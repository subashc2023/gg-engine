//! Demo 13, driven the way the host drives it (§4.2.2) — every tick through the
//! declared systems table against a `World` registered from the declared
//! components, which is `gg-runtime`'s own path.
//!
//! What these are actually about is the pair of regime transitions: a body is on
//! rails or in the bubble, never both and never neither, and the conversion in
//! each direction is computed from sim state (§2, §6 M38 items 8 and 10).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use demo_13_orbit::{
    BURN, Body, Bubble, Epoch, Event, MU_STAR, OCHRE, PARKING_RADIUS, RESTART, Rails, Ship, THRUST,
    TRACE, VERGE, WARP_UP, WARPS,
};
use gg_ecs::boundary::{
    self, AbiInfo, ActionId, ComponentsTable, HostApiV1, InputFrame, SystemsTable, TickCtx,
};
use gg_ecs::{Query, World};
use gg_math::sim;

// The symbols `gg_game!` exported into this crate's rlib.
unsafe extern "C" {
    fn gg_game_abi() -> AbiInfo;
    fn gg_game_init(api: *const HostApiV1);
    fn gg_game_components() -> ComponentsTable;
    fn gg_game_systems() -> SystemsTable;
}

const HZ: u32 = 60;

struct Game {
    world: World,
    table: SystemsTable,
    tick: u64,
    held: u64,
}

impl Game {
    fn load() -> Self {
        // SAFETY: the symbols are this binary's own, linked from the rlib.
        let info = unsafe { gg_game_abi() };
        assert_eq!(info, boundary::abi_info(), "one build, one description");
        // SAFETY: `host_api()` returns a `&'static` table (§4.2.2).
        unsafe { gg_game_init(core::ptr::from_ref(boundary::host_api())) };

        let mut world = World::new();
        // SAFETY: the tables are this binary's own and live for the process.
        let table = unsafe {
            world.adopt(&gg_game_components()).unwrap();
            gg_game_systems()
        };
        Game {
            world,
            table,
            tick: 0,
            held: 0,
        }
    }

    /// Hold an action for the ticks that follow, until [`Game::release`].
    fn hold(&mut self, action: ActionId) {
        self.held |= 1 << action.index();
    }

    fn release(&mut self) {
        self.held = 0;
    }

    /// One tick with `action` newly down — the edge a `just_pressed` verb reads.
    fn tap(&mut self, action: ActionId) {
        let previous = self.held;
        self.held |= 1 << action.index();
        self.step_with(previous);
        self.held = previous;
    }

    fn step(&mut self) {
        let previous = self.held;
        self.step_with(previous);
    }

    fn step_with(&mut self, previous: u64) {
        let ctx = TickCtx {
            tick: self.tick,
            tick_hz: HZ,
            reserved: 0,
            input: InputFrame {
                buttons: self.held,
                ..InputFrame::default()
            },
            previous: InputFrame {
                buttons: previous,
                ..InputFrame::default()
            },
        };
        // SAFETY: the table is this binary's own, its entries live for the
        // process, and `ctx` outlives the call.
        unsafe { self.world.run_systems(&self.table, &ctx) }.expect("no system panicked");
        self.tick += 1;
    }

    fn run(&mut self, ticks: u64) {
        for _ in 0..ticks {
            self.step();
        }
    }

    fn all<T: gg_ecs::Component + Copy>(&self) -> Vec<T> {
        let query = Query::<&T>::new().unwrap();
        let mut out = Vec::new();
        self.world.each_ref(&query, |_, value: &T| out.push(*value));
        out
    }

    fn one<T: gg_ecs::Component + Copy>(&self) -> T {
        let all = self.all::<T>();
        assert_eq!(all.len(), 1, "expected exactly one");
        all[0]
    }

    fn ship(&self) -> Ship {
        self.one::<Ship>()
    }

    /// The ship's conic, whichever of the two regimes it is in.
    fn ship_rails(&self) -> Option<Rails> {
        let query = Query::<(&Ship, &Rails)>::new().unwrap();
        let mut found = None;
        self.world
            .each_ref(&query, |_, (_, rails): (&Ship, &Rails)| {
                found = Some(*rails)
            });
        found
    }

    fn ship_bubble(&self) -> Option<Bubble> {
        let query = Query::<(&Ship, &Bubble)>::new().unwrap();
        let mut found = None;
        self.world
            .each_ref(&query, |_, (_, b): (&Ship, &Bubble)| found = Some(*b));
        found
    }
}

#[test]
fn the_world_opens_with_a_star_two_planets_and_a_parked_ship() {
    let mut game = Game::load();
    game.step();
    let bodies = game.all::<Body>();
    assert_eq!(bodies.len(), 3);
    assert!(bodies.iter().any(|b| b.index == 0 && b.mu == MU_STAR));
    assert!(bodies.iter().any(|b| b.index == VERGE.index));
    assert!(bodies.iter().any(|b| b.index == OCHRE.index));

    let rails = game.ship_rails().expect("the ship starts on rails");
    assert_eq!(rails.orbit.semi_major, PARKING_RADIUS);
    assert_eq!(
        rails.orbit.mu, VERGE.mu,
        "parked about the planet, not the star"
    );
    assert!(
        game.ship_bubble().is_none(),
        "nothing is integrated at rest"
    );
    assert_eq!(game.ship().fuel, demo_13_orbit::TANK);
    // Three conics, and the ship's is the third.
    assert_eq!(
        game.all::<demo_13_orbit::Marker>().len(),
        3 * TRACE as usize
    );
}

/// Re-entrant bootstrap: a reload runs it again against a populated world (§6 M5).
#[test]
fn bootstrap_run_again_spawns_nothing_new() {
    let mut game = Game::load();
    game.run(8);
    assert_eq!(game.all::<Body>().len(), 3);
    assert_eq!(game.all::<Ship>().len(), 1);
    assert_eq!(game.all::<Epoch>().len(), 1);
}

/// The whole of what warp is (§6 M38 item 10): the epoch advances by the factor,
/// the host's tick does not change rate, and nothing else in the world notices.
#[test]
fn warp_advances_the_epoch_and_not_the_clock() {
    let mut game = Game::load();
    game.step();
    assert_eq!(game.one::<Epoch>().warp, 1);

    game.tap(WARP_UP);
    game.tap(WARP_UP);
    let epoch = game.one::<Epoch>();
    assert_eq!(epoch.warp, WARPS[2]);

    let before = game.one::<Epoch>().elapsed;
    let host_tick = game.tick;
    game.run(10);
    assert_eq!(game.tick - host_tick, 10, "the host tick is one per tick");
    assert_eq!(game.one::<Epoch>().elapsed - before, 10 * WARPS[2]);
}

/// A parked body is where the closed form says it is, and it is *only* that:
/// nothing integrates while the engine is out, at any warp.
#[test]
fn a_parked_ship_keeps_its_elements_whatever_the_warp() {
    let mut game = Game::load();
    game.step();
    let start = game.ship_rails().expect("rails");
    for _ in 0..4 {
        game.tap(WARP_UP);
    }
    game.run(200);
    let after = game.ship_rails().expect("rails");
    assert_eq!(
        after.orbit, start.orbit,
        "elements are not integrated state"
    );
    assert_eq!(after.since, start.since);
    // And the position it reports has moved, or the test above proves nothing.
    let hz = f64::from(HZ);
    let epoch = game.one::<Epoch>();
    let then = start.orbit.state_at(0.0).0;
    let now = after.orbit.state_at(epoch.elapsed as f64 / hz).0;
    assert!((now - then).length() > 1.0e5);
}

/// Rails → bubble → rails, which is §2's regime boundary in both directions.
#[test]
fn lighting_the_engine_opens_the_bubble_and_cutting_it_returns_elements() {
    let mut game = Game::load();
    game.step();
    let parked = game.ship_rails().expect("rails").orbit;

    game.hold(BURN);
    game.step();
    let bubble = game.ship_bubble().expect("the bubble opened");
    assert!(
        game.ship_rails().is_none(),
        "a body is in exactly one regime"
    );
    // The state vector the elements produced, not a fresh one: the parking
    // orbit is circular at `PARKING_RADIUS`, so the radius is the invariant.
    assert!((bubble.position.length() - PARKING_RADIUS).abs() < 1.0e3);

    game.run(60);
    let burning = game.ship_bubble().expect("still lit");
    let speed = burning.velocity.length();
    game.release();
    game.step();

    let after = game.ship_rails().expect("the bubble closed");
    assert!(game.ship_bubble().is_none());
    assert!(
        after.orbit.semi_major > parked.semi_major,
        "a prograde second raises the orbit: {} -> {}",
        parked.semi_major,
        after.orbit.semi_major
    );
    // `from_state` is a conversion, not a re-simulation: the conic it returns
    // must produce the state it was given back (§6 M38 item 8 reading 1).
    let (position, velocity) = after.orbit.state_at(0.0);
    assert!((position - burning.position).length() < 1.0);
    assert!((velocity.length() - speed).abs() < 1.0);
}

/// Warp is refused while the bubble has rows — an archetype question, and the
/// one rule that keeps the integrated regime at 1x (§6 M38 item 10).
#[test]
fn warp_is_refused_while_the_engine_is_lit() {
    let mut game = Game::load();
    game.step();
    for _ in 0..3 {
        game.tap(WARP_UP);
    }
    assert_eq!(game.one::<Epoch>().warp, WARPS[3]);

    game.hold(BURN);
    game.step();
    assert_eq!(
        game.one::<Epoch>().warp,
        1,
        "lighting the engine drops warp"
    );
    game.tap(WARP_UP);
    assert_eq!(game.one::<Epoch>().warp, 1, "and it cannot be raised again");

    game.release();
    game.step();
    game.tap(WARP_UP);
    assert_eq!(
        game.one::<Epoch>().warp,
        WARPS[1],
        "back once the tank is off"
    );
}

/// The tank is spent at the rate the engine pushes, and only while it pushes.
#[test]
fn a_burn_costs_delta_v_and_a_coast_costs_none() {
    let mut game = Game::load();
    game.step();
    let full = game.ship().fuel;

    game.hold(BURN);
    game.run(60);
    let after = game.ship();
    let expected = THRUST; // one second of it
    assert!(
        (full - after.fuel - expected).abs() < expected * 0.05,
        "a second of {THRUST} m/s^2 is {expected} m/s, spent {}",
        full - after.fuel
    );
    assert!((after.spent - (full - after.fuel)).abs() < 1.0e-9);

    game.release();
    game.run(120);
    assert_eq!(game.ship().fuel, after.fuel, "a coast is free");
}

/// The flight log is entities in `(tick, entity_id)` order — §2's queue, in the
/// storage the ECS already had (§6 M38 item 12).
#[test]
fn the_flight_log_is_rows_in_the_world() {
    let mut game = Game::load();
    game.step();
    assert!(game.all::<Event>().is_empty());

    game.hold(BURN);
    game.step();
    game.release();
    game.step();
    let log = game.all::<Event>();
    assert_eq!(log.len(), 2, "lit, then cut");
    assert_eq!(log[0].kind, demo_13_orbit::EVENT_LIT);
    assert_eq!(log[1].kind, demo_13_orbit::EVENT_CUT);
    assert!(log[0].epoch <= log[1].epoch);
}

/// Restart takes the world back to an opening the same in every element — the
/// mission is a function of the elements and the epoch, and nothing else.
#[test]
fn restart_rebuilds_the_opening_exactly() {
    let mut game = Game::load();
    game.step();
    let opening = game.ship_rails().expect("rails").orbit;

    game.hold(BURN);
    game.run(30);
    game.release();
    game.run(30);
    assert!(game.ship().fuel < demo_13_orbit::TANK);

    game.tap(RESTART);
    game.step();
    assert_eq!(game.ship().fuel, demo_13_orbit::TANK);
    assert_eq!(
        game.one::<Epoch>().elapsed,
        1,
        "one tick of the fresh epoch"
    );
    assert_eq!(game.ship_rails().expect("rails").orbit, opening);
    assert!(game.all::<Event>().is_empty(), "the log restarts with it");
}

/// A conic sampled by stepping its mean anomaly closes on itself — which is
/// what the trace draws, and the reason a trajectory costs nothing.
#[test]
fn stepping_the_mean_anomaly_walks_the_same_conic() {
    let orbit = sim::Orbit {
        semi_major: PARKING_RADIUS,
        eccentricity: 0.3,
        inclination: 0.4,
        ascending_node: 0.2,
        argument_of_periapsis: 1.1,
        mean_anomaly: 0.0,
        mu: VERGE.mu,
    };
    let stepped = sim::Orbit {
        mean_anomaly: orbit.mean_anomaly + core::f64::consts::TAU,
        ..orbit
    };
    let (a, _) = orbit.state_at(0.0);
    let (b, _) = stepped.state_at(0.0);
    assert!((a - b).length() < 1.0e-6, "a full turn is the same point");

    // And the period is the other route to the same claim.
    let period = orbit.period().expect("closed");
    let (c, _) = orbit.state_at(period);
    assert!((a - c).length() < 1.0e-3);
}

/// The mission's first half, end to end: a burn big enough to leave the
/// departure planet hands the conic to the **star**, computed from state and
/// not from a distance table (§6 M38 item 8), and the flight log says so once.
#[test]
fn escaping_the_planet_hands_the_conic_to_the_star() {
    let mut game = Game::load();
    game.step();
    // ~4 km/s of prograde, which is over the 3.5 km/s a departure needs.
    game.hold(BURN);
    game.run(960);
    game.release();
    game.step();
    let escape = game.ship_rails().expect("rails");
    assert!(
        escape.orbit.eccentricity > 1.0,
        "that much delta-v is a hyperbola about the planet, e = {}",
        escape.orbit.eccentricity
    );

    for _ in 0..5 {
        game.tap(WARP_UP);
    }
    assert_eq!(game.one::<Epoch>().warp, WARPS[5]);
    game.run(400);

    let rails = game.ship_rails().expect("rails");
    let primary = game
        .all::<Body>()
        .into_iter()
        .find(|body| body.mu == rails.orbit.mu)
        .expect("a body whose gravity this conic is about");
    assert_eq!(primary.index, 0, "the star is the primary now");
    assert!(
        rails.orbit.eccentricity < 1.0,
        "and the heliocentric conic is closed: e = {}",
        rails.orbit.eccentricity
    );
    // The claim the eccentricity cannot make: a handover is a change of *frame*,
    // and a ship that has just left a planet keeps that planet's 29.8 km/s.
    // Dropping it leaves a conic that is closed, plausible, and a third of the
    // right size — which every assertion above this one forgives.
    let speed = rails.orbit.state_at(0.0).1.length();
    assert!(
        speed > 20_000.0,
        "the departure planet's own motion is part of the state that crossed: {speed} m/s"
    );
    let handovers = game
        .all::<Event>()
        .into_iter()
        .filter(|e| e.kind == demo_13_orbit::EVENT_HANDOVER)
        .count();
    assert_eq!(handovers, 1, "one crossing, not a chatter of them");
}
