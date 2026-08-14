//! The mission, flown by the script that records it (§6 M38 item 14).
//!
//! `tests/game.rs` is about the two regimes; this is about the *flight* — the
//! whole of it, parking orbit to capture, through the declared systems table.
//! What it is really guarding is that `session::FLOWN` and the target's phase
//! still agree: they were solved against each other by `gg-tools transfer`, and
//! the window is one cell wide, so anything that moves the departure or the
//! target's elements breaks the mission rather than degrading it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use demo_13_orbit::session::{self, Entry, Plan};
use demo_13_orbit::{
    CAPTURED, EVENT_CUT, EVENT_HANDOVER, EVENT_LIT, EVENT_OUTCOME, FLYING, OCHRE, TANK, VERGE,
};
use gg_ecs::boundary::{AbiInfo, ComponentsTable, HostApiV1, SystemsTable};

// The symbols `gg_game!` exported into this crate's rlib.
unsafe extern "C" {
    fn gg_game_abi() -> AbiInfo;
    fn gg_game_init(api: *const HostApiV1);
    fn gg_game_components() -> ComponentsTable;
    fn gg_game_systems() -> SystemsTable;
}

fn entry() -> Entry {
    Entry {
        abi: gg_game_abi,
        init: gg_game_init,
        components: gg_game_components,
        systems: gg_game_systems,
    }
}

/// Where the target's pull beats the star's by the handover ratio — the width
/// of the window the crossing has to land in, restated here because a test that
/// took it from the game would agree with any value the game happened to hold.
const GRIP: f64 = 9.15e7;

#[test]
fn the_mission_flies_to_a_capture() {
    let flight = session::fly(&entry(), session::FLOWN).expect("the pilot flew");
    assert_eq!(
        flight.outcome, CAPTURED,
        "the flown plan no longer captures: closest {:.4e} m against a {GRIP:.2e} m grip",
        flight.approach
    );
    let periapsis = flight.periapsis.expect("an intercept has a periapsis");
    assert!(
        periapsis > OCHRE.radius,
        "the crossing arrived inside the planet: {periapsis:.4e} m against a radius of {:.4e}",
        OCHRE.radius
    );
    assert!(
        flight.approach < GRIP,
        "an intercept means the ship got inside the grip: {:.4e}",
        flight.approach
    );
    assert!(
        flight.spent < TANK,
        "the mission has to be affordable: {:.0} m/s of {TANK:.0}",
        flight.spent
    );
    // The transfer is a real crossing and not a fall: its apoapsis reaches the
    // target's orbit and its periapsis is still near the departure planet's.
    let transfer = flight.transfer.expect("the star took the conic");
    let orbit = transfer.orbit;
    assert!(orbit.eccentricity < 1.0);
    assert!(
        orbit.semi_major * (1.0 + orbit.eccentricity)
            > OCHRE.semi_major * (1.0 - OCHRE.eccentricity),
        "the transfer never reaches the target's orbit"
    );
    assert!((orbit.semi_major * (1.0 - orbit.eccentricity) - VERGE.semi_major).abs() < 1.0e10);
}

/// The flight log, in order — §2's event queue with a whole mission in it.
#[test]
fn the_flight_log_tells_the_mission_in_order() {
    let flight = session::fly(&entry(), session::FLOWN).expect("the pilot flew");
    let all = session::progress(&entry(), &flight.frames).expect("driven");
    // Everything but the closing restart, which despawns the ship and would
    // read as a third handover to nowhere.
    let progress = &all[..all.len() - 1];

    // The regimes, as ticks: parked about the departure planet, out to the
    // star, caught by the target, and exactly one of each crossing.
    let first = |find: &dyn Fn(&session::Progress) -> bool| progress.iter().position(find);
    let escaped = first(&|p| p.primary == 0).expect("the star took it");
    let intercept = first(&|p| p.primary == OCHRE.index).expect("the target took it");
    let captured = first(&|p| p.outcome == CAPTURED).expect("captured");
    assert!(
        escaped < intercept && intercept < captured,
        "escaped {escaped}, intercept {intercept}, captured {captured}"
    );
    assert!(
        progress[..escaped].iter().all(|p| p.primary == VERGE.index),
        "the mission opens parked about the departure planet"
    );

    // Two handovers and no more: a conic that chatters across the ratio would
    // show up here as four, and the hysteresis is what stops it.
    let handovers = (1..progress.len())
        .filter(|&i| progress[i].primary != progress[i - 1].primary)
        .count();
    assert_eq!(handovers, 2, "one crossing out, one crossing in");

    // The engine is lit exactly twice — the departure and the capture — and the
    // world is on rails everywhere else, which is what makes warp reachable for
    // all but 1578 of the mission's ticks.
    let lightings = (1..progress.len())
        .filter(|&i| progress[i].lit && !progress[i - 1].lit)
        .count();
    assert_eq!(lightings, 2, "one burn out, one burn in");
    let coasting = progress.iter().filter(|p| !p.lit).count();
    assert!(
        coasting * 2 > progress.len(),
        "most of a mission is a coast: {coasting} of {}",
        progress.len()
    );
}

/// The events the world holds, as kinds in the order they landed.
#[test]
fn the_events_are_the_mission_as_rows() {
    let flight = session::fly(&entry(), session::FLOWN).expect("the pilot flew");
    let progress = session::progress(&entry(), &flight.frames).expect("driven");
    // Lit, cut, escaped, intercept, lit, cut, outcome — seven, and then the
    // restart on the last frame takes the log with it.
    let peak = progress.iter().map(|p| p.events).max().expect("ticks");
    assert_eq!(peak, 7, "the whole mission is seven rows");
    assert_eq!(
        progress.last().expect("ticks").events,
        0,
        "the closing restart clears the log with the world"
    );
    // The kinds exist as constants and the log is what orders them; naming them
    // here is what makes the count above mean something.
    assert_ne!(EVENT_LIT, EVENT_CUT);
    assert_ne!(EVENT_HANDOVER, EVENT_OUTCOME);
}

/// The same stream twice is the same world, tick for tick — §5.6 through this
/// crate's own tables, before any shell is involved.
#[test]
fn the_stream_hashes_the_same_twice() {
    let entry = entry();
    let frames = session::frames(&entry).expect("the mission");
    let first = session::hash_sequence(&entry, &frames).expect("driven");
    let second = session::hash_sequence(&entry, &frames).expect("driven again");
    assert_eq!(first.len(), frames.len());
    assert!(
        session::divergence(
            &first,
            &second.iter().map(|h| (0, h.get())).collect::<Vec<_>>()
        )
        .is_none(),
        "two runs of one stream are one world"
    );
}

/// The negative control, and the reason the constants are solved rather than
/// picked: a departure a few minutes late does not degrade the mission, it ends
/// it. The window really is one cell of the sweep wide.
#[test]
fn a_late_departure_never_arrives() {
    let late = Plan {
        wait: session::FLOWN.wait + 25,
        ..session::FLOWN
    };
    let flight = session::fly(&entry(), late).expect("the pilot flew");
    assert_eq!(flight.outcome, FLYING, "a miss is a miss, not a crash");
    assert!(
        flight.approach > GRIP * 10.0,
        "25 host ticks of parking orbit is 7 minutes and it should miss by far more than the \
         grip: {:.4e} m",
        flight.approach
    );
    assert!(flight.periapsis.is_none(), "nothing took it");
}
