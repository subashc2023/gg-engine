//! The controller, driven the way the host drives it (§4.2.2): every tick goes
//! through the declared systems table against an adopted `World`, and buttons
//! and axes are set on the [`InputFrame`] the same way a replay sets them.
//!
//! Every claim about geometry *derives* from [`ROOM`] rather than restating a
//! number out of it, so a slab that moves re-tests these rather than breaking
//! them for the wrong reason. Sim claims are exact equalities: a resolve lands
//! the body on a face, and "approximately on it" would forgive a real drift.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use demo_12_shooter::{
    ACTIONS, AIM_X, AIM_Y, ARMS, AXES, BEGIN, BUFFER_TICKS, CHART_BALLS, COYOTE_TICKS, CUE_JUMP,
    CUE_LAND, CUES, Cue, EYE_LIFT, FIRE, FIRE_TICKS, FLASH_TICKS, Gun, HALF_H, HALF_W,
    HITMARK_TICKS, HUD_ROUND, HUD_ROWS, HUD_SPEED, HUD_STATE, Hud, JUMP, JUMP_VELOCITY, LAMPS,
    LIFE_FLOOR, MENU_AA, MENU_AIM, MENU_BACK, MENU_ITEMS, MENU_LOOK, MENU_LOOK_DOWN, MENU_LOOK_UP,
    MENU_QUIT, MENU_RESTART, MENU_RESUME, MENU_RULE, MENU_SETTINGS, MENU_START, MENU_TALLY,
    MENU_TITLE, MENU_TO_TITLE, MENU_VOLUME, MENU_VOLUME_DOWN, MISSES_ALLOWED, MOVE_FORWARD,
    MOVE_RIGHT, Menu, PAGE_MAIN, PAGE_OVER, PAGE_SETTINGS, PAGE_TITLE, PAUSE, RECOIL_KICK,
    RECOIL_TICKS, RESTART, ROOM, RULE, Range, SENS_DEFAULT, SENS_MAX, SENS_MIN, SENS_ONE,
    SENS_STEP, SHELTER, SHOT_RANGE, SKIN, SPARK_TICKS, SPOTS, START, STATE_OVER, STATE_RUNNING,
    STEP_HEIGHT, STREAK_STEP, Session, Solid, Spark, TARGET_HALF, TARGET_INK, TARGET_LIFE,
    TARGET_WORTH, TARGETS_LIVE, TITLE_NAME, Target, WALK_SPEED, WAVE_HITS, Walker, accuracy,
    counts_per_turn, life_for, look_per_count, session, wave_of, worth_for,
};
use gg_ecs::boundary::{
    self, AbiInfo, ActionId, AxisId, ComponentsTable, Eye, HostApiV1, InputFrame, Light, Prefs,
    QUIET_MAX, Renderable, Sky, Sound, SystemsTable, TickCtx, Widget, aa, state,
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

/// Full deflection in the fixed-point axis encoding (§4.7's `AXIS_SCALE`).
const STICK: i32 = 1024;

struct Game {
    world: World,
    table: SystemsTable,
    tick: u64,
    /// This tick's buttons; cleared after every step, so a hold is expressed by
    /// setting it again and an *edge* by not setting it the tick before.
    held: u64,
    previous: u64,
    axes: [i32; boundary::MAX_AXES],
    /// See [`Game::forgiving`].
    forgiving: bool,
}

impl Game {
    fn load() -> Self {
        // SAFETY: the symbols are this binary's own, linked from the rlib.
        let info = unsafe { gg_game_abi() };
        assert_eq!(info, boundary::abi_info(), "one build, one description");
        // SAFETY: `host_api()` returns a `&'static` table (§4.2.2).
        unsafe { gg_game_init(core::ptr::from_ref(boundary::host_api())) };

        let mut world = World::new();
        // The host's protocol registrations, exactly as `gg_runtime::app::adopt`
        // makes them — `gg.model` included, though this game never spawns one.
        world.register::<Renderable>().unwrap();
        world.register::<Eye>().unwrap();
        world.register::<boundary::Model>().unwrap();
        world.register::<Light>().unwrap();
        world.register::<Widget>().unwrap();
        world.register::<Sky>().unwrap();
        // SAFETY: the tables are this binary's own and live for the process.
        let (table, declared) = unsafe {
            let declared = world.adopt(&gg_game_components()).unwrap();
            (gg_game_systems(), declared)
        };
        assert_eq!(declared, 18, "ten of ours and the protocol's eight");
        Game {
            world,
            table,
            tick: 0,
            held: 0,
            previous: 0,
            axes: [0; boundary::MAX_AXES],
            forgiving: false,
        }
    }

    fn hold(&mut self, action: ActionId) -> &mut Self {
        self.held |= 1 << action.index();
        self
    }

    fn push(&mut self, axis: AxisId, deflection: i32) -> &mut Self {
        self.axes[axis.index()] = deflection * STICK;
        self
    }

    fn step(&mut self) {
        let ctx = TickCtx::detached(
            self.tick,
            60,
            InputFrame {
                buttons: self.held,
                axes: self.axes,
            },
            InputFrame {
                buttons: self.previous,
                axes: [0; boundary::MAX_AXES],
            },
        );
        // SAFETY: the table is this binary's own, its entries live for the
        // process, and `ctx` outlives the call.
        unsafe { self.world.run_systems(&self.table, &ctx) }.expect("no system panicked");
        if self.forgiving {
            let query = Query::<&mut Range>::new().expect("a round query");
            self.world.each(&query, |_, range: &mut Range| {
                range.misses = 0;
                range.state = STATE_RUNNING;
            });
        }
        self.tick += 1;
        self.previous = self.held;
        self.held = 0;
        self.axes = [0; boundary::MAX_AXES];
    }

    fn steps(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Hold forward for `n` ticks.
    fn forward(&mut self, n: u64) {
        for _ in 0..n {
            self.push(MOVE_FORWARD, 1);
            self.step();
        }
    }

    /// Hold one axis for `n` ticks.
    fn drive(&mut self, axis: AxisId, deflection: i32, n: u64) {
        for _ in 0..n {
            self.push(axis, deflection);
            self.step();
        }
    }

    /// Walk off whatever the body is standing on and say how many ticks it
    /// took. Used instead of a computed distance: the arc and the accel are
    /// what the tuning constants make them, and a test that recomputed them
    /// would be asserting its own arithmetic.
    fn until_airborne(&mut self, axis: AxisId, deflection: i32, bound: u64) -> u64 {
        for taken in 0..bound {
            if !self.walker().grounded() {
                return taken;
            }
            self.push(axis, deflection);
            self.step();
        }
        panic!("still on the ground after {bound} ticks");
    }

    fn until_grounded(&mut self, bound: u64) -> u64 {
        for taken in 0..bound {
            if self.walker().grounded() {
                return taken;
            }
            self.step();
        }
        panic!("still in the air after {bound} ticks");
    }

    fn one<T: gg_ecs::Component + Copy>(&self) -> T {
        let query = Query::<&T>::new().unwrap();
        let mut found = None;
        self.world.each_ref(&query, |_, v: &T| found = Some(*v));
        found.expect("exactly one of this component")
    }

    fn all<T: gg_ecs::Component + Copy>(&self) -> Vec<T> {
        let query = Query::<&T>::new().unwrap();
        let mut out = Vec::new();
        self.world.each_ref(&query, |_, v: &T| out.push(*v));
        out
    }

    fn walker(&self) -> Walker {
        self.one::<Walker>()
    }

    /// The `seq` of one cue — the trigger idiom's whole observable.
    fn cue_seq(&self, kind: u32) -> u32 {
        let query = Query::<(&Cue, &Sound)>::new().unwrap();
        let mut found = 0;
        self.world
            .each_ref(&query, |_, (cue, sound): (&Cue, &Sound)| {
                if cue.kind == kind {
                    found = sound.seq;
                }
            });
        found
    }

    /// One press *edge*. An idle tick first if the previous one already held
    /// this action, because `just_pressed` is an edge and two holds in a row
    /// are one press — which is the whole reason `step` keeps `previous`.
    fn tap(&mut self, action: ActionId) -> &mut Self {
        if self.previous & (1 << action.index()) != 0 {
            self.step();
        }
        self.hold(action);
        self.step();
        self
    }

    /// Hold the round open under a test that is about the *controller*.
    ///
    /// A player who stands still for [`TARGET_LIFE`] loses three targets and
    /// the round with them — that is the game, and since §6 M76 losing also
    /// stops the world. So a walk that takes longer than one course now ends
    /// with a frozen body against an over screen, and a test about jumping
    /// would be asserting that a menu is up.
    ///
    /// This keeps the miss counter at zero after every tick — and the round
    /// running with it, because the course is dealt in one batch and escapes
    /// in one, so the counter reaches [`MISSES_ALLOWED`] and the state flips
    /// inside a single tick. It is the smallest untruth that lets a movement
    /// test be about movement. It is
    /// not used by anything that asserts about the round, and
    /// `three_targets_that_walk_end_the_round` is what says the counter
    /// otherwise works.
    fn forgiving(&mut self) -> &mut Self {
        self.forgiving = true;
        self
    }

    /// Leave the title screen (§6 M76): one tick for `bootstrap` to deal the
    /// world and the screen it opens on, one for the key that leaves it.
    ///
    /// Two ticks, deliberately — it is what `steps(2)` used to be, so a test
    /// that counted from a settled world still counts from the same place.
    /// The course is dealt on the tick play begins, not behind the screen:
    /// `targets` is one of the systems the title screen stops.
    fn begin(&mut self) -> &mut Self {
        self.step();
        self.tap(BEGIN);
        self
    }

    /// Every HUD widget, rows and crosshair arms alike.
    fn hud_widgets(&self) -> Vec<Widget> {
        let query = Query::<(&Hud, &Widget)>::new().unwrap();
        let mut out = Vec::new();
        self.world
            .each_ref(&query, |_, (_, widget): (&Hud, &Widget)| out.push(*widget));
        out
    }

    fn session(&self) -> Session {
        self.one::<Session>()
    }

    fn prefs(&self) -> Prefs {
        self.one::<Prefs>()
    }

    /// A menu widget by item.
    fn menu(&self, item: u32) -> Widget {
        let query = Query::<(&Menu, &Widget)>::new().unwrap();
        let mut found = None;
        self.world
            .each_ref(&query, |_, (menu, widget): (&Menu, &Widget)| {
                if menu.item == item {
                    found = Some(*widget);
                }
            });
        found.expect("a menu widget with this item")
    }

    /// Click one, the way the host does: it writes `CLICKED` into the widget at
    /// the end of a tick and the game reads it at the start of the next. Nothing
    /// here routes a pointer — that is `gg-ui`'s and it is gated where it lives
    /// (§4.9); what this covers is the game's half of the protocol.
    fn click(&mut self, item: u32) {
        let query = Query::<(&Menu, &mut Widget)>::new().unwrap();
        let mut landed = false;
        self.world
            .each(&query, |_, (menu, widget): (&Menu, &mut Widget)| {
                // Only where the host could have landed it: a hidden widget is
                // a zero rect and takes no clicks, which is the property the
                // pointer hand-back rests on.
                if menu.item == item && widget.rect[2] > 0.0 && widget.rect[3] > 0.0 {
                    widget.state |= state::CLICKED;
                    landed = true;
                }
            });
        assert!(landed, "item {item} is not on screen to be clicked");
        self.step();
        // And taken away again. The host rewrites `state` from scratch every
        // tick (§4.9), so a bit left set here would be a button clicked for the
        // rest of the session — which is exactly what it was before this line.
        self.world
            .each(&query, |_, (_, widget): (&Menu, &mut Widget)| {
                widget.state = 0;
            });
    }

    fn hud(&self, line: u32) -> String {
        let query = Query::<(&Hud, &Widget)>::new().unwrap();
        let mut found = String::new();
        self.world
            .each_ref(&query, |_, (hud, widget): (&Hud, &Widget)| {
                if hud.line == line {
                    found = widget.text().to_string();
                }
            });
        found
    }
}

/// Feet, in metres: the claim every ground assertion below is really about.
trait Feet {
    fn feet(&self) -> f64;
    fn grounded(&self) -> bool;
}

impl Feet for Walker {
    fn feet(&self) -> f64 {
        self.position.y - HALF_H
    }
    fn grounded(&self) -> bool {
        self.grounded != 0
    }
}

/// The body *centre* when it is standing on a face at `top` — the very sum
/// [`resolve_y`](demo_12_shooter) writes, reproduced rather than approximated.
/// Asserting on the centre and not the feet is what keeps these exact: the
/// subtraction `feet` does gives back a [`SKIN`] one part in 10^8 shy of the
/// constant, because a nanometre added at 0.9 m has nowhere near a nanometre
/// of room left in the mantissa.
fn resting_on(top: f64) -> f64 {
    top + HALF_H + SKIN
}

/// Ground speed, metres per tick.
fn speed(game: &Game) -> f64 {
    let v = game.walker().velocity;
    (v.x * v.x + v.z * v.z).sqrt()
}

/// The top of the first slab in [`ROOM`] whose plan footprint contains `(x, z)`
/// and whose top is at or below `ceiling` — the height a body walking there
/// stands at, derived rather than restated.
fn top_at(x: f64, z: f64, ceiling: f64) -> f64 {
    let mut best = f64::MIN;
    for (position, half, _) in ROOM {
        let top = position.y + f64::from(half.y);
        if top <= ceiling
            && (x - position.x).abs() < f64::from(half.x)
            && (z - position.z).abs() < f64::from(half.z)
            && top > best
        {
            best = top;
        }
    }
    best
}

/// Walk from the spawn to the stairs and up them. The room's one long traverse,
/// and the only ledge with enough air under it to time a coyote window against.
fn onto_the_mezzanine(game: &mut Game) {
    game.drive(MOVE_RIGHT, -1, 70);
    game.drive(MOVE_FORWARD, 1, 120);
}

#[test]
fn one_tick_deals_the_room_the_body_and_the_hud() {
    let mut game = Game::load();
    game.step();
    assert_eq!(game.all::<Walker>().len(), 1);
    assert_eq!(
        game.all::<Solid>().len(),
        ROOM.len(),
        "every slab, and the chart is not one (§6 M26)"
    );
    assert_eq!(
        game.all::<Renderable>().len(),
        ROOM.len() + CHART_BALLS,
        "the room and the chart, dealt behind the title screen"
    );
    assert!(
        game.all::<Target>().is_empty(),
        "and no course: `targets` is one of the systems the screen stops, so \
         the round is dealt on the tick play begins (§6 M76)"
    );
    assert_eq!(
        game.all::<Sky>().len(),
        2,
        "the sky over the room and the shelter's own (§6 M28)"
    );
    assert_eq!(game.all::<Eye>().len(), 1);
    assert_eq!(game.all::<Light>().len(), 3, "a sun and two lamps");
    assert_eq!(game.all::<Sound>().len(), CUES);
    assert_eq!(
        game.all::<Widget>().len(),
        ARMS + HUD_ROWS.len() + MENU_ITEMS.len(),
        "arms, the text rows and the menu"
    );
    // Nothing of the HUD is on the glass yet: it belongs to the round, and
    // the round has not started (§6 M76). The reticle is the one that would
    // be noticed — a crosshair over a title screen aims at a menu.
    assert!(game.all::<Hud>().iter().all(|_| true), "the rows exist");
    for widget in game.hud_widgets() {
        assert_eq!(
            (widget.rect[2], widget.rect[3]),
            (0.0, 0.0),
            "the HUD is put away behind a screen"
        );
    }

    // And it all comes back with the round, course included.
    game.tap(BEGIN);
    assert_eq!(game.all::<Target>().len(), TARGETS_LIVE);
    assert_eq!(
        game.all::<Renderable>().len(),
        ROOM.len() + CHART_BALLS + TARGETS_LIVE,
        "the room, the chart, and the course's first three targets"
    );
    assert!(
        game.hud_widgets().iter().all(|w| w.rect[2] > 0.0),
        "and the HUD is back"
    );
    assert_eq!(game.hud(HUD_STATE), "GROUND");
}

/// Re-entrant bootstrap: a reload runs it again against a populated world, and
/// what it finds must not double (§6 M5).
#[test]
fn bootstrap_run_again_spawns_nothing_new() {
    let mut game = Game::load();
    game.begin();
    game.steps(6);
    assert_eq!(game.all::<Walker>().len(), 1);
    assert_eq!(game.all::<Gun>().len(), 1);
    assert_eq!(game.all::<Range>().len(), 1);
    assert_eq!(
        game.all::<Renderable>().len(),
        ROOM.len() + CHART_BALLS + TARGETS_LIVE
    );
    assert_eq!(game.all::<Light>().len(), 3);
    assert_eq!(
        game.all::<Widget>().len(),
        ARMS + HUD_ROWS.len() + MENU_ITEMS.len()
    );
}

/// The body rests *on* the floor, exactly. A resolve puts a face on a face, so
/// this is an equality and not an epsilon.
#[test]
fn the_body_rests_exactly_on_the_floor() {
    let mut game = Game::load();
    game.begin();
    game.steps(2);
    let walker = game.walker();
    assert_eq!(walker.position.y, resting_on(top_at(START.x, START.z, 0.5)));
    assert!(walker.grounded());
    assert_eq!(walker.velocity.y, 0.0, "resting, not still settling");
}

/// The eye is in the head, not at the body's centre — and it follows.
#[test]
fn the_eye_rides_the_body() {
    let mut game = Game::load();
    game.begin();
    game.forward(20);
    let walker = game.walker();
    let eye = game.one::<Eye>();
    assert_eq!(eye.position.y, walker.position.y + EYE_LIFT);
    assert_eq!(eye.position.z, walker.position.z);
    assert!(eye.position.z < START.z, "twenty ticks of forward moved it");
}

/// A kerb is walked onto, not jumped onto: the step-up, positively. The body
/// never leaves the ground doing it.
#[test]
fn a_kerb_is_stepped_onto_without_leaving_the_ground() {
    let mut game = Game::load();
    game.begin();
    let kerb = top_at(0.0, 4.5, STEP_HEIGHT);
    assert!(kerb > 0.0, "there is a kerb straight ahead of the spawn");

    for _ in 0..40 {
        game.push(MOVE_FORWARD, 1);
        game.step();
        assert!(game.walker().grounded(), "a step is never a flight");
    }
    assert_eq!(game.walker().position.y, resting_on(kerb));
}

/// A waist-height crate is not: the same code path, refusing. The body stops
/// against its face with its feet still on the floor.
#[test]
fn a_crate_taller_than_the_step_blocks_instead() {
    let mut game = Game::load();
    game.begin();
    // The crate straight ahead, past the kerb — the first slab on the walk
    // whose top is above STEP_HEIGHT and below the walls.
    let (crate_z, crate_half_z) = ROOM
        .iter()
        .filter(|(p, h, _)| {
            let top = p.y + f64::from(h.y);
            top > STEP_HEIGHT && top < 1.0 && p.x.abs() < f64::from(h.x)
        })
        .map(|(p, h, _)| (p.z, f64::from(h.z)))
        .next()
        .expect("a crate on the path");

    game.forward(140);
    let walker = game.walker();
    assert_eq!(walker.position.y, resting_on(0.0), "stopped, not climbed");
    assert_eq!(
        walker.position.z,
        crate_z + crate_half_z + HALF_W + SKIN,
        "resting against the crate's near face"
    );
}

/// And a wall stops it the same way — the negative case at full height, which
/// also proves the step-up refuses when *one* overlap is a wall.
///
/// Backwards, into the south wall: it is the one straight line out of the
/// spawn with nothing on it, and a test that has to dodge scenery is a test
/// about the scenery.
#[test]
fn a_wall_stops_the_body_at_its_face() {
    let mut game = Game::load();
    game.begin();
    game.drive(MOVE_FORWARD, -1, 120);
    let wall = ROOM
        .iter()
        .find(|(p, h, _)| p.z > 12.0 && f64::from(h.x) > 10.0)
        .expect("a south wall");
    let walker = game.walker();
    assert_eq!(
        walker.position.z,
        wall.0.z - f64::from(wall.1.z) - HALF_W - SKIN,
        "against the wall's inner face"
    );
    assert_eq!(walker.position.y, resting_on(0.0), "a wall is never a step");
}

/// The space that is covered and the space that is lit differently are the same
/// space (§6 M28).
///
/// The two are placed from one constant, so this is not a tautology but the
/// assertion that the derivation is right: the roof has to be *above* the volume
/// and *cover* it in both horizontal axes, and the shelter's sky has to be the
/// bounded one rather than the room's. A roof half a metre short would light a
/// strip of open floor as though it were indoors, which is exactly the mistake a
/// number typed twice makes.
#[test]
fn the_shelter_is_lit_over_exactly_the_ground_its_roof_covers() {
    let (centre, half) = SHELTER;
    // The slab standing over the volume's centre, not merely the first one
    // higher than it — the room has floating platforms above this height
    // elsewhere, and a search by height alone would grade one of those.
    let roof = ROOM
        .iter()
        .find(|(at, extent, _)| {
            at.y > centre.y
                && (centre.x - at.x).abs() <= f64::from(extent.x)
                && (centre.z - at.z).abs() <= f64::from(extent.z)
        })
        .expect("the shelter has a roof");
    assert!(
        roof.0.y - f64::from(roof.1.y) >= centre.y + f64::from(half.y) - 1e-9,
        "the roof's underside must not dip into the volume: {roof:?}"
    );
    for (axis, at, extent, covered, reach) in [
        ("x", roof.0.x, roof.1.x, centre.x, half.x),
        ("z", roof.0.z, roof.1.z, centre.z, half.z),
    ] {
        assert!(
            at - f64::from(extent) <= covered - f64::from(reach)
                && at + f64::from(extent) >= covered + f64::from(reach),
            "the roof does not cover the lit volume in {axis}"
        );
    }

    let mut game = Game::load();
    game.step();
    let mut skies = game.all::<Sky>();
    skies.retain(|sky| sky.half_extent != gg_math::sim::Vec3::ZERO);
    assert_eq!(skies.len(), 1, "one bounded sky, and it is the shelter's");
    assert_eq!(skies[0].center, centre);
    assert_eq!(skies[0].half_extent, half);
    assert!(
        skies[0].fade > 0.0,
        "a doorway is a fade, not a tick (§6 M28)"
    );
}

/// Six 0.30 m rises walked end to end onto the mezzanine, and not one tick of
/// it airborne. The stairs are the step-up's real job.
#[test]
fn the_stairs_are_walked_to_the_top_without_a_jump() {
    let mut game = Game::load();
    game.begin();
    onto_the_mezzanine(&mut game);
    let walker = game.walker();
    assert!(walker.grounded());
    assert_eq!(
        walker.position.y,
        resting_on(top_at(walker.position.x, walker.position.z, 2.0)),
        "standing on the mezzanine's own top face"
    );
    assert!(walker.feet() > STEP_HEIGHT * 4.0, "and it is a real climb");
}

/// A jump leaves at exactly [`JUMP_VELOCITY`], and lands back where it left.
#[test]
fn a_jump_leaves_at_the_stated_velocity_and_comes_back_down() {
    let mut game = Game::load();
    game.begin();
    game.steps(2);
    let floor = game.walker().feet();
    let quiet = game.cue_seq(CUE_JUMP);

    game.hold(JUMP);
    game.step();
    let launched = game.walker();
    assert_eq!(launched.velocity.y, JUMP_VELOCITY);
    assert!(!launched.grounded());
    assert_eq!(game.cue_seq(CUE_JUMP), quiet + 1, "one blip, on the edge");
    assert_eq!(game.hud(HUD_STATE), "AIR");

    let landed_after = game.until_grounded(600);
    assert!(landed_after > 10, "a jump that lasts a tick is not a jump");
    assert_eq!(game.walker().feet(), floor);
    assert_eq!(game.hud(HUD_STATE), "GROUND");
}

/// The apex clears the floating slabs the room asks a jump to reach. Derived
/// from [`ROOM`], so a slab raised past the arc fails here rather than in the
/// operator's hands.
#[test]
fn the_apex_clears_the_rise_the_room_asks_for() {
    let mut game = Game::load();
    game.begin();
    game.steps(2);
    let mut apex: f64 = 0.0;
    game.hold(JUMP);
    game.step();
    // Held throughout, so RISE_CUT never applies — the full arc.
    for _ in 0..80 {
        game.hold(JUMP);
        game.step();
        apex = apex.max(game.walker().feet());
        if game.walker().grounded() {
            break;
        }
    }
    let lowest_slab = ROOM
        .iter()
        .map(|(p, h, _)| p.y + f64::from(h.y))
        .filter(|top| *top > STEP_HEIGHT)
        .fold(f64::MAX, f64::min);
    assert!(
        apex > lowest_slab,
        "apex {apex} m does not reach the lowest thing worth jumping onto ({lowest_slab} m)"
    );
}

/// Coyote time: a jump asked for shortly after walking off a ledge still fires,
/// and one asked for past the window does not.
#[test]
fn the_coyote_window_is_open_early_and_shut_late() {
    for (delay, expected) in [(COYOTE_TICKS - 3, true), (COYOTE_TICKS + 2, false)] {
        let mut game = Game::load();
        game.forgiving().begin();
        // Off the mezzanine's east edge. The kerb would do for the early half
        // and not the late one: a 0.3 m drop is over in eleven ticks, so a
        // press past the window would land inside the *buffer* instead and
        // fire for the right reason at the wrong test's expense.
        onto_the_mezzanine(&mut game);
        let left_at = game.until_airborne(MOVE_RIGHT, 1, 200);
        assert!(left_at > 0, "the strafe found the mezzanine's edge");
        assert!(!game.walker().grounded());

        game.steps(u64::from(delay));
        assert!(!game.walker().grounded(), "still falling when asked");
        game.hold(JUMP);
        game.step();
        let rising = game.walker().velocity.y > 0.0;
        assert_eq!(
            rising,
            expected,
            "a jump {delay} ticks after the ledge should{} fire",
            if expected { "" } else { " not" }
        );
    }
}

/// Jump buffering: a press made just before landing fires *on* the landing
/// tick, which is the counter's whole reason to exist.
#[test]
fn a_buffered_press_fires_on_the_landing_tick() {
    // Measure the flight first rather than computing it: the arc belongs to the
    // tuning constants, and a test that recomputed it would assert arithmetic.
    let flight = {
        let mut game = Game::load();
        game.forgiving().begin();
        game.steps(2);
        game.hold(JUMP);
        game.step();
        game.until_grounded(600)
    };
    assert!(flight > u64::from(BUFFER_TICKS), "room to press inside it");

    let mut game = Game::load();
    game.forgiving().begin();
    game.steps(2);
    game.hold(JUMP);
    game.step();
    // Fall to within the buffer window, then press: the edge needs the tick
    // before it to be clear, which `step` guarantees by clearing `held`.
    game.steps(flight - u64::from(BUFFER_TICKS) + 1);
    assert!(
        !game.walker().grounded(),
        "still airborne when the press lands"
    );
    let quiet = game.cue_seq(CUE_LAND);
    game.hold(JUMP);
    game.step();

    let mut fired = false;
    for _ in 0..BUFFER_TICKS {
        if game.walker().velocity.y > 0.0 {
            fired = true;
            break;
        }
        game.step();
    }
    assert!(fired, "the buffered press was dropped on landing");
    assert!(game.cue_seq(CUE_LAND) > quiet, "and it landed on the way");
}

/// The diagonal is not faster than the straight line. A fly camera's
/// traditional feature and a shooter's traditional bug — the normalize is what
/// this asserts, and it asserts it as an equality.
#[test]
fn a_diagonal_is_no_faster_than_a_straight_line() {
    // Twenty ticks: full speed arrives in seven, and a longer run would only
    // find scenery. West and north-west are the clear lines out of the spawn.
    let straight = {
        let mut game = Game::load();
        game.begin();
        game.drive(MOVE_RIGHT, -1, 20);
        speed(&game)
    };
    let diagonal = {
        let mut game = Game::load();
        game.begin();
        for _ in 0..20 {
            game.push(MOVE_FORWARD, 1);
            game.push(MOVE_RIGHT, -1);
            game.step();
        }
        speed(&game)
    };
    // Neither is bit-equal to the constant: `v + (target - v)` lands within an
    // ULP of `target` rather than on it, and the diagonal's components come
    // through a normalize besides. The claim is that the two agree, and a
    // nanometre per tick is a generous way to say so.
    assert!((straight - WALK_SPEED).abs() < 1e-9, "straight {straight}");
    assert!(
        (diagonal - straight).abs() < 1e-9,
        "diagonal {diagonal} vs straight {straight} m/tick"
    );
}

/// Restart puts the body back where a session opens, from wherever it got to.
#[test]
fn restart_returns_the_body_to_the_start() {
    let mut game = Game::load();
    game.begin();
    game.forward(60);
    assert_ne!(game.walker().position.z, START.z);
    game.hold(RESTART);
    game.step();
    // Not `position == START`: `walk` runs after `restart` in the same tick,
    // so the body has already been settled onto the floor by the time anything
    // can look at it — which is the right answer and not a rounded one.
    let walker = game.walker();
    assert_eq!(walker.position.x, START.x);
    assert_eq!(walker.position.z, START.z);
    assert_eq!(walker.position.y, resting_on(0.0));
    assert_eq!((walker.velocity.x, walker.velocity.z), (0.0, 0.0));
    assert_eq!(walker.yaw, 0.0);
}

/// The HUD reads the speed the body is actually doing, in tenths.
#[test]
fn the_speed_row_reads_the_ground_speed() {
    let mut game = Game::load();
    game.begin();
    assert_eq!(game.hud(HUD_SPEED), "SPEED 0.0");
    game.drive(MOVE_RIGHT, -1, 20);
    // 0.110 m/tick at 60 Hz is 6.6 m/s, and the row is integer tenths of it.
    assert_eq!(game.hud(HUD_SPEED), "SPEED 6.6");
}

// ---------------------------------------------------------------- menu ------

/// Nothing on screen and nothing to point at, which is the same sentence: a
/// hidden widget is a zero rect (§4.9), and the host reads "no hit-tested area"
/// as "the game keeps the mouse".
#[test]
fn a_playing_session_offers_the_pointer_nothing_to_point_at() {
    let mut game = Game::load();
    game.begin();
    assert_eq!(game.session().paused, 0);
    for item in MENU_ITEMS {
        let rect = game.menu(item).rect;
        assert_eq!(
            (rect[2], rect[3]),
            (0.0, 0.0),
            "item {item} has area while the game is playing"
        );
    }
}

/// Escape stops the world *on the tick it was pressed*, not one later — which
/// is what putting `menu` before `aim` and `walk` in the table buys.
#[test]
fn escape_stops_the_world_on_the_tick_it_is_pressed() {
    let mut game = Game::load();
    game.begin();
    game.drive(MOVE_FORWARD, 1, 10);
    let moving = game.walker();
    assert_ne!(moving.velocity.z, 0.0);

    game.hold(PAUSE);
    game.push(MOVE_FORWARD, 1);
    game.push(AIM_X, 1);
    game.step();

    let paused = game.walker();
    assert_eq!(game.session().paused, 1);
    assert_eq!(
        paused.position, moving.position,
        "the body moved on the pausing tick"
    );
    assert_eq!(paused.yaw, moving.yaw, "the view turned behind the menu");
    assert!(game.menu(MENU_RESUME).rect[2] > 0.0, "the menu is not up");
}

/// And the mouse keeps arriving while it is up — raw device motion is a device
/// delta and reaches the axis whatever the host does with the pointer — so the
/// refusal has to be the game's. This is the test that fails if `aim` stops
/// checking, and it still is after §6 M45: the host withholds the aim axes too,
/// but one tick later than this, and in-process there is no host at all.
#[test]
fn a_paused_view_does_not_turn_however_hard_the_mouse_moves() {
    let mut game = Game::load();
    game.begin();
    game.tap(PAUSE);
    let before = game.walker().yaw;
    for _ in 0..30 {
        game.push(AIM_X, 1);
        game.step();
    }
    assert_eq!(game.walker().yaw, before);
}

/// The other half, which is the host's (§6 M45): pausing raises the flag the
/// shell arbitrates on, and this crate's own map says which verbs still cross.
/// Read as text — a game crate may not link `gg-input` (§3), and the shape being
/// checked is a line of the file rather than a parse of it.
#[test]
fn pausing_raises_the_flag_the_host_withholds_on() {
    let mut game = Game::load();
    game.begin();
    assert_eq!(game.prefs().modal, 0, "a running game withholds nothing");
    game.tap(PAUSE);
    assert_eq!(game.prefs().modal, 1);
    game.tap(PAUSE);
    assert_eq!(game.prefs().modal, 0, "and it comes back down");

    let keep = include_str!("../input.toml")
        .lines()
        .find(|line| line.trim_start().starts_with("keep ="))
        .expect("demo 12 declares a keep list");
    assert!(
        !keep.contains("aim_"),
        "the camera must not turn behind a menu"
    );
    assert!(!keep.contains("\"fire\""), "nor the gun fire through it");
    assert!(keep.contains("\"pause\""), "and Escape must still leave it");
}

/// Escape layers: out of the settings page, then out of the menu. Two presses
/// from the settings panel is one level each.
#[test]
fn escape_walks_back_one_level_at_a_time() {
    let mut game = Game::load();
    game.begin();
    game.tap(PAUSE);
    game.click(MENU_SETTINGS);
    assert_eq!(game.session().page, PAGE_SETTINGS);

    game.tap(PAUSE);
    assert_eq!(game.session().page, PAGE_MAIN);
    assert_eq!(game.session().paused, 1, "still paused");

    game.tap(PAUSE);
    assert_eq!(game.session().paused, 0);
    // And it comes back to the front page, not the one it left.
    game.tap(PAUSE);
    assert_eq!(game.session().page, PAGE_MAIN);
}

/// Resume hands the world back, and the pointer with it.
#[test]
fn resume_gives_the_world_and_the_pointer_back() {
    let mut game = Game::load();
    game.begin();
    game.tap(PAUSE);
    game.click(MENU_RESUME);
    assert_eq!(game.session().paused, 0);
    assert_eq!(game.menu(MENU_RESUME).rect[2], 0.0);
    game.drive(MOVE_FORWARD, 1, 5);
    assert_ne!(game.walker().velocity.z, 0.0, "the world is still stopped");
}

/// A page shows its own rows and nobody else's — the scrim, the panel and the
/// title being the three that belong to both.
#[test]
fn each_page_shows_its_own_rows() {
    let mut game = Game::load();
    game.begin();
    game.tap(PAUSE);
    assert!(game.menu(MENU_QUIT).rect[3] > 0.0);
    assert_eq!(game.menu(MENU_LOOK_UP).rect[3], 0.0);
    assert_eq!(game.menu(MENU_TITLE).text(), "PAUSED");

    game.click(MENU_SETTINGS);
    assert_eq!(game.menu(MENU_QUIT).rect[3], 0.0);
    assert!(game.menu(MENU_LOOK_UP).rect[3] > 0.0);
    assert_eq!(game.menu(MENU_TITLE).text(), "SETTINGS");

    game.click(MENU_BACK);
    assert!(game.menu(MENU_QUIT).rect[3] > 0.0);
}

// ------------------------------------------------------------ the shape ----

/// The session opens on a screen, and the screen is not the room.
#[test]
fn a_session_opens_on_the_title_and_not_in_the_room() {
    let mut game = Game::load();
    game.step();
    let session = game.session();
    assert_eq!((session.paused, session.page), (1, PAGE_TITLE));
    assert_eq!(game.menu(MENU_TITLE).text(), TITLE_NAME);
    assert_eq!(game.menu(MENU_RULE).text(), RULE);
    assert!(
        game.menu(MENU_START).rect[3] > 0.0,
        "and it offers a way in"
    );
    // Nothing of the round has happened: no course, and a body that has not
    // taken a tick of gravity.
    assert!(game.all::<Target>().is_empty());
    assert!(!game.walker().grounded(), "no physics has run behind it");
    assert_eq!(game.prefs().modal, 1, "the pointer is the menu's");
}

/// The title screen's two ways in, and they agree.
#[test]
fn the_title_screen_is_left_by_the_key_and_by_the_button() {
    for by_key in [true, false] {
        let mut game = Game::load();
        game.step();
        match by_key {
            true => {
                game.tap(BEGIN);
            }
            false => game.click(MENU_START),
        }
        let session = game.session();
        assert_eq!((session.paused, session.page), (0, PAGE_MAIN));
        assert_eq!(
            game.all::<Target>().len(),
            TARGETS_LIVE,
            "the course is dealt"
        );
        assert_eq!(
            game.prefs().modal,
            0,
            "and the pointer goes back to the aim"
        );
        assert!(game.menu(MENU_START).rect[3] == 0.0, "the screen is gone");
    }
}

/// `R` is refused on the title screen. It has nothing to start over there, and
/// a restart would deal a course nobody asked to play — the one place the key
/// the banner advertises does nothing.
#[test]
fn the_restart_key_does_nothing_on_the_title_screen() {
    let mut game = Game::load();
    game.step();
    game.tap(RESTART);
    game.tap(RESTART);
    let session = game.session();
    assert_eq!((session.paused, session.page), (1, PAGE_TITLE));
    assert!(game.all::<Target>().is_empty(), "and no course was dealt");
}

/// One settings panel, two screens, and [`Session::from`] is what tells them
/// apart — the claim §6 M44 named and the only thing that proves the return is
/// a field rather than a hardcoded page.
#[test]
fn settings_returns_to_whichever_screen_opened_it() {
    // From the title.
    let mut game = Game::load();
    game.step();
    game.click(MENU_SETTINGS);
    assert_eq!(game.session().page, PAGE_SETTINGS);
    assert_eq!(game.session().from, PAGE_TITLE);
    game.click(MENU_BACK);
    assert_eq!(
        game.session().page,
        PAGE_TITLE,
        "back to where it came from"
    );
    assert_eq!(game.menu(MENU_TITLE).text(), TITLE_NAME);

    // And from the pause menu, in the same session, which is what stops a
    // sticky `from` from passing this by accident.
    game.tap(BEGIN);
    game.tap(PAUSE);
    game.click(MENU_SETTINGS);
    assert_eq!(game.session().from, PAGE_MAIN);
    game.click(MENU_BACK);
    assert_eq!(game.session().page, PAGE_MAIN);
    assert_eq!(game.menu(MENU_TITLE).text(), "PAUSED");
    assert_eq!(game.session().paused, 1, "the pause menu, not the round");
}

/// The round ends on a screen that says what it came to, and the body stops
/// with it. Through §6 M75 the score froze and the player went on walking
/// around a room that had stopped dealing targets.
#[test]
fn the_round_ends_on_a_screen_that_reports_it() {
    let mut game = Game::load();
    game.begin();
    // Stand still: three targets walk off and that is the whole rule.
    game.steps(u64::from(TARGET_LIFE) + 4);
    let session = game.session();
    assert_eq!((session.paused, session.page), (1, PAGE_OVER));
    assert_eq!(game.menu(MENU_TITLE).text(), "ROUND OVER");
    assert_eq!(game.menu(MENU_TALLY).text(), "SCORE 0  BEST 0");
    assert_eq!(game.menu(MENU_AIM).text(), "WAVE 1  ACCURACY 0%");
    assert_eq!(
        game.menu(MENU_RESTART).text(),
        "AGAIN",
        "the same button, the screen's word"
    );

    // The body is stopped, not merely scoreless.
    let before = game.walker().position;
    game.forward(20);
    assert_eq!(game.walker().position, before, "the world is held");
    assert_eq!(game.prefs().modal, 1, "and the pointer is the menu's");
}

/// AGAIN is a *continuation*: a fresh round that keeps what the last one was
/// worth. The tick it opens at is what says it came from the over screen and
/// not from a fresh boot — §6 M48's way of grading, at a game's scale.
#[test]
fn the_over_screen_starts_another_round_and_keeps_the_best() {
    let mut game = Game::load();
    game.begin();
    let taken = game.visible_target().position;
    game.aim_at(taken);
    game.tap(FIRE);
    let scored = game.range().score;
    assert!(scored > 0, "something was on the board");

    game.steps(u64::from(TARGET_LIFE) + 4);
    assert_eq!(game.session().page, PAGE_OVER);

    game.tap(BEGIN);
    let session = game.session();
    assert_eq!((session.paused, session.page), (0, PAGE_MAIN));
    let range = game.range();
    assert_eq!(
        (range.score, range.hits, range.misses),
        (0, 0, 0),
        "a fresh round"
    );
    assert_eq!(range.best, scored, "carrying what the last one was worth");
    assert_eq!(range.state, STATE_RUNNING);
    assert_eq!(
        game.walker().position.x,
        START.x,
        "and the body is back at the start"
    );
}

/// The pause menu and the over screen both offer the way back to the title, and
/// it is the same row on both.
#[test]
fn the_title_is_reachable_from_the_pause_menu_and_from_the_over_screen() {
    let mut game = Game::load();
    game.begin();
    game.tap(PAUSE);
    game.click(MENU_TO_TITLE);
    assert_eq!(game.session().page, PAGE_TITLE);
    assert_eq!(game.session().paused, 1);

    game.tap(BEGIN);
    game.steps(u64::from(TARGET_LIFE) + 4);
    assert_eq!(game.session().page, PAGE_OVER);
    game.click(MENU_TO_TITLE);
    assert_eq!(game.session().page, PAGE_TITLE);
}

/// Every page reads top to bottom, which is not decoration: `Widget::order` is
/// the draw order *and* the order the host's focus ring walks (§6 M74), so a
/// page whose rows are listed out of order is a page whose `Tab` jumps about.
#[test]
fn every_page_lists_its_rows_in_reading_order() {
    let mut game = Game::load();
    game.step();
    for (page, reach) in [
        (PAGE_TITLE, None),
        (PAGE_MAIN, Some(())),
        (PAGE_SETTINGS, Some(())),
        (PAGE_OVER, Some(())),
    ] {
        let _ = reach;
        // Drive the session onto the page rather than calling `page_rows`, which
        // is private — the claim is about what a player is shown.
        let mut game2 = Game::load();
        game2.step();
        match page {
            PAGE_TITLE => {}
            PAGE_SETTINGS => game2.click(MENU_SETTINGS),
            PAGE_MAIN => {
                game2.tap(BEGIN);
                game2.tap(PAUSE);
            }
            _ => {
                game2.tap(BEGIN);
                game2.steps(u64::from(TARGET_LIFE) + 4);
            }
        }
        assert_eq!(game2.session().page, page);
        // Only the buttons: the ring is a walk over what the host hit-tests,
        // and a label is not that (§6 M74). A settings row's readout sits six
        // units above the `-`/`+` beside it, so including labels would be
        // asserting about a layout nobody can tab to.
        let mut rows: Vec<(u32, f32, f32)> = Vec::new();
        for item in MENU_ITEMS {
            let widget = game2.menu(item);
            let shown = widget.rect[2] > 0.0 && widget.rect[3] > 0.0;
            if shown && widget.kind == boundary::widget::BUTTON {
                rows.push((widget.order, widget.rect[1], widget.rect[0]));
            }
        }
        assert!(rows.len() >= 3, "page {page} offers something to press");
        let mut sorted = rows.clone();
        sorted.sort_by_key(|row| row.0);
        assert_eq!(rows.len(), sorted.len());
        // Down the page, then across it: two buttons sharing a row (a `-` and
        // its `+`) are visited left to right, and only a row that goes back
        // *up* is a fault.
        for pair in sorted.windows(2) {
            let forward =
                pair[0].1 < pair[1].1 || (pair[0].1 == pair[1].1 && pair[0].2 < pair[1].2);
            assert!(
                forward,
                "page {page}: the ring goes from ({}, {}) back to ({}, {})",
                pair[0].2, pair[0].1, pair[1].2, pair[1].1
            );
        }
        let _ = &mut game;
    }
}

/// A row nobody put on a page draws nowhere. The predecessor of `page_rows` had
/// a catch-all arm that put every unclaimed row on the *settings* page, so a
/// menu grew rows it was never shown (§6 M76).
#[test]
fn no_page_shows_a_row_that_belongs_to_another() {
    let mut game = Game::load();
    game.step();
    // The title screen: the round's rows are not on it.
    for item in [
        MENU_RESUME,
        MENU_BACK,
        MENU_LOOK,
        MENU_AA,
        MENU_TALLY,
        MENU_AIM,
    ] {
        assert_eq!(
            game.menu(item).rect[3],
            0.0,
            "item {item} is not on the title"
        );
    }
    game.click(MENU_SETTINGS);
    for item in [
        MENU_START,
        MENU_RULE,
        MENU_TALLY,
        MENU_AIM,
        MENU_RESUME,
        MENU_TO_TITLE,
    ] {
        assert_eq!(
            game.menu(item).rect[3],
            0.0,
            "item {item} is not in settings"
        );
    }
}

/// A line that reports something is not a thing to press.
///
/// The three summary rows were dealt as *buttons* when they were added — the
/// arm that makes a label listed three items and the new ones fell past it —
/// which is a focus ring that stops on a score and an Enter that presses it
/// (§6 M74's ring walks exactly the host-hit-tested widgets, and that is
/// exactly the buttons).
#[test]
fn a_row_that_reports_something_is_not_a_row_to_press() {
    let mut game = Game::load();
    game.step();
    for item in [
        MENU_TITLE,
        MENU_RULE,
        MENU_TALLY,
        MENU_AIM,
        MENU_LOOK,
        MENU_VOLUME,
    ] {
        assert_ne!(
            game.menu(item).kind,
            boundary::widget::BUTTON,
            "item {item} reads something out; it is not a button"
        );
    }
    for item in [
        MENU_START,
        MENU_SETTINGS,
        MENU_QUIT,
        MENU_RESTART,
        MENU_TO_TITLE,
        MENU_BACK,
    ] {
        assert_eq!(
            game.menu(item).kind,
            boundary::widget::BUTTON,
            "item {item} does something; it is a button"
        );
    }
}

/// The HUD row that carries the arc. Two facts on one line because they are one
/// question — how the round is going — and a sixth row would be a sixth thing
/// to read while aiming.
#[test]
fn the_round_row_reads_the_wave_and_the_misses() {
    let mut game = Game::load();
    game.begin();
    assert_eq!(game.hud(HUD_ROUND), "WAVE 1  MISS 0/3");
    // Take a wave's worth, and the row moves with it.
    for _ in 0..WAVE_HITS {
        let taken = game.visible_target().position;
        game.aim_at(taken);
        game.tap(FIRE);
        game.steps(u64::from(FIRE_TICKS) + 1);
    }
    assert_eq!(game.hud(HUD_ROUND), "WAVE 2  MISS 0/3");
}

/// The title screen says the same name the window bar, the shipped folder and
/// the executable's version resource do (§6 M73, M75). Two files, so the check
/// is a test rather than a shared constant.
#[test]
fn the_title_screen_and_the_manifest_agree_on_the_name() {
    let manifest = include_str!("../game.ggproj");
    let title = manifest
        .lines()
        .find_map(|line| line.strip_prefix("title"))
        .and_then(|rest| rest.split_once('='))
        .map(|(_, value)| value.trim())
        .expect("the manifest names the game");
    assert_eq!(
        title.to_ascii_uppercase(),
        TITLE_NAME,
        "the title screen and `game.ggproj` disagree about the name"
    );
}

// ------------------------------------------------------------ the arc ------

/// The ramp: shorter every wave, down to a floor, and integer all the way —
/// demo 10's `gravity_for` in this game's units.
#[test]
fn a_target_stands_for_less_every_wave_down_to_a_floor() {
    assert_eq!(life_for(0), TARGET_LIFE, "the first wave is the constant");
    let mut last = life_for(0);
    for wave in 1..40 {
        let life = life_for(wave);
        assert!(
            life <= last,
            "wave {wave} is not longer than the one before"
        );
        assert!(life >= LIFE_FLOOR, "and never under the floor");
        last = life;
    }
    assert_eq!(life_for(40), LIFE_FLOOR, "and it gets there");

    // The floor is a floor and not a wall: taking one target costs a turn, a
    // trigger and the cooldown, and `TARGETS_LIVE` stand at once, so a life
    // under that could not be cleared by a shot that never missed. The
    // inequality rather than the number, so retuning the weapon cannot quietly
    // make the course impossible.
    assert!(
        LIFE_FLOOR >= (FIRE_TICKS + 2) * TARGETS_LIVE as u32,
        "the floor leaves no time to clear the course"
    );

    // And the pay rises where the time falls.
    assert_eq!(worth_for(0), TARGET_WORTH);
    assert!(worth_for(3) > worth_for(2) && worth_for(2) > worth_for(1));
    assert_eq!(wave_of(0), 0);
    assert_eq!(wave_of(WAVE_HITS - 1), 0);
    assert_eq!(wave_of(WAVE_HITS), 1, "the wave is read off the hits");
}

/// And the ramp reaches the world: the target dealt after the wave turns over
/// stands for less than the ones the round opened with.
#[test]
fn the_wave_shortens_the_targets_the_round_deals_next() {
    let mut game = Game::load();
    game.begin();
    for target in game.all::<Target>() {
        assert_eq!(
            target.life, TARGET_LIFE,
            "the opening course is the first wave"
        );
    }
    // Take WAVE_HITS targets, which is what turns the wave over.
    for _ in 0..WAVE_HITS {
        let taken = game.visible_target().position;
        game.aim_at(taken);
        game.tap(FIRE);
        game.steps(u64::from(FIRE_TICKS) + 1);
    }
    let range = game.range();
    assert_eq!(range.hits, WAVE_HITS, "every round found its target");
    assert_eq!(wave_of(range.hits), 1);
    let fresh: Vec<u32> = game
        .all::<Target>()
        .iter()
        .filter(|t| t.age == 0 || t.life != TARGET_LIFE)
        .map(|t| t.life)
        .collect();
    assert!(
        fresh.iter().any(|life| *life == life_for(1)),
        "a target dealt into the second wave stands for {} and the course holds {fresh:?}",
        life_for(1)
    );
    assert!(life_for(1) < TARGET_LIFE);
}

/// The number `Range::shots` has been counted for since §6 M37 and that nothing
/// showed until there was a screen to show it on.
#[test]
fn accuracy_is_hits_over_shots_and_a_round_that_fired_nothing_reads_zero() {
    let mut game = Game::load();
    game.step();
    let mut range = game.range();
    assert_eq!((range.shots, range.hits), (0, 0));
    assert_eq!(accuracy(&range), 0, "no shots is not a division");
    range.shots = 4;
    range.hits = 1;
    assert_eq!(accuracy(&range), 25);
    range.hits = 4;
    assert_eq!(accuracy(&range), 100);
}

/// Each verb is where its id says it is. A list read by `gg_game!` and by
/// `input.toml`, and the only thing keeping the two in step (§4.7).
#[test]
fn every_verb_sits_at_the_id_that_names_it() {
    assert_eq!(ACTIONS.len(), 8);
    assert_eq!(ACTIONS[JUMP.index()], "jump");
    assert_eq!(ACTIONS[RESTART.index()], "restart");
    assert_eq!(ACTIONS[PAUSE.index()], "pause");
    assert_eq!(ACTIONS[FIRE.index()], "fire");
    assert_eq!(ACTIONS[BEGIN.index()], "begin");
    assert_eq!(
        *ACTIONS.last().unwrap(),
        "begin",
        "appended, never inserted"
    );
    assert_eq!(AXES.len(), 6);
    assert_eq!(AXES[AIM_X.index()], "aim_x");
    assert_eq!(AXES[MOVE_FORWARD.index()], "move_forward");

    // The map declares every one of them, and `begin` crosses while a screen is
    // up or it could not leave one.
    let map = include_str!("../input.toml");
    for verb in ACTIONS {
        assert!(
            map.contains(&format!("{verb} = [")),
            "`{verb}` has no binding"
        );
    }
    let keep = map
        .lines()
        .find(|line| line.starts_with("keep = "))
        .expect("a modal list");
    assert!(
        keep.contains("\"begin\""),
        "a key withheld by the menu cannot start a round"
    );
}

/// The look row moves the multiplier, and the multiplier moves the aim.
/// Asserted through the *angle* and not the field: a setting that changed a
/// number nothing read would still pass a test on the number.
#[test]
fn the_look_row_changes_what_a_mouse_count_is_worth() {
    let mut game = Game::load();
    game.begin();
    game.tap(PAUSE);
    game.click(MENU_SETTINGS);
    let opening = game.session().sens;
    assert_eq!(opening, SENS_DEFAULT);

    game.click(MENU_LOOK_DOWN);
    assert_eq!(game.session().sens, opening - SENS_STEP);

    // Out of the menu, then turn: one full-deflection tick of AIM_X is one
    // unit, so the yaw it produces *is* the radians-per-count.
    game.tap(PAUSE);
    game.tap(PAUSE);
    let before = game.walker().yaw;
    game.push(AIM_X, 1);
    game.step();
    assert_eq!(
        game.walker().yaw,
        before - look_per_count(opening - SENS_STEP)
    );
}

/// Both ends of the range hold, and a click at the end is a no-op rather than a
/// wrap — `saturating_*` plus a clamp, because a `u32` sensitivity that
/// underflowed would be the fastest mouse ever sold.
#[test]
fn the_sensitivity_range_holds_at_both_ends() {
    let mut game = Game::load();
    game.begin();
    game.tap(PAUSE);
    game.click(MENU_SETTINGS);
    for _ in 0..200 {
        game.click(MENU_LOOK_DOWN);
    }
    assert_eq!(game.session().sens, SENS_MIN);
    for _ in 0..200 {
        game.click(MENU_LOOK_UP);
    }
    assert_eq!(game.session().sens, SENS_MAX);
}

/// The readout says what the setting is worth in the unit a hand is set
/// against: 12 566 counts per turn at 1.00 is 40 cm of desk at 800 DPI.
#[test]
fn the_look_row_reports_counts_per_turn() {
    let mut game = Game::load();
    game.begin();
    game.tap(PAUSE);
    game.click(MENU_SETTINGS);
    assert_eq!(game.menu(MENU_LOOK).text(), "LOOK 1.50  8377/TURN");
    assert_eq!(
        counts_per_turn(SENS_ONE),
        12566,
        "the unit, which the default is 1.5 of"
    );
    assert_eq!(counts_per_turn(SENS_DEFAULT), 8377);
    // Twice the sensitivity is half the desk, which is the whole point of the
    // unit: the two numbers cannot drift apart because one is derived.
    assert_eq!(counts_per_turn(SENS_ONE * 2), counts_per_turn(SENS_ONE) / 2);
}

/// Volume is attenuation, so the *down* button raises `quiet` — and the row
/// reads the percentage a human expects either way.
#[test]
fn the_volume_row_attenuates_and_reads_as_a_percentage() {
    let mut game = Game::load();
    game.begin();
    game.tap(PAUSE);
    game.click(MENU_SETTINGS);
    assert_eq!(game.menu(MENU_VOLUME).text(), "VOLUME 100%");
    assert_eq!(game.prefs().quiet, 0);

    game.click(MENU_VOLUME_DOWN);
    assert_eq!(game.menu(MENU_VOLUME).text(), "VOLUME 87%");
    assert!(game.prefs().quiet > 0);

    for _ in 0..20 {
        game.click(MENU_VOLUME_DOWN);
    }
    assert_eq!(game.menu(MENU_VOLUME).text(), "VOLUME 0%");
    assert_eq!(
        game.prefs().quiet,
        QUIET_MAX,
        "past silence, not through it"
    );
    assert_eq!(game.prefs().volume(), 0.0);
}

/// The edge row cycles every mode that *says* something and never `DEFAULT` —
/// a row reading OFF while the field means "ask the host" is a row that lies
/// the moment `r.aa` or `r.msaa` is on — and its text names the mode it is in.
///
/// The pairs are the point: each mode states both of the host's knobs, so an
/// MSAA row must also say the post pass is off. A mode that left `antialias()`
/// alone would run FXAA over the samples whenever `r.aa` happened to be on.
#[test]
fn the_edge_row_names_the_mode_it_is_actually_in() {
    let mut game = Game::load();
    game.begin();
    game.tap(PAUSE);
    game.click(MENU_SETTINGS);
    assert_eq!(game.prefs().aa, aa::MSAA_4, "the demo opens antialiased");

    // One full lap, starting where `bootstrap` left it.
    let lap = [
        (aa::MSAA_8, "EDGES  MSAA 8X", Some(false), Some(8)),
        (aa::OFF, "EDGES  OFF", Some(false), Some(1)),
        (aa::FXAA, "EDGES  FXAA", Some(true), Some(1)),
        (aa::MSAA_2, "EDGES  MSAA 2X", Some(false), Some(2)),
        (aa::MSAA_4, "EDGES  MSAA 4X", Some(false), Some(4)),
    ];
    assert_eq!(game.menu(MENU_AA).text(), "EDGES  MSAA 4X");
    for (mode, text, antialias, samples) in lap {
        game.click(MENU_AA);
        assert_eq!(game.prefs().aa, mode);
        assert_eq!(game.menu(MENU_AA).text(), text);
        assert_eq!(game.prefs().antialias(), antialias);
        assert_eq!(game.prefs().samples(), samples);
    }
    assert_eq!(
        game.prefs().aa,
        aa::MSAA_4,
        "the lap came back to where it started"
    );
}

/// Restarting from the menu puts the body back *and* returns to the game — and
/// leaves the settings alone, which is the reason they do not live on the body.
#[test]
fn the_menu_restart_respawns_without_clearing_the_settings() {
    let mut game = Game::load();
    game.begin();
    game.tap(PAUSE);
    game.click(MENU_SETTINGS);
    game.click(MENU_LOOK_DOWN);
    game.click(MENU_BACK);
    game.click(MENU_RESUME);
    game.forward(60);
    assert_ne!(game.walker().position.z, START.z);

    game.tap(PAUSE);
    game.click(MENU_RESTART);
    assert_eq!(game.walker().position.x, START.x);
    assert_eq!(game.walker().position.z, START.z);
    assert_eq!(game.session().paused, 0, "a restart is not a pause");
    assert_eq!(game.session().sens, SENS_DEFAULT - SENS_STEP);
}

/// Quit is the one thing a game could not say before (§6 M21): it asks through
/// `Prefs::close`, which the host reads every tick.
#[test]
fn quit_asks_the_host_to_end_the_session() {
    let mut game = Game::load();
    game.begin();
    assert!(!game.prefs().closing());
    game.tap(PAUSE);
    game.click(MENU_QUIT);
    assert!(game.prefs().closing());
    // Monotone: nothing has to remember to clear it, and nothing does.
    game.steps(5);
    assert!(game.prefs().closing());
}

// ------------------------------------------------- what the gun is read by --

impl Game {
    fn range(&self) -> Range {
        self.one::<Range>()
    }

    /// Every target standing, with the box it is.
    fn targets(&self) -> Vec<(Target, Renderable)> {
        let query = Query::<(&Target, &Renderable)>::new().unwrap();
        let mut out = Vec::new();
        self.world
            .each_ref(&query, |_, (target, shape): (&Target, &Renderable)| {
                out.push((*target, *shape));
            });
        out
    }

    /// Turn to face `at`, **through the aim axes** exactly as a mouse would —
    /// the counts are what the sensitivity turns into an angle, and a test that
    /// wrote the field would be exercising a different game. Costs one tick and
    /// lands inside one mouse count (0.043°) of the point asked for.
    fn aim_at(&mut self, at: sim::DVec3) {
        let eye = self.one::<Eye>().position;
        let to = (at - eye)
            .try_normalize()
            .expect("the eye is not on the point it is aiming at");
        // `sim::fly_basis` inverted: forward is `(-cosP sinY, sinP, -cosP cosY)`.
        let yaw = sim::atan2(-to.x, -to.z) as f32;
        let pitch = sim::asin(to.y) as f32;
        let walker = self.walker();
        let per = f64::from(look_per_count(self.session().sens));
        let counts = |delta: f32| (-f64::from(delta) / per * f64::from(STICK)) as i32;
        self.axes[AIM_X.index()] = counts(yaw - walker.yaw);
        self.axes[AIM_Y.index()] = counts(pitch - walker.pitch);
        self.step();
    }

    /// The first standing target with a clear line from the eye.
    ///
    /// The course deals spots the room can hide — a pillar is a pillar from
    /// somewhere, and that is the level being a level — so a test that wants a
    /// hit picks one it can *see* rather than the first in archetype order.
    /// The reader casts its own ray against [`ROOM`] to decide, which is
    /// choosing a scenario rather than asserting the predicate under test.
    fn visible_target(&self) -> Renderable {
        let eye = self.one::<Eye>().position;
        self.targets()
            .into_iter()
            .map(|(_, shape)| shape)
            .find(|shape| {
                let to = shape.position - eye;
                let reach = to.length();
                let ray = sim::DRay::new(eye, to / reach);
                !ROOM.iter().any(|(position, half, _)| {
                    let extent =
                        sim::DVec3::new(f64::from(half.x), f64::from(half.y), f64::from(half.z));
                    ray.obb(*position, sim::DQuat::IDENTITY, extent)
                        .is_some_and(|span| span.enter > 0.0 && span.enter < reach)
                })
            })
            .expect("something in the course is in view from here")
    }

    /// Put a target of the test's own at `at`, standing until it is shot.
    ///
    /// Its slot is past the end of [`SPOTS`], so the course neither counts it
    /// as one of its own nor ever deals a second target on top of it.
    fn plant(&mut self, at: sim::DVec3) {
        let entity = self.world.spawn();
        self.world
            .insert(
                entity,
                Target {
                    age: 0,
                    life: u32::MAX,
                    slot: SPOTS.len() as u32,
                    worth: TARGET_WORTH,
                },
            )
            .unwrap();
        self.world
            .insert(
                entity,
                Renderable::boxed(at, sim::Vec3::splat(TARGET_HALF), TARGET_INK),
            )
            .unwrap();
    }
}

/// Whether a box of half-extent `pad` centred on `at` misses `slab` entirely.
fn clear_of(at: sim::DVec3, pad: f64, slab: (sim::DVec3, sim::Vec3)) -> bool {
    let (position, half) = slab;
    (at.x - position.x).abs() >= f64::from(half.x) + pad
        || (at.y - position.y).abs() >= f64::from(half.y) + pad
        || (at.z - position.z).abs() >= f64::from(half.z) + pad
}

/// The room's own bounds, derived from every slab in [`ROOM`] — what a renderer
/// fits a grid to when the room is all there is.
fn room_bounds() -> (sim::DVec3, sim::DVec3) {
    let mut low = sim::DVec3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut high = sim::DVec3::new(f64::MIN, f64::MIN, f64::MIN);
    for (position, half, _) in ROOM {
        let (near, far) = (
            sim::DVec3::new(
                position.x - f64::from(half.x),
                position.y - f64::from(half.y),
                position.z - f64::from(half.z),
            ),
            sim::DVec3::new(
                position.x + f64::from(half.x),
                position.y + f64::from(half.y),
                position.z + f64::from(half.z),
            ),
        );
        low = sim::DVec3::new(low.x.min(near.x), low.y.min(near.y), low.z.min(near.z));
        high = sim::DVec3::new(high.x.max(far.x), high.y.max(far.y), high.z.max(far.z));
    }
    (low, high)
}

/// A rotated box's axis-aligned extent — the sum of its three turned half-axes,
/// which is what a bound has to be for a tracer that lies along a shot.
fn world_extent(shape: &Renderable) -> sim::DVec3 {
    let half = sim::DVec3::new(
        f64::from(shape.half_extent.x),
        f64::from(shape.half_extent.y),
        f64::from(shape.half_extent.z),
    );
    let x = shape.rotation.rotate(sim::DVec3::new(half.x, 0.0, 0.0));
    let y = shape.rotation.rotate(sim::DVec3::new(0.0, half.y, 0.0));
    let z = shape.rotation.rotate(sim::DVec3::new(0.0, 0.0, half.z));
    sim::DVec3::new(
        x.x.abs() + y.x.abs() + z.x.abs(),
        x.y.abs() + y.y.abs() + z.y.abs(),
        x.z.abs() + y.z.abs() + z.z.abs(),
    )
}

// ------------------------------------------------------------ the gun ------

/// Every spawn point stands in the room's clear air.
///
/// Derived from [`ROOM`] rather than restated, demo 11's pull-2 doctrine: a
/// slab that moves re-tests this instead of leaving a target inside a pillar
/// where nothing can be shot and every round ends in three misses.
#[test]
fn every_spawn_point_stands_in_the_rooms_clear_air() {
    for spot in SPOTS {
        for (position, half, _) in ROOM {
            assert!(
                clear_of(*spot, f64::from(TARGET_HALF), (*position, *half)),
                "a target at {spot:?} stands inside the slab at {position:?}"
            );
        }
        assert!(
            top_at(spot.x, spot.z, spot.y) >= 0.0,
            "a target at {spot:?} hangs over no floor at all"
        );
    }
}

/// A shot takes the target it is pointed at, and the score says so.
#[test]
fn a_shot_takes_the_target_it_is_pointed_at() {
    let mut game = Game::load();
    game.begin();
    assert_eq!(game.targets().len(), TARGETS_LIVE, "the course opens full");
    let taken = game.visible_target().position;

    game.aim_at(taken);
    game.hold(FIRE);
    game.step();

    let range = game.range();
    assert_eq!(range.shots, 1);
    assert_eq!(range.hits, 1);
    assert_eq!(range.streak, 1);
    assert_eq!(range.score, TARGET_WORTH, "the first hit pays no streak");
    assert_eq!(
        range.hitmark,
        HITMARK_TICKS - 1,
        "lit on the shot's own tick, and `targets` has already aged it once"
    );
    assert!(
        game.targets()
            .iter()
            .all(|(_, shape)| shape.position != taken),
        "the one it took is gone"
    );
    assert_eq!(
        game.targets().len(),
        TARGETS_LIVE,
        "and the course refilled"
    );
}

/// The room stops a bullet without knowing it is a wall: one cast, targets and
/// solids together, and the nearest surface wins.
///
/// Two targets planted by the test on the *same line* — one in front of a
/// pillar and one behind it — which is the pair that fails in opposite
/// directions. The pillar is found in [`ROOM`] by shape rather than named, so
/// this re-derives if the room is ever re-authored.
#[test]
fn a_pillar_between_the_eye_and_a_target_stops_the_bullet() {
    let mut game = Game::load();
    game.begin();
    let eye = game.one::<Eye>().position;
    let (pillar, half, _) = ROOM
        .iter()
        .find(|(_, half, _)| half.x < 1.0 && half.z < 1.0 && half.y > 1.5)
        .expect("the room has a pillar to hide behind");
    assert!(
        f64::from(half.y) * 2.0 > eye.y,
        "and it is tall enough to be in the way at eye height"
    );
    let toward = sim::DVec3::new(pillar.x - eye.x, 0.0, pillar.z - eye.z);
    let along = toward
        .try_normalize()
        .expect("the eye is not in the pillar");
    let reach = toward.length();

    for (side, distance, taken) in [
        ("in front of", reach - 1.0, true),
        ("behind", reach + 2.0, false),
    ] {
        let at = eye + along * distance;
        for (position, extent, _) in ROOM {
            assert!(
                clear_of(at, f64::from(TARGET_HALF), (*position, *extent)),
                "the plant {side} the pillar is inside the slab at {position:?}"
            );
        }
        let before = game.range().hits;
        game.plant(at);
        game.aim_at(at);
        game.hold(FIRE);
        game.step();
        assert_eq!(
            game.range().hits > before,
            taken,
            "a target {side} the pillar"
        );
        game.steps(u64::from(FIRE_TICKS));
    }
}

/// A held trigger answers at the fire rate and not faster.
#[test]
fn the_trigger_answers_at_the_fire_rate_and_not_faster() {
    let mut game = Game::load();
    game.begin();
    let rounds = 4;
    for _ in 0..u64::from(FIRE_TICKS) * rounds {
        game.hold(FIRE);
        game.step();
    }
    assert_eq!(
        u64::from(game.range().shots),
        rounds,
        "one round per FIRE_TICKS while it is held"
    );
}

/// The kick settles exactly where it found the view — dyadic constants, so
/// eight givings-back are the one taking, to the bit.
#[test]
fn the_kick_settles_exactly_where_it_found_the_view() {
    let mut game = Game::load();
    game.begin();
    let before = game.walker().pitch;
    game.hold(FIRE);
    game.step();
    assert_eq!(
        game.walker().pitch,
        before + RECOIL_KICK,
        "the kick lands on the shot's own tick"
    );
    game.steps(u64::from(RECOIL_TICKS));
    assert_eq!(game.walker().pitch, before, "and is given back exactly");
}

/// The streak pays, and a shot that meets nothing ends it.
#[test]
fn the_streak_pays_and_a_shot_into_the_sky_ends_it() {
    let mut game = Game::load();
    game.begin();
    let mut expected = 0;
    for hit in 0..3 {
        let at = game.visible_target().position;
        game.aim_at(at);
        game.hold(FIRE);
        game.step();
        expected += TARGET_WORTH + STREAK_STEP * hit;
        assert_eq!(game.range().score, expected, "hit {hit}");
        assert_eq!(game.range().streak, hit + 1);
        game.steps(u64::from(FIRE_TICKS));
    }

    // The room is closed on five sides and open above, so this is the one aim
    // that meets no surface at all.
    let sky = game.one::<Eye>().position + sim::DVec3::new(0.0, 10.0, -0.5);
    game.aim_at(sky);
    game.hold(FIRE);
    game.step();
    assert_eq!(game.range().streak, 0, "a shot that hits nothing is a miss");
    assert_eq!(game.range().score, expected, "and pays nothing");
}

/// Three targets that walk end the round, and a finished round takes no more
/// shots. The three the course opens with expire together, which is what makes
/// standing still for one target's life the whole script.
#[test]
fn three_targets_that_walk_end_the_round() {
    let mut game = Game::load();
    game.begin();
    game.steps(u64::from(TARGET_LIFE) + 2);

    let range = game.range();
    assert_eq!(range.misses, MISSES_ALLOWED);
    assert_eq!(range.state, STATE_OVER);
    assert_eq!(range.score, 0);
    assert!(game.targets().is_empty(), "and the course is not refilled");

    game.hold(FIRE);
    game.step();
    assert_eq!(
        game.range().shots,
        0,
        "a finished round takes no more shots"
    );
    assert_eq!(game.hud(HUD_ROUND), "OVER - R TO RESTART");
}

/// A restart reopens the course and keeps the best — the one number a session
/// accumulates, and the only thing a save would have to carry.
#[test]
fn a_restart_reopens_the_course_and_keeps_the_best() {
    let mut game = Game::load();
    game.begin();
    let at = game.visible_target().position;
    game.aim_at(at);
    game.hold(FIRE);
    game.step();
    let scored = game.range().score;
    assert!(scored > 0);

    game.tap(RESTART);
    let range = game.range();
    assert_eq!(range.score, 0);
    assert_eq!(range.best, scored);
    assert_eq!(range.state, STATE_RUNNING);
    assert_eq!(range.streak, 0);
    assert_eq!(game.walker().position.x, START.x, "and the body is back");
    assert_eq!(game.walker().position.z, START.z);
    assert_eq!(game.targets().len(), TARGETS_LIVE, "on a fresh course");
    assert_eq!(game.all::<Spark>().len(), 0, "with last round's chips gone");
}

/// The menu takes the trigger with the pointer: one physical button, two jobs,
/// and the same fact decides both. The click that closes the menu is not a
/// shot, and the trigger stays refused until it is let go of.
#[test]
fn the_menu_takes_the_trigger_and_gives_it_back_on_a_release() {
    let mut game = Game::load();
    game.begin();
    game.tap(PAUSE);
    for _ in 0..u64::from(FIRE_TICKS) * 3 {
        game.hold(FIRE);
        game.step();
    }
    assert_eq!(game.range().shots, 0, "the button is the menu's");

    game.hold(PAUSE);
    game.hold(FIRE);
    game.step();
    assert_eq!(game.session().paused, 0, "the menu closed");
    assert_eq!(
        game.range().shots,
        0,
        "the click that resumed is not a shot"
    );
    game.hold(FIRE);
    game.step();
    assert_eq!(game.range().shots, 0, "and stays refused while it is held");
    game.step();
    game.hold(FIRE);
    game.step();
    assert_eq!(
        game.range().shots,
        1,
        "released and pressed again, it fires"
    );
}

/// Everything a level shot leaves behind lives and dies inside the room
/// (§6 M37 item 3): the tracer runs muzzle to contact, a chip is spent where it
/// meets the floor, and the flash sits at the eye. The renderer fits its light
/// grid and its per-frame batch bounds to the scene, so this is the claim that
/// says neither has to move for a shot.
///
/// The exception is measured rather than hidden — see
/// [`a_shot_at_the_sky_is_the_one_transient_that_leaves`].
#[test]
fn every_transient_a_level_shot_deals_dies_inside_the_room() {
    let mut game = Game::load();
    // 240 firing ticks outlive a course, and `effects` is one of the systems
    // an over screen stops — so without this the sparks stop ageing rather
    // than failing to die, which is a different claim entirely.
    game.forgiving().begin();
    let (low, high) = room_bounds();
    let mut peak = 0;
    for _ in 0..240 {
        game.hold(FIRE);
        game.step();
        peak = peak.max(game.all::<Spark>().len());
        for shape in game.all::<Renderable>() {
            let extent = world_extent(&shape);
            let (near, far) = (shape.position - extent, shape.position + extent);
            assert!(
                near.x >= low.x && near.y >= low.y && near.z >= low.z,
                "something reaches out of the room at {:?}",
                shape.position
            );
            assert!(
                far.x <= high.x && far.y <= high.y && far.z <= high.z,
                "something reaches out of the room at {:?}",
                shape.position
            );
        }
    }
    assert!(peak > 0, "four seconds of fire dealt something");
    // A shot bursts and so does an *escape*, and a course held open by
    // `forgiving` escapes for ever — so the peak now includes a whole set of
    // three walking off inside one spark's life, which the round ending used
    // to cut short. Bounded is still the claim; the bound is what moved.
    assert!(
        peak < 48,
        "and the churn stayed bounded: {peak} at its peak"
    );

    game.steps(u64::from(SPARK_TICKS) + 2);
    assert_eq!(
        game.all::<Spark>().len(),
        0,
        "and all of it went away again"
    );
}

/// The one transient that leaves: a shot with nothing to hit draws its tracer
/// to [`SHOT_RANGE`], and the room is open above. Named and measured rather
/// than assumed away — §6 M37 item 3's answer is conditional on it.
#[test]
fn a_shot_at_the_sky_is_the_one_transient_that_leaves() {
    let mut game = Game::load();
    game.begin();
    let (_, high) = room_bounds();
    let sky = game.one::<Eye>().position + sim::DVec3::new(0.0, 10.0, -0.5);
    game.aim_at(sky);
    game.hold(FIRE);
    game.step();

    let above = game
        .all::<Renderable>()
        .into_iter()
        .map(|shape| shape.position.y + world_extent(&shape).y)
        .fold(f64::MIN, f64::max);
    assert!(
        above > high.y,
        "the tracer of a shot that met nothing stayed inside the room"
    );
    assert!(
        above < high.y + SHOT_RANGE,
        "and it reached no further than the shot did"
    );
}

/// The flash is the nearest point light to the eye while it lives, and gone
/// after — the game's half of §6 M37 item 4.
///
/// The renderer ranks casting slots by exactly that distance, so this is what
/// makes `gg_render::lamp`'s tests (which synthesize lights) claims about *this*
/// scene: the muzzle is nearer than either of [`LAMPS`], every shot, so a flash
/// always takes row zero and every established lamp shifts down one.
#[test]
fn the_flash_is_the_nearest_point_light_and_leaves_when_it_dies() {
    let mut game = Game::load();
    game.begin();
    let eye = game.one::<Eye>().position;
    let points = |game: &Game| {
        let mut out: Vec<f64> = game
            .all::<Light>()
            .into_iter()
            .filter(|light| light.kind == boundary::light::POINT)
            .map(|light| (light.position - eye).length())
            .collect();
        out.sort_by(f64::total_cmp);
        out
    };
    assert_eq!(
        points(&game).len(),
        LAMPS.len(),
        "the room's own, and no more"
    );

    game.hold(FIRE);
    game.step();
    let lit = points(&game);
    assert_eq!(lit.len(), LAMPS.len() + 1, "and the shot's");
    assert!(
        lit[0] < 1.0,
        "which is at the muzzle, not the room: {}",
        lit[0]
    );
    assert!(
        lit[0] < lit[1],
        "so it is the one a casting slot goes to first"
    );

    game.steps(u64::from(FLASH_TICKS) + 1);
    assert_eq!(
        points(&game).len(),
        LAMPS.len(),
        "and it gives the slot back"
    );
}

// ---------------------------------------------- the recorded session (M37) --

/// This crate's own exports as a [`session::Entry`] — the same four pointers
/// `xtask` resolves out of the built dylib.
fn entry() -> session::Entry {
    session::Entry {
        abi: gg_game_abi,
        init: gg_game_init,
        components: gg_game_components,
        systems: gg_game_systems,
    }
}

/// The recorded session is the game's whole shape (§6 M76): the title screen,
/// the round, the wave it crosses, the last miss, the screen that reports it
/// and one more round begun from that screen — every number here re-pinned by
/// `xtask replay --bless` when the room, the course or the weapon changes
/// (§6 M37 item 2).
#[test]
fn the_recorded_session_plays_a_round_out_to_the_last_miss() {
    let frames = session::frames(&entry()).unwrap();
    let progress = session::progress(&entry(), &frames).unwrap();
    let at =
        |find: &dyn Fn(&session::Progress) -> bool| progress.iter().position(find).unwrap() as u64;

    // It opens on a screen and not in the room.
    let first = progress[0];
    assert!(first.stopped && first.page == PAGE_TITLE);
    assert_eq!(first.hits + first.misses + first.shots, 0);

    let started = at(&|p| !p.stopped);
    let (hit, wave, escaped) = (
        at(&|p| p.hits == 1),
        at(&|p| p.hits == WAVE_HITS),
        at(&|p| p.misses == 1),
    );
    let over = at(&|p| p.page == PAGE_OVER);
    // The second round: the first tick after the over screen that is playing
    // again. Searched from `over` rather than from 0, because "playing" is
    // the state most of this stream is in.
    let again = over
        + progress[over as usize..]
            .iter()
            .position(|p| !p.stopped)
            .unwrap() as u64;

    // Every edge in order and each with room between them: a session that only
    // fired would never reach `escaped`, one that only stood still would never
    // reach `hit` or `wave`, one that opened in the room would have no
    // `started`, and one that stopped at the last miss would have no `again` —
    // and every one of those is still a legal input stream.
    assert!(
        started < hit && hit < wave && wave < escaped && escaped < over && over < again,
        "the round changed shape"
    );
    assert_eq!(
        (frames.len(), started, hit, wave, escaped, over, again),
        (303, 1, 15, 70, 212, 271, 272),
        "the round changed shape — re-bless"
    );

    // The round that ended: six taken, the course out of misses.
    let ended = progress[over as usize];
    assert_eq!(ended.hits, session::HITS_WANTED, "the bot stopped shooting");
    assert_eq!(ended.misses, MISSES_ALLOWED, "and the course ran out");
    assert!(ended.over);

    // And the one that followed it: a fresh round carrying the old one's best,
    // which is the whole difference between playing again and starting the
    // process again.
    let last = *progress.last().unwrap();
    assert!(
        !last.stopped && !last.over,
        "the tail is a round in progress"
    );
    assert_eq!((last.hits, last.misses, last.score), (0, 0, 0));
    assert_eq!(last.best, ended.score, "and it remembers what it beat");
    assert_eq!(last.standing, TARGETS_LIVE as u32, "on a course of its own");
    let last = ended;
    // Five rounds for three targets. The two that went astray are the policy
    // working, not failing: this room has spots the spawn cannot see, the bot
    // finds them the only way it can — by firing — and never fires at one
    // twice. A session that shot five for five would be covering neither the
    // dust burst nor the streak reset.
    assert!(last.shots > last.hits, "no round ever met the room");
    assert_eq!(
        (last.shots, last.score, last.standing),
        (8, 800, 2),
        "the room, the course or the streak moved — re-bless"
    );
}

/// §5.6's material claim for this demo: the session's per-tick canonical hashes
/// match the checked-in baseline, on every architecture the leg runs.
#[test]
fn the_recorded_session_reproduces_its_checked_in_hash_sequence() {
    let sequence = session::hash_sequence(&entry(), &session::frames(&entry()).unwrap()).unwrap();
    let path = session::baseline_path();
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "no baseline at {} ({e}) — run `cargo xtask replay --bless`",
            path.display()
        )
    });
    let baseline = session::parse_baseline(&text).unwrap();
    if let Some(found) = session::divergence(&sequence, &baseline) {
        let actual = std::env::temp_dir().join("demo12-shooter.hashes.actual");
        let _ = std::fs::write(&actual, session::encode_baseline(&sequence));
        panic!("{found} — fresh sequence at {}", actual.display());
    }
}

/// §5.6a on this runner: the same script, driven twice, agrees tick for tick —
/// and the script itself is a pure function of the room and the course.
#[test]
fn one_recorded_session_run_twice_on_this_runner_agrees() {
    let first = session::frames(&entry()).unwrap();
    let second = session::frames(&entry()).unwrap();
    assert_eq!(first, second, "the bot is not deterministic");
    let a = session::hash_sequence(&entry(), &first).unwrap();
    let b = session::hash_sequence(&entry(), &second).unwrap();
    assert_eq!(a, b, "one machine, two answers");
}

/// The fight the retune gate swaps under (§6 M37 item 6): rounds leaving all
/// session long, a course that is live on every tick but the one it is taken
/// again on, and a quiet window between shots long enough for a swap to land in
/// and prove a whole world crossed it.
#[test]
fn the_endless_fight_keeps_a_round_in_progress() {
    const TICKS: usize = 4_000;
    let frames = session::endless(&entry(), TICKS).unwrap();
    assert_eq!(
        frames.len(),
        TICKS,
        "the stream is not the length it was asked for"
    );
    let progress = session::progress(&entry(), &frames).unwrap();

    let fired: Vec<usize> = (1..progress.len())
        .filter(|&i| progress[i].shots > progress[i - 1].shots)
        .collect();
    let gap = fired.windows(2).map(|w| w[1] - w[0]).min().unwrap();
    // One `over` tick per round ended: the bot takes the course again on the
    // tick it sees it out, so all but eight of these four thousand ticks are a
    // fight in progress.
    let over = progress.iter().filter(|p| p.over).count();
    assert_eq!(
        (fired.len(), over, gap),
        (113, 9, 35),
        "the fight changed shape — `xtask reload --retune` reads all three, and the shortest gap          is the window it spends proving the world crossed the swap"
    );
    assert!(
        progress.iter().all(|p| p.standing <= 3),
        "the course dealt more targets than it holds at once"
    );
    let last = *progress.last().unwrap();
    assert!(
        !last.over && last.hits > 0,
        "the stream does not end mid-fight"
    );
}
