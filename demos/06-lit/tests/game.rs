//! Demo 06, driven the way the host drives it (§4.2.2).
//!
//! Nothing here calls a system directly: every tick goes through the systems
//! table against a `World` registered from the *declared* table, so what these
//! exercise is the path `gg-runtime` runs. The renderer's half of the same
//! story — that a `Light` becomes shading and a shadow — is `gg-render`'s own
//! offscreen tests and the golden suite's `atrium` scene.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use demo_06_lit::{
    ATRIUM, HOLD, LAMP_AT, LAMP_INTENSITY, LAMPS, MOVE_FORWARD, START_POSITION, SUN_PERIOD,
    Visitor, sun_direction,
};
use gg_ecs::boundary::{
    self, AbiInfo, ActionId, ComponentsTable, Eye, HostApiV1, InputFrame, Light, MAX_AXES, Model,
    SystemsTable, TickCtx, VerbsTable, asset_id, light,
};
use gg_ecs::{Query, World};

// The five symbols `gg_game!` exported into this crate's rlib, declared the way
// a host reaches a dylib's tables.
unsafe extern "C" {
    fn gg_game_abi() -> AbiInfo;
    fn gg_game_init(api: *const HostApiV1);
    fn gg_game_components() -> ComponentsTable;
    fn gg_game_verbs() -> VerbsTable;
    fn gg_game_systems() -> SystemsTable;
}

/// A loaded game and the world it drives — the shell, minus the shell.
struct Game {
    world: World,
    table: SystemsTable,
    tick: u64,
    previous: InputFrame,
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
        let (table, declared) = unsafe {
            let declared = world.adopt(&gg_game_components()).unwrap();
            (gg_game_systems(), declared)
        };
        assert_eq!(declared, 4, "visitor, and the three protocol types");
        Game {
            world,
            table,
            tick: 0,
            previous: InputFrame::default(),
        }
    }

    fn step(&mut self, frame: InputFrame) {
        let ctx = TickCtx::detached(self.tick, 60, frame, self.previous);
        // SAFETY: the table is this binary's own, its entries live for the
        // process, and `ctx` outlives the call.
        unsafe { self.world.run_systems(&self.table, &ctx) }.expect("no system panicked");
        self.previous = frame;
        self.tick += 1;
    }

    /// A press: one tick down, one tick up, so the edge is a real edge.
    fn press(&mut self, action: ActionId) {
        self.step(InputFrame {
            buttons: 1 << action.index(),
            axes: [0; MAX_AXES],
        });
        self.step(InputFrame::default());
    }

    fn all<T: gg_ecs::Component + Copy>(&self) -> Vec<T> {
        let query = Query::<&T>::new().unwrap();
        let mut out = Vec::new();
        self.world.each_ref(&query, |_, value: &T| out.push(*value));
        out
    }

    fn visitor(&self) -> Visitor {
        *self
            .all::<Visitor>()
            .first()
            .expect("the world always has a visitor after bootstrap")
    }

    fn sun(&self) -> Light {
        *self
            .all::<Light>()
            .iter()
            .find(|l| l.kind == light::DIRECTIONAL)
            .expect("the atrium has a sun")
    }

    fn lamps(&self) -> Vec<Light> {
        self.all::<Light>()
            .into_iter()
            .filter(|l| l.kind == light::POINT)
            .collect()
    }
}

fn axis(id: gg_ecs::boundary::AxisId, value: i32) -> InputFrame {
    let mut axes = [0; MAX_AXES];
    axes[id.index()] = value;
    InputFrame { buttons: 0, axes }
}

#[test]
fn the_first_tick_places_the_visitor_the_atrium_and_every_light() {
    let mut game = Game::load();
    assert!(game.all::<Model>().is_empty(), "nothing before a tick runs");
    game.step(InputFrame::default());

    assert_eq!(game.visitor().position, START_POSITION);
    let models = game.all::<Model>();
    assert_eq!(models.len(), 1, "one entity for the whole room");
    // The id the *pack* stores, arrived at from the same name `ggc` wrote.
    assert_eq!(models[0].asset, asset_id(ATRIUM));

    // One sun and four lamps. The sun is spawned first, which is what makes it
    // the one the renderer's single cascade casts (§6 M11).
    let lights = game.all::<Light>();
    assert_eq!(lights.len(), 1 + LAMP_AT.len());
    assert_eq!(
        lights[0].kind,
        light::DIRECTIONAL,
        "the sun is spawned first"
    );
    assert_eq!(game.lamps().len(), LAMP_AT.len());

    let eye = game.all::<Eye>();
    assert_eq!(eye.len(), 1);
    assert_eq!(eye[0].position, START_POSITION);
}

#[test]
fn a_second_tick_declares_no_second_atrium() {
    // `bootstrap` is idempotent by asking the world rather than by remembering,
    // which is the only form §4.2.2 allows — state may not outlive a tick.
    let mut game = Game::load();
    for _ in 0..8 {
        game.step(InputFrame::default());
    }
    assert_eq!(game.all::<Model>().len(), 1);
    assert_eq!(game.all::<Visitor>().len(), 1);
    assert_eq!(game.all::<Light>().len(), 1 + LAMP_AT.len());
}

#[test]
fn the_sun_sweeps_downward_and_the_hold_verb_stops_it() {
    let mut game = Game::load();
    game.step(InputFrame::default());
    let first = game.sun().direction;

    for _ in 0..30 {
        game.step(InputFrame::default());
    }
    let moved = game.sun().direction;
    assert_ne!(first, moved, "the sun sweeps");
    // Always downward: a sun that swept under the floor would spend half its
    // period lighting the underside of the room.
    assert!(moved.y < 0.0, "{moved:?}");

    game.press(HOLD);
    let held = game.sun().direction;
    for _ in 0..30 {
        game.step(InputFrame::default());
    }
    assert_eq!(game.sun().direction, held, "held");
    game.press(HOLD);
    for _ in 0..30 {
        game.step(InputFrame::default());
    }
    assert_ne!(game.sun().direction, held, "and released");
}

#[test]
fn the_sweep_is_a_function_of_the_phase_and_the_phase_wraps() {
    // The determinism argument in one test: the sun is the tick count and
    // nothing else, so two machines at the same tick agree (§1.3). Wrapping is
    // checked because a phase that grew forever would make the hash of a long
    // session depend on uptime.
    assert_eq!(sun_direction(0), sun_direction(SUN_PERIOD));
    assert_ne!(sun_direction(0), sun_direction(SUN_PERIOD / 4));
    for phase in [0, 1, SUN_PERIOD / 3, SUN_PERIOD - 1] {
        let direction = sun_direction(phase);
        assert!(direction.y < 0.0, "phase {phase} points up");
        let length =
            (direction.x * direction.x + direction.y * direction.y + direction.z * direction.z)
                .sqrt();
        assert!((length - 1.0).abs() < 1e-6, "phase {phase} is not unit");
    }

    let mut game = Game::load();
    for _ in 0..=SUN_PERIOD + 4 {
        game.step(InputFrame::default());
    }
    assert!(game.visitor().phase < SUN_PERIOD, "the phase wrapped");
}

#[test]
fn the_lamps_switch_off_by_going_dark_rather_than_by_leaving() {
    // Despawning would change the world's archetype shape twice a keypress, and
    // the renderer already treats zero radiance as nothing.
    let mut game = Game::load();
    game.step(InputFrame::default());
    assert!(game.lamps().iter().all(|l| l.intensity == LAMP_INTENSITY));

    game.press(LAMPS);
    assert_eq!(game.lamps().len(), LAMP_AT.len(), "still there");
    assert!(game.lamps().iter().all(|l| l.intensity == 0.0), "and dark");
    // Range and position survive the switch, so turning them back on does not
    // re-decide where they are.
    assert!(game.lamps().iter().all(|l| l.range > 0.0));

    game.press(LAMPS);
    assert!(game.lamps().iter().all(|l| l.intensity == LAMP_INTENSITY));
}

#[test]
fn walking_moves_the_eye_and_not_the_room_or_its_lights() {
    let mut game = Game::load();
    game.step(InputFrame::default());
    let before = game.visitor().position;
    let lamps = game.lamps();
    for _ in 0..10 {
        game.step(axis(MOVE_FORWARD, gg_ecs::boundary::AXIS_SCALE));
    }
    let after = game.visitor().position;
    assert!(after.z < before.z, "forward is -Z: {before:?} -> {after:?}");
    assert_eq!(game.all::<Eye>()[0].position, after);
    // The room and its lamps stay where they were put. A light that walked with
    // the camera would be a camera-relative position leaking into sim state
    // (§1.4) — the exact failure the membrane exists to make impossible.
    assert_eq!(game.all::<Model>()[0].position, gg_math::sim::DVec3::ZERO);
    assert_eq!(
        game.lamps().iter().map(|l| l.position).collect::<Vec<_>>(),
        lamps.iter().map(|l| l.position).collect::<Vec<_>>()
    );
}

#[test]
fn the_declared_verbs_are_the_id_space_the_bindings_resolve_against() {
    // SAFETY: this binary's own table, alive for the process.
    let verbs = unsafe { boundary::read_verbs(&gg_game_verbs()) };
    assert_eq!(verbs.actions, ["hold", "lamps"]);
    assert_eq!(
        verbs.axes,
        ["move_right", "move_up", "move_forward", "aim_x", "aim_y"],
        "order is the id space (§4.7), and `input.toml` is read against it"
    );
}
