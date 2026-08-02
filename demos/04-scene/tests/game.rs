//! Demo 04, driven the way the host drives it (§4.2.2).
//!
//! Nothing here calls a system directly: every tick goes through the systems
//! table against a `World` registered from the *declared* table, so what these
//! exercise is the path `gg-runtime` runs. The renderer's half of the same
//! story is `gg-render`'s `tests/pack.rs`; between the two, the only piece with
//! no automated coverage is the window, which §1.5 puts out of reach.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use demo_04_scene::{HALL, MOVE_FORWARD, START_POSITION, TINT, TINTS, Visitor};
use gg_ecs::boundary::{
    self, AbiInfo, ActionId, ComponentsTable, Eye, HostApiV1, InputFrame, MAX_AXES, Model,
    SystemsTable, TickCtx, VerbsTable, asset_id,
};
use gg_ecs::{Query, World};

// The four symbols `gg_game!` exported into this crate's rlib, declared the way
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
        assert_eq!(declared, 5, "visitor, and the four protocol types");
        Game {
            world,
            table,
            tick: 0,
            previous: InputFrame::default(),
        }
    }

    fn step(&mut self, frame: InputFrame) {
        let ctx = TickCtx {
            tick: self.tick,
            tick_hz: 60,
            reserved: 0,
            input: frame,
            previous: self.previous,
        };
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
}

fn axis(id: gg_ecs::boundary::AxisId, value: i32) -> InputFrame {
    let mut axes = [0; MAX_AXES];
    axes[id.index()] = value;
    InputFrame { buttons: 0, axes }
}

#[test]
fn the_first_tick_places_the_visitor_and_names_the_hall() {
    let mut game = Game::load();
    assert!(game.all::<Model>().is_empty(), "nothing before a tick runs");
    game.step(InputFrame::default());

    assert_eq!(game.visitor().position, START_POSITION);
    let models = game.all::<Model>();
    assert_eq!(models.len(), 1, "one entity for the whole hall");
    // The id the *pack* stores, arrived at from the same name `ggc` wrote —
    // this is the one place the two sides of §4.6 are checked against each
    // other without a pack file in the room.
    assert_eq!(models[0].asset, asset_id(HALL));
    assert_eq!(models[0].tint, TINTS[0], "authored colours to begin with");

    let eye = game.all::<Eye>();
    assert_eq!(eye.len(), 1);
    assert_eq!(eye[0].position, START_POSITION);
}

#[test]
fn a_second_tick_declares_no_second_hall() {
    // `bootstrap` is idempotent by asking the world rather than by remembering,
    // which is the only form §4.2.2 allows — state may not outlive a tick.
    let mut game = Game::load();
    for _ in 0..8 {
        game.step(InputFrame::default());
    }
    assert_eq!(game.all::<Model>().len(), 1);
    assert_eq!(game.all::<Visitor>().len(), 1);
}

#[test]
fn the_tint_verb_cycles_and_reaches_the_model() {
    let mut game = Game::load();
    game.step(InputFrame::default());
    for expected in [1, 2, 3, 0] {
        game.press(TINT);
        assert_eq!(game.visitor().tint, expected);
        assert_eq!(
            game.all::<Model>()[0].tint,
            TINTS[expected as usize],
            "the sim is the source and the protocol component is output"
        );
    }
}

#[test]
fn walking_moves_the_eye_the_model_is_seen_from_and_not_the_hall() {
    let mut game = Game::load();
    game.step(InputFrame::default());
    let before = game.visitor().position;
    for _ in 0..10 {
        game.step(axis(MOVE_FORWARD, gg_ecs::boundary::AXIS_SCALE));
    }
    let after = game.visitor().position;
    assert!(after.z < before.z, "forward is -Z: {before:?} -> {after:?}");
    assert_eq!(game.all::<Eye>()[0].position, after);
    // The hall stays where the pack put it. A scene that walked with the camera
    // would be a camera-relative position leaking into sim state (§1.4).
    assert_eq!(game.all::<Model>()[0].position, gg_math::sim::DVec3::ZERO);
}

#[test]
fn the_declared_verbs_are_the_id_space_the_bindings_resolve_against() {
    // SAFETY: this binary's own table, alive for the process.
    let verbs = unsafe { boundary::read_verbs(&gg_game_verbs()) };
    assert_eq!(verbs.actions, ["tint"]);
    assert_eq!(
        verbs.axes,
        ["move_right", "move_up", "move_forward", "aim_x", "aim_y"],
        "order is the id space (§4.7), and `input.toml` is read against it"
    );
}
