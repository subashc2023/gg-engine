//! The template, driven the way the host drives it (§4.2.2).
//!
//! These exist for the same reason the template does: a guide nobody runs is a
//! guide that rots. Nothing here calls a system directly — every tick goes
//! through the systems table against a `World` registered from the *declared*
//! table, which is the path `gg-runtime` takes.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use demo_99_template::{EYE_AT, SUN, Spinner, TICKS_PER_TURN};
use gg_ecs::boundary::{
    self, AbiInfo, ComponentsTable, Eye, HostApiV1, InputFrame, Light, Renderable, SystemsTable,
    TickCtx, light,
};
use gg_ecs::{Query, World};

// The symbols `gg_game!` exported into this crate's rlib.
unsafe extern "C" {
    fn gg_game_abi() -> AbiInfo;
    fn gg_game_init(api: *const HostApiV1);
    fn gg_game_components() -> ComponentsTable;
    fn gg_game_systems() -> SystemsTable;
}

struct Game {
    world: World,
    table: SystemsTable,
    tick: u64,
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
        assert_eq!(declared, 4, "spinner, and the three protocol types");
        Game {
            world,
            table,
            tick: 0,
        }
    }

    fn step(&mut self) {
        let ctx = TickCtx {
            tick: self.tick,
            tick_hz: 60,
            reserved: 0,
            input: InputFrame::default(),
            previous: InputFrame::default(),
        };
        // SAFETY: the table is this binary's own, its entries live for the
        // process, and `ctx` outlives the call.
        unsafe { self.world.run_systems(&self.table, &ctx) }.expect("no system panicked");
        self.tick += 1;
    }

    fn all<T: gg_ecs::Component + Copy>(&self) -> Vec<T> {
        let query = Query::<&T>::new().unwrap();
        let mut out = Vec::new();
        self.world.each_ref(&query, |_, value: &T| out.push(*value));
        out
    }
}

#[test]
fn one_tick_leaves_a_cube_a_sun_and_an_eye() {
    let mut game = Game::load();
    game.step();
    assert_eq!(game.all::<Spinner>().len(), 1);
    assert_eq!(game.all::<Renderable>().len(), 1, "one box");
    let eye = *game.all::<Eye>().first().expect("an eye");
    assert_eq!(eye.position, EYE_AT);
    let sun = *game.all::<Light>().first().expect("a sun");
    assert_eq!(sun.kind, light::DIRECTIONAL);
    assert_eq!(sun.direction, SUN);
}

/// Re-entrant bootstrap: a reload runs it again against a populated world, and
/// the world it finds must not double (§6 M5).
#[test]
fn bootstrap_run_again_spawns_nothing_new() {
    let mut game = Game::load();
    for _ in 0..8 {
        game.step();
    }
    assert_eq!(game.all::<Spinner>().len(), 1);
    assert_eq!(game.all::<Light>().len(), 1);
    assert_eq!(game.all::<Eye>().len(), 1);
}

/// The claim the module header makes: the pose is a function of the tick count,
/// so a whole turn returns to exactly where it started — bit for bit, not
/// nearly.
#[test]
fn a_whole_turn_returns_to_the_starting_pose() {
    let mut game = Game::load();
    game.step();
    let start = *game.all::<Renderable>().first().expect("a box");
    for _ in 0..TICKS_PER_TURN {
        game.step();
    }
    let after = *game.all::<Renderable>().first().expect("a box");
    assert_eq!(
        after.rotation, start.rotation,
        "a full turn is {TICKS_PER_TURN} ticks and must land on the same quaternion"
    );
}

/// Halfway round is not where it started — otherwise the test above would pass
/// on a cube that never moved.
#[test]
fn the_cube_actually_turns() {
    let mut game = Game::load();
    game.step();
    let start = *game.all::<Renderable>().first().expect("a box");
    for _ in 0..TICKS_PER_TURN / 2 {
        game.step();
    }
    let half = *game.all::<Renderable>().first().expect("a box");
    assert_ne!(half.rotation, start.rotation);
}
