//! The Ugly Game, driven the way the host drives it (§4.2.2).
//!
//! Nothing here calls a system directly. Every tick goes through the systems
//! table, against a `World` whose components were registered from the *declared*
//! table rather than from the Rust types this file also happens to have — so
//! what these tests exercise is the path `gg-runtime` runs, and a boundary that
//! stopped working would fail here rather than only in a windowed session.
//!
//! The game half of this crate is throwaway (§6 M5); these tests are what say
//! *which* throwaway behaviour is currently true, so an agent rewriting a system
//! body finds out immediately whether it rewrote the game or broke it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use demo_03_reload::{
    AIM_CONE, Cube, FIRE, MAX_CUBES, Player, RESTART, SPAWN, START_CUBES, START_POSITION, Shot,
    TRACER_TICKS, basis,
};
use gg_ecs::boundary::{
    self, AXIS_SCALE, AbiInfo, ActionId, ComponentsTable, Eye, HostApiV1, InputFrame, MAX_AXES,
    Renderable, SystemsTable, TickCtx, VerbsTable,
};
use gg_ecs::{Entity, Query, World};

// The four symbols `gg_game!` exported into this crate's rlib, declared the way
// a host reaches a dylib's tables. Declaring them here is what keeps the tests
// on the host's path instead of beside it.
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
    /// The host's load sequence, then a world carrying what the game declared.
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
        assert_eq!(
            declared, 5,
            "player, cube, shot, and the two render protocol types"
        );
        Game {
            world,
            table,
            tick: 0,
            previous: InputFrame::default(),
        }
    }

    /// One tick with `frame` held. The previous frame is carried the way the
    /// shell carries it, so `just_pressed` means what it means in a real run.
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

    fn cubes(&self) -> Vec<Cube> {
        self.all::<Cube>()
    }

    fn shots(&self) -> Vec<Shot> {
        self.all::<Shot>()
    }

    fn all<T: gg_ecs::Component + Copy>(&self) -> Vec<T> {
        let query = Query::<&T>::new().unwrap();
        let mut out = Vec::new();
        self.world.each_ref(&query, |_, value: &T| out.push(*value));
        out
    }

    fn player(&self) -> (Entity, Player) {
        let query = Query::<&Player>::new().unwrap();
        let mut found = None;
        self.world.each_ref(&query, |entity, player: &Player| {
            found = Some((entity, *player));
        });
        found.expect("the world always has a player after bootstrap")
    }
}

fn axis(id: gg_ecs::boundary::AxisId, value: i32) -> InputFrame {
    let mut axes = [0; MAX_AXES];
    axes[id.index()] = value;
    InputFrame { buttons: 0, axes }
}

#[test]
fn the_first_tick_builds_the_opening_world() {
    let mut game = Game::load();
    assert!(game.cubes().is_empty(), "nothing exists before a tick runs");
    game.step(InputFrame::default());

    let (_, player) = game.player();
    assert_eq!(player.position, START_POSITION);
    let cubes = game.cubes();
    assert_eq!(cubes.len() as u64, START_CUBES);
    // Orders are the ring's index, ascending and unique — what "the newest cube"
    // is a fact *about state* rather than about a list kept elsewhere.
    let mut orders: Vec<u64> = cubes.iter().map(|c| c.order).collect();
    orders.sort_unstable();
    assert_eq!(orders, (0..START_CUBES).collect::<Vec<_>>());
}

#[test]
fn bootstrap_asks_the_world_rather_than_remembering() {
    // §4.2.2 forbids a system state that outlives a tick, so "have I run?" has
    // to be answered from the world. Ten ticks, one opening world.
    let mut game = Game::load();
    for _ in 0..10 {
        game.step(InputFrame::default());
    }
    assert_eq!(game.cubes().len() as u64, START_CUBES);
}

#[test]
fn firing_down_the_ring_kills_exactly_the_cube_under_the_crosshair() {
    let mut game = Game::load();
    game.step(InputFrame::default());
    let (_, player) = game.player();
    let (forward, _) = basis(&player);

    // Which cube *should* die, computed here from the same rule the system
    // uses — so a changed cone or a changed opening ring moves both together.
    let expected = game
        .cubes()
        .into_iter()
        .filter(|cube| {
            (cube.position - player.position)
                .try_normalize()
                .is_some_and(|d| d.dot(forward) > AIM_CONE)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        expected.len(),
        1,
        "the opening ring aims one cube at the eye"
    );

    game.press(FIRE);
    let left = game.cubes();
    assert_eq!(left.len() as u64, START_CUBES - 1);
    assert!(
        !left.iter().any(|cube| cube.order == expected[0].order),
        "the cube in the cone is the one that died"
    );
}

#[test]
fn a_miss_still_draws_a_tracer_and_kills_nothing() {
    let mut game = Game::load();
    game.step(InputFrame::default());
    // Turn around. Straight into the world's storage, because the point is to
    // test shooting rather than to test turning — and it is the same column the
    // aim system writes.
    let (entity, _) = game.player();
    game.world.get_mut::<Player>(entity).unwrap().yaw = core::f32::consts::PI;

    game.press(FIRE);
    assert_eq!(
        game.cubes().len() as u64,
        START_CUBES,
        "nothing was in front"
    );
    let shots = game.shots();
    assert_eq!(shots.len(), 1, "a miss is still a shot");
    // Two ticks have passed since the press (down, then up), so the tracer has
    // aged by exactly that much.
    assert_eq!(shots[0].ticks_left, TRACER_TICKS - 2);
}

#[test]
fn tracers_expire_on_their_own() {
    let mut game = Game::load();
    game.press(FIRE);
    assert_eq!(game.shots().len(), 1);
    for _ in 0..TRACER_TICKS {
        game.step(InputFrame::default());
    }
    assert!(game.shots().is_empty(), "a tracer is not a leak");
}

#[test]
fn holding_spawn_is_one_edge_and_the_ceiling_holds() {
    let mut game = Game::load();
    let held = InputFrame {
        buttons: 1 << SPAWN.index(),
        axes: [0; MAX_AXES],
    };
    for _ in 0..50 {
        game.step(held);
    }
    assert_eq!(
        game.cubes().len() as u64,
        START_CUBES + 1,
        "held is one edge, not fifty"
    );

    for _ in 0..MAX_CUBES * 2 {
        game.press(SPAWN);
    }
    assert_eq!(game.cubes().len(), MAX_CUBES, "the ceiling is a ceiling");
}

#[test]
fn restart_rebuilds_the_opening_world_inside_one_tick() {
    // The M0X finding (a) affordance: state persistence makes edits to
    // initial-state code invisible, and this is the verb that makes them visible
    // again. `restart` runs before `bootstrap`, so the wipe and the rebuild are
    // the same tick rather than two.
    let mut game = Game::load();
    game.step(InputFrame::default());
    game.press(SPAWN);
    game.press(FIRE);
    game.step(axis(demo_03_reload::MOVE_FORWARD, AXIS_SCALE));
    assert_ne!(game.player().1.position, START_POSITION);

    let before = game.tick;
    game.step(InputFrame {
        buttons: 1 << RESTART.index(),
        axes: [0; MAX_AXES],
    });
    assert_eq!(game.tick, before + 1, "one tick, not two");

    assert_eq!(game.player().1.position, START_POSITION);
    assert_eq!(game.cubes().len() as u64, START_CUBES);
    assert!(game.shots().is_empty(), "the wipe takes the tracers too");
}

#[test]
fn walking_goes_where_the_axes_point_and_aiming_stops_at_the_pole() {
    let mut game = Game::load();
    game.step(InputFrame::default());
    game.step(axis(demo_03_reload::MOVE_UP, AXIS_SCALE));
    let (_, player) = game.player();
    assert_eq!(
        player.position.y,
        START_POSITION.y + demo_03_reload::MOVE_PER_TICK
    );
    assert_eq!(
        player.position.x, START_POSITION.x,
        "only where the axis points"
    );

    for _ in 0..500 {
        game.step(axis(demo_03_reload::AIM_Y, -AXIS_SCALE * 4));
    }
    let (_, player) = game.player();
    assert_eq!(player.pitch, demo_03_reload::PITCH_LIMIT);
    // ...and the basis is still a basis at the limit.
    let (forward, right) = basis(&player);
    assert!(forward.cross(right).length() > 0.5);
}

#[test]
fn the_world_describes_itself_to_a_host_that_knows_no_cubes() {
    // What the shell actually reads (§4.5 v0): it has no `Cube` type and no
    // `Player` type, so everything it draws has to arrive as `Renderable` and
    // `Eye`. This test is that host, and it never mentions a game type.
    let mut game = Game::load();
    game.step(InputFrame::default());

    let eyes = game.all::<Eye>();
    assert_eq!(eyes.len(), 1, "one eye");
    assert_eq!(eyes[0].position, START_POSITION);

    // One box per cube, plus the floor — which carries no gameplay component at
    // all, and is therefore the case a sweep by game type would miss — plus the
    // crosshair, which rides on the player.
    let boxes = game.all::<Renderable>();
    assert_eq!(boxes.len() as u64, START_CUBES + 2);
    let floor = boxes
        .iter()
        .find(|r| r.half_extent.x > 10.0)
        .expect("the floor is a box like everything else");
    assert!(floor.position.y < 0.0, "its top sits at y = 0");
    assert!(
        boxes.iter().filter(|r| r.color != floor.color).count() >= 2,
        "the ring is not all one colour"
    );
}

#[test]
fn a_tracer_is_a_box_turned_onto_the_shot() {
    let mut game = Game::load();
    game.step(InputFrame::default());
    game.press(FIRE);
    let shot = game.shots()[0];

    let beam = game
        .all::<Renderable>()
        .into_iter()
        .find(|r| r.half_extent.x == demo_03_reload::TRACER_RADIUS)
        .expect("the shot is drawn");
    assert_eq!(beam.position, (shot.from + shot.to) * 0.5, "centred");
    // +Z turned onto the shot direction, checked by rotating it back.
    let along = (shot.to - shot.from).try_normalize().unwrap();
    let turned = beam
        .rotation
        .rotate(gg_math::sim::DVec3::new(0.0, 0.0, 1.0));
    assert!((turned - along).length() < 1e-9, "{turned:?} vs {along:?}");
}

#[test]
fn restarting_does_not_leave_a_floor_behind() {
    // The floor is the one entity `restart`'s gameplay-type sweeps cannot see,
    // so it is the one that would pile up once per press.
    let mut game = Game::load();
    game.step(InputFrame::default());
    let opening = game.all::<Renderable>().len();
    for _ in 0..3 {
        game.press(RESTART);
    }
    assert_eq!(game.all::<Renderable>().len(), opening);
}

#[test]
fn the_declared_verbs_are_the_ids_the_systems_index_by() {
    // The macro's lists are the id space (§4.7) and the `ActionId` constants are
    // indices into them. Nothing in the language ties the two together, so it is
    // tied here instead.
    Game::load();
    // SAFETY: the table is this binary's own and lives for the process.
    let verbs = unsafe { boundary::read_verbs(&gg_game_verbs()) };
    assert_eq!(verbs.actions[FIRE.index()], "fire");
    assert_eq!(verbs.actions[SPAWN.index()], "spawn");
    assert_eq!(verbs.actions[RESTART.index()], "restart");
    assert_eq!(verbs.axes[demo_03_reload::MOVE_RIGHT.index()], "move_right");
    assert_eq!(verbs.axes[demo_03_reload::MOVE_UP.index()], "move_up");
    assert_eq!(
        verbs.axes[demo_03_reload::MOVE_FORWARD.index()],
        "move_forward"
    );
    assert_eq!(verbs.axes[demo_03_reload::AIM_X.index()], "aim_x");
    assert_eq!(verbs.axes[demo_03_reload::AIM_Y.index()], "aim_y");
    assert_eq!(verbs.actions.len(), 3);
    assert_eq!(verbs.axes.len(), 5);
}
