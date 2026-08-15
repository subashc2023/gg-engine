//! Demo 05, driven the way the host drives it (§4.2.2) — systems table *and*
//! hierarchy propagation, in that order, because that is the tick.
//!
//! Nothing here calls a system directly. The propagation step is `gg-scene`'s
//! own, run between the systems and where the canonical hash would be taken, so
//! what these exercise is the path `gg-runtime` runs. A test that skipped it
//! would be asserting about half a tick: every child's world position is
//! written there and nowhere else.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use demo_05_many::{
    FREEZE, GRID, HUBS, Hub, MESHES, MOVE_FORWARD, Observer, PER_HUB, START_POSITION,
    child_placement, hub_position,
};
use gg_ecs::boundary::{
    self, AbiInfo, ActionId, ComponentsTable, Eye, HostApiV1, InputFrame, MAX_AXES, Model, Node,
    SystemsTable, TickCtx, VerbsTable, asset_id,
};
use gg_ecs::{Entity, Query, World};
use gg_math::sim;
use gg_scene::{Hierarchy, Propagated};

// The four symbols `gg_game!` exported into this crate's rlib, declared the way
// a host reaches a dylib's tables.
unsafe extern "C" {
    fn gg_game_abi() -> AbiInfo;
    fn gg_game_init(api: *const HostApiV1);
    fn gg_game_components() -> ComponentsTable;
    fn gg_game_verbs() -> VerbsTable;
    fn gg_game_systems() -> SystemsTable;
}

/// A loaded game, the world it drives, and the host's hierarchy — the shell,
/// minus the shell.
struct Game {
    world: World,
    table: SystemsTable,
    hierarchy: Hierarchy,
    propagated: Propagated,
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
        assert_eq!(declared, 6, "observer, hub, and the four protocol types");
        Game {
            world,
            table,
            hierarchy: Hierarchy::new(),
            propagated: Propagated::default(),
            tick: 0,
            previous: InputFrame::default(),
        }
    }

    fn step(&mut self, frame: InputFrame) {
        let ctx = TickCtx::detached(self.tick, 60, frame, self.previous);
        // SAFETY: the table is this binary's own, its entries live for the
        // process, and `ctx` outlives the call.
        unsafe { self.world.run_systems(&self.table, &ctx) }.expect("no system panicked");
        // §4.7, and where the shell runs it: after the game's systems, before
        // the hash. A child's `Model` is written here and by nothing else.
        self.propagated = self
            .hierarchy
            .propagate(&mut self.world)
            .expect("the hierarchy resolves");
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

    fn all<T: gg_ecs::Component + Copy>(&self) -> Vec<(Entity, T)> {
        let query = Query::<&T>::new().unwrap();
        let mut out = Vec::new();
        self.world
            .each_ref(&query, |entity, value: &T| out.push((entity, *value)));
        out
    }

    fn observer(&self) -> Observer {
        self.all::<Observer>()
            .first()
            .expect("the world always has an observer after bootstrap")
            .1
    }

    /// Every child of `parent`, in world iteration order.
    fn children_of(&self, parent: Entity) -> Vec<(Entity, Node)> {
        self.all::<Node>()
            .into_iter()
            .filter(|(_, node)| node.parent == parent)
            .collect()
    }

    fn model_of(&self, entity: Entity) -> Model {
        *self.world.get::<Model>(entity).expect("entity draws")
    }

    /// The hubs, in world iteration order — the order `bootstrap` spawned them.
    fn hubs(&self) -> Vec<(Entity, Hub)> {
        self.all::<Hub>()
    }
}

fn axis(id: gg_ecs::boundary::AxisId, value: i32) -> InputFrame {
    let mut axes = [0; MAX_AXES];
    axes[id.index()] = value;
    InputFrame { buttons: 0, axes }
}

#[test]
fn the_first_tick_raises_a_hundred_hubs_and_ten_thousand_children() {
    let mut game = Game::load();
    assert!(game.all::<Model>().is_empty(), "nothing before a tick runs");
    game.step(InputFrame::default());

    assert_eq!(game.observer().position, START_POSITION);
    assert_eq!(game.hubs().len(), HUBS);
    assert_eq!(game.all::<Node>().len(), HUBS * PER_HUB);
    // §6 M10's exit row asks for ten thousand objects. The hubs are the change.
    assert_eq!(game.all::<Model>().len(), HUBS * PER_HUB + HUBS);
    const {
        assert!(
            HUBS * PER_HUB >= 10_000,
            "the demo has to be the scale it claims"
        )
    };
}

/// The verb lists are an id space a replay records against (§4.7), so their
/// *order* is the contract `input.toml` resolves names through — not the names
/// themselves, which is why this asserts positions rather than membership.
#[test]
fn the_verb_order_is_the_id_space_a_replay_records() {
    let _loaded = Game::load();
    // SAFETY: this binary's own table, static for the process.
    let verbs = unsafe { boundary::read_verbs(&gg_game_verbs()) };
    assert_eq!(verbs.actions[FREEZE.index()], "freeze");
    assert_eq!(verbs.actions.len(), 1);
    assert_eq!(verbs.axes[MOVE_FORWARD.index()], "move_forward");
    assert_eq!(verbs.axes.len(), 5);
}

#[test]
fn a_second_tick_raises_nothing_further() {
    let mut game = Game::load();
    game.step(InputFrame::default());
    let after_one = game.all::<Model>().len();
    for _ in 0..4 {
        game.step(InputFrame::default());
    }
    assert_eq!(
        game.all::<Model>().len(),
        after_one,
        "bootstrap is idempotent"
    );
}

#[test]
fn the_field_names_four_meshes_and_the_hubs_name_the_first() {
    let mut game = Game::load();
    game.step(InputFrame::default());

    let ids: Vec<u64> = MESHES.iter().copied().map(asset_id).collect();
    let (hub, _) = game.hubs()[0];
    assert_eq!(game.model_of(hub).asset, ids[0]);

    // Every mesh is used, so the renderer's sort key has more than one material
    // to order and its batcher more than one run to build (§6 M10).
    let children = game.children_of(hub);
    assert_eq!(children.len(), PER_HUB);
    for id in &ids {
        assert!(
            children.iter().any(|(e, _)| game.model_of(*e).asset == *id),
            "no child draws {id:#x}"
        );
    }
}

#[test]
fn a_childs_world_position_is_its_hubs_pose_composed_with_its_own_offset() {
    let mut game = Game::load();
    game.step(InputFrame::default());

    // Hub 1, the *still* one: `spin` runs before propagation in the same tick,
    // so hub 0 has already turned by 0.013 rad when this reads it. On a hub that
    // never turns the composition reduces to an addition, which is what makes
    // this readable as an assertion about the host's arithmetic rather than a
    // restatement of it. The rotating case is the next test's subject.
    let (hub, _) = game.hubs()[1];
    let base = hub_position(1);
    assert_eq!(game.model_of(hub).position, base);

    let (child, node) = game.children_of(hub)[0];
    let (offset, _) = child_placement(0);
    assert_eq!(node.position, offset, "the game wrote the *local* offset");
    let world = game.model_of(child).position;
    assert!((world - (base + offset)).length() < 1e-12, "{world:?}");

    // Grid corners, so a hub's own placement is not silently the origin.
    assert_eq!(hub_position(0).x, hub_position(GRID).x);
    assert_ne!(hub_position(0).z, hub_position(GRID).z);
}

#[test]
fn a_spinning_hub_swings_its_ring_and_a_still_one_does_not() {
    let mut game = Game::load();
    game.step(InputFrame::default());

    let hubs = game.hubs();
    let (spinning, _) = hubs[0];
    let (still, _) = hubs[1];
    assert_ne!(game.hubs()[0].1.spin, 0.0);
    assert_eq!(game.hubs()[1].1.spin, 0.0, "every other hub is still");

    let moving_child = game.children_of(spinning)[0].0;
    let still_child = game.children_of(still)[0].0;
    let before = (game.model_of(moving_child), game.model_of(still_child));

    for _ in 0..30 {
        game.step(InputFrame::default());
    }
    let after = (game.model_of(moving_child), game.model_of(still_child));
    assert_ne!(before.0.position, after.0.position, "the ring swung");
    assert_eq!(before.1.position, after.1.position, "the still one held");

    // Swung, not slid: a rotation about the hub keeps the child's distance.
    let hub_at = game.model_of(spinning).position;
    let radius = |p: sim::DVec3| (p - hub_at).length();
    assert!((radius(before.0.position) - radius(after.0.position)).abs() < 1e-9);
}

#[test]
fn freezing_the_field_costs_a_scan_and_no_composes() {
    let mut game = Game::load();
    game.step(InputFrame::default());
    // Half the hubs spin, so half the children recompose every tick.
    game.step(InputFrame::default());
    let moving = game.propagated;
    assert_eq!(moving.nodes, HUBS * PER_HUB, "every node is still scanned");
    assert_eq!(
        moving.composed,
        HUBS / 2 * PER_HUB,
        "and exactly the spinning hubs' children recomposed"
    );

    game.press(FREEZE);
    game.step(InputFrame::default());
    let frozen = game.propagated;
    assert_eq!(frozen.nodes, HUBS * PER_HUB, "still scanned");
    assert_eq!(frozen.composed, 0, "and nothing composed");
    assert_eq!(frozen.written, 0);
}

#[test]
fn walking_moves_the_eye_the_protocol_carries() {
    let mut game = Game::load();
    game.step(InputFrame::default());
    let before = game.observer().position;
    for _ in 0..10 {
        game.step(axis(MOVE_FORWARD, gg_ecs::boundary::AXIS_SCALE));
    }
    let after = game.observer().position;
    assert!(
        after.z < before.z,
        "forward is -Z at yaw 0: {before:?} {after:?}"
    );

    let eyes = game.all::<Eye>();
    assert_eq!(eyes.len(), 1);
    assert_eq!(
        eyes[0].1.position, after,
        "the eye is this tick's, not last's"
    );
}
