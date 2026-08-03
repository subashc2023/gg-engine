//! The §4.2.2 seam, driven end to end in one process.
//!
//! This file is shaped like a game crate — components, systems, one `gg_game!`
//! — and then plays the host: it resolves the four symbols, hands over the real
//! [`HostApiV1`], and calls the systems table against a real `World`. Every line
//! of the guest wrappers, the generated shims and the host vtable runs.
//!
//! What it deliberately does **not** prove is that a *separately compiled*
//! artifact agrees, because nothing in the workspace builds as a dylib until
//! demo 03. Same-process means the version and fingerprint trivially match; the
//! checks that they *are* compared live in `gg-core`'s reload tests.
//!
//! One file, not several: `gg_game!` emits `#[unsafe(no_mangle)]` symbols, so
//! two invocations in one test binary would be a duplicate-symbol link error —
//! and each integration test file is its own binary.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::boundary::{
    self, AbiInfo, ComponentsTable, GameWorld, HostApiV1, SystemsTable, TickCtx,
};
use gg_ecs::{Component, Entity, World};

// ---- the game crate half ------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "boundary.position")]
#[repr(C)]
struct Position {
    x: i32,
    y: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "boundary.velocity")]
#[repr(C)]
struct Velocity {
    dx: i32,
    dy: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "boundary.tag")]
#[repr(C)]
struct Tag {
    seen_at_tick: u64,
}

/// Marker components whose only job is to put otherwise-identical entities in
/// different archetypes, so a query has more matches than one page holds.
macro_rules! markers {
    ($($name:ident = $id:literal),+ $(,)?) => {
        $(
            // The id is spelled out rather than built with `concat!`: the derive
            // takes a string literal, which is the same insistence §4.2 makes
            // everywhere — a declared id is written down, never computed.
            #[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable, Component)]
            #[component(id = $id)]
            #[repr(C)]
            struct $name {
                _pad: u32,
            }
        )+

        const MARKERS: usize = [$($id),+].len();

        fn insert_marker(world: &mut World, entity: Entity, at: usize) {
            let mut next = 0;
            $(
                if at == next {
                    world.insert(entity, $name::default()).unwrap();
                    return;
                }
                next += 1;
            )+
            let _ = next;
            unreachable!("marker {at} past MARKERS");
        }
    };
}

markers!(
    M0 = "boundary.marker.0",
    M1 = "boundary.marker.1",
    M2 = "boundary.marker.2",
    M3 = "boundary.marker.3",
    M4 = "boundary.marker.4",
    M5 = "boundary.marker.5",
    M6 = "boundary.marker.6",
    M7 = "boundary.marker.7",
    M8 = "boundary.marker.8",
);

/// Integer motion on purpose: this test is about the seam, and a float would
/// make a failure ambiguous between "the column is wrong" and "the arithmetic
/// is".
fn movement(world: &mut GameWorld) {
    world
        .each::<(&mut Position, &Velocity)>(|_entity, (position, velocity)| {
            position.x += velocity.dx;
            position.y += velocity.dy;
        })
        .expect("a two-component query is not an aliasing error");
}

/// Writes the tick it ran on, so ordering within a table is observable.
fn stamp(world: &mut GameWorld) {
    let tick = world.tick();
    world
        .each::<&mut Tag>(|_entity, tag| tag.seen_at_tick = tick)
        .expect("a single-column query");
}

/// Structural change from inside a system, after iteration rather than during.
fn spawner(world: &mut GameWorld) {
    if world.tick() != 3 {
        return;
    }
    let entity = world.spawn();
    world
        .insert(entity, Position { x: 100, y: 100 })
        .expect("the host registered Position");
}

/// The deliberately planted failure of §6 M5: a panic inside a system.
fn exploder(world: &mut GameWorld) {
    assert!(world.tick() != 9, "the ugly game broke on purpose");
}

/// Game output goes through the host's stream, not the game's own (§4.2.2).
fn talker(world: &mut GameWorld) {
    world.log(
        gg_ecs::boundary::log_level::INFO,
        "the ugly game says hello",
    );
}

/// Names one component mutably twice — refused host-side, before any pointer
/// exists.
fn aliaser(world: &mut GameWorld) {
    let outcome = world.each::<(&mut Position, &mut Position)>(|_, _| {});
    assert!(
        matches!(
            outcome,
            Err(boundary::BoundaryError {
                status: boundary::AbiStatus::Alias,
                ..
            })
        ),
        "aliasing must come back as a status, never as two live borrows"
    );
}

gg_ecs::gg_game! {
    components: [Position, Velocity, Tag],
    actions: ["fire", "restart"],
    axes: ["aim_x"],
    systems: [movement, stamp, spawner, talker, exploder, aliaser],
}

/// Where `set_logger` sends this binary's game output. A `Mutex<Vec<_>>` because
/// the hook is a plain `fn` — the host installs one route, not a closure over
/// per-test state.
static LOGGED: std::sync::Mutex<Vec<(u32, String)>> = std::sync::Mutex::new(Vec::new());

fn record(level: u32, message: &str) {
    LOGGED.lock().unwrap().push((level, message.to_owned()));
}

/// What the installed [`boundary::SystemZone`] saw, in call order.
static ZONED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The thread whose zones swallow the system instead of running it — the failure
/// `run_systems` has to report. Keyed to a thread rather than a bare flag so the
/// one test that arms it cannot reach a test running beside it.
static SWALLOW: std::sync::Mutex<Option<std::thread::ThreadId>> = std::sync::Mutex::new(None);

fn zone(name: &str, body: &mut dyn FnMut()) {
    ZONED.lock().unwrap().push(format!("enter {name}"));
    if *SWALLOW.lock().unwrap() == Some(std::thread::current().id()) {
        return;
    }
    body();
    ZONED.lock().unwrap().push(format!("leave {name}"));
}

// ---- the host half ------------------------------------------------------

// The generated symbols are exported, not named: `const _` scopes the items and
// `#[unsafe(no_mangle)]` exports them. Declaring them is how a host reaches a
// dylib's table, and doing it here keeps this test on the same path.
unsafe extern "C" {
    fn gg_game_abi() -> AbiInfo;
    fn gg_game_init(api: *const HostApiV1);
    fn gg_game_components() -> ComponentsTable;
    fn gg_game_verbs() -> gg_ecs::boundary::VerbsTable;
    fn gg_game_systems() -> SystemsTable;
}

struct Loaded {
    table: SystemsTable,
    systems: Vec<gg_ecs::boundary::SystemEntry>,
}

/// Play the host's load sequence: check, hand over the table, read the systems.
fn load() -> Loaded {
    // SAFETY: the symbols are this binary's own, emitted by `gg_game!` above.
    let info = unsafe { gg_game_abi() };
    assert_eq!(info, boundary::abi_info(), "one build, one description");
    // SAFETY: `host_api()` returns a `&'static` table (§4.2.2).
    unsafe { gg_game_init(core::ptr::from_ref(boundary::host_api())) };

    // SAFETY: as above; the table's entries live for the process.
    let table = unsafe { gg_game_systems() };
    // SAFETY: `entries`/`len` describe one live slice owned by the game side.
    let systems = unsafe { core::slice::from_raw_parts(table.entries, table.len as usize) };
    Loaded {
        table,
        systems: systems.to_vec(),
    }
}

impl Loaded {
    fn name(&self, at: usize) -> String {
        let entry = self.systems[at];
        // SAFETY: `name`/`name_len` describe UTF-8 the game side owns.
        let bytes = unsafe { core::slice::from_raw_parts(entry.name, entry.name_len as usize) };
        String::from_utf8(bytes.to_vec()).expect("system names are UTF-8")
    }

    /// One tick, through the host driver the shell uses — not a loop written
    /// here, so what these tests exercise is what `gg-runtime` runs.
    fn tick(&self, world: &mut World, tick: u64) -> Result<(), gg_ecs::boundary::SystemPanic> {
        let ctx = TickCtx {
            tick,
            tick_hz: 60,
            reserved: 0,
            input: gg_abi::InputFrame::default(),
            previous: gg_abi::InputFrame::default(),
        };
        // SAFETY: the table is this binary's own and its entries live for the
        // process; `ctx` outlives the call.
        unsafe { world.run_systems(&self.table, &ctx) }
    }
}

/// The game's declared components, as the host reads them off a loaded dylib.
fn declared() -> &'static [gg_ecs::boundary::ComponentLayout] {
    // SAFETY: the game side owns the table for the process's lifetime.
    let table = unsafe { gg_game_components() };
    // SAFETY: `entries`/`len` describe one live slice.
    unsafe { core::slice::from_raw_parts(table.entries, table.len as usize) }
}

/// A world carrying the game's components, registered the way the host does it:
/// from the declaration table, with no Rust type named here.
fn world() -> World {
    load();
    let mut world = World::new();
    // SAFETY: the table is this binary's own and never unloaded.
    unsafe { world.adopt(&gg_game_components()) }.unwrap();
    world
}

// ---- the tests ----------------------------------------------------------

#[test]
fn the_table_lists_every_system_in_declaration_order() {
    // Order in the table *is* execution order (§4.1), which is what makes
    // ordering game code and therefore reloadable.
    let loaded = load();
    let names: Vec<String> = (0..loaded.systems.len())
        .map(|at| loaded.name(at))
        .collect();
    assert_eq!(
        names,
        [
            "movement", "stamp", "spawner", "talker", "exploder", "aliaser"
        ]
    );
}

#[test]
fn a_systems_id_is_hashed_from_its_name_and_survives_reordering() {
    let loaded = load();
    for (at, entry) in loaded.systems.iter().enumerate() {
        assert_eq!(
            entry.id,
            gg_ecs::boundary::system_id(&loaded.name(at)),
            "the id must be derivable from the name alone — an index would make \
             a reordered table look like five replaced systems"
        );
    }
    let ids: Vec<u64> = loaded.systems.iter().map(|e| e.id).collect();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), ids.len(), "distinct names, distinct ids");
}

#[test]
fn the_declared_components_are_what_the_host_would_have_computed_itself() {
    // The dylib publishes schemas the host migrates against, so "the table says
    // what the types say" is the property the whole migration path rests on.
    load();
    // SAFETY: the game side owns the table for the process's lifetime.
    let table = unsafe { gg_game_components() };
    let layouts =
        // SAFETY: `entries`/`len` describe one live slice.
        unsafe { core::slice::from_raw_parts(table.entries, table.len as usize) };
    assert_eq!(layouts.len(), 3);

    let position = layouts
        .iter()
        .find(|l| l.id == gg_ecs::component_id::<Position>().get())
        .expect("Position is declared");
    assert_eq!(
        u128::from_le_bytes(position.schema_hash),
        gg_ecs::schema_hash::<Position>().get()
    );
    assert_eq!(position.size as usize, size_of::<Position>());
    assert_eq!(position.align as usize, align_of::<Position>());
    assert_eq!(position.field_len, 2);

    // SAFETY: `fields`/`field_len` describe one live slice of descriptors.
    let fields =
        unsafe { core::slice::from_raw_parts(position.fields, position.field_len as usize) };
    let names: Vec<&str> = fields
        .iter()
        .map(|f| {
            // SAFETY: `name`/`name_len` describe UTF-8 owned by the game side.
            let bytes = unsafe { core::slice::from_raw_parts(f.name, f.name_len as usize) };
            core::str::from_utf8(bytes).expect("field names are UTF-8")
        })
        .collect();
    assert_eq!(
        names,
        ["x", "y"],
        "declaration order, as migration matches on"
    );
}

#[test]
fn adopting_the_table_registers_what_the_host_has_no_type_for() {
    // The host holds no Rust type for a dylib's components — this is the whole
    // reason `ComponentLayout` exists — so registration has to come off the wire
    // and still produce an entry indistinguishable from a typed one.
    let world = world();
    let registry = world.registry();
    assert_eq!(registry.len(), declared().len());

    let info = registry
        .get(gg_ecs::component_id::<Position>())
        .expect("the declared Position is registered");
    assert_eq!(info.schema, gg_ecs::schema_hash::<Position>());
    assert_eq!(info.size, size_of::<Position>());
    assert_eq!(info.align, align_of::<Position>());
    // Persisted identity, not the Rust path (§4.2). It crosses precisely so the
    // host can name it in a collision or a migration line.
    assert_eq!(info.declared_id, "boundary.position");
    assert_eq!(info.type_name, "Position");
    let fields: Vec<&str> = info.fields.iter().map(|f| f.name).collect();
    assert_eq!(fields, ["x", "y"]);

    // Registry order is by stable id, ascending — the canonical walk (§4.2.1),
    // and not the order the dylib happened to declare things in.
    let ids: Vec<u64> = registry.iter().map(|i| i.id.get()).collect();
    assert!(ids.windows(2).all(|w| w[0] < w[1]), "{ids:?}");
}

#[test]
fn adopting_the_same_build_twice_is_a_no_op_but_a_moved_schema_is_refused() {
    // A reload registers into a *fresh* world for exactly this reason: an id
    // coming back with a different schema describes columns that no longer
    // exist, and answering it with a silent `Ok` would leave the world
    // describing itself wrongly.
    let mut world = world();
    // SAFETY: this binary's own table, never unloaded.
    unsafe { world.adopt(&gg_game_components()) }.expect("an unchanged build re-registers");

    let mut moved = declared().to_vec();
    let position = moved
        .iter_mut()
        .find(|l| l.id == gg_ecs::component_id::<Position>().get())
        .expect("Position is declared");
    position.schema_hash[0] ^= 1;
    let table = ComponentsTable {
        entries: moved.as_ptr(),
        len: moved.len() as u32,
        reserved: 0,
    };

    // SAFETY: `moved` outlives the call and holds the original's pointers.
    let refused = unsafe { world.adopt(&table) }.expect_err("a moved schema is not idempotent");
    let refused = refused.to_string();
    assert!(
        refused.contains("boundary.position") && refused.contains("different schema"),
        "named by declared id, and by what is wrong: {refused}"
    );
}

#[test]
fn the_host_registering_its_own_protocol_types_is_what_makes_adopt_refuse_them() {
    // §4.5's `Renderable`/`Eye`/`Model`/`Light` are declared by the game *and*
    // read by the host through typed queries at extract. Adopting into a fresh
    // world takes the insert branch for every one of them, so a dylib that laid
    // one out differently is accepted and the disagreement surfaces later as a
    // host-side panic in `ArchetypeView::assert_shape` — outside every shim,
    // unwinding through the window callback, the one place the boundary is not
    // unwind-free. Registering the host's compiled-in schema *first* is what
    // turns that into a refusal by name at load, and it is what `gg-runtime`
    // does before every adopt.
    let mut world = World::new();
    world
        .register::<gg_ecs::boundary::Renderable>()
        .expect("the host's own protocol type registers");

    let mut moved = boundary::layout_of::<gg_ecs::boundary::Renderable>();
    moved.schema_hash[0] ^= 1; // a field reordered, resized or retyped
    let table = ComponentsTable {
        entries: core::ptr::from_ref(&moved),
        len: 1,
        reserved: 0,
    };
    // SAFETY: `moved` outlives the call and carries this binary's own pointers.
    let refused = unsafe { world.adopt(&table) }
        .expect_err("a dylib laying the protocol out differently is not adoptable")
        .to_string();
    assert!(
        refused.contains("gg.renderable") && refused.contains("different schema"),
        "refused by declared id, and by what is wrong: {refused}"
    );

    // The agreeing case must still adopt, or the check would refuse every game.
    let agreed = boundary::layout_of::<gg_ecs::boundary::Renderable>();
    let table = ComponentsTable {
        entries: core::ptr::from_ref(&agreed),
        len: 1,
        reserved: 0,
    };
    // SAFETY: as above.
    assert_eq!(unsafe { world.adopt(&table) }.unwrap(), 1);
}

#[test]
fn a_reload_of_an_unchanged_build_is_canonical_hash_neutral() {
    // The sequence `gg-runtime` runs at every swap — snapshot, adopt into a
    // fresh world, restore — with the new build identical to the old. §6 M5
    // requires the hash to be untouched across that, which is what makes "state
    // intact" a measurement rather than an impression.
    let loaded = load();
    let mut world = world();
    for tick in 0..5 {
        let entity = world.spawn();
        world
            .insert(entity, Position { x: tick, y: -tick })
            .unwrap();
        world.insert(entity, Velocity { dx: 1, dy: 2 }).unwrap();
        loaded.tick(&mut world, tick as u64).unwrap();
    }
    let before = world.canonical_hash();

    let snapshot = world.snapshot();
    let mut swapped = World::new();
    // SAFETY: this binary's own table, never unloaded.
    unsafe { swapped.adopt(&gg_game_components()) }.unwrap();
    let report = swapped.restore(&snapshot).unwrap();

    assert!(report.is_clean(), "an unchanged build migrates nothing");
    assert_eq!(swapped.canonical_hash(), before, "state survived the swap");
    // And it keeps producing the pre-swap world's results: driving both one more
    // tick has to leave them identical, which the entity allocator's freelist is
    // half of (§4.8) and the systems seeing the same columns is the other.
    loaded.tick(&mut swapped, 5).unwrap();
    loaded.tick(&mut world, 5).unwrap();
    assert_eq!(swapped.canonical_hash(), world.canonical_hash());
}

#[test]
fn a_query_crossing_the_boundary_sees_the_hosts_columns() {
    let loaded = load();
    let mut world = world();

    let moving = world.spawn();
    world.insert(moving, Position { x: 0, y: 0 }).unwrap();
    world.insert(moving, Velocity { dx: 2, dy: -1 }).unwrap();
    // A second archetype, so the query pages over more than one match.
    let still = world.spawn();
    world.insert(still, Position { x: 7, y: 7 }).unwrap();
    world.insert(still, Tag { seen_at_tick: 0 }).unwrap();

    loaded
        .tick(&mut world, 1)
        .expect("no system fails at tick 1");

    assert_eq!(
        world.get::<Position>(moving),
        Some(&Position { x: 2, y: -1 }),
        "the system wrote through a column pointer into host storage"
    );
    assert_eq!(
        world.get::<Position>(still),
        Some(&Position { x: 7, y: 7 }),
        "an archetype without Velocity does not match the two-column query"
    );
    assert_eq!(
        world.get::<Tag>(still).map(|t| t.seen_at_tick),
        Some(1),
        "the tick reached game code as a count"
    );
}

#[test]
fn a_query_pages_over_more_archetypes_than_fit_in_one_call() {
    // The paging loop is the part of `each` a small world never exercises: one
    // call per page, `skip` advancing past what was already delivered, and a
    // short page ending it. Producing it needs *archetypes*, not entities — one
    // distinct marker component each, which is what `markers!` is for. Ten
    // against a page of eight means two full pages and a short one.
    let loaded = load();
    let mut world = world();
    let mut entities = Vec::new();
    for at in 0..MARKERS + 1 {
        let entity = world.spawn();
        let dx = at as i32 + 1;
        world.insert(entity, Position { x: 0, y: 0 }).unwrap();
        world.insert(entity, Velocity { dx, dy: 0 }).unwrap();
        // The last one keeps the bare `Position + Velocity` archetype.
        if at < MARKERS {
            insert_marker(&mut world, entity, at);
        }
        entities.push((entity, dx));
    }

    // Ten archetypes, or the test is measuring nothing: `each` would deliver all
    // of them in one call and the paging loop would never take its second lap.
    let query = gg_ecs::Query::<(&Position, &Velocity)>::new().unwrap();
    let mut matches = 0;
    world.views(query.access(), |_| matches += 1);
    assert_eq!(matches, 10, "one archetype per marker, plus the bare pair");

    loaded.tick(&mut world, 1).unwrap();
    for (entity, dx) in entities {
        assert_eq!(
            world.get::<Position>(entity).map(|p| p.x),
            Some(dx),
            "every match ran exactly once, across pages"
        );
    }
}

#[test]
fn a_system_spawns_and_inserts_through_the_host_table() {
    let loaded = load();
    let mut world = world();
    let before = world.len();

    loaded
        .tick(&mut world, 3)
        .expect("the spawner does not fail");

    assert_eq!(world.len(), before + 1, "the spawn crossed the boundary");
    let spawned = world
        .entity_hashes()
        .into_iter()
        .map(|(entity, _)| entity)
        .find(|&entity| world.get::<Position>(entity) == Some(&Position { x: 100, y: 100 }));
    assert!(spawned.is_some(), "and so did the insert");
}

#[test]
fn a_panicking_system_is_a_status_naming_the_system_and_the_process_survives() {
    // §4.2.2's load-bearing claim: the shim catches inside the dylib, so a
    // panic is a value the host reads rather than a process abort. If this test
    // aborts instead of failing, the shim is gone.
    let loaded = load();
    let mut world = world();

    let panicked = loaded
        .tick(&mut world, 9)
        .expect_err("tick 9 detonates the exploder");
    assert_eq!(panicked.system, "exploder", "reported by name");
    assert!(
        panicked.message.contains("the ugly game broke on purpose"),
        "the payload crosses: {panicked}"
    );
    assert!(
        panicked.message.contains("boundary.rs"),
        "and so does the location: {panicked}"
    );

    // The point of catching: the next tick runs.
    loaded.tick(&mut world, 10).expect("the process survived");
}

#[test]
fn game_output_arrives_on_the_hosts_stream() {
    // §4.2.2: game log lines land in the same stream — and the same Tracy
    // capture — as engine output, which is only true if they cross the boundary
    // rather than being printed dylib-side.
    let loaded = load();
    let _ = boundary::set_logger(record);
    let mut world = world();
    loaded.tick(&mut world, 1).unwrap();

    let logged = LOGGED.lock().unwrap();
    assert!(
        logged
            .iter()
            .any(|(level, message)| *level == boundary::log_level::INFO
                && message == "the ugly game says hello"),
        "the line crossed with its level intact: {logged:?}"
    );
}

#[test]
fn the_system_zone_brackets_each_system_by_its_declared_name() {
    // §4.8's per-system CPU zones. What a profiler needs is the *name* — table
    // data read out of a dylib, which is why this is a hook and not a
    // `profiling::scope!` — and a bracket that closes around the call rather
    // than beside it.
    let loaded = load();
    let _ = boundary::set_system_zone(zone);
    let mut world = world();
    ZONED.lock().unwrap().clear();
    loaded.tick(&mut world, 1).unwrap();

    let zoned = ZONED.lock().unwrap();
    assert_eq!(
        *zoned,
        [
            "enter movement",
            "leave movement",
            "enter stamp",
            "leave stamp",
            "enter spawner",
            "leave spawner",
            "enter talker",
            "leave talker",
            "enter exploder",
            "leave exploder",
            "enter aliaser",
            "leave aliaser",
        ],
        "one closed bracket per system, in the table's order (§4.1)"
    );
}

#[test]
fn a_zone_that_never_runs_the_system_is_reported_rather_than_passed_off_as_a_tick() {
    // An instrument that swallowed the call would leave the tick half
    // simulated with nothing to show for it — a determinism break whose cause
    // is the profiler. It halts the sim like any other system failure.
    let loaded = load();
    let _ = boundary::set_system_zone(zone);
    let mut world = world();
    *SWALLOW.lock().unwrap() = Some(std::thread::current().id());
    let outcome = loaded.tick(&mut world, 1);
    *SWALLOW.lock().unwrap() = None;

    let failed = outcome.expect_err("a swallowed system is not a completed tick");
    assert_eq!(
        failed.system, "movement",
        "reported by name, at the first one"
    );
    assert!(
        failed.message.contains("never ran"),
        "and says what happened: {failed}"
    );
}

#[test]
fn a_panic_outside_a_system_is_left_to_whatever_hook_was_already_installed() {
    // The capture hook has to be scoped to systems. A hook that swallowed every
    // panic in the process would make the engine's own failures silent — and a
    // failing assertion in this very file print nothing — while the next system
    // panic got reported with somebody else's message.
    let loaded = load();
    let mut world = world();
    let escaped = std::panic::catch_unwind(|| panic!("not a system, and not ours to swallow"));
    assert!(escaped.is_err());

    let panicked = loaded.tick(&mut world, 9).expect_err("tick 9 detonates");
    assert!(
        panicked.message.contains("the ugly game broke on purpose"),
        "the system's own panic, not the one before it: {panicked}"
    );
}

#[test]
fn a_system_after_a_panicking_one_does_not_run() {
    // The sim halts where it stood (§4.2.2) rather than finishing the tick with
    // half its systems skipped and no record of which.
    let loaded = load();
    let mut world = world();
    let entity = world.spawn();
    world.insert(entity, Tag { seen_at_tick: 0 }).unwrap();

    // `stamp` runs before `exploder`, `aliaser` after.
    assert!(loaded.tick(&mut world, 9).is_err());
    assert_eq!(
        world.get::<Tag>(entity).map(|t| t.seen_at_tick),
        Some(9),
        "systems before the panic did run"
    );
}

#[test]
fn an_aliasing_query_is_refused_before_any_pointer_exists() {
    // `aliaser` asserts the status itself; reaching the end of a tick proves the
    // host refused it rather than handing over two mutable views of one column.
    let loaded = load();
    let mut world = world();
    let entity = world.spawn();
    world.insert(entity, Position { x: 1, y: 1 }).unwrap();
    loaded
        .tick(&mut world, 1)
        .expect("refusal is not a failure");
}

#[test]
fn iteration_order_across_the_boundary_is_the_hosts_order() {
    // §4.2 guarantee 1 has to survive the seam, or a replay recorded in dev and
    // replayed in dist would diverge on system iteration alone.
    load();
    let mut world = world();
    for i in 0..8i32 {
        let entity = world.spawn();
        world.insert(entity, Position { x: i, y: 0 }).unwrap();
        world.insert(entity, Velocity { dx: 1, dy: 0 }).unwrap();
    }

    // Scoped, not dropped: `handle` is a raw pointer, so nothing but this block
    // stops the world being touched while a `GameWorld` over it is alive.
    let mut seen = Vec::new();
    {
        let handle = boundary::handle(&mut world);
        let ctx = TickCtx {
            tick: 0,
            tick_hz: 60,
            reserved: 0,
            input: gg_abi::InputFrame::default(),
            previous: gg_abi::InputFrame::default(),
        };
        // SAFETY: `handle` is this world's and `ctx` outlives the borrow; the
        // host table was installed by `load`.
        let mut game = unsafe { GameWorld::new(handle, &ctx) };
        game.each::<&Position>(|entity, position| seen.push((entity, position.x)))
            .unwrap();
    }

    let expected: Vec<(Entity, i32)> = {
        let query = gg_ecs::Query::<&Position>::new().unwrap();
        let mut out = Vec::new();
        world.each(&query, |entity, position: &Position| {
            out.push((entity, position.x));
        });
        out
    };
    assert_eq!(seen, expected, "one order, both sides of the boundary");
}

#[test]
fn the_tick_context_answers_edges_without_the_system_keeping_state() {
    // §4.2.2 forbids state that outlives a tick, so `just_pressed` has to come
    // out of the two frames the context carries.
    load();
    let action = gg_abi::ActionId::new(3);
    let mut world = world();
    let handle = boundary::handle(&mut world);
    let mut input = gg_abi::InputFrame::default();
    input.buttons |= 1 << action.index();

    let ctx = TickCtx {
        tick: 12,
        tick_hz: 60,
        reserved: 0,
        input,
        previous: gg_abi::InputFrame::default(),
    };
    // SAFETY: as the previous test.
    let game = unsafe { GameWorld::new(handle, &ctx) };
    assert!(game.pressed(action));
    assert!(game.just_pressed(action));
    assert_eq!(game.tick(), 12);
    assert_eq!(game.tick_hz(), 60);

    let held = TickCtx {
        previous: input,
        ..ctx
    };
    // SAFETY: as above.
    let game = unsafe { GameWorld::new(handle, &held) };
    assert!(game.pressed(action), "still down");
    assert!(!game.just_pressed(action), "but no longer an edge");
}

#[test]
fn the_declared_verbs_cross_in_id_order() {
    // The list's order *is* the id space a replay records (§4.7), so what the
    // host reads back has to be the declaration, not a set.
    load();
    // SAFETY: the table is this binary's own and lives for the process.
    let verbs = unsafe { boundary::read_verbs(&gg_game_verbs()) };
    assert_eq!(verbs.actions, ["fire", "restart"]);
    assert_eq!(verbs.axes, ["aim_x"]);
    // A game that takes no input declares none, and the empty case must be a
    // pair of empty slices rather than a null the host follows.
    let empty = gg_ecs::boundary::VerbsTable {
        actions: core::ptr::null(),
        axes: core::ptr::null(),
        action_len: 0,
        axis_len: 0,
    };
    // SAFETY: both lengths are zero, which `read_verbs` documents as the one
    // case where the pointers are never followed.
    let none = unsafe { boundary::read_verbs(&empty) };
    assert_eq!(none, gg_ecs::boundary::Verbs::default());
}
