//! Hashed side tables (§4.2.1): sim state above the columns, hashed in the
//! canonical pass, free of the component `Pod` rule.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::{Component, Entity, SideTable, StateHash, World};
use gg_math::sim;

#[derive(Clone, Copy, PartialEq, Debug, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "position")]
#[repr(C)]
struct Position {
    p: sim::DVec3,
}

/// A component holds the handle; the payload lives in the table. This is the
/// documented shape for strategy-layer state (§4.2.1).
#[derive(Clone, Copy, PartialEq, Debug, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "queue-handle")]
#[repr(C)]
struct QueueHandle {
    slot: u32,
}

#[derive(Clone, Debug, StateHash)]
struct Order {
    owner: Entity,
    item: u32,
    remaining: f64,
}

/// Growable, heap-backed, keyed by entity id — none of which a component may be.
#[derive(Clone, Debug, Default, SideTable)]
#[side_table(id = "production-queue")]
struct ProductionQueue {
    orders: Vec<Order>,
    label: String,
}

#[derive(Clone, Debug, Default, SideTable)]
#[side_table(id = "trade-routes")]
struct TradeRoutes {
    edges: Vec<u64>,
}

fn order(owner: Entity, item: u32, remaining: f64) -> Order {
    Order {
        owner,
        item,
        remaining,
    }
}

#[test]
fn a_side_table_holds_what_a_component_cannot() {
    let mut w = World::new();
    let e = w.spawn();
    w.insert(e, QueueHandle { slot: 0 }).unwrap();
    w.insert_side_table(ProductionQueue {
        orders: vec![order(e, 7, 12.5)],
        label: "shipyard".into(),
    })
    .unwrap();

    let q = w.side_table::<ProductionQueue>().unwrap();
    assert_eq!(q.orders.len(), 1);
    assert_eq!(q.orders[0].owner, e);
    assert_eq!(q.label, "shipyard");

    w.side_table_mut::<ProductionQueue>()
        .unwrap()
        .orders
        .push(order(e, 8, 1.0));
    assert_eq!(w.side_table::<ProductionQueue>().unwrap().orders.len(), 2);
}

#[test]
fn side_table_state_reaches_the_canonical_hash() {
    let mut w = World::new();
    let e = w.spawn();
    w.insert(
        e,
        Position {
            p: sim::DVec3::new(1.0, 2.0, 3.0),
        },
    )
    .unwrap();
    let bare = w.canonical_hash();

    w.insert_side_table(ProductionQueue::default()).unwrap();
    let installed = w.canonical_hash();
    assert_ne!(bare, installed, "installing a table is a state change");

    w.side_table_mut::<ProductionQueue>()
        .unwrap()
        .orders
        .push(order(e, 1, 2.0));
    let filled = w.canonical_hash();
    assert_ne!(installed, filled, "table contents must be hashed");

    // Removal is state leaving the world, and returns to the pre-install value.
    let taken = w.remove_side_table::<ProductionQueue>().unwrap();
    assert_eq!(taken.orders.len(), 1);
    assert_eq!(w.canonical_hash(), bare);
}

#[test]
fn the_hash_does_not_depend_on_installation_order() {
    let build = |flip: bool| {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(
            e,
            Position {
                p: sim::DVec3::new(0.5, 0.0, 0.0),
            },
        )
        .unwrap();
        let queue = ProductionQueue {
            orders: vec![order(e, 3, 4.0)],
            label: "a".into(),
        };
        let routes = TradeRoutes {
            edges: vec![1, 2, 3],
        };
        if flip {
            w.insert_side_table(routes).unwrap();
            w.insert_side_table(queue).unwrap();
        } else {
            w.insert_side_table(queue).unwrap();
            w.insert_side_table(routes).unwrap();
        }
        w.canonical_hash()
    };
    assert_eq!(build(false), build(true), "tables walk sorted by id");
}

#[test]
fn side_tables_do_not_split_the_fast_path_from_the_protocol() {
    // Tables are protocol-encoded in both walks; the §4.2.1 equality must
    // survive their presence.
    let mut w = World::new();
    for i in 0..16u32 {
        let e = w.spawn();
        w.insert(
            e,
            Position {
                p: sim::DVec3::new(f64::from(i), 0.0, 0.0),
            },
        )
        .unwrap();
        w.insert(e, QueueHandle { slot: i }).unwrap();
    }
    w.insert_side_table(ProductionQueue {
        orders: (0..8)
            .map(|i| order(Entity::NONE, i, f64::from(i)))
            .collect(),
        label: "mixed".into(),
    })
    .unwrap();
    w.insert_side_table(TradeRoutes {
        edges: (0..32).collect(),
    })
    .unwrap();
    assert_eq!(w.canonical_hash(), w.canonical_hash_via_protocol());
}

#[test]
fn installing_one_table_twice_is_an_error_not_a_silent_replacement() {
    let mut w = World::new();
    w.insert_side_table(TradeRoutes::default()).unwrap();
    let err = w.insert_side_table(TradeRoutes::default()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("already installed"), "{msg}");
    assert!(msg.contains("TradeRoutes"), "{msg}");
}

#[test]
fn two_types_claiming_one_id_is_an_error_naming_both() {
    // A different type in a different module claiming the same declared id:
    // identity is the string, so this is a collision, not a rename.
    mod elsewhere {
        #[derive(Default, gg_ecs::SideTable)]
        #[side_table(id = "trade-routes")]
        pub struct Routes {
            pub edges: Vec<u64>,
        }
    }
    let mut w = World::new();
    w.insert_side_table(TradeRoutes::default()).unwrap();
    let err = w
        .insert_side_table(elsewhere::Routes::default())
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("TradeRoutes"), "{msg}");
    assert!(msg.contains("Routes"), "{msg}");
    assert!(msg.contains("trade-routes"), "{msg}");
}

/// The side-table half of the folded-id check (§4.2.1). Its domain is the one
/// thing the derive restates rather than shares, so the two namespaces are
/// compared under one name — a domain pasted from the component branch would
/// pass the first assertion and fail the last.
#[test]
fn the_folded_side_table_id_is_the_one_of_would_have_computed() {
    assert_eq!(
        ProductionQueue::SIDE_TABLE_ID,
        gg_ecs::SideTableId::of(ProductionQueue::DECLARED_ID)
    );
    assert_eq!(
        TradeRoutes::SIDE_TABLE_ID,
        gg_ecs::side_table_id_of!("trade-routes")
    );
    assert_ne!(
        ProductionQueue::SIDE_TABLE_ID,
        TradeRoutes::SIDE_TABLE_ID,
        "a fold that ignored the declared id would pass the equalities above"
    );
    assert_ne!(
        gg_ecs::side_table_id_of!("production-queue").get(),
        gg_ecs::component_id_of!("production-queue").get(),
        "the two folds must keep §4.2.1's domains apart, as `of` does"
    );
}

#[test]
fn a_component_and_a_side_table_sharing_a_name_do_not_share_an_id() {
    // Different domains (§4.2.1): the two namespaces are deliberately disjoint.
    assert_ne!(
        gg_ecs::ComponentId::of("production-queue").get(),
        gg_ecs::SideTableId::of("production-queue").get()
    );
}

// ---- the reload boundary, which is what decides who may declare one ------
//
// `App::swap`'s migrating path is snapshot → `World::new()` → adopt the rebuilt
// dylib's schemas → restore, and nothing between those steps installs a side
// table. The two tests below are that shape with the game's table in it, and
// together they are why §6 M38 item 12 put demo 13's event queue in components
// instead: a table the *game* declares either kills the reload or survives it
// empty, and which of the two happens depends on whether some unrelated
// component's schema moved in the same edit.

#[test]
fn the_migrating_reload_refuses_a_table_the_fresh_world_does_not_have() {
    let mut live = World::new();
    live.register::<Position>().unwrap();
    let e = live.spawn();
    live.insert(
        e,
        Position {
            p: sim::DVec3::new(1.0, 0.0, 0.0),
        },
    )
    .unwrap();
    live.insert_side_table(ProductionQueue {
        orders: vec![order(e, 7, 12.5)],
        label: "shipyard".into(),
    })
    .unwrap();

    let mut fresh = World::new();
    fresh.register::<Position>().unwrap();
    let msg = fresh.restore(&live.snapshot()).unwrap_err().to_string();
    assert!(msg.contains("production-queue"), "{msg}");
    assert!(msg.contains("host-owned"), "{msg}");
}

#[test]
fn re_installing_it_restores_green_with_the_contents_gone() {
    // The worse half: the id check is by *name*, so a fresh table under the same
    // id passes it and the restore reports a clean migration while the queue's
    // rows are gone. The hash is where that shows, and a hash is exactly what
    // §5.6c's reload legs compare either side of a swap.
    let mut live = World::new();
    live.register::<Position>().unwrap();
    let e = live.spawn();
    live.insert(
        e,
        Position {
            p: sim::DVec3::new(1.0, 0.0, 0.0),
        },
    )
    .unwrap();
    live.insert_side_table(ProductionQueue {
        orders: vec![order(e, 7, 12.5)],
        label: "shipyard".into(),
    })
    .unwrap();

    let mut fresh = World::new();
    fresh.register::<Position>().unwrap();
    fresh.insert_side_table(ProductionQueue::default()).unwrap();
    let report = fresh.restore(&live.snapshot()).unwrap();
    assert_eq!(report.entities, 1, "the columns migrate perfectly");
    assert!(
        fresh
            .side_table::<ProductionQueue>()
            .unwrap()
            .orders
            .is_empty()
    );
    assert_ne!(
        live.canonical_hash(),
        fresh.canonical_hash(),
        "a reload that loses table contents is a divergence, not a migration"
    );
}

// ---- §1.13 hazard 6 over non-`Pod` state --------------------------------
//
// A side table's floats live behind a `Vec`, so the column scan's layout walk
// cannot reach them; they are witnessed as they pass `StateHasher` instead.
// That is the same hash a divergence would be found by, which is what makes the
// coverage structural rather than a second list to keep in sync.

fn nan_f64() -> f64 {
    f64::from_bits(0x7ff8_0000_0000_0001)
}

#[test]
fn a_nan_behind_a_vec_is_found_and_names_its_table() {
    let mut w = World::new();
    let e = w.spawn();
    w.insert_side_table(ProductionQueue {
        orders: vec![order(e, 7, 1.0), order(e, 8, nan_f64())],
        label: "forge".to_owned(),
    })
    .unwrap();

    let site = w.scan_for_nan().expect("the NaN must be found");
    let gg_ecs::NanSite::SideTable(hit) = site else {
        panic!("a side-table NaN must not be reported as a column: {site}");
    };
    assert_eq!(hit.declared, "production-queue");
    assert_eq!(hit.type_name, "ProductionQueue");
    assert_eq!(hit.width, 64);
    assert_eq!(
        hit.bits, 0x7ff8_0000_0000_0001,
        "the payload is what differs between architectures, so it is reported"
    );
    assert!(site.to_string().contains("ProductionQueue"));
}

#[test]
fn a_clean_side_table_reports_nothing() {
    let mut w = World::new();
    let e = w.spawn();
    w.insert_side_table(ProductionQueue {
        orders: vec![order(e, 7, 12.5), order(e, 8, f64::NEG_INFINITY)],
        label: "forge".to_owned(),
    })
    .unwrap();
    assert!(
        w.scan_for_nan().is_none(),
        "±inf is not a NaN: its mantissa is zero and it is architecture-agnostic"
    );
}

#[test]
fn integer_bytes_that_spell_a_nan_in_a_table_are_not_reported() {
    // The false positive that would make the gate useless — the same control
    // the column scan carries, on the path that reaches values by type rather
    // than by layout.
    let mut w = World::new();
    w.insert_side_table(TradeRoutes {
        edges: vec![0x7ff8_0000_0000_0001, 0xffff_ffff_ffff_ffff],
    })
    .unwrap();
    assert!(
        w.scan_for_nan().is_none(),
        "a `Vec<u64>` is hashed as integers, whatever its bytes spell"
    );
}

#[test]
fn a_column_nan_outranks_a_side_table_one() {
    // Order is a property of the world, not of the search (§1.13 hazard 6), and
    // the column answer is the precise one — it names a field and a lane.
    let mut w = World::new();
    let e = w.spawn();
    w.insert(
        e,
        Position {
            p: sim::DVec3::new(0.0, nan_f64(), 0.0),
        },
    )
    .unwrap();
    w.insert_side_table(ProductionQueue {
        orders: vec![order(e, 7, nan_f64())],
        label: String::new(),
    })
    .unwrap();
    let hit = w
        .scan_for_nan()
        .and_then(gg_ecs::NanSite::column)
        .expect("the column hit must win");
    assert_eq!(hit.field, "p");
    assert_eq!(hit.lane, 1);
}
