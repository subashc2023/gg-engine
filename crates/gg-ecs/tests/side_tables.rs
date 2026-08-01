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

#[test]
fn a_component_and_a_side_table_sharing_a_name_do_not_share_an_id() {
    // Different domains (§4.2.1): the two namespaces are deliberately disjoint.
    assert_ne!(
        gg_ecs::ComponentId::of("production-queue").get(),
        gg_ecs::SideTableId::of("production-queue").get()
    );
}
