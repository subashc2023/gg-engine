//! World behaviour and the §4.2 determinism guarantees.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::{Component, Entity, World};
use gg_math::sim;

#[derive(Clone, Copy, PartialEq, Debug, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "position")]
#[repr(C)]
struct Position {
    p: sim::DVec3,
}

#[derive(Clone, Copy, PartialEq, Debug, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "velocity")]
#[repr(C)]
struct Velocity {
    v: sim::DVec3,
}

#[derive(Clone, Copy, PartialEq, Debug, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "health")]
#[repr(C)]
struct Health {
    hp: u32,
    shield: u32,
}

#[derive(Clone, Copy, PartialEq, Debug, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "frozen")]
#[repr(C)]
struct Frozen;

fn pos(x: f64) -> Position {
    Position {
        p: sim::DVec3::new(x, 0.0, 0.0),
    }
}

#[test]
fn insert_get_and_overwrite() {
    let mut w = World::new();
    let e = w.spawn();
    assert!(w.get::<Position>(e).is_none());

    w.insert(e, pos(1.0)).unwrap();
    assert_eq!(w.get::<Position>(e), Some(&pos(1.0)));

    // Overwriting an existing component must not move the entity.
    w.insert(e, pos(2.0)).unwrap();
    assert_eq!(w.get::<Position>(e), Some(&pos(2.0)));

    w.get_mut::<Position>(e).unwrap().p.x = 5.0;
    assert_eq!(w.get::<Position>(e), Some(&pos(5.0)));
}

#[test]
fn adding_a_component_carries_the_others_across_the_archetype_move() {
    let mut w = World::new();
    let e = w.spawn();
    w.insert(e, pos(1.0)).unwrap();
    w.insert(e, Health { hp: 7, shield: 3 }).unwrap();
    w.insert(
        e,
        Velocity {
            v: sim::DVec3::new(0.0, 9.0, 0.0),
        },
    )
    .unwrap();

    // Three archetype moves later, everything inserted first is still intact.
    assert_eq!(w.get::<Position>(e), Some(&pos(1.0)));
    assert_eq!(w.get::<Health>(e), Some(&Health { hp: 7, shield: 3 }));
    assert_eq!(w.get::<Velocity>(e).unwrap().v.y, 9.0);
}

#[test]
fn removing_a_component_keeps_the_rest() {
    let mut w = World::new();
    let e = w.spawn();
    w.insert(e, pos(1.0)).unwrap();
    w.insert(e, Health { hp: 7, shield: 3 }).unwrap();

    assert!(w.remove::<Health>(e));
    assert!(
        !w.remove::<Health>(e),
        "second remove reports nothing to do"
    );
    assert!(w.get::<Health>(e).is_none());
    assert_eq!(w.get::<Position>(e), Some(&pos(1.0)));
}

#[test]
fn zero_sized_marker_components_work() {
    let mut w = World::new();
    let e = w.spawn();
    w.insert(e, pos(1.0)).unwrap();
    w.insert(e, Frozen).unwrap();
    assert!(w.contains::<Frozen>(e));
    assert_eq!(w.get::<Position>(e), Some(&pos(1.0)));
    assert!(w.remove::<Frozen>(e));
    assert!(!w.contains::<Frozen>(e));
}

#[test]
fn despawn_repairs_the_location_of_the_row_swap_remove_moved() {
    // The classic archetype-storage bug: despawning a middle row moves the
    // last row into the hole, and the moved entity's location must follow.
    let mut w = World::new();
    let entities: Vec<Entity> = (0..4)
        .map(|i| {
            let e = w.spawn();
            w.insert(e, pos(f64::from(i))).unwrap();
            e
        })
        .collect();

    assert!(w.despawn(entities[1]));
    assert!(!w.is_alive(entities[1]));
    for (i, &e) in entities.iter().enumerate() {
        if i == 1 {
            continue;
        }
        assert_eq!(
            w.get::<Position>(e),
            Some(&pos(i as f64)),
            "entity {i} lost its row after a swap-remove elsewhere"
        );
    }
}

#[test]
fn operations_on_dead_entities_are_reported_not_fatal() {
    let mut w = World::new();
    let e = w.spawn();
    w.insert(e, pos(1.0)).unwrap();
    assert!(w.despawn(e));

    assert!(!w.despawn(e));
    assert!(w.get::<Position>(e).is_none());
    assert!(!w.contains::<Position>(e));
    assert!(!w.remove::<Position>(e));
    assert!(w.insert(e, pos(2.0)).is_err());
}

#[test]
fn a_reused_index_does_not_inherit_the_previous_components() {
    let mut w = World::new();
    let a = w.spawn();
    w.insert(a, pos(1.0)).unwrap();
    assert!(w.despawn(a));

    let b = w.spawn();
    assert_eq!(b.index(), a.index(), "the freelist should reuse the index");
    assert!(
        w.get::<Position>(b).is_none(),
        "a fresh entity must not see the previous occupant's state"
    );
}

// ---- the canonical hash (§4.2.1) --------------------------------------

fn build(order: &[usize]) -> World {
    // Same final state, different insertion order and different archetype
    // creation order.
    let mut w = World::new();
    let mut entities = Vec::new();
    for _ in 0..4 {
        entities.push(w.spawn());
    }
    for &i in order {
        let e = entities[i];
        w.insert(e, pos(i as f64)).unwrap();
        w.insert(
            e,
            Health {
                hp: i as u32,
                shield: 1,
            },
        )
        .unwrap();
    }
    w
}

#[test]
fn the_canonical_hash_is_independent_of_storage_order() {
    // The property the whole design rests on (§4.2.1): sorting by entity id
    // means archetype layout cannot reach the hash, so internal storage rework
    // can demand identical hashes across a rewrite.
    let a = build(&[0, 1, 2, 3]);
    let b = build(&[3, 1, 0, 2]);
    assert_eq!(a.canonical_hash(), b.canonical_hash());
}

#[test]
fn the_canonical_hash_moves_when_state_moves() {
    let a = build(&[0, 1, 2, 3]);
    let mut b = build(&[0, 1, 2, 3]);
    let victim = b.spawn();
    b.insert(victim, pos(99.0)).unwrap();
    assert_ne!(a.canonical_hash(), b.canonical_hash());

    // And a single bit of component payload is enough.
    let mut c = build(&[0, 1, 2, 3]);
    let mut d = build(&[0, 1, 2, 3]);
    let (ce, de) = (c.spawn(), d.spawn());
    c.insert(ce, Health { hp: 1, shield: 0 }).unwrap();
    d.insert(de, Health { hp: 1, shield: 1 }).unwrap();
    assert_ne!(c.canonical_hash(), d.canonical_hash());
}

#[test]
fn the_canonical_hash_is_stable_across_a_replayed_sequence() {
    fn run() -> Vec<u128> {
        let mut w = World::new();
        let mut live: Vec<Entity> = Vec::new();
        let mut seq = Vec::new();
        for tick in 0..200u32 {
            for k in 0..3 {
                let e = w.spawn();
                w.insert(e, pos(f64::from(tick * 3 + k))).unwrap();
                if (tick + k) % 2 == 0 {
                    w.insert(
                        e,
                        Health {
                            hp: tick,
                            shield: k,
                        },
                    )
                    .unwrap();
                }
                if (tick + k) % 5 == 0 {
                    w.insert(e, Frozen).unwrap();
                }
                live.push(e);
            }
            if tick % 3 == 0 && !live.is_empty() {
                let victim = live.swap_remove((tick as usize * 7) % live.len());
                w.despawn(victim);
            }
            if tick % 7 == 0 && !live.is_empty() {
                let target = live[(tick as usize * 11) % live.len()];
                w.remove::<Health>(target);
            }
            seq.push(w.canonical_hash().get());
        }
        seq
    }
    assert_eq!(
        run(),
        run(),
        "a replayed sequence must reproduce every hash"
    );
}

#[test]
fn an_empty_world_still_has_a_domain_specific_hash() {
    let a = World::new().canonical_hash();
    let b = World::new().canonical_hash();
    assert_eq!(a, b);
    let mut w = World::new();
    let e = w.spawn();
    assert_ne!(a, w.canonical_hash());
    w.despawn(e);
    // Not equal to a fresh world: the freelist and generation counters differ,
    // and they decide what the next spawn is called.
    assert_ne!(a, w.canonical_hash());
}

#[test]
fn entity_hashes_locate_the_one_row_that_moved() {
    // The divergence probe (§5.6): two worlds that disagree must disagree at a
    // *named* entity, not merely in the whole-world hash.
    let build = || {
        let mut w = World::new();
        let mut entities = Vec::new();
        for i in 0..8 {
            let e = w.spawn();
            w.insert(e, pos(i as f64)).unwrap();
            entities.push(e);
        }
        (w, entities)
    };
    let (mut a, entities) = build();
    let (b, _) = build();
    assert_eq!(
        a.entity_hashes(),
        b.entity_hashes(),
        "identical worlds agree"
    );

    // Nudge one entity by one ULP-ish amount and find it by hash alone.
    a.get_mut::<Position>(entities[5]).unwrap().p.x += 1.0;
    assert_ne!(a.canonical_hash(), b.canonical_hash());
    let (ha, hb) = (a.entity_hashes(), b.entity_hashes());
    let first = ha
        .iter()
        .zip(&hb)
        .find(|(x, y)| x != y)
        .map(|(x, _)| x.0)
        .expect("the worlds differ, so some entity must");
    assert_eq!(first, entities[5]);
    // ...and exactly one row moved.
    assert_eq!(ha.iter().zip(&hb).filter(|(x, y)| x != y).count(), 1);
}
