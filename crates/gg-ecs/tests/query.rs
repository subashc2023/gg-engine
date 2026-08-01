//! The typed query layer over the column views (§4.2).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::{Component, Entity, Query, World};
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

/// `n` entities with position+velocity; every other one also has health, so a
/// two-component query has to match two archetypes.
fn world_with(n: u32) -> (World, Vec<Entity>) {
    let mut w = World::new();
    let mut es = Vec::new();
    for i in 0..n {
        let e = w.spawn();
        w.insert(
            e,
            Position {
                p: sim::DVec3::new(f64::from(i), 0.0, 0.0),
            },
        )
        .unwrap();
        w.insert(
            e,
            Velocity {
                v: sim::DVec3::new(1.0, 0.0, 0.0),
            },
        )
        .unwrap();
        if i % 2 == 0 {
            w.insert(e, Health { hp: i, shield: 0 }).unwrap();
        }
        es.push(e);
    }
    (w, es)
}

#[test]
fn a_read_write_tuple_integrates_across_archetypes() {
    let (mut w, es) = world_with(8);
    let q = Query::<(&mut Position, &Velocity)>::new().unwrap();

    let mut seen = 0;
    w.each(&q, |_, (pos, vel)| {
        pos.p += vel.v;
        seen += 1;
    });
    assert_eq!(seen, 8, "every matching entity exactly once");

    for (i, &e) in es.iter().enumerate() {
        assert_eq!(w.get::<Position>(e).unwrap().p.x, i as f64 + 1.0);
    }
}

#[test]
fn the_entity_arrives_alongside_its_row() {
    let (mut w, _) = world_with(6);
    let q = Query::<&Position>::new().unwrap();
    let mut pairs: Vec<(u32, f64)> = Vec::new();
    w.each(&q, |e, pos| pairs.push((e.index(), pos.p.x)));
    pairs.sort_by_key(|&(i, _)| i);
    for (i, (index, x)) in pairs.iter().enumerate() {
        assert_eq!(*index as usize, i);
        assert_eq!(*x, i as f64, "entity {index} paired with the wrong row");
    }
}

#[test]
fn a_narrower_query_matches_supersets() {
    let (mut w, _) = world_with(10);
    // Health is on half the entities; Position is on all of them.
    let all = Query::<&Position>::new().unwrap();
    let half = Query::<&Health>::new().unwrap();
    let (mut a, mut h) = (0, 0);
    w.each(&all, |_, _| a += 1);
    w.each(&half, |_, _| h += 1);
    assert_eq!((a, h), (10, 5));
}

#[test]
fn two_writes_of_distinct_components_are_granted_together() {
    let (mut w, es) = world_with(4);
    let q = Query::<(&mut Position, &mut Velocity)>::new().unwrap();
    w.each(&q, |_, (pos, vel)| {
        vel.v = vel.v * 2.0;
        pos.p += vel.v;
    });
    for &e in &es {
        assert_eq!(w.get::<Velocity>(e).unwrap().v.x, 2.0);
    }
    assert_eq!(w.get::<Position>(es[3]).unwrap().p.x, 5.0);
}

#[test]
fn aliasing_is_refused_when_the_query_is_built_not_when_it_runs() {
    let err = Query::<(&mut Position, &mut Position)>::new().unwrap_err();
    assert!(err.to_string().contains("mutably more than once"), "{err}");

    let err = Query::<(&Position, &mut Position)>::new().unwrap_err();
    assert!(
        err.to_string().contains("both mutably and immutably"),
        "{err}"
    );
}

#[test]
fn a_marker_component_queries_as_rows_with_no_payload() {
    let mut w = World::new();
    for _ in 0..5 {
        let e = w.spawn();
        w.insert(e, Frozen).unwrap();
    }
    let q = Query::<&Frozen>::new().unwrap();
    let mut rows = 0;
    w.each(&q, |_, _| rows += 1);
    assert_eq!(rows, 5);
}

#[test]
fn a_query_matching_nothing_visits_nothing() {
    let (mut w, _) = world_with(4);
    let q = Query::<(&Position, &Frozen)>::new().unwrap();
    let mut visits = 0;
    w.each(&q, |_, _| visits += 1);
    assert_eq!(visits, 0);
}

#[test]
fn the_typed_layer_reproduces_the_raw_view_result_exactly() {
    // The typed layer must be a convenience over `views`, not a second
    // implementation with its own iteration order.
    let (mut w, _) = world_with(9);
    let q = Query::<(&Position, &Velocity)>::new().unwrap();
    let mut typed: Vec<(u64, f64)> = Vec::new();
    w.each(&q, |e, (pos, _)| typed.push((e.to_bits(), pos.p.x)));

    let mut raw: Vec<(u64, f64)> = Vec::new();
    w.views(q.access(), |view| {
        let entities = view.entities();
        let pos = view.read_of::<Position>();
        for (row, &e) in entities.iter().enumerate() {
            raw.push((e.to_bits(), pos[row].p.x));
        }
    });
    assert_eq!(typed, raw);
}

#[test]
#[should_panic(expected = "already taken")]
fn taking_one_write_column_twice_panics_rather_than_aliasing() {
    let (mut w, _) = world_with(2);
    let q = Query::<&mut Position>::new().unwrap();
    w.views(q.access(), |mut view| {
        let a = view.write_of::<Position>();
        // Without the take-once bit this would hand out a second `&mut [T]` to
        // the same bytes: the returned slices live for the world borrow, not
        // for the `&mut view` that produced them.
        let b = view.write_of::<Position>();
        a[0].p.x += 1.0;
        b[0].p.x += 1.0;
    });
}

#[test]
#[should_panic(expected = "does not write")]
fn taking_a_column_the_query_never_asked_for_panics() {
    let (mut w, _) = world_with(2);
    let q = Query::<&Position>::new().unwrap();
    w.views(q.access(), |mut view| {
        let _ = view.write_of::<Position>();
    });
}
