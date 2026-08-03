//! Column views (§4.2.2 prototype) and the fast-path-equals-protocol proof
//! (§4.2.1).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::{Component, Entity, QueryAccess, World, component_id};
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
        // Half the entities get a third component, so the query has to skip a
        // non-matching archetype and match two others.
        if i % 2 == 0 {
            w.insert(e, Health { hp: i, shield: 0 }).unwrap();
        }
        es.push(e);
    }
    (w, es)
}

#[test]
fn a_write_and_a_read_column_iterate_natively() {
    let (mut w, es) = world_with(8);
    let access =
        QueryAccess::new(&[component_id::<Velocity>()], &[component_id::<Position>()]).unwrap();

    let mut seen = 0usize;
    w.views(&access, |mut view| {
        let vel = view.read::<Velocity>(0).to_vec();
        let pos = view.write::<Position>(0);
        assert_eq!(pos.len(), vel.len());
        for (p, v) in pos.iter_mut().zip(&vel) {
            p.p.x += v.v.x;
        }
        seen += view.len();
    });
    assert_eq!(seen, 8, "every entity must be visited exactly once");

    for (i, &e) in es.iter().enumerate() {
        assert_eq!(w.get::<Position>(e).unwrap().p.x, i as f64 + 1.0);
    }
}

#[test]
fn a_query_matches_supersets_not_only_exact_archetypes() {
    let (mut w, _) = world_with(8);
    // Position+Velocity matches both the {P,V} and {P,V,H} archetypes.
    let access = QueryAccess::new(
        &[component_id::<Position>(), component_id::<Velocity>()],
        &[],
    )
    .unwrap();
    let mut total = 0;
    let mut archetypes = 0;
    w.views(&access, |view| {
        total += view.len();
        archetypes += 1;
    });
    assert_eq!(total, 8);
    assert_eq!(archetypes, 2, "one match per archetype, not one per entity");

    // A component nobody has matches nothing.
    let none = QueryAccess::new(&[component_id::<Frozen>()], &[]).unwrap();
    let mut visits = 0;
    w.views(&none, |_| visits += 1);
    assert_eq!(visits, 0);
}

#[test]
fn entities_line_up_with_dense_rows() {
    let (mut w, _) = world_with(6);
    let access = QueryAccess::new(&[component_id::<Position>()], &[]).unwrap();
    w.views(&access, |view| {
        let entities = view.entities();
        let pos = view.read::<Position>(0);
        assert_eq!(entities.len(), pos.len());
        // The dense index is the row index in both, which is the whole
        // contract extract relies on (§4.2).
        for (row, &e) in entities.iter().enumerate() {
            assert_eq!(e.index() as f64, pos[row].p.x);
        }
    });
}

#[test]
fn aliasing_is_rejected_at_view_construction_naming_the_component() {
    let p = component_id::<Position>();
    let v = component_id::<Velocity>();

    // Two mutable borrows of one component.
    let err = QueryAccess::new(&[], &[p, v, p]).unwrap_err();
    assert!(err.to_string().contains("mutably more than once"), "{err}");

    // Mutable and shared at once.
    let err = QueryAccess::new(&[p], &[p]).unwrap_err();
    assert!(
        err.to_string().contains("both mutably and immutably"),
        "{err}"
    );

    // Reading the same component twice is harmless and allowed.
    let ok = QueryAccess::new(&[p, p], &[v]).unwrap();
    assert_eq!(ok.reads().len(), 1);
    assert_eq!(ok.writes().len(), 1);
}

#[test]
fn zero_sized_components_view_as_counted_rows() {
    let mut w = World::new();
    for _ in 0..5 {
        let e = w.spawn();
        w.insert(e, Frozen).unwrap();
    }
    let access = QueryAccess::new(&[component_id::<Frozen>()], &[]).unwrap();
    let mut rows = 0;
    w.views(&access, |view| {
        let markers = view.read::<Frozen>(0);
        assert_eq!(markers.len(), view.len());
        rows += markers.len();
    });
    assert_eq!(rows, 5);
}

#[test]
fn the_raw_views_carry_the_shape_the_boundary_needs() {
    // §4.2.2's payload: pointer, length, stride — one call per archetype, then
    // native iteration. This asserts the shape rather than the pointer value.
    let (mut w, _) = world_with(4);
    let access = QueryAccess::new(&[], &[component_id::<Position>()]).unwrap();
    let mut checked = 0;
    w.views(&access, |view| {
        for raw in view.raw_writes() {
            assert_eq!(raw.stride, size_of::<Position>());
            assert_eq!(raw.len, view.len());
            assert!(!raw.ptr.is_null());
            assert_eq!(raw.ptr as usize % align_of::<Position>(), 0);
            checked += 1;
        }
    });
    assert!(checked > 0);
}

#[test]
#[should_panic(expected = "wrong type for this access index")]
fn reading_a_column_at_the_wrong_type_panics_rather_than_reinterpreting() {
    let (mut w, _) = world_with(2);
    let access = QueryAccess::new(&[component_id::<Position>()], &[]).unwrap();
    w.views(&access, |view| {
        // Position is 24 bytes, Health is 8 — a silent reinterpretation here
        // would read three entities' worth of garbage as one.
        let _ = view.read::<Health>(0);
    });
}

// ---- §4.2.1: the fast path is an optimization, not the definition ------

#[test]
fn the_raw_fast_path_equals_the_explicit_protocol_encoding() {
    // The CI proof §4.2.1 requires. Both walks sort by entity id and differ
    // only in how a component's bytes are absorbed: raw memory versus the
    // derived field-by-field encoding.
    let (w, _) = world_with(64);
    assert_eq!(w.canonical_hash(), w.canonical_hash_via_protocol());
}

#[test]
// Volume, not aliasing: 300 ticks of blake3 over a growing world holds no
// `unsafe` Miri has not already checked in the eight tests above, and under
// interpretation it dominates the whole Miri leg's runtime.
#[cfg_attr(
    miri,
    ignore = "600 canonical hashes over a growing world; nothing new for Miri"
)]
fn the_two_paths_agree_across_a_churned_world() {
    let mut w = World::new();
    let mut live: Vec<Entity> = Vec::new();
    for tick in 0..300u32 {
        for k in 0..3 {
            let e = w.spawn();
            w.insert(
                e,
                Position {
                    p: sim::DVec3::new(f64::from(tick), f64::from(k), -1.5),
                },
            )
            .unwrap();
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
            if (tick + k) % 3 == 0 {
                w.insert(e, Frozen).unwrap();
            }
            live.push(e);
        }
        if tick % 4 == 0 && !live.is_empty() {
            let victim = live.swap_remove((tick as usize * 13) % live.len());
            w.despawn(victim);
        }
        assert_eq!(
            w.canonical_hash(),
            w.canonical_hash_via_protocol(),
            "fast path and protocol diverged at tick {tick}"
        );
    }
}

/// The untyped pair (§6 M15): a caller holding a `ComponentInfo` and no `T`
/// reaches the same bytes the typed pair does, and a write through it lands.
///
/// This is the whole of what the M15 inspector needs from storage — a component
/// id chosen at runtime has no type to name — and it is a *safe* spelling of
/// what `raw_reads`/`raw_writes` already handed over.
#[test]
fn untyped_columns_are_the_typed_ones_seen_as_bytes() {
    let (mut w, _) = world_with(6);
    // Copied out of the registry rather than borrowed: `views` wants `&mut
    // World` and a live `&ComponentInfo` would hold the shared borrow. The
    // inspector does the same, for the same reason.
    let (id, size, fields) = {
        let info = w.registry().get(component_id::<Health>()).unwrap();
        (info.id, info.size, info.fields)
    };
    let read = QueryAccess::new(&[id], &[]).unwrap();
    let mut seen = Vec::new();
    w.views_ref(&read, |view| {
        let bytes = view.read_bytes(0);
        assert_eq!(bytes.len(), view.len() * size);
        assert_eq!(
            bytemuck::cast_slice::<u8, Health>(bytes),
            view.read_of::<Health>()
        );
        // Field-addressed, which is how an inspector reads one: `hp` by its
        // offset out of the schema rather than by a Rust type.
        let hp = fields.iter().find(|f| f.name == "hp").unwrap();
        for row in 0..view.len() {
            let at = row * size + hp.offset;
            seen.push(u32::from_ne_bytes(bytes[at..at + 4].try_into().unwrap()));
        }
    });
    assert_eq!(seen, vec![0, 2, 4]);

    let write = QueryAccess::new(&[], &[id]).unwrap();
    let shield = *fields.iter().find(|f| f.name == "shield").unwrap();
    w.views(&write, |mut view| {
        let bytes = view.write_bytes(0);
        for row in 0..bytes.len() / size {
            let at = row * size + shield.offset;
            bytes[at..at + shield.size].copy_from_slice(&7u32.to_ne_bytes());
        }
    });
    let mut shields = Vec::new();
    w.views_ref(&read, |view| {
        shields.extend(view.read_of::<Health>().iter().map(|h| h.shield));
    });
    assert_eq!(
        shields,
        vec![7, 7, 7],
        "the untyped write reached the column"
    );
}

/// The take-once rule is the untyped column's too — the returned slice outlives
/// the `&mut self` that made it, so a second take would alias.
#[test]
#[should_panic(expected = "already taken")]
fn an_untyped_write_column_cannot_be_taken_twice() {
    let (mut w, _) = world_with(2);
    let access = QueryAccess::new(&[], &[component_id::<Position>()]).unwrap();
    w.views(&access, |mut view| {
        let _first = view.write_bytes(0);
        let _second = view.write_bytes(0);
    });
}
