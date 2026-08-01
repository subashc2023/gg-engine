//! Snapshot, restore and migration (§4.8, §4.2.2).
//!
//! The migration cases all use two Rust types sharing one **declared id** —
//! `PositionV1` and `PositionV2` are "the same component" as far as persisted
//! identity is concerned, which is exactly what a game crate does to itself when
//! an agent edits a struct and the dylib reloads. Two worlds keep them apart,
//! because a registry legitimately refuses to hold both at once.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::{Component, ComponentOutcome, SideTable, World};

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "position")]
#[repr(C)]
struct PositionV1 {
    x: f32,
    y: f32,
}

/// A field appended — the agent-edits-a-struct case.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "position")]
#[repr(C)]
struct PositionV2 {
    x: f32,
    y: f32,
    z: f32,
}

/// The same fields, reordered, with one retyped. Reordering must move data with
/// the names; retyping must not reinterpret bytes.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "position")]
#[repr(C)]
struct PositionShuffled {
    y: f32,
    x: u32,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "velocity")]
#[repr(C)]
struct Velocity {
    dx: f32,
}

#[derive(Default, SideTable)]
#[side_table(id = "notes")]
struct Notes {
    lines: Vec<u32>,
}

fn populated() -> World {
    let mut world = World::new();
    for i in 0..8u32 {
        let e = world.spawn();
        world
            .insert(
                e,
                PositionV1 {
                    x: i as f32,
                    y: 10.0 + i as f32,
                },
            )
            .unwrap();
        // Half the entities get a velocity, so more than one archetype exists
        // and the restore has to rebuild the set, not just one column.
        if i % 2 == 0 {
            world.insert(e, Velocity { dx: -(i as f32) }).unwrap();
        }
    }
    // A hole in the middle, so the freelist is non-empty and the allocator state
    // is something a restore can get wrong.
    let victim = world.entity_hashes()[3].0;
    world.despawn(victim);
    world
}

#[test]
fn a_restored_world_is_the_captured_world_bit_for_bit() {
    let world = populated();
    let before = world.canonical_hash();
    let snapshot = world.snapshot();

    let mut restored = World::new();
    restored.register::<PositionV1>().unwrap();
    restored.register::<Velocity>().unwrap();
    let report = restored.restore(&snapshot).unwrap();

    assert!(report.is_clean(), "no schema moved: {report:?}");
    assert_eq!(restored.canonical_hash(), before);
    assert_eq!(restored.len(), world.len());
}

#[test]
fn the_allocator_resumes_where_it_was_captured() {
    // Restoring live entities but not the freelist would hand out an id the old
    // world had already retired — a divergence exactly one spawn later, which is
    // the kind that survives every test that only checks the current state.
    let mut world = populated();
    let snapshot = world.snapshot();
    let next_in_original = world.spawn();

    let mut restored = World::new();
    restored.register::<PositionV1>().unwrap();
    restored.register::<Velocity>().unwrap();
    restored.restore(&snapshot).unwrap();

    assert_eq!(restored.spawn(), next_in_original);
}

#[test]
fn an_added_field_defaults_and_the_rest_survives() {
    let world = populated();
    let snapshot = world.snapshot();
    let kept: Vec<(gg_ecs::Entity, f32, f32)> = {
        let mut out = Vec::new();
        for (entity, _) in world.entity_hashes() {
            if let Some(p) = world.get::<PositionV1>(entity) {
                out.push((entity, p.x, p.y));
            }
        }
        out
    };

    let mut next = World::new();
    next.register::<PositionV2>().unwrap();
    next.register::<Velocity>().unwrap();
    let report = next.restore(&snapshot).unwrap();

    let outcome = &report
        .components
        .iter()
        .find(|(id, _)| id == "position")
        .unwrap()
        .1;
    match outcome {
        ComponentOutcome::Migrated {
            copied,
            defaulted,
            retyped,
        } => {
            assert_eq!(copied, &["x", "y"]);
            assert_eq!(defaulted, &["z"]);
            assert!(retyped.is_empty());
        }
        other => panic!("expected a migration, got {other:?}"),
    }
    for (entity, x, y) in kept {
        let p = next.get::<PositionV2>(entity).expect("entity survived");
        assert_eq!((p.x, p.y), (x, y), "old fields kept their values");
        assert_eq!(p.z, 0.0, "the new field defaulted");
    }
}

#[test]
fn a_reorder_moves_data_with_the_name_and_a_retype_defaults_it() {
    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, PositionV1 { x: 3.0, y: 7.0 }).unwrap();
    let snapshot = world.snapshot();

    let mut next = World::new();
    next.register::<PositionShuffled>().unwrap();
    let report = next.restore(&snapshot).unwrap();

    let p = next.get::<PositionShuffled>(e).unwrap();
    assert_eq!(p.y, 7.0, "y followed its name across the reorder");
    assert_eq!(
        p.x, 0,
        "x is an f32 in the snapshot and a u32 now — defaulted, never \
         reinterpreted"
    );
    match &report.components[0].1 {
        ComponentOutcome::Migrated {
            copied, retyped, ..
        } => {
            assert_eq!(copied, &["y"]);
            assert_eq!(retyped, &["x"]);
        }
        other => panic!("expected a migration, got {other:?}"),
    }
}

#[test]
fn a_component_the_new_build_dropped_takes_its_archetypes_with_it() {
    let world = populated();
    let live = world.len();
    let snapshot = world.snapshot();

    // The new build declares position only. Every entity now has the same
    // component set, so the two captured archetypes must collapse into one.
    let mut next = World::new();
    next.register::<PositionV1>().unwrap();
    let report = next.restore(&snapshot).unwrap();

    assert_eq!(next.len(), live, "entities survive a dropped component");
    assert_eq!(
        report
            .components
            .iter()
            .find(|(id, _)| id == "velocity")
            .unwrap()
            .1,
        ComponentOutcome::Dropped
    );
    for (entity, _) in next.entity_hashes() {
        assert!(next.get::<Velocity>(entity).is_none());
        assert!(next.get::<PositionV1>(entity).is_some());
    }
}

#[test]
fn a_snapshot_will_not_restore_into_a_different_host() {
    // Side tables cannot cross the reload boundary, so a mismatch means the
    // *host* changed — which a snapshot does not carry, and must not pretend to.
    let mut world = World::new();
    world.insert_side_table(Notes::default()).unwrap();
    let snapshot = world.snapshot();

    let mut bare = World::new();
    let err = bare.restore(&snapshot).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("notes"), "names the table: {message}");
}

#[test]
fn a_world_survives_the_trip_through_bytes() {
    // The rejuvenation path (§4.2.2): the world is handed to a *successor
    // process*, so it crosses as bytes and must arrive as the same world —
    // canonical hash and next allocated id included.
    let mut world = populated();
    let before = world.canonical_hash();
    let image = world.snapshot().encode();
    let next_in_original = world.spawn();

    let mut restored = World::new();
    restored.register::<PositionV1>().unwrap();
    restored.register::<Velocity>().unwrap();
    let report = restored
        .restore(&gg_ecs::Snapshot::decode(&image).unwrap())
        .unwrap();

    assert!(report.is_clean(), "no schema moved: {report:?}");
    assert_eq!(restored.canonical_hash(), before);
    assert_eq!(restored.spawn(), next_in_original);
}

#[test]
fn the_manifest_survives_the_trip_and_migration_still_runs() {
    // The bytes carry the *old build's* field layout, which is the half of a
    // snapshot that migration reads. A restart that dropped it would silently
    // default every field instead of moving it.
    let world = populated();
    let image = world.snapshot().encode();

    let mut restored = World::new();
    restored.register::<PositionV2>().unwrap();
    restored.register::<Velocity>().unwrap();
    let report = restored
        .restore(&gg_ecs::Snapshot::decode(&image).unwrap())
        .unwrap();

    let (_, outcome) = report
        .components
        .iter()
        .find(|(declared, _)| declared == "position")
        .unwrap();
    let ComponentOutcome::Migrated {
        copied, defaulted, ..
    } = outcome
    else {
        panic!("a field was appended: {outcome:?}");
    };
    assert_eq!(copied, &["x".to_owned(), "y".to_owned()]);
    assert_eq!(defaulted, &["z".to_owned()]);
}

#[test]
fn a_truncated_image_is_refused_at_every_length() {
    // The image is written by a process that is *exiting*, so a short file is a
    // disk that filled rather than an attack — but it still must not be trusted
    // into an out-of-bounds column read.
    let image = populated().snapshot().encode();
    for end in 0..image.len() {
        assert!(
            gg_ecs::Snapshot::decode(&image[..end]).is_err(),
            "{end} bytes of {} decoded as a whole world",
            image.len()
        );
    }
    assert!(gg_ecs::Snapshot::decode(&image).is_ok());
}
