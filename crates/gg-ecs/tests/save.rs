//! The durable save and its load policy (§6 M14).
//!
//! Same idiom as `snapshot.rs`: two Rust types sharing one **declared id** are
//! "the same component" across a version boundary, and two worlds keep them
//! apart because one registry legitimately refuses to hold both. What differs is
//! the verdict — `snapshot.rs` asserts that a reload *migrates through* loss,
//! this file asserts that a save *refuses* it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::{Component, Query, Save, SaveError, SideTable, World};

/// The writing build's component.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo.progress")]
#[repr(C)]
struct ProgressV1 {
    score: u64,
    level: u32,
    _pad: u32,
}

/// Version N+1: a field appended, one reordered, none lost. The whole of what a
/// save is allowed to cross.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo.progress")]
#[repr(C)]
struct ProgressV2 {
    level: u32,
    _pad: u32,
    score: u64,
    deaths: u64,
}

/// `score` kept its name and changed its meaning. Bit-identical, semantically
/// nonsense — the case the save policy exists to catch.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo.progress")]
#[repr(C)]
struct ProgressRetyped {
    score: f64,
    level: u32,
    _pad: u32,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo.inventory")]
#[repr(C)]
struct Inventory {
    coins: u32,
}

#[derive(Default, SideTable)]
#[side_table(id = "notes")]
struct Notes {
    lines: Vec<u32>,
}

/// What the shell passes as provenance: the game dylib's content hash (§4.2.2).
const PROVENANCE: u128 = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210;
const TICK: u64 = 4200;

fn played() -> World {
    let mut world = World::new();
    for i in 0..8u32 {
        let e = world.spawn();
        world
            .insert(
                e,
                ProgressV1 {
                    score: u64::from(i) * 100,
                    level: i,
                    _pad: 0,
                },
            )
            .unwrap();
        // Two archetypes, so a load rebuilds a set rather than one column.
        if i % 2 == 0 {
            world.insert(e, Inventory { coins: i * 3 }).unwrap();
        }
    }
    let victim = world.entity_hashes()[5].0;
    world.despawn(victim);
    world
}

fn saved() -> Vec<u8> {
    Save::new(played().snapshot(), TICK, PROVENANCE).encode()
}

/// A world with V2 registered and nothing in it — the next build, before it
/// loads anything.
fn next_build() -> World {
    let mut world = World::new();
    world.register::<ProgressV2>().unwrap();
    world.register::<Inventory>().unwrap();
    world
}

#[test]
fn a_save_survives_the_trip_through_bytes_and_the_world_comes_back_whole() {
    let world = played();
    let before = world.canonical_hash();

    let save = Save::decode(&saved()).unwrap();
    assert_eq!(save.tick(), TICK);
    assert_eq!(save.provenance(), PROVENANCE);

    let mut loaded = World::new();
    loaded.register::<ProgressV1>().unwrap();
    loaded.register::<Inventory>().unwrap();
    let report = loaded.load(&save).unwrap();

    assert!(report.is_clean(), "no schema moved: {report:?}");
    assert_eq!(loaded.canonical_hash(), before);
    assert_eq!(loaded.len(), world.len());
}

#[test]
fn a_save_written_against_version_n_migrates_forward_into_n_plus_one() {
    let save = Save::decode(&saved()).unwrap();
    let mut world = next_build();
    let report = world.load(&save).unwrap();

    assert!(!report.is_clean(), "the schema moved: {report:?}");
    // The player's numbers arrived where the *new* build keeps them, and the
    // field the new build added is zero rather than absent.
    let mut seen = Vec::new();
    let q = Query::<&ProgressV2>::new().unwrap();
    world.each(&q, |_, p| seen.push((p.level, p.score, p.deaths)));
    seen.sort_unstable();
    assert_eq!(
        seen,
        vec![
            (0, 0, 0),
            (1, 100, 0),
            (2, 200, 0),
            (3, 300, 0),
            (4, 400, 0),
            (6, 600, 0),
            (7, 700, 0)
        ]
    );
}

#[test]
fn a_component_this_build_no_longer_declares_is_refused_by_its_name() {
    // The reload path drops it and says so in a report; a save may not, because
    // the data is the player's. Same image, two verdicts.
    let save = Save::decode(&saved()).unwrap();
    let mut world = World::new();
    world.register::<ProgressV2>().unwrap();

    let refusal = world.load(&save).unwrap_err();
    assert!(
        matches!(&refusal, SaveError::Dropped { declared } if declared == "demo.inventory"),
        "{refusal}"
    );
    assert!(refusal.to_string().contains("demo.inventory"));
    // The same image through `restore` is accepted, which is what makes the
    // refusal above a *policy* rather than a limitation.
    assert!(world.restore(save.snapshot()).is_ok());
}

#[test]
fn a_field_that_kept_its_name_and_changed_its_type_is_refused_by_field() {
    let save = Save::decode(&saved()).unwrap();
    let mut world = World::new();
    world.register::<ProgressRetyped>().unwrap();
    world.register::<Inventory>().unwrap();

    let refusal = world.load(&save).unwrap_err();
    assert!(
        matches!(
            &refusal,
            SaveError::Retyped { declared, field } if declared == "demo.progress" && field == "score"
        ),
        "{refusal}"
    );
}

#[test]
fn a_refused_save_leaves_the_world_the_player_had() {
    // The refusal is reached through `dry_run`, so nothing is torn down. A load
    // that emptied the world before deciding it could not fill it would be worse
    // than one that loaded wrong.
    let save = Save::decode(&saved()).unwrap();
    let mut world = World::new();
    world.register::<ProgressRetyped>().unwrap();
    world.register::<Inventory>().unwrap();
    let e = world.spawn();
    world.insert(e, Inventory { coins: 99 }).unwrap();
    let before = world.canonical_hash();

    assert!(world.load(&save).is_err());
    assert_eq!(world.canonical_hash(), before);
    assert_eq!(world.len(), 1);
}

#[test]
fn a_save_from_a_newer_game_is_refused_by_the_component_that_build_added() {
    // The claim the format makes by carrying no version number: "written by a
    // build you do not understand" is not a word in the header, it is a name in
    // the manifest. Here the *older* build is the one with no `Inventory`.
    let save = Save::decode(&saved()).unwrap();
    let mut older = World::new();
    older.register::<ProgressV1>().unwrap();

    let refusal = older.load(&save).unwrap_err().to_string();
    assert!(refusal.contains("demo.inventory"), "{refusal}");
}

#[test]
fn one_flipped_bit_anywhere_is_caught_before_a_length_is_believed() {
    // A snapshot handoff carries no checksum by design (§4.2.2) — it is written
    // by an exiting process to a successor that starts immediately. A save is on
    // a disk, so this is the difference the container exists for.
    let clean = saved();
    for at in (0..clean.len()).step_by(7) {
        let mut damaged = clean.clone();
        damaged[at] ^= 0x40;
        let refusal = Save::decode(&damaged).unwrap_err();
        assert!(
            matches!(refusal, SaveError::Checksum { .. }),
            "byte {at} flipped and the reader said {refusal}"
        );
    }
}

#[test]
fn a_truncated_save_is_refused_at_every_length() {
    let clean = saved();
    for len in 0..clean.len() {
        let refusal = Save::decode(&clean[..len]);
        assert!(refusal.is_err(), "{len} of {} bytes loaded", clean.len());
    }
    assert!(Save::decode(&clean).is_ok());
}

#[test]
fn a_file_that_is_not_a_save_is_refused_as_one() {
    // Checksummed first, so the magic check has to be reachable with a *valid*
    // digest over the wrong content — an empty snapshot image is exactly that.
    let mut impostor = b"GGXX".to_vec();
    impostor.extend_from_slice(&[0u8; 32]);
    impostor.extend_from_slice(blake3::hash(&impostor).as_bytes());
    let refusal = Save::decode(&impostor).unwrap_err();
    assert!(matches!(refusal, SaveError::Magic { .. }), "{refusal}");
}

#[test]
fn a_save_will_not_load_into_a_differently_equipped_host() {
    // Side-table contents do not ride along (P2, see the module docs), so the
    // ids are the only thing standing between a load and a world quietly short
    // of state.
    let save = Save::decode(&saved()).unwrap();
    let mut world = next_build();
    world.insert_side_table(Notes::default()).unwrap();

    let refusal = world.load(&save).unwrap_err();
    assert!(refusal.to_string().contains("notes"), "{refusal}");
}
