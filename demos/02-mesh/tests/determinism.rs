//! The replay determinism gate (§5.6), as ordinary tests.
//!
//! Ordinary tests on purpose: the third architecture reaches this file by
//! adding `-p demo-02-mesh` to a `cargo nextest --target aarch64-…` line, and
//! nothing that needs an `xtask` binary can run under qemu-user. Which legs
//! this covers:
//!
//! | §5.6 | here |
//! |---|---|
//! | (a) twice on one runner | [`one_replay_run_twice_on_this_runner_agrees`] |
//! | (b) once per architecture | the curated baseline, checked in, compared by every leg |
//! | (c) across tiers | the same test under `--cargo-profile dist` (`xtask ci`) |
//! | (d) FP baseline per profile | `gg-math`'s, since M0B |
//! | (e) across link modes | dormant until the static path activates (M5) |
//!
//! Nothing here needs a GPU, a window, or a display: this is sim state and
//! hashes, which is exactly why the gate can be everywhere.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use demo_02_mesh::gate;
use demo_02_mesh::sim::{self, Plant};
use gg_input::Replay;

fn curated() -> Replay {
    // The checked-in file, not a freshly generated one: the point is that the
    // *artifact* reproduces, and a test that regenerated its own input would
    // pass happily the day the generator changed.
    let bytes = gate::read(&gate::replay_path(gate::CURATED)).expect(
        "the curated replay is missing — `cargo xtask replay --bless` writes it, and its \
         absence means the gate is checking nothing",
    );
    Replay::decode(&bytes).expect("the curated replay does not decode")
}

fn baseline(name: &str) -> Vec<(u64, u128)> {
    let text = gate::read(&gate::baseline_path(name)).expect("baseline missing");
    gate::parse_baseline(&String::from_utf8(text).expect("baseline is not UTF-8")).unwrap()
}

#[test]
fn the_curated_replay_reproduces_its_checked_in_hash_sequence() {
    // §5.6b on this architecture. The other two legs run this same test against
    // this same file; "identical across architectures" is that, three times.
    let replay = curated();
    assert_eq!(replay.ticks(), gate::CURATED_TICKS);
    assert_eq!(
        replay.meta().contract,
        gg_math::DETERMINISM_CONTRACT,
        "the replay was recorded under a different Determinism Contract (§4.2.1)"
    );
    let sequence = sim::hash_sequence(&replay).unwrap();
    let expected = baseline(gate::CURATED);
    assert_eq!(expected.len(), sequence.len(), "baseline length");

    if let Some(divergence) = gate::compare(&sequence, &expected) {
        // Write the fresh sequence beside the baseline rather than replacing
        // it — the same discipline the FP baseline keeps (§6 M0A spike 3):
        // regenerating to make CI green is the move that turns a gate into a
        // rubber stamp.
        let fresh = std::env::temp_dir().join("demo02-curated.hashes.actual");
        let _ = std::fs::write(
            &fresh,
            gate::encode_baseline(&gate::sequence_entries(&sequence)),
        );
        panic!(
            "{divergence}\nfresh sequence written to {}\nIf this diff is intended, \
             `cargo xtask replay --bless` — and the diff belongs in the review.",
            fresh.display()
        );
    }
}

#[test]
fn one_replay_run_twice_on_this_runner_agrees() {
    // §5.6a: catches scheduling and hash-order nondeterminism, which a single
    // run compared to a file cannot — a stable-but-wrong order matches its own
    // baseline forever.
    let replay = curated();
    let first = sim::hash_sequence(&replay).unwrap();
    let second = sim::hash_sequence(&replay).unwrap();
    assert_eq!(
        gate::compare_runs(&replay, &first, &second, None).unwrap(),
        None
    );
}

#[test]
fn the_protocol_and_fast_path_hashes_agree_across_the_whole_replay() {
    // §4.2.1: the raw-column fast path is an *optimization of* the protocol
    // encoding, and a replay is the widest state this repo can point at both.
    let replay = curated();
    let (sim, _) = sim::run(&replay, gate::CURATED_TICKS, None).unwrap();
    assert_eq!(
        sim.world.canonical_hash(),
        sim.world.canonical_hash_via_protocol()
    );
}

#[test]
fn a_planted_divergence_is_caught_and_reported_with_its_frame_and_entity() {
    // The M4B exit criterion. The plant is one shoved cube at a known tick —
    // small, late, and on an entity that only exists because the replay spawned
    // it, so nothing about finding it is arranged in advance.
    let replay = curated();
    let plant = Plant {
        tick: 317,
        nudge: 1.0e-9,
    };
    let clean = sim::hash_sequence(&replay).unwrap();
    let (_, planted) = sim::run(&replay, replay.ticks(), Some(plant)).unwrap();

    let divergence = gate::compare_runs(&replay, &clean, &planted, Some(plant))
        .unwrap()
        .expect("a planted divergence that the gate does not catch is not a gate");
    assert_eq!(divergence.tick, plant.tick, "reported the wrong frame");

    let entity = divergence.entity.expect("the entity was not localized");
    // It must be the cube that was shoved, not merely *an* entity: the world
    // holds a camera and several cubes at that tick.
    let world = sim::world_at(&replay, plant.tick + 1, None).unwrap();
    let cube = world.get::<sim::Cube>(entity).expect("named a non-cube");
    assert!(cube.order > 0, "named the original cube, not the newest");

    // ...and a nudge below the last bit of the position changes nothing, which
    // is what makes the gate a measurement rather than a coin flip.
    let (_, unplanted) = sim::run(
        &replay,
        replay.ticks(),
        Some(Plant {
            tick: 317,
            nudge: 0.0,
        }),
    )
    .unwrap();
    assert_eq!(clean, unplanted);
}

/// The locator merges two index-ascending lists, so its comparator must be
/// index-major too: `to_bits()` is generation-major, and keyed on it the walk
/// named an entity from the *other* world whenever a recycled index sat at the
/// first difference.
#[test]
fn the_locator_names_the_recycled_entity_not_its_neighbour() {
    let mut a = gg_ecs::World::new();
    let (_, _) = (a.spawn(), a.spawn());
    let recycled = a.spawn();
    a.despawn(recycled);
    let recycled = a.spawn(); // index 2 again, generation 3

    let mut b = gg_ecs::World::new();
    let (_, _) = (b.spawn(), b.spawn());
    let gone = b.spawn();
    b.spawn(); // index 3 — the neighbour the generation-major walk misnamed
    b.despawn(gone); // index 2 lives only in `a`

    assert_eq!(gate::locate_entity(&a, &b), Some(recycled));
}

#[test]
fn every_chaos_seed_matches_its_checked_in_checkpoints() {
    // §5.11 v0. Generated streams, so the file holds checkpoints rather than
    // 4,800 lines of hash; the curated replay is where tick-exact reporting
    // lives, and a failing seed is promoted to one before it is fixed.
    let expected = baseline("chaos");
    let mut entries = Vec::new();
    for &seed in gate::CHAOS_SEEDS {
        let replay = gate::chaos_replay(seed, gate::CHAOS_TICKS);
        let sequence = sim::hash_sequence(&replay).unwrap();
        for (tick, hash) in gate::checkpoints(&sequence) {
            entries.push((seed, tick, hash.get()));
        }
    }
    assert_eq!(
        entries.len(),
        expected.len(),
        "the chaos baseline covers a different number of checkpoints"
    );
    for ((seed, tick, mine), (_, theirs)) in entries.iter().zip(&expected) {
        assert_eq!(
            mine, theirs,
            "chaos seed {seed} diverges at tick {tick}: {mine:032x} here, {theirs:032x} expected"
        );
    }
}

#[test]
fn the_chaos_streams_actually_churn() {
    // A chaos suite that generated a quiet stream would pass forever while
    // testing nothing, so the generator's *coverage* is asserted rather than
    // assumed: cubes must come and go, and the camera must move.
    let replay = gate::chaos_replay(CHAOS_WITNESS, gate::CHAOS_TICKS);
    let (sim, hashes) = sim::run(&replay, gate::CHAOS_TICKS, None).unwrap();
    assert!(
        hashes
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            > 500,
        "the stream barely moved the world"
    );
    let cubes = sim.cube_count().unwrap();
    assert!(cubes > 1, "no cube survived the churn: {cubes}");
    assert_ne!(
        sim.camera().unwrap().position,
        sim::CameraRig::GOLDEN.position
    );
    assert_ne!(sim.camera().unwrap().yaw, sim::CameraRig::GOLDEN.yaw);
}

/// One seed's worth of coverage assertions; the rest are gated by hash alone.
const CHAOS_WITNESS: u64 = 8;
