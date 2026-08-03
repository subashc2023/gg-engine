//! §6 M14's last exit row: **demo 05's ten thousand entities save and load
//! inside one frame budget** (§4.11).
//!
//! The world is built from the demo's own layout functions — the same ones
//! `gg-golden` takes it for — so what is timed is ten thousand parented objects
//! in the archetypes the game produces, not a synthetic column of the right
//! length. Through the layout functions rather than through `gg_game_*`, because
//! `[profile.bench]` inherits `instrumented`'s thin LTO and that internalizes an
//! rlib's `no_mangle` symbols; the demo's tests reach the table under the test
//! profile, where it survives.
//!
//! No benchmark framework, for `gg-ecs`'s reason (§3, §4.11): what is asserted
//! here is an *absolute* budget rather than a ratio, but the budget is a frame
//! and a frame is three orders of magnitude above timer noise. The minimum of
//! several repeats is reported, because noise on a desktop is one-sided.
//!
//! Run: `cargo bench -p demo-05-many` (or `cargo xtask bench`, the nightly tier).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use demo_05_many::{
    HUBS, Hub, MESHES, PER_HUB, SPIN_PER_TICK, START_POSITION, SUN_COLOR, SUN_DIRECTION,
    SUN_INTENSITY, child_placement, hub_position,
};
use gg_ecs::boundary::{Light, Model, Node};
use gg_ecs::{Save, World};
use gg_math::sim;

/// One frame at the shell's tick rate. The criterion is "inside one frame
/// budget"; 60 Hz is that budget, spelled out here because a demo crate cannot
/// reach `gg_core::DEFAULT_TICK_HZ` across §3's deny pin.
const FRAME_MS: f64 = 1000.0 / 60.0;

/// Repeats. Few, because each one moves megabytes and the minimum settles fast.
const REPS: u32 = 9;

/// The demo's world after `bootstrap`: a hundred hubs, ten thousand children,
/// the observer and the sun — every position from the demo's own functions, so
/// the columns hold what the game would have put in them.
fn populated() -> World {
    let mut world = World::new();
    let observer = world.spawn();
    world
        .insert(
            observer,
            demo_05_many::Observer {
                position: START_POSITION,
                yaw: 0.0,
                pitch: -0.12,
                frozen: 0,
                _pad: 0,
            },
        )
        .unwrap();
    let sun = world.spawn();
    world
        .insert(sun, Light::sun(SUN_DIRECTION, SUN_COLOR, SUN_INTENSITY))
        .unwrap();

    for index in 0..HUBS {
        let hub = world.spawn();
        world
            .insert(hub, Model::at(MESHES[0], hub_position(index)))
            .unwrap();
        world
            .insert(
                hub,
                Hub {
                    angle: 0.0,
                    spin: if index % 2 == 0 { SPIN_PER_TICK } else { 0.0 },
                },
            )
            .unwrap();
        for slot in 0..PER_HUB {
            let (offset, mesh) = child_placement(slot);
            let child = world.spawn();
            world.insert(child, Node::at(hub, offset)).unwrap();
            world
                .insert(child, Model::at(MESHES[mesh], sim::DVec3::ZERO))
                .unwrap();
        }
    }
    world
}

fn measure(name: &str, mut body: impl FnMut()) -> Duration {
    body(); // warm the allocator and the caches
    let mut best = Duration::MAX;
    for _ in 0..REPS {
        let start = Instant::now();
        body();
        best = best.min(start.elapsed());
    }
    let ms = best.as_secs_f64() * 1e3;
    println!(
        "{name:<28} {ms:>8.3} ms   {:>5.1}% of a 60 Hz frame",
        ms / FRAME_MS * 100.0
    );
    best
}

fn main() {
    let world = populated();
    let entities = world.len();
    let bytes = Save::new(world.snapshot(), 0, 0).encode().len();
    println!(
        "demo 05: {entities} entities, {:.2} MiB of save\n",
        bytes as f64 / (1024.0 * 1024.0)
    );
    assert!(
        entities >= 10_000,
        "{entities} entities is not the ten thousand §6 M10 built and §6 M14 gates"
    );

    let saving = measure("save (snapshot + encode)", || {
        black_box(Save::new(world.snapshot(), 0, 0).encode().len());
    });

    // The destination is built once and reloaded into, which is what an editor's
    // load does: the components are already registered and only the rows move.
    let file = Save::new(world.snapshot(), 0, 0).encode();
    let mut into = populated();
    let loading = measure("load (decode + restore)", || {
        let save = Save::decode(&file).unwrap();
        black_box(into.load(&save).unwrap().entities);
    });

    let (save_ms, load_ms) = (saving.as_secs_f64() * 1e3, loading.as_secs_f64() * 1e3);
    println!(
        "\nround trip {:.3} ms against a {FRAME_MS:.3} ms frame",
        save_ms + load_ms
    );

    // Emitted before the assertions so a red run still archives how far off it
    // was — the reader wants the number, not just the verdict (§4.11).
    if std::env::args().any(|a| a == "--json") {
        println!(
            "{{\"entities\":{entities},\"save_bytes\":{bytes},\"reps\":{REPS},\
             \"ms\":{{\"save\":{save_ms:.4},\"load\":{load_ms:.4},\"frame_budget\":{FRAME_MS:.4}}}}}"
        );
    }

    // Each side separately, not the round trip: an editor pressing stop pays the
    // load alone, and a game autosaving pays the save alone. A budget on the sum
    // would let one of them quietly eat the whole frame.
    assert!(
        save_ms < FRAME_MS,
        "saving {entities} entities took {save_ms:.3} ms against a {FRAME_MS:.3} ms frame (§6 M14)"
    );
    assert!(
        load_ms < FRAME_MS,
        "loading {entities} entities took {load_ms:.3} ms against a {FRAME_MS:.3} ms frame (§6 M14)"
    );
}
