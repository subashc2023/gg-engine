//! Asset-load micro-benchmarks (§4.11's fourth named hot path).
//!
//! No benchmark framework, for the reason `gg-ecs`'s query bench states: what
//! is worth trusting here is a *ratio* measured in one process (§3's budget,
//! §4.11's list).
//!
//! Two claims, both about §4.6's zero-copy story:
//!
//! 1. **Resolving a mesh does not copy it.** Every non-texture asset is a view
//!    over the pack's own bytes, so loading *and reading* all of them must cost
//!    less than one copy of the container. The reference copy is measured in
//!    the same run — `Pack::from_bytes` copies into an aligned buffer, which is
//!    exactly one pass over the bytes and therefore the honest unit.
//! 2. **A second load of the same asset is a lookup.** `Assets` is keyed by id;
//!    a repeat `load` that re-resolved would turn every shared material
//!    reference in a scene into real work.
//!
//! Neither is a wall-clock budget, and neither claims `from_bytes` is fast: it
//! is the *baseline*, not the subject. `Pack::open` mmaps instead and copies
//! nothing, which is what the shell uses (§4.6).
//!
//! It needs a pack, and packs are build output (§4.6) — so the bench builds one
//! in memory rather than reading `target/assets`, which may not exist. That also
//! keeps the numbers independent of what the demos happen to ship.
//!
//! Run: `cargo bench -p gg-assets` (or `cargo xtask bench`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use gg_assets::load::{Assets, asset};
use gg_assets::mesh::{self, Vertex};
use gg_assets::pack::{AssetId, AssetKind, Pack};
use gg_assets::write::PackWriter;

/// Meshes in the synthetic pack, and vertices in each. Sized so the whole thing
/// is a few MiB — a level's worth of directory, not a toy.
const MESHES: u32 = 256;
const VERTICES: u32 = 512;
/// Repeats per measurement; the minimum is reported.
const REPS: u32 = 25;

fn synthetic_pack() -> Vec<u8> {
    let mut writer = PackWriter::default();
    for m in 0..MESHES {
        let vertices: Vec<Vertex> = (0..VERTICES)
            .map(|v| {
                let f = (m * VERTICES + v) as f32 * 0.001;
                Vertex {
                    position: [f, f * 2.0, f * 3.0],
                    normal: [0.0, 1.0, 0.0],
                    uv: [f.fract(), (f * 2.0).fract()],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                }
            })
            .collect();
        let indices: Vec<u32> = (0..VERTICES).collect();
        writer
            .add(
                &format!("mesh-{m}"),
                AssetKind::Mesh,
                u64::from(m),
                mesh::encode(&vertices, &indices, AssetId::NONE),
            )
            .expect("stage mesh");
    }
    writer.finish().expect("lay out pack")
}

fn measure(mut body: impl FnMut()) -> Duration {
    let mut best = Duration::MAX;
    for _ in 0..REPS {
        let start = Instant::now();
        body();
        best = best.min(start.elapsed());
    }
    best
}

fn main() {
    let bytes = synthetic_pack();
    let names: Vec<String> = (0..MESHES).map(|m| format!("mesh-{m}")).collect();

    // One pass over the container: the unit everything below is stated in.
    let copy = measure(|| {
        let pack = Pack::from_bytes(&bytes).expect("pack");
        black_box(pack.entries().len());
    });

    // The same copy, plus resolving every mesh and reading its geometry — which
    // is what `gg-render`'s residency does before it uploads.
    let copy_and_read = measure(|| {
        let pack = Pack::from_bytes(&bytes).expect("pack");
        let mut assets = Assets::new(pack);
        let handles: Vec<_> = names
            .iter()
            .map(|name| assets.load::<asset::Mesh>(name).expect("load"))
            .collect();
        assets.pump_until_idle();
        for handle in &handles {
            let mesh = assets.mesh(handle).expect("mesh view");
            black_box(mesh.vertices().len() + mesh.indices().len());
        }
    });

    // Every mesh again, into an `Assets` that already holds them all.
    let pack = Pack::from_bytes(&bytes).expect("pack");
    let mut warm = Assets::new(pack);
    for name in &names {
        warm.load::<asset::Mesh>(name).expect("load");
    }
    warm.pump_until_idle();
    let tracked = warm.tracked();
    let reload = measure(|| {
        for name in &names {
            black_box(warm.load::<asset::Mesh>(name).expect("load").id());
        }
    });

    let us = |d: Duration| d.as_secs_f64() * 1e6;
    // Saturating: a read cheaper than the noise floor would otherwise wrap.
    let read = copy_and_read.saturating_sub(copy);
    let read_ratio = read.as_secs_f64() / copy.as_secs_f64();
    let reload_ratio = reload.as_secs_f64() / copy_and_read.as_secs_f64();
    let vertices = u64::from(MESHES) * u64::from(VERTICES);
    let ns_per_vertex = read.as_secs_f64() * 1e9 / vertices as f64;

    if std::env::args().any(|a| a == "--json") {
        println!(
            // One line: `xtask bench --record` takes the first line that parses
            // as JSON, so a pretty-printed record would archive as nothing.
            "{{\"meshes\":{MESHES},\"vertices_each\":{VERTICES},\"reps\":{REPS},\
             \"pack_bytes\":{},\"tracked\":{tracked},\
             \"us\":{{\"container_copy\":{:.3},\"copy_and_read\":{:.3},\"reload_warm\":{:.3}}},\
             \"ns_per_vertex\":{ns_per_vertex:.4},\
             \"ratios\":{{\"read_vs_copy\":{read_ratio:.5},\"reload_vs_read\":{reload_ratio:.5}}}}}",
            bytes.len(),
            us(copy),
            us(copy_and_read),
            us(reload)
        );
    } else {
        println!(
            "assets: {MESHES} meshes x {VERTICES} vertices ({} KiB pack), min of {REPS}:\n  \
             container copy {:>10.3} us   (the baseline: one pass over the bytes)\n  \
             copy + read    {:>10.3} us   ({tracked} assets tracked)\n  \
             reload warm    {:>10.3} us\n  \
             read/copy {read_ratio:.4} ({ns_per_vertex:.4} ns/vertex)   \
             reload/read {reload_ratio:.4}",
            bytes.len() / 1024,
            us(copy),
            us(copy_and_read),
            us(reload)
        );
    }

    let mut failures = Vec::new();
    // Resolving and reading every mesh must cost less than copying the pack
    // once. If it ever exceeds that, something is copying geometry that §4.6
    // says is a view.
    if read_ratio > 1.0 {
        failures.push(format!(
            "resolving and reading every mesh costs {read_ratio:.3} of one copy of the              container — non-texture assets are meant to be views over the pack's bytes (§4.6)"
        ));
    }
    // A cache hit is a lookup.
    if reload_ratio > 0.25 {
        failures.push(format!(
            "re-loading resident assets costs {reload_ratio:.3} of the cold pass — `Assets` is \
             keyed by id and a repeat is meant to be a lookup, not a re-resolve"
        ));
    }
    assert!(
        failures.is_empty(),
        "{}",
        failures.join(
            "
"
        )
    );
}
