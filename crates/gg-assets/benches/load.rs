//! Asset-load micro-benchmarks (§4.11's fourth named hot path).
//!
//! No benchmark framework, for the reason `gg-ecs`'s query bench states: what
//! is worth trusting here is a *ratio* measured in one process (§3's budget,
//! §4.11's list).
//!
//! Two claims, both about §4.6's zero-copy story:
//!
//! 1. **Resolving a mesh does not copy it.** Every non-texture asset is a view
//!    over the pack's own bytes, so resolving and reading one costs what the
//!    asset *count* costs and nothing per byte. [`VERTICES_FAT`] is the test:
//!    four times the geometry, the same 256 assets, and the same time.
//! 2. **A second load of the same asset is a lookup.** `Assets` is keyed by id;
//!    a repeat `load` that re-resolved would turn every shared material
//!    reference in a scene into real work.
//!
//! Absolute microseconds are printed for the archive; only the ratios assert.
//!
//! # The container copy is a number, not a gate
//!
//! Until M17 claim 1 was stated as "resolving every mesh must cost less than
//! one copy of the container", against `Pack::from_bytes`. That gate measured
//! the host's allocator. `from_bytes` allocates 6.7 MiB per rep, and Windows
//! serves that from fresh pages every time (~6.2 GB/s) while glibc returns a
//! warm arena (~28 GB/s) — so the *same* resolve, ~250 µs on both, read 0.23
//! on the desk and 1.36–1.61 on the WSL lane, and the lane failed a 1.0 bound
//! on its first run of this leg. A 6x spread in a denominator the header
//! itself called "not the subject" is not a property of the engine. It is
//! still printed, because what a resolve costs against one pass over the bytes
//! is worth knowing; it just cannot be asserted across hosts.
//!
//! Claim 1's honest form needs no cross-host baseline at all: it is a leg
//! against *itself* at two sizes, so the allocator, the page size and the
//! memcpy rate all cancel.
//!
//! # What is inside the clock
//!
//! `Pack::from_bytes` and `Assets::new` are built before it. Both are per-rep
//! setup — the second spawns the loader's worker threads, which would price a
//! thread pool the way the extract bench's 10k world priced rayon's. The timed
//! region is exactly what claim 1 names: `load` every mesh, pump, and read
//! each one's vertex and index runs.
//!
//! Gates are paired medians — one pass of each leg per rep, quotient taken
//! within the rep — for the reason `gg-extract`'s bench header gives at length.
//!
//! Both bounds were made to fail by name before being trusted, each against the
//! regression it names: a `Mesh::read` that copied its two runs into `Vec`s
//! instead of borrowing them put the scaling ratio at 3.60–3.81, and a
//! `handle_for` with its cache-hit return removed put the reload ratio at
//! 0.376–0.453 against a 0.12–0.20 band.
//!
//! The first of those is the case for having replaced the old gate rather than
//! retuned it: with that copying `Mesh::read` in place, `read/copy` reads 0.42
//! on the desk — under the 1.0 bound that stood here until M17. A resolve that
//! copied every mesh would have shipped, and only the lane's faster allocator
//! would have caught it.
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
/// The scaling leg: four times the geometry under the same asset count. A view
/// costs per asset, so this leg must cost what the small one does; a resolve
/// that copied would cost four times its geometry share. Four and not two
/// because the per-asset floor is real work and would halve any smaller signal.
const VERTICES_FAT: u32 = VERTICES * 4;
/// Repeats per measurement; absolutes report the minimum, gates the paired
/// median over these (see [`ratio`]).
const REPS: u32 = 25;
/// Passes over the name list inside one timed reload, divided back out. Eight
/// puts that leg on the same order as the resolve it is quoted against.
const RELOAD_PASSES: u32 = 8;
/// Passes before any of them counts — caches, the allocator's arena, and the
/// loader's workers, which park between reps and pay a wake-up on the first.
const WARMUP: u32 = 3;

fn synthetic_pack(vertices_each: u32) -> Vec<u8> {
    let mut writer = PackWriter::default();
    for m in 0..MESHES {
        let vertices: Vec<Vertex> = (0..vertices_each)
            .map(|v| {
                let f = (m * vertices_each + v) as f32 * 0.001;
                Vertex {
                    position: [f, f * 2.0, f * 3.0],
                    normal: [0.0, 1.0, 0.0],
                    uv: [f.fract(), (f * 2.0).fract()],
                    tangent: [1.0, 0.0, 0.0, 1.0],
                }
            })
            .collect();
        let indices: Vec<u32> = (0..vertices_each).collect();
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

/// One pass over the container: `from_bytes` copies into an aligned buffer.
/// Reported, never asserted — see the module header.
fn copy_once(bytes: &[u8]) -> Duration {
    let start = Instant::now();
    let pack = Pack::from_bytes(bytes).expect("pack");
    let elapsed = start.elapsed();
    black_box(pack.entries().len());
    elapsed
}

/// One cold resolve of every mesh, timed — what `gg-render`'s residency does
/// before it uploads. The pack and the loader are built before the clock; the
/// handle vector is allocated before it too, so a push cannot land a realloc
/// inside the measurement.
fn resolve_once(bytes: &[u8], names: &[String]) -> (Duration, usize) {
    let pack = Pack::from_bytes(bytes).expect("pack");
    let mut assets = Assets::new(pack);
    let mut handles = Vec::with_capacity(names.len());

    let start = Instant::now();
    for name in names {
        handles.push(assets.load::<asset::Mesh>(name).expect("load"));
    }
    assets.pump_until_idle();
    let mut touched = 0;
    for handle in &handles {
        let view = assets.mesh(handle).expect("mesh view");
        touched += view.vertices().len() + view.indices().len();
    }
    let elapsed = start.elapsed();
    black_box(touched);
    (elapsed, assets.tracked())
}

/// Every mesh again, into an `Assets` that already holds them all — [`RELOAD_PASSES`]
/// times over, divided back down, so one pass is not what the clock has to
/// resolve. One pass is ~13 µs beside a ~60 µs resolve and lands right after two
/// legs that allocate 34 MiB between them; measured that way the ratio spanned
/// 0.25–0.40 across the hosts, which is the width of the 1.6x signal it has to
/// discriminate. This is the extract bench's lesson in the small: a leg shorter
/// than the disturbance around it measures the disturbance.
fn reload_once(warm: &mut Assets, names: &[String]) -> Duration {
    let start = Instant::now();
    for _ in 0..RELOAD_PASSES {
        for name in names {
            black_box(warm.load::<asset::Mesh>(name).expect("load").id());
        }
    }
    start.elapsed() / RELOAD_PASSES
}

/// The four legs interleaved — one pass of each per rep, every sample kept.
/// Returns `[copy, resolve, resolve_fat, reload]`.
fn sweep(small: &[u8], fat: &[u8], names: &[String], warm: &mut Assets) -> [Vec<Duration>; 4] {
    let mut samples: [Vec<Duration>; 4] = Default::default();
    for rep in 0..WARMUP + REPS {
        let pass = [
            copy_once(small),
            resolve_once(small, names).0,
            resolve_once(fat, names).0,
            reload_once(warm, names),
        ];
        if rep >= WARMUP {
            for (leg, pass) in samples.iter_mut().zip(pass) {
                leg.push(pass);
            }
        }
    }
    samples
}

/// The floor — right for an absolute, which is all it is used for here.
fn best(leg: &[Duration]) -> Duration {
    leg.iter().copied().min().unwrap()
}

/// Median of the per-rep quotients, paired within the rep.
fn ratio(num: &[Duration], den: &[Duration]) -> f64 {
    let mut quotients: Vec<f64> = num
        .iter()
        .zip(den)
        .map(|(n, d)| n.as_secs_f64() / d.as_secs_f64())
        .collect();
    quotients.sort_by(f64::total_cmp);
    quotients[quotients.len() / 2]
}

fn main() {
    let small = synthetic_pack(VERTICES);
    let fat = synthetic_pack(VERTICES_FAT);
    let names: Vec<String> = (0..MESHES).map(|m| format!("mesh-{m}")).collect();

    // Untimed, and before the sweep: how many assets a resolve leaves resident
    // is a statement about the dataset, and checking it inside the clock would
    // price the check.
    let (_, tracked) = resolve_once(&small, &names);

    let mut warm = Assets::new(Pack::from_bytes(&small).expect("pack"));
    for name in &names {
        warm.load::<asset::Mesh>(name).expect("load");
    }
    warm.pump_until_idle();

    let [copy, resolve, resolve_fat, reload] = sweep(&small, &fat, &names, &mut warm);

    let scale_ratio = ratio(&resolve_fat, &resolve);
    let reload_ratio = ratio(&reload, &resolve);
    let read_ratio = ratio(&resolve, &copy);
    let (copy, resolve, resolve_fat, reload) = (
        best(&copy),
        best(&resolve),
        best(&resolve_fat),
        best(&reload),
    );
    let us = |d: Duration| d.as_secs_f64() * 1e6;
    // Per *asset*, not per vertex: the claim under test is that there is no
    // per-vertex cost, so reporting one would beg the question.
    let ns_per_asset = resolve.as_secs_f64() * 1e9 / f64::from(MESHES);

    if std::env::args().any(|a| a == "--json") {
        println!(
            // One line: `xtask bench --record` takes the last line that parses
            // as JSON, so a pretty-printed record would archive as nothing.
            "{{\"meshes\":{MESHES},\"vertices_each\":{VERTICES},\"vertices_fat\":{VERTICES_FAT},\
             \"reps\":{REPS},\"pack_bytes\":{},\"fat_pack_bytes\":{},\"tracked\":{tracked},\
             \"us\":{{\"container_copy\":{:.3},\"resolve\":{:.3},\"resolve_fat\":{:.3},\
             \"reload_warm\":{:.3}}},\"ns_per_asset\":{ns_per_asset:.1},\
             \"ratios\":{{\"geometry_scaling\":{scale_ratio:.5},\
             \"reload_vs_resolve\":{reload_ratio:.5},\"read_vs_copy\":{read_ratio:.5}}}}}",
            small.len(),
            fat.len(),
            us(copy),
            us(resolve),
            us(resolve_fat),
            us(reload)
        );
    } else {
        println!(
            "assets: {MESHES} meshes x {VERTICES} vertices ({} KiB pack), paired median of \
             {REPS}:\n  \
             container copy {:>10.3} us   (reported only: it prices the host allocator)\n  \
             resolve + read {:>10.3} us   ({tracked} assets tracked, {ns_per_asset:.0} ns/asset)\n  \
             resolve 4x geo {:>10.3} us   (the same {MESHES} assets, four times the geometry)\n  \
             reload warm    {:>10.3} us\n  \
             geometry scaling {scale_ratio:.4}   reload/resolve {reload_ratio:.4}   \
             (read/copy {read_ratio:.4})",
            small.len() / 1024,
            us(copy),
            us(resolve),
            us(resolve_fat),
            us(reload)
        );
    }

    let mut failures = Vec::new();
    assert!(
        tracked == MESHES as usize,
        "a resolve left {tracked} of {MESHES} assets resident — this bench measures nothing \
         about resolving unless every mesh actually resolved"
    );
    // Claim 1. A view's cost is per asset, so four times the geometry must cost
    // what one times it did. The bound sits between the ~1.0 a view reads and
    // the 3.9 a copying `Mesh::read` reads — well clear of both, because what
    // fails here is a resolve that touches bytes at all, and that is a 4x
    // signal rather than a marginal one.
    if scale_ratio > 1.6 {
        failures.push(format!(
            "resolving 4x the geometry costs {scale_ratio:.2}x resolving 1x under the same \
             {MESHES} assets — a non-texture asset is meant to be a view over the pack's bytes, \
             and this one is paying per byte (§4.6)"
        ));
    }
    // Claim 2, stated against the resolve leg — the M17 bound of 0.25 divided by
    // the cold pass *including* `Pack::from_bytes`, a host-dependent denominator
    // five times larger, so this is not that number nudged. The band is
    // 0.12–0.14 on the desk and 0.19–0.20 on the lane; a `handle_for` with its
    // cache-hit return removed reads 0.376–0.453 on both. 0.28 is the geometric
    // middle of that gap, ~1.4x either side. The separation is bought by
    // [`RELOAD_PASSES`]: at one pass the band alone spanned 0.25–0.40 — as wide
    // as the signal — and no bound placed anywhere could have discriminated.
    if reload_ratio > 0.28 {
        failures.push(format!(
            "re-loading resident assets costs {reload_ratio:.3} of the cold resolve — `Assets` is \
             keyed by id and a repeat is meant to be a lookup, not a re-resolve"
        ));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
