//! Extract micro-benchmarks (§4.11's first named hot path).
//!
//! No benchmark framework, for the reason `gg-ecs`'s query bench states and
//! this crate inherits: the numbers worth trusting are *ratios* measured in one
//! process on one dataset, which needs no harness and no dependency tree (§3).
//!
//! Two claims are under test, and both are §6 M10's:
//!
//! 1. **Culling pays.** A frustum that rejects most of the world must cost
//!    measurably less than one that keeps it — otherwise the culler is a
//!    correct picture and a wasted pass, the regression §4.11 calls hardest to
//!    notice.
//! 2. **Extract scales with what survives, not with what exists.** Doubling the
//!    world while holding the visible set roughly fixed must not double the
//!    time by more than the scan itself costs.
//!
//! Absolute nanoseconds are printed for the archive; only the ratios assert.
//!
//! # Claim 1 holds in bandwidth, not in arithmetic
//!
//! Six plane dots against a sphere are not cheaper than narrowing one instance
//! and storing it. Held to one thread the cull ratio is ~0.97 here and *1.06* at
//! demo 05's ten thousand — a loss — and both hosts agree to the second decimal,
//! so it is the code and not the desk. What pays is the parallel section:
//! keeping 3 416 of 100 000 instances writes ~6 MiB less, and the unbounded pass
//! is bandwidth-bound long before it is arithmetic-bound.
//!
//! So both are measured and both assert, and they are not redundant: a plane
//! test made sixteen times dearer reads 3.2 on the serial ratio against 1.45 on
//! the parallel one. The serial leg is the sensitive instrument for what the
//! cull itself costs, and the host-independent one; the parallel leg is where
//! claim 1's margin exists at all. The gate that stood here until M17 asserted
//! the *serial* sentence against the *parallel* number, and so asserted neither.
//!
//! All three bounds were made to fail by name before being trusted, by that
//! same dearer plane test — at 16x for the two cull ratios and 48x for the
//! scaling one, which is the shape of the regression it names: a cost paid per
//! entity examined rather than per entity kept.
//!
//! # Two traps in measuring a quotient, both of which this bench fell into
//!
//! **A ratio of two minima is not a ratio.** The minima can come from different
//! reps, so the number mixes two moments of the machine. Every gate here is a
//! paired median: one pass of each leg per rep, quotient taken *within* the rep,
//! median over reps. Min-based, the parallel cull ratio spanned 0.09 to 1.43 on
//! the WSL lane against a 1.05 gate; paired it spans 0.71 to 0.92 over thirteen
//! runs, and the desk's 0.55–0.68 differs from it by scheduler, not by noise.
//!
//! **A pass shorter than the pool's own jitter measures the pool.** At ten
//! thousand entities the parallel pass is ~40 µs and rayon's park/unpark is
//! single-digit µs of it, which is most of that 15x swing. [`ENTITIES`] is set
//! where the instrument can see the code instead.
//!
//! Run: `cargo bench -p gg-extract` (or `cargo xtask bench`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::{Extracted, Frustum};
use gg_math::{render, sim};

/// Entities in the base world — an order of magnitude over demo 05, which is
/// what §6 M10's exit row is stated against, because a 40 µs parallel pass is
/// inside rayon's wake-up noise and the ratios there price the pool. The
/// absolutes stay per-entity and so stay comparable in kind.
const ENTITIES: u32 = 100_000;
/// Repeats per measurement. The minimum is the reported *absolute*; the gated
/// ratios are paired medians over these (see [`ratio`]).
const REPS: u32 = 25;
/// Passes before any of them counts — caches, branch predictors, and rayon's
/// pool, whose threads park and pay a wake-up on the first pass through.
const WARMUP: u32 = 3;
/// Lights, at the caps `gg-extract` enforces — a full set, so the sort and the
/// range cull are both doing work.
const LIGHTS: u32 = 40;

/// A world of `count` renderables spread over a cube, plus a full light set.
///
/// Spread rather than clustered: a frustum test against coincident points would
/// branch identically every row and measure a predictor, not a culler.
fn world_of(count: u32) -> World {
    let mut world = World::new();
    for i in 0..count {
        let entity = world.spawn();
        let f = f64::from(i);
        let position = sim::DVec3::new(
            (f * 0.7).rem_euclid(200.0) - 100.0,
            (f * 0.31).rem_euclid(60.0) - 30.0,
            (f * 1.13).rem_euclid(200.0) - 100.0,
        );
        world
            .insert(
                entity,
                Renderable {
                    position,
                    rotation: sim::DQuat::from_axis_angle(sim::DVec3::new(0.0, 1.0, 0.0), f * 0.01),
                    half_extent: sim::Vec3::new(0.5, 0.5, 0.5),
                    color: 0x00c0_c0c0,
                },
            )
            .unwrap();
    }
    for i in 0..LIGHTS {
        let entity = world.spawn();
        let f = f64::from(i);
        let light = if i < 4 {
            Light::sun(
                sim::Vec3::new(-0.4, -1.0, -0.3),
                0x00ff_f2d8,
                1.0 + i as f32,
            )
        } else {
            Light::point(
                sim::DVec3::new(f * 3.0 - 60.0, 4.0, f * 1.7 - 30.0),
                0x00ff_b060,
                12.0,
                9.0,
            )
        };
        world.insert(entity, light).unwrap();
    }
    world
}

/// One frame's extract, timed. The `black_box` is after the clock so the length
/// the optimizer must not fold away is not itself in the measurement.
fn once(world: &World, frustum: Frustum, out: &mut Extracted) -> Duration {
    let start = Instant::now();
    out.clear(sim::DVec3::ZERO, frustum);
    out.append::<Renderable>(world).unwrap();
    out.append_lights(world).unwrap();
    let elapsed = start.elapsed();
    black_box(out.instances.len());
    elapsed
}

/// The three legs interleaved — one pass of each per rep, every sample kept.
/// Returns `[unbounded, culled, scaled]`.
///
/// Called twice, once per pool, rather than run as five legs in one loop: a
/// second pool in the same process is not free to the first. Sixteen workers and
/// a seventeenth on sixteen cores put the parallel unbounded leg at 8.8
/// ns/entity against 4.9 measured alone — the denominator of claim 1, degraded
/// by the instrument.
fn sweep(
    world: &World,
    doubled: &World,
    narrow: Frustum,
    out: &mut Extracted,
) -> [Vec<Duration>; 3] {
    let mut samples: [Vec<Duration>; 3] = Default::default();
    for rep in 0..WARMUP + REPS {
        let pass = [
            once(world, Frustum::UNBOUNDED, out),
            once(world, narrow, out),
            once(doubled, narrow, out),
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

/// Median of the per-rep quotients, paired within the rep — the statistic every
/// gate here uses, for the reason the module header gives.
fn ratio(num: &[Duration], den: &[Duration]) -> f64 {
    let mut quotients: Vec<f64> = num
        .iter()
        .zip(den)
        .map(|(n, d)| n.as_secs_f64() / d.as_secs_f64())
        .collect();
    quotients.sort_by(f64::total_cmp);
    quotients[quotients.len() / 2]
}

/// A frustum tight enough to reject most of `world_of`'s spread: a narrow fov
/// looking down -Z from the origin, which is where the camera sits here.
fn narrow() -> Frustum {
    Frustum::from_view_projection(render::perspective_reverse_z(0.35, 16.0 / 9.0, 0.05))
}

fn main() {
    let world = world_of(ENTITIES);
    let doubled = world_of(ENTITIES * 2);
    let narrow = narrow();
    let mut out = Extracted::default();

    // Counts first and untimed: what the dataset does is a statement about the
    // dataset, and asserting it inside a timed loop would price the assertion.
    once(&world, Frustum::UNBOUNDED, &mut out);
    let kept_all = out.instances.len();
    once(&world, narrow, &mut out);
    let (kept_narrow, rejected) = (out.instances.len(), out.culled);

    let [unbounded, culled, scaled] = sweep(&world, &doubled, narrow, &mut out);
    // `install` runs the sweep on this pool, so the `par_iter` inside `append`
    // is serial for its duration and nothing else about the code path changes.
    let serial = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let [s_unbounded, s_culled, _] = serial.install(|| sweep(&world, &doubled, narrow, &mut out));

    let ns = |d: Duration, n: usize| d.as_secs_f64() * 1e9 / n as f64;
    let per_entity = ns(best(&unbounded), ENTITIES as usize);
    let cull_ratio = ratio(&culled, &unbounded);
    let serial_ratio = ratio(&s_culled, &s_unbounded);
    let scale_ratio = ratio(&scaled, &culled);
    let culled = best(&culled);
    let (s_unbounded, s_culled) = (best(&s_unbounded), best(&s_culled));
    let threads = rayon::current_num_threads();

    if std::env::args().any(|a| a == "--json") {
        println!(
            "{{\"entities\":{ENTITIES},\"reps\":{REPS},\"lights\":{LIGHTS},\"threads\":{threads},\
             \"kept_unbounded\":{kept_all},\"kept_narrow\":{kept_narrow},\"culled\":{rejected},\
             \"ns_per_entity\":{{\"unbounded\":{:.3},\"culled\":{:.3},\"unbounded_serial\":{:.3},\
             \"culled_serial\":{:.3}}},\
             \"ratios\":{{\"cull\":{cull_ratio:.4},\"cull_serial\":{serial_ratio:.4},\
             \"double_the_world\":{scale_ratio:.4}}}}}",
            per_entity,
            ns(culled, ENTITIES as usize),
            ns(s_unbounded, ENTITIES as usize),
            ns(s_culled, ENTITIES as usize)
        );
    } else {
        println!(
            "extract {ENTITIES} entities + {LIGHTS} lights, {threads} threads, \
             paired median of {REPS}:\n  \
             unbounded {:>8.3} ns/entity ({kept_all} kept)   serial {:>8.3}\n  \
             narrow    {:>8.3} ns/entity ({kept_narrow} kept, {rejected} culled)   \
             serial {:>8.3}\n  \
             cull ratio {cull_ratio:.3} ({serial_ratio:.3} serial)   \
             double-the-world ratio {scale_ratio:.3}",
            per_entity,
            ns(s_unbounded, ENTITIES as usize),
            ns(culled, ENTITIES as usize),
            ns(s_culled, ENTITIES as usize)
        );
    }

    let mut failures = Vec::new();
    assert!(
        rejected > kept_narrow,
        "the narrow frustum kept {kept_narrow} and culled {rejected} — this bench measures \
         nothing about culling unless most of the world is rejected"
    );
    // Claim 1, priced on the pool the engine actually uses. The bound sits where
    // the *claim* fails rather than where the measurement sits: the desk reads
    // 0.55–0.71 and the lane 0.71–0.94 over nineteen runs, and the two differ by
    // how well the scheduler delivers sixteen threads, which is not something to
    // gate on. A culler that has stopped paying is one that costs more than not
    // culling, so the bound is 1.0 and the lane's margin is ~6% — tight on
    // purpose. Firing at 0.98–1.02 is a finding, not a flake: the pathological
    // variance is gone (this is a paired median), so what is left there is the
    // cull's margin genuinely at zero on that host.
    if cull_ratio > 1.0 {
        failures.push(format!(
            "culling {rejected}/{kept_all} instances costs {cull_ratio:.2}x an unbounded extract \
             on {threads} threads — the bandwidth the cull saves is what pays for it, and it has \
             stopped paying"
        ));
    }
    // The other half, and the sensitive end: 0.86–1.18 over twenty-four runs
    // across both hosts, so the frustum test is within ~20% of the narrowing it
    // skips in either direction and is allowed to stay there. A cheaper plane
    // test lowers this and is welcome. The bound sits in the gap between that
    // band and the 3.2 a sixteen-times-dearer plane test reads — nearer the
    // noise, because the signal is 2.3x clear of it either way. A gate whose
    // margin is smaller than its own spread reports the desk, which is how the
    // 1.05 that stood here until M17 came to flake.
    if serial_ratio > 1.4 {
        failures.push(format!(
            "single-threaded, the frustum test makes extract {serial_ratio:.2}x its unbounded \
             cost — the plane tests have grown past the narrowing they skip"
        ));
    }
    // Twice the entities, roughly the same survivors: the extra cost must be
    // the scan, not the gather. Well under 2x is the claim.
    if scale_ratio > 1.8 {
        failures.push(format!(
            "doubling the world cost {scale_ratio:.2}x with the visible set unchanged — extract \
             is scaling with what exists rather than with what survives (§6 M10)"
        ));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
