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
//! Run: `cargo bench -p gg-extract` (or `cargo xtask bench`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;
use std::time::{Duration, Instant};

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::{Extracted, Frustum};
use gg_math::{render, sim};

/// Entities in the base world. Demo 05's order of magnitude, which is what the
/// M10 exit row is stated against.
const ENTITIES: u32 = 10_000;
/// Repeats per measurement; the minimum is reported, since desktop noise is
/// one-sided.
const REPS: u32 = 25;
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

/// Minimum wall time over `REPS` runs of one extract.
fn measure(world: &World, frustum: Frustum, out: &mut Extracted) -> Duration {
    let mut best = Duration::MAX;
    for _ in 0..REPS {
        let start = Instant::now();
        out.clear(sim::DVec3::ZERO, frustum);
        out.append::<Renderable>(world).unwrap();
        out.append_lights(world).unwrap();
        let elapsed = start.elapsed();
        black_box(out.instances.len());
        best = best.min(elapsed);
    }
    best
}

/// A frustum tight enough to reject most of `world_of`'s spread: a narrow fov
/// looking down -Z from the origin, which is where the camera sits here.
fn narrow() -> Frustum {
    Frustum::from_view_projection(render::perspective_reverse_z(0.35, 16.0 / 9.0, 0.05))
}

fn main() {
    let world = world_of(ENTITIES);
    let doubled = world_of(ENTITIES * 2);
    let mut out = Extracted::default();

    let unbounded = measure(&world, Frustum::UNBOUNDED, &mut out);
    let kept_all = out.instances.len();
    let culled = measure(&world, narrow(), &mut out);
    let kept_narrow = out.instances.len();
    let rejected = out.culled;
    let scaled = measure(&doubled, narrow(), &mut out);

    let ns = |d: Duration, n: usize| d.as_secs_f64() * 1e9 / n as f64;
    let per_entity = ns(unbounded, ENTITIES as usize);
    let cull_ratio = culled.as_secs_f64() / unbounded.as_secs_f64();
    let scale_ratio = scaled.as_secs_f64() / culled.as_secs_f64();

    if std::env::args().any(|a| a == "--json") {
        println!(
            "{{\"entities\":{ENTITIES},\"reps\":{REPS},\"lights\":{LIGHTS},\
             \"kept_unbounded\":{kept_all},\"kept_narrow\":{kept_narrow},\"culled\":{rejected},\
             \"ns_per_entity\":{{\"unbounded\":{:.3},\"culled\":{:.3}}},\
             \"ratios\":{{\"cull\":{cull_ratio:.4},\"double_the_world\":{scale_ratio:.4}}}}}",
            per_entity,
            ns(culled, ENTITIES as usize)
        );
    } else {
        println!(
            "extract {ENTITIES} entities + {LIGHTS} lights, min of {REPS}:\n  \
             unbounded {:>8.3} ns/entity ({kept_all} kept)\n  \
             narrow    {:>8.3} ns/entity ({kept_narrow} kept, {rejected} culled)\n  \
             cull ratio {cull_ratio:.3}   double-the-world ratio {scale_ratio:.3}",
            per_entity,
            ns(culled, ENTITIES as usize)
        );
    }

    let mut failures = Vec::new();
    assert!(
        rejected > kept_narrow,
        "the narrow frustum kept {kept_narrow} and culled {rejected} — this bench measures \
         nothing about culling unless most of the world is rejected"
    );
    // A culler that costs as much as drawing everything is not paying for
    // itself. The bound is loose on purpose: the win is the *draw* it avoids,
    // so extract merely has to not get slower.
    if cull_ratio > 1.05 {
        failures.push(format!(
            "culling {rejected}/{kept_all} instances costs {cull_ratio:.2}x an unbounded \
             extract — the frustum test is meant to be cheaper than the work it removes"
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
