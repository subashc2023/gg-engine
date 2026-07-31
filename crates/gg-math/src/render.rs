//! `gg_math::render` — full-SIMD `glam`, platform intrinsics welcome
//! (§4.2.1). Used render-side only; correctness here is judged by `gg-golden`,
//! never by a hash. Determinism-critical crates must not depend on this module
//! (§3 — deny.toml holds the crate-level ban on direct `glam`).
//!
//! Crossing the membrane is explicit and one-way: sim positions are `f64` and
//! narrow **only** through [`camera_relative`], which demands a camera origin —
//! there is deliberately no `From<sim::DVec3> for Vec3`, so an absolute-position
//! `f32` cannot exist render-side by accident (§1.4, §4.2.1).

pub use glam::*;

use crate::sim;

/// Narrow a sim-space position to render space, relative to the camera
/// origin: subtract in `f64`, *then* round once to `f32`. The subtraction is
/// what restores precision near the camera at Expanse scale — narrowing first
/// would throw it away before the subtract.
pub fn camera_relative(world: sim::DVec3, camera_origin: sim::DVec3) -> Vec3 {
    Vec3::new(
        (world.x - camera_origin.x) as f32,
        (world.y - camera_origin.y) as f32,
        (world.z - camera_origin.z) as f32,
    )
}

/// Narrow a sim rotation for render use. Rotations are origin-free, so this
/// is a plain per-component rounding.
pub fn narrow_rotation(q: sim::DQuat) -> Quat {
    Quat::from_xyzw(q.x as f32, q.y as f32, q.z as f32, q.w as f32)
}

/// Convert a *non-positional* sim `f32` vector (a color, a local offset, a
/// normal) for render use. Positions never take this path — they are `f64`
/// sim-side and must go through [`camera_relative`].
pub fn to_render(v: sim::Vec3) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_relative_preserves_precision_near_the_camera() {
        // 10 million units out — beyond f32 integer precision (2^24 ≈ 16.7M,
        // where spacing is already 1.0) — but 0.25 from the camera. The f64
        // subtract-first path must recover the 0.25 exactly.
        let world = sim::DVec3::new(10_000_000.25, 0.0, 0.0);
        let camera = sim::DVec3::new(10_000_000.0, 0.0, 0.0);
        assert_eq!(camera_relative(world, camera), Vec3::new(0.25, 0.0, 0.0));

        // Narrow-first would have quantized to whole units at that range.
        let narrowed_first = world.x as f32 - camera.x as f32;
        assert_ne!(narrowed_first, 0.25);
    }

    #[test]
    fn narrowing_rotations_and_vectors() {
        let q = narrow_rotation(sim::DQuat::IDENTITY);
        assert_eq!(q, Quat::IDENTITY);
        assert_eq!(
            to_render(sim::Vec3::new(0.1, 0.2, 0.3)),
            Vec3::new(0.1, 0.2, 0.3)
        );
    }
}
