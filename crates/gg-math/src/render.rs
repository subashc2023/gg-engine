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

/// The engine's one projection (§2, Math row): right-handed Y-up view space in,
/// Vulkan NDC out, **reverse-Z with an infinite far plane** — `near` maps to
/// depth 1 and infinity to depth 0.
///
/// It exists as a named function rather than a call to glam's so the convention
/// has exactly one home: reverse-Z touches every depth comparison
/// (`GREATER_OR_EQUAL`), every depth clear (`0.0`), and every golden image, and
/// a second projection built the ordinary way would render a scene that is
/// subtly, consistently wrong. `vertical_fov` is in radians.
///
/// Built here rather than delegated to glam's `perspective_infinite_reverse`
/// for one reason beyond that, and it is not a determinism one — this is the
/// `f32` side of §1.4's membrane and nothing hashed passes through it. It is a
/// *golden-image portability* one. The cotangent below is the only transcendental
/// in the engine's projection, it scales every pixel of every frame, and glam
/// reaches std for it — which put a `sincosf` import in the aarch64 dev artifact
/// (§6 M17 item 6) and made the picture a function of the host's libm. Measured
/// over vertical fovs from 0.05 to 3.0 rad: the MSVC CRT and `libm` disagree on
/// `cos/sin` at 86 of 29 509 angles, by up to 2 ulp. Small, and not nothing —
/// `r.fov` is a runtime CVar, so which side of that 0.29% a session lands on is
/// the operator's to choose, and §4.10 expects the two lavapipe reference sets
/// to stay byte-identical. Through `sim` they agree by construction instead of
/// by luck, at no cost in pixels: every fov in this tree came out bit-identical
/// either way, which `the_cotangent_is_the_one_the_golden_sets_were_blessed_against`
/// keeps true.
pub fn perspective_reverse_z(vertical_fov: f32, aspect_ratio: f32, near: f32) -> Mat4 {
    // glam's own form, and deliberately `cos/sin` rather than the algebraically
    // equal `1/tan`: the two differ by an ulp at fovs where sin and cos agree,
    // so reproducing the matrix bit-for-bit means reproducing the expression.
    let (sin_half, cos_half) = sim::sin_cos(vertical_fov * 0.5);
    let h = cos_half / sin_half;
    Mat4::from_cols(
        Vec4::new(h / aspect_ratio, 0.0, 0.0, 0.0),
        // Vulkan's clip space is Y-down.
        Vec4::new(0.0, -h, 0.0, 0.0),
        // Infinite far: `near` maps to depth 1 and infinity to 0, and w carries
        // the perspective divide's -z.
        Vec4::new(0.0, 0.0, 0.0, -1.0),
        Vec4::new(0.0, 0.0, near, 0.0),
    )
}

/// The engine's other projection: right-handed Y-up view space in, Vulkan NDC
/// out, **reverse-Z with a finite far plane** — `near` maps to depth 1 and
/// `far` to depth 0. Born as the shadow map's (§6 M11); since §6 M20 also the
/// camera's, when a game declares an orthographic `Eye`.
///
/// Finite rather than infinite, unlike [`perspective_reverse_z`], and that is
/// the difference worth naming: parallel rays have no position for depth to
/// diverge from, so the depth range is a *choice* — the slab of the world the
/// cascade (or the playfield) covers. An infinite far plane would put every
/// unlit metre of the scene into the same depth budget as the metre the shadow
/// is being tested at.
///
/// Reverse-Z is not an optimisation here but a compatibility requirement: the
/// shadow map is compared against with the same `GREATER_OR_EQUAL` and cleared
/// to the same `0.0` as every other depth buffer in the engine (§2, Math row),
/// and one buffer with the other convention would read as a shadow covering
/// exactly what is lit.
pub fn orthographic_reverse_z(half_width: f32, half_height: f32, near: f32, far: f32) -> Mat4 {
    // Built rather than borrowed from glam for the same reason
    // `perspective_reverse_z` is a named function: the convention has one home.
    // Columns, because glam is column-major.
    //
    // `z_ndc = (z + far) / (far - near)` for a view-space z in [-far, -near],
    // which is the *reverse* of the ordinary orthographic's mapping. The Y row
    // is negated because Vulkan's clip space is Y-down and glam's `vulkan`
    // perspective already is — a shadow map built the other way up would put
    // every shadow on the wrong side of its caster.
    let depth = far - near;
    Mat4::from_cols(
        Vec4::new(1.0 / half_width, 0.0, 0.0, 0.0),
        Vec4::new(0.0, -1.0 / half_height, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0 / depth, 0.0),
        Vec4::new(0.0, 0.0, far / depth, 1.0),
    )
}

/// A `float4x4` push-constant field, row-major — the layout `gg-shaders`
/// reflects and asserts (§4.4). glam is column-major, so this transposes: a
/// missing transpose here is the classic silently-wrong render.
pub fn rows(m: Mat4) -> [[f32; 4]; 4] {
    m.transpose().to_cols_array_2d()
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

    /// The cotangent is the projection's only transcendental and so the only
    /// place two libms can disagree. These are its bits at every fov this tree
    /// uses, and they are **what glam produced on the day the projection moved
    /// off it** — which is why that change re-blessed no golden, and what this
    /// test exists to keep true.
    ///
    /// Pinned rather than still compared against glam, for two reasons. The
    /// comparison would put the call back in the binary and undo the point: the
    /// aarch64 leg's artifacts import no math routine (§6 M17 item 6), and a
    /// test is one of those artifacts. And it would re-tie the picture to a
    /// dependency's choice of formula, which is the tie being cut — glam is now
    /// free to change its mind without moving a pixel here.
    ///
    /// Every other entry is exact arithmetic on this number and the two
    /// arguments, so the structural claims stay in the test below rather than
    /// being restated per fov.
    #[test]
    fn the_cotangent_is_the_one_the_golden_sets_were_blessed_against() {
        for (fov, bits) in [
            (1.0_f32, 0x3fea_4d6b_u32),                 // r.fov's default
            (std::f32::consts::FRAC_PI_3, 0x3fdd_b3d7), // 60°, the unit tests'
            (0.35, 0x40b4_fc96),                        // the extract bench's
            (1.2, 0x3fbb_18da),
            (0.7, 0x402f_542b),
        ] {
            // Column 1's y is -h and a sign flip is exact, so this reads the
            // cotangent back out of the matrix rather than recomputing it.
            let h = -perspective_reverse_z(fov, 1.0, 0.1).y_axis.y;
            assert_eq!(
                h.to_bits(),
                bits,
                "fov {fov} produced {h} ({:08x})",
                h.to_bits()
            );
        }
    }

    #[test]
    fn reverse_z_puts_the_near_plane_at_one_and_infinity_at_zero() {
        let p = perspective_reverse_z(std::f32::consts::FRAC_PI_3, 16.0 / 9.0, 0.1);
        // View space is right-handed with -Z forward, so a point in front of
        // the camera has a negative z.
        let near = p * Vec4::new(0.0, 0.0, -0.1, 1.0);
        assert!(
            (near.z / near.w - 1.0).abs() < 1e-5,
            "near mapped to {near:?}"
        );
        let far = p * Vec4::new(0.0, 0.0, -1.0e9, 1.0);
        assert!((far.z / far.w).abs() < 1e-5, "far mapped to {far:?}");
        // Nearer must compare *greater* — the whole reason the depth test is
        // GREATER_OR_EQUAL and the buffer clears to 0.0.
        let mid = p * Vec4::new(0.0, 0.0, -10.0, 1.0);
        assert!(mid.z / mid.w < near.z / near.w);
        assert!(mid.z / mid.w > far.z / far.w);
    }

    #[test]
    fn the_shadow_projection_agrees_with_the_camera_on_every_axis() {
        let o = orthographic_reverse_z(10.0, 10.0, 1.0, 51.0);
        // Reverse-Z, the same way round as the perspective: near is 1, far is 0.
        let near = o * Vec4::new(0.0, 0.0, -1.0, 1.0);
        let far = o * Vec4::new(0.0, 0.0, -51.0, 1.0);
        assert!((near.z / near.w - 1.0).abs() < 1e-6, "near {near:?}");
        assert!((far.z / far.w).abs() < 1e-6, "far {far:?}");
        // Orthographic: w is 1 everywhere, which is what makes depth linear and
        // is the reason a directional light gets this projection and not the
        // other one.
        assert_eq!(near.w, 1.0);
        assert_eq!(far.w, 1.0);
        // And nearer still compares greater, so one `GREATER_OR_EQUAL` serves
        // both buffers.
        let mid = o * Vec4::new(0.0, 0.0, -26.0, 1.0);
        assert!(mid.z < near.z && mid.z > far.z);

        // The Y flip is the bug that survives every scalar check and shows up
        // as shadows on the wrong side of their casters, so it is compared
        // against the projection that already has it right rather than asserted
        // from memory.
        let p = perspective_reverse_z(std::f32::consts::FRAC_PI_3, 1.0, 0.1);
        let above = Vec4::new(0.0, 1.0, -4.0, 1.0);
        let right = Vec4::new(1.0, 0.0, -4.0, 1.0);
        assert_eq!(
            (o * above).y.signum(),
            (p * above).y.signum(),
            "Y disagrees"
        );
        assert_eq!(
            (o * right).x.signum(),
            (p * right).x.signum(),
            "X disagrees"
        );
        // Half-extents are half-extents: the edge of the slab lands on the edge
        // of clip space.
        assert!(((o * Vec4::new(10.0, 0.0, -2.0, 1.0)).x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rows_transposes_out_of_glams_column_major_storage() {
        let m = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        // Translation lives in the last *column* of a row-major float4x4, which
        // is what `mul(M, v)` in the shader reads.
        assert_eq!(rows(m)[0][3], 1.0);
        assert_eq!(rows(m)[1][3], 2.0);
        assert_eq!(rows(m)[2][3], 3.0);
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
