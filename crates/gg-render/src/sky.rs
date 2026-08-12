//! The environment, and its projection onto spherical harmonics (§6 M24).
//!
//! Image-based lighting needs the *integral* of the environment against a lobe,
//! not the environment: a diffuse surface gathers a whole hemisphere and a rough
//! specular one gathers most of it, so what both actually see is the low-order
//! part. Nine spherical-harmonic coefficients hold that part exactly — the
//! classic result is that a clamped-cosine kernel has only three nonzero bands,
//! so an order-2 projection is not an approximation of the diffuse term, it is
//! the diffuse term ([RH01]).
//!
//! So this module is the projection, and the game-declared `Sky` behind
//! [`SkyLook`](gg_extract::SkyLook) is only what happens to feed it
//! today. The quadrature below takes an arbitrary function of
//! direction; the day the source is a loaded panorama, [`radiance`] samples it
//! instead and nothing downstream — not the coefficients, not the shader, not
//! the frame block — changes shape.
//!
//! # References
//!
//!   [RH01] Ramamoorthi & Hanrahan, "An Efficient Representation for Irradiance
//!          Environment Maps", SIGGRAPH 2001 — the order-2 result and the three
//!          band factors below.
//!   [Slo08] Sloan, "Stupid Spherical Harmonics (SH) Tricks", GDC 2008 — the
//!          basis in the polynomial form used here, and the convolution weights
//!          the specular path applies per band.

use gg_extract::SkyLook as Sky;
use gg_math::render;

use crate::srgb_to_linear;

/// Coefficients in an order-2 real SH projection: bands 0, 1 and 2 are 1 + 3 + 5.
pub(crate) const SH_COEFFICIENTS: usize = 9;

/// Directions the quadrature below integrates over.
///
/// A Fibonacci sphere at this count leaves the residual on a *smooth* integrand
/// four orders of magnitude below the mean it is estimating, which the zonal
/// test pins directly: an altitude-only sky must project to exactly zero in the
/// five coefficients that are odd in x or z, and at 2048 samples the largest of
/// them measures 0.014 % of band 0.
const SAMPLES: usize = 2048;

/// The golden angle, radians — `2π(2 - φ)`. Successive samples advance by it,
/// which is what makes the point set uniform at *every* prefix rather than only
/// at the full count.
const GOLDEN_ANGLE: f32 = 2.399_963_2;

/// The real SH basis, bands 0..=2, in the polynomial form [Slo08] gives.
///
/// **Duplicated in `include/pbr.slang`'s `sh_basis`, deliberately and with the
/// same literals.** The two sides never meet at runtime — this writes
/// coefficients, that one evaluates them — so there is nothing to assert
/// against and the only defence is that both are the textbook polynomials and
/// neither is derived from the other. The reconstruction test below is what
/// would catch this half being wrong.
///
/// `z` is the polar axis in these formulas and the engine's up is `y`. That is
/// not a bug and is not corrected: the basis is a complete orthonormal set
/// whatever axis one calls polar, and the only requirement is that the
/// projection and the evaluation use the same formulas. What it means in
/// practice is that an altitude-only sky lands in coefficients 0, 1, 6 and 8
/// rather than in 0, 2 and 6.
pub(crate) fn basis(d: render::Vec3) -> [f32; SH_COEFFICIENTS] {
    [
        0.282_094_79,
        0.488_602_51 * d.y,
        0.488_602_51 * d.z,
        0.488_602_51 * d.x,
        1.092_548_4 * d.x * d.y,
        1.092_548_4 * d.y * d.z,
        0.315_391_57 * (3.0 * d.z * d.z - 1.0),
        1.092_548_4 * d.x * d.z,
        0.546_274_2 * (d.x * d.x - d.y * d.y),
    ]
}

/// What the environment radiates along `direction`, linear rgb.
///
/// A vertical gradient met along the **square root** of altitude rather than
/// linearly, in both hemispheres. A linear ramp puts half its change in the top
/// half of the sky, where a real one has almost none — the transition a person
/// reads as "sky" is concentrated in the twenty degrees above the horizon, and
/// the square root is the cheapest curve that puts it there.
pub(crate) fn radiance(sky: &Sky, direction: render::Vec3) -> render::Vec3 {
    let horizon = linear(sky.horizon);
    let pole = if direction.y >= 0.0 {
        linear(sky.zenith)
    } else {
        linear(sky.ground)
    };
    // `direction` is unit, so `y` is already the sine of the altitude and this
    // needs no clamp beyond the one an unnormalized caller would deserve.
    let k = direction.y.abs().min(1.0).sqrt();
    horizon.lerp(pole, k) * sky.intensity.max(0.0)
}

fn linear(color: u32) -> render::Vec3 {
    let c = srgb_to_linear(color);
    render::Vec3::new(c[0], c[1], c[2])
}

/// The environment projected onto [`SH_COEFFICIENTS`] radiance coefficients,
/// laid out as the frame block wants them — `float4` a piece, `w` unused,
/// because a `float3` array in a std430-ish block is a stride nobody should have
/// to remember.
///
/// Numerical quadrature and not the closed form the current sky admits. The
/// closed form exists — an altitude-only gradient integrates against the three
/// zonal harmonics analytically — and it would be both exact and free. It is not
/// used because it is a property of *this* environment rather than of the
/// projection, and the first environment that is not a vertical gradient would
/// delete it. Two thousand samples of a smooth function, recomputed only when
/// the sky changes, is not a cost worth specializing.
pub(crate) fn project(sky: &Sky) -> [[f32; 4]; SH_COEFFICIENTS] {
    let mut out = [[0.0f32; 4]; SH_COEFFICIENTS];
    // Monte Carlo over a uniform point set: the measure of the sphere divided
    // by the count, which is the weight every sample carries because the set is
    // uniform by construction rather than by importance.
    let weight = 4.0 * core::f32::consts::PI / SAMPLES as f32;
    for k in 0..SAMPLES {
        let direction = fibonacci(k);
        let radiance = radiance(sky, direction) * weight;
        let basis = basis(direction);
        for (coefficient, y) in out.iter_mut().zip(basis) {
            coefficient[0] += radiance.x * y;
            coefficient[1] += radiance.y * y;
            coefficient[2] += radiance.z * y;
        }
    }
    out
}

/// Sample `k` of a Fibonacci sphere: equal-area in `z`, advanced by the golden
/// angle in longitude.
#[expect(
    clippy::disallowed_methods,
    reason = "render-side quadrature over a smooth analytic function, never \
              hashed and never crossing a tick — the §3 ban is about the sim"
)]
fn fibonacci(k: usize) -> render::Vec3 {
    // Offset by a half so neither pole is a sample: the basis is finite there,
    // but a point set including both ends is one that is not equal-area.
    let z = 1.0 - (2.0 * k as f32 + 1.0) / SAMPLES as f32;
    let radius = (1.0 - z * z).max(0.0).sqrt();
    let (sin, cos) = (k as f32 * GOLDEN_ANGLE).sin_cos();
    render::Vec3::new(radius * cos, radius * sin, z)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// `gg_ecs::boundary::Sky::daylight`'s colours, restated rather than
    /// reached for: this crate does not link the one that owns that constructor
    /// (§3), and a test that needed it would be a dependency the shipping graph
    /// then carries.
    fn daylight() -> Sky {
        Sky {
            zenith: 0x0059_8fd8,
            horizon: 0x00c8_d4e0,
            ground: 0x0035_3129,
            intensity: 1.0,
            environment: 0,
        }
    }

    /// The gradient is met at the horizon from both sides and reaches each pole
    /// exactly — the property the square root must not break.
    #[test]
    fn the_gradient_runs_from_horizon_to_both_poles() {
        let sky = daylight();
        let at = |y: f32| radiance(&sky, render::Vec3::new(0.0, y, 0.0));
        assert!((at(1.0) - linear(sky.zenith)).length() < 1e-6);
        assert!((at(-1.0) - linear(sky.ground)).length() < 1e-6);
        assert!((at(0.0) - linear(sky.horizon)).length() < 1e-6);
        // Concentrated low: at a tenth of the way up, a square root is already a
        // third of the way to the zenith where a straight line would be a tenth.
        let third = (at(0.01) - at(0.0)).length() / (at(1.0) - at(0.0)).length();
        assert!(
            (0.09..0.11).contains(&third),
            "sqrt(0.01) = 0.1, got {third}"
        );
    }

    /// Intensity is the switch as well as the scale: zero is no environment,
    /// which is what a world declaring no `Sky` gets and what a migration
    /// zeroing the field must leave behind.
    #[test]
    fn a_zero_intensity_sky_radiates_nothing_in_every_direction() {
        let sky = Sky {
            intensity: 0.0,
            ..daylight()
        };
        for d in [render::Vec3::Y, render::Vec3::NEG_Y, render::Vec3::X] {
            assert_eq!(radiance(&sky, d), render::Vec3::ZERO);
        }
        assert!(project(&sky).iter().flatten().all(|c| *c == 0.0));
    }

    /// An environment that depends only on altitude has no longitude to project
    /// onto: every coefficient odd in `x` or `z` must integrate to zero.
    ///
    /// This is the quadrature's own gate rather than the sky's. A point set that
    /// clumped — the failure mode of every "uniform" sphere sampling that is not
    /// — leaves exactly this residue, and it would otherwise show up as an
    /// environment that lights a wall differently depending on which way it
    /// faces for no reason anyone could see.
    #[test]
    fn an_altitude_only_sky_projects_onto_the_zonal_coefficients_alone() {
        let sh = project(&daylight());
        let magnitude = |i: usize| render::Vec3::from_slice(&sh[i][..3]).length();
        // 0 is the mean, 1 is y, 6 and 8 both carry y through a square.
        for zonal in [0, 1, 6, 8] {
            assert!(magnitude(zonal) > 1e-3, "coefficient {zonal} is the sky");
        }
        // Against band 0 rather than against an absolute number: what makes a
        // residue harmless is that it is small *compared to the sky*, and an
        // absolute bound would pass or fail on the sky's brightness instead of
        // on the point set's uniformity, which is the thing under test.
        let mean = magnitude(0);
        for off_axis in [2, 3, 4, 5, 7] {
            let residue = magnitude(off_axis) / mean;
            assert!(
                residue < 1e-3,
                "coefficient {off_axis} is {residue} of band 0"
            );
        }
    }

    /// The whole claim, checked against the definition: irradiance rebuilt from
    /// nine coefficients must match a brute-force cosine-weighted integral of
    /// the environment itself.
    ///
    /// Six per cent, and that is not slack — it is [RH01]'s own error, which
    /// they report as ~1 % average and 9 % worst case over real environments.
    /// An order-2 projection is exact for the *lobe*: the clamped cosine has no
    /// band above 2. What it is not exact for is the environment, which does —
    /// a square-root gradient has energy at band 4, the lobe attenuates it
    /// rather than deleting it, and the projection drops it entirely.
    ///
    /// The worst normal is straight **down**, at 5.1 %, and that is where the
    /// theory says it should be: the gradient's own curvature is sharpest
    /// across the horizon, so a hemisphere centred on a pole integrates the
    /// most of what band 2 cannot hold. A sky with a sun disk in it would blow
    /// this open entirely, which is one more reason the sun is a `Light` and
    /// not a pixel.
    #[test]
    fn irradiance_rebuilt_from_nine_coefficients_matches_the_integral() {
        let sky = daylight();
        let sh = project(&sky);
        for normal in [
            render::Vec3::Y,
            render::Vec3::NEG_Y,
            render::Vec3::X,
            render::Vec3::new(0.3, 0.8, -0.5).normalize(),
        ] {
            let rebuilt = irradiance(&sh, normal);
            let integrated = cosine_integral(&sky, normal);
            let error = (rebuilt - integrated).length() / integrated.length();
            assert!(error < 0.06, "normal {normal}: {rebuilt} vs {integrated}");
        }
    }

    /// [RH01]'s three band factors, applied to the coefficients — the shader's
    /// `sh_irradiance` in Rust, and the only reason it is here is that the claim
    /// above needs something to check.
    fn irradiance(sh: &[[f32; 4]; SH_COEFFICIENTS], normal: render::Vec3) -> render::Vec3 {
        // pi, 2pi/3, pi/4 — [RH01]'s A0, A1, A2, spelled as the fractions of pi
        // they are rather than as decimals, which is both what the paper writes
        // and what stops a transcription error from being invisible.
        use core::f32::consts::{FRAC_PI_4, PI};
        const A1: f32 = 2.0 * PI / 3.0;
        const BANDS: [f32; SH_COEFFICIENTS] = [
            PI, A1, A1, A1, FRAC_PI_4, FRAC_PI_4, FRAC_PI_4, FRAC_PI_4, FRAC_PI_4,
        ];
        let basis = basis(normal);
        let mut total = render::Vec3::ZERO;
        for ((coefficient, y), band) in sh.iter().zip(basis).zip(BANDS) {
            total += render::Vec3::from_slice(&coefficient[..3]) * (y * band);
        }
        total
    }

    /// The definition: the environment integrated against a clamped cosine.
    fn cosine_integral(sky: &Sky, normal: render::Vec3) -> render::Vec3 {
        let mut total = render::Vec3::ZERO;
        for k in 0..SAMPLES {
            let d = fibonacci(k);
            total += radiance(sky, d) * normal.dot(d).max(0.0);
        }
        total * (4.0 * core::f32::consts::PI / SAMPLES as f32)
    }
}
