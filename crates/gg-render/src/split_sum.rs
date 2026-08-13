//! The split-sum's second integral, computed rather than fitted (§6 M34).
//!
//! [Laz13]'s four-constant fit is what `include/pbr.slang` shipped through M33,
//! and §6 M33 found the one thing it cannot express: it returns
//! `(-1.04a + z, 1.04a + w)`, so `scale + bias` — the directional albedo at
//! `f0 = 1`, which is the whole of what the multiscatter correction divides by —
//! is `z + w` and carries no view angle at all. The real integral falls toward
//! the silhouette. A table can say so and a fit of that shape cannot.
//!
//! Everything here mirrors `include/pbr.slang` term for term — the same
//! height-correlated Smith visibility, the same spherical-Gaussian Schlick —
//! because what the table has to approximate is *this renderer's* BRDF and not
//! the one the citation integrated. [`fit`] is kept beside it so the instrument
//! can grade one against the other in one binary.
//!
//! `sim` transcendentals throughout: the table is generated on the host at
//! startup, and one that differed between Windows and Linux would put a
//! host difference into every reference image that samples it.

use gg_math::sim;

/// The table's edge, in both axes: `n·v` across, perceptual roughness down.
///
/// Sixteen, and it is a *measured* sixteen — `gg-tools split-sum` sweeps this
/// against 8, 32 and 64 and 16 is the knee: 0.21 % mean and 1.45 % worst error
/// on `1/E`, which 64² improves to 0.19 % and 1.67 % — that is, not at all, and
/// in the worst case not even in the right direction. Sampling a finer grid puts
/// texels closer to `n·v = 0` where the integral itself is hardest to estimate,
/// so past the knee the table buys interpolation accuracy and pays for it in
/// texel accuracy.
///
/// Both axes are read at the **corners** rather than at texel centres — no
/// half-texel offset — which is what a manual bilinear read buys over a sampler.
/// It matters at exactly one place and that place is load-bearing: roughness 0
/// must read the mirror's albedo of 1 *exactly*, because the correction divides
/// by it.
pub const EXTENT: usize = 16;

/// Samples per texel when the renderer builds its table.
///
/// The whole build is `EXTENT² × SAMPLES` GGX samples once per process, ~8 ms in
/// the dev profile. Quadrupling it moves the mean from 0.21 % to 0.12 % and the
/// worst from 1.45 % to 1.38 %, which is not worth four times the startup —
/// again `gg-tools split-sum`, and again the residual is the corner rather than
/// the count.
pub const SAMPLES: u32 = 1024;

/// The `n·v` axis is stored **squared**: texel `x` holds `n·v = (x/edge)²`, and
/// a read at `n·v` lands at `sqrt(n·v) * edge`.
///
/// One `sqrt` in the shader, and it is worth more than four times the texels:
/// at 16² it takes the worst error from 8.06 % to 1.45 %. Everything that makes
/// this function hard to interpolate is packed against `n·v = 0`, where a
/// near-mirror seen at the silhouette turns sharply, and a linear axis spends
/// its texels where the function is flat.
fn axis(n_dot_v: f32) -> f32 {
    sim::sqrt(n_dot_v.clamp(0.0, 1.0))
}

/// [Laz13]'s analytic fit — the four constants `include/pbr.slang` shipped
/// through M33, in the shader's own arithmetic.
///
/// Here rather than only in Slang so the instrument can grade the table against
/// the thing it replaced without two builds (`r.shadow_cull`'s argument, §6
/// M32). Returns the scale and bias applied to `f0`: `f0 * a + b`.
#[must_use]
pub fn fit(roughness: f32, n_dot_v: f32) -> (f32, f32) {
    let (c0, c1) = ([-1.0, -0.027_5, -0.572, 0.022], [1.0, 0.042_5, 1.04, -0.04]);
    let r = [
        roughness * c0[0] + c1[0],
        roughness * c0[1] + c1[1],
        roughness * c0[2] + c1[2],
        roughness * c0[3] + c1[3],
    ];
    let a004 = (r[0] * r[0]).min(sim::exp2(-9.28 * n_dot_v)) * r[0] + r[1];
    (-1.04 * a004 + r[2], 1.04 * a004 + r[3])
}

/// The integral itself, by GGX importance sampling — `samples` of it.
///
/// The reference and the generator are the same function: a table texel is this
/// at [`SAMPLES`], and the truth it is graded against is this at a count large
/// enough that the estimator's own noise is below what is being resolved. That
/// is deliberate — a reference implemented separately would be grading two
/// authors rather than one approximation (§6 M33's argument for making `ggc`'s
/// prefilter its own ground truth).
///
/// `roughness` is perceptual and `n_dot_v` in `0..=1`; both are clamped the way
/// the shader clamps them, so a corner of the domain returns what a fragment
/// there would get and not a division by zero.
#[must_use]
pub fn integrate(roughness: f32, n_dot_v: f32, samples: u32) -> (f32, f32) {
    // `make_surface`'s clamp exactly: alpha 0 is a Dirac delta no finite sample
    // count represents, and the shader never presents one.
    let alpha = (roughness * roughness).max(1e-3);
    let a2 = alpha * alpha;
    let n_dot_v = n_dot_v.clamp(1e-4, 1.0);
    let view = sim::Vec3::new(sim::sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    let (mut scale, mut bias) = (0.0f32, 0.0f32);
    for i in 0..samples {
        let (u1, u2) = (i as f32 / samples as f32, radical_inverse(i));
        // GGX's own distribution of half vectors, so the D term and the pdf
        // cancel and what is left is the visibility and the Fresnel split.
        let cos_theta = sim::sqrt((1.0 - u2) / (1.0 + (a2 - 1.0) * u2));
        let sin_theta = sim::sqrt((1.0 - cos_theta * cos_theta).max(0.0));
        let (sin_phi, cos_phi) = sim::sin_cos(core::f32::consts::TAU * u1);
        let half = sim::Vec3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        let v_dot_h = view.dot(half);
        let light = half * (2.0 * v_dot_h) - view;
        if light.z <= 0.0 {
            continue;
        }
        let (n_dot_l, n_dot_h, v_dot_h) = (light.z, half.z.max(1e-7), v_dot_h.clamp(0.0, 1.0));
        // integrand / pdf, with D cancelled: 4 * Vis * NoL * VoH / NoH.
        let weight = 4.0 * visibility(n_dot_v, n_dot_l, alpha) * n_dot_l * v_dot_h / n_dot_h;
        // `f_schlick`'s spherical-Gaussian fifth power and not `(1 - VoH)^5`:
        // the split has to factor the Fresnel the *direct* path applies, or a
        // metal's ambient and its highlight disagree about their own f0.
        let fc = sim::exp2((-5.554_73 * v_dot_h - 6.983_16) * v_dot_h);
        scale += (1.0 - fc) * weight;
        bias += fc * weight;
    }
    let n = samples as f32;
    (scale / n, bias / n)
}

/// The same albedo by **uniform hemisphere sampling** — the cross-check, and
/// the only reason [`integrate`] is believable.
///
/// [`integrate`] samples GGX's own half-vector distribution and cancels `D`
/// against the pdf analytically; this one samples directions uniformly and
/// evaluates the BRDF as written. They share [`d_ggx`] and [`visibility`] — the
/// two terms that mirror the shader and *should* be shared — and share none of
/// the sampling algebra, which is where an importance-sampled estimator goes
/// wrong. Returns `E` at `f0 = 1` directly rather than a scale/bias pair,
/// because with `F ≡ 1` the split does not exist.
///
/// Converges badly below roughness ~0.3: a near-delta lobe is what importance
/// sampling exists for, and a uniform estimator needs an unreasonable count to
/// find it. Read it where it converges and read [`integrate`] everywhere.
#[must_use]
pub fn integrate_uniform(roughness: f32, n_dot_v: f32, samples: u32) -> f32 {
    let alpha = (roughness * roughness).max(1e-3);
    let n_dot_v = n_dot_v.clamp(1e-4, 1.0);
    let view = sim::Vec3::new(sim::sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    let mut total = 0.0f32;
    for i in 0..samples {
        let (u1, u2) = (i as f32 / samples as f32, radical_inverse(i));
        // Uniform on the hemisphere: z is the uniform coordinate, not the angle.
        let cos_theta = u2;
        let sin_theta = sim::sqrt((1.0 - cos_theta * cos_theta).max(0.0));
        let (sin_phi, cos_phi) = sim::sin_cos(core::f32::consts::TAU * u1);
        let light = sim::Vec3::new(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
        let Some(half) = (light + view).try_normalize() else {
            continue;
        };
        let (n_dot_l, n_dot_h) = (light.z, half.z.max(0.0));
        total += d_ggx(n_dot_h, alpha) * visibility(n_dot_v, n_dot_l, alpha) * n_dot_l;
    }
    // pdf is 1/2π, uniform.
    core::f32::consts::TAU * total / samples as f32
}

/// The table the renderer uploads: `EXTENT * EXTENT` pairs, `n·v` fastest.
#[must_use]
pub fn table() -> Vec<[f32; 2]> {
    let edge = (EXTENT - 1) as f32;
    let mut out = Vec::with_capacity(EXTENT * EXTENT);
    for y in 0..EXTENT {
        for x in 0..EXTENT {
            // [`axis`] inverted: the texel's own `n·v`.
            let v = x as f32 / edge;
            let (a, b) = integrate(y as f32 / edge, v * v, SAMPLES);
            out.push([a, b]);
        }
    }
    out
}

/// What the shader's bilinear read of `table` returns — the same arithmetic, on
/// the host, so the instrument grades the *sampled* table rather than its texels.
///
/// Panics if `table` is shorter than [`EXTENT`] squared.
#[must_use]
pub fn sample(table: &[[f32; 2]], roughness: f32, n_dot_v: f32) -> (f32, f32) {
    let edge = (EXTENT - 1) as f32;
    let (fx, fy) = (axis(n_dot_v) * edge, roughness.clamp(0.0, 1.0) * edge);
    let (x0, y0) = (fx.floor().min(edge - 1.0), fy.floor().min(edge - 1.0));
    let (tx, ty) = (fx - x0, fy - y0);
    let at = |x: f32, y: f32| table[y as usize * EXTENT + x as usize];
    let (c00, c10) = (at(x0, y0), at(x0 + 1.0, y0));
    let (c01, c11) = (at(x0, y0 + 1.0), at(x0 + 1.0, y0 + 1.0));
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    (
        lerp(lerp(c00[0], c10[0], tx), lerp(c01[0], c11[0], tx), ty),
        lerp(lerp(c00[1], c10[1], tx), lerp(c01[1], c11[1], tx), ty),
    )
}

/// GGX's normal distribution — `d_ggx` in the shader, term for term.
fn d_ggx(n_dot_h: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    a2 / (core::f32::consts::PI * d * d).max(1e-7)
}

/// Height-correlated Smith visibility — `v_smith_correlated` in the shader,
/// term for term.
fn visibility(n_dot_v: f32, n_dot_l: f32, alpha: f32) -> f32 {
    let a2 = alpha * alpha;
    let lambda_v = n_dot_l * sim::sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    let lambda_l = n_dot_v * sim::sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    0.5 / (lambda_v + lambda_l).max(1e-7)
}

/// Van der Corput radical inverse, base 2 — the second Hammersley coordinate.
///
/// Integer bit reversal, so it is exact on every host and needs no `sim`.
pub(crate) fn radical_inverse(bits: u32) -> f32 {
    let mut b = bits.rotate_left(16);
    b = ((b & 0x5555_5555) << 1) | ((b & 0xaaaa_aaaa) >> 1);
    b = ((b & 0x3333_3333) << 2) | ((b & 0xcccc_cccc) >> 2);
    b = ((b & 0x0f0f_0f0f) << 4) | ((b & 0xf0f0_f0f0) >> 4);
    b = ((b & 0x00ff_00ff) << 8) | ((b & 0xff00_ff00) >> 8);
    b as f32 * 2.328_306_4e-10
}
