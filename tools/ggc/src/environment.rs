//! An equirectangular `.hdr` → an order-2 SH projection and a prefiltered
//! octahedral radiance chain (§6 M27).
//!
//! # Why both halves are computed here and not at load
//!
//! Prefiltering is the split-sum's first integral, and it is the expensive one:
//! every texel of every level is a few dozen importance samples of the source.
//! Doing it once, at import, makes it a cost the pack already amortizes (§4.6's
//! incremental rebuild reuses the blob whenever the `.hdr` and this compiler are
//! unchanged), makes it byte-reproducible like everything else in a pack, and
//! keeps the runtime with no compute pass, no storage images and no
//! mip-generation path — none of which exist in `gg-rhi` and none of which this
//! buys anything by adding.
//!
//! # Why octahedral and not a cube
//!
//! A cubemap is six array layers and a `VK_IMAGE_VIEW_TYPE_CUBE`, and §4.3's
//! bindless set is one array of `Texture2D`. An octahedral map *is* a 2D
//! texture: it costs a descriptor write, and adding a texture is supposed to
//! cost exactly that. The distortion is also slightly kinder than a cube's — the
//! worst-case solid angle per texel runs about 1.2x the mean where a cube's
//! corner reaches 1.5x.
//!
//! What it costs is the seam. The octahedron's outer edge is a fold, so a
//! bilinear tap that straddles it reads the wrong hemisphere. That is answered
//! by construction rather than by a border: **every level is integrated over
//! the sphere from the source**, not downsampled from the level above, so a
//! texel near the fold is filtered from the directions actually around it. Only
//! the final bilinear tap in the shader can straddle, and at these lobe widths
//! it straddles between two nearly equal values.
//!
//! # References
//!
//!   [Kar13] Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013 — the
//!           split sum, the `n = v = r` approximation, and the GGX importance
//!           sample below.
//!   [Cig14] Cigolle et al., "A Survey of Efficient Representations for
//!           Independent Unit Vectors", JCGT 2014 — the octahedral mapping.

use anyhow::{Result, ensure};
use gg_assets::{TextureFormat, texture};
use gg_math::render::Vec3;
// `sim`'s transcendentals and not `std`'s, and here the §3 ban earns more than
// it does in the engine: `libm` is correctly rounded and identical on every
// host, so a pack compiled on this desk and one compiled in WSL agree *byte for
// byte* rather than agreeing only with themselves. §4.6's reproducibility gate
// builds twice on one machine and could never have caught the difference.
use gg_math::sim;
use intel_tex_2::{RgbaSurface, bc6h};

/// Coefficients in the order-2 projection. **Must equal
/// `gg_assets::environment::SH_COEFFICIENTS`.**
const SH_COEFFICIENTS: usize = gg_assets::environment::SH_COEFFICIENTS;

/// The octahedral chain's base extent.
///
/// 256 square holds about as much detail as a 128 cube face, which is the size
/// the split-sum's first integral is conventionally stored at — past it the
/// mirror level is the only one that can carry the extra texels, and this
/// renderer's mirror is bounded by the source's own resolution rather than by
/// this. The whole chain is 87 KiB of BC6H.
pub const EXTENT: u32 = 256;

/// Levels in the chain: 256 down to 4. The last is two blocks square, which is
/// the smallest extent whose block fit still has interior texels to fit.
pub const LEVELS: u32 = 7;

/// The working resolution the source is resampled to before any integral runs.
///
/// Every lookup below is a *filtered* one — an importance sample of a lobe, or
/// a mirror tap that a 256-texel octahedral level cannot out-resolve anyway —
/// so the full-resolution source is only ever a source of fireflies here. A
/// 1024x512 box-average of it is bounded work regardless of what an artist
/// exported, which is what keeps a 16k panorama from turning import into a
/// minute.
const WORKING: (u32, u32) = (1024, 512);

/// Importance samples per texel at the roughest level, scaled down toward the
/// mirror where the lobe collapses to a point.
///
/// The count is per *level* rather than fixed because the two ends want opposite
/// things: level 0 is a mirror and one tap is exact, while the roughest level
/// integrates most of a hemisphere and a thin sample set there reads as blotches
/// on a smooth surface — the one artifact of this whole path a viewer would name
/// unprompted.
const MAX_SAMPLES: u32 = 256;

/// What one `.hdr` compiled to.
#[derive(Debug)]
pub struct Compiled {
    /// The order-2 projection, in linear radiance, laid out as the frame block
    /// wants it. The **same basis and the same normalization** as
    /// `gg_render::sky::basis`, which is the only thing making these
    /// interchangeable with a gradient sky's.
    pub sh: [[f32; 4]; SH_COEFFICIENTS],
    /// The prefiltered chain, as a KTX2 blob.
    pub radiance: Vec<u8>,
}

/// Compile one equirectangular float image.
///
/// `source` is `width * height` linear RGB triples, row 0 at the *top* — the
/// convention every `.hdr` exporter writes and `KTXorientation rd` records.
///
/// # Errors
/// An image with no texels, or one whose buffer disagrees with its extent.
pub fn compile(source: &[[f32; 3]], width: u32, height: u32) -> Result<Compiled> {
    ensure!(
        width > 0 && height > 0,
        "a {width}x{height} panorama has no texels"
    );
    let expected = width as usize * height as usize;
    ensure!(
        source.len() == expected,
        "a {width}x{height} panorama is {expected} texels, not {}",
        source.len()
    );

    // The SH projection reads the *source*, not the working copy: it is an
    // integral over the whole sphere weighted by solid angle, so it costs one
    // pass over the texels that exist and resampling first would only throw
    // energy away before integrating it.
    let sh = project(source, width, height);
    let working = resample(source, width, height);
    let radiance = prefilter(&working)?;
    Ok(Compiled { sh, radiance })
}

/// The order-2 real SH basis, bands 0..=2.
///
/// **The same literals as `gg_render::sky::basis` and `pbr.slang`'s `sh_basis`,
/// deliberately.** Three copies, no shared crate: `ggc` is not in the shipping
/// graph, the renderer is, and the shader is neither — what makes them agree is
/// that all three are the textbook polynomials and none is derived from another.
/// `the_basis_is_orthonormal` is what would catch this copy drifting.
fn basis(d: Vec3) -> [f32; SH_COEFFICIENTS] {
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

/// Project the panorama onto the basis, weighting each texel by its solid angle.
///
/// Exact quadrature over the texels rather than a Monte Carlo point set: an
/// equirectangular image *is* a partition of the sphere, so the integral is a
/// weighted sum with no sampling residual at all. `gg_render::sky` samples
/// instead because its environment is a function it can evaluate anywhere and
/// has no texels to sum over.
fn project(source: &[[f32; 3]], width: u32, height: u32) -> [[f32; 4]; SH_COEFFICIENTS] {
    let mut out = [[0.0f32; 4]; SH_COEFFICIENTS];
    // dφ dθ, constant across the image; the sin θ that completes the solid angle
    // is per row and applied there.
    let cell = (core::f32::consts::TAU / width as f32) * (core::f32::consts::PI / height as f32);
    for y in 0..height {
        let v = (y as f32 + 0.5) / height as f32;
        let theta = v * core::f32::consts::PI;
        let weight = cell * sim::sin(theta);
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let texel = source[(y * width + x) as usize];
            let basis = basis(direction_of(u, v));
            for (coefficient, y) in out.iter_mut().zip(basis) {
                let scale = weight * y;
                coefficient[0] += texel[0] * scale;
                coefficient[1] += texel[1] * scale;
                coefficient[2] += texel[2] * scale;
            }
        }
    }
    out
}

/// The direction an equirectangular `(u, v)` names. `v` runs from the +Y pole
/// down, which is what "row 0 at the top" means once the image is a sphere.
///
/// Paired with [`uv_of`], and the pair is what fixes the sky's *orientation*:
/// any self-consistent choice is correct, and a disagreement between the two is
/// a panorama that reflects a rotated copy of itself.
fn direction_of(u: f32, v: f32) -> Vec3 {
    let theta = v * core::f32::consts::PI;
    let phi = u * core::f32::consts::TAU - core::f32::consts::PI;
    let (sin_theta, cos_theta) = sim::sin_cos(theta);
    let (sin_phi, cos_phi) = sim::sin_cos(phi);
    Vec3::new(sin_theta * sin_phi, cos_theta, -sin_theta * cos_phi)
}

/// [`direction_of`] inverted, for a unit `d`.
fn uv_of(d: Vec3) -> (f32, f32) {
    let v = sim::acos(d.y.clamp(-1.0, 1.0)) / core::f32::consts::PI;
    let phi = sim::atan2(d.x, -d.z);
    let u = (phi + core::f32::consts::PI) / core::f32::consts::TAU;
    (u, v)
}

/// Box-average the source down to [`WORKING`].
///
/// A whole-texel box average — every source texel lands in exactly one
/// destination — rather than a resampling filter, because the input is a
/// radiance and the only property that must survive is its integral. A kernel
/// with negative lobes would ring around a sun and put *negative* radiance in
/// the ring, which the prefilter would then dutifully spread.
fn resample(source: &[[f32; 3]], width: u32, height: u32) -> Vec<[f32; 3]> {
    let (dw, dh) = WORKING;
    let mut out = vec![[0.0f32; 3]; (dw * dh) as usize];
    let mut counts = vec![0u32; (dw * dh) as usize];
    for y in 0..height {
        // Nearest-destination assignment: with `dw` and `dh` fixed powers of two
        // and any source extent, this is the partition a box filter needs and
        // costs no division per texel beyond these two.
        let dy = (y as u64 * dh as u64 / height as u64).min(dh as u64 - 1) as u32;
        for x in 0..width {
            let dx = (x as u64 * dw as u64 / width as u64).min(dw as u64 - 1) as u32;
            let to = (dy * dw + dx) as usize;
            let texel = source[(y * width + x) as usize];
            for c in 0..3 {
                out[to][c] += texel[c];
            }
            counts[to] += 1;
        }
    }
    // A destination texel with no source texel happens only when the source is
    // smaller than the working copy, where the right answer is the nearest
    // source texel rather than black.
    for (i, (texel, count)) in out.iter_mut().zip(&counts).enumerate() {
        if *count > 0 {
            for c in texel.iter_mut() {
                *c /= *count as f32;
            }
        } else {
            let (x, y) = ((i as u32 % dw), (i as u32 / dw));
            let sx = (x as u64 * width as u64 / dw as u64).min(width as u64 - 1) as u32;
            let sy = (y as u64 * height as u64 / dh as u64).min(height as u64 - 1) as u32;
            *texel = source[(sy * width + sx) as usize];
        }
    }
    out
}

/// Bilinear lookup into the working copy along `direction`.
///
/// Wrapping in longitude and clamped in latitude — the two are different
/// topologies and a single addressing mode would be wrong for one of them: the
/// seam at φ = ±π is continuous and must blend across, while the pole is an
/// edge no tap should walk past.
fn sample(working: &[[f32; 3]], direction: Vec3) -> Vec3 {
    let (dw, dh) = WORKING;
    let (u, v) = uv_of(direction);
    let x = u * dw as f32 - 0.5;
    let y = (v * dh as f32 - 0.5).clamp(0.0, dh as f32 - 1.0);
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let row = |yi: i64| (yi.clamp(0, dh as i64 - 1) as u32 * dw) as usize;
    let col = |xi: i64| xi.rem_euclid(dw as i64) as usize;
    let (r0, r1) = (row(y0 as i64), row(y0 as i64 + 1));
    let (c0, c1) = (col(x0 as i64), col(x0 as i64 + 1));
    let at = |i: usize| Vec3::from_array(working[i]);
    let top = at(r0 + c0).lerp(at(r0 + c1), fx);
    let bottom = at(r1 + c0).lerp(at(r1 + c1), fx);
    top.lerp(bottom, fy)
}

/// Prefilter the working copy into the octahedral chain and encode it.
fn prefilter(working: &[[f32; 3]]) -> Result<Vec<u8>> {
    let mut levels = Vec::with_capacity(LEVELS as usize);
    for level in 0..LEVELS {
        let extent = EXTENT >> level;
        // Roughness across the chain's *whole* span, so the last level is the
        // fully rough end of the axis rather than an arbitrary blur. The shader
        // inverts exactly this to pick a mip.
        let roughness = level as f32 / (LEVELS - 1) as f32;
        let mut texels = vec![[0.0f32; 3]; (extent * extent) as usize];
        for y in 0..extent {
            for x in 0..extent {
                let u = (x as f32 + 0.5) / extent as f32;
                let v = (y as f32 + 0.5) / extent as f32;
                let normal = octahedral_decode(u, v);
                let filtered = if level == 0 {
                    // A mirror integrates a delta, and an importance sample set
                    // of a delta is the same tap N times with worse rounding.
                    sample(working, normal)
                } else {
                    integrate(working, normal, roughness)
                };
                texels[(y * extent + x) as usize] = filtered.to_array();
            }
        }
        levels.push(compress(&texels, extent));
    }
    Ok(texture::encode(
        TextureFormat::Bc6hUfloat,
        EXTENT,
        EXTENT,
        &levels,
    )?)
}

/// The GGX lobe around `normal`, integrated against the environment — the split
/// sum's first term, under [Kar13]'s `n = v = r` assumption.
///
/// That assumption is what makes a single chain enough: the real integral
/// depends on the view direction too, and storing that axis would be a fourth
/// dimension. What it costs is the stretched highlight at grazing angles, which
/// this renderer does not have and a reference render does.
fn integrate(working: &[[f32; 3]], normal: Vec3, roughness: f32) -> Vec3 {
    let alpha = roughness * roughness;
    // Sample count follows the lobe: a narrow one needs few taps to be smooth
    // and a wide one needs many, and spending the roughest level's budget on
    // level 1 would be most of the import time for none of the artifact.
    let samples = (MAX_SAMPLES as f32 * roughness).ceil().max(1.0) as u32;
    let (tangent, bitangent) = basis_around(normal);
    let mut total = Vec3::ZERO;
    let mut weight = 0.0f32;
    for i in 0..samples {
        let (e1, e2) = hammersley(i, samples);
        let half = ggx_half(e1, e2, alpha, tangent, bitangent, normal);
        // `v = n`, so the reflected direction is the half vector mirrored about
        // the normal rather than about a separate view.
        let light = 2.0 * normal.dot(half) * half - normal;
        let n_dot_l = normal.dot(light);
        if n_dot_l <= 0.0 {
            // Below the horizon. Dropped from the *weight* as well as the sum,
            // which is what keeps this a weighted average rather than a
            // darkening — a lobe that half-misses must not return half its
            // radiance.
            continue;
        }
        total += sample(working, light) * n_dot_l;
        weight += n_dot_l;
    }
    if weight > 0.0 {
        total / weight
    } else {
        sample(working, normal)
    }
}

/// Any orthonormal pair perpendicular to `n`. The choice of *which* pair is free
/// — the lobe is rotationally symmetric about `n` — so this takes the branchless
/// one and only guards the degenerate axis.
fn basis_around(n: Vec3) -> (Vec3, Vec3) {
    let up = if n.y.abs() < 0.999 { Vec3::Y } else { Vec3::X };
    let tangent = up.cross(n).normalize();
    (tangent, n.cross(tangent))
}

/// The `i`th of `n` points of the Hammersley set: van der Corput in base 2
/// against a uniform first coordinate.
///
/// A deterministic low-discrepancy set and not a random one, because a pack is
/// byte-reproducible (§4.6) and "compile it twice, get two files" is exactly
/// what a random sequence buys.
fn hammersley(i: u32, n: u32) -> (f32, f32) {
    (
        i as f32 / n as f32,
        i.reverse_bits() as f32 * 2.328_306_4e-10,
    )
}

/// Importance-sample a GGX half vector, `e` uniform in the unit square.
fn ggx_half(e1: f32, e2: f32, alpha: f32, tangent: Vec3, bitangent: Vec3, normal: Vec3) -> Vec3 {
    let phi = core::f32::consts::TAU * e1;
    // The GGX distribution's inverse CDF in cos θ. Clamped below one because
    // e2 = 0 is a legal sample and would otherwise divide by zero at alpha = 0.
    let cos_theta = ((1.0 - e2) / (1.0 + (alpha * alpha - 1.0) * e2))
        .max(0.0)
        .sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let (sin_phi, cos_phi) = sim::sin_cos(phi);
    tangent * (sin_theta * cos_phi) + bitangent * (sin_theta * sin_phi) + normal * cos_theta
}

/// The direction an octahedral `(u, v)` names — [Cig14]'s mapping, folded about
/// **y**, because that is the engine's up and the fold belongs on the axis whose
/// hemispheres a sky actually differs across.
///
/// **Mirrored in `pbr.slang`'s `octahedral_uv`, inverted.** The two never meet
/// at runtime, exactly as the SH basis does not, and the same defence applies.
fn octahedral_decode(u: f32, v: f32) -> Vec3 {
    let (x, z) = (u * 2.0 - 1.0, v * 2.0 - 1.0);
    let y = 1.0 - x.abs() - z.abs();
    // The lower hemisphere is the outer ring, folded across the diagonal edges.
    let fold = (-y).max(0.0);
    let x = x - fold * if x >= 0.0 { 1.0 } else { -1.0 };
    let z = z - fold * if z >= 0.0 { 1.0 } else { -1.0 };
    Vec3::new(x, y, z).normalize()
}

/// Compress one octahedral level to BC6H.
fn compress(texels: &[[f32; 3]], extent: u32) -> Vec<u8> {
    // The ISPC kernel reads RGBA halves; the alpha channel is unread by BC6H and
    // is written as one so a debugger showing the staging buffer shows opaque
    // texels rather than a transparent image that is not transparent.
    let mut halves = Vec::with_capacity((extent * extent) as usize * 4);
    for texel in texels {
        for c in texel {
            halves.push(to_half(*c));
        }
        halves.push(0x3c00); // 1.0
    }
    // Every level here is square and a power of two down to 4, so the block grid
    // divides exactly and the edge-replication `texture::gather` needs has
    // nothing to do.
    debug_assert_eq!(extent % 4, 0);
    bc6h::compress_blocks(
        &bc6h::basic_settings(),
        &RgbaSurface {
            data: bytemuck::cast_slice(&halves),
            width: extent,
            height: extent,
            stride: extent * 8,
        },
    )
}

/// `f32` → unsigned `f16` bits, round-to-nearest-even, clamped into range.
///
/// Written out rather than pulled from a crate: it is twenty lines of integer
/// arithmetic, it is the same twenty lines on every host — which is what a
/// byte-reproducible pack needs — and BC6H's *unsigned* variant makes the
/// negative half of the domain a clamp rather than a case.
fn to_half(x: f32) -> u16 {
    // Negative radiance is not a measurement, and BC6H_UFLOAT cannot hold one.
    // It reaches here from a resampling filter's ringing or from an exporter,
    // and either way zero is the honest answer.
    let x = if x > 0.0 { x } else { 0.0 };
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x007f_ffff;
    if exponent >= 0x1f {
        // Overflow, and infinity too: clamped to the largest finite half rather
        // than to `inf`, because an `inf` texel poisons every block it shares an
        // endpoint fit with — one bad texel would take a 4x4 neighbourhood.
        return 0x7bff;
    }
    if exponent <= 0 {
        // Subnormal, or below the subnormal range entirely.
        if exponent < -10 {
            return 0;
        }
        let mantissa = mantissa | 0x0080_0000;
        let shift = (14 - exponent) as u32;
        let half = mantissa >> shift;
        // Round to nearest, ties to even, on the bits being dropped.
        let dropped = mantissa & ((1 << shift) - 1);
        let midpoint = 1 << (shift - 1);
        let round = u32::from(dropped > midpoint || (dropped == midpoint && half & 1 == 1));
        return (half + round) as u16;
    }
    let half = ((exponent as u32) << 10) | (mantissa >> 13);
    let dropped = mantissa & 0x1fff;
    let round = u32::from(dropped > 0x1000 || (dropped == 0x1000 && half & 1 == 1));
    // The carry out of the mantissa lands in the exponent, which is the encoding
    // doing the right thing rather than an overflow to guard: 0x7bff + 1 is
    // 0x7c00, and that is only reachable from a value already clamped above.
    (half + round) as u16
}
