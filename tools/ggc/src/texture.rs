//! Images → block-compressed KTX2 (§4.6).
//!
//! **The material slot decides everything.** Format, channel packing, filter
//! space and the asset's own name all come from *which slot referenced the
//! image*, never from the file. That is what §4.6 means by "sRGB/linear decided
//! at import, never guessed at runtime": the same PNG referenced as base colour
//! and as an occlusion map compiles twice, to two assets, because it is two
//! different measurements that happen to share a source.
//!
//! **Mip generation is integer for everything but normals.** A box filter over
//! sRGB bytes is the classic wrong mip — averaging encoded values darkens the
//! result, and the error grows with every level — so the sRGB path goes through
//! a 16-bit linear table, averages there, and comes back through a second
//! table. Fixed-point throughout, so the chain is bit-identical on every host
//! rather than identical-if-the-libm-agrees, which is what a byte-reproducible
//! pack (§4.6) needs from a filter that runs on two operating systems.

use std::sync::LazyLock;

use anyhow::{Result, ensure};
use gg_assets::TextureFormat;
use gg_assets::texture;
use intel_tex_2::{RSurface, RgSurface, RgbaSurface, bc4, bc5, bc7};

/// Which material slot referenced an image, and therefore how it compiles.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    /// glTF `baseColorTexture`: the one role whose texels are a colour.
    BaseColor,
    /// glTF `normalTexture`: tangent-space x and y, z reconstructed in the
    /// shader from `sqrt(1 - x² - y²)`, which is why the blue channel is not
    /// stored and not missed.
    Normal,
    /// glTF `metallicRoughnessTexture`. glTF puts roughness in green and
    /// metalness in blue; BC5 has two channels and they arrive as red and
    /// green, so the pair is repacked here and the shader reads `.rg`.
    MetallicRoughness,
    /// glTF `occlusionTexture`: red only.
    Occlusion,
}

impl Role {
    /// The name suffix an asset of this role carries. Part of the id, so a
    /// glTF that points two slots at one image gets two assets rather than one
    /// asset compiled two incompatible ways.
    #[must_use]
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::BaseColor => "base_color",
            Self::Normal => "normal",
            Self::MetallicRoughness => "metallic_roughness",
            Self::Occlusion => "occlusion",
        }
    }

    /// The block format this role compiles to.
    #[must_use]
    pub const fn format(self) -> TextureFormat {
        match self {
            Self::BaseColor => TextureFormat::Bc7Srgb,
            Self::Normal | Self::MetallicRoughness => TextureFormat::Bc5Unorm,
            Self::Occlusion => TextureFormat::Bc4Unorm,
        }
    }

    /// Which source channels end up in the compressed texels, in order.
    const fn channels(self) -> &'static [usize] {
        match self {
            Self::BaseColor => &[0, 1, 2, 3],
            Self::Normal => &[0, 1],
            // Roughness then metalness: glTF's green and blue, in that order.
            Self::MetallicRoughness => &[1, 2],
            Self::Occlusion => &[0],
        }
    }

    const fn filter(self) -> Filter {
        match self {
            Self::BaseColor => Filter::Srgb,
            Self::Normal => Filter::Normal,
            Self::MetallicRoughness | Self::Occlusion => Filter::Linear,
        }
    }
}

/// How a level is halved. Not a cosmetic choice: each is the only correct
/// average for what the channel *is*.
#[derive(Clone, Copy)]
enum Filter {
    /// Decode to linear, average, re-encode. Alpha is already linear.
    Srgb,
    /// Average the bytes.
    Linear,
    /// Average as a unit vector and renormalize — a box filter over normals
    /// shortens them, and a short normal is a surface that reads as darker with
    /// every mip level rather than as smoother.
    Normal,
}

/// Compile one decoded RGBA8 image into a texture blob with a full mip chain.
pub fn compile(rgba: &[u8], width: u32, height: u32, role: Role) -> Result<Vec<u8>> {
    ensure!(
        width > 0 && height > 0,
        "a {width}x{height} image has no texels to compress"
    );
    let expected = width as usize * height as usize * 4;
    ensure!(
        rgba.len() == expected,
        "a {width}x{height} RGBA8 image is {expected} bytes, not {}",
        rgba.len()
    );

    let settings = role.settings(rgba);
    let mut levels = Vec::with_capacity(texture::full_mip_count(width, height) as usize);
    let (mut texels, mut w, mut h) = (rgba.to_vec(), width, height);
    loop {
        levels.push(compress(&texels, w, h, role, settings.as_ref()));
        if w == 1 && h == 1 {
            break;
        }
        texels = downsample(&texels, w, h, role.filter());
        (w, h) = ((w / 2).max(1), (h / 2).max(1));
    }
    debug_assert_eq!(levels.len() as u32, texture::full_mip_count(width, height));
    Ok(texture::encode(role.format(), width, height, &levels)?)
}

impl Role {
    /// BC7's encoder settings, or `None` for the roles that do not use BC7.
    ///
    /// `basic` is the middle of the five presets — chosen rather than measured,
    /// and the thing to move first if §6 M9's 30 s cold budget for Sponza turns
    /// out to be spent here. Opaque images take the cheaper mode set: with no
    /// alpha to fit, the modes that carry one are wasted search.
    fn settings(self, rgba: &[u8]) -> Option<bc7::EncodeSettings> {
        if self != Self::BaseColor {
            return None;
        }
        Some(if rgba.chunks_exact(4).all(|texel| texel[3] == u8::MAX) {
            bc7::opaque_basic_settings()
        } else {
            bc7::alpha_basic_settings()
        })
    }
}

/// Block-compress one level.
fn compress(
    rgba: &[u8],
    w: u32,
    h: u32,
    role: Role,
    settings: Option<&bc7::EncodeSettings>,
) -> Vec<u8> {
    // The ISPC kernels take whole blocks only, and their own `calc_output_size`
    // rounds `w * h` rather than each axis — wrong for anything not already
    // block-aligned. Padding here rather than trusting it keeps both problems in
    // one place.
    let (pw, ph) = (w.next_multiple_of(4), h.next_multiple_of(4));
    let channels = role.channels();
    let data = gather(rgba, w, h, pw, ph, channels);
    let stride = pw * channels.len() as u32;
    match role.format() {
        TextureFormat::Bc7Srgb => bc7::compress_blocks(
            settings.unwrap_or(&bc7::opaque_basic_settings()),
            &RgbaSurface {
                data: &data,
                width: pw,
                height: ph,
                stride,
            },
        ),
        TextureFormat::Bc5Unorm => bc5::compress_blocks(&RgSurface {
            data: &data,
            width: pw,
            height: ph,
            stride,
        }),
        TextureFormat::Bc4Unorm => bc4::compress_blocks(&RSurface {
            data: &data,
            width: pw,
            height: ph,
            stride,
        }),
    }
}

/// Pull `channels` out of every texel into a tightly packed buffer, padded out
/// to the block grid by replicating the edge.
///
/// Replicated rather than zeroed: a padded texel still takes part in its
/// block's endpoint fit, and a black column dragged into the fit shows as a
/// dark seam along an edge whose texels were never black.
fn gather(rgba: &[u8], w: u32, h: u32, pw: u32, ph: u32, channels: &[usize]) -> Vec<u8> {
    let mut out = Vec::with_capacity((pw * ph) as usize * channels.len());
    for y in 0..ph {
        let sy = y.min(h - 1) as usize;
        for x in 0..pw {
            let at = (sy * w as usize + x.min(w - 1) as usize) * 4;
            out.extend(channels.iter().map(|&c| rgba[at + c]));
        }
    }
    out
}

/// Halve a level, in whatever space the role's channels live in.
fn downsample(src: &[u8], w: u32, h: u32, filter: Filter) -> Vec<u8> {
    let (dw, dh) = ((w / 2).max(1), (h / 2).max(1));
    let mut out = Vec::with_capacity((dw * dh) as usize * 4);
    for y in 0..dh {
        // An axis already at one texel samples the same row twice, which is the
        // 2x1 and 1x1 tail of a non-square chain averaging with itself.
        let (y0, y1) = ((2 * y).min(h - 1), (2 * y + 1).min(h - 1));
        for x in 0..dw {
            let (x0, x1) = ((2 * x).min(w - 1), (2 * x + 1).min(w - 1));
            let texel = |cx: u32, cy: u32| {
                let at = (cy as usize * w as usize + cx as usize) * 4;
                [src[at], src[at + 1], src[at + 2], src[at + 3]]
            };
            let quad = [texel(x0, y0), texel(x1, y0), texel(x0, y1), texel(x1, y1)];
            out.extend_from_slice(&blend(&quad, filter));
        }
    }
    out
}

/// Average four texels.
fn blend(quad: &[[u8; 4]; 4], filter: Filter) -> [u8; 4] {
    let mean = |c: usize| {
        let sum: u32 = quad.iter().map(|t| u32::from(t[c])).sum();
        ((sum + 2) / 4) as u8
    };
    match filter {
        Filter::Linear => [mean(0), mean(1), mean(2), mean(3)],
        Filter::Srgb => {
            // Integer throughout: the tables are the only float work, and they
            // are built once. Alpha is a coverage value, not a colour, and
            // glTF says so — it is averaged where it lies.
            let linear = |c: usize| {
                let sum: u32 = quad
                    .iter()
                    .map(|t| u32::from(SRGB_TO_LINEAR[t[c] as usize]))
                    .sum();
                LINEAR_TO_SRGB[((sum + 2) / 4) as usize]
            };
            [linear(0), linear(1), linear(2), mean(3)]
        }
        Filter::Normal => {
            let mut sum = [0.0f32; 3];
            for texel in quad {
                for (axis, total) in sum.iter_mut().enumerate() {
                    // 0..255 maps to -1..1 the way the shader will read it back.
                    *total += f32::from(texel[axis]) / 127.5 - 1.0;
                }
            }
            let length = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
            // A quad of opposing normals cancels; leaving it at zero would put a
            // hole in the mip, so it falls back to the surface's own +z.
            let unit = if length > 1e-6 {
                [sum[0] / length, sum[1] / length, sum[2] / length]
            } else {
                [0.0, 0.0, 1.0]
            };
            let encode = |v: f32| (((v + 1.0) * 127.5) + 0.5).clamp(0.0, 255.0) as u8;
            [encode(unit[0]), encode(unit[1]), encode(unit[2]), mean(3)]
        }
    }
}

/// sRGB byte → linear, in 16-bit fixed point. Exact for all 256 inputs, so the
/// decode half of the filter loses nothing.
static SRGB_TO_LINEAR: LazyLock<[u16; 256]> = LazyLock::new(|| {
    core::array::from_fn(|encoded| {
        let c = encoded as f32 / 255.0;
        let linear = if c <= 0.040_45 {
            c / 12.92
        } else {
            // libm rather than the CRT's: this table is upstream of every mip
            // texel in the pack, and two hosts disagreeing in the last bit
            // would make a pack built on Windows differ from one built on
            // Linux (§4.6).
            gg_math::sim::powf((c + 0.055) / 1.055, 2.4)
        };
        (linear * 65_535.0 + 0.5) as u16
    })
});

/// Linear 16-bit fixed point → sRGB byte. 64 KiB, built once, and the reason
/// the filter's inner loop touches no float at all.
static LINEAR_TO_SRGB: LazyLock<Box<[u8; 65_536]>> = LazyLock::new(|| {
    let mut table = Box::new([0u8; 65_536]);
    for (fixed, out) in table.iter_mut().enumerate() {
        let c = fixed as f32 / 65_535.0;
        let encoded = if c <= 0.003_130_8 {
            c * 12.92
        } else {
            1.055 * gg_math::sim::powf(c, 1.0 / 2.4) - 0.055
        };
        *out = (encoded * 255.0 + 0.5) as u8;
    }
    table
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_srgb_tables_are_inverses_of_each_other() {
        // Round-tripping every byte is what makes the mip filter lossless on a
        // flat region: a level whose texels are all one colour must produce a
        // level whose texels are that same colour, or a solid texture drifts
        // as it shrinks.
        for encoded in 0..=255u8 {
            let linear = SRGB_TO_LINEAR[encoded as usize];
            assert_eq!(
                LINEAR_TO_SRGB[linear as usize], encoded,
                "sRGB {encoded} round-tripped to {}",
                LINEAR_TO_SRGB[linear as usize]
            );
        }
    }

    #[test]
    fn a_flat_colour_survives_every_level_of_the_chain() {
        let flat = [180u8, 90, 40, 255].repeat(16 * 16);
        let down = downsample(&flat, 16, 16, Filter::Srgb);
        assert_eq!(down.len(), 8 * 8 * 4);
        assert!(down.chunks_exact(4).all(|t| t == [180, 90, 40, 255]));
    }

    #[test]
    fn halving_black_and_white_lands_where_light_does_not_where_bytes_do() {
        // Half black, half white, averaged in linear light, is ~188 in sRGB —
        // the mid-grey of the *encoded* bytes would be 128, which is the mip
        // bug this filter exists to avoid.
        let checker = [[0u8, 0, 0, 255], [255, 255, 255, 255]].concat().repeat(2);
        let down = downsample(&checker, 2, 2, Filter::Srgb);
        assert_eq!(down.len(), 4);
        assert!(
            (187..=189).contains(&down[0]),
            "linear-light average should be ~188, got {}",
            down[0]
        );

        let bytes = downsample(&checker, 2, 2, Filter::Linear);
        assert_eq!(
            bytes[0], 128,
            "the linear filter averages the bytes as given"
        );
    }

    #[test]
    fn normals_stay_unit_length_through_a_halving() {
        // Two normals 90° apart average to one 45° between them — of unit
        // length, not of the shorter length a byte-wise average would give.
        let left = [0u8, 128, 128, 255];
        let up = [128, 128, 255, 255];
        let quad = [left, up, left, up].concat();
        let down = downsample(&quad, 2, 2, Filter::Normal);
        let v: Vec<f32> = down[..3]
            .iter()
            .map(|&b| f32::from(b) / 127.5 - 1.0)
            .collect();
        let length = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((length - 1.0).abs() < 0.02, "normal length {length}");
    }

    #[test]
    fn the_slot_decides_the_format_and_the_channels() {
        assert_eq!(Role::BaseColor.format(), TextureFormat::Bc7Srgb);
        assert!(Role::BaseColor.format().is_srgb());
        assert_eq!(Role::Normal.format(), TextureFormat::Bc5Unorm);
        assert!(!Role::Normal.format().is_srgb());
        assert_eq!(Role::Occlusion.format(), TextureFormat::Bc4Unorm);
        // glTF's green and blue become BC5's red and green, in that order.
        assert_eq!(Role::MetallicRoughness.channels(), &[1, 2]);
    }

    #[test]
    fn a_non_block_aligned_image_pads_by_replicating_its_edge() {
        // 6x6 is two blocks across and two down, with the last two texels of
        // each replicated — the copy the ISPC kernels cannot do themselves.
        let mut rgba = vec![0u8; 6 * 6 * 4];
        for (i, texel) in rgba.chunks_exact_mut(4).enumerate() {
            texel[0] = i as u8;
        }
        let padded = gather(&rgba, 6, 6, 8, 8, &[0]);
        assert_eq!(padded.len(), 64);
        assert_eq!(padded[5], 5, "the last real texel of row 0");
        assert_eq!(padded[6], 5, "and the two padded ones repeat it");
        assert_eq!(padded[7], 5);
        assert_eq!(&padded[48..56], &padded[56..64], "the last row, twice");
    }
}
