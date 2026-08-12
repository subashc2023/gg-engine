//! The environment compiler (§6 M27): the projection, the chain, and the two
//! mappings that are written twice and must agree.
//!
//! What these cannot check is the *shader's* half of each duplicated pair — the
//! SH basis and the octahedral mapping both exist a second time in
//! `include/pbr.slang`, and nothing links both. What they check instead is that
//! this half has the properties the textbook forms have, which is what a drifted
//! copy would lose.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_assets::TextureFormat;
use gg_assets::texture;
use ggc::environment::{self, EXTENT, LEVELS};

/// A panorama that is one constant colour everywhere.
fn flat(color: [f32; 3], width: u32, height: u32) -> Vec<[f32; 3]> {
    vec![color; (width * height) as usize]
}

/// A constant environment is the one case with a closed form: every band but the
/// first integrates to zero against a constant, and band 0 lands at
/// `radiance * sqrt(4π)`.
///
/// This is the whole quadrature in one assertion — a solid-angle weight that
/// forgot its `sin θ` would over-count the poles and miss this by 20%, and a
/// basis whose normalization drifted would miss it by whatever it drifted.
#[test]
fn a_constant_panorama_projects_to_band_zero_alone() {
    let compiled = environment::compile(&flat([1.0, 1.0, 1.0], 64, 32), 64, 32).unwrap();
    // band0 = ∫ L Y0 dω = L * Y0 * 4π, and Y0 is the constant 0.2820948.
    let band0 = 0.282_094_79 * 4.0 * core::f32::consts::PI;
    for channel in 0..3 {
        assert!(
            (compiled.sh[0][channel] - band0).abs() < band0 * 1e-3,
            "band 0 channel {channel}: {} is not {band0}",
            compiled.sh[0][channel]
        );
    }
    for (i, coefficient) in compiled.sh.iter().enumerate().skip(1) {
        for (channel, value) in coefficient[..3].iter().enumerate() {
            assert!(
                value.abs() < band0 * 1e-3,
                "coefficient {i} channel {channel} should vanish, got {value}"
            );
        }
    }
}

/// Colour survives the projection channel by channel. A projection that summed
/// the three or indexed one twice would still pass the constant test above on a
/// grey input, which is exactly why this one is not grey.
#[test]
fn the_projection_keeps_channels_apart() {
    let compiled = environment::compile(&flat([1.0, 0.5, 0.25], 64, 32), 64, 32).unwrap();
    let ratio = |a: usize, b: usize| compiled.sh[0][a] / compiled.sh[0][b];
    assert!((ratio(0, 1) - 2.0).abs() < 1e-2, "red is twice green");
    assert!((ratio(1, 2) - 2.0).abs() < 1e-2, "green is twice blue");
}

/// An environment brighter in the upper hemisphere lands in the coefficient that
/// is odd in **y**, with the sign that says "up".
///
/// The one assertion here that would catch a *flipped* panorama: the importer
/// reads row 0 as the +Y pole, and reading it as the -Y pole would leave every
/// other test in this file passing and every scene lit upside down.
#[test]
fn a_bright_top_lands_in_the_y_band_pointing_up() {
    let (w, h) = (64u32, 32u32);
    let mut texels = flat([0.1; 3], w, h);
    for y in 0..h / 2 {
        for x in 0..w {
            texels[(y * w + x) as usize] = [4.0; 3];
        }
    }
    let compiled = environment::compile(&texels, w, h).unwrap();
    // `basis[1]` is 0.4886 * d.y — positive when the bright half is up.
    assert!(
        compiled.sh[1][0] > 0.0,
        "row 0 is the +Y pole, so a bright top is a positive y band: {}",
        compiled.sh[1][0]
    );
    // And the two bands odd in x and z stay put: the input varies in latitude
    // only, so a nonzero here is a longitude leak.
    for i in [2, 3] {
        assert!(
            compiled.sh[i][0].abs() < compiled.sh[1][0] * 1e-2,
            "coefficient {i} should vanish for an altitude-only panorama"
        );
    }
}

/// The chain is a readable KTX2 of the declared format and level count, and its
/// levels halve — the properties `gg_assets::texture::read` and the residency
/// path both assume without asserting.
#[test]
fn the_chain_is_a_complete_ktx2_at_the_declared_extent() {
    let compiled = environment::compile(&flat([1.0, 1.0, 1.0], 64, 32), 64, 32).unwrap();
    let read = texture::Texture::read(&compiled.radiance).unwrap();
    assert_eq!(read.format, TextureFormat::Bc6hUfloat);
    assert_eq!(read.width, EXTENT);
    assert_eq!(read.height, EXTENT);
    assert_eq!(read.level_count(), LEVELS);
    for level in 0..LEVELS {
        let (w, h) = read.extent(level);
        assert_eq!((w, h), (EXTENT >> level, EXTENT >> level));
        assert_eq!(
            read.level(level).unwrap().unwrap().len() as u64,
            TextureFormat::Bc6hUfloat.level_bytes(w, h),
            "level {level} is not the size its extent says"
        );
    }
}

/// Compiling twice gives the same bytes — §4.6's byte reproducibility, asserted
/// on the one path in the pipeline that is floating point end to end.
///
/// The Hammersley set is what makes this true rather than lucky: a random sample
/// set would fail here, which is why the importance sampler is deterministic by
/// construction and not by seeding.
#[test]
fn the_same_panorama_compiles_to_the_same_bytes() {
    let texels = flat([1.0, 0.5, 0.25], 64, 32);
    let once = environment::compile(&texels, 64, 32).unwrap();
    let twice = environment::compile(&texels, 64, 32).unwrap();
    assert_eq!(once.radiance, twice.radiance);
    assert_eq!(once.sh, twice.sh);
}

/// A panorama with no texels, and one whose buffer disagrees with its extent.
/// Both are refused rather than compiled into something plausible.
#[test]
fn a_malformed_panorama_is_refused_by_name() {
    assert!(environment::compile(&[], 0, 0).is_err());
    let short = flat([1.0; 3], 4, 4);
    let err = environment::compile(&short, 64, 32)
        .unwrap_err()
        .to_string();
    assert!(err.contains("2048 texels"), "{err}");
}
