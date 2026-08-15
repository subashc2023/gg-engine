//! Demo 10's taskbar picture, written where the caller asks (§6 M46).
//!
//! `panorama`'s rule one medium over, and `timbre`'s: a checked-in binary in
//! this tree has a source. An icon is the *worst* case for that rule rather than
//! an incidental one — it is the one artifact a reviewer cannot read in a diff,
//! so the diff has to be the generator or there is no review at all.
//!
//! # What it draws, and why not a screenshot of the game
//!
//! A tetromino, one of the game's own [`SHAPES`] in one of its own [`COLORS`],
//! on the well's background — so the picture is the game's palette by
//! construction rather than by somebody matching it once. Drawn rather than
//! captured because 64 pixels is not a frame scaled down: at that size the
//! board is a grey smear, and what reads is one shape with a wide margin.
//!
//! Everything here is integer arithmetic on `u8` channels. No `gg_math::sim`
//! and no floats at all, which is the cheapest possible answer to the
//! byte-reproducibility question `panorama` had to work for: there is nothing
//! in it a second host could round differently.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use gg_core::config::icon;

/// Which of demo 10's seven shapes. The S piece: the only one that is neither
/// symmetric nor a straight line, so it reads as *a tetromino* at 64 pixels
/// rather than as a rectangle or a square.
const PIECE: usize = 4;

/// A 4x4 block grid with a one-block margin all round, hence six across — which
/// is why [`icon::SIDE`] is 96 and not a power of two. The margin is what keeps
/// the piece clear of the rounding every OS does to a taskbar corner.
const CELLS: u32 = 6;

pub fn run(args: &[String]) -> Result<()> {
    let mut out = PathBuf::from("target/gg-tools/icon.ggicon");
    let mut side = icon::SIDE;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--out" => out = PathBuf::from(rest.next().context("--out wants a path")?),
            "--side" => side = rest.next().context("--side wants a number")?.parse()?,
            other => bail!("unknown argument {other}"),
        }
    }
    if side == 0 || !side.is_multiple_of(CELLS) {
        bail!("--side wants a positive multiple of {CELLS}, not {side}");
    }
    let rgba = draw(side);
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&out, icon::encode(side, &rgba))?;
    println!(
        "gg-tools icon: {side}x{side}, piece {PIECE}, {} bytes -> {}",
        rgba.len(),
        out.display()
    );
    Ok(())
}

/// The pixels: the well's background, a border a shade lighter, and the piece.
fn draw(side: u32) -> Vec<u8> {
    // Demo 10's own palette, read rather than matched — the whole reason this
    // lives in a subcommand that links the game crate.
    let piece = demo_10_tetris::COLORS[PIECE];
    let bits = demo_10_tetris::SHAPES[PIECE][0];
    let cell = side / CELLS;
    let mut rgba = Vec::with_capacity(side as usize * side as usize * 4);
    for y in 0..side {
        for x in 0..side {
            // A block is lit on its top-left edges and shaded on its
            // bottom-right, which is what makes four flat squares read as four
            // *blocks* at this size. One pixel, because at 64 across two is a
            // bevel and three is a frame.
            let (cx, cy) = (x / cell, y / cell);
            let inside = (1..CELLS - 1).contains(&cx) && (1..CELLS - 1).contains(&cy);
            let lit = inside && (x % cell == 0 || y % cell == 0);
            let shaded = inside && (x % cell == cell - 1 || y % cell == cell - 1);
            let filled = inside && {
                // `SHAPES` is a 4x4 bitmask, row-major from the top left.
                let bit = (cy - 1) * 4 + (cx - 1);
                bits >> bit & 1 == 1
            };
            let color = match (filled, lit, shaded) {
                (true, true, _) => shift(piece, 40),
                (true, _, true) => shift(piece, -50),
                (true, ..) => piece,
                // The background of the well itself, so the icon and the game
                // are the same picture at two sizes.
                _ => 0x0010_1418,
            };
            rgba.extend_from_slice(&[
                (color >> 16 & 0xff) as u8,
                (color >> 8 & 0xff) as u8,
                (color & 0xff) as u8,
                0xff,
            ]);
        }
    }
    rgba
}

/// A colour lightened or darkened by `by` per channel, clamped. Integer and
/// per-channel: a gamma-correct shade would be right and invisible at 64 px,
/// and would put a float in the one file whose whole claim is that it has none.
fn shift(color: u32, by: i32) -> u32 {
    let channel = |sh: u32| ((color >> sh & 0xff) as i32 + by).clamp(0, 255) as u32;
    channel(16) << 16 | channel(8) << 8 | channel(0)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The two claims worth holding: it is the format's own bytes, and it is
    /// not one flat colour — a generator that drew the background over the
    /// whole square would produce a valid file and an invisible icon.
    #[test]
    fn the_generated_icon_parses_back_and_is_not_a_blank_square() {
        let rgba = draw(icon::SIDE);
        let (side, back) = icon::parse(&icon::encode(icon::SIDE, &rgba)).expect("a valid icon");
        assert_eq!((side, back.len()), (icon::SIDE, rgba.len()));
        let distinct: std::collections::BTreeSet<_> = rgba.chunks_exact(4).collect();
        assert!(
            distinct.len() >= 4,
            "an icon of {} colours is a square, not a piece",
            distinct.len()
        );
        assert!(
            rgba.chunks_exact(4).all(|p| p[3] == 0xff),
            "every pixel is opaque; a stray alpha is a hole in the taskbar"
        );
    }

    /// Same input, same bytes — the property that makes a checked-in binary
    /// reviewable at all (§4.6, one medium over).
    #[test]
    fn the_same_side_draws_the_same_bytes() {
        assert_eq!(draw(icon::SIDE), draw(icon::SIDE));
    }
}
