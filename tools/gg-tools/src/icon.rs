//! Demo 10's taskbar picture, written where the caller asks (§6 M46).
//!
//! `panorama`'s rule one medium over, and `timbre`'s: a checked-in binary in
//! this tree has a source. An icon is the *worst* case for that rule rather than
//! an incidental one — it is the one artifact a reviewer cannot read in a diff,
//! so the diff has to be the generator or there is no review at all.
//!
//! # What it draws, and why not a screenshot of the game
//!
//! One shape per game, in that game's own constants — a tetromino from demo
//! 10's [`SHAPES`] and [`COLORS`], a target and a crosshair from demo 12's
//! [`TARGET_INK`] and [`CROSS`] — so each picture is its game's palette by
//! construction rather than by somebody matching it once. Drawn rather than
//! captured because 64 pixels is not a frame scaled down: at that size a board
//! is a grey smear and a room is mud, and what reads is one shape with a wide
//! margin.
//!
//! Which game is `--game`, and it is a *table* rather than a flag per demo
//! (§6 M75): a third one is a row, and the row is the whole of what a new game
//! contributes here.
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

/// What a game contributes here: the name `--game` takes, and how to draw it.
type Drawn = (&'static str, fn(u32) -> Vec<u8>);

/// The games that have a picture. A third one is a row.
const GAMES: [Drawn; 2] = [("10-tetris", tetris), ("12-shooter", shooter)];

/// Blocks of margin around the piece's **own** extent, split between the two
/// sides of its long axis — so one is half a block at each edge.
///
/// Not around the 4x4 mask (§6 M73). A `SHAPES` entry is a piece inside its
/// spawn box, and where it sits in that box is a *rule of the game* — the S
/// piece occupies three columns of the top two rows, so drawing the box put the
/// picture in a corner with two thirds of the square empty. At 96 px nobody
/// notices; at the 16 px Explorer asks for it is a red smudge with a margin.
/// What the picture wants is the piece, centred, as large as it goes.
const MARGIN: u32 = 1;

pub fn run(args: &[String]) -> Result<()> {
    let mut out = PathBuf::from("target/gg-tools/icon.ggicon");
    let mut side = icon::SIDE;
    let mut game = GAMES[0].0.to_owned();
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--out" => out = PathBuf::from(rest.next().context("--out wants a path")?),
            "--side" => side = rest.next().context("--side wants a number")?.parse()?,
            "--game" => game = rest.next().context("--game wants a demo name")?.clone(),
            other => bail!("unknown argument {other}"),
        }
    }
    if side < 16 {
        bail!("--side wants at least 16, not {side} — below that a block is one pixel");
    }
    let (_, draw) = GAMES
        .iter()
        .find(|(name, _)| *name == game)
        .with_context(|| {
            let known: Vec<&str> = GAMES.iter().map(|(name, _)| *name).collect();
            format!(
                "no icon for `{game}` — this command draws {}",
                known.join(", ")
            )
        })?;
    let rgba = draw(side);
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&out, icon::encode(side, &rgba))?;
    println!(
        "gg-tools icon: {game} at {side}x{side}, {} bytes -> {}",
        rgba.len(),
        out.display()
    );
    Ok(())
}

/// Which cells of the 4x4 spawn mask the piece actually occupies, as
/// `(left, top, width, height)`.
fn extent(bits: u16) -> (u32, u32, u32, u32) {
    let (mut x0, mut y0, mut x1, mut y1) = (4, 4, 0, 0);
    for bit in 0..16u32 {
        if bits >> bit & 1 == 1 {
            let (x, y) = (bit % 4, bit / 4);
            (x0, y0) = (x0.min(x), y0.min(y));
            (x1, y1) = (x1.max(x + 1), y1.max(y + 1));
        }
    }
    (x0, y0, x1 - x0, y1 - y0)
}

/// Demo 10: the pixels are the well's background, and the piece as large as it
/// goes.
///
/// The cell size divides and the remainder goes into the margin, so nothing
/// here constrains `side` and no size rounds twice. What that costs is up to a
/// few pixels of extra margin on one axis; what it buys is that the picture is
/// the piece rather than the box the game spawns it in.
fn tetris(side: u32) -> Vec<u8> {
    // Demo 10's own palette, read rather than matched — the whole reason this
    // lives in a subcommand that links the game crate.
    let piece = demo_10_tetris::COLORS[PIECE];
    let bits = demo_10_tetris::SHAPES[PIECE][0];
    let (bx, by, bw, bh) = extent(bits);
    let cell = side / (bw.max(bh) + MARGIN);
    // Centred on both axes: the short one gets the leftover as margin, which is
    // the only sense in which a non-square piece is framed differently.
    let (ox, oy) = ((side - bw * cell) / 2, (side - bh * cell) / 2);
    let mut rgba = Vec::with_capacity(side as usize * side as usize * 4);
    for y in 0..side {
        for x in 0..side {
            let (dx, dy) = (x.wrapping_sub(ox), y.wrapping_sub(oy));
            // A block is lit on its top-left edges and shaded on its
            // bottom-right, which is what makes four flat squares read as four
            // *blocks* at this size. One pixel, because at 64 across two is a
            // bevel and three is a frame.
            let inside = dx < bw * cell && dy < bh * cell;
            let filled = inside && {
                // `SHAPES` is a 4x4 bitmask, row-major from the top left, and
                // the extent above is what turns a cell here into one of its
                // bits.
                let bit = (dy / cell + by) * 4 + (dx / cell + bx);
                bits >> bit & 1 == 1
            };
            let color = match (filled, dx % cell, dy % cell) {
                (true, 0, _) | (true, _, 0) => shift(piece, 40),
                (true, a, b) if a == cell - 1 || b == cell - 1 => shift(piece, -50),
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

/// Demo 12: a target with the crosshair on it.
///
/// The one shape that *is* this game — a room and a rifle read as neither at 16
/// pixels, and what a player is doing for the whole session is putting the
/// second of these on the first. Proportioned by the game's own [`CROSS`]
/// (arm, thickness, gap) so the picture and the HUD are one drawing, and in its
/// own inks: the disc is what a target is, the arms are what the player sees,
/// and the ring between them is [`HIT_INK`]'s, which is the colour of the
/// moment the two meet.
fn shooter(side: u32) -> Vec<u8> {
    use demo_12_shooter as game;
    // Read off the game's own crosshair rather than chosen: `CROSS` is
    // `(arm, thickness, gap)` in canvas units — a bar `arm` long and `thick`
    // wide, starting `gap` out from the middle — and what matters here is their
    // *ratio*, because the picture is that HUD at another size.
    let (arm, thick, gap) = game::CROSS;
    // The crosshair spans the square less a block of margin, on `tetris`'s
    // rule. One conversion from canvas units to pixels, here, by rounding; every
    // comparison below is integer, so the shape is the same on every machine.
    let reach = (side - side / 8) / 2;
    let unit = |v: f32| ((v * reach as f32) / (gap + arm)) as u32;
    let half = unit(thick).max(2) / 2;
    let gap = unit(gap).max(half + 1);
    // The disc fills the crosshair's own gap and the ring closes it, so the
    // target is the thing the reticle is *around* rather than a backdrop behind
    // it — which is what a player is looking at for the whole session.
    let ring = gap + half;
    let mid = side / 2;
    // The room's **darkest** surface, in this game's shadow. Demo 10's icon uses
    // the well's background because that is what its pieces are drawn on, and
    // the analogue here would be the floor — but every surface in this room is a
    // light neutral, so a white reticle on any of them disappears. Darkest and
    // then shaded: both are rules rather than tastes, `shift` is the same one
    // that bevels a tetromino, and a retinted room moves this with it. What it
    // buys is the one thing a room does not have to provide and a taskbar does —
    // an icon legible against somebody else's wallpaper.
    let ground = shift(
        game::ROOM
            .iter()
            .map(|(_, _, ink)| *ink)
            .min_by_key(|ink| (ink >> 16 & 0xff) * 2 + (ink >> 8 & 0xff) * 5 + (ink & 0xff))
            .unwrap_or(0),
        -90,
    );
    let mut rgba = Vec::with_capacity(side as usize * side as usize * 4);
    for y in 0..side {
        for x in 0..side {
            // Squared distances, so nothing here needs a square root.
            let (dx, dy) = (x.abs_diff(mid), y.abs_diff(mid));
            let r2 = dx * dx + dy * dy;
            let along = dx.max(dy);
            let across = dx.min(dy);
            let on_arm = across <= half && (ring..=reach).contains(&along);
            let color = if on_arm {
                game::CROSS_INK & 0x00ff_ffff
            } else if r2 <= gap * gap {
                game::TARGET_INK
            } else if r2 <= ring * ring {
                game::HIT_INK
            } else {
                ground
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
        let rgba = tetris(icon::SIDE);
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
    /// reviewable at all (§4.6, one medium over). Every game's, by the table, so
    /// a third one inherits the claim rather than needing its own line.
    #[test]
    fn the_same_side_draws_the_same_bytes() {
        for (name, draw) in GAMES {
            assert_eq!(draw(icon::SIDE), draw(icon::SIDE), "{name}");
            assert_eq!(
                draw(icon::SIDE).len(),
                (icon::SIDE * icon::SIDE * 4) as usize,
                "{name}"
            );
            assert!(
                draw(icon::SIDE).chunks_exact(4).all(|p| p[3] == 0xff),
                "{name}: a stray alpha is a hole in the taskbar"
            );
        }
    }

    /// Demo 12's picture, graded on the two things a *taskbar* needs and a room
    /// does not have to provide (§6 M75): the reticle has to be **legible
    /// against its own ground**, and the whole thing has to be centred and large
    /// — the same claim `the_picture_is_the_piece_...` makes one game over.
    ///
    /// Contrast is measured rather than eyeballed because the ground is derived:
    /// it is the room's darkest surface, shaded, and a retinted room moves it.
    /// The first version drew on the *floor* and the white arms vanished.
    #[test]
    fn the_reticle_reads_against_the_room_it_is_drawn_on() {
        let side = icon::SIDE;
        let rgba = shooter(side);
        let at = |x: u32, y: u32| {
            let i = ((y * side + x) * 4) as usize;
            [rgba[i], rgba[i + 1], rgba[i + 2]]
        };
        let luma = |p: [u8; 3]| (u32::from(p[0]) * 2 + u32::from(p[1]) * 5 + u32::from(p[2])) / 8;
        let ground = at(0, 0);
        // An arm, halfway out along +x from the middle, and the disc's centre.
        let arm = at(side * 7 / 8 - 2, side / 2);
        let disc = at(side / 2, side / 2);
        assert!(
            luma(arm).abs_diff(luma(ground)) > 90,
            "the reticle is {arm:?} on {ground:?} and would disappear"
        );
        assert!(
            luma(disc).abs_diff(luma(ground)) > 30,
            "the target is {disc:?} on {ground:?}"
        );
        // Four-fold symmetric about the middle, which is what says it is a
        // reticle rather than a shape that happens to have arms.
        for (x, y) in [(side / 2, side / 4), (side / 4, side / 2)] {
            assert_eq!(at(x, y), at(side - 1 - x, y), "not mirrored in x");
            assert_eq!(at(x, y), at(x, side - 1 - y), "not mirrored in y");
        }
        // And it fills the square rather than sitting in the middle of a margin.
        let painted = (0..side)
            .filter(|x| at(*x, side / 2) != ground)
            .fold((side, 0), |(lo, hi), x| (lo.min(x), hi.max(x + 1)));
        assert!(
            (painted.1 - painted.0) * 4 >= side * 3,
            "the reticle spans {} of {side}",
            painted.1 - painted.0
        );
    }

    /// The defect §6 M73 found, as a claim rather than a look: the piece is
    /// **centred** and **most of the square**, which is what a 16 px taskbar
    /// entry needs and what drawing the 4x4 spawn box did not give.
    ///
    /// Graded on the piece's bounding box in the *pixels*, so it fails for the
    /// old placement (a box in the top-left quadrant) and for a piece drawn too
    /// small, in opposite directions.
    #[test]
    fn the_picture_is_the_piece_and_not_the_box_the_game_spawns_it_in() {
        assert_eq!(extent(demo_10_tetris::SHAPES[PIECE][0]), (0, 0, 3, 2));
        // A square piece, and a straight one, to prove the fit is read off the
        // mask rather than assumed of the S.
        assert_eq!(extent(0b0000_0000_0110_0110), (1, 0, 2, 2));
        assert_eq!(extent(0b0000_0000_1111_0000), (0, 1, 4, 1));

        let side = icon::SIDE;
        let rgba = tetris(side);
        let back = [0x10, 0x14, 0x18];
        let (mut x0, mut y0, mut x1, mut y1) = (side, side, 0, 0);
        for y in 0..side {
            for x in 0..side {
                let at = ((y * side + x) * 4) as usize;
                if rgba[at..at + 3] != back {
                    (x0, y0) = (x0.min(x), y0.min(y));
                    (x1, y1) = (x1.max(x + 1), y1.max(y + 1));
                }
            }
        }
        let (w, h) = (x1 - x0, y1 - y0);
        assert!(w * 4 >= side * 3, "the piece is {w} of {side} across");
        // Centred to within a pixel of rounding on both axes.
        assert!(x0.abs_diff(side - x1) <= 1, "off centre: {x0}..{x1}");
        assert!(y0.abs_diff(side - y1) <= 1, "off centre: {y0}..{y1}");
        // Two rows of blocks by three columns, so the height follows the width.
        assert_eq!(h * 3, w * 2, "{w}x{h} is not the S piece's own shape");
    }
}
