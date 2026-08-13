//! The fallback glyph source: a 5×7 bitmap font, ASCII `0x20..=0x7E`.
//!
//! Ours, embedded, one bit per texel, and *not* what §4.9 rents — it is what
//! draws when the rented stack has nothing to draw with, which on a crash path
//! or a stripped install is a real state. The seam is the point: [`atlas`] hands
//! the renderer coverage texels and [`uv`] hands the draw layer a rectangle, and
//! neither says where the coverage came from.
//!
//! It also carries [`solid_uv`], which is not a glyph at all: the always-inked
//! texel that lets a plain rectangle be a glyph quad, and therefore lets the
//! whole layer be one draw.
//!
//! The rows below are written as binary literals so the glyphs are legible as
//! shapes — a font whose bugs are invisible in its source is a font nobody
//! proofreads.

/// Glyph cell in texels, including the one-texel gap that keeps a nearest
/// sample off its neighbour.
pub const CELL: (u32, u32) = (6, 8);
/// Inked area of a cell.
pub const GLYPH: (u32, u32) = (5, 7);
/// Cells across the band.
const COLUMNS: u32 = 16;
/// Cells down it — 96 cells for 95 glyphs plus the solid one.
const ROWS: u32 = 6;
/// The corner of the atlas this font owns, at the origin. Everything below it
/// belongs to [`crate::atlas`]'s packer.
pub const BAND: (u32, u32) = (COLUMNS * CELL.0, ROWS * CELL.1);
/// Atlas size in texels, **fixed** — every uv in the layer is a division by it,
/// so a growing atlas would move the *v* of every glyph already placed and
/// invalidate a frame mid-build. 256 KiB on the CPU, 1 MiB on the device once
/// [`gg_render::ui`] expands it to RGBA.
///
/// P2: when a real face at several sizes fills this, the answer is an LRU over
/// [`crate::atlas`]'s shelves, not a bigger constant. **How far that is, now
/// measured rather than guessed:** the editor's own face at four scales packs
/// to 277 of these 512 rows — 54 %, which is closer than it reads. It is not
/// reachable today for a reason that is not headroom, though:
/// `gg_editor::rent_face` builds a fresh `Fonts` per size, so sizes never
/// accumulate in one atlas and the full case needs a design change before it
/// needs an LRU. `text::tests` packs the four-scale case anyway and fails at
/// 75 %, so the change that makes this reachable arrives with the answer to
/// whether it also made the LRU due.
pub const EXTENT: (u32, u32) = (512, 512);
/// The first character [`FONT`] covers.
const FIRST: u8 = 0x20;
/// The cell filled solid, so a plain rectangle is a glyph quad whose uv lands
/// on a texel that is always 1 — one pipeline and one draw for the whole layer.
const SOLID_CELL: u32 = 95;

/// Rows top to bottom; bit 4 is the leftmost column.
#[rustfmt::skip]
const FONT: [[u8; 7]; 95] = [
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000], // ' '
    [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100], // '!'
    [0b01010, 0b01010, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000], // '"'
    [0b01010, 0b01010, 0b11111, 0b01010, 0b11111, 0b01010, 0b01010], // '#'
    [0b00100, 0b01111, 0b10100, 0b01110, 0b00101, 0b11110, 0b00100], // '$'
    [0b11000, 0b11001, 0b00010, 0b00100, 0b01000, 0b10011, 0b00011], // '%'
    [0b01100, 0b10010, 0b10100, 0b01000, 0b10101, 0b10010, 0b01101], // '&'
    [0b00100, 0b00100, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000], // '\''
    [0b00010, 0b00100, 0b01000, 0b01000, 0b01000, 0b00100, 0b00010], // '('
    [0b01000, 0b00100, 0b00010, 0b00010, 0b00010, 0b00100, 0b01000], // ')'
    [0b00000, 0b00100, 0b10101, 0b01110, 0b10101, 0b00100, 0b00000], // '*'
    [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000], // '+'
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00100, 0b01000], // ','
    [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000], // '-'
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100], // '.'
    [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000], // '/'
    [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110], // '0'
    [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110], // '1'
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111], // '2'
    [0b11111, 0b00010, 0b00100, 0b00010, 0b00001, 0b10001, 0b01110], // '3'
    [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010], // '4'
    [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110], // '5'
    [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110], // '6'
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000], // '7'
    [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110], // '8'
    [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100], // '9'
    [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b01100, 0b00000], // ':'
    [0b00000, 0b01100, 0b01100, 0b00000, 0b01100, 0b00100, 0b01000], // ';'
    [0b00010, 0b00100, 0b01000, 0b10000, 0b01000, 0b00100, 0b00010], // '<'
    [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000], // '='
    [0b01000, 0b00100, 0b00010, 0b00001, 0b00010, 0b00100, 0b01000], // '>'
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b00000, 0b00100], // '?'
    [0b01110, 0b10001, 0b10111, 0b10101, 0b10111, 0b10000, 0b01110], // '@'
    [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001], // 'A'
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110], // 'B'
    [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110], // 'C'
    [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100], // 'D'
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111], // 'E'
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000], // 'F'
    [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111], // 'G'
    [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001], // 'H'
    [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110], // 'I'
    [0b00111, 0b00010, 0b00010, 0b00010, 0b00010, 0b10010, 0b01100], // 'J'
    [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001], // 'K'
    [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111], // 'L'
    [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001], // 'M'
    [0b10001, 0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001], // 'N'
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110], // 'O'
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000], // 'P'
    [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101], // 'Q'
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001], // 'R'
    [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110], // 'S'
    [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100], // 'T'
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110], // 'U'
    [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100], // 'V'
    [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001], // 'W'
    [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001], // 'X'
    [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100], // 'Y'
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111], // 'Z'
    [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110], // '['
    [0b10000, 0b01000, 0b01000, 0b00100, 0b00010, 0b00010, 0b00001], // '\\'
    [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110], // ']'
    [0b00100, 0b01010, 0b10001, 0b00000, 0b00000, 0b00000, 0b00000], // '^'
    [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111], // '_'
    [0b01000, 0b00100, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000], // '`'
    [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111], // 'a'
    [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110], // 'b'
    [0b00000, 0b00000, 0b01110, 0b10001, 0b10000, 0b10001, 0b01110], // 'c'
    [0b00001, 0b00001, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111], // 'd'
    [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110], // 'e'
    [0b00110, 0b01001, 0b01000, 0b11100, 0b01000, 0b01000, 0b01000], // 'f'
    [0b00000, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110], // 'g'
    [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001], // 'h'
    [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110], // 'i'
    [0b00010, 0b00000, 0b00110, 0b00010, 0b00010, 0b10010, 0b01100], // 'j'
    [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010], // 'k'
    [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110], // 'l'
    [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001], // 'm'
    [0b00000, 0b00000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001], // 'n'
    [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110], // 'o'
    [0b00000, 0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000], // 'p'
    [0b00000, 0b01111, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001], // 'q'
    [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000], // 'r'
    [0b00000, 0b00000, 0b01111, 0b10000, 0b01110, 0b00001, 0b11110], // 's'
    [0b01000, 0b01000, 0b11100, 0b01000, 0b01000, 0b01001, 0b00110], // 't'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101], // 'u'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100], // 'v'
    [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010], // 'w'
    [0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001], // 'x'
    [0b00000, 0b10001, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110], // 'y'
    [0b00000, 0b00000, 0b11111, 0b00010, 0b00100, 0b01000, 0b11111], // 'z'
    [0b00110, 0b01000, 0b01000, 0b11000, 0b01000, 0b01000, 0b00110], // '{'
    [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100], // '|'
    [0b01100, 0b00010, 0b00010, 0b00011, 0b00010, 0b00010, 0b01100], // '}'
    [0b00000, 0b00000, 0b01000, 0b10101, 0b00010, 0b00000, 0b00000], // '~'
];

/// A coverage atlas holding nothing but this font: one byte per texel, `0` or
/// `255`, row-major over [`EXTENT`], the band at the origin and the rest blank
/// for the packer. Built rather than embedded — the table is 665 bytes and the
/// expansion is a loop that runs once.
pub fn atlas() -> Vec<u8> {
    let mut texels = vec![0u8; (EXTENT.0 * EXTENT.1) as usize];
    let mut put = |x: u32, y: u32| texels[(y * EXTENT.0 + x) as usize] = 0xff;
    for (index, glyph) in FONT.iter().enumerate() {
        let (ox, oy) = origin(index as u32);
        for (row, bits) in glyph.iter().enumerate() {
            for column in 0..GLYPH.0 {
                if bits & (1 << (GLYPH.0 - 1 - column)) != 0 {
                    put(ox + column, oy + row as u32);
                }
            }
        }
    }
    let (ox, oy) = origin(SOLID_CELL);
    for y in 0..CELL.1 {
        for x in 0..CELL.0 {
            put(ox + x, oy + y);
        }
    }
    texels
}

/// The inked rectangle for `c` in atlas coordinates `[0, 1]`, as
/// `(u0, v0, u1, v1)`. Anything outside the table draws as `?`, which is a
/// visible wrong answer rather than a hole.
pub fn uv(c: char) -> (f32, f32, f32, f32) {
    let index = u8::try_from(c)
        .ok()
        .filter(|b| (FIRST..FIRST + FONT.len() as u8).contains(b))
        .map_or(u32::from(b'?' - FIRST), |b| u32::from(b - FIRST));
    let (x, y) = origin(index);
    (
        x as f32 / EXTENT.0 as f32,
        y as f32 / EXTENT.1 as f32,
        (x + GLYPH.0) as f32 / EXTENT.0 as f32,
        (y + GLYPH.1) as f32 / EXTENT.1 as f32,
    )
}

/// The always-inked texel a solid rectangle samples — a point, not a
/// rectangle, so no filtering can reach a neighbour whatever the sampler.
pub fn solid_uv() -> (f32, f32) {
    let (x, y) = origin(SOLID_CELL);
    (
        (x as f32 + 0.5) / EXTENT.0 as f32,
        (y as f32 + 0.5) / EXTENT.1 as f32,
    )
}

/// Top-left texel of a cell.
fn origin(cell: u32) -> (u32, u32) {
    (cell % COLUMNS * CELL.0, cell / COLUMNS * CELL.1)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// Every cell stays inside the atlas, and the solid one is genuinely solid
    /// — a rectangle sampling a blank texel would draw nothing at all, and the
    /// overlay would be invisible with no error anywhere.
    #[test]
    fn the_atlas_holds_every_cell_and_the_solid_one_is_inked() {
        let texels = atlas();
        assert_eq!(texels.len(), (EXTENT.0 * EXTENT.1) as usize);
        let (u, v) = solid_uv();
        let x = (u * EXTENT.0 as f32) as u32;
        let y = (v * EXTENT.1 as f32) as u32;
        assert_eq!(texels[(y * EXTENT.0 + x) as usize], 0xff);
        assert_eq!(origin(SOLID_CELL).0 + CELL.0, BAND.0, "last column");
        assert_eq!(origin(SOLID_CELL).1 + CELL.1, BAND.1, "last row");
        // And nothing of this font's is written below the band, which is what
        // the packer is about to hand out.
        assert!(
            texels[(BAND.1 * EXTENT.0) as usize..]
                .iter()
                .all(|t| *t == 0)
        );
    }

    /// Space is blank and everything else is not. A glyph silently left empty
    /// is a character that vanishes from the overlay, which reads as a layout
    /// bug rather than a missing glyph.
    #[test]
    fn every_printable_character_but_space_has_ink() {
        for (index, glyph) in FONT.iter().enumerate() {
            let c = (FIRST + index as u8) as char;
            let ink = glyph.iter().any(|row| *row != 0);
            assert_eq!(ink, c != ' ', "{c:?} at index {index}");
            for row in glyph {
                assert!(*row < 1 << GLYPH.0, "{c:?} has ink outside its 5 columns");
            }
        }
    }

    /// The cells a string reaches, spot-checked against shapes that would
    /// survive an off-by-one in the table: `I` is symmetric, `L` is not, and a
    /// table shifted by one would swap them for `H` and `K`.
    #[test]
    fn glyphs_land_on_the_characters_they_are_drawn_for() {
        let rows = |c: char| FONT[(c as u8 - FIRST) as usize];
        assert_eq!(rows('I')[0], 0b01110);
        assert_eq!(rows('I')[3], 0b00100);
        assert_eq!(rows('L')[6], 0b11111);
        assert_eq!(rows('L')[0], 0b10000);
        assert_eq!(rows(' '), [0; 7]);
        // An unknown character borrows `?`'s cell rather than drawing nothing.
        assert_eq!(uv('€'), uv('?'));
    }
}
