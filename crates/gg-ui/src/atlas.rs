//! The coverage atlas every UI draw samples (§4.9: "glyph atlas management is
//! ours; what fills the atlas is not").
//!
//! One texture, therefore one draw — which is why the fallback band, the solid
//! texel a plain rectangle samples, and every glyph a rented rasterizer hands
//! over all live in the same bitmap. [`crate::font`] owns the corner above
//! `font::BAND.1`; the packer below is shelf-based, appending left to right and
//! opening a new shelf when a glyph is taller than the shelves already open.
//!
//! Nothing here evicts. A full atlas refuses the insert and the caller draws the
//! fallback's `?`, which is a visible wrong answer rather than a silent one —
//! see [`font::EXTENT`](crate::font::EXTENT) for the P2 that fixes it, and for
//! how much room is left, which is a measurement now and not a guess.

use crate::font;
use gg_render::ui::Coverage;

/// One texel of margin around every packed glyph. The sampler is nearest and
/// the uvs land on texel edges, so a fragment exactly on a boundary may sample
/// either side — the gap is what makes "either" harmless.
const GAP: u32 = 1;

/// What a glyph is keyed by. Integer sizes only: subpixel positioning would
/// multiply the cache by its phase count, and the layer draws at integer pixel
/// positions anyway. (P2: subpixel phases, when a HUD animates text.)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key {
    /// Which face, as [`crate::text::FaceId`] numbers them.
    pub face: u16,
    /// Pixels per em, rounded.
    pub px: u16,
    /// The face's own glyph index — *not* a character. Shaping already turned
    /// a cluster into these, and two characters can share one.
    pub glyph: u16,
}

/// What a rasterizer produced for one glyph: its coverage, its size, and where
/// it sits relative to the pen.
///
/// The offsets pass straight through the packer and come back out of a cache
/// hit, so a resident glyph is one lookup rather than a slot here and a bearing
/// in a second structure keyed identically.
pub struct Raster<'a> {
    /// Width in texels.
    pub w: u32,
    /// Height in texels.
    pub h: u32,
    /// Pen to the glyph's left edge; negative for a glyph that overhangs.
    pub left: i32,
    /// Baseline to the glyph's top edge, positive up.
    pub top: i32,
    /// Row-major, `w * h` long.
    pub coverage: &'a [u8],
}

/// Where a glyph landed, in texels, and how it sits on the pen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Slot {
    /// Left edge.
    pub x: u32,
    /// Top edge.
    pub y: u32,
    /// Width.
    pub w: u32,
    /// Height.
    pub h: u32,
    /// [`Raster::left`], carried.
    pub left: i32,
    /// [`Raster::top`], carried.
    pub top: i32,
}

impl Slot {
    /// The rectangle to sample, as `(u0, v0, u1, v1)` over the whole atlas.
    #[must_use]
    pub fn uv(&self) -> (f32, f32, f32, f32) {
        let (ex, ey) = (font::EXTENT.0 as f32, font::EXTENT.1 as f32);
        (
            self.x as f32 / ex,
            self.y as f32 / ey,
            (self.x + self.w) as f32 / ex,
            (self.y + self.h) as f32 / ey,
        )
    }
}

/// A shelf: a horizontal band of one height, filled left to right.
struct Shelf {
    top: u32,
    height: u32,
    pen: u32,
}

/// The atlas: texels, the shelves below the fallback band, and what is in them.
pub struct Atlas {
    texels: Vec<u8>,
    shelves: Vec<Shelf>,
    /// Sorted by key. A sorted `Vec` rather than a map because §3 bans
    /// `HashMap` outright and this holds a few hundred entries — a binary
    /// search over them is cheaper than hashing one.
    entries: Vec<(Key, Slot)>,
    /// Bumped whenever [`Atlas::texels`] changed, so an owner can re-upload
    /// exactly when there is something new and never in steady state.
    version: u64,
    /// Set once the packer has refused an insert, so the warning is one line
    /// rather than one per glyph per frame.
    full: bool,
}

impl Default for Atlas {
    fn default() -> Self {
        Atlas {
            texels: font::atlas(),
            shelves: Vec::new(),
            entries: Vec::new(),
            version: 1,
            full: false,
        }
    }
}

impl Atlas {
    /// What the renderer uploads. Hand it over whenever [`Atlas::version`]
    /// moved; in steady state it never does.
    #[must_use]
    pub fn coverage(&self) -> Coverage<'_> {
        Coverage {
            extent: font::EXTENT,
            texels: &self.texels,
        }
    }

    /// Rises every time the texels changed and never otherwise.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Rows the packer has committed, against the rows it has. Test-only: it
    /// exists so [`crate::font::EXTENT`]'s deferral can be a measurement instead
    /// of a guess, and a host has nothing to do with the answer — which is also
    /// why it is not public surface `gg-ui`'s baseline would have to carry.
    #[cfg(test)]
    pub(crate) fn packed_rows(&self) -> (u32, u32) {
        let used = self
            .shelves
            .last()
            .map_or(font::BAND.1, |s| s.top + s.height);
        (used, font::EXTENT.1)
    }

    /// Where `key` is, if it is here.
    #[must_use]
    pub fn get(&self, key: Key) -> Option<Slot> {
        self.entries
            .binary_search_by_key(&key, |(k, _)| *k)
            .ok()
            .map(|at| self.entries[at].1)
    }

    /// Pack `raster` under `key` and return where it landed, or `None` when the
    /// atlas is full.
    ///
    /// A zero-area glyph (a space) takes a real entry with an empty slot, so it
    /// is asked for once rather than rasterized every frame.
    ///
    /// # Panics
    ///
    /// `raster.coverage.len() != raster.w * raster.h`.
    pub fn insert(&mut self, key: Key, raster: &Raster<'_>) -> Option<Slot> {
        let Raster { w, h, coverage, .. } = *raster;
        assert_eq!(coverage.len(), (w * h) as usize, "coverage is not w*h");
        if let Some(slot) = self.get(key) {
            return Some(slot);
        }
        let mut slot = match w == 0 || h == 0 {
            true => Slot::default(),
            false => self.reserve(w, h)?,
        };
        slot.left = raster.left;
        slot.top = raster.top;
        for row in 0..h {
            let from = (row * w) as usize;
            let to = ((slot.y + row) * font::EXTENT.0 + slot.x) as usize;
            self.texels[to..to + w as usize].copy_from_slice(&coverage[from..from + w as usize]);
        }
        let at = self.entries.partition_point(|(k, _)| *k < key);
        self.entries.insert(at, (key, slot));
        self.version += 1;
        Some(slot)
    }

    /// Find room for `w`×`h`: the first open shelf that is tall enough and has
    /// the width left, else a new shelf under the lowest one.
    ///
    /// First-fit rather than best-fit deliberately: best-fit would pack a
    /// mixed-size atlas tighter, and a UI asks for one or two sizes of one face
    /// in a run, so the shelves are near-uniform and the tie-breaking would only
    /// cost a scan.
    fn reserve(&mut self, w: u32, h: u32) -> Option<Slot> {
        let (need_w, need_h) = (w + GAP, h + GAP);
        if let Some(shelf) = self
            .shelves
            .iter_mut()
            .find(|s| s.height >= need_h && s.pen + need_w <= font::EXTENT.0)
        {
            let x = shelf.pen;
            shelf.pen += need_w;
            return Some(Slot {
                x,
                y: shelf.top,
                w,
                h,
                ..Slot::default()
            });
        }
        let top = self
            .shelves
            .last()
            .map_or(font::BAND.1, |s| s.top + s.height);
        if need_w > font::EXTENT.0 || top + need_h > font::EXTENT.1 {
            if !self.full {
                self.full = true;
                tracing::warn!(
                    extent = ?font::EXTENT,
                    glyphs = self.entries.len(),
                    "ui glyph atlas full; further glyphs draw as the fallback"
                );
            }
            return None;
        }
        self.shelves.push(Shelf {
            top,
            height: need_h,
            pen: need_w,
        });
        Some(Slot {
            x: 0,
            y: top,
            w,
            h,
            ..Slot::default()
        })
    }
}

/// The atlas a host has before it has loaded a font: [`crate::font`]'s band and
/// nothing else.
///
/// Expanded on first call and kept, because the renderer may be rebuilt (a
/// device loss, a second window) and re-expanding it per bring-up is not worth
/// a second copy of the bytes.
#[must_use]
pub fn fallback() -> Coverage<'static> {
    static TEXELS: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    Coverage {
        extent: font::EXTENT,
        texels: TEXELS.get_or_init(font::atlas),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn key(glyph: u16) -> Key {
        Key {
            face: 1,
            px: 16,
            glyph,
        }
    }

    /// A solid block `w`×`h` with no bearing.
    fn block(w: u32, h: u32) -> (Vec<u8>, u32, u32) {
        (vec![0xff; (w * h) as usize], w, h)
    }

    fn put(atlas: &mut Atlas, key: Key, w: u32, h: u32) -> Option<Slot> {
        let (coverage, w, h) = block(w, h);
        atlas.insert(
            key,
            &Raster {
                w,
                h,
                left: 0,
                top: 0,
                coverage: &coverage,
            },
        )
    }

    /// The band is untouched by the packer, because the solid texel and every
    /// fallback glyph are addressed by constants that predate any font.
    #[test]
    fn packing_never_reaches_the_fallback_band() {
        let mut atlas = Atlas::default();
        for glyph in 0..200u16 {
            put(&mut atlas, key(glyph), 9, 15);
        }
        let before = font::atlas();
        let band = (font::BAND.1 * font::EXTENT.0) as usize;
        assert_eq!(atlas.texels[..band], before[..band]);
    }

    /// Coverage lands where the slot says it does, row by row — an off-by-one
    /// in the row stride is a glyph sheared across the atlas, which looks like
    /// a rasterizer bug and is not one.
    #[test]
    fn a_glyph_is_written_at_the_slot_it_was_given() {
        let mut atlas = Atlas::default();
        let raster = Raster {
            w: 3,
            h: 2,
            left: -1,
            top: 9,
            coverage: &[1, 2, 3, 4, 5, 6],
        };
        let slot = atlas.insert(key(0), &raster).unwrap();
        assert!(slot.y >= font::BAND.1);
        assert_eq!((slot.left, slot.top), (-1, 9), "the bearing rides along");
        let at =
            |x: u32, y: u32| atlas.texels[((slot.y + y) * font::EXTENT.0 + slot.x + x) as usize];
        assert_eq!([at(0, 0), at(1, 0), at(2, 0)], [1, 2, 3]);
        assert_eq!([at(0, 1), at(1, 1), at(2, 1)], [4, 5, 6]);
        // And the texel past its right edge is the gap, not the next glyph.
        let next = atlas.insert(key(1), &raster).unwrap();
        assert_eq!(next.x, slot.x + 3 + GAP);
    }

    /// A second ask for the same key is the cache hit the whole type exists
    /// for: same slot, no new texels, and the version does not move — which is
    /// what makes a steady-state frame cost no upload.
    #[test]
    fn a_repeated_key_is_the_same_slot_and_no_new_version() {
        let mut atlas = Atlas::default();
        let first = put(&mut atlas, key(7), 4, 4).unwrap();
        let version = atlas.version();
        let again = atlas
            .insert(
                key(7),
                &Raster {
                    w: 4,
                    h: 4,
                    left: 0,
                    top: 0,
                    coverage: &[0x00; 16],
                },
            )
            .unwrap();
        assert_eq!(first, again);
        assert_eq!(atlas.version(), version, "no re-upload for a hit");
        assert_eq!(atlas.get(key(7)), Some(first));
        assert_eq!(atlas.get(key(8)), None);
    }

    /// Keys differing only in face or size are different glyphs. Collapsing
    /// them would draw 12-pixel text with 24-pixel coverage, which reads as a
    /// blurry font rather than as a cache bug.
    #[test]
    fn face_and_size_are_part_of_the_key() {
        let mut atlas = Atlas::default();
        let mut at = |face, px| put(&mut atlas, Key { face, px, glyph: 3 }, 4, 4);
        let (a, b, c) = (at(1, 16), at(1, 24), at(2, 16));
        assert_ne!(a, b);
        assert_ne!(a, c);
    }

    /// A shelf that runs out of width opens the next one below it, and a glyph
    /// too tall for an open shelf does not squeeze into it.
    #[test]
    fn shelves_wrap_and_a_taller_glyph_opens_its_own() {
        let mut atlas = Atlas::default();
        let wide = font::EXTENT.0 / 2 - GAP;
        let a = put(&mut atlas, key(0), wide, 8).unwrap();
        let b = put(&mut atlas, key(1), wide, 8).unwrap();
        assert_eq!(a.y, b.y, "same shelf");
        let c = put(&mut atlas, key(2), 4, 8).unwrap();
        assert!(c.y > a.y, "no room left, so a new shelf");
        let tall = put(&mut atlas, key(3), 4, 40).unwrap();
        assert!(tall.y > c.y, "too tall for the open shelf");
    }

    /// Full is refused, not wrapped: a packer that reused occupied texels would
    /// corrupt glyphs already on screen this frame.
    #[test]
    fn a_full_atlas_refuses_rather_than_overwriting() {
        let mut atlas = Atlas::default();
        let (w, h) = (font::EXTENT.0 - GAP, 64);
        let mut placed = 0;
        for glyph in 0..64u16 {
            if put(&mut atlas, key(glyph), w, h).is_none() {
                break;
            }
            placed += 1;
        }
        assert!(placed > 0 && placed < 64, "placed {placed}");
        assert_eq!(put(&mut atlas, key(999), w, h), None);
        // And every slot that did land is inside the atlas.
        for (_, slot) in &atlas.entries {
            assert!(slot.x + slot.w <= font::EXTENT.0 && slot.y + slot.h <= font::EXTENT.1);
        }
    }

    /// A space shapes to a glyph with no ink. It gets an entry so the
    /// rasterizer is asked once, and an empty slot so nothing is drawn.
    #[test]
    fn a_blank_glyph_is_cached_without_taking_room() {
        let mut atlas = Atlas::default();
        let slot = put(&mut atlas, key(1), 0, 0).unwrap();
        assert_eq!((slot.w, slot.h), (0, 0));
        assert!(atlas.shelves.is_empty(), "no shelf opened for a space");
        assert_eq!(atlas.get(key(1)), Some(slot));
    }
}
