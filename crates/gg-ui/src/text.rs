//! Text: the rented half of §4.9.
//!
//! Shaping and rasterization are `swash`'s — bidi, ligatures, hinting and
//! fallback chains are not our fight, and the one thing this module refuses to
//! do is reimplement any of them. What stays ours is everything either side:
//! which glyphs are resident ([`crate::atlas`]), where a run lands, and the
//! quads that come out.
//!
//! The output is deliberately *not* a string: [`Fonts::layout`] leaves
//! [`Glyph`]s in a scratch the caller reads and [`crate::DrawList::glyphs`]
//! emits. That is what keeps a steady-state frame free of allocation — the
//! scratch, the shaper's buffers and the rasterizer's image are all reused, and
//! a run whose glyphs are already resident touches the atlas not at all.

use crate::atlas::{Atlas, Key, Raster};
use crate::draw::{Glyph, Rect};
use swash::FontRef;
use swash::scale::{Render, ScaleContext, Source};
use swash::shape::ShapeContext;
use swash::zeno::Format;

/// Which face a run is set in. `0` is [`crate::font`]'s fallback bitmap, which
/// is not a [`Font`] and never enters the packer; loaded faces start at `1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FaceId(u16);

impl FaceId {
    /// The built-in bitmap. Drawn through [`crate::DrawList::text`], not here.
    pub const FALLBACK: FaceId = FaceId(0);
}

/// Why a face would not load.
#[derive(Debug, thiserror::Error)]
pub enum TextError {
    /// Not a font file this build can parse, or an index past its last face.
    #[error("no face at index {0} in {1} bytes")]
    NoSuchFace(usize, usize),
    /// More faces than a [`Key`](crate::atlas::Key) can tell apart. A UI with
    /// 65k faces has a different problem than this error.
    #[error("too many faces")]
    TooManyFaces,
}

/// A loaded face: the file, kept, plus where its tables start.
///
/// The bytes are owned rather than borrowed because a `swash::FontRef` borrows
/// them and a self-referential struct is the alternative — this way a
/// `FontRef` is rebuilt per call, which costs three field copies.
struct Font {
    data: Box<[u8]>,
    offset: u32,
    key: swash::CacheKey,
}

impl Font {
    fn as_ref(&self) -> FontRef<'_> {
        FontRef {
            data: &self.data,
            offset: self.offset,
            key: self.key,
        }
    }
}

/// Vertical metrics of one face at one size, in pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Metrics {
    /// Baseline to the top of the line box; positive.
    pub ascent: f32,
    /// Baseline to the bottom of the line box; positive.
    pub descent: f32,
    /// Top of one line box to the top of the next.
    pub line_height: f32,
}

/// A glyph after shaping and before rasterization.
#[derive(Clone, Copy)]
struct Shaped {
    id: u16,
    x: f32,
    y: f32,
}

/// The faces, the atlas they share, and the reusable machinery between them.
///
/// One per host. Sharing the atlas across faces is the point: two faces are
/// still one texture and therefore still one draw (§4.9).
#[derive(Default)]
pub struct Fonts {
    faces: Vec<Font>,
    atlas: Atlas,
    shape: ShapeContext,
    scale: ScaleContext,
    /// The rasterizer's target, reused — a fresh `Image` per glyph is an
    /// allocation per glyph, which is the whole of what the M13 exit criterion
    /// on steady-state allocation forbids.
    image: swash::scale::image::Image,
    run: Vec<Shaped>,
    glyphs: Vec<Glyph>,
}

/// Rasterize outlines only. `swash` will hand over colour bitmaps and colour
/// outlines too, and deliberately is not asked: a colour glyph is not a
/// coverage texel, and the UI pass samples coverage. (P2: emoji, when a demo
/// asks for one.)
const SOURCES: &[Source] = &[Source::Outline];

impl Fonts {
    /// Load a face from a font file's bytes.
    ///
    /// # Errors
    ///
    /// Bytes that hold no face at `index`, or a 65537th face.
    pub fn load(&mut self, data: Vec<u8>, index: usize) -> Result<FaceId, TextError> {
        let id = u16::try_from(self.faces.len() + 1).map_err(|_| TextError::TooManyFaces)?;
        let data = data.into_boxed_slice();
        let font =
            FontRef::from_index(&data, index).ok_or(TextError::NoSuchFace(index, data.len()))?;
        let (offset, key) = (font.offset, font.key);
        self.faces.push(Font { data, offset, key });
        Ok(FaceId(id))
    }

    /// What the renderer should be holding. Re-upload when [`Fonts::version`]
    /// moves and at no other time.
    #[must_use]
    pub fn coverage(&self) -> gg_render::ui::Coverage<'_> {
        self.atlas.coverage()
    }

    /// Rises whenever a glyph was added to the atlas.
    #[must_use]
    pub fn version(&self) -> u64 {
        self.atlas.version()
    }

    /// `face` at `px` pixels per em. Zeroes for a face that was never loaded.
    #[must_use]
    pub fn metrics(&self, face: FaceId, px: u16) -> Metrics {
        let Some(font) = self.face(face) else {
            return Metrics::default();
        };
        let m = font.as_ref().metrics(&[]).scale(f32::from(px));
        Metrics {
            ascent: m.ascent,
            descent: m.descent,
            // Rounded, because a line box of 18.37 px puts every second row on
            // a half texel and a nearest-sampled glyph then loses a row.
            line_height: (m.ascent + m.descent + m.leading).round(),
        }
    }

    /// Shape `text` in `face` at `px`, rasterize whatever is not resident, and
    /// leave the run in [`Fonts::glyphs`]. Returns the advance width.
    ///
    /// One line: `\n` is not a break here, because breaking is layout's and
    /// layout is a caller. A face that was never loaded lays out nothing.
    pub fn layout(&mut self, face: FaceId, px: u16, text: &str) -> f32 {
        let Fonts {
            faces,
            atlas,
            shape,
            scale,
            image,
            run,
            glyphs,
        } = self;
        run.clear();
        glyphs.clear();
        let Some(font) = usize::from(face.0)
            .checked_sub(1)
            .and_then(|at| faces.get(at))
        else {
            return 0.0;
        };
        let font = font.as_ref();

        let mut shaper = shape.builder(font).size(f32::from(px)).build();
        let ascent = shaper.metrics().ascent;
        shaper.add_str(text);
        let mut pen = 0.0f32;
        shaper.shape_with(|cluster| {
            for glyph in cluster.glyphs {
                run.push(Shaped {
                    id: glyph.id,
                    x: pen + glyph.x,
                    y: glyph.y,
                });
                pen += glyph.advance;
            }
        });

        let mut scaler = scale.builder(font).size(f32::from(px)).hint(true).build();
        // `Format::Alpha` explicitly: the UI pass samples one coverage byte per
        // texel, and a subpixel format would hand back three and pack wrongly.
        let mut render = Render::new(SOURCES);
        let render = render.format(Format::Alpha);
        for shaped in run.iter() {
            let key = Key {
                face: face.0,
                px,
                glyph: shaped.id,
            };
            let slot = match atlas.get(key) {
                Some(slot) => Some(slot),
                None => {
                    // Cleared rather than trusted: `render_into` resizes the
                    // buffer to the new glyph and a shorter one would otherwise
                    // keep the tail of a taller neighbour.
                    image.clear();
                    // A glyph with no outline (a space) still takes an entry,
                    // so it is asked for once instead of every frame.
                    let drawn = render.render_into(&mut scaler, shaped.id, image);
                    let p = image.placement;
                    let raster = match drawn {
                        true => Raster {
                            w: p.width,
                            h: p.height,
                            left: p.left,
                            top: p.top,
                            coverage: &image.data,
                        },
                        false => Raster {
                            w: 0,
                            h: 0,
                            left: 0,
                            top: 0,
                            coverage: &[],
                        },
                    };
                    atlas.insert(key, &raster)
                }
            };
            let Some(slot) = slot.filter(|slot| slot.w != 0 && slot.h != 0) else {
                continue;
            };
            glyphs.push(Glyph {
                // Rounded to whole texels: the sampler is nearest, and a glyph
                // landing on a half texel drops a row of its own coverage.
                rect: Rect::new(
                    (shaped.x + slot.left as f32).round(),
                    (ascent - shaped.y - slot.top as f32).round(),
                    slot.w as f32,
                    slot.h as f32,
                ),
                uv: slot.uv(),
            });
        }
        pen
    }

    /// The run [`Fonts::layout`] left behind, valid until the next call.
    #[must_use]
    pub fn glyphs(&self) -> &[Glyph] {
        &self.glyphs
    }

    fn face(&self, face: FaceId) -> Option<&Font> {
        usize::from(face.0)
            .checked_sub(1)
            .and_then(|at| self.faces.get(at))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::font;

    /// The vendored face (`assets/fonts`, OFL-1.1). Monospace on purpose: the
    /// overlay's rows are column-aligned by format specifiers, so a
    /// proportional face would be a feature loss the M13 criterion names.
    fn face_bytes() -> Vec<u8> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("crates/gg-ui is two below the workspace root");
        std::fs::read(root.join("assets/fonts/FiraMono-Regular.ttf")).expect("the vendored face")
    }

    fn loaded() -> (Fonts, FaceId) {
        let mut fonts = Fonts::default();
        let face = fonts.load(face_bytes(), 0).expect("a parseable face");
        (fonts, face)
    }

    #[test]
    fn bytes_that_are_not_a_font_are_refused_by_name() {
        let mut fonts = Fonts::default();
        let err = fonts.load(vec![0u8; 64], 0).unwrap_err();
        assert!(matches!(err, TextError::NoSuchFace(0, 64)), "{err}");
        // And a face nobody loaded lays out nothing rather than panicking.
        assert_eq!(fonts.layout(FaceId(7), 16, "hello"), 0.0);
        assert!(fonts.glyphs().is_empty());
        assert_eq!(fonts.metrics(FaceId(7), 16), Metrics::default());
    }

    /// The whole point of the rental: real outlines, at a real size, with ink.
    #[test]
    fn a_run_shapes_rasterizes_and_lands_in_the_atlas() {
        let (mut fonts, face) = loaded();
        let advance = fonts.layout(face, 24, "Hello");
        assert!(advance > 0.0, "the run has width");
        assert_eq!(fonts.glyphs().len(), 5);
        let metrics = fonts.metrics(face, 24);
        assert!(
            metrics.ascent > 0.0 && metrics.line_height >= 24.0,
            "{metrics:?}"
        );
        for glyph in fonts.glyphs() {
            assert!(glyph.rect.w > 0.0 && glyph.rect.h > 0.0, "{glyph:?}");
            // Below the fallback band, since the packer owns everything there.
            assert!(glyph.uv.1 >= font::BAND.1 as f32 / font::EXTENT.1 as f32);
            assert!(glyph.uv.3 <= 1.0 && glyph.uv.2 <= 1.0);
        }
        // Ink, not an empty box: an atlas of zeroes draws nothing and every
        // assertion above would still pass.
        let texels = fonts.coverage().texels;
        let band = (font::BAND.1 * font::EXTENT.0) as usize;
        assert!(texels[band..].iter().any(|t| *t > 0x40), "no coverage");
    }

    /// Monospace is what the overlay's alignment rests on, so it is asserted
    /// rather than assumed of the vendored face.
    #[test]
    fn the_vendored_face_is_monospace() {
        let (mut fonts, face) = loaded();
        let one = fonts.layout(face, 16, "i");
        let wide = fonts.layout(face, 16, "W");
        assert_eq!(one, wide);
        assert_eq!(fonts.layout(face, 16, "iiii"), one * 4.0);
    }

    /// A space advances the pen and emits no quad — and takes an atlas entry
    /// anyway, so the rasterizer is asked about it exactly once.
    #[test]
    fn a_space_advances_without_drawing() {
        let (mut fonts, face) = loaded();
        let advance = fonts.layout(face, 16, "a a");
        assert_eq!(fonts.glyphs().len(), 2);
        assert!(advance > fonts.layout(face, 16, "aa"));
        let version = fonts.version();
        fonts.layout(face, 16, "a a");
        assert_eq!(fonts.version(), version, "nothing new to upload");
    }

    /// Steady state is the criterion: a second identical run adds no glyph, so
    /// nothing is re-uploaded and nothing is rasterized. The capacity check is
    /// the allocation half — the scratches are reused, never regrown.
    #[test]
    fn a_settled_run_costs_no_atlas_work_and_no_growth() {
        let (mut fonts, face) = loaded();
        let text = "the quick brown fox 0123456789";
        fonts.layout(face, 18, text);
        let version = fonts.version();
        let (run, glyphs) = (fonts.run.capacity(), fonts.glyphs.capacity());
        for _ in 0..8 {
            fonts.layout(face, 18, text);
        }
        assert_eq!(fonts.version(), version);
        assert_eq!(
            (fonts.run.capacity(), fonts.glyphs.capacity()),
            (run, glyphs)
        );
    }

    /// Same input, same output, down to the atlas bytes — a UI golden
    /// reference is only a gate if the frame it judges reproduces.
    #[test]
    fn the_same_sequence_produces_the_same_atlas_and_the_same_run() {
        let build = || {
            let (mut fonts, face) = loaded();
            fonts.layout(face, 20, "gg");
            fonts.layout(face, 20, "typography");
            let glyphs = fonts.glyphs().to_vec();
            (fonts, glyphs)
        };
        let (a, run_a) = build();
        let (b, run_b) = build();
        assert_eq!(run_a, run_b);
        assert_eq!(a.version(), b.version());
        assert_eq!(a.coverage().texels, b.coverage().texels);
    }

    /// Where a run *draws* must not depend on what was cached before it — only
    /// which texels it samples may. A rect that moved with cache state would be
    /// text that jitters as the atlas fills.
    #[test]
    fn a_runs_positions_do_not_depend_on_the_cache_state() {
        let rects =
            |fonts: &Fonts| -> Vec<Rect> { fonts.glyphs().iter().map(|g| g.rect).collect() };
        let (mut cold, face) = loaded();
        cold.layout(face, 20, "typography");
        let (mut warm, face) = loaded();
        warm.layout(face, 20, "gg");
        warm.layout(face, 20, "typography");
        assert_eq!(rects(&cold), rects(&warm));
        assert_ne!(cold.glyphs()[0].uv, warm.glyphs()[0].uv, "different slots");
    }

    /// Sizes are separate cache entries and separate pictures. A run at 32 that
    /// reused 16's coverage would be half-size glyphs on a full-size advance.
    #[test]
    fn a_second_size_is_a_second_set_of_glyphs() {
        let (mut fonts, face) = loaded();
        let small = fonts.layout(face, 12, "M");
        let small_rect = fonts.glyphs()[0].rect;
        let large = fonts.layout(face, 32, "M");
        let large_rect = fonts.glyphs()[0].rect;
        assert!(large > small);
        assert!(large_rect.h > small_rect.h);
        assert_ne!(small_rect, large_rect);
    }
}
