//! The batched draw layer and its clip/transform stacks (§4.9) — built at M8 as
//! the overlay's kernel inside `gg-debug`, moved here whole at M13.
//!
//! Clipping is geometric rather than a scissor rectangle: a quad is intersected
//! with the clip rect and its uvs are moved with it. That is what makes the
//! whole layer one draw call — a scissor would be a state change, and a state
//! change is a second draw.

use crate::font;
use gg_render::ui::UiVertex;

/// An axis-aligned rectangle in pixels, origin top-left.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    /// Left edge.
    pub x: f32,
    /// Top edge.
    pub y: f32,
    /// Width; non-positive means empty.
    pub w: f32,
    /// Height; non-positive means empty.
    pub h: f32,
}

impl Rect {
    /// A rectangle from its edges. `const` so a layout can be a table of them
    /// (§6 M15's panels are exactly that).
    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Rect { x, y, w, h }
    }

    /// Right edge.
    pub fn right(&self) -> f32 {
        self.x + self.w
    }

    /// Bottom edge.
    pub fn bottom(&self) -> f32 {
        self.y + self.h
    }

    /// Whether it encloses any area at all.
    pub fn is_empty(&self) -> bool {
        self.w <= 0.0 || self.h <= 0.0
    }

    /// Whether `(x, y)` is inside. Half-open on the right and bottom edges, so
    /// two abutting rectangles cannot both claim the point between them.
    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x && y >= self.y && x < self.right() && y < self.bottom()
    }

    /// Shrunk by `by` on every side; a negative `by` grows it. Padding, in
    /// whichever direction the caller happens to need it — a panel insets to
    /// find its content area and content outsets to find its panel.
    pub fn inset(&self, by: f32) -> Rect {
        Rect {
            x: self.x + by,
            y: self.y + by,
            w: self.w - by * 2.0,
            h: self.h - by * 2.0,
        }
    }

    /// The smallest rectangle covering both.
    ///
    /// An empty operand is *not* special-cased: a zero-sized rectangle is still
    /// a point the result must cover, which is what lets [`crate::Stack`] start
    /// from a bare origin and grow.
    pub fn union(&self, other: &Rect) -> Rect {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        Rect {
            x,
            y,
            w: self.right().max(other.right()) - x,
            h: self.bottom().max(other.bottom()) - y,
        }
    }

    /// The overlap, or an empty rectangle when there is none.
    pub fn intersect(&self, other: &Rect) -> Rect {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        Rect {
            x,
            y,
            w: self.right().min(other.right()) - x,
            h: self.bottom().min(other.bottom()) - y,
        }
    }
}

/// A quad and the atlas rectangle to cut it from, positioned relative to the
/// top-left of the run it belongs to.
///
/// This is the whole of what [`crate::text`] hands back: the draw layer is told
/// where and from where, never what character it is drawing. That is the same
/// seam `gg_render::ui` keeps one level down, one level up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glyph {
    /// Where to draw it.
    pub rect: Rect,
    /// Where to cut it from, as `(u0, v0, u1, v1)` over the whole atlas.
    pub uv: (f32, f32, f32, f32),
}

/// Offset and uniform scale — what a panel needs to place its contents and what
/// a `d.scale` CVar changes. Not a matrix: rotation and shear have no consumer
/// here, and a 2×3 nobody multiplies is a 2×3 nobody tests.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Transform {
    offset: (f32, f32),
    scale: f32,
}

impl Transform {
    fn apply(&self, r: &Rect) -> Rect {
        Rect {
            x: self.offset.0 + r.x * self.scale,
            y: self.offset.1 + r.y * self.scale,
            w: r.w * self.scale,
            h: r.h * self.scale,
        }
    }

    /// `inner` read in this transform's coordinates.
    fn compose(&self, inner: &Transform) -> Transform {
        Transform {
            offset: (
                self.offset.0 + inner.offset.0 * self.scale,
                self.offset.1 + inner.offset.1 * self.scale,
            ),
            scale: self.scale * inner.scale,
        }
    }
}

/// One frame's UI geometry, batched into a single vertex stream.
///
/// Rebuilt each frame — immediate mode (§4.8) — and reusing one instance is how
/// that stays allocation-free after the first few frames.
#[derive(Default)]
pub struct DrawList {
    vertices: Vec<UiVertex>,
    clips: Vec<Rect>,
    transforms: Vec<Transform>,
}

impl DrawList {
    /// Drop last frame's geometry and both stacks, keeping the capacity.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.clips.clear();
        self.transforms.clear();
    }

    /// This frame's vertices, six per surviving quad.
    pub fn vertices(&self) -> &[UiVertex] {
        &self.vertices
    }

    /// Bound everything drawn until the matching [`DrawList::pop_clip`] to
    /// `rect`, intersected with whatever clip is already in force — a nested
    /// clip can only ever shrink.
    pub fn push_clip(&mut self, rect: Rect) {
        let rect = self.transform().apply(&rect);
        let rect = match self.clips.last() {
            Some(outer) => outer.intersect(&rect),
            None => rect,
        };
        self.clips.push(rect);
    }

    /// Undo the innermost [`DrawList::push_clip`].
    pub fn pop_clip(&mut self) {
        self.clips.pop();
    }

    /// Read the coordinates of everything until the matching
    /// [`DrawList::pop_transform`] in `offset`/`scale`, composed with whatever
    /// is already in force.
    pub fn push_transform(&mut self, offset: (f32, f32), scale: f32) {
        let inner = Transform { offset, scale };
        self.transforms.push(self.transform().compose(&inner));
    }

    /// Undo the innermost [`DrawList::push_transform`].
    pub fn pop_transform(&mut self) {
        self.transforms.pop();
    }

    /// Where `rect` actually lands: through the transform stack, cut by the clip
    /// stack. Empty when the clip excludes it entirely.
    ///
    /// This is the router's half of the seam (§4.9) — a widget hit-tests the
    /// rectangle it *drew into*, so a panel that scrolled its contents or a clip
    /// that cut a row off cannot leave a hit region floating where nothing is
    /// visible.
    pub fn place(&self, rect: Rect) -> Rect {
        let placed = self.transform().apply(&rect);
        match self.clips.last() {
            Some(clip) => placed.intersect(clip),
            None => placed,
        }
    }

    /// A solid rectangle. `color` is `0xAARRGGBB`.
    pub fn rect(&mut self, rect: Rect, color: u32) {
        let (u, v) = font::solid_uv();
        self.quad(&rect, (u, v, u, v), color);
    }

    /// `text` at `(x, y)`, one line, left-aligned, top of the cell at `y`.
    /// Returns the pen position after it, so a caller can chain runs of
    /// different colours without knowing the font's advance.
    pub fn text(&mut self, x: f32, y: f32, text: &str, color: u32) -> f32 {
        let (cell, glyph) = (font::CELL.0 as f32, font::GLYPH);
        let mut pen = x;
        for c in text.chars() {
            // Space is a blank cell in the table, so advancing past it without
            // a quad costs nothing and saves six vertices per space — which on
            // a padded overlay is most of them.
            if c != ' ' {
                let rect = Rect::new(pen, y, glyph.0 as f32, glyph.1 as f32);
                self.quad(&rect, font::uv(c), color);
            }
            pen += cell;
        }
        pen
    }

    /// A shaped run from [`crate::text::Fonts::layout`], with its top-left at
    /// `(x, y)` — the same origin [`DrawList::text`] takes, so switching a call
    /// site from the fallback to a real face does not move the line.
    pub fn glyphs(&mut self, x: f32, y: f32, glyphs: &[Glyph], color: u32) {
        for glyph in glyphs {
            let rect = Rect::new(
                x + glyph.rect.x,
                y + glyph.rect.y,
                glyph.rect.w,
                glyph.rect.h,
            );
            self.quad(&rect, glyph.uv, color);
        }
    }

    /// Width of `text` in this font, unscaled.
    ///
    /// The game's own measurement (§6 M44), not a second one that agrees with
    /// it today — a label is clipped to a rect the game sized, so the two
    /// answers have to be one answer.
    pub fn width(text: &str) -> f32 {
        gg_ecs::boundary::text_width(text)
    }

    /// Height of one line, unscaled.
    pub fn line_height() -> f32 {
        font::CELL.1 as f32
    }

    /// Two triangles, transformed, clipped, and appended. `uv` is
    /// `(u0, v0, u1, v1)` over the whole atlas; a degenerate one is a solid
    /// fill.
    fn quad(&mut self, rect: &Rect, uv: (f32, f32, f32, f32), color: u32) {
        let placed = self.transform().apply(rect);
        if placed.is_empty() {
            return;
        }
        // Through `place` rather than a second copy of it: the rectangle a
        // widget hit-tests and the rectangle it draws are the same rectangle by
        // construction, not by two functions staying in step.
        let clipped = self.place(*rect);
        if clipped.is_empty() {
            return;
        }
        // The uvs move with the cut edges rather than the quad being redrawn
        // whole: a half-clipped glyph shows its left half, not a squeezed one.
        let (u0, v0, u1, v1) = uv;
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let (cu0, cu1) = (
            lerp(u0, u1, (clipped.x - placed.x) / placed.w),
            lerp(u0, u1, (clipped.right() - placed.x) / placed.w),
        );
        let (cv0, cv1) = (
            lerp(v0, v1, (clipped.y - placed.y) / placed.h),
            lerp(v0, v1, (clipped.bottom() - placed.y) / placed.h),
        );
        let corner = |x: f32, y: f32, u: f32, v: f32| UiVertex {
            pos: [x, y],
            uv: [u, v],
            color,
        };
        let (tl, tr) = (
            corner(clipped.x, clipped.y, cu0, cv0),
            corner(clipped.right(), clipped.y, cu1, cv0),
        );
        let (br, bl) = (
            corner(clipped.right(), clipped.bottom(), cu1, cv1),
            corner(clipped.x, clipped.bottom(), cu0, cv1),
        );
        self.vertices.extend_from_slice(&[tl, tr, br, tl, br, bl]);
    }

    fn transform(&self) -> Transform {
        self.transforms.last().copied().unwrap_or(Transform {
            offset: (0.0, 0.0),
            scale: 1.0,
        })
    }
}

/// The cursor's fill and its one-unit halo — the halo is what keeps it visible
/// over a light panel.
const CURSOR: u32 = 0xffff_ffff;
const CURSOR_HALO: u32 = 0xd004_0608;

/// The classic arrow silhouette as `(dx, dy, width)` runs of unit cells, tip at
/// the origin:
///
/// ```text
/// X            The base cut and the hanging tail are not decoration: a 45°
/// XX           body alone is the bottom-left half of a square, and nothing
/// XXX          on it says which corner points (§6 M19).
/// XXXX
/// XXXXX
/// XXXXXX
/// XXXXXXX
/// XXXXXXXX
/// XXXXX
/// XX XX
///     XX
/// ```
pub(crate) const ARROW: [(f32, f32, f32); 12] = [
    (0.0, 0.0, 1.0),
    (0.0, 1.0, 2.0),
    (0.0, 2.0, 3.0),
    (0.0, 3.0, 4.0),
    (0.0, 4.0, 5.0),
    (0.0, 5.0, 6.0),
    (0.0, 6.0, 7.0),
    (0.0, 7.0, 8.0),
    (0.0, 8.0, 5.0),
    (0.0, 9.0, 2.0),
    (3.0, 9.0, 2.0),
    (4.0, 10.0, 2.0),
];

/// An arrow with its tip at `at`, in canvas units.
///
/// Host-drawn because the pointer is an integral of the replayed axis stream
/// ([`crate::Router`]), not where the OS thinks the mouse is, so the OS arrow
/// cannot stand in for it. Runs of rects because the draw list has one
/// primitive; the halo pass is every run dilated by a unit, which is also what
/// fills the notch between runs as outline.
pub fn cursor(list: &mut DrawList, at: (f32, f32)) {
    for (color, grow) in [(CURSOR_HALO, 1.0), (CURSOR, 0.0)] {
        for (dx, dy, w) in ARROW {
            list.rect(Rect::new(at.0 + dx, at.1 + dy, w, 1.0).inset(-grow), color);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const WHITE: u32 = 0xffff_ffff;

    fn bounds(list: &DrawList) -> Rect {
        let xs: Vec<f32> = list.vertices().iter().map(|v| v.pos[0]).collect();
        let ys: Vec<f32> = list.vertices().iter().map(|v| v.pos[1]).collect();
        let (x0, y0) = (
            xs.iter().copied().fold(f32::MAX, f32::min),
            ys.iter().copied().fold(f32::MAX, f32::min),
        );
        Rect::new(
            x0,
            y0,
            xs.iter().copied().fold(f32::MIN, f32::max) - x0,
            ys.iter().copied().fold(f32::MIN, f32::max) - y0,
        )
    }

    #[test]
    fn a_rectangle_is_six_vertices_at_the_corners_it_was_given() {
        let mut list = DrawList::default();
        list.rect(Rect::new(10.0, 20.0, 30.0, 40.0), WHITE);
        assert_eq!(list.vertices().len(), 6);
        assert_eq!(bounds(&list), Rect::new(10.0, 20.0, 30.0, 40.0));
        // A solid fill samples one texel, so every corner shares its uv —
        // nothing a filter could reach a neighbouring glyph through.
        let uv = list.vertices()[0].uv;
        assert!(list.vertices().iter().all(|v| v.uv == uv));
    }

    #[test]
    fn text_advances_by_the_cell_and_skips_blank_glyphs() {
        let mut list = DrawList::default();
        let pen = list.text(0.0, 0.0, "AB", WHITE);
        assert_eq!(pen, DrawList::width("AB"));
        assert_eq!(list.vertices().len(), 12);
        list.clear();
        // Two glyphs and a space: the space advances the pen but draws nothing.
        let pen = list.text(0.0, 0.0, "A B", WHITE);
        assert_eq!(pen, DrawList::width("A B"));
        assert_eq!(list.vertices().len(), 12);
    }

    /// A clip cuts geometry *and* uv together. Half a glyph must show its left
    /// half rather than the whole glyph squeezed into half the width — the bug
    /// a geometric clip has and a scissor cannot.
    #[test]
    fn a_clip_cuts_the_uvs_with_the_geometry() {
        let mut list = DrawList::default();
        let full = {
            list.text(0.0, 0.0, "W", WHITE);
            list.vertices().to_vec()
        };
        list.clear();
        list.push_clip(Rect::new(0.0, 0.0, 2.0, 100.0));
        list.text(0.0, 0.0, "W", WHITE);
        list.pop_clip();
        let cut = list.vertices();
        assert_eq!(cut.len(), 6);
        assert_eq!(bounds(&list).w, 2.0, "geometry is cut at the clip");
        let width = |v: &[UiVertex]| {
            v.iter().map(|v| v.uv[0]).fold(f32::MIN, f32::max)
                - v.iter().map(|v| v.uv[0]).fold(f32::MAX, f32::min)
        };
        let ratio = width(cut) / width(&full);
        assert!(
            (ratio - 2.0 / font::GLYPH.0 as f32).abs() < 1e-5,
            "uv is cut in the same proportion, got {ratio}"
        );
    }

    #[test]
    fn a_clip_that_excludes_everything_emits_nothing() {
        let mut list = DrawList::default();
        list.push_clip(Rect::new(0.0, 0.0, 10.0, 10.0));
        list.rect(Rect::new(50.0, 50.0, 10.0, 10.0), WHITE);
        assert!(list.vertices().is_empty());
        list.pop_clip();
        list.rect(Rect::new(50.0, 50.0, 10.0, 10.0), WHITE);
        assert_eq!(list.vertices().len(), 6, "and the pop restores it");
    }

    /// Nested clips shrink and never grow — an inner panel cannot draw outside
    /// the one that contains it by asking for more room.
    #[test]
    fn a_nested_clip_can_only_shrink() {
        let mut list = DrawList::default();
        list.push_clip(Rect::new(0.0, 0.0, 10.0, 10.0));
        list.push_clip(Rect::new(0.0, 0.0, 100.0, 100.0));
        list.rect(Rect::new(0.0, 0.0, 100.0, 100.0), WHITE);
        assert_eq!(bounds(&list), Rect::new(0.0, 0.0, 10.0, 10.0));
    }

    /// The router's seam: what `place` reports and what a quad occupies are the
    /// same rectangle under every stack state. A drift here is a hit region that
    /// is not where the widget is.
    #[test]
    fn place_reports_the_rectangle_a_quad_actually_occupies() {
        let mut list = DrawList::default();
        list.push_transform((10.0, 20.0), 2.0);
        list.push_clip(Rect::new(0.0, 0.0, 30.0, 30.0));
        let rect = Rect::new(5.0, 0.0, 40.0, 5.0);
        let placed = list.place(rect);
        list.rect(rect, WHITE);
        assert_eq!(bounds(&list), placed);
        // And a rect the clip excludes places empty and draws nothing.
        list.clear();
        list.push_clip(Rect::new(0.0, 0.0, 10.0, 10.0));
        assert!(list.place(Rect::new(50.0, 50.0, 5.0, 5.0)).is_empty());
        list.rect(Rect::new(50.0, 50.0, 5.0, 5.0), WHITE);
        assert!(list.vertices().is_empty());
    }

    #[test]
    fn transforms_compose_and_a_clip_is_taken_in_the_transform_that_pushed_it() {
        let mut list = DrawList::default();
        list.push_transform((10.0, 10.0), 2.0);
        list.push_transform((5.0, 0.0), 3.0);
        list.rect(Rect::new(1.0, 0.0, 1.0, 1.0), WHITE);
        // offset 10 + 5*2 = 20, then x=1 at scale 6.
        assert_eq!(bounds(&list), Rect::new(26.0, 10.0, 6.0, 6.0));
        list.clear();

        list.push_transform((100.0, 0.0), 1.0);
        list.push_clip(Rect::new(0.0, 0.0, 5.0, 5.0));
        list.pop_transform();
        // The clip stayed where it was pushed; the rect at 0 is now outside it.
        list.rect(Rect::new(0.0, 0.0, 50.0, 50.0), WHITE);
        assert!(list.vertices().is_empty());
    }
}
