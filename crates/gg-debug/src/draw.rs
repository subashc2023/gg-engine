//! The batched draw layer and its clip/transform stacks — the kernel of
//! `gg-ui` (§4.9), landing here because §4.8's overlay is what first needs one.
//!
//! **And no further.** There is no hit-testing, no focus, no capture and no
//! widget: those are M13 features that M13 pays for. What is here is the part
//! the overlay genuinely uses and the part M13 would otherwise have to invent
//! twice — one vertex stream, rectangles and glyphs batched into it, and stacks
//! that let a panel position and bound its contents.
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
    /// A rectangle from its edges.
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
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

    /// Width of `text` in this font, unscaled.
    pub fn width(text: &str) -> f32 {
        text.chars().count() as f32 * font::CELL.0 as f32
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
        let clipped = match self.clips.last() {
            Some(clip) => placed.intersect(clip),
            None => placed,
        };
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
