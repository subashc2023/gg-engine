//! Absolute and stack layout (§4.9's owned list), which in immediate mode is a
//! pen and a bounding box rather than a tree.
//!
//! There is no node graph here and no measure/arrange protocol: a caller places
//! cells in the order it draws them and [`Stack`] answers where each one landed
//! and what they occupy together. Absolute layout is [`crate::Rect`] itself —
//! naming a rectangle is the whole of it — so what this module adds is the one
//! case a rectangle cannot state: a run of cells whose positions depend on the
//! sizes of the ones before them. `taffy` arrives when a demo asks for flex or
//! grid; a panel of rows is not that demo.

use crate::draw::Rect;

/// The direction a [`Stack`] advances its pen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// Cells stack downwards and the pen advances by height.
    Vertical,
    /// Cells stack rightwards and the pen advances by width.
    Horizontal,
}

/// Cells laid end to end along one [`Axis`].
///
/// The two-pass shape is the point. A panel whose background must fit its
/// contents cannot know its own size until those contents are measured, and an
/// immediate-mode draw list has to emit the background *first* — so a caller
/// pushes its cells, reads [`Stack::content`], draws the background, then
/// [`Stack::rewind`]s and pushes the same cells again to place them. Two walks
/// of a pen, no retained tree, and the rectangle the background was sized
/// against is by construction the one the cells land in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stack {
    origin: (f32, f32),
    pen: (f32, f32),
    axis: Axis,
    gap: f32,
    content: Rect,
}

impl Stack {
    /// A stack running down from `at`, with `gap` pixels between cells.
    #[must_use]
    pub fn vertical(at: (f32, f32), gap: f32) -> Self {
        Stack::new(at, Axis::Vertical, gap)
    }

    /// A stack running right from `at`.
    #[must_use]
    pub fn horizontal(at: (f32, f32), gap: f32) -> Self {
        Stack::new(at, Axis::Horizontal, gap)
    }

    fn new(at: (f32, f32), axis: Axis, gap: f32) -> Self {
        Stack {
            origin: at,
            pen: at,
            axis,
            gap,
            content: Rect::new(at.0, at.1, 0.0, 0.0),
        }
    }

    /// Reserve a `w` × `h` cell at the pen, advance past it, and report where
    /// it went. The gap follows the cell, so a trailing one never widens
    /// [`Stack::content`].
    pub fn push(&mut self, w: f32, h: f32) -> Rect {
        let cell = Rect::new(self.pen.0, self.pen.1, w, h);
        self.content = self.content.union(&cell);
        match self.axis {
            Axis::Vertical => self.pen.1 += h + self.gap,
            Axis::Horizontal => self.pen.0 += w + self.gap,
        }
        cell
    }

    /// Everything pushed so far as one rectangle — zero-sized at the origin
    /// before the first push, which is what an empty panel should measure.
    #[must_use]
    pub fn content(&self) -> Rect {
        self.content
    }

    /// Put the pen back at the origin, **keeping** the content box, so a second
    /// pass over the same cells lands where the first pass measured.
    ///
    /// Pushing a *different* set after a rewind grows the box to cover both.
    /// That is a caller error and is deliberately not detected: the check costs
    /// a per-cell record, and the failure it would catch is visible on screen.
    pub fn rewind(&mut self) {
        self.pen = self.origin;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn cells_follow_one_another_along_the_axis_with_the_gap_between_them() {
        let mut down = Stack::vertical((10.0, 20.0), 2.0);
        assert_eq!(down.push(30.0, 5.0), Rect::new(10.0, 20.0, 30.0, 5.0));
        assert_eq!(down.push(40.0, 5.0), Rect::new(10.0, 27.0, 40.0, 5.0));

        let mut right = Stack::horizontal((10.0, 20.0), 2.0);
        assert_eq!(right.push(30.0, 5.0), Rect::new(10.0, 20.0, 30.0, 5.0));
        assert_eq!(right.push(40.0, 5.0), Rect::new(42.0, 20.0, 40.0, 5.0));
    }

    /// The trailing gap is not content: a stack of three rows is three rows
    /// tall, not three rows and a gap, or every panel gains a dead stripe.
    #[test]
    fn content_covers_the_cells_and_not_the_gap_after_them() {
        let mut stack = Stack::vertical((0.0, 0.0), 4.0);
        assert_eq!(stack.content(), Rect::new(0.0, 0.0, 0.0, 0.0), "untouched");
        for width in [10.0, 30.0, 20.0] {
            stack.push(width, 6.0);
        }
        assert_eq!(stack.content(), Rect::new(0.0, 0.0, 30.0, 26.0));
    }

    /// The idiom the whole type exists for: measure, size the background,
    /// rewind, place. Every cell of the second pass is inside the panel the
    /// first pass sized — which is what makes the clip safe.
    #[test]
    fn a_rewound_pass_lands_inside_the_box_the_first_pass_measured() {
        let sizes = [(40.0, 6.0), (90.0, 6.0), (55.0, 6.0)];
        let mut stack = Stack::vertical((8.0, 8.0), 1.0);
        let first: Vec<Rect> = sizes.iter().map(|&(w, h)| stack.push(w, h)).collect();
        let panel = stack.content().inset(-4.0);

        stack.rewind();
        let second: Vec<Rect> = sizes.iter().map(|&(w, h)| stack.push(w, h)).collect();
        assert_eq!(first, second, "the same cells, in the same places");
        assert_eq!(stack.content().inset(-4.0), panel, "and the same box");
        for cell in second {
            assert_eq!(cell.intersect(&panel), cell, "{cell:?} escapes {panel:?}");
        }
    }
}
