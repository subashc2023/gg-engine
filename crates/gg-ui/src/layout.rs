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
use gg_ecs::boundary::CANVAS;
use gg_input::AXIS_SCALE;

/// How a canvas sits inside a surface: a scale, an offset, and the extent the
/// canvas itself has (§4.9).
///
/// One copy of an arithmetic that had grown three — the game's UI, the editor's
/// panels, and the host mapping an OS cursor into the space both hit-test in.
/// Three copies of a rounding rule is three chances for the arrow to sit a
/// pixel away from the widget it is over.
///
/// Two constructors because there are two jobs. A *game* wants [`Fit::new`]: a
/// fixed [`CANVAS`] scaled uniformly and centred, so a HUD is the same picture
/// at every window size and the mismatch shows as letterboxing. The *editor*
/// wants [`Fit::fill`]: the canvas is whatever the window is at the chosen
/// scale, so the panels reach the edges and there is no letterbox to leave
/// showing (§6 M15.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fit {
    /// Canvas units to surface pixels.
    pub scale: f32,
    /// Where the canvas's top-left lands, in surface pixels. Floored, so the
    /// letterboxing never straddles a pixel.
    pub offset: (f32, f32),
    /// The canvas this fits, in canvas units — what [`Fit::to_canvas`] clamps
    /// into and what a [`Router`](crate::Router) advances the pointer across.
    pub canvas: (u32, u32),
}

impl Fit {
    /// Fit [`CANVAS`] into a surface `target` pixels across.
    #[must_use]
    pub fn new(target: (u32, u32)) -> Fit {
        let scale = (target.0 as f32 / CANVAS.0 as f32).min(target.1 as f32 / CANVAS.1 as f32);
        let axis =
            |extent: u32, canvas: u32| ((extent as f32 - canvas as f32 * scale) * 0.5).floor();
        Fit {
            scale,
            offset: (axis(target.0, CANVAS.0), axis(target.1, CANVAS.1)),
            canvas: CANVAS,
        }
    }

    /// Cover `target` at `scale`, the canvas being however much of it that is.
    ///
    /// No offset and nothing left over — which is the whole point, and why the
    /// canvas is a *quotient* rather than a constant. A non-integer `scale`
    /// works but should be rare: the bitmap font is sampled nearest, so a
    /// fractional scale turns every stem into porridge.
    ///
    /// The quotient rounds **up**, so the canvas covers the surface rather than
    /// fitting inside it. A floor leaves `target % scale` pixels down the right
    /// edge and along the bottom that no pane reaches and only the clear
    /// covers — a hairline of clear colour at every extent the scale does not
    /// divide, which at whole scales is every odd one (§6 M15.1 item 5).
    /// Covering costs the opposite and the scissor already discards it: the
    /// last unit runs a fraction past the surface.
    #[must_use]
    pub fn fill(target: (u32, u32), scale: f32) -> Fit {
        // A zero scale is a divide, and a minimized window reaches here.
        let scale = scale.max(f32::MIN_POSITIVE);
        let cover = |extent: u32| (extent as f32 / scale).ceil() as u32;
        Fit {
            scale,
            offset: (0.0, 0.0),
            canvas: (cover(target.0), cover(target.1)),
        }
    }

    /// A surface-pixel position in canvas units, clamped to the canvas — the
    /// inverse of what [`crate::DrawList::push_transform`] does with this fit.
    ///
    /// Clamped rather than allowed off-canvas because the caller is a cursor:
    /// a pointer in the letterbox is at the edge of the UI, not outside it.
    #[must_use]
    pub fn to_canvas(&self, x: f32, y: f32) -> (f32, f32) {
        let axis = |v: f32, off: f32, limit: u32| {
            match self.scale > 0.0 {
                true => ((v - off) / self.scale).clamp(0.0, limit as f32),
                // A zero-pixel surface is a minimized window, not a divide.
                false => 0.0,
            }
        };
        (
            axis(x, self.offset.0, self.canvas.0),
            axis(y, self.offset.1, self.canvas.1),
        )
    }

    /// The inverse of [`to_canvas`](Self::to_canvas): where a canvas position
    /// lands on the surface, in pixels.
    ///
    /// Not clamped, because it cannot be off-surface — a canvas position is
    /// already inside the canvas, and outside is where the letterbox is. Here
    /// beside the forward map so a host warping the system cursor to the
    /// pointer and a host reading the cursor into it cannot disagree (§6 M15.1).
    #[must_use]
    pub fn to_surface(&self, x: f32, y: f32) -> (f32, f32) {
        (
            x * self.scale + self.offset.0,
            y * self.scale + self.offset.1,
        )
    }

    /// As [`to_canvas`](Self::to_canvas), in the [`AXIS_SCALE`]ths the pointer
    /// accumulates in — the units a host must difference in for its delta to
    /// land the derived pointer exactly on the cursor (§6 M15.1).
    #[must_use]
    pub fn to_canvas_fixed(&self, x: f32, y: f32) -> (i32, i32) {
        let (cx, cy) = self.to_canvas(x, y);
        let fixed = |v: f32| (v * AXIS_SCALE as f32).round() as i32;
        (fixed(cx), fixed(cy))
    }

    /// The motion that moves a pointer at `from` onto the system cursor at
    /// `(x, y)` — surface pixels in, [`AXIS_SCALE`]ths out (§6 M15.1).
    ///
    /// **A delta and not an assignment, deliberately.** The pointer is an
    /// integral of the recorded axis stream ([`crate::Router`]) and that is what
    /// makes a replayed click land on the same widget; a host that *set* the
    /// position would be a second input path and the recording would no longer
    /// describe the session. Steering keeps the arrow the operator sees and the
    /// arrow a hit test uses on the same pixel while every bit of it stays in
    /// the frame the recorder writes.
    ///
    /// Exact in one tick: [`Router::begin`](crate::Router::begin) adds this and
    /// clamps to the same canvas this clamped to, so the pointer lands on
    /// [`to_canvas_fixed`](Self::to_canvas_fixed) and not near it.
    #[must_use]
    pub fn steer(&self, from: (i32, i32), x: f32, y: f32) -> (i32, i32) {
        let to = self.to_canvas_fixed(x, y);
        (to.0 - from.0, to.1 - from.1)
    }
}

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

    /// The property the host relies on: a canvas rectangle drawn through the
    /// fit and a surface pixel mapped back through it are the same place. If
    /// these two disagree the cursor sits beside the widget it is over.
    #[test]
    fn a_surface_pixel_maps_back_to_the_canvas_unit_that_drew_it() {
        for target in [
            (1280, 720),
            (1920, 1080),
            (640, 360),
            (1000, 1000),
            (800, 300),
        ] {
            let fit = Fit::new(target);
            for canvas in [(0.0, 0.0), (100.0, 50.0), (639.0, 359.0), (320.5, 180.25)] {
                let px = (
                    canvas.0 * fit.scale + fit.offset.0,
                    canvas.1 * fit.scale + fit.offset.1,
                );
                let back = fit.to_canvas(px.0, px.1);
                assert!(
                    (back.0 - canvas.0).abs() < 1e-3 && (back.1 - canvas.1).abs() < 1e-3,
                    "{target:?}: {canvas:?} -> {px:?} -> {back:?}"
                );
            }
        }
    }

    /// §6 M15.1's first exit row, in the form that does not need a window: a
    /// steer puts the derived pointer *on* the system cursor, in one tick, at
    /// every window size — and through a real [`Router`], because the clamp
    /// that could disagree is the router's.
    ///
    /// What this cannot cover is whether the OS reports the position winit
    /// hands over; that is `xtask interactive`'s, and it is the only part of
    /// the claim a headless run has no access to.
    #[test]
    fn a_steer_lands_the_derived_pointer_on_the_system_cursor() {
        use crate::router::{Router, Tick};

        for target in [(1280, 720), (1920, 1080), (2560, 1080), (900, 900)] {
            let fit = Fit::new(target);
            let mut router = Router::default();
            // Corners, centre, and inside the letterbox on both axes — the
            // cases where a clamp or a rounding rule would show.
            for at in [
                (0.0, 0.0),
                (target.0 as f32 - 1.0, target.1 as f32 - 1.0),
                (target.0 as f32 * 0.5, target.1 as f32 * 0.5),
                (7.0, 3.0),
                (target.0 as f32 * 0.5 + 0.5, target.1 as f32 * 0.5 - 0.5),
            ] {
                let motion = fit.steer(router.pointer().raw(), at.0, at.1);
                router.begin(
                    &Tick {
                        motion,
                        ..Tick::default()
                    },
                    CANVAS,
                );
                assert_eq!(
                    router.pointer().raw(),
                    fit.to_canvas_fixed(at.0, at.1),
                    "{target:?} at {at:?}"
                );
            }
        }
    }

    /// What a filled canvas is for: every pixel of the surface is inside some
    /// pane. A floor here is a hairline of clear colour down the right edge and
    /// along the bottom, visible on any window the scale does not divide — and
    /// on a drag it appears and vanishes with the parity of the extent, which
    /// is the complaint that produced this test (§6 M15.1 item 5).
    #[test]
    fn a_filled_canvas_covers_every_pixel_of_its_surface() {
        for scale in [1.0, 2.0, 3.0, 1.5] {
            for target in [(1280, 720), (1281, 721), (1919, 1079), (2561, 1441), (1, 1)] {
                let fit = Fit::fill(target, scale);
                let covered = (fit.canvas.0 as f32 * scale, fit.canvas.1 as f32 * scale);
                let want = (target.0 as f32, target.1 as f32);
                assert!(
                    covered.0 >= want.0 && covered.1 >= want.1,
                    "{target:?} at {scale}: {:?} covers {covered:?}",
                    fit.canvas
                );
                // And by under a unit, or the layout is aiming off the surface
                // for room it was never given.
                assert!(
                    covered.0 - want.0 < scale && covered.1 - want.1 < scale,
                    "{target:?} at {scale}: {:?} overshoots to {covered:?}",
                    fit.canvas
                );
            }
        }
        // A minimized surface is still a zero, not a one.
        assert_eq!(Fit::fill((0, 0), 2.0).canvas, (0, 0));
    }

    /// A pointer in the letterbox is at the edge of the UI, not off it — and a
    /// minimized surface is a zero, not a divide.
    #[test]
    fn the_letterbox_clamps_and_a_zero_surface_does_not_divide() {
        let wide = Fit::new((2000, 360));
        assert!(wide.offset.0 > 0.0 && wide.offset.1 == 0.0);
        assert_eq!(wide.to_canvas(0.0, 0.0), (0.0, 0.0), "left of the canvas");
        assert_eq!(wide.to_canvas(1999.0, 0.0).0, 640.0, "right of it");
        assert_eq!(Fit::new((0, 0)).to_canvas(4.0, 8.0), (0.0, 0.0));
    }

    /// Fixed point is what the pointer accumulates in, so the host differences
    /// there and nowhere else.
    #[test]
    fn the_fixed_point_form_is_the_pixel_form_scaled() {
        let fit = Fit::new((1280, 720));
        assert_eq!(fit.to_canvas_fixed(0.0, 0.0), (0, 0));
        assert_eq!(fit.to_canvas(20.0, 40.0), (10.0, 20.0));
        assert_eq!(
            fit.to_canvas_fixed(20.0, 40.0),
            (10 * AXIS_SCALE, 20 * AXIS_SCALE)
        );
    }

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
