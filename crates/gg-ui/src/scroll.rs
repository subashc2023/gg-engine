//! Scrolling (§4.9's owned list, §6 M15.1) — geometry only, and deliberately
//! not a container.
//!
//! Nothing here retains anything: a caller keeps its own offset, hands it in
//! with the content's height, and gets back where its rows land and where the
//! bar is. That is the same shape [`crate::dock`] and [`crate::menu`] take, and
//! it is what lets an offset be *derived state* — the editor's is a pure
//! function of the click and notch stream, so it replays without being hashed.
//!
//! # A row outside the view must not be declared
//!
//! Clipping is a drawing operation: [`crate::DrawList`] cuts the quads and the
//! router never hears about it, so a widget declared off the top of a scrolled
//! pane is invisible and still takes clicks — under whatever pane is actually
//! up there. [`Scroll::row`] answers `None` for exactly those, and a caller
//! that skips them is a caller whose hit-testing agrees with its picture.

use crate::draw::Rect;

/// How wide the bar is, in the caller's units. Thin: it is chrome down the edge
/// of somebody else's pane, and the pane is what the operator is reading.
pub const BAR: f32 = 4.0;

/// The shortest a thumb may be drawn. A thousand rows in a ten-row pane would
/// otherwise give a thumb thinner than the line it is drawn with, which is a
/// bar with nothing to grab.
const MIN_THUMB: f32 = 6.0;

/// Where a scrolled pane's content lands, and where its bar is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scroll {
    /// Where content may be seen: the body, less the bar when there is one.
    pub view: Rect,
    /// The clamped offset — what the caller stores back, since a pane that
    /// shrank has an offset its content no longer reaches.
    pub offset: f32,
    /// The furthest the content may be moved up. Zero when it all fits.
    pub max: f32,
    /// The track and the thumb, or `None` when there is nothing to scroll.
    pub bar: Option<(Rect, Rect)>,
}

impl Scroll {
    /// Resolve `body` holding `content` units of rows, scrolled by `offset`.
    #[must_use]
    pub fn new(body: Rect, content: f32, offset: f32) -> Scroll {
        let max = (content - body.h).max(0.0);
        let offset = offset.clamp(0.0, max);
        if max <= 0.0 {
            return Scroll {
                view: body,
                offset: 0.0,
                max: 0.0,
                bar: None,
            };
        }
        let view = Rect::new(body.x, body.y, (body.w - BAR).max(0.0), body.h);
        let track = Rect::new(body.right() - BAR, body.y, BAR, body.h);
        // The thumb is the visible fraction of the content, and its travel is
        // whatever the track has left over — so a thumb at the bottom of its
        // track and content at `max` are the same state by construction.
        let length = (body.h * body.h / content).clamp(MIN_THUMB.min(body.h), body.h);
        let at = track.y + (track.h - length) * (offset / max);
        Scroll {
            view,
            offset,
            max,
            bar: Some((track, Rect::new(track.x, at, track.w, length))),
        }
    }

    /// Where a row `at` units down the content lands, or `None` when it is
    /// entirely outside [`Scroll::view`] — see this module's second half.
    #[must_use]
    pub fn row(&self, at: f32, h: f32) -> Option<Rect> {
        let y = self.view.y - self.offset + at;
        (y + h > self.view.y && y < self.view.bottom())
            .then(|| Rect::new(self.view.x, y, self.view.w, h))
    }

    /// The offset that puts the thumb's middle under `y` — what a held bar
    /// steers by.
    ///
    /// Centred on the pointer rather than anchored where the press landed: an
    /// anchor is one more piece of retained state, and the difference is only
    /// visible on the first tick of a drag that started away from the thumb's
    /// middle — which is a press on the track, where jumping there is what the
    /// operator asked for anyway.
    #[must_use]
    pub fn offset_at(&self, y: f32) -> f32 {
        let Some((track, thumb)) = self.bar else {
            return 0.0;
        };
        let travel = track.h - thumb.h;
        if travel <= 0.0 {
            return 0.0;
        }
        ((y - track.y - thumb.h * 0.5) / travel * self.max).clamp(0.0, self.max)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const BODY: Rect = Rect::new(10.0, 20.0, 100.0, 50.0);

    /// Content that fits takes the whole body and grows no bar — the case every
    /// pane is in until it has something to say.
    #[test]
    fn content_that_fits_has_no_bar() {
        let scroll = Scroll::new(BODY, 30.0, 12.0);
        assert_eq!(scroll.view, BODY);
        assert_eq!(scroll.bar, None);
        assert_eq!(scroll.offset, 0.0, "an offset with nowhere to go is zero");
        assert_eq!(
            scroll.row(0.0, 8.0),
            Some(Rect::new(10.0, 20.0, 100.0, 8.0))
        );
    }

    /// The offset is clamped to what the content has, the view narrows by the
    /// bar, and the thumb is the visible fraction of the whole.
    #[test]
    fn a_scrolled_pane_narrows_and_clamps() {
        let scroll = Scroll::new(BODY, 200.0, 1000.0);
        assert_eq!(scroll.max, 150.0);
        assert_eq!(scroll.offset, 150.0);
        assert_eq!(scroll.view.w, BODY.w - BAR);
        let (track, thumb) = scroll.bar.expect("200 units in 50 needs a bar");
        assert_eq!(track.right(), BODY.right());
        assert!((thumb.h - 12.5).abs() < 0.01, "{thumb:?}");
        assert!(
            (thumb.bottom() - track.bottom()).abs() < 0.01,
            "a full offset leaves the thumb at the bottom: {thumb:?}"
        );
    }

    /// A row scrolled off either edge is refused rather than moved, because a
    /// caller that drew it would also have declared it clickable.
    #[test]
    fn rows_outside_the_view_are_refused() {
        let scroll = Scroll::new(BODY, 200.0, 60.0);
        assert_eq!(scroll.row(0.0, 8.0), None, "above");
        assert_eq!(scroll.row(190.0, 8.0), None, "below");
        // Straddling the top edge is still visible, and lands where it should.
        let row = scroll.row(56.0, 8.0).expect("straddles the top");
        assert!((row.y - 16.0).abs() < 0.01, "{row:?}");
        let inside = scroll.row(60.0, 8.0).expect("the first whole row");
        assert_eq!(inside.y, BODY.y);
        assert_eq!(inside.w, scroll.view.w);
    }

    /// Steering the bar and reading the thumb back are inverses, which is what
    /// makes a drag land where the pointer is.
    #[test]
    fn steering_the_thumb_round_trips() {
        let scroll = Scroll::new(BODY, 200.0, 0.0);
        let (_, thumb) = scroll.bar.expect("a bar");
        assert_eq!(scroll.offset_at(thumb.y + thumb.h * 0.5), 0.0);
        for want in [0.0, 37.5, 75.0, 150.0] {
            let moved = Scroll::new(BODY, 200.0, want);
            let (_, thumb) = moved.bar.expect("a bar");
            let back = moved.offset_at(thumb.y + thumb.h * 0.5);
            assert!((back - want).abs() < 0.01, "{want} came back {back}");
        }
        assert_eq!(scroll.offset_at(-1000.0), 0.0, "clamped");
        assert_eq!(scroll.offset_at(1000.0), scroll.max, "clamped");
    }
}
