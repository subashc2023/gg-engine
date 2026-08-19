//! Input routing (§4.9): where the pointer is, what it is over, what has focus,
//! and what took the press.
//!
//! # The pointer is derived, never read
//!
//! There is no "ask the OS where the cursor is" here, and there is deliberately
//! no way to set the position from outside. [`Pointer`] is an *integral of the
//! replayed axis stream*: motion arrives as [`AXIS_SCALE`]ths of a pixel in an
//! [`InputFrame`], the same fixed-point integers a replay file holds, and the
//! position accumulates and clamps in that same fixed point. Nothing here
//! divides until a hit test needs pixels, and that divide is exact.
//!
//! That is the whole of §4.7's "a replayed click lands on the same widget": the
//! cursor is sim-shaped state derived from the recorded stream, so a replay
//! reproduces it by construction rather than by the mouse driver agreeing with
//! itself. The cost is that the OS cursor and this one are different things — a
//! host in UI mode hides the system cursor and draws this one.
//!
//! # One frame of lag, on purpose
//!
//! Immediate mode declares widgets in draw order, and the topmost hit is the
//! *last* one declared — which cannot be known until the frame is built. So
//! [`Router::begin`] resolves hover, capture and focus against the geometry the
//! **previous** frame declared, and [`Router::hit`] answers from that resolution
//! while recording this frame's geometry for the next one. The lag is uniform
//! and deterministic, which is what the replay criterion actually needs; a
//! two-pass build would remove it and cost every caller a second traversal.
//!
//! # Not here yet
//!
//! Scroll chaining, named in §4.9's owned list, has no scrollable container to
//! chain through. It arrives with the widget that needs it, by the same rule
//! that keeps widgets demo-pulled.

use crate::draw::Rect;
use gg_input::{ActionId, AxisId, Input};

pub use gg_input::AXIS_SCALE;

/// A widget's identity across frames — what hover, focus and capture are
/// remembered against.
///
/// FNV-1a over the path, `const`-evaluable so an id costs nothing per frame:
/// `const OK: WidgetId = WidgetId::new("settings.ok");`. Deliberately *not*
/// `gg-ecs`'s blake3 stable-id scheme — a widget id is process-local, never
/// persisted and never crosses an artifact boundary, so nothing here needs the
/// property that scheme is paying for. What catches a collision is
/// [`Router::duplicate`], which sees the case that actually matters: two live
/// widgets sharing one id in one frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WidgetId(u64);

impl WidgetId {
    /// The id for `path`. Paths are conventionally dotted (`"settings.vsync"`)
    /// so a panel's widgets share a prefix and a grep finds them.
    ///
    /// The hash itself is `gg_ecs::boundary::widget_id`'s, not a second copy of
    /// it: a game declares its buttons on the far side of §3's deny pin and
    /// cannot call this constructor, so the two sides would be one silent
    /// divergence away from routing every click to nothing.
    #[must_use]
    pub const fn new(path: &str) -> Self {
        WidgetId(gg_ecs::boundary::widget_id(path))
    }

    /// An id a game already computed, arriving through
    /// [`Widget::id`](gg_ecs::boundary::Widget::id). Any `u64` is a valid
    /// identity — what a duplicate costs is reported by [`Router::duplicate`],
    /// not prevented here.
    #[must_use]
    pub const fn from_hash(hash: u64) -> Self {
        WidgetId(hash)
    }

    /// The `index`th instance of this id — for a row in a list, where one path
    /// names many widgets.
    #[must_use]
    pub const fn indexed(self, index: u64) -> Self {
        let mut hash = self.0 ^ index;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        WidgetId(hash)
    }

    /// The raw hash, for a caller that wants to log or sort by it.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// The pointer position, in [`AXIS_SCALE`]ths of a physical pixel.
///
/// Fixed point rather than `f32` because this is an *accumulator*: a float
/// position summed from float deltas over thousands of ticks drifts by rounding
/// alone, and two runs of the same replay would drift the same way only by luck.
/// Integers accumulate exactly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pointer {
    x: i32,
    y: i32,
}

impl Pointer {
    /// Position in [`AXIS_SCALE`]ths, which is what it actually is.
    ///
    /// For a host differencing against an OS cursor position (§6 M15.1): the
    /// delta it must feed is `desired - raw()`, and computing that in pixels
    /// would put a rounding step in the one place the two must agree exactly.
    #[must_use]
    pub fn raw(self) -> (i32, i32) {
        (self.x, self.y)
    }

    /// Position in pixels. Exact: [`AXIS_SCALE`] is a power of two.
    #[must_use]
    pub fn position(self) -> (f32, f32) {
        (
            self.x as f32 / AXIS_SCALE as f32,
            self.y as f32 / AXIS_SCALE as f32,
        )
    }

    /// Whether the pointer is inside `rect`, which must already be in physical
    /// pixels — [`crate::DrawList::place`] is what puts it there.
    #[must_use]
    pub fn inside(self, rect: &Rect) -> bool {
        let (x, y) = self.position();
        rect.contains(x, y)
    }

    /// Add this tick's motion and clamp to a surface `extent` pixels across.
    ///
    /// Saturating throughout: an absurd delta parks the pointer at an edge
    /// rather than wrapping it to the other one.
    fn advance(&mut self, motion: (i32, i32), extent: (u32, u32)) {
        let limit = |pixels: u32| {
            i64::from(pixels)
                .saturating_mul(i64::from(AXIS_SCALE))
                .clamp(0, i64::from(i32::MAX)) as i32
        };
        self.x = self.x.saturating_add(motion.0).clamp(0, limit(extent.0));
        self.y = self.y.saturating_add(motion.1).clamp(0, limit(extent.1));
    }
}

/// Which actions and axes the host's UI context binds (§4.7). The router never
/// reaches into an action map itself — it is handed the ids, so what "click"
/// means stays a rebindable config decision.
#[derive(Clone, Copy, Debug)]
pub struct Binding {
    /// Pointer motion, horizontal.
    pub x: AxisId,
    /// Pointer motion, vertical.
    pub y: AxisId,
    /// The click/drag button.
    pub primary: ActionId,
    /// Move focus to the next widget.
    pub advance_focus: ActionId,
    /// Press the focused widget, if this build declared a verb for it — the
    /// keyboard's half of a click (§6 M74).
    pub activate: Option<ActionId>,
    /// One wheel notch away from the operator, if this build declared one.
    pub scroll_up: Option<ActionId>,
    /// One toward.
    pub scroll_down: Option<ActionId>,
}

/// One tick of input as the router consumes it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tick {
    /// Pointer motion this tick, in [`AXIS_SCALE`]ths of a pixel.
    pub motion: (i32, i32),
    /// The primary button, *held* — the router derives its own edges, because
    /// capture spans the whole press and a caller's edge would not.
    pub primary: bool,
    /// Focus moves on this tick. An edge, not a held state: a held key must not
    /// walk the whole panel in a second.
    pub advance_focus: bool,
    /// The focused widget is pressed on this tick. An edge for
    /// [`advance_focus`](Self::advance_focus)'s reason, one key later.
    pub activate: bool,
    /// Wheel notches this tick: `+1` away from the operator, `-1` toward, `0`
    /// for the overwhelming majority of ticks.
    ///
    /// One notch at most in each direction because that is what the source is
    /// (`gg_input::Wheel`) — a tick either notched or it did not. Both at once
    /// is a wheel spun both ways inside 16 ms, which cancels.
    pub scroll: i32,
}

impl Tick {
    /// Resolve a tick out of live or replayed action state. Identical either
    /// way — that is what [`Input`] is for (§4.7).
    #[must_use]
    pub fn from_input(input: &Input, binding: &Binding) -> Tick {
        let axes = input.frame().axes;
        // `pressed` and not `just_pressed`: a notch is already an edge — it is
        // set for the one tick it arrived in — and asking for the edge of an
        // edge drops every second notch of a steady scroll.
        let notch = |verb: Option<ActionId>| i32::from(verb.is_some_and(|id| input.pressed(id)));
        Tick {
            motion: (axes[binding.x.index()], axes[binding.y.index()]),
            primary: input.pressed(binding.primary),
            advance_focus: input.just_pressed(binding.advance_focus),
            activate: binding.activate.is_some_and(|id| input.just_pressed(id)),
            scroll: notch(binding.scroll_up) - notch(binding.scroll_down),
        }
    }
}

/// What one widget learned about this frame's input.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Response {
    /// The pointer is over this widget and nothing is drawn on top of it.
    pub hovered: bool,
    /// This widget took the press and the button is still down — a drag.
    pub held: bool,
    /// The press this widget took was released over it. A press dragged off and
    /// released elsewhere is deliberately *not* a click.
    pub clicked: bool,
    /// Keys are going here.
    pub focused: bool,
}

/// Hover, focus and capture across frames.
///
/// Allocation-free in steady state: the per-frame geometry lives in two `Vec`s
/// that are cleared and refilled, never reallocated once a UI reaches its usual
/// size (§6 M13's exit criterion).
#[derive(Debug, Default)]
pub struct Router {
    pointer: Pointer,
    hovered: Option<WidgetId>,
    focused: Option<WidgetId>,
    /// The widget the current press belongs to. Outlives the pointer leaving it
    /// — that is what makes a drag a drag.
    captured: Option<WidgetId>,
    /// Resolved at [`Router::begin`] and read by every [`Router::hit`] in the
    /// frame, so exactly one widget can report a click.
    clicked: Option<WidgetId>,
    primary: bool,
    /// This frame's declarations, in draw order. Read at the *next* `begin`.
    declared: Vec<(WidgetId, Rect)>,
    /// Scratch for the duplicate check, sorted in place so `declared` keeps the
    /// draw order the hit test depends on.
    sorted: Vec<WidgetId>,
    duplicate: Option<WidgetId>,
}

impl Router {
    /// Where the pointer is.
    #[must_use]
    pub fn pointer(&self) -> Pointer {
        self.pointer
    }

    /// The focused widget, if any.
    #[must_use]
    pub fn focused(&self) -> Option<WidgetId> {
        self.focused
    }

    /// An id two widgets declared in the last frame, if any.
    ///
    /// The classic immediate-mode footgun: two widgets sharing an id fight over
    /// focus and capture, and the symptom is a button that sometimes belongs to
    /// a different button. Exposed rather than only logged so a UI test can
    /// assert there are none.
    #[must_use]
    pub fn duplicate(&self) -> Option<WidgetId> {
        self.duplicate
    }

    /// Open a frame: advance the pointer, then resolve hover, capture and focus
    /// against the geometry the previous frame declared.
    ///
    /// `extent` is the surface in physical pixels — the same space
    /// [`crate::DrawList::place`] resolves a rect into.
    pub fn begin(&mut self, tick: &Tick, extent: (u32, u32)) {
        self.pointer.advance(tick.motion, extent);
        self.check_duplicates();

        self.hovered = self.resolve_hover();
        let pressed = tick.primary && !self.primary;
        let released = !tick.primary && self.primary;
        self.primary = tick.primary;

        self.clicked = None;
        if pressed {
            self.captured = self.hovered;
            // Pressing empty space clears focus, so Escape is not the only way
            // out of a text field.
            self.focused = self.hovered;
        }
        if released {
            // A click is a release over the widget that took the press. Dragging
            // off a button and letting go is the standard way to change your
            // mind, and it must not fire.
            self.clicked = self.captured.take().filter(|id| Some(*id) == self.hovered);
        }
        // The keyboard's half of a click (§6 M74), and deliberately not folded
        // into the button's press/release above: a click is a *gesture* that can
        // be dragged off a widget and abandoned, and a key has nowhere to drag
        // to. So this writes the click directly and touches neither capture nor
        // focus — pressing a button with Enter leaves the ring where it was,
        // which is what a menu wants.
        //
        // Before the advance, so a tick carrying both presses what the ring was
        // on rather than what it is about to move to.
        //
        // A screen that has just appeared has *nothing* focused — no game can
        // set the ring (there is no setter, deliberately: focus is the
        // player's) — so M74's keyboard only worked on a page somebody had
        // already pressed Tab on, and a title screen offering a key nobody
        // advertised is a title screen with no keyboard at all (§6 M76). The
        // fallback is the *first* row, which is the one a reader's eye is on,
        // and it leaves the ring alone on purpose: focus is a choice and this
        // is the absence of one, so the glass shows no highlight for a
        // decision the player has not made. `after(None)` is already "the
        // first declaration with somewhere to be pressed".
        if tick.activate {
            let target = self
                .focused
                .filter(|id| self.area(*id))
                .or_else(|| self.after(None));
            if let Some(id) = target {
                self.clicked = Some(id);
            }
        }
        if tick.advance_focus {
            self.focused = self.after(self.focused);
        }

        self.declared.clear();
    }

    /// Declare a widget occupying `rect` and read what the frame did to it.
    ///
    /// `rect` is in physical pixels; pass [`crate::DrawList::place`]'s result so
    /// the clip and transform stacks that decide where a widget *drew* are the
    /// same ones that decide where it can be *hit*.
    pub fn hit(&mut self, id: WidgetId, rect: Rect) -> Response {
        self.declared.push((id, rect));
        Response {
            hovered: self.hovered == Some(id),
            held: self.captured == Some(id) && self.primary,
            clicked: self.clicked == Some(id),
            focused: self.focused == Some(id),
        }
    }

    /// The last-declared widget under the pointer — last, because later
    /// declarations draw over earlier ones and the topmost is what a human aims
    /// at.
    ///
    /// A live press narrows this to its owner: during a drag nothing else may
    /// light up, and the owner itself lights up only while the pointer is
    /// genuinely over it. Both halves are load-bearing. Letting capture *claim*
    /// hover unconditionally would make "released over the widget" trivially
    /// true and destroy the drag-off-to-cancel gesture; letting geometry win
    /// outright would highlight whatever a drag passes across.
    fn resolve_hover(&self) -> Option<WidgetId> {
        let under = self
            .declared
            .iter()
            .rev()
            .find(|(_, rect)| self.pointer.inside(rect))
            .map(|(id, _)| *id);
        match self.captured {
            Some(owner) => (under == Some(owner)).then_some(owner),
            None => under,
        }
    }

    /// The widget declared after `current`, wrapping; the first when nothing is
    /// focused or the focused widget is gone.
    /// The next widget with **area**, wrapping.
    ///
    /// A zero rect is how a game hides a widget (§4.9), and every hidden menu in
    /// the tree is still declared every frame — so a walk over `declared` visits
    /// a pause menu's five buttons while the player is mid-game. That was merely
    /// untidy while focus was the only thing it moved and became a hazard the
    /// moment a key could *press* what it landed on (§6 M74). Same rule as
    /// `resolve_hover`'s, for the same reason: a widget nobody can point at is
    /// not a widget anybody can reach.
    fn after(&self, current: Option<WidgetId>) -> Option<WidgetId> {
        let shown = |(id, rect): &(WidgetId, Rect)| (rect.w > 0.0 && rect.h > 0.0).then_some(*id);
        let at = current.and_then(|id| self.declared.iter().position(|(d, _)| *d == id));
        // From just after the current one, wrapping — which is one pass over a
        // rotation of the list rather than two searches.
        let from = at.map_or(0, |at| at + 1);
        self.declared[from..]
            .iter()
            .chain(&self.declared[..from])
            .find_map(shown)
            .or(current)
    }

    /// Whether a declared widget has somewhere to be pressed.
    fn area(&self, id: WidgetId) -> bool {
        self.declared
            .iter()
            .any(|(d, rect)| *d == id && rect.w > 0.0 && rect.h > 0.0)
    }

    fn check_duplicates(&mut self) {
        self.sorted.clear();
        self.sorted.extend(self.declared.iter().map(|(id, _)| *id));
        self.sorted.sort_unstable();
        self.duplicate = self
            .sorted
            .windows(2)
            .find(|pair| pair[0] == pair[1])
            .map(|pair| pair[0]);
        if let Some(id) = self.duplicate {
            tracing::warn!(id = id.get(), "two widgets declared one id in a frame");
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const EXTENT: (u32, u32) = (200, 100);
    const A: WidgetId = WidgetId::new("a");
    const B: WidgetId = WidgetId::new("b");

    fn moved(dx: f32, dy: f32) -> Tick {
        Tick {
            motion: (
                (dx * AXIS_SCALE as f32) as i32,
                (dy * AXIS_SCALE as f32) as i32,
            ),
            ..Tick::default()
        }
    }

    /// Two rectangles side by side; declaring both every frame is what a real
    /// immediate-mode build does and what the lag is measured against.
    fn declare(router: &mut Router) -> (Response, Response) {
        let a = router.hit(A, Rect::new(0.0, 0.0, 50.0, 50.0));
        let b = router.hit(B, Rect::new(50.0, 0.0, 50.0, 50.0));
        (a, b)
    }

    #[test]
    fn the_pointer_accumulates_in_fixed_point_and_clamps_to_the_surface() {
        let mut router = Router::default();
        router.begin(&moved(10.5, 4.25), EXTENT);
        assert_eq!(router.pointer().position(), (10.5, 4.25));
        router.begin(&moved(10.5, 0.0), EXTENT);
        assert_eq!(router.pointer().position(), (21.0, 4.25));
        // Off the right edge and back: a clamp, not a wrap.
        router.begin(&moved(1e6, 1e6), EXTENT);
        assert_eq!(router.pointer().position(), (200.0, 100.0));
        router.begin(&moved(-1e6, -1e6), EXTENT);
        assert_eq!(router.pointer().position(), (0.0, 0.0));
    }

    /// The property the whole design exists for: the same motion stream puts the
    /// pointer in the same place, bit for bit, however it is chopped into ticks.
    /// A float accumulator fails the second half of this.
    #[test]
    fn the_same_motion_stream_lands_on_the_same_pixel_every_run() {
        let run = |chunks: &[(f32, f32)]| {
            let mut router = Router::default();
            for &(dx, dy) in chunks {
                router.begin(&moved(dx, dy), EXTENT);
            }
            router.pointer()
        };
        let tenth = 0.1_f32;
        let many: Vec<(f32, f32)> = (0..300).map(|_| (tenth, tenth)).collect();
        assert_eq!(run(&many), run(&many));
        // 0.1 is not representable; the quantized delta is, so 300 of them is an
        // exact integer count of AXIS_SCALE-ths and not 300 accumulated errors.
        let quantized = (tenth * AXIS_SCALE as f32) as i32;
        assert_eq!(
            run(&many),
            Pointer {
                x: quantized * 300,
                y: quantized * 300,
            }
        );
    }

    /// Hover is resolved against the *previous* frame's declarations, so the
    /// first frame a widget exists it cannot be hovered — and the second it can.
    #[test]
    fn hover_costs_one_frame_and_then_tracks() {
        let mut router = Router::default();
        router.begin(&moved(10.0, 10.0), EXTENT);
        let (a, b) = declare(&mut router);
        assert!(!a.hovered && !b.hovered, "nothing was declared last frame");

        router.begin(&Tick::default(), EXTENT);
        let (a, b) = declare(&mut router);
        assert!(a.hovered && !b.hovered);

        router.begin(&moved(60.0, 0.0), EXTENT);
        let (a, b) = declare(&mut router);
        assert!(!a.hovered && b.hovered);
    }

    /// The topmost declaration wins, because later ones draw over earlier ones.
    #[test]
    fn overlapping_widgets_are_claimed_by_the_one_drawn_last() {
        let mut router = Router::default();
        router.begin(&moved(10.0, 10.0), EXTENT);
        router.hit(A, Rect::new(0.0, 0.0, 50.0, 50.0));
        router.hit(B, Rect::new(0.0, 0.0, 50.0, 50.0));

        router.begin(&Tick::default(), EXTENT);
        let a = router.hit(A, Rect::new(0.0, 0.0, 50.0, 50.0));
        let b = router.hit(B, Rect::new(0.0, 0.0, 50.0, 50.0));
        assert!(b.hovered && !a.hovered);
    }

    /// Abutting rectangles: the shared edge belongs to exactly one of them, so
    /// no pointer position is over two widgets at once.
    #[test]
    fn a_shared_edge_belongs_to_one_widget() {
        let mut router = Router::default();
        router.begin(&moved(50.0, 10.0), EXTENT);
        declare(&mut router);
        router.begin(&Tick::default(), EXTENT);
        let (a, b) = declare(&mut router);
        assert!(!a.hovered && b.hovered);
    }

    /// Press, drag off, release: not a click. This is how a human changes their
    /// mind, and a UI that fires anyway is a UI nobody trusts with a delete
    /// button.
    #[test]
    fn a_press_dragged_off_its_widget_is_not_a_click() {
        let mut router = Router::default();
        let down = Tick {
            primary: true,
            ..Tick::default()
        };
        router.begin(&moved(10.0, 10.0), EXTENT);
        declare(&mut router);
        router.begin(&Tick::default(), EXTENT);
        declare(&mut router);

        router.begin(&down, EXTENT);
        let (a, _) = declare(&mut router);
        assert!(a.held && !a.clicked);

        // Dragged onto B, still held. A keeps the press; nothing lights up —
        // not B, which does not own it, and not A, which the pointer has left.
        router.begin(
            &Tick {
                motion: moved(60.0, 0.0).motion,
                ..down
            },
            EXTENT,
        );
        let (a, b) = declare(&mut router);
        assert!(
            a.held && !a.hovered,
            "the press is still A's, the pointer is not"
        );
        assert!(
            !b.hovered && !b.held,
            "a drag lights nothing it passes over"
        );

        router.begin(&Tick::default(), EXTENT);
        let (a, b) = declare(&mut router);
        assert!(
            !a.clicked && !b.clicked,
            "released off A, and B never had it"
        );
    }

    #[test]
    fn a_press_and_release_over_one_widget_is_a_click_exactly_once() {
        let mut router = Router::default();
        let down = Tick {
            primary: true,
            ..Tick::default()
        };
        router.begin(&moved(10.0, 10.0), EXTENT);
        declare(&mut router);
        router.begin(&Tick::default(), EXTENT);
        declare(&mut router);

        router.begin(&down, EXTENT);
        declare(&mut router);
        router.begin(&Tick::default(), EXTENT);
        let (a, b) = declare(&mut router);
        assert!(a.clicked && !b.clicked);
        // And the click does not repeat while the button stays up.
        router.begin(&Tick::default(), EXTENT);
        let (a, _) = declare(&mut router);
        assert!(!a.clicked);
    }

    /// The keyboard's whole path (§6 M74): Tab lands the ring somewhere, the
    /// press verb clicks *that*, and none of it moves the pointer or the ring.
    #[test]
    fn a_focused_widget_can_be_pressed_without_a_pointer() {
        let mut router = Router::default();
        let tab = Tick {
            advance_focus: true,
            ..Tick::default()
        };
        let enter = Tick {
            activate: true,
            ..Tick::default()
        };
        // The pointer is parked off both widgets for the whole test, so nothing
        // below can be a hover in disguise.
        router.begin(&moved(150.0, 90.0), EXTENT);
        declare(&mut router);
        router.begin(&tab, EXTENT);
        let (a, b) = declare(&mut router);
        assert_eq!(router.focused(), Some(A));
        assert!(!a.clicked && !b.clicked, "moving the ring presses nothing");

        router.begin(&enter, EXTENT);
        let (a, b) = declare(&mut router);
        assert!(a.clicked && !b.clicked);
        assert!(!a.hovered && !a.held, "a key is not a pointer");
        assert_eq!(router.focused(), Some(A), "the ring stays where it was");

        // Once, and only while the key is down — an edge like every other.
        router.begin(&Tick::default(), EXTENT);
        assert!(!declare(&mut router).0.clicked);
        router.begin(&tab, EXTENT);
        declare(&mut router);
        router.begin(&enter, EXTENT);
        let (a, b) = declare(&mut router);
        assert!(
            b.clicked && !a.clicked,
            "the ring moved on and so did the press"
        );

        // With nothing focused the press falls to the first row (§6 M76) —
        // and *only* the first, so it is a default rather than a broadcast.
        // Through M75 this asserted that nothing was pressed at all, which
        // is the state a screen appears in and therefore the state a title
        // screen would have been stuck in.
        router.begin(&moved(0.0, 0.0), EXTENT);
        router.focused = None;
        declare(&mut router);
        router.begin(&enter, EXTENT);
        let (a, b) = declare(&mut router);
        assert!(a.clicked && !b.clicked);
        assert_eq!(
            router.focused(),
            None,
            "a default is not a choice, so the ring stays dark"
        );
    }

    /// The fallback above has somewhere to fall to only if the page has a row
    /// a pointer could reach. A page whose buttons are all hidden — every
    /// menu in the tree, most of the time (§4.9) — presses nothing, which is
    /// what stops Enter mid-game from reaching a pause menu's first button.
    #[test]
    fn a_page_with_no_reachable_row_presses_nothing() {
        let mut router = Router::default();
        let enter = Tick {
            activate: true,
            ..Tick::default()
        };
        let hidden = |router: &mut Router| {
            let a = router.hit(A, Rect::new(0.0, 0.0, 0.0, 0.0));
            let b = router.hit(B, Rect::new(0.0, 0.0, 50.0, 0.0));
            (a, b)
        };
        router.begin(&moved(500.0, 500.0), EXTENT);
        hidden(&mut router);
        router.begin(&enter, EXTENT);
        let (a, b) = hidden(&mut router);
        assert!(!a.clicked && !b.clicked, "a zero rect is not a row");

        // The same tick with one of them given a body does press it, so the
        // refusal above is the area rule and not a dead code path.
        router.begin(&moved(0.0, 0.0), EXTENT);
        declare(&mut router);
        router.begin(&enter, EXTENT);
        assert!(declare(&mut router).0.clicked);
    }

    /// A zero rect is how a game hides a widget (§4.9) and hidden menus are
    /// still declared every frame, so the walk has to skip them — untidy while
    /// focus was cosmetic, and a hazard once a key can press what it lands on.
    ///
    /// Graded in both directions: the ring never stops on the hidden one, and a
    /// press cannot reach it even when focus is put there by hand — which is
    /// also the state a screen change leaves behind, focus being stale on a
    /// button the new page does not show. Where the press goes *instead* is
    /// §6 M76's: the first row that has somewhere to be pressed, so a stale
    /// ring and no ring at all behave alike.
    #[test]
    fn the_ring_walks_past_a_widget_a_pointer_could_not_reach() {
        let mut router = Router::default();
        let hidden = |router: &mut Router| {
            let a = router.hit(A, Rect::new(0.0, 0.0, 50.0, 50.0));
            let b = router.hit(B, Rect::new(0.0, 0.0, 0.0, 0.0));
            (a, b)
        };
        router.begin(&moved(150.0, 90.0), EXTENT);
        hidden(&mut router);
        for _ in 0..3 {
            router.begin(
                &Tick {
                    advance_focus: true,
                    ..Tick::default()
                },
                EXTENT,
            );
            hidden(&mut router);
            assert_eq!(router.focused(), Some(A), "B has nowhere to be pressed");
        }
        router.focused = Some(B);
        router.begin(
            &Tick {
                activate: true,
                ..Tick::default()
            },
            EXTENT,
        );
        let (a, b) = hidden(&mut router);
        assert!(!b.clicked, "B has nowhere to be pressed");
        assert!(a.clicked, "so the press falls to the row that has");
        assert_eq!(router.focused(), Some(B), "and the ring is not moved by it");
    }

    #[test]
    fn a_press_moves_focus_and_pressing_nothing_clears_it() {
        let mut router = Router::default();
        let down = Tick {
            primary: true,
            ..Tick::default()
        };
        router.begin(&moved(10.0, 10.0), EXTENT);
        declare(&mut router);
        router.begin(&down, EXTENT);
        declare(&mut router);
        assert_eq!(router.focused(), Some(A));

        // Up, then down again over empty space.
        router.begin(&moved(150.0, 0.0), EXTENT);
        declare(&mut router);
        router.begin(&down, EXTENT);
        declare(&mut router);
        assert_eq!(router.focused(), None);
    }

    #[test]
    fn focus_advances_in_declaration_order_and_wraps() {
        let mut router = Router::default();
        let tab = Tick {
            advance_focus: true,
            ..Tick::default()
        };
        router.begin(&Tick::default(), EXTENT);
        declare(&mut router);

        router.begin(&tab, EXTENT);
        let (a, b) = declare(&mut router);
        assert!(a.focused && !b.focused);
        router.begin(&tab, EXTENT);
        let (a, b) = declare(&mut router);
        assert!(!a.focused && b.focused);
        router.begin(&tab, EXTENT);
        let (a, b) = declare(&mut router);
        assert!(a.focused && !b.focused, "wrapped");
    }

    /// Two widgets, one id — the bug that makes a button belong to a different
    /// button. Reported rather than silently arbitrated.
    #[test]
    fn one_id_declared_twice_is_reported() {
        let mut router = Router::default();
        router.begin(&Tick::default(), EXTENT);
        assert_eq!(router.duplicate(), None);
        router.hit(A, Rect::new(0.0, 0.0, 10.0, 10.0));
        router.hit(A, Rect::new(20.0, 0.0, 10.0, 10.0));
        router.begin(&Tick::default(), EXTENT);
        assert_eq!(router.duplicate(), Some(A));
        // And it is *this* frame's answer, not a latch.
        declare(&mut router);
        router.begin(&Tick::default(), EXTENT);
        assert_eq!(router.duplicate(), None);
    }

    #[test]
    fn ids_are_distinct_by_path_and_by_index() {
        assert_ne!(A, B);
        assert_ne!(A, A.indexed(0));
        assert_ne!(A.indexed(0), A.indexed(1));
        assert_ne!(WidgetId::new("a.b"), WidgetId::new("ab"));
    }

    /// Steady state allocates nothing: the two frame `Vec`s reach their size and
    /// stay there. Measured as capacity rather than through an allocator hook,
    /// because the hook is the *shell's* proof (§6 M13) and this is the crate's.
    #[test]
    fn a_settled_frame_reuses_its_buffers() {
        let mut router = Router::default();
        for _ in 0..8 {
            router.begin(&Tick::default(), EXTENT);
            declare(&mut router);
        }
        let (declared, sorted) = (router.declared.capacity(), router.sorted.capacity());
        for _ in 0..64 {
            router.begin(&moved(1.0, 1.0), EXTENT);
            declare(&mut router);
        }
        assert_eq!(
            (router.declared.capacity(), router.sorted.capacity()),
            (declared, sorted)
        );
    }
}
