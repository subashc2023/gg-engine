//! The polling API (§4.7): `input.pressed(SPAWN)`, `input.axis(MOVE_RIGHT)`.
//! Game code never sees a keycode, and — more load-bearing — never sees the
//! difference between a live key and a replayed one, because both arrive as the
//! same [`InputFrame`].
//!
//! **Axis values are fixed-point on the way in.** Pointer motion is the one
//! continuous input, and a replay file full of `f32` deltas would make the
//! bit-exactness of the sim depend on the bit-exactness of a mouse driver. So
//! motion is quantized to [`AXIS_SCALE`]ths of a unit *before* the frame is
//! recorded, and [`Input::axis`] divides by a power of two — exact on every
//! target, and the same value on replay as on the day it was recorded.

use crate::key::{Key, MouseAxis, MouseButton, Wheel};
use crate::map::{ActionId, ActionMap, AxisId, ContextId, MAX_AXES, Source};

/// The recorded frame and its fixed-point unit live in `gg-abi`: the frame is
/// what [`TickCtx`](gg_abi::TickCtx) carries into a game dylib by value, so its
/// layout is a cross-artifact contract (§4.2.2). Everything that *decides* the
/// bits stays here.
pub use gg_abi::{AXIS_SCALE, InputFrame};

/// Mouse buttons tracked as held state. Bit `n` is button number `n + 1`.
///
/// The number itself is [`MouseButton::TRACKED`], where a *name* can be refused
/// against it (§6 M81) — the guard below still stands, because
/// [`MouseButton::Extra`] arrives from a platform's own numbering as well as
/// from a config line.
const TRACKED_BUTTONS: u8 = MouseButton::TRACKED;

/// Action state for the current tick, plus the previous one for edges.
#[derive(Debug)]
pub struct Input {
    map: ActionMap,
    stack: Vec<ContextId>,
    keys_held: [u64; Input::KEY_WORDS],
    buttons_held: u16,
    /// Raw device motion accumulated since the last tick, still in device
    /// units — the only float in the live path, and it never reaches the sim
    /// un-quantized.
    motion: [f32; 2],
    /// Cursor motion accumulated since the last tick, already in
    /// [`AXIS_SCALE`]ths. Integers all the way because a host derives these by
    /// *differencing positions* it has already quantized (§6 M15.1): passing
    /// them through a float would put a rounding step between the delta a host
    /// computed and the delta the sim sees, and the whole point of the source
    /// is that the two agree exactly.
    cursor: [i32; 2],
    /// Wheel notches accumulated since the last tick, signed, positive up. An
    /// impulse rather than a held state: [`Input::tick`] spends it and clears
    /// it, so a notch is down for exactly the tick it arrived in ([`Wheel`]).
    wheel: f32,
    /// Printable characters accumulated since the last tick (§6 M16), for a
    /// host that has somewhere to put text. An impulse like [`Input::wheel`],
    /// and for the same reason: typing belongs to the tick it happened on.
    ///
    /// Beside the action map rather than in it. A character is not a verb — it
    /// has no id, no binding and no meaning to a game — so it reaches
    /// [`InputFrame`] never and `gg_input::replay`'s text channel only.
    typed: String,
    /// Wall time the accumulators above span, in *tick periods* — the
    /// denominator [`Input::tick`] spends against, set by
    /// [`Input::frame_covered`] and decremented by one as each tick takes its
    /// period. One is the resting value, and it means "this tick takes
    /// everything".
    covered: f32,
    /// The **motion-only** part of what the last tick folded into each axis, in
    /// [`AXIS_SCALE`]ths — [`Input::spent`]. Kept separately because
    /// [`InputFrame::axes`] has the digital deflection added on top and the two
    /// cannot be told apart afterwards, and a late latch has to treat a travel
    /// and a level differently (§6 M56).
    spent: [i32; MAX_AXES],
    current: InputFrame,
    previous: InputFrame,
    /// Derived from `map` and `stack`, rebuilt by `reread` whenever either
    /// moves — see [`Input::spellings`].
    spellings: Vec<gg_abi::Binding>,
    keep: crate::map::Keep,
}

impl Input {
    const KEY_WORDS: usize = Key::COUNT.div_ceil(64);

    /// An input state over `map`, with no context active — until something is
    /// pushed, nothing is bound (§4.7: layering is explicit).
    pub fn new(map: ActionMap) -> Self {
        let mut input = Input {
            map,
            stack: Vec::new(),
            keys_held: [0; Self::KEY_WORDS],
            buttons_held: 0,
            motion: [0.0; 2],
            cursor: [0; 2],
            wheel: 0.0,
            typed: String::new(),
            covered: 1.0,
            spent: [0; MAX_AXES],
            current: InputFrame::default(),
            previous: InputFrame::default(),
            spellings: Vec::new(),
            keep: crate::map::Keep::default(),
        };
        input.reread();
        input
    }

    /// The map this state polls through.
    pub fn map(&self) -> &ActionMap {
        &self.map
    }

    /// Apply the player's own bindings over the map, reporting what it could not
    /// use (§6 M45) — [`ActionMap::overlay`], plus the reread the caches need.
    ///
    /// Here rather than in the host because the caches below are this type's:
    /// a shell that edited the map through a `&mut` would be a shell that has to
    /// remember to rebuild them, which is one more thing to forget at a reload.
    pub fn rebind(&mut self, text: &str) -> Vec<String> {
        let stale = self.map.overlay(text);
        self.reread();
        stale
    }

    /// What survives a modal screen under the layers active right now.
    pub fn keep(&self) -> crate::map::Keep {
        self.keep
    }

    /// What the map calls each verb, as the boundary carries it (§6 M45). Built
    /// per map rather than per tick: 64 verbs of it would be four kilobytes
    /// copied every tick to say something that did not change.
    pub fn spellings(&self) -> &[gg_abi::Binding] {
        &self.spellings
    }

    /// Rebuild what is derived from the map and the stack.
    fn reread(&mut self) {
        self.spellings = self.map.spellings();
        self.keep = self.map.keep(&self.stack);
    }

    /// Activate a layer. The most recently pushed context sees a source first.
    pub fn push_context(&mut self, context: ContextId) {
        self.stack.push(context);
        self.reread();
    }

    /// Activate the layer `name`, reporting whether the map declared one.
    ///
    /// The by-name form exists so a host does not have to reach into the map for
    /// an id and hand it straight back; whether a missing context is worth a
    /// complaint depends on why the caller asked, so this answers rather than
    /// warns.
    pub fn push_named(&mut self, name: &str) -> bool {
        match self.map.context(name) {
            Some(context) => {
                self.push_context(context);
                true
            }
            None => false,
        }
    }

    /// Whether the active layers bind `key` — see [`ActionMap::claims`]. What a
    /// shell asks before treating a key as its own.
    pub fn claims_key(&self, key: Key) -> bool {
        self.map.claims(&self.stack, Source::Key(key))
    }

    /// Whether the active layers bind raw device motion to any axis — see
    /// [`ActionMap::claims_motion`]. True for a game that does mouse-look.
    pub fn looks(&self) -> bool {
        [MouseAxis::X, MouseAxis::Y]
            .into_iter()
            .any(|m| self.map.claims_motion(&self.stack, m))
    }

    /// Deactivate the topmost layer.
    pub fn pop_context(&mut self) -> Option<ContextId> {
        let popped = self.stack.pop();
        self.reread();
        popped
    }

    /// Record a key edge. Auto-repeat must already be filtered (gg-platform
    /// does): a key held across ticks is a *state*, and a repeat that re-set an
    /// already-set bit would be harmless, but one that re-fired `just_pressed`
    /// would not.
    pub fn key(&mut self, key: Key, pressed: bool) {
        let (word, bit) = (key.index() / 64, key.index() % 64);
        if pressed {
            self.keys_held[word] |= 1 << bit;
        } else {
            self.keys_held[word] &= !(1 << bit);
        }
    }

    /// Whether a key is down right now, by physical position.
    ///
    /// Beside the action map and not in it, like [`Input::type_char`]'s text and
    /// for the same reason: this answers about a *key*, and a key has no verb id
    /// and reaches no [`InputFrame`], so nothing read through here is in a
    /// replay. Its one caller is the host asking whether a modifier is down, a
    /// chord being the one thing a map cannot spell (§6 M46).
    #[must_use]
    pub fn held(&self, key: Key) -> bool {
        let (word, bit) = (key.index() / 64, key.index() % 64);
        self.keys_held[word] & (1 << bit) != 0
    }

    /// Record a mouse-button edge.
    pub fn mouse_button(&mut self, button: MouseButton, pressed: bool) {
        let number = button.number();
        if number == 0 || number > TRACKED_BUTTONS {
            return;
        }
        let mask = 1u16 << (number - 1);
        if pressed {
            self.buttons_held |= mask;
        } else {
            self.buttons_held &= !mask;
        }
    }

    /// Accumulate raw device motion for this tick —
    /// [`MouseAxis::X`]/[`MouseAxis::Y`].
    pub fn motion(&mut self, dx: f32, dy: f32) {
        // A non-finite delta is a driver bug, not an input: dropping it keeps
        // one bad event from poisoning every later tick through the accumulator.
        if dx.is_finite() && dy.is_finite() {
            self.motion[0] += dx;
            self.motion[1] += dy;
        }
    }

    /// Accumulate cursor motion for this tick —
    /// [`MouseAxis::PointerX`]/[`MouseAxis::PointerY`] — in [`AXIS_SCALE`]ths of
    /// a surface unit.
    ///
    /// Saturating, for the reason `gg_ui`'s pointer clamps one layer up: an
    /// absurd delta parks the cursor at an edge rather than wrapping it to the
    /// other one.
    pub fn cursor(&mut self, dx: i32, dy: i32) {
        self.cursor[0] = self.cursor[0].saturating_add(dx);
        self.cursor[1] = self.cursor[1].saturating_add(dy);
    }

    /// Accumulate wheel travel for this tick, in notches, positive away from the
    /// operator. A pixel-precise device divides by its own line height first —
    /// what reaches here is notches or a fraction of one, and a fraction that
    /// never reaches a whole notch never presses anything.
    pub fn wheel(&mut self, notches: f32) {
        if notches.is_finite() {
            self.wheel += notches;
        }
    }

    /// Accumulate a character this tick produced (§6 M16).
    ///
    /// Printable only: control codes are the *keys* Backspace and Enter, which
    /// are ordinary verbs in the action map, and a field that took both would
    /// act on one keystroke twice.
    pub fn type_char(&mut self, c: char) {
        if !c.is_control() {
            self.typed.push(c);
        }
    }

    /// Take this tick's characters, or drop them where `wanted` is `false`.
    ///
    /// Drains either way, and that is the interesting half: a `false` means
    /// nothing is focused, and keeping the buffer would deliver these
    /// keystrokes to whatever *is* focused three seconds from now. The caller
    /// answers `wanted` because focus is a UI's fact, not this crate's.
    #[must_use]
    pub fn take_typed(&mut self, wanted: bool) -> String {
        let typed = core::mem::take(&mut self.typed);
        match wanted {
            true => typed,
            false => String::new(),
        }
    }

    /// How much wall time the accumulated motion spans, in tick periods —
    /// `gg_core::Due::covered`, handed over before the frame's first tick.
    ///
    /// Motion arrives on a frame's clock and is spent on a tick's, and the two do
    /// not divide. Spending the accumulator whole gives a tick however much travel
    /// a *variable* number of frames delivered: four at 240 Hz and sometimes three
    /// or five, so the view turns at a rate that wobbles by a quarter while held
    /// keys deflect an axis identically throughout. Translation walks on evenly,
    /// rotation lurches — which reads as the *world* juddering rather than the
    /// mouse, and gets worse the further the panel is from the sim rate.
    ///
    /// So a tick takes one period's worth, `1/covered` of what is in hand, and
    /// leaves the rest. It is lossless and it does not lag: the leftover is the
    /// travel of the leftover time, and it is still there when that time's tick
    /// comes due. A frame owing two ticks is the same statement with `covered`
    /// near two — half each — which is why there is no separate tick count here.
    ///
    /// A host that never calls this leaves it at one and every tick takes the
    /// whole accumulator, which is what a one-tick-per-frame run and every
    /// locked-pace tier (§5.6) already had — `covered` is exactly the tick count
    /// there, alpha being zero throughout.
    pub fn frame_covered(&mut self, ticks: f32) {
        // Set, not added: the clock's own residue already carries the frames that
        // owed nothing, so accumulating here would count that time twice.
        self.covered = ticks;
    }

    /// Fold this tick's share of the accumulated input into a frame. Call exactly
    /// once per sim tick; the returned frame is what the recorder writes.
    ///
    /// Impulses are *not* shared — a wheel notch and a typed character belong to
    /// the tick they arrived on and cannot be a third of anything, so they land on
    /// the first tick of a frame and are spent there.
    pub fn tick(&mut self) -> InputFrame {
        // One tick period out of however many the accumulator spans. Never above
        // one — a tick may not spend travel that has not happened — and a
        // non-finite or non-positive `covered` takes everything, which is the
        // resting behaviour rather than a NaN in the sim.
        let take = match self.covered > 1.0 {
            true => 1.0 / self.covered,
            false => 1.0,
        };
        self.covered = (self.covered - 1.0).max(0.0);
        let mut buttons = 0u64;
        // Digital deflection is *tallied by direction*, not summed: two keys
        // bound the same way are not twice the stick, and a key bound each way
        // cancels. Summing signs gets both of those wrong.
        let mut positive = [false; MAX_AXES];
        let mut negative = [false; MAX_AXES];
        let mut motion = [0i32; MAX_AXES];

        for &key in Key::ALL {
            let (word, bit) = (key.index() / 64, key.index() % 64);
            if self.keys_held[word] & (1 << bit) == 0 {
                continue;
            }
            let source = Source::Key(key);
            for action in self.map.actions_for(&self.stack, source) {
                buttons |= 1 << action.index();
            }
            for (axis, sign) in self.map.axes_for(&self.stack, source) {
                deflect(&mut positive, &mut negative, axis.index(), sign);
            }
        }
        for number in 1..=TRACKED_BUTTONS {
            if self.buttons_held & (1u16 << (number - 1)) == 0 {
                continue;
            }
            let Some(button) = MouseButton::from_number(number) else {
                continue;
            };
            let source = Source::Button(button);
            for action in self.map.actions_for(&self.stack, source) {
                buttons |= 1 << action.index();
            }
            for (axis, sign) in self.map.axes_for(&self.stack, source) {
                deflect(&mut positive, &mut negative, axis.index(), sign);
            }
        }
        // The wheel, as the edge it is: whichever direction crossed a whole
        // notch this tick presses for this tick only. Both directions are asked
        // so a map may bind them to one axis with opposite signs, which is the
        // shape a "zoom" verb wants.
        for wheel in Wheel::ALL {
            let notched = match wheel {
                Wheel::Up => self.wheel >= 1.0,
                Wheel::Down => self.wheel <= -1.0,
            };
            if !notched {
                continue;
            }
            let source = Source::Wheel(wheel);
            for action in self.map.actions_for(&self.stack, source) {
                buttons |= 1 << action.index();
            }
            for (axis, sign) in self.map.axes_for(&self.stack, source) {
                deflect(&mut positive, &mut negative, axis.index(), sign);
            }
        }
        // This tick's period of the accumulated motion, subtracted rather than
        // cleared so the remainder is the next tick's. Truncation on the cursor
        // keeps that exact — an integer source stays integral either side — while
        // the raw accumulator is a float whose residue stays un-quantized, which
        // is the same rounding a single tick already had.
        let share = [self.motion[0] * take, self.motion[1] * take];
        // `f64` and an exact identity at `take == 1.0`: the cursor is an integer
        // the host already quantized (§6 M15.1), and a delta past 2^24 does not
        // survive an `f32` multiply — scaling it there would move a number that
        // was supposed to arrive unchanged.
        let steps = match take < 1.0 {
            true => [scaled(self.cursor[0], take), scaled(self.cursor[1], take)],
            false => self.cursor,
        };
        // Device motion quantizes here; cursor motion arrived quantized. Same
        // loop either way — what differs is which accumulator was in whose
        // units, and a source is only ever in one of them.
        let deltas = [quantize(share[0]), quantize(share[1]), steps[0], steps[1]];
        for (quantized, which) in deltas.into_iter().zip(MouseAxis::ALL) {
            for (axis, sign) in self.map.motion_axes(&self.stack, which) {
                motion[axis.index()] += sign * quantized;
            }
        }

        let mut frame = InputFrame {
            buttons,
            axes: [0; MAX_AXES],
        };
        for i in 0..MAX_AXES {
            // Motion rides on top of the digital deflection *unclamped*: a fast
            // flick is a large delta, and clipping it would silently cap turn
            // speed at one unit per tick.
            let digital = i32::from(positive[i]) - i32::from(negative[i]);
            frame.axes[i] = digital * AXIS_SCALE + motion[i];
        }
        // The motion half alone, kept for `Input::spent`: `frame.axes` has the
        // digital deflection added on top and the two cannot be separated back
        // out of it (§6 M56).
        self.spent = motion;
        self.motion = [self.motion[0] - share[0], self.motion[1] - share[1]];
        self.cursor = [self.cursor[0] - steps[0], self.cursor[1] - steps[1]];
        // Spent whole rather than decremented: a notch that pressed this tick
        // must not press again next tick, and travel that never reached one is
        // a hand resting on the wheel rather than a scroll being saved up.
        self.wheel = 0.0;
        self.set_frame(frame);
        frame
    }

    /// Take this tick's frame from a replay instead of from the hardware. Live
    /// key state keeps updating underneath — a human mashing keys during a
    /// replay changes nothing the sim can see, which is the property that makes
    /// a replay a replay.
    pub fn tick_from(&mut self, frame: InputFrame) {
        self.motion = [0.0; 2];
        self.cursor = [0; 2];
        // A recorded frame is one number an axis, with no motion to separate out
        // of it — so a replayed session latches nothing (§6 M56).
        self.spent = [0; MAX_AXES];
        // Spent whole rather than decremented: a notch that pressed this tick
        // must not press again next tick, and travel that never reached one is
        // a hand resting on the wheel rather than a scroll being saved up.
        self.wheel = 0.0;
        self.set_frame(frame);
    }

    /// This tick's frame.
    pub fn frame(&self) -> InputFrame {
        self.current
    }

    /// Is the action down this tick?
    pub fn pressed(&self, action: ActionId) -> bool {
        self.current.buttons & (1 << action.index()) != 0
    }

    /// Did it go down *this* tick?
    pub fn just_pressed(&self, action: ActionId) -> bool {
        let mask = 1 << action.index();
        self.current.buttons & mask != 0 && self.previous.buttons & mask == 0
    }

    /// Did it come up this tick?
    pub fn just_released(&self, action: ActionId) -> bool {
        let mask = 1 << action.index();
        self.current.buttons & mask == 0 && self.previous.buttons & mask != 0
    }

    /// The axis in units. Exact: [`AXIS_SCALE`] is a power of two.
    pub fn axis(&self, axis: AxisId) -> f32 {
        self.current.axes[axis.index()] as f32 / AXIS_SCALE as f32
    }

    /// Travel this axis has taken that no tick has spent yet, in [`Input::axis`]'
    /// units (§6 M56) — what the picture is missing when a frame lands between
    /// two ticks.
    ///
    /// Continuous sources only, because they are the only ones with a remainder:
    /// [`Input::tick`] leaves a share of the motion accumulators behind and
    /// spends every button, key and notch whole. A held key therefore reports
    /// zero here and that is the right answer — what a late latch is missing is
    /// the hand's motion since the last tick, never a button's state.
    ///
    /// **Un-quantized** where [`Input::tick`] rounds to [`AXIS_SCALE`]ths. The
    /// difference is the point: this and the next tick's spend then differ by at
    /// most that rounding, so a view built on it steps onto the sim's own angle
    /// rather than drifting away from it by a residue that never resolves.
    #[must_use]
    pub fn pending(&self, axis: AxisId) -> f32 {
        // `tick`'s four sources in `tick`'s order, in the two units they arrive
        // in: raw device counts pass through, and the cursor arrived already
        // quantized into `AXIS_SCALE`ths of a surface unit.
        let deltas = [
            self.motion[0],
            self.motion[1],
            self.cursor[0] as f32 / AXIS_SCALE as f32,
            self.cursor[1] as f32 / AXIS_SCALE as f32,
        ];
        let mut total = 0.0;
        for (delta, which) in deltas.into_iter().zip(MouseAxis::ALL) {
            for (bound, sign) in self.map.motion_axes(&self.stack, which) {
                if bound.index() == axis.index() {
                    total += sign as f32 * delta;
                }
            }
        }
        total
    }

    /// What the last tick spent of this axis' **continuous** sources, in
    /// [`Input::axis`]' units (§6 M56).
    ///
    /// [`Input::axis`] minus the digital deflection, and the split is the point:
    /// a late latch has to know how much of the tick's turn came from the hand,
    /// because that part must not be interpolated backwards while a key's
    /// deflection — which is a level, not a travel — must. A game binding both
    /// to one axis gets each half treated as what it is.
    ///
    /// Zero after [`Input::tick_from`]: a replayed frame arrives as one number
    /// with no motion to separate out of it, so a replay latches nothing and
    /// renders exactly the interpolated tick it did before M56.
    #[must_use]
    pub fn spent(&self, axis: AxisId) -> f32 {
        self.spent[axis.index()] as f32 / AXIS_SCALE as f32
    }

    /// Drop the travel in hand (§6 M56).
    ///
    /// For the two states where nothing will ever spend it: a suspended session
    /// (§6 M49) runs no tick at all, and a halted one returns before
    /// [`Input::tick`]. An accumulator nobody empties grows for as long as the
    /// operator is away, and both the latch that reads it and the first tick
    /// after the pause would then be handed a turn measured in minutes.
    ///
    /// Motion only, on [`Input::pending`]'s reasoning: a key still held across a
    /// pause is still held after it, and a notch that never reached a tick is a
    /// hand resting on the wheel.
    pub fn forget_motion(&mut self) {
        self.motion = [0.0; 2];
        self.cursor = [0; 2];
    }

    fn set_frame(&mut self, frame: InputFrame) {
        self.previous = self.current;
        self.current = frame;
    }
}

fn deflect(
    positive: &mut [bool; MAX_AXES],
    negative: &mut [bool; MAX_AXES],
    axis: usize,
    sign: i32,
) {
    if sign >= 0 {
        positive[axis] = true;
    } else {
        negative[axis] = true;
    }
}

/// A fraction of an already-quantized integer delta, toward zero. `f64` because
/// it holds every `i32` exactly, which `f32` stops doing at 2^24 — and the
/// residue this leaves behind is the next tick's, so a rounding that drifted
/// would drift into the sim.
fn scaled(v: i32, take: f32) -> i32 {
    (f64::from(v) * f64::from(take)) as i32
}

/// Device units → [`AXIS_SCALE`]ths, saturating. `as` on a float saturates in
/// Rust, so an absurd delta clamps instead of wrapping into a turn the other
/// way.
fn quantize(v: f32) -> i32 {
    if v.is_finite() {
        (v * AXIS_SCALE as f32).round() as i32
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const ACTIONS: &[&str] = &["look", "spawn"];
    const AXES: &[&str] = &["move_right", "look_x", "ui_x"];
    const LOOK: ActionId = ActionId::new(0);
    const SPAWN: ActionId = ActionId::new(1);
    const MOVE_RIGHT: AxisId = AxisId::new(0);
    const LOOK_X: AxisId = AxisId::new(1);
    const UI_X: AxisId = AxisId::new(2);

    const MAP: &str = "
        [game.actions]
        look = [\"Tab\"]
        spawn = [\"F\", \"Mouse1\", \"WheelUp\"]

        [game.axes]
        move_right = [\"+D\", \"+Right\", \"-A\"]
        look_x = [\"MouseX\"]
        ui_x = [\"PointerX\"]
    ";

    fn input() -> Input {
        let map = ActionMap::parse(MAP, ACTIONS, AXES).unwrap();
        let mut input = Input::new(map);
        let game = input.map().context("game").unwrap();
        input.push_context(game);
        input
    }

    #[test]
    fn a_held_key_is_a_state_and_its_edges_are_one_tick_each() {
        let mut i = input();
        i.key(Key::Tab, true);
        i.tick();
        assert!(i.pressed(LOOK) && i.just_pressed(LOOK));
        i.tick();
        assert!(i.pressed(LOOK) && !i.just_pressed(LOOK), "no repeat edge");
        i.key(Key::Tab, false);
        i.tick();
        assert!(!i.pressed(LOOK) && i.just_released(LOOK));
    }

    #[test]
    fn two_keys_bound_the_same_way_are_not_twice_the_stick() {
        let mut i = input();
        i.key(Key::D, true);
        i.key(Key::Right, true);
        i.tick();
        assert_eq!(i.axis(MOVE_RIGHT), 1.0);
        i.key(Key::A, true);
        i.tick();
        assert_eq!(i.axis(MOVE_RIGHT), 0.0, "opposed keys cancel");
    }

    /// Why the shell may claim a chord's press and let its release fall through
    /// to `feed` (§6 M81's open row, closed here rather than in `play.rs`).
    /// `Event::Key` is level, not edge: `key` sets or clears a bit and `tick`
    /// samples it, so a release whose press nobody recorded clears a bit already
    /// zero and the derived edges compare two frames that agree. The unpaired
    /// release is therefore not observable — not in `pressed`, not in
    /// `just_released`, and not in the `InputFrame` a replay is made of.
    ///
    /// Here rather than beside the arm that produces it: this is the *reason*
    /// the arm is harmless, and it is a property of this crate's input model. An
    /// edge queue would make the shell's asymmetry a defect at the same moment
    /// it made this red, which is the coupling worth having.
    #[test]
    fn a_release_whose_press_was_claimed_is_not_an_edge() {
        let (mut claimed, mut quiet) = (input(), input());
        // The chord: the shell swallows Tab's press, so only the release feeds.
        claimed.key(Key::Tab, false);
        assert_eq!(claimed.tick(), quiet.tick(), "same frame as none");
        assert!(!claimed.pressed(LOOK) && !claimed.just_released(LOOK));
    }

    #[test]
    fn a_mouse_button_and_a_key_reach_the_same_action() {
        let mut i = input();
        i.mouse_button(MouseButton::Left, true);
        i.tick();
        assert!(i.pressed(SPAWN));
        i.mouse_button(MouseButton::Left, false);
        i.key(Key::F, true);
        i.tick();
        assert!(i.pressed(SPAWN), "still down, now by key");
    }

    /// The claim §6 M56's whole containment story rests on: under a locked pace
    /// — every replay, golden run and headless tier (§5.6) — a frame's ticks
    /// spend the accumulator down to nothing, so there is no travel left for a
    /// latch to show and the picture is the interpolated tick whatever the knob
    /// says. By construction, not by a flag.
    ///
    /// Both shapes a locked pace takes: one tick a frame, and `Locked(n)`, where
    /// `covered == count` and the shares are thirds that must still sum whole.
    #[test]
    fn a_locked_frame_leaves_no_travel_for_a_latch_to_show() {
        let mut i = input();
        i.motion(0.75, -0.25);
        i.frame_covered(1.0);
        i.tick();
        assert_eq!(i.pending(LOOK_X), 0.0);

        let mut i = input();
        i.motion(1.0, 0.0);
        // `Locked(3)`: the clock reports its tick count exactly, alpha being
        // zero throughout.
        i.frame_covered(3.0);
        for _ in 0..3 {
            i.tick();
        }
        assert_eq!(i.pending(LOOK_X), 0.0, "three thirds is all of it");
    }

    /// The other half of the same story: a replayed frame arrives as one number
    /// an axis with no motion inside it to separate out, so a replay latches
    /// nothing and renders what it rendered before M56.
    #[test]
    fn a_replayed_tick_offers_a_latch_nothing() {
        let mut i = input();
        // A *live* tick first, so there is a spend on the books for the replayed
        // one to have to clear. Without this the assertion below passes against
        // a `tick_from` that clears nothing, which is what the first version of
        // this test did.
        i.motion(4.0, 0.0);
        i.tick();
        assert!(i.spent(LOOK_X) > 0.0, "the live tick attributed its travel");

        i.motion(2.0, 1.0);
        i.tick_from(InputFrame {
            buttons: 0,
            axes: [512; MAX_AXES],
        });
        assert_eq!(i.axis(LOOK_X), 0.5, "the recorded value, undisturbed");
        assert_eq!(i.pending(LOOK_X), 0.0, "the live accumulator was dropped");
        assert_eq!(i.spent(LOOK_X), 0.0, "and none of it is attributable");
    }

    /// Pending and spent are the two halves of one travel, and the split is what
    /// makes the latch exact rather than approximate: what the tick took plus
    /// what it left is what the hand delivered.
    #[test]
    fn what_a_tick_spent_and_what_it_left_are_the_travel_that_arrived() {
        let mut i = input();
        i.motion(1.0, 0.0);
        i.frame_covered(4.0);
        i.tick();
        // A quarter taken, three quarters still in hand.
        assert_eq!(i.spent(LOOK_X), 0.25);
        assert_eq!(i.pending(LOOK_X), 0.75);
    }

    /// A key is a level, not a travel: it deflects its axis on the tick and has
    /// nothing outstanding, so neither number a latch reads may include it. A
    /// game binding a key and the mouse to one axis is the case that tells them
    /// apart.
    #[test]
    fn a_held_key_deflects_an_axis_without_offering_a_latch_anything() {
        let mut i = input();
        i.key(Key::D, true);
        i.motion(0.5, 0.0);
        i.frame_covered(2.0);
        i.tick();
        assert_eq!(i.axis(MOVE_RIGHT), 1.0, "the key, whole");
        assert_eq!(i.pending(MOVE_RIGHT), 0.0, "and nothing owed on it");
        assert_eq!(i.spent(LOOK_X), 0.25, "the mouse's half is still separable");
    }

    #[test]
    fn forgetting_motion_empties_what_a_latch_would_read() {
        let mut i = input();
        i.motion(3.0, 3.0);
        i.cursor(64, 64);
        assert!(i.pending(LOOK_X) > 0.0);
        i.forget_motion();
        assert_eq!(i.pending(LOOK_X), 0.0);
        // And the tick that follows is not handed the dropped travel either,
        // which is the point: a five-minute alt-tab must not resume with a turn
        // measured in minutes.
        i.tick();
        assert_eq!(i.axis(LOOK_X), 0.0);
    }

    #[test]
    fn motion_accumulates_within_a_tick_and_resets_across_one() {
        let mut i = input();
        i.motion(0.5, 0.0);
        i.motion(0.25, 0.0);
        i.tick();
        assert_eq!(i.axis(LOOK_X), 0.75);
        i.tick();
        assert_eq!(i.axis(LOOK_X), 0.0);
    }

    /// A frame owing two ticks turns the view half as far on each rather than
    /// wholly on the first: held keys deflect an axis identically on both ticks,
    /// so motion that lands all on one is rotation lurching against translation
    /// that did not — the *world* juddering, which is how it reads from a chair.
    #[test]
    fn a_frames_motion_is_spent_over_the_time_that_frame_covered() {
        let mut i = input();
        i.motion(1.0, 0.0);
        i.frame_covered(2.0);
        i.tick();
        assert_eq!(i.axis(LOOK_X), 0.5, "half on the first tick");
        i.tick();
        assert_eq!(i.axis(LOOK_X), 0.5, "and the remainder on the last");
        i.tick();
        assert_eq!(i.axis(LOOK_X), 0.0, "with nothing left over");

        // The control: the same travel over one period is undivided, which is
        // what every locked-pace tier runs and what this must not have moved.
        i.motion(1.0, 0.0);
        i.frame_covered(1.0);
        i.tick();
        assert_eq!(i.axis(LOOK_X), 1.0);
    }

    /// The 240 Hz case, which the tick *count* cannot see: every frame owes zero
    /// ticks or one, so a count-based divisor is always one, and yet the tick that
    /// comes due has whatever four frames delivered — or five, when the two clocks
    /// slip a beat. Spending that whole is a fifth of the turn rate appearing out
    /// of nothing, for one tick, while held keys deflect identically throughout.
    ///
    /// A quarter unit a frame, and a tick due after five of them rather than four.
    #[test]
    fn a_tick_spends_one_period_however_many_frames_delivered_it() {
        let mut i = input();
        for frame in 1u8..=5 {
            i.motion(0.25, 0.0);
            // What the clock reports: five quarter-periods in hand, and the tick
            // comes due on the fifth.
            i.frame_covered(0.25 * f32::from(frame));
        }
        i.tick();
        assert_eq!(i.axis(LOOK_X), 0.25 * 4.0, "one period's travel, not five");

        // And the fifth quarter is not lost — it is the next tick's, along with
        // whatever arrives before it. Three more frames complete that period.
        for frame in 1u8..=3 {
            i.motion(0.25, 0.0);
            i.frame_covered(0.25 * f32::from(frame + 1));
        }
        i.tick();
        assert_eq!(i.axis(LOOK_X), 0.25 * 4.0, "the leftover quarter arrived");
    }

    /// Lossless, which is the property that makes spending a fraction safe to do
    /// at all: the tick that exhausts the covered time takes the remainder, so a
    /// division that does not come out even neither drops travel nor invents it.
    /// Three periods over a delta that is not a multiple of three, and one that is
    /// not a multiple of `AXIS_SCALE`.
    #[test]
    fn spending_motion_by_period_neither_drops_it_nor_invents_it() {
        let mut i = input();
        i.cursor(100, 0);
        i.frame_covered(3.0);
        let mut total = 0;
        for _ in 0..3 {
            i.tick();
            total += i.frame().axes[UI_X.index()];
        }
        assert_eq!(total, 100, "every quantum arrived, once");
        assert_eq!(
            i.frame().axes[UI_X.index()],
            34,
            "the last takes the residue"
        );
    }

    /// A notch is not a fraction of anything. Impulses land on the first tick of
    /// the frame they arrived in and are spent there, however much time it covered.
    #[test]
    fn a_wheel_notch_is_not_spent_by_period() {
        let mut i = input();
        i.wheel(1.0);
        i.frame_covered(3.0);
        i.tick();
        assert!(i.pressed(SPAWN), "the whole notch, on the first tick");
        i.tick();
        assert!(!i.pressed(SPAWN), "and not again");
    }

    /// The two pointer sources are two sources (§6 M15.1). One mouse moving
    /// feeds whichever of them the host chose to feed, and a verb bound to the
    /// other one does not move — which is what stops a cursor crossing a
    /// viewport from also turning the camera.
    #[test]
    fn device_motion_and_cursor_motion_reach_different_verbs() {
        let mut i = input();
        i.motion(2.0, 0.0);
        i.tick();
        assert_eq!(i.axis(LOOK_X), 2.0);
        assert_eq!(i.axis(UI_X), 0.0, "a look delta is not cursor motion");

        i.cursor(AXIS_SCALE * 3, 0);
        i.tick();
        assert_eq!(i.axis(UI_X), 3.0);
        assert_eq!(i.axis(LOOK_X), 0.0, "and the reverse");
    }

    /// Cursor deltas never touch a float: a host differences two positions it
    /// already quantized, and what the sim sees is that integer and not a
    /// re-rounding of it. Checked at a value `f32` would not have kept.
    #[test]
    fn a_cursor_delta_arrives_as_the_integer_the_host_computed() {
        let mut i = input();
        let odd = 16_777_217; // 2^24 + 1 — the first integer an f32 cannot hold
        i.cursor(odd, 0);
        i.tick();
        assert_eq!(i.frame().axes[UI_X.index()], odd);
        i.tick();
        assert_eq!(i.frame().axes[UI_X.index()], 0, "and does not survive");
    }

    #[test]
    fn a_replayed_frame_outranks_whatever_the_hardware_is_doing() {
        let mut i = input();
        i.key(Key::Tab, true);
        i.motion(10.0, 0.0);
        let mut axes = [0; MAX_AXES];
        axes[LOOK_X.index()] = 512;
        i.tick_from(InputFrame {
            buttons: 1 << SPAWN.index(),
            axes,
        });
        assert!(i.pressed(SPAWN) && !i.pressed(LOOK));
        assert_eq!(i.axis(LOOK_X), 0.5);
        // ...and the live motion did not survive to leak into the next tick.
        i.tick();
        assert_eq!(i.axis(LOOK_X), 0.0);
    }

    #[test]
    fn quantization_is_exact_both_ways_and_survives_garbage() {
        assert_eq!(quantize(1.0), AXIS_SCALE);
        assert_eq!(quantize(-0.5), -512);
        assert_eq!(quantize(f32::NAN), 0);
        assert_eq!(quantize(f32::INFINITY), 0);
        let mut i = input();
        let mut axes = [0; MAX_AXES];
        axes[LOOK_X.index()] = AXIS_SCALE * 3;
        i.tick_from(InputFrame { buttons: 0, axes });
        assert_eq!(i.axis(LOOK_X), 3.0, "a flick is not clipped");
    }
}
