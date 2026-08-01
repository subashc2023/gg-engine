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

use crate::key::{Key, MouseAxis, MouseButton};
use crate::map::{ActionId, ActionMap, AxisId, ContextId, MAX_AXES, Source};

/// Fixed-point unit for axis values: `1.0` is `AXIS_SCALE`. A power of two, so
/// the conversion back to `f32` is exact and identical on every target.
pub const AXIS_SCALE: i32 = 1024;

/// Mouse buttons tracked as held state. Bit `n` is button number `n + 1`.
const TRACKED_BUTTONS: u8 = 16;

/// One tick of input, as recorded and as replayed. This *is* the replay
/// stream's element — the whole sim-visible input surface in 40 bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InputFrame {
    /// Bit `i` is the action with index `i`, down this tick.
    pub buttons: u64,
    /// Axis values in [`AXIS_SCALE`]ths.
    pub axes: [i32; MAX_AXES],
}

/// Action state for the current tick, plus the previous one for edges.
#[derive(Debug)]
pub struct Input {
    map: ActionMap,
    stack: Vec<ContextId>,
    keys_held: [u64; Input::KEY_WORDS],
    buttons_held: u16,
    /// Raw pointer motion accumulated since the last tick, still in device
    /// units — the only float in the live path, and it never reaches the sim
    /// un-quantized.
    motion: [f32; 2],
    current: InputFrame,
    previous: InputFrame,
}

impl Input {
    const KEY_WORDS: usize = Key::COUNT.div_ceil(64);

    /// An input state over `map`, with no context active — until something is
    /// pushed, nothing is bound (§4.7: layering is explicit).
    pub fn new(map: ActionMap) -> Self {
        Input {
            map,
            stack: Vec::new(),
            keys_held: [0; Self::KEY_WORDS],
            buttons_held: 0,
            motion: [0.0; 2],
            current: InputFrame::default(),
            previous: InputFrame::default(),
        }
    }

    /// The map this state polls through.
    pub fn map(&self) -> &ActionMap {
        &self.map
    }

    /// Activate a layer. The most recently pushed context sees a source first.
    pub fn push_context(&mut self, context: ContextId) {
        self.stack.push(context);
    }

    /// Deactivate the topmost layer.
    pub fn pop_context(&mut self) -> Option<ContextId> {
        self.stack.pop()
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

    /// Accumulate relative pointer motion for this tick.
    pub fn motion(&mut self, dx: f32, dy: f32) {
        // A non-finite delta is a driver bug, not an input: dropping it keeps
        // one bad event from poisoning every later tick through the accumulator.
        if dx.is_finite() && dy.is_finite() {
            self.motion[0] += dx;
            self.motion[1] += dy;
        }
    }

    /// Fold accumulated raw input into this tick's frame. Call exactly once per
    /// sim tick; the returned frame is what the recorder writes.
    pub fn tick(&mut self) -> InputFrame {
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
        for (raw, which) in self.motion.into_iter().zip([MouseAxis::X, MouseAxis::Y]) {
            let quantized = quantize(raw);
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
        self.motion = [0.0; 2];
        self.set_frame(frame);
        frame
    }

    /// Take this tick's frame from a replay instead of from the hardware. Live
    /// key state keeps updating underneath — a human mashing keys during a
    /// replay changes nothing the sim can see, which is the property that makes
    /// a replay a replay.
    pub fn tick_from(&mut self, frame: InputFrame) {
        self.motion = [0.0; 2];
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
    const AXES: &[&str] = &["move_right", "look_x"];
    const LOOK: ActionId = ActionId::new(0);
    const SPAWN: ActionId = ActionId::new(1);
    const MOVE_RIGHT: AxisId = AxisId::new(0);
    const LOOK_X: AxisId = AxisId::new(1);

    const MAP: &str = "
        [game.actions]
        look = [\"Tab\"]
        spawn = [\"F\", \"Mouse1\"]

        [game.axes]
        move_right = [\"+D\", \"+Right\", \"-A\"]
        look_x = [\"MouseX\"]
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

    #[test]
    fn a_replayed_frame_outranks_whatever_the_hardware_is_doing() {
        let mut i = input();
        i.key(Key::Tab, true);
        i.motion(10.0, 0.0);
        i.tick_from(InputFrame {
            buttons: 1 << SPAWN.index(),
            axes: [0, 512, 0, 0, 0, 0, 0, 0],
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
        i.tick_from(InputFrame {
            buttons: 0,
            axes: [0, AXIS_SCALE * 3, 0, 0, 0, 0, 0, 0],
        });
        assert_eq!(i.axis(LOOK_X), 3.0, "a flick is not clipped");
    }
}
