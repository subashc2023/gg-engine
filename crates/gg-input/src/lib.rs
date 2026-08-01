//! `gg-input` (§4.7) — action maps, the polling API, and input recording and
//! replay.
//!
//! Three responsibilities that belong together because they are three views of
//! one thing: game code says `input.pressed(SPAWN)`, the map decides what
//! produced that, and the recorder writes down the answer so the same tick can
//! be produced again on another machine, another architecture, or another build
//! tier (§5.6).
//!
//! # Where this crate sits
//!
//! **Below `gg-platform`, not above it.** The obvious direction — input as a
//! consumer of the windowing layer — would put winit in the dependency graph of
//! every replay runner, every determinism gate and every qemu leg, and those
//! are headless *by linkage* (§1.5, §5 gate 7). So [`Key`] is born here, with
//! no dependencies at all, and `gg-platform` translates winit into it.
//!
//! # The two paths through it
//!
//! ```text
//! live:    key/motion events → Input::tick()      → InputFrame → Recorder
//! replay:  Replay::frame(tick) → Input::tick_from(frame)
//! ```
//!
//! Both end at the same [`InputFrame`], and the sim cannot tell them apart —
//! which is the entire point, and the reason the poll API has no "am I being
//! replayed" question to ask.

#![warn(missing_docs)]

pub mod drive;
pub mod key;
pub mod map;
pub mod replay;
pub mod state;

pub use drive::Drive;
pub use key::{Key, MouseAxis, MouseButton};
pub use map::{
    ActionId, ActionMap, AxisId, AxisSource, ContextId, MAX_ACTIONS, MAX_AXES, MapError, Source,
};
pub use replay::{ENGINE_COMMIT, Recorder, Replay, ReplayError, ReplayMeta, Segment};
pub use state::{AXIS_SCALE, Input, InputFrame};
