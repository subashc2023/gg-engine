//! The sim-visible input surface (§4.7), at the boundary because game code
//! cannot link `gg-input`.
//!
//! Game crates are deny-pinned to `gg-abi`, `gg-ecs`, `gg-ecs-derive` and
//! `gg-math` (§3) — that pin is what makes the fingerprint's scope and a
//! dylib's possible link set the same four crates by construction. So the
//! *shape* of an input tick lives here and `gg-input` re-exports it, while
//! everything that decides what produced a given bit — the map, contexts, key
//! identity, the recorder — stays host-side where it belongs.
//!
//! **Axis values are fixed-point.** Pointer motion is the one continuous input,
//! and a replay full of `f32` deltas would make the bit-exactness of the sim
//! depend on the bit-exactness of a mouse driver. Motion is quantized to
//! [`AXIS_SCALE`]ths *before* it is recorded, and reading an axis divides by a
//! power of two — exact on every target, identical on replay.

/// Actions one map may declare. Sets the width of [`InputFrame::buttons`].
pub const MAX_ACTIONS: usize = 64;

/// Axes one map may declare.
///
/// Sixteen because eight ran out, and it is worth naming where: demo 05 declares
/// five, an editor opened over it appends a cursor pair and a raw-motion pair,
/// and nine does not fit in eight. Under the old cap the editor's look had to
/// difference the *cursor* — which the OS stops at the window edge, so a drag
/// stopped turning there (§6 M15.2's named residual). What it costs is 32 bytes
/// on [`InputFrame`], which is also 32 bytes on every replay change record — the
/// reason to double rather than to keep raising it one pair at a time.
pub const MAX_AXES: usize = 16;

/// Fixed-point unit for axis values: `1.0` is `AXIS_SCALE`. A power of two, so
/// the conversion back to `f32` is exact and identical on every target.
pub const AXIS_SCALE: i32 = 1024;

/// A digital action's index within the map.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ActionId(u8);

impl ActionId {
    /// # Panics
    ///
    /// If `index >= MAX_ACTIONS` — a const-evaluable panic, so the usual use
    /// (a `const` per action) fails to compile rather than at runtime.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        assert!(index < MAX_ACTIONS, "action index past MAX_ACTIONS");
        Self(index as u8)
    }

    /// Bit position within [`InputFrame::buttons`].
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// An analog axis's index within the map.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct AxisId(u8);

impl AxisId {
    /// # Panics
    ///
    /// If `index >= MAX_AXES`.
    #[must_use]
    pub const fn new(index: usize) -> Self {
        assert!(index < MAX_AXES, "axis index past MAX_AXES");
        Self(index as u8)
    }

    /// Position within [`InputFrame::axes`].
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// One tick of input, as recorded and as replayed.
///
/// This *is* the replay stream's element — the whole sim-visible input surface
/// in 72 bytes — and it crosses the reload boundary by value inside
/// [`TickCtx`](crate::TickCtx). A system cannot tell a live tick from a replayed
/// one, which is the property that makes a replay a replay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct InputFrame {
    /// Bit `i` is the action with index `i`, down this tick.
    pub buttons: u64,
    /// Axis values in [`AXIS_SCALE`]ths.
    pub axes: [i32; MAX_AXES],
}

impl InputFrame {
    /// Is the action down this tick?
    #[must_use]
    pub const fn pressed(&self, action: ActionId) -> bool {
        self.buttons & (1 << action.index()) != 0
    }

    /// The axis in units. Exact: [`AXIS_SCALE`] is a power of two.
    #[must_use]
    pub fn axis(&self, axis: AxisId) -> f32 {
        self.axes[axis.index()] as f32 / AXIS_SCALE as f32
    }
}
