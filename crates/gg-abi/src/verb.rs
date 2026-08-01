//! The verbs a game build declares (§4.7) — action and axis names, in id order.
//!
//! # Why names cross at all
//!
//! Everything else about input is already the host's: the action map binds names
//! to physical keys, the replay header records them, and both files are loaded
//! by the host. Only the *list* is the game's, because the list's order is the
//! id space every recorded frame is indexed by.
//!
//! The cheaper alternative — declare the list host-side in config and have game
//! code address it by index — drifts silently: adding an action to a reloaded
//! build would leave the two lists disagreeing with nothing in the process able
//! to notice. Crossing them makes the game the authority it already is for
//! components (§4.2), so a binding or a replay naming a verb *this build* does
//! not declare is refused by name at load.

/// One verb name.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct VerbName {
    /// UTF-8, not NUL-terminated.
    pub name: *const u8,
    /// Bytes of `name`.
    pub name_len: u32,
    /// Must be zero.
    pub reserved: u32,
}

/// A build's verbs, each list in **id order**.
///
/// Order is the contract: appending a verb is free, reordering one is a
/// replay-format change, and [`ActionId`](crate::ActionId) /
/// [`AxisId`](crate::AxisId) are indices into these two arrays.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct VerbsTable {
    /// `action_len` names, valid for the dylib's lifetime.
    pub actions: *const VerbName,
    /// `axis_len` names, same lifetime.
    pub axes: *const VerbName,
    /// Entries in `actions`. At most [`MAX_ACTIONS`](crate::MAX_ACTIONS) — a
    /// longer list cannot be recorded, so the host refuses it rather than
    /// truncating a replay's id space.
    pub action_len: u32,
    /// Entries in `axes`, at most [`MAX_AXES`](crate::MAX_AXES).
    pub axis_len: u32,
}
