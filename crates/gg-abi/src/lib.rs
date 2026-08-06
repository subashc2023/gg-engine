//! `gg-abi` (§4.2.2) — the game-code boundary, and nothing else.
//!
//! Game code presents itself to the host as a handful of `extern "C"` symbols
//! over an opaque, host-owned `World`. Everything crossing that seam is here,
//! `repr(C)`, because the seam's compatibility story is deliberately an **API
//! version number governing a layout the language defines** rather than a
//! fingerprint asserting that two `repr(Rust)` artifacts *should have been*
//! built from equivalent inputs. The fingerprint exists too ([`AbiInfo`]), as
//! defense-in-depth behind the version — never as the load-bearing check.
//!
//! # The symbols a game dylib exports
//!
//! | Symbol | Type | When the host calls it |
//! |---|---|---|
//! | [`SYM_GAME_ABI`] | [`GameAbiFn`] | first, before anything else is consulted |
//! | [`SYM_GAME_INIT`] | [`GameInitFn`] | once the version and fingerprint pass |
//! | [`SYM_GAME_COMPONENTS`] | [`GameComponentsFn`] | to register schemas and drive migration |
//! | [`SYM_GAME_VERBS`] | [`GameVerbsFn`] | to load the action map and check a replay |
//! | [`SYM_GAME_SYSTEMS`] | [`GameSystemsFn`] | to get the tick's execution order |
//!
//! Order is a contract: a dylib built against another host API must be refused
//! by number *before* its component or system tables are read, since those
//! tables' layout is exactly what the version governs.
//!
//! # Which side of the boundary produced a value
//!
//! This crate distrusts the two directions differently, and the asymmetry is
//! deliberate rather than sloppy. Values the **host** produces ([`AbiStatus`],
//! the `bool` returns on [`HostApiV1`]) are ordinary Rust enums and `bool`s: the
//! producer is the trusted, statically-linked engine, so an out-of-range
//! discriminant cannot exist. Values a **dylib** produces ([`SystemStatus::code`])
//! are plain integers with associated constants, because a mismatched or
//! miscompiled dylib is precisely the thing this boundary exists to survive, and
//! an invalid `repr(u32)` enum discriminant is instant UB rather than an error
//! anyone can report.
//!
//! # No allocation, no behaviour
//!
//! `#![no_std]`. Two exceptions, both of them the *same* argument the layouts
//! above make: [`host`] is a pointer stash whose per-dylib copy is a linkage
//! fact, and [`asset`] is a `const fn` hash both sides of the seam must agree
//! on to the bit. Neither allocates, and neither could grow a subsystem.

#![no_std]
#![warn(missing_docs)]

pub mod asset;
pub mod entity;
pub mod host;
pub mod input;
pub mod layout;
pub mod query;
pub mod system;
pub mod verb;

pub use asset::asset_id;
pub use entity::Entity;
pub use host::{AbiStatus, HostApiV1, WorldHandle};
pub use input::{AXIS_SCALE, ActionId, AxisId, InputFrame, MAX_ACTIONS, MAX_AXES};
pub use layout::{ComponentLayout, ComponentsTable, FieldLayout};
pub use query::{ArchetypeMatch, ColumnView, MAX_QUERY_COLUMNS, QueryDesc};
pub use system::{SystemEntry, SystemFn, SystemStatus, SystemsTable, TickCtx};
pub use verb::{VerbName, VerbsTable};

/// Which host API this build of the boundary is.
///
/// The load-bearing compatibility check (§4.2.2). Bump it for **any** change to
/// a type in this crate — adding a field, reordering one, widening an integer.
/// A dylib whose [`AbiInfo::host_api_version`] differs is refused by number with
/// a named error, before its tables are read.
// v4: `MAX_AXES` 8 → 16 — proof the rule covers a widened constant, not only
// a type. Earlier bumps were ordinary type/symbol changes; git carries them.
pub const HOST_API_VERSION: u32 = 4;

/// Bytes in a boundary fingerprint: a blake3 output, named here so the host and
/// the dylib agree on the width without `gg-abi` depending on a hash.
pub const FINGERPRINT_BYTES: usize = 32;

// `BOUNDARY_FINGERPRINT` and its hex form, computed by `build.rs` over the four
// boundary crates' source and the toolchain version. Generated rather than
// written because a hand-maintained fingerprint is a fingerprint that stops
// moving; see build.rs for the scope and why it is those four crates.
include!(concat!(env!("OUT_DIR"), "/fingerprint.rs"));

/// What a game dylib says about itself, before it is trusted with anything.
///
/// Returned by [`SYM_GAME_ABI`]. `repr(C)` and free of pointers on purpose: this
/// is the one value read from an artifact that has not yet been proven
/// compatible, so it must be interpretable under *any* version of this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct AbiInfo {
    /// The [`HOST_API_VERSION`] the dylib was compiled against.
    pub host_api_version: u32,
    /// Content hash of the boundary crates (`gg-abi`, `gg-ecs`, `gg-math`) plus
    /// the toolchain version — defense-in-depth for semantic drift the version
    /// number cannot see, such as a changed component invariant or a resized
    /// math type behind an unchanged API. Deliberately scoped to those crates: a
    /// comment edit in `gg-render` must not invalidate a running session for a
    /// layout that cannot have changed.
    pub fingerprint: [u8; FINGERPRINT_BYTES],
}

/// `gg_game_abi` — version and fingerprint. Resolved and called first.
pub const SYM_GAME_ABI: &[u8] = b"gg_game_abi\0";
/// `gg_game_init` — hands the dylib the host function table.
pub const SYM_GAME_INIT: &[u8] = b"gg_game_init\0";
/// `gg_game_components` — the component schemas this build declares.
pub const SYM_GAME_COMPONENTS: &[u8] = b"gg_game_components\0";
/// `gg_game_verbs` — the action and axis names this build declares (§4.7).
pub const SYM_GAME_VERBS: &[u8] = b"gg_game_verbs\0";
/// `gg_game_systems` — the tick's systems, in execution order.
pub const SYM_GAME_SYSTEMS: &[u8] = b"gg_game_systems\0";

/// Signature of [`SYM_GAME_ABI`].
pub type GameAbiFn = unsafe extern "C" fn() -> AbiInfo;
/// Signature of [`SYM_GAME_INIT`]. The pointer must outlive every later call
/// into the dylib; the host owns the table and never moves it.
pub type GameInitFn = unsafe extern "C" fn(*const HostApiV1);
/// Signature of [`SYM_GAME_COMPONENTS`].
pub type GameComponentsFn = unsafe extern "C" fn() -> ComponentsTable;
/// Signature of [`SYM_GAME_VERBS`].
pub type GameVerbsFn = unsafe extern "C" fn() -> VerbsTable;
/// Signature of [`SYM_GAME_SYSTEMS`].
pub type GameSystemsFn = unsafe extern "C" fn() -> SystemsTable;

#[cfg(test)]
mod tests;
