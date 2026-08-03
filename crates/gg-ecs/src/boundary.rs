//! The §4.2.2 game-code boundary — both sides of it.
//!
//! `gg-abi` defines the seam's *layout*; this module is the code on either end
//! of it. [`host_api`] implements the versioned function table over a real
//! [`World`](crate::World); [`GameWorld`] and [`gg_game!`](crate::gg_game) are
//! what a game crate writes against, so gameplay code names components and never
//! sees a raw pointer.
//!
//! # Why it lives in `gg-ecs`
//!
//! Not preference — arithmetic. A game crate may link `gg-abi`, `gg-ecs` and
//! `gg-math` and nothing else (§3's deny pin, which is what makes the
//! fingerprint's scope and a dylib's link set the same list). `gg-abi` is
//! `no_std` and sits *below* [`Component`](crate::Component), so it cannot
//! assemble a table keyed on component identity; `gg-core` may not depend on
//! `gg-ecs` at all (§3's closed charter, and the reason `GameLib::load` takes
//! the table as an argument). That leaves one crate, and it is this one.
//!
//! It is not a widening of §7's non-goals: no scheduler, no system graph. The
//! table's order *is* the execution order, chosen by the game and read by the
//! loop (§4.1).
//!
//! # The two directions are trusted differently
//!
//! Host-side entry points are handed pointers by a dylib that may have been
//! built from anything, so every one of them validates before it dereferences,
//! and an out-of-range request is an [`AbiStatus`] rather than a panic. Guest
//! side, the host is the trusted, statically-linked engine — a column whose
//! stride disagrees with its component is a host bug, and the assertion that
//! catches it is a panic the shim converts into a status.

mod declare;
mod game;
mod host;
mod load;
mod macros;
mod present;
mod scene;
mod ui;

pub use declare::{Table, entry, init, layout_of, run, verbs};
pub use game::{Access, BoundaryData, BoundaryError, Columns, GameWorld};
pub use host::{SystemZone, host_api, set_logger, set_system_zone};
pub use load::{SystemPanic, Verbs, read_verbs};
pub use present::{Eye, Light, Model, Renderable, light};
pub use scene::Node;
pub use ui::{CANVAS, TEXT, Widget, state, widget, widget_id};

pub use crate::hash::system_id;

// Re-exported so `gg_game!`'s expansion names exactly one crate: a game crate is
// pinned to three engine dependencies (§3) and a macro that silently required a
// fourth path in scope would be a pin held by hope.
pub use gg_abi::host::log_level;
pub use gg_abi::{
    AXIS_SCALE, AbiInfo, AbiStatus, ActionId, AxisId, ComponentLayout, ComponentsTable, HostApiV1,
    InputFrame, MAX_AXES, SystemEntry, SystemStatus, SystemsTable, TickCtx, VerbName, VerbsTable,
    WorldHandle, asset_id,
};

/// What this build of the boundary says about itself — the value
/// `gg_game_abi()` returns.
///
/// Both halves come from `gg-abi` rather than from anything local: the whole
/// point of the check is that host and dylib compare *the same constant*
/// compiled twice.
#[must_use]
pub const fn abi_info() -> AbiInfo {
    AbiInfo {
        host_api_version: gg_abi::HOST_API_VERSION,
        fingerprint: gg_abi::BOUNDARY_FINGERPRINT,
    }
}

/// The world as the boundary sees it: an opaque handle, never a layout.
///
/// The cast is the entire mechanism — [`WorldHandle`] is a zero-sized `repr(C)`
/// marker, so this is a pointer with a different name and no promise attached to
/// it. The dylib cannot do anything with the result except hand it back.
#[must_use]
pub fn handle(world: &mut crate::World) -> *mut WorldHandle {
    core::ptr::from_mut(world).cast()
}
