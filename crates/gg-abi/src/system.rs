//! The systems table and the per-tick context (§4.2.2).
//!
//! System order in the table *is* execution order (§4.1), which is what makes
//! ordering part of game code and therefore part of what a reload can change.
//!
//! # Why a status code and not an unwind
//!
//! Since Rust 1.81 a panic unwinding out of an `extern "C"` fn is a defined
//! process abort, so a host-side `catch_unwind` around a system call could never
//! fire — the process would be dead before the catch. Panics are therefore
//! caught **inside the dylib** by a generated shim and cross as
//! [`SystemStatus`], which makes the boundary unwind-free by construction and
//! sidesteps cross-dylib unwinding entirely.

use crate::host::WorldHandle;
use crate::input::InputFrame;

/// What a system is handed each tick.
///
/// Both input frames travel **by value**, sized by `MAX_AXES` rather than a
/// byte count worth pinning here — cheaper than a pointer whose lifetime the
/// dylib would have to be trusted with — and carrying the previous frame is
/// what lets the generated wrappers answer `just_pressed` without the game
/// retaining state across ticks, which the §4.2.2 rules forbid anyway.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TickCtx {
    /// Sim time (§2): a `u64` tick count, never a float.
    pub tick: u64,
    /// Fixed timestep as a rate. The wrapper derives `dt` from it by dividing —
    /// a value computed identically everywhere, rather than a float *field*,
    /// which §2's Sim time row bans from sim-visible state.
    pub tick_hz: u32,
    /// Must be zero. Named rather than implicit: the boundary never leaves
    /// padding to chance, because a struct with an unwritten hole is a struct
    /// whose bytes two compilers may disagree about.
    pub reserved: u32,
    /// This tick's actions and axes.
    pub input: InputFrame,
    /// Last tick's, for edges.
    pub previous: InputFrame,
}

/// A system's outcome.
///
/// `code` is a plain integer with associated constants rather than a
/// `repr(u32)` enum: a dylib is the untrusted side (see the crate docs), and an
/// out-of-range enum discriminant is instant UB instead of a report.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct SystemStatus {
    /// [`SystemStatus::OK`] or [`SystemStatus::PANICKED`]. Anything else is
    /// treated as a panic with no message, which is the safe reading.
    pub code: u32,
    /// Bytes of `message`; zero when `message` is null.
    pub message_len: u32,
    /// UTF-8 payload and location, or null. Dylib-owned and valid for the
    /// process's lifetime: the shim leaks it, which costs one string per halted
    /// sim and is free of the lifetime question a borrowed buffer would raise —
    /// dev never unloads a dylib anyway (§4.2.2).
    pub message: *const u8,
}

impl SystemStatus {
    /// The system ran to completion.
    pub const OK: u32 = 0;
    /// The system panicked; the dylib-side shim caught it.
    pub const PANICKED: u32 = 1;

    /// A successful, message-free status.
    #[must_use]
    pub const fn ok() -> Self {
        Self {
            code: Self::OK,
            message_len: 0,
            message: core::ptr::null(),
        }
    }

    /// Did this system run? Anything that is not exactly [`SystemStatus::OK`]
    /// reads as a failure.
    #[must_use]
    pub const fn is_ok(&self) -> bool {
        self.code == Self::OK
    }
}

/// A system's `extern "C"` entry point.
///
/// # Safety
///
/// `world` must be the handle the host passed for this tick and `ctx` must
/// point to a live [`TickCtx`]; neither may be retained past the call.
pub type SystemFn =
    unsafe extern "C" fn(world: *mut WorldHandle, ctx: *const TickCtx) -> SystemStatus;

/// One named system.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct SystemEntry {
    /// UTF-8, not NUL-terminated. Names the system in a panic report, which is
    /// the whole reason it crosses at all.
    pub name: *const u8,
    /// The call.
    pub run: SystemFn,
    /// Stable id, hashed from the declared name. Survives reordering, so the
    /// host can tell "the table was reordered" from "a system was replaced".
    pub id: u64,
    /// Bytes of `name`.
    pub name_len: u32,
    /// Must be zero.
    pub reserved: u32,
}

/// The tick's systems, in execution order.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct SystemsTable {
    /// `len` entries, valid for as long as the dylib is loaded — which is
    /// forever, since dev leaks rather than unloads (§4.2.2).
    pub entries: *const SystemEntry,
    /// Entries in `entries`.
    pub len: u32,
    /// Must be zero.
    pub reserved: u32,
}
