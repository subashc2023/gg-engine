//! The versioned host function table (§4.2.2) — everything a dylib can ask the
//! engine to do, and the opaque handle it does it through.
//!
//! The dylib never sees `World`'s layout. Structural operations cross here one
//! call at a time; the *hot* path does not cross at all, because
//! [`HostApiV1::query`] hands over whole columns and the dylib iterates them
//! natively (see [`crate::query`]).

use crate::entity::Entity;
use crate::query::{ArchetypeMatch, QueryDesc};

/// The host's `World`, opaque.
///
/// Zero-sized with a `PhantomPinned` marker: the C idiom for a type that is only
/// ever pointed at, never constructed or dereferenced on this side.
#[repr(C)]
pub struct WorldHandle {
    _data: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Why a host call did not do what was asked.
///
/// A real `repr(u32)` enum, unlike [`SystemStatus`](crate::SystemStatus): the
/// *host* produces these, and the host is the trusted, statically-linked side,
/// so an out-of-range discriminant cannot exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum AbiStatus {
    /// The call did what was asked.
    Ok = 0,
    /// No component with that stable id is registered. The host registers every
    /// component the dylib declares at load, so this means the dylib asked for
    /// one it did not declare.
    UnknownComponent = 1,
    /// The entity is not alive — despawned, or never allocated.
    DeadEntity = 2,
    /// The byte count handed over does not match the registered component's
    /// size. Catches a stale id pointed at a resized type.
    SizeMismatch = 3,
    /// The query names one component mutably twice, or both mutably and
    /// immutably. Checked before any pointer is produced, so this is an error
    /// rather than aliasing UB.
    Alias = 4,
    /// The query names more than [`MAX_QUERY_COLUMNS`](crate::MAX_QUERY_COLUMNS)
    /// columns, or a null output buffer.
    BadRequest = 5,
}

/// Log levels for [`HostApiV1::log`], matching `tracing`'s.
pub mod log_level {
    /// Something failed.
    pub const ERROR: u32 = 0;
    /// Something is suspicious.
    pub const WARN: u32 = 1;
    /// Ordinary progress.
    pub const INFO: u32 = 2;
    /// Detail for a developer.
    pub const DEBUG: u32 = 3;
    /// Firehose.
    pub const TRACE: u32 = 4;
}

/// Host operations, version 1.
///
/// The host owns this table and never moves it; the pointer handed to
/// `gg_game_init` stays valid for as long as the dylib can be called, which is
/// the process's lifetime (dev leaks dylibs rather than unloading them).
///
/// Every function takes the `world` handle explicitly rather than reading an
/// ambient one: a system that could reach the world without being handed it
/// would be a system whose access the host cannot bound.
#[repr(C)]
pub struct HostApiV1 {
    /// [`HOST_API_VERSION`](crate::HOST_API_VERSION), repeated inside the table
    /// so a mis-cast table is caught by inspection rather than by a jump.
    pub version: u32,
    /// Must be zero.
    pub reserved: u32,

    /// Allocate an entity with no components.
    pub spawn: unsafe extern "C" fn(world: *mut WorldHandle) -> Entity,
    /// Destroy an entity and everything on it. `false` if it was already dead.
    pub despawn: unsafe extern "C" fn(world: *mut WorldHandle, entity: Entity) -> bool,
    /// Is this handle live? A stale generation reads as dead.
    pub is_alive: unsafe extern "C" fn(world: *const WorldHandle, entity: Entity) -> bool,

    /// Add or overwrite a component. `data` is `len` bytes of the component's
    /// `Pod` representation.
    pub insert: unsafe extern "C" fn(
        world: *mut WorldHandle,
        entity: Entity,
        component: u64,
        data: *const u8,
        len: u32,
    ) -> AbiStatus,
    /// Drop a component. `false` if the entity did not have it.
    pub remove:
        unsafe extern "C" fn(world: *mut WorldHandle, entity: Entity, component: u64) -> bool,
    /// Pointer to one entity's component, or null if absent or dead.
    ///
    /// Invalidated by the next structural change, exactly like the column
    /// pointers in an [`ArchetypeMatch`].
    pub get:
        unsafe extern "C" fn(world: *mut WorldHandle, entity: Entity, component: u64) -> *mut u8,

    /// Fill `out` with up to `cap` archetype matches, skipping the first `skip`.
    ///
    /// Paging rather than a cursor: a cursor would be host state whose lifetime
    /// spans dylib code, and this boundary keeps every such lifetime inside one
    /// call. Loop while `written == cap`, adding `written` to `skip`.
    pub query: unsafe extern "C" fn(
        world: *mut WorldHandle,
        desc: *const QueryDesc,
        skip: u32,
        out: *mut ArchetypeMatch,
        cap: u32,
        written: *mut u32,
    ) -> AbiStatus,

    /// Emit a log line through the host's `tracing` subscriber, so game output
    /// lands in the same stream — and the same Tracy capture — as engine output.
    /// `level` is one of [`log_level`]; `msg` is UTF-8, `len` bytes.
    pub log: unsafe extern "C" fn(level: u32, msg: *const u8, len: u32),
}

/// The dylib's copy of the host table.
///
/// A `static` in an rlib is per-linkage-unit, and this crate is linked
/// statically into both the host and every game dylib — so each artifact gets
/// its own cell and there is nothing shared to race on. That is the mechanism,
/// not a coincidence to rely on quietly.
///
/// It is not a violation of the §4.2.2 rule against retained state in game
/// crates: the rule bans state that outlives a tick *and survives a reload*, and
/// this is re-initialized by `gg_game_init` on every load, in the fresh dylib's
/// own cell. It is also an `AtomicPtr` rather than a `static mut`, so the grep
/// ban's underlying hazard is absent as well as its text.
static HOST: core::sync::atomic::AtomicPtr<HostApiV1> =
    core::sync::atomic::AtomicPtr::new(core::ptr::null_mut());

/// Stash the host table for [`get`].
///
/// # Safety
///
/// `api` must be non-null, point to a valid [`HostApiV1`], and remain valid for
/// as long as this dylib can be called into — which the host guarantees by
/// owning the table for the process's lifetime.
pub unsafe fn set(api: *const HostApiV1) {
    HOST.store(api.cast_mut(), core::sync::atomic::Ordering::Release);
}

/// The host table this dylib was initialized with.
///
/// # Panics
///
/// If `gg_game_init` has not run. That is a host sequencing bug, not a
/// recoverable condition: no game code can do anything useful without it.
#[must_use]
pub fn get() -> &'static HostApiV1 {
    let ptr = HOST.load(core::sync::atomic::Ordering::Acquire);
    assert!(
        !ptr.is_null(),
        "gg-abi: host API used before `gg_game_init` — the host must hand over \
         the table before calling any system"
    );
    // SAFETY: non-null was just checked, and `set`'s contract is that the
    // pointer stays valid for as long as this dylib can be called into. The
    // host owns the table for the whole process.
    unsafe { &*ptr }
}
