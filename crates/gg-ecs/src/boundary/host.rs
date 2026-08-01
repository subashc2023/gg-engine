//! The host half: [`HostApiV1`] over a real [`World`] (§4.2.2).
//!
//! Every function here is reached from a dylib the host has version-checked but
//! not otherwise trusted, so each one validates its arguments before
//! dereferencing anything and reports refusals as [`AbiStatus`]. None of them
//! may panic: a panic here would unwind *out of* `extern "C"` back into the
//! dylib's frame, which is a defined process abort since Rust 1.81 — the exact
//! failure the dylib-side shims exist to avoid, arriving from the other side.

use gg_abi::{
    AbiStatus, ArchetypeMatch, ColumnView, Entity, HostApiV1, MAX_QUERY_COLUMNS, QueryDesc,
    WorldHandle,
};

use crate::hash::ComponentId;
use crate::view::QueryAccess;
use crate::world::World;

/// Where a game's `log` calls go.
///
/// A hook rather than a `tracing` call, because `gg-ecs` does not depend on
/// `tracing` and buying a dependency for one forwarding function would spend the
/// crate's budget (§3) on a line the shell can write itself. Unset until the
/// shell sets it: game output before the host is up has nowhere to go, and
/// silently dropping it is better than a second logging path.
static LOGGER: std::sync::OnceLock<fn(level: u32, message: &str)> = std::sync::OnceLock::new();

/// Route game `log` calls into the host's subscriber. Once per process.
///
/// # Errors
///
/// Returns the logger back if one is already installed — two loggers would mean
/// two answers to "where did that line go".
pub fn set_logger(logger: fn(level: u32, message: &str)) -> Result<(), fn(u32, &str)> {
    LOGGER.set(logger)
}

/// The host function table for this build.
///
/// A `&'static` because the dylib keeps the pointer for as long as it can be
/// called into, which is the process's lifetime (§4.2.2 leaks dylibs rather than
/// unloading them).
#[must_use]
pub fn host_api() -> &'static HostApiV1 {
    &TABLE
}

static TABLE: HostApiV1 = HostApiV1 {
    version: gg_abi::HOST_API_VERSION,
    reserved: 0,
    spawn,
    despawn,
    is_alive,
    insert,
    remove,
    get,
    query,
    log,
};

/// # Safety
///
/// `handle` must be a pointer produced by [`handle`](super::handle) from a
/// `&mut World` that is live and not otherwise borrowed for this call.
unsafe fn world_mut<'a>(handle: *mut WorldHandle) -> Option<&'a mut World> {
    if handle.is_null() {
        return None;
    }
    // SAFETY: the caller's contract — `handle` came from `&mut World` and the
    // host does not call into a dylib while holding another borrow of it.
    Some(unsafe { &mut *handle.cast::<World>() })
}

/// # Safety
///
/// As [`world_mut`], for a shared borrow.
unsafe fn world_ref<'a>(handle: *const WorldHandle) -> Option<&'a World> {
    if handle.is_null() {
        return None;
    }
    // SAFETY: as `world_mut`.
    Some(unsafe { &*handle.cast::<World>() })
}

unsafe extern "C" fn spawn(handle: *mut WorldHandle) -> Entity {
    // SAFETY: the boundary's contract for every entry point (§4.2.2).
    match unsafe { world_mut(handle) } {
        // A null world cannot spawn; `Entity::NONE` is already the value every
        // accessor treats as dead, so the dylib's next call reports it rather
        // than this one aborting.
        None => Entity::NONE,
        Some(world) => world.spawn(),
    }
}

unsafe extern "C" fn despawn(handle: *mut WorldHandle, entity: Entity) -> bool {
    // SAFETY: as `spawn`.
    unsafe { world_mut(handle) }.is_some_and(|world| world.despawn(entity))
}

unsafe extern "C" fn is_alive(handle: *const WorldHandle, entity: Entity) -> bool {
    // SAFETY: as `spawn`, shared.
    unsafe { world_ref(handle) }.is_some_and(|world| world.is_alive(entity))
}

unsafe extern "C" fn insert(
    handle: *mut WorldHandle,
    entity: Entity,
    component: u64,
    data: *const u8,
    len: u32,
) -> AbiStatus {
    // SAFETY: as `spawn`.
    let Some(world) = (unsafe { world_mut(handle) }) else {
        return AbiStatus::BadRequest;
    };
    if data.is_null() && len != 0 {
        return AbiStatus::BadRequest;
    }
    // SAFETY: the dylib's contract for `insert` — `data` is `len` readable bytes
    // of the component's `Pod` image. A zero length takes the empty slice rather
    // than a null pointer, which `from_raw_parts` forbids even for zero.
    let bytes = if len == 0 {
        &[][..]
    } else {
        unsafe { core::slice::from_raw_parts(data, len as usize) }
    };
    world.insert_raw(entity, ComponentId::from_raw(component), bytes)
}

unsafe extern "C" fn remove(handle: *mut WorldHandle, entity: Entity, component: u64) -> bool {
    // SAFETY: as `spawn`.
    unsafe { world_mut(handle) }
        .is_some_and(|world| world.remove_raw(entity, ComponentId::from_raw(component)))
}

unsafe extern "C" fn get(handle: *mut WorldHandle, entity: Entity, component: u64) -> *mut u8 {
    // SAFETY: as `spawn`.
    match unsafe { world_mut(handle) } {
        None => core::ptr::null_mut(),
        Some(world) => world.get_raw(entity, ComponentId::from_raw(component)),
    }
}

unsafe extern "C" fn query(
    handle: *mut WorldHandle,
    desc: *const QueryDesc,
    skip: u32,
    out: *mut ArchetypeMatch,
    cap: u32,
    written: *mut u32,
) -> AbiStatus {
    if desc.is_null() || out.is_null() || written.is_null() {
        return AbiStatus::BadRequest;
    }
    // SAFETY: as `spawn`.
    let Some(world) = (unsafe { world_mut(handle) }) else {
        return AbiStatus::BadRequest;
    };
    // SAFETY: the dylib's contract — `desc` points at one live `QueryDesc`, and
    // its two arrays hold `read_len`/`write_len` ids. Both are `Copy` plain data.
    let (read, write) = unsafe {
        let desc = *desc;
        match (
            ids(desc.read, desc.read_len),
            ids(desc.write, desc.write_len),
        ) {
            (Some(read), Some(write)) => (read, write),
            _ => return AbiStatus::BadRequest,
        }
    };
    if read.len() + write.len() > MAX_QUERY_COLUMNS {
        return AbiStatus::BadRequest;
    }

    // The access set sorts and dedups; the *output* must stay in the caller's
    // order, because that is the order the dylib's generated wrappers count
    // columns off in. `read_index`/`write_index` are the translation.
    let access = match QueryAccess::new(&read, &write) {
        Ok(access) => access,
        Err(_) => return AbiStatus::Alias,
    };

    // SAFETY: the dylib's contract — `out` is `cap` initialized `ArchetypeMatch`
    // values it does not touch until this call returns.
    let page = unsafe { core::slice::from_raw_parts_mut(out, cap as usize) };
    let mut produced = 0usize;
    let mut skipped = 0u32;
    world.views(&access, |view| {
        if produced >= page.len() {
            return;
        }
        if skipped < skip {
            skipped += 1;
            return;
        }
        let entities = view.entities();
        let (reads, writes) = (view.raw_reads(), view.raw_writes());
        let matched = &mut page[produced];
        matched.entities = entities.as_ptr();
        matched.len = entities.len();
        for (at, id) in read.iter().chain(write.iter()).enumerate() {
            // `read` first, so an id found among the reads is one; the write
            // lookup is only reached for the tail.
            matched.columns[at] = access
                .read_index(*id)
                .map(|i| reads[i])
                .or_else(|| access.write_index(*id).map(|i| writes[i]))
                .unwrap_or(EMPTY_COLUMN);
        }
        produced += 1;
    });

    // SAFETY: `written` was null-checked above and the dylib owns one `u32`
    // there for the duration of the call.
    unsafe { *written = produced as u32 };
    AbiStatus::Ok
}

const EMPTY_COLUMN: ColumnView = ColumnView {
    ptr: core::ptr::null_mut(),
    len: 0,
    stride: 0,
};

/// # Safety
///
/// `ptr` must be null or point at `len` readable `u64`s.
unsafe fn ids(ptr: *const u64, len: u32) -> Option<Vec<ComponentId>> {
    if len == 0 {
        return Some(Vec::new());
    }
    if ptr.is_null() {
        return None;
    }
    // SAFETY: the caller's contract, with the null and zero cases already split
    // off — `from_raw_parts` forbids null even at length zero.
    let raw = unsafe { core::slice::from_raw_parts(ptr, len as usize) };
    Some(
        raw.iter()
            .map(|&bits| ComponentId::from_raw(bits))
            .collect(),
    )
}

unsafe extern "C" fn log(level: u32, message: *const u8, len: u32) {
    let Some(logger) = LOGGER.get() else {
        return;
    };
    if message.is_null() || len == 0 {
        return;
    }
    // SAFETY: the dylib's contract — `message` is `len` readable bytes.
    let bytes = unsafe { core::slice::from_raw_parts(message, len as usize) };
    // Lossless or nothing: a dylib that sent non-UTF-8 has a bug worth hearing
    // about, and mangling it into replacement characters would hide it.
    if let Ok(text) = core::str::from_utf8(bytes) {
        logger(level, text);
    }
}
