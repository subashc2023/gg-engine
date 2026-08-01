//! What [`gg_game!`](crate::gg_game) expands into calls of (§4.2.2).
//!
//! The macro stays thin on purpose: every line of `unsafe` a game crate would
//! otherwise carry lives here instead, written once and reviewed once, so a
//! regenerated table cannot regenerate a soundness bug.
//!
//! # Panics are caught here, on this side of the seam
//!
//! Since Rust 1.81 a panic unwinding out of an `extern "C"` fn is a defined
//! process abort, so a host-side `catch_unwind` around a system call could never
//! fire — the process would be dead before the catch. [`run`] therefore catches
//! *inside the dylib* and returns a [`SystemStatus`], which makes the boundary
//! unwind-free by construction. Under `panic = "abort"` (dist) the same
//! `catch_unwind` compiles to a straight call, so this is one code path in every
//! tier rather than two.

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, Once, OnceLock, PoisonError};

use gg_abi::{
    ComponentLayout, ComponentsTable, FieldLayout, HostApiV1, SystemEntry, SystemFn, SystemStatus,
    SystemsTable, TickCtx, VerbName, VerbsTable, WorldHandle,
};

use super::game::GameWorld;
use crate::component::Component;

/// A table the generated `gg_game_*` symbols hand back by pointer.
///
/// The values are built on first call and never dropped: the boundary hands the
/// host raw pointers into them, and the host may read those for as long as the
/// dylib is loaded — which is the process's lifetime, since dev leaks dylibs
/// rather than unloading them (§4.2.2).
pub struct Table<T: 'static> {
    entries: OnceLock<Vec<T>>,
}

// SAFETY: `ComponentLayout` and `SystemEntry` hold raw pointers, which is why
// they are not `Sync` on their own. Every pointer they hold points at `'static`,
// immutable data — string literals in this artifact and slices leaked by
// `layout_of` — and the only mutation of the `Vec` is `OnceLock`'s own
// initialization, which is synchronized. Sharing the table across threads is
// therefore sharing immutable data.
unsafe impl<T> Sync for Table<T> {}

// SAFETY: as above — there is nothing thread-affine behind those pointers.
unsafe impl<T> Send for Table<T> {}

impl<T: 'static> Table<T> {
    /// An empty table, for a `static` in the generated code.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            entries: OnceLock::new(),
        }
    }
}

impl<T: 'static> Default for Table<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl Table<ComponentLayout> {
    /// Build (once) and describe this build's component schemas.
    ///
    /// Takes constructors rather than values so nothing is computed until the
    /// host asks: a schema hash is a blake3 pass per component, and a dylib that
    /// is about to be refused by version number should not have paid for one.
    pub fn components(&'static self, build: &[fn() -> ComponentLayout]) -> ComponentsTable {
        let entries = self
            .entries
            .get_or_init(|| build.iter().map(|make| make()).collect());
        ComponentsTable {
            entries: entries.as_ptr(),
            len: entries.len() as u32,
            reserved: 0,
        }
    }
}

impl Table<VerbName> {
    /// Build (once) one verb list, in the order it was declared.
    fn names(&'static self, names: &[&'static str]) -> (*const VerbName, u32) {
        let entries = self.entries.get_or_init(|| {
            names
                .iter()
                .map(|name| VerbName {
                    name: name.as_ptr(),
                    name_len: name.len() as u32,
                    reserved: 0,
                })
                .collect()
        });
        (entries.as_ptr(), entries.len() as u32)
    }
}

/// This build's verbs, from two lists whose order is their id space (§4.7).
///
/// Two tables rather than one because a `static` holds one `OnceLock`, and the
/// two lists are built independently.
#[must_use]
pub fn verbs(
    actions: &'static Table<VerbName>,
    axes: &'static Table<VerbName>,
    action_names: &[&'static str],
    axis_names: &[&'static str],
) -> VerbsTable {
    let (actions, action_len) = actions.names(action_names);
    let (axes, axis_len) = axes.names(axis_names);
    VerbsTable {
        actions,
        axes,
        action_len,
        axis_len,
    }
}

impl Table<SystemEntry> {
    /// Build (once) and describe this build's systems, in execution order.
    pub fn systems(&'static self, build: &[SystemEntry]) -> SystemsTable {
        let entries = self.entries.get_or_init(|| build.to_vec());
        SystemsTable {
            entries: entries.as_ptr(),
            len: entries.len() as u32,
            reserved: 0,
        }
    }
}

/// One component's schema, in the C form the host migrates against.
///
/// Leaks one field array per component per process. That is deliberate and
/// bounded: the boundary needs `'static` `repr(C)` descriptors, `FieldDesc` is
/// not one, and a dylib is never unloaded — so "leaked" and "lives as long as
/// the table" are the same statement.
#[must_use]
pub fn layout_of<T: Component>() -> ComponentLayout {
    let fields: &'static [FieldLayout] = Vec::leak(
        T::FIELDS
            .iter()
            .map(|field| FieldLayout {
                name: field.name.as_ptr(),
                ty: field.ty.as_ptr(),
                name_len: field.name.len() as u32,
                ty_len: field.ty.len() as u32,
                offset: field.offset as u32,
                size: field.size as u32,
            })
            .collect::<Vec<_>>(),
    );
    ComponentLayout {
        name: T::TYPE_NAME.as_ptr(),
        fields: fields.as_ptr(),
        declared: T::DECLARED_ID.as_ptr(),
        id: crate::component_id::<T>().get(),
        schema_hash: crate::schema_hash::<T>().get().to_le_bytes(),
        name_len: T::TYPE_NAME.len() as u32,
        declared_len: T::DECLARED_ID.len() as u32,
        field_len: fields.len() as u32,
        size: size_of::<T>() as u32,
        align: align_of::<T>() as u32,
    }
}

/// One systems-table entry: the declared name, its stable id, and the shim.
#[must_use]
pub fn entry(name: &'static str, run: SystemFn) -> SystemEntry {
    SystemEntry {
        name: name.as_ptr(),
        run,
        id: crate::hash::system_id(name),
        name_len: name.len() as u32,
        reserved: 0,
    }
}

/// Take the host table and arm the panic capture. What `gg_game_init` does.
///
/// # Safety
///
/// `api` must point at a valid [`HostApiV1`] the host owns for as long as this
/// dylib can be called into.
pub unsafe fn init(api: *const HostApiV1) {
    // SAFETY: forwarded to the caller, which is the host's `gg_game_init`
    // contract (§4.2.2).
    unsafe { gg_abi::host::set(api) };
    install_hook();
}

/// Run one system, converting a panic into a status instead of an abort.
///
/// # Safety
///
/// `world` must be the handle the host built for this tick and `ctx` must point
/// at a live [`TickCtx`]; neither may outlive the call.
pub unsafe fn run(
    world: *mut WorldHandle,
    ctx: *const TickCtx,
    system: fn(&mut GameWorld<'_>),
) -> SystemStatus {
    if ctx.is_null() {
        return panicked_with("gg-ecs: the host called a system with a null tick context");
    }
    // SAFETY: the caller's contract — `ctx` points at one live `TickCtx` that
    // outlives this call.
    let ctx = unsafe { &*ctx };

    // `AssertUnwindSafe` is the honest annotation, not a bypass: the only state
    // reachable through `GameWorld` is host-owned, and a panicking system halts
    // the sim (§4.2.2) rather than continuing over half-updated state.
    IN_SYSTEM.fetch_add(1, Ordering::Relaxed);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the caller's contract, forwarded — `world` is this tick's
        // handle and `ctx` outlives the borrow.
        let mut game = unsafe { GameWorld::new(world, ctx) };
        system(&mut game);
    }));
    IN_SYSTEM.fetch_sub(1, Ordering::Relaxed);

    match outcome {
        Ok(()) => SystemStatus::ok(),
        Err(payload) => panicked(payload),
    }
}

/// The last panic the hook saw, payload and location together.
///
/// A `Mutex<Option<String>>` rather than a channel or a thread-local: systems
/// run on the tick thread, the value is written and taken within one call, and
/// the lock is uncontended by construction. `thread_local!` is grep-banned in
/// game crates (§3) and this is the machinery those crates use.
static CAPTURED: Mutex<Option<String>> = Mutex::new(None);
static HOOK: Once = Once::new();

/// Whether a system is on the stack. Systems run on the tick thread and the sim
/// is single-threaded (§4.1), so a counter is enough and a `thread_local!` would
/// be the state §4.2.2 bans, spelled the banned way.
static IN_SYSTEM: AtomicUsize = AtomicUsize::new(0);

/// Install the capture hook, once.
///
/// Inside a system the hook **replaces** the default: the host reports a
/// panicking system by name (§4.2.2), so letting the default hook also print
/// would be the same failure twice, once without the name. Outside a system it
/// **chains** to whatever hook was already installed — swallowing every panic in
/// the process would make the engine's own failures silent, which is a far worse
/// trade than one duplicated line.
///
/// `set_hook` reaches the `std` this artifact was linked against. A game dylib
/// links its own copy, so installing here does not disturb the host's hook; a
/// test binary that calls [`init`] is the one place both are the same `std`, and
/// the chaining above is what keeps that case usable.
fn install_hook() {
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if IN_SYSTEM.load(Ordering::Relaxed) == 0 {
                previous(info);
                return;
            }
            let location = info
                .location()
                .map_or_else(|| "an unknown location".to_owned(), ToString::to_string);
            let message = format!("{} (at {location})", payload_text(info.payload()));
            *CAPTURED.lock().unwrap_or_else(PoisonError::into_inner) = Some(message);
        }));
    });
}

fn payload_text(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("a panic with a non-string payload")
}

fn panicked(payload: Box<dyn Any + Send>) -> SystemStatus {
    // The hook has the location; the payload alone does not. Falling back to the
    // payload keeps this total for a panic raised before `init` armed the hook.
    let message = CAPTURED
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take()
        .unwrap_or_else(|| payload_text(payload.as_ref()).to_owned());
    status_from(message)
}

fn panicked_with(message: &str) -> SystemStatus {
    status_from(message.to_owned())
}

/// Leaks the message. §4.2.2: dylib-owned and valid for the process's lifetime —
/// one string per halted sim, and free of the lifetime question a borrowed
/// buffer would raise across a seam that never unloads.
fn status_from(message: String) -> SystemStatus {
    let leaked: &'static str = String::leak(message);
    SystemStatus {
        code: SystemStatus::PANICKED,
        message_len: leaked.len() as u32,
        message: leaked.as_ptr(),
    }
}
