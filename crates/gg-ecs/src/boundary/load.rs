//! What the host does with a verified dylib's tables (§4.2.2): register the
//! components it declares, read the verbs it declares, then run its systems in
//! order.
//!
//! Both live here rather than in the shell for the reason §3 gives the shell:
//! zero engine logic. Driving a systems table needs a [`World`], the handle cast
//! and the panic-status protocol; registering a foreign component needs
//! [`ComponentInfo`]. All three are `gg-ecs`'s, and none of them is something an
//! app shell should be able to get subtly wrong.
//!
//! # Decoding is unsafe once, here
//!
//! The tables are pointers-and-lengths written by an artifact the host has
//! version- and fingerprint-checked but not otherwise trusted. Every read of one
//! is confined to this file, and every `&'static` it produces rests on the same
//! single fact: **a dylib is never unloaded** (§4.2.2, and [`GameLib`]'s
//! `ManuallyDrop` is the mechanism). Where that is not enough — a `FieldDesc` is
//! a different type from the `FieldLayout` on the wire — the conversion leaks,
//! which for a few hundred bytes beside a multi-megabyte leaked dylib is not a
//! budget anyone can notice.
//!
//! [`GameLib`]: https://docs.rs/gg-core

use gg_abi::{ComponentLayout, ComponentsTable, SystemsTable, TickCtx, VerbsTable};

use crate::component::FieldDesc;
use crate::hash::{ComponentId, SchemaHash};
use crate::registry::{ComponentInfo, RegistryError};
use crate::world::World;

/// A system that panicked, as the host reports it (§4.2.2).
///
/// Owned strings: the message is leaked dylib memory and the name points into
/// the systems table, and the value outlives both a swap and — through a
/// rejuvenation snapshot — the process that produced it.
#[derive(Clone, Debug, thiserror::Error)]
#[error("system `{system}` panicked: {message}")]
pub struct SystemPanic {
    /// The declared system name, which is why it crosses the boundary at all.
    pub system: String,
    /// Payload and source location, as the dylib-side shim captured them.
    pub message: String,
}

impl World {
    /// Register every component a game build declares.
    ///
    /// Returns how many were registered. Idempotent for an unchanged build, and
    /// a startup error (never a silent remap, §4.2) for a collision — including
    /// the reload-only one where an id comes back with a different schema, which
    /// is the caller's cue that this world needs [`restore`](World::restore)
    /// rather than reuse.
    ///
    /// # Safety
    ///
    /// `table` must be the `gg_game_components()` result of a dylib that passed
    /// [`verify`] and is never unloaded: every pointer it holds is read as
    /// `&'static` on exactly that basis.
    ///
    /// [`verify`]: https://docs.rs/gg-core
    pub unsafe fn adopt(&mut self, table: &ComponentsTable) -> Result<usize, RegistryError> {
        // SAFETY: the caller's obligation — `entries`/`len` describe one array
        // in the dylib's immutable static data, live for the process.
        let entries = unsafe { slice(table.entries, table.len) };
        for layout in entries {
            // SAFETY: as above, for the pointers inside each entry.
            self.registry_mut()
                .register_declared(unsafe { info(layout) })?;
        }
        Ok(entries.len())
    }

    /// Run one tick's systems in table order (§4.1: the table's order *is* the
    /// schedule), stopping at the first that panicked.
    ///
    /// Stopping is the point: a panicking system halts the sim cleanly (§4.2.2),
    /// and running the rest of the tick over state the failed one left half
    /// written would be a second, quieter bug.
    ///
    /// # Safety
    ///
    /// `table` must be the `gg_game_systems()` result of a verified, loaded
    /// dylib — its entries are called.
    pub unsafe fn run_systems(
        &mut self,
        table: &SystemsTable,
        ctx: &TickCtx,
    ) -> Result<(), SystemPanic> {
        let handle = super::handle(self);
        // Read once per tick, not per system: a `OnceLock` cannot change under
        // the loop, and the zone-less path is then one branch on a local.
        let zone = super::host::system_zone();
        // SAFETY: the caller's obligation — one live array of entries.
        for entry in unsafe { slice(table.entries, table.len) } {
            // SAFETY: `handle` is this call's world and `ctx` outlives the call,
            // which is the whole of `SystemFn`'s contract. The shim on the other
            // side catches panics, so this cannot unwind.
            let call = || unsafe { (entry.run)(handle, core::ptr::from_ref(ctx)) };
            let status = match zone {
                None => call(),
                Some(zone) => {
                    // SAFETY: the name is the table's, valid for the process.
                    let name = unsafe { text(entry.name, entry.name_len) };
                    let mut status = None;
                    zone(name, &mut || status = Some(call()));
                    let Some(status) = status else {
                        // A hook that returned without running the body left the
                        // tick half-simulated. Reported rather than assumed away:
                        // a silent `ok` here is a determinism break with an
                        // instrument as its cause (§4.8).
                        return Err(SystemPanic {
                            system: name.to_owned(),
                            message: "the installed system zone never ran the system".to_owned(),
                        });
                    };
                    status
                }
            };
            if !status.is_ok() {
                // SAFETY: the name is the table's; the message is leaked by the
                // shim and valid for the process (§4.2.2).
                return Err(unsafe {
                    SystemPanic {
                        system: text(entry.name, entry.name_len).to_owned(),
                        message: text(status.message, status.message_len).to_owned(),
                    }
                });
            }
        }
        Ok(())
    }
}

/// The verb names a build declares, in id order (§4.7).
///
/// Borrowed from the dylib's static data rather than owned: the lists outlive
/// every reader for the same reason the systems table does, and the host hands
/// them straight to `ActionMap::parse`, which wants `&[&str]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Verbs {
    /// Action names; index is [`ActionId`](gg_abi::ActionId).
    pub actions: &'static [&'static str],
    /// Axis names; index is [`AxisId`](gg_abi::AxisId).
    pub axes: &'static [&'static str],
}

/// Read a build's verb lists.
///
/// Leaks the two `&str` arrays — a `VerbName` is pointer-and-length and a `&str`
/// is a fat pointer, so this is the same conversion `info` makes for fields, at
/// the same cost and for the same reason.
///
/// # Safety
///
/// `table` must be the `gg_game_verbs()` result of a verified dylib that is
/// never unloaded.
#[must_use]
pub unsafe fn read_verbs(table: &VerbsTable) -> Verbs {
    // SAFETY: the caller's obligation, per list and then per name.
    unsafe {
        Verbs {
            actions: names(slice(table.actions, table.action_len)),
            axes: names(slice(table.axes, table.axis_len)),
        }
    }
}

/// # Safety
///
/// Every entry's `name`/`name_len` must describe live UTF-8.
unsafe fn names(entries: &[gg_abi::VerbName]) -> &'static [&'static str] {
    Vec::leak(
        entries
            .iter()
            // SAFETY: the caller's obligation.
            .map(|verb| unsafe { text(verb.name, verb.name_len) })
            .collect::<Vec<_>>(),
    )
}

/// One `ComponentLayout` as the registry sees it.
///
/// # Safety
///
/// Every pointer in `layout` must be valid UTF-8 of the stated length and live
/// for the process (§4.2.2: dylibs are never unloaded).
unsafe fn info(layout: &ComponentLayout) -> ComponentInfo {
    // SAFETY: the caller's obligation.
    let fields = unsafe { slice(layout.fields, layout.field_len) };
    let fields: &'static [FieldDesc] = Vec::leak(
        fields
            .iter()
            .map(|f| FieldDesc {
                // SAFETY: as above, per field.
                name: unsafe { text(f.name, f.name_len) },
                // SAFETY: as above, per field.
                ty: unsafe { text(f.ty, f.ty_len) },
                offset: f.offset as usize,
                size: f.size as usize,
            })
            .collect::<Vec<_>>(),
    );
    ComponentInfo {
        // Taken from the wire rather than recomputed from `declared`: the two
        // are the same function by construction — the fingerprint's scope is
        // `gg-abi` + `gg-ecs` + `gg-math`, so a dylib that hashed ids differently
        // was already refused (§4.2.2).
        id: ComponentId::from_raw(layout.id),
        schema: SchemaHash::from_raw(u128::from_le_bytes(layout.schema_hash)),
        // SAFETY: the caller's obligation.
        declared_id: unsafe { text(layout.declared, layout.declared_len) },
        // SAFETY: as above.
        type_name: unsafe { text(layout.name, layout.name_len) },
        size: layout.size as usize,
        align: layout.align as usize,
        fields,
        // A component the host has no type for encodes as its bytes, which is
        // what the raw fast path already does — so the two canonical hashes
        // agree here by construction rather than by comparison. The field-wise
        // protocol check that makes that a *derivation* runs in the dylib, where
        // the type exists (§4.2.1).
        protocol_hash: |bytes, h| h.bytes(bytes),
    }
}

/// # Safety
///
/// `ptr`/`len` must describe one live array, or `len` must be 0.
unsafe fn slice<'a, T>(ptr: *const T, len: u32) -> &'a [T] {
    if len == 0 {
        return &[];
    }
    // SAFETY: the caller's obligation.
    unsafe { core::slice::from_raw_parts(ptr, len as usize) }
}

/// Boundary text: UTF-8, not NUL-terminated, length beside it.
///
/// Invalid UTF-8 becomes a placeholder rather than a panic — every caller is on
/// a diagnostic path, and dying while describing a failure is the worst possible
/// place to be strict.
///
/// # Safety
///
/// `ptr`/`len` must describe one live byte array, or `len` must be 0.
unsafe fn text<'a>(ptr: *const u8, len: u32) -> &'a str {
    // SAFETY: the caller's obligation.
    core::str::from_utf8(unsafe { slice(ptr, len) }).unwrap_or("<not utf-8>")
}
