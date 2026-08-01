//! The query payload: what "one FFI call per archetype match" actually hands
//! over (§4.2.2).
//!
//! The hot path across this boundary must not pay FFI per entity. Every
//! component type that crosses is `repr(C)` by rule — hashed components already
//! are, since `bytemuck`'s `Pod` derive requires it — so a query materializes as
//! raw column pointers, lengths and strides, and the dylib iterates them
//! natively, inline, at full speed.
//!
//! **The borrow check happens host-side, at view construction, before any
//! pointer is handed over.** A [`QueryDesc`] naming one component twice mutably,
//! or both mutably and immutably, comes back as
//! [`AbiStatus::Alias`](crate::AbiStatus) — an error, not aliasing UB.

use crate::entity::Entity;

/// Columns one query may name, reads and writes together.
///
/// Fixed rather than heap-allocated so a match is one flat `repr(C)` value and
/// the dylib can page a query through stack space alone. A query touching more
/// than this many components is a design error long before it is a limit — the
/// same judgement `gg-ecs`'s `MAX_WRITES` already makes.
pub const MAX_QUERY_COLUMNS: usize = 16;

/// One column of one archetype match: base pointer, row count, stride.
///
/// A stride of zero is a marker component — `len` still counts rows. `gg-ecs`
/// re-exports this type; it is defined here because it is the payload, not
/// because storage borrowed it.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ColumnView {
    /// Base of the column. Aligned to the component's requirement (columns are
    /// 16-aligned; registration refuses anything stricter). Never null: an
    /// empty column still points at a valid, zero-length allocation base.
    pub ptr: *mut u8,
    /// Rows in this column.
    pub len: usize,
    /// Bytes per row.
    pub stride: usize,
}

/// The components a query reads and writes, by stable id.
///
/// Ids are raw `u64` here: `ComponentId` is a `gg-ecs` newtype, and `gg-ecs`
/// sits *above* this crate.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct QueryDesc {
    /// Stable ids read. May be null when `read_len` is 0.
    pub read: *const u64,
    /// Stable ids written. May be null when `write_len` is 0.
    pub write: *const u64,
    /// Entries in `read`.
    pub read_len: u32,
    /// Entries in `write`.
    pub write_len: u32,
}

/// One archetype holding every component the [`QueryDesc`] asked for.
///
/// Pointer fields are valid until the next call into the host through
/// [`HostApiV1`](crate::HostApiV1) — spawning or inserting can reallocate a
/// column, and a structural change moves rows between archetypes. Systems
/// iterate a match to completion, then mutate; that ordering is the contract,
/// and it is why the generated wrappers own the loop rather than the game.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ArchetypeMatch {
    /// The entity of each row, indexed by dense row (§4.2). Never null.
    pub entities: *const Entity,
    /// Rows in this archetype.
    pub len: usize,
    /// Reads first, in `QueryDesc::read` order, then writes in
    /// `QueryDesc::write` order. Entries past `read_len + write_len` are
    /// unspecified and must not be read.
    pub columns: [ColumnView; MAX_QUERY_COLUMNS],
}

impl ArchetypeMatch {
    /// A match describing nothing.
    ///
    /// The host writes into caller-provided storage, so the caller needs a value
    /// to fill an output page with before the call — `[ArchetypeMatch::EMPTY; N]`
    /// rather than `MaybeUninit`, since every field here is a pointer or a count
    /// and there is nothing about "empty" the boundary cannot spell.
    pub const EMPTY: Self = Self {
        entities: core::ptr::null(),
        len: 0,
        columns: [ColumnView {
            ptr: core::ptr::null_mut(),
            len: 0,
            stride: 0,
        }; MAX_QUERY_COLUMNS],
    };
}
