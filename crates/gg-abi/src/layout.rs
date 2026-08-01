//! Component layout, as the migration path needs to see it (§4.2.2).
//!
//! The host pulls these from the dylib at load rather than the dylib pushing
//! registrations: the host needs the descriptors anyway to compare this build's
//! schemas against the running world's, and a pull keeps the ordering — refuse
//! by version, then read tables — that the crate docs make a contract.
//!
//! Migration for `Pod` components is a field-name-matched, offset-mapped column
//! copy: matched names copy, new fields default, removed fields drop, retyped
//! fields default with a logged warning. That is why [`FieldLayout`] is a
//! *retained descriptor* rather than something folded into the schema hash and
//! discarded.
//!
//! `gg-ecs` has its own `FieldDesc` over `&'static str`. It is not this type and
//! must not become it: a `&str` is a fat pointer with no defined cross-artifact
//! layout, so the boundary spells out pointer and length itself.

/// One field of a component, in the C form.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct FieldLayout {
    /// The Rust field name — migration's matching key. UTF-8, not
    /// NUL-terminated.
    pub name: *const u8,
    /// The field's source type token, e.g. `"f32"`, `"[u32; 4]"`, `"Entity"`.
    /// Syntactic, so an alias swap that preserves layout still moves the schema
    /// hash — erring toward migrating unnecessarily, never toward reusing a
    /// column it should not have.
    pub ty: *const u8,
    /// Bytes of `name`.
    pub name_len: u32,
    /// Bytes of `ty`.
    pub ty_len: u32,
    /// Byte offset within the component.
    pub offset: u32,
    /// Byte size of the field.
    pub size: u32,
}

/// One component type a game build declares.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ComponentLayout {
    /// The Rust type name. Diagnostics only — never hashed, never persisted, so
    /// refactors stay free.
    pub name: *const u8,
    /// `field_len` field descriptors, in declaration order.
    pub fields: *const FieldLayout,
    /// The declared id string `id` is the hash of. Crosses as well as the hash
    /// because the host is what reports a collision and a migration outcome, and
    /// both are about *persisted identity*: naming the Rust type there would name
    /// the one thing §4.2 says identity is not.
    pub declared: *const u8,
    /// Stable id: the 64-bit hash of the *declared* id string, not the type
    /// path. Persisted identity; changing it is a schema change and migration
    /// treats it as one.
    pub id: u64,
    /// Layout fingerprint, little-endian. Equal hashes mean the column can be
    /// reused in place across a reload; unequal means snapshot, migrate,
    /// restore.
    pub schema_hash: [u8; 16],
    /// Bytes of `name`.
    pub name_len: u32,
    /// Bytes of `declared`.
    pub declared_len: u32,
    /// Entries in `fields`.
    pub field_len: u32,
    /// `size_of` the component.
    pub size: u32,
    /// `align_of` the component.
    pub align: u32,
}

/// Every component type a game build declares.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ComponentsTable {
    /// `len` entries, valid for the dylib's lifetime.
    pub entries: *const ComponentLayout,
    /// Entries in `entries`.
    pub len: u32,
    /// Must be zero.
    pub reserved: u32,
}
