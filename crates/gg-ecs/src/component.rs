//! Component identity and schema fingerprints (§4.2).
//!
//! Two facts every component carries, both emitted by `#[derive(Component)]`:
//!
//! - a **stable id**, hashed from a *declared* string rather than the Rust type
//!   path, so moving `game::ships::Velocity` to `game::physics::Velocity` is a
//!   source-organization event and not a persisted-schema event;
//! - a **schema hash** over [`FieldDesc`]s, so a reload can tell "reuse this
//!   column in place" from "snapshot, migrate, restore" (§4.2.2).
//!
//! # Why components are `Pod`
//!
//! Not for hashing's sake — [`StateHash`](crate::StateHash) works fine on
//! non-`Pod` types, which is what hashed side tables are built on. `Pod` is
//! load-bearing three other ways (§4.2.1): the reload boundary hands raw
//! `repr(C)` column pointers across a C ABI, heap-backed component data would
//! put a leaked dylib's allocations inside host-owned state, and offset-mapped
//! migration is only a column copy because columns are flat.
//!
//! The tax is real, and the sanctioned patterns are fixed rather than
//! improvised per component: **enums** become newtyped integers with validated
//! accessors, **flags** become explicit bit sets, **dynamic data** becomes
//! fixed-capacity inline arrays or indices into host-owned side arenas.

use crate::hash::{ComponentId, SchemaHash, StateHasher};

/// One field of a component's layout, emitted as a `const` by the derive.
///
/// This is also the input to §4.2.2's offset-mapped migration: matching by
/// `name` across two versions gives copy/default/drop decisions directly, which
/// is why the descriptor is retained rather than being folded into the hash and
/// discarded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldDesc {
    /// The Rust field name — migration's matching key (§4.2.2).
    pub name: &'static str,
    /// The field's source type token, e.g. `"f32"`, `"[u32; 4]"`, `"Entity"`.
    ///
    /// Syntactic, so a type alias swap that preserves layout still moves the
    /// schema hash. That errs toward migrating unnecessarily, never toward
    /// reusing a column it should not have — the safe direction.
    pub ty: &'static str,
    /// Byte offset within the component, from `core::mem::offset_of!`.
    pub offset: usize,
    /// Byte size, from `core::mem::size_of`.
    pub size: usize,
}

impl FieldDesc {
    fn hash_into(&self, h: &mut StateHasher) {
        h.str(self.name);
        h.str(self.ty);
        h.u64(self.offset as u64);
        h.u64(self.size as u64);
    }
}

/// A type storable in an archetype column.
///
/// Implement via `#[derive(Component)]` with a mandatory
/// `#[component(id = "…")]`. The attribute is mandatory so the choice is made
/// once, at birth, when it costs nothing — a component that ships without one
/// would have to acquire an id later, which is a schema break wearing a
/// bugfix's clothes.
///
/// The `Pod` bound is where §4.2.1 hazard 4 becomes a compile error: a `bool`
/// next to a `u32` does not reach review.
pub trait Component: bytemuck::Pod + crate::StateHash + Send + Sync + 'static {
    /// The declared id string. Persisted identity; changing it is a schema
    /// change and the migration path treats it as one.
    const DECLARED_ID: &'static str;
    /// [`DECLARED_ID`](Self::DECLARED_ID) hashed, folded by the derive at
    /// expansion time so a lookup is a constant and not a blake3 (§4.2).
    /// [`Registry::register`](crate::Registry::register) proves it equals
    /// [`ComponentId::of`] on every registration; nothing else should hash an
    /// id that is already known.
    const COMPONENT_ID: ComponentId;
    /// The Rust type name. Diagnostics only — never hashed, never persisted,
    /// so refactors stay free.
    const TYPE_NAME: &'static str;
    /// Fields in declaration order.
    const FIELDS: &'static [FieldDesc];
}

/// A component's stable id (§4.2) — a constant read, not a hash.
///
/// It was a hash per call until the query micro priced it: `read_of`/`write_of`
/// resolve a term *per matched archetype*, so a pass over 256 small archetypes
/// paid 512 blake3s to look up ids it knew before the pass began. The derive
/// folds the same construction at expansion time against the same pinned
/// blake3, and [`Registry::register`](crate::Registry::register) proves the two
/// agree — so the determinism hazard the old note feared (a second
/// implementation) is neither taken nor left unchecked.
#[must_use]
pub fn component_id<T: Component>() -> ComponentId {
    T::COMPONENT_ID
}

/// A component's layout fingerprint (§4.2).
///
/// Covers the declared id as well as the fields: two components with identical
/// layouts but different ids must not be mistaken for one another by a
/// migration pass that only compared schemas.
#[must_use]
pub fn schema_hash<T: Component>() -> SchemaHash {
    let mut h = StateHasher::schema();
    h.str(T::DECLARED_ID);
    h.u64(T::FIELDS.len() as u64);
    for field in T::FIELDS {
        field.hash_into(&mut h);
    }
    h.u64(core::mem::size_of::<T>() as u64);
    h.u64(core::mem::align_of::<T>() as u64);
    h.finish_schema()
}
