//! Hashed side tables (§4.2.1) — sim state that is hashed but not stored in
//! columns.
//!
//! Components are `Pod` for three reasons that have nothing to do with hashing:
//! the reload boundary hands raw `repr(C)` column pointers across a C ABI
//! (§4.2.2), heap-backed component data would put a leaked dylib's allocations
//! inside host-owned state, and offset-mapped migration is only a column copy
//! because columns are flat. The `StateHash` protocol has no such requirement —
//! a growable list hashes as length-then-elements perfectly deterministically.
//!
//! This module is where that decoupling pays out. A side table is host-owned
//! sim state (a production queue, a trade-route graph, an AI plan store), keyed
//! by entity id, free to hold `Vec`s and `String`s, and hashed in the canonical
//! pass in a defined order alongside the columns. The strategy layer of a
//! Paradox-shaped sim lives here, with components holding handles — which is
//! recorded now so nobody later mistakes the component `Pod` rule for the
//! gameplay data model.
//!
//! # Order
//!
//! Registration order is irrelevant: tables are walked ascending by
//! [`SideTableId`], the same rule the component registry follows, so the
//! canonical hash cannot depend on which system happened to register first
//! (§4.2.1 hazard 3).

use core::any::Any;

use crate::hash::{SideTableId, StateHasher};
use crate::state_hash::StateHash;

/// Host-owned sim state hashed alongside the columns (§4.2.1).
///
/// Derive it — `#[derive(SideTable)]` with `#[side_table(id = "…")]` emits this
/// and the [`StateHash`] impl together, and applies the same field-type
/// rejection components get. Implementing by hand is legal and occasionally
/// right (a table whose encoding is not its field order), but then the encoding
/// is yours to keep stable.
pub trait SideTable: StateHash + Send + Sync + 'static {
    /// Persisted identity, decoupled from the Rust type path exactly as
    /// [`Component::DECLARED_ID`](crate::Component::DECLARED_ID) is.
    const DECLARED_ID: &'static str;
    /// The Rust type name. Diagnostics only.
    const TYPE_NAME: &'static str;
}

/// Registration failures — startup errors, never silent remaps (§4.2).
#[derive(Debug, thiserror::Error)]
pub enum SideTableError {
    #[error(
        "side-table id collision: `{first_type}` (id \"{first_declared}\") and `{second_type}` \
         (id \"{second_declared}\") both hash to {id:?}. Change one declared id — they are \
         persisted identities and cannot be shared."
    )]
    IdCollision {
        id: SideTableId,
        first_type: &'static str,
        first_declared: &'static str,
        second_type: &'static str,
        second_declared: &'static str,
    },
    #[error(
        "side-table id \"{declared}\" is declared by two types, `{first_type}` and \
         `{second_type}`. Two types claiming one id would load each other's saved state."
    )]
    DuplicateDeclaration {
        declared: &'static str,
        first_type: &'static str,
        second_type: &'static str,
    },
    #[error(
        "side table `{type_name}` (id \"{declared}\") is already installed. Take it out with \
         `remove_side_table` first, or mutate it in place."
    )]
    AlreadyInstalled {
        declared: &'static str,
        type_name: &'static str,
    },
}

/// The object-safe face of [`SideTable`], so the world can hold a sorted list of
/// heterogeneous tables.
pub(crate) trait ErasedSideTable: Any + Send + Sync {
    fn hash_into(&self, h: &mut StateHasher);
    fn declared_id(&self) -> &'static str;
    fn type_name(&self) -> &'static str;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn into_any(self: Box<Self>) -> Box<dyn Any>;
}

impl<T: SideTable> ErasedSideTable for T {
    fn hash_into(&self, h: &mut StateHasher) {
        self.state_hash(h);
    }

    fn declared_id(&self) -> &'static str {
        T::DECLARED_ID
    }

    fn type_name(&self) -> &'static str {
        T::TYPE_NAME
    }

    // `Any` here is a within-process downcast, not an identity claim. §4.2 bans
    // `TypeId` as *persisted* identity — that job belongs to `DECLARED_ID`,
    // which is what the id, the hash and any future save format are built on.
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

/// Installed side tables, kept sorted by [`SideTableId`].
#[derive(Default)]
pub(crate) struct SideTables {
    sorted: Vec<(SideTableId, Box<dyn ErasedSideTable>)>,
}

impl SideTables {
    pub(crate) fn insert<T: SideTable>(&mut self, table: T) -> Result<SideTableId, SideTableError> {
        let id = SideTableId::of(T::DECLARED_ID);
        match self.sorted.binary_search_by_key(&id, |(k, _)| *k) {
            Ok(at) => {
                let existing = &self.sorted[at].1;
                Err(if existing.type_name() == T::TYPE_NAME {
                    SideTableError::AlreadyInstalled {
                        declared: T::DECLARED_ID,
                        type_name: T::TYPE_NAME,
                    }
                } else if existing.declared_id() == T::DECLARED_ID {
                    SideTableError::DuplicateDeclaration {
                        declared: T::DECLARED_ID,
                        first_type: existing.type_name(),
                        second_type: T::TYPE_NAME,
                    }
                } else {
                    SideTableError::IdCollision {
                        id,
                        first_type: existing.type_name(),
                        first_declared: existing.declared_id(),
                        second_type: T::TYPE_NAME,
                        second_declared: T::DECLARED_ID,
                    }
                })
            }
            Err(at) => {
                self.sorted.insert(at, (id, Box::new(table)));
                Ok(id)
            }
        }
    }

    pub(crate) fn get<T: SideTable>(&self) -> Option<&T> {
        let id = SideTableId::of(T::DECLARED_ID);
        let at = self.sorted.binary_search_by_key(&id, |(k, _)| *k).ok()?;
        self.sorted[at].1.as_any().downcast_ref::<T>()
    }

    pub(crate) fn get_mut<T: SideTable>(&mut self) -> Option<&mut T> {
        let id = SideTableId::of(T::DECLARED_ID);
        let at = self.sorted.binary_search_by_key(&id, |(k, _)| *k).ok()?;
        self.sorted[at].1.as_any_mut().downcast_mut::<T>()
    }

    pub(crate) fn remove<T: SideTable>(&mut self) -> Option<T> {
        let id = SideTableId::of(T::DECLARED_ID);
        let at = self.sorted.binary_search_by_key(&id, |(k, _)| *k).ok()?;
        // Probe before removing: an id collision must not silently hand back
        // someone else's table, and a failed downcast after `remove` would have
        // already taken it out of the world.
        self.sorted[at].1.as_any().downcast_ref::<T>()?;
        let (_, table) = self.sorted.remove(at);
        table.into_any().downcast::<T>().ok().map(|b| *b)
    }

    pub(crate) fn len(&self) -> usize {
        self.sorted.len()
    }

    /// Ascending by id, so the canonical pass sees one order regardless of when
    /// each table was installed.
    pub(crate) fn hash_into(&self, h: &mut StateHasher) {
        h.u64(self.sorted.len() as u64);
        for (id, table) in &self.sorted {
            h.u64(id.get());
            table.hash_into(h);
        }
    }
}
