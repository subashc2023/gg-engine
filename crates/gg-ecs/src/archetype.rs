//! Archetypes and the graph that connects them (§4.2).
//!
//! An archetype is one exact component set, stored as parallel columns plus a
//! row→entity map. Adding or removing a component moves an entity's row to
//! another archetype; the **graph** is the cache of those edges, so the second
//! `insert::<Velocity>()` on a given archetype is a lookup rather than a
//! set-union and a search.
//!
//! # Ordering, everywhere, on purpose
//!
//! Component ids within an archetype are sorted, edges are sorted, and the
//! archetype list is walked in creation order. None of that is incidental:
//! §4.2 guarantee 1 promises stable iteration for a given operation sequence,
//! and the cheapest way to keep that promise is to leave no unordered
//! container anywhere in the structure (§4.2.1 hazard 3).

use crate::column::Column;
use crate::entity::Entity;
use crate::hash::ComponentId;

/// Index into [`World`](crate::World)'s archetype list.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct ArchetypeId(pub(crate) u32);

impl ArchetypeId {
    /// The empty archetype — every world has it at index 0, and a bare
    /// `spawn()` lands there.
    pub(crate) const EMPTY: Self = Self(0);

    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// Where one entity's data lives.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Location {
    pub(crate) archetype: ArchetypeId,
    /// The dense index (§4.2). Deterministic, not persistent: swap-remove
    /// moves the last row into any hole, so this is stable only between
    /// structural changes to this archetype.
    pub(crate) row: u32,
}

pub(crate) struct Archetype {
    /// Component ids, ascending. `columns` is parallel to this.
    ids: Vec<ComponentId>,
    columns: Vec<Column>,
    /// Row → entity. The inverse of [`Location::row`], and what swap-remove
    /// needs in order to tell the world which entity just moved.
    entities: Vec<Entity>,
    /// `(component, destination)` for adding a component, ascending.
    add_edges: Vec<(ComponentId, ArchetypeId)>,
    /// `(component, destination)` for removing one, ascending.
    remove_edges: Vec<(ComponentId, ArchetypeId)>,
}

impl Archetype {
    /// `ids` must be sorted and deduplicated; `strides` is parallel to it.
    pub(crate) fn new(ids: Vec<ComponentId>, strides: &[usize]) -> Self {
        debug_assert!(ids.windows(2).all(|w| w[0] < w[1]), "ids sorted and unique");
        debug_assert_eq!(ids.len(), strides.len());
        Self {
            columns: strides.iter().map(|&s| Column::new(s)).collect(),
            ids,
            entities: Vec::new(),
            add_edges: Vec::new(),
            remove_edges: Vec::new(),
        }
    }

    pub(crate) fn ids(&self) -> &[ComponentId] {
        &self.ids
    }

    pub(crate) fn len(&self) -> usize {
        self.entities.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Raw base of the row→entity map, for [`crate::view::build`], which must
    /// not hold a shared reference to this archetype alongside the mutable
    /// column pointers it also takes.
    pub(crate) fn entities_ptr(&mut self) -> *const Entity {
        self.entities.as_ptr()
    }

    /// The same, for [`crate::view::build_ref`], where the archetype is only
    /// ever borrowed shared and no column pointer is written through.
    pub(crate) fn entities_ptr_shared(&self) -> *const Entity {
        self.entities.as_ptr()
    }

    /// The row→entity map as a slice, for the snapshot walk (§4.8) — which
    /// wants the values, not a pointer to alias.
    pub(crate) fn entities_slice(&self) -> &[Entity] {
        &self.entities
    }

    /// Position of `id` in this archetype's parallel arrays.
    pub(crate) fn column_index(&self, id: ComponentId) -> Option<usize> {
        self.ids.binary_search(&id).ok()
    }

    pub(crate) fn contains(&self, id: ComponentId) -> bool {
        self.column_index(id).is_some()
    }

    /// Does this archetype hold every id in `needed`? Both sides are sorted, so
    /// this is a merge rather than a nested scan.
    pub(crate) fn contains_all(&self, needed: &[ComponentId]) -> bool {
        let mut have = self.ids.iter();
        needed.iter().all(|want| have.any(|got| got == want))
    }

    pub(crate) fn column(&self, at: usize) -> &Column {
        &self.columns[at]
    }

    pub(crate) fn column_mut(&mut self, at: usize) -> &mut Column {
        &mut self.columns[at]
    }

    /// Append `entity` with every column zeroed. The caller writes real values
    /// into the returned row.
    pub(crate) fn push_zeroed(&mut self, entity: Entity) -> u32 {
        for column in &mut self.columns {
            column.push_zeroed();
        }
        self.entities.push(entity);
        (self.entities.len() - 1) as u32
    }

    /// Remove `row` by swap-remove.
    ///
    /// Returns the entity that was moved into the hole, if any — the world must
    /// update that entity's location, and returning it is how this type avoids
    /// needing to know about the world at all.
    pub(crate) fn swap_remove(&mut self, row: u32) -> Option<Entity> {
        let row = row as usize;
        for column in &mut self.columns {
            column.swap_remove(row);
        }
        self.entities.swap_remove(row);
        // `swap_remove` on a Vec moves the last element into `row`, so anything
        // now at `row` is the entity that moved — unless `row` was the last.
        self.entities.get(row).copied()
    }

    pub(crate) fn add_edge(&self, id: ComponentId) -> Option<ArchetypeId> {
        Self::edge(&self.add_edges, id)
    }

    pub(crate) fn remove_edge(&self, id: ComponentId) -> Option<ArchetypeId> {
        Self::edge(&self.remove_edges, id)
    }

    pub(crate) fn set_add_edge(&mut self, id: ComponentId, to: ArchetypeId) {
        Self::insert_edge(&mut self.add_edges, id, to);
    }

    pub(crate) fn set_remove_edge(&mut self, id: ComponentId, to: ArchetypeId) {
        Self::insert_edge(&mut self.remove_edges, id, to);
    }

    fn edge(edges: &[(ComponentId, ArchetypeId)], id: ComponentId) -> Option<ArchetypeId> {
        edges
            .binary_search_by_key(&id, |&(c, _)| c)
            .ok()
            .map(|at| edges[at].1)
    }

    fn insert_edge(edges: &mut Vec<(ComponentId, ArchetypeId)>, id: ComponentId, to: ArchetypeId) {
        if let Err(at) = edges.binary_search_by_key(&id, |&(c, _)| c) {
            edges.insert(at, (id, to));
        }
    }
}
