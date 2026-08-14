//! The world: entities, their components, and the canonical hash over both
//! (§4.2, §4.2.1).

use crate::archetype::{Archetype, ArchetypeId, Location};
use crate::component::Component;
use crate::entity::{Entities, Entity};
use crate::hash::{Block, CanonicalHash, ComponentId, StateHasher};
use crate::query::{Query, QueryData, ReadOnly};
use crate::registry::{Registry, RegistryError};
use crate::side_table::{SideTable, SideTableError, SideTables};
use crate::view::{ArchetypeView, QueryAccess};

pub mod save;
pub mod snapshot;

/// Which encoding a canonical walk uses. Both must agree; see
/// [`World::canonical_hash_via_protocol`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum Encoding {
    /// Raw row bytes — valid because components are `Pod` + `repr(C)` on
    /// little-endian targets.
    RawFastPath,
    /// The explicit `StateHash` protocol, via the registry's fn pointer.
    Protocol,
}

/// Operations that can fail for a reason the caller might act on.
///
/// Deliberately narrow: a dead entity is reported as `None`/`false` by the
/// accessors rather than as an error, because "this entity was despawned last
/// tick" is ordinary control flow in gameplay code, not an exceptional
/// condition.
#[derive(Debug, thiserror::Error)]
pub enum WorldError {
    /// Component registration failed.
    #[error(transparent)]
    Registry(#[from] RegistryError),
    /// Side-table installation failed.
    #[error(transparent)]
    SideTable(#[from] SideTableError),
    /// An operation that must name a live entity was given a dead one. The
    /// *accessors* report a dead entity as `None`/`false` instead — only the
    /// mutating paths raise this.
    #[error("entity {entity:?} is not alive")]
    DeadEntity {
        /// The entity that was not alive.
        entity: Entity,
    },
}

/// Sim state: an entity allocator, a component registry, and the archetypes.
///
/// # What is guaranteed (§4.2)
///
/// 1. **Stable iteration order** for a given sequence of spawns, despawns and
///    mutations.
/// 2. **Deterministic entity-id allocation** — see [`crate::entity`].
/// 3. **Stable component identity across compilations** — declared ids, never
///    the Rust type path.
/// 4. **Identical canonical hash** for a given replay on every supported
///    architecture, across build tiers and link modes.
///
/// Each has a test that fails if the behaviour changes. Internals below that
/// surface are explicitly fair game for rework.
#[derive(Default)]
pub struct World {
    entities: Entities,
    registry: Registry,
    /// Index is [`ArchetypeId`]. Index 0 is always the empty archetype.
    archetypes: Vec<Archetype>,
    /// `(component set, archetype)`, sorted by the set — the lookup that turns
    /// a component set into an archetype without a hash map (hazard 3).
    by_ids: Vec<(Vec<ComponentId>, ArchetypeId)>,
    /// Hashed side tables (§4.2.1), sorted by id — sim state above the columns.
    side_tables: SideTables,
    /// Entity index → location. Sparse: `None` for dead or never-used indices.
    locations: Vec<Option<Location>>,
    /// Reused across structural changes to avoid a fresh buffer per move;
    /// `move_row` still allocates `carried` plus a `to_vec` per shared column.
    scratch: Vec<u8>,
}

impl World {
    /// An empty world holding only the empty archetype.
    #[must_use]
    pub fn new() -> Self {
        let mut world = Self::default();
        // The empty archetype must be index 0 so `spawn` needs no lookup.
        world.archetypes.push(Archetype::new(Vec::new(), &[]));
        world.by_ids.push((Vec::new(), ArchetypeId::EMPTY));
        world
    }

    /// Live entities.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.entities.len()
    }

    /// Whether no entity is live.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// The component registry, for reading what is registered and with which
    /// schema. Registration itself goes through [`Self::register`].
    #[must_use]
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    /// For `boundary::load`, which registers what a dylib declares and is a
    /// sibling module rather than a child of this one. Not public: registration
    /// is a checked act (§4.2) and the checks live in [`Registry`].
    pub(crate) fn registry_mut(&mut self) -> &mut Registry {
        &mut self.registry
    }

    /// Whether `entity` is live — generation included, so a recycled index
    /// does not resurrect an old handle.
    #[must_use]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    /// Register a component type up front.
    ///
    /// Optional — [`insert`](Self::insert) registers on demand. Doing it at
    /// startup is still worth it: an id collision found during world setup
    /// names two types, while one found mid-tick names them from inside a
    /// system's call stack.
    pub fn register<T: Component>(&mut self) -> Result<ComponentId, RegistryError> {
        self.registry.register::<T>()
    }

    /// Create an entity with no components.
    pub fn spawn(&mut self) -> Entity {
        let entity = self.entities.alloc();
        let row = self.archetypes[0].push_zeroed(entity);
        self.set_location(
            entity,
            Some(Location {
                archetype: ArchetypeId::EMPTY,
                row,
            }),
        );
        entity
    }

    /// Destroy an entity. Returns whether it was alive.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }
        if let Some(loc) = self.location(entity) {
            self.remove_row(loc);
        }
        self.set_location(entity, None);
        self.entities.free(entity)
    }

    /// Add or overwrite a component.
    pub fn insert<T: Component>(&mut self, entity: Entity, value: T) -> Result<(), WorldError> {
        if !self.entities.is_alive(entity) {
            return Err(WorldError::DeadEntity { entity });
        }
        let id = self.registry.register::<T>()?;
        let loc = self
            .location(entity)
            .ok_or(WorldError::DeadEntity { entity })?;

        // Already present: overwrite in place, no archetype move.
        if let Some(at) = self.archetypes[loc.archetype.index() as usize].column_index(id) {
            let column = self.archetypes[loc.archetype.index() as usize].column_mut(at);
            column
                .row_mut(loc.row as usize)
                .copy_from_slice(bytemuck::bytes_of(&value));
            return Ok(());
        }

        let target = self.archetype_with(loc.archetype, id);
        let new_row = self.move_row(entity, loc, target);
        let at = self.archetypes[target.index() as usize]
            .column_index(id)
            .unwrap_or_else(|| unreachable!("target archetype was built to contain {id:?}"));
        self.archetypes[target.index() as usize]
            .column_mut(at)
            .row_mut(new_row as usize)
            .copy_from_slice(bytemuck::bytes_of(&value));
        Ok(())
    }

    /// A new entity carrying a copy of every component `entity` has. `None` if
    /// it is dead.
    ///
    /// Public where [`World::insert_raw`](Self::insert_raw) deliberately is not,
    /// and the difference is the contract rather than the mechanism: an untyped
    /// insert is "here are some bytes, trust me", which is the §4.2.2 boundary's
    /// bargain and nobody else's. This is a *closed* operation — the bytes come
    /// out of this world's own columns and go straight back into them, so there
    /// is no way to spell a call that puts the wrong ones anywhere. What it buys
    /// is an editor that can duplicate an entity of a type it was never compiled
    /// against (§6 M15.4 item 5).
    ///
    /// The copy is a new [`Entity`] with its own index and generation. Nothing
    /// dereferences it — no component here holds an entity id the engine knows
    /// how to follow — so a game whose components point at each other by id gets
    /// a duplicate pointing where the original did, which is the only answer a
    /// copy without a schema for references can give.
    pub fn duplicate(&mut self, entity: Entity) -> Option<Entity> {
        let loc = self.location(entity)?;
        let source = &self.archetypes[loc.archetype.index() as usize];
        // Copied out before anything moves: `spawn` and the inserts below both
        // reshape storage, and a borrow of a column would not survive either.
        let rows: Vec<(ComponentId, Vec<u8>)> = source
            .ids()
            .iter()
            .enumerate()
            .map(|(at, id)| (*id, source.column(at).row(loc.row as usize).to_vec()))
            .collect();
        let copy = self.spawn();
        for (id, bytes) in rows {
            if self.insert_raw(copy, id, &bytes) != gg_abi::AbiStatus::Ok {
                // All or nothing: half an entity is worse than none, and the
                // only way here is a registry that lost a component between two
                // statements.
                self.despawn(copy);
                return None;
            }
        }
        Some(copy)
    }

    /// Remove a component. Returns whether the entity had it.
    pub fn remove<T: Component>(&mut self, entity: Entity) -> bool {
        let id = crate::component_id::<T>();
        let Some(loc) = self.location(entity) else {
            return false;
        };
        if !self.archetypes[loc.archetype.index() as usize].contains(id) {
            return false;
        }
        let target = self.archetype_without(loc.archetype, id);
        self.move_row(entity, loc, target);
        true
    }

    /// Whether `entity` carries `T`. `false` for a dead entity — a despawn last
    /// tick is ordinary control flow, not an error.
    #[must_use]
    pub fn contains<T: Component>(&self, entity: Entity) -> bool {
        self.location(entity).is_some_and(|loc| {
            self.archetypes[loc.archetype.index() as usize].contains(crate::component_id::<T>())
        })
    }

    /// Read one component. `None` if the entity is dead or does not carry `T`.
    #[must_use]
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        let loc = self.location(entity)?;
        let archetype = &self.archetypes[loc.archetype.index() as usize];
        let at = archetype.column_index(crate::component_id::<T>())?;
        Some(bytemuck::from_bytes(
            archetype.column(at).row(loc.row as usize),
        ))
    }

    /// As [`Self::get`], mutably. Writing through this changes the canonical
    /// hash, which is the point: it is sim state.
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        let loc = self.location(entity)?;
        let archetype = &mut self.archetypes[loc.archetype.index() as usize];
        let at = archetype.column_index(crate::component_id::<T>())?;
        Some(bytemuck::from_bytes_mut(
            archetype.column_mut(at).row_mut(loc.row as usize),
        ))
    }

    // ---- by stable id, for the §4.2.2 boundary ---------------------------
    //
    // The dylib names components by `u64`, never by type: it has no `T` to hand
    // us and we have no way to trust one if it did. These three mirror
    // `insert`/`remove`/`get` with the type replaced by a registry lookup, which
    // is also what makes them *refuse* an unregistered id — the host registers
    // everything a dylib declares at load, so an unknown id means the dylib
    // asked for something it never declared. Not public: raw bytes into a column
    // is the boundary's contract, not the ECS's.

    pub(crate) fn insert_raw(
        &mut self,
        entity: Entity,
        id: ComponentId,
        bytes: &[u8],
    ) -> gg_abi::AbiStatus {
        let Some(info) = self.registry.get(id) else {
            return gg_abi::AbiStatus::UnknownComponent;
        };
        if info.size != bytes.len() {
            return gg_abi::AbiStatus::SizeMismatch;
        }
        let Some(loc) = self.location(entity) else {
            return gg_abi::AbiStatus::DeadEntity;
        };

        if let Some(at) = self.archetypes[loc.archetype.index() as usize].column_index(id) {
            self.archetypes[loc.archetype.index() as usize]
                .column_mut(at)
                .row_mut(loc.row as usize)
                .copy_from_slice(bytes);
            return gg_abi::AbiStatus::Ok;
        }

        let target = self.archetype_with(loc.archetype, id);
        let new_row = self.move_row(entity, loc, target);
        let Some(at) = self.archetypes[target.index() as usize].column_index(id) else {
            // The target archetype was built to contain `id`; reaching here is a
            // storage bug, and the boundary reports rather than panics inside a
            // host call the dylib cannot unwind out of.
            return gg_abi::AbiStatus::UnknownComponent;
        };
        self.archetypes[target.index() as usize]
            .column_mut(at)
            .row_mut(new_row as usize)
            .copy_from_slice(bytes);
        gg_abi::AbiStatus::Ok
    }

    pub(crate) fn remove_raw(&mut self, entity: Entity, id: ComponentId) -> bool {
        let Some(loc) = self.location(entity) else {
            return false;
        };
        if !self.archetypes[loc.archetype.index() as usize].contains(id) {
            return false;
        }
        let target = self.archetype_without(loc.archetype, id);
        self.move_row(entity, loc, target);
        true
    }

    /// A pointer to one row of one column, or null. Invalidated by the next
    /// structural change — the same rule the column views carry.
    pub(crate) fn get_raw(&mut self, entity: Entity, id: ComponentId) -> *mut u8 {
        let Some(loc) = self.location(entity) else {
            return core::ptr::null_mut();
        };
        let archetype = &mut self.archetypes[loc.archetype.index() as usize];
        let Some(at) = archetype.column_index(id) else {
            return core::ptr::null_mut();
        };
        archetype
            .column_mut(at)
            .row_mut(loc.row as usize)
            .as_mut_ptr()
    }

    // ---- hashed side tables (§4.2.1) -------------------------------------

    /// Install a hashed side table — sim state above the columns, free of the
    /// `Pod` rule and hashed in the canonical pass alongside them.
    ///
    /// # Errors
    ///
    /// If a table with this id is already installed, or two types claim one id.
    pub fn insert_side_table<T: SideTable>(&mut self, table: T) -> Result<(), SideTableError> {
        self.side_tables.insert(table).map(|_| ())
    }

    /// The installed side table of type `T`, if one is installed.
    #[must_use]
    pub fn side_table<T: SideTable>(&self) -> Option<&T> {
        self.side_tables.get::<T>()
    }

    /// As [`Self::side_table`], mutably.
    pub fn side_table_mut<T: SideTable>(&mut self) -> Option<&mut T> {
        self.side_tables.get_mut::<T>()
    }

    /// Take a side table back out. Removing one changes the canonical hash — it
    /// is state leaving the world, not a view being closed.
    pub fn remove_side_table<T: SideTable>(&mut self) -> Option<T> {
        self.side_tables.remove::<T>()
    }

    /// How many side tables are installed.
    #[must_use]
    pub fn side_table_count(&self) -> usize {
        self.side_tables.len()
    }

    /// The canonical state hash (§4.2.1) — the contract, and the only hash any
    /// CI gate reads.
    ///
    /// Walks **sorted by entity id**, not in storage order, which is what makes
    /// it independent of archetype layout: internal storage rework can demand
    /// identical hashes across a rewrite, and an iteration-order change that
    /// genuinely does not affect results correctly does not fail CI. An order
    /// change that *does* affect results still fails, because it changes state.
    ///
    /// Cost is a full pass. The chunked accelerator (§4.2.1, designed in
    /// [`crate::chunked`]) is what makes that cheap at scale; it is deliberately
    /// not built until demo 05-many proves the need.
    #[must_use]
    pub fn canonical_hash(&self) -> CanonicalHash {
        self.hash_walk(Encoding::RawFastPath)
    }

    /// The same hash computed through the **explicit protocol** — the
    /// definition that [`canonical_hash`](Self::canonical_hash) is an
    /// optimization of (§4.2.1).
    ///
    /// Components are `Pod` and `repr(C)`, and every supported target is
    /// little-endian, so the protocol's field-by-field encoding and the raw row
    /// bytes coincide. That is a *derivation*, not an axiom, which is why both
    /// exist and why CI compares them: the day a component acquires a field
    /// whose protocol encoding differs from its memory image, the fast path
    /// must stop being valid loudly rather than silently *become* the
    /// definition.
    #[must_use]
    pub fn canonical_hash_via_protocol(&self) -> CanonicalHash {
        self.hash_walk(Encoding::Protocol)
    }

    /// Per-entity canonical hashes, ascending by entity id.
    ///
    /// The divergence probe under [`canonical_hash`](Self::canonical_hash):
    /// when two runs of one replay disagree at some tick, "which entity" is the
    /// difference between a bug report and a wasted day (§5.6). Two runs' lists
    /// are compared elementwise; the first pair that differs — or the first id
    /// present in one list and not the other — is the answer.
    ///
    /// Not part of the whole-world hash and not a substitute for it: this walk
    /// omits the registry, the entity allocator and side tables, so a world can
    /// have identical entity hashes and a different canonical hash. That is
    /// deliberate — it is a *locator*, and CI still gates on the contract.
    #[must_use]
    pub fn entity_hashes(&self) -> Vec<(Entity, CanonicalHash)> {
        let mut out = Vec::with_capacity(self.entities.len() as usize);
        // One buffer across every entity's digest, not one per hasher: the
        // absorbs are per-entity but the allocation need not be (§6 M38 item 9).
        let mut block = Block::new();
        for entity in self.entities.iter() {
            let mut h = StateHasher::canonical();
            block.u64(&mut h, entity.to_bits());
            if let Some(loc) = self.location(entity) {
                let archetype = &self.archetypes[loc.archetype.index() as usize];
                block.u64(&mut h, archetype.ids().len() as u64);
                for (at, id) in archetype.ids().iter().enumerate() {
                    block.u64(&mut h, id.get());
                    block.bytes(&mut h, archetype.column(at).row(loc.row as usize));
                }
            } else {
                block.u64(&mut h, u64::MAX);
            }
            block.flush(&mut h);
            out.push((entity, h.finish_canonical()));
        }
        out
    }

    /// The first NaN in any hashed column, or `None` (§1.13 hazard 6).
    ///
    /// One hit per call on purpose: a NaN propagates to thousands of entities
    /// within a tick, and sixty thousand log lines a second is not a diagnostic.
    /// Which one is *first* is a function of the world, not of the search — see
    /// [`nan_scan`](crate::nan_scan) for the order and for how the float
    /// positions are derived.
    ///
    /// Columns first, then side tables — every hashed float, by two mechanisms
    /// because the two are shaped differently. A column has a declared layout
    /// and is walked; a side table has none and is made to enumerate itself
    /// through [`StateHasher`], the funnel its floats must pass anyway. Columns
    /// first because their answer is the precise one.
    #[must_use]
    pub fn scan_for_nan(&self) -> Option<crate::nan_scan::NanSite> {
        use crate::nan_scan::NanSite;
        crate::nan_scan::scan(&self.archetypes, &self.registry)
            .map(NanSite::Column)
            .or_else(|| self.side_tables.scan_for_nan().map(NanSite::SideTable))
    }

    fn hash_walk(&self, encoding: Encoding) -> CanonicalHash {
        let mut h = StateHasher::canonical();
        self.entities.hash_into(&mut h);
        self.registry.hash_into(&mut h);

        // Staged for the fast path, passthrough for the protocol one — so the
        // two walks differ in staging as well as in encoding and §4.2.1's proof
        // covers both (§6 M38 item 9).
        let mut block = match encoding {
            Encoding::RawFastPath => Block::new(),
            Encoding::Protocol => Block::unstaged(),
        };
        block.u64(&mut h, u64::from(self.entities.len()));
        // Ascending entity id (§4.2.1) — never storage order, which is what
        // keeps archetype layout out of the contract.
        for entity in self.entities.iter() {
            block.u64(&mut h, entity.to_bits());
            let Some(loc) = self.location(entity) else {
                // Allocated but unplaced cannot happen; hash a discriminant
                // rather than panic, so a bug surfaces as a CI divergence
                // instead of as a crash in a player's build.
                block.u64(&mut h, u64::MAX);
                continue;
            };
            let archetype = &self.archetypes[loc.archetype.index() as usize];
            // The component *set*, never the archetype index — archetype
            // numbering is creation order, which is storage layout.
            block.u64(&mut h, archetype.ids().len() as u64);
            for (at, id) in archetype.ids().iter().enumerate() {
                block.u64(&mut h, id.get());
                let bytes = archetype.column(at).row(loc.row as usize);
                match encoding {
                    Encoding::RawFastPath => block.bytes(&mut h, bytes),
                    Encoding::Protocol => match self.registry.get(*id) {
                        // Writes through the hasher directly, so nothing may be
                        // staged behind it. `unstaged` guarantees that here;
                        // the flush is what would make it true if it were not.
                        Some(info) => {
                            block.flush(&mut h);
                            (info.protocol_hash)(bytes, &mut h);
                        }
                        // Unregistered is impossible — a column exists only
                        // because registration created it — but the walk stays
                        // total rather than panicking mid-hash.
                        None => block.bytes(&mut h, bytes),
                    },
                }
            }
        }
        block.flush(&mut h);

        // Side tables last, ascending by id (§4.2.1). They are protocol-encoded
        // in both walks — being non-`Pod` is the point — so they cannot make the
        // two encodings disagree, and their presence in the hash is what stops
        // state migrating out of the columns from silently leaving the contract.
        self.side_tables.hash_into(&mut h);
        h.finish_canonical()
    }

    /// Visit every archetype holding all of `access`'s components, in archetype
    /// creation order — §4.2 guarantee 1's stable iteration order.
    ///
    /// One call per archetype match, each yielding whole columns: the shape
    /// §4.2.2's boundary uses so a query never pays FFI per entity. The borrow
    /// check already happened, when `access` was constructed.
    pub fn views<'q>(&mut self, access: &'q QueryAccess, mut f: impl FnMut(ArchetypeView<'_, 'q>)) {
        for archetype in &mut self.archetypes {
            if archetype.is_empty() {
                continue;
            }
            if let Some(view) = crate::view::build(archetype, access) {
                f(view);
            }
        }
    }

    /// [`views`](Self::views) over `&self`, yielding views that outlive the
    /// call.
    ///
    /// The lifetime is the point: the columns come back as `&'w` slices tied to
    /// the world borrow rather than to the callback, so a caller can *collect*
    /// them and work over them afterwards. That is what lets the extract stage
    /// (§4.1) split one archetype into row ranges and narrow them in parallel —
    /// per-archetype parallelism would be no parallelism at all when ten
    /// thousand entities share a component set, which is the normal case.
    ///
    /// No [`ReadOnly`] bound because there is no `D` here; `access` carrying a
    /// write is what would be unsound, and `QueryAccess` is only constructible
    /// from a [`Query`], which checks it.
    pub fn views_ref<'w, 'q>(
        &'w self,
        access: &'q QueryAccess,
        mut f: impl FnMut(ArchetypeView<'w, 'q>),
    ) {
        for archetype in &self.archetypes {
            if archetype.is_empty() {
                continue;
            }
            if let Some(view) = crate::view::build_ref(archetype, access) {
                f(view);
            }
        }
    }

    /// Visit every matching entity as a typed tuple of borrows (§4.2).
    ///
    /// The ergonomic face of [`views`](Self::views), and the same cost model:
    /// columns are resolved once per archetype, then indexed. Iteration order is
    /// archetype creation order, then dense row — guarantee 1, not an accident
    /// of layout.
    pub fn each<D: QueryData>(&mut self, query: &Query<D>, mut f: impl FnMut(Entity, D::Item<'_>)) {
        self.views(query.access(), |mut view| {
            let entities = view.entities();
            let mut columns = D::columns(&mut view);
            for (row, &entity) in entities.iter().enumerate() {
                f(entity, D::get(&mut columns, row));
            }
        });
    }

    /// [`each`](Self::each) over `&self`, for queries that write nothing.
    ///
    /// The extract stage takes `&World` by contract (§4.1): "there is no type
    /// through which a render result could travel back into the sim" is meant
    /// to be checked, and `&mut World` would leave it as a habit. The
    /// [`ReadOnly`] bound is what makes the shared column views sound — a
    /// `&mut T` query does not satisfy it.
    ///
    /// Same iteration order as [`each`](Self::each), and the same cost model.
    pub fn each_ref<D: QueryData + ReadOnly>(
        &self,
        query: &Query<D>,
        mut f: impl FnMut(Entity, D::Item<'_>),
    ) {
        for archetype in &self.archetypes {
            if archetype.is_empty() {
                continue;
            }
            let Some(mut view) = crate::view::build_ref(archetype, query.access()) else {
                continue;
            };
            let entities = view.entities();
            let mut columns = D::columns(&mut view);
            for (row, &entity) in entities.iter().enumerate() {
                f(entity, D::get(&mut columns, row));
            }
        }
    }

    // ---- internals ------------------------------------------------------

    fn location(&self, entity: Entity) -> Option<Location> {
        if !self.entities.is_alive(entity) {
            return None;
        }
        self.locations
            .get(entity.index() as usize)
            .copied()
            .flatten()
    }

    fn set_location(&mut self, entity: Entity, loc: Option<Location>) {
        let at = entity.index() as usize;
        if at >= self.locations.len() {
            self.locations.resize(at + 1, None);
        }
        self.locations[at] = loc;
    }

    /// Drop a row, repairing the location of whichever entity swap-remove moved
    /// into the hole.
    fn remove_row(&mut self, loc: Location) {
        let moved = self.archetypes[loc.archetype.index() as usize].swap_remove(loc.row);
        if let Some(moved) = moved {
            self.set_location(moved, Some(loc));
        }
    }

    /// Move an entity's row to `target`, carrying every component both
    /// archetypes share. Returns the new row.
    ///
    /// Shared columns are copied through [`World::scratch`] rather than by
    /// borrowing two archetypes at once. Structural change is not the hot path
    /// — queries are — and one reused buffer keeps this function obviously
    /// correct without a disjoint-borrow dance.
    fn move_row(&mut self, entity: Entity, loc: Location, target: ArchetypeId) -> u32 {
        debug_assert_ne!(
            loc.archetype, target,
            "move_row called with no move to make"
        );

        self.scratch.clear();
        let source = &self.archetypes[loc.archetype.index() as usize];
        // (component, offset in scratch, length) for every column we carry.
        let mut carried: Vec<(ComponentId, usize, usize)> = Vec::with_capacity(source.ids().len());
        for (at, &id) in source.ids().iter().enumerate() {
            let bytes = source.column(at).row(loc.row as usize);
            carried.push((id, self.scratch.len(), bytes.len()));
            self.scratch.extend_from_slice(bytes);
        }

        let new_row = self.archetypes[target.index() as usize].push_zeroed(entity);
        for (id, offset, len) in carried {
            let Some(at) = self.archetypes[target.index() as usize].column_index(id) else {
                continue; // dropped by a `remove`
            };
            let bytes = self.scratch[offset..offset + len].to_vec();
            self.archetypes[target.index() as usize]
                .column_mut(at)
                .row_mut(new_row as usize)
                .copy_from_slice(&bytes);
        }

        self.remove_row(loc);
        self.set_location(
            entity,
            Some(Location {
                archetype: target,
                row: new_row,
            }),
        );
        new_row
    }

    /// The archetype for `from`'s components plus `id`, via the graph edge if
    /// it has been walked before.
    fn archetype_with(&mut self, from: ArchetypeId, id: ComponentId) -> ArchetypeId {
        if let Some(to) = self.archetypes[from.index() as usize].add_edge(id) {
            return to;
        }
        let mut ids = self.archetypes[from.index() as usize].ids().to_vec();
        if let Err(at) = ids.binary_search(&id) {
            ids.insert(at, id);
        }
        let to = self.archetype_for(ids);
        self.archetypes[from.index() as usize].set_add_edge(id, to);
        self.archetypes[to.index() as usize].set_remove_edge(id, from);
        to
    }

    /// The archetype for `from`'s components minus `id`.
    fn archetype_without(&mut self, from: ArchetypeId, id: ComponentId) -> ArchetypeId {
        if let Some(to) = self.archetypes[from.index() as usize].remove_edge(id) {
            return to;
        }
        let mut ids = self.archetypes[from.index() as usize].ids().to_vec();
        if let Ok(at) = ids.binary_search(&id) {
            ids.remove(at);
        }
        let to = self.archetype_for(ids);
        self.archetypes[from.index() as usize].set_remove_edge(id, to);
        self.archetypes[to.index() as usize].set_add_edge(id, from);
        to
    }

    /// Look up or create the archetype for an exact, sorted component set.
    fn archetype_for(&mut self, ids: Vec<ComponentId>) -> ArchetypeId {
        match self
            .by_ids
            .binary_search_by(|(set, _)| set.as_slice().cmp(&ids))
        {
            Ok(at) => self.by_ids[at].1,
            Err(at) => {
                let strides: Vec<usize> = ids
                    .iter()
                    .map(|id| self.registry.get(*id).map_or(0, |info| info.size))
                    .collect();
                let new = ArchetypeId(self.archetypes.len() as u32);
                self.archetypes.push(Archetype::new(ids.clone(), &strides));
                self.by_ids.insert(at, (ids, new));
                new
            }
        }
    }
}
