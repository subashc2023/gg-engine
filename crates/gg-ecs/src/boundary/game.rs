//! The guest half: what game code actually writes against (§4.2.2).
//!
//! §4.2.2 promises the ergonomics back — `fn movement(world: &mut GameWorld)`,
//! never a raw pointer — while the seam underneath stays a `repr(C)` table. That
//! is this file: [`GameWorld`] wraps the opaque handle and the host table,
//! [`BoundaryData`] resolves a tuple of `&T`/`&mut T` against one archetype
//! match, and iteration happens *inside* the dylib over whole columns, so a
//! query costs one FFI call per page of archetypes rather than one per entity.

use gg_abi::{
    AbiStatus, ActionId, ArchetypeMatch, AxisId, Binding, ColumnView, Entity, HostApiV1,
    InputFrame, MAX_QUERY_COLUMNS, QueryDesc, TickCtx, WorldHandle,
};

use crate::component::Component;

/// Archetype matches fetched per FFI call.
///
/// The page is a stack array in [`GameWorld::each`], so this trades ~3 KB of
/// stack against a call per archetype. Eight covers the archetype count of any
/// query a single system writes; more only costs another round of the loop.
const PAGE: usize = 8;

/// A host call that did not do what was asked.
///
/// Carries the operation because [`AbiStatus`] alone reads the same for a query
/// and an insert, and the two have entirely different remedies.
#[derive(Debug, thiserror::Error)]
#[error("the host refused `{op}`: {status:?}")]
pub struct BoundaryError {
    /// Which host operation was refused.
    pub op: &'static str,
    /// Why.
    pub status: AbiStatus,
}

/// The world, from inside a game dylib.
///
/// Every method is a call through the host table; nothing here holds ECS state,
/// because the world is host-owned and a system owns nothing that outlives a
/// tick (§4.2.2). Borrows returned by [`get`](Self::get) and
/// [`get_mut`](Self::get_mut) are tied to this value, which is what makes the
/// "invalidated by the next host call" rule a borrow-check error rather than a
/// footnote.
pub struct GameWorld<'a> {
    handle: *mut WorldHandle,
    api: &'static HostApiV1,
    ctx: &'a TickCtx,
}

impl<'a> GameWorld<'a> {
    /// Wrap the handle and context the host passed to a systems-table entry.
    ///
    /// Game code never calls this — [`gg_game!`](crate::gg_game)'s generated
    /// shims do, once per system per tick.
    ///
    /// # Safety
    ///
    /// `handle` must be the handle the host built for this tick, `ctx` must
    /// outlive the returned value, and `gg_game_init` must already have run.
    ///
    /// # Panics
    ///
    /// If `gg_game_init` has not run: no game code can do anything without the
    /// host table, so the shim converting this into a named status is the whole
    /// report.
    #[must_use]
    pub unsafe fn new(handle: *mut WorldHandle, ctx: &'a TickCtx) -> Self {
        Self {
            handle,
            api: gg_abi::host::get(),
            ctx,
        }
    }

    /// This tick's number. Sim time is a count, never a float (§2).
    #[must_use]
    pub const fn tick(&self) -> u64 {
        self.ctx.tick
    }

    /// The fixed timestep, as a rate.
    #[must_use]
    pub const fn tick_hz(&self) -> u32 {
        self.ctx.tick_hz
    }

    /// Seconds per tick, computed rather than stored — a float *field* on
    /// sim-visible state is banned (§2), a value derived identically everywhere
    /// is not.
    #[must_use]
    pub fn dt(&self) -> f32 {
        1.0 / self.ctx.tick_hz as f32
    }

    /// This tick's input.
    #[must_use]
    pub const fn input(&self) -> &InputFrame {
        &self.ctx.input
    }

    /// Last tick's, for edges.
    #[must_use]
    pub const fn previous(&self) -> &InputFrame {
        &self.ctx.previous
    }

    /// Is the action down this tick?
    #[must_use]
    pub const fn pressed(&self, action: ActionId) -> bool {
        self.ctx.input.pressed(action)
    }

    /// Did the action go down *this* tick? Answered from the two frames the
    /// context carries, so a system needs no state of its own (§4.2.2).
    #[must_use]
    pub const fn just_pressed(&self, action: ActionId) -> bool {
        self.ctx.input.pressed(action) && !self.ctx.previous.pressed(action)
    }

    /// An axis in units. Exact — axes are fixed-point (§4.7).
    #[must_use]
    pub fn axis(&self, axis: AxisId) -> f32 {
        self.ctx.input.axis(axis)
    }

    /// What `action` is bound to, spelled as the map spells it, in the map's own
    /// order (§6 M45) — `["A", "Left"]` for a verb bound twice.
    ///
    /// Empty when the session was given no map, which every windowed run has one
    /// of and a bare `--frames` run does not. A legend built from this is a
    /// legend that cannot drift from `input.toml`, which is the whole point:
    /// before M45 a game that wanted to tell the player its keys retyped them.
    pub fn bindings(&self, action: ActionId) -> impl Iterator<Item = &str> {
        self.binding_table()
            .iter()
            .filter(move |b| b.action as usize == action.index())
            .map(Binding::name)
    }

    /// The whole table, host-owned and valid for this call.
    fn binding_table(&self) -> &[Binding] {
        if self.ctx.bindings.is_null() {
            return &[];
        }
        // SAFETY: the host's contract for `TickCtx::bindings` — non-null with a
        // nonzero length, `bindings_len` initialized entries, live for the call.
        // The `&self` borrow ties the slice to the context, which cannot outlive
        // the tick it was handed for.
        unsafe { core::slice::from_raw_parts(self.ctx.bindings, self.ctx.bindings_len as usize) }
    }

    /// Emit a line into the host's log stream.
    pub fn log(&self, level: u32, message: &str) {
        // SAFETY: `log` takes a pointer and a length into memory this call owns
        // and the host only reads during the call.
        unsafe { (self.api.log)(level, message.as_ptr(), message.len() as u32) }
    }

    /// Create an entity with no components.
    pub fn spawn(&mut self) -> Entity {
        // SAFETY: `handle` is the host's, valid for this tick (`new`'s contract).
        unsafe { (self.api.spawn)(self.handle) }
    }

    /// Destroy an entity. `false` if it was already dead.
    pub fn despawn(&mut self, entity: Entity) -> bool {
        // SAFETY: as `spawn`.
        unsafe { (self.api.despawn)(self.handle, entity) }
    }

    /// Is this handle live? A stale generation reads as dead.
    #[must_use]
    pub fn is_alive(&self, entity: Entity) -> bool {
        // SAFETY: as `spawn`, shared.
        unsafe { (self.api.is_alive)(self.handle.cast_const(), entity) }
    }

    /// Add or overwrite a component.
    ///
    /// # Errors
    ///
    /// If the entity is dead, or the host has no component registered under
    /// `T`'s id — which means this build declared a component the host was never
    /// told about, a load-sequence bug rather than a gameplay condition.
    pub fn insert<T: Component>(&mut self, entity: Entity, value: T) -> Result<(), BoundaryError> {
        let bytes = bytemuck::bytes_of(&value);
        // SAFETY: as `spawn`; `bytes` is `T`'s image, read only for this call.
        let status = unsafe {
            (self.api.insert)(
                self.handle,
                entity,
                crate::component_id::<T>().get(),
                bytes.as_ptr(),
                bytes.len() as u32,
            )
        };
        match status {
            AbiStatus::Ok => Ok(()),
            status => Err(BoundaryError {
                op: "insert",
                status,
            }),
        }
    }

    /// [`insert`](Self::insert), logging a refusal instead of returning it.
    ///
    /// Every game crate wrote this by hand — five identical `fn report`s, and
    /// the M12 template criterion is what counted them (§6 M12). There is
    /// nothing a system can *do* with a `BoundaryError`: it means this build
    /// declared a component the host was never told about, so the only correct
    /// response is to say so loudly and carry on. [`insert`](Self::insert)
    /// stays for the caller that genuinely wants to branch.
    pub fn put<T: Component>(&mut self, entity: Entity, value: T) {
        if let Err(refused) = self.insert(entity, value) {
            self.log(crate::boundary::log_level::ERROR, &refused.to_string());
        }
    }

    /// Spawn one entity carrying `value`, as [`put`](Self::put) reports.
    pub fn spawn_with<T: Component>(&mut self, value: T) -> Entity {
        let entity = self.spawn();
        self.put(entity, value);
        entity
    }

    /// Drop a component. `false` if the entity did not have it.
    pub fn remove<T: Component>(&mut self, entity: Entity) -> bool {
        // SAFETY: as `spawn`.
        unsafe { (self.api.remove)(self.handle, entity, crate::component_id::<T>().get()) }
    }

    /// One entity's component, or `None` if absent or dead.
    #[must_use]
    pub fn get<T: Component>(&self, entity: Entity) -> Option<&T> {
        let ptr = self.raw(entity, crate::component_id::<T>().get())?;
        // SAFETY: the host returns either null or a pointer to one row of `T`'s
        // column, valid until the next host call — which the `&self` borrow on
        // the returned reference is exactly what prevents.
        Some(unsafe { &*ptr.cast::<T>() })
    }

    /// One entity's component, mutably.
    pub fn get_mut<T: Component>(&mut self, entity: Entity) -> Option<&mut T> {
        let ptr = self.raw(entity, crate::component_id::<T>().get())?;
        // SAFETY: as `get`, and the `&mut self` borrow rules out any other live
        // reference into the world for the returned reference's lifetime.
        Some(unsafe { &mut *ptr.cast::<T>() })
    }

    fn raw(&self, entity: Entity, component: u64) -> Option<*mut u8> {
        // SAFETY: as `spawn`. `get` takes `*mut` because the host cannot know
        // which way the caller will use it; the returned `&`/`&mut` is what
        // decides, and both are produced under a borrow of `self`.
        let ptr = unsafe { (self.api.get)(self.handle, entity, component) };
        (!ptr.is_null()).then_some(ptr)
    }

    /// Visit every entity holding all of `D`'s components.
    ///
    /// One FFI call per [`PAGE`] of archetypes; the per-entity loop runs inside
    /// the dylib over whole columns (§4.2.2). Iteration order is the host's
    /// archetype order then dense row — §4.2 guarantee 1, reaching across the
    /// boundary unchanged.
    ///
    /// The visitor is handed column borrows and never the world, so a structural
    /// change cannot happen mid-iteration and invalidate the pointers under it.
    ///
    /// # Errors
    ///
    /// If `D` borrows one component mutably twice, or both mutably and
    /// immutably — checked host-side before any pointer is produced.
    ///
    /// # Panics
    ///
    /// If a matched column's stride disagrees with its component's size. That
    /// means the host registered a differently-sized type under this id, which
    /// the version and fingerprint checks exist to prevent; the shim turns it
    /// into a named status rather than a reinterpretation of live bytes.
    pub fn each<D: BoundaryData>(
        &mut self,
        mut visit: impl FnMut(Entity, D::Item<'_>),
    ) -> Result<(), BoundaryError> {
        let mut access = Access::new();
        D::access(&mut access);
        let desc = access.desc();
        let mut page = [ArchetypeMatch::EMPTY; PAGE];
        let mut skip = 0u32;

        loop {
            let mut written = 0u32;
            // SAFETY: `desc` borrows `access`, which outlives this call; `page`
            // is `PAGE` initialized matches the host may overwrite; `written` is
            // one live `u32`.
            let status = unsafe {
                (self.api.query)(
                    self.handle,
                    &raw const desc,
                    skip,
                    page.as_mut_ptr(),
                    PAGE as u32,
                    &raw mut written,
                )
            };
            if status != AbiStatus::Ok {
                return Err(BoundaryError {
                    op: "query",
                    status,
                });
            }

            for matched in &page[..written as usize] {
                let mut cursor = Columns::new(matched, access.read_len);
                // SAFETY: `matched` was produced for `desc`, which `D::access`
                // built — so the columns are `D`'s, in `D`'s declaration order.
                let mut columns = unsafe { D::columns(&mut cursor) };
                // SAFETY: the host writes `entities`/`len` as one live slice of
                // the matched archetype's row→entity map.
                let entities =
                    unsafe { core::slice::from_raw_parts(matched.entities, matched.len) };
                for (row, &entity) in entities.iter().enumerate() {
                    visit(entity, D::get(&mut columns, row));
                }
            }

            if (written as usize) < PAGE {
                return Ok(());
            }
            skip += written;
        }
    }

    /// [`each`](Self::each), logging a refusal instead of returning it — the
    /// [`put`](Self::put)/[`insert`](Self::insert) pair, for queries.
    ///
    /// Six game crates carried the same four-line `fn report` shim over `each`
    /// because there is nothing a system can do with the error: a refused query
    /// names a component this build declared and the host never registered, so
    /// the only correct response is to say so and carry on. [`each`](Self::each)
    /// stays for the caller that genuinely wants to branch.
    pub fn visit<D: BoundaryData>(&mut self, body: impl FnMut(Entity, D::Item<'_>)) {
        if let Err(refused) = self.each::<D>(body) {
            self.log(crate::boundary::log_level::ERROR, &refused.to_string());
        }
    }
}

impl core::fmt::Debug for GameWorld<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("GameWorld")
            .field("tick", &self.ctx.tick)
            .finish_non_exhaustive()
    }
}

/// The component ids one query names, in declaration order.
///
/// Fixed arrays rather than `Vec`: this is built once per `each` call on the
/// tick path, and the boundary already caps a query at
/// [`MAX_QUERY_COLUMNS`](gg_abi::MAX_QUERY_COLUMNS) columns.
pub struct Access {
    read: [u64; MAX_QUERY_COLUMNS],
    write: [u64; MAX_QUERY_COLUMNS],
    read_len: usize,
    write_len: usize,
}

impl Access {
    /// An empty access set.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            read: [0; MAX_QUERY_COLUMNS],
            write: [0; MAX_QUERY_COLUMNS],
            read_len: 0,
            write_len: 0,
        }
    }

    /// Name a component this query reads.
    ///
    /// # Panics
    ///
    /// Past [`MAX_QUERY_COLUMNS`](gg_abi::MAX_QUERY_COLUMNS) reads.
    pub const fn read(&mut self, id: u64) {
        assert!(
            self.read_len < MAX_QUERY_COLUMNS,
            "a query may name at most MAX_QUERY_COLUMNS components"
        );
        self.read[self.read_len] = id;
        self.read_len += 1;
    }

    /// Name a component this query writes.
    ///
    /// # Panics
    ///
    /// Past [`MAX_QUERY_COLUMNS`](gg_abi::MAX_QUERY_COLUMNS) writes.
    pub const fn write(&mut self, id: u64) {
        assert!(
            self.write_len < MAX_QUERY_COLUMNS,
            "a query may name at most MAX_QUERY_COLUMNS components"
        );
        self.write[self.write_len] = id;
        self.write_len += 1;
    }

    fn desc(&self) -> QueryDesc {
        QueryDesc {
            read: self.read.as_ptr(),
            write: self.write.as_ptr(),
            read_len: self.read_len as u32,
            write_len: self.write_len as u32,
        }
    }
}

impl Default for Access {
    fn default() -> Self {
        Self::new()
    }
}

/// Walks one match's columns in the order [`BoundaryData::access`] named them.
///
/// The match lays reads out first and writes after, so a `&T` counts off the
/// front and a `&mut T` counts off from `read_len`. Two counters, no lookup:
/// declaration order is the only index either side needs.
pub struct Columns<'m> {
    matched: &'m ArchetypeMatch,
    read: usize,
    write: usize,
    read_len: usize,
}

impl<'m> Columns<'m> {
    fn new(matched: &'m ArchetypeMatch, read_len: usize) -> Self {
        Self {
            matched,
            read: 0,
            write: read_len,
            read_len,
        }
    }

    /// The next read column.
    fn next_read(&mut self) -> ColumnView {
        assert!(self.read < self.read_len, "read columns exhausted");
        let view = self.matched.columns[self.read];
        self.read += 1;
        view
    }

    /// The next write column.
    fn next_write(&mut self) -> ColumnView {
        assert!(
            self.write < MAX_QUERY_COLUMNS,
            "write columns exhausted — more than MAX_QUERY_COLUMNS in one query"
        );
        let view = self.matched.columns[self.write];
        self.write += 1;
        view
    }

    /// Rows in this match.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.matched.len
    }

    /// Whether this match holds no rows.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.matched.len == 0
    }
}

/// What one query row yields on the game side of the boundary.
///
/// Implemented for `&T`, `&mut T`, and tuples of those — the same shape as
/// [`QueryData`](crate::QueryData), against the `repr(C)` payload instead of a
/// host-side view. Not meant to be implemented downstream: the column
/// bookkeeping rests on [`access`](Self::access) naming exactly what
/// [`columns`](Self::columns) later takes, in the same order.
pub trait BoundaryData {
    /// One row's worth of borrows.
    type Item<'r>;
    /// One archetype's worth, resolved once per match.
    type Columns<'w>;

    /// Name the components this data borrows, in declaration order.
    fn access(access: &mut Access);

    /// Resolve against one matched archetype.
    ///
    /// # Safety
    ///
    /// `cursor`'s match must have been produced for a query [`access`] built,
    /// and the columns must not already have been resolved for this match.
    ///
    /// [`access`]: Self::access
    unsafe fn columns<'w>(cursor: &mut Columns<'_>) -> Self::Columns<'w>;

    /// Borrow row `row`. Callers pass rows in `0..len`.
    fn get<'r, 'w: 'r>(columns: &'r mut Self::Columns<'w>, row: usize) -> Self::Item<'r>;
}

/// # Safety
///
/// `view` must describe a live column of `T`s, valid until the next host call.
unsafe fn slice<'w, T: Component>(view: ColumnView) -> &'w [T] {
    assert_stride::<T>(view);
    // SAFETY: the caller's contract. A zero-sized `T` takes the dangling
    // pointer, which a ZST slice must use rather than a real address.
    unsafe {
        if size_of::<T>() == 0 {
            core::slice::from_raw_parts(core::ptr::NonNull::<T>::dangling().as_ptr(), view.len)
        } else {
            core::slice::from_raw_parts(view.ptr.cast::<T>(), view.len)
        }
    }
}

/// # Safety
///
/// As [`slice`], and the column must not be borrowed anywhere else — which the
/// host proved when it built the match (§4.2.2: the borrow check happens before
/// any pointer is handed over).
unsafe fn slice_mut<'w, T: Component>(view: ColumnView) -> &'w mut [T] {
    assert_stride::<T>(view);
    // SAFETY: the caller's contract, as `slice`.
    unsafe {
        if size_of::<T>() == 0 {
            core::slice::from_raw_parts_mut(core::ptr::NonNull::<T>::dangling().as_ptr(), view.len)
        } else {
            core::slice::from_raw_parts_mut(view.ptr.cast::<T>(), view.len)
        }
    }
}

fn assert_stride<T: Component>(view: ColumnView) {
    assert_eq!(
        view.stride,
        size_of::<T>(),
        "gg-ecs: the host's column for `{}` is {} bytes per row but this build's type is {} — the \
         two sides disagree about a component's layout behind a matching ABI version",
        T::TYPE_NAME,
        view.stride,
        size_of::<T>()
    );
}

impl<T: Component> BoundaryData for &T {
    type Item<'r> = &'r T;
    type Columns<'w> = &'w [T];

    fn access(access: &mut Access) {
        access.read(crate::component_id::<T>().get());
    }

    unsafe fn columns<'w>(cursor: &mut Columns<'_>) -> Self::Columns<'w> {
        // SAFETY: the caller's contract — this cursor's next read column is
        // `T`'s, because `access` put it there.
        unsafe { slice(cursor.next_read()) }
    }

    fn get<'r, 'w: 'r>(columns: &'r mut Self::Columns<'w>, row: usize) -> Self::Item<'r> {
        &columns[row]
    }
}

impl<T: Component> BoundaryData for &mut T {
    type Item<'r> = &'r mut T;
    type Columns<'w> = &'w mut [T];

    fn access(access: &mut Access) {
        access.write(crate::component_id::<T>().get());
    }

    unsafe fn columns<'w>(cursor: &mut Columns<'_>) -> Self::Columns<'w> {
        // SAFETY: as the shared impl, plus: the host rejected a component named
        // mutably twice with `AbiStatus::Alias` before producing this match, so
        // no other slice in this resolution names these bytes.
        unsafe { slice_mut(cursor.next_write()) }
    }

    fn get<'r, 'w: 'r>(columns: &'r mut Self::Columns<'w>, row: usize) -> Self::Item<'r> {
        &mut columns[row]
    }
}

macro_rules! tuple_boundary_data {
    ($($name:ident),+) => {
        impl<$($name: BoundaryData),+> BoundaryData for ($($name,)+) {
            type Item<'r> = ($($name::Item<'r>,)+);
            type Columns<'w> = ($($name::Columns<'w>,)+);

            fn access(access: &mut Access) {
                $($name::access(access);)+
            }

            unsafe fn columns<'w>(cursor: &mut Columns<'_>) -> Self::Columns<'w> {
                // SAFETY: forwarded per element. Evaluation order of a tuple
                // expression is left to right, which is what keeps the cursor in
                // step with the order `access` above named the columns in.
                unsafe { ($($name::columns(cursor),)+) }
            }

            fn get<'r, 'w: 'r>(columns: &'r mut Self::Columns<'w>, row: usize) -> Self::Item<'r> {
                // Type and binding names coincide; different namespaces.
                #[allow(non_snake_case)]
                let ($($name,)+) = columns;
                ($($name::get($name, row),)+)
            }
        }
    };
}

tuple_boundary_data!(A);
tuple_boundary_data!(A, B);
tuple_boundary_data!(A, B, C);
tuple_boundary_data!(A, B, C, D);
tuple_boundary_data!(A, B, C, D, E);
tuple_boundary_data!(A, B, C, D, E, F);
tuple_boundary_data!(A, B, C, D, E, F, G);
tuple_boundary_data!(A, B, C, D, E, F, G, H);
