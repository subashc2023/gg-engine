//! Column views: borrow-checked, `repr(C)`-shaped access to whole columns
//! (§4.2, §4.2.2).
//!
//! This is both the ECS's own query substrate and the **prototype of the reload
//! boundary's hot path**. §4.2.2 promises that a query crossing into a game
//! dylib does not pay FFI per entity: one call per archetype match yields raw
//! column pointers, lengths and strides, and the dylib then iterates natively
//! at full speed. [`ColumnView`] is exactly that payload, and it is `repr(C)`
//! for the same reason.
//!
//! It is deliberately **not** the M5 API. M5 builds the versioned `extern "C"`
//! surface; M3 proves the shape is cheap, two milestones before M5 stakes its
//! budget on the claim — because the cheapest moment to learn the storage
//! design makes it awkward is while the internals are still wet.
//!
//! # Where the borrow check happens
//!
//! At **view construction**, before any pointer is handed over — the same
//! ordering §4.2.2 specifies for the real boundary. [`QueryAccess::new`]
//! rejects a component requested mutably twice, or both mutably and immutably,
//! with a message naming the component. That makes aliasing a deterministic
//! panic-free error rather than UB, which is the M3 exit criterion.

use crate::archetype::Archetype;
use crate::component::Component;
use crate::entity::Entity;
use crate::hash::ComponentId;

/// One column of one archetype match: base pointer, row count, stride.
///
/// Defined in `gg-abi` and re-exported, because "crosses the §4.2.2 boundary
/// unchanged" is stronger stated as *there is only one of it*. M3 built this
/// shape here to prove it cheap; M5 moved the definition down to the crate whose
/// subject is layout, and nothing about the shape changed in the move.
pub use gg_abi::ColumnView;

/// A validated set of component accesses.
///
/// Construct once and reuse across ticks: validation is the only cost, and
/// paying it per frame would make the check feel expensive enough to skip.
#[derive(Clone, Debug)]
pub struct QueryAccess {
    /// Ascending, deduplicated.
    read: Vec<ComponentId>,
    /// Ascending, deduplicated, disjoint from `read`.
    write: Vec<ComponentId>,
    /// `read ∪ write`, ascending — the archetype match set.
    all: Vec<ComponentId>,
}

/// Columns one query may name, reads and writes together — the cap that lets
/// [`ArchetypeView`] hold them inline instead of allocating per archetype.
///
/// It *is* [`MAX_QUERY_COLUMNS`](gg_abi::MAX_QUERY_COLUMNS) rather than a
/// number that happens to match: the boundary's payload has been one flat
/// `repr(C)` array of this length since M5, and a host that allocated two
/// `Vec`s to copy into it was paying for storage the shape had already ruled
/// out. One number, so the two cannot drift.
pub const MAX_TERMS: usize = gg_abi::MAX_QUERY_COLUMNS;

// `taken` is one word of take-once bits, one per write column.
const _: () = assert!(MAX_TERMS <= u64::BITS as usize);

/// Filler for the slots of [`ArchetypeView::columns`] past `reads + writes`.
/// Never handed out — [`raw_reads`](ArchetypeView::raw_reads) and its siblings
/// cut the array to length first — and null so a slot that escaped that would
/// fault rather than read somewhere plausible.
pub(crate) const EMPTY_COLUMN: ColumnView = ColumnView {
    ptr: core::ptr::null_mut(),
    len: 0,
    stride: 0,
};

/// Why a set of accesses cannot be granted.
#[derive(Debug, thiserror::Error)]
pub enum AliasError {
    /// One component borrowed mutably twice.
    #[error(
        "component {id:?} is borrowed mutably more than once by this query. Two mutable borrows \
         of one component would alias; split the work or borrow it once."
    )]
    WriteWrite {
        /// The doubly-borrowed component.
        id: ComponentId,
    },
    /// One component borrowed both shared and mutably.
    #[error(
        "component {id:?} is borrowed both mutably and immutably by this query. Drop the shared \
         borrow, or read it through the mutable one."
    )]
    ReadWrite {
        /// The component on both sides of the borrow.
        id: ComponentId,
    },
    /// More columns than [`MAX_TERMS`], counted after duplicate reads collapse.
    /// The refusal is what makes [`ArchetypeView`]'s inline storage safe: past
    /// the cap there is nowhere to put the columns, so this is an error at
    /// validation rather than a truncation at build.
    #[error(
        "a query may name at most {MAX_TERMS} components across its reads and writes; this one \
         asks for {count}"
    )]
    TooManyTerms {
        /// How many were asked for, deduplicated.
        count: usize,
    },
}

impl QueryAccess {
    /// Validate an access set. This is the borrow check (§4.2.2): it happens
    /// here, before any raw pointer exists.
    pub fn new(read: &[ComponentId], write: &[ComponentId]) -> Result<Self, AliasError> {
        let mut w = write.to_vec();
        w.sort_unstable();
        if let Some(dup) = w.windows(2).find(|p| p[0] == p[1]) {
            return Err(AliasError::WriteWrite { id: dup[0] });
        }
        let mut r = read.to_vec();
        r.sort_unstable();
        r.dedup(); // reading twice is harmless, unlike writing twice
        if let Some(id) = r.iter().find(|id| w.binary_search(id).is_ok()) {
            return Err(AliasError::ReadWrite { id: *id });
        }
        // After the aliasing checks, so a query that is both over the cap and
        // self-aliasing is named by the more specific fault; after the dedup,
        // because the column count is what the view must hold, not what the
        // caller typed.
        if r.len() + w.len() > MAX_TERMS {
            return Err(AliasError::TooManyTerms {
                count: r.len() + w.len(),
            });
        }
        let mut all = r.clone();
        all.extend_from_slice(&w);
        all.sort_unstable();
        Ok(Self {
            read: r,
            write: w,
            all,
        })
    }

    /// Index of `id` within [`reads`](Self::reads), if this query reads it.
    pub(crate) fn read_index(&self, id: ComponentId) -> Option<usize> {
        self.read.binary_search(&id).ok()
    }

    /// Index of `id` within [`writes`](Self::writes), if this query writes it.
    pub(crate) fn write_index(&self, id: ComponentId) -> Option<usize> {
        self.write.binary_search(&id).ok()
    }

    /// Components this query borrows shared, sorted by id.
    #[must_use]
    pub fn reads(&self) -> &[ComponentId] {
        &self.read
    }

    /// Components this query borrows mutably, sorted by id.
    #[must_use]
    pub fn writes(&self) -> &[ComponentId] {
        &self.write
    }

    pub(crate) fn matched(&self) -> &[ComponentId] {
        &self.all
    }
}

/// One archetype's worth of columns, in the order the [`QueryAccess`] listed
/// them.
///
/// Every field is raw, including the entity slice. That is deliberate: the view
/// is produced by splitting one unique `&mut Archetype` into disjoint pieces,
/// and keeping a *shared* reference to the archetype alongside pointers later
/// used mutably would be exactly the aliasing pattern that makes such a split
/// unsound.
pub struct ArchetypeView<'w, 'q> {
    entities: *const Entity,
    entities_len: usize,
    /// Reads then writes, in access-set order — the flat shape
    /// [`ArchetypeMatch`](gg_abi::ArchetypeMatch) already hands the dylib.
    /// Inline, so building one archetype's view allocates nothing (§4.2.2).
    columns: [ColumnView; MAX_TERMS],
    /// `columns[..reads]`.
    reads: usize,
    /// `columns[reads..reads + writes]`.
    writes: usize,
    /// Which write columns have already been handed out. The returned slices
    /// live for `'w`, not for the `&mut self` that produced them, so nothing but
    /// this bit stops a second call from minting a second `&mut` to one column.
    taken: u64,
    /// Lets `*_of::<T>()` map a component to its index in the access set.
    access: &'q QueryAccess,
    /// Ties the raw pointers to the world borrow they came from.
    _lifetime: core::marker::PhantomData<&'w mut ()>,
}

impl<'w, 'q> ArchetypeView<'w, 'q> {
    /// Entities in this archetype, indexed by dense row (§4.2).
    #[must_use]
    pub fn entities(&self) -> &'w [Entity] {
        // SAFETY: `entities`/`entities_len` came from the archetype's live
        // entity `Vec` under the `&'w mut` borrow this view holds, and nothing
        // structurally changes the archetype while the view exists (the world
        // is mutably borrowed for `'w`). The entity slice is disjoint from
        // every column, so it never aliases a `write` slice.
        unsafe { core::slice::from_raw_parts(self.entities, self.entities_len) }
    }

    /// Rows in this archetype — the length every column view shares.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entities_len
    }

    /// Whether the archetype holds no rows. A matched-but-empty archetype is
    /// ordinary, not a bug.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities_len == 0
    }

    /// The raw views, as the §4.2.2 boundary would hand them over.
    ///
    /// Cut to length here rather than indexed in place, so every `[i]` below
    /// still panics past the *query's* column count instead of reading a
    /// [`EMPTY_COLUMN`] slot the cap left behind.
    #[must_use]
    pub fn raw_reads(&self) -> &[ColumnView] {
        &self.columns[..self.reads]
    }

    /// As [`Self::raw_reads`], for the mutable half. Handing one out does not
    /// consume it — [`Self::write`] is what tracks that.
    #[must_use]
    pub fn raw_writes(&self) -> &[ColumnView] {
        &self.columns[self.reads..self.reads + self.writes]
    }

    /// The `i`-th read column as a typed slice, where `i` indexes
    /// [`QueryAccess::reads`].
    ///
    /// # Panics
    ///
    /// If `i` is out of range, or `T` does not match the column's stride — the
    /// latter catches a caller that passed the right index with the wrong type,
    /// which is otherwise a silent reinterpretation.
    #[must_use]
    pub fn read<T: Component>(&self, i: usize) -> &'w [T] {
        let view = self.raw_reads()[i];
        Self::assert_shape::<T>(view, "read");
        // SAFETY: `view` was built from a live column of this archetype under
        // the `&mut World` borrow this view holds. `QueryAccess` proved `T`'s
        // component is not also borrowed mutably here, so no `&mut [T]` to the
        // same bytes exists. `assert_shape` checked the stride, and columns are
        // over-aligned past every legal component alignment. A zero-sized `T`
        // takes the dangling branch, since a ZST slice must not use a real
        // pointer it never reads.
        unsafe {
            if size_of::<T>() == 0 {
                core::slice::from_raw_parts(core::ptr::NonNull::<T>::dangling().as_ptr(), view.len)
            } else {
                core::slice::from_raw_parts(view.ptr.cast::<T>(), view.len)
            }
        }
    }

    /// The read column for `T`, wherever the access set put it.
    ///
    /// # Panics
    ///
    /// If this query does not read `T`.
    #[must_use]
    pub fn read_of<T: Component>(&self) -> &'w [T] {
        let id = crate::component_id::<T>();
        let i = self.access.read_index(id).unwrap_or_else(|| {
            panic!(
                "gg-ecs: query does not read `{}` — a read column can only be taken for a \
                 component the access set names",
                T::TYPE_NAME
            )
        });
        self.read(i)
    }

    /// The write column for `T`, wherever the access set put it.
    ///
    /// # Panics
    ///
    /// If this query does not write `T`, or if `T`'s column was already taken —
    /// see [`write`](Self::write).
    pub fn write_of<T: Component>(&mut self) -> &'w mut [T] {
        let id = crate::component_id::<T>();
        let i = self.access.write_index(id).unwrap_or_else(|| {
            panic!(
                "gg-ecs: query does not write `{}` — a mutable column can only be taken for a \
                 component the access set names",
                T::TYPE_NAME
            )
        });
        self.write(i)
    }

    /// The `i`-th write column as a typed mutable slice, where `i` indexes
    /// [`QueryAccess::writes`].
    ///
    /// Each write column may be taken **once**. The returned slice outlives the
    /// `&mut self` that produced it, so a second take would mint a second `&mut`
    /// to the same bytes; that is a panic here rather than UB, which is the M3
    /// exit criterion for aliasing.
    ///
    /// # Panics
    ///
    /// If `i` is out of range, if `T` does not match the column's stride, or if
    /// this column was already taken.
    pub fn write<T: Component>(&mut self, i: usize) -> &'w mut [T] {
        let view = self.raw_writes()[i];
        Self::assert_shape::<T>(view, "write");
        let bit = 1u64 << i;
        assert_eq!(
            self.taken & bit,
            0,
            "gg-ecs: write column {i} (`{}`) was already taken from this view; a second mutable \
             borrow of one column would alias",
            T::TYPE_NAME
        );
        self.taken |= bit;
        // SAFETY: as `read`, plus: `QueryAccess::new` rejected any duplicate in
        // the write set, so no other index in this view names these bytes, and
        // the `taken` bit just set rules out a second slice from this index.
        unsafe {
            if size_of::<T>() == 0 {
                core::slice::from_raw_parts_mut(
                    core::ptr::NonNull::<T>::dangling().as_ptr(),
                    view.len,
                )
            } else {
                core::slice::from_raw_parts_mut(view.ptr.cast::<T>(), view.len)
            }
        }
    }

    /// The `i`-th read column as raw bytes — `len * stride` of them, rows back
    /// to back — where `i` indexes [`QueryAccess::reads`].
    ///
    /// The untyped sibling of [`read`](Self::read), for a caller holding a
    /// [`ComponentInfo`](crate::registry::ComponentInfo) rather than a type: a
    /// component id chosen at runtime has no `T` to name. Nothing new is
    /// *reachable* through it — [`raw_reads`](Self::raw_reads) already hands the
    /// same pointer, length and stride out — it is the safe way to spend them.
    ///
    /// # Panics
    ///
    /// If `i` is out of range.
    #[must_use]
    pub fn read_bytes(&self, i: usize) -> &'w [u8] {
        let view = self.raw_reads()[i];
        if view.stride == 0 {
            return &[]; // a zero-sized component has no bytes and no pointer
        }
        // SAFETY: as `read`, minus the type. Every `Component` is `Pod`, so a
        // column holds no bit pattern a `&[u8]` could misread, and the stride
        // check `read` needs has nothing to check here.
        unsafe { core::slice::from_raw_parts(view.ptr, view.len * view.stride) }
    }

    /// The `i`-th write column as raw bytes, where `i` indexes
    /// [`QueryAccess::writes`]. Taken **once**, exactly as
    /// [`write`](Self::write) is and for the same reason.
    ///
    /// # Panics
    ///
    /// If `i` is out of range, or if this column was already taken.
    pub fn write_bytes(&mut self, i: usize) -> &'w mut [u8] {
        let view = self.raw_writes()[i];
        let bit = 1u64 << i;
        assert_eq!(
            self.taken & bit,
            0,
            "gg-ecs: write column {i} was already taken from this view; a second mutable borrow \
             of one column would alias"
        );
        self.taken |= bit;
        if view.stride == 0 {
            return &mut [];
        }
        // SAFETY: as `write`, minus the type — and `Pod` is what makes the
        // untyped form no weaker: every byte pattern a caller can write is a
        // value the component could already have held.
        unsafe { core::slice::from_raw_parts_mut(view.ptr, view.len * view.stride) }
    }

    fn assert_shape<T: Component>(view: ColumnView, kind: &str) {
        assert_eq!(
            view.stride,
            size_of::<T>(),
            "gg-ecs: {kind} column has stride {} but `{}` is {} bytes — wrong type for this \
             access index",
            view.stride,
            T::TYPE_NAME,
            size_of::<T>()
        );
    }
}

/// Split one archetype into the disjoint column views `access` asks for, or
/// `None` if the archetype does not hold every requested component.
///
/// The whole function takes raw pointers out of a unique borrow and hands them
/// to a lifetime-tagged view; disjointness is [`QueryAccess`]'s job, already
/// done before this is called.
///
/// Allocation-free: [`MAX_TERMS`] is what [`QueryAccess::new`] already refused
/// past, so the columns land in the view's inline array. It reads as one write
/// per column into `columns` rather than a push into a `Vec` per half, which is
/// the whole of the difference — every `each` over a live world used to charge
/// the allocator once per matching archetype per tick, for storage the §4.2.2
/// payload was going to copy into a fixed array anyway.
pub(crate) fn build<'w, 'q>(
    archetype: &'w mut Archetype,
    access: &'q QueryAccess,
) -> Option<ArchetypeView<'w, 'q>> {
    if !archetype.contains_all(access.matched()) {
        return None;
    }
    let entities_len = archetype.len();
    let entities = archetype.entities_ptr();

    let mut columns = [EMPTY_COLUMN; MAX_TERMS];
    for (at, &id) in access.read.iter().chain(access.write.iter()).enumerate() {
        let column = archetype.column_mut(archetype.column_index(id)?);
        columns[at] = ColumnView {
            ptr: column.base_ptr(),
            len: column.rows(),
            stride: column.stride(),
        };
    }
    Some(ArchetypeView {
        entities,
        entities_len,
        columns,
        reads: access.read.len(),
        writes: access.write.len(),
        taken: 0,
        access,
        _lifetime: core::marker::PhantomData,
    })
}

/// [`build`] over a shared archetype, for a query that writes nothing.
///
/// The extract stage takes `&World` by contract (§4.1) — one-way-ness is
/// supposed to be a *type* fact, not a discipline — so the read-only path needs
/// a read-only view. Callers reach this only through
/// [`World::each_ref`](crate::World::each_ref), whose `ReadOnly` bound is what
/// makes the empty write list a compile-time property; the assertion below is
/// the belt to that bracing, and it is `debug_assert` because a release build
/// cannot get here with a non-empty write list without the bound being unsound.
pub(crate) fn build_ref<'w, 'q>(
    archetype: &'w Archetype,
    access: &'q QueryAccess,
) -> Option<ArchetypeView<'w, 'q>> {
    debug_assert!(
        access.write.is_empty(),
        "build_ref reached with a write list — the ReadOnly bound was bypassed"
    );
    if !archetype.contains_all(access.matched()) {
        return None;
    }
    let mut columns = [EMPTY_COLUMN; MAX_TERMS];
    for (at, &id) in access.read.iter().enumerate() {
        let column = archetype.column(archetype.column_index(id)?);
        columns[at] = ColumnView {
            ptr: column.base_ptr_shared(),
            len: column.rows(),
            stride: column.stride(),
        };
    }
    Some(ArchetypeView {
        entities: archetype.entities_ptr_shared(),
        entities_len: archetype.len(),
        columns,
        reads: access.read.len(),
        writes: 0,
        taken: 0,
        access,
        _lifetime: core::marker::PhantomData,
    })
}
