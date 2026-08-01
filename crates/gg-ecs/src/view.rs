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
/// `repr(C)` because this crosses the §4.2.2 boundary unchanged. A stride of
/// zero is a marker component — `len` still counts rows.
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

/// Mutable columns per query, capped so [`ArchetypeView`]'s take-once bookkeeping
/// fits one word. A query writing 64 distinct components is a design error long
/// before it is a limit.
pub const MAX_WRITES: usize = 64;

/// Why a set of accesses cannot be granted.
#[derive(Debug, thiserror::Error)]
pub enum AliasError {
    #[error(
        "component {id:?} is borrowed mutably more than once by this query. Two mutable borrows \
         of one component would alias; split the work or borrow it once."
    )]
    WriteWrite { id: ComponentId },
    #[error(
        "component {id:?} is borrowed both mutably and immutably by this query. Drop the shared \
         borrow, or read it through the mutable one."
    )]
    ReadWrite { id: ComponentId },
    #[error(
        "a query may borrow at most {MAX_WRITES} components mutably; this one asks for {count}"
    )]
    TooManyWrites { count: usize },
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
        if w.len() > MAX_WRITES {
            return Err(AliasError::TooManyWrites { count: w.len() });
        }
        let mut r = read.to_vec();
        r.sort_unstable();
        r.dedup(); // reading twice is harmless, unlike writing twice
        if let Some(id) = r.iter().find(|id| w.binary_search(id).is_ok()) {
            return Err(AliasError::ReadWrite { id: *id });
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

    #[must_use]
    pub fn reads(&self) -> &[ComponentId] {
        &self.read
    }

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
    read: Vec<ColumnView>,
    write: Vec<ColumnView>,
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

    #[must_use]
    pub fn len(&self) -> usize {
        self.entities_len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities_len == 0
    }

    /// The raw views, as the §4.2.2 boundary would hand them over.
    #[must_use]
    pub fn raw_reads(&self) -> &[ColumnView] {
        &self.read
    }

    #[must_use]
    pub fn raw_writes(&self) -> &[ColumnView] {
        &self.write
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
        let view = self.read[i];
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
        let view = self.write[i];
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
pub(crate) fn build<'w, 'q>(
    archetype: &'w mut Archetype,
    access: &'q QueryAccess,
) -> Option<ArchetypeView<'w, 'q>> {
    if !archetype.contains_all(access.matched()) {
        return None;
    }
    let entities_len = archetype.len();
    let entities = archetype.entities_ptr();

    let mut read = Vec::with_capacity(access.read.len());
    let mut write = Vec::with_capacity(access.write.len());
    for (out, ids) in [
        (&mut read, access.read.as_slice()),
        (&mut write, access.write.as_slice()),
    ] {
        for &id in ids {
            let at = archetype.column_index(id)?;
            let column = archetype.column_mut(at);
            out.push(ColumnView {
                ptr: column.base_ptr(),
                len: column.rows(),
                stride: column.stride(),
            });
        }
    }
    Some(ArchetypeView {
        entities,
        entities_len,
        read,
        write,
        taken: 0,
        access,
        _lifetime: core::marker::PhantomData,
    })
}
