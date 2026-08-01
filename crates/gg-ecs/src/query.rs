//! Typed queries over the column views (§4.2).
//!
//! [`view`](crate::view) is the substrate: whole columns, one call per archetype
//! match. This is the thin typed layer over it — `&T` for shared access, `&mut T`
//! for exclusive, tuples for both — so gameplay code names components instead of
//! access indices, and the compiler checks the types the view would otherwise
//! only assert at runtime.
//!
//! Thin on purpose. It resolves a tuple to slices once per archetype and then
//! indexes them; there is no per-entity dispatch, no filter DSL, and no
//! iterator adaptors. §7 keeps the scheduler out of this crate, and a query
//! language is how a scheduler gets in by increments.

use crate::component::Component;
use crate::hash::ComponentId;
use crate::view::{AliasError, ArchetypeView, QueryAccess};

/// What one query row yields, and how to resolve it against an archetype.
///
/// Implemented for `&T`, `&mut T`, and tuples of those. Not meant to be
/// implemented downstream: the aliasing argument in [`QueryAccess`] rests on
/// [`access`](Self::access) reporting exactly the columns
/// [`columns`](Self::columns) later takes.
pub trait QueryData {
    /// One row's worth of borrows.
    type Item<'r>;
    /// One archetype's worth, resolved once per match.
    type Columns<'w>;

    /// Report the components this data borrows, shared into `read` and
    /// exclusive into `write`.
    fn access(read: &mut Vec<ComponentId>, write: &mut Vec<ComponentId>);

    /// Resolve against one matched archetype.
    fn columns<'w>(view: &mut ArchetypeView<'w, '_>) -> Self::Columns<'w>;

    /// Borrow row `row`. Callers pass rows in `0..view.len()`.
    fn get<'r, 'w: 'r>(columns: &'r mut Self::Columns<'w>, row: usize) -> Self::Item<'r>;
}

impl<T: Component> QueryData for &T {
    type Item<'r> = &'r T;
    type Columns<'w> = &'w [T];

    fn access(read: &mut Vec<ComponentId>, _write: &mut Vec<ComponentId>) {
        read.push(crate::component_id::<T>());
    }

    fn columns<'w>(view: &mut ArchetypeView<'w, '_>) -> Self::Columns<'w> {
        view.read_of::<T>()
    }

    fn get<'r, 'w: 'r>(columns: &'r mut Self::Columns<'w>, row: usize) -> Self::Item<'r> {
        &columns[row]
    }
}

impl<T: Component> QueryData for &mut T {
    type Item<'r> = &'r mut T;
    type Columns<'w> = &'w mut [T];

    fn access(_read: &mut Vec<ComponentId>, write: &mut Vec<ComponentId>) {
        write.push(crate::component_id::<T>());
    }

    fn columns<'w>(view: &mut ArchetypeView<'w, '_>) -> Self::Columns<'w> {
        view.write_of::<T>()
    }

    fn get<'r, 'w: 'r>(columns: &'r mut Self::Columns<'w>, row: usize) -> Self::Item<'r> {
        &mut columns[row]
    }
}

macro_rules! tuple_query {
    ($($name:ident),+) => {
        impl<$($name: QueryData),+> QueryData for ($($name,)+) {
            type Item<'r> = ($($name::Item<'r>,)+);
            type Columns<'w> = ($($name::Columns<'w>,)+);

            fn access(read: &mut Vec<ComponentId>, write: &mut Vec<ComponentId>) {
                $($name::access(read, write);)+
            }

            fn columns<'w>(view: &mut ArchetypeView<'w, '_>) -> Self::Columns<'w> {
                ($($name::columns(view),)+)
            }

            fn get<'r, 'w: 'r>(columns: &'r mut Self::Columns<'w>, row: usize) -> Self::Item<'r> {
                // Type and binding names coincide; they live in different
                // namespaces, which is what makes the expansion readable.
                #[allow(non_snake_case)]
                let ($($name,)+) = columns;
                ($($name::get($name, row),)+)
            }
        }
    };
}

tuple_query!(A);
tuple_query!(A, B);
tuple_query!(A, B, C);
tuple_query!(A, B, C, D);
tuple_query!(A, B, C, D, E);
tuple_query!(A, B, C, D, E, F);
tuple_query!(A, B, C, D, E, F, G);
tuple_query!(A, B, C, D, E, F, G, H);

/// A validated query: an access set with the type that produced it.
///
/// Build once, reuse every tick. The type parameter is what removes the
/// footgun the raw view API still has — an access set and a set of `read::<T>`
/// calls that disagree — because here one derives the other.
pub struct Query<D: QueryData> {
    access: QueryAccess,
    _data: core::marker::PhantomData<fn() -> D>,
}

impl<D: QueryData> Query<D> {
    /// Validate `D`'s accesses (§4.2.2).
    ///
    /// # Errors
    ///
    /// If `D` borrows one component mutably twice, or both mutably and
    /// immutably — the same check the raw API makes, named by component.
    pub fn new() -> Result<Self, AliasError> {
        let (mut read, mut write) = (Vec::new(), Vec::new());
        D::access(&mut read, &mut write);
        Ok(Self {
            access: QueryAccess::new(&read, &write)?,
            _data: core::marker::PhantomData,
        })
    }

    #[must_use]
    pub fn access(&self) -> &QueryAccess {
        &self.access
    }
}

impl<D: QueryData> core::fmt::Debug for Query<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Query")
            .field("access", &self.access)
            .finish()
    }
}
