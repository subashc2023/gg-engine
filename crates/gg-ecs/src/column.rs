//! Type-erased component columns (§4.2).
//!
//! A column is flat bytes plus a stride. Components are `Pod`, so there is no
//! drop glue, no element-wise move, and no partially-initialized state to
//! reason about — insertion and removal are `copy_from_slice`.
//!
//! # Why the backing store is over-aligned rather than hand-allocated
//!
//! Columns must satisfy each component's alignment, and §4.2.2 hands raw column
//! pointers across a C ABI where a misaligned pointer is UB rather than a slow
//! path. The obvious implementation is `alloc`/`realloc`/`dealloc` against a
//! per-component `Layout`, which is also several hundred lines of `unsafe` that
//! Miri has to be trusted to cover.
//!
//! Instead the backing store is `Vec<Block>` where [`Block`] is 16-byte
//! aligned, so every column base pointer is 16-aligned by construction and
//! `Vec` keeps ownership of the allocation. Components with alignment above 16
//! are refused at registration with a named error — sim components have no
//! reason to be SIMD-aligned, that being a render-side concern on the far side
//! of the §1.4 membrane. The result is a column with **no `unsafe` at all**.

/// Backing unit: 16 bytes, 16-aligned. The alignment is the entire point.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Block([u8; 16]);

// SAFETY: a `repr(C, align(16))` wrapper around a byte array — no padding, no
// interior mutability, every bit pattern valid.
unsafe impl bytemuck::Zeroable for Block {}
// SAFETY: as above, plus `Copy`.
unsafe impl bytemuck::Pod for Block {}

/// The largest component alignment a column can serve.
pub(crate) const MAX_ALIGN: usize = align_of::<Block>();

/// One component's storage within one archetype.
#[derive(Clone)]
pub(crate) struct Column {
    blocks: Vec<Block>,
    /// Row count. Not derivable from `blocks.len()`, which is rounded up to
    /// whole blocks and is meaningless for a zero-sized component.
    rows: usize,
    /// Bytes per row. Zero for marker components, which the whole file handles
    /// by arithmetic rather than by special case.
    stride: usize,
}

impl Column {
    pub(crate) fn new(stride: usize) -> Self {
        Self {
            blocks: Vec::new(),
            rows: 0,
            stride,
        }
    }

    pub(crate) const fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) const fn stride(&self) -> usize {
        self.stride
    }

    /// The live bytes: `rows * stride`, never the block-rounded capacity.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &bytemuck::cast_slice(&self.blocks)[..self.rows * self.stride]
    }

    pub(crate) fn as_bytes_mut(&mut self) -> &mut [u8] {
        let end = self.rows * self.stride;
        &mut bytemuck::cast_slice_mut(&mut self.blocks)[..end]
    }

    /// One row's bytes.
    pub(crate) fn row(&self, row: usize) -> &[u8] {
        debug_assert!(row < self.rows, "column row {row} out of range");
        &self.as_bytes()[row * self.stride..(row + 1) * self.stride]
    }

    pub(crate) fn row_mut(&mut self, row: usize) -> &mut [u8] {
        debug_assert!(row < self.rows, "column row {row} out of range");
        let (start, end) = (row * self.stride, (row + 1) * self.stride);
        &mut self.as_bytes_mut()[start..end]
    }

    /// Base pointer for a [`ColumnView`](crate::view::ColumnView).
    ///
    /// Takes `&mut self` because the resulting pointer may be written through.
    /// For an empty or zero-stride column this is `Vec`'s dangling-but-aligned
    /// base, which is correct: `len` is zero, so nothing dereferences it.
    pub(crate) fn base_ptr(&mut self) -> *mut u8 {
        bytemuck::cast_slice_mut::<Block, u8>(&mut self.blocks).as_mut_ptr()
    }

    /// Base pointer for a read-only view (`ReadOnly` queries, §4.2).
    ///
    /// The `*mut` in [`ColumnView`](crate::view::ColumnView) is shape, not
    /// permission: this pointer carries shared provenance and writing through
    /// it is UB. Only [`view::build_ref`](crate::view::build_ref) may produce
    /// it, and only for an access set whose write list is empty — enforced at
    /// the type level by the `ReadOnly` bound, not by this comment.
    pub(crate) fn base_ptr_shared(&self) -> *mut u8 {
        bytemuck::cast_slice::<Block, u8>(&self.blocks)
            .as_ptr()
            .cast_mut()
    }

    /// Grow the block store so `rows` rows fit.
    fn reserve_rows(&mut self, rows: usize) {
        let needed = rows * self.stride;
        let blocks = needed.div_ceil(size_of::<Block>());
        if blocks > self.blocks.len() {
            self.blocks.resize(blocks, Block([0; 16]));
        }
    }

    /// Append a row. `bytes.len()` must equal the stride.
    ///
    /// Only the tests call this — `World` writes through `row_mut` after
    /// `push_zeroed`, because an archetype move fills every column at once.
    /// It stays because it is the clearest way to exercise block growth.
    #[cfg(test)]
    pub(crate) fn push(&mut self, bytes: &[u8]) {
        assert_eq!(
            bytes.len(),
            self.stride,
            "gg-ecs: column push size mismatch"
        );
        self.reserve_rows(self.rows + 1);
        self.rows += 1;
        let row = self.rows - 1;
        self.row_mut(row).copy_from_slice(bytes);
    }

    /// Append a zeroed row — the default for a component added to an existing
    /// entity, and for a field that migration has no source for (§4.2.2).
    /// Zero is always a valid value: every component is `Pod`, hence `Zeroable`.
    pub(crate) fn push_zeroed(&mut self) {
        self.reserve_rows(self.rows + 1);
        self.rows += 1;
        let row = self.rows - 1;
        self.row_mut(row).fill(0);
    }

    /// Remove `row`, moving the last row into the hole.
    ///
    /// Swap-remove is the specified removal discipline (§4.2): it is what makes
    /// storage placement a deterministic function of the operation sequence,
    /// which is the property the dense-index contract actually promises.
    pub(crate) fn swap_remove(&mut self, row: usize) {
        assert!(row < self.rows, "gg-ecs: column swap_remove out of range");
        let last = self.rows - 1;
        if row != last && self.stride != 0 {
            let stride = self.stride;
            let (start, end) = (last * stride, (last + 1) * stride);
            self.as_bytes_mut().copy_within(start..end, row * stride);
        }
        self.rows -= 1;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn push_and_read_back_round_trips() {
        let mut c = Column::new(4);
        c.push(&[1, 2, 3, 4]);
        c.push(&[5, 6, 7, 8]);
        assert_eq!(c.rows(), 2);
        assert_eq!(c.row(0), &[1, 2, 3, 4]);
        assert_eq!(c.row(1), &[5, 6, 7, 8]);
        assert_eq!(c.as_bytes(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn growth_across_block_boundaries_preserves_every_row() {
        // Stride 3 deliberately does not divide the 16-byte block, so rows
        // straddle block boundaries and an off-by-one in reserve_rows shows up.
        let mut c = Column::new(3);
        for i in 0..40u8 {
            c.push(&[i, i.wrapping_add(1), i.wrapping_add(2)]);
        }
        assert_eq!(c.rows(), 40);
        for i in 0..40u8 {
            assert_eq!(
                c.row(i as usize),
                &[i, i.wrapping_add(1), i.wrapping_add(2)]
            );
        }
    }

    #[test]
    fn swap_remove_moves_the_last_row_into_the_hole() {
        let mut c = Column::new(1);
        for i in 0..4u8 {
            c.push(&[i]);
        }
        c.swap_remove(1);
        assert_eq!(c.rows(), 3);
        assert_eq!(c.as_bytes(), &[0, 3, 2]);
        // Removing the last row is a plain truncation.
        c.swap_remove(2);
        assert_eq!(c.as_bytes(), &[0, 3]);
    }

    #[test]
    fn zero_sized_components_are_counted_not_stored() {
        let mut c = Column::new(0);
        c.push(&[]);
        c.push_zeroed();
        assert_eq!(c.rows(), 2);
        assert!(c.as_bytes().is_empty());
        c.swap_remove(0);
        assert_eq!(c.rows(), 1);
    }

    #[test]
    fn the_backing_store_is_over_aligned_for_every_legal_component() {
        let mut c = Column::new(8);
        c.push(&[0; 8]);
        let addr = c.as_bytes().as_ptr() as usize;
        assert_eq!(
            addr % MAX_ALIGN,
            0,
            "column base must be {MAX_ALIGN}-aligned"
        );
    }

    #[test]
    fn capacity_padding_never_reaches_the_hash() {
        // 3 rows of stride 3 is 9 live bytes inside one 16-byte block; the
        // other 7 must not be absorbed, or the hash would depend on capacity.
        let mut c = Column::new(3);
        for i in 0..3u8 {
            c.push(&[i, i, i]);
        }
        assert_eq!(c.as_bytes().len(), 9);
    }
}
