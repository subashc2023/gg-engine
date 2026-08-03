//! The one hash primitive (§4.2, §4.2.1).
//!
//! Component ids, schema hashes and the canonical state hash all come from
//! blake3 — pinned at `=1.8.5`, `pure` backend. The choice is documented in
//! PLAN.md §6/M3; the short version is that blake3's output is a frozen
//! published spec, so a patch release cannot move bits, and `pure` keeps
//! runtime CPUID dispatch out of the determinism path.
//!
//! Every hash here is **domain-separated**: the domain string is absorbed
//! first, so a component id and a schema hash over the same bytes differ. Two
//! kinds of value that can never be confused for one another is worth more
//! than the handful of bytes it costs.

use core::fmt;

/// Domain separators. Changing one of these is a Determinism Contract event —
/// it moves every id and hash in the domain at once.
const DOMAIN_COMPONENT_ID: &[u8] = b"gg-ecs/component-id/v1\0";
const DOMAIN_SIDE_TABLE_ID: &[u8] = b"gg-ecs/side-table-id/v1\0";
const DOMAIN_SYSTEM_ID: &[u8] = b"gg-ecs/system-id/v1\0";
const DOMAIN_SCHEMA: &[u8] = b"gg-ecs/schema/v1\0";
const DOMAIN_CANONICAL: &[u8] = b"gg-ecs/canonical-state/v1\0";

/// A component's persisted identity: `blake3(domain ‖ declared id)`, first 8
/// bytes little-endian (§4.2). Derived from the **declared** `#[component(id =
/// "…")]` string, never from the Rust type path — moving `game::ships::Velocity`
/// to `game::physics::Velocity` is a source-organization event, not a schema
/// event.
///
/// 64 bits is a birthday collision around 2^32 components, which no game
/// reaches; the registry checks for collisions anyway and reports both
/// declaring types, because "unreachable" and "unchecked" are different words.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComponentId(u64);

impl ComponentId {
    /// Hash a declared component id string. `const`-callable is deliberately
    /// *not* offered: the derive calls this at registration, and a `const`
    /// blake3 would be a second implementation to keep bit-identical.
    #[must_use]
    pub fn of(declared: &str) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(DOMAIN_COMPONENT_ID);
        h.update(declared.as_bytes());
        let mut out = [0u8; 8];
        out.copy_from_slice(&h.finalize().as_bytes()[..8]);
        Self(u64::from_le_bytes(out))
    }

    /// The raw 64 bits, for the `repr(C)` boundary (§4.2.2) and for tests.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Reconstruct from the wire. The boundary hands these across `extern "C"`;
    /// an unknown value is a lookup miss, never unsafe.
    #[must_use]
    pub const fn from_raw(bits: u64) -> Self {
        Self(bits)
    }
}

impl fmt::Debug for ComponentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ComponentId({:#018x})", self.0)
    }
}

/// A system's stable id for the §4.2.2 systems table: the same construction as
/// [`ComponentId`] under its own domain, over the *declared* system name.
///
/// Raw `u64` rather than a newtype because it exists only to fill
/// [`SystemEntry::id`](gg_abi::SystemEntry) — nothing in this crate keys on it.
/// What it buys the host is telling "the table was reordered" from "a system was
/// replaced", which is why it must not be the table index.
#[must_use]
pub fn system_id(name: &str) -> u64 {
    let mut h = blake3::Hasher::new();
    h.update(DOMAIN_SYSTEM_ID);
    h.update(name.as_bytes());
    let mut out = [0u8; 8];
    out.copy_from_slice(&h.finalize().as_bytes()[..8]);
    u64::from_le_bytes(out)
}

/// A hashed side table's persisted identity (§4.2.1) — the same construction as
/// [`ComponentId`] under its own domain.
///
/// Separate domains rather than one shared namespace: a side table and a
/// component are different kinds of state with different storage, and the day
/// one is promoted to the other the id *should* change, loudly.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SideTableId(u64);

impl SideTableId {
    /// The id for a declared side-table name, under the side-table domain.
    #[must_use]
    pub fn of(declared: &str) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(DOMAIN_SIDE_TABLE_ID);
        h.update(declared.as_bytes());
        let mut out = [0u8; 8];
        out.copy_from_slice(&h.finalize().as_bytes()[..8]);
        Self(u64::from_le_bytes(out))
    }

    /// The raw bits, for storage and diagnostics.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Debug for SideTableId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SideTableId({:#018x})", self.0)
    }
}

/// A component's layout fingerprint (§4.2): field names, offsets, sizes and
/// element types, hashed by the derive at compile time into a `const`.
///
/// Unchanged across a reload means in-place state reuse; changed routes through
/// snapshot → migrate → restore (§4.2.2). It covers *layout*, not semantics —
/// renaming a field changes it, changing what the field means does not. That
/// gap is what §4.2.2's dylib fingerprint exists to cover.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SchemaHash(u128);

impl SchemaHash {
    /// Wrap bits the derive computed at compile time. The only intended caller
    /// is generated code — a hand-built value is a schema claim nothing checked.
    #[must_use]
    pub const fn from_raw(bits: u128) -> Self {
        Self(bits)
    }

    /// The raw bits, for storage and diagnostics.
    #[must_use]
    pub const fn get(self) -> u128 {
        self.0
    }
}

impl fmt::Debug for SchemaHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SchemaHash({:#034x})", self.0)
    }
}

/// The canonical state hash (§4.2.1) — the contract, and the only hash any CI
/// gate reads.
///
/// 128 bits, because this is the value cross-architecture determinism is
/// decided by and a 64-bit birthday bound is uncomfortably close to plausible
/// entity counts over a long chaos-replay campaign.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CanonicalHash(u128);

impl CanonicalHash {
    /// The raw bits. Compare these across architectures; do not compare them
    /// against a [`ChunkedHash`](crate::chunked::ChunkedHash), which is
    /// version-local by construction (§4.2.1).
    #[must_use]
    pub const fn get(self) -> u128 {
        self.0
    }
}

impl fmt::Debug for CanonicalHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Lowercase zero-padded hex, so hash sequences diff cleanly across
        // architectures in CI output.
        write!(f, "{:032x}", self.0)
    }
}

impl fmt::Display for CanonicalHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:032x}", self.0)
    }
}

/// Streaming hasher for the state-hash protocol (§4.2.1).
///
/// The protocol is defined by what this type absorbs, in what order. Every
/// method is a documented wire commitment: changing one is a Determinism
/// Contract event, not a refactor. Integers go in little-endian by explicit
/// `to_le_bytes` rather than by memcpy — the supported targets are all
/// little-endian, but "we only support LE" and "the encoding is LE" are
/// different claims and only the second one survives a port.
pub struct StateHasher {
    inner: blake3::Hasher,
    /// The first NaN absorbed through [`Self::f32`]/[`Self::f64`], as
    /// `(width, bits)` — §1.13 hazard 6's witness for hashed state the column
    /// scan cannot reach. Side tables are not flat `Pod` memory and carry no
    /// field layout, so this funnel is the only place all of their floats are
    /// guaranteed to pass.
    ///
    /// Not feature-gated, deliberately: `gg-ecs` is on the game-crate pin (§3),
    /// so a dylib and a host disagreeing about this field would disagree about
    /// the layout of the `&mut StateHasher` a side table's `state_hash` writes
    /// through. One compare per float, on a path the raw column fast path does
    /// not take, is the price of that not being a question.
    nan: Option<(u32, u64)>,
}

impl StateHasher {
    /// A hasher for the canonical pass. The domain is absorbed immediately, so
    /// an empty world still has a domain-specific hash rather than blake3's
    /// hash of nothing.
    #[must_use]
    pub fn canonical() -> Self {
        let mut inner = blake3::Hasher::new();
        inner.update(DOMAIN_CANONICAL);
        Self { inner, nan: None }
    }

    /// A hasher for schema fingerprints — same protocol, different domain, so
    /// the derive can reuse the encoding methods.
    #[must_use]
    pub fn schema() -> Self {
        let mut inner = blake3::Hasher::new();
        inner.update(DOMAIN_SCHEMA);
        Self { inner, nan: None }
    }

    /// Raw bytes, no length prefix. For `Pod` column slices, where the length
    /// is committed separately by the caller (`raw_column` below).
    pub fn bytes(&mut self, bytes: &[u8]) {
        self.inner.update(bytes);
    }

    /// A length-prefixed byte run. Prefixing is what keeps `["ab", "c"]` and
    /// `["a", "bc"]` apart — the classic concatenation ambiguity, and the one
    /// encoding bug that would show up as a phantom cross-architecture
    /// divergence months later.
    pub fn delimited(&mut self, bytes: &[u8]) {
        self.u64(bytes.len() as u64);
        self.inner.update(bytes);
    }

    /// A `str`, length-prefixed. UTF-8 bytes; no normalization.
    pub fn str(&mut self, s: &str) {
        self.delimited(s.as_bytes());
    }

    /// One byte. The integer family below is **little-endian regardless of
    /// host**, which is what makes the encoding an architecture-independent
    /// contract rather than a memory dump (§4.2.1).
    pub fn u8(&mut self, v: u8) {
        self.inner.update(&[v]);
    }

    /// Two bytes, little-endian.
    pub fn u16(&mut self, v: u16) {
        self.inner.update(&v.to_le_bytes());
    }

    /// Four bytes, little-endian.
    pub fn u32(&mut self, v: u32) {
        self.inner.update(&v.to_le_bytes());
    }

    /// Eight bytes, little-endian.
    pub fn u64(&mut self, v: u64) {
        self.inner.update(&v.to_le_bytes());
    }

    /// Sixteen bytes, little-endian.
    pub fn u128(&mut self, v: u128) {
        self.inner.update(&v.to_le_bytes());
    }

    /// One byte, two's complement.
    pub fn i8(&mut self, v: i8) {
        self.inner.update(&v.to_le_bytes());
    }

    /// Two bytes, little-endian two's complement.
    pub fn i16(&mut self, v: i16) {
        self.inner.update(&v.to_le_bytes());
    }

    /// Four bytes, little-endian two's complement.
    pub fn i32(&mut self, v: i32) {
        self.inner.update(&v.to_le_bytes());
    }

    /// Eight bytes, little-endian two's complement.
    pub fn i64(&mut self, v: i64) {
        self.inner.update(&v.to_le_bytes());
    }

    /// Sixteen bytes, little-endian two's complement.
    pub fn i128(&mut self, v: i128) {
        self.inner.update(&v.to_le_bytes());
    }

    /// Floats by `to_bits()`, never by value: `-0.0 == 0.0` and `NaN != NaN`
    /// would both make the hash disagree with equality in the wrong direction.
    /// A NaN reaching hashed state is itself the bug (§4.2.1 hazard 6) — this
    /// encoding does not paper over it, and [`Self::nan_witness`] records it.
    pub fn f32(&mut self, v: f32) {
        let bits = v.to_bits();
        // First wins, so the answer is a function of the encoding order and not
        // of how many NaNs followed. `to_bits` is a bitcast: the payload the
        // architectures disagree about survives to the test.
        if self.nan.is_none() && crate::nan_scan::is_nan32(bits) {
            self.nan = Some((32, u64::from(bits)));
        }
        self.u32(bits);
    }

    /// As [`Self::f32`], by `to_bits()`.
    pub fn f64(&mut self, v: f64) {
        let bits = v.to_bits();
        if self.nan.is_none() && crate::nan_scan::is_nan64(bits) {
            self.nan = Some((64, bits));
        }
        self.u64(bits);
    }

    /// The first NaN this hasher absorbed as a float, `(width in bits, bits)`.
    ///
    /// Only floats offered through [`Self::f32`]/[`Self::f64`] are seen. A type
    /// that hashes its floats as raw bytes is invisible here for the same reason
    /// an unclassifiable `ty` token is invisible to the column scan — a hole,
    /// and the same kind.
    #[must_use]
    pub const fn nan_witness(&self) -> Option<(u32, u64)> {
        self.nan
    }

    /// Bool as one byte, `0` or `1`. Not reachable from a `Pod` component —
    /// `bool` is not `Pod` — but side tables (§4.2.1) are not `Pod` and do
    /// carry them.
    pub fn bool(&mut self, v: bool) {
        self.u8(u8::from(v));
    }

    /// A whole `Pod` column: element count, element stride, then the bytes.
    /// This is the raw-column fast path's *only* entry point, so the CI proof
    /// that fast path == protocol has exactly one thing to compare (§4.2.1).
    ///
    /// Committing count and stride separately matters: without them a 2-element
    /// column of 8-byte values and a 4-element column of 4-byte values over the
    /// same 16 bytes would hash identically.
    pub fn raw_column(&mut self, count: usize, stride: usize, bytes: &[u8]) {
        self.u64(count as u64);
        self.u64(stride as u64);
        self.bytes(bytes);
    }

    /// Finish, as the contract-bearing canonical value.
    #[must_use]
    pub fn finish_canonical(self) -> CanonicalHash {
        CanonicalHash(u128::from_le_bytes(truncate_16(self.inner.finalize())))
    }

    /// Finish, as a schema fingerprint.
    #[must_use]
    pub fn finish_schema(self) -> SchemaHash {
        SchemaHash(u128::from_le_bytes(truncate_16(self.inner.finalize())))
    }
}

/// blake3 emits 32 bytes; we keep the first 16. Truncating a blake3 digest is
/// explicitly sanctioned by the spec (its XOF output is uniform), unlike
/// truncating a Merkle–Damgård hash.
fn truncate_16(digest: blake3::Hash) -> [u8; 16] {
    let mut out = [0u8; 16];
    out.copy_from_slice(&digest.as_bytes()[..16]);
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn component_ids_are_stable_values_not_just_stable_within_a_run() {
        // Frozen on 2026-07-31. These are persisted-identity values: a change
        // here is a save-format break, so the test exists to make that change
        // loud rather than to check that hashing works.
        assert_eq!(ComponentId::of("position").get(), 0x2770_34ae_e10d_0b6e);
        assert_eq!(ComponentId::of("velocity").get(), 0x6ba2_db89_e4d3_408b);
    }

    #[test]
    fn domains_separate_identical_input() {
        let mut a = StateHasher::canonical();
        let mut b = StateHasher::schema();
        a.u64(7);
        b.u64(7);
        assert_ne!(a.finish_canonical().get(), b.finish_schema().get());
    }

    #[test]
    fn length_prefixing_defeats_concatenation_ambiguity() {
        let mut a = StateHasher::canonical();
        a.str("ab");
        a.str("c");
        let mut b = StateHasher::canonical();
        b.str("a");
        b.str("bc");
        assert_ne!(a.finish_canonical(), b.finish_canonical());
    }

    #[test]
    fn raw_column_commits_shape_not_only_bytes() {
        let bytes = [1u8; 16];
        let mut a = StateHasher::canonical();
        a.raw_column(2, 8, &bytes);
        let mut b = StateHasher::canonical();
        b.raw_column(4, 4, &bytes);
        assert_ne!(a.finish_canonical(), b.finish_canonical());
    }

    #[test]
    fn float_encoding_distinguishes_signed_zero() {
        let mut a = StateHasher::canonical();
        a.f64(0.0);
        let mut b = StateHasher::canonical();
        b.f64(-0.0);
        assert_ne!(a.finish_canonical(), b.finish_canonical());
    }
}
