//! The chunked hash accelerator — **designed here, built later** (§4.2.1).
//!
//! [`World::canonical_hash`](crate::World::canonical_hash) is a full pass over
//! every component of every live entity, sorted by entity id. That is the
//! contract and it is not negotiable; it is also linear in world size, and the
//! dev loop pays it every tick it wants divergence detection. The accelerator
//! is the answer *when scale demands one* — not before demo 05-many proves the
//! need (§6, M9). What lands at M3 is the contract, so the implementation is
//! later a feature rather than a redesign.
//!
//! # What it is not
//!
//! Not the canonical hash, and not comparable to it. Chunk membership is
//! storage order, so a chunked hash is **layout-dependent** by construction:
//! two worlds in identical simulation states can carry different chunked
//! hashes if their archetypes were built up in a different order. It is valid
//! for exactly two jobs:
//!
//! - **Divergence detection** on one fixed engine version, where §4.2 guarantee
//!   1 makes iteration order reproducible.
//! - **Bisection**: CI names the diverging chunk, then the entity inside it,
//!   instead of scanning the world by hand.
//!
//! The canonical hash stays the only hash any CI gate reads and the only one
//! with cross-version meaning (§4.2.1). [`ChunkedHash`] is version-locked in the
//! type system so the quiet failure — a green-looking comparison that means
//! nothing — cannot be written (§8 risk register).
//!
//! # Chunk boundaries
//!
//! A chunk is a **half-open row range within one archetype's columns**:
//! `(ArchetypeId, start_row, len)` with `len <= CHUNK_ROWS`, `start_row` always
//! a multiple of `CHUNK_ROWS`. Three consequences worth stating because they
//! constrain the later implementation:
//!
//! - Chunks never span archetypes. An archetype's column set is fixed, so a
//!   chunk hash absorbs a known list of `(component id, stride)` pairs and can
//!   use the raw-column fast path over each range with no per-row dispatch.
//! - The final chunk of an archetype is short rather than padded. Padding to a
//!   fixed size would mean hashing bytes that are not state.
//! - Row `r` lives in chunk `r / CHUNK_ROWS`, which makes a structural change
//!   dirty a bounded set: swap-remove touches the chunk of the removed row and
//!   the chunk of the last row, and nothing else.
//!
//! # The change-tick contract
//!
//! Change ticks are opt-in per component type and exist only when a consumer
//! proves the need (§4.2); the accelerator is the second named consumer, and
//! this is the contract it needs from them.
//!
//! - A tick is a `u64` counter owned by the world, incremented once per
//!   simulation step. `u64` so wraparound is not a case anyone has to handle,
//!   and integral because §4.2.1 hazard 7 forbids float time.
//! - A component type that opts in carries one tick **per row**, written on
//!   every path that can change a row's bytes: `insert` (both the fresh and the
//!   overwrite path), the archetype move in `remove`, and any mutable column
//!   handed out by [`view`](crate::view) or [`query`](crate::query).
//! - The mutable-access rule is deliberately **conservative**: taking a write
//!   column marks every row in it, whether or not the system wrote anything.
//!   A missed mark is a stale hash, which is a wrong answer; a spurious mark is
//!   a redundant re-hash, which is only slower. Precision would need per-row
//!   write tracking, which costs more than the re-hash it saves.
//! - A chunk's cached hash is valid iff no row in it has a tick newer than the
//!   cache's. Structural change invalidates by row range, per the boundary
//!   rules above.
//! - Entity despawn and archetype moves invalidate through the same row-range
//!   path; there is no separate structural-change signal to keep in sync.
//!
//! The frame value is the fold of the per-chunk hashes in archetype order, then
//! chunk order — reproducible on one version, meaningless across two, which is
//! exactly what [`ChunkedHash`]'s type parameter encodes.

/// Rows per chunk.
///
/// A power of two so `row / CHUNK_ROWS` is a shift, and small enough that a
/// dirty chunk re-hashes a few kilobytes rather than a column. Tuning this is a
/// version-local change by definition — it moves chunk membership, so every
/// cached [`ChunkedHash`] from before the change is incomparable, which the
/// version parameter already says.
pub const CHUNK_ROWS: usize = 256;

/// A chunked hash, locked to the engine version that produced it.
///
/// The type parameter is the whole point: a value from another version is a
/// different type, so comparing across versions is a compile error rather than
/// a green-looking test. Use [`Local`] for the current version — naming a
/// literal version parameter is how you opt into meaninglessness.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct ChunkedHash<const VERSION: u64>(u128);

impl<const VERSION: u64> ChunkedHash<VERSION> {
    /// Wrap a value at `VERSION`. The version parameter is the point: it is
    /// what makes a cross-version comparison fail to compile.
    #[must_use]
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    /// The raw bits. Meaningful only against another value at the same
    /// `VERSION`.
    #[must_use]
    pub const fn get(self) -> u128 {
        self.0
    }

    /// The version this value is locked to — for reports, never for a
    /// runtime comparison that the type system already made impossible.
    #[must_use]
    pub const fn version(self) -> u64 {
        VERSION
    }
}

/// Packed `major.minor.patch` of this crate.
///
/// Storage-layout rework must bump at least the patch version. That is a
/// convention, not a machine check — the machine check is the canonical hash,
/// which is layout-independent and therefore *must* survive such a rework
/// unchanged (§4.2.1). This constant only stops two versions' accelerator
/// values from being compared by accident.
pub const LOCAL_VERSION: u64 = pack_version(
    parse_u64(env!("CARGO_PKG_VERSION_MAJOR")),
    parse_u64(env!("CARGO_PKG_VERSION_MINOR")),
    parse_u64(env!("CARGO_PKG_VERSION_PATCH")),
);

/// The chunked hash of the running engine version.
pub type Local = ChunkedHash<LOCAL_VERSION>;

const fn pack_version(major: u64, minor: u64, patch: u64) -> u64 {
    (major << 40) | (minor << 20) | patch
}

/// Decimal parse in const context; `env!` hands us strings, and this runs at
/// compile time on values cargo generated, so a non-digit is a build failure.
const fn parse_u64(s: &str) -> u64 {
    let bytes = s.as_bytes();
    let mut value = 0u64;
    let mut i = 0;
    while i < bytes.len() {
        assert!(
            bytes[i].is_ascii_digit(),
            "crate version component is not decimal"
        );
        value = value * 10 + (bytes[i] - b'0') as u64;
        i += 1;
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_version_is_this_crates_own() {
        assert_eq!(
            Local::new(0).version(),
            pack_version(
                parse_u64(env!("CARGO_PKG_VERSION_MAJOR")),
                parse_u64(env!("CARGO_PKG_VERSION_MINOR")),
                parse_u64(env!("CARGO_PKG_VERSION_PATCH")),
            )
        );
    }

    #[test]
    fn packing_orders_versions_the_way_semver_does() {
        assert!(pack_version(0, 1, 0) < pack_version(0, 1, 1));
        assert!(pack_version(0, 1, 999) < pack_version(0, 2, 0));
        assert!(pack_version(0, 999, 0) < pack_version(1, 0, 0));
    }

    #[test]
    fn chunk_rows_is_a_power_of_two_so_the_divide_is_a_shift() {
        assert!(CHUNK_ROWS.is_power_of_two());
    }
}
