//! Asset identity: the name a game says, as the number a pack stores (§4.6).
//!
//! # Why the hash is here and not in `gg-assets`
//!
//! A game crate may link `gg-abi`, `gg-ecs`, `gg-ecs-derive` and `gg-math`
//! (§3's deny pin), so it cannot call anything in `gg-assets` — and naming
//! content is exactly what the render protocol's [`Model`] asks it to do. The
//! alternatives were a second definition of the hash on the game side (two
//! definitions of one convention, which is how a format acquires a dialect) or
//! widening the pin to admit a crate that mmaps files into a dylib's address
//! space. This is the third: one definition, at the leaf both sides already
//! link, with `gg-assets` calling it rather than restating it.
//!
//! [`Model`]: https://docs.rs/gg-ecs
//!
//! # Why not xxhash3
//!
//! Not quality — reach. The hash has to be a `const fn` with no dependency
//! behind it (`gg-abi` allocates nothing and links one crate), which xxhash3's
//! block structure is not. FNV-1a accumulates a byte at a time, and the
//! splitmix64 finalizer below fixes the avalanche it is otherwise weak on.
//! Collisions are not a probability anyone has to reason about: `PackWriter`
//! sorts by id and refuses a duplicate by name, so two names that collided
//! would fail a build rather than silently draw the wrong mesh.

/// FNV-1a's 64-bit offset basis.
const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a's 64-bit prime.
const PRIME: u64 = 0x0000_0100_0000_01b3;

/// Hash an asset name into its id.
///
/// The name is hashed exactly as given: normalizing it — separators, case — is
/// the importer's job and must happen before this, or a pack built on Windows
/// and one built on Linux name the same asset differently.
///
/// `const`, so a game that names an asset spends nothing at runtime doing it.
#[must_use]
pub const fn asset_id(name: &str) -> u64 {
    let bytes = name.as_bytes();
    let mut hash = OFFSET_BASIS;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(PRIME);
        i += 1;
    }
    finalize(hash)
}

/// splitmix64's finalizer. FNV-1a's low bits move well and its high bits barely
/// move at all; a pack's entry table is sorted by the whole id and its lookup
/// is a binary search, so an id whose top bits track the name's first byte
/// would cluster every asset of one source file into one region of the table.
const fn finalize(mut hash: u64) -> u64 {
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(0x94d0_49bb_1331_11eb);
    hash ^ (hash >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_hashes_the_same_way_twice_and_two_names_differently() {
        assert_eq!(asset_id("hall/mesh/0.0"), asset_id("hall/mesh/0.0"));
        assert_ne!(asset_id("hall/mesh/0.0"), asset_id("hall/mesh/0.1"));
        // Case and separators are the importer's to normalize, not this.
        assert_ne!(asset_id("hall/scene"), asset_id("Hall/scene"));
    }

    #[test]
    fn the_empty_name_is_not_the_reserved_id() {
        // `AssetId::NONE` is 0 and a pack may not contain it. Nothing forces a
        // hash away from zero, so this pins the one input anyone would reach
        // for by accident.
        assert_ne!(asset_id(""), 0);
    }

    #[test]
    fn one_flipped_bit_moves_more_than_the_low_byte() {
        // The finalizer's whole job. Without it FNV-1a leaves the top bits of
        // two names differing in their last character nearly identical, and the
        // pack's entry table clusters.
        let (a, b) = (asset_id("hall/mesh/0.0"), asset_id("hall/mesh/0.1"));
        let moved = (a ^ b).count_ones();
        assert!(moved > 16, "only {moved} bits moved: {a:#018x} {b:#018x}");
    }

    #[test]
    fn it_is_usable_where_a_constant_is_required() {
        const ID: u64 = asset_id("hall/scene");
        assert_eq!(ID, asset_id("hall/scene"));
    }
}
