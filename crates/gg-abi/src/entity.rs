//! Entity identity (§4.2), defined at the boundary rather than in `gg-ecs`.
//!
//! It lives here for one reason: an archetype match hands the dylib a
//! `*const Entity` over the host's live entity array, and components holding
//! entity references cross inside `Pod` column bytes. Both make [`Entity`] a
//! layout contract between two separately-compiled artifacts, which is this
//! crate's entire subject. `gg-ecs` re-exports it and still owns the
//! *allocator* and the allocation contract — but the type has one definition,
//! and no fingerprint is asked to stand in for one.

use core::fmt;

/// A live-or-dead handle: a dense `index` plus the `generation` that index was
/// on when the handle was made.
///
/// `repr(C)` and `Pod` because components carrying entity references must stay
/// `Pod` (§4.2.1 hazard 4) and these cross the §4.2.2 reload boundary.
///
/// `Ord` sorts by index then generation — load-bearing, not a convenience impl:
/// it is the order the canonical hash walks state in.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(C)]
pub struct Entity {
    // `pub(crate)` only so the layout assertions in `tests` can name offsets;
    // the halves stay private to every caller outside this crate.
    pub(crate) index: u32,
    pub(crate) generation: u32,
}

// SAFETY: `repr(C)`, two `u32`s — 8 bytes, no padding, no interior mutability.
// Every bit pattern is a valid `Entity`; an unrecognized one is simply not
// alive, which `Entities::is_alive` reports without dereferencing anything.
unsafe impl bytemuck::Zeroable for Entity {}
// SAFETY: as above, plus `Copy`.
unsafe impl bytemuck::Pod for Entity {}

impl Entity {
    /// The all-zeroes entity — never alive, because generation parity is
    /// liveness and 0 is even (§4.2, allocation contract rule 3). This is the
    /// "no entity" value for `Pod` components, where `Option<Entity>` is
    /// unavailable (niche optimization is not a layout guarantee).
    pub const NONE: Self = Self {
        index: 0,
        generation: 0,
    };

    /// Assemble a handle from its halves.
    ///
    /// For `gg-ecs`'s allocator, which lives a crate away now, and for tests.
    /// Not a capability the boundary did not already have: [`Entity::from_bits`]
    /// mints arbitrary handles too, and liveness is the allocator's answer, not
    /// the handle's.
    #[must_use]
    pub const fn new(index: u32, generation: u32) -> Self {
        Self { index, generation }
    }

    /// Dense slot this handle refers to.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// How many times that slot has turned over — odd live, even dead.
    #[must_use]
    pub const fn generation(self) -> u32 {
        self.generation
    }

    /// Whether this is [`Entity::NONE`]. Not the same question as "is it
    /// alive" — ask `Entities::is_alive` for that.
    #[must_use]
    pub const fn is_none(self) -> bool {
        self.generation == 0
    }

    /// The `u64` the state-hash protocol encodes an entity reference as
    /// (§4.2.1): generation high, index low. Both halves are committed, so a
    /// stale handle and a live one at the same index hash differently.
    ///
    /// A value encoding, never a reinterpretation of the struct's bytes — the
    /// halves are in the opposite order from memory. Anything that memcpys
    /// entities keeps [`Entity`] itself.
    #[must_use]
    pub const fn to_bits(self) -> u64 {
        ((self.generation as u64) << 32) | self.index as u64
    }

    /// Inverse of [`Entity::to_bits`].
    #[must_use]
    pub const fn from_bits(bits: u64) -> Self {
        Self {
            index: bits as u32,
            generation: (bits >> 32) as u32,
        }
    }
}

impl fmt::Debug for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_none() {
            f.write_str("Entity::NONE")
        } else {
            write!(f, "Entity({}v{})", self.index, self.generation)
        }
    }
}
