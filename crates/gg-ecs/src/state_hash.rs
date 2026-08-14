//! The state-hash protocol (§4.2.1).
//!
//! `Pod` was solving padding (hazard 4), but `Pod` is not a deterministic wire
//! format, and the gap between the two is exactly where a portable hash quietly
//! stops being one. [`StateHash`] closes it: an explicit field-by-field
//! encoding in declaration order, with every primitive's representation pinned
//! by [`StateHasher`] rather than by whatever the target's memory layout
//! happens to be.
//!
//! # Rejection is by absence
//!
//! The forbidden field types of §4.2.1 — `usize`/`isize`, pointers, references,
//! bare enums, every `gg_math::render` type, unordered `std` containers —
//! are rejected because **no impl exists for them**, so a field of such a type
//! fails to compile. That is total by construction: a type nobody thought to
//! ban is still rejected unless someone deliberately wrote an impl with a
//! documented encoding. The derive additionally intercepts the four most likely
//! mistakes syntactically so the error names the field and the reason instead
//! of reading `the trait bound is not satisfied`.
//!
//! `Vec<T>`, `String` and `&str` *are* implemented, despite §4.2.1's
//! "any `std` container" phrasing: the same section requires that "a growable
//! list hashes as length-then-elements perfectly deterministically", which is
//! the encoding used here. The property that matters is that iteration order is
//! determined by content, which is true of `Vec` and false of `HashMap` —
//! and `HashMap`/`HashSet` have no impl, on top of already being clippy-banned
//! workspace-wide (§3).

use gg_math::sim;

use crate::entity::Entity;
use crate::hash::{ComponentId, StateHasher};

/// Types with a defined, platform-independent encoding for the canonical hash.
///
/// Derive it (`#[derive(StateHash)]`) rather than writing it by hand. A manual
/// impl is a new wire format entry and belongs with the others in this file,
/// where the encodings can be reviewed together.
#[diagnostic::on_unimplemented(
    message = "`{Self}` has no defined encoding for the canonical state hash (§4.2.1)",
    label = "this field's type cannot be hashed state",
    note = "hashed state must encode identically on every supported architecture, so a type \
            reaches the protocol only by having an impl deliberately written for it",
    note = "bare enums: use the newtyped-integer pattern — a #[repr(transparent)] struct over a \
            u8/u32 with validated accessors",
    note = "`gg_math::render` types and `usize`/`isize`: use `gg_math::sim` and an explicit width",
    note = "`HashMap`/`HashSet`: iteration order is randomly seeded (hazard 3) — use an ordered \
            container"
)]
pub trait StateHash {
    /// Absorb `self` into `h`. Implementations must be a pure function of the
    /// value: no addresses, no lengths in native width, no iteration order that
    /// content does not determine.
    fn state_hash(&self, h: &mut StateHasher);
}

macro_rules! impl_primitive {
    ($($t:ty => $method:ident),* $(,)?) => {
        $(
            impl StateHash for $t {
                fn state_hash(&self, h: &mut StateHasher) {
                    h.$method(*self);
                }
            }
        )*
    };
}

// Deliberately absent: usize, isize. Their width is the platform's, which is
// the definition of the thing this protocol refuses (§4.2.1).
impl_primitive! {
    u8 => u8, u16 => u16, u32 => u32, u64 => u64, u128 => u128,
    i8 => i8, i16 => i16, i32 => i32, i64 => i64, i128 => i128,
    f32 => f32, f64 => f64,
    bool => bool,
}

impl StateHash for Entity {
    fn state_hash(&self, h: &mut StateHasher) {
        h.u64(self.to_bits());
    }
}

impl StateHash for ComponentId {
    fn state_hash(&self, h: &mut StateHasher) {
        h.u64(self.get());
    }
}

impl<T: StateHash, const N: usize> StateHash for [T; N] {
    /// Index order. `N` is not absorbed — it is part of the type, so two arrays
    /// of different length are different fields, not different values of one.
    fn state_hash(&self, h: &mut StateHasher) {
        for item in self {
            item.state_hash(h);
        }
    }
}

impl<T: StateHash> StateHash for [T] {
    /// Length-then-elements. The length *is* absorbed here, unlike the array
    /// case, because it varies at runtime.
    fn state_hash(&self, h: &mut StateHasher) {
        h.u64(self.len() as u64);
        for item in self {
            item.state_hash(h);
        }
    }
}

impl<T: StateHash> StateHash for Vec<T> {
    fn state_hash(&self, h: &mut StateHasher) {
        self.as_slice().state_hash(h);
    }
}

impl<T: StateHash> StateHash for Option<T> {
    /// Tag byte, then the payload if any. The tag is absorbed for `None` too,
    /// so `None` and `Some(0u8)` cannot collide.
    fn state_hash(&self, h: &mut StateHasher) {
        match self {
            None => h.u8(0),
            Some(v) => {
                h.u8(1);
                v.state_hash(h);
            }
        }
    }
}

impl StateHash for str {
    fn state_hash(&self, h: &mut StateHasher) {
        h.str(self);
    }
}

impl StateHash for String {
    fn state_hash(&self, h: &mut StateHasher) {
        h.str(self);
    }
}

/// `gg_math::sim` types, field by field in declaration order.
///
/// Written out rather than blanket-implemented over `Pod`: a blanket impl would
/// hash padding and defeat the whole point (§4.2.1 hazard 4), and it would also
/// silently accept every future `Pod` type without anyone choosing an encoding.
macro_rules! impl_sim {
    ($($t:ty { $($field:ident),* $(,)? })*) => {
        $(
            impl StateHash for $t {
                fn state_hash(&self, h: &mut StateHasher) {
                    $( self.$field.state_hash(h); )*
                }
            }
        )*
    };
}

impl_sim! {
    sim::Vec2 { x, y }
    sim::Vec3 { x, y, z }
    sim::Vec4 { x, y, z, w }
    sim::Quat { x, y, z, w }
    sim::Mat3 { x_axis, y_axis, z_axis }
    sim::Mat4 { x_axis, y_axis, z_axis, w_axis }
    sim::DVec2 { x, y }
    sim::DVec3 { x, y, z }
    sim::DVec4 { x, y, z, w }
    sim::DQuat { x, y, z, w }
    sim::DMat3 { x_axis, y_axis, z_axis }
    sim::DMat4 { x_axis, y_axis, z_axis, w_axis }
    // The analytic regime's whole state (§6 M38 item 7). Element order is the
    // declaration's, so an `Orbit` hashes as the seven doubles it is — and it
    // is here because a body on rails carries its conic as a component, which
    // demo 13 is the first to do.
    sim::Orbit {
        semi_major,
        eccentricity,
        inclination,
        ascending_node,
        argument_of_periapsis,
        mean_anomaly,
        mu,
    }
}

/// The generator's stream position, which is the whole of its state (§6 M18).
///
/// Not in [`impl_sim`] because the field is private: an `Rng` says its state
/// through `to_bits` rather than exposing a counter that is not an output. It
/// is in the protocol at all because a game that draws pieces from a generator
/// the hash does not cover is a game whose replay diverges on the first draw
/// with nothing to report it.
impl StateHash for sim::Rng {
    fn state_hash(&self, h: &mut StateHasher) {
        h.u64(self.to_bits());
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn hash_of(v: &impl StateHash) -> crate::CanonicalHash {
        let mut h = StateHasher::canonical();
        v.state_hash(&mut h);
        h.finish_canonical()
    }

    #[test]
    fn slices_commit_their_length() {
        // The encoding bug that would surface as a phantom divergence months
        // later: [[1],[2]] and [[1,2]] flattening to the same absorption.
        let a: Vec<Vec<u8>> = vec![vec![1], vec![2]];
        let b: Vec<Vec<u8>> = vec![vec![1, 2]];
        assert_ne!(hash_of(&a), hash_of(&b));
    }

    #[test]
    fn option_tag_separates_none_from_zero() {
        assert_ne!(hash_of(&None::<u8>), hash_of(&Some(0u8)));
    }

    #[test]
    fn sim_types_hash_field_by_field_not_as_raw_memory() {
        // Mat3 is three Vec3s; hashing it must equal hashing the nine scalars,
        // which is what makes the encoding independent of any padding a future
        // layout change might introduce.
        let m = sim::Mat3 {
            x_axis: sim::Vec3::new(1.0, 2.0, 3.0),
            y_axis: sim::Vec3::new(4.0, 5.0, 6.0),
            z_axis: sim::Vec3::new(7.0, 8.0, 9.0),
        };
        let mut a = StateHasher::canonical();
        m.state_hash(&mut a);
        let mut b = StateHasher::canonical();
        for v in 1..=9 {
            b.f32(v as f32);
        }
        assert_eq!(a.finish_canonical(), b.finish_canonical());
    }

    #[test]
    fn entities_commit_generation_not_only_index() {
        let live = Entity::from_bits((1u64 << 32) | 7);
        let stale = Entity::from_bits((3u64 << 32) | 7);
        assert_ne!(hash_of(&live), hash_of(&stale));
    }
}
