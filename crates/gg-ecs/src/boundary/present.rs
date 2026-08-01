//! What a game tells the host to draw — the render half of the §4.2.2 boundary.
//!
//! The host holds no game types. It cannot instantiate a query over `Cube`, and
//! §3's deny pin means a type both sides name can live in exactly three crates —
//! `gg-abi` (below [`Component`](crate::Component), so it cannot carry persisted
//! identity), `gg-math` (types, no identity either), and this one. So the render
//! protocol is *two ordinary components* defined here, declared by the game like
//! any other, and read by the host through a typed query.
//!
//! The consequence worth naming: **nothing about how the game looks lives in the
//! host.** A system fills these in from whatever the game's own components say,
//! which puts colour, size and pose inside the dylib — reloadable, and editable
//! while someone is playing. The host's renderer knows one shape and one colour
//! channel, and that is the whole of its opinion.
//!
//! Layout agreement is not asserted here because it is already checked where it
//! matters: the schema hash covers every field's name, type token, offset and
//! size plus the struct's size and alignment, and `World::adopt` refuses a
//! declared component whose schema disagrees with the one already registered
//! (§4.2.2). Two compilations that laid these out differently would be refused
//! by name at load rather than drawing garbage.

use gg_math::sim;

use crate::Component;

/// One box to draw, in world space.
///
/// A box because M5's renderer draws exactly one primitive (§6 M5, deliberately
/// ugly): a cube, a floor slab and a tracer are all this type with different
/// [`half_extent`](Renderable::half_extent)s.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "gg.renderable")]
#[repr(C)]
pub struct Renderable {
    /// World-space centre, `f64` and un-narrowed — `gg-extract` is what makes it
    /// camera-relative `f32` (§1.4).
    pub position: sim::DVec3,
    /// Orientation. Identity for anything axis-aligned, which is most of it.
    pub rotation: sim::DQuat,
    /// Half-extent per axis, metres, *before* rotation. Non-uniform on purpose:
    /// a beam is a long thin box, and one primitive that stretches beats a
    /// second primitive.
    pub half_extent: sim::Vec3,
    /// `0x00RRGGBB`, sRGB. An integer rather than three floats: it is a colour
    /// the game picked, not a value anything computes with.
    pub color: u32,
}

impl Renderable {
    /// An axis-aligned box. The common case, spelled once so game code does not
    /// repeat `DQuat::IDENTITY`.
    #[must_use]
    pub fn boxed(position: sim::DVec3, half_extent: sim::Vec3, color: u32) -> Self {
        Renderable {
            position,
            rotation: sim::DQuat::IDENTITY,
            half_extent,
            color,
        }
    }
}

/// Where the game is looked at from.
///
/// Yaw and pitch rather than a quaternion: the host builds a fly camera's basis
/// from them in one order (Y then X, never roll), and a quaternion here would
/// make "which way is up" the game's problem to get right per frame.
///
/// The first one the host finds wins, in world iteration order. A game with no
/// eye renders from the origin — visible, wrong, and obviously so, which beats
/// a black screen that could mean anything.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "gg.eye")]
#[repr(C)]
pub struct Eye {
    /// World-space eye position.
    pub position: sim::DVec3,
    /// Rotation about +Y, radians.
    pub yaw: f32,
    /// Rotation about the camera's right axis, radians.
    pub pitch: f32,
}

impl Eye {
    /// Unrotated, at the world origin — what a game that declares no eye is
    /// rendered from.
    pub const ORIGIN: Eye = Eye {
        position: sim::DVec3::ZERO,
        yaw: 0.0,
        pitch: 0.0,
    };

    /// The eye a host renders from: the first one in world iteration order, or
    /// [`Eye::ORIGIN`].
    ///
    /// Visible and obviously wrong beats a black screen, which could mean the
    /// game declared no eye, spawned nothing, or crashed.
    ///
    /// # Errors
    ///
    /// If the world refuses the query, which one read alone cannot cause.
    pub fn of(world: &crate::World) -> Result<Eye, crate::AliasError> {
        let query = crate::Query::<&Eye>::new()?;
        let mut found = None;
        world.each_ref(&query, |_, eye: &Eye| found = found.or(Some(*eye)));
        Ok(found.unwrap_or(Eye::ORIGIN))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_protocol_types_are_flat_and_padding_free() {
        // `Pod` already refuses padding; these pin the numbers so a field added
        // to either one is a visible edit rather than a silent layout move.
        assert_eq!(size_of::<Renderable>(), 72);
        assert_eq!(align_of::<Renderable>(), 8);
        assert_eq!(size_of::<Eye>(), 32);
        assert_eq!(align_of::<Eye>(), 8);
    }

    #[test]
    fn a_boxed_renderable_is_unrotated() {
        let r = Renderable::boxed(sim::DVec3::ZERO, sim::Vec3::splat(0.5), 0x00ff_8000);
        assert_eq!(r.rotation, sim::DQuat::IDENTITY);
        assert_eq!(r.color, 0x00ff_8000);
    }
}
