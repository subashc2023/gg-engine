//! `gg-extract` (§4.1) — the extract stage: `&World` in, flat owned arrays out.
//!
//! This is the determinism membrane of §1.4 expressed as a crate boundary. It
//! is the **one** crate permitted both `gg_math::sim` and `gg_math::render`
//! (§3), because it exists precisely to hold the conversion between them, and
//! putting that conversion inside a determinism-critical crate would trip the
//! lint that keeps every other crate honest.
//!
//! Two properties are structural rather than promised:
//!
//! - **One-way.** [`Extracted`] is owned data built from `&World`; there is no
//!   type here through which a render result could travel back into the sim,
//!   and nothing sim-side depends on this crate.
//! - **Camera-relative.** World positions are `f64` and can be planetary; the
//!   camera's position is subtracted *in `f64`* and the residue narrows to
//!   `f32` once ([`gg_math::render::camera_relative`]). A render-side absolute
//!   position therefore cannot exist by accident — and a camera 10^12 m from
//!   the origin renders without jitter, which is the whole point of paying for
//!   `f64` sim positions in the first place.
//!
//! The renderer never holds a reference into the world, which is what buys
//! trivially parallel render prep, a later render-thread split, and clean
//! interpolation — all without refactoring the sim.

#![warn(missing_docs)]

use gg_ecs::{AliasError, Component, Entity, Query, World};
use gg_math::{render, sim};

/// What a component must answer for extract to place it in render space.
///
/// Implemented by *game* components, which is why it takes `&self` rather than
/// prescribing field names: `gg-extract` should not have opinions about how a
/// game spells its transform, only about the arithmetic that crosses the
/// membrane.
pub trait SimTransform: Component {
    /// World-space position, `f64` — absolute, un-narrowed, possibly enormous.
    fn world_position(&self) -> sim::DVec3;

    /// World-space orientation. Defaults to unrotated, since plenty of things
    /// that have a position do not have a facing.
    fn orientation(&self) -> sim::DQuat {
        sim::DQuat::IDENTITY
    }
}

/// One render-space instance: an entity, where it is *relative to the camera*,
/// and how it is turned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Instance {
    /// Which entity produced it. Extract is where render-facing indices are
    /// generated (§4.2), so this is the only place the two identities meet.
    pub entity: Entity,
    /// Position relative to the camera origin, in `f32`.
    pub offset: render::Vec3,
    /// Orientation, narrowed.
    pub rotation: render::Quat,
}

/// The per-frame arrays, reused across frames.
///
/// Reused rather than rebuilt: extract runs every frame, and the allocation
/// pattern of "one fresh `Vec` per frame per array" is the kind of thing that
/// looks free in a demo and is not at scale.
#[derive(Clone, Debug)]
pub struct Extracted {
    /// Instances, in world iteration order (deterministic, §4.2).
    pub instances: Vec<Instance>,
    /// The camera origin every offset in [`Extracted::instances`] is relative
    /// to. Kept alongside so a consumer cannot pair offsets with the wrong eye.
    pub camera_origin: sim::DVec3,
}

impl Default for Extracted {
    fn default() -> Self {
        Extracted {
            instances: Vec::new(),
            // Not derived: `sim` types carry no `Default`, on purpose — a
            // zero position is a *place*, and defaulting to it silently is how
            // an unset transform reaches the origin instead of an error.
            camera_origin: sim::DVec3::ZERO,
        }
    }
}

impl Extracted {
    /// Empty, with whatever capacity previous frames earned.
    pub fn clear(&mut self, camera_origin: sim::DVec3) {
        self.instances.clear();
        self.camera_origin = camera_origin;
    }

    /// Append every entity carrying `T`, narrowed through `camera_origin`.
    ///
    /// Clears first: one call is one frame's worth of that component. Call
    /// [`Extracted::append`] to add a second component type to the same frame.
    pub fn transforms<T: SimTransform>(
        &mut self,
        world: &World,
        camera_origin: sim::DVec3,
    ) -> Result<(), AliasError> {
        self.clear(camera_origin);
        self.append::<T>(world)
    }

    /// Append without clearing — for a frame whose instances come from more
    /// than one component type.
    ///
    /// Uses [`Extracted::camera_origin`], so the eye cannot drift between the
    /// arrays of one frame.
    pub fn append<T: SimTransform>(&mut self, world: &World) -> Result<(), AliasError> {
        let query = Query::<&T>::new()?;
        let origin = self.camera_origin;
        let instances = &mut self.instances;
        // `each` visits archetypes in the world's deterministic order, so two
        // runs of the same sim produce the same array in the same order — which
        // is what lets a golden image be compared byte-for-byte at all.
        world.each_ref(&query, |entity, transform: &T| {
            instances.push(Instance {
                entity,
                offset: render::camera_relative(transform.world_position(), origin),
                rotation: render::narrow_rotation(transform.orientation()),
            });
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use gg_ecs::Component;

    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
    #[component(id = "test.body")]
    #[repr(C)]
    struct Body {
        position: sim::DVec3,
        rotation: sim::DQuat,
    }

    impl SimTransform for Body {
        fn world_position(&self) -> sim::DVec3 {
            self.position
        }
        fn orientation(&self) -> sim::DQuat {
            self.rotation
        }
    }

    fn world_with(positions: &[sim::DVec3]) -> World {
        let mut world = World::new();
        for &position in positions {
            let e = world.spawn();
            let _ = world.insert(
                e,
                Body {
                    position,
                    rotation: sim::DQuat::IDENTITY,
                },
            );
        }
        world
    }

    #[test]
    fn offsets_are_relative_to_the_camera_and_the_origin_travels_with_them() {
        let world = world_with(&[sim::DVec3::new(3.0, 0.0, 0.0), sim::DVec3::ZERO]);
        let eye = sim::DVec3::new(1.0, 2.0, 0.0);
        let mut out = Extracted::default();
        out.transforms::<Body>(&world, eye).unwrap();

        assert_eq!(out.camera_origin, eye);
        assert_eq!(out.instances.len(), 2);
        assert_eq!(out.instances[0].offset, render::Vec3::new(2.0, -2.0, 0.0));
        assert_eq!(out.instances[1].offset, render::Vec3::new(-1.0, -2.0, 0.0));
    }

    #[test]
    fn a_planetary_camera_still_resolves_metres() {
        // The membrane's reason to exist: 10^12 m out, an f32 absolute position
        // has ~65 km of resolution, and subtract-then-narrow keeps the metre.
        let far = 1.0e12;
        let world = world_with(&[sim::DVec3::new(far + 1.0, 0.0, 0.0)]);
        let mut out = Extracted::default();
        out.transforms::<Body>(&world, sim::DVec3::new(far, 0.0, 0.0))
            .unwrap();
        assert_eq!(out.instances[0].offset, render::Vec3::new(1.0, 0.0, 0.0));
        // Narrowing first is what this test is a control for.
        assert_ne!((far + 1.0) as f32 - far as f32, 1.0);
    }

    #[test]
    fn clearing_keeps_capacity_and_appending_shares_one_eye() {
        let world = world_with(&[sim::DVec3::ZERO; 4]);
        let mut out = Extracted::default();
        out.transforms::<Body>(&world, sim::DVec3::ZERO).unwrap();
        let capacity = out.instances.capacity();
        out.transforms::<Body>(&world, sim::DVec3::new(5.0, 0.0, 0.0))
            .unwrap();
        assert_eq!(out.instances.capacity(), capacity, "reused, not rebuilt");
        out.append::<Body>(&world).unwrap();
        assert_eq!(out.instances.len(), 8);
        // Every offset used the eye set by the last clear, including appended.
        assert!(
            out.instances
                .iter()
                .all(|i| i.offset == render::Vec3::new(-5.0, 0.0, 0.0))
        );
    }
}
