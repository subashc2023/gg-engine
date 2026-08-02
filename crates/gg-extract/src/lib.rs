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

    /// Half-extent per axis, metres, before rotation. `f32` on both sides of
    /// the membrane already: a size is a local quantity and never gains
    /// planetary magnitude, which is the only thing `f64` buys here.
    fn half_extent(&self) -> sim::Vec3 {
        sim::Vec3::splat(0.5)
    }

    /// `0x00RRGGBB`, sRGB. Carried across because the game picks it and the
    /// renderer must not — a host with an opinion about colour is a host the
    /// game cannot restyle without a rebuild of the *engine* (§6 M5).
    fn color(&self) -> u32 {
        0x00ff_ffff
    }

    /// The pack asset this instance draws, or 0 for none (§4.6). Zero for
    /// anything the host draws as a box, which is why it defaults.
    fn asset(&self) -> u64 {
        0
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
    /// Half-extent per axis, metres — or, for an instance that names an asset,
    /// the scale to draw it at.
    pub half_extent: render::Vec3,
    /// `0x00RRGGBB`, sRGB.
    pub color: u32,
    /// The pack asset to draw, or 0 (§4.6). Always 0 in
    /// [`Extracted::instances`], never 0 in [`Extracted::models`] — the two
    /// arrays are exactly this field's two cases, split so the renderer picks a
    /// pipeline once per array rather than once per instance.
    pub asset: u64,
}

/// One mesh placed inside a scene asset, in that scene's own space (§4.6).
///
/// `f64` translation because the pack stores it that way and narrowing it
/// belongs *here*, on the far side of a camera origin — a scene's nodes are
/// small in practice and enormous in principle, and a format that decided which
/// in advance would have baked the membrane's failure into the file.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    /// The mesh asset this node draws.
    pub mesh: u64,
    /// Offset from the scene's origin.
    pub translation: sim::DVec3,
    /// Orientation within the scene.
    pub rotation: sim::DQuat,
    /// Scale within the scene.
    pub scale: sim::Vec3,
}

/// What a scene asset expands to, asked of whoever holds the pack.
///
/// A trait rather than a dependency: resolving an id is `gg-assets`' business
/// and mmapping a file has no place in the crate that owns the §1.4 membrane.
/// What this crate contributes is the one thing only it may do — compose the
/// placement with the game's own transform *in `f64`* and narrow the result
/// once, so a scene 10^12 m out draws without jitter exactly as a box does.
pub trait Scenes {
    /// Call `visit` once per mesh `asset` places. An asset that is not a scene
    /// — a mesh, or one the pack does not contain — places nothing, and the
    /// caller draws it as itself.
    fn expand(&self, asset: u64, visit: &mut dyn FnMut(Placement));
}

/// Expands nothing: every model is drawn as itself. What a host with no pack
/// passes, and what the box-only tests use.
impl Scenes for () {
    fn expand(&self, _asset: u64, _visit: &mut dyn FnMut(Placement)) {}
}

/// The per-frame arrays, reused across frames.
///
/// Reused rather than rebuilt: extract runs every frame, and the allocation
/// pattern of "one fresh `Vec` per frame per array" is the kind of thing that
/// looks free in a demo and is not at scale.
#[derive(Clone, Debug)]
pub struct Extracted {
    /// Instances the host draws as boxes, in world iteration order
    /// (deterministic, §4.2).
    pub instances: Vec<Instance>,
    /// Instances that name pack content, same order and same narrowing.
    pub models: Vec<Instance>,
    /// The camera origin every offset in both arrays is relative to. Kept
    /// alongside so a consumer cannot pair offsets with the wrong eye.
    pub camera_origin: sim::DVec3,
}

impl Default for Extracted {
    fn default() -> Self {
        Extracted {
            instances: Vec::new(),
            models: Vec::new(),
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
        self.models.clear();
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
        let origin = self.camera_origin;
        collect::<T>(world, origin, &mut self.instances)
    }

    /// Append every entity carrying `T` to [`Extracted::models`] instead — the
    /// ones whose [`SimTransform::asset`] names pack content — expanding any
    /// that name a scene into one instance per node.
    ///
    /// Never clears: models are a second component type over the same frame, so
    /// the eye is already fixed by whoever filled the first array.
    ///
    /// One entity can therefore produce hundreds of instances. That is the
    /// point: what leaves here is flat and camera-relative, so the renderer
    /// resolves a mesh id and nothing else.
    pub fn append_models<T: SimTransform>(
        &mut self,
        world: &World,
        scenes: &dyn Scenes,
    ) -> Result<(), AliasError> {
        let query = Query::<&T>::new()?;
        let origin = self.camera_origin;
        let models = &mut self.models;
        world.each_ref(&query, |entity, transform: &T| {
            let placed = place(entity, transform, origin);
            let mut expanded = false;
            scenes.expand(transform.asset(), &mut |node| {
                expanded = true;
                models.push(compose(&placed, transform, origin, node));
            });
            if !expanded {
                models.push(placed);
            }
        });
        Ok(())
    }
}

/// One entity's own transform, narrowed.
fn place<T: SimTransform>(entity: Entity, transform: &T, origin: sim::DVec3) -> Instance {
    let half = transform.half_extent();
    Instance {
        entity,
        offset: render::camera_relative(transform.world_position(), origin),
        rotation: render::narrow_rotation(transform.orientation()),
        half_extent: render::Vec3::new(half.x, half.y, half.z),
        color: transform.color(),
        asset: transform.asset(),
    }
}

/// One scene node under an entity's transform. The composition is `f64`
/// throughout and narrows exactly once, at the same `camera_relative` a
/// standalone instance goes through — a node's offset added to the eye-relative
/// residue in `f32` would reintroduce the jitter the membrane exists to remove.
fn compose<T: SimTransform>(
    placed: &Instance,
    transform: &T,
    origin: sim::DVec3,
    node: Placement,
) -> Instance {
    let model_scale = transform.half_extent();
    let scaled = sim::DVec3::new(
        node.translation.x * f64::from(model_scale.x),
        node.translation.y * f64::from(model_scale.y),
        node.translation.z * f64::from(model_scale.z),
    );
    let rotation = transform.orientation();
    let world = transform.world_position() + rotation.rotate(scaled);
    Instance {
        entity: placed.entity,
        offset: render::camera_relative(world, origin),
        rotation: render::narrow_rotation(rotation.mul(node.rotation)),
        half_extent: render::Vec3::new(
            model_scale.x * node.scale.x,
            model_scale.y * node.scale.y,
            model_scale.z * node.scale.z,
        ),
        color: placed.color,
        asset: node.mesh,
    }
}

fn collect<T: SimTransform>(
    world: &World,
    origin: sim::DVec3,
    into: &mut Vec<Instance>,
) -> Result<(), AliasError> {
    let query = Query::<&T>::new()?;
    // `each` visits archetypes in the world's deterministic order, so two runs
    // of the same sim produce the same array in the same order — which is what
    // lets a golden image be compared byte-for-byte at all.
    world.each_ref(&query, |entity, transform: &T| {
        into.push(place(entity, transform, origin));
    });
    Ok(())
}

/// The §4.5 v0 render protocol as an extract source (§4.2.2): the host draws
/// [`Renderable`] and nothing else, so this is the one impl that ships.
///
/// A game's own component *may* implement [`SimTransform`] — demo 02's does —
/// but a component the host has no Rust type for cannot, which is why the
/// protocol exists at all.
impl SimTransform for gg_ecs::boundary::Renderable {
    fn world_position(&self) -> sim::DVec3 {
        self.position
    }

    fn orientation(&self) -> sim::DQuat {
        self.rotation
    }

    fn half_extent(&self) -> sim::Vec3 {
        self.half_extent
    }

    fn color(&self) -> u32 {
        self.color
    }
}

/// The same protocol's pack half (§4.6): a name, and where to put what it
/// resolves to. `half_extent` is the model's scale — one field, because the
/// arithmetic crossing the membrane is identical and a second one would be a
/// second chance to narrow it differently.
impl SimTransform for gg_ecs::boundary::Model {
    fn world_position(&self) -> sim::DVec3 {
        self.position
    }

    fn orientation(&self) -> sim::DQuat {
        self.rotation
    }

    fn half_extent(&self) -> sim::Vec3 {
        self.scale
    }

    fn color(&self) -> u32 {
        self.tint
    }

    fn asset(&self) -> u64 {
        self.asset
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
    fn the_render_protocol_carries_size_and_colour_across() {
        use gg_ecs::boundary::Renderable;
        let mut world = World::new();
        world.register::<Renderable>().unwrap();
        let e = world.spawn();
        let mut beam = Renderable::boxed(
            sim::DVec3::new(0.0, 0.0, -2.0),
            sim::Vec3::new(0.05, 0.05, 3.0),
            0x0033_66ff,
        );
        beam.rotation = sim::DQuat::from_axis_angle(sim::DVec3::Y, core::f64::consts::FRAC_PI_2);
        world.insert(e, beam).unwrap();

        let mut out = Extracted::default();
        out.transforms::<Renderable>(&world, sim::DVec3::ZERO)
            .unwrap();
        let instance = out.instances[0];
        assert_eq!(instance.half_extent, render::Vec3::new(0.05, 0.05, 3.0));
        assert_eq!(instance.color, 0x0033_66ff);
        // The defaults exist for game components that predate the protocol, and
        // must not leak into a type that answers for itself.
        assert_ne!(instance.rotation, render::Quat::IDENTITY);
    }

    /// A scene of two nodes, one metre apart along +X.
    struct TwoNodes;

    const SCENE: u64 = 0xabc;

    impl Scenes for TwoNodes {
        fn expand(&self, asset: u64, visit: &mut dyn FnMut(Placement)) {
            if asset != SCENE {
                return;
            }
            for (index, x) in [0.0, 1.0].into_iter().enumerate() {
                visit(Placement {
                    mesh: 100 + index as u64,
                    translation: sim::DVec3::new(x, 0.0, 0.0),
                    rotation: sim::DQuat::IDENTITY,
                    scale: sim::Vec3::splat(1.0),
                });
            }
        }
    }

    fn model_world(asset: u64, position: sim::DVec3, scale: sim::Vec3) -> World {
        use gg_ecs::boundary::Model;
        let mut world = World::new();
        world.register::<Model>().unwrap();
        let e = world.spawn();
        world
            .insert(
                e,
                Model {
                    asset,
                    scale,
                    ..Model::at("", position)
                },
            )
            .unwrap();
        world
    }

    #[test]
    fn a_model_naming_a_scene_becomes_one_instance_per_node() {
        use gg_ecs::boundary::Model;
        let world = model_world(
            SCENE,
            sim::DVec3::new(10.0, 0.0, 0.0),
            sim::Vec3::splat(2.0),
        );
        let mut out = Extracted::default();
        out.clear(sim::DVec3::ZERO);
        out.append_models::<Model>(&world, &TwoNodes).unwrap();

        assert!(out.instances.is_empty(), "models are not boxes");
        assert_eq!(out.models.len(), 2);
        // The model's scale multiplies the node's *offset*, not only its size:
        // scaling a scene by two moves its pieces apart as well as growing them.
        assert_eq!(out.models[0].offset, render::Vec3::new(10.0, 0.0, 0.0));
        assert_eq!(out.models[1].offset, render::Vec3::new(12.0, 0.0, 0.0));
        assert_eq!(out.models[0].half_extent, render::Vec3::splat(2.0));
        // What leaves here names meshes, never the scene the game named.
        assert_eq!(out.models[0].asset, 100);
        assert_eq!(out.models[1].asset, 101);
    }

    #[test]
    fn a_scene_at_planetary_distance_narrows_once_and_keeps_its_nodes_apart() {
        // The whole reason node translations stay `f64` in the pack. Composing
        // in f32 would round both nodes onto the same 65 km-resolution point.
        use gg_ecs::boundary::Model;
        let far = 1.0e12;
        let world = model_world(SCENE, sim::DVec3::new(far, 0.0, 0.0), sim::Vec3::splat(1.0));
        let mut out = Extracted::default();
        out.clear(sim::DVec3::new(far, 0.0, 0.0));
        out.append_models::<Model>(&world, &TwoNodes).unwrap();
        assert_eq!(out.models[0].offset, render::Vec3::ZERO);
        assert_eq!(out.models[1].offset, render::Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn a_model_naming_something_that_is_not_a_scene_is_drawn_as_itself() {
        use gg_ecs::boundary::Model;
        let world = model_world(77, sim::DVec3::ZERO, sim::Vec3::splat(1.0));
        let mut out = Extracted::default();
        out.clear(sim::DVec3::ZERO);
        out.append_models::<Model>(&world, &TwoNodes).unwrap();
        assert_eq!(out.models.len(), 1);
        assert_eq!(out.models[0].asset, 77, "a mesh id passes straight through");

        // And with no pack at all, every model is itself.
        let mut bare = Extracted::default();
        bare.clear(sim::DVec3::ZERO);
        bare.append_models::<Model>(&world, &()).unwrap();
        assert_eq!(bare.models.len(), 1);
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
