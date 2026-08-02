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

mod frustum;

pub use frustum::Frustum;

/// The light-kind discriminants, re-exported so a consumer downstream of the
/// membrane can read [`ExtractedLight::kind`] without linking `gg-ecs` — which
/// `gg-render` does not, and should not have to for two constants.
pub use gg_ecs::boundary::light;

use gg_ecs::{AliasError, Component, Entity, Query, World};
use gg_math::{render, sim};
use rayon::prelude::*;

/// Rows per parallel chunk.
///
/// Chunks are cut *inside* an archetype, not per archetype: ten thousand
/// entities that share a component set are one archetype, so per-archetype
/// parallelism would be none at all in the case that needs it (§6 M10). Large
/// enough that rayon's per-task cost is noise against 512 narrowings, small
/// enough that a 10k world still splits across every core on the desk.
const CHUNK_ROWS: usize = 512;

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

/// The bounding radius of a box of `half_extent` — its corner distance, which
/// is what makes the sphere test rotation-invariant.
fn box_radius(half_extent: sim::Vec3) -> f32 {
    render::Vec3::new(half_extent.x, half_extent.y, half_extent.z).length()
}

/// The largest axis of a scale, for turning a local radius into a world one.
/// The largest rather than the average: a mesh scaled 1x1x10 bounds to a sphere
/// of ten times its radius, and anything smaller would cull what it should keep.
fn max_scale(scale: sim::Vec3) -> f32 {
    scale.x.abs().max(scale.y.abs()).max(scale.z.abs())
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
    /// World-space bounding radius about [`offset`](Instance::offset), metres.
    /// Carried rather than recomputed downstream because the culler already
    /// needed it and a second derivation is a second chance to disagree.
    pub radius: f32,
}

/// Directional lights one frame may carry. Small on purpose: a scene has a sun,
/// occasionally a fill, and a game that wants a third is describing something
/// the shading model would be the wrong place to fix.
pub const MAX_DIRECTIONAL: usize = 4;

/// Point lights one frame may carry.
///
/// A forward pass loops over every one of these per fragment, so the cap is a
/// per-pixel cost and not a memory one — which is why it is a small number
/// rather than a large one, and why raising it is a measurement rather than a
/// preference. Clustered assignment is the answer that makes it large, and it
/// is P1.
pub const MAX_POINT: usize = 32;

/// One light in render space: camera-relative, narrowed, ready to shade with.
///
/// Colour stays `0x00RRGGBB` rather than becoming a linear triple here. Extract
/// carries what the game said and converts geometry, not colour — the sRGB
/// decode belongs beside the one that already happens to every tint, in the
/// renderer (§4.5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtractedLight {
    /// Position relative to the camera origin. Zero for a directional light,
    /// which has no position to be relative to.
    pub offset: render::Vec3,
    /// The direction the light travels, unit length. Zero for a point light.
    pub direction: render::Vec3,
    /// `0x00RRGGBB`, sRGB.
    pub color: u32,
    /// Linear radiance multiplier.
    pub intensity: f32,
    /// Metres at which a point light reaches exactly zero. Zero for a
    /// directional one, which never does.
    pub range: f32,
    /// One of [`gg_ecs::boundary::light`]'s constants.
    pub kind: u32,
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
    /// The mesh's bounding radius in the scene's own space, metres — the pack
    /// computed it at import (§4.6), because a mesh's extent is content and
    /// only whoever holds the file knows it.
    pub radius: f32,
}

/// What a scene asset expands to, asked of whoever holds the pack.
///
/// A trait rather than a dependency: resolving an id is `gg-assets`' business
/// and mmapping a file has no place in the crate that owns the §1.4 membrane.
/// What this crate contributes is the one thing only it may do — compose the
/// placement with the game's own transform *in `f64`* and narrow the result
/// once, so a scene 10^12 m out draws without jitter exactly as a box does.
/// `Sync` because extract expands scenes from several threads at once, and a
/// pack is a read-only mapping — the one shape that makes that free.
pub trait Scenes: Sync {
    /// Call `visit` once per mesh `asset` places. An asset that is not a scene
    /// — a mesh, or one the pack does not contain — places nothing, and the
    /// caller draws it as itself.
    fn expand(&self, asset: u64, visit: &mut dyn FnMut(Placement));

    /// `asset`'s bounding radius in its own space, or `None` when the pack does
    /// not know it yet.
    ///
    /// `None` means *do not cull*, which is the safe direction: an asset that
    /// vanishes while it streams in is a visible bug, and one drawn a few
    /// frames longer than it had to be is a cost.
    fn radius(&self, asset: u64) -> Option<f32>;
}

/// Expands nothing and knows no bounds: every model is drawn as itself and
/// never culled. What a host with no pack passes, and what the box-only tests
/// use.
impl Scenes for () {
    fn expand(&self, _asset: u64, _visit: &mut dyn FnMut(Placement)) {}

    fn radius(&self, _asset: u64) -> Option<f32> {
        None
    }
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
    /// What this frame culled against. Kept for the same reason the origin is:
    /// the arrays are only meaningful paired with the eye that produced them.
    pub frustum: Frustum,
    /// Instances the frustum rejected this frame, across both arrays. The
    /// number worth a line on the overlay — "10k entities, 400 drawn" is the
    /// claim §6 M10 makes, and an unmeasured culler is one that might be
    /// keeping everything.
    pub culled: usize,
    /// This frame's lights, directional first (§6 M11). Never more than
    /// [`MAX_DIRECTIONAL`] + [`MAX_POINT`].
    pub lights: Vec<ExtractedLight>,
    /// Lights this frame had and could not carry, because a cap was reached.
    /// Counted rather than logged: a corner of a room going dark is the symptom,
    /// and a number on the overlay is what turns it into a diagnosis.
    pub lights_dropped: usize,
    /// Per-chunk staging for the parallel pass, kept so a frame allocates
    /// nothing. Never read by a consumer: [`gather`] is what turns it into the
    /// arrays above, in chunk order.
    scratch: Vec<Chunk>,
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
            frustum: Frustum::UNBOUNDED,
            culled: 0,
            lights: Vec::new(),
            lights_dropped: 0,
            scratch: Vec::new(),
        }
    }
}

impl Extracted {
    /// Empty, with whatever capacity previous frames earned, and pointed at a
    /// new eye and frustum.
    ///
    /// Both are arguments rather than settable fields so that forgetting one is
    /// a compile error. A caller that does not cull passes
    /// [`Frustum::UNBOUNDED`], which is a decision spelled out loud — silently
    /// disabling the culler yields a correct picture and a wrong frame time,
    /// which is the hardest kind of regression to notice.
    pub fn clear(&mut self, camera_origin: sim::DVec3, frustum: Frustum) {
        self.instances.clear();
        self.models.clear();
        self.camera_origin = camera_origin;
        self.frustum = frustum;
        self.culled = 0;
        self.lights.clear();
        self.lights_dropped = 0;
    }

    /// Append every entity carrying `T`, narrowed through `camera_origin` and
    /// culled against `frustum`.
    ///
    /// Clears first: one call is one frame's worth of that component. Call
    /// [`Extracted::append`] to add a second component type to the same frame.
    pub fn transforms<T: SimTransform>(
        &mut self,
        world: &World,
        camera_origin: sim::DVec3,
        frustum: Frustum,
    ) -> Result<(), AliasError> {
        self.clear(camera_origin, frustum);
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
        let frustum = self.frustum;
        let chunks = chunks_of::<T>(world, &query);
        let scratch = fit(&mut self.scratch, chunks.len());
        scratch
            .par_iter_mut()
            .zip(chunks.par_iter())
            .for_each(|(chunk, (entities, rows))| {
                for (&entity, transform) in entities.iter().zip(rows.iter()) {
                    let radius = box_radius(transform.half_extent());
                    chunk.keep(place(entity, transform, origin, radius), frustum);
                }
            });
        gather(scratch, &mut self.instances, &mut self.culled);
        Ok(())
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
        let frustum = self.frustum;
        let chunks = chunks_of::<T>(world, &query);
        let scratch = fit(&mut self.scratch, chunks.len());
        scratch
            .par_iter_mut()
            .zip(chunks.par_iter())
            .for_each(|(chunk, (entities, rows))| {
                for (&entity, transform) in entities.iter().zip(rows.iter()) {
                    let asset = transform.asset();
                    let scale = max_scale(transform.half_extent());
                    // The entity's own radius, for the mesh case. A scene's
                    // nodes each carry their own and never consult this.
                    let own = scenes
                        .radius(asset)
                        .map_or(f32::INFINITY, |radius| radius * scale);
                    let placed = place(entity, transform, origin, own);
                    let mut expanded = false;
                    scenes.expand(asset, &mut |node| {
                        expanded = true;
                        // Culled per node, not per entity: a scene is expanded
                        // exactly so the half of a building behind the camera
                        // costs nothing, and testing the whole scene's bound
                        // would hand that back.
                        chunk.keep(compose(&placed, transform, origin, node), frustum);
                    });
                    if !expanded {
                        chunk.keep(placed, frustum);
                    }
                }
            });
        gather(scratch, &mut self.models, &mut self.culled);
        Ok(())
    }

    /// Fill [`Extracted::lights`] from the render protocol's
    /// [`Light`](gg_ecs::boundary::Light) (§6 M11).
    ///
    /// Concrete rather than generic, unlike [`Extracted::append`]: a game with
    /// its own lamp component writes a system that fills in `Light`, which is
    /// what the protocol is for. A trait here would be surface added for a
    /// caller that does not exist.
    ///
    /// Serial, and that is not an oversight — there are tens of lights where
    /// there are ten thousand instances, so rayon's per-task cost would be the
    /// whole of the work. It also keeps this the one place order comes from.
    ///
    /// # Selection, stated
    ///
    /// - **Directional lights are never culled** and are kept in world order up
    ///   to [`MAX_DIRECTIONAL`]. They have no position to test.
    /// - **Point lights are culled by their own range**, through the same
    ///   frustum every instance goes through: a light whose sphere of influence
    ///   misses the view lights nothing in it. That is exact rather than
    ///   heuristic, which is why [`Light::range`](gg_ecs::boundary::Light::range)
    ///   is defined as the distance the falloff *reaches zero* at.
    /// - Of the survivors, the **nearest [`MAX_POINT`]** are kept. Nearest and
    ///   not brightest: brightest would need the shading model here, and a
    ///   ranking that disagrees with the one the fragment shader would compute
    ///   is worse than a simple rule stated out loud.
    ///
    /// # Errors
    ///
    /// If the world refuses the query, which one read alone cannot cause.
    pub fn append_lights(&mut self, world: &World) -> Result<(), AliasError> {
        use gg_ecs::boundary::{Light, light};

        let query = Query::<&Light>::new()?;
        let origin = self.camera_origin;
        let frustum = self.frustum;
        // Two passes over a handful of rows, because directional lights go
        // first in the output and the point ones need ranking before they can
        // be truncated. Collected rather than streamed for the same reason.
        let mut suns = Vec::new();
        let mut points: Vec<(f32, ExtractedLight)> = Vec::new();
        world.each_ref(&query, |_, light: &Light| match light.kind {
            light::DIRECTIONAL => suns.push(ExtractedLight {
                offset: render::Vec3::ZERO,
                direction: render::to_render(light.direction).normalize_or_zero(),
                color: light.color,
                intensity: light.intensity,
                range: 0.0,
                kind: light.kind,
            }),
            light::POINT => {
                let offset = render::camera_relative(light.position, origin);
                if !frustum.contains(offset, light.range) {
                    return;
                }
                points.push((
                    offset.length_squared(),
                    ExtractedLight {
                        offset,
                        direction: render::Vec3::ZERO,
                        color: light.color,
                        intensity: light.intensity,
                        range: light.range,
                        kind: light.kind,
                    },
                ));
            }
            // A kind this build does not know shades nothing. Silent rather
            // than an error: a dylib built against a later boundary is a
            // reload away from being right, and a frame is not the place to
            // fail over it (§4.2.2).
            _ => {}
        });

        // Stable, and keyed on the squared distance rather than the distance:
        // the ordering is the same and the square root is not. `total_cmp`
        // because a NaN position would otherwise make the sort's output depend
        // on the comparison order.
        points.sort_by(|a, b| a.0.total_cmp(&b.0));

        self.lights_dropped +=
            suns.len().saturating_sub(MAX_DIRECTIONAL) + points.len().saturating_sub(MAX_POINT);
        suns.truncate(MAX_DIRECTIONAL);
        self.lights.extend(suns);
        self.lights
            .extend(points.into_iter().take(MAX_POINT).map(|(_, light)| light));
        Ok(())
    }
}

/// One parallel chunk's output. Reused across frames — the allocation pattern
/// of a fresh `Vec` per chunk per frame is exactly what [`Extracted`] exists to
/// avoid, and there are more chunks than there are arrays.
#[derive(Clone, Debug, Default)]
struct Chunk {
    out: Vec<Instance>,
    culled: usize,
}

impl Chunk {
    /// Keep `instance` if the frustum admits it, and count it if not.
    fn keep(&mut self, instance: Instance, frustum: Frustum) {
        if frustum.contains(instance.offset, instance.radius) {
            self.out.push(instance);
        } else {
            self.culled += 1;
        }
    }
}

/// This query's matching rows, cut into chunks of at most [`CHUNK_ROWS`].
///
/// Archetypes come back in the world's deterministic order and each one's rows
/// are cut front to back, so the chunk *sequence* is a function of world state
/// alone. That is the whole of the order-stability argument: rayon may run the
/// chunks in any order it likes, and [`gather`] concatenates them in this one.
fn chunks_of<'w, T: Component>(
    world: &'w World,
    query: &Query<&T>,
) -> Vec<(&'w [Entity], &'w [T])> {
    let mut chunks = Vec::new();
    world.views_ref(query.access(), |view| {
        let entities = view.entities();
        let rows = view.read_of::<T>();
        for pair in entities.chunks(CHUNK_ROWS).zip(rows.chunks(CHUNK_ROWS)) {
            chunks.push(pair);
        }
    });
    chunks
}

/// `scratch`, grown to `chunks` empty chunks.
fn fit(scratch: &mut Vec<Chunk>, chunks: usize) -> &mut [Chunk] {
    if scratch.len() < chunks {
        scratch.resize_with(chunks, Chunk::default);
    }
    let used = &mut scratch[..chunks];
    for chunk in used.iter_mut() {
        chunk.out.clear();
        chunk.culled = 0;
    }
    used
}

/// Concatenate the chunks **in chunk order**, whatever order they ran in.
fn gather(scratch: &[Chunk], into: &mut Vec<Instance>, culled: &mut usize) {
    into.reserve(scratch.iter().map(|c| c.out.len()).sum());
    for chunk in scratch {
        into.extend_from_slice(&chunk.out);
        *culled += chunk.culled;
    }
}

/// One entity's own transform, narrowed, with the world-space bounding radius
/// its caller worked out.
fn place<T: SimTransform>(
    entity: Entity,
    transform: &T,
    origin: sim::DVec3,
    radius: f32,
) -> Instance {
    let half = transform.half_extent();
    Instance {
        entity,
        offset: render::camera_relative(transform.world_position(), origin),
        rotation: render::narrow_rotation(transform.orientation()),
        half_extent: render::Vec3::new(half.x, half.y, half.z),
        color: transform.color(),
        asset: transform.asset(),
        radius,
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
        // The node's own radius, scaled by everything above it. `placed.radius`
        // is the *scene's* and would over-keep every node by the size of the
        // whole building.
        radius: node.radius * max_scale(model_scale) * max_scale(node.scale),
    }
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
        out.transforms::<Body>(&world, eye, Frustum::UNBOUNDED)
            .unwrap();

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
        out.transforms::<Body>(&world, sim::DVec3::new(far, 0.0, 0.0), Frustum::UNBOUNDED)
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
        out.transforms::<Renderable>(&world, sim::DVec3::ZERO, Frustum::UNBOUNDED)
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
                    radius: 0.5,
                });
            }
        }

        fn radius(&self, asset: u64) -> Option<f32> {
            (asset == SCENE).then_some(1.0)
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
        out.clear(sim::DVec3::ZERO, Frustum::UNBOUNDED);
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
        out.clear(sim::DVec3::new(far, 0.0, 0.0), Frustum::UNBOUNDED);
        out.append_models::<Model>(&world, &TwoNodes).unwrap();
        assert_eq!(out.models[0].offset, render::Vec3::ZERO);
        assert_eq!(out.models[1].offset, render::Vec3::new(1.0, 0.0, 0.0));
    }

    #[test]
    fn a_model_naming_something_that_is_not_a_scene_is_drawn_as_itself() {
        use gg_ecs::boundary::Model;
        let world = model_world(77, sim::DVec3::ZERO, sim::Vec3::splat(1.0));
        let mut out = Extracted::default();
        out.clear(sim::DVec3::ZERO, Frustum::UNBOUNDED);
        out.append_models::<Model>(&world, &TwoNodes).unwrap();
        assert_eq!(out.models.len(), 1);
        assert_eq!(out.models[0].asset, 77, "a mesh id passes straight through");

        // And with no pack at all, every model is itself.
        let mut bare = Extracted::default();
        bare.clear(sim::DVec3::ZERO, Frustum::UNBOUNDED);
        bare.append_models::<Model>(&world, &()).unwrap();
        assert_eq!(bare.models.len(), 1);
    }

    #[test]
    fn clearing_keeps_capacity_and_appending_shares_one_eye() {
        let world = world_with(&[sim::DVec3::ZERO; 4]);
        let mut out = Extracted::default();
        out.transforms::<Body>(&world, sim::DVec3::ZERO, Frustum::UNBOUNDED)
            .unwrap();
        let capacity = out.instances.capacity();
        out.transforms::<Body>(&world, sim::DVec3::new(5.0, 0.0, 0.0), Frustum::UNBOUNDED)
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

    /// Enough rows to cross several [`CHUNK_ROWS`] boundaries and give rayon
    /// something to reorder. A test at 100 entities would pass on a serial
    /// implementation and prove nothing about a parallel one.
    const MANY: usize = CHUNK_ROWS * 7 + 13;

    #[test]
    fn parallel_extract_produces_one_order_however_the_threads_ran() {
        let positions: Vec<sim::DVec3> = (0..MANY)
            .map(|i| sim::DVec3::new(i as f64, 0.0, 0.0))
            .collect();
        let world = world_with(&positions);

        let mut first = Extracted::default();
        first
            .transforms::<Body>(&world, sim::DVec3::ZERO, Frustum::UNBOUNDED)
            .unwrap();
        assert_eq!(first.instances.len(), MANY);
        // Position i sits at x = i, so the array is in world order iff the
        // offsets ascend — a permutation between chunks would show here even if
        // every chunk were internally correct.
        for (i, instance) in first.instances.iter().enumerate() {
            assert_eq!(instance.offset.x, i as f32, "row {i} moved");
        }

        // And it is the same order every run, not merely a sorted one: chunk
        // boundaries and archetype order are both functions of world state.
        for _ in 0..8 {
            let mut again = Extracted::default();
            again
                .transforms::<Body>(&world, sim::DVec3::ZERO, Frustum::UNBOUNDED)
                .unwrap();
            assert_eq!(again.instances, first.instances);
        }
    }

    #[test]
    fn culling_across_chunk_boundaries_keeps_one_order_and_one_count() {
        // Every third entity behind the camera, so survivors and casualties are
        // interleaved across every chunk rather than falling on a seam.
        let positions: Vec<sim::DVec3> = (0..MANY)
            .map(|i| {
                let z = if i % 3 == 0 {
                    5.0
                } else {
                    -5.0 - (i % 11) as f64
                };
                sim::DVec3::new(((i % 5) as f64 - 2.0) * 0.5, 0.0, z)
            })
            .collect();
        let world = world_with(&positions);

        let mut first = Extracted::default();
        first
            .transforms::<Body>(&world, sim::DVec3::ZERO, looking_forward())
            .unwrap();
        assert!(first.culled > CHUNK_ROWS, "culled {}", first.culled);
        assert_eq!(first.instances.len() + first.culled, MANY);
        for _ in 0..8 {
            let mut again = Extracted::default();
            again
                .transforms::<Body>(&world, sim::DVec3::ZERO, looking_forward())
                .unwrap();
            assert_eq!(again.instances, first.instances);
            assert_eq!(again.culled, first.culled);
        }
    }

    #[test]
    fn reusing_one_extracted_gives_the_same_answer_as_a_fresh_one() {
        // The chunk scratch is reused across frames; a stale chunk left over
        // from a longer frame would append last frame's instances to this one.
        let long: Vec<sim::DVec3> = (0..MANY)
            .map(|i| sim::DVec3::new(i as f64, 0.0, 0.0))
            .collect();
        let big = world_with(&long);
        let small = world_with(&long[..3]);

        let mut reused = Extracted::default();
        reused
            .transforms::<Body>(&big, sim::DVec3::ZERO, Frustum::UNBOUNDED)
            .unwrap();
        reused
            .transforms::<Body>(&small, sim::DVec3::ZERO, Frustum::UNBOUNDED)
            .unwrap();

        let mut fresh = Extracted::default();
        fresh
            .transforms::<Body>(&small, sim::DVec3::ZERO, Frustum::UNBOUNDED)
            .unwrap();
        assert_eq!(reused.instances, fresh.instances);
        assert_eq!(reused.instances.len(), 3);
    }

    /// Looking down -Z from the origin, 60° vertical, square.
    fn looking_forward() -> Frustum {
        Frustum::from_view_projection(render::perspective_reverse_z(
            core::f32::consts::FRAC_PI_3,
            1.0,
            0.1,
        ))
    }

    #[test]
    fn what_is_behind_the_camera_is_culled_and_counted() {
        let world = world_with(&[
            sim::DVec3::new(0.0, 0.0, -10.0), // ahead
            sim::DVec3::new(0.0, 0.0, 10.0),  // behind
            sim::DVec3::new(0.0, 0.0, -20.0), // ahead
        ]);
        let mut out = Extracted::default();
        out.transforms::<Body>(&world, sim::DVec3::ZERO, looking_forward())
            .unwrap();
        assert_eq!(out.instances.len(), 2);
        assert_eq!(out.culled, 1);
    }

    #[test]
    fn culling_removes_instances_without_reordering_the_survivors() {
        // The property golden images depend on: the culled list must be a
        // subsequence of the unculled one, not a permutation of what is left.
        let positions: Vec<sim::DVec3> = (0..40)
            .map(|i| {
                let x = f64::from(i % 7) * 4.0 - 12.0;
                let z = if i % 3 == 0 { 8.0 } else { -8.0 };
                sim::DVec3::new(x, 0.0, z)
            })
            .collect();
        let world = world_with(&positions);

        let mut all = Extracted::default();
        all.transforms::<Body>(&world, sim::DVec3::ZERO, Frustum::UNBOUNDED)
            .unwrap();
        let mut kept = Extracted::default();
        kept.transforms::<Body>(&world, sim::DVec3::ZERO, looking_forward())
            .unwrap();

        assert!(kept.culled > 0, "the test proves nothing if nothing culled");
        assert!(!kept.instances.is_empty());
        assert_eq!(kept.instances.len() + kept.culled, all.instances.len());
        let mut survivors = kept.instances.iter();
        let mut matched = 0;
        for instance in &all.instances {
            if survivors.clone().next() == Some(instance) {
                let _ = survivors.next();
                matched += 1;
            }
        }
        assert_eq!(matched, kept.instances.len(), "survivors kept their order");
    }

    #[test]
    fn a_scene_is_culled_a_node_at_a_time() {
        use gg_ecs::boundary::Model;
        // TwoNodes places meshes at x=0 and x=1. Put the scene so that one node
        // is inside the frustum and the other is well outside it.
        let world = model_world(
            SCENE,
            sim::DVec3::new(0.0, 0.0, -10.0),
            sim::Vec3::new(60.0, 1.0, 1.0),
        );
        let mut out = Extracted::default();
        out.clear(sim::DVec3::ZERO, looking_forward());
        out.append_models::<Model>(&world, &TwoNodes).unwrap();
        assert_eq!(out.models.len(), 1, "the near node survived");
        assert_eq!(out.culled, 1, "and the one 60 m to the side did not");
        assert_eq!(out.models[0].asset, 100);
    }

    #[test]
    fn a_model_whose_bounds_the_pack_does_not_know_is_never_culled() {
        use gg_ecs::boundary::Model;
        // 77 is not a scene and `TwoNodes::radius` does not know it — what an
        // asset still streaming in looks like (§4.6).
        let world = model_world(77, sim::DVec3::new(0.0, 0.0, 500.0), sim::Vec3::splat(1.0));
        let mut out = Extracted::default();
        out.clear(sim::DVec3::ZERO, looking_forward());
        out.append_models::<Model>(&world, &TwoNodes).unwrap();
        assert_eq!(out.models.len(), 1, "kept, though it is behind the camera");
        assert_eq!(out.culled, 0);
        assert!(!out.models[0].radius.is_finite());
    }

    fn lit_world(lights: &[gg_ecs::boundary::Light]) -> World {
        use gg_ecs::boundary::Light;
        let mut world = World::new();
        world.register::<Light>().unwrap();
        for &light in lights {
            let e = world.spawn();
            world.insert(e, light).unwrap();
        }
        world
    }

    fn extract_lights(world: &World, origin: sim::DVec3, frustum: Frustum) -> Extracted {
        let mut out = Extracted::default();
        out.clear(origin, frustum);
        out.append_lights(world).unwrap();
        out
    }

    #[test]
    fn a_point_light_narrows_through_the_camera_and_a_sun_has_no_position() {
        use gg_ecs::boundary::{Light, light};
        // The membrane again, on a light: 10^12 m out, an absolute `f32`
        // position would put the lamp and the wall it lights on the same point.
        let far = 1.0e12;
        let world = lit_world(&[
            Light::sun(sim::Vec3::new(0.0, -1.0, 0.0), 0x00ff_ffff, 3.0),
            Light::point(
                sim::DVec3::new(far + 2.0, 0.0, 0.0),
                0x00ff_8800,
                20.0,
                50.0,
            ),
        ]);
        let out = extract_lights(&world, sim::DVec3::new(far, 0.0, 0.0), Frustum::UNBOUNDED);

        assert_eq!(out.lights.len(), 2);
        // Directional first, whatever order the world held them in.
        assert_eq!(out.lights[0].kind, light::DIRECTIONAL);
        assert_eq!(out.lights[0].offset, render::Vec3::ZERO);
        assert_eq!(out.lights[0].direction, render::Vec3::NEG_Y);
        assert_eq!(out.lights[1].kind, light::POINT);
        assert_eq!(out.lights[1].offset, render::Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(out.lights[1].direction, render::Vec3::ZERO);
        assert_eq!(out.lights_dropped, 0);
    }

    #[test]
    fn a_point_light_whose_reach_misses_the_view_is_culled_and_a_sun_never_is() {
        use gg_ecs::boundary::Light;
        let world = lit_world(&[
            // Behind the camera and short-range: nothing it lights is on screen.
            Light::point(sim::DVec3::new(0.0, 0.0, 40.0), 0x00ff_ffff, 5.0, 1.0),
            // Also behind the camera, but reaching well past it — a lamp behind
            // your head still lights the wall in front of you.
            Light::point(sim::DVec3::new(0.0, 0.0, 4.0), 0x00ff_ffff, 5.0, 60.0),
            Light::sun(sim::Vec3::new(0.0, -1.0, 0.0), 0x00ff_ffff, 1.0),
        ]);
        let out = extract_lights(&world, sim::DVec3::ZERO, looking_forward());
        assert_eq!(out.lights.len(), 2, "{:?}", out.lights);
        assert_eq!(out.lights[1].range, 60.0, "the far-reaching one survived");
    }

    #[test]
    fn past_the_cap_the_nearest_lights_win_and_the_rest_are_counted() {
        use gg_ecs::boundary::Light;
        // Spawned far-to-near, so world order is the *opposite* of the answer:
        // a truncation that kept spawn order would keep exactly the wrong set.
        let lights: Vec<Light> = (0..MAX_POINT + 8)
            .map(|i| {
                let z = -((MAX_POINT + 8 - i) as f64);
                Light::point(sim::DVec3::new(0.0, 0.0, z), 0x00ff_ffff, 1.0, 100.0)
            })
            .collect();
        let out = extract_lights(&lit_world(&lights), sim::DVec3::ZERO, Frustum::UNBOUNDED);

        assert_eq!(out.lights.len(), MAX_POINT);
        assert_eq!(out.lights_dropped, 8);
        // Nearest first, and the nearest of all is the last one spawned.
        assert_eq!(out.lights[0].offset.z, -1.0);
        let mut previous = f32::NEG_INFINITY;
        for light in &out.lights {
            let distance = light.offset.length();
            assert!(distance >= previous, "{:?} is out of order", light.offset);
            previous = distance;
        }
    }

    #[test]
    fn the_light_list_is_the_same_every_run() {
        use gg_ecs::boundary::Light;
        // Equal distances on purpose: the sort has to be stable for the frame
        // to be, and ties are where an unstable one shows.
        let lights: Vec<Light> = (0..12)
            .map(|i| {
                Light::point(
                    sim::DVec3::new(0.0, 0.0, -5.0),
                    0x0011_0000 * (i + 1),
                    1.0,
                    10.0,
                )
            })
            .collect();
        let world = lit_world(&lights);
        let first = extract_lights(&world, sim::DVec3::ZERO, Frustum::UNBOUNDED);
        for _ in 0..8 {
            let again = extract_lights(&world, sim::DVec3::ZERO, Frustum::UNBOUNDED);
            assert_eq!(again.lights, first.lights);
        }
    }

    #[test]
    fn clearing_a_frame_forgets_last_frames_lights() {
        use gg_ecs::boundary::Light;
        let world = lit_world(&[Light::sun(sim::Vec3::new(0.0, -1.0, 0.0), 0x00ff_ffff, 1.0)]);
        let mut out = Extracted::default();
        out.clear(sim::DVec3::ZERO, Frustum::UNBOUNDED);
        out.append_lights(&world).unwrap();
        out.clear(sim::DVec3::ZERO, Frustum::UNBOUNDED);
        out.append_lights(&world).unwrap();
        assert_eq!(out.lights.len(), 1, "appended twice, not accumulated");
    }

    #[test]
    fn a_box_is_bounded_by_its_corner_so_turning_it_cannot_cull_it() {
        let mut world = World::new();
        world.register::<gg_ecs::boundary::Renderable>().unwrap();
        let e = world.spawn();
        // A long beam, just outside the frustum edge broadside, whose corner
        // reaches in. A half-extent-as-radius bound would have dropped it.
        world
            .insert(
                e,
                gg_ecs::boundary::Renderable::boxed(
                    sim::DVec3::new(6.0, 0.0, -10.0),
                    sim::Vec3::new(0.2, 0.2, 4.0),
                    0x00ff_ffff,
                ),
            )
            .unwrap();
        let mut out = Extracted::default();
        out.transforms::<gg_ecs::boundary::Renderable>(&world, sim::DVec3::ZERO, looking_forward())
            .unwrap();
        assert_eq!(out.instances.len(), 1);
        assert!((out.instances[0].radius - 4.01).abs() < 0.01);
    }
}
