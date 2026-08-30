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
//!
//! # Interpolation
//!
//! The sim ticks at a fixed rate and a window presents at the monitor's, so most
//! frames land *between* two ticks. Drawing the newer of the two puts each pose
//! on screen for two refreshes and then three, which reads as judder on
//! everything moving and as discrete steps on a mouse-look — the camera is the
//! one transform a whole screen of pixels hangs off.
//!
//! [`Extracted::capture`] records the pose every entity held when the last tick
//! began; [`Extracted::interpolate`] states how far past it this frame sits
//! (§4.1's `alpha`), and extract then blends. It costs one tick of latency and
//! is the standard trade — a frame drawn between two simulated states is a state
//! the sim did pass through, where an extrapolated one is a guess that overshoots
//! every wall. A locked pace reports `alpha == 0.0` throughout, so a replay, a
//! golden run and every headless tier extract the world exactly as it stands.

#![warn(missing_docs)]

mod frustum;

pub use frustum::Frustum;

/// The light-kind discriminants, re-exported so a consumer downstream of the
/// membrane can read [`ExtractedLight::kind`] without linking `gg-ecs` — which
/// `gg-render` does not, and should not have to for two constants.
pub use gg_ecs::boundary::light;

use gg_ecs::boundary::{Eye, Look};
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

    /// How polished and how metallic, `0.0..=1.0` each — carried for
    /// [`color`](SimTransform::color)'s reason, and defaulting to what the box
    /// pass hard-coded before the material crossed the boundary, so a game
    /// component that does not answer draws exactly as it did.
    ///
    /// One method rather than two because the pair is always read together and
    /// the default has to be the pair — a `metallic` that defaulted separately
    /// could be answered without its smoothness and silver a surface.
    fn surface(&self) -> (f32, f32) {
        (gg_ecs::boundary::DEFAULT_SMOOTHNESS, 0.0)
    }

    /// Which primitive the host draws this as (§6 M26). Defaults to the box,
    /// so a game component written before spheres existed answers correctly
    /// without knowing the question.
    fn shape(&self) -> u64 {
        gg_ecs::boundary::shape::BOX
    }

    /// How much of what stands behind the surface reads through it, `0.0..=1.0`
    /// (§6 M92). Zero — opaque — by default, for
    /// [`shape`](SimTransform::shape)'s reason: a component that has never
    /// heard the question is the opaque world it always was.
    fn transparency(&self) -> f32 {
        0.0
    }
}

/// The bounding radius of a box of `half_extent` — its corner distance, which
/// is what makes the sphere test rotation-invariant.
fn box_radius(half_extent: sim::Vec3) -> f32 {
    render::Vec3::new(half_extent.x, half_extent.y, half_extent.z).length()
}

/// What [`Instance::shape`] means, re-exported rather than restated (§6 M26).
///
/// `gg-render` does not depend on `gg-ecs` and must not start (§3), but it is
/// the crate that has to turn a shape into a vertex buffer. Extract is already
/// the seam both sides reach across, so the constants cross here — one
/// definition, named twice.
pub use gg_ecs::boundary::shape;

/// The cull radius `shape` implies, so a primitive bounds itself and the caller
/// does not choose.
///
/// An ellipsoid's is its longest semi-axis, not its box's corner: it is
/// inscribed in that box, so the `sqrt(3)` would keep plenty that never reaches
/// the frustum and cull nothing extra in exchange.
fn shape_radius(shape: u64, half_extent: sim::Vec3) -> f32 {
    if shape == gg_ecs::boundary::shape::SPHERE {
        max_scale(half_extent)
    } else {
        box_radius(half_extent)
    }
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
    /// Smoothness and metallic, `0.0..=1.0` each — the box pass's material.
    /// Unread for an instance naming an asset: a pack's own material is the
    /// authored one, and a game overriding it per placement is a feature this
    /// is not.
    pub surface: (f32, f32),
    /// The pack asset to draw, or 0 (§4.6). Always 0 in
    /// [`Extracted::instances`], never 0 in [`Extracted::models`] — the two
    /// arrays are exactly this field's two cases, split so the renderer picks a
    /// pipeline once per array rather than once per instance.
    pub asset: u64,
    /// World-space bounding radius about [`offset`](Instance::offset), metres.
    /// Carried rather than recomputed downstream because the culler already
    /// needed it and a second derivation is a second chance to disagree.
    pub radius: f32,
    /// Which primitive to draw it as (§6 M26). Unread for an instance naming an
    /// asset — a pack's content is whatever mesh it holds, and a shape field
    /// over it would be a second opinion about the same geometry.
    pub shape: u64,
    /// How much of what is behind reads through, `0.0..=1.0`, zero opaque
    /// (§6 M92). Nonzero routes a primitive to the transparent pass and out of
    /// the prepass and every caster list; unread for an instance naming an
    /// asset, [`surface`](Instance::surface)'s reason.
    pub transparency: f32,
}

/// Directional lights one frame may carry. Small on purpose: a scene has a sun,
/// occasionally a fill, and a game that wants a third is describing something
/// the shading model would be the wrong place to fix.
pub const MAX_DIRECTIONAL: usize = 4;

/// Point lights one frame may carry.
///
/// **256 since §6 M30**, where it was 32 for four milestones because a forward
/// pass looped over every one of them per fragment and the cap was therefore a
/// per-pixel cost. It is a memory cost now: the frame's lights are assigned to
/// froxels (`gg_render::cluster`) and a fragment iterates its own froxel's list,
/// which in a scene with a hundred lamps holds single digits.
///
/// What decides the number is still a measurement rather than a preference —
/// `gg-tools lights` is the sweep, and what it prices is the *assignment*, which
/// is per-light-per-froxel on the host and is what grows here. A frame past this
/// keeps the nearest and counts the rest, as it always has.
pub const MAX_POINT: usize = 256;

/// Environments one frame may carry (§6 M28).
///
/// Four, matching [`MAX_DIRECTIONAL`] and the cascade count, and for a related
/// reason: a fragment composites them front to back, so this is a per-pixel loop
/// bound. Four is one room, the corridor outside it, a pocket inside the room
/// and the world — deeper nesting than any level here has, and the drop is
/// counted rather than silent.
pub const MAX_SKY: usize = 4;

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

/// What a sky *radiates*, with nothing about where it applies (§6 M28).
///
/// Split out for one reason and it is a performance trap rather than a taste:
/// the projection is a 2048-sample quadrature cached on its input, and
/// [`ExtractedSky`]'s volume is camera-relative, so a cache keyed on the whole
/// thing would miss on every frame the camera moved. This half is what the
/// projection actually depends on, and it is stationary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyLook {
    /// `0x00RRGGBB`, sRGB — straight up.
    pub zenith: u32,
    /// The same, at the horizon.
    pub horizon: u32,
    /// The same, straight down.
    pub ground: u32,
    /// Linear multiplier over all three. Always positive here: a zero-intensity
    /// sky is dropped on the way in, so downstream has one answer to the
    /// question "is there an environment" rather than two.
    pub intensity: f32,
    /// A compiled panorama in the pack, or 0 for the gradient (§6 M27). Carried
    /// un-resolved: this crate does not know what a pack holds, and an id the
    /// renderer cannot find is the gradient rather than an error.
    pub environment: u64,
}

/// The environment in render space (§6 M24), and since §6 M28 the volume it is
/// the environment *of*.
///
/// A distinct type from [`Sky`](gg_ecs::boundary::Sky) for
/// [`ExtractedLight`]'s reason and one more: `gg-render` does not link `gg-ecs`
/// (§3's per-crate dependency budget), so a protocol component cannot be what
/// crosses into it. Colour stays `0x00RRGGBB` here exactly as a light's does —
/// extract converts geometry, never colour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExtractedSky {
    /// What it radiates.
    pub look: SkyLook,
    /// The volume's centre, **camera-relative** and narrowed — as
    /// [`ExtractedLight::offset`], and for §1.4's reason. Meaningless when
    /// [`unbounded`](ExtractedSky::unbounded).
    pub offset: render::Vec3,
    /// Half the box's size. Zero is unbounded, which is the whole of "this is
    /// the world's sky" — see [`Sky::half_extent`](gg_ecs::boundary::Sky::half_extent).
    pub half_extent: render::Vec3,
    /// Metres outside the box over which it falls to nothing.
    pub fade: f32,
}

impl ExtractedSky {
    /// Whether this is the world's sky rather than a room's.
    #[must_use]
    pub fn unbounded(&self) -> bool {
        self.half_extent == render::Vec3::ZERO
    }

    /// Distance from the camera to the volume — zero when the camera is inside
    /// it, and zero for an unbounded sky, which contains everything.
    ///
    /// What [`Extracted::append_lights`] keeps the nearest of, on
    /// [`MAX_POINT`]'s reasoning: a cap has to drop the ones that matter least,
    /// and for an environment that is the one furthest from being looked out of.
    #[must_use]
    pub fn camera_distance(&self) -> f32 {
        if self.unbounded() {
            return 0.0;
        }
        // The camera sits at the origin of this space, so the point-to-box
        // distance is the box's own offset pushed outward by its extent.
        let outside = (self.offset.abs() - self.half_extent).max(render::Vec3::ZERO);
        outside.length()
    }

    /// The box's volume, as the compositing order sorts on. Infinite when
    /// unbounded, which is what puts the world's sky underneath every room's
    /// without a second rule.
    #[must_use]
    pub fn scale(&self) -> f32 {
        if self.unbounded() {
            return f32::INFINITY;
        }
        self.half_extent.x * self.half_extent.y * self.half_extent.z
    }

    /// How much of this environment applies at `point`, camera-relative: 1
    /// inside the box, 0 beyond [`fade`](ExtractedSky::fade) metres outside it,
    /// smoothstepped between.
    ///
    /// **Duplicated in `include/pbr.slang`'s `environment_weight`**, on the same
    /// terms as the SH basis and the octahedral mapping: the two never meet at
    /// runtime — this decides which background to draw, that decides how to
    /// shade — so there is nothing to assert against, and the only defence is
    /// that both are three lines of the same arithmetic. A drift shows as a
    /// reflection that does not match the sky behind it.
    #[must_use]
    pub fn weight_at(&self, point: render::Vec3) -> f32 {
        if self.unbounded() {
            return 1.0;
        }
        let outside = ((point - self.offset).abs() - self.half_extent).max(render::Vec3::ZERO);
        let distance = outside.length();
        if self.fade <= 0.0 {
            // A hard edge, and the comparison rather than a division: `fade` of
            // zero would otherwise be a NaN weight and a black room.
            return f32::from(distance <= 0.0);
        }
        let t = (distance / self.fade).clamp(0.0, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    }
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
    /// Instances the host draws as boxes, **visible ones first** (§6 M34), each
    /// half in world iteration order (deterministic, §4.2).
    pub instances: Vec<Instance>,
    /// How many of the above the *view* frustum admits — see
    /// [`Extracted::visible`]. Equal to `instances.len()` in every frame nothing
    /// widened the cull, which is every frame with no sun.
    visible_instances: usize,
    /// Instances that name pack content, same order, same narrowing, same split.
    pub models: Vec<Instance>,
    /// [`Extracted::visible_instances`] for [`Extracted::models`].
    visible_models: usize,
    /// The camera origin every offset in both arrays is relative to. Kept
    /// alongside so a consumer cannot pair offsets with the wrong eye.
    pub camera_origin: sim::DVec3,
    /// What this frame culled against. Kept for the same reason the origin is:
    /// the arrays are only meaningful paired with the eye that produced them.
    pub frustum: Frustum,
    /// What *instances* are culled against — [`Extracted::frustum`] until
    /// [`Extracted::cast_along`] widens it, and reset to it by
    /// [`Extracted::clear`]. Private because it is derived: a second frustum
    /// anyone could set is a second way to have the two disagree.
    casters: Frustum,
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
    /// The environments this frame is lit by, empty for a world that declared
    /// no [`Sky`](gg_ecs::boundary::Sky) — which is the flat ambient term §6 M11
    /// shipped with, and every scene blessed before M24.
    ///
    /// **Innermost first**, by box volume, with an unbounded sky last (§6 M28):
    /// the shader composites them in this order and stops when the weight runs
    /// out, so the order *is* the priority and a room does not have to know what
    /// contains it. Never more than [`MAX_SKY`].
    pub skies: Vec<ExtractedSky>,
    /// Environments this frame had and could not carry. [`lights_dropped`]'s
    /// reasoning exactly — a room lit as though it were outdoors is a symptom
    /// with no other diagnosis on the overlay.
    ///
    /// [`lights_dropped`]: Extracted::lights_dropped
    pub skies_dropped: usize,
    /// Per-chunk staging for the parallel pass, kept so a frame allocates
    /// nothing. Never read by a consumer: [`gather`] is what turns it into the
    /// arrays above, in chunk order.
    scratch: Vec<Chunk>,
    /// Where [`Extracted::instances`]' entities were when the last tick began.
    previous: Poses,
    /// The same for [`Extracted::models`]. A second table and not a second
    /// component type in the first: one entity may carry both protocol
    /// components, and one slot per entity would have them overwrite each other.
    previous_models: Poses,
    /// How far past those poses this frame sits, `0.0..1.0`. `f64` because the
    /// blend happens on un-narrowed positions — see [`Blend::pose`].
    alpha: f64,
    /// The eye as the last tick began. Here rather than in the caller because it
    /// is captured and blended on the same two clocks everything above it is,
    /// and a shell keeping its own copy would be a second place to get the
    /// pairing wrong. `None` until [`Extracted::capture_eye`] has been called.
    previous_eye: Option<Eye>,
}

impl Default for Extracted {
    fn default() -> Self {
        Extracted {
            instances: Vec::new(),
            visible_instances: 0,
            visible_models: 0,
            models: Vec::new(),
            // Not derived: `sim` types carry no `Default`, on purpose — a
            // zero position is a *place*, and defaulting to it silently is how
            // an unset transform reaches the origin instead of an error.
            camera_origin: sim::DVec3::ZERO,
            frustum: Frustum::UNBOUNDED,
            casters: Frustum::UNBOUNDED,
            culled: 0,
            lights: Vec::new(),
            lights_dropped: 0,
            skies: Vec::new(),
            skies_dropped: 0,
            scratch: Vec::new(),
            previous: Poses::default(),
            previous_models: Poses::default(),
            alpha: 0.0,
            previous_eye: None,
        }
    }
}

impl Extracted {
    /// Record the pose every entity carrying `T` holds *now*, as the one the
    /// coming frames blend away from.
    ///
    /// Called once per sim tick, before the tick runs — so what it holds is the
    /// world the previous tick left, which is the near end of the interval a
    /// frame lands in. A tick that skipped this leaves the previous capture in
    /// place; that is one tick of smear on a paused sim and nothing at all on a
    /// halted one, where the two poses are the same world.
    ///
    /// Read-only and render-side: nothing recorded here re-enters the sim, and
    /// a run that never calls it renders exactly as it did before this existed.
    ///
    /// # Errors
    ///
    /// If the world refuses the query, which one read alone cannot cause.
    pub fn capture<T: SimTransform>(&mut self, world: &World) -> Result<(), AliasError> {
        self.previous.capture::<T>(world)
    }

    /// The same, for the array [`Extracted::append_models`] fills.
    ///
    /// # Errors
    ///
    /// If the world refuses the query, which one read alone cannot cause.
    pub fn capture_models<T: SimTransform>(&mut self, world: &World) -> Result<(), AliasError> {
        self.previous_models.capture::<T>(world)
    }

    /// The same, for the eye [`Eye::of`] picks. Concrete where the two above are
    /// generic, and for [`Extracted::append_lights`]' reason: a game with its own
    /// camera component writes a system that fills this one in, which is what the
    /// protocol is for.
    ///
    /// # Errors
    ///
    /// If the world refuses the query, which one read alone cannot cause.
    pub fn capture_eye(&mut self, world: &World) -> Result<(), AliasError> {
        self.previous_eye = Some(Eye::of(world)?);
        Ok(())
    }

    /// The eye to render this frame through: `current`, blended back towards the
    /// captured one by whatever [`Extracted::interpolate`] was told, and turned
    /// by whatever travel `latch` says no tick has spent (§6 M56).
    ///
    /// Taken rather than read out of the world so a host may substitute its own
    /// camera afterwards — the editor's, which is not in any archetype — without
    /// this having an opinion about which one it got.
    ///
    /// **The latch adds to the blend rather than replacing it**, which is the
    /// one shape that gets both halves right. Split the angle a tick produced
    /// into the part the hand drove and the part the game did: the hand's part
    /// must not be walked backwards at all — that walk *is* the lag — while the
    /// game's part (a recoil kick, a cutscene, a turret) is sim motion like a
    /// position and wants exactly the interpolation everything else gets.
    ///
    /// So the correction is the travel no tick has spent, plus the `1 - alpha`
    /// of the last tick's *hand* travel that the blend is still holding back.
    /// The two together put the hand's contribution at where the hand is now and
    /// leave the game's contribution blended, and the algebra is exact rather
    /// than approximate: the tick's own spend is what the host reports.
    ///
    /// A latch whose four angles are zero is therefore the identity, which is
    /// what a frame with no mouse travel gets — every golden, every replay
    /// (`Input::tick_from` spends no motion) and every headless tier. Position
    /// keeps the blend regardless, because a body walks on the sim's clock and
    /// extrapolating it puts the camera through walls the sim will not let it
    /// reach.
    #[must_use]
    pub fn eye(&self, current: Eye, latch: Option<Latch>) -> Eye {
        let blended = match self.previous_eye {
            Some(previous) => blend_eye(previous, current, self.alpha as f32),
            None => current,
        };
        match latch {
            Some(latch) => latched(blended, latch, self.alpha as f32),
            None => blended,
        }
    }

    /// How far past the captured poses this frame sits, `0.0..1.0` (§4.1).
    ///
    /// Zero — the default, and what a locked pace reports for every frame it
    /// ever runs — extracts the world exactly as it stands, so a replay and a
    /// golden run are untouched by any of this. Clamped rather than trusted: a
    /// clock that handed over 1.5 would extrapolate, which is a different
    /// decision than the one this crate has made.
    ///
    /// Frame state, not frame *output*: [`Extracted::clear`] leaves it alone, so
    /// one call covers the whole frame's several appends.
    pub fn interpolate(&mut self, alpha: f32) {
        self.alpha = f64::from(alpha.clamp(0.0, 1.0));
    }

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
        self.visible_instances = 0;
        self.visible_models = 0;
        self.camera_origin = camera_origin;
        self.frustum = frustum;
        // Reset, not kept: a caster volume outlives neither the frustum it was
        // swept from nor the sun it was swept along, so a frame that forgets to
        // re-widen culls tightly rather than against last frame's camera.
        self.casters = frustum;
        self.culled = 0;
        self.lights.clear();
        self.lights_dropped = 0;
        self.skies.clear();
        self.skies_dropped = 0;
    }

    /// Widen *instance* culling to whatever can cast a shadow into the view
    /// (§6 M11): [`Extracted::frustum`] swept `reach` metres up-light, which is
    /// [`Frustum::swept`] along the sun [`Extracted::sun`] found.
    ///
    /// Call between [`Extracted::append_lights`] — the sun has to be in hand — and
    /// the instance appends, with the reach `gg-render`'s `View::caster_reach`
    /// gives, so what is kept and what the cascades can hold are read off one
    /// scheme. It takes the sun from this frame's own light list rather than from
    /// an argument, which is the only way a caller cannot pair a frame with the
    /// wrong one; a frame with no sun sweeps along the zero vector, relaxes no
    /// plane, and culls exactly as it did before this existed — which is every
    /// headless run and every test below.
    ///
    /// Instances only. [`Extracted::frustum`] still culls lights and is still what
    /// the overlay reports, because a light off screen lights nothing on it while
    /// a box off screen shadows plenty.
    ///
    /// What it admits beyond the view is kept **behind** what the view admits
    /// (§6 M34), so the main pass draws [`Extracted::visible`] and the shadow
    /// passes walk the whole array. Before that the main pass submitted every
    /// caster too and the rasterizer discarded them off screen — correct, and
    /// paid for per frame in vertex work and draw calls.
    pub fn cast_shadows(&mut self, reach: f32) {
        let sun = self.sun().unwrap_or(render::Vec3::ZERO);
        self.casters = self.frustum.swept(sun, reach);
    }

    /// The prefix of [`Extracted::instances`] the *view* frustum admits — what
    /// a main pass draws (§6 M34).
    ///
    /// The rest of the array is off screen and kept only because
    /// [`Extracted::cast_shadows`] widened the cull, so a shadow pass wants all
    /// of it and nothing else does. Equal to the whole array whenever nothing
    /// widened it, which is every frame with no sun — the split costs a caller
    /// that ignores it nothing, and a caller that uses it cannot forget the
    /// tail exists, because the tail is where the array still is.
    #[must_use]
    pub fn visible(&self) -> &[Instance] {
        &self.instances[..self.visible_instances]
    }

    /// [`Extracted::visible`] for [`Extracted::models`].
    #[must_use]
    pub fn visible_models(&self) -> &[Instance] {
        &self.models[..self.visible_models]
    }

    /// How many instances across both arrays are kept for their shadows alone —
    /// the number a `gg::draws` line reports beside `culled`, and the whole of
    /// what §6 M34 stopped submitting to the main pass.
    #[must_use]
    pub fn casting_only(&self) -> usize {
        (self.instances.len() - self.visible_instances) + (self.models.len() - self.visible_models)
    }

    /// The travel direction of the light this frame's shadows come from — the
    /// first casting directional in [`Extracted::lights`], unit.
    ///
    /// The same light `gg-render` fits its cascades to, and read out of the same
    /// array, so the sweep and the fit cannot pick different suns. `None` before
    /// [`Extracted::append_lights`] has run, and for a world with no sun.
    #[must_use]
    pub fn sun(&self) -> Option<render::Vec3> {
        self.lights
            .iter()
            .find(|l| {
                l.kind == gg_ecs::boundary::light::DIRECTIONAL && l.direction != render::Vec3::ZERO
            })
            .map(|l| l.direction)
    }

    /// How far the point `camera-relative `at`` sits along `axis`, in units of
    /// `period`, wrapped to one cell.
    ///
    /// A grid built in camera-relative space slides with the camera, so anything
    /// quantized to it crawls across world geometry as the eye moves — which is
    /// what a shadow map's texel boundaries do (§6 M11). Subtracting this phase
    /// back out locks the grid to absolute world space.
    ///
    /// It lives here because the quantity it is a phase *of* is
    /// `dot(camera_origin + at, axis)` on an un-narrowed position, and the
    /// fractional part of a planetary coordinate is precisely what `f32`
    /// destroys (§1.4). The two terms are summed *before* the wrap and in `f64`,
    /// which is the whole of why this cannot be written on the render side: the
    /// camera term is enormous, `at` is metres, and their sum's fraction is the
    /// answer.
    ///
    /// `axis` is a direction in the grid's space and `period` one cell's width
    /// along it, both in the caller's own units, so the quotient counts cells
    /// whatever those are. A `period` of zero or a non-finite one yields zero: a
    /// grid with no spacing has no phase.
    ///
    /// In `[0, 1]` rather than `[0, 1)` — a phase just under one narrows to
    /// exactly one, and nothing may depend on the difference, because a whole
    /// cell of shift is the same aligned grid.
    #[must_use]
    pub fn grid_phase(&self, axis: render::Vec3, at: render::Vec3, period: f32) -> f32 {
        let period = f64::from(period);
        if period == 0.0 || !period.is_finite() {
            return 0.0;
        }
        let axis64 = sim::DVec3::new(f64::from(axis.x), f64::from(axis.y), f64::from(axis.z));
        // Widening, never narrowing: the origin is the one position that must not
        // be rounded before it is used, and `at` is small enough that its own
        // narrowing already happened when extract built it.
        let at64 = sim::DVec3::new(f64::from(at.x), f64::from(at.y), f64::from(at.z));
        let along = self.camera_origin.dot(axis64) + at64.dot(axis64);
        let cells = along / period;
        // `floor`, not `%`: the remainder keeps the dividend's sign, so a camera
        // on the negative side of the origin would get a phase in `(-1, 0]` and
        // shift the grid the wrong way across the world's midpoint.
        let phase = cells - cells.floor();
        if phase.is_finite() { phase as f32 } else { 0.0 }
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
        // Both volumes: the view, and the caster volume it was swept into unless
        // nobody widened it. An instance off screen may still be laying a shadow
        // across it, and since §6 M34 which of the two admitted it is recorded
        // rather than forgotten.
        let (view, casters) = (self.frustum, self.casters);
        let blend = self.previous.blend(self.alpha);
        let chunks = chunks_of::<T>(world, &query);
        let scratch = fit(&mut self.scratch, chunks.len());
        scratch
            .par_iter_mut()
            .zip(chunks.par_iter())
            .for_each(|(chunk, (entities, rows))| {
                for (&entity, transform) in entities.iter().zip(rows.iter()) {
                    let radius = shape_radius(transform.shape(), transform.half_extent());
                    let pose = blend.pose(entity, transform);
                    chunk.keep(
                        place(entity, transform, origin, radius, pose),
                        view,
                        casters,
                    );
                }
            });
        gather(
            scratch,
            &mut self.instances,
            &mut self.visible_instances,
            &mut self.culled,
        );
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
        let (view, casters) = (self.frustum, self.casters);
        let blend = self.previous_models.blend(self.alpha);
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
                    let pose = blend.pose(entity, transform);
                    let placed = place(entity, transform, origin, own, pose);
                    let mut expanded = false;
                    scenes.expand(asset, &mut |node| {
                        expanded = true;
                        // Culled per node, not per entity: a scene is expanded
                        // exactly so the half of a building behind the camera
                        // costs nothing, and testing the whole scene's bound
                        // would hand that back.
                        chunk.keep(
                            compose(&placed, transform, origin, node, pose),
                            view,
                            casters,
                        );
                    });
                    if !expanded {
                        chunk.keep(placed, view, casters);
                    }
                }
            });
        gather(
            scratch,
            &mut self.models,
            &mut self.visible_models,
            &mut self.culled,
        );
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
        use gg_ecs::boundary::{Light, Sky, light};

        let origin = self.camera_origin;
        // The environments, taken here rather than through a call of their own:
        // a sky *is* a light, it is read on the same tick as the rest of them,
        // and a second entry point is a second thing a host can forget — which
        // would show up as a scene that lights itself flatly for no visible
        // reason. A world with none leaves the array empty (§6 M24).
        let skies = Query::<&Sky>::new()?;
        world.each_ref(&skies, |_, sky: &Sky| {
            // Zero intensity is *no environment* rather than a black one, so it
            // is filtered here and the renderer downstream sees one answer.
            if sky.intensity > 0.0 {
                self.skies.push(ExtractedSky {
                    look: SkyLook {
                        zenith: sky.zenith,
                        horizon: sky.horizon,
                        ground: sky.ground,
                        intensity: sky.intensity,
                        environment: sky.environment,
                    },
                    offset: render::camera_relative(sky.center, origin),
                    half_extent: render::to_render(sky.half_extent).abs(),
                    fade: sky.fade.max(0.0),
                });
            }
        });
        // Two orders, and they are different questions (§6 M28). The **cap**
        // keeps what is nearest, on `MAX_POINT`'s reasoning — a volume the
        // camera is far outside of shades the fewest fragments, and an unbounded
        // sky measures zero here, so the fallback can never be the one dropped.
        // The **composite** order is by volume, innermost first, which is what
        // makes "the smallest box containing this fragment wins" fall out of
        // walking the array rather than out of a rule the shader has to know.
        // Both stable and both `total_cmp`: a NaN would otherwise make the
        // picture depend on the comparison order.
        if self.skies.len() > MAX_SKY {
            self.skies
                .sort_by(|a, b| a.camera_distance().total_cmp(&b.camera_distance()));
            self.skies_dropped += self.skies.len() - MAX_SKY;
            self.skies.truncate(MAX_SKY);
        }
        self.skies.sort_by(|a, b| a.scale().total_cmp(&b.scale()));

        let query = Query::<&Light>::new()?;
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

/// An entity's placement for one frame: its own, or blended towards it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Posed {
    position: sim::DVec3,
    rotation: sim::DQuat,
}

/// Where entities were when the last tick began.
///
/// Keyed by entity **index**, not by extract order. Extract order is stable
/// between two ticks that spawned nothing and shifts wholesale on the one that
/// did — a table matched by position would pop every entity after the new row,
/// which in a game that spawns is every frame.
///
/// `stamp` is what makes a slot's answer "captured *this* tick" rather than
/// "captured once". A slot is overwritten in place and never cleared (a per-tick
/// memset over the index space is the one cost this must not have), so without
/// it an entity that stopped carrying `T` and later got it back would blend out
/// of a pose it left minutes ago, across the whole level.
#[derive(Clone, Debug, Default)]
struct Poses {
    /// Indexed by [`Entity::index`]. Sparse: dead and never-captured slots sit
    /// at [`Pose::EMPTY`], whose entity is [`Entity::NONE`] and matches nothing.
    slots: Vec<Pose>,
    /// Captures so far. Starts at zero, which [`Pose::EMPTY`] also holds — so
    /// the first capture is stamp 1 and an untouched slot never matches.
    stamp: u64,
}

/// One entity's captured pose.
#[derive(Clone, Copy, Debug)]
struct Pose {
    /// The whole handle, generation included: an index reused by a new entity
    /// must not inherit the dead one's pose.
    entity: Entity,
    stamp: u64,
    position: sim::DVec3,
    rotation: sim::DQuat,
}

impl Pose {
    /// Not `Default`: `sim` types carry none on purpose, because a zero position
    /// is a *place* and defaulting to it silently is how an unset transform
    /// reaches the origin instead of an error. Here the guard is the entity.
    const EMPTY: Pose = Pose {
        entity: Entity::NONE,
        stamp: 0,
        position: sim::DVec3::ZERO,
        rotation: sim::DQuat::IDENTITY,
    };
}

impl Poses {
    /// Overwrite every `T`-carrying entity's slot with the pose it holds now.
    fn capture<T: SimTransform>(&mut self, world: &World) -> Result<(), AliasError> {
        let query = Query::<&T>::new()?;
        self.stamp += 1;
        let (slots, stamp) = (&mut self.slots, self.stamp);
        world.each_ref(&query, |entity, transform: &T| {
            let index = entity.index() as usize;
            if index >= slots.len() {
                slots.resize(index + 1, Pose::EMPTY);
            }
            slots[index] = Pose {
                entity,
                stamp,
                position: transform.world_position(),
                rotation: transform.orientation(),
            };
        });
        Ok(())
    }

    /// This frame's blend against these poses. `alpha` of zero — and a table
    /// nothing was ever captured into — yields the identity blend, which is the
    /// path every locked-pace run takes.
    fn blend(&self, alpha: f64) -> Blend<'_> {
        Blend {
            poses: self,
            alpha: if self.stamp == 0 { 0.0 } else { alpha },
        }
    }

    /// `entity`'s pose from the **latest** capture, or `None` — not captured,
    /// captured in an earlier tick and not this one, or a reused index.
    fn of(&self, entity: Entity) -> Option<&Pose> {
        self.slots
            .get(entity.index() as usize)
            .filter(|pose| pose.entity == entity && pose.stamp == self.stamp)
    }
}

/// A frame's position between the captured poses and the world as it stands.
#[derive(Clone, Copy)]
struct Blend<'a> {
    poses: &'a Poses,
    alpha: f64,
}

impl Blend<'_> {
    /// Where to draw `transform` this frame.
    ///
    /// The blend is `f64` and happens **before** the narrowing, which is the
    /// whole membrane argument (§1.4) applied to a second pair of positions: two
    /// planetary coordinates blended in `f32` would round to the same 65 km grid
    /// point and the interpolation would be pure noise.
    ///
    /// An entity with no captured pose draws at its own — which is what a
    /// spawn should do. Popping into existence at the right place beats sliding
    /// in from wherever the table's zero happened to be.
    fn pose<T: SimTransform>(&self, entity: Entity, transform: &T) -> Posed {
        let position = transform.world_position();
        let rotation = transform.orientation();
        if self.alpha == 0.0 {
            return Posed { position, rotation };
        }
        match self.poses.of(entity) {
            Some(previous) => Posed {
                position: previous.position.lerp(position, self.alpha),
                rotation: blend_rotation(previous.rotation, rotation, self.alpha),
            },
            None => Posed { position, rotation },
        }
    }
}

/// Shortest-arc normalized lerp between two orientations.
///
/// Nlerp and not slerp: over one tick the angular error against a constant-rate
/// slerp peaks below a thousandth of the arc, which is well under a pixel at any
/// rotation rate a frame covers, and slerp costs an `acos`/`sin` pair per
/// instance to fix it. The sign flip is not optional — `q` and `-q` are the same
/// orientation, and lerping between the two representations of a small turn goes
/// the long way round, which is a full revolution in one frame.
///
/// A blend that cancels to zero length (two exactly opposite quaternions, which
/// is a half-turn in one tick) keeps `b`: there is no shortest arc to pick
/// between, and hazard 6 forbids handing back the NaN a normalize would give.
fn blend_rotation(a: sim::DQuat, b: sim::DQuat, t: f64) -> sim::DQuat {
    let b = if a.dot(b) < 0.0 {
        sim::DQuat::from_xyzw(-b.x, -b.y, -b.z, -b.w)
    } else {
        b
    };
    sim::DQuat::from_xyzw(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
        a.w + (b.w - a.w) * t,
    )
    .try_normalize()
    .unwrap_or(b)
}

/// The eye a frame `alpha` of a tick past `previous` looks through (§4.1).
///
/// Separate from [`Extracted`] because the eye is not an instance: it is chosen
/// by [`Eye::of`]'s rule, it sets the origin every offset in the frame is
/// relative to, and a host may substitute its own (the editor's camera) after
/// the game's has been read.
///
/// Angles take the short way round. A yaw wrapped from `+π` to `-π` between two
/// ticks is a few milliradians of turn, and blending it the arithmetic way spins
/// the camera through a full revolution inside one frame — which is the one
/// visible artifact a naive interpolator has, and it fires exactly when someone
/// is turning quickly.
///
/// `ortho` blends only between two eyes that agree about what a projection is:
/// zero is the perspective sentinel (§6 M20), so a frame halfway between an
/// orthographic eye and a perspective one would render a third thing that is
/// neither. A game that switched projection between two ticks switches on the
/// tick, like the discrete fact it is.
#[must_use]
pub fn blend_eye(previous: Eye, current: Eye, alpha: f32) -> Eye {
    let alpha = alpha.clamp(0.0, 1.0);
    let flat = (previous.ortho == 0.0) == (current.ortho == 0.0);
    Eye {
        position: previous.position.lerp(current.position, f64::from(alpha)),
        yaw: blend_angle(previous.yaw, current.yaw, alpha),
        pitch: blend_angle(previous.pitch, current.pitch, alpha),
        ortho: match flat {
            true => previous.ortho + (current.ortho - previous.ortho) * alpha,
            false => current.ortho,
        },
        reserved: current.reserved,
    }
}

/// The turn a frame owes the hand: travel no tick has spent, already in radians
/// (§6 M56).
///
/// Radians and not axis units because this crate cannot see `gg-input` and must
/// not learn to (§3) — the host reads the accumulator, [`Latch::of`] applies the
/// game's own declared rate, and what crosses is an angle.
///
/// Render-side by construction, exactly as [`Extracted::interpolate`]'s alpha
/// is: it is computed from an accumulator, consumed by [`Extracted::eye`], and
/// written back nowhere. No canonical hash, replay or save can see it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Latch {
    /// Radians the hand has turned that no tick has spent.
    pub yaw: f32,
    /// The same for pitch, before [`Latch::limit`].
    pub pitch: f32,
    /// Radians of the **last tick's** turn that came from the hand rather than
    /// from the game — the part of the blend that must not be walked backwards.
    /// Separate from [`Latch::yaw`] because it is scaled by `1 - alpha` and that
    /// one is not.
    pub yaw_spent: f32,
    /// The same for pitch.
    pub pitch_spent: f32,
    /// The game's own pitch clamp, radians; `0.0` leaves the angle alone.
    pub pitch_limit: f32,
}

impl Latch {
    /// What `look` makes of the two axis readings that matter: what is pending
    /// (`gg_input::Input::pending`) and what the last tick spent of the same
    /// source (`gg_input::Input::spent`).
    ///
    /// The multiply lives here rather than in the host so the sign convention is
    /// applied in one place: [`Look`] states a *signed* rate precisely so that
    /// neither this crate nor the shell has to know which way a mouse turns a
    /// camera, and two copies of that decision is one too many.
    #[must_use]
    pub fn of(look: Look, pending: (f32, f32), spent: (f32, f32)) -> Self {
        Latch {
            yaw: pending.0 * look.yaw_rate,
            pitch: pending.1 * look.pitch_rate,
            yaw_spent: spent.0 * look.yaw_rate,
            pitch_spent: spent.1 * look.pitch_rate,
            pitch_limit: look.pitch_limit,
        }
    }

    /// `pitch` held inside the game's declared limit, or untouched if it
    /// declared none.
    ///
    /// The latch clamps because the *tick* clamps: without this the picture
    /// turns past a limit the next tick refuses, and the hand feels a snap back
    /// — which is a worse artifact than the lag being removed.
    #[must_use]
    pub fn limit(&self, pitch: f32) -> f32 {
        match self.pitch_limit > 0.0 {
            true => pitch.clamp(-self.pitch_limit, self.pitch_limit),
            false => pitch,
        }
    }
}

/// [`Extracted::eye`]'s second half on an eye already blended: the turn no tick
/// has spent, plus the `1 - alpha` of last tick's *hand* travel the blend is
/// still holding back.
///
/// Separate from the method since §6 M63, because there are two cameras that
/// want it and only one of them is in the world. The editor's is host state
/// blended from a pair the editor keeps ([`blend_eye`] directly), and a shell
/// applying this arithmetic itself would be the second copy of an algebra whose
/// whole correctness argument — see [`Extracted::eye`] — is that the two terms
/// are scaled differently on purpose.
#[must_use]
pub fn latched(blended: Eye, latch: Latch, alpha: f32) -> Eye {
    let held = 1.0 - alpha.clamp(0.0, 1.0);
    Eye {
        yaw: wrap_angle(blended.yaw + latch.yaw + held * latch.yaw_spent),
        pitch: latch.limit(blended.pitch + latch.pitch + held * latch.pitch_spent),
        ..blended
    }
}

/// Keep an angle in ±π. An unbounded yaw loses the precision a 240 Hz frame's
/// share of a turn is made of — the same wrap every fly camera in this tree
/// applies, restated here because the latch adds to an angle after the sim
/// stopped looking at it.
fn wrap_angle(yaw: f32) -> f32 {
    use core::f32::consts::{PI, TAU};
    let yaw = yaw % TAU;
    match yaw {
        y if y > PI => y - TAU,
        y if y < -PI => y + TAU,
        y => y,
    }
}

/// Blend two angles along the shorter of the two arcs between them, radians.
///
/// `%` and not a transcendental: the remainder is exact, so this is portable by
/// construction like everything else that crosses the membrane.
fn blend_angle(a: f32, b: f32, t: f32) -> f32 {
    use core::f32::consts::{PI, TAU};
    let mut delta = (b - a) % TAU;
    if delta > PI {
        delta -= TAU;
    } else if delta < -PI {
        delta += TAU;
    }
    a + delta * t
}

/// One parallel chunk's output. Reused across frames — the allocation pattern
/// of a fresh `Vec` per chunk per frame is exactly what [`Extracted`] exists to
/// avoid, and there are more chunks than there are arrays.
#[derive(Clone, Debug, Default)]
struct Chunk {
    /// What the *view* frustum admits.
    visible: Vec<Instance>,
    /// What only the swept caster volume admits — off screen, but laying a
    /// shadow across it (§6 M11). Empty in every frame nothing widened the cull.
    casting: Vec<Instance>,
    culled: usize,
}

impl Chunk {
    /// Keep `instance` in the tighter of the two volumes that admits it, and
    /// count it if neither does.
    ///
    /// Two tests where M11 did one, and the second is only reached by what the
    /// first rejected — so a frame with no sun (`casters == view`) pays exactly
    /// what it paid before, and one with a sun pays a second halfspace test on
    /// the minority of instances that are off screen.
    fn keep(&mut self, instance: Instance, view: Frustum, casters: Frustum) {
        if view.contains(instance.offset, instance.radius) {
            self.visible.push(instance);
        } else if casters.contains(instance.offset, instance.radius) {
            self.casting.push(instance);
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
        chunk.visible.clear();
        chunk.casting.clear();
        chunk.culled = 0;
    }
    used
}

/// Concatenate the chunks **in chunk order**, whatever order they ran in,
/// keeping `into` partitioned visible-then-casting across repeated appends
/// (§6 M34).
///
/// The rotate is what makes a second `append` legal. `into` is already
/// `[visible | casting]`; this appends the new visible block after the existing
/// casters and turns `[casting | new visible]` back into `[new visible |
/// casting]` in place — one memmove of the caster tail, no allocation, and the
/// relative order inside each half is untouched, which is what the determinism
/// argument above rests on. In a frame with no sun the tail is empty and the
/// rotate is a no-op.
fn gather(scratch: &[Chunk], into: &mut Vec<Instance>, visible: &mut usize, culled: &mut usize) {
    into.reserve(
        scratch
            .iter()
            .map(|c| c.visible.len() + c.casting.len())
            .sum(),
    );
    let tail = into.len() - *visible;
    let start = into.len();
    for chunk in scratch {
        into.extend_from_slice(&chunk.visible);
    }
    let added = into.len() - start;
    into[*visible..].rotate_left(tail);
    *visible += added;
    for chunk in scratch {
        into.extend_from_slice(&chunk.casting);
        *culled += chunk.culled;
    }
}

/// One entity's own transform, narrowed, with the world-space bounding radius
/// its caller worked out.
///
/// `pose` is where it is *this frame* — its own pose, or the blend towards it —
/// and everything else still comes off the transform: only position and
/// orientation move between two ticks, and a size or a colour that changed is a
/// discrete fact rather than a quantity to be halfway through.
fn place<T: SimTransform>(
    entity: Entity,
    transform: &T,
    origin: sim::DVec3,
    radius: f32,
    pose: Posed,
) -> Instance {
    let half = transform.half_extent();
    Instance {
        entity,
        offset: render::camera_relative(pose.position, origin),
        rotation: render::narrow_rotation(pose.rotation),
        half_extent: render::Vec3::new(half.x, half.y, half.z),
        color: transform.color(),
        surface: transform.surface(),
        asset: transform.asset(),
        radius,
        shape: transform.shape(),
        transparency: transform.transparency(),
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
    pose: Posed,
) -> Instance {
    let model_scale = transform.half_extent();
    let scaled = sim::DVec3::new(
        node.translation.x * f64::from(model_scale.x),
        node.translation.y * f64::from(model_scale.y),
        node.translation.z * f64::from(model_scale.z),
    );
    let rotation = pose.rotation;
    let world = pose.position + rotation.rotate(scaled);
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
        surface: placed.surface,
        asset: node.mesh,
        // The node's own radius, scaled by everything above it. `placed.radius`
        // is the *scene's* and would over-keep every node by the size of the
        // whole building.
        radius: node.radius * max_scale(model_scale) * max_scale(node.scale),
        // Carried rather than set: a node draws a mesh, so nothing reads this,
        // and inheriting it beats inventing a second answer.
        shape: placed.shape,
        transparency: placed.transparency,
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

    fn surface(&self) -> (f32, f32) {
        (self.smoothness, self.metallic)
    }

    fn shape(&self) -> u64 {
        self.shape
    }

    fn transparency(&self) -> f32 {
        self.transparency
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

    /// Two boxes and a sun: one on screen, one overhead and off it — and the
    /// overhead one is what lays a shadow across the first.
    fn a_room_under(direction: sim::Vec3) -> World {
        use gg_ecs::boundary::Light;
        let mut world = world_with(&[
            sim::DVec3::new(0.0, 0.0, -10.0),  // on screen
            sim::DVec3::new(0.0, 30.0, -10.0), // overhead, casting down into it
        ]);
        world.register::<Light>().unwrap();
        let e = world.spawn();
        world
            .insert(e, Light::sun(direction, 0x00ff_ffff, 3.0))
            .unwrap();
        world
    }

    /// The post-M21 defect, in the instance list: a box high above the view is
    /// off screen and its shadow is not, so culling it by the view is a shadow
    /// that vanishes while its own is still in frame.
    #[test]
    fn a_caster_off_screen_survives_a_sweep_up_light_and_the_view_alone_drops_it() {
        let world = a_room_under(sim::Vec3::new(0.0, -1.0, 0.0));
        let mut out = Extracted::default();
        out.transforms::<Body>(&world, sim::DVec3::ZERO, looking_forward())
            .unwrap();
        assert_eq!(out.culled, 1, "the view alone drops the caster");

        let mut swept = |world: &World, reach| {
            out.clear(sim::DVec3::ZERO, looking_forward());
            out.append_lights(world).unwrap();
            out.cast_shadows(reach);
            out.append::<Body>(world).unwrap();
            out.culled
        };
        assert_eq!(swept(&world, 40.0), 0, "swept up-light it is kept");

        // Two ways of failing to be the fix. A sun from *below* reaches nothing
        // above the view, and 40 m of any sun does not reach a caster at 100.
        let below = a_room_under(sim::Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(
            swept(&below, 40.0),
            1,
            "a sun from below casts nothing down"
        );
        assert_eq!(swept(&world, 5.0), 1, "and the reach is a reach");
    }

    /// §6 M34: the caster the sweep kept is kept **behind** what the view
    /// admitted, so a main pass can draw a prefix and a shadow pass the whole.
    ///
    /// This is the half of the split a picture cannot show. Submitting the
    /// casters to the main pass too renders the identical frame — the rasterizer
    /// discards them off screen — so nothing in `tests/gg-images` would ever
    /// notice the partition being lost, or an off-by-one putting a visible
    /// instance in the tail. Both directions are asserted here, and the second
    /// one is why the room's own box is checked rather than only the count: a
    /// prefix that is merely the right *length* is not a prefix.
    #[test]
    fn what_only_the_sweep_kept_sits_behind_what_the_view_admitted() {
        let world = a_room_under(sim::Vec3::new(0.0, -1.0, 0.0));
        let mut out = Extracted::default();
        out.clear(sim::DVec3::ZERO, looking_forward());
        out.append_lights(&world).unwrap();
        out.cast_shadows(40.0);
        out.append::<Body>(&world).unwrap();

        assert_eq!(out.culled, 0, "the sweep keeps both");
        assert_eq!(out.instances.len(), 2);
        assert_eq!(out.visible().len(), 1, "one of the two is on screen");
        assert_eq!(out.casting_only(), 1);
        // The prefix is the *right* instance and not merely one of them: the
        // floor is in frame and the box above the camera is not.
        assert!(
            out.visible()[0].offset.y < out.instances[1].offset.y,
            "the visible prefix holds the floor, the tail the box overhead — {:?}",
            out.instances
        );

        // Nothing widened is the common frame, and it must cost nothing: the
        // whole array is the prefix, so a caller using `visible()` and a caller
        // using `instances` are reading the same slice.
        out.clear(sim::DVec3::ZERO, looking_forward());
        out.append::<Body>(&world).unwrap();
        assert_eq!(out.visible().len(), out.instances.len());
        assert_eq!(out.casting_only(), 0);
    }

    /// The partition survives a second append, which is the whole reason
    /// [`gather`] rotates rather than concatenates (§6 M34).
    ///
    /// A frame whose instances come from two component types appends twice, and
    /// the naive version leaves `[visible, casting, visible, casting]` — an
    /// array whose prefix silently stops being the visible set on the second
    /// call. Nothing downstream would report that; it would draw half a scene.
    #[test]
    fn a_second_append_keeps_the_visible_ones_in_front() {
        let world = a_room_under(sim::Vec3::new(0.0, -1.0, 0.0));
        let mut out = Extracted::default();
        out.clear(sim::DVec3::ZERO, looking_forward());
        out.append_lights(&world).unwrap();
        out.cast_shadows(40.0);
        out.append::<Body>(&world).unwrap();
        out.append::<Body>(&world).unwrap();

        assert_eq!(out.instances.len(), 4);
        assert_eq!(out.visible().len(), 2, "both appends' visible ones, first");
        assert_eq!(out.casting_only(), 2);
        let overhead = out.instances[3].offset.y;
        assert!(
            out.visible().iter().all(|i| i.offset.y < overhead),
            "a caster reached the prefix: {:?}",
            out.instances
        );
        // And the halves kept their own order — the determinism argument
        // `chunks_of` makes is about the sequence within each, not across both.
        assert_eq!(out.instances[0].entity, out.instances[1].entity);
        assert_eq!(out.instances[2].entity, out.instances[3].entity);
    }

    /// `clear` resets the sweep, so a frame that forgets to re-widen culls by its
    /// own camera rather than by the one two frames ago.
    #[test]
    fn a_caster_volume_does_not_outlive_the_frame_it_was_swept_for() {
        let world = a_room_under(sim::Vec3::new(0.0, -1.0, 0.0));
        let mut out = Extracted::default();
        out.clear(sim::DVec3::ZERO, looking_forward());
        out.append_lights(&world).unwrap();
        out.cast_shadows(40.0);
        out.append::<Body>(&world).unwrap();
        assert_eq!(out.culled, 0);

        out.clear(sim::DVec3::ZERO, looking_forward());
        out.append_lights(&world).unwrap();
        out.append::<Body>(&world).unwrap();
        assert_eq!(out.culled, 1, "the sweep did not carry over");
    }

    /// The sixth plane doing its job in a real extract, not only in `Frustum`'s
    /// own tests (post-M20 audit). The exit row asks for the finite-far culler
    /// *proven able to fail*; `gg-golden`'s `verify-gates` proves it in pixels
    /// and this proves it in the instance list, which is where a five-plane
    /// regression would first show as "nothing was ever dropped".
    #[test]
    fn an_orthographic_extract_drops_what_is_past_its_far_plane() {
        // A 16 m box from 0.1 to 60 m: the middle entity is beyond far and
        // dead ahead, so no side plane can take the credit for removing it.
        let ortho =
            Frustum::from_view_projection(render::orthographic_reverse_z(16.0, 16.0, 0.1, 60.0));
        let world = world_with(&[
            sim::DVec3::new(0.0, 0.0, -10.0),
            sim::DVec3::new(0.0, 0.0, -600.0),
            sim::DVec3::new(0.0, 0.0, -50.0),
        ]);
        let mut out = Extracted::default();
        out.transforms::<Body>(&world, sim::DVec3::ZERO, ortho)
            .unwrap();
        assert_eq!(out.instances.len(), 2);
        assert_eq!(out.culled, 1, "the far plane dropped it");

        // And the same world under the infinite far keeps all three, so the
        // count above is the sixth plane's doing and not the box's width.
        let mut wide = Extracted::default();
        wide.transforms::<Body>(&world, sim::DVec3::ZERO, looking_forward())
            .unwrap();
        assert_eq!(wide.culled, 0, "perspective has no far plane to fail on");
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

    fn skied_world(skies: &[gg_ecs::boundary::Sky]) -> World {
        use gg_ecs::boundary::Sky;
        let mut world = World::new();
        world.register::<Sky>().unwrap();
        for &sky in skies {
            let e = world.spawn();
            world.insert(e, sky).unwrap();
        }
        world
    }

    /// The compositing order, which is the whole of "the innermost volume wins"
    /// (§6 M28) — the shader walks the array and spends a budget, so the array's
    /// order *is* the priority and nothing downstream re-decides it.
    #[test]
    fn the_smallest_volume_composites_first_and_the_unbounded_one_last() {
        use gg_ecs::boundary::Sky;
        let boxed = |half: f32| {
            Sky::daylight(1.0).within(sim::DVec3::ZERO, sim::Vec3::new(half, half, half), 0.5)
        };
        // Spawned largest-first with the world's sky in the middle, so world
        // order is not the answer in any of the three positions.
        let world = skied_world(&[boxed(10.0), Sky::daylight(1.0), boxed(2.0), boxed(6.0)]);
        let out = extract_lights(&world, sim::DVec3::ZERO, Frustum::UNBOUNDED);
        let extents: Vec<f32> = out.skies.iter().map(|s| s.half_extent.x).collect();
        assert_eq!(extents, vec![2.0, 6.0, 10.0, 0.0]);
        assert!(out.skies[3].unbounded(), "the world's sky is the fallback");
        assert_eq!(out.skies_dropped, 0);
    }

    /// The cap keeps what is *nearest*, and can never drop the fallback: an
    /// unbounded sky measures zero distance because it contains everything, so
    /// it sorts first however far the camera has walked (§6 M28).
    #[test]
    fn past_the_cap_the_nearest_volumes_win_and_the_world_sky_survives() {
        use gg_ecs::boundary::Sky;
        let at = |x: f64| {
            Sky::daylight(1.0).within(
                sim::DVec3::new(x, 0.0, 0.0),
                sim::Vec3::new(1.0, 1.0, 1.0),
                0.0,
            )
        };
        // Six bounded volumes marching away from the camera, and the world's sky
        // spawned last so a truncation in spawn order would drop exactly it.
        let mut skies: Vec<Sky> = (0..MAX_SKY + 2).map(|i| at(4.0 * i as f64)).collect();
        skies.push(Sky::daylight(1.0));
        let out = extract_lights(&skied_world(&skies), sim::DVec3::ZERO, Frustum::UNBOUNDED);
        assert_eq!(out.skies.len(), MAX_SKY);
        assert_eq!(out.skies_dropped, 3);
        assert!(
            out.skies.last().is_some_and(ExtractedSky::unbounded),
            "the fallback is never the one dropped: {:?}",
            out.skies
        );
        let kept: Vec<f32> = out.skies[..MAX_SKY - 1]
            .iter()
            .map(|s| s.offset.x)
            .collect();
        assert_eq!(kept, vec![0.0, 4.0, 8.0], "the three nearest boxes");
    }

    /// A zero-intensity sky is no environment, as it has been since §6 M24 — and
    /// it is dropped rather than carried as a black one, so a room that switched
    /// its light off falls through to whatever contains it.
    #[test]
    fn a_dark_volume_is_absent_rather_than_black() {
        use gg_ecs::boundary::Sky;
        let world = skied_world(&[
            Sky::daylight(0.0).within(sim::DVec3::ZERO, sim::Vec3::splat(2.0), 0.0),
            Sky::daylight(1.0),
        ]);
        let out = extract_lights(&world, sim::DVec3::ZERO, Frustum::UNBOUNDED);
        assert_eq!(out.skies.len(), 1);
        assert!(out.skies[0].unbounded());
    }

    /// The weight the shader duplicates: 1 inside, 0 past the fade, and *smooth*
    /// between — a monotone ramp with no step in it, because the step is exactly
    /// what a per-volume environment is built to avoid.
    #[test]
    fn a_volume_fades_outward_from_its_face_and_never_jumps() {
        use gg_ecs::boundary::Sky;
        let world = skied_world(&[Sky::daylight(1.0).within(
            sim::DVec3::ZERO,
            sim::Vec3::new(1.0, 1.0, 1.0),
            2.0,
        )]);
        let sky = extract_lights(&world, sim::DVec3::ZERO, Frustum::UNBOUNDED).skies[0];
        assert_eq!(sky.weight_at(render::Vec3::ZERO), 1.0, "inside is whole");
        assert_eq!(
            sky.weight_at(render::Vec3::new(1.0, 0.0, 0.0)),
            1.0,
            "the face is inside"
        );
        assert_eq!(
            sky.weight_at(render::Vec3::new(3.5, 0.0, 0.0)),
            0.0,
            "past the fade is nothing"
        );
        let mut previous = 1.0;
        for step in 0..=40 {
            let x = 1.0 + step as f32 * 0.05;
            let weight = sky.weight_at(render::Vec3::new(x, 0.0, 0.0));
            assert!(weight <= previous + 1e-6, "the ramp must not rise: at {x}");
            assert!(
                (weight - previous).abs() < 0.15,
                "a fade with a step of {} in it is a pop at {x}",
                (weight - previous).abs()
            );
            previous = weight;
        }
        // And a zero fade is the hard edge it is documented to be, which is the
        // one case the smoothstep cannot be asked to compute.
        let hard = Sky::daylight(1.0).within(sim::DVec3::ZERO, sim::Vec3::splat(1.0), 0.0);
        let hard =
            extract_lights(&skied_world(&[hard]), sim::DVec3::ZERO, Frustum::UNBOUNDED).skies[0];
        assert_eq!(hard.weight_at(render::Vec3::new(1.0, 0.0, 0.0)), 1.0);
        assert_eq!(hard.weight_at(render::Vec3::new(1.01, 0.0, 0.0)), 0.0);
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

    fn eye_at(origin: sim::DVec3) -> Extracted {
        let mut out = Extracted::default();
        out.clear(origin, Frustum::UNBOUNDED);
        out
    }

    #[test]
    fn a_grid_phase_is_a_fraction_of_a_cell_and_repeats_every_cell() {
        // The property the whole thing rests on: two eyes a whole number of
        // cells apart see the same grid, and one a fraction apart does not.
        let axis = render::Vec3::X;
        let period = 0.25;
        let base =
            eye_at(sim::DVec3::new(1.1, 0.0, 0.0)).grid_phase(axis, render::Vec3::ZERO, period);
        assert!((0.0..=1.0).contains(&base), "{base}");
        for cells in [-8.0, -1.0, 3.0, 400.0] {
            let moved = eye_at(sim::DVec3::new(1.1 + cells * 0.25, 0.0, 0.0));
            assert!(
                (moved.grid_phase(axis, render::Vec3::ZERO, period) - base).abs() < 1e-5,
                "{cells} cells along moved the grid"
            );
        }
        let between = eye_at(sim::DVec3::new(1.1 + 0.125, 0.0, 0.0));
        assert!((between.grid_phase(axis, render::Vec3::ZERO, period) - base).abs() > 0.4);
    }

    #[test]
    fn the_phase_wraps_forward_on_both_sides_of_the_origin() {
        // `%` would hand back a negative phase here, and a grid shifted the
        // wrong way is worse than an unshifted one: it crawls at twice the rate.
        let phase = eye_at(sim::DVec3::new(-0.75, 0.0, 0.0)).grid_phase(
            render::Vec3::X,
            render::Vec3::ZERO,
            1.0,
        );
        assert!((phase - 0.25).abs() < 1e-6, "{phase}");
        let none = eye_at(sim::DVec3::new(-4.0, 0.0, 0.0)).grid_phase(
            render::Vec3::X,
            render::Vec3::ZERO,
            1.0,
        );
        assert_eq!(none, 0.0, "a whole number of cells out is no phase at all");
    }

    #[test]
    fn a_planetary_eye_still_resolves_a_fraction_of_a_shadow_texel() {
        // Why this is `f64` and why it is in this crate. At 10^12 m an absolute
        // `f32` has ~65 km of resolution, so a 39 mm texel's phase would be pure
        // noise — and noise in a snap is the crawl it was meant to remove.
        let far = 1.0e12;
        let texel = 0.039_062_5;
        let here = eye_at(sim::DVec3::new(far, 0.0, 0.0)).grid_phase(
            render::Vec3::X,
            render::Vec3::ZERO,
            texel,
        );
        let step = eye_at(sim::DVec3::new(far + f64::from(texel) * 8.0, 0.0, 0.0)).grid_phase(
            render::Vec3::X,
            render::Vec3::ZERO,
            texel,
        );
        assert!((step - here).abs() < 0.01, "{here} vs {step}");
    }

    #[test]
    fn a_grid_with_no_spacing_has_no_phase() {
        let eye = eye_at(sim::DVec3::new(3.7, 0.0, 0.0));
        for period in [0.0, f32::INFINITY, f32::NAN] {
            assert_eq!(
                eye.grid_phase(render::Vec3::X, render::Vec3::ZERO, period),
                0.0,
                "{period}"
            );
        }
    }

    /// A world of one body, movable, so a capture and a move can be staged.
    fn one_body(position: sim::DVec3) -> (World, Entity) {
        let mut world = World::new();
        let e = world.spawn();
        let _ = world.insert(
            e,
            Body {
                position,
                rotation: sim::DQuat::IDENTITY,
            },
        );
        (world, e)
    }

    fn move_to(world: &mut World, entity: Entity, position: sim::DVec3) {
        world.get_mut::<Body>(entity).unwrap().position = position;
    }

    fn drawn(world: &World, out: &mut Extracted) -> Vec<render::Vec3> {
        out.transforms::<Body>(world, sim::DVec3::ZERO, Frustum::UNBOUNDED)
            .unwrap();
        out.instances.iter().map(|i| i.offset).collect()
    }

    #[test]
    fn a_frame_between_two_ticks_draws_between_two_poses() {
        let (mut world, e) = one_body(sim::DVec3::ZERO);
        let mut out = Extracted::default();
        out.capture::<Body>(&world).unwrap();
        move_to(&mut world, e, sim::DVec3::new(4.0, 0.0, 0.0));

        // The three frames a 60 Hz tick is shown across at 240 Hz, and the tick
        // boundary itself — which must land exactly on the new pose, or every
        // tick would end a hair short of where the sim actually put things.
        for (alpha, x) in [(0.0, 4.0), (0.25, 1.0), (0.5, 2.0), (1.0, 4.0)] {
            out.interpolate(alpha);
            assert_eq!(drawn(&world, &mut out)[0].x, x, "alpha {alpha}");
        }
    }

    #[test]
    fn nothing_interpolates_until_a_pose_was_captured_for_it() {
        // The spawn case, and the whole-run case: an entity with no previous
        // pose draws where it is. Sliding in from a table's zero would put every
        // new projectile through the middle of the level on its first frame.
        let (mut world, _) = one_body(sim::DVec3::new(9.0, 0.0, 0.0));
        let mut out = Extracted::default();
        out.interpolate(0.5);
        assert_eq!(drawn(&world, &mut out)[0].x, 9.0, "no capture, no blend");

        out.capture::<Body>(&world).unwrap();
        let fresh = world.spawn();
        world
            .insert(
                fresh,
                Body {
                    position: sim::DVec3::new(-3.0, 0.0, 0.0),
                    rotation: sim::DQuat::IDENTITY,
                },
            )
            .unwrap();
        let offsets = drawn(&world, &mut out);
        assert_eq!(offsets.len(), 2);
        assert!(offsets.iter().any(|o| o.x == -3.0), "the new one popped in");
    }

    #[test]
    fn a_reused_index_does_not_inherit_the_dead_entitys_pose() {
        // The generation check, in the one situation that reaches it: the
        // allocator hands indices back LIFO, so the *next* spawn after a despawn
        // lands on the slot the dead body's pose is still sitting in.
        let (mut world, e) = one_body(sim::DVec3::new(100.0, 0.0, 0.0));
        let mut out = Extracted::default();
        out.capture::<Body>(&world).unwrap();
        world.despawn(e);
        let reborn = world.spawn();
        assert_eq!(reborn.index(), e.index(), "the test needs the reuse");
        world
            .insert(
                reborn,
                Body {
                    position: sim::DVec3::ZERO,
                    rotation: sim::DQuat::IDENTITY,
                },
            )
            .unwrap();
        out.interpolate(0.5);
        assert_eq!(drawn(&world, &mut out)[0].x, 0.0, "not halfway from 100");
    }

    #[test]
    fn a_capture_an_entity_missed_does_not_blend_it_out_of_an_old_pose() {
        // What the stamp is for. Slots are overwritten and never cleared, so a
        // body that stops being drawn and starts again keeps a row in the table
        // — and without the stamp it would slide back in from wherever it was
        // when it left, which may be the other side of the level.
        let (mut world, e) = one_body(sim::DVec3::new(50.0, 0.0, 0.0));
        let mut out = Extracted::default();
        out.capture::<Body>(&world).unwrap();
        assert!(world.remove::<Body>(e));
        out.capture::<Body>(&world).unwrap(); // this capture does not see it
        world
            .insert(
                e,
                Body {
                    position: sim::DVec3::ZERO,
                    rotation: sim::DQuat::IDENTITY,
                },
            )
            .unwrap();
        out.interpolate(0.5);
        assert_eq!(drawn(&world, &mut out)[0].x, 0.0, "not halfway from 50");
    }

    #[test]
    fn the_blend_happens_before_the_narrowing() {
        // The membrane (§1.4) applied to the second pair of positions this crate
        // now holds. At 10^12 m an absolute `f32` resolves ~65 km, so two poses
        // one metre apart would blend to noise if either narrowed first.
        let far = 1.0e12;
        let (mut world, e) = one_body(sim::DVec3::new(far, 0.0, 0.0));
        let mut out = Extracted::default();
        out.capture::<Body>(&world).unwrap();
        move_to(&mut world, e, sim::DVec3::new(far + 1.0, 0.0, 0.0));
        out.interpolate(0.5);
        out.transforms::<Body>(&world, sim::DVec3::new(far, 0.0, 0.0), Frustum::UNBOUNDED)
            .unwrap();
        assert_eq!(out.instances[0].offset, render::Vec3::new(0.5, 0.0, 0.0));
    }

    #[test]
    fn a_locked_pace_extracts_the_tick_exactly() {
        // Zero alpha is what every replay, golden run and headless tier reports,
        // and it must reproduce the pre-interpolation picture bit for bit — not
        // approximately, or a blessed image would drift the day this landed.
        let (mut world, e) = one_body(sim::DVec3::new(1.0 / 3.0, 7.7, -0.1));
        let mut captured = Extracted::default();
        captured.capture::<Body>(&world).unwrap();
        move_to(&mut world, e, sim::DVec3::new(2.0 / 3.0, 1.1, 9.0));

        let mut untouched = Extracted::default();
        assert_eq!(drawn(&world, &mut captured), drawn(&world, &mut untouched));
    }

    #[test]
    fn a_turn_takes_the_short_way_round_and_stays_a_unit_quaternion() {
        // `q` and `-q` are one orientation. A body that turned a few degrees can
        // arrive at the far representation, and lerping the components then goes
        // the long way — a full revolution inside one frame, at exactly the
        // moment someone is turning quickly.
        let small = sim::DQuat::from_axis_angle(sim::DVec3::Y, 0.1);
        let far = sim::DQuat::from_xyzw(-small.x, -small.y, -small.z, -small.w);
        let half = blend_rotation(sim::DQuat::IDENTITY, far, 0.5);
        let short = blend_rotation(sim::DQuat::IDENTITY, small, 0.5);
        assert!((half.dot(short).abs() - 1.0).abs() < 1e-12, "{half:?}");
        assert!((half.length() - 1.0).abs() < 1e-12, "not normalized");
        // And the degenerate case hazard 6 forbids handing back a NaN for.
        let opposite = sim::DQuat::from_xyzw(0.0, 0.0, 0.0, -1.0);
        let flipped = blend_rotation(sim::DQuat::IDENTITY, opposite, 0.5);
        assert!(flipped.length().is_finite());
    }

    #[test]
    fn an_eye_blends_its_angles_the_short_way_and_its_position_in_f64() {
        use gg_ecs::boundary::Eye;
        let far = 1.0e12;
        let previous = Eye {
            position: sim::DVec3::new(far, 0.0, 0.0),
            // Just under +π, turning positive: the wrap is the case that spins
            // a camera through a whole revolution in one frame if unhandled.
            yaw: core::f32::consts::PI - 0.05,
            pitch: 0.2,
            ortho: 0.0,
            reserved: 0,
        };
        let current = Eye {
            position: sim::DVec3::new(far + 2.0, 0.0, 0.0),
            yaw: -core::f32::consts::PI + 0.05,
            pitch: 0.4,
            ..previous
        };
        let mid = blend_eye(previous, current, 0.5);
        assert_eq!(mid.position.x - far, 1.0, "narrowed nothing");
        assert!((mid.pitch - 0.3).abs() < 1e-6);
        // Halfway across the seam is π itself, whichever sign it comes out as.
        assert!(
            (mid.yaw.abs() - core::f32::consts::PI).abs() < 1e-5,
            "yaw went the long way: {}",
            mid.yaw
        );
    }

    #[test]
    fn an_eye_never_blends_across_the_perspective_sentinel() {
        use gg_ecs::boundary::Eye;
        // `ortho == 0.0` means perspective (§6 M20), so a frame halfway between
        // a flat eye and a perspective one would render a third projection that
        // is neither. Two eyes that agree about which they are still blend.
        let flat = Eye {
            ortho: 8.0,
            ..Eye::ORIGIN
        };
        let deep = Eye::ORIGIN;
        assert_eq!(
            blend_eye(flat, deep, 0.5).ortho,
            0.0,
            "switches on the tick"
        );
        assert_eq!(blend_eye(deep, flat, 0.5).ortho, 8.0);
        let zoomed = Eye {
            ortho: 4.0,
            ..Eye::ORIGIN
        };
        assert_eq!(blend_eye(flat, zoomed, 0.5).ortho, 6.0, "a zoom does blend");
    }

    /// An `Extracted` holding `previous` as its captured eye, `alpha` past it.
    fn staged(previous: Eye, alpha: f32) -> Extracted {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, previous).unwrap();
        let mut extracted = Extracted::default();
        extracted.capture_eye(&world).unwrap();
        extracted.interpolate(alpha);
        extracted
    }

    /// The property every golden, replay and headless tier rests on (§6 M56):
    /// a latch with nothing in it is the identity, so a frame with no mouse
    /// travel renders bit-for-bit what it rendered before the latch existed.
    ///
    /// Asserted against a *nonzero* alpha and two different eyes, because at
    /// alpha zero, or between two equal eyes, an `eye` that ignored its latch
    /// entirely would also pass.
    #[test]
    fn a_latch_with_no_travel_in_it_renders_exactly_the_interpolated_tick() {
        let previous = Eye::at(sim::DVec3::new(1.0, 2.0, 3.0), 0.4, -0.2);
        let current = Eye::at(sim::DVec3::new(4.0, 2.0, 3.0), 0.9, 0.3);
        let extracted = staged(previous, 0.37);
        let blended = extracted.eye(current, None);
        assert_ne!(blended, current, "alpha is doing something to blend past");
        assert_eq!(
            extracted.eye(current, Some(Latch::default())),
            blended,
            "an empty latch moves nothing"
        );
    }

    /// The algebra the milestone is: a hand turning at a constant rate must land
    /// on the angle the hand has actually reached, whatever the frame caught the
    /// tick at. The blend alone cannot — that is the lag being removed.
    #[test]
    fn a_latched_view_lands_on_the_hand_and_the_blend_lands_behind_it() {
        // One tick turned -0.10 rad, all of it the hand's; the frame sits 0.25
        // of a tick past it with another -0.025 rad in hand unspent.
        let look = Look::fly(0, 1, 1.0, 0.0);
        let previous = Eye::at(sim::DVec3::ZERO, 0.0, 0.0);
        let current = Eye::at(sim::DVec3::ZERO, -0.10, 0.0);
        let extracted = staged(previous, 0.25);
        let latch = Latch::of(look, (0.025, 0.0), (0.10, 0.0));

        // Where the hand is: a tick's worth plus a quarter of one more.
        let hand = -0.125;
        let shown = extracted.eye(current, Some(latch)).yaw;
        assert!((shown - hand).abs() < 1e-6, "{shown} is not {hand}");

        // And the interpolated frame is a whole tick period behind it, which is
        // the number `gg-tools pace` prints as `age`.
        let blended = extracted.eye(current, None).yaw;
        assert!((blended - -0.025).abs() < 1e-6, "{blended}");
    }

    /// The half the additive form exists for: a rotation the *game* drove — a
    /// recoil kick, a cutscene — is sim motion and keeps its interpolation. A
    /// latch that replaced the blend instead of adding to it would step it at
    /// the tick rate, which is the judder this whole line of work removes.
    #[test]
    fn a_turn_the_game_drove_still_interpolates_under_a_latch() {
        let look = Look::fly(0, 1, 1.0, 0.0);
        let previous = Eye::at(sim::DVec3::ZERO, 0.0, 0.0);
        let current = Eye::at(sim::DVec3::ZERO, 0.8, 0.0);
        let extracted = staged(previous, 0.25);
        // No hand at all: nothing pending, and the tick spent no motion.
        let latch = Latch::of(look, (0.0, 0.0), (0.0, 0.0));
        let shown = extracted.eye(current, Some(latch)).yaw;
        assert!((shown - 0.2).abs() < 1e-6, "{shown} is not the blend");
    }

    /// A latch turns the angles and nothing else — position keeps the blend,
    /// because extrapolating a body puts the camera through a wall the sim will
    /// not let it reach.
    #[test]
    fn a_latch_moves_no_position() {
        let look = Look::fly(0, 1, 1.0, 0.0);
        let previous = Eye::at(sim::DVec3::ZERO, 0.0, 0.0);
        let current = Eye::at(sim::DVec3::new(10.0, 0.0, 0.0), 0.0, 0.0);
        let extracted = staged(previous, 0.5);
        let latch = Latch::of(look, (1.0, 1.0), (1.0, 1.0));
        assert_eq!(
            extracted.eye(current, Some(latch)).position,
            extracted.eye(current, None).position
        );
    }

    /// The clamp is the game's, and the latch has to honour it: without this the
    /// picture turns past a limit the next tick refuses and the hand feels the
    /// snap back.
    #[test]
    fn a_latched_pitch_stops_where_the_game_said_it_would() {
        let look = Look::fly(0, 1, 1.0, 1.55);
        let previous = Eye::at(sim::DVec3::ZERO, 0.0, 1.5);
        let current = Eye::at(sim::DVec3::ZERO, 0.0, 1.54);
        let extracted = staged(previous, 1.0);
        // A flick that would put the picture at 2.54 rad — past vertical.
        let latch = Latch::of(look, (-1.0, -1.0), (0.0, 0.0));
        assert_eq!(extracted.eye(current, Some(latch)).pitch, 1.55);
        // And a game that declared no limit is left alone.
        let free = Latch::of(Look::fly(0, 1, 1.0, 0.0), (-1.0, -1.0), (0.0, 0.0));
        assert!(extracted.eye(current, Some(free)).pitch > 2.5);
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
