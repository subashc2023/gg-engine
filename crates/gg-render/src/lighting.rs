//! One frame's lighting environment, and the sun's shadow projection (§6 M11).
//!
//! Every shading pipeline reads the same block through one device address rather
//! than through push constants: the sun's matrix alone is 64 bytes and Vulkan
//! guarantees 128 of push, so a per-draw copy would leave no room for a draw's
//! own parameters. It was also the shape a clustered light list wanted, which is
//! what §6 M30 cashed in: the froxel ranges and their indices follow the lights
//! in this same region, at constant offsets, so the grid cost the shader no
//! second address and [`MAX_POINT`](gg_extract::MAX_POINT) could go from 32 to
//! 256 without one.
//!
//! # What is stated rather than implemented
//!
//! **Cascades, and the first directional light casts them** (§6 M15.3). A second
//! sun casting its own set is a second set of passes and a second cascade
//! decision; the rest of the directional lights still light, they just do not
//! occlude — visible and explicable rather than silently wrong.
//!
//! **Every cascade is texel-snapped**, and the snap is split across the §1.4
//! membrane on purpose. A cascade is centred on a slice of the view frustum, so
//! its texel grid slides whenever the camera moves *or turns*, and every
//! quantized shadow edge would crawl across the world with it. Locking the grid
//! needs the camera's *absolute* position, which is `f64` and deliberately
//! unreachable from this crate — so the phase is
//! [`Extracted::grid_phase`](gg_extract::Extracted::grid_phase)'s to compute and
//! this crate's only to apply. The rotation half is why the fit is a bounding
//! sphere: a snap can only lock a grid whose *spacing* is already stable.

use bytemuck::Zeroable as _;
use gg_extract::{
    Extracted, ExtractedLight, ExtractedSky, MAX_DIRECTIONAL, MAX_POINT, MAX_SKY, SkyLook, light,
};
use gg_math::render;
use gg_rhi::{BufferDesc, BufferHandle, DeviceAddress, FRAMES_IN_FLIGHT, RhiError, Sampler};

use crate::cluster::{self, Assignment, Grid};
use crate::{GpuHost, View, cvars, lamp, probe, sky, srgb_to_linear};

/// The per-frame block, mirroring `include/pbr.slang`'s `Frame`. The shader
/// reads it as scalars through a device address, so this layout is the whole of
/// the agreement — and `FRAME_STRIDE` there is the other half of it.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuFrame {
    /// Camera-relative world → clip. Read by the box pass, whose push block has
    /// no room for it; the pack pass carries a finished product per instance.
    view_projection: [[f32; 4]; 4],
    /// Linear rgb; `w` unused.
    ambient: [f32; 4],
    light_count: u32,
    shadow_sampler: u32,
    /// Cascades this frame fitted. Zero is the whole of "nothing casts" — a
    /// separate enabled flag would be a second thing to keep in step with the
    /// graph, which allocates exactly this many shadow maps.
    cascade_count: u32,
    /// 1 / shadow map edge, in uv. One value, because every cascade is the same
    /// size in texels — which is what makes the 3x3 kernel mean the same thing
    /// in each of them.
    shadow_texel: f32,
    /// The direction the casting light travels, unit; zero when none casts. The
    /// shader needs the angle, and a matrix it cannot cheaply invert is what it
    /// otherwise has.
    sun_direction: [f32; 3],
    /// Normal-offset reach in **shadow texels**, not metres — the shader scales
    /// it by the selected cascade's `texel_world` and by sin(incidence).
    shadow_normal_bias: f32,
    /// The angle-free part of the normal offset, also in texels.
    shadow_depth_bias: f32,
    /// Fraction of a cascade over which it cross-fades into the next.
    shadow_blend: f32,
    /// Tangent of the sun's angular **radius** — the penumbra's growth per metre
    /// of blocker-to-receiver gap (§6 M23). The tangent and not the angle: the
    /// shader multiplies, and a `tan` per fragment for a constant is waste.
    sun_tan: f32,
    /// The screen-pixel penumbra floor, in pixels.
    shadow_softness: f32,
    /// Penumbra cap and blocker-search radius, in shadow texels.
    shadow_penumbra: f32,
    /// Taps in the filter disk. The search uses a quarter of them.
    shadow_taps: u32,
    /// How many of [`environments`](GpuFrame::environments) are live, innermost
    /// first. Zero is the whole of "this frame has no environment" — fall back
    /// to the flat [`ambient`](GpuFrame::ambient), which is every scene blessed
    /// before §6 M24 and every game that declares no sky.
    ///
    /// A count where M24 had a flag (§6 M28), and the same thing at one: the
    /// shader's guard is still `> 0` and a scene with one sky reads exactly the
    /// bytes it read before.
    environment_count: u32,
    /// Directional lights at the **front** of the light array (§6 M30).
    ///
    /// The shader loops these by count and the rest by froxel list, which is the
    /// whole of the split: a light with no position is in every froxel, so
    /// putting it in one would be storing the same answer three thousand times.
    /// Extract is what makes the front of the array the right place to look.
    directional_count: u32,
    /// The direction the camera looks, unit — what a fragment's camera-relative
    /// position projects onto to get the view depth its froxel is chosen by.
    ///
    /// Carried rather than derived: under perspective the projection's row 2 is
    /// `[0, 0, 0, near]` by construction, so the matrix the shader already holds
    /// does not contain it.
    view_forward: [f32; 3],
    /// `slice = log2(depth) * scale + bias`, floored and clamped — `cluster.rs`.
    cluster_scale: f32,
    cluster_bias: f32,
    /// Froxels per pixel, across and down. The shader multiplies
    /// `SV_Position.xy` by this, so a viewport-sized attachment and a
    /// window-sized one give the same grid over their own pixels.
    cluster_tile: [f32; 2],
    /// Point lights that cast, sitting immediately after the directional ones
    /// (§6 M31). The shader's whole test for "does this light occlude" is
    /// `index < directional_count + lamp_count` — extract's nearest-first order
    /// is what makes a prefix the right answer, and `lamp`'s header is where
    /// that is argued.
    lamp_count: u32,
    /// The face atlas in the global sampled-image array. Unread at
    /// [`lamp_count`](GpuFrame::lamp_count) zero, which is the only reason it
    /// may be slot zero.
    lamp_texture: u32,
    /// Tile space → atlas uv: one sixth across, one row of live lamps down.
    lamp_uv_scale: [f32; 2],
    /// One face texel in tile space — the unit the lamp filter's taps are in,
    /// and what its clamp insets by half of.
    lamp_inv_tile: f32,
    /// Normal-offset reach and the angle-free part, both in **face** texels —
    /// [`shadow_normal_bias`](GpuFrame::shadow_normal_bias)'s unit against a
    /// texel whose world size grows with distance from the lamp rather than
    /// staying fixed across a slab.
    lamp_normal_bias: f32,
    lamp_depth_bias: f32,
    /// Taps in the lamp filter. Its own knob rather than
    /// [`shadow_taps`](GpuFrame::shadow_taps): this one is paid per casting
    /// lamp per fragment, where the sun's is paid once.
    lamp_taps: u32,
    /// Which shading corrections are on — `SHADING_MULTISCATTER`,
    /// `SHADING_LOBE`, mirrored in `include/pbr.slang` (§6 M33).
    ///
    /// A bitfield where every knob above is its own field, and that is the
    /// block's own arithmetic rather than a preference: this is the *last* word
    /// of the padding M27–M31 spent down, and the arrays behind it start at 208
    /// because 208 is a multiple of 16. A fifty-third scalar would have moved
    /// every offset in the file to carry one bit.
    ///
    /// Both default on. Off is the pre-M33 shading exactly rather than a model
    /// of it, which is what lets `gg-tools furnace` measure two algorithms in
    /// one binary instead of two builds — `r.shadow_cull`'s argument at §6 M32.
    shading: u32,
    /// §6 M34's split-sum table — `EXTENT²` pairs of `f32`, `n·v` fastest and
    /// stored against `sqrt(n·v)`.
    ///
    /// An address and not a texture, which is §4.3's rule rather than a
    /// preference: this is data a shader indexes, not an image, and a bindless
    /// slot would additionally have made it a half-float — the one place in the
    /// frame where precision is spent on a reciprocal.
    ///
    /// Where the padding went. M33's `shading` was the last of the words M27–M31
    /// spent down, so this is a fresh 16 bytes and the arrays behind it moved
    /// from 208 to 224. The three that follow are the next milestone's, and
    /// naming them that way is what stops the *next* scalar from being the one
    /// that moves every offset in the file.
    split_sum: u64,
    /// §6 M35's occlusion target in the global sampled-image array, or zero.
    ///
    /// One of the two words M34 set aside for the next milestone, which is why
    /// this block did not grow and every offset below is unchanged. Zero is the
    /// whole of "this frame has no AO" — `SHADING_AO` is what the shader tests,
    /// and the graph is what makes the two agree: with `r.ao` off neither pass
    /// is declared, so there is no target to have a slot.
    ao_texture: u32,
    /// §6 M36's irradiance field: the L1 records, and the octahedral distance
    /// tiles beside them. Both are images the renderer owns across frames rather
    /// than graph transients (`graph::Frame::imported`), which is why their slots
    /// are stable and this block carries them like any other texture.
    ///
    /// Zero is a legitimate slot, so `SHADING_GI` is the flag — `lamp_texture`'s
    /// argument at §6 M31. It is clear whenever the grid has not been fitted,
    /// and then nothing below is read.
    gi_sh_texture: u32,
    gi_moment_texture: u32,
    /// One distance tile's edge in texels. A float because every use of it in
    /// the shader is a uv scale.
    gi_moment_edge: f32,
    /// Metres between probes, and the grid's corner probe camera-relative — what
    /// turns a shading point into a cell and three interpolants.
    gi_spacing: f32,
    gi_counts: [u32; 3],
    gi_origin: [f32; 3],
    /// Metres a shading point is pushed along its normal before it is located in
    /// the grid. Without it every surface is at distance zero from the geometry
    /// the probes recorded — itself — and every visibility bound reads as
    /// occluded by the very surface being shaded.
    gi_bias: f32,
    /// This frame's gather batch: six projections and a position per probe,
    /// `probe::SLOT_BYTES` apart.
    ///
    /// An address rather than an array in this block, unlike the lamps' — the
    /// batch is `r.gi_rate` probes and that is a session's knob, so the fixed
    /// array would have to be sized for the largest rate anyone might set and
    /// written for the rate they did. Zero when nothing is gathering, which the
    /// graph guarantees by declaring no probe pass then.
    probe_views: u64,
    /// This frame's environments, innermost first (§6 M28).
    ///
    /// A fixed array in the block rather than a second buffer, `cascades`'
    /// reasoning at three times the size: it is 768 bytes against one address
    /// the shader already holds, and the fragment reads only the entries whose
    /// weight is nonzero — which in every scene with one sky is one of them.
    environments: [GpuEnvironment; MAX_SKY],
    /// Nearest first, `cascade_count` of them live. A fixed array rather than a
    /// second buffer: it is 320 bytes, and one address the shader already has
    /// beats a second one it would have to be handed.
    cascades: [GpuCascade; MAX_CASCADES],
    /// This frame's casting lamps, `lamp_count` of them live (§6 M31).
    ///
    /// Three kilobytes at the ceiling, which is the largest thing in this block
    /// by some way and is still the same trade `cascades` and `environments`
    /// made: one address the shader already holds, against a second one every
    /// pipeline would have to be handed. The vertex shader reads a face's matrix
    /// to render it and the fragment shader reads the same matrix to look it up,
    /// so there is exactly one projection per face and no convention for the two
    /// to disagree about — which is the bug §6 M30's grid shipped with.
    lamps: [GpuLamp; lamp::MAX_LAMPS],
}

/// One casting lamp as the shader reads it, mirroring `include/pbr.slang`'s
/// `Lamp`.
///
/// Nothing but the six projections: the light's position and range are already
/// in its [`GpuLight`] record, which the fragment has in hand when it asks, and
/// the lamp's row in the atlas is its index.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLamp {
    faces: [[[f32; 4]; 4]; lamp::FACES],
}

/// One environment as the shader reads it, mirroring `include/pbr.slang`'s
/// `Environment` (§6 M28).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuEnvironment {
    /// Order-2 spherical-harmonic **radiance** coefficients (§6 M24) — not
    /// irradiance: the diffuse path applies the cosine kernel's band factors and
    /// the specular path applies a roughness kernel's, and pre-convolving here
    /// would serve one of them and lie to the other.
    ///
    /// `float4` a piece with `w` unused, because a `float3` array's stride in a
    /// block read as scalars is a fact nobody should have to hold in their head.
    ///
    /// [`Sky::intensity`](gg_ecs::boundary::Sky::intensity) is already folded in
    /// — nine multiplies once a frame rather than one per fragment.
    sh: [[f32; 4]; sky::SH_COEFFICIENTS],
    /// The volume's centre, camera-relative. Unread when
    /// [`half_extent`](GpuEnvironment::half_extent) is zero.
    offset: [f32; 3],
    /// Metres outside the box over which this falls to nothing. Zero is a hard
    /// edge.
    fade: f32,
    /// Half the box. **Zero is unbounded**, and that is the one value the
    /// shader tests: an unbounded environment answers weight 1 everywhere, which
    /// is what makes the last entry a fallback without a second flag.
    half_extent: [f32; 3],
    /// [`Sky::intensity`](gg_ecs::boundary::Sky::intensity) again, *unapplied* —
    /// what the shader multiplies the chain's texels by.
    ///
    /// The texels are the panorama's own absolute radiance and are not scaled at
    /// import: a pack holds the measurement, and exposure is the world's to set.
    /// Both halves must carry the same number, or a mirror stops matching the
    /// room it stands in (§6 M27).
    radiance_scale: f32,
    /// The prefiltered chain in the global sampled-image array, or zero — which
    /// is slot zero and therefore not a valid answer on its own, so
    /// [`radiance_levels`](GpuEnvironment::radiance_levels) is the flag.
    radiance_texture: u32,
    /// Levels in that chain, and the switch: zero is no chain at all, which is
    /// every sky that is still a gradient. The shader turns a roughness into a
    /// mip by scaling with this, so it has to be here rather than queried off
    /// the texture.
    radiance_levels: u32,
    /// Zero, to a multiple of 16 — [`GpuFrame::reserved`]'s reason.
    reserved: [u32; 2],
}

/// A compiled environment the pack resolved, ready for the frame block (§6 M27).
///
/// Assembled by the caller rather than looked up here, because it takes both
/// halves of the content layer — the blob's SH out of [`Content`](crate::content::Content)
/// and the chain's bindless index out of
/// [`Residency`](crate::residency::Residency) — and `Lighting` holds neither. It
/// exists at all only when *both* are ready: a chain still streaming falls back
/// to the gradient for the frame rather than sampling slot zero.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Radiance {
    /// The prefiltered chain's slot in the global sampled-image array.
    pub texture: gg_rhi::TextureIndex,
    /// Levels in it. Never zero — a chain with no levels is not one.
    pub levels: u32,
    /// The panorama's own order-2 projection, which *replaces* the gradient's
    /// rather than adding to it. Scaled by the sky's intensity on the way in,
    /// so the frame block's `sky` stays the flag it has been since M24.
    pub sh: [[f32; 4]; sky::SH_COEFFICIENTS],
    /// That same intensity, unapplied — what the shader multiplies the chain's
    /// texels by, since those are the panorama's absolute radiance.
    pub scale: f32,
}

/// The bindless slots the graph acquired for this frame's shadow maps: one per
/// cascade, then the lamp atlas.
///
/// Handed in together rather than as two arguments because they are one fact —
/// what the graph actually allocated — and because a `Lighting` that guessed
/// either would be writing a block whose lookups land in whatever else is in
/// those slots. Both are short of what was *planned* whenever the graph declined
/// to allocate, and both truncate the block's counts rather than erroring.
pub(crate) struct Maps<'a> {
    pub(crate) cascades: &'a [gg_rhi::TextureIndex],
    pub(crate) lamps: Option<gg_rhi::TextureIndex>,
    /// The occlusion target the AO pass wrote (§6 M35), if this frame ran one.
    pub(crate) occlusion: Option<gg_rhi::TextureIndex>,
    /// The irradiance field, when one exists and both its images reached the
    /// bindless array (§6 M36). `None` leaves `SHADING_GI` clear, which is the
    /// shading path exactly as M35 left it.
    pub(crate) field: Option<probe::Field>,
    /// This frame's gather batch, staged. Separate from `field` because a frame
    /// gathers before it samples: the first frame after the images are created
    /// has no field to read and six projections per probe to render through, and
    /// a shader handed zero for this would dereference it.
    pub(crate) probe_views: gg_rhi::DeviceAddress,
}

/// One cascade as the shader reads it, mirroring `include/pbr.slang`'s
/// `Cascade`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuCascade {
    view_projection: [[f32; 4]; 4],
    /// This cascade's own map, in the global sampled-image array.
    texture: u32,
    /// Metres one of its texels covers — the bias's unit, and different per
    /// cascade, which is exactly what a single-slab engine did not have to say.
    texel_world: f32,
    /// View-space distance it reaches. Unread by the shading path; carried so a
    /// debug view can colour by cascade without a second source of truth.
    split_far: f32,
    /// Metres one unit of this cascade's clip depth covers — what turns the gap
    /// between a blocker's recorded depth and the receiver's back into the
    /// distance the penumbra grows with (§6 M23). Per cascade for the same
    /// reason `texel_world` is: each slab is fitted to its own sphere.
    depth_world: f32,
}

/// Multiple-scattering compensation is on — `r.multiscatter` (§6 M33).
const SHADING_MULTISCATTER: u32 = 1;
/// The chain is sampled along the lobe's dominant direction — `r.lobe` (§6 M33).
const SHADING_LOBE: u32 = 2;
/// The split-sum's second integral comes from the table — `r.lut` (§6 M34).
const SHADING_LUT: u32 = 4;
/// The ambient terms are occluded by the depth buffer — `r.ao` (§6 M35).
const SHADING_AO: u32 = 8;
/// The diffuse term comes from the probe field — `r.gi` (§6 M36).
const SHADING_GI: u32 = 16;

/// [`GpuFrame::shading`], read off the CVars once a frame.
///
/// Read here and not in the shader because a CVar is a host thing and a bit is
/// what crosses; the shader tests bits and has no idea either was a knob.
fn shading_flags() -> u32 {
    let mut flags = 0;
    if cvars::MULTISCATTER.bool() {
        flags |= SHADING_MULTISCATTER;
    }
    if cvars::LOBE.bool() {
        flags |= SHADING_LOBE;
    }
    if cvars::LUT.bool() {
        flags |= SHADING_LUT;
    }
    flags
}

/// [`shading_flags`] with the occlusion bit set to what this *frame* actually
/// got. Separate because `r.ao` is a request and a slot is a fact: a frame that
/// asked for AO and whose target never reached the bindless array must not tell
/// the shader to sample slot zero.
fn shading_flags_with(ao: Option<gg_rhi::TextureIndex>, gi: bool) -> u32 {
    shading_flags() | if ao.is_some() { SHADING_AO } else { 0 } | if gi { SHADING_GI } else { 0 }
}

const _: () = {
    assert!(core::mem::size_of::<GpuCascade>() == 80);
    assert!(core::mem::size_of::<GpuEnvironment>() == 192);
    assert!(core::mem::size_of::<GpuLamp>() == 384);
    assert!(core::mem::size_of::<GpuFrame>() == 4432);
    assert!(core::mem::offset_of!(GpuFrame, ambient) == 64);
    assert!(core::mem::offset_of!(GpuFrame, light_count) == 80);
    assert!(core::mem::offset_of!(GpuFrame, sun_direction) == 96);
    assert!(core::mem::offset_of!(GpuFrame, directional_count) == 140);
    assert!(core::mem::offset_of!(GpuFrame, lamp_count) == 172);
    assert!(core::mem::offset_of!(GpuFrame, split_sum) == 208);
    // §6 M36's eleven words, and the first block growth since M24: 220 was the
    // second and last of the words M34 set aside, so the arrays moved from 224
    // to 272 and every base in `pbr.slang` moved with them.
    assert!(core::mem::offset_of!(GpuFrame, gi_sh_texture) == 220);
    assert!(core::mem::offset_of!(GpuFrame, gi_counts) == 236);
    assert!(core::mem::offset_of!(GpuFrame, gi_origin) == 248);
    assert!(core::mem::offset_of!(GpuFrame, probe_views) == 264);
    assert!(core::mem::offset_of!(GpuFrame, environments) == 272);
    assert!(core::mem::offset_of!(GpuFrame, cascades) == 1040);
    assert!(core::mem::offset_of!(GpuFrame, lamps) == 1360);
    // `pbr.slang` derives these two from `FRAME_STRIDE`, `LIGHT_STRIDE` and its
    // own grid constants. Written out here so the derivation has something to be
    // wrong against — a froxel read at the wrong offset is a picture, not a
    // crash.
    assert!(MAX_LIGHTS == 260);
    assert!(CLUSTER_BASE == 16912);
    assert!(INDEX_BASE == 41488);
    assert!(cluster::CLUSTERS == 3072);
    // The coefficients are still the first thing in an environment, so the
    // shader's `SH_BASE` is `ENV_BASE` and M24's loader reads the innermost
    // environment by doing exactly what it always did.
    assert!(core::mem::offset_of!(GpuEnvironment, sh) == 0);
    assert!(core::mem::offset_of!(GpuEnvironment, half_extent) == 160);
    // `pbr.slang`'s SPLIT_SUM_EXTENT indexes the table by this; a drift reads
    // the wrong row and returns a plausible wrong albedo rather than crashing.
    assert!(crate::split_sum::EXTENT == 16);
};

/// One light as the shader reads it, mirroring `include/pbr.slang`'s `Light`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLight {
    position: [f32; 3],
    range: f32,
    direction: [f32; 3],
    kind: u32,
    /// Linear rgb — the sRGB decode happens here, beside the one every tint
    /// already goes through, because extract carries colour and does not convert
    /// it (§4.5).
    color: [f32; 3],
    intensity: f32,
}

const _: () = assert!(core::mem::size_of::<GpuLight>() == 48);

/// Lights one frame's buffer has room for — extract's two caps, which is the
/// only place they are decided.
const MAX_LIGHTS: usize = MAX_DIRECTIONAL + MAX_POINT;

/// Where the froxel ranges start, and where their indices do (§6 M30). Both are
/// constants because the grid is: a fixed count of froxels means a fixed offset,
/// and a fixed offset means the shader needs neither a second address nor a
/// third field to find them.
const CLUSTER_BASE: u64 =
    (core::mem::size_of::<GpuFrame>() + MAX_LIGHTS * core::mem::size_of::<GpuLight>()) as u64;
const INDEX_BASE: u64 = CLUSTER_BASE + (cluster::CLUSTERS * 8) as u64;

const SLOT_BYTES: u64 = INDEX_BASE + (cluster::MAX_INDICES * 4) as u64;

/// Cascades one frame may carry. Four, because the practical split scheme puts
/// the fourth's far plane at ~30x the first's and a fifth buys range nothing in
/// view is at; each is a full [`cvars::SHADOW_SIZE`]² map, so the cap is memory
/// and draw calls as much as it is quality.
pub(crate) const MAX_CASCADES: usize = 4;

/// One cascade: a slice of the view frustum, fitted with its own shadow map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cascade {
    /// Camera-relative world → this cascade's clip space.
    pub(crate) view_projection: render::Mat4,
    /// Camera-relative centre of the slab and the sphere bounding it. What the
    /// per-cascade caster cull tests against — a cascade that redrew the whole
    /// world would multiply the shadow pass by [`MAX_CASCADES`] for geometry it
    /// cannot see.
    pub(crate) centre: render::Vec3,
    pub(crate) radius: f32,
    /// Metres one shadow texel covers. The unit every acne knob is expressed in
    /// (§6 M11's exit row): the map's sampling footprint is a texel wide however
    /// large the cascade, so a bias in metres is a bias that stops being right
    /// the moment a split moves — which, with cascades, it does per cascade.
    pub(crate) texel_world: f32,
    /// View-space distance this cascade reaches. Carried for the graph dump;
    /// selection is by containment, not by this.
    pub(crate) split_far: f32,
    /// Metres one unit of clip depth covers — the slab's full depth, since
    /// `orthographic_reverse_z` maps `[near, far]` onto `[1, 0]`. What the
    /// blocker search turns a depth difference back into a distance with.
    pub(crate) depth_world: f32,
}

/// What the sun asks of the graph this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sun {
    /// The direction the light travels, unit.
    pub(crate) direction: render::Vec3,
    /// Shadow map edge, in texels — one size for every cascade, which is what
    /// makes the kernel's three-texel width mean the same thing in each.
    pub(crate) size: u32,
    /// The light basis every cascade shares, as the two axes across the slab.
    /// Carried because the caster cull has to know which way the slab's *square*
    /// is turned — a distance from its axis cannot tell a corner from an edge
    /// (see [`crate::casts_into`]).
    pub(crate) right: render::Vec3,
    pub(crate) above: render::Vec3,
    cascades: [Cascade; MAX_CASCADES],
    count: usize,
}

impl Sun {
    /// The cascades, nearest first.
    pub(crate) fn cascades(&self) -> &[Cascade] {
        &self.cascades[..self.count]
    }

    /// The casting light's cascades, or `None` when nothing casts.
    ///
    /// The first directional light in extract's order — which is world iteration
    /// order, and therefore a function of world state (§4.2).
    pub(crate) fn of(extracted: &Extracted, view: &View, extent: (u32, u32)) -> Option<Self> {
        // Asked of extract rather than found here: the same call the shell swept
        // its caster frustum along, so the light the cascades are fitted to and
        // the light the culler kept casters for are one light by construction.
        let direction = extracted.sun()?.normalize_or_zero();
        let size = cvars::SHADOW_SIZE.int().clamp(256, 4096) as u32;
        let count = (cvars::SHADOW_CASCADES.int().max(1) as usize).min(MAX_CASCADES);
        let near = view.near.max(1e-3);
        let far = (cvars::SHADOW_DISTANCE.float() as f32).max(near * 2.0);

        // The light's basis is shared by every cascade: they differ in where
        // they sit and how wide they are, never in which way the map is turned.
        // A per-cascade `up` could flip one map against its neighbour, which is
        // a seam at the split no blend hides.
        let up = up_for(direction);
        let basis = render::camera::rh::view::look_to_mat4(render::Vec3::ZERO, direction, up);
        let mut cascades = [Cascade {
            view_projection: render::Mat4::IDENTITY,
            centre: render::Vec3::ZERO,
            radius: 0.0,
            texel_world: 0.0,
            split_far: 0.0,
            depth_world: 0.0,
        }; MAX_CASCADES];
        // One depth for every cascade (§6 M60), computed before the loop because
        // it is the *widest* cascade's and a near one must not derive its own.
        let reach = match cvars::SHADOW_REACH.bool() {
            true => depth_reach(view, extent),
            false => 0.0,
        };
        let mut slice_near = near;
        for (index, cascade) in cascades.iter_mut().enumerate().take(count) {
            let slice_far = split(near, far, index + 1, count);
            *cascade = fit(
                extracted, view, extent, direction, up, &basis, size, slice_near, slice_far, reach,
            );
            // Butted, not overlapping: the band the shader cross-fades over
            // lives *inside* each cascade's extent, so widening the slices to
            // overlap would pay for the same band twice.
            slice_near = slice_far;
        }
        Some(Sun {
            direction,
            size,
            right: basis.row(0).truncate(),
            above: basis.row(1).truncate(),
            cascades,
            count,
        })
    }
}

/// How far up-light a caster can sit and still land in a cascade — the depth the
/// maps record, over the widest one.
///
/// What `Extracted::cast_along` sweeps the view frustum by, and the reason it is
/// computed here: a shorter sweep drops a caster the map would have held, a
/// longer one keeps instances [`crate::casts_into`] then rejects, and only this
/// module knows the split scheme that decides which. The widest cascade is the
/// last, so its slab contains every other's — which since §6 M60 is exact rather
/// than conservative: [`depth_reach`] *is* this number, and every cascade records
/// it.
pub(crate) fn caster_reach(view: &View, extent: (u32, u32)) -> f32 {
    // `r.shadow_cull 0`'s half of the switch. Large and finite rather than
    // infinite: the sweep multiplies this by a plane normal, and `inf * 0` is the
    // `NaN` that would quietly cull *everything* instead of nothing.
    if cvars::SHADOW_CULL.int() == 0 {
        return 1.0e6;
    }
    depth_reach(view, extent)
}

/// The depth **every** cascade records, in metres — the widest one's slab, shared
/// (§6 M60).
///
/// It was each cascade's own `radius * 4` until M60, and that is a correctness
/// bug rather than a tightness one: absence of depth is absence of a blocker, so
/// a caster further up-light than a cascade's light eye is not merely coarse in
/// that map, it is *missing* from it and the shader reads the receiver as lit. A
/// near cascade is small by construction — that is the whole point of splitting —
/// so the reach it derived from its own width was the shortest exactly where the
/// picture is largest. Sponza's gallery went from shadowed to sunlit as
/// `r.shadow_distance` came *down*, which is the tuning move that improves every
/// other number.
///
/// Shared rather than fitted per cascade to the casters actually present: a reach
/// that follows content changes as content enters the view, and `depth_world`
/// scales the blocker search, so a fitted one would wobble a penumbra's width
/// with the contents of the frame. This is a function of the CVars and the
/// projection alone.
///
/// What it costs the near cascade is depth precision, and the margin is wide
/// enough not to be a knob: an orthographic reverse-Z map is linear, so
/// `D32_SFLOAT`'s worst quantization is `reach * 2⁻²⁴` and the normal offset it
/// has to stay under is `r.shadow_normal_bias` texels of the *near* cascade —
/// a ratio of `2041 * radius₀ / radius₃`, which is 83x at the shipping split and
/// degrades only with the spread between the first cascade and the last.
fn depth_reach(view: &View, extent: (u32, u32)) -> f32 {
    let near = view.near.max(1e-3);
    let far = (cvars::SHADOW_DISTANCE.float() as f32).max(near * 2.0);
    let count = (cvars::SHADOW_CASCADES.int().max(1) as usize).min(MAX_CASCADES);
    let size = cvars::SHADOW_SIZE.int().clamp(256, 4096) as u32;
    let (_, radius) = slice_sphere(view, extent, split(near, far, count - 1, count), far);
    // The light eye sits half of this up-light of the centre and the far plane
    // half beyond — over the same `radius + texel` the cull widens a cascade by.
    (radius + 2.0 * radius / size as f32) * 4.0
}

/// One cascade's up-light reach against the casters that need it (§6 M60).
///
/// Every field is metres or a count, and the two that matter fail in opposite
/// directions: [`dropped`] is shadow the frame will not have, and [`reach`] is
/// the depth range whose size costs blocker-search precision and cull work. A
/// measurement rather than a gate — `gg-tools shadow-reach` prints it.
///
/// [`dropped`]: CascadeReach::dropped
/// [`reach`]: CascadeReach::reach
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CascadeReach {
    /// Half-width of the cascade's square footprint, metres.
    pub radius: f32,
    /// How far up-light of the centre this map records — the light eye's offset.
    pub reach: f32,
    /// How far up-light the furthest caster over the footprint actually sits.
    /// A `needed` above `reach` is missing shadow, in metres of miss. Floored at
    /// zero, which is a real reading rather than a failed one: the widest cascade
    /// sits tens of metres ahead of the eye and every caster is *down*-light of
    /// its centre, so it needs no up-light reach at all.
    pub needed: f32,
    /// Casters whose footprint overlaps this cascade's square at all.
    pub over: u32,
    /// How many of those sit further up-light than `reach` — each one a blocker
    /// the map has no depth for, which the shader reads as no blocker.
    pub dropped: u32,
}

/// [`CascadeReach`] per cascade, nearest first, for the frame `extracted`
/// describes. Empty when nothing casts.
///
/// The caster set is both arrays whole rather than [`Extracted::visible`]: a
/// caster off screen is exactly the case a cascade's depth range decides, and the
/// prefix is the one that cannot show it.
#[must_use]
pub fn cascade_reach(extracted: &Extracted, view: &View, extent: (u32, u32)) -> Vec<CascadeReach> {
    let Some(sun) = Sun::of(extracted, view, extent) else {
        return Vec::new();
    };
    let (right, above) = (sun.right, sun.above);
    sun.cascades()
        .iter()
        .map(|cascade| {
            let reach = cascade.depth_world * 0.5;
            let mut report = CascadeReach {
                radius: cascade.radius,
                reach,
                ..CascadeReach::default()
            };
            let casters = extracted.instances.iter().chain(extracted.models.iter());
            for instance in casters {
                let to = instance.offset - cascade.centre;
                let signed = to.dot(sun.direction);
                let across = to - sun.direction * signed;
                // The square footprint, per axis — `cull::slab`'s test, and the
                // reason it is restated here rather than called is that `slab`
                // answers with the depth test folded in and this needs the two
                // apart.
                let out = |d: f32| (d.abs() - cascade.radius).max(0.0);
                let (x, y) = (out(across.dot(right)), out(across.dot(above)));
                if x * x + y * y > instance.radius * instance.radius {
                    continue;
                }
                report.over += 1;
                // Up-light is against the direction of travel, and a sphere
                // reaches `radius` further than its centre.
                let up = -signed + instance.radius;
                report.needed = report.needed.max(up);
                if up > reach {
                    report.dropped += 1;
                }
            }
            report
        })
        .collect()
}

/// The far distance of split `index` of `count` over `[near, far]`.
///
/// The practical split scheme [ZSXL06]: `r.shadow_split_lambda` blends a
/// logarithmic division, which gives every cascade the same *relative* texel
/// density and is what perspective actually wants, against a uniform one, which
/// is what keeps the first cascade from being centimetres deep.
///
/// [ZSXL06]: Zhang, Sun, Xu & Lu, "Parallel-Split Shadow Maps for Large-scale
/// Virtual Environments", 2006.
#[expect(
    clippy::disallowed_methods,
    reason = "render-side only and never hashed — a split distance decides which map a fragment \
              reads, and §3 keeps `gg_math::sim` out of this crate entirely"
)]
fn split(near: f32, far: f32, index: usize, count: usize) -> f32 {
    let ratio = index as f32 / count as f32;
    let logarithmic = near * (far / near).powf(ratio);
    let uniform = near + (far - near) * ratio;
    let lambda = (cvars::SHADOW_SPLIT_LAMBDA.float() as f32).clamp(0.0, 1.0);
    uniform + lambda * (logarithmic - uniform)
}

/// Tangent of the sun's angular **radius**, from `r.sun_angle`'s diameter.
///
/// The tangent because the shader multiplies it by a distance and wants no
/// trigonometry per fragment, and the radius because a penumbra grows by the
/// half-angle either side of the shadow's geometric edge.
#[expect(
    clippy::disallowed_methods,
    reason = "the sun's angular size is a tangent by definition; render-side only and never \
              hashed, and §3 keeps `gg_math::sim` out of this crate entirely"
)]
fn sun_tan() -> f32 {
    let degrees = (cvars::SUN_ANGLE.float() as f32).clamp(0.0, 20.0);
    (degrees.to_radians() * 0.5).tan()
}

/// One cascade fitted to the frustum slice from `slice_near` to `slice_far`.
///
/// The slab is sized by the slice's **bounding sphere**, not its corners, and
/// that is the whole reason a turning camera does not shimmer: a sphere's radius
/// is invariant under rotation, so a cascade's extent — and therefore its texel
/// size — depends only on the split distances and the projection. An AABB-fitted
/// cascade breathes as the camera turns, and a map whose texel size changes
/// every frame cannot be snapped to anything.
#[expect(
    clippy::too_many_arguments,
    reason = "every one is a distinct axis of the fit and bundling them into a struct would put \
              the argument list one indirection away rather than shortening it"
)]
fn fit(
    extracted: &Extracted,
    view: &View,
    extent: (u32, u32),
    direction: render::Vec3,
    up: render::Vec3,
    basis: &render::Mat4,
    size: u32,
    slice_near: f32,
    slice_far: f32,
    reach: f32,
) -> Cascade {
    let (centre_depth, radius) = slice_sphere(view, extent, slice_near, slice_far);
    // Down the camera's own forward axis, which is -Z before the view rotation.
    let centre = view.rotation() * render::Vec3::new(0.0, 0.0, -centre_depth);
    let texel_world = 2.0 * radius / size as f32;

    // The caller's, shared by every cascade — and the same two axes `Sun` hands
    // the caster cull, which is what keeps the snap and the cull in one frame.
    let (right, above) = (basis.row(0).truncate(), basis.row(1).truncate());
    // The shimmer fix (§6 M11's exit row), now covering rotation as well as
    // travel: the slab's centre moves whenever the camera moves *or turns*, so
    // its texel grid slides under the world unless the centre is quantized. The
    // phase is `gg-extract`'s because the quantity is absolute and `f64` — see
    // `Extracted::grid_phase`, and this module's header for why that split is
    // not an accident.
    let shift = |axis: render::Vec3| extracted.grid_phase(axis, centre, texel_world) * texel_world;
    let snapped = centre - right * shift(right) - above * shift(above);

    // The light's eye half the depth up-light and the far plane half beyond,
    // which is what puts a caster behind the slab — a wall the sun is on the far
    // side of — inside the map rather than outside it. `reach` is every
    // cascade's, from the widest (§6 M60); `r.shadow_reach 0` is the pre-M60
    // spelling, where a small cascade derived a short one from its own width and
    // lost every blocker above it.
    let depth = match reach > 0.0 {
        true => reach,
        false => radius * 4.0,
    };
    let eye = snapped - direction * depth * 0.5;
    let light = render::camera::rh::view::look_to_mat4(eye, direction, up);
    let projection = render::orthographic_reverse_z(radius, radius, 0.0, depth);
    Cascade {
        view_projection: projection * light,
        // The *unsnapped* centre bounds the geometry; the snap moves the grid by
        // under a texel, and a cull that followed it would drop a caster on the
        // seam. The radius grows by the same amount, for the same reason.
        centre,
        radius: radius + texel_world,
        texel_world,
        split_far: slice_far,
        // The projection's own far plane — the same `depth` the line above builds
        // it from, and the reason this is derived here rather than in the shader
        // is that the shader is handed the finished matrix and cannot see the
        // number that went into it. Also the cull's depth extent (`cull::slab`),
        // which is why the two cannot drift.
        depth_world: depth,
    }
}

/// `(distance along the view axis, radius)` of the sphere bounding the frustum
/// slice between `slice_near` and `slice_far`.
///
/// Closed form: for a symmetric perspective frustum the centre lies on the view
/// axis, and equating the distance to a near corner against the distance to a
/// far one solves it in one step. The branch is the degenerate case — a slice so
/// wide relative to its depth that the solved centre lands past the far plane,
/// where the far corners alone bound it.
#[expect(
    clippy::disallowed_methods,
    reason = "the frustum's half-extents are a tangent of the fov; render-side only, never \
              hashed, and §3 keeps `gg_math::sim` out of this crate"
)]
fn slice_sphere(view: &View, extent: (u32, u32), slice_near: f32, slice_far: f32) -> (f32, f32) {
    let aspect = extent.0.max(1) as f32 / extent.1.max(1) as f32;
    // An orthographic view's slice is a box, not a pyramid slice (§6 M20): the
    // half-extents are the view's own and depth adds nothing, so the sphere is
    // centred mid-slab and reaches a corner. What the sun *does* with a fit
    // this shape over a flat playfield is the M20 golden's question to answer.
    if view.ortho > 0.0 {
        let (half_h, half_v) = (view.ortho * aspect, view.ortho);
        let half_depth = (slice_far - slice_near) * 0.5;
        return (
            (slice_near + slice_far) * 0.5,
            (half_h * half_h + half_v * half_v + half_depth * half_depth).sqrt(),
        );
    }
    let half_v = (view.fov_y * 0.5).tan();
    let half_h = half_v * aspect;
    let spread = half_h * half_h + half_v * half_v;
    let (n, f) = (slice_near, slice_far);
    let centre = (spread + 1.0) * (n + f) * 0.5;
    if centre >= f {
        // Wider than it is deep: the far corners are the extremes and the
        // sphere sits on the far plane.
        (f, f * spread.sqrt())
    } else {
        let behind = centre - f;
        (centre, (spread * f * f + behind * behind).sqrt())
    }
}

/// A world axis to use as "up" for a light looking along `direction`.
///
/// The least aligned one, because `look_to` builds its basis from a cross
/// product and a near-parallel up makes that cross product near-zero — which is
/// a shadow map whose orientation flips as the sun crosses straight down.
fn up_for(direction: render::Vec3) -> render::Vec3 {
    let (x, y, z) = (direction.x.abs(), direction.y.abs(), direction.z.abs());
    if x <= y && x <= z {
        render::Vec3::X
    } else if y <= z {
        render::Vec3::Y
    } else {
        render::Vec3::Z
    }
}

/// The per-frame lighting buffer: one region per frame in flight, because it is
/// rewritten every frame and nothing waits (§4.3's `BufferKind::Dynamic`
/// contract).
pub(crate) struct Lighting {
    buffer: BufferHandle,
    address: DeviceAddress,
    /// §6 M34's split-sum table, and where the shader reads it.
    ///
    /// Its own buffer rather than a region of the one above: it is written once
    /// and read every frame, where everything else here is rebuilt every frame,
    /// and a constant living in a slot-per-frame ring would be memcpy'd 4368
    /// bytes at a time forever. Two kilobytes, so the address in the block costs
    /// more than the contents do.
    lut: BufferHandle,
    lut_address: DeviceAddress,
    /// Scratch, reused across frames.
    lights: Vec<GpuLight>,
    /// This frame's sun, once [`Lighting::plan`] has decided whether one casts.
    sun: Option<Sun>,
    /// This frame's casting lamps, decided by the same call (§6 M31).
    lamps: lamp::Lamps,
    /// The froxel lists [`Lighting::write`] built, kept for their allocation and
    /// for the drop count a `gg::draws` line reports.
    assignment: Assignment,
    /// The environments projected lately, and what they projected to (§6 M24).
    ///
    /// Cached on the *value* and not on a dirty flag, because the value is a
    /// small `Copy` record a game may rewrite every tick and comparing it is
    /// cheaper than the projection by three orders of magnitude. A game that
    /// really does animate its sky pays the quadrature once a frame, which is
    /// what it asked for.
    ///
    /// Keyed on [`SkyLook`] and not on the whole [`ExtractedSky`], which is the
    /// trap M28 had to avoid: the volume is *camera-relative*, so a key holding
    /// it would miss on every frame the camera moved and turn a cache into a
    /// 2048-sample quadrature per environment per frame. A linear scan because
    /// it holds at most [`MAX_SKY`] entries.
    projected: Vec<(SkyLook, [[f32; 4]; sky::SH_COEFFICIENTS])>,
    /// This frame's environments as the shader will read them, and how many are
    /// live — filled by [`Lighting::resolve`] and copied by
    /// [`Lighting::write`].
    ///
    /// Two calls rather than one argument: resolving needs the content layer and
    /// the projection cache, writing needs the RHI and the graph's shadow slots,
    /// and one function taking all four is over §3's arity limit for no gain.
    blocks: [GpuEnvironment; MAX_SKY],
    block_count: u32,
    /// The environments the last [`Lighting::resolve`] was handed — what
    /// `dump_graph` reports, on the same terms as [`Lighting::sun`]: whether
    /// the *next* frame has a sky is not knowable before it is extracted.
    environments: Vec<ExtractedSky>,
}

impl Lighting {
    /// Allocate the per-slot regions.
    ///
    /// # Errors
    /// Whatever the allocation refuses.
    pub(crate) fn new(rhi: &mut impl GpuHost) -> Result<Self, RhiError> {
        let buffer = rhi.create_buffer(&BufferDesc {
            name: "render.lighting",
            size: SLOT_BYTES * FRAMES_IN_FLIGHT,
            kind: gg_rhi::BufferKind::Dynamic,
        })?;
        let table = split_sum_table();
        let lut = rhi.create_buffer(&BufferDesc {
            name: "render.lighting.split-sum",
            size: core::mem::size_of_val(table) as u64,
            kind: gg_rhi::BufferKind::Storage,
        })?;
        // The staging ring and not `write_buffer`: this outlives every frame
        // that reads it, which is exactly the line §4.3 draws between the two.
        rhi.upload_buffer(lut, 0, bytemuck::cast_slice(table))?;
        rhi.flush_uploads()?;
        Ok(Lighting {
            address: rhi.buffer_address(buffer)?,
            buffer,
            lut_address: rhi.buffer_address(lut)?,
            lut,
            lights: Vec::new(),
            sun: None,
            lamps: lamp::Lamps::default(),
            assignment: Assignment::default(),
            projected: Vec::new(),
            blocks: [GpuEnvironment::zeroed(); MAX_SKY],
            block_count: 0,
            environments: Vec::new(),
        })
    }

    /// Decide this frame's sun before the graph is declared, so the caller knows
    /// whether to ask for a shadow map at all.
    ///
    /// Separate from [`Lighting::write`] because the shadow map's bindless index
    /// does not exist until the graph has acquired it, and that acquisition
    /// depends on this answer.
    pub(crate) fn plan(
        &mut self,
        extracted: &Extracted,
        view: &View,
        extent: (u32, u32),
    ) -> Option<Sun> {
        self.sun = Sun::of(extracted, view, extent);
        // Beside the sun rather than at the call site, and for the same reason
        // the sun is here: the atlas's extent is a function of how many lamps
        // cast, so the count must exist before the graph acquires anything —
        // and holding it here is what stops the frame that was *planned* from
        // differing from the one that gets written.
        self.lamps = lamp::Lamps::build(&extracted.lights);
        self.sun
    }

    /// The lamps the last [`Lighting::plan`] settled on (§6 M31).
    pub(crate) fn lamps(&self) -> &lamp::Lamps {
        &self.lamps
    }

    /// The sun the last [`Lighting::plan`] settled on — what a graph dump prints
    /// when there is no frame to plan.
    pub(crate) fn sun(&self) -> Option<Sun> {
        self.sun
    }

    /// Stage this frame's block into `slot`'s region.
    ///
    /// `shadow` is the bindless slot each cascade's map landed in, in the order
    /// [`Sun::cascades`] returned them. Short of that — the empty slice, when no
    /// pass wrote one — is how a frame says nothing casts, and it truncates the
    /// cascade count rather than being an error: the graph and this must agree
    /// on how many maps exist, and the graph is the one that knows.
    ///
    /// # Errors
    /// Whatever the buffer write refuses.
    pub(crate) fn write(
        &mut self,
        rhi: &mut impl GpuHost,
        slot: u64,
        view_projection: render::Mat4,
        grid: &Grid,
        lights: &[ExtractedLight],
        maps: Maps<'_>,
    ) -> Result<(), RhiError> {
        let Maps {
            cascades: shadow,
            lamps: atlas,
            field,
            ..
        } = maps;
        // Truncated first, so the froxel lists index the array the shader will
        // actually read. Extract's own caps make this a no-op in every frame it
        // produced; a caller assembling lights by hand is who it is for.
        let lights = &lights[..lights.len().min(MAX_LIGHTS)];
        self.lights.clear();
        self.lights.extend(lights.iter().map(|light| {
            let color = srgb_to_linear(light.color);
            GpuLight {
                position: light.offset.to_array(),
                range: light.range,
                direction: light.direction.to_array(),
                kind: light.kind,
                color: [color[0], color[1], color[2]],
                intensity: light.intensity,
            }
        }));
        self.assignment.build(grid, lights, cvars::CLUSTERS.bool());

        let sun = self.sun.filter(|_| !shadow.is_empty());
        let (forward, cluster_scale, cluster_bias, cluster_tile) = grid.shader_terms();
        let ambient = cvars::AMBIENT.float().max(0.0) as f32;
        let mut cascades = [GpuCascade::zeroed(); MAX_CASCADES];
        let mut cascade_count = 0;
        if let Some(sun) = sun {
            // Zipped, so the count is the shorter of what was fitted and what
            // the graph actually acquired a slot for. A cascade whose map has no
            // bindless index would be sampled as texture 0, which is whatever
            // else is in that slot.
            for ((gpu, fitted), texture) in cascades.iter_mut().zip(sun.cascades()).zip(shadow) {
                *gpu = GpuCascade {
                    view_projection: render::rows(fitted.view_projection),
                    texture: texture.get(),
                    texel_world: fitted.texel_world,
                    split_far: fitted.split_far,
                    depth_world: fitted.depth_world,
                };
                cascade_count += 1;
            }
        }
        // The same zip as the cascades', for the same reason: the atlas's
        // bindless slot is the graph's to hand over, and a lamp array longer
        // than the texture backing it would look its faces up in slot zero.
        let planned = &self.lamps;
        let mut gpu_lamps = [GpuLamp::zeroed(); lamp::MAX_LAMPS];
        let mut lamp_count = 0;
        if atlas.is_some() {
            for (gpu, fitted) in gpu_lamps.iter_mut().zip(planned.lamps()) {
                gpu.faces = core::array::from_fn(|face| render::rows(fitted.faces[face]));
                lamp_count += 1;
            }
        }
        let frame = GpuFrame {
            view_projection: render::rows(view_projection),
            ambient: [ambient, ambient, ambient, 0.0],
            light_count: self.lights.len() as u32,
            // Nearest, because the comparison is per tap in the shader: a linear
            // filter would average *depths* and then compare once, which softens
            // nothing and puts a wrong edge halfway between two casters.
            shadow_sampler: Sampler::NearestClamp.index(),
            cascade_count,
            shadow_texel: sun.map_or(0.0, |s| 1.0 / s.size as f32),
            sun_direction: sun.map_or([0.0; 3], |s| s.direction.to_array()),
            shadow_normal_bias: cvars::SHADOW_NORMAL_BIAS.float().max(0.0) as f32,
            shadow_depth_bias: cvars::SHADOW_DEPTH_BIAS.float().max(0.0) as f32,
            shadow_blend: (cvars::SHADOW_BLEND.float() as f32).clamp(0.0, 0.5),
            sun_tan: sun_tan(),
            shadow_softness: (cvars::SHADOW_SOFTNESS.float() as f32).clamp(0.0, 8.0),
            shadow_penumbra: (cvars::SHADOW_PENUMBRA.float() as f32).clamp(1.0, 64.0),
            shadow_taps: cvars::SHADOW_TAPS.int().clamp(4, 32) as u32,
            environment_count: self.block_count,
            // The array's *prefix*, not its total: the shader loops these by
            // count, so what it needs is how far the suns reach from index
            // zero. Extract puts them there (`Extracted::append_lights`).
            directional_count: lights
                .iter()
                .take_while(|l| l.kind == light::DIRECTIONAL)
                .count() as u32,
            view_forward: forward,
            cluster_scale,
            cluster_bias,
            cluster_tile,
            lamp_count,
            lamp_texture: atlas.map_or(0, gg_rhi::TextureIndex::get),
            lamp_uv_scale: planned.uv_scale(),
            lamp_inv_tile: 1.0 / planned.tile() as f32,
            lamp_normal_bias: cvars::LAMP_NORMAL_BIAS.float().max(0.0) as f32,
            lamp_depth_bias: cvars::LAMP_DEPTH_BIAS.float().max(0.0) as f32,
            lamp_taps: cvars::LAMP_TAPS.int().clamp(4, 32) as u32,
            shading: shading_flags_with(maps.occlusion, field.is_some()),
            split_sum: self.lut_address,
            ao_texture: maps.occlusion.map_or(0, gg_rhi::TextureIndex::get),
            gi_sh_texture: field.map_or(0, |f| f.sh.get()),
            gi_moment_texture: field.map_or(0, |f| f.moments.get()),
            gi_moment_edge: field.map_or(0.0, |f| f.moment_edge as f32),
            gi_spacing: field.map_or(0.0, |f| f.grid.spacing()),
            gi_counts: field.map_or([0; 3], |f| f.grid.counts()),
            gi_origin: field.map_or([0.0; 3], |f| f.origin.to_array()),
            gi_bias: field.map_or(0.0, |f| f.bias),
            probe_views: maps.probe_views,
            environments: self.blocks,
            cascades,
            lamps: gpu_lamps,
        };

        let slot = slot % FRAMES_IN_FLIGHT;
        let offset = slot * SLOT_BYTES;
        rhi.write_buffer(self.buffer, offset, bytemuck::bytes_of(&frame))?;
        if !self.lights.is_empty() {
            rhi.write_buffer(
                self.buffer,
                offset + core::mem::size_of::<GpuFrame>() as u64,
                bytemuck::cast_slice(&self.lights),
            )?;
            // The ranges always, the indices only up to what was used: the
            // array behind them is 128 KiB and a frame that filled it would be
            // one that had dropped pairs anyway.
            rhi.write_buffer(
                self.buffer,
                offset + CLUSTER_BASE,
                bytemuck::cast_slice(self.assignment.ranges()),
            )?;
            if !self.assignment.indices().is_empty() {
                rhi.write_buffer(
                    self.buffer,
                    offset + INDEX_BASE,
                    bytemuck::cast_slice(self.assignment.indices()),
                )?;
            }
        }
        Ok(())
    }

    /// What the last [`Lighting::write`]'s assignment came to — reported on the
    /// `gg::draws` line and read by `gg-tools lights`, on §6 M29's terms: a
    /// number nothing prints is a number nobody checks.
    pub(crate) fn cluster_load(&self) -> cluster::ClusterLoad {
        self.assignment.load()
    }

    /// Turn this frame's skies into the blocks the shader reads: each one's
    /// coefficients, its volume, and the compiled chain it named if the pack
    /// holds one and the chain is resident (§6 M28).
    ///
    /// Separate from [`Lighting::write`] because the two need different things —
    /// this the content layer and the projection cache, that the RHI and the
    /// graph's shadow slots — and because resolving must happen while the RHI
    /// borrow is free.
    pub(crate) fn resolve(
        &mut self,
        content: Option<&crate::content::Content>,
        skies: &[ExtractedSky],
    ) {
        let mut blocks = [GpuEnvironment::zeroed(); MAX_SKY];
        for (index, sky) in skies.iter().take(MAX_SKY).enumerate() {
            let radiance = crate::radiance(content, *sky);
            // The pack's projection wins outright where there is one. Not a
            // blend: the two describe the same sky and averaging them would be a
            // third sky neither half of the split sum was integrated against.
            let sh = match radiance {
                Some(radiance) => radiance.sh,
                None => self.project(sky.look),
            };
            blocks[index] = GpuEnvironment {
                sh,
                offset: sky.offset.to_array(),
                fade: sky.fade,
                half_extent: sky.half_extent.to_array(),
                radiance_scale: radiance.map_or(0.0, |r| r.scale),
                radiance_texture: radiance.map_or(0, |r| r.texture.get()),
                radiance_levels: radiance.map_or(0, |r| r.levels),
                reserved: [0; 2],
            };
        }
        self.blocks = blocks;
        self.block_count = skies.len().min(MAX_SKY) as u32;
        self.environments.clear();
        self.environments
            .extend(skies.iter().take(MAX_SKY).copied());
    }

    /// The environments the last frame was resolved with, innermost first.
    pub(crate) fn environments(&self) -> &[ExtractedSky] {
        &self.environments
    }

    /// This environment's coefficients, projected unless a recent frame already
    /// did it for the same sky.
    fn project(&mut self, look: SkyLook) -> [[f32; 4]; sky::SH_COEFFICIENTS] {
        if let Some((_, sh)) = self.projected.iter().find(|(cached, _)| *cached == look) {
            return *sh;
        }
        let sh = sky::project(&look);
        // Cleared rather than evicted when full: the cache exists for a sky that
        // holds still, and a world genuinely animating more than `MAX_SKY` of
        // them is paying the quadrature every frame whatever this does.
        if self.projected.len() >= MAX_SKY {
            self.projected.clear();
        }
        self.projected.push((look, sh));
        sh
    }

    /// Where `slot`'s block starts on the device.
    ///
    /// Known before anything is written, which is what lets the draws that carry
    /// it be built before the graph exists — the shadow map's bindless index is
    /// not known until the graph has acquired it, and acquiring it holds the RHI
    /// borrow that [`Lighting::write`] needs.
    pub(crate) fn slot_address(&self, slot: u64) -> DeviceAddress {
        self.address + (slot % FRAMES_IN_FLIGHT) * SLOT_BYTES
    }

    /// Release the buffer.
    ///
    /// # Errors
    /// A handle the RHI does not recognise, which cannot happen for one issued
    /// through here.
    pub(crate) fn destroy(self, rhi: &mut impl GpuHost) -> Result<(), RhiError> {
        rhi.destroy_buffer(self.lut)?;
        rhi.destroy_buffer(self.buffer)
    }
}

/// The split-sum table, built once per process (§6 M34).
///
/// A `OnceLock` because the table is a function of nothing — the same 2 KB for
/// every renderer in the process — and building it costs ~8 ms of GGX sampling
/// that a second `OffscreenRenderer` in the same test has no reason to pay
/// again.
fn split_sum_table() -> &'static [[f32; 2]] {
    static TABLE: std::sync::OnceLock<Vec<[f32; 2]>> = std::sync::OnceLock::new();
    TABLE.get_or_init(crate::split_sum::table)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    // Only the tests name a light *kind* now: `Sun::of` asks extract which light
    // casts rather than deciding for itself.
    use gg_extract::light;

    use super::*;

    fn sun(direction: render::Vec3) -> ExtractedLight {
        ExtractedLight {
            offset: render::Vec3::ZERO,
            direction,
            color: 0x00ff_ffff,
            intensity: 1.0,
            range: 0.0,
            kind: light::DIRECTIONAL,
        }
    }

    /// One frame's lights, seen from `eye` — the absolute camera position the
    /// snap is a phase of, and the only reason [`Sun::of`] takes a whole frame.
    fn seen_from(eye: gg_math::sim::DVec3, lights: &[ExtractedLight]) -> Extracted {
        let mut out = Extracted::default();
        out.clear(eye, gg_extract::Frustum::UNBOUNDED);
        out.lights.extend_from_slice(lights);
        out
    }

    fn at_origin(lights: &[ExtractedLight]) -> Extracted {
        seen_from(gg_math::sim::DVec3::ZERO, lights)
    }

    /// A fixed framing for the fitter. The cascades are fitted to a frustum, so
    /// every test needs one — and a *stated* one, since the defaults are CVars a
    /// session may have moved.
    const EXTENT: (u32, u32) = (1280, 720);

    fn looking(yaw: f32, pitch: f32) -> View {
        View {
            yaw,
            pitch,
            fov_y: 1.0,
            near: 0.05,
            ortho: 0.0,
            ortho_far: 500.0,
            aa: false,
            samples: gg_rhi::Samples::X1,
        }
    }

    fn cast(extracted: &Extracted) -> Sun {
        Sun::of(extracted, &looking(0.0, 0.0), EXTENT).expect("a caster")
    }

    /// The sweep and the cull read off one scheme (post-M21). `caster_reach` is
    /// what extract widens by and [`casts_into`](crate::casts_into) is what then
    /// rejects: short of the widest slab drops a caster the map would have held,
    /// past it keeps instances nothing can use. Equality is the only right answer,
    /// so this fails in both directions.
    #[test]
    fn the_caster_reach_is_the_widest_cascades_slab_and_nothing_over() {
        let cast = cast(&at_origin(&[sun(render::Vec3::new(-0.4, -1.0, -0.35))]));
        let widest = cast
            .cascades()
            .iter()
            .map(|c| c.radius)
            .fold(0.0f32, f32::max);
        assert!(widest > 0.0, "a fitted cascade has an extent");
        // `casts_into` takes a caster up to two radii up-light of the centre and
        // two down; four radii is the depth the map records.
        let slab = widest * 4.0;
        let reach = caster_reach(&looking(0.0, 0.0), EXTENT);
        assert!(
            (reach - slab).abs() <= slab * 1e-5,
            "{reach} against {slab}"
        );
    }

    fn lamp() -> ExtractedLight {
        ExtractedLight {
            offset: render::Vec3::new(0.0, 1.0, 0.0),
            direction: render::Vec3::ZERO,
            color: 0x00ff_8800,
            intensity: 10.0,
            range: 8.0,
            kind: light::POINT,
        }
    }

    #[test]
    fn only_a_directional_light_casts_and_a_zero_direction_is_not_one() {
        assert!(
            Sun::of(&at_origin(&[]), &looking(0.0, 0.0), EXTENT).is_none(),
            "no lights, no shadow pass"
        );
        assert!(
            Sun::of(&at_origin(&[lamp()]), &looking(0.0, 0.0), EXTENT).is_none(),
            "a point light casts nothing"
        );
        // A game that left `direction` at zero has described no direction; the
        // basis built from it would be degenerate, so it does not cast either.
        assert!(
            Sun::of(
                &at_origin(&[sun(render::Vec3::ZERO)]),
                &looking(0.0, 0.0),
                EXTENT
            )
            .is_none()
        );
        assert!(
            Sun::of(
                &at_origin(&[sun(render::Vec3::new(0.0, -1.0, 0.0))]),
                &looking(0.0, 0.0),
                EXTENT
            )
            .is_some()
        );
    }

    #[test]
    fn the_shadow_slab_is_reverse_z_and_holds_what_the_camera_can_see() {
        let cast = cast(&at_origin(&[sun(render::Vec3::new(0.0, -1.0, 0.0))]));
        let near = cast.cascades()[0];
        // Reverse-Z, the same convention every other depth buffer here uses
        // (§2, Math row): a point inside the slab lands strictly between the
        // planes, and something nearer the light lands *greater*.
        let centre = near.view_projection * near.centre.extend(1.0);
        assert_eq!(centre.w, 1.0, "orthographic: w is 1 everywhere");
        assert!(centre.z > 0.0 && centre.z < 1.0, "{centre:?}");
        let above = near.view_projection * (near.centre + render::Vec3::Y * 2.0).extend(1.0);
        assert!(above.z > centre.z, "{above:?} vs {centre:?}");

        // And it is centred on the *slice*, not on the eye — which is the whole
        // difference cascades make. The camera looks down -Z, so the near
        // cascade sits in front of it and the far ones further still.
        assert!(near.centre.z < 0.0, "{near:?}");
        for pair in cast.cascades().windows(2) {
            assert!(pair[1].centre.z < pair[0].centre.z, "{pair:?}");
        }
        // The eye itself is inside the near cascade: a shadow that started a
        // metre in front of the player would be the most visible bug there is.
        assert!(near.centre.length() <= near.radius, "{near:?}");
    }
    #[test]
    fn a_sun_straight_down_still_gets_a_usable_basis() {
        // The case a fixed world-up would break: `look_to` builds its basis from
        // a cross product, and an up parallel to the view direction makes it
        // zero — a shadow map that flips orientation as the sun crosses noon.
        for direction in [
            render::Vec3::new(0.0, -1.0, 0.0),
            render::Vec3::new(0.0, 1.0, 0.0),
            render::Vec3::new(1.0, 0.0, 0.0),
            render::Vec3::new(0.0, 0.0, -1.0),
        ] {
            let cast = Sun::of(&at_origin(&[sun(direction)]), &looking(0.0, 0.0), EXTENT).unwrap();
            let here = cast.cascades()[0].view_projection * render::Vec4::new(0.0, 0.0, 0.0, 1.0);
            assert!(here.z.is_finite() && here.x.is_finite(), "{direction:?}");
            assert!(here.z > 0.0 && here.z < 1.0, "{direction:?} -> {here:?}");
        }
    }

    #[test]
    fn the_gpu_records_are_what_the_shader_strides_by() {
        // `include/pbr.slang` hardcodes FRAME_STRIDE and LIGHT_STRIDE; this is
        // the other half of that agreement, the way the vertex assertions are.
        assert_eq!(core::mem::size_of::<GpuFrame>(), 4432);
        assert_eq!(core::mem::size_of::<GpuCascade>(), 80);
        assert_eq!(core::mem::size_of::<GpuLight>(), 48);
        assert_eq!(core::mem::offset_of!(GpuLight, direction), 16);
        assert_eq!(core::mem::offset_of!(GpuLight, color), 32);
        // And where the froxel arrays begin (§6 M30) — `pbr.slang` derives both
        // from the three constants above, so these are what the derivation is
        // checked against.
        assert_eq!(CLUSTER_BASE, 16912);
        assert_eq!(INDEX_BASE, 41488);
        // The lamps (§6 M31): a face is a bare 4x4, so the stride is six of them
        // and `LAMP_BASE` is where the shader starts counting.
        assert_eq!(core::mem::size_of::<GpuLamp>(), 384);
        assert_eq!(core::mem::offset_of!(GpuFrame, lamps), 1360);
        // ENV_STRIDE, and the offsets `load_environment` reads past the
        // coefficients (§6 M28).
        assert_eq!(core::mem::size_of::<GpuEnvironment>(), 192);
        assert_eq!(core::mem::offset_of!(GpuEnvironment, offset), 144);
        assert_eq!(core::mem::offset_of!(GpuEnvironment, radiance_texture), 176);
        // §6 M36's eleven words, and where they pushed the arrays to.
        assert_eq!(core::mem::offset_of!(GpuFrame, gi_sh_texture), 220);
        assert_eq!(core::mem::offset_of!(GpuFrame, probe_views), 264);
        assert_eq!(core::mem::offset_of!(GpuFrame, environments), 272);
        assert_eq!(core::mem::size_of::<probe::GpuProbeView>(), 400);
    }

    #[test]
    fn every_cascade_is_its_slab_over_the_map_and_they_coarsen_outward() {
        // The unit every acne knob is in, and now a *per-cascade* one — if this
        // drifts, an offset tuned in texels silently becomes a different
        // distance in each cascade rather than uniformly.
        let cast = cast(&at_origin(&[sun(render::Vec3::new(0.0, -1.0, 0.0))]));
        let cascades = cast.cascades();
        assert_eq!(cascades.len(), 4, "the shipping default");
        for c in cascades {
            // `radius` is the fit's own, grown by a texel for the cull; the map
            // is sized by the fit, so recover it the way `fit` computed it.
            let fitted = c.texel_world * cast.size as f32 / 2.0;
            assert!((c.radius - (fitted + c.texel_world)).abs() < 1e-4, "{c:?}");
        }
        // Strictly coarsening outward, which is the whole shape of a split
        // scheme: equal cascades would be a single slab drawn four times.
        for pair in cascades.windows(2) {
            assert!(pair[1].texel_world > pair[0].texel_world * 1.5, "{pair:?}");
            assert!(pair[1].split_far > pair[0].split_far, "{pair:?}");
        }
        // And the near cascade is worth the milestone: the single 40 m slab it
        // replaced was 39.06 mm a texel, which is what put a visible staircase
        // under a one-metre cube (§6 M15.3).
        assert!(cascades[0].texel_world < 0.006, "{:?}", cascades[0]);
        assert!(cascades[3].split_far >= 79.0, "the range is covered");
    }

    #[test]
    fn the_splits_cover_the_range_without_a_gap_and_lambda_moves_them() {
        // Butted, not overlapping — the shader's cross-fade lives inside a
        // cascade's own extent, so a gap here is an unshadowed band and an
        // overlap is paying for the same metres twice.
        let near = 0.05;
        let far = 80.0;
        let logarithmic: Vec<f32> = (1..=4).map(|i| split(near, far, i, 4)).collect();
        assert!(
            logarithmic.windows(2).all(|p| p[1] > p[0]),
            "{logarithmic:?}"
        );
        assert!((logarithmic[3] - far).abs() < 1e-3, "{logarithmic:?}");

        // Lambda 0 is the uniform division, which is the control: it is the one
        // value whose splits are arithmetic, so it can be checked by hand.
        cvars::SHADOW_SPLIT_LAMBDA.set_float(0.0);
        for i in 1..=4 {
            let want = near + (far - near) * i as f32 / 4.0;
            assert!((split(near, far, i, 4) - want).abs() < 1e-3, "split {i}");
        }
        // A uniform first cascade is *much* deeper than the logarithmic one —
        // which is exactly why the default is not uniform.
        assert!(split(near, far, 1, 4) > logarithmic[0] * 4.0);
        cvars::SHADOW_SPLIT_LAMBDA.set_float(0.85);
    }

    #[test]
    fn a_turning_camera_does_not_change_a_cascades_size() {
        // Why the fit is a bounding *sphere*: an AABB-fitted cascade breathes as
        // the camera turns, and a map whose texel size changes every frame
        // cannot be snapped to anything — the shimmer would come back through
        // the one door the snap does not cover.
        let lit = at_origin(&[sun(render::Vec3::new(-0.4, -1.0, -0.35))]);
        let base = Sun::of(&lit, &looking(0.0, 0.0), EXTENT).unwrap();
        for (yaw, pitch) in [(0.7, 0.0), (2.5, -0.4), (-1.1, 0.9), (3.0, 1.2)] {
            let turned = Sun::of(&lit, &looking(yaw, pitch), EXTENT).unwrap();
            for (a, b) in base.cascades().iter().zip(turned.cascades()) {
                assert!(
                    (a.texel_world - b.texel_world).abs() < 1e-6,
                    "yaw {yaw} pitch {pitch}: {a:?} vs {b:?}"
                );
            }
        }
    }

    #[test]
    fn the_shadow_grid_stands_still_in_the_world_while_the_camera_walks() {
        use gg_math::sim;
        // The crawl, as a measurement. One fixed absolute point, seen from a
        // camera stepping by a deliberately non-texel amount: its shadow texel
        // must move by *whole* texels only, because a fractional move is the
        // grid sliding under the geometry and that is what shimmers.
        let target = sim::DVec3::new(3.25, -1.75, 2.5);
        let step = 0.017_3; // ~0.44 of a texel at the shipping defaults
        let light = sun(render::Vec3::new(-0.4, -1.0, -0.35));
        let mut first: Option<render::Vec2> = None;
        let mut travelled: f32 = 0.0;
        for i in 0..16 {
            let eye = sim::DVec3::new(f64::from(i) * step, 0.0, f64::from(i) * -step * 0.63);
            let cast = Sun::of(&seen_from(eye, &[light]), &looking(0.0, 0.0), EXTENT).unwrap();
            // The outermost cascade: it is the one the target at 4 m is inside
            // for the whole walk, so the sequence is comparable across steps.
            let last = *cast.cascades().last().unwrap();
            let clip = last.view_projection
                * render::Vec4::new(
                    (target.x - eye.x) as f32,
                    (target.y - eye.y) as f32,
                    (target.z - eye.z) as f32,
                    1.0,
                );
            let texel = render::Vec2::new(clip.x, clip.y) * 0.5 * cast.size as f32;

            let moved = texel - *first.get_or_insert(texel);
            // What is left of the slide: f32 rounding in the narrowing above,
            // four orders below a texel. Without the snap this reaches 0.5.
            let residue = moved - moved.round();
            assert!(residue.abs().max_element() < 0.01, "step {i}: {residue:?}");
            travelled = travelled.max(moved.abs().max_element());
        }
        // And the control: the grid did move, by several whole texels, so the
        // assertion above is not passing on a camera that never went anywhere.
        assert!(travelled > 2.0, "the eye barely moved: {travelled} texels");
    }

    #[test]
    fn the_recorded_sun_direction_is_unit_and_is_what_the_projection_looks_along() {
        for direction in [
            render::Vec3::new(0.0, -1.0, 0.0),
            render::Vec3::new(-0.4, -1.0, -0.35),
            render::Vec3::new(3.0, -1.0, 0.0),
        ] {
            let cast = Sun::of(&at_origin(&[sun(direction)]), &looking(0.0, 0.0), EXTENT).unwrap();
            // The shader takes sin(incidence) from `dot(normal, -sun_direction)`,
            // which is only an angle if this is unit.
            assert!((cast.direction.length() - 1.0).abs() < 1e-6, "{cast:?}");
            // And it must be the same direction the slab was fitted along, or the
            // offset leans one way while the map looks another.
            let along = direction.normalize();
            assert!(
                cast.direction.distance(along) < 1e-6,
                "{cast:?} vs {along:?}"
            );
        }
    }
}
