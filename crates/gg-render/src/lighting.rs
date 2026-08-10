//! One frame's lighting environment, and the sun's shadow projection (§6 M11).
//!
//! Every shading pipeline reads the same block through one device address rather
//! than through push constants: the sun's matrix alone is 64 bytes and Vulkan
//! guarantees 128 of push, so a per-draw copy would leave no room for a draw's
//! own parameters. It is also the shape a clustered light list wants when
//! [`MAX_POINT`](gg_extract::MAX_POINT) stops being a small number (P1).
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
use gg_extract::{Extracted, ExtractedLight, MAX_DIRECTIONAL, MAX_POINT, light};
use gg_math::render;
use gg_rhi::{BufferDesc, BufferHandle, DeviceAddress, FRAMES_IN_FLIGHT, RhiError, Sampler};

use crate::{GpuHost, View, cvars, srgb_to_linear};

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
    /// Zero. Named rather than implicit: `Pod` refuses padding, and the shader's
    /// `FRAME_STRIDE` has to be a number both sides can write down. Padding to a
    /// multiple of 16 besides, so the cascade array behind it starts on the
    /// alignment a `float4` read would want.
    reserved: [u32; 2],
    /// Nearest first, `cascade_count` of them live. A fixed array rather than a
    /// second buffer: it is 320 bytes, and one address the shader already has
    /// beats a second one it would have to be handed.
    cascades: [GpuCascade; MAX_CASCADES],
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
    reserved: u32,
}

const _: () = {
    assert!(core::mem::size_of::<GpuCascade>() == 80);
    assert!(core::mem::size_of::<GpuFrame>() == 448);
    assert!(core::mem::offset_of!(GpuFrame, ambient) == 64);
    assert!(core::mem::offset_of!(GpuFrame, light_count) == 80);
    assert!(core::mem::offset_of!(GpuFrame, sun_direction) == 96);
    assert!(core::mem::offset_of!(GpuFrame, cascades) == 128);
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

const SLOT_BYTES: u64 =
    (core::mem::size_of::<GpuFrame>() + MAX_LIGHTS * core::mem::size_of::<GpuLight>()) as u64;

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
}

/// What the sun asks of the graph this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sun {
    /// The direction the light travels, unit.
    pub(crate) direction: render::Vec3,
    /// Shadow map edge, in texels — one size for every cascade, which is what
    /// makes the kernel's three-texel width mean the same thing in each.
    pub(crate) size: u32,
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
    fn of(extracted: &Extracted, view: &View, extent: (u32, u32)) -> Option<Self> {
        let sun = extracted
            .lights
            .iter()
            .find(|l| l.kind == light::DIRECTIONAL && l.direction != render::Vec3::ZERO)?;
        let direction = sun.direction.normalize_or_zero();
        let size = cvars::SHADOW_SIZE.int().clamp(256, 4096) as u32;
        let count = (cvars::SHADOW_CASCADES.int().max(1) as usize).min(MAX_CASCADES);
        let near = view.near.max(1e-3);
        let far = (cvars::SHADOW_DISTANCE.float() as f32).max(near * 2.0);

        // The light's basis is shared by every cascade: they differ in where
        // they sit and how wide they are, never in which way the map is turned.
        // A per-cascade `up` could flip one map against its neighbour, which is
        // a seam at the split no blend hides.
        let up = up_for(direction);
        let mut cascades = [Cascade {
            view_projection: render::Mat4::IDENTITY,
            centre: render::Vec3::ZERO,
            radius: 0.0,
            texel_world: 0.0,
            split_far: 0.0,
        }; MAX_CASCADES];
        let mut slice_near = near;
        for (index, cascade) in cascades.iter_mut().enumerate().take(count) {
            let slice_far = split(near, far, index + 1, count);
            *cascade = fit(
                extracted, view, extent, direction, up, size, slice_near, slice_far,
            );
            // Butted, not overlapping: the band the shader cross-fades over
            // lives *inside* each cascade's extent, so widening the slices to
            // overlap would pay for the same band twice.
            slice_near = slice_far;
        }
        Some(Sun {
            direction,
            size,
            cascades,
            count,
        })
    }
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
    size: u32,
    slice_near: f32,
    slice_far: f32,
) -> Cascade {
    let (centre_depth, radius) = slice_sphere(view, extent, slice_near, slice_far);
    // Down the camera's own forward axis, which is -Z before the view rotation.
    let centre = view.rotation() * render::Vec3::new(0.0, 0.0, -centre_depth);
    let texel_world = 2.0 * radius / size as f32;

    let basis = render::camera::rh::view::look_to_mat4(render::Vec3::ZERO, direction, up);
    let (right, above) = (basis.row(0).truncate(), basis.row(1).truncate());
    // The shimmer fix (§6 M11's exit row), now covering rotation as well as
    // travel: the slab's centre moves whenever the camera moves *or turns*, so
    // its texel grid slides under the world unless the centre is quantized. The
    // phase is `gg-extract`'s because the quantity is absolute and `f64` — see
    // `Extracted::grid_phase`, and this module's header for why that split is
    // not an accident.
    let shift = |axis: render::Vec3| extracted.grid_phase(axis, centre, texel_world) * texel_world;
    let snapped = centre - right * shift(right) - above * shift(above);

    // The light's eye two radii up-light and the far plane four, which is what
    // puts a caster behind the slab — a wall the sun is on the far side of —
    // inside the map rather than outside it.
    let eye = snapped - direction * radius * 2.0;
    let light = render::camera::rh::view::look_to_mat4(eye, direction, up);
    let projection = render::orthographic_reverse_z(radius, radius, 0.0, radius * 4.0);
    Cascade {
        view_projection: projection * light,
        // The *unsnapped* centre bounds the geometry; the snap moves the grid by
        // under a texel, and a cull that followed it would drop a caster on the
        // seam. The radius grows by the same amount, for the same reason.
        centre,
        radius: radius + texel_world,
        texel_world,
        split_far: slice_far,
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
    /// Scratch, reused across frames.
    lights: Vec<GpuLight>,
    /// This frame's sun, once [`Lighting::plan`] has decided whether one casts.
    sun: Option<Sun>,
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
        Ok(Lighting {
            address: rhi.buffer_address(buffer)?,
            buffer,
            lights: Vec::new(),
            sun: None,
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
        self.sun
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
        lights: &[ExtractedLight],
        shadow: &[gg_rhi::TextureIndex],
    ) -> Result<(), RhiError> {
        self.lights.clear();
        self.lights
            .extend(lights.iter().take(MAX_LIGHTS).map(|light| {
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

        let sun = self.sun.filter(|_| !shadow.is_empty());
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
                    reserved: 0,
                };
                cascade_count += 1;
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
            reserved: [0; 2],
            cascades,
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
        }
        Ok(())
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
        rhi.destroy_buffer(self.buffer)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

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
        }
    }

    fn cast(extracted: &Extracted) -> Sun {
        Sun::of(extracted, &looking(0.0, 0.0), EXTENT).expect("a caster")
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
        assert_eq!(core::mem::size_of::<GpuFrame>(), 448);
        assert_eq!(core::mem::size_of::<GpuCascade>(), 80);
        assert_eq!(core::mem::size_of::<GpuLight>(), 48);
        assert_eq!(core::mem::offset_of!(GpuLight, direction), 16);
        assert_eq!(core::mem::offset_of!(GpuLight, color), 32);
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
