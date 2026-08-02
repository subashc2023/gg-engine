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
//! **One cascade, and the first directional light casts it.** A second sun
//! casting a second map is a second pass and a cascade decision; cascades are P1
//! (§6 M11). The rest of the directional lights still light — they just do not
//! occlude, which is visible and explicable rather than silently wrong.
//!
//! **The map is not texel-snapped**, so its edges shimmer slightly while the
//! camera walks. The fix quantizes the light-space origin to whole shadow texels
//! *in world space*, and world space here is the camera's absolute `f64`
//! position — which lives on the far side of the §1.4 membrane and is
//! deliberately unreachable from this crate. Doing it properly means the
//! quantization happens in `gg-extract`; doing it improperly means giving
//! `gg-render` a sim type. Named as a limitation instead.

use gg_extract::{ExtractedLight, MAX_DIRECTIONAL, MAX_POINT, light};
use gg_math::render;
use gg_rhi::{BufferDesc, BufferHandle, DeviceAddress, FRAMES_IN_FLIGHT, RhiError, Sampler};

use crate::{GpuHost, cvars, srgb_to_linear};

/// The per-frame block, mirroring `include/pbr.slang`'s `Frame`. The shader
/// reads it as scalars through a device address, so this layout is the whole of
/// the agreement — and `FRAME_STRIDE` there is the other half of it.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuFrame {
    /// Camera-relative world → clip. Read by the box pass, whose push block has
    /// no room for it; the pack pass carries a finished product per instance.
    view_projection: [[f32; 4]; 4],
    /// Camera-relative world → the sun's clip space.
    sun_view_projection: [[f32; 4]; 4],
    /// Linear rgb; `w` unused.
    ambient: [f32; 4],
    light_count: u32,
    shadow_texture: u32,
    shadow_sampler: u32,
    shadow_enabled: u32,
    shadow_texel: f32,
    shadow_normal_bias: f32,
    /// Zero. Named rather than implicit: `Pod` refuses padding, and the shader's
    /// `FRAME_STRIDE` has to be a number both sides can write down.
    reserved: [u32; 2],
}

const _: () = {
    assert!(core::mem::size_of::<GpuFrame>() == 176);
    assert!(core::mem::offset_of!(GpuFrame, sun_view_projection) == 64);
    assert!(core::mem::offset_of!(GpuFrame, ambient) == 128);
    assert!(core::mem::offset_of!(GpuFrame, light_count) == 144);
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

/// What the sun asks of the graph this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Sun {
    /// Camera-relative world → the sun's clip space.
    pub(crate) view_projection: render::Mat4,
    /// Shadow map edge, in texels.
    pub(crate) size: u32,
}

impl Sun {
    /// The casting light's projection, or `None` when nothing casts.
    ///
    /// The first directional light in extract's order — which is world iteration
    /// order, and therefore a function of world state (§4.2).
    fn of(lights: &[ExtractedLight]) -> Option<Self> {
        let sun = lights
            .iter()
            .find(|l| l.kind == light::DIRECTIONAL && l.direction != render::Vec3::ZERO)?;
        let radius = cvars::SHADOW_RADIUS.float().max(1.0) as f32;
        let size = cvars::SHADOW_SIZE.int().clamp(256, 4096) as u32;
        let direction = sun.direction.normalize_or_zero();

        // The camera is the origin (§1.4), so the slab is centred on it. The
        // light's eye sits two radii up-light and the far plane four, which is
        // what puts a caster behind the viewer — a wall the sun is on the far
        // side of — inside the map rather than outside it.
        let eye = -direction * radius * 2.0;
        let view = render::camera::rh::view::look_to_mat4(eye, direction, up_for(direction));
        let projection = render::orthographic_reverse_z(radius, radius, 0.0, radius * 4.0);
        Some(Sun {
            view_projection: projection * view,
            size,
        })
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
    pub(crate) fn plan(&mut self, lights: &[ExtractedLight]) -> Option<Sun> {
        self.sun = Sun::of(lights);
        self.sun
    }

    /// The sun the last [`Lighting::plan`] settled on — what a graph dump prints
    /// when there is no frame to plan.
    pub(crate) fn sun(&self) -> Option<Sun> {
        self.sun
    }

    /// Stage this frame's block into `slot`'s region.
    ///
    /// `shadow` is the bindless slot the shadow map landed in, or `None` when no
    /// pass wrote one — in which case nothing samples it and the matrix below is
    /// whatever [`Lighting::plan`] produced, read by no one.
    ///
    /// # Errors
    /// Whatever the buffer write refuses.
    pub(crate) fn write(
        &mut self,
        rhi: &mut impl GpuHost,
        slot: u64,
        view_projection: render::Mat4,
        lights: &[ExtractedLight],
        shadow: Option<gg_rhi::TextureIndex>,
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

        let sun = self.sun.filter(|_| shadow.is_some());
        let ambient = cvars::AMBIENT.float().max(0.0) as f32;
        let frame = GpuFrame {
            view_projection: render::rows(view_projection),
            sun_view_projection: render::rows(
                sun.map_or(render::Mat4::IDENTITY, |s| s.view_projection),
            ),
            ambient: [ambient, ambient, ambient, 0.0],
            light_count: self.lights.len() as u32,
            shadow_texture: shadow.map_or(0, gg_rhi::TextureIndex::get),
            // Nearest, because the comparison is per tap in the shader: a linear
            // filter would average *depths* and then compare once, which softens
            // nothing and puts a wrong edge halfway between two casters.
            shadow_sampler: Sampler::NearestClamp.index(),
            shadow_enabled: u32::from(sun.is_some()),
            shadow_texel: sun.map_or(0.0, |s| 1.0 / s.size as f32),
            shadow_normal_bias: cvars::SHADOW_NORMAL_BIAS.float() as f32,
            reserved: [0; 2],
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

    /// The depth bias the shadow pass draws with, straight off the CVars (§6
    /// M11's exit row). Read per frame rather than cached: turning the knob at
    /// the console while looking at the wall is the point of it being one.
    pub(crate) fn shadow_bias() -> gg_rhi::DepthBias {
        gg_rhi::DepthBias {
            constant: cvars::SHADOW_BIAS.float() as f32,
            slope: cvars::SHADOW_SLOPE_BIAS.float() as f32,
        }
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
        assert!(Sun::of(&[]).is_none(), "no lights, no shadow pass");
        assert!(Sun::of(&[lamp()]).is_none(), "a point light casts nothing");
        // A game that left `direction` at zero has described no direction; the
        // basis built from it would be degenerate, so it does not cast either.
        assert!(Sun::of(&[sun(render::Vec3::ZERO)]).is_none());
        assert!(Sun::of(&[sun(render::Vec3::new(0.0, -1.0, 0.0))]).is_some());
    }

    #[test]
    fn the_shadow_slab_is_reverse_z_and_centred_on_the_camera() {
        let cast = Sun::of(&[sun(render::Vec3::new(0.0, -1.0, 0.0))]).unwrap();
        // The camera origin is inside the slab, and reverse-Z puts it at a depth
        // strictly between the planes — the same convention every other depth
        // buffer in the engine uses (§2, Math row).
        let here = cast.view_projection * render::Vec4::new(0.0, 0.0, 0.0, 1.0);
        assert_eq!(here.w, 1.0, "orthographic: w is 1 everywhere");
        assert!(here.z > 0.0 && here.z < 1.0, "{here:?}");
        assert!(here.x.abs() < 1e-5 && here.y.abs() < 1e-5, "{here:?}");

        // And something between the sun and the camera is *nearer* the light,
        // which under reverse-Z means a greater depth.
        let above = cast.view_projection * render::Vec4::new(0.0, 5.0, 0.0, 1.0);
        assert!(above.z > here.z, "{above:?} vs {here:?}");
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
            let cast = Sun::of(&[sun(direction)]).unwrap();
            let here = cast.view_projection * render::Vec4::new(0.0, 0.0, 0.0, 1.0);
            assert!(here.z.is_finite() && here.x.is_finite(), "{direction:?}");
            assert!(here.z > 0.0 && here.z < 1.0, "{direction:?} -> {here:?}");
        }
    }

    #[test]
    fn the_gpu_records_are_what_the_shader_strides_by() {
        // `include/pbr.slang` hardcodes FRAME_STRIDE and LIGHT_STRIDE; this is
        // the other half of that agreement, the way the vertex assertions are.
        assert_eq!(core::mem::size_of::<GpuFrame>(), 176);
        assert_eq!(core::mem::size_of::<GpuLight>(), 48);
        assert_eq!(core::mem::offset_of!(GpuLight, direction), 16);
        assert_eq!(core::mem::offset_of!(GpuLight, color), 32);
    }
}
