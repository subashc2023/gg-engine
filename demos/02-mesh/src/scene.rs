//! Demo 02's **scene**: the mesh, the BC7 texture, the render-side camera pose,
//! and everything needed to get them onto the GPU. gg-golden renders exactly
//! this — same buffers, same upload path, same push constants — so the golden
//! test guards the demo rather than a lookalike.
//!
//! Everything here is downstream of the §1.4 membrane: it reads sim state that
//! [`crate::sim`] owns and turns it into `f32` the GPU can have. Nothing here
//! is hashed, and nothing here may write back.

use crate::bc7;
use crate::shaders_gen::mesh as shader;
use crate::sim::{CameraRig, Cube, Sim, SimError};
use gg_extract::{Extracted, Instance};
use gg_math::{render, sim};
use gg_rhi::{
    BufferDesc, BufferHandle, BufferKind, DeviceAddress, ImageDesc, ImageFormat, ImageHandle,
    ImageUse, PipelineDesc, PipelineHandle, RhiError, Sampler, TextureIndex,
};

/// Background clear color (linear values; the sRGB target encodes).
pub const CLEAR: [f32; 4] = [0.02, 0.025, 0.04, 1.0];

/// The extent gg-golden captures at (§4.10 v0: fixed-size offscreen target).
pub const GOLDEN_EXTENT: (u32, u32) = (640, 360);

/// Vertical field of view, radians.
pub const FOV_Y: f32 = 1.0;

/// Near plane. With reverse-Z and an infinite far plane this is the *only*
/// depth-precision knob there is (§2, Math row).
pub const NEAR: f32 = 0.05;

/// The texture's edge length in texels. A multiple of 4 so it is a whole
/// number of BC7 blocks.
pub const TEXTURE_EXTENT: (u32, u32) = (64, 64);

/// One vertex, mirroring `mesh.slang`'s `VERTEX_STRIDE` byte layout. The
/// shader reads these fields at hardcoded offsets through a device address, so
/// the layout is frozen below rather than trusted.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// Object-space position.
    pub position: [f32; 3],
    /// Object-space normal.
    pub normal: [f32; 3],
    /// Texture coordinates.
    pub uv: [f32; 2],
}

const _: () = {
    // `mesh.slang` hardcodes VERTEX_STRIDE = 32 and the three field offsets;
    // HLSL has no `sizeof` to derive them from, so this is the other half of
    // that agreement. A drift here is a build error, not a garbled mesh.
    assert!(core::mem::size_of::<Vertex>() == 32);
    assert!(core::mem::offset_of!(Vertex, position) == 0);
    assert!(core::mem::offset_of!(Vertex, normal) == 12);
    assert!(core::mem::offset_of!(Vertex, uv) == 24);
};

/// The render-side camera pose: a *copy* of the sim's rig, on this side of the
/// membrane. Position stays `f64` right up to the narrowing, which
/// [`gg_extract`] performs — [`Camera::view_projection`] hands out a matrix
/// whose eye is the origin of camera-relative space, and instance offsets
/// arrive already narrowed to meet it (§1.4, §4.2.1).
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// World-space eye position.
    pub position: sim::DVec3,
    /// Rotation about +Y, radians.
    pub yaw: f32,
    /// Rotation about the camera's right axis, radians.
    pub pitch: f32,
}

impl Camera {
    /// The demo's opening shot, and the one gg-golden renders. Derived from the
    /// sim's pose rather than repeating its numbers: one camera, two sides.
    pub const GOLDEN: Camera = Camera::from_rig(CameraRig::GOLDEN);

    /// Read a sim rig into a render pose. The only direction that exists.
    pub const fn from_rig(rig: CameraRig) -> Camera {
        Camera {
            position: rig.position,
            yaw: rig.yaw,
            pitch: rig.pitch,
        }
    }

    /// Where the camera is looking, in render space.
    pub fn forward(&self) -> render::Vec3 {
        self.rotation() * render::Vec3::NEG_Z
    }

    /// The camera's right axis.
    pub fn right(&self) -> render::Vec3 {
        self.rotation() * render::Vec3::X
    }

    fn rotation(&self) -> render::Quat {
        // YXZ: yaw about world +Y first, then pitch about the rotated right
        // axis — the fly-camera order, and the one that never rolls.
        render::Quat::from_euler(render::EulerRot::YXZ, self.yaw, self.pitch, 0.0)
    }

    /// World → clip, for geometry already narrowed to camera-relative space.
    /// The view matrix carries rotation only: the translation happened in
    /// `f64`, before the narrowing, which is what makes a camera 10^12 m from
    /// the origin render without jitter (§4.2.1).
    pub fn view_projection(&self, aspect: f32) -> render::Mat4 {
        let up = self.rotation() * render::Vec3::Y;
        let view = render::camera::rh::view::look_to_mat4(render::Vec3::ZERO, self.forward(), up);
        render::perspective_reverse_z(FOV_Y, aspect, NEAR) * view
    }
}

/// The scene's pipeline, straight from the embedded offline build.
pub fn pipeline_desc() -> PipelineDesc<'static> {
    PipelineDesc {
        name: "mesh",
        vs_spirv: shader::VS_MAIN_SPIRV,
        vs_entry: shader::VS_MAIN_ENTRY,
        fs_spirv: shader::FS_MAIN_SPIRV,
        fs_entry: shader::FS_MAIN_ENTRY,
        push_constant_size: core::mem::size_of::<shader::MeshPush>() as u32,
        color: gg_rhi::ColorTarget::Backbuffer,
        // A closed mesh drawn without back-face culling: every face is
        // rasterized and depth decides, which is exactly what makes this a
        // test of reverse-Z rather than of winding order.
        blend: gg_rhi::Blend::Off,
        depth: gg_rhi::DepthMode::Write,
    }
}

/// The scene's graph: one forward pass into wherever the frame lands, over a
/// depth attachment the graph pools (§4.5). Shared by the app and by gg-golden
/// — which appends a readback pass to this same list, so the image it judges is
/// the frame the demo renders rather than a lookalike (§4.10).
pub fn declare<'a>(
    backbuffer: gg_render::graph::ResourceId,
    depth: gg_render::graph::ResourceId,
    draws: &'a [gg_rhi::DrawSpec<'a>],
) -> [gg_render::graph::Declared<'a>; 1] {
    [gg_render::graph::Declared {
        name: "forward-opaque",
        body: gg_render::graph::Body::Draw {
            color: Some((backbuffer, gg_render::graph::Load::Clear(CLEAR))),
            depth: Some((depth, gg_render::graph::DepthUse::Write)),
            samples: &[],
            draws,
        },
    }]
}

/// The matrix the vertex shader receives: object → clip for one extracted
/// instance, seen from `camera` through a viewport of `extent`.
///
/// Both halves of the model matrix now come from the sim: the offset is the
/// instance's camera-relative position and the rotation is its spin, neither of
/// which anything on this side of the membrane can change (§6 M4B).
pub fn view_projection_for(
    camera: &Camera,
    extent: (u32, u32),
    instance: &Instance,
) -> render::Mat4 {
    let aspect = extent.0.max(1) as f32 / extent.1.max(1) as f32;
    let model = render::Mat4::from_rotation_translation(instance.rotation, instance.offset);
    camera.view_projection(aspect) * model
}

/// Push constants for one instance.
pub fn push_for(
    camera: &Camera,
    extent: (u32, u32),
    instance: &Instance,
    scene: &SceneResources,
) -> shader::MeshPush {
    shader::MeshPush::new(
        render::rows(view_projection_for(camera, extent, instance)),
        scene.vertex_address,
        scene.texture_index.get(),
        Sampler::LinearRepeat.index(),
    )
}

/// A unit cube: 24 vertices so every face carries its own normal and its own
/// copy of the texture, 36 indices.
pub fn cube() -> (Vec<Vertex>, Vec<u32>) {
    // (normal, u axis, v axis) per face, u x v pointing outward.
    const FACES: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ];
    const CORNERS: [(f32, f32); 4] = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
    const HALF: f32 = 0.5;

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (normal, u_axis, v_axis) in FACES {
        let base = vertices.len() as u32;
        for (u, v) in CORNERS {
            let position = [
                (normal[0] + u * u_axis[0] + v * v_axis[0]) * HALF,
                (normal[1] + u * u_axis[1] + v * v_axis[1]) * HALF,
                (normal[2] + u * u_axis[2] + v * v_axis[2]) * HALF,
            ];
            vertices.push(Vertex {
                position,
                normal,
                uv: [(u + 1.0) * 0.5, (1.0 - v) * 0.5],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    (vertices, indices)
}

/// The demo's texture as BC7 blocks: a two-tone checker inside a dark border,
/// which makes a UV or sampling regression obvious in a golden diff instead of
/// merely dimmer.
pub fn texture_bc7() -> Vec<u8> {
    let (w, h) = TEXTURE_EXTENT;
    let mut rgba = vec![0u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let border = x < 2 || y < 2 || x >= w - 2 || y >= h - 2;
            let checker = ((x / 8) + (y / 8)) % 2 == 0;
            let color: [u8; 4] = match (border, checker) {
                (true, _) => [18, 18, 26, 255],
                (false, true) => [232, 196, 84, 255],
                (false, false) => [46, 92, 140, 255],
            };
            let at = ((y * w + x) * 4) as usize;
            rgba[at..at + 4].copy_from_slice(&color);
        }
    }
    bc7::encode(&rgba, TEXTURE_EXTENT)
}

/// What the scene owns on the GPU once [`upload`] has run.
pub struct SceneResources {
    /// Vertex array, read through [`SceneResources::vertex_address`].
    pub vertices: BufferHandle,
    /// Index stream for the indexed draw.
    pub indices: BufferHandle,
    /// How many indices the draw consumes.
    pub index_count: u32,
    /// The BC7 texture.
    pub texture: ImageHandle,
    /// Its slot in the global sampled-image array (§4.3: materials are
    /// indices).
    pub texture_index: TextureIndex,
    /// The GPU pointer the vertex shader pulls from.
    pub vertex_address: DeviceAddress,
    /// The scene's pipeline.
    pub pipeline: PipelineHandle,
}

/// The slice of an RHI a scene needs to put itself on the GPU.
///
/// It lives here, not in `gg-rhi`: §3's budget for that crate says no backend
/// abstraction, and this is a demo's convenience over two concrete entry
/// points, not an engine seam. What it buys is that gg-golden and the windowed
/// demo run *the same* upload code, so the golden image guards the transfer
/// path too.
pub trait SceneHost {
    /// See [`gg_rhi::Rhi::create_buffer`].
    fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> Result<BufferHandle, RhiError>;
    /// See [`gg_rhi::Rhi::buffer_address`].
    fn buffer_address(&self, handle: BufferHandle) -> Result<DeviceAddress, RhiError>;
    /// See [`gg_rhi::Rhi::create_image`].
    fn create_image(&mut self, desc: &ImageDesc<'_>) -> Result<ImageHandle, RhiError>;
    /// See [`gg_rhi::Rhi::upload_buffer`].
    fn upload_buffer(
        &mut self,
        handle: BufferHandle,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), RhiError>;
    /// See [`gg_rhi::Rhi::upload_image`].
    fn upload_image(&mut self, handle: ImageHandle, bytes: &[u8]) -> Result<(), RhiError>;
    /// See [`gg_rhi::Rhi::flush_uploads`].
    fn flush_uploads(&mut self) -> Result<(), RhiError>;
    /// See [`gg_rhi::Rhi::register_texture`].
    fn register_texture(&mut self, handle: ImageHandle) -> Result<TextureIndex, RhiError>;
    /// See [`gg_rhi::Rhi::create_pipeline`].
    fn create_pipeline(&mut self, desc: &PipelineDesc<'_>) -> Result<PipelineHandle, RhiError>;
}

macro_rules! impl_scene_host {
    ($ty:ty) => {
        impl SceneHost for $ty {
            fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> Result<BufferHandle, RhiError> {
                <$ty>::create_buffer(self, desc)
            }
            fn buffer_address(&self, handle: BufferHandle) -> Result<DeviceAddress, RhiError> {
                <$ty>::buffer_address(self, handle)
            }
            fn create_image(&mut self, desc: &ImageDesc<'_>) -> Result<ImageHandle, RhiError> {
                <$ty>::create_image(self, desc)
            }
            fn upload_buffer(
                &mut self,
                handle: BufferHandle,
                offset: u64,
                bytes: &[u8],
            ) -> Result<(), RhiError> {
                <$ty>::upload_buffer(self, handle, offset, bytes)
            }
            fn upload_image(&mut self, handle: ImageHandle, bytes: &[u8]) -> Result<(), RhiError> {
                <$ty>::upload_image(self, handle, bytes)
            }
            fn flush_uploads(&mut self) -> Result<(), RhiError> {
                <$ty>::flush_uploads(self)
            }
            fn register_texture(&mut self, handle: ImageHandle) -> Result<TextureIndex, RhiError> {
                <$ty>::register_texture(self, handle)
            }
            fn create_pipeline(
                &mut self,
                desc: &PipelineDesc<'_>,
            ) -> Result<PipelineHandle, RhiError> {
                <$ty>::create_pipeline(self, desc)
            }
        }
    };
}

impl_scene_host!(gg_rhi::Rhi);
impl_scene_host!(gg_rhi::OffscreenRhi);

/// Create the scene's buffers, texture and pipeline, upload everything through
/// the staging ring, and wait for the transfer to land.
pub fn upload(host: &mut impl SceneHost) -> Result<SceneResources, RhiError> {
    let (vertices, indices) = cube();
    let vertex_bytes: &[u8] = bytemuck::cast_slice(&vertices);
    let index_bytes: &[u8] = bytemuck::cast_slice(&indices);

    let vertex_buffer = host.create_buffer(&BufferDesc {
        name: "demo02.vertices",
        size: vertex_bytes.len() as u64,
        kind: BufferKind::Storage,
    })?;
    let index_buffer = host.create_buffer(&BufferDesc {
        name: "demo02.indices",
        size: index_bytes.len() as u64,
        kind: BufferKind::Index,
    })?;
    let texture = host.create_image(&ImageDesc {
        name: "demo02.checker",
        extent: TEXTURE_EXTENT,
        format: ImageFormat::Bc7Srgb,
        usage: ImageUse::Sampled,
    })?;

    host.upload_buffer(vertex_buffer, 0, vertex_bytes)?;
    host.upload_buffer(index_buffer, 0, index_bytes)?;
    host.upload_image(texture, &texture_bc7())?;
    host.flush_uploads()?;

    let texture_index = host.register_texture(texture)?;
    let vertex_address = host.buffer_address(vertex_buffer)?;
    let pipeline = host.create_pipeline(&pipeline_desc())?;

    Ok(SceneResources {
        vertices: vertex_buffer,
        indices: index_buffer,
        index_count: indices.len() as u32,
        texture,
        texture_index,
        vertex_address,
        pipeline,
    })
}

/// The frame's extract stage (§4.1): sim world in, render-space instances out,
/// plus the camera they are relative to.
///
/// The demo and gg-golden both call this, so the golden image is produced by
/// the demo's actual path — including the narrowing — rather than by a
/// lookalike that happens to agree today.
pub fn extract(sim: &Sim, out: &mut Extracted) -> Result<Camera, SimError> {
    let rig = sim.camera()?;
    out.transforms::<Cube>(&sim.world, rig.position)?;
    Ok(Camera::from_rig(rig))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::sim::TICKS_PER_TURN;

    /// The opening frame, extracted — one camera, one cube.
    fn frame_at(tick: u64) -> (Camera, Instance) {
        frame_at_origin(tick, sim::DVec3::new(0.0, 0.0, 0.0))
    }

    fn frame_at_origin(tick: u64, origin: sim::DVec3) -> (Camera, Instance) {
        let mut sim = Sim::new_at(0, origin).unwrap();
        let input = crate::sim::input().unwrap();
        for _ in 0..tick {
            sim.tick(&input).unwrap();
        }
        let mut out = Extracted::default();
        let camera = extract(&sim, &mut out).unwrap();
        assert_eq!(out.instances.len(), 1, "the opening sim holds one cube");
        (camera, out.instances[0])
    }

    #[test]
    fn the_cube_is_closed_and_its_normals_point_outward() {
        let (vertices, indices) = cube();
        assert_eq!(vertices.len(), 24);
        assert_eq!(indices.len(), 36);
        for v in &vertices {
            // Every corner of a unit cube is half a unit out on each axis.
            for axis in v.position {
                assert!(
                    (axis.abs() - 0.5).abs() < 1e-6,
                    "corner at {:?}",
                    v.position
                );
            }
            // Outward normals: the dot with the corner is positive on the axis
            // the face faces.
            let dot = v.position[0] * v.normal[0]
                + v.position[1] * v.normal[1]
                + v.position[2] * v.normal[2];
            assert!(dot > 0.0, "normal {:?} faces inward", v.normal);
            for c in v.uv {
                assert!((0.0..=1.0).contains(&c), "uv out of range: {:?}", v.uv);
            }
        }
    }

    #[test]
    fn the_texture_is_a_whole_number_of_bc7_blocks() {
        let bytes = texture_bc7();
        let blocks = (TEXTURE_EXTENT.0 / 4) * (TEXTURE_EXTENT.1 / 4);
        assert_eq!(bytes.len(), blocks as usize * bc7::BLOCK_BYTES);
        // What the RHI will check the upload against.
        assert_eq!(
            bytes.len() as u64,
            ImageFormat::Bc7Srgb.packed_size(TEXTURE_EXTENT)
        );
    }

    #[test]
    fn the_spin_is_a_function_of_the_tick_count_alone() {
        // A replay that reaches tick N sees the same orientation every time,
        // because the sim counts ticks rather than accumulating a quaternion
        // (§2, Sim time row) — so a full turn lands back on itself exactly.
        let at = |tick| {
            let (camera, instance) = frame_at(tick);
            view_projection_for(&camera, GOLDEN_EXTENT, &instance)
        };
        assert_eq!(at(7), at(7 + TICKS_PER_TURN));
        assert_ne!(at(7), at(8));
    }

    #[test]
    fn a_camera_a_trillion_metres_out_frames_the_cube_the_same_way() {
        // The exit criterion behind `f64` sim positions (§4.2.1), measured
        // rather than eyeballed: at 10^12 m an `f64` ulp is ~0.24 mm, so
        // subtract-then-narrow lands every corner well inside a *tenth* of a
        // pixel of the near frame. Narrowing first would land it ~65 km out,
        // which is the jitter this whole membrane exists to prevent.
        let (near_camera, near) = frame_at_origin(0, sim::DVec3::new(0.0, 0.0, 0.0));
        let (far_camera, far) = frame_at_origin(0, crate::sim::FAR_ORIGIN);
        let near_m = view_projection_for(&near_camera, GOLDEN_EXTENT, &near);
        let far_m = view_projection_for(&far_camera, GOLDEN_EXTENT, &far);

        let tenth_of_a_pixel = 0.1 * 2.0 / GOLDEN_EXTENT.0 as f32;
        for v in cube().0 {
            let p = render::Vec4::new(v.position[0], v.position[1], v.position[2], 1.0);
            let (a, b) = (near_m * p, far_m * p);
            let (a, b) = (a.truncate() / a.w, b.truncate() / b.w);
            assert!(
                (a.x - b.x).abs() < tenth_of_a_pixel && (a.y - b.y).abs() < tenth_of_a_pixel,
                "corner {:?} moved from {a:?} to {b:?} at 10^12 m",
                v.position
            );
        }
    }

    #[test]
    fn the_whole_cube_lands_inside_the_golden_frame() {
        // Not merely "in front of the camera": every corner must be inside the
        // NDC box. A camera aimed past the mesh still puts corners in front of
        // it, which is how the first version of this pose shipped an empty
        // golden image, so the assertion is the framing, not the hemisphere.
        let (camera, instance) = frame_at(0);
        let m = view_projection_for(&camera, GOLDEN_EXTENT, &instance);
        for v in cube().0 {
            let p = m * render::Vec4::new(v.position[0], v.position[1], v.position[2], 1.0);
            assert!(p.w > 0.0, "corner {:?} is behind the camera", v.position);
            let ndc = p.truncate() / p.w;
            assert!(
                (-1.0..=1.0).contains(&ndc.x) && (-1.0..=1.0).contains(&ndc.y),
                "corner {:?} is off-frame at {ndc:?}",
                v.position
            );
            // Reverse-Z: nearer is greater, and nothing is past the far plane
            // because there isn't one.
            assert!(
                (0.0..=1.0).contains(&ndc.z),
                "corner {:?} has depth {}",
                v.position,
                ndc.z
            );
        }
    }
}
