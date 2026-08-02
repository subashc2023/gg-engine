//! The UI pass: one pipeline, one draw, one vertex format (§4.9's draw layer).
//!
//! Here rather than in `gg-debug` because the *pass* is a renderer's — a
//! pipeline, a coverage atlas and a per-frame vertex stream — while what fills
//! it is not. §4.9 splits the same way ("glyph atlas management is ours; what
//! fills the atlas is not"), and it is the split that lets M8's overlay be
//! replaced by M13's `gg-ui` without the renderer noticing: what crosses is a
//! slice of [`UiVertex`] and a coverage bitmap, and neither knows what a
//! console is.

use crate::shaders_gen::ui as shader;
use gg_rhi::{
    Blend, BufferDesc, BufferHandle, BufferKind, ColorTarget, DepthMode, DeviceAddress, DrawSpec,
    FRAMES_IN_FLIGHT, ImageDesc, ImageFormat, ImageHandle, ImageUse, PipelineDesc, PipelineHandle,
    RhiError, Sampler, TextureIndex,
};

/// One UI vertex: a position in pixels, a coverage-atlas coordinate, and a
/// colour. Mirrors `ui.slang`'s `UI_VERTEX_STRIDE` byte layout — the shader
/// reads these at hardcoded scalar offsets through a device address, so the
/// layout is frozen below rather than trusted.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiVertex {
    /// Pixels, origin top-left.
    pub pos: [f32; 2],
    /// Coverage atlas coordinate in `[0, 1]`.
    pub uv: [f32; 2],
    /// `0xAARRGGBB` — the same byte order the render protocol's `Renderable`
    /// uses, so one colour constant means one thing engine-wide.
    pub color: u32,
}

const _: () = {
    assert!(core::mem::size_of::<UiVertex>() == 20);
    assert!(core::mem::offset_of!(UiVertex, uv) == 8);
    assert!(core::mem::offset_of!(UiVertex, color) == 16);
};

/// Vertices one frame may draw. Six per quad; the overlay's fullest frame is a
/// few thousand glyphs, and the buffer costs 20 bytes each per frame slot.
/// Overflow truncates rather than reallocating mid-frame — see [`UiPass::write`].
pub const MAX_VERTICES: usize = 24 * 1024;

/// A grayscale coverage atlas: `extent.0 * extent.1` bytes, one per texel.
///
/// One byte per texel rather than the four the format holds, because the
/// caller authors coverage and expanding it is this side's business.
pub struct Coverage<'a> {
    /// Width and height in texels.
    pub extent: (u32, u32),
    /// Row-major, `extent.0 * extent.1` long.
    pub texels: &'a [u8],
}

/// The pass's GPU state: the pipeline, the per-frame vertex stream, and the
/// atlas its glyphs are cut from.
pub(crate) struct UiPass {
    pipeline: PipelineHandle,
    vertices: BufferHandle,
    address: DeviceAddress,
    atlas: Option<(ImageHandle, TextureIndex)>,
    /// This frame's vertex count and which slot's region holds it.
    pending: (u32, u64),
}

/// Bytes one frame slot's vertices occupy.
const SLOT_BYTES: u64 = (MAX_VERTICES * core::mem::size_of::<UiVertex>()) as u64;

impl UiPass {
    /// Build the pipeline and the vertex stream. No atlas yet: a pass with
    /// nothing to draw needs none, and the shell may never open an overlay.
    pub(crate) fn new(rhi: &mut impl super::GpuHost) -> Result<Self, RhiError> {
        let vertices = rhi.create_buffer(&BufferDesc {
            name: "render.ui.vertices",
            // One region per frame in flight: [`BufferKind::Dynamic`] does not
            // wait, so a single region would be this frame overwriting what its
            // predecessor is still reading.
            size: SLOT_BYTES * FRAMES_IN_FLIGHT,
            kind: BufferKind::Dynamic,
        })?;
        Ok(UiPass {
            pipeline: rhi.create_pipeline(&pipeline_desc())?,
            vertices,
            address: rhi.buffer_address(vertices)?,
            atlas: None,
            pending: (0, 0),
        })
    }

    /// Upload a coverage atlas and give it a bindless slot. Replaces any
    /// previous one, whose image is released.
    pub(crate) fn set_atlas(
        &mut self,
        rhi: &mut impl super::GpuHost,
        coverage: &Coverage<'_>,
    ) -> Result<(), RhiError> {
        let texels = (coverage.extent.0 as usize) * (coverage.extent.1 as usize);
        if coverage.texels.len() != texels {
            return Err(RhiError::Loader(format!(
                "coverage atlas {:?} needs {texels} texels, got {}",
                coverage.extent,
                coverage.texels.len()
            )));
        }
        // Rgba8Unorm rather than a single-channel format: §4.3 keeps one
        // pixel-format entry per job that exists, and "masks and data textures"
        // is already that job. Four bytes a texel on an 18 KiB atlas is not the
        // cost worth adding a format for.
        let mut rgba = Vec::with_capacity(texels * 4);
        for texel in coverage.texels {
            rgba.extend_from_slice(&[*texel; 4]);
        }
        let image = rhi.create_image(&ImageDesc {
            name: "render.ui.atlas",
            extent: coverage.extent,
            format: ImageFormat::Rgba8Unorm,
            usage: ImageUse::Sampled,
            mip_levels: 1,
        })?;
        rhi.upload_image(image, 0, &rgba)?;
        rhi.flush_uploads()?;
        let index = rhi.register_texture(image)?;
        if let Some((old, _)) = self.atlas.replace((image, index)) {
            rhi.destroy_image(old)?;
        }
        Ok(())
    }

    /// Whether a frame's vertices can be drawn at all — an atlas has to exist
    /// for a glyph to be cut from.
    pub(crate) fn ready(&self) -> bool {
        self.atlas.is_some()
    }

    /// Copy this frame's vertices into `slot`'s region. Truncates at
    /// [`MAX_VERTICES`] with a warning: a UI that outgrows its buffer is a
    /// budget to raise deliberately, and dropping the frame or reallocating
    /// mid-graph are both worse answers than a short overlay.
    pub(crate) fn write(
        &mut self,
        rhi: &mut impl super::GpuHost,
        slot: u64,
        vertices: &[UiVertex],
    ) -> Result<(), RhiError> {
        let count = vertices.len().min(MAX_VERTICES);
        if count < vertices.len() {
            tracing::warn!(
                wanted = vertices.len(),
                cap = MAX_VERTICES,
                "ui vertices truncated"
            );
        }
        // Quads only: a trailing partial triangle would rasterize as garbage.
        let count = count - count % 3;
        let offset = slot % FRAMES_IN_FLIGHT * SLOT_BYTES;
        rhi.write_buffer(
            self.vertices,
            offset,
            bytemuck::cast_slice(&vertices[..count]),
        )?;
        self.pending = (count as u32, offset);
        Ok(())
    }

    /// The push constants for the draw [`Self::write`] staged.
    pub(crate) fn push(&self, extent: (u32, u32)) -> shader::UiPush {
        shader::UiPush::new(
            self.address + self.pending.1,
            self.atlas.map_or(0, |(_, index)| index.get()),
            Sampler::NearestClamp.index(),
            [1.0 / extent.0.max(1) as f32, 1.0 / extent.1.max(1) as f32],
        )
    }

    /// The one draw, or none when the frame drew nothing.
    pub(crate) fn draw<'a>(&self, push: &'a shader::UiPush) -> Option<DrawSpec<'a>> {
        (self.pending.0 > 0 && self.ready()).then_some(DrawSpec {
            pipeline: self.pipeline,
            push_constants: bytemuck::bytes_of(push),
            count: self.pending.0,
            index_buffer: None,
        })
    }

    pub(crate) fn destroy(self, rhi: &mut impl super::GpuHost) -> Result<(), RhiError> {
        if let Some((image, _)) = self.atlas {
            rhi.destroy_image(image)?;
        }
        rhi.destroy_pipeline(self.pipeline)?;
        rhi.destroy_buffer(self.vertices)
    }
}

/// Alpha-blended, depth-free, onto whatever the frame lands in: the UI is the
/// last thing over the resolved scene.
fn pipeline_desc() -> PipelineDesc<'static> {
    PipelineDesc {
        name: "ui",
        vs_spirv: shader::VS_MAIN_SPIRV,
        vs_entry: shader::VS_MAIN_ENTRY,
        fs_spirv: shader::FS_MAIN_SPIRV,
        fs_entry: shader::FS_MAIN_ENTRY,
        push_constant_size: core::mem::size_of::<shader::UiPush>() as u32,
        color: ColorTarget::Backbuffer,
        blend: Blend::Alpha,
        depth: DepthMode::Off,
    }
}
