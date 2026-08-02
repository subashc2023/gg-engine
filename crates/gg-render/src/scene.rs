//! The pack pass (§4.5's v1 list, fed by §4.6): pull a mesh's vertices by
//! device address, sample its base colour out of the global descriptor set.
//!
//! It is the [`BoxPass`](crate::BoxPass) with two differences that matter: the
//! vertex stride is a *file's* rather than a struct's, and the geometry is
//! per-draw rather than one buffer for every instance. Both passes render into
//! the same two attachments and share the graph, so a frame drawing boxes and
//! pack meshes together is one prepass and one forward pass, not two of each.
//!
//! # The white texel
//!
//! A material naming no base-colour map samples a 1x1 white image this pass
//! allocates once. The alternative — a branch on a sentinel index — puts a
//! divergent read in the fragment shader for a case the artist could fix by
//! assigning a texture, and costs four bytes of VRAM to avoid.

use gg_extract::Extracted;
use gg_math::render;
use gg_rhi::{
    Blend, ColorTarget, DepthMode, DrawSpec, ImageDesc, ImageFormat, ImageHandle, ImageUse,
    PipelineDesc, PipelineHandle, RhiError, Sampler, TextureIndex,
};

use crate::content::Content;
use crate::shaders_gen::scene as shader;
use crate::{GpuHost, SCENE_FORMAT, View, srgb_to_linear};

/// The pack vertex the shader indexes. Not declared here — this asserts that
/// `gg_assets::Vertex` is what `scene.slang`'s `VERTEX_STRIDE` says it is, so a
/// change to the file format is a build error rather than a garbled mesh.
const _: () = {
    assert!(core::mem::size_of::<gg_assets::Vertex>() == 32);
    assert!(core::mem::offset_of!(gg_assets::Vertex, normal) == 12);
    assert!(core::mem::offset_of!(gg_assets::Vertex, uv) == 24);
};

/// One mesh instance, ready to draw. The push constants have to outlive the
/// [`DrawSpec`]s that borrow them, and the index buffer varies per mesh — which
/// is the one thing `BoxPass` gets to keep constant and this cannot.
struct Drawable {
    push: shader::ScenePush,
    indices: gg_rhi::BufferHandle,
    index_count: u32,
}

/// The pack pass: two pipelines, one fallback texel, this frame's draws.
pub(crate) struct ScenePass {
    prepass: PipelineHandle,
    forward: PipelineHandle,
    white: ImageHandle,
    white_index: TextureIndex,
    drawables: Vec<Drawable>,
}

impl ScenePass {
    /// Build the pipelines and the white texel. One flush, at startup.
    pub(crate) fn new(rhi: &mut impl GpuHost) -> Result<Self, RhiError> {
        let white = rhi.create_image(&ImageDesc {
            name: "render.scene.white",
            extent: (1, 1),
            format: ImageFormat::Rgba8Srgb,
            usage: ImageUse::Sampled,
            mip_levels: 1,
        })?;
        rhi.upload_image(white, 0, &[0xff; 4])?;
        rhi.flush_uploads()?;
        Ok(ScenePass {
            prepass: rhi.create_pipeline(&prepass_desc())?,
            forward: rhi.create_pipeline(&forward_desc())?,
            white,
            white_index: rhi.register_texture(white)?,
            drawables: Vec::new(),
        })
    }

    /// Rebuild this frame's draws from the models extract produced.
    ///
    /// An instance whose mesh is not resident yet is skipped, not deferred: the
    /// frame draws what has arrived, and the next one draws more. That is the
    /// whole visible behaviour of streaming, and it is why nothing here waits.
    pub(crate) fn build(
        &mut self,
        extent: (u32, u32),
        extracted: &Extracted,
        view: &View,
        content: Option<&Content>,
    ) {
        self.drawables.clear();
        let Some(content) = content else {
            return;
        };
        let view_projection = view.view_projection(extent);
        for instance in &extracted.models {
            let id = gg_assets::AssetId(instance.asset);
            let Some(mesh) = content.mesh(id) else {
                continue;
            };
            let Some(indices) = mesh.indices.filter(|_| mesh.index_count > 0) else {
                continue;
            };
            let material = content.material(id);
            // The material's base colour is linear in the file; the game's tint
            // is sRGB bytes it chose. Both multiply, so both must be linear
            // first — the one place the two colour spaces meet.
            let tint = srgb_to_linear(instance.color);
            let model = render::Mat4::from_scale_rotation_translation(
                instance.half_extent,
                instance.rotation,
                instance.offset,
            );
            let texture = content
                .texture(material.base_color_texture)
                .map_or(self.white_index, |resident| resident.index);
            self.drawables.push(Drawable {
                push: shader::ScenePush::new(
                    render::rows(view_projection * model),
                    mesh.address,
                    [
                        tint[0] * material.base_color[0],
                        tint[1] * material.base_color[1],
                        tint[2] * material.base_color[2],
                        1.0,
                    ],
                    instance.rotation.to_array(),
                    texture.get(),
                    // Repeat, not clamp: a glTF uv is free to leave [0,1] and
                    // tiling is how a floor is authored.
                    Sampler::LinearRepeat.index(),
                ),
                indices,
                index_count: mesh.index_count,
            });
        }
    }

    /// This frame's mesh draws through `pipeline`.
    pub(crate) fn draws(&self, pipeline: PipelineHandle) -> Vec<DrawSpec<'_>> {
        self.drawables
            .iter()
            .map(|drawable| DrawSpec {
                pipeline,
                push_constants: bytemuck::bytes_of(&drawable.push),
                count: drawable.index_count,
                index_buffer: Some(drawable.indices),
            })
            .collect()
    }

    /// The depth-prepass pipeline.
    pub(crate) fn prepass(&self) -> PipelineHandle {
        self.prepass
    }

    /// The forward pipeline.
    pub(crate) fn forward(&self) -> PipelineHandle {
        self.forward
    }

    /// Release the pipelines and the white texel.
    pub(crate) fn destroy(self, rhi: &mut impl GpuHost) -> Result<(), RhiError> {
        rhi.destroy_pipeline(self.prepass)?;
        rhi.destroy_pipeline(self.forward)?;
        rhi.destroy_image(self.white)
    }
}

/// Position only, depth stored — see `ugly.slang` for why it shares the block.
fn prepass_desc() -> PipelineDesc<'static> {
    PipelineDesc {
        name: "scene.prepass",
        vs_spirv: shader::VS_DEPTH_SPIRV,
        vs_entry: shader::VS_DEPTH_ENTRY,
        fs_spirv: shader::FS_DEPTH_SPIRV,
        fs_entry: shader::FS_DEPTH_ENTRY,
        push_constant_size: core::mem::size_of::<shader::ScenePush>() as u32,
        color: ColorTarget::None,
        blend: Blend::Off,
        depth: DepthMode::Write,
    }
}

/// The forward pass into the scene attachment, depth tested against the
/// prepass's result — the same arrangement `ugly.forward` uses.
fn forward_desc() -> PipelineDesc<'static> {
    PipelineDesc {
        name: "scene.forward",
        vs_spirv: shader::VS_MAIN_SPIRV,
        vs_entry: shader::VS_MAIN_ENTRY,
        fs_spirv: shader::FS_MAIN_SPIRV,
        fs_entry: shader::FS_MAIN_ENTRY,
        push_constant_size: core::mem::size_of::<shader::ScenePush>() as u32,
        color: ColorTarget::Format(SCENE_FORMAT),
        blend: Blend::Off,
        depth: DepthMode::TestOnly,
    }
}
