//! The pack pass (§4.5's v1 list, fed by §4.6): pull a mesh's vertices by
//! device address, sample its base colour out of the global descriptor set.
//!
//! It is the [`BoxPass`](crate::BoxPass) with two differences that matter: the
//! vertex stride is a *file's* rather than a struct's, and the geometry is
//! per-mesh rather than one buffer for every instance. Both passes render into
//! the same two attachments and share the graph, so a frame drawing boxes and
//! pack meshes together is one prepass and one forward pass, not two of each.
//!
//! # Sorted, batched, indirect (§6 M10)
//!
//! What arrives from extract is a flat, culled, order-stable array of instances.
//! This turns it into draws in three steps:
//!
//! 1. **Key.** Each instance gets a `u64` of pipeline, then material, then
//!    depth. Sorting by it puts everything sharing a mesh together and orders
//!    the rest front-to-back, which is what a depth prepass wants.
//! 2. **Batch.** A run of equal mesh becomes one draw of many instances. The
//!    per-instance data — matrix, tint, orientation — goes into one array the
//!    shader indexes by `SV_InstanceID`.
//! 3. **Indirect.** The draw's counts are read out of device memory rather than
//!    recorded, so a GPU culling pass (P2, §7) has somewhere to write them.
//!
//! The sort is **stable**, so instances with equal keys keep the order extract
//! produced — which is a function of world state alone (§4.2). An unstable sort
//! here would make the draw list depend on the sort's own internal choices, a
//! second source of frame-to-frame variation downstream of the one extract
//! works to remove.
//!
//! Sorting happens here rather than in `gg-extract` because a material is
//! *content*: extract resolves no ids and links no pack, deliberately.
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
    Blend, BufferDesc, BufferHandle, BufferKind, ColorTarget, DepthMode, DeviceAddress, DrawSpec,
    FRAMES_IN_FLIGHT, ImageDesc, ImageFormat, ImageHandle, ImageUse, Indirect, IndirectCommand,
    PipelineDesc, PipelineHandle, RhiError, Sampler, Samples, TextureIndex,
};

use crate::content::Content;
use crate::shaders_gen::scene as shader;
use crate::{GpuHost, SCENE_FORMAT, View, srgb_to_linear};

/// The pack vertex the shader indexes. Not declared here — this asserts that
/// `gg_assets::Vertex` is what `scene.slang`'s `VERTEX_STRIDE` says it is, so a
/// change to the file format is a build error rather than a garbled mesh.
const _: () = {
    assert!(core::mem::size_of::<gg_assets::Vertex>() == 48);
    assert!(core::mem::offset_of!(gg_assets::Vertex, normal) == 12);
    assert!(core::mem::offset_of!(gg_assets::Vertex, uv) == 24);
    assert!(core::mem::offset_of!(gg_assets::Vertex, tangent) == 32);
};

/// One instance as the shader reads it: the finished object-to-clip matrix, the
/// colour to multiply, and the orientation normals are rotated by.
///
/// The matrix is the *product*, computed here rather than assembled in the
/// shader from a view-projection and a model. Multiplying in a different place
/// moves the low bits, and the blessed `hall` reference (§4.10) is a byte
/// comparison against pixels that depend on them.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuInstance {
    mvp: [[f32; 4]; 4],
    tint: [f32; 4],
    rotation: [f32; 4],
    /// Camera-relative translation in `xyz`; `w` unused. Carried *beside* the
    /// matrix rather than read out of it because lighting needs the world
    /// position and a clip coordinate cannot be inverted back to one — and
    /// because the matrix is the finished product, whose low bits are the whole
    /// reason it is precomputed.
    offset: [f32; 4],
    /// Per-axis scale in `xyz`, applied before the rotation; `w` unused.
    scale: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<GpuInstance>() == 128);

/// Instances one frame may draw, per frame-in-flight slot.
///
/// A budget rather than a growable buffer: reallocating mid-graph would orphan
/// a buffer the previous frame is still reading, and a frame that draws more
/// than this is a budget to raise deliberately — [`ScenePass::batch`] says so
/// out loud rather than silently dropping the tail.
const MAX_INSTANCES: usize = 64 * 1024;

/// Draws one frame may issue. One per distinct mesh after batching, so this
/// counts *meshes on screen* rather than objects.
const MAX_BATCHES: usize = 4 * 1024;

const INSTANCE_SLOT_BYTES: u64 = (MAX_INSTANCES * core::mem::size_of::<GpuInstance>()) as u64;
const COMMAND_SLOT_BYTES: u64 = (MAX_BATCHES * core::mem::size_of::<IndirectCommand>()) as u64;

/// One instance's *place* in the draw order, before sorting and batching.
///
/// The 128-byte [`GpuInstance`] this points at deliberately does not travel with
/// it. Sorting the payload inline moved ~1.2 MB a frame at ten thousand objects
/// to answer a question about eight bytes, and measured 1.27 ms — 43 % of the
/// renderer's host frame (§6 M25). The permutation is unchanged: the sort is
/// still stable on the same `key`, so the bytes reaching the GPU are identical
/// and no golden moves.
struct Keyed {
    key: u64,
    mesh: u64,
    /// Index into [`ScenePass::built`], which stays in extract order.
    at: u32,
}

/// One draw: a run of instances that share a mesh.
struct Batch {
    push: shader::ScenePush,
    indices: BufferHandle,
    /// Byte offset of this batch's command, absolute within the buffer.
    command_offset: u64,
}

/// The pack pass: two pipelines, one fallback texel, this frame's draws.
pub(crate) struct ScenePass {
    /// Prepass + forward, per sample count (§6 M21).
    variants: crate::Variants,
    shadow: PipelineHandle,
    white: ImageHandle,
    white_index: TextureIndex,
    /// The flat-normal fallback. A separate texel from `white`, because white
    /// decodes to a tangent normal of `(1, 1, ?)` whose reconstructed z is the
    /// square root of a negative number — a NaN normal, and a black mesh.
    flat: ImageHandle,
    flat_index: TextureIndex,
    /// Per-instance data, one region per frame in flight — `BufferKind::Dynamic`
    /// does not wait, so a single region would be this frame overwriting what
    /// its predecessor is still reading.
    instances: BufferHandle,
    instance_address: DeviceAddress,
    /// The CPU-built draw commands: same slotting, same reason.
    commands: BufferHandle,
    /// Scratch, reused across frames.
    keyed: Vec<Keyed>,
    /// This frame's instances in *extract* order — what `keyed` indexes into.
    /// Separate from `staged`, which is the same data in *draw* order.
    built: Vec<GpuInstance>,
    staged: Vec<GpuInstance>,
    batches: Vec<Batch>,
    /// One push per batch per cascade, parallel to `batches` (§6 M15.3). Held
    /// beside the batch's own rather than replacing it: the forward and prepass
    /// draws borrow that one, and rewriting it per cascade would rewrite what
    /// they are still pointing at.
    shadow_pushes: Vec<Vec<shader::ScenePush>>,
    written: Vec<IndirectCommand>,
    /// This frame's lighting block, patched into every batch's push.
    frame: DeviceAddress,
}

impl ScenePass {
    /// Build the pipelines, the white texel and the per-frame streams. One
    /// flush, at startup.
    pub(crate) fn new(rhi: &mut impl GpuHost) -> Result<Self, RhiError> {
        let white = rhi.create_image(&ImageDesc {
            name: "render.scene.white",
            extent: (1, 1),
            format: ImageFormat::Rgba8Srgb,
            usage: ImageUse::Sampled,
            mip_levels: 1,
            samples: Samples::X1,
        })?;
        rhi.upload_image(white, 0, &[0xff; 4])?;
        let flat = rhi.create_image(&ImageDesc {
            name: "render.scene.flat-normal",
            extent: (1, 1),
            // Unorm, not sRGB: a normal map is data and must not be decoded.
            format: ImageFormat::Rgba8Unorm,
            usage: ImageUse::Sampled,
            mip_levels: 1,
            samples: Samples::X1,
        })?;
        // (0.5, 0.5) decodes to a tangent-space (0, 0, 1) — the flat normal.
        rhi.upload_image(flat, 0, &[0x80, 0x80, 0xff, 0xff])?;
        rhi.flush_uploads()?;
        let instances = rhi.create_buffer(&BufferDesc {
            name: "render.scene.instances",
            size: INSTANCE_SLOT_BYTES * FRAMES_IN_FLIGHT,
            kind: BufferKind::Dynamic,
        })?;
        let commands = rhi.create_buffer(&BufferDesc {
            name: "render.scene.commands",
            size: COMMAND_SLOT_BYTES * FRAMES_IN_FLIGHT,
            kind: BufferKind::Indirect,
        })?;
        // 1× eagerly, every other count on first ask — see `BoxPass::new`.
        let mut variants = crate::Variants::default();
        variants.get(rhi, Samples::X1, |s| [prepass_desc(s), forward_desc(s)])?;

        Ok(ScenePass {
            variants,
            shadow: rhi.create_pipeline(&shadow_desc())?,
            white,
            white_index: rhi.register_texture(white)?,
            flat,
            flat_index: rhi.register_texture(flat)?,
            instances,
            instance_address: rhi.buffer_address(instances)?,
            commands,
            keyed: Vec::new(),
            built: Vec::new(),
            staged: Vec::new(),
            batches: Vec::new(),
            written: Vec::new(),
            shadow_pushes: Vec::new(),
            frame: 0,
        })
    }

    /// Rebuild this frame's draws from the models extract produced, and stage
    /// them into `slot`'s regions.
    ///
    /// An instance whose mesh is not resident yet is skipped, not deferred: the
    /// frame draws what has arrived, and the next one draws more. That is the
    /// whole visible behaviour of streaming, and it is why nothing here waits.
    ///
    /// # Errors
    ///
    /// Whatever the two buffer writes refuse.
    #[expect(clippy::too_many_arguments, reason = "one frame's inputs")]
    pub(crate) fn build(
        &mut self,
        rhi: &mut impl GpuHost,
        slot: u64,
        extent: (u32, u32),
        extracted: &Extracted,
        view: &View,
        content: Option<&Content>,
        frame: DeviceAddress,
        cascades: usize,
    ) -> Result<(), RhiError> {
        self.frame = frame;
        self.shadow_pushes.resize_with(cascades, Vec::new);
        self.shadow_pushes.truncate(cascades);
        self.keyed.clear();
        self.built.clear();
        self.staged.clear();
        self.batches.clear();
        self.written.clear();
        let Some(content) = content else {
            return Ok(());
        };
        let view_projection = view.view_projection(extent);
        {
            gg_core::zone!("scene.key");
            for instance in &extracted.models {
                let id = gg_assets::AssetId(instance.asset);
                let Some(mesh) = content.mesh(id) else {
                    continue;
                };
                if mesh.index_count == 0 || mesh.indices.is_none() {
                    continue;
                }
                let material = content.material(id);
                // The material's base colour is linear in the file; the game's
                // tint is sRGB bytes it chose. Both multiply, so both must be
                // linear first — the one place the two colour spaces meet.
                let tint = srgb_to_linear(instance.color);
                let model = render::Mat4::from_scale_rotation_translation(
                    instance.half_extent,
                    instance.rotation,
                    instance.offset,
                );
                self.built.push(GpuInstance {
                    mvp: render::rows(view_projection * model),
                    tint: [
                        tint[0] * material.base_color[0],
                        tint[1] * material.base_color[1],
                        tint[2] * material.base_color[2],
                        1.0,
                    ],
                    rotation: instance.rotation.to_array(),
                    offset: instance.offset.extend(0.0).to_array(),
                    scale: instance.half_extent.extend(0.0).to_array(),
                });
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "MAX_INSTANCES caps this far below u32::MAX"
                )]
                self.keyed.push(Keyed {
                    key: sort_key(instance.asset, instance.offset),
                    mesh: instance.asset,
                    at: (self.built.len() - 1) as u32,
                });
            }
        }
        {
            // Stable: equal keys keep extract's order, which is a function of
            // world state and nothing else (§4.2).
            gg_core::zone!("scene.sort");
            self.keyed.sort_by_key(|k| k.key);
        }
        {
            gg_core::zone!("scene.batch");
            self.batch(content);
        }
        gg_core::zone!("scene.stage");
        self.stage(rhi, slot)
    }

    /// Turn the sorted list into runs of equal mesh, one draw each.
    fn batch(&mut self, content: &Content) {
        let mut start = 0;
        while start < self.keyed.len() {
            let mesh = self.keyed[start].mesh;
            let mut end = start;
            while end < self.keyed.len() && self.keyed[end].mesh == mesh {
                end += 1;
            }
            if self.staged.len() + (end - start) > MAX_INSTANCES
                || self.batches.len() == MAX_BATCHES
            {
                tracing::warn!(
                    instances = self.keyed.len(),
                    batches = self.batches.len(),
                    instance_cap = MAX_INSTANCES,
                    batch_cap = MAX_BATCHES,
                    "scene draw list truncated — raise the budget rather than living with a \
                     frame that silently stops drawing"
                );
                break;
            }
            let id = gg_assets::AssetId(mesh);
            let material = content.material(id);
            // Residency was already proven when the instance was keyed; this
            // re-reads it for the batch's own geometry rather than carrying a
            // borrow of the pack through the sort.
            let Some(resident) = content.mesh(id) else {
                start = end;
                continue;
            };
            let Some(indices) = resident.indices else {
                start = end;
                continue;
            };
            // A map still on its way falls back to the neutral texel rather than
            // to nothing: a mesh that turned black for the two frames its normal
            // map took to arrive would look like a shading bug (§4.6).
            let map = |id, fallback: TextureIndex| {
                content.texture(id).map_or(fallback, |t| t.index).get()
            };
            self.batches.push(Batch {
                push: shader::ScenePush::new(
                    // Patched in `stage`, once the slot's byte offset is known.
                    0,
                    resident.address,
                    self.frame,
                    self.staged.len() as u32,
                    map(material.base_color_texture, self.white_index),
                    map(material.normal_texture, self.flat_index),
                    map(material.metallic_roughness_texture, self.white_index),
                    map(material.occlusion_texture, self.white_index),
                    // Repeat, not clamp: a glTF uv is free to leave [0,1] and
                    // tiling is how a floor is authored.
                    Sampler::LinearRepeat.index(),
                    material.metallic,
                    material.roughness,
                    // Overwritten per cascade in `stage`; the forward and
                    // prepass pipelines never read the field.
                    0,
                ),
                indices,
                command_offset: (self.written.len() * core::mem::size_of::<IndirectCommand>())
                    as u64,
            });
            self.written.push(IndirectCommand {
                index_count: resident.index_count,
                instance_count: (end - start) as u32,
                ..Default::default()
            });
            // Indexed rather than `extend`-ed off an iterator: `keyed` and
            // `built` are two fields being read while `staged` is written, and
            // the borrow checker splits that per statement, not per closure.
            for slot in start..end {
                let at = self.keyed[slot].at as usize;
                let instance = self.built[at];
                self.staged.push(instance);
            }
            start = end;
        }
    }

    /// Copy this frame's instances and commands into `slot`'s regions, and
    /// point every batch at them.
    fn stage(&mut self, rhi: &mut impl GpuHost, slot: u64) -> Result<(), RhiError> {
        if self.staged.is_empty() {
            return Ok(());
        }
        let slot = slot % FRAMES_IN_FLIGHT;
        let instance_offset = slot * INSTANCE_SLOT_BYTES;
        let command_offset = slot * COMMAND_SLOT_BYTES;
        rhi.write_buffer(
            self.instances,
            instance_offset,
            bytemuck::cast_slice(&self.staged),
        )?;
        rhi.write_buffer(
            self.commands,
            command_offset,
            bytemuck::cast_slice(&self.written),
        )?;
        let base = self.instance_address + instance_offset;
        for batch in &mut self.batches {
            batch.push.instances = base;
            batch.command_offset += command_offset;
        }
        // After the patch above, never before it: a cascade copy taken while
        // `instances` was still the placeholder would draw the shadow pass
        // through address zero.
        for (index, pushes) in self.shadow_pushes.iter_mut().enumerate() {
            pushes.clear();
            pushes.extend(self.batches.iter().map(|batch| {
                let mut push = batch.push;
                push.cascade = index as u32;
                push
            }));
        }
        Ok(())
    }

    /// This frame's mesh draws through `pipeline`.
    ///
    /// No `depth_bias` on any of them, the shadow one included: the RHI refuses
    /// a bias the pipeline did not declare, and none of these declare one
    /// (`crate::shadow_draws` has the argument).
    pub(crate) fn draws(&self, pipeline: PipelineHandle) -> Vec<DrawSpec<'_>> {
        self.draws_with(
            pipeline,
            &self.batches.iter().map(|b| &b.push).collect::<Vec<_>>(),
        )
    }

    /// The same batches through one cascade's projection (§6 M15.3).
    ///
    /// A parallel push array rather than a mutated one: the batch's own push is
    /// what the forward and prepass draws borrow, and a frame that rewrote it per
    /// cascade would be rewriting what those two are still pointing at.
    pub(crate) fn shadow_draws(&self, cascade: usize) -> Vec<DrawSpec<'_>> {
        let pushes = self
            .shadow_pushes
            .get(cascade)
            .map_or(&[][..], Vec::as_slice);
        self.draws_with(self.shadow, &pushes.iter().collect::<Vec<_>>())
    }

    fn draws_with<'a>(
        &'a self,
        pipeline: PipelineHandle,
        pushes: &[&'a shader::ScenePush],
    ) -> Vec<DrawSpec<'a>> {
        self.batches
            .iter()
            .zip(pushes)
            .map(|(batch, push)| DrawSpec {
                pipeline,
                depth_bias: None,
                push_constants: bytemuck::bytes_of(*push),
                // Unread: the indirect command carries the counts (§6 M10).
                count: 0,
                index_buffer: Some(batch.indices),
                indirect: Some(Indirect {
                    buffer: self.commands,
                    offset: batch.command_offset,
                }),
            })
            .collect()
    }

    /// Instances staged and draws issued this frame — the "ten thousand
    /// objects, four draws" claim as two numbers rather than an assertion.
    pub(crate) fn counts(&self) -> (usize, usize) {
        (self.staged.len(), self.batches.len())
    }

    /// The prepass + forward pair for `samples`, built on first ask (§6 M21).
    ///
    /// # Errors
    ///
    /// Pipeline creation — including a device that does not do this count.
    pub(crate) fn variant(
        &mut self,
        rhi: &mut impl GpuHost,
        samples: Samples,
    ) -> Result<crate::Variant, RhiError> {
        self.variants
            .get(rhi, samples, |s| [prepass_desc(s), forward_desc(s)])
    }

    /// Release the pipelines, both fallback texels and the per-frame streams.
    pub(crate) fn destroy(self, rhi: &mut impl GpuHost) -> Result<(), RhiError> {
        self.variants.destroy(rhi)?;
        rhi.destroy_pipeline(self.shadow)?;
        rhi.destroy_buffer(self.instances)?;
        rhi.destroy_buffer(self.commands)?;
        rhi.destroy_image(self.flat)?;
        rhi.destroy_image(self.white)
    }
}

/// A draw's sort key: pipeline, then material, then depth, most significant
/// first (§6 M10).
///
/// - **Pipeline** is 4 bits and currently always zero, because this pass has one
///   forward pipeline. The field exists because the *order* is the decision — a
///   pipeline change is the most expensive thing a frame can ask for, so it has
///   to be the outermost sort — and a second pipeline should not re-open the
///   key's shape.
/// - **Material** is 36 bits of the mesh id. A mesh has exactly one material
///   (§4.6), so batching by mesh batches by material, and the id is already a
///   name's hash. Truncation is harmless in a way it would not be in the pack:
///   two colliding meshes sort adjacently and still batch separately, because
///   [`ScenePass::batch`] compares the *full* id.
/// - **Depth** is 24 bits, front to back, because a depth prepass rejects the
///   most fragments when the nearest thing lands first.
fn sort_key(mesh: u64, offset: render::Vec3) -> u64 {
    const PIPELINE: u64 = 0;
    // Quantized in metres/64 and saturated rather than reinterpreting the
    // float's bits: bit-order sorting only works for positive floats, and
    // saturation is what stops something a hundred kilometres out from wrapping
    // around and sorting in front of what is at arm's length.
    let quantized = (offset.length().max(0.0) * 64.0).min(16_777_215.0) as u64;
    (PIPELINE << 60) | ((mesh & 0xf_ffff_ffff) << 24) | quantized
}

/// Position only, depth stored — see `ugly.slang` for why it shares the block.
fn prepass_desc(samples: Samples) -> PipelineDesc<'static> {
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
        samples,
        depth_bias: false,
    }
}

/// The forward pass into the scene attachment, depth tested against the
/// prepass's result — the same arrangement `ugly.forward` uses.
fn forward_desc(samples: Samples) -> PipelineDesc<'static> {
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
        samples,
        depth_bias: false,
    }
}

#[cfg(feature = "hot-reload")]
impl ScenePass {
    /// Rebuild the three `scene.slang` pipelines from a hot recompile (§4.4).
    pub(crate) fn swap_shaders(
        &mut self,
        rhi: &mut impl crate::GpuHost,
        module: &gg_shaders::CompiledModule,
    ) -> Result<(), String> {
        let push = core::mem::size_of::<shader::ScenePush>();
        let (vs_main, fs_main) = crate::hot::pair(module, "vs_main", "fs_main", push)?;
        let (vs_depth, fs_depth) = crate::hot::pair(module, "vs_depth", "fs_depth", push)?;
        let (vs_shadow, _) = crate::hot::pair(module, "vs_shadow", "fs_depth", push)?;
        // Every live count, for the reason `BoxPass::swap_ugly` gives.
        let mut swaps: Vec<(&mut PipelineHandle, PipelineDesc<'_>)> = Vec::new();
        for (samples, variant) in self.variants.each() {
            swaps.push((
                &mut variant.prepass,
                PipelineDesc {
                    vs_spirv: &vs_depth.spirv,
                    vs_entry: &vs_depth.spirv_entry,
                    fs_spirv: &fs_depth.spirv,
                    fs_entry: &fs_depth.spirv_entry,
                    ..prepass_desc(samples)
                },
            ));
            swaps.push((
                &mut variant.forward,
                PipelineDesc {
                    vs_spirv: &vs_main.spirv,
                    vs_entry: &vs_main.spirv_entry,
                    fs_spirv: &fs_main.spirv,
                    fs_entry: &fs_main.spirv_entry,
                    ..forward_desc(samples)
                },
            ));
        }
        swaps.push((
            &mut self.shadow,
            PipelineDesc {
                vs_spirv: &vs_shadow.spirv,
                vs_entry: &vs_shadow.spirv_entry,
                fs_spirv: &fs_depth.spirv,
                fs_entry: &fs_depth.spirv_entry,
                ..shadow_desc()
            },
        ));
        crate::hot::swap_all(rhi, &mut swaps)
    }
}

/// The shadow pass: the same geometry through the sun's projection, depth only
/// and **unbiased** — see [`crate::shadow_draws`] for why the rasterizer does
/// not do it.
fn shadow_desc() -> PipelineDesc<'static> {
    PipelineDesc {
        name: "scene.shadow",
        vs_spirv: shader::VS_SHADOW_SPIRV,
        vs_entry: shader::VS_SHADOW_ENTRY,
        fs_spirv: shader::FS_DEPTH_SPIRV,
        fs_entry: shader::FS_DEPTH_ENTRY,
        push_constant_size: core::mem::size_of::<shader::ScenePush>() as u32,
        color: ColorTarget::None,
        blend: Blend::Off,
        depth: DepthMode::Write,
        samples: Samples::X1,
        depth_bias: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_orders_by_pipeline_then_material_then_depth() {
        let near = render::Vec3::new(0.0, 0.0, -1.0);
        let far = render::Vec3::new(0.0, 0.0, -100.0);
        // Same mesh: the nearer one sorts first, which is what a prepass wants.
        assert!(sort_key(7, near) < sort_key(7, far));
        // Different mesh: material wins over depth, however near the other is.
        assert!(sort_key(7, far) < sort_key(8, near));
    }

    #[test]
    fn the_depth_field_saturates_instead_of_wrapping() {
        // A model a thousand kilometres out must not sort in front of one at
        // arm's length because its quantized depth wrapped into the mesh bits.
        let close = sort_key(1, render::Vec3::new(0.0, 0.0, -1.0));
        let miles = sort_key(1, render::Vec3::splat(1.0e9));
        assert!(close < miles, "{close:#x} vs {miles:#x}");
        assert_eq!(
            miles >> 24,
            close >> 24,
            "and it stayed out of the mesh bits"
        );
    }

    #[test]
    fn the_instance_record_is_what_the_shader_strides_by() {
        // `scene.slang` hardcodes INSTANCE_STRIDE; this is the other half of
        // that agreement, the way the vertex assertions above are.
        assert_eq!(core::mem::size_of::<GpuInstance>(), 128);
        assert_eq!(core::mem::offset_of!(GpuInstance, tint), 64);
        assert_eq!(core::mem::offset_of!(GpuInstance, rotation), 80);
        assert_eq!(core::mem::offset_of!(GpuInstance, offset), 96);
        assert_eq!(core::mem::offset_of!(GpuInstance, scale), 112);
    }
}
