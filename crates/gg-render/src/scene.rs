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
    PipelineDesc, PipelineHandle, RhiError, Sampler, Samples, TextureIndex, Viewport,
};

use crate::content::Content;
use crate::cull;
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

impl GpuInstance {
    /// The camera-relative centre, as the cull wants it.
    fn offset_vec(&self) -> render::Vec3 {
        render::Vec3::new(self.offset[0], self.offset[1], self.offset[2])
    }
}

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

/// Indirect commands one frame may hold: the batches, plus one per surviving
/// (batch, shadow view) pair (§6 M32).
///
/// Four times the batch cap rather than `MAX_BATCHES * (1 + MAX_CASCADES +
/// MAX_LAMPS * FACES)`, which would be 56× for a worst case the cull exists to
/// prevent — a frame where every batch reaches every one of four cascades and
/// forty-eight lamp faces has already lost. [`ScenePass::cull`] degrades into
/// the uncompacted draw rather than dropping one, so the ceiling costs frame
/// time and never a shadow.
const MAX_COMMANDS: usize = MAX_BATCHES * 4;

const INSTANCE_SLOT_BYTES: u64 = (MAX_INSTANCES * core::mem::size_of::<GpuInstance>()) as u64;
const COMMAND_SLOT_BYTES: u64 = (MAX_COMMANDS * core::mem::size_of::<IndirectCommand>()) as u64;

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
    /// Whether the *view* frustum admitted this instance, or only the swept
    /// caster volume did (§6 M34). Sorted on ahead of [`Keyed::key`], so the
    /// batches a main pass draws are a prefix of the batch list.
    visible: bool,
    /// Index into [`ScenePass::built`], which stays in extract order.
    at: u32,
    /// The instance's world-space bounding radius, carried through the sort so
    /// that batching can union it without a second pass over extract's array.
    radius: f32,
}

/// One draw: a run of instances that share a mesh.
struct Batch {
    push: shader::ScenePush,
    indices: BufferHandle,
    /// Byte offset of this batch's command, absolute within the buffer.
    command_offset: u64,
    /// The sphere bounding every instance in the run, camera-relative — the
    /// bounds §6 M15.3 said a batch did not have, and without which a shadow
    /// view had nothing to reject (§6 M32).
    ///
    /// A sphere about the *mean* of the instance centres rather than about the
    /// first: a run of a thousand shrubs down one edge of a level bounds to a
    /// sphere centred on the row, where growing one from whichever shrub sorted
    /// first would bound to nearly twice the radius.
    centre: render::Vec3,
    radius: f32,
    /// How many instances the run holds. Where they *sit* is the batch's own
    /// `push.instance_base` and its chunks' ranges; a third copy of the same
    /// offset was one to keep in step for nothing.
    count: u32,
    /// The mesh's index count, so a compacted draw can write its own command
    /// without reading the batch's back out of the buffer.
    index_count: u32,
    /// This batch's slice of [`ScenePass::chunks`].
    chunk_first: u32,
    chunk_count: u32,
}

/// Instances per chunk — the second level of the cull, under the batch and over
/// the instance (§6 M32).
///
/// One sphere over a whole batch is too coarse to *do* anything on its own: a
/// mesh scattered across a level bounds to the level, so every view says
/// `Partial` and pays for every instance one at a time. That cost is invisible
/// on a software rasterizer, where the draws it saves dwarf it, and plainly
/// visible on a real one — the 4090 ran the cull at a net *loss* until this
/// existed, because 3 485 instances against 28 views is a hundred thousand
/// sphere tests a frame.
///
/// The instances are in draw order, which is depth order within a mesh, so a
/// chunk is a shell at roughly one distance from the eye rather than a compact
/// blob. That is weaker than a spatial sort would give and is still enough: a
/// lamp reaching six metres rejects every shell that is not at its own distance,
/// which is most of them.
const CHUNK: usize = 128;

/// One batch through one shadow view: the same geometry, its own instances.
///
/// Separate from [`Batch`] rather than a mutated copy of it for the reason §6
/// M15.3 gave for the cascade push array — the forward and prepass draws borrow
/// the batch's own push, and a frame rewriting it per view would be rewriting
/// what those two still point at.
struct ViewDraw {
    push: shader::ScenePush,
    indices: BufferHandle,
    command_offset: u64,
}

/// The pack pass: two pipelines, one fallback texel, this frame's draws.
pub(crate) struct ScenePass {
    /// Prepass + forward, per sample count (§6 M21).
    variants: crate::Variants,
    shadow: PipelineHandle,
    /// One probe face's shading pass (§6 M36) — the forward pipeline writing
    /// depth, because a probe face has no prepass in front of it.
    probe: PipelineHandle,
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
    /// How many of the above the *view* frustum admits — the prefix
    /// [`ScenePass::draws`] returns (§6 M34). The rest exist for the shadow
    /// views, which walk all of them.
    visible_batches: usize,
    /// Each instance's bounding sphere, in `staged` order and covering the main
    /// region only — what [`ScenePass::cull`] tests when a batch straddles a
    /// view. Compacted copies are appended past the end of this and never read
    /// back through it.
    spheres: Vec<(render::Vec3, f32)>,
    /// `(centre, radius, first, count)` per chunk, in batch order.
    chunks: Vec<(render::Vec3, f32, u32, u32)>,
    /// One view's surviving instance indices, reused across views. Held so the
    /// fit test runs *once* per instance and the decision to compact is taken
    /// after the count is known rather than during the copy.
    survivors: Vec<u32>,
    /// The draws that reach each cascade, one list per cascade (§6 M15.3).
    shadow_views: Vec<Vec<ViewDraw>>,
    /// The same per lamp face (§6 M31), flat as `lamp * 6 + face`.
    lamp_views: Vec<Vec<ViewDraw>>,
    /// And per gathering probe face (§6 M36), flat as `slot * 6 + face`.
    probe_views: Vec<Vec<ViewDraw>>,
    /// What the cull came to this frame, for the overlay and `gg-tools lamps` —
    /// a cull nothing reports is a cull nobody can tell stopped working.
    cull: cull::ShadowCull,
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
            visible_batches: 0,
            spheres: Vec::new(),
            chunks: Vec::new(),
            survivors: Vec::new(),
            written: Vec::new(),
            shadow_views: Vec::new(),
            lamp_views: Vec::new(),
            probe_views: Vec::new(),
            probe: rhi.create_pipeline(&probe_desc())?,
            cull: cull::ShadowCull::default(),
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
        sun: Option<&crate::lighting::Sun>,
        lamps: &crate::lamp::Lamps,
        probes: &[crate::probe::Gather],
    ) -> Result<(), RhiError> {
        self.frame = frame;
        self.keyed.clear();
        self.built.clear();
        self.staged.clear();
        self.spheres.clear();
        self.chunks.clear();
        self.batches.clear();
        self.visible_batches = 0;
        self.written.clear();
        self.cull = cull::ShadowCull::default();
        let Some(content) = content else {
            self.shadow_views.clear();
            self.probe_views.clear();
            self.lamp_views.clear();
            return Ok(());
        };
        let view_projection = view.view_projection(extent);
        {
            gg_core::zone!("scene.key");
            let seen = extracted.visible_models().len();
            for (at, instance) in extracted.models.iter().enumerate() {
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
                    visible: at < seen,
                    at: (self.built.len() - 1) as u32,
                    radius: instance.radius,
                });
            }
        }
        {
            // Stable: equal keys keep extract's order, which is a function of
            // world state and nothing else (§4.2).
            gg_core::zone!("scene.sort");
            // Visibility outermost (§6 M34), so `batches[..visible_batches]` is
            // exactly what the main pass draws. Stable, so equal keys still keep
            // extract's order, which is a function of world state alone (§4.2).
            self.keyed.sort_by_key(|k| (!k.visible, k.key));
        }
        {
            gg_core::zone!("scene.batch");
            self.batch(content);
        }
        {
            gg_core::zone!("scene.cull");
            self.cull(sun, lamps, probes);
        }
        gg_core::zone!("scene.stage");
        self.stage(rhi, slot)
    }

    /// Turn the sorted list into runs of equal mesh, one draw each.
    fn batch(&mut self, content: &Content) {
        let mut start = 0;
        while start < self.keyed.len() {
            let (mesh, visible) = (self.keyed[start].mesh, self.keyed[start].visible);
            let mut end = start;
            // Broken on visibility as well as on mesh: the two halves must not
            // merge into one run, or the main pass's prefix would contain
            // instances the view frustum rejected (§6 M34). One mesh present in
            // both halves is two batches, which is the cost of the split and is
            // paid only by a frame with a sun.
            while end < self.keyed.len()
                && self.keyed[end].mesh == mesh
                && self.keyed[end].visible == visible
            {
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
            let first = self.staged.len() as u32;
            let chunk_first = self.chunks.len() as u32;
            let sphere = |keyed: &Keyed, built: &[GpuInstance]| {
                (built[keyed.at as usize].offset_vec(), keyed.radius)
            };
            // Chunk bounds first, then the batch's as the union of theirs — one
            // walk of the instances rather than one per level.
            for base in (start..end).step_by(CHUNK) {
                let stop = (base + CHUNK).min(end);
                let (centre, radius) = bound(
                    self.keyed[base..stop]
                        .iter()
                        .map(|k| sphere(k, &self.built)),
                );
                self.chunks.push((
                    centre,
                    radius,
                    first + (base - start) as u32,
                    (stop - base) as u32,
                ));
            }
            let chunk_count = self.chunks.len() as u32 - chunk_first;
            let (centre, radius) = bound(
                self.chunks[chunk_first as usize..]
                    .iter()
                    .map(|&(c, r, ..)| (c, r)),
            );
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
                centre,
                radius,
                count: (end - start) as u32,
                index_count: resident.index_count,
                chunk_first,
                chunk_count,
            });
            // The batch list is sorted visible-first, so the running high-water
            // mark *is* the prefix boundary (§6 M34).
            if visible {
                self.visible_batches = self.batches.len();
            }
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
                self.spheres
                    .push((instance.offset_vec(), self.keyed[slot].radius));
            }
            start = end;
        }
    }

    /// Build each shadow view's draw list out of the batches that reach it
    /// (§6 M32).
    ///
    /// This is what §6 M15.3 deferred and §6 M31 inherited with a six-times
    /// multiplier: before it, every batch was drawn into every cascade and every
    /// lamp face, because a batch had no bounds to reject it by. Now it has one,
    /// and each (batch, view) pair takes one of three roads —
    ///
    /// - **Outside**: no draw at all, which is the case that pays.
    /// - **Inside**: the batch's own instance range and its own command, reused
    ///   untouched. No walk, no copy, no second command — a cascade holding the
    ///   whole level costs exactly what it cost before.
    /// - **Partial**: the survivors compacted into a fresh range with a command
    ///   of their own. [`ScenePush::instance_base`](shader::ScenePush) already
    ///   existed for the batch's own offset, so a compacted range needs no
    ///   shader change to draw through.
    ///
    /// Runs before [`ScenePass::stage`] because it appends to both streams that
    /// call writes.
    fn cull(
        &mut self,
        sun: Option<&crate::lighting::Sun>,
        lamps: &crate::lamp::Lamps,
        batch: &[crate::probe::Gather],
    ) {
        // Taken and put back so the per-view list being filled is not a borrow
        // of `self` while the compaction writes `self.staged`.
        let mut shadow = core::mem::take(&mut self.shadow_views);
        let cascades = sun.map_or(&[][..], crate::lighting::Sun::cascades);
        shadow.resize_with(cascades.len(), Vec::new);
        shadow.truncate(cascades.len());
        if let Some(sun) = sun {
            let basis = cull::Basis::of(sun);
            for (index, (draws, cascade)) in shadow.iter_mut().zip(cascades).enumerate() {
                let view = cull::View::Slab { cascade, basis };
                self.cull_view(view, index as u32, draws);
            }
        }
        self.shadow_views = shadow;

        let mut lamp_views = core::mem::take(&mut self.lamp_views);
        let faces = lamps.lamps().len() * crate::lamp::FACES;
        lamp_views.resize_with(faces, Vec::new);
        lamp_views.truncate(faces);
        for (index, draws) in lamp_views.iter_mut().enumerate() {
            let lamp = &lamps.lamps()[index / crate::lamp::FACES];
            let view = cull::View::Face {
                lamp,
                face: index % crate::lamp::FACES,
            };
            // The index space is the shader's, cascades first — `LAMP_VIEW_BASE`
            // in `pbr.slang` is the other half of this sentence.
            let shadow_view = (crate::lighting::MAX_CASCADES + index) as u32;
            self.cull_view(view, shadow_view, draws);
        }
        self.lamp_views = lamp_views;

        // The probes this frame is gathering, one segment further along the
        // same index space (§6 M36). Six views a probe and no reach to reject
        // by, so this is the cull that pays most per view.
        let mut probe_views = core::mem::take(&mut self.probe_views);
        let probes = batch.len() * crate::probe::PROBE_FACES;
        probe_views.resize_with(probes, Vec::new);
        probe_views.truncate(probes);
        for (index, draws) in probe_views.iter_mut().enumerate() {
            let view = cull::View::Probe {
                gather: &batch[index / crate::probe::PROBE_FACES],
                face: index % crate::probe::PROBE_FACES,
            };
            self.cull_view(view, (crate::probe::VIEW_BASE + index) as u32, draws);
        }
        self.probe_views = probe_views;
        self.cull.views = (cascades.len() + faces + probes) as u32;
    }

    /// One view's draws, appending whatever compaction it needs to `staged` and
    /// `written`.
    fn cull_view(&mut self, view: cull::View<'_>, shadow_view: u32, draws: &mut Vec<ViewDraw>) {
        draws.clear();
        for index in 0..self.batches.len() {
            let batch = &self.batches[index];
            let (centre, radius, count) = (batch.centre, batch.radius, batch.count);
            let whole = ViewDraw {
                push: shader::ScenePush {
                    shadow_view,
                    ..batch.push
                },
                indices: batch.indices,
                command_offset: batch.command_offset,
            };
            match view.fit(centre, radius) {
                cull::Fit::Outside => {
                    self.cull.rejected += count;
                    self.cull.dropped += 1;
                }
                cull::Fit::Inside => {
                    self.cull.drawn += count;
                    draws.push(whole);
                }
                // Either budget exhausted: draw the batch whole. A frame that
                // ran out of room to be precise renders the right picture and
                // costs what it used to, which is the only degradation this pass
                // is allowed — the alternative drops a shadow.
                cull::Fit::Partial
                    if self.staged.len() + count as usize > MAX_INSTANCES
                        || self.written.len() == MAX_COMMANDS =>
                {
                    tracing::warn!(
                        instances = self.staged.len(),
                        commands = self.written.len(),
                        instance_cap = MAX_INSTANCES,
                        command_cap = MAX_COMMANDS,
                        "shadow cull out of room — drawing a batch uncompacted rather than \
                         dropping it; raise the budget"
                    );
                    self.cull.drawn += count;
                    draws.push(whole);
                }
                cull::Fit::Partial => {
                    // Which of them survive, before deciding whether moving them
                    // is worth it. One fit test per instance either way.
                    self.survivors.clear();
                    let chunks = batch.chunk_first as usize
                        ..(batch.chunk_first + batch.chunk_count) as usize;
                    for chunk in chunks {
                        let (centre, radius, first, count) = self.chunks[chunk];
                        let range = first..first + count;
                        match view.fit(centre, radius) {
                            // The level this exists for: a shell of the batch
                            // that this view cannot see costs one test, not a
                            // hundred and twenty-eight.
                            cull::Fit::Outside => continue,
                            cull::Fit::Inside => self.survivors.extend(range),
                            cull::Fit::Partial => {
                                for at in range {
                                    let (centre, radius) = self.spheres[at as usize];
                                    if view.fit(centre, radius) != cull::Fit::Outside {
                                        self.survivors.push(at);
                                    }
                                }
                            }
                        }
                    }
                    let kept = self.survivors.len() as u32;
                    if kept == 0 {
                        self.cull.rejected += count;
                        self.cull.dropped += 1;
                        continue;
                    }
                    // Keeping nearly all of it: draw the batch whole. Copying
                    // eight hundred instances to avoid rasterizing twenty is a
                    // loss twice over — the copy costs more than the draw saved,
                    // and the compacted range costs instance budget the *next*
                    // view then has to do without. A batch spanning a level is
                    // never wholly inside a cascade, so without this rule the
                    // common case is a full copy per view that rejects nothing.
                    if kept * 4 >= count * 3 {
                        self.cull.drawn += count;
                        draws.push(whole);
                        continue;
                    }
                    let first = self.staged.len();
                    for at in 0..self.survivors.len() {
                        let instance = self.staged[self.survivors[at] as usize];
                        self.staged.push(instance);
                    }
                    self.cull.rejected += count - kept;
                    self.cull.drawn += kept;
                    let batch = &self.batches[index];
                    let command_offset =
                        (self.written.len() * core::mem::size_of::<IndirectCommand>()) as u64;
                    self.written.push(IndirectCommand {
                        index_count: batch.index_count,
                        instance_count: kept,
                        ..Default::default()
                    });
                    draws.push(ViewDraw {
                        push: shader::ScenePush {
                            shadow_view,
                            instance_base: first as u32,
                            ..batch.push
                        },
                        indices: batch.indices,
                        command_offset,
                    });
                }
            }
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
        // The view draws were built before this call and carry the same two
        // placeholders, so they take the same patch. Never before it: a copy
        // taken while `instances` was still zero would draw the shadow pass
        // through address zero.
        let views = self
            .shadow_views
            .iter_mut()
            .chain(self.lamp_views.iter_mut())
            .chain(self.probe_views.iter_mut());
        for draw in views.flatten() {
            draw.push.instances = base;
            draw.command_offset += command_offset;
        }
        Ok(())
    }

    /// This frame's mesh draws through `pipeline`.
    ///
    /// No `depth_bias` on any of them, the shadow one included: the RHI refuses
    /// a bias the pipeline did not declare, and none of these declare one
    /// (`crate::shadow_draws` has the argument).
    pub(crate) fn draws(&self, pipeline: PipelineHandle) -> Vec<DrawSpec<'_>> {
        self.batches[..self.visible_batches]
            .iter()
            .map(|batch| self.spec(pipeline, &batch.push, batch.indices, batch.command_offset))
            .collect()
    }

    /// The batches that reach cascade `cascade` (§6 M15.3, culled at §6 M32).
    pub(crate) fn shadow_draws(&self, cascade: usize) -> Vec<DrawSpec<'_>> {
        let draws = self
            .shadow_views
            .get(cascade)
            .map_or(&[][..], Vec::as_slice);
        draws
            .iter()
            .map(|d| self.spec(self.shadow, &d.push, d.indices, d.command_offset))
            .collect()
    }

    /// The batches that reach one lamp face, landing in that face's tile.
    ///
    /// The tile is the *draw's* rectangle rather than the pass's: six faces of
    /// eight lamps share one atlas and one pass, so a pass-level viewport could
    /// only ever have named one of them (§6 M31).
    pub(crate) fn lamp_draws(&self, face: usize, tile: Viewport) -> Vec<DrawSpec<'_>> {
        let draws = self.lamp_views.get(face).map_or(&[][..], Vec::as_slice);
        draws
            .iter()
            .map(|d| {
                let mut spec = self.spec(self.shadow, &d.push, d.indices, d.command_offset);
                spec.viewport = Some(tile);
                spec
            })
            .collect()
    }

    /// The batches that reach one probe face, landing in that face's tile —
    /// [`ScenePass::lamp_draws`]'s shape, through the probe pipeline because
    /// this one shades rather than only recording depth.
    pub(crate) fn probe_draws(&self, face: usize, tile: Viewport) -> Vec<DrawSpec<'_>> {
        let draws = self.probe_views.get(face).map_or(&[][..], Vec::as_slice);
        draws
            .iter()
            .map(|d| {
                let mut spec = self.spec(self.probe, &d.push, d.indices, d.command_offset);
                spec.viewport = Some(tile);
                spec
            })
            .collect()
    }

    fn spec<'a>(
        &self,
        pipeline: PipelineHandle,
        push: &'a shader::ScenePush,
        indices: BufferHandle,
        command_offset: u64,
    ) -> DrawSpec<'a> {
        DrawSpec {
            pipeline,
            depth_bias: None,
            viewport: None,
            push_constants: bytemuck::bytes_of(push),
            // Unread: the indirect command carries the counts (§6 M10).
            count: 0,
            index_buffer: Some(indices),
            indirect: Some(Indirect {
                buffer: self.commands,
                offset: command_offset,
            }),
        }
    }

    /// Instances staged and draws issued this frame — the "ten thousand
    /// objects, four draws" claim as two numbers rather than an assertion.
    ///
    /// The instance count is the *main* region's: what the camera draws, not
    /// what the shadow views compacted behind it, so the number still answers
    /// the question the overlay asks it.
    pub(crate) fn counts(&self) -> (usize, usize) {
        (self.spheres.len(), self.batches.len())
    }

    /// What this pass's shadow cull came to (§6 M32).
    pub(crate) fn shadow_cull(&self) -> cull::ShadowCull {
        self.cull
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
        rhi.destroy_pipeline(self.probe)?;
        rhi.destroy_buffer(self.instances)?;
        rhi.destroy_buffer(self.commands)?;
        rhi.destroy_image(self.flat)?;
        rhi.destroy_image(self.white)
    }
}

/// The sphere bounding a set of spheres: the mean of their centres, then the
/// furthest surface from it.
///
/// Centred on the **mean** rather than on whichever came first, which is what
/// keeps a long row of shrubs bounded by a sphere over the row instead of one
/// over twice its length. Two passes, over data already in cache.
fn bound(spheres: impl Iterator<Item = (render::Vec3, f32)> + Clone) -> (render::Vec3, f32) {
    let mut centre = render::Vec3::ZERO;
    let mut count = 0.0_f32;
    for (at, _) in spheres.clone() {
        centre += at;
        count += 1.0;
    }
    if count == 0.0 {
        return (render::Vec3::ZERO, 0.0);
    }
    centre /= count;
    let mut radius: f32 = 0.0;
    for (at, r) in spheres {
        radius = radius.max(at.distance(centre) + r);
    }
    (centre, radius)
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

/// One probe face (§6 M36): pack geometry shaded into a tile of the radiance
/// atlas. `crate::probe_desc`'s twin, and the reason the field sees a pack at
/// all — see it for why depth is written rather than tested.
fn probe_desc() -> PipelineDesc<'static> {
    PipelineDesc {
        name: "scene.probe",
        vs_spirv: shader::VS_PROBE_SPIRV,
        vs_entry: shader::VS_PROBE_ENTRY,
        fs_spirv: shader::FS_PROBE_SPIRV,
        fs_entry: shader::FS_PROBE_ENTRY,
        push_constant_size: core::mem::size_of::<shader::ScenePush>() as u32,
        color: ColorTarget::Format(crate::SCENE_FORMAT),
        blend: Blend::Off,
        depth: DepthMode::Write,
        samples: Samples::X1,
        depth_bias: false,
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
