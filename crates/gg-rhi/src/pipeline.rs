//! Graphics pipelines (§4.4, M2): SPIR-V in, dynamic-rendering pipeline out —
//! no render passes, no descriptor sets yet (bindless lands at M4A; M2's one
//! resource path is push constants). The pipeline cache is serialized to disk
//! and every creation is timed and logged, so cold/warm cost is a printed
//! fact instead of a feeling.

use crate::RhiError;
use crate::deletion::{Deferred, DeletionQueue};
use crate::device::Device;
use ash::vk;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// What a graphics pipeline is made of at M2: one vertex + one fragment entry
/// point and a push-constant block. Vertex data comes from the shader itself
/// (`SV_VertexID`); vertex buffers arrive with real geometry (M4A).
pub struct PipelineDesc<'a> {
    /// Debug name (§1.6) — shows up in validation messages and logs.
    pub name: &'a str,
    /// Vertex stage SPIR-V.
    pub vs_spirv: &'a [u8],
    /// Vertex entry point name inside `vs_spirv`.
    pub vs_entry: &'a str,
    /// Fragment stage SPIR-V.
    pub fs_spirv: &'a [u8],
    /// Fragment entry point name inside `fs_spirv`.
    pub fs_entry: &'a str,
    /// Push-constant block size in bytes (0 for none). Draws must pass
    /// exactly this many bytes — checked, not assumed.
    pub push_constant_size: u32,
}

/// An opaque handle to a created pipeline. Plain data on purpose: handles
/// outliving their pipeline (hot reload retires the old one) fail a draw with
/// a precise error instead of dangling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PipelineHandle(u64);

// Copy so a draw can be resolved to an owned value *before* the swapchain
// acquire (see Rhi::render_frame) without holding a borrow of the store across
// it — three handles and a length, so copying is free.
#[derive(Clone, Copy)]
pub(crate) struct PipelineEntry {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub push_constant_size: u32,
}

/// The pipelines plus the disk-backed `vk::PipelineCache` behind them.
pub(crate) struct PipelineStore {
    cache: vk::PipelineCache,
    cache_path: PathBuf,
    next_id: u64,
    entries: BTreeMap<u64, PipelineEntry>,
}

impl PipelineStore {
    /// Create the store, seeding the Vulkan pipeline cache from disk when a
    /// previous run left one (§4.4: cache serialized to disk). One file per
    /// device: the driver would reject a foreign blob anyway (header check),
    /// but rejecting it would also silently discard the *right* device's
    /// warm cache every time the driver under test changes.
    pub fn new(device: &Device) -> Result<Self, RhiError> {
        let device_key: String = device
            .report()
            .chosen
            .to_lowercase()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect();
        let cache_path = std::env::var_os("GG_PIPELINE_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("target/gg-cache"))
            .join(format!("pipeline-cache-{device_key}.bin"));
        let initial = std::fs::read(&cache_path).unwrap_or_default();
        let info = vk::PipelineCacheCreateInfo::default().initial_data(&initial);
        // SAFETY: device is live; initial data is either empty or a previous
        // vkGetPipelineCacheData blob — the driver validates its own header
        // and falls back to empty on mismatch.
        let cache =
            unsafe { device.raw().create_pipeline_cache(&info, None) }.map_err(RhiError::Vk)?;
        device.set_name(cache, "gg.pipeline-cache");
        tracing::info!(
            path = %cache_path.display(),
            loaded_bytes = initial.len(),
            "pipeline cache"
        );
        Ok(Self {
            cache,
            cache_path,
            next_id: 1,
            entries: BTreeMap::new(),
        })
    }

    /// Build a graphics pipeline for `color_format` via dynamic rendering.
    /// Creation is timed and logged (§4.4).
    pub fn create(
        &mut self,
        device: &Device,
        desc: &PipelineDesc<'_>,
        color_format: vk::Format,
    ) -> Result<PipelineHandle, RhiError> {
        let started = std::time::Instant::now();
        let vs = create_shader_module(device, desc.vs_spirv, desc.name)?;
        let fs = match create_shader_module(device, desc.fs_spirv, desc.name) {
            Ok(fs) => fs,
            Err(e) => {
                // SAFETY: vs was created just above and is unused.
                unsafe { device.raw().destroy_shader_module(vs, None) };
                return Err(e);
            }
        };

        let result = self.create_with_modules(device, desc, color_format, vs, fs);
        // SAFETY: pipeline creation retains no reference to the modules.
        unsafe {
            device.raw().destroy_shader_module(vs, None);
            device.raw().destroy_shader_module(fs, None);
        }
        let handle = result?;
        tracing::info!(
            name = desc.name,
            ms = started.elapsed().as_secs_f64() * 1e3,
            "pipeline created"
        );
        Ok(handle)
    }

    fn create_with_modules(
        &mut self,
        device: &Device,
        desc: &PipelineDesc<'_>,
        color_format: vk::Format,
        vs: vk::ShaderModule,
        fs: vk::ShaderModule,
    ) -> Result<PipelineHandle, RhiError> {
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(desc.push_constant_size)];
        let mut layout_info = vk::PipelineLayoutCreateInfo::default();
        if desc.push_constant_size > 0 {
            layout_info = layout_info.push_constant_ranges(&push_ranges);
        }
        // SAFETY: device is live; ranges outlive the call.
        let layout = unsafe { device.raw().create_pipeline_layout(&layout_info, None) }
            .map_err(RhiError::Vk)?;
        device.set_name(layout, &format!("gg.pipeline.{}.layout", desc.name));

        let vs_entry = std::ffi::CString::new(desc.vs_entry)
            .map_err(|e| RhiError::Loader(format!("vs entry name: {e}")))?;
        let fs_entry = std::ffi::CString::new(desc.fs_entry)
            .map_err(|e| RhiError::Loader(format!("fs entry name: {e}")))?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vs)
                .name(&vs_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(fs)
                .name(&fs_entry),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let color_formats = [color_format];
        let mut rendering =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&blend)
            .dynamic_state(&dynamic)
            .layout(layout)
            .push_next(&mut rendering);

        // SAFETY: all pointed-to state outlives the call; cache and layout are
        // live objects of this device.
        let pipelines = unsafe {
            device
                .raw()
                .create_graphics_pipelines(self.cache, &[info], None)
        };
        let pipeline = match pipelines {
            Ok(p) => p
                .into_iter()
                .next()
                .ok_or_else(|| RhiError::Loader("pipeline creation returned nothing".into()))?,
            Err((_, e)) => {
                // SAFETY: layout was created above and is now orphaned.
                unsafe { device.raw().destroy_pipeline_layout(layout, None) };
                return Err(RhiError::Vk(e));
            }
        };
        device.set_name(pipeline, &format!("gg.pipeline.{}", desc.name));

        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(
            id,
            PipelineEntry {
                pipeline,
                layout,
                push_constant_size: desc.push_constant_size,
            },
        );
        Ok(PipelineHandle(id))
    }

    pub fn get(&self, handle: PipelineHandle) -> Result<&PipelineEntry, RhiError> {
        self.entries.get(&handle.0).ok_or_else(|| {
            RhiError::Loader(format!(
                "pipeline handle {} is not live (destroyed by hot reload?)",
                handle.0
            ))
        })
    }

    /// Retire a pipeline behind the timeline: destroyed once the GPU is
    /// provably past `after_value` — never mid-frame (§4.4 hot path contract).
    pub fn retire(
        &mut self,
        handle: PipelineHandle,
        deletions: &mut DeletionQueue,
        after_value: u64,
    ) -> Result<(), RhiError> {
        let entry = self.entries.remove(&handle.0).ok_or_else(|| {
            RhiError::Loader(format!("pipeline handle {} retired twice", handle.0))
        })?;
        deletions.defer(after_value, Deferred::Pipeline(entry.pipeline));
        deletions.defer(after_value, Deferred::PipelineLayout(entry.layout));
        Ok(())
    }

    /// Destroy a pipeline now. Caller proves nothing is in flight (the
    /// offscreen path is synchronous; the swapchain path uses [`Self::retire`]).
    pub fn remove_now(&mut self, device: &Device, handle: PipelineHandle) -> Result<(), RhiError> {
        let entry = self.entries.remove(&handle.0).ok_or_else(|| {
            RhiError::Loader(format!("pipeline handle {} destroyed twice", handle.0))
        })?;
        // SAFETY: caller contract — no submitted work references the pipeline.
        unsafe {
            device.raw().destroy_pipeline(entry.pipeline, None);
            device.raw().destroy_pipeline_layout(entry.layout, None);
        }
        Ok(())
    }

    /// Teardown: persist the cache, destroy everything. Caller waited idle.
    pub fn destroy(&mut self, device: &Device) {
        for (_, entry) in std::mem::take(&mut self.entries) {
            // SAFETY: handles belong to this device; GPU idle per contract.
            unsafe {
                device.raw().destroy_pipeline(entry.pipeline, None);
                device.raw().destroy_pipeline_layout(entry.layout, None);
            }
        }
        // SAFETY: cache is live until the destroy below.
        match unsafe { device.raw().get_pipeline_cache_data(self.cache) } {
            Ok(data) => {
                let write = self
                    .cache_path
                    .parent()
                    .map(std::fs::create_dir_all)
                    .transpose()
                    .and_then(|_| std::fs::write(&self.cache_path, &data).map(Some));
                match write {
                    Ok(_) => tracing::info!(
                        path = %self.cache_path.display(),
                        bytes = data.len(),
                        "pipeline cache saved"
                    ),
                    Err(e) => tracing::warn!("pipeline cache not saved: {e}"),
                }
            }
            Err(e) => tracing::warn!("pipeline cache data unavailable: {e:?}"),
        }
        // SAFETY: cache belongs to this device; no creation is in flight.
        unsafe { device.raw().destroy_pipeline_cache(self.cache, None) };
    }
}

/// Record one draw inside an active dynamic-rendering pass: full-target
/// viewport/scissor, push constants, `vk::CmdDraw`. Shared by the swapchain
/// and offscreen paths so a draw means the same thing everywhere.
///
/// # Safety
/// `cmd` must be recording inside `cmd_begin_rendering`, and `entry` must be
/// a live pipeline compatible with the pass's color format.
pub(crate) unsafe fn record_draw(
    device: &Device,
    cmd: vk::CommandBuffer,
    extent: (u32, u32),
    entry: &PipelineEntry,
    push_constants: &[u8],
    vertex_count: u32,
) -> Result<(), RhiError> {
    if push_constants.len() != entry.push_constant_size as usize {
        return Err(RhiError::Loader(format!(
            "draw passed {} push-constant bytes; the pipeline declares {} (§4.4: layouts are \
             checked, not assumed)",
            push_constants.len(),
            entry.push_constant_size
        )));
    }
    let device = device.raw();
    let viewport = [vk::Viewport::default()
        .width(extent.0 as f32)
        .height(extent.1 as f32)
        .min_depth(0.0)
        .max_depth(1.0)];
    let scissor = [vk::Rect2D {
        offset: vk::Offset2D::default(),
        extent: vk::Extent2D {
            width: extent.0,
            height: extent.1,
        },
    }];
    // SAFETY: caller contract — recording command buffer, live pipeline.
    unsafe {
        device.cmd_set_viewport(cmd, 0, &viewport);
        device.cmd_set_scissor(cmd, 0, &scissor);
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, entry.pipeline);
        if !push_constants.is_empty() {
            device.cmd_push_constants(
                cmd,
                entry.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                push_constants,
            );
        }
        device.cmd_draw(cmd, vertex_count, 1, 0, 0);
    }
    Ok(())
}

fn create_shader_module(
    device: &Device,
    spirv: &[u8],
    name: &str,
) -> Result<vk::ShaderModule, RhiError> {
    if !spirv.len().is_multiple_of(4) || spirv.len() < 4 {
        return Err(RhiError::Loader(format!(
            "shader `{name}`: SPIR-V length {} is not a multiple of 4",
            spirv.len()
        )));
    }
    // SPIR-V is a word stream; the bytes on disk are little-endian words.
    let words: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if words.first() != Some(&0x0723_0203) {
        return Err(RhiError::Loader(format!(
            "shader `{name}`: bytes are not SPIR-V (bad magic)"
        )));
    }
    let info = vk::ShaderModuleCreateInfo::default().code(&words);
    // SAFETY: device is live; code outlives the call.
    unsafe { device.raw().create_shader_module(&info, None) }.map_err(RhiError::Vk)
}
