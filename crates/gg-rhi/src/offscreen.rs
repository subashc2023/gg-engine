//! Offscreen rendering (§4.10 v0): no window, no surface, no swapchain — an
//! RGBA8-sRGB target image, a host-visible readback buffer, and a single-shot
//! render → readback path. This is gg-golden's whole GPU footprint, which is
//! what lets the harness be headless *by linkage* (§5 gate 7: no winit
//! symbols), not just by configuration.

use crate::device::Device;
use crate::instance::{Instance, validation_message_count};
use crate::pipeline::{PipelineDesc, PipelineHandle, PipelineStore, record_draw};
use crate::{DrawSpec, RhiError, ShutdownReport};
use ash::vk;

/// Bytes per pixel of [`OffscreenRhi`]'s fixed RGBA8 target.
const BYTES_PER_PIXEL: usize = 4;

/// A window-less rendering context: bring-up without a display, render into
/// an offscreen image, read the pixels back. Same accounting rules as [`Rhi`]
/// (§4.3): every byte allocated is tracked, shutdown reports leaks and
/// validation messages.
///
/// [`Rhi`]: crate::Rhi
pub struct OffscreenRhi {
    dead: bool,
    timeline_value: u64,
    extent: (u32, u32),
    format: vk::Format,
    image: vk::Image,
    image_view: vk::ImageView,
    image_alloc: Option<gpu_allocator::vulkan::Allocation>,
    readback: vk::Buffer,
    readback_alloc: Option<gpu_allocator::vulkan::Allocation>,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    pipelines: PipelineStore,
    device: Device,
    instance: Instance,
}

impl OffscreenRhi {
    /// Bring up a device with no display attached and an offscreen RGBA8-sRGB
    /// target of `extent` pixels.
    pub fn new(extent: (u32, u32)) -> Result<Self, RhiError> {
        if extent.0 == 0 || extent.1 == 0 {
            return Err(RhiError::Loader("offscreen extent must be nonzero".into()));
        }
        let mut instance = Instance::new(None)?;
        let device = Device::new(&instance, None);
        let mut device = match device {
            Ok(d) => d,
            Err(e) => {
                instance.destroy();
                return Err(e);
            }
        };

        match Self::create_resources(&mut device, extent) {
            Ok((
                image,
                image_alloc,
                image_view,
                readback,
                readback_alloc,
                pool,
                cmd,
                pipelines,
            )) => Ok(Self {
                dead: false,
                timeline_value: 0,
                extent,
                format: vk::Format::R8G8B8A8_SRGB,
                image,
                image_view,
                image_alloc: Some(image_alloc),
                readback,
                readback_alloc: Some(readback_alloc),
                pool,
                cmd,
                pipelines,
                device,
                instance,
            }),
            Err(e) => {
                device.destroy();
                instance.destroy();
                Err(e)
            }
        }
    }

    #[expect(clippy::type_complexity, reason = "internal one-shot bring-up tuple")]
    fn create_resources(
        device: &mut Device,
        extent: (u32, u32),
    ) -> Result<
        (
            vk::Image,
            gpu_allocator::vulkan::Allocation,
            vk::ImageView,
            vk::Buffer,
            gpu_allocator::vulkan::Allocation,
            vk::CommandPool,
            vk::CommandBuffer,
            PipelineStore,
        ),
        RhiError,
    > {
        let format = vk::Format::R8G8B8A8_SRGB;
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.0,
                height: extent.1,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: device is live; info valid.
        let image =
            unsafe { device.raw().create_image(&image_info, None) }.map_err(RhiError::Vk)?;
        device.set_name(image, "gg.offscreen.target");
        // SAFETY: image is live.
        let requirements = unsafe { device.raw().get_image_memory_requirements(image) };
        let image_alloc = device.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "gg.offscreen.target",
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;
        // SAFETY: fresh image, fresh memory, offsets from the allocator.
        unsafe {
            device
                .raw()
                .bind_image_memory(image, image_alloc.memory(), image_alloc.offset())
        }
        .map_err(RhiError::Vk)?;

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );
        // SAFETY: image is live and bound.
        let image_view =
            unsafe { device.raw().create_image_view(&view_info, None) }.map_err(RhiError::Vk)?;
        device.set_name(image_view, "gg.offscreen.target.view");

        let readback_size = extent.0 as u64 * extent.1 as u64 * BYTES_PER_PIXEL as u64;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(readback_size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: device is live; info valid.
        let readback =
            unsafe { device.raw().create_buffer(&buffer_info, None) }.map_err(RhiError::Vk)?;
        device.set_name(readback, "gg.offscreen.readback");
        // SAFETY: buffer is live.
        let buf_requirements = unsafe { device.raw().get_buffer_memory_requirements(readback) };
        let readback_alloc = device.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: "gg.offscreen.readback",
            requirements: buf_requirements,
            location: gpu_allocator::MemoryLocation::GpuToCpu,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;
        // SAFETY: fresh buffer, fresh memory.
        unsafe {
            device.raw().bind_buffer_memory(
                readback,
                readback_alloc.memory(),
                readback_alloc.offset(),
            )
        }
        .map_err(RhiError::Vk)?;

        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(device.graphics.family);
        // SAFETY: device is live.
        let pool =
            unsafe { device.raw().create_command_pool(&pool_info, None) }.map_err(RhiError::Vk)?;
        device.set_name(pool, "gg.offscreen.pool");
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: pool is live.
        let cmd = unsafe { device.raw().allocate_command_buffers(&alloc_info) }
            .map_err(RhiError::Vk)?
            .into_iter()
            .next()
            .ok_or_else(|| RhiError::Loader("no command buffer allocated".into()))?;
        device.set_name(cmd, "gg.offscreen.cmd");

        let pipelines = PipelineStore::new(device)?;
        Ok((
            image,
            image_alloc,
            image_view,
            readback,
            readback_alloc,
            pool,
            cmd,
            pipelines,
        ))
    }

    /// The device-selection report (§4.3).
    pub fn device_report(&self) -> &crate::DeviceReport {
        self.device.report()
    }

    /// Target extent in pixels.
    pub fn extent(&self) -> (u32, u32) {
        self.extent
    }

    /// Create a graphics pipeline targeting the offscreen format.
    pub fn create_pipeline(&mut self, desc: &PipelineDesc<'_>) -> Result<PipelineHandle, RhiError> {
        self.pipelines.create(&self.device, desc, self.format)
    }

    /// Destroy a pipeline. Offscreen renders are synchronous ([`Self::render`]
    /// waits for completion before returning), so nothing can be in flight
    /// and destruction is immediate.
    pub fn destroy_pipeline(&mut self, handle: PipelineHandle) -> Result<(), RhiError> {
        self.pipelines.remove_now(&self.device, handle)
    }

    /// Render one frame — clear to `color`, run `draw` if given — wait for
    /// completion, and return the target's pixels: tightly packed RGBA8,
    /// row-major, sRGB-encoded (PNG-ready bytes, §4.10).
    pub fn render(
        &mut self,
        color: [f32; 4],
        draw: Option<&DrawSpec<'_>>,
    ) -> Result<Vec<u8>, RhiError> {
        let draw = draw
            .map(|d| {
                self.pipelines
                    .get(d.pipeline)
                    .map(|entry| (entry, d.push_constants, d.vertex_count))
            })
            .transpose()?;
        let device = self.device.raw();
        let subresource = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);

        // SAFETY: single-shot recording; the wait below proves the previous
        // submission retired before the pool is reset.
        unsafe {
            device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .map_err(RhiError::Vk)?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.cmd, &begin)
                .map_err(RhiError::Vk)?;

            let to_attachment = [vk::ImageMemoryBarrier2::default()
                .image(self.image)
                .subresource_range(subresource)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)];
            device.cmd_pipeline_barrier2(
                self.cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_attachment),
            );

            let clear = vk::ClearValue {
                color: vk::ClearColorValue { float32: color },
            };
            let attachment = [vk::RenderingAttachmentInfo::default()
                .image_view(self.image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear)];
            let rendering = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D {
                        width: self.extent.0,
                        height: self.extent.1,
                    },
                })
                .layer_count(1)
                .color_attachments(&attachment);
            device.cmd_begin_rendering(self.cmd, &rendering);
            if let Some((entry, push, vertex_count)) = draw {
                // SAFETY: cmd is recording inside the pass; entry is live and
                // targets this format.
                record_draw(
                    &self.device,
                    self.cmd,
                    self.extent,
                    entry,
                    push,
                    vertex_count,
                )?;
            }
            device.cmd_end_rendering(self.cmd);

            let to_transfer = [vk::ImageMemoryBarrier2::default()
                .image(self.image)
                .subresource_range(subresource)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
            device.cmd_pipeline_barrier2(
                self.cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_transfer),
            );

            let region = vk::BufferImageCopy2::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: self.extent.0,
                    height: self.extent.1,
                    depth: 1,
                });
            let regions = [region];
            let copy = vk::CopyImageToBufferInfo2::default()
                .src_image(self.image)
                .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .dst_buffer(self.readback)
                .regions(&regions);
            device.cmd_copy_image_to_buffer2(self.cmd, &copy);

            let to_host = [vk::BufferMemoryBarrier2::default()
                .buffer(self.readback)
                .size(vk::WHOLE_SIZE)
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                .dst_access_mask(vk::AccessFlags2::HOST_READ)];
            device.cmd_pipeline_barrier2(
                self.cmd,
                &vk::DependencyInfo::default().buffer_memory_barriers(&to_host),
            );

            device.end_command_buffer(self.cmd).map_err(RhiError::Vk)?;
        }

        let signal_value = self.timeline_value + 1;
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.device.graphics.timeline)
            .value(signal_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let cmd_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(self.cmd)];
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&cmd_infos)
            .signal_semaphore_infos(&signal);
        // SAFETY: queue and semaphore are live; cmd finished recording above.
        unsafe {
            self.device
                .raw()
                .queue_submit2(self.device.graphics.raw, &[submit], vk::Fence::null())
        }
        .map_err(RhiError::Vk)?;
        self.timeline_value = signal_value;
        self.device.wait_graphics_timeline(signal_value)?;

        let alloc = self
            .readback_alloc
            .as_ref()
            .ok_or_else(|| RhiError::Allocator("readback memory gone".into()))?;
        let mapped = alloc
            .mapped_slice()
            .ok_or_else(|| RhiError::Allocator("readback memory is not host-visible".into()))?;
        let len = self.extent.0 as usize * self.extent.1 as usize * BYTES_PER_PIXEL;
        Ok(mapped[..len].to_vec())
    }

    /// Orderly teardown with the §4.3 accounting.
    pub fn shutdown(mut self) -> ShutdownReport {
        self.teardown()
    }

    fn teardown(&mut self) -> ShutdownReport {
        if self.dead {
            return ShutdownReport {
                validation_messages: validation_message_count(),
                leaked_allocations: Vec::new(),
            };
        }
        self.device.wait_idle();
        // SAFETY: GPU idle; handles belong to this device; destroyed once —
        // dead is set below and Drop re-entry returns early.
        unsafe {
            self.device.raw().destroy_command_pool(self.pool, None);
            self.device.raw().destroy_image_view(self.image_view, None);
            self.device.raw().destroy_image(self.image, None);
            self.device.raw().destroy_buffer(self.readback, None);
        }
        if let Some(alloc) = self.image_alloc.take() {
            let _ = self.device.free(alloc);
        }
        if let Some(alloc) = self.readback_alloc.take() {
            let _ = self.device.free(alloc);
        }
        self.pipelines.destroy(&self.device);
        let report = ShutdownReport {
            leaked_allocations: self.device.leak_report(),
            validation_messages: validation_message_count(),
        };
        if !report.leaked_allocations.is_empty() {
            tracing::error!(leaks = ?report.leaked_allocations, "GPU allocations leaked (§4.3)");
        }
        self.device.destroy();
        self.instance.destroy();
        self.dead = true;
        report
    }
}

impl Drop for OffscreenRhi {
    fn drop(&mut self) {
        // Backstop for early-exit paths; the accountable path is shutdown().
        self.teardown();
    }
}
