//! `gg-rhi` — the only crate that speaks raw Vulkan (§4.3), a machine-enforced
//! property: `ash` is deny-banned elsewhere and vk:: tokens outside this crate
//! fail the §3 grep. The public API is safe, small, and engine-shaped: bring
//! the device up with a precise report, clear and present frames with 2 frames
//! in flight on timeline semaphores, treat swapchain recreation as a normal
//! event, upload through a staging ring on the transfer queue, reach every
//! texture through one global bindless set and every buffer through its device
//! address — and account for every byte and every validation message at
//! shutdown.
//!
//! Complexity budget (§3): no backend abstraction, no render concepts (meshes,
//! materials live above), resource newtypes only.

#![warn(missing_docs)]

mod bindless;
mod deletion;
mod device;
mod frame;
mod gpu;
mod instance;
mod offscreen;
mod pipeline;
mod resource;
mod suppressions;
mod surface;
mod swapchain;
mod upload;

pub use bindless::{StorageImageIndex, TextureIndex};
pub use device::{Candidate, DeviceReport};
pub use frame::FRAMES_IN_FLIGHT;
pub use instance::validation_message_count;
pub use offscreen::OffscreenRhi;
pub use pipeline::{PipelineDesc, PipelineHandle};
pub use resource::{
    BufferDesc, BufferHandle, BufferKind, DeviceAddress, ImageDesc, ImageFormat, ImageHandle,
    ImageUse, Sampler,
};
pub use suppressions::{parse as parse_suppressions, validated as validated_suppressions};

use ash::vk;
use frame::Frames;
use gpu::Gpu;
use instance::Instance;
use pipeline::ResolvedDraw;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use surface::Surface;
use swapchain::{Acquired, Swapchain};

/// One draw call: a pipeline, its push constants, and how many vertices — or,
/// with `index_buffer` set, how many indices. Vertex *data* is never bound:
/// the shader pulls it through a device address it receives in the push
/// constants (§4.3).
pub struct DrawSpec<'a> {
    /// The pipeline to draw with.
    pub pipeline: PipelineHandle,
    /// Push-constant bytes; length must equal the pipeline's declared size.
    pub push_constants: &'a [u8],
    /// Vertices to draw, or indices to consume when `index_buffer` is set.
    pub count: u32,
    /// Index stream for an indexed draw (`u32` indices from byte 0). `None`
    /// draws non-indexed, as shader-generated geometry does.
    pub index_buffer: Option<BufferHandle>,
}

/// Errors from the RHI. Vulkan result codes surface as their spec names; the
/// §3 containment grep keeps callers from matching on raw codes anyway.
#[derive(Debug, thiserror::Error)]
pub enum RhiError {
    /// Loader, handle, or bookkeeping failure outside a Vulkan call.
    #[error("{0}")]
    Loader(String),
    /// A Vulkan call failed.
    #[error("vulkan: {0:?}")]
    Vk(vk::Result),
    /// The `validation` feature needs the Khronos layer and it is absent.
    #[error("{0}")]
    MissingLayer(String),
    /// No physical device satisfies §4.3; the message is the per-device report.
    #[error("{0}")]
    NoSuitableDevice(String),
    /// GPU allocator failure.
    #[error("allocator: {0}")]
    Allocator(String),
}

/// What a [`Rhi::render_clear_frame`] call did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameOutcome {
    /// A frame was recorded, submitted, and presented.
    Presented {
        /// The WSI flagged the swapchain suboptimal; recreation is queued.
        suboptimal: bool,
    },
    /// The surface is zero-extent (minimized); nothing was rendered.
    SkippedSuspended,
    /// Acquire found the swapchain out of date; recreation is queued and no
    /// frame was rendered. The next call renders.
    SkippedOutOfDate,
}

/// The shutdown accounting (§4.3, §5.4): CI-facing truth, not a log line.
#[derive(Clone, Debug)]
pub struct ShutdownReport {
    /// Validation messages heard this process (post-suppression). Must be 0.
    pub validation_messages: u64,
    /// Allocations still live at shutdown, named. Must be empty.
    pub leaked_allocations: Vec<String>,
}

impl ShutdownReport {
    /// Zero validation messages and zero leaks.
    pub fn clean(&self) -> bool {
        self.validation_messages == 0 && self.leaked_allocations.is_empty()
    }
}

/// The windowed RHI: one window, one device, one swapchain, frames that clear,
/// draw, and present. Grows engine-shaped surface with its consumers (§9).
pub struct Rhi {
    frame_index: u64,
    desired_extent: (u32, u32),
    pending_recreate: bool,
    dead: bool,
    /// Transfer timeline value the graphics queue has already waited on, so a
    /// flush is ordered before the first frame that reads it exactly once.
    transfer_waited: u64,
    depth: ImageHandle,
    depth_extent: (u32, u32),
    gpu: Gpu,
    frames: Frames,
    swapchain: Swapchain,
    surface: Surface,
    instance: Instance,
}

impl Rhi {
    /// Bring up instance, surface, device, swapchain, and frame slots for
    /// `window`.
    ///
    /// **The window must outlive the returned value.** `ash_window` requires
    /// the window and display connection to stay valid for the lifetime of the
    /// `VkSurfaceKHR` derived from them, and nothing in this signature enforces
    /// it — `HasWindowHandle` hands out a borrowed handle whose lifetime is
    /// dropped at the FFI boundary, so a lifetime parameter here would have to
    /// propagate into every caller's storage to mean anything.
    ///
    /// Callers driving `gg_platform::run` uphold this by tearing down in the
    /// `Event::Exiting` arm, which is dispatched while the window is still
    /// alive; doing it after `run` returns is too late. Callers owning their
    /// own window must destroy the `Rhi` (via [`Rhi::shutdown`] or by dropping
    /// it) before the window.
    pub fn new(
        window: &(impl HasWindowHandle + HasDisplayHandle),
        extent: (u32, u32),
    ) -> Result<Self, RhiError> {
        let display = window
            .display_handle()
            .map_err(|e| RhiError::Loader(format!("no display handle: {e}")))?;
        let mut instance = Instance::new(Some(display.as_raw()))?;
        let mut surface = match Surface::new(&instance, window) {
            Ok(s) => s,
            Err(e) => {
                instance.destroy();
                return Err(e);
            }
        };
        let mut gpu = match Gpu::new(&instance, Some(&surface)) {
            Ok(g) => g,
            Err(e) => {
                surface.destroy();
                instance.destroy();
                return Err(e);
            }
        };
        let mut swapchain = match Swapchain::new(&gpu.device, &surface, extent) {
            Ok(s) => s,
            Err(e) => {
                gpu.destroy();
                surface.destroy();
                instance.destroy();
                return Err(e);
            }
        };
        let mut frames = match Frames::new(&gpu.device) {
            Ok(f) => f,
            Err(e) => {
                swapchain.destroy(&gpu.device);
                gpu.destroy();
                surface.destroy();
                instance.destroy();
                return Err(e);
            }
        };
        // The depth target follows the swapchain's extent, not the requested
        // one: a minimized surface clamps to a nonzero swapchain extent, and
        // the attachment has to match what the pass renders into.
        let depth_extent = swapchain.extent();
        let depth = match gpu.create_depth(depth_extent) {
            Ok(d) => d,
            Err(e) => {
                frames.destroy(&gpu.device);
                swapchain.destroy(&gpu.device);
                gpu.destroy();
                surface.destroy();
                instance.destroy();
                return Err(e);
            }
        };
        Ok(Self {
            frame_index: 0,
            desired_extent: extent,
            pending_recreate: false,
            dead: false,
            transfer_waited: 0,
            depth,
            depth_extent,
            gpu,
            frames,
            swapchain,
            surface,
            instance,
        })
    }

    /// Create a graphics pipeline targeting this window's swapchain format
    /// (§4.4: dynamic rendering, disk-backed cache, creation timed + logged).
    pub fn create_pipeline(&mut self, desc: &PipelineDesc<'_>) -> Result<PipelineHandle, RhiError> {
        let format = self.swapchain.format(&self.gpu.device, &self.surface)?;
        let set_layout = self.gpu.bindless.layout();
        self.gpu
            .pipelines
            .create(&self.gpu.device, desc, format, set_layout)
    }

    /// Retire a pipeline behind the frame timeline — safe to call while
    /// frames using it are still in flight; destruction is deferred (§4.4:
    /// hot reload rebuilds behind the timeline, never mid-frame).
    pub fn destroy_pipeline(&mut self, handle: PipelineHandle) -> Result<(), RhiError> {
        self.gpu
            .pipelines
            .retire(handle, &mut self.gpu.deletions, self.frame_index)
    }

    /// Allocate a buffer (§4.3: reached by device address, never by descriptor).
    pub fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> Result<BufferHandle, RhiError> {
        self.gpu.create_buffer(desc)
    }

    /// The GPU pointer a shader reaches `handle` by — what goes in the push
    /// constant or scene buffer.
    pub fn buffer_address(&self, handle: BufferHandle) -> Result<DeviceAddress, RhiError> {
        self.gpu.buffer_address(handle)
    }

    /// Allocate an image.
    pub fn create_image(&mut self, desc: &ImageDesc<'_>) -> Result<ImageHandle, RhiError> {
        self.gpu.create_image(desc)
    }

    /// Record a buffer upload into the staging batch (§4.3).
    pub fn upload_buffer(
        &mut self,
        handle: BufferHandle,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), RhiError> {
        self.gpu.upload_buffer(handle, offset, bytes)
    }

    /// Record an image upload into the staging batch. `bytes` is the whole
    /// image, tightly packed for its format.
    pub fn upload_image(&mut self, handle: ImageHandle, bytes: &[u8]) -> Result<(), RhiError> {
        self.gpu.upload_image(handle, bytes)
    }

    /// Submit the staging batch and wait for it — the §4.3 "upload now" path.
    /// The next frame records the graphics-side ownership acquires.
    pub fn flush_uploads(&mut self) -> Result<(), RhiError> {
        self.gpu.flush_uploads_blocking()
    }

    /// Give an image a slot in the global sampled-image array; the returned
    /// index is what a material stores and a shader indexes (§4.3).
    pub fn register_texture(&mut self, handle: ImageHandle) -> Result<TextureIndex, RhiError> {
        self.gpu.register_texture(handle)
    }

    /// Retire a buffer behind the frame timeline.
    pub fn destroy_buffer(&mut self, handle: BufferHandle) -> Result<(), RhiError> {
        self.gpu.retire_buffer(handle, self.frame_index)
    }

    /// Retire an image behind the frame timeline.
    pub fn destroy_image(&mut self, handle: ImageHandle) -> Result<(), RhiError> {
        self.gpu.retire_image(handle, self.frame_index)
    }

    /// Whether uploads cross a queue-family boundary on this device (§4.3).
    /// `false` on single-family devices such as lavapipe — a fact tests assert
    /// on rather than assume.
    pub fn transfer_crosses_queue_families(&self) -> bool {
        self.gpu.device.transfer_crosses_families()
    }

    /// The device-selection report (§4.3: a diagnostic document, kept).
    pub fn device_report(&self) -> &DeviceReport {
        self.gpu.device.report()
    }

    /// Note a new surface size; the swapchain recreates on the next frame.
    /// `(0, 0)` (minimized) suspends rendering until a nonzero resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.desired_extent = (width, height);
        self.pending_recreate = true;
    }

    /// Swapchain recreations so far (tests assert the torture paths recreate).
    pub fn swapchain_generation(&self) -> u64 {
        self.swapchain.generation()
    }

    /// The swapchain's current extent.
    pub fn swapchain_extent(&self) -> (u32, u32) {
        self.swapchain.extent()
    }

    /// Frames successfully presented is not tracked separately; this is the
    /// monotonically increasing count of frames *submitted*.
    pub fn frames_submitted(&self) -> u64 {
        self.frame_index
    }

    /// Validation messages heard so far (0 without the `validation` feature).
    pub fn validation_messages(&self) -> u64 {
        validation_message_count()
    }

    /// Record, submit, and present one frame that clears the swapchain image
    /// to `color` (linear values; the sRGB target encodes). Handles resize,
    /// minimize, out-of-date, and suboptimal as normal events (§4.3).
    pub fn render_clear_frame(&mut self, color: [f32; 4]) -> Result<FrameOutcome, RhiError> {
        self.render_frame(color, None)
    }

    /// Record, submit, and present one frame: clear to `color`, then run
    /// `draw` if given (the one-pass world; the render graph owns passes from
    /// M6). Resize/minimize/out-of-date handled as in
    /// [`Rhi::render_clear_frame`].
    pub fn render_frame(
        &mut self,
        color: [f32; 4],
        draw: Option<&DrawSpec<'_>>,
    ) -> Result<FrameOutcome, RhiError> {
        if self.pending_recreate || self.swapchain.suspended() {
            // Recreation is a normal event but a *structural* one: presents
            // in flight hold the retired per-image semaphores with no signal
            // to key their deletion to, so this path — and only this path —
            // waits the queue idle before retiring (see wait_graphics_idle).
            self.gpu.device.wait_graphics_idle();
            let retire = self.frame_index;
            let rebuilt = self.swapchain.recreate(
                &self.gpu.device,
                &self.surface,
                self.desired_extent,
                &mut self.gpu.deletions,
                retire,
            )?;
            self.pending_recreate = false;
            if !rebuilt {
                return Ok(FrameOutcome::SkippedSuspended);
            }
            self.resize_depth(retire)?;
        }
        if self.swapchain.suspended() {
            return Ok(FrameOutcome::SkippedSuspended);
        }

        // §4.3: frames-in-flight = 2, expressed as a wait on
        // timeline >= frame_index - 1 (submit N signals value N + 1).
        let horizon = self.frame_index.saturating_sub(FRAMES_IN_FLIGHT - 1);
        if horizon > 0 {
            self.gpu.device.wait_graphics_timeline(horizon)?;
        }
        let completed = self.gpu.device.graphics_timeline_value()?;
        self.gpu.collect(completed);

        // Resolve the draw's handles BEFORE acquiring. A stale handle (retired
        // by hot reload) is a normal, caller-reachable error, and acquire
        // signals `slot.acquire` — so returning between the two would leave a
        // signaled semaphore with no waiter, which the next frame to reuse this
        // slot trips over as a validation error, permanently failing the §4.3
        // zero-message shutdown report. Nothing may fail in between.
        let draw = self.resolve_draw(draw)?;
        let depth = self.gpu.resources.image(self.depth)?;
        let (depth_image, depth_view) = (depth.raw, depth.view);

        let slot = &self.frames.slots[(self.frame_index % FRAMES_IN_FLIGHT) as usize];
        let (image_index, acquire_suboptimal) =
            match self.swapchain.acquire(&self.gpu.device, slot.acquire)? {
                Acquired::Image { index, suboptimal } => (index, suboptimal),
                Acquired::OutOfDate => {
                    self.pending_recreate = true;
                    return Ok(FrameOutcome::SkippedOutOfDate);
                }
            };

        let acquires = self.gpu.take_acquires();
        self.record_pass(
            slot.pool,
            slot.cmd,
            image_index,
            color,
            (depth_image, depth_view),
            &acquires,
            draw.as_ref(),
        )?;

        // Submit: wait the acquire binary (and the transfer timeline when an
        // upload has been flushed since the last frame), signal the per-image
        // render-done binary (for present) and the graphics timeline (frame
        // pacing and the deletion queue's clock).
        let signal_value = self.frame_index + 1;
        let mut wait = vec![
            vk::SemaphoreSubmitInfo::default()
                .semaphore(slot.acquire)
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
        ];
        let transfer_value = self.gpu.uploader.timeline_value();
        if transfer_value > self.transfer_waited {
            wait.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.gpu.device.transfer.timeline)
                    .value(transfer_value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }
        let signal = [
            // ALL_COMMANDS, not COLOR_ATTACHMENT_OUTPUT: the present-layout
            // transition happens outside that stage, and the present must
            // chain after it (sync validation: PRESENT_AFTER_WRITE otherwise).
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.swapchain.render_done(image_index))
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.gpu.device.graphics.timeline)
                .value(signal_value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        ];
        let cmd_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(slot.cmd)];
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait)
            .command_buffer_infos(&cmd_infos)
            .signal_semaphore_infos(&signal);
        // SAFETY: queue, semaphores, and command buffer are live; the buffer
        // finished recording in record_pass.
        unsafe {
            self.gpu.device.raw().queue_submit2(
                self.gpu.device.graphics.raw,
                &[submit],
                vk::Fence::null(),
            )
        }
        .map_err(RhiError::Vk)?;
        self.transfer_waited = transfer_value;

        // Present, waiting the per-image render-done semaphore.
        let wait_semaphores = [self.swapchain.render_done(image_index)];
        let swapchains = [self.swapchain.raw()];
        let indices = [image_index];
        let present = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&indices);
        // SAFETY: swapchain and queue are live; image_index came from acquire.
        let present_result = unsafe {
            self.gpu
                .device
                .swapchain_fns()
                .queue_present(self.gpu.device.graphics.raw, &present)
        };
        self.frame_index += 1;

        match present_result {
            Ok(false) => Ok(FrameOutcome::Presented {
                suboptimal: acquire_suboptimal,
            }),
            Ok(true) => {
                self.pending_recreate = true;
                Ok(FrameOutcome::Presented { suboptimal: true })
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.pending_recreate = true;
                Ok(FrameOutcome::Presented { suboptimal: true })
            }
            Err(err) => Err(RhiError::Vk(err)),
        }
    }

    /// Resolve a draw's handles to Vulkan objects, so the record path cannot
    /// fail. See the call site for why this must happen before acquire.
    fn resolve_draw<'a>(
        &self,
        draw: Option<&DrawSpec<'a>>,
    ) -> Result<Option<ResolvedDraw<'a>>, RhiError> {
        draw.map(|d| {
            let entry = *self.gpu.pipelines.get(d.pipeline)?;
            let index_buffer = d
                .index_buffer
                .map(|h| self.gpu.resources.buffer(h).map(|b| b.raw))
                .transpose()?;
            Ok(ResolvedDraw {
                entry,
                push_constants: d.push_constants,
                count: d.count,
                index_buffer,
            })
        })
        .transpose()
    }

    /// Rebuild the depth target to match a recreated swapchain. The old one is
    /// retired behind the timeline like any other resource — the queue-idle
    /// wait above has already proven it unused, but the deletion queue is the
    /// one path that accounts for it in the leak report.
    fn resize_depth(&mut self, retire_at: u64) -> Result<(), RhiError> {
        let extent = self.swapchain.extent();
        if extent == self.depth_extent {
            return Ok(());
        }
        let old = self.depth;
        self.depth = self.gpu.create_depth(extent)?;
        self.depth_extent = extent;
        self.gpu.retire_image(old, retire_at)
    }

    /// Record the frame's pass: the ownership acquires an upload left owing,
    /// the color and depth layout barriers, dynamic rendering with clear load
    /// ops (plus the draw, when given), then color-attachment → present.
    /// `draw` arrives already resolved (see `resolve_draw`).
    #[expect(clippy::too_many_arguments, reason = "one frame's recording inputs")]
    fn record_pass(
        &self,
        pool: vk::CommandPool,
        cmd: vk::CommandBuffer,
        image_index: u32,
        color: [f32; 4],
        depth: (vk::Image, vk::ImageView),
        acquires: &[upload::Acquire],
        draw: Option<&ResolvedDraw<'_>>,
    ) -> Result<(), RhiError> {
        let device = self.gpu.device.raw();
        // SAFETY: the timeline wait proved this slot's previous frame retired,
        // so its pool and command buffer are free to reset and rerecord.
        unsafe {
            device
                .reset_command_pool(pool, vk::CommandPoolResetFlags::empty())
                .map_err(RhiError::Vk)?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(cmd, &begin)
                .map_err(RhiError::Vk)?;

            // SAFETY: cmd records on the graphics family and this submit waits
            // on the transfer timeline value the releases signaled.
            upload::Uploader::record_acquires(&self.gpu.device, cmd, acquires);

            let subresource = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            let image = self.swapchain.image(image_index);

            let to_attachment = [
                vk::ImageMemoryBarrier2::default()
                    .image(image)
                    .subresource_range(subresource)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE),
                depth_barrier(depth.0),
            ];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_attachment),
            );

            let clear = vk::ClearValue {
                color: vk::ClearColorValue { float32: color },
            };
            let attachment = [vk::RenderingAttachmentInfo::default()
                .image_view(self.swapchain.view(image_index))
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear)];
            let depth_attachment = depth_attachment_info(depth.1);
            let extent = self.swapchain.extent();
            let rendering = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D {
                        width: extent.0,
                        height: extent.1,
                    },
                })
                .layer_count(1)
                .color_attachments(&attachment)
                .depth_attachment(&depth_attachment);
            device.cmd_begin_rendering(cmd, &rendering);
            if let Some(draw) = draw {
                // SAFETY: cmd is recording inside the rendering pass; the entry
                // came from the live store and targets these formats, and the
                // set is the one every pipeline layout declares.
                pipeline::record_draw(
                    &self.gpu.device,
                    cmd,
                    extent,
                    self.gpu.bindless.set(),
                    draw,
                )?;
            }
            device.cmd_end_rendering(cmd);

            let to_present = [vk::ImageMemoryBarrier2::default()
                .image(image)
                .subresource_range(subresource)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_access_mask(vk::AccessFlags2::NONE)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_present),
            );

            device.end_command_buffer(cmd).map_err(RhiError::Vk)?;
        }
        Ok(())
    }

    /// Orderly teardown with the §4.3 accounting: wait, drain the deletion
    /// queue, destroy everything, and report validation messages + leaks.
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
        self.gpu.device.wait_idle();
        self.frames.destroy(&self.gpu.device);
        self.swapchain.destroy(&self.gpu.device);
        let report = ShutdownReport {
            leaked_allocations: self.gpu.destroy(),
            validation_messages: validation_message_count(),
        };
        self.surface.destroy();
        self.instance.destroy();
        self.dead = true;
        report
    }
}

impl Drop for Rhi {
    fn drop(&mut self) {
        // Backstop for early-exit paths; the accountable path is shutdown().
        self.teardown();
    }
}

/// Depth attachments are cleared every frame, so the previous contents are
/// discarded: `UNDEFINED` as the old layout is the cheap, correct answer, not
/// a shortcut.
pub(crate) fn depth_barrier(image: vk::Image) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .level_count(1)
                .layer_count(1),
        )
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
        .src_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
        .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
}

/// Reverse-Z (§2, Math row): clear to `0.0` — the far plane — and let nearer
/// fragments win by comparing *greater*. Nothing reads depth after the pass at
/// M4A, so it is never stored.
pub(crate) fn depth_attachment_info(view: vk::ImageView) -> vk::RenderingAttachmentInfo<'static> {
    vk::RenderingAttachmentInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .clear_value(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 0.0,
                stencil: 0,
            },
        })
}
