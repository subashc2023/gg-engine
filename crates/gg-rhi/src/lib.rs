//! `gg-rhi` — the only crate that speaks raw Vulkan (§4.3), a machine-enforced
//! property: `ash` is deny-banned elsewhere and vk:: tokens outside this crate
//! fail the §3 grep. The public API is safe, small, and engine-shaped — at M1
//! that shape is [`Rhi`]: bring the device up with a precise report, clear and
//! present frames with 2 frames in flight on timeline semaphores, treat
//! swapchain recreation as a normal event, and account for every byte and
//! every validation message at shutdown.
//!
//! Complexity budget (§3): no backend abstraction, no render concepts (meshes,
//! materials live above), resource newtypes only.

#![warn(missing_docs)]

mod deletion;
mod device;
mod frame;
mod instance;
mod suppressions;
mod surface;
mod swapchain;

pub use device::{Candidate, DeviceReport};
pub use frame::FRAMES_IN_FLIGHT;
pub use instance::validation_message_count;
pub use suppressions::{parse as parse_suppressions, validated as validated_suppressions};

use ash::vk;
use deletion::DeletionQueue;
use device::Device;
use frame::Frames;
use instance::Instance;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use surface::Surface;
use swapchain::{Acquired, Swapchain};

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

/// The M1 RHI: one window, one device, one swapchain, clear-and-present
/// frames. Grows engine-shaped surface with its consumers (§9) — pipelines at
/// M2, bindless and uploads at M4A.
pub struct Rhi {
    frame_index: u64,
    desired_extent: (u32, u32),
    pending_recreate: bool,
    dead: bool,
    deletions: DeletionQueue,
    frames: Frames,
    swapchain: Swapchain,
    device: Device,
    surface: Surface,
    instance: Instance,
}

impl Rhi {
    /// Bring up instance, surface, device, swapchain, and frame slots for
    /// `window`. The window must outlive the returned value.
    pub fn new(
        window: &(impl HasWindowHandle + HasDisplayHandle),
        extent: (u32, u32),
    ) -> Result<Self, RhiError> {
        let display = window
            .display_handle()
            .map_err(|e| RhiError::Loader(format!("no display handle: {e}")))?;
        let mut instance = Instance::new(display.as_raw())?;
        let mut surface = match Surface::new(&instance, window) {
            Ok(s) => s,
            Err(e) => {
                instance.destroy();
                return Err(e);
            }
        };
        let mut device = match Device::new(&instance, &surface) {
            Ok(d) => d,
            Err(e) => {
                surface.destroy();
                instance.destroy();
                return Err(e);
            }
        };
        let mut swapchain = match Swapchain::new(&device, &surface, extent) {
            Ok(s) => s,
            Err(e) => {
                device.destroy();
                surface.destroy();
                instance.destroy();
                return Err(e);
            }
        };
        let frames = match Frames::new(&device) {
            Ok(f) => f,
            Err(e) => {
                swapchain.destroy(&device);
                device.destroy();
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
            deletions: DeletionQueue::default(),
            frames,
            swapchain,
            device,
            surface,
            instance,
        })
    }

    /// The device-selection report (§4.3: a diagnostic document, kept).
    pub fn device_report(&self) -> &DeviceReport {
        self.device.report()
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
        if self.pending_recreate || self.swapchain.suspended() {
            // Recreation is a normal event but a *structural* one: presents
            // in flight hold the retired per-image semaphores with no signal
            // to key their deletion to, so this path — and only this path —
            // waits the queue idle before retiring (see wait_graphics_idle).
            self.device.wait_graphics_idle();
            let retire = self.frame_index;
            let rebuilt = self.swapchain.recreate(
                &self.device,
                &self.surface,
                self.desired_extent,
                &mut self.deletions,
                retire,
            )?;
            self.pending_recreate = false;
            if !rebuilt {
                return Ok(FrameOutcome::SkippedSuspended);
            }
        }
        if self.swapchain.suspended() {
            return Ok(FrameOutcome::SkippedSuspended);
        }

        // §4.3: frames-in-flight = 2, expressed as a wait on
        // timeline >= frame_index - 1 (submit N signals value N + 1).
        let horizon = self.frame_index.saturating_sub(FRAMES_IN_FLIGHT - 1);
        if horizon > 0 {
            self.device.wait_graphics_timeline(horizon)?;
        }
        let completed = self.device.graphics_timeline_value()?;
        self.deletions.collect(&self.device, completed);

        let slot = &self.frames.slots[(self.frame_index % FRAMES_IN_FLIGHT) as usize];
        let (image_index, acquire_suboptimal) =
            match self.swapchain.acquire(&self.device, slot.acquire)? {
                Acquired::Image { index, suboptimal } => (index, suboptimal),
                Acquired::OutOfDate => {
                    self.pending_recreate = true;
                    return Ok(FrameOutcome::SkippedOutOfDate);
                }
            };

        self.record_clear(slot.pool, slot.cmd, image_index, color)?;

        // Submit: wait the acquire binary, signal the per-image render-done
        // binary (for present) and the graphics timeline (frame pacing and
        // the deletion queue's clock).
        let signal_value = self.frame_index + 1;
        let wait = [vk::SemaphoreSubmitInfo::default()
            .semaphore(slot.acquire)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];
        let signal = [
            // ALL_COMMANDS, not COLOR_ATTACHMENT_OUTPUT: the present-layout
            // transition happens outside that stage, and the present must
            // chain after it (sync validation: PRESENT_AFTER_WRITE otherwise).
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.swapchain.render_done(image_index))
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.device.graphics.timeline)
                .value(signal_value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        ];
        let cmd_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(slot.cmd)];
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait)
            .command_buffer_infos(&cmd_infos)
            .signal_semaphore_infos(&signal);
        // SAFETY: queue, semaphores, and command buffer are live; the buffer
        // finished recording in record_clear.
        unsafe {
            self.device
                .raw()
                .queue_submit2(self.device.graphics.raw, &[submit], vk::Fence::null())
        }
        .map_err(RhiError::Vk)?;

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
            self.device
                .swapchain_fns()
                .queue_present(self.device.graphics.raw, &present)
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

    /// Record the clear pass: undefined → color-attachment barrier, dynamic
    /// rendering with a clear load op, then color-attachment → present.
    fn record_clear(
        &self,
        pool: vk::CommandPool,
        cmd: vk::CommandBuffer,
        image_index: u32,
        color: [f32; 4],
    ) -> Result<(), RhiError> {
        let device = self.device.raw();
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

            let subresource = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);
            let image = self.swapchain.image(image_index);

            let to_attachment = [vk::ImageMemoryBarrier2::default()
                .image(image)
                .subresource_range(subresource)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)];
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
                .color_attachments(&attachment);
            device.cmd_begin_rendering(cmd, &rendering);
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
        self.device.wait_idle();
        self.frames.destroy(&self.device);
        self.deletions.drain_all(&self.device);
        self.swapchain.destroy(&self.device);
        let report = ShutdownReport {
            leaked_allocations: self.device.leak_report(),
            validation_messages: validation_message_count(),
        };
        if !report.leaked_allocations.is_empty() {
            tracing::error!(leaks = ?report.leaked_allocations, "GPU allocations leaked (§4.3)");
        }
        self.device.destroy();
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
