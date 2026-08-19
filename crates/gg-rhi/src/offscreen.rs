//! Offscreen rendering (§4.10 v0): no window, no surface, no swapchain — an
//! RGBA8-sRGB target image and a single-shot execute-and-wait path. This is
//! gg-golden's whole GPU footprint, which is what lets the harness be headless
//! *by linkage* (§5 gate 7: no winit symbols), not just by configuration.
//!
//! It runs the *same* [`Pass`] list a windowed frame does (§4.5): the target
//! stands in for the swapchain image as [`Target::Backbuffer`], so a graph
//! written for the screen renders here unchanged, and the pixels come back
//! through the graph's own readback pass rather than a path only this file
//! knows.

use crate::gpu::Gpu;
use crate::graph::{self, Bound};
use crate::instance::{Instance, Presentation, validation_message_count};
use crate::pipeline::{PipelineDesc, PipelineHandle};
use crate::resource::{
    BufferDesc, BufferHandle, DeviceAddress, ImageDesc, ImageFormat, ImageHandle, ImageUse,
};
use crate::{Pass, RhiError, ShutdownReport, TextureIndex};
use ash::vk;
use std::path::Path;

/// The offscreen target's format. Fixed: PNG-ready bytes are the point (§4.10).
const TARGET_FORMAT: ImageFormat = ImageFormat::Rgba8Srgb;

/// A window-less rendering context: bring-up without a display, execute a graph
/// into an offscreen target, read the pixels back. Same accounting rules as
/// [`Rhi`] (§4.3): every byte allocated is tracked, shutdown reports leaks and
/// validation messages.
///
/// [`Rhi`]: crate::Rhi
pub struct OffscreenRhi {
    dead: bool,
    timeline_value: u64,
    transfer_waited: u64,
    extent: (u32, u32),
    target: ImageHandle,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    timings: Option<crate::timing::Timings>,
    gpu: Gpu,
    instance: Instance,
}

impl OffscreenRhi {
    /// Bring up a device with no display attached and an offscreen RGBA8-sRGB
    /// target of `extent` pixels. The warm pipeline cache lands under the dev
    /// tree's `target/`, which is every offscreen caller's right answer — a
    /// harness and an instrument have no player to keep bytes for.
    pub fn new(extent: (u32, u32)) -> Result<Self, RhiError> {
        Self::with_cache(extent, None)
    }

    /// As [`OffscreenRhi::new`], naming where the pipeline cache goes.
    ///
    /// Exists for one caller: the gate that has to prove the directory is
    /// *obeyed* (§6 M52), which cannot be written against a default without
    /// asserting the defect it is checking for.
    pub fn with_cache(extent: (u32, u32), cache_dir: Option<&Path>) -> Result<Self, RhiError> {
        if extent.0 == 0 || extent.1 == 0 {
            return Err(RhiError::Loader("offscreen extent must be nonzero".into()));
        }
        let mut instance = Instance::new(Presentation::None)?;
        let mut gpu = match Gpu::new(&instance, None, 1, cache_dir) {
            Ok(g) => g,
            Err(e) => {
                instance.destroy();
                return Err(e);
            }
        };

        let (target, pool, cmd) = match Self::create_resources(&mut gpu, extent) {
            Ok(r) => r,
            Err(e) => {
                gpu.destroy();
                instance.destroy();
                return Err(e);
            }
        };
        // One slot: an offscreen execute blocks until the GPU is done with it,
        // so there is never a second frame's timestamps to keep apart.
        let timings = match crate::timing::Timings::new(&gpu.device, 1) {
            Ok(t) => t,
            Err(e) => {
                // SAFETY: the pool was just created on this device and nothing
                // has been submitted, so no command buffer of it is in flight.
                unsafe { gpu.device.raw().destroy_command_pool(pool, None) };
                gpu.destroy();
                instance.destroy();
                return Err(e);
            }
        };
        Ok(Self {
            dead: false,
            timeline_value: 0,
            transfer_waited: 0,
            extent,
            target,
            pool,
            cmd,
            timings,
            gpu,
            instance,
        })
    }

    fn create_resources(
        gpu: &mut Gpu,
        extent: (u32, u32),
    ) -> Result<(ImageHandle, vk::CommandPool, vk::CommandBuffer), RhiError> {
        // A `ColorTarget` image and nothing bespoke: the target is allocated,
        // named and leak-accounted by the same path every other image is.
        let target = gpu.create_image(&ImageDesc {
            name: "gg.offscreen.target",
            extent,
            format: TARGET_FORMAT,
            usage: ImageUse::ColorTarget,
            mip_levels: 1,
            // The offscreen target is where a resolve lands, never what is
            // resolved: the harness reads it back, and a multisample image is
            // not a legal copy source.
            samples: crate::Samples::X1,
        })?;

        let device = &mut gpu.device;
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(device.graphics.family);
        // SAFETY: device is live.
        let pool =
            unsafe { device.raw().create_command_pool(&pool_info, None) }.map_err(RhiError::vk)?;
        device.set_name(pool, "gg.offscreen.pool");
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: pool is live.
        let cmd = unsafe { device.raw().allocate_command_buffers(&alloc_info) }
            .map_err(RhiError::vk)?
            .into_iter()
            .next()
            .ok_or_else(|| RhiError::Loader("no command buffer allocated".into()))?;
        device.set_name(cmd, "gg.offscreen.cmd");
        Ok((target, pool, cmd))
    }

    /// The device-selection report (§4.3).
    pub fn device_report(&self) -> &crate::DeviceReport {
        self.gpu.device.report()
    }

    /// Target extent in pixels.
    pub fn extent(&self) -> (u32, u32) {
        self.extent
    }

    /// The image [`Target::Backbuffer`] resolves to here — what a graph's final
    /// pass writes and a readback pass copies out.
    ///
    /// [`Target::Backbuffer`]: crate::Target::Backbuffer
    pub fn backbuffer(&self) -> ImageHandle {
        self.target
    }

    /// Whether uploads cross a queue-family boundary on this device (§4.3).
    pub fn transfer_crosses_queue_families(&self) -> bool {
        self.gpu.device.transfer_crosses_families()
    }

    /// Create a graphics pipeline. [`ColorTarget::Backbuffer`] means the
    /// offscreen target's format here.
    ///
    /// [`ColorTarget::Backbuffer`]: crate::ColorTarget::Backbuffer
    pub fn create_pipeline(&mut self, desc: &PipelineDesc<'_>) -> Result<PipelineHandle, RhiError> {
        let set_layout = self.gpu.bindless.layout();
        self.gpu
            .pipelines
            .create(&self.gpu.device, desc, TARGET_FORMAT.vk(), set_layout)
    }

    /// Destroy a pipeline. Offscreen renders are synchronous ([`Self::execute`]
    /// waits for completion before returning), so nothing can be in flight
    /// and destruction is immediate.
    pub fn destroy_pipeline(&mut self, handle: PipelineHandle) -> Result<(), RhiError> {
        self.gpu.pipelines.remove_now(&self.gpu.device, handle)
    }

    /// Write the warm pipeline cache out now — see
    /// [`Rhi::persist_pipeline_cache`](crate::Rhi::persist_pipeline_cache).
    pub fn persist_pipeline_cache(&mut self) {
        self.gpu.pipelines.persist(&self.gpu.device);
    }

    /// Allocate a buffer (§4.3: reached by device address).
    pub fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> Result<BufferHandle, RhiError> {
        self.gpu.create_buffer(desc)
    }

    /// The GPU pointer a shader reaches `handle` by.
    pub fn buffer_address(&self, handle: BufferHandle) -> Result<DeviceAddress, RhiError> {
        self.gpu.buffer_address(handle)
    }

    /// A readback buffer's bytes, valid once [`Self::execute`] returned.
    ///
    /// # Errors
    ///
    /// The buffer is not [`BufferKind::Readback`](crate::BufferKind).
    pub fn map_buffer(&self, handle: BufferHandle) -> Result<&[u8], RhiError> {
        self.gpu.resources.mapped(handle)
    }

    /// Allocate an image.
    pub fn create_image(&mut self, desc: &ImageDesc<'_>) -> Result<ImageHandle, RhiError> {
        self.gpu.create_image(desc)
    }

    /// Record a buffer upload into the staging batch.
    pub fn upload_buffer(
        &mut self,
        handle: BufferHandle,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), RhiError> {
        self.gpu.upload_buffer(handle, offset, bytes)
    }

    /// Record one mip level's upload into the staging batch.
    pub fn upload_image(
        &mut self,
        handle: ImageHandle,
        level: u32,
        bytes: &[u8],
    ) -> Result<(), RhiError> {
        self.gpu.upload_image(handle, level, bytes)
    }

    /// Write into a dynamic buffer's mapping — see [`crate::Rhi::write_buffer`].
    ///
    /// # Errors
    ///
    /// A buffer that is not host-visible, or a write past its end.
    pub fn write_buffer(
        &mut self,
        handle: BufferHandle,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), RhiError> {
        self.gpu.write_buffer(handle, offset, bytes)
    }

    /// Live buffer and image allocations (§4.8).
    pub fn memory(&self) -> crate::MemoryUse {
        self.gpu.memory()
    }

    /// Submit the staging batch and wait for it (§4.3 "upload now").
    pub fn flush_uploads(&mut self) -> Result<(), RhiError> {
        self.gpu.flush_uploads_blocking()
    }

    /// Give an image a slot in the global sampled-image array.
    pub fn register_texture(&mut self, handle: ImageHandle) -> Result<TextureIndex, RhiError> {
        self.gpu.register_texture(handle)
    }

    /// Give an image a slot in the global storage-image array, transitioning
    /// it into the layout that array declares.
    pub fn register_storage_image(
        &mut self,
        handle: ImageHandle,
    ) -> Result<crate::StorageImageIndex, RhiError> {
        let index = self.gpu.register_storage_image(handle)?;
        let image = self.gpu.resources.image(handle)?.raw;
        self.one_shot(|gpu, cmd| {
            // SAFETY: cmd is recording on the graphics family (see one_shot).
            unsafe { gpu.record_storage_image_transition(cmd, image) };
            Ok(())
        })?;
        Ok(index)
    }

    /// Destroy a buffer. Renders are synchronous, so nothing is in flight.
    pub fn destroy_buffer(&mut self, handle: BufferHandle) -> Result<(), RhiError> {
        self.gpu.retire_buffer(handle, self.timeline_value)?;
        let completed = self.gpu.device.graphics_timeline_value()?;
        self.gpu.collect(completed);
        Ok(())
    }

    /// Destroy an image. Renders are synchronous, so nothing is in flight.
    pub fn destroy_image(&mut self, handle: ImageHandle) -> Result<(), RhiError> {
        self.gpu.retire_image(handle, self.timeline_value)?;
        let completed = self.gpu.device.graphics_timeline_value()?;
        self.gpu.collect(completed);
        Ok(())
    }

    /// Record and run one compiled graph, blocking until it retires.
    ///
    /// Synchronous by design: a harness that has to poll for its own pixels is
    /// a harness with a race in it. Pixels come back through a readback pass
    /// into a [`BufferKind::Readback`](crate::BufferKind) buffer, read with
    /// [`Self::map_buffer`].
    ///
    /// # Errors
    ///
    /// A stale handle, or a draw whose push constants do not match its
    /// pipeline — both found while resolving, before anything is recorded.
    pub fn execute(&mut self, passes: &[Pass<'_>]) -> Result<(), RhiError> {
        let resolved = graph::resolve(&self.gpu, passes)?;
        if let Some(timings) = &mut self.timings {
            crate::timing::warn_pass_overflow(passes.len());
            timings.expect(0, passes.iter().map(|p| p.name.to_owned()));
        }
        let stamps = self.timings.as_ref().map(|t| t.stamps(0));
        let crumbs = self.gpu.prepare_crumbs(0, passes.iter().map(|p| p.name))?;
        let image = self.gpu.resources.image(self.target)?;
        let backbuffer = Bound {
            image: image.raw,
            view: image.view,
            extent: image.extent,
            aspect: image.format.aspect(),
        };
        let acquires = self.gpu.take_acquires();
        let device = self.gpu.device.raw();
        // SAFETY: single-shot recording; the wait below proves the previous
        // submission retired before the pool is reset.
        unsafe {
            device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .map_err(RhiError::vk)?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.cmd, &begin)
                .map_err(RhiError::vk)?;
            // SAFETY: cmd records on the graphics family and this submit waits
            // on the transfer timeline value the releases signaled.
            crate::upload::Uploader::record_acquires(&self.gpu.device, self.cmd, &acquires);
            // SAFETY: as above; every handle came from resolve.
            graph::record(
                &self.gpu.device,
                self.cmd,
                self.gpu.bindless.set(),
                &resolved,
                backbuffer,
                graph::Instruments { stamps, crumbs },
            );
            device.end_command_buffer(self.cmd).map_err(RhiError::vk)?;
        }
        self.submit_and_wait()?;
        // The wait is inside submit_and_wait, so by here the timestamps are
        // complete and this read never blocks.
        if let Some(timings) = &mut self.timings {
            timings.collect(&self.gpu.device, 0);
        }
        Ok(())
    }

    /// Per-pass GPU time from the last [`Self::execute`] — current rather than a
    /// frame behind, because an offscreen render blocks until it retires.
    /// Empty without the `gpu-timings` feature or on a device whose graphics
    /// queue writes no timestamp.
    pub fn pass_timings(&self) -> &[crate::PassTiming] {
        self.timings.as_ref().map(|t| t.last()).unwrap_or_default()
    }

    /// The device clock now — see [`crate::Rhi::gpu_clock`].
    pub fn gpu_clock(&self) -> Option<crate::GpuClock> {
        self.timings.as_ref()?.clock(&self.gpu.device)
    }

    /// What the last [`Self::execute`]'s breadcrumbs say it reached — see
    /// [`crate::Rhi::breadcrumbs`].
    pub fn breadcrumbs(&self) -> String {
        self.gpu.breadcrumbs()
    }

    /// Record and run a one-shot graphics command buffer to completion. Used
    /// by paths that need a barrier without a frame around it.
    fn one_shot(
        &mut self,
        record: impl FnOnce(&Gpu, vk::CommandBuffer) -> Result<(), RhiError>,
    ) -> Result<(), RhiError> {
        let device = self.gpu.device.raw();
        // SAFETY: renders are synchronous, so the previous submission retired.
        unsafe {
            device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
                .map_err(RhiError::vk)?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.cmd, &begin)
                .map_err(RhiError::vk)?;
        }
        record(&self.gpu, self.cmd)?;
        // SAFETY: the buffer has been recording since the begin above.
        unsafe { device.end_command_buffer(self.cmd) }.map_err(RhiError::vk)?;
        self.submit_and_wait()
    }

    /// Submit `self.cmd` on the graphics queue and block until it retires,
    /// chaining after any flushed upload batch.
    fn submit_and_wait(&mut self) -> Result<(), RhiError> {
        let signal_value = self.timeline_value + 1;
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(self.gpu.device.graphics.timeline)
            .value(signal_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let transfer_value = self.gpu.uploader.timeline_value();
        let wait = if transfer_value > self.transfer_waited {
            vec![
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.gpu.device.transfer.timeline)
                    .value(transfer_value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            ]
        } else {
            Vec::new()
        };
        let cmd_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(self.cmd)];
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait)
            .command_buffer_infos(&cmd_infos)
            .signal_semaphore_infos(&signal);
        // SAFETY: queue and semaphore are live; cmd finished recording.
        unsafe {
            self.gpu.device.raw().queue_submit2(
                self.gpu.device.graphics.raw,
                &[submit],
                vk::Fence::null(),
            )
        }
        .map_err(|e| self.gpu.explain(e))?;
        self.timeline_value = signal_value;
        self.transfer_waited = transfer_value;
        self.gpu
            .device
            .wait_graphics_timeline(signal_value)
            .map_err(|e| self.gpu.detail(e))?;
        // A completed wait is not proof of a live device — see `check_alive`.
        // Only this path needs the probe: the windowed frame path presents
        // immediately after its wait, and `queue_present` reports the loss
        // while the crumbs still belong to the frame that caused it.
        self.gpu
            .device
            .check_alive()
            .map_err(|e| self.gpu.detail(e))
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
        self.gpu.device.wait_idle();
        if let Some(timings) = &mut self.timings {
            timings.destroy(&self.gpu.device);
        }
        // SAFETY: GPU idle; the pool belongs to this device and is destroyed
        // once — `dead` is set below and Drop re-entry returns early.
        unsafe { self.gpu.device.raw().destroy_command_pool(self.pool, None) };
        let report = ShutdownReport {
            leaked_allocations: self.gpu.destroy(),
            validation_messages: validation_message_count(),
        };
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
