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

#![deny(missing_docs)]

mod bindless;
mod crash;
mod deletion;
mod device;
mod frame;
mod gpu;
mod graph;
mod instance;
mod offscreen;
mod pipeline;
mod resource;
// Dead without the layer it serves — `validation` off means no messenger, so
// nothing consults a lease. The tests inside still run in every tier and are the
// §5.4 gate on the checked-in file, which is why the module is not itself gated.
#[cfg_attr(not(feature = "validation"), allow(dead_code))]
mod suppressions;
mod surface;
mod swapchain;
mod timing;
mod upload;

pub use bindless::{StorageImageIndex, TextureIndex};
pub use device::{Candidate, DeviceReport};
pub use frame::FRAMES_IN_FLIGHT;
pub use graph::{
    Access, ColorAttachment, DepthAttachment, Pass, PassKind, Target, Transition, Viewport,
};
use instance::validation_message_count;
pub use offscreen::OffscreenRhi;
pub use pipeline::{Blend, ColorTarget, DepthMode, PipelineDesc, PipelineHandle};
pub use resource::{
    BufferDesc, BufferHandle, BufferKind, DeviceAddress, ImageDesc, ImageFormat, ImageHandle,
    ImageUse, MemoryUse, Sampler, Samples, full_mip_count, mip_extent,
};
pub use timing::{GpuClock, PassTiming};

use ash::vk;
use frame::Frames;
use gpu::Gpu;
use graph::Bound;
use instance::{Instance, Presentation};
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
    /// Where the draw's parameters come from. `None` takes them from
    /// [`count`](DrawSpec::count) on the CPU; `Some` reads a
    /// [`VkDrawIndexedIndirectCommand`] out of GPU memory (§6 M10).
    ///
    /// [`VkDrawIndexedIndirectCommand`]: Indirect
    pub indirect: Option<Indirect>,
    /// Depth bias for this draw. Required exactly when the pipeline declares
    /// [`PipelineDesc::depth_bias`], and refused otherwise — a pipeline whose
    /// dynamic state nothing set draws with undefined bias, which is a shadow
    /// that acnes on one driver and not another (§6 M11).
    pub depth_bias: Option<DepthBias>,
}

/// Depth bias, in the two terms Vulkan takes.
///
/// Dynamic rather than baked into the pipeline because §6 M11's exit row makes
/// these CVars: a value that needed a pipeline rebuild to change is a value
/// nobody tunes against a real scene, and acne and peter-panning are found by
/// turning the knob while looking at the wall.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthBias {
    /// Constant offset, in units of the smallest resolvable depth difference.
    pub constant: f32,
    /// Offset proportional to the fragment's depth slope — the term that covers
    /// a surface seen edge-on, where a constant one cannot.
    pub slope: f32,
}

/// Where an indirect draw reads its parameters.
///
/// # What this buys today, and what it does not
///
/// One command per call — not a multi-draw. That is deliberate: `drawCount > 1`
/// would need every batch's geometry in one index buffer, and each mesh owns its
/// own (§4.6). What the indirect path buys *now* is that the index count and the
/// instance count live in device memory, which is the whole prerequisite for a
/// GPU culling pass writing them (P2, §7). Saying it that way rather than
/// implying a draw-call reduction is the honest version: against
/// `cmd_draw_indexed` with the same instance count, this costs one buffer read.
///
/// It also needs neither `multiDrawIndirect` nor `drawIndirectFirstInstance`,
/// which is why no capability probe row moved for it.
#[derive(Clone, Copy, Debug)]
pub struct Indirect {
    /// The buffer holding the command. Must be [`BufferKind::Indirect`].
    pub buffer: BufferHandle,
    /// Byte offset of the command. Must be 4-aligned, per the spec.
    pub offset: u64,
}

/// One `VkDrawIndexedIndirectCommand`, in the layout the spec fixes.
///
/// Declared here rather than reached through `ash` because §3 keeps `vk::`
/// inside this crate, and a caller building a draw list needs to *write* these.
/// `Pod`, so a whole list is one `write_buffer`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct IndirectCommand {
    /// Indices to consume.
    pub index_count: u32,
    /// Instances to draw. Zero draws nothing — which is how a culling pass
    /// rejects a batch without rewriting the list.
    pub instance_count: u32,
    /// First index, in elements.
    pub first_index: u32,
    /// Added to every index before the vertex fetch. Zero here: geometry is
    /// pulled by device address (§4.3), so there is no vertex buffer to offset.
    pub vertex_offset: i32,
    /// First instance id, as seen by the shader. Zero here too — the batch's
    /// base is a push constant instead, which is what keeps
    /// `drawIndirectFirstInstance` off the required-capability list.
    pub first_instance: u32,
}

const _: () = assert!(core::mem::size_of::<IndirectCommand>() == 20);

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
    /// The device was lost. Separate from [`RhiError::Vk`] because the code
    /// itself says nothing: what is worth reporting is the pass the breadcrumbs
    /// caught in flight and whatever `VK_EXT_device_fault` saw (§4.8), and this
    /// carries that whole report.
    #[error("{0}")]
    DeviceLost(String),
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

/// What a [`Rhi::execute`] call did.
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

/// A frame whose swapchain is current, waiting for a graph.
///
/// Held between [`Rhi::begin_frame`] and [`Rhi::execute`] so the two are one
/// frame rather than two calls that happen to be adjacent: recreation is
/// already done, the extent below is final, and no image has been acquired yet
/// — which is what lets graph compilation (and its allocations) fail safely.
#[must_use = "a begun frame must be executed or the swapchain stalls"]
pub struct FrameToken {
    extent: (u32, u32),
}

impl FrameToken {
    /// The swapchain's extent, which the graph sizes its attachments to.
    pub fn extent(&self) -> (u32, u32) {
        self.extent
    }
}

/// What [`Rhi::begin_frame`] produced.
#[must_use]
pub enum FrameStart {
    /// Compile a graph against this and call [`Rhi::execute`].
    Ready(FrameToken),
    /// Nothing to render into — the surface is zero-extent (minimized).
    Skipped(FrameOutcome),
}

/// The windowed RHI: one window, one device, one swapchain, and frames that
/// execute a compiled graph and present. Grows engine-shaped surface with its
/// consumers (§9).
pub struct Rhi {
    frame_index: u64,
    desired_extent: (u32, u32),
    pending_recreate: bool,
    dead: bool,
    /// Transfer timeline value the graphics queue has already waited on, so a
    /// flush is ordered before the first frame that reads it exactly once.
    transfer_waited: u64,
    gpu: Gpu,
    frames: Frames,
    timings: Option<timing::Timings>,
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
        let mut instance = Instance::new(Presentation::Window(display.as_raw()))?;
        let surface = match Surface::new(&instance, window) {
            Ok(s) => s,
            Err(e) => {
                instance.destroy();
                return Err(e);
            }
        };
        Self::bring_up(instance, surface, extent)
    }

    /// Whether [`Rhi::headless`] can run here — that is, whether the loader
    /// offers `VK_EXT_headless_surface`.
    ///
    /// A gate that needs the swapchain path calls this and skips rather than
    /// fails: the extension is a Mesa build option, present on the Linux
    /// lavapipe and absent from the pinned Windows one (§6 M12).
    #[must_use]
    pub fn headless_supported() -> bool {
        Instance::headless_supported()
    }

    /// Bring up the full windowed path — swapchain, frames in flight, present —
    /// against a surface with **no window behind it**.
    ///
    /// This is not [`OffscreenRhi`], which has no surface and no swapchain at
    /// all: everything here is the real WSI code, and present really is called.
    /// What it buys is the one thing §1.5 otherwise makes impossible — swapchain
    /// recreation under automated CI. What it does *not* cover is the OS event
    /// path (resize and minimize as a window manager delivers them), which needs
    /// a window by definition and stays in `cargo xtask interactive`.
    ///
    /// # Errors
    /// [`RhiError::Loader`] if the loader has no `VK_EXT_headless_surface` —
    /// check [`Rhi::headless_supported`] first to skip instead.
    pub fn headless(extent: (u32, u32)) -> Result<Self, RhiError> {
        if !Instance::headless_supported() {
            return Err(RhiError::Loader(
                "VK_EXT_headless_surface is not available on this loader — \
                 Rhi::headless_supported() is the question to ask first"
                    .into(),
            ));
        }
        let mut instance = Instance::new(Presentation::Headless)?;
        let surface = match Surface::headless(&instance) {
            Ok(s) => s,
            Err(e) => {
                instance.destroy();
                return Err(e);
            }
        };
        Self::bring_up(instance, surface, extent)
    }

    /// Device, swapchain, frames and timings over a surface that is already
    /// live — shared by the windowed and headless constructors so bring-up
    /// order and failure-unwind order exist once (§4.3).
    fn bring_up(
        mut instance: Instance,
        mut surface: Surface,
        extent: (u32, u32),
    ) -> Result<Self, RhiError> {
        let mut gpu = match Gpu::new(&instance, Some(&surface), FRAMES_IN_FLIGHT as usize) {
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
        let timings = match timing::Timings::new(&gpu.device, FRAMES_IN_FLIGHT as usize) {
            Ok(t) => t,
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
            gpu,
            frames,
            timings,
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

    /// Record one mip level's upload into the staging batch. `bytes` is that
    /// level, tightly packed for its format; an image with no chain takes one
    /// call at level 0.
    pub fn upload_image(
        &mut self,
        handle: ImageHandle,
        level: u32,
        bytes: &[u8],
    ) -> Result<(), RhiError> {
        self.gpu.upload_image(handle, level, bytes)
    }

    /// Write into a [`BufferKind::Dynamic`] buffer's mapping — the per-frame
    /// stream path, with no staging copy and no submit (§4.3).
    ///
    /// Visible to the next submit on this queue without a barrier: a host write
    /// before a submit is ordered by the submit itself. What is *not* ordered
    /// is a write over bytes an in-flight frame is still reading, which is the
    /// caller's to avoid — see [`BufferKind::Dynamic`].
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

    /// Live buffer and image allocations — the overlay's memory row (§4.8).
    pub fn memory(&self) -> MemoryUse {
        self.gpu.memory()
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

    /// A readback buffer's bytes, valid once the frame that filled it retired.
    ///
    /// # Errors
    ///
    /// The buffer is not [`BufferKind::Readback`], or the handle is stale.
    pub fn map_buffer(&self, handle: BufferHandle) -> Result<&[u8], RhiError> {
        self.gpu.resources.mapped(handle)
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

    /// Per-pass GPU time from the most recently *retired* frame — one to two
    /// frames behind what was last submitted, which is the price of not
    /// stalling to ask (§4.8). Empty without the `gpu-timings` feature, on a
    /// device whose graphics queue writes no timestamp, and before the first
    /// frame has come back.
    pub fn pass_timings(&self) -> &[PassTiming] {
        self.timings.as_ref().map(|t| t.last()).unwrap_or_default()
    }

    /// The device clock now, paired with its period — what puts a
    /// [`PassTiming`]'s raw ticks on a profiler's CPU timeline (§4.8). `None`
    /// without `VK_EXT_calibrated_timestamps`, which is reported and never
    /// required.
    pub fn gpu_clock(&self) -> Option<GpuClock> {
        self.timings.as_ref()?.clock(&self.gpu.device)
    }

    /// What the last recorded frame's breadcrumbs say it reached (§4.8).
    ///
    /// On a live device this is a health check — every pass completed — and the
    /// reading worth having comes after a loss, where it names the pass that was
    /// in flight. A device-lost error already carries this text; this is for
    /// tests and for a canary that survives to ask.
    pub fn breadcrumbs(&self) -> String {
        self.gpu.breadcrumbs()
    }

    /// Make the swapchain current and hand back the extent to compile against.
    ///
    /// Recreation, minimize and the frames-in-flight wait all happen here, so
    /// that everything after it — graph compilation and the transient
    /// allocations it makes — sees a final extent and may still fail safely
    /// (see [`Rhi::execute`]).
    pub fn begin_frame(&mut self) -> Result<FrameStart, RhiError> {
        if self.pending_recreate || self.swapchain.suspended() {
            // Recreation is a normal event but a *structural* one: presents
            // in flight hold the retired per-image semaphores with no signal
            // to key their deletion to, so this path — and only this path —
            // waits the queue idle before retiring (see wait_graphics_idle).
            self.gpu.device.wait_graphics_idle();
            let rebuilt = self.swapchain.recreate(
                &self.gpu.device,
                &self.surface,
                self.desired_extent,
                &mut self.gpu.deletions,
                self.frame_index,
            )?;
            self.pending_recreate = false;
            if !rebuilt {
                return Ok(FrameStart::Skipped(FrameOutcome::SkippedSuspended));
            }
        }
        if self.swapchain.suspended() {
            return Ok(FrameStart::Skipped(FrameOutcome::SkippedSuspended));
        }

        // §4.3: frames-in-flight = 2, expressed as a wait on
        // timeline >= frame_index - 1 (submit N signals value N + 1).
        let horizon = self.frame_index.saturating_sub(FRAMES_IN_FLIGHT - 1);
        if horizon > 0 {
            self.gpu
                .device
                .wait_graphics_timeline(horizon)
                .map_err(|e| self.gpu.detail(e))?;
        }
        let completed = self.gpu.device.graphics_timeline_value()?;
        self.gpu.collect(completed);
        // The wait above is exactly the proof the timestamps this slot holds are
        // written and complete, so the read never blocks (§4.8).
        if let Some(timings) = &mut self.timings {
            timings.collect(
                &self.gpu.device,
                (self.frame_index % FRAMES_IN_FLIGHT) as usize,
            );
        }
        Ok(FrameStart::Ready(FrameToken {
            extent: self.swapchain.extent(),
        }))
    }

    /// Record, submit and present one compiled graph.
    ///
    /// The passes run in the order given; every barrier between them is
    /// derived from the [`Transition`]s they carry, and this crate writes no
    /// others (§4.5). Out-of-date and suboptimal are handled as normal events.
    ///
    /// # Errors
    ///
    /// A stale handle or a draw whose push constants do not match its pipeline.
    /// Both are found while resolving, which happens *before* the swapchain
    /// acquire: acquire signals a semaphore that only a submit can wait on, so
    /// returning between the two would leave it signaled with no waiter — which
    /// the next frame to reuse that slot trips over as a validation error,
    /// permanently failing the §4.3 zero-message shutdown report. What *can*
    /// still fail after the acquire is the device's own weather (pool OOM,
    /// device loss) out of `record` or the submit itself; those paths hand the
    /// orphaned semaphore to [`Rhi::drain_acquire`] before returning, so the
    /// slot stays reusable by whatever survives the error.
    pub fn execute(
        &mut self,
        frame: FrameToken,
        passes: &[Pass<'_>],
    ) -> Result<FrameOutcome, RhiError> {
        let _ = frame;
        let resolved = graph::resolve(&self.gpu, passes)?;

        let slot_index = (self.frame_index % FRAMES_IN_FLIGHT) as usize;
        let stamps = self.timings.as_ref().map(|t| t.stamps(slot_index));
        // Before the acquire, like everything else that can fail (see above):
        // this clears the slot's marks and names the passes about to write them.
        let crumbs = self
            .gpu
            .prepare_crumbs(slot_index, passes.iter().map(|p| p.name))?;
        let slot = &self.frames.slots[slot_index];
        let (image_index, acquire_suboptimal) =
            match self.swapchain.acquire(&self.gpu.device, slot.acquire)? {
                Acquired::Image { index, suboptimal } => (index, suboptimal),
                Acquired::OutOfDate => {
                    self.pending_recreate = true;
                    return Ok(FrameOutcome::SkippedOutOfDate);
                }
            };

        let backbuffer = Bound {
            image: self.swapchain.image(image_index),
            view: self.swapchain.view(image_index),
            extent: self.swapchain.extent(),
            aspect: vk::ImageAspectFlags::COLOR,
        };
        let acquires = self.gpu.take_acquires();
        let instruments = graph::Instruments { stamps, crumbs };
        if let Err(e) = self.record(
            slot.pool,
            slot.cmd,
            &acquires,
            &resolved,
            backbuffer,
            instruments,
        ) {
            self.drain_acquire(slot.acquire);
            return Err(e);
        }

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
        if let Err(e) = unsafe {
            self.gpu.device.raw().queue_submit2(
                self.gpu.device.graphics.raw,
                &[submit],
                vk::Fence::null(),
            )
        } {
            self.drain_acquire(slot.acquire);
            return Err(self.gpu.explain(e));
        }
        self.transfer_waited = transfer_value;

        // Registered after the submit, never before: a frame returning between
        // `resolve` and here — an out-of-date acquire is the ordinary case —
        // reset no query and wrote none, yet leaves `frame_index` alone, so the
        // retry's `begin_frame` collects this same slot. An expectation left
        // behind by such a frame reads queries never reset since pool creation
        // on frame 0 (VUID-vkGetQueryPoolResults-None-09401, fatal to the §4.3
        // zero-message report) and, later, frame N-2's ticks under the aborted
        // frame's names. The graph's borrows outlive this point.
        if let Some(timings) = &mut self.timings {
            timing::warn_pass_overflow(passes.len());
            timings.expect(slot_index, passes.iter().map(|p| p.name.to_owned()));
        }

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
            Err(err) => Err(self.gpu.explain(err)),
        }
    }

    /// Consume an acquire semaphore orphaned by a failure between acquire and
    /// submit: a binary semaphore cannot be reset from the host, so the one way
    /// to leave the slot reusable is an empty submit that waits it out.
    ///
    /// Best-effort by design — the plausible causes of the failure being
    /// recovered from (pool OOM, device loss) can fail this submit too, and
    /// then teardown is next anyway; the warning names the semaphore the §4.3
    /// report will otherwise blame.
    fn drain_acquire(&self, semaphore: vk::Semaphore) {
        let wait = [vk::SemaphoreSubmitInfo::default()
            .semaphore(semaphore)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submit = vk::SubmitInfo2::default().wait_semaphore_infos(&wait);
        // SAFETY: queue and semaphore are live; nothing rides along to record.
        let drained = unsafe {
            self.gpu.device.raw().queue_submit2(
                self.gpu.device.graphics.raw,
                &[submit],
                vk::Fence::null(),
            )
        };
        if let Err(e) = drained {
            tracing::warn!(error = ?e, "an orphaned acquire semaphore could not be drained");
        }
    }

    /// Record the frame: the ownership acquires an upload left owing, then the
    /// graph. Everything here arrived resolved (see [`Rhi::execute`]).
    fn record(
        &self,
        pool: vk::CommandPool,
        cmd: vk::CommandBuffer,
        acquires: &[upload::Acquire],
        passes: &[graph::ResolvedPass<'_>],
        backbuffer: Bound,
        instruments: graph::Instruments,
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
            // SAFETY: as above; the backbuffer is this frame's acquired image.
            graph::record(
                &self.gpu.device,
                cmd,
                self.gpu.bindless.set(),
                passes,
                backbuffer,
                instruments,
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
        if let Some(timings) = &mut self.timings {
            timings.destroy(&self.gpu.device);
        }
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

/// What a render graph needs from the context it runs on: the attachments it
/// owns, and which frame slot it is building for.
///
/// **Not a backend abstraction** (§3): that ban is over graphics *APIs*, and
/// both implementors here are the same Vulkan. What varies is where the frame
/// lands — a swapchain or an offscreen target — and this exists so one graph
/// serves the shell, the demos and the golden harness rather than three that
/// drift.
pub trait GraphContext {
    /// Allocate an attachment.
    ///
    /// # Errors
    /// Out of memory, or a format/usage pair the RHI refuses.
    fn create_image(&mut self, desc: &ImageDesc<'_>) -> Result<ImageHandle, RhiError>;

    /// Retire one, behind the timeline where there is one.
    ///
    /// # Errors
    /// The handle is stale.
    fn destroy_image(&mut self, handle: ImageHandle) -> Result<(), RhiError>;

    /// Give an attachment a bindless slot so a later pass can sample it.
    ///
    /// # Errors
    /// The array is full, or the image is a depth attachment.
    fn register_texture(&mut self, handle: ImageHandle) -> Result<TextureIndex, RhiError>;

    /// Which frame-in-flight slot the next frame records into. A transient
    /// shared between two frames in flight is a write racing a read with no
    /// barrier between them, so the pool keys on this.
    fn frame_slot(&self) -> u64;

    /// Whether the backbuffer is handed to a presentation engine. Only a
    /// swapchain image may end a frame in [`Access::Present`]; an offscreen
    /// target put there is an invalid layout for an image no WSI owns.
    fn presents(&self) -> bool;
}

impl GraphContext for Rhi {
    fn create_image(&mut self, desc: &ImageDesc<'_>) -> Result<ImageHandle, RhiError> {
        Rhi::create_image(self, desc)
    }
    fn destroy_image(&mut self, handle: ImageHandle) -> Result<(), RhiError> {
        Rhi::destroy_image(self, handle)
    }
    fn register_texture(&mut self, handle: ImageHandle) -> Result<TextureIndex, RhiError> {
        Rhi::register_texture(self, handle)
    }
    fn frame_slot(&self) -> u64 {
        self.frame_index % FRAMES_IN_FLIGHT
    }
    fn presents(&self) -> bool {
        true
    }
}

impl GraphContext for OffscreenRhi {
    fn create_image(&mut self, desc: &ImageDesc<'_>) -> Result<ImageHandle, RhiError> {
        OffscreenRhi::create_image(self, desc)
    }
    fn destroy_image(&mut self, handle: ImageHandle) -> Result<(), RhiError> {
        OffscreenRhi::destroy_image(self, handle)
    }
    fn register_texture(&mut self, handle: ImageHandle) -> Result<TextureIndex, RhiError> {
        OffscreenRhi::register_texture(self, handle)
    }
    /// One slot: an offscreen execute blocks until the GPU is done with it.
    fn frame_slot(&self) -> u64 {
        0
    }
    fn presents(&self) -> bool {
        false
    }
}

/// Reverse-Z's clear value (§2, Math row): the far plane is `0.0`, and nearer
/// fragments win by comparing *greater*. Passes name this rather than a
/// literal, because a depth attachment cleared the ordinary way renders a
/// scene that is subtly, consistently wrong.
pub const DEPTH_CLEAR: f32 = 0.0;
