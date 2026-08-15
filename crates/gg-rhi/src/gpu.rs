//! Everything a rendering context owns below the presentation layer: the
//! device, the allocator's resources, the global bindless set, the staging
//! ring, the pipeline store, and the deletion queue.
//!
//! This exists so [`Rhi`] and [`OffscreenRhi`] share one bring-up ladder and
//! one teardown order rather than two that drift. Both differ only in how
//! pixels leave — a swapchain or a readback buffer — and that difference stays
//! in their own modules.
//!
//! [`Rhi`]: crate::Rhi
//! [`OffscreenRhi`]: crate::OffscreenRhi

use crate::RhiError;
use crate::bindless::{Bindless, StorageImageIndex, TextureIndex};
use crate::crash::{self, Breadcrumbs, Crumbs};
use crate::deletion::DeletionQueue;
use crate::device::Device;
use crate::instance::Instance;
use crate::pipeline::PipelineStore;
use crate::resource::{BufferDesc, BufferHandle, DeviceAddress, ImageDesc, ImageHandle, Resources};
use crate::surface::Surface;
use crate::upload::{Acquire, Uploader};
use ash::vk;

/// A bindless slot waiting to be reusable. Returning one the instant its image
/// is retired would let the next registration overwrite a descriptor an
/// in-flight draw still indexes — update-after-bind permits writes to
/// descriptors pending commands do *not* use, and this is exactly the case
/// where they do.
enum PendingSlot {
    Texture(TextureIndex),
    Storage(StorageImageIndex),
}

pub(crate) struct Gpu {
    pub device: Device,
    pub resources: Resources,
    pub bindless: Bindless,
    pub uploader: Uploader,
    pub pipelines: PipelineStore,
    pub deletions: DeletionQueue,
    crumbs: Breadcrumbs,
    pending_slots: Vec<(u64, PendingSlot)>,
}

impl Gpu {
    /// Bring up the device and everything under it. `surface` is `None` for
    /// offscreen contexts (§4.10). `frame_slots` is the caller's frames in
    /// flight, which the breadcrumb buffer is divided by. `cache_dir` is where
    /// the warm pipeline cache belongs — `None` is the dev tree's `target/`
    /// (§6 M52). Every failure path unwinds what it built.
    pub fn new(
        instance: &Instance,
        surface: Option<&Surface>,
        frame_slots: usize,
        cache_dir: Option<&std::path::Path>,
    ) -> Result<Self, RhiError> {
        let mut device = Device::new(instance, surface)?;

        let mut resources = match Resources::new(&device) {
            Ok(r) => r,
            Err(e) => {
                device.destroy();
                return Err(e);
            }
        };
        let mut bindless = match Bindless::new(&device, &resources) {
            Ok(b) => b,
            Err(e) => {
                resources.destroy(&mut device);
                device.destroy();
                return Err(e);
            }
        };
        let mut uploader = match Uploader::new(&mut device) {
            Ok(u) => u,
            Err(e) => {
                bindless.destroy(&device);
                resources.destroy(&mut device);
                device.destroy();
                return Err(e);
            }
        };
        let mut pipelines = match PipelineStore::new(&device, cache_dir) {
            Ok(p) => p,
            Err(e) => {
                uploader.destroy(&mut device);
                bindless.destroy(&device);
                resources.destroy(&mut device);
                device.destroy();
                return Err(e);
            }
        };

        let crumbs = match Breadcrumbs::new(&mut device, &mut resources, frame_slots) {
            Ok(c) => c,
            Err(e) => {
                pipelines.destroy(&device);
                uploader.destroy(&mut device);
                bindless.destroy(&device);
                resources.destroy(&mut device);
                device.destroy();
                return Err(e);
            }
        };

        Ok(Self {
            device,
            resources,
            bindless,
            uploader,
            pipelines,
            deletions: DeletionQueue::default(),
            crumbs,
            pending_slots: Vec::new(),
        })
    }

    /// Clear `slot`'s breadcrumbs and record the names the frame about to be
    /// recorded will write there (§4.8). Called once per frame, before
    /// recording; the caller has already proven that slot's last frame retired.
    pub fn prepare_crumbs<'a>(
        &mut self,
        slot: usize,
        names: impl Iterator<Item = &'a str>,
    ) -> Result<Crumbs, RhiError> {
        self.crumbs.prepare(&mut self.resources, slot, names)
    }

    /// Turn a Vulkan result into an error, explaining a lost device rather than
    /// forwarding its code (§4.8). Every submit, present and timeline wait goes
    /// through here: `DEVICE_LOST` is the one code whose cause is somewhere else
    /// entirely, and the breadcrumbs are the only record of where.
    pub fn explain(&self, err: vk::Result) -> RhiError {
        if err != vk::Result::ERROR_DEVICE_LOST {
            return RhiError::Vk(err);
        }
        let report = crash::report(
            self.device.marker_mechanism(),
            &self.breadcrumbs(),
            self.device.fault_info().as_ref(),
        );
        // Logged here rather than left to whoever receives the error: this is
        // the one path where the caller may never get to print anything, and
        // the log tail is what a crash report attaches (§4.8).
        tracing::error!(target: "gg::crash", "{report}");
        RhiError::DeviceLost(report)
    }

    /// Re-explain an error a device call already wrapped. A lost device reaches
    /// a caller through whichever call happened to notice — a wait, a submit, a
    /// present — and only this crate can turn that code into the report.
    pub fn detail(&self, err: RhiError) -> RhiError {
        match err {
            RhiError::Vk(code) => self.explain(code),
            other => other,
        }
    }

    /// What the last recorded frame's marks say it reached.
    pub fn breadcrumbs(&self) -> String {
        self.crumbs.report(&self.resources)
    }

    /// Everything the GPU is provably past: deferred destructions and the
    /// bindless slots retired images were holding.
    pub fn collect(&mut self, completed_value: u64) {
        self.deletions.collect(&mut self.device, completed_value);
        let mut keep = Vec::new();
        for (value, slot) in std::mem::take(&mut self.pending_slots) {
            if value > completed_value {
                keep.push((value, slot));
                continue;
            }
            match slot {
                PendingSlot::Texture(i) => self.bindless.release_sampled(i),
                PendingSlot::Storage(i) => self.bindless.release_storage(i),
            }
        }
        self.pending_slots = keep;
    }

    /// Retire a buffer behind the timeline.
    pub fn retire_buffer(&mut self, handle: BufferHandle, after: u64) -> Result<(), RhiError> {
        // Before the retire, and see `Uploader::forget_image_acquires`: an
        // upload the graphics queue has not yet acquired leaves a barrier
        // naming this handle, and the retire is what makes it a dead one.
        let raw = self.resources.buffer(handle)?.raw;
        self.uploader.forget_buffer_acquires(raw);
        self.resources
            .retire_buffer(handle, &mut self.deletions, after)
    }

    /// Retire an image behind the timeline, queueing its bindless slots for
    /// reuse at the same value.
    pub fn retire_image(&mut self, handle: ImageHandle, after: u64) -> Result<(), RhiError> {
        let raw = self.resources.image(handle)?.raw; // as `retire_buffer`
        self.uploader.forget_image_acquires(raw);
        let slots = self
            .resources
            .retire_image(handle, &mut self.deletions, after)?;
        if let Some(i) = slots.texture {
            self.pending_slots.push((after, PendingSlot::Texture(i)));
        }
        if let Some(i) = slots.storage {
            self.pending_slots.push((after, PendingSlot::Storage(i)));
        }
        Ok(())
    }

    /// Allocate a buffer. Shader-readable through
    /// [`Gpu::buffer_address`] and through nothing else (§4.3).
    pub fn create_buffer(&mut self, desc: &BufferDesc<'_>) -> Result<BufferHandle, RhiError> {
        self.resources.create_buffer(&mut self.device, desc)
    }

    /// The GPU pointer a shader reaches this buffer by.
    pub fn buffer_address(&self, handle: BufferHandle) -> Result<DeviceAddress, RhiError> {
        Ok(self.resources.buffer(handle)?.address)
    }

    pub fn create_image(&mut self, desc: &ImageDesc<'_>) -> Result<ImageHandle, RhiError> {
        self.resources.create_image(&mut self.device, desc)
    }

    /// Record a buffer upload into the staging batch.
    pub fn upload_buffer(
        &mut self,
        handle: BufferHandle,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), RhiError> {
        let dst = self.resources.buffer(handle)?;
        let (raw, size) = (dst.raw, dst.size);
        self.uploader
            .upload_buffer(&self.device, raw, size, offset, bytes)
    }

    /// Copy straight into a [`BufferKind::Dynamic`](crate::BufferKind::Dynamic)
    /// buffer's mapping — no staging, no submit.
    pub fn write_buffer(
        &mut self,
        handle: BufferHandle,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), RhiError> {
        let mapping = self.resources.mapped_mut(handle)?;
        let at = offset as usize;
        let end = at
            .checked_add(bytes.len())
            .filter(|end| *end <= mapping.len());
        let Some(end) = end else {
            return Err(RhiError::Loader(format!(
                "write of {} bytes at offset {offset} runs off a {}-byte buffer",
                bytes.len(),
                mapping.len()
            )));
        };
        mapping[at..end].copy_from_slice(bytes);
        Ok(())
    }

    /// What the device is holding right now (§4.8).
    pub fn memory(&self) -> crate::MemoryUse {
        self.resources.memory()
    }

    /// Record one mip level's upload into the staging batch. `bytes` is the
    /// tightly packed content of that level.
    pub fn upload_image(
        &mut self,
        handle: ImageHandle,
        level: u32,
        bytes: &[u8],
    ) -> Result<(), RhiError> {
        let image = self.resources.image(handle)?;
        let (raw, format, extent, levels) =
            (image.raw, image.format, image.extent, image.mip_levels);
        if level >= levels {
            return Err(RhiError::Loader(format!(
                "image upload: level {level} of an image allocated with {levels}"
            )));
        }
        self.uploader
            .upload_image(&self.device, raw, format, extent, level, bytes)
    }

    /// Submit the staging batch and wait for it — the §4.3 "upload now" path.
    pub fn flush_uploads_blocking(&mut self) -> Result<(), RhiError> {
        self.uploader.flush_blocking(&self.device)
    }

    /// Give an image a slot in the global sampled-image array (§4.3: materials
    /// are indices). Update-after-bind, so this is legal mid-flight.
    pub fn register_texture(&mut self, handle: ImageHandle) -> Result<TextureIndex, RhiError> {
        let image = self.resources.image(handle)?;
        // A depth image gets a slot only if it asked for one at creation:
        // `SAMPLED` has to be in the usage flags, and a prepass target that
        // never declared it would fail validation at the descriptor write
        // rather than here, where the name is still in hand.
        if image.format.is_depth() && image.usage != crate::ImageUse::DepthSampled {
            return Err(RhiError::Loader(
                "a depth image is an attachment, not a bindless texture — create it as \
                 `ImageUse::DepthSampled` if a later pass reads it"
                    .into(),
            ));
        }
        let view = image.view;
        let index = self.bindless.register_sampled(&self.device, view)?;
        self.resources.image_mut(handle)?.texture_slot = Some(index);
        Ok(index)
    }

    /// Give an image a slot in the global storage-image array.
    pub fn register_storage_image(
        &mut self,
        handle: ImageHandle,
    ) -> Result<StorageImageIndex, RhiError> {
        let view = self.resources.image(handle)?.view;
        let index = self.bindless.register_storage(&self.device, view)?;
        self.resources.image_mut(handle)?.storage_slot = Some(index);
        Ok(index)
    }

    /// Transition a storage image into the layout the bindless storage array
    /// declares. One-shot on the graphics queue: storage images are written by
    /// shaders rather than uploaded, so nothing else puts them in `GENERAL`.
    ///
    /// The barrier itself is `graph.rs`'s, like every other one (§4.5).
    ///
    /// # Safety
    /// `cmd` must be recording on the graphics family.
    pub unsafe fn record_storage_image_transition(&self, cmd: vk::CommandBuffer, image: vk::Image) {
        // SAFETY: caller contract — cmd is recording on the graphics family and
        // the image is live.
        unsafe {
            crate::graph::one_shot_transition(
                &self.device,
                cmd,
                image,
                vk::ImageAspectFlags::COLOR,
                crate::Access::None,
                crate::Access::StorageReadWrite,
            );
        }
    }

    /// The ownership-transfer barriers the graphics queue owes, taken out.
    pub fn take_acquires(&mut self) -> Vec<Acquire> {
        self.uploader.take_acquires()
    }

    /// Teardown in dependency order, returning the §4.3 leak report. Caller
    /// destroyed its own presentation resources first and waited idle.
    pub fn destroy(&mut self) -> Vec<String> {
        self.device.wait_idle();
        self.deletions.drain_all(&mut self.device);
        self.pipelines.destroy(&self.device);
        self.uploader.destroy(&mut self.device);
        self.bindless.destroy(&self.device);
        self.resources.destroy(&mut self.device);
        let leaks = self.device.leak_report();
        if !leaks.is_empty() {
            tracing::error!(leaks = ?leaks, "GPU allocations leaked (§4.3)");
        }
        self.device.destroy();
        leaks
    }
}
