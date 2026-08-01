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
use crate::deletion::DeletionQueue;
use crate::device::Device;
use crate::instance::Instance;
use crate::pipeline::PipelineStore;
use crate::resource::{
    BufferDesc, BufferHandle, DeviceAddress, ImageDesc, ImageHandle, ImageUse, Resources,
};
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
    pending_slots: Vec<(u64, PendingSlot)>,
}

impl Gpu {
    /// Bring up the device and everything under it. `surface` is `None` for
    /// offscreen contexts (§4.10). Every failure path unwinds what it built.
    pub fn new(instance: &Instance, surface: Option<&Surface>) -> Result<Self, RhiError> {
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
        let pipelines = match PipelineStore::new(&device) {
            Ok(p) => p,
            Err(e) => {
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
            pending_slots: Vec::new(),
        })
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
        self.resources
            .retire_buffer(handle, &mut self.deletions, after)
    }

    /// Retire an image behind the timeline, queueing its bindless slots for
    /// reuse at the same value.
    pub fn retire_image(&mut self, handle: ImageHandle, after: u64) -> Result<(), RhiError> {
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

    /// Record an image upload into the staging batch. `bytes` is the tightly
    /// packed content of the whole image.
    pub fn upload_image(&mut self, handle: ImageHandle, bytes: &[u8]) -> Result<(), RhiError> {
        let image = self.resources.image(handle)?;
        let (raw, format, extent) = (image.raw, image.format, image.extent);
        self.uploader
            .upload_image(&self.device, raw, format, extent, bytes)?;
        // The upload's barrier leaves it here; the acquire, when there is one,
        // does not change the layout it lands in.
        self.resources.image_mut(handle)?.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        Ok(())
    }

    /// Submit the staging batch and wait for it — the §4.3 "upload now" path.
    pub fn flush_uploads_blocking(&mut self) -> Result<(), RhiError> {
        self.uploader.flush_blocking(&self.device)
    }

    /// Give an image a slot in the global sampled-image array (§4.3: materials
    /// are indices). Update-after-bind, so this is legal mid-flight.
    pub fn register_texture(&mut self, handle: ImageHandle) -> Result<TextureIndex, RhiError> {
        let image = self.resources.image(handle)?;
        if image.format.is_depth() {
            return Err(RhiError::Loader(
                "a depth image is an attachment, not a bindless texture".into(),
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
    /// # Safety
    /// `cmd` must be recording on the graphics family.
    pub unsafe fn record_storage_image_transition(&self, cmd: vk::CommandBuffer, image: vk::Image) {
        let barriers = [vk::ImageMemoryBarrier2::default()
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            )
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )];
        // SAFETY: caller contract — cmd is recording on the graphics family and
        // the image is live.
        unsafe {
            self.device.raw().cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            )
        };
    }

    /// The ownership-transfer barriers the graphics queue owes, taken out.
    pub fn take_acquires(&mut self) -> Vec<Acquire> {
        self.uploader.take_acquires()
    }

    /// A depth attachment sized for `extent`.
    pub fn create_depth(&mut self, extent: (u32, u32)) -> Result<ImageHandle, RhiError> {
        self.create_image(&ImageDesc {
            name: "gg.depth",
            extent,
            format: crate::resource::ImageFormat::Depth32,
            usage: ImageUse::Depth,
        })
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
