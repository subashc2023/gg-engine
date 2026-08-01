//! The swapchain (§4.3): recreation is a normal event — resize, `SUBOPTIMAL`,
//! out-of-date — not an error path. Retired swapchains and their views go
//! through the deletion queue keyed to the frame that last used them; a
//! zero-extent surface (minimized) suspends rendering instead of recreating.

use crate::RhiError;
use crate::deletion::{Deferred, DeletionQueue};
use crate::device::Device;
use crate::surface::Surface;
use ash::vk;

/// The swapchain plus its per-image state.
pub(crate) struct Swapchain {
    raw: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    views: Vec<vk::ImageView>,
    images: Vec<vk::Image>,
    /// Per-image "render finished" binary semaphores — signaled by the frame's
    /// submit, waited by present. Per image, not per frame: present may hold
    /// its wait until the image is next reacquired.
    render_done: Vec<vk::Semaphore>,
    /// True while the surface is zero-extent (minimized): no swapchain
    /// operations are legal, frames are skipped (§4.3).
    suspended: bool,
    /// Bumped on every successful recreation; test-visible.
    generation: u64,
}

/// What [`Swapchain::acquire`] produced.
pub(crate) enum Acquired {
    /// An image index, plus whether the WSI flagged the swapchain suboptimal.
    Image { index: u32, suboptimal: bool },
    /// The swapchain no longer matches the surface; recreate and retry.
    OutOfDate,
}

impl Swapchain {
    /// Create the first swapchain. A zero-extent surface yields a suspended
    /// swapchain that materializes on the first nonzero resize.
    pub fn new(device: &Device, surface: &Surface, desired: (u32, u32)) -> Result<Self, RhiError> {
        let mut swapchain = Self {
            raw: vk::SwapchainKHR::null(),
            format: vk::Format::UNDEFINED,
            extent: vk::Extent2D::default(),
            views: Vec::new(),
            images: Vec::new(),
            render_done: Vec::new(),
            suspended: true,
            generation: 0,
        };
        // First creation retires nothing, so no deletion queue is involved.
        swapchain.recreate(device, surface, desired, &mut DeletionQueue::default(), 0)?;
        Ok(swapchain)
    }

    /// Recreate against current surface capabilities, clamping `desired` where
    /// the surface dictates extent. Retires the old swapchain (if any) through
    /// `deletions` at `retire_value`. Returns `false` when the surface is
    /// zero-extent and the swapchain suspended instead.
    pub fn recreate(
        &mut self,
        device: &Device,
        surface: &Surface,
        desired: (u32, u32),
        deletions: &mut DeletionQueue,
        retire_value: u64,
    ) -> Result<bool, RhiError> {
        let caps = surface.capabilities(device.physical())?;
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: desired.0.clamp(
                    caps.min_image_extent.width,
                    caps.max_image_extent.width.max(caps.min_image_extent.width),
                ),
                height: desired.1.clamp(
                    caps.min_image_extent.height,
                    caps.max_image_extent
                        .height
                        .max(caps.min_image_extent.height),
                ),
            }
        };
        if extent.width == 0 || extent.height == 0 {
            self.suspended = true;
            return Ok(false);
        }

        let formats = surface.formats(device.physical())?;
        let format = formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| formats.first())
            .copied()
            .ok_or_else(|| RhiError::Loader("surface reports no formats".into()))?;

        let mut min_images = caps.min_image_count + 1;
        if caps.max_image_count > 0 {
            min_images = min_images.min(caps.max_image_count);
        }
        let composite = if caps
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
        {
            vk::CompositeAlphaFlagsKHR::OPAQUE
        } else {
            // First supported bit; the WSI must offer at least one.
            vk::CompositeAlphaFlagsKHR::from_raw(
                1 << caps.supported_composite_alpha.as_raw().trailing_zeros(),
            )
        };

        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.raw())
            .min_image_count(min_images)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(caps.current_transform)
            .composite_alpha(composite)
            // FIFO: the one mode the spec guarantees; latency tuning is a
            // later milestone's problem, correctness is this one's.
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(self.raw);

        // SAFETY: surface is live; old_swapchain (possibly null) is retired,
        // never used again, and destroyed via the deletion queue below.
        let new_raw = unsafe { device.swapchain_fns().create_swapchain(&info, None) }
            .map_err(RhiError::Vk)?;

        // Retire the outgoing swapchain at the caller's timeline value.
        if self.raw != vk::SwapchainKHR::null() {
            for view in self.views.drain(..) {
                deletions.defer(retire_value, Deferred::ImageView(view));
            }
            for semaphore in self.render_done.drain(..) {
                deletions.defer(retire_value, Deferred::Semaphore(semaphore));
            }
            deletions.defer(retire_value, Deferred::Swapchain(self.raw));
        }

        self.raw = new_raw;
        self.format = format.format;
        self.extent = extent;
        self.generation += 1;
        self.suspended = false;
        device.set_name(new_raw, &format!("gg.swapchain.gen{}", self.generation));

        // SAFETY: swapchain is live.
        self.images = unsafe { device.swapchain_fns().get_swapchain_images(new_raw) }
            .map_err(RhiError::Vk)?;
        self.views = Vec::with_capacity(self.images.len());
        self.render_done = Vec::with_capacity(self.images.len());
        for (i, &image) in self.images.iter().enumerate() {
            device.set_name(
                image,
                &format!("gg.swapchain.gen{}.image{i}", self.generation),
            );
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format.format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );
            // SAFETY: image belongs to the live swapchain.
            let view = unsafe { device.raw().create_image_view(&view_info, None) }
                .map_err(RhiError::Vk)?;
            device.set_name(
                view,
                &format!("gg.swapchain.gen{}.view{i}", self.generation),
            );
            self.views.push(view);

            let semaphore_info = vk::SemaphoreCreateInfo::default();
            // SAFETY: device is live.
            let semaphore = unsafe { device.raw().create_semaphore(&semaphore_info, None) }
                .map_err(RhiError::Vk)?;
            device.set_name(
                semaphore,
                &format!("gg.swapchain.gen{}.render_done{i}", self.generation),
            );
            self.render_done.push(semaphore);
        }
        Ok(true)
    }

    /// Acquire the next image, signaling `semaphore` when it is usable.
    /// Out-of-date is a value, not an error — recreation is a normal event.
    pub fn acquire(&self, device: &Device, semaphore: vk::Semaphore) -> Result<Acquired, RhiError> {
        debug_assert!(!self.suspended, "acquire on a suspended swapchain");
        // SAFETY: swapchain and semaphore are live; no fence; the semaphore is
        // unsignaled (its previous wait completed — Rhi's timeline pacing).
        match unsafe {
            device.swapchain_fns().acquire_next_image(
                self.raw,
                u64::MAX,
                semaphore,
                vk::Fence::null(),
            )
        } {
            Ok((index, suboptimal)) => Ok(Acquired::Image { index, suboptimal }),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(Acquired::OutOfDate),
            Err(err) => Err(RhiError::Vk(err)),
        }
    }

    pub fn suspended(&self) -> bool {
        self.suspended
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn extent(&self) -> (u32, u32) {
        (self.extent.width, self.extent.height)
    }

    pub fn raw(&self) -> vk::SwapchainKHR {
        self.raw
    }

    pub fn view(&self, index: u32) -> vk::ImageView {
        self.views[index as usize]
    }

    pub fn image(&self, index: u32) -> vk::Image {
        self.images[index as usize]
    }

    pub fn render_done(&self, index: u32) -> vk::Semaphore {
        self.render_done[index as usize]
    }

    /// Teardown: destroy everything now. Caller waited the device idle.
    pub fn destroy(&mut self, device: &Device) {
        // SAFETY (all): handles belong to this device; GPU idle per contract.
        unsafe {
            for view in self.views.drain(..) {
                device.raw().destroy_image_view(view, None);
            }
            for semaphore in self.render_done.drain(..) {
                device.raw().destroy_semaphore(semaphore, None);
            }
            if self.raw != vk::SwapchainKHR::null() {
                device.swapchain_fns().destroy_swapchain(self.raw, None);
                self.raw = vk::SwapchainKHR::null();
            }
        }
    }
}
