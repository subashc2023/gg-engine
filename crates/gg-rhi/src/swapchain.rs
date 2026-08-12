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
    /// What the caller asked for, kept so a recreation does not silently
    /// downgrade a display that came back.
    want: Output,
    /// What it got. `want` when the surface offered it, else [`Output::Sdr`].
    output: Output,
    /// True while the surface is zero-extent (minimized): no swapchain
    /// operations are legal, frames are skipped (§4.3).
    suspended: bool,
    /// Bumped on every successful recreation; test-visible.
    generation: u64,
}

/// What the swapchain hands the display, and therefore what the post pass must
/// encode into (§6 M23).
///
/// Not a quality setting with a scale — three genuinely different contracts
/// about what the numbers in the backbuffer *mean*, and a shader that guessed
/// wrong renders a picture that is not merely worse but a different colour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    /// 8-bit sRGB, `SRGB_NONLINEAR`. The **hardware** applies the OETF on write,
    /// so the shader hands it linear values and encodes nothing — which is why
    /// this path is the one every golden (§4.10) is blessed against.
    Sdr,
    /// 10-bit PQ, Rec.2020 primaries — HDR10, `HDR10_ST2084_EXT`. An absolute
    /// encoding: a code value names a luminance in nits rather than a fraction
    /// of whatever the display can do, so the shader must be told what to call
    /// white and how far it may go above it.
    Hdr10,
    /// 16-bit float, linear Rec.709 with values past 1.0 meaning brighter than
    /// SDR white — scRGB, `EXTENDED_SRGB_LINEAR_EXT`. No transfer function and
    /// no quantizer worth dithering, at twice the bandwidth of the other two.
    ScRgb,
}

impl Output {
    /// The format and colour space this asks for, nearest-first.
    ///
    /// Two candidates for HDR10 because the ordering of the ten-bit packing is
    /// the driver's business and both spellings are common; one for the others.
    fn candidates(self) -> &'static [(vk::Format, vk::ColorSpaceKHR)] {
        match self {
            Output::Sdr => &[
                (vk::Format::B8G8R8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
                (vk::Format::R8G8B8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR),
            ],
            Output::Hdr10 => &[
                (
                    vk::Format::A2B10G10R10_UNORM_PACK32,
                    vk::ColorSpaceKHR::HDR10_ST2084_EXT,
                ),
                (
                    vk::Format::A2R10G10B10_UNORM_PACK32,
                    vk::ColorSpaceKHR::HDR10_ST2084_EXT,
                ),
            ],
            Output::ScRgb => &[(
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
            )],
        }
    }
}

/// The surface format the swapchain uses (and pipelines must target), plus the
/// output contract it actually got.
///
/// **Falls back rather than failing**, and the returned [`Output`] is how the
/// caller finds out: an HDR colour space exists only when the display, the
/// compositor *and* the driver all agree, and none of that is knowable before
/// asking. A run that quietly encoded PQ into an sRGB swapchain would be a
/// washed-out grey picture with no error anywhere, so the resolved value is
/// returned rather than assumed and the renderer reconciles `r.hdr` against it.
fn choose(
    formats: &[vk::SurfaceFormatKHR],
    want: Output,
) -> Option<(vk::SurfaceFormatKHR, Output)> {
    // What was asked for, then SDR. The second pass is redundant when SDR was
    // what was asked for and costs two comparisons to keep the loop one shape.
    for mode in [want, Output::Sdr] {
        for &(format, color_space) in mode.candidates() {
            if let Some(found) = formats
                .iter()
                .find(|f| f.format == format && f.color_space == color_space)
            {
                return Some((*found, mode));
            }
        }
    }
    // Whatever it lists first, called SDR: a surface offering none of the six
    // above is not one this engine has a colour management story for, and
    // guessing that an unknown space is HDR is the more damaging guess.
    formats.first().map(|f| (*f, Output::Sdr))
}

fn preferred_format(
    device: &Device,
    surface: &Surface,
    want: Output,
) -> Result<(vk::SurfaceFormatKHR, Output), RhiError> {
    let formats = surface.formats(device.physical())?;
    choose(&formats, want).ok_or_else(|| RhiError::Loader("surface reports no formats".into()))
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
    pub fn new(
        device: &Device,
        surface: &Surface,
        desired: (u32, u32),
        want: Output,
    ) -> Result<Self, RhiError> {
        let mut swapchain = Self {
            raw: vk::SwapchainKHR::null(),
            format: vk::Format::UNDEFINED,
            want,
            output: Output::Sdr,
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

        let (format, output) = preferred_format(device, surface, self.want)?;
        self.output = output;

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

    /// The color format pipelines must target. Valid even while the swapchain
    /// is suspended (zero-extent): it falls back to the surface's preferred
    /// format, which is exactly what the next materialization will pick.
    pub fn format(&self, device: &Device, surface: &Surface) -> Result<vk::Format, RhiError> {
        if self.format != vk::Format::UNDEFINED {
            return Ok(self.format);
        }
        Ok(preferred_format(device, surface, self.want)?.0.format)
    }

    /// The output contract the swapchain actually got — what the post pass must
    /// encode into, and never assumed from what was asked for.
    pub fn output(&self) -> Output {
        self.output
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn offered(pairs: &[(vk::Format, vk::ColorSpaceKHR)]) -> Vec<vk::SurfaceFormatKHR> {
        pairs
            .iter()
            .map(|&(format, color_space)| vk::SurfaceFormatKHR {
                format,
                color_space,
            })
            .collect()
    }

    const SDR: (vk::Format, vk::ColorSpaceKHR) =
        (vk::Format::B8G8R8A8_SRGB, vk::ColorSpaceKHR::SRGB_NONLINEAR);
    const PQ: (vk::Format, vk::ColorSpaceKHR) = (
        vk::Format::A2B10G10R10_UNORM_PACK32,
        vk::ColorSpaceKHR::HDR10_ST2084_EXT,
    );
    const SCRGB: (vk::Format, vk::ColorSpaceKHR) = (
        vk::Format::R16G16B16A16_SFLOAT,
        vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT,
    );

    /// The whole of the HDR contract that can be gated windowless (§1.5), and it
    /// is the half worth gating: what a surface *offers* is the driver's and the
    /// display's to say, but which of them is picked is ours, and picking wrong
    /// is a picture that is a different colour rather than a picture that errors.
    #[test]
    fn an_hdr_surface_is_taken_when_asked_for_and_never_otherwise() {
        let everything = offered(&[SDR, PQ, SCRGB]);
        for (want, format) in [
            (Output::Sdr, SDR),
            (Output::Hdr10, PQ),
            (Output::ScRgb, SCRGB),
        ] {
            let (got, mode) = choose(&everything, want).expect("a surface offering three formats");
            assert_eq!(mode, want, "asked for {want:?}");
            assert_eq!((got.format, got.color_space), format, "asked for {want:?}");
        }
    }

    /// The case every desk that is not in HDR mode takes, and the reason
    /// `choose` returns the mode rather than the caller assuming it: the surface
    /// simply does not list an HDR pair, and the *only* correct outcome is a
    /// working SDR picture plus an honest report of what happened.
    #[test]
    fn asking_for_hdr_on_an_sdr_surface_falls_back_and_says_so() {
        let sdr_only = offered(&[SDR]);
        for want in [Output::Hdr10, Output::ScRgb] {
            let (got, mode) = choose(&sdr_only, want).expect("an sRGB surface");
            assert_eq!(mode, Output::Sdr, "asked for {want:?} on an SDR surface");
            assert_eq!((got.format, got.color_space), SDR);
        }
    }

    /// A surface listing something we have no colour management story for is
    /// still a surface. It is taken, and it is called SDR — the safer of the two
    /// guesses, since encoding PQ into a space that is not PQ is a washed-out
    /// grey frame with no error anywhere to explain it.
    #[test]
    fn an_unknown_colour_space_is_used_and_called_sdr() {
        let odd = offered(&[(
            vk::Format::R5G6B5_UNORM_PACK16,
            vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT,
        )]);
        let (got, mode) = choose(&odd, Output::Hdr10).expect("a surface with one odd format");
        assert_eq!(mode, Output::Sdr);
        assert_eq!(got.format, vk::Format::R5G6B5_UNORM_PACK16);
        assert_eq!(choose(&[], Output::Sdr), None, "no formats is not a choice");
    }
}
