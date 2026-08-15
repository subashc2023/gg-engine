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
    /// The present mode asked for — settable, unlike [`Self::want`], and read
    /// by the next recreation.
    want_present: Present,
    /// What it got. [`Present::Vsync`] wherever the driver offered nothing
    /// nearer.
    present: Present,
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

/// When a finished frame reaches the glass (§6 M46).
///
/// [`Output`]'s shape — a request, not a setting, because only [`Present::Vsync`]
/// is guaranteed to exist and the rest are the driver's to offer. Unlike an
/// output contract this one is **not** fixed for the swapchain's life: a colour
/// space decides what the numbers in the backbuffer mean and a present mode does
/// not, so changing it is an ordinary recreation.
///
/// Named for what the player gets rather than for the token behind it: the
/// engine-shaped rule §4.3 states, and here it is load-bearing rather than
/// stylistic, since `MAILBOX` and `IMMEDIATE` say nothing about tearing to
/// anyone who has not read the spec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Present {
    /// Wait for the vertical blank. No tearing, and the frame rate is the
    /// refresh rate. `FIFO` — the one mode the spec requires of every surface,
    /// which is what makes it the fallback for the other two.
    Vsync,
    /// A finished frame replaces the one already queued. No tearing and no
    /// wait, at one more swapchain image and however many frames are drawn and
    /// discarded. `MAILBOX`.
    Fast,
    /// Straight to the glass. The lowest latency there is, and the one mode that
    /// tears. `IMMEDIATE`.
    Immediate,
}

impl Present {
    /// The present modes this asks for, nearest-first, always ending in `FIFO`.
    ///
    /// `Fast` falls back to `Immediate` before `Vsync` and not the other way
    /// round: a player who turned vsync off asked for latency, and the nearest
    /// thing to mailbox on a driver without it is the mode that also does not
    /// wait. `Immediate` does *not* fall back to `Fast` — asking not to tear is
    /// a preference, asking to tear is a latency floor, and quietly buffering it
    /// would be answering a different question.
    fn candidates(self) -> &'static [vk::PresentModeKHR] {
        match self {
            Present::Vsync => &[vk::PresentModeKHR::FIFO],
            Present::Fast => &[
                vk::PresentModeKHR::MAILBOX,
                vk::PresentModeKHR::IMMEDIATE,
                vk::PresentModeKHR::FIFO,
            ],
            Present::Immediate => &[vk::PresentModeKHR::IMMEDIATE, vk::PresentModeKHR::FIFO],
        }
    }

    /// Which mode a granted `VkPresentModeKHR` is, as the caller asked about it.
    /// `FIFO_RELAXED` is [`Present::Vsync`]: it waits like one, and the tear it
    /// permits happens only on a frame that already missed its blank.
    fn of(mode: vk::PresentModeKHR) -> Present {
        match mode {
            vk::PresentModeKHR::MAILBOX => Present::Fast,
            vk::PresentModeKHR::IMMEDIATE => Present::Immediate,
            _ => Present::Vsync,
        }
    }
}

/// The present mode the surface will actually take, nearest to `want`.
///
/// `FIFO` at the end of every candidate list is what makes the `unwrap_or` here
/// a statement rather than a guess — the spec requires it of every surface, so
/// a driver reporting none of the three is one that reported nothing at all.
fn choose_present(modes: &[vk::PresentModeKHR], want: Present) -> vk::PresentModeKHR {
    want.candidates()
        .iter()
        .find(|mode| modes.contains(mode))
        .copied()
        .unwrap_or(vk::PresentModeKHR::FIFO)
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
            want_present: Present::Vsync,
            present: Present::Vsync,
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
        let present = choose_present(
            &surface.present_modes(device.physical())?,
            self.want_present,
        );
        self.present = Present::of(present);

        // Mailbox wants a third image to have somewhere to put the frame it is
        // replacing; with two it degenerates into FIFO's wait with none of its
        // ordering. `min_image_count` is the surface's floor and not the mode's,
        // so this is the one place the mode moves the count.
        let mut min_images = (caps.min_image_count + 1).max(match self.present {
            Present::Fast => 3,
            _ => 0,
        });
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
            .present_mode(present)
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

    pub fn present(&self) -> Present {
        self.present
    }

    /// Ask for a present mode. Takes effect at the next recreation, which is
    /// the caller's to trigger — the swapchain does not decide when it is safe
    /// to retire itself.
    ///
    /// Returns whether this changed the request, so a caller can skip a
    /// recreation nobody asked for: this is polled from a per-tick preference
    /// and is the same value on all but one of those ticks.
    pub fn want_present(&mut self, want: Present) -> bool {
        let moved = self.want_present != want;
        self.want_present = want;
        moved
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

    /// The present mode's half of the same contract, and gated for the same
    /// reason: which mode a driver offers is its business, which of them we take
    /// is ours, and taking the wrong one is a session that tears when the player
    /// asked it not to.
    ///
    /// The asymmetry is the content. `Fast` degrades to `Immediate` because a
    /// player who turned vsync off asked for latency; `Immediate` does **not**
    /// degrade to `Fast`, because a request to tear is a floor and answering it
    /// with a buffered mode is answering a different question.
    #[test]
    fn a_present_mode_degrades_toward_what_was_asked_for_and_never_past_fifo() {
        use vk::PresentModeKHR as M;
        let all = [M::FIFO, M::MAILBOX, M::IMMEDIATE];
        for (want, offers, expect) in [
            (Present::Vsync, &all[..], M::FIFO),
            (Present::Fast, &all[..], M::MAILBOX),
            (Present::Immediate, &all[..], M::IMMEDIATE),
            // The pinned lavapipe's own shape, and the desk's second vendor:
            // no mailbox, so `Fast` is the mode that also does not wait.
            (Present::Fast, &[M::FIFO, M::IMMEDIATE][..], M::IMMEDIATE),
            (Present::Fast, &[M::FIFO][..], M::FIFO),
            (Present::Immediate, &[M::FIFO, M::MAILBOX][..], M::FIFO),
            // A surface that reported nothing is one that reported no FIFO,
            // which the spec does not permit; taking it is the only answer that
            // is not a panic on a driver bug.
            (Present::Fast, &[][..], M::FIFO),
        ] {
            assert_eq!(
                choose_present(offers, want),
                expect,
                "{want:?} of {offers:?}"
            );
        }
    }

    /// What a granted mode is called on the way back out. The round trip has to
    /// hold or `Rhi::present` reports a mode the swapchain is not in — and
    /// `FIFO_RELAXED` is the one the driver may substitute without being asked.
    #[test]
    fn a_granted_mode_names_itself_and_relaxed_fifo_is_still_vsync() {
        for want in [Present::Vsync, Present::Fast, Present::Immediate] {
            let granted = choose_present(
                &[
                    vk::PresentModeKHR::FIFO,
                    vk::PresentModeKHR::MAILBOX,
                    vk::PresentModeKHR::IMMEDIATE,
                ],
                want,
            );
            assert_eq!(Present::of(granted), want, "{want:?} round-trips");
        }
        assert_eq!(
            Present::of(vk::PresentModeKHR::FIFO_RELAXED),
            Present::Vsync,
            "it waits like one; the tear it allows is a frame that already missed"
        );
    }
}
