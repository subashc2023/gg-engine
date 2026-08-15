//! The presentation surface — the seam between `gg-platform`'s window (raw
//! handles only; winit never appears in this crate's graph) and the WSI.

use crate::RhiError;
use crate::instance::Instance;
use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

/// A live `VkSurfaceKHR` plus the extension functions that query it.
pub struct Surface {
    fns: ash::khr::surface::Instance,
    raw: vk::SurfaceKHR,
}

impl Surface {
    /// Create the surface for a window. The window must outlive the surface —
    /// upheld by `Rhi`'s teardown order.
    pub fn new(
        instance: &Instance,
        window: &(impl HasWindowHandle + HasDisplayHandle),
    ) -> Result<Self, RhiError> {
        let display = window
            .display_handle()
            .map_err(|e| RhiError::Loader(format!("no display handle: {e}")))?;
        let window_handle = window
            .window_handle()
            .map_err(|e| RhiError::Loader(format!("no window handle: {e}")))?;
        // SAFETY: both handles come from a live window; instance is live.
        let raw = unsafe {
            ash_window::create_surface(
                instance.entry(),
                instance.raw(),
                display.as_raw(),
                window_handle.as_raw(),
                None,
            )
        }
        .map_err(RhiError::Vk)?;
        Ok(Self {
            fns: ash::khr::surface::Instance::new(instance.entry(), instance.raw()),
            raw,
        })
    }

    /// Create a headless surface — a presentable surface with no window behind
    /// it. Its capabilities report `current_extent` as "you decide", so a
    /// recreation gate drives the extent itself rather than being told it by a
    /// window manager (§6 M12).
    ///
    /// The instance must have been created with
    /// [`Presentation::Headless`](crate::instance::Presentation::Headless).
    pub fn headless(instance: &Instance) -> Result<Self, RhiError> {
        let fns = ash::ext::headless_surface::Instance::new(instance.entry(), instance.raw());
        let info = vk::HeadlessSurfaceCreateInfoEXT::default();
        // SAFETY: instance is live and enabled VK_EXT_headless_surface; the
        // create-info holds no pointers.
        let raw = unsafe { fns.create_headless_surface(&info, None) }.map_err(RhiError::Vk)?;
        Ok(Self {
            fns: ash::khr::surface::Instance::new(instance.entry(), instance.raw()),
            raw,
        })
    }

    pub(crate) fn raw(&self) -> vk::SurfaceKHR {
        self.raw
    }

    /// Whether `family` on `pd` can present to this surface.
    pub(crate) fn supports_present(&self, pd: vk::PhysicalDevice, family: u32) -> bool {
        // SAFETY: surface and pd are live.
        unsafe {
            self.fns
                .get_physical_device_surface_support(pd, family, self.raw)
        }
        .unwrap_or(false)
    }

    pub(crate) fn capabilities(
        &self,
        pd: vk::PhysicalDevice,
    ) -> Result<vk::SurfaceCapabilitiesKHR, RhiError> {
        // SAFETY: surface and pd are live.
        unsafe {
            self.fns
                .get_physical_device_surface_capabilities(pd, self.raw)
        }
        .map_err(RhiError::Vk)
    }

    pub(crate) fn formats(
        &self,
        pd: vk::PhysicalDevice,
    ) -> Result<Vec<vk::SurfaceFormatKHR>, RhiError> {
        // SAFETY: surface and pd are live.
        unsafe { self.fns.get_physical_device_surface_formats(pd, self.raw) }.map_err(RhiError::Vk)
    }

    pub(crate) fn present_modes(
        &self,
        pd: vk::PhysicalDevice,
    ) -> Result<Vec<vk::PresentModeKHR>, RhiError> {
        // SAFETY: surface and pd are live.
        unsafe {
            self.fns
                .get_physical_device_surface_present_modes(pd, self.raw)
        }
        .map_err(RhiError::Vk)
    }

    pub(crate) fn destroy(&mut self) {
        // SAFETY: no swapchain refers to this surface anymore (teardown order).
        unsafe { self.fns.destroy_surface(self.raw, None) };
    }
}
