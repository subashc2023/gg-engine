//! Device bring-up (§4.3): every physical device scored and logged with every
//! missing feature — the startup log is a diagnostic document — required
//! 1.2/1.3 features asserted or we exit with a precise report (§1.10), queues
//! selected (graphics+present, dedicated transfer when the hardware has one),
//! one timeline semaphore per queue, and the allocator that owns every byte of
//! GPU memory and tattles on leaks at shutdown.

use crate::RhiError;
use crate::instance::Instance;
use crate::surface::Surface;
use ash::vk;

/// One queue: family, handle, and its monotonic timeline semaphore (§4.3:
/// timeline semaphores only, plus WSI binaries where presentation demands).
pub(crate) struct Queue {
    pub family: u32,
    pub raw: vk::Queue,
    pub timeline: vk::Semaphore,
}

/// What device selection saw and decided — kept, not just logged, so tests
/// and bug reports can assert on it.
#[derive(Clone, Debug)]
pub struct DeviceReport {
    /// Every enumerated device, in enumeration order.
    pub candidates: Vec<Candidate>,
    /// Name of the chosen device.
    pub chosen: String,
    /// Chosen device's Vulkan version (major, minor, patch).
    pub api_version: (u32, u32, u32),
    /// Whether a dedicated transfer family existed (lavapipe has one family
    /// total, so `false` there — recorded, not papered over).
    pub transfer_dedicated: bool,
}

/// One scored candidate from device selection.
#[derive(Clone, Debug)]
pub struct Candidate {
    /// Driver-reported device name.
    pub name: String,
    /// Human-readable device type.
    pub device_type: &'static str,
    /// Selection score; higher wins. Rejected devices score 0.
    pub score: u32,
    /// Required capabilities this device lacks (§4.3's precise report).
    pub missing: Vec<&'static str>,
}

/// The logical device, its queues, and the allocator.
pub struct Device {
    physical: vk::PhysicalDevice,
    raw: ash::Device,
    swapchain_fns: ash::khr::swapchain::Device,
    #[cfg(feature = "validation")]
    debug_fns: ash::ext::debug_utils::Device,
    pub(crate) graphics: Queue,
    pub(crate) transfer: Queue,
    allocator: Option<gpu_allocator::vulkan::Allocator>,
    report: DeviceReport,
}

/// The §4.3 required-feature list. Names match `xtask probe`'s table so a
/// bring-up failure and a probe failure read as the same fact.
fn missing_features(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    api_version: u32,
    surface: Option<&Surface>,
    families: &[vk::QueueFamilyProperties],
) -> Vec<&'static str> {
    let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut f2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut f11)
        .push_next(&mut f12)
        .push_next(&mut f13);
    // SAFETY: pd comes from this live instance; structs are default-init.
    unsafe { instance.get_physical_device_features2(pd, &mut f2) };

    // Offscreen contexts (§4.10) have no surface: any graphics family will
    // do, and the row says so honestly.
    let graphics_present = (0..families.len() as u32).any(|i| {
        families[i as usize]
            .queue_flags
            .contains(vk::QueueFlags::GRAPHICS)
            && surface.is_none_or(|s| s.supports_present(pd, i))
    });
    let queue_row = if surface.is_some() {
        "graphics+present queue family"
    } else {
        "graphics queue family"
    };

    let rows: [(&'static str, bool); 16] = [
        (
            "Vulkan >= 1.3",
            api_version >= vk::make_api_version(0, 1, 3, 0),
        ),
        (queue_row, graphics_present),
        // Slang's vertex SPIR-V declares DrawParameters (BaseVertex) — a 1.1
        // core feature every M2+ pipeline rides on.
        (
            "shaderDrawParameters",
            f11.shader_draw_parameters == vk::TRUE,
        ),
        ("timelineSemaphore", f12.timeline_semaphore == vk::TRUE),
        ("bufferDeviceAddress", f12.buffer_device_address == vk::TRUE),
        ("descriptorIndexing", f12.descriptor_indexing == vk::TRUE),
        (
            "runtimeDescriptorArray",
            f12.runtime_descriptor_array == vk::TRUE,
        ),
        (
            "shaderSampledImageArrayNonUniformIndexing",
            f12.shader_sampled_image_array_non_uniform_indexing == vk::TRUE,
        ),
        (
            "shaderStorageBufferArrayNonUniformIndexing",
            f12.shader_storage_buffer_array_non_uniform_indexing == vk::TRUE,
        ),
        (
            "descriptorBindingPartiallyBound",
            f12.descriptor_binding_partially_bound == vk::TRUE,
        ),
        (
            "descriptorBindingSampledImageUpdateAfterBind",
            f12.descriptor_binding_sampled_image_update_after_bind == vk::TRUE,
        ),
        (
            "descriptorBindingStorageBufferUpdateAfterBind",
            f12.descriptor_binding_storage_buffer_update_after_bind == vk::TRUE,
        ),
        (
            "descriptorBindingUpdateUnusedWhilePending",
            f12.descriptor_binding_update_unused_while_pending == vk::TRUE,
        ),
        ("dynamicRendering", f13.dynamic_rendering == vk::TRUE),
        ("synchronization2", f13.synchronization2 == vk::TRUE),
        ("maintenance4", f13.maintenance4 == vk::TRUE),
    ];
    rows.iter().filter(|(_, ok)| !ok).map(|(n, _)| *n).collect()
}

fn device_type_name(t: vk::PhysicalDeviceType) -> (&'static str, u32) {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => ("discrete", 1000),
        vk::PhysicalDeviceType::INTEGRATED_GPU => ("integrated", 500),
        vk::PhysicalDeviceType::VIRTUAL_GPU => ("virtual", 250),
        vk::PhysicalDeviceType::CPU => ("cpu", 100),
        _ => ("other", 50),
    }
}

impl Device {
    /// Select and create the device. `surface` is `None` for offscreen
    /// contexts (§4.10), which need no present support. Logs every candidate;
    /// on failure the error carries the full report (§1.10).
    pub fn new(instance: &Instance, surface: Option<&Surface>) -> Result<Self, RhiError> {
        let inst = instance.raw();
        // SAFETY: instance is live.
        let physical_devices =
            unsafe { inst.enumerate_physical_devices() }.map_err(RhiError::Vk)?;

        let mut candidates = Vec::new();
        let mut chosen: Option<(vk::PhysicalDevice, u32, usize)> = None;
        for pd in physical_devices {
            // SAFETY: pd from the live instance.
            let props = unsafe { inst.get_physical_device_properties(pd) };
            // SAFETY: as above.
            let families = unsafe { inst.get_physical_device_queue_family_properties(pd) };
            let name = props
                .device_name_as_c_str()
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "<unnamed>".into());
            let (device_type, type_score) = device_type_name(props.device_type);
            let missing = missing_features(inst, pd, props.api_version, surface, &families);
            let score = if missing.is_empty() { type_score } else { 0 };
            tracing::info!(
                device = %name,
                r#type = device_type,
                score,
                missing = ?missing,
                "physical device"
            );
            if missing.is_empty() && chosen.map(|(_, best, _)| score > best).unwrap_or(true) {
                chosen = Some((pd, score, candidates.len()));
            }
            candidates.push(Candidate {
                name,
                device_type,
                score,
                missing,
            });
        }

        let Some((pd, _, chosen_idx)) = chosen else {
            let mut report = String::from(
                "no Vulkan device satisfies the §4.3 requirements — per-device report:",
            );
            for c in &candidates {
                report.push_str(&format!(
                    "\n  {} ({}): missing {}",
                    c.name,
                    c.device_type,
                    if c.missing.is_empty() {
                        "nothing (outscored)".to_string()
                    } else {
                        c.missing.join(", ")
                    }
                ));
            }
            return Err(RhiError::NoSuitableDevice(report));
        };

        // SAFETY: pd from the live instance.
        let props = unsafe { inst.get_physical_device_properties(pd) };
        // SAFETY: as above.
        let families = unsafe { inst.get_physical_device_queue_family_properties(pd) };

        // Queues: graphics+present is required (selection proved it exists);
        // transfer prefers a family that is neither graphics nor compute
        // (a real DMA queue), falls back to any other transfer-capable
        // family, and finally shares the graphics queue — recorded honestly,
        // because lavapipe has exactly one family (§4.3).
        let graphics_family = (0..families.len() as u32)
            .find(|&i| {
                families[i as usize]
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS)
                    && surface.is_none_or(|s| s.supports_present(pd, i))
            })
            .ok_or_else(|| RhiError::NoSuitableDevice("graphics family vanished".into()))?;
        let transfer_family = (0..families.len() as u32)
            .filter(|&i| i != graphics_family)
            .filter(|&i| {
                families[i as usize]
                    .queue_flags
                    .contains(vk::QueueFlags::TRANSFER)
            })
            .min_by_key(|&i| {
                let flags = families[i as usize].queue_flags;
                u32::from(flags.contains(vk::QueueFlags::GRAPHICS))
                    + u32::from(flags.contains(vk::QueueFlags::COMPUTE))
            });
        let transfer_dedicated = transfer_family.is_some();

        let priorities = [1.0f32];
        let mut queue_infos = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(graphics_family)
                .queue_priorities(&priorities),
        ];
        if let Some(tf) = transfer_family {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(tf)
                    .queue_priorities(&priorities),
            );
        }

        // Enable exactly the asserted feature set — nothing speculative.
        let mut f11 = vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
        let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
            .timeline_semaphore(true)
            .buffer_device_address(true)
            .descriptor_indexing(true)
            .runtime_descriptor_array(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_storage_buffer_array_non_uniform_indexing(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_update_unused_while_pending(true);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true)
            .maintenance4(true);
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f11)
            .push_next(&mut f12)
            .push_next(&mut f13);

        let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut features2);

        // SAFETY: all pointed-to arrays outlive the call; pd is valid.
        let raw = unsafe { inst.create_device(pd, &device_info, None) }.map_err(RhiError::Vk)?;
        let swapchain_fns = ash::khr::swapchain::Device::new(inst, &raw);
        #[cfg(feature = "validation")]
        let debug_fns = ash::ext::debug_utils::Device::new(inst, &raw);

        // SAFETY: families/indices were created above.
        let graphics_queue = unsafe { raw.get_device_queue(graphics_family, 0) };
        let transfer_queue = match transfer_family {
            // SAFETY: as above.
            Some(tf) => unsafe { raw.get_device_queue(tf, 0) },
            None => graphics_queue,
        };

        let make_timeline = |device: &ash::Device| -> Result<vk::Semaphore, RhiError> {
            let mut type_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let info = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
            // SAFETY: device is live; info valid.
            unsafe { device.create_semaphore(&info, None) }.map_err(RhiError::Vk)
        };
        let graphics_timeline = make_timeline(&raw)?;
        let transfer_timeline = make_timeline(&raw)?;

        let allocator =
            gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
                instance: inst.clone(),
                device: raw.clone(),
                physical_device: pd,
                debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
                buffer_device_address: true,
                allocation_sizes: gpu_allocator::AllocationSizes::default(),
            })
            .map_err(|e| RhiError::Allocator(e.to_string()))?;

        let api = props.api_version;
        let report = DeviceReport {
            chosen: candidates[chosen_idx].name.clone(),
            candidates,
            api_version: (
                vk::api_version_major(api),
                vk::api_version_minor(api),
                vk::api_version_patch(api),
            ),
            transfer_dedicated,
        };
        tracing::info!(
            chosen = %report.chosen,
            api = ?report.api_version,
            transfer_dedicated,
            "device created"
        );

        let device = Self {
            physical: pd,
            raw,
            swapchain_fns,
            #[cfg(feature = "validation")]
            debug_fns,
            graphics: Queue {
                family: graphics_family,
                raw: graphics_queue,
                timeline: graphics_timeline,
            },
            transfer: Queue {
                family: transfer_family.unwrap_or(graphics_family),
                raw: transfer_queue,
                timeline: transfer_timeline,
            },
            allocator: Some(allocator),
            report,
        };
        device.set_name(graphics_timeline, "gg.graphics.timeline");
        device.set_name(transfer_timeline, "gg.transfer.timeline");
        Ok(device)
    }

    /// What selection saw and decided.
    pub fn report(&self) -> &DeviceReport {
        &self.report
    }

    pub(crate) fn raw(&self) -> &ash::Device {
        &self.raw
    }

    pub(crate) fn physical(&self) -> vk::PhysicalDevice {
        self.physical
    }

    pub(crate) fn swapchain_fns(&self) -> &ash::khr::swapchain::Device {
        &self.swapchain_fns
    }

    /// Name any Vulkan object (§1.6). Required at creation sites — the
    /// signature always exists; the call compiles to nothing without the
    /// `validation` feature, which is how names unbolt in dist (§2).
    pub(crate) fn set_name<T: vk::Handle>(&self, handle: T, name: &str) {
        #[cfg(feature = "validation")]
        {
            if let Ok(name) = std::ffi::CString::new(name) {
                let info = vk::DebugUtilsObjectNameInfoEXT::default()
                    .object_handle(handle)
                    .object_name(&name);
                // SAFETY: device is live; handle belongs to it.
                let _ = unsafe { self.debug_fns.set_debug_utils_object_name(&info) };
            }
        }
        #[cfg(not(feature = "validation"))]
        {
            let _ = (handle, name);
        }
    }

    /// Block until the graphics timeline reaches `value` (§4.3 frame pacing).
    pub(crate) fn wait_graphics_timeline(&self, value: u64) -> Result<(), RhiError> {
        let semaphores = [self.graphics.timeline];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: semaphore is live and a timeline.
        unsafe { self.raw.wait_semaphores(&info, u64::MAX) }.map_err(RhiError::Vk)
    }

    /// The graphics timeline's completed value — the deletion queue's clock.
    pub(crate) fn graphics_timeline_value(&self) -> Result<u64, RhiError> {
        // SAFETY: semaphore is live and a timeline.
        unsafe { self.raw.get_semaphore_counter_value(self.graphics.timeline) }
            .map_err(RhiError::Vk)
    }

    /// Wait the graphics queue idle. Swapchain recreation only: presentation
    /// holds its wait semaphores until the present completes, and WSI offers
    /// no signal to key their retirement to (short of swapchain-maintenance1,
    /// a §7-scoped upgrade) — so the rare structural event pays a queue stall
    /// and the per-frame path never does.
    pub(crate) fn wait_graphics_idle(&self) {
        // SAFETY: queue is live; no other thread submits (M1 is single-threaded).
        let _ = unsafe { self.raw.queue_wait_idle(self.graphics.raw) };
    }

    /// Wait for the whole device — teardown and swapchain-suspend paths only;
    /// the frame path paces on the timeline, never on idle.
    pub(crate) fn wait_idle(&self) {
        // SAFETY: device is live.
        let _ = unsafe { self.raw.device_wait_idle() };
    }

    /// Allocate GPU memory. Every byte in the engine flows through here so
    /// the shutdown leak report is total (§4.3).
    pub(crate) fn allocate(
        &mut self,
        desc: &gpu_allocator::vulkan::AllocationCreateDesc<'_>,
    ) -> Result<gpu_allocator::vulkan::Allocation, RhiError> {
        self.allocator
            .as_mut()
            .ok_or_else(|| RhiError::Allocator("allocator already torn down".into()))?
            .allocate(desc)
            .map_err(|e| RhiError::Allocator(e.to_string()))
    }

    /// Return an allocation to the allocator.
    pub(crate) fn free(
        &mut self,
        allocation: gpu_allocator::vulkan::Allocation,
    ) -> Result<(), RhiError> {
        self.allocator
            .as_mut()
            .ok_or_else(|| RhiError::Allocator("allocator already torn down".into()))?
            .free(allocation)
            .map_err(|e| RhiError::Allocator(e.to_string()))
    }

    /// Live allocations at shutdown, named — the §4.3 leak report. CI fails
    /// on a nonzero count.
    pub(crate) fn leak_report(&self) -> Vec<String> {
        self.allocator
            .as_ref()
            .map(|a| {
                a.generate_report()
                    .allocations
                    .iter()
                    .map(|alloc| format!("{} ({} bytes)", alloc.name, alloc.size))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Tear down; call after all child resources are destroyed.
    pub(crate) fn destroy(&mut self) {
        self.wait_idle();
        drop(self.allocator.take());
        // SAFETY: semaphores belong to this device; GPU is idle.
        unsafe {
            self.raw.destroy_semaphore(self.graphics.timeline, None);
            self.raw.destroy_semaphore(self.transfer.timeline, None);
            self.raw.destroy_device(None);
        }
    }
}
