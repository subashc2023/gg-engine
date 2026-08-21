//! Device bring-up (§4.3): every physical device scored and logged with every
//! missing feature — the startup log is a diagnostic document — required
//! 1.2/1.3 features asserted or we exit with a precise report (§1.10), queues
//! selected (graphics+present, dedicated transfer when the hardware has one),
//! one timeline semaphore per queue, and the allocator that owns every byte of
//! GPU memory and tattles on leaks at shutdown.

use crate::RhiError;
use crate::crash::{Fault, FaultAddress, FaultVendor, address_kind};
use crate::instance::Instance;
use crate::surface::Surface;
use ash::vk;
use std::ffi::CStr;

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
    /// The *driver* behind the chosen device, name and version, e.g.
    /// `("llvmpipe", "Mesa 26.1.3 (LLVM 21.1.0)")` or `("NVIDIA", "580.97.0")`.
    ///
    /// Separate from [`Self::chosen`] because a device name is not a build: two
    /// Mesa releases are the same `llvmpipe` and do not render the same picture,
    /// which is what left this tree's two lavapipe reference sets two minor
    /// versions apart with nothing recording it (§6 M81). The version is the
    /// driver's own prose rather than the encoded `driverVersion`, which each
    /// vendor packs its own way and no reader can compare by eye.
    pub driver: (String, String),
    /// Whether a dedicated transfer family existed (lavapipe has one family
    /// total, so `false` there — recorded, not papered over).
    pub transfer_dedicated: bool,
    /// Every MSAA count this device advertises for **both** a color and a depth
    /// framebuffer (§6 M21), ascending and always starting at 1×.
    ///
    /// The intersection of the two masks, because the scene pass needs the pair
    /// at one count: a device offering 8× color and 4× depth can only do 4×.
    /// The whole set rather than a maximum, because support is *membership* —
    /// a driver may advertise 1, 4 and 8 and not 2, and clamping to a maximum
    /// would hand it a count it never claimed.
    pub samples: Vec<crate::Samples>,
}

impl DeviceReport {
    /// Whether this device does `samples`.
    #[must_use]
    pub fn supports_samples(&self, samples: crate::Samples) -> bool {
        self.samples.contains(&samples)
    }

    /// The largest count it does, which is at worst 1×.
    #[must_use]
    pub fn max_samples(&self) -> crate::Samples {
        self.samples.last().copied().unwrap_or_default()
    }

    /// `asked` cut down to the largest count this device actually does. The
    /// one place a request is reduced rather than refused: something has to be
    /// drawn, and a black window is not a better answer than 4×.
    #[must_use]
    pub fn afforded(&self, asked: crate::Samples) -> crate::Samples {
        self.samples
            .iter()
            .copied()
            .rfind(|s| *s <= asked)
            .unwrap_or_default()
    }
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

/// How large the global bindless arrays may be on this device (§4.3). Both
/// are the min of the per-set and per-stage update-after-bind limits: our one
/// set *is* the per-stage budget, so the smaller of the two is the real cap.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DescriptorLimits {
    pub sampled_images: u32,
    pub storage_images: u32,
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
    descriptor_limits: DescriptorLimits,
    /// Nanoseconds per timestamp tick, and which bits of a tick are real.
    /// A zero mask means the graphics queue writes no usable timestamp, which
    /// is a device property and not a failure (§4.8).
    timestamps: (f32, u64),
    /// `VK_EXT_calibrated_timestamps`, when the device has it and this build
    /// times passes. What it buys is the device clock *without a submit*, which
    /// is the only way to anchor a GPU zone to a CPU one without the anchor
    /// itself costing a round trip (§4.8).
    calibrated: Option<ash::ext::calibrated_timestamps::Device>,
    /// `VK_AMD_buffer_marker`, when advertised: a breadcrumb written *at* a
    /// pipeline stage rather than as a loose transfer command (§4.8).
    buffer_marker: Option<ash::amd::buffer_marker::Device>,
    /// `VK_EXT_device_fault`, when advertised *and* its feature enabled — what
    /// turns `DEVICE_LOST` into an address and a vendor code. Absent on
    /// lavapipe (§6 M8, measured by `xtask probe`).
    fault: Option<ash::ext::device_fault::Device>,
    allocator: Option<gpu_allocator::vulkan::Allocator>,
    report: DeviceReport,
}

/// Whether the physical device advertises `extension`.
fn has_extension(instance: &ash::Instance, pd: vk::PhysicalDevice, extension: &CStr) -> bool {
    // SAFETY: pd comes from the live instance.
    let extensions = unsafe { instance.enumerate_device_extension_properties(pd) };
    extensions.is_ok_and(|list| {
        list.iter().any(|e| {
            e.extension_name_as_c_str()
                .is_ok_and(|name| name == extension)
        })
    })
}

/// Whether the device *implements* device fault, not merely advertises the
/// extension. Only ever asked once the extension is known present: querying a
/// feature struct a device does not support is invalid usage, not a `false`.
fn has_device_fault(instance: &ash::Instance, pd: vk::PhysicalDevice) -> bool {
    let mut fault = vk::PhysicalDeviceFaultFeaturesEXT::default();
    let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut fault);
    // SAFETY: pd comes from the live instance; the chain is default-initialized
    // and the extension was proven advertised by the caller.
    unsafe { instance.get_physical_device_features2(pd, &mut f2) };
    fault.device_fault == vk::TRUE
}

/// Whether the *device* time domain is calibrateable. Advertising the extension
/// is not the same claim — a driver may offer host domains only — and asking for
/// a domain it did not list is invalid usage, not an empty answer.
fn device_time_domain(
    entry: &ash::Entry,
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
) -> bool {
    let fns = ash::ext::calibrated_timestamps::Instance::new(entry, instance);
    // SAFETY: pd comes from the live instance, and the instance-level entry
    // points of this extension need no device-level enable.
    let domains = unsafe { fns.get_physical_device_calibrateable_time_domains(pd) };
    domains.is_ok_and(|d| d.contains(&vk::TimeDomainEXT::DEVICE))
}

/// What a machine with nothing to select is told. Read by two people and
/// written for both (§6 M47): a head a player can act on, then rows a bug report
/// is pasted from. What is deliberately *not* here is a citation of PLAN.md — a
/// section number is unactionable to the only person who ever sees this.
///
/// A function rather than the three-armed `match` it was, because the arm that
/// matters most is the one no desk with a working GPU can execute (§6 M55): the
/// caller can only reach this with devices in hand, so an empty `candidates` is
/// checked by construction or not at all.
fn refusal(want: Option<&str>, candidates: &[Candidate]) -> String {
    let mut report = match (want, candidates.is_empty()) {
        // Vulkan works and enumerated nothing — a different machine from the two
        // below, wanting a different sentence: there is no card here whose
        // driver could be "too old", and no rows are coming. Until M55 an empty
        // list took the third head, which promised "the features listed below"
        // and then "Per-device report:" with nothing after it — on what is the
        // single most common way to arrive here at all.
        (_, true) => String::from(
            "Vulkan is working on this machine, but it reports no graphics device at all. If \
             this machine has a graphics card, its driver is most likely not installed; if it \
             is a virtual machine or a remote desktop session, it may have no graphics card to \
             offer.",
        ),
        (Some(w), false) => format!(
            "no graphics device matches GG_ADAPTER={w:?} and can also run this engine, which \
             needs Vulkan 1.3 — per-device report:"
        ),
        (None, false) => String::from(
            "no graphics device on this machine can run this engine. It needs Vulkan 1.3 and \
             the features listed below; on a card that has them the usual cause is a graphics \
             driver too old. Per-device report:",
        ),
    };
    for c in candidates {
        report.push_str(&format!(
            "\n  {} ({}): missing {}",
            c.name,
            c.device_type,
            match c.missing.is_empty() {
                true => "nothing (outscored)".to_string(),
                false => c.missing.join(", "),
            }
        ));
    }
    report
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
    // Read out of the chain head before the rows below borrow its links.
    let shader_int64 = f2.features.shader_int64 == vk::TRUE;

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

    let rows: [(&'static str, bool); 18] = [
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
        // Buffer device addresses are 64-bit values a shader does arithmetic
        // on, so §4.3's "all buffer access by address" implies this row.
        ("shaderInt64", shader_int64),
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
        // The global set's storage-image array carries the same
        // update-after-bind flag as its sampled-image array, so it needs the
        // matching feature — not a spare row: the layout is refused without it.
        (
            "descriptorBindingStorageImageUpdateAfterBind",
            f12.descriptor_binding_storage_image_update_after_bind == vk::TRUE,
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

/// A driver-supplied fixed-size description. Bounded rather than pointer-walked:
/// this reads memory a *lost* device's driver filled in, which is the worst
/// place to trust a terminator that may not be there.
///
/// Generic over the length because Vulkan's three of these — description, driver
/// name, driver info — are all 256 today and agree by coincidence, which is the
/// kind of agreement §2.1 is about.
fn c_str<const N: usize>(bytes: &[std::ffi::c_char; N]) -> String {
    // SAFETY: c_char and u8 have the same layout and size; the array is live.
    let bytes = unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u8>(), bytes.len()) };
    CStr::from_bytes_until_nul(bytes)
        .map(|c| c.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "<undescribed>".into())
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
            unsafe { inst.enumerate_physical_devices() }.map_err(RhiError::vk)?;

        let mut candidates = Vec::new();
        let mut chosen: Option<(vk::PhysicalDevice, u32, usize)> = None;
        // `GG_ADAPTER` — case-insensitive substring of the device name — is how
        // the second vendor gets tested at all on a desk whose highest score is
        // always the same card. No match is an error below, never a fallback: a
        // run reported green on one vendor must not have been green on another.
        let want = std::env::var("GG_ADAPTER")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase());
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
            let wanted = want
                .as_ref()
                .is_none_or(|w| name.to_lowercase().contains(w));
            tracing::info!(
                device = %name,
                r#type = device_type,
                score,
                wanted,
                missing = ?missing,
                "physical device"
            );
            if missing.is_empty()
                && wanted
                && chosen.map(|(_, best, _)| score > best).unwrap_or(true)
            {
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
            return Err(RhiError::NoSuitableDevice(refusal(
                want.as_deref(),
                &candidates,
            )));
        };

        // SAFETY: pd from the live instance.
        let props = unsafe { inst.get_physical_device_properties(pd) };
        // SAFETY: as above.
        let families = unsafe { inst.get_physical_device_queue_family_properties(pd) };

        let mut p12 = vk::PhysicalDeviceVulkan12Properties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut p12);
        // SAFETY: pd from the live instance; the chain is default-initialized.
        unsafe { inst.get_physical_device_properties2(pd, &mut props2) };
        let descriptor_limits = DescriptorLimits {
            sampled_images: p12
                .max_descriptor_set_update_after_bind_sampled_images
                .min(p12.max_per_stage_descriptor_update_after_bind_sampled_images),
            storage_images: p12
                .max_descriptor_set_update_after_bind_storage_images
                .min(p12.max_per_stage_descriptor_update_after_bind_storage_images),
        };

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
            .descriptor_binding_storage_image_update_after_bind(true)
            .descriptor_binding_update_unused_while_pending(true);
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true)
            .maintenance4(true);
        let base = vk::PhysicalDeviceFeatures::default().shader_int64(true);
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .features(base)
            .push_next(&mut f11)
            .push_next(&mut f12)
            .push_next(&mut f13);

        let mut device_extensions = vec![ash::khr::swapchain::NAME.as_ptr()];
        // Mandatory rather than optional, in the spec's own wording: a device
        // that advertises the portability row must be created with it enabled.
        // The instance half is `instance::optional_extensions`; this is the only
        // other place a translated implementation costs a line.
        if has_extension(inst, pd, ash::khr::portability_subset::NAME) {
            device_extensions.push(ash::khr::portability_subset::NAME.as_ptr());
        }
        // Optional and asked for only by a build that profiles: absence costs a
        // Tracy column, never the engine, so it is checked rather than required
        // (§4.8) — and `cfg!` keeps dist from requesting it at all.
        let wants_calibration = cfg!(feature = "gpu-timings")
            && has_extension(inst, pd, ash::ext::calibrated_timestamps::NAME)
            && device_time_domain(instance.entry(), inst, pd);
        if wants_calibration {
            device_extensions.push(ash::ext::calibrated_timestamps::NAME.as_ptr());
        }
        // The crash path, in every tier: §1.6 promises no mystery crashes where
        // it matters most, which is the build nobody can attach a debugger to.
        let wants_markers = has_extension(inst, pd, ash::amd::buffer_marker::NAME);
        if wants_markers {
            device_extensions.push(ash::amd::buffer_marker::NAME.as_ptr());
        }
        let wants_fault =
            has_extension(inst, pd, ash::ext::device_fault::NAME) && has_device_fault(inst, pd);
        let mut fault_features = vk::PhysicalDeviceFaultFeaturesEXT::default().device_fault(true);
        if wants_fault {
            device_extensions.push(ash::ext::device_fault::NAME.as_ptr());
            features2 = features2.push_next(&mut fault_features);
        }
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extensions)
            .push_next(&mut features2);

        // SAFETY: all pointed-to arrays outlive the call; pd is valid.
        let raw = unsafe { inst.create_device(pd, &device_info, None) }.map_err(RhiError::vk)?;
        let swapchain_fns = ash::khr::swapchain::Device::new(inst, &raw);
        let calibrated =
            wants_calibration.then(|| ash::ext::calibrated_timestamps::Device::new(inst, &raw));
        let buffer_marker = wants_markers.then(|| ash::amd::buffer_marker::Device::new(inst, &raw));
        let fault = wants_fault.then(|| ash::ext::device_fault::Device::new(inst, &raw));
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
            unsafe { device.create_semaphore(&info, None) }.map_err(RhiError::vk)
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

        // 64 valid bits is the common case and `1u64 << 64` is UB, so the full
        // width is spelled separately rather than reached by shifting.
        let valid_bits = families[graphics_family as usize].timestamp_valid_bits;
        let timestamp_mask = match valid_bits {
            0 => 0,
            64.. => u64::MAX,
            bits => (1u64 << bits) - 1,
        };

        // Both masks, because the scene pass needs a color and a depth
        // attachment at one count and only what they share is reachable.
        let sample_counts = props.limits.framebuffer_color_sample_counts
            & props.limits.framebuffer_depth_sample_counts;
        let samples: Vec<crate::Samples> = crate::Samples::ALL
            .into_iter()
            .filter(|s| sample_counts.contains(s.vk()))
            .collect();

        let api = props.api_version;
        let driver_version = match c_str(&p12.driver_info) {
            // A driver may leave this empty. The encoded number is then the only
            // identity there is, and it is printed raw rather than decoded
            // because the packing is the vendor's and NVIDIA's is not Vulkan's.
            info if info.is_empty() => format!("driverVersion {:#010x}", props.driver_version),
            info => info,
        };
        let report = DeviceReport {
            chosen: candidates[chosen_idx].name.clone(),
            candidates,
            api_version: (
                vk::api_version_major(api),
                vk::api_version_minor(api),
                vk::api_version_patch(api),
            ),
            driver: (c_str(&p12.driver_name), driver_version),
            transfer_dedicated,
            samples,
        };
        tracing::info!(
            chosen = %report.chosen,
            driver = %report.driver.0,
            driver_version = %report.driver.1,
            api = ?report.api_version,
            transfer_dedicated,
            samples = ?report.samples.iter().map(|s| s.count()).collect::<Vec<_>>(),
            device_fault = wants_fault,
            buffer_marker = wants_markers,
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
            descriptor_limits,
            timestamps: (props.limits.timestamp_period, timestamp_mask),
            calibrated,
            buffer_marker,
            fault,
            allocator: Some(allocator),
            report,
        };
        device.set_name(graphics_timeline, "gg.graphics.timeline");
        device.set_name(transfer_timeline, "gg.transfer.timeline");
        Ok(device)
    }

    /// Whether this device advertises `samples` for a color *and* depth
    /// framebuffer. Membership, not a comparison against the maximum.
    pub(crate) fn supports_samples(&self, samples: crate::Samples) -> bool {
        self.report.supports_samples(samples)
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

    pub(crate) fn descriptor_limits(&self) -> DescriptorLimits {
        self.descriptor_limits
    }

    /// Nanoseconds one timestamp tick represents.
    pub(crate) fn timestamp_period(&self) -> f32 {
        self.timestamps.0
    }

    /// Which bits of a written timestamp carry a value. `0` means the graphics
    /// queue cannot time anything — a legal device, so callers degrade rather
    /// than fail (§4.8).
    pub(crate) fn timestamp_mask(&self) -> u64 {
        self.timestamps.1
    }

    /// The device clock right now, in the same ticks a pass timestamp is
    /// written in — read host-side, with no submit and no wait. `None` when the
    /// device lacks `VK_EXT_calibrated_timestamps` or this build does not
    /// profile, which costs a profiler its anchor and costs the engine nothing.
    pub(crate) fn gpu_ticks_now(&self) -> Option<u64> {
        let calibrated = self.calibrated.as_ref()?;
        let info =
            [vk::CalibratedTimestampInfoEXT::default().time_domain(vk::TimeDomainEXT::DEVICE)];
        // SAFETY: the extension was enabled at device creation, and `DEVICE` was
        // proven calibrateable before that handle was kept.
        let (ticks, _deviation) = unsafe { calibrated.get_calibrated_timestamps(&info) }.ok()?;
        Some(ticks.first()? & self.timestamps.1)
    }

    /// Write one breadcrumb (§4.8).
    ///
    /// `VK_AMD_buffer_marker` writes it *at* `stage`, which is what lets a mark
    /// mean "the pass reached this stage". The portable fallback is an ordinary
    /// transfer write, unordered against the draws around it — looser, and
    /// still enough to name the pass a lost device was inside.
    ///
    /// # Safety
    /// `cmd` must be recording outside a render pass, `buffer` must be live with
    /// `TRANSFER_DST` usage, and `offset` must be 4-byte aligned and in range.
    pub(crate) unsafe fn write_marker(
        &self,
        cmd: vk::CommandBuffer,
        stage: vk::PipelineStageFlags,
        buffer: vk::Buffer,
        offset: u64,
        value: u32,
    ) {
        match &self.buffer_marker {
            // SAFETY: caller contract; the extension was enabled at creation.
            Some(fns) => unsafe { fns.cmd_write_buffer_marker(cmd, stage, buffer, offset, value) },
            // SAFETY: caller contract.
            None => unsafe { self.raw.cmd_fill_buffer(cmd, buffer, offset, 4, value) },
        }
    }

    /// Which mechanism [`Device::write_marker`] uses — named in the report,
    /// because how much a mark's position can be trusted depends on it.
    pub(crate) fn marker_mechanism(&self) -> &'static str {
        match self.buffer_marker {
            Some(_) => "VK_AMD_buffer_marker (stage-ordered)",
            None => "cmd_fill_buffer (submission-ordered)",
        }
    }

    /// What the driver saw, after the device was lost. `None` without
    /// `VK_EXT_device_fault`, and on any driver that declines to answer.
    ///
    /// Called *only* on a lost device: the extension's contract is about the
    /// fault that lost it, and there is nothing to ask about before then.
    pub(crate) fn fault_info(&self) -> Option<Fault> {
        let fns = self.fault.as_ref()?;
        let mut counts = vk::DeviceFaultCountsEXT::default();
        // SAFETY: the extension was enabled at device creation, the device
        // handle is this one, and a null info pointer is the spec's own way of
        // asking for counts only.
        let result = unsafe {
            (fns.fp().get_device_fault_info_ext)(
                self.raw.handle(),
                &mut counts,
                std::ptr::null_mut(),
            )
        };
        if result != vk::Result::SUCCESS {
            return None;
        }
        let mut addresses =
            vec![vk::DeviceFaultAddressInfoEXT::default(); counts.address_info_count as usize];
        let mut vendors =
            vec![vk::DeviceFaultVendorInfoEXT::default(); counts.vendor_info_count as usize];
        // The vendor binary is a crash dump for the vendor's own tooling, not
        // something a report can read; asking for none keeps the call to two.
        counts.vendor_binary_size = 0;
        let mut info = vk::DeviceFaultInfoEXT {
            p_address_infos: addresses.as_mut_ptr(),
            p_vendor_infos: vendors.as_mut_ptr(),
            ..Default::default()
        };
        // SAFETY: as above, and both arrays are sized by the counts just read.
        let result = unsafe {
            (fns.fp().get_device_fault_info_ext)(self.raw.handle(), &mut counts, &mut info)
        };
        if result != vk::Result::SUCCESS {
            return None;
        }
        addresses.truncate(counts.address_info_count as usize);
        vendors.truncate(counts.vendor_info_count as usize);
        Some(Fault {
            description: c_str(&info.description),
            addresses: addresses
                .iter()
                .map(|a| FaultAddress {
                    kind: address_kind(a.address_type),
                    address: a.reported_address,
                    precision: a.address_precision,
                })
                .collect(),
            vendors: vendors
                .iter()
                .map(|v| FaultVendor {
                    description: c_str(&v.description),
                    code: v.vendor_fault_code,
                    data: v.vendor_fault_data,
                })
                .collect(),
        })
    }

    /// Whether uploads cross a queue-family boundary. `false` on lavapipe,
    /// which has one family total — recorded honestly rather than assumed
    /// away, because the ownership-transfer path only exists when this is
    /// `true` and a test that cannot tell would silently prove nothing.
    pub(crate) fn transfer_crosses_families(&self) -> bool {
        self.transfer.family != self.graphics.family
    }

    /// Name any Vulkan object (§1.6). Required at creation sites — the
    /// signature always exists; the call compiles to nothing without the
    /// `validation` feature, which is how names unbolt in dist (§2).
    pub(crate) fn set_name<T: vk::Handle>(&self, handle: T, name: &str) {
        // Gated on the *runtime* answer as well as the feature (§6 M58):
        // `VK_EXT_debug_utils` goes in with the layer, so under
        // `GG_VALIDATION=0` these entry points are not loaded and calling one
        // aborts the process inside `ash`'s loader stub.
        #[cfg(feature = "validation")]
        if crate::instance::validation_requested()
            && let Ok(name) = std::ffi::CString::new(name)
        {
            let info = vk::DebugUtilsObjectNameInfoEXT::default()
                .object_handle(handle)
                .object_name(&name);
            // SAFETY: device is live; handle belongs to it.
            let _ = unsafe { self.debug_fns.set_debug_utils_object_name(&info) };
        }
        #[cfg(not(feature = "validation"))]
        {
            let _ = (handle, name);
        }
    }

    /// Open a debug label around a pass (§4.5: every pass gets one). Like
    /// [`Device::set_name`] it compiles to nothing without `validation`, so a
    /// capture in dev names every pass and dist carries no strings.
    ///
    /// # Safety
    /// `cmd` must be recording, and every label must be closed by
    /// [`Device::end_label`] on the same command buffer.
    pub(crate) unsafe fn begin_label(&self, cmd: vk::CommandBuffer, name: &str) {
        // [`Device::set_name`]'s guard, for its reason.
        #[cfg(feature = "validation")]
        if crate::instance::validation_requested()
            && let Ok(name) = std::ffi::CString::new(name)
        {
            let label = vk::DebugUtilsLabelEXT::default().label_name(&name);
            // SAFETY: caller contract — cmd is recording on this device.
            unsafe { self.debug_fns.cmd_begin_debug_utils_label(cmd, &label) };
        }
        #[cfg(not(feature = "validation"))]
        {
            let _ = (cmd, name);
        }
    }

    /// Close the innermost [`Device::begin_label`].
    ///
    /// # Safety
    /// `cmd` must be recording with a label open.
    pub(crate) unsafe fn end_label(&self, cmd: vk::CommandBuffer) {
        // [`Device::set_name`]'s guard, for its reason. Paired with
        // `begin_label`'s by the same condition, so a label is never left open.
        #[cfg(feature = "validation")]
        if crate::instance::validation_requested() {
            // SAFETY: caller contract — a label is open on this command buffer.
            unsafe { self.debug_fns.cmd_end_debug_utils_label(cmd) };
        }
        #[cfg(not(feature = "validation"))]
        {
            let _ = cmd;
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
        unsafe { self.raw.wait_semaphores(&info, u64::MAX) }.map_err(RhiError::vk)
    }

    /// The graphics timeline's completed value — the deletion queue's clock.
    pub(crate) fn graphics_timeline_value(&self) -> Result<u64, RhiError> {
        // SAFETY: semaphore is live and a timeline.
        unsafe { self.raw.get_semaphore_counter_value(self.graphics.timeline) }
            .map_err(RhiError::vk)
    }

    /// Block until the transfer timeline reaches `value` — the staging ring's
    /// reclaim wait and the synchronous upload path's completion wait (§4.3).
    pub(crate) fn wait_transfer_timeline(&self, value: u64) -> Result<(), RhiError> {
        let semaphores = [self.transfer.timeline];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: semaphore is live and a timeline.
        unsafe { self.raw.wait_semaphores(&info, u64::MAX) }.map_err(RhiError::vk)
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

    /// Ask the device whether it is still there, after a wait that said yes.
    ///
    /// A reset does not fail the wait it unblocks: a TDR force-signals the
    /// timeline and the loss surfaces on some *later* call (measured on
    /// NVIDIA/Windows, §6 M8 — a hung draw's `vkWaitSemaphores` returned
    /// `SUCCESS` and the next frame's submit reported the loss). One frame late
    /// is too late for §4.8: by then the next frame's `prepare_crumbs` has
    /// cleared the marks naming the pass that hung. Free on the path that calls
    /// it — the queue it idles has just been waited to completion.
    pub(crate) fn check_alive(&self) -> Result<(), RhiError> {
        // SAFETY: device is live or lost; both are legal receivers here.
        unsafe { self.raw.device_wait_idle() }.map_err(RhiError::vk)
    }

    /// Allocate GPU memory. Every byte in the engine flows through here so
    /// the shutdown leak report is total (§4.3) — which is also what makes it
    /// the one seam a failure can be injected at (§6 M85).
    pub(crate) fn allocate(
        &mut self,
        desc: &gpu_allocator::vulkan::AllocationCreateDesc<'_>,
    ) -> Result<gpu_allocator::vulkan::Allocation, RhiError> {
        // Before the allocator, not after: an injected failure must leave the
        // real one untouched, or the unwind is grading a state no shortage
        // produces.
        if crate::inject::allocating() {
            return Err(RhiError::Allocator(format!(
                "injected allocation failure at `{}` (§6 M85)",
                desc.name
            )));
        }
        let allocation = self
            .allocator
            .as_mut()
            .ok_or_else(|| RhiError::Allocator("allocator already torn down".into()))?
            .allocate(desc)
            .map_err(|e| RhiError::Allocator(e.to_string()))?;
        crate::inject::allocated();
        Ok(allocation)
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
            .map_err(|e| RhiError::Allocator(e.to_string()))?;
        crate::inject::freed();
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(name: &str, missing: &[&'static str]) -> Candidate {
        Candidate {
            name: name.to_owned(),
            device_type: "discrete GPU",
            score: 0,
            missing: missing.to_vec(),
        }
    }

    /// The arm this desk cannot reach and a player reaches most often (§6 M55):
    /// a loader and a driver, and no device behind them — a GPU disabled in
    /// Device Manager, a container with no passthrough, an RDP session. Two
    /// claims and both are about what it must *not* do: promise a list nothing
    /// follows, and blame a driver version on a machine with no card to have
    /// one.
    #[test]
    fn a_machine_that_enumerated_nothing_is_not_promised_a_list() {
        let body = refusal(None, &[]);
        assert!(
            !body.contains("listed below") && !body.contains("report:"),
            "nothing follows this, so it may not announce that something does: {body}"
        );
        assert!(
            !body.contains("too old"),
            "there is no card here whose driver could be the wrong version: {body}"
        );
        assert!(
            body.lines().count() == 1,
            "and no rows are appended: {body}"
        );
    }

    /// `GG_ADAPTER` is a developer's variable and this is the only head that
    /// names it — a player has not set one, so a message about it would send
    /// them looking for something they never touched.
    #[test]
    fn the_two_heads_that_have_rows_keep_them() {
        let devices = [
            candidate("Card A", &["shaderInt64"]),
            candidate("Card B", &[]),
        ];
        let player = refusal(None, &devices);
        assert!(player.contains("Vulkan 1.3"), "{player}");
        assert!(
            !player.contains("GG_ADAPTER"),
            "not a player's word: {player}"
        );
        assert!(
            player.contains("Card A (discrete GPU): missing shaderInt64"),
            "{player}"
        );
        // The row that says a device was fine and simply lost, which is the one
        // that stops a bug report blaming the wrong card.
        assert!(
            player.contains("Card B (discrete GPU): missing nothing (outscored)"),
            "{player}"
        );
        assert!(refusal(Some("radeon"), &devices).contains("GG_ADAPTER=\"radeon\""));
    }
}
