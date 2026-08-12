//! Instance bring-up (§4.3): `VK_EXT_debug_utils`, and — under the
//! `validation` feature — the Khronos validation layer with synchronization
//! checking, its messages routed to `tracing` and *counted*, because the §5.4
//! contract is zero messages and a counter is what makes that a gate instead
//! of a scroll-past. `GG_GPUAV=1` adds shader instrumentation on top (§5 gate
//! 4's nightly half) — see [`gpuav_requested`].

use crate::RhiError;
use ash::vk;
use std::sync::atomic::{AtomicU64, Ordering};

/// Warnings + errors from the validation layer since process start, after
/// suppressions (§5.4). Any nonzero value at shutdown is a failed run.
static VALIDATION_MESSAGES: AtomicU64 = AtomicU64::new(0);

/// Validation messages counted so far in this process (0 without the
/// `validation` feature — dist has no layer to hear from).
pub fn validation_message_count() -> u64 {
    VALIDATION_MESSAGES.load(Ordering::Relaxed)
}

/// Whether `GG_GPUAV=1` asked for GPU-assisted validation (§5 gate 4's nightly
/// half, §8's sync-bug row): the layer instruments every shader so descriptor
/// and buffer-device-address faults are caught *at shader execution*, which no
/// CPU-side check can see. Opt-in because instrumentation recompiles every
/// shader and slows the run by an order of magnitude — `xtask gpuav` is the
/// only thing that sets it. Read once: the answer must not change between
/// instance creation and the messages that come back through the callback.
#[cfg(feature = "validation")]
fn gpuav_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("GG_GPUAV").is_ok_and(|v| v == "1"))
}

/// The Vulkan instance plus the debug plumbing that rides with it.
pub struct Instance {
    entry: ash::Entry,
    raw: ash::Instance,
    #[cfg(feature = "validation")]
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

/// Where this instance's surface will come from. Chosen at instance creation
/// because it decides the extension set, and an extension not asked for then
/// cannot be used later.
pub(crate) enum Presentation {
    /// A real window, via the WSI extensions its display handle names.
    Window(raw_window_handle::RawDisplayHandle),
    /// `VK_EXT_headless_surface` — a real `VkSwapchainKHR` whose present
    /// discards, with no OS window anywhere. The only shape in which a
    /// swapchain gate and §1.5 can both hold (§6 M12). Not universally
    /// available: ask [`Instance::headless_supported`] first.
    Headless,
    /// No surface at all — [`OffscreenRhi`](crate::OffscreenRhi)'s instance.
    None,
}

impl Instance {
    /// Whether this machine's loader offers `VK_EXT_headless_surface`.
    ///
    /// A capability question asked of the *instance* layer, so it is answerable
    /// before anything is created — which is what lets a gate skip cleanly on a
    /// driver that lacks it instead of failing.
    pub fn headless_supported() -> bool {
        // SAFETY: loading the system Vulkan loader; sound to call anytime.
        let Ok(entry) = (unsafe { ash::Entry::load() }) else {
            return false;
        };
        // SAFETY: entry is live; `None` asks for the implementation's own list.
        let Ok(available) = (unsafe { entry.enumerate_instance_extension_properties(None) }) else {
            return false;
        };
        available.iter().any(|e| {
            e.extension_name_as_c_str()
                .is_ok_and(|n| n == ash::ext::headless_surface::NAME)
        })
    }

    /// Create the instance with the extensions `presentation` implies. With
    /// `validation`: the Khronos layer, synchronization validation, and the
    /// debug messenger — all lab equipment that unbolts in dist (§1.13).
    pub fn new(presentation: Presentation) -> Result<Self, RhiError> {
        // SAFETY: loading the system Vulkan loader; sound to call anytime.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| RhiError::Loader(format!("Vulkan loader not found: {e}")))?;

        // `mut` feeds the colour-space push below and the validation-only
        // debug_utils one after it.
        let mut extensions = match presentation {
            Presentation::Window(display) => ash_window::enumerate_required_extensions(display)
                .map_err(RhiError::Vk)?
                .to_vec(),
            Presentation::Headless => vec![
                ash::khr::surface::NAME.as_ptr(),
                ash::ext::headless_surface::NAME.as_ptr(),
            ],
            // Offscreen still enables VK_KHR_surface alone: it is what device
            // creation's unconditional VK_KHR_swapchain extension requires,
            // and enabling it costs nothing without a surface object.
            Presentation::None => vec![ash::khr::surface::NAME.as_ptr()],
        };

        // The HDR colour spaces are behind an instance extension, and asking for
        // one that is not enabled is undefined rather than an error — so this is
        // checked and pushed rather than assumed. Absent it, `Swapchain::choose`
        // simply never sees an HDR pair to match and falls back to SDR, which is
        // the same path a display without HDR takes.
        if matches!(presentation, Presentation::Window(_)) {
            // SAFETY: entry is live; `None` asks for the implementation's own list.
            if let Ok(available) = unsafe { entry.enumerate_instance_extension_properties(None) } {
                let named = |n: &std::ffi::CStr| {
                    available
                        .iter()
                        .any(|e| e.extension_name_as_c_str().is_ok_and(|a| a == n))
                };
                if named(ash::ext::swapchain_colorspace::NAME) {
                    extensions.push(ash::ext::swapchain_colorspace::NAME.as_ptr());
                }
            }
        }

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"golden")
            .api_version(vk::make_api_version(0, 1, 3, 0));
        let mut info = vk::InstanceCreateInfo::default().application_info(&app_info);

        #[cfg(feature = "validation")]
        let layer_names = [c"VK_LAYER_KHRONOS_validation".as_ptr()];
        #[cfg(feature = "validation")]
        let enabled = [vk::ValidationFeatureEnableEXT::SYNCHRONIZATION_VALIDATION];
        #[cfg(feature = "validation")]
        let mut validation_features;
        // Declared here so they outlive `create_instance` — the layer reads
        // through these pointers during the call, not before it.
        #[cfg(feature = "validation")]
        let gpuav_on = vk::TRUE;
        #[cfg(feature = "validation")]
        let gpuav_settings;
        #[cfg(feature = "validation")]
        let mut layer_settings;
        #[cfg(feature = "validation")]
        {
            // §1.10, no silent fallbacks: if the layer is absent the run is
            // not the run CI promised, so fail loudly instead of degrading.
            // SAFETY: entry is live.
            let layers =
                unsafe { entry.enumerate_instance_layer_properties() }.map_err(RhiError::Vk)?;
            let have = layers.iter().any(|l| {
                l.layer_name_as_c_str()
                    .is_ok_and(|n| n == c"VK_LAYER_KHRONOS_validation")
            });
            if !have {
                return Err(RhiError::MissingLayer(
                    "VK_LAYER_KHRONOS_validation is not installed — the `validation` feature \
                     requires it (Vulkan SDK supplies it); dist builds compile it out"
                        .into(),
                ));
            }
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            validation_features =
                vk::ValidationFeaturesEXT::default().enabled_validation_features(&enabled);
            info = info
                .enabled_layer_names(&layer_names)
                .push_next(&mut validation_features);

            // GPU-AV goes in through `VK_EXT_layer_settings`, not through
            // `VkValidationFeaturesEXT`: the 1.4.335 layer marks the
            // GPU_ASSISTED enable bit deprecated in favour of `gpuav_enable`,
            // and the settings route is the one it honours natively. The
            // extension is not *enabled* — a settings chain is read by layers
            // during instance creation whether or not the app asked for it,
            // and enabling it would make instance creation fail on a machine
            // with no layer at all.
            let mut setting = vk::LayerSettingEXT::default()
                .layer_name(c"VK_LAYER_KHRONOS_validation")
                .setting_name(c"gpuav_enable")
                .ty(vk::LayerSettingTypeEXT::BOOL32);
            // Not `.values()`: ash sets `value_count` from the byte length,
            // which for a 4-byte BOOL32 would claim four settings values.
            setting.value_count = 1;
            setting.p_values = (&raw const gpuav_on).cast();
            gpuav_settings = [setting];
            layer_settings = vk::LayerSettingsCreateInfoEXT::default().settings(&gpuav_settings);
            if gpuav_requested() {
                tracing::info!("GPU-assisted validation on (GG_GPUAV=1) — instrumented, slow");
                info = info.push_next(&mut layer_settings);
            }
        }

        info = info.enabled_extension_names(&extensions);

        // SAFETY: create-info and all pointed-to arrays outlive the call.
        let raw = unsafe { entry.create_instance(&info, None) }.map_err(RhiError::Vk)?;

        #[cfg(feature = "validation")]
        let debug = {
            let fns = ash::ext::debug_utils::Instance::new(&entry, &raw);
            let messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            // SAFETY: instance is live; callback is 'static.
            let messenger = unsafe { fns.create_debug_utils_messenger(&messenger_info, None) }
                .map_err(RhiError::Vk)?;
            Some((fns, messenger))
        };

        Ok(Self {
            entry,
            raw,
            #[cfg(feature = "validation")]
            debug,
        })
    }

    pub(crate) fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    pub(crate) fn raw(&self) -> &ash::Instance {
        &self.raw
    }

    /// Tear down; call after every child object is destroyed.
    pub(crate) fn destroy(&mut self) {
        #[cfg(feature = "validation")]
        if let Some((fns, messenger)) = self.debug.take() {
            // SAFETY: messenger was created from this instance and is live.
            unsafe { fns.destroy_debug_utils_messenger(messenger, None) };
        }
        // SAFETY: caller upholds child-objects-first; double-destroy is
        // prevented by taking self by &mut exactly once from Rhi teardown.
        unsafe { self.raw.destroy_instance(None) };
    }
}

/// The messenger callback: route to `tracing`, apply §5.4 suppressions,
/// count what survives. Never panics — unwinding across the FFI boundary
/// would be UB, and a diagnostic channel must not be able to kill the run.
#[cfg(feature = "validation")]
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if data.is_null() {
        return vk::FALSE;
    }
    // SAFETY: the layer passes a valid callback-data struct for the call.
    let data = unsafe { &*data };
    let id = if data.p_message_id_name.is_null() {
        String::new()
    } else {
        // SAFETY: non-null id names are NUL-terminated per the spec.
        unsafe { std::ffi::CStr::from_ptr(data.p_message_id_name) }
            .to_string_lossy()
            .into_owned()
    };
    let message = if data.p_message.is_null() {
        String::new()
    } else {
        // SAFETY: as above.
        unsafe { std::ffi::CStr::from_ptr(data.p_message) }
            .to_string_lossy()
            .into_owned()
    };

    // GPU-AV announces the limits and features *it* forces on to instrument
    // (fragmentStoresAndAtomics, vulkanMemoryModel, …) as layer-internal
    // warnings. They describe the layer's own configuration, not our usage, so
    // they are not §5.4 messages — but only under GG_GPUAV, so the ordinary
    // zero-messages contract keeps every id it had.
    if gpuav_requested() && id == "WARNING-Setting-Limit-Adjusted" {
        tracing::info!(target: "gg_rhi::validation", %id, "{message}");
        return vk::FALSE;
    }

    if suppressed().iter().any(|s| s.vuid == id) {
        tracing::debug!(target: "gg_rhi::validation", %id, "suppressed (§5.4 lease)");
        return vk::FALSE;
    }

    VALIDATION_MESSAGES.fetch_add(1, Ordering::Relaxed);
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        tracing::error!(target: "gg_rhi::validation", %id, "{message}");
    } else {
        tracing::warn!(target: "gg_rhi::validation", %id, "{message}");
    }
    vk::FALSE
}

/// The checked-in suppression leases, parsed once. An invalid file logs and
/// suppresses nothing — the suppressions tests fail CI for real.
#[cfg(feature = "validation")]
fn suppressed() -> &'static [crate::suppressions::Suppression] {
    use std::sync::OnceLock;
    static LEASES: OnceLock<Vec<crate::suppressions::Suppression>> = OnceLock::new();
    LEASES.get_or_init(|| {
        crate::suppressions::validated(crate::suppressions::today()).unwrap_or_else(|err| {
            tracing::error!("validation-suppressions.toml invalid ({err}); suppressing nothing");
            Vec::new()
        })
    })
}
