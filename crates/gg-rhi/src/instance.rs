//! Instance bring-up (§4.3): `VK_EXT_debug_utils`, and — under the
//! `validation` feature — the Khronos validation layer with synchronization
//! checking, its messages routed to `tracing` and *counted*, because the §5.4
//! contract is zero messages and a counter is what makes that a gate instead
//! of a scroll-past.

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

/// The Vulkan instance plus the debug plumbing that rides with it.
pub struct Instance {
    entry: ash::Entry,
    raw: ash::Instance,
    #[cfg(feature = "validation")]
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl Instance {
    /// Create the instance with the surface extensions the display needs.
    /// With `validation`: the Khronos layer, synchronization validation, and
    /// the debug messenger — all lab equipment that unbolts in dist (§1.13).
    pub fn new(display: raw_window_handle::RawDisplayHandle) -> Result<Self, RhiError> {
        // SAFETY: loading the system Vulkan loader; sound to call anytime.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| RhiError::Loader(format!("Vulkan loader not found: {e}")))?;

        let mut extensions = ash_window::enumerate_required_extensions(display)
            .map_err(RhiError::Vk)?
            .to_vec();

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
