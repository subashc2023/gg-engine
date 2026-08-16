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

/// Whether the validation layer is loaded at all — on unless `GG_VALIDATION=0`
/// (§6 M58).
///
/// The layer is a **frame** cost, not a call cost, and nothing in this tree could
/// see that until `gg-tools frame` could print a frame's CPU zones: on demo 12 at
/// 1080p it adds **5.4 ms** to `render.execute` and leaves the device's own time
/// unchanged to three decimals. That is the whole of a `tier-dev` window running
/// at 150 fps on a 240 Hz panel. Only 0.70 ms of it is synchronization
/// validation, which is why this switch is the layer and not that feature — the
/// expensive part is the core layer's per-command interception.
///
/// Default **on**: every test, every gate and `xtask gpu` keep the checks §4.3's
/// barrier discipline exists to be graded by, and §1.10 forbids a silent
/// downgrade. Off is for the one command whose subject is a human playing —
/// `xtask run` sets it and says so, having nothing to prove about API misuse that
/// the gates do not already prove on the same code. Read once and logged either
/// way, for [`gpuav_requested`]'s reason.
///
/// `pub(crate)` because it also answers "is `VK_EXT_debug_utils` enabled" — the
/// extension goes in exactly when the layer does, so object names and pass
/// labels are gated on the same answer rather than on a second flag that could
/// disagree with it.
#[cfg(feature = "validation")]
pub(crate) fn validation_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| wants_validation(std::env::var("GG_VALIDATION").ok().as_deref()))
}

/// [`validation_requested`]'s policy, without the environment. Split out because
/// the answer is cached in a `OnceLock` for a whole process, so a test that set
/// the variable would decide it for every other test in the binary — the policy
/// is what is worth checking, and it is only checkable apart from the cache.
#[cfg(feature = "validation")]
fn wants_validation(value: Option<&str>) -> bool {
    // Exactly `0` turns it off. Not "anything falsy": this is a *downgrade* of a
    // correctness check (§1.10), so a typo must fail toward the safe answer
    // rather than silently disarm the layer for a run somebody trusted.
    value != Some("0")
}

#[cfg(all(test, feature = "validation"))]
mod tests {
    use super::wants_validation;

    #[test]
    fn only_an_exact_zero_takes_the_validation_layer_off() {
        assert!(wants_validation(None), "unset is the default, and on");
        assert!(!wants_validation(Some("0")));
        // Every other spelling stays on, including the ones that look like an
        // off switch — a downgrade has to be asked for precisely.
        for text in ["", "1", "false", "off", "no", "00", " 0", "0 "] {
            assert!(wants_validation(Some(text)), "`{text}` should stay on");
        }
    }
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

/// The two ways a machine says no before there is anything to choose between,
/// written for the person holding the mouse (§6 M55).
///
/// Both arrive as codes only this crate can read — a `libloading` message about
/// `vulkan-1.dll`, or `ERROR_INCOMPATIBLE_DRIVER` — and a message box is the
/// only place either of them is ever going to be read. The head differs because
/// the *action* does not quite: no loader is a machine that has never had a
/// Vulkan driver, a loader with nothing behind it is one whose driver is gone or
/// predates Vulkan, and telling those apart is what stops a player reinstalling
/// something they already have.
///
/// `detail` is kept, parenthesized and last, because the other reader is a bug
/// report — the same two-audience shape as the per-device report in
/// [`Device::new`](crate::Device).
fn no_vulkan(head: &str, detail: impl std::fmt::Display) -> RhiError {
    RhiError::NoVulkan(format!(
        "{head}\n\nThis game needs a graphics card with Vulkan 1.3 support and an up-to-date \
         driver. Installing the newest driver from your graphics card's maker — NVIDIA, AMD or \
         Intel — is what usually fixes this.\n\n({detail})"
    ))
}

/// A result code as its spec name. `Debug` and not `Display`: ash's `Display`
/// renders the unknown codes a driver may invent as bare integers, and the name
/// is the half worth pasting.
fn e_name(e: vk::Result) -> String {
    format!("{e:?}")
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
        // A machine that has never had a Vulkan driver has no `vulkan-1.dll` /
        // `libvulkan.so.1` either — it is not part of either OS, it arrives with
        // the driver — so this is the *first* thing a player without one hits
        // (§6 M55). It said `Vulkan loader not found: <libloading>` until then.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| no_vulkan("Vulkan is not installed on this machine.", e))?;

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
        if validation_requested() {
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
        let raw = unsafe { entry.create_instance(&info, None) }.map_err(|e| match e {
            // The loader answered and had nothing behind it: every ICD it knows
            // of is absent or refused. A *driver* problem and not a card one,
            // which is the distinction the head draws — the loader being present
            // means this machine had a Vulkan driver at some point (§6 M55).
            vk::Result::ERROR_INCOMPATIBLE_DRIVER => no_vulkan(
                "Vulkan is installed on this machine, but no graphics driver provides it.",
                e_name(e),
            ),
            // Ambiguous by spec — a driver that failed to start, or a layer that
            // did. Dist has no layer (§1.13), so for a player it is the former.
            vk::Result::ERROR_INITIALIZATION_FAILED => no_vulkan(
                "Vulkan is installed on this machine, but it could not be started.",
                e_name(e),
            ),
            other => RhiError::Vk(other),
        })?;

        // No layer, no messenger: `VK_EXT_debug_utils` is only in `extensions`
        // when the layer went in, and creating a messenger against an extension
        // the instance never enabled is undefined rather than merely useless.
        #[cfg(feature = "validation")]
        let debug = if !validation_requested() {
            None
        } else {
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
