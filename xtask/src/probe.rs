//! `xtask probe` — spike 2 (§6 M0A): print the capability table the M4A
//! bindless path needs, against the *pinned* lavapipe, and exit nonzero on any
//! missing capability — so a lavapipe or pin regression reports itself instead
//! of waiting to be rediscovered at M7. `--system` probes the default driver
//! stack instead (useful against the host GPU).
//!
//! Pin provenance (§5.4): on Windows, a mesa-dist-win release by version and
//! SHA-256, fetched on first run into target/xtask-cache — pinned fetches are
//! within §9's fresh-clone bar. The WSL pin is a container image by digest —
//! a §5.4 deferred machine that lands with the golden suite (M7), which is
//! what the pin protects; until then Linux probes the system ICD.

use crate::util::{sha256_hex, workspace_root};
use std::path::PathBuf;

const MESA_VERSION: &str = "26.1.3";
const MESA_SHA256: &str = "6dd431f4620cea73970b13e3ffa94f721f2a3924306b8a4283c97648cdb6eb9c";

pub fn run(system: bool) -> anyhow::Result<()> {
    if !system {
        if cfg!(windows) {
            let icd = ensure_lavapipe()?;
            // SAFETY: single-threaded at this point; set before the Vulkan
            // loader is first touched, so the pinned ICD is the only driver.
            unsafe { std::env::set_var("VK_DRIVER_FILES", &icd) };
        } else {
            println!(
                "probe: WSL lavapipe is the system ICD, not a digest-pinned container — \
                 §5.4's named residual: an apt Mesa upgrade can re-author the Linux golden \
                 baseline ungated. Pass --system to silence this note"
            );
        }
    }
    probe_device(system)
}

/// The pinned lavapipe's ICD manifest — fetched (SHA-256-checked) on first
/// use. Shared by the probe, the nightly GPU tests, and the demo-run gates.
pub(crate) fn ensure_lavapipe() -> anyhow::Result<PathBuf> {
    let cache = workspace_root()
        .join("target/xtask-cache")
        .join(format!("mesa3d-{MESA_VERSION}-msvc"));
    let icd = cache.join("x64").join("lvp_icd.x86_64.json");
    if icd.exists() {
        return Ok(icd);
    }

    let url = format!(
        "https://github.com/pal1000/mesa-dist-win/releases/download/{MESA_VERSION}/mesa3d-{MESA_VERSION}-release-msvc.7z"
    );
    let archive = cache.with_extension("7z");
    std::fs::create_dir_all(&cache)?;
    println!("probe: fetching pinned lavapipe {MESA_VERSION} (first run only)");
    crate::util::run(
        std::process::Command::new("curl").args([
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "-o",
            &archive.to_string_lossy(),
            &url,
        ]),
        "download mesa-dist-win",
    )?;

    let bytes = std::fs::read(&archive)?;
    let digest = sha256_hex(&bytes);
    if digest != MESA_SHA256 {
        let _ = std::fs::remove_file(&archive);
        anyhow::bail!(
            "mesa-dist-win {MESA_VERSION} SHA-256 mismatch: got {digest}, pinned {MESA_SHA256} — \
             refusing to use it; the network can change nothing (§5)"
        );
    }

    // bsdtar (ships with Windows) reads 7z.
    crate::util::run(
        std::process::Command::new("tar").args([
            "-xf",
            &archive.to_string_lossy(),
            "-C",
            &cache.to_string_lossy(),
        ]),
        "extract mesa-dist-win",
    )?;
    let _ = std::fs::remove_file(&archive);
    anyhow::ensure!(
        icd.exists(),
        "extracted mesa-dist-win lacks {}",
        icd.display()
    );
    Ok(icd)
}

/// The M4A bindless path (§6 M0A spike 2): every row here is load-bearing for
/// the CI quality story in §5, which is why absence is an error, not a warning.
fn probe_device(system: bool) -> anyhow::Result<()> {
    // SAFETY: `load` dlopen's the system Vulkan loader, whose obligation is that
    // nothing else in the process is mid-teardown of it — this is the first
    // Vulkan call the probe makes, and `entry` outlives every use below.
    let entry = unsafe { ash::Entry::load() }
        .map_err(|e| anyhow::anyhow!("Vulkan loader not found: {e}"))?;

    let app_info =
        ash::vk::ApplicationInfo::default().api_version(ash::vk::make_api_version(0, 1, 3, 0));
    let create_info = ash::vk::InstanceCreateInfo::default().application_info(&app_info);
    // SAFETY: valid create-info; instance destroyed below before return.
    let instance = unsafe { entry.create_instance(&create_info, None)? };

    let result = (|| -> anyhow::Result<()> {
        // SAFETY: instance is live.
        let devices = unsafe { instance.enumerate_physical_devices()? };
        // Same override gg-rhi honours, so the table and the run that follows it
        // describe one device.
        let want = std::env::var("GG_ADAPTER")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_lowercase());
        let mut chosen = None;
        for pd in devices {
            // SAFETY: pd comes from the live instance.
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let name = props
                .device_name_as_c_str()
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_default();
            let lower = name.to_lowercase();
            let hit = match (&want, system) {
                (Some(w), true) => lower.contains(w.as_str()),
                (_, true) => true,
                _ => lower.contains("llvmpipe"),
            };
            if hit {
                chosen = Some((pd, props, name));
                break;
            }
        }
        let (pd, props, name) = chosen.ok_or_else(|| match (&want, system) {
            (Some(w), true) => anyhow::anyhow!("no device matches GG_ADAPTER={w:?}"),
            _ => anyhow::anyhow!("no lavapipe (llvmpipe) device found — pin broken?"),
        })?;

        let api = props.api_version;
        println!(
            "probe: device `{name}` — Vulkan {}.{}.{}",
            ash::vk::api_version_major(api),
            ash::vk::api_version_minor(api),
            ash::vk::api_version_patch(api),
        );

        let mut f11 = ash::vk::PhysicalDeviceVulkan11Features::default();
        let mut f12 = ash::vk::PhysicalDeviceVulkan12Features::default();
        let mut f13 = ash::vk::PhysicalDeviceVulkan13Features::default();
        let mut f2 = ash::vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f11)
            .push_next(&mut f12)
            .push_next(&mut f13);
        // SAFETY: pd from the live instance; feature structs are default-initialized.
        unsafe { instance.get_physical_device_features2(pd, &mut f2) };
        // Read out of the chain head before the rows below borrow its links.
        let shader_int64 = f2.features.shader_int64 == ash::vk::TRUE;

        let rows: Vec<(&str, bool)> = vec![
            (
                "Vulkan >= 1.3",
                api >= ash::vk::make_api_version(0, 1, 3, 0),
            ),
            (
                // M2's pipelines, not M4A's bindless: Slang vertex SPIR-V
                // declares DrawParameters; names match gg-rhi's device rows.
                "shaderDrawParameters",
                f11.shader_draw_parameters == ash::vk::TRUE,
            ),
            (
                // §4.3's "all buffer access by device address" is 64-bit
                // arithmetic in every shader that reads a buffer.
                "shaderInt64",
                shader_int64,
            ),
            (
                "descriptorIndexing",
                f12.descriptor_indexing == ash::vk::TRUE,
            ),
            (
                "runtimeDescriptorArray",
                f12.runtime_descriptor_array == ash::vk::TRUE,
            ),
            (
                "shaderSampledImageArrayNonUniformIndexing",
                f12.shader_sampled_image_array_non_uniform_indexing == ash::vk::TRUE,
            ),
            (
                "shaderStorageBufferArrayNonUniformIndexing",
                f12.shader_storage_buffer_array_non_uniform_indexing == ash::vk::TRUE,
            ),
            (
                "descriptorBindingPartiallyBound",
                f12.descriptor_binding_partially_bound == ash::vk::TRUE,
            ),
            (
                "descriptorBindingSampledImageUpdateAfterBind",
                f12.descriptor_binding_sampled_image_update_after_bind == ash::vk::TRUE,
            ),
            (
                "descriptorBindingStorageBufferUpdateAfterBind",
                f12.descriptor_binding_storage_buffer_update_after_bind == ash::vk::TRUE,
            ),
            (
                "descriptorBindingStorageImageUpdateAfterBind",
                f12.descriptor_binding_storage_image_update_after_bind == ash::vk::TRUE,
            ),
            (
                "descriptorBindingUpdateUnusedWhilePending",
                f12.descriptor_binding_update_unused_while_pending == ash::vk::TRUE,
            ),
            (
                "bufferDeviceAddress",
                f12.buffer_device_address == ash::vk::TRUE,
            ),
            ("timelineSemaphore", f12.timeline_semaphore == ash::vk::TRUE),
            ("dynamicRendering", f13.dynamic_rendering == ash::vk::TRUE),
            ("synchronization2", f13.synchronization2 == ash::vk::TRUE),
        ];

        let mut missing = 0;
        for (cap, ok) in &rows {
            println!("  {} {cap}", if *ok { "PASS" } else { "MISS" });
            missing += usize::from(!ok);
        }
        anyhow::ensure!(
            missing == 0,
            "{missing} required capabilities missing — the M4A bindless path is not viable on this pin (§6 M0A spike 2)"
        );

        // Reported, not required: the engine renders at 1× on anything, so no
        // count is load-bearing (§6 M21). What this line decides is which MSAA
        // modes a *gate* can prove here — pinned lavapipe advertises 1, 4 and 8
        // and not 2, so 2× is reachable only on the desk's own GPU, and the
        // difference is why the clamp tests membership rather than a maximum.
        let counts = props.limits.framebuffer_color_sample_counts
            & props.limits.framebuffer_depth_sample_counts;
        let advertised: Vec<u32> = [1u32, 2, 4, 8, 16, 32, 64]
            .into_iter()
            .filter(|n| counts.contains(ash::vk::SampleCountFlags::from_raw(*n)))
            .collect();
        println!("  MSAA color+depth sample counts: {advertised:?}");

        println!("probe: all capabilities present");
        report_instruments(&instance, pd, &props);
        Ok(())
    })();

    // SAFETY: created above; no live child objects.
    unsafe { instance.destroy_instance(None) };
    result
}

/// §4.8's instruments, reported and never required — an absent row costs a
/// column on the overlay or detail in a crash report, not the engine. Kept
/// beside the required table anyway, because "which backend can tell me what"
/// is the first question of every device-lost investigation.
fn report_instruments(
    instance: &ash::Instance,
    pd: ash::vk::PhysicalDevice,
    props: &ash::vk::PhysicalDeviceProperties,
) {
    // SAFETY: pd comes from the live instance.
    let extensions =
        unsafe { instance.enumerate_device_extension_properties(pd) }.unwrap_or_default();
    let has = |name: &str| {
        extensions.iter().any(|e| {
            e.extension_name_as_c_str()
                .is_ok_and(|c| c.to_string_lossy() == name)
        })
    };
    // SAFETY: as above.
    let families = unsafe { instance.get_physical_device_queue_family_properties(pd) };
    let bits = families
        .iter()
        .filter(|f| f.queue_flags.contains(ash::vk::QueueFlags::GRAPHICS))
        .map(|f| f.timestamp_valid_bits)
        .max()
        .unwrap_or(0);

    println!("probe: optional instruments (§4.8) — absence degrades, never fails");
    let mark = |ok: bool| if ok { "HAVE" } else { "none" };
    println!(
        "  {} graphics-queue timestamps ({bits} valid bits, {} ns/tick)",
        mark(bits > 0),
        props.limits.timestamp_period
    );
    println!(
        "  {} VK_EXT_calibrated_timestamps (GPU/CPU clock correlation for Tracy)",
        mark(has("VK_EXT_calibrated_timestamps"))
    );
    println!(
        "  {} VK_EXT_device_fault (address and vendor code in a device-lost report)",
        mark(has("VK_EXT_device_fault"))
    );
    println!(
        "  {} VK_AMD_buffer_marker (stage-ordered breadcrumbs; else cmd_fill_buffer)",
        mark(has("VK_AMD_buffer_marker"))
    );
    // An *instance* extension, unlike everything above it — reported here anyway
    // because its absence is what makes `gg-rhi`'s swapchain gate skip, and a
    // skip nobody can see is the vacuous pass §5.8 exists to refuse (§6 M12).
    println!(
        "  {} VK_EXT_headless_surface (windowless swapchain recreation gate, §1.5)",
        mark(has_headless_surface())
    );
}

/// Whether the loader offers `VK_EXT_headless_surface`. Loads its own entry: it
/// is an instance-level question, and the probe's instance is already built.
fn has_headless_surface() -> bool {
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
            .is_ok_and(|c| c.to_string_lossy() == "VK_EXT_headless_surface")
    })
}
