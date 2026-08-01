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

use crate::util::workspace_root;
use sha2::Digest;
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
                "probe: WSL lavapipe is the system ICD, not yet a digest-pinned container \
                 (§5.4 deferred machine — lands with the golden suite, M7, whose goldens \
                 the pin protects); pass --system to silence this note"
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
    let digest = format!("{:x}", sha2::Sha256::digest(&bytes));
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
        let mut chosen = None;
        for pd in devices {
            // SAFETY: pd comes from the live instance.
            let props = unsafe { instance.get_physical_device_properties(pd) };
            let name = props
                .device_name_as_c_str()
                .map(|c| c.to_string_lossy().into_owned())
                .unwrap_or_default();
            if system || name.to_lowercase().contains("llvmpipe") {
                chosen = Some((pd, props, name));
                break;
            }
        }
        let (pd, props, name) = chosen
            .ok_or_else(|| anyhow::anyhow!("no lavapipe (llvmpipe) device found — pin broken?"))?;

        let api = props.api_version;
        println!(
            "probe: device `{name}` — Vulkan {}.{}.{}",
            ash::vk::api_version_major(api),
            ash::vk::api_version_minor(api),
            ash::vk::api_version_patch(api),
        );

        let mut f12 = ash::vk::PhysicalDeviceVulkan12Features::default();
        let mut f13 = ash::vk::PhysicalDeviceVulkan13Features::default();
        let mut f2 = ash::vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut f12)
            .push_next(&mut f13);
        // SAFETY: pd from the live instance; feature structs are default-initialized.
        unsafe { instance.get_physical_device_features2(pd, &mut f2) };

        let rows: Vec<(&str, bool)> = vec![
            (
                "Vulkan >= 1.3",
                api >= ash::vk::make_api_version(0, 1, 3, 0),
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
        println!("probe: all capabilities present");
        Ok(())
    })();

    // SAFETY: created above; no live child objects.
    unsafe { instance.destroy_instance(None) };
    result
}
