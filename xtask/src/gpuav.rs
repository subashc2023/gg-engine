//! `cargo xtask gpuav` — GPU-assisted validation, the gate PLAN §5 gate 4 and
//! §8's sync-bug row have promised since M0 and nothing ran.
//!
//! Ordinary validation reads the API calls; GPU-AV rewrites every shader so the
//! *shader's* memory accesses are checked at execution. That is the only place
//! this engine's two load-bearing unchecked indices are visible at all (§4.3):
//! a buffer reached by device address, and a texture reached by its slot in the
//! one global descriptor set. Neither has a CPU-side check to fail.
//!
//! Windowless throughout (§1.5): the offscreen path in `gg-rhi`'s tests and
//! `gg-golden`, which is headless by linkage. Instrumentation is opt-in via
//! `GG_GPUAV=1`, read at instance creation and nowhere else, so nothing outside
//! this leg pays for it.
//!
//! The first leg below is the leg's own gate. A layer that quietly declines to
//! instrument would leave every later leg green and meaningless, so the deliberate
//! out-of-bounds device-address read runs *first* and must be caught (§4.10's
//! "a suite that cannot fail is not a gate", applied here).

use crate::util::{cargo, run as exec};

/// Frames per macro. Enough that instrumented shaders execute many times over;
/// small enough that the leg stays a gate rather than a benchmark — the numbers
/// an instrumented run prints are meaningless as performance data anyway.
const FRAMES: &str = "16";

pub fn run(args: &[&str]) -> anyhow::Result<()> {
    let adapter = adapter_arg(args)?;
    match adapter {
        Some(name) => println!(
            "xtask gpuav: instrumented run on the desk's own GPU (GG_ADAPTER={name}) — \
             desk-hardware-dependent, a §8 residual; this is also the only device with a \
             dedicated transfer family, so it is the only place the §4.3 ownership transfer \
             is instrumented at all"
        ),
        None => println!("xtask gpuav: instrumented run on the pinned lavapipe (§5.4)"),
    }

    // Falsifiability first: an in-bounds read through the same instrumented
    // shader must stay silent, and a read past the buffer must be caught.
    exec(
        driver(
            cargo().args(["nextest", "run", "-p", "gg-rhi", "--test", "gpuav"]),
            adapter,
        )?,
        "GPU-AV catches a deliberate out-of-bounds device-address read (and forgives the \
         in-bounds one)",
    )?;

    // The API-shaped half: staging-ring wraps, mip chains, bindless sampled and
    // storage images, indirect draws, reverse-Z, HDR targets — every mechanism
    // §4.3 owns, now with its shaders instrumented. Each test enforces zero
    // validation messages itself through the shutdown report.
    exec(
        driver(cargo().args(["nextest", "run", "-p", "gg-rhi"]), adapter)?,
        "gg-rhi's offscreen suite, instrumented",
    )?;

    // The engine-shaped half: the real v1 pass list with the real shaders, which
    // is where a descriptor index or a vertex-pull address would actually be
    // wrong. `bench`/`load` rather than the golden suite on purpose — these
    // render and assert a clean shutdown without comparing images, so this leg
    // reports on validation and cannot go red for a re-bless.
    exec(
        driver(
            cargo().args([
                "run",
                "-q",
                "-p",
                "gg-golden",
                "--",
                "bench",
                "--frames",
                FRAMES,
            ]),
            adapter,
        )?,
        "the boxes pass list, instrumented",
    )?;
    exec(
        driver(
            cargo().args([
                "run",
                "-q",
                "-p",
                "gg-golden",
                "--",
                "bench",
                "field",
                "--frames",
                FRAMES,
            ]),
            adapter,
        )?,
        "ten thousand objects through the instrumented pass list (indirect draws, §6 M10)",
    )?;
    // A real pack: meshes pulled by device address, textures sampled out of the
    // global set by index — the two §4.3 rules GPU-AV exists to check, driven by
    // data rather than by a fixture.
    exec(
        driver(
            cargo().args(["run", "-q", "-p", "gg-golden", "--", "load"]),
            adapter,
        )?,
        "a pack streamed and drawn, instrumented",
    )?;

    println!("xtask gpuav: green");
    Ok(())
}

/// `--adapter <substring>` selects the desk's own GPU instead of the pin.
fn adapter_arg<'a>(args: &[&'a str]) -> anyhow::Result<Option<&'a str>> {
    match args.iter().position(|a| *a == "--adapter") {
        None => Ok(None),
        Some(at) => args
            .get(at + 1)
            .copied()
            .filter(|n| !n.starts_with("--"))
            .map(Some)
            .ok_or_else(|| anyhow::anyhow!("--adapter needs a device-name substring")),
    }
}

/// Point a child at the driver this leg runs on, and turn instrumentation on.
///
/// The two are mutually exclusive by construction: a pinned ICD file and a
/// device-name filter would fight, and the filter is an error rather than a
/// fallback when it matches nothing, so leaving both set would silently
/// re-pin the run to lavapipe.
fn driver<'a>(
    cmd: &'a mut std::process::Command,
    adapter: Option<&str>,
) -> anyhow::Result<&'a mut std::process::Command> {
    cmd.env("GG_GPUAV", "1");
    match adapter {
        Some(name) => {
            cmd.env_remove("VK_DRIVER_FILES").env("GG_ADAPTER", name);
        }
        None => {
            cmd.env_remove("GG_ADAPTER");
            if cfg!(windows) {
                cmd.env("VK_DRIVER_FILES", crate::probe::ensure_lavapipe()?);
            } else {
                cmd.env("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/lvp_icd.json");
            }
        }
    }
    Ok(cmd)
}
