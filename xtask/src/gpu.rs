//! `cargo xtask gpu` — the manual real-GPU leg (§4.3, §4.10, §8): everything
//! headless the pinned lavapipe cannot prove, on the desk's own driver. Manual
//! the way `bench --record` is: the device is the user's gaming GPU, and what
//! the leg proves depends on desk hardware — a §8 residual by name, never a
//! tier's claim. Windowless throughout (§1.5): offscreen suites and readbacks,
//! no swapchain — the windowed suite stays `cargo xtask interactive`.
//!
//! What only this leg proves:
//! - **the queue-family ownership transfer** (§4.3): the release/acquire pair
//!   exists only where a dedicated transfer family does, and lavapipe has one
//!   family — `gg-rhi`'s offscreen suite here is the only gate that runs the
//!   pair at all;
//! - **the golden suite against a real compiler and scheduler** (§4.10): the
//!   per-device reference set (`tests/gg-images/<device>-<os>/`) exists for
//!   exactly this run, and no automated tier reads it;
//! - `gg-render`'s pass list on hardware.
//!
//! The instrumented half is `cargo xtask gpuav --adapter <name>` (§8), which
//! keeps its own command: GPU-assisted validation reruns the suite with
//! rewritten shaders, and folding a 28-second instrumented pass into every
//! run of this leg would tax the run nobody instruments.

use crate::util::{cargo, run as exec, run_capture};

pub fn run(args: &[&str]) -> anyhow::Result<()> {
    let adapter = match args {
        [] => None,
        ["--adapter", name] => Some(*name),
        _ => anyhow::bail!("usage: cargo xtask gpu [--adapter <name>]"),
    };

    // Identity first, and refused rather than noted: under a stray
    // VK_DRIVER_FILES this leg silently becomes a second lavapipe run — and
    // the golden half would even pass, against the lavapipe reference set — a
    // vacuous green wearing the real leg's name (§1.10). `device()` scrubs the
    // children's environment; this probe proves what actually enumerates,
    // through the same bring-up the reference sets are keyed by.
    let mut probe = cargo();
    probe.args(["run", "-q", "-p", "gg-golden", "--", "backend"]);
    let backend = run_capture(device(&mut probe, adapter), "gg-golden backend")?;
    let backend = backend
        .trim()
        .lines()
        .last()
        .unwrap_or("")
        .trim()
        .to_owned();
    anyhow::ensure!(!backend.is_empty(), "`gg-golden backend` printed nothing");
    anyhow::ensure!(
        !crate::bench::is_software(&backend),
        "the real-GPU leg enumerated `{backend}` — a software rasterizer. Unset \
         VK_DRIVER_FILES (or fix GG_ADAPTER): this leg exists to prove the driver \
         lavapipe is not (§4.10, §8)"
    );
    println!("xtask gpu: real-GPU leg on `{backend}`");

    // The offscreen suites. `gg-rhi` is where the ownership-transfer pair
    // lives (§4.3) and its upload tests are what record it; `gg-render` is the
    // pass list. Window-creating tests stay `#[ignore]` on any driver (§1.5).
    let mut tests = cargo();
    tests.args(["nextest", "run", "-p", "gg-rhi", "-p", "gg-render"]);
    exec(
        device(&mut tests, adapter),
        "gg-rhi + gg-render offscreen suites on the real device",
    )?;

    // The other half of the pair (§6 M85): what a frame that *fails* between
    // taking the acquires and submitting them owes afterwards. Vacuous on one
    // family — the list is empty either way — so the test skips on lavapipe and
    // this is the only place in the matrix it asserts anything.
    let mut faults = cargo();
    faults.args([
        "nextest",
        "run",
        "-p",
        "gg-rhi",
        "--features",
        "inject",
        "-E",
        "binary(injected)",
    ]);
    exec(
        device(&mut faults, adapter),
        "injected failure sweep on the real device (§6 M85)",
    )?;

    // The golden suite against this device's own reference set. A device with
    // no set fails with "no reference", which is the suite saying a bless is
    // owed — a deliberate, reviewed act (§4.10), not something this leg does.
    let mut golden = cargo();
    golden.args(["run", "-p", "gg-golden", "--", "run"]);
    exec(
        device(&mut golden, adapter),
        "golden suite on the real device",
    )?;

    println!(
        "xtask gpu: green on `{backend}` — the §4.3 ownership transfer and the §4.10 \
         per-device references ran for real"
    );
    Ok(())
}

/// Point a child at the real driver: no ICD pin, and `GG_ADAPTER` only when
/// the caller named one — in `gg-rhi` a name that matches nothing is an error,
/// never a fallback, so passing it through unchecked is safe.
fn device<'c>(
    cmd: &'c mut std::process::Command,
    adapter: Option<&str>,
) -> &'c mut std::process::Command {
    cmd.env_remove("VK_DRIVER_FILES");
    match adapter {
        Some(name) => cmd.env("GG_ADAPTER", name),
        None => cmd.env_remove("GG_ADAPTER"),
    }
}
