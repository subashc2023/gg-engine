//! The failures a stranger's machine is most likely to hit, read as they would
//! read them (§6 M47, §6 M55).
//!
//! Provoked here rather than through the shell for a reason that is structural:
//! a headless run opens no window and brings no renderer up (§1.5), so the shell
//! can never reach bring-up in a tier a gate may run. What that costs is stated
//! rather than papered over — the message box itself is `xtask interactive`'s,
//! and these are the strings that go in it.
//!
//! Three machines, in the order a launch meets them: no loader, a loader with no
//! driver, and a driver whose devices will not do. The first two reach a player
//! as a `libloading` message and a Vulkan enum unless something translates them,
//! and this crate is the only thing that can — §3 bans the tokens everywhere
//! else, so a downstream crate cannot so much as name `ERROR_INCOMPATIBLE_DRIVER`
//! to match on it.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_rhi::{OffscreenRhi, RhiError};

/// What every one of these has to end with. A player who reads none of the rest
/// still gets the one action that fixes the majority of them.
fn is_advice(body: &str) -> bool {
    body.contains("Vulkan 1.3") && body.contains("driver")
}

/// The report has two audiences and has to serve both in one string: a head a
/// player can act on, and rows a bug report can be pasted from. The third
/// assertion is the one with a history — the head cited `§4.3` until M47, which
/// is a sentence for us about a document the only person who ever sees it does
/// not have.
#[test]
fn a_machine_with_no_usable_device_is_refused_in_words_a_player_can_use() {
    // SAFETY: process-per-test (nextest); no other thread reads the env yet.
    unsafe { std::env::set_var("GG_ADAPTER", "no-such-adapter-exists") };
    let refused = match OffscreenRhi::new((8, 8)) {
        Err(RhiError::NoSuitableDevice(report)) => report,
        Err(other) => panic!("expected a device refusal, got {other}"),
        Ok(_) => panic!("a device matched a name no device has"),
    };
    assert!(
        refused.contains("Vulkan 1.3"),
        "the head names what is wanted: {refused}"
    );
    assert!(
        refused.contains("per-device report:"),
        "and is followed by the rows: {refused}"
    );
    assert!(
        !refused.contains('§'),
        "a section of PLAN.md is unactionable to whoever reads this: {refused}"
    );
    // Every candidate is still enumerated by name — the half that was always
    // right, and the half a driver-version bug report is built out of.
    assert!(
        refused.lines().count() >= 2,
        "the report lists the devices it looked at: {refused}"
    );
}

/// A loader with nothing behind it — every ICD it knows of absent or refusing.
/// The player is a fresh Windows install whose graphics driver never landed, or
/// a machine whose card was pulled; it is by some distance the most common way
/// a Vulkan game fails to start for a stranger.
///
/// `VK_DRIVER_FILES` is the whole provocation: the loader honours it in place of
/// its registry and `/usr/share/vulkan` lists, identically on both hosts, so a
/// path to nothing is a machine with no driver and needs no privileges to be
/// one. Until M55 this reached the message box as **`vulkan:
/// ERROR_INCOMPATIBLE_DRIVER`** and nothing else.
#[test]
fn a_machine_with_no_graphics_driver_is_not_shown_a_vulkan_enum() {
    // SAFETY: process-per-test (nextest); no other thread reads the env yet.
    unsafe { std::env::set_var("VK_DRIVER_FILES", "no-such-icd-anywhere.json") };
    let refused = match OffscreenRhi::new((8, 8)) {
        Err(RhiError::NoVulkan(body)) => body,
        Err(other) => panic!("expected a bring-up refusal, got {other}"),
        Ok(_) => panic!("a device came from an ICD list naming nothing"),
    };
    assert!(
        refused.starts_with("Vulkan is installed on this machine, but no graphics driver"),
        "the head separates the loader from the driver, because the actions differ: {refused}"
    );
    assert!(is_advice(&refused), "{refused}");
    // Kept, last, parenthesized: unreadable to the player and the first thing a
    // bug report needs. Losing it to make the box read well would be the same
    // mistake in the other direction.
    assert!(
        refused.contains("(ERROR_INCOMPATIBLE_DRIVER)"),
        "the code survives for whoever is reading the report: {refused}"
    );
}

/// Set for the child half of the test below, whose whole job is to print what
/// bring-up said in a directory where the loader is not a loader.
const NO_LOADER_CHILD: &str = "GG_M55_NO_LOADER_CHILD";

/// libtest's own filter needs the name spelled, and nothing in Rust hands a test
/// its own. A typo here runs no test in the child, which the marker assertion
/// catches by name rather than by silence.
const NO_LOADER_TEST: &str = "a_machine_with_no_vulkan_loader_is_told_what_to_install";

/// What the child prints, so the parent is reading an answer and not a log line.
const MARKER: &str = "M55-CHILD-SAID: ";

/// The failure a machine that has *never* had a Vulkan driver hits first, and
/// the only one here needing a second process: `vulkan-1.dll` / `libvulkan.so.1`
/// is not part of either operating system — it arrives with the driver, which is
/// why §6 M53's baseline does not list it and why `ash` loads it by hand — and
/// which file answers to that name is settled by the loader's search path before
/// any code of ours runs.
///
/// So the provocation is a **file that is not a library** where the library
/// would be found: beside the executable on Windows, whose search order starts
/// with its own directory, and on `LD_LIBRARY_PATH` on Linux, which `dlopen`
/// consults ahead of the cache. §6 M54's shape one layer over, and like it,
/// needing no privileges and touching nothing outside a temp directory.
///
/// One function in two roles rather than a second `#[ignore]`d test: in this
/// tree `#[ignore]` means "creates a window" (§1.5) and nothing else should
/// learn to mean it.
#[test]
fn a_machine_with_no_vulkan_loader_is_told_what_to_install() {
    if std::env::var_os(NO_LOADER_CHILD).is_some() {
        // The child. Its stdout is the test's whole result, so `Ok` is printed
        // too — a plant that did not take must be a legible failure upstream and
        // not a silent pass.
        match OffscreenRhi::new((8, 8)) {
            Err(e) => println!("{MARKER}{e}"),
            Ok(_) => println!("{MARKER}bring-up succeeded"),
        }
        return;
    }

    let dir = std::env::temp_dir().join("gg-m55-no-loader");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // `libvulkan.so.1` reached through `LD_LIBRARY_PATH`; `vulkan-1.dll` beside
    // the binary. Neither host looks for the other's name, so both are planted
    // and the one that matters answers.
    for name in ["vulkan-1.dll", "libvulkan.so.1"] {
        std::fs::write(dir.join(name), b"not a shared library").unwrap();
    }

    let exe = dir.join(format!(
        "child{}",
        std::env::consts::EXE_SUFFIX // the copy is what makes its directory ours
    ));
    std::fs::copy(std::env::current_exe().unwrap(), &exe).unwrap();

    let mut child = std::process::Command::new(&exe);
    child
        .args(["--exact", NO_LOADER_TEST, "--nocapture"])
        .env(NO_LOADER_CHILD, "1")
        // Or the child inherits this run's own ICD list and reports a driver
        // failure from the *previous* test's cause instead of a loader one.
        .env_remove("VK_DRIVER_FILES")
        .env_remove("GG_ADAPTER");
    if cfg!(unix) {
        child.env("LD_LIBRARY_PATH", &dir);
    }
    let out = child.output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);

    let Some(said) = stdout
        .lines()
        .find_map(|l| l.strip_prefix(MARKER).map(str::to_owned))
    else {
        panic!("the child printed no verdict — did {NO_LOADER_TEST} move?\n{stdout}");
    };
    // The whole point of the second process, asserted before the words are: a
    // `dlopen` that fell through to the real loader would report a *working*
    // machine and this test would then be checking nothing at all.
    assert!(
        said != "bring-up succeeded",
        "the planted loader was not the one loaded — this host searched past it"
    );
    assert!(
        said.starts_with("Vulkan is not installed on this machine."),
        "a machine with no loader is told that, and not what `libloading` said: {said}"
    );
    assert!(is_advice(&stdout), "{stdout}");
    let _ = std::fs::remove_dir_all(&dir);
}
