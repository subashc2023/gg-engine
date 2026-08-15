//! The one failure a stranger is most likely to hit, read as they would read it
//! (§6 M47).
//!
//! Provoked here rather than through the shell for a reason that is structural:
//! a headless run opens no window and brings no renderer up (§1.5), so the shell
//! can never reach device selection in a tier a gate may run. `GG_ADAPTER` is
//! the seam that makes the path reachable at all — a name matching nothing is an
//! error and never a fallback, which is a rule this test now depends on twice.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_rhi::{OffscreenRhi, RhiError};

/// The report has two audiences and has to serve both in one string: a head a
/// player can act on, and rows a bug report can be pasted from. The third
/// assertion is the one with a history — the head cited `§4.3` until this
/// milestone, which is a sentence for us about a document the only person who
/// ever sees it does not have.
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
