//! The unwind ladders, executed (§6 M85).
//!
//! `gg-rhi` has nine of them — five in `Gpu::new`, four in `Rhi::bring_up` —
//! plus an `Err` arm on every resource this crate creates, and until this file
//! not one had ever run. They are reviewed instead, which is a method with a
//! measured hit rate: the audit that prompted this milestone read all nine,
//! reported three defects, and one was real.
//!
//! What runs them is [`gg_rhi::inject`], a seam at the two kinds of place this
//! crate can fail. What grades them is not this file's opinion but the two
//! things §4.3 already demands of a clean run — no allocation outstanding, no
//! validation message — asked after a failure instead of after a success.
//!
//! The counters behind the seam are process-global, so every test here takes
//! [`ALONE`] first. Nextest gives a process per test and would not need it, but
//! a file that is only correct under one runner is a trap for whoever runs the
//! other, and `cargo test` runs these as threads.
//!
//! Requires `--features inject`, which no tier builds; `xtask ci --nightly`
//! runs it, and by hand it is
//! `cargo nextest run -p gg-rhi --features inject injected`.

#![cfg(feature = "inject")]
// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use gg_rhi::{
    ImageDesc, ImageFormat, ImageUse, OffscreenRhi, Rhi, RhiError, Samples, ShutdownReport, inject,
};

/// Held for the length of every test here: the seam's counters are one set per
/// process. Poisoning is not a concern — a panicking test has already failed.
static ALONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Small on purpose: this sweep takes a device per leg, and what it grades is
/// the bring-up ladder rather than any picture.
const EXTENT: (u32, u32) = (64, 64);

/// A whole offscreen session — bring up, render a frame, tear down — returning
/// the §4.3 report when there was a device to get one from.
///
/// Teardown runs **disarmed**: a leg is about the unwind after one failure, and
/// a second failure during cleanup would grade a state no shortage produces.
fn offscreen_session() -> Result<ShutdownReport, RhiError> {
    let mut rhi = OffscreenRhi::new(EXTENT)?;
    let frame = common::render(&mut rhi, [0.1, 0.2, 0.3, 1.0], &[]);
    inject::disarm();
    let report = rhi.shutdown();
    // After the teardown, never instead of it: a frame that failed still owns a
    // device, and returning its error first would leak everything under it.
    frame?;
    Ok(report)
}

/// The windowed path's bring-up over a surface with no window behind it — the
/// only way to reach `Rhi::bring_up`'s four ladders, and windowless, so §1.5
/// holds. No frame: the ladders are all in the constructor.
fn windowed_session() -> Result<ShutdownReport, RhiError> {
    let rhi = Rhi::headless(EXTENT)?;
    inject::disarm();
    Ok(rhi.shutdown())
}

/// Arm each site in turn and prove the unwind left nothing behind. Returns the
/// refusals in order, which is the roster of what was covered.
fn sweep(what: &str, session: fn() -> Result<ShutdownReport, RhiError>) -> Vec<String> {
    inject::disarm();
    inject::reset();
    let healthy = session().unwrap_or_else(|e| panic!("{what}: the control leg must succeed: {e}"));
    assert!(
        healthy.clean(),
        "{what}: the control leg was not clean: {healthy:?}"
    );
    let census = inject::seen();
    assert_eq!(inject::live(), 0, "{what}: the control leg leaked");
    let baseline = inject::validation_messages();
    let mut sites = Vec::new();

    for nth in 0..census {
        inject::reset();
        inject::arm(nth);
        let outcome = session();
        inject::disarm();

        // Aimed where we said: an unwind allocates nothing and constructs
        // nothing, so a leg that saw more sites than the one it was armed at
        // swallowed the failure and carried on without a resource it asked for.
        assert_eq!(
            inject::seen(),
            nth + 1,
            "{what} site {nth} of {census}: {} sites were reached, so the failure was swallowed",
            inject::seen()
        );
        assert_eq!(
            inject::live(),
            0,
            "{what} site {nth} of {census}: {} allocations outstanding after the unwind",
            inject::live()
        );
        assert_eq!(
            inject::validation_messages(),
            baseline,
            "{what} site {nth} of {census}: the unwind drew {} validation messages",
            inject::validation_messages() - baseline
        );
        let Err(e) = outcome else {
            panic!("{what} site {nth} of {census}: the session completed without it");
        };
        let text = e.to_string();
        assert!(
            text.contains("§6 M85"),
            "{what} site {nth} of {census}: refused for the wrong reason: {text}"
        );
        sites.push(text);
    }
    sites
}

/// Every ladder this leg is supposed to reach, by name.
///
/// A floor on the *count* was the first spelling and it was too forgiving by
/// exactly the amount that matters: deleting a checkpoint took the census from
/// 11 to 10 and the assertion still passed, so the sweep could quietly shrink
/// toward the four sites the allocator alone can see. A name is what a ladder
/// has; a count is what a sweep has.
fn expect(what: &str, sites: &[String], wanted: &[&str], allocations: usize) {
    for name in wanted {
        assert!(
            sites.iter().any(|s| s.contains(name)),
            "{what}: `{name}` was never armed, so its unwind ladder is still unrun — the sweep \
             covered {} sites and not that one",
            sites.len()
        );
    }
    // The allocator's own sites are counted rather than named: which resource
    // needs memory is this crate's business to change, that some do is not. The
    // floor is per leg because the legs differ — a windowed bring-up allocates
    // the staging ring and the breadcrumbs and stops, where an offscreen
    // session also has a target and a readback buffer.
    let found = sites.iter().filter(|s| s.contains("allocator:")).count();
    assert!(
        found >= allocations,
        "{what}: {found} allocation sites where {allocations} were expected — the seam is out \
         of the resource path"
    );
}

/// What was covered, named. A sweep whose reach is invisible is one nobody
/// notices shrinking.
fn roster(what: &str, sites: &[String]) {
    println!(
        "{what}: {} sites swept, each refused and unwound to nothing:",
        sites.len()
    );
    for (nth, site) in sites.iter().enumerate() {
        println!("  {nth:>2}  {site}");
    }
}

#[test]
fn every_failure_this_crate_can_have_unwinds_to_nothing() {
    let _guard = ALONE.lock();
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();

    let offscreen = sweep("offscreen", offscreen_session);
    expect(
        "offscreen",
        &offscreen,
        &[
            // `Gpu::new`'s five ladders, in the order it builds them.
            "Resources::new",
            "Bindless::new",
            "Uploader::new",
            "PipelineStore::new",
            "Breadcrumbs::new",
            // `OffscreenRhi::with_cache`'s two.
            "Timings::new",
            // The window between taking the acquires and submitting them.
            "OffscreenRhi::record",
        ],
        4,
    );
    roster("offscreen", &offscreen);

    // The windowed ladders need `VK_EXT_headless_surface`, which is a Mesa build
    // option — present on the Linux lavapipe, absent from the pinned Windows one
    // (§6 M12). Skipping loudly rather than failing is this crate's existing
    // answer, and it makes these four the WSL lane's to prove.
    if !Rhi::headless_supported() {
        println!("windowed: skipped — this loader has no VK_EXT_headless_surface (§6 M12)");
        return;
    }
    let windowed = sweep("windowed", windowed_session);
    expect(
        "windowed",
        &windowed,
        &[
            "Resources::new",
            "Bindless::new",
            "Uploader::new",
            "PipelineStore::new",
            "Breadcrumbs::new",
            // `Rhi::bring_up`'s own three, which no offscreen session reaches.
            "Swapchain::new",
            "Frames::new",
            "Timings::new",
        ],
        2,
    );
    roster("windowed", &windowed);
}

/// The transfer queue's release is half an ownership transfer; a frame that
/// takes the acquires and then fails still owes them (§4.3, §6 M85).
///
/// Real only where the transfer family is its own, which in this matrix is the
/// desk's GPU and nothing else — `cargo xtask gpu`. On a single-family device
/// the list is empty by construction, so this skips loudly rather than
/// asserting `0 == 0` and reporting it as coverage.
#[test]
fn a_frame_that_fails_still_owes_its_acquires() {
    let _guard = ALONE.lock();
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    inject::disarm();
    inject::reset();

    let mut rhi = OffscreenRhi::new(EXTENT).unwrap();
    if !rhi.device_report().transfer_dedicated {
        println!(
            "acquires: skipped — this device has one queue family, so the release/acquire pair \
             never exists (§4.3); `cargo xtask gpu` is where this runs"
        );
        assert!(rhi.shutdown().clean());
        return;
    }

    let texture = rhi
        .create_image(&ImageDesc {
            name: "m85.texture",
            extent: (4, 4),
            format: ImageFormat::Rgba8Srgb,
            usage: ImageUse::Sampled,
            mip_levels: 1,
            samples: Samples::X1,
        })
        .unwrap();
    rhi.upload_image(texture, 0, &[0xFF; 4 * 4 * 4]).unwrap();
    rhi.flush_uploads().unwrap();
    let owed = rhi.acquires_owed();
    assert!(
        owed > 0,
        "a dedicated transfer family flushed an upload and owes no acquire — this test is \
         measuring nothing"
    );

    inject::arm_site("OffscreenRhi::record");
    let refused = common::render(&mut rhi, [0.0; 4], &[]);
    inject::disarm();
    assert!(refused.is_err(), "the frame was supposed to be refused");
    assert_eq!(
        rhi.acquires_owed(),
        owed,
        "the refused frame dropped {} of {owed} acquires: no later frame will record them and \
         the texture arrives with undefined contents (§4.3)",
        owed - rhi.acquires_owed()
    );

    // And the debt is discharged by the next frame that does submit, rather
    // than accumulating for the rest of the session.
    common::render(&mut rhi, [0.0; 4], &[]).unwrap();
    assert_eq!(
        rhi.acquires_owed(),
        0,
        "the frame after the failure submitted and still left acquires owed"
    );

    rhi.destroy_image(texture).unwrap();
    assert!(rhi.shutdown().clean());
}
