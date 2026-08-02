//! Breadcrumbs on a *healthy* device (§4.8).
//!
//! What CI can prove is that the marks are written and the report reads them:
//! the host only ever writes zeros into that buffer, so a mark that reads back
//! set was written by the device, through whichever mechanism this driver has.
//! What the same marks say after a device loss is unit-tested in `crash.rs`, and
//! proving it on real hardware is the deliberate-hang canary's job (§6 M8) —
//! wedging lavapipe wedges this process, not a device.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use gg_rhi::OffscreenRhi;

/// The report's per-pass rows as `(name, fate)`, in the order it lists them.
fn rows(report: &str) -> Vec<(String, String)> {
    report
        .lines()
        .skip(1)
        .filter_map(|line| line.trim().split_once(char::is_whitespace))
        .map(|(name, fate)| (name.to_owned(), fate.trim().to_owned()))
        .collect()
}

/// Every pass of a frame that retired opened and closed, and the report says
/// nothing was in flight — which is the reading that makes the *other* one
/// mean something.
#[test]
fn a_retired_frame_leaves_both_marks_on_every_pass() {
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
    common::render(&mut rhi, [0.0, 0.0, 0.0, 1.0], &[]).unwrap();
    let report = rhi.breadcrumbs();
    assert!(report.contains("no pass was in flight"), "{report}");
    assert_eq!(
        rows(&report),
        vec![
            ("test.draw".into(), "completed".into()),
            ("test.readback".into(), "completed".into()),
            ("test.host-visible".into(), "completed".into()),
        ],
        "{report}"
    );
    // The marker writes are ordinary recorded commands: if they were malformed
    // the shutdown report would say so rather than the marks quietly missing.
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// A second frame's report is the second frame's: the slot is cleared and its
/// names refilled, so a stale row can never be read as this frame's suspect.
#[test]
fn each_frame_reports_its_own_passes() {
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
    common::render(&mut rhi, [0.0, 0.0, 0.0, 1.0], &[]).unwrap();
    let first = rows(&rhi.breadcrumbs());
    // A depth attachment changes the frame's transitions and not its pass list,
    // which is the point: same rows, freshly written.
    common::render_with_depth(&mut rhi, [0.0, 0.0, 0.0, 1.0], &[]).unwrap();
    assert_eq!(rows(&rhi.breadcrumbs()), first);
    assert!(first.iter().all(|(_, fate)| fate == "completed"));
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}
