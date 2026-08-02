//! The Tracy GPU path, executed rather than only compiled.
//!
//! Only under `--features tracy`, because without it `GpuZones::new` is `None`
//! by construction and the test would prove that a disabled thing is disabled.
//! The windowed shell is where this runs for real (§1.5 keeps that out of every
//! automated tier), so an offscreen context stands in: it produces the same
//! `PassTiming`s from the same query pool.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![cfg(feature = "tracy")]

use gg_debug::GpuZones;
use gg_rhi::{Access, OffscreenRhi, Pass, PassKind, Target, Transition};

/// A real context, opened against a real device clock and fed real readings.
/// No Tracy server is listening, which is the normal case for a dev build and
/// exactly the path that must not fall over.
#[test]
fn zones_are_emitted_for_a_frame_that_actually_ran() {
    // The guard owns the client; dropping it would discard the queue.
    let _guard = gg_debug::init("debug").expect("first subscriber in this process");
    let mut rhi = OffscreenRhi::new((32, 32)).unwrap();
    let clock = rhi.gpu_clock().expect("a device clock on the test backend");
    let mut zones = GpuZones::new("gg.test", clock).expect("a Tracy context");

    let transitions = [Transition {
        target: Target::Backbuffer,
        from: Access::None,
        to: Access::ColorWrite,
    }];
    let passes = [
        Pass {
            name: "test.first",
            transitions: &transitions,
            kind: PassKind::Barriers,
        },
        Pass {
            name: "test.second",
            transitions: &[],
            kind: PassKind::Barriers,
        },
    ];

    // Twice, because the second frame is the one that exercises the watermark
    // carried across frames — the state a single frame cannot reach.
    for _ in 0..2 {
        rhi.execute(&passes).unwrap();
        let timings = rhi.pass_timings();
        assert_eq!(timings.len(), 2, "both passes timed: {timings:?}");
        zones.frame(timings, rhi.gpu_clock());
    }

    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}
