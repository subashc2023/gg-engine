//! Per-pass GPU timestamps (§4.8). Headless by construction, like every other
//! automated gg-rhi test (§1.5) — the windowed context times passes through the
//! same `graph::record`, so what is proven here is the whole mechanism minus the
//! swapchain.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use gg_rhi::{Access, OffscreenRhi, Pass, PassKind, Target, Transition};

fn init_tracing() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
}

/// Every pass in the executed list comes back, in order, under the name it was
/// declared with — the property the overlay's rows are built on.
#[test]
fn every_pass_reports_its_own_time_under_its_own_name() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((64, 64)).unwrap();
    assert!(
        rhi.pass_timings().is_empty(),
        "nothing has executed yet, so there is nothing to have timed"
    );

    common::render(&mut rhi, [0.0, 0.0, 1.0, 1.0], &[]).unwrap();

    let names: Vec<&str> = rhi.pass_timings().iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["test.draw", "test.readback", "test.host-visible"]);
    for timing in rhi.pass_timings() {
        // A device tick is unsigned and the readings only ever move forward, so
        // a negative or absurd duration means the period or the mask is wrong —
        // which is the failure this catches, not the GPU being slow.
        assert!(
            timing.gpu_ms.is_finite() && timing.gpu_ms >= 0.0 && timing.gpu_ms < 10_000.0,
            "{} timed at {} ms",
            timing.name,
            timing.gpu_ms
        );
    }

    // A pool that is reset and never written reads back as zeros, and zeros
    // would satisfy every assertion above — so the frame has to have cost
    // *something*. Per pass would be flaky (a 64x64 clear can land inside one
    // tick); across the frame it cannot be, and a wrong period or a wrong mask
    // still shows up in the bound above.
    let total: f32 = rhi.pass_timings().iter().map(|t| t.gpu_ms).sum();
    assert!(total > 0.0, "the whole frame timed at zero: {total} ms");

    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// The device clock reads host-side, advances, and lands in the same tick space
/// the passes were stamped in — which is the whole claim a GPU zone rests on. An
/// anchor in a different unit or a different era would put every zone on the
/// profiler's timeline at a plausible-looking wrong place.
#[test]
fn the_gpu_clock_anchors_pass_ticks_to_a_readable_device_clock() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((16, 16)).unwrap();
    let Some(before) = rhi.gpu_clock() else {
        // Reported, never required (§4.8): a device without the extension keeps
        // its overlay milliseconds and loses only Tracy's alignment.
        eprintln!("no VK_EXT_calibrated_timestamps on this device — clock untested");
        assert!(rhi.shutdown().clean());
        return;
    };
    assert!(before.period_ns > 0.0, "a zero period converts nothing");

    common::render(&mut rhi, [0.0, 0.0, 1.0, 1.0], &[]).unwrap();
    let after = rhi.gpu_clock().unwrap();
    assert!(
        after.ticks > before.ticks,
        "the device clock stood still across a frame: {} then {}",
        before.ticks,
        after.ticks
    );
    assert_eq!(
        after.period_ns, before.period_ns,
        "the period is a constant"
    );

    for timing in rhi.pass_timings() {
        assert!(
            timing.begin >= before.ticks && timing.end <= after.ticks,
            "{} ran at [{}, {}], outside the [{}, {}] the host clock bracketed it with",
            timing.name,
            timing.begin,
            timing.end,
            before.ticks,
            after.ticks
        );
    }

    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// A second render replaces the readings rather than accumulating them: the
/// overlay shows one frame, not every frame since bring-up.
#[test]
fn each_render_replaces_the_previous_readings() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((16, 16)).unwrap();
    common::render(&mut rhi, [0.0, 0.0, 0.0, 1.0], &[]).unwrap();
    let first = rhi.pass_timings().len();
    common::render(&mut rhi, [1.0, 1.0, 1.0, 1.0], &[]).unwrap();
    assert_eq!(rhi.pass_timings().len(), first);

    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// A graph deeper than the pool times what it can and stops there. Asserted
/// rather than left to be discovered: a prefix silently reported as the whole
/// frame is a profiler that lies by omission, so the cap is a documented number
/// with a warning behind it.
#[test]
fn a_graph_deeper_than_the_pool_is_timed_as_far_as_the_pool_reaches() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();

    // Pure-barrier passes: no attachment, no draw, nothing but the timestamp
    // pair under test.
    let transitions = [Transition {
        target: Target::Backbuffer,
        from: Access::None,
        to: Access::ColorWrite,
    }];
    let passes: Vec<Pass<'_>> = (0..40)
        .map(|i| Pass {
            name: "deep",
            // Only the first moves the target; the rest are empty on purpose.
            transitions: if i == 0 { &transitions } else { &[] },
            kind: PassKind::Barriers,
        })
        .collect();
    rhi.execute(&passes).unwrap();

    assert_eq!(
        rhi.pass_timings().len(),
        32,
        "the pool times 32 passes; a deeper graph reports the prefix and warns"
    );

    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// The per-pass readings **tile**: each pass opens no earlier than the previous
/// one closed, so the column sums to the frame rather than to more than it
/// (§6 M79).
///
/// This is what a `TOP_OF_PIPE` opening stamp gets wrong, and it is worth a
/// structural assertion rather than a threshold because the failure is not
/// "slow", it is *arithmetic*: a top-of-pipe stamp fires when the command
/// processor reaches the pass, which on a device that runs ahead across a
/// barrier is while the previous pass is still executing. Every interval then
/// contains its predecessor's tail. On the desk's integrated Radeon that read
/// `ao-blur` at 20.5 ms for work worth 3.8 and summed the table to 46 % more
/// than the frame it described.
///
/// It does **not** fire on every device, and that is the point rather than a
/// weakness: lavapipe and the 4090 both stalled at the barrier and reported
/// almost the right answer, which is why this table read fine from §6 M8, where
/// it was written, to M79.
/// The assertion is true everywhere and only *provable* where the hardware
/// misbehaves — `gg-tools frame --attribute` is the empirical half, on a real
/// device, and it is manual for that reason.
#[test]
fn a_passs_time_does_not_include_the_pass_before_it() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((64, 64)).unwrap();
    common::render(&mut rhi, [0.0, 0.5, 0.0, 1.0], &[]).unwrap();

    let timings = rhi.pass_timings();
    assert!(timings.len() > 1, "one pass cannot overlap anything");
    // Raw device ticks, masked. A counter wrap inside one frame would read as a
    // descent here and is the one false failure available — it cannot happen:
    // Vulkan guarantees at least 36 valid bits on a queue that timestamps at
    // all, which is 687 seconds even at the coarsest period on this desk (10 ns
    // on the Radeon iGPU), against a frame measured in microseconds.
    for pair in timings.windows(2) {
        let (before, after) = (&pair[0], &pair[1]);
        assert!(
            after.begin >= before.end,
            "{} opened at {} before {} closed at {} — the two readings overlap, \
             so {}'s {} ms contains work that is not its own",
            after.name,
            after.begin,
            before.name,
            before.end,
            after.name,
            after.gpu_ms
        );
    }

    // The same claim as a duration, which is the one a reader of the table acts
    // on: what every pass claims, against the span the whole frame occupied.
    // Tiling makes this an identity minus whatever gaps the queue sat idle for,
    // so the bound is one-sided.
    let claimed: f32 = timings.iter().map(|t| t.gpu_ms).sum();
    let span = span_ms(timings);
    assert!(
        claimed <= span * 1.001 + 1e-4,
        "the passes claim {claimed} ms of a frame that spanned {span} ms"
    );

    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// The frame's whole span in milliseconds, in the same units the rows report —
/// derived from one pass's own ticks-to-ms ratio rather than from the device
/// period, which this crate does not hand out.
fn span_ms(timings: &[gg_rhi::PassTiming]) -> f32 {
    let scale = timings
        .iter()
        .find(|t| t.end > t.begin && t.gpu_ms > 0.0)
        .map(|t| t.gpu_ms / (t.end - t.begin) as f32);
    match (scale, timings.first(), timings.last()) {
        (Some(scale), Some(first), Some(last)) => (last.end - first.begin) as f32 * scale,
        // Every pass landed inside one tick: nothing to compare, and the
        // ordering assertion above has already done the real work.
        _ => f32::INFINITY,
    }
}
