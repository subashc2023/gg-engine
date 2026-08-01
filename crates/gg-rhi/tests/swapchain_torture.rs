//! The swapchain torture gates (§4.3, §6 M1), split into what each actually
//! tests:
//!
//! - `recreation_survives_1000_synthetic_extents` — the *recreation logic*
//!   (where every young engine hides its crashes): synthetic extent changes
//!   against an invisible window's swapchain.
//! - `interactive_resize_minimize_spam_1000` — the *OS event path* (real
//!   resize/minimize plumbing through winit and WSI).
//!
//! Both create OS windows and present to them, so both are `#[ignore]`: the
//! automated tiers are windowless by construction (§1.5 — presenting maps a
//! Wayland surface, un-minimize maps an X11/Win32 one, and WSLg mirrors any
//! mapped window onto the real desktop). They run in the manual windowed
//! suite, `cargo xtask interactive`.
//!
//! Both end with the §4.3 accounting: zero validation messages, zero leaks.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_platform::{Event, Pump, WindowDesc};
use gg_rhi::{
    Access, ColorAttachment, FrameOutcome, FrameStart, Pass, PassKind, Rhi, Target, Transition,
};

/// Deterministic extent stream — no `rand`, reproducible failures.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, lo: u32, hi: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        lo + ((self.0 >> 33) as u32) % (hi - lo)
    }
}

/// One frame, one pass: clear the backbuffer and present it. The smallest
/// graph there is — declared here rather than borrowed from `gg-render`, which
/// this crate cannot depend on, and which these tests are below anyway.
fn clear_frame(rhi: &mut Rhi, color: [f32; 4]) -> FrameOutcome {
    let token = match rhi.begin_frame().expect("begin frame") {
        FrameStart::Ready(token) => token,
        FrameStart::Skipped(outcome) => return outcome,
    };
    let to_attachment = [Transition {
        target: Target::Backbuffer,
        from: Access::None,
        to: Access::ColorWrite,
    }];
    let to_present = [Transition {
        target: Target::Backbuffer,
        from: Access::ColorWrite,
        to: Access::Present,
    }];
    let passes = [
        Pass {
            name: "clear",
            transitions: &to_attachment,
            kind: PassKind::Render {
                color: Some(ColorAttachment {
                    target: Target::Backbuffer,
                    clear: Some(color),
                }),
                depth: None,
                draws: &[],
            },
        },
        Pass {
            name: "present",
            transitions: &to_present,
            kind: PassKind::Barriers,
        },
    ];
    rhi.execute(token, &passes).expect("frame")
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
}

fn shutdown_clean(rhi: Rhi, what: &str) {
    let report = rhi.shutdown();
    assert!(
        report.clean(),
        "{what}: {} validation message(s), leaks {:?}",
        report.validation_messages,
        report.leaked_allocations
    );
}

#[test]
#[ignore = "creates a window + presents (WSI) — manual windowed suite: cargo xtask interactive (§1.5)"]
fn recreation_survives_1000_synthetic_extents() {
    init_tracing();
    let mut pump = Pump::new(WindowDesc::invisible("gg swapchain recreation", (640, 360)))
        .expect("event loop");
    let mut rhi = Rhi::new(pump.window(), pump.window().inner_size()).expect("rhi");
    let start_generation = rhi.swapchain_generation();

    let mut lcg = Lcg(0x9E37_79B9_7F4A_7C15);
    let mut presented = 0u64;
    for i in 0..1000u32 {
        // Synthetic extent: exercises the clamp + old-swapchain handoff +
        // deletion-queue path every frame, whatever the surface finally says.
        let extent = (lcg.next(1, 3000), lcg.next(1, 2000));
        rhi.resize(extent.0, extent.1);
        match clear_frame(&mut rhi, [0.1, 0.2, 0.3, 1.0]) {
            FrameOutcome::Presented { .. } => presented += 1,
            FrameOutcome::SkippedSuspended | FrameOutcome::SkippedOutOfDate => {}
        }
        if i % 64 == 0 {
            let _ = pump.pump();
        }
    }

    let recreations = rhi.swapchain_generation() - start_generation;
    assert!(
        recreations >= 1000,
        "expected 1000 recreations, swapchain saw {recreations}"
    );
    assert!(presented > 0, "no frame ever presented");
    shutdown_clean(rhi, "synthetic-extent recreation");
}

/// The OS event path (§4.3): real resize and minimize events. The window is
/// invisible, non-activating and parked off-screen, but minimize/restore maps
/// windows on every OS — so this is manual-suite only (§1.5).
#[test]
#[ignore = "creates a window + presents (WSI) — manual windowed suite: cargo xtask interactive (§1.5)"]
fn interactive_resize_minimize_spam_1000() {
    init_tracing();
    let mut pump =
        Pump::new(WindowDesc::invisible("gg interactive resize", (640, 360))).expect("event loop");
    let mut rhi = Rhi::new(pump.window(), pump.window().inner_size()).expect("rhi");

    let mut lcg = Lcg(0xB5AD_4ECE_DA1C_E2A9);
    let mut resizes_seen = 0u64;
    let mut presented = 0u64;
    for i in 0..1000u32 {
        if i % 97 == 13 {
            pump.window().set_minimized(true);
        } else if i % 97 == 20 {
            pump.window().set_minimized(false);
        } else {
            let (w, h) = (lcg.next(64, 1600), lcg.next(64, 1000));
            pump.window().request_inner_size(w, h);
        }
        for event in pump.pump() {
            match event {
                Event::Resized(w, h) => {
                    resizes_seen += 1;
                    rhi.resize(w, h);
                }
                // Pump never emits Exiting (it has no event loop to end), and
                // nothing types at an off-screen window; the arms exist so this
                // match stays exhaustive.
                Event::CloseRequested
                | Event::WindowReady
                | Event::Frame
                | Event::Exiting
                | Event::Key { .. }
                | Event::MouseButton { .. }
                | Event::MouseMotion { .. } => {}
            }
        }
        match clear_frame(&mut rhi, [0.3, 0.2, 0.1, 1.0]) {
            FrameOutcome::Presented { .. } => presented += 1,
            FrameOutcome::SkippedSuspended | FrameOutcome::SkippedOutOfDate => {}
        }
    }

    assert!(
        resizes_seen > 100,
        "OS event path barely exercised: {resizes_seen} resize events in 1000 iterations"
    );
    assert!(presented > 0, "no frame ever presented");
    shutdown_clean(rhi, "interactive resize/minimize spam");
}
