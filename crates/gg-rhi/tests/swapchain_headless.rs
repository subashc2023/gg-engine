//! Swapchain recreation torture with **no window anywhere** (§6 M12, §1.5).
//!
//! M12 asks for the swapchain torture in the nightly tier; §1.5 forbids an
//! automated tier from creating an OS window at all. `VK_EXT_headless_surface`
//! is what makes both true at once: a real `VkSurfaceKHR`, a real
//! `VkSwapchainKHR`, real `vkQueuePresentKHR` — and nothing mapped on any
//! desktop. This is not [`gg_rhi::OffscreenRhi`], which has no surface and no
//! swapchain; every line under test here is the windowed path.
//!
//! What a headless surface additionally buys is *fidelity*, not just legality.
//! It reports `currentExtent` as `0xFFFFFFFF`, meaning the application chooses —
//! so the synthetic extents actually take effect. Behind a Win32 window the
//! surface dictates `currentExtent`, and a thousand random extents all collapse
//! to the window's real size, exercising the clamp and not the resize.
//!
//! Two halves cannot move here, and both are window-shaped rather than
//! inconvenient. The **OS event path** — resize and minimize as a window manager
//! delivers them — needs a window by definition. And **suspend/resume** is not
//! reachable at all: suspension is entered when the *surface* reports a zero
//! extent, which only a minimized window does; a headless surface answers
//! "you choose" forever, so `resize(0, 0)` clamps up to `minImageExtent` and
//! presents. Both stay in `cargo xtask interactive`, which is the honest split.
//!
//! Skips rather than fails where the extension is absent: it is a Mesa build
//! option, present on the Linux lavapipe and absent from the pinned Windows one.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_rhi::{
    Access, ColorAttachment, FrameOutcome, FrameStart, Pass, PassKind, Rhi, Target, Transition,
};

/// Deterministic extent stream — no `rand`, so a failure reproduces.
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

/// One frame, one pass: clear the backbuffer and present it.
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
                    resolve: None,
                }),
                depth: None,
                viewport: None,
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

fn shutdown_clean(rhi: Rhi, what: &str) {
    let report = rhi.shutdown();
    assert!(
        report.clean(),
        "{what}: {} validation message(s), leaks {:?}",
        report.validation_messages,
        report.leaked_allocations
    );
}

/// Print once and return `false` where the driver cannot answer. A skip, not a
/// silent pass: §1.10 forbids a fallback that looks like a run.
fn headless_available() -> bool {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    if Rhi::headless_supported() {
        return true;
    }
    eprintln!(
        "SKIP: VK_EXT_headless_surface is absent from this loader. The Linux lavapipe has it; \
         the pinned Windows build does not, so this leg runs on the WSL lane (§6 M12)."
    );
    false
}

#[test]
fn a_headless_rhi_brings_up_the_whole_windowed_path() {
    if !headless_available() {
        return;
    }
    let mut rhi = Rhi::headless((640, 360)).expect("headless rhi");
    let outcome = clear_frame(&mut rhi, [0.1, 0.2, 0.3, 1.0]);
    assert_eq!(
        outcome,
        FrameOutcome::Presented { suboptimal: false },
        "a headless swapchain must present like any other"
    );
    assert_eq!(rhi.frames_submitted(), 1);
    shutdown_clean(rhi, "headless bring-up");
}

#[test]
fn recreation_survives_1000_synthetic_extents() {
    if !headless_available() {
        return;
    }
    let mut rhi = Rhi::headless((640, 360)).expect("headless rhi");
    let start_generation = rhi.swapchain_generation();

    let mut lcg = Lcg(0x9E37_79B9_7F4A_7C15);
    let mut presented = 0u64;
    // Every iteration drives the clamp, the old-swapchain handoff and the
    // deletion queue — and here the extent is genuinely honoured, so each
    // recreation allocates images at a size no earlier one used.
    for _ in 0..1000u32 {
        let extent = (lcg.next(1, 3000), lcg.next(1, 2000));
        rhi.resize(extent.0, extent.1);
        match clear_frame(&mut rhi, [0.1, 0.2, 0.3, 1.0]) {
            FrameOutcome::Presented { .. } => presented += 1,
            FrameOutcome::SkippedSuspended | FrameOutcome::SkippedOutOfDate => {}
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

/// The fidelity claim in the module docs, asserted rather than asserted-about:
/// a resize here reaches the images.
///
/// Behind a Win32 window the surface dictates `currentExtent` and every
/// requested size collapses to the window's, so the equivalent assertion could
/// not be written at all — which is the concrete reason this leg is worth more
/// than the ignored one it replaces.
#[test]
fn a_resize_reaches_the_images_rather_than_being_overridden() {
    if !headless_available() {
        return;
    }
    let mut rhi = Rhi::headless((640, 360)).expect("headless rhi");
    for extent in [(800u32, 600u32), (1920, 1080), (17, 4099), (1, 1)] {
        rhi.resize(extent.0, extent.1);
        let token = match rhi.begin_frame().expect("begin frame") {
            FrameStart::Ready(token) => token,
            FrameStart::Skipped(outcome) => panic!("skipped at {extent:?}: {outcome:?}"),
        };
        assert_eq!(
            token.extent(),
            extent,
            "the swapchain took an extent the caller did not ask for"
        );
        // Still a full frame: presenting from anything but PRESENT layout is
        // four validation messages, and this suite's contract is zero.
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
                        clear: Some([0.0, 0.0, 0.0, 1.0]),
                        resolve: None,
                    }),
                    depth: None,
                    viewport: None,
                    draws: &[],
                },
            },
            Pass {
                name: "present",
                transitions: &to_present,
                kind: PassKind::Barriers,
            },
        ];
        rhi.execute(token, &passes).expect("frame");
    }
    shutdown_clean(rhi, "extent fidelity");
}
