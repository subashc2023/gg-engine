//! Demo 00 — First Light (§6 M1): an animated clear color, resize-stable.
//! The exit-criteria run: headless-capable (`GG_HEADLESS=1` gives an invisible
//! window), `--frames N` for CI's bounded runs, and a nonzero exit if the run
//! heard a validation message or leaked an allocation — zero-mystery rules
//! (§1.6) enforced at the demo's own front door.

use gg_platform::{Control, Event, WindowDesc};
use gg_render::graph::{Load, Transients, single_pass};
use gg_rhi::{FrameOutcome, FrameStart, Rhi, RhiError};

/// One frame, one pass: clear the backbuffer through the graph (§4.5). The
/// present handoff is the graph's, which is why nothing here mentions one.
fn clear_frame(
    rhi: &mut Rhi,
    transients: &mut Transients,
    color: [f32; 4],
) -> Result<FrameOutcome, RhiError> {
    let token = match rhi.begin_frame()? {
        FrameStart::Ready(token) => token,
        FrameStart::Skipped(outcome) => return Ok(outcome),
    };
    let mut frame = transients.frame(rhi, token.extent())?;
    let backbuffer = frame.backbuffer();
    let declared = single_pass("clear", backbuffer, Load::Clear(color), &[]);
    let compiled = frame.compile(&declared)?;
    rhi.execute(token, &compiled.passes())
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Hazard 5 (§4.2.1): assert the FP environment before any dependency has
    // a chance to have vandalized it unnoticed.
    #[cfg(feature = "fp-assert")]
    gg_math::fpenv::assert_fp_env();

    let frames_arg = std::env::args()
        .skip_while(|a| a != "--frames")
        .nth(1)
        .map(|n| n.parse::<u64>())
        .transpose()?;
    // Headless runs are CI runs and must terminate: default to the §5.7
    // 100-frame contract when no explicit count is given.
    let target_frames = frames_arg.or_else(|| gg_platform::headless().then_some(100));

    let mut rhi: Option<Rhi> = None;
    let mut transients = Transients::default();
    let mut failure: Option<anyhow::Error> = None;
    let mut report = None;
    let mut frame_count: u64 = 0;

    let desc = WindowDesc::visible_unless_headless("golden — 00-clear", (1280, 720));
    gg_platform::run(desc, |window, event| match event {
        Event::WindowReady => match Rhi::new(window, window.inner_size(), gg_rhi::Output::Sdr) {
            Ok(r) => {
                let d = r.device_report();
                tracing::info!(
                    device = %d.chosen,
                    api = ?d.api_version,
                    transfer_dedicated = d.transfer_dedicated,
                    "first light"
                );
                rhi = Some(r);
                Control::Continue
            }
            Err(e) => {
                failure = Some(e.into());
                Control::Exit
            }
        },
        Event::Resized(w, h) => {
            if let Some(r) = rhi.as_mut() {
                r.resize(w, h);
            }
            Control::Continue
        }
        Event::Frame => {
            // Hazard 5 again, per frame in debug builds — this loop's stand-in
            // for "per sim tick" until the loop skeleton exists.
            #[cfg(all(feature = "fp-assert", debug_assertions))]
            gg_math::fpenv::assert_fp_env();

            let Some(r) = rhi.as_mut() else {
                return Control::Continue;
            };
            match clear_frame(r, &mut transients, clear_color(frame_count)) {
                Ok(FrameOutcome::Presented { .. }) => frame_count += 1,
                Ok(FrameOutcome::SkippedSuspended | FrameOutcome::SkippedOutOfDate) => {}
                Err(e) => {
                    failure = Some(e.into());
                    return Control::Exit;
                }
            }
            if target_frames.is_some_and(|t| frame_count >= t) {
                // Shutdown before the window dies with the event loop: the
                // surface must never outlive its window.
                report = rhi.take().map(Rhi::shutdown);
                return Control::Exit;
            }
            Control::Continue
        }
        Event::CloseRequested => {
            report = rhi.take().map(Rhi::shutdown);
            Control::Exit
        }
        Event::Exiting => {
            // The backstop for every path that exits without its own teardown
            // (the failure arms above, or the loop ending on its own). Still
            // inside the closure, so the window is alive — which is the whole
            // point: the surface must never outlive it.
            if let Some(r) = rhi.take() {
                report = Some(r.shutdown());
            }
            Control::Exit
        }
        // Demos 00 and 01 predate raw input (§4.7) and ignore it.
        Event::Key { .. }
        | Event::MouseButton { .. }
        | Event::MouseMotion { .. }
        | Event::CursorMoved { .. }
        | Event::MouseWheel { .. }
        | Event::Focused(_) => Control::Continue,
    })?;

    if let Some(err) = failure {
        return Err(err);
    }
    let report = report.ok_or_else(|| anyhow::anyhow!("event loop ended before first light"))?;

    tracing::info!(
        frames = frame_count,
        validation_messages = report.validation_messages,
        leaks = report.leaked_allocations.len(),
        "00-clear done"
    );
    anyhow::ensure!(
        report.clean(),
        "unclean run: {} validation message(s), {} leaked allocation(s) {:?} (§4.3, §5.4)",
        report.validation_messages,
        report.leaked_allocations.len(),
        report.leaked_allocations,
    );
    Ok(())
}

/// The animation: a slow hue sweep, render-side float math (the sim never
/// sees it, and no float time is stored — `frame_count` is the clock).
fn clear_color(frame: u64) -> [f32; 4] {
    let hue = (frame % 600) as f32 / 600.0 * 6.0;
    let sector = hue as u32 % 6;
    let f = hue - hue.floor();
    let (v, p) = (0.55_f32, 0.08_f32);
    let (q, t) = (v - (v - p) * f, p + (v - p) * f);
    let (r, g, b) = match sector {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    };
    [r, g, b, 1.0]
}
