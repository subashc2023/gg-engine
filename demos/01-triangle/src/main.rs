//! Demo 01 — the Golden Triangle (§6 M2), app half: a window, the scene from
//! the lib target, and — in the dev tier — shader hot reload: edit
//! `shaders/triangle.slang` while this runs and the new triangle is on screen
//! in under 500 ms; a broken edit keeps the last-good pipeline and prints the
//! compile errors (§4.4). Same CI front door as demo 00: headless-capable,
//! `--frames N`, nonzero exit on validation messages or leaks.

use demo_01_triangle as scene;
use gg_platform::{Control, Event, WindowDesc};
use gg_rhi::{DrawSpec, FrameOutcome, Rhi};

/// Dev-tier hot reload state (§4.4 hot path). Compiled out of dist entirely —
/// the dist gate proves no compiler or watcher symbols ship (§5.8).
#[cfg(feature = "hot-reload")]
struct HotReload {
    watcher: gg_shaders::hot::ShaderWatcher,
    /// Set when a rebuilt pipeline is swapped in; the next presented frame
    /// closes the save-to-screen measurement.
    swapped_at: Option<std::time::Instant>,
}

#[cfg(feature = "hot-reload")]
impl HotReload {
    fn new() -> Option<Self> {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders");
        match gg_shaders::hot::ShaderWatcher::new(dir, "triangle.slang") {
            Ok(watcher) => Some(Self {
                watcher,
                swapped_at: None,
            }),
            Err(e) => {
                // Dev convenience must not kill the demo (e.g. running the
                // binary away from the source tree) — but say so loudly.
                tracing::warn!("hot reload disabled: {e}");
                None
            }
        }
    }

    /// Poll the watcher; on a successful recompile build the new pipeline and
    /// retire the old one behind the timeline. Returns the new handle, or
    /// `None` to keep the current pipeline (no event, or last-good kept).
    fn poll(
        &mut self,
        rhi: &mut Rhi,
        current: gg_rhi::PipelineHandle,
    ) -> Option<gg_rhi::PipelineHandle> {
        let result = self.watcher.poll()?;
        let recompiled = match result {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("shader edit failed to compile — keeping last-good pipeline:\n{e}");
                return None;
            }
        };
        let module = &recompiled.module;
        let find =
            |stage: gg_shaders::Stage| module.entry_points.iter().find(|ep| ep.stage == stage);
        let (Some(vs), Some(fs)) = (
            find(gg_shaders::Stage::Vertex),
            find(gg_shaders::Stage::Fragment),
        ) else {
            tracing::error!("shader edit dropped an entry point — keeping last-good pipeline");
            return None;
        };
        let push_size = module
            .push_constants
            .as_ref()
            .map(|p| p.size as u32)
            .unwrap_or(0);
        if push_size != core::mem::size_of::<scene::shaders_gen::triangle::TrianglePush>() as u32 {
            tracing::error!(
                "shader edit changed the push-constant layout — that needs `cargo xtask shaders` \
                 and a rebuild (§4.4 codegen); keeping last-good pipeline"
            );
            return None;
        }
        let desc = gg_rhi::PipelineDesc {
            name: "triangle (hot)",
            vs_spirv: &vs.spirv,
            vs_entry: &vs.spirv_entry,
            fs_spirv: &fs.spirv,
            fs_entry: &fs.spirv_entry,
            push_constant_size: push_size,
            color: gg_rhi::ColorTarget::Backbuffer,
            blend: gg_rhi::Blend::Off,
            depth: gg_rhi::DepthMode::Off,
            depth_bias: false,
        };
        match rhi.create_pipeline(&desc) {
            Ok(handle) => {
                if let Err(e) = rhi.destroy_pipeline(current) {
                    tracing::warn!("retiring old pipeline: {e}");
                }
                tracing::info!(
                    compile_ms = recompiled.compile_time.as_secs_f64() * 1e3,
                    "hot reload: pipeline rebuilt behind the timeline"
                );
                self.swapped_at = Some(recompiled.saved_at);
                Some(handle)
            }
            Err(e) => {
                tracing::error!("hot reload pipeline creation failed — keeping last-good: {e}");
                None
            }
        }
    }

    /// Call after a presented frame: closes the save-to-screen clock.
    fn frame_presented(&mut self) {
        if let Some(saved_at) = self.swapped_at.take() {
            let ms = saved_at.elapsed().as_secs_f64() * 1e3;
            tracing::info!(
                save_to_screen_ms = ms,
                "hot reload: new shader on screen (§4.4 budget: 500 ms)"
            );
        }
    }
}

/// One frame through the graph. The present handoff is the graph's.
fn frame(
    rhi: &mut Rhi,
    transients: &mut gg_render::graph::Transients,
    draws: &[DrawSpec<'_>],
) -> Result<FrameOutcome, gg_rhi::RhiError> {
    let token = match rhi.begin_frame()? {
        gg_rhi::FrameStart::Ready(token) => token,
        gg_rhi::FrameStart::Skipped(outcome) => return Ok(outcome),
    };
    let mut graph = transients.frame(rhi, token.extent())?;
    let backbuffer = graph.backbuffer();
    let declared = scene::declare(backbuffer, draws);
    let compiled = graph.compile(&declared)?;
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
    let target_frames = frames_arg.or_else(|| gg_platform::headless().then_some(100));

    let mut rhi: Option<Rhi> = None;
    let mut transients = gg_render::graph::Transients::default();
    let mut pipeline: Option<gg_rhi::PipelineHandle> = None;
    let mut failure: Option<anyhow::Error> = None;
    let mut report = None;
    let mut frame_count: u64 = 0;
    #[cfg(feature = "hot-reload")]
    let mut hot: Option<HotReload> = None;

    let desc = WindowDesc::visible_unless_headless("golden — 01-triangle", (1280, 720));
    gg_platform::run(desc, |window, event| match event {
        Event::WindowReady => match Rhi::new(window, window.inner_size()) {
            Ok(mut r) => {
                let d = r.device_report();
                tracing::info!(
                    device = %d.chosen,
                    api = ?d.api_version,
                    "golden triangle"
                );
                match r.create_pipeline(&scene::pipeline_desc()) {
                    Ok(p) => pipeline = Some(p),
                    Err(e) => {
                        failure = Some(e.into());
                        return Control::Exit;
                    }
                }
                #[cfg(feature = "hot-reload")]
                {
                    hot = HotReload::new();
                }
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
            #[cfg(all(feature = "fp-assert", debug_assertions))]
            gg_math::fpenv::assert_fp_env();

            let (Some(r), Some(p)) = (rhi.as_mut(), pipeline) else {
                return Control::Continue;
            };
            #[cfg(feature = "hot-reload")]
            if let Some(hot) = hot.as_mut()
                && let Some(new_pipeline) = hot.poll(r, p)
            {
                pipeline = Some(new_pipeline);
            }
            let p = pipeline.unwrap_or(p);

            let push = scene::push_for_extent(r.swapchain_extent());
            let draw = DrawSpec {
                pipeline: p,
                push_constants: bytemuck::bytes_of(&push),
                count: scene::VERTEX_COUNT,
                index_buffer: None,
                indirect: None,
                depth_bias: None,
            };
            match frame(r, &mut transients, std::slice::from_ref(&draw)) {
                Ok(FrameOutcome::Presented { .. }) => {
                    frame_count += 1;
                    #[cfg(feature = "hot-reload")]
                    if let Some(hot) = hot.as_mut() {
                        hot.frame_presented();
                    }
                }
                Ok(FrameOutcome::SkippedSuspended | FrameOutcome::SkippedOutOfDate) => {}
                Err(e) => {
                    failure = Some(e.into());
                    return Control::Exit;
                }
            }
            if target_frames.is_some_and(|t| frame_count >= t) {
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
        | Event::MouseWheel { .. } => Control::Continue,
    })?;

    if let Some(err) = failure {
        return Err(err);
    }
    let report = report.ok_or_else(|| anyhow::anyhow!("event loop ended before the triangle"))?;

    tracing::info!(
        frames = frame_count,
        validation_messages = report.validation_messages,
        leaks = report.leaked_allocations.len(),
        "01-triangle done"
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
