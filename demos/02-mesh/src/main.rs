//! Demo 02 — Real Geometry (M4A), Real Sim (M4B), app half: a window, the
//! scene from the lib target, and a fly camera that is no longer this file's
//! business. Everything the camera does now happens in [`scene::sim`] as ECS
//! state advanced from action maps; this half owns the window, the swapchain,
//! and the choice between live input and a replay.
//!
//! Three front doors, all of them the same sim:
//!
//! ```text
//! demo-02-mesh                        play
//! demo-02-mesh --record path.ggrp     play, and write down every tick
//! demo-02-mesh --replay path.ggrp     play it back — same camera path, exactly
//! ```
//!
//! `--frames N` bounds any of them, which is how CI runs it headlessly, and
//! `--expect-hash <hex>` turns a replay run into an assertion: the windowed app,
//! driving the real renderer, must land on the canonical state hash the
//! determinism gate recorded for that file (§5.6). It is the end-to-end form of
//! the claim the gate makes at the sim level.

use demo_02_mesh as scene;
use gg_extract::Extracted;
use gg_input::{Drive, Recorder, Replay, ReplayMeta};
use gg_platform::{Control, Event, Key, WindowDesc};
use gg_render::graph::Transients;
use gg_rhi::{DrawSpec, FrameOutcome, FrameStart, Rhi, RhiError};

/// One frame through the graph (§4.5): the scene's declarations, compiled, with
/// the depth attachment and the present handoff both the graph's business.
fn frame(
    rhi: &mut Rhi,
    transients: &mut Transients,
    draws: &[DrawSpec<'_>],
) -> Result<FrameOutcome, RhiError> {
    let token = match rhi.begin_frame()? {
        FrameStart::Ready(token) => token,
        FrameStart::Skipped(outcome) => return Ok(outcome),
    };
    let mut graph = transients.frame(rhi, token.extent())?;
    let backbuffer = graph.backbuffer();
    let depth = graph.depth("scene.depth")?;
    let declared = scene::declare(backbuffer, depth, draws);
    let compiled = graph.compile(&declared)?;
    rhi.execute(token, &compiled.passes())
}

/// Which tier this binary was built as, for the replay header (§4.7). Not a
/// guess: the tier features are the ones the build actually enabled.
const TIER: &str = if cfg!(feature = "validation") {
    "dev"
} else {
    "dist"
};

fn value_of(name: &str) -> Option<String> {
    std::env::args().skip_while(|a| a != name).nth(1)
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

    let frames_arg = value_of("--frames").map(|n| n.parse::<u64>()).transpose()?;
    let record_to = value_of("--record");
    let replay_from = value_of("--replay");
    let expect_hash = value_of("--expect-hash");
    let seed = value_of("--seed")
        .map(|s| s.parse::<u64>())
        .transpose()?
        .unwrap_or(0);

    let mut sim = scene::sim::Sim::new(seed)?;
    let mut input = scene::sim::input()?;
    let mut extracted = Extracted::default();
    // Reused across frames: extract already refuses to allocate per frame, and
    // the push buffer beside it should not undo that.
    let mut pushes: Vec<scene::shaders_gen::mesh::MeshPush> = Vec::new();

    let mut drive = match &replay_from {
        Some(path) => {
            let replay = Replay::decode(&std::fs::read(path)?)?;
            replay.check_verbs(scene::sim::ACTIONS, scene::sim::AXES)?;
            let meta = replay.meta();
            tracing::info!(
                path,
                commit = %meta.engine_commit,
                contract = meta.contract,
                tier = %meta.tier,
                ticks = replay.ticks(),
                seed = meta.seed,
                "replaying"
            );
            // The seed is the file's, not the flag's: a replay carries its own
            // reproduction environment (§4.7), and honouring a command line
            // over the header is how a replay stops reproducing.
            sim = scene::sim::Sim::new(meta.seed)?;
            Drive::Replay(Box::new(replay))
        }
        None => Drive::Live(record_to.is_some().then(|| {
            Box::new(Recorder::new(ReplayMeta {
                engine_commit: gg_input::ENGINE_COMMIT.to_owned(),
                contract: gg_math::DETERMINISM_CONTRACT,
                tier: TIER.to_owned(),
                tick_hz: scene::sim::TICK_HZ,
                seed,
                actions: scene::sim::ACTIONS
                    .iter()
                    .map(|s| (*s).to_owned())
                    .collect(),
                axes: scene::sim::AXES.iter().map(|s| (*s).to_owned()).collect(),
            }))
        })),
    };

    let target_frames = frames_arg
        .or_else(|| drive.ticks())
        .or_else(|| gg_platform::headless().then_some(100));

    let mut rhi: Option<Rhi> = None;
    let mut transients = Transients::default();
    let mut resources: Option<scene::SceneResources> = None;
    let mut failure: Option<anyhow::Error> = None;
    let mut report = None;

    let desc = WindowDesc::visible_unless_headless("golden — 02-mesh", (1280, 720));
    gg_platform::run(desc, |window, event| match event {
        Event::WindowReady => match Rhi::new(window, window.inner_size()) {
            Ok(mut r) => {
                let d = r.device_report();
                tracing::info!(
                    device = %d.chosen,
                    api = ?d.api_version,
                    transfer_dedicated = d.transfer_dedicated,
                    "mesh demo"
                );
                match scene::upload(&mut r) {
                    Ok(s) => resources = Some(s),
                    Err(e) => {
                        failure = Some(e.into());
                        return Control::Exit;
                    }
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
        // Escape is the *app's* key, not the sim's: quitting is not simulated
        // state and must work identically while a replay is driving.
        Event::Key { key, pressed, .. } if key == Key::Escape && pressed => {
            report = rhi.take().map(Rhi::shutdown);
            Control::Exit
        }
        Event::Key { key, pressed, .. } => {
            input.key(key, pressed);
            Control::Continue
        }
        Event::MouseButton { button, pressed } => {
            input.mouse_button(button, pressed);
            Control::Continue
        }
        Event::MouseMotion { dx, dy } => {
            input.motion(dx, dy);
            Control::Continue
        }
        Event::Frame => {
            #[cfg(all(feature = "fp-assert", debug_assertions))]
            gg_math::fpenv::assert_fp_env();

            let (Some(r), Some(s)) = (rhi.as_mut(), resources.as_ref()) else {
                return Control::Continue;
            };

            // One sim tick per presented frame. The accumulator that decouples
            // the two is `gg-core`'s (§4.1) and arrives with the loop skeleton;
            // until then the replay stream is indexed by presented frame, which
            // is a fixed timestep in every sense the determinism gate cares
            // about.
            let tick = sim.tick_count();
            drive.frame(&mut input, tick);
            if let Err(e) = sim.tick(&input) {
                failure = Some(e.into());
                return Control::Exit;
            }

            let camera = match scene::extract(&sim, &mut extracted) {
                Ok(camera) => camera,
                Err(e) => {
                    failure = Some(e.into());
                    return Control::Exit;
                }
            };
            // One draw per cube, one pass, one present. The push constants have
            // to outlive the `DrawSpec`s that borrow them, so they are built
            // into a buffer that survives the frame; instancing and batching are
            // the render graph's (M6), and a draw call per cube is the honest
            // cost of a demo that draws at most a few dozen.
            let extent = r.swapchain_extent();
            pushes.clear();
            pushes.extend(
                extracted
                    .instances
                    .iter()
                    .map(|instance| scene::push_for(&camera, extent, instance, s)),
            );
            let draws: Vec<DrawSpec<'_>> = pushes
                .iter()
                .map(|push| DrawSpec {
                    pipeline: s.pipeline,
                    push_constants: bytemuck::bytes_of(push),
                    count: s.index_count,
                    index_buffer: Some(s.indices),
                })
                .collect();
            match frame(r, &mut transients, &draws) {
                Ok(FrameOutcome::Presented { .. })
                | Ok(FrameOutcome::SkippedSuspended | FrameOutcome::SkippedOutOfDate) => {}
                Err(e) => {
                    failure = Some(e.into());
                    return Control::Exit;
                }
            }
            if target_frames.is_some_and(|t| sim.tick_count() >= t) {
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
    })?;

    if let Some(err) = failure {
        return Err(err);
    }
    let report = report.ok_or_else(|| anyhow::anyhow!("event loop ended before the mesh"))?;

    if let (Drive::Live(Some(recorder)), Some(path)) = (drive, &record_to) {
        let replay = recorder.finish();
        std::fs::write(path, replay.encode())?;
        tracing::info!(
            path,
            ticks = replay.ticks(),
            changes = replay.change_count(),
            "replay written"
        );
    }

    let camera = sim.camera()?;
    let hash = sim.world.canonical_hash();
    if let Some(expected) = &expect_hash {
        let expected = u128::from_str_radix(expected.trim_start_matches("0x"), 16)?;
        anyhow::ensure!(
            hash.get() == expected,
            "the app path diverged from the gate: canonical hash {:032x} after {} ticks,              expected {expected:032x} (§5.6)",
            hash.get(),
            sim.tick_count(),
        );
        tracing::info!(
            ticks = sim.tick_count(),
            "replay reproduced its recorded hash"
        );
    }
    tracing::info!(
        frames = sim.tick_count(),
        cubes = sim.cube_count()?,
        camera = ?camera.position,
        // The contract-bearing number (§4.2.1). Two runs of one replay print
        // the same one, on any architecture — which is the whole milestone.
        canonical_hash = %hash,
        validation_messages = report.validation_messages,
        leaks = report.leaked_allocations.len(),
        "02-mesh done"
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
