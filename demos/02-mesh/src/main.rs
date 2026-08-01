//! Demo 02 — Real Geometry (§6 M4A), app half: a window, the scene from the
//! lib target, and a fly camera on raw keyboard + mouse look. Same CI front
//! door as the earlier demos: headless-capable, `--frames N`, nonzero exit on
//! validation messages or leaks.
//!
//! The camera is driven straight from `gg-platform` events because action maps
//! do not exist yet — `gg-input` and the recorder arrive at M4B (§4.7), and
//! this demo is upgraded in place then. Motion is per *tick*, never per
//! wall-clock second (§2, Sim time row), so a headless 100-frame run and an
//! interactive one put the camera in the same place.

use demo_02_mesh as scene;
use gg_platform::{Control, Event, Key, WindowDesc};
use gg_rhi::{DrawSpec, FrameOutcome, Rhi};

/// Metres per tick at full stick, and radians per unit of mouse motion.
const MOVE_PER_TICK: f32 = 0.035;
const LOOK_PER_UNIT: f32 = 0.0025;
/// Pitch stops just short of straight up: at exactly ±π/2 the view matrix's up
/// vector becomes parallel to the forward one and the basis collapses.
const PITCH_LIMIT: f32 = 1.55;

/// Which movement keys are down. A held key is a *state*, and the platform
/// reports edges, so the state lives here.
#[derive(Default)]
struct Fly {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    look: bool,
}

impl Fly {
    /// Returns `true` if the event was a camera input.
    fn on_key(&mut self, key: Key, pressed: bool) -> bool {
        let slot = match key {
            Key::W => &mut self.forward,
            Key::S => &mut self.back,
            Key::A => &mut self.left,
            Key::D => &mut self.right,
            Key::E | Key::Space => &mut self.up,
            Key::Q | Key::ShiftLeft => &mut self.down,
            _ => return false,
        };
        *slot = pressed;
        true
    }

    fn advance(&self, camera: &mut scene::Camera) {
        let axis = |plus: bool, minus: bool| f32::from(plus) - f32::from(minus);
        let forward = camera.forward() * axis(self.forward, self.back);
        let right = camera.right() * axis(self.right, self.left);
        let up = gg_math::render::Vec3::Y * axis(self.up, self.down);
        let step = (forward + right + up) * MOVE_PER_TICK;
        // Widen once and add in f64: the camera's position is sim-space, and
        // accumulating it in f32 is precisely the precision loss the membrane
        // exists to prevent (§4.2.1).
        camera.position +=
            gg_math::sim::DVec3::new(f64::from(step.x), f64::from(step.y), f64::from(step.z));
    }
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
    let mut resources: Option<scene::SceneResources> = None;
    let mut failure: Option<anyhow::Error> = None;
    let mut report = None;
    let mut tick: u64 = 0;
    let mut camera = scene::Camera::GOLDEN;
    let mut fly = Fly::default();

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
        Event::Key { key, pressed } => {
            if key == Key::Escape && pressed {
                report = rhi.take().map(Rhi::shutdown);
                return Control::Exit;
            }
            // Mouse look is modal so the pointer stays usable: hold Tab (or
            // any mouse motion while it is held) to turn.
            if key == Key::Tab {
                fly.look = pressed;
            }
            fly.on_key(key, pressed);
            Control::Continue
        }
        Event::MouseMotion { dx, dy } => {
            if fly.look {
                camera.yaw -= dx * LOOK_PER_UNIT;
                camera.pitch = (camera.pitch - dy * LOOK_PER_UNIT).clamp(-PITCH_LIMIT, PITCH_LIMIT);
            }
            Control::Continue
        }
        Event::Frame => {
            #[cfg(all(feature = "fp-assert", debug_assertions))]
            gg_math::fpenv::assert_fp_env();

            let (Some(r), Some(s)) = (rhi.as_mut(), resources.as_ref()) else {
                return Control::Continue;
            };
            fly.advance(&mut camera);
            let push = scene::push_for(&camera, r.swapchain_extent(), tick, s);
            let draw = DrawSpec {
                pipeline: s.pipeline,
                push_constants: bytemuck::bytes_of(&push),
                count: s.index_count,
                index_buffer: Some(s.indices),
            };
            match r.render_frame(scene::CLEAR, Some(&draw)) {
                Ok(FrameOutcome::Presented { .. }) => tick += 1,
                Ok(FrameOutcome::SkippedSuspended | FrameOutcome::SkippedOutOfDate) => {}
                Err(e) => {
                    failure = Some(e.into());
                    return Control::Exit;
                }
            }
            if target_frames.is_some_and(|t| tick >= t) {
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

    tracing::info!(
        frames = tick,
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
