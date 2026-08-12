//! §4.11's frame macro: N warmup + M measured frames of the engine's own v1
//! pass list, reporting CPU frame percentiles and per-pass GPU milliseconds.
//!
//! It lives here because this is the only binary that renders headlessly by
//! linkage (§1.5) — the windowed shell's frame loop cannot be measured by an
//! automated tier at all — and because the per-pass numbers come from the same
//! `pass_timings()` the overlay draws and Tracy's zones read (§4.8), so a
//! regression seen here is the one a profile would show.
//!
//! **The CPU number includes the readback the screen does not pay.** An
//! offscreen frame ends in a copy to a mapped buffer and a `Vec` of the pixels;
//! there is no no-readback path through `OffscreenRenderer` and inventing one
//! would be a second frame path to keep honest. It is a constant, so the
//! percentiles still track regressions — but it is wall clock for an offscreen
//! frame, not a frame budget, and the readback pass's own GPU row is itemised
//! below so its share is visible rather than assumed.

use std::time::Instant;

use gg_extract::Extracted;
use gg_render::{OffscreenRenderer, View};

/// 720p rather than the golden scenes' 320x180: this measures passes, and a
/// frame small enough to be free measures the submit instead.
pub const EXTENT: (u32, u32) = (1280, 720);

const CLEAR: [f32; 4] = [0.02, 0.02, 0.03, 1.0];

/// One pass's accumulated device time. A `Vec` with a linear probe rather than a
/// map: pass counts are single digits, and `HashMap` is banned (§3).
struct Pass {
    name: String,
    sum_ms: f64,
    max_ms: f64,
    frames: u32,
}

/// One CPU zone's accumulated host time, keyed by name *and* depth so a zone
/// that appears at two nesting levels stays two rows (§6 M25).
pub struct Zone {
    name: &'static str,
    depth: u16,
    sum_ms: f64,
    max_ms: f64,
    hits: u32,
}

/// Fold one frame's drained samples into the running table.
///
/// Close order is preserved, so the first time a `(name, depth)` pair is seen
/// fixes its row's position — which makes the printed table read top-down as the
/// frame ran rather than alphabetically.
pub fn accumulate(zones: &mut Vec<Zone>, samples: &[gg_core::zone::Sample]) {
    for sample in samples {
        #[expect(
            clippy::cast_precision_loss,
            reason = "a zone is microseconds; f64 holds nanoseconds exactly to 2^53"
        )]
        let ms = sample.nanos as f64 / 1e6;
        match zones
            .iter_mut()
            .find(|z| z.name == sample.name && z.depth == sample.depth)
        {
            Some(zone) => {
                zone.sum_ms += ms;
                zone.max_ms = zone.max_ms.max(ms);
                zone.hits += 1;
            }
            None => zones.push(Zone {
                name: sample.name,
                depth: sample.depth,
                sum_ms: ms,
                max_ms: ms,
                hits: 1,
            }),
        }
    }
}

/// The zone table, indented by depth so the nesting reads as the tree it is.
///
/// Sorted by first appearance rather than by cost: this is a flame graph printed
/// sideways, and re-ordering it by size would break the one property that makes
/// it readable — a child sits under its parent.
pub fn zone_table(zones: &[Zone], frames: usize) -> String {
    use std::fmt::Write as _;

    #[expect(clippy::cast_precision_loss, reason = "frame counts are thousands")]
    let per = frames as f64;
    let mut out = String::new();
    let _ = write!(
        out,
        "\n  {:<30} {:>10} {:>10} {:>8}",
        "cpu zone", "mean ms", "max ms", "per frame"
    );
    // Sorted by close order is what `accumulate` already produced; a child
    // closes before its parent, so printing it as-is would put children above
    // the zone containing them. Reversed, the outermost comes first.
    for zone in zones.iter().rev() {
        let indent = "  ".repeat(usize::from(zone.depth));
        let _ = write!(
            out,
            "\n  {indent}{:<width$} {:>10.4} {:>10.4} {:>8.2}",
            zone.name,
            zone.sum_ms / per,
            zone.max_ms,
            f64::from(zone.hits) / per,
            width = 30usize.saturating_sub(indent.len()),
        );
    }
    out
}

/// The boxes macro: three static instances, extracted once. What M8 archived,
/// and what a *pass-list* regression shows up in.
pub fn run(extracted: &Extracted, frames: usize, json: bool) -> anyhow::Result<()> {
    let renderer = OffscreenRenderer::new(EXTENT)?;
    let view = View::default();
    measure("boxes", renderer, frames, json, |renderer| {
        renderer.frame(extracted, &view, CLEAR, &[])?;
        Ok(())
    })
}

/// The field macro (§6 M10's exit row): ten thousand parented objects, spun,
/// propagated, extracted, culled, sorted, batched and drawn — every frame.
///
/// Everything a real tick pays on the CPU is inside the measured span except
/// the game's systems, which run through the reload boundary and are the
/// shell's to drive. The hub advance below is what `demo_05_many::spin` does,
/// inlined for the same reason the golden scene builds its world by hand: a
/// measurement should not depend on a dylib being present.
///
/// The camera moves, so the culler's answer changes every frame — a static
/// camera would let a cache that should not exist look free.
pub fn field(frames: usize, json: bool) -> anyhow::Result<()> {
    use gg_ecs::World;
    use gg_ecs::boundary::{Light, Model, Node};
    use gg_math::sim;

    let pack = std::path::PathBuf::from("target/assets/05-many.ggpack");
    anyhow::ensure!(
        pack.is_file(),
        "{} is not there — `cargo xtask assets` compiles it (§4.6)",
        pack.display()
    );

    let mut world = World::new();
    world.register::<Model>()?;
    world.register::<Node>()?;
    world.register::<Light>()?;
    // The demo's own sun (§6 M11). A measurement of an *unlit* frame would be a
    // measurement of a frame nobody renders: the shading loop, the shadow pass
    // and ten thousand casters are all part of what a lit frame costs, and the
    // exit budget is about the frame a player sees.
    let sun = world.spawn();
    world.insert(
        sun,
        Light::sun(
            demo_05_many::SUN_DIRECTION,
            demo_05_many::SUN_COLOR,
            demo_05_many::SUN_INTENSITY,
        ),
    )?;
    let mut hubs = Vec::with_capacity(demo_05_many::HUBS);
    for index in 0..demo_05_many::HUBS {
        let hub = world.spawn();
        world.insert(
            hub,
            Model::at(demo_05_many::MESHES[0], demo_05_many::hub_position(index)),
        )?;
        hubs.push(hub);
        for slot in 0..demo_05_many::PER_HUB {
            let (offset, mesh) = demo_05_many::child_placement(slot);
            let child = world.spawn();
            world.insert(child, Node::at(hub, offset))?;
            world.insert(
                child,
                Model::at(demo_05_many::MESHES[mesh], sim::DVec3::ZERO),
            )?;
        }
    }
    let objects = demo_05_many::HUBS * (demo_05_many::PER_HUB + 1);

    let mut renderer = OffscreenRenderer::new(EXTENT)?;
    renderer.open_pack(&pack)?;
    // Stream the pack in before the clock starts: this measures a steady-state
    // frame, and the first few frames of any pack are a load (§4.6, measured on
    // its own by `gg-golden load`).
    let mut hierarchy = gg_scene::Hierarchy::new();
    let mut extracted = Extracted::default();
    let mut angle = 0.0_f64;
    let mut tick = 0_u32;
    let settle = View {
        pitch: -0.16,
        ..View::default()
    };
    for _ in 0..8 {
        hierarchy.propagate(&mut world)?;
        extracted.clear(demo_05_many::START_POSITION, settle.frustum(EXTENT));
        extracted.append_models::<Model>(&world, renderer.scenes())?;
        extracted.append_lights(&world)?;
        renderer.frame(&extracted, &settle, CLEAR, &[])?;
    }
    let pending = renderer
        .pack()
        .map_or(0, gg_render::content::Content::pending);
    anyhow::ensure!(pending == 0, "{pending} asset(s) still streaming");

    let (instances, draws) = renderer.draw_counts();
    println!(
        "  field: {objects} objects, {} culled, {instances} instances, {draws} draw(s)",
        extracted.culled
    );

    measure("field", renderer, frames, json, move |renderer| {
        // One tick's worth of movement: every other hub turns, and the eye
        // orbits so the frustum's answer is never the previous frame's.
        angle += demo_05_many::SPIN_PER_TICK;
        let turn = sim::DQuat::from_axis_angle(sim::DVec3::Y, angle);
        for (index, hub) in hubs.iter().enumerate() {
            if index % 2 == 0
                && let Some(model) = world.get_mut::<Model>(*hub)
            {
                model.rotation = turn;
            }
        }
        hierarchy.propagate(&mut world)?;

        tick += 1;
        let view = View {
            yaw: f32::from(u16::try_from(tick % 360).unwrap_or(0)) * 0.0175,
            pitch: -0.16,
            ..View::default()
        };
        let (sin, cos) = sim::sin_cos(f64::from(view.yaw));
        let eye = sim::DVec3::new(sin * 90.0, 12.0, cos * 90.0);

        extracted.clear(eye, view.frustum(EXTENT));
        extracted.append_models::<Model>(&world, renderer.scenes())?;
        extracted.append_lights(&world)?;
        renderer.frame(&extracted, &view, CLEAR, &[])?;
        Ok(())
    })
}

/// The measured loop, shared by both scenes. `frame` is one whole frame's work
/// — for the field that includes the tick's propagation and the extract, which
/// is the point: at ten thousand objects the renderer is not where the CPU
/// time is, and a macro that measured only `renderer.frame` would report a
/// budget the engine does not actually spend (§6 M10).
fn measure(
    scene: &str,
    mut renderer: OffscreenRenderer,
    frames: usize,
    json: bool,
    mut frame: impl FnMut(&mut OffscreenRenderer) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    anyhow::ensure!(frames > 0, "a measured run of zero frames measures nothing");
    // Enough to have paged in the pipelines, warmed the allocator and let the
    // clock boost settle; a tenth of the run, floored, so a smoke run still
    // warms up at all.
    let warmup = (frames / 10).max(4);
    let device = renderer.device().chosen.clone();
    for _ in 0..warmup {
        frame(&mut renderer)?;
    }

    // Drained and dropped: the warmup's zones are a cold cache and a first
    // pipeline build, and folding them in would report a frame nobody ran.
    let _ = gg_core::zone::take();

    let mut cpu_ms = Vec::with_capacity(frames);
    let mut passes: Vec<Pass> = Vec::new();
    let mut zones: Vec<Zone> = Vec::new();
    for _ in 0..frames {
        let start = Instant::now();
        frame(&mut renderer)?;
        cpu_ms.push(start.elapsed().as_secs_f64() * 1e3);
        accumulate(&mut zones, &gg_core::zone::take());
        for timing in renderer.pass_timings() {
            let ms = f64::from(timing.gpu_ms);
            match passes.iter_mut().find(|p| p.name == timing.name) {
                Some(pass) => {
                    pass.sum_ms += ms;
                    pass.max_ms = pass.max_ms.max(ms);
                    pass.frames += 1;
                }
                None => passes.push(Pass {
                    name: timing.name.clone(),
                    sum_ms: ms,
                    max_ms: ms,
                    frames: 1,
                }),
            }
        }
    }

    let report = renderer.shutdown();
    anyhow::ensure!(
        report.clean(),
        "unclean bench run: {} validation message(s), {} leak(s) — a measurement taken through a \
         leaking device is not a measurement (§4.3)",
        report.validation_messages,
        report.leaked_allocations.len(),
    );

    cpu_ms.sort_by(f64::total_cmp);
    let stats = Percentiles::of(&cpu_ms);
    if json {
        print!("{}", encode(scene, &device, frames, warmup, stats, &passes));
    } else {
        human(scene, &device, frames, warmup, stats, &passes);
        // After the pass table, because it answers the question the pass table
        // raises: the GPU rows sum to a fraction of the CPU frame, and this is
        // where the rest of it went.
        if gg_core::zone::enabled() {
            println!("{}", zone_table(&zones, frames));
        } else {
            println!("\n  no cpu zones — built without `cpu-timings` (§4.8)");
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Percentiles {
    min: f64,
    p50: f64,
    p95: f64,
    p99: f64,
}

impl Percentiles {
    /// Nearest-rank over an ascending slice — no interpolation, so every number
    /// printed is a frame that actually happened.
    fn of(sorted: &[f64]) -> Self {
        let at = |p: f64| -> f64 {
            #[expect(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "frame counts are thousands at most; the rank is an index"
            )]
            let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
            sorted
                .get(rank.saturating_sub(1).min(sorted.len().saturating_sub(1)))
                .copied()
                .unwrap_or(f64::NAN)
        };
        Percentiles {
            min: at(0.0),
            p50: at(50.0),
            p95: at(95.0),
            p99: at(99.0),
        }
    }
}

fn human(
    scene: &str,
    device: &str,
    frames: usize,
    warmup: usize,
    cpu: Percentiles,
    passes: &[Pass],
) {
    println!(
        "gg-golden bench — {scene}, v1 pass list, {}x{}, {frames} measured frames after {warmup} \
         warmup\n  device {device}\n",
        EXTENT.0, EXTENT.1
    );
    println!(
        "  cpu frame (readback included)   p50 {:>7.3} ms   p95 {:>7.3} ms   p99 {:>7.3} ms   min \
         {:>7.3} ms",
        cpu.p50, cpu.p95, cpu.p99, cpu.min
    );
    if passes.is_empty() {
        println!("  no per-pass GPU timings — built without `gpu-timings` (§4.8)");
        return;
    }
    println!("\n  {:<28} {:>10} {:>10}", "pass", "mean ms", "max ms");
    for pass in passes {
        println!(
            "  {:<28} {:>10.4} {:>10.4}",
            pass.name,
            pass.sum_ms / f64::from(pass.frames),
            pass.max_ms
        );
    }
    let total: f64 = passes.iter().map(|p| p.sum_ms / f64::from(p.frames)).sum();
    println!("  {:<28} {total:>10.4}", "gpu total");
}

/// §4.11 says the frame macro reports as JSON, so it does — hand-written,
/// because one flat record is not worth a serialization dependency in a crate
/// whose whole job is pixels.
fn encode(
    scene: &str,
    device: &str,
    frames: usize,
    warmup: usize,
    cpu: Percentiles,
    passes: &[Pass],
) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    // Infallible: writing to a String cannot fail.
    let _ = write!(
        out,
        "{{\"scene\":\"{scene}\",\"extent\":[{},{}],\"device\":\"{}\",\"frames\":{frames},\
         \"warmup\":{warmup},\"cpu_frame_ms\":{{\"p50\":{:.4},\"p95\":{:.4},\"p99\":{:.4},\
         \"min\":{:.4}}},\"passes\":[",
        EXTENT.0,
        EXTENT.1,
        escape(device),
        cpu.p50,
        cpu.p95,
        cpu.p99,
        cpu.min
    );
    for (i, pass) in passes.iter().enumerate() {
        let _ = write!(
            out,
            "{}{{\"name\":\"{}\",\"mean_ms\":{:.4},\"max_ms\":{:.4}}}",
            if i == 0 { "" } else { "," },
            escape(&pass.name),
            pass.sum_ms / f64::from(pass.frames),
            pass.max_ms
        );
    }
    out.push_str("]}");
    out
}

/// Device names come from a driver, so they are the one string here nobody
/// controls.
fn escape(text: &str) -> String {
    text.chars()
        .flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            c if c.is_control() => vec![' '],
            c => vec![c],
        })
        .collect()
}
