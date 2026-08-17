//! `gg-tools field` — whether the probe field survives the camera moving (§6 M36).
//!
//! `gg-tools bounce` grades the field's *accuracy* from one chair. This grades
//! its **stability** from a moving one, which is a different failure and the one
//! a player reports: the grid is fitted to the bounds of what the frustum left
//! (`probe::bounds_of` reads `visible()`), so a camera that changes what it can
//! see changes the fit, and a fit that changes throws the whole field away —
//! `Probes::update` marks every probe ungathered and restarts the round robin.
//! The field then re-converges over `probes / r.gi_rate` frames with the images
//! still holding irradiance addressed against a grid that no longer exists.
//!
//! Three legs, chosen to isolate the axis rather than to look like play: the eye
//! **walks** at a fixed height, **turns** on the spot, and **jumps** on the spot
//! along demo 12's own arc. A defect that only the third one shows is a defect in
//! the height term, which is what makes this a diagnosis rather than a
//! reproduction.
//!
//! Two numbers that fail in opposite directions. **Refits** is how often the
//! field is discarded — a grid that never moves scores zero and may still be the
//! wrong grid. **Settled** is the mean fraction of it that was gathered against
//! the grid in hand — a field refitted every frame still shows a full batch of
//! fresh probes and is nearly all stale. Neither alone is the complaint; the
//! complaint is **swing**, the frame-to-frame change in what the room's light
//! actually is, and it is printed beside them so the two structural columns are
//! attributed to a picture rather than asserted against one.

use anyhow::Result;
use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable, Sky};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

use demo_12_shooter as demo;

use crate::views;

/// The frame the legs are measured over. Small for `bounce`'s reason — the
/// subject is a *mean* over the picture, which converges long before a
/// silhouette does — and small again because every frame here pays
/// `r.gi_rate * 6` views of the scene on a software rasterizer.
const EXTENT: (u32, u32) = (240, 135);

/// Where the walk starts: demo 12's spawn at eye height, `bounce`'s chair, so
/// the three instruments describe the same room from the same place.
const START: sim::DVec3 = sim::DVec3::new(0.0, 1.62, 8.0);
const PITCH: f32 = -0.22;

/// Frames per leg. Past demo 12's own jump arc (42 ticks at [`demo::GRAVITY`]),
/// so the height leg lands and stands still for the remainder — which is what
/// makes its tail comparable with the level leg's.
const FRAMES: usize = 60;

/// The per-frame trace is long; only the leg named here prints one by default.
const TRACED: &str = "jump";

pub fn run(args: &[String]) -> Result<()> {
    // `--set r.gi_filter=0` and its like: the cost table's only way to price two
    // reads of the same tile in one binary (§6 M69), and `views`' own flag so an
    // operator does not have to learn a second spelling.
    views::apply_sets(args)?;
    let traced = match args.iter().position(|a| a == "--trace") {
        Some(i) => args.get(i + 1).map_or("", String::as_str).to_owned(),
        None => TRACED.to_owned(),
    };
    let world = world()?;
    if args.iter().any(|a| a == "--cost") {
        return cost(&world);
    }
    println!(
        "demo 12's room from the spawn, {}x{} — r.gi_spacing {:.2}, r.gi_rate {}, {} frames a leg\n",
        EXTENT.0,
        EXTENT.1,
        cvars::GI_SPACING.float(),
        cvars::GI_RATE.int(),
        FRAMES
    );
    // Refits beside scrolls, because they are opposite events: the first is the
    // field discarded and is what has to stay at zero, the second is the window
    // sliding with every record it can keep, and a player walking produces them.
    println!("  leg      refits  scrolls  settled   swing%   worst%   grids");
    let mut traces = Vec::new();
    for leg in LEGS {
        // A renderer a leg, and not for tidiness: the fit only ever grows while
        // it covers, so a leg inheriting the last one's grid would score its
        // predecessor's refits as stability. `Probes` has no reset and should
        // not grow one for an instrument.
        let mut renderer = OffscreenRenderer::new(EXTENT)?;
        let trace = walk(&mut renderer, &world, leg)?;
        report(leg.name, &trace);
        if leg.name == traced {
            traces.push(trace);
        }
        let report = renderer.shutdown();
        anyhow::ensure!(report.clean(), "unclean render: {report:?}");
    }
    for trace in &traces {
        detail(&traced, trace);
    }
    Ok(())
}

/// What the field costs a frame that is standing still, which is the other half
/// of a player's complaint about it (`--cost`).
///
/// A probe is **six views of the scene** and [`cvars::GI_RATE`] of them are taken
/// every frame *forever* — the round robin has no rest state, so a converged
/// field over a scene where nothing moves is re-gathered at the same rate as one
/// chasing a collapsing building. Sixteen is ninety-six extra views a frame
/// beside the four lamps' twenty-four and the sun's cascades. Whether that is the
/// right trade is a judgement; what it is not is a number anyone had.
///
/// Priced at a play resolution and on a real GPU, because the pinned rasterizer
/// answers a question about lavapipe. `r.gi_rate 0` is one whole batch rather
/// than none (see [`cvars::GI_RATE`]), so the floor row turns `r.gi` off.
const COST_EXTENT: (u32, u32) = (1920, 1080);
const COST_RATES: [i64; 5] = [1, 4, 8, 16, 32];
const WARMUP: usize = 20;
const TIMED: usize = 40;

/// Frames past which an unsettled field is a defect rather than a transient — the
/// largest grid `probe::MAX_PER_AXIS` allows, one batch a frame, and margin.
const SETTLE_CEILING: usize = 256;

fn cost(world: &World) -> Result<()> {
    let mut renderer = OffscreenRenderer::new(COST_EXTENT)?;
    println!(
        "the field's price — demo 12's room at {}x{}, median of {TIMED} frames\n",
        COST_EXTENT.0, COST_EXTENT.1
    );
    println!("  r.gi_rate   views   probe ms   lamp ms   frame ms");
    let rate = cvars::GI_RATE.int();
    cvars::GI.set_bool(false);
    let (probe, lamp, frame) = price(&mut renderer, world)?;
    println!(
        "  {:>9}   {:>5}   {probe:>8.3}   {lamp:>7.3}   {frame:>8.3}",
        "off", 0
    );
    cvars::GI.set_bool(true);
    for r in COST_RATES {
        cvars::GI_RATE.set_int(r);
        let (probe, lamp, frame) = price(&mut renderer, world)?;
        // Six faces a probe — `crate::lamp::FACES`, which a probe shares and
        // which is a cube's and not a knob.
        println!(
            "  {r:>9}   {:>5}   {probe:>8.3}   {lamp:>7.3}   {frame:>8.3}",
            r * 6
        );
    }
    cvars::GI_RATE.set_int(rate);
    let report = renderer.shutdown();
    anyhow::ensure!(report.clean(), "unclean render: {report:?}");
    Ok(())
}

/// Median probe-pass, lamp-pass and wall-clock frame milliseconds.
fn price(renderer: &mut OffscreenRenderer, world: &World) -> Result<(f64, f64, f64)> {
    let (mut probes, mut lamps, mut frames) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..WARMUP + TIMED {
        let started = std::time::Instant::now();
        frame(renderer, world, START, 0.0, COST_EXTENT)?;
        if i < WARMUP {
            continue;
        }
        frames.push(started.elapsed().as_secs_f64() * 1e3);
        let sum = |m: fn(&str) -> bool| -> f64 {
            renderer
                .pass_timings()
                .iter()
                .filter(|t| m(&t.name))
                .map(|t| f64::from(t.gpu_ms))
                .sum()
        };
        probes.push(sum(|n| n.contains("probe")));
        lamps.push(sum(|n| n.contains("lamp")));
    }
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    Ok((median(probes), median(lamps), median(frames)))
}

/// One frame's observation: what the eye was, what the renderer decided, and
/// what came out.
struct Frame {
    eye: sim::DVec3,
    yaw: f32,
    visible: usize,
    grid: Option<(sim::DVec3, f32, [u32; 3])>,
    pending: usize,
    probes: usize,
    /// Grids discarded and grids slid, as the renderer counts them (§6 M68).
    ///
    /// **Counted and not inferred.** This instrument used to read "the grid tuple
    /// changed" as a refit, which was exact while every grid was fitted to the scene
    /// and became wrong the moment one could follow the eye: an anchored grid moves
    /// its origin every time the eye crosses a cell, keeping every record, and a
    /// walking player would have read as a field thrown away thirty times. The
    /// distinction is the whole of §6 M57's claim, so it comes from the renderer.
    events: (usize, usize),
    /// Mean linear luminance of the picture — the room's light in one number.
    mean: f32,
}

struct Leg {
    name: &'static str,
    /// Eye and yaw at frame `i`. A closure rather than a table so the arc is the
    /// demo's constants and not a transcription of them.
    at: fn(usize) -> (sim::DVec3, f32),
}

const LEGS: &[Leg] = &[
    Leg {
        name: "level",
        // Forward at the demo's own walk speed, height and heading fixed.
        at: |i| {
            let mut eye = START;
            eye.z -= demo::WALK_SPEED * i as f64;
            (eye, 0.0)
        },
    },
    Leg {
        name: "turn",
        // On the spot through a half circle — the frustum sweeps the whole room
        // without the eye moving a millimetre.
        at: |i| {
            let sweep = core::f32::consts::PI * i as f32 / FRAMES as f32;
            (START, sweep)
        },
    },
    Leg {
        name: "jump",
        // Demo 12's arc exactly: `JUMP_VELOCITY` up, `GRAVITY` a tick, clamped
        // at the ground so the tail of the leg is a standing camera.
        at: |i| {
            let t = i as f64;
            let rise = (demo::JUMP_VELOCITY * t - 0.5 * demo::GRAVITY * t * t).max(0.0);
            let mut eye = START;
            eye.y += rise;
            (eye, 0.0)
        },
    },
];

/// Drive one leg, one render a frame, in the shell's own extract order.
fn walk(renderer: &mut OffscreenRenderer, world: &World, leg: &Leg) -> Result<Vec<Frame>> {
    settle(renderer, world, leg)?;
    let mut trace = Vec::new();
    for i in 0..FRAMES {
        let (eye, yaw) = (leg.at)(i);
        trace.push(frame(renderer, world, eye, yaw, EXTENT)?);
    }
    Ok(trace)
}

/// Bring the leg's first frame to a converged field, so what the trace measures
/// is the walk and not the renderer's first-frame cold start.
fn settle(renderer: &mut OffscreenRenderer, world: &World, leg: &Leg) -> Result<()> {
    let (eye, yaw) = (leg.at)(0);
    let rate = cvars::GI_RATE.int();
    cvars::GI_RATE.set_int(0);
    // **Until it is settled, not for two frames** (§6 M68). `r.gi_rate 0` is one
    // *batch* and not one grid, so this gathered 128 probes — which was most of a
    // 192-probe field and is a seventh of a 900-probe one. What it cost is the
    // column beside it: `settled` read 0.66 and the swing it reports was partly the
    // field still arriving, which is the transient this leg exists to distinguish
    // from a flicker.
    let mut frames = 0;
    loop {
        frame(renderer, world, eye, yaw, EXTENT)?;
        frames += 1;
        let (pending, probes) = renderer.field_pending();
        if pending == 0 {
            break;
        }
        anyhow::ensure!(
            frames < SETTLE_CEILING,
            "the field did not settle in {frames} frames: {pending} of {probes} ungathered"
        );
    }
    cvars::GI_RATE.set_int(rate);
    Ok(())
}

/// One render, and everything observable about it.
fn frame(
    renderer: &mut OffscreenRenderer,
    world: &World,
    eye: sim::DVec3,
    yaw: f32,
    extent: (u32, u32),
) -> Result<Frame> {
    let view = View {
        pitch: PITCH,
        yaw,
        ..View::default()
    };
    // `gg-runtime`'s order (`App::extract`): lights, then the caster sweep, then
    // the instances the sweep widened. Restating it here would measure a
    // different visible set from the one the shell hands the field.
    let mut extracted = Extracted::default();
    extracted.clear(eye, view.frustum(extent));
    extracted.append_lights(world)?;
    extracted.cast_shadows(view.caster_reach(extent));
    extracted.append::<Renderable>(world)?;
    let pixels = renderer
        .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?
        .pixels;
    let sum: f64 = pixels
        .chunks_exact(4)
        .map(|p| f64::from(0.2126 * decode(p[0]) + 0.7152 * decode(p[1]) + 0.0722 * decode(p[2])))
        .sum();
    let (pending, probes) = renderer.field_pending();
    Ok(Frame {
        eye,
        yaw,
        visible: extracted.visible().len(),
        grid: renderer.field_grid(),
        pending,
        probes,
        events: renderer.field_events(),
        mean: (sum / (pixels.len() / 4) as f64) as f32,
    })
}

/// sRGB decode — `bounce`'s, restated for that module's reason: the two never
/// meet at runtime.
fn decode(code: u8) -> f32 {
    let e = f32::from(code) / 255.0;
    if e <= 0.040_45 {
        e / 12.92
    } else {
        sim::powf((e + 0.055) / 1.055, 2.4)
    }
}

fn report(name: &str, trace: &[Frame]) {
    let moved = |pick: fn(&(usize, usize)) -> usize| match (trace.first(), trace.last()) {
        (Some(a), Some(b)) => pick(&b.events).saturating_sub(pick(&a.events)),
        _ => 0,
    };
    let refits = moved(|e| e.0);
    let scrolls = moved(|e| e.1);
    let settled = trace
        .iter()
        .map(|f| match f.probes {
            0 => 1.0,
            n => 1.0 - f.pending as f64 / n as f64,
        })
        .sum::<f64>()
        / trace.len() as f64;
    let swings: Vec<f64> = trace
        .windows(2)
        .map(|w| match w[0].mean {
            m if m > 1e-6 => 100.0 * f64::from((w[1].mean - m).abs() / m),
            _ => 0.0,
        })
        .collect();
    let mean = swings.iter().sum::<f64>() / swings.len().max(1) as f64;
    let worst = swings.iter().copied().fold(0.0, f64::max);
    // Distinct grids, not refits: a fit that oscillates between two settles on
    // neither, and the count is what tells that apart from one that grew once.
    let mut grids: Vec<String> = trace.iter().map(|f| format!("{:?}", f.grid)).collect();
    grids.sort();
    grids.dedup();
    println!(
        "  {name:<8} {refits:>6}  {scrolls:>7}   {settled:>6.2}   {mean:>6.2}   {worst:>6.2}            {:>5}",
        grids.len()
    );
}

/// The frame-by-frame trace of one leg, which is where the mechanism is legible
/// rather than merely counted.
fn detail(name: &str, trace: &[Frame]) {
    println!("\n  {name}, frame by frame\n");
    println!("  frame   eye y     yaw   visible   spacing   counts    ungathered   mean L     d%");
    let mut last: Option<&Frame> = None;
    for (i, f) in trace.iter().enumerate() {
        let (spacing, counts) = match f.grid {
            Some((_, s, c)) => (s, format!("{}x{}x{}", c[0], c[1], c[2])),
            None => (0.0, "-".to_owned()),
        };
        let delta = match last {
            Some(p) if p.mean > 1e-6 => {
                format!("{:6.2}", 100.0 * (f.mean - p.mean).abs() / p.mean)
            }
            _ => "     -".to_owned(),
        };
        let mark = match last {
            Some(p) if p.events.0 != f.events.0 => "  refit",
            Some(p) if p.events.1 != f.events.1 => "  scroll",
            _ => "",
        };
        println!(
            "  {i:>5}  {:>6.2}  {:>6.2}   {:>7}   {spacing:>7.2}   {counts:<8}  {:>4} / {:<4}  {:>7.5} {delta}{mark}",
            f.eye.y, f.yaw, f.visible, f.pending, f.probes, f.mean
        );
        last = Some(f);
    }
}

/// Demo 12's room as the renderer draws it, with the sky volumes and the lights
/// the demo declares — `bounce`'s two world builders joined, because the subject
/// here is the *whole* lit picture rather than the indirect term in isolation.
pub fn world() -> Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Sky>()?;
    world.register::<Light>()?;
    for &(center, half_extent, color) in demo::ROOM {
        let entity = world.spawn();
        world.insert(entity, Renderable::boxed(center, half_extent, color))?;
    }
    let mut sky = |s: Sky| -> Result<()> {
        let entity = world.spawn();
        world.insert(entity, s)?;
        Ok(())
    };
    sky(Sky::daylight(demo::SKY_INTENSITY))?;
    sky(Sky {
        zenith: 0x0056_5a60,
        horizon: 0x006a_6862,
        ground: 0x0048_423a,
        ..Sky::daylight(demo::SHELTER_INTENSITY)
    }
    .within(demo::SHELTER.0, demo::SHELTER.1, demo::SHELTER_FADE))?;
    let entity = world.spawn();
    world.insert(
        entity,
        Light::sun(demo::SUN, demo::SUN_INK, demo::SUN_INTENSITY),
    )?;
    for at in demo::LAMPS {
        let entity = world.spawn();
        world.insert(
            entity,
            Light::point(at, demo::LAMP_INK, demo::LAMP_INTENSITY, demo::LAMP_RANGE),
        )?;
    }
    Ok(world)
}
