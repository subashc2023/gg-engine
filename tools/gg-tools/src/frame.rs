//! `gg-tools frame` — where a frame's milliseconds go (§6 M58).
//!
//! Every other instrument here answers a question about a *picture*. This one
//! answers the question a player asks, which is why the picture arrives late,
//! and it exists because the report that prompted it — ninety frames a second on
//! a 240 Hz panel, a hundred and fifty uncapped — could not be answered from
//! anything the tree wrote down. There is no profile: `xtask run` builds a shell
//! whose log goes to a console, a demo without a `game.ggproj` has no data
//! directory to write one into (§6 M41), and `gg_core::zone`'s CPU collector —
//! built at §6 M25 for exactly this — is in **no tier at all** and has no reader.
//! So the first half of the milestone is that the zones can be printed, and this
//! is what prints them.
//!
//! Two tables that answer in opposite directions. The **device** table is the
//! per-pass GPU time the render graph already measures: if the sum is a large
//! fraction of the frame, the renderer is the subject. The **host** table is
//! `gg-render`'s own `zone!`s: if the sum is the frame and the device is idle
//! inside it, the subject is CPU and the passes are innocent. A frame budget
//! argued from one of them alone is the mistake this exists to prevent — a 4090
//! finishing every pass in a third of a millisecond says nothing about whether
//! the frame took six.
//!
//! **What it is not.** Offscreen, so there is no present and no swapchain wait,
//! and the offscreen path submits *blocking* — `render.execute` therefore holds
//! record, submit **and** the device, which is why the device total is printed
//! beside it rather than under it. `render.readback` is the copy off the mapped
//! buffer, a cost a screen never pays; it is itemised so a reader subtracts it
//! rather than believing it. And the sim is absent: this crate takes demo 12
//! with `default-features = false`, so there are no `gg_game!` exports to tick.
//! What is left is the half of the frame the shell hands the renderer, measured
//! through the shipping code and on the desk's own GPU.

use std::time::Instant;

use anyhow::Result;
use gg_core::zone;
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View};

use crate::{field, views};

/// A play resolution, because the question is about a session. Overridable —
/// half the cost of a fragment-bound pass is the pixel count, and a reader
/// checking whether a frame is fill-bound wants two points.
const EXTENT: (u32, u32) = (1920, 1080);

/// Enough that a median is one, and few enough that the run is seconds. Warmup
/// is separate and larger than it looks necessary: the probe field converges
/// over `probes / r.gi_rate` frames and a frame taken while it is still
/// gathering prices a transient (§6 M57).
const WARMUP: usize = 30;
const TIMED: usize = 60;

/// Where the eye sits — `field`'s and `bounce`'s chair, so the three
/// instruments describe the same room from the same place.
const PITCH: f32 = -0.22;

pub fn run(args: &[String]) -> Result<()> {
    // `--set r.gi_filter=0` and its like, `views`' own flag: the per-pass table is
    // where a change to what a pass *does* gets priced, and pricing two of them
    // needs one binary rather than two builds (§6 M32, M69).
    views::apply_sets(args)?;
    let extent = match args.iter().position(|a| a == "--extent") {
        Some(i) => parse_extent(args.get(i + 1).map_or("", String::as_str))?,
        None => EXTENT,
    };
    let frames = match args.iter().position(|a| a == "--frames") {
        Some(i) => args
            .get(i + 1)
            .and_then(|n| n.parse().ok())
            .unwrap_or(TIMED),
        None => TIMED,
    };
    anyhow::ensure!(
        zone::enabled(),
        "this binary collects no CPU zones — `gg-tools` must resolve \
         `gg-render/cpu-timings` for the host table to exist at all"
    );

    if args.iter().any(|a| a == "--sweep") {
        return sweep(frames);
    }

    let measured = measure(extent, frames)?;
    println!(
        "demo 12's room at {}x{} on {} — median of {frames} frames after {WARMUP}, \
         {} codegen\n",
        extent.0,
        extent.1,
        measured.name,
        profile()
    );
    let wall = median_ms(&measured.wall_ns);
    passes(&measured.device);
    zones(&measured.host, &measured.extract_ns, wall);
    Ok(())
}

/// The tier the operator's window runs is the debug profile too (`xtask run`
/// builds `-p gg-runtime` with no `--release`), so a debug-profile reading is
/// the comparable one — and a release reading says how much of the frame is the
/// profile.
fn profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

/// What one extent cost, kept per frame rather than reduced here: every table
/// below takes a median, and one stalled frame is not the frame.
struct Measured {
    name: String,
    device: Vec<Vec<(String, f32)>>,
    host: Vec<Vec<zone::Sample>>,
    extract_ns: Vec<u64>,
    wall_ns: Vec<u64>,
}

/// A device per extent, deliberately — [`OffscreenRenderer`] takes its extent at
/// bring-up, so a sweep has to rebuild anyway, and a renderer carrying the
/// previous row's transients would price this row's allocation as the last
/// one's (§6 M67's lesson, one instrument over).
fn measure(extent: (u32, u32), frames: usize) -> Result<Measured> {
    let world = field::world()?;
    let mut renderer = OffscreenRenderer::new(extent)?;
    let name = renderer.device().chosen.clone();

    let mut host: Vec<Vec<zone::Sample>> = Vec::new();
    let mut device: Vec<Vec<(String, f32)>> = Vec::new();
    let mut extract_ns: Vec<u64> = Vec::new();
    let mut wall_ns: Vec<u64> = Vec::new();
    for i in 0..WARMUP + frames {
        // Drained every frame whether or not it is kept: the collector grows
        // until someone takes it, so a warmup frame left in would be charged to
        // the first timed one.
        let _ = zone::take();
        let started = Instant::now();
        let built = Instant::now();
        let extracted = extract(&world, extent)?;
        let extracted_ns = built.elapsed().as_nanos() as u64;
        let view = View {
            pitch: PITCH,
            ..View::default()
        };
        renderer.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?;
        if i < WARMUP {
            continue;
        }
        wall_ns.push(started.elapsed().as_nanos() as u64);
        extract_ns.push(extracted_ns);
        host.push(zone::take());
        device.push(
            renderer
                .pass_timings()
                .iter()
                .map(|t| (t.name.clone(), t.gpu_ms))
                .collect(),
        );
    }

    let report = renderer.shutdown();
    anyhow::ensure!(report.clean(), "unclean render: {report:?}");
    Ok(Measured {
        name,
        device,
        host,
        extract_ns,
        wall_ns,
    })
}

/// Sixteen-by-nine from a play resolution down to an eighth of its pixels. Real
/// presented sizes rather than fractions of the first, so a reader finds their
/// own panel in the left column instead of doing arithmetic to reach it.
const SWEEP: [(u32, u32); 5] = [
    (1920, 1080),
    (1600, 900),
    (1280, 720),
    (960, 540),
    (640, 360),
];

/// The sweep: one row per extent on **one** device, which is the only way to ask
/// whether a frame is *fill-bound* — a single reading cannot, and two readings
/// on two machines answer a different question (this room costs 0.94 ms on a
/// 4090 and 79.4 on the integrated Radeon beside it, so the desk that develops
/// the renderer is the one that cannot feel its cost).
///
/// The column that answers is **ms/Mpx**: flat means every millisecond is
/// fragments and a render scale buys the lot; falling as the extent grows means
/// a fixed cost is being amortised and the scale buys less than the pixel ratio
/// suggests.
fn sweep(frames: usize) -> Result<()> {
    let mut rows: Vec<Row> = Vec::new();
    for extent in SWEEP {
        let measured = measure(extent, frames)?;
        rows.push(Row {
            extent,
            name: measured.name,
            device: measured.device,
        });
    }
    let name = rows.first().map_or("", |row| row.name.as_str());
    println!(
        "demo 12's room on {name} — median of {frames} frames after {WARMUP}, {} codegen\n",
        profile()
    );

    let totals: Vec<(f64, f64)> = rows
        .iter()
        .map(|row| (mpx(row.extent), total_ms(&row.device)))
        .collect();
    let full = totals.first().map_or(0.0, |(_, ms)| *ms);
    println!("  extent          Mpx   device ms   ms/Mpx   against full");
    for (row, (px, ms)) in rows.iter().zip(&totals) {
        // Against the first row rather than against the ideal: what a reader
        // wants is what the knob buys them, and the ideal is the next column's
        // job.
        let against = match full {
            f if f > 0.0 => format!("{:>6.2}x", f / ms),
            _ => "     -".to_owned(),
        };
        let per = match px {
            p if *p > 0.0 => ms / px,
            _ => 0.0,
        };
        println!(
            "  {:>4}x{:<9} {px:>5.3}   {ms:>9.3}   {per:>6.2}   {against}",
            row.extent.0, row.extent.1
        );
    }
    let (floor, per) = fit(&totals);
    println!(
        "\n  fill {per:.2} ms/Mpx, floor {floor:.3} ms — {:.0}% of the frame at {}x{} is \
         fragments\n",
        match full {
            f if f > 0.0 => 100.0 * (f - floor) / f,
            _ => 0.0,
        },
        SWEEP[0].0,
        SWEEP[0].1
    );

    // Per pass, the same two numbers — and they fail in opposite directions: a
    // pass with no floor is bought outright by a render scale, one with no
    // slope is untouched by it and stays whatever it was.
    let mut names: Vec<&str> = rows
        .iter()
        .flat_map(|row| {
            row.device
                .iter()
                .flat_map(|f| f.iter().map(|(n, _)| n.as_str()))
        })
        .collect();
    names.sort_unstable();
    names.dedup();
    let mut per_pass: Vec<(&str, f64, f64, f64)> = names
        .iter()
        .map(|name| {
            let points: Vec<(f64, f64)> = rows
                .iter()
                .map(|row| (mpx(row.extent), pass_ms(&row.device, name)))
                .collect();
            let (floor, per) = fit(&points);
            (*name, points.first().map_or(0.0, |(_, ms)| *ms), floor, per)
        })
        .collect();
    per_pass.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("  pass                              at full   ms/Mpx    floor   fill");
    for (name, at_full, floor, per) in &per_pass {
        let share = match at_full {
            f if *f > 0.0 => format!("{:>5.0}%", 100.0 * (f - floor).max(0.0) / f),
            _ => "    -".to_owned(),
        };
        println!("  {name:<32}  {at_full:>7.3}   {per:>6.2}   {floor:>6.3}   {share}");
    }
    Ok(())
}

/// One extent's reading, kept whole so the two tables below read the same rows.
struct Row {
    extent: (u32, u32),
    name: String,
    device: Vec<Vec<(String, f32)>>,
}

/// Megapixels, the axis every fit here runs against — a fragment-bound pass
/// costs its pixel *count* and not its width.
fn mpx(extent: (u32, u32)) -> f64 {
    f64::from(extent.0) * f64::from(extent.1) / 1e6
}

fn total_ms(device: &[Vec<(String, f32)>]) -> f64 {
    let per_frame: Vec<u64> = device
        .iter()
        .map(|f| (f.iter().map(|(_, ms)| f64::from(*ms)).sum::<f64>() * 1e6) as u64)
        .collect();
    median_ms(&per_frame)
}

fn pass_ms(device: &[Vec<(String, f32)>], name: &str) -> f64 {
    let per_frame: Vec<u64> = device
        .iter()
        .map(|f| {
            let sum: f64 = f
                .iter()
                .filter(|(n, _)| n == name)
                .map(|(_, ms)| f64::from(*ms))
                .sum();
            (sum * 1e6) as u64
        })
        .collect();
    median_ms(&per_frame)
}

/// Least squares of `ms = floor + per * mpx`. Reported as two numbers rather
/// than as one goodness-of-fit, because the two are what a reader acts on and a
/// correlation coefficient is what they would have to interpret.
fn fit(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    if n < 2.0 {
        return (points.first().map_or(0.0, |(_, y)| *y), 0.0);
    }
    let mean_x = points.iter().map(|(x, _)| x).sum::<f64>() / n;
    let mean_y = points.iter().map(|(_, y)| y).sum::<f64>() / n;
    let spread: f64 = points.iter().map(|(x, _)| (x - mean_x).powi(2)).sum();
    if spread <= 0.0 {
        return (mean_y, 0.0);
    }
    let cross: f64 = points
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum();
    let per = cross / spread;
    (mean_y - per * mean_x, per)
}

/// The shell's extract order (`App::extract`), the one `field` restates for the
/// same reason: a different visible set is a different frame.
fn extract(world: &gg_ecs::World, extent: (u32, u32)) -> Result<Extracted> {
    let view = View {
        pitch: PITCH,
        ..View::default()
    };
    let eye = sim::DVec3::new(0.0, 1.62, 8.0);
    let mut extracted = Extracted::default();
    extracted.clear(eye, view.frustum(extent));
    extracted.append_lights(world)?;
    extracted.cast_shadows(view.caster_reach(extent));
    extracted.append::<gg_ecs::boundary::Renderable>(world)?;
    Ok(extracted)
}

/// The device table: what each pass cost the GPU, and the sum.
///
/// Sorted by cost rather than by graph order on purpose — the question is where
/// the time is, and a reader who wants the order has `gg-golden graph`.
fn passes(frames: &[Vec<(String, f32)>]) {
    let mut names: Vec<&str> = frames
        .iter()
        .flat_map(|f| f.iter().map(|(n, _)| n.as_str()))
        .collect();
    names.sort_unstable();
    names.dedup();
    let mut rows: Vec<(&str, f64)> = names
        .iter()
        .map(|name| {
            // Summed within a frame before the median is taken across frames:
            // the graph declares one name per shadow slice and per probe face,
            // so a per-row median would report one slice and call it the pass.
            let per_frame: Vec<u64> = frames
                .iter()
                .map(|f| {
                    let sum: f64 = f
                        .iter()
                        .filter(|(n, _)| n == name)
                        .map(|(_, ms)| f64::from(*ms))
                        .sum();
                    (sum * 1e6) as u64
                })
                .collect();
            (*name, median_ms(&per_frame))
        })
        .collect();
    rows.sort_by(|a, b| b.1.total_cmp(&a.1));
    let total: f64 = rows.iter().map(|(_, ms)| ms).sum();
    println!("  device                                gpu ms    share");
    for (name, ms) in &rows {
        // Below a microsecond is below what a timestamp period resolves on most
        // queues; printed as a dash rather than as a suspiciously exact zero.
        let share = match total {
            t if t > 0.0 => format!("{:>6.1}%", 100.0 * ms / t),
            _ => "     -".to_owned(),
        };
        println!("  {name:<34}  {ms:>7.3}   {share}");
    }
    println!("  {:<34}  {total:>7.3}\n", "device total");
}

/// The host table: `gg-render`'s own zones, plus the two costs measured here
/// rather than inside it — the extract the shell does before the call, and the
/// wall clock around the whole thing.
fn zones(frames: &[Vec<zone::Sample>], extract_ns: &[u64], wall: f64) {
    let mut names: Vec<(&'static str, u16)> = frames
        .iter()
        .flat_map(|f| f.iter().map(|s| (s.name, s.depth)))
        .collect();
    names.sort_unstable();
    names.dedup();
    let mut rows: Vec<(&str, u16, f64)> = names
        .iter()
        .map(|&(name, depth)| {
            let per_frame: Vec<u64> = frames
                .iter()
                .map(|f| f.iter().filter(|s| s.name == name).map(|s| s.nanos).sum())
                .collect();
            (name, depth, median_ms(&per_frame))
        })
        .collect();
    // Depth first so the outermost zone reads as the total it is, then cost —
    // the two orderings a reader wants and neither of them alphabetical.
    rows.sort_by(|a, b| a.1.cmp(&b.1).then(b.2.total_cmp(&a.2)));
    let outer: f64 = rows
        .iter()
        .filter(|(_, depth, _)| *depth == 0)
        .map(|(_, _, ms)| ms)
        .sum();
    println!("  host                                   cpu ms    share");
    let extracted = median_ms(extract_ns);
    println!(
        "  {:<34}  {extracted:>7.3}   {:>6.1}%",
        "extract (this harness)",
        share(extracted, wall)
    );
    for (name, depth, ms) in &rows {
        let indent = " ".repeat(usize::from(*depth) * 2);
        println!(
            "  {indent}{name:<width$}  {ms:>7.3}   {:>6.1}%",
            share(*ms, wall),
            width = 34 - usize::from(*depth) * 2
        );
    }
    println!(
        "  {:<34}  {wall:>7.3}   (render.frame {outer:.3} + extract)",
        "wall, extract to pixels"
    );
}

fn share(part: f64, whole: f64) -> f64 {
    match whole {
        w if w > 0.0 => 100.0 * part / w,
        _ => 0.0,
    }
}

fn median_ms(nanos: &[u64]) -> f64 {
    if nanos.is_empty() {
        return 0.0;
    }
    let mut v = nanos.to_vec();
    v.sort_unstable();
    v[v.len() / 2] as f64 / 1e6
}

fn parse_extent(text: &str) -> Result<(u32, u32)> {
    let (w, h) = text
        .split_once('x')
        .ok_or_else(|| anyhow::anyhow!("--extent wants `<w>x<h>`, got {text:?}"))?;
    Ok((w.parse()?, h.parse()?))
}
