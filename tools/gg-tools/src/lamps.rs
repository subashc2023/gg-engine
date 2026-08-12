//! `gg-tools lamps` — what a *casting* lamp costs, and where its bias belongs.
//!
//! §6 M30's `lights` priced a lamp that only lights. This prices the one that
//! also occludes, which is a different question with a different shape: the
//! froxel loop is per fragment and flat in the light count, while a lamp's six
//! faces are per *lamp* and paid in geometry — so where `lights` found a flat
//! line this finds a slope, and the number worth knowing is how steep.
//!
//! Two tables, and they answer to different masters.
//!
//! **Cost** sweeps the casting budget against `r.lamp_shadows 0`, which is not a
//! second code path but the same frame with `lamp::Lamps` empty. What it reports
//! per row is the frame either way, the difference *per casting lamp*, and the
//! atlas that budget allocates — because the two limits on `r.lamps` are draw
//! calls and memory and they run out at different places.
//!
//! **Bias** is `shadow-bias`' shape at a lamp: two numbers that fail in opposite
//! directions, and a plateau between them that is the answer.
//!
//! - **lost** — light missing from a floor with *nothing* between it and the
//!   lamp. That is acne, and it falls as the offset grows.
//! - **leaked** — light arriving on a floor squarely behind a wall. That is the
//!   offset pushing a receiver through its own occluder, and it rises.
//!
//! A knob whose right value is a range is not a knob one run can settle, so what
//! is printed is both columns at every bias and the **plateau** between them —
//! the first offset that clears the acne and the last that has not yet pushed a
//! receiver through its blocker. `r.lamp_normal_bias` and `r.lamp_depth_bias`
//! are read off that range, one step above its floor.

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

/// A real frame for the cost table: the shadow lookup is per fragment per
/// casting lamp, so a small extent would price the passes and not the lookup.
const EXTENT: (u32, u32) = (1280, 720);

/// Square and small for the bias table, which measures two floor regions rather
/// than a frame time.
const BIAS_EXTENT: (u32, u32) = (256, 256);

const WARMUP: usize = 3;
const FRAMES: usize = 11;

/// Casting budgets swept. 0 is `r.lamp_shadows 0` in all but name and is the
/// row every other row is read against; 8 is `lamp::MAX_LAMPS`, the ceiling the
/// frame block has room for.
const BUDGETS: [i64; 5] = [0, 1, 2, 4, 8];

/// Face edges swept, in texels — the clamp `lamp.rs` imposes, end to end.
const SIZES: [i64; 3] = [128, 256, 512];

/// Biases swept, in face texels.
const BIASES: [f64; 7] = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];

/// Lamps in the cost scene. More than any budget, so what the budget changes is
/// how many of them *cast* and never how many of them light.
const LAMPS: usize = 12;

/// A corridor with pillars — geometry every face has to record, because a lamp
/// whose faces see nothing prices the passes and not the pass.
fn hall() -> anyhow::Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Light>()?;
    let floor = world.spawn();
    world.insert(
        floor,
        Renderable::boxed(
            sim::DVec3::new(0.0, -0.1, -20.0),
            sim::Vec3::new(4.0, 0.1, 30.0),
            0x0088_8c90,
        ),
    )?;
    for i in 0u32..24 {
        let side = if i.is_multiple_of(2) { -1.0 } else { 1.0 };
        let pillar = world.spawn();
        world.insert(
            pillar,
            Renderable::boxed(
                sim::DVec3::new(side * 2.6, 1.1, -1.0 - f64::from(i / 2) * 2.4),
                sim::Vec3::new(0.25, 1.1, 0.25),
                0x00b0_a898,
            ),
        )?;
    }
    for i in 0..LAMPS {
        let side = if i.is_multiple_of(2) { -1.0 } else { 1.0 };
        let lamp = world.spawn();
        world.insert(
            lamp,
            Light::point(
                sim::DVec3::new(side * 1.6, 1.6, -2.0 - f64::from(i as u32 / 2) * 3.0),
                0x00ff_e8c0,
                14.0,
                5.0,
            ),
        )?;
    }
    Ok(world)
}

/// How thick the occluder is, half-extent in metres. **Thin on purpose**: a
/// normal offset punches a receiver through its own blocker, so an occluder
/// thicker than the largest offset swept could not leak at any bias and the
/// column meant to bound the offset from above would read the same number
/// forever. Four centimetres is inside the reach of the top of the sweep and
/// outside the reach of the plateau, which is what makes the two ends meet.
const WALL_HALF: f32 = 0.04;

/// The bias scene: a lamp, a thin wall, and a floor either side of it — the
/// offscreen test's geometry, because the question is the same one.
///
/// `lit` is what makes the ambient floor measurable: with the lamp gone, what
/// is left on both regions is [`cvars::AMBIENT`], and a "leak" measured without
/// subtracting it would be a constant the bias cannot move.
fn walled(lit: bool) -> anyhow::Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Light>()?;
    let floor = world.spawn();
    world.insert(
        floor,
        Renderable::boxed(
            sim::DVec3::new(0.0, -0.1, 0.0),
            sim::Vec3::new(8.0, 0.1, 8.0),
            0x00c0_c0c0,
        ),
    )?;
    let wall = world.spawn();
    world.insert(
        wall,
        Renderable::boxed(
            sim::DVec3::new(0.0, 0.9, 0.0),
            sim::Vec3::new(8.0, 0.9, WALL_HALF),
            0x0060_6060,
        ),
    )?;
    if lit {
        let lamp = world.spawn();
        world.insert(
            lamp,
            Light::point(sim::DVec3::new(1.3, 1.6, -3.0), 0x00ff_ffff, 40.0, 12.0),
        )?;
    }
    Ok(world)
}

fn median_frame(
    renderer: &mut OffscreenRenderer,
    world: &World,
    view: &View,
    eye: sim::DVec3,
    extent: (u32, u32),
) -> anyhow::Result<(f64, Vec<u8>)> {
    let mut extracted = Extracted::default();
    let mut times: Vec<f64> = Vec::new();
    let mut pixels = Vec::new();
    for frame in 0..WARMUP + FRAMES {
        extracted.clear(eye, view.frustum(extent));
        extracted.append::<Renderable>(world)?;
        extracted.append_lights(world)?;
        let started = std::time::Instant::now();
        let rendered = renderer.frame(&extracted, view, [0.0, 0.0, 0.0, 1.0], &[])?;
        if frame >= WARMUP {
            times.push(started.elapsed().as_secs_f64() * 1e3);
            pixels = rendered.pixels;
        }
    }
    times.sort_by(f64::total_cmp);
    Ok((times[times.len() / 2], pixels))
}

/// Mean green over image rows `rows`, in [0, 255].
fn brightness(pixels: &[u8], extent: (u32, u32), rows: core::ops::Range<u32>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0.0_f64;
    for y in rows {
        for x in 0..extent.0 {
            let i = ((y * extent.0 + x) * 4 + 1) as usize;
            sum += f64::from(pixels[i]);
            count += 1.0;
        }
    }
    sum / count.max(1.0)
}

/// Rows of floor with nothing between them and the lamp.
const OPEN_ROWS: core::ops::Range<u32> = 6..106;
/// Rows squarely behind the wall.
const BEHIND_ROWS: core::ops::Range<u32> = 150..250;

fn cost() -> anyhow::Result<()> {
    let world = hall()?;
    let mut renderer = OffscreenRenderer::new(EXTENT)?;
    println!("gg-tools lamps — {}", renderer.device().chosen);
    println!("{EXTENT:?}, median of {FRAMES} frames after {WARMUP} warm-up\n");
    let view = View::default();
    let eye = sim::DVec3::new(0.0, 1.5, 4.0);

    for size in SIZES {
        cvars::LAMP_SIZE.set_int(size);
        println!(
            "{:>4}px  {:>8} {:>9} {:>9} {:>10} {:>9}",
            size, "casting", "frame ms", "vs none", "per lamp", "atlas MiB",
        );
        cvars::LAMP_SHADOWS.set_bool(false);
        let (base, _) = median_frame(&mut renderer, &world, &view, eye, EXTENT)?;
        cvars::LAMP_SHADOWS.set_bool(true);
        for budget in BUDGETS {
            cvars::LAMPS.set_int(budget);
            let (ms, _) = median_frame(&mut renderer, &world, &view, eye, EXTENT)?;
            // Six faces of `size`² at four bytes, one row per casting lamp.
            let atlas =
                (budget.max(0) as f64) * 6.0 * (size * size) as f64 * 4.0 / (1024.0 * 1024.0);
            let per = if budget > 0 {
                (ms - base) / budget as f64
            } else {
                0.0
            };
            println!(
                "        {:>8} {:>9.2} {:>8.2}x {:>9.2} {:>9.1}",
                budget,
                ms,
                ms / base,
                per,
                atlas,
            );
        }
        println!();
    }
    cvars::LAMP_SIZE.set_int(512);
    cvars::LAMPS.set_int(4);
    let report = renderer.shutdown();
    anyhow::ensure!(
        report.clean(),
        "unclean render: {} validation message(s), {} leak(s)",
        report.validation_messages,
        report.leaked_allocations.len(),
    );
    Ok(())
}

fn bias() -> anyhow::Result<()> {
    let world = walled(true)?;
    let dark = walled(false)?;
    let mut renderer = OffscreenRenderer::new(BIAS_EXTENT)?;
    let view = View {
        pitch: -core::f32::consts::FRAC_PI_2,
        ..View::default()
    };
    let eye = sim::DVec3::new(0.0, 9.0, 0.0);

    // Three fixed points, measured once. The lamp removed gives what ambient
    // alone puts on each region — the floor neither column can go below — and
    // the lamp restored with shadows off gives what it puts there unobstructed.
    let (_, unlit) = median_frame(&mut renderer, &dark, &view, eye, BIAS_EXTENT)?;
    let (open_zero, behind_zero) = (
        brightness(&unlit, BIAS_EXTENT, OPEN_ROWS),
        brightness(&unlit, BIAS_EXTENT, BEHIND_ROWS),
    );
    cvars::LAMP_SHADOWS.set_bool(false);
    let (_, unshadowed) = median_frame(&mut renderer, &world, &view, eye, BIAS_EXTENT)?;
    let open_full = brightness(&unshadowed, BIAS_EXTENT, OPEN_ROWS) - open_zero;
    let behind_full = brightness(&unshadowed, BIAS_EXTENT, BEHIND_ROWS) - behind_zero;
    cvars::LAMP_SHADOWS.set_bool(true);
    anyhow::ensure!(
        open_full > 1.0 && behind_full > 1.0,
        "this framing has no lamp light to lose ({open_full:.2}) or to leak ({behind_full:.2})"
    );
    println!(
        "\nbias sweep — lamp light on the open floor {open_full:.1}, behind the wall \
         {behind_full:.1}, over an ambient floor of {open_zero:.1}/{behind_zero:.1}\n"
    );
    println!(
        "{:>8} {:>8} {:>9} {:>9} {:>8}",
        "normal", "depth", "lost %", "leaked %", "gap",
    );

    // The plateau, per depth term: the first offset large enough to clear the
    // filter's own acne, and the last one small enough not to push a receiver
    // through the wall in front of it. Reported as a *range* because a knob
    // whose right value is a range is not a knob one run can settle.
    let mut plateau: Vec<(f64, Option<f64>, Option<f64>)> = Vec::new();
    for depth in [0.0, 1.0, 2.0] {
        let (mut floor, mut ceiling) = (None, None);
        for normal in BIASES {
            cvars::LAMP_NORMAL_BIAS.set_float(normal);
            cvars::LAMP_DEPTH_BIAS.set_float(depth);
            let (_, pixels) = median_frame(&mut renderer, &world, &view, eye, BIAS_EXTENT)?;
            let open = brightness(&pixels, BIAS_EXTENT, OPEN_ROWS) - open_zero;
            let behind = brightness(&pixels, BIAS_EXTENT, BEHIND_ROWS) - behind_zero;
            // Both as fractions of the lamp's own contribution, so the columns
            // are comparable: `gap` is how much of the right answer this pair
            // got, and 100 is a shadow that is exactly where the geometry is.
            let lost = 100.0 * (open_full - open) / open_full;
            let leaked = 100.0 * behind / behind_full;
            if floor.is_none() && lost < 1.0 {
                floor = Some(normal);
            }
            if leaked < 1.0 {
                ceiling = Some(normal);
            }
            println!(
                "{normal:>8.1} {depth:>8.1} {lost:>9.2} {leaked:>9.2} {:>8.2}",
                100.0 - lost - leaked
            );
        }
        plateau.push((depth, floor, ceiling));
    }
    println!("\nplateau, in face texels of normal offset:");
    for (depth, floor, ceiling) in plateau {
        match (floor, ceiling) {
            (Some(lo), Some(hi)) if lo <= hi => {
                println!("  r.lamp_depth_bias {depth:.1}: {lo:.1} to {hi:.1}");
            }
            // Not a formatting case but a result: the offset that clears the
            // acne is already large enough to punch through the wall, and no
            // value of this pair is right for both.
            _ => println!("  r.lamp_depth_bias {depth:.1}: none — the two ends cross"),
        }
    }
    // Read the ceiling with suspicion when it is the top of the sweep, because
    // it usually is, and the reason is worth knowing rather than worth widening
    // the sweep over: the offset runs along the *receiver's* normal, so on a
    // floor it goes straight up. Raising a floor by a fifth of a metre does not
    // move it past a wall standing on that floor at any offset this sweep
    // reaches — which makes the acne floor the bound that binds here, and
    // peter-panning a failure of geometry this scene does not contain.
    println!(
        "  (a ceiling at {:.1} is the top of the sweep and not a measured limit — see the note \
         in this file)",
        BIASES[BIASES.len() - 1]
    );
    cvars::LAMP_NORMAL_BIAS.set_float(2.0);
    cvars::LAMP_DEPTH_BIAS.set_float(1.0);
    let report = renderer.shutdown();
    anyhow::ensure!(
        report.clean(),
        "unclean render: {} validation message(s), {} leak(s)",
        report.validation_messages,
        report.leaked_allocations.len(),
    );
    Ok(())
}

pub fn run(args: &[String]) -> anyhow::Result<()> {
    let (want_cost, want_bias) = match args.first().map(String::as_str) {
        None => (true, true),
        Some("--cost") => (true, false),
        Some("--bias") => (false, true),
        Some(other) => anyhow::bail!("gg-tools lamps: unknown argument `{other}`"),
    };
    if want_cost {
        cost()?;
    }
    if want_bias {
        bias()?;
    }
    Ok(())
}
