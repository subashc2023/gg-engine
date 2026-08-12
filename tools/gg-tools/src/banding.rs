//! `gg-tools banding` — what the 8-bit output does to a smooth gradient, and
//! what `r.dither` buys back (§6 M22).
//!
//! The desk reported "banding on the floor and walls from the light" and asked
//! whether the engine is properly HDR. It is, and that is exactly why the
//! question needed a number rather than an answer: the scene attachment is
//! `Rgba16F`, the tonemapper is the M11 curve, and the banding is manufactured
//! *after* all of it, by the swapchain's 8 bits a channel. A point light's
//! inverse-square falloff across a floor moves a code value every twenty-odd
//! pixels, and a step of one code value that wide is a contour line. No amount
//! of precision upstream of the quantizer can help; only breaking the quantizer's
//! correlation with the signal can.
//!
//! Two numbers, failing in opposite directions (the `shadow-bias` shape):
//!
//! - **run** — the mean run of *identical* code values along a scanline of a
//!   frame that is nothing but gradient, in pixels. This is the band, measured:
//!   an undithered ramp holds a value for as long as the signal takes to cross
//!   an LSB, and the eye finds the edge of that plateau unerringly. The *mean*
//!   and not the longest, which is a subtler point than it looks: correct dither
//!   leaves long runs wherever the signal sits near a code value exactly, since
//!   there is no error to spread there. Those runs are not contours and a metric
//!   that counted them would report the fix as having failed.
//! - **grain** — mean absolute residual from a local linear fit, in code values.
//!   This is what dither *costs*, and it is the number that stops "more dither"
//!   from being free: a frame dithered hard has no runs at all and looks like
//!   film.
//!
//! Neither is a gate. What the sweep decides is where `r.dither` sits: one code
//! value takes the mean run from 23 px to under 4 for a quarter of a code value
//! of grain, and every step past it buys a fifth as much run for the same grain
//! again. Neither number has a cliff, so the choice is the knee and not a
//! threshold — which is exactly why it is a CVar with an instrument behind it.

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

/// Wide enough for a run to be a run. The metric counts along x, so the width is
/// the one that matters and 720 rows is plenty of sample.
const EXTENT: (u32, u32) = (1280, 720);

/// Dither amplitudes to sweep, in output code values. `0.0` is the shipping
/// build before this milestone and is the control every other row is read
/// against.
const AMOUNTS: &[f64] = &[0.0, 0.25, 0.5, 1.0, 2.0, 4.0];

/// Window of the local linear fit that **grain** is the residual from, in
/// pixels. Wide enough that a real gradient is straight across it and narrow
/// enough that the falloff's curvature is not counted as noise.
const FIT: usize = 9;

/// Rows are skipped unless they cross at least this many code values: a row with
/// nothing happening in it has no bands to find and would dilute both numbers
/// toward zero.
const MIN_SPREAD: i32 = 6;

pub fn run(args: &[String]) -> anyhow::Result<()> {
    if let Some(arg) = args.first() {
        anyhow::bail!("unknown flag {arg:?} — banding takes none");
    }
    let world = scene()?;
    let mut renderer = OffscreenRenderer::new(EXTENT)?;

    println!();
    println!("  r.dither | run px   grain   levels");
    println!("  ---------+-------------------------");
    for &amount in AMOUNTS {
        cvars::DITHER.set_float(amount);
        let pixels = frame(&mut renderer, &world)?;
        let green = channel(&pixels);
        let (run, grain, rows) = measure(&green);
        anyhow::ensure!(
            rows > EXTENT.1 as usize / 2,
            "only {rows} of {} rows carried a gradient — the framing stopped being a ramp and \
             every number here is about something else",
            EXTENT.1
        );
        let levels = distinct(&green);
        println!("  {amount:>8.2} | {run:>6.2}  {grain:>6.3}  {levels:>6}");
        write_png(&pixels, &format!("dither-{amount}"))?;
    }
    cvars::DITHER.set_float(1.0);

    let report = renderer.shutdown();
    anyhow::ensure!(
        report.clean(),
        "unclean render: {} validation message(s), {} leak(s)",
        report.validation_messages,
        report.leaked_allocations.len(),
    );
    println!();
    println!("  frames under target/gg-tools/banding-*.png");
    Ok(())
}

/// A frame that is nothing but floor, lit by one point light off to the side.
///
/// Deliberately not a room: walls, a horizon and a shadow would all put real
/// edges in the picture, and a run metric cannot tell a real edge from the end of
/// a band. What is left is the pure case — a single smooth falloff filling every
/// pixel — which is the case the complaint is about and the only one where a run
/// length means what it says.
fn scene() -> anyhow::Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Light>()?;
    let floor = world.spawn();
    world.insert(
        floor,
        Renderable::boxed(
            sim::DVec3::new(0.0, -0.1, 0.0),
            sim::Vec3::new(60.0, 0.1, 60.0),
            0x009a_9488,
        ),
    )?;
    // Demo 12's own lamp — colour, intensity and range verbatim, because the
    // gradient's steepness is what decides how wide a band is and a proxy light
    // would measure a different picture than the one that was complained about.
    let lamp = world.spawn();
    world.insert(
        lamp,
        Light::point(sim::DVec3::new(2.0, 2.6, -5.0), 0x00ff_c890, 14.0, 11.0),
    )?;
    Ok(world)
}

fn frame(renderer: &mut OffscreenRenderer, world: &World) -> anyhow::Result<Vec<u8>> {
    // Low over the floor and pitched down, so the falloff runs across the frame
    // rather than toward the horizon — a ramp compressed into ten rows would
    // have no bands wide enough to count.
    let view = View {
        pitch: -0.5,
        ..View::default()
    };
    let eye = sim::DVec3::new(0.0, 1.7, 0.0);
    let mut extracted = Extracted::default();
    extracted.clear(eye, view.frustum(EXTENT));
    extracted.append_lights(world)?;
    extracted.append::<Renderable>(world)?;
    Ok(renderer
        .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?
        .pixels)
}

/// Green alone, and not luminance: luminance is a weighted sum of three channels
/// whose quantization noise partly cancels, which would report a picture less
/// banded than the one on the screen. Green is where two thirds of the
/// perceived light is and it is quantized on its own.
fn channel(pixels: &[u8]) -> Vec<i32> {
    pixels.chunks_exact(4).map(|p| i32::from(p[1])).collect()
}

/// `(mean run, mean residual, rows counted)`.
///
/// Both numbers come off the same rows, which is what makes them comparable: a
/// row that is flat has neither bands nor a fit worth taking, and counting it in
/// one and not the other would let a framing change move the pair in opposite
/// directions on its own.
fn measure(image: &[i32]) -> (f64, f64, usize) {
    let (w, h) = (EXTENT.0 as usize, EXTENT.1 as usize);
    let (mut runs, mut changes) = (0usize, 0usize);
    let (mut residual, mut samples, mut rows) = (0.0f64, 0usize, 0usize);
    for y in 0..h {
        let row = &image[y * w..(y + 1) * w];
        let spread = row.iter().max().unwrap_or(&0) - row.iter().min().unwrap_or(&0);
        if spread < MIN_SPREAD {
            continue;
        }
        rows += 1;
        runs += w;
        changes += (1..w).filter(|&x| row[x] != row[x - 1]).count() + 1;
        for x in FIT / 2..w - FIT / 2 {
            let window = &row[x - FIT / 2..=x + FIT / 2];
            residual += (f64::from(row[x]) - fit_at(window)).abs();
            samples += 1;
        }
    }
    let grain = match samples {
        0 => 0.0,
        n => residual / n as f64,
    };
    // Pixels over runs: a row of `w` pixels broken by `changes` boundaries holds
    // `changes` runs, so this is the mean run length whatever the row did.
    let run = match changes {
        0 => 0.0,
        n => runs as f64 / n as f64,
    };
    (run, grain, rows)
}

/// The centre of a least-squares line through `window`, which for a symmetric
/// window is just the mean — the slope term drops out at the midpoint. Written
/// as the mean with that fact named rather than as a regression nobody would
/// check.
fn fit_at(window: &[i32]) -> f64 {
    let sum: i32 = window.iter().sum();
    f64::from(sum) / window.len() as f64
}

fn distinct(image: &[i32]) -> usize {
    let mut seen = [false; 256];
    for &v in image {
        if (0..256).contains(&v) {
            seen[v as usize] = true;
        }
    }
    seen.iter().filter(|s| **s).count()
}

fn write_png(pixels: &[u8], name: &str) -> anyhow::Result<()> {
    let path = crate::output_dir()?.join(format!("banding-{name}.png"));
    let file = std::fs::File::create(&path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), EXTENT.0, EXTENT.1);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(pixels)?;
    Ok(())
}
