//! `gg-tools shadow-edge` — how wide a shadow's boundary is on the screen, and
//! how straight (§6 M22, corrected at M23).
//!
//! The desk reported that further shadows are "kinda super aliased", and the
//! screenshot that came with the second report is what finally settled it: the
//! boundary crossed **140 levels in one pixel**, while the silhouette of the very
//! cube casting it came out smooth two centimetres away. That pairing is the
//! whole diagnosis. A cube's silhouette is *coverage*, which MSAA resolves. A
//! shadow boundary is a shading discontinuity **inside** a triangle — every
//! sample in the pixel is on the same triangle and takes the same shadow tap —
//! so no sample count has ever touched it, and the only thing that can is a
//! filter at least a pixel wide.
//!
//! Two numbers, failing in opposite directions (the `shadow-bias` shape):
//!
//! - **soft** — the 20%→80% width of the crossing, in pixels. This is the one
//!   that matters and this instrument used to bury it. A boundary under a pixel
//!   wide is a staircase whatever else is true of it.
//! - **jag** — RMS deviation of the half-lit crossing from the straight line a
//!   straight caster over a flat floor must produce, in pixels. This is what
//!   stops "blur it more" from winning outright, together with the frame written
//!   out beside every row: a kernel wide enough to have no jag at all has turned
//!   a contact shadow into a smudge.
//!
//! # What this instrument got wrong, twice
//!
//! Worth keeping, because both failures were failures of the *measurement* while
//! the picture on the desk was right all along. First it led with **jag**, and
//! jag on a near-straight edge is small whether or not the edge is one pixel
//! hard — three reports in a row ended "did not reproduce" on the strength of a
//! number that could not see the defect. Second, its sun had no lateral
//! component, so the boundary lay *along* the map's texel grid: the single
//! framing where no filter has anything to do.
//!
//! Nothing here gates. The gate this feeds is
//! `gg-render/tests/shadow_softness.rs`.

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

/// 16:9 at the desk's own width, because the metric is in *pixels* and a
/// narrower frame would put fewer of them across the same texel.
const EXTENT: (u32, u32) = (1280, 720);

/// Where the eye stands, in metres back from the caster along +Z.
const DISTANCES: &[f64] = &[18.0, 35.0];

/// Shadow map edges to sweep. The axis that matters is **texels per screen
/// pixel**, and this is the clean way to move it: at the shipping 2048 a far
/// cascade's texel is under a pixel at these distances, and every step down
/// doubles what one texel covers. A world four times the size of demo 12's
/// reaches the same ratios at 2048, which is what the sweep is standing in for.
const SIZES: &[i64] = &[2048, 1024, 512];

/// Screen-pixel floors to sweep — `r.shadow_softness`'s own range. 0.0 is the
/// pre-M23 picture and is the control every other row is read against; 2.0 is
/// what ships.
const SOFTNESS: &[f64] = &[0.0, 1.0, 2.0, 3.0, 4.0];

/// The caster: a long slab held above the floor, running along X so its shadow
/// crosses every scanline exactly once and the edge to fit is horizontal in the
/// world and vertical in nothing — a shadow boundary that lay along a scanline
/// would be sampled once per row and could not be fitted at all.
const SLAB_HALF: [f32; 3] = [60.0, 0.15, 2.0];
const SLAB_HEIGHT: f64 = 3.0;

/// Off the world axes, so the boundary crosses the map's own texel grid
/// obliquely. An axis-aligned sun lays the edge *along* the grid, which is the
/// single framing where no filter has anything to do — and is how two earlier
/// rounds of this instrument measured a perfectly straight edge either way.
const SUN: [f32; 3] = [0.4, -0.55, -1.0];

/// How the eye is placed at each leg, as multiples of the leg's distance:
/// `distance` metres of it up and a quarter back, pitched a radian down.
///
/// Steep on purpose. The obvious framing — an eye near the floor looking along
/// it — puts the receiver at a grazing angle, and everything about the
/// measurement then fights it: a pixel covers several metres along the view and
/// a few centimetres across it, so a penumbra read *down a column* is compressed
/// by the anisotropy, and a window wide enough to contain the ramp is also wide
/// enough to contain the floor's own shading gradient. Between them those two
/// held every width past about 1.4 px to 1.4 px. Looking down at the floor costs
/// nothing the measurement needs and removes both.
const EYE_UP: f64 = 0.78;
const EYE_BACK: f64 = 0.25;
const EYE_PITCH: f32 = -1.0;

/// The shipping defaults, restated so every leg sets every knob and no row
/// inherits the previous row's state.
const SHIP_SIZE: i64 = 2048;
const SHIP_DISTANCE: f64 = 80.0;
const SHIP_CASCADES: i64 = 4;
const SHIP_SOFTNESS: f64 = 2.0;
const SHIP_SUN_ANGLE: f64 = 0.53;
const SHIP_TAPS: i64 = 16;
const SHIP_PENUMBRA: f64 = 16.0;

/// Columns whose crossing is this far from the *median* crossing are dropped
/// before the line is fitted — a column that clipped the slab's end, or found no
/// shadow at all, is not a sample of the edge. Taken off the median and not off a
/// first fit: a fit is the thing the outliers break, so using it to find them is
/// the circle this exists to avoid, and the edge is near horizontal by
/// construction so a median is a fair centre for it.
const OUTLIER: f64 = 24.0;

pub fn run(args: &[String]) -> anyhow::Result<()> {
    if let Some(arg) = args.first() {
        anyhow::bail!("unknown flag {arg:?} — shadow-edge takes none");
    }
    let mut renderer = OffscreenRenderer::new(EXTENT)?;
    ship();

    println!();
    println!("  the map's grain, at the shipping filter");
    println!("  eye m  mm/px | r.shadow_size | soft px  jag px | columns");
    println!("  -------------+---------------+-----------------+--------");
    for &distance in DISTANCES {
        for &size in SIZES {
            ship();
            cvars::SHADOW_SIZE.set_int(size);
            let edge = measure(&mut renderer, distance, &format!("{distance:.0}m-{size}"))?;
            println!(
                "  {distance:>5.0}  {:>5.1} | {size:>13} | {:>6.2}  {:>6.2} | {:>7}",
                1000.0 * millimetres_per_pixel(distance),
                edge.soft,
                edge.jag,
                edge.rows,
            );
        }
    }

    println!();
    println!("  the screen floor, at the shipping map");
    println!("  eye m  mm/px | r.shadow_softness | soft px  jag px | columns");
    println!("  -------------+-------------------+-----------------+--------");
    for &distance in DISTANCES {
        for &width in SOFTNESS {
            ship();
            cvars::SHADOW_SOFTNESS.set_float(width);
            let edge = measure(&mut renderer, distance, &format!("{distance:.0}m-s{width}"))?;
            println!(
                "  {distance:>5.0}  {:>5.1} | {width:>17.1} | {:>6.2}  {:>6.2} | {:>7}",
                1000.0 * millimetres_per_pixel(distance),
                edge.soft,
                edge.jag,
                edge.rows,
            );
        }
    }
    ship();

    let report = renderer.shutdown();
    anyhow::ensure!(
        report.clean(),
        "unclean render: {} validation message(s), {} leak(s)",
        report.validation_messages,
        report.leaked_allocations.len(),
    );
    println!();
    println!("  frames under target/gg-tools/shadow-edge-*.png");
    Ok(())
}

fn ship() {
    cvars::SHADOW_SIZE.set_int(SHIP_SIZE);
    cvars::SHADOW_DISTANCE.set_float(SHIP_DISTANCE);
    cvars::SHADOW_CASCADES.set_int(SHIP_CASCADES);
    cvars::SHADOW_SOFTNESS.set_float(SHIP_SOFTNESS);
    cvars::SUN_ANGLE.set_float(SHIP_SUN_ANGLE);
    cvars::SHADOW_TAPS.set_int(SHIP_TAPS);
    cvars::SHADOW_PENUMBRA.set_float(SHIP_PENUMBRA);
}

/// One leg: render, reduce, fit, and write the frame out beside the number so a
/// row that looks wrong can be looked at rather than argued about.
fn measure(renderer: &mut OffscreenRenderer, distance: f64, name: &str) -> anyhow::Result<Edge> {
    // Rebuilt per leg because the caster's placement depends on the distance:
    // aiming its shadow down the view axis is what makes `distance` mean how far
    // the *measured edge* is rather than how far some slab is.
    let world = scene(distance)?;
    let pixels = frame(renderer, &world, distance)?;
    let edge = Edge::of(&column_profile(&pixels))?;
    write_png(&pixels, name)?;
    Ok(edge)
}

/// A wide flat floor and one slab over it, the slab placed so its shadow lands
/// under the frame's centre at this leg's distance.
///
/// No second caster and no wall: every other edge in the frame is one the fit
/// below would have to be taught to ignore, and a metric with exceptions in it
/// is a metric nobody trusts. The slab is painted the floor's own colour for the
/// same reason — it is in frame at these framings, and a slab that shades like
/// the surface under it puts no step of its own into a column.
fn scene(distance: f64) -> anyhow::Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Light>()?;
    let floor = world.spawn();
    world.insert(
        floor,
        Renderable::boxed(
            sim::DVec3::new(0.0, -0.1, 0.0),
            sim::Vec3::new(120.0, 0.1, 120.0),
            0x009a_9488,
        ),
    )?;
    // Back along the light by however far it falls, so the shadow lands on the
    // aim point rather than wherever the sun's angle happens to put it.
    let drop = SLAB_HEIGHT / f64::from(-SUN[1]);
    let slab = world.spawn();
    world.insert(
        slab,
        Renderable::boxed(
            sim::DVec3::new(
                -f64::from(SUN[0]) * drop,
                SLAB_HEIGHT,
                aim(distance) - f64::from(SUN[2]) * drop,
            ),
            sim::Vec3::new(SLAB_HALF[0], SLAB_HALF[1], SLAB_HALF[2]),
            0x009a_9488,
        ),
    )?;
    let sun = world.spawn();
    world.insert(
        sun,
        Light::sun(sim::Vec3::new(SUN[0], SUN[1], SUN[2]), 0x00ff_f4e0, 3.4),
    )?;
    Ok(world)
}

/// Where the shadow is aimed on the floor at this leg — straight down the view
/// axis from an eye `EYE_UP * distance` metres up and `EYE_BACK * distance`
/// back, so `distance` really is how far the measured edge is from the camera.
fn aim(distance: f64) -> f64 {
    EYE_BACK * distance - EYE_UP * distance / sim::tan(f64::from(-EYE_PITCH))
}

/// The room from `distance` metres back, pitched to put the shadow across the
/// middle of the frame at every leg — a shadow that walked off the bottom as
/// the eye retreated would make the four rows measure four different things.
fn frame(
    renderer: &mut OffscreenRenderer,
    world: &World,
    distance: f64,
) -> anyhow::Result<Vec<u8>> {
    let eye = sim::DVec3::new(0.0, EYE_UP * distance, EYE_BACK * distance);
    let view = View {
        pitch: EYE_PITCH,
        ..View::default()
    };
    let mut extracted = Extracted::default();
    extracted.clear(eye, view.frustum(EXTENT));
    extracted.append_lights(world)?;
    extracted.cast_shadows(view.caster_reach(EXTENT));
    extracted.append::<Renderable>(world)?;
    let frame = renderer.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?;
    anyhow::ensure!(
        frame.order.iter().any(|name| name.starts_with("shadow")),
        "no shadow pass ran — there would be nothing to measure"
    );
    Ok(frame.pixels)
}

/// Metres of floor one screen pixel covers at the frame's centre — the scale
/// that turns **jag** from a pixel count into a statement about the world.
///
/// Reported rather than the cascade's texel size: the fitter is not public and a
/// second copy of the split scheme here would be a second thing to keep in sync.
/// What the column is for is the *ratio* between rows — walking from 6 m to 70 m
/// makes a pixel eleven times bigger, and a defect that grew with it is a defect
/// in the map's own grid rather than in the picture's.
fn millimetres_per_pixel(distance: f64) -> f64 {
    let half = 0.5 * f64::from(View::default().fov_y);
    2.0 * distance * sim::tan(half) / f64::from(EXTENT.1)
}

/// Per-column luminance down the frame's middle band, one profile a column.
///
/// The shadow runs along X in the world and therefore across the frame, so the
/// crossing being fitted is *down* each column and the fit is a line in x.
fn column_profile(pixels: &[u8]) -> Vec<Vec<i32>> {
    let (w, h) = (EXTENT.0 as usize, EXTENT.1 as usize);
    (0..w)
        .map(|x| {
            (0..h)
                .map(|y| {
                    let at = (y * w + x) * 4;
                    (2126 * i32::from(pixels[at])
                        + 7152 * i32::from(pixels[at + 1])
                        + 722 * i32::from(pixels[at + 2]))
                        / 10000
                })
                .collect()
        })
        .collect()
}

/// The measured edge: where each column crosses half-lit, how far those
/// crossings sit off a straight line, and how wide the crossing is.
struct Edge {
    jag: f64,
    soft: f64,
    rows: usize,
}

impl Edge {
    fn of(profile: &[Vec<i32>]) -> anyhow::Result<Self> {
        let mut crossings = Vec::new();
        for (x, column) in profile.iter().enumerate() {
            if let Some((at, width)) = crossing(column) {
                crossings.push((x as f64, at, width));
            }
        }
        anyhow::ensure!(
            crossings.len() > profile.len() / 4,
            "only {} of {} columns crossed a shadow edge — the framing has moved the shadow off \
             the frame and there is nothing to fit",
            crossings.len(),
            profile.len()
        );
        // Reject against the median, then fit once. A fit-reject-refit loop
        // converges on whatever subset happens to be straight, which is a way of
        // measuring zero jag on any picture at all.
        let mut middle: Vec<f64> = crossings.iter().map(|&(_, at, _)| at).collect();
        middle.sort_by(f64::total_cmp);
        let median = middle[middle.len() / 2];
        crossings.retain(|&(_, at, _)| (at - median).abs() < OUTLIER);
        anyhow::ensure!(
            !crossings.is_empty(),
            "every crossing was more than {OUTLIER} px from the median of the crossings"
        );
        let fitted = line(&crossings);
        let jag = (crossings
            .iter()
            .map(|&(x, at, _)| (at - fitted.at(x)).powi(2))
            .sum::<f64>()
            / crossings.len() as f64)
            .sqrt();
        let soft = crossings.iter().map(|&(_, _, w)| w).sum::<f64>() / crossings.len() as f64;
        Ok(Self {
            jag,
            soft,
            rows: crossings.len(),
        })
    }
}

/// Rows either side of the edge that the lit and shadowed levels are read from.
///
/// It has to clear the widest penumbra any leg produces, and that number moved
/// at M23: sized for the old two-pixel kernel, this window sat *inside* the ramp
/// at the wide end of the softness sweep, read a lit level that was already half
/// dark, and reported every width past about 1.4 px as 1.4 px. A metric that
/// saturates exactly where the knob starts working is worse than no metric.
const MARGIN: usize = 14;

/// Rows either side that the *locator* compares, which is a different question
/// from where the levels are read and wants a different number: wide enough that
/// a one-row artifact cannot outscore a real ramp, narrow enough not to average
/// the ramp away.
const STEP: usize = 6;

/// `(sub-pixel row of the half-lit crossing, 20%→80% width)` down one column.
///
/// Found as the largest *sustained* fall down the column — the mean of the
/// [`STEP`] rows below a row against the mean of the [`STEP`] above it — rather
/// than as a crossing of a global level or as the steepest single row. All three
/// halves of that matter.
///
/// Sustained, because the steepest single row is the *wrong* locator for
/// precisely the thing being measured: a soft edge spreads its contrast over
/// several rows, so the wider the penumbra the smaller its steepest row gets,
/// and a one-row artifact elsewhere in the column wins. A locator that goes
/// blind as the knob works is not a locator.
///
/// A fall and not a rise, and A threshold taken off the
/// column's own range finds whatever the frame's widest contrast happens to be.
/// And the shadow's two boundaries are not equally worth measuring: its near one
/// sits directly under the caster, near enough under a steeply pitched eye that
/// what the column actually crosses there is the caster's *silhouette* — a
/// geometry edge, hard at every filter width, and the reason this instrument
/// once reported 0.60 px for every row of a sweep. The far boundary is a shadow
/// boundary over open floor with nothing else within a margin of it, and looking
/// down at the floor it is the largest step *down* there is.
///
/// Levels are read *locally*, a margin either side, so the measurement is of
/// this edge and not of the frame's contrast. Linear interpolation between the
/// two rows straddling each level makes the answer sub-pixel — a crossing
/// quantized to whole rows could not tell a straight edge from a staircase,
/// which is the entire question.
fn crossing(column: &[i32]) -> Option<(f64, f64)> {
    // The eye looks down steeply enough that every row is floor, so the whole
    // column is fair game bar the margins the levels are read from.
    let from = MARGIN.max(STEP) + 1;
    let to = column.len().checked_sub(MARGIN.max(STEP) + 2)?;
    let mean = |range: std::ops::Range<usize>| -> f64 {
        let n = range.len() as f64;
        column[range].iter().map(|&v| f64::from(v)).sum::<f64>() / n
    };
    let step = |y: usize| mean(y + 1..y + 1 + STEP) - mean(y - STEP..y);
    let edge = (from..to).min_by(|&a, &b| step(a).total_cmp(&step(b)))?;
    let lit = mean(edge - MARGIN..edge - 1);
    let dark = mean(edge + 3..edge + MARGIN);
    // No shadow in this column, or all shadow: nothing to find, and a threshold
    // taken off its own noise would find something anyway.
    if lit - dark < 24.0 {
        return None;
    }
    let at = |f: f64| -> Option<f64> {
        let want = dark + f * (lit - dark);
        (edge - MARGIN..edge + MARGIN).find_map(|y| {
            let (a, b) = (f64::from(column[y]), f64::from(column[y + 1]));
            match (a >= want) != (b >= want) && (a - b).abs() > 1e-6 {
                true => Some(y as f64 + (a - want) / (a - b)),
                false => None,
            }
        })
    };
    let half = at(0.5)?;
    let (hi, lo) = (at(0.8)?, at(0.2)?);
    Some((half, (lo - hi).abs()))
}

/// A least-squares line through `(x, y)`, ignoring the third field.
struct Line {
    slope: f64,
    intercept: f64,
}

impl Line {
    fn at(&self, x: f64) -> f64 {
        self.slope * x + self.intercept
    }
}

fn line(points: &[(f64, f64, f64)]) -> Line {
    let n = points.len() as f64;
    let mean_x = points.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = points.iter().map(|p| p.1).sum::<f64>() / n;
    let cov: f64 = points.iter().map(|p| (p.0 - mean_x) * (p.1 - mean_y)).sum();
    let var: f64 = points.iter().map(|p| (p.0 - mean_x).powi(2)).sum();
    let slope = match var > 1e-9 {
        true => cov / var,
        false => 0.0,
    };
    Line {
        slope,
        intercept: mean_y - slope * mean_x,
    }
}

fn write_png(pixels: &[u8], name: &str) -> anyhow::Result<()> {
    let path = crate::output_dir()?.join(format!("shadow-edge-{name}.png"));
    let file = std::fs::File::create(&path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), EXTENT.0, EXTENT.1);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(pixels)?;
    Ok(())
}
