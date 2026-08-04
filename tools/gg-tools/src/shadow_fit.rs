//! `gg-tools shadow-fit` — what splitting the shadow range buys the picture.
//!
//! One cascade has to cover the whole of `r.shadow_distance` with one map, so its
//! texel is the range over `r.shadow_size` however big the thing being shadowed
//! is: 80 m across 2048 texels is 39 mm, a whole shadow map spent on an 80 m
//! circle whether the scene is 80 m across or one metre. That is what put a
//! visible staircase under a one-metre cube (§6 M15.3).
//!
//! This renders the *same* scene at each `r.shadow_cascades` count, writes each
//! leg out to look at, and reports two numbers against a deliberately tight
//! reference:
//!
//! - **boundary** — disagreement in the band around a shadow edge. A **coarse**
//!   indicator only, and deliberately labelled as one: the 3x3 kernel is three
//!   texels wide, so the penumbra itself scales with the radius and dominates
//!   this band. It says the boundary landed differently, not that it landed
//!   jagged. The images are what show the staircase.
//! - **seam** — disagreement on surfaces the sun *faces*, measured strictly
//!   away from any boundary. This one is clean: a lit face has no penumbra to
//!   confound it, so what shows here is self-shadowing, which is what a steep
//!   receiver meeting a shallow one inside one texel produces.
//!
//! Both fall with the texel when the cause is density, but the *rate* is not
//! the finding — a metric with a floor cannot prove a fix, only point at one.
//! What this instrument is for is deciding **whether the cascade's fit is where
//! the problem lives**, and it answers that by holding everything else fixed.

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

const EXTENT: (u32, u32) = (640, 480);

/// Cascade counts swept. 1 is the pre-cascade engine — one slab over the whole
/// range — and 4 is the shipping default.
const COUNTS: &[i64] = &[1, 2, 3, 4];

/// The range and map size every test leg shares, so the *only* thing moving
/// across the table is how many maps the range is divided between.
const TEST_DISTANCE: f64 = 80.0;
const TEST_SIZE: i64 = 2048;

/// The reference leg: one cascade over a range short enough to be all subject,
/// at the densest map `Sun::of` will clamp to.
///
/// Short but *not* shorter than the subject is 4 m away, and that is the trap
/// this constant exists to name: shrinking it further does not make a better
/// reference, it makes an empty one, because the cascade stops containing the
/// scene before the texels get interesting. [`has_a_shadow`] is the assertion.
const REFERENCE_DISTANCE: f64 = 6.0;
const REFERENCE_SIZE: i64 = 4096;

/// Luminance levels of disagreement that count as one, matching `shadow-bias`.
const DISAGREEMENT: i32 = 8;

/// Below this in both legs the pixel is background rather than surface.
const BACKGROUND: i32 = 4;

/// Reference 3x3 luminance range that marks a shadow or geometric edge.
const EDGE_GRADIENT: i32 = 20;

/// Pixels either side of an edge. `boundary` is measured *inside* this band and
/// `seam` strictly outside it — the whole point is to separate the two failures.
const EDGE_BAND: usize = 4;

pub fn run(args: &[String]) -> anyhow::Result<()> {
    if let Some(arg) = args.first() {
        anyhow::bail!("unknown flag {arg:?} — shadow-fit takes none");
    }
    let mut renderer = OffscreenRenderer::new(EXTENT)?;
    println!(
        "gg-tools shadow-fit: {}x{} on {}",
        EXTENT.0,
        EXTENT.1,
        renderer.device().chosen
    );
    println!(
        "  the template's scene at r.shadow_distance {TEST_DISTANCE} / size {TEST_SIZE}, \
         against one cascade over {REFERENCE_DISTANCE} m at size {REFERENCE_SIZE}"
    );

    let world = scene()?;
    cvars::SHADOW_DISTANCE.set_float(REFERENCE_DISTANCE);
    cvars::SHADOW_CASCADES.set_int(1);
    cvars::SHADOW_SIZE.set_int(REFERENCE_SIZE);
    let reference = render(&mut renderer, &world)?;
    write_png(&reference.pixels, "reference")?;
    let reference = reference.luminance;
    let (edge, surface) = bands(&reference);
    anyhow::ensure!(
        has_a_shadow(&reference, &surface),
        "the reference frame has no shadow in it — at {REFERENCE_DISTANCE} m the cascade no \
         longer contains the subject, and every number below would be measuring the test legs \
         against an unshadowed frame"
    );
    cvars::SHADOW_SIZE.set_int(TEST_SIZE);

    println!();
    println!("  cascades | boundary%    seam%");
    println!("  ---------+------------------");
    let mut rows = Vec::new();
    for &count in COUNTS {
        cvars::SHADOW_CASCADES.set_int(count);
        let frame = render(&mut renderer, &world)?;
        write_png(&frame.pixels, &format!("cascades-{count}"))?;
        let test = frame.luminance;
        let boundary = disagree(&test, &reference, &edge);
        let seam = disagree(&test, &reference, &surface);
        println!(
            "  {count:>8} | {:>7.3}  {:>7.3}",
            boundary * 100.0,
            seam * 100.0
        );
        rows.push((count, boundary, seam));
    }

    let report = renderer.shutdown();
    anyhow::ensure!(
        report.clean(),
        "unclean render: {} validation message(s), {} leak(s)",
        report.validation_messages,
        report.leaked_allocations.len(),
    );

    if let (Some(one), Some(many)) = (rows.first(), rows.last()) {
        println!();
        println!(
            "  {} cascades against 1: boundary {:.2}x smaller, seam {:.2}x smaller",
            many.0,
            ratio_of(one.1, many.1),
            ratio_of(one.2, many.2),
        );
    }
    Ok(())
}

/// Whether a frame contains a cast shadow at all, judged off surface pixels
/// clear of any edge.
///
/// The reference's own sanity check, and it earns its lines: a slab centred on
/// the camera silently stops containing the subject as the radius drops, and a
/// reference with no shadow in it makes every disagreement below read as one.
/// A fifth of the surface a fifth darker than its median is a shadow; noise and
/// shading gradients are neither that dark nor that common.
fn has_a_shadow(reference: &[i32], surface: &[bool]) -> bool {
    let mut lit: Vec<i32> = reference
        .iter()
        .zip(surface)
        .filter(|&(_, &keep)| keep)
        .map(|(&l, _)| l)
        .collect();
    if lit.is_empty() {
        return false;
    }
    lit.sort_unstable();
    let median = lit[lit.len() / 2];
    let dark = lit.iter().filter(|&&l| l * 5 < median * 4).count();
    dark * 100 > lit.len()
}

fn ratio_of(coarse: f64, fine: f64) -> f64 {
    if fine > 0.0 {
        coarse / fine
    } else {
        f64::INFINITY
    }
}

/// The template's scene exactly: a 1.2 m cube resting on a wide floor, lit by
/// the template's own sun. The scene the complaint came from, so the numbers
/// are about it and not about a proxy.
fn scene() -> anyhow::Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Light>()?;
    let mut put = |r: Renderable| -> anyhow::Result<()> {
        let e = world.spawn();
        world.insert(e, r)?;
        Ok(())
    };
    let half = 0.6;
    let mut cube = Renderable::boxed(sim::DVec3::ZERO, sim::Vec3::splat(half), 0x0060_a0ff);
    // Turned, because an axis-aligned cube puts its shadow's edges along the
    // map's own grid and hides the staircase this is here to count.
    cube.rotation = sim::DQuat::from_axis_angle(sim::DVec3::Y, 0.65);
    put(cube)?;
    put(Renderable::boxed(
        sim::DVec3::new(0.0, f64::from(-half - 0.1), 0.0),
        sim::Vec3::new(6.0, 0.1, 6.0),
        0x009a_9488,
    ))?;
    let sun = world.spawn();
    world.insert(
        sun,
        Light::sun(sim::Vec3::new(-0.4, -1.0, -0.35), 0x00ff_f4e0, 3.5),
    )?;
    Ok(world)
}

/// One render's pixels and their luminance — the image to look at and the
/// numbers to count, from one pass rather than two.
struct Frame {
    pixels: Vec<u8>,
    luminance: Vec<i32>,
}

fn write_png(pixels: &[u8], name: &str) -> anyhow::Result<()> {
    let path = crate::output_dir()?.join(format!("shadow-fit-{name}.png"));
    let file = std::fs::File::create(&path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), EXTENT.0, EXTENT.1);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(pixels)?;
    Ok(())
}

/// One render at the template's own camera.
fn render(renderer: &mut OffscreenRenderer, world: &World) -> anyhow::Result<Frame> {
    let view = View {
        pitch: -0.2,
        ..View::default()
    };
    // The template's eye is 4 m back and 1 m up; extract is camera-relative, so
    // that is the origin the scene is narrowed through.
    let eye = sim::DVec3::new(0.0, 1.0, 4.0);
    let mut extracted = Extracted::default();
    extracted.transforms::<Renderable>(world, eye, view.frustum(EXTENT))?;
    extracted.append_lights(world)?;
    let frame = renderer.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?;
    anyhow::ensure!(
        frame.order.iter().any(|name| name.starts_with("shadow")),
        "no shadow pass ran — there would be nothing to measure"
    );
    let luminance = frame
        .pixels
        .chunks_exact(4)
        .map(|p| (2126 * i32::from(p[0]) + 7152 * i32::from(p[1]) + 722 * i32::from(p[2])) / 10000)
        .collect();
    Ok(Frame {
        pixels: frame.pixels,
        luminance,
    })
}

/// `(near an edge, well clear of one)` — both restricted to surface pixels.
///
/// Taken from the reference, for the reason `shadow-bias` takes its mask there:
/// a staircase is itself high-gradient, so a band defined from the test leg
/// would move with the artifact it is trying to bracket.
fn bands(reference: &[i32]) -> (Vec<bool>, Vec<bool>) {
    let (w, h) = (EXTENT.0 as usize, EXTENT.1 as usize);
    let at = |x: usize, y: usize| reference[y * w + x];
    let mut edge = vec![false; reference.len()];
    for y in 0..h {
        for x in 0..w {
            let (mut lo, mut hi) = (i32::MAX, i32::MIN);
            for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    lo = lo.min(at(nx, ny));
                    hi = hi.max(at(nx, ny));
                }
            }
            edge[y * w + x] = hi - lo > EDGE_GRADIENT;
        }
    }
    let mut wide = vec![false; edge.len()];
    for y in 0..h {
        for x in 0..w {
            let lo = x.saturating_sub(EDGE_BAND);
            let hi = (x + EDGE_BAND).min(w - 1);
            wide[y * w + x] = edge[y * w + lo..=y * w + hi].iter().any(|e| *e);
        }
    }
    let (mut near, mut clear) = (vec![false; edge.len()], vec![false; edge.len()]);
    for y in 0..h {
        for x in 0..w {
            let lo = y.saturating_sub(EDGE_BAND);
            let hi = (y + EDGE_BAND).min(h - 1);
            let dilated = (lo..=hi).any(|ny| wide[ny * w + x]);
            let surface = at(x, y) >= BACKGROUND;
            near[y * w + x] = dilated && surface;
            clear[y * w + x] = !dilated && surface;
        }
    }
    (near, clear)
}

/// Fraction of the masked pixels the test leg gets visibly wrong, either way.
fn disagree(test: &[i32], reference: &[i32], mask: &[bool]) -> f64 {
    let (mut counted, mut wrong) = (0u64, 0u64);
    for ((&t, &r), &keep) in test.iter().zip(reference).zip(mask) {
        if !keep {
            continue;
        }
        counted += 1;
        if (t - r).abs() > DISAGREEMENT {
            wrong += 1;
        }
    }
    if counted == 0 {
        return 0.0;
    }
    wrong as f64 / counted as f64
}
