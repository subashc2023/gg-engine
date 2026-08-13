//! `gg-tools bounce` — how much of the light in a room got there by bouncing,
//! and how close the environment a level author *drew* comes to it (§6 M36).
//!
//! Two questions, and the first has to be answered before the second is worth
//! asking — `ao`'s shape, one term along.
//!
//! **Is there anything here?** Indirect light is the term the engine has never
//! derived: `r.ambient` is a constant and §6 M28's environments are boxes
//! somebody positioned by hand. Whether that is a visible wrong or a rounding
//! error is a property of the *content*, so it is measured against demo 12's
//! room — shipped geometry, a table of axis-aligned boxes with colours, which
//! [`gg_render::bounce`] represents exactly rather than approximately, under
//! the demo's own sun, lamps and two skies.
//!
//! **And how close is the authored version?** The shipped path composites those
//! two `Sky` volumes front to back and evaluates nine coefficients; the
//! reference casts paths. The gap is what a hand-placed box cannot know —
//! which is everything about where the geometry is — and the point of printing
//! it is that a room lit by a constant looks *fine* until something in it
//! should have been coloured by the wall beside it.
//!
//! **And how close is the field that replaced it?** [`field`] grades the
//! *shipped* §6 M36 path against the same paths, split into the two failures an
//! absolute error cannot tell apart — light **invented**, which is a probe
//! leaking through a wall, and light **missed**, which is a probe the visibility
//! term rejected too hard. `r.gi_spacing` and `r.gi_moments` are read off the
//! plateau between them.

use anyhow::Result;
use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable, Sky};
use gg_extract::{Extracted, SkyLook};
use gg_math::{render, sim};
use gg_render::{OffscreenRenderer, View, ao, bounce, cvars, sky, srgb_to_linear};

use demo_12_shooter as demo;

/// The frame the tables are measured over. Small on purpose: every pixel costs a
/// primary cast plus [`SAMPLES`] paths of up to [`BOUNCES`] segments each, and
/// how a room's light divides between direct and bounced converges long before
/// its silhouettes do.
const EXTENT: (u32, u32) = (240, 135);

/// Every `STRIDE`th pixel in each axis is a sample point. The framing is `ao`'s
/// so the two instruments describe the same room from the same chair, and the
/// decimation is what makes a path-traced table finish in a minute: how a room's
/// light divides between direct and bounced is a distribution over surfaces, and
/// a thousand of them settle it long before thirty thousand would.
const STRIDE: u32 = 6;

/// Vertical field of view, radians — `r.fov`'s default.
const FOV: f32 = 1.0;

/// Where the tables stand: demo 12's spawn, at eye height, looking across the
/// floor at the stairs and the shelter. `ao`'s camera exactly, so the two
/// instruments describe the same pixels.
const EYE: sim::DVec3 = sim::DVec3::new(0.0, 1.62, 8.0);
const YAW: f32 = 0.0;
const PITCH: f32 = -0.22;

/// Paths per sample point. The reference's own convergence is what licenses it
/// and the `--slow` leg is what checks the licence.
const SAMPLES: u32 = 256;

/// Bounce vertices per path. Four is past where the series matters at these
/// albedos — the fourth bounce of a 0.7 wall carries 34 % of the first's and
/// the room's mean albedo is below that — and the table prints the truncation
/// so the claim is not taken on trust.
const BOUNCES: u32 = 4;

pub fn run(args: &[String]) -> Result<()> {
    let slow = args.iter().any(|a| a == "--slow");
    let scene = room();
    let suns = [sun()];
    let lamps = lamps();
    let world = world()?;
    let cast = bounce::Scene {
        solids: &scene,
        suns: &suns,
        lamps: &lamps,
    };
    let points = surfaces(&scene);
    println!(
        "demo 12's room, {} solids, {} sun, {} lamps, {}x{} from the spawn — {} surface points\n",
        scene.len(),
        suns.len(),
        lamps.len(),
        EXTENT.0,
        EXTENT.1,
        points.len()
    );
    convergence(&cast, &points, slow)?;
    split(&cast, &points)?;
    authored(&cast, &points, &world)?;
    field(&scene, &points)?;
    Ok(())
}

/// A point the camera can see, with the normal of the surface it sits on.
struct Sample {
    point: sim::Vec3,
    normal: sim::Vec3,
    /// Which pixel found it. [`field`]'s tables read the rendered frame at this
    /// index, so the picture and the paths grade the same surface rather than
    /// two framings of one room.
    pixel: usize,
}

/// Where the frame's primary rays land. Camera-relative throughout, which is the
/// space [`ao::Occluder`] is already in.
fn surfaces(scene: &[ao::Occluder]) -> Vec<Sample> {
    let mut out = Vec::new();
    for y in (0..EXTENT.1).step_by(STRIDE as usize) {
        for x in (0..EXTENT.0).step_by(STRIDE as usize) {
            let direction = ray(x, y);
            let Some(hit) = ao::trace(sim::Vec3::ZERO, direction, scene, 1e-3, 1e4) else {
                continue;
            };
            out.push(Sample {
                point: direction * hit.distance,
                normal: hit.normal,
                pixel: (y * EXTENT.0 + x) as usize,
            });
        }
    }
    out
}

/// The estimator's own convergence, printed rather than assumed — the licence
/// for [`SAMPLES`], and the first thing a reader of any number below should be
/// able to check.
fn convergence(scene: &bounce::Scene, points: &[Sample], slow: bool) -> Result<()> {
    let counts: &[u32] = if slow {
        &[64, 128, 256, 512, 1024]
    } else {
        &[64, 128, 256, 512]
    };
    println!("  paths |   mean E | vs the finest");
    let finest = mean_irradiance(scene, points, *counts.last().unwrap_or(&SAMPLES), BOUNCES);
    for &samples in counts {
        let mean = mean_irradiance(scene, points, samples, BOUNCES);
        println!(
            "  {samples:>5} | {:>8.4} | {:>+7.2} %",
            luminance(mean),
            (luminance(mean) / luminance(finest) - 1.0) * 100.0
        );
    }
    println!();
    Ok(())
}

/// Where the diffuse light in this room actually comes from.
///
/// Three quantities that sum to the whole, each of which the engine treats
/// differently: what the lights deliver (exact, through a shadow map), what the
/// sky delivers through the directions geometry leaves open (approximated —
/// nine coefficients, occluded by §6 M35's screen-space term), and what
/// **bounced** off a surface on the way (not represented at all).
///
/// The direct column is not part of the sum and is printed beside it for scale:
/// it is what the shadow map already delivers exactly, and the question here is
/// how the *rest* divides. The bounced column is the milestone's justification
/// and it is computed by subtracting the sky leg from the whole rather than by a
/// second estimator, so the two cannot fail to add up and a bug that moved light
/// between them would have to move it into the leg with a closed form.
fn split(scene: &bounce::Scene, points: &[Sample]) -> Result<()> {
    let n = points.len().max(1) as f64;
    let mut direct = 0.0f64;
    let mut sky_only = 0.0f64;
    // Albedo forced to zero everywhere: a path meets a surface, that surface
    // reflects nothing, and what is left is exactly the sky through whatever
    // directions the geometry left open. Built once — it does not depend on the
    // bounce count, and neither does the direct term.
    let dark = blacken(scene.solids);
    let unlit = bounce::Scene {
        solids: &dark,
        suns: scene.suns,
        lamps: scene.lamps,
    };
    for s in points {
        direct += f64::from(luminance(bounce::direct(s.point, s.normal, scene)));
        sky_only += f64::from(luminance(bounce::irradiance(
            s.point,
            s.normal,
            &unlit,
            environment,
            bounce::Params {
                samples: SAMPLES,
                bounces: 1,
                ..bounce::Params::default()
            },
        )));
    }
    println!("  bounces |   direct |      sky |  bounced | bounced share of the indirect");
    for bounces in [1u32, 2, 3, 4] {
        let mut total = 0.0f64;
        for s in points {
            total += f64::from(luminance(bounce::irradiance(
                s.point,
                s.normal,
                scene,
                environment,
                bounce::Params {
                    samples: SAMPLES,
                    bounces,
                    ..bounce::Params::default()
                },
            )));
        }
        let bounced = (total - sky_only).max(0.0);
        println!(
            "  {bounces:>7} | {:>8.4} | {:>8.4} | {:>8.4} | {:>29.1} %",
            direct / n,
            sky_only / n,
            bounced / n,
            bounced / (bounced + sky_only).max(1e-9) * 100.0
        );
    }
    println!();
    Ok(())
}

/// What the shipped environment delivers at each point, against what the paths
/// say arrives there.
///
/// The engine's ambient term is `ambient_light`'s composite restated: the
/// extract's own sky array, already sorted innermost first, walked front to back
/// against a budget of weight, accumulated as coefficients because projection is
/// linear (§6 M28), and evaluated once. Compared against the reference's
/// non-direct half — sky plus bounce — because that is the quantity it stands
/// for; the direct term is the shadow map's and is not in question here.
///
/// Two failures in opposite directions, which is the shape a knob's plateau is
/// read off and the shape a *replacement* has to beat: light the authored
/// version **invents** where the paths say a surface is shadowed from the sky,
/// and light it **misses** where a bounce delivered something it has no term
/// for.
fn authored(scene: &bounce::Scene, points: &[Sample], world: &World) -> Result<()> {
    let view = View {
        pitch: PITCH,
        yaw: YAW,
        ..View::default()
    };
    let mut extracted = Extracted::default();
    extracted.clear(EYE, view.frustum(EXTENT));
    extracted.append_lights(world)?;
    let projected: Vec<_> = extracted
        .skies
        .iter()
        .map(|s| (s, sky::project(&s.look)))
        .collect();
    println!(
        "  {} environment(s) composited, innermost first\n",
        projected.len()
    );

    let (mut invented, mut missed, mut truth, mut shipped) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut worst, mut worst_at) = (0.0f64, 0usize);
    let mut graded: Vec<(f32, f32)> = Vec::with_capacity(points.len());
    for (i, s) in points.iter().enumerate() {
        let reference = luminance(bounce::irradiance(
            s.point,
            s.normal,
            scene,
            environment,
            bounce::Params {
                samples: SAMPLES,
                bounces: BOUNCES,
                ..bounce::Params::default()
            },
        ));
        let authored = luminance_render(composite(&projected, s));
        let delta = f64::from(authored - reference);
        if delta > 0.0 {
            invented += delta;
        } else {
            missed += -delta;
        }
        truth += f64::from(reference);
        shipped += f64::from(authored);
        graded.push((reference, authored));
        if delta.abs() > worst {
            (worst, worst_at) = (delta.abs(), i);
        }
    }
    let n = points.len().max(1) as f64;
    println!("  mean indirect, paths      | {:>8.4}", truth / n);
    println!("  mean indirect, authored   | {:>8.4}", shipped / n);
    println!(
        "  invented (authored > true) | {:>8.4}  ({:>5.1} % of the truth)",
        invented / n,
        invented / truth.max(1e-9) * 100.0
    );
    println!(
        "  missed   (authored < true) | {:>8.4}  ({:>5.1} % of the truth)",
        missed / n,
        missed / truth.max(1e-9) * 100.0
    );
    let Some(at) = points.get(worst_at) else {
        return Ok(());
    };
    println!(
        "  worst point                | {:>8.4}  at ({:.2}, {:.2}, {:.2}) facing ({:.2}, {:.2}, {:.2})",
        worst,
        at.point.x + EYE.x as f32,
        at.point.y + EYE.y as f32,
        at.point.z + EYE.z as f32,
        at.normal.x,
        at.normal.y,
        at.normal.z
    );
    // Where the disagreement lives, against how much indirect light the paths
    // say a point gets. A mean that matches while the buckets do not is the
    // signature of a hand-placed volume: it was tuned until the room read right
    // overall, which is a statement about one number and about nothing else.
    println!(
        "
  true indirect |  points |    paths | authored |     error"
    );
    for (lo, hi) in [(0.0f32, 0.4), (0.4, 0.8), (0.8, 1.2), (1.2, f32::INFINITY)] {
        let (mut count, mut t, mut a) = (0usize, 0.0f64, 0.0f64);
        for (reference, authored) in &graded {
            if *reference >= lo && *reference < hi {
                count += 1;
                t += f64::from(*reference);
                a += f64::from(*authored);
            }
        }
        if count == 0 {
            continue;
        }
        let c = count as f64;
        let label = if hi.is_finite() {
            format!("{lo:.1}..{hi:.1}")
        } else {
            format!("{lo:.1}+   ")
        };
        println!(
            "  {label:>13} | {count:>7} | {:>8.4} | {:>8.4} | {:>+9.4}",
            t / c,
            a / c,
            (a - t) / c
        );
    }
    println!();
    Ok(())
}

/// The uniform sky [`field`] grades under, as a radiance.
///
/// `ao`'s constant and for its reason: the tonemapper's knee is at 0.76 and
/// nothing in that leg reaches a third of it, so a ratio of two decoded code
/// values needs no curve inverted. Here it is also the *only* light there is.
const AMBIENT: f64 = 0.25;

/// Frames the field is given to converge before it is graded.
///
/// At `r.gi_rate 0` a frame gathers one batch — `probe::MAX_BATCH`, 64 — against
/// a grid of at most 512, so eight frames is the floor and this is the margin a
/// sweep down to a fine spacing needs. `gg-golden`'s `FIELD_FRAMES` is the same
/// argument for a harness that renders one scene.
const FIELD_FRAMES: usize = 32;

/// The shipped field against the paths, in the two failures that fail in
/// opposite directions (§6 M36).
///
/// **A ratio of the same pixel**, which is what makes this exact rather than
/// calibrated — `ao`'s argument one term along, and the reason nothing about the
/// shader's sampling is restated here. The leg is lit by nothing but
/// `r.ambient`: with no `Sky` declared and no `Light` in the world the pre-M36
/// ambient term is that constant at every point, and a probe whose face texel
/// escaped records exactly it back (`probe.slang`'s miss). So one render with
/// `r.gi 0` divided into one with `r.gi 1` is the field's own answer **in units
/// of an unoccluded sky** — 1.0 where the field agrees a point is open, below
/// where geometry took the sky away, above where a coloured wall gave some back.
///
/// The reference is that same quantity with nothing approximated: paths under a
/// uniform environment of the same radiance, over what an unoccluded point would
/// receive.
///
/// Reading the two columns: light **invented** is the field brighter than the
/// paths, which is a probe leaking through a wall and reads as a room that will
/// not go dark; light **missed** is the field darker, which is a probe the
/// visibility term rejected too hard and reads as a crease that never fills in.
/// An absolute error averages them into a number that improves when they trade.
fn field(solids: &[ao::Occluder], points: &[Sample]) -> Result<()> {
    cvars::AMBIENT.set_float(AMBIENT);
    // The dither is a deliberate plus-or-minus one code value and this reads
    // single pixels; a ratio would carry it twice.
    cvars::DITHER.set_float(0.0);
    // Off for headroom rather than for correctness: a screen-space term
    // multiplies the ambient half of *both* renders and cancels in the ratio,
    // but it also darkens the pixels the ratio is quantised out of.
    cvars::AO.set_bool(false);
    cvars::GI_RATE.set_int(0);
    let world = boxes()?;
    let cast = bounce::Scene {
        solids,
        suns: &[],
        lamps: &[],
    };
    let reference = openness(&cast, points);
    let mut renderer = OffscreenRenderer::new(EXTENT)?;

    println!(
        "the field against the paths — the same room under a uniform sky of {AMBIENT}, no lights\n"
    );
    let shipped = rendered(&mut renderer, &world, points)?;
    let inside = covered(&renderer, points);
    let (_, probes) = renderer.field_pending();
    println!(
        "  shipped defaults: {probes} probes at {:.2} m, distance tile {} texels, {} of {} points inside the grid",
        cvars::GI_SPACING.float(),
        cvars::GI_MOMENTS.int(),
        inside.iter().filter(|c| **c).count(),
        points.len()
    );
    report(&shipped, &reference, &inside, points);
    buckets(&shipped, &reference, &inside);

    spacings(&mut renderer, &world, points, &reference)?;
    tiles(&mut renderer, &world, points, &reference)?;

    cvars::DITHER.set_float(1.0);
    cvars::AO.set_bool(true);
    cvars::GI_RATE.set_int(16);
    let report = renderer.shutdown();
    anyhow::ensure!(report.clean(), "unclean render: {report:?}");
    Ok(())
}

/// The paths' answer in the units the ratio is already in: indirect irradiance
/// over what a point open to the whole sky would receive, `π` times its
/// radiance.
fn openness(scene: &bounce::Scene, points: &[Sample]) -> Vec<f32> {
    let radiance = AMBIENT as f32;
    let open = core::f32::consts::PI * radiance;
    points
        .iter()
        .map(|s| {
            let e = bounce::irradiance(
                s.point,
                s.normal,
                scene,
                |_| sim::Vec3::new(radiance, radiance, radiance),
                bounce::Params {
                    samples: SAMPLES,
                    bounces: BOUNCES,
                    ..bounce::Params::default()
                },
            );
            luminance(e) / open
        })
        .collect()
}

/// The field's answer at each sample point, as the ratio of two renders.
///
/// The `r.gi 1` leg runs first and is given [`FIELD_FRAMES`] to converge; the
/// `r.gi 0` leg is one frame after it, which costs nothing the field cares about
/// — a frame with the term switched off still gathers, and gathering is the only
/// thing that writes.
fn rendered(
    renderer: &mut OffscreenRenderer,
    world: &World,
    points: &[Sample],
) -> Result<Vec<f32>> {
    cvars::GI.set_bool(true);
    let mut lit = frame(renderer, world)?;
    for _ in 0..FIELD_FRAMES {
        if renderer.field_pending().0 == 0 {
            break;
        }
        lit = frame(renderer, world)?;
    }
    let (pending, probes) = renderer.field_pending();
    anyhow::ensure!(
        pending == 0,
        "the field did not converge in {FIELD_FRAMES} frames: {pending} of {probes} ungathered"
    );
    cvars::GI.set_bool(false);
    let flat = frame(renderer, world)?;
    cvars::GI.set_bool(true);
    Ok(points
        .iter()
        .map(|s| match flat.get(s.pixel) {
            Some(d) if *d > 1e-4 => lit.get(s.pixel).copied().unwrap_or(*d) / d,
            _ => 1.0,
        })
        .collect())
}

/// One render of the graded room, as decoded linear luminance per pixel.
fn frame(renderer: &mut OffscreenRenderer, world: &World) -> Result<Vec<f32>> {
    let view = View {
        pitch: PITCH,
        yaw: YAW,
        ..View::default()
    };
    let mut extracted = Extracted::default();
    extracted.clear(EYE, view.frustum(EXTENT));
    extracted.append::<Renderable>(world)?;
    extracted.append_lights(world)?;
    let pixels = renderer
        .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?
        .pixels;
    Ok(pixels
        .chunks_exact(4)
        .map(|p| 0.2126 * decode(p[0]) + 0.7152 * decode(p[1]) + 0.0722 * decode(p[2]))
        .collect())
}

/// sRGB decode, IEC 61966-2-1 — `post.slang`'s encode run backwards. `ao`'s, and
/// restated for that module's reason: the two never meet at runtime.
fn decode(code: u8) -> f32 {
    let e = f32::from(code) / 255.0;
    if e <= 0.040_45 {
        e / 12.92
    } else {
        sim::powf((e + 0.055) / 1.055, 2.4)
    }
}

/// Demo 12's room as the renderer draws it: the same boxes with the same
/// declared colours, fully rough so nothing here is a reflection, and **no sky
/// and no lights** — see [`field`] for why that is the whole measurement.
fn boxes() -> Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    for &(center, half_extent, color) in demo::ROOM {
        let entity = world.spawn();
        world.insert(
            entity,
            Renderable::boxed(center, half_extent, color).surfaced(0.0, 0.0),
        )?;
    }
    Ok(world)
}

/// Which sample points the grid actually reaches.
///
/// **The distinction the table lives on.** The grid is clamped at eight probes
/// an axis, so a fine `r.gi_spacing` covers a *corner* of a large room and every
/// point outside it is lit by the flat ambient the field replaces — which reads
/// as fully open, and against a truth of 0.66 scores as the largest leak there
/// is. A sweep that did not separate the two would report a coarse grid as the
/// better one for a reason that has nothing to do with the field.
fn covered(renderer: &OffscreenRenderer, points: &[Sample]) -> Vec<bool> {
    let Some((origin, spacing, counts)) = renderer.field_grid() else {
        return vec![false; points.len()];
    };
    points
        .iter()
        .map(|s| {
            let at = [
                f64::from(s.point.x) + EYE.x,
                f64::from(s.point.y) + EYE.y,
                f64::from(s.point.z) + EYE.z,
            ];
            let lo = [origin.x, origin.y, origin.z];
            (0..3).all(|i| {
                let reach = f64::from(counts[i] - 1) * f64::from(spacing);
                at[i] >= lo[i] && at[i] <= lo[i] + reach
            })
        })
        .collect()
}

/// The headline four numbers, and where the worst of them is.
fn report(shipped: &[f32], reference: &[f32], inside: &[bool], points: &[Sample]) {
    let (invented, missed, truth, field) = totals(shipped, reference, inside);
    let n = inside.iter().filter(|c| **c).count().max(1) as f64;
    println!("  mean openness, paths      | {:>8.4}", truth / n);
    println!("  mean openness, field      | {:>8.4}", field / n);
    println!(
        "  invented (field > true)   | {:>8.4}  ({:>5.1} % of the truth)",
        invented / n,
        invented / truth.max(1e-9) * 100.0
    );
    println!(
        "  missed   (field < true)   | {:>8.4}  ({:>5.1} % of the truth)",
        missed / n,
        missed / truth.max(1e-9) * 100.0
    );
    let mut worst = (0.0f32, 0usize);
    for (i, ((s, t), on)) in shipped.iter().zip(reference).zip(inside).enumerate() {
        if *on && (s - t).abs() > worst.0.abs() {
            worst = (s - t, i);
        }
    }
    if let Some(at) = points.get(worst.1) {
        println!(
            "  worst point               | {:>+8.4}  at ({:.2}, {:.2}, {:.2}) facing ({:.2}, {:.2}, {:.2})",
            worst.0,
            at.point.x + EYE.x as f32,
            at.point.y + EYE.y as f32,
            at.point.z + EYE.z as f32,
            at.normal.x,
            at.normal.y,
            at.normal.z
        );
    }
    println!();
}

/// Invented, missed, and the two means, over the points the grid reaches.
fn totals(shipped: &[f32], reference: &[f32], inside: &[bool]) -> (f64, f64, f64, f64) {
    let (mut invented, mut missed, mut truth, mut field) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for ((s, t), on) in shipped.iter().zip(reference).zip(inside) {
        if !on {
            continue;
        }
        let delta = f64::from(s - t);
        if delta > 0.0 {
            invented += delta;
        } else {
            missed += -delta;
        }
        truth += f64::from(*t);
        field += f64::from(*s);
    }
    (invented, missed, truth, field)
}

/// Where the disagreement lives, against how open the paths say a point is.
///
/// The row that separates the two failures by *where* they happen rather than by
/// their sign alone: a field that leaks does it in the buckets the paths call
/// dark, and one that over-rejects does it in the buckets they call open.
fn buckets(shipped: &[f32], reference: &[f32], inside: &[bool]) {
    println!("  true openness |  points |    paths |    field |     error");
    for (lo, hi) in [(0.0f32, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, f32::MAX)] {
        let (mut count, mut t, mut a) = (0usize, 0.0f64, 0.0f64);
        for ((s, r), on) in shipped.iter().zip(reference).zip(inside) {
            if *on && *r >= lo && *r < hi {
                count += 1;
                t += f64::from(*r);
                a += f64::from(*s);
            }
        }
        if count == 0 {
            continue;
        }
        let c = count as f64;
        let label = match hi {
            f32::MAX => format!("{lo:.2}+     "),
            _ => format!("{lo:.2}..{hi:.2}"),
        };
        println!(
            "  {label:>13} | {count:>7} | {:>8.4} | {:>8.4} | {:>+9.4}",
            t / c,
            a / c,
            (a - t) / c
        );
    }
    println!();
}

/// `r.gi_spacing`'s plateau, in the units the clamp leaves it in.
///
/// **The asked-for column is not the axis and the effective one is.** Eight
/// probes an axis over a room 25 m across is a spacing of at least 3.6 m
/// whatever the CVar says, so every request below that lands on the same handful
/// of whole multiples — which is the reading. In a room this size the knob has
/// no choices left, and what is being graded is the *clamp*: the sweep asks for
/// values that come out spread rather than values that look evenly spaced.
///
/// Denser is not monotonically better either, and the two columns are what shows
/// it: a coarse grid puts probes further inside walls, which leaks, while a fine
/// one at this clamp is the same grid with a different label on it.
fn spacings(
    renderer: &mut OffscreenRenderer,
    world: &World,
    points: &[Sample],
    reference: &[f32],
) -> Result<()> {
    println!(
        "   asked | effective | probes | inside | invented |   missed |     mean |    field | worst"
    );
    for spacing in [2.0f64, 4.0, 4.5, 5.0, 6.0, 8.0] {
        cvars::GI_SPACING.set_float(spacing);
        let shipped = rendered(renderer, world, points)?;
        let inside = covered(renderer, points);
        let (_, probes) = renderer.field_pending();
        let effective = renderer.field_grid().map_or(0.0, |(_, s, _)| s);
        let n = inside.iter().filter(|c| **c).count();
        row(
            &format!("{spacing:>8.2} | {effective:>9.2} | {probes:>6} | {n:>6}"),
            &shipped,
            reference,
            &inside,
        );
    }
    cvars::GI_SPACING.set_float(2.0);
    println!();
    Ok(())
}

/// The distance tile's edge, which is the moment kernel's sharpness: `fs_moments`
/// derives its lobe from a texel's own solid angle, so a wider tile is a *harder*
/// visibility test and not merely a finer one. Two texels is one moment per
/// octant and cannot represent a doorway; eight is where a wall's own edge stops
/// being blurred across the probe's whole sphere.
fn tiles(
    renderer: &mut OffscreenRenderer,
    world: &World,
    points: &[Sample],
    reference: &[f32],
) -> Result<()> {
    println!(
        "    tile | effective | probes | inside | invented |   missed |     mean |    field | worst"
    );
    for tile in [2i64, 4, 8] {
        cvars::GI_MOMENTS.set_int(tile);
        let shipped = rendered(renderer, world, points)?;
        let inside = covered(renderer, points);
        let (_, probes) = renderer.field_pending();
        let n = inside.iter().filter(|c| **c).count();
        let effective = renderer.field_grid().map_or(0.0, |(_, s, _)| s);
        row(
            &format!("{tile:>8} | {effective:>9.2} | {probes:>6} | {n:>6}"),
            &shipped,
            reference,
            &inside,
        );
    }
    cvars::GI_MOMENTS.set_int(4);
    println!();
    Ok(())
}

/// One sweep row: the two columns that disagree, then the ones that hide it.
fn row(label: &str, shipped: &[f32], reference: &[f32], inside: &[bool]) {
    let (invented, missed, _, field) = totals(shipped, reference, inside);
    let n = inside.iter().filter(|c| **c).count().max(1) as f64;
    let mut worst = 0.0f32;
    for ((s, t), on) in shipped.iter().zip(reference).zip(inside) {
        if *on && (s - t).abs() > worst.abs() {
            worst = s - t;
        }
    }
    // The mean is beside them because it is the number a single-figure error
    // would have reported, and watching it stand still while the two columns
    // trade is the whole argument for printing two.
    println!(
        "  {label} | {:>8.4} | {:>8.4} | {:>8.4} | {:>8.4} | {worst:>+7.4}",
        invented / n,
        missed / n,
        (invented + missed) / n,
        field / n
    );
}

/// `ambient_light`'s front-to-back composite, in Rust. Restated rather than
/// shared for the same reason `environment_weight` is: the two never meet at
/// runtime, and both are the same six lines.
fn composite(
    projected: &[(&gg_extract::ExtractedSky, [[f32; 4]; sky::SH_COEFFICIENTS])],
    at: &Sample,
) -> render::Vec3 {
    let point = render::Vec3::new(at.point.x, at.point.y, at.point.z);
    let normal = render::Vec3::new(at.normal.x, at.normal.y, at.normal.z);
    let mut sh = [[0.0f32; 4]; sky::SH_COEFFICIENTS];
    let mut budget = 1.0f32;
    for (extracted, coefficients) in projected {
        if budget <= 1e-3 {
            break;
        }
        let weight = extracted.weight_at(point).min(1.0) * budget;
        if weight <= 0.0 {
            continue;
        }
        budget -= weight;
        for (out, c) in sh.iter_mut().zip(coefficients) {
            for k in 0..3 {
                out[k] += c[k] * weight;
            }
        }
    }
    sky::irradiance(&sh, normal)
}

/// The demo's world, for the environments and lights the extract path reads.
/// Only the components this instrument needs are registered — a `Walker` and a
/// `Session` would be gameplay state nothing here asks a question about.
fn world() -> Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Sky>()?;
    world.register::<Light>()?;
    let put = |world: &mut World, sky: Sky| -> Result<()> {
        let entity = world.spawn();
        world.insert(entity, sky)?;
        Ok(())
    };
    put(&mut world, Sky::daylight(demo::SKY_INTENSITY))?;
    put(
        &mut world,
        Sky {
            zenith: 0x0056_5a60,
            horizon: 0x006a_6862,
            ground: 0x0048_423a,
            ..Sky::daylight(demo::SHELTER_INTENSITY)
        }
        .within(demo::SHELTER.0, demo::SHELTER.1, demo::SHELTER_FADE),
    )?;
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

/// The world's sky along an escaping direction — the outdoor one only.
///
/// The shelter's volume is *not* consulted here and that is the point rather
/// than an omission: a path that escapes the room has left every box in it, and
/// what it sees is the sky. An authored volume applies to a *fragment's
/// position*, which is a different question and the one [`authored`] compares
/// against.
fn environment(direction: sim::Vec3) -> sim::Vec3 {
    let look = daylight();
    let r = sky::radiance(
        &look,
        render::Vec3::new(direction.x, direction.y, direction.z),
    );
    sim::Vec3::new(r.x, r.y, r.z)
}

fn daylight() -> SkyLook {
    let declared = Sky::daylight(demo::SKY_INTENSITY);
    SkyLook {
        zenith: declared.zenith,
        horizon: declared.horizon,
        ground: declared.ground,
        intensity: declared.intensity,
        environment: 0,
    }
}

/// The same solids with every albedo at zero — the control leg that isolates
/// "the sky through what the geometry left open" from "the sky after it
/// bounced".
fn blacken(solids: &[ao::Occluder]) -> Vec<ao::Occluder> {
    solids
        .iter()
        .map(|s| ao::Occluder {
            albedo: sim::Vec3::ZERO,
            ..*s
        })
        .collect()
}

fn mean_irradiance(
    scene: &bounce::Scene,
    points: &[Sample],
    samples: u32,
    bounces: u32,
) -> sim::Vec3 {
    let mut total = sim::Vec3::ZERO;
    for s in points {
        total += bounce::irradiance(
            s.point,
            s.normal,
            scene,
            environment,
            bounce::Params {
                samples,
                bounces,
                ..bounce::Params::default()
            },
        );
    }
    total / points.len().max(1) as f32
}

/// Rec. 709 luminance — the one scalar a three-channel comparison reduces to
/// without picking a channel.
fn luminance(v: sim::Vec3) -> f32 {
    0.2126 * v.x + 0.7152 * v.y + 0.0722 * v.z
}

fn luminance_render(v: render::Vec3) -> f32 {
    0.2126 * v.x + 0.7152 * v.y + 0.0722 * v.z
}

/// Demo 12's room as solids, camera-relative, with each box's declared colour
/// decoded as its diffuse albedo — which is what the shipped shader does with
/// the same `u32` (`scene.rs`'s tint), so the reference reflects what the
/// renderer reflects.
fn room() -> Vec<ao::Occluder> {
    demo::ROOM
        .iter()
        .map(|&(center, half_extent, color)| {
            let linear = srgb_to_linear(color);
            ao::Occluder {
                center: sim::Vec3::new(
                    (center.x - EYE.x) as f32,
                    (center.y - EYE.y) as f32,
                    (center.z - EYE.z) as f32,
                ),
                rotation: sim::Quat::IDENTITY,
                half_extent,
                sphere: false,
                albedo: sim::Vec3::new(linear[0], linear[1], linear[2]),
                emission: sim::Vec3::ZERO,
            }
        })
        .collect()
}

fn sun() -> bounce::Sun {
    let ink = srgb_to_linear(demo::SUN_INK);
    bounce::Sun {
        direction: demo::SUN.try_normalize().unwrap_or(-sim::Vec3::Y),
        radiance: sim::Vec3::new(ink[0], ink[1], ink[2]) * demo::SUN_INTENSITY,
    }
}

fn lamps() -> Vec<bounce::Lamp> {
    let ink = srgb_to_linear(demo::LAMP_INK);
    demo::LAMPS
        .iter()
        .map(|at| bounce::Lamp {
            position: sim::Vec3::new(
                (at.x - EYE.x) as f32,
                (at.y - EYE.y) as f32,
                (at.z - EYE.z) as f32,
            ),
            radiance: sim::Vec3::new(ink[0], ink[1], ink[2]) * demo::LAMP_INTENSITY,
            range: demo::LAMP_RANGE,
        })
        .collect()
}

/// Ray through pixel `(x, y)`, in the eye's frame — `ao`'s, restated because
/// the two instruments frame the same room from the same chair.
fn ray(x: u32, y: u32) -> sim::Vec3 {
    let aspect = EXTENT.0 as f32 / EXTENT.1 as f32;
    let tan = sim::tan(FOV * 0.5);
    let sx = ((x as f32 + 0.5) / EXTENT.0 as f32 * 2.0 - 1.0) * tan * aspect;
    let sy = (1.0 - (y as f32 + 0.5) / EXTENT.1 as f32 * 2.0) * tan;
    let (sin_yaw, cos_yaw) = sim::sin_cos(YAW);
    let (sin_pitch, cos_pitch) = sim::sin_cos(PITCH);
    let forward = sim::Vec3::new(-sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch);
    let right = sim::Vec3::new(cos_yaw, 0.0, -sin_yaw);
    let up = forward.cross(right) * -1.0;
    (forward + right * sx + up * sy)
        .try_normalize()
        .unwrap_or(forward)
}
