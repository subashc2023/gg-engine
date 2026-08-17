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
    // `--energy` alone, because that leg needs no path tracer and the rest of this
    // command spends about a minute on one: it is the leg an experiment on the
    // field's own constants is run against, over and over.
    if args.iter().any(|a| a == "--energy") {
        return only_energy();
    }
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

/// Frames past which an unconverged field is a defect rather than a transient —
/// the largest grid `probe::MAX_PER_AXIS` allows at `r.gi_rate 1`, and margin.
const FIELD_CEILING: usize = 8192;

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

    spacings(&world, points, &reference)?;
    burials(&world, points, &reference)?;
    biases(&world, points, &reference)?;
    offsets(&world, points, &reference)?;
    tiles(&world, points, &reference)?;
    energy()?;

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
    // **Every frame of the budget, and no early break on `pending`** (§6 M67).
    //
    // `pending` counts probes never gathered *since the grid was fitted*, which is
    // not the question a sweep asks: a leg that changes the world, or the moment
    // tile's edge, or anything else the records depend on but the grid does not,
    // leaves `covers` satisfied — so no refit, so `pending` is already zero, so the
    // loop used to stop after **one** frame with two thirds of the field still
    // holding the previous leg's records.
    //
    // What that cost is worth writing down: the slab leg read 0.73 against a truth
    // of 1.0 when it ran after the room, and 0.96 on its own device — the whole
    // 27 % was the room's walls, still in the field, lighting a world with no walls
    // in it. The tile table is the other one it reached: `r.gi_moments` re-lays out
    // the moment image without moving the grid, so rows 4 and 8 were graded on tiles
    // two thirds written at the *previous* edge, which is where "8 nearly triples the
    // leak" (§6 M36) came from.
    // A floor and then a *condition*, because how many frames a grid needs is a
    // property of the grid: `probes / r.gi_rate`, which a constant can only be
    // right about for one axis cap. It was 32 against a 512-probe ceiling, and §6
    // M68 raised that ceiling.
    let mut lit = frame(renderer, world)?;
    let mut frames = 1;
    while frames < FIELD_FRAMES || renderer.field_pending().0 > 0 {
        lit = frame(renderer, world)?;
        frames += 1;
        let (pending, probes) = renderer.field_pending();
        anyhow::ensure!(
            frames <= FIELD_CEILING,
            "the field did not converge in {frames} frames: {pending} of {probes} ungathered"
        );
    }
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
fn spacings(world: &World, points: &[Sample], reference: &[f32]) -> Result<()> {
    println!(
        "   asked | effective | probes | inside | invented |   missed |     mean |    field | worst |    whole |   level |   grain"
    );
    // **This sweep is the extent/spacing trade and not only a cell size** (§6 M68).
    // Demo 12's room is 24 m across, so anything under 24/7 anchors and buys finer
    // cells over a window smaller than the level, while 4 m and up still fit the
    // whole room the way every grid did before M68 — the 4.00 row *is* the shipped
    // pre-M68 grid, which is what makes `whole` a before-and-after rather than a
    // curve with no zero on it.
    let was = cvars::GI_SPACING.float();
    for spacing in [1.0f64, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0] {
        cvars::GI_SPACING.set_float(spacing);
        graded(&format!("{spacing:>8.2}"), world, points, reference)?;
    }
    cvars::GI_SPACING.set_float(was);
    println!();
    Ok(())
}

/// The field's own energy, in the one case whose answer needs no reference at all
/// (§6 M67) — `furnace`'s standard, one term along.
///
/// **One slab under a uniform sky, sampled on its own top face.** A point there
/// with an up normal receives the whole sky hemisphere and nothing else: the slab
/// takes none of that hemisphere away, and its own bounce is coplanar with the
/// point and so reaches it at grazing incidence with zero measure. So the truth is
/// `pi * L` over `pi * L`, which is **1.0 exactly**, at every sampled pixel, with
/// no path tracer and no tolerance to argue about.
///
/// It is also the case that puts the *whole* answer in the L1 record's first band:
/// the surrounding radiance is a hemispherical step, sky above and a dim floor
/// below, and `probe.rs`'s header claims that exact case is reconstructed exactly —
/// band 1 and the constant cancelling against the cosine kernel. So a number below
/// 1.0 here is not a sampling error and not a placement error. It is the record
/// itself, and it is the one reading that separates *the field is approximate* from
/// *the field is short of energy*.
///
/// Which matters because §6 M67 found the graded room to be propped up by a leak:
/// every change that removed invented light — a physically correct back face, a
/// lattice off the level's own numbers, a stricter burial slope — made the total
/// error worse, because the field is under-lit and the leak was paying for it. This
/// leg is where that claim stops being an inference across three sweeps.
/// [`energy`] and its sweep alone, for `--energy` — both bring their own devices.
fn only_energy() -> Result<()> {
    cvars::AMBIENT.set_float(AMBIENT);
    cvars::DITHER.set_float(0.0);
    cvars::AO.set_bool(false);
    cvars::GI_RATE.set_int(0);
    energy()?;
    scales()?;
    cvars::DITHER.set_float(1.0);
    cvars::AO.set_bool(true);
    cvars::GI_RATE.set_int(16);
    Ok(())
}

/// **The one measurement in this file with no reference implementation and no
/// scene dependence at all**, swept over the probe spacing (§6 M68).
///
/// [`energy`]'s slab, whose truth is `1.0` at every point by construction, at every
/// spacing worth shipping. The truth does not depend on the spacing, so **any**
/// dependence the reading has is a defect and not a trade: a finer grid resolves a
/// flat slab under a uniform sky no better and no worse than a coarse one, because
/// there is nothing there to resolve.
///
/// It is the leg that answers the question §6 M68's room tables could not. Those say
/// a finer field is darker and cannot say whether that is the resolution buying
/// accuracy somewhere a path tracer disagrees about, or the weighting failing at
/// short probe distances. Here there is no somewhere: one plane, one answer, and a
/// spacing column that ought to be constant.
fn scales() -> Result<()> {
    let world = slab()?;
    let solids = vec![room()[0]];
    let points = &surfaces(&solids);
    let (was_spacing, was_bias) = (cvars::GI_SPACING.float(), cvars::GI_BIAS.float());
    println!("\nthe same slab over the spacing — every row's truth is still 1.0\n");
    println!("  spacing |    bias | probes | inside |    field |  short | worst");
    for spacing in [1.0f64, 1.5, 2.0, 3.0, 4.0, 6.0] {
        for bias in [was_bias, 0.15 * spacing] {
            cvars::GI_SPACING.set_float(spacing);
            cvars::GI_BIAS.set_float(bias);
            let mut renderer = OffscreenRenderer::new(EXTENT)?;
            let shipped = rendered(&mut renderer, &world, points)?;
            let inside = covered(&renderer, points);
            let (_, probes) = renderer.field_pending();
            let n = inside.iter().filter(|c| **c).count();
            let (mut field, mut worst) = (0.0f64, 0.0f32);
            for (s, on) in shipped.iter().zip(&inside) {
                if *on {
                    field += f64::from(*s);
                    if (s - 1.0).abs() > worst.abs() {
                        worst = s - 1.0;
                    }
                }
            }
            let field = field / n.max(1) as f64;
            println!(
                "  {spacing:>7.2} | {bias:>7.2} | {probes:>6} | {n:>6} | {field:>8.4} | \
                 {:>5.1} % | {worst:>+7.4}",
                (1.0 - field) * 100.0
            );
            let report = renderer.shutdown();
            anyhow::ensure!(report.clean(), "unclean render: {report:?}");
        }
    }
    cvars::GI_SPACING.set_float(was_spacing);
    cvars::GI_BIAS.set_float(was_bias);
    println!();
    Ok(())
}

fn energy() -> Result<()> {
    let world = slab()?;
    // The slab's own surfaces, not the room's: a pixel that used to find a wall
    // finds sky here, and grading it against 1.0 would grade the background.
    let solids = vec![room()[0]];
    let points = &surfaces(&solids);
    // **Its own device, and this leg is the one that needed it most** (§6 M69).
    // It took the caller's renderer until then, which is the renderer that had just
    // rendered the *room* — and `Grid::covers` is satisfied by a grid that still
    // reaches the scene, so a 28 m room's grid covers a slab and no refit happens.
    // The leg read 1.0072 that way, 0.7 % *over* a truth of 1.0, against 0.9526 on
    // a device of its own. Which is the honest number is not a close call: a leg is
    // graded on the grid *its own world* produces, or the reference-free claim
    // ("the truth here is exactly 1.0, at every point") is about a different scene
    // than the one on screen. §6 M67 found this class twice and §6 M68 twice more;
    // this is the fourth site.
    let mut renderer = OffscreenRenderer::new(EXTENT)?;
    let shipped = rendered(&mut renderer, &world, points)?;
    let inside = covered(&renderer, points);
    let open = vec![1.0f32; points.len()];
    let (_, probes) = renderer.field_pending();
    println!("one slab under the same uniform sky — every point's truth is 1.0\n");
    println!(
        "  {probes} probes at {:.2} m effective, {} of {} points on the slab and inside the grid",
        renderer.field_grid().map_or(0.0, |(_, s, _)| s),
        inside.iter().filter(|c| **c).count(),
        points.len()
    );
    report(&shipped, &open, &inside, points);
    let report = renderer.shutdown();
    anyhow::ensure!(report.clean(), "unclean render: {report:?}");
    Ok(())
}

/// Demo 12's floor and nothing else — [`energy`]'s world.
///
/// The room's own first box, so the slab's size, thickness and albedo are the ones
/// every other table here was measured against rather than a second set of numbers
/// to keep in step.
fn slab() -> Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    let (center, half_extent, color) = demo::ROOM[0];
    let entity = world.spawn();
    world.insert(
        entity,
        Renderable::boxed(center, half_extent, color).surfaced(0.0, 0.0),
    )?;
    Ok(world)
}

/// Where the probe lattice sits between the multiples of its own spacing (§6 M67)
/// — `r.gi_offset`, as a fraction of a cell.
///
/// **The one knob here whose subject is the level rather than the renderer.** A
/// lattice through the world origin lands on whatever an author rounded to, and a
/// probe on a surface is worth a fraction of one: half its sphere is the inside of
/// that surface, so the burial term discounts it and the cosine wrap halves it
/// again for being coplanar with everything it should be lighting. Demo 12's room
/// is the worst case and got there by being ordinary — floor top at `y = 0`, wall
/// faces at `x, z = ±12`, wall tops at `y = 4`, all divisible by the 4 m spacing
/// the clamp widens to.
///
/// So the sweep is over *coincidence*, and the two ends are what say so: 0 is every
/// plane on a round number and 0.5 is every plane on a half — both authorable, and
/// the values between them are not. What it costs is in the probe column, since an
/// axis whose span is a whole multiple of the spacing spends one more plane to
/// reach its far bound.
/// `r.gi_burial`'s plateau: how fast a probe's weight falls with the fraction of its
/// own sphere that came back a back face (§6 M67).
///
/// The two columns fail in opposite directions and this is the knob that trades them
/// most directly — a probe inside a wall recorded the wall's inside, so trusting it
/// leaks light through the wall, and discounting it leaves the crease beside it lit by
/// the fallback. **Read the mean beside them**: the weights are *normalized*, so
/// discounting a buried probe does not remove its light, it hands its share to the
/// unburied probes in the cell — which are the brighter ones. That is why a higher
/// slope makes the room *brighter*, which is the opposite of what the name suggests.
///
/// One device, unlike [`offsets`]: the slope never moves the grid.
fn burials(world: &World, points: &[Sample], reference: &[f32]) -> Result<()> {
    println!(
        "  burial | effective | probes | inside | invented |   missed |     mean |    field | worst |    whole |   level |   grain"
    );
    let was = cvars::GI_BURIAL.float();
    for slope in [0.0f64, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0] {
        cvars::GI_BURIAL.set_float(slope);
        graded(&format!("{slope:>8.2}"), world, points, reference)?;
    }
    // The value on entry and not a literal: this reset read `1.0` while the
    // shipped default was 1.5, so every table below it in one run was measured at
    // a burial slope nobody chose (§6 M68).
    cvars::GI_BURIAL.set_float(was);
    println!();
    Ok(())
}

/// **A device per row, and that is not a detail** — [`Grid::covers`] is hysteresis
/// by design, so a renderer that already holds a grid reaching the scene refuses to
/// refit, and a knob that only moves the *lattice* would never take effect. Five of
/// this table's six rows read identically to four decimal places before that was
/// understood, because they were one grid graded five times.
/// `r.gi_bias`'s plateau, **and its unit** (§6 M68).
///
/// Two questions in one table, which is why it sweeps the offset at three
/// spacings rather than at the shipped one. The plateau itself is the ordinary
/// question — too small a push and a surface occludes itself, so the field misses
/// light in exactly the creases it exists for; too large and the shading point is
/// located a cell away from where it is, so the field invents light from the far
/// side of thin geometry.
///
/// The unit is the interesting half. This term was `0.15 * spacing` from §6 M36 to
/// M68, so it was **0.60 m at the 4 m grid the axis cap forced** and halved every
/// time somebody made the cells finer. If its plateau is at a fixed fraction the
/// old unit was right and the three curves agree; if it is at a fixed *distance*
/// they agree only after multiplying, and the coupling was hiding a defect that
/// looks exactly like "a finer field is worse".
///
/// A renderer per row, and per spacing: `Grid::covers`'s hysteresis refuses to
/// refit a grid that still reaches the scene (§6 M67).
fn biases(world: &World, points: &[Sample], reference: &[f32]) -> Result<()> {
    let (was_bias, was_spacing) = (cvars::GI_BIAS.float(), cvars::GI_SPACING.float());
    for spacing in [2.0f64, 3.0, 4.0] {
        cvars::GI_SPACING.set_float(spacing);
        println!(
            "    bias | in cells | probes | inside | invented |   missed |     mean |    field | \
             worst |    whole |   level |   grain"
        );
        for bias in [0.05f64, 0.15, 0.3, 0.45, 0.6, 0.9, 1.2] {
            cvars::GI_BIAS.set_float(bias);
            let mut renderer = OffscreenRenderer::new(EXTENT)?;
            let shipped = rendered(&mut renderer, world, points)?;
            let inside = covered(&renderer, points);
            let (_, probes) = renderer.field_pending();
            let effective = renderer.field_grid().map_or(1.0, |(_, s, _)| f64::from(s));
            let n = inside.iter().filter(|c| **c).count();
            // The fraction beside the metres, because the old unit is the column
            // the reader is being asked to compare against.
            row(
                &format!(
                    "{bias:>8.2} | {:>8.3} | {probes:>6} | {n:>6}",
                    bias / effective.max(1e-6)
                ),
                &shipped,
                reference,
                &inside,
                points,
            );
            let report = renderer.shutdown();
            anyhow::ensure!(report.clean(), "unclean render at bias {bias}: {report:?}");
        }
        println!("    ^ at {spacing:.2} m spacing\n");
    }
    cvars::GI_BIAS.set_float(was_bias);
    cvars::GI_SPACING.set_float(was_spacing);
    Ok(())
}

fn offsets(world: &World, points: &[Sample], reference: &[f32]) -> Result<()> {
    println!(
        "  offset | effective | probes | inside | invented |   missed |     mean |    field | worst |    whole |   level |   grain"
    );
    for offset in [0.0f64, 0.125, 0.25, 1.0 / 3.0, 0.4, 0.5] {
        cvars::GI_OFFSET.set_float(offset);
        graded(&format!("{offset:>8.3}"), world, points, reference)?;
    }
    cvars::GI_OFFSET.set_float(gg_render::cvars::LATTICE_OFFSET);
    println!();
    Ok(())
}

/// The distance tile's edge, which is the moment kernel's sharpness: `fs_moments`
/// derives its lobe from a texel's own solid angle, so a wider tile is a *harder*
/// visibility test and not merely a finer one. Two texels is one moment per
/// octant and cannot represent a doorway; eight is where a wall's own edge stops
/// being blurred across the probe's whole sphere.
fn tiles(world: &World, points: &[Sample], reference: &[f32]) -> Result<()> {
    println!(
        "    tile | effective | probes | inside | invented |   missed |     mean |    field | worst |    whole |   level |   grain"
    );
    let was = (cvars::GI_MOMENTS.int(), cvars::GI_FILTER.bool());
    // **Both reads of the tile, because this table is where the *accuracy* half of
    // §6 M69 is settled.** `gg-tools facets` says filtering removes the facets; what
    // it cannot say is whether a softer bound leaks — a bilinear read is a blur of
    // the blocker distance, and a blurred blocker is a blocker in slightly the
    // wrong place. `invented` against `missed` is the pair that would show it.
    for filter in [false, true] {
        cvars::GI_FILTER.set_bool(filter);
        for tile in [2i64, 4, 8] {
            cvars::GI_MOMENTS.set_int(tile);
            let read = if filter { "filt" } else { "pt  " };
            graded(&format!("{tile:>3} {read}"), world, points, reference)?;
        }
    }
    cvars::GI_MOMENTS.set_int(was.0);
    cvars::GI_FILTER.set_bool(was.1);
    println!();
    Ok(())
}

/// One sweep row: the two columns that disagree, then the ones that hide it.
/// One sweep row on **its own device**, which is not an optimisation detail but the
/// only way the rows mean the same thing (§6 M68).
///
/// Three separate pieces of deliberate state make a shared renderer carry one row's
/// answer into the next: `Grid::covers`'s hysteresis holds a grid that still reaches
/// the scene (§6 M67 found this), `Grid::place`'s `sticky` holds an axis that has
/// once had to anchor, and the records themselves survive anything that is not a
/// refit. On one device the spacing sweep read 680 of 786 points inside the grid at
/// 2 m while a fresh renderer at the same spacing read 786 — the sweep had anchored
/// at 1 m three rows earlier and never let go.
fn graded(
    label: &str,
    world: &World,
    points: &[Sample],
    reference: &[f32],
) -> Result<(f64, usize)> {
    let mut renderer = OffscreenRenderer::new(EXTENT)?;
    let shipped = rendered(&mut renderer, world, points)?;
    let inside = covered(&renderer, points);
    let (_, probes) = renderer.field_pending();
    let effective = renderer.field_grid().map_or(0.0, |(_, s, _)| s);
    let n = inside.iter().filter(|c| **c).count();
    row(
        &format!("{label} | {effective:>9.2} | {probes:>6} | {n:>6}"),
        &shipped,
        reference,
        &inside,
        points,
    );
    let report = renderer.shutdown();
    anyhow::ensure!(report.clean(), "unclean render at {label}: {report:?}");
    Ok((f64::from(effective), n))
}

/// The error's **level** against its **grain** — a mean and a roughness, and the
/// pair that finally lets this command answer the question it was asked (§6 M68).
///
/// Every other column here is a mean absolute error, and a mean cannot see
/// structure. It therefore prefers a coarse field at every spacing, because the
/// field's error is dominated by *systematic darkness* and probes further from
/// surfaces are buried and rejected less often. That ranking is real and it is also
/// useless for the defect this milestone exists to fix: what was reported is a
/// chevron on a flat wall, which is a bilinear cell's iso-contours — the saddle a
/// trilinear interpolation makes when its eight corners disagree — and a room a
/// uniform 6 % dark scores worse on every column above while looking *correct*.
///
/// So: **level** is the signed mean of the error, the part a player reads as "this
/// room is a bit dim" and forgives, and **grain** is the mean absolute step in that
/// error between neighbouring sample points on the *same surface*, which is the part
/// they read as a stain on the wall and report. They fail in opposite directions —
/// a field can be exactly right on average and hideous, or uniformly wrong and
/// invisible — and the spacing is read off the second.
///
/// Neighbours are the decimated frame's own: [`STRIDE`] to the right and one row
/// down, taken only when both samples exist, share a normal, and are close enough
/// together to be the same surface rather than two sides of a silhouette. No
/// reference is needed for the grain beyond the truth already in hand, and none at
/// all for the claim that a smooth truth should not produce a rough error.
fn grain(shipped: &[f32], reference: &[f32], inside: &[bool], points: &[Sample]) -> (f64, f64) {
    let mut level = 0.0f64;
    let mut counted = 0usize;
    for ((s, t), on) in shipped.iter().zip(reference).zip(inside) {
        if *on {
            level += f64::from(s - t);
            counted += 1;
        }
    }
    // Pixel -> sample, so a neighbour is a lookup rather than a search.
    let mut at = vec![usize::MAX; (EXTENT.0 * EXTENT.1) as usize];
    for (i, p) in points.iter().enumerate() {
        if let Some(slot) = at.get_mut(p.pixel) {
            *slot = i;
        }
    }
    let error = |i: usize| f64::from(shipped[i] - reference[i]);
    let mut rough = 0.0f64;
    let mut pairs = 0usize;
    for (i, p) in points.iter().enumerate() {
        if !inside[i] {
            continue;
        }
        for step in [STRIDE as usize, (STRIDE * EXTENT.0) as usize] {
            let Some(&j) = at.get(p.pixel + step) else {
                continue;
            };
            if j == usize::MAX || !inside[j] {
                continue;
            }
            let other = &points[j];
            // Same plane, and adjacent on it: a silhouette edge puts two unrelated
            // surfaces one pixel apart, and the step across it is geometry rather
            // than a defect in the field.
            if p.normal.dot(other.normal) < 0.99 {
                continue;
            }
            let apart = (p.point - other.point).length();
            if !(apart > 1e-4 && apart < 1.0) {
                continue;
            }
            rough += (error(i) - error(j)).abs();
            pairs += 1;
        }
    }
    (level / counted.max(1) as f64, rough / pairs.max(1) as f64)
}

fn row(label: &str, shipped: &[f32], reference: &[f32], inside: &[bool], points: &[Sample]) {
    let (invented, missed, _, field) = totals(shipped, reference, inside);
    let n = inside.iter().filter(|c| **c).count().max(1) as f64;
    // **The whole scene, and the only column comparable across rows whose coverage
    // differs** (§6 M68). Every column beside it is normalised by the points the
    // field *reached*, which was one population while a grid always spanned the
    // scene and is a different one per row now that a window can be smaller than
    // the level. A placement that covers a third of the room perfectly reads
    // beautifully there and is not better; what a player sees is this, because a
    // point outside the field is shaded by the fallback rather than skipped.
    let whole = {
        let every = vec![true; inside.len()];
        let (invented, missed, _, _) = totals(shipped, reference, &every);
        (invented + missed) / inside.len().max(1) as f64
    };
    let mut worst = 0.0f32;
    for ((s, t), on) in shipped.iter().zip(reference).zip(inside) {
        if *on && (s - t).abs() > worst.abs() {
            worst = s - t;
        }
    }
    // The mean is beside them because it is the number a single-figure error
    // would have reported, and watching it stand still while the two columns
    // trade is the whole argument for printing two.
    let (level, rough) = grain(shipped, reference, inside, points);
    println!(
        "  {label} | {:>8.4} | {:>8.4} | {:>8.4} | {:>8.4} | {worst:>+7.4} | {whole:>8.4} |          {level:>+7.4} | {rough:>7.4}",
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
