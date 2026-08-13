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

use anyhow::Result;
use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable, Sky};
use gg_extract::{Extracted, SkyLook};
use gg_math::{render, sim};
use gg_render::{View, ao, bounce, sky, srgb_to_linear};

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
    Ok(())
}

/// A point the camera can see, with the normal of the surface it sits on.
struct Sample {
    point: sim::Vec3,
    normal: sim::Vec3,
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
