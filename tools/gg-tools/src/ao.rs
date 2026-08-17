//! `gg-tools ao` — how much ambient occlusion a scene actually has, and how far
//! a depth buffer gets toward it (§6 M35).
//!
//! Two questions, and the first one had to be answered before the second was
//! worth asking. **Is there anything here?** The engine's ambient term arrives
//! unoccluded on any surface whose material carries no baked occlusion map,
//! which is every `Renderable` in the tree; whether that is a visible wrong or a
//! rounding error is a property of the *content*, so it is measured against
//! demo 12's room — shipped geometry, and a table of axis-aligned boxes, which
//! `gg_render::ao` represents exactly rather than approximately.
//!
//! **And how close does the screen-space pass get?** Screen-space AO has an
//! error that no amount of tuning removes: it sees a depth buffer, so an
//! occluder off the edge of the frame or hidden behind a nearer surface is not
//! there at all. The reference casts against the scene, so the gap between them
//! is exactly that structural error plus whatever the approximation adds, and
//! the point of printing it is that "AO looks about right" cannot tell the two
//! apart.

use anyhow::Result;
use gg_ecs::World;
use gg_ecs::boundary::Renderable;
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, ao, cvars};

/// The frame the truth pass is measured over. Small on purpose: every pixel
/// costs a primary cast plus [`ao::Params::samples`] shadow rays, and the
/// distribution of a scene's occlusion converges long before its silhouettes do.
const EXTENT: (u32, u32) = (320, 180);

/// Vertical field of view, radians — `r.fov`'s default, so the frame is the one
/// the demo renders.
const FOV: f32 = 1.0;

/// Where the truth pass stands in demo 12's room: the player's spawn, at eye
/// height, looking across the floor at the stairs and the crates.
const EYE: sim::DVec3 = sim::DVec3::new(0.0, 1.62, 8.0);
const YAW: f32 = 0.0;
const PITCH: f32 = -0.22;

/// Radii swept for the "is there anything here" table. AO is a local term and
/// what counts as local is the knob with the widest reasonable range — a
/// centimetre-scale radius creases contacts only, a metre-scale one darkens
/// whole alcoves.
const RADII: [f32; 4] = [0.25, 0.5, 1.0, 2.0];

pub fn run(_args: &[String]) -> Result<()> {
    truth();
    grade()?;
    Ok(())
}

/// Demo 12's room as occluders, camera-relative.
fn room() -> Vec<ao::Occluder> {
    demo_12_shooter::ROOM
        .iter()
        .map(|&(center, half_extent, _)| ao::Occluder {
            center: sim::Vec3::new(
                (center.x - EYE.x) as f32,
                (center.y - EYE.y) as f32,
                (center.z - EYE.z) as f32,
            ),
            rotation: sim::Quat::IDENTITY,
            half_extent,
            sphere: false,
            // Occlusion is a property of shape: what a box is made of does not
            // change what it blocks. `bounce` is where these matter (§6 M36).
            albedo: sim::Vec3::ZERO,
            emission: sim::Vec3::ZERO,
        })
        .collect()
}

/// Ray through pixel `(x, y)`, in the eye's frame.
fn ray(x: u32, y: u32) -> sim::Vec3 {
    let aspect = EXTENT.0 as f32 / EXTENT.1 as f32;
    let tan = sim::tan(FOV * 0.5);
    let sx = ((x as f32 + 0.5) / EXTENT.0 as f32 * 2.0 - 1.0) * tan * aspect;
    let sy = (1.0 - (y as f32 + 0.5) / EXTENT.1 as f32 * 2.0) * tan;
    let (sin_yaw, cos_yaw) = sim::sin_cos(YAW);
    let (sin_pitch, cos_pitch) = sim::sin_cos(PITCH);
    // Right-handed, looking down -Z at rest — `gg-extract`'s convention.
    let forward = sim::Vec3::new(-sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch);
    let right = sim::Vec3::new(cos_yaw, 0.0, -sin_yaw);
    let up = forward.cross(right) * -1.0;
    (forward + right * sx + up * sy)
        .try_normalize()
        .unwrap_or(forward)
}

/// What the scene's real occlusion is, per radius: how much of the frame it
/// darkens and by how much.
///
/// The mean over *occluded* pixels rather than over the frame is the number that
/// matters — a term that moves 4 % of the picture by 40 % is a visible wrong,
/// and one that moves 90 % of it by 2 % is a slightly darker exposure.
fn truth() {
    let scene = room();
    println!(
        "demo 12's room, {} boxes, {}x{} from the spawn\n",
        scene.len(),
        EXTENT.0,
        EXTENT.1
    );
    println!("  radius |    hit |  <0.98 |  <0.75 |  <0.50 |   mean | mean where occluded");
    for radius in RADII {
        let params = ao::Params {
            radius,
            ..ao::Params::default()
        };
        let (mut hits, mut total, mut occluded) = (0u32, 0.0f64, 0.0f64);
        let (mut some, mut quarter, mut half) = (0u32, 0u32, 0u32);
        for y in 0..EXTENT.1 {
            for x in 0..EXTENT.0 {
                let direction = ray(x, y);
                let Some(hit) = ao::trace(sim::Vec3::ZERO, direction, &scene, 1e-3, 1e4) else {
                    continue;
                };
                let point = direction * hit.distance;
                let open = ao::occlusion(point, hit.normal, &scene, params);
                hits += 1;
                total += f64::from(open);
                if open < 0.98 {
                    some += 1;
                    occluded += f64::from(open);
                }
                quarter += u32::from(open < 0.75);
                half += u32::from(open < 0.5);
            }
        }
        let pixels = f64::from(EXTENT.0 * EXTENT.1);
        let share = |n: u32| f64::from(n) / f64::from(hits.max(1)) * 100.0;
        println!(
            "  {radius:>6.2} | {:>5.1}% | {:>5.1}% | {:>5.1}% | {:>5.1}% | {:>6.3} | {:>6.3}",
            f64::from(hits) / pixels * 100.0,
            share(some),
            share(quarter),
            share(half),
            total / f64::from(hits.max(1)),
            occluded / f64::from(some.max(1)),
        );
    }
    println!();
}

// ------------------------------------------------------------------ grade ----

/// The graded frame. Larger than [`EXTENT`] because what is being measured is a
/// *screen-space* estimate, and how well it does is partly a question of how
/// many pixels a radius covers.
const FRAME: (u32, u32) = (480, 270);

/// Linear ambient the graded scene is lit by.
///
/// Low enough that the tonemapper is the identity — PBR Neutral's knee is at
/// 0.76 and nothing here reaches a third of it — which is what makes the
/// comparison below a ratio of two decoded code values rather than an inversion
/// of three curves. The furnace has to invert them because it compares a surface
/// against a *background*; this compares a pixel against itself.
const AMBIENT: f64 = 0.25;

/// Where the graded scene is viewed from.
const GRADE_EYE: sim::DVec3 = sim::DVec3::new(0.0, 1.4, 4.2);
const GRADE_PITCH: f32 = -0.30;

/// The graded scene: a floor, a back wall, boxes in contact with the floor, and
/// a slab held clear of it.
///
/// Built here rather than taken from a demo, for the furnace's reason: a scene
/// whose job is to make an estimator disagree with the truth should contain the
/// cases that make it disagree. The slab is the sharpest of them — its underside
/// is occluded by geometry that is *behind the receiver*, which a depth buffer
/// cannot see and no tuning recovers.
fn graded_scene() -> Vec<(sim::DVec3, sim::Vec3)> {
    vec![
        (
            sim::DVec3::new(0.0, -0.5, 0.0),
            sim::Vec3::new(6.0, 0.5, 6.0),
        ),
        (
            sim::DVec3::new(0.0, 1.5, -2.5),
            sim::Vec3::new(6.0, 2.0, 0.5),
        ),
        (
            sim::DVec3::new(-1.3, 0.35, 0.2),
            sim::Vec3::new(0.35, 0.35, 0.35),
        ),
        (
            sim::DVec3::new(0.0, 0.2, -0.6),
            sim::Vec3::new(0.6, 0.2, 0.3),
        ),
        (
            sim::DVec3::new(1.2, 0.5, -0.2),
            sim::Vec3::new(0.25, 0.5, 0.25),
        ),
        (
            sim::DVec3::new(-0.2, 0.95, -1.4),
            sim::Vec3::new(1.1, 0.08, 0.5),
        ),
    ]
}

fn graded_world() -> Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    for (position, half_extent) in graded_scene() {
        let entity = world.spawn();
        // White and fully rough. With no sky declared, the ambient path is
        // `ambient * diffuse * occlusion` and nothing else, so the picture *is*
        // the occlusion times a constant.
        world.insert(
            entity,
            Renderable::boxed(position, half_extent, 0x00ff_ffff).surfaced(0.0, 0.0),
        )?;
    }
    Ok(world)
}

fn graded_occluders() -> Vec<ao::Occluder> {
    graded_scene()
        .into_iter()
        .map(|(center, half_extent)| ao::Occluder {
            center: sim::Vec3::new(
                (center.x - GRADE_EYE.x) as f32,
                (center.y - GRADE_EYE.y) as f32,
                (center.z - GRADE_EYE.z) as f32,
            ),
            rotation: sim::Quat::IDENTITY,
            half_extent,
            sphere: false,
            // Occlusion is a property of shape: what a box is made of does not
            // change what it blocks. `bounce` is where these matter (§6 M36).
            albedo: sim::Vec3::ZERO,
            emission: sim::Vec3::ZERO,
        })
        .collect()
}

/// Half-height of the orthographic leg, metres. Chosen so the graded scene fills
/// a similar share of the frame as the perspective leg does, since the whole
/// question is whether the *estimator* behaves and not how big the boxes look.
const ORTHO: f32 = 2.2;

fn graded_view_at(ortho: bool) -> View {
    View {
        pitch: GRADE_PITCH,
        ortho: if ortho { ORTHO } else { 0.0 },
        ortho_far: 60.0,
        ..View::default()
    }
}

/// The primary ray through pixel `(x, y)` of the graded frame — origin and
/// direction, in the eye's frame.
///
/// Two projections, because the pass has two and one of them was silently doing
/// nothing until it was measured. Under perspective every ray leaves the eye;
/// under an orthographic view they are parallel and it is the *origin* that
/// moves, which is the same split `View::view_projection` makes.
fn grade_ray(x: u32, y: u32, ortho: bool) -> (sim::Vec3, sim::Vec3) {
    let aspect = FRAME.0 as f32 / FRAME.1 as f32;
    let view = graded_view_at(ortho);
    let ndc_x = (x as f32 + 0.5) / FRAME.0 as f32 * 2.0 - 1.0;
    let ndc_y = 1.0 - (y as f32 + 0.5) / FRAME.1 as f32 * 2.0;
    let (sin_pitch, cos_pitch) = sim::sin_cos(view.pitch);
    let forward = sim::Vec3::new(0.0, sin_pitch, -cos_pitch);
    let right = sim::Vec3::new(1.0, 0.0, 0.0);
    let up = forward.cross(right) * -1.0;
    if ortho {
        let origin = right * (ndc_x * ORTHO * aspect) + up * (ndc_y * ORTHO);
        return (origin, forward);
    }
    let tan = sim::tan(view.fov_y * 0.5);
    let direction = (forward + right * (ndc_x * tan * aspect) + up * (ndc_y * tan))
        .try_normalize()
        .unwrap_or(forward);
    (sim::Vec3::ZERO, direction)
}

/// sRGB decode, IEC 61966-2-1 — `post.slang`'s encode run backwards.
fn decode(code: u8) -> f32 {
    let e = f32::from(code) / 255.0;
    if e <= 0.040_45 {
        e / 12.92
    } else {
        sim::powf((e + 0.055) / 1.055, 2.4)
    }
}

/// One render of the graded scene, as decoded linear luminance per pixel.
fn render_scene(renderer: &mut OffscreenRenderer, world: &World) -> Result<Vec<f32>> {
    render_scene_at(renderer, world, false)
}

fn render_scene_at(
    renderer: &mut OffscreenRenderer,
    world: &World,
    ortho: bool,
) -> Result<Vec<f32>> {
    let view = graded_view_at(ortho);
    let mut extracted = Extracted::default();
    extracted.clear(GRADE_EYE, view.frustum(FRAME));
    extracted.append::<Renderable>(world)?;
    extracted.append_lights(world)?;
    let pixels = renderer
        .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?
        .pixels;
    Ok(pixels.chunks_exact(4).map(|p| decode(p[1])).collect())
}

/// The rendered occlusion, per pixel: the same frame with the pass on and off.
///
/// A **ratio of the same pixel**, which is what makes this exact rather than
/// calibrated. Occlusion multiplies the ambient term and nothing else touches
/// it, so whatever the normal and the encode did to a pixel, they did to both —
/// and below the tonemapper's knee the encode is the only nonlinearity left,
/// which `decode` undoes.
fn rendered_occlusion(renderer: &mut OffscreenRenderer, world: &World) -> Result<Vec<f32>> {
    rendered_occlusion_at(renderer, world, false)
}

fn rendered_occlusion_at(
    renderer: &mut OffscreenRenderer,
    world: &World,
    ortho: bool,
) -> Result<Vec<f32>> {
    cvars::AO.set_bool(false);
    let open = render_scene_at(renderer, world, ortho)?;
    cvars::AO.set_bool(true);
    let occluded = render_scene_at(renderer, world, ortho)?;
    Ok(occluded
        .iter()
        .zip(&open)
        .map(|(a, b)| if *b > 1e-5 { (a / b).min(1.0) } else { 1.0 })
        .collect())
}

/// The same frame, cast rather than rendered.
fn reference_occlusion(radius: f32) -> (Vec<f32>, Vec<bool>) {
    reference_occlusion_at(radius, false)
}

fn reference_occlusion_at(radius: f32, ortho: bool) -> (Vec<f32>, Vec<bool>) {
    let scene = graded_occluders();
    let params = ao::Params {
        radius,
        ..ao::Params::default()
    };
    let mut values = vec![1.0f32; (FRAME.0 * FRAME.1) as usize];
    let mut covered = vec![false; values.len()];
    for y in 0..FRAME.1 {
        for x in 0..FRAME.0 {
            let (origin, direction) = grade_ray(x, y, ortho);
            let Some(hit) = ao::trace(origin, direction, &scene, 1e-3, 1e4) else {
                continue;
            };
            let at = (y * FRAME.0 + x) as usize;
            let point = origin + direction * hit.distance;
            values[at] = ao::occlusion(point, hit.normal, &scene, params);
            covered[at] = true;
        }
    }
    (values, covered)
}

/// Where the error lives, split by what the truth says about the pixel.
///
/// The single most useful row here, because the two failures are opposite and an
/// absolute mean cannot tell them apart: a screen-space estimate that **invents**
/// occlusion darkens open surfaces and reads as dirt, and one that **misses** it
/// leaves a contact crease lit and reads as nothing at all. Signed, so the sign
/// is the diagnosis — negative is too dark.
fn buckets(rendered: &[f32], reference: &[f32], covered: &[bool]) {
    const EDGES: [f32; 4] = [0.999, 0.9, 0.7, 0.0];
    println!("  reference |  pixels |  signed |   worst | reading");
    let mut lo = 1.001f32;
    for edge in EDGES {
        let (mut total, mut worst, mut n) = (0.0f64, 0.0f32, 0u32);
        for ((r, t), on) in rendered.iter().zip(reference).zip(covered) {
            if !on || *t >= lo || *t < edge {
                continue;
            }
            total += f64::from(r - t);
            if (r - t).abs() > worst.abs() {
                worst = r - t;
            }
            n += 1;
        }
        let reading = if edge >= 0.999 {
            "open surface; negative is invented occlusion"
        } else {
            "occluded; positive is occlusion the depth buffer could not see"
        };
        println!(
            "  >={edge:<7.3} | {n:>7} | {:>7.4} | {worst:>7.4} | {reading}",
            (total / f64::from(n.max(1))) as f32
        );
        lo = edge;
    }
    println!();
}

/// Mean and worst absolute error over the pixels the scene covers.
fn error(rendered: &[f32], reference: &[f32], covered: &[bool]) -> (f32, f32) {
    let (mut total, mut worst, mut n) = (0.0f64, 0.0f32, 0u32);
    for ((r, t), on) in rendered.iter().zip(reference).zip(covered) {
        if !on {
            continue;
        }
        let e = (r - t).abs();
        total += f64::from(e);
        worst = worst.max(e);
        n += 1;
    }
    ((total / f64::from(n.max(1))) as f32, worst)
}

/// What the screen-space pass costs against the truth, and what each knob buys.
///
/// Read in this order. The first table is the headline — how far the shipped
/// configuration sits from the integral, with the reference's own convergence
/// beside it so a reader can tell the estimator's error from the sampler's. The
/// sweeps under it are what the defaults in `cvars.rs` are read off, and every
/// one of them is a plateau rather than a minimum: a knob whose right value is a
/// range is not a knob one run can settle.
fn grade() -> Result<()> {
    cvars::AMBIENT.set_float(AMBIENT);
    // The dither is a deliberate plus-or-minus one code value and this reads
    // single pixels; the ratio would carry it twice.
    cvars::DITHER.set_float(0.0);
    let world = graded_world()?;
    let mut renderer = OffscreenRenderer::new(FRAME)?;

    println!(
        "graded scene: {} boxes, {}x{}, ambient {AMBIENT}, no sky and no lights\n",
        graded_scene().len(),
        FRAME.0,
        FRAME.1
    );

    let radius = cvars::AO_RADIUS.float() as f32;
    let (reference, covered) = reference_occlusion(radius);
    let rendered = rendered_occlusion(&mut renderer, &world)?;
    let (mean, worst) = error(&rendered, &reference, &covered);
    let coverage = covered.iter().filter(|c| **c).count();
    println!(
        "shipped defaults, radius {radius}: mean error {mean:.4}, worst {worst:.4} over {coverage} \
         covered pixels"
    );
    println!(
        "  for scale, the reference's own mean occlusion here is {:.4}\n",
        mean_of(&reference, &covered)
    );
    buckets(&rendered, &reference, &covered);

    projections(&mut renderer, &world)?;
    bias(&mut renderer, &world, &reference, &covered)?;
    slices_and_steps(&mut renderer, &world, &reference, &covered)?;
    pattern(&mut renderer, &world, &covered)?;
    direction(&mut renderer, &world, &covered)?;
    blur(&mut renderer, &world, &reference, &covered)?;
    cost(&mut renderer, &world, &reference, &covered)?;
    radii(&mut renderer, &world)?;

    cvars::DITHER.set_float(1.0);
    let report = renderer.shutdown();
    anyhow::ensure!(report.clean(), "unclean render: {report:?}");
    Ok(())
}

fn mean_of(values: &[f32], covered: &[bool]) -> f32 {
    let (mut total, mut n) = (0.0f64, 0u32);
    for (v, on) in values.iter().zip(covered) {
        if *on {
            total += f64::from(*v);
            n += 1;
        }
    }
    (total / f64::from(n.max(1))) as f32
}

/// Error against slice and step count — the sampling budget, in the two axes it
/// is spent on.
fn slices_and_steps(
    renderer: &mut OffscreenRenderer,
    world: &World,
    reference: &[f32],
    covered: &[bool],
) -> Result<()> {
    println!("  slices | steps |   mean |  worst");
    for slices in [1i64, 2, 3, 4] {
        for steps in [4i64, 8, 16] {
            cvars::AO_SLICES.set_int(slices);
            cvars::AO_STEPS.set_int(steps);
            let rendered = rendered_occlusion(renderer, world)?;
            let (mean, worst) = error(&rendered, reference, covered);
            println!("  {slices:>6} | {steps:>5} | {mean:>6.4} | {worst:>6.4}");
        }
    }
    cvars::AO_SLICES.set_int(2);
    cvars::AO_STEPS.set_int(8);
    println!();
    Ok(())
}

/// Lags in pixels, powers of two so a dip names a period directly.
const LAGS: [usize; 4] = [1, 2, 4, 8];

/// Mean absolute difference between covered pixels `lag` apart along one axis.
///
/// Both operands have to be covered: a difference taken against the background
/// is a silhouette, which is the scene and not the pattern.
fn lag_difference(field: &[f32], covered: &[bool], lag: usize, along_x: bool) -> f32 {
    let (w, h) = (FRAME.0 as usize, FRAME.1 as usize);
    let (mut total, mut n) = (0.0f64, 0u32);
    for y in 0..h {
        for x in 0..w {
            let (bx, by) = if along_x { (x + lag, y) } else { (x, y + lag) };
            if bx >= w || by >= h {
                continue;
            }
            let (a, b) = (y * w + x, by * w + bx);
            if !covered[a] || !covered[b] {
                continue;
            }
            total += f64::from((field[a] - field[b]).abs());
            n += 1;
        }
    }
    (total / f64::from(n.max(1))) as f32
}

/// The smallest lag whose difference falls *below* a nearer one's, if any.
fn period(d: &[f32; LAGS.len()]) -> Option<usize> {
    (1..LAGS.len()).find(|i| d[*i] < d[i - 1]).map(|i| LAGS[i])
}

/// The four residue classes of one index family, and how far apart their means
/// are — a spread in code values (§6 M66).
///
/// [`lag_difference`] walks x and y, so it can say a period is four pixels and
/// cannot say **along what**. That gap is not academic: `ao.slang` keys its
/// rotation on `(x + y) & 3`, which is *constant along every anti-diagonal*, so
/// every pixel on one of those lines estimates visibility from the same pair of
/// slice directions and carries the same directional bias. Four such lines repeat
/// every four pixels — which is what the x and y lags report, from the one
/// direction in which the structure is weakest.
fn class_spread(field: &[f32], covered: &[bool], key: fn(usize, usize) -> usize) -> f32 {
    let (w, h) = (FRAME.0 as usize, FRAME.1 as usize);
    let mut total = [0.0f64; 4];
    let mut n = [0u32; 4];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !covered[i] {
                continue;
            }
            let k = key(x, y);
            total[k] += f64::from(field[i]);
            n[k] += 1;
        }
    }
    let means: Vec<f64> = (0..4).map(|k| total[k] / f64::from(n[k].max(1))).collect();
    let (lo, hi) = means
        .iter()
        .fold((f64::MAX, f64::MIN), |(lo, hi), m| (lo.min(*m), hi.max(*m)));
    (hi - lo) as f32
}

/// Index families, chosen so the one the shader keys on has a control beside it.
/// The anti-diagonal is the rotation's own index and the main diagonal is its
/// mirror, which is the comparison that means anything — a spread large on both
/// is the scene's own gradient and not a tile.
#[expect(clippy::type_complexity, reason = "a labelled table, read once below")]
const FAMILIES: [(&str, fn(usize, usize) -> usize); 4] = [
    ("x+y", |x, y| (x + y) & 3),
    ("x-y", |x, y| x.wrapping_sub(y) & 3),
    ("x", |x, _| x & 3),
    ("y", |_, y| y & 3),
];

/// Which *direction* the pattern runs in (§6 M66).
///
/// [`pattern`]'s table says a period exists and how long it is; this says along
/// what, and the two together are what let §6 M66 rule the occlusion pass out as
/// the cause of a complaint about "very liney" shading under demo 12's shelter
/// roof. The raw target is organized on the anti-diagonal by an order of
/// magnitude over anything else, exactly as the rotation index is — and the 4x4
/// box takes that to a hundredth of a code value, which is why the answer was
/// that the lines were the *irradiance field's* and not this pass's.
fn direction(renderer: &mut OffscreenRenderer, world: &World, covered: &[bool]) -> Result<()> {
    println!("  slices | blur | spread of the four residue-class means, code values");
    println!("         |      | {}", {
        let mut header = String::new();
        for (label, _) in FAMILIES {
            header.push_str(&format!("{label:>8} "));
        }
        header
    });
    for (slices, blur) in [(2i64, false), (4, false), (2, true)] {
        cvars::AO_SLICES.set_int(slices);
        cvars::AO_BLUR.set_bool(blur);
        let rendered = rendered_occlusion(renderer, world)?;
        let mut row = String::new();
        for (_, key) in FAMILIES {
            row.push_str(&format!(
                "{:>8.3} ",
                class_spread(&rendered, covered, key) * 255.0
            ));
        }
        println!("  {slices:>6} | {:>4} | {row}", u32::from(blur));
    }
    cvars::AO_SLICES.set_int(2);
    cvars::AO_BLUR.set_bool(true);
    println!();
    Ok(())
}

/// Does the pass leave a *pattern* rather than noise (§6 M62)?
///
/// The one table here that needs no reference renderer, and the reason it exists
/// is that the ones that do are blind to exactly this: mean absolute error is a
/// scalar over a whole frame, so occlusion that is wrong in a repeating grid and
/// occlusion that is wrong at random score the same — and only one of them is
/// something a player sees. It is `gg-tools shadow-edge`'s jag argument in a
/// second place: a quality no ground truth can express, measured against what
/// the geometry alone says the answer must look like.
///
/// A field with no periodic structure decorrelates monotonically, so the mean
/// difference at lag 1, 2, 4, 8 rises. **A dip is a period**, and there is no
/// second reading of it. Four is the design — the rotation is a 4x4 tile and the
/// denoiser's window is matched to it — so `4 px` is the pass working and
/// anything *shorter* is phases landing on top of each other. A row reading `-`
/// with a lag-1 difference near zero is the other failure and is why the
/// amplitudes are printed beside the verdict: no period because no variety.
///
/// Both of those were live before §6 M62, and the slice sweep is what separates
/// them: at two slices x read a **2 px** period, at four it read `-` with a
/// lag-1 difference of 0.2 code values — the rotation inert — and only at one
/// and three did it do what the comment in `ao.slang` claimed. The accuracy
/// tables above are blind to all of it and moved by 0.001 when it was fixed.
///
/// Blur off, because the denoiser's whole job is to average one period away and
/// a table measured through it would report on the filter rather than on the
/// pass. The shipped configuration is printed through it at the foot, which is
/// what says whether the filter is in fact doing that.
fn pattern(renderer: &mut OffscreenRenderer, world: &World, covered: &[bool]) -> Result<()> {
    println!("  slices | blur | axis | lag 1 | lag 2 | lag 4 | lag 8 | period");
    cvars::AO_BLUR.set_bool(false);
    for (slices, blur) in [(1i64, false), (2, false), (3, false), (4, false), (2, true)] {
        cvars::AO_SLICES.set_int(slices);
        cvars::AO_BLUR.set_bool(blur);
        let rendered = rendered_occlusion(renderer, world)?;
        for along_x in [true, false] {
            let mut d = [0.0f32; LAGS.len()];
            for (slot, lag) in d.iter_mut().zip(LAGS) {
                *slot = lag_difference(&rendered, covered, lag, along_x);
            }
            println!(
                "  {slices:>6} | {:>4} | {:>4} | {:>5.1} | {:>5.1} | {:>5.1} | {:>5.1} | {}",
                u32::from(blur),
                if along_x { "x" } else { "y" },
                // Code values of 255, which is what an eye is looking at — the
                // field is a ratio in 0..1 and four decimal places of it read as
                // noise about noise.
                d[0] * 255.0,
                d[1] * 255.0,
                d[2] * 255.0,
                d[3] * 255.0,
                match period(&d) {
                    Some(p) => format!("{p} px"),
                    None => "-".to_owned(),
                }
            );
        }
    }
    cvars::AO_SLICES.set_int(2);
    cvars::AO_BLUR.set_bool(true);
    println!();
    Ok(())
}

/// The same scene under both projections.
///
/// Here because an orthographic eye has **no eye point**, so the direction a
/// horizon is measured against is the camera's axis and not the direction to the
/// fragment — and no blessed reference can say whether that is done right. The
/// suite's one orthographic scene is demo 11's flat framing, whose visible faces
/// are a single coplanar sheet: it has almost no screen-space occlusion to show
/// and moves by a code value whichever vector the pass uses. This row is the
/// only place the two spellings are ever compared.
fn projections(renderer: &mut OffscreenRenderer, world: &World) -> Result<()> {
    println!("  projection |   mean |  worst | reference mean | rendered mean");
    for ortho in [false, true] {
        let (reference, covered) = reference_occlusion_at(cvars::AO_RADIUS.float() as f32, ortho);
        let rendered = rendered_occlusion_at(renderer, world, ortho)?;
        let (mean, worst) = error(&rendered, &reference, &covered);
        println!(
            "  {:>10} | {mean:>6.4} | {worst:>6.4} | {:>14.4} | {:>13.4}",
            if ortho { "ortho" } else { "perspective" },
            mean_of(&reference, &covered),
            mean_of(&rendered, &covered),
        );
    }
    println!();
    Ok(())
}

/// The bias sweep — the plateau `r.ao_bias` is read off.
///
/// Two columns that fail in opposite directions, which is the only honest way to
/// report a knob like this (`gg-tools shadow-bias`'s shape, §6 M11): **invented**
/// is what a nearly-open surface loses to self-occlusion and goes wrong as the
/// bias falls, **missed** is the crease occlusion stepped over and goes wrong as
/// it rises. The right value is the range where neither is bad, not the row with
/// the lowest total.
fn bias(
    renderer: &mut OffscreenRenderer,
    world: &World,
    reference: &[f32],
    covered: &[bool],
) -> Result<()> {
    println!("  r.ao_bias | invented | missed |   mean |  worst");
    for pixels in [0.0f64, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] {
        cvars::AO_BIAS.set_float(pixels);
        let rendered = rendered_occlusion(renderer, world)?;
        let (mean, worst) = error(&rendered, reference, covered);
        let (invented, missed) = sides(&rendered, reference, covered);
        println!("  {pixels:>9.1} | {invented:>8.4} | {missed:>6.4} | {mean:>6.4} | {worst:>6.4}");
    }
    cvars::AO_BIAS.set_float(2.0);
    println!();
    Ok(())
}

/// The two failures, each as a mean over the pixels it can happen on: occlusion
/// invented on surfaces the truth calls open, and occlusion missed on surfaces
/// the truth calls occluded.
fn sides(rendered: &[f32], reference: &[f32], covered: &[bool]) -> (f32, f32) {
    let (mut invented, mut open_n) = (0.0f64, 0u32);
    let (mut missed, mut shut_n) = (0.0f64, 0u32);
    for ((r, t), on) in rendered.iter().zip(reference).zip(covered) {
        if !on {
            continue;
        }
        if *t >= 0.999 {
            invented += f64::from((t - r).max(0.0));
            open_n += 1;
        } else {
            missed += f64::from((r - t).max(0.0));
            shut_n += 1;
        }
    }
    (
        (invented / f64::from(open_n.max(1))) as f32,
        (missed / f64::from(shut_n.max(1))) as f32,
    )
}

/// What the denoiser is worth, at the budget it is there to make survivable.
fn blur(
    renderer: &mut OffscreenRenderer,
    world: &World,
    reference: &[f32],
    covered: &[bool],
) -> Result<()> {
    println!("  r.ao_blur |   mean |  worst");
    for on in [false, true] {
        cvars::AO_BLUR.set_bool(on);
        let rendered = rendered_occlusion(renderer, world)?;
        let (mean, worst) = error(&rendered, reference, covered);
        println!("  {:>9} | {mean:>6.4} | {worst:>6.4}", u32::from(on));
    }
    cvars::AO_BLUR.set_bool(true);
    println!();
    Ok(())
}

/// Error against radius, each row graded by the reference **at its own radius**.
///
/// That is the whole point of the row: a wider radius asks the depth buffer
/// about geometry further outside the frame and behind nearer surfaces, so the
/// estimator's structural blindness grows with it. What the sweep prints is the
/// distance between an estimate and the quantity it is estimating, not the
/// distance between two different quantities.
fn radii(renderer: &mut OffscreenRenderer, world: &World) -> Result<()> {
    println!("  radius |   mean |  worst | reference mean occlusion");
    for radius in RADII {
        cvars::AO_RADIUS.set_float(f64::from(radius));
        let (reference, covered) = reference_occlusion(radius);
        let rendered = rendered_occlusion(renderer, world)?;
        let (mean, worst) = error(&rendered, &reference, &covered);
        println!(
            "  {radius:>6.2} | {mean:>6.4} | {worst:>6.4} | {:>6.4}",
            mean_of(&reference, &covered)
        );
    }
    cvars::AO_RADIUS.set_float(0.5);
    println!();
    Ok(())
}

/// Warm-up frames discarded, and frames measured, for the cost table.
const WARMUP: usize = 3;
const FRAMES: usize = 15;

/// Median frame time, in milliseconds.
///
/// Wall clock around a whole `frame` call, which is the shape `lamps` uses and
/// the honest one here: the occlusion passes are two fullscreen triangles, so
/// what they cost is fragment work on whatever device this runs on and not a
/// number a CPU-side timer could miss.
fn median_frame(renderer: &mut OffscreenRenderer, world: &World) -> Result<f64> {
    for _ in 0..WARMUP {
        render_scene(renderer, world)?;
    }
    let mut times = Vec::with_capacity(FRAMES);
    for _ in 0..FRAMES {
        let started = std::time::Instant::now();
        render_scene(renderer, world)?;
        times.push(started.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    Ok(times[times.len() / 2])
}

/// What the term costs, beside what it buys.
///
/// `cull`'s lesson applies unchanged: the frame column is the honest half. This
/// is a 480x270 frame of six boxes on a software rasterizer, so the absolute
/// numbers are the pin's and not a budget — what transfers is the *shape*, which
/// is that the cost is per-pixel and scales with `slices * steps` while the
/// accuracy stops improving after three slices.
fn cost(
    renderer: &mut OffscreenRenderer,
    world: &World,
    reference: &[f32],
    covered: &[bool],
) -> Result<()> {
    cvars::AO.set_bool(false);
    let base = median_frame(renderer, world)?;
    cvars::AO.set_bool(true);
    println!("  slices | steps |   mean |  frame ms | over r.ao 0 ({base:.2} ms)");
    for (slices, steps) in [(1i64, 8i64), (2, 8), (3, 8), (4, 8), (2, 16), (3, 16)] {
        cvars::AO_SLICES.set_int(slices);
        cvars::AO_STEPS.set_int(steps);
        let rendered = rendered_occlusion(renderer, world)?;
        let (mean, _) = error(&rendered, reference, covered);
        let ms = median_frame(renderer, world)?;
        println!(
            "  {slices:>6} | {steps:>5} | {mean:>6.4} | {ms:>9.2} | {:>+6.2} ms",
            ms - base
        );
    }
    // Back to the shipped pair. Three slices is measurably better and is not the
    // default: on the pin it is +3.93 ms against two slices' +3.07, a 28 % wider
    // pass, and on the 4090 the difference between them is inside what one
    // binary's frame time moves between runs — `cull`'s lesson, unchanged. What
    // it buys is 17 % of the error (0.0299 → 0.0247), and that 17 % comes off
    // the *smaller* half: the structural blindness in the crease rows above is
    // 0.13 and no slice count touches it. Two is what CI's rasterizer pays for;
    // a player on real hardware can raise it for nothing.
    cvars::AO_SLICES.set_int(2);
    cvars::AO_STEPS.set_int(8);
    println!();
    Ok(())
}
