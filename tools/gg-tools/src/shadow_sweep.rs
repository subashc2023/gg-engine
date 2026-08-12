//! `gg-tools shadow-sweep` — what a turning camera does to the shadows in a room
//! standing still around it.
//!
//! The complaint this answers: sweeping the view makes shadows appear and
//! disappear, "as if a rectangle of non-shadow eats the shadow and moves with my
//! view". Nothing about a sun, a room or a floor depends on where the camera
//! points, so **a patch of floor that is shadowed at one yaw and lit at another
//! is a defect**, and no reference renderer is needed to say so.
//!
//! So the measurement is per *world point*, not per pixel. A grid is laid over
//! the floor once; the camera stands at five spots in demo 12's room at demo
//! 12's eye height and turns through a whole circle at each; at every step the
//! grid points that are on screen and unoccluded are projected back into the
//! frame and classified lit or shadowed. A point two views disagreed about is one
//! count of the defect, and `shadow-sweep-disagreement.png` is where they are —
//! a plan view of the room, so a hole with a shape reads as a shape.
//!
//! The second table is *why*, by elimination — the same frame rendered four
//! ways at the views that disagreed most:
//!
//! - **shipping** — the shell's own path (`gg_runtime::App::extract`): the view
//!   frustum swept up-light by the shadow maps' reach, so a caster off screen
//!   still casts into what is on it.
//! - **view only** — the frustum alone, which is what the shell did before the
//!   post-M21 fix. The gap against `shipping` is what that fix bought, and it is
//!   the one number here that is allowed to be zero only if the cull never bit.
//! - **unculled** — [`Frustum::UNBOUNDED`], so the only thing that removes a
//!   caster is the per-cascade slab test (`gg_render::casts_into`). The gap
//!   against `shipping` is what the *swept* frustum still costs.
//! - **one slab** — unculled, `r.shadow_cascades 1`: one 89 m slab that swallows
//!   the room whole, so no cascade edge is near anything. The gap against
//!   `unculled` is the cascades' own, cull and texel density together.

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::{Extracted, Frustum};
use gg_math::{render, sim};
use gg_render::{OffscreenRenderer, View, cvars};

use crate::shadow_image::{DISAGREEMENT, luminance};

/// 16:9, and small enough that the sweep is seconds. The cascade fit reads the
/// aspect, so the shape is part of the measurement even though the pixels are
/// only how a world point gets classified.
const EXTENT: (u32, u32) = (960, 540);

/// Demo 12's sun and eye height, verbatim — this reproduces that room's
/// complaint, not a synthetic one.
const SUN: [f32; 3] = [-0.45, -1.0, -0.30];
const EYE_Y: f64 = 1.62;

/// Yaw steps through a full turn. 24 is 15 degrees apart, finer than the ~30 a
/// cascade footprint takes to cross a pillar.
const STEPS: u32 = 24;

/// Pitches swept at each yaw, in degrees, negative down.
///
/// Not a refinement: a cascade's centre is `view.rotation() * (0, 0, -depth)`,
/// so pitch moves every slab *vertically* while yaw only swings it around. A
/// sweep at one pitch leaves half the fit's freedom untouched, and "sweeping my
/// view over a place" is both axes at once.
///
/// `-70` is not symmetry: a caster leaves the picture *upward* long before its
/// shadow does, so looking steeply down at your own feet is the framing the
/// complaint describes and the one a gentle pitch sweep never reaches.
const PITCHES: &[f32] = &[-70.0, -40.0, -20.0, 0.0, 15.0];

/// The floor grid, in metres: half-width, and the spacing between samples. The
/// room's inner faces are at ±12, so 11.5 keeps every point clear of the walls.
const REACH: f64 = 11.5;
const SPACING: f64 = 0.25;

/// Under the lit floor's upper quartile by this much and the point is in shadow.
/// Two disagreements, so a PCF tap's worth of penumbra does not count as one.
const SHADED: i32 = 2 * DISAGREEMENT;

/// Where the camera stands. A cascade is fitted to the camera's *position* as
/// well as its heading, so a sweep from one spot leaves that half of the fit
/// unexercised — the middle of the room, in among the crates, out by the floating
/// slabs, and two more added with the caster cull in mind: hard against a pillar
/// and up on the mezzanine, where the thing casting is nearest to leaving frame.
const EYES: &[[f64; 3]] = &[
    [0.0, EYE_Y, 0.0],
    [-4.0, EYE_Y, 4.0],
    [5.0, EYE_Y, -4.0],
    [2.0, EYE_Y, 4.2],
    [-7.0, EYE_Y + 1.8, -2.0],
];

/// One heading. A pair rather than two arguments so no call site can hand the
/// projection one pitch and the sampler another — the whole measurement is that
/// a world point is looked up in the frame it was drawn in.
#[derive(Clone, Copy)]
struct Look {
    yaw: f32,
    pitch: f32,
}

impl Look {
    fn view(self) -> View {
        View {
            yaw: self.yaw,
            pitch: self.pitch,
            ..View::default()
        }
    }
}

pub fn run(args: &[String]) -> anyhow::Result<()> {
    // `--msaa N` because demo 12 *opens* at 4x (§6 M21's `Prefs::aa`), and the
    // scene pass is where a sample count lands — so a sweep at 1x would be
    // measuring a renderer the complaint was not made against.
    let samples = match args {
        [] => 1,
        [flag, count] if flag == "--msaa" => count.parse()?,
        [arg, ..] => anyhow::bail!("unknown flag {arg:?} — shadow-sweep takes `--msaa N`"),
    };
    cvars::MSAA.set_int(samples);
    let mut renderer = OffscreenRenderer::new(EXTENT)?;
    println!(
        "gg-tools shadow-sweep: {}x{} on {}",
        EXTENT.0,
        EXTENT.1,
        renderer.device().chosen
    );
    println!(
        "  demo 12's room and sun, eye {EYE_Y} m up at the origin, turning through {STEPS} yaws"
    );
    println!(
        "  {}x samples, {} cascades over {} m",
        renderer.samples().count(),
        cvars::SHADOW_CASCADES.int(),
        cvars::SHADOW_DISTANCE.float()
    );
    println!("  the sun does not move, so no patch of floor may change its mind");
    println!();

    let world = room()?;
    let grid = grid();
    // One row per view: what luminance each grid point showed, or `None` where
    // that view could not see it. Kept whole rather than reduced as it goes,
    // because the *reference* a point is judged against is the brightest it was
    // ever seen — which is not known until every view has been rendered.
    //
    // A per-point reference and not a per-frame one, and that correction is the
    // instrument's own history: the first version took the lit level from the
    // lower two fifths of each frame, which is floor at pitch 0 and is wall,
    // ceiling and background the moment the camera looks up. It reported 20% of
    // the room unstable, every one of the worst views at a positive pitch, and
    // all of it was the baseline collapsing rather than a shadow moving.
    let mut rows: Vec<(sim::DVec3, Look, Vec<Option<i32>>)> = Vec::new();

    ship();
    for eye in EYES {
        let eye = sim::DVec3::new(eye[0], eye[1], eye[2]);
        let reachable: Vec<bool> = grid.iter().map(|p| visible(eye, *p)).collect();
        for step in 0..STEPS {
            let yaw = step as f32 * std::f32::consts::TAU / STEPS as f32;
            for &degrees in PITCHES {
                let look = Look {
                    yaw,
                    pitch: degrees.to_radians(),
                };
                let lum = luminance(&frame_of(&mut renderer, &world, eye, look, Cull::Shipping)?);
                let row = grid
                    .iter()
                    .enumerate()
                    .map(|(i, point)| sample(eye, *point, look, &lum).filter(|_| reachable[i]))
                    .collect();
                rows.push((eye, look, row));
            }
        }
    }

    // The brightest each point was ever rendered. A point every view agreed was
    // shadowed has its own shadowed level as its reference and reads as stable,
    // which is the right answer: nothing about it changed.
    let reference: Vec<i32> = (0..grid.len())
        .map(|i| rows.iter().filter_map(|r| r.2[i]).max().unwrap_or(0))
        .collect();
    let shadowed = |value: Option<i32>, i: usize| value.map(|v| v < reference[i] - SHADED);

    let mut seen = vec![0u32; grid.len()];
    let mut dark = vec![0u32; grid.len()];
    for (_, _, row) in &rows {
        for i in 0..grid.len() {
            if let Some(is_dark) = shadowed(row[i], i) {
                seen[i] += 1;
                dark[i] += u32::from(is_dark);
            }
        }
    }

    let judged: Vec<usize> = (0..grid.len()).filter(|&i| seen[i] >= 2).collect();
    let split: Vec<usize> = judged
        .iter()
        .copied()
        .filter(|&i| dark[i] > 0 && dark[i] < seen[i])
        .collect();
    println!(
        "  {} of {} floor points were seen from two or more views; {} of them ({:.1}%) changed \
         their answer",
        judged.len(),
        grid.len(),
        split.len(),
        100.0 * split.len() as f64 / judged.len().max(1) as f64,
    );

    // Which views are responsible: a view is charged for a point when it is in
    // the minority about it. That is what says *where to look*, and the leg table
    // below is what says why.
    let mut blame: Vec<(sim::DVec3, Look, usize)> = Vec::new();
    for (eye, look, row) in &rows {
        let wrong = split
            .iter()
            .filter(|&&i| shadowed(row[i], i) == Some(dark[i] * 2 <= seen[i]))
            .count();
        blame.push((*eye, *look, wrong));
    }
    blame.sort_by_key(|b| std::cmp::Reverse(b.2));
    plan_view(&grid, &seen, &dark)?;

    println!();
    println!("  the same frame four ways, at the views that disagreed most:");
    println!(
        "       eye       yaw pitch  split | shipping view-only unculled one slab |  sweep \
         frustum  fit"
    );
    println!(
        "    -------------------------------+------------------------------------+---------------\
         ------"
    );
    for &(eye, look, wrong) in blame.iter().take(6) {
        ship();
        let mut leg = |cull| shaded(&mut renderer, &world, eye, look, cull, &grid, &reference);
        let shipping = leg(Cull::Shipping)?;
        let view_only = leg(Cull::View)?;
        let unculled = leg(Cull::Off)?;
        cvars::SHADOW_CASCADES.set_int(1);
        let one = shaded(
            &mut renderer,
            &world,
            eye,
            look,
            Cull::Off,
            &grid,
            &reference,
        )?;
        ship();
        println!(
            "    ({:>5.1},{:>5.1}) {:>5} {:>5} {:>6} | {shipping:>8} {view_only:>9} \
             {unculled:>8} {one:>7} | {:>6} {:>7} {:>4}",
            eye.x,
            eye.z,
            look.yaw.to_degrees() as i32,
            look.pitch.to_degrees() as i32,
            wrong,
            shipping.saturating_sub(view_only),
            unculled.saturating_sub(shipping),
            one.saturating_sub(unculled),
        );
    }
    if let Some(&(eye, look, _)) = blame.first() {
        dump(&mut renderer, &world, eye, look)?;
    }
    println!();
    println!("  wrote {}", crate::output_dir()?.display());

    let report = renderer.shutdown();
    anyhow::ensure!(
        report.clean(),
        "unclean render: {} validation message(s), {} leak(s)",
        report.validation_messages,
        report.leaked_allocations.len(),
    );
    Ok(())
}

/// The shipping cascade knobs, restated per leg so no row inherits the last.
fn ship() {
    cvars::SHADOW_SIZE.set_int(2048);
    cvars::SHADOW_DISTANCE.set_float(80.0);
    cvars::SHADOW_CASCADES.set_int(4);
    cvars::SHADOW_SPLIT_LAMBDA.set_float(0.85);
    cvars::SHADOW_NORMAL_BIAS.set_float(0.5);
    cvars::SHADOW_DEPTH_BIAS.set_float(0.75);
}

/// The floor sample grid, world space, a hair above the floor's top face.
fn grid() -> Vec<sim::DVec3> {
    let steps = (2.0 * REACH / SPACING) as i32;
    let mut out = Vec::new();
    for iz in 0..=steps {
        for ix in 0..=steps {
            out.push(sim::DVec3::new(
                -REACH + f64::from(ix) * SPACING,
                0.02,
                -REACH + f64::from(iz) * SPACING,
            ));
        }
    }
    out
}

/// Where one world point lands in the frame at `yaw`, or `None` when it is off
/// screen. Occlusion is [`visible`]'s, answered once per eye.
///
/// Camera-relative from the start: the eye is the origin of the render space
/// (§1.4), so the projection this rebuilds is the one the frame was drawn with.
fn sample(eye: sim::DVec3, point: sim::DVec3, look: Look, lum: &[i32]) -> Option<i32> {
    let view = look.view();
    let relative = render::Vec3::new(
        (point.x - eye.x) as f32,
        (point.y - eye.y) as f32,
        (point.z - eye.z) as f32,
    );
    let clip = view.view_projection(EXTENT) * relative.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = render::Vec3::new(clip.x, clip.y, clip.z) / clip.w;
    // Inset by a few pixels: the frame's own edge is where a cascade's border and
    // the frame's border coincide, and a point half off screen is not a
    // measurement of either.
    if ndc.x.abs() > 0.97 || ndc.y.abs() > 0.97 {
        return None;
    }
    let col = ((ndc.x * 0.5 + 0.5) * EXTENT.0 as f32) as usize;
    let row = ((ndc.y * 0.5 + 0.5) * EXTENT.1 as f32) as usize;
    lum.get(row.min(EXTENT.1 as usize - 1) * EXTENT.0 as usize + col.min(EXTENT.0 as usize - 1))
        .copied()
}

/// A plan view of the room: grey where every yaw agreed the floor was lit, black
/// where every yaw agreed it was shadowed, red where they did not.
///
/// North is -z and the eye is the centre pixel, so this is the room as a map —
/// which is the only projection in which "the hole is a rectangle that slides
/// with the view" is a statement a picture can settle.
fn plan_view(grid: &[sim::DVec3], seen: &[u32], dark: &[u32]) -> anyhow::Result<()> {
    // Nearest-neighbour to 8 px a sample: at one pixel per 250 mm the whole room
    // is 93 px and the difference between a fringe and a region — which is the
    // only question this picture is asked — is below the resolution it is drawn
    // at. No filtering, so a single unstable sample stays a single square.
    const SCALE: usize = 8;
    let cells = (2.0 * REACH / SPACING) as usize + 1;
    let mut cell = Vec::with_capacity(grid.len() * 4);
    for i in 0..grid.len() {
        let rgba = match (seen[i], dark[i]) {
            (0 | 1, _) => [24, 24, 40, 255],
            (_, 0) => [170, 165, 150, 255],
            (s, d) if d == s => [40, 38, 34, 255],
            (s, d) => {
                // Red by how split it was: a point half the yaws disagreed about
                // is the brightest, since that is the strongest evidence.
                let strength = 255 - (255 * (2 * d.abs_diff(s / 2)) / s.max(1)).min(200) as u8;
                [strength.max(80), 30, 30, 255]
            }
        };
        cell.extend_from_slice(&rgba);
    }
    let edge = (cells * SCALE) as u32;
    let mut pixels = Vec::with_capacity(cell.len() * SCALE * SCALE);
    for row in 0..cells * SCALE {
        for col in 0..cells * SCALE {
            let i = ((row / SCALE) * cells + col / SCALE) * 4;
            pixels.extend_from_slice(&cell[i..i + 4]);
        }
    }
    let path = crate::output_dir()?.join("shadow-sweep-disagreement.png");
    let file = std::fs::File::create(&path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), edge, edge);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&pixels)?;
    Ok(())
}

/// Demo 12's room as a world, plus its sun.
fn room() -> anyhow::Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Light>()?;
    for (position, half, color) in SHAPES {
        let e = world.spawn();
        world.insert(
            e,
            Renderable::boxed(
                sim::DVec3::new(position[0], position[1], position[2]),
                sim::Vec3::new(half[0], half[1], half[2]),
                *color,
            ),
        )?;
    }
    let sun = world.spawn();
    world.insert(
        sun,
        Light::sun(sim::Vec3::new(SUN[0], SUN[1], SUN[2]), 0x00ff_f4e0, 3.4),
    )?;
    Ok(world)
}

/// Whether the segment from `eye` to `point` reaches it without crossing a box.
///
/// An exact slab test against every shape rather than a test against the few
/// that look like occluders: with the mezzanine and the stairs in, "what can
/// stand between the eye and the floor" is most of the room, and a grid point
/// *under* the mezzanine is excluded by the same arithmetic that excludes one
/// behind a pillar. Independent of yaw, so it is answered once per eye.
fn visible(eye: sim::DVec3, point: sim::DVec3) -> bool {
    let d = [point.x - eye.x, point.y - eye.y, point.z - eye.z];
    for (centre, half, _) in SHAPES {
        let (mut enter, mut exit) = (0.0f64, 1.0f64);
        let mut missed = false;
        for axis in 0..3 {
            let origin = [eye.x, eye.y, eye.z][axis];
            let (lo, hi) = (
                centre[axis] - f64::from(half[axis]),
                centre[axis] + f64::from(half[axis]),
            );
            if d[axis].abs() < 1e-12 {
                // Parallel to this pair of planes: a miss here is a miss for the
                // whole box, and no interval to intersect.
                missed |= origin < lo || origin > hi;
                continue;
            }
            let (mut t0, mut t1) = ((lo - origin) / d[axis], (hi - origin) / d[axis]);
            if t0 > t1 {
                core::mem::swap(&mut t0, &mut t1);
            }
            enter = enter.max(t0);
            exit = exit.min(t1);
        }
        // A hair off each end: the point sits 20 mm over the floor and the ray
        // starts inside nothing, so a grazing touch at either end is contact
        // rather than occlusion.
        if !missed && enter <= exit && exit > 1e-4 && enter < 0.999 {
            return false;
        }
    }
    true
}

/// Demo 12's `ROOM`, verbatim — the whole table and not a trim of it.
///
/// The first attempt at this instrument kept the floor, the walls and the four
/// pillars, and measured a room whose shadows were nearly stable: with a sun this
/// steep a pillar's shadow lies within a metre of its own base, so a caster
/// leaving the frustum takes its shadow with it. The mezzanine, the stairs and
/// the floating slabs are the casters that put shadow *metres* away from
/// themselves, which is the arrangement the complaint is about.
const SHAPES: &[([f64; 3], [f32; 3], u32)] = &[
    ([0.0, -0.25, 0.0], [12.5, 0.25, 12.5], 0x00b4_aea2),
    ([0.0, 2.0, -12.25], [12.5, 2.0, 0.25], 0x008f_9298),
    ([0.0, 2.0, 12.25], [12.5, 2.0, 0.25], 0x008f_9298),
    ([-12.25, 2.0, 0.0], [0.25, 2.0, 12.5], 0x008f_9298),
    ([12.25, 2.0, 0.0], [0.25, 2.0, 12.5], 0x008f_9298),
    ([-7.0, 0.15, 3.0], [2.5, 0.15, 0.45], 0x00a0_8c74),
    ([-7.0, 0.30, 2.1], [2.5, 0.30, 0.45], 0x00a0_8c74),
    ([-7.0, 0.45, 1.2], [2.5, 0.45, 0.45], 0x00a0_8c74),
    ([-7.0, 0.60, 0.3], [2.5, 0.60, 0.45], 0x00a0_8c74),
    ([-7.0, 0.75, -0.6], [2.5, 0.75, 0.45], 0x00a0_8c74),
    ([-7.0, 0.90, -1.5], [2.5, 0.90, 0.45], 0x00a0_8c74),
    ([-7.25, 0.90, -6.95], [4.75, 0.90, 5.05], 0x0096_846c),
    ([2.0, 2.0, 6.0], [0.4, 2.0, 0.4], 0x00c8_c4bc),
    ([2.0, 2.0, -6.0], [0.4, 2.0, 0.4], 0x00c8_c4bc),
    ([9.0, 2.0, 6.0], [0.4, 2.0, 0.4], 0x00c8_c4bc),
    ([9.0, 2.0, -6.0], [0.4, 2.0, 0.4], 0x00c8_c4bc),
    ([4.0, 1.0, 0.0], [1.2, 0.2, 1.2], 0x00d0_8840),
    ([7.5, 1.9, -3.0], [1.2, 0.2, 1.2], 0x00d0_8840),
    ([10.5, 2.8, -6.5], [1.2, 0.2, 1.2], 0x00d0_8840),
    ([0.0, 0.15, 4.5], [0.6, 0.15, 0.6], 0x0078_b4a8),
    ([-2.9, 0.15, 5.2], [0.6, 0.15, 0.6], 0x0078_b4a8),
    ([0.0, 0.35, 1.5], [0.5, 0.35, 0.5], 0x004f_9a8c),
    ([-4.2, 0.35, 5.2], [0.5, 0.35, 0.5], 0x004f_9a8c),
    ([-4.2, 1.05, 5.2], [0.5, 0.35, 0.5], 0x004f_9a8c),
];

/// Grid points this leg put in shadow, against the sweep's own per-point
/// reference — the same yardstick the disagreement count uses, so a leg number
/// and a split number are the same kind of thing.
fn shaded(
    renderer: &mut OffscreenRenderer,
    world: &World,
    eye: sim::DVec3,
    look: Look,
    cull: Cull,
    grid: &[sim::DVec3],
    reference: &[i32],
) -> anyhow::Result<usize> {
    let lum = luminance(&frame_of(renderer, world, eye, look, cull)?);
    Ok(grid
        .iter()
        .enumerate()
        .filter(|(i, point)| {
            visible(eye, **point)
                && sample(eye, **point, look, &lum).is_some_and(|v| v < reference[*i] - SHADED)
        })
        .count())
}

/// What extract was told to keep for one leg.
#[derive(Clone, Copy, PartialEq)]
enum Cull {
    /// The shell's own path: the frustum swept up-light by the maps' reach.
    Shipping,
    /// The frustum alone — the shell before the post-M21 fix.
    View,
    /// Nothing; only `gg_render::casts_into` removes a caster.
    Off,
}

/// One frame of the room at `look`, extract culling the way `cull` says.
///
/// Written as `clear` + `append_lights` + `append` rather than `transforms`
/// because that is the order the shell now runs in, and for its reason: the
/// sweep is aimed along the sun, so the light has to be in hand before the
/// instances are culled.
fn frame_of(
    renderer: &mut OffscreenRenderer,
    world: &World,
    eye: sim::DVec3,
    look: Look,
    cull: Cull,
) -> anyhow::Result<Vec<u8>> {
    let view = look.view();
    let mut extracted = Extracted::default();
    extracted.clear(
        eye,
        match cull {
            Cull::Off => Frustum::UNBOUNDED,
            _ => view.frustum(EXTENT),
        },
    );
    extracted.append_lights(world)?;
    if cull == Cull::Shipping {
        extracted.cast_shadows(view.caster_reach(EXTENT));
    }
    extracted.append::<Renderable>(world)?;
    let frame = renderer.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?;
    anyhow::ensure!(
        frame.order.iter().any(|name| name.starts_with("shadow")),
        "no shadow pass ran — there would be nothing to measure"
    );
    Ok(frame.pixels)
}

/// The legs at one yaw, written out: the frame, and a map of where the shipping
/// path is *lighter* than the leg it is being compared against.
///
/// A count says how much shadow went missing; only the picture says whether the
/// hole has a shape — and "a rectangle that slides with the view" is a claim
/// about shape, which a number can neither confirm nor refute.
fn dump(
    renderer: &mut OffscreenRenderer,
    world: &World,
    eye: sim::DVec3,
    look: Look,
) -> anyhow::Result<()> {
    ship();
    let shipping = frame_of(renderer, world, eye, look, Cull::Shipping)?;
    let view_only = frame_of(renderer, world, eye, look, Cull::View)?;
    let unculled = frame_of(renderer, world, eye, look, Cull::Off)?;
    cvars::SHADOW_CASCADES.set_int(1);
    let one = frame_of(renderer, world, eye, look, Cull::Off)?;
    ship();
    let degrees = format!(
        "{}y{}p",
        look.yaw.to_degrees() as i32,
        look.pitch.to_degrees() as i32
    );

    write_png(&shipping, &format!("{degrees}-shipping"))?;
    write_png(&one, &format!("{degrees}-one-slab"))?;
    // Red where `lighter` shows a pixel the `darker` leg had in shadow. The last
    // row runs the other way round on purpose: shadow the *fix* put back is the
    // one thing here that is a gain, and it belongs in a picture of its own
    // rather than as a negative number in the row above.
    for (lighter, darker, name) in [
        (&shipping, &unculled, "lost-vs-unculled"),
        (&shipping, &one, "lost-vs-one-slab"),
        (&view_only, &shipping, "recovered-by-the-sweep"),
    ] {
        let (a, b) = (luminance(lighter), luminance(darker));
        let mut out = lighter.clone();
        for i in 0..a.len() {
            let moved = a[i] - b[i] > DISAGREEMENT;
            out[i * 4] = if moved { 255 } else { out[i * 4] / 3 };
            out[i * 4 + 1] /= 3;
            out[i * 4 + 2] /= 3;
        }
        write_png(&out, &format!("{degrees}-{name}"))?;
    }
    Ok(())
}

fn write_png(pixels: &[u8], name: &str) -> anyhow::Result<()> {
    let path = crate::output_dir()?.join(format!("shadow-sweep-{name}.png"));
    let file = std::fs::File::create(&path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), EXTENT.0, EXTENT.1);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(pixels)?;
    Ok(())
}
