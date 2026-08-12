//! The shadow filter's two widths (§6 M23), proven offscreen where a test can
//! read the pixels.
//!
//! The kernel sizes its penumbra from the larger of two quantities, and they
//! answer to different masters, so they are asserted separately and they fail in
//! opposite directions:
//!
//! - **The screen floor.** A boundary is never narrower than `r.shadow_softness`
//!   pixels. This is the defect this milestone exists for: a shadow boundary is a
//!   shading discontinuity *inside* a triangle, so no MSAA count resolves it, and
//!   a kernel whose width is stated in shadow texels projects to whatever it
//!   happens to project to — which on the desk was **0.01 px**, a 140-level step
//!   in a single pixel with the silhouette of the very cube casting it coming out
//!   smooth alongside. The control is `r.shadow_softness 0`, which puts that
//!   frame back and is what stops the claim from passing on a framing that had no
//!   sharp edge to begin with.
//! - **The physical penumbra.** With the floor switched off, a shadow is wider
//!   where it falls further from what cast it. That is `r.sun_angle` doing its
//!   job, and it is what makes a floor filter a *soft shadow* rather than a blur:
//!   set the sun to a point and the same pair of casters must draw the same edge.
//!
//! `gg-tools shadow-edge` is where the widths were chosen; these are the claims a
//! gate can hold.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

/// Wide, because every number here is in pixels: a narrower frame makes each
/// pixel cover more world and would report the defect as smaller than it is.
const EXTENT: (u32, u32) = (512, 288);

/// High and pitched steeply down, so the floor is well resolved across a wide
/// range of distances instead of being crushed into ten rows under a horizon.
const PITCH: f32 = -1.0;
const EYE: [f64; 3] = [0.0, 25.0, 8.0];

/// Rows the shadow band crosses at this framing.
const BAND: std::ops::Range<u32> = 30..240;

/// Where every caster's shadow is made to land, on the floor.
///
/// Holding it fixed is what makes a height sweep a measurement of the **gap**
/// and of nothing else: raise the caster and move it back along the light by
/// exactly as much, and all that changed is how far the light travelled between
/// caster and receiver — not how far the camera is from the result. That
/// confound is real and it cost this test a round: a slab at 12 m first measured
/// *narrower* than one at 1.5 m, because its shadow had also moved five times
/// further away and the penumbra was being read in pixels.
const TARGET: [f64; 2] = [0.0, -12.0];

/// The shipping map, because texels-per-pixel is the axis the defect lives on
/// and a coarse map would hide it: at 2048 over four cascades a texel is a
/// fraction of a pixel, which is exactly the case a texel-width kernel loses.
const SHADOW_SIZE: i64 = 2048;

/// Off the world axes, so the boundary crosses the map's texel grid obliquely.
/// An axis-aligned sun lays the edge along the grid — the single framing where
/// no filter has anything to do.
const SUN: [f32; 3] = [0.4, -0.55, -1.0];

/// A floor, and a long slab held `height` metres over it.
fn world(height: f64) -> World {
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    world.register::<Light>().unwrap();
    let floor = world.spawn();
    world
        .insert(
            floor,
            Renderable::boxed(
                sim::DVec3::new(0.0, -0.1, 0.0),
                sim::Vec3::new(60.0, 0.1, 60.0),
                0x009a_9488,
            ),
        )
        .unwrap();
    // Back along the light by however far it falls, so the shadow lands on
    // `TARGET` whatever the height. Painted the floor's own colour: the caster
    // is in frame at the low end of the sweep, and a slab that shades like the
    // surface it stands over puts no step of its own into a column that `soft`
    // would then have to be taught to ignore.
    let drop = height / f64::from(-SUN[1]);
    let slab = world.spawn();
    world
        .insert(
            slab,
            Renderable::boxed(
                sim::DVec3::new(
                    TARGET[0] - f64::from(SUN[0]) * drop,
                    height,
                    TARGET[1] - f64::from(SUN[2]) * drop,
                ),
                sim::Vec3::new(45.0, 0.15, 2.0),
                0x009a_9488,
            ),
        )
        .unwrap();
    let sun = world.spawn();
    world
        .insert(
            sun,
            Light::sun(sim::Vec3::new(SUN[0], SUN[1], SUN[2]), 0x00ff_f4e0, 3.4),
        )
        .unwrap();
    world
}

fn render(renderer: &mut OffscreenRenderer, world: &World) -> Vec<i32> {
    let view = View {
        pitch: PITCH,
        ..View::default()
    };
    let mut extracted = Extracted::default();
    extracted.clear(
        sim::DVec3::new(EYE[0], EYE[1], EYE[2]),
        view.frustum(EXTENT),
    );
    extracted.append_lights(world).unwrap();
    extracted.cast_shadows(view.caster_reach(EXTENT));
    extracted.append::<Renderable>(world).unwrap();
    let pixels = renderer
        .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])
        .unwrap()
        .pixels;
    pixels
        .chunks_exact(4)
        .map(|p| (2126 * i32::from(p[0]) + 7152 * i32::from(p[1]) + 722 * i32::from(p[2])) / 10000)
        .collect()
}

/// The shipping defaults, restated so no leg inherits the previous leg's state.
fn ship() {
    cvars::SHADOW_SIZE.set_int(SHADOW_SIZE);
    cvars::SHADOW_CASCADES.set_int(4);
    cvars::SHADOW_DISTANCE.set_float(80.0);
    cvars::SHADOW_SOFTNESS.set_float(2.0);
    cvars::SUN_ANGLE.set_float(0.53);
    cvars::SHADOW_TAPS.set_int(16);
    cvars::SHADOW_PENUMBRA.set_float(16.0);
}

/// Median 20%→80% width of the steepest downward step in each column, in
/// pixels — the boundary's width, measured the way the desk's own screenshot
/// was.
///
/// The median and not the mean: a column that clipped the end of the slab, or
/// found no boundary at all, is not a sample of the edge, and a median needs no
/// threshold to say so.
fn soft(image: &[i32]) -> f64 {
    let w = EXTENT.0 as usize;
    let mut widths: Vec<f64> = Vec::new();
    for x in 0..w {
        let column: Vec<i32> = BAND.map(|y| image[y as usize * w + x]).collect();
        let Some(step) = (1..column.len()).min_by_key(|&i| column[i] - column[i - 1]) else {
            continue;
        };
        // Local levels, from a window around the step: the frame's global
        // extremes belong to whatever else is in it.
        let lo = step.saturating_sub(10);
        let hi = (step + 10).min(column.len());
        let dark = f64::from(*column[lo..hi].iter().min().unwrap());
        let lit = f64::from(*column[lo..hi].iter().max().unwrap());
        if lit - dark < 25.0 {
            continue;
        }
        // Walk back up from the step to where the column last held each level,
        // interpolating between the two samples that bracket it.
        let cross = |frac: f64| -> Option<f64> {
            let level = dark + frac * (lit - dark);
            (0..step).rev().find_map(|i| {
                let (a, b) = (f64::from(column[i]), f64::from(column[i + 1]));
                (a >= level).then(|| {
                    if a == b {
                        i as f64
                    } else {
                        i as f64 + (a - level) / (a - b)
                    }
                })
            })
        };
        if let (Some(p20), Some(p80)) = (cross(0.2), cross(0.8)) {
            widths.push(p20 - p80);
        }
    }
    assert!(
        widths.len() > EXTENT.0 as usize / 4,
        "only {} columns held a boundary — this framing has no shadow edge across it",
        widths.len()
    );
    widths.sort_by(f64::total_cmp);
    widths[widths.len() / 2]
}

fn mean_light(image: &[i32]) -> f64 {
    image.iter().map(|&v| f64::from(v)).sum::<f64>() / image.len() as f64
}

#[test]
fn a_shadow_boundary_is_never_narrower_than_a_pixel() {
    let world = world(10.0);
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();

    let widths = [0.0, 2.0, 4.0, 6.0];
    let mut measured = Vec::new();
    let mut light = Vec::new();
    for &softness in &widths {
        ship();
        // A point sun, so the *only* thing widening the boundary is the floor.
        // Left at its real angular size this framing softens to 1.4 px on its
        // own and the control below would pass with the floor never wired up.
        cvars::SUN_ANGLE.set_float(0.0);
        cvars::SHADOW_SOFTNESS.set_float(softness);
        let image = render(&mut renderer, &world);
        measured.push(soft(&image));
        light.push(mean_light(&image));
    }
    println!("boundary width by r.shadow_softness {widths:?}: {measured:?}");

    // The control. Without it the claim below would pass on a framing whose
    // edge was already soft for some other reason, which is how three earlier
    // rounds of this measurement went wrong.
    assert!(
        measured[0] < 1.0,
        "with the floor off the boundary is {:.2} px wide — this framing does not reproduce the \
         sub-pixel edge, so the floor has nothing to prove here",
        measured[0]
    );
    for (&softness, &width) in widths.iter().zip(&measured).skip(1) {
        assert!(
            width >= 1.0,
            "r.shadow_softness {softness} left the boundary {width:.2} px wide — a step inside one \
             pixel is what no MSAA count can resolve, which is the whole reason for the floor"
        );
    }
    for pair in measured.windows(2) {
        assert!(
            pair[1] > pair[0],
            "a wider floor did not widen the boundary: {measured:?} over {widths:?}"
        );
    }

    // And the other direction, which a width alone cannot see: a filter
    // redistributes light across a boundary, it does not add or remove it. The
    // slip this catches is dividing by a fixed tap count while the taps are
    // weighted, which would sail through every claim above.
    println!("mean luminance by softness: {light:?}");
    let widest = *light.last().unwrap();
    for (&softness, &mean) in widths.iter().zip(&light) {
        assert!(
            (mean - widest).abs() < widest * 0.02,
            "r.shadow_softness {softness} moved the frame's mean luminance from {widest:.3} to \
             {mean:.3}"
        );
    }
}

#[test]
fn a_penumbra_widens_with_the_distance_to_what_cast_it() {
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    let (low, high) = (world(1.5), world(12.0));

    // The floor off and the sun deliberately larger than the real one, so what
    // is being measured is the physical term alone and it is well clear of the
    // noise a sixteen-tap disk leaves.
    let mut edges = |angle: f64| {
        ship();
        cvars::SHADOW_SOFTNESS.set_float(0.0);
        cvars::SUN_ANGLE.set_float(angle);
        cvars::SHADOW_PENUMBRA.set_float(48.0);
        (
            soft(&render(&mut renderer, &low)),
            soft(&render(&mut renderer, &high)),
        )
    };

    let (near_sun, far_sun) = edges(4.0);
    println!("penumbra width at r.sun_angle 4: {near_sun:.2} px low, {far_sun:.2} px high");
    assert!(
        far_sun > near_sun * 2.0,
        "a slab at 12 m cast a {far_sun:.2} px penumbra and one at 1.5 m cast {near_sun:.2} px — a \
         soft shadow is one that widens with the gap to its caster, and these barely differ"
    );

    // The control: with a point sun the gap has nothing to scale, so the same
    // pair must draw the same edge. Without this the claim above is also
    // satisfied by a filter that simply widens with distance from the camera.
    let (near_point, far_point) = edges(0.0);
    println!("penumbra width at r.sun_angle 0: {near_point:.2} px low, {far_point:.2} px high");
    assert!(
        (far_point - near_point).abs() < 0.5,
        "a point sun still drew a {far_point:.2} px edge against a {near_point:.2} px one — the \
         width is tracking something other than the sun's angular size"
    );
}
