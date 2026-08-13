//! The gate an image cannot be, for §6 M33's energy half: **a surface that
//! absorbs nothing is invisible against the environment lighting it.**
//!
//! Put a perfect white metal — `metallic 1`, white, so `f0` is exactly 1 —
//! inside a sphere of uniform radiance. Whatever it does with the light it
//! radiates all of it back, so it comes out the same colour as the background
//! behind it and the picture is a flat field. That is the white furnace, and it
//! is the one test of this kind that needs no reference to argue with: the
//! *background is the reference*, drawn by the skybox from the same environment,
//! through the same exposure, the same tonemap and the same quantizer. Nothing
//! here has to know what any of those curves are, only that both halves of the
//! comparison travelled all of them — which is why this holds as an equality of
//! **code values** and not as a tolerance on radiance.
//!
//! What it is guarding is not the picture. A cull that stops culling renders the
//! same frame (§6 M32's argument, one milestone along); a compensation that
//! stops compensating renders a *different* one, so the goldens would catch a
//! revert. What they cannot catch is the difference between compensating
//! correctly and compensating by some other amount that happens to look
//! plausible — every blessed reference would move once, be re-blessed, and never
//! object again. This asserts the number.
//!
//! The second leg is the control, and it is the one that makes the first mean
//! anything: with `r.multiscatter 0` the same frame must **fail** to close, by a
//! margin far outside the tolerance. A furnace that closes either way is
//! measuring a flat field of sky with the surface accidentally invisible for
//! some other reason.
//!
//! `gg-tools furnace` is where the numbers live — the absolute energy per
//! roughness, the direct path's agreement with it, and the lobe's aim. This is
//! the one property of them all that must never regress silently.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::{Renderable, Sky};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

/// Small: every reading is one window in the middle and one in a corner.
const EXTENT: (u32, u32) = (192, 192);

/// Half-width of the windows averaged, in pixels.
const WINDOW: u32 = 8;

/// Linear radiance of the environment. Low enough to sit on the steep part of
/// the output curve, where a code value is worth a fraction of a per cent of it
/// — the shoulder near white would forgive a real difference.
const RADIANCE: f32 = 0.25;

/// The roughest surface, which is where the single-scatter loss is largest and
/// therefore where both legs have the most to say.
const ROUGHNESS: f32 = 1.0;

/// How far apart the surface and its background may sit, in output code values.
///
/// Not zero, and the reasons are all downstream of the shading: the split-sum's
/// second integral is an analytic fit rather than a table, the compensation
/// closes it in closed form rather than exactly, and the last thing that happens
/// to both numbers is a quantization to 256 levels. Two code values out of the
/// ~103 this environment lands at is under a fifth of one per cent of the
/// radiance, and the control below misses by more than twenty times it.
const TOLERANCE: f32 = 2.0;

/// A wall of perfect white metal, and a sky that is the same white everywhere.
///
/// The gradient projects to spherical harmonics *exactly* when it is a constant
/// — only the DC band is nonzero — so every direction returns `RADIANCE` with no
/// interpolation and no chain to resolve. That is what makes this a furnace
/// rather than an approximation of one, and it is why the scene needs no pack.
fn furnace() -> World {
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    world.register::<Sky>().unwrap();
    let sky = world.spawn();
    world
        .insert(
            sky,
            Sky {
                zenith: 0x00ff_ffff,
                horizon: 0x00ff_ffff,
                ground: 0x00ff_ffff,
                intensity: RADIANCE,
                ..Sky::daylight(RADIANCE)
            },
        )
        .unwrap();
    let wall = world.spawn();
    world
        .insert(
            wall,
            // Smaller than the orthographic frame on purpose: the reference this
            // compares against is the *background*, so the sky has to be
            // somewhere in the picture. A wall that filled the frame would be
            // measured against itself and close perfectly however wrong it was.
            Renderable::boxed(sim::DVec3::ZERO, sim::Vec3::new(2.0, 2.0, 0.5), 0x00ff_ffff)
                .surfaced(1.0 - ROUGHNESS, 1.0),
        )
        .unwrap();
    world
}

/// The centre window, which is wall, and a corner window, which is sky.
///
/// Orthographic for the reason `gg-tools furnace` is: under parallel rays the
/// eye direction is a constant across the frame, so the whole face shades
/// identically and the window reads one number many times rather than averaging
/// a gradient (§6 M20 is what put the projection's axis where the shader can
/// find it).
fn measure(renderer: &mut OffscreenRenderer, world: &World) -> (f32, f32) {
    let view = View {
        ortho: 4.0,
        ..View::default()
    };
    let eye = sim::DVec3::new(0.0, 0.0, 30.0);
    let mut extracted = Extracted::default();
    extracted.clear(eye, view.frustum(EXTENT));
    extracted.append::<Renderable>(world).unwrap();
    extracted.append_lights(world).unwrap();
    let pixels = renderer
        .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])
        .unwrap()
        .pixels;
    let mean = |cx: u32, cy: u32| {
        let (mut total, mut count) = (0.0f32, 0u32);
        for y in cy - WINDOW..=cy + WINDOW {
            for x in cx - WINDOW..=cx + WINDOW {
                total += f32::from(pixels[((y * EXTENT.0 + x) * 4 + 1) as usize]);
                count += 1;
            }
        }
        total / count as f32
    };
    (
        mean(EXTENT.0 / 2, EXTENT.1 / 2),
        mean(WINDOW + 2, WINDOW + 2),
    )
}

#[test]
fn a_white_metal_gives_back_everything_the_sky_gave_it() {
    // The dither is a deliberate ±1 code value and would be noise on a
    // comparison this tight; the windows are averaged, but the control's margin
    // is what this is protecting and it deserves a clean signal.
    cvars::DITHER.set_float(0.0);
    let world = furnace();
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();

    cvars::MULTISCATTER.set_bool(true);
    let (surface, background) = measure(&mut renderer, &world);
    assert!(
        (surface - background).abs() <= TOLERANCE,
        "the furnace does not close: the wall came out at {surface:.1} against a background of \
         {background:.1}, {:.1} code values apart. A surface that absorbs nothing must be \
         invisible against the environment lighting it.",
        (surface - background).abs()
    );

    // The control. Without the compensation the same frame must miss by a mile,
    // or the assertion above is measuring a sky with nothing in front of it.
    cvars::MULTISCATTER.set_bool(false);
    let (single, sky) = measure(&mut renderer, &world);
    cvars::MULTISCATTER.set_bool(true);
    assert!(
        sky - single > 10.0 * TOLERANCE,
        "with r.multiscatter off the wall came out at {single:.1} against {sky:.1} — the \
         single-scatter lobe is supposed to drop over half the energy at roughness {ROUGHNESS}, \
         so a furnace that nearly closes without the correction is not measuring the surface"
    );
    cvars::DITHER.set_float(1.0);

    let report = renderer.shutdown();
    assert!(report.clean(), "unclean render: {report:?}");
}

/// Steps of the radiance→code-value calibration, from nothing to a little over
/// [`RADIANCE`]. Measured rather than modelled, `gg-tools furnace`'s reason: the
/// exposure, the tonemap and the sRGB encode are three curves this would
/// otherwise hold a second, staleable copy of.
const CALIBRATION: usize = 65;

/// How far the rendered single-scatter albedo may sit from the integral, in
/// units of `E`.
///
/// Everything between the two is in here: the table's own 1.45 % worst
/// interpolation error, the shader's bilinear read of it, an eight-bit output
/// inverted through a 65-step curve, and the dither. `gg-tools furnace` measures
/// the whole path at 0.008 worst over a finer grid; this is that with room.
const ALBEDO_TOLERANCE: f32 = 0.03;

/// §6 M34: the split-sum's second integral has a view axis, and the table has it.
///
/// This is the assertion §6 M33 could not make and did not know it was missing.
/// The furnace above is *structurally blind* to `Ess`: at `f0 = 1` the ambient
/// path returns `Ess + (1 - Ess)`, which is 1 for any value at all, right or
/// wrong. So it closed against [Laz13]'s fit and closes against the table, and
/// between those two the fit was asking for 2.22x the light that arrived at a
/// grazing rough metal where the truth is 1.00x. What catches that is measuring
/// `Ess` itself — `r.multiscatter 0`, which is the same picture with the
/// correction taken off — against the integral it approximates.
///
/// The control is the fit, which is the *other* value of `r.lut` rather than
/// another build: it must fail here, and it must fail by being flat, because a
/// view axis that cancels is the specific thing wrong with it.
#[test]
fn a_rough_metal_gives_back_what_the_integral_says_at_every_view_angle() {
    cvars::DITHER.set_float(0.0);
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    let curve = calibrate(&mut renderer);
    // The correction off, so what is measured is the albedo and not 1 (above).
    cvars::MULTISCATTER.set_bool(false);

    cvars::LUT.set_bool(true);
    let mut rendered = Vec::new();
    for (roughness, cosine) in GRID {
        let e = albedo(&mut renderer, &turned(roughness, cosine), &curve);
        let (a, b) = gg_render::split_sum::integrate(roughness, cosine, 65_536);
        assert!(
            (e - (a + b)).abs() <= ALBEDO_TOLERANCE,
            "at roughness {roughness}, n·v {cosine}: the wall returned {e:.3} of the sky against \
             the integral's {:.3}. The single-scatter albedo is what §6 M33's correction divides \
             by, so a wrong one is a wrong amount of invented light.",
            a + b
        );
        rendered.push(e);
    }
    // The axis exists in the picture and not only in the arithmetic: a rough
    // metal is darkest head-on and recovers toward the silhouette, where the
    // facets a viewer can see are the ones turned toward them.
    assert!(
        // The real gap is 0.19; four tolerances is 0.12, and the control's is 0.
        rendered[2] > rendered[0] + 4.0 * ALBEDO_TOLERANCE,
        "roughness 1 returned {:.3} head-on and {:.3} at n·v 0.4 — barely apart, so whatever is \
         being read has no view angle in it",
        rendered[0],
        rendered[2]
    );

    // The control. The fit's `scale + bias` is `r.z + r.w` with the view term
    // cancelled, so this pair must come out identical — and wrong.
    cvars::LUT.set_bool(false);
    let flat: Vec<f32> = [GRID[0], GRID[2]]
        .iter()
        .map(|&(r, c)| albedo(&mut renderer, &turned(r, c), &curve))
        .collect();
    cvars::LUT.set_bool(true);
    cvars::MULTISCATTER.set_bool(true);
    cvars::DITHER.set_float(1.0);
    assert!(
        (flat[1] - flat[0]).abs() <= ALBEDO_TOLERANCE,
        "with r.lut off the two view angles returned {:.3} and {:.3} — the analytic fit is \
         supposed to be incapable of telling them apart, so this test is no longer measuring the \
         thing that replaced it",
        flat[0],
        flat[1]
    );

    let report = renderer.shutdown();
    assert!(report.clean(), "unclean render: {report:?}");
}

/// The `(roughness, n·v)` cells asserted. Two on the roughest row, where the fit
/// is worst and the axis steepest, and one smooth cell that both must agree on —
/// a table that had shifted a row would pass the rough pair and fail this one.
const GRID: [(f32, f32); 3] = [(1.0, 1.0), (0.2, 1.0), (1.0, 0.4)];

/// A white metal wall turned `cosine` away from an orthographic eye.
///
/// Orthographic for the reason [`measure`] is, and turned about `Y` so `n·v` is
/// exactly the cosine asked for at every pixel of it rather than only at the
/// centre. Wider than it is deep so the turned face still covers the window and
/// the sky still reaches the corner.
fn turned(roughness: f32, cosine: f32) -> World {
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    world.register::<Sky>().unwrap();
    let sky = world.spawn();
    world
        .insert(
            sky,
            Sky {
                zenith: 0x00ff_ffff,
                horizon: 0x00ff_ffff,
                ground: 0x00ff_ffff,
                intensity: RADIANCE,
                ..Sky::daylight(RADIANCE)
            },
        )
        .unwrap();
    let angle = sim::acos(f64::from(cosine.clamp(-1.0, 1.0)));
    let mut surface =
        Renderable::boxed(sim::DVec3::ZERO, sim::Vec3::new(3.0, 3.0, 0.2), 0x00ff_ffff)
            .surfaced(1.0 - roughness, 1.0);
    surface.rotation = sim::DQuat::from_axis_angle(sim::DVec3::Y, angle);
    let wall = world.spawn();
    world.insert(wall, surface).unwrap();
    world
}

/// Radiance → output code value, read off the *background* so the calibration
/// and the measurement travel the identical post chain.
fn calibrate(renderer: &mut OffscreenRenderer) -> Vec<(f32, f32)> {
    (0..CALIBRATION)
        .map(|step| {
            let radiance = RADIANCE * 1.2 * step as f32 / (CALIBRATION - 1) as f32;
            let mut world = World::new();
            world.register::<Renderable>().unwrap();
            world.register::<Sky>().unwrap();
            let sky = world.spawn();
            world
                .insert(
                    sky,
                    Sky {
                        zenith: 0x00ff_ffff,
                        horizon: 0x00ff_ffff,
                        ground: 0x00ff_ffff,
                        intensity: radiance,
                        ..Sky::daylight(radiance)
                    },
                )
                .unwrap();
            (radiance, measure(renderer, &world).0)
        })
        .collect()
}

/// What the wall returned, over [`RADIANCE`] — the furnace's own number, with
/// the output curve inverted out of it.
fn albedo(renderer: &mut OffscreenRenderer, world: &World, curve: &[(f32, f32)]) -> f32 {
    let code = measure(renderer, world).0;
    let at = curve.partition_point(|&(_, c)| c < code);
    let radiance = if at == 0 {
        curve[0].0
    } else if at >= curve.len() {
        curve[curve.len() - 1].0
    } else {
        let ((r0, c0), (r1, c1)) = (curve[at - 1], curve[at]);
        // A flat run of the curve means the honest answer is its middle.
        if (c1 - c0).abs() < f32::EPSILON {
            (r0 + r1) * 0.5
        } else {
            r0 + (r1 - r0) * (code - c0) / (c1 - c0)
        }
    };
    radiance / RADIANCE
}
