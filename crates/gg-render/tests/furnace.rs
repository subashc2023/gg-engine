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
