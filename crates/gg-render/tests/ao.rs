//! The gate for §6 M35's occlusion: **the picture must agree with the integral,
//! and it must be flat without it.**
//!
//! `gg-tools ao` is where the numbers live — the bias plateau, the slice budget,
//! what the denoiser is worth, what the pass costs. This asserts the two things
//! that must never regress silently, and they fail in opposite directions:
//!
//! - **Open surfaces stay open.** A screen-space estimate that invents occlusion
//!   on a flat floor is the classic SSAO artifact and it does not look like a
//!   bug, it looks like dirt — a blessed reference would move once, be
//!   re-blessed, and never object again. The bias sweep found exactly that
//!   failure at 0.14 of invented occlusion, and it was found by measuring rather
//!   than by looking.
//! - **Creases are actually darkened.** The opposite failure is a term that
//!   costs two fullscreen passes and does nothing, which no image gate would
//!   notice either — a slightly lighter frame is what a re-bless is for.
//!
//! The control under both is `r.ao 0`, which must render the frame *flat*: not
//! merely different, but with no occlusion anywhere. A test whose two assertions
//! both passed against an unoccluded picture would be measuring the ambient term.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::Renderable;
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, Samples, View, ao, cvars};

/// Small, but not *too* small, and that is a property of the thing being tested
/// rather than a budget: `r.ao_bias` is in pixels, so at a quarter of the linear
/// resolution the bias eats a sixth of the radius instead of a fifteenth and the
/// term genuinely weakens. 320x180 is where a crease still has pixels to march
/// through; the reference is cast at every third one of them.
const EXTENT: (u32, u32) = (320, 180);

/// Linear ambient. Below the tonemapper's knee, which is what makes the
/// comparison a ratio of two decoded code values rather than an inversion of
/// three curves — see `gg-tools ao`, whose reasoning this shares.
const AMBIENT: f64 = 0.25;

const EYE: sim::DVec3 = sim::DVec3::new(0.0, 1.4, 4.2);
const PITCH: f32 = -0.30;

/// Rays per reference point. Fewer than the instrument's, because what is being
/// resolved here is tenths and not thousandths.
const SAMPLES: u32 = 96;

/// How far a rendered pixel may sit from the integral on an **open** surface.
///
/// The number this is protecting is 0.0116, which is where `gg-tools ao` puts
/// the shipped configuration's invented occlusion. Three times it, so an
/// ordinary drift does not fail the tier and the 0.14 the missing bias produced
/// fails it twelve times over.
const OPEN_TOLERANCE: f32 = 0.035;

/// A floor, a back wall, and a box sitting on the floor — the least a contact
/// crease needs. `gg-tools ao`'s scene with the cases that only *it* reports
/// taken out: this is a gate, not a measurement.
fn scene() -> Vec<(sim::DVec3, sim::Vec3)> {
    vec![
        (
            sim::DVec3::new(0.0, -0.5, 0.0),
            sim::Vec3::new(6.0, 0.5, 6.0),
        ),
        (
            sim::DVec3::new(0.0, 1.5, -2.5),
            sim::Vec3::new(6.0, 2.0, 0.5),
        ),
        // Two blocks with a narrow slot between them, rather than one box on a
        // floor. A single box's contact crease is a band a couple of pixels wide
        // whose *true* occlusion barely reaches 0.85 — real, and too shallow to
        // assert on. The slot's floor is genuinely enclosed, which is what gives
        // the assertion below two populations that are actually far apart.
        (
            sim::DVec3::new(-0.45, 0.4, -0.8),
            sim::Vec3::new(0.35, 0.4, 0.5),
        ),
        (
            sim::DVec3::new(0.45, 0.4, -0.8),
            sim::Vec3::new(0.35, 0.4, 0.5),
        ),
    ]
}

fn world() -> World {
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    for (position, half_extent) in scene() {
        let entity = world.spawn();
        // White, fully rough, dielectric: with no sky declared the ambient path
        // is `ambient * diffuse * occlusion` and nothing else, so the picture is
        // the occlusion times a constant.
        world
            .insert(
                entity,
                Renderable::boxed(position, half_extent, 0x00ff_ffff).surfaced(0.0, 0.0),
            )
            .unwrap();
    }
    world
}

fn occluders() -> Vec<ao::Occluder> {
    scene()
        .into_iter()
        .map(|(center, half_extent)| ao::Occluder {
            center: sim::Vec3::new(
                (center.x - EYE.x) as f32,
                (center.y - EYE.y) as f32,
                (center.z - EYE.z) as f32,
            ),
            rotation: sim::Quat::IDENTITY,
            half_extent,
            sphere: false,
            // Unread here: occlusion is a property of shape (§6 M36).
            albedo: sim::Vec3::ZERO,
            emission: sim::Vec3::ZERO,
        })
        .collect()
}

/// Half-height of the orthographic leg, metres.
const ORTHO: f32 = 2.2;

fn view(leg: Leg) -> View {
    View {
        pitch: PITCH,
        ortho: if leg.ortho { ORTHO } else { 0.0 },
        ortho_far: 60.0,
        samples: leg.samples,
        ..View::default()
    }
}

/// One configuration of the pass. Three of them, and each covers a branch no
/// blessed reference in the tree reaches.
#[derive(Clone, Copy)]
struct Leg {
    name: &'static str,
    ortho: bool,
    samples: Samples,
}

const LEGS: [Leg; 3] = [
    Leg {
        name: "perspective",
        ortho: false,
        samples: Samples::X1,
    },
    // The orthographic view vector — see the loop below.
    Leg {
        name: "orthographic",
        ortho: true,
        samples: Samples::X1,
    },
    // And multisampled, which is the only thing that exercises §6 M35's depth
    // resolve: above one sample the prepass target cannot be sampled at all (the
    // global set declares no `2DMS` descriptor), so the occlusion pass reads a
    // single-sample reduction of it written by fixed function as the prepass
    // ends. A resolve that did not happen is an occlusion pass reading an image
    // nothing wrote, which these assertions see and no golden does — the suite
    // renders every scene at one sample.
    Leg {
        name: "multisampled",
        ortho: false,
        samples: Samples::X4,
    },
];

/// The primary ray through pixel `(x, y)` — origin and direction, in the eye's
/// frame. `View::view_projection`'s two projections, undone.
fn ray(x: u32, y: u32, ortho: bool) -> (sim::Vec3, sim::Vec3) {
    let aspect = EXTENT.0 as f32 / EXTENT.1 as f32;
    let ndc_x = (x as f32 + 0.5) / EXTENT.0 as f32 * 2.0 - 1.0;
    let ndc_y = 1.0 - (y as f32 + 0.5) / EXTENT.1 as f32 * 2.0;
    let (sin_pitch, cos_pitch) = sim::sin_cos(PITCH);
    let forward = sim::Vec3::new(0.0, sin_pitch, -cos_pitch);
    let right = sim::Vec3::new(1.0, 0.0, 0.0);
    let up = forward.cross(right) * -1.0;
    if ortho {
        return (
            right * (ndc_x * ORTHO * aspect) + up * (ndc_y * ORTHO),
            forward,
        );
    }
    let tan = sim::tan(View::default().fov_y * 0.5);
    let direction = (forward + right * (ndc_x * tan * aspect) + up * (ndc_y * tan))
        .try_normalize()
        .unwrap_or(forward);
    (sim::Vec3::ZERO, direction)
}

/// sRGB decode — `post.slang`'s encode run backwards.
fn decode(code: u8) -> f32 {
    let e = f32::from(code) / 255.0;
    if e <= 0.040_45 {
        e / 12.92
    } else {
        sim::powf((e + 0.055) / 1.055, 2.4)
    }
}

fn render(renderer: &mut OffscreenRenderer, world: &World, leg: Leg) -> Vec<f32> {
    let view = view(leg);
    let mut extracted = Extracted::default();
    extracted.clear(EYE, view.frustum(EXTENT));
    extracted.append::<Renderable>(world).unwrap();
    extracted.append_lights(world).unwrap();
    let pixels = renderer
        .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])
        .unwrap()
        .pixels;
    pixels.chunks_exact(4).map(|p| decode(p[1])).collect()
}

/// The rendered occlusion per pixel: the same frame with the pass on and off.
///
/// A ratio of the same pixel, which is what makes it exact rather than
/// calibrated — occlusion multiplies the ambient term and nothing else touches
/// it, so whatever the normal and the encode did to a pixel they did to both.
fn rendered(renderer: &mut OffscreenRenderer, world: &World, leg: Leg) -> Vec<f32> {
    cvars::AO.set_bool(false);
    let open = render(renderer, world, leg);
    cvars::AO.set_bool(true);
    let occluded = render(renderer, world, leg);
    occluded
        .iter()
        .zip(&open)
        .map(|(a, b)| if *b > 1e-5 { (a / b).min(1.0) } else { 1.0 })
        .collect()
}

/// Every third pixel, cast: `(reference, rendered)` pairs over the covered part
/// of the frame. Third and not eighth because the crease this measures is a
/// band a few pixels wide at this extent, and a coarse grid steps over it.
fn pairs(rendered: &[f32], ortho: bool) -> Vec<(f32, f32)> {
    let occluders = occluders();
    let params = ao::Params {
        radius: cvars::AO_RADIUS.float() as f32,
        samples: SAMPLES,
    };
    let mut out = Vec::new();
    for y in (0..EXTENT.1).step_by(3) {
        for x in (0..EXTENT.0).step_by(3) {
            let (origin, direction) = ray(x, y, ortho);
            let Some(hit) = ao::trace(origin, direction, &occluders, 1e-3, 1e4) else {
                continue;
            };
            let point = origin + direction * hit.distance;
            let truth = ao::occlusion(point, hit.normal, &occluders, params);
            out.push((truth, rendered[(y * EXTENT.0 + x) as usize]));
        }
    }
    out
}

#[test]
fn the_occlusion_pass_agrees_with_the_integral_it_estimates() {
    cvars::AMBIENT.set_float(AMBIENT);
    // The dither is a deliberate ±1 code value and this reads single pixels; the
    // ratio would carry it twice.
    cvars::DITHER.set_float(0.0);
    let world = world();
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();

    // **Three legs**, because no golden covers two of them. The only
    // orthographic scene in the suite is demo 11's flat framing, whose visible
    // faces are a single coplanar sheet — so it has almost no screen-space
    // occlusion to show and moves by one code value whatever this pass does.
    // That makes a legitimately-still reference and a dead branch look exactly
    // alike, which is what this leg exists to tell apart. It is coverage and not
    // a trap for one bug: `gg-tools ao`'s projection row is where the two
    // spellings of the view vector are compared as numbers.
    for leg in LEGS {
        let projection = leg.name;
        let pairs = pairs(&rendered(&mut renderer, &world, leg), leg.ortho);
        assert!(
            pairs.len() > 100,
            "{projection}: only {} covered sample points — the scene has moved out of frame and              nothing below is measuring occlusion",
            pairs.len()
        );

        // Open surfaces stay open. The failure this catches darkens a flat
        // floor, which reads as dirt rather than as a bug.
        let (mut invented, mut open) = (0.0f64, 0u32);
        let mut worst = 0.0f32;
        for (truth, got) in &pairs {
            if *truth < 0.999 {
                continue;
            }
            invented += f64::from((truth - got).max(0.0));
            worst = worst.max(truth - got);
            open += 1;
        }
        let invented = (invented / f64::from(open.max(1))) as f32;
        assert!(
            invented <= OPEN_TOLERANCE,
            "{projection}: the pass invented {invented:.4} of occlusion on {open} points the              integral calls fully open (worst {worst:.4}). A screen-space estimate that darkens a              flat surface is the artifact this term is most likely to ship — see `gg-tools ao`'s              bias sweep."
        );

        // And creases are darkened, in the right places. The opposite failure is
        // a term that costs two fullscreen passes and does nothing.
        //
        // Asserted as a *gap between two populations* rather than as an
        // absolute, because the honest expectation is not that the pass matches
        // the integral in a crease — it cannot, and `gg-tools ao` measures how
        // far short it falls (0.24 in the deepest bucket, most of it geometry
        // the depth buffer has no record of). What must hold is that the points
        // the integral calls occluded come out measurably darker than the ones
        // it calls open.
        let mean = |select: &dyn Fn(f32) -> bool| {
            let (mut total, mut n) = (0.0f64, 0u32);
            for (truth, got) in &pairs {
                if select(*truth) {
                    total += f64::from(*got);
                    n += 1;
                }
            }
            ((total / f64::from(n.max(1))) as f32, n)
        };
        let (open_mean, _) = mean(&|t| t >= 0.999);
        let (crease_mean, crease) = mean(&|t| t < 0.75);
        assert!(
            crease >= 24,
            "{projection}: only {crease} sample points are genuinely occluded — the scene no              longer has a crease to measure"
        );
        assert!(
            open_mean - crease_mean >= 0.05,
            "{projection}: the integral calls {crease} points at least 25 % occluded and the pass              rendered them at {crease_mean:.4} against {open_mean:.4} on the open ones — the              occlusion is being computed and thrown away"
        );

        // The control, and the reason the two assertions above mean anything:
        // with `r.ao` off the same frame must be *flat*. Not different — flat.
        // Both assertions pass trivially against an unoccluded picture, so a
        // test that could not tell one apart would be measuring the ambient
        // term.
        cvars::AO.set_bool(false);
        let flat = render(&mut renderer, &world, leg);
        cvars::AO.set_bool(true);
        let base = render(&mut renderer, &world, leg);
        let moved = flat
            .iter()
            .zip(&base)
            .filter(|(a, b)| (*a - *b).abs() > 1e-4)
            .count();
        assert!(
            moved * 5 >= flat.len(),
            "{projection}: only {moved} of {} pixels differ between `r.ao` on and off — the pass              is not reaching the picture",
            flat.len()
        );
    }

    cvars::DITHER.set_float(1.0);
    let report = renderer.shutdown();
    assert!(report.clean(), "unclean render: {report:?}");
}
