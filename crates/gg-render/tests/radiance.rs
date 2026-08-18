//! `OffscreenRenderer::capture_radiance` (§6 M70) — the frame in linear radiance,
//! beside the frame a screen would get.
//!
//! Why it needs its own gate rather than riding on a picture: three instruments
//! grade a render by a **ratio of the same pixel**, and they had been taking that
//! ratio from the backbuffer. Everything between the scene attachment and the
//! backbuffer is display encoding — the exposure, `pbr_neutral`, the antialiaser and
//! eight bits — and the first of those is not linear and the last is not fine. The
//! claim this file holds is that the new readback is upstream of all of it, and it is
//! held four ways because each alone passes for something wrong:
//!
//! - the background's radiance **is the clear colour**, exactly, while the pixels at
//!   the same place are not — the tonemapper's pedestal, stated as a test;
//! - `r.exposure` and `r.dither` live in the post pass, so they must move the pixels
//!   and leave the radiance **bit-identical**;
//! - a pixel at code 255 still has its radiance, which is the range eight bits do not
//!   have at all;
//! - and a frame nobody asked declares **no** copy, because a pass declared here
//!   lands in the render-graph dump the push tier compares against every golden.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::Renderable;
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenFrame, OffscreenRenderer, View, cvars};

const EXTENT: (u32, u32) = (64, 36);

/// Deliberately not a value the chain reproduces: each channel is well above the
/// pedestal's 0.08 branch, so all three come back four hundredths light, and none of
/// them lands on a code value.
const CLEAR: [f32; 4] = [0.37, 0.21, 0.11, 1.0];

/// Above every code the quantizer has — `pbr_neutral(40)` encodes to 255.
const BLOWN: [f32; 4] = [40.0, 40.0, 40.0, 1.0];

/// A pixel the geometry cannot reach at this framing: the top-left corner, which is
/// sky. What the clear claim is measured at.
const SKY: (u32, u32) = (1, 1);

/// One small box well below the eye, so the top of the frame is the clear and
/// nothing in this file depends on how it is shaded.
fn world() -> World {
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    let box_ = world.spawn();
    world
        .insert(
            box_,
            Renderable::boxed(
                sim::DVec3::new(0.0, -4.0, -6.0),
                sim::Vec3::new(2.0, 2.0, 2.0),
                0x0080_8080,
            ),
        )
        .unwrap();
    world
}

fn render(renderer: &mut OffscreenRenderer, world: &World, clear: [f32; 4]) -> OffscreenFrame {
    let view = View::default();
    let mut extracted = Extracted::default();
    extracted.clear(sim::DVec3::new(0.0, 0.0, 0.0), view.frustum(EXTENT));
    extracted.append::<Renderable>(world).unwrap();
    renderer.frame(&extracted, &view, clear, &[]).unwrap()
}

fn at(frame: &OffscreenFrame, x: u32, y: u32) -> ([u8; 4], [f32; 4]) {
    let i = (y * EXTENT.0 + x) as usize;
    let p = &frame.pixels[i * 4..i * 4 + 4];
    let r = &frame.radiance[i * 4..i * 4 + 4];
    ([p[0], p[1], p[2], p[3]], [r[0], r[1], r[2], r[3]])
}

/// sRGB decode, so the pixel half of the claim below is in the same units as the
/// radiance half.
fn decode(code: u8) -> f32 {
    let e = f32::from(code) / 255.0;
    if e <= 0.040_45 {
        e / 12.92
    } else {
        sim::powf((e + 0.055) / 1.055, 2.4)
    }
}

#[test]
fn the_backgrounds_radiance_is_the_clear_colour_and_its_pixels_are_not() {
    let world = world();
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    renderer.capture_radiance(true);
    cvars::DITHER.set_float(0.0);
    let frame = render(&mut renderer, &world, CLEAR);
    let (pixel, radiance) = at(&frame, SKY.0, SKY.1);
    println!("clear {CLEAR:?} -> pixels {pixel:?} -> radiance {radiance:?}");
    for c in 0..3 {
        // Ten bits of mantissa is the tolerance: `SCENE_FORMAT` is `Rgba16F`, so a
        // value near 0.37 is resolved to about 2e-4 and no better.
        assert!(
            (radiance[c] - CLEAR[c]).abs() < 1e-3,
            "channel {c}: the readback says {} where the clear was {}",
            radiance[c],
            CLEAR[c]
        );
        // And the other half of the claim, which is the reason the readback exists:
        // the backbuffer is *not* this number. `pbr_neutral` took 0.04 off every
        // channel of a colour whose darkest is above its 0.08 branch.
        let decoded = decode(pixel[c]);
        assert!(
            (decoded - CLEAR[c]).abs() > 0.02,
            "channel {c}: the pixels decode to {decoded}, which is the clear {} — this framing \
             is not exercising the tonemapper, so the claim above would pass for free",
            CLEAR[c]
        );
    }
    cvars::DITHER.set_float(1.0);
    assert!(renderer.shutdown().clean());
}

#[test]
fn the_post_passs_own_knobs_move_the_pixels_and_leave_the_radiance_alone() {
    let world = world();
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    renderer.capture_radiance(true);
    cvars::DITHER.set_float(0.0);
    // **Off, and the first version of this test failed for want of it.** The field is
    // gathered a slice of probes per frame, so consecutive frames of one renderer are
    // legitimately different pictures until it has converged — and this test compares
    // frames, so it would have read that convergence as the exposure moving the scene
    // attachment. Nothing here is about the field: the claim is what the *post* pass
    // is downstream of.
    cvars::GI.set_bool(false);
    let base = render(&mut renderer, &world, CLEAR).radiance;
    let before = render(&mut renderer, &world, CLEAR).pixels;
    assert_eq!(
        render(&mut renderer, &world, CLEAR).radiance,
        base,
        "two frames of one renderer already disagree, so nothing below is measurable"
    );
    for knob in ["r.exposure", "r.dither"] {
        match knob {
            "r.exposure" => cvars::EXPOSURE.set_float(1.5),
            _ => cvars::DITHER.set_float(1.0),
        }
        let frame = render(&mut renderer, &world, CLEAR);
        match knob {
            "r.exposure" => cvars::EXPOSURE.set_float(0.0),
            _ => cvars::DITHER.set_float(0.0),
        }
        assert_ne!(
            frame.pixels, before,
            "{knob} moved no pixel, so the claim below holds for the wrong reason"
        );
        // Bit-identical, not close: the copy is of an image neither knob is upstream
        // of, so there is no rounding for either to cause.
        assert_eq!(
            frame.radiance, base,
            "{knob} moved the scene attachment, which means the readback is downstream of it"
        );
    }
    cvars::GI.set_bool(true);
    cvars::DITHER.set_float(1.0);
    assert!(renderer.shutdown().clean());
}

#[test]
fn a_pixel_the_quantizer_blew_still_has_its_radiance() {
    let world = world();
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    renderer.capture_radiance(true);
    let frame = render(&mut renderer, &world, BLOWN);
    let (pixel, radiance) = at(&frame, SKY.0, SKY.1);
    println!("clear {BLOWN:?} -> pixels {pixel:?} -> radiance {radiance:?}");
    assert_eq!(
        [pixel[0], pixel[1], pixel[2]],
        [255, 255, 255],
        "the framing is not blowing the quantizer, so this test has no subject"
    );
    for c in 0..3 {
        assert!(
            (radiance[c] - BLOWN[c]).abs() < 0.05,
            "channel {c}: {} where the clear was {}",
            radiance[c],
            BLOWN[c]
        );
    }
    assert!(renderer.shutdown().clean());
}

/// The reason it is opt-in, as a test rather than as a comment: a second copy pass
/// would otherwise appear in the render-graph dump `--push` compares for every
/// golden scene.
#[test]
fn no_frame_declares_the_copy_unless_it_was_asked() {
    let world = world();
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    let quiet = render(&mut renderer, &world, CLEAR);
    assert!(
        quiet.radiance.is_empty(),
        "a frame nobody asked handed back {} floats",
        quiet.radiance.len()
    );
    let one = quiet.order.iter().filter(|p| *p == "readback").count();
    assert_eq!(
        one, 1,
        "the default graph's readback passes: {:?}",
        quiet.order
    );
    renderer.capture_radiance(true);
    let asked = render(&mut renderer, &world, CLEAR);
    assert_eq!(
        asked.radiance.len(),
        (EXTENT.0 * EXTENT.1 * 4) as usize,
        "asked for radiance and got {} floats",
        asked.radiance.len()
    );
    assert_eq!(
        asked.order.iter().filter(|p| *p == "readback").count(),
        2,
        "asked for radiance and the graph declared {:?}",
        asked.order
    );
    // And back off again, because a sweep turns it on for one leg.
    renderer.capture_radiance(false);
    assert!(render(&mut renderer, &world, CLEAR).radiance.is_empty());
    assert!(renderer.shutdown().clean());
}
