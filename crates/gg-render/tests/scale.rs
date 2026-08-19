//! The scene drawn at a fraction of the window — `r.scale` (§6 M78).
//!
//! Why it exists is a measurement rather than a preference: `gg-tools frame
//! --sweep` reads demo 12's room as **99.7 % fragments** on the integrated
//! Radeon in this desk — 37.7 ms per megapixel against a floor of 0.2 — so the
//! frame is very nearly the pixel count and nothing else, and 79.4 ms at
//! 1920x1080 is 34.0 at 1280x720. The same room is 0.94 ms on the 4090 beside
//! it. Eighty-five times, in one machine, which is why nothing here noticed for
//! seventy-seven milestones.
//!
//! What the knob has to be, and what these tests are for, is that it is **not an
//! approximation of a smaller window** — it is a smaller window, with the
//! window's own size kept for the two things that are really the window's: where
//! the post pass puts the result, and the UI layer drawn over it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

/// Off black, so a pixel the scene never reached is telling apart from one it
/// did.
const CLEAR: [f32; 4] = [0.02, 0.02, 0.03, 1.0];

/// Boxes spread across the frame rather than one on the axis: a picture that is
/// right in the middle and wrong at the edges is exactly what a resolution
/// change breaks, and one box would not see it.
fn world() -> World {
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    world.register::<Light>().unwrap();
    for x in [-2.4, -0.8, 0.9, 2.5] {
        let entity = world.spawn();
        world
            .insert(
                entity,
                Renderable::boxed(
                    sim::DVec3::new(x, x * 0.3, -4.0),
                    sim::Vec3::splat(0.45),
                    0x00c0_4020,
                ),
            )
            .unwrap();
    }
    let sun = world.spawn();
    world
        .insert(
            sun,
            Light::sun(sim::Vec3::new(0.2, -0.6, -1.0), 0x00ff_ffff, 3.0),
        )
        .unwrap();
    world
}

/// One frame at `extent` with `r.scale` set to `scale`, handing back the
/// presented pixels, the scene attachment's own radiance, and the extent that
/// attachment was allocated at.
fn render(extent: (u32, u32), scale: f64) -> (Vec<u8>, Vec<f32>, (u32, u32)) {
    // The field is off by default since §6 M71 and is turned off here anyway:
    // an unconverged probe grid moves between frames, and every claim below is
    // an equality between two renders (§6 M67's lesson).
    cvars::GI.set_bool(false);
    cvars::SCALE.set_float(scale);
    let world = world();
    let mut renderer = OffscreenRenderer::new(extent).unwrap();
    renderer.capture_radiance(true);
    let view = View::default();
    let mut extracted = Extracted::default();
    // As the shell culls — against the extent the projection is built from,
    // which is the scaled one.
    extracted.clear(sim::DVec3::ZERO, view.frustum(renderer.view_extent()));
    extracted.append::<Renderable>(&world).unwrap();
    extracted.append_lights(&world).unwrap();
    let at = renderer.view_extent();
    let frame = renderer.frame(&extracted, &view, CLEAR, &[]).unwrap();
    let out = (frame.pixels.clone(), frame.radiance.clone(), at);
    assert!(renderer.shutdown().clean(), "unclean render");
    out
}

/// The claim the whole milestone rests on: a half-scale frame's scene
/// attachment is **byte-identical** to the same scene rendered into a window
/// half the size.
///
/// Exact rather than tolerant, and it can be: the scene passes never see the
/// backbuffer. The projection, the culling frustum, the depth buffer, the
/// occlusion pass and the cluster grid are all built from `view_extent`, so two
/// frames that agree on that number are the same frame — what differs is only
/// where the post pass puts the result, which happens after everything measured
/// here.
///
/// Graded on radiance rather than on pixels because the presented image is the
/// one thing that *must* differ, and because a ratio through the tonemapper is
/// not the quantity it looks like (§6 M70).
#[test]
fn a_render_scale_is_a_smaller_window_and_not_an_approximation_of_one() {
    let (small_pixels, small, small_at) = render((128, 72), 1.0);
    let (big_pixels, scaled, scaled_at) = render((256, 144), 0.5);

    assert_eq!(small_at, (128, 72), "the unscaled frame moved");
    assert_eq!(scaled_at, (128, 72), "half of 256x144 is not 128x72");
    // Both halves of "not vacuous": the readback happened at all, and what it
    // read is a lit room rather than a cleared buffer. An equality between two
    // empty vectors would pass every version of this test ever written.
    assert_eq!(small.len(), 128 * 72 * 4, "nothing was read back");
    assert_eq!(
        small.len(),
        scaled.len(),
        "two attachments of one size read back different lengths"
    );
    let lit = small.iter().filter(|v| **v > 0.05).count();
    assert!(lit > small.len() / 10, "only {lit} samples carry any light");
    assert_eq!(
        small, scaled,
        "the scene attachment is not the frame a smaller window would have drawn"
    );

    // And the window is still the window: the presented picture is the target's
    // size in both, which is the half of this that a reader would otherwise have
    // to take on trust.
    assert_eq!(small_pixels.len(), 128 * 72 * 4);
    assert_eq!(big_pixels.len(), 256 * 144 * 4);
}

/// The small picture is *stretched over* the target, not placed in a corner of
/// it — the failure a 1:1 blit of a smaller source produces, and the one that
/// looks fine in the middle of the screen.
#[test]
fn the_scaled_frame_covers_the_whole_target() {
    let extent = (192, 108);
    let (pixels, ..) = render(extent, 0.5);
    let at = |x: u32, y: u32| {
        let i = ((y * extent.0 + x) * 4) as usize;
        [pixels[i], pixels[i + 1], pixels[i + 2]]
    };
    // The clear as it lands after the tonemapper, read from the frame itself
    // rather than predicted: what matters is that the far corner is not it.
    let lit = |p: [u8; 3]| u32::from(p[0]) + u32::from(p[1]) + u32::from(p[2]);
    let middle = lit(at(extent.0 / 2, extent.1 / 2));
    assert!(middle > 0, "the scene is missing from the middle");
    for (x, y) in [
        (0, 0),
        (extent.0 - 1, 0),
        (0, extent.1 - 1),
        (extent.0 - 1, extent.1 - 1),
        (extent.0 - 1, extent.1 / 2),
    ] {
        // A corner is sky, not an unwritten pixel: the post pass wrote it, so it
        // carries the clear through the tonemapper rather than a zero.
        assert!(
            lit(at(x, y)) > 0,
            "({x}, {y}) was never written — the source was blitted 1:1 instead of fitted"
        );
    }
}

/// The knob's own arithmetic: rounded, clamped at both ends, and never zero.
///
/// Read through `view_extent`, which is what every consumer of it uses — a test
/// against the private multiply would prove the multiply and not the frame.
#[test]
fn the_knob_rounds_and_clamps_and_never_allocates_nothing() {
    let renderer = |extent, scale| {
        cvars::SCALE.set_float(scale);
        let r = OffscreenRenderer::new(extent).unwrap();
        let at = r.view_extent();
        assert!(r.shutdown().clean());
        at
    };

    assert_eq!(renderer((100, 50), 1.0), (100, 50), "off is off");
    assert_eq!(renderer((100, 50), 0.5), (50, 25));
    // Rounds rather than truncates: half of 51 is the nearer pixel, and a floor
    // would make every odd width one short.
    assert_eq!(renderer((51, 51), 0.5), (26, 26));
    // Clamped, not refused — a knob a player can type into is a knob that will
    // hold nonsense, and the alternative to clamping is a window that does not
    // open.
    assert_eq!(renderer((100, 50), 0.01), (25, 13), "clamped at a quarter");
    assert_eq!(renderer((100, 50), 9.0), (200, 100), "clamped at two");
    // A target small enough that a quarter of it rounds to nothing still has to
    // allocate something: the attachment pool refuses a zero extent.
    assert_eq!(renderer((2, 1), 0.25), (1, 1), "an axis never reaches zero");
}

/// Supersampling is the same seam upward, and it is the case that catches a
/// readback buffer sized from the window rather than from the scene.
///
/// The radiance buffer is allocated once at bring-up; before §6 M78 it was sized
/// `extent`, and the scene attachment could not be larger than that so nobody
/// had to think about it. At `r.scale 2` it is four times larger.
#[test]
fn supersampling_reads_back_the_attachment_and_not_the_window() {
    let (pixels, radiance, at) = render((64, 36), 2.0);
    assert_eq!(at, (128, 72), "the scene is drawn at twice the window");
    assert_eq!(
        radiance.len(),
        128 * 72 * 4,
        "the readback was sized from the window, not from the scene"
    );
    assert_eq!(pixels.len(), 64 * 36 * 4, "and the window is still 64x36");
}
