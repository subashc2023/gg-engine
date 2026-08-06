//! The frame inside a rectangle rather than over the whole target (§4.5).
//!
//! What the editor needs of the renderer (§6 M15): its panels frame a viewport,
//! and until the renderer had one the game was composed for the *window* and
//! shown through a hole — so an object at the edge of the panel was off-screen,
//! and the letterbox `gg_ui::Fit` opens at a non-16:9 window was bare game down
//! both edges. The shell is windowed and therefore manual (§1.5), so the claim
//! is proven here, offscreen, where a test can read the pixels.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, Viewport};

/// Linear, and far enough off black that the tonemapper cannot land the scene
/// on the same value as the letterbox.
const CLEAR: [f32; 4] = [0.30, 0.05, 0.05, 1.0];

/// A pixel, RGBA8 as the readback packs it.
type Pixel = [u8; 4];

const BLACK: Pixel = [0, 0, 0, 255];

fn pixel(pixels: &[u8], extent: (u32, u32), x: u32, y: u32) -> Pixel {
    let at = ((y * extent.0 + x) * 4) as usize;
    pixels[at..at + 4].try_into().unwrap()
}

/// Boxes four metres down -Z, spread across the frame and lit so they are not
/// the clear colour.
///
/// Spread deliberately: a single box on the axis barely moves when the aspect
/// it was composed for changes, so a test built on one would pass on a picture
/// that is wrong everywhere but the middle.
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

/// One frame of `world` at `extent`, composed for `viewport` when there is one.
fn render(extent: (u32, u32), viewport: Option<Viewport>) -> Vec<u8> {
    let world = world();
    let mut renderer = OffscreenRenderer::new(extent).unwrap();
    renderer.set_viewport(viewport);
    let view = View::default();
    let mut extracted = Extracted::default();
    // `view_extent`, exactly as the shell culls (`gg_runtime`'s extract stage):
    // a frustum built from the surface while the picture is built for the
    // viewport is the bug this pairing exists to prevent.
    extracted.clear(sim::DVec3::ZERO, view.frustum(renderer.view_extent()));
    extracted.append::<Renderable>(&world).unwrap();
    extracted.append_lights(&world).unwrap();
    renderer
        .frame(&extracted, &view, CLEAR, &[])
        .unwrap()
        .pixels
}

/// The composite lands on the declared rectangle and nowhere else, to the pixel.
///
/// Both halves matter. Scene inside proves the viewport is not simply shrinking
/// the picture to a corner; black outside proves the clear reaches the pixels
/// the draws no longer do — an unwritten backbuffer is whatever the last frame
/// in that slot left there, which on a swapchain is a stale frame and not black.
#[test]
fn the_viewport_places_the_frame_and_blacks_everything_outside_it() {
    let extent = (64, 64);
    let region = Viewport {
        x: 12,
        y: 8,
        width: 40,
        height: 24,
    };
    let pixels = render(extent, Some(region));
    let at = |x, y| pixel(&pixels, extent, x, y);

    for y in 0..extent.1 {
        for x in 0..extent.0 {
            let inside = x >= region.x
                && x < region.x + region.width
                && y >= region.y
                && y < region.y + region.height;
            match inside {
                true => assert_ne!(at(x, y), BLACK, "the scene is missing at ({x}, {y})"),
                false => assert_eq!(at(x, y), BLACK, "the frame leaked to ({x}, {y})"),
            }
        }
    }
}

/// How far apart two same-sized pictures are: how many pixels exceed
/// [`CHANNEL_TOLERANCE`] on any channel, and the mean worst-channel difference.
///
/// Two numbers because they fail for different reasons — the count catches a
/// picture that moved, the mean catches one that is subtly everywhere wrong.
fn compare(a: &[u8], b: &[u8], b_extent: (u32, u32), at: Viewport) -> (usize, f64) {
    let (mut differing, mut total) = (0usize, 0u64);
    for y in 0..at.height {
        for x in 0..at.width {
            let want = pixel(a, (at.width, at.height), x, y);
            let got = pixel(b, b_extent, at.x + x, at.y + y);
            let worst = want
                .iter()
                .zip(got.iter())
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            total += u64::from(worst);
            differing += usize::from(worst > CHANNEL_TOLERANCE);
        }
    }
    (differing, total as f64 / f64::from(at.width * at.height))
}

/// A viewport is a render target, not a crop of a window-sized frame.
///
/// The same rectangle rendered two ways: alone at 40×24, and as a viewport
/// inside a 96×64 surface. Same size and same aspect, so the same projection —
/// the pictures must match.
///
/// The second half is the control, and it is the half that makes the first mean
/// anything: the *old* composition — a window-sized frame with the panel over a
/// hole in it — is rendered here too, and the same comparison has to reject it.
/// A tolerance loose enough to forgive a crop would forgive the bug (§4.10's
/// rule that a suite which cannot fail is not a gate).
///
/// Tolerant per channel rather than exact: the viewport transform shifts every
/// vertex by the rectangle's offset, and a triangle edge that lands on a pixel
/// boundary either way is one rounding away from covering a different pixel.
#[test]
fn the_frame_is_composed_for_the_viewport_and_not_cropped_from_the_window() {
    let surface = (96, 64);
    let inset = Viewport {
        x: 30,
        y: 20,
        width: 40,
        height: 24,
    };
    let alone = render((inset.width, inset.height), None);

    // A silhouette's worth of rounding, and no more: on the pin the composite
    // scores 0 and 0.0 against a crop's 442 and 18.0. The budget is not slack
    // the composite needs — it is room for another driver's edge rounding, and
    // it was wide enough to hide a lighting block projected from the surface's
    // aspect instead of the viewport's, which is what a 0 here now pins.
    let budget = (inset.width * inset.height) as usize / 8;
    let judge = |(differing, mean): (usize, f64)| differing <= budget && mean <= MEAN_TOLERANCE;

    let composited = render(surface, Some(inset));
    let scored = compare(&alone, &composited, surface, inset);
    assert!(
        judge(scored),
        "{} of {} pixels differ by more than {CHANNEL_TOLERANCE} (mean {:.2}, budget {budget}) — \
         the viewport is not composed for its own rectangle",
        scored.0,
        inset.width * inset.height,
        scored.1
    );

    let cropped = render(surface, None);
    let control = compare(&alone, &cropped, surface, inset);
    assert!(
        !judge(control),
        "the comparison cannot tell a composite from a crop: a window-sized frame read through the \
         same rectangle scored {} differing (budget {budget}) and mean {:.2}",
        control.0,
        control.1
    );
}

/// Per channel, over the 8-bit values the readback packs. Only the silhouette
/// should reach it: the two frames differ by which side of a pixel boundary a
/// triangle edge rounded to.
const CHANNEL_TOLERANCE: u8 = 4;
/// Averaged over every pixel, which no rounding budget hides.
const MEAN_TOLERANCE: f64 = 8.0;

/// The extent everything downstream is sized from follows the viewport, and a
/// degenerate one falls back to the surface rather than asking the pool for a
/// zero-sized attachment.
#[test]
fn the_view_extent_is_the_viewport_and_a_degenerate_one_is_the_surface() {
    let mut renderer = OffscreenRenderer::new((64, 64)).unwrap();
    assert_eq!(renderer.view_extent(), (64, 64));
    renderer.set_viewport(Some(Viewport {
        x: 8,
        y: 8,
        width: 32,
        height: 16,
    }));
    assert_eq!(renderer.view_extent(), (32, 16));
    // Past the surface: clamped, and what is left of it is what the frame is
    // built for. A window that shrank under a stale viewport lands here.
    renderer.set_viewport(Some(Viewport {
        x: 48,
        y: 0,
        width: 32,
        height: 64,
    }));
    assert_eq!(renderer.view_extent(), (16, 64));
    renderer.set_viewport(Some(Viewport {
        x: 64,
        y: 64,
        width: 8,
        height: 8,
    }));
    assert_eq!(
        renderer.view_extent(),
        (64, 64),
        "nothing left to render into"
    );
    renderer.set_viewport(None);
    assert_eq!(renderer.view_extent(), (64, 64));
    drop(renderer.shutdown());
}

/// A pane docked behind another asks for a *zero-sized* viewport
/// (`gg_editor::viewport_rect`), and the two ways to honour that are not
/// equivalent: a zero-extent scissor discards the draw and is valid, while a
/// zero-width viewport is VUID-VkViewport-width-01770 — a validation message,
/// and therefore an unclean §4.3 shutdown report, once per draw per frame.
///
/// Asserted on both, because each half fails alone: a viewport floored to a unit
/// that forgot its scissor would leave a stray lit pixel, and a correct picture
/// says nothing about what the layer heard.
#[test]
fn a_hidden_pane_draws_nothing_and_says_nothing() {
    let extent = (64, 64);
    let world = world();
    let mut renderer = OffscreenRenderer::new(extent).unwrap();
    renderer.set_viewport(Some(Viewport {
        x: 20,
        y: 16,
        width: 0,
        height: 0,
    }));
    let view = View::default();
    let mut extracted = Extracted::default();
    extracted.clear(sim::DVec3::ZERO, view.frustum(renderer.view_extent()));
    extracted.append::<Renderable>(&world).unwrap();
    extracted.append_lights(&world).unwrap();
    let pixels = renderer
        .frame(&extracted, &view, CLEAR, &[])
        .unwrap()
        .pixels;
    for y in 0..extent.1 {
        for x in 0..extent.0 {
            assert_eq!(pixel(&pixels, extent, x, y), BLACK, "drew at ({x}, {y})");
        }
    }
    let report = renderer.shutdown();
    assert!(report.clean(), "{report:?}");
}
