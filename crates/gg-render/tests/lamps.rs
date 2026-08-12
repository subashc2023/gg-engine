//! What §6 M31 claims, as geometry rather than as a look: **a wall between a
//! lamp and a floor makes the floor behind it dark, and leaves the floor in
//! front of it alone.**
//!
//! Two numbers failing in opposite directions, which is the shape every shadow
//! measurement in this tree has (`gg-tools shadow-flat` is the other): a lamp
//! that shadows nothing fails the first, and a lamp that shadows everything —
//! acne, a mirrored grid, a filter reading the neighbouring tile — fails the
//! second. One of them alone can be passed by doing nothing at all.
//!
//! The camera looks straight down, so the two regions are rectangles in the
//! image and no projection has to be reasoned about to say which pixels belong
//! to which. The lamp is *off centre* on purpose: a symmetric arrangement would
//! be passed by a lookup that mirrored the grid, which is exactly the defect
//! §6 M30 shipped and its lattice test caught.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

const EXTENT: (u32, u32) = (256, 256);

/// Where the wall stands. The lamp is at -3 and the camera looks down the y
/// axis, so world -z is up the image.
const WALL_Z: f64 = 0.0;
const LAMP_Z: f64 = -3.0;
const LAMP_HEIGHT: f64 = 1.6;
const RANGE: f32 = 12.0;

fn world(wall: bool) -> World {
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    world.register::<Light>().unwrap();
    let floor = world.spawn();
    world
        .insert(
            floor,
            Renderable::boxed(
                sim::DVec3::new(0.0, -0.1, 0.0),
                sim::Vec3::new(8.0, 0.1, 8.0),
                0x00c0_c0c0,
            ),
        )
        .unwrap();
    if wall {
        // Thin, tall, and wide enough to cross the whole frame: what is being
        // measured is the shadow's *side*, not its ends.
        let blocker = world.spawn();
        world
            .insert(
                blocker,
                Renderable::boxed(
                    sim::DVec3::new(0.0, 0.9, WALL_Z),
                    sim::Vec3::new(8.0, 0.9, 0.15),
                    0x0060_6060,
                ),
            )
            .unwrap();
    }
    let lamp = world.spawn();
    world
        .insert(
            lamp,
            Light::point(
                sim::DVec3::new(1.3, LAMP_HEIGHT, LAMP_Z),
                0x00ff_ffff,
                40.0,
                RANGE,
            ),
        )
        .unwrap();
    world
}

/// Straight down from above, so image rows are world `z`.
fn render(renderer: &mut OffscreenRenderer, world: &World) -> Vec<u8> {
    let view = View {
        pitch: -core::f32::consts::FRAC_PI_2,
        ..View::default()
    };
    let mut extracted = Extracted::default();
    extracted.clear(sim::DVec3::new(0.0, 9.0, 0.0), view.frustum(EXTENT));
    extracted.append_lights(world).unwrap();
    extracted.append::<Renderable>(world).unwrap();
    renderer
        .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])
        .unwrap()
        .pixels
}

/// Mean of the green channel over rows `rows`, ignoring the wall's own pixels.
fn brightness(pixels: &[u8], rows: core::ops::Range<u32>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0.0;
    for y in rows {
        for x in 0..EXTENT.0 {
            let i = ((y * EXTENT.0 + x) * 4 + 1) as usize;
            sum += f64::from(pixels[i]);
            count += 1.0;
        }
    }
    sum / count
}

/// Behind the wall, from the lamp's point of view.
///
/// The gap between this and [`LIT`] is the wall's own rows plus a margin, left
/// out of both so that neither region is measuring the blocker's lit top.
const SHADOWED: core::ops::Range<u32> = 150..250;
/// Between the lamp and the wall.
const LIT: core::ops::Range<u32> = 6..106;

#[test]
fn a_wall_between_a_lamp_and_a_floor_casts() {
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    cvars::LAMP_SHADOWS.set_bool(true);

    let open = render(&mut renderer, &world(false));
    let walled = render(&mut renderer, &world(true));
    cvars::LAMP_SHADOWS.set_bool(false);
    let unshadowed = render(&mut renderer, &world(true));
    cvars::LAMP_SHADOWS.set_bool(true);

    let (open_lit, open_far) = (brightness(&open, LIT), brightness(&open, SHADOWED));
    let (walled_lit, walled_far) = (brightness(&walled, LIT), brightness(&walled, SHADOWED));
    let unshadowed_far = brightness(&unshadowed, SHADOWED);
    println!(
        "open {open_lit:.1}/{open_far:.1}  walled {walled_lit:.1}/{walled_far:.1}  \
         unshadowed far {unshadowed_far:.1}"
    );

    // The control the whole test rests on: with the wall present but shadows
    // off, the far floor is as bright as it is with no wall at all. If this
    // fails, the far region is dark for a reason that is not the shadow —
    // framing, falloff, the wall occluding the *camera* — and the assertions
    // below would pass on that instead.
    assert!(
        (unshadowed_far - open_far).abs() < 1.0,
        "the wall changes the far floor with shadows off ({unshadowed_far:.1} against \
         {open_far:.1}) — this framing does not isolate the shadow",
    );
    assert!(
        open_far > 24.0,
        "the far floor is not lit to begin with ({open_far:.1}), so darkening it proves nothing",
    );

    // The shadow: the floor behind the wall loses most of its light.
    assert!(
        walled_far * 4.0 < open_far,
        "the floor behind the wall is {walled_far:.1} against {open_far:.1} unobstructed — the \
         lamp is lighting through the wall",
    );
    // And the other direction, which is what an over-shadowing lookup fails:
    // the floor *between* lamp and wall is untouched.
    assert!(
        walled_lit > open_lit * 0.95,
        "the floor in front of the wall dropped to {walled_lit:.1} from {open_lit:.1} — nothing \
         stands between it and the lamp, so this is the lamp shadowing itself",
    );
}
