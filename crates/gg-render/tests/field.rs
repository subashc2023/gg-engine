//! The gate for §6 M36's irradiance field: **the term must change the picture,
//! and `r.gi 0` must take it back off — however long it ran.**
//!
//! `gg-tools bounce` is where the numbers live: what fraction of a room's light
//! bounced, how far the field lands from the paths, and the leak-against-loss
//! plateau `r.gi_spacing` and `r.gi_moments` are read off. This asserts the two
//! things that must never regress silently, and they fail in opposite
//! directions:
//!
//! - **The field darkens an enclosed room.** A field that agrees with the flat
//!   ambient everywhere costs three passes and does nothing, and no image gate
//!   would object — a slightly different frame is what a re-bless is for.
//! - **Switching it off is the pre-M36 renderer.** Not "close to it": the same
//!   bytes. `Probes::refit` stops the *gathering*, which is invisible to a
//!   session that already fitted a grid, so the switch has to be read where the
//!   frame block is written as well. It was not, and a field gathered before the
//!   CVar went off kept lighting every frame after — found by `gg-tools bounce`,
//!   whose field table grades a ratio of these two renders and read exactly 1
//!   everywhere.
//!
//! The control under both is the *first* frame, rendered before any probe has
//! been gathered. A test that compared the switched-off frame only against
//! itself would pass against a renderer whose switch does nothing at all.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::Renderable;
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

const EXTENT: (u32, u32) = (192, 108);

/// Linear ambient, below the tonemapper's knee — `tests/ao.rs`'s constant and
/// its reasoning. With no sky declared it is also the whole of the pre-M36
/// ambient term, which is what the field replaces.
const AMBIENT: f64 = 0.25;

const EYE: sim::DVec3 = sim::DVec3::new(0.0, 1.4, 4.0);

/// Frames the field is given to converge. At `r.gi_rate 0` a frame gathers one
/// batch, and this scene's grid is well inside eight of them.
const FRAMES: usize = 24;

/// A floor and three walls: a room with one open side, which is the least a
/// bounce needs to be worth measuring. The side walls are saturated on purpose —
/// a grey room's field differs from a flat ambient only by its occlusion, and
/// this test wants the colour term in the picture too.
fn scene() -> Vec<(sim::DVec3, sim::Vec3, u32)> {
    vec![
        (
            sim::DVec3::new(0.0, -0.25, 0.0),
            sim::Vec3::new(4.0, 0.25, 4.0),
            0x00cc_cccc,
        ),
        (
            sim::DVec3::new(0.0, 2.0, -4.0),
            sim::Vec3::new(4.0, 2.0, 0.25),
            0x00cc_cccc,
        ),
        (
            sim::DVec3::new(-4.0, 2.0, 0.0),
            sim::Vec3::new(0.25, 2.0, 4.0),
            0x00cc_2020,
        ),
        (
            sim::DVec3::new(4.0, 2.0, 0.0),
            sim::Vec3::new(0.25, 2.0, 4.0),
            0x0020_20cc,
        ),
        (
            sim::DVec3::new(0.0, 4.0, 0.0),
            sim::Vec3::new(4.0, 0.25, 4.0),
            0x00cc_cccc,
        ),
    ]
}

fn world() -> World {
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    for (position, half_extent, color) in scene() {
        let entity = world.spawn();
        // Fully rough dielectric, and no sky and no lights anywhere: the only
        // light in this room is the ambient term and whatever bounced off it.
        world
            .insert(
                entity,
                Renderable::boxed(position, half_extent, color).surfaced(0.0, 0.0),
            )
            .unwrap();
    }
    world
}

fn frame(renderer: &mut OffscreenRenderer, world: &World) -> Vec<u8> {
    let view = View {
        pitch: -0.15,
        ..View::default()
    };
    let mut extracted = Extracted::default();
    extracted.clear(EYE, view.frustum(EXTENT));
    extracted.append::<Renderable>(world).unwrap();
    extracted.append_lights(world).unwrap();
    renderer
        .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])
        .unwrap()
        .pixels
}

fn mean(pixels: &[u8]) -> f64 {
    let total: u64 = pixels
        .chunks_exact(4)
        .map(|p| u64::from(p[0]) + u64::from(p[1]) + u64::from(p[2]))
        .sum();
    total as f64 / (pixels.len() / 4).max(1) as f64 / 3.0
}

#[test]
fn the_field_darkens_an_enclosed_room_and_switching_it_off_gives_the_frame_back() {
    cvars::AMBIENT.set_float(AMBIENT);
    // A ± one code value pattern would break the byte comparison below for a
    // reason that has nothing to do with the field.
    cvars::DITHER.set_float(0.0);
    // As many probes a frame as one batch holds: this is a gate, not a session,
    // and what it wants is the converged field in the fewest frames.
    cvars::GI_RATE.set_int(0);
    let world = world();
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();

    // Before anything has been gathered — the pre-M36 renderer by construction.
    cvars::GI.set_bool(false);
    let flat = frame(&mut renderer, &world);

    cvars::GI.set_bool(true);
    let mut lit = frame(&mut renderer, &world);
    for _ in 0..FRAMES {
        if renderer.field_pending().0 == 0 {
            break;
        }
        lit = frame(&mut renderer, &world);
    }
    let (pending, probes) = renderer.field_pending();
    assert_eq!(pending, 0, "the field did not converge: {probes} probes");

    // A room open on one side, lit by a uniform ambient: every probe inside it
    // sees walls where the flat term assumed sky, so the field is darker. The
    // margin is wide because the failure this guards is a term that does
    // *nothing*, not one that drifts.
    let (before, after) = (mean(&flat), mean(&lit));
    assert!(
        after < before * 0.95,
        "the field did not darken an enclosed room: {before:.2} -> {after:.2}"
    );

    // And off is off. Bytes, not a tolerance: `r.gi 0` is either the renderer
    // that shipped before M36 or it is a second approximation of it.
    cvars::GI.set_bool(false);
    let again = frame(&mut renderer, &world);
    let differing = again.iter().zip(&flat).filter(|(a, b)| a != b).count();
    assert_eq!(
        differing, 0,
        "a gathered field kept lighting the frame with r.gi off: {differing} bytes differ"
    );

    // Off is not a clear, either: the grid and its records outlive the toggle,
    // so turning the term back on costs the frame it costs and not the field.
    cvars::GI.set_bool(true);
    assert_eq!(
        renderer.field_pending().0,
        0,
        "switching the field off threw away what it had gathered"
    );

    cvars::DITHER.set_float(1.0);
    cvars::GI_RATE.set_int(16);
    let report = renderer.shutdown();
    assert!(report.clean(), "unclean render: {report:?}");
}
