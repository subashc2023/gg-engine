//! The claim that lets §6 M30 raise `MAX_POINT` from 32 to 256 without moving a
//! blessed pixel: **a fragment reading its own froxel's light list gets the same
//! answer, bit for bit, as one looping the whole frame's.**
//!
//! It rests on two things and neither can be checked by reading the code. The
//! shader's point falloff is exactly zero past `Light::range`, so a light the
//! old loop reached and rejected added `0.0`; and `cluster::Assignment` may
//! over-include but must never under-include, so nothing that would have added
//! more than zero is missing from a run. Together those make the froxel's sum a
//! *subsequence* of the frame's, in the same order, with the omitted terms all
//! exactly zero — which is equality and not approximation, and is therefore
//! testable by comparing bytes rather than by a tolerance.
//!
//! `r.clusters 0` is what the other leg renders with, and it is deliberately not
//! a second code path: it is the assignment answering "every light, every
//! froxel", so the shader, the buffer and the draw are identical between the two
//! runs and the only difference is the contents of three thousand runs.
//!
//! The occupancy this scene actually produces is `gg-tools lights`' business.
//! What is here is the equality, and the control that proves the scene has
//! enough lights for the equality to be worth anything.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

/// Small: this compares whole frames rather than measuring anything in them, and
/// two renders of a hundred lights is what the test costs.
const EXTENT: (u32, u32) = (320, 180);

/// Lamps down a hall. Well past the 32 that was the cap before this milestone,
/// and spread through depth as well as across the screen so froxels differ from
/// screen tiles.
const LAMPS: usize = 96;

const EYE: [f64; 3] = [0.0, 1.6, 6.0];

fn world() -> World {
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    world.register::<Light>().unwrap();
    let floor = world.spawn();
    world
        .insert(
            floor,
            Renderable::boxed(
                sim::DVec3::new(0.0, -0.1, -20.0),
                sim::Vec3::new(12.0, 0.1, 40.0),
                0x0088_8c90,
            ),
        )
        .unwrap();
    // Two walls, so a light near one of them lights something the camera can
    // see it fail to light. A hall of lamps over an empty floor would grade the
    // floor alone.
    for side in [-1.0, 1.0] {
        let wall = world.spawn();
        world
            .insert(
                wall,
                Renderable::boxed(
                    sim::DVec3::new(side * 6.0, 2.0, -20.0),
                    sim::Vec3::new(0.2, 2.0, 40.0),
                    0x0090_8c84,
                ),
            )
            .unwrap();
    }
    for i in 0..LAMPS {
        let side = if i.is_multiple_of(2) { -1.0 } else { 1.0 };
        let z = -1.0 - (i / 2) as f64 * 1.2;
        let lamp = world.spawn();
        // A short range on purpose: the whole point is that most of these reach
        // no part of most froxels. Ranges that spanned the hall would put every
        // light in every list and the two legs would agree for the wrong reason
        // — which is what the occupancy control below refuses.
        world
            .insert(
                lamp,
                Light::point(
                    sim::DVec3::new(side * 4.5, 1.4 + (i % 3) as f64 * 0.4, z),
                    match i % 3 {
                        0 => 0x00ff_d0a0,
                        1 => 0x00a0_d0ff,
                        _ => 0x00d0_ffb0,
                    },
                    6.0,
                    3.0,
                ),
            )
            .unwrap();
    }
    world
}

fn render(renderer: &mut OffscreenRenderer, world: &World) -> Vec<u8> {
    let view = View {
        pitch: -0.15,
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
    assert_eq!(
        extracted.lights.len(),
        LAMPS,
        "the frustum culled lamps this framing was built to keep",
    );
    renderer
        .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])
        .unwrap()
        .pixels
}

#[test]
fn a_froxel_list_shades_exactly_as_the_whole_frame_did() {
    let world = world();
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    // The field held still, and this is not a convenience: it is **stateful
    // across frames** (`r.gi_rate` gathers a few probes a frame), so two renders
    // taken one after the other are two different fields and a test whose whole
    // assertion is byte equality would be grading the round robin. Off rather
    // than converged — nothing here is about the field, and
    // `probe::Probes::field` makes off mean off (§6 M36).
    cvars::GI.set_bool(false);

    cvars::CLUSTERS.set_bool(true);
    let clustered = render(&mut renderer, &world);
    // Read between the two renders: the second one's assignment is the
    // every-light-everywhere value, and reporting *that* as the occupancy would
    // be the control agreeing with itself.
    let load = renderer.cluster_load();
    cvars::CLUSTERS.set_bool(false);
    let looped = render(&mut renderer, &world);
    assert_eq!(
        renderer.cluster_load().worst,
        LAMPS as u32,
        "`r.clusters 0` is supposed to be every light in every froxel",
    );
    cvars::CLUSTERS.set_bool(true);
    cvars::GI.set_bool(true);

    // Two controls first, because equality is exactly what a grid that let
    // everybody into every froxel would also produce, and what an unlit framing
    // would produce as well.
    let mean = looped.chunks_exact(4).map(|p| u32::from(p[0])).sum::<u32>() as f64
        / (EXTENT.0 * EXTENT.1) as f64;
    assert!(
        (12.0..200.0).contains(&mean),
        "mean red {mean:.1} — this framing is not lit by its lamps, so the comparison below \
         grades nothing",
    );
    println!("{LAMPS} lamps: {load:?}");
    assert_eq!(load.dropped, 0);
    assert!(
        load.worst * 3 < LAMPS as u32,
        "the busiest froxel holds {} of {LAMPS} — this grid is not selecting, so the equality \
         below is the trivial one",
        load.worst,
    );

    let differing = clustered
        .chunks_exact(4)
        .zip(looped.chunks_exact(4))
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing, 0,
        "{differing} pixels differ between a froxel's list and the whole frame's — assignment \
         under-included, which is the one failure §6 M30 has no tolerance for",
    );
}
