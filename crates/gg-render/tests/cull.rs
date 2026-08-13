//! The two halves of §6 M32's claim, which need two different kinds of gate.
//!
//! **The picture must not move.** Culling a shadow view is a pure performance
//! change: every pixel it alters is a shadow it lost. That half is gated by the
//! golden suite, which compares twenty-two blessed references byte for byte —
//! and by [`the_cull_does_not_change_what_is_drawn`] here, which renders one
//! frame with `r.shadow_cull` on and off and compares the two.
//!
//! **The cull must actually cull.** No image gate can see this half: a cull that
//! quietly began letting everything through renders the *identical* picture and
//! only costs more, so a suite of blessed references stays green while the
//! milestone silently reverts. That is what [`gg_render::ShadowCull`] is for and
//! what the rest of this file asserts against — the numbers, not the pixels.
//!
//! The scene is boxes rather than pack meshes because both passes cull now, and
//! a world declared in thirty lines is a better subject than one needing a
//! `.ggpack` on disk to exist at all.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, ShadowCull, View, cvars};

const EXTENT: (u32, u32) = (320, 180);

const EYE: [f64; 3] = [0.0, 3.0, 10.0];
const PITCH: f32 = -0.2;

/// A long row of posts under a sun, with one short-range lamp near the camera.
///
/// Long **on purpose**: the row runs 45 m each way down `x`, so no one cascade
/// holds it and no one lamp face sees much of it — which is what gives the cull
/// something to reject. A scene fitting inside every view would pass the
/// assertions below with the cull switched off, which is the way this test could
/// quietly stop testing anything.
fn world() -> World {
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    world.register::<Light>().unwrap();

    let floor = world.spawn();
    world
        .insert(
            floor,
            Renderable::boxed(
                sim::DVec3::new(0.0, -0.1, 0.0),
                sim::Vec3::new(50.0, 0.1, 10.0),
                0x0088_8c90,
            ),
        )
        .unwrap();
    for i in 0..60 {
        let post = world.spawn();
        world
            .insert(
                post,
                Renderable::boxed(
                    sim::DVec3::new((i as f64 - 30.0) * 1.5, 0.8, 0.0),
                    sim::Vec3::new(0.3, 0.8, 0.3),
                    0x00c0_a070,
                ),
            )
            .unwrap();
    }
    let sun = world.spawn();
    world
        .insert(
            sun,
            Light::sun(sim::Vec3::new(-0.35, -1.0, -0.25), 0x00ff_f4e0, 3.4),
        )
        .unwrap();
    let lamp = world.spawn();
    world
        .insert(
            lamp,
            Light::point(sim::DVec3::new(1.0, 2.0, 4.0), 0x00ff_d8a0, 24.0, 6.0),
        )
        .unwrap();
    world
}

/// One frame, and what the cull came to rendering it. The instance count comes
/// back too: it is the denominator the accounting assertion needs, and reading
/// it off extract rather than off the world is what keeps the assertion true
/// when extract's own frustum cull drops a post.
fn render(renderer: &mut OffscreenRenderer, world: &World) -> (Vec<u8>, ShadowCull, u32) {
    let view = View {
        pitch: PITCH,
        ..View::default()
    };
    let eye = sim::DVec3::new(EYE[0], EYE[1], EYE[2]);
    let mut extracted = Extracted::default();
    extracted.clear(eye, view.frustum(EXTENT));
    extracted.append_lights(world).unwrap();
    extracted.cast_shadows(view.caster_reach(EXTENT));
    extracted.append::<Renderable>(world).unwrap();
    let instances = extracted.instances.len() as u32;
    let pixels = renderer
        .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])
        .unwrap()
        .pixels;
    (pixels, renderer.shadow_cull(), instances)
}

#[test]
fn the_cull_rejects_most_of_a_row_no_view_can_hold() {
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    cvars::SHADOW_CULL.set_int(1);
    cvars::LAMP_SHADOWS.set_bool(true);
    let (_, cull, instances) = render(&mut renderer, &world());
    renderer.shutdown();
    println!("{cull:?} over {instances} instances");

    // Two controls, without which everything below can pass on a frame that is
    // not testing anything: there have to be views to cull against, and
    // instances to cull.
    assert!(cull.views >= 4, "too few shadow views: {cull:?}");
    assert!(instances >= 20, "too few instances: {instances}");

    // The milestone, as a number. Before §6 M32 this was zero by construction —
    // every batch went into every view — and a regression switching the cull
    // back off returns it to zero while every blessed reference stays green.
    assert!(cull.rejected > 0, "the cull rejected nothing: {cull:?}");

    // And it is the *majority*, not a rounding: a row 90 m long against four
    // cascades and the six faces of a 6 m lamp is mostly out of view.
    assert!(
        cull.rejected > cull.drawn,
        "the cull kept more than it dropped: {cull:?}"
    );

    // Nothing invented or lost on the way: every (instance, view) pair is
    // accounted for exactly once. This is what catches a counter double-counting
    // a compacted batch, which would otherwise make `rejected` look healthy
    // while the cull did nothing at all.
    assert_eq!(
        cull.drawn + cull.rejected,
        instances * cull.views,
        "{cull:?} against {instances} instances"
    );
}

#[test]
fn the_cull_does_not_change_what_is_drawn() {
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    cvars::LAMP_SHADOWS.set_bool(true);
    let world = world();
    // The field held still, and this is the one place in the suite where that is
    // not a convenience: it is **stateful across frames** (`r.gi_rate` gathers a
    // few probes a frame), so two renders taken one after the other are two
    // different fields, and a test whose whole assertion is byte equality would
    // be grading the round robin. Off rather than converged — nothing here is
    // about the field, and `probe::Probes::field` makes off mean off (§6 M36).
    cvars::GI.set_bool(false);

    cvars::SHADOW_CULL.set_int(1);
    let (culled, on, _) = render(&mut renderer, &world);
    cvars::SHADOW_CULL.set_int(0);
    let (whole, off, _) = render(&mut renderer, &world);
    cvars::SHADOW_CULL.set_int(1);
    cvars::GI.set_bool(true);
    renderer.shutdown();

    // The off switch means what it says: everything, in every view, uncompacted.
    // Without this the comparison below could be two identical culled runs.
    assert_eq!(off.rejected, 0, "the off switch still culled: {off:?}");
    assert_eq!(off.dropped, 0, "the off switch still dropped: {off:?}");
    assert!(on.rejected > 0, "the on switch culled nothing: {on:?}");

    // And the two frames are one frame. Byte equality rather than a tolerance,
    // because the geometry reaching each shadow map is a *subset* chosen by the
    // cull — the depths written are the same depths, or the cull dropped a
    // caster that mattered.
    assert_eq!(culled.len(), whole.len(), "two renders disagreed on size");
    let differing = culled.iter().zip(&whole).filter(|(a, b)| a != b).count();
    assert_eq!(
        differing, 0,
        "{differing} bytes moved when the cull came on"
    );
}
