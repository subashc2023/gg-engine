//! The gate for §6 M60: **a blocker above a near cascade still shadows.**
//!
//! A cascade is an orthographic slab with a light eye some distance up-light of
//! its centre, and a caster past that eye is not in the map. The shader reads
//! absent depth as absent blocker, so the receiver renders *lit* — which is the
//! one shadow bug that does not look like a shadow bug. Until M60 that distance
//! came from the cascade's own width, so it was shortest exactly where the
//! cascade was tightest.
//!
//! Two assertions, in opposite directions, and the second is what makes the
//! first mean anything:
//!
//! - the floor under a high blocker is **dark**, at a range where the near
//!   cascade is metres wide and the blocker is tens of metres up;
//! - with `r.shadow_reach 0` — the pre-M60 fit, one flag away in the same binary
//!   (`r.shadow_cull`'s argument, §6 M32) — the same floor is **lit**. Without
//!   this arm the test would pass on a renderer that shadowed the floor for some
//!   other reason, and on one where nothing casts at all.
//!
//! No demo could have carried it. `gg-tools shadow-reach` drops five casters out
//! of demo 12's near cascade at the shipped range and the picture does not move
//! by one code value, because a low open room has nothing high enough over the
//! near band to matter. It took Sponza's gallery, where the drop is 12 % of the
//! frame.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

/// Small on purpose: what is asserted is a luminance, and it is as true at
/// 160x90 as at 1080p.
const EXTENT: (u32, u32) = (160, 90);

/// Close to the floor, so the patch under the blocker lands in cascade 0 — the
/// tightest one, which is where the old fit was shortest.
const EYE: sim::DVec3 = sim::DVec3::new(0.0, 0.9, 2.6);
const PITCH: f32 = -0.34;

/// Straight down, so the blocker's shadow lands directly under it and no part of
/// this test depends on where a slanted sun would put it.
const SUN: sim::Vec3 = sim::Vec3::new(0.0, -1.0, 0.0);

/// Metres up. Far enough above the floor that the pre-M60 near cascade — whose
/// eye sat two of its own radii up-light, and it is metres wide — cannot hold
/// it, and well inside the shared reach, which is the widest cascade's.
const BLOCKER_HEIGHT: f64 = 30.0;

/// A floor and one wide slab high above it. Nothing else: a second caster would
/// make it possible for the floor to be dark for a reason this test is not
/// about.
fn world() -> World {
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    world.register::<Light>().unwrap();
    for (position, half_extent) in [
        (
            sim::DVec3::new(0.0, -0.5, 0.0),
            sim::Vec3::new(20.0, 0.5, 20.0),
        ),
        (
            sim::DVec3::new(0.0, BLOCKER_HEIGHT, 0.0),
            sim::Vec3::new(8.0, 0.5, 8.0),
        ),
    ] {
        let entity = world.spawn();
        world
            .insert(
                entity,
                Renderable::boxed(position, half_extent, 0x00ff_ffff).surfaced(0.0, 0.0),
            )
            .unwrap();
    }
    let sun = world.spawn();
    world
        .insert(sun, Light::sun(SUN, 0x00ff_ffff, 4.0))
        .unwrap();
    world
}

/// Mean luminance of the bottom third of the frame — floor, and nothing else at
/// this framing. The blocker is thirty metres up and out of shot.
fn floor_luminance(renderer: &mut OffscreenRenderer, world: &World) -> f64 {
    let view = View {
        pitch: PITCH,
        ..View::default()
    };
    // Twice: the probe field carries state across frames, so the first after a
    // CVar move is a picture of the transition (§6 M57).
    let mut pixels = Vec::new();
    for _ in 0..2 {
        let mut extracted = Extracted::default();
        extracted.clear(EYE, view.frustum(EXTENT));
        extracted.append_lights(world).unwrap();
        extracted.cast_shadows(view.caster_reach(EXTENT));
        extracted.append::<Renderable>(world).unwrap();
        pixels = renderer
            .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])
            .unwrap()
            .pixels;
    }
    let start = (EXTENT.1 as usize * 2 / 3) * EXTENT.0 as usize * 4;
    let rows = &pixels[start..];
    let sum: f64 = rows
        .chunks_exact(4)
        .map(|p| 0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2]))
        .sum();
    sum / (rows.len() / 4) as f64
}

#[test]
fn a_blocker_above_the_near_cascade_still_shadows() {
    let world = world();
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();

    cvars::SHADOW_REACH.set_bool(true);
    let shadowed = floor_luminance(&mut renderer, &world);
    cvars::SHADOW_REACH.set_bool(false);
    let lit = floor_luminance(&mut renderer, &world);
    cvars::SHADOW_REACH.set_bool(true);

    assert!(renderer.shutdown().clean());
    // Wide apart rather than merely ordered: a sun this bright against a shadow
    // lit by ambient alone is a factor, and a threshold that only asked for
    // `<` would pass on a one-code-value difference that no player could see.
    assert!(
        lit > shadowed * 2.0,
        "the pre-M60 fit was supposed to lose this blocker and light the floor: reach on {shadowed:.1}, \
         off {lit:.1} — if these agree, nothing here is casting and the assertion below is vacuous"
    );
    assert!(
        shadowed < 40.0,
        "a floor under a solid slab, lit by a sun straight overhead, is not {shadowed:.1} bright"
    );
}
