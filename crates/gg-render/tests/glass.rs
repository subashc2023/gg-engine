//! The transparent pass (§6 M92): a `Renderable` with `transparency` above zero
//! is glass — blended over the finished opaque frame, absent from the depth
//! prepass and from every caster list, sorted back-to-front.
//!
//! Every claim here is graded in radiance (`capture_radiance`), because the
//! tonemapper's pedestal exaggerates a ratio near black and eight bits put a
//! floor under one (§6 M70) — and blending *is* a ratio.
//!
//! What each test falsifies:
//! - a pane at `transparency` 1.0 must leave the frame **bit-identical** while
//!   still declaring the pass — straight alpha at zero coverage is the identity,
//!   so any movement is the pass writing depth, blocking the sky, or shading
//!   something it should not;
//! - a pane at 0.5 must land on `0.5 * surface + 0.5 * behind` exactly (to f16),
//!   where both halves are read out of the two opaque renders — the blend
//!   equation asserted as arithmetic rather than as a look;
//! - a glass slab out of view must leave the whole frame alone while the same
//!   slab opaque moves it — the caster exclusion, falsified in both directions;
//! - two worlds spawning the same two overlapping panes in opposite orders must
//!   render identically — the back-to-front sort, which is what makes the blend
//!   a function of the world rather than of iteration order.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenFrame, OffscreenRenderer, View, cvars};

const EXTENT: (u32, u32) = (64, 36);
const CLEAR: [f32; 4] = [0.05, 0.06, 0.08, 1.0];
/// On the pane and on the wall behind it from the default eye.
const CENTER: (u32, u32) = (32, 18);

/// The backdrop every scene here shades against: a red wall filling the frame
/// behind everything, and a sun angled so nothing relevant shadows anything.
fn wall(world: &mut World) {
    let entity = world.spawn();
    world
        .insert(
            entity,
            Renderable::boxed(
                sim::DVec3::new(0.0, 0.0, -8.0),
                sim::Vec3::new(6.0, 6.0, 0.5),
                0x00c0_3030,
            ),
        )
        .unwrap();
}

fn lit_world() -> World {
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    world.register::<Light>().unwrap();
    let sun = world.spawn();
    world
        .insert(
            sun,
            Light::sun(sim::Vec3::new(-0.2, -0.6, -0.75), 0x00ff_f4e0, 3.2),
        )
        .unwrap();
    wall(&mut world);
    world
}

/// The pane in front of the wall, at `transparency`.
fn pane(world: &mut World, transparency: f32) {
    let entity = world.spawn();
    world
        .insert(
            entity,
            Renderable::boxed(
                sim::DVec3::new(0.0, 0.0, -4.0),
                sim::Vec3::new(2.0, 2.0, 0.1),
                0x00e8_f0ff,
            )
            .glazed(transparency),
        )
        .unwrap();
}

fn render(renderer: &mut OffscreenRenderer, world: &World) -> OffscreenFrame {
    let view = View::default();
    let mut extracted = Extracted::default();
    extracted.clear(sim::DVec3::ZERO, view.frustum(EXTENT));
    // Lights first, then the sweep, then the instances — the order
    // `cast_shadows` contracts, and what keeps an off-screen caster in the
    // array at all.
    extracted.append_lights(world).unwrap();
    extracted.cast_shadows(40.0);
    extracted.append::<Renderable>(world).unwrap();
    renderer.frame(&extracted, &view, CLEAR, &[]).unwrap()
}

fn radiance_at(frame: &OffscreenFrame, (x, y): (u32, u32)) -> [f32; 3] {
    let i = ((y * EXTENT.0 + x) * 4) as usize;
    [
        frame.radiance[i],
        frame.radiance[i + 1],
        frame.radiance[i + 2],
    ]
}

/// One renderer with the knobs this file depends on pinned: the field gathers
/// across frames (a converging picture is not a comparable one) and this file
/// compares frames.
fn renderer() -> OffscreenRenderer {
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    renderer.capture_radiance(true);
    cvars::GI.set_bool(false);
    renderer
}

fn done(renderer: OffscreenRenderer) {
    cvars::GI.set_bool(true);
    assert!(renderer.shutdown().clean());
}

#[test]
fn a_fully_transparent_pane_leaves_the_frame_alone_and_still_declares_the_pass() {
    let mut r = renderer();
    let bare = render(&mut r, &lit_world());
    assert!(
        !bare.order.iter().any(|p| *p == "forward-transparent"),
        "a frame with no glass declared the pass: {:?}",
        bare.order
    );
    let mut world = lit_world();
    pane(&mut world, 1.0);
    let glassed = render(&mut r, &world);
    let opaque_at = |o: &[String]| o.iter().position(|p| *p == "forward-opaque").unwrap();
    let post_at = |o: &[String]| o.iter().position(|p| *p == "post").unwrap();
    let at = glassed
        .order
        .iter()
        .position(|p| *p == "forward-transparent")
        .expect("a frame holding glass declares the pass");
    assert!(
        opaque_at(&glassed.order) < at && at < post_at(&glassed.order),
        "glass out of place: {:?}",
        glassed.order
    );
    // Straight alpha at zero coverage is `dst`, exactly — so any difference is
    // the pass doing something other than blending: writing depth, occluding
    // the sky draw, or landing in a caster list.
    assert_eq!(
        bare.radiance, glassed.radiance,
        "an invisible pane moved the frame"
    );
    done(r);
}

#[test]
fn glass_lands_exactly_between_its_surface_and_what_stands_behind() {
    let mut r = renderer();
    let behind = render(&mut r, &lit_world());
    let mut opaque = lit_world();
    pane(&mut opaque, 0.0);
    let surface = render(&mut r, &opaque);
    let mut glassed = lit_world();
    pane(&mut glassed, 0.5);
    let blended = render(&mut r, &glassed);

    let b = radiance_at(&behind, CENTER);
    let s = radiance_at(&surface, CENTER);
    let g = radiance_at(&blended, CENTER);
    println!("behind {b:?}  surface {s:?}  blended {g:?}");
    // The framing check: if the wall and the lit pane read the same, the assert
    // below would pass with the blend broken in either direction.
    assert!(
        (0..3).any(|c| (b[c] - s[c]).abs() > 0.05),
        "the two extremes agree, so this scene cannot grade a blend"
    );
    for c in 0..3 {
        let expected = 0.5 * s[c] + 0.5 * b[c];
        // f16 storage plus one blend's rounding; the claim is the equation, not
        // a vibe, so the tolerance is the format's and no wider.
        assert!(
            (g[c] - expected).abs() < 2e-3 + expected * 5e-3,
            "channel {c}: blended {} against {expected} (surface {}, behind {})",
            g[c],
            s[c],
            b[c]
        );
    }
    done(r);
}

#[test]
fn glass_out_of_view_casts_nothing_and_the_same_slab_opaque_casts() {
    // High above the frustum and forward of the wall, angled sun: the slab is
    // in no picture, and its shadow would land square in one.
    let slab = |world: &mut World, transparency: f32| {
        let entity = world.spawn();
        world
            .insert(
                entity,
                Renderable::boxed(
                    sim::DVec3::new(0.0, 4.0, -2.0),
                    sim::Vec3::new(1.5, 0.25, 1.5),
                    0x00ff_ffff,
                )
                .glazed(transparency),
            )
            .unwrap();
    };
    let mut r = renderer();
    let bare = render(&mut r, &lit_world());
    let mut shadowed = lit_world();
    slab(&mut shadowed, 0.0);
    let opaque = render(&mut r, &shadowed);
    assert!(
        !opaque.order.iter().any(|p| *p == "forward-transparent"),
        "an opaque slab reached the glass pass"
    );
    // The falsification arm: this slab's shadow reaches the wall, or the glass
    // half below is vacuously green.
    assert_ne!(
        bare.radiance, opaque.radiance,
        "the slab shadows nothing from here, so the test has no subject"
    );
    let mut glassed = lit_world();
    slab(&mut glassed, 0.5);
    let glass = render(&mut r, &glassed);
    // Out of view and casting nothing, the slab's existence is invisible: not a
    // shadow texel, not a depth write, not a probe face. The whole frame, not a
    // pixel — the cascade fit is view-derived, so nothing else may move either.
    assert_eq!(
        bare.radiance, glass.radiance,
        "a glass slab out of view still reached some pass"
    );
    done(r);
}

#[test]
fn spawn_order_does_not_choose_the_blend_order() {
    // Two overlapping panes at different depths. `over` does not commute, so
    // without the back-to-front sort the spawn order would pick the picture.
    let two = |near_first: bool| {
        let mut world = lit_world();
        let near = (sim::DVec3::new(0.0, 0.0, -3.0), 0x0030_60ff);
        let far = (sim::DVec3::new(0.0, 0.0, -5.0), 0x00ff_d040);
        for (position, color) in match near_first {
            true => [near, far],
            false => [far, near],
        } {
            let entity = world.spawn();
            world
                .insert(
                    entity,
                    Renderable::boxed(position, sim::Vec3::new(2.0, 2.0, 0.1), color).glazed(0.5),
                )
                .unwrap();
        }
        world
    };
    let mut r = renderer();
    let ab = render(&mut r, &two(true));
    let ba = render(&mut r, &two(false));
    assert_eq!(
        ab.radiance, ba.radiance,
        "two spawn orders of one world drew two pictures — the sort is not deciding"
    );
    // And both panes are in the picture: the stack reads differently from the
    // wall alone and from either pane alone, or the assert above compared two
    // copies of nothing.
    let mut one = lit_world();
    pane(&mut one, 0.5);
    let single = render(&mut r, &one);
    assert_ne!(ab.radiance, render(&mut r, &lit_world()).radiance);
    assert_ne!(ab.radiance, single.radiance);
    done(r);
}

#[test]
fn r_glass_off_is_the_world_before_the_field_existed() {
    let mut r = renderer();
    let mut opaque = lit_world();
    pane(&mut opaque, 0.0);
    let before = render(&mut r, &opaque);
    let mut glassed = lit_world();
    pane(&mut glassed, 0.5);
    cvars::GLASS.set_bool(false);
    let off = render(&mut r, &glassed);
    cvars::GLASS.set_bool(true);
    assert!(
        !off.order.iter().any(|p| *p == "forward-transparent"),
        "r.glass 0 still declared the pass"
    );
    // Filed back under opaque: prepass, forward, casters, all of it — not a
    // world with a hole where the pane was.
    assert_eq!(
        before.radiance, off.radiance,
        "r.glass 0 is not the pre-M92 world"
    );
    done(r);
}

#[test]
fn the_pass_survives_msaa_and_still_blends() {
    let mut r = renderer();
    cvars::MSAA.set_int(4);
    let bare = render(&mut r, &lit_world());
    let mut world = lit_world();
    pane(&mut world, 0.5);
    let glassed = render(&mut r, &world);
    cvars::MSAA.set_int(1);
    assert!(
        glassed.order.iter().any(|p| *p == "forward-transparent"),
        "no glass pass under MSAA: {:?}",
        glassed.order
    );
    // The resolve moved to the transparent pass (§6 M92): what post sampled has
    // the pane in it, and the clean shutdown below is the validation layer
    // agreeing nobody resolved twice or stored a discarded attachment.
    assert_ne!(
        radiance_at(&bare, CENTER),
        radiance_at(&glassed, CENTER),
        "the blend never reached the resolved frame under MSAA"
    );
    done(r);
}
