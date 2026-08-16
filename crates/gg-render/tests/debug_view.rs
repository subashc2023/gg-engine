//! The gate for §6 M59's debug view: **every view shows something, and off
//! shows the picture.**
//!
//! `gg-tools views` is where the pictures are; this asserts the three things
//! that would otherwise regress invisibly, and they fail in different
//! directions:
//!
//! - **A view that resolved declares its pass and changes the frame.** A blit
//!   whose pass is declared and whose draw list is empty produces the picture
//!   unchanged, which reads as "that view is off" rather than as a bug — nobody
//!   files it, they type the next index.
//! - **Off is off.** Index 0 and an index past the table must both render the
//!   frame the renderer would have rendered without this milestone, byte for
//!   byte. This is what keeps a debug feature out of every blessed golden.
//! - **The depth-derived views work with `r.ao 0`.** They read the camera's
//!   depth buffer, which is allocated *sampleable* only because the occlusion
//!   pass wants it — so the configuration where AO is off is exactly the one
//!   where they would silently show nothing, and it is also the configuration
//!   somebody debugging AO reaches for first.
//!
//! Not a golden, and deliberately: a reference image of the occlusion buffer is
//! a reference of a debugging aid, and re-blessing thirteen of them on three
//! backends every time a pass moves is a cost with no reader.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

/// Small: what is being asserted is which image reached the screen, and that is
/// as true at 160x90 as at 1080p.
const EXTENT: (u32, u32) = (160, 90);

const EYE: sim::DVec3 = sim::DVec3::new(0.0, 1.4, 4.2);
const PITCH: f32 = -0.30;

/// A floor, a wall and a box on the floor: enough that depth, normals and
/// occlusion all have something to say, and that a cascade is fitted so
/// `shadow.0` resolves to a real image.
fn world() -> World {
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    world.register::<Light>().unwrap();
    for (position, half_extent) in [
        (
            sim::DVec3::new(0.0, -0.5, 0.0),
            sim::Vec3::new(6.0, 0.5, 6.0),
        ),
        (
            sim::DVec3::new(0.0, 1.5, -2.5),
            sim::Vec3::new(6.0, 2.0, 0.5),
        ),
        (
            sim::DVec3::new(-0.4, 0.4, -0.8),
            sim::Vec3::new(0.4, 0.4, 0.5),
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
        .insert(
            sun,
            Light::sun(sim::Vec3::new(-0.4, -1.0, -0.35), 0x00ff_f4e0, 3.0),
        )
        .unwrap();
    world
}

fn view() -> View {
    View {
        pitch: PITCH,
        ..View::default()
    }
}

/// One frame at the current CVars, as pixels and the pass names that produced
/// them.
fn render(renderer: &mut OffscreenRenderer, world: &World) -> (Vec<u8>, Vec<String>) {
    let view = view();
    let mut extracted = Extracted::default();
    extracted.clear(EYE, view.frustum(EXTENT));
    extracted.append_lights(world).unwrap();
    extracted.cast_shadows(view.caster_reach(EXTENT));
    extracted.append::<Renderable>(world).unwrap();
    let frame = renderer
        .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])
        .unwrap();
    (frame.pixels, frame.order)
}

/// Render until two consecutive frames agree, and return that frame.
///
/// The probe field accumulates over `probes / r.gi_rate` frames (§6 M36), so a
/// baseline taken early differs from every later frame for a reason that is not
/// this test's — and the difference would read as "the debug view changed the
/// picture" on whichever view happened to be next.
fn settle(renderer: &mut OffscreenRenderer, world: &World) -> (Vec<u8>, Vec<String>) {
    let mut previous = render(renderer, world);
    for _ in 0..200 {
        let current = render(renderer, world);
        if current.0 == previous.0 {
            return current;
        }
        previous = current;
    }
    panic!("the frame never stopped changing with `r.debug_view 0` — nothing below is comparable");
}

/// Every view the frame has resolves, declares its pass, and changes the
/// picture; every view it has not leaves the picture alone.
///
/// **`r.gi 0`, and that is a finding rather than a convenience** (§6 M59). With
/// the field on, this scene's frame never reaches a fixed point: `settle` finds
/// two consecutive frames that agree and a later one differs by a code value.
/// The cause is structural — a gathering probe samples the field (§6 M36's
/// second bounce) and the round robin re-gathers a *group* per frame, so the
/// sequence has a limit cycle as long as the robin rather than a limit. It is
/// below `gg-tools field`'s measurement floor in a real scene, which is why M57
/// read zero swing on all three of its legs and this 160x90 frame does not. The
/// two field views keep their coverage in
/// [`the_field_views_resolve_when_it_is_gathering`], which asks the question
/// that does not need a stationary frame.
#[test]
fn each_view_that_resolves_replaces_the_picture() {
    let world = world();
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    cvars::GI.set_bool(false);
    cvars::DEBUG_VIEW.set_int(0);
    let (off, order) = settle(&mut renderer, &world);
    assert!(
        !order.iter().any(|pass| pass == "debug-view"),
        "`r.debug_view 0` declares no pass at all: {order:?}"
    );

    let mut resolved = 0;
    for (index, name) in cvars::DEBUG_VIEWS.iter().enumerate().skip(1) {
        cvars::DEBUG_VIEW.set_int(index as i64);
        let (pixels, order) = render(&mut renderer, &world);
        if !order.iter().any(|pass| pass == "debug-view") {
            // A view of a pass this frame did not run. Legal, and then the
            // picture must be untouched rather than partly overwritten.
            assert_eq!(pixels, off, "{name} declared no pass but changed the frame");
            continue;
        }
        resolved += 1;
        assert_ne!(
            pixels, off,
            "{name} declared `debug-view` and produced the picture unchanged — a pass whose \
             draw list is empty looks exactly like a view that is switched off"
        );
    }
    // A frame with a floor, a box and a sun has depth, normals, occlusion, a
    // cascade and the field: a run where nothing resolved would pass every
    // assertion above and prove nothing (§5.8's rule).
    assert!(
        resolved >= 6,
        "only {resolved} view(s) resolved against a frame that has depth, occlusion, a cascade \
         and the field — the resolution table has stopped finding this frame's images"
    );

    // Off by construction, and off by a number nobody declared.
    for index in [0, cvars::DEBUG_VIEWS.len() as i64, -1, 9999] {
        cvars::DEBUG_VIEW.set_int(index);
        let (pixels, _) = render(&mut renderer, &world);
        assert_eq!(
            pixels, off,
            "`r.debug_view {index}` is not a view, so it must render the picture"
        );
    }
    cvars::DEBUG_VIEW.set_int(0);
    cvars::GI.set_bool(true);
    let report = renderer.shutdown();
    assert!(
        report.clean(),
        "no leaks, no validation messages: {report:?}"
    );
}

/// The field's two views, which the test above turns `r.gi` off for.
///
/// A weaker assertion on purpose — that each declares its pass and submits
/// clean, not that it changed a frame nothing here can hold still. What it
/// catches is the failure that actually happens: a field image renamed or
/// re-ordered so the resolution table stops finding it, which would leave two
/// entries in the picker that quietly do nothing.
#[test]
fn the_field_views_resolve_when_it_is_gathering() {
    let world = world();
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    cvars::GI.set_bool(true);
    for (index, name) in [(11, "field"), (12, "moments")] {
        assert_eq!(
            cvars::DEBUG_VIEWS.get(index as usize),
            Some(&name),
            "the view table moved under this test"
        );
        cvars::DEBUG_VIEW.set_int(index);
        let (_, order) = render(&mut renderer, &world);
        assert!(
            order.iter().any(|pass| pass == "debug-view"),
            "{name} resolved to nothing in a frame that gathers: {order:?}"
        );
    }
    cvars::DEBUG_VIEW.set_int(0);
    let report = renderer.shutdown();
    assert!(
        report.clean(),
        "no leaks, no validation messages: {report:?}"
    );
}

/// The depth and normal views with `r.ao 0`.
///
/// The camera's depth buffer is allocated `SAMPLED` only because the occlusion
/// pass reads it, so with AO off there is nothing for these two to sample unless
/// the frame asks for one on their behalf — and a frame that did not would show
/// them blank on exactly the configuration an operator switches to while
/// investigating AO.
#[test]
fn the_depth_views_work_with_occlusion_off() {
    let world = world();
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    cvars::AO.set_bool(false);
    cvars::DEBUG_VIEW.set_int(0);
    let (off, order) = settle(&mut renderer, &world);
    assert!(
        !order.iter().any(|pass| pass == "ao"),
        "`r.ao 0` declares no occlusion pass: {order:?}"
    );

    for (index, name) in [(2, "depth"), (3, "normal")] {
        assert_eq!(
            cvars::DEBUG_VIEWS.get(index as usize),
            Some(&name),
            "the view table moved under this test"
        );
        cvars::DEBUG_VIEW.set_int(index);
        let (pixels, order) = render(&mut renderer, &world);
        assert!(
            order.iter().any(|pass| pass == "debug-view"),
            "{name} declared no pass with `r.ao 0` — the sampleable depth was not allocated"
        );
        assert_ne!(pixels, off, "{name} rendered the picture unchanged");
    }

    cvars::DEBUG_VIEW.set_int(0);
    cvars::AO.set_bool(true);
    let report = renderer.shutdown();
    assert!(
        report.clean(),
        "no leaks, no validation messages: {report:?}"
    );
}
