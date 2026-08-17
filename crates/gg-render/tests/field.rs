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
use gg_ecs::boundary::{Light, Renderable};
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

/// A corridor long enough that its grid cannot span it, walked (§6 M68).
///
/// Every scene this suite and the golden suite hold *fits*, so the anchored
/// placement and the wrap the shader has to undo to read a record are dead code in
/// every other gate. This is the one that executes them, and its claim is
/// **a field walked into place is the field gathered there.**
///
/// **What it does not prove, stated rather than implied.** It compares two renders
/// through the same shader, so a *consistent* indexing error is invisible to it: the
/// wrap was shifted a whole plane along x on purpose and this test stayed green,
/// because both renderers then read every record one cell over and still agreed with
/// each other. Nothing that shares the lookup can catch that — only a reference
/// outside it, which is the golden suite — the same falsification moves **ten** of
/// its scenes, checked rather than assumed. So the division is: this gates the scroll, the goldens gate the
/// indexing, and neither substitutes for the other.
///
/// A scroll keeps the records of every probe that stayed and re-gathers only the
/// plane that entered, so once both are converged the two renderers hold the same
/// records for the same places and must produce the same picture. If the wrap is off
/// by a plane the walked renderer is reading each probe's record at another probe's
/// position, which is a wrong field with no symptom until something in the level
/// happens to be there — so equality here is the gate, and a tolerance rather than
/// bytes only because the two got their records in a different order.
#[test]
fn a_field_walked_along_a_corridor_is_the_field_gathered_where_it_stopped() {
    cvars::AMBIENT.set_float(AMBIENT);
    cvars::DITHER.set_float(0.0);
    cvars::GI_RATE.set_int(0);
    // Long on x, so x anchors and y and z stay fitted to the corridor: the mixed
    // case, which is what a level actually is.
    let long = 160.0;
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    world.register::<Light>().unwrap();
    let mut add = |position: sim::DVec3, half: sim::Vec3, ink: u32| {
        let entity = world.spawn();
        world
            .insert(
                entity,
                Renderable::boxed(position, half, ink).surfaced(0.0, 0.0),
            )
            .unwrap();
    };
    add(
        sim::DVec3::new(0.0, -0.25, 0.0),
        sim::Vec3::new(long, 0.25, 4.0),
        0x00cc_cccc,
    );
    add(
        sim::DVec3::new(0.0, 2.0, -4.0),
        sim::Vec3::new(long, 2.0, 0.25),
        0x00cc_2020,
    );
    add(
        sim::DVec3::new(0.0, 2.0, 4.0),
        sim::Vec3::new(long, 2.0, 0.25),
        0x0020_20cc,
    );
    // No ceiling: the sun below has to *reach* a surface or the iteration above
    // decays to zero anyway, and a roofed corridor 320 m long lets in nothing at all
    // — adding the light under a lid changed this picture by zero bytes.
    // **A light, and the reason is a property of the field rather than of this
    // corridor.** `fs_probe` shades through `ambient_light`, which the field
    // *replaces* rather than adds to, so a sweep gathers a scene lit by the previous
    // sweep and each one multiplies by the albedo. With `r.ambient` as the only
    // source that iteration decays toward **zero** — the walked renderer here read a
    // mean of 7.2 against the fresh one's 31.7 for no reason but having run three
    // hundred sweeps against nine. A sun is what gives the map a fixed point above
    // zero, and only a fixed point is comparable between two renderers that arrived
    // by different routes.
    let sun = world.spawn();
    world
        .insert(
            sun,
            Light::sun(sim::Vec3::new(-0.3, -1.0, -0.2), 0x00ff_f4e0, 3.0),
        )
        .unwrap();

    let look = |renderer: &mut OffscreenRenderer, at: sim::DVec3| -> Vec<u8> {
        let view = View {
            pitch: -0.15,
            ..View::default()
        };
        let mut extracted = Extracted::default();
        extracted.clear(at, view.frustum(EXTENT));
        extracted.append::<Renderable>(&world).unwrap();
        extracted.append_lights(&world).unwrap();
        renderer
            .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])
            .unwrap()
            .pixels
    };
    // Enough frames for the round robin to reach every slot, which is all a step
    // along the walk needs.
    let gather = |renderer: &mut OffscreenRenderer, at: sim::DVec3| {
        for _ in 0..FRAMES * 8 {
            look(renderer, at);
            if renderer.field_pending().0 == 0 {
                return;
            }
        }
        panic!("the field did not gather at {at:?}");
    };
    // **A fixed point, not a full round robin** — and the difference is a property of
    // the field worth stating: `fs_probe` shades through `ambient_light`, which
    // samples the field, so each sweep lights the next one and the records are an
    // *iteration* whose limit is the multi-bounce answer. `pending == 0` therefore
    // means "every probe has a record", not "the records have stopped moving", and
    // two renderers that have gathered the same probes a different number of times
    // hold different fields. Comparing anything but the limit compares bounce counts.
    let settle = |renderer: &mut OffscreenRenderer, at: sim::DVec3| -> Vec<u8> {
        let mut previous = look(renderer, at);
        for _ in 0..FRAMES * 16 {
            let next = look(renderer, at);
            let moved = previous
                .iter()
                .zip(&next)
                .filter(|(a, b)| a.abs_diff(**b) > 1)
                .count();
            previous = next;
            if renderer.field_pending().0 == 0 && moved == 0 {
                return previous;
            }
        }
        panic!("the field did not settle at {at:?}");
    };

    // Walked: converge at the start, then step along the corridor a metre at a
    // time, converging at each stop so a scroll's new plane is gathered before the
    // next one.
    let start = sim::DVec3::new(-30.0, 1.4, 0.0);
    let end = sim::DVec3::new(0.0, 1.4, 0.0);
    let mut walked = OffscreenRenderer::new(EXTENT).unwrap();
    gather(&mut walked, start);
    let anchored = walked
        .field_anchored()
        .expect("the corridor produced no grid at all");
    assert_eq!(
        anchored,
        [true, false, false],
        "this corridor was supposed to anchor its long axis and fit the other two — \
         without that the walk below never scrolls and the test proves nothing"
    );
    for step in 1..=30 {
        let at = sim::DVec3::new(start.x + f64::from(step), start.y, start.z);
        gather(&mut walked, at);
    }
    let walked_pixels = settle(&mut walked, end);
    let (refits, scrolls) = walked.field_events();
    assert_eq!(
        refits, 1,
        "walking a corridor threw the field away {refits} times — one is the first fit"
    );
    assert!(
        scrolls > 0,
        "thirty metres along an anchored axis produced no scroll at all"
    );

    // Gathered: a renderer that was never anywhere else.
    let mut fresh = OffscreenRenderer::new(EXTENT).unwrap();
    let fresh_pixels = settle(&mut fresh, end);

    // The two grids must be the same window before the pictures are worth
    // comparing — otherwise this passes on two renderers that disagree about where
    // the field is and happen to look alike.
    assert_eq!(
        walked.field_grid(),
        fresh.field_grid(),
        "the walked grid is not the grid a renderer starting here would build"
    );
    // **A tolerance and not bytes, and the reason is the iteration above.** The two
    // renderers reach the same limit from different distances, so what is left is how
    // far each still is from it — three code values out of 255 here, against the
    // 4.4x difference in the mean that a field reading its records at the wrong
    // places produced while this test was being built. A plane of wrap error
    // misplaces a sixteenth of the records, which is a gross change in a corridor and
    // nothing like this residue.
    let worst = walked_pixels
        .iter()
        .zip(&fresh_pixels)
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    let (walked_mean, fresh_mean) = (mean(&walked_pixels), mean(&fresh_pixels));
    assert!(
        worst <= 8,
        "a walked field differs from the field gathered in place by {worst} code values —          the storage rotation and the probe positions disagree"
    );
    assert!(
        (walked_mean - fresh_mean).abs() < fresh_mean * 0.01,
        "a walked field is {walked_mean:.2} against {fresh_mean:.2} gathered in place"
    );

    cvars::DITHER.set_float(1.0);
    cvars::GI_RATE.set_int(16);
    for (name, renderer) in [("walked", walked), ("fresh", fresh)] {
        let report = renderer.shutdown();
        assert!(report.clean(), "unclean {name} render: {report:?}");
    }
}
