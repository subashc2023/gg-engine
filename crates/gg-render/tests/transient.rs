//! §6 M37 item 3's renderer half: **what a transient that leaves the room does
//! to the two bounds this renderer computes per frame.**
//!
//! Demo 12 fires about 47 entities a second and reclaims the same, and every one
//! of them is spent inside the room's own static bounds — chips die *at* the
//! floor plane rather than sinking through it. The exception is the room being
//! closed on five sides and open to the sky: a shot aimed up meets no surface,
//! so its tracer reaches `SHOT_RANGE` through a ceiling that is not there, for
//! three ticks. That is the one transient in the ladder that leaves its scene,
//! and two per-frame bounds have an opinion about it.
//!
//! The two fail in opposite directions and are asserted that way:
//!
//! - **M32's batch bounds absorb it.** A tracer forty metres long is inside the
//!   one box batch, so the batch sphere swallows the room and every shadow view
//!   says `Partial` — the case the chunk level exists for. What must not happen
//!   is the *room* losing rejections it had: the transient may cost its own draw
//!   in a view and nothing else, and the count of rejected pairs must not fall.
//! - **M36's `Grid::fit` does not.** It is fitted to every visible instance, so
//!   a sliver of geometry 40 m up is scene bounds like any other: the spacing
//!   doubles to reach it, the grid is refitted, and a refit throws the gathered
//!   field away. Twice per shot — once when the tracer appears and once when it
//!   dies. That is the milestone's number and it is pinned here rather than
//!   described, because the day it is fixed this test is what says so.
//!
//! The control under both is the *level* shot, which is the other 99 % of them:
//! a tracer that ends on a wall changes neither bound and costs the field
//! nothing. Without it the sky assertions would pass against a renderer that
//! refitted on every frame for any reason at all.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, ShadowCull, View, cvars};

const EXTENT: (u32, u32) = (192, 108);

/// Demo 12's eye where it spawns, near enough: the muzzle every tracer below
/// leaves from.
const EYE: sim::DVec3 = sim::DVec3::new(0.0, 1.62, 8.0);

/// How far demo 12's bullet reaches (`SHOT_RANGE`), which is what a tracer with
/// nothing to stop it is drawn to.
const RANGE: f64 = 40.0;

/// Frames the field is given to converge before a transient is introduced.
const FRAMES: usize = 24;

/// Demo 12's room in miniature: a floor and four walls 4 m tall, inner faces at
/// ±12, open to the sky. Not the demo's own table — this is a renderer gate and
/// a game crate is not a dependency it may take — but its *shape*, because the
/// property under test is the ratio between a room and a shot that leaves it.
fn room() -> Vec<(sim::DVec3, sim::Vec3, u32)> {
    let mut boxes = vec![(
        sim::DVec3::new(0.0, -0.25, 0.0),
        sim::Vec3::new(12.5, 0.25, 12.5),
        0x00b4_aea2,
    )];
    for (offset, half) in [
        (
            sim::DVec3::new(0.0, 0.0, -12.25),
            sim::Vec3::new(12.5, 2.0, 0.25),
        ),
        (
            sim::DVec3::new(0.0, 0.0, 12.25),
            sim::Vec3::new(12.5, 2.0, 0.25),
        ),
        (
            sim::DVec3::new(-12.25, 0.0, 0.0),
            sim::Vec3::new(0.25, 2.0, 12.5),
        ),
        (
            sim::DVec3::new(12.25, 0.0, 0.0),
            sim::Vec3::new(0.25, 2.0, 12.5),
        ),
    ] {
        boxes.push((offset + sim::DVec3::new(0.0, 2.0, 0.0), half, 0x008f_9298));
    }
    // Enough small furniture for the batch to hold more than one chunk, which
    // is the level M32's cull actually rejects at.
    for i in 0..24 {
        let a = f64::from(i) * 0.7;
        boxes.push((
            sim::DVec3::new(gg_math::sim::cos(a) * 9.0, 0.5, gg_math::sim::sin(a) * 9.0),
            sim::Vec3::splat(0.5),
            0x00c0_9060,
        ));
    }
    boxes
}

/// Where a shot's tracer ends, given no target: on a wall, or 40 m up through
/// the ceiling this room does not have.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shot {
    None,
    Level,
    Sky,
}

fn world(shot: Shot) -> World {
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    for (position, half_extent, color) in room() {
        let entity = world.spawn();
        world
            .insert(
                entity,
                Renderable::boxed(position, half_extent, color).surfaced(0.0, 0.0),
            )
            .unwrap();
    }
    let sun = world.spawn();
    world
        .insert(
            sun,
            Light::sun(sim::Vec3::new(-0.4, -1.0, -0.3), 0x00ff_f4e0, 3.4),
        )
        .unwrap();
    for at in [
        sim::DVec3::new(-6.0, 3.4, -4.0),
        sim::DVec3::new(6.0, 3.4, 4.0),
    ] {
        let lamp = world.spawn();
        world
            .insert(lamp, Light::point(at, 0x00ff_c890, 14.0, 11.0))
            .unwrap();
    }
    // Demo 12's tracer: a long thin box lying on the ray, half its length from
    // the muzzle (`TRACER_HALF` for the thin pair).
    let beam = match shot {
        Shot::None => None,
        // Ends on the far wall, 20 m out and well inside the bounds.
        Shot::Level => Some((
            sim::DVec3::new(0.0, 1.62, -2.0),
            sim::Vec3::new(0.012, 0.012, 10.0),
        )),
        Shot::Sky => Some((
            sim::DVec3::new(0.0, 1.62 + RANGE / 2.0, 8.0),
            sim::Vec3::new(0.012, (RANGE / 2.0) as f32, 0.012),
        )),
    };
    if let Some((position, half_extent)) = beam {
        let entity = world.spawn();
        world
            .insert(
                entity,
                Renderable::boxed(position, half_extent, 0x00ff_e8b0),
            )
            .unwrap();
    }
    world
}

/// One frame, looking up the way a player taking a sky shot is.
fn frame(renderer: &mut OffscreenRenderer, world: &World) {
    let view = View {
        pitch: 0.6,
        ..View::default()
    };
    let mut extracted = Extracted::default();
    extracted.clear(EYE, view.frustum(EXTENT));
    extracted.append::<Renderable>(world).unwrap();
    extracted.append_lights(world).unwrap();
    renderer
        .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])
        .unwrap();
}

type Grid = Option<(sim::DVec3, f32, [u32; 3])>;

/// Render `shot` for one frame and report the two bounds afterwards, plus the
/// field's `(ungathered, probes)` — how much of it survived, against how much
/// there now is to gather.
fn shoot(renderer: &mut OffscreenRenderer, shot: Shot) -> (Grid, ShadowCull, (usize, usize)) {
    frame(renderer, &world(shot));
    (
        renderer.field_grid(),
        renderer.shadow_cull(),
        renderer.field_pending(),
    )
}

#[test]
fn a_shot_that_leaves_the_room_refits_the_field_and_the_shadow_cull_absorbs_it() {
    cvars::AMBIENT.set_float(0.25);
    // The whole field in one frame: this is a gate, and what it asks is whether
    // a frame threw the field away — not how many frames it takes to get back.
    cvars::GI_RATE.set_int(0);
    cvars::GI.set_bool(true);
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();

    let calm = world(Shot::None);
    // One batch, measured rather than quoted: `r.gi_rate 0` is "as many probes
    // as a batch holds", and what a refit costs below is stated in those.
    frame(&mut renderer, &calm);
    let opened = renderer.field_pending();
    let batch = opened.1 - opened.0;
    assert!(batch > 0, "the first frame gathered nothing at r.gi_rate 0");
    for _ in 0..FRAMES {
        if renderer.field_pending().0 == 0 {
            break;
        }
        frame(&mut renderer, &calm);
    }
    let (settled, quiet, pending) = (
        renderer.field_grid(),
        renderer.shadow_cull(),
        renderer.field_pending(),
    );
    assert_eq!(pending.0, 0, "the field never converged over the calm room");
    let probes = pending.1;
    assert!(
        settled.is_some(),
        "no grid was fitted to a room with a floor"
    );

    // The level shot: inside the bounds, so neither bound notices and the field
    // is untouched. The control — without it, everything below would pass
    // against a renderer that refitted every frame.
    let (grid, cull, left) = shoot(&mut renderer, Shot::Level);
    assert_eq!(
        grid, settled,
        "a tracer that ends on a wall refitted the field's grid"
    );
    assert_eq!(left.0, 0, "a tracer inside the room threw the field away");
    assert!(
        cull.rejected >= quiet.rejected,
        "the room lost shadow-view rejections to a transient inside it: {} -> {}",
        quiet.rejected,
        cull.rejected
    );

    // The sky shot, M32's half: the batch sphere now swallows a 40 m sliver and
    // every view says `Partial`, so what is left to check is that the *room*
    // kept its rejections and the beam paid only for itself — at most one draw
    // per view, which is what "the chunk and instance levels absorbed it" means
    // as a number.
    let (grid, cull, left) = shoot(&mut renderer, Shot::Sky);
    assert!(
        cull.rejected >= quiet.rejected,
        "the room lost shadow-view rejections to a shot that left it: {} -> {}",
        quiet.rejected,
        cull.rejected
    );
    assert!(
        cull.drawn <= quiet.drawn + cull.views,
        "one transient cost {} extra draws across {} views — the batch bounds did not absorb it",
        cull.drawn.saturating_sub(quiet.drawn),
        cull.views
    );

    // And M36's half, which does not absorb it. Both halves of the cost are
    // pinned: the grid is coarser than the one the room asked for, and the
    // gathered field is *gone* rather than stale.
    let (_, spacing, counts) = grid.expect("the sky shot left the renderer with no grid at all");
    let (_, was, had) = settled.unwrap();
    assert!(
        spacing > was,
        "the sky shot did not widen the grid ({was} -> {spacing}) — if `Grid::fit` no longer \
         follows a transient out of the room, this is the good news and §6 M37 item 3's number \
         is stale"
    );
    assert_eq!(
        (was, spacing, had, counts),
        (4.0, 8.0, [8, 4, 8], [5, 8, 5]),
        "the room or the fit moved — §6 M37 item 3 quotes these two grids"
    );
    // Everything the old grid had gathered is gone: what is left to gather is
    // the *new* grid entire, less the one batch this same frame took off it.
    // `Probes::update` keeps nothing across a refit, and cannot — a probe at a
    // new spacing is at a new place.
    assert_eq!(
        left,
        (
            counts.iter().product::<u32>() as usize - batch,
            counts.iter().product::<u32>() as usize
        ),
        "the refit kept part of the field it had gathered over {probes} probes"
    );

    // The tracer dies after three ticks and the bounds shrink back. `covers`
    // is the hysteresis that would keep the wide grid, and it does not fire:
    // the fitted spacing is different, so this is a *second* refit, and one
    // shot at the sky costs the field twice.
    let (grid, _, left) = shoot(&mut renderer, Shot::None);
    assert_eq!(
        grid, settled,
        "the grid did not come back to the room's own fit once the tracer died"
    );
    assert_eq!(
        left,
        (probes - batch, probes),
        "the second refit kept part of the field — one shot at the sky costs two full regathers \
         of {probes} probes, {batch} of them a frame"
    );

    cvars::GI_RATE.set_int(16);
    let report = renderer.shutdown();
    assert!(report.clean(), "unclean render: {report:?}");
}
