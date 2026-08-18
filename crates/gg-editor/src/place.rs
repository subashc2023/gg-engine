//! Where a thing with no geometry *is* (§6 M72).
//!
//! The viewport could outline, pick and drag exactly one component:
//! [`Renderable`]. Everything else the boundary carries is a placement too — a
//! lamp is at a point and reaches so far, an environment volume is a box — and
//! an operator who cannot see either is reading coordinates out of the
//! inspector and guessing. The report that started this said so directly: *I
//! can't tell where the light is supposed to be and where its not.*
//!
//! **A `Renderable` is the carrier, not the source.** Every kind below resolves
//! to one, so [`crate::marker`]'s outline, [`crate::pick`]'s ray and the
//! gizmo's drag are the code they already were and there is no second
//! projection to keep in step. What each kind *means* by the box differs, and
//! that is the whole of [`Kind`]: a lamp's half-extent is its **range**, so the
//! shape an operator drags is the reach itself rather than a decoration sized
//! by taste.
//!
//! [`put`] is the other half and is why this is a module rather than a match in
//! `marker.rs`: a drag resolves to a box, and which fields of which component
//! that box lands in is a fact about the kind. A tool a kind has no field for is
//! **refused** ([`Kind::takes`]) rather than silently dropped — a gizmo that
//! draws an arm and does nothing when it is pulled is worse than one that draws
//! two arms.

use gg_ecs::boundary::{Light, Renderable, Sky, light};
use gg_math::sim;

use crate::Tool;

/// The least a lamp's range may be dragged to, metres. A zero-range light
/// contributes nothing anywhere and is invisible in the picture *and* in the
/// marker, which is one drag too far from being unreachable.
const LEAST_RANGE: f32 = 0.1;

/// What the marker draws where a directional light *is*, since it is nowhere:
/// the arrow's length in metres at the world origin. A screen-constant length
/// would be the editor's habit and is wrong here — a sun's arrow is the one
/// marker whose job is to sit in the scene and be compared against the shadows
/// it casts.
pub(crate) const SUN_ARM: f64 = 3.0;

/// Which component a placement came out of — what the marker draws, and what a
/// drag writes back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// A [`Renderable`]. The box is the thing itself.
    Body,
    /// A [`light::POINT`]. The centre is the lamp and the half-extent is its
    /// [`Light::range`], the same number on all three axes.
    Lamp,
    /// A [`light::DIRECTIONAL`]. Nowhere at all — the box is degenerate at the
    /// world origin and exists so the arrow has somewhere to start.
    Sun,
    /// A bounded [`Sky`]. The box is the volume, which is exactly the shape
    /// `weight_at` tests against.
    Volume,
}

impl Kind {
    /// Whether a drag with `tool` has a field to land in.
    ///
    /// A lamp and a volume have no orientation — neither component carries a
    /// rotation — and a sun has no placement at all, so the gizmo offers two
    /// tools on the first two and none on the last.
    pub(crate) fn takes(self, tool: Tool) -> bool {
        match self {
            Kind::Body => true,
            Kind::Lamp | Kind::Volume => !matches!(tool, Tool::Turn),
            Kind::Sun => false,
        }
    }

    /// The word the drag's log line names it by, so a replayed session's log
    /// says which of the three a tick moved.
    pub(crate) fn label(self) -> &'static str {
        match self {
            Kind::Body => "body",
            Kind::Lamp => "lamp",
            Kind::Sun => "sun",
            Kind::Volume => "volume",
        }
    }
}

/// The placement `entity` declared, or `None` for a row with no shape at all.
///
/// **[`Renderable`] first**, and the order is the arbitration: an entity
/// carrying both a body and a light is a lamp *on* something, and what an
/// operator clicked is the thing they can see.
pub(crate) fn of(world: &gg_ecs::World, entity: gg_ecs::Entity) -> Option<(Kind, Renderable)> {
    if let Some(box_) = world.get::<Renderable>(entity) {
        return Some((Kind::Body, *box_));
    }
    if let Some(lamp) = world.get::<Light>(entity) {
        return Some(as_light(lamp));
    }
    world.get::<Sky>(entity).and_then(as_sky)
}

/// A light as a placement: a sphere of its range for a lamp, a degenerate point
/// at the origin for a sun.
fn as_light(lamp: &Light) -> (Kind, Renderable) {
    if lamp.kind == light::DIRECTIONAL {
        return (
            Kind::Sun,
            Renderable::boxed(sim::DVec3::ZERO, sim::Vec3::ZERO, lamp.color),
        );
    }
    (
        Kind::Lamp,
        Renderable::boxed(lamp.position, sim::Vec3::splat(lamp.range), lamp.color),
    )
}

/// A bounded sky as a placement, or `None` for the world's own — an unbounded
/// environment is everywhere, and a marker for it would be a box round the
/// whole level.
fn as_sky(sky: &Sky) -> Option<(Kind, Renderable)> {
    let half = sky.half_extent;
    (half.x > 0.0 || half.y > 0.0 || half.z > 0.0).then(|| {
        (
            Kind::Volume,
            Renderable::boxed(sky.center, half, sky.horizon),
        )
    })
}

/// Every placement in the world, in the order [`crate::pick::nearest`] wants to
/// consider them: bodies first, then lamps and volumes.
///
/// Three queries rather than one walk, because they are three archetype sets —
/// and `each_ref` is what the pick already uses, so the iteration order a tie is
/// broken by is the same order it always was.
pub(crate) fn each(world: &gg_ecs::World, on: &mut dyn FnMut(gg_ecs::Entity, Kind, Renderable)) {
    if let Ok(query) = gg_ecs::Query::<&Renderable>::new() {
        world.each_ref(&query, |entity, box_: &Renderable| {
            on(entity, Kind::Body, *box_);
        });
    }
    if let Ok(query) = gg_ecs::Query::<&Light>::new() {
        world.each_ref(&query, |entity, lamp: &Light| {
            // A body carrying a lamp is one entity with two placements, and
            // `of` already said the body wins — so this skips it rather than
            // offering the same row twice under two shapes.
            if world.get::<Renderable>(entity).is_some() {
                return;
            }
            let (kind, box_) = as_light(lamp);
            on(entity, kind, box_);
        });
    }
    if let Ok(query) = gg_ecs::Query::<&Sky>::new() {
        world.each_ref(&query, |entity, sky: &Sky| {
            if world.get::<Renderable>(entity).is_some() {
                return;
            }
            if let Some((kind, box_)) = as_sky(sky) {
                on(entity, kind, box_);
            }
        });
    }
}

/// Write a dragged placement back into whichever component it came from, along
/// `axis`. `true` if the world moved.
///
/// `axis` is the arm the drag resolved onto and is read by exactly one kind: a
/// lamp's range is one number and a box's half-extent is three, so which of the
/// three the operator pulled *is* the new range. Every other kind takes the
/// whole box.
pub(crate) fn put(
    world: &mut gg_ecs::World,
    entity: gg_ecs::Entity,
    kind: Kind,
    axis: usize,
    want: &Renderable,
) -> bool {
    match kind {
        Kind::Body => match world.get_mut::<Renderable>(entity) {
            Some(target) if *target != *want => {
                *target = *want;
                true
            }
            _ => false,
        },
        Kind::Lamp => {
            let reach = [want.half_extent.x, want.half_extent.y, want.half_extent.z][axis.min(2)]
                .max(LEAST_RANGE);
            match world.get_mut::<Light>(entity) {
                Some(lamp) if lamp.position != want.position || lamp.range != reach => {
                    lamp.position = want.position;
                    lamp.range = reach;
                    true
                }
                _ => false,
            }
        }
        Kind::Volume => match world.get_mut::<Sky>(entity) {
            Some(sky) if sky.center != want.position || sky.half_extent != want.half_extent => {
                sky.center = want.position;
                sky.half_extent = want.half_extent;
                true
            }
            _ => false,
        },
        // A sun is a direction. There is nowhere to put it, which is why
        // `takes` offers it no tool in the first place.
        Kind::Sun => false,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn world_with_a_lamp() -> (gg_ecs::World, gg_ecs::Entity) {
        let mut world = gg_ecs::World::new();
        world.register::<Renderable>().unwrap();
        world.register::<Light>().unwrap();
        world.register::<Sky>().unwrap();
        let entity = world.spawn();
        world
            .insert(
                entity,
                Light::point(sim::DVec3::new(1.0, 2.0, 3.0), 0x00ff_8800, 4.0, 6.0),
            )
            .unwrap();
        (world, entity)
    }

    #[test]
    fn a_lamps_placement_is_its_reach() {
        let (world, entity) = world_with_a_lamp();
        let (kind, box_) = of(&world, entity).expect("a lamp is somewhere");
        assert_eq!(kind, Kind::Lamp);
        assert_eq!(box_.position, sim::DVec3::new(1.0, 2.0, 3.0));
        // The half-extent *is* the range, which is what makes the outline the
        // answer to "where does this light stop".
        assert_eq!(box_.half_extent, sim::Vec3::splat(6.0));
    }

    #[test]
    fn a_drag_lands_in_the_field_it_came_from() {
        let (mut world, entity) = world_with_a_lamp();
        let (kind, was) = of(&world, entity).unwrap();

        let moved = Renderable {
            position: sim::DVec3::new(1.0, 5.0, 3.0),
            ..was
        };
        assert!(put(&mut world, entity, kind, 1, &moved));
        let lamp = *world.get::<Light>(entity).unwrap();
        assert_eq!(lamp.position, sim::DVec3::new(1.0, 5.0, 3.0));
        assert_eq!(lamp.range, 6.0, "a move is not a resize");

        // A resize reads the dragged axis and nothing else — the other two are
        // the range as it was, and averaging them in would make one arm move
        // the value by a third of the drag.
        let sized = Renderable {
            half_extent: sim::Vec3::new(6.0, 6.0, 9.0),
            ..moved
        };
        assert!(put(&mut world, entity, kind, 2, &sized));
        assert_eq!(world.get::<Light>(entity).unwrap().range, 9.0);

        // Idempotent: the gizmo writes every tick of a drag, and a write that
        // changed nothing must not count as an edit.
        assert!(!put(&mut world, entity, kind, 2, &sized));
    }

    #[test]
    fn a_sun_is_nowhere_and_takes_no_tool() {
        let mut world = gg_ecs::World::new();
        world.register::<Light>().unwrap();
        let entity = world.spawn();
        world
            .insert(
                entity,
                Light::sun(sim::Vec3::new(0.0, -1.0, 0.0), 0x00ff_ffff, 3.0),
            )
            .unwrap();
        let (kind, box_) = of(&world, entity).unwrap();
        assert_eq!(kind, Kind::Sun);
        assert_eq!(box_.half_extent, sim::Vec3::ZERO);
        for tool in [Tool::Move, Tool::Scale, Tool::Turn] {
            assert!(!kind.takes(tool), "{tool:?}");
        }
        assert!(!put(&mut world, entity, kind, 0, &box_));
    }

    #[test]
    fn the_worlds_own_sky_has_no_marker_and_a_room_does() {
        let mut world = gg_ecs::World::new();
        world.register::<Sky>().unwrap();
        let outdoors = world.spawn();
        world.insert(outdoors, Sky::daylight(1.0)).unwrap();
        let room = world.spawn();
        world
            .insert(
                room,
                Sky::daylight(0.3).within(
                    sim::DVec3::new(2.0, 1.0, 0.0),
                    sim::Vec3::new(3.0, 1.0, 2.0),
                    1.0,
                ),
            )
            .unwrap();

        assert!(of(&world, outdoors).is_none(), "everywhere is not a place");
        let (kind, box_) = of(&world, room).unwrap();
        assert_eq!(kind, Kind::Volume);
        assert_eq!(box_.half_extent, sim::Vec3::new(3.0, 1.0, 2.0));

        let mut seen = Vec::new();
        each(&world, &mut |entity, kind, _| seen.push((entity, kind)));
        assert_eq!(seen, vec![(room, Kind::Volume)]);
    }

    #[test]
    fn a_body_carrying_a_lamp_is_offered_once() {
        let mut world = gg_ecs::World::new();
        world.register::<Renderable>().unwrap();
        world.register::<Light>().unwrap();
        let entity = world.spawn();
        world
            .insert(
                entity,
                Renderable::boxed(sim::DVec3::ZERO, sim::Vec3::splat(0.5), 0x00ff_ffff),
            )
            .unwrap();
        world
            .insert(
                entity,
                Light::point(sim::DVec3::ZERO, 0x00ff_ffff, 1.0, 2.0),
            )
            .unwrap();

        assert_eq!(of(&world, entity).map(|(k, _)| k), Some(Kind::Body));
        let mut seen = Vec::new();
        each(&world, &mut |entity, kind, _| seen.push((entity, kind)));
        assert_eq!(seen, vec![(entity, Kind::Body)]);
    }
}
