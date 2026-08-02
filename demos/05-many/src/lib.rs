//! Demo 05 — **the field** (§6 M10): ten thousand objects, a hundred hubs, and
//! not one transform composed in this file.
//!
//! Demo 04 put pack content on the screen, one entity for a whole scene. This is
//! the other axis: many entities, few meshes, and a parent link between them.
//! Each hub carries a [`Model`] it moves itself; each of its hundred children
//! carries a [`Node`] naming the hub and an offset in the hub's frame. What
//! turns that into a world position is `gg-scene`, host-side, inside the tick
//! and before the canonical hash — so a child's position is hashed state that
//! this crate never writes.
//!
//! # What "dirty" means when you can see it
//!
//! Only every other hub spins. The still ones' children are untouched from tick
//! to tick, and the propagation pass skips their composes entirely — so the work
//! is proportional to what *moved*, not to what exists. The `freeze` verb takes
//! that to its limit: with nothing spinning, ten thousand parented objects cost
//! one scan and zero quaternion multiplies. That is the claim §4.7 makes, and
//! [`Hierarchy::propagate`](../gg_scene/struct.Hierarchy.html#method.propagate)
//! returns the numbers that show it.
//!
//! # Four meshes, ten thousand instances
//!
//! Children cycle the pack's four meshes, so the renderer's sort key has more
//! than one material to order and its batcher has more than one run to build
//! (§6 M10). What reaches the device is four draws — the count is a property of
//! the *content*, not of the entity count, which is the whole point of batching.

use gg_ecs::boundary::{
    ActionId, AxisId, BoundaryError, Eye, GameWorld, Light, Model, Node, log_level,
};
use gg_ecs::{Component, Entity};
use gg_math::sim;

/// The pack's four meshes, as `ggc` names them: the source's path under the
/// asset root, then what it produced (§4.6 — names come from glTF's indices,
/// never from its strings).
pub const MESHES: [&str; 4] = [
    "field/mesh/0.0",
    "field/mesh/1.0",
    "field/mesh/2.0",
    "field/mesh/3.0",
];

/// Hubs across the field. A square number, because they are laid out in a grid.
pub const HUBS: usize = 100;

/// Children each hub carries. `HUBS * PER_HUB` is the ten thousand of §6 M10's
/// exit row, and the hubs themselves are the change.
pub const PER_HUB: usize = 100;

/// Hubs per row of the grid.
pub const GRID: usize = 10;

/// Metres between hubs.
pub const HUB_SPACING: f64 = 12.0;

/// Radius of a hub's ring of children, metres.
pub const RING_RADIUS: f64 = 4.0;

/// Rings a hub's children are stacked into.
pub const RINGS: usize = 4;

/// Metres between rings.
pub const RING_RISE: f64 = 1.1;

/// Radians a spinning hub turns per tick. Small and irrational-ish, so the
/// hierarchy is never accidentally back at a pose it held before.
pub const SPIN_PER_TICK: f64 = 0.013;

/// Metres per tick at full deflection. Faster than demo 04's: the field is a
/// hundred metres across.
pub const MOVE_PER_TICK: f64 = 0.35;

/// Radians per unit of pointer motion.
pub const LOOK_PER_UNIT: f32 = 0.0035;

/// Pitch stops short of the pole, where the view basis collapses.
pub const PITCH_LIMIT: f32 = 1.55;

/// Where a session opens: outside the field, looking into it, so a good share
/// of the objects start behind the camera and the culler has something to do.
pub const START_POSITION: sim::DVec3 = sim::DVec3::new(-14.0, 9.0, 78.0);

/// The demo's one light: a sun, warm, angled so a face pointing up and a face
/// pointing at the camera are lit differently (§6 M11).
///
/// Exported rather than buried in `bootstrap` because `gg-golden` builds this
/// demo's world from its constants: a sun the harness restated would be a sun
/// that could drift from the one a player sees.
pub const SUN_DIRECTION: sim::Vec3 = sim::Vec3::new(-0.38, -0.84, -0.39);
/// The sun's colour, `0x00RRGGBB`.
pub const SUN_COLOR: u32 = 0x00ff_f0d4;
/// The sun's radiance multiplier.
pub const SUN_INTENSITY: f32 = 3.5;

/// Stop and start the spin.
pub const FREEZE: ActionId = ActionId::new(0);
/// Strafe.
pub const MOVE_RIGHT: AxisId = AxisId::new(0);
/// Rise and fall.
pub const MOVE_UP: AxisId = AxisId::new(1);
/// Forward and back.
pub const MOVE_FORWARD: AxisId = AxisId::new(2);
/// Yaw, from pointer motion.
pub const AIM_X: AxisId = AxisId::new(3);
/// Pitch, from pointer motion.
pub const AIM_Y: AxisId = AxisId::new(4);

/// Where the camera is and what it is doing.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo05.observer")]
#[repr(C)]
pub struct Observer {
    /// World-space eye position, `f64` and never narrowed on this side of the
    /// membrane (§4.2.1).
    pub position: sim::DVec3,
    /// Rotation about +Y, radians.
    pub yaw: f32,
    /// Rotation about the observer's right axis, radians.
    pub pitch: f32,
    /// Non-zero while the field is frozen.
    pub frozen: u32,
    /// Padding, named — `Pod` refuses the anonymous kind.
    pub _pad: u32,
}

/// One hub: where it is in its own rotation, and whether it turns at all.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo05.hub")]
#[repr(C)]
pub struct Hub {
    /// Current angle about +Y, radians.
    pub angle: f64,
    /// Radians per tick. Zero for the still half, which is what makes the
    /// hierarchy's dirty set worth measuring.
    pub spin: f64,
}

/// Forward and right, in sim space. `f64` trig through `libm` (§4.2.1).
#[must_use]
pub fn basis(observer: &Observer) -> (sim::DVec3, sim::DVec3) {
    let (sin_yaw, cos_yaw) = sim::sin_cos(f64::from(observer.yaw));
    let (sin_pitch, cos_pitch) = sim::sin_cos(f64::from(observer.pitch));
    let forward = sim::DVec3::new(-cos_pitch * sin_yaw, sin_pitch, -cos_pitch * cos_yaw);
    let right = sim::DVec3::new(cos_yaw, 0.0, -sin_yaw);
    (forward, right)
}

/// Where hub `index` stands, centred on the origin.
#[must_use]
pub fn hub_position(index: usize) -> sim::DVec3 {
    let span = (GRID - 1) as f64 * HUB_SPACING * 0.5;
    let (row, column) = (index / GRID, index % GRID);
    sim::DVec3::new(
        column as f64 * HUB_SPACING - span,
        0.0,
        row as f64 * HUB_SPACING - span,
    )
}

/// Where child `slot` sits in its hub's frame, and which mesh it draws.
///
/// The offset is in the *hub's* rotated frame, which is what makes a spinning
/// hub swing its ring rather than slide it — the composition `gg-scene` applies
/// and this crate never writes.
#[must_use]
pub fn child_placement(slot: usize) -> (sim::DVec3, usize) {
    let ring = slot % RINGS;
    let step = slot / RINGS;
    let per_ring = PER_HUB / RINGS;
    let turn = core::f64::consts::TAU * step as f64 / per_ring as f64;
    let (sin, cos) = sim::sin_cos(turn);
    // Rings alternate radius slightly so the stack reads as a shape rather than
    // as four identical circles.
    let radius = RING_RADIUS + (ring % 2) as f64 * 0.8;
    let offset = sim::DVec3::new(cos * radius, ring as f64 * RING_RISE + 0.6, sin * radius);
    (offset, slot % MESHES.len())
}

/// Report a refused query and carry on — see demo 03 for why this is not a
/// panic: the host would report the panic and hide the load-sequence bug
/// standing behind it.
fn report(world: &GameWorld, outcome: Result<(), BoundaryError>) {
    if let Err(refused) = outcome {
        world.log(log_level::ERROR, &refused.to_string());
    }
}

/// Spawn the observer, the hubs and their children if there is no observer yet.
///
/// Idempotent by asking the world rather than by remembering, because
/// remembering would be state outliving a tick (§4.2.2, and a grep gate).
pub fn bootstrap(world: &mut GameWorld) {
    let mut exists = false;
    let outcome = world.each::<&Observer>(|_, _| exists = true);
    report(world, outcome);
    if exists {
        return;
    }

    let observer = world.spawn();
    if let Err(refused) = world.insert(
        observer,
        Observer {
            position: START_POSITION,
            yaw: 0.0,
            pitch: -0.12,
            frozen: 0,
            _pad: 0,
        },
    ) {
        world.log(log_level::ERROR, &refused.to_string());
        return;
    }

    // The sun. One directional light, so ten thousand objects are lit rather
    // than merely present (§6 M11) — and so the shadow pass has ten thousand
    // casters, which is the scale question a lit frame actually asks.
    let sun = world.spawn();
    let outcome = world.insert(sun, Light::sun(SUN_DIRECTION, SUN_COLOR, SUN_INTENSITY));
    report(world, outcome);

    for index in 0..HUBS {
        let hub = world.spawn();
        // The hub's own `Model` is its transform: it has no parent, so nothing
        // derives it and this crate owns it outright.
        let outcome = world.insert(hub, Model::at(MESHES[0], hub_position(index)));
        report(world, outcome);
        let outcome = world.insert(
            hub,
            Hub {
                angle: 0.0,
                spin: if index % 2 == 0 { SPIN_PER_TICK } else { 0.0 },
            },
        );
        report(world, outcome);

        for slot in 0..PER_HUB {
            let (offset, mesh) = child_placement(slot);
            let child = world.spawn();
            // Position lives in the `Node`. The child's `Model` position is
            // written by the host every tick a parent moves, so anything this
            // crate put there would be overwritten — which is why it passes the
            // origin and means it.
            let outcome = world.insert(child, Node::at(hub, offset));
            report(world, outcome);
            let outcome = world.insert(child, Model::at(MESHES[mesh], sim::DVec3::ZERO));
            report(world, outcome);
        }
    }
    world.log(log_level::INFO, "the field is up");
}

/// Turn.
pub fn aim(world: &mut GameWorld) {
    let (x, y) = (world.axis(AIM_X), world.axis(AIM_Y));
    let outcome = world.each::<&mut Observer>(|_, observer| {
        observer.yaw -= x * LOOK_PER_UNIT;
        observer.pitch = (observer.pitch - y * LOOK_PER_UNIT).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    });
    report(world, outcome);
}

/// Walk. Not normalized: diagonals are faster, which is a fly camera's
/// traditional bug and this demo's traditional feature.
pub fn walk(world: &mut GameWorld) {
    let axes = [
        f64::from(world.axis(MOVE_RIGHT)),
        f64::from(world.axis(MOVE_UP)),
        f64::from(world.axis(MOVE_FORWARD)),
    ];
    let outcome = world.each::<&mut Observer>(|_, observer| {
        let (forward, right) = basis(observer);
        observer.position = observer.position
            + right * (axes[0] * MOVE_PER_TICK)
            + sim::DVec3::Y * (axes[1] * MOVE_PER_TICK)
            + forward * (axes[2] * MOVE_PER_TICK);
    });
    report(world, outcome);
}

/// Toggle the spin.
pub fn freeze(world: &mut GameWorld) {
    if !world.just_pressed(FREEZE) {
        return;
    }
    let outcome = world.each::<&mut Observer>(|_, observer| {
        observer.frozen ^= 1;
    });
    report(world, outcome);
}

/// Advance the hubs, and only the hubs.
///
/// Writing a hub's `Model` rotation is the *whole* of what moves ten thousand
/// children: their offsets never change, and the host recomposes exactly the
/// subtrees whose parent's transform differs from last tick's.
pub fn spin(world: &mut GameWorld) {
    let mut frozen = false;
    let outcome = world.each::<&Observer>(|_, observer| frozen = observer.frozen != 0);
    report(world, outcome);
    if frozen {
        return;
    }
    let outcome = world.each::<(&mut Hub, &mut Model)>(|_, (hub, model)| {
        if hub.spin == 0.0 {
            return;
        }
        hub.angle += hub.spin;
        model.rotation = sim::DQuat::from_axis_angle(sim::DVec3::Y, hub.angle);
    });
    report(world, outcome);
}

/// Fill the render protocol from the sim: where the eye is. Last in the table,
/// so it sees this tick's movement.
pub fn present(world: &mut GameWorld) {
    let mut seen: Option<(Entity, Observer)> = None;
    let outcome = world.each::<&Observer>(|entity, observer| seen = Some((entity, *observer)));
    report(world, outcome);
    let Some((entity, observer)) = seen else {
        return;
    };
    let outcome = world.insert(
        entity,
        Eye {
            position: observer.position,
            yaw: observer.yaw,
            pitch: observer.pitch,
        },
    );
    report(world, outcome);
}

// Order in both verb lists is the id space (§4.7); order in `systems` is the
// execution order (§4.1). Neither is alphabetical and neither may drift.
//
// Behind `game` so a harness can link this crate's layout functions without its
// boundary symbols — see the manifest for why that is not optional.
#[cfg(feature = "game")]
gg_ecs::gg_game! {
    components: [Observer, Hub, Model, Node, Light, Eye],
    actions: ["freeze"],
    axes: ["move_right", "move_up", "move_forward", "aim_x", "aim_y"],
    systems: [bootstrap, aim, walk, freeze, spin, present],
}
