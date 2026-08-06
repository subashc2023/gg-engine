//! Demo 04 — **the hall** (§6 M9): walk around a scene that came out of a pack
//! file rather than out of this source.
//!
//! Demo 03 proved the cooperative loop over geometry the *game* spells out, one
//! `Renderable` at a time. This proves the other half: the floor, the pillars
//! and the wall below are in `assets/hall.gltf`, compiled by `ggc`, and the only
//! thing this crate says about them is their name and where to put them.
//!
//! # What a game knows about assets
//!
//! A name, and nothing else. §3's deny pin keeps `gg-assets` out of a game's
//! link set, so there is no pack type here, no load state, no handle — a
//! [`Model`] carries the hash of a name and the host resolves it, exactly as
//! [`Renderable`] carries a colour the host paints with. An asset that has not
//! finished streaming draws nothing for a frame or two and then appears; an
//! asset that does not exist draws nothing forever. Neither is an error a game
//! can see, which is what lets `ggc watch` rebuild the pack underneath a
//! session that is still running.
//!
//! # The tint verb earns its keep
//!
//! One action, cycling [`Model::tint`], because it is the smallest thing that
//! demonstrates the point of the protocol: how pack content *looks* is decided
//! by a reloadable system body in this file, not by the renderer. Edit
//! [`TINTS`], save, and the hall changes colour without the session restarting.

use gg_ecs::boundary::{ActionId, AxisId, Eye, GameWorld, Light, Model, Renderable, log_level};
use gg_ecs::{Component, Entity};
use gg_math::sim;

/// The pack asset this demo draws — `demos/04-scene/assets/hall.gltf` as `ggc`
/// names it: the source's path under the asset root, then what it produced.
pub const HALL: &str = "hall/scene";

/// Metres per tick at full deflection.
pub const MOVE_PER_TICK: f64 = 0.08;

/// Radians per unit of pointer motion.
pub const LOOK_PER_UNIT: f32 = 0.0035;

/// Pitch stops short of the pole, where the view basis collapses.
pub const PITCH_LIMIT: f32 = 1.55;

/// Where a session opens: inside the hall, eye height, facing the wall.
pub const START_POSITION: sim::DVec3 = sim::DVec3::new(0.0, 1.7, 9.0);

/// Tints the `tint` verb cycles, `0x00RRGGBB`. White is first because the
/// authored colours are the ones worth seeing before anything is done to them.
pub const TINTS: [u32; 4] = [0x00ff_ffff, 0x00ff_c080, 0x0080_c0ff, 0x0080_ff90];

/// Half-extent of the marker cube, metres.
pub const MARKER_SIZE: f32 = 0.25;

/// The demo's one light: a sun, warm, angled so a face pointing up and a face
/// pointing at the camera are lit differently (§6 M11).
///
/// Exported rather than buried in `bootstrap` because `gg-golden` builds this
/// demo's world from its constants: a sun the harness restated would be a sun
/// that could drift from the one a player sees.
pub const SUN_DIRECTION: sim::Vec3 = sim::Vec3::new(-0.42, -0.82, -0.39);
/// The sun's colour, `0x00RRGGBB`.
pub const SUN_COLOR: u32 = 0x00ff_f2dc;
/// The sun's radiance multiplier.
pub const SUN_INTENSITY: f32 = 3.5;

/// Cycle [`Model::tint`].
pub const TINT: ActionId = ActionId::new(0);
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

/// Where the visitor stands and where they look.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo04.visitor")]
#[repr(C)]
pub struct Visitor {
    /// World-space eye position, `f64` and never narrowed on this side of the
    /// membrane (§4.2.1).
    pub position: sim::DVec3,
    /// Rotation about +Y, radians.
    pub yaw: f32,
    /// Rotation about the visitor's right axis, radians.
    pub pitch: f32,
    /// Which of [`TINTS`] the hall is wearing.
    pub tint: u32,
    /// Padding, named.
    pub _pad: u32,
}

/// Spawn the visitor and the hall if there is no visitor yet.
///
/// Idempotent by asking the world rather than by remembering, because
/// remembering would be state outliving a tick (§4.2.2, and a grep gate).
pub fn bootstrap(world: &mut GameWorld) {
    let mut exists = false;
    world.visit::<&Visitor>(|_, _| exists = true);
    if exists {
        return;
    }

    let visitor = world.spawn();
    if let Err(refused) = world.insert(
        visitor,
        Visitor {
            position: START_POSITION,
            yaw: 0.0,
            pitch: 0.0,
            tint: 0,
            _pad: 0,
        },
    ) {
        world.log(log_level::ERROR, &refused.to_string());
        return;
    }

    // One entity for the whole hall. The pack's scene says where its pieces go
    // relative to this; growing that into an entity per piece is M10's job, and
    // doing it here would put a transform hierarchy in a demo.
    let hall = world.spawn();
    world.put(hall, Model::at(HALL, sim::DVec3::ZERO));

    // The sun. One directional light, so the scene is lit and the shadow pass
    // has something to cast (§6 M11). Spawned in `bootstrap` like everything
    // else here — a light is a component the game declares, not a renderer
    // setting somebody edits in the engine.
    let sun = world.spawn();
    world.put(sun, Light::sun(SUN_DIRECTION, SUN_COLOR, SUN_INTENSITY));

    // One box in a frame full of pack meshes, so both pipelines are exercised
    // by the thing a human actually looks at (§4.5 — they share the prepass).
    let marker = world.spawn();
    world.put(
        marker,
        Renderable::boxed(
            sim::DVec3::new(0.0, MARKER_SIZE as f64, 0.0),
            sim::Vec3::splat(MARKER_SIZE),
            0x00ff_e070,
        ),
    );
    world.log(log_level::INFO, "the hall is open");
}

/// Turn.
pub fn aim(world: &mut GameWorld) {
    let (x, y) = (world.axis(AIM_X), world.axis(AIM_Y));
    world.visit::<&mut Visitor>(|_, visitor| {
        visitor.yaw -= x * LOOK_PER_UNIT;
        visitor.pitch = (visitor.pitch - y * LOOK_PER_UNIT).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    });
}

/// Walk. Not normalized: diagonals are faster, which is a fly camera's
/// traditional bug and this demo's traditional feature.
pub fn walk(world: &mut GameWorld) {
    let axes = [
        f64::from(world.axis(MOVE_RIGHT)),
        f64::from(world.axis(MOVE_UP)),
        f64::from(world.axis(MOVE_FORWARD)),
    ];
    world.visit::<&mut Visitor>(|_, visitor| {
        let (forward, right) = sim::fly_basis(visitor.yaw, visitor.pitch);
        visitor.position = visitor.position
            + right * (axes[0] * MOVE_PER_TICK)
            + sim::DVec3::Y * (axes[1] * MOVE_PER_TICK)
            + forward * (axes[2] * MOVE_PER_TICK);
    });
}

/// Cycle the hall's tint. Held on the visitor rather than read back off the
/// `Model`, so the sim's state is the source and the protocol component is
/// output — the direction §4.5 asks for.
pub fn tint(world: &mut GameWorld) {
    if !world.just_pressed(TINT) {
        return;
    }
    world.visit::<&mut Visitor>(|_, visitor| {
        visitor.tint = (visitor.tint + 1) % TINTS.len() as u32;
    });
}

/// Fill the render protocol from the sim: where the eye is, and what the hall
/// is wearing. Last in the table, so it sees this tick's movement.
pub fn present(world: &mut GameWorld) {
    let mut seen: Option<(Entity, Visitor)> = None;
    world.visit::<&Visitor>(|entity, visitor| seen = Some((entity, *visitor)));
    let Some((entity, visitor)) = seen else {
        return;
    };
    world.put(
        entity,
        Eye {
            position: visitor.position,
            yaw: visitor.yaw,
            pitch: visitor.pitch,
        },
    );

    let chosen = TINTS[visitor.tint as usize % TINTS.len()];
    let mut halls = Vec::new();
    world.visit::<&Model>(|entity, model| halls.push((entity, *model)));
    for (entity, model) in halls {
        world.put(
            entity,
            Model {
                tint: chosen,
                ..model
            },
        );
    }
}

// Order in both verb lists is the id space (§4.7); order in `systems` is the
// execution order (§4.1). Neither is alphabetical and neither may drift.
gg_ecs::gg_game! {
    components: [Visitor, Model, Renderable, Light, Eye],
    actions: ["tint"],
    axes: ["move_right", "move_up", "move_forward", "aim_x", "aim_y"],
    systems: [bootstrap, aim, walk, tint, present],
}
