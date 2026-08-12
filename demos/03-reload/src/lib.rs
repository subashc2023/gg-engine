//! Demo 03 — **the Ugly Game** (§6 M5): move, aim, shoot cubes, spawn cubes,
//! restart. Deliberately ugly, deliberately throwaway, and the first artifact in
//! this workspace that is *game code* rather than a demo binary.
//!
//! Its job is not to be a good game. Its job is to be played by a human while an
//! agent edits it, so the cooperative loop this engine exists for gets falsified
//! by use rather than described. Everything below is meant to be rewritten — the
//! constants first, the system bodies next — and an API gap it exposes is a
//! finding, not a reason to grow the engine (§6 M5).
//!
//! # What makes it game code
//!
//! There is no `main`, no window, no renderer, and no loop: the host owns all
//! four. This crate exports the §4.2.2 symbols and nothing else, so every system
//! below is a stateless function over a world it does not own. Three engine
//! dependencies, held by `cargo-deny` (§3) rather than by discipline.
//!
//! # Restart is the edit affordance, not a convenience
//!
//! State persistence across a reload is the feature, which makes edits to
//! *initial* state silently invisible — the M0X spike's finding (a). `restart`
//! runs **before** `bootstrap` in the table, so pressing it wipes the world and
//! the same tick rebuilds it from whatever the initial-state code says *now*.
//! That is the affordance the finding asked for, spelled as a verb the player
//! already has.

use gg_ecs::boundary::{ActionId, AxisId, Eye, GameWorld, Light, Renderable, log_level};
use gg_ecs::{Component, Entity};
use gg_math::sim;

/// Metres per tick at full deflection.
pub const MOVE_PER_TICK: f64 = 0.08;

/// Radians per unit of pointer motion.
pub const LOOK_PER_UNIT: f32 = 0.0035;

/// Pitch stops short of the pole: at exactly ±π/2 the view basis collapses,
/// because up becomes parallel to forward.
pub const PITCH_LIMIT: f32 = 1.55;

/// Cubes the world opens with, in a ring around the origin.
/// Twelve would put a neighbour on the edge of [`AIM_CONE`] from the opening
/// pose, which is a coin-flip for the test that fires down the ring.
pub const START_CUBES: u64 = 8;

/// Radius of that ring, metres.
pub const RING_RADIUS: f64 = 6.0;

/// Where the player stands at a restart.
pub const START_POSITION: sim::DVec3 = sim::DVec3::new(0.0, 1.7, 12.0);

/// How far ahead of the player a spawned cube lands.
pub const SPAWN_DISTANCE: f64 = 4.0;

/// Cube ceiling. A spawn past it is ignored rather than growing without bound —
/// a chaos replay holding `spawn` for 100k ticks is a test, not a memory bug.
pub const MAX_CUBES: usize = 128;

/// How closely a cube must line up with the aim ray to be hit: the cosine of the
/// cone's half-angle, so bigger is tighter. About 10°.
pub const AIM_CONE: f64 = 0.985;

/// How far a miss travels before the tracer stops, metres.
pub const SHOT_RANGE: f64 = 40.0;

/// Ticks a tracer stays visible.
pub const TRACER_TICKS: u32 = 8;

/// Half-width of the floor slab, metres. Not gameplay — an FPS in a void has no
/// sense of motion, and the floor is the cheapest fix there is.
pub const FLOOR_HALF: f32 = 60.0;

/// Radius of a tracer beam, metres.
pub const TRACER_RADIUS: f32 = 0.02;

/// Half-thickness of the floor slab, metres. Its top sits at y = 0.
pub const FLOOR_THICKNESS: f32 = 0.05;

/// How far ahead of the eye the crosshair sits, metres.
pub const CROSSHAIR_DISTANCE: f64 = 1.0;

/// Half-extent of the crosshair, metres — about seven pixels at 720p.
pub const CROSSHAIR_SIZE: f32 = 0.004;

/// The demo's one light: a sun, warm, angled so a face pointing up and a face
/// pointing at the camera are lit differently (§6 M11).
///
/// Exported rather than buried in `bootstrap` because `gg-golden` builds this
/// demo's world from its constants: a sun the harness restated would be a sun
/// that could drift from the one a player sees.
pub const SUN_DIRECTION: sim::Vec3 = sim::Vec3::new(-0.35, -0.86, -0.37);
/// The sun's colour, `0x00RRGGBB`.
pub const SUN_COLOR: u32 = 0x00ff_f4e0;
/// The sun's radiance multiplier.
pub const SUN_INTENSITY: f32 = 3.2;

/// Fire at whatever is under the crosshair.
pub const FIRE: ActionId = ActionId::new(0);
/// Drop a cube ahead of the player.
pub const SPAWN: ActionId = ActionId::new(1);
/// Wipe the world; `bootstrap` rebuilds it the same tick.
pub const RESTART: ActionId = ActionId::new(2);

/// Strafe.
pub const MOVE_RIGHT: AxisId = AxisId::new(0);
/// Rise and fall — a fly camera, because the Ugly Game has no gravity.
pub const MOVE_UP: AxisId = AxisId::new(1);
/// Forward and back.
pub const MOVE_FORWARD: AxisId = AxisId::new(2);
/// Yaw, from pointer motion.
pub const AIM_X: AxisId = AxisId::new(3);
/// Pitch, from pointer motion.
pub const AIM_Y: AxisId = AxisId::new(4);

/// The player: where they stand and where they look. One entity carries it, and
/// "the player" is therefore a property of state rather than a handle living
/// outside the world.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo03.player")]
#[repr(C)]
pub struct Player {
    /// World-space eye position, `f64` and never narrowed on this side of the
    /// membrane (§4.2.1).
    pub position: sim::DVec3,
    /// Rotation about +Y, radians.
    pub yaw: f32,
    /// Rotation about the player's right axis, radians.
    pub pitch: f32,
}

/// One cube: the thing being shot at, and the thing being spawned.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo03.cube")]
#[repr(C)]
pub struct Cube {
    /// World-space position.
    pub position: sim::DVec3,
    /// Spawn order, unique and ascending — what makes "the newest cube" a fact
    /// about state rather than about a list kept somewhere else.
    pub order: u64,
    /// The tick it appeared on. A *counter*, never seconds (§2).
    pub spawned_at: u64,
    /// Colour, degrees around the wheel. The renderer's, once there is one.
    pub hue: u32,
    /// Half-extent, metres.
    pub scale: f32,
}

/// A tracer — where a shot went, fading over a few ticks.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo03.shot")]
#[repr(C)]
pub struct Shot {
    /// Muzzle.
    pub from: sim::DVec3,
    /// Where it stopped — the cube it hit, or [`SHOT_RANGE`] out.
    pub to: sim::DVec3,
    /// Ticks before it disappears.
    pub ticks_left: u32,
    /// Padding, named.
    pub _pad: u32,
}

/// Wipe the world on `restart`. First in the table, so `bootstrap` refills it in
/// the same tick — see the module docs on why that is the point.
pub fn restart(world: &mut GameWorld) {
    if !world.just_pressed(RESTART) {
        return;
    }
    let mut doomed = Vec::new();
    world.visit::<&Player>(|entity, _| doomed.push(entity));
    world.visit::<&Cube>(|entity, _| doomed.push(entity));
    world.visit::<&Shot>(|entity, _| doomed.push(entity));
    // The floor carries no gameplay component at all, so a sweep by gameplay
    // type would leave one behind per restart. Entities land here twice; the
    // second despawn is a no-op.
    world.visit::<&Renderable>(|entity, _| doomed.push(entity));
    for entity in doomed {
        world.despawn(entity);
    }
    world.log(log_level::INFO, "restarted");
}

/// Build the opening world if there is none. Also what a restart lands in.
///
/// Idempotent by construction — it asks whether a player exists rather than
/// remembering whether it has run, because remembering would be state outliving
/// a tick, which §4.2.2 forbids and the grep gates enforce.
pub fn bootstrap(world: &mut GameWorld) {
    let mut exists = false;
    world.visit::<&Player>(|_, _| exists = true);
    if exists {
        return;
    }

    let player = world.spawn();
    if let Err(refused) = world.insert(
        player,
        Player {
            position: START_POSITION,
            yaw: 0.0,
            pitch: 0.0,
        },
    ) {
        world.log(log_level::ERROR, &refused.to_string());
        return;
    }

    // The sun. One directional light, so the scene is lit and the shadow pass
    // has something to cast (§6 M11). Spawned in `bootstrap` like everything
    // else here — a light is a component the game declares, not a renderer
    // setting somebody edits in the engine.
    let sun = world.spawn();
    world.put(sun, Light::sun(SUN_DIRECTION, SUN_COLOR, SUN_INTENSITY));

    let floor = world.spawn();
    if let Err(refused) = world.insert(
        floor,
        Renderable::boxed(
            sim::DVec3::new(0.0, -FLOOR_THICKNESS as f64, 0.0),
            sim::Vec3::new(FLOOR_HALF, FLOOR_THICKNESS, FLOOR_HALF),
            0x0026_2a33,
        ),
    ) {
        world.log(log_level::ERROR, &refused.to_string());
    }

    let tick = world.tick();
    for order in 0..START_CUBES {
        let angle = order as f64 / START_CUBES as f64 * core::f64::consts::TAU;
        let (sin, cos) = sim::sin_cos(angle);
        place(
            world,
            sim::DVec3::new(cos * RING_RADIUS, 0.5, sin * RING_RADIUS),
            order,
            tick,
        );
    }
    world.log(log_level::INFO, "the ugly game is open for business");
}

/// Turn. An FPS aims with the pointer always, not while a modifier is held —
/// the Ugly Game is played, not inspected.
pub fn aim(world: &mut GameWorld) {
    let (x, y) = (world.axis(AIM_X), world.axis(AIM_Y));
    world.visit::<&mut Player>(|_, player| {
        player.yaw -= x * LOOK_PER_UNIT;
        player.pitch = (player.pitch - y * LOOK_PER_UNIT).clamp(-PITCH_LIMIT, PITCH_LIMIT);
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
    world.visit::<&mut Player>(|_, player| {
        let (forward, right) = sim::fly_basis(player.yaw, player.pitch);
        let up = sim::DVec3::new(0.0, 1.0, 0.0);
        player.position += (forward * axes[2] + right * axes[0] + up * axes[1]) * MOVE_PER_TICK;
    });
}

/// Fire: the best-aligned cube inside the cone dies, and a tracer marks the shot
/// either way.
pub fn shoot(world: &mut GameWorld) {
    if !world.just_pressed(FIRE) {
        return;
    }
    let Some(player) = the_player(world) else {
        return;
    };
    let (forward, _) = sim::fly_basis(player.yaw, player.pitch);
    let eye = player.position;

    // Best alignment, not nearest: two cubes on the same line should both be
    // hittable by walking, and picking by distance would make the far one
    // unreachable forever.
    let mut best: Option<(Entity, sim::DVec3, f64)> = None;
    world.visit::<&Cube>(|entity, cube| {
        let Some(direction) = (cube.position - eye).try_normalize() else {
            return;
        };
        let aligned = direction.dot(forward);
        if aligned > AIM_CONE && best.is_none_or(|(_, _, previous)| aligned > previous) {
            best = Some((entity, cube.position, aligned));
        }
    });

    let tracer = world.spawn();
    if let Err(refused) = world.insert(
        tracer,
        Shot {
            from: eye,
            to: best.map_or(eye + forward * SHOT_RANGE, |(_, position, _)| position),
            ticks_left: TRACER_TICKS,
            _pad: 0,
        },
    ) {
        world.log(log_level::ERROR, &refused.to_string());
    }
    if let Some((hit, _, _)) = best {
        world.despawn(hit);
    }
}

/// Drop a cube ahead of the player.
pub fn spawn(world: &mut GameWorld) {
    if !world.just_pressed(SPAWN) {
        return;
    }
    let Some(player) = the_player(world) else {
        return;
    };
    let (count, highest) = cube_extremes(world);
    if count >= MAX_CUBES {
        return;
    }
    let (forward, _) = sim::fly_basis(player.yaw, player.pitch);
    let tick = world.tick();
    place(
        world,
        player.position + forward * SPAWN_DISTANCE,
        highest.saturating_add(1),
        tick,
    );
}

/// Age the tracers out.
pub fn fade(world: &mut GameWorld) {
    let mut spent = Vec::new();
    world.visit::<&mut Shot>(|entity, shot| {
        shot.ticks_left = shot.ticks_left.saturating_sub(1);
        if shot.ticks_left == 0 {
            spent.push(entity);
        }
    });
    for entity in spent {
        world.despawn(entity);
    }
}

/// Say what the world looks like (§4.5 v0). Last in the table, so it describes
/// the tick that just happened rather than the one before it.
///
/// **This is where the Ugly Game's whole appearance lives**, and it is game code:
/// an agent changing a colour, a size or what a tracer looks like is editing a
/// reloadable system body, not the engine. The host draws boxes where it is told
/// and has no other opinion.
///
/// Collect-then-write in every pass, because a host call invalidates any borrow
/// into the world (§4.2.2) and `insert` is a host call.
pub fn present(world: &mut GameWorld) {
    let mut eye = None;
    world.visit::<&Player>(|entity, player| eye = Some((entity, *player)));
    if let Some((entity, player)) = eye {
        let seen = Eye::at(player.position, player.yaw, player.pitch);
        world.put(entity, seen);
        // The crosshair, and it rides on the player's own entity: a first-person
        // player is not drawn, so its `Renderable` is free for the one thing a
        // cone-of-fire game cannot be played without. There is no UI layer until
        // M13 and this needs none — a small box, one metre ahead, in world space.
        let (forward, _) = sim::fly_basis(player.yaw, player.pitch);
        world.put(
            entity,
            Renderable::boxed(
                player.position + forward * CROSSHAIR_DISTANCE,
                sim::Vec3::splat(CROSSHAIR_SIZE),
                0x00ff_f0c0,
            ),
        );
    }

    let mut cubes = Vec::new();
    world.visit::<&Cube>(|entity, cube| cubes.push((entity, *cube)));
    for (entity, cube) in cubes {
        let look = Renderable::boxed(cube.position, sim::Vec3::splat(cube.scale), wheel(cube.hue));
        world.put(entity, look);
    }

    let mut shots = Vec::new();
    world.visit::<&Shot>(|entity, shot| shots.push((entity, *shot)));
    for (entity, shot) in shots {
        world.put(entity, beam(&shot));
    }
}

/// A tracer as a long thin box: centred between its ends, +Z turned onto the
/// direction it travelled, fading as it ages.
fn beam(shot: &Shot) -> Renderable {
    let along = shot.to - shot.from;
    let length = along.length();
    let fade = 96 + (shot.ticks_left * 159 / TRACER_TICKS.max(1));
    Renderable {
        position: (shot.from + shot.to) * 0.5,
        rotation: turn_z_onto(along),
        half_extent: sim::Vec3::new(TRACER_RADIUS, TRACER_RADIUS, (length * 0.5) as f32),
        color: (fade << 16) | (fade << 8) | 0x40,
        smoothness: gg_ecs::boundary::DEFAULT_SMOOTHNESS,
        metallic: 0.0,
        shape: gg_ecs::boundary::shape::BOX,
    }
}

/// The rotation taking +Z onto `direction`. Identity for a zero-length or
/// already-aligned direction, and a half turn about +X for the exact opposite —
/// the axis is degenerate at both poles and a cross product there is noise.
fn turn_z_onto(direction: sim::DVec3) -> sim::DQuat {
    let Some(unit) = direction.try_normalize() else {
        return sim::DQuat::IDENTITY;
    };
    let forward = sim::DVec3::new(0.0, 0.0, 1.0);
    match forward.cross(unit).try_normalize() {
        Some(axis) => {
            sim::DQuat::from_axis_angle(axis, sim::acos(forward.dot(unit).clamp(-1.0, 1.0)))
        }
        None if unit.z < 0.0 => sim::DQuat::from_axis_angle(sim::DVec3::X, core::f64::consts::PI),
        None => sim::DQuat::IDENTITY,
    }
}

/// Degrees around the colour wheel → `0x00RRGGBB`, at full saturation.
///
/// Integer arithmetic throughout: this runs inside a hashed tick, and a colour
/// is not worth a float (§2, and it costs nothing to be exact).
fn wheel(hue: u32) -> u32 {
    let hue = hue % 360;
    let ramp = (hue % 60) * 255 / 60;
    let (r, g, b) = match hue / 60 {
        0 => (255, ramp, 0),
        1 => (255 - ramp, 255, 0),
        2 => (0, 255, ramp),
        3 => (0, 255 - ramp, 255),
        4 => (ramp, 0, 255),
        _ => (255, 0, 255 - ramp),
    };
    (r << 16) | (g << 8) | b
}

/// One cube, at a place, with an order. The one insert site, so a cube spawned
/// at bootstrap and a cube spawned by the player cannot drift apart.
fn place(world: &mut GameWorld, position: sim::DVec3, order: u64, tick: u64) {
    let entity = world.spawn();
    if let Err(refused) = world.insert(
        entity,
        Cube {
            position,
            order,
            spawned_at: tick,
            // Derived from the order rather than random: a replay draws no
            // numbers the recorder did not record (§4.7).
            hue: ((order * 47) % 360) as u32,
            scale: 0.5,
        },
    ) {
        world.log(log_level::ERROR, &refused.to_string());
    }
}

/// The player's pose, by value — a copy, because the next host call invalidates
/// any borrow into the world (§4.2.2).
fn the_player(world: &mut GameWorld) -> Option<Player> {
    let mut found = None;
    world.visit::<&Player>(|_, player| found = Some(*player));
    found
}

/// `(count, highest order)` in one pass, because the ceiling and the next order
/// are the same question and two passes could answer it twice differently.
fn cube_extremes(world: &mut GameWorld) -> (usize, u64) {
    let (mut count, mut highest) = (0usize, 0u64);
    world.visit::<&Cube>(|_, cube| {
        count += 1;
        highest = highest.max(cube.order);
    });
    (count, highest)
}

// Order in both lists is the id space (§4.7) and order in `systems` is the
// schedule (§4.1) — which is why both are game code, and both are reloadable.
gg_ecs::gg_game! {
    components: [Player, Cube, Shot, Renderable, Light, Eye],
    actions: ["fire", "spawn", "restart"],
    axes: ["move_right", "move_up", "move_forward", "aim_x", "aim_y"],
    systems: [restart, bootstrap, aim, walk, shoot, spawn, fade, present],
}
