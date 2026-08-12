//! Demo 06 — **the atrium** (§6 M11): the first screenshot we would show a
//! stranger.
//!
//! Demo 04 put pack content on the screen. This lights it: a sun that sweeps,
//! four lamps on the pillars, tiled stone carrying all four maps, and two rows
//! of spheres running from mirror to matte so a viewer can see what roughness
//! and metalness actually do.
//!
//! # Sponza, and why this is not it
//!
//! §6 M11's exit row says "lit Sponza". Sponza is about fifty megabytes and
//! cannot be checked in — §6 M9 already said so when the load-time numbers were
//! measured against a synthetic stand-in for the same reason. `assets/atrium.gltf`
//! is authored by a script and asks the same questions Sponza would: a normal
//! map bending light across a mortar line, an occlusion map darkening the same
//! groove, roughness spreading a highlight, a pillar putting a shadow on a
//! floor. What it does *not* ask is the one thing scale asks — thousands of
//! draws with hundreds of materials — and that is demo 05's subject already.
//!
//! # The smooth metal reflects a room, and that is the point of the row
//!
//! The leftmost sphere in the back row is metal at roughness 0.05 — a mirror. A
//! mirror has no diffuse lobe and reflects whatever is around it, so until §6
//! M27 it reflected *nothing*: a sun it mostly did not point at and four small
//! lamps, which is why it was nearly black. That gap was left visible on purpose
//! for five milestones. [`SKY`] closes it — a compiled panorama, and the sphere
//! now shows the room it stands in, four coloured walls and a strip of windows,
//! sharpening as the row runs left.
//!
//! The panorama is generated rather than photographed (`gg-tools panorama`), and
//! it deliberately holds no sun: the sun is a [`Light`], and a disc burnt into
//! the environment as well would be the same photon counted twice (§6 M24).
//!
//! # What a game says about light
//!
//! Where it is, which way it points, what colour it is, how bright. Not a
//! shading model, not an exposure, not a shadow-map resolution — those are the
//! renderer's, and they are CVars because they belong to the machine rather than
//! to the game (§4.8). The sun below sweeps because a static sun makes a shadow
//! look like a texture; the lamps flicker on a schedule for the same reason.

use gg_ecs::boundary::{ActionId, AxisId, Eye, GameWorld, Light, Model, Sky, log_level};
use gg_ecs::{Component, Entity};
use gg_math::sim;

/// The pack asset this demo draws — `demos/06-lit/assets/atrium.gltf` as `ggc`
/// names it: the source's path under the asset root, then what it produced.
pub const ATRIUM: &str = "atrium/scene";

/// The environment, named the same way: `demos/06-lit/assets/sky/room.hdr`
/// compiled by `ggc` (§6 M27). Written by `gg-tools panorama`, which is where
/// its shape is argued.
pub const SKY: &str = "sky/room";

/// Exposure over the panorama's own radiance.
///
/// Small, and chosen against the sun rather than by eye. The file's mean
/// radiance is about 2, so this puts the environment's irradiance near 0.25
/// against [`SUN_INTENSITY`]'s 4 — a fill at a sixteenth of the key.
///
/// It reads low written down and it is not: an environment lights *every* face
/// from *every* direction and is never shadowed, where a sun lights one side of
/// one thing and is blocked by everything. Matched by irradiance rather than by
/// eye, an environment at a quarter of the key light flattens a room — every
/// shadow fills, and a demo whose subject is a sun casting one has lost it.
///
/// The *specular* half is unaffected by that restraint, which is the point: the
/// window strip is two orders of magnitude over the walls, so at this exposure a
/// mirror still reflects panes at radiance 3 while a diffuse surface gathers a
/// tenth of one.
pub const SKY_INTENSITY: f32 = 0.04;

/// Metres per tick at full deflection.
pub const MOVE_PER_TICK: f64 = 0.08;

/// Radians per unit of pointer motion.
pub const LOOK_PER_UNIT: f32 = 0.0035;

/// Pitch stops short of the pole, where the view basis collapses.
pub const PITCH_LIMIT: f32 = 1.55;

/// Where a session opens: inside the atrium, eye height, facing the spheres.
pub const START_POSITION: sim::DVec3 = sim::DVec3::new(0.0, 1.7, 7.0);

/// Ticks the sun takes to cross the sky. Long enough that a shadow's motion is
/// visible without being a strobe, and a whole number of ticks so the sweep is
/// a function of the tick count and nothing else (§1.3).
pub const SUN_PERIOD: u64 = 1800;

/// How far off vertical the sun leans, radians. Not straight down: a sun at the
/// zenith casts a shadow directly under everything, which is the one angle at
/// which a shadow map proves the least.
pub const SUN_TILT: f64 = 0.55;

/// The sun's colour and strength — warm, and bright enough that the tonemapper
/// has something to compress.
pub const SUN_COLOR: u32 = 0x00ff_f2d8;
/// Radiance multiplier for the sun.
pub const SUN_INTENSITY: f32 = 4.0;

/// Where the four lamps hang, and how far they reach.
pub const LAMP_AT: [(f64, f64); 4] = [(-5.0, -5.0), (5.0, -5.0), (-5.0, 3.0), (5.0, 3.0)];
/// Lamp height, metres.
pub const LAMP_HEIGHT: f64 = 4.2;
/// Metres at which a lamp contributes exactly nothing.
pub const LAMP_RANGE: f32 = 9.0;
/// The lamps' colour and strength.
pub const LAMP_COLOR: u32 = 0x00ff_b060;
/// Radiance multiplier for a lamp. Larger than the sun's because inverse-square
/// falloff has already divided it by the square of a few metres.
pub const LAMP_INTENSITY: f32 = 18.0;

/// Freeze and unfreeze the sun.
pub const HOLD: ActionId = ActionId::new(0);
/// Switch the lamps off and on.
pub const LAMPS: ActionId = ActionId::new(1);
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

/// Where the visitor stands, where they look, and what the lights are doing.
///
/// The sun's phase lives here rather than being derived from the tick, because
/// holding it is a verb: a derived angle could not be stopped without the tick
/// stopping with it.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo06.visitor")]
#[repr(C)]
pub struct Visitor {
    /// World-space eye position, `f64` and never narrowed on this side of the
    /// membrane (§4.2.1).
    pub position: sim::DVec3,
    /// Rotation about +Y, radians.
    pub yaw: f32,
    /// Rotation about the visitor's right axis, radians.
    pub pitch: f32,
    /// Where the sun is in its sweep, in ticks. Integer, because a float phase
    /// would be a float clock and §1.3 has one of those and it counts (§4.7).
    pub phase: u64,
    /// Whether the sun is held.
    pub held: u32,
    /// Whether the lamps are lit.
    pub lamps: u32,
}

/// The direction the sun's light *travels* at `phase`, unit length.
///
/// A function of the phase alone, so the same tick produces the same sun on
/// every architecture — the sweep is `sim` trig through `libm` for that reason
/// and not because a light needs `f64` precision.
#[must_use]
pub fn sun_direction(phase: u64) -> sim::Vec3 {
    let turn = (phase % SUN_PERIOD) as f64 / SUN_PERIOD as f64 * core::f64::consts::TAU;
    let (sin_turn, cos_turn) = sim::sin_cos(turn);
    let (sin_tilt, cos_tilt) = sim::sin_cos(SUN_TILT);
    // Downward always: a sun that swept below the floor would spend half its
    // period lighting the underside of the atrium, which is a demo of nothing.
    let direction = sim::DVec3::new(sin_tilt * cos_turn, -cos_tilt, sin_tilt * sin_turn);
    // Already unit by construction — `sin^2 + cos^2` twice over — but divided
    // anyway, because the shader takes this as a unit vector and "by
    // construction" is the kind of guarantee an edit to `SUN_TILT` quietly ends.
    let length = direction.length();
    sim::Vec3::new(
        (direction.x / length) as f32,
        (direction.y / length) as f32,
        (direction.z / length) as f32,
    )
}

/// Spawn the visitor, the atrium, the sun and the lamps if there is no visitor
/// yet.
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
            phase: 0,
            held: 0,
            lamps: 1,
        },
    ) {
        world.log(log_level::ERROR, &refused.to_string());
        return;
    }

    // One entity for the whole room; the pack's scene says where its eighteen
    // pieces go relative to it (§4.6).
    let atrium = world.spawn();
    world.put(atrium, Model::at(ATRIUM, sim::DVec3::ZERO));

    // The sun. Spawned first of the lights, and therefore the one that casts:
    // extract keeps world iteration order and the renderer's single cascade
    // takes the first directional light it is given (§6 M11).
    let sun = world.spawn();
    world.put(sun, Light::sun(sun_direction(0), SUN_COLOR, SUN_INTENSITY));

    // The environment, as a compiled panorama in the same pack the room came
    // out of (§6 M27). It is what the mirror in the back row reflects, and the
    // intensity is exposure rather than a tint — the file is already a radiance.
    world.spawn_with(Sky::image(SKY, SKY_INTENSITY));

    for (x, z) in LAMP_AT {
        let lamp = world.spawn();
        world.put(
            lamp,
            Light::point(
                sim::DVec3::new(x, LAMP_HEIGHT, z),
                LAMP_COLOR,
                LAMP_INTENSITY,
                LAMP_RANGE,
            ),
        );
    }
    world.log(log_level::INFO, "the atrium is lit");
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

/// Advance the sun's sweep, and take the two switches.
pub fn sky(world: &mut GameWorld) {
    let hold = world.just_pressed(HOLD);
    let lamps = world.just_pressed(LAMPS);
    world.visit::<&mut Visitor>(|_, visitor| {
        if hold {
            visitor.held ^= 1;
        }
        if lamps {
            visitor.lamps ^= 1;
        }
        if visitor.held == 0 {
            // Wrapped here rather than at the read, so the stored phase is
            // bounded and the hash of it cannot drift with uptime.
            visitor.phase = (visitor.phase + 1) % SUN_PERIOD;
        }
    });
}

/// Fill the render protocol from the sim: where the eye is, and where the light
/// comes from. Last in the table, so it sees this tick's movement.
pub fn present(world: &mut GameWorld) {
    let mut seen: Option<(Entity, Visitor)> = None;
    world.visit::<&Visitor>(|entity, visitor| seen = Some((entity, *visitor)));
    let Some((entity, visitor)) = seen else {
        return;
    };
    world.put(
        entity,
        Eye::at(visitor.position, visitor.yaw, visitor.pitch),
    );

    let direction = sun_direction(visitor.phase);
    // Collected before writing: `each` holds the column borrow, and `insert`
    // takes the world (§4.2.2).
    let mut lights = Vec::new();
    world.visit::<&Light>(|entity, light| lights.push((entity, *light)));
    for (entity, light) in lights {
        let updated = match light.kind {
            gg_ecs::boundary::light::DIRECTIONAL => Light { direction, ..light },
            // Switched off by dropping the intensity to zero rather than by
            // despawning: a light that came and went would change the world's
            // archetype shape twice a keypress, and the renderer already treats
            // zero radiance as nothing.
            _ => Light {
                intensity: f32::from(visitor.lamps == 1) * LAMP_INTENSITY,
                ..light
            },
        };
        world.put(entity, updated);
    }
}

// Order in both verb lists is the id space (§4.7); order in `systems` is the
// execution order (§4.1). Neither is alphabetical and neither may drift.
#[cfg(feature = "game")]
gg_ecs::gg_game! {
    components: [Visitor, Model, Light, Eye],
    actions: ["hold", "lamps"],
    axes: ["move_right", "move_up", "move_forward", "aim_x", "aim_y"],
    systems: [bootstrap, aim, walk, sky, present],
}
