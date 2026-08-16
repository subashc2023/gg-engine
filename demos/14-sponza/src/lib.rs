//! Demo 14 — **Sponza** (§6 M59): somewhere to stand while the frame is taken
//! apart.
//!
//! Every other demo in this tree is content *we* authored, which is the honest
//! way to build a renderer and the wrong way to check one. An authored scene
//! asks the questions its author knew to ask: demo 06's atrium has a mortar line
//! because a normal map was the thing being demonstrated, and demo 12's room has
//! twenty-six boxes because a shooter needed cover. Sponza asks the questions
//! nobody chose — 262,267 triangles across 103 meshes and 69 textures, arcades
//! that occlude each other at every scale, a floor that runs thirty metres, and
//! curtains that are two-sided alpha. It is the scene the literature is written
//! against, so a number measured here is a number somebody else can argue with.
//!
//! # It is not checked in, and that is §6 M11's decision rather than a new one
//!
//! Fifty-one megabytes. Demo 06's header records the refusal; `cargo xtask
//! sponza` records the download instead, which is a file this repository can
//! hold. With the assets absent every [`Model`] here names something the pack
//! does not contain and draws nothing — documented behaviour, not a failure —
//! so the crate compiles, its tests pass, and `xtask ci` never touches
//! fifty-one megabytes of JPEG.
//!
//! # A fly camera and nothing else
//!
//! There is no game here. What a scene viewer needs is to be walked through
//! with the frame paused at whatever the eye stops on, and the two ways to do
//! that are `--editor` (whose camera is the editor's, §6 M15) and this — WASD to
//! move, `QE` for up and down, the mouse to look, and shift to cover thirty
//! metres in a reasonable number of seconds. The camera state is a component
//! for the reason every other demo's is: it survives a reload with the world
//! intact (§4.2.2), so a rebuild does not put the eye back at the door.
//!
//! Run it: `cargo xtask sponza` once, then `cargo xtask run 14-sponza --editor`.

use gg_ecs::Component;
use gg_ecs::boundary::{AxisId, Eye, GameWorld, Light, Model, Sky, log_level};
use gg_math::sim;

/// The pack asset this demo draws — `assets/Sponza.gltf` as `ggc` names it: the
/// source's stem, then what it produced.
pub const SPONZA: &str = "Sponza/scene";

/// Where the eye starts: at the west end of the nave, at head height, looking
/// down the long axis. The one framing that shows what the scene is for — the
/// arcade receding on both sides, which is the geometry every occlusion and
/// shadow question in this milestone is asked about.
pub const EYE_AT: sim::DVec3 = sim::DVec3::new(-9.0, 1.7, 0.0);
/// Yaw at spawn, radians: down +x, which is the nave.
pub const EYE_YAW: f32 = -core::f32::consts::FRAC_PI_2;

/// Which way the sun points, and it is deliberately not down the nave: a sun
/// square to the arcade puts every column's shadow on the same floor tile and
/// answers nothing about cascade fit.
pub const SUN: sim::Vec3 = sim::Vec3::new(-0.55, -0.72, -0.42);
/// Bright, because the scene is outdoors above the arcade and the interior is
/// lit by what gets in — which is the whole reason it is a hard scene.
pub const SUN_INTENSITY: f32 = 6.0;
/// The sky's own intensity. Small against the sun: it is the fill that reaches
/// the floor of a colonnade, and a bright one washes the term this scene exists
/// to make visible.
pub const SKY_INTENSITY: f32 = 0.6;

/// Metres a second on the ground, and the multiple `shift` is worth. Sponza is
/// about thirty metres end to end, so the slow speed crosses it in ten seconds
/// and the fast one in three.
pub const WALK: f64 = 3.0;
/// See [`WALK`].
pub const SPRINT: f64 = 4.0;
/// Radians per raw mouse count, matching demo 12's so the two feel the same.
pub const LOOK: f32 = 0.0022;
/// How far the eye may look up or down. Short of vertical, where the fly basis
/// stops having a usable right vector.
pub const PITCH_LIMIT: f32 = 1.5;

/// Ticks a second, so a per-tick step is a per-second speed divided by this.
/// The shell's fixed rate (§2) — named rather than inlined because a speed in
/// metres per second is the reviewable number and a speed per tick is not.
pub const TICK_HZ: f64 = 60.0;

/// Strafe, right positive.
pub const MOVE_RIGHT: AxisId = AxisId::new(0);
/// Forward, away from the eye positive.
pub const MOVE_FORWARD: AxisId = AxisId::new(1);
/// Rise, up positive.
pub const MOVE_UP: AxisId = AxisId::new(2);
/// Yaw, from raw mouse counts.
pub const AIM_X: AxisId = AxisId::new(3);
/// Pitch, from raw mouse counts.
pub const AIM_Y: AxisId = AxisId::new(4);
/// The speed multiplier, as an axis rather than an action: an action is an edge
/// and this is a state, and a held key read as an edge moves the eye once.
pub const BOOST: AxisId = AxisId::new(5);

/// Where the eye is and which way it faces. Sim state, so it crosses a reload.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "sponza.flyer")]
#[repr(C)]
pub struct Flyer {
    /// Metres, world space.
    pub position: sim::DVec3,
    /// Radians about +y.
    pub yaw: f32,
    /// Radians about the body's right axis, clamped to [`PITCH_LIMIT`].
    pub pitch: f32,
}

/// The scene, the sun, the sky and the eye — once. Re-entrant, because a reload
/// runs it again against a world that already holds them (§6 M5).
pub fn bootstrap(world: &mut GameWorld) {
    let mut exists = false;
    let _ = world.each::<&Flyer>(|_, _| exists = true);
    if exists {
        return;
    }
    world.spawn_with(Model::at(SPONZA, sim::DVec3::ZERO));
    world.spawn_with(Light::sun(SUN, 0x00ff_f0d8, SUN_INTENSITY));
    // Unbounded: the arcade is open to it on both sides and above, so a volume
    // would be a wall the geometry does not have (§6 M28).
    world.spawn_with(Sky::daylight(SKY_INTENSITY));
    let eye = world.spawn_with(Flyer {
        position: EYE_AT,
        yaw: EYE_YAW,
        pitch: 0.0,
    });
    world.put(eye, Eye::at(EYE_AT, EYE_YAW, 0.0));
    world.log(log_level::INFO, "sponza: standing in the nave");
}

/// One tick of flight, and the [`Eye`] that follows from it.
///
/// Deliberately one system rather than a look pass and a move pass: there is no
/// collision here and nothing between them, so splitting would be two queries
/// over one entity to make the file look like demo 12's.
pub fn fly(world: &mut GameWorld) {
    let (aim_x, aim_y) = (world.axis(AIM_X), world.axis(AIM_Y));
    let right = f64::from(world.axis(MOVE_RIGHT)).clamp(-1.0, 1.0);
    let ahead = f64::from(world.axis(MOVE_FORWARD)).clamp(-1.0, 1.0);
    let rise = f64::from(world.axis(MOVE_UP)).clamp(-1.0, 1.0);
    // A held key reads as 1.0, so this is `WALK` or `WALK * SPRINT` and nothing
    // between — an analogue boost would be a speed nobody can reproduce.
    let speed = match world.axis(BOOST) > 0.5 {
        true => WALK * SPRINT,
        false => WALK,
    } / TICK_HZ;
    let _ = world.each::<(&mut Flyer, &mut Eye)>(|_, (flyer, eye)| {
        flyer.yaw = wrap(flyer.yaw - aim_x * LOOK);
        flyer.pitch = (flyer.pitch - aim_y * LOOK).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        // The full basis, pitch included: this is a fly camera, so looking at
        // the ceiling and pressing forward goes to the ceiling. Demo 12 zeroes
        // the pitch here because it is a walker and this is not.
        let (forward, strafe) = sim::fly_basis(flyer.yaw, flyer.pitch);
        flyer.position = flyer.position
            + forward * (ahead * speed)
            + strafe * (right * speed)
            + sim::DVec3::Y * (rise * speed);
        *eye = Eye::at(flyer.position, flyer.yaw, flyer.pitch);
    });
}

/// Yaw into `(-π, π]`, so a session that turns in one direction for an hour
/// does not lose precision in the low bits (§1.3).
fn wrap(yaw: f32) -> f32 {
    let turn = core::f32::consts::TAU;
    let mut wrapped = yaw;
    while wrapped > core::f32::consts::PI {
        wrapped -= turn;
    }
    while wrapped <= -core::f32::consts::PI {
        wrapped += turn;
    }
    wrapped
}

// Order in `systems` is execution order (§4.1); order in the verb lists is the
// id space a replay records (§4.7). Neither is alphabetical, neither may drift.
#[cfg(feature = "game")]
gg_ecs::gg_game! {
    components: [Flyer, Model, Light, Sky, Eye],
    actions: [],
    axes: ["move_right", "move_forward", "move_up", "aim_x", "aim_y", "boost"],
    systems: [bootstrap, fly],
}
