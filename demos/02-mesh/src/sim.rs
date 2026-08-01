//! Demo 02's simulation (§6 M4B) — the half of the demo a replay reproduces.
//!
//! Everything the player can change lives in `gg-ecs` as components, is
//! advanced one tick at a time from [`gg_input`] action state, and is computed
//! with `gg_math::sim` only. No `glam`, no `std` transcendentals, no clock: the
//! spin is a **tick counter**, not accumulated seconds (§2, Sim time row), so
//! reaching tick 480 twice puts the cube in exactly the same place twice.
//!
//! That discipline is the point of the milestone rather than a style: this
//! module is what the three-way determinism gate hashes (§5.6), and a single
//! `f64::sin` or one iteration over a `HashMap` in here is what the gate exists
//! to catch. Both are lint-rejected before the gate ever runs.
//!
//! Builds with no GPU and no windowing in its graph, which is what lets the
//! aarch64-under-qemu leg run it at all.

use gg_ecs::{CanonicalHash, Component, Entity, Query, World, WorldError};
use gg_extract::SimTransform;
use gg_input::{ActionId, ActionMap, AxisId, Input, MapError, Replay, ReplayError};
use gg_math::sim;

/// The verbs the demo declares. **Order is the id space** a replay file records
/// (§4.7): appending is free, reordering is a replay-format change, and a
/// replay recorded against a different order is refused by name.
pub const ACTIONS: &[&str] = &["look", "spawn", "despawn"];

/// Axis verbs, same rule.
pub const AXES: &[&str] = &["move_right", "move_up", "move_forward", "look_x", "look_y"];

/// Held to turn the camera.
pub const LOOK: ActionId = ActionId::new(0);
/// Spawn a cube ahead of the camera.
pub const SPAWN: ActionId = ActionId::new(1);
/// Despawn the most recently spawned cube.
pub const DESPAWN: ActionId = ActionId::new(2);

/// Strafe.
pub const MOVE_RIGHT: AxisId = AxisId::new(0);
/// Rise and fall.
pub const MOVE_UP: AxisId = AxisId::new(1);
/// Forward and back.
pub const MOVE_FORWARD: AxisId = AxisId::new(2);
/// Yaw, from pointer motion.
pub const LOOK_X: AxisId = AxisId::new(3);
/// Pitch, from pointer motion.
pub const LOOK_Y: AxisId = AxisId::new(4);

/// The demo's bindings, compiled in. A shipped game would load this from disk;
/// a demo whose replay gate must run on three architectures should not need a
/// file beside the binary to have a camera.
pub const BINDINGS: &str = include_str!("../input.toml");

/// Sim rate. The replay stream is indexed by this tick, never by seconds.
pub const TICK_HZ: u32 = 60;

/// Ticks for one full turn of a cube.
pub const TICKS_PER_TURN: u64 = 480;

/// Metres per tick at full deflection.
pub const MOVE_PER_TICK: f64 = 0.035;

/// Radians per unit of pointer motion.
pub const LOOK_PER_UNIT: f32 = 0.0025;

/// Pitch stops just short of straight up: at exactly ±π/2 the view basis
/// collapses, because the up vector becomes parallel to the forward one.
pub const PITCH_LIMIT: f32 = 1.55;

/// How far ahead of the camera a spawned cube lands.
pub const SPAWN_DISTANCE: f64 = 3.0;

/// Cube ceiling. A spawn past it is ignored rather than growing without bound —
/// a chaos replay holding `spawn` for 100k ticks is a test, not a memory bug.
pub const MAX_CUBES: usize = 64;

/// Where the first cube sits, relative to the sim's origin.
pub const ORIGIN_CUBE: sim::DVec3 = sim::DVec3::new(0.0, 0.0, 0.0);

/// A deliberately absurd world origin: 10^12 m, about seven astronomical units.
///
/// The whole reason sim positions are `f64` (§4.2.1). An `f32` absolute
/// position out here resolves to ~65 km; subtract-then-narrow through the
/// camera resolves to ~0.24 mm, which is the `f64` ulp at this magnitude and
/// far below a pixel. `gg-golden`'s `mesh-far` scene renders from here and is
/// compared against a reference blessed at the same offset — the exit criterion
/// that turns "no visible jitter" into an image.
pub const FAR_ORIGIN: sim::DVec3 = sim::DVec3::new(1.0e12, 0.0, 0.0);

/// The camera, as sim state. Position is `f64` world-space and never narrows on
/// this side of the membrane (§4.2.1); the render half receives it through
/// `gg-extract` and nowhere else.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo02.camera")]
#[repr(C)]
pub struct CameraRig {
    /// World-space eye position.
    pub position: sim::DVec3,
    /// Rotation about +Y, radians.
    pub yaw: f32,
    /// Rotation about the camera's right axis, radians.
    pub pitch: f32,
}

impl CameraRig {
    /// The demo's opening shot, and the pose gg-golden renders.
    pub const GOLDEN: CameraRig = CameraRig {
        position: sim::DVec3::new(1.1, 0.8, 1.8),
        yaw: 0.55,
        pitch: -0.36,
    };

    /// Forward and right, in sim space. `f64` trig through `libm`: the same
    /// bits on every target in the contract, which a `glam` basis would not be.
    ///
    /// The render half builds its view matrix from the same yaw and pitch with
    /// SIMD `f32` glam, and the two bases differ in the last few bits. That is
    /// the membrane working as designed — the sim decides where the camera
    /// *goes*, the renderer decides what it *sees*, and only the former is
    /// hashed.
    pub fn basis(&self) -> (sim::DVec3, sim::DVec3) {
        let (sin_yaw, cos_yaw) = sim::sin_cos(f64::from(self.yaw));
        let (sin_pitch, cos_pitch) = sim::sin_cos(f64::from(self.pitch));
        let forward = sim::DVec3::new(-cos_pitch * sin_yaw, sin_pitch, -cos_pitch * cos_yaw);
        let right = sim::DVec3::new(cos_yaw, 0.0, -sin_yaw);
        (forward, right)
    }
}

/// One spinning cube.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "demo02.cube")]
#[repr(C)]
pub struct Cube {
    /// World-space position.
    pub position: sim::DVec3,
    /// Ticks into its turn. A *counter*, wrapped at [`TICKS_PER_TURN`] — the
    /// orientation is derived from it and never accumulated into a quaternion,
    /// so a cube at tick 7 is identical to one at tick 487.
    pub spin_ticks: u64,
    /// Spawn order, unique and ascending. Despawn takes the highest, which is
    /// what makes "the most recent cube" a property of state rather than of a
    /// list living outside the world.
    pub order: u64,
}

impl SimTransform for Cube {
    fn world_position(&self) -> sim::DVec3 {
        self.position
    }

    fn orientation(&self) -> sim::DQuat {
        let turn = (self.spin_ticks % TICKS_PER_TURN) as f64 / TICKS_PER_TURN as f64;
        let angle = turn * core::f64::consts::TAU;
        sim::DQuat::from_axis_angle(sim::DVec3::new(0.0, 1.0, 0.0), angle).mul(
            sim::DQuat::from_axis_angle(sim::DVec3::new(1.0, 0.0, 0.0), angle * 0.5),
        )
    }
}

/// Why the sim could not be built or driven.
#[derive(Debug, thiserror::Error)]
pub enum SimError {
    /// The world refused a component operation.
    #[error(transparent)]
    World(#[from] WorldError),
    /// The bindings did not parse.
    #[error(transparent)]
    Map(#[from] MapError),
    /// The replay is unreadable, or means something else than it did.
    #[error(transparent)]
    Replay(#[from] ReplayError),
    /// A query's borrows could not be granted — a code error, surfaced rather
    /// than unwrapped.
    #[error("query rejected: {0}")]
    Query(#[from] gg_ecs::AliasError),
    /// The camera entity is missing, which only a bug can produce.
    #[error("the sim has no camera")]
    NoCamera,
    /// The bindings parsed but declare no context by that name.
    #[error("the action map declares no `{0}` context")]
    NoContext(&'static str),
}

/// Demo 02's world and the tick it is on.
pub struct Sim {
    /// The sim state. Public because the extract stage and the tests read it,
    /// and because a demo has no reason to wrap an ECS in accessors.
    pub world: World,
    tick: u64,
    camera: Entity,
    seed: u64,
}

impl Sim {
    /// A fresh sim: one camera at [`CameraRig::GOLDEN`] and one cube at the
    /// origin. `seed` is carried for the replay header — this sim draws no
    /// random numbers, so it changes no state; the chaos *generator* is what
    /// consumes it (§5.11).
    pub fn new(seed: u64) -> Result<Self, SimError> {
        Self::new_at(seed, sim::DVec3::new(0.0, 0.0, 0.0))
    }

    /// The same sim, with the whole world translated by `origin`.
    ///
    /// Not a feature the demo needs — it is how the far-camera claim gets
    /// tested with the *actual* scene rather than a synthetic one
    /// ([`FAR_ORIGIN`]).
    pub fn new_at(seed: u64, origin: sim::DVec3) -> Result<Self, SimError> {
        let mut world = World::new();
        world.register::<CameraRig>().map_err(WorldError::from)?;
        world.register::<Cube>().map_err(WorldError::from)?;

        let camera = world.spawn();
        world.insert(
            camera,
            CameraRig {
                position: CameraRig::GOLDEN.position + origin,
                ..CameraRig::GOLDEN
            },
        )?;
        let cube = world.spawn();
        world.insert(
            cube,
            Cube {
                position: ORIGIN_CUBE + origin,
                spin_ticks: 0,
                order: 0,
            },
        )?;
        Ok(Sim {
            world,
            tick: 0,
            camera,
            seed,
        })
    }

    /// Ticks elapsed.
    pub fn tick_count(&self) -> u64 {
        self.tick
    }

    /// The seed the replay header records.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The camera's current pose.
    pub fn camera(&self) -> Result<CameraRig, SimError> {
        self.world
            .get::<CameraRig>(self.camera)
            .copied()
            .ok_or(SimError::NoCamera)
    }

    /// How many cubes exist.
    pub fn cube_count(&self) -> Result<usize, SimError> {
        let query = Query::<&Cube>::new()?;
        let mut count = 0;
        self.world.each_ref(&query, |_, _: &Cube| count += 1);
        Ok(count)
    }

    /// Advance one tick.
    ///
    /// The order of the three steps below **is** the schedule (§4.1: the loop
    /// is the scheduler). Changing it changes the simulation, which is exactly
    /// why it is written out in one place and read top to bottom.
    pub fn tick(&mut self, input: &Input) -> Result<(), SimError> {
        self.move_camera(input)?;
        self.spawn_and_despawn(input)?;
        self.spin_cubes()?;
        self.tick += 1;
        Ok(())
    }

    fn move_camera(&mut self, input: &Input) -> Result<(), SimError> {
        let looking = input.pressed(LOOK);
        let (look_x, look_y) = (input.axis(LOOK_X), input.axis(LOOK_Y));
        let axes = [
            f64::from(input.axis(MOVE_RIGHT)),
            f64::from(input.axis(MOVE_UP)),
            f64::from(input.axis(MOVE_FORWARD)),
        ];

        let rig = self
            .world
            .get_mut::<CameraRig>(self.camera)
            .ok_or(SimError::NoCamera)?;
        if looking {
            rig.yaw -= look_x * LOOK_PER_UNIT;
            rig.pitch = (rig.pitch - look_y * LOOK_PER_UNIT).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        }
        let (forward, right) = rig.basis();
        let up = sim::DVec3::new(0.0, 1.0, 0.0);
        // Not normalized: diagonals are faster, which is a fly camera's
        // traditional bug and this demo's traditional feature. Whatever it is,
        // it is the same on every architecture.
        let step = (forward * axes[2] + right * axes[0] + up * axes[1]) * MOVE_PER_TICK;
        rig.position += step;
        Ok(())
    }

    fn spawn_and_despawn(&mut self, input: &Input) -> Result<(), SimError> {
        if input.just_pressed(SPAWN) {
            let rig = self.camera()?;
            let (forward, _) = rig.basis();
            let (count, highest, _) = self.cube_extremes()?;
            if count < MAX_CUBES {
                let cube = self.world.spawn();
                self.world.insert(
                    cube,
                    Cube {
                        position: rig.position + forward * SPAWN_DISTANCE,
                        spin_ticks: 0,
                        order: highest.saturating_add(1),
                    },
                )?;
            }
        }
        if input.just_pressed(DESPAWN) {
            let (_, _, newest) = self.cube_extremes()?;
            if let Some(entity) = newest {
                self.world.despawn(entity);
            }
        }
        Ok(())
    }

    /// `(count, highest order, entity holding it)` — one pass, because both
    /// spawn and despawn want the same answer and querying twice would let
    /// them disagree.
    fn cube_extremes(&self) -> Result<(usize, u64, Option<Entity>), SimError> {
        let query = Query::<&Cube>::new()?;
        let (mut count, mut highest, mut newest) = (0usize, 0u64, None);
        self.world.each_ref(&query, |entity, cube: &Cube| {
            count += 1;
            if newest.is_none() || cube.order > highest {
                highest = cube.order;
                newest = Some(entity);
            }
        });
        Ok((count, highest, newest))
    }

    /// Shove the newest cube. Test machinery, and deliberately a method on
    /// `Sim` rather than a fork of the tick: a planted divergence must go
    /// through the same storage the real sim uses, or the gate is being proven
    /// against something the gate does not watch.
    pub fn plant(&mut self, nudge: f64) -> Result<(), SimError> {
        let (_, _, newest) = self.cube_extremes()?;
        if let Some(entity) = newest
            && let Some(cube) = self.world.get_mut::<Cube>(entity)
        {
            cube.position.x += nudge;
        }
        Ok(())
    }

    fn spin_cubes(&mut self) -> Result<(), SimError> {
        let query = Query::<&mut Cube>::new()?;
        self.world.each(&query, |_, cube: &mut Cube| {
            cube.spin_ticks = (cube.spin_ticks + 1) % TICKS_PER_TURN;
        });
        Ok(())
    }
}

/// An [`Input`] over the demo's bindings, with the game context pushed.
pub fn input() -> Result<Input, SimError> {
    let map = ActionMap::parse(BINDINGS, ACTIONS, AXES)?;
    let mut input = Input::new(map);
    let game = input
        .map()
        .context("game")
        .ok_or(SimError::NoContext("game"))?;
    input.push_context(game);
    Ok(input)
}

/// A deliberately planted divergence, for proving the gate catches one (§6 M4B
/// exit). Applied *after* the named tick's systems have run, so it is
/// indistinguishable from a sim bug that only bites on one machine.
#[derive(Clone, Copy, Debug)]
pub struct Plant {
    /// The tick to corrupt.
    pub tick: u64,
    /// How far to shove the newest cube, in metres.
    pub nudge: f64,
}

/// Drive `replay` for `until` ticks, returning the sim and the canonical state
/// hash after every tick.
///
/// The one implementation behind every replay path: the determinism gate, the
/// divergence locator, and the planted-divergence proof all come through here,
/// so none of them can accidentally simulate something slightly different from
/// the others.
pub fn run(
    replay: &Replay,
    until: u64,
    plant: Option<Plant>,
) -> Result<(Sim, Vec<CanonicalHash>), SimError> {
    replay.check_verbs(ACTIONS, AXES)?;
    let mut sim = Sim::new(replay.meta().seed)?;
    let mut input = input()?;
    let mut hashes = Vec::with_capacity(until as usize);
    for tick in 0..until {
        input.tick_from(replay.frame(tick));
        sim.tick(&input)?;
        if let Some(plant) = plant
            && plant.tick == tick
        {
            sim.plant(plant.nudge)?;
        }
        hashes.push(sim.world.canonical_hash());
    }
    Ok((sim, hashes))
}

/// The canonical state hash after every tick of `replay`.
///
/// This is what the determinism gate compares — on the same runner twice, on
/// three architectures, and across build tiers (§5.6). The **sequence**, not
/// just the final hash: a divergence at tick 40 that a later tick masks is
/// exactly the failure one end-state hash would miss.
pub fn hash_sequence(replay: &Replay) -> Result<Vec<CanonicalHash>, SimError> {
    Ok(run(replay, replay.ticks(), None)?.1)
}

/// The world as it stood at `tick` — the second half of divergence reporting,
/// once the sequence has named the tick (§5.6: "a bare hashes-differ on frame
/// 9,000 is a wasted day").
pub fn world_at(replay: &Replay, tick: u64, plant: Option<Plant>) -> Result<World, SimError> {
    Ok(run(replay, tick, plant)?.0.world)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use gg_input::{InputFrame, Recorder, ReplayMeta};

    fn meta(seed: u64) -> ReplayMeta {
        ReplayMeta {
            engine_commit: gg_input::ENGINE_COMMIT.into(),
            contract: gg_math::DETERMINISM_CONTRACT,
            tier: "test".into(),
            tick_hz: TICK_HZ,
            seed,
            actions: ACTIONS.iter().map(|s| (*s).to_owned()).collect(),
            axes: AXES.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    #[test]
    fn the_declared_verbs_and_the_bindings_agree() {
        // The map parses against the *declared* lists, so a binding naming a
        // verb the code does not have fails here rather than in a demo run.
        let map = ActionMap::parse(BINDINGS, ACTIONS, AXES).unwrap();
        assert!(map.context("game").is_some());
        assert_eq!(map.action_names().len(), ACTIONS.len());
        assert_eq!(map.axis_names().len(), AXES.len());
    }

    #[test]
    fn a_fresh_sim_holds_one_camera_and_one_cube() {
        let sim = Sim::new(0).unwrap();
        assert_eq!(sim.cube_count().unwrap(), 1);
        assert_eq!(sim.camera().unwrap(), CameraRig::GOLDEN);
        assert_eq!(sim.tick_count(), 0);
    }

    #[test]
    fn the_spin_is_a_counter_so_a_full_turn_returns_to_where_it_started() {
        let mut sim = Sim::new(0).unwrap();
        let input = input().unwrap();
        let at_start = sim.world.canonical_hash();
        for _ in 0..TICKS_PER_TURN {
            sim.tick(&input).unwrap();
        }
        assert_eq!(
            sim.world.canonical_hash(),
            at_start,
            "a full turn with no input must return the world to its opening state"
        );
    }

    #[test]
    fn spawn_and_despawn_are_last_in_first_out_by_state_alone() {
        let mut sim = Sim::new(0).unwrap();
        let mut input = input().unwrap();
        let mut frame = InputFrame::default();

        for _ in 0..3 {
            frame.buttons = 1 << SPAWN.index();
            input.tick_from(frame);
            sim.tick(&input).unwrap();
            frame.buttons = 0; // release: spawn is an edge, not a state
            input.tick_from(frame);
            sim.tick(&input).unwrap();
        }
        assert_eq!(sim.cube_count().unwrap(), 4);

        frame.buttons = 1 << DESPAWN.index();
        input.tick_from(frame);
        sim.tick(&input).unwrap();
        assert_eq!(sim.cube_count().unwrap(), 3);

        // The one left standing after despawning everything is the original.
        for _ in 0..8 {
            frame.buttons = 1 << DESPAWN.index();
            input.tick_from(frame);
            sim.tick(&input).unwrap();
            frame.buttons = 0;
            input.tick_from(frame);
            sim.tick(&input).unwrap();
        }
        assert_eq!(sim.cube_count().unwrap(), 0, "even the first cube goes");
    }

    #[test]
    fn a_held_spawn_key_spawns_once_and_the_ceiling_holds() {
        let mut sim = Sim::new(0).unwrap();
        let mut input = input().unwrap();
        for _ in 0..50 {
            input.tick_from(InputFrame {
                buttons: 1 << SPAWN.index(),
                axes: [0; gg_input::MAX_AXES],
            });
            sim.tick(&input).unwrap();
        }
        assert_eq!(sim.cube_count().unwrap(), 2, "held is one edge, not fifty");
    }

    #[test]
    fn the_camera_moves_only_where_the_axes_point() {
        let mut sim = Sim::new(0).unwrap();
        let mut input = input().unwrap();
        let mut axes = [0; gg_input::MAX_AXES];
        axes[MOVE_UP.index()] = gg_input::AXIS_SCALE;
        input.tick_from(InputFrame { buttons: 0, axes });
        sim.tick(&input).unwrap();

        let rig = sim.camera().unwrap();
        assert_eq!(rig.position.y, CameraRig::GOLDEN.position.y + MOVE_PER_TICK);
        assert_eq!(rig.position.x, CameraRig::GOLDEN.position.x);
        assert_eq!(rig.yaw, CameraRig::GOLDEN.yaw, "no look input, no turn");
    }

    #[test]
    fn look_turns_only_while_look_is_held() {
        let mut sim = Sim::new(0).unwrap();
        let mut input = input().unwrap();
        let mut axes = [0; gg_input::MAX_AXES];
        axes[LOOK_X.index()] = gg_input::AXIS_SCALE;

        input.tick_from(InputFrame { buttons: 0, axes });
        sim.tick(&input).unwrap();
        assert_eq!(sim.camera().unwrap().yaw, CameraRig::GOLDEN.yaw);

        input.tick_from(InputFrame {
            buttons: 1 << LOOK.index(),
            axes,
        });
        sim.tick(&input).unwrap();
        assert_eq!(
            sim.camera().unwrap().yaw,
            CameraRig::GOLDEN.yaw - LOOK_PER_UNIT
        );
    }

    #[test]
    fn pitch_stops_short_of_the_pole_however_hard_the_mouse_is_pushed() {
        let mut sim = Sim::new(0).unwrap();
        let mut input = input().unwrap();
        let mut axes = [0; gg_input::MAX_AXES];
        axes[LOOK_Y.index()] = -gg_input::AXIS_SCALE * 100;
        for _ in 0..200 {
            input.tick_from(InputFrame {
                buttons: 1 << LOOK.index(),
                axes,
            });
            sim.tick(&input).unwrap();
        }
        assert_eq!(sim.camera().unwrap().pitch, PITCH_LIMIT);
        // ...and the basis is still a basis at the limit.
        let (forward, right) = sim.camera().unwrap().basis();
        assert!(forward.cross(right).length() > 0.5);
    }

    #[test]
    fn one_replay_produces_one_hash_sequence_however_often_it_is_run() {
        // §5.6a in miniature: the same runner, twice. The gate runs this over
        // the checked-in curated replay; here it is the property under test.
        let mut recorder = Recorder::new(meta(9));
        for tick in 0..120u64 {
            let mut axes = [0; gg_input::MAX_AXES];
            axes[MOVE_FORWARD.index()] = gg_input::AXIS_SCALE;
            axes[LOOK_X.index()] = (tick as i32 % 7) - 3;
            recorder.record(
                tick,
                InputFrame {
                    buttons: u64::from(tick % 30 == 0) << SPAWN.index(),
                    axes,
                },
            );
        }
        let replay = recorder.finish();
        let a = hash_sequence(&replay).unwrap();
        let b = hash_sequence(&replay).unwrap();
        assert_eq!(a.len(), 120);
        assert_eq!(a, b);
        // The replay actually did something — a sequence of 120 identical
        // hashes would pass the comparison above and prove nothing.
        assert!(a.iter().collect::<std::collections::BTreeSet<_>>().len() > 100);
    }

    #[test]
    fn a_world_replayed_to_a_tick_matches_the_sim_that_walked_there() {
        let mut recorder = Recorder::new(meta(3));
        for tick in 0..40u64 {
            let mut axes = [0; gg_input::MAX_AXES];
            axes[MOVE_RIGHT.index()] = gg_input::AXIS_SCALE;
            recorder.record(tick, InputFrame { buttons: 0, axes });
        }
        let replay = recorder.finish();
        let sequence = hash_sequence(&replay).unwrap();
        for tick in [1u64, 17, 40] {
            let world = world_at(&replay, tick, None).unwrap();
            assert_eq!(
                world.canonical_hash(),
                sequence[tick as usize - 1],
                "world_at({tick}) disagreed with the sequence"
            );
        }
    }
}
