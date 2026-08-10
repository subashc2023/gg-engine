//! Demo 11 — **the platformer** (§6 M20): run, jump, fall, arrive.
//!
//! The first rung of the game ladder, and the demo that pulls the orthographic
//! camera (§6 M20 pull 1). This chunk is the *game*: a box that runs and jumps
//! through a level of boxes to a goal, with the three feel counters — coyote
//! time, jump buffering, variable jump height — as hashed sim state, so how a
//! jump feels is a reloadable edit and a replayable fact (§6 M20 pull 4).
//!
//! The camera is [`Eye::flat`] — orthographic, §6 M20 pull 1 delivered — on an
//! eased [`Rig`]. The level is the checked-in `scene.ggsave` beside
//! `input.toml` (§6 M20 pull 2): the shell opens it in every live session, the
//! editor authors it, and [`bootstrap`] deals only the player's kit — the first
//! demo whose world is not spawned by its own `bootstrap`.
//!
//! All collision is here, not in the engine: a few axis-aligned rectangles and
//! a swept point, in `gg_math::sim` terms (§6 M20 pull 3). Physics is f64
//! `+ - * /` and comparisons only — nothing transcendental crosses a tick.

use gg_ecs::Component;
use gg_ecs::boundary::{
    ActionId, AxisId, Eye, GameWorld, Light, Renderable, Sound, Widget, log_level, wave, widget_id,
};
use gg_math::sim;

pub mod session;

// ---------------------------------------------------------------- feel ------

/// Metres per tick at full stick. ~6.6 m/s at 60 Hz.
pub const RUN_SPEED: f64 = 0.11;
/// How fast the run speed is reached on the ground, metres per tick per tick.
pub const GROUND_ACCEL: f64 = 0.012;
/// Air control: same shape, weaker. A platformer with none feels like a shove.
pub const AIR_ACCEL: f64 = 0.008;
/// Downward pull, metres per tick per tick. ~3.6× real gravity — arcade, not
/// ballistics.
pub const GRAVITY: f64 = 0.010;
/// Gravity multiplier while rising with the jump key already released — the
/// variable-height half of the feel. Stateless on purpose: cutting velocity
/// once would need a "was cut" flag; a stronger pull needs nothing.
pub const RISE_CUT: f64 = 3.0;
/// Terminal fall speed, metres per tick. Also the anti-tunnelling bound: it
/// must stay below every slab's full thickness.
pub const MAX_FALL: f64 = 0.28;
/// Upward velocity a jump starts with. Peak ≈ v²/2g ≈ 2.2 m held — the bound
/// every rise the scene authors must sit under, with slack.
pub const JUMP_VELOCITY: f64 = 0.21;
/// Ticks after leaving a ledge in which a jump still fires. 100 ms at 60 Hz.
pub const COYOTE_TICKS: u32 = 6;
/// Ticks a jump press waits for the ground to arrive. Same 100 ms.
pub const BUFFER_TICKS: u32 = 6;

// ---------------------------------------------------------------- stage -----

/// Player half-width, metres.
pub const PLAYER_HALF_W: f32 = 0.35;
/// Player half-height, metres. Feet are `position.y - PLAYER_HALF_H`.
pub const PLAYER_HALF_H: f32 = 0.45;
/// Player half-depth — presentation only; the sim is the z = 0 plane.
pub const PLAYER_DEPTH: f32 = 0.35;
/// Where a run begins and a death returns to, metres. Above the start pad, so
/// the opening tick is a short fall and the first sound the game makes is a
/// landing — which is also what pins the land cue in a test.
pub const START: sim::DVec2 = sim::DVec2::new(-4.0, 1.0);
/// Below this the fall is a death and [`arrive`] respawns. Deeper than any pit
/// by a margin, so a maximal jump into a gap still dies rather than escaping.
pub const KILL_Y: f64 = -8.0;
/// The player's colour.
pub const PLAYER_INK: u32 = 0x00f0_f4ff;
/// Which way the sun points — the template's, for the template's reason: not
/// vertical, so the one shadow it casts proves something.
pub const SUN: sim::Vec3 = sim::Vec3::new(-0.4, -1.0, -0.35);

// ---------------------------------------------------------------- camera ----

/// Per-tick fraction of the remaining distance the camera closes. Exponential
/// smoothing in exact f64, so two machines agree on where the eye is.
pub const CAMERA_K: f64 = 0.12;
/// Eye height above the rig, metres.
pub const CAMERA_LIFT: f64 = 1.6;
/// Eye distance out of the playfield plane, metres. Orthographic, so this
/// frames nothing — it only keeps the slab inside `[r.near, r.ortho_far]`.
pub const CAMERA_BACK: f64 = 14.0;
/// Half the world metres the window is tall (§6 M20's pull 1): the jump apex,
/// the floaters and a slab of sky fit; width follows the window's aspect.
pub const CAMERA_HALF_HEIGHT: f32 = 4.5;

// ---------------------------------------------------------------- verbs -----

/// Jump — buffered [`BUFFER_TICKS`], honoured within [`COYOTE_TICKS`].
pub const JUMP: ActionId = ActionId::new(0);
/// Wipe the world; `bootstrap` rebuilds it the same tick (demo 03's rule).
pub const RESTART: ActionId = ActionId::new(1);
/// Left/right, −1..=1.
pub const MOVE: AxisId = AxisId::new(0);

// ---------------------------------------------------------------- cues ------

/// [`Cue::kind`]: the jump blip.
pub const CUE_JUMP: u32 = 0;
/// The landing thud.
pub const CUE_LAND: u32 = 1;
/// The falling-off-the-world slide.
pub const CUE_DEATH: u32 = 2;
/// The goal chime.
pub const CUE_GOAL: u32 = 3;
/// How many cue entities `bootstrap` deals.
pub const CUES: usize = 4;

// ---------------------------------------------------------------- hud -------

/// [`Hud::line`]: the running clock.
pub const HUD_TIME: u32 = 0;
/// The death count.
pub const HUD_DEATHS: u32 = 1;
/// The goal banner — zero-rect until the run is finished (§6 M18's rule: a
/// hidden widget is a zero rect, never a despawn).
pub const HUD_GOAL: u32 = 2;

// ---------------------------------------------------------------- state -----

/// The player: pose, motion, and the three feel counters. All sim state, all
/// hashed — a divergent coyote window is a named tick, not a feel opinion.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "platformer.player")]
#[repr(C)]
pub struct Player {
    /// Centre, metres, z = 0 plane. `f64`, never narrowed sim-side (§4.2.1).
    pub position: sim::DVec2,
    /// Metres per tick.
    pub velocity: sim::DVec2,
    /// Ticks of ledge forgiveness left. Refilled while supported.
    pub coyote: u32,
    /// Ticks the last jump press stays willing.
    pub buffer: u32,
    /// Standing on something this tick. `u32` for `Pod`; 0 or 1.
    pub grounded: u32,
    /// Padding, named (§4.2.1 hazard 4).
    pub _pad: u32,
}

/// Marks a [`Renderable`] the player collides with. The geometry lives in the
/// `Renderable` itself — what you see is what you stand on, which is the whole
/// mechanism by which an editor-authored slab is solid (§6 M20 pull 2).
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "platformer.solid")]
#[repr(C)]
pub struct Solid {
    /// Padding, named — the marker is the component.
    pub _pad: u32,
}

/// Marks the goal box. Not solid: the player finishes by overlapping it.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "platformer.goal")]
#[repr(C)]
pub struct Goal {
    /// Padding, named — the marker is the component.
    pub _pad: u32,
}

/// The run: when it started, when it finished, what it cost. One per world.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "platformer.run")]
#[repr(C)]
pub struct Run {
    /// The tick `bootstrap` dealt this run.
    pub started_at: u64,
    /// The tick the goal was reached; 0 while still running. Tick 0 cannot be
    /// a finish — the goal is forty metres from the start.
    pub finished_at: u64,
    /// Falls below [`KILL_Y`]. A death keeps the clock: the time is the run's.
    pub deaths: u32,
    /// Padding, named.
    pub _pad: u32,
}

/// The camera rig: where the eye is easing toward the player from. State,
/// because a camera that snapped on reload would report a smoothing bug that
/// is not there.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "platformer.rig")]
#[repr(C)]
pub struct Rig {
    /// Where the rig is, metres, z = 0 plane.
    pub position: sim::DVec2,
}

/// Names the [`Sound`] beside it, so a system can bump one cue without holding
/// entity ids (demo 10's shape).
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "platformer.cue")]
#[repr(C)]
pub struct Cue {
    /// One of the `CUE_*` constants.
    pub kind: u32,
}

/// Names the [`Widget`] beside it, for the same reason as [`Cue`].
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "platformer.hud")]
#[repr(C)]
pub struct Hud {
    /// One of the `HUD_*` constants.
    pub line: u32,
}

// ---------------------------------------------------------------- systems ---

/// Reset the run in place on `restart` — player home, clock and deaths zeroed,
/// camera snapped. Not a wipe: the level is the scene's (§6 M20 pull 2), not
/// this system's to redeal, so `R` cannot erase what nothing here could deal
/// back.
pub fn restart(world: &mut GameWorld) {
    if !world.just_pressed(RESTART) {
        return;
    }
    let now = world.tick();
    world.visit::<&mut Player>(|_, player| {
        *player = Player {
            position: START,
            velocity: sim::DVec2::ZERO,
            coyote: 0,
            buffer: 0,
            grounded: 0,
            _pad: 0,
        };
    });
    world.visit::<&mut Run>(|_, run| {
        *run = Run {
            started_at: now,
            finished_at: 0,
            deaths: 0,
            _pad: 0,
        };
    });
    // Snapped, not eased: a restart is a cut, and half a second of the camera
    // flying home would replay the death the player just asked to forget.
    world.visit::<&mut Rig>(|_, rig| rig.position = START);
    world.log(log_level::INFO, "platformer: restarted");
}

/// Deal the player's kit if there is none: player, sun, rig, run clock, cues,
/// HUD. Never the level — that is the checked-in scene's (§6 M20 pull 2),
/// loaded as data before this ever runs, and a world opened without one has
/// nothing to stand on, visibly. Idempotent by asking the world, not by
/// remembering (§4.2.2) — a reload no-op, and stood down entirely when the
/// scene supplied the player.
pub fn bootstrap(world: &mut GameWorld) {
    let mut exists = false;
    world.visit::<&Player>(|_, _| exists = true);
    if exists {
        return;
    }
    let tick = world.tick();

    let player = world.spawn();
    if let Err(refused) = world.insert(
        player,
        Player {
            position: START,
            velocity: sim::DVec2::ZERO,
            coyote: 0,
            buffer: 0,
            grounded: 0,
            _pad: 0,
        },
    ) {
        world.log(log_level::ERROR, &refused.to_string());
        return;
    }
    world.put(player, player_look(START));

    world.spawn_with(Light::sun(SUN, 0x00ff_f4e0, 3.5));

    let rig = world.spawn_with(Run {
        started_at: tick,
        finished_at: 0,
        deaths: 0,
        _pad: 0,
    });
    world.put(rig, Rig { position: START });
    world.put(
        rig,
        Eye::flat(
            sim::DVec3::new(START.x, START.y + CAMERA_LIFT, CAMERA_BACK),
            CAMERA_HALF_HEIGHT,
        ),
    );

    for (kind, sound) in [
        (CUE_JUMP, chirp(wave::SQUARE, 340.0, 560.0, 90, 0.22)),
        (CUE_LAND, chirp(wave::TRIANGLE, 140.0, 140.0, 50, 0.18)),
        (CUE_DEATH, chirp(wave::SQUARE, 220.0, 70.0, 300, 0.28)),
        (CUE_GOAL, chirp(wave::SINE, 520.0, 1040.0, 450, 0.30)),
    ] {
        let entity = world.spawn_with(Cue { kind });
        world.put(entity, sound);
    }

    for (line, mut widget) in [
        (
            HUD_TIME,
            Widget::label([8.0, 6.0, 150.0, 12.0], 0xffe8_f0ff, "TIME 0.0"),
        ),
        (
            HUD_DEATHS,
            Widget::label([8.0, 20.0, 150.0, 12.0], 0xffe8_f0ff, "DEATHS 0"),
        ),
        // Hidden until the run finishes — shown by giving it its rect back.
        (HUD_GOAL, Widget::label([0.0; 4], 0xffff_c832, "")),
    ] {
        widget.id = widget_id("plat.hud") ^ u64::from(line);
        widget.order = line + 1;
        let entity = world.spawn_with(Hud { line });
        world.put(entity, widget);
    }

    world.log(log_level::INFO, "platformer: ready");
}

/// One tick of movement: steer, gravity, integrate-and-resolve one axis at a
/// time, then the buffered/coyote jump. X resolves before Y so a step's wall
/// is a wall while the feet are still on last tick's resolved ground.
pub fn step(world: &mut GameWorld) {
    let stick = f64::from(world.axis(MOVE));
    let jump_held = world.pressed(JUMP);
    let jump_edge = world.just_pressed(JUMP);
    let solids = solids(world);

    let mut jumped = false;
    let mut landed = false;
    world.visit::<&mut Player>(|_, player| {
        let target = stick.clamp(-1.0, 1.0) * RUN_SPEED;
        let accel = if player.grounded != 0 {
            GROUND_ACCEL
        } else {
            AIR_ACCEL
        };
        player.velocity.x += (target - player.velocity.x).clamp(-accel, accel);

        // The variable-jump half: a released rise falls under heavier gravity.
        let pull = if player.velocity.y > 0.0 && !jump_held {
            GRAVITY * RISE_CUT
        } else {
            GRAVITY
        };
        player.velocity.y = (player.velocity.y - pull).max(-MAX_FALL);

        player.buffer = if jump_edge {
            BUFFER_TICKS
        } else {
            player.buffer.saturating_sub(1)
        };

        player.position.x += player.velocity.x;
        resolve_x(player, &solids);

        let was_grounded = player.grounded != 0;
        player.position.y += player.velocity.y;
        let supported = resolve_y(player, &solids);
        player.grounded = u32::from(supported);
        player.coyote = if supported {
            COYOTE_TICKS
        } else {
            player.coyote.saturating_sub(1)
        };
        if supported && !was_grounded {
            landed = true;
        }

        // After the landing check, so a buffered press fires on the landing
        // tick itself — the buffer's whole reason to exist.
        if player.buffer > 0 && player.coyote > 0 {
            player.velocity.y = JUMP_VELOCITY;
            player.buffer = 0;
            player.coyote = 0;
            player.grounded = 0;
            jumped = true;
        }
    });

    if jumped {
        cue(world, CUE_JUMP);
    } else if landed {
        cue(world, CUE_LAND);
    }
}

/// The run's two endings: the goal box reached, or the world fallen out of.
pub fn arrive(world: &mut GameWorld) {
    let mut pose = None;
    world.visit::<&Player>(|_, player| pose = Some(*player));
    let Some(player) = pose else {
        return;
    };

    if player.position.y < KILL_Y {
        world.visit::<&mut Player>(|_, p| {
            p.position = START;
            p.velocity = sim::DVec2::ZERO;
            p.coyote = 0;
            p.buffer = 0;
            p.grounded = 0;
        });
        world.visit::<&mut Run>(|_, run| run.deaths = run.deaths.saturating_add(1));
        cue(world, CUE_DEATH);
        world.log(log_level::INFO, "platformer: fell");
        return;
    }

    let mut goal = None;
    world.visit::<(&Goal, &Renderable)>(|_, (_, shape)| goal = Some(*shape));
    let Some(shape) = goal else {
        return;
    };
    let meets = (player.position.x - shape.position.x).abs()
        < f64::from(PLAYER_HALF_W) + f64::from(shape.half_extent.x)
        && (player.position.y - shape.position.y).abs()
            < f64::from(PLAYER_HALF_H) + f64::from(shape.half_extent.y);
    if !meets {
        return;
    }
    let tick = world.tick();
    let mut first = false;
    world.visit::<&mut Run>(|_, run| {
        if run.finished_at == 0 {
            run.finished_at = tick;
            first = true;
        }
    });
    if first {
        cue(world, CUE_GOAL);
        world.log(log_level::INFO, "platformer: goal");
    }
}

/// Say what the tick looks like: the player's box, the eased camera, the HUD.
/// Last in the table, so it describes the tick that just happened (§4.5 v0).
pub fn present(world: &mut GameWorld) {
    let mut pose = None;
    world.visit::<&Player>(|entity, player| pose = Some((entity, *player)));
    if let Some((entity, player)) = pose {
        world.put(entity, player_look(player.position));

        let mut eased = None;
        world.visit::<&mut Rig>(|rig_entity, rig| {
            rig.position.x += (player.position.x - rig.position.x) * CAMERA_K;
            rig.position.y += (player.position.y - rig.position.y) * CAMERA_K;
            eased = Some((rig_entity, rig.position));
        });
        if let Some((rig_entity, at)) = eased {
            world.put(
                rig_entity,
                Eye::flat(
                    sim::DVec3::new(at.x, at.y + CAMERA_LIFT, CAMERA_BACK),
                    CAMERA_HALF_HEIGHT,
                ),
            );
        }
    }

    let mut state = None;
    world.visit::<&Run>(|_, run| state = Some(*run));
    let Some(run) = state else {
        return;
    };
    let hz = u64::from(world.tick_hz()).max(1);
    let end = if run.finished_at != 0 {
        run.finished_at
    } else {
        world.tick()
    };
    let tenths = end.saturating_sub(run.started_at) * 10 / hz;

    use core::fmt::Write as _;
    let mut clock = Line::default();
    let _ = write!(clock, "TIME {}.{}", tenths / 10, tenths % 10);
    let mut toll = Line::default();
    let _ = write!(toll, "DEATHS {}", run.deaths);
    let mut banner = Line::default();
    let _ = write!(banner, "GOAL {}.{}", tenths / 10, tenths % 10);
    let finished = run.finished_at != 0;

    world.visit::<(&Hud, &mut Widget)>(|_, (hud, widget)| match hud.line {
        HUD_TIME => widget.set_text(clock.as_str()),
        HUD_DEATHS => widget.set_text(toll.as_str()),
        HUD_GOAL => {
            widget.rect = if finished {
                [264.0, 160.0, 140.0, 16.0]
            } else {
                [0.0; 4]
            };
            widget.set_text(banner.as_str());
        }
        _ => {}
    });
}

// ---------------------------------------------------------------- innards ---

/// The player's box, drawn where the sim says it is.
fn player_look(at: sim::DVec2) -> Renderable {
    Renderable::boxed(
        sim::DVec3::new(at.x, at.y, 0.0),
        sim::Vec3::new(PLAYER_HALF_W, PLAYER_HALF_H, PLAYER_DEPTH),
        PLAYER_INK,
    )
}

/// A swept tone. [`Sound::tone`] with the end frequency opened up.
fn chirp(wave: u32, from_hz: f32, to_hz: f32, ms: u32, gain: f32) -> Sound {
    let mut sound = Sound::tone(wave, from_hz, ms, gain);
    sound.hz_to = to_hz;
    sound
}

/// Bump one cue's sequence — the whole trigger idiom (§4.2.2's audio protocol).
fn cue(world: &mut GameWorld, kind: u32) {
    world.visit::<(&Cue, &mut Sound)>(|_, (cue, sound)| {
        if cue.kind == kind {
            sound.seq = sound.seq.wrapping_add(1);
        }
    });
}

/// A solid, widened to f64 once so the resolve arithmetic is one width.
struct Box2 {
    x: f64,
    y: f64,
    hw: f64,
    hh: f64,
}

/// Every solid's box, read from its `Renderable` — the level *is* the picture,
/// which is what lets the editor author it (§6 M20 pull 2).
fn solids(world: &mut GameWorld) -> Vec<Box2> {
    let mut out = Vec::new();
    world.visit::<(&Solid, &Renderable)>(|_, (_, shape)| {
        out.push(Box2 {
            x: shape.position.x,
            y: shape.position.y,
            hw: f64::from(shape.half_extent.x),
            hh: f64::from(shape.half_extent.y),
        });
    });
    out
}

/// Push the player out of any slab along x. Overlap is strict, so exact
/// contact — which is what a resolve leaves behind — is not a collision.
fn resolve_x(player: &mut Player, solids: &[Box2]) {
    for slab in solids {
        let dx = player.position.x - slab.x;
        let dy = player.position.y - slab.y;
        let span_x = f64::from(PLAYER_HALF_W) + slab.hw;
        let span_y = f64::from(PLAYER_HALF_H) + slab.hh;
        if dx.abs() < span_x && dy.abs() < span_y {
            let push = span_x - dx.abs();
            player.position.x += if dx >= 0.0 { push } else { -push };
            player.velocity.x = 0.0;
        }
    }
}

/// Push the player out of any slab along y. Returns whether something took the
/// player's weight — a push *up* against downward motion, i.e. ground.
fn resolve_y(player: &mut Player, solids: &[Box2]) -> bool {
    let mut supported = false;
    for slab in solids {
        let dx = player.position.x - slab.x;
        let dy = player.position.y - slab.y;
        let span_x = f64::from(PLAYER_HALF_W) + slab.hw;
        let span_y = f64::from(PLAYER_HALF_H) + slab.hh;
        if dx.abs() < span_x && dy.abs() < span_y {
            if dy >= 0.0 {
                player.position.y = slab.y + span_y;
                if player.velocity.y <= 0.0 {
                    supported = true;
                    player.velocity.y = 0.0;
                }
            } else {
                player.position.y = slab.y - span_y;
                if player.velocity.y > 0.0 {
                    player.velocity.y = 0.0;
                }
            }
        }
    }
    supported
}

/// A fixed-capacity HUD line. The HUD path runs every tick and allocates
/// nothing (§6 M18 item 2a's rule); truncation is fine — every line is ASCII.
#[derive(Default)]
struct Line {
    bytes: [u8; 32],
    len: usize,
}

impl core::fmt::Write for Line {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let take = s.len().min(self.bytes.len() - self.len);
        self.bytes[self.len..self.len + take].copy_from_slice(&s.as_bytes()[..take]);
        self.len += take;
        Ok(())
    }
}

impl Line {
    fn as_str(&self) -> &str {
        core::str::from_utf8(&self.bytes[..self.len]).unwrap_or("")
    }
}

// Order in `systems` is execution order (§4.1); order in the verb lists is the
// id space a replay records (§4.7). Neither is alphabetical, neither may drift.
#[cfg(feature = "game")]
gg_ecs::gg_game! {
    components: [Player, Solid, Goal, Run, Rig, Cue, Hud, Renderable, Light, Eye, Widget, Sound],
    actions: ["jump", "restart"],
    axes: ["move"],
    systems: [restart, bootstrap, step, arrive, present],
}
