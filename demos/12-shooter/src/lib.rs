//! Demo 12 — **the room** (the game ladder's third rung): stand up, look
//! around, walk, jump, land — and, since §6 M37, shoot.
//!
//! What Unreal opens into, in this engine's terms: a white-grey block room you
//! move through in first person. The first chunk was the *character controller*
//! and deliberately nothing else, on the argument that a controller which is
//! wrong is wrong under everything built on it. M37 is the second: the gun, the
//! targets and the score, which is what turns a room into a game with an end —
//! and an end is what a recorded session needs to record.
//!
//! # The shot
//!
//! Hitscan, and the ray is [`sim::DRay`] (§6 M37 item 1): a shot's answer
//! decides what dies, so it is in the tick and in the hash, which is why the
//! predicate lives in `gg_math::sim` rather than in the two host crates that
//! also own one. The policy on top of it is this game's alone — past the muzzle
//! (`enter > 0`, so a target you are standing inside is not one you can shoot)
//! and inside [`SHOT_RANGE`]. Targets and [`Solid`]s are cast against together
//! and the nearest wins, so a wall stops a bullet without knowing it is a wall.
//!
//! Everything the shot leaves behind is an ordinary entity with a life counter:
//! the tracer, the debris, and the muzzle flash — which is a [`Light`] that
//! exists for two ticks and is therefore the first light in this tree that
//! appears and vanishes while a scene is being played (§6 M37 item 4). Debris
//! is spent where it meets the floor rather than sinking through it, which is
//! also what keeps every transient inside the room's static bounds.
//!
//! Three things it inherits rather than invents. Mouse-look is demo 06's:
//! `aim_x`/`aim_y` are raw device deltas the shell records as fixed-point axes,
//! and `gg-platform` locks and hides the OS cursor for any game that does not
//! bind pointer verbs. Collision is demo 11's idiom lifted a dimension —
//! axis-separated integrate-and-resolve against every `(Solid, Renderable)` box,
//! strict overlap so exact contact is not a collision, horizontals before the
//! vertical so a wall is a wall while the feet still stand on last tick's
//! resolved ground. The feel counters — coyote time, jump buffering, variable
//! jump height — are `Walker` fields and therefore hashed, so a retune is a
//! reloadable edit and a replayable fact.
//!
//! What it adds is the one thing a 2D resolve never needed: a **step-up**. A
//! horizontal push-out that would block against a low ledge lifts onto it
//! instead, which is the whole difference between a character controller and a
//! box that slides — stairs and kerbs are walked, not jumped.
//!
//! # The pause menu, and one mouse doing two jobs
//!
//! Everything in the menu is ordinary declared state: [`Session`] is the
//! transport (paused, which page), the widgets are `Widget`s named by [`Menu`]
//! the way the HUD's are named by [`Hud`], and a click arrives as
//! `Widget::state` on the next tick. So the whole thing is in the hash and a
//! recorded session replays through it — the menu opening on the tick it opened,
//! the same button lighting up.
//!
//! The part worth reading the bindings for is the mouse. This game binds *both*
//! pointer protocols — raw counts for the camera, the steered cursor for the
//! menu — and the host arbitrates on one rule: it holds the mouse while the UI
//! has nothing to point at and hands it back while it has. A zero rect is how a
//! widget hides (§4.9), so [`lay_out`] shrinking the menu to nothing is also
//! what takes the pointer back, and neither half had to be told about the
//! other.
//!
//! Two settings cross to the host as [`Prefs`] rather than staying here, because
//! they are the host's to act on: master volume and the edge treatment. The
//! third — look sensitivity — never leaves, because the game is what turns a
//! mouse count into an angle.
//!
//! The level is a table here, not a `scene.ggsave`: demo 11 owns the
//! editor-authored level (§6 M20 pull 2), and this demo is about the thing that
//! moves through one. All physics is `f64` `+ - * /`, comparisons, and the one
//! `sqrt` IEEE mandates exactly — nothing transcendental crosses a tick except
//! the `sim::fly_basis` the blessed replays already pin.

pub mod session;

use gg_ecs::Component;
use gg_ecs::boundary::{
    ActionId, AxisId, Eye, GameWorld, Light, Prefs, QUIET_MAX, Renderable, Sky, Sound, Widget, aa,
    log_level, wave, widget_id,
};
use gg_math::sim;

// ---------------------------------------------------------------- feel ------

/// Metres per tick at full deflection. ~6.6 m/s at 60 Hz — brisk, near every
/// shooter's default. Deliberately *not* [`JUMP_VELOCITY`], which it nearly is:
/// two knobs that happen to agree should not read as one knob.
pub const WALK_SPEED: f64 = 0.110;
/// How fast that speed is reached on the ground, metres per tick per tick.
/// Full speed in seven ticks; the same number decelerates, so releasing the
/// keys stops you in seven.
pub const GROUND_ACCEL: f64 = 0.016;
/// Air control. Applied only while a direction is *asked for* — with no input
/// the accel is zero rather than a pull toward zero, so a jump keeps the speed
/// it left the ground with instead of being braked by the air.
pub const AIR_ACCEL: f64 = 0.004;
/// Downward pull, metres per tick per tick. ~19.8 m/s² — twice real, which is
/// what a shooter's arc has always been.
pub const GRAVITY: f64 = 0.0055;
/// Gravity multiplier while rising with the jump key already released — the
/// variable-height half of the feel. Stateless on purpose (demo 11's argument):
/// cutting the velocity once would need a "was cut" flag; a stronger pull needs
/// nothing.
pub const RISE_CUT: f64 = 2.5;
/// Upward velocity a jump starts with. Peak ≈ v²/2g ≈ 1.2 m — the bound every
/// rise in [`ROOM`] must sit under, with slack.
pub const JUMP_VELOCITY: f64 = 0.115;
/// Terminal fall speed, metres per tick. Also the anti-tunnelling bound: it
/// must stay below the full thickness of every slab in [`ROOM`], and the
/// thinnest of those is 0.4 m.
pub const MAX_FALL: f64 = 0.35;
/// Ticks after walking off a ledge in which a jump still fires. 100 ms at 60 Hz.
pub const COYOTE_TICKS: u32 = 6;
/// Ticks a jump press waits for the ground to arrive. Same 100 ms.
pub const BUFFER_TICKS: u32 = 6;
/// How far clear of a face a resolve leaves the body, metres.
///
/// A nanometre, and it is load-bearing. Ejecting *onto* a face is not enough:
/// `slab.z - slab.hz - HALF_W` and the `HALF_W + slab.hz` the overlap test
/// compares against are different sums of the same numbers, so they disagree in
/// the last bit, and half the time the disagreement leaves the body still
/// technically inside. The vertical resolve then reads that residue as a floor
/// or a ceiling — which is how a body that walked into a 4 m wall ended up
/// standing on top of it. A nanometre is ~500,000 ULPs at room scale and far
/// below what an `f32` render position can express, so it is invisible
/// everywhere except in the comparison it exists to settle.
pub const SKIN: f64 = 1e-9;

/// The tallest ledge the walk lifts onto instead of stopping against, metres.
/// Above every stair rise in [`ROOM`] (0.30) and below every crate meant to be
/// jumped (0.70) — the gap is what makes the two read as different objects.
pub const STEP_HEIGHT: f64 = 0.35;

/// Radians of turn per mouse count at [`SENS_ONE`] — the sensitivity scale's
/// unit, and the one number a look is finally made of.
///
/// 0.0005 rad is 0.0286 degrees, which puts a full turn at 12 566 counts: 40 cm
/// of desk at 800 DPI, 20 at 1600. That is an ordinary shooter's default and is
/// *seven times slower* than the 0.0035 this demo inherited from demo 06's fly
/// camera, which is worth stating because the difference is not only speed.
/// At the old value one mouse count turned the view 0.2 degrees, and 0.2 degrees
/// across a 90-degree window 1920 pixels wide is **four pixels** — so the
/// smallest motion a mouse can report was a four-pixel jump, and the camera
/// visibly stepped between two counts however slowly the hand moved. A fly
/// camera flown with a mouse never noticed; an aim does nothing else. Under a
/// count worth 0.6 of a pixel the same motion is sub-pixel and the step is gone.
pub const LOOK_PER_UNIT: f32 = 0.0005;
/// [`Session::sens`]'s unit: hundredths, so 100 is one times [`LOOK_PER_UNIT`].
/// An integer because it is hashed state and a float multiplier would put a
/// rounding rule in it (§4.7's argument about axes, applied to a menu).
pub const SENS_ONE: u32 = 100;
/// What a session opens at — one and a half, not one, because that is where
/// tuning it with a hand landed and a default is the tuned value or it is a
/// placeholder. [`SENS_ONE`] stays the scale's *unit* so [`LOOK_PER_UNIT`]'s
/// measurement above stays a true statement about one mouse count.
///
/// It is also the only sense in which the setting persists: nothing here is
/// read off disk (see [`Session`]), so every run opens here and a change lasts
/// the session.
pub const SENS_DEFAULT: u32 = 150;
/// The slowest and fastest the menu will go, and what one click moves.
pub const SENS_MIN: u32 = 10;
/// See [`SENS_MIN`].
pub const SENS_MAX: u32 = 400;
/// See [`SENS_MIN`].
pub const SENS_STEP: u32 = 5;
/// What one click of the volume row moves — an eighth of [`QUIET_MAX`], so the
/// row walks 100 % to silence in eight and every stop is exact.
pub const QUIET_STEP: u32 = QUIET_MAX / 8;
/// Pitch stops short of the pole, where the view basis collapses.
pub const PITCH_LIMIT: f32 = 1.55;

// ---------------------------------------------------------------- body ------

/// Half-width and half-depth of the body box, metres. Square in plan: a
/// rectangular body would collide differently depending on which way it faced,
/// and nothing here rotates the box.
pub const HALF_W: f64 = 0.35;
/// Half-height, metres — a 1.8 m body. Feet are `position.y - HALF_H`.
pub const HALF_H: f64 = 0.90;
/// Eye above the body centre, metres: 1.62 m off the floor.
pub const EYE_LIFT: f64 = 0.72;
/// Where a session opens and `restart` returns to — standing on the floor by
/// the south wall, facing the stairs. The body *centre*, so it is one half-height
/// up and the opening tick is a rest, not a fall.
pub const START: sim::DVec3 = sim::DVec3::new(0.0, HALF_H, 8.0);
/// Below this the body has left the world and is put back. The room is closed,
/// so this only fires if a slab moved — a safety net, and a visible one.
pub const KILL_Y: f64 = -20.0;

// ---------------------------------------------------------------- gun -------

/// Ticks between shots — 6.7 a second at 60 Hz. Above [`RECOIL_TICKS`] on
/// purpose: a weapon whose kick has not settled when the next round leaves is a
/// weapon that climbs, and a climb is a second feel decision hiding inside the
/// fire rate.
pub const FIRE_TICKS: u32 = 9;
/// How far a bullet reaches, metres. The room's longest diagonal is 34.2, so
/// this is "everywhere in here" with room to spare — a shot that expired in
/// flight would be a range mechanic nobody asked for.
pub const SHOT_RANGE: f64 = 40.0;
/// Steps the spread draw has. **Odd**, so dead centre is one of the outcomes
/// rather than the gap between two of them.
pub const SPREAD_STEPS: u32 = 9;
/// Radians one spread step is worth. Four steps either way is ±0.0072 rad —
/// 25 mm at ten metres, enough that a burst is not a laser and far under what a
/// standing shot needs to hit a 440 mm target.
pub const SPREAD_UNIT: f32 = 0.0018;
/// Radians the view kicks up per shot, and what it gives back per tick.
///
/// Dyadic on purpose: 1/64 given back as 8 × 1/512 is exact in `f32`, so a
/// settled weapon leaves the pitch bit-identical to where the kick found it.
/// A tenth of a degree of residue per shot would be invisible, deterministic,
/// and still a number that grew all session.
pub const RECOIL_KICK: f32 = 0.015_625;
/// See [`RECOIL_KICK`].
pub const RECOIL_TICKS: u32 = 8;
/// See [`RECOIL_KICK`].
pub const RECOIL_GIVE: f32 = 0.001_953_125;
/// Where the tracer is *drawn* from, relative to the eye: right, up, forward.
/// The ray still leaves the eye — item 5's honest minimum is no view model at
/// all, and a tracer that came out of the crosshair would read as one anyway.
pub const MUZZLE: (f64, f64, f64) = (0.22, -0.16, 0.35);
/// What `bootstrap` seeds the spread and the target order with. A constant, not
/// a clock: a wall-clock seed is a game that cannot be replayed (§4.7).
pub const SEED: u64 = 0x5348_4f4f_5445_5201;

// ---------------------------------------------------------------- range -----

/// [`Range::state`]: the round is live. **Zero**, so a zeroed migration is a
/// playable round rather than a finished one — [`Renderable::smoothness`]'s
/// rule, applied to a game's own state.
pub const STATE_RUNNING: u32 = 0;
/// [`Range::state`]: out of misses. The score is frozen and `R` starts again.
pub const STATE_OVER: u32 = 1;
/// Targets standing at once. Three: enough that there is always one to swing
/// to, few enough that the room does not read as a shooting gallery.
pub const TARGETS_LIVE: usize = 3;
/// Ticks a target stands before it leaves — 3.5 s, about two swings and a shot.
pub const TARGET_LIFE: u32 = 210;
/// Targets allowed to leave before the round ends.
pub const MISSES_ALLOWED: u32 = 3;
/// Half-extent of a target, metres — a 440 mm cube.
pub const TARGET_HALF: f32 = 0.22;
/// What one target is worth before the streak.
pub const TARGET_WORTH: u32 = 100;
/// How a target takes the light — matte enough to read as a board rather than a
/// mirror. Named for [`shelter_sky`]'s reason: `deal` and the golden that
/// pictures the course read one number.
pub const TARGET_SMOOTHNESS: f32 = 0.65;
/// Added per consecutive hit, up to [`STREAK_CAP`] of them. A miss — a shot
/// into the room, or a target that walked — puts it back to zero.
pub const STREAK_STEP: u32 = 25;
/// See [`STREAK_STEP`]. Eight, so a perfect run tops out at triple.
pub const STREAK_CAP: u32 = 8;
/// A target's colour, and what a hit and an escape leave behind.
pub const TARGET_INK: u32 = 0x00e0_5a3c;
/// See [`TARGET_INK`].
pub const HIT_INK: u32 = 0x00ff_d070;
/// See [`TARGET_INK`].
pub const DUST_INK: u32 = 0x00c8_c0b4;
/// See [`TARGET_INK`].
pub const ESCAPE_INK: u32 = 0x0060_78a0;

/// Where a target can stand: the room's clear air, twelve places.
///
/// A table rather than a random point in the volume, and the reason is the same
/// one [`ROOM`] is a table for: a spawn point has to be *shootable* — clear of
/// the solids, off the pillars, out of the chart's way and not inside the
/// shelter's roof — and a uniform draw over a box satisfies none of that. The
/// test that keeps it true asks the room rather than this list (`tests/game.rs`
/// checks every spot against every [`ROOM`] entry), which is demo 11's pull-2
/// doctrine: a claim derives from the scene, or it rots.
pub const SPOTS: &[sim::DVec3] = &[
    sim::DVec3::new(-3.0, 2.6, -4.0),
    sim::DVec3::new(3.5, 2.6, -8.0),
    sim::DVec3::new(6.0, 1.2, -10.0),
    sim::DVec3::new(-6.0, 2.8, -9.0),
    sim::DVec3::new(10.0, 1.6, -2.0),
    sim::DVec3::new(6.5, 2.2, 2.5),
    sim::DVec3::new(-9.0, 1.4, 7.5),
    sim::DVec3::new(-1.5, 2.4, 6.5),
    sim::DVec3::new(3.6, 1.0, 8.5),
    sim::DVec3::new(-6.5, 3.0, 0.5),
    sim::DVec3::new(0.0, 3.2, -6.0),
    sim::DVec3::new(11.0, 3.0, 5.0),
];

// ---------------------------------------------------------------- effects ---

/// [`Spark::kind`]: the bullet's streak. Drawn once, goes nowhere, gone.
pub const SPARK_TRACER: u32 = 0;
/// [`Spark::kind`]: a chip off whatever was hit. The only kind that moves.
pub const SPARK_DEBRIS: u32 = 1;
/// [`Spark::kind`]: the muzzle flash — a [`Light`] and no geometry at all.
pub const SPARK_FLASH: u32 = 2;
/// Ticks the tracer is up. Two would read as a flicker on a 60 Hz display and
/// vanish entirely on a frame the sim ticked twice.
pub const TRACER_TICKS: u32 = 3;
/// Half-thickness of the tracer box, metres. A beam is a long thin box —
/// [`Renderable`]'s own documentation says so, and this is the game that needed
/// it.
pub const TRACER_HALF: f32 = 0.012;
/// See [`TRACER_HALF`].
pub const TRACER_INK: u32 = 0x00ff_e8b0;
/// Chips one impact throws.
pub const SPARKS: u32 = 5;
/// Ticks one lives, and the pull on it per tick.
pub const SPARK_TICKS: u32 = 22;
/// See [`SPARK_TICKS`].
pub const SPARK_GRAVITY: f64 = 0.0035;
/// Metres per tick a chip leaves at, along the face's normal.
pub const SPARK_LIFT: f64 = 0.030;
/// And the scatter across it, per axis: `±(SPREAD_STEPS-1)/2` steps of this.
pub const SPARK_SCATTER: f64 = 0.008;
/// Half-extent of a chip, metres.
pub const SPARK_HALF: f32 = 0.020;
/// Ticks the muzzle flash lights the room for. Two: one is a frame a 30 Hz
/// display can miss entirely, and three starts to read as a lamp.
pub const FLASH_TICKS: u32 = 2;
/// The flash's colour, strength and reach — a hot, short, local light.
pub const FLASH_INK: u32 = 0x00ff_e0a0;
/// See [`FLASH_INK`].
pub const FLASH_INTENSITY: f32 = 9.0;
/// See [`FLASH_INK`].
pub const FLASH_RANGE: f32 = 4.5;

// ---------------------------------------------------------------- chart -----

/// Steps along each axis of [`chart`] — five, so the ends are 0 and 1 exactly,
/// the middle step lands on 0.5, and the grid is the twenty-five balls every
/// published material chart is (§6 M26).
pub const CHART_STEPS: usize = 5;
/// Balls the chart deals — what `bootstrap` adds to [`ROOM`].
pub const CHART_BALLS: usize = CHART_STEPS * CHART_STEPS;
/// Radius of one ball.
pub const CHART_RADIUS: f32 = 0.28;
/// Between neighbours, along both axes. Wider than a ball, so what separates
/// two samples is background and not a tangent.
pub const CHART_PITCH: f64 = 0.72;
/// Centre of the grid — the middle ball, smoothness and metallic both 0.5.
///
/// **Centred in the room's width, off the floor, and clear of everything that
/// casts (§6 M26).** Three separate requirements, and the room only satisfies
/// all three here:
///
/// - *Nothing shadows it.* The sun travels `-x, -y, -z`, so a caster has to
///   stand east, above and south of the chart. The nearest that qualifies is
///   the pillar at `(2, -6)`, whose shadow crosses `y = 2.1` at `z = -6.6` —
///   four metres short. The old placement failed exactly this: on the floor at
///   the east end, the pillar at `(9, -6)` laid a shadow across its bright half,
///   and a chart with a shadow on it is measuring the shadow.
/// - *Nothing stands behind it.* The room's middle is not its clear span: the
///   mezzanine reaches `x = -2.5` and a pillar stands at `x = 2`, which leaves
///   4.5 m for a 3.4 m chart and puts a column of concrete behind the smooth
///   end of it. Against the north wall the backdrop is one flat, evenly lit
///   plane with sky above — which is what a chart is read against.
/// - *Nothing walks through it.* It hangs at head height in the one part of the
///   room the course does not use.
pub const CHART_CENTRE: sim::DVec3 = sim::DVec3::new(0.0, 2.1, -10.5);
/// One base colour for every ball in it, and that is the point: the only things
/// that differ across the chart are the two knobs, so anything else the eye
/// sees is the lighting answering them.
pub const CHART_INK: u32 = 0x00b4_b4b4;

/// The material chart (§6 M24, spheres since §6 M26): `(centre, smoothness,
/// metallic)` for every ball in it, dielectric row first, roughest step first.
///
/// A grid standing up in the air rather than lying on the floor, because a
/// chart is read against a background and a floor is a background that changes
/// with distance — every sample at a different range, under a different slice
/// of the same shadow cascade. Smoothness runs left to right, roughest first;
/// metallic runs upward, dielectrics along the bottom where reading starts.
///
/// A function rather than a table because it is arithmetic — twenty-five
/// hand-written triples are twenty-five chances to mistype a step, and the
/// thing being demonstrated is precisely that the steps are even.
///
/// The metallic axis passes through values no real material has. That is
/// deliberate and [`Renderable::metallic`](gg_ecs::boundary::Renderable::metallic)
/// says so: in between is not a substance, it is the blend a texture authored
/// across a boundary needs, and a chart that skipped it would not show what the
/// knob does between its ends.
pub fn chart() -> impl Iterator<Item = (sim::DVec3, f32, f32)> {
    let span = (CHART_STEPS - 1) as f64 * CHART_PITCH;
    (0..CHART_STEPS).flat_map(move |row| {
        (0..CHART_STEPS).map(move |step| {
            let at = sim::DVec3::new(
                CHART_CENTRE.x - span / 2.0 + step as f64 * CHART_PITCH,
                CHART_CENTRE.y - span / 2.0 + row as f64 * CHART_PITCH,
                CHART_CENTRE.z,
            );
            // Perceptual and even: `Renderable::smoothness` is squared into
            // GGX's alpha by the shader, so an even step here is what looks
            // like an even step.
            let axis = |i: usize| i as f32 / (CHART_STEPS - 1) as f32;
            (at, axis(step), axis(row))
        })
    })
}

/// Two lamps, low and warm, so the shapes have a second edge to read by — one
/// sun leaves everything facing away from it a single flat tone.
///
/// Constants rather than literals inside `bootstrap` because the `chart` golden
/// scene deals this same room (§4.10: the reference guards the demo, not a
/// lookalike), and a lamp it placed itself would be a second source of truth.
pub const LAMPS: [sim::DVec3; 2] = [
    sim::DVec3::new(-6.0, 3.2, -4.0),
    sim::DVec3::new(6.0, 3.2, 4.0),
];
/// See [`LAMPS`].
pub const LAMP_INK: u32 = 0x00ff_c890;
/// See [`LAMPS`].
pub const LAMP_INTENSITY: f32 = 14.0;
/// See [`LAMPS`].
pub const LAMP_RANGE: f32 = 11.0;
/// The sun's tint and strength — [`LAMPS`]' reason for being named.
pub const SUN_INK: u32 = 0x00ff_f4e0;
/// See [`SUN_INK`].
pub const SUN_INTENSITY: f32 = 3.4;

/// How bright the environment is against the sun above it (§6 M24).
///
/// The sky's diffuse contribution is roughly `intensity * 0.3 * albedo` against
/// the sun's `3.4 / PI * albedo` — so 0.8 is a fill about a fifth of the key,
/// which is what an overcast-free afternoon actually measures. Turning it up
/// does not make the room brighter so much as make it *flatter*, which is the
/// thing a fill light is for and the thing too much of it costs.
pub const SKY_INTENSITY: f32 = 0.8;

// ---------------------------------------------------------------- verbs -----

/// Jump — buffered [`BUFFER_TICKS`], honoured within [`COYOTE_TICKS`].
pub const JUMP: ActionId = ActionId::new(0);
/// Back to [`START`], looking where a session opens.
pub const RESTART: ActionId = ActionId::new(1);
/// Open the menu, and step back out of it. Bound to Escape, which is what takes
/// Escape off the shell — see this crate's `input.toml`.
pub const PAUSE: ActionId = ActionId::new(2);
/// Fire. Id 5 because it was **appended** to the verb list: the order of that
/// list is the id space a replay records (§4.7), so a verb inserted among the
/// existing ones would renumber every stream ever made against them. Bound to
/// the same physical button as `ui_click`, which is not a conflict but the
/// arbitration itself — the menu is up or it is not, and [`shoot`] and the host
/// read the same fact off [`Session::paused`].
pub const FIRE: ActionId = ActionId::new(5);
/// Strafe.
pub const MOVE_RIGHT: AxisId = AxisId::new(0);
/// Forward and back.
pub const MOVE_FORWARD: AxisId = AxisId::new(1);
/// Yaw, from pointer motion.
pub const AIM_X: AxisId = AxisId::new(2);
/// Pitch, from pointer motion.
pub const AIM_Y: AxisId = AxisId::new(3);

// ---------------------------------------------------------------- cues ------

/// [`Cue::kind`]: the jump.
pub const CUE_JUMP: u32 = 0;
/// The landing.
pub const CUE_LAND: u32 = 1;
/// The shot — the first [`wave::NOISE`] any game in this tree has fired.
pub const CUE_SHOT: u32 = 2;
/// A target taken.
pub const CUE_HIT: u32 = 3;
/// A target that left on its own.
pub const CUE_ESCAPE: u32 = 4;
/// The round's end.
pub const CUE_OVER: u32 = 5;
/// How many cue entities `bootstrap` deals.
pub const CUES: usize = 6;

// ---------------------------------------------------------------- hud -------

/// [`Hud::line`]: ground speed.
pub const HUD_SPEED: u32 = 0;
/// Grounded or airborne — the one piece of controller state a tuner watches.
pub const HUD_STATE: u32 = 1;
/// The round's score and the streak riding on it.
pub const HUD_SCORE: u32 = 2;
/// The best this session has managed.
pub const HUD_BEST: u32 = 3;
/// Misses spent, or how the round ended.
pub const HUD_MISS: u32 = 4;
/// Every text row `bootstrap` deals, and what it reads before a tick has had
/// anything to say. One table so the rows and their count cannot drift — the
/// crosshair arms are the other kind of `Hud` and are counted by [`ARMS`].
pub const HUD_ROWS: [(u32, &str); 5] = [
    (HUD_SPEED, "SPEED 0.0"),
    (HUD_STATE, "GROUND"),
    (HUD_SCORE, "SCORE 0"),
    (HUD_BEST, "BEST 0"),
    (HUD_MISS, "MISS 0/3"),
];
/// The first crosshair arm's line. Above the text rows and never equal to one,
/// so `present`'s match reaches the arms only through its default arm.
pub const HUD_CROSS: u32 = 8;
/// Canvas centre, in [`gg_ecs::boundary::CANVAS`] units.
pub const CENTRE: (f32, f32) = (320.0, 180.0);
/// Crosshair arm length and thickness, and the gap left at the middle.
pub const CROSS: (f32, f32, f32) = (6.0, 2.0, 3.0);
/// The crosshair's colour, `0xAARRGGBB` — not quite opaque, so it sits on the
/// scene rather than in front of it.
pub const CROSS_INK: u32 = 0xe0ff_ffff;
/// What it turns for [`HITMARK_TICKS`] after a target is taken. The one piece
/// of feedback that reaches the eye already looking at where the shot went.
pub const HITMARK_INK: u32 = 0xffff_9040;
/// See [`HITMARK_INK`]. Six ticks counting the shot's own — 100 ms, the same
/// window [`COYOTE_TICKS`] uses and for the same reason: it is what reads as
/// *immediate* rather than as a flicker. [`targets`] ages it, so a hit scored
/// this tick leaves five behind it.
pub const HITMARK_TICKS: u32 = 6;
/// How many crosshair rects `bootstrap` deals.
pub const ARMS: usize = 4;

// ---------------------------------------------------------------- menu ------

/// [`Session::page`]: the pause menu's first page.
pub const PAGE_MAIN: u32 = 0;
/// [`Session::page`]: the settings panel.
pub const PAGE_SETTINGS: u32 = 1;

/// [`Menu::item`], in the order `bootstrap` deals them. The scrim is first so
/// it is behind everything else at equal order, and the value doubles as the
/// draw order — a menu that overlaps the HUD must be on top of it, and
/// [`HUD_CROSS`] is 8.
pub const MENU_SCRIM: u32 = 16;
/// The panel behind the rows.
pub const MENU_PANEL: u32 = 17;
/// Which page you are on.
pub const MENU_TITLE: u32 = 18;
/// Back to the game.
pub const MENU_RESUME: u32 = 19;
/// To [`PAGE_SETTINGS`].
pub const MENU_SETTINGS: u32 = 20;
/// Back to [`START`], without leaving the menu's own state behind.
pub const MENU_RESTART: u32 = 21;
/// End the session — [`Prefs::close`], the only way a game has to do it.
pub const MENU_QUIT: u32 = 22;
/// Back to [`PAGE_MAIN`].
pub const MENU_BACK: u32 = 23;
/// The look row's readout.
pub const MENU_LOOK: u32 = 24;
/// Slower.
pub const MENU_LOOK_DOWN: u32 = 25;
/// Faster.
pub const MENU_LOOK_UP: u32 = 26;
/// The volume row's readout.
pub const MENU_VOLUME: u32 = 27;
/// Quieter.
pub const MENU_VOLUME_DOWN: u32 = 28;
/// Louder.
pub const MENU_VOLUME_UP: u32 = 29;
/// The edge treatment, cycled by clicking it.
pub const MENU_AA: u32 = 30;
/// Every menu widget, in deal order.
pub const MENU_ITEMS: [u32; 15] = [
    MENU_SCRIM,
    MENU_PANEL,
    MENU_TITLE,
    MENU_RESUME,
    MENU_SETTINGS,
    MENU_RESTART,
    MENU_QUIT,
    MENU_BACK,
    MENU_LOOK,
    MENU_LOOK_DOWN,
    MENU_LOOK_UP,
    MENU_VOLUME,
    MENU_VOLUME_DOWN,
    MENU_VOLUME_UP,
    MENU_AA,
];

/// The menu's rectangles in [`gg_ecs::boundary::CANVAS`] units, `[x, y, w, h]`.
/// One table so the panel and the rows cannot drift apart, and so a row that
/// moves moves in one place.
const fn slot(item: u32) -> [f32; 4] {
    // Panel x 200..440, y 60..300. Rows are 208 wide inside an 8-unit margin.
    match item {
        MENU_SCRIM => [0.0, 0.0, 640.0, 360.0],
        MENU_PANEL => [200.0, 60.0, 240.0, 240.0],
        MENU_TITLE => [216.0, 76.0, 208.0, 12.0],
        MENU_RESUME => [216.0, 110.0, 208.0, 26.0],
        MENU_SETTINGS => [216.0, 146.0, 208.0, 26.0],
        MENU_RESTART => [216.0, 182.0, 208.0, 26.0],
        MENU_QUIT => [216.0, 218.0, 208.0, 26.0],
        MENU_LOOK => [216.0, 116.0, 150.0, 12.0],
        MENU_LOOK_DOWN => [370.0, 110.0, 24.0, 24.0],
        MENU_LOOK_UP => [400.0, 110.0, 24.0, 24.0],
        MENU_VOLUME => [216.0, 156.0, 150.0, 12.0],
        MENU_VOLUME_DOWN => [370.0, 150.0, 24.0, 24.0],
        MENU_VOLUME_UP => [400.0, 150.0, 24.0, 24.0],
        MENU_AA => [216.0, 188.0, 208.0, 26.0],
        MENU_BACK => [216.0, 248.0, 208.0, 26.0],
        // Not a panic: an id no arm knows draws nowhere, which is the same
        // answer a hidden widget gets and the safe one for a table edit.
        _ => HIDDEN,
    }
}

/// A zero rect — how a widget hides (§4.9), and therefore also how the menu
/// gives the mouse back.
const HIDDEN: [f32; 4] = [0.0, 0.0, 0.0, 0.0];

/// The scrim over the scene: black at two-thirds, so the room reads through it
/// and the rows do not have to compete with it.
pub const MENU_SCRIM_INK: u32 = 0xa800_0000;
/// The panel behind the rows.
pub const MENU_PANEL_INK: u32 = 0xf01c_2230;
/// A button's fill. The host lightens it on hover — the one visual decision it
/// makes, and the reason a game authors one colour and not three.
pub const MENU_BUTTON_INK: u32 = 0xff2e_3a4e;
/// Text on a button, and the title.
pub const MENU_TEXT_INK: u32 = 0xffe8_f0ff;

// ---------------------------------------------------------------- level -----

/// The sun's travel direction. Not vertical: the one angle at which a shadow
/// map proves the least.
pub const SUN: sim::Vec3 = sim::Vec3::new(-0.45, -1.0, -0.30);

/// The room: centre, half-extent, colour. Every entry is spawned `Solid`, so
/// what you see is what you stand on — the same mechanism that makes demo 11's
/// authored slabs solid without the editor learning anything.
///
/// The shapes are chosen to ask the controller different questions: a stair
/// flight it must *step* up, crates below and above [`STEP_HEIGHT`] so both
/// answers are visible side by side, floating slabs at 0.9 m rises that only a
/// jump reaches, and pillars to walk into and slide along.
pub const ROOM: &[(sim::DVec3, sim::Vec3, u32)] = &[
    // Floor, top at y = 0. Thick enough that MAX_FALL cannot cross it.
    (
        sim::DVec3::new(0.0, -0.25, 0.0),
        sim::Vec3::new(12.5, 0.25, 12.5),
        0x00b4_aea2,
    ),
    // Four walls, 4 m tall, inner faces at ±12.
    (
        sim::DVec3::new(0.0, 2.0, -12.25),
        sim::Vec3::new(12.5, 2.0, 0.25),
        0x008f_9298,
    ),
    (
        sim::DVec3::new(0.0, 2.0, 12.25),
        sim::Vec3::new(12.5, 2.0, 0.25),
        0x008f_9298,
    ),
    (
        sim::DVec3::new(-12.25, 2.0, 0.0),
        sim::Vec3::new(0.25, 2.0, 12.5),
        0x008f_9298,
    ),
    (
        sim::DVec3::new(12.25, 2.0, 0.0),
        sim::Vec3::new(0.25, 2.0, 12.5),
        0x008f_9298,
    ),
    // Six stairs, 0.30 rise each — under STEP_HEIGHT, so they are walked.
    // Each is a column from the floor rather than a tread, so nothing here
    // has an underside to get stuck beneath.
    (
        sim::DVec3::new(-7.0, 0.15, 3.0),
        sim::Vec3::new(2.5, 0.15, 0.45),
        0x00a0_8c74,
    ),
    (
        sim::DVec3::new(-7.0, 0.30, 2.1),
        sim::Vec3::new(2.5, 0.30, 0.45),
        0x00a0_8c74,
    ),
    (
        sim::DVec3::new(-7.0, 0.45, 1.2),
        sim::Vec3::new(2.5, 0.45, 0.45),
        0x00a0_8c74,
    ),
    (
        sim::DVec3::new(-7.0, 0.60, 0.3),
        sim::Vec3::new(2.5, 0.60, 0.45),
        0x00a0_8c74,
    ),
    (
        sim::DVec3::new(-7.0, 0.75, -0.6),
        sim::Vec3::new(2.5, 0.75, 0.45),
        0x00a0_8c74,
    ),
    (
        sim::DVec3::new(-7.0, 0.90, -1.5),
        sim::Vec3::new(2.5, 0.90, 0.45),
        0x00a0_8c74,
    ),
    // The mezzanine the stairs arrive on, top at 1.8. Overlaps the last step
    // by 50 mm rather than meeting it: two solids that merely touch leave a
    // seam a foot can find, and overlapping ones resolve identically.
    (
        sim::DVec3::new(-7.25, 0.90, -6.95),
        sim::Vec3::new(4.75, 0.90, 5.05),
        0x0096_846c,
    ),
    // Four pillars — something to walk into, and something for the sun to
    // put a shadow behind.
    (
        sim::DVec3::new(2.0, 2.0, 6.0),
        sim::Vec3::new(0.4, 2.0, 0.4),
        0x00c8_c4bc,
    ),
    (
        sim::DVec3::new(2.0, 2.0, -6.0),
        sim::Vec3::new(0.4, 2.0, 0.4),
        0x00c8_c4bc,
    ),
    (
        sim::DVec3::new(9.0, 2.0, 6.0),
        sim::Vec3::new(0.4, 2.0, 0.4),
        0x00c8_c4bc,
    ),
    (
        sim::DVec3::new(9.0, 2.0, -6.0),
        sim::Vec3::new(0.4, 2.0, 0.4),
        0x00c8_c4bc,
    ),
    // Three floating slabs, each 0.9 m above the last: inside a 1.2 m apex
    // with margin, outside STEP_HEIGHT by a factor of two. 0.4 m thick, the
    // thinnest thing here and still thicker than MAX_FALL.
    (
        sim::DVec3::new(4.0, 1.0, 0.0),
        sim::Vec3::new(1.2, 0.2, 1.2),
        0x00d0_8840,
    ),
    (
        sim::DVec3::new(7.5, 1.9, -3.0),
        sim::Vec3::new(1.2, 0.2, 1.2),
        0x00d0_8840,
    ),
    (
        sim::DVec3::new(10.5, 2.8, -6.5),
        sim::Vec3::new(1.2, 0.2, 1.2),
        0x00d0_8840,
    ),
    // Kerb-height crates: 0.30 tall, stepped onto without a jump.
    (
        sim::DVec3::new(0.0, 0.15, 4.5),
        sim::Vec3::new(0.6, 0.15, 0.6),
        0x0078_b4a8,
    ),
    // The second sits against the stack below, which is what makes that stack
    // climbable: 1.2 m of apex off a 0.3 m kerb clears a 1.4 m top by 100 mm.
    (
        sim::DVec3::new(-2.9, 0.15, 5.2),
        sim::Vec3::new(0.6, 0.15, 0.6),
        0x0078_b4a8,
    ),
    // Waist-height crates: 0.70 tall, and therefore jumped. The first sits
    // straight ahead of START behind the kerbs, so the opening walk meets both
    // heights in order and the difference teaches itself.
    (
        sim::DVec3::new(0.0, 0.35, 1.5),
        sim::Vec3::new(0.5, 0.35, 0.5),
        0x004f_9a8c,
    ),
    // And two stacked, top at 1.4 m — above a standing jump's 1.2 m apex, so
    // the only way up is off the kerb against them. The room's one puzzle.
    (
        sim::DVec3::new(-4.2, 0.35, 5.2),
        sim::Vec3::new(0.5, 0.35, 0.5),
        0x004f_9a8c,
    ),
    (
        sim::DVec3::new(-4.2, 1.05, 5.2),
        sim::Vec3::new(0.5, 0.35, 0.5),
        0x004f_9a8c,
    ),
    // The shelter's roof and the one wall that closes it — the room's other two
    // sides are already walls. Both derived from [`SHELTER`] rather than placed
    // beside it, so the space that is *covered* and the space that is *lit*
    // differently cannot drift apart (§6 M28).
    (
        sim::DVec3::new(
            SHELTER.0.x,
            SHELTER.0.y + SHELTER.1.y as f64 + ROOF.y as f64,
            SHELTER.0.z,
        ),
        sim::Vec3::new(SHELTER.1.x + 0.25, ROOF.y, SHELTER.1.z + 0.25),
        0x008f_9298,
    ),
    (
        sim::DVec3::new(
            SHELTER.0.x - SHELTER.1.x as f64 - 0.25,
            SHELTER.0.y,
            SHELTER.0.z,
        ),
        sim::Vec3::new(0.25, SHELTER.1.y, SHELTER.1.z + 0.25),
        0x008f_9298,
    ),
];

/// The sheltered corner: the centre and half-extent of the space *under* the
/// roof, in the far +x/+z corner where nothing else stands.
///
/// One constant because it is two facts that must agree — where the roof is, and
/// where the light stops being the sky's. Walking in is the whole demonstration
/// of §6 M28: the opening faces -z, the fade is a metre and a bit, and the
/// environment changes over the stride that crosses it rather than on one tick.
pub const SHELTER: (sim::DVec3, sim::Vec3) = (
    sim::DVec3::new(8.25, 1.05, 9.75),
    sim::Vec3::new(3.75, 1.05, 2.25),
);

/// Half-thickness of the shelter's roof slab. Thick enough that [`MAX_FALL`]
/// cannot cross it, as the floor is.
const ROOF: sim::Vec3 = sim::Vec3::new(0.0, 0.25, 0.0);

/// Metres over which the shelter's environment gives way to the sky's. A stride
/// and a half at walking speed — long enough to read as a change in the light
/// rather than as a switch, short enough that it happens in the doorway.
pub const SHELTER_FADE: f32 = 1.4;

/// What the shelter radiates: a dim, cool bounce with no sky in it. Not a
/// darkened copy of [`Sky::daylight`] — under a roof the bright half of the
/// world is gone and what is left is floor and wall, so the gradient is
/// upside-down from the outdoor one and much flatter.
pub const SHELTER_INTENSITY: f32 = 0.30;

/// The shelter's environment as one value.
///
/// A function rather than three constants because `bootstrap` and the golden
/// that pictures this room both need the whole thing, and a colour copied into
/// the harness is the second table §4.10 forbids.
#[must_use]
pub fn shelter_sky() -> Sky {
    Sky {
        zenith: 0x0056_5a60,
        horizon: 0x006a_6862,
        ground: 0x0048_423a,
        ..Sky::daylight(SHELTER_INTENSITY)
    }
    .within(SHELTER.0, SHELTER.1, SHELTER_FADE)
}

// ---------------------------------------------------------------- state -----

/// The body: pose, motion, aim, and the three feel counters. All sim state, all
/// hashed — a divergent coyote window is a named tick rather than a feel
/// opinion.
///
/// There is deliberately no [`Renderable`] on this entity. The camera sits
/// inside the body, and a box drawn there fills the screen.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "shooter.walker")]
#[repr(C)]
pub struct Walker {
    /// Centre of the body box, metres. `f64`, never narrowed sim-side (§4.2.1).
    pub position: sim::DVec3,
    /// Metres per tick.
    pub velocity: sim::DVec3,
    /// Rotation about +Y, radians, wrapped to ±π.
    pub yaw: f32,
    /// Rotation about the body's right axis, radians, clamped to
    /// [`PITCH_LIMIT`].
    pub pitch: f32,
    /// Ticks of ledge forgiveness left. Refilled while supported.
    pub coyote: u32,
    /// Ticks the last jump press stays willing.
    pub buffer: u32,
    /// Standing on something this tick. `u32` for `Pod`; 0 or 1.
    pub grounded: u32,
    /// Padding, named (§4.2.1 hazard 4).
    pub _pad: u32,
}

/// The weapon, on the body's own entity — a gun is the body's, and a respawn
/// that left the last round's cooldown behind would be a bug shaped like a
/// feature.
///
/// The stream is *in here* rather than beside it (§6 M18's rule): an `Rng` that
/// lives outside the world is one the replay gate silently stops covering.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "shooter.gun")]
#[repr(C)]
pub struct Gun {
    /// The spread stream. First, because it is the `u64` that sets the
    /// alignment and a trailing one would leave a hole `Pod` refuses.
    pub rng: sim::Rng,
    /// Ticks until the trigger answers again.
    pub cooldown: u32,
    /// Ticks of kick still owed back to the view.
    pub recover: u32,
    /// Whether the trigger has been released since the world last ran. A click
    /// that closes the menu is not a shot: the button that resumes is the
    /// button that fires, and without this the first thing a resumed session
    /// does is put a round through whatever the RESUME row was over. Zero is
    /// *disarmed*, which is the safe migration — one released tick arms it.
    pub armed: u32,
    /// Padding, named (§4.2.1 hazard 4).
    pub _pad: u32,
}

/// The round: what has been scored, what is left to lose it with, and the
/// stream that places the targets.
///
/// Separate from [`Gun`] rather than sharing one stream, because the two draw
/// for unrelated reasons and a single stream would make the *number of shots
/// fired* decide where the next target stands.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "shooter.range")]
#[repr(C)]
pub struct Range {
    /// Where the targets stand. First, as [`Gun::rng`] is.
    pub rng: sim::Rng,
    /// This round's points.
    pub score: u32,
    /// The best this session has seen. Survives a restart on purpose — it is
    /// the one number a session accumulates, and M14's save is what would carry
    /// it across sessions the day this game gets one.
    pub best: u32,
    /// Consecutive hits, capped at [`STREAK_CAP`] where it pays.
    pub streak: u32,
    /// Targets that left on their own. [`MISSES_ALLOWED`] ends the round.
    pub misses: u32,
    /// Targets taken, all round.
    pub hits: u32,
    /// Shots fired, all round — the accuracy denominator, and the one number
    /// that says whether the spread is doing anything.
    pub shots: u32,
    /// [`STATE_RUNNING`] or [`STATE_OVER`].
    pub state: u32,
    /// Ticks the crosshair stays lit after a hit.
    pub hitmark: u32,
}

/// A thing to shoot. The geometry is the [`Renderable`] beside it, [`Solid`]'s
/// rule again: what you see is what you hit.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "shooter.target")]
#[repr(C)]
pub struct Target {
    /// Ticks it has stood.
    pub age: u32,
    /// Ticks it stands for.
    pub life: u32,
    /// Which [`SPOTS`] entry it occupies — so two targets cannot share one.
    pub slot: u32,
    /// Points it pays before the streak.
    pub worth: u32,
}

/// Everything a shot leaves behind: the tracer, the chips, the flash. One
/// component for three kinds because what they share is the *life counter*, and
/// three components would be three copies of the ageing system (§6 M37 item 3 —
/// this is the churn, and it is one archetype and one pass).
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "shooter.spark")]
#[repr(C)]
pub struct Spark {
    /// Metres per tick. Zero for everything but [`SPARK_DEBRIS`].
    pub velocity: sim::DVec3,
    /// Ticks left. Zero is dead and is despawned the tick it arrives.
    pub life: u32,
    /// One of the `SPARK_*` constants.
    pub kind: u32,
}

/// Marks a [`Renderable`] the body collides with. The geometry lives in the
/// `Renderable` itself — what you see is what you stand on.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "shooter.solid")]
#[repr(C)]
pub struct Solid {
    /// Padding, named — the marker is the component.
    pub _pad: u32,
}

/// Names the [`Sound`] beside it, so a system can bump one cue without holding
/// entity ids.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "shooter.cue")]
#[repr(C)]
pub struct Cue {
    /// One of the `CUE_*` constants.
    pub kind: u32,
}

/// Names the [`Widget`] beside it, for the same reason as [`Cue`].
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "shooter.hud")]
#[repr(C)]
pub struct Hud {
    /// One of the `HUD_*` constants; crosshair arms carry [`ARMS`] and up.
    pub line: u32,
}

/// Names a menu [`Widget`], the same idiom again.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "shooter.menu")]
#[repr(C)]
pub struct Menu {
    /// One of the `MENU_*` constants.
    pub item: u32,
}

/// The session: whether the sim is running, what the menu is showing, and the
/// one setting that never leaves this crate.
///
/// Hashed like everything else, which is what makes a paused session a
/// replayable fact rather than a host mode — and what makes a sensitivity
/// change something a recorded session reproduces without anything on disk.
/// Nothing here is persisted, and a run therefore opens at [`SENS_DEFAULT`]
/// however the last one ended. Reading it off disk at boot *unconditionally*
/// would be the CVar-in-a-replay hazard (§8) with the file as the leak — a
/// stream blessed on one desk would diverge on another the moment its owner
/// touched the menu. The mechanism that would be sound is the one
/// `scene.ggsave` already uses: a probe live sessions make and recorded or
/// replayed ones do not (§6 M15.2). What it needs that does not exist yet is a
/// save narrowed to the components a *setting* lives in — `World::load` takes
/// a whole world, and a settings file that also restored where the player was
/// standing would be a different feature wearing this one's name.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "shooter.session")]
#[repr(C)]
pub struct Session {
    /// The sim is stopped and the menu is up. `u32` for `Pod`; 0 or 1.
    pub paused: u32,
    /// [`PAGE_MAIN`] or [`PAGE_SETTINGS`]. Kept while playing, so re-opening
    /// the menu comes back to the front page.
    pub page: u32,
    /// Look sensitivity in [`SENS_ONE`]ths, clamped to
    /// [`SENS_MIN`]..=[`SENS_MAX`].
    pub sens: u32,
    /// Padding, named (§4.2.1 hazard 4).
    pub _pad: u32,
}

// ---------------------------------------------------------------- systems ---

/// Put the body back at [`START`] facing where a session opens. Not a wipe: the
/// room is `bootstrap`'s and nothing here could deal it back.
pub fn restart(world: &mut GameWorld) {
    if !world.just_pressed(RESTART) {
        return;
    }
    respawn(world);
}

/// Deal the body, the room, the light and the HUD if there is no body yet.
/// Idempotent by asking the world rather than by remembering (§4.2.2) — so a
/// reload runs it again and it is a no-op.
pub fn bootstrap(world: &mut GameWorld) {
    let mut exists = false;
    world.visit::<&Walker>(|_, _| exists = true);
    if exists {
        return;
    }

    let body = world.spawn();
    if let Err(refused) = world.insert(body, at_start()) {
        world.log(log_level::ERROR, &refused.to_string());
        return;
    }
    world.put(body, Eye::at(eye_of(&at_start()), 0.0, 0.0));
    world.put(body, fresh_gun());
    // The round on its own entity, for `Session`'s reason: `restart` rewrites
    // the body, and a score the body carried would be a score a kill plane
    // could clear.
    world.spawn_with(fresh_range(0));

    for (position, half_extent, color) in ROOM {
        let slab = world.spawn_with(Solid { _pad: 0 });
        world.put(slab, Renderable::boxed(*position, *half_extent, *color));
    }

    // The material chart. **Not `Solid`, unlike everything else in the room**
    // (§6 M26): the walk resolves against axis-aligned boxes, so a solid ball
    // would collide as the cube it is inscribed in — an invisible corner to
    // catch on, in mid-air, in the room's one open span. It hangs where it is
    // read from and nothing walks into it.
    for (at, smoothness, metallic) in chart() {
        let ball = world.spawn();
        world.put(
            ball,
            Renderable::ball(at, CHART_RADIUS, CHART_INK).surfaced(smoothness, metallic),
        );
    }

    // The environment the chart is read against (§6 M24). Without one the metal
    // row is black — a conductor has no diffuse lobe, so what it shows is
    // whatever is around it and nothing else.
    world.spawn_with(Sky::daylight(SKY_INTENSITY));
    // And the shelter's own, bounded to the space under its roof (§6 M28). The
    // room is open to the sky, so before this the light under a roof was the
    // light beside it — which is the one thing a player standing in a doorway
    // notices without being told to look.
    world.spawn_with(shelter_sky());

    world.spawn_with(Light::sun(SUN, SUN_INK, SUN_INTENSITY));
    for at in LAMPS {
        world.spawn_with(Light::point(at, LAMP_INK, LAMP_INTENSITY, LAMP_RANGE));
    }

    // A rising note and a falling one, the same waveform, the landing heavier —
    // a matched pair, which is what stops two cues fired a second apart from
    // reading as two unrelated noises.
    //
    // The jump was a square sweeping 300 -> 480 Hz and it was harsh for a reason
    // worth writing down, because `gain` is the knob one reaches for and it is
    // the wrong one. A square's odd harmonics fall off as 1/n, so a 480 Hz
    // fundamental puts real energy at 1.4, 2.4 and 3.4 kHz — the octave the ear
    // is most sensitive to by something like 10 dB. Peak amplitude said 0.16 and
    // the loudness the ear reported was nowhere near it. A triangle's harmonics
    // fall off as 1/n^2, which is the whole difference; dropping the register
    // and the gain is then arithmetic on a sound that is no longer grating.
    // The shot is the tree's first `wave::NOISE`, and it is the right waveform
    // for the reason the jump cue is a triangle: a gunshot is broadband, and
    // any tone at all would read as a pitch the room does not have. Swept down
    // and short, so a burst is six separate reports rather than one buzz.
    for (kind, sound) in [
        (CUE_JUMP, chirp(wave::TRIANGLE, 180.0, 260.0, 90, 0.10)),
        (CUE_LAND, chirp(wave::TRIANGLE, 150.0, 110.0, 60, 0.20)),
        (CUE_SHOT, chirp(wave::NOISE, 1400.0, 420.0, 55, 0.14)),
        (CUE_HIT, chirp(wave::TRIANGLE, 660.0, 990.0, 70, 0.11)),
        (CUE_ESCAPE, chirp(wave::TRIANGLE, 240.0, 165.0, 140, 0.13)),
        (CUE_OVER, chirp(wave::TRIANGLE, 200.0, 85.0, 420, 0.18)),
    ] {
        let entity = world.spawn_with(Cue { kind });
        world.put(entity, sound);
    }

    let (arm, thick, gap) = CROSS;
    let (cx, cy) = CENTRE;
    for (index, rect) in [
        [cx - gap - arm, cy - thick / 2.0, arm, thick],
        [cx + gap, cy - thick / 2.0, arm, thick],
        [cx - thick / 2.0, cy - gap - arm, thick, arm],
        [cx - thick / 2.0, cy + gap, thick, arm],
    ]
    .into_iter()
    .enumerate()
    {
        let line = HUD_CROSS + index as u32;
        let mut widget = Widget::panel(rect, CROSS_INK);
        // Distinct ids even though nothing hit-tests these: two live widgets
        // sharing one is a conflict the host reports rather than arbitrates.
        widget.id = widget_id("shooter.cross") ^ u64::from(line);
        widget.order = line;
        let entity = world.spawn_with(Hud { line });
        world.put(entity, widget);
    }

    // The globals, on an entity of their own: `restart` rewrites the body and
    // nothing else, and settings that a respawn cleared would be a bug shaped
    // like a feature.
    let globals = world.spawn_with(Session {
        paused: 0,
        page: PAGE_MAIN,
        sens: SENS_DEFAULT,
        _pad: 0,
    });
    // Antialiasing on by default *here*, where it is a game's state and no gate
    // reads it — `r.aa`/`r.msaa` stay off engine-wide because every blessed
    // golden was rendered without them (§4.10).
    //
    // 4× rather than 8×: it is the count both backends in the execution matrix
    // advertise, so the opening picture is the same one everywhere, and a
    // player who wants more has a button for it.
    world.put(
        globals,
        Prefs {
            aa: aa::MSAA_4,
            ..Default::default()
        },
    );

    for item in MENU_ITEMS {
        let mut widget = match item {
            MENU_SCRIM => Widget::panel(HIDDEN, MENU_SCRIM_INK),
            MENU_PANEL => Widget::panel(HIDDEN, MENU_PANEL_INK),
            MENU_TITLE | MENU_LOOK | MENU_VOLUME => Widget::label(HIDDEN, MENU_TEXT_INK, ""),
            _ => Widget::button(0, HIDDEN, MENU_BUTTON_INK, MENU_TEXT_INK, label_of(item)),
        };
        widget.id = widget_id("shooter.menu") ^ u64::from(item);
        // The item *is* the draw order (see `MENU_SCRIM`): the scrim under the
        // panel under the rows, and the whole menu over a HUD that stops at 11.
        widget.order = item;
        let entity = world.spawn_with(Menu { item });
        world.put(entity, widget);
    }

    for (line, text) in HUD_ROWS {
        let mut widget = Widget::label(
            [8.0, 6.0 + 14.0 * line as f32, 150.0, 12.0],
            0xffe8_f0ff,
            text,
        );
        widget.id = widget_id("shooter.hud") ^ u64::from(line);
        widget.order = line;
        let entity = world.spawn_with(Hud { line });
        world.put(entity, widget);
    }

    world.log(log_level::INFO, "shooter: ready");
}

/// Turn. Before [`walk`] in the table, so this tick's look steers this tick's
/// step — a controller where they disagree feels like input lag and is.
pub fn aim(world: &mut GameWorld) {
    let session = session_of(world);
    if session.paused != 0 {
        // Raw device motion arrives whether the host holds the pointer or not —
        // it is a *device* delta, not a position — so a paused game that did not
        // refuse it would spin the camera behind its own menu.
        return;
    }
    let per_count = look_per_count(session.sens);
    let (x, y) = (world.axis(AIM_X), world.axis(AIM_Y));
    world.visit::<&mut Walker>(|_, walker| {
        walker.yaw = wrap(walker.yaw - x * per_count);
        walker.pitch = (walker.pitch - y * per_count).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    });
}

/// One tick of movement: steer, gravity, integrate-and-resolve one axis at a
/// time, then the buffered/coyote jump.
///
/// X and Z resolve before Y so a wall is a wall while the feet are still on
/// last tick's resolved ground — the ordering demo 11 established, with the
/// second horizontal axis added.
pub fn walk(world: &mut GameWorld) {
    if session_of(world).paused != 0 {
        // The whole tick of motion, not a zeroed input: gravity is in here, and
        // a body that kept falling behind the menu would land while nobody was
        // looking at it.
        return;
    }
    let strafe = f64::from(world.axis(MOVE_RIGHT)).clamp(-1.0, 1.0);
    let ahead = f64::from(world.axis(MOVE_FORWARD)).clamp(-1.0, 1.0);
    let jump_held = world.pressed(JUMP);
    let jump_edge = world.just_pressed(JUMP);
    let solids = solids(world);

    let mut jumped = false;
    let mut landed = false;
    world.visit::<&mut Walker>(|_, walker| {
        // The fly basis at *zero pitch*: looking at the ceiling must not walk
        // you into it. Yaw-only is what makes this a walk and not a fly.
        let (forward, right) = sim::fly_basis(walker.yaw, 0.0);
        let wish = right * strafe + forward * ahead;
        let reach = wish.length();
        // Normalized, unlike demo 06's fly camera: a diagonal 41 % faster than
        // a straight line is a fly camera's traditional feature and a shooter's
        // traditional bug.
        let target = if reach > 0.0 {
            wish * (WALK_SPEED / reach)
        } else {
            sim::DVec3::ZERO
        };
        let accel = if walker.grounded != 0 {
            GROUND_ACCEL
        } else if reach > 0.0 {
            AIR_ACCEL
        } else {
            // No input in the air: no accel at all rather than accel toward
            // zero, so a jump keeps the speed it left with.
            0.0
        };
        walker.velocity.x += (target.x - walker.velocity.x).clamp(-accel, accel);
        walker.velocity.z += (target.z - walker.velocity.z).clamp(-accel, accel);

        let pull = if walker.velocity.y > 0.0 && !jump_held {
            GRAVITY * RISE_CUT
        } else {
            GRAVITY
        };
        walker.velocity.y = (walker.velocity.y - pull).max(-MAX_FALL);

        walker.buffer = if jump_edge {
            BUFFER_TICKS
        } else {
            walker.buffer.saturating_sub(1)
        };

        walker.position.x += walker.velocity.x;
        resolve_x(walker, &solids);
        walker.position.z += walker.velocity.z;
        resolve_z(walker, &solids);

        let was_grounded = walker.grounded != 0;
        walker.position.y += walker.velocity.y;
        let supported = resolve_y(walker, &solids);
        walker.grounded = u32::from(supported);
        walker.coyote = if supported {
            COYOTE_TICKS
        } else {
            walker.coyote.saturating_sub(1)
        };
        if supported && !was_grounded {
            landed = true;
        }

        // After the landing check, so a buffered press fires on the landing
        // tick itself — the buffer's whole reason to exist.
        if walker.buffer > 0 && walker.coyote > 0 {
            walker.velocity.y = JUMP_VELOCITY;
            walker.buffer = 0;
            walker.coyote = 0;
            walker.grounded = 0;
            jumped = true;
        }

        if walker.position.y < KILL_Y {
            *walker = at_start();
        }
    });

    // Independently, not `else if`: a buffered press fires *on* the landing
    // tick, and that tick is both events. Silencing the thud there would make a
    // clean hop the one landing you cannot hear.
    if jumped {
        cue(world, CUE_JUMP);
    }
    if landed {
        cue(world, CUE_LAND);
    }
}

/// The shot. After [`walk`] in the table, so the ray leaves the eye where this
/// tick left it — a hitscan fired from last tick's position disagrees with the
/// picture the player aimed at, by 110 mm at walking speed.
pub fn shoot(world: &mut GameWorld) {
    if session_of(world).paused != 0 {
        // The trigger and the menu's click are one physical button (see this
        // crate's `input.toml`), and these two lines are the whole of the
        // arbitration: the menu is up, so the button is the menu's, and the
        // trigger is disarmed until it is let go of. The host decides who holds
        // the *pointer* off the same fact, by a different route (§4.9).
        world.visit::<&mut Gun>(|_, gun| gun.armed = 0);
        return;
    }
    let mut range = range_of(world);
    // Held, not the edge: this is an automatic weapon and [`FIRE_TICKS`] is
    // what makes the rate a rate rather than a measure of how fast a hand can
    // click.
    let held = world.pressed(FIRE);
    let firing = held && range.state == STATE_RUNNING;

    let mut shot = None;
    world.visit::<(&mut Walker, &mut Gun)>(|_, (walker, gun)| {
        if gun.recover > 0 {
            walker.pitch = (walker.pitch - RECOIL_GIVE).clamp(-PITCH_LIMIT, PITCH_LIMIT);
            gun.recover -= 1;
        }
        gun.cooldown = gun.cooldown.saturating_sub(1);
        if !held {
            gun.armed = 1;
        }
        if !firing || gun.cooldown != 0 || gun.armed == 0 {
            return;
        }
        gun.cooldown = FIRE_TICKS;
        // Drawn before the kick: the round goes where the player aimed and the
        // weapon answers afterwards. The other order is a gun that punishes the
        // shot you have already taken.
        let yaw = walker.yaw + spread(&mut gun.rng);
        let pitch = walker.pitch + spread(&mut gun.rng);
        walker.pitch = (walker.pitch + RECOIL_KICK).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        gun.recover = RECOIL_TICKS;
        shot = Some((eye_of(walker), yaw, pitch));
    });
    let Some((eye, yaw, pitch)) = shot else {
        return;
    };

    let (direction, right) = sim::fly_basis(yaw, pitch);
    let contact = cast(world, eye, direction);
    range.shots += 1;

    // The tracer comes off a muzzle and the bullet comes off the eye. Not an
    // inconsistency — item 5's answer is no view model at all, and a streak
    // that started at the crosshair would read as one anyway while also being
    // the one thing in the frame with no parallax.
    let muzzle =
        eye + right * MUZZLE.0 + sim::DVec3::new(0.0, MUZZLE.1, 0.0) + direction * MUZZLE.2;
    let end = contact.map_or(eye + direction * SHOT_RANGE, |hit| hit.at);
    tracer(
        world,
        muzzle,
        direction,
        yaw,
        pitch,
        (end - muzzle).length(),
    );
    flash(world, muzzle);
    cue(world, CUE_SHOT);

    match contact {
        Some(Contact {
            at,
            normal,
            target: Some((entity, worth)),
        }) => {
            world.despawn(entity);
            range.hits += 1;
            range.score += worth + STREAK_STEP * range.streak.min(STREAK_CAP);
            range.streak += 1;
            range.hitmark = HITMARK_TICKS;
            burst(world, &mut range.rng, at, normal, HIT_INK);
            cue(world, CUE_HIT);
            if range.hits == 1 {
                world.log(log_level::INFO, "shooter: hit");
            }
        }
        // A wall, or nothing at all: both end the streak, and the difference
        // between them is a puff of dust. Nothing is reachable — the room is
        // closed on five sides and open to the sky, so a shot aimed up meets
        // no surface and its tracer is the one thing this game deals that
        // leaves the room (§6 M37 item 3).
        other => {
            range.streak = 0;
            if let Some(Contact { at, normal, .. }) = other {
                burst(world, &mut range.rng, at, normal, DUST_INK);
            }
        }
    }
    world.visit::<&mut Range>(|_, held| *held = range);
}

/// The round: age the targets, count the ones that walked, and keep
/// [`TARGETS_LIVE`] of them standing. After [`shoot`], so a target taken this
/// tick frees its slot on the same tick and the course is never one short.
pub fn targets(world: &mut GameWorld) {
    if session_of(world).paused != 0 {
        return;
    }
    let mut range = range_of(world);
    range.hitmark = range.hitmark.saturating_sub(1);
    if range.state != STATE_RUNNING {
        world.visit::<&mut Range>(|_, held| *held = range);
        return;
    }

    let mut escaped = Vec::new();
    let mut held = Vec::new();
    world.visit::<(&mut Target, &Renderable)>(|entity, (target, shape)| {
        target.age += 1;
        if target.age >= target.life {
            escaped.push((entity, shape.position));
        } else {
            held.push(target.slot);
        }
    });

    for (entity, at) in escaped {
        world.despawn(entity);
        range.misses += 1;
        range.streak = 0;
        burst(
            world,
            &mut range.rng,
            at,
            sim::DVec3::new(0.0, 1.0, 0.0),
            ESCAPE_INK,
        );
        cue(world, CUE_ESCAPE);
        if range.misses == 1 {
            world.log(log_level::INFO, "shooter: escaped");
        }
    }
    // Once: the round is `STATE_OVER` from here and the early return above is
    // what makes the milestone a milestone rather than a line per tick.
    if range.misses >= MISSES_ALLOWED {
        range.state = STATE_OVER;
        range.best = range.best.max(range.score);
        cue(world, CUE_OVER);
        world.log(log_level::INFO, "shooter: over");
    }

    while range.state == STATE_RUNNING && held.len() < TARGETS_LIVE {
        let free: Vec<u32> = (0..SPOTS.len() as u32)
            .filter(|slot| !held.contains(slot))
            .collect();
        // `below` is `None` only on an empty range — more targets than places,
        // which is a table edit and not a tick this can recover inside.
        let Some(pick) = range.rng.below(free.len() as u32) else {
            break;
        };
        let slot = free[pick as usize];
        held.push(slot);
        deal(world, slot);
    }
    world.visit::<&mut Range>(|_, held| *held = range);
}

/// Age everything a shot left behind, and move the one kind that goes anywhere.
/// Last of the gameplay systems, so a chip dealt this tick is drawn where it was
/// dealt rather than one step along it.
pub fn effects(world: &mut GameWorld) {
    if session_of(world).paused != 0 {
        return;
    }
    world.visit::<(&mut Spark, &mut Renderable)>(|_, (spark, shape)| {
        if spark.kind != SPARK_DEBRIS {
            return;
        }
        spark.velocity.y -= SPARK_GRAVITY;
        shape.position += spark.velocity;
        // Spent where it meets the floor rather than sinking through it. Not a
        // resolve — a chip is not a body — but it is what keeps every transient
        // this game deals inside the room's own bounds, which is the claim
        // §6 M37 item 3 measures rather than assumes.
        if shape.position.y <= 0.0 {
            shape.position.y = 0.0;
            spark.life = 1;
        }
    });
    let mut spent = Vec::new();
    world.visit::<&mut Spark>(|entity, spark| {
        spark.life = spark.life.saturating_sub(1);
        if spark.life == 0 {
            spent.push(entity);
        }
    });
    for entity in spent {
        world.despawn(entity);
    }
}

/// The pause key, the buttons, and the rectangles they leave behind. Before
/// [`aim`] in the table, so Escape stops the world on the tick it was pressed
/// rather than one later — a pause that let a flick through is a pause you can
/// die during.
///
/// Reads clicks off `Widget::state`, which the host wrote at the end of *last*
/// tick (§4.9). That one-tick lag is the protocol's, not this system's, and it
/// is why a click is a `clicked()` edge rather than a held bit.
pub fn menu(world: &mut GameWorld) {
    let mut session = session_of(world);
    let mut prefs = prefs_of(world);

    // Escape layers: out of the settings page first, then out of the menu. One
    // key walking back one level is what every game does, and it is also what
    // keeps the page you left on from being the page you return to.
    if world.just_pressed(PAUSE) {
        match (session.paused != 0, session.page) {
            (true, PAGE_SETTINGS) => session.page = PAGE_MAIN,
            (true, _) => session.paused = 0,
            (false, _) => {
                session.paused = 1;
                session.page = PAGE_MAIN;
            }
        }
    }

    // At most one: a release lands over one widget, and the host has already
    // decided which (`Widget::order` breaks an overlap, not this).
    let mut hit = 0;
    world.visit::<(&Menu, &Widget)>(|_, (menu, widget)| {
        if widget.clicked() {
            hit = menu.item;
        }
    });
    let mut respawning = false;
    match hit {
        MENU_RESUME => session.paused = 0,
        MENU_SETTINGS => session.page = PAGE_SETTINGS,
        MENU_BACK => session.page = PAGE_MAIN,
        // Straight back into the game: a restart that left you looking at the
        // menu would need a second click to do the thing you asked for.
        MENU_RESTART => {
            respawning = true;
            session.paused = 0;
        }
        MENU_QUIT => prefs.close = 1,
        MENU_LOOK_DOWN => session.sens = session.sens.saturating_sub(SENS_STEP).max(SENS_MIN),
        MENU_LOOK_UP => session.sens = session.sens.saturating_add(SENS_STEP).min(SENS_MAX),
        // Quiet is attenuation, so the *down* button adds — see `Prefs::quiet`,
        // where zero-is-loud is a migration contract rather than a preference.
        MENU_VOLUME_DOWN => prefs.quiet = prefs.quiet.saturating_add(QUIET_STEP).min(QUIET_MAX),
        MENU_VOLUME_UP => prefs.quiet = prefs.quiet.saturating_sub(QUIET_STEP),
        // Never `DEFAULT`: a row that reads OFF while the field means "ask the
        // host" is a row that lies whenever `r.aa` or `r.msaa` is on.
        MENU_AA => prefs.aa = next_aa(prefs.aa),
        _ => {}
    }

    world.visit::<&mut Session>(|_, held| *held = session);
    world.visit::<&mut Prefs>(|_, held| *held = prefs);
    if respawning {
        respawn(world);
    }
    lay_out(world, session, prefs);
}

/// Put the menu where the session says it is — and, by the same act, decide who
/// holds the mouse: a hidden widget is a zero rect (§4.9), a canvas with no
/// hit-tested area is a canvas with nothing to point at, and the host hands the
/// pointer back to mouse-look on exactly that condition.
pub fn lay_out(world: &mut GameWorld, session: Session, prefs: Prefs) {
    use core::fmt::Write as _;
    let showing = session.paused != 0;
    let page = session.page;

    // Two numbers, because the multiplier alone answers "faster or slower" and
    // nothing else. Counts-per-turn is the one a hand can be set against: divide
    // by the mouse's DPI for inches of desk per full turn.
    let mut look = Line::default();
    let _ = write!(
        look,
        "LOOK {}.{:02}  {}/TURN",
        session.sens / SENS_ONE,
        session.sens % SENS_ONE,
        counts_per_turn(session.sens)
    );
    let mut volume = Line::default();
    let _ = write!(
        volume,
        "VOLUME {}%",
        (QUIET_MAX - prefs.quiet.min(QUIET_MAX)) * 100 / QUIET_MAX
    );

    world.visit::<(&Menu, &mut Widget)>(|_, (menu, widget)| {
        widget.rect = match showing && on_page(menu.item, page) {
            true => slot(menu.item),
            false => HIDDEN,
        };
        match menu.item {
            MENU_TITLE => widget.set_text(match page {
                PAGE_SETTINGS => "SETTINGS",
                _ => "PAUSED",
            }),
            MENU_LOOK => widget.set_text(look.as_str()),
            MENU_VOLUME => widget.set_text(volume.as_str()),
            MENU_AA => widget.set_text(aa_label(prefs.aa)),
            _ => {}
        }
    });
}

/// Say what the tick looks like: where the eye is, and what the HUD reads. Last
/// in the table, so it describes the tick that just happened (§4.5 v0).
pub fn present(world: &mut GameWorld) {
    let mut seen = None;
    world.visit::<&Walker>(|entity, walker| seen = Some((entity, *walker)));
    let Some((entity, walker)) = seen else {
        return;
    };
    world.put(entity, Eye::at(eye_of(&walker), walker.yaw, walker.pitch));

    // Ground speed in tenths of a metre per second: the tick rate is the
    // conversion, and it is an integer, so no float clock appears anywhere.
    let ground = sim::DVec3::new(walker.velocity.x, 0.0, walker.velocity.z).length();
    // Rounded, not truncated: full speed lands an ULP under the constant that
    // names it, and a truncating readout would report 6.5 for 6.6 for ever.
    let tenths = (ground * f64::from(world.tick_hz()) * 10.0 + 0.5) as u64;
    let grounded = walker.grounded != 0;

    use core::fmt::Write as _;
    let mut speed = Line::default();
    let _ = write!(speed, "SPEED {}.{}", tenths / 10, tenths % 10);

    let range = range_of(world);
    let mut score = Line::default();
    let _ = write!(score, "SCORE {}  X{}", range.score, range.streak + 1);
    let mut best = Line::default();
    let _ = write!(best, "BEST {}", range.best.max(range.score));
    let mut misses = Line::default();
    let _ = if range.state == STATE_OVER {
        write!(misses, "OVER - R TO RESTART")
    } else {
        write!(misses, "MISS {}/{MISSES_ALLOWED}", range.misses)
    };
    let cross = if range.hitmark > 0 {
        HITMARK_INK
    } else {
        CROSS_INK
    };

    world.visit::<(&Hud, &mut Widget)>(|_, (hud, widget)| match hud.line {
        HUD_SPEED => widget.set_text(speed.as_str()),
        HUD_STATE => widget.set_text(if grounded { "GROUND" } else { "AIR" }),
        HUD_SCORE => widget.set_text(score.as_str()),
        HUD_BEST => widget.set_text(best.as_str()),
        HUD_MISS => widget.set_text(misses.as_str()),
        // The four crosshair rects, and nothing else reaches here.
        _ => widget.color = cross,
    });
}

// ---------------------------------------------------------------- innards ---

/// A body at [`START`], stationary, facing the stairs. One place, so `restart`,
/// `bootstrap` and the kill plane cannot drift apart.
fn at_start() -> Walker {
    Walker {
        position: START,
        velocity: sim::DVec3::ZERO,
        yaw: 0.0,
        pitch: 0.0,
        coyote: 0,
        buffer: 0,
        grounded: 0,
        _pad: 0,
    }
}

/// Put the body back at [`START`] and start a fresh round. Not a wipe: the room
/// is `bootstrap`'s and nothing here could deal it back — and neither the
/// session nor the settings are the body's to reset.
///
/// The *round* is reset here rather than left alone because that is what `R`
/// means in a game with a score; [`Range::best`] is the one thing carried
/// across, which is also the only thing a save would have to carry the day this
/// game gets one (M14's `--best` is the shape).
fn respawn(world: &mut GameWorld) {
    world.visit::<&mut Walker>(|_, walker| *walker = at_start());
    world.visit::<&mut Gun>(|_, gun| *gun = fresh_gun());
    world.visit::<&mut Range>(|_, range| *range = fresh_range(range.best.max(range.score)));
    // The course starts empty: a target left standing would be one the new
    // round did not deal and the old one did not finish, and its slot would be
    // held by a `Target` whose age belongs to a score that no longer exists.
    let mut clear = Vec::new();
    world.visit::<&Target>(|entity, _| clear.push(entity));
    world.visit::<&Spark>(|entity, _| clear.push(entity));
    for entity in clear {
        world.despawn(entity);
    }
    world.log(log_level::INFO, "shooter: restarted");
}

/// A cooled weapon on a fresh stream. Seeded from `!`[`SEED`] rather than
/// [`SEED`]: two generators started at the same state draw the same numbers,
/// and a spread that agreed with a spawn order would be a correlation nobody
/// could see and everybody would feel.
fn fresh_gun() -> Gun {
    Gun {
        rng: sim::Rng::from_seed(!SEED),
        cooldown: 0,
        recover: 0,
        armed: 0,
        _pad: 0,
    }
}

/// A round at zero, carrying `best`.
///
/// Reseeded from the same constant every time, so every round deals the same
/// course in the same order. That is a decision and not an oversight: a skill
/// game whose targets move differently each run is one where a better score can
/// be a luckier draw, and a fixed course is also what lets a recorded session
/// be a *session* rather than a seed.
fn fresh_range(best: u32) -> Range {
    Range {
        rng: sim::Rng::from_seed(SEED),
        score: 0,
        best,
        streak: 0,
        misses: 0,
        hits: 0,
        shots: 0,
        state: STATE_RUNNING,
        hitmark: 0,
    }
}

/// The round as it stands, or the opening one — [`session_of`]'s rule.
fn range_of(world: &mut GameWorld) -> Range {
    let mut found = fresh_range(0);
    world.visit::<&Range>(|_, held| found = *held);
    found
}

/// Where a shot landed: the point, the outward normal of the face it met, and
/// the target it was — `None` being the room itself.
#[derive(Clone, Copy)]
struct Contact {
    at: sim::DVec3,
    normal: sim::DVec3,
    target: Option<(gg_ecs::Entity, u32)>,
}

/// The nearest surface the ray meets, targets and room in one pass — one
/// because a wall in front of a target has to stop the bullet, and two passes
/// with two answers would need a third to reconcile them.
///
/// `direction` must be unit ([`sim::fly_basis`]'s is), which is what makes the
/// distances metres rather than multiples of something.
fn cast(world: &mut GameWorld, from: sim::DVec3, direction: sim::DVec3) -> Option<Contact> {
    let ray = sim::DRay::new(from, direction);
    let mut best: Option<(f64, Contact)> = None;
    world.visit::<(&Target, &Renderable)>(|entity, (target, shape)| {
        if let Some((distance, normal)) = pierce(ray, shape) {
            nearer(
                &mut best,
                distance,
                Contact {
                    at: from + direction * distance,
                    normal,
                    target: Some((entity, target.worth)),
                },
            );
        }
    });
    world.visit::<(&Solid, &Renderable)>(|_, (_, shape)| {
        if let Some((distance, normal)) = pierce(ray, shape) {
            nearer(
                &mut best,
                distance,
                Contact {
                    at: from + direction * distance,
                    normal,
                    target: None,
                },
            );
        }
    });
    best.map(|(_, contact)| contact)
}

/// This game's policy on a span (§6 M37 item 1): **past the muzzle, inside the
/// weapon's reach**.
///
/// `enter <= 0` is the muzzle already inside the box, and that is a miss rather
/// than a point-blank hit: a target you are standing in is not one you can
/// shoot, and the *exit* is not the answer either, because a bullet does not
/// leave through the far face of something it never entered. The editor's pick
/// answers the same span the opposite way and M35's occlusion answers it a third
/// — which is why `gg_math::sim` returns the interval and none of the three.
fn pierce(ray: sim::DRay, shape: &Renderable) -> Option<(f64, sim::DVec3)> {
    let half = sim::DVec3::new(
        f64::from(shape.half_extent.x),
        f64::from(shape.half_extent.y),
        f64::from(shape.half_extent.z),
    );
    let span = ray.obb(shape.position, shape.rotation, half)?;
    (span.enter > 0.0 && span.enter <= SHOT_RANGE).then_some((span.enter, span.normal))
}

/// Keep the nearer of two contacts. Strictly nearer, so a target and a wall at
/// exactly the same distance leave the target standing — the pass that ran
/// first wins, and targets run first.
fn nearer(best: &mut Option<(f64, Contact)>, distance: f64, contact: Contact) {
    if best.is_none_or(|(previous, _)| distance < previous) {
        *best = Some((distance, contact));
    }
}

/// One target at `slot`.
fn deal(world: &mut GameWorld, slot: u32) {
    let entity = world.spawn_with(Target {
        age: 0,
        life: TARGET_LIFE,
        slot,
        worth: TARGET_WORTH,
    });
    world.put(
        entity,
        Renderable::boxed(
            SPOTS[slot as usize],
            sim::Vec3::splat(TARGET_HALF),
            TARGET_INK,
        )
        .surfaced(TARGET_SMOOTHNESS, 0.0),
    );
}

/// The bullet's streak: a long thin box lying along the ray that drew it.
///
/// [`Renderable`]'s own documentation offers this — "a beam is a long thin box,
/// and one primitive that stretches beats a second primitive" — and this is the
/// game that finally needed it. The rotation is built from the shot's own
/// yaw and pitch rather than fitted to its direction, so the box lies *on* the
/// ray instead of near it.
fn tracer(
    world: &mut GameWorld,
    from: sim::DVec3,
    direction: sim::DVec3,
    yaw: f32,
    pitch: f32,
    distance: f64,
) {
    let mut shape = Renderable::boxed(
        from + direction * (distance / 2.0),
        sim::Vec3::new(TRACER_HALF, TRACER_HALF, distance as f32 / 2.0),
        TRACER_INK,
    );
    shape.rotation = aim_quat(yaw, pitch);
    let entity = world.spawn_with(Spark {
        velocity: sim::DVec3::ZERO,
        life: TRACER_TICKS,
        kind: SPARK_TRACER,
    });
    world.put(entity, shape.surfaced(0.9, 0.0));
}

/// The muzzle flash: a [`Light`] with a life counter and no geometry at all.
///
/// The first light in this tree that appears and vanishes while a scene is being
/// played (§6 M37 item 4). It is also, by construction, the nearest point light
/// to the eye whenever it exists — which is exactly the thing the renderer ranks
/// casting slots by.
fn flash(world: &mut GameWorld, at: sim::DVec3) {
    let entity = world.spawn_with(Spark {
        velocity: sim::DVec3::ZERO,
        life: FLASH_TICKS,
        kind: SPARK_FLASH,
    });
    world.put(
        entity,
        Light::point(at, FLASH_INK, FLASH_INTENSITY, FLASH_RANGE),
    );
}

/// Chips off a face: [`SPARKS`] of them, out along the normal and scattered
/// across it.
fn burst(world: &mut GameWorld, rng: &mut sim::Rng, at: sim::DVec3, normal: sim::DVec3, ink: u32) {
    let from = at + normal * f64::from(SPARK_HALF);
    for _ in 0..SPARKS {
        let velocity =
            normal * SPARK_LIFT + sim::DVec3::new(scatter(rng), scatter(rng), scatter(rng));
        let entity = world.spawn_with(Spark {
            velocity,
            life: SPARK_TICKS,
            kind: SPARK_DEBRIS,
        });
        world.put(
            entity,
            Renderable::boxed(from, sim::Vec3::splat(SPARK_HALF), ink).surfaced(0.5, 0.0),
        );
    }
}

/// A centred integer draw in `±(SPREAD_STEPS-1)/2`.
///
/// Integers and one multiply at each use, because [`sim::Rng`] has **no float
/// output** on purpose (§6 M18) — and this is the shape that costs nothing for
/// it: an integer stream is bit-identical on every target with no `libm`
/// question anywhere near it.
fn step(rng: &mut sim::Rng) -> i32 {
    rng.below(SPREAD_STEPS).unwrap_or(0) as i32 - (SPREAD_STEPS as i32 - 1) / 2
}

/// Radians of aim error on one axis of one shot.
fn spread(rng: &mut sim::Rng) -> f32 {
    step(rng) as f32 * SPREAD_UNIT
}

/// Metres per tick of scatter on one axis of one chip.
fn scatter(rng: &mut sim::Rng) -> f64 {
    f64::from(step(rng)) * SPARK_SCATTER
}

/// The rotation taking a box's local `-Z` onto a shot fired at `yaw`/`pitch` —
/// yaw about world `+Y`, then pitch about the rotated right axis, which is the
/// one order [`sim::fly_basis`] builds its forward in.
fn aim_quat(yaw: f32, pitch: f32) -> sim::DQuat {
    sim::DQuat::from_axis_angle(sim::DVec3::new(0.0, 1.0, 0.0), f64::from(yaw)).mul(
        sim::DQuat::from_axis_angle(sim::DVec3::new(1.0, 0.0, 0.0), f64::from(pitch)),
    )
}

/// Radians of turn one mouse count is worth at `sens` — [`LOOK_PER_UNIT`] scaled
/// by the setting, and the only place the two meet.
#[must_use]
pub fn look_per_count(sens: u32) -> f32 {
    LOOK_PER_UNIT * sens as f32 / SENS_ONE as f32
}

/// Mouse counts in a full turn at `sens` — what the settings row reports,
/// because it is the number a hand can be set against: divide by the mouse's
/// DPI for inches of desk per 360.
#[must_use]
pub fn counts_per_turn(sens: u32) -> u32 {
    (core::f32::consts::TAU / look_per_count(sens)) as u32
}

/// The antialiasing modes this game offers, in the order the row cycles them.
///
/// The whole list the boundary has, minus `DEFAULT` — a game with a video menu
/// has an opinion about every mode, and "ask the host" is not one a player can
/// read off a button. A count this device cannot do is reduced by the host and
/// said out loud in the log; the row keeps reading what was *asked* for, which
/// is the only thing this world knows.
const AA_MODES: [u32; 5] = [aa::OFF, aa::FXAA, aa::MSAA_2, aa::MSAA_4, aa::MSAA_8];

/// The next mode along, wrapping. An unknown one — a save from a build with
/// more modes than this — lands on the first rather than sticking.
fn next_aa(mode: u32) -> u32 {
    let at = AA_MODES.iter().position(|m| *m == mode);
    AA_MODES[at.map_or(0, |i| (i + 1) % AA_MODES.len())]
}

/// What the edges row reads.
fn aa_label(mode: u32) -> &'static str {
    match mode {
        aa::FXAA => "EDGES  FXAA",
        aa::MSAA_2 => "EDGES  MSAA 2X",
        aa::MSAA_4 => "EDGES  MSAA 4X",
        aa::MSAA_8 => "EDGES  MSAA 8X",
        _ => "EDGES  OFF",
    }
}

/// The session as it stands, or the opening one before `bootstrap` has dealt
/// it. A default rather than an `Option` because every caller wants the same
/// answer for "no session yet" — running, at full sensitivity.
fn session_of(world: &mut GameWorld) -> Session {
    let mut found = Session {
        paused: 0,
        page: PAGE_MAIN,
        sens: SENS_DEFAULT,
        _pad: 0,
    };
    world.visit::<&Session>(|_, held| found = *held);
    found
}

/// The preferences as they stand — the first walked, that type's own rule.
fn prefs_of(world: &mut GameWorld) -> Prefs {
    let mut found = Prefs::default();
    world.visit::<&Prefs>(|_, held| found = *held);
    found
}

/// Whether an item belongs to the page being shown.
fn on_page(item: u32, page: u32) -> bool {
    match item {
        MENU_SCRIM | MENU_PANEL | MENU_TITLE => true,
        MENU_RESUME | MENU_SETTINGS | MENU_RESTART | MENU_QUIT => page == PAGE_MAIN,
        _ => page == PAGE_SETTINGS,
    }
}

/// A button's fixed text. The three rows whose text is a *value* are written
/// every tick by [`lay_out`] and start empty here.
fn label_of(item: u32) -> &'static str {
    match item {
        MENU_RESUME => "RESUME",
        MENU_SETTINGS => "SETTINGS",
        MENU_RESTART => "RESTART",
        MENU_QUIT => "QUIT",
        MENU_BACK => "BACK",
        MENU_LOOK_DOWN | MENU_VOLUME_DOWN => "-",
        MENU_LOOK_UP | MENU_VOLUME_UP => "+",
        _ => "",
    }
}

/// Where the eye sits for a given body.
fn eye_of(walker: &Walker) -> sim::DVec3 {
    sim::DVec3::new(
        walker.position.x,
        walker.position.y + EYE_LIFT,
        walker.position.z,
    )
}

/// Keep an angle in ±π. An unbounded yaw loses precision with uptime, and it is
/// hashed — demo 06's argument about the sun's phase, in radians.
fn wrap(yaw: f32) -> f32 {
    const TAU: f32 = core::f32::consts::TAU;
    const PI: f32 = core::f32::consts::PI;
    let mut yaw = yaw;
    // Bounded rather than `while`: a flick is worth a couple of radians and
    // eight turns is far past any tick's pointer delta, while a `while` would
    // spin for ever on the NaN this is the last line of defence against.
    for _ in 0..8 {
        if yaw > PI {
            yaw -= TAU;
        } else if yaw < -PI {
            yaw += TAU;
        } else {
            break;
        }
    }
    yaw
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

/// A solid, widened to `f64` once so the resolve arithmetic is one width.
struct Box3 {
    x: f64,
    y: f64,
    z: f64,
    hx: f64,
    hy: f64,
    hz: f64,
}

/// Every solid's box, read from its `Renderable`.
fn solids(world: &mut GameWorld) -> Vec<Box3> {
    let mut out = Vec::new();
    world.visit::<(&Solid, &Renderable)>(|_, (_, shape)| {
        out.push(Box3 {
            x: shape.position.x,
            y: shape.position.y,
            z: shape.position.z,
            hx: f64::from(shape.half_extent.x),
            hy: f64::from(shape.half_extent.y),
            hz: f64::from(shape.half_extent.z),
        });
    });
    out
}

/// Whether the body box and a slab share volume. Strict, so exact contact —
/// which is what a resolve leaves behind — is not a collision.
fn overlaps(walker: &Walker, slab: &Box3) -> bool {
    (walker.position.x - slab.x).abs() < HALF_W + slab.hx
        && (walker.position.y - slab.y).abs() < HALF_H + slab.hy
        && (walker.position.z - slab.z).abs() < HALF_W + slab.hz
}

/// How far the body would have to rise to stand on everything it is currently
/// inside, if that is a step rather than a wall.
///
/// `None` refuses the lift, and it refuses for four distinct reasons worth
/// keeping separate: airborne (a lift would be a teleport), nothing overlapping
/// (nothing to step onto), one of the overlaps taller than [`STEP_HEIGHT`] (a
/// wall — and one wall in the set is enough), or a lift that would leave the
/// body inside something else (a low ledge under a low ceiling).
fn step_lift(walker: &Walker, solids: &[Box3]) -> Option<f64> {
    if walker.grounded == 0 {
        return None;
    }
    let feet = walker.position.y - HALF_H;
    let mut lift = 0.0;
    let mut touching = false;
    for slab in solids {
        if !overlaps(walker, slab) {
            continue;
        }
        touching = true;
        let rise = slab.y + slab.hy - feet;
        if rise > STEP_HEIGHT {
            return None;
        }
        if rise > lift {
            lift = rise;
        }
    }
    if !touching || lift <= 0.0 {
        return None;
    }
    let mut lifted = *walker;
    lifted.position.y += lift;
    solids
        .iter()
        .all(|slab| !overlaps(&lifted, slab))
        .then_some(lift)
}

/// Push the body out of every slab it overlaps along x — or step onto them.
///
/// The coordinate is *set* [`SKIN`] clear of the slab's face, never nudged by
/// the penetration depth: `p += span - |d|` lands an ULP short as readily as
/// long, and an overlap a single ULP deep is one [`resolve_y`] then has to
/// interpret.
fn resolve_x(walker: &mut Walker, solids: &[Box3]) {
    if let Some(lift) = step_lift(walker, solids) {
        walker.position.y += lift;
        return;
    }
    for slab in solids {
        if !overlaps(walker, slab) {
            continue;
        }
        walker.position.x = if walker.position.x >= slab.x {
            slab.x + slab.hx + HALF_W + SKIN
        } else {
            slab.x - slab.hx - HALF_W - SKIN
        };
        walker.velocity.x = 0.0;
    }
}

/// Push the body out of every slab it overlaps along z — or step onto them.
fn resolve_z(walker: &mut Walker, solids: &[Box3]) {
    if let Some(lift) = step_lift(walker, solids) {
        walker.position.y += lift;
        return;
    }
    for slab in solids {
        if !overlaps(walker, slab) {
            continue;
        }
        walker.position.z = if walker.position.z >= slab.z {
            slab.z + slab.hz + HALF_W + SKIN
        } else {
            slab.z - slab.hz - HALF_W - SKIN
        };
        walker.velocity.z = 0.0;
    }
}

/// Push the body out along y, in the direction opposite the motion that put it
/// there. Returns whether something took its weight.
///
/// The direction is the *velocity's*, never a comparison of centres. A wall is
/// 4 m tall and a standing body's centre is 0.9 m up, so "eject toward the
/// nearer face" posts the body underneath the wall and out of the world — which
/// is precisely what it did before this comment existed.
fn resolve_y(walker: &mut Walker, solids: &[Box3]) -> bool {
    let falling = walker.velocity.y <= 0.0;
    let mut supported = false;
    for slab in solids {
        if !overlaps(walker, slab) {
            continue;
        }
        if falling {
            walker.position.y = slab.y + slab.hy + HALF_H + SKIN;
            supported = true;
        } else {
            walker.position.y = slab.y - slab.hy - HALF_H - SKIN;
        }
        walker.velocity.y = 0.0;
    }
    supported
}

/// A fixed-capacity HUD line. The HUD path runs every tick and allocates
/// nothing; truncation is fine — every line is ASCII.
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
    components: [
        Walker, Solid, Cue, Hud, Menu, Session, Gun, Range, Target, Spark,
        Renderable, Light, Sky, Eye, Widget, Sound, Prefs
    ],
    actions: ["jump", "restart", "pause", "ui_click", "ui_focus", "fire"],
    axes: ["move_right", "move_forward", "aim_x", "aim_y", "ui_x", "ui_y"],
    systems: [restart, bootstrap, menu, aim, walk, shoot, targets, effects, present],
}
