//! Demo 12 — **the room** (the game ladder's third rung): stand up, look
//! around, walk, jump, land.
//!
//! What Unreal opens into, in this engine's terms: a white-grey block room you
//! move through in first person. This chunk is the *character controller* — the
//! first 3D gameplay in the tree — and deliberately nothing else. The gun, the
//! targets and the score are a second pass; a controller that is wrong is wrong
//! under everything built on it, so it gets its own pass and its own tuning.
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

// ---------------------------------------------------------------- chart -----

/// Steps across the smoothness axis of [`chart`] — seven, so the ends are 0 and
/// 1 exactly and the middle step lands on 0.5.
pub const CHART_STEPS: usize = 7;
/// Rows: dielectric, then conductor. Two, because the interesting thing about
/// metallic is not a ramp — it is that a metal has no diffuse lobe at all, and
/// that reads as a difference between two rows rather than as a gradient.
pub const CHART_ROWS: usize = 2;
/// Boxes the chart deals — what `bootstrap` adds to [`ROOM`].
pub const CHART_BOXES: usize = CHART_STEPS * CHART_ROWS;
/// Half-extent of one chart box. The waist crates' size on purpose: the same
/// 0.70 cube, so it reads as an object in the room rather than as a diagram
/// laid over one, and it can be jumped onto like any other.
pub const CHART_HALF: f32 = 0.35;
/// Centre of the first box — dielectric row, roughest step.
///
/// The room's one large clear span: east of the mezzanine (which reaches
/// x = -2.5), north of the floating slabs, and with the far wall five metres
/// behind it so the chart is read against sky and flat wall rather than against
/// the furniture. The `chart` golden scene frames it from x = 5.5, z = -3.0,
/// which is the standoff seven boxes at this pitch need to fit a 16:9 frame.
pub const CHART_ORIGIN: sim::DVec3 = sim::DVec3::new(2.65, 0.35, -8.0);
/// Between columns and between the two rows. Wider than a box, so the gap
/// between two samples is floor and not a shared edge.
pub const CHART_PITCH: f64 = 0.95;
/// One base colour for every box in it, and that is the point: the only things
/// that differ across the chart are the two knobs, so anything else the eye
/// sees is the lighting answering them.
pub const CHART_INK: u32 = 0x00b4_b4b4;

/// The material chart on the floor (§6 M24): `(centre, smoothness, metallic)`
/// for every box in it, dielectric row first, roughest step first.
///
/// A function rather than a table because it is arithmetic — a table of 14
/// hand-written triples is 14 chances to mistype a step, and the thing being
/// demonstrated is precisely that the steps are even.
pub fn chart() -> impl Iterator<Item = (sim::DVec3, f32, f32)> {
    (0..CHART_ROWS).flat_map(|row| {
        (0..CHART_STEPS).map(move |step| {
            let at = sim::DVec3::new(
                CHART_ORIGIN.x + step as f64 * CHART_PITCH,
                CHART_ORIGIN.y,
                CHART_ORIGIN.z - row as f64 * CHART_PITCH,
            );
            // Perceptual and even: `Renderable::smoothness` is squared into
            // GGX's alpha by the shader, so an even step here is what looks
            // like an even step.
            let smoothness = step as f32 / (CHART_STEPS - 1) as f32;
            (at, smoothness, row as f32)
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
/// How many cue entities `bootstrap` deals.
pub const CUES: usize = 2;

// ---------------------------------------------------------------- hud -------

/// [`Hud::line`]: ground speed.
pub const HUD_SPEED: u32 = 0;
/// Grounded or airborne — the one piece of controller state a tuner watches.
pub const HUD_STATE: u32 = 1;
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
];

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

    for (position, half_extent, color) in ROOM {
        let slab = world.spawn_with(Solid { _pad: 0 });
        world.put(slab, Renderable::boxed(*position, *half_extent, *color));
    }

    // The material chart, solid like everything else in the room: it is
    // furniture that happens to be a diagram, so it collides, it casts, and it
    // can be stood on.
    for (at, smoothness, metallic) in chart() {
        let cube = world.spawn_with(Solid { _pad: 0 });
        world.put(
            cube,
            Renderable::boxed(at, sim::Vec3::splat(CHART_HALF), CHART_INK)
                .surfaced(smoothness, metallic),
        );
    }

    // The environment the chart is read against (§6 M24). Without one the metal
    // row is black — a conductor has no diffuse lobe, so what it shows is
    // whatever is around it and nothing else.
    world.spawn_with(Sky::daylight(SKY_INTENSITY));

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
    for (kind, sound) in [
        (CUE_JUMP, chirp(wave::TRIANGLE, 180.0, 260.0, 90, 0.10)),
        (CUE_LAND, chirp(wave::TRIANGLE, 150.0, 110.0, 60, 0.20)),
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

    for (line, text) in [(HUD_SPEED, "SPEED 0.0"), (HUD_STATE, "GROUND")] {
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

    world.visit::<(&Hud, &mut Widget)>(|_, (hud, widget)| match hud.line {
        HUD_SPEED => widget.set_text(speed.as_str()),
        HUD_STATE => widget.set_text(if grounded { "GROUND" } else { "AIR" }),
        _ => {}
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

/// Put the body back at [`START`]. Not a wipe: the room is `bootstrap`'s and
/// nothing here could deal it back — and neither the session nor the settings
/// are the body's to reset.
fn respawn(world: &mut GameWorld) {
    world.visit::<&mut Walker>(|_, walker| *walker = at_start());
    world.log(log_level::INFO, "shooter: restarted");
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
        Walker, Solid, Cue, Hud, Menu, Session, Renderable, Light, Sky, Eye, Widget, Sound, Prefs
    ],
    actions: ["jump", "restart", "pause", "ui_click", "ui_focus"],
    axes: ["move_right", "move_forward", "aim_x", "aim_y", "ui_x", "ui_y"],
    systems: [restart, bootstrap, menu, aim, walk, present],
}
