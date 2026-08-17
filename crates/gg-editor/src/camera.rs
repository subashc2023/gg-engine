//! The editor's camera (§6 M15.2 items 2 and 4): host state, flown by verbs the
//! host appended.
//!
//! Not an entity, for the same reason the panels are not `Widget`s — it is in no
//! archetype, in no save and in no canonical hash, so flying it moves nothing a
//! determinism gate can see. That is the whole point: an editor camera exists to
//! look at a scene from somewhere the game never put one.
//!
//! Three decisions a reader should not have to infer:
//!
//! - **Latched, never defaulted.** The first stop takes the game's own eye
//!   ([`Eye::of`], §6 M15.2 item 1), so the viewport does not jump at the click,
//!   and it is never re-latched — a play/stop pair would otherwise throw away
//!   exactly the viewpoint the operator flew to. It survives a reload for free,
//!   since a reload rebuilds the world and this is not in it.
//! - **Stopped only.** Play and pause render from the game's eye, because that is
//!   what play is *for*. It is also what makes shared keys safe: demo 05 binds
//!   `W` as well, and while the scene is stopped its systems are not running to
//!   read it.
//! - **Look is a raw device delta, not the pointer's.** A steered pointer stops
//!   at the window edge, so a drag that reached it would stop turning while the
//!   hand kept moving; a device delta has no edge to reach. Needs its own axis
//!   pair alongside whatever the game declares, which `gg_abi::MAX_AXES` (§4.7)
//!   is wide enough to hold.
//!
//! # Flying it (§6 M63)
//!
//! The three things a fly camera needs that a drag-to-turn one does not, and
//! the reason each is here rather than in the shell:
//!
//! - **The pointer is captured for the length of the drag** — [`Camera::flying`]
//!   is what a windowed host holds and hides the OS arrow on. Decided here
//!   because it is decided off the *recorded* frame: a replayed session then
//!   captures and releases on the ticks the operator did, with no window
//!   anywhere to capture into, and a headless gate can grade it.
//! - **The wheel is the throttle while captured** ([`SPEED`]), and the dolly it
//!   always was otherwise. One wheel, two subjects, arbitrated by the button
//!   that is already down — and the viewport prints the number, because a knob
//!   that moves invisibly is a knob nobody finds.
//! - **The turn a frame owes the hand** ([`Camera::look`]) is this camera's half
//!   of §6 M56. A tick is 16.6 ms and a panel is 4.2, so a view moved only by
//!   ticks moves in visible steps on the desk this was written for; the shell
//!   adds the unspent counts at frame time exactly as it does for a game's eye.
//!   `gg-tools pace` measures both halves of why that is the right answer and
//!   not an interpolation: latching costs no latency, and interpolating costs a
//!   tick of it.
//!
//! What is *not* here: a toggle. The only key a host would spare for leaving a
//! captured pointer is Escape, and Escape already quits (`play.rs`), so a mode
//! would be a way in whose way out is closing the window. A hold cannot strand
//! anyone, which is the whole of the argument.
//!
//! # A flat camera is not a fly camera (§6 M20 item 10)
//!
//! Under an orthographic eye the six fly directions stop describing anything an
//! operator wants. Forward and back move along the view axis, which a parallel
//! projection renders *identically* — the two keys did nothing at all — and a
//! look drag tilts a view whose whole value is being square to the level, with
//! no way back by hand, because getting yaw and pitch to exactly zero again by
//! dragging is not a gesture. So while `Eye::ortho` is nonzero:
//!
//! - the look drag is **refused**, and the flat view stays flat;
//! - forward and back **zoom**, which is what a distance means to a projection
//!   that has no perspective — the same act the wheel does, on the keys that
//!   were previously inert;
//! - lateral movement and the pan scale **with the zoom**, so a step is the
//!   same fraction of the window at any framing rather than a screenful when
//!   zoomed in and a pixel when zoomed out.
//!
//! [`verb::FRAME`] is the recovery hatch and the reason none of the above needs
//! a mode: it puts the selection in the middle at a size that fits it, and under
//! an orthographic eye it also puts the camera back to level — so a session that
//! got lost is one keypress from a picture, whatever a previous build's tilt or
//! a stray drag left behind.

use crate::host::verb;
use gg_core::cvar::CVar;
use gg_ecs::boundary::{Eye, Look, Renderable};
use gg_input::{ActionId, AxisId, Input, MAX_ACTIONS, MAX_AXES};
use gg_math::sim;

/// Metres per second held, and the wheel's own subject while the pointer is
/// captured (§6 M63). 15 m/s crosses demo 05's scene in a second rather than
/// the room it is in — which was `MOVE_PER_TICK`'s 0.25 at 60 Hz, so the
/// default is the constant it replaces and no session moves differently for
/// this existing.
///
/// A CVar because a fly speed is the one navigation number whose right value is
/// the *scene's* rather than the editor's: 15 m/s is a stroll in Sponza and a
/// blur in a Tetris well, and an operator who cannot change it flies one of the
/// two badly. `recorded` (§6 M40) for `r.fov`'s reason — where the camera is
/// decides what a recorded click picks.
pub(crate) static SPEED: CVar =
    CVar::new_float("d.editor_speed", 15.0, "editor fly speed, metres a second").recorded();

/// What the wheel multiplies [`SPEED`] by, per notch, while the pointer is
/// captured. Geometric for [`ZOOM_PER_NOTCH`]'s reason: a notch is then the same
/// *proportion* at 0.2 m/s and at 200.
const SPEED_PER_NOTCH: f64 = 1.15;

/// What [`SPEED`] may reach. The floor is an inspection crawl and the ceiling
/// crosses Sponza in a third of a second; past either the wheel is a way to lose
/// the scene rather than a way to reach it.
const SPEED_RANGE: (f64, f64) = (0.05, 500.0);

/// Multiplies [`LOOK_PER_UNIT`]. A mouse's counts per inch is a property of the
/// desk and not of this tree, so the rate a drag turns at cannot have one right
/// value here — 400 CPI and 3200 CPI are the same hand and eight times the
/// counts. `recorded` for [`SPEED`]'s reason.
///
/// One is [`LOOK_PER_UNIT`] itself and so an ordinary shooter's rate; 1.5 is
/// demo 12's own default, and past about 1.6 the step this camera turns by
/// stops being sub-pixel — which is the ceiling worth knowing about and the
/// reason it is not clamped to one.
pub(crate) static SENSITIVITY: CVar = CVar::new_float(
    "d.editor_sensitivity",
    1.0,
    "editor look sensitivity, a multiplier",
)
.recorded();

/// Push the mouse away to look up. Half of flight simulation has wanted this
/// since flight simulation existed, and the half that does not is the default.
/// `recorded` for [`SPEED`]'s reason.
pub(crate) static INVERT: CVar = CVar::new_bool(
    "d.editor_invert_y",
    false,
    "editor look inverts the pitch axis",
)
.recorded();

/// The orthographic half-height a lateral step is [`SPEED`]'s own at. Above
/// it a step is proportionally longer and below it shorter, so the key covers
/// the same fraction of the window at every zoom — 4.5 m is demo 11's own
/// framing (`CAMERA_HALF_HEIGHT`), which makes the flat default *exactly* the
/// perspective one rather than near it.
const FLAT_REFERENCE: f64 = 4.5;

/// What one wheel notch multiplies the flat view's half-height by, zooming out.
/// Geometric rather than additive: a notch is then the same fraction of the
/// window at every framing, where a fixed number of metres crawls when zoomed
/// in and jumps a level when zoomed out.
const ZOOM_PER_NOTCH: f64 = 1.1;

/// [`ZOOM_PER_NOTCH`] for a held key. Far gentler, because the wheel is one act
/// and this is sixty a second.
const ZOOM_PER_TICK: f64 = 1.012;

/// What a zoom may reach, metres of half-height. The lower bound is a hand's
/// breadth of world and the upper one a level seen whole; both exist because a
/// zoom past either is a blank window with no gesture that gets out of it.
const ZOOM_RANGE: (f64, f64) = (0.05, 4096.0);

/// How much taller than the thing it frames [`verb::FRAME`] leaves the view —
/// enough margin to see what the selection is standing on.
const FRAME_MARGIN: f64 = 2.5;

/// Metres a perspective dolly covers per wheel notch. Unlike the flat zoom this
/// really is a distance, so it is one.
const DOLLY_PER_NOTCH: f64 = 1.0;

/// The depth a *perspective* pan tracks the hand at with nothing selected.
/// A perspective pan can only be exact at one distance; this is the one it
/// picks, and roughly a room's width because that is where a scene being
/// authored usually is.
pub(crate) const PAN_DEPTH: f64 = 10.0;

/// Metres [`verb::FRAME`] leaves between a flat camera and the near face of
/// what it framed. Nothing to do with framing — a parallel projection is the
/// same picture at any distance — and everything to do with staying between
/// `r.near` and `r.ortho_far`'s 500 m.
const FLAT_STANDOFF: f64 = 16.0;

/// What the editor knows about the picture that the camera does not: the pane
/// it is drawn in, and what the operator has selected.
///
/// Passed in rather than reached for, because every one of these already has an
/// owner — the viewport rectangle is the dock's, the field of view is a CVar
/// read per tick (`panels::lens`), and the selection is the editor's. A camera
/// that fetched them would be a second reader of three things that move.
#[derive(Clone, Copy, Default)]
pub(crate) struct Nav {
    /// World metres one unit of *device* motion covers across the viewport, at
    /// the framing the last tick was drawn with. What makes a pan track the
    /// hand instead of moving by an arbitrary rate.
    pub(crate) metres_per_unit: f64,
    /// The viewport's width over its height — what decides whether framing a
    /// wide selection is bounded by its width or its height.
    pub(crate) aspect: f64,
    /// `tan(fov_y / 2)`, the perspective lens's half-height at unit depth — how
    /// far back [`verb::FRAME`] has to stand to fit something. Unread under a
    /// flat eye, exactly as the renderer leaves the field of view unread there.
    pub(crate) half_fov_tan: f64,
    /// The selection's box, for [`verb::FRAME`]. `None` frames nothing and the
    /// key is inert, which is the honest answer to "frame what?".
    pub(crate) target: Option<Renderable>,
    /// Wheel notches this tick, and zero on all but a few of them.
    pub(crate) wheel: i32,
}

/// Radians of turn one raw device count is worth (§6 M65).
///
/// 0.0005 rad is 0.0286 degrees, which puts a full turn at 12 566 counts — 20 cm
/// of desk at 1600 DPI — and is demo 12's own unit, restated rather than
/// imported because a host crate may not depend on a demo. What matters is not
/// the speed but the **step**: across `r.fov`'s 88-degree horizontal window
/// 1920 pixels wide, one count moves the picture 0.6 of a pixel, so the
/// smallest motion a mouse can report is invisible.
///
/// It was `0.005` from §6 M15.2 to M65, and that is *ten times* the value demo
/// 12 rejected at §6 M37 for exactly this reason: at 0.005 one count turned the
/// view 0.29 degrees, which is **six pixels**, so the camera visibly stepped
/// between two counts however slowly the hand moved. The number came from
/// "a 600-unit sweep turns three radians", which was true of the *cursor* drag
/// this camera had until §6 M15.2 gave it raw device motion — 600 screen pixels
/// is a gesture across the window, 600 counts at 1600 DPI is a centimetre of
/// desk. The source changed and the constant did not, which is the whole of it.
///
/// `gg-tools pace --editor` is what reads this: its stall/lurch columns are
/// blind to it by construction — they are fractions of what a frame *owed*, so
/// the rate cancels — and its quantum table is the column that is not.
const LOOK_PER_UNIT: f32 = 0.0005;

/// Just under a right angle: at exactly one the forward and world-up axes are
/// parallel and the basis below degenerates.
const PITCH_LIMIT: f32 = 1.5533;

/// The editor's camera. `Default` is unlatched — see the module docs.
#[derive(Default)]
pub(crate) struct Camera {
    /// `None` until the first stop, and `Some` for the rest of the session.
    eye: Option<Eye>,
    /// Where [`Camera::eye`] was at the end of the *previous* tick, so a host
    /// can blend the pair (§6 M63). `None` on the tick the camera latched: there
    /// is no previous, and blending against a default would swing the picture in
    /// from the origin over one tick.
    previous: Option<Eye>,
    /// The pointer is the camera's this tick: a drag is turning or sliding it,
    /// so the OS arrow should be held and hidden and the wheel means speed
    /// rather than distance (§6 M63).
    ///
    /// Derived from the *recorded* frame like every other decision here, which
    /// is what keeps a replayed session capturing and releasing on the ticks the
    /// operator did — with no window anywhere to capture into.
    flying: bool,
    /// [`flying`](Camera::flying) by the look button specifically, which is the
    /// only one of the two whose motion a late latch may spend: a pan slides the
    /// eye and the latch turns it.
    looking: bool,
    /// The scene is stopped — whether a host should be rendering from this.
    live: bool,
    /// A drag has turned it, and a key has moved it. Separately, because one
    /// line for both is a line the *other* gesture can satisfy: a session gate
    /// reading it could not tell a look that reached no axis from one that did
    /// (§6 M15.2 item 4). Each logs once, for §5.6c's reason.
    turned: bool,
    moved: bool,
    /// The same, for the three gestures §6 M20 item 10 added.
    panned: bool,
    zoomed: bool,
    /// And for §6 M63's two, on the same one-line-a-session rule.
    throttled: bool,
    captured: bool,
    /// [`verb::FRAME`] last tick, so the key acts on its press edge. Held, it
    /// would recompute the identical answer sixty times a second and log it.
    framing: bool,
}

impl Camera {
    /// One tick.
    pub(crate) fn fly(&mut self, world: &gg_ecs::World, frame: &crate::Frame, nav: Nav) {
        self.live = matches!(frame.play, crate::Play::Stopped);
        // Both cleared before the early returns below, not after: a scene that
        // starts playing while the button is down must hand the pointer back,
        // and a stale `true` here is an arrow that never comes back.
        (self.flying, self.looking) = (false, false);
        // The tick this eye ends is the tick the next one blends from. Taken
        // before the flight rather than after, so the pair a host reads is two
        // *different* ticks even on the frames it reads them twice.
        self.previous = self.eye;
        // A host that routes no input at all (a golden render) never latches,
        // which is what keeps a reference image the game's own view.
        let Some(input) = frame.input.filter(|_| self.live) else {
            return;
        };
        if self.eye.is_none() {
            // `unwrap_or` and not a `?`: one read cannot alias, and a camera
            // that refused to exist would leave the operator with no viewport
            // rather than with a wrong one.
            self.eye = Some(Eye::of(world).unwrap_or(Eye::ORIGIN));
            tracing::info!(tick = frame.tick, "editor: camera taken");
        }
        let Some(eye) = self.eye.as_mut() else { return };
        let held = |name| id(input, name).is_some_and(|action| input.pressed(action));
        // A flat eye is a different instrument, not the same one turned sideways
        // — see the module docs for which gestures it keeps.
        let flat = eye.ortho > 0.0;

        // This tick's raw device motion, read once because the look and the pan
        // are the same gesture on two buttons. No anchor and no previous value:
        // a device axis is already this tick's delta, so a press moves nothing
        // on its own and there is no state to reset on release. Integer, so a
        // replayed drag lands where the recorded one did — the same reason the
        // recorder quantizes at all (§4.7).
        let axes = input.frame().axes;
        let delta = |name| axis_id(input, name).map_or(0, |a| axes[a.index()]);
        let (dx, dy) = (delta(verb::LOOK_X), delta(verb::LOOK_Y));
        let dragging = (dx, dy) != (0, 0);

        // The pointer is the camera's for as long as a drag lasts and not a
        // moment longer (§6 M63). Held rather than toggled, and that is the
        // whole argument: a mode has to be left, the only key a host will spare
        // for leaving one is Escape, and Escape already quits — so a toggle
        // would be a way to capture the pointer whose way out is closing the
        // window. A hold cannot strand anyone.
        //
        // The look is refused under a flat eye (see the module docs) and there
        // is nothing to capture the pointer *for* there; the pan is taken under
        // both, because a pan that reaches the window edge stops for the same
        // reason a drag did before §6 M15.2 gave it device motion.
        self.looking = !flat && held(verb::LOOK);
        self.flying = self.looking || held(verb::PAN);

        // Look first: the move below is along the basis this leaves, so a drag
        // that turns and a key that pushes compose within one tick.
        let turned = self.looking && dragging && {
            let (yaw_rate, pitch_rate) = look_rates();
            eye.yaw -= dx as f32 * yaw_rate / gg_input::AXIS_SCALE as f32;
            eye.pitch = (eye.pitch - dy as f32 * pitch_rate / gg_input::AXIS_SCALE as f32)
                .clamp(-PITCH_LIMIT, PITCH_LIMIT);
            true
        };

        // The renderer's own basis, built once for this, the pan and the pick
        // ray (`crate::pick::basis`) — and in `gg_math::sim` rather than `std`,
        // since host state or not, a camera whose angles came out of a libm the
        // two hosts disagree about would put a golden scene's viewpoint on the
        // driver (§1.4, §3). The lift below is along **world** up; `up` here is
        // the camera's, because a pan slides across the picture and the picture
        // is what the camera's own axes describe.
        let (right, up, forward) = crate::pick::basis(eye.yaw, eye.pitch);

        // The pan. Against the hand on both axes: what the operator is dragging
        // is the *scene*, so the camera goes the other way and whatever was
        // under the pointer stays under it. `metres_per_unit` is the framing the
        // last tick was drawn at, which is what makes that true rather than
        // approximately true.
        let scale = f64::from(gg_input::AXIS_SCALE);
        let panned = held(verb::PAN) && dragging && {
            let (px, py) = (f64::from(dx) / scale, f64::from(dy) / scale);
            eye.position += (up * py - right * px) * nav.metres_per_unit;
            true
        };

        let axis = |plus, minus| f64::from(u8::from(held(plus))) - f64::from(u8::from(held(minus)));
        let ahead = axis(verb::FORWARD, verb::BACK);
        // While the drag has the pointer the wheel is the **throttle** (§6 M63),
        // which is the one rebinding of it an operator flying with the other
        // hand actually wants: the keys are already moving, and what is wrong is
        // their rate. A notch spent dollying mid-drag moves the eye somewhere
        // the keys were about to take it anyway, so nothing is lost by the swap
        // — and the legend prints the number, which is what makes a wheel that
        // does two things discoverable rather than surprising.
        let throttled = self.flying && nav.wheel != 0 && {
            SPEED.set_float(
                notched(SPEED.float(), nav.wheel, SPEED_PER_NOTCH)
                    .clamp(SPEED_RANGE.0, SPEED_RANGE.1),
            );
            true
        };
        // Zoom: the wheel otherwise, and forward/back as well under a flat eye,
        // where they otherwise move along the one direction a parallel
        // projection cannot show. A perspective eye dollies on the notch instead
        // — there, distance *is* framing and forward already means it.
        let wheel = match throttled {
            true => 0,
            false => nav.wheel,
        };
        let zoomed = match flat {
            true => {
                // A notch away from the operator zooms *in*, so the exponent is
                // the wheel's negation — the one place the two geometric knobs
                // this tick can move disagree about which way a notch points.
                let mut want = notched(f64::from(eye.ortho), -wheel, ZOOM_PER_NOTCH);
                if ahead != 0.0 {
                    want = notched(want, -ahead as i32, ZOOM_PER_TICK);
                }
                let want = want.clamp(ZOOM_RANGE.0, ZOOM_RANGE.1);
                let moved = want != f64::from(eye.ortho);
                eye.ortho = want as f32;
                moved
            }
            false => {
                eye.position += forward * (f64::from(wheel) * DOLLY_PER_NOTCH);
                wheel != 0
            }
        };

        // Lateral movement, scaled to the zoom under a flat eye so a key covers
        // the same fraction of the window at every framing. Forward and back are
        // absent there: they are the zoom above.
        let step = per_tick(frame.hz)
            * match flat {
                true => f64::from(eye.ortho) / FLAT_REFERENCE,
                false => 1.0,
            };
        let mut motion =
            right * axis(verb::RIGHT, verb::LEFT) + sim::DVec3::Y * axis(verb::UP, verb::DOWN);
        if !flat {
            motion += forward * ahead;
        }
        eye.position += motion * step;

        // The way back. On the press edge, because held it would recompute the
        // same answer every tick — and *last*, so a frame is not then undone by
        // a key held from the same gesture that asked for it.
        let asked = held(verb::FRAME);
        let framed = asked && !self.framing;
        self.framing = asked;
        if framed {
            // Measured in the basis the framing will *leave*, not the one it
            // found: a flat eye is levelled by this act, so bounds taken through
            // the old tilt would be the extents of a box nobody is going to see.
            let basis = match flat {
                true => crate::pick::basis(0.0, 0.0),
                false => (right, up, forward),
            };
            if let Some((centre, half)) = bounds(world, nav.target, basis) {
                fit(eye, nav, flat, centre, half, basis.2);
                tracing::info!(tick = frame.tick, "editor: camera framed");
            }
        }

        // Once each, and only once the gesture actually did something — sixty
        // lines a second would bury every other thing the session says (§5.6c).
        for (seen, did, what) in [
            (&mut self.turned, turned, "editor: camera turned"),
            (
                &mut self.moved,
                motion != sim::DVec3::ZERO,
                "editor: camera flown",
            ),
            (&mut self.panned, panned, "editor: camera panned"),
            (&mut self.zoomed, zoomed, "editor: camera zoomed"),
            (&mut self.throttled, throttled, "editor: camera throttled"),
            (&mut self.captured, self.flying, "editor: pointer captured"),
        ] {
            if !*seen && did {
                *seen = true;
                tracing::info!(tick = frame.tick, "{what}");
            }
        }
    }

    /// The eye a host should render from, given the game's own.
    ///
    /// The game's whenever the scene is not stopped, so play shows what a player
    /// would see — and before the first stop, since there is nothing latched.
    pub(crate) fn eye(&self, game: Eye) -> Eye {
        match self.live {
            true => self.eye.unwrap_or(game),
            false => game,
        }
    }

    /// The previous tick's eye and this one's, for a host that blends them
    /// (§6 M63, §4.1).
    ///
    /// Both are `game` whenever this camera is not the one being rendered from,
    /// which makes the blend the identity there rather than making the caller
    /// ask twice. The pair is also equal on the tick the camera latched — see
    /// [`Camera::previous`].
    pub(crate) fn eyes(&self, game: Eye) -> (Eye, Eye) {
        let current = self.eye(game);
        match self.live {
            true => (self.previous.unwrap_or(current), current),
            false => (game, game),
        }
    }

    /// The pointer is this camera's — held and hidden by a host that has one.
    pub(crate) fn flying(&self) -> bool {
        self.flying
    }

    /// What a *frame* may add to this camera's angles for the turn no tick has
    /// spent yet (§6 M56), or `None` on every tick the drag is not turning it.
    ///
    /// The editor's own [`Look`], built here rather than in the shell for the
    /// reason `gg_extract::Latch::of` states about a game's: the sign, the rate
    /// and the clamp are this file's three decisions and one restatement of
    /// them elsewhere is one too many. `None` while the button is
    /// up is not an optimization — raw device motion arrives whatever the
    /// pointer is doing, so a latch applied unconditionally would swing the
    /// picture whenever the operator reached for a menu, at frame rate, while
    /// the tick underneath it stood still.
    pub(crate) fn look(&self, input: &Input) -> Option<Look> {
        let (yaw, pitch) = (
            axis_id(input, verb::LOOK_X)?.index() as u32,
            axis_id(input, verb::LOOK_Y)?.index() as u32,
        );
        let (yaw_rate, pitch_rate) = look_rates();
        self.looking.then_some(Look {
            // Negated for `fly`'s reason and this file's: the tick above spends
            // `yaw -= dx * rate`, and a latch that added would turn the picture
            // one way and the tick the other.
            yaw_rate: -yaw_rate,
            pitch_rate: -pitch_rate,
            pitch_limit: PITCH_LIMIT,
            yaw_axis: yaw,
            pitch_axis: pitch,
            reserved: 0,
        })
    }
}

/// Radians per raw device count for yaw and for pitch, sensitivity and the
/// invert applied. Two numbers rather than one because [`INVERT`] is a sign on
/// exactly one of them.
fn look_rates() -> (f32, f32) {
    let rate = LOOK_PER_UNIT * SENSITIVITY.float() as f32;
    (
        rate,
        match INVERT.bool() {
            true => -rate,
            false => rate,
        },
    )
}

/// Metres a held key covers in one tick at [`SPEED`], on a sim running at `hz`.
///
/// The knob is metres a *second* because that is the unit an operator can
/// picture, and this is the one place the tick rate turns it into the unit the
/// flight is integrated in. A zero `hz` cannot happen and is defended anyway:
/// the alternative is an infinite step, which is a camera at NaN and a viewport
/// that never comes back.
pub(crate) fn per_tick(hz: u32) -> f64 {
    SPEED.float() / f64::from(hz.max(1))
}

/// `value` multiplied by `per` once for each notch **away** from the operator,
/// and divided once for each notch toward.
///
/// A loop and not `powi`: this file computes in `gg_math::sim` wherever an angle
/// is involved for §1.4's reason, and a repeated multiply is the one spelling
/// of an integer power that needs no library to agree about.
fn notched(value: f64, notches: i32, per: f64) -> f64 {
    let mut out = value;
    for _ in 0..notches.abs() {
        out *= match notches > 0 {
            true => per,
            false => 1.0 / per,
        };
    }
    out
}

/// What [`verb::FRAME`] should fill the view with: the selection's box, or —
/// with nothing selected — every [`Renderable`] there is, which is the case an
/// operator who has lost the scene entirely is in.
///
/// Returned as a centre in world space and half-extents along `basis`'s three
/// axes, because that is the shape both projections need and neither one wants
/// an axis-aligned world box: a camera that has been turned frames what it can
/// see, not what +X happens to bound.
fn bounds(
    world: &gg_ecs::World,
    selected: Option<Renderable>,
    basis: (sim::DVec3, sim::DVec3, sim::DVec3),
) -> Option<(sim::DVec3, [f64; 3])> {
    let (right, up, forward) = basis;
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    let mut take = |box_: &Renderable| {
        // The corners and not the centre: a wide platform framed by its centre
        // would sit in a view sized for a point.
        for corner in crate::pick::corners(box_) {
            for (i, axis) in [right, up, forward].into_iter().enumerate() {
                let along = corner.dot(axis);
                lo[i] = lo[i].min(along);
                hi[i] = hi[i].max(along);
            }
        }
    };
    match selected {
        Some(box_) => take(&box_),
        None => {
            let query = gg_ecs::Query::<&Renderable>::new().ok()?;
            world.each_ref(&query, |_, box_: &Renderable| take(box_));
        }
    }
    // A world with nothing in it to look at leaves the camera alone, rather
    // than framing an empty box at the origin.
    if lo[0] > hi[0] {
        return None;
    }
    let mid = |i: usize| (lo[i] + hi[i]) * 0.5;
    // Orthonormal, so the three projections reconstruct the world point.
    let centre = right * mid(0) + up * mid(1) + forward * mid(2);
    Some((
        centre,
        [
            (hi[0] - lo[0]) * 0.5,
            (hi[1] - lo[1]) * 0.5,
            (hi[2] - lo[2]) * 0.5,
        ],
    ))
}

/// Put `centre` in the middle of the view at a size that fits `half`.
///
/// The two projections need opposite things and that is the whole of the split:
/// a flat eye keeps its distance (a parallel projection frames nothing with it)
/// and changes its *extent*, while a perspective one keeps its extent and
/// changes its distance.
fn fit(
    eye: &mut Eye,
    nav: Nav,
    flat: bool,
    centre: sim::DVec3,
    half: [f64; 3],
    forward: sim::DVec3,
) {
    // Whichever of the two the pane runs out of first — a wide selection in a
    // narrow window is bounded by its width, and the aspect is what says so.
    let radius = half[1].max(half[0] / nav.aspect.max(f64::EPSILON)) * FRAME_MARGIN;
    match flat {
        true => {
            // Square to the level again: the one act that puts a tilted flat
            // camera back, since dragging yaw and pitch to exactly zero is not
            // a gesture a hand can make.
            eye.yaw = 0.0;
            eye.pitch = 0.0;
            eye.ortho = radius.clamp(ZOOM_RANGE.0, ZOOM_RANGE.1) as f32;
            // Far enough out that the whole slab is past the near plane, and
            // well inside `r.ortho_far`'s 500 m whatever the level's depth.
            eye.position = centre - forward * (half[2] + FLAT_STANDOFF);
        }
        false => {
            // Plus the selection's own depth, or a long box would be framed
            // from inside its near end.
            eye.position =
                centre - forward * (radius / nav.half_fov_tan.max(f64::EPSILON) + half[2]);
        }
    }
}

/// A verb's id in *this* build's action map.
///
/// Resolved per tick rather than cached, and that is correctness rather than
/// laziness: the map is re-parsed at every reload (§4.2.2) and an id cached
/// across one would name whatever verb an edit moved into that slot.
pub(crate) fn id(input: &Input, name: &str) -> Option<ActionId> {
    input
        .map()
        .action_names()
        .iter()
        .position(|n| n == name)
        .filter(|i| *i < MAX_ACTIONS)
        .map(ActionId::new)
}

/// [`id`] for an axis, and resolved per tick for the same reason. Named apart
/// from the local `axis` closure in [`Camera::fly`], which is a digital pair.
fn axis_id(input: &Input, name: &str) -> Option<AxisId> {
    input
        .map()
        .axis_names()
        .iter()
        .position(|n| n == name)
        .filter(|i| *i < MAX_AXES)
        .map(AxisId::new)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::{Frame, Play};
    use gg_ecs::World;
    use gg_input::{ActionMap, InputFrame};

    const AT: sim::DVec3 = sim::DVec3::new(4.0, 1.0, -30.0);

    /// What a held key covers in one tick at the shipped [`SPEED`] on the
    /// shipped pace — the constant this file held until §6 M63 made it a knob,
    /// so every distance below still reads against the number it always did.
    const PER_TICK: f64 = 0.25;

    /// A 16:9 pane a third of a metre per device unit, with nothing selected —
    /// what the editor hands the camera on an ordinary tick, spelled once here
    /// so a test that cares about one of these says which.
    fn nav() -> Nav {
        Nav {
            metres_per_unit: 1.0 / 32.0,
            aspect: 16.0 / 9.0,
            half_fov_tan: 0.5463,
            target: None,
            wheel: 0,
        }
    }

    /// Demo 11's framing in shape: an orthographic eye out of the playfield
    /// plane, and two slabs far enough apart that framing *the scene* and
    /// framing *one of them* cannot accidentally agree.
    const SLABS: [(sim::DVec3, sim::Vec3); 2] = [
        (
            sim::DVec3::new(-9.0, 0.0, 0.0),
            sim::Vec3::new(3.0, 0.5, 0.5),
        ),
        (
            sim::DVec3::new(21.0, 6.0, 0.0),
            sim::Vec3::new(1.0, 0.5, 0.5),
        ),
    ];

    fn flat_world() -> World {
        let mut world = World::new();
        world.register::<Eye>().unwrap();
        let eye = world.spawn();
        world
            .insert(eye, Eye::flat(sim::DVec3::new(0.0, 0.0, 14.0), 4.5))
            .unwrap();
        for (at, half) in SLABS {
            let slab = world.spawn();
            world
                .insert(slab, Renderable::boxed(at, half, 0x0080_c0ff))
                .unwrap();
        }
        world
    }

    /// A world holding one eye, plus a second entity so `Eye::of`'s rule is
    /// being exercised rather than a one-entity accident.
    fn world() -> World {
        let mut world = World::new();
        world.register::<Eye>().unwrap();
        let eye = world.spawn();
        world.insert(eye, Eye::at(AT, 0.0, 0.0)).unwrap();
        world.spawn();
        world
    }

    /// The map a shell builds with the editor open over a game declaring
    /// nothing — which is what makes the ids below the host's and not ours.
    fn input() -> Input {
        let (verbs, bindings) = crate::host::open(&gg_ecs::boundary::Verbs {
            actions: &[],
            axes: &[],
        });
        let map = ActionMap::parse(&bindings, verbs.actions, verbs.axes).unwrap();
        let mut input = Input::new(map);
        assert!(
            input.push_named("game"),
            "the appended text opens a context"
        );
        input
    }

    fn press(input: &mut Input, name: &str) {
        let action = id(input, name).expect("the host appended it");
        input.tick_from(InputFrame {
            buttons: 1 << action.index(),
            ..InputFrame::default()
        });
    }

    /// `held` with this tick's device motion on the pair the host appended
    /// beside it — the look and the pan are the same gesture on two buttons,
    /// so they are the same helper.
    fn gesture(input: &mut Input, held: &str, motion: (i32, i32)) {
        let action = id(input, held).expect("the host appended it");
        let mut frame = InputFrame {
            buttons: 1 << action.index(),
            ..InputFrame::default()
        };
        for (name, value) in [(verb::LOOK_X, motion.0), (verb::LOOK_Y, motion.1)] {
            let at = axis_id(input, name).expect("and the axis pair with it");
            frame.axes[at.index()] = value;
        }
        input.tick_from(frame);
    }

    fn drag(input: &mut Input, motion: (i32, i32)) {
        gesture(input, verb::LOOK, motion);
    }

    fn frame<'a>(play: Play, input: Option<&'a Input>) -> Frame<'a> {
        Frame {
            extent: gg_ecs::boundary::CANVAS,
            dpi: 1.0,
            tick: 7,
            hz: 60,
            play,
            input,
            typed: "",
            passes: &[],
            memory: gg_rhi::MemoryUse::default(),
            save_path: "",
            title: "",
            project: Some("test"),
            projects: &[],
            maximized: false,
            reload: None,
            draw_cursor: false,
        }
    }

    /// The click that stops the scene must not also move the picture, which is
    /// the whole reason the camera latches instead of starting somewhere.
    #[test]
    fn the_first_stop_latches_the_game_eye_and_the_viewport_does_not_jump() {
        let (world, input, mut camera) = (world(), input(), Camera::default());
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        assert_eq!(camera.eye(Eye::ORIGIN).position, AT, "latched the game's");
    }

    /// A held verb moves along the eye's own basis, and the world it was
    /// latched from does not move with it — §6 M15.2's Exit, as a hash.
    #[test]
    fn forward_moves_the_camera_and_nothing_the_hash_can_see() {
        let (world, mut input, mut camera) = (world(), input(), Camera::default());
        let before = world.canonical_hash();
        for _ in 0..4 {
            press(&mut input, verb::FORWARD);
            camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        }
        let flown = camera.eye(Eye::ORIGIN).position;
        // Yaw zero looks down -Z, so four ticks forward is exactly that far
        // along it and nothing on the other two axes.
        assert!(
            (flown.z - (AT.z - 4.0 * PER_TICK)).abs() < 1e-12,
            "{flown:?}"
        );
        assert_eq!((flown.x, flown.y), (AT.x, AT.y));
        assert_eq!(
            world.canonical_hash(),
            before,
            "the camera wrote to the world"
        );
    }

    /// Up is world up and not the eye's, so a pitched camera still rises level.
    #[test]
    fn up_is_world_up_however_the_camera_is_pitched() {
        let (world, mut input, mut camera) = (world(), input(), Camera::default());
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        camera.eye.as_mut().unwrap().pitch = -0.9;
        press(&mut input, verb::UP);
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        let flown = camera.eye(Eye::ORIGIN).position;
        assert_eq!(flown.y, AT.y + PER_TICK);
        assert_eq!((flown.x, flown.z), (AT.x, AT.z), "it drifted sideways");
    }

    /// The drag turns by *this tick's* delta, a press carrying no motion turns
    /// nothing, and pitch clamps rather than passing through vertical.
    #[test]
    fn a_look_drag_turns_by_the_device_delta_and_clamps_at_the_poles() {
        let (world, mut input, mut camera) = (world(), input(), Camera::default());
        // Held, with the mouse still: nothing to turn by, and no accumulated
        // wander to snap to either — which is what a device axis buys over the
        // cursor position the anchor used to have to defend against.
        drag(&mut input, (0, 0));
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        assert_eq!(
            camera.eye(Eye::ORIGIN).yaw,
            0.0,
            "the press snapped the view"
        );
        drag(&mut input, (900 * gg_input::AXIS_SCALE, 0));
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        assert!((camera.eye(Eye::ORIGIN).yaw + 900.0 * LOOK_PER_UNIT).abs() < 1e-4);
        // Straight down, well past the limit, in one absurd drag.
        for _ in 0..4 {
            drag(&mut input, (0, 900 * gg_input::AXIS_SCALE));
            camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        }
        assert_eq!(camera.eye(Eye::ORIGIN).pitch, -PITCH_LIMIT);
        // And the axes carry a delta every tick the mouse moves, so the verb is
        // the only thing deciding whether it counts. Read unconditionally, the
        // camera would swing whenever the operator reached for a menu.
        let was = camera.eye(Eye::ORIGIN).yaw;
        let at = axis_id(&input, verb::LOOK_X).expect("the host appended it");
        let mut moved = InputFrame::default();
        moved.axes[at.index()] = 900 * gg_input::AXIS_SCALE;
        input.tick_from(moved);
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        assert_eq!(
            camera.eye(Eye::ORIGIN).yaw,
            was,
            "it turned with nothing held"
        );
    }

    /// §6 M15.2's residual, closed: a drag far longer than the surface keeps
    /// turning for all of it. The cursor this used to difference stops at the
    /// window edge, so the same gesture turned until it got there and no
    /// further; the sum below is six canvas widths.
    #[test]
    fn a_drag_past_the_window_edge_keeps_turning() {
        let (world, mut input, mut camera) = (world(), input(), Camera::default());
        const STEP: i32 = 200;
        const TICKS: i32 = 40;
        for _ in 0..TICKS {
            drag(&mut input, (STEP * gg_input::AXIS_SCALE, 0));
            camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        }
        let yaw = camera.eye(Eye::ORIGIN).yaw;
        let want = -((TICKS * STEP) as f32) * LOOK_PER_UNIT;
        assert!(
            (yaw - want).abs() < 1e-2,
            "turned {yaw} of an expected {want} — a clamp somewhere ate the tail"
        );
        assert!(
            (STEP * TICKS) as f32 > 6.0 * gg_ecs::boundary::CANVAS.0 as f32,
            "the drag no longer outruns the surface, so it proves nothing"
        );
    }

    /// Play renders from the game, and the keys are the game's while it does.
    #[test]
    fn while_playing_the_host_renders_from_the_game_and_the_verbs_do_nothing() {
        let (world, mut input, mut camera) = (world(), input(), Camera::default());
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        for play in [Play::Running, Play::Paused] {
            press(&mut input, verb::FORWARD);
            camera.fly(&world, &frame(play, Some(&input)), nav());
            assert_eq!(
                camera.eye(Eye::ORIGIN),
                Eye::ORIGIN,
                "{play:?} took the camera"
            );
        }
        // And the flight resumes exactly where it was left, rather than
        // re-latching onto whatever the game's eye did meanwhile.
        press(&mut input, verb::FORWARD);
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        let flown = camera.eye(Eye::ORIGIN).position;
        assert!((flown.z - (AT.z - PER_TICK)).abs() < 1e-12, "{flown:?}");
    }

    /// A host with no action map — the golden harness — never latches, so a
    /// reference image stays the game's own view.
    #[test]
    fn a_host_that_routes_no_input_renders_from_the_game() {
        let (world, mut camera) = (world(), Camera::default());
        camera.fly(&world, &frame(Play::Stopped, None), nav());
        assert_eq!(camera.eye(Eye::ORIGIN), Eye::ORIGIN);
    }

    // ------------------------------------------- §6 M20 item 10: flat nav ----

    /// A flat eye is not tiltable, and the drag that would tilt it is refused
    /// rather than absent: the same gesture under a perspective eye turns, so
    /// this measures a rule and not a gesture that never arrived.
    #[test]
    fn a_flat_eye_refuses_the_look_drag_and_a_perspective_one_still_takes_it() {
        // The vacuity guard first, so the refusal below is measured against a
        // gesture already known to arrive somewhere.
        let (turning, mut keys, mut tilted) = (world(), input(), Camera::default());
        drag(&mut keys, (400 * gg_input::AXIS_SCALE, 0));
        tilted.fly(&turning, &frame(Play::Stopped, Some(&keys)), nav());
        assert!(
            tilted.eye(Eye::ORIGIN).yaw != 0.0,
            "the drag reaches no camera at all"
        );

        let (world, mut input, mut camera) = (flat_world(), input(), Camera::default());
        for _ in 0..8 {
            drag(
                &mut input,
                (400 * gg_input::AXIS_SCALE, 250 * gg_input::AXIS_SCALE),
            );
            camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        }
        let flat = camera.eye(Eye::ORIGIN);
        assert_eq!((flat.yaw, flat.pitch), (0.0, 0.0), "the flat view tilted");
    }

    /// The wheel and the forward key both zoom a flat eye, in the same
    /// direction and without moving it: a parallel projection has nothing to
    /// dolly, so the keys that used to translate along the view axis — visibly
    /// doing nothing — are the zoom.
    #[test]
    fn the_wheel_and_forward_zoom_a_flat_eye_without_moving_it() {
        let (world, mut input, mut camera) = (flat_world(), input(), Camera::default());
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        let (was, at) = {
            let eye = camera.eye(Eye::ORIGIN);
            (eye.ortho, eye.position)
        };
        camera.fly(
            &world,
            &frame(Play::Stopped, Some(&input)),
            Nav { wheel: 1, ..nav() },
        );
        let notched = camera.eye(Eye::ORIGIN).ortho;
        assert!(notched < was, "a notch away from the operator zooms in");
        assert_eq!(camera.eye(Eye::ORIGIN).position, at, "the notch moved it");

        press(&mut input, verb::FORWARD);
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        let held = camera.eye(Eye::ORIGIN);
        assert!(held.ortho < notched, "forward zooms in too");
        assert_eq!(held.position, at, "forward moved a flat camera");
        // Gentler than the wheel, or a key would be a notch a tick.
        assert!(notched - held.ortho < was - notched);
    }

    /// A notch is the same *fraction* of the view wherever it is taken from —
    /// the property an additive zoom does not have — and the range is clamped
    /// at both ends, because a view of nothing has no gesture that gets out.
    #[test]
    fn a_zoom_notch_is_geometric_and_clamps_at_both_ends() {
        let (world, input, mut camera) = (flat_world(), input(), Camera::default());
        let out = Nav { wheel: -1, ..nav() };
        let step = |camera: &mut Camera, nav: Nav| {
            let was = f64::from(camera.eye(Eye::ORIGIN).ortho);
            camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav);
            f64::from(camera.eye(Eye::ORIGIN).ortho) / was
        };
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        let near = step(&mut camera, out);
        for _ in 0..24 {
            step(&mut camera, out);
        }
        let far = step(&mut camera, out);
        assert!(
            (near - far).abs() < 1e-3,
            "a notch is {near} of the view here and {far} of it two hundred metres out"
        );
        // f32 storage rounds, so this is the ratio the constant asks for to
        // within the type rather than to the bit.
        assert!((near - ZOOM_PER_NOTCH).abs() < 1e-3, "{near}");

        for _ in 0..400 {
            step(&mut camera, out);
        }
        assert_eq!(f64::from(camera.eye(Eye::ORIGIN).ortho), ZOOM_RANGE.1);
        let inn = Nav { wheel: 1, ..nav() };
        for _ in 0..400 {
            step(&mut camera, inn);
        }
        assert!(
            (f64::from(camera.eye(Eye::ORIGIN).ortho) - ZOOM_RANGE.0).abs() < 1e-6,
            "{}",
            camera.eye(Eye::ORIGIN).ortho
        );
    }

    /// A perspective eye keeps the wheel as a dolly, because there distance
    /// *is* framing — the one thing a flat eye's zoom is standing in for.
    #[test]
    fn the_wheel_dollies_a_perspective_eye_instead() {
        let (world, input, mut camera) = (world(), input(), Camera::default());
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        camera.fly(
            &world,
            &frame(Play::Stopped, Some(&input)),
            Nav { wheel: 1, ..nav() },
        );
        let eye = camera.eye(Eye::ORIGIN);
        // Yaw zero looks down -Z, so a notch away from the operator is exactly
        // that far along it, and nothing on the other two axes.
        assert!((eye.position.z - (AT.z - DOLLY_PER_NOTCH)).abs() < 1e-12);
        assert_eq!((eye.position.x, eye.position.y), (AT.x, AT.y));
        assert_eq!(eye.ortho, 0.0, "a perspective eye gained a projection");
    }

    /// The pan slides the camera **against** the hand, so what was under the
    /// pointer stays under it, and it does so at the scale it was handed rather
    /// than one of its own.
    #[test]
    fn a_middle_drag_pans_against_the_hand_at_the_scale_it_was_given() {
        let (world, mut input, mut camera) = (flat_world(), input(), Camera::default());
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        let at = camera.eye(Eye::ORIGIN).position;
        // 320 device units right and 64 down, at a thirty-secondth of a metre
        // each: ten metres left, two metres up.
        gesture(
            &mut input,
            verb::PAN,
            (320 * gg_input::AXIS_SCALE, 64 * gg_input::AXIS_SCALE),
        );
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        let moved = camera.eye(Eye::ORIGIN).position;
        assert!((moved.x - (at.x - 10.0)).abs() < 1e-9, "{moved:?}");
        assert!((moved.y - (at.y + 2.0)).abs() < 1e-9, "{moved:?}");
        assert_eq!(moved.z, at.z, "a flat pan left the plane");
        // And the same motion with nothing held moves nothing: the axes carry a
        // delta every tick the mouse does, so the verb is the only thing that
        // decides whether it counts.
        let mut loose = InputFrame::default();
        let ax = axis_id(&input, verb::LOOK_X).expect("the host appended it");
        loose.axes[ax.index()] = 320 * gg_input::AXIS_SCALE;
        input.tick_from(loose);
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        assert_eq!(camera.eye(Eye::ORIGIN).position, moved, "it panned loose");
    }

    /// A lateral step is the same fraction of the window at every zoom, which
    /// is what keeps one key usable across a level and inside a doorway.
    #[test]
    fn a_flat_step_scales_with_the_zoom() {
        let mut travelled = Vec::new();
        for zoom in [FLAT_REFERENCE, FLAT_REFERENCE * 4.0] {
            let (world, mut input, mut camera) = (flat_world(), input(), Camera::default());
            camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
            camera.eye.as_mut().unwrap().ortho = zoom as f32;
            let at = camera.eye(Eye::ORIGIN).position;
            press(&mut input, verb::RIGHT);
            camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
            travelled.push(camera.eye(Eye::ORIGIN).position.x - at.x);
        }
        assert!((travelled[0] - PER_TICK).abs() < 1e-12, "{travelled:?}");
        assert!(
            (travelled[1] - PER_TICK * 4.0).abs() < 1e-12,
            "{travelled:?}"
        );
    }

    /// [`verb::FRAME`] is the way back: it centres the selection, sizes the
    /// view to it, and puts a tilted flat camera square to the level again —
    /// the one act that can, since dragging yaw and pitch to exactly zero is
    /// not a gesture.
    #[test]
    fn frame_centres_the_selection_and_squares_a_tilted_flat_camera() {
        let (world, mut input, mut camera) = (flat_world(), input(), Camera::default());
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        // However it got there — an older build, a save, a stray drag.
        let eye = camera.eye.as_mut().unwrap();
        eye.yaw = 0.7;
        eye.pitch = -0.4;
        eye.position = sim::DVec3::new(-400.0, 90.0, -12.0);

        let (at, half) = SLABS[1];
        let nav = Nav {
            target: Some(Renderable::boxed(at, half, 0)),
            ..nav()
        };
        press(&mut input, verb::FRAME);
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav);
        let eye = camera.eye(Eye::ORIGIN);
        assert_eq!((eye.yaw, eye.pitch), (0.0, 0.0), "it is still tilted");
        assert_eq!((eye.position.x, eye.position.y), (at.x, at.y));
        assert!(
            eye.position.z > f64::from(half.z),
            "the camera is inside the level: {eye:?}"
        );
        // Bounded by the width here — a metre wide over a sixteen-by-nine pane
        // is more view than half a metre tall is.
        let want = (f64::from(half.x) / nav.aspect).max(f64::from(half.y)) * FRAME_MARGIN;
        assert!((f64::from(eye.ortho) - want).abs() < 1e-5, "{eye:?}");
    }

    /// Nothing selected frames the **whole scene**, which is the state an
    /// operator who has lost it entirely is in — and the answer differs from
    /// framing one slab, or this would pass on a build that ignored the
    /// selection.
    #[test]
    fn frame_with_nothing_selected_fits_every_box_there_is() {
        let (world, mut input, mut camera) = (flat_world(), input(), Camera::default());
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        press(&mut input, verb::FRAME);
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        let eye = camera.eye(Eye::ORIGIN);
        // The two slabs span x from -12 to 22 and y from -0.5 to 6.5.
        assert!((eye.position.x - 5.0).abs() < 1e-9, "{eye:?}");
        assert!((eye.position.y - 3.0).abs() < 1e-9, "{eye:?}");
        let want = (17.0 / nav().aspect).max(3.5) * FRAME_MARGIN;
        assert!((f64::from(eye.ortho) - want).abs() < 1e-4, "{eye:?}");
    }

    /// The key acts on its press edge: held down it would recompute the same
    /// answer sixty times a second, and a pan made with a finger still on it
    /// would be undone every tick it lasted.
    #[test]
    fn frame_fires_once_per_press_and_does_not_fight_a_pan() {
        let (world, mut input, mut camera) = (flat_world(), input(), Camera::default());
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        press(&mut input, verb::FRAME);
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        let framed = camera.eye(Eye::ORIGIN).position;

        // Still held, and now panning as well.
        let (pan, ask) = (
            id(&input, verb::PAN).expect("the host appended it"),
            id(&input, verb::FRAME).expect("and that one"),
        );
        let mut both = InputFrame {
            buttons: (1 << pan.index()) | (1 << ask.index()),
            ..InputFrame::default()
        };
        let ax = axis_id(&input, verb::LOOK_X).expect("the host appended it");
        both.axes[ax.index()] = 320 * gg_input::AXIS_SCALE;
        input.tick_from(both);
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        assert!(
            (camera.eye(Eye::ORIGIN).position.x - (framed.x - 10.0)).abs() < 1e-9,
            "the held key re-framed over the pan"
        );
    }

    // ------------------------------------------------ §6 M63: flying it ----

    /// Every knob this milestone added, back where it found it. CVars are
    /// process-global and these tests move three; nextest gives each test its
    /// own process, so this is belt and braces — and it is what lets the
    /// assertions below be written against the *shipped* defaults rather than
    /// against whatever ran first.
    struct Knobs(f64, f64, bool);

    impl Knobs {
        fn take() -> Knobs {
            Knobs(SPEED.float(), SENSITIVITY.float(), INVERT.bool())
        }
    }

    impl Drop for Knobs {
        fn drop(&mut self) {
            SPEED.set_float(self.0);
            SENSITIVITY.set_float(self.1);
            INVERT.set_bool(self.2);
        }
    }

    /// The pointer is the camera's for exactly as long as a drag lasts. Both
    /// buttons take it — a pan that reached the window edge stopped for the
    /// same reason a look did — and the tick after the release gives it back,
    /// which is the half a toggle would not have.
    #[test]
    fn the_drag_captures_the_pointer_and_the_release_gives_it_back() {
        let (world, mut input, mut camera) = (world(), input(), Camera::default());
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        assert!(!camera.flying(), "captured with nothing held");
        for held in [verb::LOOK, verb::PAN] {
            press(&mut input, held);
            camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
            assert!(camera.flying(), "{held} did not capture");
            input.tick_from(InputFrame::default());
            camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
            assert!(!camera.flying(), "{held} kept the pointer after release");
        }
        // And a scene that starts playing hands it back without a release: the
        // camera is not the one being flown there, so a grab would be an arrow
        // the operator cannot get back without stopping.
        press(&mut input, verb::LOOK);
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        assert!(camera.flying());
        camera.fly(&world, &frame(Play::Running, Some(&input)), nav());
        assert!(!camera.flying(), "play kept the pointer");
    }

    /// A flat eye refuses the look, so there is nothing to capture the pointer
    /// *for* — but it keeps the pan, which is the gesture a flat scene is
    /// actually authored with. The pair is the assertion: one rule, two answers.
    #[test]
    fn a_flat_eye_captures_for_the_pan_and_not_for_the_refused_look() {
        let (world, mut input, mut camera) = (flat_world(), input(), Camera::default());
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        press(&mut input, verb::LOOK);
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        assert!(!camera.flying(), "captured for a drag it refuses to act on");
        press(&mut input, verb::PAN);
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        assert!(camera.flying(), "the flat pan lost the pointer");
    }

    /// One wheel, two subjects, arbitrated by the button already down: while the
    /// drag has the pointer a notch is the throttle and moves the eye not at
    /// all, and with nothing held it is the dolly it always was and leaves the
    /// speed alone. Asserted as a pair, because either half alone passes on a
    /// build that does only one of the two.
    #[test]
    fn the_wheel_throttles_while_captured_and_dollies_otherwise() {
        let _knobs = Knobs::take();
        let (world, mut input, mut camera) = (world(), input(), Camera::default());
        SPEED.set_float(15.0);
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        let at = camera.eye(Eye::ORIGIN).position;

        press(&mut input, verb::LOOK);
        camera.fly(
            &world,
            &frame(Play::Stopped, Some(&input)),
            Nav { wheel: 1, ..nav() },
        );
        let faster = SPEED.float();
        assert!((faster - 15.0 * SPEED_PER_NOTCH).abs() < 1e-9, "{faster}");
        assert_eq!(
            camera.eye(Eye::ORIGIN).position,
            at,
            "the throttle also dollied"
        );

        input.tick_from(InputFrame::default());
        camera.fly(
            &world,
            &frame(Play::Stopped, Some(&input)),
            Nav { wheel: 1, ..nav() },
        );
        assert_eq!(SPEED.float(), faster, "the dolly also throttled");
        assert!(
            (camera.eye(Eye::ORIGIN).position.z - (at.z - DOLLY_PER_NOTCH)).abs() < 1e-12,
            "the notch reached no camera at all"
        );
    }

    /// The throttle is geometric and clamps at both ends, for
    /// [`ZOOM_PER_NOTCH`]'s reasons: a notch is the same *proportion* at
    /// 0.2 m/s and at 200, and a speed past either bound is a way to lose the
    /// scene rather than a way to reach it.
    #[test]
    fn the_throttle_is_geometric_and_clamps_at_both_ends() {
        let _knobs = Knobs::take();
        let (world, mut input, mut camera) = (world(), input(), Camera::default());
        SPEED.set_float(15.0);
        press(&mut input, verb::LOOK);
        let notch = |camera: &mut Camera, wheel: i32| {
            let was = SPEED.float();
            camera.fly(
                &world,
                &frame(Play::Stopped, Some(&input)),
                Nav { wheel, ..nav() },
            );
            SPEED.float() / was
        };
        let near = notch(&mut camera, 1);
        for _ in 0..12 {
            notch(&mut camera, 1);
        }
        let far = notch(&mut camera, 1);
        assert!(
            (near - far).abs() < 1e-9,
            "a notch is {near} of the speed here and {far} of it eight times faster"
        );
        for _ in 0..200 {
            notch(&mut camera, 1);
        }
        assert_eq!(SPEED.float(), SPEED_RANGE.1);
        for _ in 0..400 {
            notch(&mut camera, -1);
        }
        assert_eq!(SPEED.float(), SPEED_RANGE.0);
    }

    /// The knob is metres a **second**, which is only true if the tick rate
    /// divides it: the same key held for one second covers the same ground at
    /// 60 Hz and at 240, and the per-tick distances differ by exactly four.
    #[test]
    fn a_held_key_covers_the_same_ground_a_second_at_any_pace() {
        let _knobs = Knobs::take();
        SPEED.set_float(15.0);
        let travelled = |hz: u32| {
            let (world, mut input, mut camera) = (world(), input(), Camera::default());
            fn at(hz: u32, input: &Input) -> Frame<'_> {
                Frame {
                    hz,
                    ..frame(Play::Stopped, Some(input))
                }
            }
            camera.fly(&world, &at(hz, &input), nav());
            let from = camera.eye(Eye::ORIGIN).position;
            for _ in 0..hz {
                press(&mut input, verb::FORWARD);
                camera.fly(&world, &at(hz, &input), nav());
            }
            from.z - camera.eye(Eye::ORIGIN).position.z
        };
        let (slow, fast) = (travelled(60), travelled(240));
        assert!(
            (slow - 15.0).abs() < 1e-9,
            "{slow} metres in a 60 Hz second"
        );
        assert!(
            (fast - 15.0).abs() < 1e-9,
            "{fast} metres in a 240 Hz second"
        );
        // The vacuity guard: a build ignoring `hz` entirely would pass the two
        // above only by passing this one too.
        assert!(
            (per_tick(60) - 4.0 * per_tick(240)).abs() < 1e-12,
            "the pace divides nothing"
        );
    }

    /// Sensitivity scales the turn and nothing else; invert flips the pitch and
    /// leaves the yaw where it was. Two knobs, and the failure worth catching is
    /// one of them reaching the other axis.
    #[test]
    fn sensitivity_scales_the_turn_and_the_invert_reaches_only_pitch() {
        let _knobs = Knobs::take();
        let turned = |sensitivity: f64, invert: bool| {
            SENSITIVITY.set_float(sensitivity);
            INVERT.set_bool(invert);
            let (world, mut input, mut camera) = (world(), input(), Camera::default());
            camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
            drag(
                &mut input,
                (100 * gg_input::AXIS_SCALE, 100 * gg_input::AXIS_SCALE),
            );
            camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
            let eye = camera.eye(Eye::ORIGIN);
            (eye.yaw, eye.pitch)
        };
        let (yaw, pitch) = turned(1.0, false);
        let (twice_yaw, twice_pitch) = turned(2.0, false);
        assert!((twice_yaw - 2.0 * yaw).abs() < 1e-6, "{twice_yaw} vs {yaw}");
        assert!((twice_pitch - 2.0 * pitch).abs() < 1e-6);
        let (inverted_yaw, inverted_pitch) = turned(1.0, true);
        assert_eq!(inverted_yaw, yaw, "the invert reached the yaw");
        assert!((inverted_pitch + pitch).abs() < 1e-9, "{inverted_pitch}");
    }

    /// The latch a frame adds is the tick's own rate and the tick's own sign,
    /// and it exists only while the drag does — raw device motion arrives
    /// whatever the pointer is over, so a latch offered unconditionally would
    /// swing the picture at frame rate whenever the operator reached for a menu.
    #[test]
    fn the_frames_latch_is_the_ticks_own_turn_and_only_while_the_drag_lasts() {
        let _knobs = Knobs::take();
        SENSITIVITY.set_float(1.0);
        INVERT.set_bool(false);
        let (world, mut input, mut camera) = (world(), input(), Camera::default());
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        assert!(camera.look(&input).is_none(), "latched with nothing held");

        drag(&mut input, (100 * gg_input::AXIS_SCALE, 0));
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        let look = camera.look(&input).expect("the drag is turning it");
        // Negated, because the tick spends `yaw -= dx * rate`: a latch that
        // added would turn the picture one way and the tick the other, which
        // reads as a view that shakes rather than one that leads.
        assert!((look.yaw_rate + LOOK_PER_UNIT).abs() < 1e-9, "{look:?}");
        assert_eq!(look.yaw_rate, look.pitch_rate, "the two axes disagree");
        assert_eq!(look.pitch_limit, PITCH_LIMIT, "the latch would pass a pole");
        assert_eq!(
            look.yaw_axis,
            axis_id(&input, verb::LOOK_X).expect("appended").index() as u32
        );

        // A *pan* captures the pointer and turns nothing, so it offers no latch:
        // the two are one gesture on two buttons everywhere except here.
        gesture(&mut input, verb::PAN, (100 * gg_input::AXIS_SCALE, 0));
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        assert!(camera.flying(), "the pan lost the pointer");
        assert!(camera.look(&input).is_none(), "the pan offered a turn");
    }

    /// The pair a host blends is two *different* ticks, and it is the identity
    /// on the tick the camera latched — a blend against a default would swing
    /// the picture in from the origin over one tick, which is the worst frame of
    /// a session to put a swoop in.
    #[test]
    fn the_blended_pair_is_two_ticks_and_the_first_one_is_not_a_swoop() {
        let (world, mut input, mut camera) = (world(), input(), Camera::default());
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        let (previous, current) = camera.eyes(Eye::ORIGIN);
        assert_eq!(previous, current, "the latching tick blends from nowhere");
        assert_eq!(current.position, AT, "and it is the game's own eye");

        press(&mut input, verb::FORWARD);
        camera.fly(&world, &frame(Play::Stopped, Some(&input)), nav());
        let (previous, current) = camera.eyes(Eye::ORIGIN);
        assert_eq!(previous.position, AT, "the previous tick moved with it");
        assert!((previous.position.z - current.position.z).abs() > 1e-9);

        // While playing the pair is the game's, both halves, so the blend a host
        // does over it is the identity and the game's own interpolation is the
        // only one running.
        let game = Eye::at(sim::DVec3::new(1.0, 2.0, 3.0), 0.4, -0.2);
        camera.fly(&world, &frame(Play::Running, Some(&input)), nav());
        assert_eq!(camera.eyes(game), (game, game));
    }
}
