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
use gg_ecs::boundary::{Eye, Renderable};
use gg_input::{ActionId, AxisId, Input, MAX_ACTIONS, MAX_AXES};
use gg_math::sim;

/// Metres per tick held. At 60 Hz that is 15 m/s — a second crosses demo 05's
/// scene rather than the room it is in.
const MOVE_PER_TICK: f64 = 0.25;

/// The orthographic half-height a lateral step is [`MOVE_PER_TICK`] at. Above
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

/// Radians per unit of raw device motion — a mouse count, near enough a pixel
/// on an ordinary desk. A 600-unit sweep turns three radians, which is the
/// gesture-to-rotation ratio every editor with a right-drag has settled on.
const LOOK_PER_UNIT: f32 = 0.005;

/// Just under a right angle: at exactly one the forward and world-up axes are
/// parallel and the basis below degenerates.
const PITCH_LIMIT: f32 = 1.5533;

/// The editor's camera. `Default` is unlatched — see the module docs.
#[derive(Default)]
pub(crate) struct Camera {
    /// `None` until the first stop, and `Some` for the rest of the session.
    eye: Option<Eye>,
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
    /// [`verb::FRAME`] last tick, so the key acts on its press edge. Held, it
    /// would recompute the identical answer sixty times a second and log it.
    framing: bool,
}

impl Camera {
    /// One tick.
    pub(crate) fn fly(&mut self, world: &gg_ecs::World, frame: &crate::Frame, nav: Nav) {
        self.live = matches!(frame.play, crate::Play::Stopped);
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

        // Look first: the move below is along the basis this leaves, so a drag
        // that turns and a key that pushes compose within one tick.
        let turned = !flat && held(verb::LOOK) && dragging && {
            let unit = LOOK_PER_UNIT / gg_input::AXIS_SCALE as f32;
            eye.yaw -= dx as f32 * unit;
            eye.pitch = (eye.pitch - dy as f32 * unit).clamp(-PITCH_LIMIT, PITCH_LIMIT);
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
        // Zoom: the wheel always, and forward/back as well under a flat eye,
        // where they otherwise move along the one direction a parallel
        // projection cannot show. A perspective eye dollies on the notch instead
        // — there, distance *is* framing and forward already means it.
        let zoomed = match flat {
            true => {
                let mut factor = 1.0;
                for _ in 0..nav.wheel.abs() {
                    factor *= match nav.wheel > 0 {
                        true => 1.0 / ZOOM_PER_NOTCH,
                        false => ZOOM_PER_NOTCH,
                    };
                }
                if ahead != 0.0 {
                    factor *= match ahead > 0.0 {
                        true => 1.0 / ZOOM_PER_TICK,
                        false => ZOOM_PER_TICK,
                    };
                }
                let want = (f64::from(eye.ortho) * factor).clamp(ZOOM_RANGE.0, ZOOM_RANGE.1);
                let moved = want != f64::from(eye.ortho);
                eye.ortho = want as f32;
                moved
            }
            false => {
                eye.position += forward * (f64::from(nav.wheel) * DOLLY_PER_NOTCH);
                nav.wheel != 0
            }
        };

        // Lateral movement, scaled to the zoom under a flat eye so a key covers
        // the same fraction of the window at every framing. Forward and back are
        // absent there: they are the zoom above.
        let step = MOVE_PER_TICK
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
            (flown.z - (AT.z - 4.0 * MOVE_PER_TICK)).abs() < 1e-12,
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
        assert_eq!(flown.y, AT.y + MOVE_PER_TICK);
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
        assert!(
            (flown.z - (AT.z - MOVE_PER_TICK)).abs() < 1e-12,
            "{flown:?}"
        );
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
        assert!(
            (travelled[0] - MOVE_PER_TICK).abs() < 1e-12,
            "{travelled:?}"
        );
        assert!(
            (travelled[1] - MOVE_PER_TICK * 4.0).abs() < 1e-12,
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
}
