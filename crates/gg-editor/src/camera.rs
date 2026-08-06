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

use crate::host::verb;
use gg_ecs::boundary::Eye;
use gg_input::{ActionId, AxisId, Input, MAX_ACTIONS, MAX_AXES};
use gg_math::sim;

/// Metres per tick held. At 60 Hz that is 15 m/s — a second crosses demo 05's
/// scene rather than the room it is in.
const MOVE_PER_TICK: f64 = 0.25;

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
}

impl Camera {
    /// One tick.
    pub(crate) fn fly(&mut self, world: &gg_ecs::World, frame: &crate::Frame) {
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

        // Look first: the move below is along the basis this leaves, so a drag
        // that turns and a key that pushes compose within one tick.
        //
        // No anchor and no previous value: a device axis is already this tick's
        // delta, so a press turns nothing on its own and there is no state to
        // reset on release. Integer, so a replayed drag lands where the recorded
        // one did — the same reason the recorder quantizes at all (§4.7).
        let turned = held(verb::LOOK) && {
            let axes = input.frame().axes;
            let delta = |name| axis_id(input, name).map_or(0, |a| axes[a.index()]);
            let (dx, dy) = (delta(verb::LOOK_X), delta(verb::LOOK_Y));
            let unit = LOOK_PER_UNIT / gg_input::AXIS_SCALE as f32;
            eye.yaw -= dx as f32 * unit;
            eye.pitch = (eye.pitch - dy as f32 * unit).clamp(-PITCH_LIMIT, PITCH_LIMIT);
            (dx, dy) != (0, 0)
        };

        // The renderer's own basis, built once for both this and the pick ray
        // (`crate::pick::basis`) — and in `gg_math::sim` rather than `std`, since
        // host state or not, a camera whose angles came out of a libm the two
        // hosts disagree about would put a golden scene's viewpoint on the
        // driver (§1.4, §3). Its `up` is unread here: lift is along **world** up.
        let (right, _, forward) = crate::pick::basis(eye.yaw, eye.pitch);
        let axis = |plus, minus| f64::from(u8::from(held(plus))) - f64::from(u8::from(held(minus)));
        let motion = right * axis(verb::RIGHT, verb::LEFT)
            + sim::DVec3::Y * axis(verb::UP, verb::DOWN)
            + forward * axis(verb::FORWARD, verb::BACK);
        eye.position += motion * MOVE_PER_TICK;

        // Once each, and only once the gesture actually did something — sixty
        // lines a second would bury every other thing the session says (§5.6c).
        if !self.turned && turned {
            self.turned = true;
            tracing::info!(tick = frame.tick, "editor: camera turned");
        }
        if !self.moved && motion != sim::DVec3::ZERO {
            self.moved = true;
            tracing::info!(tick = frame.tick, "editor: camera flown");
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

    /// The look verb held with this tick's device motion on the pair the host
    /// appended beside it.
    fn drag(input: &mut Input, motion: (i32, i32)) {
        let action = id(input, verb::LOOK).expect("the host appended it");
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
        camera.fly(&world, &frame(Play::Stopped, Some(&input)));
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
            camera.fly(&world, &frame(Play::Stopped, Some(&input)));
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
        camera.fly(&world, &frame(Play::Stopped, Some(&input)));
        camera.eye.as_mut().unwrap().pitch = -0.9;
        press(&mut input, verb::UP);
        camera.fly(&world, &frame(Play::Stopped, Some(&input)));
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
        camera.fly(&world, &frame(Play::Stopped, Some(&input)));
        assert_eq!(
            camera.eye(Eye::ORIGIN).yaw,
            0.0,
            "the press snapped the view"
        );
        drag(&mut input, (900 * gg_input::AXIS_SCALE, 0));
        camera.fly(&world, &frame(Play::Stopped, Some(&input)));
        assert!((camera.eye(Eye::ORIGIN).yaw + 900.0 * LOOK_PER_UNIT).abs() < 1e-4);
        // Straight down, well past the limit, in one absurd drag.
        for _ in 0..4 {
            drag(&mut input, (0, 900 * gg_input::AXIS_SCALE));
            camera.fly(&world, &frame(Play::Stopped, Some(&input)));
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
        camera.fly(&world, &frame(Play::Stopped, Some(&input)));
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
            camera.fly(&world, &frame(Play::Stopped, Some(&input)));
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
        camera.fly(&world, &frame(Play::Stopped, Some(&input)));
        for play in [Play::Running, Play::Paused] {
            press(&mut input, verb::FORWARD);
            camera.fly(&world, &frame(play, Some(&input)));
            assert_eq!(
                camera.eye(Eye::ORIGIN),
                Eye::ORIGIN,
                "{play:?} took the camera"
            );
        }
        // And the flight resumes exactly where it was left, rather than
        // re-latching onto whatever the game's eye did meanwhile.
        press(&mut input, verb::FORWARD);
        camera.fly(&world, &frame(Play::Stopped, Some(&input)));
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
        camera.fly(&world, &frame(Play::Stopped, None));
        assert_eq!(camera.eye(Eye::ORIGIN), Eye::ORIGIN);
    }
}
