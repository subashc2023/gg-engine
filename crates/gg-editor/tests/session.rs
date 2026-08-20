//! The scripted session driven in process (§6 M15).
//!
//! The half of the exit criteria that does not need a shell: the script selects
//! something, the inspector edits it, the toolbar asks for play/pause/step/save,
//! and the arithmetic lands where the step sizes say. `xtask reload --editor`
//! is the other half — the same script through the real host over demo 05, on
//! two tiers, compared by state hash.
//!
//! The world here is *shaped* like demo 05's rather than being it: a crate under
//! `crates/` may not depend on a demo, and what the script needs is only that
//! the first entity spawned carries a component whose first field is a vector.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::{Component, World};
use gg_editor::session::{Act, aim, frames, frames_from, script};
use gg_editor::{Commands, Editor, Frame, Pane};
use gg_input::{ActionId, AxisId, InputFrame};
use gg_math::sim;
use gg_rhi::MemoryUse;
use gg_ui::router::Tick;

/// Demo 05's observer, in shape: a `DVec3` first, then scalars.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "test.camera")]
#[repr(C)]
struct Camera {
    position: sim::DVec3,
    yaw: f32,
    pitch: f32,
    frozen: u32,
    _pad: u32,
}

const CLICK: ActionId = ActionId::new(1);
/// The wheel, as the host appends it (`gg_editor::host`): two actions rather
/// than an axis, so a notch is an ordinary recorded button edge.
const UP: ActionId = ActionId::new(2);
const DOWN: ActionId = ActionId::new(3);
const X: AxisId = AxisId::new(5);
const Y: AxisId = AxisId::new(6);
const TARGET: (u32, u32) = (1920, 1080);

/// More entities than one tree page holds, all of one kind.
///
/// One archetype on purpose: the tree is archetype order and dense row within it
/// (`gg_editor::scan`), so a world of one archetype is the only one where "row
/// 1" is a thing this test can name. Which entity the script lands on is the
/// *world's* property, and demo 05's gate is what asserts it there.
fn world() -> (World, gg_ecs::Entity) {
    let mut world = World::new();
    let mut second = None;
    for i in 0..64u32 {
        let entity = world.spawn();
        world
            .insert(
                entity,
                Camera {
                    position: sim::DVec3::new(4.0, f64::from(i), -30.0),
                    yaw: 0.5,
                    pitch: -0.12,
                    frozen: 0,
                    _pad: 0,
                },
            )
            .unwrap();
        if i == 1 {
            second = Some(entity);
        }
    }
    (world, second.unwrap())
}

/// An editor placed at `target`, for a caller that needs to know where the
/// panes went before it can aim at them (§6 M15.1 — the layout is no longer a
/// table of constants).
fn placed(target: (u32, u32)) -> Editor {
    let mut editor = Editor::new(None);
    editor.place(target, 1.0);
    editor
}

/// What a run of the script produced.
struct Run {
    world: World,
    camera: gg_ecs::Entity,
    editor: Editor,
    /// Every play-state change the toolbar asked for, in order.
    playing: Vec<bool>,
    steps: u32,
    saves: u32,
    /// What each stop did: `changed` before the restore and `identical` after,
    /// which is M14's pair pointed at a button (§6 M15.2). Both are required
    /// together — `identical` alone is a gate that cannot fail.
    stops: Vec<(bool, bool)>,
    /// Where the camera was on the tick before the first stop, so a test can
    /// still see the edits that the stop then discarded.
    before_stop: Option<sim::DVec3>,
}

/// Drive `frames` through the editor, playing the shell's part: it owns the
/// play state, the captured world and the save, exactly as `gg-runtime` does.
///
/// The stash is the shell's half of §6 M15.2 item 3 and is reproduced here
/// rather than stubbed, because the claim under test is about *bytes* — a
/// harness that only counted the clicks would pass over a stop that restored
/// nothing.
fn run(target: (u32, u32), input: &[InputFrame]) -> Run {
    let (mut world, camera) = world();
    let mut editor = Editor::new(None);
    let (mut playing, mut steps, mut saves) = (Vec::new(), 0, 0);
    let (mut stops, mut before_stop) = (Vec::new(), None);
    // The editor opens *Stopped* one tick in (§6 M15.2 post-close): nothing
    // captured, and the capture waits for the script's first click on the
    // transport. This harness runs no systems, so `world()` above stands in
    // for the bootstrap tick's work; the shell's rule that the bootstrap tick
    // advances with nothing captured is gated in `xtask reload --editor`, over
    // a game that bootstraps.
    let mut stash = None;
    let mut paused = true;
    for (tick, frame) in input.iter().enumerate() {
        let ui = Tick {
            motion: (frame.axes[X.index()], frame.axes[Y.index()]),
            primary: frame.pressed(CLICK),
            advance_focus: false,
            // `host` appends no `ui_press` and `Editor::tick` clears the one a
            // *game* may declare (§6 M81) — this had said the host's silence
            // settled it, which is the belief that hid the leak.
            activate: false,
            // Exactly what `Tick::from_input` derives from the same two verbs.
            scroll: i32::from(frame.pressed(UP)) - i32::from(frame.pressed(DOWN)),
        };
        let play = match (stash.is_some(), paused) {
            (false, _) => gg_editor::Play::Stopped,
            (true, false) => gg_editor::Play::Running,
            (true, true) => gg_editor::Play::Paused,
        };
        let commands: Commands = editor.tick(
            &mut world,
            &ui,
            &Frame {
                extent: target,
                dpi: 1.0,
                tick: tick as u64,
                hz: 60,
                play,
                // The camera has its own tests, against a real map; this harness
                // drives authored frames and holds none, which is also what
                // keeps every assertion below about the world and not the view.
                input: None,
                typed: "",
                passes: &[],
                memory: MemoryUse::default(),
                reload: None,
                draw_cursor: true,
                save_path: "target/editor/test.ggsv",
                title: "gg — test",
                project: Some("test"),
                projects: &[],
                maximized: false,
            },
        );
        if let Some(want) = commands.playing {
            if want && stash.is_none() {
                stash = Some(world.snapshot().encode());
            }
            paused = !want;
            playing.push(want);
        }
        // A step from `Stopped` captures first, for the shell's reason.
        if commands.step && stash.is_none() {
            stash = Some(world.snapshot().encode());
        }
        steps += u32::from(commands.step);
        saves += u32::from(commands.save);
        if commands.stop
            && let Some(bytes) = stash.take()
        {
            before_stop =
                before_stop.or_else(|| Some(world.get::<Camera>(camera).unwrap().position));
            let changed = world.snapshot().encode() != bytes;
            world
                .restore(&gg_ecs::Snapshot::decode(&bytes).unwrap())
                .unwrap();
            let identical = world.snapshot().encode() == bytes;
            stops.push((changed, identical));
            paused = true;
        }
    }
    Run {
        world,
        camera,
        editor,
        playing,
        steps,
        saves,
        stops,
        before_stop,
    }
}

fn position(run: &Run) -> sim::DVec3 {
    run.world.get::<Camera>(run.camera).unwrap().position
}

/// The whole session: what it selects, what it edits, and what it asks the
/// host for. The arithmetic is the point — four nudges at 1.0 and a +10/-10
/// pair at the coarse step is exactly +4.0, so a step button that did nothing
/// would land on +6.0 and a selection that missed would land on +0.0.
#[test]
fn the_scripted_session_edits_what_it_selected_and_asks_for_the_rest() {
    let input = frames(&script(&placed(TARGET)), CLICK, X, Y);
    let run = run(TARGET, &input);
    // Nothing, because §6 M15.4's last structural act is a delete — and that
    // the tree row *was* selected is what the arithmetic below proves, since a
    // selection that missed would leave the field where it started.
    assert_eq!(run.editor.selected(), None);
    // Read at the stop rather than at the end: the six edits were made inside
    // play mode, so the stop that follows them is *supposed* to take them back.
    assert_eq!(
        run.before_stop,
        Some(sim::DVec3::new(8.0, 1.0, -30.0)),
        "four at 1.0, then +10 and -10 at the coarse step"
    );
    assert_eq!(
        run.playing,
        vec![true, false, true, false],
        "play out of the Stopped open, pause, play, pause"
    );
    assert_eq!(run.steps, 2, "two single ticks while paused");
    assert_eq!(run.saves, 1);
    // Six nudges, then §6 M15.4's spawn, duplicate, delete and undo.
    assert_eq!(run.editor.tally(), (6 + 4, 1));
}

/// §6 M15.2 item 3, as bytes: the stop restores the world play began at, and
/// both halves of the claim are required. `changed` alone would pass over a
/// restore that did nothing; `identical` alone is a gate that cannot fail,
/// since a session that edited nothing satisfies it for free.
#[test]
fn a_stop_discards_what_play_did_and_lands_byte_for_byte_on_the_capture() {
    let input = frames(&script(&placed(TARGET)), CLICK, X, Y);
    let run = run(TARGET, &input);
    assert_eq!(run.stops, vec![(true, true)], "one stop: changed, restored");
    // The six nudges are gone with it — the same field the script moved to
    // 8.0 is back where the world was built.
    assert_eq!(position(&run), sim::DVec3::new(4.0, 1.0, -30.0));
    assert_ne!(
        run.before_stop,
        Some(position(&run)),
        "a stop that restored nothing would make these equal and prove nothing"
    );
}

/// The other half of §6 M15.2's Exit, and the half that makes the feature worth
/// having: an edit made while **stopped** is the scene, so the next play does
/// not take it back.
///
/// Scripted here rather than in [`script`] because it needs the order the gate's
/// session does not have — stop, *then* edit, then play and stop again. What it
/// is aimed at is the mistake of restoring on the play edge as well as the stop
/// one, which would leave a stopped edit alive exactly until it mattered.
#[test]
fn an_edit_made_while_stopped_survives_the_next_play_and_stop() {
    let editor = placed(TARGET);
    // The editor opens Stopped (§6 M15.2 post-close), which is exactly the
    // state this test needs first — so it starts editing straight away: select
    // row 1 and its first lane, then nudge once at the default step.
    let mut acts = Vec::new();
    for at in [
        aim::tree_row(&editor, 1),
        aim::lane(&editor, 1, 0),
        aim::plus(&editor),
    ] {
        acts.extend([Act::To(at.unwrap()), Act::Settle(3), Act::Click]);
    }
    // Play over the edit, then stop back onto it.
    acts.extend([
        Act::To(aim::play(&editor)),
        Act::Settle(3),
        Act::Click,
        Act::Settle(10),
        Act::To(aim::stop(&editor)),
        Act::Settle(3),
        Act::Click,
        Act::Settle(5),
    ]);
    let run = run(TARGET, &frames(&acts, CLICK, X, Y));
    assert_eq!(run.stops.len(), 1, "the one stop after the play");
    // 4.0 + one nudge at the default step of 1.0, and it is still there after a
    // play captured it and a stop restored what it captured.
    assert_eq!(position(&run), sim::DVec3::new(5.0, 1.0, -30.0));
    assert_eq!(
        run.stops[0],
        (false, true),
        "nothing moved during that play, so the stop changed nothing"
    );
}

/// §6 M15's fourth exit row at the level this crate can prove it, in the form
/// §6 M15.1 leaves it: the session does the same job at every window size.
///
/// The claim moved and it is worth being exact about how. Through M15 the
/// editor was a fixed canvas, so *one* tick stream landed the same clicks at
/// every extent. The panes fill the window now, so a stream is aimed at the
/// extent it was recorded at — what survives, and what an operator actually
/// cares about, is that the same *script* run at any extent leaves the same
/// world. What was given up is a stream recorded at one size replaying at
/// another, and the host records the extent instead (§6 M15.1's residual).
#[test]
fn the_same_session_leaves_the_same_world_at_any_window_size() {
    let reference = run(TARGET, &frames(&script(&placed(TARGET)), CLICK, X, Y));
    for target in [(640, 360), (1280, 720), (3840, 2160), (1024, 1024)] {
        let input = frames(&script(&placed(target)), CLICK, X, Y);
        let other = run(target, &input);
        assert_eq!(position(&other), position(&reference), "at {target:?}");
        assert_eq!(other.editor.selected(), reference.editor.selected());
        assert_eq!(other.playing, reference.playing);
        assert_eq!(other.editor.tally(), reference.editor.tally());
        assert_eq!(
            other.world.canonical_hash(),
            reference.world.canonical_hash(),
            "the world an editor session leaves behind is the window's business \
             at {target:?}"
        );
    }
}

/// And at one extent the stream is exactly reproducible, which is the half the
/// replay gate rests on: same frames in, same world out, twice.
#[test]
fn one_stream_at_one_extent_reproduces_itself() {
    let input = frames(&script(&placed(TARGET)), CLICK, X, Y);
    let first = run(TARGET, &input);
    let again = run(TARGET, &input);
    assert_eq!(first.world.canonical_hash(), again.world.canonical_hash());
    assert_eq!(first.editor.tally(), again.editor.tally());
}

/// §6 M15.1's own exit row at this level: the two gestures the script ends with
/// actually move the layout, and the layout they leave still holds every pane.
#[test]
fn the_session_drags_a_seam_and_re_docks_a_pane() {
    let before = placed(TARGET);
    let seam = before.seams()[0].rect;
    // Which strip each pane sits in, keyed by the strip's own line. Named this
    // way rather than by asserting a *particular* pane moved: the script aims
    // its tab drag from the pre-drag layout, so the seam gesture that runs first
    // shifts the strip under that point and decides which tab the drag grabs.
    // That is a property of the layout, not of what this test means — which is
    // that the gesture re-docked something.
    let strips = |editor: &Editor| {
        let mut rows: Vec<(i32, &str)> = Pane::ALL
            .iter()
            .filter_map(|p| editor.tab_rect(*p).map(|r| (r.y as i32, p.title())))
            .collect();
        rows.sort_unstable();
        rows
    };
    let before_strips = strips(&before);
    let input = frames(&script(&before), CLICK, X, Y);
    let mut run = run(TARGET, &input);
    run.editor.place(TARGET, 1.0);

    assert_ne!(
        run.editor.seams()[0].rect,
        seam,
        "the seam drag moved nothing"
    );
    assert_ne!(
        strips(&run.editor),
        before_strips,
        "no pane changed strips — the tab drag re-docked nothing"
    );
    // Every pane is still reachable — a docking gesture that loses one is worse
    // than one that does nothing.
    for pane in Pane::ALL {
        assert!(
            run.editor.tab_rect(pane).is_some(),
            "{} has no tab left",
            pane.title()
        );
    }
    // And the layout the editor is left holding is one it would accept from a
    // file, which is what makes it safe to persist.
    let root = run.editor.layout().clone();
    assert!(run.editor.set_layout(root));
}

/// A click that lands on nothing must change nothing — otherwise the session
/// above could be passing on an edit it never aimed at.
#[test]
fn a_session_that_clicks_empty_canvas_edits_nothing() {
    let editor = placed(TARGET);
    // The middle of the game pane, over a world holding no `Renderable` at all
    // — so the pick §6 M15.4 item 1 declares there finds nothing to select.
    let idle = [
        Act::To(aim::nowhere(&editor).expect("the game is up")),
        Act::Settle(4),
        Act::Click,
        Act::Click,
        Act::Settle(4),
    ];
    let run = run(TARGET, &frames(&idle, CLICK, X, Y));
    assert_eq!(run.editor.selected(), None);
    assert_eq!(run.editor.tally(), (0, 0));
    assert_eq!(position(&run), sim::DVec3::new(4.0, 1.0, -30.0));
    assert!(run.playing.is_empty() && run.steps == 0 && run.saves == 0);
}

/// A world the renderer would actually draw: an eye at the origin facing -Z,
/// and two boxes on that line. The far one is spawned **first**, so it is also
/// first in iteration order — a pick that took the first hit rather than the
/// nearest would select it and the assertion below would say so.
fn scene() -> (World, gg_ecs::Entity, gg_ecs::Entity, gg_ecs::Entity) {
    let boxed = |z: f64| {
        gg_ecs::boundary::Renderable::boxed(
            sim::DVec3::new(0.0, 0.0, z),
            sim::Vec3::splat(0.5),
            0x00ff_8000,
        )
    };
    let mut world = World::new();
    let eye = world.spawn();
    world.insert(eye, gg_ecs::boundary::Eye::ORIGIN).unwrap();
    let (far, near) = (world.spawn(), world.spawn());
    world.insert(far, boxed(-20.0)).unwrap();
    world.insert(near, boxed(-5.0)).unwrap();
    (world, near, far, eye)
}

/// Drive `input` over `world` with the scene **stopped** throughout — nothing
/// captured, so `Play::Stopped` on every tick, which is the one state a pick
/// happens in.
fn stopped(world: &mut World, editor: &mut Editor, input: &[InputFrame]) {
    for (tick, frame) in input.iter().enumerate() {
        let ui = Tick {
            motion: (frame.axes[X.index()], frame.axes[Y.index()]),
            primary: frame.pressed(CLICK),
            advance_focus: false,
            // `host` appends no `ui_press` and `Editor::tick` clears the one a
            // *game* may declare (§6 M81) — this had said the host's silence
            // settled it, which is the belief that hid the leak.
            activate: false,
            scroll: 0,
        };
        editor.tick(
            world,
            &ui,
            &Frame {
                extent: TARGET,
                dpi: 1.0,
                tick: tick as u64,
                hz: 60,
                play: gg_editor::Play::Stopped,
                input: None,
                typed: "",
                passes: &[],
                memory: MemoryUse::default(),
                reload: None,
                draw_cursor: true,
                save_path: "target/editor/test.ggsv",
                title: "gg — test",
                project: Some("test"),
                projects: &[],
                maximized: false,
            },
        );
    }
}

/// §6 M15.4 item 1, wired: a click in the stopped viewport selects the nearest
/// box the ray under it hits, and a click into empty sky puts the selection
/// back to nothing.
///
/// The deselect half is not a nicety — the tree has no way to clear a selection
/// either, so without it an operator who picked once could only ever pick
/// something else.
#[test]
fn a_click_in_the_stopped_viewport_picks_the_nearest_box_and_the_sky_clears_it() {
    let editor = placed(TARGET);
    let body = editor.pane_body(Pane::Viewport).expect("the game is up");
    let middle = aim::nowhere(&editor).expect("the game is up");
    // A tenth of the way down the pane: at the fov the console defaults to, a
    // half-metre box five metres out subtends far less than that, so this is
    // sky whichever box is nearer.
    let sky = (middle.0, body.y + body.h * 0.1);

    let (mut world, near, far, _) = scene();
    let mut run = Editor::new(None);
    // One stream, driven in two halves: `frames` glides from where the *stream*
    // left the pointer, so a second call would start over at the origin and aim
    // at the sum of the two.
    let first = [Act::To(middle), Act::Settle(4), Act::Click];
    let acts = [
        first.as_slice(),
        &[Act::To(sky), Act::Settle(4), Act::Click],
    ]
    .concat();
    let stream = frames(&acts, CLICK, X, Y);
    let split = frames(&first, CLICK, X, Y).len();
    stopped(&mut world, &mut run, &stream[..split]);
    assert_eq!(run.selected(), Some(near), "the far box is at {far:?}");
    stopped(&mut world, &mut run, &stream[split..]);
    assert_eq!(run.selected(), None, "a click on nothing selects nothing");
}

/// §6 M15.4 item 2: what is selected is drawn *in the scene*, and it is drawn
/// through the camera — turn the eye and the outline goes with it.
///
/// Measured as *any* geometry in the middle of the rendered rectangle, which is
/// the strongest signal available: with nothing selected the viewport is a hole
/// the editor draws nothing over at all
/// (`nothing_is_drawn_over_the_middle_of_the_game`), so a vertex there is the
/// marker and can be nothing else.
#[test]
fn the_selection_is_outlined_in_the_scene_and_the_outline_follows_the_camera() {
    let editor = placed(TARGET);
    let middle = aim::nowhere(&editor).expect("the game is up");
    let (mut world, _, _, eye) = scene();
    let mut run = Editor::new(None);
    // Vertices in the middle half of the pane — clear of its border and of the
    // play tag in its corner — and their mean position across the pane.
    let marks = |editor: &Editor| -> Vec<f32> {
        let view = editor.viewport_rect();
        let (w, h) = (view.width as f32, view.height as f32);
        let (x, y) = (view.x as f32, view.y as f32);
        editor
            .vertices()
            .iter()
            .filter(|v| {
                v.pos[0] > x + w * 0.25
                    && v.pos[0] < x + w * 0.75
                    && v.pos[1] > y + h * 0.25
                    && v.pos[1] < y + h * 0.75
            })
            .map(|v| v.pos[0])
            .collect()
    };
    let settle = frames(&[Act::Settle(2)], CLICK, X, Y);

    stopped(&mut world, &mut run, &settle);
    assert!(marks(&run).is_empty(), "the unpicked viewport is a hole");

    let acts = [Act::To(middle), Act::Settle(4), Act::Click, Act::Settle(2)];
    stopped(&mut world, &mut run, &frames(&acts, CLICK, X, Y));
    assert!(run.selected().is_some(), "the click found the near box");
    let before = marks(&run);
    assert!(!before.is_empty(), "nothing is drawn where the box is");

    // Turn the eye a tenth of a radian to the left. The box was dead ahead, so
    // it has to move right across the pane.
    *world.get_mut::<gg_ecs::boundary::Eye>(eye).unwrap() =
        gg_ecs::boundary::Eye::at(sim::DVec3::ZERO, 0.1, 0.0);
    stopped(&mut world, &mut run, &settle);
    let after = marks(&run);
    assert!(
        !after.is_empty(),
        "the outline vanished when the eye turned"
    );
    let mean = |xs: &[f32]| xs.iter().sum::<f32>() / xs.len() as f32;
    assert!(
        mean(&after) > mean(&before) + 10.0,
        "the outline did not follow the camera: {} then {}",
        mean(&before),
        mean(&after)
    );
}

/// §6 M15.4 item 3: a gizmo drag moves the entity by a value the inspector
/// could have typed with `+`.
///
/// Proven by doing both. Two runs from the same scene — one drags the world-X
/// handle far enough to resolve to one step, one clicks `+` once on the same
/// lane — and the two worlds are compared. The comparison is what makes this a
/// gate rather than a screenshot: the drag reaches the field through the pointer
/// and a projection, the nudge through the registry, and only a quantized drag
/// can land them on the same number.
#[test]
fn a_gizmo_drag_lands_on_a_value_the_nudge_bar_could_have_typed() {
    let editor = placed(TARGET);
    let middle = aim::nowhere(&editor).expect("the game is up");
    let pick = [Act::To(middle), Act::Settle(4), Act::Click, Act::Settle(3)];

    // The drag. The handle's position is only knowable once the pick has run,
    // so the stream is built in two pieces off one pointer (`frames_from`).
    let (mut world, near, _, _) = scene();
    let mut dragged = Editor::new(None);
    stopped(&mut world, &mut dragged, &frames(&pick, CLICK, X, Y));
    let handle = dragged.handle(0).expect("the world-X handle is up");
    // Several arms' length along it — how much *world* that is depends on the
    // pane's height and the field of view, which is exactly why the number of
    // steps it lands on is read off the result below rather than assumed here.
    let arm = handle.0 - middle.0;
    let drag = [
        Act::To(handle),
        Act::Settle(4),
        Act::Drag((handle.0 + arm * 6.0, handle.1)),
        Act::Settle(3),
    ];
    stopped(
        &mut world,
        &mut dragged,
        &gg_editor::session::frames_from(middle, &drag, CLICK, X, Y),
    );
    let moved = world
        .get::<gg_ecs::boundary::Renderable>(near)
        .unwrap()
        .position;

    // Whatever it landed on is a whole number of the inspector's own step, or
    // `+` could not reach it however many times it were pressed.
    let grain = gg_editor::STEPS[1];
    let steps = moved.x / grain;
    assert!(steps >= 1.0, "the drag moved nothing: {moved:?}");
    assert_eq!(
        steps,
        steps.round(),
        "the drag was not quantized: {moved:?}"
    );

    // Now type it: select, take the first lane of the first field, and press
    // `+` exactly that many times at that step.
    let (mut world, near, _, _) = scene();
    let mut nudged = Editor::new(None);
    let mut acts = pick.to_vec();
    for at in [aim::lane(&editor, 1, 0), aim::plus(&editor)] {
        acts.extend([Act::To(at.unwrap()), Act::Settle(3), Act::Click]);
    }
    acts.extend(core::iter::repeat_n(Act::Click, steps as usize - 1));
    acts.push(Act::Settle(3));
    stopped(&mut world, &mut nudged, &frames(&acts, CLICK, X, Y));
    let typed = world
        .get::<gg_ecs::boundary::Renderable>(near)
        .unwrap()
        .position;

    assert_eq!(nudged.tally().0 as f64, steps, "one press per step");
    assert_eq!(moved, typed, "the drag landed somewhere `+` cannot reach");
    // And it was the gizmo that did it, not the pick: the same click without a
    // drag leaves the box where it was.
    assert_eq!(dragged.selected(), Some(near));
}

/// A drag on one handle moves that axis and no other, and the same drag
/// backwards puts the value back **exactly** — the property a gizmo that
/// accumulated per tick would not have, because its rounding would compound.
#[test]
fn a_gizmo_drag_moves_one_axis_and_the_same_drag_back_restores_it() {
    let editor = placed(TARGET);
    let middle = aim::nowhere(&editor).expect("the game is up");
    let (mut world, near, _, _) = scene();
    let mut run = Editor::new(None);
    let at = |world: &World| {
        world
            .get::<gg_ecs::boundary::Renderable>(near)
            .unwrap()
            .position
    };
    let pick = [Act::To(middle), Act::Settle(4), Act::Click, Act::Settle(3)];
    stopped(&mut world, &mut run, &frames(&pick, CLICK, X, Y));
    let was = at(&world);

    let handle = run.handle(1).expect("the world-Y handle is up");
    // Most of the way to the top of the pane, rather than a multiple of the
    // arm: `Pointer::advance` clamps to the canvas, so a reach expressed in arm
    // lengths silently stops being symmetric the moment the arm gets longer —
    // and a clamped gesture *out* with an unclamped one *back* is a round trip
    // that looks like a quantization bug (§6 M20 item 10).
    let body = editor.pane_body(Pane::Viewport).expect("the game is up");
    let reach = (0.0, -(handle.1 - body.y) * 0.8);
    // Where the first gesture leaves the pointer, which is where the second one
    // has to be told it starts from: the frames carry motion, not position.
    let ended = (handle.0 + reach.0, handle.1 + reach.1);
    assert!(
        ended.1 > 0.0 && ended.1 < body.bottom(),
        "the drag leaves the canvas and clamps: {ended:?}"
    );
    let out = [
        Act::To(handle),
        Act::Settle(4),
        Act::Drag(ended),
        Act::Settle(3),
    ];
    stopped(
        &mut world,
        &mut run,
        &gg_editor::session::frames_from(middle, &out, CLICK, X, Y),
    );
    let there = at(&world);
    assert_ne!(there.y, was.y, "the Y handle moved nothing");
    assert_eq!(
        (there.x, there.z),
        (was.x, was.z),
        "it moved X or Z as well"
    );
    assert_eq!(
        there.y,
        there.y.round(),
        "the drag was not quantized: {there:?}"
    );

    // Grab it again where it is now — the arm followed the box — and make the
    // same gesture backwards.
    let handle = run.handle(1).expect("the handle followed the box");
    let back = [
        Act::To(handle),
        Act::Settle(4),
        Act::Drag((handle.0 - reach.0, handle.1 - reach.1)),
        Act::Settle(3),
    ];
    stopped(
        &mut world,
        &mut run,
        &gg_editor::session::frames_from(ended, &back, CLICK, X, Y),
    );
    assert_eq!(at(&world), was, "the drag back is not the drag's inverse");
}

/// §6 M15.4 item 4: undo restores the world the edit was made against, byte for
/// byte, and redo re-applies it.
///
/// Driven through the `edit` menu, which is the whole of the operator's reach
/// into it — and byte-compared with M14's own encoder rather than field by
/// field, so a step that put the nudged lane back and lost the selection's
/// neighbour would fail here.
#[test]
fn the_edit_menu_undoes_a_nudge_byte_for_byte_and_redoes_it() {
    let editor = placed(TARGET);
    let middle = aim::nowhere(&editor).expect("the game is up");
    let (mut world, near, _, _) = scene();
    let mut run = Editor::new(None);
    let at = |world: &World| {
        world
            .get::<gg_ecs::boundary::Renderable>(near)
            .unwrap()
            .position
    };
    let clean = world.snapshot().encode();

    // Pick the box, take the first lane of its first field, and nudge it twice.
    let mut acts = vec![Act::To(middle), Act::Settle(4), Act::Click, Act::Settle(3)];
    for target in [aim::lane(&editor, 1, 0), aim::plus(&editor)] {
        acts.extend([Act::To(target.unwrap()), Act::Settle(3), Act::Click]);
    }
    acts.extend([Act::Click, Act::Settle(3)]);
    // `edit → undo`, twice: a menu item is two clicks, the title then the item.
    let menu = |item: usize| {
        [
            Act::To(aim::menu(&editor, 1).unwrap()),
            Act::Settle(3),
            Act::Click,
            Act::To(aim::menu_item(&editor, 1, item).unwrap()),
            Act::Settle(3),
            Act::Click,
            Act::Settle(3),
        ]
    };
    let split = frames(&acts, CLICK, X, Y).len();
    acts.extend(menu(0));
    let once = frames(&acts, CLICK, X, Y).len();
    acts.extend(menu(0));
    let twice = frames(&acts, CLICK, X, Y).len();
    acts.extend(menu(1));
    let stream = frames(&acts, CLICK, X, Y);

    stopped(&mut world, &mut run, &stream[..split]);
    assert_eq!(at(&world).x, 2.0, "two presses on `+` at the default step");
    let edited = world.snapshot().encode();

    stopped(&mut world, &mut run, &stream[split..once]);
    assert_eq!(at(&world).x, 1.0, "one undo went back one nudge");
    stopped(&mut world, &mut run, &stream[once..twice]);
    assert_eq!(
        world.snapshot().encode(),
        clean,
        "two undos did not land on the world the first edit was made against"
    );
    stopped(&mut world, &mut run, &stream[twice..]);
    assert_eq!(at(&world).x, 1.0, "redo did not re-apply the first nudge");

    // And a third undo, with nothing left to go back to, changes nothing rather
    // than restoring whatever was on the bottom of the ring.
    let held = world.snapshot().encode();
    let mut acts: Vec<Act> = menu(0).into();
    acts.extend(menu(0));
    let tail =
        gg_editor::session::frames_from(aim::menu_item(&editor, 1, 1).unwrap(), &acts, CLICK, X, Y);
    stopped(&mut world, &mut run, &tail);
    assert_ne!(world.snapshot().encode(), edited, "the redo stack survived");
    assert_eq!(
        world.snapshot().encode(),
        clean,
        "and one more undo is the floor"
    );
    let floor = world.snapshot().encode();
    assert_ne!(floor, held, "the first of those two undos did nothing");
}

/// §6 M15.4 item 5: the tree spawns, duplicates and deletes, and what it leaves
/// behind is a world the schema manifest still accepts.
///
/// The manifest check is the one that matters and it is free: `snapshot().encode()`
/// writes the schema of every component present and `Snapshot::decode` reads it
/// back, so a restructure that left a half-built entity or an unregistered
/// component would fail to round-trip here rather than at the next save.
#[test]
fn the_tree_spawns_duplicates_and_deletes_and_the_world_still_encodes() {
    let editor = placed(TARGET);
    let (mut world, _, _, _) = scene();
    let mut run = Editor::new(None);
    let before = world.len();
    let click = |at: Option<(f32, f32)>| [Act::To(at.unwrap()), Act::Settle(4), Act::Click];

    // Spawn. It lands in front of the camera and comes up selected, which is
    // what makes the very next click a click on the thing just made.
    let spawn = click(aim::spawn(&editor));
    let split = frames(&spawn, CLICK, X, Y).len();
    let mut acts: Vec<Act> = spawn.into();
    acts.extend(click(aim::duplicate(&editor)));
    let duplicated = frames(&acts, CLICK, X, Y).len();
    acts.extend(click(aim::delete(&editor)));
    let stream = frames(&acts, CLICK, X, Y);

    stopped(&mut world, &mut run, &stream[..split]);
    let made = run.selected().expect("a spawn selects what it made");
    assert_eq!(world.len(), before + 1);
    let placed_at = world
        .get::<gg_ecs::boundary::Renderable>(made)
        .expect("a spawn makes something the host can draw")
        .position;
    // Five metres down the camera's forward axis, which for the scene's eye is
    // straight along -Z.
    assert_eq!(placed_at, sim::DVec3::new(0.0, 0.0, -5.0));

    stopped(&mut world, &mut run, &stream[split..duplicated]);
    let copy = run.selected().expect("a duplicate selects the copy");
    assert_ne!(copy, made, "the duplicate selected the original");
    assert_eq!(world.len(), before + 2);
    let made_box = *world.get::<gg_ecs::boundary::Renderable>(made).unwrap();
    let copy_box = *world
        .get::<gg_ecs::boundary::Renderable>(copy)
        .expect("the copy did not carry the original's components");
    assert_eq!(copy_box.half_extent, made_box.half_extent);
    assert_eq!(copy_box.color, made_box.color);
    // One nudge grain to the camera's right (§6 M20 item 10), not on top of it:
    // a copy at the original's exact position is invisible, and the pick's
    // lowest-index tie-break hands the next click back to the original, so the
    // button reads as inert while the world fills with stacked boxes.
    assert_eq!(
        copy_box.position - placed_at,
        sim::DVec3::new(gg_editor::STEPS[1], 0.0, 0.0),
        "the copy landed on top of its original"
    );

    stopped(&mut world, &mut run, &stream[duplicated..]);
    assert_eq!(run.selected(), None, "a delete left the corpse selected");
    assert_eq!(world.len(), before + 1);
    assert!(!world.is_alive(copy), "the copy is still alive");
    assert!(world.is_alive(made), "the delete took the wrong one");

    // And the world round-trips: encode, decode, restore, byte-identical.
    let bytes = world.snapshot().encode();
    let snapshot = gg_ecs::Snapshot::decode(&bytes).expect("the manifest still reads");
    world.restore(&snapshot).expect("and still applies");
    assert_eq!(world.snapshot().encode(), bytes);
}

/// The same three, undone: every structural edit records a step, so `edit → undo`
/// walks a spawn and a delete back exactly as it walks a nudge (§6 M15.4 item 4).
#[test]
fn a_spawn_and_a_delete_are_undoable_like_any_other_edit() {
    let editor = placed(TARGET);
    let (mut world, near, _, _) = scene();
    let mut run = Editor::new(None);
    let clean = world.snapshot().encode();
    let click = |at: Option<(f32, f32)>| [Act::To(at.unwrap()), Act::Settle(4), Act::Click];
    let menu = |pair: Option<((f32, f32), (f32, f32))>| {
        let (title, item) = pair.unwrap();
        [
            Act::To(title),
            Act::Settle(3),
            Act::Click,
            Act::To(item),
            Act::Settle(3),
            Act::Click,
            Act::Settle(3),
        ]
    };

    // Spawn something, then delete the box that was already there.
    let mut acts: Vec<Act> = click(aim::spawn(&editor)).into();
    let spawned = frames(&acts, CLICK, X, Y).len();
    acts.extend([Act::To(aim::nowhere(&editor).unwrap()), Act::Settle(4)]);
    acts.push(Act::Click);
    acts.push(Act::Settle(3));
    let picked = frames(&acts, CLICK, X, Y).len();
    acts.extend(click(aim::delete(&editor)));
    let did = frames(&acts, CLICK, X, Y).len();
    // Two undos: the delete, then the spawn.
    acts.extend(menu(aim::undo(&editor)));
    acts.extend(menu(aim::undo(&editor)));
    let stream = frames(&acts, CLICK, X, Y);

    stopped(&mut world, &mut run, &stream[..spawned]);
    assert_eq!(world.len(), 3 + 1, "the spawn made nothing");
    stopped(&mut world, &mut run, &stream[spawned..picked]);
    assert_eq!(
        run.selected(),
        Some(near),
        "the click picked the spawn rather than the box already there"
    );
    stopped(&mut world, &mut run, &stream[picked..did]);
    assert!(!world.is_alive(near), "the picked box was not deleted");
    assert_eq!(world.len(), 3, "the eye, the far box and the spawn");

    stopped(&mut world, &mut run, &stream[did..]);
    assert_eq!(
        world.snapshot().encode(),
        clean,
        "two undos did not walk a delete and a spawn back to the world before them"
    );
}

/// The same click while the scene is **playing** is the player's, not the
/// editor's: no widget is declared over the viewport at all, so the press goes
/// where a press over a game goes and the selection does not move.
#[test]
fn a_click_in_a_playing_viewport_picks_nothing() {
    let editor = placed(TARGET);
    let middle = aim::nowhere(&editor).expect("the game is up");
    let (mut world, _, _, _) = scene();
    let mut run = Editor::new(None);
    let input = frames(&[Act::To(middle), Act::Settle(4), Act::Click], CLICK, X, Y);
    for (tick, frame) in input.iter().enumerate() {
        let ui = Tick {
            motion: (frame.axes[X.index()], frame.axes[Y.index()]),
            primary: frame.pressed(CLICK),
            advance_focus: false,
            // `host` appends no `ui_press` and `Editor::tick` clears the one a
            // *game* may declare (§6 M81) — this had said the host's silence
            // settled it, which is the belief that hid the leak.
            activate: false,
            scroll: 0,
        };
        run.tick(
            &mut world,
            &ui,
            &Frame {
                extent: TARGET,
                dpi: 1.0,
                tick: tick as u64,
                hz: 60,
                play: gg_editor::Play::Running,
                input: None,
                typed: "",
                passes: &[],
                memory: MemoryUse::default(),
                reload: None,
                draw_cursor: true,
                save_path: "target/editor/test.ggsv",
                title: "gg — test",
                project: Some("test"),
                projects: &[],
                maximized: false,
            },
        );
    }
    assert_eq!(run.selected(), None);
}

/// Paging is what makes a tree over ten thousand entities usable, and the page
/// button is the only way to reach past the first thirty rows.
#[test]
fn the_tree_pages_and_a_row_on_page_two_selects_an_entity_page_one_never_showed() {
    let editor = placed(TARGET);
    let row = |i: usize| aim::tree_row(&editor, i).expect("the tree is up");
    let pick = [Act::To(row(0)), Act::Settle(3), Act::Click];
    let first = run(TARGET, &frames(&pick, CLICK, X, Y));
    let page_one = first.editor.selected().expect("a row on page one");

    let acts = [
        Act::To(aim::page(&editor, true).expect("the tree is up")),
        Act::Settle(3),
        Act::Click,
        Act::To(row(2)),
        Act::Settle(3),
        Act::Click,
    ];
    let paged = run(TARGET, &frames(&acts, CLICK, X, Y));
    let picked = paged.editor.selected().expect("a row on page two");
    assert_ne!(picked, page_one, "the page moved under the same row index");
}

/// A notch scrolls the pane the pointer is over, and the rows follow the clicks:
/// the same pixel selects a different entity once the list has moved under it.
///
/// The second half is the one a clip cannot provide — `DrawList` cuts the quads
/// and the router never hears about it (`gg_ui::scroll`), so a row scrolled off
/// the top would otherwise still be taking presses through whatever is up there.
#[test]
fn a_wheel_notch_scrolls_the_tree_under_the_pointer() {
    let editor = placed(TARGET);
    let row = aim::tree_row(&editor, 0).expect("the tree is up");
    let approach = frames(&[Act::To(row), Act::Settle(3)], CLICK, X, Y);
    let click = frames(&[Act::Click, Act::Settle(3)], CLICK, X, Y);
    // Notches as the recorder writes them: one action bit, one tick each, with
    // an idle tick between so the pair reads as two detents and not one long
    // press — which is what a wheel actually delivers.
    let notches = |n: usize, action: ActionId| -> Vec<InputFrame> {
        (0..n * 2)
            .map(|i| InputFrame {
                buttons: u64::from(i % 2 == 0) << action.index(),
                axes: [0; gg_input::MAX_AXES],
            })
            .collect()
    };
    let session = |wheel: Vec<InputFrame>| {
        let mut input = approach.clone();
        input.extend(wheel);
        input.extend(click.iter().copied());
        run(TARGET, &input).editor.selected().expect("a row")
    };

    let unscrolled = session(Vec::new());
    let scrolled = session(notches(4, DOWN));
    assert_ne!(
        unscrolled, scrolled,
        "four notches left the same entity under the pointer"
    );
    // And back up, past the top: the offset clamps, so the first row is the
    // first row again rather than somewhere above the world.
    let mut back = notches(4, DOWN);
    back.extend(notches(12, UP));
    assert_eq!(session(back), unscrolled, "scrolling up past zero drifted");
}

/// The launcher (§6 M15.1 item 4): with no project, the game pane is the picker,
/// and a click on a row asks the host to open that project and to close the
/// window this session is in.
///
/// The whole of item 4's Exit row that can be proven in process — what a shell
/// does with the answer is `xtask reload --launcher`'s, because it needs two
/// sessions and a real dylib.
#[test]
fn with_no_project_the_game_pane_is_a_picker_and_a_click_on_a_row_opens_one() {
    let projects: Vec<gg_editor::project::Project> = ["03-reload", "05-many"]
        .iter()
        .map(|name| gg_editor::project::Project {
            name: (*name).to_string(),
            game: std::path::PathBuf::from(format!("target/debug/demo_{name}.dll")),
            input: None,
            pack: None,
            built: true,
        })
        .collect();
    let mut world = World::new();
    let mut editor = placed(TARGET);
    // Aimed at the second row, so a picker that always reported the first would
    // fail by name rather than by luck.
    let at = aim::project(&editor, 1).expect("the game pane is up");
    let mut opened = None;
    for frame in frames(&[Act::To(at), Act::Settle(3), Act::Click], CLICK, X, Y) {
        let ui = Tick {
            motion: (frame.axes[X.index()], frame.axes[Y.index()]),
            primary: frame.pressed(CLICK),
            advance_focus: false,
            // `host` appends no `ui_press` and `Editor::tick` clears the one a
            // *game* may declare (§6 M81) — this had said the host's silence
            // settled it, which is the belief that hid the leak.
            activate: false,
            scroll: 0,
        };
        let commands: Commands = editor.tick(
            &mut world,
            &ui,
            &Frame {
                extent: TARGET,
                dpi: 1.0,
                tick: 0,
                hz: 60,
                play: gg_editor::Play::Stopped,
                input: None,
                typed: "",
                passes: &[],
                memory: MemoryUse::default(),
                reload: None,
                draw_cursor: true,
                save_path: "target/editor/test.ggsv",
                title: "gg — <no project>",
                // The mode under test, and the only field that decides it.
                project: None,
                projects: &projects,
                maximized: false,
            },
        );
        opened = commands.open.or(opened);
    }
    assert_eq!(opened, Some(1), "the picker did not report the row clicked");
    // The session is over as well: a shell is built around the dylib it was
    // pointed at, so the next project is the next session.
    assert_eq!(
        editor.take_window_command(),
        Some(gg_editor::WindowCommand::Close),
        "a pick left the session running"
    );
    // And the world is untouched — the picker is host UI over an empty world,
    // and picking is not an edit.
    assert_eq!(world.len(), 0);
    assert_eq!(editor.tally(), (0, 0));
}

/// §6 M16 item 5: the panel takes typed text, and only once something asked for
/// it.
///
/// The focus half is the load-bearing one. `Editor::wants_text` is what a host
/// asks between ticks to decide whether a character is worth recording at all,
/// so a field that took keystrokes it was never given would put every `W` a
/// player pressed into the replay's text channel — beside the `move_forward`
/// the same key already recorded as a verb.
#[test]
fn the_prompt_takes_typed_text_only_once_a_click_has_focused_it() {
    let mut world = World::new();
    let mut editor = placed(TARGET);
    // The agent pane is the fourth tab of the group under the viewport, so it
    // has no body until its tab is up — and no prompt field to aim at.
    let tab = aim::tab(&editor, Pane::Agent).expect("the agent pane has a tab");
    let mut script = frames(&[Act::To(tab), Act::Settle(3), Act::Click], CLICK, X, Y);
    let mut focus = Vec::new();
    // Typed throughout, including every tick before the click lands: the
    // characters are offered and the field must refuse them until it is the
    // thing keys are going to.
    let step = |editor: &mut Editor, world: &mut World, frame: &InputFrame, typed: &str| {
        let ui = Tick {
            motion: (frame.axes[X.index()], frame.axes[Y.index()]),
            primary: frame.pressed(CLICK),
            advance_focus: false,
            // `host` appends no `ui_press` and `Editor::tick` clears the one a
            // *game* may declare (§6 M81) — this had said the host's silence
            // settled it, which is the belief that hid the leak.
            activate: false,
            scroll: 0,
        };
        editor.tick(world, &ui, &typing(typed));
    };
    for frame in &script {
        step(&mut editor, &mut world, frame, "no");
        focus.push(editor.wants_text());
    }
    assert!(
        focus.iter().all(|f| !f),
        "nothing was focused, yet keys were wanted"
    );
    assert_eq!(
        editor.prompt(),
        "",
        "the field took characters while the game had the keyboard"
    );

    // Aimed only now: `prompt_field` is a function of a body that did not exist
    // a moment ago, which is the same reason a gizmo is aimed mid-script.
    let target = aim::prompt(&editor).expect("the agent pane is up");
    script = gg_editor::session::frames_from(
        tab,
        &[Act::To(target), Act::Settle(3), Act::Click],
        CLICK,
        X,
        Y,
    );
    for frame in &script {
        step(&mut editor, &mut world, frame, "");
    }
    assert!(editor.wants_text(), "a click on the field did not focus it");

    // And now it does take them, in the order they were typed and across ticks.
    for typed in ["why ", "so ", "slow?"] {
        step(&mut editor, &mut world, &InputFrame::default(), typed);
    }
    assert_eq!(editor.prompt(), "why so slow?");
    // Host state to the end: a prompt is not an edit, and nothing about it is in
    // the world or in the canonical hash (§4.2.1).
    assert_eq!(world.len(), 0);
    assert_eq!(editor.tally(), (0, 0));
}

/// A frame carrying `typed`, for the test above.
fn typing(typed: &str) -> Frame<'_> {
    Frame {
        extent: TARGET,
        dpi: 1.0,
        tick: 0,
        hz: 60,
        play: gg_editor::Play::Stopped,
        input: None,
        typed,
        passes: &[],
        memory: MemoryUse::default(),
        reload: None,
        draw_cursor: true,
        save_path: "target/editor/test.ggsv",
        title: "gg — test",
        project: Some("test"),
        projects: &[],
        maximized: false,
    }
}

// ---------------------------------------- §6 M20 item 10: flat authoring ----

/// Demo 11's framing, in shape: an orthographic eye out of the playfield plane
/// and a wide slab at the origin — the level an operator is actually authoring
/// when the gizmo has to work (§6 M20).
fn flat_scene() -> (World, gg_ecs::Entity) {
    let mut world = World::new();
    let eye = world.spawn();
    world
        .insert(
            eye,
            gg_ecs::boundary::Eye::flat(sim::DVec3::new(0.0, 0.0, 14.0), 4.5),
        )
        .unwrap();
    let slab = world.spawn();
    world
        .insert(
            slab,
            gg_ecs::boundary::Renderable::boxed(
                sim::DVec3::ZERO,
                sim::Vec3::new(3.0, 0.5, 0.5),
                0x0080_c0ff,
            ),
        )
        .unwrap();
    (world, slab)
}

/// Pick the slab and hand back the editor with its arms up.
fn flat_picked() -> (World, gg_ecs::Entity, Editor, (f32, f32)) {
    let editor = placed(TARGET);
    let middle = aim::nowhere(&editor).expect("the game is up");
    let (mut world, slab) = flat_scene();
    let mut run = Editor::new(None);
    let pick = [Act::To(middle), Act::Settle(4), Act::Click, Act::Settle(3)];
    stopped(&mut world, &mut run, &frames(&pick, CLICK, X, Y));
    assert_eq!(run.selected(), Some(slab), "the click picked the slab");
    (world, slab, run, middle)
}

/// A point `t` of the way from the selection's centre out along an arm.
fn along(centre: (f32, f32), tip: (f32, f32), t: f32) -> (f32, f32) {
    (
        centre.0 + (tip.0 - centre.0) * t,
        centre.1 + (tip.1 - centre.1) * t,
    )
}

fn box_of(world: &World, entity: gg_ecs::Entity) -> gg_ecs::boundary::Renderable {
    *world.get::<gg_ecs::boundary::Renderable>(entity).unwrap()
}

/// §6 M20 item 10: the **arm** is the grab target, not the dot at its end.
///
/// Through M20 only a five-unit pad at the tip was hit-tested, so a press on the
/// visible length of a handle fell through to the pick underneath and the gizmo
/// read as broken — which is how it read at the desk. Both halves are asserted:
/// a press most of the way along the arm drags, and one near the centre still
/// does not, because that is where all three arms meet and no answer to "which
/// axis" would be right.
#[test]
fn a_press_along_an_arm_drags_it_and_one_at_the_meeting_point_does_not() {
    let (mut world, slab, mut run, middle) = flat_picked();
    let handle = run.handle(0).expect("the world-X handle is up");
    let reach = (handle.0 - middle.0) * 5.0;
    let grab = along(middle, handle, 0.7);
    let drag = [
        Act::To(grab),
        Act::Settle(4),
        Act::Drag((grab.0 + reach, grab.1)),
        Act::Settle(3),
    ];
    stopped(
        &mut world,
        &mut run,
        &gg_editor::session::frames_from(middle, &drag, CLICK, X, Y),
    );
    let moved = box_of(&world, slab).position;
    assert!(
        moved.x > 0.0,
        "a press on the arm did not drag it: {moved:?}"
    );
    assert_eq!((moved.y, moved.z), (0.0, 0.0), "it moved another axis");

    // And the exclusion at the centre holds: the same gesture from inside the
    // meeting point is a *pick*, not a drag.
    let (mut world, slab, mut run, middle) = flat_picked();
    let handle = run.handle(0).expect("the world-X handle is up");
    let inner = along(middle, handle, 0.15);
    let drag = [
        Act::To(inner),
        Act::Settle(4),
        Act::Drag((inner.0 + reach, inner.1)),
        Act::Settle(3),
    ];
    stopped(
        &mut world,
        &mut run,
        &gg_editor::session::frames_from(middle, &drag, CLICK, X, Y),
    );
    assert_eq!(
        box_of(&world, slab).position,
        sim::DVec3::ZERO,
        "a press between the arms dragged one of them"
    );
}

/// The chip cycles the three tools. A mode reachable only by a key is a mode an
/// operator cannot see the state of, which is why this is a control and not
/// only `host::verb::TOOL`.
#[test]
fn the_tool_chip_cycles_move_scale_turn_and_wraps() {
    let editor = placed(TARGET);
    let chip = aim::tool(&editor).expect("the game is up");
    let (mut world, _) = flat_scene();
    let mut run = Editor::new(None);
    assert_eq!(run.tool(), gg_editor::Tool::Move, "it opens on move");
    let mut at = (0.0, 0.0);
    for want in [
        gg_editor::Tool::Scale,
        gg_editor::Tool::Turn,
        gg_editor::Tool::Move,
    ] {
        let click = [Act::To(chip), Act::Settle(4), Act::Click, Act::Settle(2)];
        stopped(
            &mut world,
            &mut run,
            &gg_editor::session::frames_from(at, &click, CLICK, X, Y),
        );
        at = chip;
        assert_eq!(run.tool(), want);
    }
}

/// The scale tool writes `half_extent` and nothing else, in whole steps of the
/// same grain the nudge bar uses — a level of boxes is authored by resizing
/// them, and through M20 the only way to was the inspector's `+`.
#[test]
fn the_scale_tool_grows_one_axis_in_whole_steps_and_moves_nothing() {
    let editor = placed(TARGET);
    let chip = aim::tool(&editor).expect("the game is up");
    let (mut world, slab, mut run, middle) = flat_picked();
    let was = box_of(&world, slab);
    let to_scale = [Act::To(chip), Act::Settle(4), Act::Click, Act::Settle(2)];
    stopped(
        &mut world,
        &mut run,
        &gg_editor::session::frames_from(middle, &to_scale, CLICK, X, Y),
    );
    assert_eq!(run.tool(), gg_editor::Tool::Scale);

    let handle = run.handle(0).expect("the world-X handle is up");
    let grab = along(middle, handle, 0.8);
    let out = (grab.0 + (handle.0 - middle.0) * 4.0, grab.1);
    let drag = [
        Act::To(grab),
        Act::Settle(4),
        Act::Drag(out),
        Act::Settle(3),
    ];
    stopped(
        &mut world,
        &mut run,
        &gg_editor::session::frames_from(chip, &drag, CLICK, X, Y),
    );
    let now = box_of(&world, slab);
    let grew = f64::from(now.half_extent.x - was.half_extent.x);
    assert!(grew > 0.0, "the scale drag resized nothing: {now:?}");
    let grain = gg_editor::STEPS[1];
    assert_eq!(grew / grain, (grew / grain).round(), "unquantized: {grew}");
    assert_eq!(now.position, was.position, "a resize moved it");
    assert_eq!(
        (now.half_extent.y, now.half_extent.z),
        (was.half_extent.y, was.half_extent.z),
        "it resized another axis"
    );
}

/// The turn tool writes `rotation`, in whole [`gg_editor::ANGLES`] degrees about
/// the arm's **world** axis, and touches neither position nor extent.
#[test]
fn the_turn_tool_rotates_about_the_arms_world_axis_in_whole_degrees() {
    let editor = placed(TARGET);
    let chip = aim::tool(&editor).expect("the game is up");
    let (mut world, slab, mut run, middle) = flat_picked();
    let was = box_of(&world, slab);
    let mut at = middle;
    for _ in 0..2 {
        let click = [Act::To(chip), Act::Settle(4), Act::Click, Act::Settle(2)];
        stopped(
            &mut world,
            &mut run,
            &gg_editor::session::frames_from(at, &click, CLICK, X, Y),
        );
        at = chip;
    }
    assert_eq!(run.tool(), gg_editor::Tool::Turn);

    let handle = run.handle(0).expect("the world-X handle is up");
    let grab = along(middle, handle, 0.8);
    let out = (grab.0 + (handle.0 - middle.0) * 4.0, grab.1);
    let drag = [
        Act::To(grab),
        Act::Settle(4),
        Act::Drag(out),
        Act::Settle(3),
    ];
    stopped(
        &mut world,
        &mut run,
        &gg_editor::session::frames_from(chip, &drag, CLICK, X, Y),
    );
    let now = box_of(&world, slab);
    assert_ne!(now.rotation, was.rotation, "the turn drag rotated nothing");
    assert_eq!(now.position, was.position, "a turn moved it");
    assert_eq!(now.half_extent, was.half_extent, "a turn resized it");
    // Whole steps of the coarse grain, or the value is one no repeated act
    // could reach — the same claim the move tool's quantization makes.
    let grain = gg_editor::ANGLES[1] * core::f64::consts::PI / 180.0;
    let landed = (1..=48).any(|k| {
        let want = sim::DQuat::from_axis_angle(sim::DVec3::X, f64::from(k) * grain);
        (want.dot(now.rotation).abs() - 1.0).abs() < 1e-9
    });
    assert!(landed, "the turn was not quantized: {:?}", now.rotation);
}

/// §6 M20 item 10: a spawn lands at the depth the operator is working at.
///
/// Two rules, and the second is the one demo 11 needed. With something selected
/// a new box arrives in *its* plane — asserted against a box seventeen metres
/// out, so a build that kept `SPAWN_AT` would land it more than three times
/// nearer. With nothing selected under a **flat** eye the plane is the scene's:
/// an orthographic camera's distance frames nothing, so a fixed number of metres
/// in front of one is a number with no meaning, and over demo 11 it put every
/// new platform nine metres out of the level and in front of the player.
#[test]
fn a_spawn_lands_in_the_plane_being_worked_in_rather_than_at_a_fixed_distance() {
    let editor = placed(TARGET);
    let middle = aim::nowhere(&editor).expect("the game is up");
    let click = |at: (f32, f32)| [Act::To(at), Act::Settle(4), Act::Click, Act::Settle(2)];

    // Perspective, with a box well past `SPAWN_AT` picked out of the viewport.
    let mut world = World::new();
    let eye = world.spawn();
    world.insert(eye, gg_ecs::boundary::Eye::ORIGIN).unwrap();
    let far = world.spawn();
    world
        .insert(
            far,
            gg_ecs::boundary::Renderable::boxed(
                sim::DVec3::new(0.0, 0.0, -17.0),
                sim::Vec3::splat(0.5),
                0x00ff_8000,
            ),
        )
        .unwrap();
    let mut run = Editor::new(None);
    let acts: Vec<Act> = [
        click(middle).as_slice(),
        click(aim::spawn(&editor).expect("the tree is up")).as_slice(),
    ]
    .concat();
    stopped(&mut world, &mut run, &frames(&acts, CLICK, X, Y));
    let made = run.selected().expect("a spawn selects what it made");
    let at = world
        .get::<gg_ecs::boundary::Renderable>(made)
        .expect("a spawn makes something the host can draw")
        .position;
    assert!(
        (at.z + 17.0).abs() < 1e-9,
        "the spawn ignored the selection: {at:?}"
    );

    // Flat, with nothing selected: the level's own plane, which is z = 0 here
    // because the eye is fourteen metres out of it.
    let (mut world, _slab) = flat_scene();
    let mut run = Editor::new(None);
    stopped(
        &mut world,
        &mut run,
        &frames(
            &click(aim::spawn(&editor).expect("the tree is up")),
            CLICK,
            X,
            Y,
        ),
    );
    let made = run.selected().expect("a spawn selects what it made");
    let at = world
        .get::<gg_ecs::boundary::Renderable>(made)
        .unwrap()
        .position;
    assert!(
        at.z.abs() < 1e-9,
        "the spawn landed out of the plane: {at:?}"
    );
}

/// §6 M61's render pane: a click on a view row is the CVar, and clicking the
/// live row puts the picture back.
///
/// The two arms fail in opposite directions. Without the first, the pane is a
/// list of names that does nothing; without the second, an operator who reached
/// `shadow.2` has to walk the table back to `off` — and the row a scrolled list
/// puts under the pointer is not the row the script aimed at, which is why the
/// same aim is re-read after every click rather than cached.
#[test]
fn the_render_pane_picks_a_view_and_puts_it_back() {
    use gg_render::cvars::{DEBUG_VIEW, DEBUG_VIEWS};

    let (mut world, _) = world();
    let mut editor = placed(TARGET);
    // The pane is a tab of the group `cvars` opens on, so it has to be raised
    // before it has a body to aim into — the same click an operator makes.
    let raise = [
        Act::To(aim::tab(&editor, Pane::Render).expect("the render tab")),
        Act::Settle(4),
        Act::Click,
        Act::Settle(2),
    ];
    stopped(&mut world, &mut editor, &frames(&raise, CLICK, X, Y));
    assert_eq!(DEBUG_VIEW.int(), 0, "the editor opens showing the picture");

    // Row 4 is `ao`, and it is named rather than numbered here: an index that
    // silently became `shadow.1` is exactly the drift this asserts against.
    let ao = DEBUG_VIEWS
        .iter()
        .position(|name| *name == "ao")
        .expect("the table has an occlusion view");
    let row = |editor: &Editor, i: usize| {
        [
            Act::To(aim::view(editor, i).expect("the view row is in the column")),
            Act::Settle(4),
            Act::Click,
            Act::Settle(2),
        ]
    };
    // `frames_from` and not `frames`: a stream carries motion, so a second
    // script starting over at the origin aims at the sum of the two.
    let input = frames_from(editor.pointer(), &row(&editor, ao), CLICK, X, Y);
    stopped(&mut world, &mut editor, &input);
    assert_eq!(DEBUG_VIEW.int(), ao as i64, "the row did not set the view");

    // The same row again is off, not a no-op.
    let input = frames_from(editor.pointer(), &row(&editor, ao), CLICK, X, Y);
    stopped(&mut world, &mut editor, &input);
    assert_eq!(DEBUG_VIEW.int(), 0, "the live row did not turn itself off");

    // And a row the aim refuses is one the column has no room for, never a
    // silent hit on its neighbour.
    assert!(aim::view(&editor, DEBUG_VIEWS.len() + 40).is_none());
}

/// `view → <pane>` closes a pane and brings it back, driven as clicks rather
/// than by calling `toggle_pane` (§6 M61).
///
/// The unit test beside `menu_action` proves the mapping; this proves the
/// *geometry* — that the row an operator's pointer lands on is the row the
/// handler reads. A menu whose items drifted one row down would toggle the
/// pane above the one it names, and no assertion about the table would notice.
#[test]
fn the_view_menu_toggles_the_pane_it_names() {
    let (mut world, _) = world();
    let mut editor = placed(TARGET);
    for pane in [Pane::Assets, Pane::Inspector, Pane::Render] {
        for want in [false, true] {
            let (title, item) = aim::pane_toggle(&editor, pane).expect("view → pane");
            let acts = [
                Act::To(title),
                Act::Settle(4),
                Act::Click,
                Act::To(item),
                Act::Settle(4),
                Act::Click,
                Act::Settle(2),
            ];
            let input = frames_from(editor.pointer(), &acts, CLICK, X, Y);
            stopped(&mut world, &mut editor, &input);
            assert_eq!(
                editor.pane_docked(pane),
                want,
                "{} did not go {}",
                pane.title(),
                if want { "back" } else { "away" }
            );
        }
    }
}
