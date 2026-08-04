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
use gg_editor::session::{Act, aim, frames, script};
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
}

/// Drive `frames` through the editor, playing the shell's part: it owns the
/// play state and the save, exactly as `gg-runtime` does.
fn run(target: (u32, u32), input: &[InputFrame]) -> Run {
    let (mut world, camera) = world();
    let mut editor = Editor::new(None);
    let (mut playing, mut steps, mut saves) = (Vec::new(), 0, 0);
    let mut running = true;
    for (tick, frame) in input.iter().enumerate() {
        let ui = Tick {
            motion: (frame.axes[X.index()], frame.axes[Y.index()]),
            primary: frame.pressed(CLICK),
            advance_focus: false,
            // Exactly what `Tick::from_input` derives from the same two verbs.
            scroll: i32::from(frame.pressed(UP)) - i32::from(frame.pressed(DOWN)),
        };
        let commands: Commands = editor.tick(
            &mut world,
            &ui,
            &Frame {
                extent: target,
                dpi: 1.0,
                tick: tick as u64,
                playing: running,
                passes: &[],
                memory: MemoryUse::default(),
                draw_cursor: true,
                save_path: "target/editor/test.ggsv",
                title: "gg — test",
                maximized: false,
            },
        );
        if let Some(want) = commands.playing {
            running = want;
            playing.push(want);
        }
        steps += u32::from(commands.step);
        saves += u32::from(commands.save);
    }
    Run {
        world,
        camera,
        editor,
        playing,
        steps,
        saves,
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
    assert_eq!(run.editor.selected(), Some(run.camera), "tree row 1");
    assert_eq!(
        position(&run),
        sim::DVec3::new(8.0, 1.0, -30.0),
        "four at 1.0, then +10 and -10 at the coarse step"
    );
    assert_eq!(run.playing, vec![false, true, false], "pause, play, pause");
    assert_eq!(run.steps, 2, "two single ticks while paused");
    assert_eq!(run.saves, 1);
    assert_eq!(run.editor.tally(), (6, 1), "six nudges, one save");
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
    let perf_group = before.pane_body(Pane::Perf);
    let input = frames(&script(&before), CLICK, X, Y);
    let mut run = run(TARGET, &input);
    run.editor.place(TARGET, 1.0);

    assert_ne!(
        run.editor.seams()[0].rect,
        seam,
        "the seam drag moved nothing"
    );
    assert_ne!(
        run.editor.pane_body(Pane::Perf),
        perf_group,
        "perf did not re-dock"
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
    // The middle of the game pane: the one rectangle with no widget in it.
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
