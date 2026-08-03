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
use gg_editor::{Commands, Editor, Frame};
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
        };
        let commands: Commands = editor.tick(
            &mut world,
            &ui,
            &Frame {
                extent: target,
                tick: tick as u64,
                playing: running,
                passes: &[],
                memory: MemoryUse::default(),
                save_path: "target/editor/test.ggsv",
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
    let input = frames(&script(), CLICK, X, Y);
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

/// §6 M15's fourth exit row at the level this crate can prove it: the same tick
/// stream produces the same edits whatever the window is. The pointer is
/// integrated in canvas units precisely so this holds (§4.9).
#[test]
fn a_replayed_session_lands_the_same_clicks_at_any_window_size() {
    let input = frames(&script(), CLICK, X, Y);
    let reference = run(TARGET, &input);
    for target in [(640, 360), (1280, 720), (3840, 2160), (1024, 1024)] {
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

/// A click that lands on nothing must change nothing — otherwise the session
/// above could be passing on an edit it never aimed at.
#[test]
fn a_session_that_clicks_empty_canvas_edits_nothing() {
    // The middle of the viewport: the one rectangle with no widget in it.
    let idle = [
        Act::To((gg_editor::session::aim::tab(0).0 + 40.0, 150.0)),
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
    let pick = [Act::To(aim::tree_row(0)), Act::Settle(3), Act::Click];
    let first = run(TARGET, &frames(&pick, CLICK, X, Y));
    let page_one = first.editor.selected().expect("a row on page one");

    // `>` sits at the right end of the tree's header row.
    let next = (gg_editor::session::aim::tree_row(0).0 * 2.0 - 6.0, 20.0);
    let acts = [
        Act::To(next),
        Act::Settle(3),
        Act::Click,
        Act::To(aim::tree_row(2)),
        Act::Settle(3),
        Act::Click,
    ];
    let paged = run(TARGET, &frames(&acts, CLICK, X, Y));
    let picked = paged.editor.selected().expect("a row on page two");
    assert_ne!(picked, page_one, "the page moved under the same row index");
}
