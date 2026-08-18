//! Lights in the viewport (§6 M72), driven through the whole editor.
//!
//! The report was one sentence — *I can't tell where the light is supposed to be
//! and where its not* — and it has three parts, which are the three tests here:
//! a light is **drawn** where it is, it can be **clicked**, and the gizmo
//! **moves** it. Every one of them was false before this milestone: the pick
//! queried `Renderable` alone, so a light was reachable only as a row in the
//! tree, and the row gave you three numbers and no picture.
//!
//! In process rather than through the shell, for `session.rs`'s reason: what is
//! under test is the editor's own arithmetic, and the shell's half of it is
//! `xtask reload --editor`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::{Eye, Light, Renderable};
use gg_editor::session::{Act, aim, frames, frames_from};
use gg_editor::{Editor, Frame, Play};
use gg_input::{ActionId, AxisId, InputFrame};
use gg_math::sim;
use gg_rhi::MemoryUse;
use gg_ui::router::Tick;

const CLICK: ActionId = ActionId::new(1);
const X: AxisId = AxisId::new(5);
const Y: AxisId = AxisId::new(6);
const TARGET: (u32, u32) = (1920, 1080);

/// Where the lamp is: straight ahead of an eye at the origin, so its pane
/// position is the viewport's centre and no test here has to reimplement the
/// projection to aim at it.
const LAMP_AT: sim::DVec3 = sim::DVec3::new(0.0, 0.0, -10.0);
/// Its reach, and the number the box a naive pick would test against is made of
/// — six metres wide, which covers the body below.
const LAMP_RANGE: f32 = 6.0;

/// An eye at the origin looking down -Z, a lamp ten metres ahead of it, and a
/// box off to one side and *inside the lamp's range*.
fn world() -> (World, gg_ecs::Entity, gg_ecs::Entity) {
    let mut world = World::new();
    world.register::<Eye>().unwrap();
    world.register::<Light>().unwrap();
    world.register::<Renderable>().unwrap();

    let eye = world.spawn();
    world
        .insert(eye, Eye::at(sim::DVec3::ZERO, 0.0, 0.0))
        .unwrap();
    let lamp = world.spawn();
    world
        .insert(lamp, Light::point(LAMP_AT, 0x00ff_c060, 5.0, LAMP_RANGE))
        .unwrap();
    let body = world.spawn();
    world
        .insert(
            body,
            Renderable::boxed(
                sim::DVec3::new(3.0, 0.0, -9.0),
                sim::Vec3::splat(0.5),
                0x00c0_c0c0,
            ),
        )
        .unwrap();
    (world, lamp, body)
}

fn placed() -> Editor {
    let mut editor = Editor::new(None);
    editor.place(TARGET, 1.0);
    editor
}

/// The shell's part, minus the transport: this milestone's subject is all
/// **Stopped**, which is where the viewport belongs to the operator.
fn drive(editor: &mut Editor, world: &mut World, input: &[InputFrame], from: u64) {
    for (i, frame) in input.iter().enumerate() {
        let ui = Tick {
            motion: (frame.axes[X.index()], frame.axes[Y.index()]),
            primary: frame.pressed(CLICK),
            advance_focus: false,
            scroll: 0,
        };
        editor.tick(
            world,
            &ui,
            &Frame {
                extent: TARGET,
                dpi: 1.0,
                tick: from + i as u64,
                hz: 60,
                play: Play::Stopped,
                input: None,
                typed: "",
                passes: &[],
                memory: MemoryUse::default(),
                reload: None,
                draw_cursor: true,
                save_path: "target/editor/markers.ggsv",
                title: "gg — markers",
                project: Some("test"),
                projects: &[],
                maximized: false,
            },
        );
    }
}

#[test]
fn a_click_in_the_viewport_selects_a_light() {
    let (mut world, lamp, body) = world();
    let mut editor = placed();
    let at = aim::viewport(&editor).expect("the game pane is docked by default");
    drive(
        &mut editor,
        &mut world,
        &frames(&[Act::To(at), Act::Settle(3), Act::Click], CLICK, X, Y),
        0,
    );
    assert_eq!(editor.selected(), Some(lamp));
    // And it is the marker that was hit rather than the range: the box sits
    // three metres off the axis and well inside six metres of the lamp, so a
    // pick testing the reach would have answered the lamp for a click on the
    // box too — which is every click in a lit room.
    assert_ne!(editor.selected(), Some(body));
}

#[test]
fn the_gizmo_moves_the_light_and_not_a_renderable() {
    let (mut world, lamp, _) = world();
    let mut editor = placed();
    let at = aim::viewport(&editor).unwrap();
    let select = frames(
        &[Act::To(at), Act::Settle(3), Act::Click, Act::Settle(2)],
        CLICK,
        X,
        Y,
    );
    let ticks = select.len() as u64;
    drive(&mut editor, &mut world, &select, 0);
    assert_eq!(editor.selected(), Some(lamp));

    // The Y arm, asked for rather than derived — its position is a property of
    // the world and the camera (`Editor::handle`). That it exists at all is
    // half the claim: through M71 a selection with no `Renderable` grew no arms.
    let arm = editor.handle(1).expect("a lamp has a move gizmo");
    let up = (arm.0, arm.1 - 40.0);
    drive(
        &mut editor,
        &mut world,
        &frames_from(
            at,
            &[Act::To(arm), Act::Settle(3), Act::Drag(up)],
            CLICK,
            X,
            Y,
        ),
        ticks,
    );

    let moved = *world.get::<Light>(lamp).unwrap();
    assert!(
        moved.position.y > LAMP_AT.y,
        "dragging the Y arm up moved the lamp to {:?}",
        moved.position
    );
    // The other two axes and the reach are untouched — a move is not a resize,
    // and `Light::range` shares the placement's half-extent with nothing else.
    assert_eq!(moved.position.x, LAMP_AT.x);
    assert_eq!(moved.position.z, LAMP_AT.z);
    assert_eq!(moved.range, LAMP_RANGE);
    assert_eq!(editor.tally().0, 1, "one gesture is one edit");
}

#[test]
fn the_marker_is_drawn_where_the_light_is_and_the_knob_removes_it() {
    /// The lamp's own ink as the marker draws it — `0x00RRGGBB` with the UI's
    /// opaque alpha. Nothing else in the editor is this colour, which is what
    /// makes the vertices below identifiable without a picture: the marker is
    /// drawn in the colour the light *emits*, so four lamps in a room are four
    /// distinguishable crosses.
    const INK: u32 = 0xffff_c060;

    let marker = |on: bool| {
        // `register` is idempotent and is what claims the name — `Editor::new`
        // calls it too, but the knob is read before the first tick here.
        gg_editor::register().ok();
        gg_core::cvar::find("d.editor_markers")
            .expect("registered by `gg_editor::register`")
            .set_bool(on);
        let (mut world, _, _) = world();
        let mut editor = placed();
        drive(
            &mut editor,
            &mut world,
            &frames(&[Act::Settle(3)], CLICK, X, Y),
            0,
        );
        assert_eq!(editor.selected(), None, "nothing has been clicked");
        let at: Vec<[f32; 2]> = editor
            .vertices()
            .iter()
            .filter(|v| v.color == INK)
            .map(|v| v.pos)
            .collect();
        let rect = editor.viewport_rect();
        let centre = (
            rect.x as f32 + rect.width as f32 * 0.5,
            rect.y as f32 + rect.height as f32 * 0.5,
        );
        (at, centre)
    };

    let (drawn, centre) = marker(true);
    assert!(!drawn.is_empty(), "an unselected lamp drew no marker");
    // Where, not merely whether. The lamp sits dead ahead of the eye, so its
    // pane position is the viewport's centre — and every vertex of the cross is
    // within its own arm of it. A marker projected through a different camera
    // than the picture, or hung off the wrong field, lands somewhere else and
    // fails here rather than looking slightly off in a screenshot.
    let arm = 8.0 * gg_editor::ui_scale(TARGET, 1.0);
    for pos in &drawn {
        assert!(
            (pos[0] - centre.0).abs() <= arm && (pos[1] - centre.1).abs() <= arm,
            "a marker vertex at {pos:?} is not on the lamp at {centre:?}"
        );
    }

    let (gone, _) = marker(false);
    assert!(gone.is_empty(), "`d.editor_markers 0` still drew {gone:?}");
    // Put it back for whatever runs next in this process — the registry is a
    // global, and a test that leaves a knob moved is a test that decides the
    // next one's answer.
    gg_core::cvar::find("d.editor_markers")
        .unwrap()
        .set_bool(true);
}
