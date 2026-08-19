//! The surface reaches the picture and nothing else (§6 M46).
//!
//! This is the claim a fullscreen toggle rests on, and it is load-bearing well
//! beyond that: it is why a golden rendered at 1280×720 and a replay driven at
//! the headless canvas resolve the same widget, why `xtask reload`'s legs can
//! click at all, and why the extent is absent from every hash in the tree.
//!
//! The mechanism is two lines apart in [`Ui::frame`]: the router is begun at
//! [`CANVAS`] — a constant — while the [`Fit`] is handed only to the draw list.
//! So a window twice the size is the same hit test at twice the scale, and a
//! player who alt-enters mid-game does not find their menu somewhere else.
//!
//! **Both directions, because either alone passes on a broken UI.** Invariance
//! on its own is satisfied by a UI that ignores the fit entirely and draws the
//! same pixels at every window size; the picture moving on its own is satisfied
//! by one that letterboxes the hit test along with the geometry. The pairing is
//! the gate.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::{CANVAS, Widget, state, widget_id};
use gg_ui::router::{AXIS_SCALE, Tick};
use gg_ui::{Fit, Ui};

/// Two surfaces chosen to disagree about everything a fit can disagree about:
/// scale, letterbox axis, and whether the canvas is bigger or smaller than the
/// window. A 720p window, and a 4K monitor in the other aspect.
const WINDOWED: (u32, u32) = (1280, 720);
const FULLSCREEN: (u32, u32) = (3840, 2160);
const TALL: (u32, u32) = (900, 1600);

const ROWS: u32 = 12;

fn world() -> World {
    let mut world = World::new();
    world.register::<Widget>().unwrap();
    for i in 0..ROWS {
        let entity = world.spawn();
        // Spread across the canvas, so a pointer walking it crosses several and
        // an off-by-a-scale hit test lands on a *different* one rather than on
        // nothing — the failure that would otherwise read as "no hover".
        // Contiguous rather than spaced: a gap between rows is somewhere the
        // pointer can sit hovering nothing, and this file's whole subject is
        // *which* widget a position resolves to.
        let height = (CANVAS.1 as f32 - 8.0) / f32::from(ROWS as u16);
        let w = Widget::button(
            widget_id("row").wrapping_add(u64::from(i)),
            [
                4.0,
                4.0 + f32::from(i as u16) * height,
                CANVAS.0 as f32 - 8.0,
                height,
            ],
            0xff1e_2c3c,
            0xffd8_e0e8,
            "row",
        );
        world.insert(entity, w).unwrap();
    }
    world
}

/// A pointer walking the canvas and pressing every fourth tick, so hover,
/// capture and click are all exercised rather than only position.
///
/// **It shoves into the corner and back**, and that is the load-bearing part
/// rather than thoroughness. The extent handed to the router is not compared
/// against anything — it is what the pointer *clamps* to — so a walk that stays
/// in the middle of the canvas resolves the same widgets whatever the extent,
/// and would pass on a hit test that read the surface. This one saturates: from
/// tick 120 it drives hard into the bottom-right, and where it comes to rest is
/// a statement about the extent. A first version of this file did not, planted
/// the defect, and went green (§6 M46).
fn tick_of(tick: u64) -> Tick {
    let (dx, dy) = match tick {
        0..120 => (AXIS_SCALE / 3, AXIS_SCALE / 2),
        120..200 => (AXIS_SCALE * 9, AXIS_SCALE * 5),
        _ => (-AXIS_SCALE, -AXIS_SCALE / 2),
    };
    Tick {
        motion: (dx, dy),
        primary: tick.is_multiple_of(4),
        advance_focus: tick.is_multiple_of(16),
        // Every third focus advance presses what the ring is on (§6 M74).
        activate: tick.is_multiple_of(48),
        scroll: 0,
    }
}

/// Every widget's hit-test state after `ticks` frames at `fit`, in world order.
fn run(fit: Fit, ticks: u64) -> (Vec<u32>, usize) {
    let mut ui = Ui::new().unwrap();
    let mut world = world();
    for tick in 0..ticks {
        ui.frame(&mut world, &tick_of(tick), fit);
    }
    let vertices = ui.vertices().len();
    let mut states = Vec::new();
    let query = gg_ecs::Query::<&Widget>::new().unwrap();
    world.each_ref(&query, |_, w: &Widget| states.push(w.state));
    (states, vertices)
}

/// The whole of it, and the two halves fail in opposite directions.
#[test]
fn the_surface_moves_the_picture_and_never_the_hit_test() {
    const TICKS: u64 = 240;
    let (windowed, windowed_vertices) = run(Fit::new(WINDOWED), TICKS);
    let (fullscreen, _) = run(Fit::new(FULLSCREEN), TICKS);
    let (tall, _) = run(Fit::new(TALL), TICKS);

    // The gate's own gate: a run where nothing was ever hovered or pressed
    // would satisfy every equality below while proving nothing at all.
    assert!(
        windowed
            .iter()
            .any(|s| s & (state::HOVERED | state::HELD | state::CLICKED) != 0),
        "the pointer walk never reached a widget, so this file is comparing idle states"
    );
    assert_eq!(
        windowed, fullscreen,
        "the same input resolved different widgets at 4K — the hit test is reading the surface, \
         which means a player who went fullscreen mid-game found their menu somewhere else"
    );
    assert_eq!(
        windowed, tall,
        "and the same in the other aspect, where the letterbox changes axis"
    );

    // The other direction. A UI that ignored the fit would pass everything
    // above, so the picture has to be shown to move — the geometry is in
    // surface pixels and the letterbox is what changes with the window.
    let mut ui = Ui::new().unwrap();
    let mut world = world();
    ui.frame(&mut world, &tick_of(0), Fit::new(WINDOWED));
    let small: Vec<_> = ui.vertices().to_vec();
    ui.frame(&mut world, &tick_of(1), Fit::new(FULLSCREEN));
    let large = ui.vertices();
    assert_eq!(
        small.len(),
        large.len(),
        "the same widgets should produce the same geometry, differently placed"
    );
    let corners = |vs: &[gg_render::ui::UiVertex]| vs.iter().map(|v| v.pos).collect::<Vec<_>>();
    assert_ne!(
        corners(&small),
        corners(large),
        "the picture did not move with the window, so the fit reaches nothing and the equalities \
         above are vacuous"
    );
    assert!(windowed_vertices > 0, "the frame under test drew nothing");
}

/// Which fit is extent-sensitive, and it is not the one a game is on.
///
/// §8's open row says "a game's own `Widget` hit-testing is extent-sensitive
/// with no override to apply the recorded surface through". The first half has
/// not been true since the steered pointer landed: [`Fit::new`] fits a *constant*
/// canvas, so a game's units do not move with the window. What does move is
/// [`Fit::fill`], the editor's — there the canvas is a **quotient** of the
/// surface, which is exactly why `--editor-extent` and the replay header's
/// recorded surface exist and why they are the editor's alone.
///
/// Recorded as a test rather than as a paragraph because the residual is scoped
/// by this and nothing else: a day when `Fit::new` starts returning a canvas
/// that depends on its argument is the day the scope changes, and it should
/// fail here rather than be discovered in a bug report about a click.
#[test]
fn a_games_canvas_is_a_constant_and_the_editors_is_a_quotient_of_the_window() {
    for surface in [WINDOWED, FULLSCREEN, TALL] {
        assert_eq!(
            Fit::new(surface).canvas,
            CANVAS,
            "a game's canvas moved with the window at {surface:?}"
        );
    }
    let editor = |s: (u32, u32)| Fit::fill(s, 2.0).canvas;
    assert_ne!(
        editor(WINDOWED),
        editor(FULLSCREEN),
        "the editor's canvas is the window's, so its hit test is the extent-sensitive one — if \
         this ever stops being true, §8's residual has changed shape"
    );
}
