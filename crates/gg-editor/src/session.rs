//! The scripted editor session, as input frames (§6 M15's fourth exit row).
//!
//! Authored here rather than in `xtask` for demo 07's reason: the clicks are
//! against *this* crate's layout, so a pane that moves moves the script with it
//! instead of silently missing. What `xtask` supplies is the verb ids, because
//! those belong to whichever game the editor was opened over (§4.7) — the same
//! four names resolve to different indices in different verb lists, and
//! hard-coding them here would bind the session to one demo.
//!
//! Every coordinate below comes off a **resolved** [`Editor`] by arithmetic.
//! None is a literal, and since §6 M15.1 none is a constant either: the layout is
//! the operator's, so a script has to ask where a pane went rather than assume.
//! A caller therefore [`Editor::place`]s the editor at the extent it will drive
//! it at, and aims after that.
//!
//! The seam drag is deliberately the **last** act. Everything before it aims at
//! rectangles resolved once up front, and dragging a seam is precisely the thing
//! that invalidates them.

use crate::{Editor, Pane, panels};
use gg_input::{AXIS_SCALE, MAX_AXES};
use gg_input::{ActionId, AxisId, InputFrame};
use gg_ui::draw::Rect;

/// One step of a session.
#[derive(Clone, Copy, Debug)]
pub enum Act {
    /// Glide the pointer to a logical position.
    To((f32, f32)),
    /// Hold still for `n` ticks — a press must find its widget already hovered,
    /// because hover resolves against the previous frame (§4.9's one-frame lag).
    Settle(u32),
    /// Press and release over whatever is under the pointer.
    Click,
    /// Press, glide to a position while held, and release there — the gesture a
    /// seam and a re-dock are both made of.
    Drag((f32, f32)),
}

fn centre(rect: Rect) -> (f32, f32) {
    (rect.x + rect.w * 0.5, rect.y + rect.h * 0.5)
}

/// Where the session aims, against a placed [`Editor`].
///
/// Every one is the centre of a rectangle the panels declare, computed the same
/// way twice. `None` means the pane is not up, which a caller should treat as a
/// script that no longer describes the layout rather than as a click to skip.
pub mod aim {
    use super::{Editor, Pane, Rect, centre, panels};

    /// The title bar's `i`-th transport button, in declaration order: play,
    /// step, stop.
    #[must_use]
    pub fn toolbar(editor: &Editor, i: usize) -> (f32, f32) {
        centre(editor.transport(i))
    }

    /// Play/pause.
    #[must_use]
    pub fn play(editor: &Editor) -> (f32, f32) {
        toolbar(editor, 0)
    }

    /// Advance one tick.
    #[must_use]
    pub fn step(editor: &Editor) -> (f32, f32) {
        toolbar(editor, 1)
    }

    /// Leave play mode, restoring the world captured when it was entered (§6
    /// M15.2).
    #[must_use]
    pub fn stop(editor: &Editor) -> (f32, f32) {
        toolbar(editor, 2)
    }

    /// The `i`-th menu's title in the strip — clicking it drops the menu down.
    #[must_use]
    pub fn menu(editor: &Editor, i: usize) -> Option<(f32, f32)> {
        let mut menus = gg_ui::menu::MenuBar::default();
        editor.menus_into(&mut menus, None);
        menus.titles().get(i).copied().map(centre)
    }

    /// The `item`th item of the `menu`th menu, aimed at *as if* that menu were
    /// down — so a script can aim before it clicks, which is the order a script
    /// is built in (`gg_ui::menu`).
    #[must_use]
    pub fn menu_item(editor: &Editor, menu: usize, item: usize) -> Option<(f32, f32)> {
        let mut menus = gg_ui::menu::MenuBar::default();
        editor.menus_into(&mut menus, Some(menu));
        menus.items().get(item).copied().map(centre)
    }

    /// Where `file → save` is: the title, then the item.
    #[must_use]
    pub fn save(editor: &Editor) -> Option<((f32, f32), (f32, f32))> {
        Some((menu(editor, 0)?, menu_item(editor, 0, 0)?))
    }

    /// The window buttons, in the platform's own order.
    #[must_use]
    pub fn window(editor: &Editor, i: usize) -> (f32, f32) {
        let buttons = panels::window_buttons(editor.bar_rect(), crate::MAC);
        centre(buttons[i.min(buttons.len() - 1)].1)
    }

    /// The `i`-th entity row *the tree is showing* — row 0 is the topmost
    /// visible one, not entity zero, since the pane scrolls (§6 M15.1).
    ///
    /// The bar's width comes off whether or not one is showing, for
    /// [`panels::lanes_in`]'s reason: a script must aim at the same pixel over a
    /// world of ten entities and a world of ten thousand.
    #[must_use]
    pub fn tree_row(editor: &Editor, i: usize) -> Option<(f32, f32)> {
        let list = panels::tree_list(editor.pane_body(Pane::Tree)?);
        Some(centre(Rect::new(
            list.x,
            list.y + i as f32 * crate::PITCH,
            list.w - gg_ui::scroll::BAR,
            crate::ROW,
        )))
    }

    /// The `i`-th button in the tree's header, counted from the right: `>` is 0,
    /// `<` is 1, then delete, duplicate and spawn (§6 M15.4 item 5).
    fn head(editor: &Editor, i: usize) -> Option<(f32, f32)> {
        let body = editor.pane_body(Pane::Tree)?;
        let head = Rect::new(body.x + 2.0, body.y + 2.0, (body.w - 4.0).max(0.0), 9.0);
        Some(centre(panels::head_button(head, i)))
    }

    /// The tree's page buttons, `<` and `>`, at the right end of its header.
    #[must_use]
    pub fn page(editor: &Editor, next: bool) -> Option<(f32, f32)> {
        head(editor, usize::from(!next))
    }

    /// New entity, in front of the camera.
    #[must_use]
    pub fn spawn(editor: &Editor) -> Option<(f32, f32)> {
        head(editor, 4)
    }

    /// Copy the selection.
    #[must_use]
    pub fn duplicate(editor: &Editor) -> Option<(f32, f32)> {
        head(editor, 3)
    }

    /// Despawn the selection.
    #[must_use]
    pub fn delete(editor: &Editor) -> Option<(f32, f32)> {
        head(editor, 2)
    }

    /// The `i`-th project row in the launcher's picker (§6 M15.1 item 4), which
    /// occupies the game pane while there is no game.
    ///
    /// Aimable blind, unlike a gizmo handle: the rows are a table this crate lays
    /// out, and which project is `i` is `project::scan`'s sorted order rather
    /// than anything about a world.
    #[must_use]
    pub fn project(editor: &Editor, i: usize) -> Option<(f32, f32)> {
        // `inset(1.0)` is the pane's border, which is what `viewport` draws into
        // and what `viewport_rect` hands the renderer — the interior, not the body.
        let list = panels::picker_list(editor.pane_body(Pane::Viewport)?.inset(1.0));
        Some(centre(Rect::new(
            list.x,
            list.y + i as f32 * crate::PITCH,
            (list.w - gg_ui::scroll::BAR).max(0.0),
            crate::ROW,
        )))
    }

    /// The agent panel's prompt field (§6 M16). Clicking it takes focus, which
    /// is what makes typed characters land there and what `Editor::wants_text`
    /// reports back to a host deciding whether to record them.
    #[must_use]
    pub fn prompt(editor: &Editor) -> Option<(f32, f32)> {
        Some(centre(
            panels::prompt_field(editor.pane_body(Pane::Agent)?).0,
        ))
    }

    /// Send whatever is in the field. Enter does the same thing and is the one a
    /// human uses; a script aims here because a verb needs a bound key and a
    /// click needs only a pixel.
    #[must_use]
    pub fn send(editor: &Editor) -> Option<(f32, f32)> {
        Some(centre(
            panels::prompt_field(editor.pane_body(Pane::Agent)?).1,
        ))
    }

    /// `edit → undo`, as the pair of clicks a menu item takes.
    #[must_use]
    pub fn undo(editor: &Editor) -> Option<((f32, f32), (f32, f32))> {
        Some((menu(editor, 1)?, menu_item(editor, 1, 0)?))
    }

    /// `edit → redo`.
    #[must_use]
    pub fn redo(editor: &Editor) -> Option<((f32, f32), (f32, f32))> {
        Some((menu(editor, 1)?, menu_item(editor, 1, 1)?))
    }

    /// The middle of the game pane — no panel widget, and since §6 M15.4 item 1
    /// a pick while the scene is stopped. What a test that must click on
    /// *nothing* needs, over a world holding nothing to pick.
    #[must_use]
    pub fn nowhere(editor: &Editor) -> Option<(f32, f32)> {
        editor.pane_body(Pane::Viewport).map(centre)
    }

    /// The `lane`-th cell of the `row`-th line of the inspector body, where row
    /// 0 is the first component's title and each field that follows takes one.
    ///
    /// `None` for a lane the pane is too narrow to show — the inspector draws
    /// only the lanes that fit, and aiming at one it does not draw would click
    /// the panel behind it.
    #[must_use]
    pub fn lane(editor: &Editor, row: usize, lane: usize) -> Option<(f32, f32)> {
        let body = editor.pane_body(Pane::Inspector)?;
        if lane >= panels::lanes_in(body) {
            return None;
        }
        let y = body.y + 14.0 + row as f32 * crate::PITCH;
        Some(centre(panels::lane_rect(body, y, lane)))
    }

    fn bar(editor: &Editor, offset: f32, width: f32) -> Option<(f32, f32)> {
        let nudge = panels::nudge_rect(editor.pane_body(Pane::Inspector)?);
        Some(centre(Rect::new(
            nudge.x + 1.0 + offset,
            nudge.y + 1.0,
            width,
            8.0,
        )))
    }

    /// Cycle the nudge step.
    #[must_use]
    pub fn grain(editor: &Editor) -> Option<(f32, f32)> {
        bar(editor, 0.0, 32.0)
    }

    /// Nudge the selected lane down.
    #[must_use]
    pub fn minus(editor: &Editor) -> Option<(f32, f32)> {
        bar(editor, 35.0, 12.0)
    }

    /// Nudge it up.
    #[must_use]
    pub fn plus(editor: &Editor) -> Option<(f32, f32)> {
        bar(editor, 49.0, 12.0)
    }

    /// The gizmo-mode chip in the viewport's corner (§6 M20 item 10).
    #[must_use]
    pub fn tool(editor: &Editor) -> Option<(f32, f32)> {
        editor
            .pane_body(Pane::Viewport)
            .map(|body| centre(panels::tool_chip(body)))
    }

    /// A pane's tab — what a click brings up and a drag re-docks.
    #[must_use]
    pub fn tab(editor: &Editor, pane: Pane) -> Option<(f32, f32)> {
        editor.tab_rect(pane).map(centre)
    }

    /// The `i`-th seam of the current layout, and the position that moves it by
    /// `by` logical units along its own axis.
    #[must_use]
    pub fn seam(editor: &Editor, i: usize, by: f32) -> Option<((f32, f32), (f32, f32))> {
        let seam = editor.seams().get(i)?;
        let at = centre(seam.rect);
        let to = match seam.axis {
            gg_ui::Axis::Horizontal => (at.0 + by, at.1),
            gg_ui::Axis::Vertical => (at.0, at.1 + by),
        };
        Some((at, to))
    }
}

/// The session §6 M15's gate replays, extended by §6 M15.1's gestures, §6
/// M15.2's stop and §6 M15.4's structural edits: pause a running game, select an
/// entity, nudge one field of it six times across two step sizes, single-step
/// twice, play and pause again, walk the instrument tabs, save out of the `file`
/// menu, **stop**, spawn an entity and pick it out of the viewport, duplicate
/// it, delete it, undo the delete — and then drag a seam and re-dock a pane,
/// which is where the layout stops being the editor's.
///
/// What is deliberately **not** here is a gizmo drag. Every act above aims at a
/// rectangle this crate declares; a handle aims *into the game*, and where it
/// lands is a property of the world and the camera at the tick it is grabbed, so
/// a blind script cannot name one (§6 M15.4's named residual). The drag is
/// covered in process, over a world the test builds, against
/// [`Editor::handle`](crate::Editor::handle).
///
/// What it deliberately does **not** touch is the title bar's drag region or its
/// window buttons: a replay that maximized the window would be a replay whose
/// own layout depends on the OS answering (§6 M15.1's Exit row — window drag,
/// resize and maximize appear nowhere in a replay).
///
/// It starts by *playing, then pausing* — two clicks on the same button. The
/// editor opens Stopped one tick in (§6 M15.2 post-close), so the first click
/// is the play edge the old editor took by itself, and the second puts the
/// session in the paused window the six nudges land in.
///
/// `editor` must already be [`Editor::place`]d at the extent the session will
/// run at.
#[must_use]
pub fn script(editor: &Editor) -> Vec<Act> {
    let mut acts = vec![
        Act::To(aim::play(editor)),
        Act::Settle(3),
        Act::Click,
        Act::Settle(3),
        Act::Click,
    ];
    // Tree rows are archetype order, not spawn order (see `crate::scan`), so
    // which entity row 1 holds is a property of the world and not of the
    // script. What the script needs is only that it holds *something* — the
    // caller's gate is what names the component the inspector then shows.
    if let Some(at) = aim::tree_row(editor, 1) {
        acts.extend([Act::To(at), Act::Settle(3), Act::Click]);
    }
    // Row 0 of the inspector body is a component title; row 1 is its first
    // field, and lane 0 of that is the field's first scalar.
    if let Some(at) = aim::lane(editor, 1, 0) {
        acts.extend([Act::To(at), Act::Settle(3), Act::Click]);
    }
    if let Some(at) = aim::plus(editor) {
        acts.push(Act::To(at));
        acts.push(Act::Settle(3));
        acts.extend([Act::Click, Act::Click, Act::Click, Act::Click]);
    }
    // A coarser step, one up and one down: the pair nets to zero, so what it
    // proves is that the step button moved the grain and not the value.
    if let Some(at) = aim::grain(editor) {
        acts.extend([Act::To(at), Act::Settle(3), Act::Click]);
    }
    if let Some(at) = aim::plus(editor) {
        acts.extend([Act::To(at), Act::Settle(3), Act::Click]);
    }
    if let Some(at) = aim::minus(editor) {
        acts.extend([Act::To(at), Act::Settle(3), Act::Click]);
    }
    // Two single ticks, then run for a while, then stop again.
    acts.extend([
        Act::To(aim::step(editor)),
        Act::Settle(3),
        Act::Click,
        Act::Click,
    ]);
    acts.extend([Act::To(aim::play(editor)), Act::Settle(3), Act::Click]);
    acts.push(Act::Settle(40));
    acts.push(Act::Click);
    // Every instrument tab, so the golden subject and the replay cover all
    // three panes that share a strip.
    for pane in [Pane::Cvars, Pane::Assets, Pane::Perf] {
        if let Some(at) = aim::tab(editor, pane) {
            acts.extend([Act::To(at), Act::Settle(3), Act::Click]);
        }
    }
    // The save is a menu item now (§6 M15.1 item 5), so it is two clicks: drop
    // `file` down, then pick `save` out of it. Which is the point of routing it
    // through the script — the gate exercises the menu without knowing there is
    // one.
    if let Some((title, item)) = aim::save(editor) {
        acts.extend([
            Act::To(title),
            Act::Settle(3),
            Act::Click,
            Act::To(item),
            Act::Settle(3),
            Act::Click,
        ]);
    }
    // §6 M15.2: leave play mode. Deliberately *after* the save, so the file on
    // disk still holds the six nudges that this then discards — which is what
    // makes "an edit made during play is discarded" checkable against two
    // artifacts rather than against a log line.
    acts.extend([Act::To(aim::stop(editor)), Act::Settle(3), Act::Click]);
    // §6 M15.4, and it runs here because every one of these needs the scene
    // *stopped*: spawn, pick what was spawned, duplicate it, delete it, and undo
    // the delete. The spawn is deliberately first, and that is what makes the
    // pick blind-authorable — a spawned entity lands a fixed distance down the
    // camera's own forward axis, so it projects to the exact centre of the game
    // pane whatever the camera is doing, which is the one thing in the scene a
    // script can aim at without knowing the world.
    for at in [
        aim::spawn(editor),
        aim::nowhere(editor),
        aim::duplicate(editor),
        aim::delete(editor),
    ]
    .into_iter()
    .flatten()
    {
        acts.extend([Act::To(at), Act::Settle(4), Act::Click]);
    }
    // And back: `edit` → `undo` returns the entity the click above deleted.
    if let Some((title, item)) = aim::undo(editor) {
        acts.extend([
            Act::To(title),
            Act::Settle(3),
            Act::Click,
            Act::To(item),
            Act::Settle(3),
            Act::Click,
        ]);
    }
    // §6 M15.1: the two gestures that move the layout itself. The seam first,
    // because a re-dock changes how many seams there are.
    if let Some((at, to)) = aim::seam(editor, 0, 24.0) {
        acts.extend([Act::To(at), Act::Settle(3), Act::Drag(to)]);
    }
    if let (Some(from), Some(onto)) = (aim::tab(editor, Pane::Perf), aim::tab(editor, Pane::Tree)) {
        acts.extend([Act::To(from), Act::Settle(3), Act::Drag(onto)]);
    }
    // A tail, or the session proves nothing about an editor that settled.
    acts.push(Act::Settle(20));
    acts
}

/// The §1 loop's clicks (§6 M16 exit row 4), in the one order a replay can
/// hold them. Asking comes *first*: while the game holds the pointer every
/// panel is unreachable, and the only recordable way it takes the pointer — a
/// press in the running viewport — is also the only way hands-on play reaches
/// the sim at all (the game gets a dead frame until then), while the way back
/// out is Escape, which is not a verb and so not in any replay.
pub struct AgentScript {
    /// Bring the agent pane up and focus its prompt, from wherever the
    /// pointer starts.
    pub focus: Vec<Act>,
    /// Where `focus` leaves the pointer — what [`frames_from`] needs next.
    pub prompt: (f32, f32),
    /// A settle the caller's typing lands in — the prompt's characters ride
    /// the replay's *text channel* rather than the action map, placed by tick
    /// against the frames `focus` produced — then send.
    pub ask: Vec<Act>,
    /// Where `ask` leaves the pointer.
    pub send: (f32, f32),
    /// The transport's play — the editor opens Stopped (§6 M15.2 post-close),
    /// and only a *running* viewport hands the pointer over — then the click
    /// into it, after which the caller's held game verbs are §1's play.
    pub play: Vec<Act>,
}

/// The loop's script against a placed editor. `editor` must also have the
/// agent pane [`Editor::raise`]d: the aims inside it are `None` while it sits
/// behind another tab. The replayed editor starts with it down, which is what
/// the first piece's tab click is for — activating a tab moves no geometry,
/// so aims taken against the raised layout land in the replay too.
#[must_use]
pub fn agent_script(editor: &Editor) -> Option<AgentScript> {
    let tab = aim::tab(editor, Pane::Agent)?;
    let prompt = aim::prompt(editor)?;
    let send = aim::send(editor)?;
    let viewport = aim::nowhere(editor)?;
    Some(AgentScript {
        focus: vec![
            Act::Settle(30),
            Act::To(tab),
            Act::Settle(3),
            Act::Click,
            Act::To(prompt),
            Act::Settle(3),
            Act::Click,
        ],
        prompt,
        ask: vec![
            Act::Settle(12),
            Act::To(send),
            Act::Settle(3),
            Act::Click,
            Act::Settle(20),
        ],
        send,
        play: vec![
            Act::To(aim::play(editor)),
            Act::Settle(3),
            Act::Click,
            Act::To(viewport),
            Act::Settle(4),
            Act::Click,
        ],
    })
}

/// Ticks a press is held, and the ones after the release.
const PRESS: u32 = 2;
const RELEASE: u32 = 6;
/// Logical units the pointer covers per tick while gliding. Slow enough that a
/// press never lands on a widget the pointer only just entered.
const GLIDE: f32 = 8.0;

/// Turn a script into recordable frames, against the verb ids the host resolved
/// [`gg_ui::boundary::verb`]'s four names to.
#[must_use]
pub fn frames(acts: &[Act], click: ActionId, x: AxisId, y: AxisId) -> Vec<InputFrame> {
    frames_from((0.0, 0.0), acts, click, x, y)
}

/// [`frames`] for a stream that is being built in more than one piece, with the
/// pointer already at `from`.
///
/// The frames carry *motion*, not positions, so a second call starting over at
/// the origin would aim every glide at the sum of the two. A caller needs this
/// exactly when the second half of a script depends on what the first half did
/// — aiming at a gizmo handle, whose position is a property of the world and the
/// camera rather than of the layout (§6 M15.4 item 3).
#[must_use]
pub fn frames_from(
    from: (f32, f32),
    acts: &[Act],
    click: ActionId,
    x: AxisId,
    y: AxisId,
) -> Vec<InputFrame> {
    let mut out = Vec::new();
    let mut at = (
        (from.0 * AXIS_SCALE as f32) as i32,
        (from.1 * AXIS_SCALE as f32) as i32,
    );
    let hold = |out: &mut Vec<InputFrame>, ticks: u32, down: bool| {
        for _ in 0..ticks {
            out.push(InputFrame {
                buttons: u64::from(down) << click.index(),
                axes: [0; MAX_AXES],
            });
        }
    };
    // The glide, as frames: motion divided by the ticks *remaining*, so the last
    // frame carries the remainder and the pointer lands exactly on the target —
    // a click one unit short names nothing. `down` rides on every frame of it,
    // which is the whole difference between a move and a drag.
    let glide = |out: &mut Vec<InputFrame>, at: &mut (i32, i32), to: &(f32, f32), down: bool| {
        let target = (
            (to.0 * AXIS_SCALE as f32) as i32,
            (to.1 * AXIS_SCALE as f32) as i32,
        );
        // Chebyshev, so no square root reaches a path that authors hashed
        // input — and the pointer arrives on both axes at once.
        let reach = (target.0 - at.0).abs().max((target.1 - at.1).abs());
        let ticks = ((reach as f32 / (GLIDE * AXIS_SCALE as f32)) as u32).max(4);
        for step in 0..ticks {
            let left = (ticks - step) as i32;
            let motion = ((target.0 - at.0) / left, (target.1 - at.1) / left);
            *at = (at.0 + motion.0, at.1 + motion.1);
            let mut axes = [0; MAX_AXES];
            axes[x.index()] = motion.0;
            axes[y.index()] = motion.1;
            out.push(InputFrame {
                buttons: u64::from(down) << click.index(),
                axes,
            });
        }
    };
    for act in acts {
        match act {
            Act::Settle(ticks) => hold(&mut out, *ticks, false),
            Act::Click => {
                hold(&mut out, PRESS, true);
                hold(&mut out, RELEASE, false);
            }
            Act::To(to) => glide(&mut out, &mut at, to, false),
            Act::Drag(to) => {
                // Pressed where it stands, dragged, released there. The press
                // has to settle first or the widget under it is still whatever
                // the previous frame declared.
                hold(&mut out, PRESS, true);
                glide(&mut out, &mut at, to, true);
                hold(&mut out, PRESS, true);
                hold(&mut out, RELEASE, false);
            }
        }
    }
    out
}

/// Frames holding `action` for `ticks`, each carrying `motion` on `axes` — a
/// camera move, and with motion a look drag (§6 M15.2 item 4).
///
/// Outside [`Act`] on purpose. Every act above aims at a rectangle this crate
/// declares, which is why they are authored here; a camera verb aims at nothing
/// at all, so what it needs is an *id*, and ids belong to whichever game the
/// editor was opened over (§4.7). A caller that resolved one composes with this
/// and keeps the script free of them.
#[must_use]
pub fn hold(
    action: ActionId,
    ticks: u32,
    axes: (AxisId, AxisId),
    motion: (i32, i32),
) -> Vec<InputFrame> {
    (0..ticks)
        .map(|_| {
            let mut frame = InputFrame {
                buttons: 1 << action.index(),
                axes: [0; MAX_AXES],
            };
            frame.axes[axes.0.index()] = motion.0;
            frame.axes[axes.1.index()] = motion.1;
            frame
        })
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const EXTENT: (u32, u32) = (1280, 720);

    fn placed() -> Editor {
        let mut editor = Editor::new(None);
        editor.place(EXTENT, 1.0);
        editor
    }

    /// The script lands where it aims. Integer fixed point accumulates exactly,
    /// so "exactly" is the assertion and not "within a unit".
    #[test]
    fn the_pointer_arrives_on_every_target() {
        let (click, x, y) = (ActionId::new(1), AxisId::new(5), AxisId::new(6));
        let editor = placed();
        let acts = script(&editor);
        let frames = frames(&acts, click, x, y);
        let fixed = |to: &(f32, f32)| {
            (
                (to.0 * AXIS_SCALE as f32) as i32,
                (to.1 * AXIS_SCALE as f32) as i32,
            )
        };
        let mut at = (0i32, 0i32);
        let mut fed = 0;
        for act in &acts {
            let (ticks, to) = match act {
                Act::Settle(n) => (*n as usize, None),
                Act::Click => ((PRESS + RELEASE) as usize, None),
                Act::To(to) => (0, Some(*to)),
                Act::Drag(to) => (PRESS as usize, Some(*to)),
            };
            fed += ticks;
            let Some(to) = to else { continue };
            let target = fixed(&to);
            while fed < frames.len() && at != target {
                at.0 += frames[fed].axes[x.index()];
                at.1 += frames[fed].axes[y.index()];
                fed += 1;
            }
            assert_eq!(at, target, "glide missed {to:?}");
            if matches!(act, Act::Drag(_)) {
                fed += (PRESS + RELEASE) as usize;
            }
        }
        assert_eq!(fed, frames.len(), "every frame belongs to an act");
    }

    /// Every aim is inside the pane that owns it — a target that drifted out of
    /// its rectangle would click on the pane behind it and still replay.
    #[test]
    fn every_target_is_inside_its_pane() {
        let editor = placed();
        let inside = |rect: Rect, at: (f32, f32), what: &str| {
            assert!(
                rect.contains(at.0, at.1),
                "{what} at {at:?} is outside {rect:?}"
            );
        };
        let bar = editor.bar_rect();
        for i in 0..panels::TOOLBAR.len() {
            inside(bar, aim::toolbar(&editor, i), "transport button");
        }
        for i in 0..crate::MENUS.len() {
            inside(
                bar,
                aim::menu(&editor, i).expect("every menu has a title"),
                "menu",
            );
        }
        for i in 0..3 {
            inside(bar, aim::window(&editor, i), "window button");
        }
        // The one aim that is deliberately *not* in the bar: an item hangs
        // below it, over whatever pane is underneath.
        let (title, item) = aim::save(&editor).expect("file → save");
        inside(bar, title, "file");
        assert!(item.1 > bar.bottom(), "save at {item:?} is inside the bar");
        let tree = editor.pane_body(Pane::Tree).expect("the tree is up");
        inside(tree, aim::tree_row(&editor, 0).unwrap(), "tree row 0");
        inside(
            tree,
            aim::tree_row(&editor, editor.per_page() - 1).unwrap(),
            "tree row last",
        );
        let inspect = editor
            .pane_body(Pane::Inspector)
            .expect("the inspector is up");
        let lanes = panels::lanes_in(inspect);
        assert_eq!(lanes, 3, "the default inspector shows a whole vector");
        for row in 0..6 {
            for lane in 0..lanes {
                inside(inspect, aim::lane(&editor, row, lane).unwrap(), "lane");
            }
        }
        for (at, what) in [
            (aim::grain(&editor), "grain"),
            (aim::minus(&editor), "minus"),
            (aim::plus(&editor), "plus"),
        ] {
            inside(inspect, at.unwrap(), what);
        }
        for pane in Pane::ALL {
            let tab = aim::tab(&editor, pane).expect("every pane has a tab");
            let rect = editor.tab_rect(pane).expect("and a rectangle");
            inside(rect, tab, pane.title());
        }
    }

    /// The click count is the gate's contract: four play/pause presses (the
    /// first starts the Stopped scene the editor opens into, §6 M15.2
    /// post-close), one tree row, one lane, six nudges, one grain, two single
    /// steps, three tabs and one save — plus §6 M15.1's two drags, which are
    /// not clicks.
    #[test]
    fn the_script_clicks_what_the_gate_expects() {
        let editor = placed();
        let acts = script(&editor);
        let clicks = acts.iter().filter(|a| matches!(a, Act::Click)).count();
        // The save is two: the `file` title, then the item under it. The single
        // one after it is §6 M15.2's stop, and the last group is §6 M15.4's
        // spawn, pick, duplicate, delete and two-click undo.
        assert_eq!(clicks, 4 + 1 + 1 + 6 + 1 + 2 + 3 + 2 + 1 + 4 + 2);
        let drags = acts.iter().filter(|a| matches!(a, Act::Drag(_))).count();
        assert_eq!(drags, 2, "a seam and a re-dock");
    }

    /// §6 M16 exit row 4: the loop's script aims inside a pane that starts
    /// behind a tab, so it declines an unraised editor rather than aiming at
    /// whatever pane is showing — and once raised, every aim is inside it.
    #[test]
    fn the_agent_script_is_authored_against_a_raised_pane() {
        let mut editor = placed();
        assert!(
            agent_script(&editor).is_none(),
            "aimed into a pane behind another tab"
        );
        editor.raise(Pane::Agent);
        let script = agent_script(&editor).expect("the pane is up");
        let body = editor.pane_body(Pane::Agent).expect("and has a body");
        assert!(
            body.contains(script.prompt.0, script.prompt.1),
            "the prompt is outside its pane"
        );
        assert!(
            body.contains(script.send.0, script.send.1),
            "send is outside its pane"
        );
        let clicks = |acts: &[Act]| acts.iter().filter(|a| matches!(a, Act::Click)).count();
        assert_eq!(clicks(&script.focus), 2, "the tab, then the field");
        assert_eq!(clicks(&script.ask), 1, "send");
        assert_eq!(
            clicks(&script.play),
            2,
            "the transport's play, then the viewport press that hands over"
        );
    }

    /// The script is the same at every extent it is placed at — the same acts
    /// in the same order, differing only in where they aim. A layout that
    /// dropped a pane at some size would show up here as a shorter script.
    #[test]
    fn the_script_has_the_same_shape_at_every_extent() {
        let shape = |extent: (u32, u32)| {
            let mut editor = Editor::new(None);
            editor.place(extent, 1.0);
            script(&editor)
                .iter()
                .map(|a| match a {
                    Act::To(_) => 'm',
                    Act::Settle(_) => 's',
                    Act::Click => 'c',
                    Act::Drag(_) => 'd',
                })
                .collect::<String>()
        };
        let at_720 = shape((1280, 720));
        for extent in [(1920, 1080), (3840, 2064), (2560, 1080)] {
            assert_eq!(shape(extent), at_720, "{extent:?}");
        }
    }
}
