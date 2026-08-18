//! The seven panes §6 M15.1 docks, each one a table of rectangles and clicks.
//!
//! Nothing here is retained: the layout is recomputed every tick and hit-tested
//! against the geometry the previous one declared, which is `gg_ui::router`'s
//! documented one frame of lag. That is also why panel state — selection, how
//! far a pane is scrolled, which pane is up — can live in plain fields and
//! still replay: every one of them moves only in response to a click or a wheel
//! notch, and both arrive through the action map.
//!
//! Every function here takes the rectangle it draws into. They took module
//! constants through M15, which is what made the layout the *editor's* rather
//! than the operator's — a pane whose position is a `const` cannot be dragged
//! anywhere.

use crate::chat;
use crate::scan::{Slot, read_row, write_row};
use crate::{
    ACCENT, BUTTON, CHROME, DANGER, DIM, EM, Editor, Frame, HEADER, INK, Lane, MENUS, PICKED,
    PITCH, Pane, ROW, STEPS, WindowCommand, fit_text, tail, text_width, value,
};
use gg_ui::WidgetId;
use gg_ui::draw::DrawList;
use gg_ui::draw::Rect;
use gg_ui::router::Tick;

const PLAY: WidgetId = WidgetId::new("editor.play");
const STEP: WidgetId = WidgetId::new("editor.step");
const STOP: WidgetId = WidgetId::new("editor.stop");
const MENU: WidgetId = WidgetId::new("editor.menu");
const DRAG: WidgetId = WidgetId::new("editor.titlebar");
const WINDOW: WidgetId = WidgetId::new("editor.window");
const ASK_EXPLAIN: WidgetId = WidgetId::new("editor.agent.explain");
const ASK_FIX: WidgetId = WidgetId::new("editor.agent.fix");
/// The prompt field. `pub(crate)` because [`Editor::wants_text`] answers about
/// it from outside this module, and the host asks between ticks.
pub(crate) const PROMPT: WidgetId = WidgetId::new("editor.agent.prompt");
const SEND: WidgetId = WidgetId::new("editor.agent.send");
const PREV: WidgetId = WidgetId::new("editor.tree.prev");
const NEXT: WidgetId = WidgetId::new("editor.tree.next");
const TREE_ROW: WidgetId = WidgetId::new("editor.tree.row");
const KNOB: WidgetId = WidgetId::new("editor.cvar");
const LANE: WidgetId = WidgetId::new("editor.lane");
const MINUS: WidgetId = WidgetId::new("editor.nudge.minus");
const PLUS: WidgetId = WidgetId::new("editor.nudge.plus");
const GRAIN: WidgetId = WidgetId::new("editor.nudge.step");
const SPAWN: WidgetId = WidgetId::new("editor.tree.spawn");
const COPY: WidgetId = WidgetId::new("editor.tree.duplicate");
const KILL: WidgetId = WidgetId::new("editor.tree.delete");
const TOOL: WidgetId = WidgetId::new("editor.gizmo.tool");
/// §6 M59's view picker, in the perf pane beside the passes it picks among.
const DEBUG: WidgetId = WidgetId::new("editor.debug.view");
/// §6 M61's render pane: the same picker as a *list*, and its own knob ids.
const VIEW_ROW: WidgetId = WidgetId::new("editor.render.view");

/// The three widget ids one [`Editor::knob`] row declares.
///
/// Per **pane**, not per crate: the cvars pane and the render pane can be up at
/// the same time showing the same CVar, and one id answering two hit tests is a
/// click that lands twice (§4.9).
struct Knobs {
    knob: WidgetId,
    minus: WidgetId,
    plus: WidgetId,
}

const CVAR_KNOBS: Knobs = Knobs {
    knob: KNOB,
    minus: MINUS,
    plus: PLUS,
};

const RENDER_KNOBS: Knobs = Knobs {
    knob: WidgetId::new("editor.render.cvar"),
    minus: WidgetId::new("editor.render.minus"),
    plus: WidgetId::new("editor.render.plus"),
};

/// One line of the render pane's right column.
enum Row<'a> {
    /// A group's name, in the accent. Not a control.
    Heading(&'a str),
    /// A CVar, its index in `gg_core::cvar::all()` — which is what makes its
    /// widget id stable as the pane scrolls — and the leading characters of its
    /// name the heading above it already said.
    Knob(usize, &'a gg_core::cvar::CVar, usize),
}

/// The render pane's groups: a heading and the prefix it claims. **First match
/// wins**, so `r.shadow` must precede the bare `r.` catch-all — a prefix listed
/// after one it is a prefix of would never claim a row.
const GROUPS: &[(&str, &str)] = &[
    ("ambient occlusion", "r.ao"),
    ("global illumination", "r.gi"),
    ("sun shadows", "r.shadow"),
    ("lamp shadows", "r.lamp"),
    ("debug", "r.debug"),
    ("frame", "r."),
];

/// Lay `cvars` out as the render pane's rows: every `r.*` under exactly one
/// heading, headings with nothing under them dropped.
///
/// Pure and separate from the drawing so it can be tested against a registry
/// this crate's test binary does not have to populate — the `r.*` names are the
/// shell's to register (`gg_runtime::run`), so a panel test over the live
/// registry would assert about whichever crates happened to have started.
fn knob_rows<'a>(cvars: &[&'a gg_core::cvar::CVar]) -> Vec<Row<'a>> {
    let mut rows = Vec::new();
    for (at, (heading, prefix)) in GROUPS.iter().enumerate() {
        let mut group = Vec::new();
        for (i, cvar) in cvars.iter().enumerate() {
            let claimed = GROUPS[..at]
                .iter()
                .any(|(_, earlier)| cvar.name().starts_with(earlier));
            if cvar.name().starts_with(prefix) && !claimed {
                group.push((i, *cvar));
            }
        }
        if group.is_empty() {
            continue;
        }
        rows.push(Row::Heading(heading));
        // The heading already says the prefix, so the rows drop it — but never
        // to nothing: `r.ao` under "ambient occlusion" would be a blank name, so
        // a row whose name *is* the prefix keeps everything past `r.`.
        for (i, cvar) in group {
            let trim = match cvar.name().len() > prefix.len() + 1 {
                true => prefix.len() + 1,
                false => 2,
            };
            rows.push(Row::Knob(i, cvar, trim));
        }
    }
    rows
}

/// How far in front of the camera a spawned entity lands, in metres. Far enough
/// to see whole at the default field of view, near enough that its gizmo arms
/// are a comfortable size (§6 M15.4 item 5).
const SPAWN_AT: f64 = 5.0;
/// What a spawned box is coloured, so it is obvious which one is new.
const SPAWN_INK: u32 = 0x00c8_d4e0;

/// Width the tree's header reserves for its five buttons, in logical units.
const HEAD_BUTTONS: f32 = 52.0;

/// The `i`-th button in a header strip, counted **from the right** so adding one
/// on the left of the row does not move the others. Nine units square, a unit
/// apart, which is the plate the rest of this panel is drawn in.
pub(crate) fn head_button(head: Rect, i: usize) -> Rect {
    Rect::new(head.right() - 10.0 * (i as f32 + 1.0), head.y, 9.0, 9.0)
}
/// The game's own rectangle, declared **only while the scene is stopped** (§6
/// M15.4 item 1). While it plays the viewport is the game's and this is the hole
/// it always was, which is what keeps a press there the player's rather than
/// something the editor and the game both answer.
const VIEWPORT: WidgetId = WidgetId::new("editor.viewport");
const OPEN: WidgetId = WidgetId::new("editor.project");

/// Where one world axis's tip lands in an axis widget whose arms are `arm`
/// long, as an offset from its middle — and whether that axis is coming at the
/// operator rather than going away (§6 M63).
///
/// A function of its own because it is the widget's one non-obvious line and
/// the two failures it can have are silent: a sign wrong on the vertical and
/// the picture is a mirror of the camera, a sign wrong on the depth and the
/// labels are on the three arms pointing into the screen.
///
/// Screen y counts down and a camera's up counts up, hence the negation — the
/// same flip `crate::pick::Lens::project` applies. An axis exactly across the
/// view counts as facing: it is neither toward nor away, and lighting it is
/// what keeps a level camera showing three labels instead of one.
fn axis_arm(
    axis: gg_math::sim::DVec3,
    basis: (
        gg_math::sim::DVec3,
        gg_math::sim::DVec3,
        gg_math::sim::DVec3,
    ),
    arm: f32,
) -> ((f32, f32), bool) {
    let (right, up, forward) = basis;
    let reach = f64::from(arm);
    (
        (
            (axis.dot(right) * reach) as f32,
            -(axis.dot(up) * reach) as f32,
        ),
        axis.dot(forward) <= 0.0,
    )
}

/// The axis widget's side, in logical units — big enough that three arms and a
/// letter are separable at scale 1, small enough to stay out of the picture.
const AXES_BOX: f32 = 46.0;
/// The labelled pad at an arm's tip. A square rather than a disc for
/// `crate::marker`'s reason: `DrawList` has rectangles, and one primitive that
/// clips is worth more than a rounder one that does not.
const AXES_TIP: f32 = 9.0;
/// What each arm is called. Single characters so a pad is a pad and not a chip.
const AXES_NAME: [&str; 3] = ["X", "Y", "Z"];

/// Draw the viewport's axis widget and controls legend (§6 M63).
///
/// A knob rather than always-on because the overlay is the one thing this editor
/// draws *into* the picture rather than around it: a screenshot, a golden bless
/// and a look at a dark corner all want it gone, and none of them wants the pane
/// closed. Not `recorded` — it moves no click, declares no widget and is read
/// only where the drawing happens.
pub(crate) static LEGEND: gg_core::cvar::CVar = gg_core::cvar::CVar::new_bool(
    "d.editor_overlay",
    true,
    "draw the viewport's axis widget and controls legend",
);

/// Transport cells, in declaration order: play, step, stop — offset and width,
/// from the start of the set. Square-ish because they are drawn and not set:
/// the glyph is the meaning, the way it is on every transport an operator has
/// used, and a cell sized for the word `pause` is a cell mostly full of
/// nothing.
///
/// Stop is **last** rather than beside play, which is where a tape deck put it.
/// The destructive one is the one an operator must reach for deliberately, and
/// the set is centred (see [`transport_at`]), so growing it from two to three
/// moved play and step by half a cell — which is why §6 M15.2 costs a re-bless
/// of the recorded editor stream.
pub(crate) const TOOLBAR: [(f32, f32); 3] = [(0.0, 20.0), (23.0, 20.0), (46.0, 20.0)];

/// A window button's width. Its height is the bar's — a caption button reaching
/// the window's top edge is the one proportion every desktop shares, and the
/// 1.4:1 cell is Windows' own 46×32 at this bar's height.
const CAPTION: f32 = 19.0;

/// The box a caption glyph is drawn in, centred in its cell. Odd, so a 1-unit
/// stroke lands on whole units either side of centre at every scale.
const ICON: f32 = 7.0;

/// Ticks inside which a second press on the drag region is a double-click.
/// A third of a second at 60 Hz, counted in ticks so a replay agrees.
pub(crate) const DOUBLE: u64 = 20;

/// The platform's own button set, on the platform's own side and in its own
/// order (§6 M15.1 item 5), against a title `bar`.
///
/// `mac` is a parameter rather than a `cfg!` so both arrangements are laid out
/// wherever the tests run — §8's matrix has no macOS host, so that ordering is
/// written down and not verified.
pub(crate) fn window_buttons(bar: Rect, mac: bool) -> [(WindowCommand, Rect); 3] {
    let order = match mac {
        true => [
            WindowCommand::Close,
            WindowCommand::Minimize,
            WindowCommand::ToggleMaximize,
        ],
        false => [
            WindowCommand::Minimize,
            WindowCommand::ToggleMaximize,
            WindowCommand::Close,
        ],
    };
    // Flush to the corner and to each other, which is what makes the set read
    // as the window's rather than as three widgets that happen to be up there.
    let start = match mac {
        true => bar.x,
        false => bar.right() - CAPTION * order.len() as f32,
    };
    std::array::from_fn(|i| {
        let x = start + i as f32 * CAPTION;
        (order[i], Rect::new(x, bar.y, CAPTION, bar.h))
    })
}

/// The `i`th transport button, given the bar and where the menu strip ended.
///
/// **Centred on the bar**, not trailing the menus: the transport is the one
/// control an operator reaches for without reading, so it goes where every
/// other transport is. The strip is still a floor — a window narrow enough for
/// the menus to reach the middle pushes the set right rather than under them.
///
/// A free function because the strip's end is the platform's business and a
/// test has to be able to lay out the arrangement it is not running under.
pub(crate) fn transport_at(bar: Rect, strip_right: f32, i: usize) -> Rect {
    let (offset, w) = TOOLBAR[i.min(TOOLBAR.len() - 1)];
    let span = TOOLBAR.last().map_or(0.0, |(o, w)| o + w);
    let start = (bar.x + (bar.w - span) * 0.5)
        .floor()
        .max(strip_right + 6.0);
    // Full height, like everything else in the bar: a 9-unit cell cannot hold a
    // line of an 8-unit face and its border both, so "play" hung its `y` out of
    // the bottom and "step" its `t` out of the top.
    Rect::new(start + offset, bar.y, w, bar.h)
}

/// Which transport glyph a cell draws. Drawn rather than set for [`caption`]'s
/// reason: a *shape* is the meaning here, and `play` spelled out is a word in a
/// language, in a bar that has no room for one.
///
/// [`caption`]: Editor::caption
#[derive(Clone, Copy)]
enum Transport {
    Play,
    Pause,
    Step,
    Stop,
}

/// The [`ICON`] box inside a bar cell, on whole units so a stroke lands on
/// whole units either side of centre.
fn centred_icon(cell: Rect) -> Rect {
    Rect::new(
        (cell.x + (cell.w - ICON) * 0.5).floor(),
        (cell.y + (cell.h - ICON) * 0.5).floor(),
        ICON,
        ICON,
    )
}

/// Where the menu strip starts: after the window buttons when the platform puts
/// them on this side, and at the margin when it does not.
pub(crate) fn strip_left(bar: Rect, mac: bool) -> f32 {
    match mac {
        true => {
            window_buttons(bar, mac)
                .iter()
                .fold(bar.x, |x, (_, cell)| x.max(cell.right()))
                + 3.0
        }
        false => bar.x + 3.0,
    }
}

impl Editor {
    /// A caption glyph, in [`ICON`]-unit geometry centred in `cell`.
    ///
    /// Drawn rather than set. The three are the one place in the editor where a
    /// *shape* is the meaning — every desktop draws the same line, box and
    /// cross, and `_`, `[]` and `x` out of the panels' text face are a
    /// pantomime of them that also cannot fit: the brackets are a full ascender
    /// tall and climbed straight out of the cell. Strokes are one logical unit,
    /// so they thicken with the UI scale exactly as a native glyph does with
    /// DPI (§6 M15.1 item 5).
    fn caption(&mut self, command: WindowCommand, cell: Rect, fill: u32, ink: u32) {
        let icon = centred_icon(cell);
        match command {
            WindowCommand::Minimize => {
                let y = icon.y + (ICON - 1.0) * 0.5;
                self.list.rect(Rect::new(icon.x, y, ICON, 1.0), ink);
            }
            // Restore when the window already is: the same two states a native
            // button has, and the reason [`crate::Frame::maximized`] is told to
            // the editor at all.
            WindowCommand::ToggleMaximize if self.maximized => {
                let back = Rect::new(icon.x + 2.0, icon.y, ICON - 2.0, ICON - 2.0);
                self.outline(back, ink);
                let front = Rect::new(icon.x, icon.y + 2.0, ICON - 2.0, ICON - 2.0);
                // Over the back square, because the front one is nearer and
                // there is no other way to say so with rectangles.
                self.list.rect(front, fill);
                self.outline(front, ink);
            }
            WindowCommand::ToggleMaximize => self.outline(icon, ink),
            // A staircase, but stepped in **pixels** rather than in logical
            // units. A unit-stepped diagonal is a row of squares meeting at
            // their corners — at scale 2 that is four-pixel dots with a
            // diagonal seam between each pair, which is exactly what it looked
            // like. A pixel row per step, `stroke` wide, is the same figure a
            // rasterizer draws for a 45° line: every row overlaps its
            // neighbour, so the stroke is continuous at every scale.
            _ => self.in_pixels(icon, |list, cell, stroke| {
                let rows = cell.h as u16;
                for step in 0..rows {
                    let i = f32::from(step);
                    list.rect(Rect::new(cell.x + i, cell.y + i, stroke, 1.0), ink);
                    let back = Rect::new(cell.right() - stroke - i, cell.y + i, stroke, 1.0);
                    list.rect(back, ink);
                }
            }),
        }
    }

    /// A transport glyph, [`ICON`]-unit geometry centred in `cell`.
    fn transport_icon(&mut self, which: Transport, cell: Rect, ink: u32) {
        let icon = centred_icon(cell);
        self.in_pixels(icon, |list, cell, stroke| {
            let (w, h) = (cell.w, cell.h);
            // The apex is on the right and the base on the left, one pixel row
            // at a time: a triangle out of unit-tall rows is the same figure a
            // rasterizer would fill, and the only one a quad list can state.
            let mut wedge = |x: f32, width: f32| {
                let half = (h - 1.0) * 0.5;
                for step in 0..h as u16 {
                    let d = (f32::from(step) - half).abs();
                    let run = (width * (1.0 - d / half.max(1.0))).round().max(1.0);
                    list.rect(Rect::new(x, cell.y + f32::from(step), run, 1.0), ink);
                }
            };
            match which {
                Transport::Play => wedge(cell.x, w),
                // Two bars, a stroke wide with a stroke of air between them.
                Transport::Pause => {
                    let bar = (stroke * 2.0).min((w - stroke) * 0.5).max(1.0);
                    list.rect(Rect::new(cell.x, cell.y, bar, h), ink);
                    list.rect(Rect::new(cell.right() - bar, cell.y, bar, h), ink);
                }
                // Skip-forward: the same wedge, stopped short of the wall it
                // advances onto. One tick and then still, which is the verb.
                Transport::Step => {
                    wedge(cell.x, w - stroke * 2.0);
                    list.rect(Rect::new(cell.right() - stroke, cell.y, stroke, h), ink);
                }
                // A filled square, which is what every transport ever built
                // draws for it — and the one glyph in the set that needs no
                // stepping, so it is a single quad at any scale.
                Transport::Stop => list.rect(Rect::new(cell.x, cell.y, w, h), ink),
            }
        });
    }

    /// Draw into `cell` in **device pixels**, through the transform that undoes
    /// the fit's — [`Editor::text`](crate::Editor::text)'s trick, for a related
    /// reason. A shape whose steps are one logical unit is the same coarse
    /// figure at every window size, magnified; a shape stepped in pixels is the
    /// figure the screen can actually draw. The stroke stays one logical unit,
    /// so it still thickens with the UI scale as a native glyph does with DPI.
    ///
    /// `cell` arrives in logical units and reaches the closure in pixels.
    fn in_pixels(&mut self, cell: Rect, draw: impl FnOnce(&mut DrawList, Rect, f32)) {
        let scale = self.fit.scale;
        let px = |v: f32| (v * scale).round();
        let cell = Rect::new(px(cell.x), px(cell.y), px(cell.w), px(cell.h));
        self.list.push_transform((0.0, 0.0), 1.0 / scale);
        draw(&mut self.list, cell, scale.max(1.0));
        self.list.pop_transform();
    }

    /// The window's title bar, which is also the toolbar (§6 M15.1 item 5): the
    /// menus, play/step, the title, what the session has done so far, and the
    /// platform's own buttons. Everything not claimed by one of those is the
    /// drag region.
    pub(crate) fn topbar(&mut self, frame: &Frame, tick: &Tick) {
        let bar = self.bar_rect();
        self.plate(bar, CHROME, HEADER);
        let pressed = tick.primary && !self.primary;
        // First, so everything below it wins the hit test: the drag region is
        // whatever the title bar has not given to a widget.
        self.drag_region(bar, frame, pressed);

        let strip: Vec<Rect> = self.menus.titles().to_vec();
        for (i, cell) in strip.iter().enumerate() {
            let title = MENUS.get(i).map_or("?", |m| m.title);
            if self.menu_title(
                MENU.indexed(i as u64),
                *cell,
                title,
                self.menus.open() == Some(i),
            ) {
                self.menus.toggle(i);
            }
        }
        // A press anywhere that is not a title or the open panel puts it away —
        // the one gesture every menu bar has and no widget owns.
        let (px, py) = self.router.pointer().position();
        if pressed && !self.menus.contains(px, py) {
            self.menus.close();
        }

        let running = frame.play.running();
        let glyph = match running {
            true => Transport::Pause,
            false => Transport::Play,
        };
        if self.icon_button(PLAY, self.transport(0), glyph, running, true) {
            self.commands.playing = Some(!running);
        }
        // A step over a running sim is meaningless — it is the paused verb.
        self.commands.step =
            self.icon_button(STEP, self.transport(1), Transport::Step, false, !running);
        // Inert with nothing captured: a stop from `Stopped` would restore
        // nothing over a scene that is already the scene. Dimmed rather than
        // lit — stop is a verb and not a state, so a plate under it would read
        // as "stopped" exactly when the session is not (§6 M15.2).
        self.commands.stop = self.icon_button(
            STOP,
            self.transport(2),
            Transport::Stop,
            false,
            frame.play.entered(),
        );

        // Bare until the pointer is on them, like every desktop's: a caption
        // button that is always a plate is a toolbar, and the close one goes
        // the colour it means rather than merely drawing in it.
        let buttons = window_buttons(bar, crate::MAC);
        for (i, (command, rect)) in buttons.iter().enumerate() {
            let response = self.router.hit(WINDOW.indexed(i as u64), *rect);
            let hot = response.hovered || response.held;
            let (fill, ink) = match (command, hot) {
                (WindowCommand::Close, true) => (DANGER, 0xffff_ffff),
                (_, true) => (BUTTON, INK),
                // Nothing drawn at rest, so the bar's own edge runs unbroken
                // behind the set — a caption button is a hover state, not a
                // plate. `CHROME` is what the glyph is then drawn *over*.
                _ => (CHROME, INK),
            };
            if hot {
                self.list.rect(*rect, fill);
            }
            self.caption(*command, *rect, fill, ink);
            if response.clicked {
                self.window = Some(*command);
            }
        }

        // Only the title — every other number has a home that answers the
        // question it was asked from: tick → F1 overlay (§4.8), entity count →
        // tree header, tallies → `perf`.
        let left = self.menus.right() + 6.0;
        let room = (self.transport(0).x - 3.0 - left).max(0.0);
        let title = fit_text(frame.title, (room / EM) as usize);
        self.label_mid(Rect::new(left, bar.y, room, bar.h), left, title, DIM);
    }

    /// A transport cell: the plate every button in this crate is, with a shape
    /// in it instead of a word.
    /// `on` is *state* — the sim is advancing — and lights the plate. `enabled`
    /// is whether the verb can act at all, and only dims the glyph: a transport
    /// button that vanished or moved when it went inert would make the set jump
    /// under the pointer. A disabled button still eats its click rather than
    /// letting the drag region behind it take one.
    fn icon_button(
        &mut self,
        id: WidgetId,
        rect: Rect,
        which: Transport,
        on: bool,
        enabled: bool,
    ) -> bool {
        let response = self.router.hit(id, rect);
        let fill = match (on, response.hovered || response.held) {
            (true, _) => crate::LIVE,
            (false, true) if enabled => PICKED,
            (false, _) => BUTTON,
        };
        self.plate(rect, fill, HEADER);
        self.transport_icon(which, rect, if enabled { INK } else { DIM });
        response.clicked && enabled
    }

    /// A menu title: the cell, with no plate under it.
    ///
    /// A menu bar is not a row of buttons — a title is bare text until the
    /// pointer is on it or its menu is down, which is the whole visual
    /// difference between a menu strip and a toolbar.
    fn menu_title(&mut self, id: WidgetId, cell: Rect, title: &str, open: bool) -> bool {
        let response = self.router.hit(id, cell);
        match (open, response.hovered || response.held) {
            (true, _) => self.list.rect(cell, PICKED),
            (false, true) => self.list.rect(cell, BUTTON),
            (false, false) => {}
        }
        let x = (cell.x + (cell.w - crate::text_width(title)) * 0.5).floor();
        self.label_mid(cell, x, title, INK);
        response.clicked
    }

    /// The `i`th transport button, after the menu strip.
    pub(crate) fn transport(&self, i: usize) -> Rect {
        transport_at(self.bar_rect(), self.menus.right(), i)
    }

    /// The whole bar as one widget: press to move the window, press twice to
    /// maximize it.
    ///
    /// Both of those come free with an OS title bar and neither comes free
    /// without one (§6 M15.1 item 5). The double-click is ours because the
    /// first press has already handed the pointer to the system's move loop,
    /// which swallows the release the OS would otherwise have counted.
    fn drag_region(&mut self, bar: Rect, frame: &Frame, pressed: bool) {
        let response = self.router.hit(DRAG, bar);
        if !(response.held && pressed) {
            return;
        }
        let doubled = self
            .last_bar_press
            .is_some_and(|last| frame.tick.saturating_sub(last) <= DOUBLE);
        self.last_bar_press = Some(frame.tick);
        self.window = Some(match doubled {
            true => WindowCommand::ToggleMaximize,
            false => WindowCommand::Drag,
        });
    }

    /// Every live entity, scrolled, with the components it carries.
    ///
    /// The list is a *window*: only the rows the pane can show are fetched
    /// ([`crate::Editor::first_row`]), so ten thousand entities cost what ten do
    /// and the bar is the only thing that knows the difference.
    pub(crate) fn tree(&mut self, world: &mut gg_ecs::World, body: Rect) {
        let head = Rect::new(body.x + 2.0, body.y + 2.0, (body.w - 4.0).max(0.0), 9.0);
        self.plate(head, HEADER, HEADER);
        let pages = self.scan.total.div_ceil(self.per_page).max(1);
        let page = self.first_row / self.per_page.max(1) + 1;
        // The entity count belongs to whichever panel answers "what is in
        // this world" — this one.
        self.label(
            Rect::new(
                head.x + 2.0,
                head.y + 1.0,
                (head.w - HEAD_BUTTONS).max(0.0),
                8.0,
            ),
            &format!("{} entities  {page}/{pages}", self.scan.total),
            ACCENT,
        );
        let list = tree_list(body);
        let page_by = self.per_page as f32 * PITCH;
        self.structure(world, head);
        if self.button(PREV, head_button(head, 1), "<", false) {
            self.scroll_by(Pane::Tree, -page_by);
        }
        if self.button(NEXT, head_button(head, 0), ">", false) {
            self.scroll_by(Pane::Tree, page_by);
        }

        let scroll = self.scrollable(Pane::Tree, list, self.scan.total as f32 * PITCH);
        // Moved out rather than copied: the loop declares buttons, which needs
        // `&mut self`, and a clone to dodge that borrow buys nothing the swap
        // back does not. Neither is ever observed empty — nothing between here
        // and the restore reads either field — and the loop's only exit is the
        // `continue` below, so the restore is unconditional.
        let rows = std::mem::take(&mut self.rows);
        let slots = std::mem::take(&mut self.scan.slots);
        let base = self.first_row;
        let names = ((scroll.view.w / EM) as usize).saturating_sub(7);
        self.list.push_clip(scroll.view);
        for (i, (entity, mask)) in rows.iter().enumerate() {
            // `row` is what keeps a half-scrolled list honest: the row above the
            // pane is not declared, so it cannot take a click through the panel
            // that is actually up there.
            let Some(rect) = scroll.row((base + i) as f32 * PITCH, ROW) else {
                continue;
            };
            let picked = self.selected == Some(*entity);
            let text = format!("{:>5} {}", entity.index(), summary(&slots, *mask, names, 8));
            if self.button(TREE_ROW.indexed((base + i) as u64), rect, &text, picked) {
                self.selected = Some(*entity);
                self.lane = None;
                tracing::info!(
                    row = base + i,
                    entity = entity.index(),
                    components = %summary(&slots, *mask, 256, 64),
                    "editor: selected"
                );
            }
        }
        self.rows = rows;
        self.scan.slots = slots;
        self.list.pop_clip();
    }

    /// Spawn, duplicate and delete — what makes the tree an editor rather than a
    /// viewer (§6 M15.4 item 5).
    ///
    /// All three reach the world through machinery that names no game type: a
    /// spawn writes the one component the *host* is allowed to have an opinion
    /// about (`Renderable` is the render protocol, §4.2.2), a duplicate is
    /// `World::duplicate`'s component-set walk, and a delete is a despawn. So a
    /// build that was compiled against none of the game's types can still
    /// restructure its world.
    fn structure(&mut self, world: &mut gg_ecs::World, head: Rect) {
        let eye =
            self.eye(gg_ecs::boundary::Eye::of(world).unwrap_or(gg_ecs::boundary::Eye::ORIGIN));
        let (right, _, forward) = crate::pick::basis(eye.yaw, eye.pitch);
        if self.button(SPAWN, head_button(head, 4), "+", false) {
            self.history.edit(world);
            // In front of the camera being looked through, at the depth the
            // operator is working at, in the one shape the host knows how to
            // draw. A spawn that produced an entity with nothing on it would be
            // a button whose whole effect is a row appearing in a list — and
            // over a game that never declared `Renderable` this *introduces* it,
            // which the save policy permits (a save may gain, never lose) and
            // the tree shows.
            let entity = world.spawn();
            let box_ = gg_ecs::boundary::Renderable::boxed(
                eye.position + forward * self.working_depth(world, eye.position, forward),
                gg_math::sim::Vec3::splat(0.5),
                SPAWN_INK,
            );
            match world.insert(entity, box_) {
                Ok(()) => {
                    self.selected = Some(entity);
                    self.lane = None;
                    self.edits += 1;
                    tracing::info!(entity = entity.index(), "editor: spawned");
                }
                Err(error) => tracing::warn!(%error, "editor: spawn refused"),
            }
        }
        let Some(entity) = self.selected else {
            // Both of the rest act on a selection, and both are still *drawn*
            // without one: a control that appears and disappears is a control an
            // operator cannot learn where to find.
            self.button(COPY, head_button(head, 3), "c", false);
            self.button(KILL, head_button(head, 2), "x", false);
            return;
        };
        if self.button(COPY, head_button(head, 3), "c", false) {
            self.history.edit(world);
            match world.duplicate(entity) {
                Some(copy) => {
                    // One nudge grain to the camera's right (§6 M20 item 10). A
                    // copy exactly on top of its original is invisible, and the
                    // pick's lowest-index tie-break hands the next click back to
                    // the *original* — so the button reads as doing nothing
                    // while the world fills with stacked boxes. One grain and
                    // not an arbitrary offset because the result must still be a
                    // value the nudge bar could have typed.
                    let grain = crate::STEPS.get(self.step).copied().unwrap_or(1.0);
                    if let Some(box_) = world.get_mut::<gg_ecs::boundary::Renderable>(copy) {
                        box_.position += right * grain;
                    }
                    self.selected = Some(copy);
                    self.lane = None;
                    self.edits += 1;
                    tracing::info!(
                        entity = entity.index(),
                        copy = copy.index(),
                        "editor: duplicated"
                    );
                }
                None => tracing::warn!(entity = entity.index(), "editor: nothing to duplicate"),
            }
        }
        if self.button(KILL, head_button(head, 2), "x", false) {
            self.history.edit(world);
            if world.despawn(entity) {
                self.edits += 1;
                tracing::info!(entity = entity.index(), "editor: deleted");
            }
            // Cleared either way: a selection naming a dead entity shows an
            // empty inspector for as long as nobody clicks elsewhere.
            self.selected = None;
            self.lane = None;
            self.gizmo = None;
        }
    }

    /// How far in front of the camera a spawn lands, in metres (§6 M20 item 10).
    ///
    /// The selection's own depth, so a new box arrives beside the thing being
    /// worked on rather than in a plane of its own. Failing that — and **only**
    /// under a flat eye — the mean depth of everything there is: an orthographic
    /// camera's distance frames nothing, so `SPAWN_AT` metres in front of one is
    /// a number with no meaning, and over demo 11 it put every new platform nine
    /// metres out of the level and in front of the player. A perspective eye with
    /// nothing selected keeps [`SPAWN_AT`], where the distance *is* the framing.
    fn working_depth(
        &self,
        world: &gg_ecs::World,
        from: gg_math::sim::DVec3,
        forward: gg_math::sim::DVec3,
    ) -> f64 {
        let along = |at: gg_math::sim::DVec3| (at - from).dot(forward);
        if let Some(depth) = self
            .selected
            .and_then(|entity| world.get::<gg_ecs::boundary::Renderable>(entity))
            .map(|box_| along(box_.position))
            .filter(|depth| *depth > 0.0)
        {
            return depth;
        }
        let flat = self
            .eye(gg_ecs::boundary::Eye::of(world).unwrap_or(gg_ecs::boundary::Eye::ORIGIN))
            .ortho
            > 0.0;
        let Some(query) = flat
            .then(gg_ecs::Query::<&gg_ecs::boundary::Renderable>::new)
            .and_then(Result::ok)
        else {
            return SPAWN_AT;
        };
        let (mut sum, mut seen) = (0.0, 0u32);
        world.each_ref(&query, |_, box_: &gg_ecs::boundary::Renderable| {
            let depth = along(box_.position);
            // In front only: a box behind the camera is not what the operator
            // is looking at, and averaging it in would drag the answer through
            // the eye and out the other side.
            if depth > 0.0 {
                sum += depth;
                seen += 1;
            }
        });
        match seen {
            0 => SPAWN_AT,
            _ => sum / f64::from(seen),
        }
    }

    /// The border and the play-state tag around the running game. Only the
    /// border: the interior is the renderer's, drawn into the rectangle
    /// [`Editor::viewport_rect`](crate::Editor::viewport_rect) hands the shell,
    /// so an object at the edge of this pane is at the edge of the picture
    /// rather than off-screen.
    pub(crate) fn viewport(&mut self, world: &mut gg_ecs::World, frame: &Frame, body: Rect) {
        self.outline(body, ACCENT);
        // Inside the border, which is what `viewport_rect` hands the renderer:
        // the picture and the rectangle a click is measured against have to be
        // the same rectangle or a pick near an edge lands off by the border.
        let view = body.inset(1.0);
        // Before the lens, and it returns: with no project there is nothing to
        // pick, nothing to outline and no game drawing behind the pane — so the
        // hole the viewport normally leaves is the picker's rectangle (§6 M15.1
        // item 4).
        if frame.project.is_none() {
            self.launcher(frame, view);
            return;
        }
        let lens = self.lens(world);
        // In this order, and it is the order that arbitrates a click: the pick
        // declares the pane, the marker draws over it, and the gizmo's handles
        // are declared **last** so a press on one is a drag rather than a pick.
        // Nothing is plumbed between them to say so — the router's capture is
        // what keeps the pane from reporting a click during a drag (§4.9).
        self.pick(world, frame, view, &lens);
        self.markers(world, view, &lens);
        self.mark(world, view, &lens);
        self.gizmo(world, frame, view, &lens);
        let tag = match frame.play {
            crate::Play::Running => "playing",
            crate::Play::Paused => "paused",
            crate::Play::Stopped => "stopped",
        };
        // The tag gets a backing plate: it reads over whatever the game happens
        // to be drawing behind it, and the game is not this crate's to know.
        let plate = Rect::new(
            body.x + 1.0,
            body.y + 1.0,
            (17.0 * EM).min(body.w - 2.0),
            9.0,
        );
        self.list.rect(plate, HEADER);
        self.label(
            Rect::new(plate.x + 2.0, plate.y + 1.0, plate.w - 3.0, 8.0),
            &format!("viewport {tag}"),
            ACCENT,
        );
        // The gizmo's mode, beside the tag and clickable (§6 M20 item 10). A
        // key alone would be a mode with no way to discover it or to see which
        // one is current, and the arms of the three tools are the same picture.
        let chip = tool_chip(body);
        if chip.right() <= body.right() - 1.0 && self.button(TOOL, chip, self.tool.label(), false) {
            self.tool = self.tool.next();
            tracing::info!(tool = self.tool.label(), "editor: tool");
        }
        self.overlay(frame, view, &lens);
    }

    /// What the operator is flying, and what with (§6 M63): the world's axes as
    /// this camera sees them, and the gestures that move it.
    ///
    /// **Stopped only**, for [`crate::camera`]'s reason rather than a second
    /// one: while the scene plays the viewport is the game's, none of these
    /// gestures do anything, and a legend for controls that are inert is worse
    /// than no legend. Nothing here is a widget — it declares no id and takes no
    /// click — so it cannot shadow the pick rectangle underneath it.
    fn overlay(&mut self, frame: &Frame, view: Rect, lens: &crate::pick::Lens) {
        if !matches!(frame.play, crate::Play::Stopped) || !LEGEND.bool() {
            return;
        }
        let (right, up, forward) = lens.basis();
        // Top-right, and laid out from that corner inward so a narrow pane
        // clips the *left* end of the rows rather than sliding them off the
        // picture. The play tag and the tool chip own the other corner.
        let box_ = Rect::new(
            view.right() - AXES_BOX - 2.0,
            view.y + 2.0,
            AXES_BOX,
            AXES_BOX,
        );
        if box_.x <= view.x || box_.bottom() >= view.bottom() {
            return;
        }
        let mid = (box_.x + box_.w * 0.5, box_.y + box_.h * 0.5);
        let arm = box_.w * 0.5 - AXES_TIP;
        // Painter's order: the arm pointing *away* is drawn first, so the one
        // coming at the operator is on top of it where they cross. Sorting three
        // axes by depth is what makes the widget read as a solid object rather
        // than as three lines.
        let mut order = [0usize, 1, 2];
        let depth = |i: usize| crate::marker::AXES[i].dot(forward);
        order.sort_by(|a, b| depth(*b).total_cmp(&depth(*a)));
        for i in order {
            let ((dx, dy), facing) = axis_arm(crate::marker::AXES[i], (right, up, forward), arm);
            let tip = (mid.0 + dx, mid.1 + dy);
            let ink = crate::marker::AXIS_INK[i];
            // Toward the operator is lit and labelled; away is the same hue at
            // the chrome's weight, which is what says "this axis is behind you"
            // without a second colour to learn.
            let shade = match facing {
                true => ink,
                false => (ink & 0x00ff_ffff) | 0x6000_0000,
            };
            crate::marker::segment(&mut self.list, box_, mid, tip, self.fit.scale, shade);
            let pad = Rect::new(
                tip.0 - AXES_TIP * 0.5,
                tip.1 - AXES_TIP * 0.5,
                AXES_TIP,
                AXES_TIP,
            );
            self.list.rect(pad, shade);
            if facing {
                self.label_mid(pad, pad.x + (AXES_TIP - EM) * 0.5, AXES_NAME[i], CHROME);
            }
        }
        // The legend, under the widget and right-aligned to the same edge. The
        // speed is in it because §6 M63 gave the wheel a second subject, and a
        // knob that moves invisibly is a knob nobody finds — this row *is* the
        // discovery path for the throttle, and its readout while it moves.
        let flying = self.camera.flying();
        let speed = format!("wheel {:.1} m/s", crate::camera::SPEED.float());
        let rows: [(&str, bool); 4] = [
            ("RMB look · MMB pan", flying),
            ("WASD + Space/Ctrl", flying),
            (speed.as_str(), flying),
            ("F frame · G gizmo", false),
        ];
        let width = rows
            .iter()
            .map(|(text, _)| text_width(text))
            .fold(0.0_f32, f32::max);
        let plate = Rect::new(
            box_.right() - width - 4.0,
            box_.bottom() + 2.0,
            width + 4.0,
            rows.len() as f32 * PITCH + 2.0,
        );
        if plate.x <= view.x || plate.bottom() >= view.bottom() {
            return;
        }
        self.list.rect(plate, HEADER);
        for (i, (text, lit)) in rows.iter().enumerate() {
            let row = Rect::new(
                plate.x + 2.0,
                plate.y + 1.0 + i as f32 * PITCH,
                plate.w - 4.0,
                ROW,
            );
            // Right-aligned inside a left-aligned label, which is what the pad
            // is: `label` cuts to its rectangle, so starting the run further in
            // is the alignment.
            let x = row.right() - text_width(text);
            self.label_mid(row, x, text, if *lit { ACCENT } else { DIM });
        }
    }

    /// The project picker, in the game pane, with no game behind it (§6 M15.1
    /// item 4).
    ///
    /// A list in the one pane that is otherwise a hole, rather than a screen of
    /// its own: the whole claim of item 4 is that opening with no project is an
    /// ordinary mode of the editor, and an editor that showed a launcher instead
    /// of itself would be a second application arguing the opposite.
    ///
    /// A project that has not been built is drawn and refuses the click. Running
    /// `cargo` is a decision about what an editor is, and this is not the line to
    /// take it on.
    fn launcher(&mut self, frame: &Frame, view: Rect) {
        // Filled, unlike every other tick of this pane: there is no game drawing
        // behind it, so leaving the hole would show the last frame or nothing.
        self.list.rect(view, CHROME);
        let head = Rect::new(view.x + 3.0, view.y + 3.0, (view.w - 6.0).max(0.0), ROW);
        self.label(head, "no project — pick one", DIM);
        let list = picker_list(view);
        if frame.projects.is_empty() {
            self.label(list, "no game crates under demos/ (§3)", DIM);
            return;
        }
        let rows = frame.projects.len();
        let scroll = self.scrollable(Pane::Viewport, list, rows as f32 * PITCH);
        self.list.push_clip(scroll.view);
        for (i, project) in frame.projects.iter().enumerate() {
            let Some(rect) = cell(&scroll, 1, rows, i) else {
                continue;
            };
            if !project.built {
                // Not a button: a control that looks clickable and is not is
                // worse than a row that says why. The name of the thing to run is
                // in the row, because the reader's next move is to run it.
                self.label(
                    rect,
                    &format!("{} — cargo build -p demo-{}", project.name, project.name),
                    DIM,
                );
                continue;
            }
            if self.button(OPEN.indexed(i as u64), rect, &project.name, false) {
                self.commands.open = Some(i);
                // The session ends here and the next one is over that game: a
                // shell is built around the dylib it was pointed at, and the
                // window closing is what a host already knows how to act on.
                // Two channels rather than one because they are two facts —
                // *that* this session is over, and *which* project follows it.
                self.window = Some(crate::WindowCommand::Close);
                tracing::info!(project = project.name, "editor: project picked");
            }
        }
        self.list.pop_clip();
    }

    /// The camera the viewport was drawn through this tick — the one both the
    /// pick and the marker read, so a click and an outline cannot disagree
    /// (`crate::pick::Lens`).
    ///
    /// Everything in it moves: the eye is the editor's while the scene is
    /// stopped and the game's otherwise (§6 M15.2 item 2), the field of view and
    /// the near plane are CVars the console edits live, and the aspect is the
    /// **rendered** rectangle's rather than the pane's — the two differ by the
    /// rounding `viewport_rect` does, and the picture came out of the first one.
    fn lens(&self, world: &gg_ecs::World) -> crate::pick::Lens {
        let rendered = self.viewport_rect();
        crate::pick::Lens::new(
            self.eye(gg_ecs::boundary::Eye::of(world).unwrap_or(gg_ecs::boundary::Eye::ORIGIN)),
            gg_render::cvars::FOV.float(),
            f64::from(rendered.width.max(1)) / f64::from(rendered.height.max(1)),
            gg_render::cvars::NEAR.float(),
        )
    }

    /// A click in the stopped viewport selects what the ray under it hits, and
    /// deselects when it hits nothing (§6 M15.4 item 1).
    ///
    /// Stopped only, and that is [`crate::camera`]'s rule rather than a second
    /// one: while the scene is stopped the viewport is the operator's and the
    /// camera being looked through is the editor's, so a pick and a picture
    /// agree by construction. While it plays, the pane is the game's — no widget
    /// is declared over it at all, so a press there is a press the player made.
    fn pick(&mut self, world: &gg_ecs::World, frame: &Frame, view: Rect, lens: &crate::pick::Lens) {
        if !matches!(frame.play, crate::Play::Stopped) {
            return;
        }
        let response = self.router.hit(VIEWPORT, view);
        if !response.clicked || view.w <= 0.0 || view.h <= 0.0 {
            return;
        }
        let (px, py) = self.router.pointer().position();
        let at = (
            f64::from((px - view.x) / view.w),
            f64::from((py - view.y) / view.h),
        );
        self.selected = crate::pick::nearest(world, &lens.ray(at), lens, f64::from(view.h));
        // The lane is an offset into whatever was selected before, so it cannot
        // survive the selection moving — including to nothing.
        self.lane = None;
        match self.selected {
            Some(entity) => tracing::info!(entity = entity.index(), "editor: picked"),
            None => tracing::info!("editor: picked nothing"),
        }
    }

    /// The §4.8 registry, over the same globals `gg_debug`'s console edits — one
    /// registry, so the two faces cannot disagree about a value.
    ///
    /// Booleans toggle on click. Numbers **nudge** since §6 M59, where the note
    /// here used to say a numeric CVar has no step this panel could guess. It
    /// does not have to guess an absolute one: a float steps by a *ratio* and an
    /// int by one, which is the pair that serves `r.ao_radius` (0.5 to 4 metres
    /// in seven clicks) and `r.ao_slices` (two to three) out of one rule.
    ///
    /// What made it worth doing is that the editor could not previously set the
    /// knobs the milestone's own findings are about — every `r.ao_*` in this
    /// pane was a read-only label, so "look at the scene and tune it" meant
    /// editing `gg.cfg` and restarting.
    pub(crate) fn cvars(&mut self, body: Rect) {
        let body = body.inset(3.0);
        let cvars = gg_core::cvar::all();
        let columns = columns_in(body);
        let rows = cvars.len().div_ceil(columns).max(1);
        let scroll = self.scrollable(Pane::Cvars, body, rows as f32 * PITCH);
        self.list.push_clip(scroll.view);
        for (i, cvar) in cvars.iter().enumerate() {
            let Some(rect) = cell(&scroll, columns, rows, i) else {
                continue;
            };
            self.knob(rect, cvar, &CVAR_KNOBS, i as u64, 0);
        }
        self.list.pop_clip();
    }

    /// One CVar's row: a bool that toggles on click, or a reading with two
    /// nudges at its right end.
    ///
    /// `id` distinguishes this row's three widgets from every other row's, and
    /// `trim` is how many leading characters of the name to drop — the render
    /// pane groups by prefix and a column of `r.ao_` repeated eight times is
    /// eight glyphs of the one thing the group heading already said.
    fn knob(&mut self, rect: Rect, cvar: &gg_core::cvar::CVar, ids: &Knobs, id: u64, trim: usize) {
        let name = cvar.name().get(trim..).unwrap_or(cvar.name());
        if matches!(cvar.kind(), gg_core::cvar::CVarKind::Bool) {
            let text = format!("{} {}", fit_text(name, 20), cvar.to_text());
            if self.button(ids.knob.indexed(id), rect, &text, cvar.bool()) {
                cvar.set_bool(!cvar.bool());
                self.edits += 1;
                tracing::info!(cvar = cvar.name(), value = cvar.bool(), "editor: cvar set");
            }
            return;
        }
        // Two nudges at the right end, the reading to the left of them. Narrow
        // on purpose: the name and the value are what a reader scans and the
        // buttons are what a tuner reaches for, so the row must not become two
        // thirds chrome.
        let step = ICON + 4.0;
        let (down, up) = (
            Rect::new(rect.right() - step * 2.0, rect.y, step, rect.h),
            Rect::new(rect.right() - step, rect.y, step, rect.h),
        );
        let room = ((rect.w - step * 2.0) / EM) as usize;
        let text = format!("{} {}", fit_text(name, 17), cvar.to_text());
        let color = if cvar.is_default() { DIM } else { INK };
        self.label(rect, fit_text(&text, room), color);
        for (button, up) in [(down, false), (up, true)] {
            let glyph = if up { "+" } else { "-" };
            let widget = if up { ids.plus } else { ids.minus };
            if !self.button(widget.indexed(id), button, glyph, false) {
                continue;
            }
            nudge(cvar, up);
            self.edits += 1;
            tracing::info!(
                cvar = cvar.name(),
                value = cvar.to_text(),
                "editor: cvar set"
            );
        }
    }

    /// The M9 pack's directory (§4.6): what is in it, not what is resident.
    pub(crate) fn assets(&mut self, body: Rect) {
        let body = body.inset(3.0);
        let Some(pack) = &self.pack else {
            self.label(body, "no --pack on this session", DIM);
            return;
        };
        let columns = columns_in(body);
        let lines: Vec<String> = pack
            .entries()
            .iter()
            .map(|entry| {
                let kind = match entry.kind() {
                    Some(gg_assets::AssetKind::Mesh) => "msh",
                    Some(gg_assets::AssetKind::Texture) => "tex",
                    Some(gg_assets::AssetKind::Material) => "mat",
                    Some(gg_assets::AssetKind::Scene) => "scn",
                    Some(gg_assets::AssetKind::Environment) => "env",
                    Some(gg_assets::AssetKind::Clip) => "pcm",
                    None => "?",
                };
                format!(
                    "{:<3} {:>7} {}",
                    kind,
                    entry.len,
                    fit_text(pack.name(entry), 18)
                )
            })
            .collect();
        // Every entry; the bar says how many.
        let rows = lines.len().div_ceil(columns).max(1);
        let scroll = self.scrollable(Pane::Assets, body, rows as f32 * PITCH);
        self.list.push_clip(scroll.view);
        for (i, text) in lines.iter().enumerate() {
            if let Some(rect) = cell(&scroll, columns, rows, i) {
                self.label(rect, text, INK);
            }
        }
        self.list.pop_clip();
    }

    /// What M8's overlay shows when it has room (§4.8): the pass list with its
    /// GPU milliseconds, and the device's memory.
    ///
    /// **Every** pass since §6 M59, on a scroll bar, where it used to be the
    /// first six of them. Six is what fits, and a frame with four cascades, a
    /// lamp atlas and three probe passes had fourteen — so the list stopped
    /// exactly where the passes an operator has a question about begin, and the
    /// "gpu total" beside it was the total of the six that were shown.
    pub(crate) fn perf(&mut self, body: Rect, frame: &Frame) {
        let body = body.inset(3.0);
        let per = rows_in(body);
        let half = body.w * 0.5;
        let total: f32 = frame.passes.iter().map(|pass| pass.gpu_ms).sum();
        // The view picker sits above the list rather than in the CVar pane,
        // because the CVar pane makes only *bools* clickable — an int there is a
        // read-only label, so before this the editor could show `r.debug_view`
        // and not set it (§6 M59).
        let picker = Rect::new(body.x, body.y, half, ROW);
        let view = gg_render::cvars::DEBUG_VIEW.int();
        let names = gg_render::cvars::DEBUG_VIEWS;
        let name = usize::try_from(view)
            .ok()
            .and_then(|i| names.get(i))
            .copied()
            .unwrap_or("off");
        if self.button(DEBUG, picker, &format!("view: {name}"), view != 0) {
            // Wrapping through the whole table, including back to off: one
            // button, and a view the frame has nothing for shows the picture,
            // so cycling past it is not a dead end (`debug_source`).
            let next = usize::try_from(view).map_or(1, |i| (i + 1) % names.len());
            gg_render::cvars::DEBUG_VIEW.set_int(next as i64);
            self.edits += 1;
            tracing::info!(view = names.get(next), "editor: debug view");
        }
        let list = Rect::new(body.x, body.y + PITCH, half, body.h - PITCH - ROW);
        let scroll = self.scrollable(Pane::Perf, list, frame.passes.len() as f32 * PITCH);
        self.list.push_clip(scroll.view);
        for (i, pass) in frame.passes.iter().enumerate() {
            let text = format!("{:<14}{:>7.3}ms", fit_text(&pass.name, 14), pass.gpu_ms);
            let y = scroll.view.y - scroll.offset + i as f32 * PITCH;
            self.label(Rect::new(list.x, y, half, 8.0), &text, INK);
        }
        self.list.pop_clip();
        if frame.passes.is_empty() {
            self.label(
                Rect::new(body.x, list.y, body.w, 8.0),
                "no gpu on this run",
                DIM,
            );
        }
        let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
        // The session's own tallies live here rather than in the title bar: a
        // count of edits is an instrument reading, and the bar it was in is the
        // window's name. The save target rides with them because it is the
        // answer to "where does `file → save` land", asked before clicking it.
        for (i, text) in [
            format!("gpu total  {total:>7.3}ms"),
            format!(
                "buffers {:>4}  {:>6.1} MiB",
                frame.memory.buffers,
                mib(frame.memory.buffer_bytes)
            ),
            format!(
                "images  {:>4}  {:>6.1} MiB",
                frame.memory.images,
                mib(frame.memory.image_bytes)
            ),
            format!("components {:>3}", self.scan.slots.len()),
            format!("{} edit(s)  {} save(s)", self.edits, self.saves),
        ]
        .iter()
        .enumerate()
        .take(per)
        {
            let row = Rect::new(body.x + half, body.y + i as f32 * PITCH, half, 8.0);
            self.label(row, text, if i == 0 { ACCENT } else { INK });
        }
        // The save target, on its own full-width row at the foot of the pane:
        // it is the longest string the editor shows and the only one whose
        // *end* is the part worth reading, so a half-width column cut it to a
        // filename with half a directory in front of it.
        let room = ((body.w / EM) as usize).saturating_sub(9);
        let cut = tail(frame.save_path, room);
        let elided = if cut.len() < frame.save_path.len() {
            "…"
        } else {
            ""
        };
        self.label(
            Rect::new(body.x, body.bottom() - ROW, body.w, ROW),
            &format!("save -> {elided}{cut}"),
            DIM,
        );
    }

    /// §6 M61: the frame's own knobs — which intermediate to show on the left,
    /// and the `r.*` registry grouped by pass on the right.
    ///
    /// It is not a second CVar pane. Two things are here that the registry as a
    /// flat list cannot give: the debug view is a **list** rather than a cycle
    /// button, so reaching `shadow.2` is one click instead of nine and an
    /// operator can see the table without walking it; and the rows are grouped
    /// by the pass they belong to, which is what turns fifty-three names into
    /// six questions. Both faces write the same globals, so an edit made here
    /// and one made in `cvars` cannot disagree (§4.8).
    ///
    /// **Whether a view resolved is one bit and it is derived**: the graph
    /// declares a `debug-view` pass exactly when `r.debug_view` named an image
    /// this frame has, so its presence in the pass list is the answer. A column
    /// claiming per-row availability would be a second copy of
    /// `scene_attachments`' table, which is the thing `cvars::DEBUG_VIEWS`'
    /// own note exists to forbid.
    pub(crate) fn render(&mut self, body: Rect, frame: &Frame) {
        let (views, knobs) = render_split(body);
        self.views(views, frame);
        self.knobs(knobs);
    }

    /// `r.debug_view`'s table as a list, one row per entry, the live one lit.
    fn views(&mut self, body: Rect, frame: &Frame) {
        let names = gg_render::cvars::DEBUG_VIEWS;
        let live = gg_render::cvars::DEBUG_VIEW.int();
        // The pass the renderer declares only when the view resolved to an
        // image — see this pane's note.
        let resolved = frame.passes.iter().any(|pass| pass.name == "debug-view");
        self.label(Rect::new(body.x, body.y, body.w, ROW), "view", ACCENT);
        let list = view_list(body);
        let scroll = self.scrollable(Pane::Render, list, names.len() as f32 * PITCH);
        self.list.push_clip(scroll.view);
        for (i, name) in names.iter().enumerate() {
            let Some(rect) = scroll.row(i as f32 * PITCH, ROW) else {
                continue;
            };
            let on = live == i as i64;
            if self.button(VIEW_ROW.indexed(i as u64), rect, name, on) {
                // Clicking the live row turns it off, so the pane can put the
                // picture back without hunting for `off` at the top of a
                // scrolled list.
                let next = if on { 0 } else { i };
                gg_render::cvars::DEBUG_VIEW.set_int(next as i64);
                self.edits += 1;
                tracing::info!(view = names.get(next), "editor: debug view");
            }
        }
        self.list.pop_clip();
        // The one honest thing to say about the picture, on the row under the
        // list: `off` is not a failure, and a named view the frame has nothing
        // for is the pass that did not run rather than a bug to report.
        let (text, color) = match (live, resolved) {
            (0, _) => ("showing the frame", DIM),
            (_, true) => ("showing the view", ACCENT),
            (_, false) => ("no such image this frame", DANGER),
        };
        self.label(
            Rect::new(body.x, body.bottom() - ROW, body.w, ROW),
            text,
            color,
        );
    }

    /// Every `r.*` CVar, grouped under the pass it belongs to.
    ///
    /// The grouping is read off the **name** rather than a curated table, so a
    /// CVar added to `gg_render::cvars` lands in its group without this file
    /// hearing about it — the cost is that a group is whatever prefix its
    /// members share, which is exactly how the names were chosen.
    fn knobs(&mut self, body: Rect) {
        let cvars = gg_core::cvar::all();
        let rows = knob_rows(&cvars);
        if rows.is_empty() {
            // Reads as a bug otherwise, and it is not one: registration is the
            // *shell's* (`gg_runtime::run`), so a harness that drives the editor
            // without it has an empty registry rather than a broken pane.
            self.label(body, "no render cvars registered", DIM);
            return;
        }
        let columns = columns_in(body);
        let count = rows.len().div_ceil(columns).max(1);
        let scroll = self.scrollable_at(crate::KNOBS_SLOT, body, count as f32 * PITCH);
        self.list.push_clip(scroll.view);
        for (i, row) in rows.iter().enumerate() {
            let Some(rect) = cell(&scroll, columns, count, i) else {
                continue;
            };
            match row {
                Row::Heading(text) => self.label(rect, text, ACCENT),
                Row::Knob(id, cvar, trim) => {
                    self.knob(rect, cvar, &RENDER_KNOBS, *id as u64, *trim);
                }
            }
        }
        self.list.pop_clip();
    }

    /// §6 M16: the conversation with the agent, and the reload it is usually
    /// about.
    ///
    /// The panel is a *reader* of `gg-agent`'s record — what the shell published
    /// and `gg-tools mcp` serves — so the window and the terminal cannot tell an
    /// operator two different stories about the same reload. What it adds is the
    /// ask, and since M16 item 5 that is anything typed rather than two canned
    /// prompts: `claude` is spawned in the workspace the game's source is in, and
    /// its answer lands in the transcript beside the failure it was about.
    ///
    /// Text entry here is not a hole in §4.7 — [`crate::chat`] states why the
    /// keystrokes are in the replay rather than beside it.
    pub(crate) fn agent(&mut self, pane: Rect, frame: &Frame) {
        let body = pane.inset(3.0);
        // The turn as it happens, not as it finished. Drained whole and into an
        // owned list first: the events borrow the `Ask`, and the transcript is
        // beside it in the same `self`.
        for event in self.ask.drain() {
            self.chat.hear(&event);
        }
        let asking = self.chat.waiting;
        // One line per reload, so an answer three turns later still has the
        // crossing it was about above it.
        if let Some(seam) = frame.reload {
            self.chat.note(seam);
        }

        let head = Rect::new(body.x, body.y, body.w, ROW);
        let (headline, tone) = match frame.reload {
            None => ("no reload yet this session".to_owned(), DIM),
            Some(seam) => match &seam.outcome {
                gg_agent::Outcome::Refused { kind, .. } => {
                    (format!("REFUSED  {kind}  tick {}", seam.tick), DANGER)
                }
                gg_agent::Outcome::Accepted => (
                    format!("accepted  tick {}  {} live", seam.tick, seam.entities),
                    ACCENT,
                ),
            },
        };
        self.label(head, &headline, tone);
        let mut y = body.y + PITCH;
        // The seam's numbers, above the conversation: §9's bar is judged by the
        // record rather than here, so the panel and an agent cannot disagree
        // about the threshold.
        if let Some(seam) = frame.reload
            && let gg_agent::Outcome::Accepted = seam.outcome
        {
            let (verdict, colour) = match seam.within_budget() {
                Some(true) => ("within 2s", ACCENT),
                Some(false) => ("OVER 2s", DANGER),
                None => ("not timed", DIM),
            };
            let ms = seam
                .save_to_swap_ms
                .map_or_else(|| "-".to_owned(), |ms| ms.to_string());
            let hashes = format!(
                "{} -> {}  {ms} ms  {verdict}",
                short(&seam.state_before),
                short(&seam.state_after)
            );
            self.label(Rect::new(body.x, y, body.w, ROW), &hashes, colour);
            y += PITCH;
            for change in seam.changes.iter().take(2) {
                let text = format!("{} {}", change.kind, fit_text(&change.component, 20));
                self.label(Rect::new(body.x, y, body.w, ROW), &text, DIM);
                y += PITCH;
            }
        }

        // The canned asks, only where [`buttons`] says there is one behind them.
        // They fill the field rather than spending on the spot: a prompt worth
        // sending is usually worth a sentence of context first, and the operator
        // now has somewhere to type it.
        let (offered, fixable) = buttons(frame.reload);
        if offered {
            let bar = Rect::new(body.x, y, body.w, ROW + 4.0);
            let half = (bar.w * 0.5).floor();
            let explain = self.button(
                ASK_EXPLAIN,
                Rect::new(bar.x, bar.y, half - 1.0, bar.h),
                "explain",
                asking,
            );
            let fix = fixable
                && self.button(
                    ASK_FIX,
                    Rect::new(bar.x + half + 1.0, bar.y, half - 1.0, bar.h),
                    "fix it",
                    asking,
                );
            if !asking
                && (explain || fix)
                && let Some(canned) = prompt(frame.reload, fix)
            {
                self.chat.prompt = canned;
            }
            y = bar.bottom() + 2.0;
        }

        // The field, laid out from the *bottom* up: a transcript grows and the
        // place you type into must not move under the hand while it does.
        let (hole, sender) = prompt_field(pane);
        let focused = self.router.hit(PROMPT, hole).focused;
        self.plate(hole, BUTTON, if focused { ACCENT } else { HEADER });
        let (back, enter) = chat::edits(frame.input);
        if focused {
            self.chat.edit(frame.typed, back);
        }
        // The tail and not the head: a long prompt is being typed at its end,
        // and a field showing its beginning hides the caret.
        let room = columns(hole.w).saturating_sub(2);
        let shown = match (self.chat.prompt.is_empty(), focused) {
            (true, false) => "click to ask claude".to_owned(),
            (true, true) => String::new(),
            (false, _) => tail(&self.chat.prompt, room).to_owned(),
        };
        let caret = match focused && chat::Chat::caret_on(frame.tick) {
            true => "_",
            false => "",
        };
        let ink = if self.chat.prompt.is_empty() && !focused {
            DIM
        } else {
            INK
        };
        self.label_mid(hole, hole.x + 3.0, &format!("{shown}{caret}"), ink);
        let go = self.button(SEND, sender, "send", asking);
        if !asking
            && (go || (enter && focused))
            && let Some(asked) = self.chat.take_prompt()
        {
            // Wherever the shell was launched, which is the workspace the game's
            // source is in. The panel does not get to choose a directory for an
            // agent that edits files.
            let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
            // The session and not a fresh one: the question after the first is a
            // follow-up, which is both what makes "fix it" mean anything and
            // most of what the first question's thirty seconds were.
            self.ask = gg_agent::Ask::spawn(&asked, &cwd, self.chat.session.as_deref());
            self.chat.asked(frame.tick);
            // The length and not the text: the log is not a second transcript.
            // §6 M16 exit row 4's gate anchors on this line — the rebuild it
            // lands is placed *after* the ask, which is §1's own order.
            tracing::info!(
                tick = frame.tick,
                chars = asked.len(),
                "editor: agent asked"
            );
            self.chat.say(chat::Who::You, asked);
        }

        // The transcript fills what is left. Wrapped once: the height it needs
        // and the lines it draws are the same list.
        let view = Rect::new(body.x, y, body.w, (hole.y - 2.0 - y).max(0.0));
        let cols = columns(view.w);
        let mut lines: Vec<(String, u32)> = Vec::new();
        for turn in &self.chat.log {
            let (prefix, colour) = match turn.who {
                chat::Who::You => ("> ", ACCENT),
                chat::Who::Claude => ("", INK),
                chat::Who::Tool => ("· ", HEADER),
                chat::Who::Host => ("- ", DIM),
            };
            for (i, line) in wrap(&turn.text, cols.saturating_sub(2))
                .into_iter()
                .enumerate()
            {
                let indent = if i == 0 { prefix } else { "  " };
                lines.push((format!("{indent}{line}"), colour));
            }
        }
        if self.chat.waiting {
            // Counted, because the honest answer to "is it stuck" is a number
            // that keeps going up. A turn that runs tools is a minute of work,
            // and the tool lines above this are what it is spending it on.
            let waited = self.chat.waited(frame.tick).unwrap_or_default();
            lines.push((format!("  working… {waited}s"), ACCENT));
        }
        let content = lines.len() as f32 * PITCH;
        // Pinned to the newest whenever there is a new one, like a terminal;
        // clamped by `scrollable`, which is the only thing that knows the view.
        if self.chat.drawn != lines.len() {
            self.chat.drawn = lines.len();
            self.scroll_by(Pane::Agent, content);
        }
        let scroll = self.scrollable(Pane::Agent, view, content);
        self.list.push_clip(scroll.view);
        let mut at = 0.0;
        for (line, colour) in &lines {
            if let Some(row) = scroll.row(at, ROW) {
                self.label(row, line, *colour);
            }
            at += PITCH;
        }
        self.list.pop_clip();
    }

    /// The selected entity's components, field by field, and the nudge bar that
    /// edits one lane of one of them.
    pub(crate) fn inspector(&mut self, world: &mut gg_ecs::World, body: Rect) {
        let head = Rect::new(body.x + 2.0, body.y + 2.0, (body.w - 4.0).max(0.0), 9.0);
        self.plate(head, HEADER, HEADER);
        let Some(entity) = self.selected else {
            self.label(
                Rect::new(head.x + 2.0, head.y + 1.0, head.w, 8.0),
                "no selection",
                DIM,
            );
            return;
        };
        self.label(
            Rect::new(head.x + 2.0, head.y + 1.0, head.w, 8.0),
            &format!("entity {}:{}", entity.index(), entity.generation()),
            ACCENT,
        );

        let mask = self.scan.mask_of(world, entity);
        let slots: Vec<Slot> = self.scan.slots.clone();
        // What the rows below will add up to, measured before any of them are
        // drawn because a bar has to be as long as the content is. One line per
        // present component and one per field of it, which is exactly what the
        // loop advances by — a component whose bytes turn out to be unreadable
        // is the one case the two disagree, and it costs a slightly long bar.
        let content: f32 = slots
            .iter()
            .enumerate()
            .filter(|(bit, _)| mask & (1 << bit) != 0)
            .map(|(_, slot)| (1 + slot.fields.len()) as f32 * PITCH)
            .sum();
        let scroll = self.scrollable(Pane::Inspector, inspector_list(body), content);
        self.list.push_clip(scroll.view);
        let mut at = 0.0;
        for (bit, slot) in slots.iter().enumerate() {
            if mask & (1 << bit) == 0 {
                continue;
            }
            if !read_row(world, entity, slot, &mut self.bytes) {
                continue;
            }
            let bytes = core::mem::take(&mut self.bytes);
            if let Some(row) = scroll.row(at, ROW) {
                let title = Rect::new(row.x + 2.0, row.y, (row.w - 4.0).max(0.0), ROW);
                self.list.rect(title, HEADER);
                self.label(title, tail(slot.declared, 27), ACCENT);
            }
            at += PITCH;
            for (index, field) in slot.fields.iter().enumerate() {
                let Some(row) = scroll.row(at, ROW) else {
                    at += PITCH;
                    continue;
                };
                self.label(
                    Rect::new(row.x + 3.0, row.y, 8.0 * EM, 8.0),
                    fit_text(field.name, 8),
                    DIM,
                );
                let lanes = value::lanes(field);
                if lanes == 0 {
                    self.label(
                        Rect::new(row.x + 3.0 + 8.0 * EM, row.y, 110.0, 8.0),
                        &value::hex(&bytes, field),
                        DIM,
                    );
                    at += PITCH;
                    continue;
                }
                for lane in 0..lanes.min(lanes_in(body)) {
                    let rect = lane_rect(body, row.y, lane);
                    let here = Lane {
                        id: slot.id,
                        field: index as u16,
                        lane: lane as u16,
                    };
                    let text = value::show(&bytes, field, lane);
                    // Identity from where the lane *is* and not from a counter:
                    // a counter that only advances over drawn rows renumbers
                    // every widget below the fold each time the pane scrolls,
                    // and hover resolves against the previous frame's ids.
                    let id = (bit as u64) << 16 | (index as u64) << 8 | lane as u64;
                    if self.button(
                        LANE.indexed(id),
                        rect,
                        fit_text(&text, 5),
                        self.lane == Some(here),
                    ) {
                        self.lane = Some(here);
                    }
                }
                at += PITCH;
            }
            self.bytes = bytes;
        }
        self.list.pop_clip();
        self.nudge_bar(world, entity, &slots, body);
    }

    /// `-` and `+` on the selected lane, and the step they move it by. The whole
    /// of the editor's write path to the world.
    fn nudge_bar(
        &mut self,
        world: &mut gg_ecs::World,
        entity: gg_ecs::Entity,
        slots: &[Slot],
        body: Rect,
    ) {
        let bar = nudge_rect(body);
        self.plate(bar, HEADER, HEADER);
        let grain = STEPS[self.step];
        if self.button(
            GRAIN,
            Rect::new(bar.x + 1.0, bar.y + 1.0, 32.0, 8.0),
            &format!("{grain}"),
            false,
        ) {
            self.step = (self.step + 1) % STEPS.len();
        }
        let Some(lane) = self.lane else {
            self.label(
                Rect::new(bar.x + 36.0, bar.y + 1.0, (bar.w - 38.0).max(0.0), 8.0),
                "pick a field",
                DIM,
            );
            return;
        };
        let Some(slot) = slots.iter().find(|s| s.id == lane.id).copied() else {
            self.lane = None;
            return;
        };
        let Some(field) = slot.fields.get(lane.field as usize).copied() else {
            self.lane = None;
            return;
        };
        let minus = self.button(
            MINUS,
            Rect::new(bar.x + 36.0, bar.y + 1.0, 12.0, 8.0),
            "-",
            false,
        );
        let plus = self.button(
            PLUS,
            Rect::new(bar.x + 50.0, bar.y + 1.0, 12.0, 8.0),
            "+",
            false,
        );
        if minus || plus {
            let by = if plus { grain } else { -grain };
            // The edge, before the write: one step per press, which is what an
            // operator undoing a nudge expects (§6 M15.4 item 4).
            self.history.edit(world);
            let applied = write_row(world, entity, &slot, |bytes| {
                value::nudge(bytes, &field, lane.lane as usize, by);
            });
            if applied {
                self.edits += 1;
                tracing::info!(
                    component = slot.declared,
                    field = field.name,
                    lane = lane.lane,
                    by,
                    entity = entity.index(),
                    "editor: field nudged"
                );
            }
        }
        if read_row(world, entity, &slot, &mut self.bytes) {
            let bytes = core::mem::take(&mut self.bytes);
            let text = format!(
                "{}[{}] {}",
                fit_text(field.name, 8),
                lane.lane,
                value::show(&bytes, &field, lane.lane as usize)
            );
            self.label(
                Rect::new(bar.x + 66.0, bar.y + 1.0, (bar.w - 68.0).max(0.0), 8.0),
                &text,
                INK,
            );
            self.bytes = bytes;
        }
    }
}

/// How many characters fit across `width`.
fn columns(width: f32) -> usize {
    ((width / EM) as usize).max(8)
}

/// A hash, cut to something that fits beside another one and still distinguishes
/// two builds — the record keeps all of it, and `gg-tools mcp` serves all of it.
fn short(hash: &str) -> &str {
    hash.get(..8).unwrap_or(hash)
}

/// `text` broken to `cols`, on spaces where it can and mid-word where it must.
///
/// Its own function rather than `fit_text`'s cut because the two strings this is
/// used on — an OS refusal and an agent's answer — are the only ones in the
/// editor whose *end* matters as much as their start.
fn wrap(text: &str, cols: usize) -> Vec<String> {
    let mut out = Vec::new();
    for paragraph in text.lines() {
        let mut line = String::new();
        for word in paragraph.split_whitespace() {
            if line.is_empty() {
                line = word.to_owned();
            } else if line.chars().count() + 1 + word.chars().count() <= cols {
                line.push(' ');
                line.push_str(word);
            } else {
                out.push(std::mem::take(&mut line));
                line = word.to_owned();
            }
            // A single word longer than the pane — a Windows path, every time.
            while line.chars().count() > cols {
                let cut: String = line.chars().take(cols).collect();
                line = line.chars().skip(cols).collect();
                out.push(cut);
            }
        }
        out.push(line);
    }
    out
}

/// The agent pane's prompt field and its send button, off the pane's body.
///
/// A function rather than two expressions inside [`Editor::agent`] for
/// `tree_list`'s reason: `session::aim` has to reach the same rectangle the
/// panel drew, and a script that aimed at a remembered pixel would silently miss
/// the day the pane grew a row.
pub(crate) fn prompt_field(pane: Rect) -> (Rect, Rect) {
    let body = pane.inset(3.0);
    let bar = Rect::new(body.x, body.bottom() - ROW - 4.0, body.w, ROW + 4.0);
    let send = (text_width("send") + 8.0).ceil().min(bar.w);
    let field = Rect::new(bar.x, bar.y, (bar.w - send - 2.0).max(0.0), bar.h);
    (field, Rect::new(field.right() + 2.0, bar.y, send, bar.h))
}

/// Which asks the panel offers: `(explain, fix it)`.
///
/// Its own function because the failure it prevents is a control that disagrees
/// with [`prompt`] — a button drawn where the click resolves to `None` spends
/// nothing and does nothing, which reads as broken rather than as empty. `fix`
/// additionally needs a refusal: sending an agent after a reload that worked is
/// a way to spend money on a question with no answer.
/// One click's worth of a numeric CVar (§6 M59).
///
/// A **ratio** for floats, because the knobs worth turning here span orders of
/// magnitude — `r.ao_radius` lives around 0.5, `r.gi_spacing` around 4, and
/// `r.peak_nits` around 1000, so one additive step cannot serve all three. The
/// additive floor beside it is what lets a value reach and leave zero, which a
/// pure ratio never can.
///
/// Clamped to nothing on purpose: the CVars clamp themselves where they must
/// (`r.ao_slices` at read, `AO_RADIUS.max(1e-3)`), and a second opinion here
/// would be a range this panel invented.
fn nudge(cvar: &gg_core::cvar::CVar, up: bool) {
    const RATIO: f64 = 1.25;
    /// Below this a ratio moves nothing worth seeing, so the step goes additive.
    const FLOOR: f64 = 0.01;
    match cvar.kind() {
        gg_core::cvar::CVarKind::Int => cvar.set_int(cvar.int() + if up { 1 } else { -1 }),
        gg_core::cvar::CVarKind::Float => {
            let value = cvar.float();
            let stepped = match (up, value.abs() < FLOOR) {
                (true, true) => FLOOR,
                (true, false) => value * RATIO,
                (false, true) => 0.0,
                (false, false) => value / RATIO,
            };
            cvar.set_float(if stepped.abs() < FLOOR && up {
                FLOOR
            } else {
                stepped
            });
        }
        // Handled by the caller, which never reaches here.
        gg_core::cvar::CVarKind::Bool => {}
    }
}

fn buttons(reload: Option<&gg_agent::Seam>) -> (bool, bool) {
    match reload.map(|seam| &seam.outcome) {
        None => (false, false),
        Some(gg_agent::Outcome::Refused { .. }) => (true, true),
        Some(gg_agent::Outcome::Accepted) => (true, false),
    }
}

/// What the panel asks, given what the seam says and which button was pressed.
///
/// A *fill*, not a send, since §6 M16 item 5 gave typing a channel: a starting
/// point a sentence of context can be added to. `fix` is the ask that closes
/// §1's loop: the agent edits the game's source and builds it, the watcher
/// sees the artifact change, and the shell reloads with the world intact — no
/// step of which is this panel's to perform.
fn prompt(reload: Option<&gg_agent::Seam>, fix: bool) -> Option<String> {
    let seam = reload?;
    let refusal = match &seam.outcome {
        gg_agent::Outcome::Refused { kind, detail } => Some((kind, detail)),
        gg_agent::Outcome::Accepted => None,
    };
    Some(match (fix, refusal) {
        (true, Some((kind, detail))) => format!(
            "The running game's hot reload was refused as `{kind}`: {detail}\n\nCall the \
             gg-engine MCP tool gg_last_reload for the full record, find the cause in the game \
             crate's source, fix it, and run `cargo build` for that crate so the artifact the \
             watcher is looking at is rebuilt. Do not restart the shell — it is playing and its \
             world is live."
        ),
        (_, Some((kind, detail))) => format!(
            "The running game's hot reload was refused as `{kind}`: {detail}\n\nExplain what \
             that means and what would fix it, in a few sentences. Do not edit anything."
        ),
        (_, None) => format!(
            "The running game hot reloaded at tick {}, restoring {} entities, with the canonical \
             state hash moving {} -> {}.\n\nCall the gg-engine MCP tool gg_last_reload and \
             explain in a few sentences what changed and whether the migration lost anything. Do \
             not edit anything.",
            seam.tick, seam.entities, seam.state_before, seam.state_after
        ),
    })
}

/// The tree's list, under its header — the part that scrolls. Shared with
/// [`crate::Editor::place`], which sizes the fetch off it before anything is
/// drawn.
pub(crate) fn tree_list(body: Rect) -> Rect {
    Rect::new(
        body.x + 2.0,
        body.y + 13.0,
        (body.w - 4.0).max(0.0),
        (body.h - 15.0).max(0.0),
    )
}

/// The `lane`-th value cell on the inspector row at `y`. Shared with
/// [`crate::session`], which has to aim at one without drawing it.
pub(crate) fn lane_rect(body: Rect, y: f32, lane: usize) -> Rect {
    Rect::new(body.x + 3.0 + 8.0 * EM + lane as f32 * 36.0, y, 35.0, 8.0)
}

/// Value lanes an inspector `body` wide has room for, capped at the three a
/// vector has. One always: a pane dragged narrower than a single lane shows a
/// clipped one rather than a row of names with nothing beside them.
///
/// The bar's width comes off whether or not one is showing, so this answers the
/// same number to the panel and to [`crate::session::aim`] — a lane that
/// appeared when the content grew short enough to stop scrolling would be a
/// lane a script cannot aim at.
pub(crate) fn lanes_in(body: Rect) -> usize {
    let room = body.w - 4.0 - 3.0 - 8.0 * EM - gg_ui::scroll::BAR;
    ((room / 36.0) as usize).clamp(1, 3)
}

/// The inspector's scrolling part: under its header, above the nudge bar.
pub(crate) fn inspector_list(body: Rect) -> Rect {
    let top = body.y + 14.0;
    Rect::new(
        body.x,
        top,
        body.w,
        (nudge_rect(body).y - 2.0 - top).max(0.0),
    )
}

/// The nudge bar, pinned to the bottom of the inspector.
pub(crate) fn nudge_rect(body: Rect) -> Rect {
    Rect::new(
        body.x + 2.0,
        body.bottom() - 12.0,
        (body.w - 4.0).max(0.0),
        10.0,
    )
}

/// The render pane's two columns, given its body: the view list and the knobs
/// (§6 M61).
///
/// A third to the views, the rest to the knobs — the view names are short and
/// fixed, while a knob row carries a name, a value and two buttons. Capped, so a
/// wide pane spends the extra width on the knobs rather than on air beside
/// `shadow.0`.
pub(crate) fn render_split(body: Rect) -> (Rect, Rect) {
    let body = body.inset(3.0);
    let at = (body.w * 0.34).min(18.0 * EM);
    (
        Rect::new(body.x, body.y, at, body.h),
        Rect::new(
            body.x + at + 3.0,
            body.y,
            (body.w - at - 3.0).max(0.0),
            body.h,
        ),
    )
}

/// Where the view rows are laid inside the render pane's left column — a
/// function rather than arithmetic at the call site so `session::aim::view` and
/// the rows it aims at cannot be two different tables (`tool_chip`'s reason).
/// A header above and the resolved/absent line below.
pub(crate) fn view_list(column: Rect) -> Rect {
    Rect::new(
        column.x,
        column.y + PITCH,
        column.w,
        (column.h - PITCH * 2.0).max(0.0),
    )
}

/// The gizmo-mode chip, beside the viewport's own tag in its top-left corner
/// (§6 M20 item 10). A function rather than four numbers at the call site so a
/// script can aim at it — `session::aim::tool` is its only other reader, and a
/// second copy of the arithmetic is a script that clicks where the chip was.
pub(crate) fn tool_chip(body: Rect) -> Rect {
    let tag = (17.0 * EM).min(body.w - 2.0);
    Rect::new(body.x + 3.0 + tag, body.y + 1.0, 5.0 * EM, 9.0)
}

/// Rows a pane has room for — what an unscrolled panel truncates to.
fn rows_in(body: Rect) -> usize {
    ((body.h / PITCH) as usize).max(1)
}

/// Where the launcher's project rows are laid, given the game pane's *interior*
/// — the same rectangle [`crate::session::aim::project`] aims into, so a click
/// authored blind and the row it lands on cannot be two different tables (§6
/// M15.1 item 4). A header row above it says what the pane is.
pub(crate) fn picker_list(view: Rect) -> Rect {
    Rect::new(
        view.x + 3.0,
        view.y + 3.0 + PITCH,
        (view.w - 6.0).max(0.0),
        (view.h - 6.0 - PITCH).max(0.0),
    )
}

/// Columns a list pane shows: two when it is wide enough for two readable ones,
/// one otherwise. A pane dragged narrow shows a single column rather than two
/// truncated ones.
fn columns_in(body: Rect) -> usize {
    if body.w >= 60.0 * EM { 2 } else { 1 }
}

/// The `i`-th cell of a `columns`×`rows` grid filling a scrolled view,
/// column-major — or `None` when it is scrolled out of sight.
///
/// `rows` is the *content's* row count and not the view's: the grid is as tall
/// as it needs to be and the bar is what says so.
fn cell(scroll: &gg_ui::Scroll, columns: usize, rows: usize, i: usize) -> Option<Rect> {
    let row = scroll.row((i % rows) as f32 * PITCH, ROW)?;
    let width = row.w / columns as f32;
    Some(Rect::new(
        row.x + (i / rows) as f32 * width,
        row.y,
        (width - 3.0).max(0.0),
        ROW,
    ))
}

/// The component names a row has room for: `chars` in total, `per` each, cut
/// from the left because the half of `demo05.observer` worth reading is the
/// right one.
fn summary(slots: &[Slot], mask: u64, chars: usize, per: usize) -> String {
    let mut out = String::new();
    for (bit, slot) in slots.iter().enumerate() {
        if mask & (1 << bit) == 0 {
            continue;
        }
        let name = crate::short(slot.declared, per);
        if out.chars().count() + name.chars().count() + 1 > chars {
            out.push('+');
            break;
        }
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(name);
    }
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The axis widget reads the world the way the picture does: +X to the
    /// right, +Y **up the screen** — which is down in these coordinates — and
    /// the axis along the view collapsed to the middle. Turning the camera a
    /// quarter turn swaps which of the two horizontal axes is which, and that
    /// swap is the whole claim: a widget that ignored the camera would pass
    /// every one of the first three assertions.
    #[test]
    fn the_axis_widget_reads_the_world_the_way_the_picture_does() {
        use gg_math::sim::DVec3;
        let level = crate::pick::basis(0.0, 0.0);
        let at = |axis| axis_arm(axis, level, 10.0);
        assert_eq!(at(DVec3::X).0, (10.0, 0.0), "+X is not to the right");
        assert_eq!(at(DVec3::Y).0, (0.0, -10.0), "+Y is not up the screen");
        // Along the view axis: no arm at all, and toward the operator, since a
        // level camera looks down -Z.
        assert_eq!(at(DVec3::Z).0, (0.0, 0.0));
        assert!(at(DVec3::Z).1, "+Z reads as behind a camera facing it");
        assert!(!at(-DVec3::Z).1, "-Z reads as in front of one facing away");

        // A quarter turn of yaw faces the camera down -X, so +X is the axis
        // collapsed to the middle now and +Z has taken the screen's *left* —
        // the camera's right is -Z there (`gg_math::sim::fly_basis`).
        let turned = crate::pick::basis(core::f32::consts::FRAC_PI_2, 0.0);
        let (arm, facing) = axis_arm(DVec3::X, turned, 10.0);
        assert!(arm.0.abs() < 1e-4 && arm.1.abs() < 1e-4, "{arm:?}");
        assert!(facing, "+X reads as ahead of a camera facing away from it");
        assert!(
            (axis_arm(DVec3::Z, turned, 10.0).0.0 + 10.0).abs() < 1e-4,
            "+Z did not take the screen's left"
        );
    }

    fn refused() -> gg_agent::Seam {
        gg_agent::Seam {
            tick: 412,
            outcome: gg_agent::Outcome::Refused {
                kind: "HostApiMismatch",
                detail: r"`C:\dev\a.dll` was built against host API v3".to_owned(),
            },
            code_before: "aa".into(),
            code_after: None,
            state_before: "1111222233334444".into(),
            state_after: "1111222233334444".into(),
            entities: 0,
            changes: Vec::new(),
            load_ms: None,
            save_to_swap_ms: None,
        }
    }

    fn accepted() -> gg_agent::Seam {
        gg_agent::Seam {
            outcome: gg_agent::Outcome::Accepted,
            code_after: Some("bb".into()),
            state_after: "5555666677778888".into(),
            entities: 3,
            changes: vec![gg_agent::Change {
                component: "demo03.cube".to_owned(),
                kind: "migrated",
                defaulted: vec!["wobble".to_owned()],
                retyped: Vec::new(),
            }],
            load_ms: Some(7),
            save_to_swap_ms: Some(180),
            ..refused()
        }
    }

    /// A path is one word and longer than any pane, which is the case a
    /// space-splitting wrap gets wrong by looping or by dropping the tail.
    #[test]
    fn an_unbreakable_word_is_cut_rather_than_dropped_or_looped() {
        let path = r"C:\dev\GGEngine\target\debug\demo_03_reload.dll";
        let lines = wrap(path, 10);
        assert!(lines.len() >= 4, "{lines:?}");
        assert_eq!(lines.concat(), path, "{lines:?}");
        for line in &lines {
            assert!(line.chars().count() <= 10, "{line:?}");
        }
    }

    #[test]
    fn wrapping_breaks_on_spaces_where_it_can_and_keeps_every_word() {
        let lines = wrap("the quick brown fox jumps", 11);
        assert_eq!(lines, ["the quick", "brown fox", "jumps"], "{lines:?}");
    }

    /// Newlines are paragraph breaks, because the prompts and the refusal detail
    /// both carry them and a wrap that flattened them would run two sentences
    /// together mid-line.
    #[test]
    fn a_newline_starts_a_line() {
        assert_eq!(wrap("one\ntwo", 40), ["one", "two"]);
    }

    #[test]
    fn a_hash_is_cut_to_something_two_of_them_still_fit_beside() {
        assert_eq!(short("1111222233334444"), "11112222");
        // Shorter than the cut is returned whole rather than panicking, which is
        // what a hash the record left empty looks like.
        assert_eq!(short("ab"), "ab");
    }

    /// The `fix` prompt is the one that closes §1's loop, so it has to ask for
    /// the build as well as the edit: the watcher looks at an *artifact*, and an
    /// agent that edited source and stopped would leave the game playing the
    /// build that was refused.
    #[test]
    fn the_fix_prompt_names_the_refusal_and_asks_for_a_build() {
        let seam = refused();
        let text = prompt(Some(&seam), true).expect("a prompt");
        assert!(text.contains("HostApiMismatch"), "{text}");
        assert!(text.contains("host API v3"), "{text}");
        assert!(text.contains("cargo build"), "{text}");
        assert!(text.contains("Do not restart the shell"), "{text}");
    }

    /// And the explain prompt must not: a panel that edited when asked to
    /// explain would be the one behaviour an operator cannot undo by not
    /// clicking again.
    #[test]
    fn the_explain_prompts_forbid_editing() {
        for seam in [refused(), accepted()] {
            let text = prompt(Some(&seam), false).expect("a prompt");
            assert!(text.contains("Do not edit anything"), "{text}");
            assert!(!text.contains("cargo build"), "{text}");
        }
    }

    /// An accepted reload's prompt carries the hashes, because "what changed"
    /// is answerable from them and from nothing else the agent can see.
    #[test]
    fn the_accepted_prompt_carries_both_hashes_and_the_mcp_tool_to_read_more() {
        let seam = accepted();
        let text = prompt(Some(&seam), false).expect("a prompt");
        assert!(text.contains("1111222233334444"), "{text}");
        assert!(text.contains("5555666677778888"), "{text}");
        assert!(text.contains("gg_last_reload"), "{text}");
    }

    /// No seam is no question. A session that has not reloaded has nothing to
    /// ask about, and a prompt built from `None` would be an agent sent after a
    /// reload that never happened.
    #[test]
    fn a_session_with_no_reload_has_nothing_to_ask() {
        assert_eq!(prompt(None, false), None);
        assert_eq!(prompt(None, true), None);
    }

    /// And so offers no button to press. The pair is what matters: every button
    /// the panel draws has to have a prompt behind it, or the click is a control
    /// that looks live and does nothing.
    #[test]
    fn every_button_offered_has_a_prompt_behind_it() {
        assert_eq!(buttons(None), (false, false));
        for seam in [refused(), accepted()] {
            let (explain, fix) = buttons(Some(&seam));
            assert!(explain && prompt(Some(&seam), false).is_some());
            assert!(!fix || prompt(Some(&seam), true).is_some());
        }
        // `fix it` follows the refusal, not the presence of a seam.
        assert!(buttons(Some(&refused())).1);
        assert!(!buttons(Some(&accepted())).1);
    }

    /// The render pane's grouping (§6 M61), over a table this test owns
    /// rather than over the live registry — which in a panel test holds
    /// whichever crates happened to have registered.
    ///
    /// Three claims, and the first is the one a curated table would break: a
    /// name nobody wrote a group for still appears, under `frame`.
    #[test]
    fn every_render_cvar_lands_under_exactly_one_heading() {
        static KNOBS: [gg_core::cvar::CVar; 7] = [
            gg_core::cvar::CVar::new_bool("r.ao", true, ""),
            gg_core::cvar::CVar::new_float("r.ao_radius", 0.5, ""),
            gg_core::cvar::CVar::new_int("r.shadow_cascades", 4, ""),
            gg_core::cvar::CVar::new_bool("r.shadow_reach", true, ""),
            gg_core::cvar::CVar::new_bool("r.vsync", true, ""),
            gg_core::cvar::CVar::new_int("r.something_new", 1, ""),
            gg_core::cvar::CVar::new_int("d.editor_scale", 0, ""),
        ];
        let cvars: Vec<&gg_core::cvar::CVar> = KNOBS.iter().collect();
        let rows = knob_rows(&cvars);
        let named: Vec<&str> = rows
            .iter()
            .map(|row| match row {
                Row::Heading(text) => *text,
                Row::Knob(_, cvar, _) => cvar.name(),
            })
            .collect();
        assert_eq!(
            named,
            vec![
                "ambient occlusion",
                "r.ao",
                "r.ao_radius",
                "sun shadows",
                "r.shadow_cascades",
                "r.shadow_reach",
                "frame",
                "r.vsync",
                "r.something_new",
            ],
            "a heading with nothing under it, a row under two, or `d.` leaking in"
        );

        // The trim leaves a name, never a blank: `r.ao` under its own heading
        // keeps everything past `r.`, and `r.ao_radius` drops the prefix whole.
        for row in &rows {
            let Row::Knob(_, cvar, trim) = row else {
                continue;
            };
            let shown = cvar.name().get(*trim..).unwrap_or("");
            assert!(!shown.is_empty(), "{} trims to nothing", cvar.name());
            assert!(cvar.name().ends_with(shown));
        }

        // And an empty registry is an empty list rather than six bare headings.
        assert!(knob_rows(&[]).is_empty());
    }
}
