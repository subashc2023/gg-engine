//! The six panes §6 M15.1 docks, each one a table of rectangles and clicks.
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

use crate::scan::{Slot, read_row, write_row};
use crate::{
    ACCENT, BUTTON, CHROME, DANGER, DIM, EM, Editor, Frame, HEADER, INK, Lane, MENUS, PICKED,
    PITCH, Pane, ROW, STEPS, WindowCommand, fit_text, tail, value,
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

        // The title, and nothing else. What used to run along here — the tick,
        // the entity count, the session's tallies — was a status bar wearing a
        // title bar's clothes, and every number in it has a home that answers
        // the question it was asked from: the tick is the F1 overlay's (§4.8),
        // the entity count is the tree's own header, and the tallies belong
        // beside the save path in `perf`.
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
        // The entity count, which used to be in the title bar. It belongs to
        // whichever panel answers "what is in this world", and this is that
        // panel — a number in a title bar is a number nobody asked.
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
        // Copied out: the loop declares buttons, which needs `&mut self`.
        let rows: Vec<(gg_ecs::Entity, u64)> = self.rows.clone();
        let slots: Vec<Slot> = self.scan.slots.clone();
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
        if self.button(SPAWN, head_button(head, 4), "+", false) {
            self.history.edit(world);
            // In front of the camera being looked through, at a fixed distance,
            // in the one shape the host knows how to draw. A spawn that produced
            // an entity with nothing on it would be a button whose whole effect
            // is a row appearing in a list — and over a game that never declared
            // `Renderable` this *introduces* it, which the save policy permits
            // (a save may gain, never lose) and the tree shows.
            let eye =
                self.eye(gg_ecs::boundary::Eye::of(world).unwrap_or(gg_ecs::boundary::Eye::ORIGIN));
            let (_, _, forward) = crate::pick::basis(eye.yaw, eye.pitch);
            let entity = world.spawn();
            let box_ = gg_ecs::boundary::Renderable::boxed(
                eye.position + forward * SPAWN_AT,
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
        let lens = self.lens(world);
        // In this order, and it is the order that arbitrates a click: the pick
        // declares the pane, the marker draws over it, and the gizmo's handles
        // are declared **last** so a press on one is a drag rather than a pick.
        // Nothing is plumbed between them to say so — the router's capture is
        // what keeps the pane from reporting a click during a drag (§4.9).
        self.pick(world, frame, view, &lens);
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
        self.selected = crate::pick::nearest(world, &lens.ray(at));
        // The lane is an offset into whatever was selected before, so it cannot
        // survive the selection moving — including to nothing.
        self.lane = None;
        match self.selected {
            Some(entity) => tracing::info!(entity = entity.index(), "editor: picked"),
            None => tracing::info!("editor: picked nothing"),
        }
    }

    /// The §4.8 registry, over the same globals `gg_debug`'s console edits — one
    /// registry, so the two faces cannot disagree about a value. Booleans toggle
    /// on click; the rest are shown and left to the console, because a numeric
    /// CVar has no step this panel could guess.
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
            let toggle = matches!(cvar.kind(), gg_core::cvar::CVarKind::Bool);
            let text = format!("{} {}", fit_text(cvar.name(), 20), cvar.to_text());
            if toggle {
                if self.button(KNOB.indexed(i as u64), rect, &text, cvar.bool()) {
                    cvar.set_bool(!cvar.bool());
                    self.edits += 1;
                    tracing::info!(cvar = cvar.name(), value = cvar.bool(), "editor: cvar set");
                }
            } else {
                let color = if cvar.is_default() { DIM } else { INK };
                self.label(rect, &text, color);
            }
        }
        self.list.pop_clip();
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
        // Every entry, and the bar says how many there are: the footer this
        // replaces ("12 of 400 entries") was the panel apologizing for a list
        // it would not let anyone reach.
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
    pub(crate) fn perf(&mut self, body: Rect, frame: &Frame) {
        let body = body.inset(3.0);
        let per = rows_in(body);
        let half = body.w * 0.5;
        let mut y = body.y;
        let mut total = 0.0;
        for pass in frame.passes.iter().take(per.min(6)) {
            let text = format!("{:<14}{:>7.3}ms", fit_text(&pass.name, 14), pass.gpu_ms);
            self.label(Rect::new(body.x, y, half, 8.0), &text, INK);
            total += pass.gpu_ms;
            y += PITCH;
        }
        if frame.passes.is_empty() {
            self.label(Rect::new(body.x, y, body.w, 8.0), "no gpu on this run", DIM);
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

/// Rows a pane has room for — what an unscrolled panel truncates to.
fn rows_in(body: Rect) -> usize {
    ((body.h / PITCH) as usize).max(1)
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
