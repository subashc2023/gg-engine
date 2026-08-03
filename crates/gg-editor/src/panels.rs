//! The five panels §6 M15 names, each one a table of rectangles and clicks.
//!
//! Nothing here is retained: the layout is recomputed every tick and hit-tested
//! against the geometry the previous one declared, which is `gg_ui::router`'s
//! documented one frame of lag. That is also why panel state — selection, page,
//! the dock tab — can live in plain fields and still replay: every one of them
//! moves only in response to a click, and the clicks come off the action map.

use crate::scan::{Slot, read_row, write_row};
use crate::{
    ACCENT, BAR, CHROME, DIM, DOCK, EM, Editor, Frame, HEADER, INK, INSPECT, Lane, PAGE, PITCH,
    STEPS, TREE, VIEW, fit, tail, value,
};
use gg_ui::draw::Rect;
use gg_ui::{WidgetId, font};

const PLAY: WidgetId = WidgetId::new("editor.play");
const STEP: WidgetId = WidgetId::new("editor.step");
const SAVE: WidgetId = WidgetId::new("editor.save");
const PREV: WidgetId = WidgetId::new("editor.tree.prev");
const NEXT: WidgetId = WidgetId::new("editor.tree.next");
const ROW: WidgetId = WidgetId::new("editor.tree.row");
const TAB: WidgetId = WidgetId::new("editor.dock.tab");
const KNOB: WidgetId = WidgetId::new("editor.cvar");
const LANE: WidgetId = WidgetId::new("editor.lane");
const MINUS: WidgetId = WidgetId::new("editor.nudge.minus");
const PLUS: WidgetId = WidgetId::new("editor.nudge.plus");
const GRAIN: WidgetId = WidgetId::new("editor.nudge.step");

/// The dock's tabs, in order.
const TABS: &[&str] = &["cvars", "assets", "perf"];

impl Editor {
    /// Play/pause, step, save, and what the session has done so far.
    pub(crate) fn toolbar(&mut self, frame: &Frame) {
        self.plate(BAR, CHROME, HEADER);
        let cell = |i: f32, w: f32| Rect::new(3.0 + i, 2.0, w, 9.0);
        let label = if frame.playing { "pause" } else { "play" };
        if self.button(PLAY, cell(0.0, 34.0), label, frame.playing) {
            self.commands.playing = Some(!frame.playing);
        }
        // A step over a running sim is meaningless — it is the paused verb.
        self.commands.step = self.button(STEP, cell(37.0, 28.0), "step", false) && !frame.playing;
        if self.button(SAVE, cell(68.0, 28.0), "save", false) {
            self.commands.save = true;
            self.saves += 1;
        }
        let status = format!(
            "tick {}  {} entities  {} edit(s)  {} save(s) -> {}",
            frame.tick,
            self.scan.total,
            self.edits,
            self.saves,
            tail(frame.save_path, 34)
        );
        self.label(Rect::new(104.0, 3.0, 532.0, 9.0), &status, DIM);
    }

    /// Every live entity, paged, with the components it carries.
    pub(crate) fn tree(&mut self) {
        self.plate(TREE, CHROME, HEADER);
        let pages = self.scan.total.div_ceil(PAGE).max(1);
        let head = Rect::new(TREE.x + 2.0, TREE.y + 2.0, TREE.w - 4.0, 9.0);
        self.plate(head, HEADER, HEADER);
        self.label(
            Rect::new(head.x + 2.0, head.y + 1.0, 84.0, 8.0),
            &format!("world {}/{}", self.page + 1, pages),
            ACCENT,
        );
        if self.button(
            PREV,
            Rect::new(head.right() - 20.0, head.y, 9.0, 9.0),
            "<",
            false,
        ) {
            self.page = self.page.saturating_sub(1);
        }
        if self.button(
            NEXT,
            Rect::new(head.right() - 10.0, head.y, 9.0, 9.0),
            ">",
            false,
        ) && self.page + 1 < pages
        {
            self.page += 1;
        }

        // Copied out: the loop declares buttons, which needs `&mut self`.
        let rows: Vec<(gg_ecs::Entity, u64)> = self.rows.clone();
        let slots: Vec<Slot> = self.scan.slots.clone();
        let base = self.page * PAGE;
        for (i, (entity, mask)) in rows.iter().enumerate() {
            let y = TREE.y + 13.0 + i as f32 * PITCH;
            let rect = Rect::new(TREE.x + 2.0, y, TREE.w - 4.0, font::CELL.1 as f32);
            let picked = self.selected == Some(*entity);
            let text = format!(
                "{:>5} {}",
                entity.index(),
                summary(&slots, *mask, ((TREE.w - 4.0) / EM) as usize - 7, 8)
            );
            if self.button(ROW.indexed((base + i) as u64), rect, &text, picked) {
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
    }

    /// The frame around the running game. The panels are opaque and abut, so
    /// what shows through this rectangle *is* the viewport — the renderer draws
    /// the whole surface and the editor covers everything else.
    ///
    /// P2: that makes the viewport a **crop and not a render target**. The
    /// game's projection is built from the window's aspect (§4.1's extract), so
    /// what shows here is the middle of a frame composed for a wider rectangle,
    /// and an object near the edge of the viewport is off-screen rather than at
    /// the edge. Fixing it means a viewport extent reaching `Renderer::frame`,
    /// which is a renderer change and not a panel one — it belongs with whoever
    /// next touches the frame's projection.
    pub(crate) fn viewport(&mut self, frame: &Frame) {
        self.outline(VIEW, ACCENT);
        let tag = match frame.playing {
            true => "playing",
            false => "paused",
        };
        // The tag gets a backing plate: it reads over whatever the game happens
        // to be drawing behind it, and the game is not this crate's to know.
        let plate = Rect::new(VIEW.x + 1.0, VIEW.y + 1.0, 17.0 * EM, 9.0);
        self.list.rect(plate, HEADER);
        self.label(
            Rect::new(plate.x + 2.0, plate.y + 1.0, plate.w - 3.0, 8.0),
            &format!("viewport {tag}"),
            ACCENT,
        );
    }

    /// CVars, the asset pack, and the frame's readings, behind three tabs.
    pub(crate) fn dock(&mut self, frame: &Frame) {
        self.plate(DOCK, CHROME, HEADER);
        for (i, name) in TABS.iter().enumerate() {
            let rect = Rect::new(DOCK.x + 3.0 + i as f32 * 40.0, DOCK.y + 2.0, 38.0, 9.0);
            if self.button(TAB.indexed(i as u64), rect, name, self.dock == i) {
                self.dock = i;
            }
        }
        let body = Rect::new(DOCK.x + 3.0, DOCK.y + 13.0, DOCK.w - 6.0, DOCK.h - 16.0);
        match self.dock {
            0 => self.cvars(body),
            1 => self.assets(body),
            _ => self.perf(body, frame),
        }
    }

    /// The §4.8 registry, over the same globals `gg_debug`'s console edits — one
    /// registry, so the two faces cannot disagree about a value. Booleans toggle
    /// on click; the rest are shown and left to the console, because a numeric
    /// CVar has no step this panel could guess.
    fn cvars(&mut self, body: Rect) {
        let cvars = gg_core::cvar::all();
        let per = ((body.h / PITCH) as usize).max(1);
        for (i, cvar) in cvars.iter().take(per * 2).enumerate() {
            let column = i / per;
            let rect = Rect::new(
                body.x + column as f32 * (body.w / 2.0),
                body.y + (i % per) as f32 * PITCH,
                body.w / 2.0 - 3.0,
                font::CELL.1 as f32,
            );
            let toggle = matches!(cvar.kind(), gg_core::cvar::CVarKind::Bool);
            let text = format!("{} {}", fit(cvar.name(), 20), cvar.to_text());
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
    }

    /// The M9 pack's directory (§4.6): what is in it, not what is resident.
    fn assets(&mut self, body: Rect) {
        let Some(pack) = &self.pack else {
            self.label(body, "no --pack on this session", DIM);
            return;
        };
        let per = ((body.h / PITCH) as usize).max(1);
        let rows: Vec<String> = pack
            .entries()
            .iter()
            .take(per * 2)
            .map(|entry| {
                let kind = match entry.kind() {
                    Some(gg_assets::AssetKind::Mesh) => "msh",
                    Some(gg_assets::AssetKind::Texture) => "tex",
                    Some(gg_assets::AssetKind::Material) => "mat",
                    Some(gg_assets::AssetKind::Scene) => "scn",
                    None => "?",
                };
                format!("{:<3} {:>7} {}", kind, entry.len, fit(pack.name(entry), 18))
            })
            .collect();
        let count = pack.entries().len();
        for (i, text) in rows.iter().enumerate() {
            let rect = Rect::new(
                body.x + (i / per) as f32 * (body.w / 2.0),
                body.y + (i % per) as f32 * PITCH,
                body.w / 2.0 - 3.0,
                font::CELL.1 as f32,
            );
            self.label(rect, text, INK);
        }
        if count > rows.len() {
            self.label(
                Rect::new(body.x, body.bottom() - 8.0, body.w, 8.0),
                &format!("{} of {count} entries", rows.len()),
                DIM,
            );
        }
    }

    /// What M8's overlay shows when it has room (§4.8): the pass list with its
    /// GPU milliseconds, and the device's memory.
    fn perf(&mut self, body: Rect, frame: &Frame) {
        let mut y = body.y;
        let mut total = 0.0;
        for pass in frame.passes.iter().take(6) {
            let text = format!("{:<14}{:>7.3}ms", fit(&pass.name, 14), pass.gpu_ms);
            self.label(Rect::new(body.x, y, body.w / 2.0, 8.0), &text, INK);
            total += pass.gpu_ms;
            y += PITCH;
        }
        if frame.passes.is_empty() {
            self.label(Rect::new(body.x, y, body.w, 8.0), "no gpu on this run", DIM);
        }
        let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
        let right = Rect::new(body.x + body.w / 2.0, body.y, body.w / 2.0, 8.0);
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
            format!("entities {:>5}", self.scan.total),
            format!("components {:>3}", self.scan.slots.len()),
        ]
        .iter()
        .enumerate()
        {
            let row = Rect::new(right.x, right.y + i as f32 * PITCH, right.w, 8.0);
            self.label(row, text, if i == 0 { ACCENT } else { INK });
        }
    }

    /// The selected entity's components, field by field, and the nudge bar that
    /// edits one lane of one of them.
    pub(crate) fn inspector(&mut self, world: &mut gg_ecs::World) {
        self.plate(INSPECT, CHROME, HEADER);
        let head = Rect::new(INSPECT.x + 2.0, INSPECT.y + 2.0, INSPECT.w - 4.0, 9.0);
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
        let mut y = INSPECT.y + 14.0;
        let floor = INSPECT.bottom() - 24.0;
        let mut widget = 0u64;
        for (bit, slot) in slots.iter().enumerate() {
            if mask & (1 << bit) == 0 || y > floor {
                continue;
            }
            if !read_row(world, entity, slot, &mut self.bytes) {
                continue;
            }
            let bytes = core::mem::take(&mut self.bytes);
            let title = Rect::new(INSPECT.x + 2.0, y, INSPECT.w - 4.0, 8.0);
            self.list.rect(title, HEADER);
            self.label(title, tail(slot.declared, 27), ACCENT);
            y += PITCH;
            for (index, field) in slot.fields.iter().enumerate() {
                if y > floor {
                    break;
                }
                self.label(
                    Rect::new(INSPECT.x + 3.0, y, 8.0 * EM, 8.0),
                    fit(field.name, 8),
                    DIM,
                );
                let lanes = value::lanes(field);
                if lanes == 0 {
                    self.label(
                        Rect::new(INSPECT.x + 3.0 + 8.0 * EM, y, 110.0, 8.0),
                        &value::hex(&bytes, field),
                        DIM,
                    );
                    y += PITCH;
                    continue;
                }
                for lane in 0..lanes.min(3) {
                    let rect = Rect::new(
                        INSPECT.x + 3.0 + 8.0 * EM + lane as f32 * 36.0,
                        y,
                        35.0,
                        8.0,
                    );
                    let here = Lane {
                        id: slot.id,
                        field: index as u16,
                        lane: lane as u16,
                    };
                    let text = value::show(&bytes, field, lane);
                    widget += 1;
                    if self.button(
                        LANE.indexed(widget),
                        rect,
                        fit(&text, 5),
                        self.lane == Some(here),
                    ) {
                        self.lane = Some(here);
                    }
                }
                y += PITCH;
            }
            self.bytes = bytes;
        }
        self.nudge_bar(world, entity, &slots);
    }

    /// `-` and `+` on the selected lane, and the step they move it by. The whole
    /// of the editor's write path to the world.
    fn nudge_bar(&mut self, world: &mut gg_ecs::World, entity: gg_ecs::Entity, slots: &[Slot]) {
        let bar = Rect::new(
            INSPECT.x + 2.0,
            INSPECT.bottom() - 12.0,
            INSPECT.w - 4.0,
            10.0,
        );
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
                Rect::new(bar.x + 36.0, bar.y + 1.0, bar.w - 38.0, 8.0),
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
                fit(field.name, 8),
                lane.lane,
                value::show(&bytes, &field, lane.lane as usize)
            );
            self.label(
                Rect::new(bar.x + 66.0, bar.y + 1.0, bar.w - 68.0, 8.0),
                &text,
                INK,
            );
            self.bytes = bytes;
        }
    }
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
