//! `gg-editor` (§6 M15) — the v1 artifact: an immediate-mode editor built on
//! `gg-ui`, hosted by the same `gg-runtime` the demos run under.
//!
//! Not a second engine and not a second renderer. The game renders as it always
//! does and the editor's panels are opaque rectangles over it, so what shows
//! through is the viewport; the panels are `gg_ui::DrawList` geometry appended
//! to the frame the shell already submits.
//!
//! # The editor is host UI, the game's UI is not
//!
//! A game declares its UI as `gg_ecs::boundary::Widget` components because it
//! may not link `gg-ui` (§3's deny pin, §4.9). The editor is host code and has
//! no such wall, so it declares widgets against [`Router`] directly. The
//! consequence is worth stating both ways round: the editor's *panels* are not
//! in the world and so are absent from the canonical hash, while every *edit*
//! it makes goes through `World` like any other write and is hashed like any
//! other state. §6 M15's fourth exit row — a recorded editor session replays to
//! the same final hash — rests on exactly that split, plus the fact that every
//! editor input is an ordinary verb through the ordinary action map (§4.7).
//!
//! # Why there is no text entry
//!
//! See [`value`]: keystrokes outside the action map are not in a replay, so the
//! inspector edits by clicking. Every panel state — selection, page, step size,
//! which dock tab is up — is a pure function of the replayed input stream and
//! therefore reproduces without being hashed.

#![warn(missing_docs)]

pub mod host;
pub mod panels;
pub mod scan;
pub mod session;
pub mod value;

use gg_ecs::Entity;
use gg_ecs::boundary::CANVAS;
use gg_ecs::hash::ComponentId;
use gg_render::ui::UiVertex;
use gg_ui::draw::{DrawList, Rect};
use gg_ui::router::{Router, Tick};
use gg_ui::{WidgetId, font};

/// What the editor asks the host to do this tick. The editor owns the world and
/// does its own edits; these are the three things only the shell can perform.
///
/// A record rather than a stream, and returned by value rather than borrowed:
/// the panels are hit-tested once per tick so two presses cannot land in one,
/// and a `&[Command]` would hold the editor's borrow across the shell applying
/// them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Commands {
    /// `Some(true)` to start advancing the sim, `Some(false)` to stop.
    pub playing: Option<bool>,
    /// Advance exactly one tick, then stop again.
    pub step: bool,
    /// Write a save where the operator named (§6 M14).
    pub save: bool,
}

/// What the editor is told about the frame it is drawing over. Everything here
/// is the host's to know and none of it is in the world.
pub struct Frame<'a> {
    /// The surface in physical pixels; the canvas fits into it (§4.9).
    pub extent: (u32, u32),
    /// The tick about to run.
    pub tick: u64,
    /// Whether the sim is advancing.
    pub playing: bool,
    /// This frame's per-pass GPU readings, empty in a headless run.
    pub passes: &'a [gg_rhi::PassTiming],
    /// Device memory in use.
    pub memory: gg_rhi::MemoryUse,
    /// Where a save would land, shown so the operator knows before clicking.
    pub save_path: &'a str,
}

/// Which field lane the nudge bar acts on. Held by component *id* rather than
/// by index, because a reload can renumber the registry under a live selection.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Lane {
    id: ComponentId,
    field: u16,
    lane: u16,
}

/// Nudge sizes, cycled by the step button. Coarse enough that a position moves
/// visibly, fine enough that a colour channel does not saturate in two clicks.
pub const STEPS: &[f64] = &[0.25, 1.0, 10.0];

/// Panel geometry, in canvas units. Opaque and abutting: what is *not* covered
/// is the viewport, so these five rectangles are the whole layout.
pub(crate) const BAR: Rect = Rect::new(0.0, 0.0, 640.0, 13.0);
pub(crate) const TREE: Rect = Rect::new(0.0, 13.0, 150.0, 347.0);
pub(crate) const VIEW: Rect = Rect::new(150.0, 13.0, 324.0, 261.0);
pub(crate) const DOCK: Rect = Rect::new(150.0, 274.0, 324.0, 86.0);
pub(crate) const INSPECT: Rect = Rect::new(474.0, 13.0, 166.0, 347.0);

/// Row pitch: the font cell plus a unit of air.
pub(crate) const PITCH: f32 = font::CELL.1 as f32 + 1.0;
/// Character advance, for sizing a column in glyphs.
pub(crate) const EM: f32 = font::CELL.0 as f32;
/// Entity rows a tree page shows.
pub(crate) const PAGE: usize = 30;

pub(crate) const INK: u32 = 0xffc8_d4e0;
pub(crate) const DIM: u32 = 0xff76_8496;
pub(crate) const ACCENT: u32 = 0xff7f_d0a0;
pub(crate) const CHROME: u32 = 0xff14_1a22;
pub(crate) const HEADER: u32 = 0xff1e_2632;
pub(crate) const BUTTON: u32 = 0xff2a_3444;
pub(crate) const LIVE: u32 = 0xff3a_6a4a;
pub(crate) const PICKED: u32 = 0xff2a_4a6a;

/// The editor.
///
/// One per session, ticked by the shell inside `sim_tick` so its edits land in
/// the world before the canonical hash reads it.
pub struct Editor {
    list: DrawList,
    router: Router,
    scan: scan::Scan,
    /// This page's entities and the component mask each carries.
    rows: Vec<(Entity, u64)>,
    /// Scratch for one entity's bytes, refilled per component per tick.
    bytes: Vec<u8>,
    /// The pack the asset browser lists, if the shell was given one.
    pack: Option<gg_assets::Pack>,
    selected: Option<Entity>,
    lane: Option<Lane>,
    step: usize,
    page: usize,
    dock: usize,
    /// Saves issued this session — the status line, and what a gate greps for.
    saves: u32,
    /// Edits applied this session, for the same reason.
    edits: u32,
    commands: Commands,
}

impl Editor {
    /// Build an editor. `pack` is the asset pack the browser lists; a session
    /// without one gets an empty browser rather than no browser.
    #[must_use]
    pub fn new(pack: Option<&std::path::Path>) -> Editor {
        let pack = pack.and_then(|path| match gg_assets::Pack::open(path) {
            Ok(pack) => Some(pack),
            Err(error) => {
                tracing::warn!(%error, path = %path.display(), "editor: pack unreadable");
                None
            }
        });
        Editor {
            list: DrawList::default(),
            router: Router::default(),
            scan: scan::Scan::default(),
            rows: Vec::new(),
            bytes: Vec::new(),
            pack,
            selected: None,
            lane: None,
            step: 1,
            page: 0,
            dock: 0,
            saves: 0,
            edits: 0,
            commands: Commands::default(),
        }
    }

    /// Run one tick of editor over `world` and return what it asks the host to
    /// do.
    ///
    /// P2: the panels format their labels with `format!` and so allocate a few
    /// dozen strings a tick — M13's zero-allocation gate covers `gg-ui` and the
    /// overlay and deliberately does not reach here, because an editor is not
    /// in a shipping frame budget. It is still the wrong shape at ten thousand
    /// rows; `gg_ui::Scratch` is what it should be writing into.
    ///
    /// Called from the sim tick and not the frame, for `gg_ui::Ui`'s reason: an
    /// edit has to be in the world before the hash reads it, and a headless run
    /// must route clicks exactly as a windowed one does — which is what lets a
    /// gate replay an editor session with no window anywhere (§1.5).
    pub fn tick(&mut self, world: &mut gg_ecs::World, tick: &Tick, frame: &Frame) -> Commands {
        self.commands = Commands::default();
        self.router.begin(tick, CANVAS);
        self.scan.run(world);
        let pages = self.scan.total.div_ceil(PAGE).max(1);
        self.page = self.page.min(pages - 1);
        self.scan
            .page(world, self.page * PAGE, PAGE, &mut self.rows);

        self.list.clear();
        let scale = (f32::from(frame.extent.0 as u16) / CANVAS.0 as f32)
            .min(f32::from(frame.extent.1 as u16) / CANVAS.1 as f32);
        let fit =
            |extent: u32, canvas: u32| ((extent as f32 - canvas as f32 * scale) * 0.5).floor();
        self.list.push_transform(
            (fit(frame.extent.0, CANVAS.0), fit(frame.extent.1, CANVAS.1)),
            scale,
        );

        self.toolbar(frame);
        self.tree();
        self.viewport(frame);
        self.dock(frame);
        self.inspector(world);
        cursor(&mut self.list, self.router.pointer().position());
        self.list.pop_transform();
        self.commands
    }

    /// Whether the pointer is over a panel rather than over the viewport.
    ///
    /// The host feeds the game a dead input frame while it is: the editor and
    /// the game share one physical mouse, and a click on `pause` must not also
    /// fire whatever the game bound to that button. Reads the pointer the last
    /// tick left, which is the same frame of lag every hit test already has.
    #[must_use]
    pub fn over_panels(&self) -> bool {
        let (x, y) = self.router.pointer().position();
        !VIEW.inset(1.0).contains(x, y)
    }

    /// The geometry the last [`tick`](Self::tick) built, for the shell to append
    /// to the frame's UI stream.
    #[must_use]
    pub fn vertices(&self) -> &[UiVertex] {
        self.list.vertices()
    }

    /// The selected entity, if any — read by tests and by the session gate.
    #[must_use]
    pub fn selected(&self) -> Option<Entity> {
        self.selected
    }

    /// Edits applied and saves requested this session.
    #[must_use]
    pub fn tally(&self) -> (u32, u32) {
        (self.edits, self.saves)
    }

    /// A filled rectangle with a one-unit inner border — the shape every panel
    /// and every button in this crate is.
    pub(crate) fn plate(&mut self, rect: Rect, fill: u32, edge: u32) {
        self.list.rect(rect, edge);
        self.list.rect(rect.inset(1.0), fill);
    }

    /// A clickable cell. Hit-tested against the *canvas* rect for
    /// `gg_ui::boundary`'s reason — the pointer is integrated in canvas units,
    /// so a click replays at any window size.
    pub(crate) fn button(&mut self, id: WidgetId, rect: Rect, label: &str, on: bool) -> bool {
        let response = self.router.hit(id, rect);
        let fill = match (on, response.hovered || response.held) {
            (true, _) => LIVE,
            (false, true) => PICKED,
            (false, false) => BUTTON,
        };
        self.plate(rect, fill, HEADER);
        let x = (rect.x + (rect.w - DrawList::width(label)) * 0.5).floor();
        let y = (rect.y + (rect.h - font::CELL.1 as f32) * 0.5).floor();
        self.list.push_clip(rect);
        self.list.text(x, y, label, INK);
        self.list.pop_clip();
        response.clicked
    }

    /// A one-unit frame with nothing inside it. Four edges rather than a filled
    /// rectangle under a transparent one, because the interior here is the
    /// *game* and anything drawn over it is the editor deciding what a viewport
    /// looks like.
    pub(crate) fn outline(&mut self, rect: Rect, color: u32) {
        for edge in [
            Rect::new(rect.x, rect.y, rect.w, 1.0),
            Rect::new(rect.x, rect.bottom() - 1.0, rect.w, 1.0),
            Rect::new(rect.x, rect.y, 1.0, rect.h),
            Rect::new(rect.right() - 1.0, rect.y, 1.0, rect.h),
        ] {
            self.list.rect(edge, color);
        }
    }

    /// Left-aligned text, cut to `rect` rather than drawn over its neighbour.
    pub(crate) fn label(&mut self, rect: Rect, text: &str, color: u32) {
        self.list.push_clip(rect);
        self.list.text(rect.x, rect.y, text, color);
        self.list.pop_clip();
    }
}

/// The pointer, drawn by the editor for the reason `gg_ui::boundary` draws one:
/// this cursor is an integral of the replayed axis stream and is *not* where the
/// OS thinks the mouse is, so the system cursor cannot stand in for it.
fn cursor(list: &mut DrawList, at: (f32, f32)) {
    for (color, grow) in [(0xd004_0608u32, 1.0), (0xffff_ffff, 0.0)] {
        for row in 0..9u32 {
            let i = f32::from(row as u16);
            list.rect(Rect::new(at.0, at.1 + i, i + 1.0, 1.0).inset(-grow), color);
        }
    }
}

/// Truncate to `chars` glyphs on a character boundary — every panel here is
/// narrower than the names it shows.
pub(crate) fn fit(text: &str, chars: usize) -> &str {
    match text.char_indices().nth(chars) {
        Some((at, _)) => &text[..at],
        None => text,
    }
}

/// The last dotted segment of a declared id, cut to `chars`: `demo05.observer`
/// in a narrow column is `observer`, not `.observer`. Ambiguous by
/// construction — two crates may both declare a `model` — which is why the
/// inspector's own titles use [`tail`] and show the id.
pub(crate) fn short(declared: &str, chars: usize) -> &str {
    fit(declared.rsplit('.').next().unwrap_or(declared), chars)
}

/// Keep the last `chars` glyphs instead of the first. A component id is
/// `demo05.observer` and the half worth showing is the right one.
pub(crate) fn tail(text: &str, chars: usize) -> &str {
    let len = text.chars().count();
    match len > chars {
        true => text
            .char_indices()
            .nth(len - chars)
            .map_or(text, |(at, _)| &text[at..]),
        false => text,
    }
}
