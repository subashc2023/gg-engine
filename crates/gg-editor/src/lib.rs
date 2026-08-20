//! `gg-editor` (§6 M15, §6 M15.1) — the v1 artifact: an immediate-mode editor
//! built on `gg-ui`, hosted by the same `gg-runtime` the demos run under.
//!
//! Not a second engine and not a second renderer. The panels are
//! `gg_ui::DrawList` geometry appended to the frame the shell already submits,
//! and the game renders as it always does — into the rectangle
//! [`Editor::viewport_rect`] names, which the shell hands
//! `gg_render::Renderer::set_viewport`. The panels surround that rectangle
//! rather than covering a window-sized frame around it.
//!
//! # The editor fills the window; a game's UI does not
//!
//! A game declares its UI in `gg_ecs::boundary::CANVAS` units and the host
//! letterboxes that canvas into the window, because a HUD should be the same
//! picture on every screen (§4.9). An editor should not: a tool that leaves
//! bands of nothing down both edges of a 4K window is a tool that has decided
//! the window is 640×360. So the editor takes [`gg_ui::Fit::fill`] at a whole
//! [`ui_scale`], lays out in the logical units that leaves, and its panes tile
//! the whole surface (§6 M15.1).
//!
//! What that costs is stated where it is paid: hit-testing now depends on the
//! window, so a session replays where it was recorded unless the host is told
//! the extent. See [`ui_scale`].
//!
//! Text is the one thing *not* laid out in those units. The panels are set in
//! §4.9's rented face at [`ROW`] pixels per em of scale and emitted in device
//! pixels ([`Editor::text`]), so the tables are the canvas's and the glyphs are
//! the screen's — magnifying a 5×7 bitmap by scale instead would keep every
//! glyph's on-screen size fixed at the source resolution regardless of DPI.
//!
//! # The title bar is the toolbar
//!
//! The window is undecorated with the editor open (§6 M15.1 item 5), so the
//! strip across the top is the *window's* title bar and not a second one under
//! the OS's: it carries the menus, play/step, the window's name, and the
//! platform's own three buttons on the platform's own side. What the OS frame
//! did, [`panels`] now does — a drag region, a resize border and a
//! double-press that maximizes — and the verbs those produce leave through
//! [`Editor::take_window_command`] rather than through the world.
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
//! # Where text may be typed, and where it may not
//!
//! Exactly one place: the agent panel's prompt (§6 M16), whose keystrokes cross
//! as recorded text and two appended verbs, so a typed session replays character
//! for character. The inspector has none — see [`value`] — and that is what keeps
//! an *edit* replayable. Every panel state either way — selection, step size, how
//! far a pane is scrolled, which pane is docked where — is a pure function of
//! the replayed input stream and therefore reproduces without being hashed.

#![warn(missing_docs)]

mod camera;
mod chat;
mod history;
pub mod host;
mod marker;
pub mod panels;
pub mod persist;
mod pick;
mod place;
pub mod project;
pub mod scan;
pub mod session;
pub mod value;

use gg_ecs::Entity;
use gg_ecs::hash::ComponentId;
use gg_render::ui::UiVertex;
use gg_ui::dock::{self, Dock, Node, PaneId};
use gg_ui::draw::{DrawList, Rect};
use gg_ui::router::{Response, Router, Tick};
use gg_ui::scroll::Scroll;
use gg_ui::{Axis, FaceId, Fit, Fonts, WidgetId, font};

/// What the editor asks the host to do this tick. The editor owns the world and
/// does its own edits; these are the three things only the shell can perform.
///
/// A record rather than a stream, and returned by value rather than borrowed:
/// the panels are hit-tested once per tick so two presses cannot land in one,
/// and a `&[Command]` would hold the editor's borrow across the shell applying
/// them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Commands {
    /// `Some(true)` to advance the sim, `Some(false)` to hold it. Entering play
    /// from [`Play::Stopped`] is what captures the world a [`stop`] returns to;
    /// the host owns that edge, because the bytes are the host's.
    ///
    /// [`stop`]: Commands::stop
    pub playing: Option<bool>,
    /// Advance exactly one tick, then hold again. Implies play from
    /// [`Play::Stopped`]: a step with no capture behind it would advance the
    /// scene itself, which is the one thing stop exists to prevent.
    pub step: bool,
    /// Leave play mode, restoring the world captured when it was entered (§6
    /// M15.2). An edit made during play is discarded by this; an edit made
    /// while stopped is the scene.
    pub stop: bool,
    /// Write a save where the operator named (§6 M14).
    pub save: bool,
    /// Open the `n`th of [`Frame::projects`] (§6 M15.1 item 4) — which ends this
    /// session and starts one over that game, because a shell is built around
    /// the dylib it was pointed at.
    ///
    /// An index and not a path, so [`Commands`] stays `Copy` and so the editor
    /// never holds a path it did not read from the host in the first place.
    pub open: Option<usize>,
}

/// What the transport is showing, and what a tick does (§6 M15.2).
///
/// Three states rather than one `playing` bool, because there are three: the
/// difference between [`Paused`] and [`Stopped`] is not whether the sim is
/// advancing — neither is — but whether the host is holding a world to go back
/// to. Which is also why the editor cannot decide this on its own: the capture
/// is bytes, and the bytes are the shell's (§3).
///
/// [`Paused`]: Play::Paused
/// [`Stopped`]: Play::Stopped
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Play {
    /// Not advancing and nothing captured: the world **is** the scene, and an
    /// edit made here survives the next play.
    #[default]
    Stopped,
    /// Advancing, over a world captured at the play edge.
    Running,
    /// A play that is not advancing — the capture is still held, so an edit
    /// made here is still discarded at stop.
    Paused,
}

impl Play {
    /// Whether the sim advances this tick.
    #[must_use]
    pub fn running(self) -> bool {
        matches!(self, Play::Running)
    }

    /// Whether a world is captured — true in play mode, paused or not, and the
    /// only state a stop can be asked for from.
    #[must_use]
    pub fn entered(self) -> bool {
        matches!(self, Play::Running | Play::Paused)
    }

    /// Whether a press hands the physical mouse to the game — the other side of
    /// the pick rule in [`Editor::over_panels`], and stated here so the host
    /// cannot hold a second one. `over_panels` is the caller's because the
    /// pointer is (§6 M15.1).
    ///
    /// [`Running`](Play::Running) and not [`entered`](Play::entered). Stopped is
    /// the state that bit: the viewport is the operator's there, a press in it
    /// is §6 M15.4's pick, and taking the mouse on the same click made selecting
    /// an entity cost the cursor until Escape gave it back. Paused goes the same
    /// way for a weaker reason — nothing reads the input either, and the panels
    /// are what a pause is *for*.
    #[must_use]
    pub fn takes_pointer(self, over_panels: bool) -> bool {
        self.running() && !over_panels
    }
}

/// What a click on the editor's own title bar asks the host to do.
///
/// Host requests, not sim state: a headless replay produces these and there is
/// no window to apply them to, which is exactly why they are commands rather
/// than anything the world or the hash can see. Nothing about them reaches a
/// replay — the clicks that produce them are ordinary `ui_click` frames, and
/// this enum is derived from those the same way `save` is.
///
/// Not in [`Commands`] with the other three, and the difference is timing: a
/// save is applied inside the tick that asked for it, while a window verb is
/// applied by whatever *has* the window — a different moment in a different
/// stack. So the editor holds it until [`Editor::take_window_command`], rather
/// than every host inventing the same field to park it in (§3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WindowCommand {
    /// Hand the window to the system's move loop — a press on the drag region.
    Drag,
    /// The same for a border; see [`Edge`].
    Resize(Edge),
    /// To the taskbar.
    Minimize,
    /// Maximize, or restore if it already is.
    ToggleMaximize,
    /// Close the window, which ends the session.
    Close,
}

/// Which border a resize drags, as a direction: `-1`, `0` or `1` per axis, so
/// `(-1, 0)` is the left edge and `(1, 1)` the bottom-right corner.
///
/// A pair rather than an enum of eight because the hit test that produces it
/// *is* one — the pointer is compared against four sides — and because the
/// translation into a platform's naming then happens once, in the crate that
/// owns the platform (`gg_platform::Window::begin_resize`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Edge {
    /// `-1` left, `1` right, `0` neither.
    pub x: i8,
    /// `-1` top, `1` bottom, `0` neither.
    pub y: i8,
}

/// What the editor is told about the frame it is drawing over. Everything here
/// is the host's to know and none of it is in the world.
pub struct Frame<'a> {
    /// The surface in **physical** pixels. [`ui_scale`] turns it into the
    /// logical units the panes are laid out in.
    pub extent: (u32, u32),
    /// The monitor's scale factor, `1.0` at 96 DPI — the other half of what
    /// [`ui_scale`] decides with. A host with no window (a headless replay, a
    /// golden render) says `1.0`, which is the truth there and the reason no
    /// gate's layout depends on this.
    pub dpi: f32,
    /// The tick about to run.
    pub tick: u64,
    /// Ticks a second the sim is running at (§6 M63).
    ///
    /// Here because the camera's speed knob is metres a *second* — the only unit
    /// an operator can picture — and this is what turns it into the metres a
    /// tick the flight is integrated in. The shell is the one thing that knows
    /// it; a `60` written in this crate would be a lie on every other pace.
    pub hz: u32,
    /// Whether the sim is advancing, and whether a world is captured behind it.
    pub play: Play,
    /// This host's action map, as of this tick — what the editor's own appended
    /// verbs are read through (`host`, §6 M15.2 item 4).
    ///
    /// The map and not a frame, because the ids are the *host's*: `editor_up` is
    /// a different index over every game, so the editor resolves its verbs by
    /// name against the list the host bound (§4.7). `None` for a host that
    /// routes no input at all — a golden render — which is exactly the host
    /// whose reference image must keep showing the game's own view.
    pub input: Option<&'a gg_input::Input>,
    /// Characters typed this tick, in order (§6 M16).
    ///
    /// Not in [`InputFrame`](gg_input::InputFrame) and never in the world: text
    /// is not an action, and a `[u8; N]` of it crossing the ABI would make a
    /// keyboard layout part of a cross-artifact contract (§4.2.2). It arrives
    /// through `gg_input::replay`'s own channel, so a typed prompt records and
    /// replays like every other input this editor takes — see [`crate::chat`].
    ///
    /// Empty on all but the ticks somebody typed on, which is almost all of
    /// them. A host that routes no input at all says `""`.
    pub typed: &'a str,
    /// This frame's per-pass GPU readings, empty in a headless run.
    pub passes: &'a [gg_rhi::PassTiming],
    /// Device memory in use.
    pub memory: gg_rhi::MemoryUse,
    /// Where a save would land, shown so the operator knows before clicking.
    pub save_path: &'a str,
    /// What the window is called. Drawn in the title bar, because since §6
    /// M15.1 item 5 the title bar is this one and the OS is not drawing one.
    pub title: &'a str,
    /// Whether the OS considers the window maximized — which is what the middle
    /// caption button draws, restore or maximize.
    ///
    /// The window's answer and not the editor's: `ToggleMaximize` is a request,
    /// and a shell may refuse it or the operator may maximize by other means.
    /// A host with no window (a headless replay, a golden render) says `false`
    /// and gets the maximize glyph, which is the truth there.
    pub maximized: bool,
    /// The project this session is open over, or `None` for a launcher that has
    /// not been given one (§6 M15.1 item 4).
    ///
    /// `None` is a real mode and not an error path: the panels are all there,
    /// the tree is empty because the world is, and the game pane shows
    /// [`Frame::projects`] instead of a hole with nothing behind it.
    pub project: Option<&'a str>,
    /// What could be opened, in the order the picker lists them. Empty for every
    /// host that is not a launcher — including a session already over a game,
    /// which is why the picker is not a way to switch projects mid-session.
    pub projects: &'a [project::Project],
    /// The last crossing of the reload seam, or `None` in a session that has not
    /// had one (§6 M16).
    ///
    /// The host's to know and emphatically not in the world: it is a fact about
    /// two *builds*, and a component holding it would put the last refusal in
    /// the canonical hash. Borrowed from the shell's own journal, so the panel
    /// and the record `gg-tools mcp` serves cannot disagree.
    pub reload: Option<&'a gg_agent::Seam>,
    /// Draw the pointer.
    ///
    /// `false` whenever something else is already showing one — a windowed
    /// session, where the host keeps the *system* cursor on the same pixel and
    /// two arrows would be one too many, or one where the game has taken the
    /// pointer and the editor's is parked (§6 M15.1). `true` is the case with
    /// no OS in it at all: a golden render, and any host that would rather draw
    /// than borrow.
    pub draw_cursor: bool,
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

/// What [`STEPS`]'s three grains mean to [`Tool::Turn`]: degrees, on the same
/// index, because one step control the operator can see beats a second one that
/// only appears in one mode. A quarter-metre and a degree are both "the fine
/// one" — the ratio between the entries is what the button is *for*.
pub const ANGLES: &[f64] = &[1.0, 15.0, 45.0];

/// What a gizmo drag does to the selection (§6 M20 item 10).
///
/// One gizmo in three modes rather than three gizmos: the arms, the arbitration
/// and the quantization are the same geometry in all three, and only the write
/// at the end differs. Cycled by [`host::verb::TOOL`] and by the chip in the
/// viewport's corner, so it is reachable without knowing the key.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Tool {
    /// Translate along one world axis. The mode M15.4 shipped, and the default.
    #[default]
    Move,
    /// Grow or shrink `Renderable::half_extent` on one axis — what authoring a
    /// level out of boxes is mostly made of, and the reason this enum exists.
    Scale,
    /// Turn about one world axis, in whole [`ANGLES`] degrees.
    Turn,
}

impl Tool {
    /// The three, in cycle order.
    pub const ALL: [Tool; 3] = [Tool::Move, Tool::Scale, Tool::Turn];

    /// The next one, wrapping.
    #[must_use]
    pub fn next(self) -> Tool {
        match self {
            Tool::Move => Tool::Scale,
            Tool::Scale => Tool::Turn,
            Tool::Turn => Tool::Move,
        }
    }

    /// The chip's label — four characters, because the chip sits in the corner
    /// of a pane whose whole width belongs to the game.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Tool::Move => "move",
            Tool::Scale => "size",
            Tool::Turn => "turn",
        }
    }
}

/// A dockable panel.
///
/// The three that were tabs of one fixed "dock" panel through M15 are ordinary
/// panes here, which is most of what docking bought: `perf` can sit beside the
/// viewport instead of under it, and `cvars` can be a tab of the tree.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pane {
    /// Every live entity, paged.
    Tree,
    /// The game. The one pane whose interior the renderer draws.
    Viewport,
    /// The §4.8 registry.
    Cvars,
    /// The M9 pack's directory.
    Assets,
    /// This frame's pass timings and memory.
    Perf,
    /// The selected entity's components, and the nudge bar.
    Inspector,
    /// §6 M16: what the last reload did, and the questions worth asking about
    /// it.
    Agent,
    /// §6 M61: the frame's own knobs — which intermediate to show, and the
    /// `r.*` registry grouped by the pass each row belongs to. Last in
    /// [`Pane::ALL`] because the position is the persisted id, and a layout
    /// written by a build without this pane must still mean what it said.
    Render,
}

impl Pane {
    /// Every pane, in a fixed order — the order [`PaneId`]s are assigned in, so
    /// a persisted layout keeps meaning across a rebuild that added one at the
    /// end.
    pub const ALL: [Pane; 8] = [
        Pane::Tree,
        Pane::Viewport,
        Pane::Cvars,
        Pane::Assets,
        Pane::Perf,
        Pane::Inspector,
        Pane::Agent,
        Pane::Render,
    ];

    /// Its dock identity.
    #[must_use]
    pub fn id(self) -> PaneId {
        PaneId(Pane::ALL.iter().position(|p| *p == self).unwrap_or(0) as u16)
    }

    /// The pane a [`PaneId`] names, or `None` for one this build does not have
    /// — which is what a layout persisted by a newer editor looks like.
    #[must_use]
    pub fn from_id(id: PaneId) -> Option<Pane> {
        Pane::ALL.get(id.0 as usize).copied()
    }

    /// What its tab says.
    #[must_use]
    pub fn title(self) -> &'static str {
        match self {
            Pane::Tree => "world",
            Pane::Viewport => "game",
            Pane::Cvars => "cvars",
            Pane::Assets => "assets",
            Pane::Perf => "perf",
            Pane::Inspector => "inspect",
            Pane::Agent => "agent",
            Pane::Render => "render",
        }
    }
}

/// Which group a pane re-opened from the `view` menu joins: whatever it shares
/// a tab group with in [`default_layout`] and is currently docked, else the
/// pane the default layout puts nearest it.
///
/// Off the default layout rather than a table of neighbours, because a second
/// table is a second thing to keep true — a pane moved in `default_layout` would
/// otherwise keep re-opening where it used to live.
#[must_use]
fn home(pane: Pane, docked: &dyn Fn(Pane) -> bool) -> Option<Pane> {
    let mut groups: Vec<Vec<PaneId>> = Vec::new();
    collect(&default_layout(), &mut groups);
    let own = groups.iter().find(|panes| panes.contains(&pane.id()))?;
    // Its own group first, then any other, so a pane whose whole default group
    // is closed still lands somewhere rather than refusing to open.
    let live = |panes: &[PaneId]| {
        panes
            .iter()
            .filter(|id| **id != pane.id())
            .filter_map(|id| Pane::from_id(*id))
            .find(|p| docked(*p))
    };
    live(own).or_else(|| groups.iter().find_map(|panes| live(panes)))
}

/// Every tab group of a layout tree, in tree order.
fn collect(node: &Node, out: &mut Vec<Vec<PaneId>>) {
    match node {
        Node::Tabs { panes, .. } => out.push(panes.clone()),
        Node::Split { first, second, .. } => {
            collect(first, out);
            collect(second, out);
        }
    }
}

/// The layout the editor opens on, and what a reset returns to: the tree down
/// the left, the inspector down the right, the game above the three instrument
/// panes.
#[must_use]
pub fn default_layout() -> Node {
    // The fractions reproduce M15's fixed table at a 640-unit canvas, which is
    // what the golden reference was blessed against and what a reader comparing
    // the two should see: a 150-unit tree, a 166-unit inspector, and the game
    // taking three quarters of the height between them.
    Node::split(
        Axis::Horizontal,
        0.234,
        Node::pane(Pane::Tree.id()),
        Node::split(
            Axis::Horizontal,
            0.65,
            Node::split(
                Axis::Vertical,
                0.75,
                Node::pane(Pane::Viewport.id()),
                // `render` is appended rather than placed next to `cvars` where
                // it belongs by subject: a tab's rectangle is where a recorded
                // session clicks, and inserting one shifts every tab to its
                // right (§4.7).
                Node::Tabs {
                    panes: vec![
                        Pane::Cvars.id(),
                        Pane::Assets.id(),
                        Pane::Perf.id(),
                        Pane::Agent.id(),
                        Pane::Render.id(),
                    ],
                    active: 0,
                },
            ),
            Node::pane(Pane::Inspector.id()),
        ),
    )
}

/// CVar-overridable UI scale; `0` is auto. Read through [`ui_scale`], which is
/// where the auto rule lives.
/// `recorded` (§6 M40): this is the divisor every physical click is turned into
/// a logical position by, so a session replayed at another value clicks
/// elsewhere.
static SCALE: gg_core::cvar::CVar = gg_core::cvar::CVar::new_int(
    "d.editor_scale",
    0,
    "editor UI scale in whole pixels; 0 picks one from the window",
)
.recorded();

/// Logical units to physical pixels for a surface `extent` across, on a monitor
/// reporting `dpi` (1.0 at 96 DPI; a host with no window says 1.0).
///
/// Integer, always: glyph coverage is sampled nearest, so a fractional scale
/// puts a rectangle's edge and a stem's edge on different sub-pixels.
///
/// **Auto follows the monitor's DPI and not its resolution.** A row's
/// *physical* size is what an operator complains about, and the desktop's own
/// scale factor is the number that tracks it — the window extent does not,
/// since the same logical layout at four times the pixels is four times the
/// row, not the same row. [`PER_DPI`] rows of it, and the window is only ever
/// a cap: it may shrink the scale so a small one still has [`MIN`] units of
/// working area, never grow it. `d.editor_scale` overrules the whole thing,
/// because "what fits" and "what I can read" are different opinions and only
/// one of them is the machine's.
///
/// **This is the layout's dependence on the window**, and therefore on
/// hit-testing: a click is at a logical position, and a logical position is a
/// physical one divided by this. A session recorded at one extent replays at
/// that extent — the host records it (`--editor-extent`) rather than the replay
/// header carrying it, which is §6 M15.1's named residual. The DPI is *not* a
/// second such dependence for any gate: every headless and golden host reports
/// 1.0, and a windowed one at 1.0 lays out exactly as they do.
#[must_use]
pub fn ui_scale(extent: (u32, u32), dpi: f32) -> f32 {
    match SCALE.int() {
        0 => {
            // Halves round *down*: 125% desktop scale asks for 2.5, and 3 is a
            // UI 20% larger than the desktop asked for while 2 is the 100% look
            // the operator already reads — oversize is the complaint, undersize
            // is the status quo (§6 M19).
            let want = dpi.max(0.0) * PER_DPI;
            let want = if want.fract() == 0.5 {
                want.floor()
            } else {
                want.round()
            };
            // Whole scales only, so this is a floor division: at 1279 across,
            // scale 2 would leave 639 units and the panes would be a column
            // short of what every layout here is written for.
            let room = (extent.0 / MIN.0).min(extent.1 / MIN.1) as f32;
            want.clamp(1.0, room.clamp(1.0, 6.0))
        }
        forced => forced as f32,
    }
}

/// Whole scales per unit of monitor scale factor. Two, so a 96-DPI screen gets
/// the 8-pixel row this UI was drawn at doubled to 16 — the size the editor has
/// been at 720p and 1080p since M13, and the one every constant here was picked
/// against.
const PER_DPI: f32 = 2.0;

/// The smallest working area the auto scale will leave, in logical units.
///
/// A floor under the canvas, where M15.1's `WORKING` was a ceiling over it. The
/// extents the gates are aimed at (720p, 900p, and `boundary::CANVAS` headless)
/// come out at the scale they always did, because at DPI 1.0 the window is what
/// binds for all three.
const MIN: (u32, u32) = (640, 360);

/// How the editor's canvas sits on a surface `extent` physical pixels across,
/// on a monitor reporting `dpi`.
///
/// A free function and not a method because the host needs it before the first
/// tick and after the last one — steering an OS cursor onto the editor's
/// pointer is the same arithmetic and must not be a second copy of it (§6
/// M15.1).
#[must_use]
pub fn fit(extent: (u32, u32), dpi: f32) -> Fit {
    Fit::fill(extent, ui_scale(extent, dpi))
}

/// Height of the title bar, in logical units. Outside the dock: the play button
/// is not a pane and there is nothing to dock it to.
pub(crate) const BAR_H: f32 = 13.0;

/// The render pane's second scrolling list (§6 M61), past the last [`PaneId`].
pub(crate) const KNOBS_SLOT: u16 = Pane::ALL.len() as u16;

/// Scroll offsets [`Editor`] keeps: one per pane, plus [`KNOBS_SLOT`].
const SCROLLS: usize = Pane::ALL.len() + 1;

/// The menus, in strip order. What each item *does* is [`menu_action`], and
/// `every_menu_item_does_something` holds the two together.
pub(crate) const MENUS: &[gg_ui::menu::Menu<'static>] = &[
    gg_ui::menu::Menu {
        title: "file",
        items: &["save", "quit"],
    },
    gg_ui::menu::Menu {
        title: "edit",
        items: &["undo", "redo"],
    },
    gg_ui::menu::Menu {
        title: "view",
        // Every pane, then the reset (§6 M61). The pane rows are in
        // [`Pane::ALL`] order and `menu_action` indexes them by position, which
        // is why [`VIEW_PANES`] and this list are the same table read twice
        // rather than two lists — `every_menu_item_does_something` holds them
        // together, and a pane added to `ALL` fails it by name until its row is
        // here.
        items: VIEW_PANES,
    },
];

/// The `view` menu's pane rows, in [`Pane::ALL`] order.
const VIEW_PANES: &[&str] = &[
    "world",
    "game",
    "cvars",
    "assets",
    "perf",
    "inspect",
    "agent",
    "render",
    "reset layout",
];

/// What the `item`th item of the `menu`th menu does. `None` is an index
/// [`MENUS`] does not have.
fn menu_action(menu: usize, item: usize) -> Option<MenuAction> {
    Some(match (menu, item) {
        (0, 0) => MenuAction::Save,
        (0, 1) => MenuAction::Quit,
        (1, 0) => MenuAction::Undo,
        (1, 1) => MenuAction::Redo,
        (2, i) if i == Pane::ALL.len() => MenuAction::ResetLayout,
        (2, i) => MenuAction::Toggle(*Pane::ALL.get(i)?),
        _ => return None,
    })
}

/// One menu item's effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MenuAction {
    Save,
    Quit,
    Undo,
    Redo,
    ResetLayout,
    /// Show the pane if it is closed, close it if it is up (§6 M61).
    Toggle(Pane),
}

/// How thick the resize border around the surface is, in logical units.
///
/// The window has no OS frame (§6 M15.1 item 5), so this *is* the resize border
/// — declared over the panes it overlaps, which is why it is as thin as it can
/// be and still be hit: at the usual scale of 2 it is six physical pixels, and
/// what it costs is the outermost three units of whatever pane touches an edge.
pub(crate) const GRIP: f32 = 3.0;

/// The eight resize grips around a `canvas`, corners first so a corner wins the
/// overlap when the hit test walks them in order.
pub(crate) fn grips(canvas: (u32, u32)) -> [(Edge, Rect); 8] {
    let (w, h) = (canvas.0 as f32, canvas.1 as f32);
    let (far_x, far_y) = ((w - GRIP).max(0.0), (h - GRIP).max(0.0));
    let corner = |x: f32, y: f32| Rect::new(x, y, GRIP, GRIP);
    [
        (Edge { x: -1, y: -1 }, corner(0.0, 0.0)),
        (Edge { x: 1, y: -1 }, corner(far_x, 0.0)),
        (Edge { x: -1, y: 1 }, corner(0.0, far_y)),
        (Edge { x: 1, y: 1 }, corner(far_x, far_y)),
        (
            Edge { x: 0, y: -1 },
            Rect::new(GRIP, 0.0, w - GRIP * 2.0, GRIP),
        ),
        (
            Edge { x: 0, y: 1 },
            Rect::new(GRIP, far_y, w - GRIP * 2.0, GRIP),
        ),
        (
            Edge { x: -1, y: 0 },
            Rect::new(0.0, GRIP, GRIP, h - GRIP * 2.0),
        ),
        (
            Edge { x: 1, y: 0 },
            Rect::new(far_x, GRIP, GRIP, h - GRIP * 2.0),
        ),
    ]
}

/// Which grip `(x, y)` is in, if any.
pub(crate) fn edge_at(canvas: (u32, u32), x: f32, y: f32) -> Option<Edge> {
    grips(canvas)
        .into_iter()
        .find(|(_, rect)| rect.contains(x, y))
        .map(|(edge, _)| edge)
}

/// Whether this host puts its window buttons on the left, in close/minimize/
/// zoom order, rather than on the right in minimize/maximize/close order.
///
/// §8's matrix has no macOS host, so the macOS arrangement is **written down and
/// not verified** — a named residual (§6 M15.1 item 5) rather than a claim. It
/// is still a parameter of [`window_buttons`] and not a `cfg!` inside it, so
/// both arrangements are laid out and asserted wherever the tests run.
pub(crate) const MAC: bool = cfg!(target_os = "macos");

/// One line of text, in logical units. Also the em: the face is rasterized at
/// exactly this many pixels per em, which is what makes a row's ink fill the
/// row it is in.
pub(crate) const ROW: f32 = font::CELL.1 as f32;
/// Row pitch: the row, the unit of air a descender hangs into, and a unit
/// between two rows' ink. Two rather than M15's one because a real face has
/// descenders and a 5×7 bitmap did not.
pub(crate) const PITCH: f32 = ROW + 2.0;
/// Character advance, for sizing a column in glyphs — [`FACE`]'s own 0.6 em,
/// asserted against the loaded face by `the_advance_constant_is_the_faces_own`
/// rather than trusted. The bitmap fallback's cell is 6 and is *not* this: a
/// column measured in glyphs measures in the glyphs it will actually draw.
pub(crate) const EM: f32 = ROW * 0.6;

/// The face the editor sets its panels in: the workspace's vendored FiraMono
/// (`assets/fonts`, OFL-1.1), embedded rather than opened — an editor whose
/// text disappears because the process was started from another directory is
/// worse than one that costs 170 KiB. Dev-only by construction: §3's deny pin
/// and `xtask dist` keep this crate out of what ships.
const FACE: &[u8] = include_bytes!("../../../assets/fonts/FiraMono-Regular.ttf");

pub(crate) const INK: u32 = 0xffc8_d4e0;
pub(crate) const DIM: u32 = 0xff76_8496;
pub(crate) const ACCENT: u32 = 0xff7f_d0a0;
pub(crate) const CHROME: u32 = 0xff14_1a22;
pub(crate) const HEADER: u32 = 0xff1e_2632;
pub(crate) const BUTTON: u32 = 0xff2a_3444;
/// What a destructive control goes under the pointer — Windows' own close red,
/// because the one control wearing it is the one every desktop paints that
/// colour and an approximation reads as a different button.
pub(crate) const DANGER: u32 = 0xffc4_2b1c;
pub(crate) const LIVE: u32 = 0xff3a_6a4a;
pub(crate) const PICKED: u32 = 0xff2a_4a6a;

/// Rows one wheel notch moves a pane. Three is the desktop's own number, and
/// `gg_platform` divides a trackpad's pixels by the same figure.
pub(crate) const NOTCH: f32 = 3.0;

const SEAM: WidgetId = WidgetId::new("editor.seam");
const BAR_ID: WidgetId = WidgetId::new("editor.scrollbar");
const TAB: WidgetId = WidgetId::new("editor.tab");
const GRIP_ID: WidgetId = WidgetId::new("editor.grip");
const ITEM: WidgetId = WidgetId::new("editor.menu.item");

/// A tab drag in progress.
#[derive(Clone, Copy)]
struct Grab {
    pane: PaneId,
    /// The tab it started on. The drag only *becomes* a drag once the pointer
    /// leaves this — otherwise every click on a tab would end in a drop, and a
    /// drop onto the group you are already in is a rearrangement nobody asked
    /// for.
    from: Rect,
    escaped: bool,
}

/// The editor.
///
/// One per session, ticked by the shell inside `sim_tick` so its edits land in
/// the world before the canonical hash reads it.
pub struct Editor {
    list: DrawList,
    router: Router,
    scan: scan::Scan,
    dock: Dock,
    /// This page's entities and the component mask each carries.
    rows: Vec<(Entity, u64)>,
    /// Scratch for one entity's bytes, refilled per component per tick.
    bytes: Vec<u8>,
    /// The pack the asset browser lists, if the shell was given one.
    pack: Option<gg_assets::Pack>,
    selected: Option<Entity>,
    lane: Option<Lane>,
    step: usize,
    /// The first tree row fetched this tick — the scroll offset in rows, which
    /// is what makes the tree a *window* onto the world rather than a list of
    /// it: ten thousand entities cost the rows the pane can show.
    first_row: usize,
    /// Tree rows the pane had room for last tick — the page size, which is a
    /// property of the layout now that the layout is the operator's.
    per_page: usize,
    /// How far each pane's content is scrolled, in logical units, indexed by
    /// [`PaneId`] — plus [`KNOBS_SLOT`] on the end. Plain state for [`panels`]'
    /// reason: it moves only in response to a click or a wheel notch, and both
    /// arrive through the action map.
    scroll: [f32; SCROLLS],
    /// Wheel notches this tick, spent by whichever pane the pointer is over.
    wheel: i32,
    grab: Option<Grab>,
    /// The menus in the title bar, and which one is down (§6 M15.1 item 5).
    menus: gg_ui::menu::MenuBar,
    /// The tick of the last press on the drag region, for the double-click that
    /// maximizes. Deliberately a *tick* count and not a clock: it is the same
    /// number on a replay, so a gesture nobody can time by hand is still exact.
    /// `None` until there has been one — tick zero is a real tick, and a
    /// sentinel of `0` would make the session's first press a double.
    last_bar_press: Option<u64>,
    /// The primary button last tick, for the release edge a drop needs and the
    /// press edge the title bar does. The router derives its own edges but does
    /// not expose them, and a drag ends somewhere no widget is.
    primary: bool,
    /// The fit the last [`Editor::tick`] laid out under.
    fit: Fit,
    /// The rented face and the atlas its glyphs are packed into (§4.9).
    ///
    /// The editor is the first UI in the engine to set text at the *surface's*
    /// resolution rather than the canvas's: a logical unit is `fit.scale`
    /// pixels, so glyphs are rasterized at [`Editor::px`] and emitted through a
    /// device-space transform — magnifying `gg_ui::font`'s 5×7 bitmap by scale
    /// instead would keep every glyph fixed at the source resolution.
    fonts: Fonts,
    face: FaceId,
    /// Pixels per em the resident glyphs were rasterized at. A changed scale
    /// *replaces* them rather than packing a second size beside the first: the
    /// atlas is one 512² bitmap and six scales of 95 glyphs do not fit in it.
    px: u16,
    /// [`ink_lift`] at `px`, in pixels. Cached with the face it was measured
    /// off, because it is read once per run and moves only when `px` does.
    lift: f32,
    /// Rises whenever [`Editor::coverage`] changed — the atlas's own counter
    /// plus the reloads that reset it.
    font_rev: u64,
    /// The atlas version `font_rev` last accounted for.
    atlas_seen: u64,
    /// What the title bar last asked of the window, until a host takes it.
    window: Option<WindowCommand>,
    /// [`Frame::maximized`], as of this tick — read by the caption button one
    /// call below the frame that carries it.
    maximized: bool,
    /// The editor's own camera (§6 M15.2 item 2) — host state, and held here
    /// rather than by the shell because it is flown by ordinary routed input and
    /// this is what routes it.
    camera: camera::Camera,
    /// The translate handle a press is holding, if any (§6 M15.4 item 3). Host
    /// state like every other panel field, and for the same reason: it is a pure
    /// function of the replayed input stream.
    gizmo: Option<marker::Gizmo>,
    /// Which write a gizmo drag makes (§6 M20 item 10). Host state, like the
    /// drag itself: the mode decides what an *input* means, and the world only
    /// ever sees the result.
    tool: Tool,
    /// [`host::verb::TOOL`] last tick, so the key cycles on its press edge
    /// rather than sixty times a second while it is held.
    tool_was: bool,
    /// Worlds as they were before each edit (§6 M15.4 item 4) — host state,
    /// absent from the world and so from the hash, though re-applying a step
    /// writes through `World` like every other edit and is.
    history: history::History,
    /// The play state the last tick ran under, so a change of it can drop the
    /// history — see [`history`].
    play_was: Play,
    /// Where each axis handle's grab pad was drawn last tick, in logical units.
    /// Read by [`Editor::handle`] — a script cannot compute one, because unlike
    /// every other target in this editor a handle's position is a property of
    /// the world and the camera rather than of the layout.
    arms: [Option<(f32, f32)>; 3],
    /// [`Frame::project`] was `None` this tick — the launcher's state (§6 M15.1
    /// item 4). Held because [`Editor::over_panels`] is asked between ticks, by a
    /// host that has the answer and no frame in its hand.
    launching: bool,
    /// Saves issued this session — the status line, and what a gate greps for.
    saves: u32,
    /// Edits applied this session, for the same reason.
    edits: u32,
    /// §6 M16's question in flight. Owned rather than routed through
    /// [`Commands`]: the answer is text and `Commands` is `Copy`, and the panel
    /// is the only thing that draws it.
    ask: gg_agent::Ask,
    /// The conversation around it — transcript and the line being typed. Host
    /// state, so no prompt is ever in the world or the canonical hash, and a
    /// pure function of the replayed input stream (§6 M16, [`chat`]).
    chat: chat::Chat,
    commands: Commands,
}

/// Claim the editor's knob names, beside every other crate's and *before*
/// config is applied (§6 M40).
///
/// Called by the shell at startup rather than by [`Editor::new`], for the reason
/// the shell states where it calls it: a name the config file or `--set` uses is
/// unknown by an accident of ordering otherwise. Idempotent — [`Editor::new`]
/// calls it too, so a host that forgets still gets the console's completion.
pub fn register() -> Result<(), gg_core::cvar::CVarError> {
    gg_core::cvar::register_all(&[
        &SCALE,
        &history::DEPTH,
        &camera::SPEED,
        &camera::SENSITIVITY,
        &camera::INVERT,
        &panels::LEGEND,
        &marker::MARKERS,
    ])
}

impl Editor {
    /// Build an editor. `pack` is the asset pack the browser lists; a session
    /// without one gets an empty browser rather than no browser.
    #[must_use]
    pub fn new(pack: Option<&std::path::Path>) -> Editor {
        // Best effort, and a second editor in one process is the only way it
        // fails: `ui_scale` reads the static directly, so registration buys the
        // console and the config file a name, not the read path.
        //
        // Late, and that is why [`register`] exists: this runs when the *editor*
        // is built, which is long after `gg_core::config::boot` has applied the
        // config file and `--set`, so a knob named there was "no such cvar"
        // until the shell started registering these two up front (§6 M40).
        let _ = register();
        let pack = pack.and_then(|path| match gg_assets::Pack::open(path) {
            Ok(pack) => Some(pack),
            Err(error) => {
                tracing::warn!(%error, path = %path.display(), "editor: pack unreadable");
                None
            }
        });
        // No window yet, so no monitor to ask: the canvas headless hosts use,
        // at the scale a 96-DPI one would.
        let fit = fit(gg_ecs::boundary::CANVAS, 1.0);
        let px = text_px(fit.scale);
        let (fonts, face, lift) = rent_face(px);
        Editor {
            list: DrawList::default(),
            router: Router::default(),
            scan: scan::Scan::default(),
            dock: Dock::new(default_layout()),
            rows: Vec::new(),
            bytes: Vec::new(),
            pack,
            selected: None,
            lane: None,
            ask: gg_agent::Ask::idle(),
            chat: chat::Chat::default(),
            step: 1,
            first_row: 0,
            per_page: 1,
            scroll: [0.0; SCROLLS],
            wheel: 0,
            grab: None,
            menus: gg_ui::menu::MenuBar::default(),
            last_bar_press: None,
            primary: false,
            fit,
            fonts,
            face,
            px,
            lift,
            font_rev: 1,
            atlas_seen: 0,
            window: None,
            maximized: false,
            camera: camera::Camera::default(),
            gizmo: None,
            tool: Tool::default(),
            tool_was: false,
            history: history::History::default(),
            play_was: Play::default(),
            arms: [None; 3],
            launching: false,
            saves: 0,
            edits: 0,
            commands: Commands::default(),
        }
    }

    /// The dock tree, for a host persisting the layout.
    #[must_use]
    pub fn layout(&self) -> &Node {
        self.dock.root()
    }

    /// Adopt a layout, dropping it for the default if it names a pane this
    /// build has not got, names one twice, or is empty.
    ///
    /// Validated rather than trusted because the source is a file. It no longer
    /// insists on **every** pane (§6 M61): a closed pane is a thing an operator
    /// can now ask for, so a tree missing the inspector is a persisted choice
    /// rather than an editor with no way to get it back — the `view` menu is
    /// the way back, and it is drawn off `Pane::ALL` rather than off the tree.
    /// The other two arms stand: a pane named twice is two widgets sharing an
    /// id every frame, an unknown id is a strip with a nameless tab in it, and
    /// an empty tree is no strip at all.
    ///
    /// Returns whether it was taken.
    pub fn set_layout(&mut self, root: Node) -> bool {
        let mut seen = [0u8; Pane::ALL.len()];
        let mut count = 0;
        let mut unknown = 0;
        walk_panes(&root, &mut |pane| {
            count += 1;
            match seen.get_mut(pane.0 as usize) {
                Some(slot) => *slot += 1,
                None => unknown += 1,
            }
        });
        let sound = count > 0 && unknown == 0 && seen.iter().all(|n| *n <= 1);
        if sound {
            self.dock.set_root(root);
        } else {
            tracing::warn!(count, unknown, "editor: layout is not sound; default kept");
        }
        sound
    }

    /// Close `pane` if it is docked, or put it back if it is not (§6 M61).
    ///
    /// Where it lands is [`home`]'s answer, and the last pane cannot be closed
    /// — [`gg_ui::dock::Dock::close`]'s floor, since a dock with nothing in it
    /// has no strip to reopen from.
    pub fn toggle_pane(&mut self, pane: Pane) {
        let shown = match self.dock.holds(pane.id()) {
            true => !self.dock.close(pane.id()),
            false => match home(pane, &|other| self.dock.holds(other.id())) {
                Some(onto) => self.dock.open(pane.id(), onto.id()),
                None => false,
            },
        };
        tracing::info!(pane = pane.title(), shown, "editor: pane toggled");
    }

    /// Put the layout back the way it opens.
    pub fn reset_layout(&mut self) {
        self.dock.set_root(default_layout());
    }

    /// Lay the panes out for a surface `extent` across, at `dpi`, **without**
    /// ticking.
    ///
    /// What a caller that needs to know where a widget will be before clicking
    /// it uses — [`session`]'s aiming helpers, and so the gate and the golden
    /// scene that drive them.
    pub fn place(&mut self, extent: (u32, u32), dpi: f32) {
        self.fit = fit(extent, dpi);
        // A resize across a scale boundary is the only thing that gets here
        // with a new size, and it re-rents rather than packing beside the old
        // one — see [`Editor::px`].
        let px = text_px(self.fit.scale);
        if px != self.px {
            let (fonts, face, lift) = rent_face(px);
            self.lift = lift;
            self.fonts = fonts;
            self.face = face;
            self.px = px;
            self.font_rev += 1;
            self.atlas_seen = self.fonts.version();
        }
        let open = self.menus.open();
        self.resolve_menus(open);
        self.resolve_dock();
        self.per_page = self.dock.body_of(Pane::Tree.id()).map_or(1, |body| {
            ((panels::tree_list(body).h / PITCH) as usize).max(1)
        });
        // Which rows to fetch, off the offset the last tick clamped. A row of
        // slack at each end: the top row is usually half-scrolled and the
        // bottom one always is.
        self.first_row = (self.scroll[Pane::Tree.id().0 as usize] / PITCH).max(0.0) as usize;
    }

    /// Run one tick of editor over `world` and return what it asks the host to
    /// do.
    ///
    /// The panels format their labels with `format!` and so allocate a few dozen
    /// strings a tick. M13's zero-allocation gate covers `gg-ui` and the overlay
    /// and deliberately does not reach here, because an editor is not in a
    /// shipping frame budget — and the reason this was once a P2, that it is the
    /// wrong shape at ten thousand rows, stopped being true when the tree became
    /// paged: [`scan::Scan::page`] fills `rows` with a screenful, so the count
    /// formatted is bounded by the pane's height and not by the world's size.
    /// `gg_ui::Scratch` is still the shape that would allocate nothing, and it
    /// costs every label a span lookup to dodge the `&mut self` the buttons
    /// need — which is a worse trade here than the allocations are a cost.
    ///
    /// Called from the sim tick and not the frame, for `gg_ui::Ui`'s reason: an
    /// edit has to be in the world before the hash reads it, and a headless run
    /// must route clicks exactly as a windowed one does — which is what lets a
    /// gate replay an editor session with no window anywhere (§1.5).
    pub fn tick(&mut self, world: &mut gg_ecs::World, tick: &Tick, frame: &Frame) -> Commands {
        self.commands = Commands::default();
        // `self.window` is deliberately *not* cleared here: it is produced by a
        // tick and consumed by a frame, and a frame owing two ticks would
        // otherwise clear the press edge's command before the host ever read it
        // (§6 M15.1 item 5). Below 60 fps that is every frame. The `take` in
        // `take_window_command` is the reset.
        self.wheel = tick.scroll;
        self.maximized = frame.maximized;
        self.launching = frame.project.is_none();
        // A step recorded on one side of the transport cannot be restored on the
        // other: it would put a play-mode world back into a stopped scene, or a
        // stopped one on top of a running sim (§6 M15.4 item 4).
        if frame.play != self.play_was {
            self.play_was = frame.play;
            self.history.clear();
        }
        self.place(frame.extent, frame.dpi);
        self.router.begin(tick, self.fit.canvas);
        // Before the panels, so a camera moved this tick is the one the viewport
        // tag and the shell's extract both see. It no longer reads the router's
        // pointer at all — a look drag is a device delta (`camera`) — so the
        // order against `begin` above is now free rather than load-bearing.
        //
        // The nav is assembled here because every part of it has an owner
        // elsewhere: the pane is the dock's, the field of view a CVar, the
        // selection this struct's. Built from the framing the *last* tick drew,
        // which is the same frame of lag every hit test already has.
        self.camera.fly(world, frame, self.nav(world));
        // The tool cycles on the key's press edge — and unconditionally, unlike
        // the gizmo it steers: pressing it while the scene plays should leave
        // the mode the operator wanted waiting when they stop.
        let cycling = frame
            .input
            .and_then(|input| camera::id(input, host::verb::TOOL).map(|a| input.pressed(a)))
            .unwrap_or(false);
        if cycling && !self.tool_was {
            self.tool = self.tool.next();
            tracing::info!(tool = self.tool.label(), "editor: tool");
        }
        self.tool_was = cycling;
        self.scan.run(world);
        self.scan
            .page(world, self.first_row, self.per_page + 1, &mut self.rows);

        self.list.clear();
        self.list.push_transform(self.fit.offset, self.fit.scale);

        self.topbar(frame, tick);
        self.draw_seams();
        self.panes(world, frame);
        // After the panes, because a dropped-down menu hangs over them and the
        // last declaration is the one a click reaches (§4.9's router).
        self.menu_popup(world, frame);
        self.resize_grips(tick);
        self.drag(tick);

        if frame.draw_cursor {
            // Not a fallback for a missing OS cursor so much as the definition:
            // this position is an integral of the replayed axis stream and
            // exists whether or not a mouse does. A windowed host steers the
            // two into agreement instead (§6 M15.1), which is why
            // `Frame::draw_cursor` is false there.
            let at = self.router.pointer().position();
            gg_ui::draw::cursor(&mut self.list, at);
        }
        self.list.pop_transform();
        // Whatever this tick had to rasterize — a status line the warm-up did
        // not cover is one glyph, and the host still has to hear about it.
        if self.atlas_seen != self.fonts.version() {
            self.atlas_seen = self.fonts.version();
            self.font_rev += 1;
        }
        self.primary = tick.primary;
        self.commands
    }

    /// What the camera needs to know about the picture it is flying through
    /// (§6 M20 item 10) — the pane's shape, the lens, and what is selected.
    ///
    /// The scales come out of [`pick::Lens`] rather than being derived a second
    /// time here, for the reason the lens exists at all: a pan that moved by a
    /// different metres-per-pixel than the picture was drawn with is a scene
    /// that slides out from under the pointer, and nothing about either number
    /// alone would look wrong.
    fn nav(&self, world: &gg_ecs::World) -> camera::Nav {
        let rendered = self.viewport_rect();
        let (w, h) = (
            f64::from(rendered.width.max(1)),
            f64::from(rendered.height.max(1)),
        );
        let aspect = w / h;
        let eye =
            self.eye(gg_ecs::boundary::Eye::of(world).unwrap_or(gg_ecs::boundary::Eye::ORIGIN));
        let fov = gg_render::cvars::FOV.float();
        let lens = pick::Lens::new(eye, fov, aspect, gg_render::cvars::NEAR.float());
        let target = self
            .selected
            .and_then(|entity| world.get::<gg_ecs::boundary::Renderable>(entity).copied());
        // A perspective pan tracks the hand only at one depth, and this picks
        // it: the selection's, or a room's width when there is none. Under a
        // flat eye the argument goes unread — there is only one scale.
        let depth = target.map_or(camera::PAN_DEPTH, |box_| {
            lens.project(box_.position).1.max(camera::PAN_DEPTH * 0.1)
        });
        camera::Nav {
            metres_per_unit: lens.metres_per_unit(depth, h),
            aspect,
            half_fov_tan: gg_math::sim::tan(fov * 0.5),
            target,
            // Only over the game: the panes tile, so a notch anywhere else is
            // some pane's scroll and taking it here would be taking it twice.
            wheel: match self.over_panels() {
                true => 0,
                false => self.wheel,
            },
        }
    }

    /// The title bar's strip, across the top of the canvas and outside the dock.
    #[must_use]
    pub fn bar_rect(&self) -> Rect {
        Rect::new(0.0, 0.0, self.fit.canvas.0 as f32, BAR_H)
    }

    /// Lay the menu strip out with `open` dropped down, into `menus`.
    ///
    /// The bar to write into is an argument so a caller can ask where an item
    /// *would* be without opening it — which is how [`session::aim`] aims at
    /// `file → save` off a closed editor.
    pub(crate) fn menus_into(&self, menus: &mut gg_ui::menu::MenuBar, open: Option<usize>) {
        let bar = self.bar_rect();
        menus.resolve(bar, panels::strip_left(bar, MAC), MENUS, open, &text_width);
    }

    fn resolve_menus(&mut self, open: Option<usize>) {
        let mut menus = core::mem::take(&mut self.menus);
        self.menus_into(&mut menus, open);
        self.menus = menus;
    }

    /// The eight resize grips, declared last so a press at the window's edge is
    /// a resize and not whatever pane happens to reach that far (§6 M15.1 item
    /// 5). Nothing is drawn: a border you can see is chrome, and this is the
    /// frame the OS is no longer providing.
    fn resize_grips(&mut self, tick: &Tick) {
        let pressed = tick.primary && !self.primary;
        for (i, (edge, rect)) in grips(self.fit.canvas).into_iter().enumerate() {
            let response = self.router.hit(GRIP_ID.indexed(i as u64), rect);
            if response.held && pressed {
                self.window = Some(WindowCommand::Resize(edge));
            }
        }
    }

    /// The open menu, hanging over the panes.
    fn menu_popup(&mut self, world: &mut gg_ecs::World, frame: &Frame) {
        let Some(panel) = self.menus.panel() else {
            return;
        };
        let Some(menu) = self.menus.open() else {
            return;
        };
        // Lighter than the bar it hangs off, not accented: a menu is a surface
        // floating over the panes and the only thing marking it is its edge.
        self.plate(panel, HEADER, BUTTON);
        let items: Vec<Rect> = self.menus.items().to_vec();
        for (i, rect) in items.iter().enumerate() {
            let label = MENUS
                .get(menu)
                .and_then(|m| m.items.get(i))
                .copied()
                .unwrap_or("?");
            // Left-aligned, unlike every other cell here: a menu is a list of
            // names and centring them would make it a row of buttons.
            let response = self.router.hit(ITEM.indexed(i as u64), *rect);
            // A docked pane's row is lit rather than ticked (§6 M61): a marker
            // column would widen every panel in the strip by a glyph the other
            // two menus have no use for, and `gg_ui::menu` measures a label and
            // knows nothing about state.
            let on = matches!(menu_action(menu, i), Some(MenuAction::Toggle(pane)) if self.dock.holds(pane.id()));
            if on {
                self.list.rect(*rect, LIVE);
            }
            if response.hovered || response.held {
                self.list.rect(*rect, PICKED);
            }
            self.label_mid(*rect, rect.x + gg_ui::menu::PAD, label, INK);
            if response.clicked {
                self.menus.close();
                self.run_menu(world, menu, i, frame);
            }
        }
    }

    /// What a menu item does, once it has been clicked.
    fn run_menu(&mut self, world: &mut gg_ecs::World, menu: usize, item: usize, frame: &Frame) {
        match menu_action(menu, item) {
            Some(MenuAction::Save) => {
                self.commands.save = true;
                self.saves += 1;
                tracing::info!(path = frame.save_path, "editor: save requested");
            }
            Some(MenuAction::Quit) => self.window = Some(WindowCommand::Close),
            // Counted as an edit, because it is one: the world changed, through
            // `World`, and the canonical hash will say so.
            Some(MenuAction::Undo) => {
                let (back, forward) = self.history.depths();
                // Logged inside the branch that did something: a line printed
                // whether or not the ring had a step in it would say a menu item
                // was clicked, which is not the claim any gate wants.
                match self.history.undo(world) {
                    true => {
                        self.edits += 1;
                        tracing::info!(back, forward, "editor: undo");
                    }
                    false => tracing::info!("editor: nothing to undo"),
                }
            }
            Some(MenuAction::Redo) => {
                let (back, forward) = self.history.depths();
                match self.history.redo(world) {
                    true => {
                        self.edits += 1;
                        tracing::info!(back, forward, "editor: redo");
                    }
                    false => tracing::info!("editor: nothing to redo"),
                }
            }
            Some(MenuAction::ResetLayout) => {
                self.reset_layout();
                tracing::info!("editor: layout reset");
            }
            Some(MenuAction::Toggle(pane)) => self.toggle_pane(pane),
            None => tracing::warn!(menu, item, "editor: menu item does nothing"),
        }
    }

    /// Whether the layout holds `pane` at all (§6 M61) — as opposed to holding
    /// it behind another tab, which is [`pane_body`](Self::pane_body) saying
    /// `None`.
    #[must_use]
    pub fn pane_docked(&self, pane: Pane) -> bool {
        self.dock.holds(pane.id())
    }

    /// Where a pane's contents are, in logical units, or `None` when it is
    /// behind another tab.
    #[must_use]
    pub fn pane_body(&self, pane: Pane) -> Option<Rect> {
        self.dock.body_of(pane.id())
    }

    /// Bring `pane`'s tab up, as a click on it would (§6 M16 exit row 4).
    /// A session author needs this: aims are geometry against a layout, and
    /// [`pane_body`](Self::pane_body) declines a pane behind another tab.
    /// Requires a prior [`place`](Self::place) — resolving needs a canvas.
    pub fn raise(&mut self, pane: Pane) {
        self.dock.activate(pane.id());
        self.resolve_dock();
    }

    /// Lay the dock out over the placed canvas — what changing which pane a
    /// group shows must be followed by, since the resolved snapshot is what
    /// [`pane_body`](Self::pane_body) answers from.
    fn resolve_dock(&mut self) {
        let canvas = self.fit.canvas;
        let area = Rect::new(
            0.0,
            BAR_H,
            canvas.0 as f32,
            (canvas.1 as f32 - BAR_H).max(0.0),
        );
        self.dock.resolve(area, tab_width);
    }

    /// Where a pane's tab is, in logical units. Present whether or not the pane
    /// is the visible one — the tab is how it becomes visible.
    #[must_use]
    pub fn tab_rect(&self, pane: Pane) -> Option<Rect> {
        self.dock
            .tabs()
            .iter()
            .find(|t| t.pane == pane.id())
            .map(|t| t.rect)
    }

    /// The seams of the current layout, for a caller that wants to drag one.
    #[must_use]
    pub fn seams(&self) -> &[dock::Seam] {
        self.dock.seams()
    }

    /// Tree rows the world pane has room for at the current layout.
    #[must_use]
    pub fn per_page(&self) -> usize {
        self.per_page
    }

    /// The seams between panes: chrome, and the only widget in the editor whose
    /// response is read while *held* rather than on a click.
    fn draw_seams(&mut self) {
        // Copied out: `steer` takes the dock mutably and the loop declares
        // widgets, which takes the router mutably.
        let seams: Vec<dock::Seam> = self.dock.seams().to_vec();
        for seam in seams {
            let response = self.router.hit(SEAM.indexed(seam.path.key()), seam.rect);
            let color = match response.held || response.hovered {
                true => ACCENT,
                false => HEADER,
            };
            self.list.rect(seam.rect, color);
            if response.held {
                let (x, y) = self.router.pointer().position();
                let along = match seam.axis {
                    Axis::Horizontal => x,
                    Axis::Vertical => y,
                };
                self.dock.steer(seam.path, seam.span, along);
            }
        }
    }

    /// Every group: its strip of tabs, then whichever pane is up.
    fn panes(&mut self, world: &mut gg_ecs::World, frame: &Frame) {
        let groups: Vec<dock::Group> = self.dock.groups().to_vec();
        let tabs: Vec<dock::Tab> = self.dock.tabs().to_vec();
        for group in &groups {
            let showing = self.dock.active_of(group).and_then(Pane::from_id);
            // The game's body is a hole, not a plate: the renderer has already
            // drawn into it (`viewport_rect`) and an opaque fill over it is a
            // rectangle of chrome where the game was. Its border is
            // `panels::viewport`'s, in the accent, and the strip still needs its
            // own — `region` is exactly `strip ∪ body`, so nothing else is left
            // uncovered.
            if showing != Some(Pane::Viewport) {
                self.plate(group.region, CHROME, HEADER);
            }
            self.list.rect(group.strip, HEADER);
            let active = group.first + group.active;
            for (i, tab) in tabs
                .iter()
                .enumerate()
                .skip(group.first as usize)
                .take(group.count as usize)
            {
                let Some(pane) = Pane::from_id(tab.pane) else {
                    continue;
                };
                let up = i as u32 == active;
                let response = self.cell(
                    TAB.indexed(u64::from(tab.pane.0)),
                    tab.rect,
                    pane.title(),
                    up,
                );
                if response.clicked {
                    self.dock.activate(tab.pane);
                }
                // `held` and not `clicked`: a press that leaves the tab is a
                // drag, and a click never reports one because it is a release
                // over the widget that took the press (§4.9).
                if response.held && self.grab.is_none() {
                    self.grab = Some(Grab {
                        pane: tab.pane,
                        from: tab.rect,
                        escaped: false,
                    });
                }
            }
            let Some(pane) = showing else {
                continue;
            };
            let body = group.body;
            match pane {
                Pane::Tree => self.tree(world, body),
                Pane::Viewport => self.viewport(world, frame, body),
                Pane::Cvars => self.cvars(body),
                Pane::Assets => self.assets(body),
                Pane::Perf => self.perf(body, frame),
                Pane::Inspector => self.inspector(world, body),
                Pane::Agent => self.agent(body, frame),
                Pane::Render => self.render(body, frame),
            }
        }
    }

    /// Carry a tab drag, and land it on release.
    fn drag(&mut self, tick: &Tick) {
        let Some(mut grab) = self.grab else {
            return;
        };
        let (x, y) = self.router.pointer().position();
        grab.escaped |= !grab.from.contains(x, y);
        self.grab = Some(grab);

        let released = self.primary && !tick.primary;
        if !grab.escaped {
            // Still a click as far as anyone can tell.
            if released {
                self.grab = None;
            }
            return;
        }
        let drop = self.dock.drop_at(x, y);
        if let Some(rect) = drop.and_then(|d| self.dock.preview(d)) {
            // Two edges of the target rather than a filled overlay: the pane
            // underneath is what the operator is aiming at and a wash over it
            // hides the thing being aimed at.
            self.outline(rect, ACCENT);
            self.outline(rect.inset(1.0), ACCENT);
        }
        if released {
            self.grab = None;
            if let Some(drop) = drop.filter(|d| d.onto != grab.pane) {
                self.dock.move_pane(grab.pane, drop);
                tracing::info!(
                    pane = Pane::from_id(grab.pane).map(Pane::title).unwrap_or("?"),
                    onto = Pane::from_id(drop.onto).map(Pane::title).unwrap_or("?"),
                    zone = ?drop.zone,
                    "editor: pane docked"
                );
            }
        }
    }

    /// Where the pointer is, in logical units — what a host warps a system
    /// cursor to when it hands the pointer back (§6 M15.1).
    #[must_use]
    pub fn pointer(&self) -> (f32, f32) {
        self.router.pointer().position()
    }

    /// Where the game belongs on the surface, in **physical** pixels, for
    /// `gg_render::Renderer::set_viewport`.
    ///
    /// This is what makes the frame a render target sized to the pane rather
    /// than the middle of a window-sized frame showing through a hole: an
    /// object at the edge of the pane is at the edge of the picture.
    ///
    /// **Zero-sized** when the game is docked behind another tab, which draws
    /// nothing rather than falling back to the whole window — a hidden pane is
    /// hidden, and `None` would mean "no viewport" and fill the surface.
    ///
    /// Rounded to whole pixels from the *edges* rather than by scaling the
    /// size, so the rectangle cannot end a pixel short of the outline that
    /// frames it.
    #[must_use]
    pub fn viewport_rect(&self) -> gg_render::Viewport {
        // Inset by the one unit `panels::viewport` outlines it with: the border
        // says where the game ends, and a frame drawn over it would be the game
        // saying.
        let Some(inner) = self.dock.body_of(Pane::Viewport.id()).map(|b| b.inset(1.0)) else {
            return gg_render::Viewport {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            };
        };
        let px = |v: f32| (v * self.fit.scale).round().max(0.0) as u32;
        let (x, y) = (px(inner.x), px(inner.y));
        gg_render::Viewport {
            x,
            y,
            width: px(inner.right()).saturating_sub(x),
            height: px(inner.bottom()).saturating_sub(y),
        }
    }

    /// Where the viewport is looked at from, given the eye the *game* declared
    /// (§6 M15.2 item 2).
    ///
    /// `game` back whenever the scene is not stopped, and back unchanged before
    /// the first stop — so a host may call this unconditionally and a session
    /// that never stops renders exactly as it did before this existed.
    #[must_use]
    pub fn eye(&self, game: gg_ecs::boundary::Eye) -> gg_ecs::boundary::Eye {
        self.camera.eye(game)
    }

    /// [`Editor::eye`] as the pair a frame blends between — the previous tick's
    /// and this one's (§6 M63, §4.1).
    ///
    /// A host that renders once per tick may keep calling [`Editor::eye`] and
    /// see no difference; one whose panel refreshes four times a tick wants
    /// this, for the same reason `gg_extract::Extracted::interpolate` exists for
    /// the game's instances. Both halves are `game` whenever the editor's camera
    /// is not the one being rendered from, so the blend is the identity there.
    #[must_use]
    pub fn eyes(
        &self,
        game: gg_ecs::boundary::Eye,
    ) -> (gg_ecs::boundary::Eye, gg_ecs::boundary::Eye) {
        self.camera.eyes(game)
    }

    /// The editor's camera has the pointer this tick: a host with a window
    /// should hold and hide the OS arrow (§6 M63).
    ///
    /// Distinct from `gg_runtime`'s "the *game* took the pointer" — that one
    /// also stops feeding the editor and starts feeding the game, and this one
    /// changes nothing about who is fed. The only thing it asks for is the
    /// arrow, which is why a host that ignores it still routes every click
    /// correctly and merely lets the cursor wander out of the window mid-drag,
    /// which is what every session before this did.
    #[must_use]
    pub fn flying(&self) -> bool {
        self.camera.flying()
    }

    /// The turn a *frame* may add to the editor's camera for counts no tick has
    /// spent (§6 M56's latch, §6 M63's camera), or `None` on every tick the
    /// operator is not turning it.
    ///
    /// Handed the host's [`gg_input::Input`] because the axis *ids* are the
    /// host's — `editor_look_x` is a different index over every game — and the
    /// rate, the sign and the clamp are the camera's. What comes back is exactly
    /// the shape a game's own `Look` has, so the shell latches both through one
    /// path.
    #[must_use]
    pub fn look(&self, input: &gg_input::Input) -> Option<gg_ecs::boundary::Look> {
        self.camera.look(input)
    }

    /// What the agent panel's prompt field holds, unsent (§6 M16). Empty is
    /// both "nothing typed" and "just sent", which is the same thing to look at.
    #[must_use]
    pub fn prompt(&self) -> &str {
        &self.chat.prompt
    }

    /// Whether keystrokes are wanted — the agent panel's prompt has focus.
    ///
    /// Asked by the host *between* ticks, which is where a keystroke arrives, so
    /// it answers about the layout the operator can currently see. A `false`
    /// keeps the character out of the recording entirely (§6 M16): a replay
    /// carrying every `W` pressed while playing would be a text channel full of
    /// input that already has verbs.
    #[must_use]
    pub fn wants_text(&self) -> bool {
        self.router.focused() == Some(panels::PROMPT)
    }

    /// Whether the pointer is over a panel rather than over the viewport.
    ///
    /// One half of what the host decides a *press* is (§6 M15.1): the editor
    /// and the game share one physical mouse, and a click on `pause` must not
    /// also fire whatever the game bound to that button. The other half is the
    /// transport — the viewport is the game's only while the game runs, which
    /// is `panels`' pick rule and `gg_runtime`'s `takes_pointer` reading it
    /// from the host's side. Not the dead-frame rule either; that one is "the
    /// editor holds the pointer", which is true over the viewport as well until
    /// a press there takes it. Reads the pointer the last tick left, which is
    /// the same frame of lag every hit test already has.
    #[must_use]
    pub fn over_panels(&self) -> bool {
        let (x, y) = self.router.pointer().position();
        // A dropped-down menu and a resize grip both sit *over* the game when
        // the game is under them, and a press on either is the editor's.
        // And the picker, which *is* panels: with no project the game pane holds
        // a list of them (§6 M15.1 item 4), so a press there is the editor's and
        // handing the pointer to a game that does not exist would leave the
        // launcher unclickable for the rest of the session.
        if self.menus.contains(x, y) || self.resize_edge().is_some() || self.launching {
            return true;
        }
        !self
            .dock
            .body_of(Pane::Viewport.id())
            .is_some_and(|r| r.inset(1.0).contains(x, y))
    }

    /// The geometry the last [`tick`](Self::tick) built, for the shell to append
    /// to the frame's UI stream.
    #[must_use]
    pub fn vertices(&self) -> &[UiVertex] {
        self.list.vertices()
    }

    /// The coverage atlas [`Editor::vertices`] cut their glyphs from (§4.9).
    ///
    /// A host that draws them must be holding *this* atlas and not
    /// `gg_ui::atlas::fallback()` — the editor's are rasterized outlines packed
    /// below the bitmap band, and against the fallback they sample blank texels
    /// and the panels come out with no text on them.
    #[must_use]
    pub fn coverage(&self) -> gg_render::ui::Coverage<'_> {
        self.fonts.coverage()
    }

    /// Rises whenever [`Editor::coverage`] changed; a host uploads on a change
    /// and never otherwise. Not the atlas's own counter, which restarts when a
    /// new size replaces the resident glyphs.
    #[must_use]
    pub fn font_revision(&self) -> u64 {
        self.font_rev
    }

    /// Which resize border the pointer is over, if any (§6 M15.1 item 5).
    ///
    /// For the host to say so with the system cursor: the border is deliberately
    /// not drawn, so the arrows are the only thing that reveals it. Reads the
    /// pointer the last tick left, which is the same frame of lag the hit test
    /// that acts on it has — so the shape and the gesture never disagree.
    #[must_use]
    pub fn resize_edge(&self) -> Option<Edge> {
        let (x, y) = self.router.pointer().position();
        edge_at(self.fit.canvas, x, y)
    }

    /// What the title bar asked of the window, taken once (§6 M15.1 item 5).
    /// `None` in the overwhelming majority of ticks, and in *every* tick of a
    /// session that never touched the bar.
    pub fn take_window_command(&mut self) -> Option<WindowCommand> {
        self.window.take()
    }

    /// What a gizmo drag currently writes (§6 M20 item 10) — read by tests and
    /// by the session gate, which cannot see the chip that says so.
    #[must_use]
    pub fn tool(&self) -> Tool {
        self.tool
    }

    /// The selected entity, if any — read by tests and by the session gate.
    #[must_use]
    pub fn selected(&self) -> Option<Entity> {
        self.selected
    }

    /// Where the `axis`-th translate handle was drawn last tick, in logical
    /// units — `0` is world X, `1` Y, `2` Z (§6 M15.4 item 3).
    ///
    /// `None` when there is no gizmo to grab: nothing selected, the selection is
    /// not a `Renderable`, the scene is not stopped, or it is behind the camera.
    /// The one aiming point in this editor that a script cannot compute for
    /// itself, which is why it is asked for rather than derived.
    #[must_use]
    pub fn handle(&self, axis: usize) -> Option<(f32, f32)> {
        self.arms.get(axis).copied().flatten()
    }

    /// Edits applied and saves requested this session.
    #[must_use]
    pub fn tally(&self) -> (u32, u32) {
        (self.edits, self.saves)
    }

    /// Scroll `body`, which holds `content` units of rows, for `pane`.
    ///
    /// Spends this tick's wheel notches if the pointer is over the body, drags
    /// the bar if it is held, draws the bar, and stores the clamped offset back.
    ///
    /// Two obligations on the caller, both of which the router would otherwise
    /// silently break: clip to [`Scroll::view`], and declare **only** the rows
    /// [`Scroll::row`] answers for — a widget outside the view is invisible and
    /// still clickable, under whichever pane is actually up there.
    pub(crate) fn scrollable(&mut self, pane: Pane, body: Rect, content: f32) -> Scroll {
        self.scrollable_at(pane.id().0, body, content)
    }

    /// [`Editor::scrollable`] against a bare slot, for the one pane with two
    /// lists in it (§6 M61's render pane, whose views and knobs scroll apart).
    /// A pane's own slot is its [`PaneId`]; [`KNOBS_SLOT`] is past the last of
    /// them, which is why [`Editor::scroll`] is one longer than `Pane::ALL`.
    pub(crate) fn scrollable_at(&mut self, slot: u16, body: Rect, content: f32) -> Scroll {
        let mut offset = self.scroll.get(slot as usize).copied().unwrap_or(0.0);
        let (px, py) = self.router.pointer().position();
        // Whichever pane the pointer is over takes the notch, and no chaining
        // is needed to say so: the bodies tile, so at most one contains it.
        if self.wheel != 0 && body.contains(px, py) {
            offset -= self.wheel as f32 * NOTCH * PITCH;
        }
        let mut scroll = Scroll::new(body, content, offset);
        if let Some((track, thumb)) = scroll.bar {
            let response = self.router.hit(BAR_ID.indexed(u64::from(slot)), track);
            if response.held {
                scroll = Scroll::new(body, content, scroll.offset_at(py));
            }
            let lit = response.held || response.hovered;
            self.list.rect(track, HEADER);
            let thumb = scroll.bar.map_or(thumb, |(_, moved)| moved);
            // `DIM` and not `BUTTON` at rest: the rows behind it are plates in
            // `BUTTON`, so a thumb in that colour is a control the operator can
            // only find by knowing where it is.
            self.list.rect(thumb, if lit { ACCENT } else { DIM });
        }
        if let Some(stored) = self.scroll.get_mut(slot as usize) {
            *stored = scroll.offset;
        }
        scroll
    }

    /// Move a pane's offset by `by` logical units — a page button, or anything
    /// else that scrolls without a pointer on the bar. Clamped by the next
    /// [`Editor::scrollable`], which is the only place that knows the content's
    /// height.
    pub(crate) fn scroll_by(&mut self, pane: Pane, by: f32) {
        if let Some(offset) = self.scroll.get_mut(pane.id().0 as usize) {
            *offset = (*offset + by).max(0.0);
        }
    }

    /// A filled rectangle with a one-unit inner border — the shape every panel
    /// and every button in this crate is.
    pub(crate) fn plate(&mut self, rect: Rect, fill: u32, edge: u32) {
        self.list.rect(rect, edge);
        self.list.rect(rect.inset(1.0), fill);
    }

    /// A clickable cell. Hit-tested against the *logical* rect, which is the
    /// space the pointer is integrated in — so a click replays at the extent it
    /// was recorded at (see [`ui_scale`]).
    pub(crate) fn button(&mut self, id: WidgetId, rect: Rect, label: &str, on: bool) -> bool {
        self.cell(id, rect, label, on).clicked
    }

    /// [`Editor::button`] with the whole response, for the one caller that needs
    /// more than "was it clicked" — a tab, where a press that walks off is a
    /// drag rather than a change of mind.
    fn cell(&mut self, id: WidgetId, rect: Rect, label: &str, on: bool) -> Response {
        let response = self.router.hit(id, rect);
        let fill = match (on, response.hovered || response.held) {
            (true, _) => LIVE,
            (false, true) => PICKED,
            (false, false) => BUTTON,
        };
        self.plate(rect, fill, HEADER);
        let x = (rect.x + (rect.w - text_width(label)) * 0.5).floor();
        self.label_mid(rect, x, label, INK);
        response
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
    ///
    /// The run sits at the *top* of `rect`, so a table's rows are its own rows
    /// and not `rect.h` divided by anything. A cell taller than [`ROW`] wants
    /// [`Editor::label_mid`].
    pub(crate) fn label(&mut self, rect: Rect, text: &str, color: u32) {
        self.clipped_text(rect, rect.x, rect.y, text, color);
    }

    /// One row of text from `x`, centred down `rect` and cut to it — what a
    /// cell taller than a row needs so its ink is on the cell's middle rather
    /// than hanging off its top edge.
    pub(crate) fn label_mid(&mut self, rect: Rect, x: f32, text: &str, color: u32) {
        let y = (rect.y + (rect.h - ROW) * 0.5).floor();
        self.clipped_text(rect, x, y, text, color);
    }

    /// One run at `(x, y)`, cut to `rect`.
    ///
    /// The clip is `rect` horizontally — which is the whole point of it, a
    /// column that does not write into its neighbour — and [`BLEED`] looser top
    /// and bottom, because a real face's braces and descenders reach past the
    /// row its ink is sized to and a tight clip would slice them.
    fn clipped_text(&mut self, rect: Rect, x: f32, y: f32, text: &str, color: u32) {
        self.list.push_clip(Rect::new(
            rect.x,
            rect.y - BLEED,
            rect.w,
            rect.h + BLEED * 2.0,
        ));
        self.text(x, y, text, color);
        self.list.pop_clip();
    }

    /// One line of text, `(x, y)` being the top-left of its row in logical
    /// units, drawn at the surface's own resolution.
    ///
    /// Two halves and both matter. The run is rasterized at [`Editor::px`] —
    /// pixels, not units — and emitted through a transform that undoes the
    /// fit's, so a stem is one *pixel* wide rather than one unit magnified.
    /// And the origin is rounded to a whole pixel, because `Fonts::layout`
    /// rounds each glyph to whole texels *relative to it*: a run starting on a
    /// half pixel puts every glyph in it half a texel off, which for a
    /// nearest-sampled atlas is a lost row of coverage.
    fn text(&mut self, x: f32, y: f32, text: &str, color: u32) {
        if self.face == FaceId::FALLBACK {
            self.list.text(x, y, text, color);
            return;
        }
        let scale = self.fit.scale;
        self.fonts.layout(self.face, self.px, text);
        self.list.push_transform((0.0, 0.0), 1.0 / scale);
        self.list.glyphs(
            (x * scale).round(),
            (y * scale).round() - self.lift,
            self.fonts.glyphs(),
            color,
        );
        self.list.pop_transform();
    }
}

/// A tab's width: its label plus the padding a plate needs either side.
pub(crate) fn tab_width(pane: PaneId) -> f32 {
    let label = Pane::from_id(pane).map_or("?", Pane::title);
    text_width(label) + 8.0
}

/// How wide `text` will be drawn, in logical units. One advance per character:
/// [`FACE`] is monospace, which is what lets a column be a count of glyphs.
pub(crate) fn text_width(text: &str) -> f32 {
    text.chars().count() as f32 * EM
}

/// Pixels per em at `scale`. [`ROW`] of them, which is what makes [`EM`] the
/// face's own advance and a line of its ink about a row tall.
fn text_px(scale: f32) -> u16 {
    (ROW * scale).round().clamp(1.0, f32::from(u16::MAX)) as u16
}

/// How far above its row a run is set, in pixels: what centres the **cap band**
/// on a [`ROW`]-tall row, a row being exactly `px` tall by [`text_px`].
///
/// Centring the *line box* is the obvious thing and is wrong: half the descent
/// is air, so a label without a descender ("file", "view") sits that much high
/// and its ascenders climb out of the cell above it. Cap-centred, a label reads
/// level whatever letters are in it, and only a descender leaves the row — into
/// the [`BLEED`] `clipped_text` allows.
///
/// Measured off a rasterized `H` rather than the face's `cap_height`, because
/// the scaler *hints*: at 8 px per em it grid-fits a 5.6 px cap to 6, and a lift
/// derived from the metric is then a whole pixel out — which at that size is an
/// eighth of the row. Held at every scale by `the_face_fills_its_row`.
fn ink_lift(fonts: &mut Fonts, face: FaceId, px: u16) -> f32 {
    fonts.layout(face, px, "H");
    let Some(cap) = fonts.glyphs().first().map(|g| g.rect) else {
        return 0.0;
    };
    ((cap.y + cap.bottom() - f32::from(px)) * 0.5).round()
}

/// How far outside its row a run may draw, in logical units. One line of ink is
/// about 1.25 rows tall — a brace over a descender — so a row centred on its
/// text overhangs by an eighth of a row either side, and a clip that did not
/// allow it would take the top off every `{`. Exactly the air [`PITCH`] leaves,
/// so the permission stops at the next row's own.
const BLEED: f32 = PITCH - ROW;

/// Load [`FACE`] and warm the atlas with the glyphs the panels are made of.
///
/// Warmed on purpose: the alternative is an atlas that grows over the first few
/// frames, and every growth is a re-upload the host has to notice mid-frame.
/// A face that will not parse leaves [`FaceId::FALLBACK`], which draws the
/// bitmap — smaller and blockier, and visibly still an editor.
///
/// Returns [`ink_lift`] with them, because it is measured off the raster and so
/// is a property of this face at this size and of nothing else.
fn rent_face(px: u16) -> (Fonts, FaceId, f32) {
    let mut fonts = Fonts::default();
    match fonts.load(FACE.to_vec(), 0) {
        Ok(face) => {
            let warm: String = (0x20u8..0x7f).map(char::from).collect();
            fonts.layout(face, px, &warm);
            let lift = ink_lift(&mut fonts, face, px);
            (fonts, face, lift)
        }
        Err(error) => {
            tracing::warn!(%error, "editor: vendored face unreadable; drawing the bitmap");
            (fonts, FaceId::FALLBACK, 0.0)
        }
    }
}

fn walk_panes(node: &Node, on: &mut impl FnMut(PaneId)) {
    match node {
        Node::Tabs { panes, .. } => panes.iter().copied().for_each(on),
        Node::Split { first, second, .. } => {
            walk_panes(first, on);
            walk_panes(second, on);
        }
    }
}

/// Truncate to `chars` glyphs on a character boundary — every panel here is
/// narrower than the names it shows.
pub(crate) fn fit_text(text: &str, chars: usize) -> &str {
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
    fit_text(declared.rsplit('.').next().unwrap_or(declared), chars)
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// A world with something in every pane: two archetypes so the tree has
    /// more than one, `Renderable`s so the markers and the gizmo have geometry
    /// to draw, and an `Eye` on some of them.
    fn populated() -> (gg_ecs::World, gg_ecs::Entity) {
        use gg_ecs::boundary::{Eye, Renderable};
        use gg_math::sim;
        let mut world = gg_ecs::World::new();
        let mut first = None;
        for i in 0..6u32 {
            let e = world.spawn();
            let at = sim::DVec3::new(f64::from(i), 0.0, -4.0);
            world
                .insert(e, Renderable::boxed(at, sim::Vec3::splat(0.5), 0x00ff_8040))
                .unwrap();
            if i % 2 == 0 {
                world.insert(e, Eye::at(at, 0.0, 0.0)).unwrap();
            }
            first.get_or_insert(e);
        }
        (world, first.unwrap())
    }

    /// The gate [`gg_ui::Router::duplicate`] was built for and never had (§6
    /// M81) — its own doc says "exposed rather than only logged so a UI test can
    /// assert there are none", and no test did.
    ///
    /// Two widgets sharing an id fight over hover, capture and focus, and the
    /// symptom is a button that *sometimes* belongs to a different button. The
    /// editor is where it would happen: ~40 hand-written const ids across seven
    /// panels, plus every list row spelled `ID.indexed(n)`, where forgetting the
    /// `indexed` is one character and looks right. Nothing below the shell would
    /// catch it — the check is a `tracing::warn!` in a crate whose warnings no
    /// gate reads — and above the shell it is `xtask reload --editor`, five
    /// manual minutes over one layout.
    ///
    /// Ticked twice per pane because a duplicate is resolved in
    /// `Router::begin` against what the *previous* frame declared: one tick
    /// declares, the next reports. Every pane in [`Pane::ALL`], each tool, and
    /// with a selection, since panes only declare what they are showing.
    #[test]
    fn no_two_widgets_in_any_of_the_editors_panes_declare_one_id() {
        let (mut world, entity) = populated();
        let mut editor = Editor::new(None);
        editor.selected = Some(entity);
        let mut at = 0u64;
        let mut tick_twice = |editor: &mut Editor, world: &mut gg_ecs::World, what: &str| {
            for _ in 0..2 {
                at += 1;
                editor.tick(world, &Tick::default(), &frame_at((1600, 900), at));
            }
            assert_eq!(
                editor.router.duplicate().map(WidgetId::get),
                None,
                "{what}: two widgets declared one id"
            );
        };
        for pane in Pane::ALL {
            editor.dock.activate(pane.id());
            tick_twice(&mut editor, &mut world, pane.title());
        }
        // The viewport's overlay is the one that changes shape with state
        // rather than with which tab is up: the gizmo declares an arm per axis
        // per tool, and the markers a widget per placement.
        editor.dock.activate(Pane::Viewport.id());
        for tool in Tool::ALL {
            editor.tool = tool;
            tick_twice(&mut editor, &mut world, tool.label());
        }
        // And with nothing selected, which is a different set of declarations
        // and not a subset of the one above.
        editor.selected = None;
        tick_twice(&mut editor, &mut world, "no selection");

        // The vacuity guard, and it has to be here rather than in `gg-ui`: what
        // the loop above proves is only worth the assertion if the assertion can
        // fire through *this* editor's own router on *this* tick loop. Declared
        // after a tick, so it joins that frame's set and the next `begin`
        // resolves it.
        let clash = WidgetId::new("test.clash");
        let rect = gg_ui::Rect {
            x: 0.0,
            y: 0.0,
            w: 1.0,
            h: 1.0,
        };
        editor.router.hit(clash, rect);
        editor.router.hit(clash, rect);
        editor.tick(&mut world, &Tick::default(), &frame_at((1600, 900), at + 1));
        assert_eq!(editor.router.duplicate(), Some(clash), "the check is inert");
    }

    /// The mouse rule in both directions, because the bug was one of them: §6
    /// M15.4's pick and the game's mouse-look share one click in one rectangle,
    /// and the pointer used to win it whatever the transport said — so selecting
    /// an entity in a stopped scene cost the cursor until Escape gave it back.
    #[test]
    fn the_mouse_is_the_games_only_while_the_game_runs() {
        assert!(Play::Running.takes_pointer(false));
        for play in [Play::Stopped, Play::Paused] {
            assert!(!play.takes_pointer(false), "{play:?}");
        }
        // And a press on a panel is never the game's, whatever the transport
        // says: a click on `pause` must not also fire what the game bound to
        // that button ([`Editor::over_panels`]).
        for play in [Play::Stopped, Play::Running, Play::Paused] {
            assert!(!play.takes_pointer(true), "{play:?}");
        }
    }

    /// The three numbers the tables rest on are the *face's* and not chosen:
    /// [`EM`] is its advance at [`ROW`] pixels per em, letters sit inside the
    /// row they are given, and the tallest punctuation stays inside the unit of
    /// bleed `clipped_text` allows it. A face swapped for one with other
    /// proportions fails here rather than by overflowing every column in the
    /// editor.
    #[test]
    fn the_face_fills_its_row() {
        let printable: String = (0x20u8..0x7f).map(char::from).collect();
        for scale in [1.0, 2.0, 3.0, 4.0, 6.0] {
            let px = text_px(scale);
            let (mut fonts, face, lift) = rent_face(px);
            assert_ne!(face, FaceId::FALLBACK, "the vendored face must parse");
            let advance = fonts.layout(face, px, "MMMMMMMMMM") / 10.0;
            assert!(
                (advance - EM * scale).abs() <= 0.5,
                "px {px}: {advance} per glyph is not EM ({})",
                EM * scale
            );
            // Relative to the row a table gave it, which is where
            // `Editor::text` puts the run less the lift it was rented with.
            let ink = |fonts: &Fonts| {
                let top = fonts
                    .glyphs()
                    .iter()
                    .map(|g| g.rect.y)
                    .fold(f32::MAX, f32::min);
                let bottom = fonts
                    .glyphs()
                    .iter()
                    .map(|g| g.rect.y + g.rect.h)
                    .fold(f32::MIN, f32::max);
                (top - lift, bottom - lift)
            };
            // Capitals are the band the row is centred on — the assertion the
            // whole of `ink_lift` exists for. Judged on a word with no
            // descender, because that is the case centring the *line box* gets
            // wrong: half the descent is air, and a label set on it sits that
            // much high and climbs out of the cell above it.
            fonts.layout(face, px, "HEX");
            let (top, bottom) = ink(&fonts);
            assert!(
                (top - (ROW * scale - bottom)).abs() <= 1.5,
                "px {px}: capitals {top}..{bottom} sit off-centre on a {} row",
                ROW * scale
            );
            // And a descender is the only thing that leaves the row, downward.
            fonts.layout(face, px, "Alg0");
            let (top, bottom) = ink(&fonts);
            assert!(
                top >= -0.5 && bottom > ROW * scale,
                "px {px}: letters {top}..{bottom} on a {} row",
                ROW * scale
            );
            // The title bar's cells, which are the bar's full height and where
            // `label_mid` puts a row in one: every printable glyph stays inside
            // the one-unit border a plate draws. This is what a 9-unit cell
            // could not do at any lift — an 8-unit face needs more than seven
            // units of interior, so `f` and `l` climbed out of the top of the
            // menu titles and `y` out of the bottom of `play`.
            fonts.layout(face, px, &printable);
            let (top, bottom) = ink(&fonts);
            let row = ((BAR_H - ROW) * 0.5).floor() * scale;
            assert!(
                row + top >= scale && row + bottom <= (BAR_H - 1.0) * scale,
                "px {px}: ink {}..{} leaves a {} cell's border",
                row + top,
                row + bottom,
                BAR_H * scale
            );
            assert!(
                top >= -BLEED * scale && bottom <= (ROW + BLEED) * scale,
                "px {px}: ink {top}..{bottom} leaves the bleed `clipped_text` allows"
            );
            // And two rows of it never touch, which is what the second unit of
            // `PITCH` buys.
            assert!(bottom - top <= PITCH * scale, "px {px}: {top}..{bottom}");
        }
    }

    /// The panels are set in the rented face and not the bitmap band. Judged on
    /// the atlas rectangles the frame actually samples, because the failure
    /// this guards is silent: a face that would not load draws the fallback,
    /// which is legible, smaller, and not what any of the widths were sized
    /// for.
    #[test]
    fn the_panels_are_set_in_the_rented_face() {
        let mut world = gg_ecs::World::new();
        let mut editor = Editor::new(None);
        editor.tick(&mut world, &Tick::default(), &frame((1280, 720)));
        let band = font::BAND.1 as f32 / font::EXTENT.1 as f32;
        let glyphs = editor.vertices().iter().filter(|v| v.uv[1] >= band).count();
        assert!(glyphs > 100, "only {glyphs} vertices below the bitmap band");
    }

    /// A frame's worth of geometry for `extent`, with the fields a panel reads.
    fn frame(extent: (u32, u32)) -> Frame<'static> {
        frame_at(extent, 0)
    }

    fn frame_at(extent: (u32, u32), tick: u64) -> Frame<'static> {
        Frame {
            extent,
            dpi: 1.0,
            tick,
            hz: 60,
            play: Play::Paused,
            input: None,
            typed: "",
            passes: &[],
            memory: gg_rhi::MemoryUse::default(),
            save_path: "target/editor/test.ggsv",
            title: "gg — test",
            project: Some("test"),
            projects: &[],
            maximized: false,
            reload: None,
            draw_cursor: false,
        }
    }

    /// A pointer, a world and an editor, driven a tick at a time — what a test
    /// about the title bar needs, since every gesture up there is an *edge* and
    /// an edge takes two ticks to state.
    struct Driver {
        world: gg_ecs::World,
        editor: Editor,
        extent: (u32, u32),
        at: (i32, i32),
        tick: u64,
        /// What the title bar asked for since the last [`Driver::click`], taken
        /// off the editor the way the shell takes it.
        window: Option<WindowCommand>,
    }

    impl Driver {
        fn new(extent: (u32, u32)) -> Driver {
            Driver {
                world: gg_ecs::World::new(),
                editor: Editor::new(None),
                extent,
                at: (0, 0),
                tick: 0,
                window: None,
            }
        }

        /// One tick: glide to `to` if given, with the button in `down`.
        fn step(&mut self, to: Option<(f32, f32)>, down: bool) -> Commands {
            let scale = gg_ui::router::AXIS_SCALE as f32;
            let motion = match to {
                Some((x, y)) => {
                    let target = ((x * scale) as i32, (y * scale) as i32);
                    let motion = (target.0 - self.at.0, target.1 - self.at.1);
                    self.at = target;
                    motion
                }
                None => (0, 0),
            };
            let tick = Tick {
                motion,
                primary: down,
                ..Tick::default()
            };
            self.tick += 1;
            let commands = self.editor.tick(
                &mut self.world,
                &tick,
                &frame_at(self.extent, self.tick - 1),
            );
            self.window = self.editor.take_window_command().or(self.window);
            commands
        }

        /// Move somewhere, let hover settle, press and release. Every tick's
        /// commands, because a press and a release ask for different things.
        fn click(&mut self, at: (f32, f32)) -> Vec<Commands> {
            self.window = None;
            self.step(Some(at), false);
            self.step(None, false);
            vec![self.step(None, true), self.step(None, false)]
        }

        fn settle(&mut self, ticks: u32) {
            for _ in 0..ticks {
                self.step(None, false);
            }
        }
    }

    /// The title bar holds the whole of what an OS one did, and nothing in it
    /// sits on top of anything else — including on the platform this test is
    /// not running on (§6 M15.1 item 5's macOS residual).
    #[test]
    fn the_title_bar_lays_out_on_both_platforms() {
        for mac in [false, true] {
            for extent in [(1280, 720), (1920, 1080), (800, 600)] {
                let mut editor = Editor::new(None);
                editor.place(extent, 1.0);
                let bar = editor.bar_rect();
                let buttons = panels::window_buttons(bar, mac);
                // Laid out for `mac` rather than read off the editor, which can
                // only hold the arrangement it was compiled for.
                let mut strip = gg_ui::menu::MenuBar::default();
                strip.resolve(bar, panels::strip_left(bar, mac), MENUS, None, &text_width);
                let mut cells: Vec<Rect> = strip.titles().to_vec();
                cells.extend(
                    (0..panels::TOOLBAR.len()).map(|i| panels::transport_at(bar, strip.right(), i)),
                );
                cells.extend(buttons.iter().map(|(_, rect)| *rect));
                for (i, cell) in cells.iter().enumerate() {
                    assert!(
                        cell.y >= bar.y && cell.bottom() <= bar.bottom(),
                        "{mac} {extent:?}: {cell:?} leaves {bar:?}"
                    );
                    assert!(cell.x >= bar.x && cell.right() <= bar.right() + 0.01);
                    for other in &cells[i + 1..] {
                        assert!(
                            cell.intersect(other).is_empty(),
                            "{mac} {extent:?}: {cell:?} overlaps {other:?}"
                        );
                    }
                }
                // The platform's own order, on the platform's own side.
                let side = |rect: Rect| rect.x < bar.w * 0.5;
                assert!(buttons.iter().all(|(_, r)| side(*r) == mac));
                let order: Vec<WindowCommand> = buttons.iter().map(|(c, _)| *c).collect();
                assert_eq!(
                    order,
                    match mac {
                        true => vec![
                            WindowCommand::Close,
                            WindowCommand::Minimize,
                            WindowCommand::ToggleMaximize,
                        ],
                        false => vec![
                            WindowCommand::Minimize,
                            WindowCommand::ToggleMaximize,
                            WindowCommand::Close,
                        ],
                    }
                );
            }
        }
    }

    /// The transport sits on the bar's middle at every window size, and gives
    /// way to the menus rather than sliding under them.
    #[test]
    fn the_transport_is_centred_until_the_menus_reach_it() {
        for extent in [(1280, 720), (3840, 2160), (2560, 1080)] {
            let mut editor = Editor::new(None);
            editor.place(extent, 1.0);
            let bar = editor.bar_rect();
            // The whole set, not a numbered pair: what is centred is the
            // transport, so a button added to it must not need this line edited.
            let (play, last) = (
                editor.transport(0),
                editor.transport(panels::TOOLBAR.len() - 1),
            );
            let middle = (play.x + last.right()) * 0.5;
            assert!(
                (middle - bar.w * 0.5).abs() <= 1.0,
                "{extent:?}: the transport's middle is {middle} on a {} bar",
                bar.w
            );
            assert!(play.x > editor.menus.right(), "{extent:?}: under the menus");
        }
        // A bar too narrow to centre in: the set is pushed clear of the strip
        // instead of overlapping it, which is what `strip_right` is a floor for.
        let narrow = Rect::new(0.0, 0.0, 90.0, BAR_H);
        assert!(panels::transport_at(narrow, 60.0, 0).x >= 60.0);
    }

    /// Press the bar to move the window; press it twice to maximize. Both are
    /// what an OS title bar was doing before the editor took it over.
    #[test]
    fn the_bar_drags_the_window_and_a_double_press_maximizes_it() {
        let mut driver = Driver::new((1280, 720));
        // Over the status text, which is drawn and not hit-tested — and below
        // the top resize grip, which is.
        let bar = driver.editor.bar_rect();
        driver.step(None, false);
        // Past the *last* transport cell, not a numbered one: the set is
        // centred, so adding a button moves everything and a literal index
        // here would silently start pressing whatever grew into that gap.
        let last = panels::TOOLBAR.len() - 1;
        let empty = (
            driver.editor.transport(last).right() + 6.0,
            bar.bottom() - 1.5,
        );
        driver.click(empty);
        assert_eq!(driver.window, Some(WindowCommand::Drag));
        // Straight away again: the same press is a double-click.
        driver.click(empty);
        assert_eq!(driver.window, Some(WindowCommand::ToggleMaximize));
        // And once the window has gone quiet, it is a drag again.
        driver.settle(panels::DOUBLE as u32 + 2);
        driver.click(empty);
        assert_eq!(driver.window, Some(WindowCommand::Drag));
    }

    /// The middle button draws the state the window is in, not the one it
    /// asks for: maximize while the window is normal, restore while it is not.
    /// Judged on the geometry, because the two are drawn and not set — a glyph
    /// that ignored [`Frame::maximized`] would emit the same quads for both.
    #[test]
    fn the_maximize_button_draws_restore_once_the_window_is_maximized() {
        let quads = |maximized: bool| {
            let mut world = gg_ecs::World::new();
            let mut editor = Editor::new(None);
            let mut frame = frame((1280, 720));
            frame.maximized = maximized;
            editor.tick(&mut world, &Tick::default(), &frame);
            editor.vertices().len()
        };
        assert_ne!(quads(false), quads(true), "the same button in both states");
    }

    /// The three buttons ask for the three things they are drawn as.
    #[test]
    fn the_window_buttons_ask_for_what_they_show() {
        let mut driver = Driver::new((1280, 720));
        let bar = driver.editor.bar_rect();
        for (command, rect) in panels::window_buttons(bar, MAC) {
            let at = (rect.x + rect.w * 0.5, rect.y + rect.h * 0.5);
            driver.click(at);
            assert_eq!(driver.window, Some(command), "{command:?}");
            driver.settle(panels::DOUBLE as u32 + 2);
        }
    }

    /// The window has no OS frame, so the border that resizes it is this one:
    /// every edge and corner names its own direction, and a press there is the
    /// editor's rather than the game's.
    #[test]
    fn the_resize_border_names_the_edge_it_is_on() {
        let extent = (1280, 720);
        let mut driver = Driver::new(extent);
        driver.step(None, false);
        let canvas = fit(extent, 1.0).canvas;
        let (w, h) = (canvas.0 as f32, canvas.1 as f32);
        for (edge, at) in [
            (Edge { x: -1, y: -1 }, (1.0, 1.0)),
            (Edge { x: 1, y: -1 }, (w - 1.0, 1.0)),
            (Edge { x: -1, y: 1 }, (1.0, h - 1.0)),
            (Edge { x: 1, y: 1 }, (w - 1.0, h - 1.0)),
            (Edge { x: 0, y: -1 }, (w * 0.5, 1.0)),
            (Edge { x: 0, y: 1 }, (w * 0.5, h - 1.0)),
            (Edge { x: -1, y: 0 }, (1.0, h * 0.5)),
            (Edge { x: 1, y: 0 }, (w - 1.0, h * 0.5)),
        ] {
            assert_eq!(edge_at(canvas, at.0, at.1), Some(edge), "{at:?}");
            driver.click(at);
            assert_eq!(driver.window, Some(WindowCommand::Resize(edge)), "{at:?}");
            // Never the game's, wherever the viewport happens to have been
            // dragged to: a press on the frame resizes the window.
            assert!(
                driver.editor.over_panels(),
                "{at:?} handed the game a press"
            );
        }
        assert_eq!(edge_at(canvas, w * 0.5, h * 0.5), None, "the middle");
    }

    /// A window verb is produced by a *tick* and consumed by a *frame*, and a
    /// frame owing more than one tick is the ordinary case under 60 fps. So the
    /// command outlives the ticks after the one that raised it, and only
    /// [`Editor::take_window_command`] clears it — a `Driver` that drains every
    /// tick is a host no real shell is (§6 M15.1 item 5).
    #[test]
    fn a_window_verb_survives_the_ticks_between_it_and_the_frame() {
        let extent = (1280, 720);
        let mut world = gg_ecs::World::new();
        let mut editor = Editor::new(None);
        let scale = gg_ui::router::AXIS_SCALE as f32;
        let canvas = fit(extent, 1.0).canvas;
        // Mid-height, so this is the right *edge* and not the corner above it.
        let at = (
            ((canvas.0 as f32 - 1.0) * scale) as i32,
            (canvas.1 as f32 * 0.5 * scale) as i32,
        );
        let mut step = |editor: &mut Editor, motion, primary| {
            editor.tick(
                &mut world,
                &Tick {
                    motion,
                    primary,
                    ..Tick::default()
                },
                &frame(extent),
            );
        };
        // Onto the right border, settle so the hit test has seen it, then press
        // and hold — the press edge is the first of the ticks this frame owes.
        step(&mut editor, at, false);
        step(&mut editor, (0, 0), false);
        step(&mut editor, (0, 0), true);
        for _ in 0..gg_core::MAX_TICKS_PER_FRAME - 1 {
            step(&mut editor, (0, 0), true);
        }
        assert_eq!(
            editor.take_window_command(),
            Some(WindowCommand::Resize(Edge { x: 1, y: 0 })),
            "the ticks after the press edge ate the command"
        );
        assert_eq!(editor.take_window_command(), None, "and it is taken once");
    }

    /// `file → save` is two clicks and asks the host to write; `file → quit`
    /// closes the window. The menu is the only place either lives.
    #[test]
    fn the_file_menu_saves_and_quits() {
        let mut driver = Driver::new((1280, 720));
        driver.step(None, false);
        let file = session::aim::menu(&driver.editor, 0).expect("a file menu");
        let (save, quit) = (
            session::aim::menu_item(&driver.editor, 0, 0).expect("save"),
            session::aim::menu_item(&driver.editor, 0, 1).expect("quit"),
        );
        driver.click(file);
        assert_eq!(driver.editor.menus.open(), Some(0), "the menu dropped down");
        let commands = driver.click(save);
        assert!(commands.iter().any(|c| c.save), "save asked for nothing");
        assert_eq!(driver.editor.menus.open(), None, "and put itself away");
        assert_eq!(driver.editor.tally().1, 1);

        driver.click(file);
        driver.click(quit);
        assert_eq!(driver.window, Some(WindowCommand::Close));
    }

    /// A press anywhere else puts the menu away, and while it is down it is the
    /// editor's — including over the game, which it hangs across.
    #[test]
    fn an_open_menu_covers_what_is_under_it() {
        let mut driver = Driver::new((1280, 720));
        driver.step(None, false);
        let file = session::aim::menu(&driver.editor, 0).expect("a file menu");
        let save = session::aim::menu_item(&driver.editor, 0, 0).expect("save");
        driver.click(file);
        driver.step(Some(save), false);
        driver.step(None, false);
        assert!(
            driver.editor.over_panels(),
            "an open menu is not a hole to the game"
        );
        // Somewhere neither the strip nor the panel is.
        let elsewhere = (400.0, 300.0);
        driver.step(Some(elsewhere), false);
        driver.step(None, true);
        assert_eq!(
            driver.editor.menus.open(),
            None,
            "a press outside closed it"
        );
    }

    /// Every index `MENUS` declares does something, and nothing else does. The
    /// table and the handler are two lists that have to agree, and this is what
    /// stops an item from being drawn with nothing behind it.
    #[test]
    fn every_menu_item_does_something() {
        for (m, menu) in MENUS.iter().enumerate() {
            assert!(!menu.title.is_empty() && !menu.items.is_empty());
            for i in 0..menu.items.len() {
                assert!(menu_action(m, i).is_some(), "{} {i}", menu.title);
            }
            assert!(menu_action(m, menu.items.len()).is_none());
        }
        assert!(menu_action(MENUS.len(), 0).is_none());

        // And the `view` menu's pane rows say what they toggle (§6 M61). A pane
        // added to `ALL` without a row here fails on the length; one added out
        // of order fails by name, which is the failure worth having — the rows
        // are matched to panes by *position*, so a menu reading `perf` that
        // toggles `assets` is otherwise a silent swap.
        assert_eq!(VIEW_PANES.len(), Pane::ALL.len() + 1);
        for (row, pane) in VIEW_PANES.iter().zip(Pane::ALL) {
            assert_eq!(*row, pane.title());
            assert_eq!(
                menu_action(2, pane.id().0 as usize),
                Some(MenuAction::Toggle(pane))
            );
        }
        assert_eq!(
            menu_action(2, Pane::ALL.len()),
            Some(MenuAction::ResetLayout)
        );
    }

    /// A pane closed from the `view` menu leaves the dock and comes back, and
    /// the last one standing refuses to go — an editor with no panes has no
    /// menu bar to bring one back from, the bar being drawn outside the dock.
    #[test]
    fn the_view_menu_closes_and_reopens_every_pane() {
        let mut editor = Editor::new(None);
        editor.place((1280, 720), 1.0);
        for pane in Pane::ALL {
            assert!(
                editor.dock.holds(pane.id()),
                "{} did not open",
                pane.title()
            );
            editor.toggle_pane(pane);
            assert!(
                !editor.dock.holds(pane.id()),
                "{} did not close",
                pane.title()
            );
            editor.toggle_pane(pane);
            assert!(
                editor.dock.holds(pane.id()),
                "{} did not come back",
                pane.title()
            );
        }
        // Close everything. The last one refuses, so the tree is never empty and
        // the operator is never locked out.
        for pane in Pane::ALL {
            editor.toggle_pane(pane);
        }
        assert_eq!(editor.dock.panes(), 1);
        let left = Pane::ALL
            .into_iter()
            .find(|p| editor.dock.holds(p.id()))
            .expect("one pane survives");
        // And from there every other pane can be asked back, which is what
        // `home`'s fallback to any live group is for.
        for pane in Pane::ALL.into_iter().filter(|p| *p != left) {
            editor.toggle_pane(pane);
            assert!(editor.dock.holds(pane.id()), "{} is stranded", pane.title());
        }
    }

    /// A persisted layout missing a pane is now taken rather than dropped (§6
    /// M61) — but one naming a pane twice, one naming a pane this build has not
    /// got, and an empty one are still refused.
    #[test]
    fn a_layout_may_be_short_a_pane_but_not_wrong_about_one() {
        let mut editor = Editor::new(None);
        let two = Node::split(
            Axis::Horizontal,
            0.5,
            Node::pane(Pane::Tree.id()),
            Node::pane(Pane::Viewport.id()),
        );
        assert!(editor.set_layout(two.clone()));
        assert_eq!(editor.dock.panes(), 2);

        let twice = Node::split(
            Axis::Horizontal,
            0.5,
            Node::pane(Pane::Tree.id()),
            Node::pane(Pane::Tree.id()),
        );
        assert!(!editor.set_layout(twice));
        let unknown = Node::pane(PaneId(Pane::ALL.len() as u16));
        assert!(!editor.set_layout(unknown));
        assert!(!editor.set_layout(Node::Tabs {
            panes: Vec::new(),
            active: 0
        }));
        // Each refusal left the sound one in place rather than half-applying.
        assert_eq!(editor.dock.panes(), 2);
    }

    /// The bug §6 M15.1 exists for: at no window size does the editor leave a
    /// pixel to nobody. Every logical unit of the surface is the toolbar, a
    /// group, or a seam — there is no letterbox to show the game through.
    #[test]
    fn the_panes_cover_every_window_size() {
        for extent in [
            (1280, 720),
            (1920, 1080),
            (3840, 2064),
            (2560, 1080),
            (800, 600),
        ] {
            let mut editor = Editor::new(None);
            editor.place(extent, 1.0);
            let canvas = editor.fit.canvas;
            let total = canvas.0 as f32 * canvas.1 as f32;
            let mut covered = BAR_H * canvas.0 as f32;
            for group in editor.dock.groups() {
                covered += group.region.w * group.region.h;
            }
            for seam in editor.dock.seams() {
                covered += seam.rect.w * seam.rect.h;
            }
            let mut overlap = 0.0;
            let seams = editor.dock.seams();
            for (i, a) in seams.iter().enumerate() {
                for b in &seams[i + 1..] {
                    let hit = a.rect.intersect(&b.rect);
                    if !hit.is_empty() {
                        overlap += hit.w * hit.h;
                    }
                }
            }
            assert!(
                (covered - overlap - total).abs() < 0.5,
                "{extent:?}: {covered} - {overlap} covers {total}"
            );
        }
    }

    /// The scale is whole at every size, so the bitmap font is never resampled.
    #[test]
    fn the_ui_scale_is_always_an_integer() {
        for extent in [
            (640, 360),
            (1280, 720),
            (1920, 1080),
            (3840, 2160),
            (77, 41),
        ] {
            for dpi in [0.0, 1.0, 1.25, 1.5, 2.0, 3.0] {
                let scale = ui_scale(extent, dpi);
                assert_eq!(scale, scale.floor(), "{extent:?} at {dpi}");
                assert!(scale >= 1.0, "{extent:?} at {dpi}");
            }
        }
    }

    /// A row is a row on every monitor, which is what §6 M15.1's last finding
    /// was about: the scale follows the *DPI*, and the window only caps it, so
    /// more pixels buy more rows rather than bigger ones.
    #[test]
    fn the_scale_follows_the_monitor_and_not_the_resolution() {
        // Every extent a gate is aimed at, at the 1.0 a host with no window
        // reports — all unchanged, which is why none of them moved.
        assert_eq!(ui_scale(gg_ecs::boundary::CANVAS, 1.0), 1.0);
        assert_eq!(ui_scale((1280, 720), 1.0), 2.0);
        assert_eq!(ui_scale((1600, 900), 1.0), 2.0);
        assert_eq!(ui_scale((1920, 1080), 1.0), 2.0);
        // And the finding itself: 4K stops being 1080p with bigger pixels. At
        // 100% it is twice the rows; at the 150% Windows gives a 4K panel it is
        // a third more, which is what "150%" means everywhere else on the
        // desktop and what the editor was ignoring.
        assert_eq!(ui_scale((3840, 2160), 1.0), 2.0, "M15.1 answered 4 here");
        assert_eq!(ui_scale((3840, 2160), 1.5), 3.0);
        assert_eq!(ui_scale((3840, 2160), 2.0), 4.0);
        // Halves round down (§6 M19): 125% asks for 2.5, and the answer is the
        // 100% look rather than a UI 20% larger than the desktop asked for.
        // 175% asks for 3.5 and gets 3, same rule; 130% is past the half and
        // rounds up as ever.
        assert_eq!(ui_scale((1920, 1080), 1.25), 2.0, "2.5 rounded to 3 before");
        assert_eq!(ui_scale((3840, 2160), 1.75), 3.0);
        assert_eq!(ui_scale((3840, 2160), 1.3), 3.0);
        assert_eq!(fit((3840, 2160), 1.0).canvas, (1920, 1080));
        // The window is a cap and never a floor: a small window on a dense
        // monitor still leaves `MIN` units to lay panes out in.
        assert_eq!(ui_scale((1280, 720), 3.0), 2.0);
        assert_eq!(ui_scale(MIN, 6.0), 1.0);
        for dpi in [1.0, 1.5, 2.0, 3.0] {
            for extent in [(1280, 720), (1920, 1080), (2560, 1080), (3840, 2160)] {
                let canvas = fit(extent, dpi).canvas;
                assert!(
                    canvas.0 >= MIN.0 && canvas.1 >= MIN.1,
                    "{extent:?} at {dpi}: {canvas:?} is under {MIN:?}"
                );
            }
        }
    }

    /// The rectangle handed to the renderer is inside the pane that frames it,
    /// in physical pixels, at every size.
    #[test]
    fn the_viewport_is_inside_the_pane_that_outlines_it() {
        for extent in [(1280, 720), (3840, 2064), (2560, 1080)] {
            let mut editor = Editor::new(None);
            editor.place(extent, 1.0);
            let view = editor.viewport_rect();
            let body = editor.dock.body_of(Pane::Viewport.id()).expect("placed");
            let scale = editor.fit.scale;
            assert!(view.width > 0 && view.height > 0, "{extent:?}: {view:?}");
            assert!(
                view.x as f32 >= body.x * scale && view.y as f32 >= body.y * scale,
                "{extent:?}: {view:?} starts before {body:?}"
            );
            assert!(
                (view.x + view.width) as f32 <= body.right() * scale + 0.5
                    && (view.y + view.height) as f32 <= body.bottom() * scale + 0.5,
                "{extent:?}: {view:?} runs past {body:?}"
            );
            assert!(view.x + view.width <= extent.0 && view.y + view.height <= extent.1);
        }
    }

    /// The regression that made the game a rectangle of chrome: the editor
    /// draws *around* the viewport, never over it. Judged at the centre of the
    /// pane rather than over the whole of it, because the play-state tag is
    /// deliberately inside the top-left corner.
    ///
    /// The world here is empty, which is now load-bearing: §6 M15.4 item 2 draws
    /// the selection's outline into this rectangle on purpose, and it is the one
    /// thing that ever does. `the_selection_is_outlined_in_the_scene…` is the
    /// other side of the same claim.
    #[test]
    fn nothing_is_drawn_over_the_middle_of_the_game() {
        for extent in [(1280, 720), (2560, 1080), (3840, 2064)] {
            let mut world = gg_ecs::World::new();
            let mut editor = Editor::new(None);
            editor.tick(&mut world, &Tick::default(), &frame(extent));
            // Physical pixels: that is the space the vertices are in, and
            // `viewport_rect` is the same rectangle the renderer was handed.
            let view = editor.viewport_rect();
            let cx = view.x as f32 + view.width as f32 * 0.5;
            let cy = view.y as f32 + view.height as f32 * 0.5;
            // Six vertices per quad (`gg_render::ui`), so a bounding box per
            // chunk is the quad — glyphs included, which is what we want.
            for quad in editor.vertices().chunks(6) {
                let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
                for vertex in quad {
                    for axis in 0..2 {
                        lo[axis] = lo[axis].min(vertex.pos[axis]);
                        hi[axis] = hi[axis].max(vertex.pos[axis]);
                    }
                }
                assert!(
                    !(lo[0] <= cx && cx < hi[0] && lo[1] <= cy && cy < hi[1]),
                    "{extent:?}: {lo:?}..{hi:?} covers the game at ({cx}, {cy})"
                );
            }
        }
    }

    /// Panes and their ids agree in both directions, which is what a persisted
    /// layout is stored as.
    #[test]
    fn every_pane_round_trips_through_its_id() {
        for pane in Pane::ALL {
            assert_eq!(Pane::from_id(pane.id()), Some(pane));
        }
        assert_eq!(Pane::from_id(PaneId(Pane::ALL.len() as u16)), None);
    }
}
