//! The wiring: a world, a game dylib, a window, and the stages that drive them
//! (§4.1).
//!
//! Every line chooses *which* engine piece runs and in what order; none
//! implements one. The three that would otherwise be logic — registering what a
//! dylib declares, calling its systems, drawing what it asks for — are
//! `gg-ecs`'s [`World::adopt`], [`World::run_systems`] and `gg-render`'s
//! [`Renderer`], because §3 caps this crate at zero engine logic.
//!
//! **Dev and dist are this file with one field.** The loader is unconditional
//! (§4.2.2): dist loads the table once at startup and checks it exactly as dev
//! does. `hot-reload` adds the watcher and the swap at [`Stages::reload_check`],
//! which in dist stays the trait's empty default — absent, not disabled.
//!
//! **The GPU half is optional, and that is §1.5 rather than a convenience.** No
//! automated tier may create a window at all, so a headless run holds no
//! [`Renderer`] and the extract and render stages return immediately. The sim, the reload
//! path and the replay stream are identical either way, which is what lets CI
//! exercise the shell it ships.

use std::path::Path;

use gg_core::reload::rejuvenate::Rejuvenator;
use gg_core::{GameLib, Stages};
use gg_ecs::boundary::{Eye, Light, Look, Model, Prefs, Renderable, TickCtx, Widget, host_api};
use gg_ecs::{ComponentOutcome, Save, Snapshot, World};
use gg_extract::{Extracted, Latch};
use gg_input::{
    ActionMap, AxisId, Drive, Input, InputFrame, MAX_AXES, Recorder, Replay, ReplayMeta,
};
use gg_platform::Window;
use gg_render::{Renderer, View, ui::UiVertex};
use gg_scene::Hierarchy;
use gg_ui::boundary::binding;
use gg_ui::router::{Binding, Tick as UiTick};
use tracing::{error, info, warn};

/// Background clear (linear values; the sRGB target encodes).
const CLEAR: [f32; 4] = [0.02, 0.025, 0.04, 1.0];

/// The context a shell binds in. One name, so a game's bindings file and the
/// shell agree without a flag for it.
const CONTEXT: &str = "game";

pub struct App {
    world: World,
    lib: GameLib,
    /// The leaked-dylib budget and the handoff crossing it stages (§4.2.2).
    /// Never charged in dist — one library, loaded once, nothing retired — but
    /// held there too: what dist must not contain is the *watcher*, and gating a
    /// pair of counters would buy a second body for every method that reads it.
    rejuvenate: Rejuvenator,
    /// What the window last reported as its inner size, in physical pixels —
    /// the surface every UI on it is laid out for ([`App::surface`]). Starts at
    /// the headless canvas and is only ever moved by [`App::resize`], so a run
    /// with no window keeps the extent its recorded session was made at.
    extent: (u32, u32),
    hz: u32,
    /// A panicking system halts the sim and leaves the process running
    /// (§4.2.2); a reload is what clears this, which is the "agent broke it,
    /// agent fixes it, nobody restarts" loop.
    halted: bool,
    /// §6 M16's record of that loop. Written at the seam and at nothing else, so
    /// a session with no reloads publishes once and never again.
    #[cfg(feature = "agent")]
    journal: gg_agent::Journal,
    #[cfg(feature = "hot-reload")]
    watch: gg_core::reload::watch::Watch,
    /// The bindings text, kept because a reload can move the verb list and the
    /// map has to be parsed against the *new* one (§4.7). Dev only: dist parses
    /// once at load and has no second build to parse against.
    #[cfg(feature = "hot-reload")]
    bindings: String,
    /// The player's own bindings text, kept for `bindings`'s reason: a reload
    /// reparses the map and the overlay has to go back on top of it (§6 M45).
    #[cfg(feature = "hot-reload")]
    rebinds: String,
    input: Input,
    drive: Drive,
    /// The recorded-CVar diff (§6 M40). The shell is the only place a replay and
    /// the registry are both in scope, which is why the two lines that join them
    /// are here rather than in either.
    knobs: gg_core::cvar::Watch,
    /// The tick that has not run yet — the sim clock's own resume point, which
    /// is why it survives a rejuvenation (§4.2.2) rather than restarting at zero.
    next_tick: u64,
    previous: InputFrame,
    /// The §4.7 hierarchy's frame-to-frame memory. Host-owned rather than a
    /// side table: what it holds is derived from state the hash already covers.
    hierarchy: Hierarchy,
    extracted: Extracted,
    view: View,
    /// The game's declared UI (§4.9). Driven on the *tick* and not the frame:
    /// hit state lands in `Widget::state`, which the canonical hash covers, so
    /// "a replayed click lands on the same widget" is §5.6c's existing gate
    /// rather than a new kind of proof — and a headless run must therefore
    /// route clicks exactly as a windowed one does.
    ui: gg_ui::Ui,
    /// The game's declared audio (§6 M18 item 2). Driven on the tick beside the
    /// UI and for half the same reason — a cue fires on the tick a row cleared,
    /// and a frame that covered two ticks would drop one of them.
    ///
    /// The other half is the opposite of the UI's: this stage writes *nothing*.
    /// A silent run and a loud one are the same run, which is what keeps §5.6c
    /// comparing sims rather than speakers. Silent under `GG_HEADLESS=1` and on
    /// a machine with no device, both without failing (§1.5).
    audio: gg_audio::Audio,
    /// Which verbs feed it, resolved against *this build's* verb list and
    /// re-resolved at every swap for the reason [`bind`] gives. `None` is a
    /// game that declared none: its UI draws and cannot be clicked.
    ui_binding: Option<Binding>,
    /// Whether the map binds raw device motion — "this game does mouse-look",
    /// which is what decides the pointer is the game's to hold at all (§6 M21).
    /// Re-read at every swap beside `ui_binding`, and for the same reason.
    looks: bool,
    /// The frame's UI geometry, game first and instruments over the top. One
    /// buffer because [`Renderer::frame`] takes one slice; cleared and refilled,
    /// so it allocates once (§6 M13).
    ui_geometry: Vec<UiVertex>,
    /// The editor's play/stop, when `--play` asked for it (§6 M14). In every
    /// tier for the reason `--save` is: the capture and the restore either side
    /// of it are the shipping code path, and a play mode that existed only where
    /// the instruments do would make dist the untested one (§1.10).
    play: Option<PlayMode>,
    /// The editor, when `--editor` asked for it (§6 M15). One field rather than
    /// four: pause, single-step and the save target only exist because it does.
    #[cfg(feature = "editor")]
    editor: Option<Editing>,
    /// The OS cursor, and who has the pointer (§6 M15.1).
    cursor: Cursor,
    /// Where a windowed loop should warp the arrow, once (§6 M63). See
    /// [`App::take_warp`].
    warp: Option<(f32, f32)>,
    /// `gg_editor::Editor::font_revision` as last uploaded. Zero is "the
    /// fallback band, from `attach`" — no editor ever reports it.
    #[cfg(feature = "editor")]
    ui_atlas_rev: u64,
    /// The window's GPU state, absent in a headless run — and that absence is
    /// the extract and render stages' off switch.
    gpu: Option<Renderer>,
    /// The monitor's scale factor, `1.0` until a window says otherwise — which
    /// is what a headless run and a golden render both stay at (§6 M15.1).
    dpi: f32,
    /// The pack, opened against the renderer once there is one (§4.6). Held as
    /// a path rather than opened here because a headless run has no renderer to
    /// stream into and mapping a file for nobody would be work with no reader.
    pack: Option<std::path::PathBuf>,
    /// Where the warm pipeline cache goes: the player's own directory (§6 M42),
    /// or `None` for a dev run, which keeps it under `target/` like everything
    /// else. **Not** `player_file`'s — that gates on the run being *live*,
    /// because a preference is hashed world state (§6 M42). A pipeline cache is
    /// a driver artifact no tick can observe, so a replayed session deserves a
    /// warm one exactly as much as a live one does.
    cache: Option<std::path::PathBuf>,
    /// Tracy's GPU zones, fed the same readings the overlay shows, so the two
    /// views of one frame cannot disagree (§4.8).
    #[cfg(feature = "debug-tools")]
    zones: Option<gg_debug::GpuZones>,
    /// The overlay, and the UI vertices it built this frame (§4.8). Absent from
    /// dist entirely — `gg-debug` is not in that graph (§3).
    #[cfg(feature = "overlay")]
    overlay: gg_debug::Overlay,
    /// What the player's settings file asked for, until this session's first
    /// tick spends it (§6 M42). `None` in every run that may not read one — see
    /// `player_file`.
    settings: Option<Prefs>,
    /// The writers that keep the player's two files current while the session
    /// runs (§6 M48). Each is `None` on exactly the terms its exit write is
    /// skipped on, which is the rule rather than a coincidence: a checkpoint is
    /// the exit write happening sooner and more than once, so a run that may not
    /// write a file at the end may not write it in the middle either.
    ///
    /// Two writers and not one queue: a queue of one is what keeps the sim
    /// thread off the disk, and sharing it would let a settings offer evict the
    /// session the interval just encoded.
    checkpoint: Option<crate::player::Checkpoint>,
    /// `settings.cfg`'s (§6 M42), on the same cadence. A crash cost a player
    /// every preference they had touched that session — the same defect as the
    /// board, one file over and cheaper to write.
    prefs_checkpoint: Option<crate::player::Checkpoint>,
    /// What the OS last said about the window filling a monitor (§6 M46).
    /// Always false in a windowless run, where there is nothing to fill.
    window_is_fullscreen: bool,
    /// Whether a tick of this session has run. What [`settings`](Self::settings)
    /// is spent on, and a `bool` rather than a tick number because a resumed
    /// session's first tick is whatever the save carried (§6 M44).
    opened: bool,
    /// Whether the window is the one being typed into (§6 M49). True in a
    /// windowless run and true until a window says otherwise: a session nobody
    /// told about focus is a session that runs, which is what keeps every gate
    /// that predates M49 byte-unchanged.
    focused: bool,
    /// `--away`'s script, a window losing focus on frame numbers instead of on a
    /// player's attention. Drives the same decision the event does rather than a
    /// second one, so what a tier grades is the shipping path.
    away: Option<Away>,
    /// What [`Stages::suspended`] last answered, so the transition is logged once
    /// rather than sixty times a second.
    waiting: bool,
}

impl App {
    /// The turn the hand has made that no tick has spent yet (§6 M56), or `None`
    /// for a frame that must render the tick as it stands.
    ///
    /// Four ways to get `None`, and only the first is a decision about quality:
    /// the knob is off; the game declares no [`Look`], which is every game that
    /// predates M56 and every game whose camera is not a hand's; the sim is
    /// **halted**, where `sim_tick` returns before `Input::tick` and the
    /// accumulator therefore grows without bound; or the session is
    /// **suspended** (§6 M49), which runs no tick at all. The last two also
    /// empty the accumulator where they set that state, so this is belt and
    /// braces on a hazard that reads as the view swinging by however long the
    /// operator was away.
    ///
    /// Nothing here is `None` for a locked pace, and nothing needs to be — the
    /// two readings are zero there on two *different* structural grounds, and
    /// both are worth stating separately because they cover different tiers.
    /// `pending` is zero because a locked frame spends the accumulator down to
    /// nothing across its own ticks (`covered == count` exactly). `spent` is
    /// zero because a replay's frame carries no motion to attribute
    /// (`Input::tick_from` clears it) and a windowless run has no device to
    /// produce any. So every replay, golden and headless tier (§5.6) adds a
    /// latch of zeros, which is the identity — **by construction** rather than
    /// by a flag someone has to remember to clear.
    fn latch(&self) -> anyhow::Result<Option<Latch>> {
        if self.halted || self.waiting || !gg_render::cvars::LATE_LATCH.bool() {
            return Ok(None);
        }
        let Some(look) = Look::of(&self.world)? else {
            return Ok(None);
        };
        Ok(Some(self.latch_of(look)))
    }

    /// [`App::latch`] for a `Look` already in hand — the editor camera declares
    /// its own rather than putting one in the world (§6 M63).
    fn latch_of(&self, look: Look) -> Latch {
        let (yaw, pitch) = (self.reading(look.yaw_axis), self.reading(look.pitch_axis));
        Latch::of(look, (yaw.0, pitch.0), (yaw.1, pitch.1))
    }

    /// The editor camera's eye for a frame `alpha` of a tick past the last one,
    /// or `game` when that camera is not the one being rendered from (§6 M63).
    ///
    /// The same two corrections the game's eye gets and in the same order: blend
    /// the two ticks, then add the turn no tick has spent. `r.late_latch` gates
    /// the second half here as it does there — one knob, both cameras, because
    /// an operator turning it off to see what it was doing wants to see that in
    /// the viewport they are looking through.
    ///
    /// Every reading is zero for a windowless or replayed session by
    /// construction (`App::latch`'s note): `alpha` is zero under a locked pace,
    /// and a replay's frame carries no device motion to latch. So this is the
    /// identity in every tier a gate runs, and no golden or hash moves for it.
    #[cfg(feature = "editor")]
    fn editor_eye(&self, game: Eye, alpha: f32) -> Eye {
        let Some(editing) = self.editor.as_ref() else {
            return game;
        };
        let (previous, current) = editing.ui.eyes(game);
        let blended = gg_extract::blend_eye(previous, current, alpha);
        match gg_render::cvars::LATE_LATCH
            .bool()
            .then(|| editing.ui.look(&self.input))
            .flatten()
        {
            Some(look) => gg_extract::latched(blended, self.latch_of(look), alpha),
            None => blended,
        }
    }

    /// `Input::pending` and `Input::spent` for the axis number a game wrote into
    /// its [`Look`].
    ///
    /// Checked rather than trusted: the index is game data — from a system, a
    /// save or an older build's world — and `AxisId::new` *asserts*. An axis
    /// this build's map does not have is a view that does not turn, which is the
    /// same nothing a game declaring no `Look` gets, and never a panic in the
    /// host on a number a game got wrong.
    fn reading(&self, axis: u32) -> (f32, f32) {
        match usize::try_from(axis) {
            Ok(index) if index < MAX_AXES => {
                let id = AxisId::new(index);
                (self.input.pending(id), self.input.spent(id))
            }
            _ => (0.0, 0.0),
        }
    }

    /// Load `game`, register what it declares, and start a world over it. In dev
    /// load number one already goes through the staging copy: loading
    /// `target/debug/game.dll` in place would make the next `cargo build` fail
    /// rather than the reload (§4.2.2).
    pub fn new(
        args: &crate::Args,
        staging: &Path,
        hz: u32,
        bindings: String,
        rebinds: String,
        replay: Option<Box<Replay>>,
    ) -> anyhow::Result<Self> {
        let (game, _) = (args.game.as_path(), staging);
        // No game is the launcher (§6 M15.1 item 4), and the absence is the
        // loader's own variant rather than an `Option` here: everything below
        // reads a `GameLib` and every one of those reads has a true answer with
        // nothing loaded.
        let none = game.as_os_str().is_empty();
        #[cfg(feature = "hot-reload")]
        let (watch, lib) = if none {
            (gg_core::reload::watch::Watch::absent(), GameLib::absent())
        } else {
            let mut watch = gg_core::reload::watch::Watch::new(game, staging)?;
            // SAFETY: `game` is the artifact the operator named — the only
            // provenance any host can establish (§4.2.2) — and `host_api()` is
            // `&'static`. How long to wait and how hard is the watcher's, not
            // the shell's (§3).
            let lib = unsafe { watch.block_until_ready(host_api()) }?.lib;
            (watch, lib)
        };
        #[cfg(not(feature = "hot-reload"))]
        // SAFETY: `game` is the artifact the operator named, which is the only
        // provenance any host can establish (§4.2.2); `host_api()` is `&'static`.
        let lib = match none {
            true => GameLib::absent(),
            false => unsafe { GameLib::load(game, host_api())? },
        };

        let mut world = World::new();
        let declared = adopt(&mut world, &lib)?;
        let mut drive = match replay {
            Some(replay) => Drive::Replay(replay),
            None => Drive::Live(
                args.record
                    .is_some()
                    .then(|| Box::new(Recorder::new(meta(&lib, args.editor, hz)))),
            ),
        };
        // Segment zero names the build that produced tick zero (§4.7).
        drive.open_segment(0, lib.code_hash());
        // SAFETY: `lib` is verified and never unloaded.
        let (verbs, extra) = unsafe { verbs_for(&lib, args.editor) };
        let input = bind(&format!("{bindings}{extra}"), &rebinds, &drive, verbs)?;
        let ui_binding = binding(&verbs);
        // Before the struct literal takes `input`.
        let input_looks = input.looks();
        info!(
            path = %lib.path().display(),
            components = declared,
            systems = lib.systems().len,
            contexts = input.map().context_count(),
            ui = ui_binding.is_some(),
            replaying = drive.ticks(),
            "game loaded"
        );
        // Before the struct literal takes `lib`: the save target defaults to the
        // dylib's own name, so a session with no `--save` still has somewhere
        // for the button to write.
        #[cfg(feature = "editor")]
        let editing = args
            .editor
            .then(|| Editing::new(args, &lib, drive.surface()));
        // Same reason, one line later: the record names the game it is about.
        #[cfg(feature = "agent")]
        let lib_name = lib.name().to_string();
        Ok(Self {
            world,
            lib,
            rejuvenate: Rejuvenator::new(args.leak_budget),
            extent: gg_ecs::boundary::CANVAS,
            hz,
            halted: false,
            #[cfg(feature = "agent")]
            journal: journal(args, &lib_name),
            #[cfg(feature = "hot-reload")]
            watch,
            #[cfg(feature = "hot-reload")]
            bindings,
            #[cfg(feature = "hot-reload")]
            rebinds,
            input,
            drive,
            knobs: gg_core::cvar::Watch::new(),
            next_tick: 0,
            previous: InputFrame::default(),
            hierarchy: Hierarchy::new(),
            extracted: Extracted::default(),
            view: View::default(),
            ui: gg_ui::Ui::new()?,
            audio: {
                let mut audio = gg_audio::Audio::device_unless_headless()?;
                if let Some(pack) = &args.pack {
                    audio.install(clips(pack));
                }
                audio
            },
            cursor: Cursor::new(args.editor || ui_binding.is_some()),
            warp: None,
            looks: input_looks,
            ui_binding,
            #[cfg(feature = "editor")]
            ui_atlas_rev: 0,
            ui_geometry: Vec::new(),
            play: args.play.as_deref().map(PlayMode::parse).transpose()?,
            focused: true,
            away: args.away.as_deref().map(Away::parse).transpose()?,
            waiting: false,
            #[cfg(feature = "editor")]
            editor: editing,
            gpu: None,
            dpi: 1.0,
            pack: args.pack.clone(),
            cache: args.data.clone(),
            #[cfg(feature = "debug-tools")]
            zones: None,
            #[cfg(feature = "overlay")]
            overlay: gg_debug::Overlay::default(),
            settings: None,
            checkpoint: None,
            prefs_checkpoint: None,
            window_is_fullscreen: false,
            opened: false,
        })
    }

    /// What the player's settings file asked for, spent on the first tick this
    /// session runs (§6 M42, corrected at M44).
    pub fn want_settings(&mut self, prefs: Prefs) {
        self.settings = Some(prefs);
    }

    /// Keep the player's files current while this session runs (§6 M48). Each
    /// argument is the path its *exit* write already targets, or `None` where
    /// that write is skipped — the caller owns that decision and this must not
    /// second-guess it. Idempotent by replacement: a second call retires the
    /// first writers, which is what makes dropping them the only spelling of
    /// "stop".
    pub fn checkpoint_to(
        &mut self,
        session: Option<std::path::PathBuf>,
        prefs: Option<std::path::PathBuf>,
    ) {
        self.checkpoint = session.map(crate::player::Checkpoint::new);
        self.prefs_checkpoint = prefs.map(crate::player::Checkpoint::new);
    }

    /// Stop checkpointing and wait for the last one to land. **Before the exit
    /// write and never after**: bytes still in flight would land on top of the
    /// newer ones and roll the session back by up to an interval.
    pub fn checkpoint_stop(&mut self) {
        self.checkpoint = None;
        self.prefs_checkpoint = None;
    }

    /// The preferences as the last tick left them — what a session writes back
    /// out. Read off the UI stage's cache rather than the world, because that
    /// stage walks for one already and two answers would be one too many.
    pub fn prefs(&self) -> Prefs {
        self.ui.prefs()
    }

    /// The window gained or lost keyboard focus (§6 M49).
    ///
    /// A fact and not a decision: what it costs a session is
    /// [`Stages::suspended`]'s to say, and only once the preference has been
    /// read. Never called in a windowless run, where the field stays true.
    pub fn set_focused(&mut self, focused: bool) {
        self.focused = focused;
    }

    /// Whether the game has a modal screen up, from the same cache (§6 M45) —
    /// so what suppresses this tick's input is the `Prefs` the *last* one left,
    /// which is the tick whose widgets are on the glass.
    fn modal(&self) -> bool {
        self.prefs().modal()
    }

    /// Offer a key to the instruments before anything else sees it. `true` means
    /// the overlay took it — see `gg_debug::Overlay::key` for why only presses
    /// are ever taken.
    #[cfg(feature = "overlay")]
    pub fn debug_key(&mut self, key: gg_input::Key, pressed: bool, text: Option<char>) -> bool {
        // Text first: the keystroke that *opens* the console also produces a
        // character, and it is only filtered once the console is open.
        if let Some(c) = text {
            self.overlay.text(c);
        }
        self.overlay.key(key, pressed)
    }

    /// Dist has no instruments to offer it to (§3).
    #[cfg(not(feature = "overlay"))]
    pub fn debug_key(&mut self, _key: gg_input::Key, _pressed: bool, _text: Option<char>) -> bool {
        false
    }

    /// Bring up the GPU against a live window. Called once, from
    /// `Event::WindowReady` — the surface may not outlive the window it came
    /// from, which is what [`App::detach`] is for at the other end.
    pub fn attach(&mut self, window: &Window) -> anyhow::Result<()> {
        // Before the first `Resized`, which winit sends only after this: a
        // window that laid its first frame out at `boundary::CANVAS` would
        // relayout on the very next event for no reason.
        self.extent = window.inner_size();
        let mut renderer = Renderer::new(window, window.inner_size(), self.cache.as_deref())?;
        // The renderer never learns what a glyph is; it takes coverage texels
        // and a rectangle (§4.9). Unconditional since M13: the game's own UI
        // draws from this atlas in every tier, the overlay is a second caller.
        //
        // The editor's, when there is one, rather than uploading the fallback
        // and replacing it on the first frame: its atlas *contains* the
        // fallback band, so one upload serves all three callers, and an image
        // uploaded and thrown away inside a frame is churn nobody asked for.
        #[cfg(feature = "editor")]
        let coverage = match &self.editor {
            Some(editing) => {
                self.ui_atlas_rev = editing.ui.font_revision();
                editing.ui.coverage()
            }
            None => gg_ui::atlas::fallback(),
        };
        #[cfg(not(feature = "editor"))]
        let coverage = gg_ui::atlas::fallback();
        renderer.set_ui_atlas(&coverage)?;
        if let Some(pack) = &self.pack {
            renderer.open_pack(pack)?;
        }
        let device = renderer.device();
        info!(device = %device.chosen, api = ?device.api_version, "gpu online");
        // Opened here and not at construction: the context is anchored to the
        // device clock, and there is no device until now.
        #[cfg(feature = "debug-tools")]
        {
            self.zones = renderer
                .gpu_clock()
                .and_then(|clock| gg_debug::GpuZones::new("gg.graphics", clock));
        }
        self.gpu = Some(renderer);
        Ok(())
    }

    /// Tear the GPU down while the window is still alive. Idempotent: every
    /// exit path calls it and `Event::Exiting` is the backstop for the ones that
    /// did not get there first.
    pub fn detach(&mut self) {
        let Some(renderer) = self.gpu.take() else {
            return;
        };
        let report = renderer.shutdown();
        if report.clean() {
            info!("gpu shutdown clean");
        } else {
            error!(
                validation_messages = report.validation_messages,
                leaks = report.leaked_allocations.len(),
                "unclean shutdown (§4.3, §5.4)"
            );
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        // Recorded here as well as queued below, because the two are not the
        // same size until the next frame: `Rhi::resize` only raises a flag, so
        // the swapchain is still the old one while this tick lays the editor
        // out. Laying out against the swapchain would put the whole UI one
        // resize event behind the window for the length of a drag (§6 M15.1).
        self.extent = (width, height);
        if let Some(renderer) = &mut self.gpu {
            renderer.resize(width, height);
        }
    }

    /// What the monitor says a logical pixel is worth (§6 M15.1) — stated by
    /// the windowed loop, since a window is the only thing that can ask. A
    /// headless run leaves the 1.0 it starts at, which is the truth there.
    pub fn set_dpi(&mut self, dpi: f32) {
        self.dpi = dpi;
    }

    /// The live input state, for `gg_platform::feed` to apply raw events to.
    /// Escape and the close button never get here: quitting is not simulated
    /// state and must work identically while a replay is driving.
    pub fn input(&mut self) -> &mut Input {
        &mut self.input
    }

    /// Where the OS cursor is, in physical pixels — `gg_platform::feed` will
    /// not take it, for the reason `Event::CursorMoved` documents.
    pub fn cursor_at(&mut self, x: f32, y: f32) {
        self.cursor.at = Some((x, y));
    }

    /// What the window does with the pointer this frame: `(held, hidden)`.
    /// Held is mouse-look's grab; hidden is the software-cursor case — the UI
    /// drew the arrow at the steered pointer last tick, so the OS arrow is a
    /// second arrow on the same pixel (§4.9). Whether it draws is the UI
    /// stage's own decision (`Prefs::cursor`, §6 M19); the shell only relays
    /// it. Applied by the windowed loop, the only place a window exists.
    ///
    /// Never hidden while the editor is hosting: `self.ui` is the *game's* UI,
    /// its arrow lives inside the game pane, and the editor draws none of its
    /// own — so obeying it would leave the panels with no cursor at all.
    pub fn pointer(&self) -> (bool, bool) {
        let hidden = !self.held() && !self.editing() && self.ui.cursor_drawn();
        (self.held(), hidden)
    }

    /// The OS pointer should be grabbed and hidden: the game took it, **or** the
    /// editor's camera is mid-drag (§6 M63).
    ///
    /// Two very different facts behind one window call, and they stay separate
    /// everywhere else: `cursor.held` also decides who is *fed* — a held pointer
    /// is the game's input and a dead frame for the editor — while a flying
    /// camera changes nothing about routing at all. Only the arrow is shared,
    /// because a window has one.
    fn held(&self) -> bool {
        self.cursor.held || self.flying()
    }

    /// The editor's camera has the pointer this tick. Always false without the
    /// editor compiled in, which is the tier that ships.
    fn flying(&self) -> bool {
        #[cfg(feature = "editor")]
        return self.editor.as_ref().is_some_and(|e| e.ui.flying());
        #[cfg(not(feature = "editor"))]
        false
    }

    /// Where a host should warp the OS cursor now that a grab has ended, once
    /// (§6 M63).
    ///
    /// Consumed rather than read: the warp is an *edge*, and a windowed loop that
    /// polls this every frame would otherwise pin the arrow to one spot for the
    /// rest of the session. `None` on every frame but the one a drag ended on.
    ///
    /// It exists because releasing a grab does not put the arrow back. Windows
    /// and X11 get `Confined` rather than `Locked` (`gg_platform::Window::
    /// set_pointer`), so the pointer really moves during the drag and reappears
    /// wherever the turn left it — which for a long turn is against a window
    /// edge, nowhere near the button the operator pressed.
    pub fn take_warp(&mut self) -> Option<(f32, f32)> {
        self.warp.take()
    }

    /// Follow the editor camera's grab, and put the arrow back where the
    /// operator left it when the drag ends (§6 M63).
    ///
    /// The release is the interesting half. `Cursor::steer` is frozen for the
    /// length of the grab, so the editor's own pointer has not moved and is
    /// exactly where the press happened; warping the OS cursor onto it — and
    /// telling this side that is where it now is — makes the next steer's delta
    /// zero, which is what "the arrow came back" means. Without both halves the
    /// software and system cursors part company on the first motion after.
    #[cfg(feature = "editor")]
    fn note_flying(&mut self) {
        let flying = self.flying();
        if self.cursor.flying
            && !flying
            && let Some((x, y)) = self.editor_pointer()
        {
            let at = self.ui_fit().to_surface(x, y);
            self.cursor.at = Some(at);
            self.warp = Some(at);
        }
        self.cursor.flying = flying;
    }

    /// Whether the window should be filling a monitor, or `None` where the game
    /// declared no preference and the window it was launched with stands (§6
    /// M46).
    ///
    /// Off the UI stage's cached `Prefs` like every other read of it, so what
    /// this frame acts on is what last tick's menu said — the tick
    /// [`Prefs::modal`] documents, and the same one the picture came from.
    pub fn fullscreen(&self) -> Option<bool> {
        self.prefs().fullscreen()
    }

    /// The player's own way in and out, for the window rather than for the game
    /// (§6 M46): a toggle the *host* owns, spelled Alt+Enter.
    ///
    /// Escape's rule and Escape's reason. A game whose menu is the only way out
    /// of fullscreen is a game that traps a player the first time that menu is
    /// unreachable — and unlike Escape this cannot be claimed by a map, because
    /// a chord has no spelling the action map can hold (§4.7: a binding is one
    /// physical key). Writes the game's own `Prefs` so the choice is the one
    /// thing it must be — the game's to persist — and so a menu drawn next tick
    /// shows what the window actually is.
    pub fn toggle_fullscreen(&mut self) -> anyhow::Result<()> {
        let want = match self.fullscreen() {
            // No preference yet: the toggle is against what the window *is*,
            // which the caller knows and this side does not.
            None => self.window_is_fullscreen,
            Some(on) => on,
        };
        let display = match want {
            true => gg_ecs::boundary::display::WINDOWED,
            false => gg_ecs::boundary::display::FULLSCREEN,
        };
        self.world
            .each(&gg_ecs::Query::<&mut Prefs>::new()?, |_, p: &mut Prefs| {
                p.display = display
            });
        Ok(())
    }

    /// What the windowed loop last observed the OS saying (§6 M46) — needed
    /// because a player can leave fullscreen by ways the shell never hears
    /// about, and the first Alt+Enter after that must put them back in rather
    /// than toggle a stale flag.
    pub fn note_fullscreen(&mut self, is: bool) {
        self.window_is_fullscreen = is;
    }

    /// Hand the pointer back, reporting where a system cursor should be warped
    /// to — in physical pixels, so the arrow reappears where the operator left
    /// it rather than wherever the OS parked it while it was hidden.
    ///
    /// `None` means there was nothing to release and Escape keeps its older
    /// meaning. Two ways to get it, and both are deliberate: the pointer is
    /// already free, or **there is no editor** — a plain run has nowhere to
    /// hand a pointer back *to*, and Escape has quit every demo since M4 (§6
    /// M15.1).
    pub fn release_pointer(&mut self) -> Option<(f32, f32)> {
        let (x, y) = self.editor_pointer().filter(|_| self.cursor.held)?;
        self.cursor.held = false;
        let at = self.ui_fit().to_surface(x, y);
        self.cursor.at = Some(at);
        Some(at)
    }

    /// The surface the canvas is fitted into: what the *window* last said it
    /// is, which leads the swapchain by one recreate (see [`App::resize`]).
    ///
    /// A headless run never resizes, so this stays the canvas it starts at —
    /// which is what a headless editor session is laid out at, and so what a
    /// recorded one must be replayed at (`gg_editor::ui_scale`, §6 M15.1's
    /// residual).
    fn surface(&self) -> (u32, u32) {
        self.extent
    }

    /// The swapchain's extent, or `None` before a renderer exists. For the
    /// windowed loop's `gg::resize` diagnostic: the question a black band
    /// during a drag asks is whether the swapchain followed the window, and
    /// only this side can answer it.
    pub fn surface_extent(&self) -> Option<(u32, u32)> {
        self.gpu.as_ref().map(Renderer::extent)
    }

    /// How the UI's canvas sits on the surface.
    ///
    /// Two different answers, because there are two different UIs: the editor
    /// fills the window at a whole scale (§6 M15.1) and a game's canvas is
    /// letterboxed into it (§4.9). One function so the arrow the OS draws and
    /// the pointer a hit test uses cannot be mapped by two different rules.
    fn ui_fit(&self) -> gg_ui::Fit {
        #[cfg(feature = "editor")]
        if self.editor.is_some() {
            return gg_editor::fit(self.editor_surface(), self.dpi);
        }
        gg_ui::Fit::new(self.surface())
    }

    /// Where the *game's* canvas sits: the whole surface, or the editor's game
    /// pane — the widget half of the rule [`gg_render::Renderer::set_viewport`]
    /// states for the scene: a picture is composed for the rectangle that shows
    /// it. Moves only the picture — the pointer and the hit test are in canvas
    /// units ([`gg_ui::Ui::frame`]), which is what keeps this out of replays.
    fn game_fit(&self) -> gg_ui::Fit {
        #[cfg(feature = "editor")]
        if let Some(editing) = &self.editor {
            // The pane the *last* tick laid out — `Editor::tick` runs after the
            // game's UI — so the picture trails a pane drag by one tick, same
            // as the resize paint `play.rs` documents.
            return gg_ui::Fit::inside(editing.ui.viewport_rect());
        }
        gg_ui::Fit::new(self.surface())
    }

    /// The surface the editor lays its panes out for: `--editor-extent` when
    /// one was named, the real one otherwise (§6 M15.1).
    #[cfg(feature = "editor")]
    fn editor_surface(&self) -> (u32, u32) {
        self.editor
            .as_ref()
            .and_then(|editing| editing.extent)
            .unwrap_or_else(|| self.surface())
    }

    /// The editor's pointer in canvas units, if there is an editor.
    fn editor_pointer(&self) -> Option<(f32, f32)> {
        #[cfg(feature = "editor")]
        return self.editor.as_ref().map(|e| e.ui.pointer());
        #[cfg(not(feature = "editor"))]
        None
    }

    /// Ticks the replay covers — what bounds a `--replay` run that named no
    /// frame count of its own.
    pub fn ticks(&self) -> Option<u64> {
        self.drive.ticks()
    }

    /// Take over a session a predecessor staged (§4.2.2).
    ///
    /// The world lands in the components *this* build declares, so a restart
    /// that crossed a schema change migrates exactly as a reload would — the
    /// same `restore`, the same report. Call before the loop starts.
    pub fn restore(&mut self, handoff: &gg_core::Handoff) -> anyhow::Result<()> {
        let report = self.world.restore(&Snapshot::decode(&handoff.world)?)?;
        self.next_tick = handoff.tick;
        let migrated = !report.is_clean();
        info!(
            entities = report.entities,
            tick = handoff.tick,
            migrated,
            "rejuvenated"
        );
        Ok(())
    }

    /// The tick the sim clock resumes at — zero, or where a predecessor stopped.
    pub fn next_tick(&self) -> u64 {
        self.next_tick
    }

    /// Load a save written by *any* build (§6 M14). Call before the loop starts,
    /// for the same reason [`App::restore`] is: the world it installs is the one
    /// tick zero of this run acts on.
    ///
    /// Unlike a handoff, this may cross a schema change and may not cross a
    /// loss — `gg_ecs::world::save` holds that policy, and a refusal here is a
    /// named component rather than a failed run.
    pub fn load_save(&mut self, path: &Path) -> anyhow::Result<()> {
        let save = Save::decode(&std::fs::read(path)?)?;
        let report = self.world.load(&save)?;
        self.next_tick = save.tick();
        for (declared, outcome) in &report.components {
            if !matches!(outcome, ComponentOutcome::Reused) {
                info!(component = %declared, ?outcome, "migrated");
            }
        }
        info!(
            entities = report.entities,
            tick = save.tick(),
            provenance = format!("{:032x}", save.provenance()),
            migrated = !report.is_clean(),
            "save loaded"
        );
        Ok(())
    }

    /// Write the played world where `--save` said. After the loop, so what lands
    /// is the session someone actually had.
    ///
    /// The provenance is the game dylib's content hash — the same number a
    /// replay segment names (§4.7) — so a save and a recording of the run that
    /// produced it agree about which build made them.
    pub fn write_save(&self, path: &Path) -> anyhow::Result<()> {
        let save = Save::new(self.world.snapshot(), self.next_tick, self.lib.code_hash());
        let bytes = save.encode();
        // The directory is made, not required: with the editor open the default
        // target is this shell's own invention (`Editing::new`), and a save
        // button that fails because nobody had run one before is not a missing
        // directory, it is a lost session.
        // Atomically (§6 M48), and the directory is `replace`'s business now:
        // what a save replaces is a file somebody already has, and half of one
        // is worse than the last one.
        let wrote = crate::player::replace(path, &bytes);
        crate::player::note(path, &wrote);
        wrote?;
        info!(
            path = %path.display(),
            tick = save.tick(),
            entities = save.snapshot().entity_count(),
            bytes = bytes.len(),
            "save written"
        );
        Ok(())
    }

    /// The player's preferences as text, beside their saves (§6 M42).
    ///
    /// Read out of the world, never echoed back from the file that seeded it —
    /// which is what makes a menu's edit outlive the process that took it, and
    /// is the whole point of the round trip. A failure is swallowed here and
    /// counted in [`player::note`](crate::player::note): a game that cannot write
    /// a preference is a game that opens at defaults next time, and refusing to
    /// exit over it would be worse than that — but the *directory* it could not
    /// write is the player's, and that is a sentence somebody owes them (§6 M54).
    pub fn write_settings(&self, path: &Path) {
        let prefs = self.prefs();
        let write =
            crate::player::replace(path, gg_ecs::boundary::settings::encode(&prefs).as_bytes());
        crate::player::note(path, &write);
        if write.is_ok() {
            info!(path = %path.display(), quiet = prefs.quiet, "settings written");
        }
    }

    /// The world staged for a successor, once the loop is over and the window is
    /// gone. `Some` means this process exists only to start the next one.
    pub fn handoff(&mut self) -> Option<gg_core::Handoff> {
        self.rejuvenate.take()
    }

    /// The recording, once the loop is over. Also where the editor's layout is
    /// written down (§6 M15.1) — the one point every exit path passes through.
    pub fn finish(self) -> Option<Box<Recorder>> {
        #[cfg(feature = "editor")]
        if let Some(editing) = &self.editor {
            editing.remember();
        }
        self.drive.recorder()
    }

    /// One editor tick, returning where a save was asked for (§6 M15).
    ///
    /// Split out of [`Stages::sim_tick`] because the borrows are: the panels
    /// take the world mutably while the profiler view reads the renderer, and
    /// `write_save` needs both released before it runs.
    #[cfg(feature = "editor")]
    fn editor_tick(
        &mut self,
        tick: u64,
        ui: &UiTick,
        extent: (u32, u32),
    ) -> anyhow::Result<Option<std::path::PathBuf>> {
        // Before the borrow below, which takes the editor mutably while this
        // reads the dylib beside it.
        let title = self.title();
        let project = (!self.lib.is_absent()).then(|| self.lib.name());
        // Through the recording and not around it (§6 M16): live keystrokes go
        // to the replay's text channel here and a replayed tick reads its own
        // back, so the panel cannot tell which run it is in. Gated on focus, or
        // every `W` a player pressed would be written down twice — once as text
        // and once as the verb it already is. `take_typed` drains either way.
        let wants = self.editor.as_ref().is_some_and(|e| e.ui.wants_text());
        let typed = self.drive.text(tick, &self.input.take_typed(wants));
        let Some(editing) = self.editor.as_mut() else {
            return Ok(None);
        };
        let path = editing.save.display().to_string();
        // The editor opens *Stopped*, one tick in: the bootstrap tick has just
        // run (`Editing::advance`'s `!opened` arm), the world now exists, and
        // Stopped — nothing captured — is where an edit sticks. Play is the
        // operator's first click, which is where the capture moved. A preloaded
        // world (§6 M14's save as the opening scene) constructs `opened` true
        // and never reaches this edge.
        if !core::mem::replace(&mut editing.opened, true) {
            editing.paused = true;
        }
        let commands = editing.ui.tick(
            &mut self.world,
            ui,
            &gg_editor::Frame {
                extent,
                dpi: self.dpi,
                tick,
                hz: self.hz,
                play: editing.play(),
                input: Some(&self.input),
                typed: &typed,
                passes: self.gpu.as_ref().map_or(&[][..], Renderer::pass_timings),
                memory: self.gpu.as_ref().map(Renderer::memory).unwrap_or_default(),
                save_path: &path,
                title: &title,
                project: project.as_deref(),
                projects: &editing.projects,
                maximized: editing.maximized,
                // The panel is a reader of the same record `gg-tools mcp`
                // serves, so the window and the terminal cannot tell an operator
                // two different stories about one reload (§6 M16). Unconditional
                // because `editor` implies `agent`, which is this arm's cfg.
                reload: self.journal.last(),
                // Never: the panels steer the system cursor rather than an
                // arrow of their own, which is why `App::pointer` refuses to
                // hide it while an editor is hosting. What draws its own is
                // `gg-golden`, which has no window to put one in (§6 M15.1).
                draw_cursor: false,
            },
        );
        if let Some(playing) = commands.playing {
            // Entering from `Stopped` is what captures; entering from `Paused`
            // is a no-op inside `Stash`, because the world to go back to is the
            // one play *began* at and not the one it was last unpaused at.
            if playing {
                editing.stash.enter(tick, &self.world);
            }
            editing.paused = !playing;
            info!(
                tick,
                playing,
                entities = self.world.len(),
                "editor: play state"
            );
        }
        // A step from `Stopped` is a play of exactly one tick, so it captures
        // first: stepping the scene itself is the one thing stop exists to
        // prevent, and `Editing::advance` refuses a step with nothing held.
        if commands.step {
            editing.stash.enter(tick, &self.world);
        }
        editing.step = commands.step;
        editing.open = commands.open.or(editing.open);
        if commands.stop {
            // Paused rather than left advancing: `Stopped` is where edits stick,
            // and a stop that kept running would discard the next one too.
            editing.paused = true;
            // Unlogged here on purpose — `Stash::stop` states `changed` and
            // `identical`, which is the pair the gate reads, and a second line
            // beside it would be a second thing to keep true.
            editing.stash.stop(tick, &mut self.world)?;
            // Unconditional: a stop that restored nothing costs one re-registered
            // cue bank and no sound, where a stop that *did* restore rewound
            // every `seq` — and a rewind is a difference like any other.
            self.audio.forget();
        }
        Ok(commands.save.then(|| editing.save.clone()))
    }

    /// The project a click in the picker asked for (§6 M15.1 item 4) — read
    /// after the loop, because the session that answers it is the next one.
    #[cfg(feature = "editor")]
    pub fn opening(&self) -> Option<&gg_editor::project::Project> {
        let editing = self.editor.as_ref()?;
        editing.projects.get(editing.open?)
    }

    /// What the title bar asked of the window (§6 M15.1 item 5). Applied by the
    /// windowed loop, which is the only place a window exists — a headless
    /// session produces these and drops them, which is the whole reason they
    /// are commands and not state.
    #[cfg(feature = "editor")]
    pub fn take_window_command(&mut self) -> Option<gg_editor::WindowCommand> {
        self.editor.as_mut()?.ui.take_window_command()
    }

    /// Which resize border the editor's pointer is over, for the system cursor
    /// (§6 M15.1 item 5). `None` while the game holds the pointer: the arrow is
    /// hidden then, and the shape it would carry back out is stale.
    #[cfg(feature = "editor")]
    pub fn resize_edge(&self) -> Option<gg_editor::Edge> {
        self.editor
            .as_ref()
            .filter(|_| !self.cursor.held)?
            .ui
            .resize_edge()
    }

    /// Tell the editor what the window is, for the caption button that draws
    /// maximize or restore. Stated by the windowed loop for `pointer_held`'s
    /// reason: the answer is the OS's and this is the only side that can ask.
    #[cfg(feature = "editor")]
    pub fn set_maximized(&mut self, maximized: bool) {
        if let Some(editing) = self.editor.as_mut() {
            editing.maximized = maximized;
        }
    }

    /// What to call the window: the game this shell was pointed at. Also what
    /// the editor's own title bar says, since there is no OS one to say it (§6
    /// M15.1 item 5).
    pub fn title(&self) -> String {
        format!("gg — {}", self.lib.name())
    }

    /// Whether the editor is open, asked from paths that exist without it — the
    /// reload's verb list, and the window's decorations.
    pub fn editing(&self) -> bool {
        #[cfg(feature = "editor")]
        return self.editor.is_some();
        #[cfg(not(feature = "editor"))]
        false
    }

    /// Adopt a rebuilt dylib. Every fallible step runs before any state moves,
    /// so a refusal leaves the last-good build playing an untouched session —
    /// then one of §4.2.2's two shapes, chosen by the schema check per hashed
    /// component: all fingerprints unchanged → the world is already the new
    /// build's layout, so the swap is the systems-table pointer and nothing
    /// else — state untouched, between ticks; any changed → snapshot, register
    /// the new schemas into a *fresh* world, restore through the migration
    /// path, then swap. Fresh rather than re-registered, because a component
    /// whose schema moved needs its column rebuilt, and rebuilding under live
    /// rows is what snapshot/restore exists to avoid.
    #[cfg(feature = "hot-reload")]
    fn swap(
        &mut self,
        reloaded: gg_core::reload::watch::Reloaded,
    ) -> anyhow::Result<gg_ecs::MigrationReport> {
        // Re-parsed rather than carried: an edit that appends an action moves
        // the id space, and a map bound against the old one would fire the wrong
        // verb (§4.7). Key state resets with it, which is why this is the one
        // reload effect a player can feel. Into a local, not `self.input` — the
        // migration below can still refuse, and a refusal must not have
        // replaced the running build's map.
        // SAFETY: `reloaded.lib` is verified and never unloaded.
        let (verbs, extra) = unsafe { verbs_for(&reloaded.lib, self.editing()) };
        let input = bind(
            &format!("{}{extra}", self.bindings),
            &self.rebinds,
            &self.drive,
            verbs,
        )
        .map_err(|e| reloaded.lib.refuse(e))?;
        // SAFETY: both dylibs are verified and never unloaded.
        let unchanged = unsafe {
            gg_ecs::boundary::same_schemas(self.lib.components(), reloaded.lib.components())
        };
        let report = if unchanged {
            // Pointer swap: the world's columns already are the new build's
            // layout, and the registry keeps the retired build's descriptors —
            // sound for the reason every boundary `&'static` is: a dylib is
            // never unloaded. The audio observer's memory is kept too, because
            // an untouched world's `seq`s carry on (§6 M18 item 2).
            gg_ecs::MigrationReport::untouched(self.world.len())
        } else {
            let snapshot = self.world.snapshot();
            let mut world = World::new();
            // Every failure from here to the swap is charged to the leak budget:
            // the image is mapped and never unloaded, so a refused adopt costs
            // exactly what an accepted one does (§4.2.2).
            adopt(&mut world, &reloaded.lib).map_err(|e| reloaded.lib.refuse(e))?;
            let report = world
                .restore(&snapshot)
                .map_err(|e| reloaded.lib.refuse(e))?;
            // One line per component that actually moved (§4.2.2). The reused
            // ones are the majority of every reload, and printing them would
            // bury the one line the reader came for.
            for (declared, outcome) in &report.components {
                if !matches!(outcome, ComponentOutcome::Reused) {
                    info!(component = %declared, ?outcome, "migrated");
                }
            }
            self.world = world;
            // The migrated world is one the audio observer has not seen: a
            // `Sound` the new build declares afresh comes back with `seq` at
            // zero, and a difference is a trigger. Forgotten rather than
            // carried, so a reload is silent at the seam instead of playing
            // whatever the retired build's cue bank ended on (§6 M18 item 2).
            self.audio.forget();
            report
        };
        self.looks = input.looks();
        self.input = input;
        // Same move, same reason: an edit that appends a verb moves the id
        // space the UI's four names resolved to (§4.7). An edit that *deletes*
        // one leaves the UI unclickable rather than clicking the wrong thing.
        self.ui_binding = binding(&verbs);
        self.rejuvenate.retire(self.lib.bytes());
        self.lib = reloaded.lib;
        // The old segment closes and a new one opens at the first tick the new
        // code will run, so a recording made across a reload still says which
        // build produced which ticks (§4.2.2, §4.7).
        self.drive
            .open_segment(self.next_tick, self.lib.code_hash());
        self.halted = false;
        // Staged after the swap, so a successor inherits the *migrated* world
        // rather than the one the retired dylib understood.
        if self.rejuvenate.due() {
            let staged = self.world.snapshot().encode();
            self.rejuvenate
                .stage(self.next_tick, staged, self.drive.survives_restart());
        }
        info!(
            entities = report.entities,
            // The first tick the new code runs, which is also where the replay
            // segment opened — named so a recording and a log agree (§4.7).
            tick = self.next_tick,
            load_ms = reloaded.load_time.as_millis(),
            // The M5 budget's clock, measured end to end: file event to the tick
            // boundary the new code first runs at.
            save_to_swap_ms = reloaded.saved_at.elapsed().as_millis(),
            // Which §4.2.2 shape this reload took — what lets a gate insist the
            // fast path actually ran instead of grading the 2 s bar on the slow
            // one for every trivial edit.
            migration = if unchanged { "pointer-swap" } else { "restore" },
            "game reloaded"
        );
        Ok(report)
    }
}

/// What [`App::open_seam`] takes before the swap destroys it (§6 M16).
#[cfg(all(feature = "agent", feature = "hot-reload"))]
struct Opened {
    code_before: String,
    code_after: String,
    state_before: String,
    load_ms: u64,
    /// Held rather than elapsed here: §9's bar is file event to the tick the new
    /// code first runs at, and that tick is on the far side of the swap.
    saved_at: std::time::Instant,
}

#[cfg(all(feature = "agent", feature = "hot-reload"))]
impl App {
    fn open_seam(&self, reloaded: &gg_core::reload::watch::Reloaded) -> Opened {
        Opened {
            code_before: format!("{:032x}", self.lib.code_hash()),
            code_after: format!("{:032x}", reloaded.lib.code_hash()),
            state_before: self.world.canonical_hash().to_string(),
            load_ms: reloaded.load_time.as_millis().min(u128::from(u64::MAX)) as u64,
            saved_at: reloaded.saved_at,
        }
    }

    /// Record the crossing and republish. Both outcomes, because a refusal is
    /// the case the record exists for — an accepted reload is already visible in
    /// the game.
    fn close_seam(
        &mut self,
        opened: Option<Opened>,
        result: Result<&gg_ecs::MigrationReport, &anyhow::Error>,
    ) {
        let outcome = match result {
            Ok(_) => gg_agent::Outcome::Accepted,
            Err(refused) => gg_agent::Outcome::Refused {
                kind: refused
                    .downcast_ref::<gg_core::ReloadError>()
                    .map_or("Other", gg_core::ReloadError::kind),
                detail: refused.to_string(),
            },
        };
        // Only the components that moved, for §4.2.2's reason: a clean reload is
        // the majority of every session, and listing it would bury the one line
        // the reader came for.
        let changes = result.map_or_else(
            |_| Vec::new(),
            |report| {
                report
                    .components
                    .iter()
                    .filter(|(_, o)| !matches!(o, ComponentOutcome::Reused))
                    .map(|(component, outcome)| {
                        let (kind, defaulted, retyped) = match outcome {
                            ComponentOutcome::Reused => ("reused", Vec::new(), Vec::new()),
                            ComponentOutcome::Dropped => ("dropped", Vec::new(), Vec::new()),
                            ComponentOutcome::Migrated {
                                defaulted, retyped, ..
                            } => ("migrated", defaulted.clone(), retyped.clone()),
                        };
                        gg_agent::Change {
                            component: component.clone(),
                            kind,
                            defaulted,
                            retyped,
                        }
                    })
                    .collect()
            },
        );
        // Read again rather than carried: on a refusal nothing was swapped, so
        // this is `state_before` — and a record that said otherwise would be
        // claiming a migration that did not happen.
        let state_after = self.world.canonical_hash().to_string();
        let code_before = opened.as_ref().map_or_else(
            || format!("{:032x}", self.lib.code_hash()),
            |o| o.code_before.clone(),
        );
        self.journal.record(gg_agent::Seam {
            tick: self.next_tick,
            outcome,
            code_before,
            // `None` and not a default, throughout: verification can refuse an
            // artifact before it is opened, and there is then no second build to
            // name and nothing that was timed. A zero here would be the fastest
            // reload of the session (§6 M16).
            code_after: opened.as_ref().map(|o| o.code_after.clone()),
            state_before: opened
                .as_ref()
                .map_or_else(|| state_after.clone(), |o| o.state_before.clone()),
            state_after,
            entities: result.map_or(0, |report| report.entities),
            changes,
            load_ms: opened.as_ref().map(|o| o.load_ms),
            save_to_swap_ms: opened
                .as_ref()
                .map(|o| o.saved_at.elapsed().as_millis().min(u128::from(u64::MAX)) as u64),
        });
        self.publish_journal();
    }
}

#[cfg(feature = "agent")]
impl App {
    /// Publish, and say so once if the write fails. A record that cannot be
    /// written is not worth halting a play session over — the game is the thing
    /// running, and the record is what an agent reads *about* it (§6 M16).
    fn publish_journal(&mut self) {
        self.journal.at_tick(self.next_tick);
        if let Err(e) = self.journal.publish(&agent_dir()) {
            warn!(error = %e, "could not publish the agent record");
        }
    }
}

/// Where the record is published. Overridable so two sessions on one tree do not
/// overwrite each other's — see [`gg_agent::Journal::publish`], which keys the
/// temp file by pid but writes one final path per directory.
#[cfg(feature = "agent")]
fn agent_dir() -> std::path::PathBuf {
    std::env::var_os("GG_AGENT_DIR").map_or_else(
        || std::path::PathBuf::from("target/gg-agent"),
        std::path::PathBuf::from,
    )
}

#[cfg(feature = "agent")]
fn journal(args: &crate::Args, game: &str) -> gg_agent::Journal {
    // Keyed on the tier meta-features, not on `tracy`: `xtask run --tracy` adds
    // tracy to a *dev* shell, which would otherwise publish a record claiming
    // instruments the build does not carry (an agent reading it needs the truth).
    let mut journal = gg_agent::Journal::new(game, format!("tier-{}", crate::active_tier()));
    // The third thing the record hands an agent: what the human just did, as
    // something replayable rather than described (§4.7).
    if let Some(path) = &args.record {
        journal.recording(path);
    }
    journal
}

/// Whether a held pointer goes back to the operator this tick (§6 M15.1).
///
/// `running` is the **editor's** transport, so `Some(false)` — a scene stopped
/// under an open editor — is the one case that hands the mouse back. `None` is a
/// run with no editor, which has nowhere to hand a pointer *to*; that is the
/// same rule [`App::release_pointer`] states about Escape, and it is a rule
/// rather than a shortcut because a plain game is *supposed* to hold the mouse
/// for the life of the session.
///
/// Its own function because reading `None` as "stopped" is what handed every
/// mouse-look game its pointer back on tick 0 of any build carrying the editor
/// feature — which is `tier-dev`, so every run anyone plays. Raw device motion
/// arrives grabbed or not, so looking still worked perfectly and the only
/// symptom was the OS arrow sitting on top of the game for the whole session.
#[cfg(feature = "editor")]
fn hands_back(held: bool, running: Option<bool>) -> bool {
    held && running.is_some_and(|running| !running)
}

/// The clips in `path`, as a bank `gg-audio` can be handed (§6 M43).
///
/// Read here rather than in `gg-audio`, which links no pack and parses no
/// format (§3), and not through the renderer, which opens the same file for the
/// GPU's sake and does not exist at all under `GG_HEADLESS=1` — a headless run
/// still fires cues, and a gate has to be able to ask whether one resolved.
///
/// A pack that will not open is silence and a line in the log, never a refusal:
/// the renderer is the half that needs one to draw, and a game whose audio
/// failed to load should still be playable enough to say so.
fn clips(path: &Path) -> gg_audio::Bank {
    let mut bank = gg_audio::Bank::new();
    let pack = match gg_assets::Pack::open(path) {
        Ok(pack) => pack,
        Err(error) => {
            warn!(pack = %path.display(), %error, "audio: no clips — the pack did not open");
            return bank;
        }
    };
    for entry in pack.entries() {
        if entry.kind() != Some(gg_assets::AssetKind::Clip) {
            continue;
        }
        match gg_assets::Clip::read(pack.blob(entry)) {
            Ok(clip) => bank.add(entry.id.0, clip.rate(), clip.samples()),
            Err(error) => warn!(name = pack.name(entry), %error, "audio: clip skipped"),
        }
    }
    bank
}

/// The OS cursor, and which of the two things a mouse does is happening (§6
/// M15.1).
///
/// Two motion sources exist because there are two jobs — raw device deltas for
/// looking, cursor motion for pointing — and this decides which one the sim is
/// being fed. Held is the game's: the pointer is locked and hidden and the
/// deltas are raw. Free is the editor's: the OS draws the arrow, and the shell
/// *steers* the derived pointer onto it by feeding the difference, so the
/// cursor a human sees and the cursor a hit test uses are the same pixel.
///
/// Steering rather than assigning is what keeps this out of the sim: the
/// difference is an ordinary axis value in an ordinary recorded frame, so a
/// session replays without anything here existing (§4.7).
struct Cursor {
    /// Where the OS last said the cursor is, in physical pixels. `None` until
    /// it has been over the surface at all — a session driven by a replay never
    /// sets it, which is exactly right.
    at: Option<(f32, f32)>,
    /// The canvas position, in `AXIS_SCALE`ths, the last steer aimed at. Not a
    /// copy of the router's pointer: it is what this shell *asked* for, and the
    /// router applies precisely that delta and lands there.
    steered: (i32, i32),
    /// The game holds the pointer.
    held: bool,
    /// The editor's camera holds it instead (§6 M63) — a drag is turning or
    /// sliding the view. Frozen for [`Cursor::steer`]'s purposes exactly as
    /// `held` is, and for a different reason: the OS pointer is grabbed, so the
    /// positions still arriving are the grab's rather than the hand's.
    flying: bool,
}

impl Cursor {
    /// `free` starts the pointer unheld: an editor opening, or a game whose UI
    /// binds the pointer verbs — a menu is aimed with an arrow the operator can
    /// see, and grabbing it would park the steer's only input. Everything else
    /// keeps the old start: mouse-look wants the pointer inside the window
    /// before the first frame, and every demo has always started that way.
    fn new(free: bool) -> Cursor {
        Cursor {
            at: None,
            steered: (0, 0),
            held: !free,
            flying: false,
        }
    }

    /// The cursor delta to feed this tick, if any.
    ///
    /// `None` while the game holds the pointer — the arrow is locked and hidden
    /// and the editor's pointer parks where it was — and `None` before the OS
    /// has reported a position at all.
    fn steer(&mut self, fit: gg_ui::Fit) -> Option<(i32, i32)> {
        let (x, y) = self.at.filter(|_| !self.held && !self.flying)?;
        let delta = fit.steer(self.steered, x, y);
        self.steered = fit.to_canvas_fixed(x, y);
        (delta != (0, 0)).then_some(delta)
    }
}

/// The editor and the three shell states only it creates (§6 M15).
///
/// One struct rather than three fields, so the whole of what `--editor` costs
/// the shell is `Option<Editing>` and the tier that lacks the crate lacks the
/// concept — pause, single-step and a save *button* are meaningless without it.
#[cfg(feature = "editor")]
struct Editing {
    ui: gg_editor::Editor,
    /// The sim is not advancing. Starts **false** so the bootstrap tick can
    /// run — the world does not exist before it — and flips at the `opened`
    /// edge: the editor opens *Stopped*, one tick in (§6 M15.2 post-close).
    paused: bool,
    /// The world captured at the play edge (§6 M15.2). Empty is `Stopped`, and
    /// `Stopped` is the only state an inspector edit survives.
    stash: Stash,
    /// False until the bootstrap tick has run. Constructed **true** when the
    /// world was preloaded (§6 M14's save as the opening scene): data supplied
    /// the world, so no game tick runs before the operator asks for one.
    /// Read by [`advance`](Editing::advance): the bootstrap tick is the one
    /// tick that must run with nothing captured.
    opened: bool,
    /// Advance one tick, then pause again. Consumed by [`Editing::advance`].
    step: bool,
    /// Where the save button writes — `--save` if given, so a gate can name it.
    save: std::path::PathBuf,
    /// `--editor-extent`, when the layout is to be built for a surface other
    /// than this run's (§6 M15.1).
    extent: Option<(u32, u32)>,
    /// Where the dock layout is remembered between sessions, or `None` for a
    /// run that must not remember one.
    layout: Option<std::path::PathBuf>,
    /// What the window last said it was, for the caption button that draws it.
    /// Stated by the windowed loop, so a headless session leaves it false and
    /// draws the maximize glyph — which is the truth with no window (§1.5).
    maximized: bool,
    /// What the picker offers, empty in a session already over a game (§6 M15.1
    /// item 4) — so the picker is not a way to switch projects mid-session.
    projects: Vec<gg_editor::project::Project>,
    /// Which of them a click asked for. Read after the loop by [`App::opening`].
    open: Option<usize>,
}

#[cfg(feature = "editor")]
impl Editing {
    fn new(args: &crate::Args, lib: &GameLib, recorded: Option<(u32, u32)>) -> Editing {
        use gg_editor::persist;
        let stem = &lib.name();
        // A live session's save button writes the project's opening scene
        // (`crate::opening_scene`), which is what makes the scene *mutable
        // data*: stop, edit, save, and the next session opens from it. Gates
        // name `--save` and replays keep the out-of-tree default — a replayed
        // session must not leave a scene behind for the next run to open.
        let save = args.save.clone().unwrap_or_else(|| {
            crate::scene_path(args.input.as_deref())
                .filter(|_| crate::may_touch_project(args))
                .unwrap_or_else(|| persist::save_path(stem))
        });
        // The layout *is* hit-testing (§6 M15.1), so a session that started from
        // a file the gate never saw would land its clicks somewhere else.
        let layout = crate::may_touch_project(args).then(|| persist::layout_path(stem));
        let mut ui = gg_editor::Editor::new(args.pack.as_deref());
        if let Some(path) = &layout {
            persist::restore(&mut ui, path);
        }
        // Data supplied the world, so the bootstrap tick is not owed: the
        // session opens Stopped at the save's own tick, zero game ticks run.
        let preloaded = args.load.is_some() || crate::opening_scene(args).is_some();
        Editing {
            // The working directory, for `gg.cfg`'s reason: a launcher is
            // started *in* a workspace, and a flag pointing elsewhere would be a
            // second answer to where a project is. Scanned in every editor
            // session and drawn in none that has a game — the picker's condition
            // is `Frame::project`, not this list being empty.
            projects: gg_editor::project::scan(Path::new(".")),
            open: None,
            ui,
            paused: preloaded,
            stash: Stash::default(),
            opened: preloaded,
            step: false,
            save,
            // The flag wins over the file: it is what authors a session for a
            // surface other than the one it was recorded at (§6 M15.1), and a
            // header that overrode it would delete that use.
            extent: args.editor_extent.or(recorded),
            layout,
            maximized: false,
        }
    }

    /// Remember the layout, if this run is one that may.
    fn remember(&self) {
        if let Some(path) = &self.layout {
            gg_editor::persist::remember(&self.ui, path);
        }
    }

    /// Whether the sim advances this tick, spending a single step if one is due.
    ///
    /// A stopped session does not advance whatever the step says: `Commands`
    /// turns a step from `Stopped` into a play first, so a step that reached
    /// here with nothing captured would be one the editor never asked for.
    ///
    /// The `!opened` arm is the bootstrap tick, which advances with nothing
    /// captured: the world does not exist before it, and the Stopped state the
    /// editor opens into needs a world to show (§6 M15.2 post-close). A
    /// preloaded session constructs `opened` true and skips it — the save is
    /// the world, and running a game tick over it before the operator asked
    /// would make "load paused" a lie by one tick.
    fn advance(&mut self) -> bool {
        let step = core::mem::take(&mut self.step);
        (!self.opened || self.stash.held()) && (!self.paused || step)
    }

    /// What the transport draws, derived rather than stored — two bits already
    /// say it, and a third field could disagree with them.
    fn play(&self) -> gg_editor::Play {
        match (self.stash.held(), self.paused) {
            (false, _) => gg_editor::Play::Stopped,
            (true, false) => gg_editor::Play::Running,
            (true, true) => gg_editor::Play::Paused,
        }
    }
}

/// The world as it was when play began — the bytes a stop returns to (§6 M14,
/// §6 M15.2).
///
/// Kept **encoded**. Comparing bytes at stop then proves the round trip and the
/// equality at once; a `Snapshot` held by value would prove neither, because
/// nothing would have serialized it.
///
/// One type for both clocks that drive it: `--play <enter>:<stop>` reaches these
/// two methods from a tick number and the editor's transport reaches them from a
/// button, which is what makes "both reach the same two calls" checkable rather
/// than a claim about two similar functions.
#[derive(Default)]
struct Stash(Option<Vec<u8>>);

impl Stash {
    /// Whether a world is captured — play mode, paused or not.
    fn held(&self) -> bool {
        self.0.is_some()
    }

    /// The play edge. Re-entering without a stop is a no-op rather than a
    /// second capture: the world to go back to is the one play *began* at.
    fn enter(&mut self, tick: u64, world: &World) {
        if self.held() {
            return;
        }
        let stash = world.snapshot().encode();
        info!(
            tick,
            entities = world.len(),
            bytes = stash.len(),
            "play mode entered"
        );
        self.0 = Some(stash);
    }

    /// The stop edge: restore what was captured, and report both halves of the
    /// claim. `Ok(false)` means nothing was captured, which is a stop from
    /// `Stopped` and does nothing to the world.
    ///
    /// `identical` over a session that changed nothing is a gate that cannot
    /// fail, so `changed` is read *before* the restore and reported beside it —
    /// the caller requires both.
    fn stop(&mut self, tick: u64, world: &mut World) -> anyhow::Result<bool> {
        let Some(stash) = self.0.take() else {
            return Ok(false);
        };
        let changed = world.snapshot().encode() != stash;
        world.restore(&Snapshot::decode(&stash)?)?;
        let identical = world.snapshot().encode() == stash;
        info!(
            tick,
            entities = world.len(),
            changed,
            identical,
            "play mode stopped"
        );
        Ok(true)
    }
}

/// The editor's play/stop on a script, so a tier can run it (§6 M14).
///
/// M15.2 drives these two edges from a button; a `<enter>:<stop>` spec drives
/// them from a tick number, and both reach the same [`Stash`]. It reports rather
/// than asserts: a gate that lived inside the program under test would be graded
/// by the thing it grades, so the shell states what happened and `xtask` decides.
struct PlayMode {
    enter: u64,
    stop: u64,
    stash: Stash,
}

impl PlayMode {
    fn parse(spec: &str) -> anyhow::Result<PlayMode> {
        let (enter, stop) = spec.split_once(':').ok_or_else(|| {
            anyhow::anyhow!("--play wants `<enter tick>:<stop tick>`, got `{spec}`")
        })?;
        let (enter, stop) = (enter.parse()?, stop.parse()?);
        anyhow::ensure!(
            enter < stop,
            "--play {spec}: nothing happens between {enter} and {stop}"
        );
        Ok(PlayMode {
            enter,
            stop,
            stash: Stash::default(),
        })
    }

    /// Called at the top of every tick, so the world compared at `stop` is the
    /// world captured at `enter` — the same point in the frame, with nothing
    /// half-advanced between the two readings.
    /// `true` when this tick restored the world — the caller's cue to forget
    /// what its observers knew about it.
    fn edge(&mut self, tick: u64, world: &mut World) -> anyhow::Result<bool> {
        if tick == self.enter {
            self.stash.enter(tick, world);
        } else if tick == self.stop {
            return self.stash.stop(tick, world);
        }
        Ok(false)
    }
}

/// A window losing focus on a script, so a tier can run it (§6 M49).
///
/// [`PlayMode`]'s shape one flag over, and with the one difference that is the
/// milestone: **frames, not ticks.** A suspension is measured in the clock it
/// does not stop, and a `<until tick>` would name a tick the suspension itself
/// guarantees never arrives.
///
/// `Copy` and half-open, so `covers` is the whole of it — there is no state to
/// keep, because a frame number is the only thing this has ever needed to know.
#[derive(Clone, Copy)]
struct Away {
    from: u64,
    until: u64,
}

impl Away {
    fn parse(spec: &str) -> anyhow::Result<Away> {
        let (from, until) = spec.split_once(':').ok_or_else(|| {
            anyhow::anyhow!("--away wants `<from frame>:<until frame>`, got `{spec}`")
        })?;
        let (from, until) = (from.parse()?, until.parse()?);
        anyhow::ensure!(
            from < until,
            "--away {spec}: nobody is away between {from} and {until}"
        );
        Ok(Away { from, until })
    }

    fn covers(self, frame: u64) -> bool {
        (self.from..self.until).contains(&frame)
    }
}

/// Register the host's own §4.5 protocol types, then everything the dylib
/// declares. Returns the dylib's count.
///
/// The order *is* the check. The host reads these five through typed queries at
/// extract and at the UI tick, so putting its compiled-in schemas in first is
/// what gives `World::adopt`'s comparison something to disagree with — into a
/// fresh world every declaration would take the insert branch, and a dylib that
/// laid one out differently would be accepted and then panic mid-extract,
/// outside every shim (§4.2.2).
fn adopt(world: &mut World, lib: &GameLib) -> anyhow::Result<usize> {
    world.register::<Renderable>()?;
    world.register::<Eye>()?;
    world.register::<Model>()?;
    world.register::<Light>()?;
    world.register::<Widget>()?;
    // SAFETY: `lib` is verified — `GameLib::load` returned it — and is never
    // unloaded, which is what makes its descriptors `&'static`.
    let declared = unsafe { world.adopt(lib.components()) }?;
    // §1.13 hazard 6: a field the scan cannot place is a hole in the gate, and
    // load is the once-per-build moment to say so — the alternative is a claim
    // in PLAN.md that quietly covers less than it says.
    #[cfg(all(feature = "nan-scan", debug_assertions))]
    for hole in world.registry().unclassified_fields() {
        warn!(%hole, "nan-scan cannot classify this field — its bytes are not scanned");
    }
    Ok(declared)
}

/// This build's verbs (§4.7).
///
/// # Safety
///
/// A verified, never-unloaded `lib`, so its table and every name in it live as
/// long as the process (§4.2.2).
unsafe fn verbs_of(lib: &GameLib) -> gg_ecs::boundary::Verbs {
    // SAFETY: forwarded to the caller, one line above.
    unsafe { gg_ecs::boundary::read_verbs(lib.verbs()) }
}

/// This build's verbs as the shell actually binds them, and the bindings text
/// for whatever had to be appended (§4.7, §6 M15).
///
/// Without the editor this is the dylib's own lists and an empty string, which
/// is why every caller goes through it rather than through [`verbs_of`]: the
/// three places that need verbs — the map, the replay header, and the UI
/// binding — must agree, and one of them disagreeing would route a replayed
/// click to a different verb.
///
/// # Safety
///
/// As [`verbs_of`].
unsafe fn verbs_for(lib: &GameLib, _editor: bool) -> (gg_ecs::boundary::Verbs, String) {
    // SAFETY: forwarded to the caller.
    let verbs = unsafe { verbs_of(lib) };
    #[cfg(feature = "editor")]
    if _editor {
        return gg_editor::host::open(&verbs);
    }
    (verbs, String::new())
}

/// The replay header this build would record (§4.7): the reproduction
/// environment, and the verb lists whose *order* is the id space it writes.
fn meta(lib: &GameLib, editor: bool, hz: u32) -> ReplayMeta {
    // SAFETY: `lib` is verified and never unloaded.
    let (verbs, _) = unsafe { verbs_for(lib, editor) };
    let contract = gg_math::DETERMINISM_CONTRACT;
    ReplayMeta::new(
        contract,
        crate::active_tier(),
        hz,
        verbs.actions,
        verbs.axes,
    )
}

/// Parse the bindings against *this build's* declared verbs and open the game
/// context. An empty file is legal and means no bindings — a run with nothing
/// pressed, which is exactly what `--frames N` in CI wants.
///
/// A replay is checked here too, and again at every swap: an edit that appends
/// or reorders a verb moves the id space the file was recorded against, and
/// carrying on would replay the wrong verbs rather than fail (§4.7).
fn bind(
    bindings: &str,
    rebinds: &str,
    drive: &Drive,
    verbs: gg_ecs::boundary::Verbs,
) -> anyhow::Result<Input> {
    if let Some(replay) = drive.replay() {
        replay.check_verbs(verbs.actions, verbs.axes)?;
    }
    let mut input = Input::new(ActionMap::parse(bindings, verbs.actions, verbs.axes)?);
    if !input.push_named(CONTEXT) && !bindings.trim().is_empty() {
        warn!("the bindings declare no `{CONTEXT}` context — nothing is bound");
    }
    // The player's own keys, over the project's (§6 M45), after the context is
    // pushed so the caches `rebind` rebuilds see the layer that is up.
    match input.rebind(rebinds) {
        _ if rebinds.trim().is_empty() => {}
        stale if stale.is_empty() => {
            info!(file = %gg_input::BINDINGS_FILE, "the player's own bindings applied");
        }
        stale => warn!(
            ?stale,
            "bindings: lines this build does not answer to — ignored"
        ),
    }
    Ok(input)
}

impl Stages for App {
    type Error = anyhow::Error;

    #[cfg(feature = "hot-reload")]
    fn reload_check(&mut self) -> anyhow::Result<()> {
        // SAFETY: the watcher stages a byte copy of the artifact this shell was
        // pointed at, and `host_api()` is `&'static`.
        let Some(ready) = (unsafe { self.watch.poll(host_api()) }) else {
            return Ok(());
        };
        // Every failure below leaves the last-good dylib playing. A broken edit
        // usually produces no artifact at all, so the running game never
        // notices; when one does arrive and is refused, the refusal belongs in
        // the terminal and not in the player's session (§6 M0X).
        // Bound, not matched in place: the closure is a scrutinee temporary and
        // would hold the `&mut self` borrow across the arms.
        //
        // Opened before the swap because half the record is state the swap
        // destroys — the retired build's code hash and the pre-migration state
        // hash. `None` is the arm where verification failed before a `Reloaded`
        // existed at all, which is where `HostApiMismatch` lands: there is a
        // refusal to record and no timings to record it with (§6 M16).
        #[cfg(feature = "agent")]
        let mut opened = None;
        let outcome = match ready {
            Err(e) => Err(anyhow::Error::from(e)),
            Ok(r) => {
                #[cfg(feature = "agent")]
                {
                    opened = Some(self.open_seam(&r));
                }
                self.swap(r)
            }
        };
        match outcome {
            Ok(_report) => {
                #[cfg(feature = "agent")]
                self.close_seam(opened, Ok(&_report));
                Ok(())
            }
            Err(refused) => {
                // A refused dylib was mapped and ran its initializers before any
                // check could, and is never unloaded (§4.2.2) — so its bytes are
                // this session's whether or not the swap happened. Charged, not
                // acted on: the restart still waits for a *successful* swap,
                // because rejuvenating into the same refusal only repeats it.
                if let Some(reload) = refused.downcast_ref::<gg_core::ReloadError>() {
                    self.rejuvenate.retire(reload.leaked_bytes());
                }
                #[cfg(feature = "agent")]
                self.close_seam(opened, Err(&refused));
                error!(error = %refused, "reload refused — still running the last good build");
                Ok(())
            }
        }
    }

    fn quitting(&self) -> bool {
        // A menu's quit button (§6 M21), beside rejuvenation's staged successor.
        // Read off the UI stage's cached `Prefs` rather than the world: this
        // takes `&self` and a query does not.
        self.rejuvenate.pending() || self.ui.prefs().closing()
    }

    fn suspended(&mut self, frame: u64) -> bool {
        // The script and the event are one fact, not two: `--away` says the
        // window is not focused on these frames, and everything downstream is
        // the path a real alt-tab takes.
        let focused = self.focused && !self.away.is_some_and(|away| away.covers(frame));
        // Read off the same cache `quitting` uses, so what decides is the `Prefs`
        // the last tick left. A suspended session runs no tick, so the value
        // cannot move while it is waiting — the tick that resumes it is the
        // earliest anything could have changed its mind, which is why this is
        // safe to ask every frame.
        let waiting = !focused && self.prefs().pauses_unfocused();
        if waiting {
            // Every suspended frame rather than on the edge: `hush` is a level,
            // and one missed edge is a game that plays its music to an empty
            // desk for as long as the player is gone.
            self.audio.hush();
            // And on the same reasoning, for the accumulator no tick will empty
            // (§6 M56): a pointer that kept reporting across a five-minute
            // alt-tab would otherwise be spent whole by the tick that resumes,
            // which is a turn measured in minutes on one frame.
            self.input.forget_motion();
        }
        if std::mem::replace(&mut self.waiting, waiting) != waiting {
            info!(frame, waiting, "window focus");
        }
        waiting
    }

    fn ticks_due(&mut self, due: gg_core::Due) {
        self.input.frame_covered(due.covered);
    }

    fn sim_tick(&mut self, tick: u64) -> anyhow::Result<()> {
        // Hazard 5's per-tick call site (§4.2.1, §2's Build-tiers row). An ICD,
        // layer or audio DLL initializer that sets FTZ/DAZ changes every denormal
        // result process-wide and only on the host that loaded it — the two
        // native legs of §5.6 then diverge with nothing naming the cause.
        #[cfg(all(feature = "fp-assert", debug_assertions))]
        gg_math::fpenv::assert_fp_env();
        self.next_tick = tick + 1;
        // What the frames between this tick and the next blend away from (§4.1).
        // At the top, before anything this tick writes — including a `--play`
        // edge and the editor's own edits — so it is exactly the world the
        // previous tick left, whatever path the rest of this one takes.
        //
        // Unconditional, and cheap enough to be: it is two reads over the same
        // rows extract already walks, and a locked pace pays for a table it then
        // never blends with (`alpha` is zero throughout, §4.1). Making it
        // conditional would put the render pace inside the sim tick, which is
        // the one place §1.4 says it may not be.
        self.extracted.capture::<Renderable>(&self.world)?;
        self.extracted.capture_models::<Model>(&self.world)?;
        self.extracted.capture_eye(&self.world)?;
        // A second's cadence, not a tick's: the record's readers poll on a human
        // clock, and a file write inside the sim loop is jitter charged to every
        // frame to keep a number nobody reads that fast. Tick zero publishes on
        // its own, so a tier with the record and no watcher — instrumented — has
        // one from the start instead of only after a reload it will never see.
        #[cfg(feature = "agent")]
        if tick.is_multiple_of(u64::from(self.hz.max(1))) {
            self.publish_journal();
        }
        // Before the halt check: play mode is the *editor's* clock, not the
        // sim's, and stopping a session that a panicking system halted is
        // exactly when someone wants their world back.
        if let Some(play) = &mut self.play
            && play.edge(tick, &mut self.world)?
        {
            // A stop rewinds every `Sound`'s `seq` to what it was at enter, and
            // a rewind is a difference like any other — without this, stopping
            // plays the whole session's cue bank at once (§6 M18 item 2).
            self.audio.forget();
        }
        if self.halted {
            // `Input::tick` is below this return, so nothing empties the motion
            // accumulator while a panicked system keeps the sim stopped (§6
            // M56). A halt is recoverable — a rebuild clears it — and the tick
            // that recovers must not be handed every count the mouse produced in
            // between.
            self.input.forget_motion();
            return Ok(());
        }
        // Before anything reads a knob this tick — `ui_fit` below is already one
        // such reader (§6 M40). Live: what moved is written down. Replayed: what
        // the file says moved is applied, and an unknown name stops the run.
        let moved = self.knobs.moved();
        let replayed = self.drive.knobs(tick, &moved);
        gg_core::cvar::apply(replayed.iter().map(|(_, n, v)| (n.as_str(), v.as_str())))?;
        // The surface the session opened at, once. `attach` has run by now in a
        // windowed session and a headless one never resizes, so this is the
        // extent every click below was scaled by (§6 M40, M15.1).
        if tick == 0 {
            self.drive.record_surface(self.surface());
        }
        // The cursor's own accumulator, filled before the latch below for the
        // same reason platform events are: it is this tick's input (§6 M15.1).
        // A steer emitted after the latch would arrive one tick late and the
        // arrow would trail the OS cursor by a frame.
        if let Some((dx, dy)) = self.cursor.steer(self.ui_fit()) {
            self.input.cursor(dx, dy);
        }
        // Latched here rather than at `poll_input`, because the frame is a
        // *tick's* input: `just_pressed` is an edge between two ticks, and a
        // frame that owes three of them must not report the same edge to all
        // three. Platform events accumulate in `self.input` as they arrive.
        let input = self.drive.frame(&mut self.input, tick);
        // The transport, and where the pointer is: read once by value, because
        // both decisions below want them and the second writes through `self`.
        #[cfg(feature = "editor")]
        let mouse = self.editor.as_ref().map(|e| (e.play(), e.ui.over_panels()));
        // Handed back the moment the scene stops running, which is the take's
        // own rule (`Play::takes_pointer`) read the other way rather than a
        // second edge to keep in step with it. What it closes is `--play`'s stop
        // with the mouse held: the arrow would have stayed hidden over a scene
        // nothing advances, with Escape — which a script cannot press — as the
        // only way out.
        //
        #[cfg(feature = "editor")]
        if hands_back(self.cursor.held, mouse.map(|(play, _)| play.running())) {
            self.cursor.held = false;
            info!(tick, "editor: pointer handed back — the scene is stopped");
        }
        // The editor and the game share one physical mouse (§6 M15), and while
        // the editor holds it the game gets a dead frame on *both* channels —
        // the tick's input and the UI stage's — wherever the pointer is, not
        // merely over a panel. Raw device motion arrives whatever the pointer
        // is over (it is a *device* delta, not a position) and demo 05 binds it
        // to `aim_x`; a press over a panel would otherwise also hit whatever
        // game widget sits at that canvas position, both routers integrating
        // the same stream. Read before the take below, so the press that hands
        // the pointer over is still the editor's. Recorded before this, never
        // after: a replay holds what the operator did, not what the game saw.
        let dead = self.editing() && !self.cursor.held;
        #[cfg(feature = "editor")]
        let input = match dead {
            true => InputFrame::default(),
            false => input,
        };
        // A press in a *running* viewport is how the game takes the pointer, and
        // it is read off the *recorded* click rather than off a window event —
        // so a replayed session enters and leaves mouse-look exactly where the
        // operator did, with no window anywhere (§6 M15.1). Escape hands it back
        // too; `play.rs` owns that edge, Escape not being a verb.
        #[cfg(feature = "editor")]
        if !self.cursor.held
            && let Some(binding) = self.ui_binding
            && self.input.just_pressed(binding.primary)
            && mouse.is_some_and(|(play, panels)| play.takes_pointer(panels))
        {
            self.cursor.held = true;
            info!(tick, "editor: pointer taken by the game");
        }
        // Arbitration (§6 M45), at *delivery* and after the recorder: the game
        // said last tick that a modal screen is up, which is the tick that
        // screen reaches the glass on, so what is withheld and what is visible
        // move together. `previous` takes the withheld frame too — an edge
        // against the raw one would fire the moment a menu closed.
        let input = match self.modal() {
            true => self.input.keep().apply(input),
            false => input,
        };
        let ctx = TickCtx {
            tick,
            tick_hz: self.hz,
            reserved: 0,
            input,
            previous: self.previous,
            bindings: self.input.spellings().as_ptr(),
            bindings_len: self.input.spellings().len() as u32,
            reserved2: 0,
        };
        self.previous = input;
        // Pause is the editor's, and it stops exactly the sim: the tick still
        // happens, input is still recorded, and the panels still route clicks —
        // which is what makes a paused editor usable at all.
        #[cfg(feature = "editor")]
        let running = self.editor.as_mut().is_none_or(Editing::advance);
        #[cfg(not(feature = "editor"))]
        let running = true;
        // SAFETY: `self.lib` is verified and loaded, and `ctx` outlives the call.
        if running
            && let Err(panicked) = unsafe { self.world.run_systems(self.lib.systems(), &ctx) }
        {
            // Halt, do not exit: the sim stops cleanly, the process survives,
            // and a fixed reload resumes it (§4.2.2).
            error!(
                system = %panicked.system,
                message = %panicked.message,
                tick,
                "system panicked — sim halted until the next reload"
            );
            self.halted = true;
        }
        // §4.7, and *before* the hash on purpose: a derived world transform is
        // hashed state, so the three-way gate covers the compose and not merely
        // the locals feeding it. A refused hierarchy halts like a panicking
        // system — same reason, and a reload is what clears it.
        if running
            && !self.halted
            && let Err(refused) = self.hierarchy.propagate(&mut self.world)
        {
            error!(error = %refused, tick, "hierarchy refused — sim halted until the next reload");
            self.halted = true;
        }
        // The player's own settings (§6 M42), after the systems of this
        // session's *first* tick — which is the only moment they can land: a
        // game spawns its `Prefs` in its bootstrap, so before it there is no
        // component to write, the one way this differs from a scene, which
        // arrives as the whole world. Before the UI tick below, so the stage
        // that caches preferences reads them on the tick they arrive.
        //
        // The first tick, not tick 0 (§6 M44): a resumed session opens at the
        // tick it stopped on, and a condition written `tick == 0` silently
        // stopped reading the settings file the moment a session could carry a
        // tick — a player whose saved game ignored the volume slider.
        //
        // The file is the *complete* statement of what the player asked for, so
        // it is assigned rather than merged: a key they deleted is a preference
        // they gave back. The fields that are **not** preferences survive it —
        // `close` is the quit button's edge and `modal` is which screen is up
        // (§6 M45), and both belong to this session rather than to the file.
        // `settings::KEYS` refuses to persist either, and this is the same rule
        // read from the other side: a file that carried them would end every
        // session that opened one, or open one with its controls withheld.
        // `--keys` is what found the second: it went red because a run with a
        // settings file zeroed the title screen's `modal` on its first tick.
        if !std::mem::replace(&mut self.opened, true)
            && let Some(want) = self.settings.take()
        {
            let mut applied = 0usize;
            self.world
                .each(&gg_ecs::Query::<&mut Prefs>::new()?, |_, p: &mut Prefs| {
                    *p = Prefs {
                        close: p.close,
                        modal: p.modal,
                        ..want
                    };
                    applied += 1;
                });
            info!(applied, "settings applied");
        }
        // The UI tick (§4.9). *Before* the hash and after the systems that
        // declared it, so a click lands in `Widget::state` inside the tick that
        // took it and the game reads it on the next one. A halt stops it with
        // everything else — the early return above is the sim's clock, and a UI
        // that kept ticking would route clicks against a world nothing is
        // advancing.
        //
        // The fit moves the picture and never the hit test: the pointer is
        // integrated in canvas units precisely so a headless tick and a 4K one
        // resolve the same widget (`gg_ui::boundary`'s docs), which is what
        // lets the determinism gate cover clicks at all — and what lets the
        // editor's pane carry the picture without touching a replay.
        let ui_tick = self
            .ui_binding
            .map(|binding| UiTick::from_input(&self.input, &binding))
            .unwrap_or_default();
        // Dead on the same terms as `TickCtx::input` above, and only for the
        // *game's* stage: `ui_tick` still reaches the editor's own router below,
        // which is the one the operator is pointing at.
        let game_ui = if dead { UiTick::default() } else { ui_tick };
        let fit = self.game_fit();
        self.ui.frame(&mut self.world, &game_ui, fit);
        // Mouse-look and a menu are one physical mouse (§4.9). A game that binds
        // raw device motion holds it — that binding is what tells a camera from
        // a cursor — and gives it back for exactly as long as its UI has
        // something to point at. Both halves are world state, so a replayed
        // session changes hands on the same tick with no window anywhere (§4.7),
        // and a game with no menu never reaches the second half at all. Left to
        // the editor's own take/hand-back rules while it is hosting.
        if self.looks && !self.editing() {
            let held = !self.ui.wants_pointer();
            if self.cursor.held && !held {
                // Where `Window::set_pointer` warped it when the game took it,
                // which is the only thing that knows where a locked cursor is:
                // without this the software arrow spends its first frames in
                // the top-left corner and jumps under the OS one on the first
                // motion event.
                let (w, h) = self.surface();
                self.cursor.at = Some((w as f32 / 2.0, h as f32 / 2.0));
            }
            self.cursor.held = held;
        }
        // The audio tick (§6 M18 item 2). After the UI so a cue a game fires in
        // response to a click lands in the same tick as the click, and *outside*
        // the hash entirely — it takes `&World` and the compiler is what proves
        // it. A halt stops it with everything else: the early return above is
        // the sim's clock, and a world nothing is advancing has no new cues.
        self.audio.tick(&self.world);
        // The editor tick (§6 M15), after the game's UI and before the hash for
        // the same reason: an inspector edit is ordinary world state and belongs
        // in the tick that took the click.
        #[cfg(feature = "editor")]
        if let Some(path) = self.editor_tick(tick, &ui_tick, self.editor_surface())? {
            self.write_save(&path)?;
        }
        // Immediately after, because the camera decided it in there and the
        // steer above reads it at the top of the *next* tick (§6 M63).
        #[cfg(feature = "editor")]
        self.note_flying();
        // §1.13 hazard 6's per-tick call site. The canonical hash absorbs raw
        // bits and NaN payloads differ by architecture, so a NaN in hashed state
        // is a §5.6 divergence with a math bug as its cause — banned, not
        // canonicalized around. It halts like a panicking system, for the same
        // reason: the rest of the tick would run over state already corrupt.
        #[cfg(all(feature = "nan-scan", debug_assertions))]
        if !self.halted
            && let Some(hit) = self.world.scan_for_nan()
        {
            error!(%hit, tick, "NaN in hashed state — sim halted until the next reload");
            self.halted = true;
        }
        // §5.6c's material: one canonical hash per tick, on a target of its own
        // so a human's terminal never sees 60 of these a second. Emitted rather
        // than accumulated — the shell keeps no determinism ledger, and a run's
        // stdout is a thing every tier already captures.
        #[cfg(feature = "state-hash")]
        tracing::debug!(target: "gg::hash", tick, hash = %self.world.canonical_hash());
        // The session, while it is still one (§6 M48). Last in the tick, so what
        // crosses is the world every stage above has finished with — the same
        // bytes the exit write would produce, which is what makes a resumed
        // checkpoint indistinguishable from a resumed exit.
        //
        // Halted is skipped rather than written: a halt froze the world, so the
        // interval would spend an encode to produce the file already on disk.
        // Not while the editor hosts, either — that session's target is the save
        // button's, and `run` is what withholds the path.
        if !self.halted
            && tick != 0
            && tick.is_multiple_of(u64::from(self.hz.max(1)) * crate::player::CHECKPOINT_SECONDS)
        {
            if let Some(checkpoint) = &self.checkpoint {
                checkpoint.offer(
                    Save::new(self.world.snapshot(), self.next_tick, self.lib.code_hash()).encode(),
                );
            }
            // Read off the same cache the exit write reads, so the two produce
            // the same bytes from the same tick and a resumed session cannot
            // tell which one it got.
            if let Some(checkpoint) = &self.prefs_checkpoint {
                checkpoint.offer(gg_ecs::boundary::settings::encode(&self.prefs()).into_bytes());
            }
        }
        Ok(())
    }

    /// `alpha` is how far past the last tick this frame landed (§4.1), and what
    /// it buys is that a 60 Hz sim does not present each pose for two refreshes
    /// and then three. `sim_tick` captured the far side of the interval; this
    /// end of it is the world as it stands. Zero under a locked pace, so a
    /// replay, a golden run and every headless tier extract the tick exactly.
    fn extract(&mut self, alpha: f32) -> anyhow::Result<()> {
        // Restated every frame rather than on `Resized`: the pane's pixels move
        // with the window *and* with a seam the operator drags, and one
        // assignment ahead of the two stages that read it is cheaper than an
        // event path that could be missed. The editor is the only thing that
        // ever sets it — a plain run renders into the whole window, which is
        // what `None` means.
        #[cfg(feature = "editor")]
        if let (Some(editing), Some(renderer)) = (self.editor.as_ref(), self.gpu.as_mut()) {
            renderer.set_viewport(Some(editing.ui.viewport_rect()));
        }
        // The player's ask beats `r.vsync`, and only where they made one — the
        // `aa` rule below, one knob over (§6 M46). Restated every frame for the
        // viewport's reason and at the viewport's price: the renderer compares
        // it and recreates nothing on the frames it did not move.
        let present = self.ui.prefs().present_mode();
        if let Some(renderer) = self.gpu.as_mut() {
            renderer.set_present(present);
        }
        let Some(renderer) = &self.gpu else {
            return Ok(());
        };
        self.extracted.interpolate(alpha);
        // Blended before the editor's substitution below, never after: what the
        // two ticks bracket is the *game's* eye, and the editor's camera is host
        // state moved by a tick of its own.
        let eye = self.extracted.eye(Eye::of(&self.world)?, self.latch()?);
        // The editor's own camera while the scene is stopped (§6 M15.2 item 2):
        // host state, in no archetype and in no save, so what the operator flies
        // moves nothing the canonical hash can see. `game` back in every other
        // state, which is why this is unconditional.
        //
        // Blended and latched here rather than substituted whole (§6 M63): the
        // editor's camera steps once a tick like the game's, so on a 240 Hz
        // panel it presented each of its poses for four refreshes and turned in
        // visible stairs. Both corrections are §4.1's and §6 M56's, applied to a
        // second eye — `blend_eye` for where it *was*, the latch for the counts
        // the hand has produced since the tick that moved it. Neither is written
        // back anywhere; the picture leads the tick and the tick is unaware.
        #[cfg(feature = "editor")]
        let eye = self.editor_eye(eye, alpha);
        // Rebuilt, not mutated: `View::default()` is where the render CVars are
        // read, so a console edit lands on the next frame (§4.8).
        self.view = View {
            yaw: eye.yaw,
            pitch: eye.pitch,
            // The projection is the eye's (§6 M20): zero is perspective, and an
            // editor camera latched from an ortho game carries the game's flat
            // view with it — which is what makes authoring in-plane.
            ortho: eye.ortho,
            // The player's ask beats the host's `r.aa`/`r.msaa`, and only where
            // they made one: both are `None` for a game that declares no video
            // menu, which is every game before this one (§6 M21). The count is
            // still a *request* — the renderer cuts it to what the device does
            // and logs the cut.
            aa: self.ui.prefs().antialias().unwrap_or(View::default().aa),
            samples: self
                .ui
                .prefs()
                .samples()
                .and_then(gg_render::Samples::from_count)
                .unwrap_or(View::default().samples),
            ..View::default()
        };
        // The frustum crosses from the renderer, which owns the projection, to
        // extract, which owns the narrowing. Building it here would put §2's
        // reverse-Z convention in the shell.
        //
        // `view_extent` and not `extent`: with the editor open the picture is
        // composed for the viewport panel, and a frustum built from the whole
        // window would cull what the panel can still see.
        let extent = renderer.view_extent();
        self.extracted
            .clear(eye.position, self.view.frustum(extent));
        // The lights the game declared (§6 M11), and *before* the instances now
        // rather than after: the sun is what the caster sweep below is aimed
        // along, and a directional light is culled by nothing, so there is
        // nothing it needs settled first. A game with no lights renders unlit —
        // dim and obviously so, which is the same rule a missing `Eye` follows.
        self.extracted.append_lights(&self.world)?;
        // A caster does not have to be on screen. An object just past the edge
        // still lays its shadow across what is, and culling casters by the view
        // is a shadow that vanishes while its own is still in frame — widened
        // up-light by exactly the depth the cascades record (§6 M11).
        self.extracted.cast_shadows(self.view.caster_reach(extent));
        self.extracted.append::<Renderable>(&self.world)?;
        // Pack content, expanded through whatever the renderer has mapped. A
        // run with no `--pack` expands nothing and this is a query over an
        // empty archetype (§4.6).
        self.extracted
            .append_models::<Model>(&self.world, renderer.scenes())?;
        Ok(())
    }

    fn render(&mut self, _alpha: f32) -> anyhow::Result<()> {
        let Some(renderer) = &mut self.gpu else {
            return Ok(());
        };
        // The editor sets its text in a rented face, so the atlas the game's UI
        // was given at attach is not the one the editor's glyphs are cut from
        // (§4.9). Uploaded on a change and never otherwise — which after the
        // warm-up means resizes across a scale boundary, and nothing else.
        #[cfg(feature = "editor")]
        if let Some(editing) = &self.editor
            && self.ui_atlas_rev != editing.ui.font_revision()
        {
            self.ui_atlas_rev = editing.ui.font_revision();
            renderer.set_ui_atlas(&editing.ui.coverage())?;
        }
        // Game UI first, instruments over the top — the overlay is a lab
        // instrument and must never be the thing a click lands under. Copied
        // into one buffer because the renderer takes one slice; cleared and
        // refilled, so this allocates once and then never (§6 M13).
        self.ui_geometry.clear();
        self.ui_geometry.extend_from_slice(self.ui.vertices());
        // Over the game and under the instruments: the editor is what the
        // operator clicks, the overlay is a readout and must never be the thing
        // a click lands beneath.
        #[cfg(feature = "editor")]
        if let Some(editing) = &self.editor {
            self.ui_geometry.extend_from_slice(editing.ui.vertices());
        }
        // The overlay reads the readings the frame already took rather than
        // asking the device again, so its rows and Tracy's zones cannot
        // disagree about one frame (§4.8).
        #[cfg(feature = "overlay")]
        self.ui_geometry
            .extend_from_slice(self.overlay.build(&gg_debug::overlay::Stats {
                extent: renderer.extent(),
                tick: self.next_tick,
                passes: renderer.pass_timings(),
                memory: renderer.memory(),
                luminance: renderer.luminance(),
            }));
        // Dropped at the end of this method, which is before the device is and
        // after the only submit — what RenderDoc records is exactly one frame.
        #[cfg(feature = "debug-tools")]
        let _capture = gg_debug::capture::frame();
        renderer.frame(&self.extracted, &self.view, CLEAR, &self.ui_geometry)?;
        // A frame or two behind, which is what not stalling costs; a zone is
        // placed by its GPU timestamps, not by when it was sent (§4.8).
        #[cfg(feature = "debug-tools")]
        if let Some(zones) = &mut self.zones {
            zones.frame(renderer.pass_timings(), renderer.gpu_clock());
        }
        Ok(())
    }
}
