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
use gg_ecs::boundary::{Eye, Light, Model, Renderable, TickCtx, Widget, host_api};
use gg_ecs::{ComponentOutcome, Save, Snapshot, World};
use gg_extract::Extracted;
use gg_input::{ActionMap, Drive, Input, InputFrame, Recorder, Replay, ReplayMeta};
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
    input: Input,
    drive: Drive,
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
    /// Tracy's GPU zones, fed the same readings the overlay shows, so the two
    /// views of one frame cannot disagree (§4.8).
    #[cfg(feature = "debug-tools")]
    zones: Option<gg_debug::GpuZones>,
    /// The overlay, and the UI vertices it built this frame (§4.8). Absent from
    /// dist entirely — `gg-debug` is not in that graph (§3).
    #[cfg(feature = "overlay")]
    overlay: gg_debug::Overlay,
}

impl App {
    /// Load `game`, register what it declares, and start a world over it. In dev
    /// load number one already goes through the staging copy: loading
    /// `target/debug/game.dll` in place would make the next `cargo build` fail
    /// rather than the reload (§4.2.2).
    pub fn new(
        args: &crate::Args,
        staging: &Path,
        hz: u32,
        bindings: String,
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
        let input = bind(&format!("{bindings}{extra}"), &drive, verbs)?;
        let ui_binding = binding(&verbs);
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
        let editing = args.editor.then(|| Editing::new(args, &lib));
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
            input,
            drive,
            next_tick: 0,
            previous: InputFrame::default(),
            hierarchy: Hierarchy::new(),
            extracted: Extracted::default(),
            view: View::default(),
            ui: gg_ui::Ui::new()?,
            audio: gg_audio::Audio::device_unless_headless()?,
            ui_binding,
            cursor: Cursor::new(args.editor),
            #[cfg(feature = "editor")]
            ui_atlas_rev: 0,
            ui_geometry: Vec::new(),
            play: args.play.as_deref().map(PlayMode::parse).transpose()?,
            #[cfg(feature = "editor")]
            editor: editing,
            gpu: None,
            dpi: 1.0,
            pack: args.pack.clone(),
            #[cfg(feature = "debug-tools")]
            zones: None,
            #[cfg(feature = "overlay")]
            overlay: gg_debug::Overlay::default(),
        })
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
        let mut renderer = Renderer::new(window, window.inner_size())?;
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

    /// Whether the game should hold and hide the pointer this frame. Applied by
    /// the windowed loop, which is the only place a window exists.
    pub fn pointer_held(&self) -> bool {
        self.cursor.held
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
        if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, &bytes)?;
        info!(
            path = %path.display(),
            tick = save.tick(),
            entities = save.snapshot().entity_count(),
            bytes = bytes.len(),
            "save written"
        );
        Ok(())
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
        // The editor opens playing, so the first tick is the play edge nobody
        // clicked. Here rather than in `Editing::new` because that is where the
        // world first exists to capture.
        if !core::mem::replace(&mut editing.opened, true) {
            editing.stash.enter(tick, &self.world);
        }
        let commands = editing.ui.tick(
            &mut self.world,
            ui,
            &gg_editor::Frame {
                extent,
                dpi: self.dpi,
                tick,
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
                // Never: a windowed session has the system cursor on the same
                // pixel, and a headless one has no frame to draw into (§6
                // M15.1). What draws its own is `gg-golden`.
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

    /// Adopt a rebuilt dylib: snapshot, register the new schemas into a *fresh*
    /// world, restore through the migration path (§4.2.2), then swap. Fresh
    /// rather than re-registered, because a component whose schema moved needs
    /// its column rebuilt, and rebuilding under live rows is what
    /// snapshot/restore exists to avoid.
    #[cfg(feature = "hot-reload")]
    fn swap(
        &mut self,
        reloaded: gg_core::reload::watch::Reloaded,
    ) -> anyhow::Result<gg_ecs::MigrationReport> {
        let snapshot = self.world.snapshot();
        let mut world = World::new();
        // Every failure from here to the swap is charged to the leak budget: the
        // image is mapped and never unloaded, so a refused adopt costs exactly
        // what an accepted one does (§4.2.2).
        adopt(&mut world, &reloaded.lib).map_err(|e| reloaded.lib.refuse(e))?;
        let report = world
            .restore(&snapshot)
            .map_err(|e| reloaded.lib.refuse(e))?;
        // One line per component that actually moved (§4.2.2). The reused ones
        // are the majority of every reload, and printing them would bury the one
        // line the reader came for.
        for (declared, outcome) in &report.components {
            if !matches!(outcome, ComponentOutcome::Reused) {
                info!(component = %declared, ?outcome, "migrated");
            }
        }
        // Re-parsed rather than carried: an edit that appends an action moves
        // the id space, and a map bound against the old one would fire the wrong
        // verb (§4.7). Key state resets with it, which is why this is the one
        // reload effect a player can feel.
        // SAFETY: `reloaded.lib` is verified and never unloaded.
        let (verbs, extra) = unsafe { verbs_for(&reloaded.lib, self.editing()) };
        self.input = bind(&format!("{}{extra}", self.bindings), &self.drive, verbs)
            .map_err(|e| reloaded.lib.refuse(e))?;
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
        self.world = world;
        self.halted = false;
        // The world on the other side of a migration is one the audio observer
        // has not seen: a `Sound` the new build declares afresh comes back with
        // `seq` at zero, and a difference is a trigger. Forgotten rather than
        // carried, so a reload is silent at the seam instead of playing whatever
        // the retired build's cue bank happened to end on (§6 M18 item 2).
        self.audio.forget();
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

/// The tier this shell was built as. An agent reading a record needs it: half
/// the advice worth giving names machinery that a dist build does not have.
#[cfg(feature = "agent")]
const TIER: &str = if cfg!(feature = "tracy") {
    "tier-instrumented"
} else {
    "tier-dev"
};

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
    let mut journal = gg_agent::Journal::new(game, TIER);
    // The third thing the record hands an agent: what the human just did, as
    // something replayable rather than described (§4.7).
    if let Some(path) = &args.record {
        journal.recording(path);
    }
    journal
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
}

impl Cursor {
    /// Held from the start unless an editor is opening: mouse-look wants the
    /// pointer inside the window before the first frame, and every demo has
    /// always started that way.
    fn new(editor: bool) -> Cursor {
        Cursor {
            at: None,
            steered: (0, 0),
            held: !editor,
        }
    }

    /// The cursor delta to feed this tick, if any.
    ///
    /// `None` while the game holds the pointer — the arrow is locked and hidden
    /// and the editor's pointer parks where it was — and `None` before the OS
    /// has reported a position at all.
    fn steer(&mut self, fit: gg_ui::Fit) -> Option<(i32, i32)> {
        let (x, y) = self.at.filter(|_| !self.held)?;
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
    /// The sim is not advancing. Starts **false**: an editor that opened paused
    /// would show a world whose bootstrap system had never run.
    paused: bool,
    /// The world captured at the play edge (§6 M15.2). Empty is `Stopped`, and
    /// `Stopped` is the only state an inspector edit survives.
    stash: Stash,
    /// False until the first tick has captured. The editor opens *playing* for
    /// `paused`'s reason, one level further out — and a capture needs a world,
    /// which exists at the *end* of the first tick and not at construction.
    /// Read by [`advance`](Editing::advance) too: the tick that fills the stash
    /// is the one tick that must run without it.
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
    fn new(args: &crate::Args, lib: &GameLib) -> Editing {
        use gg_editor::persist;
        let stem = &lib.name();
        let save = args
            .save
            .clone()
            .unwrap_or_else(|| persist::save_path(stem));
        // Not while recording or replaying. The layout *is* hit-testing (§6
        // M15.1), so a session that started from a file the gate never saw
        // would land its clicks somewhere else.
        let layout =
            (args.record.is_none() && args.replay.is_none()).then(|| persist::layout_path(stem));
        let mut ui = gg_editor::Editor::new(args.pack.as_deref());
        if let Some(path) = &layout {
            persist::restore(&mut ui, path);
        }
        Editing {
            // The working directory, for `gg.cfg`'s reason: a launcher is
            // started *in* a workspace, and a flag pointing elsewhere would be a
            // second answer to where a project is. Scanned in every editor
            // session and drawn in none that has a game — the picker's condition
            // is `Frame::project`, not this list being empty.
            projects: gg_editor::project::scan(Path::new(".")),
            open: None,
            ui,
            paused: false,
            stash: Stash::default(),
            opened: false,
            step: false,
            save,
            extent: args.editor_extent,
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
    /// The `!opened` arm is the first tick, which advances with nothing
    /// captured yet — the capture is taken at the *end* of it, in `editor_tick`,
    /// because the world a stop returns to is the bootstrapped one and bootstrap
    /// has not run before the first tick. Without it the editor opens playing in
    /// name only: tick 0 runs no systems, the play edge captures the empty
    /// pre-bootstrap world, and the first stop hands that back (§6 M15.2).
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
fn bind(bindings: &str, drive: &Drive, verbs: gg_ecs::boundary::Verbs) -> anyhow::Result<Input> {
    if let Some(replay) = drive.replay() {
        replay.check_verbs(verbs.actions, verbs.axes)?;
    }
    let mut input = Input::new(ActionMap::parse(bindings, verbs.actions, verbs.axes)?);
    if !input.push_named(CONTEXT) && !bindings.trim().is_empty() {
        warn!("the bindings declare no `{CONTEXT}` context — nothing is bound");
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
        self.rejuvenate.pending()
    }

    fn sim_tick(&mut self, tick: u64) -> anyhow::Result<()> {
        // Hazard 5's per-tick call site (§4.2.1, §2's Build-tiers row). An ICD,
        // layer or audio DLL initializer that sets FTZ/DAZ changes every denormal
        // result process-wide and only on the host that loaded it — the two
        // native legs of §5.6 then diverge with nothing naming the cause.
        #[cfg(all(feature = "fp-assert", debug_assertions))]
        gg_math::fpenv::assert_fp_env();
        self.next_tick = tick + 1;
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
            return Ok(());
        }
        // The cursor's own accumulator, filled before the latch below for the
        // same reason platform events are: it is this tick's input (§6 M15.1).
        // A steer emitted after the latch would arrive one tick late and the
        // arrow would trail the OS cursor by a frame.
        let surface = self.surface();
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
        #[cfg(feature = "editor")]
        if self.cursor.held && mouse.is_none_or(|(play, _)| !play.running()) {
            self.cursor.held = false;
            info!(tick, "editor: pointer handed back — the scene is stopped");
        }
        // The editor and the game share one physical mouse (§6 M15), and while
        // the editor holds it the game gets a dead frame — wherever the pointer
        // is, not merely over a panel. Hovering the viewport is not playing:
        // raw device motion arrives whatever the pointer is over (it is a
        // *device* delta, not a position), demo 05 binds it to `aim_x`, and a
        // pointer crossing the pane on its way between two panels used to pan
        // the camera as it went. Recorded before this, never after: what a
        // replay holds is what the operator did, not what the game saw.
        #[cfg(feature = "editor")]
        let input = match self.editor.is_some() && !self.cursor.held {
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
        let ctx = TickCtx {
            tick,
            tick_hz: self.hz,
            reserved: 0,
            input,
            previous: self.previous,
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
        // The UI tick (§4.9). *Before* the hash and after the systems that
        // declared it, so a click lands in `Widget::state` inside the tick that
        // took it and the game reads it on the next one. A halt stops it with
        // everything else — the early return above is the sim's clock, and a UI
        // that kept ticking would route clicks against a world nothing is
        // advancing.
        //
        // The extent moves the picture and never the hit test: the pointer is
        // integrated in canvas units precisely so a headless tick and a 4K one
        // resolve the same widget (`gg_ui::boundary`'s docs), which is what
        // lets the determinism gate cover clicks at all.
        let ui_tick = self
            .ui_binding
            .map(|binding| UiTick::from_input(&self.input, &binding))
            .unwrap_or_default();
        self.ui.frame(&mut self.world, &ui_tick, surface);
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
        Ok(())
    }

    /// `alpha` is unread: interpolating would need last tick's transforms kept
    /// beside this tick's, and the render protocol (§4.5 v0) carries one pose
    /// per entity. The loop hands it over anyway so the day a second buffer
    /// exists, the signature does not move.
    fn extract(&mut self, _alpha: f32) -> anyhow::Result<()> {
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
        let Some(renderer) = &self.gpu else {
            return Ok(());
        };
        let eye = Eye::of(&self.world)?;
        // The editor's own camera while the scene is stopped (§6 M15.2 item 2):
        // host state, in no archetype and in no save, so what the operator flies
        // moves nothing the canonical hash can see. `game` back in every other
        // state, which is why this is unconditional.
        #[cfg(feature = "editor")]
        let eye = self.editor.as_ref().map_or(eye, |e| e.ui.eye(eye));
        // Rebuilt, not mutated: `View::default()` is where the render CVars are
        // read, so a console edit lands on the next frame (§4.8).
        self.view = View {
            yaw: eye.yaw,
            pitch: eye.pitch,
            ..View::default()
        };
        // The frustum crosses from the renderer, which owns the projection, to
        // extract, which owns the narrowing. Building it here would put §2's
        // reverse-Z convention in the shell.
        //
        // `view_extent` and not `extent`: with the editor open the picture is
        // composed for the viewport panel, and a frustum built from the whole
        // window would cull what the panel can still see.
        let frustum = self.view.frustum(renderer.view_extent());
        self.extracted
            .transforms::<Renderable>(&self.world, eye.position, frustum)?;
        // Pack content, expanded through whatever the renderer has mapped. A
        // run with no `--pack` expands nothing and this is a query over an
        // empty archetype (§4.6).
        self.extracted
            .append_models::<Model>(&self.world, renderer.scenes())?;
        // The lights the game declared (§6 M11). A game with none renders unlit
        // — dim and obviously so, which is the same rule a missing `Eye` follows.
        self.extracted.append_lights(&self.world)?;
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
