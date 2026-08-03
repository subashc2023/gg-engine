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
    hz: u32,
    /// A panicking system halts the sim and leaves the process running
    /// (§4.2.2); a reload is what clears this, which is the "agent broke it,
    /// agent fixes it, nobody restarts" loop.
    halted: bool,
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
    /// The window's GPU state, absent in a headless run — and that absence is
    /// the extract and render stages' off switch.
    gpu: Option<Renderer>,
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
        #[cfg(feature = "hot-reload")]
        let (watch, lib) = {
            let mut watch = gg_core::reload::watch::Watch::new(game, staging)?;
            // SAFETY: `game` is the artifact the operator named — the only
            // provenance any host can establish (§4.2.2) — and `host_api()` is
            // `&'static`. How long to wait and how hard is the watcher's, not
            // the shell's (§3).
            let lib = unsafe { watch.block_until_ready(host_api()) }?.lib;
            (watch, lib)
        };
        // SAFETY: `game` is the artifact the operator named, which is the only
        // provenance any host can establish (§4.2.2); `host_api()` is `&'static`.
        #[cfg(not(feature = "hot-reload"))]
        let lib = unsafe { GameLib::load(game, host_api())? };

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
        Ok(Self {
            world,
            lib,
            rejuvenate: Rejuvenator::new(args.leak_budget),
            hz,
            halted: false,
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
            ui_binding,
            ui_geometry: Vec::new(),
            play: args.play.as_deref().map(PlayMode::parse).transpose()?,
            #[cfg(feature = "editor")]
            editor: editing,
            gpu: None,
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
        let mut renderer = Renderer::new(window, window.inner_size())?;
        // The renderer never learns what a glyph is; it takes coverage texels
        // and a rectangle (§4.9). Unconditional since M13: the game's own UI
        // draws from this atlas in every tier, the overlay is a second caller.
        renderer.set_ui_atlas(&gg_ui::atlas::fallback())?;
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
        if let Some(renderer) = &mut self.gpu {
            renderer.resize(width, height);
        }
    }

    /// The live input state, for `gg_platform::feed` to apply raw events to.
    /// Escape and the close button never get here: quitting is not simulated
    /// state and must work identically while a replay is driving.
    pub fn input(&mut self) -> &mut Input {
        &mut self.input
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

    /// The recording, once the loop is over.
    pub fn finish(self) -> Option<Box<Recorder>> {
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
    ) -> Option<std::path::PathBuf> {
        let editing = self.editor.as_mut()?;
        let path = editing.save.display().to_string();
        let commands = editing.ui.tick(
            &mut self.world,
            ui,
            &gg_editor::Frame {
                extent,
                tick,
                playing: !editing.paused,
                passes: self.gpu.as_ref().map_or(&[][..], Renderer::pass_timings),
                memory: self.gpu.as_ref().map(Renderer::memory).unwrap_or_default(),
                save_path: &path,
            },
        );
        if let Some(playing) = commands.playing {
            editing.paused = !playing;
            info!(
                tick,
                playing,
                entities = self.world.len(),
                "editor: play state"
            );
        }
        editing.step = commands.step;
        commands.save.then(|| editing.save.clone())
    }

    /// Whether the editor is open, asked from a path that exists without it.
    #[cfg(feature = "hot-reload")]
    fn editing(&self) -> bool {
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
    fn swap(&mut self, reloaded: gg_core::reload::watch::Reloaded) -> anyhow::Result<()> {
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
        Ok(())
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
    /// Advance one tick, then pause again. Consumed by [`Editing::advance`].
    step: bool,
    /// Where the save button writes — `--save` if given, so a gate can name it.
    save: std::path::PathBuf,
}

#[cfg(feature = "editor")]
impl Editing {
    fn new(args: &crate::Args, lib: &GameLib) -> Editing {
        let save = args.save.clone().unwrap_or_else(|| {
            let stem = lib.path().file_stem().unwrap_or_default().to_string_lossy();
            std::path::Path::new("target/editor").join(format!("{stem}.ggsv"))
        });
        Editing {
            ui: gg_editor::Editor::new(args.pack.as_deref()),
            paused: false,
            step: false,
            save,
        }
    }

    /// Whether the sim advances this tick, spending a single step if one is due.
    fn advance(&mut self) -> bool {
        !self.paused || core::mem::take(&mut self.step)
    }
}

/// The editor's play/stop on a script, so a tier can run it (§6 M14).
///
/// M15 drives these two edges from a button; a `<enter>:<stop>` spec drives them
/// from a tick number, and both reach the same two calls. It reports rather than
/// asserts: a gate that lived inside the program under test would be graded by
/// the thing it grades, so the shell states what happened and `xtask` decides.
struct PlayMode {
    enter: u64,
    stop: u64,
    /// The captured world, kept *encoded*. Comparing bytes at stop then proves
    /// the round trip and the equality at once — a `Snapshot` held by value
    /// would prove neither, because nothing would have serialized it.
    stash: Option<Vec<u8>>,
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
            stash: None,
        })
    }

    /// Called at the top of every tick, so the world compared at `stop` is the
    /// world captured at `enter` — the same point in the frame, with nothing
    /// half-advanced between the two readings.
    fn edge(&mut self, tick: u64, world: &mut World) -> anyhow::Result<()> {
        if tick == self.enter {
            let stash = world.snapshot().encode();
            info!(
                tick,
                entities = world.len(),
                bytes = stash.len(),
                "play mode entered"
            );
            self.stash = Some(stash);
        } else if tick == self.stop
            && let Some(stash) = self.stash.take()
        {
            // Read before the restore as well as after. `identical` over a
            // session that changed nothing is a gate that cannot fail, so the
            // *other* half of the claim is reported beside it and the caller
            // requires both.
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
        }
        Ok(())
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
        let outcome = ready
            .map_err(anyhow::Error::from)
            .and_then(|r| self.swap(r));
        match outcome {
            Ok(()) => Ok(()),
            Err(refused) => {
                // A refused dylib was mapped and ran its initializers before any
                // check could, and is never unloaded (§4.2.2) — so its bytes are
                // this session's whether or not the swap happened. Charged, not
                // acted on: the restart still waits for a *successful* swap,
                // because rejuvenating into the same refusal only repeats it.
                if let Some(reload) = refused.downcast_ref::<gg_core::ReloadError>() {
                    self.rejuvenate.retire(reload.leaked_bytes());
                }
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
        // Before the halt check: play mode is the *editor's* clock, not the
        // sim's, and stopping a session that a panicking system halted is
        // exactly when someone wants their world back.
        if let Some(play) = &mut self.play {
            play.edge(tick, &mut self.world)?;
        }
        if self.halted {
            return Ok(());
        }
        // Latched here rather than at `poll_input`, because the frame is a
        // *tick's* input: `just_pressed` is an edge between two ticks, and a
        // frame that owes three of them must not report the same edge to all
        // three. Platform events accumulate in `self.input` as they arrive.
        let input = self.drive.frame(&mut self.input, tick);
        // The editor and the game share one physical mouse (§6 M15). While the
        // pointer is over a panel the game gets a dead frame, or a click on
        // `pause` would also fire whatever the game bound to that button — and
        // the reading is the *previous* tick's pointer, which is the same frame
        // of lag every hit test already has. Recorded before this, never after:
        // what a replay holds is what the operator did, not what the game saw.
        #[cfg(feature = "editor")]
        let input = match self.editor.as_ref().is_some_and(|e| e.ui.over_panels()) {
            true => InputFrame::default(),
            false => input,
        };
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
        let extent = self
            .gpu
            .as_ref()
            .map_or(gg_ecs::boundary::CANVAS, Renderer::extent);
        self.ui.frame(&mut self.world, &ui_tick, extent);
        // The editor tick (§6 M15), after the game's UI and before the hash for
        // the same reason: an inspector edit is ordinary world state and belongs
        // in the tick that took the click.
        #[cfg(feature = "editor")]
        if let Some(path) = self.editor_tick(tick, &ui_tick, extent) {
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
        let Some(renderer) = &self.gpu else {
            return Ok(());
        };
        let eye = Eye::of(&self.world)?;
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
        let frustum = self.view.frustum(renderer.extent());
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
