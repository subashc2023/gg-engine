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
#[cfg(feature = "hot-reload")]
use gg_ecs::ComponentOutcome;
use gg_ecs::boundary::{Eye, Model, Renderable, TickCtx, host_api};
use gg_ecs::{Snapshot, World};
use gg_extract::Extracted;
use gg_input::{ActionMap, Drive, Input, InputFrame, Recorder, Replay, ReplayMeta};
use gg_platform::Window;
use gg_render::{Renderer, View};
use gg_scene::Hierarchy;
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
            watch.request();
            let lib = loop {
                // Bounded by the watcher's own staging attempts, so a missing or
                // permanently locked artifact ends as a named error, not a spin.
                //
                // SAFETY: `game` is the artifact the operator named — the only
                // provenance any host can establish (§4.2.2) — and `host_api()`
                // is `&'static`.
                if let Some(ready) = unsafe { watch.poll(host_api()) } {
                    break ready?.lib;
                }
            };
            (watch, lib)
        };
        // SAFETY: `game` is the artifact the operator named, which is the only
        // provenance any host can establish (§4.2.2); `host_api()` is `&'static`.
        #[cfg(not(feature = "hot-reload"))]
        let lib = unsafe { GameLib::load(game, host_api())? };

        let mut world = World::new();
        // SAFETY: `lib` is verified — `GameLib::load` returned it — and is never
        // unloaded, which is what makes its descriptors `&'static`.
        let declared = unsafe { world.adopt(lib.components()) }?;
        let mut drive = match replay {
            Some(replay) => Drive::Replay(replay),
            None => Drive::Live(
                args.record
                    .is_some()
                    .then(|| Box::new(Recorder::new(meta(&lib, hz)))),
            ),
        };
        // Segment zero names the build that produced tick zero (§4.7).
        drive.open_segment(0, lib.code_hash());
        let input = bind(&bindings, &drive, &lib)?;
        info!(
            path = %lib.path().display(),
            components = declared,
            systems = lib.systems().len,
            contexts = input.map().context_count(),
            replaying = drive.ticks(),
            "game loaded"
        );
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
        // and a rectangle (§4.9).
        #[cfg(feature = "overlay")]
        renderer.set_ui_atlas(&gg_debug::atlas())?;
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

    /// The world staged for a successor, once the loop is over and the window is
    /// gone. `Some` means this process exists only to start the next one.
    pub fn handoff(&mut self) -> Option<gg_core::Handoff> {
        self.rejuvenate.take()
    }

    /// The recording, once the loop is over.
    pub fn finish(self) -> Option<Box<Recorder>> {
        self.drive.recorder()
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
        // SAFETY: `reloaded.lib` passed `GameLib::load`'s checks and is never
        // unloaded.
        unsafe { world.adopt(reloaded.lib.components()) }?;
        let report = world.restore(&snapshot)?;
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
        self.input = bind(&self.bindings, &self.drive, &reloaded.lib)?;
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

/// The replay header this build would record (§4.7): the reproduction
/// environment, and the verb lists whose *order* is the id space it writes.
fn meta(lib: &GameLib, hz: u32) -> ReplayMeta {
    // SAFETY: `lib` is verified and never unloaded.
    let verbs = unsafe { verbs_of(lib) };
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
fn bind(bindings: &str, drive: &Drive, lib: &GameLib) -> anyhow::Result<Input> {
    // SAFETY: `lib` is verified and never unloaded.
    let verbs = unsafe { verbs_of(lib) };
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
        match ready
            .map_err(anyhow::Error::from)
            .and_then(|r| self.swap(r))
        {
            Ok(()) => Ok(()),
            Err(refused) => {
                error!(error = %refused, "reload refused — still running the last good build");
                Ok(())
            }
        }
    }

    fn quitting(&self) -> bool {
        self.rejuvenate.pending()
    }

    fn sim_tick(&mut self, tick: u64) -> anyhow::Result<()> {
        self.next_tick = tick + 1;
        if self.halted {
            return Ok(());
        }
        // Latched here rather than at `poll_input`, because the frame is a
        // *tick's* input: `just_pressed` is an edge between two ticks, and a
        // frame that owes three of them must not report the same edge to all
        // three. Platform events accumulate in `self.input` as they arrive.
        let input = self.drive.frame(&mut self.input, tick);
        let ctx = TickCtx {
            tick,
            tick_hz: self.hz,
            reserved: 0,
            input,
            previous: self.previous,
        };
        self.previous = input;
        // SAFETY: `self.lib` is verified and loaded, and `ctx` outlives the call.
        if let Err(panicked) = unsafe { self.world.run_systems(self.lib.systems(), &ctx) } {
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
        if !self.halted
            && let Err(refused) = self.hierarchy.propagate(&mut self.world)
        {
            error!(error = %refused, tick, "hierarchy refused — sim halted until the next reload");
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
        Ok(())
    }

    fn render(&mut self, _alpha: f32) -> anyhow::Result<()> {
        let Some(renderer) = &mut self.gpu else {
            return Ok(());
        };
        // The overlay reads the readings the frame already took rather than
        // asking the device again, so its rows and Tracy's zones cannot
        // disagree about one frame (§4.8).
        #[cfg(feature = "overlay")]
        let ui = self.overlay.build(&gg_debug::overlay::Stats {
            extent: renderer.extent(),
            tick: self.next_tick,
            passes: renderer.pass_timings(),
            memory: renderer.memory(),
        });
        #[cfg(not(feature = "overlay"))]
        let ui = &[];
        // Dropped at the end of this method, which is before the device is and
        // after the only submit — what RenderDoc records is exactly one frame.
        #[cfg(feature = "debug-tools")]
        let _capture = gg_debug::capture::frame();
        renderer.frame(&self.extracted, &self.view, CLEAR, ui)?;
        // A frame or two behind, which is what not stalling costs; a zone is
        // placed by its GPU timestamps, not by when it was sent (§4.8).
        #[cfg(feature = "debug-tools")]
        if let Some(zones) = &mut self.zones {
            zones.frame(renderer.pass_timings(), renderer.gpu_clock());
        }
        Ok(())
    }
}
