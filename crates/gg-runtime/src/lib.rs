//! `gg-runtime`: THE host shell — the one executable game code ever runs under,
//! in every tier (§2 Game-code boundary, §3). Thin in code, fat in linkage: zero
//! engine logic, zero game logic, no public API. Complexity budget: 600
//! CI-counted *code* lines (§3) — 300 through M4, 500 at M5 when the shell took
//! delivery of the window, the renderer, live input and record/replay, 600 at M8
//! for the observability stack. The number lives in `xtask`; this restates it.
//!
//! Boot, observability, the game dylib, and `gg-core`'s loop driven to
//! completion. Everything it does is a choice of *which* engine piece runs;
//! the wiring proper is in [`app`], the window in [`play`], and the loop
//! skeleton both drive is `gg-core`'s (§4.1) — never reimplemented here.
//!
//! # Why this is a library as well as a binary
//!
//! `apps/gg-editor` (§6 M15.1 item 4) opens the editor with no game by driving
//! this shell as [`run`] rather than holding a second copy of it, so `main` is
//! just the argv in front of it — and §3's line budget counts a launcher's own
//! lines as the application's, not the shell's.

use std::path::PathBuf;

use anyhow::Context as _;
use gg_core::{DEFAULT_TICK_HZ, FrameLoop};
use gg_input::Replay;
use tracing::{info, info_span, warn};

mod app;
mod play;

/// `--set`, for a host with its own argv to consume it the same way (§4.8).
pub use gg_core::config::SET_FLAG;

/// What the shell was told to run. Hand-parsed: a handful of values, and a
/// parser dependency would be the shell's first gram of fat. Passed to [`App`]
/// whole rather than field by field — one struct is what a session *is*.
///
/// `Default` is derived rather than written out at the one construction site: a
/// new flag then costs the field and its arm, not a third place to forget.
#[derive(Default)]
pub struct Args {
    /// The game dylib. **Empty is a real value** and means no project: the
    /// launcher's opening state, where the host registers its own §4.5 protocol
    /// types and the world stays empty (§6 M15.1 item 4). `parse_args` still
    /// requires the flag — a shell run from a command line with no game is a
    /// mistake, and a launcher does not have a command line.
    pub game: PathBuf,
    /// Bounded run. Headless it is `Pace::Locked` — wall time ignored, so a
    /// run's tick count is a property of the run and not of the machine (§5.6).
    pub frames: Option<u64>,
    /// The action map (§4.7). Resolved against the *game's* declared verbs, so
    /// a binding naming a verb this build does not declare is refused by name.
    pub input: Option<PathBuf>,
    /// Where to write this session's recording.
    pub record: Option<PathBuf>,
    /// A recording to drive this session from.
    pub replay: Option<PathBuf>,
    /// A world staged by this shell's *predecessor* (§4.2.2). Passed by a
    /// rejuvenating process to its successor, never by hand.
    pub restore: Option<PathBuf>,
    /// A save to load before the first tick (§6 M14). Unlike `--restore` this is
    /// the *player's* file: written by one build, read by another, and refused by
    /// name rather than migrated into loss. In every tier, because that is the
    /// point of it.
    pub load: Option<PathBuf>,
    /// Where to write one after the last tick.
    pub save: Option<PathBuf>,
    /// `<enter tick>:<stop tick>` — the editor's play/stop, on a script (§6 M14).
    pub play: Option<String>,
    /// Open the editor over the game (§6 M15). Lab equipment: the flag exists
    /// only in a tier that has the crate, and is refused elsewhere by name
    /// rather than ignored. With it, `--save` names where its save button
    /// writes rather than what the shell writes at exit.
    pub editor: bool,
    /// `<w>x<h>` — lay the editor out for this surface instead of the real one
    /// (§6 M15.1).
    ///
    /// The panes fill their surface, so where a widget *is* depends on how big
    /// that surface is, and a session recorded at one size lands its clicks
    /// somewhere else at another. This is how a recording made in a window is
    /// replayed headlessly: name the extent it was recorded at. It changes
    /// nothing else — the window, the swapchain and the game's own canvas are
    /// still the real one, so passing it to a *windowed* run deliberately draws
    /// an editor that does not match its window.
    pub editor_extent: Option<(u32, u32)>,
    /// Leaked-dylib bytes this session tolerates before rejuvenating. Present so
    /// the forced case — zero, restart on the first reload — is exercisable on
    /// demand instead of after a thousand edits.
    pub leak_budget: Option<u64>,
    /// The pack a game draws out of (§4.6). Which file, and nothing else: what
    /// to draw from it is the game's to say, through `Model`.
    pub pack: Option<PathBuf>,
}

/// Where a project keeps its opening scene: `scene.ggsave` beside the action
/// map, so a project is still the directory `xtask run` points at (§6 M15.2
/// post-close). The one spelling — the editor's save button writes here and
/// [`opening_scene`] reads here, and a drift between the two would leak a scene
/// one session wrote past the next one's probe.
pub(crate) fn scene_path(input: Option<&std::path::Path>) -> Option<PathBuf> {
    Some(input?.parent()?.join("scene.ggsave"))
}

/// The save a live session opens from, found rather than named. Editor-only
/// until §6 M20 pull 2 made the scene the *project's* data: a game whose level
/// is checked in as `scene.ggsave` must open it in every tier, dist included,
/// or the level would be lab equipment. The probe stays implicit, so a recorded
/// or replayed session still refuses it — a scene appearing beside the project
/// later must not diverge a stream blessed without one — and `--load` stays the
/// explicit spelling, taking precedence so the two cannot both load.
pub(crate) fn opening_scene(args: &Args) -> Option<PathBuf> {
    let live = may_touch_project(args) && args.load.is_none();
    live.then(|| scene_path(args.input.as_deref()))
        .flatten()
        .filter(|scene| scene.is_file())
}

/// Whether this run may read or write the project's own files — its scene, its
/// dock layout (§6 M15.1, M15.2). False under `--record` and `--replay`: a
/// blessed stream must land its clicks against the layout the gate recorded
/// with, and must leave no `scene.ggsave` behind for the next run to open.
pub(crate) fn may_touch_project(args: &Args) -> bool {
    args.record.is_none() && args.replay.is_none()
}

/// One spelling of the handoff flag: [`Args`] parses it and
/// [`gg_core::reload::rejuvenate::restart`] passes it on, and a drift between
/// the two would make a twice-rejuvenated session accumulate argv.
const RESTORE_FLAG: &str = "--restore";

/// The config file, read from the working directory. Not having written one is
/// the normal case (§4.8), so there is no flag to point elsewhere: a run that
/// wants one value wants `--set`, and a run that wants a different *file* is
/// choosing a different working directory anyway.
const CONFIG: &str = "gg.cfg";

/// Boot, then sessions until nothing asks for another.
///
/// `argv` is what `gg_core::config` reads its own flags out of; `args` is what
/// this shell was told, which a launcher constructs rather than parses.
///
/// **The loop is one pass in every ordinary run.** A session ends by asking to
/// open a project (§6 M15.1 item 4) and nothing else does that, so a shell
/// pointed at a game runs exactly once and returns.
pub fn run(mut args: Args, argv: &[String]) -> anyhow::Result<()> {
    // Bound to a named local, not `_`: the guard *is* the Tracy client's
    // lifetime, and `let _ = ..` would drop it here (see `Observability`).
    // Dist has no guard to hold, so the binding is a unit there and the lint is
    // right about that one tier and wrong about the shape.
    #[cfg_attr(not(feature = "debug-tools"), allow(clippy::let_unit_value))]
    let _observability = init_observability()?;
    // Hazard 5's startup call site (§4.2.1): before any dependency's initializer
    // has had a chance to vandalize MXCSR/FPCR unnoticed. Demos 00–02 do the
    // same; the shell is what every demo from 03 on actually runs under.
    #[cfg(feature = "fp-assert")]
    gg_math::fpenv::assert_fp_env();

    {
        let _startup = info_span!("startup").entered();
        info!(
            version = env!("CARGO_PKG_VERSION"),
            tier = active_tier(),
            headless = gg_platform::headless(),
            game = %args.game.display(),
            "golden runtime online"
        );
        // Game output has nowhere to go until this is set, and two loggers would
        // mean two answers to where a line went — so it is set once, here.
        if gg_ecs::boundary::set_logger(log_from_game).is_err() {
            warn!("a logger was already installed");
        }
        // Per-system CPU zones (§4.8). Only the instrumented graph has anywhere
        // to send them: `gg-debug` is absent from dist (§3), so the ECS's hook
        // stays unset there and its loop keeps a null check.
        #[cfg(feature = "debug-tools")]
        if gg_ecs::boundary::set_system_zone(gg_debug::cpu::system_zone).is_err() {
            warn!("a system zone was already installed");
        }
        // Every crate's knobs registered before anything is applied, or a name
        // the config file uses would be unknown by an accident of ordering.
        gg_render::cvars::register()?;
        #[cfg(feature = "debug-tools")]
        gg_debug::register()?;
        gg_core::config::boot(std::path::Path::new(CONFIG), argv)?;
    }

    while let Some(next) = session(&args)? {
        args = next;
    }
    Ok(())
}

/// One session: a world over a game, driven to the end of its loop.
///
/// `Some` is a project the operator picked out of the launcher (§6 M15.1 item
/// 4) — the arguments the *next* session runs under, which is a new shell state
/// and not a swap, because a session is built around the dylib it was pointed
/// at. Everything else about the run carries over, `--frames` included.
fn session(args: &Args) -> anyhow::Result<Option<Args>> {
    let staging = std::env::temp_dir().join(format!("gg-runtime-{}", std::process::id()));
    let bindings = match &args.input {
        Some(path) => {
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?
        }
        None => String::new(),
    };
    let replay = args
        .replay
        .as_ref()
        .map(|path| -> anyhow::Result<_> { Ok(Box::new(Replay::decode(&std::fs::read(path)?)?)) })
        .transpose()?;
    let mut app = app::App::new(args, &staging, DEFAULT_TICK_HZ, bindings, replay)?;
    // Before the first frame: the loop's clock resumes at the tick this carries.
    if let Some(path) = &args.restore {
        app.restore(&gg_core::Handoff::take(path)?)?;
    }
    // After the handoff, and they are not two answers to one question: a
    // predecessor's world is this process continuing, a save is a session
    // resuming. A run given both continues into the saved one.
    if let Some(path) = &args.load {
        app.load_save(path)?;
    }
    // The project's opening scene (§6 M15.2 post-close; §6 M20 pull 2): the
    // world as data — the editor opens Stopped at the save's tick with no game
    // code run, and a plain run opens the level the project checked in.
    if let Some(scene) = opening_scene(args) {
        info!(scene = %scene.display(), "opening scene");
        app.load_save(&scene)?;
    }

    // Headless is windowless, not invisible-windowed: §1.5 forbids an automated
    // tier from creating an OS window *at all*, and every `xtask ci` tier sets
    // `GG_HEADLESS=1`. The sim, the reload path and the replay stream are the
    // same either way, which is what lets CI exercise the shell it ships.
    // A replay bounds its own run: the file says how many ticks it covers, and
    // honouring `--frames` over it is how a replay stops reproducing (§4.7).
    let target = args.frames.or_else(|| app.ticks());
    let frames = if gg_platform::headless() {
        let target = target.with_context(
            || "a headless run needs --frames N or --replay: there is no window to close it (§1.5)",
        )?;
        FrameLoop::locked(DEFAULT_TICK_HZ)
            .resuming_at(app.next_tick())
            .run(&mut app, target)?
    } else {
        let title = app.title();
        play::play(&mut app, &title, target)?
    };

    // The window is down and the GPU is accounted for (§4.3), which is the only
    // point a session may be handed on. Never returns on success.
    if let Some(handoff) = app.handoff() {
        gg_core::reload::rejuvenate::restart(&handoff, RESTORE_FLAG)?;
    }
    // Before `finish`, which consumes the app: what a save holds is the world,
    // and the world is gone once the recorder has been taken out of it.
    // Not with the editor open: there `--save` names where its *button* writes,
    // and a second write at exit would bury what the operator actually did.
    if let Some(path) = &args.save
        && !args.editor
    {
        app.write_save(path)?;
    }
    // Also before `finish`, and the reason is the same one: the pick is the
    // editor's and the editor goes with the app.
    #[cfg(feature = "editor")]
    let next = app.opening().map(|project| {
        info!(project = project.name, game = %project.game.display(), "opening project");
        Args {
            game: project.game.clone(),
            input: project.input.clone(),
            pack: project.pack.clone(),
            // Only what describes the *run* carries. A recording cannot cross
            // this seam at all — a replay's id space is the verb list it was
            // made against (§4.7), and the launcher's is the editor's appended
            // verbs over no game, which is not any game's. The save and restore
            // flags are the same argument one step out: they named a session,
            // and this is the next one.
            frames: args.frames,
            editor: args.editor,
            editor_extent: args.editor_extent,
            leak_budget: args.leak_budget,
            ..Args::default()
        }
    });
    #[cfg(not(feature = "editor"))]
    let next = None;
    if let (Some(path), Some(recorder)) = (&args.record, app.finish()) {
        let replay = recorder.finish();
        std::fs::write(path, replay.encode())?;
        info!(path = %path.display(), ticks = replay.ticks(),
              changes = replay.change_count(), "replay written");
    }
    info!(frames, "golden runtime clean exit");
    Ok(next)
}

/// `--editor-extent`'s value. Public because the launcher takes the same flag
/// (§6 M15.1 item 4) and two spellings of one parse would drift.
pub fn parse_extent(text: &str) -> anyhow::Result<(u32, u32)> {
    let (w, h) = text
        .split_once(['x', 'X'])
        .with_context(|| format!("--editor-extent wants <w>x<h>, got `{text}`"))?;
    Ok((w.trim().parse()?, h.trim().parse()?))
}

/// The command line, as [`Args`].
pub fn parse_args(argv: &[String]) -> anyhow::Result<Args> {
    let mut args = Args::default();
    let mut argv = argv.iter().cloned();
    while let Some(flag) = argv.next() {
        let mut value = || argv.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            // Consumed and dropped: `gg_core::config` reads the same argv for
            // these, and the shell's job here is only to not call them unknown.
            gg_core::config::SET_FLAG => drop(value()?),
            "--game" => args.game = PathBuf::from(value()?),
            "--frames" => args.frames = Some(value()?.parse()?),
            "--input" => args.input = Some(PathBuf::from(value()?)),
            "--record" => args.record = Some(PathBuf::from(value()?)),
            "--replay" => args.replay = Some(PathBuf::from(value()?)),
            RESTORE_FLAG => args.restore = Some(PathBuf::from(value()?)),
            "--load" => args.load = Some(PathBuf::from(value()?)),
            "--save" => args.save = Some(PathBuf::from(value()?)),
            "--play" => args.play = Some(value()?),
            "--editor" => {
                anyhow::ensure!(
                    cfg!(feature = "editor"),
                    "--editor: this tier has no editor (§3 keeps it out of the dist graph)"
                );
                args.editor = true;
            }
            "--editor-extent" => args.editor_extent = Some(parse_extent(&value()?)?),
            "--leak-budget" => args.leak_budget = Some(value()?.parse()?),
            "--pack" => args.pack = Some(PathBuf::from(value()?)),
            other => anyhow::bail!("unknown argument `{other}`"),
        }
    }
    anyhow::ensure!(
        args.record.is_none() || args.replay.is_none(),
        "--record and --replay are two answers to where this run's input comes from"
    );
    anyhow::ensure!(
        args.editor_extent.is_none_or(|(w, h)| w > 0 && h > 0),
        "--editor-extent must be positive: a zero-sized editor lays out nothing"
    );
    // The launcher reaches the same state by *constructing* `Args`, not by
    // omitting a flag: a command line with no `--game` is a mistake, and a
    // shell that opened an empty editor over one would hide it (§6 M15.1 item 4).
    anyhow::ensure!(
        !args.game.as_os_str().is_empty(),
        "--game <path to the game dylib> is required: the shell is the same program in every \
         tier and the game is what makes it a game (§2)"
    );
    Ok(args)
}

/// Game `log` calls, on the host's stream. A forwarding function rather than a
/// `tracing` dependency inside `gg-ecs` (§3) — this is the line that buys.
fn log_from_game(level: u32, message: &str) {
    use gg_ecs::boundary::log_level;
    match level {
        log_level::ERROR => tracing::error!(target: "game", "{message}"),
        log_level::WARN => tracing::warn!(target: "game", "{message}"),
        log_level::DEBUG => tracing::debug!(target: "game", "{message}"),
        log_level::TRACE => tracing::trace!(target: "game", "{message}"),
        _ => tracing::info!(target: "game", "{message}"),
    }
}

/// Which named tier combination this binary was built as (§3). Tiers are
/// meta-features, so exactly one of these is expected per build.
pub fn active_tier() -> &'static str {
    if cfg!(feature = "tier-dev") {
        "dev"
    } else if cfg!(feature = "tier-instrumented") {
        "instrumented"
    } else if cfg!(feature = "tier-dist-verify") {
        "dist-verify"
    } else {
        "dist"
    }
}

/// `gg::hash` off by default: it is one line per sim tick (§5.6c) and wanted
/// only by the gate that compares two runs, which asks for it by `RUST_LOG`.
const LOG_FILTER: &str = "debug,gg::hash=off";

/// The instruments (§4.8): Tracy, the log tail a crash report attaches, the
/// console. `gg-debug` is absent from every dist graph by §3, so this is two
/// bodies rather than one with a flag in it — dist keeps the terminal and loses
/// the rest, which is the whole difference.
#[cfg(feature = "debug-tools")]
fn init_observability() -> anyhow::Result<gg_debug::Guard> {
    let guard = gg_debug::init(LOG_FILTER)?;
    // After the tail exists, never before: a report from a process whose logging
    // had not come up yet would attach nothing (§4.8).
    gg_debug::crash::install(gg_debug::crash::Product {
        name: env!("CARGO_PKG_NAME"),
        version: env!("CARGO_PKG_VERSION"),
        tier: active_tier(),
    });
    Ok(guard)
}

#[cfg(not(feature = "debug-tools"))]
fn init_observability() -> anyhow::Result<()> {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(LOG_FILTER));
    tracing_subscriber::fmt()
        .with_target(true)
        .with_env_filter(filter)
        .try_init()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}
