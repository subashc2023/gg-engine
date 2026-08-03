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

use std::path::PathBuf;

use anyhow::Context as _;
use gg_core::{DEFAULT_TICK_HZ, FrameLoop};
use gg_input::Replay;
use tracing::{info, info_span, warn};

mod app;
mod play;

/// What the shell was told to run. Hand-parsed: a handful of values, and a
/// parser dependency would be the shell's first gram of fat. Passed to [`App`]
/// whole rather than field by field — one struct is what a session *is*.
///
/// `Default` is derived rather than written out at the one construction site: a
/// new flag then costs the field and its arm, not a third place to forget.
#[derive(Default)]
pub struct Args {
    game: PathBuf,
    /// Bounded run. Headless it is `Pace::Locked` — wall time ignored, so a
    /// run's tick count is a property of the run and not of the machine (§5.6).
    frames: Option<u64>,
    /// The action map (§4.7). Resolved against the *game's* declared verbs, so
    /// a binding naming a verb this build does not declare is refused by name.
    input: Option<PathBuf>,
    record: Option<PathBuf>,
    replay: Option<PathBuf>,
    /// A world staged by this shell's *predecessor* (§4.2.2). Passed by a
    /// rejuvenating process to its successor, never by hand.
    restore: Option<PathBuf>,
    /// A save to load before the first tick, and where to write one after the
    /// last (§6 M14). Unlike `--restore` these are the *player's* file: written
    /// by one build, read by another, and refused by name rather than migrated
    /// into loss. In every tier, because that is the point of them.
    load: Option<PathBuf>,
    save: Option<PathBuf>,
    /// `<enter tick>:<stop tick>` — the editor's play/stop, on a script (§6 M14).
    play: Option<String>,
    /// Open the editor over the game (§6 M15). Lab equipment: the flag exists
    /// only in a tier that has the crate, and is refused elsewhere by name
    /// rather than ignored. With it, `--save` names where its save button
    /// writes rather than what the shell writes at exit.
    editor: bool,
    /// Leaked-dylib bytes this session tolerates before rejuvenating. Present so
    /// the forced case — zero, restart on the first reload — is exercisable on
    /// demand instead of after a thousand edits.
    leak_budget: Option<u64>,
    /// The pack a game draws out of (§4.6). Which file, and nothing else: what
    /// to draw from it is the game's to say, through `Model`.
    pack: Option<PathBuf>,
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

fn main() -> anyhow::Result<()> {
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
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = parse_args(&argv)?;

    {
        let _startup = info_span!("startup").entered();
        info!(
            version = env!("CARGO_PKG_VERSION"),
            tier = active_tier(),
            headless = std::env::var_os("GG_HEADLESS").is_some(),
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
        gg_core::config::boot(std::path::Path::new(CONFIG), &argv)?;
    }

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
    let mut app = app::App::new(&args, &staging, DEFAULT_TICK_HZ, bindings, replay)?;
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
        play::play(&mut app, "golden", target)?
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
    if let (Some(path), Some(recorder)) = (&args.record, app.finish()) {
        let replay = recorder.finish();
        std::fs::write(path, replay.encode())?;
        info!(path = %path.display(), ticks = replay.ticks(),
              changes = replay.change_count(), "replay written");
    }
    info!(frames, "golden runtime clean exit");
    Ok(())
}

fn parse_args(argv: &[String]) -> anyhow::Result<Args> {
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
