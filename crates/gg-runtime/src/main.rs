//! `gg-runtime`: THE host shell — the one executable game code ever runs under,
//! in every tier (§2 Game-code boundary, §3). Thin in code, fat in linkage: zero
//! engine logic, zero game logic, no public API. Complexity budget: 500
//! CI-counted *code* lines (§3) — 300 through M4, raised at M5 when the shell
//! took delivery of the window, the renderer, live input and record/replay.
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

/// What the shell was told to run. Hand-parsed: five values, and a parser
/// dependency would be the shell's first gram of fat.
struct Args {
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
    /// Leaked-dylib bytes this session tolerates before rejuvenating. Present so
    /// the forced case — zero, restart on the first reload — is exercisable on
    /// demand instead of after a thousand edits.
    leak_budget: Option<u64>,
}

/// One spelling of the handoff flag: [`Args`] parses it and
/// [`gg_core::reload::rejuvenate::restart`] passes it on, and a drift between
/// the two would make a twice-rejuvenated session accumulate argv.
const RESTORE_FLAG: &str = "--restore";

fn main() -> anyhow::Result<()> {
    // Bound to a named local, not `_`: the guard *is* the Tracy client's
    // lifetime, and `let _ = ..` would drop it here (see `Observability`).
    let _observability = init_observability()?;
    let args = parse_args()?;

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
    let mut app = app::App::new(
        &args.game,
        &staging,
        DEFAULT_TICK_HZ,
        bindings,
        replay,
        args.record.is_some(),
        args.leak_budget,
    )?;
    // Before the first frame: the loop's clock resumes at the tick this carries.
    if let Some(path) = &args.restore {
        app.restore(&gg_core::Handoff::take(path)?)?;
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
    if let (Some(path), Some(recorder)) = (&args.record, app.finish()) {
        let replay = recorder.finish();
        std::fs::write(path, replay.encode())?;
        info!(path = %path.display(), ticks = replay.ticks(),
              changes = replay.change_count(), "replay written");
    }
    info!(frames, "golden runtime clean exit");
    Ok(())
}

fn parse_args() -> anyhow::Result<Args> {
    let mut args = Args {
        game: PathBuf::new(),
        frames: None,
        input: None,
        record: None,
        replay: None,
        restore: None,
        leak_budget: None,
    };
    let mut argv = std::env::args().skip(1);
    while let Some(flag) = argv.next() {
        let mut value = || argv.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--game" => args.game = PathBuf::from(value()?),
            "--frames" => args.frames = Some(value()?.parse()?),
            "--input" => args.input = Some(PathBuf::from(value()?)),
            "--record" => args.record = Some(PathBuf::from(value()?)),
            "--replay" => args.replay = Some(PathBuf::from(value()?)),
            RESTORE_FLAG => args.restore = Some(PathBuf::from(value()?)),
            "--leak-budget" => args.leak_budget = Some(value()?.parse()?),
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

/// Process-lifetime guard for the observability stack. Tracy stays enabled only
/// while a client guard is alive — dropping the last one discards anything not
/// yet delivered — so `start()` with the result thrown away starts and
/// immediately stops it. `TracyLayer` holds a client of its own, which would
/// mask that; relying on it makes all output hostage to construction order.
struct Observability {
    #[cfg(feature = "tracy")]
    _tracy: tracy_client::Client,
}

fn init_observability() -> anyhow::Result<Observability> {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    #[cfg(feature = "tracy")]
    let tracy = tracy_client::Client::start();
    let fmt = tracing_subscriber::fmt::layer().with_target(true);
    // `gg::hash` off by default: it is one line per sim tick (§5.6c) and wanted
    // only by the gate that compares two runs, which asks for it by `RUST_LOG`.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug,gg::hash=off"));
    let registry = tracing_subscriber::registry().with(filter).with(fmt);
    #[cfg(feature = "tracy")]
    let registry = registry.with(tracing_tracy::TracyLayer::default());

    registry.try_init()?;
    Ok(Observability {
        #[cfg(feature = "tracy")]
        _tracy: tracy,
    })
}
