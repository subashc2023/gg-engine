//! `gg-runtime`: THE host shell — the one executable game code ever runs under,
//! in every tier (§2 Game-code boundary, §3). Thin in code, fat in linkage: zero
//! engine logic, zero game logic, no public API. Complexity budget: 300 lines,
//! CI-counted (§3). M0A scope: boot, observability hookup (tracing + Tracy),
//! a heartbeat proving zones and structured logs flow, clean exit. The real loop
//! skeleton arrives with `gg-core` (M1) and is *wired* here, never implemented here.

use tracing::{debug, info, info_span};

fn main() -> anyhow::Result<()> {
    init_observability()?;

    {
        let _startup = info_span!("startup").entered();
        info!(
            version = env!("CARGO_PKG_VERSION"),
            tier = active_tier(),
            headless = std::env::var_os("GG_HEADLESS").is_some(),
            "golden runtime online"
        );
    }

    // M0A heartbeat: a few labeled ticks so the M0A exit criterion — "a
    // hello-world binary emits Tracy zones and structured logs" — is observable.
    for tick in 0u64..3 {
        run_tick(tick);
    }

    info!("golden runtime clean exit");
    Ok(())
}

fn run_tick(tick: u64) {
    profiling::scope!("tick");
    let _span = info_span!("tick", tick).entered();
    debug!(tick, "heartbeat");
    profiling::finish_frame!();
}

/// Which named tier combination this binary was built as (§3). Tiers are
/// meta-features, so exactly one of these is expected per build.
fn active_tier() -> &'static str {
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

fn init_observability() -> anyhow::Result<()> {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    #[cfg(feature = "tracy")]
    tracy_client::Client::start();

    let fmt = tracing_subscriber::fmt::layer().with_target(true);
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug"));

    let registry = tracing_subscriber::registry().with(filter).with(fmt);

    #[cfg(feature = "tracy")]
    let registry = registry.with(tracing_tracy::TracyLayer::default());

    registry.try_init()?;
    Ok(())
}
