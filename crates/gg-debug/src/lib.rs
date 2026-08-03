//! `gg-debug` (§4.8) — the instruments.
//!
//! Tracy, the log tail a crash report attaches, GPU zones fed by `gg-rhi`'s
//! per-pass timestamps, and the CVar console. §3 keeps this crate out of every
//! dist graph, which is the whole shape of the crate: nothing here may be the
//! only implementation of something a shipped build needs. The CVar *registry*
//! is `gg-core`'s for that reason (§4.8) — every crate that declares a knob
//! would otherwise point its dependency arrow here.
//!
//! The shell's share is [`init`], one [`gpu::GpuZones`] and one [`Overlay`]
//! when it holds a renderer. A host that cannot link this crate keeps a console
//! subscriber and loses the rest.
//!
//! The draw layer that used to live here is `gg-ui`'s from M13 — it was always
//! that crate's kernel (§4.9), and the overlay being an ordinary client of it
//! is the acceptance test §4.8 named.

#![warn(missing_docs)]

pub mod capture;
pub mod console;
pub mod cpu;
pub mod crash;
pub mod gpu;
pub mod log_tail;
pub mod overlay;

pub use gpu::GpuZones;
pub use log_tail::tail;
pub use overlay::Overlay;

use tracing_subscriber::layer::SubscriberExt as _;
use tracing_subscriber::util::SubscriberInitExt as _;

/// Why the instruments did not come up. Only ever at startup, and only ever
/// because something already installed a subscriber.
#[derive(Debug, thiserror::Error)]
pub enum DebugError {
    /// A global `tracing` subscriber was already installed. Two subscribers
    /// would mean two answers to where a line went.
    #[error("a tracing subscriber is already installed: {0}")]
    Subscriber(String),
}

/// Register every knob this crate declares. One call, so a host taking delivery
/// of the instruments does not have to track which module grew a CVar this
/// milestone.
///
/// # Errors
///
/// A name already taken.
pub fn register() -> Result<(), gg_core::CVarError> {
    overlay::register()?;
    capture::register()
}

/// The instruments' lifetime.
///
/// Tracy stays enabled only while a client guard is alive — dropping the last
/// one discards anything not yet delivered — so `Client::start()` with the
/// result thrown away starts and immediately stops it. `TracyLayer` holds a
/// client of its own, which would mask that; relying on it makes all output
/// hostage to construction order.
pub struct Guard {
    #[cfg(feature = "tracy")]
    _tracy: tracy_client::Client,
}

/// Install the process's subscriber: terminal, log tail, and Tracy when this
/// build has it. `default_filter` is used only when `RUST_LOG` says nothing —
/// the caller owns that string because which targets are noisy is a property of
/// the host, not of the instruments.
///
/// Call once, first thing. The returned guard must outlive everything worth
/// profiling.
pub fn init(default_filter: &str) -> Result<Guard, DebugError> {
    #[cfg(feature = "tracy")]
    let tracy = tracy_client::Client::start();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_filter));
    let registry = tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(true))
        .with(log_tail::LogTail);
    #[cfg(feature = "tracy")]
    let registry = registry.with(tracing_tracy::TracyLayer::default());
    registry
        .try_init()
        .map_err(|e| DebugError::Subscriber(e.to_string()))?;
    Ok(Guard {
        #[cfg(feature = "tracy")]
        _tracy: tracy,
    })
}
