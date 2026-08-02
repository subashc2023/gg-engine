//! `gg-core` (§4.1, §3) — the loop skeleton, the tick clock, app events, config
//! and the CVar registry. The reload host (§4.2.2) joins them.
//!
//! # The charter is closed
//!
//! §3 names two crates with dumping-ground gravity, and this is the one whose
//! answer is a *closed* list rather than an open one: lifecycle, loop skeleton,
//! tick clock, events, config, CVar registry, reload host, **and nothing else**.
//! A ninth responsibility arriving here edits §3's budget in its own PR; it does
//! not arrive quietly. The dependency budget (≤ 8 external crates, `cargo-deny`
//! counted) is the machine behind that sentence.
//!
//! What that buys is direction. `gg-core` sits below `gg-render`, `gg-extract`
//! and `gg-platform` (§3), so it owns the *shape* of a frame and never its
//! contents: it cannot call a renderer, cannot touch an ECS world, and does not
//! link winit. The app shell wires the concrete stages in at startup
//! ([`Stages`]), and the loop calls them in the one order there is.
//!
//! ```
//! use core::time::Duration;
//! use gg_core::{FrameLoop, Flow, Stages};
//!
//! struct Game { ticks: u64 }
//! impl Stages for Game {
//!     type Error = std::convert::Infallible;
//!     fn sim_tick(&mut self, _tick: u64) -> Result<(), Self::Error> {
//!         self.ticks += 1;
//!         Ok(())
//!     }
//! }
//!
//! let mut game = Game { ticks: 0 };
//! // Locked pace: one tick per frame, wall time ignored — a bounded run's tick
//! // count is a property of the run, not of the machine (§5.6).
//! let mut app = FrameLoop::locked(60);
//! app.run(&mut game, 100)?;
//! assert_eq!(game.ticks, 100);
//! # Ok::<(), std::convert::Infallible>(())
//! ```

#![warn(missing_docs)]

pub mod clock;
pub mod config;
pub mod cvar;
pub mod event;
pub mod frame;
pub mod reload;

pub use clock::{DEFAULT_TICK_HZ, Due, MAX_TICKS_PER_FRAME, Pace, TickClock};
pub use cvar::{CVar, CVarError, CVarKind, CVarSource};
pub use event::{AppEvent, Events};
pub use frame::{Flow, FrameLoop, Stages};
pub use reload::rejuvenate::Handoff;
pub use reload::{GameLib, LeakBudget, ReloadError};
