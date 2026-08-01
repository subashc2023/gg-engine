//! Rejuvenation: hand this world to a successor process and leave (§4.2.2).
//!
//! Dev leaks every dylib it loads, deliberately — `dlclose` on a Rust dylib with
//! lingering TLS destructors is a classic crash source. That makes leaked bytes a
//! budget, and crossing it is answered by *restarting* rather than by an unload
//! nobody can make safe: snapshot, restart, restore, continue.
//!
//! The successor is a **sibling, not a child**. Waiting on it would keep the
//! process whose memory is the whole reason for the restart alive for the entire
//! new session, which is the one outcome that makes rejuvenation pointless.
//!
//! Not gated on `hot-reload`: only the *trigger* is dev-only. §8's named fallback
//! if reload latency ever goes bad is this same restart, and so is the editor's
//! Play/Stop.

use std::path::{Path, PathBuf};

use super::{LeakBudget, ReloadError};

const MAGIC: [u8; 4] = *b"GGRJ";
const FORMAT: u32 = 1;
/// Magic, format, tick.
const HEADER: usize = 16;

/// What a predecessor left for its successor: the sim clock, and the world as
/// `Snapshot::encode` flattened it.
///
/// The tick rides here rather than inside the world image because it is the
/// *host's* state — `gg-core` owns the clock — and a game that reads `tick` would
/// see time run backwards if a restart silently resumed at zero.
#[derive(Clone, Debug)]
pub struct Handoff {
    /// The tick to resume at: the first one that had not run when the world was
    /// captured.
    pub tick: u64,
    /// The encoded world. Opaque here: `gg-core` sits below the ECS (§3) and
    /// carries these bytes without reading them.
    pub world: Vec<u8>,
}

impl Handoff {
    /// Flatten for the file. A header and a byte run — the world image already
    /// checks itself, and doubling that here would be a second answer to one
    /// question.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER + self.world.len());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT.to_le_bytes());
        out.extend_from_slice(&self.tick.to_le_bytes());
        out.extend_from_slice(&self.world);
        out
    }

    /// Read back what [`Handoff::encode`] wrote.
    pub fn decode(bytes: &[u8]) -> Result<Handoff, ReloadError> {
        let bad = |detail: String| ReloadError::Handoff { detail };
        let header = bytes.get(..HEADER).ok_or_else(|| {
            bad(format!(
                "{} bytes, shorter than the {HEADER}-byte header",
                bytes.len()
            ))
        })?;
        if header[..4] != MAGIC {
            return Err(bad(format!(
                "magic is {:02x?}, expected {MAGIC:02x?}",
                &header[..4]
            )));
        }
        let mut format = [0u8; 4];
        format.copy_from_slice(&header[4..8]);
        let format = u32::from_le_bytes(format);
        if format != FORMAT {
            return Err(bad(format!("format {format}, this build writes {FORMAT}")));
        }
        let mut tick = [0u8; 8];
        tick.copy_from_slice(&header[8..HEADER]);
        Ok(Handoff {
            tick: u64::from_le_bytes(tick),
            world: bytes[HEADER..].to_vec(),
        })
    }

    /// Take the handoff at `path`, removing the file.
    ///
    /// Removed rather than left: it is a message from one process to one other,
    /// and a stale one on disk is a session that resumes somebody else's world.
    pub fn take(path: &Path) -> Result<Handoff, ReloadError> {
        let bytes = std::fs::read(path).map_err(|source| ReloadError::Rejuvenate {
            detail: "reading the staged world",
            source,
        })?;
        let _ = std::fs::remove_file(path);
        Handoff::decode(&bytes)
    }
}

/// The leaked-dylib budget, and the handoff that crossing it produces.
///
/// Policy rather than wiring, which is why it is here and not in the shell (§3
/// caps that crate at zero engine logic): *when* a session is retired, *whether*
/// it may be, and what the operator is told are all decisions about the
/// process's own lifecycle.
#[derive(Debug, Default)]
pub struct Rejuvenator {
    budget: LeakBudget,
    staged: Option<Handoff>,
}

impl Rejuvenator {
    /// A rejuvenator over `budget` bytes of leaked dylib, or the default.
    ///
    /// Zero restarts on the first reload — how the path is exercised on demand
    /// instead of after a thousand edits (§6 M5).
    #[must_use]
    pub fn new(budget: Option<u64>) -> Self {
        Self {
            budget: budget.map_or_else(LeakBudget::default, LeakBudget::new),
            staged: None,
        }
    }

    /// Charge a swapped-out dylib's bytes to the budget.
    pub fn retire(&mut self, bytes: u64) {
        self.budget.charge(bytes);
    }

    /// Whether this session should be handed on — spent, and not staged already.
    /// The caller flattens its world only when this says so.
    #[must_use]
    pub fn due(&self) -> bool {
        self.budget.exhausted() && self.staged.is_none()
    }

    /// Stage `world` for a successor process, to resume at `tick`.
    ///
    /// `restartable` is whether the run's input stream could survive the restart
    /// at all; a run that is recording or replaying is refused here rather than
    /// corrupted later, and says so.
    pub fn stage(&mut self, tick: u64, world: Vec<u8>, restartable: bool) {
        let (spent, budget) = (self.budget.spent(), self.budget.budget());
        if !restartable {
            tracing::warn!(
                spent,
                budget,
                "leak budget spent — rejuvenation deferred: this run is recording or replaying, \
                 and its input stream cannot cross a restart"
            );
            return;
        }
        tracing::warn!(
            spent,
            budget,
            tick,
            "leaked dylib budget spent — rejuvenating"
        );
        self.staged = Some(Handoff { tick, world });
    }

    /// Whether a session is waiting to be handed on — the loop's cue to stop.
    #[must_use]
    pub fn pending(&self) -> bool {
        self.staged.is_some()
    }

    /// Take the staged session. Call once the window and the GPU are down: the
    /// snapshot is worth nothing if the teardown that accounts for them (§4.3)
    /// is skipped to reach the restart sooner.
    pub fn take(&mut self) -> Option<Handoff> {
        self.staged.take()
    }
}

/// Stage `handoff`, start a fresh copy of this process over it, and exit.
///
/// Never returns on success — the process is gone. `flag` is the argument the
/// successor receives the staged path under: the shell owns its own CLI, so the
/// spelling is passed in, and an existing occurrence in this process's argv is
/// *dropped* rather than accumulated, because a long session rejuvenates more
/// than once.
///
/// # Panics
///
/// Never. A failure returns `Err` and the caller keeps playing: a restart that
/// cannot happen must leave the session running rather than end it.
pub fn restart(handoff: &Handoff, flag: &str) -> Result<std::convert::Infallible, ReloadError> {
    let failed = |detail: &'static str| move |source| ReloadError::Rejuvenate { detail, source };
    let path: PathBuf =
        std::env::temp_dir().join(format!("gg-handoff-{}.world", std::process::id()));
    std::fs::write(&path, handoff.encode()).map_err(failed("staging the world"))?;

    let exe = std::env::current_exe().map_err(failed("finding this executable"))?;
    let mut command = std::process::Command::new(exe);
    let mut argv = std::env::args_os().skip(1);
    while let Some(arg) = argv.next() {
        if arg == *flag {
            let _ = argv.next();
            continue;
        }
        command.arg(arg);
    }
    command.arg(flag).arg(&path);
    command.spawn().map_err(failed("starting the successor"))?;

    // Only once the successor is alive. Everything above this line is reversible
    // and reportable; nothing below it is.
    std::process::exit(0)
}
