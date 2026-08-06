//! The save-watcher and copy-then-load staging (§4.2.2) — the dev half of the
//! reload host, and the only half dist does not have.
//!
//! Lab equipment by construction: this module exists only under `hot-reload`,
//! and the dist gate proves `notify` absent from a shipping binary (§5.8).
//!
//! # Why the copy
//!
//! Windows holds a loaded DLL open, so loading `target/debug/game.dll` in place
//! would make the *next* `cargo build` fail rather than the reload. Each load
//! takes a numbered copy aside and loads that, which also means the artifact a
//! `GameLib` names still exists after the next build overwrote the original —
//! useful in a crash dump, and free.
//!
//! # Why a rebuilt file is not loaded the moment it appears
//!
//! The watcher fires while the linker is still writing. [`Watch::poll`] waits
//! for the artifact's size and modification time to hold still for
//! [`SETTLE_QUIET`] before staging it — a wall-clock quiet period rather than a
//! number of polls, because `poll` is called wherever the loop calls it and a
//! bounded headless run calls it hundreds of thousands of times a second.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime};

use gg_abi::HostApiV1;
use notify::Watcher as _;

use super::{GameLib, ReloadError};

/// How long the artifact must hold the same size and mtime before it counts as
/// fully written.
///
/// **Time, not a poll count.** `poll` runs wherever the loop calls it, and that
/// is 60 Hz under a player but hundreds of thousands per second in a bounded
/// headless run — so "two consecutive polls agreed" can mean "two microseconds
/// apart, in the middle of someone's 2 MB write". That is not a settle rule, it
/// is a race the fast lane loses, and losing it stages a truncated dylib and
/// reports `file too short` for an artifact that was perfectly good a
/// millisecond later.
/// Deliberately not `ggc::watch`'s 120 ms: that rule waits for a *tree's*
/// event burst to go quiet (outlast a save); this one holds one artifact's
/// size and mtime still (outlast a linker). Same shape, different
/// measurement — change one and read the other's note first (§6 M16 item 4).
const SETTLE_QUIET: Duration = Duration::from_millis(40);

/// How long a stubbornly-unreadable artifact is given before the wait is
/// reported as a failure — long enough for any linker, short enough that a
/// genuinely stuck file is not silence. Also in time, for the same reason.
const STAGING_PATIENCE: Duration = Duration::from_secs(2);

/// How long [`Watch::block_until_ready`] spins before it starts sleeping, and
/// how long it sleeps after that.
///
/// The first load settles in two polls (`request` sets no quiet period), so the
/// good case never reaches the sleep and pays nothing against M9's
/// load-to-first-frame budget. Only an absent or locked artifact polls
/// repeatedly, and that case has [`STAGING_PATIENCE`] to burn: without the
/// sleep it is a `stat` per iteration at full core for two seconds, to produce
/// an error message.
const FIRST_LOAD_SPIN: Duration = Duration::from_millis(2);
const FIRST_LOAD_NAP: Duration = Duration::from_millis(1);

/// A dylib that arrived, was staged, and passed every check.
pub struct Reloaded {
    /// The new library. The caller swaps it in at `reload_check` and retires the
    /// old one into a [`LeakBudget`](super::LeakBudget).
    pub lib: GameLib,
    /// When the file event arrived — the zero of M5's "< 2 s save to new
    /// behaviour" clock, which is the exit criterion this field exists for.
    pub saved_at: Instant,
    /// How long staging, loading and verification took.
    pub load_time: Duration,
}

impl std::fmt::Debug for Reloaded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reloaded")
            .field("lib", &self.lib)
            .field("load_time", &self.load_time)
            .finish()
    }
}

/// A rebuild seen but not yet acted on.
struct Pending {
    saved_at: Instant,
    /// The size and mtime last observed, and when they were first observed —
    /// the artifact is settled once that pair has held for [`Self::quiet`].
    settled_at: Option<((u64, SystemTime), Instant)>,
    /// How long that pair must hold. [`SETTLE_QUIET`] when a file event started
    /// this, and zero when [`Watch::request`] did: the quiet period exists to
    /// outlast a *writer*, and at startup there is none — charging every launch
    /// 40 ms to wait for a build that finished yesterday would be paying M9's
    /// load-to-first-frame budget for nothing.
    quiet: Duration,
}

/// Watches for a rebuilt game dylib and stages it for loading.
///
/// Nothing here swaps anything: [`Watch::poll`] hands back a loaded, verified
/// [`GameLib`] and the caller installs it at `reload_check`, between ticks
/// (§4.1). Keeping the swap out of this type is what lets the loop own *when* a
/// reload lands.
pub struct Watch {
    source: PathBuf,
    staging_dir: PathBuf,
    generation: u32,
    pending: Option<Pending>,
    rx: mpsc::Receiver<Instant>,
    // Held for its Drop: dropping the watcher stops the OS watch. `None` is
    // `absent` — there is no artifact, so there is no directory to watch.
    _watcher: Option<notify::RecommendedWatcher>,
}

impl Watch {
    /// Watch `source`'s directory for the artifact being rebuilt.
    ///
    /// The *directory* rather than the file: a linker replaces an artifact by
    /// rename, which leaves a watch on the old inode watching nothing.
    pub fn new(source: &Path, staging_dir: &Path) -> Result<Self, ReloadError> {
        let staging_err = |source: std::io::Error| ReloadError::Staging {
            path: staging_dir.to_owned(),
            source,
        };
        std::fs::create_dir_all(staging_dir).map_err(staging_err)?;

        let dir = source
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map_or_else(|| PathBuf::from("."), Path::to_owned);
        let file = source.file_name().map(std::ffi::OsString::from);

        let (tx, rx) = mpsc::channel();
        let mut watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
                let Ok(event) = event else { return };
                if !(event.kind.is_modify() || event.kind.is_create()) {
                    return;
                }
                // One artifact in a directory full of them: `cargo build` touches
                // every rlib in `deps/`, and reloading on each would be a rebuild
                // storm rather than a reload.
                let ours = event
                    .paths
                    .iter()
                    .any(|p| p.file_name().map(std::ffi::OsString::from) == file);
                if ours {
                    let _ = tx.send(Instant::now());
                }
            })
            .map_err(|e| ReloadError::Staging {
                path: source.to_owned(),
                source: std::io::Error::other(e),
            })?;
        watcher
            .watch(&dir, notify::RecursiveMode::NonRecursive)
            .map_err(|e| ReloadError::Staging {
                path: dir.clone(),
                source: std::io::Error::other(e),
            })?;

        Ok(Self {
            source: source.to_owned(),
            staging_dir: staging_dir.to_owned(),
            generation: 0,
            pending: None,
            rx,
            _watcher: Some(watcher),
        })
    }

    /// A watcher over no artifact — [`GameLib::absent`]'s companion (§6 M15.1
    /// item 4).
    ///
    /// [`poll`](Self::poll) returns `None` forever: the sender is dropped here,
    /// so the channel is closed and `try_recv` never yields. Infallible, unlike
    /// [`new`](Self::new), because there is nothing to stage and no directory to
    /// fail to watch — which is what lets a launcher hold the same field a
    /// session over a game does.
    ///
    /// [`GameLib::absent`]: crate::reload::GameLib::absent
    #[must_use]
    pub fn absent() -> Self {
        let (_, rx) = mpsc::channel();
        Self {
            source: PathBuf::from(crate::reload::NO_PROJECT),
            staging_dir: PathBuf::new(),
            generation: 0,
            pending: None,
            rx,
            _watcher: None,
        }
    }

    /// The artifact being watched.
    pub fn source(&self) -> &Path {
        &self.source
    }

    /// Whether a rebuild has been seen and not yet loaded.
    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Request the artifact and wait for it: the startup load, which has no
    /// frame loop to be polled from yet.
    ///
    /// Ends in a staged, loaded, verified library or in a named error — a
    /// missing path and a linker that never lets go both terminate through
    /// [`STAGING_PATIENCE`].
    ///
    /// # Safety
    ///
    /// Inherits [`Self::poll`]'s obligations.
    pub unsafe fn block_until_ready(
        &mut self,
        host_api: &'static HostApiV1,
    ) -> Result<Reloaded, ReloadError> {
        self.request();
        let started = Instant::now();
        loop {
            // SAFETY: forwarded to this method's caller, one doc block above.
            if let Some(ready) = unsafe { self.poll(host_api) } {
                return ready;
            }
            if started.elapsed() > FIRST_LOAD_SPIN {
                std::thread::sleep(FIRST_LOAD_NAP);
            }
        }
    }

    /// Act as though the artifact had just been rebuilt — a manual reload key,
    /// and the first load at startup.
    pub fn request(&mut self) {
        // Nothing to load: an absent watcher that took a request would stat a
        // path that is not a path and refuse it once `STAGING_PATIENCE` ran out.
        if self._watcher.is_none() {
            return;
        }
        self.pending.get_or_insert(Pending {
            saved_at: Instant::now(),
            settled_at: None,
            quiet: Duration::ZERO,
        });
    }

    /// Stage and load a rebuilt dylib, if one is ready.
    ///
    /// Returns `None` while nothing has changed, and while a rebuild is still
    /// being written — the wait is not an error and not a log line. Call it from
    /// `reload_check`, which is a tick boundary by construction (§4.1).
    ///
    /// # Safety
    ///
    /// Inherits [`GameLib::load`]'s obligations: this loads and executes whatever
    /// is at the watched path, and `host_api` must outlive every later call into
    /// the dylib.
    pub unsafe fn poll(
        &mut self,
        host_api: &'static HostApiV1,
    ) -> Option<Result<Reloaded, ReloadError>> {
        // Coalesce: one `cargo build` produces a burst of events, and the
        // earliest is the one the save-to-screen clock starts at.
        while let Ok(seen) = self.rx.try_recv() {
            match &mut self.pending {
                Some(pending) => {
                    pending.saved_at = pending.saved_at.min(seen);
                    // A write during the settling wait restarts it — and proves
                    // there is a writer, if this began as a startup request.
                    pending.settled_at = None;
                    pending.quiet = SETTLE_QUIET;
                }
                None => {
                    self.pending = Some(Pending {
                        saved_at: seen,
                        settled_at: None,
                        quiet: SETTLE_QUIET,
                    });
                }
            }
        }

        let pending = self.pending.as_mut()?;
        let stuck = pending.saved_at.elapsed() > STAGING_PATIENCE;

        let stat = std::fs::metadata(&self.source)
            .and_then(|m| Ok((m.len(), m.modified()?)))
            .ok();
        match stat {
            // Held long enough to be believed: fall through and stage it.
            Some(stat)
                if pending.settled_at.is_some_and(|(seen, since)| {
                    seen == stat && since.elapsed() >= pending.quiet
                }) => {}
            Some(stat) if !stuck => {
                // A different size or mtime restarts the clock; the same one
                // keeps the instant it was first seen, or the wait never ends.
                if pending.settled_at.is_none_or(|(seen, _)| seen != stat) {
                    pending.settled_at = Some((stat, Instant::now()));
                }
                return None;
            }
            _ if !stuck => return None,
            // Out of patience. Which failure it is turns on whether the file is
            // *there*: an absent artifact is a wrong path or a build that never
            // ran, and telling that reader about linker races sends them to
            // debug a race that is not happening.
            stat => {
                let pending = self.pending.take()?;
                let detail = match stat {
                    Some(_) => format!(
                        "still being written {:?} after the rebuild was seen",
                        pending.saved_at.elapsed()
                    ),
                    None => "no such artifact — was the game crate built as a dylib?".to_owned(),
                };
                return Some(Err(ReloadError::Staging {
                    path: self.source.clone(),
                    source: std::io::Error::other(detail),
                }));
            }
        }

        let saved_at = pending.saved_at;
        self.generation += 1;
        let staged = self.staged_path(self.generation);
        if let Err(source) = std::fs::copy(&self.source, &staged) {
            // Almost always the linker still holding the file. Clearing the
            // settle marker restarts the quiet period and retries;
            // `STAGING_PATIENCE` is what stops this being forever.
            if !stuck {
                if let Some(pending) = self.pending.as_mut() {
                    pending.settled_at = None;
                }
                return None;
            }
            self.pending = None;
            return Some(Err(ReloadError::Staging {
                path: self.source.clone(),
                source,
            }));
        }

        self.pending = None;
        let started = Instant::now();
        // SAFETY: the caller's obligation, forwarded — `staged` is a byte-for-byte
        // copy of the watched artifact, so its provenance is the watched path's.
        let loaded = unsafe { GameLib::load(&staged, host_api) };
        Some(loaded.map(|lib| Reloaded {
            lib,
            saved_at,
            load_time: started.elapsed(),
        }))
    }

    /// Where generation `n` of the artifact is staged. Numbered rather than
    /// overwritten: the previous copy is still loaded and still executing.
    fn staged_path(&self, generation: u32) -> PathBuf {
        let stem = self
            .source
            .file_stem()
            .unwrap_or_else(|| std::ffi::OsStr::new("game"))
            .to_string_lossy()
            .into_owned();
        let mut name = format!("{stem}-{generation}");
        if let Some(ext) = self.source.extension() {
            name.push('.');
            name.push_str(&ext.to_string_lossy());
        }
        self.staging_dir.join(name)
    }
}

impl std::fmt::Debug for Watch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Watch")
            .field("source", &self.source)
            .field("generation", &self.generation)
            .field("pending", &self.pending.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_watcher_over_no_artifact_has_nothing_to_wait_for() {
        // The launcher's field (§6 M15.1 item 4). A request here would put the
        // watcher into a wait that only `STAGING_PATIENCE` ends, and it would end
        // it with a refusal naming a path that is not a path.
        let mut watch = Watch::absent();
        assert_eq!(watch.source(), Path::new(crate::reload::NO_PROJECT));
        watch.request();
        assert!(
            !watch.is_pending(),
            "an absent watcher took a reload request"
        );
    }
}
