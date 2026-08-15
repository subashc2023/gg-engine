//! The player's own files, and the two ways a session is lost (§6 M48).
//!
//! `progress.ggsave`, `settings.cfg` and a `--save` are all *replacements* of
//! bytes somebody already has, so one failure arrives at two scales: a write
//! interrupted halfway leaves half a file, and a session written only at exit
//! leaves nothing at all when the process never reaches one. [`replace`]
//! answers the first and [`Checkpoint`] the second.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};

use tracing::{debug, warn};

/// How often a running session reaches the disk, in seconds of sim time —
/// multiplied by the tick rate at the call site, so a game at another Hz keeps
/// the interval rather than the tick count.
///
/// Five, and what sets it is neither the disk nor the encode: it is what a
/// player would agree to replay. A crash costs them the seconds since the last
/// one, and a Tetris board five seconds stale is the same board.
pub const CHECKPOINT_SECONDS: u64 = 5;

/// What the player's disk has actually said, this process (§6 M54).
///
/// Counted rather than repeated, and the first reason kept: a disk that refuses
/// refuses every interval, so a `warn` per attempt regrows exactly the log §6
/// M51 cut out of the success path.
static FAILURES: AtomicU64 = AtomicU64::new(0);
static WRITES: AtomicU64 = AtomicU64::new(0);
static REASON: OnceLock<String> = OnceLock::new();

/// The last outcome of each player file, `true` meaning it failed. **Per path
/// and not one flag**, which is the difference between a verdict and an
/// accident of ordering: settings are written before the session at exit, so a
/// single latch cleared by the last write would report a refused save as fine
/// whenever the preferences beside it happened to land.
///
/// A `Vec` because there are three of these and a lookup is a memcmp — and a
/// lock the sim thread never takes, since the writes are the checkpoint
/// thread's and the exit path's.
static OUTCOMES: std::sync::Mutex<Vec<(PathBuf, bool)>> = std::sync::Mutex::new(Vec::new());

/// A session's disk, as of its last write. Built only when there is something
/// to say — see [`verdict`].
pub struct Verdict {
    /// Writes that failed. Every player file, not one of them: what a player
    /// needs to hear is about the directory.
    pub failures: u64,
    /// Writes that landed. Zero and nonzero are different sentences — nothing
    /// was saved, against some of it was.
    pub writes: u64,
    /// The first failure's own words, which name the cause the way the OS did.
    pub reason: String,
}

/// Record one player-file write, whatever it was for.
///
/// **Every one of them goes through here.** How often a failing disk is allowed
/// to speak, and whether the session's last words mention it, is one policy;
/// three call sites each logging their own failure is three policies, and the
/// one thing none of them can see is that the *directory* is the problem.
pub fn note(path: &Path, result: &std::io::Result<()>) {
    match result {
        Ok(()) => WRITES.fetch_add(1, Relaxed),
        Err(e) => {
            if REASON.set(e.to_string()).is_ok() {
                warn!(path = %path.display(), error = %e,
                      "the player's disk refused a write - counted from here, said once");
            }
            FAILURES.fetch_add(1, Relaxed)
        }
    };
    // Overwritten rather than latched: a scanner holding one file for one
    // interval is not a disk that refuses, and telling a player their progress
    // is gone when the next write landed is its own kind of wrong.
    if let Ok(mut outcomes) = OUTCOMES.lock() {
        let failed = result.is_err();
        match outcomes.iter_mut().find(|(seen, _)| seen == path) {
            Some((_, last)) => *last = failed,
            None => outcomes.push((path.to_path_buf(), failed)),
        }
    }
}

/// What to tell the player on the way out, or `None` when the disk did its job.
///
/// `Some` when any player file's *last* write failed: a file that recovered
/// lost nothing, and a box about a write that eventually landed teaches a
/// player to ignore the next one.
#[must_use]
pub fn verdict() -> Option<Verdict> {
    let failing = OUTCOMES
        .lock()
        .map(|outcomes| outcomes.iter().any(|(_, failed)| *failed))
        .unwrap_or(false);
    failing.then(|| Verdict {
        failures: FAILURES.load(Relaxed),
        writes: WRITES.load(Relaxed),
        reason: REASON.get().cloned().unwrap_or_default(),
    })
}

/// Replace `path` with `bytes`, atomically: a reader sees the old file or the
/// new one, never a prefix of it.
///
/// The temp is a **sibling** — `rename` is atomic only within a filesystem, and
/// a system temp directory is routinely a different one. It is removed when the
/// write fails, so a refusal leaves nothing behind either; a process *killed*
/// between the two steps leaves one, which the next run ignores by name.
///
/// `sync_all` before the rename rather than after, which is the whole claim:
/// without it the directory entry can reach the platter ahead of the data, and
/// the crash that follows produces exactly the truncated file this exists to
/// prevent.
pub fn replace(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent().filter(|dir| !dir.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = temp(path);
    let written = (|| {
        let mut file = std::fs::File::create(&tmp)?;
        file.write_all(bytes)?;
        file.sync_all()
    })();
    // `rename` over an existing file replaces it on both hosts — POSIX says so
    // and Windows' `MoveFileEx` is given `REPLACE_EXISTING` by `std`.
    match written.and_then(|()| std::fs::rename(&tmp, path)) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// The sibling [`replace`] writes through. A fixed name and not a unique one:
/// two processes sharing one player directory is not a thing this engine
/// supports, and a random name would leave a fresh orphan behind every crash
/// instead of reusing the one.
#[must_use]
pub fn temp(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".tmp");
    PathBuf::from(name)
}

/// The session, written while it is still a session (§6 M48).
///
/// Before this every player file was written between the loop ending and the
/// app being consumed, so a process that never reached that line — a panic
/// under `panic = "abort"`, a kill, a power cut — left a player exactly what
/// they had before they started.
///
/// **The encode is the game thread's and the disk is not.** Bytes come off the
/// world, so nobody else can build them; `sync_all` is the reason they are then
/// handed away, because its latency belongs to the player's drive and no
/// measurement of this desk would bound it for theirs.
///
/// A queue of one, and a full queue *drops* rather than blocks: a checkpoint is
/// a level and not an event, so the interval that follows carries a strictly
/// better answer than the one refused. What that buys is the property worth
/// having — the sim thread never waits for a disk, whatever the disk is doing.
pub struct Checkpoint {
    tx: Option<std::sync::mpsc::SyncSender<Vec<u8>>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// Set once a writer exists, read by the panic hook — which is installed before
/// there is a session to ask and must not promise a player a file that this run
/// was never going to write (§6 M48).
static RUNNING: AtomicBool = AtomicBool::new(false);

/// Whether a checkpoint has actually reached the disk, and the last one still
/// did. A thread that spawned is not a file that exists (§6 M54).
static LANDED: AtomicBool = AtomicBool::new(false);

/// Whether this session is checkpointing — meaning bytes have landed, not that
/// a writer exists.
///
/// The distinction is the one sentence in the crash box a player acts on
/// ([`crate::crashed`]): on a disk that refuses, "your progress was saved up to
/// about five seconds before this" is false, and it is false in the direction
/// that stops them from doing anything about it.
#[must_use]
pub fn checkpointing() -> bool {
    RUNNING.load(Relaxed) && LANDED.load(Relaxed)
}

impl Checkpoint {
    /// Start the writer for `path`. Failing to spawn is not fatal — the session
    /// then behaves as every session did before M48, and the exit write still
    /// lands.
    #[must_use]
    pub fn new(path: PathBuf) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(1);
        let thread = std::thread::Builder::new()
            .name("gg-checkpoint".to_owned())
            .spawn(move || {
                while let Ok(bytes) = rx.recv() {
                    let wrote = replace(&path, &bytes);
                    LANDED.store(wrote.is_ok(), Relaxed);
                    note(&path, &wrote);
                    // `debug`, not `info`, and the level is the whole point (§6
                    // M51). This fires every `CHECKPOINT_SECONDS` for as long as
                    // a session lasts — twice, since prefs checkpoint beside the
                    // save — so at `info` a twenty minute game writes 480 lines
                    // saying nothing happened, and an evening's writes megabytes.
                    // In a tier without `debug-tools` that log is a *file* on the
                    // player's disk (§6 M47) and nothing rotates it. The failure
                    // is `note`'s and is said once for the same reason (§6 M54);
                    // it is never fatal, since the session is still playable and
                    // the exit write may still land.
                    if wrote.is_ok() {
                        debug!(path = %path.display(), bytes = bytes.len(), "checkpoint written");
                    }
                }
            })
            .inspect_err(
                |e| warn!(error = %e, "no checkpoint thread — this session is exit-write only"),
            )
            .ok();
        RUNNING.store(thread.is_some(), std::sync::atomic::Ordering::Relaxed);
        Self {
            tx: thread.is_some().then_some(tx),
            thread,
        }
    }

    /// Offer this tick's session to the writer, or drop it if the last one has
    /// not landed. Never blocks.
    pub fn offer(&self, bytes: Vec<u8>) {
        if let Some(tx) = &self.tx {
            let _ = tx.try_send(bytes);
        }
    }
}

impl Drop for Checkpoint {
    /// Hang up, then wait. **Dropped before the exit write, never after**: a
    /// checkpoint still in flight would land on top of the newer bytes the exit
    /// wrote and quietly roll the session back by up to one interval.
    fn drop(&mut self) {
        self.tx = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
