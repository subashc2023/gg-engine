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
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

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

/// Writers of the player's **progress** whose *last* write reached the disk
/// (§6 M81).
///
/// Scoped to progress and counted rather than latched, and both halves are the
/// same finding. Two `Checkpoint`s share this module — the session's and the
/// preferences' — and one `LANDED` flag between them could not tell "the save
/// landed" from "the preferences beside it did", which is precisely the
/// accident of ordering [`OUTCOMES`] already refuses for the exit verdict (§6
/// M54), reaching the crash box four milestones later. Its `RUNNING` half was
/// written at *construction* by whichever writer was built last, so a session
/// with a save and no preferences file, or the reverse, answered about the
/// wrong one.
///
/// A count and not a flag because a writer can be replaced while another is
/// still retiring, and because the transitions are what make it a verdict on
/// the last write rather than a latch on the first.
///
/// Not locked: the reader is a panic hook, and a `Mutex` there deadlocks
/// whenever the panicking thread is the one holding it.
static PROGRESS_LANDED: AtomicUsize = AtomicUsize::new(0);

/// Whether the player's progress is on their disk as of this session's last
/// checkpoint — bytes that landed, not a writer that exists.
///
/// The distinction is the one sentence in the crash box a player acts on
/// ([`crate::crashed`]): on a disk that refuses, "your progress was saved up to
/// about five seconds before this" is false, and it is false in the direction
/// that stops them from doing anything about it. **Preferences do not count** —
/// a `settings.cfg` that landed says nothing about a game a player lost. A
/// *retired* writer does: the bytes it wrote are still there, which is why the
/// count survives the thread and only a later failure takes it back.
#[must_use]
pub fn checkpointing() -> bool {
    PROGRESS_LANDED.load(Relaxed) > 0
}

/// [`Checkpoint::new`]'s second argument, named at the call site: the player's
/// session, which is the file [`checkpointing`] answers about.
pub const PROGRESS: bool = true;

/// The other one — their preferences. Counted by [`note`] like every other
/// write, and deliberately invisible to [`checkpointing`].
pub const PREFERENCES: bool = false;

impl Checkpoint {
    /// Start the writer for `path`, keeping [`PROGRESS`] or [`PREFERENCES`].
    /// Failing to spawn is not fatal — the session then behaves as every
    /// session did before M48, and the exit write still lands.
    #[must_use]
    pub fn new(path: PathBuf, progress: bool) -> Self {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(1);
        let thread = std::thread::Builder::new()
            .name("gg-checkpoint".to_owned())
            .spawn(move || {
                // This writer's own contribution, adjusted on transitions so
                // the counter stays a population rather than becoming a tally
                // of every write ever made.
                let mut counted = false;
                while let Ok(bytes) = rx.recv() {
                    let wrote = replace(&path, &bytes);
                    if progress && wrote.is_ok() != counted {
                        counted = wrote.is_ok();
                        if counted {
                            PROGRESS_LANDED.fetch_add(1, Relaxed);
                        } else {
                            PROGRESS_LANDED.fetch_sub(1, Relaxed);
                        }
                    }
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

/// The first unit tests in this crate (§6 M81), and what made them writable is
/// §3's budget no longer counting a `#[cfg(test)]` item.
///
/// All four are about [`checkpointing`], which is one `bool` reaching one
/// sentence in the crash box, and which no other gate can reach: `xtask reload
/// --crash` grades the *file* a killed process left behind, and this is what
/// the box says while the process is still dying.
///
/// The state is process-global and nextest gives each test its own process, so
/// these need no ordering between them. `cargo test` would thread them and they
/// would fight; the tree's runner is nextest everywhere.
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// A directory this test owns, named for it so a failure leaves evidence
    /// rather than a collision.
    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("gg-player-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Spin until the writer has caught up, or give up. There is nothing to
    /// wait *on*: a checkpoint's whole point is that no caller blocks on the
    /// disk, and `drop` is the only join there is.
    fn settles(done: impl Fn() -> bool) -> bool {
        (0..2000).any(|_| {
            std::thread::sleep(std::time::Duration::from_millis(1));
            done()
        })
    }

    /// The promise itself, which is what most of these are waiting for.
    fn settles_at(want: bool) -> bool {
        settles(|| checkpointing() == want)
    }

    /// The dangerous direction, and the reason the counter is scoped: a
    /// `settings.cfg` that landed is not a game that was saved. One flag
    /// between the two writers said it was.
    #[test]
    fn preferences_landing_is_not_progress_being_saved() {
        let dir = scratch("prefs");
        let prefs = Checkpoint::new(dir.join("settings.cfg"), PREFERENCES);
        prefs.offer(b"prefs".to_vec());
        drop(prefs); // Joins, so the write has happened by here.
        assert!(!checkpointing(), "preferences promised a saved game");
        // And a progress writer in the same session does say so.
        let session = Checkpoint::new(dir.join("progress.ggsave"), PROGRESS);
        session.offer(b"session".to_vec());
        assert!(settles_at(true), "a landed save said nothing");
    }

    /// The same defect with its sign flipped: `RUNNING` was written at
    /// construction by whichever writer was built last, so a session whose
    /// preferences write failed reported a saved game as lost.
    #[test]
    fn a_preferences_writer_neither_gives_nor_takes_the_promise() {
        let dir = scratch("both");
        let session = Checkpoint::new(dir.join("progress.ggsave"), PROGRESS);
        session.offer(b"session".to_vec());
        assert!(settles_at(true));
        let blocked = dir.join("wall");
        std::fs::write(&blocked, b"not a directory").unwrap();
        let prefs = Checkpoint::new(blocked.join("settings.cfg"), PREFERENCES);
        prefs.offer(b"prefs".to_vec());
        drop(prefs);
        assert!(checkpointing(), "a refused settings file unsaved the game");
    }

    /// The refusal M54 is about, at the one place M54 did not reach: a writer
    /// whose last write the disk refused must promise nothing. The parent is a
    /// **file**, which is a real OS error on both hosts and needs no
    /// privileges.
    #[test]
    fn a_refused_write_promises_nothing() {
        let dir = scratch("refused");
        let blocked = dir.join("wall");
        std::fs::write(&blocked, b"not a directory").unwrap();
        let session = Checkpoint::new(blocked.join("progress.ggsave"), PROGRESS);
        session.offer(b"session".to_vec());
        drop(session);
        assert!(!checkpointing(), "a refused disk promised a saved game");
        assert!(verdict().is_some(), "the refusal reached no verdict");
    }

    /// And a writer that recovers is believed again — a population and not a
    /// latch, which is [`note`]'s own rule one level up.
    #[test]
    fn a_writer_that_recovers_is_believed_again() {
        let dir = scratch("recovers");
        let blocked = dir.join("wall");
        std::fs::write(&blocked, b"not a directory").unwrap();
        let session = Checkpoint::new(blocked.join("progress.ggsave"), PROGRESS);
        session.offer(b"first".to_vec());
        // Waiting on the *refusal*, not on the promise: the promise is already
        // false before the first write and waiting for it would prove nothing.
        assert!(settles(|| verdict().is_some()), "the write never failed");
        assert!(!checkpointing());
        std::fs::remove_file(&blocked).unwrap();
        session.offer(b"second".to_vec());
        assert!(settles_at(true), "a recovered disk was still disbelieved");
    }
}
