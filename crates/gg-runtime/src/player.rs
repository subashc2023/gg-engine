//! The player's own files, and the two ways a session is lost (§6 M48).
//!
//! `progress.ggsave`, `settings.cfg` and a `--save` are all *replacements* of
//! bytes somebody already has, so one failure arrives at two scales: a write
//! interrupted halfway leaves half a file, and a session written only at exit
//! leaves nothing at all when the process never reaches one. [`replace`]
//! answers the first and [`Checkpoint`] the second.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use tracing::{debug, warn};

/// How often a running session reaches the disk, in seconds of sim time —
/// multiplied by the tick rate at the call site, so a game at another Hz keeps
/// the interval rather than the tick count.
///
/// Five, and what sets it is neither the disk nor the encode: it is what a
/// player would agree to replay. A crash costs them the seconds since the last
/// one, and a Tetris board five seconds stale is the same board.
pub const CHECKPOINT_SECONDS: u64 = 5;

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
static RUNNING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether this session is checkpointing.
#[must_use]
pub fn checkpointing() -> bool {
    RUNNING.load(std::sync::atomic::Ordering::Relaxed)
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
                    match replace(&path, &bytes) {
                        // `debug`, not `info`, and the level is the whole point
                        // (§6 M51). This fires every `CHECKPOINT_SECONDS` for as
                        // long as a session lasts — twice, since prefs
                        // checkpoint beside the save — so at `info` a twenty
                        // minute game writes 480 lines saying nothing happened,
                        // and an evening's writes megabytes. In a tier without
                        // `debug-tools` that log is a *file* on the player's
                        // disk (§6 M47) and nothing rotates it. What a bug
                        // report needs from this path is the failure below,
                        // which is still `warn`, and the cadence, which is
                        // `xtask reload --crash`'s to ask for.
                        Ok(()) => {
                            debug!(path = %path.display(), bytes = bytes.len(), "checkpoint written")
                        }
                        // Never fatal: the session is still playable and the
                        // exit write may still land. A disk that is full says
                        // so once per interval, which is the honest cadence.
                        Err(e) => {
                            warn!(path = %path.display(), error = %e, "checkpoint not written")
                        }
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
