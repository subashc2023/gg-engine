//! The player's own bytes, and what survives a process that never came back
//! (§6 M48). In `tests/` for `refusal.rs`'s reason: §3's shell budget counts
//! `src/`, and a test that costs the shell lines is one somebody deletes to fit
//! under a cap.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use gg_runtime::player;

/// A scratch directory of this test's own. Under `target/`, because a test that
/// wrote into the operator's data directory would be a test with a side effect
/// (§6 M42's rule, one crate over).
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The ordinary case, and the two things that make it atomic rather than merely
/// correct: the bytes arrive, and the sibling the rename came from is gone —
/// a temp left behind is a file the player's folder keeps forever and the next
/// `replace` would append to.
#[test]
fn a_replaced_file_holds_the_new_bytes_and_leaves_no_sibling() {
    let dir = scratch("replace");
    let path = dir.join("progress.ggsave");
    player::replace(&path, b"first").unwrap();
    player::replace(&path, b"second").unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"second");
    assert!(
        !player::temp(&path).exists(),
        "the sibling survived the rename: {}",
        player::temp(&path).display()
    );
}

/// The failure this exists for, and the only one a test can provoke without a
/// power cut: the write does not finish. What must hold is that the *old* file
/// is untouched — before M48 the destination was already truncated by the time
/// anything could fail, so a full disk cost a player their save rather than
/// their last five seconds.
///
/// Provoked by making the sibling a directory, which `File::create` cannot
/// open. That is the one interruption reachable from here, and it lands in
/// exactly the window the milestone is about.
#[test]
fn a_write_that_fails_leaves_the_old_file_whole() {
    let dir = scratch("refused");
    let path = dir.join("progress.ggsave");
    player::replace(&path, b"the session a player already had").unwrap();
    std::fs::create_dir(player::temp(&path)).unwrap();
    // Not asserted to be an error: the pre-M48 write *succeeds* here, and the
    // defect is where its bytes went — so the destination is the whole claim
    // and the outcome is context for reading the failure.
    let outcome = player::replace(&path, b"the one that never landed");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        b"the session a player already had",
        "the replacement ({outcome:?}) reached the destination before it could fail — this is \
         the window where an interrupted write leaves a player half a file"
    );
}

/// One offer, and the two claims that make a checkpoint worth having: the bytes
/// reach the disk without the offering thread waiting for one, and the panic
/// hook can tell there was a session to promise (§6 M48).
///
/// The join is `drop`'s, which is the same edge the shell uses before its exit
/// write. Without it this test races the file it reads — and so would the shell.
///
/// **The promise is `false` until bytes land** (§6 M54). Until that milestone it
/// was true as soon as the *thread* existed, so the sentence a player reads in
/// the crash box — "your progress was saved up to about five seconds before
/// this" — was a claim about a thread having spawned rather than about anything
/// being on their disk.
#[test]
fn a_checkpoint_lands_and_says_a_session_is_being_kept() {
    let dir = scratch("checkpoint");
    let path = dir.join("progress.ggsave");
    let checkpoint = player::Checkpoint::new(path.clone(), player::PROGRESS);
    assert!(
        !player::checkpointing(),
        "a writer that has written nothing is promising a player a file that does not exist"
    );
    checkpoint.offer(b"the session as it stood".to_vec());
    drop(checkpoint);
    assert_eq!(std::fs::read(&path).unwrap(), b"the session as it stood");
    assert!(
        player::checkpointing(),
        "the bytes landed and the hook still has nothing to promise"
    );
}

/// What the shell knows about the player's disk when the session ends (§6 M54),
/// and the one thing a single latch gets wrong.
///
/// Every claim in one test on purpose: the record is process-wide, so under a
/// runner that shares a process between tests — which `nextest` is not — two of
/// these would be one.
#[test]
fn a_verdict_is_per_file_and_not_the_last_write_to_happen() {
    let refused = || Err(std::io::Error::other("the disk said no"));
    let save = Path::new("progress.ggsave");
    let prefs = Path::new("settings.cfg");

    player::note(save, &refused());
    assert!(
        player::verdict().is_some(),
        "a refused save is not something a session may exit quietly on"
    );
    // The ordering the shell actually has: preferences are written before the
    // session at exit, so a latch cleared by the last write to land would call
    // this run fine.
    player::note(prefs, &Ok(()));
    let verdict =
        player::verdict().expect("a preference that landed vouched for a save that did not");
    assert_eq!(verdict.failures, 1);
    assert_eq!(verdict.writes, 1);
    assert!(
        verdict.reason.contains("the disk said no"),
        "the verdict carries no reason to print: {}",
        verdict.reason
    );
    // And a file that recovers takes its own failure back with it: a scanner
    // holding one write is not a disk that refuses.
    player::note(save, &Ok(()));
    assert!(
        player::verdict().is_none(),
        "a write that eventually landed still reported a loss"
    );
}

/// Offered faster than any disk will take them, which is the sim thread's real
/// case: a headless tier ticks as fast as it can and a 60 Hz session hits an
/// interval whenever the encode says so, not whenever the drive is ready.
///
/// What is asserted is *drop*, not order — a checkpoint is a level, so a refused
/// offer is one the next interval supersedes. The file must therefore hold one
/// of the whole values offered and never a splice of two, which is the property
/// [`player::replace`] is for, arriving here through the thread that uses it.
#[test]
fn offers_a_disk_cannot_keep_up_with_are_dropped_and_never_spliced() {
    let dir = scratch("flood");
    let path = dir.join("progress.ggsave");
    let checkpoint = player::Checkpoint::new(path.clone(), player::PROGRESS);
    let offered: Vec<Vec<u8>> = (0..64u32)
        .map(|tick| format!("the session at tick {tick}").into_bytes())
        .collect();
    for bytes in &offered {
        checkpoint.offer(bytes.clone());
    }
    drop(checkpoint);
    let landed = std::fs::read(&path).unwrap();
    assert!(
        offered.contains(&landed),
        "the file is not one of the sessions offered: {:?}",
        String::from_utf8_lossy(&landed)
    );
}
