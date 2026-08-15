//! What a player is told when the game does not start (§6 M47).
//!
//! Here rather than in a `#[cfg(test)]` module because §3's shell budget counts
//! `src/`, and a test that costs the shell lines is a test somebody deletes to
//! fit under a cap. The box itself is the operator's — this is the string that
//! goes in it.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use anyhow::Context as _;

fn layered() -> anyhow::Error {
    let inner: anyhow::Result<()> = Err(anyhow::anyhow!("the file is not there"));
    inner
        .context("reading demo_10_tetris.dll")
        .context("opening the game")
        .unwrap_err()
}

/// Three things, and each is missing from a different failure this milestone
/// found: the title, because a box with no name is a box from nowhere; every
/// cause, because the top of an `anyhow` chain is the vaguest line in it; and
/// the log's path, which is the only reason a bug report ever contains anything.
#[test]
fn the_box_names_the_game_the_whole_chain_and_where_the_log_is() {
    let log = std::path::Path::new("C:/Users/p/AppData/Roaming/falling-blocks/log.txt");
    let body = gg_runtime::refusal("Falling Blocks", &layered(), Some(log));
    assert!(
        body.starts_with("Falling Blocks could not start."),
        "{body}"
    );
    for cause in ["opening the game", "demo_10_tetris.dll", "not there"] {
        assert!(
            body.contains(cause),
            "the chain is whole: {cause} in {body}"
        );
    }
    assert!(body.contains("log.txt"), "{body}");
}

/// A crash is not a refusal and the box must not read like one: the session
/// existed, so the sentence a player is actually looking for is how much of it
/// is still there (§6 M48).
#[test]
fn a_crash_says_what_stopped_and_how_much_of_the_session_is_left() {
    let log = std::path::Path::new("C:/Users/p/AppData/Roaming/falling-blocks/log.txt");
    let panicked = "panicked at demos/10-tetris/src/lib.rs:412: index out of bounds";
    let body = gg_runtime::crashed("Falling Blocks", panicked, Some(log), true);
    assert!(
        body.starts_with("Falling Blocks stopped unexpectedly."),
        "{body}"
    );
    assert!(body.contains("index out of bounds"), "{body}");
    assert!(body.contains("Your progress was saved"), "{body}");
    assert!(body.contains("log.txt"), "{body}");
}

/// The half of that sentence which would be a lie. A session that was never
/// checkpointing — a replay, a recording, a run with no project — has nothing
/// on the disk to point at, and telling a player otherwise sends them looking
/// for a file that is not there.
#[test]
fn a_crash_with_nothing_kept_promises_nothing() {
    let body = gg_runtime::crashed("gg-runtime", "panicked at src/lib.rs:1: no", None, false);
    assert!(!body.contains("Your progress"), "{body}");
    assert!(!body.contains("The full log"), "{body}");
    assert!(body.contains("stopped unexpectedly"), "{body}");
}

/// A dev run has a console somebody is already looking at and no log file, so
/// the message must not point at one. The failure this refuses is worse than a
/// missing line: a path that is not there reads as a file the player deleted.
#[test]
fn with_nowhere_to_point_it_points_nowhere() {
    let body = gg_runtime::refusal("gg-runtime", &layered(), None);
    assert!(!body.contains("The full log"), "{body}");
    assert!(
        body.contains("not there"),
        "the failure still arrives: {body}"
    );
}
