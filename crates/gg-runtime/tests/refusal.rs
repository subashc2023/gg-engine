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

/// The third box, and the only one whose subject is a session that went
/// perfectly (§6 M54): the game played, and none of it reached the disk.
///
/// What it has to carry is the *directory*, because that is the only part a
/// player can act on — moving the folder out of Program Files is a thing they
/// can do, and diagnosing `os error 5` is not.
#[test]
fn a_session_that_saved_nothing_says_where_it_was_trying_to_write() {
    let dir = std::path::Path::new("C:/Program Files/falling-blocks");
    let verdict = gg_runtime::player::Verdict {
        failures: 12,
        writes: 0,
        reason: "Access is denied. (os error 5)".to_owned(),
    };
    let body = gg_runtime::unsaved("Falling Blocks", Some(dir), &verdict);
    assert!(
        body.starts_with("Falling Blocks could not save your progress."),
        "{body}"
    );
    assert!(body.contains("C:/Program Files/falling-blocks"), "{body}");
    assert!(body.contains("os error 5"), "{body}");
    assert!(
        body.contains("Nothing from this session was saved"),
        "{body}"
    );
}

/// The same failure after a session that *had* been saving, which is a
/// different evening and must not read like the one above: everything up to the
/// last checkpoint is still on the disk and still worth reopening.
#[test]
fn a_session_that_saved_some_of_itself_says_so() {
    let verdict = gg_runtime::player::Verdict {
        failures: 3,
        writes: 40,
        reason: "There is not enough space on the disk. (os error 112)".to_owned(),
    };
    let body = gg_runtime::unsaved("Falling Blocks", None, &verdict);
    assert!(body.contains("Some of this session was saved"), "{body}");
    assert!(!body.contains("Nothing from this session"), "{body}");
    assert!(!body.contains("It was writing to"), "{body}");
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

/// Bring-up is the one refusal whose body is a *paragraph* rather than a phrase
/// (§6 M55): `gg-rhi` writes several sentences and a parenthesized Vulkan code,
/// because §3 bans the tokens everywhere else and nothing downstream can turn
/// `ERROR_INCOMPATIBLE_DRIVER` into advice. This is the half that must not
/// mangle it — the title, a blank line, then the body exactly as written, with
/// no `caused by:` inserted into the middle of a sentence.
///
/// A stand-in error and not the real one: the words are graded where they are
/// written (`gg-rhi`'s `refusal.rs`), and importing the renderer here to restate
/// them would only prove that two copies of a string match.
#[test]
fn a_paragraph_from_below_reaches_the_box_unbroken() {
    let paragraph = "Vulkan is not installed on this machine.\n\n\
                     This game needs a graphics card with Vulkan 1.3 support.\n\n\
                     (LoadLibraryExW failed)";
    let body = gg_runtime::refusal("Falling Blocks", &anyhow::anyhow!(paragraph), None);
    assert_eq!(
        body,
        format!("Falling Blocks could not start.\n\n{paragraph}"),
        "a box is read standing up: nothing is added between the title and the advice"
    );
    // The `?` at `App::attach` makes an `anyhow` root and not a link, so a
    // bring-up refusal has no chain — the day something adds context to it, this
    // says so rather than the player finding the paragraph pushed down the box.
    assert!(!body.contains("caused by:"), "{body}");
}
