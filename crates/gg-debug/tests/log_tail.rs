//! The log tail (§4.8) — what a crash report attaches.
//!
//! One test, not several: the tail is process-global and installing the
//! subscriber is a once-per-process act, so two tests in one binary would race
//! over whose lines the ring holds.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[test]
fn the_tail_holds_the_most_recent_lines_and_forgets_the_rest() {
    gg_debug::init("debug").expect("first subscriber in this process");
    assert!(
        gg_debug::init("debug").is_err(),
        "a second subscriber would mean two answers to where a line went"
    );

    tracing::info!(entities = 3, "world adopted");
    let tail = gg_debug::tail();
    let line = tail.last().expect("the line just logged");
    // Message and fields both: a tail that kept only the sentence would drop the
    // numbers, which are usually the reason the report is being read.
    assert!(line.contains("world adopted"), "{line}");
    assert!(line.contains("entities=3"), "{line}");
    assert!(line.contains("INFO"), "{line}");

    // Bounded: a process that logs all day must not hold all day's logs.
    for i in 0..1000 {
        tracing::debug!(i, "filler");
    }
    let tail = gg_debug::tail();
    assert!(tail.len() <= 256, "the ring grew to {}", tail.len());
    assert!(
        tail.last().is_some_and(|l| l.contains("i=999")),
        "the newest line is the one kept: {:?}",
        tail.last()
    );
    assert!(
        !tail.iter().any(|l| l.contains("world adopted")),
        "and the oldest is the one dropped"
    );
}
