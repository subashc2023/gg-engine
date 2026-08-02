//! The panic path end to end (§4.8): hook installed, report composed, file
//! written, log tail inside it.
//!
//! Its own binary, and one test in it: a panic hook is process-global, and a
//! test that installs one must not be sharing a process with tests that do not
//! expect their panics rewritten.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

const PRODUCT: gg_debug::crash::Product = gg_debug::crash::Product {
    name: "gg-crash-test",
    version: "0.1.0",
    tier: "dev",
};

#[test]
fn a_panic_leaves_a_report_carrying_the_message_and_the_log_tail() {
    // Instruments first, hook second — the ordering `install` documents, and
    // the reason the report has a tail to attach at all.
    let _guard = gg_debug::init("info").unwrap();
    tracing::info!(canary = "before-the-crash", "the frame before");
    gg_debug::crash::install(PRODUCT);

    let path = std::env::temp_dir().join(format!("gg-crash-{}.txt", std::process::id()));
    let _ = std::fs::remove_file(&path);
    // The hook runs during the unwind; catching it is what lets the test read
    // what the hook wrote.
    let panicked = std::panic::catch_unwind(|| panic!("deliberate: {}", "a test panic"));
    assert!(panicked.is_err());

    let report = std::fs::read_to_string(&path).unwrap();
    assert!(report.contains("deliberate: a test panic"), "{report}");
    assert!(
        report.contains("gg-crash-test 0.1.0 (dev tier)"),
        "{report}"
    );
    assert!(report.contains("crash_report.rs"), "{report}");
    // The whole point of the file over the terminal line.
    assert!(report.contains("canary=before-the-crash"), "{report}");
    std::fs::remove_file(&path).unwrap();
}
