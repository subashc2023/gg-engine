//! `ggc watch` (§4.6) — rebuild the pack whenever the source tree changes.
//!
//! The dev half of the asset loop: an artist saves a PNG, this notices, the
//! incremental build recompiles only what that file produced, and the pack is
//! replaced by rename — which is what makes the runtime's mapping see a new
//! inode rather than bytes changing underneath it ([`gg_assets::Pack::open`]).
//!
//! # The settle rule is wall-clock, and that is the interesting part
//!
//! One save is a burst of events, and a large source file is still being
//! written when the first of them arrives. [`next_change`] therefore waits for
//! the tree to be *quiet* for [`SETTLE_QUIET`] rather than acting on the first
//! event or counting them — the same rule `gg_core`'s dylib watcher arrived at,
//! for the same reason: a count is a proxy for time that stops being one the
//! moment anything changes how fast the loop around it runs.

use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{Context, Result};
use notify::Watcher as _;

use crate::build;

/// How long the tree must be quiet before a rebuild starts. Long enough to
/// outlast an editor writing a texture, short enough to feel immediate.
const SETTLE_QUIET: Duration = Duration::from_millis(120);

/// Build once, then rebuild on every change until interrupted.
///
/// Never returns on its own: a watch is a session, and the way out is Ctrl-C.
/// A build that fails is logged and the watch continues — a half-saved glTF is
/// a normal event in an editor, and exiting on one would mean restarting the
/// watcher every time an artist hits save mid-write.
pub fn watch(source: &Path, out: &Path) -> Result<()> {
    report(build::build(source, out), out);

    let (tx, rx) = mpsc::channel();
    let mut watcher = notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
        let Ok(event) = event else { return };
        // Access events are not changes: some editors and every backup tool
        // read the tree, and rebuilding on a read is a loop that never idles.
        if event.kind.is_access() {
            return;
        }
        let _ = tx.send(());
    })
    .context("could not start a file watcher")?;
    // Recursive: a source tree is directories of glTF and images, and the file
    // that changed is as likely to be three levels down as at the root.
    watcher
        .watch(source, notify::RecursiveMode::Recursive)
        .with_context(|| format!("could not watch {}", source.display()))?;

    tracing::info!(source = %source.display(), out = %out.display(), "ggc: watching");
    while next_change(&rx).is_some() {
        report(build::build(source, out), out);
    }
    Ok(())
}

/// Block until something changed *and* the tree has since been quiet for
/// [`SETTLE_QUIET`]. `None` once the watcher is gone.
///
/// The drain loop is what turns one save's burst of events into one rebuild,
/// and what waits out a file still being written.
fn next_change(rx: &mpsc::Receiver<()>) -> Option<()> {
    rx.recv().ok()?;
    while rx.recv_timeout(SETTLE_QUIET).is_ok() {}
    Some(())
}

/// Log what a rebuild did. Errors are reported, never fatal — see [`watch`].
fn report(result: Result<build::Stats>, out: &Path) {
    match result {
        Ok(stats) => tracing::info!(
            assets = stats.assets,
            compiled = stats.compiled,
            reused = stats.reused,
            textures = stats.textures,
            pack = %out.display(),
            "ggc: built"
        ),
        Err(error) => tracing::error!(error = %format!("{error:#}"), "ggc: build failed"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    /// One save is a burst; one burst is one rebuild. Without the drain loop
    /// this would return five times and recompile the tree five times over.
    #[test]
    fn a_burst_of_events_settles_into_one_change() {
        let (tx, rx) = mpsc::channel();
        let sender = thread::spawn(move || {
            for _ in 0..5 {
                if tx.send(()).is_err() {
                    return;
                }
                thread::sleep(SETTLE_QUIET / 4);
            }
            // Then quiet, and then one more burst — two changes in total.
            thread::sleep(SETTLE_QUIET * 3);
            let _ = tx.send(());
        });

        assert!(next_change(&rx).is_some(), "the first burst");
        assert!(next_change(&rx).is_some(), "the second, after the quiet");
        assert!(
            next_change(&rx).is_none(),
            "the sender is gone and the watch ends rather than spinning"
        );
        assert!(sender.join().is_ok(), "the sender thread panicked");
    }
}
