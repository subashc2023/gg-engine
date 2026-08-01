//! The hot path (§4.4): watch a shader directory, recompile on save.
//! Dev-tier lab equipment by construction — this module exists only under the
//! `hot-reload` feature, and the dist gate proves `notify` absent from
//! shipping binaries (§5.8).
//!
//! The watcher thread only forwards event timestamps; compilation happens in
//! [`ShaderWatcher::poll`] on the caller's thread, so the consumer decides
//! when a rebuild is safe (behind the frame timeline, never mid-frame) and
//! the save-to-screen clock starts at the OS event.

use crate::{CompiledModule, ShaderError, compile_module};
use notify::Watcher as _;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

/// A successful recompile, with the timing the <500 ms exit criterion measures.
pub struct Recompiled {
    /// The freshly compiled module.
    pub module: CompiledModule,
    /// When the file event arrived — the save-to-screen clock's zero.
    pub saved_at: Instant,
    /// How long compilation itself took.
    pub compile_time: std::time::Duration,
}

/// Watches one `.slang` module and recompiles it on save.
pub struct ShaderWatcher {
    search_dir: String,
    module_name: String,
    rx: mpsc::Receiver<Instant>,
    // Held for its Drop: dropping the watcher stops the OS watch.
    _watcher: notify::RecommendedWatcher,
}

impl ShaderWatcher {
    /// Watch `search_dir` for changes to any `.slang` file and recompile
    /// `module_name` (the same arguments [`compile_module`] takes).
    pub fn new(search_dir: &str, module_name: &str) -> Result<Self, ShaderError> {
        let (tx, rx) = mpsc::channel();
        let mut watcher =
            notify::recommended_watcher(move |event: notify::Result<notify::Event>| {
                let Ok(event) = event else { return };
                let slang_touched = event
                    .paths
                    .iter()
                    .any(|p: &PathBuf| p.extension().is_some_and(|e| e == "slang"));
                if slang_touched && (event.kind.is_modify() || event.kind.is_create()) {
                    let _ = tx.send(Instant::now());
                }
            })
            .map_err(|e| ShaderError::BadPath(format!("watcher: {e}")))?;
        watcher
            .watch(Path::new(search_dir), notify::RecursiveMode::NonRecursive)
            .map_err(|e| ShaderError::BadPath(format!("watch {search_dir}: {e}")))?;
        Ok(Self {
            search_dir: search_dir.to_string(),
            module_name: module_name.to_string(),
            rx,
            _watcher: watcher,
        })
    }

    /// Non-blocking: if the file changed since the last poll, recompile and
    /// return the result. Editors fire bursts of events per save; the burst is
    /// drained and compiled once, keyed to its earliest event so the reported
    /// save-to-screen time is honest.
    pub fn poll(&mut self) -> Option<Result<Recompiled, ShaderError>> {
        let mut saved_at: Option<Instant> = None;
        while let Ok(at) = self.rx.try_recv() {
            saved_at = Some(saved_at.map_or(at, |prev| prev.min(at)));
        }
        let saved_at = saved_at?;
        let start = Instant::now();
        Some(
            compile_module(&self.search_dir, &self.module_name).map(|module| Recompiled {
                module,
                saved_at,
                compile_time: start.elapsed(),
            }),
        )
    }
}
