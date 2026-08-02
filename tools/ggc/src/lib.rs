//! `ggc` (§4.6) — the asset compiler, as a library so its gates can call it.
//!
//! The binary beside this is the CLI and nothing else. Everything that parses a
//! source format lives here and in no other crate, which is what makes §2's
//! "the runtime never parses JSON" a property of the dependency graph rather
//! than a rule someone has to remember.

pub mod build;
pub mod import;
pub mod texture;
pub mod watch;
