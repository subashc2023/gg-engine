//! `gg-shaders` — Slang compilation, reflection, and codegen (§4.4). M2 scope:
//! the **offline path** (compile to SPIR-V + reflect push-constant layouts,
//! consumed by `xtask shaders` which freezes them into generated Rust), and —
//! behind the `hot-reload` feature — the **hot path** (a file watcher that
//! recompiles on save, dev tier only; the dist gate proves its absence).
//!
//! Compilation is in-process through the `shader-slang` bindings (linking the
//! Vulkan SDK's Slang) — that is what makes hot reload fast and is the only
//! way to reach the reflection API. `slangc` remains the named CLI fallback
//! should the bindings regress (§8); the spike test is the canary.

#![warn(missing_docs)]

pub mod codegen;
mod compile;
#[cfg(feature = "hot-reload")]
pub mod hot;

pub use compile::{
    CompiledEntryPoint, CompiledModule, FieldType, Scalar, Stage, StructField, StructLayout,
    compile_module, slang_build_tag,
};

/// Errors from compilation or reflection. `Slang` carries the compiler's own
/// diagnostics verbatim — the hot path shows them to the human who typed the
/// broken shader (§4.4).
#[derive(Debug, thiserror::Error)]
pub enum ShaderError {
    /// The Slang global session could not be created.
    #[error("slang global session could not be created (is the Vulkan SDK's Slang available?)")]
    NoGlobalSession,
    /// The Slang session could not be created.
    #[error("slang session could not be created")]
    NoSession,
    /// A path was not valid UTF-8/C-string data.
    #[error("shader path is not valid UTF-8/C-string data: {0}")]
    BadPath(String),
    /// Compilation or linking failed; the message is Slang's diagnostic output.
    #[error("slang: {0}")]
    Slang(String),
    /// The module reflects fine but uses a shape codegen does not cover yet.
    /// Loud by design (§1.10): the alternative is a silently wrong layout.
    #[error("reflection→codegen cannot represent this module: {0}")]
    Unsupported(String),
}

#[cfg(test)]
mod tests;
