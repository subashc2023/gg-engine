//! `ggc` (§4.6) — the asset compiler. glTF in, `gg-pack` out.
//!
//! Usage:
//!   ggc build <source-dir> -o <pack>   compile a source tree into a pack
//!   ggc watch <source-dir> -o <pack>   the same, again on every change
//!   ggc list  <pack>                   print a pack's table
//!
//! Everything that parses a source format is in this binary and in no other, so
//! §2's "the runtime never parses JSON" is a fact about the dependency graph.

use std::path::PathBuf;

use anyhow::{Result, bail};
use ggc::{build, watch};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("build") => {
            let (source, out) = build_args(&args[1..])?;
            let stats = build::build(&source, &out)?;
            println!(
                "ggc: {} assets in {} — {} source(s) compiled, {} reused",
                stats.assets,
                out.display(),
                stats.compiled,
                stats.reused
            );
            Ok(())
        }
        Some("watch") => {
            let (source, out) = build_args(&args[1..])?;
            watch::watch(&source, &out)
        }
        Some("list") => match args.get(1) {
            Some(pack) => build::list(&PathBuf::from(pack)),
            None => bail!("ggc list needs a pack"),
        },
        Some(verb) => bail!("ggc: unknown verb {verb}"),
        None => {
            println!("usage: ggc build|watch <source-dir> -o <pack> | ggc list <pack>");
            Ok(())
        }
    }
}

/// `<source-dir> -o <pack>`, shared by `build` and `watch`. `-o` is required
/// rather than defaulted: a compiler that guesses where its output goes is one
/// that overwrites something.
fn build_args(args: &[String]) -> Result<(PathBuf, PathBuf)> {
    let mut source = None;
    let mut out = None;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "-o" | "--out" => out = rest.next().map(PathBuf::from),
            other if other.starts_with('-') => bail!("ggc build: unknown flag {other}"),
            other => source = Some(PathBuf::from(other)),
        }
    }
    match (source, out) {
        (Some(source), Some(out)) => Ok((source, out)),
        _ => bail!("usage: ggc build <source-dir> -o <pack>"),
    }
}
