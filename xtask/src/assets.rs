//! `cargo xtask assets` (§4.6) — compile every demo's asset source tree.
//!
//! Calls `ggc` as a library for the same reason `xtask shaders` calls
//! `gg-shaders`: the gate must run the compiler the build runs, not a second
//! one that agrees with it today.
//!
//! A demo declares assets by having a `demos/<name>/assets/` directory. Packs
//! are build output and land under `target/assets/`; they are not checked in,
//! because a pack is derived and the sources beside it are the record.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

use crate::util::workspace_root;

/// Where compiled packs land, under the workspace root — `cargo xtask` runs in
/// the caller's directory and never chdirs, so every path here is rooted.
const OUT: &str = "target/assets";

pub fn run(args: &[&str]) -> Result<()> {
    let check = args.contains(&"--check");
    let sources = demo_sources()?;
    // Demos do ship `assets/` trees, so an empty set means the walk lost them
    // rather than that none exist — and a §4.6 gate that compiled nothing used
    // to report it as success (§5.8's rule: a vacuous pass is not a green one).
    ensure!(
        !sources.is_empty(),
        "no demo declares an `assets/` source tree — §4.6's byte-reproducibility gate has nothing \
         to compile, which is a broken walk and not a fact about the repository"
    );
    let packs = workspace_root().join(OUT);
    for (demo, source) in sources {
        let out = packs.join(format!("{demo}.ggpack"));
        if check {
            // A *clean* run, twice: an incremental build that reused the file
            // it is being compared against would compare it to itself.
            let _ = std::fs::remove_file(&out);
        }
        let stats = ggc::build::build(&source, &out)?;
        println!(
            "xtask assets: {demo} — {} assets, {} compiled, {} reused → {}",
            stats.assets,
            stats.compiled,
            stats.reused,
            out.display()
        );
        if check {
            let twin = out.with_extension("twin");
            let _ = std::fs::remove_file(&twin);
            ggc::build::build(&source, &twin)?;
            let (first, second) = (read(&out)?, read(&twin)?);
            ensure!(
                first == second,
                "{demo}: two clean `ggc` runs over identical inputs produced different packs \
                 ({} vs {} bytes) — §4.6's byte-reproducibility is what the incremental cache \
                 rests on, so this is a correctness failure and not a cosmetic one",
                first.len(),
                second.len()
            );
            std::fs::remove_file(&twin).ok();
            println!("xtask assets: {demo} — bit-identical across two clean runs (§4.6)");
        }
    }
    Ok(())
}

fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("reading {}", path.display()))
}

/// `(demo name, source directory)` for every demo with an `assets/` tree,
/// sorted, because a build log whose order is the filesystem's is a build log
/// that reads differently on two machines.
fn demo_sources() -> Result<Vec<(String, PathBuf)>> {
    let mut found = Vec::new();
    // Rooted, not relative: every other gate resolves through `workspace_root`,
    // and a bare `demos` here read the *caller's* directory — from anywhere but
    // the root it found nothing and the gate above passed on an empty set.
    let demos = workspace_root().join("demos");
    if !demos.is_dir() {
        return Ok(found);
    }
    for entry in std::fs::read_dir(&demos).context("reading demos/")? {
        let demo = entry?.path();
        let assets = demo.join("assets");
        if !assets.is_dir() {
            continue;
        }
        let name = demo
            .file_name()
            .and_then(|n| n.to_str())
            .context("a demo directory with no utf-8 name")?
            .to_owned();
        found.push((name, assets));
    }
    found.sort();
    Ok(found)
}
