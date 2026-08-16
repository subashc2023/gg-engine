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

/// Demos whose sources are **fetched** rather than checked in, and which this
/// gate therefore skips (§6 M59).
///
/// The gate's subject is byte-reproducibility over sources this repository
/// holds. Demo 14's are fifty-one megabytes downloaded by `cargo xtask sponza`,
/// so on a clean clone there is nothing here to compile — and on a machine that
/// *has* fetched them there are two clean 84 MiB builds, which is a minute of
/// the push tier spent proving the same property four other demos already prove.
/// `xtask sponza` builds that pack itself, so the demo is not left without one.
///
/// A list rather than a heuristic: "large" is not a property, and a gate that
/// skipped a tree for being big would silently stop covering one that grew.
const FETCHED: &[&str] = &["14-sponza"];

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
        if demo == "10-tetris" {
            clips_resolve(&out)?;
        }
    }
    Ok(())
}

/// Every clip demo 10's cue table names is in the pack that was just built
/// (§6 M43).
///
/// The one failure nothing else in the tree can see. `Sound::clip("lock", …)`
/// hashes a string literal at compile time and a pack names assets after
/// filenames, so a renamed `.wav`, a typo or a deleted source produces a cue
/// that is *silent* — not a build error, not a panic, not a failed test, because
/// a clip the bank does not hold is silence by contract (`Sound::clip`), which
/// is the right answer for a pack mid-rebuild and the wrong one for a cue that
/// will never resolve again.
///
/// Demo 10 by name rather than every demo, because there is no way to enumerate
/// a game's `Sound`s without loading its dylib, and a gate that guessed would be
/// a gate that stops catching things when a second game gets clips.
fn clips_resolve(pack: &Path) -> Result<()> {
    let pack = gg_assets::Pack::open(pack)
        .with_context(|| format!("opening {} to check its clips", pack.display()))?;
    let voices = (0..demo_10_tetris::CUES as u8)
        .map(demo_10_tetris::voice_of)
        .chain(std::iter::once(demo_10_tetris::music_voice()));
    let mut found = 0;
    for voice in voices {
        if voice.clip == 0 {
            continue;
        }
        ensure!(
            pack.find(gg_assets::AssetId(voice.clip)).is_some(),
            "demo 10 names a clip its pack does not hold ({:#018x}) — a cue whose asset went away \
             is silent rather than broken, so nothing but this notices",
            voice.clip
        );
        found += 1;
    }
    println!("xtask assets: 10-tetris — {found} cue(s) resolve against the pack (§6 M43)");
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
        if FETCHED.contains(&name.as_str()) {
            println!("xtask assets: {name} — skipped, its sources are fetched (§6 M59)");
            continue;
        }
        found.push((name, assets));
    }
    found.sort();
    Ok(found)
}
