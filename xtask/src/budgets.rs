//! The §3 complexity budgets, as machines rather than as review standards.
//!
//! Three of them, and they fail for three different reasons: the shell grows
//! *code* (orphan logic hiding in the app shell), an owned crate grows
//! *dependencies* (a charter widening one `Cargo.toml` line at a time), and a
//! game crate grows an *engine* dependency (the reload boundary's blast radius,
//! which the fingerprint's scope is sized against — §4.2.2).
//!
//! §3 called the dependency budgets "`cargo-deny`-counted" and no such count
//! existed; cargo-deny bans crates, it does not budget them. The count is here
//! instead, with its definition stated rather than implied: **every entry in a
//! crate's `[dependencies]`, workspace-internal ones included** (§3 says the
//! forced `gg-ecs-derive` leaf "counts as one"), and dev-dependencies excluded
//! (`gg-ecs`'s own manifest already says they are outside the budget, because
//! they are absent from every runtime graph).

use std::path::Path;

use crate::util::{cargo, run_capture, walk_rs, workspace_root};

/// The `gg-runtime` code-line budget (§3). Raised 300 → 500 at M5 when the shell
/// grew the window, the renderer's three calls, live input and record/replay,
/// and 500 → 600 at M8 for the observability stack: config, the instruments,
/// the overlay, the crash handler, the capture trigger. Both raises are the same
/// argument — the shell *chooses* these and implements none of them — and both
/// were spent in a PR that said so.
const SHELL_BUDGET: usize = 600;

/// Per-crate dependency budgets (§3). Only the crates §3 actually names carry
/// one; a budget invented here would be a rule this file made up.
const DEPENDENCY_BUDGETS: &[(&str, usize)] = &[("gg-ecs", 6), ("gg-core", 8)];

/// What a game crate may reach engine-side (§3's deny pin, §4.2.2's blast
/// radius). `gg-ecs-derive` is the forced proc-macro leaf `gg-ecs` re-exports,
/// so a game crate reaches it whether or not it names it.
const GAME_CRATE_PIN: &[&str] = &["gg-abi", "gg-ecs", "gg-ecs-derive", "gg-math"];

/// §4.10's reference-set cap. Per-backend PNG sets grow monotonically and a repo
/// that takes minutes to clone fails §9's fresh-clone bar long before git
/// complains; crossing this is a decision (LFS, or a references sub-repo) made in
/// a PR, not discovered when a clone crawls.
const REFERENCE_BUDGET: u64 = 50 * 1024 * 1024;

pub fn check() -> anyhow::Result<()> {
    shell_lines()?;
    dependencies()?;
    game_crate_pin()?;
    reference_images()
}

/// The golden reference sets, weighed (§4.10).
fn reference_images() -> anyhow::Result<()> {
    fn weigh(dir: &Path, total: &mut u64, count: &mut usize) -> anyhow::Result<()> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Ok(()); // no references yet is not a budget failure
        };
        for entry in entries {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                weigh(&entry.path(), total, count)?;
            } else {
                *total += entry.metadata()?.len();
                *count += 1;
            }
        }
        Ok(())
    }
    let (mut total, mut count) = (0u64, 0usize);
    weigh(
        &workspace_root().join("tests/gg-images"),
        &mut total,
        &mut count,
    )?;
    anyhow::ensure!(
        total <= REFERENCE_BUDGET,
        "golden references are {total} B across {count} file(s) against a {REFERENCE_BUDGET} B \
         budget (§4.10) — Git LFS or a references sub-repository is the decision to make, not a \
         higher number"
    );
    println!(
        "xtask: golden references {} KiB / {} MiB across {count} file(s) (§4.10)",
        total / 1024,
        REFERENCE_BUDGET / (1024 * 1024)
    );
    Ok(())
}

/// Complexity budgets, the CI-counted lines (§3).
///
/// Code rather than every line, because §3's phrase is "thin in code" and the
/// house comment style is dense and inline — counting both would make the two
/// rules fight, and the way that fight resolves is by deleting comments to fit a
/// shell-size cap. The total is printed beside it so comment mass stays visible.
fn shell_lines() -> anyhow::Result<()> {
    let mut files = Vec::new();
    walk_rs(&workspace_root().join("crates/gg-runtime/src"), &mut files);
    let (mut code, mut total) = (0usize, 0usize);
    for text in files.iter().filter_map(|f| std::fs::read_to_string(f).ok()) {
        for line in text.lines() {
            total += 1;
            let line = line.trim_start();
            if !line.is_empty() && !line.starts_with("//") {
                code += 1;
            }
        }
    }
    anyhow::ensure!(
        code <= SHELL_BUDGET,
        "gg-runtime is {code} code lines against a {SHELL_BUDGET}-line budget (§3) — \
         raising the budget is a PR, not a drift"
    );
    println!("xtask: gg-runtime line budget {code}/{SHELL_BUDGET} code lines ({total} total)");
    Ok(())
}

/// The per-crate dependency budgets of §3, counted off the manifests.
fn dependencies() -> anyhow::Result<()> {
    let root = workspace_root();
    for (name, budget) in DEPENDENCY_BUDGETS {
        let manifest: toml::Value = toml::from_str(&std::fs::read_to_string(
            root.join("crates").join(name).join("Cargo.toml"),
        )?)?;
        let deps = manifest
            .get("dependencies")
            .and_then(toml::Value::as_table)
            .map(toml::Table::len)
            .unwrap_or(0);
        anyhow::ensure!(
            deps <= *budget,
            "{name} declares {deps} dependencies against a {budget} budget (§3) — raising it is \
             a PR that says what the crate took delivery of, not a drift"
        );
        println!("xtask: {name} dependency budget {deps}/{budget}");
    }
    Ok(())
}

/// §3's game-crate deny pin, which had no machine behind it: a game crate may
/// reach [`GAME_CRATE_PIN`] engine-side and nothing else.
///
/// This is what makes the boundary fingerprint's scope and a dylib's possible
/// link set the same list (§4.2.2) — an engine crate arriving in a game graph
/// widens the blast radius without widening the fingerprint, and that is a
/// silent hole rather than a loud one. Third-party dependencies are the game's
/// own business; the pin is engine-side by construction.
///
/// Game crates are found rather than listed: a `demos/` package that builds a
/// `cdylib` *is* game code (§2's Game-code boundary row), so demo 04 is covered
/// the day it exists and not the day someone remembers to add it here.
fn game_crate_pin() -> anyhow::Result<()> {
    let root = workspace_root();
    let games = game_crates()?;
    anyhow::ensure!(
        !games.is_empty(),
        "the game-crate deny pin matched no crate — a check that finds nothing to check passes \
         vacuously (§5.8's rule, applied to §3's pin)"
    );
    for name in &games {
        check_one_game_crate(name, &root)?;
    }
    Ok(())
}

/// Every game crate in the workspace, by package name.
///
/// Found rather than listed: a `demos/` package that builds a `cdylib` *is* game
/// code (§2's Game-code boundary row), so demo 04 is covered the day it exists
/// and not the day someone remembers to add it to a constant. Shared with the
/// dist gate, which has the same question and must not answer it differently.
pub fn game_crates() -> anyhow::Result<Vec<String>> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(workspace_root().join("demos"))?.flatten() {
        let manifest = entry.path().join("Cargo.toml");
        if !manifest.is_file() {
            continue;
        }
        let parsed: toml::Value = toml::from_str(&std::fs::read_to_string(&manifest)?)?;
        let builds_a_cdylib = parsed
            .get("lib")
            .and_then(|l| l.get("crate-type"))
            .and_then(toml::Value::as_array)
            .is_some_and(|kinds| kinds.iter().any(|k| k.as_str() == Some("cdylib")));
        if !builds_a_cdylib {
            continue;
        }
        found.push(
            parsed
                .get("package")
                .and_then(|p| p.get("name"))
                .and_then(toml::Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("{} has no package name", manifest.display()))?
                .to_owned(),
        );
    }
    found.sort();
    Ok(found)
}

fn check_one_game_crate(name: &str, root: &Path) -> anyhow::Result<()> {
    // The resolved graph, not the manifest: the pin is about what a dylib can
    // *link*, and a transitive engine crate links exactly as hard as a declared
    // one.
    let tree = run_capture(
        cargo()
            .current_dir(root)
            .args(["tree", "-p", name, "-e", "normal", "--prefix", "none"]),
        &format!("cargo tree ({name} link set)"),
    )?;
    let mut offenders: Vec<&str> = tree
        .lines()
        .filter_map(|l| l.split_whitespace().next())
        .filter(|c| c.starts_with("gg-") && !GAME_CRATE_PIN.contains(c))
        .collect();
    offenders.sort_unstable();
    offenders.dedup();
    anyhow::ensure!(
        offenders.is_empty(),
        "game crate {name} links {offenders:?} — §3 pins game crates to {GAME_CRATE_PIN:?}, which \
         is what keeps the §4.2.2 fingerprint's scope and a dylib's link set the same list"
    );
    println!("xtask: {name} links only the pinned boundary crates (§3)");
    Ok(())
}
