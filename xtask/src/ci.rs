//! `cargo xtask ci` — the four tiers of §3/§5. `--fast` is the Stop hook's tier
//! and lives under a <30 s warm budget: it must never be the reason agent turns
//! balloon. Every tier is headless (§1.5) via util::cargo().

use crate::util::{cargo, run as exec, run_capture, walk_rs, workspace_root};
use std::collections::BTreeSet;
use std::path::PathBuf;

pub fn run_tier(tier: &str) -> anyhow::Result<()> {
    match tier {
        "fast" => fast(),
        "push" => push(),
        "nightly" => nightly(),
        "weekly" => weekly(),
        other => anyhow::bail!("unknown ci tier `{other}`"),
    }
}

pub fn run(args: &[&str]) -> anyhow::Result<()> {
    let tier = args
        .iter()
        .find(|a| a.starts_with("--") && **a != "--hook")
        .map(|a| &a[2..])
        .unwrap_or("fast");
    run_tier(tier)
}

/// Stop-hook tier: fmt + clippy on changed crates + tests for changed crates.
/// A clean tree passes by definition — the hook blocks on dirty-and-red only.
fn fast() -> anyhow::Result<()> {
    let changed = changed_build_paths()?;
    if changed.is_empty() {
        println!("xtask ci --fast: tree clean — green by definition");
        return Ok(());
    }
    let crates = crates_touched(&changed);
    exec(cargo().args(["fmt", "--check"]), "cargo fmt --check")?;
    clippy(&crates)?;
    tests(&crates)?;
    println!("xtask ci --fast: green");
    Ok(())
}

/// Pre-push tier: gates 1-3 in full, plus dist/dist-verify feature checks (§5).
/// Replay determinism joins when replays exist (M4B); named absence, not a gap.
fn push() -> anyhow::Result<()> {
    exec(cargo().args(["fmt", "--check"]), "cargo fmt --check")?;
    clippy(&All)?;
    exec(cargo().args(["deny", "check"]), "cargo deny check")?;
    greps()?;
    allowlist_crosscheck()?;
    line_budgets()?;
    tests(&All)?;
    fp_baseline_dist_profile()?;
    crate::shaders::build_all()?;
    for feats in ["tier-dist", "tier-dist-verify"] {
        exec(
            cargo().args([
                "check",
                "-p",
                "gg-runtime",
                "--no-default-features",
                "--features",
                feats,
            ]),
            &format!("cargo check gg-runtime --features {feats}"),
        )?;
    }
    println!("xtask ci --push: green (replay determinism gates join at M4B)");
    Ok(())
}

fn nightly() -> anyhow::Result<()> {
    push()?;
    crate::dist::gate()?;
    crate::probe::run(false)?;
    aarch64_leg()?;
    println!(
        "xtask ci --nightly: green (golden suite M7, GPU tests M1, demo runs M1, \
         chaos replays M4B — each lands with its milestone)"
    );
    Ok(())
}

/// The third architecture of the §5 matrix (§6 M0B): gg-math — the FP baseline
/// and its unit tests — cross-compiled to aarch64 and run under qemu-user, in
/// both the dev and dist profiles (§5.6d). Linker and runner come from
/// .cargo/config.toml; the leg runs from the WSL lane, so on Windows it defers
/// rather than pretending.
fn aarch64_leg() -> anyhow::Result<()> {
    if cfg!(windows) {
        println!("xtask: aarch64 qemu leg runs in the WSL lane (§5) — skipped on Windows");
        return Ok(());
    }
    for profile in ["dev", "dist"] {
        exec(
            cargo().args([
                "nextest",
                "run",
                "-p",
                "gg-math",
                "--target",
                "aarch64-unknown-linux-gnu",
                "--cargo-profile",
                profile,
            ]),
            &format!("gg-math tests on aarch64 under qemu ({profile} profile)"),
        )?;
    }
    Ok(())
}

fn weekly() -> anyhow::Result<()> {
    nightly()?;
    println!(
        "xtask ci --weekly: green (fresh-clone gate and cargo-update canary go standing at M4B; \
         GPU-assisted validation at M1)"
    );
    Ok(())
}

// ---- crate selection ----------------------------------------------------

struct All;
trait CrateSet {
    fn args(&self) -> Vec<String>;
}
impl CrateSet for All {
    fn args(&self) -> Vec<String> {
        vec!["--workspace".into()]
    }
}
impl CrateSet for BTreeSet<String> {
    fn args(&self) -> Vec<String> {
        if self.contains("*") {
            vec!["--workspace".into()]
        } else {
            self.iter().flat_map(|c| ["-p".into(), c.clone()]).collect()
        }
    }
}

fn clippy(crates: &dyn CrateSet) -> anyhow::Result<()> {
    let mut cmd = cargo();
    cmd.arg("clippy").args(crates.args()).arg("--all-targets");
    // `#![deny(warnings)]` is CI-only (§3): the flag lives here, not in source.
    cmd.args(["--", "-D", "warnings"]);
    exec(&mut cmd, "cargo clippy -D warnings")
}

fn tests(crates: &dyn CrateSet) -> anyhow::Result<()> {
    let mut cmd = cargo();
    cmd.args(["nextest", "run", "--no-tests=pass"])
        .args(crates.args());
    exec(&mut cmd, "cargo nextest run")
}

/// §5.6d: the FP baseline must hold under dist codegen (fat LTO,
/// codegen-units=1), not just dev — optimization level may not touch sim bits.
/// The dev-profile run is already part of the workspace test pass.
fn fp_baseline_dist_profile() -> anyhow::Result<()> {
    exec(
        cargo().args(["nextest", "run", "-p", "gg-math", "--cargo-profile", "dist"]),
        "gg-math tests under the dist profile (§5.6d)",
    )
}

/// Dirty paths that can affect a build. Docs, hook config, and logs cannot;
/// PLAN.md is deliberately untracked either way.
fn changed_build_paths() -> anyhow::Result<Vec<String>> {
    let out = run_capture(
        std::process::Command::new("git")
            .current_dir(workspace_root())
            .args(["status", "--porcelain"]),
        "git status",
    )?;
    Ok(out
        .lines()
        .filter_map(|l| l.get(3..).map(str::trim))
        .map(|p| p.trim_matches('"').replace('\\', "/"))
        .filter(|p| !(p.ends_with(".md") || p.starts_with(".claude/") || p == ".gitignore"))
        .collect())
}

fn crates_touched(paths: &[String]) -> BTreeSet<String> {
    let mut crates = BTreeSet::new();
    for p in paths {
        if let Some(rest) = p.strip_prefix("crates/")
            && let Some((name, _)) = rest.split_once('/')
        {
            crates.insert(name.to_string());
            continue;
        }
        if p.starts_with("xtask/") {
            crates.insert("xtask".to_string());
        } else {
            // Workspace-level file (Cargo.toml, deny.toml, clippy.toml, ...):
            // everything is potentially affected.
            crates.insert("*".to_string());
        }
    }
    crates
}

// ---- gate 1 extras: greps, cross-checks, budgets (§3) -------------------

fn greps() -> anyhow::Result<()> {
    let root = workspace_root();
    let mut files = Vec::new();
    walk_rs(&root.join("crates"), &mut files);
    walk_rs(&root.join("demos"), &mut files);

    let mut violations = Vec::new();
    for file in &files {
        let rel = file.strip_prefix(&root).unwrap_or(file);
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        let text = std::fs::read_to_string(file)?;

        // API containment (§3): vk::/ash:: tokens live in gg-rhi alone.
        if !rel_str.starts_with("crates/gg-rhi/")
            && (text.contains("ash::") || text.contains("vk::"))
        {
            violations.push(format!("{rel_str}: vk::/ash:: token outside gg-rhi (§3)"));
        }

        // Float-time fields (§2 Sim time row). Scoped to hashed components for
        // real at M3; until then any struct-field-shaped hit in engine/game
        // code is treated as a violation.
        for (lineno, line) in text.lines().enumerate() {
            let t = line
                .trim_start()
                .strip_prefix("pub ")
                .unwrap_or(line.trim_start());
            for field in ["time", "elapsed", "dt", "seconds", "duration", "timer"] {
                for width in ["f32", "f64"] {
                    if (t.starts_with(&format!("{field}: {width}"))
                        || t.starts_with(&format!("{field}: Option<{width}>")))
                        && !line.contains("fn ")
                    {
                        violations.push(format!(
                            "{rel_str}:{}: float time field `{field}` (§2 Sim time row)",
                            lineno + 1
                        ));
                    }
                }
            }
        }

        // Game crates: no smuggled state across reloads (§4.2.2). Demos 03+
        // are the game crates; the grep stands ready from day one.
        if rel_str.starts_with("demos/")
            && (text.contains("static mut") || text.contains("thread_local!"))
        {
            violations.push(format!(
                "{rel_str}: static mut / thread_local! in a game crate (§4.2.2)"
            ));
        }
    }
    // Hand-written barrier grep joins at M6, when the render graph owns barriers.

    anyhow::ensure!(
        violations.is_empty(),
        "grep gate failed:\n{}",
        violations.join("\n")
    );
    println!("xtask: grep gates clean ({} files)", files.len());
    Ok(())
}

/// The `rayon` ban's wrappers in deny.toml must cover every exemption in
/// determinism-allowlist.toml (§4.1) — two files, one truth, machine-checked.
fn allowlist_crosscheck() -> anyhow::Result<()> {
    let root = workspace_root();
    let allow: toml::Value =
        std::fs::read_to_string(root.join("determinism-allowlist.toml"))?.parse()?;
    let deny: toml::Value = std::fs::read_to_string(root.join("deny.toml"))?.parse()?;

    let exempt: BTreeSet<&str> = allow
        .get("exemptions")
        .and_then(|e| e.as_array())
        .map(|a| {
            a.iter()
                .filter(|e| e.get("dependency").and_then(|d| d.as_str()) == Some("rayon"))
                .filter_map(|e| e.get("crate").and_then(|c| c.as_str()))
                .collect()
        })
        .unwrap_or_default();

    let wrappers: BTreeSet<&str> = deny
        .get("bans")
        .and_then(|b| b.get("deny"))
        .and_then(|d| d.as_array())
        .and_then(|entries| {
            entries
                .iter()
                .find(|e| e.get("crate").and_then(|c| c.as_str()) == Some("rayon"))
        })
        .and_then(|e| e.get("wrappers"))
        .and_then(|w| w.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();

    let missing: Vec<&&str> = exempt.iter().filter(|c| !wrappers.contains(**c)).collect();
    anyhow::ensure!(
        missing.is_empty(),
        "determinism-allowlist.toml exempts {missing:?} for rayon, but deny.toml's rayon wrappers do not include them"
    );
    println!("xtask: rayon allowlist and deny.toml wrappers agree");
    Ok(())
}

/// Complexity budgets, the CI-counted lines (§3). gg-runtime: 300.
fn line_budgets() -> anyhow::Result<()> {
    let mut files: Vec<PathBuf> = Vec::new();
    walk_rs(&workspace_root().join("crates/gg-runtime/src"), &mut files);
    let lines: usize = files
        .iter()
        .map(|f| {
            std::fs::read_to_string(f)
                .map(|t| t.lines().count())
                .unwrap_or(0)
        })
        .sum();
    anyhow::ensure!(
        lines <= 300,
        "gg-runtime is {lines} lines against a 300-line budget (§3) — raising the budget is a PR, not a drift"
    );
    println!("xtask: gg-runtime line budget {lines}/300");
    Ok(())
}
