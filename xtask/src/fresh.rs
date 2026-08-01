//! The weekly gates that check the *repository* rather than the code (§5,
//! "Cadence"): a run from a pristine clone, and the `cargo update` canary.
//! Standing from M4B (§6, M0A's deferred-machine schedule).
//!
//! # Why a fresh clone is a gate and not a ritual
//!
//! Local CI has no clean-machine guarantee. Every other tier runs in a tree
//! that has been built in for months, with a warm `target/`, generated files
//! already present, and whatever `xtask` cached last time. §9's recovery bar
//! says a pristine clone plus the Vulkan SDK, Rust and pinned fetches
//! reconstructs the entire CI definition — and the standing rule underneath it
//! is that **nothing about CI may live only in machine state git does not
//! hold**. This gate is that rule, run weekly, against the one thing that can
//! disprove it.
//!
//! It runs the `--push` tier rather than `--nightly`: from a cold target dir
//! the nightly tier's fat-LTO dist builds and golden suite would put the weekly
//! run into hours, and everything the *clone* can get wrong — a missing
//! generated file, an uncommitted fixture, a path that only resolves because
//! this machine has it — fails inside `--push` already.

use crate::util::{cargo, run as exec, run_capture, workspace_root};

/// `xtask ci` from a pristine clone of **HEAD**, in a scratch directory.
///
/// HEAD and not the working tree, deliberately: the question is whether *what
/// git holds* rebuilds, so uncommitted work is out of scope by construction —
/// and a red run here after a local fix means the fix is not committed yet,
/// which is exactly what it should mean.
pub fn clone_gate() -> anyhow::Result<()> {
    let root = workspace_root();
    let scratch = root.join("target/xtask-cache/fresh-clone");
    if scratch.exists() {
        std::fs::remove_dir_all(&scratch)?;
    }
    std::fs::create_dir_all(&scratch)?;

    // Clone the local repository, not a remote: the gate asks whether *what git
    // holds* builds, and going via a remote would also be testing the push.
    exec(
        std::process::Command::new("git").args([
            "clone",
            "--no-hardlinks",
            "--quiet",
            &root.display().to_string(),
            &scratch.display().to_string(),
        ]),
        "git clone (fresh-clone gate, §5/§9)",
    )?;

    let head = run_capture(
        std::process::Command::new("git")
            .current_dir(&root)
            .args(["rev-parse", "HEAD"]),
        "git rev-parse HEAD",
    )?;
    println!(
        "xtask fresh-clone: pristine clone of {} at {}",
        head.trim(),
        scratch.display()
    );

    // Its own target dir, its own everything. The clone gets no help from this
    // tree beyond the toolchain and the pinned fetches §9 permits.
    exec(
        cargo()
            .current_dir(&scratch)
            .args(["xtask", "ci", "--push"]),
        "cargo xtask ci --push (from a pristine clone)",
    )?;

    // Left behind on failure, cleaned on success: a red fresh-clone run is
    // worth looking at, and a green one is worth a few gigabytes back.
    std::fs::remove_dir_all(&scratch)?;
    println!("xtask fresh-clone: green — CI's definition lives in git, not in this machine (§9)");
    Ok(())
}

/// The `cargo update` canary (§5): does the workspace still build against the
/// newest semver-compatible dependency set?
///
/// Run against a copy of the lockfile, never the tree's: this answers "would
/// updating break us", and answering it by updating would make the answer moot.
/// The determinism-critical pins (`libm`, `blake3`) are exact by design, so a
/// green canary is not a claim that updating *them* is safe — those are
/// contract events with their own ceremony (§4.2.1).
pub fn update_canary() -> anyhow::Result<()> {
    let lock = workspace_root().join("Cargo.lock");
    let before = std::fs::read(&lock)?;

    let result = (|| -> anyhow::Result<bool> {
        exec(cargo().args(["update"]), "cargo update (canary)")?;
        // Compared against the bytes this function saved, never against `git
        // diff`: the tree may legitimately carry an uncommitted lockfile, and a
        // canary that reported *those* changes as "available updates" would be
        // reporting the developer to themselves.
        let moved = std::fs::read(&lock)? != before;
        exec(
            cargo().args(["check", "--workspace", "--all-targets"]),
            "cargo check --workspace (against the updated lockfile)",
        )?;
        Ok(moved)
    })();

    // Restore whichever way the check went: the canary observes, it does not
    // decide. Taking an update is a deliberate act with a review behind it.
    std::fs::write(&lock, before)?;

    if result? {
        println!("xtask canary: the workspace builds against the newest compatible deps");
    } else {
        println!("xtask canary: no semver-compatible updates available");
    }
    println!("xtask canary: Cargo.lock restored — updating is a reviewed act, not a side effect");
    Ok(())
}
