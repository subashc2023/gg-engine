//! `cargo xtask run <demo>` — launch the shell over a game crate (§2).
//!
//! **Manual and windowed.** It builds the game as a dylib, then runs `gg-runtime`
//! pointed at the artifact and the crate's own bindings — which is the whole of
//! what "shipping is the shell beside the game dylib" means, in the working tree.
//! No automated tier calls this (§1.5); `GG_HEADLESS` is deliberately *not* set,
//! because a window is the point.
//!
//! It does not watch: `cargo build -p <game>` in another terminal is the reload
//! loop, and the shell's watcher is what notices. Keeping the two apart is what
//! lets an agent rebuild while a human keeps playing (§6 M5).

use std::path::{Path, PathBuf};

use crate::util;

/// Run `<demo>` under the shell. Extra flags after the demo name are forwarded
/// to `gg-runtime` (`--frames`, `--record`, `--replay`), except `--tracy`,
/// which is this command's own and builds the shell with the profiler in.
pub fn run(args: &[&str]) -> anyhow::Result<()> {
    let demo = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .ok_or_else(|| anyhow::anyhow!("usage: cargo xtask run <demo> [-- shell flags]"))?;
    let root = util::workspace_root();
    let crate_dir = root.join("demos").join(demo);
    anyhow::ensure!(
        crate_dir.join("Cargo.toml").is_file(),
        "no demo crate at {} — game crates live under demos/ (§3)",
        crate_dir.display()
    );
    let package = format!("demo-{demo}");

    util::run(
        util::cargo().args(["build", "-p", &package]),
        "cargo build (game dylib)",
    )?;
    // `--tracy` rather than always: the client binds a TCP listener from a
    // static constructor, so a shell built with it prompts the firewall on
    // every fresh build path even when nobody attaches a profiler (§6 M9).
    // Consumed here, never forwarded — the shell has no such flag.
    let features = if args.contains(&"--tracy") {
        "tier-dev,tracy"
    } else {
        "tier-dev"
    };
    util::run(
        util::cargo().args(["build", "-p", "gg-runtime", "--features", features]),
        "cargo build (shell)",
    )?;

    let dylib = root.join("target/debug").join(dylib_name(&package));
    anyhow::ensure!(
        dylib.is_file(),
        "{} is not there — is `crate-type = [\"cdylib\", \"rlib\"]` set? (§4.2.2)",
        dylib.display()
    );

    let mut shell = std::process::Command::new(shell_binary(&root));
    shell.arg("--game").arg(&dylib);
    // The bindings are the game crate's, beside its source (§4.7). Passing them
    // rather than having the shell guess keeps "where does the map come from" a
    // flag instead of a convention the shell has to encode.
    let bindings = crate_dir.join("input.toml");
    if bindings.is_file() {
        shell.arg("--input").arg(&bindings);
    }
    // Same argument for the pack (§4.6): a demo that declares an `assets/` tree
    // gets it compiled and passed. Built here rather than assumed present,
    // because a pack is build output and a stale one is worse than none — and
    // `ggc watch` in another terminal is the asset half of the reload loop, the
    // way `cargo build -p <game>` is the code half.
    if crate_dir.join("assets").is_dir() {
        crate::assets::run(&[])?;
        shell
            .arg("--pack")
            .arg(root.join(format!("target/assets/{demo}.ggpack")));
    }
    shell.args(args.iter().filter(|a| *a != demo && **a != "--tracy"));
    util::run(&mut shell, "gg-runtime")
}

fn shell_binary(root: &Path) -> PathBuf {
    root.join("target/debug").join(if cfg!(windows) {
        "gg-runtime.exe"
    } else {
        "gg-runtime"
    })
}

fn dylib_name(package: &str) -> String {
    let stem = package.replace('-', "_");
    if cfg!(windows) {
        format!("{stem}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else {
        format!("lib{stem}.so")
    }
}
