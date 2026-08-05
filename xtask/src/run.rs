//! `cargo xtask run <demo>` — launch the shell over a game crate (§2).
//!
//! **Manual and windowed.** It builds the game as a dylib, then runs `gg-runtime`
//! pointed at the artifact and the crate's own bindings — which is the whole of
//! what "shipping is the shell beside the game dylib" means, in the working tree.
//! No automated tier calls this (§1.5); `GG_HEADLESS` is deliberately *not* set,
//! because a window is the point.
//!
//! The shell watches the *artifact*; `--watch` is what watches the **source**.
//! Without it, `cargo build -p <game>` in another terminal is the code half of
//! the reload loop — which is right when an agent is the one rebuilding, and a
//! two-terminal chore when a human is (§6 M16). Keeping the halves apart is
//! still what lets a rebuild land while play continues (§6 M5); `--watch` only
//! removes the need for a second prompt to type it in.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::util;

/// How often a watching run looks up from the source tree to see whether the
/// shell it launched is still alive. Long enough to cost nothing, short enough
/// that closing the window ends the command rather than leaving a watcher on a
/// game nobody is playing.
const CHILD_POLL: Duration = Duration::from_millis(250);

/// Run `<demo>` under the shell. Extra flags after the demo name are forwarded
/// to `gg-runtime` (`--frames`, `--record`, `--replay`), except `--tracy` and
/// `--watch`, which are this command's own.
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
    shell.args(
        args.iter()
            .filter(|a| *a != demo && **a != "--tracy" && **a != "--watch"),
    );
    if !args.contains(&"--watch") {
        return util::run(&mut shell, "gg-runtime");
    }
    watch(&mut shell, &crate_dir, &package)
}

/// Run the shell and rebuild `package` whenever its source changes, until the
/// shell exits.
///
/// The rebuild is a plain `cargo build -p <game>` — the same command the other
/// terminal was typing, and the same one the agent panel's `fix it` prompt asks
/// for. Nothing here touches the shell: the artifact changes and the shell's own
/// watcher notices, which is the seam §4.2.2 already gates. A build that fails
/// is reported and the watch continues, because a half-written source file is a
/// normal event under an editor and exiting on one would mean relaunching the
/// game to get back to where you were — the exact cost the reload loop exists to
/// avoid.
fn watch(shell: &mut std::process::Command, crate_dir: &Path, package: &str) -> anyhow::Result<()> {
    let source = crate_dir.join("src");
    let changes = ggc::watch::Changes::watching(&source)?;
    let mut child = shell
        .spawn()
        .map_err(|e| anyhow::anyhow!("could not start gg-runtime ({e})"))?;
    println!(
        "xtask: watching {} — save to rebuild {package}, close the window to stop",
        source.display()
    );
    loop {
        if let Some(status) = child.try_wait()? {
            anyhow::ensure!(status.success(), "gg-runtime exited with {status}");
            return Ok(());
        }
        match changes.next_within(CHILD_POLL) {
            ggc::watch::Change::Settled => {}
            ggc::watch::Change::Quiet => continue,
            // Returns immediately and forever, so continuing here would spin a
            // core until the window closed. The run is still worth finishing —
            // the game is playing — just without the rebuilds.
            ggc::watch::Change::Gone => {
                println!("xtask: the source watcher stopped; {package} will not rebuild");
                let status = child.wait()?;
                anyhow::ensure!(status.success(), "gg-runtime exited with {status}");
                return Ok(());
            }
        }
        // Reported rather than propagated, and by hand rather than through
        // `util::run`, which treats a nonzero exit as fatal — here it is a
        // compile error the operator is about to fix in the editor they are
        // already in.
        match util::cargo().args(["build", "-p", package]).status() {
            Ok(status) if status.success() => println!("xtask: rebuilt {package}"),
            Ok(status) => println!("xtask: {package} did not build ({status}) — still watching"),
            Err(e) => println!("xtask: could not run cargo ({e}) — still watching"),
        }
    }
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
