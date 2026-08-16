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
/// to `gg-runtime` (`--frames`, `--record`, `--replay`), except `--tracy`,
/// `--watch`, `--validate` and `--profile`, which are this command's own.
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
    // §6 M58's reader. Its own flag rather than part of `tier-dev` for the
    // reason the feature's declaration gives: a play session should not pay for
    // a table nobody asked for. Composes with `--tracy`, which is the same
    // measurement through a GUI instead of a terminal.
    let mut features = String::from("tier-dev");
    if args.contains(&"--tracy") {
        features.push_str(",tracy");
    }
    if args.contains(&"--profile") {
        features.push_str(",cpu-timings");
    }
    util::run(
        util::cargo().args(["build", "-p", "gg-runtime", "--features", &features]),
        "cargo build (shell)",
    )?;

    // Package name → library stem: cargo underscores it, and the artifact is
    // named after the lib, not the package.
    let dylib = root
        .join("target/debug")
        .join(util::dylib_name(&package.replace('-', "_")));
    anyhow::ensure!(
        dylib.is_file(),
        "{} is not there — is `crate-type = [\"cdylib\", \"rlib\"]` set? (§4.2.2)",
        dylib.display()
    );

    let mut shell = std::process::Command::new(shell_binary(&root));
    // §6 M58. The validation layer is a *frame* cost and a large one — 5.4 ms
    // on demo 12 at 1080p on the desk's 4090, with the device's own time
    // unchanged, which is the whole of a 240 Hz panel reporting 150. This is
    // the one command whose subject is a human playing, and a play session has
    // nothing to prove about API misuse that `ci --push`, `xtask gpu` and every
    // nextest run do not already prove on the same code. Said out loud both
    // ways: a quiet downgrade is what §1.10 forbids, and the operator has to
    // know which build the number they are about to read came from.
    match args.contains(&"--validate") {
        true => println!("xtask: validation layer on (--validate) — expect a slower frame"),
        false => {
            shell.env("GG_VALIDATION", "0");
            println!("xtask: validation layer off for play — `--validate` puts it back");
        }
    }
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
    shell.args(args.iter().filter(|a| {
        *a != demo
            && **a != "--tracy"
            && **a != "--watch"
            && **a != "--validate"
            && **a != "--profile"
    }));
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
