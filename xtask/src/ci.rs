//! `cargo xtask ci` — the four tiers of §3/§5. `--fast` is the Stop hook's tier
//! and lives under [`FAST_TIER_BUDGET`], now measured rather than claimed: it
//! must never be the reason agent turns balloon. Every tier is headless (§1.5)
//! via util::cargo() — and *windowless by construction*: no automated tier
//! creates an OS window at all (presenting maps a Wayland surface,
//! minimize/restore maps X11/Win32 ones, and WSLg mirrors any mapped window onto
//! the real desktop — "invisible" is not a property CI may rely on). Everything
//! windowed lives in the manual suite, `cargo xtask interactive`, run by a human
//! who expects windows.
//!
//! `--hook`'s exit 2 is decided in `main`, so it can only report a tier that
//! *ran*. A hook that fails closed when `xtask` itself will not compile needs a
//! layer above the binary, and that layer is the `|| exit 2` in
//! `.claude/settings.json`: cargo answers a build error with 101, and Claude Code
//! reads any nonzero-but-not-2 exit as non-blocking — an agent that broke `xtask`
//! would otherwise have disabled the gate that catches it breaking anything else.

use crate::util::{cargo, run as exec, run_capture, walk_rs, workspace_root};
use std::collections::BTreeSet;
use std::time::Duration;

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

/// §5's verification budget for the Stop hook's tier: `--fast` **< 30 s** warm.
///
/// The §3 budgets that count *artifacts* — shell lines, per-crate dependencies,
/// reference-image bytes — live in `budgets.rs`; this one is a property of the
/// tier running and can only be weighed where it runs. Raising it is a PR that
/// says what the tier took delivery of, exactly as raising `SHELL_BUDGET` is;
/// the standing rule is a faster `--fast`, never a bigger number.
const FAST_TIER_BUDGET: Duration = Duration::from_secs(30);

/// How many recent `--fast` runs the budget is judged over, by their **minimum**.
///
/// A cold cache, a toolchain bump and a `Cargo.toml` touch all rebuild the world,
/// and none of them is the tier being slow — but every one of them would block an
/// agent turn if a single sample decided. Contention and rebuilds only ever *add*
/// time, so the fastest of the recent runs is the warm figure the budget is
/// about; §6 M4B took the same best-of-N against §4.4's save-to-screen budget for
/// the same reason. A tier that is over budget five runs running has no warm case
/// left to appeal to.
const FAST_TIER_WINDOW: usize = 5;

/// Stop-hook tier: fmt + clippy on changed crates + tests for changed crates.
/// A clean tree passes by definition — the hook blocks on dirty-and-red only.
fn fast() -> anyhow::Result<()> {
    let changed = changed_build_paths()?;
    if changed.is_empty() {
        println!("xtask ci --fast: tree clean — green by definition");
        return Ok(());
    }
    let started = std::time::Instant::now();
    let crates = crates_touched(&changed);
    exec(cargo().args(["fmt", "--check"]), "cargo fmt --check")?;
    clippy(&crates)?;
    tests(&crates)?;
    println!("xtask ci --fast: green");
    fast_tier_budget(started.elapsed())
}

/// Hold `--fast` to [`FAST_TIER_BUDGET`] on the evidence of its own recent runs.
///
/// Deliberately *not* measured on the clean-tree path above: that branch does no
/// work, and recording its milliseconds would pin the minimum at zero forever —
/// a budget its own no-op satisfies is not a budget. The ledger lives under
/// `target/`, so `cargo clean` costs the window and nothing else.
fn fast_tier_budget(took: Duration) -> anyhow::Result<()> {
    let took_ms = u64::try_from(took.as_millis()).unwrap_or(u64::MAX);
    let ledger = workspace_root().join("target/xtask-fast-tier.txt");
    let mut window: Vec<u64> = std::fs::read_to_string(&ledger)
        .unwrap_or_default()
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    window.push(took_ms);
    let window = &window[window.len().saturating_sub(FAST_TIER_WINDOW)..];
    // Best effort: a target directory that cannot be written is not a red tier.
    let _ = std::fs::write(
        &ledger,
        window
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(" "),
    );

    println!("xtask: --fast took {took_ms} ms");
    fast_tier_verdict(window)
}

/// The verdict on a window of `--fast` durations, in milliseconds.
///
/// Split from the measurement so the budget can be *shown* to fail, the way §3's
/// greps are shown to (`mod tests` below): a number nobody has ever seen go red
/// is a budget on the same footing as a gate nobody has ever seen reject.
fn fast_tier_verdict(window: &[u64]) -> anyhow::Result<()> {
    let budget_ms = u64::try_from(FAST_TIER_BUDGET.as_millis()).unwrap_or(u64::MAX);
    let best = window.iter().copied().min().unwrap_or(0);
    println!(
        "xtask: best of the last {} --fast runs is {best} ms against a {budget_ms} ms budget (§5)",
        window.len()
    );
    anyhow::ensure!(
        best <= budget_ms || window.len() < FAST_TIER_WINDOW,
        "the fast tier has not been under its {budget_ms} ms budget once in {FAST_TIER_WINDOW} \
         runs (best {best} ms) — the Stop hook is what keeps agent turns sub-minute (§5), so the \
         fix is a faster `--fast` or a check moved down the tier ladder, never a bigger number"
    );
    Ok(())
}

/// Pre-push tier: gates 1-3 in full, the native legs of replay determinism
/// (§5.6a and 6b's x86 half, plus the dist-profile leg of 6c), and
/// dist/dist-verify feature checks (§5).
fn push() -> anyhow::Result<()> {
    exec(cargo().args(["fmt", "--check"]), "cargo fmt --check")?;
    clippy(&All)?;
    // The workspace lint above resolves features *unified*, and since M10 no
    // tier turns `gg-debug/tracy` on, so it is now the Tracy-*ful* build that
    // nothing lints. Same leg, opposite polarity: one package, one feature, on
    // its own (§6 M8 found dead code hiding in the unlinted half twice).
    exec(
        cargo().args([
            "clippy",
            "-p",
            "gg-debug",
            "--features",
            "tracy",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ]),
        "cargo clippy -p gg-debug --features tracy (the half no tier links)",
    )?;
    tracy_stays_on_loopback()?;
    exec(cargo().args(["deny", "check"]), "cargo deny check")?;
    greps()?;
    allowlist_crosscheck()?;
    crate::budgets::check()?;
    crate::public_api::check()?;
    tests(&All)?;
    fp_baseline_dist_profile()?;
    // §5.6a and the x86 half of 6b: the curated replay's hash sequence, run
    // twice here and compared against the checked-in baseline every leg shares.
    // The tests above already ran it under the dev profile; this is the report.
    crate::replay::run(&[])?;
    replay_dist_profile()?;
    // Gate 3 (§5): entry points compile + reflection codegen diff-clean. Check
    // mode: CI verifies the checked-in artifacts, it never rewrites the tree.
    crate::shaders::build_all(true)?;
    // §4.6's byte-reproducibility, which §6 M9's exit row asks to be CI-tested:
    // every demo's asset tree compiled twice *cleanly* and compared. Here and
    // not in the nightly tier, because the incremental cache rests on it — a
    // pack that differs run to run makes every warm build's reuse a guess.
    crate::assets::run(&["--check"])?;
    rejuvenation()?;
    static_link()?;
    for (pkg, feats) in [
        ("gg-runtime", "tier-dist"),
        ("gg-runtime", "tier-dist-verify"),
        ("demo-00-clear", "tier-dist"),
        ("demo-01-triangle", "tier-dist"),
        ("demo-02-mesh", "tier-dist"),
    ] {
        exec(
            cargo().args([
                "check",
                "-p",
                pkg,
                "--no-default-features",
                "--features",
                feats,
            ]),
            &format!("cargo check {pkg} --features {feats}"),
        )?;
    }
    println!("xtask ci --push: green");
    Ok(())
}

fn nightly() -> anyhow::Result<()> {
    push()?;
    crate::dist::gate()?;
    crate::probe::run(false)?;
    stress_and_miri()?;
    aarch64_leg()?;
    replay_instrumented_profile()?;
    // §5.6c in full, replay segments across a reload, §5.11's reload cases, and
    // the reload-latency instrument (§6 M5): everything that needs the shell
    // driving a real game dylib.
    crate::shell::gates(&[])?;
    crate::bench::run(&[])?;
    // Not `--record`: the numbers are a manual act like `bench --record` (§8).
    // What the tier is for is the other failure — every scenario is a text edit
    // against real source, and a rename should turn this red rather than
    // quietly measuring nothing.
    crate::dx::run(&[])?;
    gpu_tests()?;
    // Instrumented shaders catch what the layer alone cannot see: an out-of-range
    // bindless index, a read off the end of a device address (§8's sync/upload
    // risk row). Nightly rather than weekly — 28 s windowless on the pin.
    crate::gpuav::run(&[])?;
    golden_suite()?;
    println!(
        "xtask ci --nightly: green (windowless by construction — windowed WSI coverage is \
         `cargo xtask interactive`, manual)"
    );
    Ok(())
}

/// M3's two slow gates (§4.2): the 10k-tick archetype-churn stress, and Miri
/// over `gg-ecs`'s `unsafe`.
///
/// Nightly rather than push because the churn run is minutes, not seconds, and
/// Miri is an order of magnitude slower still. Both are cheap to run and
/// expensive to skip: the churn digest is the cross-architecture claim, and Miri
/// is the only thing that reads the aliasing argument in `view.rs` rather than
/// trusting its comment.
fn stress_and_miri() -> anyhow::Result<()> {
    exec(
        cargo().args([
            "nextest",
            "run",
            "-p",
            "gg-ecs",
            "--run-ignored",
            "all",
            "-E",
            "test(churn_at_full_scale_neither_leaks_nor_diverges)",
        ]),
        "gg-ecs archetype-churn stress (100k entities, 50 archetypes, 10k ticks)",
    )?;
    // `views`/`query` only: they hold every raw pointer in the crate. `churn`
    // installs a global allocator and `reject` shells out to cargo — neither is
    // something Miri should be asked to interpret.
    exec(
        cargo().env("MIRIFLAGS", "-Zmiri-strict-provenance").args([
            &format!("+{}", crate::public_api::NIGHTLY),
            "miri",
            "test",
            "-p",
            "gg-ecs",
            "--test",
            "views",
            "--test",
            "query",
        ]),
        "miri over gg-ecs column views",
    )
}

/// `cargo xtask interactive` — the manual windowed suite (§1.5): everything
/// that creates an OS window, extracted from the automated tiers because
/// "invisible" is not a property CI may rely on (WSLg mirrors mapped windows
/// onto the real desktop; minimize/restore maps them everywhere). A human runs
/// this when touching WSI code — expect window activity on WSLg/taskbar.
/// The manual WSI suite — and what it actually puts on the screen is less than
/// its name suggests, which is worth writing down rather than rediscovering.
///
/// Every test leg below uses `WindowDesc::invisible`, and both demo legs run
/// under `GG_HEADLESS=1`: **on Windows this suite is very nearly windowless.**
/// They are `#[ignore]`d anyway because §1.5's enforcement is real on Win32 and
/// X11 only — on Wayland `set_visible` is a no-op, so the same tests do reach a
/// desktop there, and an automated tier that ran them would violate §1.5 on one
/// platform out of three. What *does* present visibly, on every platform, is
/// [`shell_run`] and [`replay_run`]: 100 frames each, a second or two apiece.
pub fn interactive() -> anyhow::Result<()> {
    // The window-creating tests: swapchain recreation + resize/minimize storm
    // (gg-rhi) and the off-screen-parking §1.5 regressions (gg-platform).
    let mut cmd = cargo();
    cmd.args([
        "nextest",
        "run",
        "-p",
        "gg-rhi",
        "-p",
        "gg-platform",
        "--run-ignored",
        "ignored-only",
        // `#[ignore]` is not one reason. Every other ignored test in these two
        // crates is ignored for §1.5 — it creates a window — and this suite is
        // where those run. `device_lost` is ignored because it *wedges the GPU
        // on purpose*, and sweeping it in here ran it on the pinned lavapipe,
        // where the hang is this process on every core rather than a device
        // with a watchdog behind it. The test now refuses a software rasterizer
        // itself; this is the other half of that fix (§6 M8).
        "-E",
        "not binary(device_lost)",
    ]);
    lavapipe_env(&mut cmd)?;
    // Linux runs over X11 (XWayland): WSLg's Weston drops the Wayland
    // connection under minimize-spam on an unmapped surface (broken pipe →
    // VK_ERROR_SURFACE_LOST_KHR); the suite's job is the resize/minimize event
    // storm, and X11 delivers it. The contract platform is the Windows host.
    if !cfg!(windows) {
        cmd.env_remove("WAYLAND_DISPLAY");
    }
    exec(
        &mut cmd,
        "windowed suite: swapchain torture + resize/minimize storms (§4.3, §1.5)",
    )?;
    demo_runs()?;
    shell_run()?;
    replay_run()?;
    crate::dist::demo_runs()?;
    println!("xtask interactive: green (manual windowed suite — not part of any automated tier)");
    Ok(())
}

/// The end-to-end half of §5.6, and the only place it can live: the *windowed*
/// demo replays the curated file and must land on the hash the gate recorded
/// for it. Everything below the window — action state, the sim, extract — is
/// the same code the gate drives, but this is the path a player's bug report
/// actually takes, and a divergence between the two would be invisible to a
/// sim-level test.
///
/// Windowed, therefore manual (§1.5).
fn replay_run() -> anyhow::Result<()> {
    let replay = demo_02_mesh::gate::replay_path(demo_02_mesh::gate::CURATED);
    let baseline = demo_02_mesh::gate::parse_baseline(&std::fs::read_to_string(
        demo_02_mesh::gate::baseline_path(demo_02_mesh::gate::CURATED),
    )?)?;
    let (_, last) = baseline
        .last()
        .copied()
        .ok_or_else(|| anyhow::anyhow!("the curated baseline is empty"))?;

    let mut cmd = cargo();
    cmd.args([
        "run",
        "-p",
        "demo-02-mesh",
        "--",
        "--replay",
        &replay.display().to_string(),
        "--expect-hash",
        &format!("{last:032x}"),
    ]);
    cmd.env("GG_HEADLESS", "1");
    lavapipe_env(&mut cmd)?;
    exec(
        &mut cmd,
        "demo 02 replays the curated file and lands on its recorded hash (§5.6)",
    )
}

/// Point child processes at the pinned lavapipe (§5.4). Windows pins via
/// mesa-dist-win by SHA-256; the WSL container pin is still the deferred
/// machine the probe names — until it lands, Linux forces the system lvp ICD
/// so at least the *driver* is lavapipe, not whatever enumerates first.
fn lavapipe_env(cmd: &mut std::process::Command) -> anyhow::Result<()> {
    if cfg!(windows) {
        cmd.env("VK_DRIVER_FILES", crate::probe::ensure_lavapipe()?);
    } else {
        cmd.env("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/lvp_icd.json");
    }
    Ok(())
}

/// Gate 4 (§5): the headless GPU tests — gg-rhi's offscreen suite and
/// gg-platform's headless-law tests — against the pinned lavapipe, validation
/// on, any message a failure (the tests enforce that themselves via the
/// shutdown report). Window-creating tests are `#[ignore]` and belong to
/// `cargo xtask interactive` (§1.5).
fn gpu_tests() -> anyhow::Result<()> {
    let mut cmd = cargo();
    cmd.args(["nextest", "run", "-p", "gg-rhi", "-p", "gg-platform"]);
    lavapipe_env(&mut cmd)?;
    exec(&mut cmd, "headless GPU tests on pinned lavapipe (§5.4)")?;
    // A build of its own, because Tracy's GPU path is `#[cfg]`-absent without
    // the feature and the shell that exercises it for real needs a window
    // (§1.5). An offscreen context stands in and produces the same readings.
    let mut cmd = cargo();
    cmd.args(["nextest", "run", "-p", "gg-debug", "--features", "tracy"]);
    lavapipe_env(&mut cmd)?;
    exec(&mut cmd, "Tracy GPU zones on pinned lavapipe (§4.8)")
}

/// Gate 5 (§5), §4.10's harness: the golden suite on the pinned lavapipe —
/// offscreen render, readback, both gates against the checked-in references.
/// Also gate 7's winit half: the harness binary must be headless *by linkage*,
/// so its bytes are scanned for winit before it runs.
fn golden_suite() -> anyhow::Result<()> {
    exec(
        cargo().args(["build", "-p", "gg-golden"]),
        "build gg-golden",
    )?;
    let exe = workspace_root()
        .join("target/debug")
        .join(if cfg!(windows) {
            "gg-golden.exe"
        } else {
            "gg-golden"
        });
    let bytes = std::fs::read(&exe)?;
    let needle = b"winit";
    anyhow::ensure!(
        !bytes.windows(needle.len()).any(|w| w == needle),
        "gg-golden links winit — the harness must be headless by linkage (§1.5, §5 gate 7)"
    );
    println!("xtask: gg-golden is winit-free by linkage (§5 gate 7)");

    let mut cmd = cargo();
    cmd.args(["run", "-p", "gg-golden", "--", "run"]);
    lavapipe_env(&mut cmd)?;
    exec(&mut cmd, "golden suite on pinned lavapipe (§4.10)")?;

    // The gate's own gate (§4.10, M7): a suite that cannot fail is not a suite.
    // It runs *after* the compare so a green suite is never the reason this was
    // skipped, and on the same pin so the numbers it prints are the tier's.
    let mut cmd = cargo();
    cmd.args(["run", "-p", "gg-golden", "--", "verify-gates"]);
    lavapipe_env(&mut cmd)?;
    exec(
        &mut cmd,
        "golden gates reject a one-pixel change and forgive rounding noise (§4.10)",
    )?;

    // §6 M9's "load to first frame < 500 ms". Here rather than in the shell's
    // gates because the shell needs a window to hold a renderer (§1.5) and this
    // harness is windowless by the linkage proven above. Demo 04's pack is two
    // kilobytes, so what this leg gates is that the *clock exists and stops* —
    // the number that means anything is a level-sized one, and it is measured
    // by hand and recorded, the way `xtask bench --record` is (§4.11).
    let mut cmd = cargo();
    cmd.args(["run", "-p", "gg-golden", "--", "load"]);
    lavapipe_env(&mut cmd)?;
    exec(&mut cmd, "pack load-to-first-frame under budget (§6 M9)")?;
    render_graph_dump()
}

/// §4.5's `--dump-render-graph`, as a gate rather than a convenience.
///
/// It runs against `gg-golden` and not the shell for a reason worth stating: the
/// shell is *windowless* under `GG_HEADLESS`, so it has no renderer to ask and a
/// flag on it could never run in an automated tier (§1.5) — an artifact no gate
/// can produce is an artifact that rots. Here the dump comes off the same
/// compiled graph the harness just rendered with, so it is the executed order by
/// construction; `gg-render`'s offscreen test asserts that equality directly.
fn render_graph_dump() -> anyhow::Result<()> {
    let mut cmd = cargo();
    cmd.args(["run", "-q", "-p", "gg-golden", "--", "graph"]);
    lavapipe_env(&mut cmd)?;
    let dump = run_capture(&mut cmd, "gg-golden graph")?;
    for expected in ["forward-opaque", "readback", "frame-end", "barrier"] {
        anyhow::ensure!(
            dump.contains(expected),
            "the render-graph dump names no `{expected}` — §4.5's dump must be readable and \
             complete:\n{dump}"
        );
    }
    println!(
        "xtask: render-graph dump readable ({} lines)",
        dump.lines().count()
    );
    Ok(())
}

/// The demo WSI runs (part of the manual windowed suite): every demo runs 100
/// frames against a real (invisible) window's swapchain without validation
/// errors or leaks — the demo binaries exit nonzero on either. Creates
/// windows, so never part of an automated tier (§1.5).
fn demo_runs() -> anyhow::Result<()> {
    for demo in ["demo-00-clear", "demo-01-triangle", "demo-02-mesh"] {
        let mut cmd = cargo();
        cmd.args(["run", "-p", demo, "--", "--frames", "100"]);
        cmd.env("GG_HEADLESS", "1");
        lavapipe_env(&mut cmd)?;
        exec(
            &mut cmd,
            &format!("demo {demo}, 100 frames headless on pinned lavapipe"),
        )?;
    }
    Ok(())
}

/// The shell's own WSI leg: a game dylib loaded, a real window, the §4.5 v0 pass
/// presenting (§6 M5).
///
/// `GG_HEADLESS` is deliberately *unset* — the shell answers it by skipping
/// windowing entirely, so a headless run proves nothing about the swapchain path
/// a player takes. That is precisely why this is in `interactive` and in no
/// automated tier (§1.5).
/// The forced-rejuvenation criterion (§6 M5): a session whose leak budget is
/// zero rejuvenates on its first reload — snapshot, restart the host, restore,
/// resume — and the world it comes back to is the world it left.
///
/// Windowless, so this is a CI tier and not `interactive` (§1.5). The successor
/// process inherits this pipe, which is what makes "did it come back" observable
/// at all: `wait_with_output` returns only once the *last* process in the chain
/// has closed it.
fn rejuvenation() -> anyhow::Result<()> {
    use std::process::Stdio;

    let root = crate::util::workspace_root();
    exec(
        cargo().args(["build", "-p", "demo-03-reload", "-p", "gg-runtime"]),
        "build demo 03 + the shell",
    )?;

    // A directory of its own: the watcher watches a *directory*, so rewriting the
    // artifact under target/debug would be a reload event for anything else
    // pointed there.
    let name = if cfg!(windows) {
        "demo_03_reload.dll"
    } else {
        "libdemo_03_reload.so"
    };
    let built = root.join("target/debug").join(name);
    let dir = root.join("target/rejuvenate");
    std::fs::create_dir_all(&dir)?;
    let game = dir.join(name);
    std::fs::copy(&built, &game)?;

    let mut cmd = std::process::Command::new(root.join("target/debug").join(if cfg!(windows) {
        "gg-runtime.exe"
    } else {
        "gg-runtime"
    }));
    cmd.arg("--game")
        .arg(&game)
        .args(["--frames", "300000", "--leak-budget", "0"])
        .env("GG_HEADLESS", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let child = cmd.spawn()?;

    // Only a *reload* charges the leak budget, so the rewrite is the trigger and
    // there is no rejuvenation without one. 300k headless frames is around a
    // second of runtime; a busier machine makes that window wider, never
    // narrower, which is the direction a timing assumption should fail in.
    std::thread::sleep(std::time::Duration::from_millis(400));
    std::fs::copy(&built, &game)?;
    let out = child.wait_with_output()?;

    let log = format!(
        "{}{}",
        crate::util::plain(&out.stdout),
        crate::util::plain(&out.stderr)
    );
    let line = |needle: &str| {
        log.lines()
            .find(|l| l.contains(needle))
            .map(str::to_owned)
            .ok_or_else(|| anyhow::anyhow!("no `{needle}` line in:\n{log}"))
    };
    let staged = line("rejuvenating")?;
    let resumed = line("rejuvenated")?;
    let (left_at, came_back) = (
        crate::util::field_u64(&staged, "tick")?,
        crate::util::field_u64(&resumed, "tick")?,
    );
    anyhow::ensure!(
        left_at > 0 && left_at == came_back,
        "resumed at tick {came_back}, left off at {left_at}"
    );
    anyhow::ensure!(
        crate::util::field_u64(&resumed, "entities")? > 0,
        "came back to an empty world: {resumed}"
    );
    // The demo's bootstrap is idempotent and logs once. Twice would mean the
    // successor rebuilt the world instead of restoring it — passing the tick
    // assertion above while failing the criterion it exists for.
    let births = log.matches("open for business").count();
    anyhow::ensure!(births == 1, "the game bootstrapped {births} times");
    anyhow::ensure!(
        log.contains("clean exit"),
        "the successor did not finish its run:\n{log}"
    );
    println!(
        "xtask: rejuvenation: restarted at tick {left_at}, resumed with \
         {} entities, one bootstrap",
        crate::util::field_u64(&resumed, "entities")?
    );
    Ok(())
}

/// Gate 9 (§5), live from M5: the statically-linked systems-table variant still
/// compiles. Dormant — it runs no world and gates nothing further until the
/// fallback is activated, at which point §5.6e's link-mode equivalence gate
/// activates with it (§2, Game-code boundary row).
///
/// The crate's own test goes one step further and *links* it, which is where the
/// interesting failure lives: a game crate contributes `#[no_mangle]` symbols
/// and no Rust items, so rustc leaves its rlib out of the link unless something
/// names it — a variant that compiles and does not link is exactly the rot this
/// gate exists to catch, and a `cargo check` cannot see it.
fn static_link() -> anyhow::Result<()> {
    exec(
        cargo().args(["check", "-p", "gg-static-link"]),
        "dormant static-link variant compiles (§5.9)",
    )
}

fn shell_run() -> anyhow::Result<()> {
    let mut build = cargo();
    build.args(["build", "-p", "demo-03-reload", "-p", "gg-runtime"]);
    exec(&mut build, "build demo 03 + the shell")?;

    let root = crate::util::workspace_root();
    let dylib = root.join("target/debug").join(if cfg!(windows) {
        "demo_03_reload.dll"
    } else {
        "libdemo_03_reload.so"
    });
    let mut cmd = std::process::Command::new(root.join("target/debug").join(if cfg!(windows) {
        "gg-runtime.exe"
    } else {
        "gg-runtime"
    }));
    cmd.arg("--game")
        .arg(&dylib)
        .arg("--input")
        .arg(root.join("demos/03-reload/input.toml"))
        .args(["--frames", "100"]);
    lavapipe_env(&mut cmd)?;
    exec(&mut cmd, "demo 03 under the shell, 100 windowed frames")
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
        // gg-ecs joins at M3: the canonical hash and the churn digest are
        // cross-architecture claims, so the leg that proves aarch64 has to run
        // them. The full-scale churn stays `#[ignore]`d here — under qemu-user
        // it would cost most of an hour to prove what the moderate-scale frozen
        // digest already proves.
        //
        // demo-02-mesh joins at M4B as the *third leg of §5.6b*: the curated
        // replay and the chaos seeds, compared against the same checked-in
        // baselines the two x86 hosts compare against. `--no-default-features`
        // is what makes that possible at all — the sim half has no GPU and no
        // winit in its graph, so qemu never has to emulate a Vulkan loader.
        exec(
            cargo().args([
                "nextest",
                "run",
                "-p",
                "gg-math",
                "-p",
                "gg-ecs",
                "-p",
                "demo-02-mesh",
                "--no-default-features",
                "--features",
                "gate",
                "--target",
                "aarch64-unknown-linux-gnu",
                "--cargo-profile",
                profile,
            ]),
            &format!(
                "gg-math + gg-ecs + replay determinism on aarch64 under qemu ({profile} profile)"
            ),
        )?;
        // demo-05-many joins at M10, and it is a separate invocation because it
        // has no `gate` feature to select — it is a game crate whose whole graph
        // is sim, plus `gg-scene` as a dev-dependency. What it adds to the leg
        // is the *hierarchy*: five thousand transforms composed host-side per
        // tick, inside what the canonical hash covers (§4.7).
        exec(
            cargo().args([
                "nextest",
                "run",
                "-p",
                "demo-05-many",
                "--target",
                "aarch64-unknown-linux-gnu",
                "--cargo-profile",
                profile,
            ]),
            &format!("hierarchy determinism on aarch64 under qemu ({profile} profile)"),
        )?;
    }
    Ok(())
}

/// §1.5's neighbour: no automated tier may put a *socket* on the user's network
/// either. Tracy's client binds a listener from a static constructor — before
/// `main`, with no profiler attached — and Windows raises its firewall dialog
/// once per binary path that binds a non-loopback one, which nextest's build
/// hashes make new on every rebuild. `only-localhost` is what holds it at
/// 127.0.0.1. Asserted rather than assumed because features arrive by *union*:
/// `profiling` already re-enables `broadcast` through `profile-with-tracy`, and
/// the same route could as easily drop ours in a bump.
fn tracy_stays_on_loopback() -> anyhow::Result<()> {
    // Absence is proven, never inferred from a failure. `cargo tree -i` exits
    // nonzero for "not in the graph" *and* for a renamed package, a dropped
    // feature or a resolver error — reading the second as the first made this
    // gate fail open. So the tier is resolved once without `-i`, which errors
    // only if `gg-runtime`/`tier-instrumented` stopped existing, and the answer
    // is read out of that.
    let args = [
        "tree",
        "-p",
        "gg-runtime",
        "--no-default-features",
        "--features",
        "tier-instrumented",
        "-e",
        "features",
    ];
    let tree = run_capture(
        cargo().args(args),
        "cargo tree (tier-instrumented feature graph)",
    )?;
    if !tree.contains("tracy-client") {
        println!("xtask: tier-instrumented links no tracy-client — nothing binds");
        return Ok(());
    }
    // Present: the inverted query must now succeed too, and name the feature.
    let inverted = run_capture(
        cargo().args(args).args(["-i", "tracy-client"]),
        "cargo tree -i tracy-client",
    )?;
    anyhow::ensure!(
        inverted.contains("only-localhost"),
        "tier-instrumented resolves tracy-client without `only-localhost`: its listener would \
         bind every interface, and the desk would collect a firewall prompt per build hash. \
         Fix the feature list in the workspace Cargo.toml, not this gate."
    );
    println!("xtask: tracy's listener stays on loopback (`only-localhost` resolved)");
    Ok(())
}

/// The one codegen configuration no hash ever measured (§5.6c): thin LTO at
/// full optimization, which is neither dev's nor dist's. It is also the tier a
/// bug replay gets profiled under, where a silent divergence would cost most.
fn replay_instrumented_profile() -> anyhow::Result<()> {
    exec(
        cargo().args([
            "nextest",
            "run",
            "-p",
            "demo-02-mesh",
            "--no-default-features",
            "--features",
            "gate",
            "--cargo-profile",
            "instrumented",
        ]),
        "replay determinism under the instrumented profile (§5.6c)",
    )
}

fn weekly() -> anyhow::Result<()> {
    nightly()?;
    // Standing from M4B (§6, M0A's deferred schedule): the two gates that check
    // the repository rather than the code.
    crate::fresh::clone_gate()?;
    crate::fresh::update_canary()?;
    println!("xtask ci --weekly: green");
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

/// §5.6c, as far as this milestone can take it: the replay determinism tests
/// under **dist codegen** — fat LTO, `codegen-units = 1`, full optimization —
/// against the same checked-in baseline the dev-profile run used. Optimization
/// level may not touch sim results, and dev alone never exercises the two knobs
/// that would.
///
/// The full 6c gate (record under dist, replay under dist-verify, dev and
/// instrumented) needs `gg-runtime` to own the loop and the recorder, which is
/// M5's; this is the half that can be proven with the demo driving itself.
fn replay_dist_profile() -> anyhow::Result<()> {
    exec(
        cargo().args([
            "nextest",
            "run",
            "-p",
            "demo-02-mesh",
            "--no-default-features",
            "--features",
            "gate",
            "--cargo-profile",
            "dist",
        ]),
        "replay determinism under the dist profile (§5.6c)",
    )
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
        if let Some(rest) = p.strip_prefix("demos/")
            && let Some((dir, _)) = rest.split_once('/')
        {
            // demos/00-clear → package demo-00-clear (§3 layout convention).
            crates.insert(format!("demo-{dir}"));
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

/// `needle` as a whole path segment, not as a suffix of a longer identifier.
///
/// The distinction is not pedantry: `gg_ecs::hash::` ends in `ash::`, and a
/// plain substring search reports every file in the crate. The gate means the
/// *crate* `ash`, so a preceding identifier character disqualifies the hit.
fn contains_path(text: &str, needle: &str) -> bool {
    let bytes = text.as_bytes();
    text.match_indices(needle)
        .any(|(at, _)| at == 0 || !(bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_'))
}

fn greps() -> anyhow::Result<()> {
    let (violations, scanned) = scan(&workspace_root())?;
    anyhow::ensure!(
        violations.is_empty(),
        "grep gate failed:\n{}",
        violations.join("\n")
    );
    println!("xtask: grep gates clean ({scanned} files)");
    Ok(())
}

/// The §3 greps as a function of a source tree, so the gate can be *pointed at*
/// a tree with each violation deliberately planted (`mod tests`) rather than
/// only ever at a clean one. A gate that has never once been red is a gate
/// nobody has tested — §5's "reject a plant" criterion in its cheapest form.
///
/// Returns the violations and how many files were read.
fn scan(root: &std::path::Path) -> anyhow::Result<(Vec<String>, usize)> {
    let mut files = Vec::new();
    walk_rs(&root.join("crates"), &mut files);
    walk_rs(&root.join("demos"), &mut files);
    // The harness and CI's own source sat outside every §3 grep on no stated
    // ground: no SAFETY rule over `tools/`, none over the `unsafe` in this
    // crate's own Vulkan probe. Two rules below carve `xtask` back out, and
    // they say why at the site.
    walk_rs(&root.join("tools"), &mut files);
    walk_rs(&root.join("xtask"), &mut files);

    let mut violations = Vec::new();
    for file in &files {
        let rel = file.strip_prefix(root).unwrap_or(file);
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        let text = std::fs::read_to_string(file)?;
        let lines: Vec<&str> = text.lines().collect();
        // `xtask` is outside the containment seam by charter — deny.toml already
        // names it in `ash`'s wrappers for the §6 M0A capability probe — and
        // this file must spell both bans literally to be one.
        let spells_the_bans = rel_str.starts_with("xtask/");

        // API containment (§3): vk::/ash:: tokens live in gg-rhi alone.
        if !rel_str.starts_with("crates/gg-rhi/")
            && !spells_the_bans
            && let Some(tok) = ["ash::", "vk::"]
                .into_iter()
                .find(|t| contains_path(&text, t))
        {
            violations.push(format!("{rel_str}: `{tok}` token outside gg-rhi (§3)"));
        }

        // Float-time declarations (§2 Sim time row), matched off the
        // declaration: visibility of any shape, whitespace either side of the
        // colon, and the newtype spelling. Scoped to the sim half — see
        // [`NON_SIM_TREES`]. What is still owed is M3's scoping to *hashed
        // components*: nothing here knows which those are, so [`is_time_ish`]
        // stands in for the type question.
        if !NON_SIM_TREES.iter().any(|tree| rel_str.starts_with(tree)) {
            for (lineno, line) in lines.iter().enumerate() {
                let t = line.trim_start();
                if t.starts_with("//") {
                    continue;
                }
                if let Some(name) = float_time_newtype(t) {
                    violations.push(format!(
                        "{rel_str}:{}: float time newtype `{name}` (§2 Sim time row)",
                        lineno + 1
                    ));
                    continue;
                }
                // A method is not a field — and `fn` inside a comment used to
                // trip this escape, silencing the field on the next line.
                if t.contains("fn ") {
                    continue;
                }
                let Some((name, ty)) = strip_visibility(t).split_once(':') else {
                    continue;
                };
                let (name, ty) = (name.trim_end(), ty.trim_start());
                // One bare identifier or it is not a field: segment matching
                // would otherwise read `let ms: f32 = …` — a local, and none of
                // this row's business — as a declaration.
                if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                    continue;
                }
                if is_time_ish(name) && is_float(ty) {
                    violations.push(format!(
                        "{rel_str}:{}: float time field `{name}` (§2 Sim time row)",
                        lineno + 1
                    ));
                }
            }
        }

        // Every `unsafe` block and `unsafe impl` carries **its own** `// SAFETY:`
        // within the preceding 8 lines (§4.2 M3 exit, and the standing rule for
        // gg-rhi). `unsafe fn` *declarations* are exempt: their obligation is on
        // the caller and belongs in the doc comment, not in a SAFETY note.
        let mut claimed = vec![false; lines.len()];
        for (lineno, line) in lines.iter().enumerate() {
            if unsafe_site(line) && !claim_safety_note(&lines, &mut claimed, lineno) {
                violations.push(format!(
                    "{rel_str}:{}: unsafe without a `// SAFETY:` note of its own (§4.2)",
                    lineno + 1
                ));
            }
        }

        // Game crates: no state smuggled across a reload (§4.2.2). `static mut`
        // is one spelling of a dozen — a `static` holding an atomic, a lock, a
        // cell or a lazy initializer survives a reload exactly as well — so the
        // match is on the *declaration*, which is also what keeps a `let` local
        // and a comment naming one out of it.
        if rel_str.starts_with("demos/") {
            for (lineno, line) in lines.iter().enumerate() {
                if line.trim_start().starts_with("//") {
                    continue;
                }
                if let Some(what) = retained_state(line) {
                    violations.push(format!(
                        "{rel_str}:{}: {what} in a game crate (§4.2.2)",
                        lineno + 1
                    ));
                }
            }
        }

        // Hand-written barriers (§4.5, §6 M6): every `synchronization2` barrier
        // in the engine is *derived* by the render graph, so the tokens that
        // spell one live in the graph's execution layer and in the staging
        // ring's queue-family ownership transfers — which are a property of a
        // transfer paired across two queues, not a pass dependency the
        // single-queue v1 graph models.
        if !BARRIER_SITES.contains(&rel_str.as_str())
            && !spells_the_bans
            && let Some(tok) = BARRIER_TOKENS.into_iter().find(|t| contains_path(&text, t))
        {
            violations.push(format!(
                "{rel_str}: `{tok}` — barriers are derived by the render graph, not written \
                 (§4.5); the derivation lives in {}",
                BARRIER_SITES[0]
            ));
        }
    }

    Ok((violations, files.len()))
}

/// A declaration with its visibility removed — `pub`, `pub(crate)`, `pub(in …)`.
/// Which of the three it is has never been a question either gate above asks,
/// and stripping only `"pub "` is how `pub(crate) dt: f32` used to pass.
fn strip_visibility(line: &str) -> &str {
    let t = line.trim_start();
    let Some(rest) = t.strip_prefix("pub") else {
        return t;
    };
    match rest.strip_prefix('(').and_then(|r| r.split_once(')')) {
        Some((_, after)) => after.trim_start(),
        // Not an identifier that merely begins with those three letters.
        None if rest.starts_with(char::is_whitespace) => rest.trim_start(),
        None => t,
    }
}

/// Where §2's Sim time row does **not** bind: the halves that measure a real
/// clock. A GPU timestamp, a frame p99, a Tracy zone and a rebuild latency are
/// wall clock in milliseconds, and float is the correct type for every one of
/// them — §1.4's membrane is exactly this line. Scoping the gate is what keeps it
/// from growing the per-file exception list that ends with the field it was
/// written for exempted. `xtask/` is additionally here because this file must
/// spell the planted declarations literally in order to *be* the gate.
const NON_SIM_TREES: [&str; 5] = [
    "crates/gg-debug/",
    "crates/gg-render/",
    "crates/gg-rhi/",
    "tools/",
    "xtask/",
];

/// Name segments that name a clock (§2 Sim time row).
///
/// A pattern rather than the six literals this replaces: the escapes were never
/// exotic — `cooldown`, `timestamp`, `age_secs`, `accumulator` — and a name list
/// is a game of whack-a-mole the gate loses by construction. Matched
/// whole-segment (see [`segments`]), because a substring test reads `damage` as
/// an `age` and a gate that cries wolf is one that gets switched off.
///
/// `remaining` and `left` are deliberately absent: bare, they are as often a
/// count as a clock — `gg-ecs`'s side-table fixture has an order's `remaining:
/// f64` — while `remaining_secs` and `time_remaining` are caught by their other
/// segment anyway.
const TIME_WORDS: [&str; 29] = [
    "accum",
    "accumulator",
    "age",
    "clock",
    "cooldown",
    "countdown",
    "deadline",
    "delay",
    "dt",
    "duration",
    "elapsed",
    "interval",
    "lifetime",
    "millis",
    "ms",
    "nanos",
    "ns",
    "period",
    "sec",
    "seconds",
    "secs",
    "since",
    "time",
    "timer",
    "timers",
    "times",
    "timestamp",
    "ttl",
    "uptime",
];

/// Whether an identifier names a clock: any whole segment in [`TIME_WORDS`].
fn is_time_ish(name: &str) -> bool {
    segments(name).iter().any(|s| TIME_WORDS.contains(&&**s))
}

/// Lowercase word segments of an identifier, split on `_` and on camel humps —
/// so one predicate reads a snake_case field and a CamelCase newtype alike.
fn segments(name: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut word = String::new();
    for c in name.chars() {
        if c == '_' || !(c.is_ascii_alphanumeric()) {
            if !word.is_empty() {
                out.push(std::mem::take(&mut word));
            }
        } else if c.is_ascii_uppercase() && !word.is_empty() {
            out.push(std::mem::take(&mut word));
            word.push(c.to_ascii_lowercase());
        } else {
            word.push(c.to_ascii_lowercase());
        }
    }
    if !word.is_empty() {
        out.push(word);
    }
    out
}

/// Whether a type position opens with a float width, `Option` included.
fn is_float(ty: &str) -> bool {
    ["f32", "f64"]
        .into_iter()
        .any(|w| ty.starts_with(w) || ty.starts_with(&format!("Option<{w}>")))
}

/// `struct Timer(f32);` — the same ban worn as a newtype, which the field match
/// cannot see: there is no field name, and the name under test is the type's.
/// Any float in the tuple counts, so `struct Cooldown(u32, f32)` is still a
/// float clock.
fn float_time_newtype(line: &str) -> Option<&str> {
    let decl = strip_visibility(line).strip_prefix("struct")?;
    // The keyword, not the head of an identifier.
    if !decl.starts_with(char::is_whitespace) {
        return None;
    }
    let (name, rest) = decl.trim_start().split_once('(')?;
    // Generics are not part of the name: `struct Timer<T>(f32)`.
    let name = name.trim_end().split('<').next()?;
    let floaty = rest
        .split(')')
        .next()?
        .split(',')
        .any(|field| is_float(strip_visibility(field.trim())));
    (is_time_ish(name) && floaty).then_some(name)
}

/// An `unsafe` *site*: a block or an `unsafe impl`. A declaration is not one —
/// its obligation is the caller's — and neither is the word inside a string
/// literal, which is what this crate's own needles and its planted fixtures
/// became when the scan reached `xtask/`. Quote parity rather than a lexer: no
/// real site in the tree follows an unbalanced quote on its own line.
fn unsafe_site(line: &str) -> bool {
    if line.trim_start().starts_with("//") {
        return false;
    }
    const KW: &str = "unsafe";
    line.match_indices(KW)
        .filter(|(at, _)| line[..*at].matches('"').count().is_multiple_of(2))
        .any(|(at, _)| {
            let rest = line[at + KW.len()..].trim_start();
            rest.starts_with('{') || rest.starts_with("impl")
        })
}

/// Claim the nearest unclaimed `// SAFETY:` note in the eight lines above
/// `site`, reporting whether there was one. One note, one site: a single line
/// above three blocks used to justify all three, which is a decoy rather than a
/// justification (§4.2).
fn claim_safety_note(lines: &[&str], claimed: &mut [bool], site: usize) -> bool {
    for at in (site.saturating_sub(8)..site).rev() {
        if !claimed[at] && is_safety_note(lines[at]) {
            claimed[at] = true;
            return true;
        }
    }
    false
}

/// A real `// SAFETY:` note: a line comment whose content *opens* with the
/// token, optionally qualified the way the teardown paths' `SAFETY (all arms):`
/// is. What this rejects is everything a `contains("SAFETY")` accepted — a `///`
/// doc line about safety, a string literal, `SAFETY_MARGIN`, and the note that
/// says there is nothing to note.
fn is_safety_note(line: &str) -> bool {
    let Some(body) = line.trim_start().strip_prefix("//") else {
        return false;
    };
    // `///` and `//!` fall out here: neither `/` nor `!` opens `SAFETY`.
    let Some(rest) = body.trim_start().strip_prefix("SAFETY") else {
        return false;
    };
    let rest = rest.trim_start();
    rest.strip_prefix('(')
        .and_then(|r| r.split_once(')'))
        .map_or(rest, |(_, after)| after.trim_start())
        .starts_with(':')
}

/// What this line declares that would outlive a reload behind the host's back
/// (§4.2.2), if anything.
fn retained_state(line: &str) -> Option<String> {
    let macros = ["thread_local!", "lazy_static!"];
    if let Some(mac) = macros.into_iter().find(|m| line.contains(m)) {
        return Some(format!("`{mac}`"));
    }
    let decl = strip_visibility(line).strip_prefix("static")?;
    // The keyword, not the head of an identifier — and `static   mut` too.
    if !decl.starts_with(char::is_whitespace) {
        return None;
    }
    let decl = decl.trim_start();
    if decl
        .strip_prefix("mut")
        .is_some_and(|r| r.starts_with(char::is_whitespace))
    {
        return Some("`static mut`".to_string());
    }
    RETAINED_STATE_TYPES
        .into_iter()
        .find(|t| contains_path(decl, t))
        .map(|t| format!("`static` holding `{t}`"))
}

/// What makes a `static` mutable or lazily initialized — the quiet spellings of
/// `static mut` (§4.2.2). Whole-segment matched, so `Cell` does not also fire on
/// the `RefCell` line above it.
const RETAINED_STATE_TYPES: [&str; 20] = [
    "AtomicBool",
    "AtomicPtr",
    "AtomicI8",
    "AtomicI16",
    "AtomicI32",
    "AtomicI64",
    "AtomicIsize",
    "AtomicU8",
    "AtomicU16",
    "AtomicU32",
    "AtomicU64",
    "AtomicUsize",
    "OnceLock",
    "OnceCell",
    "Lazy",
    "Mutex",
    "RwLock",
    "RefCell",
    "UnsafeCell",
    "Cell",
];

/// What spelling a hand-written barrier out looks like.
const BARRIER_TOKENS: [&str; 4] = [
    "cmd_pipeline_barrier2",
    "ImageMemoryBarrier2",
    "BufferMemoryBarrier2",
    "MemoryBarrier2",
];

/// The two files allowed to name them, and why (see the gate above).
const BARRIER_SITES: [&str; 2] = ["crates/gg-rhi/src/graph.rs", "crates/gg-rhi/src/upload.rs"];

/// The `rayon` ban's wrappers in deny.toml must cover every exemption in
/// determinism-allowlist.toml (§4.1) — two files, one truth, machine-checked.
fn allowlist_crosscheck() -> anyhow::Result<()> {
    let root = workspace_root();
    // toml::from_str, not str::parse: since toml 0.9 `FromStr for Value` parses a
    // TOML *value*, not a document, so `.parse()` on a file starting with a table
    // header reads it as an array literal and errors.
    let allow: toml::Value = toml::from_str(&std::fs::read_to_string(
        root.join("determinism-allowlist.toml"),
    )?)?;
    let deny: toml::Value = toml::from_str(&std::fs::read_to_string(root.join("deny.toml"))?)?;

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

    // The direction that can actually break, and the one this gate did not
    // check: a `wrappers` entry is *how* the rayon ban gets silenced, so a crate
    // added there to make cargo-deny green is an exemption taken without the
    // review §4.1 requires. The other direction can only fire if someone writes
    // an allowlist row and then declines to use it.
    let unreviewed: Vec<&&str> = wrappers.iter().filter(|c| !exempt.contains(**c)).collect();
    anyhow::ensure!(
        unreviewed.is_empty(),
        "deny.toml lets {unreviewed:?} wrap rayon with no reviewed row in \
         determinism-allowlist.toml (§4.1) — write the exemption with its scope and its reason, \
         or drop the wrapper"
    );
    let missing: Vec<&&str> = exempt.iter().filter(|c| !wrappers.contains(**c)).collect();
    anyhow::ensure!(
        missing.is_empty(),
        "determinism-allowlist.toml exempts {missing:?} for rayon, but deny.toml's rayon wrappers do not include them"
    );
    println!(
        "xtask: rayon allowlist and deny.toml wrappers agree ({} exemption(s))",
        exempt.len()
    );
    Ok(())
}

/// Every §3 grep, pointed at a tree where the thing it bans is present.
///
/// The gates have only ever run against a clean workspace, which proves they do
/// not fire and nothing else — and §6 M5 asks for the opposite evidence ("the
/// grep bans on `static mut` / `thread_local!` in game crates are live and
/// **reject a plant**"). Each test below plants one violation and asserts the
/// scan names the file; the last plants none and asserts silence, so a scan that
/// reported everything would fail too.
#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    /// A throwaway source tree. Named after the test, and nextest gives each
    /// test its own process, so two of these never collide.
    fn plant(test: &str, files: &[(&str, &str)]) -> PathBuf {
        let root = std::env::temp_dir().join(format!("gg-grep-plant-{test}"));
        let _ = std::fs::remove_dir_all(&root);
        for (rel, text) in files {
            let path = root.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, text).unwrap();
        }
        root
    }

    fn violations(root: &Path) -> Vec<String> {
        super::scan(root).unwrap().0
    }

    /// The bans that keep game state from surviving a reload behind the host's
    /// back (§4.2.2). Scoped to `demos/`, which is where game crates live, and
    /// matched on the declaration: `static mut` is one spelling of a dozen, and
    /// the quiet ones below retain exactly as much across a reload.
    #[test]
    fn retained_state_in_a_game_crate_is_rejected() {
        for planted in [
            "static mut COUNT: u32 = 0;",
            "static  mut COUNT: u32 = 0;",
            "pub(crate) static COUNT: AtomicU32 = AtomicU32::new(0);",
            "static REGISTRY: OnceLock<Vec<u8>> = OnceLock::new();",
            "static STATE: Mutex<u32> = Mutex::new(0);",
            "thread_local! { static X: u8 }",
        ] {
            let root = plant("retained-state", &[("demos/03-x/src/lib.rs", planted)]);
            let found = violations(&root);
            assert_eq!(found.len(), 1, "planted `{planted}`, got {found:?}");
            assert!(found[0].contains("(§4.2.2)"), "{found:?}");
        }
        // A comment, a `&'static` bound and a local are none of the above — the
        // ban is on what a `static` *holds*, not on the words.
        let root = plant(
            "retained-state-innocent",
            &[(
                "demos/03-x/src/lib.rs",
                "// static mut here\nfn f(s: &'static str) {}\nlet c = RefCell::new(0);\n",
            )],
        );
        assert!(violations(&root).is_empty(), "{:?}", violations(&root));
        // The same text in an engine crate is legal: the ban is about code that
        // crosses the reload seam, not about the tokens.
        let root = plant(
            "retained-state-engine",
            &[("crates/gg-x/src/lib.rs", "static mut COUNT: u32 = 0;")],
        );
        assert!(violations(&root).is_empty());
    }

    /// The walk reaches the harness and CI's own source (§3). `tools/` answers
    /// to every rule; `xtask` answers to every rule but the two it must spell
    /// literally in order to *be* them (deny.toml names it in `ash`'s wrappers).
    #[test]
    fn the_walk_reaches_tools_and_xtask() {
        let root = plant(
            "walk",
            &[
                ("tools/gg-t/src/a.rs", "fn f() {\n    unsafe { g() }\n}\n"),
                ("tools/gg-t/src/b.rs", "let f = vk::Format::UNDEFINED;\n"),
                ("xtask/src/probe.rs", "let f = vk::Format::UNDEFINED;\n"),
                ("xtask/src/bare.rs", "fn f() {\n    unsafe { g() }\n}\n"),
            ],
        );
        let found = violations(&root);
        for expect in [
            "tools/gg-t/src/a.rs",
            "tools/gg-t/src/b.rs",
            "xtask/src/bare.rs",
        ] {
            assert!(found.iter().any(|v| v.starts_with(expect)), "{found:?}");
        }
        assert_eq!(found.len(), 3, "{found:?}");
    }

    /// The containment seam (§3): `gg-rhi` is the only crate that speaks Vulkan.
    #[test]
    fn a_vulkan_token_outside_gg_rhi_is_rejected() {
        let root = plant(
            "vk-token",
            &[
                (
                    "crates/gg-render/src/pass.rs",
                    "let f = vk::Format::UNDEFINED;",
                ),
                (
                    "crates/gg-rhi/src/device.rs",
                    "let f = vk::Format::UNDEFINED;",
                ),
            ],
        );
        let found = violations(&root);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].starts_with("crates/gg-render/"), "{found:?}");
    }

    /// The whole-segment rule the gate is built on: `gg_ecs::hash::` ends in
    /// `ash::` and is not the crate `ash`.
    #[test]
    fn a_path_ending_in_ash_is_not_the_ash_crate() {
        let root = plant(
            "ash-suffix",
            &[(
                "crates/gg-ecs/src/world.rs",
                "use gg_ecs::hash::ComponentId;",
            )],
        );
        assert!(violations(&root).is_empty());
    }

    /// §2's Sim time row: sim-visible time is a tick count, never a float. The
    /// method below is not a field — that escape hatch used to fire on any line
    /// with `fn ` in it, comments included.
    #[test]
    fn a_float_time_field_is_rejected() {
        let root = plant(
            "float-time",
            &[(
                "crates/gg-x/src/lib.rs",
                "struct S {\n    pub elapsed: f32,\n}\nfn dt(&self) -> f32 { 0.0 }\n",
            )],
        );
        let found = violations(&root);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].contains("float time field `elapsed`"), "{found:?}");

        // Four spellings of the same field, all of which walked through a match
        // that stripped exactly `"pub "` and demanded exactly one space.
        for planted in [
            "pub(crate) dt: f32,",
            "pub(super) elapsed: f64,",
            "dt : f32,",
            "time: Option<f64>,",
        ] {
            let root = plant(
                "float-time-spellings",
                &[("crates/gg-x/src/lib.rs", planted)],
            );
            let found = violations(&root);
            assert_eq!(found.len(), 1, "planted `{planted}`, got {found:?}");
        }
    }

    /// The escapes a six-word list could not close: any clock whose name was not
    /// one of the six, and the newtype spelling, which has no field name at all.
    #[test]
    fn a_float_clock_under_another_name_is_rejected() {
        for planted in [
            "cooldown: f32,",
            "pub timestamp: f64,",
            "age_seconds: f32,",
            "remaining_secs: f32,",
            "pub(crate) accumulator: f64,",
            "spawn_delay: Option<f32>,",
            "struct Timer(f32);",
            "pub struct Cooldown(pub f32);",
            "pub struct SinceSpawn(u32, f64);",
            "struct Uptime<T>(f64, T);",
        ] {
            let root = plant("float-time-widened", &[("demos/07-x/src/lib.rs", planted)]);
            let found = violations(&root);
            assert_eq!(found.len(), 1, "planted `{planted}`, got {found:?}");
            assert!(found[0].contains("Sim time row"), "{found:?}");
        }
    }

    /// The other direction, which is what makes the widening affordable: segments
    /// and not substrings, floats and not the tick counts §2 asks for, fields and
    /// not locals.
    #[test]
    fn a_time_ish_substring_is_not_a_clock() {
        for innocent in [
            "damage: f32,",
            "average: f64,",
            "pub metallic: f32,",
            "cooldown_ticks: u32,",
            "struct Timer(u32);",
            "let ms: f32 = 1.0;",
            "// elapsed: f32,",
        ] {
            let root = plant(
                "float-time-innocent",
                &[("demos/07-x/src/lib.rs", innocent)],
            );
            let found = violations(&root);
            assert!(found.is_empty(), "planted `{innocent}`, got {found:?}");
        }
    }

    /// The row's scope, and the reason it has one (§1.4): a GPU timestamp, a
    /// harness frame number and a Tracy zone are wall clock, and wall clock is
    /// float. Scoping beats exempting the files one at a time.
    #[test]
    fn a_float_clock_in_the_render_half_is_not_the_sim_row() {
        let root = plant(
            "float-time-scope",
            &[
                ("crates/gg-rhi/src/timing.rs", "pub gpu_ms: f32,"),
                ("crates/gg-render/src/lib.rs", "pub elapsed: f32,"),
                ("tools/gg-golden/src/bench.rs", "sum_ms: f64,"),
                ("crates/gg-core/src/loop.rs", "pub elapsed_ms: f32,"),
            ],
        );
        let found = violations(&root);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].starts_with("crates/gg-core/"), "{found:?}");
    }

    /// Every `unsafe` block carries a `// SAFETY:` within eight lines (§4.2).
    /// An `unsafe fn` *declaration* does not: that obligation is the caller's.
    /// The last four plants are what a `contains("SAFETY")` accepted.
    #[test]
    fn an_unsafe_block_without_a_safety_note_is_rejected() {
        let root = plant(
            "unsafe-note",
            &[
                (
                    "crates/gg-a/src/lib.rs",
                    "fn f() {\n    unsafe { g() }\n}\n",
                ),
                (
                    "crates/gg-b/src/lib.rs",
                    "fn f() {\n    // SAFETY: g is trivially fine.\n    unsafe { g() }\n}\n",
                ),
                ("crates/gg-c/src/lib.rs", "pub unsafe fn h() {}\n"),
                (
                    // A qualified note is still a note: the teardown paths write
                    // one per `match`, not one per arm.
                    "crates/gg-d/src/lib.rs",
                    "// SAFETY (all arms): handles are this device's.\nunsafe { d.destroy() };\n",
                ),
                (
                    "crates/gg-e/src/lib.rs",
                    "/// Nothing here is a SAFETY: concern.\nunsafe { g() }\n",
                ),
                (
                    "crates/gg-f/src/lib.rs",
                    "const SAFETY_MARGIN: u32 = 4;\nunsafe { g() }\n",
                ),
                (
                    // One note, one site.
                    "crates/gg-g/src/lib.rs",
                    "// SAFETY: the first one.\nunsafe { g() }\nunsafe { h() }\n",
                ),
            ],
        );
        let found = violations(&root);
        for expect in [
            "crates/gg-a/",
            "crates/gg-e/",
            "crates/gg-f/",
            "crates/gg-g/",
        ] {
            assert!(found.iter().any(|v| v.starts_with(expect)), "{found:?}");
        }
        assert_eq!(found.len(), 4, "{found:?}");
    }

    /// §6 M6: a barrier written by hand is rejected wherever it is — including
    /// inside `gg-rhi`, where the tokens are legal Rust — and the graph's own
    /// execution layer is not.
    #[test]
    fn a_hand_written_barrier_outside_the_graph_is_rejected() {
        let root = plant(
            "barrier",
            &[
                (
                    "crates/gg-rhi/src/frame.rs",
                    "// SAFETY: fine.\nunsafe { d.cmd_pipeline_barrier2(cmd, &info) };\n",
                ),
                (
                    "crates/gg-rhi/src/graph.rs",
                    "// SAFETY: fine.\nunsafe { d.cmd_pipeline_barrier2(cmd, &info) };\n",
                ),
                (
                    "crates/gg-render/src/lib.rs",
                    "let b = ImageMemoryBarrier2::default();\n",
                ),
            ],
        );
        let found = violations(&root);
        assert!(
            found.iter().any(|v| v.contains("frame.rs")),
            "the derivation belongs to the graph alone: {found:?}"
        );
        assert!(
            found.iter().any(|v| v.contains("gg-render")),
            "and above the seam it is not even spellable: {found:?}"
        );
        assert!(
            !found
                .iter()
                .any(|v| v.starts_with("crates/gg-rhi/src/graph.rs")),
            "the graph's own execution layer is where they live: {found:?}"
        );
    }

    /// §5's Stop-hook budget, planted the same way the greps are: a number that
    /// has never been seen to go red is a budget on the same footing as a gate
    /// that has never rejected anything.
    #[test]
    fn the_fast_tier_budget_rejects_a_tier_that_is_slow_every_time() {
        let over = super::FAST_TIER_BUDGET.as_millis() as u64 + 1;
        let slow = vec![over; super::FAST_TIER_WINDOW];
        let err = super::fast_tier_verdict(&slow).unwrap_err().to_string();
        assert!(err.contains("never a bigger number"), "{err}");

        // A cold cache, a toolchain bump, a `Cargo.toml` touch: one slow run
        // among warm ones is a rebuild, not a slow tier, and blocking an agent
        // turn on it is the false failure the minimum exists to avoid.
        let mut one_hitch = slow.clone();
        one_hitch[2] = 1;
        assert!(super::fast_tier_verdict(&one_hitch).is_ok());

        // And a window that has not filled yet may be *all* cold — a fresh clone
        // is the ordinary case — so it reports and does not judge.
        assert!(super::fast_tier_verdict(&slow[..super::FAST_TIER_WINDOW - 1]).is_ok());
        assert!(super::fast_tier_verdict(&[]).is_ok());
    }

    /// The other half of the evidence: a tree with nothing planted is silent.
    #[test]
    fn a_clean_tree_reports_nothing() {
        let root = plant(
            "clean",
            &[(
                "crates/gg-x/src/lib.rs",
                "pub fn add(a: u32, b: u32) -> u32 {\n    a + b\n}\n",
            )],
        );
        assert!(violations(&root).is_empty());
    }
}
