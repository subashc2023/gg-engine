//! `cargo xtask ci` — the four tiers of §3/§5. `--fast` is the Stop hook's tier
//! and lives under a <30 s warm budget: it must never be the reason agent turns
//! balloon. Every tier is headless (§1.5) via util::cargo() — and *windowless
//! by construction*: no automated tier creates an OS window at all (presenting
//! maps a Wayland surface, minimize/restore maps X11/Win32 ones, and WSLg
//! mirrors any mapped window onto the real desktop — "invisible" is not a
//! property CI may rely on). Everything windowed lives in the manual suite,
//! `cargo xtask interactive`, run by a human who expects windows.

use crate::util::{cargo, run as exec, run_capture, walk_rs, workspace_root};
use std::collections::BTreeSet;

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
    gpu_tests()?;
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

    let mut violations = Vec::new();
    for file in &files {
        let rel = file.strip_prefix(root).unwrap_or(file);
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        let text = std::fs::read_to_string(file)?;

        // API containment (§3): vk::/ash:: tokens live in gg-rhi alone.
        if !rel_str.starts_with("crates/gg-rhi/")
            && let Some(tok) = ["ash::", "vk::"]
                .into_iter()
                .find(|t| contains_path(&text, t))
        {
            violations.push(format!("{rel_str}: `{tok}` token outside gg-rhi (§3)"));
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

        // Every `unsafe` block and `unsafe impl` carries a `// SAFETY:` within
        // the preceding 8 lines (§4.2 M3 exit, and the standing rule for
        // gg-rhi). `unsafe fn` *declarations* are exempt: their obligation is on
        // the caller and belongs in the doc comment, not in a SAFETY note.
        let lines: Vec<&str> = text.lines().collect();
        for (lineno, line) in lines.iter().enumerate() {
            let t = line.trim_start();
            if t.starts_with("//") {
                continue;
            }
            let is_site = t.contains("unsafe impl")
                || line
                    .split("unsafe")
                    .skip(1)
                    .any(|rest| rest.trim_start().starts_with('{'));
            if is_site
                && !lines[lineno.saturating_sub(8)..lineno]
                    .iter()
                    .any(|l| l.contains("SAFETY"))
            {
                violations.push(format!(
                    "{rel_str}:{}: unsafe without a `// SAFETY:` note (§4.2)",
                    lineno + 1
                ));
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

        // Hand-written barriers (§4.5, §6 M6): every `synchronization2` barrier
        // in the engine is *derived* by the render graph, so the tokens that
        // spell one live in the graph's execution layer and in the staging
        // ring's queue-family ownership transfers — which are a property of a
        // transfer paired across two queues, not a pass dependency the
        // single-queue v1 graph models.
        if !BARRIER_SITES.contains(&rel_str.as_str())
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

    let missing: Vec<&&str> = exempt.iter().filter(|c| !wrappers.contains(**c)).collect();
    anyhow::ensure!(
        missing.is_empty(),
        "determinism-allowlist.toml exempts {missing:?} for rayon, but deny.toml's rayon wrappers do not include them"
    );
    println!("xtask: rayon allowlist and deny.toml wrappers agree");
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

    /// The two bans that keep game state from surviving a reload behind the
    /// host's back (§4.2.2). Scoped to `demos/`, which is where game crates live.
    #[test]
    fn retained_state_in_a_game_crate_is_rejected() {
        for planted in [
            "static mut COUNT: u32 = 0;",
            "thread_local! { static X: u8 }",
        ] {
            let root = plant("retained-state", &[("demos/03-x/src/lib.rs", planted)]);
            let found = violations(&root);
            assert_eq!(found.len(), 1, "planted `{planted}`, got {found:?}");
            assert!(found[0].contains("static mut / thread_local!"), "{found:?}");
        }
        // The same text in an engine crate is legal: the ban is about code that
        // crosses the reload seam, not about the tokens.
        let root = plant(
            "retained-state-engine",
            &[("crates/gg-x/src/lib.rs", "static mut COUNT: u32 = 0;")],
        );
        assert!(violations(&root).is_empty());
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

    /// §2's Sim time row: sim-visible time is a tick count, never a float.
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
    }

    /// Every `unsafe` block carries a `// SAFETY:` within eight lines (§4.2).
    /// An `unsafe fn` *declaration* does not: that obligation is the caller's.
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
            ],
        );
        let found = violations(&root);
        assert_eq!(found.len(), 1, "{found:?}");
        assert!(found[0].starts_with("crates/gg-a/"), "{found:?}");
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
