//! The M5 gates that can only be proven by *running the shell over a game
//! dylib* — §5.6c across tiers, replay segments across a reload, and the reload
//! chaos cases (§5.11).
//!
//! Everything here is windowless (§1.5): `GG_HEADLESS=1`, so the shell opens no
//! window and its extract and render stages return immediately. The sim, the
//! reload path and the replay stream are identical either way, which is the
//! whole reason CI can exercise the shell it ships.
//!
//! # Why the curated stream is authored rather than recorded
//!
//! §5.6c says "a replay recorded under the true dist configuration". A headless
//! CI run has no hands: with nothing pressed, demo 03's world is static and its
//! canonical hash never moves, so a recorded-under-dist file would be an empty
//! stream and comparing it across tiers would prove that zero equals zero. The
//! criterion is therefore split rather than weakened, and both halves are real:
//!
//! - **The recorder ships and stamps its tier** — the dist shell records a run
//!   and the file decodes with `tier = "dist"`. That is what "recorded under
//!   dist" is protecting: §1.2's bug-report channel out of a shipped build.
//! - **Optimization cannot touch sim results** — a curated stream (scripted
//!   here, like demo 02's, so its phases are readable rather than a blob) drives
//!   the shell under dist-verify, dev and instrumented, and all three canonical
//!   hash sequences must be identical.
//!
//! The one thing the split gives up is a stream whose *bits* came out of a dist
//! binary. Nothing in the sim can tell where an [`InputFrame`] was authored —
//! that is the property that makes a replay a replay — so what is lost is the
//! appearance of provenance, not a check.
//!
//! # No checked-in hash baseline for demo 03, on purpose
//!
//! Demo 02's gate compares against a blessed sequence. Demo 03's must not: its
//! gameplay code is explicitly throwaway and is the crate agents edit all day
//! (§6 M5), so a per-tick baseline would go red on every gameplay tweak and be
//! re-blessed without being read — a gate that trains you to ignore it. The
//! tiers are compared *against each other* in one invocation instead, which is
//! exactly the claim 6c makes and needs no file.

use std::path::{Path, PathBuf};
use std::process::Command;

use gg_input::{InputFrame, Recorder, Replay, ReplayMeta};

use crate::util::{cargo, plain, run as exec, workspace_root};

/// The curated stream's name in `tests/replays/`.
pub const CURATED: &str = "demo03-curated";

/// Ticks it covers. Long enough for every phase below to leave a mark on the
/// hash, short enough that three tier runs stay a nightly cost.
const CURATED_TICKS: u64 = 300;

/// Demo 03's verbs, in the id order the game declares them (§4.7). Repeated here
/// rather than read from the dylib because the shell already refuses a replay
/// whose verb list disagrees with the game's, *by name* — so a drift is a loud
/// failure at load, not a stream replayed onto the wrong actions.
const ACTIONS: &[&str] = &["fire", "spawn", "restart"];
const AXES: &[&str] = &["move_right", "move_up", "move_forward", "aim_x", "aim_y"];

/// The tiers §5.6c compares. `tier-dist` is absent because dist computes no
/// canonical hash — which is the whole reason `dist-verify` exists (§2).
struct Tier {
    name: &'static str,
    features: &'static str,
    profile: &'static str,
    /// Cargo's output directory for that profile.
    out: &'static str,
}

const HASHED_TIERS: &[Tier] = &[
    Tier {
        name: "dev",
        features: "tier-dev",
        profile: "dev",
        out: "debug",
    },
    Tier {
        name: "instrumented",
        features: "tier-instrumented",
        profile: "instrumented",
        out: "instrumented",
    },
    Tier {
        name: "dist-verify",
        features: "tier-dist-verify",
        profile: "dist",
        out: "dist",
    },
];

// ---- the curated stream ------------------------------------------------

fn meta() -> ReplayMeta {
    let mut meta = ReplayMeta::new(
        gg_math::DETERMINISM_CONTRACT,
        "curated",
        gg_core::DEFAULT_TICK_HZ,
        ACTIONS,
        AXES,
    );
    // Not the running binary's commit: a checked-in replay whose header moved
    // with every commit would produce a diff on every commit (demo 02's rule).
    meta.engine_commit = "generated".to_owned();
    meta
}

fn axes(right: i32, up: i32, forward: i32, aim_x: i32, aim_y: i32) -> [i32; gg_input::MAX_AXES] {
    let mut out = [0; gg_input::MAX_AXES];
    let unit = gg_input::AXIS_SCALE;
    out[0] = right * unit;
    out[1] = up * unit;
    out[2] = forward * unit;
    // Aim is pointer motion: already in AXIS_SCALEths, and deliberately not a
    // whole unit — a mouse delta is small and the fixed point is what makes it
    // reproducible (§4.7).
    out[3] = aim_x;
    out[4] = aim_y;
    out
}

/// The scripted session: walk, look, shoot, spawn, restart, coast.
///
/// Each phase exists to move some state the canonical hash covers, and the
/// phases are in the order a person would actually do them — which is also the
/// order that makes a divergence readable ("it went wrong when we started
/// shooting").
pub fn curated_replay() -> Replay {
    let mut recorder = Recorder::new(meta());
    for tick in 0..CURATED_TICKS {
        let frame = match tick / 50 {
            // Walk forward while turning: the eye's f64 position moves and yaw
            // enters the hash, so libm's sin/cos decide the path.
            0 => InputFrame {
                buttons: 0,
                axes: axes(0, 0, 1, 18, -4),
            },
            // Strafe the other way, pitching back.
            1 => InputFrame {
                buttons: 0,
                axes: axes(-1, 0, 0, -12, 6),
            },
            // Fire on a pulse: shots spawn, which is entity allocation plus an
            // archetype the previous phases never touched.
            2 => InputFrame {
                buttons: u64::from(tick % 7 < 2),
                axes: axes(0, 0, 1, 4, 0),
            },
            // Spawn cubes on a slower pulse, while still moving.
            3 => InputFrame {
                buttons: u64::from(tick % 11 < 2) << 1,
                axes: axes(1, 0, 0, -6, 2),
            },
            // Restart once, then coast: despawn-everything followed by a fresh
            // bootstrap is the largest structural change the game can make, and
            // the freelist state it leaves behind is what §4.8 says a restore
            // has to carry.
            4 => InputFrame {
                buttons: u64::from(tick == 200) << 2,
                axes: axes(0, 0, 0, 0, 0),
            },
            // Shots fading out with nothing pressed — the sequence must keep
            // moving after the player has stopped, or the tail proves nothing.
            _ => InputFrame {
                buttons: 0,
                axes: axes(0, 0, 0, 0, 0),
            },
        };
        recorder.record(tick, frame);
    }
    recorder.finish()
}

/// Where the curated stream lives.
pub fn curated_path() -> PathBuf {
    workspace_root()
        .join("tests/replays")
        .join(format!("{CURATED}.ggrp"))
}

/// Rewrite it. Called by `xtask replay --bless`, beside demo 02's.
pub fn bless(commit: &str) -> anyhow::Result<()> {
    let mut replay = curated_replay();
    replay.set_engine_commit(commit);
    std::fs::write(curated_path(), replay.encode())?;
    println!(
        "xtask replay: blessed {CURATED} ({} ticks, {} change records) at {commit}",
        replay.ticks(),
        replay.change_count()
    );
    Ok(())
}

/// Read it back and say what it claims — the check half.
pub fn describe() -> anyhow::Result<()> {
    let replay = Replay::decode(&std::fs::read(curated_path())?)?;
    println!(
        "xtask replay: {CURATED} — {} ticks, {} change record(s), contract v{}, blessed at {}",
        replay.ticks(),
        replay.change_count(),
        replay.meta().contract,
        replay.meta().engine_commit
    );
    Ok(())
}

// ---- building and running the shell over demo 03 -----------------------

fn exe(dir: &str, name: &str) -> PathBuf {
    workspace_root()
        .join("target")
        .join(dir)
        .join(if cfg!(windows) {
            format!("{name}.exe")
        } else {
            name.to_owned()
        })
}

fn dylib(dir: &str) -> PathBuf {
    workspace_root()
        .join("target")
        .join(dir)
        .join(if cfg!(windows) {
            "demo_03_reload.dll"
        } else {
            "libdemo_03_reload.so"
        })
}

/// Build the shell and the game for one tier, and stage both under
/// `target/tiers/<name>/`.
///
/// Staged rather than run in place because `tier-dist` and `tier-dist-verify`
/// share the `dist` profile and therefore the same output path: building the
/// second would overwrite the first, and a gate that silently compared a tier
/// against itself would be green for the wrong reason.
fn stage(tier: &Tier) -> anyhow::Result<(PathBuf, PathBuf)> {
    exec(
        cargo().args([
            "build",
            "-p",
            "gg-runtime",
            "--profile",
            tier.profile,
            "--no-default-features",
            "--features",
            tier.features,
        ]),
        &format!("build gg-runtime [{}]", tier.name),
    )?;
    exec(
        cargo().args(["build", "-p", "demo-03-reload", "--profile", tier.profile]),
        &format!("build demo 03 [{} profile]", tier.profile),
    )?;

    let dir = workspace_root().join("target/tiers").join(tier.name);
    std::fs::create_dir_all(&dir)?;
    let host = dir.join(exe(tier.out, "gg-runtime").file_name().unwrap_or_default());
    let game = dir.join(dylib(tier.out).file_name().unwrap_or_default());
    std::fs::copy(exe(tier.out, "gg-runtime"), &host)?;
    std::fs::copy(dylib(tier.out), &game)?;
    Ok((host, game))
}

/// Run a staged shell over a staged game and return its whole log.
fn play(host: &Path, game: &Path, args: &[&str], hashes: bool) -> anyhow::Result<String> {
    let mut cmd = Command::new(host);
    cmd.arg("--game").arg(game).args(args);
    cmd.env("GG_HEADLESS", "1");
    // The hash target is off by default (one line per tick); this is the gate
    // that wants it.
    cmd.env(
        "RUST_LOG",
        if hashes {
            "info,gg::hash=debug"
        } else {
            "info"
        },
    );
    let out = cmd
        .output()
        .map_err(|e| anyhow::anyhow!("failed to spawn {}: {e}", host.display()))?;
    let log = format!("{}{}", plain(&out.stdout), plain(&out.stderr));
    anyhow::ensure!(
        out.status.success(),
        "{} exited {}\n{log}",
        host.display(),
        out.status
    );
    Ok(log)
}

/// The `(tick, hash)` sequence a run emitted, in tick order.
fn sequence(log: &str) -> anyhow::Result<Vec<(u64, String)>> {
    let mut out = Vec::new();
    for line in log.lines().filter(|l| l.contains("gg::hash")) {
        out.push((
            crate::util::field_u64(line, "tick")?,
            crate::util::field(line, "hash")?.to_owned(),
        ));
    }
    anyhow::ensure!(!out.is_empty(), "the run emitted no state hashes:\n{log}");
    Ok(out)
}

/// Where two sequences first disagree, named the way §5.6 requires — a bare
/// "hashes differ" on tick 9,000 is a wasted day.
fn divergence(a: &(&str, Vec<(u64, String)>), b: &(&str, Vec<(u64, String)>)) -> Option<String> {
    for (left, right) in a.1.iter().zip(&b.1) {
        if left != right {
            return Some(format!(
                "{} and {} diverge at tick {}: {} vs {}",
                a.0, b.0, left.0, left.1, right.1
            ));
        }
    }
    (a.1.len() != b.1.len()).then(|| {
        format!(
            "{} ran {} ticks and {} ran {}",
            a.0,
            a.1.len(),
            b.0,
            b.1.len()
        )
    })
}

// ---- game variants ------------------------------------------------------

/// Demo 03's source, which every variant below is an edit of.
fn game_source() -> anyhow::Result<String> {
    Ok(std::fs::read_to_string(
        workspace_root().join("demos/03-reload/src/lib.rs"),
    )?)
}

/// The migration edit: one more field on a hashed component.
///
/// `u64` rather than `f32` because `bytemuck::Pod` refuses a struct with
/// padding, and `Cube` is 48 bytes at align 8 — a four-byte field would need a
/// second one beside it and say nothing extra.
fn with_an_extra_field(source: &str) -> anyhow::Result<String> {
    // Two edits, because a field is not just a declaration: the struct literal
    // that builds a `Cube` has to name it too, and that pair is exactly what an
    // agent types when it adds one.
    let edits = [
        (
            "    /// Half-extent, metres.\n    pub scale: f32,\n}",
            "    /// Half-extent, metres.\n    pub scale: f32,\n    /// Added by the reload \
             gates: a field the running world has never seen, so\n    /// adopting this build is \
             a migration rather than a pointer swap (§4.2.2).\n    pub wobble: u64,\n}",
        ),
        (
            "            scale: 0.5,\n",
            "            scale: 0.5,\n            wobble: 0,\n",
        ),
    ];
    let mut out = source.to_owned();
    for (anchor, replacement) in edits {
        anyhow::ensure!(
            out.contains(anchor),
            "demo 03's source no longer contains `{}` — the migration variant is a text edit, so \
             a rename here is a gate to re-point rather than one that quietly stops migrating \
             anything",
            anchor.trim()
        );
        out = out.replace(anchor, replacement);
    }
    Ok(out)
}

fn dylib_name() -> &'static str {
    if cfg!(windows) {
        "demo_03_reload.dll"
    } else {
        "libdemo_03_reload.so"
    }
}

/// Build a variant of demo 03 — its source with one edit applied — as a
/// standalone game dylib the host can be pointed at.
///
/// Generated rather than checked in, and *edited from demo 03's own source*
/// rather than written fresh, because the interesting property is that the two
/// builds differ by exactly the edit under test. A second checked-in game crate
/// would drift away from the first the week after it was added.
///
/// It is its own workspace (the empty `[workspace]` table) so cargo does not try
/// to adopt a package sitting under `target/`, and every variant shares one
/// target directory so the engine crates compile once rather than once per
/// variant. The artifact is copied aside afterwards because they all build a
/// library called `demo_03_reload` — which is the point: the host's path never
/// changes, only the bytes behind it.
fn variant(name: &str, source: &str) -> anyhow::Result<PathBuf> {
    let dir = write_variant(name, source)?;
    build_variant(name)?;
    let kept = dir.join(dylib_name());
    std::fs::copy(shared_target().join("debug").join(dylib_name()), &kept)?;
    Ok(kept)
}

/// One target directory for every variant, so the engine crates compile once
/// rather than once per variant.
fn shared_target() -> PathBuf {
    workspace_root().join("target/variants/_target")
}

/// Lay a variant's source and manifest down without building it.
fn write_variant(name: &str, source: &str) -> anyhow::Result<PathBuf> {
    let root = workspace_root();
    let dir = root.join("target/variants").join(name);
    std::fs::create_dir_all(dir.join("src"))?;
    std::fs::write(dir.join("src/lib.rs"), source)?;
    let crates = root.join("crates").display().to_string().replace('\\', "/");
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            "# Generated by `xtask reload` — a variant of demo 03 (§6 M5 gates).\n\
             # Not checked in, not a workspace member, and rewritten on every run.\n\
             [workspace]\n\n\
             [package]\n\
             name = \"gg-variant-{name}\"\n\
             version = \"0.0.0\"\n\
             edition = \"2024\"\n\n\
             [lib]\n\
             name = \"demo_03_reload\"\n\
             crate-type = [\"cdylib\"]\n\
             path = \"src/lib.rs\"\n\n\
             [dependencies]\n\
             bytemuck = {{ version = \"1\", features = [\"derive\"] }}\n\
             gg-ecs = {{ path = \"{crates}/gg-ecs\" }}\n\
             gg-math = {{ path = \"{crates}/gg-math\" }}\n\n\
             # The workspace's own dev profile, copied: the latency instrument \
             measures a\n# rebuild, and a rebuild at a different optimization \
             level is a different\n# number (§3 profiles).\n\
             [profile.dev]\n\
             opt-level = 1\n\
             [profile.dev.package.\"*\"]\n\
             opt-level = 3\n"
        ),
    )?;
    Ok(dir)
}

/// Build a variant already written, and return the artifact cargo produced.
fn build_variant(name: &str) -> anyhow::Result<PathBuf> {
    let manifest = workspace_root()
        .join("target/variants")
        .join(name)
        .join("Cargo.toml");
    exec(
        cargo().env("CARGO_TARGET_DIR", shared_target()).args([
            "build",
            "--manifest-path",
            &manifest.display().to_string(),
        ]),
        &format!("build game variant `{name}`"),
    )?;
    Ok(shared_target().join("debug").join(dylib_name()))
}

// ---- the gates ----------------------------------------------------------

/// `cargo xtask reload` — every M5 gate that needs the shell driving a game
/// dylib, in one command. The nightly tier calls the same functions
/// individually; this exists so a human can run the set after touching the
/// boundary without waiting out a whole nightly.
pub fn gates(args: &[&str]) -> anyhow::Result<()> {
    let only = |name: &str| args.is_empty() || args.contains(&name);
    if only("--cross-tier") {
        cross_tier()?;
        dist_records()?;
    }
    if only("--segments") {
        segments()?;
    }
    if only("--chaos") {
        chaos_reload()?;
    }
    if only("--latency") {
        latency()?;
    }
    println!("xtask reload: green");
    Ok(())
}

/// The two points the latency instrument measures, as generated-system counts.
///
/// Each generated system is a real query over a real component, because compile
/// time is bought by *typed* code — a thousand `fn f() {}` would measure the
/// parser and nothing that grows with a game.
///
/// Two points rather than one because they answer different questions. The
/// **small** one is roughly twice demo 03 and is gated against M5's budget: it
/// is the size at which the cooperative loop is supposed to feel immediate, and
/// a regression there is a regression in the thesis. The **fat** one is four
/// times larger again and is gated only against a loose ceiling: its job is to
/// keep the *curve* visible, and asserting a 2 s budget at a size that has
/// already been measured past it would be a gate that is red by design.
const SMALL_SYSTEMS: usize = 25;
const FAT_SYSTEMS: usize = 100;

/// M5's exit budget: save → new behaviour.
const BUDGET_MS: u128 = 2_000;

/// What the fat point may cost before something has gone wrong with the
/// toolchain rather than with the project. Measured at ~3.2 s (2026-08-01).
const FAT_CEILING_MS: u128 = 10_000;

/// Demo 03 with a game's worth of extra code bolted on (§6 M5 finding (e)).
///
/// The cost that grows with a project is the **game crate's** rebuild-and-link —
/// the engine is a compiled dependency and does not recompile — so the
/// instrument grows the one thing that matters and leaves everything else alone.
fn fat_source(source: &str, systems: usize) -> anyhow::Result<String> {
    let mut out = String::with_capacity(source.len() + systems * 400);
    out.push_str(source);
    out.push_str("\n// ---- generated by `xtask reload` (§6 M5 latency instrument) ----\n");
    for i in 0..systems {
        // Behind a condition that never holds: what this instrument buys is
        // *compile and link* time, and three hundred systems each sweeping the
        // world every tick would turn the measured run into a benchmark of the
        // instrument. The code is compiled in full either way — the condition is
        // a runtime value, so nothing here is folded away.
        out.push_str(&format!(
            "\n/// Generated system {i}.\npub fn generated_{i}(world: &mut GameWorld) {{\n    \
             if world.tick() != u64::MAX {{\n        return;\n    }}\n    let mut acc = \
             {i}u64;\n    let _ = world.each::<(&Cube,)>(|_, (cube,)| {{\n        acc = \
             acc.wrapping_add(cube.order ^ {i});\n        acc = acc.rotate_left({});\n    }});\n \
             let _ = world.each::<(&Shot,)>(|_, (shot,)| {{\n        acc = \
             acc.wrapping_mul(3).wrapping_add(u64::from(shot.ticks_left));\n    }});\n    if acc \
             == u64::MAX {{\n        world.log(log_level::TRACE, \"{i}\");\n    }}\n}}\n",
            i % 63
        ));
    }
    out.push_str("\n/// Runs every generated system, so none of them is dead code.\npub fn generated(world: &mut GameWorld) {\n");
    for i in 0..systems {
        out.push_str(&format!("    generated_{i}(world);\n"));
    }
    out.push_str("}\n");

    let anchor = "    systems: [restart, bootstrap, aim, walk, shoot, spawn, fade, present],";
    anyhow::ensure!(
        out.contains(anchor),
        "demo 03's systems list is not where this gate expected it"
    );
    Ok(out.replace(
        anchor,
        "    systems: [restart, bootstrap, aim, walk, shoot, spawn, fade, present, generated],",
    ))
}

/// §6 M5 finding (e): **reload latency, measured at a scale that is not a toy.**
///
/// Every latency number in this project so far came from a game crate of a few
/// hundred lines, and the thing that grows is not the one people assume. World
/// size grows the *snapshot*, which is a column memcpy and is not even paid on
/// the common reload — an edit that moves no schema is a pointer swap with no
/// snapshot at all. Game-crate code size grows the *rebuild and link*, which was
/// 80–90% of every measurement, and it is the only part with a real curve.
///
/// So the instrument is a synthetic fat game crate, edited one line at a time,
/// with the whole loop timed: the rebuild a save triggers, plus the swap the
/// host performs. The budget it is measured against is M5's < 2 s, and the two
/// halves are reported separately because they degrade for different reasons and
/// have different fixes (a faster linker versus a cheaper migration).
fn latency() -> anyhow::Result<()> {
    let small = measure_loop("small", SMALL_SYSTEMS, BUDGET_MS)?;
    anyhow::ensure!(
        small <= BUDGET_MS,
        "at roughly twice demo 03's size the loop takes {small} ms against M5's {BUDGET_MS} ms          budget. The fixes are compile-side and none of them is an architecture change (§6 M5):          split the game across several dylibs behind the same table, a dev-profile codegen          backend, a faster linker, or keep the fast-iteration systems in a small crate."
    );
    let fat = measure_loop("fat", FAT_SYSTEMS, FAT_CEILING_MS)?;
    anyhow::ensure!(
        fat <= FAT_CEILING_MS,
        "the fat point cost {fat} ms against a {FAT_CEILING_MS} ms ceiling — that is a toolchain          or machine event, not a project one"
    );
    Ok(())
}

/// Time one save → new-behaviour loop at a given game-crate size, in
/// milliseconds, and print the breakdown.
/// `budget` is the ceiling *this* point is judged against, passed in rather than
/// read from a constant: the fat point answers to `FAT_CEILING_MS`, and printing
/// M5's 2 s beside a number that is not measured against it reads as a gate that
/// forgives its own failure.
fn measure_loop(name: &str, systems: usize, budget: u128) -> anyhow::Result<u128> {
    let source = fat_source(&game_source()?, systems)?;
    let lines = source.lines().count();
    let dir = workspace_root().join("target/latency").join(name);
    std::fs::create_dir_all(&dir)?;
    let variant_name = format!("latency-{name}");

    // Build it once, cold, and keep the artifact. This build is *not* the
    // measurement: a game crate compiled for the first time is not what a save
    // costs, and timing it would flatter or damn the loop by whichever the
    // machine's cache happened to hold.
    write_variant(&variant_name, &source)?;
    let before = dir.join("before.bin");
    std::fs::copy(build_variant(&variant_name)?, &before)?;

    // The edit: one line inside one system body, which is what M5's exit
    // criterion actually says and the cheapest thing a person types.
    let edited = source.replace(
        "pub const MOVE_PER_TICK: f64 = 0.08;",
        "pub const MOVE_PER_TICK: f64 = 0.09;",
    );
    anyhow::ensure!(edited != source, "the one-line edit changed nothing");
    write_variant(&variant_name, &edited)?;

    // The rebuild a save triggers, timed on its own. Deliberately *not* measured
    // with the host running underneath: coupling the two would make the run's
    // frame budget a function of how long the compiler took, and a gate that
    // goes red because a build was slow in the wrong way is a gate nobody trusts
    // (§5's flake budget).
    let started = std::time::Instant::now();
    let after = build_variant(&variant_name)?;
    let rebuild_ms = started.elapsed().as_millis();

    // The swap, measured the way the host measures it: from the file event to
    // the tick boundary the new code first runs at.
    exec(
        cargo().args(["build", "-p", "gg-runtime"]),
        "build the shell [dev]",
    )?;
    let game = dir.join(dylib_name());
    std::fs::copy(&before, &game)?;
    let log = reload_midway(&game, &after, &["--frames", "60000"])?;
    let reloaded = log
        .lines()
        .find(|l| l.contains("game reloaded"))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "the rebuild produced no reload:
{log}"
            )
        })?;
    let swap_ms = u128::from(crate::util::field_u64(reloaded, "save_to_swap_ms")?);
    let total = rebuild_ms + swap_ms;
    println!(
        "xtask: reload latency, {name} ({lines} game-crate lines, {systems} generated systems): \
         rebuild {rebuild_ms} ms + swap {swap_ms} ms = {total} ms (budget {budget} ms)"
    );
    Ok(total)
}

/// A seeded chaos stream for demo 03 (§5.11), long enough to still be running
/// when a rewrite lands.
///
/// xorshift64\*, ours on purpose, for the reason demo 02's generator gives: a
/// crate's random number generator is not a stability guarantee, and a seed has
/// to mean the same stream in five years or a checked-in failure is not a
/// regression test.
fn chaos_stream(seed: u64, ticks: u64) -> Replay {
    let mut recorder = Recorder::new(meta());
    let mut state = seed | 1;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    for tick in 0..ticks {
        let r = next();
        // Held for a stretch rather than re-rolled every tick: input that
        // changes every frame is not what a player does, and a stream of pure
        // noise compresses to one record per tick, which makes the file the
        // slowest part of the gate.
        if tick % 16 == 0 {
            let frame = InputFrame {
                buttons: r & 0b111,
                axes: axes(
                    (r >> 8) as i32 % 2 - 1,
                    0,
                    (r >> 16) as i32 % 2,
                    (r >> 24) as i32 % 33 - 16,
                    (r >> 32) as i32 % 17 - 8,
                ),
            };
            recorder.record(tick, frame);
        }
    }
    recorder.finish()
}

/// §5 gate 11, from M5: **the chaos generator exercises the reload path.**
///
/// Two cases, and they are the two halves of what a reload is allowed to do:
///
/// - **Reloading an identical dylib is hash-neutral.** The same seeded stream
///   run with and without a same-bytes rewrite mid-flight must produce the same
///   canonical hash sequence, tick for tick. This is the check that would catch
///   a swap leaking allocator state, dropping a tick, or rebuilding a column in
///   a different order — none of which a "did it not crash" test can see.
/// - **A schema-migration reload produces the documented result.** The swap onto
///   a build with one more field on `Cube` reports it by name, and the sequence
///   diverges *at the swap tick and not before* — a migration that started
///   changing state a tick early would be a silent corruption of everything
///   recorded before it.
///
/// Under a chaos stream rather than the curated one because the world has to be
/// busy when the swap lands: a reload into a quiet world is the easy case, and
/// the interesting failures are the ones involving live archetype churn.
fn chaos_reload() -> anyhow::Result<()> {
    const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
    // Set by wall time, not by the check: a headless tick is microseconds and
    // the watcher's debounce alone is 120 ms.
    const TICKS: u64 = 60_000;

    let source = game_source()?;
    let before = variant("baseline", &source)?;
    let after = variant("migrated", &with_an_extra_field(&source)?)?;

    let dir = workspace_root().join("target/chaos-reload");
    std::fs::create_dir_all(&dir)?;
    let stream = dir.join("chaos.ggrp");
    std::fs::write(&stream, chaos_stream(SEED, TICKS).encode())?;
    let stream = stream.display().to_string();
    let game = dir.join(dylib_name());

    exec(
        cargo().args(["build", "-p", "gg-runtime"]),
        "build the shell [dev]",
    )?;
    let host = exe("debug", "gg-runtime");

    std::fs::copy(&before, &game)?;
    let quiet = sequence(&play(&host, &game, &["--replay", &stream], true)?)?;
    anyhow::ensure!(
        quiet.iter().any(|(_, h)| h != &quiet[0].1),
        "the chaos stream never moved the world's hash — it would make both cases below vacuous"
    );

    // Case one: the same bytes, swapped in mid-flight.
    std::fs::copy(&before, &game)?;
    let log = reload_midway(&game, &before, &["--replay", &stream])?;
    let neutral = sequence(&log)?;
    if let Some(found) = divergence(
        &("no reload", quiet.clone()),
        &("identical reload", neutral),
    ) {
        anyhow::bail!("§5.11: reloading an identical dylib was not hash-neutral — {found}");
    }
    let swap = crate::util::field_u64(
        log.lines()
            .find(|l| l.contains("game reloaded"))
            .unwrap_or_default(),
        "tick",
    )?;
    println!("xtask: chaos reload — an identical dylib swapped in at tick {swap} changed nothing");

    // Case two: the migration.
    std::fs::copy(&before, &game)?;
    let log = reload_midway(&game, &after, &["--replay", &stream])?;
    let migrated = sequence(&log)?;
    let swap = crate::util::field_u64(
        log.lines()
            .find(|l| l.contains("game reloaded"))
            .unwrap_or_default(),
        "tick",
    )?;
    let report = log
        .lines()
        .find(|l| l.contains("migrated") && l.contains("demo03.cube"))
        .ok_or_else(|| {
            anyhow::anyhow!("the swap reported no migration of `demo03.cube`:\n{log}")
        })?;
    for expected in ["copied", "position", "defaulted", "wobble"] {
        anyhow::ensure!(
            report.contains(expected),
            "the migration report does not name `{expected}`: {report}"
        );
    }
    // Every component that did *not* move stays quiet — §4.2.2's one line per
    // component that actually changed, which is the difference between a report
    // and a wall of text.
    anyhow::ensure!(
        log.matches("migrated").count() == 1,
        "the swap reported {} migrations; only `demo03.cube` moved",
        log.matches("migrated").count()
    );

    let at = |seq: &[(u64, String)], tick: u64| seq.iter().find(|(t, _)| *t == tick).cloned();
    for tick in 0..swap {
        anyhow::ensure!(
            at(&migrated, tick) == at(&quiet, tick),
            "the migration changed tick {tick}, which ran before the swap at {swap}"
        );
    }
    anyhow::ensure!(
        at(&migrated, swap) != at(&quiet, swap),
        "tick {swap} is unchanged by a component gaining a field, so this gate is comparing a \
         sequence to itself"
    );
    println!(
        "xtask: chaos reload — a migrating dylib swapped in at tick {swap}: reported by name, \
         and the sequence moved there and not one tick earlier"
    );
    Ok(())
}

/// A dev shell playing `game`, with the artifact rewritten underneath it partway
/// through. Returns the whole log.
///
/// The rewrite is the reload trigger, and 400 ms is chosen the way the
/// rejuvenation gate chooses it: a busier machine makes the window wider, never
/// narrower, which is the direction a timing assumption should fail in.
fn reload_midway(game: &Path, replacement: &Path, args: &[&str]) -> anyhow::Result<String> {
    use std::process::Stdio;

    exec(
        cargo().args(["build", "-p", "gg-runtime"]),
        "build the shell [dev]",
    )?;
    let mut cmd = Command::new(exe("debug", "gg-runtime"));
    cmd.arg("--game")
        .arg(game)
        .args(args)
        .env("GG_HEADLESS", "1")
        .env("RUST_LOG", "info,gg::hash=debug")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let child = cmd.spawn()?;
    std::thread::sleep(std::time::Duration::from_millis(400));
    std::fs::copy(replacement, game)?;
    let out = child.wait_with_output()?;
    let log = format!("{}{}", plain(&out.stdout), plain(&out.stderr));
    anyhow::ensure!(
        out.status.success(),
        "the shell exited {}\n{log}",
        out.status
    );
    anyhow::ensure!(
        log.contains("game reloaded"),
        "the rewrite did not produce a reload:\n{log}"
    );
    Ok(log)
}

/// §6 M5: **replay segments close and open across a reload, and the pre-reload
/// segment replays deterministically.**
///
/// The reload is a *migration* — the replacement build has one more field on a
/// hashed component — which is what makes the second half a real comparison
/// rather than a tautology. A headless run has no hands, so demo 03's world only
/// moves when its code does; a same-bytes reload would leave the hash sequence
/// flat on both sides of the swap and "the pre-reload segment reproduces" would
/// be true of any two runs whatsoever.
///
/// So the gate asserts both directions:
///
/// - replaying the recorded stream under the **old** build reproduces the
///   pre-reload ticks exactly, and
/// - it does **not** reproduce the post-reload ticks, because those were
///   produced by code this replay is not running.
///
/// The second is what proves the first is load-bearing. A segment that named the
/// wrong build would be invisible to the first check alone.
fn segments() -> anyhow::Result<()> {
    let source = game_source()?;
    let before = variant("baseline", &source)?;
    let after = variant("migrated", &with_an_extra_field(&source)?)?;

    // Its own directory: the watcher watches a *directory*, so rewriting an
    // artifact under `target/debug` would be a reload event for anything else
    // pointed there.
    let dir = workspace_root().join("target/segments");
    std::fs::create_dir_all(&dir)?;
    let game = dir.join(dylib_name());
    std::fs::copy(&before, &game)?;
    let recorded = dir.join("across-a-reload.ggrp");

    let log = reload_midway(
        &game,
        &after,
        // Long enough to still be running when the rewrite lands: a headless
        // tick is microseconds and the watcher's debounce alone is 120 ms, so
        // the frame count is set by *wall time* rather than by how many ticks
        // the comparison below needs. Sixty thousand is about three
        // seconds here, and a slower machine only makes the window wider.
        &[
            "--frames",
            "60000",
            "--record",
            &recorded.display().to_string(),
        ],
    )?;
    let swap = crate::util::field_u64(
        log.lines()
            .find(|l| l.contains("game reloaded"))
            .unwrap_or_default(),
        "tick",
    )?;
    // The migration itself, reported by name — the same line a human sees when
    // an agent adds a field mid-play (§6 M5).
    anyhow::ensure!(
        log.contains("migrated") && log.contains("demo03.cube"),
        "the swap reported no migration of `demo03.cube`:\n{log}"
    );

    let replay = Replay::decode(&std::fs::read(&recorded)?)?;
    let seg = replay.segments();
    anyhow::ensure!(
        seg.len() == 2,
        "recorded {} segment(s) across one reload, expected 2: {seg:?}",
        seg.len()
    );
    anyhow::ensure!(
        seg[0].first_tick == 0,
        "segment 0 starts at {}",
        seg[0].first_tick
    );
    anyhow::ensure!(
        seg[1].first_tick == swap,
        "the segment opened at tick {} and the swap happened at {swap}",
        seg[1].first_tick
    );
    anyhow::ensure!(
        seg[0].code_hash != seg[1].code_hash && seg[0].code_hash != 0,
        "both segments name the same build, so a stream recorded across a reload cannot say \
         which code produced which ticks: {seg:?}"
    );

    // Replay the whole recording under the *old* build.
    let recorded_hashes = sequence(&log)?;
    let again = play(
        &exe("debug", "gg-runtime"),
        &before,
        &["--replay", &recorded.display().to_string()],
        true,
    )?;
    let replayed = sequence(&again)?;
    let at = |seq: &[(u64, String)], tick: u64| {
        seq.iter().find(|(t, _)| *t == tick).map(|(_, h)| h.clone())
    };
    for tick in 0..swap {
        anyhow::ensure!(
            at(&recorded_hashes, tick) == at(&replayed, tick),
            "the pre-reload segment did not reproduce: tick {tick} was {:?} when recorded and \
             {:?} when replayed under the build segment 0 names",
            at(&recorded_hashes, tick),
            at(&replayed, tick)
        );
    }
    anyhow::ensure!(
        at(&recorded_hashes, swap) != at(&replayed, swap),
        "tick {swap} reproduced under the *old* build, so the migration changed nothing the hash \
         can see and this gate is comparing a constant to itself"
    );
    println!(
        "xtask: replay segments — segment 0 (build {:032x}) covers ticks 0..{swap} and replays \
         exactly; segment 1 (build {:032x}) opens at the swap and does not",
        seg[0].code_hash, seg[1].code_hash
    );
    Ok(())
}

/// §5.6c in full, and the M5 exit criterion that carries it: the same curated
/// stream, replayed by the shell under three codegen configurations, must
/// produce one canonical hash sequence.
///
/// dist-verify is what makes this mean anything (§2): dist proper compiles the
/// hash out, so without it "replayed under dev, hashes identical" would have
/// nothing on the dist side to be identical *to*. Instrumented joins because
/// thin-LTO-at-full-optimization is the one codegen no hash ever measured, and
/// it is the tier a bug replay gets profiled under.
pub fn cross_tier() -> anyhow::Result<()> {
    let replay = curated_path();
    anyhow::ensure!(
        replay.is_file(),
        "no curated stream at {} — `cargo xtask replay --bless` authors it",
        replay.display()
    );
    let replay = replay.display().to_string();

    let mut runs: Vec<(&str, Vec<(u64, String)>)> = Vec::new();
    for tier in HASHED_TIERS {
        let (host, game) = stage(tier)?;
        let log = play(&host, &game, &["--replay", &replay], true)?;
        let seq = sequence(&log)?;
        println!(
            "xtask: {} replayed {} ticks, ending {}",
            tier.name,
            seq.len(),
            seq.last().map(|(_, h)| h.as_str()).unwrap_or("?")
        );
        runs.push((tier.name, seq));
    }

    // A constant sequence would satisfy "all identical" while proving nothing,
    // which is exactly the trap an input-free recorded stream would have walked
    // into. The curated stream is supposed to move the world.
    let first = &runs[0].1;
    anyhow::ensure!(
        first.iter().any(|(_, h)| h != &first[0].1),
        "the curated stream never changed the world's hash — it proves nothing about determinism"
    );

    for pair in runs.windows(2) {
        if let Some(found) = divergence(&pair[0], &pair[1]) {
            anyhow::bail!("§5.6c: {found}");
        }
    }
    println!(
        "xtask: §5.6c green — dev, instrumented and dist-verify agree tick for tick over {} ticks",
        first.len()
    );
    Ok(())
}

/// The other half of §5.6c's "recorded under dist", and §5.8's recorder-presence
/// check made behavioural: the *shipping* configuration records a replay, and
/// the file it writes decodes and names the tier that wrote it.
///
/// This is §1.2's bug-report channel proven end to end rather than by looking
/// for the format's magic in the binary's bytes.
pub fn dist_records() -> anyhow::Result<()> {
    let tier = Tier {
        name: "dist",
        features: "tier-dist",
        profile: "dist",
        out: "dist",
    };
    let (host, game) = stage(&tier)?;
    let out = workspace_root().join("target/tiers/dist/recorded.ggrp");
    let log = play(
        &host,
        &game,
        &["--frames", "60", "--record", &out.display().to_string()],
        false,
    )?;

    // §2's other dist claim, and the one only a run can make: the dylib arrives
    // once. In dev the same line appears again at every swap, so counting it is
    // how "the reload machinery is absent, not merely idle" reads from outside
    // the process — the graph check upstream says `notify` is not linked, and
    // this says nothing behaves as though it were.
    let loads = log.matches("game loaded").count();
    anyhow::ensure!(
        loads == 1,
        "the dist shell loaded the game {loads} times — dist is load-once (§2)"
    );
    // The dev-only path announces itself; dist must never reach it.
    anyhow::ensure!(
        !log.contains("game reloaded"),
        "the dist shell reloaded a dylib:\n{log}"
    );
    let recorded = Replay::decode(&std::fs::read(&out)?)?;
    anyhow::ensure!(
        recorded.meta().tier == "dist",
        "a replay recorded by the dist shell says tier `{}`",
        recorded.meta().tier
    );
    anyhow::ensure!(
        recorded.ticks() == 60,
        "the dist shell recorded {} ticks of 60",
        recorded.ticks()
    );
    recorded.check_verbs(ACTIONS, AXES)?;
    println!(
        "xtask: the dist shell loaded the game once and recorded a readable replay ({} ticks, \
         tier `{}`, verbs intact) — §1.2's bug-report channel out of a shipped build",
        recorded.ticks(),
        recorded.meta().tier
    );
    Ok(())
}
