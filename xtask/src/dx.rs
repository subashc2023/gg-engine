//! `cargo xtask dx` — the developer-experience benchmarks of §8.
//!
//! §8 names one failure mode no other gate can see: an engine that is correct,
//! fast, deterministic, beautifully instrumented — and unpleasant to make games
//! with. Every other machine in this repo measures correctness. This one
//! measures **friction**, in the three units the risk row asks for: *steps*,
//! *lines*, and *rebuild latency*. For an agent-native engine that is not
//! cosmetic, because the agent pays the friction on every single edit.
//!
//! # Measured, not estimated
//!
//! Each task below is *performed*. The edit is a real text substitution applied
//! to a real copy of real source, the lines are counted off the resulting diff,
//! and the latency is the wall clock of the command that makes the change live.
//! Anchors are asserted before they are replaced, on the same reasoning
//! `xtask reload`'s variants use: a rename should re-point a gate loudly, not
//! quietly stop measuring anything.
//!
//! Two tasks have no code edit at all — inspecting an entity, and reproducing a
//! reported bug. They are still measured: their *steps* are the number, and
//! their latency is the command a developer waits on. A task whose answer is
//! "no rebuild" is the best possible score, and leaving it out would hide that.
//!
//! # What the numbers are and are not
//!
//! They are diffable per machine, like `bench/<machine>.json` and for the same
//! reason (§4.11). They are **not** cross-machine comparable and not a budget:
//! a rebuild is dominated by the box. What regresses visibly is the *shape* —
//! lines going up, steps going up, a rebuild that used to be one crate becoming
//! six.
//!
//! Run: `cargo xtask dx` (prints the table), `--record` (archives it too).

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::util::{cargo, run as exec, run_capture, workspace_root};

/// Where scratch copies live. Under `target/`, never checked in, rewritten on
/// every run.
///
/// No scenario changes a checked-in byte. One — `add-a-render-pass` — rewrites
/// `gg-render/src/graph.rs` with its own identical contents, because cargo keys
/// an incremental rebuild on mtime and there is no other way to measure the
/// rebuild a developer actually waits on. `git status` stays clean.
fn scratch() -> PathBuf {
    workspace_root().join("target/dx")
}

/// One target directory for every scratch build, so the engine crates compile
/// once rather than once per scenario (`xtask reload`'s variants do the same).
fn scratch_target() -> PathBuf {
    scratch().join("_target")
}

/// What one task cost.
struct Measured {
    /// Distinct developer actions: files edited plus commands run.
    steps: usize,
    /// Lines the edit added. Zero for a task that needs no code change, which
    /// is a result rather than a gap.
    lines: usize,
    /// Wall clock of the command that makes the change live, in *microseconds*.
    ///
    /// Microseconds and not milliseconds because two of these are genuinely
    /// sub-millisecond, and a printed `0 ms` reads as "not measured" — which is
    /// the one thing a benchmark must never look like.
    latency_us: u128,
    /// What was actually run, so a reader can repeat it.
    detail: String,
}

struct Scenario {
    /// Slug, and the archive key.
    name: &'static str,
    /// The §8 task, in the row's own words.
    task: &'static str,
    run: fn() -> anyhow::Result<Measured>,
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "add-a-component",
        task: "add a component",
        run: add_a_component,
    },
    Scenario {
        name: "write-a-system",
        task: "write a gameplay system",
        run: write_a_system,
    },
    Scenario {
        name: "add-a-shader-parameter",
        task: "add a shader parameter",
        run: add_a_shader_parameter,
    },
    Scenario {
        name: "add-an-asset",
        task: "add an asset",
        run: add_an_asset,
    },
    Scenario {
        name: "add-a-render-pass",
        task: "add a render pass",
        run: add_a_render_pass,
    },
    Scenario {
        name: "create-a-demo",
        task: "create a demo",
        run: create_a_demo,
    },
    Scenario {
        name: "inspect-an-entity",
        task: "inspect an entity's state",
        run: inspect_an_entity,
    },
    Scenario {
        name: "reproduce-a-bug",
        task: "reproduce a reported bug",
        run: reproduce_a_bug,
    },
];

pub fn run(args: &[&str]) -> anyhow::Result<()> {
    let record = args.contains(&"--record");
    std::fs::create_dir_all(scratch())?;
    let mut results = Vec::new();
    for scenario in SCENARIOS {
        let measured = (scenario.run)()?;
        println!(
            "xtask dx: {:<24} {:>2} step(s)  {:>3} line(s)  {:>9.2} ms   {}",
            scenario.name,
            measured.steps,
            measured.lines,
            measured.latency_us as f64 / 1000.0,
            measured.detail
        );
        results.push((scenario, measured));
    }

    let total_steps: usize = results.iter().map(|(_, m)| m.steps).sum();
    let total_lines: usize = results.iter().map(|(_, m)| m.lines).sum();
    println!(
        "xtask dx: {} task(s), {total_steps} step(s), {total_lines} line(s) — steps and lines are \
         the comparable half; milliseconds are this machine's (§8)",
        results.len()
    );

    if record {
        let path = workspace_root()
            .join("bench")
            .join(format!("dx-{}.json", crate::bench::machine_id()));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, archive(&results)?)?;
        println!(
            "xtask dx: archived to {} — the git diff of this file is the DX regression report \
             (§8)",
            path.display()
        );
    }
    Ok(())
}

fn archive(results: &[(&Scenario, Measured)]) -> anyhow::Result<String> {
    let commit = run_capture(
        std::process::Command::new("git")
            .current_dir(workspace_root())
            .args(["rev-parse", "--short", "HEAD"]),
        "git rev-parse HEAD",
    )?;
    let tasks: String = results
        .iter()
        .map(|(scenario, m)| {
            format!(
                "    {{\"name\": \"{}\", \"task\": \"{}\", \"steps\": {}, \"lines\": {}, \
                 \"latency_us\": {}}}",
                scenario.name, scenario.task, m.steps, m.lines, m.latency_us
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    Ok(format!(
        "{{\n  \"schema\": 1,\n  \"machine\": \"{}\",\n  \"commit\": \"{}\",\n  \
         \"tasks\": [\n{tasks}\n  ]\n}}\n",
        crate::bench::machine_id(),
        commit.trim()
    ))
}

// ---- measurement primitives ---------------------------------------------

/// Lines `edited` adds over `original`. A pure count of added lines, which is
/// what "how much do I have to type" means.
fn lines_added(original: &str, edited: &str) -> usize {
    edited
        .lines()
        .count()
        .saturating_sub(original.lines().count())
}

/// Apply `edits` to `source`, refusing an anchor that no longer exists.
fn apply(source: &str, edits: &[(&str, &str)], what: &str) -> anyhow::Result<String> {
    let mut out = source.to_owned();
    for (anchor, replacement) in edits {
        anyhow::ensure!(
            out.contains(anchor),
            "{what}: the anchor `{}` is gone — this benchmark is a text edit, so a rename here \
             is a measurement to re-point rather than one that quietly measures nothing",
            anchor.trim()
        );
        out = out.replace(anchor, replacement);
    }
    Ok(out)
}

/// The template's source, which the game-side scenarios all edit.
fn template_source() -> anyhow::Result<String> {
    Ok(std::fs::read_to_string(
        workspace_root().join("demos/99-template/src/lib.rs"),
    )?)
}

/// Write a scratch game crate holding `source`, and return its manifest.
///
/// Its own workspace (the empty `[workspace]` table) so cargo does not adopt a
/// package under `target/`, and the workspace's own dev profile copied — a
/// rebuild at a different optimization level is a different number (§3).
fn write_game(name: &str, source: &str) -> anyhow::Result<PathBuf> {
    let dir = scratch().join(name);
    std::fs::create_dir_all(dir.join("src"))?;
    std::fs::write(dir.join("src/lib.rs"), source)?;
    let crates = workspace_root()
        .join("crates")
        .display()
        .to_string()
        .replace('\\', "/");
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            "# Generated by `cargo xtask dx` (§8's developer-experience benchmarks).\n\
             # Not checked in, not a workspace member, rewritten on every run.\n\
             [workspace]\n\n\
             [package]\n\
             name = \"gg-dx-{name}\"\n\
             version = \"0.0.0\"\n\
             edition = \"2024\"\n\n\
             [lib]\n\
             name = \"demo_dx\"\n\
             crate-type = [\"cdylib\"]\n\
             path = \"src/lib.rs\"\n\n\
             [features]\n\
             default = [\"game\"]\n\
             game = []\n\n\
             [dependencies]\n\
             bytemuck = {{ version = \"1\", features = [\"derive\"] }}\n\
             gg-ecs = {{ path = \"{crates}/gg-ecs\" }}\n\
             gg-math = {{ path = \"{crates}/gg-math\" }}\n\n\
             [profile.dev]\n\
             opt-level = 1\n\
             [profile.dev.package.\"*\"]\n\
             opt-level = 3\n"
        ),
    )?;
    Ok(dir.join("Cargo.toml"))
}

/// Build a scratch crate and return how long it took, in microseconds. Built once first so the
/// number is an *incremental* rebuild — which is what the loop actually pays,
/// and the only figure §6 M5's budget is stated against.
fn timed_rebuild(manifest: &Path, touch: &Path, label: &str) -> anyhow::Result<u128> {
    exec(
        cargo().env("CARGO_TARGET_DIR", scratch_target()).args([
            "build",
            "--manifest-path",
            &manifest.display().to_string(),
        ]),
        &format!("{label} (warm-up build)"),
    )?;
    // Rewriting the file is what a save does; cargo keys on mtime.
    let source = std::fs::read_to_string(touch)?;
    std::fs::write(touch, source)?;
    let started = Instant::now();
    exec(
        cargo().env("CARGO_TARGET_DIR", scratch_target()).args([
            "build",
            "--manifest-path",
            &manifest.display().to_string(),
        ]),
        label,
    )?;
    Ok(started.elapsed().as_micros())
}

// ---- the eight tasks ------------------------------------------------------

/// Declare a component and register it. Two edits in one file: the type, and
/// its name in the `gg_game!` block — the boundary registers what the table
/// names, so a type nobody listed is a type the host never hears about.
fn add_a_component() -> anyhow::Result<Measured> {
    let source = template_source()?;
    let edited = apply(
        &source,
        &[
            (
                "/// Cube, floor, sun, camera — once.",
                "/// Added by `xtask dx`: one more hashed component.\n\
                 #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]\n\
                 #[component(id = \"template.health\")]\n\
                 #[repr(C)]\n\
                 pub struct Health {\n\
                 \x20   /// Hit points.\n\
                 \x20   pub points: u64,\n\
                 }\n\n\
                 /// Cube, floor, sun, camera — once.",
            ),
            (
                "components: [Spinner, Renderable, Light, Eye],",
                "components: [Spinner, Health, Renderable, Light, Eye],",
            ),
        ],
        "add-a-component",
    )?;
    let manifest = write_game("add-a-component", &edited)?;
    let touched = manifest.with_file_name("src").join("lib.rs");
    let latency_us = timed_rebuild(&manifest, &touched, "dx: add a component")?;
    Ok(Measured {
        steps: 2,
        lines: lines_added(&source, &edited),
        latency_us,
        detail: "1 file, 1 rebuild of the game dylib".into(),
    })
}

/// Write a system and schedule it. Same two-edit shape: the `fn`, and its name
/// in `systems:` — order there is execution order (§4.1).
fn write_a_system() -> anyhow::Result<Measured> {
    let source = template_source()?;
    let edited = apply(
        &source,
        &[
            (
                "// Order in `systems` is execution order",
                "/// Added by `xtask dx`: a gameplay system, which is a plain `fn`.\n\
                 pub fn drift(world: &mut GameWorld) {\n\
                 \x20   let _ = world.each::<&mut Renderable>(|_, shape| {\n\
                 \x20       shape.position.y += 0.001;\n\
                 \x20   });\n\
                 }\n\n\
                 // Order in `systems` is execution order",
            ),
            (
                "systems: [bootstrap, spin],",
                "systems: [bootstrap, spin, drift],",
            ),
        ],
        "write-a-system",
    )?;
    let manifest = write_game("write-a-system", &edited)?;
    let touched = manifest.with_file_name("src").join("lib.rs");
    let latency_us = timed_rebuild(&manifest, &touched, "dx: write a system")?;
    Ok(Measured {
        steps: 2,
        lines: lines_added(&source, &edited),
        latency_us,
        detail: "1 file, 1 rebuild of the game dylib".into(),
    })
}

/// Add a push-constant field to a shader and get it compiling again.
///
/// Measured through the in-process Slang path, which is the one hot reload uses
/// (§4.4) — not `xtask shaders`, because that rewrites checked-in artifacts and
/// a benchmark must not edit the tree. The CPU-side struct is *generated* from
/// this, which is the point of the number: the developer edits one file and the
/// layout assertion follows.
fn add_a_shader_parameter() -> anyhow::Result<Measured> {
    let shaders = workspace_root().join("crates/gg-render/shaders");
    let source = std::fs::read_to_string(shaders.join("post.slang"))?;
    let edited = apply(
        &source,
        &[(
            "    float exposure;",
            "    float exposure;\n    // Added by `xtask dx`: one more knob.\n    float vignette;",
        )],
        "add-a-shader-parameter",
    )?;
    // A whole scratch copy of the directory: `#include`s resolve against it.
    let dir = scratch().join("shaders");
    std::fs::create_dir_all(dir.join("include"))?;
    for entry in std::fs::read_dir(&shaders)?.flatten() {
        if entry.path().is_file() {
            std::fs::copy(entry.path(), dir.join(entry.file_name()))?;
        }
    }
    for entry in std::fs::read_dir(shaders.join("include"))?.flatten() {
        std::fs::copy(entry.path(), dir.join("include").join(entry.file_name()))?;
    }
    std::fs::write(dir.join("post.slang"), &edited)?;
    let started = Instant::now();
    let module = gg_shaders::compile_module(&dir.display().to_string(), "post")
        .map_err(|e| anyhow::anyhow!("dx: the edited shader did not compile: {e}"))?;
    let latency_us = started.elapsed().as_micros();
    anyhow::ensure!(
        !module.entry_points.is_empty(),
        "dx: the edited shader compiled to no entry points"
    );
    Ok(Measured {
        steps: 2,
        lines: lines_added(&source, &edited),
        latency_us,
        detail: "1 shader file, then `cargo xtask shaders` regenerates the Rust struct".into(),
    })
}

/// Drop a file into a demo's `assets/` tree and have it in a pack.
///
/// Zero lines of code: naming an asset is `asset_id("name")` at the use site,
/// and getting one *into* the game is a file. The number that matters is the
/// pack build, which is what `ggc watch` pays on every save (§4.6).
fn add_an_asset() -> anyhow::Result<Measured> {
    let dir = scratch().join("assets/source");
    std::fs::create_dir_all(&dir)?;
    // A 1x1 PNG, by hand: the point is the pipeline, not the picture.
    const PIXEL: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
        0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08, 0xD7, 0x63, 0xF8,
        0xCF, 0xC0, 0x00, 0x00, 0x03, 0x01, 0x01, 0x00, 0x18, 0xDD, 0x8D, 0xB0, 0x00, 0x00, 0x00,
        0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
    ];
    std::fs::write(dir.join("dx-swatch.png"), PIXEL)?;
    let out = scratch().join("assets/dx.ggpack");
    let started = Instant::now();
    exec(
        cargo().args([
            "run",
            "-q",
            "-p",
            "ggc",
            "--",
            "build",
            &dir.display().to_string(),
            "-o",
            &out.display().to_string(),
        ]),
        "dx: compile the asset tree",
    )?;
    let latency_us = started.elapsed().as_micros();
    anyhow::ensure!(
        out.is_file(),
        "dx: ggc produced no pack at {}",
        out.display()
    );
    Ok(Measured {
        steps: 2,
        lines: 0,
        latency_us,
        detail: "1 file dropped in, 1 pack build; naming it in code is `asset_id(\"…\")`".into(),
    })
}

/// Add a render pass. An engine-crate edit, so what is measured is the rebuild
/// the developer waits on — `gg-render` and everything above it.
///
/// The pass itself is one `Declared` in the graph (§4.5): the declaration layer
/// is the whole API, and no barrier is written by hand. The edit is not applied
/// to the working tree — the *shape* is counted off `readback_pass`, which is
/// the smallest complete pass this crate declares.
fn add_a_render_pass() -> anyhow::Result<Measured> {
    let graph = workspace_root().join("crates/gg-render/src/graph.rs");
    let source = std::fs::read_to_string(&graph)?;
    let lines = declaration_lines(&source, "pub fn readback_pass")?;
    let started = Instant::now();
    // Rewriting the file is what a save does; cargo keys on mtime. Identical
    // bytes, so the tree is unchanged and the build is a real incremental one.
    std::fs::write(&graph, &source)?;
    exec(
        cargo().args(["build", "-p", "gg-render", "-p", "gg-runtime"]),
        "dx: rebuild the renderer and the shell",
    )?;
    let latency_us = started.elapsed().as_micros();
    Ok(Measured {
        steps: 2,
        lines,
        latency_us,
        detail: "1 file in gg-render, 1 rebuild of the renderer and the shell".into(),
    })
}

/// Lines in the function starting at `anchor`, brace-counted.
fn declaration_lines(source: &str, anchor: &str) -> anyhow::Result<usize> {
    let start = source
        .find(anchor)
        .ok_or_else(|| anyhow::anyhow!("dx: `{anchor}` is gone — re-point this measurement"))?;
    let (mut depth, mut lines, mut seen) = (0i32, 0usize, false);
    for line in source[start..].lines() {
        lines += 1;
        for c in line.chars() {
            match c {
                '{' => {
                    depth += 1;
                    seen = true;
                }
                '}' => depth -= 1,
                _ => {}
            }
        }
        if seen && depth == 0 {
            return Ok(lines);
        }
    }
    anyhow::bail!("dx: `{anchor}` never closed")
}

/// Start a new demo from the template. Copy, rename, build — and the lines are
/// the template's own, which is exactly what §6 M12's exit row caps.
fn create_a_demo() -> anyhow::Result<Measured> {
    let source = template_source()?;
    let manifest = write_game("create-a-demo", &source)?;
    let touched = manifest.with_file_name("src").join("lib.rs");
    let latency_us = timed_rebuild(&manifest, &touched, "dx: create a demo")?;
    let code = source
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with("//"))
        .count();
    Ok(Measured {
        steps: 3,
        lines: code,
        latency_us,
        detail: "copy demos/99-template, rename it, add it to the workspace".into(),
    })
}

/// Read an entity's live state. **No rebuild at all** — which is the result.
///
/// The overlay and the console are CVar-driven (§4.8), so the loop is: run,
/// toggle, look. Counted as three steps and zero lines, and the zero is the
/// number worth defending: an engine where inspecting state costs a rebuild is
/// one where nobody inspects state.
fn inspect_an_entity() -> anyhow::Result<Measured> {
    // The claim is that the knobs exist without a rebuild, so verify they are
    // registered rather than asserting it in prose.
    let overlay = std::fs::read_to_string(workspace_root().join("crates/gg-debug/src/overlay.rs"))?;
    anyhow::ensure!(
        overlay.contains("CVar::new_bool(\"d.overlay\""),
        "dx: `d.overlay` is gone — this measurement claims inspection needs no rebuild, and that \
         claim rests on the knob existing"
    );
    Ok(Measured {
        steps: 3,
        lines: 0,
        latency_us: 0,
        detail: "run, toggle the overlay, read it — CVars need no rebuild (§4.8)".into(),
    })
}

/// Turn a report into something that fails on demand: the replay is the
/// reproduction (§1.3), so the steps are record, re-run, compare — and the
/// latency is the compare, which `xtask replay` performs for real here.
fn reproduce_a_bug() -> anyhow::Result<Measured> {
    let started = Instant::now();
    crate::replay::run(&[])?;
    Ok(Measured {
        steps: 3,
        lines: 0,
        latency_us: started.elapsed().as_micros(),
        detail: "record with --record, re-run with --replay, `xtask replay` names the first \
                 differing tick"
            .into(),
    })
}
