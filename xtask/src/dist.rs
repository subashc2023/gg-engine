//! The dist gate (§5.8), M0A form: the exact `tier-dist` combination builds and
//! runs through gg-runtime, and the lab equipment provably unbolted (§1.13) —
//! no Tracy/notify/tools in the resolved dist graph, no tracy strings in the
//! binary. Demos join the run at M1+.
//!
//! The gate has *presence* checks too, and they are not a symmetry exercise:
//! embedded SPIR-V must be there (§4.4), and so must the input recorder, which
//! is the §1.2 bug-report channel and explicitly not lab equipment (§2). A
//! shipped build that cannot record a replay cannot produce a bug report anyone
//! can reproduce, so its absence is a gate failure exactly like Tracy's
//! presence is.

use std::path::{Path, PathBuf};

use crate::util::{cargo, run as exec, run_capture, workspace_root};

/// Shipped binaries that must carry the input recorder (§5.8, §2). Demos 00 and
/// 01 predate the sim and record nothing; `gg-runtime` joins when it owns the
/// loop and the recorder with it (M5). An explicit list rather than "everything
/// linking gg-input", because linking it for `Key` is not shipping a recorder.
const RECORDER_DEMOS: &[&str] = &["demo-02-mesh"];

const BANNED_DIST_CRATES: &[&str] = &[
    "tracy-client",
    "tracing-tracy",
    "notify",
    "ggc",
    // §2's "the runtime never parses JSON", as a machine rather than a habit.
    // `gltf` and `serde_json` are `ggc`'s and reach no runtime graph; banning
    // them by name means a dependency added three crates away that happens to
    // pull one in fails the gate instead of quietly shipping a JSON parser.
    "gltf",
    "serde_json",
    // The codecs, same argument (§4.6). A ship decodes no PNG, compresses no
    // block and reorders no index buffer — every one of those already happened,
    // offline, and the pack is the evidence. `zstd` is deliberately *not* here:
    // a KTX2 level is a zstd frame, so the runtime decompresses one on the way
    // to the staging ring.
    "image",
    "intel_tex_2",
    "meshopt",
    "gg-golden",
    // The instruments (§3: instrumentation never in the dist graph, §4.8).
    // Banning the crate and not only Tracy is what keeps the console and the
    // overlay out too — they are absent, not disabled.
    "gg-debug",
    // The editor (§6 M15), on §7's "shipping the lab" — an editor first of all.
    // Named separately from `gg-debug` because it is a separate feature on a
    // separate crate: a tier could want instruments without an editor, and the
    // absence proof has to say which one it checked.
    "gg-editor",
    // §6 M16's seam record. A third name for the same reason the second one
    // exists: a shipped game has no reload seam, so a binary that could still
    // publish one would be carrying a machine with nothing to report on — and
    // the file it writes names the player's paths.
    "gg-agent",
];

/// Byte needles no shipped binary may contain (§1.13, §5.8). The validation
/// layer name earns its place from a feature default: `gg-rhi` is
/// `default = ["validation", "gpu-timings"]`, so dist safety rests entirely on
/// every consumer writing `default-features = false` — this is what turns a
/// forgotten one red here instead of shipping a binary that demands a layer on
/// the player's machine.
const BANNED_DIST_BYTES: &[&str] = &[
    "tracy",
    "Tracy",
    "VK_LAYER_KHRONOS_validation",
    // The editor's own log lines (§6 M15). The crate ban above is the real
    // proof; this catches the case the graph cannot see — a panel's text
    // reaching a shipped binary through some other crate's constant.
    "editor: field nudged",
];

/// Byte needles a shipped binary must contain: the two log lines that exist only
/// on the paths reaching `World::snapshot` + `Snapshot::encode` (§6 M14).
///
/// Log messages rather than the format magics, which are `[u8; 4]` constants a
/// compiler is free to materialize as an immediate rather than as a string — a
/// presence check that can pass or fail on codegen mood is not a check.
const SAVE_PATH_STRINGS: &[&str] = &["save written", "play mode entered"];

pub fn gate() -> anyhow::Result<()> {
    // Build and run the exact shipping combination (§1.10: an untested dist is
    // an untested code path wearing a nicer name).
    exec(
        cargo().args([
            "build",
            "-p",
            "gg-runtime",
            "--profile",
            "dist",
            "--no-default-features",
            "--features",
            "tier-dist",
        ]),
        "build gg-runtime [tier-dist, dist profile]",
    )?;
    let exe = workspace_root().join("target/dist").join(if cfg!(windows) {
        "gg-runtime.exe"
    } else {
        "gg-runtime"
    });
    // Running it needs a game (§2: the shell is the same program in every tier
    // and the game is what makes it a game), which has been true since demo 03
    // existed — before that the shell ran bare and this line was a bare spawn.
    let games = game_dylibs()?;
    let game = games
        .first()
        .ok_or_else(|| anyhow::anyhow!("no game dylib to run the dist shell over"))?;
    exec(
        std::process::Command::new(&exe)
            .arg("--game")
            .arg(game)
            .args(["--frames", "100"])
            .env("GG_HEADLESS", "1"),
        "run dist gg-runtime over a dist game dylib, 100 frames headless",
    )?;

    // Graph absence check — the authoritative half.
    let tree = run_capture(
        cargo().args([
            "tree",
            "-p",
            "gg-runtime",
            "--no-default-features",
            "--features",
            "tier-dist",
            "-e",
            "normal",
            "--prefix",
            "none",
        ]),
        "cargo tree (dist graph)",
    )?;
    let offenders: Vec<&str> = BANNED_DIST_CRATES
        .iter()
        .copied()
        .filter(|c| {
            tree.lines()
                .any(|l| l.split_whitespace().next() == Some(*c))
        })
        .collect();
    anyhow::ensure!(
        offenders.is_empty(),
        "dist graph contains banned crates {offenders:?} (§5.8, §1.13)"
    );

    // Symbol/string absence check — the belt to the graph's suspenders.
    let bytes = std::fs::read(&exe)?;
    for needle in BANNED_DIST_BYTES {
        anyhow::ensure!(
            !bytes.windows(needle.len()).any(|w| w == needle.as_bytes()),
            "dist binary contains `{needle}` bytes — lab equipment failed to unbolt (§1.13)"
        );
    }
    // And the presence check the shell earned at M5, when it took delivery of
    // the recorder (§5.8, §2): the shipping shell is the binary a player's bug
    // report comes out of. `xtask reload` proves it behaviourally by recording a
    // file and reading it back; this is the byte-level half, which holds even
    // when nobody runs it.
    anyhow::ensure!(
        bytes.windows(4).any(|w| w == gg_input::replay::MAGIC),
        "the dist shell carries no replay magic — the input recorder ships in every tier (§2, \
         §5.8): it is the bug-report channel, not lab equipment"
    );
    // §6 M14's fourth exit row, as a byte check. The *read* half of the snapshot
    // core already shipped before M14 — `--restore` is unconditional, so a dist
    // binary could always take a handoff — and the *write* half did not, because
    // its only caller was the hot-reload swap and fat LTO strips what nothing
    // calls. `--save` is that caller. This is the check that would have caught
    // the milestone's premise being false, so it is the one that keeps it true.
    for needle in SAVE_PATH_STRINGS {
        anyhow::ensure!(
            bytes.windows(needle.len()).any(|w| w == needle.as_bytes()),
            "the dist shell carries no `{needle}` — a shipping build must be able to *write* a \
             save, not only read one (§6 M14)"
        );
    }

    // Demos join the dist gate at their milestones (§5.8): the exact
    // tier-dist combination builds, and the binary passes the absence/presence
    // byte checks. Running it presents to a window, so the run itself is
    // `cargo xtask interactive` (§1.5).
    //
    // A presence check that matches nothing passes vacuously, so the gate keeps
    // score and insists at least one shipped binary carried the recorder.
    let mut recorder_seen = false;
    for demo in ["demo-00-clear", "demo-01-triangle", "demo-02-mesh"] {
        // Graph absence per demo (§4.4): no compiler, no watcher, no harness
        // in what ships — dist embeds SPIR-V and nothing that makes it.
        let tree = run_capture(
            cargo().args([
                "tree",
                "-p",
                demo,
                "--no-default-features",
                "--features",
                "tier-dist",
                "-e",
                "normal",
                "--prefix",
                "none",
            ]),
            &format!("cargo tree ({demo} dist graph)"),
        )?;
        let offenders: Vec<&str> = BANNED_DIST_CRATES
            .iter()
            .chain(&["gg-shaders", "shader-slang"])
            .copied()
            .filter(|c| {
                tree.lines()
                    .any(|l| l.split_whitespace().next() == Some(*c))
            })
            .collect();
        anyhow::ensure!(
            offenders.is_empty(),
            "{demo} dist graph contains banned crates {offenders:?} (§5.8, §4.4)"
        );
        exec(
            cargo().args([
                "build",
                "-p",
                demo,
                "--profile",
                "dist",
                "--no-default-features",
                "--features",
                "tier-dist",
            ]),
            &format!("build {demo} [tier-dist, dist profile]"),
        )?;
        // The demo *run* creates a window (§1.5) and lives in the manual
        // windowed suite, `cargo xtask interactive`; the automated gate stops
        // at build + absence/presence checks on the binary.
        let demo_exe = workspace_root().join("target/dist").join(if cfg!(windows) {
            format!("{demo}.exe")
        } else {
            demo.to_string()
        });

        // Same absence check as the shell: lab equipment unbolted (§1.13).
        let bytes = std::fs::read(&demo_exe)?;
        for needle in BANNED_DIST_BYTES {
            anyhow::ensure!(
                !bytes.windows(needle.len()).any(|w| w == needle.as_bytes()),
                "dist {demo} contains `{needle}` bytes — lab equipment failed to unbolt (§1.13)"
            );
        }

        // Recorder presence (§5.8, live from M4B): the replay format's magic in
        // the shipped bytes. Graph presence would not do — every windowed demo
        // links gg-input transitively through gg-platform for `Key`, and only a
        // binary that *records* keeps the codec past dead-code elimination,
        // which is exactly the distinction the check is about.
        if RECORDER_DEMOS.contains(&demo) {
            anyhow::ensure!(
                bytes.windows(4).any(|w| w == gg_input::replay::MAGIC),
                "dist {demo} carries no replay magic — the input recorder ships in every tier \
                 (§2, §5.8): it is the bug-report channel, not lab equipment"
            );
            recorder_seen = true;
        }

        // And one *presence* check for the shader-bearing demos (§6 M2 exit):
        // the dist binary contains embedded SPIR-V — the magic word, in the
        // little-endian byte order include_bytes! preserves.
        if demo != "demo-00-clear" {
            let spirv_magic: &[u8] = &[0x03, 0x02, 0x23, 0x07];
            anyhow::ensure!(
                bytes.windows(4).any(|w| w == spirv_magic),
                "dist {demo} contains no embedded SPIR-V (§4.4 offline path)"
            );
        }
    }

    // Before the dist-verify build below, which overwrites the shell's binary
    // and its artifact: what a release archives is the `tier-dist` pair, and
    // archiving whatever was linked last is how the two get swapped silently.
    symbolization()?;

    // dist-verify must also build here; it is exercised for real by §5.6c (M4B).
    exec(
        cargo().args([
            "build",
            "-p",
            "gg-runtime",
            "--profile",
            "dist",
            "--no-default-features",
            "--features",
            "tier-dist-verify",
        ]),
        "build gg-runtime [tier-dist-verify, dist profile]",
    )?;

    anyhow::ensure!(
        recorder_seen,
        "no dist binary carried the input recorder — the presence check (§5.8) matched nothing,          which is a vacuous pass, not a green one"
    );

    println!("xtask dist: gate green (lab equipment absent, SPIR-V and the recorder present)");
    Ok(())
}

/// The frame the canary's backtrace must name. Cargo builds the example as
/// `crash_canary`, hyphen normalized — the module path a backtrace prints, not
/// the target name.
const CANARY_FRAME: &str = "crash_canary::the_pass_that_faulted";

/// The same function as a bare symbol, which is how it appears in an artifact's
/// string tables — `.debug_str` in a `.dwp`, the symbol records in a PDB.
const CANARY_SYMBOL: &str = "the_pass_that_faulted";

/// §6 M8's last exit criterion: a dist crash dump symbolizes cleanly against the
/// archived split-debug artifact.
///
/// Two claims, and the second is not reachable everywhere:
///
/// 1. **The symbols left the binary and landed in the artifact** — the stripped
///    dist binary does not contain the faulting function's name and the archived
///    artifact does. Checked on every platform, and the half that would catch a
///    `strip` that stopped stripping or a `debug` setting that stopped emitting.
/// 2. **A crash actually resolves against it**, run rather than asserted: the
///    artifact is *moved* into the archive and the binary crashed alone, whose
///    backtrace must be anonymous; then the artifact is put back beside it and
///    the same crash must name the frame and its line. That pair is the release
///    workflow. It runs on MSVC, where the platform's own symbolizer loads the
///    PDB the binary points at. On ELF it does not: `strip = "symbols"` leaves
///    the backtrace with addresses and nothing else, and a `.dwp` is not
///    something the in-process symbolizer will pick up — putting a Linux dump
///    back together is `llvm-symbolizer`'s job, offline, and driving one from
///    here would gate a tool rather than the artifact. Named as a residual in §6
///    M8 rather than skipped quietly.
fn symbolization() -> anyhow::Result<()> {
    let root = workspace_root();
    // Outside `target/`, deliberately: §9 asks that a dump from the field
    // symbolize against the artifact its release archived, and under `target/`
    // that artifact is one `cargo clean` from gone — which would make the claim
    // permanently false for a tag already cut. `.gitignore`d instead; a PDB is
    // megabytes and belongs in a release, not in the history.
    let archive = root.join("dist-symbols").join(commit()?);
    std::fs::create_dir_all(&archive)?;

    // The shell's own artifact first: it is the one a release actually archives.
    // Nothing runs against it here — `gg-runtime` has no way to be told to die
    // and should not grow one — so what is checked is that the build produced it
    // at all, which is what §3's "archived per release" had been promising about
    // a file that did not exist until the profile said `debug`.
    let shell = split_debug(&root.join("target/dist"), "gg-runtime")?;
    let bytes = std::fs::copy(&shell, archive.join(file_name(&shell)?))?;
    anyhow::ensure!(
        bytes > 64 * 1024,
        "the archived split-debug artifact for the dist shell is {bytes} bytes — too small to \
         carry symbols, so `strip` has nothing to have stripped (§3)"
    );
    println!(
        "xtask dist: archived {} ({} KiB) → {}",
        file_name(&shell)?,
        bytes / 1024,
        archive.display()
    );

    // The canary is an example target of the shell's own crate, so it is this
    // profile's codegen — fat LTO, one codegen unit, stripped — and enters no
    // shipped binary (§6 M8).
    exec(
        cargo().args([
            "build",
            "-p",
            "gg-runtime",
            "--example",
            "crash-canary",
            "--profile",
            "dist",
            "--no-default-features",
            "--features",
            "tier-dist",
        ]),
        "build the dist symbolization canary [tier-dist, dist profile]",
    )?;
    let examples = root.join("target/dist/examples");
    let exe = examples.join(if cfg!(windows) {
        "crash-canary.exe"
    } else {
        "crash-canary"
    });
    let artifact = split_debug(&examples, "crash-canary")?;
    let archived = archive.join(file_name(&artifact)?);

    // Claim 1, on every platform: the names are in the artifact and out of the
    // binary. A byte scan rather than a parser — one function name in a string
    // table is not worth a PDB reader and a DWARF reader to read it out of two
    // formats, and what would be wrong here is `strip` or `debug`, not parsing.
    anyhow::ensure!(
        contains(&artifact, CANARY_SYMBOL)?,
        "the split-debug artifact {} carries no `{CANARY_SYMBOL}` — the dist profile's `debug` \
         emitted nothing worth archiving (§3)",
        artifact.display()
    );
    anyhow::ensure!(
        !contains(&exe, CANARY_SYMBOL)?,
        "the shipped dist binary {} still carries `{CANARY_SYMBOL}` — `strip` did not strip, so \
         the artifact beside it is a copy rather than the only place the symbols live (§3)",
        exe.display()
    );

    // A *move*, not a copy: what ships is the binary with nothing beside it, and
    // a check run with the artifact still in place would pass for the wrong
    // reason. On MSVC the binary records the artifact's absolute path, so this
    // genuinely takes it out of reach.
    std::fs::rename(&artifact, &archived)?;
    if !cfg!(target_env = "msvc") {
        println!(
            "xtask dist: {} archived; the symbols are in it and out of the binary. The crash \
             round trip is MSVC-only — an ELF dump is symbolized offline (§6 M8 residual).",
            file_name(&archived)?
        );
        return Ok(());
    }

    let stripped = crash(&exe)?;
    anyhow::ensure!(
        !stripped.contains(CANARY_FRAME),
        "the stripped dist binary named `{CANARY_FRAME}` with its split-debug artifact archived \
         away — the symbols never left the binary, so this gate is proving nothing:\n{stripped}"
    );

    // And back together, the way an engineer meets a dump from the field.
    std::fs::copy(&archived, &artifact)?;
    let symbolized = crash(&exe)?;
    anyhow::ensure!(
        symbolized.contains(CANARY_FRAME),
        "a dist crash did not symbolize against the archived artifact {}: no `{CANARY_FRAME}` \
         frame (§6 M8 exit)\n{symbolized}",
        archived.display()
    );
    anyhow::ensure!(
        symbolized.contains("crash-canary.rs:"),
        "the archived artifact named the frame but no line — `debug` is too thin to symbolize a \
         crash report against (§3)\n{symbolized}"
    );
    println!(
        "xtask dist: a dist crash is anonymous stripped and names `{CANARY_FRAME}` with its \
         archived artifact restored (§6 M8)"
    );
    Ok(())
}

/// Whether `path`'s bytes contain `needle`. Artifacts are megabytes, which is
/// the same scale the gate's other byte checks already read whole.
fn contains(path: &Path, needle: &str) -> anyhow::Result<bool> {
    let bytes = std::fs::read(path)?;
    Ok(bytes.windows(needle.len()).any(|w| w == needle.as_bytes()))
}

/// Run the canary and hand back its panic output. The failure *is* the result,
/// so a successful exit is the error case.
fn crash(exe: &Path) -> anyhow::Result<String> {
    let out = std::process::Command::new(exe)
        .env("RUST_BACKTRACE", "1")
        .output()
        .map_err(|e| anyhow::anyhow!("failed to spawn the symbolization canary: {e}"))?;
    anyhow::ensure!(
        !out.status.success(),
        "the symbolization canary exited cleanly — it exists to crash"
    );
    Ok(crate::util::plain(&out.stderr))
}

/// The `.pdb`/`.dwp` a `split-debuginfo = "packed"` build left beside `target`.
/// Matched on a hyphen-normalized stem because cargo names the binary after the
/// target and the artifact after the crate, and the two spell the same name
/// differently.
fn split_debug(dir: &Path, target: &str) -> anyhow::Result<PathBuf> {
    let wanted = target.replace('-', "_");
    let found = std::fs::read_dir(dir)?.flatten().find(|entry| {
        let path = entry.path();
        path.extension().is_some_and(|e| e == "pdb" || e == "dwp")
            && path
                .file_stem()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.replace('-', "_") == wanted)
    });
    found.map(|e| e.path()).ok_or_else(|| {
        anyhow::anyhow!(
            "no split-debug artifact for `{target}` in {} — the dist profile must carry `debug` \
             and `split-debuginfo` for §3's archived artifact to exist",
            dir.display()
        )
    })
}

fn file_name(path: &Path) -> anyhow::Result<String> {
    Ok(path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow::anyhow!("unnamed path {}", path.display()))?
        .to_string())
}

/// The commit the archive is keyed on. A dirty tree still archives — the
/// artifact is keyed by what produced it, and refusing here would only mean no
/// artifact for exactly the builds most likely to crash.
fn commit() -> anyhow::Result<String> {
    let out = run_capture(
        std::process::Command::new("git")
            .current_dir(workspace_root())
            .args(["rev-parse", "--short", "HEAD"]),
        "git rev-parse HEAD",
    )?;
    Ok(out.trim().to_string())
}

/// Game crates under dist (§5.8, live from M5). They are dylibs, not binaries,
/// so the checks above do not fit: there is no `tier-dist` feature to select —
/// a game crate has no tiers, it is the same code in every one — and no embedded
/// SPIR-V or recorder of its own. What is checked is the pair of claims §2's
/// Game-code boundary row makes about dist: the artifact builds under the
/// shipping profile, and nothing that reloads it is in its graph.
///
/// The behavioural half — the dylib is loaded *once*, with no watcher behind it
/// — is `xtask reload`'s, where a dist shell actually runs one.
fn game_dylibs() -> anyhow::Result<Vec<std::path::PathBuf>> {
    let mut built = Vec::new();
    for game in crate::budgets::game_crates()? {
        let tree = run_capture(
            cargo().args(["tree", "-p", &game, "-e", "normal", "--prefix", "none"]),
            &format!("cargo tree ({game} dist graph)"),
        )?;
        let offenders: Vec<&str> = BANNED_DIST_CRATES
            .iter()
            .chain(&["gg-shaders", "shader-slang"])
            .copied()
            .filter(|c| {
                tree.lines()
                    .any(|l| l.split_whitespace().next() == Some(*c))
            })
            .collect();
        anyhow::ensure!(
            offenders.is_empty(),
            "game crate {game} links {offenders:?} (§5.8) — a game dylib that carried the watcher \
             would reload itself"
        );
        exec(
            cargo().args(["build", "-p", &game, "--profile", "dist"]),
            &format!("build {game} [dist profile]"),
        )?;
        built.push(workspace_root().join("target/dist").join(if cfg!(windows) {
            format!("{}.dll", game.replace('-', "_"))
        } else {
            format!("lib{}.so", game.replace('-', "_"))
        }));
        println!("xtask dist: {game} builds as a shipping game dylib, watcher-free (§5.8)");
    }
    Ok(built)
}

/// The dist demo runs — part of the manual windowed suite (`cargo xtask
/// interactive`, §1.5): the exact tier-dist binaries run 100 frames against a
/// real window's swapchain on the pinned lavapipe, exiting nonzero on
/// validation messages or leaks.
pub fn demo_runs() -> anyhow::Result<()> {
    for demo in ["demo-00-clear", "demo-01-triangle", "demo-02-mesh"] {
        exec(
            cargo().args([
                "build",
                "-p",
                demo,
                "--profile",
                "dist",
                "--no-default-features",
                "--features",
                "tier-dist",
            ]),
            &format!("build {demo} [tier-dist, dist profile]"),
        )?;
        let demo_exe = workspace_root().join("target/dist").join(if cfg!(windows) {
            format!("{demo}.exe")
        } else {
            demo.to_string()
        });
        let mut run = std::process::Command::new(&demo_exe);
        run.args(["--frames", "100"]).env("GG_HEADLESS", "1");
        if cfg!(windows) {
            run.env("VK_DRIVER_FILES", crate::probe::ensure_lavapipe()?);
        } else {
            run.env("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/lvp_icd.json");
        }
        exec(&mut run, &format!("run dist {demo}, 100 frames headless"))?;
    }
    Ok(())
}
