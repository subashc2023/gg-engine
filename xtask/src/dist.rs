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
    "gg-golden",
];

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
    exec(
        std::process::Command::new(&exe).env("GG_HEADLESS", "1"),
        "run dist gg-runtime headless",
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
    for needle in [b"tracy" as &[u8], b"Tracy"] {
        anyhow::ensure!(
            !bytes.windows(needle.len()).any(|w| w == needle),
            "dist binary contains `{}` bytes — lab equipment failed to unbolt (§1.13)",
            String::from_utf8_lossy(needle)
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
        for needle in [b"tracy" as &[u8], b"Tracy"] {
            anyhow::ensure!(
                !bytes.windows(needle.len()).any(|w| w == needle),
                "dist {demo} contains `{}` bytes — lab equipment failed to unbolt (§1.13)",
                String::from_utf8_lossy(needle)
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
