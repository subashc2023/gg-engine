//! The dist gate (§5.8), M0A form: the exact `tier-dist` combination builds and
//! runs through gg-runtime, and the lab equipment provably unbolted (§1.13) —
//! no Tracy/notify/tools in the resolved dist graph, no tracy strings in the
//! binary. Demos join the run at M1+; the recorder *presence* check activates
//! at M4B when the recorder exists.

use crate::util::{cargo, run as exec, run_capture, workspace_root};

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

    println!("xtask dist: gate green (recorder presence check activates at M4B)");
    Ok(())
}
