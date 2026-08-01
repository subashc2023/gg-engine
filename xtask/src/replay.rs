//! `cargo xtask replay [--bless]` — the other side of the determinism gate
//! (§5.6).
//!
//! The gate itself is a test in `demo-02-mesh`, because it has to run wherever
//! the sim runs, including under qemu where no `xtask` binary can. This is the
//! *authoring* half: it regenerates the curated replay and the checked-in hash
//! baselines from the same code the test reads them with, so the two cannot
//! drift.
//!
//! `--bless` is deliberately a separate verb rather than a fallback. A gate
//! that repairs itself on failure is a rubber stamp (§6 M0A, spike 3's rule for
//! the FP baseline), so a red gate writes its fresh sequence *beside* the
//! baseline and stops; replacing the file requires being able to say which
//! change produced the diff, in a review.

use crate::util::{run_capture, workspace_root};
use demo_02_mesh::{gate, sim};

pub fn run(args: &[&str]) -> anyhow::Result<()> {
    if args.contains(&"--bless") {
        bless()
    } else {
        check()
    }
}

/// Run the gate here, and say what the checked-in files claim.
fn check() -> anyhow::Result<()> {
    let dir = gate::replay_dir();
    let replay = gg_input::Replay::decode(&gate::read(&gate::replay_path(gate::CURATED))?)?;
    let meta = replay.meta();
    println!(
        "xtask replay: {} — {} ticks, {} change record(s), contract v{}, blessed at {}",
        gate::CURATED,
        replay.ticks(),
        replay.change_count(),
        meta.contract,
        meta.engine_commit,
    );

    let sequence = sim::hash_sequence(&replay)?;
    let baseline = gate::parse_baseline(&String::from_utf8(gate::read(&gate::baseline_path(
        gate::CURATED,
    ))?)?)?;
    if let Some(divergence) = gate::compare(&sequence, &baseline) {
        let fresh = workspace_root().join("target/determinism/demo02-curated.hashes.actual");
        if let Some(parent) = fresh.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(
            &fresh,
            gate::encode_baseline(&gate::sequence_entries(&sequence)),
        )?;
        anyhow::bail!(
            "{divergence}\nfresh sequence written to {} — diff it against {}\n\
             If the change is intended, `cargo xtask replay --bless`; the diff belongs in the review.",
            fresh.display(),
            gate::baseline_path(gate::CURATED).display(),
        );
    }
    println!(
        "xtask replay: curated sequence matches ({} ticks); chaos: {} seed(s) x {} ticks",
        sequence.len(),
        gate::CHAOS_SEEDS.len(),
        gate::CHAOS_TICKS,
    );
    println!("xtask replay: baselines live in {}", dir.display());
    Ok(())
}

/// Rewrite the curated replay and both baselines. A reviewed act.
fn bless() -> anyhow::Result<()> {
    let dir = gate::replay_dir();
    std::fs::create_dir_all(&dir)?;

    let commit = run_capture(
        std::process::Command::new("git")
            .current_dir(workspace_root())
            .args(["rev-parse", "HEAD"]),
        "git rev-parse HEAD",
    )
    .map(|s| s.trim().to_owned())
    .unwrap_or_else(|_| "unknown".to_owned());

    // The replay names the commit it was blessed at (§4.7: the file identifies
    // its own reproduction environment). Everything else about it is scripted,
    // so re-blessing an unchanged tree changes exactly this one field.
    let mut replay = gate::curated_replay();
    replay.set_engine_commit(&commit);
    std::fs::write(gate::replay_path(gate::CURATED), replay.encode())?;

    let sequence = sim::hash_sequence(&replay)?;
    std::fs::write(
        gate::baseline_path(gate::CURATED),
        gate::encode_baseline(&gate::sequence_entries(&sequence)),
    )?;

    let mut chaos = Vec::new();
    for &seed in gate::CHAOS_SEEDS {
        let replay = gate::chaos_replay(seed, gate::CHAOS_TICKS);
        let sequence = sim::hash_sequence(&replay)?;
        chaos.extend(gate::checkpoints(&sequence));
    }
    std::fs::write(gate::baseline_path("chaos"), gate::encode_baseline(&chaos))?;

    println!(
        "xtask replay: blessed {} ({} ticks) and {} chaos checkpoint(s) at {commit} — \
         a deliberate, reviewed act; the diff belongs in the review (§5.6)",
        gate::CURATED,
        sequence.len(),
        chaos.len(),
    );
    Ok(())
}
