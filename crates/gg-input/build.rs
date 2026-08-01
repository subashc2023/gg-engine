//! Stamps the engine commit into the binary so a replay header can name its own
//! reproduction environment with no external bookkeeping (§4.7, §1.2).
//!
//! Absent git — a source tarball, or the rsync'd WSL lane, which deliberately
//! excludes `.git` — the stamp reads `unknown`. That is the honest answer and
//! not a build failure: a replay recorded from a tree we cannot identify should
//! say so rather than claim a commit.

fn main() {
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-env-changed=GG_ENGINE_COMMIT");

    let commit = std::env::var("GG_ENGINE_COMMIT").ok().or_else(|| {
        let out = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
    });
    let commit = commit.filter(|c| !c.is_empty()).unwrap_or("unknown".into());
    println!("cargo:rustc-env=GG_ENGINE_COMMIT={commit}");
}
