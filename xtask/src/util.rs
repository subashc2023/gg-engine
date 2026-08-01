//! Shared plumbing: workspace paths, command running with headless + polite
//! mode (§1.5, §5), and source-tree walking for the grep gates.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Lowercase hex of a SHA-256 digest.
///
/// Spelled out by hand because sha2 0.11 returns `hybrid_array::Array` rather
/// than the old `GenericArray`, and `Array` implements no `LowerHex` — so the
/// `format!("{:x}", ..)` this replaces no longer compiles. The output is
/// byte-identical to the old spelling (two lowercase digits per byte), which
/// matters: this string is frozen into checked-in codegen headers and into the
/// pinned-lavapipe checksum, so a formatting change would be a silent
/// invalidation of both.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    use std::fmt::Write as _;

    sha2::Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut s, b| {
            // Infallible: writing to a String cannot fail.
            let _ = write!(s, "{b:02x}");
            s
        })
}

pub fn workspace_root() -> PathBuf {
    // xtask lives at <root>/xtask; CARGO_MANIFEST_DIR is compile-time truth.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// A `cargo` invocation rooted at the workspace, headless by law (§1.5), and
/// polite when the machine is flagged in-use via `GG_POLITE` (§5): bounded
/// build jobs; priority is left to the OS scheduler until a real need shows.
pub fn cargo() -> Command {
    let mut cmd = Command::new("cargo");
    cmd.current_dir(workspace_root());
    cmd.env("GG_HEADLESS", "1");
    if std::env::var_os("GG_POLITE").is_some() {
        let jobs = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(1))
            .unwrap_or(4);
        cmd.env("CARGO_BUILD_JOBS", jobs.to_string());
    }
    // Slang and the Vulkan loader live in the SDK; make its Bin reachable for
    // child processes (tests load slang.dll at runtime) without touching the
    // user's shell profile.
    if let (Ok(sdk), Some(path)) = (std::env::var("VULKAN_SDK"), std::env::var_os("PATH")) {
        let mut paths: Vec<PathBuf> =
            vec![Path::new(&sdk).join("Bin"), Path::new(&sdk).join("bin")];
        paths.extend(std::env::split_paths(&path));
        if let Ok(joined) = std::env::join_paths(paths) {
            cmd.env("PATH", joined);
        }
    }
    cmd
}

pub fn run(cmd: &mut Command, what: &str) -> anyhow::Result<()> {
    println!("xtask: {what}");
    let status = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("failed to spawn `{what}`: {e}"))?;
    anyhow::ensure!(status.success(), "`{what}` failed ({status})");
    Ok(())
}

pub fn run_capture(cmd: &mut Command, what: &str) -> anyhow::Result<String> {
    let out = cmd
        .output()
        .map_err(|e| anyhow::anyhow!("failed to spawn `{what}`: {e}"))?;
    anyhow::ensure!(
        out.status.success(),
        "`{what}` failed ({})\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// All `.rs` files under `dir`, skipping `target/` — fuel for the §3 greps.
pub fn walk_rs(dir: &Path, files: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "target") {
                continue;
            }
            walk_rs(&path, files);
        } else if path.extension().is_some_and(|e| e == "rs") {
            files.push(path);
        }
    }
}
