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

/// Drop ANSI colour sequences. The `tracing` subscriber writes them whether or
/// not it is talking to a terminal, and every gate that reads a child process's
/// log has to get past them first.
pub fn plain(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' {
            for c in chars.by_ref() {
                if c == 'm' {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// What the shell logs once its watcher exists (`app.rs`, after `Watch::new`).
///
/// Every gate that rewrites a dylib under a running shell waits for this line
/// first — a file event with nobody listening is not late, it is *gone*, which
/// was the push tier's one flaky gate until M14's correction retired the sleep.
pub const READY: &str = "game loaded";

/// Read a child stream to EOF on its own thread, signalling the first time a
/// line contains `marker`, and hand back the whole thing ANSI-stripped.
///
/// Per line rather than at the end, because `tracing` writes escapes *inside* a
/// record — a marker matched against the raw bytes would work until someone
/// coloured the message.
pub fn drain<R: std::io::Read + Send + 'static>(
    stream: Option<R>,
    marker: String,
    tx: std::sync::mpsc::Sender<()>,
) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        use std::io::BufRead as _;
        let Some(stream) = stream else {
            return String::new();
        };
        let mut log = String::new();
        let mut seen = false;
        for line in std::io::BufReader::new(stream)
            .lines()
            .map_while(Result::ok)
        {
            let line = plain(line.as_bytes());
            if !seen && line.contains(&marker) {
                seen = true;
                let _ = tx.send(()); // the other stream may have got there first
            }
            log.push_str(&line);
            log.push('\n');
        }
        log
    })
}

/// A `tracing` field's value, as text up to the next space. Run [`plain`] first.
pub fn field<'a>(line: &'a str, name: &str) -> anyhow::Result<&'a str> {
    let at = line
        .find(&format!("{name}="))
        .ok_or_else(|| anyhow::anyhow!("no `{name}=` in `{line}`"))?
        + name.len()
        + 1;
    Ok(line[at..].split_whitespace().next().unwrap_or(""))
}

/// The same, parsed as a number.
pub fn field_u64(line: &str, name: &str) -> anyhow::Result<u64> {
    let text = field(line, name)?;
    let digits: String = text.chars().take_while(char::is_ascii_digit).collect();
    Ok(digits.parse()?)
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
