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

/// Attach a **static** CRT to every build of the shipping profile, on the one
/// argument that makes a build a shipping one (§6 M53).
///
/// It is here rather than at the twelve call sites that spell `--profile dist`
/// because forgetting it is silent and expensive: a shipped `.exe` linked
/// against the dynamic CRT imports `VCRUNTIME140.dll`, which is not a component
/// of Windows. The loader resolves that before the first instruction of ours
/// runs, so §6 M47's message box never opens and no `log.txt` is written — the
/// player reads an OS error naming a file they have never heard of.
///
/// **Windows only.** A statically linked glibc cannot `dlopen`, and the game
/// arrives through `dlopen` in every tier (§4.2.2), so the Linux answer is a
/// version floor held in `dist-deps.txt` instead.
///
/// Two consequences worth knowing rather than discovering. The flag is part of
/// cargo's fingerprint, so a `cargo build --profile dist` typed by hand rebuilds
/// against this one — go through `xtask` and it does not. And with no `--target`
/// on the command line the flag reaches host artifacts too, so proc macros are
/// built against the static CRT as well; rustc loads them regardless (measured,
/// not assumed), and passing `--target` to avoid it would move every dist
/// artifact under a triple directory that four other gates spell by hand.
fn static_crt(cmd: &mut Command) {
    if !cfg!(target_env = "msvc") {
        return;
    }
    let args: Vec<&std::ffi::OsStr> = cmd.get_args().collect();
    if !args
        .windows(2)
        .any(|w| w[0] == "--profile" && w[1] == "dist")
    {
        return;
    }
    // Appended rather than assigned: RUSTFLAGS in the environment replaces
    // config's `rustflags` wholesale, so clobbering an operator's would silently
    // drop whatever they were measuring.
    let mut flags = std::env::var("RUSTFLAGS").unwrap_or_default();
    if !flags.is_empty() {
        flags.push(' ');
    }
    flags.push_str("-C target-feature=+crt-static");
    cmd.env("RUSTFLAGS", flags);
}

pub fn run(cmd: &mut Command, what: &str) -> anyhow::Result<()> {
    println!("xtask: {what}");
    static_crt(cmd);
    let status = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("failed to spawn `{what}`: {e}"))?;
    anyhow::ensure!(status.success(), "`{what}` failed ({status})");
    Ok(())
}

pub fn run_capture(cmd: &mut Command, what: &str) -> anyhow::Result<String> {
    static_crt(cmd);
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

/// A cdylib's file name as cargo spells it on this host: `<stem>.dll` on
/// Windows, `lib<stem>.dylib` on macOS, `lib<stem>.so` elsewhere.
///
/// `stem` is the *library* name (underscores), not the package name — the two
/// differ for every `demo-NN-name` crate. One home because this was written
/// twice and the copies had already drifted apart on macOS.
pub fn dylib_name(stem: &str) -> String {
    if cfg!(windows) {
        format!("{stem}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else {
        format!("lib{stem}.so")
    }
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
