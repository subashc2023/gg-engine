//! Computes [`BOUNDARY_FINGERPRINT`] — the defense-in-depth half of §4.2.2's
//! compatibility check.
//!
//! The load-bearing check is `HOST_API_VERSION`, a number a human bumps. This is
//! the belt to that suspender: a content hash of the boundary crates' source
//! plus the toolchain version, which catches the drift a version number cannot
//! see — a changed component invariant, a resized math type behind an unchanged
//! API, or a `HOST_API_VERSION` somebody forgot to bump.
//!
//! Scope is the whole of `gg-abi`, `gg-ecs` and `gg-math` and nothing else,
//! which is §3's game-crate deny pin read as a hash: a game dylib may link those
//! three engine crates and no others, so the fingerprint's scope and the dylib's
//! possible link set are the same list by construction. A comment edit in
//! `gg-render` must not invalidate a running session for a layout that cannot
//! have changed.
//!
//! Both sides of the seam link `gg-abi`, so both get the constant from whichever
//! `gg-abi` they compiled against — which is what makes comparing them mean
//! anything at all.

use std::path::{Path, PathBuf};

/// The crates the fingerprint covers (§4.2.2). Adding one here without adding it
/// to `deny.toml`'s game-crate pin would make the hash claim coverage the link
/// set does not have.
const BOUNDARY_CRATES: [&str; 3] = ["gg-abi", "gg-ecs", "gg-math"];

fn main() {
    let manifest = PathBuf::from(env("CARGO_MANIFEST_DIR"));
    let Some(crates_dir) = manifest.parent() else {
        panic!("gg-abi must live under crates/");
    };

    let mut hasher = blake3::Hasher::new();
    // The toolchain first: two builds of identical source under different rustc
    // versions are different artifacts, and `repr(Rust)` layout is exactly the
    // thing a rustc bump is allowed to move.
    hasher.update(rustc_version().as_bytes());

    let mut files = Vec::new();
    for name in BOUNDARY_CRATES {
        let root = crates_dir.join(name);
        assert!(
            root.is_dir(),
            "boundary crate `{name}` is missing at {}",
            root.display()
        );
        files.push(root.join("Cargo.toml"));
        collect_rs(&root.join("src"), &mut files);
        // A directory watch catches files *added*, which a per-file list cannot.
        println!("cargo:rerun-if-changed={}", root.join("src").display());
    }
    // Sorted so the hash is a property of the source, not of directory order.
    files.sort();

    for file in &files {
        println!("cargo:rerun-if-changed={}", file.display());
        let rel = file
            .strip_prefix(crates_dir)
            .unwrap_or(file)
            .to_string_lossy()
            // Path separators are not content: the same tree must hash the same
            // whichever host is reading it.
            .replace('\\', "/");
        hasher.update(rel.as_bytes());
        let bytes = std::fs::read(file).unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }

    let digest = hasher.finalize();
    let bytes = digest
        .as_bytes()
        .iter()
        .map(|b| b.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let out = PathBuf::from(env("OUT_DIR")).join("fingerprint.rs");
    std::fs::write(
        &out,
        format!(
            "/// Content hash of the boundary crates plus the toolchain version \
             (§4.2.2).\n\
             ///\n\
             /// Defense-in-depth behind [`HOST_API_VERSION`], never the load-bearing \
             /// check. A dylib reporting a different one was built against a different \
             /// boundary, whatever its version number says.\n\
             pub const BOUNDARY_FINGERPRINT: [u8; FINGERPRINT_BYTES] = [{bytes}];\n\
             \n\
             /// The fingerprint as hex, for error messages and log lines.\n\
             pub const BOUNDARY_FINGERPRINT_HEX: &str = \"{digest}\";\n"
        ),
    )
    .unwrap_or_else(|e| panic!("write {}: {e}", out.display()));
}

fn collect_rs(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            println!("cargo:rerun-if-changed={}", path.display());
            collect_rs(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn rustc_version() -> String {
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let output = std::process::Command::new(rustc)
        .arg("-V")
        .output()
        .unwrap_or_else(|e| panic!("running rustc -V: {e}"));
    String::from_utf8_lossy(&output.stdout).trim().to_owned()
}

fn env(key: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| panic!("{key} is not set"))
}
