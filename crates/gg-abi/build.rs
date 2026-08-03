//! Computes [`BOUNDARY_FINGERPRINT`] — the defense-in-depth half of §4.2.2's
//! compatibility check.
//!
//! The load-bearing check is `HOST_API_VERSION`, a number a human bumps. This is
//! the belt to that suspender: a content hash of the boundary crates' source
//! plus the toolchain version, which catches the drift a version number cannot
//! see — a changed component invariant, a resized math type behind an unchanged
//! API, or a `HOST_API_VERSION` somebody forgot to bump.
//!
//! Scope is the whole of the four crates in [`BOUNDARY_CRATES`] plus the
//! workspace-root manifest and lock file, and nothing else. The crate list is
//! §3's game-crate deny pin read as a hash — it must stay character-for-character
//! `xtask`'s `GAME_CRATE_PIN`, or the fingerprint claims coverage the link set
//! does not have. The workspace manifest is in scope because every third-party
//! version in those crates is `{ workspace = true }` and resolved there:
//! repinning `libm` is a Determinism Contract event that moves sim result bits,
//! and hashing only the crate manifests would accept a game dylib built against
//! the old pin. A comment edit in `gg-render` still must not invalidate a running
//! session, for a layout that cannot have changed.
//!
//! Both sides of the seam link `gg-abi`, so both get the constant from whichever
//! `gg-abi` they compiled against — which is what makes comparing them mean
//! anything at all.

use std::path::{Path, PathBuf};

/// The crates the fingerprint covers (§4.2.2) — the same list as `xtask`'s
/// `GAME_CRATE_PIN`, and a divergence between the two is a silent hole rather
/// than a loud one. `gg-ecs-derive` is on it because it *generates* `DECLARED_ID`,
/// `TYPE_NAME` and `FIELDS`: an edit that requalifies a field's type token moves
/// every `schema_hash` while leaving the fingerprint still, and `migrate_fields`
/// then zeroes every field it can no longer type-match — the whole world silently
/// resetting on the next reload.
const BOUNDARY_CRATES: [&str; 4] = ["gg-abi", "gg-ecs", "gg-ecs-derive", "gg-math"];

fn main() {
    let manifest = PathBuf::from(env("CARGO_MANIFEST_DIR"));
    let Some(crates_dir) = manifest.parent() else {
        panic!("gg-abi must live under crates/");
    };
    let Some(workspace) = crates_dir.parent() else {
        panic!("crates/ must live under the workspace root");
    };

    let mut hasher = blake3::Hasher::new();
    // The toolchain first: two builds of identical source under different rustc
    // versions are different artifacts, and `repr(Rust)` layout is exactly the
    // thing a rustc bump is allowed to move.
    hasher.update(rustc_version().as_bytes());

    // Where `{ workspace = true }` resolves: the versions the boundary crates
    // actually compile against are pinned here, not in their own manifests.
    let mut files = vec![workspace.join("Cargo.toml")];
    // And the manifest resolves a *range*: `libm = "0.2"` is 0.2.15 or 0.2.16
    // depending on the lock alone, so a `cargo update` moves sim result bits
    // while leaving every other hashed byte where it was — precisely the
    // Determinism Contract event this fingerprint exists to catch. Hashing the
    // whole file is deliberately over-broad: an unrelated `gg-render` bump now
    // invalidates a running hot-reload session, which errs toward refusing a
    // dylib that would have been fine and never toward accepting one that is not.
    let lock = workspace.join("Cargo.lock");
    assert!(
        lock.is_file(),
        "no Cargo.lock at {} — a skipped input is the hole this hashes it to close",
        lock.display()
    );
    files.push(lock);
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
        // Relative to the workspace root, which is the one prefix every hashed
        // file shares — an absolute path would make the hash the checkout's.
        let rel = file
            .strip_prefix(workspace)
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
