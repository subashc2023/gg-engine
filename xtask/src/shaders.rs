//! `xtask shaders` — the offline path (§4.4), M2 form: compile every `.slang`
//! module under `*/shaders/` **in-process** through gg-shaders (the same
//! compiler the hot path uses), and freeze the results into the tree —
//! `src/shaders_gen/<stem>.rs` (reflection→codegen with layout assertions)
//! plus the sibling `<stem>.<entry>.spv` blobs it `include_bytes!`s. All
//! checked in: dist embeds them, and CI's gate 3 holds them diff-clean
//! against the sources via `--check`.
//!
//! Structured per-target from day one: the one-variant enum below is the
//! console door (§2, Console portability row). Content-hashed incremental:
//! the generated header carries the source hash + generator version, and a
//! matching header skips the recompile.

use crate::util::{sha256_hex, workspace_root};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
pub enum Target {
    Spirv,
}

impl Target {
    /// File suffix for this target's compiled blobs.
    fn blob_extension(self) -> &'static str {
        match self {
            Target::Spirv => "spv",
        }
    }
}

const TARGETS: &[Target] = &[Target::Spirv];

/// Assert the linked Slang matches this host's row in `slang-pin.toml` (§4.4).
///
/// `shader-slang` pins no Slang version — bindgen runs against whatever header
/// the machine has and links the installed dylib — so the compiler behind our
/// reflection and SPIR-V is machine state, invisible to Cargo.lock. Gate 3
/// requires the generated Rust to be byte-identical across hosts; that is only
/// a real gate if the compiler producing it is pinned. Without this, a Slang
/// upgrade on one lane turns gate 3 red with no attributable cause, or changes
/// a reflected offset that both lanes then quietly agree on.
pub fn assert_slang_pin() -> anyhow::Result<()> {
    let root = workspace_root();
    let pin_path = root.join("slang-pin.toml");
    let pins: toml::Value = toml::from_str(&std::fs::read_to_string(&pin_path)?)?;

    // Host key by OS family, matching how the two lanes are provisioned: the
    // Windows host takes Slang from the LunarG SDK, the WSL lane from SLANG_DIR.
    let host = if cfg!(windows) { "windows" } else { "linux" };
    let expected = pins
        .get("hosts")
        .and_then(|h| h.get(host))
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "slang-pin.toml has no [hosts].{host} row — every host that compiles shaders \
                 needs a pinned Slang version (§4.4)"
            )
        })?;

    let actual = gg_shaders::slang_build_tag()
        .map_err(|e| anyhow::anyhow!("could not ask Slang for its build tag: {e}"))?;

    anyhow::ensure!(
        actual == expected,
        "Slang version drift on `{host}`: linked {actual}, pinned {expected} (slang-pin.toml).\n\
         The generated shader Rust must be byte-identical across hosts, so the compiler that \
         produces it is pinned. If this upgrade is intended: re-run `cargo xtask shaders`, read \
         the diff (an empty one means the upgrade was layout-neutral), then move the row."
    );
    println!("xtask: slang {actual} matches the {host} pin (§4.4)");
    Ok(())
}

pub fn build_all(check: bool) -> anyhow::Result<()> {
    assert_slang_pin()?;
    let root = workspace_root();
    let mut modules = Vec::new();
    for base in ["crates", "demos"] {
        find_slang(&root.join(base), &mut modules);
    }
    if modules.is_empty() {
        println!("xtask shaders: no .slang modules in the tree");
        return Ok(());
    }
    modules.sort(); // deterministic order, deterministic logs

    let mut built = 0usize;
    let mut skipped = 0usize;
    for module in &modules {
        if process_module(module, check)? {
            built += 1;
        } else {
            skipped += 1;
        }
    }
    println!(
        "xtask shaders{}: {built} module(s) {}, {skipped} up-to-date",
        if check { " --check" } else { "" },
        if check { "verified" } else { "built" },
    );
    Ok(())
}

/// Compile one module and write (or, under `check`, verify) its artifacts.
/// Returns false when the source hash already matched and nothing was done.
fn process_module(module: &Path, check: bool) -> anyhow::Result<bool> {
    let stem = module
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .ok_or_else(|| anyhow::anyhow!("shader path has no stem: {}", module.display()))?;
    let shaders_dir = module
        .parent()
        .ok_or_else(|| anyhow::anyhow!("shader has no parent dir: {}", module.display()))?;
    // Convention (§4.4): <crate>/shaders/<stem>.slang → <crate>/src/shaders_gen/.
    let crate_root = shaders_dir
        .parent()
        .ok_or_else(|| anyhow::anyhow!("shaders dir has no parent: {}", shaders_dir.display()))?;
    let gen_dir = crate_root.join("src/shaders_gen");
    let rs_path = gen_dir.join(format!("{stem}.rs"));

    // Hashed and byte-compared exactly as checked out — no line-ending
    // normalization here on purpose. `.gitattributes` pins every checkout to LF
    // (§5, Filesystem hygiene), so a CRLF file reaching this point means the
    // repository itself has one, and that is worth failing over rather than
    // papering over. The weekly fresh-clone gate found this the hard way on its
    // first run: without `.gitattributes`, a Windows clone got CRLF and every
    // generated artifact read as stale in a tree nobody had touched.
    let mut source = std::fs::read(module)?;
    // Anything under `shaders/include/` is `#include`d rather than compiled on
    // its own (`find_slang` skips it — the parent directory is not `shaders`),
    // so it has to enter the hash here or editing the shared BRDF would leave
    // every module's artifacts stale with a header that still matched. Absent
    // include directory appends nothing, which is why every pre-M11 shader keeps
    // the hash it had.
    for included in includes(&shaders_dir.join("include"))? {
        source.extend_from_slice(&std::fs::read(&included)?);
    }
    let hash = sha256_hex(&source);
    let header = gg_shaders::codegen::header_line(&hash);

    // Incremental (build mode only): an on-disk header matching source hash +
    // generator version means the artifacts are current. Check mode never
    // trusts the header — it recompiles and compares bytes, so a hand-edited
    // generated file cannot ride a matching hash through the gate.
    if !check
        && let Ok(existing) = std::fs::read_to_string(&rs_path)
        && existing.lines().nth(1) == Some(header.as_str())
    {
        return Ok(false);
    }

    let compiled =
        gg_shaders::compile_module(&shaders_dir.to_string_lossy(), &format!("{stem}.slang"))
            .map_err(|e| anyhow::anyhow!("{}: {e}", module.display()))?;
    anyhow::ensure!(
        !compiled.entry_points.is_empty(),
        "{}: no [shader(...)] entry points",
        module.display()
    );
    let generated = gg_shaders::codegen::generate(&stem, &compiled, &hash)
        .map_err(|e| anyhow::anyhow!("{}: {e}", module.display()))?;

    if check {
        // Gate 3 (§5): reflection codegen diff-clean. The .rs must match the
        // regeneration byte for byte. The .spv blobs are validated for
        // presence + SPIR-V magic, not bytes: the two hosts pin different
        // Slang builds, and the codegen layout facts are the cross-host truth.
        let on_disk = std::fs::read_to_string(&rs_path).map_err(|_| {
            anyhow::anyhow!(
                "{}: missing — run `cargo xtask shaders` and commit (§5 gate 3)",
                rs_path.display()
            )
        })?;
        anyhow::ensure!(
            on_disk == generated,
            "{}: stale against {} — run `cargo xtask shaders` and commit (§5 gate 3: \
             reflection codegen diff-clean)",
            rs_path.display(),
            module.display()
        );
        for ep in &compiled.entry_points {
            let blob = gen_dir.join(format!("{stem}.{}.spv", ep.name));
            let bytes = std::fs::read(&blob)
                .map_err(|_| anyhow::anyhow!("{}: missing (§5 gate 3)", blob.display()))?;
            anyhow::ensure!(
                bytes.len() >= 4 && bytes[0..4] == [0x03, 0x02, 0x23, 0x07],
                "{}: not SPIR-V (§5 gate 3)",
                blob.display()
            );
        }
        return Ok(true);
    }

    std::fs::create_dir_all(&gen_dir)?;
    for target in TARGETS {
        for ep in &compiled.entry_points {
            let blob = gen_dir.join(format!("{stem}.{}.{}", ep.name, target.blob_extension()));
            std::fs::write(&blob, &ep.spirv)?;
        }
    }
    std::fs::write(&rs_path, &generated)?;
    println!(
        "xtask: shaders/{stem}.slang -> {} ({} entry points{})",
        rs_path.display(),
        compiled.entry_points.len(),
        if compiled.push_constants.is_some() {
            ", push constants"
        } else {
            ""
        }
    );
    Ok(true)
}

/// Every `.slang` under an `include/` directory, sorted. Sorted because the
/// concatenation is hashed and a directory listing's order is the filesystem's.
fn includes(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(Vec::new());
    };
    let mut found: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "slang"))
        .collect();
    found.sort();
    Ok(found)
}

/// All `<crate>/shaders/*.slang` modules. `fixtures/` trees are compiler test
/// data (compiled by gg-shaders' own tests), not engine shaders.
fn find_slang(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path
                .file_name()
                .is_some_and(|n| n == "target" || n == "fixtures")
            {
                continue;
            }
            find_slang(&path, out);
        } else if path.extension().is_some_and(|e| e == "slang")
            && path
                .parent()
                .and_then(Path::file_name)
                .is_some_and(|n| n == "shaders")
        {
            out.push(path);
        }
    }
}
