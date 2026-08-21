//! `cargo xtask deps` — what a shipped artifact demands of the machine it lands
//! on, held against a checked-in baseline (§6 M53).
//!
//! # The gate every other gate cannot be
//!
//! Every check in §5 runs on a machine that just built the thing it is checking,
//! so a dependency the *build host* happens to have is invisible to all of them:
//! the artifact runs here because here is where it was made. What a player has
//! is a different set, and the failure mode is the worst shape available —
//! the loader resolves imports before the first instruction of ours executes, so
//! §6 M47's message box never opens, no `log.txt` is written, and the player
//! reads an OS error naming a file they have never heard of. It is the one
//! refusal this tree cannot speak into, which is why it is answered by removing
//! the dependency (Windows) or by pinning its floor in a file someone reviews
//! (Linux) rather than by a better message.
//!
//! # What a name in the baseline is claiming
//!
//! One standard, and it is not "this exists on my machine": every entry must be
//! either **a component of the operating system itself** — present on a clean
//! install of the oldest release we claim — or **a file `xtask ship` stages into
//! the folder**. Nothing else may appear. `VCRUNTIME140.dll` is the example the
//! standard was written against: it is on virtually every gaming PC because
//! something else installed it, and on none of the machines that matter for a
//! bug report.
//!
//! # What it cannot see
//!
//! A library loaded at *runtime* is not an import: `vulkan-1.dll` /
//! `libvulkan.so.1` arrive through `LoadLibrary`/`dlopen` inside `ash`, and the
//! game dylib itself arrives the same way (§4.2.2). Their absence is a refusal
//! this tree *can* speak into and already does (§6 M47), which is the division
//! of labour — this gate covers exactly the half where our code never runs.
//!
//! Usage:
//!   cargo xtask deps [--bless]

use anyhow::Context as _;

use crate::util::{cargo, run as exec, workspace_root};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The reviewed list, beside the shell whose artifact it describes.
fn baseline_path() -> PathBuf {
    workspace_root().join("crates/gg-runtime/dist-deps.txt")
}

/// One library an artifact needs supplied, with the newest interface version it
/// asks that library for — `None` where the format has no such concept, which
/// is every PE import and most ELF ones.
///
/// The version is the half that decides *which* Linux runs the download: a
/// symbol versioned `GLIBC_2.39` is refused outright by an older `ld.so`, so
/// this number is the floor under the distro list, not a detail.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Need {
    pub lib: String,
    pub version: Option<String>,
}

impl std::fmt::Display for Need {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.version {
            Some(v) => write!(f, "{} {v}", self.lib),
            None => write!(f, "{}", self.lib),
        }
    }
}

/// Read one artifact's dependency surface, dispatching on the file's own magic
/// rather than on `cfg!` — the WSL lane reads Windows artifacts out of `/mnt/c`
/// when a question comes up, and a parser keyed on the host would refuse them.
pub fn surface(path: &Path) -> anyhow::Result<Vec<Need>> {
    let bytes =
        std::fs::read(path).map_err(|e| anyhow::anyhow!("cannot read {} : {e}", path.display()))?;
    let mut needs = if bytes.starts_with(b"MZ") {
        pe_imports(&bytes)
    } else if bytes.starts_with(b"\x7fELF") {
        elf_needed(&bytes)
    } else {
        anyhow::bail!("{} is neither PE nor ELF", path.display())
    }
    .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    needs.sort();
    needs.dedup();
    Ok(needs)
}

// ---------------------------------------------------------------- PE (Windows)

fn u16le(b: &[u8], at: usize) -> anyhow::Result<u16> {
    let s = b
        .get(at..at + 2)
        .ok_or_else(|| anyhow::anyhow!("short read at {at:#x}"))?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}

fn u32le(b: &[u8], at: usize) -> anyhow::Result<u32> {
    let s = b
        .get(at..at + 4)
        .ok_or_else(|| anyhow::anyhow!("short read at {at:#x}"))?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn u64le(b: &[u8], at: usize) -> anyhow::Result<u64> {
    let s = b
        .get(at..at + 8)
        .ok_or_else(|| anyhow::anyhow!("short read at {at:#x}"))?;
    Ok(u64::from_le_bytes([
        s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
    ]))
}

/// NUL-terminated ASCII at a file offset.
fn cstr(b: &[u8], at: usize) -> anyhow::Result<String> {
    let rest = b
        .get(at..)
        .ok_or_else(|| anyhow::anyhow!("string past EOF at {at:#x}"))?;
    let end = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
    Ok(String::from_utf8_lossy(&rest[..end]).into_owned())
}

/// The PE import directory: every DLL the loader must resolve before `main`.
///
/// Delay-loaded imports (directory 13) are deliberately *not* read: those fail
/// at the call rather than at load, which is a refusal our own code is present
/// to handle. Nothing in this tree delay-loads today, and a first one appearing
/// is a decision, not a regression this gate should silently absorb.
fn pe_imports(b: &[u8]) -> anyhow::Result<Vec<Need>> {
    let pe = u32le(b, 0x3c)? as usize;
    anyhow::ensure!(
        b.get(pe..pe + 4) == Some(b"PE\0\0"),
        "no PE signature at e_lfanew"
    );
    let sections = u16le(b, pe + 6)? as usize;
    let opt_size = u16le(b, pe + 20)? as usize;
    let opt = pe + 24;
    // PE32+ (0x20b) puts the data directories 16 bytes later than PE32; we build
    // x86-64 only, but reading the magic is cheaper than assuming it.
    let dirs = opt + if u16le(b, opt)? == 0x20b { 112 } else { 96 };
    let import_rva = u32le(b, dirs + 8)?;
    if import_rva == 0 {
        return Ok(Vec::new()); // no imports at all — a resource-only DLL
    }

    // Section table, for the one thing every RVA in the file needs: a file offset.
    let mut map = Vec::with_capacity(sections);
    for i in 0..sections {
        let s = opt + opt_size + i * 40;
        map.push((
            u32le(b, s + 12)?,
            u32le(b, s + 8)?,
            u32le(b, s + 20)?,
            u32le(b, s + 16)?,
        ));
    }
    let offset = |rva: u32| -> Option<usize> {
        map.iter().find_map(|&(va, vsz, raw, rsz)| {
            (rva >= va && rva < va + vsz.max(rsz)).then(|| (raw + (rva - va)) as usize)
        })
    };

    let mut out = Vec::new();
    let mut at = offset(import_rva).ok_or_else(|| anyhow::anyhow!("import RVA in no section"))?;
    loop {
        // IMAGE_IMPORT_DESCRIPTOR: the terminator is an all-zero entry, and the
        // name RVA is the field worth testing — a descriptor with no name is it.
        let name_rva = u32le(b, at + 12)?;
        if name_rva == 0 {
            break;
        }
        let name = offset(name_rva).ok_or_else(|| anyhow::anyhow!("import name in no section"))?;
        // Lowercased, because the loader is: the shell's table says `kernel32.dll`
        // and the game dylib's says `KERNEL32.dll`, and a baseline holding both
        // would make a linker's capitalisation look like a dependency event. ELF
        // names are left alone — there the difference is real.
        out.push(Need {
            lib: cstr(b, name)?.to_lowercase(),
            version: None,
        });
        at += 20;
    }
    Ok(out)
}

// ------------------------------------------------------------------ ELF (Linux)

const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const DT_NULL: u64 = 0;
const DT_NEEDED: u64 = 1;
const DT_STRTAB: u64 = 5;
const DT_VERNEED: u64 = 0x6fff_fffe;
const DT_VERNEEDNUM: u64 = 0x6fff_ffff;

/// `DT_NEEDED` plus the newest symbol version demanded of each library.
///
/// Both halves come out of the dynamic segment, which is why this reads program
/// headers rather than sections: a stripped artifact is exactly the case that
/// matters and `strip` is free to discard the section table.
fn elf_needed(b: &[u8]) -> anyhow::Result<Vec<Need>> {
    anyhow::ensure!(b.get(4) == Some(&2), "not ELF64");
    anyhow::ensure!(b.get(5) == Some(&1), "not little-endian ELF");
    let phoff = u64le(b, 0x20)? as usize;
    let phentsize = u16le(b, 0x36)? as usize;
    let phnum = u16le(b, 0x38)? as usize;

    let mut loads = Vec::new();
    let mut dynamic = None;
    for i in 0..phnum {
        let p = phoff + i * phentsize;
        let (kind, off, vaddr, filesz) = (
            u32le(b, p)?,
            u64le(b, p + 8)?,
            u64le(b, p + 16)?,
            u64le(b, p + 32)?,
        );
        match kind {
            PT_LOAD => loads.push((vaddr, filesz, off)),
            PT_DYNAMIC => dynamic = Some((off as usize, filesz as usize)),
            _ => {}
        }
    }
    let file_at = |vaddr: u64| -> Option<usize> {
        loads.iter().find_map(|&(va, sz, off)| {
            (vaddr >= va && vaddr < va + sz).then(|| (off + (vaddr - va)) as usize)
        })
    };
    let (dyn_off, dyn_size) = dynamic.ok_or_else(|| anyhow::anyhow!("no PT_DYNAMIC"))?;

    let mut needed_offsets = Vec::new();
    let (mut strtab, mut verneed, mut verneed_num) = (None, None, 0u64);
    for i in 0..dyn_size / 16 {
        let e = dyn_off + i * 16;
        let (tag, val) = (u64le(b, e)?, u64le(b, e + 8)?);
        match tag {
            DT_NULL => break,
            DT_NEEDED => needed_offsets.push(val as usize),
            DT_STRTAB => strtab = Some(val),
            DT_VERNEED => verneed = Some(val),
            DT_VERNEEDNUM => verneed_num = val,
            _ => {}
        }
    }
    let strtab = file_at(strtab.ok_or_else(|| anyhow::anyhow!("no DT_STRTAB"))?)
        .ok_or_else(|| anyhow::anyhow!("DT_STRTAB outside every PT_LOAD"))?;

    // Newest version demanded per library, which is the floor the loader
    // enforces: an `ld.so` whose `libc.so.6` defines no `GLIBC_2.39` refuses the
    // process outright rather than binding the symbol to null.
    let mut floors: BTreeMap<String, String> = BTreeMap::new();
    if let Some(vn) = verneed.and_then(file_at) {
        let mut at = vn;
        for _ in 0..verneed_num {
            let (cnt, file, aux, next) = (
                u16le(b, at + 2)? as usize,
                u32le(b, at + 4)? as usize,
                u32le(b, at + 8)? as usize,
                u32le(b, at + 12)? as usize,
            );
            let lib = cstr(b, strtab + file)?;
            let mut a = at + aux;
            for _ in 0..cnt {
                let name = cstr(b, strtab + u32le(b, a + 8)? as usize)?;
                floors
                    .entry(lib.clone())
                    .and_modify(|cur| {
                        if newer(&name, cur) {
                            *cur = name.clone();
                        }
                    })
                    .or_insert(name);
                let step = u32le(b, a + 12)? as usize;
                if step == 0 {
                    break;
                }
                a += step;
            }
            if next == 0 {
                break;
            }
            at += next;
        }
    }

    needed_offsets
        .into_iter()
        .map(|o| {
            let lib = cstr(b, strtab + o)?;
            let version = floors.get(&lib).cloned();
            Ok(Need { lib, version })
        })
        .collect()
}

/// Symbol-version ordering: `GLIBC_2.39` is newer than `GLIBC_2.4`, which plain
/// string comparison gets backwards — the whole point of this file is a number
/// that decides which distros can run the download.
fn newer(a: &str, b: &str) -> bool {
    let parts = |s: &str| -> (String, Vec<u64>) {
        let (prefix, tail) = s.rsplit_once('_').unwrap_or((s, ""));
        (
            prefix.to_string(),
            tail.split('.').filter_map(|n| n.parse().ok()).collect(),
        )
    };
    let (pa, na) = parts(a);
    let (pb, nb) = parts(b);
    // Different families (`GLIBC_` vs `GCC_`) are not comparable; keeping the
    // first seen is arbitrary but stable, and no shipped library carries two.
    if pa != pb {
        return false;
    }
    na > nb
}

// ------------------------------------------------------------------- the gate

/// The shipping set: the shell every player launches, plus every game dylib —
/// the dylib matters as much as the exe, since the loader resolves *its* imports
/// too and a game that cannot load is a game that cannot start.
fn artifacts() -> anyhow::Result<Vec<(String, PathBuf)>> {
    let dist = workspace_root().join("target/dist");
    let mut out = vec![(
        "gg-runtime".to_string(),
        dist.join(if cfg!(windows) {
            "gg-runtime.exe"
        } else {
            "gg-runtime"
        }),
    )];
    for (game, _dir) in crate::budgets::game_crate_dirs()? {
        out.push((
            game.clone(),
            dist.join(crate::util::dylib_name(&game.replace('-', "_"))),
        ));
    }
    Ok(out)
}

/// Build what the gate reads, so `xtask deps` stands alone. Inside `xtask dist`
/// these are no-ops: the artifacts are already there and cargo says so.
fn build() -> anyhow::Result<()> {
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
    for (game, _dir) in crate::budgets::game_crate_dirs()? {
        exec(
            cargo().args(["build", "-p", &game, "--profile", "dist"]),
            &format!("build {game} [dist profile]"),
        )?;
    }
    Ok(())
}

/// The union across the shipping set, which is what a player's machine has to
/// satisfy — a dependency only the dylib carries fails just as hard.
///
/// Folded to the *newest* version each library is asked for anywhere in the set,
/// because that is the floor: the shell wants `GLIBC_2.39` and a game dylib wants
/// only `2.34`, and a machine that satisfies the second and not the first runs
/// neither. Listing both would read as two dependencies and hide the number.
fn measured() -> anyhow::Result<Vec<Need>> {
    let mut floors: BTreeMap<String, Option<String>> = BTreeMap::new();
    for (_what, path) in artifacts()? {
        for need in surface(&path)? {
            let slot = floors.entry(need.lib).or_default();
            match (&slot, &need.version) {
                (None, Some(_)) => *slot = need.version,
                (Some(cur), Some(new)) if newer(new, cur) => *slot = need.version,
                _ => {}
            }
        }
    }
    Ok(floors
        .into_iter()
        .map(|(lib, version)| Need { lib, version })
        .collect())
}

/// Which host's section of the baseline this run is reading. The file holds all
/// of them, because the question "which machines can run the download" has a
/// different answer per platform and both belong in one review.
fn host_key() -> &'static str {
    if cfg!(windows) {
        "x86_64-pc-windows-msvc"
    } else {
        "x86_64-unknown-linux-gnu"
    }
}

/// Parse the baseline into its per-host sections, preserving nothing else — the
/// file is generated, and its prose lives in this module instead.
fn read_baseline(text: &str) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut section = String::new();
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            section = name.to_string();
            out.entry(section.clone()).or_default();
        } else if !section.is_empty() {
            out.entry(section.clone())
                .or_default()
                .push(line.to_string());
        }
    }
    out
}

fn write_baseline(sections: &BTreeMap<String, Vec<String>>) -> String {
    let mut out = String::from(
        "# What a shipped artifact demands of the machine it lands on (§6 M53).\n\
         #\n\
         # Every name here is either a component of the operating system itself or a file\n\
         # `xtask ship` stages beside the game. Nothing else may appear: a library that is\n\
         # merely *usually installed* fails in the loader, before our code runs, where no\n\
         # message of ours can reach the player (§6 M47).\n\
         #\n\
         # A version is the floor that library's interface is asked for, and it decides which\n\
         # releases can run the download at all.\n\
         #\n\
         # Rewritten by `cargo xtask deps --bless`; the diff belongs in the review.\n",
    );
    for (host, needs) in sections {
        out.push_str(&format!("\n[{host}]\n"));
        for need in needs {
            out.push_str(need);
            out.push('\n');
        }
    }
    out
}

/// Compare the shipping set against the baseline. Called standalone and from
/// `xtask dist`, where the artifacts have just been built.
pub fn check() -> anyhow::Result<()> {
    let needs: Vec<String> = measured()?.iter().map(Need::to_string).collect();
    let path = baseline_path();
    let text = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("no dependency baseline at {}: {e}", path.display()))?;
    let baseline = read_baseline(&text);
    let expected = baseline.get(host_key()).ok_or_else(|| {
        anyhow::anyhow!(
            "the dependency baseline has no [{}] section — `cargo xtask deps --bless` on that \
             host writes it (§6 M53)",
            host_key()
        )
    })?;

    let (before, after): (std::collections::BTreeSet<_>, std::collections::BTreeSet<_>) = (
        expected.iter().map(String::as_str).collect(),
        needs.iter().map(String::as_str).collect(),
    );
    if before == after {
        println!(
            "xtask deps: the shipping set demands {} libraries, all of them in the baseline (§6 M53)",
            needs.len()
        );
        return Ok(());
    }
    let mut diff = String::new();
    for line in after.difference(&before) {
        diff.push_str(&format!("  + {line}\n"));
    }
    for line in before.difference(&after) {
        diff.push_str(&format!("  - {line}\n"));
    }
    anyhow::bail!(
        "the shipped artifacts' dependency surface changed (§6 M53)\n{diff}\nA `+` line is a \
         machine that could run the game yesterday and cannot today, and the loader says so \
         before any code of ours runs. If it is genuinely part of the OS — or a file `xtask \
         ship` stages — `cargo xtask deps --bless` records it."
    )
}

/// Rewrite this host's section, leaving the other host's alone: neither machine
/// can measure the other, and a bless that dropped the absent half would make
/// the two lanes overwrite each other forever.
pub fn bless() -> anyhow::Result<()> {
    build()?;
    let path = baseline_path();
    let mut sections = std::fs::read_to_string(&path)
        .map(|t| read_baseline(&t))
        .unwrap_or_default();
    sections.insert(
        host_key().to_string(),
        measured()?.iter().map(Need::to_string).collect(),
    );
    std::fs::write(&path, write_baseline(&sections))?;
    println!("xtask deps: blessed [{}] in {}", host_key(), path.display());
    Ok(())
}

/// The report: per artifact, what it demands. This is the half that answers a
/// question rather than passing or failing, and it is here rather than in
/// `gg-tools` because the measurement *is* the threshold — there is no plateau
/// to find and no picture to read, only a list that is or is not a subset.
pub fn report() -> anyhow::Result<()> {
    build()?;
    println!(
        "\n  the shipping set, and what each artifact needs supplied ({})\n",
        host_key()
    );
    for (what, path) in artifacts()? {
        let needs = surface(&path)?;
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        println!("{what}  ({} KiB)", size / 1024);
        for need in &needs {
            println!("    {need}");
        }
        println!();
    }
    Ok(())
}

pub fn run(args: &[&str]) -> anyhow::Result<()> {
    if args.contains(&"--bless") {
        bless()
    } else if args.contains(&"--check") {
        build()?;
        check()
    } else {
        report()
    }
}

/// `xtask ship`'s own reading: the folder as assembled, which is the one place
/// the *staged* files are visible — a dependency satisfied by a neighbour in the
/// zip is fine, and a dependency satisfied by nothing is the bug this catches.
pub fn check_folder(folder: &Path) -> anyhow::Result<()> {
    let staged: std::collections::BTreeSet<String> = std::fs::read_dir(folder)?
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().to_lowercase())
        .collect();
    let text = std::fs::read_to_string(baseline_path())?;
    let baseline = read_baseline(&text);
    let allowed: std::collections::BTreeSet<String> = baseline
        .get(host_key())
        .map(|v| {
            v.iter()
                .map(|l| l.split(' ').next().unwrap_or(l).to_lowercase())
                .collect()
        })
        .unwrap_or_default();

    let mut read = 0usize;
    for entry in std::fs::read_dir(folder)?.filter_map(Result::ok) {
        let path = entry.path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !path.is_file() || (!matches!(ext, "exe" | "dll" | "so") && path.extension().is_some()) {
            continue;
        }
        // Propagated, never skipped (§6 M81). This gate's whole subject is what a
        // stranger's loader will demand, and the artifact it cannot parse is the
        // one most likely to be malformed — swallowing the error passed the
        // folder by *reading nothing in it*.
        let needs = surface(&path).with_context(|| {
            format!(
                "cannot read the dependency surface of {} — the one artifact this gate could not \
                 read is not the one to wave through (§6 M53)",
                path.display()
            )
        })?;
        read += 1;
        for need in needs {
            let lower = need.lib.to_lowercase();
            anyhow::ensure!(
                allowed.contains(&lower) || staged.contains(&lower),
                "{} needs `{}`, which is neither part of the OS baseline nor staged in the \
                 folder (§6 M53) — a player without it gets a loader error before the game \
                 says anything",
                path.display(),
                need.lib
            );
        }
    }
    // The vacuity guard §6 M87 made a roster of — this comment used to claim
    // *every* "find and check" gate here carried one, and nine did against seven
    // that did not. Two rather than one: a folder whose executable was renamed
    // past the extension filter satisfies the loop above by holding nothing to
    // check, and a shipped folder holds at least the shell and the game dylib.
    crate::census::graded(
        read,
        "the shipped folder's artifacts",
        &format!(
            "nothing in {} matched the extension filter, so no import table was read",
            folder.display()
        ),
    )?;
    anyhow::ensure!(
        read >= 2,
        "xtask ship: only {read} artifact(s) in {} had a dependency surface — a shipped folder \
         holds at least the shell and the game dylib",
        folder.display()
    );
    println!(
        "xtask ship: every library the folder's {read} artifacts need is the OS's or its own \
         (§6 M53)"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host's own `xtask` binary is a real artifact of the format this
    /// parses, which is a better fixture than bytes we made up: a parser that
    /// only reads its own synthetic input is a parser that has never met a
    /// linker.
    #[test]
    fn the_parser_reads_a_real_artifact_of_this_host() {
        let me = std::env::current_exe().unwrap();
        let needs = surface(&me).unwrap();
        assert!(
            !needs.is_empty(),
            "a test binary that imports nothing at all is not credible"
        );
        let names: Vec<&str> = needs.iter().map(|n| n.lib.as_str()).collect();
        if cfg!(windows) {
            assert!(
                names.iter().any(|n| n.eq_ignore_ascii_case("kernel32.dll")),
                "no kernel32 among {names:?}"
            );
        } else {
            assert!(
                names.iter().any(|n| n.starts_with("libc.so")),
                "no libc among {names:?}"
            );
        }
    }

    /// The planted violation for the skip this replaced (§6 M81): a folder whose
    /// binary is unreadable must fail, and a `let Ok(..) else { continue }` there
    /// passed it while reading nothing at all.
    #[test]
    fn a_folder_holding_an_unreadable_binary_is_refused_rather_than_skipped() {
        let dir = std::env::temp_dir().join("gg-deps-unreadable");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Real enough to be staged, malformed enough that no parser reads it:
        // the PE magic and then nothing a header could live in.
        std::fs::write(dir.join("game.dll"), b"MZ\0\0truncated").unwrap();
        std::fs::copy(std::env::current_exe().unwrap(), dir.join("game.exe")).unwrap();

        let refusal = check_folder(&dir).expect_err("an unreadable artifact is not a pass");
        let message = format!("{refusal:#}");
        assert!(message.contains("game.dll"), "names the file: {message}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// And the vacuity guard on the same loop: a folder with nothing to read is
    /// not a folder that passed.
    #[test]
    fn a_folder_with_no_artifacts_at_all_is_refused() {
        let dir = std::env::temp_dir().join("gg-deps-empty");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("readme.txt"), b"controls\n").unwrap();
        assert!(check_folder(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_version_floor_orders_numerically_and_not_alphabetically() {
        assert!(newer("GLIBC_2.39", "GLIBC_2.4"), "2.39 is newer than 2.4");
        assert!(!newer("GLIBC_2.4", "GLIBC_2.39"));
        assert!(newer("GLIBC_2.34", "GLIBC_2.2.5"));
        // Families are not comparable, and pretending otherwise would report a
        // `GCC_` floor as a glibc one.
        assert!(!newer("GCC_3.0", "GLIBC_2.39"));
    }

    #[test]
    fn the_baseline_round_trips_through_its_own_writer() {
        let mut sections = BTreeMap::new();
        sections.insert("a-host".to_string(), vec!["kernel32.dll".to_string()]);
        sections.insert(
            "b-host".to_string(),
            vec!["libc.so.6 GLIBC_2.39".to_string()],
        );
        assert_eq!(read_baseline(&write_baseline(&sections)), sections);
    }

    /// The checked-in file must name both lanes, or a green nightly on one host
    /// means the other's floor is unheld — which is the failure this gate is
    /// least able to notice about itself.
    #[test]
    fn the_checked_in_baseline_covers_both_hosts() {
        let text = std::fs::read_to_string(baseline_path()).unwrap();
        let sections = read_baseline(&text);
        for host in ["x86_64-pc-windows-msvc", "x86_64-unknown-linux-gnu"] {
            let needs = sections
                .get(host)
                .unwrap_or_else(|| panic!("no [{host}] section in dist-deps.txt"));
            assert!(!needs.is_empty(), "[{host}] is empty");
        }
    }

    /// The standard in the module docs, as far as a machine can hold it: the
    /// Windows section may not name the redistributable this milestone exists to
    /// remove. A `+crt-static` that silently stopped applying would otherwise
    /// come back only as a blessed line nobody looked at twice.
    #[test]
    fn the_windows_baseline_names_no_redistributable() {
        let text = std::fs::read_to_string(baseline_path()).unwrap();
        let sections = read_baseline(&text);
        for need in sections.get("x86_64-pc-windows-msvc").into_iter().flatten() {
            let lower = need.to_lowercase();
            assert!(
                !lower.starts_with("vcruntime") && !lower.starts_with("msvcp"),
                "`{need}` is the Visual C++ redistributable, which is not part of Windows \
                 (§6 M53) — the dist profile's static CRT is what keeps it out"
            );
        }
    }
}
