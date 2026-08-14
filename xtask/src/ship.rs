//! `cargo xtask ship <demo>` (§6 M41) — the folder a player unzips.
//!
//! `xtask dist` is a *gate*: it proves the lab equipment unbolted and then
//! leaves a `tier-dist-verify` binary at `target/dist/gg-runtime.exe`, which is
//! not the shipping build. So this builds its own, and the two never share an
//! artifact — the obvious hand-assembly is a trap that gets walked into once.
//!
//! What a demo contributes is `demos/<demo>/game.ggproj`: a real manifest,
//! naming its files by the names they have *in the folder*, so shipping is a
//! copy and not a rewrite. A demo without one is not shippable, and says so —
//! a title is the operator's decision and not a slug this file invents.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use gg_core::config::project::{MANIFEST, Project};

use crate::util::{cargo, dylib_name, run, workspace_root};

/// Where folders are staged. Under `target/`, because a shipped folder is build
/// output like any other and `cargo clean` should take it.
const OUT: &str = "target/ship";

pub fn run_cmd(args: &[&str]) -> Result<()> {
    let demo = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .context("usage: cargo xtask ship <demo> — e.g. `10-tetris`")?;
    let root = workspace_root();
    let crate_dir = root.join("demos").join(demo);
    ensure!(
        crate_dir.is_dir(),
        "no such demo `{demo}` — {} does not exist",
        crate_dir.display()
    );

    let source = crate_dir.join(MANIFEST);
    ensure!(
        source.is_file(),
        "{demo} declares no {MANIFEST}, so it has no name to ship under (§6 M41). Write one \
         beside its `input.toml`:\n\n  game  = {}\n  input = input.toml\n  title = <what the \
         window says>\n",
        dylib_name(&package(demo).replace('-', "_"))
    );
    let project =
        Project::open(&source).with_context(|| format!("reading {}", source.display()))?;

    let folder = root.join(OUT).join(&project.slug);
    if folder.exists() {
        std::fs::remove_dir_all(&folder)
            .with_context(|| format!("clearing {}", folder.display()))?;
    }
    std::fs::create_dir_all(&folder)?;

    let package = package(demo);
    build(&package)?;
    let exe = stage_exe(&folder, &project.slug)?;
    // Names, not paths: `Project::open` resolved everything against the source
    // manifest's directory, and what a folder needs is the same file under the
    // same name one directory over. A mismatch is caught by `stage` failing to
    // find its source, which is the manifest naming a file the demo does not
    // have.
    let dylib = dylib_name(&package.replace('-', "_"));
    stage(&root.join("target/dist").join(&dylib), &folder.join(&dylib))?;
    std::fs::write(folder.join(MANIFEST), render(&project, &dylib))?;
    if let Some(input) = &project.input {
        stage(input, &folder.join(file_name(input)?))?;
    } else {
        println!(
            "xtask ship: WARNING — {demo} ships no action map, so nothing the player does \
             reaches it"
        );
    }
    // A pack the manifest does not name is a pack the game will not open, and a
    // textured game opening no pack is a black screen three layers from the
    // omission. Both directions are errors.
    let has_assets = crate_dir.join("assets").is_dir();
    match (&project.pack, has_assets) {
        (Some(pack), true) => {
            crate::assets::run(&[])?;
            stage(
                &root.join(format!("target/assets/{demo}.ggpack")),
                &folder.join(file_name(pack)?),
            )?;
        }
        (Some(pack), false) => bail!(
            "{MANIFEST} names `pack = {}` but {demo} has no `assets/` tree to compile",
            file_name(pack)?
        ),
        (None, true) => bail!(
            "{demo} has an `assets/` tree but its {MANIFEST} names no pack — the game would \
             launch and draw nothing"
        ),
        (None, false) => {}
    }
    // The project's opening scene is found beside the action map (§6 M20 pull
    // 2), so a level that is data ships as data or the game opens empty.
    let scene = crate_dir.join("scene.ggsave");
    if scene.is_file() {
        stage(&scene, &folder.join("scene.ggsave"))?;
    }

    let files = walk(&folder)?;
    let bytes: u64 = files
        .iter()
        .filter_map(|p| p.metadata().ok())
        .map(|m| m.len())
        .sum();
    println!(
        "xtask ship: {} — {} files, {:.1} MiB → {}",
        project.title,
        files.len(),
        bytes as f64 / (1024.0 * 1024.0),
        folder.display()
    );
    let archive = folder.with_extension("zip");
    let zipped = zip(&folder, &files, &project.slug, &archive, &exe)?;
    println!(
        "xtask ship: {:.1} MiB → {}",
        zipped as f64 / (1024.0 * 1024.0),
        archive.display()
    );
    println!("xtask ship: launch it as a player does — {}", exe.display());
    Ok(())
}

/// The folder's own manifest. Rendered rather than copied for exactly one line:
/// a dylib is `.dll` here and `lib….so` there, so a folder names the file it has
/// and a demo names everything a host cannot decide. The slug is written out
/// even when it was derived, because a folder should say what directory a
/// player's bytes are in rather than make the next build re-derive it.
fn render(project: &Project, dylib: &str) -> String {
    let mut out = format!(
        "# Written by `cargo xtask ship` (§6 M41). The source is demos/<demo>/{MANIFEST};\n\
         # editing this copy changes one folder and nothing that produced it.\n\
         game  = {dylib}\n\
         title = {}\n\
         slug  = {}\n",
        project.title, project.slug
    );
    for (key, path) in [("input", &project.input), ("pack", &project.pack)] {
        if let Some(path) = path
            && let Some(name) = path.file_name()
        {
            out.push_str(&format!("{key} = {}\n", name.to_string_lossy()));
        }
    }
    if let Some((w, h)) = project.window {
        out.push_str(&format!("window = {w}x{h}\n"));
    }
    out
}

/// A stored (uncompressed) zip, written by hand.
///
/// A deflate dependency would buy about a third of 3 MiB and cost the one thing
/// this file can otherwise promise: every byte is a function of the inputs, so
/// two runs over one folder produce one archive (§4.6's habit, applied to the
/// thing a player downloads). Timestamps are pinned to the DOS epoch for the
/// same reason — an archive that differs by when it was built cannot be
/// compared.
fn zip(folder: &Path, files: &[PathBuf], prefix: &str, to: &Path, exe: &Path) -> Result<u64> {
    let (mut out, mut central) = (Vec::new(), Vec::new());
    for path in files {
        // Unix modes, written from a Windows host too: a zip's external
        // attributes are how an extracted executable keeps its `+x`, and a
        // folder whose game cannot be launched is the download failing on the
        // one machine nobody here has.
        let mode: u32 = if path == exe { 0o755 } else { 0o644 };
        let name = format!(
            "{prefix}/{}",
            path.strip_prefix(folder)?
                .to_string_lossy()
                .replace('\\', "/")
        );
        let body = std::fs::read(path)?;
        let (crc, at) = (crc32(&body), out.len() as u32);
        // The ten fields a local header and a central entry share, in the one
        // order both spell them.
        let mut common = Vec::from(20u16.to_le_bytes()); // version needed: 2.0, stored
        common.extend(0u16.to_le_bytes()); // flags
        common.extend(0u16.to_le_bytes()); // method: stored
        common.extend(0u16.to_le_bytes()); // time
        common.extend(0x0021u16.to_le_bytes()); // date: 1980-01-01, the DOS epoch
        common.extend(crc.to_le_bytes());
        common.extend((body.len() as u32).to_le_bytes());
        common.extend((body.len() as u32).to_le_bytes());
        common.extend((name.len() as u16).to_le_bytes());
        common.extend(0u16.to_le_bytes()); // extra field length

        out.extend(0x0403_4b50u32.to_le_bytes());
        out.extend(&common);
        out.extend(name.as_bytes());
        out.extend(&body);

        let mut entry = Vec::from(0x0201_4b50u32.to_le_bytes());
        entry.extend(0x0314u16.to_le_bytes()); // version made by: Unix, 2.0
        entry.extend(&common);
        entry.extend([0u16.to_le_bytes(); 3].concat()); // comment, disk, internal attrs
        entry.extend((mode << 16).to_le_bytes()); // external attrs: the Unix mode
        entry.extend(at.to_le_bytes());
        entry.extend(name.as_bytes());
        central.push(entry);
    }
    let (offset, size) = (out.len() as u32, central.concat().len() as u32);
    out.extend(central.concat());
    out.extend(0x0605_4b50u32.to_le_bytes());
    out.extend([0u16.to_le_bytes(); 2].concat()); // this disk, disk with the directory
    out.extend((files.len() as u16).to_le_bytes());
    out.extend((files.len() as u16).to_le_bytes());
    out.extend(size.to_le_bytes());
    out.extend(offset.to_le_bytes());
    out.extend(0u16.to_le_bytes()); // comment length
    std::fs::write(to, &out).with_context(|| format!("writing {}", to.display()))?;
    Ok(out.len() as u64)
}

/// The one in the zip spec. A table would be faster and this runs once per file.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// The shipping shell, built here rather than found: see the module note.
///
/// **The console goes away at the linker, not in the source.** `windows_subsystem`
/// as a `cfg_attr` would have to key off a feature, and the only feature that
/// separates `tier-dist` from `tier-dist-verify` is the state hash — so a shell
/// that hid its console under "not debug-tools" would take dist-verify's stdout
/// with it, and every §5.6c gate reads exactly that stream. Here the *code* is
/// byte-identical to the gated build and one field of the PE header is not,
/// which is the smallest difference that can exist between them.
fn build(package: &str) -> Result<()> {
    let mut shell = cargo();
    shell.args([
        "rustc",
        "-p",
        "gg-runtime",
        // gg-runtime is a library as well as a binary (the launcher drives it),
        // and link args can only be aimed at one target.
        "--bin",
        "gg-runtime",
        "--profile",
        "dist",
        "--no-default-features",
        "--features",
        "tier-dist",
    ]);
    if cfg!(windows) {
        // `mainCRTStartup` because the entry point is still an ordinary `main`:
        // /SUBSYSTEM:WINDOWS alone makes the linker look for `WinMain`.
        shell.args([
            "--",
            "-Clink-arg=/SUBSYSTEM:WINDOWS",
            "-Clink-arg=/ENTRY:mainCRTStartup",
        ]);
    }
    run(
        &mut shell,
        "build gg-runtime [tier-dist, dist profile, no console]",
    )?;
    run(
        cargo().args(["build", "-p", package, "--profile", "dist"]),
        &format!("build {package} [dist profile]"),
    )
}

/// The shell, under the game's own name. A player's task manager should say
/// what they think they are running.
fn stage_exe(folder: &Path, slug: &str) -> Result<PathBuf> {
    let name = if cfg!(windows) {
        format!("{slug}.exe")
    } else {
        slug.to_owned()
    };
    let to = folder.join(name);
    let from = workspace_root().join("target/dist").join(if cfg!(windows) {
        "gg-runtime.exe"
    } else {
        "gg-runtime"
    });
    stage(&from, &to)?;
    Ok(to)
}

fn stage(from: &Path, to: &Path) -> Result<()> {
    if let Some(dir) = to.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::copy(from, to)
        .with_context(|| format!("staging {} → {}", from.display(), to.display()))?;
    Ok(())
}

fn file_name(path: &Path) -> Result<String> {
    Ok(path
        .file_name()
        .with_context(|| format!("{} names no file", path.display()))?
        .to_string_lossy()
        .into_owned())
}

/// `demos/10-tetris` is package `demo-10-tetris` — the one place the two
/// spellings meet.
fn package(demo: &str) -> String {
    format!("demo-{demo}")
}

fn walk(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            found.extend(walk(&path)?);
        } else {
            found.push(path);
        }
    }
    found.sort();
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest names files by the names they have in the folder, so it is
    /// exactly the kind of thing a rename rots silently — and it would rot into
    /// "the only shippable game does not ship", found the next time someone
    /// ships rather than the next time someone renames.
    #[test]
    fn every_shipped_project_names_files_the_demo_actually_has() {
        let demos = workspace_root().join("demos");
        let mut checked = 0;
        for entry in std::fs::read_dir(&demos).expect("demos/") {
            let dir = entry.expect("entry").path();
            let source = dir.join(MANIFEST);
            if !source.is_file() {
                continue;
            }
            let demo = dir
                .file_name()
                .expect("named")
                .to_string_lossy()
                .into_owned();
            let project = Project::open(&source).expect("a project");
            assert!(
                project.game.as_os_str().is_empty(),
                "{demo}: a demo's manifest must not name a dylib — the spelling is the shipping \
                 host's, and `xtask ship` renders that line"
            );
            if let Some(input) = &project.input {
                assert!(input.is_file(), "{demo}: `input` names {}", input.display());
            }
            assert_eq!(
                project.pack.is_some(),
                dir.join("assets").is_dir(),
                "{demo}: a pack and an assets/ tree exist together or not at all"
            );
            checked += 1;
        }
        assert!(
            checked > 0,
            "no demo declares a {MANIFEST} — the walk lost them"
        );
    }

    /// The archive is a function of the folder and of nothing else — no clock,
    /// no filesystem order — which is what lets two builds of one commit be
    /// compared rather than merely both existing.
    #[test]
    fn an_archive_is_a_function_of_its_input() {
        let dir = std::env::temp_dir().join(format!("gg-zip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp");
        std::fs::write(dir.join("a.txt"), b"one").expect("a");
        std::fs::write(dir.join("b.bin"), [0u8; 500]).expect("b");
        let files = walk(&dir).expect("walk");
        let (first, second) = (dir.join("1.zip"), dir.join("2.zip"));
        let exe = dir.join("a.txt");
        zip(&dir, &files, "game", &first, &exe).expect("zip");
        zip(&dir, &files, "game", &second, &exe).expect("zip");
        let bytes = std::fs::read(&first).expect("read");
        assert_eq!(bytes, std::fs::read(&second).expect("read"));
        // The launchable one carries `+x` and its neighbour does not — the
        // difference an extracted folder needs and a Windows host cannot see.
        let modes: Vec<u32> = bytes
            .windows(4)
            .enumerate()
            .filter(|(_, w)| *w == 0x0201_4b50u32.to_le_bytes())
            .map(|(at, _)| u32::from_le_bytes(bytes[at + 38..at + 42].try_into().expect("4")) >> 16)
            .collect();
        assert_eq!(modes, [0o755, 0o644]);
        // End of central directory, and both counts agreeing with the walk.
        let end = bytes.len() - 22;
        assert_eq!(&bytes[end..end + 4], &0x0605_4b50u32.to_le_bytes());
        assert_eq!(u16::from_le_bytes([bytes[end + 8], bytes[end + 9]]), 2);
        assert_eq!(u16::from_le_bytes([bytes[end + 10], bytes[end + 11]]), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The CRC the spec's own test vector names.
    #[test]
    fn the_checksum_is_the_one_in_the_spec() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }
}
