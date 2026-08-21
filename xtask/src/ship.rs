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

use crate::notes;
use crate::rsrc;
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
    // The executable's own picture and version (§6 M73), decided before the link
    // because that is the only moment they can be put in: a `.rsrc` section is
    // the linker's output, and adding one afterwards means moving every section
    // that follows it.
    let marks = marks(&project)?;
    let built = resource(&project, &marks)?;
    build(&package, built.as_ref().map(|(path, _)| path.as_path()))?;
    let exe = stage_exe(&folder, &project.slug)?;
    if let Some((_, built)) = &built {
        let bytes = std::fs::read(&exe).with_context(|| format!("reading {}", exe.display()))?;
        let leaves = rsrc::verify(&bytes, built, &marks)
            .with_context(|| format!("checking what {} tells Explorer", exe.display()))?;
        println!(
            "xtask ship: {} carries {leaves} resources — {} sizes of icon and the version (§6 M73)",
            file_name(&exe)?,
            built.icons.len()
        );
    } else if cfg!(windows) {
        println!(
            "xtask ship: WARNING — {demo} declares neither an icon nor a version, so Explorer \
             shows the default picture and a blank Details tab (§6 M73)"
        );
    }
    // Names, not paths: `Project::open` resolved everything against the source
    // manifest's directory, and what a folder needs is the same file under the
    // same name one directory over. A mismatch is caught by `stage` failing to
    // find its source, which is the manifest naming a file the demo does not
    // have.
    let dylib = dylib_name(&package.replace('-', "_"));
    writable(&project)?;
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
    // The taskbar picture (§6 M46), staged like the action map and unlike the
    // pack: it is a checked-in file beside the manifest rather than build output,
    // because a window icon is wanted before a pack is open.
    if let Some(icon) = &project.icon {
        stage(icon, &folder.join(file_name(icon)?))?;
    }
    // The project's opening scene is found beside the action map (§6 M20 pull
    // 2), so a level that is data ships as data or the game opens empty.
    let scene = crate_dir.join("scene.ggsave");
    if scene.is_file() {
        stage(&scene, &folder.join("scene.ggsave"))?;
    }
    // The two files for reading (§6 M73), rendered from the folder rather than
    // written once — and the readme's controls come from the *staged* map, so
    // it describes this folder and not the demo it was assembled from.
    let map = match &project.input {
        Some(input) => std::fs::read_to_string(folder.join(file_name(input)?)).ok(),
        None => None,
    };
    // Before the readme is written from it (§6 M81): an action map is a subset
    // of TOML and a wrapped array is valid TOML the engine refuses, so this is a
    // folder whose game starts with no bindings at all. The readme would have
    // described the scheme anyway, because it parses with the full `toml` crate.
    if let Some(line) = map.as_deref().and_then(notes::wrapped_array) {
        bail!(
            "the staged action map opens an array on line {line} and does not close it there — \
             `gg-input` reads a subset of TOML in which an array is on one line (§6 M74), so \
             this folder ships a game with no bindings. A formatter did this; put the array back \
             on one line."
        );
    }
    std::fs::write(
        folder.join(notes::README),
        notes::readme(&project, map.as_deref(), &marks.engine, &file_name(&exe)?),
    )?;
    // Last, because it is the slowest thing here and the only one that resolves
    // a dependency graph.
    std::fs::write(
        folder.join(notes::NOTICES),
        notes::notices(&package, &project.title)?,
    )?;

    // Before the zip, not after: what the folder demands of a stranger's machine
    // is the one property a zip cannot carry and the operator cannot see (§6 M53).
    crate::deps::check_folder(&folder)?;

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
    // Built by hand rather than by `with_extension`, which would eat everything
    // after the first dot in a version. The version is *in* the name because two
    // downloads called `falling-blocks.zip` are two files a player cannot tell
    // apart, and itch keys an upload on its file name.
    let archive = folder.with_file_name(match &project.version {
        Some(v) => format!("{}-{}.zip", project.slug, v),
        None => format!("{}.zip", project.slug),
    });
    let zipped = zip(&folder, &files, &project.slug, &archive, &exe)?;
    println!(
        "xtask ship: {:.1} MiB → {}",
        zipped as f64 / (1024.0 * 1024.0),
        archive.display()
    );
    println!("xtask ship: launch it as a player does — {}", exe.display());
    Ok(())
}

/// Every value [`render`] will write comes back out of the file unchanged
/// (§6 M87).
///
/// The manifest format has no escaping and is not growing any: `#` opens a
/// comment, a newline ends the line, and both sides of a value are trimmed. So
/// a title holding one of those is a folder whose game is called something
/// else — silently, since the reader's answer to a mangled value is a shorter
/// name and not an error. Refused here, where the operator can still change it.
fn writable(project: &Project) -> Result<()> {
    let names: Vec<String> = [&project.input, &project.pack, &project.icon]
        .into_iter()
        .flatten()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect();
    let version = project.version.as_ref().map(ToString::to_string);
    for value in [&project.title, &project.slug]
        .into_iter()
        .cloned()
        .chain(names)
        .chain(version)
    {
        ensure!(
            !value.contains('#') && !value.contains(['\n', '\r']) && value.trim() == value,
            "`{value}` cannot be written to a {MANIFEST} and read back: the format is flat \
             `name = value` with no escaping, so `#` opens a comment, a newline ends the line, \
             and surrounding spaces are trimmed away. Rename it rather than teaching the format \
             to quote (§6 M87)"
        );
    }
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
    for (key, path) in [
        ("input", &project.input),
        ("pack", &project.pack),
        ("icon", &project.icon),
    ] {
        if let Some(path) = path
            && let Some(name) = path.file_name()
        {
            out.push_str(&format!("{key} = {}\n", name.to_string_lossy()));
        }
    }
    if let Some(version) = &project.version {
        out.push_str(&format!("version = {version}\n"));
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
/// What the executable will say about itself (§6 M73). Everything but the
/// engine line is the manifest's; the engine line is this tree's, and carries
/// the commit because that is what a bug report is answered against and the one
/// fact a player has no other way to read off the file.
fn marks(project: &Project) -> Result<rsrc::Marks> {
    let engine = workspace_root().join("crates/gg-runtime/Cargo.toml");
    let text = std::fs::read_to_string(&engine)
        .with_context(|| format!("reading {}", engine.display()))?;
    let version: toml::Value = toml::from_str(&text).context("parsing gg-runtime's manifest")?;
    let version = version
        .get("package")
        .and_then(|p| p.get("version"))
        .and_then(toml::Value::as_str)
        .context("gg-runtime declares no version")?;
    Ok(rsrc::Marks {
        title: project.title.clone(),
        version: project.version.clone(),
        slug: project.slug.clone(),
        exe: if cfg!(windows) {
            format!("{}.exe", project.slug)
        } else {
            project.slug.clone()
        },
        engine: format!("GGEngine {version} ({})", crate::dist::commit()?),
    })
}

/// The resource object, written where a *changed* one has a different path.
///
/// That is not tidiness. Cargo fingerprints a link argument by its text, so an
/// object at a fixed path whose bytes changed is an object cargo has no reason
/// to relink against — a new icon would silently ship the old one, and the gate
/// downstream would pass because it compares the artifact against what it
/// re-derives rather than against what it meant. Hashing the bytes into the name
/// makes a changed resource a changed command line.
///
/// `None` off Windows and for a game with no icon: an ELF has no resource
/// section, and the manifest's own picture is what there is to put in one.
fn resource(project: &Project, marks: &rsrc::Marks) -> Result<Option<(PathBuf, rsrc::Built)>> {
    // A version with no icon is a resource too (§6 M81). Until then the icon was
    // the only trigger, so such a manifest shipped a blank *Details* tab and —
    // worse — skipped `rsrc::verify`, which is the only check that covers what
    // the linker did with any of this.
    if !cfg!(windows) || (project.icon.is_none() && project.version.is_none()) {
        return Ok(None);
    }
    let icon = project
        .icon
        .as_ref()
        .map(|icon| std::fs::read(icon).with_context(|| format!("reading {}", icon.display())))
        .transpose()?;
    let built = rsrc::object(icon.as_deref(), marks)?;
    let digest = <sha2::Sha256 as sha2::Digest>::digest(&built.object);
    let dir = workspace_root().join("target/xtask-cache/rsrc");
    std::fs::create_dir_all(&dir)?;
    let name: String = digest.iter().take(8).map(|b| format!("{b:02x}")).collect();
    let path = dir.join(format!("{name}.obj"));
    std::fs::write(&path, &built.object).with_context(|| format!("writing {}", path.display()))?;
    Ok(Some((path, built)))
}

fn build(package: &str, resource: Option<&Path>) -> Result<()> {
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
        // An ordinary input object, which is the whole trick: `.rsrc$01` and
        // `.rsrc$02` sort and merge into one `.rsrc` and the linker fills the
        // resource data directory from it, exactly as it would for `cvtres`.
        if let Some(path) = resource {
            shell.arg(format!("-Clink-arg={}", path.display()));
        }
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
            // The icon is a checked-in file rather than build output, so unlike
            // the pack it is asserted to *exist* — and parsed, because an
            // unreadable one is dropped at runtime with a warning and would
            // otherwise ship as a game that silently has no picture (§6 M46).
            if let Some(icon) = &project.icon {
                let bytes = std::fs::read(icon)
                    .unwrap_or_else(|e| panic!("{demo}: `icon` names {}: {e}", icon.display()));
                gg_core::config::icon::parse(&bytes)
                    .unwrap_or_else(|e| panic!("{demo}: {} is not an icon: {e}", icon.display()));
            }
            checked += 1;
        }
        assert!(
            checked > 0,
            "no demo declares a {MANIFEST} — the walk lost them"
        );
    }

    /// The folder's manifest is the demo's, and the shell reads it back whole
    /// (§6 M87).
    ///
    /// The round trip was asserted for `version` and for nothing else, which is
    /// one of eight keys — so a field [`render`] stopped writing, or wrote in a
    /// spelling [`Project::parse`] does not accept, would reach a player as a
    /// game that lost its icon, its pack or its window and said nothing. Graded
    /// as **struct equality**, not field by field: a ninth key added to
    /// `Project` and forgotten here fails this test rather than passing a list
    /// nobody widened.
    #[test]
    fn every_key_the_writer_puts_in_a_folder_comes_back_out_of_it() {
        let whole = Project::parse(
            "game  = ignored.dll\n\
             title = Falling Blocks\n\
             slug  = tetris\n\
             input = input.toml\n\
             pack  = tetris.ggpack\n\
             icon  = tetris.ggicon\n\
             version = 0.9.0-rc2\n\
             window  = 1280x720\n",
        )
        .expect("a manifest with every key set");
        // `render` names the dylib itself, since `.dll` against `lib….so` is the
        // shipping host's answer; everything else must survive untouched.
        let round = Project::parse(&render(&whole, "tetris.dll")).expect("the folder's manifest");
        assert_eq!(
            round,
            Project {
                game: PathBuf::from("tetris.dll"),
                ..whole
            }
        );
    }

    /// A value the format cannot carry is refused where it is written, not
    /// discovered where it is read (§6 M87).
    ///
    /// The manifest has no escaping and is not growing any: `#` opens a comment
    /// and a value is trimmed, so a title holding either is a folder whose game
    /// is called something else. Round-tripping is the property this file can
    /// actually promise, and the promise is kept by refusing rather than by
    /// mangling.
    #[test]
    fn a_title_the_manifest_cannot_carry_is_refused_rather_than_truncated() {
        for hostile in ["Hash # Run", "Two\nLines", "  padded  "] {
            let project = Project::parse("title = X\nslug = x\n")
                .map(|p| Project {
                    title: hostile.to_owned(),
                    ..p
                })
                .expect("a project");
            assert!(
                writable(&project).is_err(),
                "`{hostile}` is not a title a manifest can hold, so `ship` must say so"
            );
        }
        let fine = Project::parse("title = Falling Blocks").expect("a project");
        assert!(writable(&fine).is_ok());
    }

    /// The version reaches the folder's own manifest and the archive's name
    /// (§6 M73), which are the two places a *player* can read it without opening
    /// anything — and the second is the one that stops two downloads from being
    /// the same file name.
    ///
    /// The suffix case is here because it is the one a naive path join gets
    /// wrong: `with_extension` on `falling-blocks-0.9.0-rc2` truncates at the
    /// first dot and ships `falling-blocks-0.zip`.
    #[test]
    fn a_version_reaches_the_folder_and_the_name_of_the_download() {
        let of = |text: &str| {
            Project::parse(&format!("title = Falling Blocks\nversion = {text}\n")).expect("parsed")
        };
        let rendered = render(&of("1.2.3"), "game.dll");
        assert!(rendered.contains("\nversion = 1.2.3\n"), "{rendered}");
        // A rendered manifest is re-read by the shell beside the executable, so
        // the round trip is the claim rather than the substring.
        assert_eq!(
            Project::parse(&rendered).expect("re-read").version,
            of("1.2.3").version
        );
        let named = |project: &Project| {
            std::path::Path::new("target/ship/falling-blocks")
                .with_file_name(match &project.version {
                    Some(v) => format!("{}-{}.zip", project.slug, v),
                    None => format!("{}.zip", project.slug),
                })
                .to_string_lossy()
                .into_owned()
        };
        assert!(named(&of("1.2.3")).ends_with("falling-blocks-1.2.3.zip"));
        assert!(named(&of("0.9.0-rc2")).ends_with("falling-blocks-0.9.0-rc2.zip"));
        let bare = Project::parse("title = Falling Blocks").expect("parsed");
        assert!(named(&bare).ends_with("falling-blocks.zip"));
        assert!(!render(&bare, "game.dll").contains("version"));
    }

    /// The archive is a function of the folder and of nothing else — no clock,
    /// no filesystem order — which is what lets two builds of one commit be
    /// compared rather than merely both existing.
    #[test]
    fn an_archive_is_a_function_of_its_input() {
        let dir = std::env::temp_dir().join(format!("gg-zip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
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
