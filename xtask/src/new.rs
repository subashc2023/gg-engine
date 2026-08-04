//! `cargo xtask new <name>` — a new game crate, from the template (§6 M12).
//!
//! Copy, rename, register, build. It holds no second copy of the template's
//! code: what a new crate gets is the crate CI already compiles (§9 — "the
//! template is also the boundary's minimal example"), so this cannot drift from
//! the template. It can only fail to *find* it, which it does by name.
//!
//! Manual and windowless: it writes to the source tree, so no tier calls it.
//! The §3 gates need no teaching about the result — `budgets::game_crates`
//! finds a `demos/` package that builds a `cdylib` rather than reading a list,
//! so the new crate is under the deny pin from its first `--push`.

use std::path::{Path, PathBuf};

use anyhow::Context as _;

use crate::util;

const TEMPLATE: &str = "99-template";
const TEMPLATE_PKG: &str = "demo-99-template";
const TEMPLATE_LIB: &str = "demo_99_template";
/// The template's `#[component(id = "…")]` namespace. Persisted identity (§4.2)
/// and not source identity, so a copy must not inherit it: two games sharing
/// `template.spinner` is one game's save loading into the other under a name
/// that agrees by accident.
const TEMPLATE_ID_NS: &str = "template.";

/// Where the copied module header is cut. The lines above [`HEADER_KEEP`] and
/// the section at [`HEADER_DROP`] state the template's own role and are false in
/// a copy; the rest is advice a new crate wants, and is therefore *taken from*
/// the template rather than restated here where it would rot separately.
const HEADER_KEEP: &str = "//! Everything below the `gg_game!` block is yours.";
const HEADER_DROP: &str = "//! # Why this is a demo and not a document";
const HEADER_RESUME: &str = "//! # Three things worth knowing";

pub fn run(args: &[&str]) -> anyhow::Result<()> {
    let name = *args
        .iter()
        .find(|a| !a.starts_with("--"))
        .context("usage: cargo xtask new <name> — a demo directory name, e.g. `09-orbit`")?;
    validate(name)?;
    let root = util::workspace_root();
    let src = root.join("demos").join(TEMPLATE);
    let dst = root.join("demos").join(name);
    anyhow::ensure!(
        src.is_dir(),
        "no template at {} — `xtask new` copies it rather than holding its own (§6 M12)",
        src.display()
    );
    anyhow::ensure!(
        !dst.exists(),
        "{} already exists; `xtask new` never writes over a crate",
        dst.display()
    );

    let mut files = Vec::new();
    copy_tree(&src, &dst, Path::new(""), name, &mut files)?;
    register(&root, name)?;
    println!(
        "xtask: demos/{name} — {} files, registered in [workspace] members",
        files.len()
    );

    // Built and tested, not merely written: "created a crate" and "created a
    // crate that works" are different claims, and the second is the only one
    // worth making to someone who has not built this workspace before. The
    // tests are what catch a bad rename — a lib whose identifiers are wrong
    // still compiles when nothing names them.
    util::run(
        util::cargo()
            .args(["nextest", "run", "-p"])
            .arg(format!("demo-{name}")),
        "cargo nextest run (the new crate)",
    )?;

    println!(
        "\n  cargo xtask run {name}            play it\n  \
         cargo xtask run {name} --editor   play it with the editor\n\n\
         demos/{name}/src/lib.rs is the whole game; `cargo build -p demo-{name}` in a second \
         terminal is the reload loop (§6 M5)."
    );
    Ok(())
}

/// The directory name, which becomes package `demo-<name>` and lib
/// `demo_<name>` — so it is constrained by cargo's rules, not only by taste.
fn validate(name: &str) -> anyhow::Result<()> {
    let shaped = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !name.starts_with('-')
        && !name.ends_with('-')
        && !name.contains("--");
    anyhow::ensure!(
        shaped,
        "`{name}` is not a demo directory name: lowercase ASCII, digits, single inner hyphens. \
         The convention is `NN-slug` (`09-orbit`), and the number is what orders `demos/`"
    );
    Ok(())
}

/// The component-id namespace a copy declares: `09-orbit` → `orbit`.
///
/// The leading number orders the directory listing and is not part of what a
/// component *is* — baking it into persisted identity would make renumbering a
/// demo a schema change (§4.2).
fn id_namespace(name: &str) -> &str {
    name.split_once('-')
        .filter(|(head, _)| !head.is_empty() && head.chars().all(|c| c.is_ascii_digit()))
        .map_or(name, |(_, slug)| slug)
}

fn copy_tree(
    src: &Path,
    dst: &Path,
    rel: &Path,
    name: &str,
    files: &mut Vec<PathBuf>,
) -> anyhow::Result<()> {
    let here = src.join(rel);
    std::fs::create_dir_all(dst.join(rel))?;
    for entry in std::fs::read_dir(&here).with_context(|| format!("reading {}", here.display()))? {
        let entry = entry?;
        let rel = rel.join(entry.file_name());
        if entry.path().is_dir() {
            copy_tree(src, dst, &rel, name, files)?;
            continue;
        }
        let out = dst.join(&rel);
        match std::fs::read_to_string(entry.path()) {
            Ok(text) => std::fs::write(&out, rewrite(&text, &rel, name)?)?,
            // An `assets/` tree the template may grow: bytes, not identifiers.
            Err(_) => drop(std::fs::copy(entry.path(), &out)?),
        }
        files.push(rel);
    }
    Ok(())
}

fn rewrite(text: &str, rel: &Path, name: &str) -> anyhow::Result<String> {
    // Longest spelling first: `demo-99-template` *contains* `99-template`, so
    // the bare replacement last, or the package name loses its prefix.
    let text = text
        .replace(TEMPLATE_PKG, &format!("demo-{name}"))
        .replace(TEMPLATE_LIB, &format!("demo_{}", name.replace('-', "_")))
        .replace(TEMPLATE, name)
        .replace(TEMPLATE_ID_NS, &format!("{}.", id_namespace(name)));
    match rel.file_name().and_then(|f| f.to_str()).unwrap_or_default() {
        "lib.rs" => lib_header(&text, name),
        "Cargo.toml" => manifest_header(&text, name),
        _ => Ok(text),
    }
}

/// The copy's module header: one generated line, then the template's own words
/// on the contract and on what not to delete.
fn lib_header(text: &str, name: &str) -> anyhow::Result<String> {
    let at = |needle: &str| {
        text.find(needle).with_context(|| {
            format!(
                "the template's module header no longer contains `{needle}`, which is where \
                 `xtask new` cuts its copy — re-anchor xtask/src/new.rs against the header as it \
                 now reads (§6 M12)"
            )
        })
    };
    let (keep, drop, resume) = (at(HEADER_KEEP)?, at(HEADER_DROP)?, at(HEADER_RESUME)?);
    anyhow::ensure!(
        keep < drop && drop < resume,
        "the template's header sections were reordered; `xtask new` assumes contract, then the \
         template's own role, then the advice a copy keeps"
    );
    Ok(format!(
        "//! `{name}` — a game crate, copied from the template (§6 M12) by `cargo xtask new`.\n\
         //!\n{}{}",
        &text[keep..drop],
        &text[resume..]
    ))
}

/// Same cut in the manifest: its leading block is about the template's standing
/// in CI. Everything from `[package]` down is contract — the `cdylib`/`rlib`
/// pair, the `game` feature's fixed symbol names — and stays.
fn manifest_header(text: &str, name: &str) -> anyhow::Result<String> {
    let at = text.find("[package]").context(
        "the template's Cargo.toml has no `[package]` table — `xtask new` replaces the comment \
         block above it",
    )?;
    Ok(format!(
        "# demo {name}, from `cargo xtask new`: the template (§6 M12) renamed.\n\
         # Three engine dependencies and no fourth, like every game crate — §3's\n\
         # deny pin is a gate, and `xtask ci --push` is where it fails.\n\n{}",
        &text[at..]
    ))
}

/// The new crate in `[workspace] members`, in sorted position.
///
/// Text insertion rather than a `toml` round-trip: the manifest is mostly
/// comments carrying §3's reasoning, and a serializer would drop every one. The
/// list is hand-sorted, so a row appended at the bottom is the first line of
/// drift.
fn register(root: &Path, name: &str) -> anyhow::Result<()> {
    let path = root.join("Cargo.toml");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let ending = if text.contains("\r\n") { "\r\n" } else { "\n" };
    let row = format!("\"demos/{name}\",");

    let (at, indent) = {
        let demos: Vec<(usize, &str)> = lines(&text)
            .filter(|(_, line)| line.trim_start().starts_with("\"demos/"))
            .collect();
        let (_, first) = *demos.first().context(
            "no `demos/…` rows in [workspace] members — `xtask new` registers the crate beside \
             them, and cannot guess where they went",
        )?;
        let indent = first[..first.len() - first.trim_start().len()].to_owned();
        // Sorted position, or after the last demo row when the name sorts last.
        let after_all = demos
            .iter()
            .map(|(start, line)| start + line.len() + ending.len())
            .max()
            .unwrap_or(0);
        let at = demos
            .iter()
            .find(|(_, line)| line.trim() > row.as_str())
            .map_or(after_all, |(start, _)| *start);
        (at, indent)
    };

    let mut out = text;
    out.insert_str(at, &format!("{indent}{row}{ending}"));
    std::fs::write(&path, out).with_context(|| format!("writing {}", path.display()))
}

/// `(byte offset, line without its ending)`. Offsets rather than a rebuilt line
/// vector so the file's own endings survive the edit on both hosts.
fn lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    let mut at = 0;
    text.split_inclusive('\n').map(move |line| {
        let start = at;
        at += line.len();
        (start, line.trim_end_matches(['\r', '\n']))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_cargo_shaped() {
        for ok in ["09-orbit", "orbit", "99-template", "a1"] {
            assert!(validate(ok).is_ok(), "{ok}");
        }
        for bad in ["", "-x", "x-", "a--b", "Orbit", "my_game", "09 orbit"] {
            assert!(validate(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn the_number_is_not_part_of_persisted_identity() {
        assert_eq!(id_namespace("09-orbit"), "orbit");
        assert_eq!(id_namespace("99-template"), "template");
        assert_eq!(id_namespace("orbit"), "orbit");
        // Not a leading *number*, so nothing is stripped.
        assert_eq!(id_namespace("my-game"), "my-game");
    }

    /// The package name must not lose its prefix to the bare-name replacement.
    #[test]
    fn identifiers_are_rewritten_longest_first() {
        let text = "demo-99-template demo_99_template 99-template template.spinner";
        let out = rewrite(text, Path::new("input.toml"), "09-orbit").unwrap();
        assert_eq!(out, "demo-09-orbit demo_09_orbit 09-orbit orbit.spinner");
    }

    #[test]
    fn the_copied_header_drops_the_templates_own_role() {
        let text = format!(
            "//! **The template** (§6 M12): a spinning lit mesh.\n\
             //!\n{HEADER_KEEP} The block is the contract.\n\
             //!\n{HEADER_DROP}\n//!\n//! CI builds it.\n\
             //!\n{HEADER_RESUME} before you delete anything\n//!\n//! - Renderable is a box.\n\
             \nuse gg_ecs::Component;\n"
        );
        let out = lib_header(&text, "09-orbit").unwrap();
        assert!(out.starts_with("//! `09-orbit` — a game crate, copied from the template"));
        assert!(out.contains("The block is the contract."), "contract kept");
        assert!(out.contains("- Renderable is a box."), "advice kept");
        assert!(
            !out.contains("CI builds it."),
            "the template's role dropped"
        );
        assert!(out.ends_with("use gg_ecs::Component;\n"), "body untouched");
    }

    #[test]
    fn a_moved_header_anchor_fails_by_name() {
        let err = lib_header("//! nothing to anchor against\n", "09-orbit").unwrap_err();
        assert!(format!("{err}").contains(HEADER_KEEP));
    }

    /// Against the real template, because no tier runs `xtask new` — without
    /// this the anchors rot silently and the next person to start a game is the
    /// one who finds out. Cheap enough to sit in the fast tier.
    #[test]
    fn the_real_template_still_copies() {
        let root = util::workspace_root().join("demos").join(TEMPLATE);
        for file in ["src/lib.rs", "Cargo.toml", "tests/game.rs", "input.toml"] {
            let path = root.join(file);
            let text = std::fs::read_to_string(&path).unwrap();
            let out = rewrite(&text, Path::new(file), "09-orbit").unwrap();
            assert!(!out.contains(TEMPLATE), "{file} still names the template");
            assert!(!out.contains(TEMPLATE_PKG), "{file} keeps the package name");
            assert!(!out.contains(TEMPLATE_LIB), "{file} keeps the lib name");
            assert!(!out.contains(TEMPLATE_ID_NS), "{file} keeps a component id");
        }
    }
}
