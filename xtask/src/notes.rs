//! The two files in the folder that are for **reading** (§6 M73): what the game
//! is and how to play it, and what it is made of.
//!
//! Both are *rendered*, and that is the whole design rather than a convenience.
//! A readme written by hand is a file that describes the build it was written
//! for: the day a key moves in `input.toml`, the checked-in prose is wrong and
//! nothing says so. Here the controls **are** the shipped action map, the paths
//! **are** the slug the folder was assembled under, and the versions are the
//! manifest's and the tree's — so the only way to make the readme wrong is to
//! make the folder wrong.
//!
//! `notices.txt` is the one file here that is not a nicety. Almost everything
//! this artifact links is MIT or Apache-2.0, and both require the notice to
//! travel with the binary; a zip on itch with no attribution is out of
//! compliance with about a hundred licences at once. It is generated from
//! `cargo tree`'s own `{l}` field rather than from a list somebody maintains,
//! for the readme's reason one severity up: a dependency added next month is in
//! the file the next time `xtask ship` runs, and a list nobody updates is worse
//! than no list because it looks like one.

use anyhow::Result;
use gg_core::config::project::Project;

use crate::util::{cargo, run_capture};

/// What a folder's `readme.txt` is called. Lowercase and `.txt`, because the
/// reader is a stranger on Windows double-clicking it and not a repository.
pub const README: &str = "readme.txt";

/// And the attributions.
pub const NOTICES: &str = "notices.txt";

/// The readme's text, from the project and its action map.
///
/// `input` is the map's own source, already staged — read rather than resolved
/// from the demo, so what the file describes is what the folder holds.
pub fn readme(project: &Project, input: Option<&str>, engine: &str, exe: &str) -> String {
    let mut out = String::new();
    let head = match &project.version {
        Some(v) => format!("{} {v}", project.title),
        None => project.title.clone(),
    };
    out.push_str(&rule(&head));
    out.push_str(&format!("\nDouble-click {exe}.\n"));

    if let Some(text) = input {
        let bound = controls(text);
        if !bound.is_empty() {
            out.push_str(&rule("Controls"));
            let width = bound.iter().map(|(v, _)| v.len()).max().unwrap_or(0);
            for (verb, keys) in &bound {
                out.push_str(&format!("\n  {verb:width$}   {keys}"));
            }
            out.push_str(
                "\n\n  Keys are physical positions rather than letters, so this means the\n  \
                 same thing on any keyboard layout. To change them, write a file called\n  \
                 `bindings.cfg` in the folder below with the lines you want to differ.\n",
            );
        }
    }

    out.push_str(&rule("Your files"));
    out.push_str(&format!(
        "\n  Windows   %LOCALAPPDATA%\\{slug}\n  Linux     ~/.local/share/{slug}\n\n  \
         Saved games, settings, `bindings.cfg` and `log.txt` are there and\n  nowhere else — \
         nothing is written beside this folder, so it can live on a\n  memory stick. Deleting \
         that directory resets the game and loses your\n  scores.\n",
        slug = project.slug
    ));

    out.push_str(&rule("If it will not start"));
    out.push_str(
        "\n  The game says why, in a box and in `log.txt` in the directory above.\n  \
         The usual cause is a machine with no Vulkan driver, and the message\n  names what it \
         looked for.\n\n  Move the whole folder somewhere you can write to — a game unzipped \
         into\n  Program Files cannot save.\n",
    );

    out.push_str(&rule("This build"));
    out.push_str(&format!(
        "\n  {head}\n  {engine}\n\n  Licences for everything this game links are in `{NOTICES}`.\n"
    ));
    out
}

/// `cargo tree --format "{p}|{l}|{r}"` into `(package, licence, repository)`.
///
/// Two things about that output are traps. `(*)` — cargo's mark for a subtree it
/// has already printed — lands at the end of the **line**, so it arrives glued
/// to the repository and a naive parse files one crate under two spellings. And
/// a path dependency prints its directory in the package field, which is this
/// workspace: the game's own crates are not a third party and their licence is
/// the one the folder ships under.
///
/// Proc macros stay in, and that is a decision rather than an oversight. Their
/// code is not linked into the artifact, but what they expand to is — and a
/// notice that lists too much costs a reader a line, while one that lists too
/// little is the failure this file exists to prevent.
fn absorb(out: &str, rows: &mut Vec<(String, String, String)>) {
    for line in out.lines() {
        let line = line.trim().trim_end_matches("(*)").trim();
        let mut parts = line.split('|');
        let (Some(name), Some(licence), Some(repo)) = (parts.next(), parts.next(), parts.next())
        else {
            continue;
        };
        let (name, licence) = (name.trim(), licence.trim());
        // A path dependency's field ends in `(<absolute directory>)`; a registry
        // crate's is `name vX.Y.Z` with nothing after it but `(proc-macro)`.
        if name.is_empty() || (name.ends_with(')') && !name.ends_with("(proc-macro)")) {
            continue;
        }
        let row = (
            name.to_owned(),
            // An empty licence field is a crate that declares none, and saying
            // so is the point — it is the row somebody has to go and read.
            if licence.is_empty() {
                "<none declared>".to_owned()
            } else {
                licence.to_owned()
            },
            repo.trim().to_owned(),
        );
        if !rows.contains(&row) {
            rows.push(row);
        }
    }
}

/// A heading and its underline, which is all the structure a text file gets.
fn rule(text: &str) -> String {
    format!("\n{text}\n{}\n", "-".repeat(text.chars().count()))
}

/// `verb → keys` out of an action map's `[game.actions]`, in the file's own
/// order.
///
/// Parsed here rather than through [`gg_input::ActionMap`], which wants the
/// game's declared verb list to resolve ids against — and this file has no ids
/// in it. What a reader wants is the two columns the file already is.
fn controls(text: &str) -> Vec<(String, String)> {
    if wrapped_array(text).is_some() {
        return Vec::new();
    }
    let Ok(doc) = toml::from_str::<toml::Value>(text) else {
        return Vec::new();
    };
    let Some(actions) = doc.get("game").and_then(|g| g.get("actions")) else {
        return Vec::new();
    };
    let Some(table) = actions.as_table() else {
        return Vec::new();
    };
    table
        .iter()
        .filter_map(|(verb, keys)| {
            let keys: Vec<&str> = keys
                .as_array()?
                .iter()
                .filter_map(toml::Value::as_str)
                .collect();
            // `ui_click`/`ui_focus` and the pointer axes are the menu's plumbing
            // rather than a control anyone rebinds, and a readme that lists
            // "ui_x  PointerX" has spent its reader's attention on nothing.
            (!keys.is_empty() && !verb.starts_with("ui_"))
                .then(|| (verb.replace('_', " "), keys.join(", ")))
        })
        .collect()
}

/// The line an array opens on and does not close on, if there is one (§6 M74,
/// §6 M81).
///
/// An action map is a *subset* of TOML and `gg-input` has no `toml` dependency:
/// an array must be on one line, and a formatter that wraps a long one produces
/// a file that is valid TOML and is not a map. This file parses with the full
/// `toml` crate, so without this the readme happily described a control scheme
/// the game would refuse to load — the readme's whole claim being that its
/// controls *are* the staged map.
///
/// Textual and deliberately so: reusing `ActionMap::parse` needs the game's
/// declared verb list, which is exactly what this side does not have.
pub fn wrapped_array(text: &str) -> Option<usize> {
    text.lines().enumerate().find_map(|(at, line)| {
        let line = line.split('#').next().unwrap_or(line);
        let (_, rest) = line.split_once('=')?;
        let rest = rest.trim();
        (rest.starts_with('[') && !rest.ends_with(']')).then_some(at + 1)
    })
}

/// Every crate this artifact links, with the licence it carries.
///
/// Read out of `cargo tree`'s `{l}` rather than a checked-in list: the graph is
/// the truth and a list is a copy of it that stops being one. Both graphs go in
/// — the shell's and the game dylib's — because a player receives both files and
/// a dependency reached only by the game is still in the zip.
///
/// # Errors
/// If either `cargo tree` fails, which is a broken feature combination rather
/// than a missing licence.
pub fn notices(package: &str, title: &str) -> Result<String> {
    let mut rows: Vec<(String, String, String)> = Vec::new();
    for args in [
        vec![
            "tree",
            "-p",
            "gg-runtime",
            "--no-default-features",
            "--features",
            "tier-dist",
        ],
        vec!["tree", "-p", package],
    ] {
        let mut cmd = cargo();
        cmd.args(args);
        cmd.args([
            "-e",
            "normal",
            "--prefix",
            "none",
            "--format",
            "{p}|{l}|{r}",
        ]);
        absorb(
            &run_capture(&mut cmd, "cargo tree for the licence list")?,
            &mut rows,
        );
    }
    // Sorted and deduplicated, so two runs of one commit write one file — the
    // property the archive around it already has.
    rows.sort();
    rows.dedup();
    let mut out = format!(
        "{title} links the free software listed below, and this file is the notice\n\
         those licences ask to travel with it. Each entry is the crate, the licence\n\
         its authors chose, and where to read both the terms and the source.\n\n\
         Written by `cargo xtask ship` (§6 M73) from the build's own dependency\n\
         graph, so it is this artifact's list rather than a list of some other\n\
         build's. {} crates.\n",
        rows.len()
    );
    let width = rows.iter().map(|(n, _, _)| n.len()).max().unwrap_or(0);
    for (name, licence, repo) in &rows {
        out.push_str(&format!("\n  {name:width$}  {licence}"));
        if !repo.is_empty() {
            out.push_str(&format!("\n  {:width$}  {repo}", ""));
        }
    }
    out.push('\n');
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(version: &str) -> Project {
        Project::parse(&format!("title = Falling Blocks\nversion = {version}\n")).expect("parsed")
    }

    /// The claim that makes a rendered readme worth having: every fact in it
    /// came from somewhere, so none of it can be true of a different build.
    #[test]
    fn a_readme_says_what_the_folder_says_and_nothing_it_was_told_once() {
        let map = "[game.actions]\nhard_drop = [\"W\", \"Up\"]\nhold = [\"Space\"]\n\
                   ui_click = [\"Mouse1\"]\n[game.axes]\nui_x = [\"PointerX\"]\n";
        let text = readme(
            &project("1.2.3"),
            Some(map),
            "GGEngine 0.1.0 (abc1234)",
            "falling-blocks.exe",
        );
        assert!(text.contains("Falling Blocks 1.2.3"));
        assert!(text.contains("falling-blocks.exe"));
        assert!(text.contains("%LOCALAPPDATA%\\falling-blocks"));
        assert!(text.contains("~/.local/share/falling-blocks"));
        assert!(text.contains("GGEngine 0.1.0 (abc1234)"));
        // The controls are the map's, both spellings of a two-key verb, and with
        // the underscore gone because a reader is not reading source.
        assert!(text.contains("hard drop"), "{text}");
        assert!(text.contains("W, Up"));
        assert!(text.contains("hold"));
        // The menu's own plumbing is not a control.
        assert!(!text.contains("ui_click"), "{text}");
        assert!(!text.contains("PointerX"));
        // A game with no map still gets a readme, without an empty heading in it.
        let bare = readme(&project("1.0"), None, "GGEngine", "g.exe");
        assert!(!bare.contains("Controls"));
        assert!(bare.contains("Your files"));
        // An unversioned build says its name once rather than its name and a
        // blank where a version goes.
        let none = Project::parse("title = Falling Blocks").expect("parsed");
        let text = readme(&none, None, "GGEngine", "g.exe");
        assert!(text.starts_with("\nFalling Blocks\n----"), "{text}");
    }

    /// The three shapes `cargo tree` emits that a naive split gets wrong, and
    /// the reason this parse is a function rather than a loop body: `(*)` marks
    /// the *line* and would otherwise file one crate twice, a path dependency is
    /// this workspace rather than a third party, and a crate that declares no
    /// licence must show up saying so.
    #[test]
    fn a_dependency_graph_lists_each_crate_once_and_this_workspace_never() {
        let out = "anyhow v1.0.104|MIT OR Apache-2.0|https://github.com/dtolnay/anyhow\n\
                   ash v0.38.0|MIT OR Apache-2.0|https://github.com/ash-rs/ash\n\
                   ash v0.38.0|MIT OR Apache-2.0|https://github.com/ash-rs/ash (*)\n\
                   gg-core v0.1.0 (C:\\dev\\GGEngine\\crates\\gg-core)||\n\
                   demo-10-tetris v0.1.0 (/home/x/gg/demos/10-tetris)||\n\
                   bytemuck_derive v1.11.0 (proc-macro)|Zlib|https://example.invalid\n\
                   mystery v0.1.0||\n";
        let mut rows = Vec::new();
        absorb(out, &mut rows);
        absorb(out, &mut rows); // the second graph overlaps the first entirely
        let names: Vec<&str> = rows.iter().map(|(n, _, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "anyhow v1.0.104",
                "ash v0.38.0",
                "bytemuck_derive v1.11.0 (proc-macro)",
                "mystery v0.1.0"
            ]
        );
        assert_eq!(
            rows[1].2, "https://github.com/ash-rs/ash",
            "the mark is not a URL"
        );
        assert_eq!(rows[3].1, "<none declared>");
    }

    /// A malformed or foreign map is a readme without a controls section, never
    /// a failed ship: the folder is already assembled by the time this runs and
    /// a prose file is not worth refusing one over.
    #[test]
    fn a_map_this_cannot_read_costs_a_section_and_not_the_folder() {
        assert!(controls("not toml at all {{{").is_empty());
        assert!(controls("[game]\nactions = 3\n").is_empty());
        assert!(controls("[other.actions]\njump = [\"Space\"]\n").is_empty());
        assert!(controls("[game.actions]\njump = \"Space\"\n").is_empty());
        assert_eq!(
            controls("[game.actions]\njump = [\"Space\"]\n"),
            vec![("jump".to_owned(), "Space".to_owned())]
        );
    }

    /// The one map that is valid TOML and is not an action map (§6 M74, §6
    /// M81): a formatter wrapped the array, so the engine refuses it and the
    /// full-`toml` parse here read it happily. The readme must not describe a
    /// scheme the game will not load, and `xtask ship` refuses the folder.
    #[test]
    fn a_wrapped_array_is_found_by_the_line_it_opens_on() {
        const WRAPPED: &str = "[game.actions]\nlook = [\n  \"Tab\",\n  \"Escape\",\n]\n";
        assert_eq!(wrapped_array(WRAPPED), Some(2));
        assert!(controls(WRAPPED).is_empty(), "and no section is rendered");
        // The one-line spelling of the same intent is untouched, which is what
        // says the check is about the wrapping and not about the length.
        assert_eq!(
            wrapped_array("[game.actions]\nlook = [\"Tab\", \"Escape\"]\n"),
            None
        );
        // A `]` inside a comment on the opening line is still an open array.
        assert_eq!(
            wrapped_array("[game.actions]\nlook = [\"Tab\", # ]\n  \"Escape\"]\n"),
            Some(2)
        );
        // And a section header is not an assignment.
        assert_eq!(wrapped_array("[game.actions]\n"), None);
    }
}
