//! What the editor can be opened over, and how it is found on disk (§6 M15.1
//! item 4).
//!
//! The launcher shows the ordinary editor with an empty world; this is the list
//! it offers instead of the game pane's hole, and the paths a host needs to
//! start a session over one. Here rather than in the shell for §3's reason —
//! `gg-runtime` is wiring and this is a directory convention — and rather than
//! in `xtask` because a shipped-shaped launcher cannot depend on the build tool.
//!
//! # The convention is `xtask run`'s
//!
//! Game crates live under `demos/` (§3's tree), a crate directory `05-many` is
//! package `demo-05-many` and dylib stem `demo_05_many`, its bindings are
//! `input.toml` beside its source (§4.7), and its pack is `target/assets/<dir>.ggpack`
//! (§4.6). Nothing here builds anything: a project whose dylib is not there is
//! listed and refuses to open, because "run cargo for me" is a decision about
//! what an editor is and not a line in a scan.

use std::path::{Path, PathBuf};

/// Where game crates live, relative to the root a scan is given (§3).
const GAMES: &str = "demos";

/// A game the editor can be opened over.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Project {
    /// The crate directory's name — `05-many` — which is what the picker shows
    /// and what a layout file is keyed by.
    pub name: String,
    /// The dylib a shell would be pointed at. Present whether or not it exists;
    /// [`Project::built`] is the question of whether it does.
    pub game: PathBuf,
    /// The action map beside its source, when it declared one (§4.7).
    pub input: Option<PathBuf>,
    /// The compiled pack, when the crate has an `assets/` tree (§4.6). The path
    /// `xtask assets` writes, which is build output and may not be there.
    pub pack: Option<PathBuf>,
    /// Whether [`Project::game`] is on disk. A project that has never been built
    /// is a real project someone has not run `cargo build` for — listed, so the
    /// answer to "where is my game" is the list and not an empty one.
    pub built: bool,
}

/// Every project under `root`, by name.
///
/// **Sorted**, and that is not presentation: `read_dir` order is the
/// filesystem's, so an unsorted list would make the picker's rows — and so a
/// recorded session's clicks — depend on the order a checkout happened to create
/// directories in. Never fails: an unreadable root is no projects, which is what
/// a launcher started outside a workspace should show.
#[must_use]
pub fn scan(root: &Path) -> Vec<Project> {
    let Ok(entries) = std::fs::read_dir(root.join(GAMES)) else {
        return Vec::new();
    };
    let mut found: Vec<Project> = entries
        .flatten()
        .filter(|entry| entry.path().join("Cargo.toml").is_file())
        .filter_map(|entry| {
            let dir = entry.path();
            let name = dir.file_name()?.to_str()?.to_owned();
            let game = root.join("target/debug").join(dylib(&name));
            let pack = root.join("target/assets").join(format!("{name}.ggpack"));
            Some(Project {
                built: game.is_file(),
                game,
                input: Some(dir.join("input.toml")).filter(|p| p.is_file()),
                pack: dir.join("assets").is_dir().then_some(pack),
                name,
            })
        })
        .collect();
    found.sort_by(|a, b| a.name.cmp(&b.name));
    found
}

/// The dylib file name a crate directory called `name` builds to — the
/// platform's own decoration around `xtask run`'s stem rule.
fn dylib(name: &str) -> String {
    let stem = format!("demo_{}", name.replace('-', "_"));
    if cfg!(windows) {
        format!("{stem}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{stem}.dylib")
    } else {
        format!("lib{stem}.so")
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// The workspace this crate is in — two levels up from `crates/gg-editor`.
    fn root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap()
    }

    #[test]
    fn the_scan_finds_this_workspace_own_demos_by_the_convention_xtask_uses() {
        // A real scan of the real tree, because the whole content of this module
        // is a claim about where things are — a scan tested against a fabricated
        // directory would pass while the convention it encodes had rotted.
        let found = scan(&root());
        let names: Vec<&str> = found.iter().map(|p| p.name.as_str()).collect();
        assert!(
            names.contains(&"05-many") && names.contains(&"03-reload"),
            "the scan did not find the demos: {names:?}"
        );
        assert!(
            names.windows(2).all(|w| w[0] < w[1]),
            "the list is not sorted, so a picker's rows are the filesystem's order: {names:?}"
        );
        let many = found.iter().find(|p| p.name == "05-many").unwrap();
        assert!(
            many.game.ends_with(dylib("05-many")),
            "{}",
            many.game.display()
        );
        assert!(
            many.input.is_some(),
            "demo 05 declares an action map (§4.7)"
        );
    }

    #[test]
    fn a_root_with_no_games_under_it_is_no_projects_and_not_an_error() {
        // What a launcher started from a home directory sees. An error here would
        // make "opened outside a workspace" a failed run rather than an empty
        // list, which is the case item 4 exists to make ordinary.
        assert!(scan(Path::new("/definitely/not/a/workspace")).is_empty());
    }

    #[test]
    fn the_dylib_name_is_the_stem_rule_with_the_platform_decoration() {
        let name = dylib("09-orbit");
        assert!(name.contains("demo_09_orbit"), "{name}");
        assert!(
            !name.contains('-'),
            "a crate name is not a library name: {name}"
        );
    }
}
