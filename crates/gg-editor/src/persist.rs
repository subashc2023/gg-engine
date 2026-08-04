//! The editor's own files: where a session keeps them, and the dock layout
//! written down and read back (§6 M15.1).
//!
//! Here rather than in the host because they are the *editor's* conventions —
//! a shell chooses to have an editor and chooses whether this run may remember
//! anything (§3), and choosing neither the directory nor the format is what
//! keeps it thin.
//!
//! Prefix notation over whitespace-separated tokens, so a parser is an iterator
//! and a recursive call rather than a grammar:
//!
//! ```text
//! split h 0.234 tabs 0 1 world split h 0.65 split v 0.75 tabs 0 1 game …
//! ```
//!
//! Panes are written by **name**, never by index, for the reason every other
//! format in this engine is name-keyed (§4.2.1, §4.6): a build that inserts a
//! pane renumbers the indices, and a layout file that survived that would put
//! the inspector where the asset browser used to be. An unknown name fails the
//! parse, the caller keeps the default, and nothing silently lands wrong.
//!
//! # Not a replay input
//!
//! A host loads this only when it is neither recording nor replaying (§6 M15.1).
//! The layout is hit-testing, so a session that started from a file the gate
//! never saw would land its clicks somewhere else — the default tree is the one
//! fixed starting point every recorded session shares.

use std::path::{Path, PathBuf};

use tracing::{info, warn};

use crate::{Editor, Pane};
use gg_ui::dock::Node;
use gg_ui::layout::Axis;

/// Where a session's files live, keyed by the game's file stem. Under `target/`
/// because they are a *working* copy of a session and not an artifact: this
/// crate is dev-only by §3's deny pin, so nothing here survives a `cargo clean`
/// and nothing here should.
const DIR: &str = "target/editor";

/// Where the save button writes when the host named no target (`--save`).
#[must_use]
pub fn save_path(stem: &str) -> PathBuf {
    // Appended and not `with_extension`, which would eat everything after a dot
    // in a stem that has one.
    Path::new(DIR).join(format!("{stem}.ggsv"))
}

/// Where the dock layout is remembered between sessions.
///
/// A host asks only for a run that may have one — neither recording nor
/// replaying, for the reason under *Not a replay input* above.
#[must_use]
pub fn layout_path(stem: &str) -> PathBuf {
    Path::new(DIR).join(format!("{stem}.layout"))
}

/// Restore `editor`'s layout from `path`, if there is one this build can read.
///
/// Total by construction — every way it can fail leaves the default tree, which
/// is the only outcome a caller could act on anyway. No file is the ordinary
/// first run and says nothing; a file that does not parse, or parses to a tree
/// missing a pane, is a *changed* build meeting an old layout and logs so.
pub fn restore(editor: &mut Editor, path: &Path) {
    let Some(root) = std::fs::read_to_string(path).ok().and_then(|t| decode(&t)) else {
        return;
    };
    match editor.set_layout(root) {
        true => info!(path = %path.display(), "editor: layout restored"),
        false => warn!(path = %path.display(), "editor: layout refused; default kept"),
    }
}

/// Write `editor`'s layout to `path`, making the directory if it is missing.
///
/// Also total, and for a stronger reason than [`restore`]: this runs on the way
/// out of a session that has already done everything it was asked for. A layout
/// nobody could write is a worse session next time, not a failed run.
pub fn remember(editor: &Editor, path: &Path) {
    let written = path
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| std::fs::write(path, encode(editor.layout())));
    match written {
        Ok(()) => info!(path = %path.display(), "editor: layout remembered"),
        Err(error) => warn!(%error, path = %path.display(), "editor: layout not written"),
    }
}

/// Write `node` as one line.
#[must_use]
pub fn encode(node: &Node) -> String {
    let mut out = String::new();
    write(node, &mut out);
    out
}

fn write(node: &Node, out: &mut String) {
    if !out.is_empty() {
        out.push(' ');
    }
    match node {
        Node::Split {
            axis,
            at,
            first,
            second,
        } => {
            let axis = match axis {
                Axis::Horizontal => 'h',
                Axis::Vertical => 'v',
            };
            // Four decimals: a seam is placed in whole pixels on any surface
            // anyone has, and a full-precision float would make the file churn
            // on every drag by an amount no one can see.
            out.push_str(&format!("split {axis} {at:.4}"));
            write(first, out);
            write(second, out);
        }
        Node::Tabs { panes, active } => {
            out.push_str(&format!("tabs {active} {}", panes.len()));
            for pane in panes {
                out.push(' ');
                out.push_str(Pane::from_id(*pane).map_or("?", Pane::title));
            }
        }
    }
}

/// Read a layout back, or `None` if the text is not one this build understands.
///
/// Never partially applied: a tree is returned whole or not at all, so a
/// truncated file cannot leave half a layout behind.
#[must_use]
pub fn decode(text: &str) -> Option<Node> {
    let mut tokens = text.split_whitespace();
    let node = read(&mut tokens)?;
    // Trailing tokens mean the file says more than one tree, which is a file
    // written by something else.
    tokens.next().is_none().then_some(node)
}

fn read<'a>(tokens: &mut impl Iterator<Item = &'a str>) -> Option<Node> {
    match tokens.next()? {
        "split" => {
            let axis = match tokens.next()? {
                "h" => Axis::Horizontal,
                "v" => Axis::Vertical,
                _ => return None,
            };
            let at: f32 = tokens.next()?.parse().ok()?;
            // Refused rather than clamped: a fraction outside the unit interval
            // is a corrupt file, and clamping it would hide that.
            if !(0.0..=1.0).contains(&at) {
                return None;
            }
            let first = read(tokens)?;
            let second = read(tokens)?;
            Some(Node::split(axis, at, first, second))
        }
        "tabs" => {
            let active: usize = tokens.next()?.parse().ok()?;
            let count: usize = tokens.next()?.parse().ok()?;
            // A group with no panes cannot be resolved and would be pruned the
            // first time anything moved; refuse it here where it is still one
            // line of a file.
            if count == 0 || count > Pane::ALL.len() || active >= count {
                return None;
            }
            let mut panes = Vec::with_capacity(count);
            for _ in 0..count {
                let name = tokens.next()?;
                panes.push(Pane::ALL.iter().find(|p| p.title() == name)?.id());
            }
            Some(Node::Tabs { panes, active })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// The property the file exists for: what the editor was holding is what it
    /// gets back.
    #[test]
    fn a_layout_round_trips_through_the_text() {
        let tree = crate::default_layout();
        let text = encode(&tree);
        assert_eq!(decode(&text), Some(tree), "{text}");
        // And the text is one line, because a layout is one line.
        assert!(!text.contains('\n'));
    }

    /// A layout the operator rearranged round-trips too — the case that is not
    /// the default and therefore the case worth persisting at all.
    #[test]
    fn a_rearranged_layout_round_trips() {
        let mut editor = crate::Editor::new(None);
        editor.place((1280, 720), 1.0);
        assert!(editor.set_layout(
            decode("split v 0.5 tabs 1 3 world game cvars tabs 0 3 assets perf inspect").unwrap()
        ));
        let text = encode(editor.layout());
        assert_eq!(decode(&text).as_ref(), Some(editor.layout()));
    }

    /// Every way a file can be wrong ends at the default, not at a half-applied
    /// tree or a panic.
    #[test]
    fn a_file_this_build_cannot_read_is_refused_whole() {
        for bad in [
            "",
            "tabs",
            "tabs 0 1",
            "tabs 0 1 nosuchpane",
            "tabs 3 1 world",
            "tabs 0 0",
            "tabs 0 99 world",
            "split h 0.5 tabs 0 1 world",
            "split d 0.5 tabs 0 1 world tabs 0 1 game",
            "split h 1.5 tabs 0 1 world tabs 0 1 game",
            "split h nan tabs 0 1 world tabs 0 1 game",
            "tabs 0 1 world tabs 0 1 game",
            "{\"root\": []}",
        ] {
            assert_eq!(decode(bad), None, "{bad:?} was accepted");
        }
    }

    /// The pair as a host uses it: what one session left is what the next one
    /// opens with, through the file and not through a value handed across.
    #[test]
    fn a_layout_survives_the_file() {
        let path = std::path::Path::new("target/gg-editor-tests/survives.layout");
        let mut first = Editor::new(None);
        first.place((1280, 720), 1.0);
        assert!(first.set_layout(
            decode("split v 0.5 tabs 1 3 world game cvars tabs 0 3 assets perf inspect").unwrap()
        ));
        remember(&first, path);

        let mut second = Editor::new(None);
        second.place((1280, 720), 1.0);
        restore(&mut second, path);
        assert_eq!(second.layout(), first.layout());
    }

    /// Both halves are total: a missing file and an unreadable one leave the
    /// default, because a host has nothing better to do with either and an
    /// editor that refuses to open over a stale layout is worse than one that
    /// forgets it.
    #[test]
    fn a_file_that_is_missing_or_wrong_leaves_the_default() {
        let dir = std::path::Path::new("target/gg-editor-tests");
        let mut editor = Editor::new(None);
        editor.place((1280, 720), 1.0);
        let default = editor.layout().clone();

        restore(&mut editor, &dir.join("no-such.layout"));
        assert_eq!(editor.layout(), &default);

        let bad = dir.join("bad.layout");
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(&bad, "tabs 0 1 nosuchpane").unwrap();
        restore(&mut editor, &bad);
        assert_eq!(editor.layout(), &default);
    }

    /// The names are the pane titles, so a file is legible and a renamed pane
    /// is a refused file rather than a silently different layout.
    #[test]
    fn panes_are_written_by_name() {
        let text = encode(&crate::default_layout());
        for pane in Pane::ALL {
            assert!(text.contains(pane.title()), "{} is missing", pane.title());
        }
    }
}
