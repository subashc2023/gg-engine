//! `cargo xtask backlog` — the P1/P2 items, found rather than listed (§6 M12).
//!
//! M12 asks for the backlog "triaged into issues rather than into a second plan
//! document". There is no issue tracker on this desk, and the honest answer is
//! not to invent a `BACKLOG.md` — that is precisely the second plan document,
//! and §1's rule about second copies applies: it would be a list of intentions
//! kept somewhere other than where the intention is, drifting from the day it
//! was written.
//!
//! So the backlog is a **query over the source**. Every deferred item already
//! sits in a doc comment next to the code that defers it, because that is where
//! a reader needs it; this command collects them. Adding an item is writing the
//! sentence where it belongs, and closing one is deleting it — neither is a
//! bookkeeping step anybody can forget.
//!
//! # Why only doc comments
//!
//! `P1` and `P2` are ordinary tokens. Demo 02's BC7 encoder writes `// P1` for a
//! *partition index*, which is not a backlog item and never will be. Requiring
//! `///` or `//!` is what separates "this is deferred work" from "this is a
//! field called P1", without either file needing to know about this one.

use std::path::Path;

use crate::util::{walk_rs, workspace_root};

/// One deferred item: where it is, and the sentence that defers it.
struct Item {
    priority: &'static str,
    location: String,
    text: String,
}

pub fn run() -> anyhow::Result<()> {
    let root = workspace_root();
    let mut files = Vec::new();
    for dir in ["crates", "demos", "tools", "xtask"] {
        walk_rs(&root.join(dir), &mut files);
    }
    files.sort();

    let mut items = Vec::new();
    for path in &files {
        // This file is *about* the markers; scanning it would list its own prose.
        if path.ends_with("backlog.rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(path) else {
            continue;
        };
        let shown = path.strip_prefix(&root).unwrap_or(path);
        collect(shown, &text, &mut items);
    }

    for priority in ["P1", "P2"] {
        let matching: Vec<&Item> = items.iter().filter(|i| i.priority == priority).collect();
        println!("\n{priority} — {} item(s)", matching.len());
        for item in matching {
            println!("  {}\n      {}", item.location, item.text);
        }
    }
    println!(
        "\nxtask backlog: {} item(s) across {} file(s). They live in the doc comments beside \
         what defers them (§6 M12) — closing one is deleting a sentence, not updating a list.",
        items.len(),
        files.len()
    );
    Ok(())
}

/// Pull every doc-comment line naming `P1`/`P2` out of one file, with the
/// sentence around it.
fn collect(path: &Path, text: &str, out: &mut Vec<Item>) {
    let lines: Vec<&str> = text.lines().collect();
    for (index, raw) in lines.iter().enumerate() {
        let line = raw.trim_start();
        let Some(body) = doc_body(line) else {
            continue;
        };
        let Some(priority) = marker(body) else {
            continue;
        };
        // The whole paragraph, not the matching line: these sentences wrap, and
        // a marker landing on a continuation line would otherwise read as
        // "is P1." — true, and useless.
        let (mut first, mut last) = (index, index);
        while first > 0 && doc_body(lines[first - 1]).is_some_and(|b| !b.trim().is_empty()) {
            first -= 1;
        }
        while last + 1 < lines.len()
            && doc_body(lines[last + 1]).is_some_and(|b| !b.trim().is_empty())
        {
            last += 1;
        }
        let text = lines[first..=last]
            .iter()
            .filter_map(|l| doc_body(l))
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(" ");
        // Duplicate paragraphs: one paragraph naming P1 twice is one item.
        let location = format!(
            "{}:{}",
            path.display().to_string().replace('\\', "/"),
            first + 1
        );
        if out.iter().any(|i: &Item| i.location == location) {
            continue;
        }
        out.push(Item {
            priority,
            location,
            text,
        });
    }
}

/// The text of a `///` or `//!` line, or `None` for anything else.
fn doc_body(line: &str) -> Option<&str> {
    let line = line.trim_start();
    line.strip_prefix("///")
        .or_else(|| line.strip_prefix("//!"))
}

/// `P1`/`P2` as a whole token — `P1X` and `sP2` are not markers.
fn marker(body: &str) -> Option<&'static str> {
    for (needle, priority) in [("P1", "P1"), ("P2", "P2")] {
        for (at, _) in body.match_indices(needle) {
            let before = body[..at].chars().next_back();
            let after = body[at + needle.len()..].chars().next();
            let bounded = |c: Option<char>| c.is_none_or(|c| !c.is_alphanumeric() && c != '_');
            if bounded(before) && bounded(after) {
                return Some(priority);
            }
        }
    }
    None
}
