//! The CVar console (§4.8) — the third of the three front doors, and the only
//! one that answers back.
//!
//! Parsing and the reply live here; drawing the input line is the overlay's.
//! Splitting it that way is what lets the command language be tested without a
//! window, and it is the reason this file has no idea what a glyph is.
//!
//! The language is deliberately four things wide: `name` reads, `name value`
//! sets, `list [prefix]` enumerates, `reset name` restores the declaration. No
//! expressions, no history substitution, no aliases — a console that needs its
//! own manual is a second language to be wrong in at 3 a.m.

use gg_core::cvar::{self, CVarSource};

/// What a command produced: lines to show, in order. Never a `Result` — every
/// outcome here is text for a human, including the failures.
pub fn run(line: &str) -> Vec<String> {
    let line = line.trim();
    if line.is_empty() {
        return Vec::new();
    }
    let mut words = line.split_whitespace();
    let Some(head) = words.next() else {
        return Vec::new();
    };
    let rest: Vec<&str> = words.collect();
    match (head, rest.as_slice()) {
        ("list", prefix) => list(prefix.first().copied().unwrap_or("")),
        ("reset", [name]) => match cvar::find(name) {
            Some(cvar) => {
                cvar.reset();
                vec![format!("{name} = {} (default)", cvar.to_text())]
            }
            None => vec![unknown(name)],
        },
        ("help", [name]) => match cvar::find(name) {
            Some(cvar) => vec![format!(
                "{name} ({}) — {}",
                cvar.kind().as_str(),
                cvar.help()
            )],
            None => vec![unknown(name)],
        },
        // A bare name reads; a name and a value sets. Anything longer is a
        // typo, and guessing which half was meant is how a console sets the
        // wrong knob.
        (name, []) => match cvar::find(name) {
            Some(cvar) => vec![format!(
                "{name} = {} ({})",
                cvar.to_text(),
                cvar.source().as_str()
            )],
            None => vec![unknown(name)],
        },
        (name, [value]) => match cvar::set(name, value, CVarSource::Console) {
            Ok(()) => vec![format!("{name} = {value}")],
            Err(e) => vec![e.to_string()],
        },
        (name, _) => vec![format!(
            "`{name}` takes one value; quote nothing, a cvar is a number or a flag"
        )],
    }
}

/// Every CVar whose name starts with `prefix`, with its value and who set it.
/// Sorted, because the registry is (§4.2.1 hazard 3) and a listing that moved
/// between runs would be unreadable.
fn list(prefix: &str) -> Vec<String> {
    let mut lines: Vec<String> = cvar::all()
        .iter()
        .filter(|c| c.name().starts_with(prefix))
        .map(|c| {
            format!(
                "{} = {} ({}) — {}",
                c.name(),
                c.to_text(),
                c.source().as_str(),
                c.help()
            )
        })
        .collect();
    if lines.is_empty() {
        lines.push(format!("no cvar starts with `{prefix}`"));
    }
    lines
}

fn unknown(name: &str) -> String {
    format!("no cvar named `{name}`")
}
