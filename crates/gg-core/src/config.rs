//! Config: the file and the command line, both of which only ever set CVars
//! (§4.8).
//!
//! The format is flat `name = value` lines, and flat is not a shortcut — a CVar
//! name *is* the path (`r.renderscale`), so nesting would be a second way to
//! spell the same thing, and a TOML/RON dependency would buy tables, arrays and
//! dates for a file that has none of them. §3's dependency budget is easier to
//! defend by not spending it here.
//!
//! ```text
//! # comments run to end of line
//! r.vsync = 0
//! r.renderscale = 0.75
//! ```
//!
//! **Precedence is call order**, and the shell applies defaults → the game's
//! file → the player's (§6 M42) → CLI, so a flag beats every file and a file
//! beats the declaration. Nothing here reads the environment: an engine that
//! silently obeys `GG_*` variables is an engine whose bug reports do not
//! reproduce.
//!
//! Nothing here decides what an unknown name means, either. [`Report`] carries
//! them back and the caller picks the policy — because "unknown cvar" is a typo
//! in a dev config and a stale line in a player's, and those two want opposite
//! answers.

use std::path::Path;

use crate::cvar::{self, CVarSource};

pub mod icon;
pub mod project;

/// The command-line form: `--set name=value`, repeatable.
pub const SET_FLAG: &str = "--set";

/// A setting that did not stick, with the reason already rendered — CVar errors
/// borrow `&'static str` names the report cannot promise, and a report exists to
/// be logged, not matched on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rejection {
    /// The cvar name, or the whole line when it was not a setting at all.
    pub name: String,
    /// What it was offered. Empty when the line had no value to offer.
    pub value: String,
    /// Why it did not stick, already rendered for a log line.
    pub reason: String,
}

/// What one `apply_*` call did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// CVars actually set.
    pub applied: usize,
    /// Names no CVar answers to. Separate from [`Self::rejected`] because a name
    /// nobody claims is usually a stale config line, while a bad value is
    /// usually a typo being made right now.
    pub unknown: Vec<String>,
    /// Names that exist and refused the value, and lines that were not settings
    /// at all.
    pub rejected: Vec<Rejection>,
}

impl Report {
    /// Nothing unknown and nothing rejected.
    pub fn is_clean(&self) -> bool {
        self.unknown.is_empty() && self.rejected.is_empty()
    }

    fn merge(&mut self, other: Report) {
        self.applied += other.applied;
        self.unknown.extend(other.unknown);
        self.rejected.extend(other.rejected);
    }

    /// One `tracing` line per problem, at the severity each deserves. What a
    /// shell that has no better policy should do with a report.
    pub fn log(&self, source: &str) {
        for name in &self.unknown {
            tracing::warn!(source, cvar = name, "no such cvar — line ignored");
        }
        for r in &self.rejected {
            tracing::error!(source, cvar = %r.name, value = %r.value, reason = %r.reason, "setting rejected");
        }
        tracing::debug!(source, applied = self.applied, "config applied");
    }
}

/// Apply the contents of a config file.
pub fn apply_str(text: &str) -> Report {
    apply_str_from(text, CVarSource::Config)
}

fn apply_str_from(text: &str, source: CVarSource) -> Report {
    let mut report = Report::default();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        match line.split_once('=') {
            Some((name, value)) => apply_one(name.trim(), value.trim(), source, &mut report),
            None => report.rejected.push(Rejection {
                name: line.to_owned(),
                value: String::new(),
                reason: "expected `name = value`".to_owned(),
            }),
        }
    }
    report
}

/// Apply a config file, if it is there.
///
/// A missing file is not an error and not a rejection: the config file is the
/// player's, and not having written one yet is the normal case. Anything else —
/// a directory, a permissions failure — is returned, because it means the file
/// the user *did* write was not read.
pub fn apply_file(path: &Path) -> std::io::Result<Report> {
    apply_file_from(path, CVarSource::Config)
}

fn apply_file_from(path: &Path, source: CVarSource) -> std::io::Result<Report> {
    match std::fs::read_to_string(path) {
        Ok(text) => Ok(apply_str_from(&text, source)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Report::default()),
        Err(e) => Err(e),
    }
}

/// Apply every `--set name=value` in an argument list, ignoring everything else —
/// the shell's other flags are its own business and are not errors here.
pub fn apply_args<I, S>(args: I) -> Report
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut report = Report::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg.as_ref() != SET_FLAG {
            continue;
        }
        let Some(setting) = args.next() else {
            report.rejected.push(Rejection {
                name: SET_FLAG.to_owned(),
                value: String::new(),
                reason: "trailing `--set` with nothing to set".to_owned(),
            });
            break;
        };
        match setting.as_ref().split_once('=') {
            Some((name, value)) => {
                apply_one(name.trim(), value.trim(), CVarSource::Cli, &mut report)
            }
            None => report.rejected.push(Rejection {
                name: setting.as_ref().to_owned(),
                value: String::new(),
                reason: "expected `--set name=value`".to_owned(),
            }),
        }
    }
    report
}

/// Defaults → file → CLI, in that order, as one report. The order *is* the
/// precedence: each pass overwrites what the last one set.
pub fn apply(path: &Path, args: &[String]) -> std::io::Result<Report> {
    let mut report = apply_file(path)?;
    report.merge(apply_args(args));
    Ok(report)
}

/// [`apply`], plus the two logs a shell owes §4.8: what each pass refused, and
/// then one line per CVar naming the source that won. Whole reason the shell's
/// share of the CVar system is a single call — precedence is a property of the
/// engine, not a thing each host re-decides.
///
/// Each pass logs under its own source name, because a stale line is a different
/// bug in a config file than in a flag typed thirty seconds ago.
///
/// `player` is the same format in the player's own directory (§6 M42), applied
/// **between** the two: the game's file is what shipped, the player's is what
/// they changed about it, and a flag is still someone at a keyboard right now.
/// `None` is every run without one — a dev run, and any run the caller decided
/// must not read it.
pub fn boot(path: &Path, player: Option<&Path>, args: &[String]) -> std::io::Result<Report> {
    let mut report = apply_file(path)?;
    report.log(CVarSource::Config.as_str());
    if let Some(path) = player {
        let player = apply_file_from(path, CVarSource::Player)?;
        player.log(CVarSource::Player.as_str());
        report.merge(player);
    }
    let cli = apply_args(args);
    cli.log(CVarSource::Cli.as_str());
    report.merge(cli);
    cvar::log_sources();
    Ok(report)
}

fn apply_one(name: &str, value: &str, source: CVarSource, report: &mut Report) {
    match cvar::set(name, value, source) {
        Ok(()) => report.applied += 1,
        Err(cvar::CVarError::Unknown { .. }) => report.unknown.push(name.to_owned()),
        Err(e) => report.rejected.push(Rejection {
            name: name.to_owned(),
            value: value.to_owned(),
            reason: e.to_string(),
        }),
    }
}
