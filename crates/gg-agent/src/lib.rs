//! §6 M16: the reload seam's outcome as a value.
//!
//! §1's loop is play → describe → edit → reload → keep playing, and the step an
//! agent cannot see is the middle one. A refusal reaching `error!` is legible to
//! a human watching a terminal and invisible to anything else; a migration that
//! zeroed a field is one `info!` per component in a stream of them. What M16
//! builds is the same information as a *record* — which build, which tick, which
//! hash either side, and what the refusal was called — so a fix can be checked
//! against the failure instead of against a description of it.
//!
//! # No engine dependencies, on purpose
//!
//! Nothing here knows what a `World` or a `ReloadError` is. The shell owns the
//! conversion, and this crate owns the shape, because the *reader* is
//! out-of-process: an agent asking what just happened must not link the ECS to
//! find out. That also makes the record the one artifact of §1's loop that
//! survives the process that produced it.
//!
//! # Lab equipment (§3)
//!
//! Optional in the shell, banned from the dist graph, and named in the dist
//! gate's absent list beside `gg-debug` and `gg-editor`. A shipped game has no
//! reload seam to report on.

pub mod ask;
mod json;

pub use ask::{Ask, Event, Status};

use std::path::{Path, PathBuf};

/// Seams kept before the oldest is dropped.
///
/// A session is hundreds of reloads (§4.2.2's leak budget is sized for exactly
/// that), and an agent asked to explain the last failure wants the last few. A
/// bound rather than a file that grows all session, because the record is
/// rewritten in full on every publish.
const KEPT_SEAMS: usize = 16;

/// What the host did with a rebuilt dylib.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Swapped in. The game is running the new code as of [`Seam::tick`].
    Accepted,
    /// Refused; the last good build is still playing.
    Refused {
        /// The refusal's *variant* name — `HostApiMismatch`, `Adoption`, …
        ///
        /// M16's exit row asks that a refusal surface "by name", and this is the
        /// name: a stable token a panel can branch on and an agent can match,
        /// where `detail` is prose that names paths and numbers and is not.
        kind: &'static str,
        /// The refusal rendered, which is what the log line carries too.
        detail: String,
    },
}

/// What a restore did to one component type, flattened out of `gg-ecs`'
/// `ComponentOutcome` — the fields, not the type, so the reader links nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Change {
    /// The component's declared id.
    pub component: String,
    /// `reused`, `migrated`, or `dropped`.
    pub kind: &'static str,
    /// Present in the new build only, zeroed.
    pub defaulted: Vec<String>,
    /// Present in both under different types, zeroed — the case where data was
    /// *lost* rather than never present, and the one an agent should be told
    /// about without being asked.
    pub retyped: Vec<String>,
}

/// One crossing of the §4.2.2 seam.
#[derive(Clone, Debug)]
pub struct Seam {
    /// The first tick the new code runs at, which is also where the replay
    /// segment opened — so a recording and this record agree (§4.7).
    pub tick: u64,
    /// Accepted, or refused and why.
    pub outcome: Outcome,
    /// The retired build's code hash, hex. What a replay segment is named by.
    pub code_before: String,
    /// The arriving build's, or `None` where verification refused the artifact
    /// before it was opened and there is no second build to name.
    ///
    /// Unknown rather than empty, for the reason `ReloadError::Unreadable` gives
    /// about a zero byte count: a default that reads as a value makes two
    /// different facts indistinguishable at the reader.
    pub code_after: Option<String>,
    /// Canonical state hash before the swap (§4.2.1).
    pub state_before: String,
    /// After the migration. On a refusal this equals `state_before`: nothing was
    /// swapped, so nothing moved.
    pub state_after: String,
    /// Live entities the restore brought back.
    pub entities: u32,
    /// Per-component, and only the ones that moved — a clean reload is the
    /// majority of every session and listing it would bury the one line the
    /// reader came for (§4.2.2).
    pub changes: Vec<Change>,
    /// Mapping the artifact, in milliseconds. `None` on the refusals that
    /// happened before there was an artifact to time.
    pub load_ms: Option<u64>,
    /// §9's two-second bar, measured end to end: file event to the tick boundary
    /// the new code first runs at. The number the milestone is graded on, and
    /// `None` where no tick boundary was ever reached.
    pub save_to_swap_ms: Option<u64>,
}

impl Seam {
    /// Whether this crossing met §9's bar, or `None` if the question does not
    /// apply — a refusal has no save-to-*behaviour* time, because there was no
    /// new behaviour. Stated here rather than at each reader so a panel and an
    /// agent cannot disagree about either the threshold or when it is meaningful.
    #[must_use]
    pub fn within_budget(&self) -> Option<bool> {
        match self.outcome {
            Outcome::Accepted => self.save_to_swap_ms.map(|ms| ms <= SAVE_TO_BEHAVIOUR_MS),
            Outcome::Refused { .. } => None,
        }
    }
}

/// §9's save-to-behaviour bar, in milliseconds.
pub const SAVE_TO_BEHAVIOUR_MS: u64 = 2_000;

/// The running session, as an agent sees it.
///
/// Published in full on every change rather than appended to: a reader that
/// opened the file mid-append would see a truncated record, and the fix for that
/// is [`Journal::publish`]'s rename, which only works on a whole file.
#[derive(Clone, Debug)]
pub struct Journal {
    /// The game dylib's name, as the shell reports it.
    game: String,
    /// Which tier this shell is, so an agent reading a record knows whether the
    /// thing it is about to suggest is even compiled in.
    tier: String,
    /// This shell's process id. One published file per directory, so a second
    /// concurrent session overwrites the first — the pid is how a reader tells
    /// whose record it is holding rather than assuming.
    pid: u32,
    /// The sim tick at the last publish.
    tick: u64,
    /// The recording this session is writing, if any (§4.7) — "the replay of
    /// what the human just did", which is the third thing M16 hands an agent.
    recording: Option<PathBuf>,
    /// Most recent last, capped at [`KEPT_SEAMS`].
    seams: Vec<Seam>,
}

impl Journal {
    /// A journal for a session over `game`, built in `tier`.
    #[must_use]
    pub fn new(game: impl Into<String>, tier: impl Into<String>) -> Journal {
        Journal {
            game: game.into(),
            tier: tier.into(),
            pid: std::process::id(),
            tick: 0,
            recording: None,
            seams: Vec::new(),
        }
    }

    /// Name the recording this session is writing (§4.7).
    pub fn recording(&mut self, path: impl Into<PathBuf>) {
        self.recording = Some(path.into());
    }

    /// Advance the tick the next publish reports.
    pub fn at_tick(&mut self, tick: u64) {
        self.tick = tick;
    }

    /// Record a crossing, dropping the oldest past [`KEPT_SEAMS`].
    pub fn record(&mut self, seam: Seam) {
        if self.seams.len() == KEPT_SEAMS {
            self.seams.remove(0);
        }
        self.seams.push(seam);
    }

    /// The seams, oldest first.
    #[must_use]
    pub fn seams(&self) -> &[Seam] {
        &self.seams
    }

    /// The most recent crossing, which is what a panel shows and what an agent
    /// asked "what just happened" is answered with.
    #[must_use]
    pub fn last(&self) -> Option<&Seam> {
        self.seams.last()
    }

    /// Write the record under `dir` as `session.json`.
    ///
    /// Written to a sibling temp file and renamed, so a reader either sees the
    /// previous record or this one and never a half-written file — the reader is
    /// a separate process polling on its own clock, so torn reads are the normal
    /// case rather than a race worth ignoring.
    ///
    /// # Errors
    ///
    /// If `dir` cannot be created, or the write or rename fails.
    pub fn publish(&self, dir: &Path) -> std::io::Result<PathBuf> {
        std::fs::create_dir_all(dir)?;
        let final_path = dir.join("session.json");
        // Pid-keyed so two shells publishing at once cannot rename each other's
        // temp file out from under themselves. They still overwrite the same
        // final path, which is what `pid` in the record is for.
        let temp = dir.join(format!("session.{}.tmp", self.pid));
        std::fs::write(&temp, self.to_json())?;
        std::fs::rename(&temp, &final_path)?;
        Ok(final_path)
    }

    /// The record as JSON.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\n  \"game\": ");
        push_str_json(&mut out, &self.game);
        out.push_str(",\n  \"tier\": ");
        push_str_json(&mut out, &self.tier);
        out.push_str(",\n  \"pid\": ");
        out.push_str(&self.pid.to_string());
        out.push_str(",\n  \"tick\": ");
        out.push_str(&self.tick.to_string());
        out.push_str(",\n  \"recording\": ");
        match &self.recording {
            // The absence is a fact an agent needs: no recording means "the
            // replay of what the human just did" is not available to verify a
            // fix against, and it should say so rather than look for a file.
            None => out.push_str("null"),
            Some(path) => push_str_json(&mut out, &path.display().to_string()),
        }
        out.push_str(",\n  \"seams\": [");
        for (i, seam) in self.seams.iter().enumerate() {
            out.push_str(if i == 0 { "\n" } else { ",\n" });
            push_seam_json(&mut out, seam);
        }
        if !self.seams.is_empty() {
            out.push_str("\n  ");
        }
        out.push_str("]\n}\n");
        out
    }
}

fn push_seam_json(out: &mut String, seam: &Seam) {
    out.push_str("    {\n      \"tick\": ");
    out.push_str(&seam.tick.to_string());
    out.push_str(",\n      \"outcome\": ");
    match &seam.outcome {
        Outcome::Accepted => out.push_str("\"accepted\""),
        Outcome::Refused { .. } => out.push_str("\"refused\""),
    }
    if let Outcome::Refused { kind, detail } = &seam.outcome {
        out.push_str(",\n      \"refusal\": ");
        push_str_json(out, kind);
        out.push_str(",\n      \"detail\": ");
        push_str_json(out, detail);
    }
    for (key, value) in [
        ("code_before", Some(&seam.code_before)),
        ("code_after", seam.code_after.as_ref()),
        ("state_before", Some(&seam.state_before)),
        ("state_after", Some(&seam.state_after)),
    ] {
        out.push_str(",\n      \"");
        out.push_str(key);
        out.push_str("\": ");
        match value {
            Some(v) => push_str_json(out, v),
            None => out.push_str("null"),
        }
    }
    out.push_str(",\n      \"entities\": ");
    out.push_str(&seam.entities.to_string());
    for (key, value) in [
        ("load_ms", seam.load_ms),
        ("save_to_swap_ms", seam.save_to_swap_ms),
    ] {
        out.push_str(",\n      \"");
        out.push_str(key);
        out.push_str("\": ");
        out.push_str(&value.map_or_else(|| "null".to_owned(), |v| v.to_string()));
    }
    // `null` and not `false`, for the same reason: `false` reads as "it was
    // slow" where the truth is that nothing arrived to be timed.
    out.push_str(",\n      \"within_budget\": ");
    out.push_str(match seam.within_budget() {
        Some(true) => "true",
        Some(false) => "false",
        None => "null",
    });
    out.push_str(",\n      \"changes\": [");
    for (i, change) in seam.changes.iter().enumerate() {
        out.push_str(if i == 0 { "\n" } else { ",\n" });
        out.push_str("        {\"component\": ");
        push_str_json(out, &change.component);
        out.push_str(", \"kind\": ");
        push_str_json(out, change.kind);
        out.push_str(", \"defaulted\": ");
        push_list_json(out, &change.defaulted);
        out.push_str(", \"retyped\": ");
        push_list_json(out, &change.retyped);
        out.push('}');
    }
    if !seam.changes.is_empty() {
        out.push_str("\n      ");
    }
    out.push_str("]\n    }");
}

fn push_list_json(out: &mut String, items: &[String]) {
    out.push('[');
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        push_str_json(out, item);
    }
    out.push(']');
}

/// A JSON string literal. The backslash case is not theoretical here — every
/// path this record carries is a Windows one on the desk, and an unescaped
/// `C:\dev` is a parse error at the reader rather than a wrong value.
fn push_str_json(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            // JSON forbids raw control characters in a string, and a refusal's
            // `detail` is arbitrary prose from an OS error message.
            c if (c as u32) < 0x20 => {
                out.push_str("\\u");
                for shift in [12, 8, 4, 0] {
                    let nibble = (c as u32 >> shift) & 0xf;
                    out.push(char::from_digit(nibble, 16).unwrap_or('0'));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn seam(tick: u64, outcome: Outcome) -> Seam {
        Seam {
            tick,
            outcome,
            code_before: "aa".into(),
            code_after: Some("bb".into()),
            state_before: "11".into(),
            state_after: "22".into(),
            entities: 3,
            changes: Vec::new(),
            load_ms: Some(4),
            save_to_swap_ms: Some(5),
        }
    }

    #[test]
    fn the_ring_keeps_the_most_recent_and_drops_the_oldest() {
        let mut journal = Journal::new("demo-03-reload", "tier-dev");
        for tick in 0..(KEPT_SEAMS as u64 + 5) {
            journal.record(seam(tick, Outcome::Accepted));
        }
        assert_eq!(journal.seams().len(), KEPT_SEAMS);
        assert_eq!(journal.seams()[0].tick, 5);
        assert_eq!(journal.last().map(|s| s.tick), Some(KEPT_SEAMS as u64 + 4));
    }

    /// The exit row is "by name", so the name has to survive into the record
    /// separately from the prose — a reader that had to regex the detail string
    /// would be reading a log line again with extra steps.
    #[test]
    fn a_refusal_carries_its_variant_name_beside_its_prose() {
        let mut journal = Journal::new("demo-03-reload", "tier-dev");
        journal.record(seam(
            9,
            Outcome::Refused {
                kind: "HostApiMismatch",
                detail: "`a.dll` was built against host API v3".into(),
            },
        ));
        let json = journal.to_json();
        assert!(json.contains("\"outcome\": \"refused\""), "{json}");
        assert!(json.contains("\"refusal\": \"HostApiMismatch\""), "{json}");
        assert!(json.contains("host API v3"), "{json}");
    }

    /// Every path in this record is a Windows path on the desk.
    #[test]
    fn a_backslash_path_survives_as_an_escape_rather_than_a_parse_error() {
        let mut journal = Journal::new("demo-03-reload", "tier-dev");
        journal.recording(PathBuf::from(r"C:\dev\GGEngine\out.ggrp"));
        let json = journal.to_json();
        assert!(json.contains(r"C:\\dev\\GGEngine\\out.ggrp"), "{json}");
        assert!(!json.contains(r"C:\dev"), "{json}");
    }

    #[test]
    fn a_control_character_in_a_refusal_is_escaped() {
        let mut out = String::new();
        push_str_json(&mut out, "a\u{1}b\"c");
        assert_eq!(out, r#""a\u0001b\"c""#);
    }

    /// §9's bar is stated once, here, so a panel and an agent cannot disagree.
    #[test]
    fn the_two_second_bar_is_judged_on_the_record_and_not_at_the_reader() {
        let mut slow = seam(1, Outcome::Accepted);
        assert_eq!(slow.within_budget(), Some(true));
        slow.save_to_swap_ms = Some(SAVE_TO_BEHAVIOUR_MS + 1);
        assert_eq!(slow.within_budget(), Some(false));
        let mut journal = Journal::new("g", "tier-dev");
        journal.record(slow);
        assert!(journal.to_json().contains("\"within_budget\": false"));
    }

    /// Zero is a measurement and `null` is the absence of one. A refusal that
    /// never reached a tick boundary has no save-to-*behaviour* time, and a
    /// record reporting `0` would let a reader conclude the fastest reload of
    /// the session was the one that did not happen.
    #[test]
    fn a_refusal_reports_unknown_timings_rather_than_zero_ones() {
        let mut unopened = seam(
            7,
            Outcome::Refused {
                kind: "Open",
                detail: "LoadLibraryExW failed".into(),
            },
        );
        unopened.code_after = None;
        unopened.load_ms = None;
        unopened.save_to_swap_ms = None;
        assert_eq!(unopened.within_budget(), None);
        let mut journal = Journal::new("g", "tier-dev");
        journal.record(unopened);
        let json = journal.to_json();
        for absent in [
            "\"code_after\": null",
            "\"load_ms\": null",
            "\"save_to_swap_ms\": null",
            "\"within_budget\": null",
        ] {
            assert!(json.contains(absent), "{json}");
        }
    }

    /// A refusal that *did* get far enough to be timed still declines the
    /// budget question: §9's bar is about new behaviour arriving, and none did.
    #[test]
    fn a_timed_refusal_is_still_not_judged_against_the_behaviour_bar() {
        let refused = seam(
            8,
            Outcome::Refused {
                kind: "Adoption",
                detail: "loaded but could not be adopted".into(),
            },
        );
        assert_eq!(refused.save_to_swap_ms, Some(5));
        assert_eq!(refused.within_budget(), None);
    }

    /// The rename is the whole point of `publish`: a reader polls on its own
    /// clock and must never open a half-written record.
    #[test]
    fn publishing_leaves_one_whole_file_and_no_temp_behind() {
        let dir = std::env::temp_dir().join(format!("gg-agent-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut journal = Journal::new("demo-03-reload", "tier-dev");
        journal.at_tick(412);
        journal.record(seam(412, Outcome::Accepted));
        let path = journal.publish(&dir).expect("publish");
        let text = std::fs::read_to_string(&path).expect("read");
        assert!(text.starts_with('{') && text.ends_with("}\n"), "{text}");
        assert!(text.contains("\"tick\": 412"), "{text}");
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .expect("dir")
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
        // Republishing over an existing record must succeed — rename onto an
        // existing file is the case Windows is stricter about than POSIX.
        journal.at_tick(413);
        journal.publish(&dir).expect("republish");
        assert!(
            std::fs::read_to_string(&path)
                .expect("reread")
                .contains("\"tick\": 413")
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_session_with_no_recording_says_so_rather_than_omitting_the_key() {
        let json = Journal::new("demo-03-reload", "tier-dev").to_json();
        assert!(json.contains("\"recording\": null"), "{json}");
        assert!(json.contains("\"seams\": []"), "{json}");
    }
}
