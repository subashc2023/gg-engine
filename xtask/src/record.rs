//! What a scheduled tier leaves behind (§6 M82).
//!
//! Until this milestone the record was one word in one file, written by the
//! *scheduler's* shell (`… && echo ok || echo RED`), and that spelling cannot
//! represent the two states this desk actually produces. A launch refused for
//! not being idle runs no command at all, and a run killed by `StopOnIdleEnd`
//! runs neither arm — so in both cases the file keeps whatever the last run
//! that *completed* put there, with no time on it and no tree.
//!
//! So the tier writes its own record, on M48's argument: whatever is on the
//! disk afterwards was put there during the run or not at all. A `started`
//! line at entry and a `finished` line at exit, appended — which makes a start
//! with no finish a **kill** rather than a stale pass (or a run still in
//! progress, inside [`RUN_LIMIT`]: `ci::nightly` calls `ci::push`, which
//! reports, so a live tier reads its own start seconds after writing it), and
//! nothing at all where the schedule says there should be something a **stale**
//! tier rather than a green one.
//!
//! [`standing`] is pure over `(events, now)` so the verdict can be shown to go
//! red in tests, the way `ci::fast_tier_verdict` is. A staleness rule nobody has
//! watched reject is a rule on exactly the footing this milestone exists to fix.

use crate::timers::{TIERS, status_dir};
use std::time::{SystemTime, UNIX_EPOCH};

/// Keeps the ledger bounded without a rotation policy: two events a night is
/// ~730 lines a year, so this is years of history and a fixed cost either way.
const KEEP: usize = 500;

/// How long an unfinished run may still be in progress, after which its silence
/// means it died. The scheduler's own `ExecutionTimeLimit` (`PT6H`, and the
/// systemd units are unbounded), so nothing it starts outlives this and nothing
/// it kills is mistaken for live.
const RUN_LIMIT: u64 = 6 * 3600;

/// One line of `ci-status/history.txt`.
///
/// Fields are space-separated and `reason` is last, so it may contain spaces
/// and a reader that gains a field stays compatible with a file that lacks one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub at: u64,
    pub tier: String,
    pub kind: Kind,
    /// `git describe --always --dirty` — the tree, not merely the commit. The
    /// nightly that wrote M82's motivating `ok` ran over uncommitted work.
    pub commit: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    Started,
    Finished {
        ok: bool,
        secs: u64,
        /// The failing gate's own first line. The log is truncated on every
        /// launch, so without this a red that has since been superseded leaves
        /// nothing at all behind.
        reason: String,
    },
}

/// What a tier's history amounts to right now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// No record at all — a fresh clone, or a ledger someone removed.
    /// Deliberately the loudest reading rather than the quietest (§6 M82).
    /// `cargo clean` left this list at §6 M90, when the ledger moved out of
    /// `target/`: a state that arrives by routine housekeeping is not one a
    /// verdict can carry, since the reader cannot tell it from the real thing.
    Never,
    /// Started, no finish yet, and still inside [`RUN_LIMIT`] — this run is in
    /// progress. It is a real state and not a degenerate kill: `ci::nightly`
    /// calls `ci::push`, which reports, so the tier reads its own `started`
    /// line within a second of writing it.
    Running { at: u64, commit: String },
    /// Started and never finished, past the window in which it could still be
    /// going: killed by `StopOnIdleEnd`, or the machine went down.
    Killed { at: u64, commit: String },
    Ran {
        at: u64,
        commit: String,
        ok: bool,
        secs: u64,
        reason: String,
    },
}

impl Verdict {
    /// When this verdict was laid down, if it was.
    pub fn at(&self) -> Option<u64> {
        match self {
            Verdict::Never => None,
            Verdict::Running { at, .. } | Verdict::Killed { at, .. } | Verdict::Ran { at, .. } => {
                Some(*at)
            }
        }
    }

    fn healthy(&self) -> bool {
        matches!(
            self,
            Verdict::Ran { ok: true, .. } | Verdict::Running { .. }
        )
    }
}

/// A tier's verdict together with whether it is older than that tier allows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Standing {
    pub tier: &'static str,
    pub verdict: Verdict,
    /// Nothing has been recorded inside the tier's own tolerance. A refused
    /// launch produces exactly this and nothing else — it is the state the
    /// pre-M82 file had no way to hold.
    pub stale: bool,
}

impl Standing {
    /// Green means recorded, passing, and recent. Any other combination is
    /// something the desk should be told about.
    pub fn green(&self) -> bool {
        self.verdict.healthy() && !self.stale
    }

    /// One clause naming what is wrong, for the line `ci --fast` prints.
    pub fn complaint(&self, now: u64) -> Option<String> {
        if self.green() {
            return None;
        }
        let age = self
            .verdict
            .at()
            .map(|at| format!(" ({} ago)", ago(now.saturating_sub(at))));
        Some(match &self.verdict {
            Verdict::Never => format!("{} has never run", self.tier),
            // Healthy, so `green` already returned above; here for the match.
            Verdict::Running { .. } => format!("{} is running", self.tier),
            Verdict::Killed { .. } => format!(
                "{} was killed mid-run{}",
                self.tier,
                age.unwrap_or_default()
            ),
            Verdict::Ran {
                ok: false, reason, ..
            } => format!("{} is RED{}: {reason}", self.tier, age.unwrap_or_default()),
            // Passing but past its tolerance: the refusal case, and the one the
            // old `ok` reported as green for as long as it liked.
            Verdict::Ran { .. } => format!("{} has not run{}", self.tier, age.unwrap_or_default()),
        })
    }
}

// ---- the transcript, for the launch the ledger cannot see ----------------

/// How much of a tier's log is read to characterise it. A green nightly's
/// transcript is ~900 KiB and `--fast` reads this on every agent turn, so the
/// window is the tail — which is also the only part worth quoting, since the
/// case this exists for is a run that ended badly.
const TAIL: u64 = 8 * 1024;

/// What a tier's **transcript** says, where the ledger cannot speak at all.
///
/// [`around`] brackets the tier from inside the process running it, so a launch
/// that dies before `xtask` exists is one it can never record. Both hosts'
/// actions are `cargo xtask ci --<tier>`, which makes the runner something the
/// run must build first — and a desk in use holds `target/debug/xtask.exe`, so
/// cargo waits on the build lock and then fails to relink it. That is how this
/// tree's **weekly tier had never once completed** at §6 M90: a 115-byte log
/// from one Sunday reading `failed to remove file … Access is denied`, and a
/// ledger reading *never run* for the four days after it.
///
/// A log proves a **launch** and never an outcome, so this is read only where
/// the ledger is silent — never to contradict a record, only to fill a hole one
/// could not reach.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sighting {
    pub tier: &'static str,
    /// The transcript's own last write.
    pub at: u64,
    /// One line of it: the first that announces an error, else the last it
    /// wrote. Both readings answer *what was happening when it stopped*.
    pub note: String,
}

/// Every scheduled tier's transcript as the disk holds it now.
pub fn sightings() -> Vec<Sighting> {
    TIERS
        .iter()
        .filter_map(|tier| {
            let path = status_dir().join(format!("{}.log", tier.name));
            let meta = std::fs::metadata(&path).ok()?;
            let at = meta
                .modified()
                .ok()?
                .duration_since(UNIX_EPOCH)
                .ok()?
                .as_secs();
            Some(Sighting {
                tier: tier.name,
                at,
                note: note(&path, meta.len()),
            })
        })
        .collect()
}

fn note(path: &std::path::Path, len: u64) -> String {
    use std::io::{Read, Seek, SeekFrom};
    let mut text = String::new();
    let Ok(mut file) = std::fs::File::open(path) else {
        return String::new();
    };
    if len > TAIL && file.seek(SeekFrom::End(-(TAIL as i64))).is_err() {
        return String::new();
    }
    // Lossy: a tail cut at a fixed offset lands mid-codepoint by construction.
    let mut bytes = Vec::new();
    if file.read_to_end(&mut bytes).is_err() {
        return String::new();
    }
    text.push_str(&String::from_utf8_lossy(&bytes));
    let lines: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    lines
        .iter()
        .find(|l| l.to_ascii_lowercase().starts_with("error"))
        .or_else(|| lines.last())
        .map(|l| one_line(l))
        .unwrap_or_default()
}

/// A launch that happened and left no record of itself.
///
/// Pure over its arguments, like everything else that decides a reading here.
/// The rule is deliberately narrow: a transcript **newer than anything the
/// ledger holds** is a launch the ledger missed. A `Running` tier's log is
/// newer than its own `started` line by construction and a `Killed` one's is
/// newer than the start it never finished, so neither is a hole — both already
/// have a reading, and adding a second would be this module's own §6 M86.
///
/// It states the fact and names no cause, because the log cannot tell the two
/// apart and this desk has produced one of each: a weekly whose transcript is
/// 115 bytes of cargo failing to relink `xtask.exe`, and a nightly whose
/// transcript ends in its own green line — a run that bracketed itself and
/// whose record went missing afterwards. [`Sighting::note`] is what separates
/// them, and `timers --status` quotes it; one clause per tier is all the line
/// `ci --fast` prints on every agent turn can afford to be.
pub fn unrecorded(standing: &Standing, log: Option<&Sighting>, now: u64) -> Option<String> {
    let log = log?;
    let missed = match &standing.verdict {
        Verdict::Never => true,
        Verdict::Ran { at, .. } => log.at > *at,
        Verdict::Running { .. } | Verdict::Killed { .. } => false,
    };
    missed.then(|| {
        format!(
            "{} launched {} ago and left no record",
            standing.tier,
            ago(now.saturating_sub(log.at)),
        )
    })
}

// ---- what the scheduler was asked, as against what the tier did ----------

/// What the **scheduler** says about its own last launch — never a verdict.
///
/// It is one slot, overwritten by every attempt including a refused one, which
/// is the pre-§6 M82 defect exactly: this desk's 03:00 nightly ran green and
/// finished at 07:10, and a refusal at 14:47 the same day overwrote the result
/// it left. So this answers *was the tier asked*, and [`Standing`] answers what
/// it did; §6 M83 exists because [`Verdict::Never`] was covering both a task
/// declined every evening and one that was never registered at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Asked {
    /// No scheduler, or it would not say. Never a complaint on its own.
    Unknown,
    /// Installed and never launched — the scheduler's own sentinel.
    Never,
    /// `RunOnlyIfIdle` declined the launch. Windows-only by construction: the
    /// politeness contract is spelled `Nice=15` on the other host (§5), and a
    /// nice level never refuses to start anything.
    Refused { at: Option<u64> },
    /// Launched and then stopped from outside — `StopOnIdleEnd`, which this
    /// tree sets itself. The corroborating half of [`Verdict::Killed`].
    Stopped { at: Option<u64> },
    /// The scheduler believes it is running right now.
    Running { at: Option<u64> },
    /// It launched and the scheduler saw the process to an end.
    Launched {
        at: Option<u64>,
        ok: bool,
        /// Printed rather than interpreted when no table knows it, on
        /// `fp-isa`'s rule: a code guessed at is worse than a code quoted.
        code: String,
    },
}

impl Asked {
    fn at(&self) -> Option<u64> {
        match self {
            Asked::Unknown | Asked::Never => None,
            Asked::Refused { at } | Asked::Stopped { at } | Asked::Running { at } => *at,
            Asked::Launched { at, .. } => *at,
        }
    }
}

/// `<epoch|-> <code>` as `Get-ScheduledTaskInfo` reports it, or `none`.
///
/// Only the codes this desk was observed to produce are named, plus the one
/// `StopOnIdleEnd` documents; anything else is quoted as hex. The four measured
/// across 198 registered tasks were `0` (109), `0x41303` (77), `0x800710E0` (5)
/// and `0x41301` (4) — a census, not a recollection.
pub fn windows_asked(line: &str) -> Asked {
    let mut it = line.split_whitespace();
    let (Some(when), Some(code)) = (it.next(), it.next()) else {
        return Asked::Unknown;
    };
    // Windows tools disagree on sign for the same bits, so normalize through
    // `u32`: `schtasks /FO CSV` prints `-2147020576` for `0x800710E0` and
    // `Get-ScheduledTaskInfo` prints `2147946720`.
    let Ok(code) = code.parse::<i64>().map(|c| c as u32) else {
        return Asked::Unknown;
    };
    // 1999-11-30 is the never-run sentinel, and every date before this tree
    // existed means the same thing.
    let at = when.parse::<u64>().ok().filter(|t| *t > 1_262_304_000);
    match code {
        0 => Asked::Launched {
            at,
            ok: true,
            code: "0".to_owned(),
        },
        0x0004_1301 => Asked::Running { at },
        0x0004_1303 => Asked::Never,
        0x0004_1306 => Asked::Stopped { at },
        0x8007_10E0 => Asked::Refused { at },
        other => Asked::Launched {
            at,
            ok: false,
            code: format!("0x{other:08X}"),
        },
    }
}

/// The timer's last trigger and the service's `Result`. systemd has no idle
/// condition, so [`Asked::Refused`] is unreachable here — that is a difference
/// between the hosts' politeness contracts, not a gap in this reader.
pub fn systemd_asked(at: Option<u64>, result: Option<&str>) -> Asked {
    match result {
        None => Asked::Unknown,
        // A timer that has never fired leaves the property empty, and `date`
        // refuses the `n/a` systemd prints for it.
        _ if at.is_none() => Asked::Never,
        Some("success") => Asked::Launched {
            at,
            ok: true,
            code: "success".to_owned(),
        },
        Some("running") | Some("start-limit-hit") => Asked::Running { at },
        Some("signal") | Some("timeout") => Asked::Stopped { at },
        Some(other) => Asked::Launched {
            at,
            ok: false,
            code: other.to_owned(),
        },
    }
}

/// The reading that needs both halves, or `None` when the scheduler adds
/// nothing to what the ledger already said.
///
/// The load-bearing row is the third: a launch the scheduler saw succeed with
/// no record on disk is §6 M82's own wiring broken, and it is the one state
/// here that indicts this tree rather than the desk. `ledger_seen` is what
/// keeps that accusation honest — a `history.txt` that does not exist means
/// nothing has ever recorded, which a run predating §6 M82 or a `cargo clean`
/// both produce, while a file that exists and lacks the run is the wiring. The
/// WSL lane hit exactly that on M83's first run of this code and the message
/// was wrong about it, which is the misdiagnosis M82 exists to not ship.
pub fn reconcile(
    standing: &Standing,
    asked: &Asked,
    now: u64,
    ledger_seen: bool,
) -> Option<String> {
    let when = asked
        .at()
        .map(|t| format!(" ({} ago)", ago(now.saturating_sub(t))))
        .unwrap_or_default();
    let tier = standing.tier;
    Some(match (&standing.verdict, asked) {
        (_, Asked::Unknown) => return None,
        // Silence with a cause. The desk is the dev machine and the gaming PC,
        // so a declined launch is the politeness contract working, not a fault
        // — but it is not "never installed" and it is not "never fired".
        (Verdict::Never, Asked::Refused { .. }) => format!(
            "{tier}: the scheduler declined its last launch{when} — idle-only (§5), so the desk \
             was in use. Not silence, and nothing to repair."
        ),
        (Verdict::Never, Asked::Never) => format!(
            "{tier}: installed and never launched — the scheduler has not reached its hour yet."
        ),
        (Verdict::Never, Asked::Launched { ok: true, .. }) if ledger_seen => format!(
            "{tier}: the scheduler ran it{when} and it recorded nothing — the ledger is broken, \
             not the timer (§6 M83)."
        ),
        (Verdict::Never, Asked::Launched { ok: true, .. }) => format!(
            "{tier}: the scheduler ran it{when} and there is no ledger yet — that run predates \
             §6 M82, or the ledger has not been written since it moved out of `target/` at \
             §6 M90. The next scheduled run settles it."
        ),
        // The scheduler corroborating a kill, which is the one thing the ledger
        // infers rather than observes: a start with no finish.
        (Verdict::Killed { .. }, Asked::Stopped { .. }) => {
            format!("{tier}: the scheduler stopped it{when} — `StopOnIdleEnd`, the desk came back.")
        }
        (Verdict::Killed { .. }, Asked::Refused { .. }) => {
            format!("{tier}: killed mid-run, and the scheduler has since declined a launch{when}.")
        }
        _ => return None,
    })
}

// ---- writing ------------------------------------------------------------

/// Run `f` as `tier`, bracketed by the two records.
///
/// Both halves are written here so neither can be forgotten, and the result is
/// returned untouched: a tier that fails must still fail its caller. Recording
/// is best-effort — a `target/` that cannot be written is not a red tier, the
/// same judgement `ci::fast_tier_budget` makes about its own ledger.
pub fn around(tier: &str, f: impl FnOnce() -> anyhow::Result<()>) -> anyhow::Result<()> {
    let commit = describe();
    let started = now();
    append(&Event {
        at: started,
        tier: tier.into(),
        kind: Kind::Started,
        commit: commit.clone(),
    });
    let result = f();
    append(&Event {
        at: now(),
        tier: tier.into(),
        kind: Kind::Finished {
            ok: result.is_ok(),
            secs: now().saturating_sub(started),
            reason: match &result {
                Ok(()) => String::new(),
                Err(e) => one_line(&e.to_string()),
            },
        },
        commit,
    });
    result
}

/// Whether a tier's runs are worth recording. Only the *scheduled* ones are:
/// `--fast` runs on every agent turn and its freshness is not in question, and
/// a tier that recorded itself would then be reading its own noise.
pub fn scheduled(tier: &str) -> bool {
    TIERS.iter().any(|t| t.name == tier)
}

fn append(event: &Event) {
    let path = status_dir().join("history.txt");
    let _ = std::fs::create_dir_all(status_dir());
    let mut lines: Vec<String> = std::fs::read_to_string(&path)
        .unwrap_or_default()
        .lines()
        .map(str::to_owned)
        .collect();
    lines.push(event.encode());
    let keep = lines.len().saturating_sub(KEEP);
    let _ = std::fs::write(&path, lines[keep..].join("\n") + "\n");
}

impl Event {
    fn encode(&self) -> String {
        match &self.kind {
            Kind::Started => format!("{} {} started {}", self.at, self.tier, self.commit),
            Kind::Finished { ok, secs, reason } => format!(
                "{} {} {} {} {secs}{}",
                self.at,
                self.tier,
                if *ok { "ok" } else { "red" },
                self.commit,
                if reason.is_empty() {
                    String::new()
                } else {
                    format!(" {reason}")
                },
            ),
        }
    }

    fn decode(line: &str) -> Option<Event> {
        let mut it = line.splitn(6, ' ');
        let at = it.next()?.parse().ok()?;
        let tier = it.next()?.to_owned();
        let verb = it.next()?;
        let commit = it.next()?.to_owned();
        let kind = match verb {
            "started" => Kind::Started,
            "ok" | "red" => Kind::Finished {
                ok: verb == "ok",
                secs: it.next().unwrap_or("0").parse().unwrap_or(0),
                reason: it.next().unwrap_or("").to_owned(),
            },
            // An unknown verb is a file from a later build; skipping it beats
            // refusing to report at all.
            _ => return None,
        };
        Some(Event {
            at,
            tier,
            kind,
            commit,
        })
    }
}

// ---- reading ------------------------------------------------------------

/// Whether anything has ever recorded, as distinct from whether this tier has.
/// The discriminator [`reconcile`] needs to tell a broken recorder from a
/// ledger that simply does not exist yet.
pub fn ledger_seen() -> bool {
    status_dir().join("history.txt").is_file()
}

pub fn events() -> Vec<Event> {
    std::fs::read_to_string(status_dir().join("history.txt"))
        .unwrap_or_default()
        .lines()
        .filter_map(Event::decode)
        .collect()
}

/// Every scheduled tier's verdict as of `now`.
///
/// Pure over its two arguments on purpose — this is the claim the milestone
/// makes, and `mod tests` is where it is shown to reject.
pub fn standing(events: &[Event], now: u64) -> Vec<Standing> {
    TIERS
        .iter()
        .map(|tier| {
            let mine: Vec<&Event> = events.iter().filter(|e| e.tier == tier.name).collect();
            let verdict = match mine.last() {
                None => Verdict::Never,
                // The newest record is a start: nothing has written the other
                // half. Either it is still being written, or nothing ever will.
                Some(e) if e.kind == Kind::Started => {
                    let (at, commit) = (e.at, e.commit.clone());
                    if now.saturating_sub(at) <= RUN_LIMIT {
                        Verdict::Running { at, commit }
                    } else {
                        Verdict::Killed { at, commit }
                    }
                }
                Some(e) => match &e.kind {
                    Kind::Finished { ok, secs, reason } => Verdict::Ran {
                        at: e.at,
                        commit: e.commit.clone(),
                        ok: *ok,
                        secs: *secs,
                        reason: reason.clone(),
                    },
                    Kind::Started => unreachable!("matched above"),
                },
            };
            let stale = match verdict.at() {
                None => true,
                Some(at) => now.saturating_sub(at) > tier.stale_after_days * 86_400,
            };
            Standing {
                tier: tier.name,
                verdict,
                stale,
            }
        })
        .collect()
}

/// The one line an interactive tier prints, or nothing when every scheduled
/// tier is recent and green.
///
/// §5 calls a red nightly "the stop-the-line event a red main used to be", and
/// until M82 the only reader of that verdict was a manual command with no
/// caller. This is the reader: the Stop hook's cadence is the only one this
/// desk reliably has.
pub fn verdict_line(events: &[Event], now: u64, logs: &[Sighting]) -> Option<String> {
    let complaints: Vec<String> = standing(events, now)
        .iter()
        .flat_map(|s| {
            let missed = unrecorded(s, logs.iter().find(|l| l.tier == s.tier), now);
            match (&s.verdict, missed) {
                // A launch nothing recorded *is* the whole of "has never run",
                // with the cause attached; anywhere else it stands beside the
                // complaint, since a superseded red is still a red.
                (Verdict::Never, Some(line)) => vec![line],
                (_, missed) => s.complaint(now).into_iter().chain(missed).collect(),
            }
        })
        .collect();
    if complaints.is_empty() {
        return None;
    }
    Some(format!(
        "xtask: scheduled tiers — {} (§5; `cargo xtask timers` for the history)",
        complaints.join("; ")
    ))
}

/// Print [`verdict_line`] if there is one. Never fails: a scheduled task that
/// did not fire is a condition of the machine, and refusing to build over it
/// would block every commit on something no edit can repair (§6 M82).
pub fn report() {
    if let Some(line) = verdict_line(&events(), now(), &sightings()) {
        println!("{line}");
    }
}

// ---- time ---------------------------------------------------------------

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `2026-08-20 07:10` UTC. Written out rather than rented: `xtask` has no date
/// dependency and this is the whole of what one would be for.
pub fn stamp(at: u64) -> String {
    let (y, m, d) = civil_from_days(i64::try_from(at / 86_400).unwrap_or(0));
    let rem = at % 86_400;
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}",
        rem / 3600,
        (rem % 3600) / 60
    )
}

/// Days since 1970-01-01 → civil date. Hinnant's `civil_from_days`, exact for
/// every day in the proleptic Gregorian calendar.
fn civil_from_days(z: i64) -> (i64, u64, u64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = yoe as i64 + era * 400;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// A duration a human reads at a glance. Coarse on purpose: the question this
/// answers is "is this recent", never "how long exactly".
pub fn ago(secs: u64) -> String {
    match secs {
        s if s < 90 * 60 => format!("{}m", s / 60),
        s if s < 48 * 3600 => format!("{}h", s / 3600),
        s => format!("{}d", s / 86_400),
    }
}

fn one_line(s: &str) -> String {
    s.lines().next().unwrap_or("").chars().take(160).collect()
}

/// The tree a run happened over, not merely the commit — `-dirty` is the whole
/// point (§6 M82 row 3).
fn describe() -> String {
    crate::util::run_capture(
        std::process::Command::new("git")
            .current_dir(crate::util::workspace_root())
            .args(["describe", "--always", "--dirty"]),
        "git describe",
    )
    .map(|s| s.trim().to_owned())
    .unwrap_or_else(|_| "unknown".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY: u64 = 86_400;

    fn ev(at: u64, tier: &str, kind: Kind) -> Event {
        Event {
            at,
            tier: tier.into(),
            kind,
            commit: "abc1234".into(),
        }
    }

    fn done(ok: bool) -> Kind {
        Kind::Finished {
            ok,
            secs: 900,
            reason: if ok {
                String::new()
            } else {
                "clippy failed".into()
            },
        }
    }

    fn nightly(events: &[Event], now: u64) -> Standing {
        standing(events, now)
            .into_iter()
            .find(|s| s.tier == "nightly")
            .expect("nightly is a scheduled tier")
    }

    fn seen(at: u64, note: &str) -> Sighting {
        Sighting {
            tier: "nightly",
            at,
            note: note.into(),
        }
    }

    /// The third witness, and the four readings it has to keep apart (§6 M90).
    ///
    /// The case it exists for is the first: this tree's weekly tier launched
    /// every Sunday and had never once completed, because the action is
    /// `cargo xtask ci --weekly` and a desk in use holds the `xtask.exe` the
    /// run must relink — so it died before there was anything to bracket it,
    /// and the ledger, which can only be written from inside, said *never run*.
    #[test]
    fn a_transcript_the_ledger_cannot_account_for_is_a_launch_it_missed() {
        let now = 100 * DAY;
        let never = nightly(&[], now);
        let line = unrecorded(&never, Some(&seen(now - 4 * DAY, "error: …")), now)
            .expect("a log with no record at all is the hole this fills");
        assert!(line.contains("nightly launched 4d ago"), "{line}");

        // A run that *did* record itself, with nothing newer on disk: the
        // ledger is the better witness and this one keeps quiet.
        let ran = [
            ev(now - 3600, "nightly", Kind::Started),
            ev(now - 900, "nightly", done(true)),
        ];
        let standing = nightly(&ran, now);
        assert_eq!(
            unrecorded(&standing, Some(&seen(now - 1000, "…")), now),
            None
        );
        // The same run, and a transcript from *after* it — a second launch the
        // ledger never saw.
        assert!(unrecorded(&standing, Some(&seen(now - 60, "…")), now).is_some());

        // A live run writes its log continuously and its `finished` line only
        // at the end, so `Running` is newer-log by construction and must not
        // read as a miss. Same for a `Killed` start that was never finished.
        let live = nightly(&[ev(now - 600, "nightly", Kind::Started)], now);
        assert!(matches!(live.verdict, Verdict::Running { .. }));
        assert_eq!(unrecorded(&live, Some(&seen(now - 5, "…")), now), None);
        let killed = nightly(&[ev(now - 8 * 3600, "nightly", Kind::Started)], now);
        assert!(matches!(killed.verdict, Verdict::Killed { .. }));
        assert_eq!(
            unrecorded(&killed, Some(&seen(now - 7 * 3600, "…")), now),
            None
        );

        // And no transcript is no claim: a tier that never wrote one is the
        // ordinary fresh-clone case, which `Verdict::Never` already reports.
        assert_eq!(unrecorded(&never, None, now), None);
    }

    /// A launch nothing recorded *replaces* "has never run" rather than joining
    /// it — the two are one fact, and the second has the cause attached.
    #[test]
    fn the_one_line_prefers_the_witness_that_knows_more() {
        let now = 100 * DAY;
        let line = verdict_line(&[], now, &[seen(now - 2 * DAY, "…")]).expect("a complaint");
        assert!(
            line.contains("nightly launched 2d ago and left no record"),
            "{line}"
        );
        assert!(!line.contains("nightly has never run"), "not both: {line}");
        // The weekly has no transcript here, so it keeps the plain reading.
        assert!(line.contains("weekly has never run"), "{line}");
    }

    #[test]
    fn a_tier_that_passed_recently_is_green_and_says_nothing() {
        let now = 100 * DAY;
        let events = [
            ev(now - 3600, "nightly", Kind::Started),
            ev(now - 900, "nightly", done(true)),
            ev(now - 3 * DAY, "weekly", Kind::Started),
            ev(now - 3 * DAY + 900, "weekly", done(true)),
        ];
        assert!(nightly(&events, now).green());
        assert_eq!(verdict_line(&events, now, &[]), None);
    }

    /// The whole defect: a launch refused for not being idle writes nothing, so
    /// the newest record stays green and merely gets old. Before M82 that read
    /// as `ok` forever.
    #[test]
    fn a_pass_that_stopped_being_recent_is_not_green() {
        let now = 100 * DAY;
        let events = [
            ev(now - 9 * DAY, "nightly", Kind::Started),
            ev(now - 9 * DAY + 900, "nightly", done(true)),
        ];
        let s = nightly(&events, now);
        assert!(s.verdict.healthy(), "the run itself passed");
        assert!(s.stale, "and it is nine days old against a nightly");
        assert!(!s.green());
        assert!(
            s.complaint(now)
                .expect("stale is a complaint")
                .contains("has not run"),
            "the complaint names the absence, not the old pass"
        );
    }

    /// `StopOnIdleEnd` kills the run between the two records. The pre-M82 file
    /// kept the *previous* verdict; this one cannot.
    #[test]
    fn a_start_with_no_finish_is_a_kill_and_not_the_pass_before_it() {
        let now = 100 * DAY;
        let events = [
            ev(now - 2 * DAY, "nightly", Kind::Started),
            ev(now - 2 * DAY + 900, "nightly", done(true)),
            ev(now - RUN_LIMIT - 60, "nightly", Kind::Started),
        ];
        let s = nightly(&events, now);
        assert!(matches!(s.verdict, Verdict::Killed { .. }));
        assert!(!s.green(), "a kill is not a pass, however recent");
        assert!(
            s.complaint(now)
                .expect("a kill is a complaint")
                .contains("killed")
        );
    }

    /// The same shape inside the window is a run in progress, and must not be
    /// reported as a kill: `ci::nightly` calls `ci::push`, which reports, so a
    /// live nightly reads its own `started` line seconds after writing it.
    #[test]
    fn a_start_inside_the_window_is_a_run_in_progress_and_says_nothing() {
        let now = 100 * DAY;
        let live = [ev(now - 90 * 60, "nightly", Kind::Started)];
        let s = nightly(&live, now);
        assert!(matches!(s.verdict, Verdict::Running { .. }));
        assert!(s.green(), "a run in progress is not a complaint");
        assert_eq!(s.complaint(now), None);

        // And the boundary is the scheduler's own execution limit, either side.
        let dead = [ev(now - RUN_LIMIT - 1, "nightly", Kind::Started)];
        assert!(matches!(
            nightly(&dead, now).verdict,
            Verdict::Killed { .. }
        ));
        let alive = [ev(now - RUN_LIMIT, "nightly", Kind::Started)];
        assert!(matches!(
            nightly(&alive, now).verdict,
            Verdict::Running { .. }
        ));
    }

    #[test]
    fn a_red_names_the_gate_that_failed() {
        let now = 100 * DAY;
        let events = [
            ev(now - 3600, "nightly", Kind::Started),
            ev(now - 900, "nightly", done(false)),
        ];
        let line = verdict_line(&events, now, &[]).expect("a red is a complaint");
        assert!(line.contains("nightly is RED"), "{line}");
        assert!(line.contains("clippy failed"), "the reason travels: {line}");
    }

    /// A cleaned tree reports the loudest thing it can, not the quietest.
    #[test]
    fn no_history_at_all_reads_as_never_run() {
        let line = verdict_line(&[], 100 * DAY, &[]).expect("an empty ledger is a complaint");
        assert!(line.contains("nightly has never run"), "{line}");
        assert!(line.contains("weekly has never run"), "{line}");
    }

    #[test]
    fn every_event_survives_the_round_trip() {
        for kind in [Kind::Started, done(true), done(false)] {
            let e = ev(1_755_678_000, "nightly", kind);
            assert_eq!(Event::decode(&e.encode()), Some(e));
        }
    }

    /// A reason with spaces in it is why the encoder puts it last.
    #[test]
    fn a_reason_keeps_its_spaces() {
        let e = ev(
            1_755_678_000,
            "weekly",
            Kind::Finished {
                ok: false,
                secs: 16,
                reason: "failed to remove file target/debug/xtask.exe (os error 5)".into(),
            },
        );
        assert_eq!(Event::decode(&e.encode()), Some(e));
    }

    #[test]
    fn a_line_from_a_later_build_is_skipped_rather_than_fatal() {
        assert_eq!(Event::decode("123 nightly paused abc1234"), None);
        assert_eq!(Event::decode("not-a-number nightly started abc"), None);
    }

    /// The dates in `--status` are read by a human, so they had better be right.
    #[test]
    fn the_calendar_is_the_real_one() {
        assert_eq!(stamp(0), "1970-01-01 00:00");
        assert_eq!(stamp(1_755_678_000), "2025-08-20 08:20");
        assert_eq!(stamp(1_755_693_000), "2025-08-20 12:30");
        // 2024-02-29: the leap day a naive 365-day fold loses. 2000-02-29 is
        // the other one — divisible by 100 *and* by 400, so a rule that stops
        // at the century gets it wrong in the opposite direction.
        assert_eq!(stamp(1_709_164_800), "2024-02-29 00:00");
        assert_eq!(stamp(951_782_400), "2000-02-29 00:00");
    }

    #[test]
    fn only_the_scheduled_tiers_keep_a_record() {
        assert!(scheduled("nightly") && scheduled("weekly"));
        assert!(!scheduled("fast") && !scheduled("push"));
    }

    /// The write path, end to end and on a real disk — the half of this module
    /// the pure functions above cannot reach.
    ///
    /// One test rather than four because it moves a process-wide environment
    /// variable: nextest gives each *test* its own process, and this stays
    /// correct if it ever does not.
    #[test]
    fn a_run_writes_both_halves_and_a_failure_keeps_its_cause() {
        let dir = std::env::temp_dir().join(format!("gg-m82-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // SAFETY: nextest runs each test in its own process, so nothing else in
        // this program observes the variable. Restored below regardless.
        unsafe { std::env::set_var("GG_CI_STATUS_DIR", &dir) };

        let green = around("nightly", || Ok(()));
        let after_green = events();
        let red = around("nightly", || {
            anyhow::bail!("clippy failed\nand a second line")
        });
        let after_red = events();

        // SAFETY: as above.
        unsafe { std::env::remove_var("GG_CI_STATUS_DIR") };
        let _ = std::fs::remove_dir_all(&dir);

        assert!(green.is_ok());
        assert_eq!(after_green.len(), 2, "a run leaves a start and a finish");
        assert_eq!(after_green[0].kind, Kind::Started);
        let now = after_green[1].at;
        assert!(
            standing(&after_green, now)
                .iter()
                .find(|s| s.tier == "nightly")
                .expect("nightly")
                .green(),
            "a run that just passed is green"
        );

        // The tier still fails its caller: recording is a side effect, never a
        // verdict of its own.
        assert!(red.is_err());
        assert_eq!(after_red.len(), 4, "appended, not overwritten");
        match &after_red[3].kind {
            Kind::Finished { ok, reason, .. } => {
                assert!(!ok);
                assert_eq!(
                    reason, "clippy failed",
                    "the cause travels as one line, so it fits a record"
                );
            }
            Kind::Started => panic!("the fourth event is a finish"),
        }
        assert!(
            after_red[3].commit.len() >= 7,
            "the tree is recorded, not merely the fact of a run: {:?}",
            after_red[3].commit
        );
    }

    // ---- §6 M83: what the scheduler was asked -----------------------------

    /// A tier with no ledger at all, which is where every M83 row starts.
    fn never(tier: &'static str) -> Standing {
        Standing {
            tier,
            verdict: Verdict::Never,
            stale: true,
        }
    }

    /// The bits this desk actually produced, read out of `Get-ScheduledTaskInfo`
    /// at §6 M83 — `2147946720` is `0x800710E0` unsigned, and `schtasks` prints
    /// the same code as `-2147020576`. Both must land on the same state, which
    /// is the whole reason the parse normalizes through `u32`.
    #[test]
    fn the_two_windows_spellings_of_one_refusal_agree() {
        let unsigned = windows_asked("1787255220 2147946720");
        let signed = windows_asked("1787255220 -2147020576");
        assert_eq!(unsigned, signed);
        assert_eq!(
            unsigned,
            Asked::Refused {
                at: Some(1_787_255_220)
            }
        );
    }

    /// Row 1 of the milestone. Before it, this desk's `--status` said "never
    /// run" of a task the scheduler had declined that afternoon — the same
    /// three words it would have used for one that was never registered.
    #[test]
    fn a_refused_launch_is_not_silence_and_not_a_missing_task() {
        let now = 1_787_255_220 + 3 * 3600;
        let refused = reconcile(
            &never("nightly"),
            &Asked::Refused {
                at: Some(1_787_255_220),
            },
            now,
            false,
        )
        .expect("a refusal is worth saying");
        assert!(refused.contains("declined"), "{refused}");
        assert!(refused.contains("3h ago"), "{refused}");
        assert!(
            refused.contains("nothing to repair"),
            "a refusal is the politeness contract working: {refused}"
        );

        let unlaunched = reconcile(&never("nightly"), &Asked::Never, now, false)
            .expect("never-launched is worth saying");
        assert!(
            unlaunched != refused,
            "the two states the pre-M83 reading conflated"
        );
        assert!(unlaunched.contains("never launched"), "{unlaunched}");
    }

    /// The one row that indicts this tree rather than the desk: the scheduler
    /// saw the tier run to a clean exit and the ledger holds nothing, which is
    /// §6 M82's own wiring broken. It must not read like a quiet night.
    #[test]
    fn a_run_the_scheduler_saw_succeed_with_no_record_blames_the_ledger() {
        let ran = Asked::Launched {
            at: Some(1_787_000_000),
            ok: true,
            code: "0".into(),
        };
        let now = 1_787_000_000 + 600;
        let line = reconcile(&never("weekly"), &ran, now, true)
            .expect("a run that recorded nothing is the loudest row here");
        assert!(line.contains("ledger is broken"), "{line}");
        assert!(!line.contains("declined"), "{line}");

        // The same scheduler answer with no ledger file at all is a different
        // fact, and the WSL lane was in exactly this state at §6 M83: its last
        // two runs predate the milestone that taught a tier to record itself.
        let fresh = reconcile(&never("weekly"), &ran, now, false)
            .expect("a run with no ledger yet is still worth saying");
        assert!(
            !fresh.contains("broken"),
            "a ledger that does not exist yet is not a broken one: {fresh}"
        );
        assert!(fresh.contains("predates"), "{fresh}");
    }

    /// A kill is the one verdict the ledger *infers* — a start with no finish —
    /// so the scheduler saying it stopped the task is corroboration, and names
    /// the setting this tree itself asked for.
    #[test]
    fn the_scheduler_corroborates_a_kill_it_caused() {
        let killed = Standing {
            tier: "nightly",
            verdict: Verdict::Killed {
                at: 1_787_000_000,
                commit: "abc1234".into(),
            },
            stale: false,
        };
        let line = reconcile(
            &killed,
            &Asked::Stopped {
                at: Some(1_787_010_000),
            },
            1_787_010_000,
            true,
        )
        .expect("a stop is worth saying beside an inferred kill");
        assert!(line.contains("StopOnIdleEnd"), "{line}");
    }

    /// The scheduler is never a verdict: it holds one slot, overwritten by
    /// every attempt, and this desk proved why — a green 03:00 run whose result
    /// a 14:47 refusal replaced the same day. A tier the ledger says passed is
    /// not made sick by the newest attempt having been declined.
    #[test]
    fn the_newest_attempt_does_not_overrule_a_recorded_pass() {
        let passed = Standing {
            tier: "nightly",
            verdict: Verdict::Ran {
                at: 1_787_200_000,
                commit: "abc1234".into(),
                ok: true,
                secs: 15_000,
                reason: String::new(),
            },
            stale: false,
        };
        assert!(passed.green());
        assert_eq!(
            reconcile(
                &passed,
                &Asked::Refused {
                    at: Some(1_787_255_220)
                },
                1_787_255_220,
                true
            ),
            None,
            "a refusal after a recorded pass is an ordinary evening"
        );
    }

    /// A code no table knows is quoted, never guessed at — `fp-isa`'s rule for
    /// an unrecognized mnemonic, and the reason the four codes that *are* named
    /// came from a census of 198 registered tasks rather than from memory.
    #[test]
    fn a_result_code_no_table_knows_is_quoted_as_hex() {
        let Asked::Launched { ok, code, .. } = windows_asked("1787255220 2147746065") else {
            panic!("an unknown code is still a launch that ended");
        };
        assert!(!ok);
        assert_eq!(code, "0x80040111");
    }

    /// The never-run sentinel is a date, not only a code: every task on this
    /// desk that had never run reported `11/30/1999`, which is before this tree
    /// existed and cannot be a real run time.
    #[test]
    fn the_never_run_sentinel_date_is_not_reported_as_a_time() {
        assert_eq!(windows_asked("-  267011"), Asked::Never);
        // 1999-11-30 as an epoch, which is what the field converts to.
        assert_eq!(windows_asked("943920000 267011"), Asked::Never);
        assert_eq!(
            windows_asked("943920000 0").at(),
            None,
            "a sentinel date is not a run time whatever the code beside it"
        );
    }

    /// systemd has no idle condition, so a refusal is unreachable on that host.
    /// The politeness contract is `Nice=15` there (§5) and a nice level never
    /// declines to start anything — a difference between the hosts, stated.
    #[test]
    fn the_linux_host_cannot_refuse_a_launch() {
        let states = ["success", "exit-code", "signal", "timeout", "running"];
        for result in states {
            let asked = systemd_asked(Some(1_787_216_216), Some(result));
            assert!(
                !matches!(asked, Asked::Refused { .. }),
                "{result} read as a refusal"
            );
        }
        assert_eq!(systemd_asked(None, None), Asked::Unknown);
        assert_eq!(
            systemd_asked(None, Some("success")),
            Asked::Never,
            "a timer that never fired leaves no trigger time to parse"
        );
    }

    /// An unknown scheduler adds nothing, and must never turn into a complaint
    /// of its own — the host without the other's scheduler is the common case,
    /// not a fault.
    #[test]
    fn an_unknown_scheduler_says_nothing_at_all() {
        for verdict in [
            Verdict::Never,
            Verdict::Killed {
                at: 1,
                commit: "abc1234".into(),
            },
        ] {
            let standing = Standing {
                tier: "nightly",
                verdict,
                stale: true,
            };
            assert_eq!(reconcile(&standing, &Asked::Unknown, 100_000, true), None);
        }
    }
}
