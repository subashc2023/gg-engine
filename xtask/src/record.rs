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

/// One line of `target/ci-status/history.txt`.
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
    /// No record at all — a fresh clone, or a `cargo clean`. Deliberately the
    /// loudest reading rather than the quietest (§6 M82).
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
pub fn verdict_line(events: &[Event], now: u64) -> Option<String> {
    let complaints: Vec<String> = standing(events, now)
        .iter()
        .filter_map(|s| s.complaint(now))
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
    if let Some(line) = verdict_line(&events(), now()) {
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
        assert_eq!(verdict_line(&events, now), None);
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
        let line = verdict_line(&events, now).expect("a red is a complaint");
        assert!(line.contains("nightly is RED"), "{line}");
        assert!(line.contains("clippy failed"), "the reason travels: {line}");
    }

    /// A cleaned tree reports the loudest thing it can, not the quietest.
    #[test]
    fn no_history_at_all_reads_as_never_run() {
        let line = verdict_line(&[], 100 * DAY).expect("an empty ledger is a complaint");
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
}
