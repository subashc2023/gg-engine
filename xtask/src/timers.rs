//! `cargo xtask timers [--install|--status|--uninstall]` — the scheduled tiers
//! (§5, "Scheduling and politeness"), deferred from M0A to M4B because until
//! this milestone there were no nightly-tier gates to schedule (§6 M0A).
//!
//! Two hosts, two mechanisms, one shape: **one run a night** at 03:00 —
//! nightly Monday to Saturday, weekly on Sunday, which is a superset of it —
//! both at **below-normal priority**, both under `GG_HEADLESS=1`, both leaving
//! a record under `target/ci-status/`. Windows uses Task Scheduler through a
//! generated XML — `schtasks /Create` on a command line cannot express
//! idle-only or priority, and those are the politeness half of the contract,
//! not decoration. WSL uses systemd user timers.
//!
//! What this module answers is *is a tier installed*. Whether it has actually
//! **run** is `record.rs`, and §6 M82 is the milestone that found those to be
//! different questions: these two tasks were installed, enabled, and returning
//! `0x800710E0` — the scheduler declining a launch whose idle condition is
//! unmet — while the one word on disk went on reading `ok`.
//!
//! **Installing a timer changes the machine**, so this command does exactly
//! what its flag says and nothing implicitly: `--status` reads, `--install`
//! writes, `--uninstall` removes. No tier calls it.
//!
//! The politeness contract (§5): the dev machine is also the gaming PC. A
//! nightly that steals frames from whatever is on the screen is worse than no
//! nightly, so the tasks run only when idle, stop when the machine stops being
//! idle, and never create a window (§1.5 — the tiers guarantee that themselves).

use crate::record;
use crate::util::{run_capture, workspace_root};
use std::path::PathBuf;

/// A scheduled tier: what it is called, when it fires on each host, and how
/// long its silence may go unremarked. Whole trigger elements rather than
/// fragments, so a schedule change is one edit here and not a templating
/// puzzle — and one table, because `record.rs` grades the same two tiers and a
/// second list is a list that drifts (§6 M81's whole finding).
pub struct Tier {
    pub name: &'static str,
    pub windows_trigger: &'static str,
    pub on_calendar: &'static str,
    /// Days of no record before the tier reads stale. Slack over the period on
    /// purpose: `RunOnlyIfIdle` means a night the desk is in use produces no
    /// run at all, and a rule that cried wolf on one busy evening would be
    /// ignored inside a week.
    pub stale_after_days: u64,
}

/// **One scheduled run a night, and never two at once** (§6 M82). Until that
/// milestone the nightly fired daily at 03:00 and the weekly at 04:00 on Sunday
/// — but `ci::weekly` *is* `ci::nightly` plus two gates, and the nightly takes
/// four hours on this desk, so every Sunday the weekly started an hour into a
/// run it duplicated and died sixteen seconds later relinking an `xtask.exe`
/// the first one held (`os error 5`). Sunday is the weekly's night and the
/// nightly stands down; the superset semantics are unchanged.
pub const TIERS: &[Tier] = &[
    Tier {
        name: "nightly",
        windows_trigger: "<CalendarTrigger><StartBoundary>2026-01-01T03:00:00</StartBoundary>\
             <Enabled>true</Enabled><ScheduleByWeek><DaysOfWeek><Monday /><Tuesday />\
             <Wednesday /><Thursday /><Friday /><Saturday /></DaysOfWeek>\
             <WeeksInterval>1</WeeksInterval></ScheduleByWeek></CalendarTrigger>",
        on_calendar: "Mon..Sat *-*-* 03:00:00",
        stale_after_days: 3,
    },
    Tier {
        name: "weekly",
        windows_trigger: "<CalendarTrigger><StartBoundary>2026-01-04T03:00:00</StartBoundary>\
             <Enabled>true</Enabled><ScheduleByWeek><DaysOfWeek><Sunday /></DaysOfWeek>\
             <WeeksInterval>1</WeeksInterval></ScheduleByWeek></CalendarTrigger>",
        on_calendar: "Sun *-*-* 03:00:00",
        stale_after_days: 10,
    },
];

pub fn run(args: &[&str]) -> anyhow::Result<()> {
    match args {
        a if a.contains(&"--install") => install(),
        a if a.contains(&"--uninstall") => uninstall(),
        _ => status(),
    }
}

/// Where a run leaves its record. Read by `--status`, and the artifact §5 asks
/// a scheduled tier to write.
///
/// `GG_CI_STATUS_DIR` redirects it, and exists for exactly one caller:
/// `record`'s test of the write path. `around` is the load-bearing function of
/// §6 M82 and writing to the desk's own ledger to prove it works would corrupt
/// the record the milestone exists to make trustworthy — a seam is the cheaper
/// of the two, and a milestone whose thesis is that an ungated claim rots does
/// not get to leave its own centre ungated.
pub fn status_dir() -> PathBuf {
    std::env::var_os("GG_CI_STATUS_DIR")
        .map_or_else(|| workspace_root().join("target/ci-status"), PathBuf::from)
}

/// Installed-ness is the scheduler's answer; whether the tier *ran* is the
/// history's, and before §6 M82 only the first question had one. A task can be
/// installed, enabled, and refused every night for a month.
fn installed(tier: &str) -> bool {
    if cfg!(windows) {
        run_capture(
            std::process::Command::new("schtasks").args(["/Query", "/TN", &task_name(tier)]),
            "schtasks /Query",
        )
        .is_ok()
    } else {
        run_capture(
            std::process::Command::new("systemctl").args([
                "--user",
                "is-enabled",
                &format!("gg-{tier}.timer"),
            ]),
            "systemctl is-enabled",
        )
        .is_ok()
    }
}

fn status() -> anyhow::Result<()> {
    let events = record::events();
    let now = record::now();
    for standing in record::standing(&events, now) {
        let detail = match &standing.verdict {
            record::Verdict::Never => "no record — never run, or `cargo clean`".to_owned(),
            record::Verdict::Running { at, commit } => format!(
                "{}  running now      {commit}  (started {} ago)",
                record::stamp(*at),
                record::ago(now.saturating_sub(*at)),
            ),
            record::Verdict::Killed { at, commit } => format!(
                "{}  KILLED mid-run  {commit}  ({} ago)",
                record::stamp(*at),
                record::ago(now.saturating_sub(*at)),
            ),
            record::Verdict::Ran {
                at,
                commit,
                ok,
                secs,
                reason,
            } => format!(
                "{}  {:<6} took {:<5} {commit}  ({} ago){}",
                record::stamp(*at),
                if *ok { "ok" } else { "RED" },
                record::ago(*secs),
                record::ago(now.saturating_sub(*at)),
                if reason.is_empty() {
                    String::new()
                } else {
                    format!("  {reason}")
                },
            ),
        };
        println!(
            "xtask timers: {:<8} {:<13} {}{}",
            standing.tier,
            if installed(standing.tier) {
                "installed"
            } else {
                "not installed"
            },
            detail,
            if standing.stale { "  [STALE]" } else { "" },
        );
    }

    // The history itself, because "did it fire at all?" is the question the
    // single-verdict file could never answer (§6 M82 row 4).
    let recent: Vec<&record::Event> = events.iter().rev().take(8).collect();
    if !recent.is_empty() {
        println!("xtask timers: last {} events, newest first:", recent.len());
        for e in recent {
            let what = match &e.kind {
                record::Kind::Started => "started".to_owned(),
                record::Kind::Finished { ok, secs, reason } => format!(
                    "{} after {}{}",
                    if *ok { "ok" } else { "RED" },
                    record::ago(*secs),
                    if reason.is_empty() {
                        String::new()
                    } else {
                        format!(" — {reason}")
                    },
                ),
            };
            println!(
                "xtask timers:   {}  {:<8} {what}",
                record::stamp(e.at),
                e.tier
            );
        }
    }

    record::report();
    println!(
        "xtask timers: `--install` writes them (Task Scheduler on Windows, systemd --user in \
         WSL); nothing installs them implicitly."
    );
    Ok(())
}

fn task_name(tier: &str) -> String {
    format!("GGEngine {tier}")
}

fn install() -> anyhow::Result<()> {
    std::fs::create_dir_all(status_dir())?;
    for tier in TIERS {
        // The pre-M82 verdict file, superseded by `history.txt`. Left in place
        // it would sit in the status directory reading `ok` beside a history
        // that says otherwise, which is the exact failure this milestone is
        // about — a word on disk that nothing keeps true.
        let _ = std::fs::remove_file(status_dir().join(format!("{}.txt", tier.name)));
        if cfg!(windows) {
            install_windows(tier.name, tier.windows_trigger)?;
        } else {
            install_systemd(tier.name, tier.on_calendar)?;
        }
    }
    status()
}

/// What the scheduler is asked to do, and the whole of it: run the tier, keep
/// its transcript.
///
/// Nothing about the **verdict** appears here any more. Until §6 M82 this line
/// ended `&& echo ok > … || echo RED > …`, which decides a run's outcome from
/// outside the run — so a launch the scheduler refused, and a run it killed on
/// `StopOnIdleEnd`, both left the previous verdict standing. The tier records
/// itself now (`record::around`), and this is the gate that keeps it that way.
fn windows_action(tier: &str, status: &std::path::Path) -> String {
    format!(
        "/c set GG_HEADLESS=1&amp; cargo xtask ci --{tier} 1&gt; \"{}\\{tier}.log\" 2&gt;&amp;1",
        status.display()
    )
}

/// Task Scheduler, via XML because the flags that matter are not command-line
/// options: `Priority 7` is below-normal, and the idle settings are the
/// politeness contract.
fn install_windows(tier: &str, trigger: &str) -> anyhow::Result<()> {
    let root = workspace_root();
    let status = status_dir();
    let action = windows_action(tier, &status);
    let xml = format!(
        r#"<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>GGEngine {tier} CI tier (PLAN.md §5). Idle-only, below-normal priority, headless.</Description>
  </RegistrationInfo>
  <Triggers>{trigger}</Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfIdle>true</RunOnlyIfIdle>
    <IdleSettings>
      <StopOnIdleEnd>true</StopOnIdleEnd>
      <RestartOnIdle>true</RestartOnIdle>
    </IdleSettings>
    <Priority>7</Priority>
    <ExecutionTimeLimit>PT6H</ExecutionTimeLimit>
  </Settings>
  <Actions>
    <Exec>
      <Command>cmd.exe</Command>
      <Arguments>{action}</Arguments>
      <WorkingDirectory>{root}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"#,
        root = root.display(),
    );
    let path = status.join(format!("{tier}-task.xml"));
    // UTF-16 LE with BOM: what the header declares, and what schtasks accepts.
    let mut bytes = vec![0xff, 0xfe];
    for unit in xml.encode_utf16() {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    std::fs::write(&path, bytes)?;
    run_capture(
        std::process::Command::new("schtasks").args([
            "/Create",
            "/TN",
            &task_name(tier),
            "/XML",
            &path.display().to_string(),
            "/F",
        ]),
        "schtasks /Create",
    )?;
    println!("xtask timers: installed `{}`", task_name(tier));
    Ok(())
}

/// systemd user units. `--user` and not system: the tier builds as this user,
/// against this user's toolchain and cargo home.
fn install_systemd(tier: &str, on_calendar: &str) -> anyhow::Result<()> {
    let root = workspace_root();
    let dir = dirs_config()?.join("systemd/user");
    std::fs::create_dir_all(&dir)?;
    let status = status_dir();
    std::fs::write(
        dir.join(format!("gg-{tier}.service")),
        format!(
            "[Unit]\n\
             Description=GGEngine {tier} CI tier (PLAN.md §5)\n\n\
             [Service]\n\
             Type=oneshot\n\
             WorkingDirectory={root}\n\
             Environment=GG_HEADLESS=1\n\
             # Politeness (§5): the dev box is also the gaming PC.\n\
             Nice=15\n\
             IOSchedulingClass=idle\n\
             ExecStart=/bin/sh -lc 'cargo xtask ci --{tier} > {status}/{tier}.log 2>&1'\n",
            root = root.display(),
            status = status.display(),
        ),
    )?;
    std::fs::write(
        dir.join(format!("gg-{tier}.timer")),
        format!(
            "[Unit]\n\
             Description=GGEngine {tier} CI tier (PLAN.md §5)\n\n\
             [Timer]\n\
             OnCalendar={on_calendar}\n\
             Persistent=true\n\n\
             [Install]\n\
             WantedBy=timers.target\n"
        ),
    )?;
    run_capture(
        std::process::Command::new("systemctl").args(["--user", "daemon-reload"]),
        "systemctl daemon-reload",
    )?;
    run_capture(
        std::process::Command::new("systemctl").args([
            "--user",
            "enable",
            "--now",
            &format!("gg-{tier}.timer"),
        ]),
        "systemctl enable --now",
    )?;
    println!("xtask timers: installed gg-{tier}.timer");
    Ok(())
}

fn uninstall() -> anyhow::Result<()> {
    for Tier { name: tier, .. } in TIERS {
        let result = if cfg!(windows) {
            run_capture(
                std::process::Command::new("schtasks").args([
                    "/Delete",
                    "/TN",
                    &task_name(tier),
                    "/F",
                ]),
                "schtasks /Delete",
            )
        } else {
            run_capture(
                std::process::Command::new("systemctl").args([
                    "--user",
                    "disable",
                    "--now",
                    &format!("gg-{tier}.timer"),
                ]),
                "systemctl disable --now",
            )
        };
        match result {
            Ok(_) => println!("xtask timers: removed {tier}"),
            Err(e) => println!("xtask timers: {tier} not removed ({e})"),
        }
    }
    Ok(())
}

fn dirs_config() -> anyhow::Result<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(xdg));
    }
    let home = std::env::var_os("HOME")
        .ok_or_else(|| anyhow::anyhow!("neither XDG_CONFIG_HOME nor HOME is set"))?;
    Ok(PathBuf::from(home).join(".config"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const WEEK: [&str; 7] = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ];

    /// The weekdays a Task Scheduler trigger names, in week order.
    fn windows_days(trigger: &str) -> Vec<&'static str> {
        WEEK.iter()
            .copied()
            .filter(|d| trigger.contains(&format!("<{d} />")))
            .collect()
    }

    /// The weekdays an `OnCalendar` names, in week order. Handles the two forms
    /// this table uses — a bare day and a `Mon..Sat` range.
    fn systemd_days(on_calendar: &str) -> Vec<&'static str> {
        let spec = on_calendar
            .split_whitespace()
            .next()
            .expect("an OnCalendar with a day spec");
        let index = |abbrev: &str| {
            WEEK.iter()
                .position(|d| d.starts_with(abbrev))
                .expect("a real weekday abbreviation")
        };
        let (from, to) = match spec.split_once("..") {
            Some((a, b)) => (index(a), index(b)),
            None => (index(spec), index(spec)),
        };
        WEEK[from..=to].to_vec()
    }

    /// The claim §6 M82 exists for. Before it, the nightly fired every day at
    /// 03:00 and the weekly — which *is* the nightly plus two gates — fired at
    /// 04:00 on Sunday, into a four-hour run still holding `xtask.exe`. Two
    /// tiers on one night is the defect whatever the hour, so the property
    /// asserted is a partition of the week and not a gap between two times.
    #[test]
    fn one_scheduled_run_a_night_and_never_two_at_once() {
        let mut claimed: Vec<&str> = Vec::new();
        for tier in TIERS {
            let days = windows_days(tier.windows_trigger);
            assert!(!days.is_empty(), "{} names no day", tier.name);
            for day in days {
                assert!(
                    !claimed.contains(&day),
                    "{day} is claimed twice — {} overlaps another tier, which is the collision \
                     that left the weekly red for four days (§6 M82 row 1)",
                    tier.name
                );
                claimed.push(day);
            }
        }
        assert_eq!(claimed.len(), 7, "every night belongs to exactly one tier");
    }

    /// Two hosts, two spellings, one intent — the shape §6 M81 found rotting
    /// everywhere it was written down twice.
    #[test]
    fn both_hosts_name_the_same_nights_and_the_same_hour() {
        for tier in TIERS {
            assert_eq!(
                windows_days(tier.windows_trigger),
                systemd_days(tier.on_calendar),
                "{} fires on different days depending on the host",
                tier.name
            );
            assert!(
                tier.windows_trigger.contains("T03:00:00")
                    && tier.on_calendar.contains(" 03:00:00"),
                "{} does not fire at 03:00 on both hosts",
                tier.name
            );
        }
    }

    /// The scheduler runs the tier and keeps its transcript. It does **not**
    /// decide whether the tier passed — that was the pre-M82 spelling, and it
    /// is unable to represent a launch that was refused or a run that was
    /// killed, because in both cases neither arm of its `&&`/`||` executes.
    #[test]
    fn the_scheduler_decides_nothing_about_the_verdict() {
        for tier in TIERS {
            let action = windows_action(tier.name, std::path::Path::new("C:/x"));
            assert!(
                action.contains(&format!("cargo xtask ci --{}", tier.name)),
                "{action}"
            );
            for forbidden in ["echo", "&amp;&amp;", "||"] {
                assert!(
                    !action.contains(forbidden),
                    "the scheduled action still decides a verdict (`{forbidden}`): {action}"
                );
            }
        }
    }

    /// Two tiers only, and they are the two `record` grades. One table.
    #[test]
    fn the_tier_table_is_the_one_record_reads() {
        assert!(TIERS.iter().all(|t| record::scheduled(t.name)));
        assert_eq!(TIERS.len(), 2);
    }
}
