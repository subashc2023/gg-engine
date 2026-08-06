//! `cargo xtask timers [--install|--status|--uninstall]` — the scheduled tiers
//! (§5, "Scheduling and politeness"), deferred from M0A to M4B because until
//! this milestone there were no nightly-tier gates to schedule (§6 M0A).
//!
//! Two hosts, two mechanisms, one shape: nightly at 03:00 and weekly on Sunday
//! at 04:00, both at **below-normal priority**, both under `GG_HEADLESS=1`, both
//! writing a status artifact under `target/ci-status/`. Windows uses Task
//! Scheduler through a generated XML — `schtasks /Create` on a command line
//! cannot express idle-only or priority, and those are the politeness half of
//! the contract, not decoration. WSL uses systemd user timers.
//!
//! **Installing a timer changes the machine**, so this command does exactly
//! what its flag says and nothing implicitly: `--status` reads, `--install`
//! writes, `--uninstall` removes. No tier calls it.
//!
//! The politeness contract (§5): the dev machine is also the gaming PC. A
//! nightly that steals frames from whatever is on the screen is worse than no
//! nightly, so the tasks run only when idle, stop when the machine stops being
//! idle, and never create a window (§1.5 — the tiers guarantee that themselves).

use crate::util::{run_capture, workspace_root};
use std::path::PathBuf;

/// What the timers run, and when: `(tier, complete Task Scheduler trigger,
/// systemd OnCalendar)`. Whole elements rather than fragments, so a schedule
/// change is one edit here and not a templating puzzle.
const TIERS: &[(&str, &str, &str)] = &[
    (
        "nightly",
        "<CalendarTrigger><StartBoundary>2026-01-01T03:00:00</StartBoundary><Enabled>true</Enabled>\
         <ScheduleByDay><DaysInterval>1</DaysInterval></ScheduleByDay></CalendarTrigger>",
        "*-*-* 03:00:00",
    ),
    (
        "weekly",
        "<CalendarTrigger><StartBoundary>2026-01-04T04:00:00</StartBoundary><Enabled>true</Enabled>\
         <ScheduleByWeek><DaysOfWeek><Sunday /></DaysOfWeek><WeeksInterval>1</WeeksInterval>\
         </ScheduleByWeek></CalendarTrigger>",
        "Sun *-*-* 04:00:00",
    ),
];

pub fn run(args: &[&str]) -> anyhow::Result<()> {
    match args {
        a if a.contains(&"--install") => install(),
        a if a.contains(&"--uninstall") => uninstall(),
        _ => status(),
    }
}

/// Where a run leaves its verdict. Read by `--status`, and the artifact §5 asks
/// a scheduled tier to write.
pub fn status_dir() -> PathBuf {
    workspace_root().join("target/ci-status")
}

fn status() -> anyhow::Result<()> {
    for (tier, _, _) in TIERS {
        let installed = if cfg!(windows) {
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
        };
        let last = std::fs::read_to_string(status_dir().join(format!("{tier}.txt")))
            .unwrap_or_else(|_| "never run".into());
        println!(
            "xtask timers: {tier:<8} {:<13} last: {}",
            if installed {
                "installed"
            } else {
                "not installed"
            },
            last.trim(),
        );
    }
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
    for (tier, windows_schedule, on_calendar) in TIERS {
        if cfg!(windows) {
            install_windows(tier, windows_schedule)?;
        } else {
            install_systemd(tier, on_calendar)?;
        }
    }
    status()
}

/// Task Scheduler, via XML because the flags that matter are not command-line
/// options: `Priority 7` is below-normal, and the idle settings are the
/// politeness contract.
fn install_windows(tier: &str, trigger: &str) -> anyhow::Result<()> {
    let root = workspace_root();
    let status = status_dir();
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
      <Arguments>/c set GG_HEADLESS=1&amp; cargo xtask ci --{tier} 1&gt; "{status}\{tier}.log" 2&gt;&amp;1 &amp;&amp; echo ok&gt; "{status}\{tier}.txt" || echo RED&gt; "{status}\{tier}.txt"</Arguments>
      <WorkingDirectory>{root}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"#,
        status = status.display(),
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
             ExecStart=/bin/sh -lc 'cargo xtask ci --{tier} > {status}/{tier}.log 2>&1 && \
             echo ok > {status}/{tier}.txt || (echo RED > {status}/{tier}.txt; exit 1)'\n",
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
    for (tier, _, _) in TIERS {
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
