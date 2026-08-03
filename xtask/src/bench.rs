//! `cargo xtask bench` (§4.11) — two tiers, because a benchmark and a smoke
//! test want opposite things from a machine.
//!
//! Plain, it is the nightly gate (§5 gate 11): do the benches still run, and do
//! the `gg-ecs` ratios still hold? That leg is happy on the pinned lavapipe,
//! whose timings §4.11 rules out as performance data in as many words — a
//! software rasterizer measures LLVM, not the frame.
//!
//! `--record` is the other thing: the real numbers, from our own machine's real
//! GPU, under the **instrumented** tier — which is what §2 means by "the only
//! build performance numbers come from" — archived per machine so the diff of
//! the archive *is* the regression report. Two records, one per shape: the
//! `gg-ecs` micro (§4.2.2's seam, ratios against a native loop) and §4.11's
//! frame macro through `gg-golden`.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::util::{cargo, run as exec, run_capture, workspace_root};

pub fn run(args: &[&str]) -> anyhow::Result<()> {
    if args.contains(&"--record") {
        record()
    } else {
        smoke()
    }
}

/// Frames a recording measures. The macro's own default; named here so the
/// smoke leg's short run reads as the deliberate difference it is.
const RECORD_FRAMES: &str = "300";
const SMOKE_FRAMES: &str = "24";

/// The CPU micros of §4.11, as `(package, bench target, archive key)`.
///
/// Harness-free, every one of them: each asserts *ratios* measured in its own
/// process, which is what makes the numbers survive a noisy desk and keeps a
/// benchmark framework out of crates with dependency budgets (§3, §4.11).
const MICROS: &[(&str, &str, &str)] = &[
    ("gg-ecs", "query", "ecs"),
    ("gg-extract", "extract", "extract"),
    ("gg-assets", "load", "assets"),
    // §6 M13's "no frame-time regression": the overlay's row shape against the
    // one it replaced, both timed in the same run.
    ("gg-debug", "overlay", "overlay"),
    // §6 M14's last exit row, and the one micro whose assertion is an absolute
    // budget rather than a ratio: ten thousand entities save and load inside one
    // frame. A frame is three orders of magnitude above timer noise, which is
    // what makes the exception safe.
    ("demo-05-many", "save", "save"),
];

fn smoke() -> anyhow::Result<()> {
    for (package, bench, _) in MICROS {
        exec(
            cargo().args(["bench", "-p", package, "--bench", bench]),
            &format!("cargo bench -p {package} --bench {bench}"),
        )?;
    }
    let mut macro_run = cargo();
    macro_run
        .args(["run", "-q", "-p", "gg-golden", "--", "bench", "--frames"])
        .arg(SMOKE_FRAMES);
    if cfg!(windows) {
        macro_run.env("VK_DRIVER_FILES", crate::probe::ensure_lavapipe()?);
    } else {
        macro_run.env("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/lvp_icd.json");
    }
    exec(
        &mut macro_run,
        "gg-golden frame macro (smoke, pinned lavapipe)",
    )?;
    println!(
        "xtask bench: smoke only — the frame macro ran on the pinned lavapipe and its \
         milliseconds are not performance data (§4.11). Real numbers: `cargo xtask bench --record` \
         on a box with a GPU."
    );
    Ok(())
}

/// A device name that means the frame was drawn by a CPU. Recording one would
/// archive a number that moves with an LLVM release.
fn is_software(device: &str) -> bool {
    let lower = device.to_lowercase();
    ["llvmpipe", "lavapipe", "swiftshader", "software"]
        .iter()
        .any(|name| lower.contains(name))
}

fn record() -> anyhow::Result<()> {
    // The micro first: it needs no GPU, so a machine without one still records
    // half a baseline rather than nothing. `cargo bench` runs it under the
    // `bench` profile, which inherits `instrumented` (§3) — the tier §4.11 says
    // benchmarks are defined to run under.
    println!("xtask bench: recording the CPU micro baselines (bench profile = instrumented)");
    // Named bench targets and not whole packages: a bare `cargo bench -p` also
    // runs the lib's unit tests under libtest, which rejects `--json` as an
    // option it does not have.
    let mut micros = Vec::new();
    for (package, bench, key) in MICROS {
        let out = run_capture(
            cargo().args(["bench", "-p", package, "--bench", bench, "--", "--json"]),
            &format!("cargo bench -p {package} --bench {bench} -- --json"),
        )?;
        let record = json_line(&out)
            .ok_or_else(|| anyhow::anyhow!("the {package} bench emitted no JSON record:\n{out}"))?
            .to_owned();
        micros.push(((*key).to_owned(), record));
    }

    // No `VK_DRIVER_FILES`: a recording wants whatever this desk's best device
    // is (or `GG_ADAPTER`'s pick), which is the opposite of every other gate.
    println!("xtask bench: recording the frame macro on this machine's own device");
    let macro_out = run_capture(
        cargo().args([
            "run",
            "-q",
            "-p",
            "gg-golden",
            "--profile",
            "instrumented",
            "--",
            "bench",
            "--json",
            "--frames",
            RECORD_FRAMES,
        ]),
        "gg-golden frame macro (instrumented, real device)",
    )?;
    let frame = json_line(&macro_out)
        .ok_or_else(|| anyhow::anyhow!("the frame macro emitted no JSON record:\n{macro_out}"))?;
    let device = field(frame, "device").unwrap_or_default();

    // §6 M10's exit row: ten thousand parented objects, culled, at under 4 ms of
    // CPU frame. A second run rather than a second scene inside one, because the
    // pack has to stream in first and a warmup that included a load would put a
    // one-off in the percentiles.
    println!("xtask bench: recording the ten-thousand-object frame (§6 M10)");
    crate::assets::run(&[])?;
    let field_out = run_capture(
        cargo().args([
            "run",
            "-q",
            "-p",
            "gg-golden",
            "--profile",
            "instrumented",
            "--",
            "bench",
            "field",
            "--json",
            "--frames",
            RECORD_FRAMES,
        ]),
        "gg-golden field macro (instrumented, real device)",
    )?;
    let field_frame = json_line(&field_out).ok_or_else(|| {
        anyhow::anyhow!(
            "the field macro emitted no JSON record:
{field_out}"
        )
    })?;
    anyhow::ensure!(
        !is_software(&device),
        "refusing to archive a baseline measured on `{device}` — software-rasterizer timings are \
         not performance data (§4.11). Unset VK_DRIVER_FILES, or point GG_ADAPTER at a real device."
    );

    let machine = machine_id();
    let path = workspace_root()
        .join("bench")
        .join(format!("{machine}.json"));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, archive(&machine, &micros, frame, field_frame)?)?;

    // The human half, straight from the run that was archived — a recording that
    // only writes a file leaves the operator with nothing to react to.
    println!("\n{}", macro_out.trim());
    println!("\n{}", field_out.trim());
    println!(
        "\nxtask bench: baseline archived to {} on `{device}`. It is one file per machine, \
         rewritten in place: the git diff of this file is the regression report (§4.11).",
        path.display()
    );
    Ok(())
}

/// The record. Provenance first, because a number without the machine, the
/// commit and the toolchain that produced it is not a baseline — it is a
/// rumour, and §5's recovery bar says baselines are re-recorded rather than
/// compared when the box changes.
fn archive(
    machine: &str,
    micros: &[(String, String)],
    frame: &str,
    field: &str,
) -> anyhow::Result<String> {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let rustc = run_capture(
        std::process::Command::new("rustc").arg("--version"),
        "rustc --version",
    )?;
    let commit = run_capture(
        std::process::Command::new("git")
            .current_dir(workspace_root())
            .args(["rev-parse", "--short", "HEAD"]),
        "git rev-parse HEAD",
    )?;
    // Schema 2: the micros became a set rather than one entry, so a reader that
    // knew schema 1 would look for a top-level `ecs` key and quietly find none.
    let micro_json: String = micros
        .iter()
        .map(|(key, record)| format!("  \"{key}\": {record},\n"))
        .collect();
    Ok(format!(
        "{{\n  \"schema\": 2,\n  \"recorded\": \"{}\",\n  \"unix_time\": {secs},\n  \
         \"machine\": \"{machine}\",\n  \"commit\": \"{}\",\n  \"rustc\": \"{}\",\n  \
         \"profile\": \"instrumented\",\n{micro_json}  \"frame\": {frame},\n  \
         \"field\": {field}\n}}\n",
        date(secs),
        commit.trim(),
        rustc.trim()
    ))
}

/// `YYYY-MM-DD` from a Unix timestamp — Howard Hinnant's civil-from-days, which
/// is a dozen lines against a date-time crate for one string in one file.
#[expect(
    clippy::cast_possible_wrap,
    reason = "timestamps in range; the algorithm is integer arithmetic throughout"
)]
fn date(secs: u64) -> String {
    let days = (secs / 86_400) as i64 + 719_468;
    let era = days.div_euclid(146_097);
    let doe = days.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}")
}

/// This box, as the archive's key. Hostname plus architecture: §5 keys
/// baselines to a machine, and the same hostname on two architectures is a
/// machine the numbers do not transfer between.
pub(crate) fn machine_id() -> String {
    let host = std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .ok()
        .or_else(|| {
            std::fs::read_to_string("/proc/sys/kernel/hostname")
                .ok()
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());
    let clean: String = host
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect();
    format!(
        "{clean}-{}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    )
}

/// The last line that is a JSON object. Both benches print a human table first
/// and their record last, so nothing has to be quiet to be parseable.
fn json_line(output: &str) -> Option<&str> {
    output
        .lines()
        .map(str::trim)
        .rfind(|line| line.starts_with('{') && line.ends_with('}'))
}

/// One string field out of a flat JSON record — enough to read the device name
/// back, and not a parser.
fn field(record: &str, name: &str) -> Option<String> {
    let key = format!("\"{name}\":\"");
    let at = record.find(&key)? + key.len();
    Some(record[at..].split('"').next()?.to_string())
}
