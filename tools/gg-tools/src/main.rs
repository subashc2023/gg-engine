//! `gg-tools` — the instruments (CLAUDE.md, "Build the tool instead of doing it
//! by hand").
//!
//! Each subcommand answers one question that got asked twice by hand, prints a
//! report, and writes whatever it measured under `target/gg-tools/`. None of
//! them gate: a number that hardens into a threshold moves to `xtask`, and this
//! keeps the microscope.
//!
//! Usage:
//!   gg-tools shadow-bias [--json]   sweep the §6 M11 acne knobs against an
//!                                   8x-denser reference and print the plateau
//!   gg-tools shadow-fit             what the single cascade's radius costs the
//!                                   picture, per radius, against a tight leg

mod shadow_bias;
mod shadow_fit;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let (command, rest) = args
        .split_first()
        .map_or(("", &[][..]), |(c, r)| (c.as_str(), r));
    match command {
        "shadow-bias" => shadow_bias::run(rest),
        "shadow-fit" => shadow_fit::run(rest),
        other => {
            anyhow::bail!(
                "unknown subcommand {other:?} — the roster is: shadow-bias, shadow-fit. \
                 A new instrument is a new subcommand here, not a new crate"
            )
        }
    }
}

/// Where an instrument leaves what it measured. Under `target/` because it is
/// build output, not a reference — anything worth keeping gets archived
/// explicitly the way `xtask bench --record` archives.
pub fn output_dir() -> anyhow::Result<std::path::PathBuf> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .ok_or_else(|| anyhow::anyhow!("no workspace root above the manifest"))?
        .join("target/gg-tools");
    std::fs::create_dir_all(&root)?;
    Ok(root)
}
