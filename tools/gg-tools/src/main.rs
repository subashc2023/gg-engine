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
//!   gg-tools shadow-flat            the flat camera's contact band (§6 M20):
//!                                   band vs shade over bias and fit
//!   gg-tools shadow-sweep           what a turning camera does to the shadows
//!                                   in a room standing still around it
//!   gg-tools pace                   what a display rate does to a turn the hand
//!                                   made at a constant speed
//!   gg-tools fp-isa [--target T]    which floating-point instructions the
//!                                   determinism path contains, by how much
//!                                   freedom the ISA leaves them (§8's qemu row)
//!   gg-tools mcp                    serve a running session's reload record to
//!                                   an agent over MCP on stdio (§6 M16)

mod fp_isa;
mod mcp;
mod pace;
mod shadow_bias;
mod shadow_fit;
mod shadow_flat;
mod shadow_image;
mod shadow_sweep;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        // stderr, not the default stdout: `mcp` speaks line-delimited JSON-RPC
        // on stdout, and one stray log line is a parse error at the client. The
        // other instruments are unaffected — they report with `println!`, and a
        // log on stderr is where a log belongs anyway.
        .with_writer(std::io::stderr)
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
        "shadow-flat" => shadow_flat::run(rest),
        "shadow-sweep" => shadow_sweep::run(rest),
        "pace" => pace::run(rest),
        "fp-isa" => fp_isa::run(rest),
        "mcp" => mcp::run(rest),
        other => {
            anyhow::bail!(
                "unknown subcommand {other:?} — the roster is: shadow-bias, shadow-fit, \
                 shadow-flat, shadow-sweep, pace, fp-isa, mcp. A new instrument is a new \
                 subcommand here, not a \
                 new crate"
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
