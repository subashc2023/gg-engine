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
//!   gg-tools shadow-edge            how straight a distant shadow's edge is,
//!                                   against how soft the filter made it
//!   gg-tools shadow-reach [--pack]  the blocker a cascade has no depth for
//!                                   (§6 M60): dropped casters against reach
//!   gg-tools lights                 what a light costs a frame, swept over
//!                                   count and over `r.clusters` (§6 M30)
//!   gg-tools lamps [--cost|--bias]  what a *casting* lamp costs, swept over
//!                                   `r.lamps` and face size, and where its
//!                                   bias belongs — lost against leaked (§6 M31)
//!   gg-tools cull                  what giving a batch bounds bought the
//!                                   shadow passes, over *pack* geometry (§6 M32)
//!   gg-tools furnace                whether a white metal gives back what it
//!                                   was given, and whether the lobe it gives it
//!                                   back along points where the lobe is (§6 M33)
//!   gg-tools split-sum              what [Laz13]'s fit costs where §6 M33 spends
//!                                   it, and what a table buys back (§6 M34)
//!   gg-tools ao                     how much occlusion a scene has, and how far
//!                                   a depth buffer gets toward it (§6 M35)
//!   gg-tools bounce [--slow]        how much of a room's light got there by
//!                                   bouncing, against the volume somebody drew
//!                                   by hand (§6 M36)
//!   gg-tools facets                 how flat the field's answer is on a flat
//!                                   wall — a second difference at a percentile,
//!                                   which is a crease's own number (§6 M69)
//!   gg-tools frame [--extent WxH]   where a frame's milliseconds go: the
//!                                   per-pass device table against the host's
//!                                   own zones (§6 M58)
//!   gg-tools banding                what the 8-bit output does to a smooth
//!                                   gradient, swept over `r.dither`
//!   gg-tools pace [--editor]        what a display rate does to a turn the hand
//!                                   made at a constant speed; `--editor` asks
//!                                   the same of the editor's camera, which is a
//!                                   second eye down a second composition
//!   gg-tools hash-scale             what the per-tick full-world passes cost at
//!                                   the scale §6 M38 brings — the contract's
//!                                   sorted walk against a storage-order floor
//!   gg-tools orbit                  whether §2's two regimes agree about the
//!                                   same body — the orbit's shape against its
//!                                   phase, which fail in opposite directions
//!   gg-tools panorama [--out P]   write the synthetic equirectangular `.hdr`
//!   gg-tools icon [--out P]       write demo 10's taskbar picture
//!   gg-tools timbre [--out D]     write demo 10's clips, and grade what a tone cannot reach
//!                                   demo 06's environment is compiled from
//!   gg-tools fp-isa [--target T]    which floating-point instructions the
//!                                   determinism path contains, by how much
//!                                   freedom the ISA leaves them (§8's qemu row)
//!   gg-tools mcp                    serve a running session's reload record to
//!                                   an agent over MCP on stdio (§6 M16)

mod ao;
mod banding;
mod bounce;
mod cull;
mod facets;
mod field;
mod fp_isa;
mod frame;
mod furnace;
mod hash_scale;
mod icon;
mod lamps;
mod lights;
mod map;
mod mcp;
mod orbit;
mod pace;
mod panorama;
mod shadow_bias;
mod shadow_edge;
mod shadow_fit;
mod shadow_flat;
mod shadow_image;
mod shadow_reach;
mod shadow_sweep;
mod split_sum;
mod timbre;
mod transfer;
mod views;

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
        "shadow-edge" => shadow_edge::run(rest),
        "shadow-reach" => shadow_reach::run(rest),
        "lights" => lights::run(rest),
        "lamps" => lamps::run(rest),
        "cull" => cull::run(rest),
        "furnace" => furnace::run(rest),
        "split-sum" => split_sum::run(rest),
        "ao" => ao::run(rest),
        "bounce" => bounce::run(rest),
        "field" => field::run(rest),
        "facets" => facets::run(rest),
        "frame" => frame::run(rest),
        "views" => views::run(rest),
        "banding" => banding::run(rest),
        "pace" => pace::run(rest),
        "hash-scale" => hash_scale::run(rest),
        "orbit" => orbit::run(rest),
        "map" => map::run(rest),
        "panorama" => panorama::run(rest),
        "icon" => icon::run(rest),
        "timbre" => timbre::run(rest),
        "transfer" => transfer::run(rest),
        "fp-isa" => fp_isa::run(rest),
        "mcp" => mcp::run(rest),
        other => {
            anyhow::bail!(
                "unknown subcommand {other:?} — the roster is: shadow-bias, shadow-fit, \
                 shadow-flat, shadow-sweep, shadow-edge, shadow-reach, lights, lamps, cull, furnace, \
                 split-sum, ao, bounce, field, facets, frame, views, banding, pace, hash-scale, orbit, map, \
                 panorama, icon, timbre, transfer, fp-isa, mcp. A new instrument is a new \
                 subcommand here, not a new crate"
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
