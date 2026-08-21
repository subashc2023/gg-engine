//! `gg-tools` — the instruments (CLAUDE.md, "Build the tool instead of doing it
//! by hand").
//!
//! Each subcommand answers one question that got asked twice by hand, prints a
//! report, and writes whatever it measured under `target/gg-tools/`. None of
//! them gate: a number that hardens into a threshold moves to `xtask`, and this
//! keeps the microscope.
//!
//! Usage, in dispatch order. This list, `main`'s match arms and the
//! unknown-subcommand message are one roster written three times, and `tests`
//! holds them to each other — it carried 25 of 30 until §6 M88, which is how
//! `map` and `transfer` were undocumented twice over (§6 M81 corrected
//! CLAUDE.md's copy and walked past this one).
//!
//!   shadow-bias [--json]   the §6 M11 acne knobs against an 8x-denser
//!                          reference — the plateau between them
//!   shadow-fit             what a cascade's radius costs the picture
//!   shadow-flat            demo 11's contact band: band against shade (§6 M20)
//!   shadow-sweep           what a turning camera does to the shadows in a room
//!                          standing still around it
//!   shadow-edge            a distant edge's straightness against its softness
//!   shadow-reach [--extent WxH]
//!                          the blocker a cascade has no depth for: dropped
//!                          casters against reach (§6 M60)
//!   lights                 what a light costs, over count and `r.clusters`
//!   lamps [--cost|--bias]  what a *casting* lamp costs, and where its bias
//!                          belongs — lost against leaked (§6 M31)
//!   cull                   what batch bounds bought the shadow passes, over
//!                          *pack* geometry (§6 M32)
//!   clicks                 the step across an audio event, over the step the
//!                          same signal makes on its own (§6 M77)
//!   furnace                whether a white metal gives back what it was given,
//!                          and along the lobe it should (§6 M33)
//!   split-sum              what [Laz13]'s fit costs and a table buys (§6 M34)
//!   ao [--crease]          how much occlusion a scene has, how far the pass
//!                          gets toward it, and whether a seam is a line (M71)
//!   bounce [--energy|--slow]
//!                          how much of a room's light bounced, against the
//!                          volume somebody drew by hand (§6 M36)
//!   field [--cost|--trace] the field's stability from a moving chair (§6 M57)
//!   facets                 how flat the field is on a flat wall — a second
//!                          difference, counted rather than averaged (§6 M69)
//!   frame [--extent WxH|--sweep|--attribute|--devices a,b]
//!                          where a frame's milliseconds go: per-pass device
//!                          time against the host's own zones (§6 M58); the
//!                          ms/Mpx table `r.scale` is read off (M78); each row
//!                          falsified by switching its pass off (M79)
//!   governor               does the automatic render scale settle — the
//!                          shipped controller against the obvious one (§6 M80)
//!   views [--pack P --scene N --eye x,y,z --yaw r --set k=v]
//!                          one render per `r.debug_view` entry, and which
//!                          views this frame had (§6 M59)
//!   banding                what 8-bit output does to a gradient, over
//!                          `r.dither`
//!   pace [--editor]        what a display rate does to a turn the hand made at
//!                          a constant speed; `--editor` asks it of the
//!                          editor's camera, a second eye down a second
//!                          composition (§6 M65)
//!   hash-scale             what the per-tick full-world passes cost at the
//!                          scale §6 M38 brings
//!   orbit                  whether §2's two regimes agree about one body —
//!                          shape against phase, failing opposite ways
//!   map                    demo 13's schematic under one inverse-square lux:
//!                          blown against crushed (§6 M38)
//!   panorama [--out P]     write the synthetic equirectangular `.hdr`
//!   icon [--game G]        write a demo's taskbar picture (§6 M46, M75)
//!   timbre [--out D]       write demo 10's clips, and grade what one `Sound`
//!                          cannot reach (§6 M43)
//!   transfer               the burn a pilot cannot see (§6 M38 item 14)
//!   fp-isa [--target T]    which floating-point instructions the determinism
//!                          path holds, by how much freedom the ISA leaves them
//!   mcp                    serve a session's reload record to an agent over
//!                          MCP on stdio (§6 M16)

mod ao;
mod banding;
mod bounce;
mod clicks;
mod cull;
mod facets;
mod field;
mod fp_isa;
mod frame;
mod furnace;
mod governor;
mod hash_scale;
mod icon;
mod lamps;
mod lights;
mod map;
mod mcp;
mod orbit;
mod pace;
mod panorama;
mod radiance;
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
        "clicks" => clicks::run(rest),
        "furnace" => furnace::run(rest),
        "split-sum" => split_sum::run(rest),
        "ao" => ao::run(rest),
        "bounce" => bounce::run(rest),
        "field" => field::run(rest),
        "facets" => facets::run(rest),
        "frame" => frame::run(rest),
        "governor" => governor::run(rest),
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
        // Dispatch order, held to the arms above and to the `//!` header by
        // `tests` — three copies of one list is §6 M86's class, and the header
        // was five names short of these two when M88 checked (§6 M87).
        other => {
            anyhow::bail!(
                "unknown subcommand {other:?} — the roster is: shadow-bias, shadow-fit, \
                 shadow-flat, shadow-sweep, shadow-edge, shadow-reach, lights, lamps, cull, \
                 clicks, furnace, split-sum, ao, bounce, field, facets, frame, governor, views, \
                 banding, pace, hash-scale, orbit, map, panorama, icon, timbre, transfer, fp-isa, \
                 mcp. A new instrument is a new subcommand here, not a new crate"
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

/// The subcommands `main` dispatches, read out of this file's own source.
///
/// A text scan for the same reason `xtask`'s cross-file gates are one (§6 M87):
/// the alternative is a table the match arms are generated from, which buys
/// nothing a rename cannot break and costs the plainest `match` in the tree.
#[cfg(test)]
fn rosters() -> (Vec<String>, Vec<String>, Vec<String>) {
    let source = include_str!("main.rs");
    let dispatched = source
        .lines()
        .filter_map(|l| l.trim().strip_suffix("::run(rest),"))
        .filter_map(|l| l.split_once("\" => "))
        .filter_map(|(name, _)| name.strip_prefix('"'))
        .map(str::to_owned)
        .collect();
    // A header entry starts in the first text column; its continuations are
    // indented past it, which is what keeps a wrapped description out of this.
    let documented = source
        .lines()
        .filter_map(|l| l.strip_prefix("//!   "))
        .filter(|l| !l.starts_with(' '))
        .filter_map(|l| l.split_whitespace().next())
        .map(str::to_owned)
        .collect();
    let named = source
        .split_once("the roster is: ")
        .and_then(|(_, tail)| tail.split_once(". A new instrument"))
        .map(|(list, _)| {
            list.split(',')
                // `\` is the source's line continuation, not part of a name.
                .map(|n| {
                    n.split_whitespace()
                        .filter(|t| *t != "\\")
                        .collect::<String>()
                })
                .collect()
        })
        .unwrap_or_default();
    (dispatched, documented, named)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    /// The `//!` header, `main`'s arms and the unknown-subcommand message are
    /// one roster written three times. Ordered, not set-wise: three orderings
    /// of one list is the drift §6 M86 deleted elsewhere.
    #[test]
    fn every_subcommand_is_dispatched_documented_and_named_in_the_same_order() {
        let (dispatched, documented, named) = super::rosters();
        // Each scan matches a shape this file could stop having (§6 M87) — a
        // reformatted match, a rewrapped header, a reworded bail — and an empty
        // population would make all three comparisons below hold vacuously.
        assert!(
            !dispatched.is_empty() && !documented.is_empty() && !named.is_empty(),
            "a roster scan found nothing: {} dispatched, {} documented, {} named — the parse \
             stopped matching this file rather than the file losing its subcommands",
            dispatched.len(),
            documented.len(),
            named.len()
        );
        assert_eq!(
            dispatched, documented,
            "the `//!` usage header and `main`'s match arms disagree — a subcommand is only \
             reachable by reading the source, which is how `map` and `transfer` went undocumented \
             for a year (§6 M81, M88)"
        );
        assert_eq!(
            dispatched, named,
            "the unknown-subcommand message and `main`'s match arms disagree — a typo would be \
             answered with a roster that does not name the command the user wanted"
        );
    }
}
