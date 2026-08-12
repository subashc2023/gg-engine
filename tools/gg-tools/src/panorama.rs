//! A synthetic equirectangular `.hdr`, written where the caller asks (§6 M27).
//!
//! # Why the environment in this tree is generated and not photographed
//!
//! A captured HDRI is somebody else's work under somebody else's licence, and a
//! checked-in binary nobody can regenerate is the kind of asset that quietly
//! becomes load-bearing. This writes one from a formula, so the file in the tree
//! has a source, a diff that changes it is reviewable as *code*, and nothing in
//! the repository is a photograph we did not take.
//!
//! # What it is shaped like, and why that shape
//!
//! Structure a mirror can show and three SH bands provably cannot: a horizon
//! band, four coloured wall panels at cardinal headings, a bright strip of
//! windows, and a dim floor. It is deliberately **not** a sky with a sun in it —
//! §6 M24 put the sun in a [`Light`] precisely so a renderer never lights from
//! the same photon twice, and a panorama with a disc burnt into it would undo
//! that the moment a scene declared both.
//!
//! The dynamic range is the point of the format: the window strip runs an order
//! of magnitude above the walls and well past one, which is what makes the
//! difference between the prefiltered chain and the three-band convolution
//! *visible* rather than merely present — and what nothing eight bits deep could
//! carry.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
// `sim::log2` and not `f32`'s, for `ggc`'s reason one step earlier in the
// pipeline: this file is checked in, so the encoder that wrote it must give the
// same bytes on every host or a regenerated panorama is a diff nobody intended.
use gg_math::sim;

/// Default extent. 1024x512 is `ggc`'s working resolution (§4.6), so the
/// importer's box filter is a copy rather than a resample and what a reader sees
/// in the file is what the prefilter integrated.
const DEFAULT: (u32, u32) = (1024, 512);

pub fn run(args: &[String]) -> Result<()> {
    let mut out = PathBuf::from("target/gg-tools/panorama.hdr");
    let (mut width, mut height) = DEFAULT;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--out" => {
                out = rest
                    .next()
                    .context("--out wants a path")?
                    .parse()
                    .context("--out")?;
            }
            "--width" => width = rest.next().context("--width wants a number")?.parse()?,
            "--height" => height = rest.next().context("--height wants a number")?.parse()?,
            other => bail!("unknown argument {other}"),
        }
    }
    if width == 0 || height == 0 || width != height * 2 {
        bail!("an equirectangular panorama is 2:1 and non-empty, not {width}x{height}");
    }

    let texels = render(width, height);
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let bytes = encode(&texels, width, height);
    std::fs::write(&out, &bytes).with_context(|| format!("writing {}", out.display()))?;

    let peak = texels
        .iter()
        .flatten()
        .copied()
        .fold(0.0f32, |a, b| if b > a { b } else { a });
    let mean = texels.iter().flatten().sum::<f32>() / (texels.len() * 3) as f32;
    println!("panorama: {width}x{height}, {} bytes", bytes.len());
    println!(
        "  peak {peak:.1}, mean {mean:.3}, ratio {:.0}x",
        peak / mean
    );
    println!("  {}", out.display());
    Ok(())
}

/// The environment as a function of direction. Everything is a smooth blend of
/// bands except the window strip's ends, which are deliberately hard — a soft
/// edge everywhere would be an environment the SH path could nearly match.
fn render(width: u32, height: u32) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        // Latitude, 0 at the +Y pole. The room is upright, so almost everything
        // below is a function of this alone.
        let v = (y as f32 + 0.5) / height as f32;
        let altitude = (0.5 - v) * 2.0; // +1 up, -1 down
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            out.push(texel(u, altitude));
        }
    }
    out
}

fn texel(u: f32, altitude: f32) -> [f32; 3] {
    // The four walls, by cardinal heading. A hue per quadrant with a soft
    // crossfade at each corner: enough structure that a mirror shows *which way
    // it is facing*, which is the single clearest read on whether the octahedral
    // mapping is oriented the way both sides think it is.
    let quadrant = u * 4.0;
    let index = quadrant as usize % 4;
    let blend = smoothstep(quadrant.fract());
    //
    // The absolute levels matter as much as the hues, and they are set by what a
    // *mirror* must show rather than by what a diffuse surface gathers. A window
    // a hundred times its wall is physically ordinary and renders as a white
    // band on black — the walls crush out, and a reflection with no colour in it
    // demonstrates nothing the SH path could not already do. Ten to one keeps
    // both readable through one exposure, which is the range this environment
    // exists to be looked at in.
    const WALLS: [[f32; 3]; 4] = [
        [2.10, 1.50, 1.20], // warm plaster
        [1.00, 1.30, 1.70], // cool grey-blue
        [1.50, 1.70, 1.10], // olive
        [1.70, 1.10, 1.30], // dull rose
    ];
    let wall = lerp3(WALLS[index], WALLS[(index + 1) % 4], blend);

    if altitude > 0.55 {
        // The ceiling, unlit and nearly flat — the half of an environment that
        // makes an upward-facing surface dim rather than black.
        return scale(wall, 0.35);
    }
    if altitude < -0.35 {
        // The floor: darker still, and warm, because a floor is lit by the room
        // rather than by the windows.
        return scale([0.26, 0.22, 0.18], 2.5);
    }
    // The window strip, and the whole reason this file is not eight bits per
    // channel. Two orders of magnitude over the walls, hard-edged in longitude
    // so there are *four* of them rather than one continuous band — a mirror
    // reflecting four distinct sources is unmistakable, and three SH bands can
    // hold none of it.
    //
    // The strip is *narrow* on purpose, and that is a lighting decision rather
    // than a drawing one: solid angle times radiance is irradiance, so a band
    // wide enough to look generous is one that lights the room like an overcast
    // sky and leaves nothing for the sun to do. These panes cover about five per
    // cent of the sphere, which is roughly what a real room's windows do.
    let in_strip = (0.14..0.26).contains(&altitude);
    let pane = (u * 8.0).fract();
    if in_strip && (0.30..0.70).contains(&pane) {
        // Slightly cool, and brighter toward the top of the pane: a window is a
        // view of a sky, and a sky is brighter overhead.
        let up = (altitude - 0.14) / 0.12;
        return scale([0.92, 0.96, 1.0], 14.0 + 22.0 * up);
    }
    // The wall itself, picking up a little bounce from the strip above it.
    let bounce = (1.0 - (altitude - 0.20).abs() * 3.0).max(0.0);
    scale(wall, 1.0 + 2.5 * bounce)
}

fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn scale(c: [f32; 3], k: f32) -> [f32; 3] {
    [c[0] * k, c[1] * k, c[2] * k]
}

/// Radiance RGBE, uncompressed scanlines.
///
/// Flat rather than RLE on purpose: the format's run-length encoding is where
/// every third-party writer disagrees with every other, the file is 1.5 MB
/// either way, and this one exists to be *read back identically* by an importer
/// rather than to be small.
fn encode(texels: &[[f32; 3]], width: u32, height: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(texels.len() * 4 + 128);
    let _ = write!(out, "#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n");
    // -Y then +X: rows top to bottom, columns left to right — the same
    // orientation `ggc` assumes and `KTXorientation rd` records downstream.
    let _ = writeln!(out, "-Y {height} +X {width}");
    for texel in texels {
        out.extend_from_slice(&rgbe(*texel));
    }
    out
}

/// One texel as RGBE: a shared exponent and three 8-bit mantissas — Ward's
/// encoding, which every reader of this format implements the same way.
fn rgbe(c: [f32; 3]) -> [u8; 4] {
    let peak = c[0].max(c[1]).max(c[2]);
    if peak < 1e-32 {
        return [0, 0, 0, 0];
    }
    // The exponent that puts `peak / 2^e` in [0.5, 1), so the brightest channel
    // uses the top half of its byte and the quantization step is as fine as a
    // shared exponent allows.
    let exponent = sim::log2(peak).floor() as i32 + 1;
    let scale = 256.0 * exp2i(-exponent);
    [
        (c[0] * scale).clamp(0.0, 255.0) as u8,
        (c[1] * scale).clamp(0.0, 255.0) as u8,
        (c[2] * scale).clamp(0.0, 255.0) as u8,
        (exponent + 128) as u8,
    ]
}

fn exp2i(e: i32) -> f32 {
    // `powi` on a literal two rather than `exp2`, so the scaling is exact binary
    // and nothing here depends on a libm's rounding.
    2.0f32.powi(e)
}
