//! `gg-tools shadow-reach` — the blocker a cascade has no depth for (§6 M60).
//!
//! Every other shadow instrument here grades a *quality*: how wide an edge is,
//! how much acne a bias leaves, how a fit holds up as the camera turns. This one
//! grades a **correctness** bound, and the difference matters because the failure
//! it names does not look like a bad shadow. It looks like no shadow.
//!
//! A cascade is an orthographic slab with a light eye some distance up-light of
//! its centre. A caster further up-light than that eye is not rasterized into the
//! map at all, and the shader reads absent depth as absent blocker — so the
//! receiver renders lit. Until M60 the distance was `radius * 2`, derived from
//! the cascade's *own* width, which makes it shortest exactly where the cascade
//! is tightest: the near field, which is most of the screen. A room with a
//! ceiling five metres up and a near cascade two metres wide has no ceiling in
//! its shadow map.
//!
//! It also gets worse as the range comes *down*, which is the tuning move that
//! improves every other number — Sponza's gallery is correctly shadowed at the
//! shipped `r.shadow_distance 80` and sunlit at 20.
//!
//! Two numbers, failing in opposite directions:
//!
//! - **dropped** — casters over a cascade's footprint sitting further up-light
//!   than it records. Truth is zero and no reference renderer is needed to say
//!   so: the caster is in the frame's own extracted set, and either its depth is
//!   in the map or it is not.
//! - **reach** — the depth range itself, in metres. Long is not free: it is the
//!   scale the blocker search turns a depth difference back into a distance with,
//!   and the along-light extent every caster cull tests against.
//!
//! The third table is the picture, because a count of dropped casters says
//! nothing about how much of the screen one of them shades. `r.shadow_reach` is
//! the same frame a flag apart (`r.shadow_cull`'s argument, §6 M32), so the
//! difference between the two renders *is* the defect, in pixels.

use anyhow::Result;
use gg_render::{OffscreenRenderer, cvars};

use crate::views;

/// 16:9, and the aspect is part of the measurement: the fitted cascade's width
/// comes out of the frustum slice, so it moves the reach as well as the texel.
const EXTENT: (u32, u32) = (1280, 720);

/// A pixel counts as changed when any channel moves by more than this. Above
/// dither and above the field's own limit cycle (§6 M59), below anything a lost
/// blocker does — a surface that was shadowed and is now lit moves by tens.
const CHANGED: u8 = 8;

/// The range sweep. Down rather than up on purpose: tightening the range is the
/// move an operator makes to sharpen shadows, and it is the move that breaks
/// them.
const DISTANCES: &[f64] = &[80.0, 40.0, 20.0, 10.0];

pub fn run(args: &[String]) -> Result<()> {
    let extent = match views::flag(args, "--extent") {
        Some(text) => parse_extent(&text)?,
        None => EXTENT,
    };
    views::apply_sets(args)?;
    let scene = views::scene_from(args)?;
    let mut renderer = OffscreenRenderer::new(extent)?;
    let label = views::open(&mut renderer, &scene)?;
    // Read after `apply_sets`, so `--set r.shadow_distance=X` names the row the
    // first table reports and the row every sweep below returns to.
    let shipped = cvars::SHADOW_DISTANCE.float();
    println!(
        "{label} at {}x{} on {}, r.shadow_distance {shipped}",
        extent.0,
        extent.1,
        renderer.device().chosen
    );

    // A pack streams, so nothing below describes the scene until it is resident.
    for _ in 0..views::WARMUP {
        let extracted = views::extract(&scene, extent, renderer.scenes())?;
        renderer.frame(&extracted, &scene.view, [0.0; 4], &[])?;
    }
    let extracted = views::extract(&scene, extent, renderer.scenes())?;
    println!(
        "\n  {} caster(s), {} of them off screen\n",
        extracted.instances.len() + extracted.models.len(),
        extracted.casting_only()
    );

    println!("  the fit, at the shipped range");
    println!("  reach | cas | radius m | reach m | needed m | over | dropped");
    for on in [false, true] {
        cvars::SHADOW_REACH.set_bool(on);
        for (index, c) in gg_render::cascade_reach(&extracted, &scene.view, extent)
            .iter()
            .enumerate()
        {
            println!(
                "  {:<5} |  {index}  | {:8.2} | {:7.1} | {:8.1} | {:4} | {:7}",
                on as u8, c.radius, c.reach, c.needed, c.over, c.dropped
            );
        }
    }

    println!("\n  swept over r.shadow_distance — dropped casters per cascade");
    println!("  reach | range m | cas 0 | cas 1 | cas 2 | cas 3 | reach m");
    for on in [false, true] {
        cvars::SHADOW_REACH.set_bool(on);
        for distance in DISTANCES {
            cvars::SHADOW_DISTANCE.set_float(*distance);
            // The sweep changes the caster set as well as the fit: `cast_shadows`
            // takes its reach from the same CVars, so re-extracting is not
            // optional — a stale `Extracted` would grade the new cascades against
            // the old frame's casters.
            let extracted = views::extract(&scene, extent, renderer.scenes())?;
            let report = gg_render::cascade_reach(&extracted, &scene.view, extent);
            let at = |i: usize| report.get(i).map_or(0, |c| c.dropped);
            println!(
                "  {:<5} | {distance:7.0} | {:5} | {:5} | {:5} | {:5} | {:7.1}",
                on as u8,
                at(0),
                at(1),
                at(2),
                at(3),
                report.first().map_or(0.0, |c| c.reach)
            );
        }
    }
    cvars::SHADOW_DISTANCE.set_float(shipped);

    println!("\n  the same frame, a flag apart — what the drop costs the picture");
    println!("  range m | changed px | mean |dL| | file");
    let out = std::path::PathBuf::from("target/gg-tools");
    std::fs::create_dir_all(&out)?;
    for distance in DISTANCES {
        cvars::SHADOW_DISTANCE.set_float(*distance);
        let mut frames = Vec::new();
        for on in [false, true] {
            cvars::SHADOW_REACH.set_bool(on);
            // A full warmup per leg, not one frame: the probe field carries state
            // across frames and every leg above has moved the range under it, so
            // anything shorter photographs the transition rather than the frame
            // (§6 M57).
            let mut pixels = Vec::new();
            for _ in 0..views::WARMUP {
                let extracted = views::extract(&scene, extent, renderer.scenes())?;
                pixels = renderer
                    .frame(&extracted, &scene.view, [0.0; 4], &[])?
                    .pixels;
            }
            let path = out.join(format!("shadow-reach-{distance:.0}m-{}.png", on as u8));
            views::write_png(&pixels, extent, &path)?;
            frames.push((pixels, path));
        }
        let (off, on) = (&frames[0].0, &frames[1].0);
        let mut changed = 0u64;
        let mut sum = 0u64;
        for (a, b) in off.chunks_exact(4).zip(on.chunks_exact(4)) {
            let delta = (0..3).map(|c| a[c].abs_diff(b[c])).max().unwrap_or(0);
            changed += u64::from(delta > CHANGED);
            sum += u64::from(delta);
        }
        let total = (extent.0 as u64) * (extent.1 as u64);
        println!(
            "  {distance:7.0} | {:9.2}% | {:9.2} | {}",
            100.0 * changed as f64 / total as f64,
            sum as f64 / total as f64,
            frames[1].1.display()
        );
    }
    cvars::SHADOW_DISTANCE.set_float(shipped);
    cvars::SHADOW_REACH.set_bool(true);

    let report = renderer.shutdown();
    anyhow::ensure!(report.clean(), "unclean render: {report:?}");
    Ok(())
}

fn parse_extent(text: &str) -> Result<(u32, u32)> {
    let (w, h) = text
        .split_once('x')
        .ok_or_else(|| anyhow::anyhow!("--extent wants `<w>x<h>`, got {text:?}"))?;
    Ok((w.trim().parse()?, h.trim().parse()?))
}
