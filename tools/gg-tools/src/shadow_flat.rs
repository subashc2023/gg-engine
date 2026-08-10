//! `gg-tools shadow-flat` — the flat camera's contact band (§6 M20), measured.
//!
//! Demo 11's framing puts ~1 shadow texel on every screen pixel: an ortho
//! half-height of 4.5 m is 12.5 mm a pixel at this extent, and the cascade the
//! play plane lands in — the eye sits 14 m out of it — fits a slice sphere the
//! *view width* dominates, ~11 mm a texel however the range is turned. At that
//! ratio a corner residual the perspective demos never resolve is a visible dark
//! band where the player meets the ground: the front face is grazing-lit (the
//! sun's z component against the face normal), and at the concave corner the
//! kernel's taps cross onto the ground while the receiver-plane term
//! extrapolates the face's steep plane past it. This instrument is what settled
//! that: the band ignored every sane bias, and an analytic receiver plane
//! reproduced the derivative one's pixels exactly — so the defect was the
//! *two-sided* extrapolation, not the plane, and `pcf` is one-sided now. What
//! remains at the shipping fit is a one-row sub-texel stipple that halves with
//! the map edge.
//!
//! Two numbers, failing in opposite directions (the `shadow-bias` shape):
//!
//! - **band** — rows of false shadow up the player's front face at the contact.
//!   The face belongs to a convex caster with nothing between it and the sun,
//!   so its truth is *zero*: every dark row is an artifact, and no reference
//!   leg is needed to say so.
//! - **shade** — shadowed pixels of the one cast shadow this projection shows
//!   (a floater's, smeared down a backdrop close behind the play plane —
//!   receivers *in* the plane only ever catch a silhouette edge-on). Bias eats
//!   a shadow from its boundary, and this is the eating, counted.
//!
//! Swept two ways: the bias grid at the shipping fit, and the fit knobs at the
//! shipping bias — which is what says whether the fix is a knob, a denser map,
//! or an ortho-aware fit. Nothing here gates; defaults move only by review.

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

use crate::shadow_image::{DISAGREEMENT, luminance};

/// 16:9 like the desk the complaint came from — the aspect feeds the fitted
/// cascade's width and therefore its texel, so it is part of the measurement.
const EXTENT: (u32, u32) = (1280, 720);

/// Demo 11's camera, verbatim: `CAMERA_HALF_HEIGHT`, and an eye `CAMERA_LIFT`
/// above the feet and `CAMERA_BACK` out of the plane.
const HALF_HEIGHT: f64 = 4.5;
const EYE: [f64; 3] = [0.0, 1.6, 14.0];

/// Demo 11's sun, verbatim — the incidence against the front face is the trap.
const SUN: [f32; 3] = [-0.4, -1.0, -0.35];

/// The player's box, demo 11's `PLAYER_HALF_*`, resting on the ground at x = 0.
const PLAYER_HALF: [f64; 3] = [0.35, 0.45, 0.35];

/// The shipping defaults (`gg-render/src/cvars.rs`), restated so every leg sets
/// every knob and no table inherits the previous table's state.
const SHIP_SIZE: i64 = 2048;
const SHIP_DISTANCE: f64 = 80.0;
const SHIP_CASCADES: i64 = 4;
const SHIP_NORMAL: f64 = 0.5;
const SHIP_DEPTH: f64 = 0.75;

/// `(size, distance, cascades)` legs at the shipping bias. 16 m is the tightest
/// range that still contains a plane 14 m from the eye plus the backdrop.
const FIT_LEGS: &[(i64, f64, i64)] = &[
    (SHIP_SIZE, SHIP_DISTANCE, SHIP_CASCADES),
    (4096, SHIP_DISTANCE, SHIP_CASCADES),
    (SHIP_SIZE, 16.0, SHIP_CASCADES),
    (SHIP_SIZE, 16.0, 1),
    (4096, 16.0, 1),
];

/// The bias grid at the shipping fit, in shadow texels.
const NORMAL_BIASES: &[f64] = &[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0];
const DEPTH_BIASES: &[f64] = &[0.0, 0.75, 1.5, 2.5, 3.5];

pub fn run(args: &[String]) -> anyhow::Result<()> {
    if let Some(arg) = args.first() {
        anyhow::bail!("unknown flag {arg:?} — shadow-flat takes none");
    }
    let mut renderer = OffscreenRenderer::new(EXTENT)?;
    println!(
        "gg-tools shadow-flat: {}x{} on {}",
        EXTENT.0,
        EXTENT.1,
        renderer.device().chosen
    );
    println!(
        "  demo 11's framing: ortho half-height {HALF_HEIGHT} ({:.1} mm a pixel), eye {} up / \
         {} out, sun ({}, {}, {})",
        2.0 * HALF_HEIGHT / f64::from(EXTENT.1) * 1000.0,
        EYE[1],
        EYE[2],
        SUN[0],
        SUN[1],
        SUN[2],
    );
    println!("  band: false-shadow rows at the contact — the face is convex-lit, truth is 0");
    println!("  shade: surviving cast-shadow pixels on the backdrop — what bias eats");

    let world = scene()?;

    println!();
    println!("  fit at shipping bias (normal {SHIP_NORMAL} / depth {SHIP_DEPTH}):");
    println!("    size  dist casc | band px  shade px");
    println!("    ---------------+------------------");
    for &(size, distance, cascades) in FIT_LEGS {
        set(size, distance, cascades, SHIP_NORMAL, SHIP_DEPTH);
        let frame = render(&mut renderer, &world)?;
        let lum = luminance(&frame);
        let (band, shade) = (band_rows(&lum), shade_pixels(&lum));
        let ship = size == SHIP_SIZE && distance == SHIP_DISTANCE && cascades == SHIP_CASCADES;
        println!(
            "    {size:>5} {distance:>5} {cascades:>4} | {band:>7}  {shade:>8}{}",
            if ship { "   <- shipping" } else { "" }
        );
        if ship {
            anyhow::ensure!(
                shade > 500,
                "the backdrop shadow is missing at the shipping fit ({shade} px) — the scene \
                 no longer contains the cast shadow every shade number is counted against"
            );
            write_png(&frame, "shipping")?;
        }
    }

    println!();
    println!(
        "  bias at shipping fit ({SHIP_SIZE} over {SHIP_DISTANCE} m, {SHIP_CASCADES} cascades):"
    );
    println!("    normal depth | band px  shade px");
    println!("    -------------+------------------");
    let mut cells = Vec::new();
    for &normal in NORMAL_BIASES {
        for &depth in DEPTH_BIASES {
            set(SHIP_SIZE, SHIP_DISTANCE, SHIP_CASCADES, normal, depth);
            let frame = render(&mut renderer, &world)?;
            let lum = luminance(&frame);
            let (band, shade) = (band_rows(&lum), shade_pixels(&lum));
            println!("    {normal:>6} {depth:>5} | {band:>7}  {shade:>8}");
            cells.push((normal, depth, band, shade));
        }
    }

    // The floor rather than zero: the contact row itself mixes the two inks
    // whatever the shadows do, so "clean" is the least any cell achieves.
    let floor = cells.iter().map(|c| c.2).min().unwrap_or(0);
    let picked = cells
        .iter()
        .filter(|c| c.2 <= floor)
        .max_by_key(|c| (c.3, std::cmp::Reverse((c.0 + c.1).to_bits())));
    println!();
    if let Some(&(normal, depth, band, shade)) = picked {
        set(SHIP_SIZE, SHIP_DISTANCE, SHIP_CASCADES, normal, depth);
        write_png(&render(&mut renderer, &world)?, "picked")?;
        println!(
            "  floor: {floor} band row(s). Most shadow kept there: normal {normal} depth \
             {depth} (band {band}, shade {shade})"
        );
    }

    let report = renderer.shutdown();
    anyhow::ensure!(
        report.clean(),
        "unclean render: {} validation message(s), {} leak(s)",
        report.validation_messages,
        report.leaked_allocations.len(),
    );
    Ok(())
}

fn set(size: i64, distance: f64, cascades: i64, normal: f64, depth: f64) {
    cvars::SHADOW_SIZE.set_int(size);
    cvars::SHADOW_DISTANCE.set_float(distance);
    cvars::SHADOW_CASCADES.set_int(cascades);
    cvars::SHADOW_NORMAL_BIAS.set_float(normal);
    cvars::SHADOW_DEPTH_BIAS.set_float(depth);
}

/// Demo 11's contact, miniature: ground top at y = 0, the player resting on it,
/// a floater whose shadow the backdrop catches, and the backdrop itself close
/// behind the play plane — it has to be close, because the sun's z slope is
/// shallow and a wall 3 m back only receives shadows cast from 8 m up.
fn scene() -> anyhow::Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Light>()?;
    let mut put = |r: Renderable| -> anyhow::Result<()> {
        let e = world.spawn();
        world.insert(e, r)?;
        Ok(())
    };
    put(Renderable::boxed(
        sim::DVec3::new(0.0, -1.0, 0.0),
        sim::Vec3::new(12.0, 1.0, 0.4),
        0x003c_8a28,
    ))?;
    put(Renderable::boxed(
        sim::DVec3::new(0.0, PLAYER_HALF[1], 0.0),
        sim::Vec3::new(
            PLAYER_HALF[0] as f32,
            PLAYER_HALF[1] as f32,
            PLAYER_HALF[2] as f32,
        ),
        0x00f0_f4ff,
    ))?;
    put(Renderable::boxed(
        sim::DVec3::new(-1.5, 3.0, 0.0),
        sim::Vec3::new(0.9, 0.18, 0.4),
        0x00c8_5a3c,
    ))?;
    put(Renderable::boxed(
        sim::DVec3::new(0.0, 3.0, -1.05),
        sim::Vec3::new(12.0, 4.0, 0.6),
        0x009a_9488,
    ))?;
    let sun = world.spawn();
    world.insert(
        sun,
        Light::sun(sim::Vec3::new(SUN[0], SUN[1], SUN[2]), 0x00ff_f4e0, 3.5),
    )?;
    Ok(world)
}

/// One render at demo 11's camera, returned as raw RGBA for the PNGs.
fn render(renderer: &mut OffscreenRenderer, world: &World) -> anyhow::Result<Vec<u8>> {
    let view = View {
        ortho: HALF_HEIGHT as f32,
        ..View::default()
    };
    let eye = sim::DVec3::new(EYE[0], EYE[1], EYE[2]);
    let mut extracted = Extracted::default();
    extracted.transforms::<Renderable>(world, eye, view.frustum(EXTENT))?;
    extracted.append_lights(world)?;
    let frame = renderer.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?;
    anyhow::ensure!(
        frame.order.iter().any(|name| name.starts_with("shadow")),
        "no shadow pass ran — there would be nothing to measure"
    );
    // The mapping every metric depends on: +y is a *smaller* row. The ground is
    // green and the backdrop is not, so a flipped readback fails here by name
    // rather than as a table of zeros.
    let below = (row(-0.5) * EXTENT.0 as usize + col(0.0)) * 4;
    let (r, g) = (
        i32::from(frame.pixels[below]),
        i32::from(frame.pixels[below + 1]),
    );
    anyhow::ensure!(
        g > r,
        "the pixel below the contact is not green (r {r}, g {g}) — the row mapping is upside \
         down and every band count would walk the wrong way"
    );
    Ok(frame.pixels)
}

/// World x to pixel column: the ortho view is axis-aligned, so the projection
/// the metrics need is one linear map, not a matrix.
fn col(x: f64) -> usize {
    let half_w = HALF_HEIGHT * f64::from(EXTENT.0) / f64::from(EXTENT.1);
    (((x - EYE[0]) / (2.0 * half_w) + 0.5) * f64::from(EXTENT.0)) as usize
}

/// World y to pixel row, +y up meaning row 0 at the top.
fn row(y: f64) -> usize {
    ((0.5 - (y - EYE[1]) / (2.0 * HALF_HEIGHT)) * f64::from(EXTENT.1)) as usize
}

/// Worst run of dark rows walking up the player's front face from the contact.
///
/// Judged per column over the middle of the face against that column's own
/// mid-face median, so face shading (which varies with nothing here) is its own
/// baseline and the number needs no cross-leg reference.
fn band_rows(lum: &[i32]) -> usize {
    let w = EXTENT.0 as usize;
    let (top, bottom) = (row(2.0 * PLAYER_HALF[1]), row(0.0));
    let mut worst = 0;
    for c in col(-0.6 * PLAYER_HALF[0])..=col(0.6 * PLAYER_HALF[0]) {
        let mut interior: Vec<i32> = (row(1.4 * PLAYER_HALF[1])..row(0.6 * PLAYER_HALF[1]))
            .map(|r| lum[r * w + c])
            .collect();
        interior.sort_unstable();
        let median = interior[interior.len() / 2];
        let mut dark = 0;
        let mut r = bottom - 1;
        while r > top && lum[r * w + c] < median - DISAGREEMENT {
            dark += 1;
            r -= 1;
        }
        worst = worst.max(dark);
    }
    worst
}

/// Shadowed pixels in the backdrop window left of the player, judged against
/// the window's own median — lit backdrop, since the shadow covers under a
/// third of it. The window stops clear of the player's own pixels.
fn shade_pixels(lum: &[i32]) -> usize {
    let w = EXTENT.0 as usize;
    let mut values: Vec<i32> = (row(2.55)..row(0.15))
        .flat_map(|r| (col(-6.0)..col(-0.6)).map(move |c| lum[r * w + c]))
        .collect();
    values.sort_unstable();
    let median = values[values.len() / 2];
    values
        .iter()
        .filter(|&&l| l < median - 2 * DISAGREEMENT)
        .count()
}

fn write_png(pixels: &[u8], name: &str) -> anyhow::Result<()> {
    let path = crate::output_dir()?.join(format!("shadow-flat-{name}.png"));
    let file = std::fs::File::create(&path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), EXTENT.0, EXTENT.1);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(pixels)?;
    Ok(())
}
