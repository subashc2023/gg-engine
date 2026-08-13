//! `gg-tools furnace` — whether a surface gives back what it was given, and
//! whether the lobe it gives it back along points where the lobe is (§6 M33).
//!
//! Two questions that fail in different ways and are therefore measured against
//! different references. Both are the same milestone because they are the same
//! line of shader — one tap into a prefiltered chain, scaled by a split-sum —
//! and the milestone's whole claim is that neither half of that line was right.
//!
//! # The furnace
//!
//! Put a surface that absorbs nothing inside a sphere of uniform radiance `L`.
//! Whatever it does with the light, conservation says it radiates `L` back: the
//! surface is *invisible* against its own background, which is the classic white
//! furnace and the only test of this kind that needs no reference implementation
//! to argue with. A perfect white metal — `metallic 1`, colour white, so `f0` is
//! exactly 1 — is that surface, and the number below is what it actually
//! returned as a fraction of `L`. **1.000 closes.**
//!
//! What a single-scatter renderer returns instead is the *directional albedo* of
//! the GGX lobe, which is 1 at roughness 0 and falls to under a third at
//! roughness 1 head-on (§6 M34 measured it; M33 read 0.450 through a fit that
//! could not see the view): the microsurface shadows itself, the ray that would have bounced
//! a second time is dropped, and nothing puts it back. That is the defect, and
//! it is a curve rather than a constant, which is why it reads as "the rough end
//! of the chart is too dark" and never as a bug.
//!
//! **sky** is that measurement: a uniform environment and nothing else, so the
//! only code that runs is `ambient_light`. **sun** is the *direct* path's share
//! of the same question, and it is a ratio rather than an energy because it has
//! to be. A furnace integrates every direction at once; a light is one
//! direction, and no finite set of them integrates a lobe that is nearly a delta
//! at the smooth end — a thousand lights over the hemisphere still put about two
//! inside a roughness-0.2 lobe, so the sum's own error would dwarf the effect.
//! What a single light can be asked is what the correction *did to it*, and for
//! `f0 = 1` on a surface with no diffuse lobe that must come out at exactly
//! `1 / E_off`: the reciprocal of the loss the sky column measured. The two legs
//! share no arithmetic below the BRDF, so the agreement is a real cross-check
//! rather than a restatement, and the run fails on a disagreement instead of
//! printing one.
//!
//! # The lobe
//!
//! A furnace cannot see this half at all, and that is not a gap — it is the
//! reason the two corrections are separate knobs. Under a uniform environment
//! every direction returns the same radiance, so *where* the chain is sampled
//! cannot change what comes back. `r.lobe` is asserted to move no furnace digit,
//! which is the strongest statement available that it manufactures no energy.
//!
//! What it does change needs a real environment and a ground truth, and both are
//! on the CPU here. The chain is [Kar13]'s `n = v = r` prefilter: every texel is
//! the GGX lobe integrated as though the viewer stood on the normal. `ggc`
//! computes those texels, and since §6 M33 it computes them through
//! [`ggc::environment::reference_value`], which is the same integral with the
//! view direction left free — so the truth this table grades against is not a
//! model of the prefilter, it *is* the prefilter with the assumption removed.
//! Three numbers per cell:
//!
//! - **mirror** — what the chain holds along `reflect(-v, n)`: pre-M33.
//! - **dom-r** — [LdR14] §4.9.3's dominant direction, its lerp factor read on
//!   the perceptual roughness.
//! - **dom-a** — the same factor read on `alpha`. Both are printed because the
//!   published listing says "roughness" and its own text uses that word for both
//!   quantities, the two differ by a lot at the smooth end, and which one the
//!   shader should run is a question this can answer instead of cite.
//!
//! Reported as relative error against truth, so the columns are comparable
//! across a panorama whose radiance spans two orders of magnitude. None is
//! expected to reach zero: one tap into a chain that has thrown the view axis
//! away cannot, and the question this settles is only whether moving the tap
//! moves it toward the answer or away from it. The answer is *both*, in
//! different places — which is why the table is printed per cell and the
//! correction is a knob.

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable, Sky};
use gg_extract::Extracted;
use gg_math::{render, sim};
use gg_render::{OffscreenRenderer, View, cvars, split_sum};

/// Small: every measurement is a window at the middle of the frame, and the rest
/// of the pixels are only there to prove the middle is not an edge.
const EXTENT: (u32, u32) = (256, 256);

/// Half-width of the window averaged at the centre of the frame. Small enough to
/// sit well inside the wall at the most oblique angle swept, large enough that
/// the dither averages out — 169 pixels of decorrelated noise is where the
/// eight-bit output stops being the limit on what this can resolve.
const WINDOW: u32 = 6;

/// The environment's radiance, linear.
///
/// Low on purpose. The output is eight bits through a tonemap curve, and the
/// curve's shoulder compresses code values exactly where it is brightest — a
/// furnace run at 1.0 would be measuring the shoulder's slope rather than the
/// surface. At a quarter the whole `[0.6, 1.0]` band this instrument cares about
/// lands on the steep part, where a code value is worth about 0.3 % of `L`.
const RADIANCE: f32 = 0.25;

/// Steps of the radiance→code-value calibration, from nothing to a little over
/// [`RADIANCE`].
///
/// The transfer function is **measured, not modelled**: exposure, the Khronos
/// Neutral curve and the sRGB encode are three things this instrument would
/// otherwise hold a second copy of, and a second copy is a thing that goes stale
/// the day one of them is retuned. Rendering the background at a known radiance
/// and reading what came out asks the shipping post chain what it does.
const CALIBRATION: usize = 193;

/// Where the direct leg's dimmer half is aimed, as a fraction of [`RADIANCE`],
/// and the fraction above which a reading is treated as off the end of the
/// curve. The gap between them is the headroom the compensated leg needs.
///
/// 0.2 since §6 M34 and it was 0.3 under M33's fit, which is worth recording
/// because the reason is the milestone: the compensated leg is `1/E` brighter,
/// and correcting `E` at roughness 1 head-on from the fit's 0.450 to the true
/// 0.307 took that factor from 2.2 to 3.3. The instrument's own headroom was
/// sized against a number that was wrong.
const AIM: f32 = 0.2;
const CLIPPING: f32 = 0.95;

/// Roughness rows.
const ROUGHNESS: [f32; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

/// `n · v` columns, reached by turning the wall away from the camera. 1.0 is
/// head-on; 0.2 is 78 degrees off, which is where the single-scatter loss and
/// the lobe's stretch are both at their worst.
const VIEW_COSINE: [f32; 4] = [1.0, 0.7, 0.4, 0.2];

/// Where the camera stands, and how big the wall is. The wall has to cover the
/// centre window at every angle swept and the camera has to stay outside it once
/// it is turned, which is the whole of why these two numbers are what they are:
/// turned 78 degrees, a wall of half-width 20 reaches 19.6 m along the view axis.
const WALL: (f32, f32) = (20.0, 40.0);
const EYE_Z: f64 = 40.0;

/// Half the height of the orthographic frame, metres. Turned 78 degrees the wall
/// still projects 4 m either side of centre, which is five times the window.
const ORTHO_HALF_HEIGHT: f32 = 8.0;

/// The panorama the lobe table integrates. Demo 06's own, so what is graded is
/// an environment the engine ships rather than a test pattern chosen to flatter
/// the correction — it holds hard-edged windows eighteen times brighter than its
/// walls, which is exactly the content a lobe aimed a few degrees off will miss.
const PANORAMA: &str = "demos/06-lit/assets/sky/room.hdr";

/// Importance samples per cell of the lobe table, for **all three** of truth,
/// mirror and dominant.
///
/// Far above what `ggc` spends filling a texel, and the same number on every
/// column, because the difference being resolved is a few per cent and the pack's
/// own count leaves an estimator noise larger than that — the first run of this
/// table was measuring its own sampling, which reads as the correction helping in
/// one cell and hurting in the next by the same amount.
const REFERENCE: u32 = 8192;

pub fn run(args: &[String]) -> anyhow::Result<()> {
    if let Some(arg) = args.first() {
        anyhow::bail!("unknown flag {arg:?} — furnace takes none");
    }
    println!("gg-tools furnace — a white metal inside a uniform sky (§6 M33)");
    println!();
    println!(
        "  A surface that absorbs nothing radiates back what it received. E is what it did\n  \
         return, over the environment's own radiance. 1.000 closes; below is energy the\n  \
         single-scatter lobe dropped, above is energy the correction invented."
    );

    sky_and_sun()?;
    lobe()?;
    Ok(())
}

/// The rendered half: what the furnace returns, and whether the *direct* lobe
/// was given back the same fraction the furnace says it lost.
///
/// The second column pair is the only absolute statement available about a
/// direct light, and it is a ratio rather than an energy on purpose. A furnace
/// integrates over every direction; a light is one direction, and no finite set
/// of them can integrate a lobe that is nearly a delta at the smooth end — at
/// roughness 0.2 a hemisphere of a thousand lights still has about two inside
/// the lobe. So the direct path is not asked what it returns in total. It is
/// asked what the correction *did to it*, which for `f0 = 1` must be exactly
/// `1 / E_off` — the reciprocal of the loss the sky column measured, on a
/// surface with no diffuse lobe to dilute it. The two legs share no code below
/// the BRDF, so the agreement is a real cross-check and not a restatement.
fn sky_and_sun() -> anyhow::Result<()> {
    // Dither *on* and averaged over the window, which buys the precision the
    // eight-bit output does not have: the noise is decorrelated from the signal
    // by construction (§6 M22), so the mean of a few hundred dithered pixels
    // resolves well inside a code value where a single undithered one cannot.
    // The calibration is read the same way, so the two travel one curve.
    let dither = cvars::DITHER.float();
    cvars::DITHER.set_float(1.0);

    let mut renderer = OffscreenRenderer::new(EXTENT)?;
    let curve = calibrate(&mut renderer)?;

    println!();
    println!("  rough | sky: E off   E on  | sun: on/off   1/E off");
    println!("  ------+--------------------+----------------------");
    for roughness in ROUGHNESS {
        let sky = wall(roughness, 1.0, None)?;
        cvars::MULTISCATTER.set_bool(false);
        let off = energy(&mut renderer, &sky, &curve)?;
        cvars::MULTISCATTER.set_bool(true);
        let on = energy(&mut renderer, &sky, &curve)?;

        let intensity = aim(&mut renderer, &curve, roughness)?;
        let sun = wall(roughness, 1.0, Some(intensity))?;
        cvars::MULTISCATTER.set_bool(false);
        let direct_off = radiance(&mut renderer, &sun, &curve)?;
        cvars::MULTISCATTER.set_bool(true);
        let direct_on = radiance(&mut renderer, &sun, &curve)?;
        anyhow::ensure!(
            direct_on < CLIPPING * RADIANCE && direct_off > 1.0e-3 * RADIANCE,
            "the direct leg read {direct_off:.4} and {direct_on:.4} against an environment of \
             {RADIANCE} — one of them is off the end of the calibration and the ratio below \
             would be the curve's own clipping rather than the shading's"
        );
        let ratio = direct_on / direct_off.max(1e-6);
        let predicted = 1.0 / off.max(1e-6);
        anyhow::ensure!(
            (ratio - predicted).abs() < 0.05 * predicted,
            "at roughness {roughness} the direct lobe gained {ratio:.3}x where the furnace \
             says it lost 1/{off:.3} = {predicted:.3}x — the two paths are compensating by \
             different amounts, so one of them is not using the albedo the other measured"
        );
        println!("  {roughness:>5.2} | {off:>9.3} {on:>7.3}  | {ratio:>9.3} {predicted:>9.3}");
    }

    view_dependence(&mut renderer, &curve)?;

    let report = renderer.shutdown();
    anyhow::ensure!(
        report.clean(),
        "unclean render: {} validation message(s), {} leak(s)",
        report.validation_messages,
        report.leaked_allocations.len(),
    );
    cvars::DITHER.set_float(dither);
    Ok(())
}

/// Two controls, both of which are expected to find nothing, and both of which
/// would be the milestone being silently wrong if they found something.
///
/// **The view axis is flat, and that is the fit's doing rather than the rig's.**
/// [Laz13]'s approximation returns `(-1.04 a + z, 1.04 a + w)` where only `a`
/// carries `n·v` — so `scale + bias`, which is the directional albedo at
/// `f0 = 1` and the whole of what a white furnace measures, is `z + w` and the
/// view term has cancelled exactly. The real GGX albedo does fall toward the
/// silhouette; this fit cannot say so, and neither therefore can the
/// compensation built on it. Swept anyway, because "the correction is
/// view-independent" is a claim worth holding a measurement against rather than
/// an algebra nobody re-derives.
///
/// **`r.lobe` may not move a furnace digit.** Every direction in a uniform
/// environment returns the same radiance, so aiming the lobe elsewhere must
/// change nothing here. A digit moving would mean the correction had become an
/// energy term, which is the one way it could be wrong without looking wrong.
fn view_dependence(renderer: &mut OffscreenRenderer, curve: &[(f32, f32)]) -> anyhow::Result<()> {
    println!();
    println!(
        "single-scatter E by view angle — rendered / CPU reference / [Laz13] fit (§6 M34)\n  \
         the third estimator: the first two are arithmetic, this one is a picture."
    );
    print!("  r\\n·v |");
    for cosine in VIEW_COSINE {
        print!("{cosine:>22.1}");
    }
    println!();
    let (mut worst, mut fit_worst, mut lobe_spread) = (0.0f32, 0.0f32, 0.0f32);
    for roughness in ROUGHNESS {
        print!("  {roughness:>4.1} |");
        for cosine in VIEW_COSINE {
            let world = wall(roughness, cosine, None)?;
            cvars::MULTISCATTER.set_bool(false);
            let e = energy(renderer, &world, curve)?;
            // `integrate` and not `sample`: what is being graded is the whole
            // path — the table, the shader's bilinear read of it, the BRDF, the
            // tonemap and the calibration — against the integral, not the
            // table against itself.
            let truth = {
                let (a, b) = split_sum::integrate(roughness, cosine, 65_536);
                a + b
            };
            let fit = {
                let (a, b) = split_sum::fit(roughness, cosine);
                a + b
            };
            worst = worst.max((e - truth).abs());
            fit_worst = fit_worst.max((fit - truth).abs());
            print!(" {e:>6.3}/{truth:.3}/{fit:.3}");

            cvars::MULTISCATTER.set_bool(true);
            cvars::LOBE.set_bool(true);
            let aimed = energy(renderer, &world, curve)?;
            cvars::LOBE.set_bool(false);
            let mirrored = energy(renderer, &world, curve)?;
            cvars::LOBE.set_bool(true);
            lobe_spread = lobe_spread.max((aimed - mirrored).abs());
        }
        println!();
    }
    anyhow::ensure!(
        lobe_spread < 5.0e-3,
        "r.lobe moved the furnace by {lobe_spread:.4} — aiming the lobe elsewhere changed how \
         much came back, so it has stopped being only a direction"
    );
    println!(
        "\n  rendered vs reference: worst {worst:.4}; the fit it replaced: {fit_worst:.4}\n  \
         control | r.lobe moves E by {lobe_spread:.4} — a direction, not an energy."
    );
    Ok(())
}

/// Radiance → output code value, as the shipping post chain actually maps it.
///
/// Read off the **background**, which is the skybox drawing the environment's
/// own radiance with no surface in front of it — so the calibration and the
/// measurement travel the identical curve and nothing about that curve has to be
/// known here. Monotone by construction; asserted, because an inversion of a
/// table that folds back on itself silently returns the wrong branch.
fn calibrate(renderer: &mut OffscreenRenderer) -> anyhow::Result<Vec<(f32, f32)>> {
    let mut curve = Vec::with_capacity(CALIBRATION);
    for step in 0..CALIBRATION {
        let radiance = RADIANCE * 1.2 * step as f32 / (CALIBRATION - 1) as f32;
        let world = empty_sky(radiance)?;
        let pixels = frame(renderer, &world)?;
        // The same centre window the measurement reads, over a frame with
        // nothing in it: the calibration has to average the dither exactly as
        // the measurement does, or the two are on different curves.
        let (code, _) = window(&pixels);
        if let Some(&(_, last)) = curve.last() {
            anyhow::ensure!(
                code >= last,
                "the output curve is not monotone at {radiance:.4} ({last} then {code}) — \
                 nothing below can be inverted"
            );
        }
        curve.push((radiance, code));
    }
    Ok(curve)
}

/// The intensity that puts this roughness's specular highlight in the middle of
/// the measurable range.
///
/// A head-on mirror concentrates a whole light into a lobe a thousandth of a
/// steradian across and comes out four orders of magnitude brighter than the
/// same light on a rough surface — one intensity cannot serve the column, and an
/// intensity that clips reads the *curve's* ceiling identically on both legs and
/// reports a ratio of exactly 1.000 with nothing wrong anywhere.
///
/// Chosen by looking rather than by a formula, which is the point: a closed form
/// for where the highlight lands would be a second copy of the BRDF living in
/// the instrument that measures it. Shading is linear in a light's intensity, so
/// one render and one multiplication land it exactly; the loop is there to walk
/// in from a first guess that may be off by orders of magnitude.
///
/// The target is low in the range because the *other* leg is up to 3.3x brighter
/// and must fit above it (§6 M34; it was 2.2x under the fit).
fn aim(
    renderer: &mut OffscreenRenderer,
    curve: &[(f32, f32)],
    roughness: f32,
) -> anyhow::Result<f32> {
    let alpha = (roughness * roughness).max(1.0e-3);
    let mut intensity = RADIANCE * alpha * alpha;
    cvars::MULTISCATTER.set_bool(false);
    for _ in 0..24 {
        let world = wall(roughness, 1.0, Some(intensity))?;
        let got = radiance(renderer, &world, curve)?;
        if got >= CLIPPING * RADIANCE {
            intensity *= 0.1;
            continue;
        }
        if got <= 0.02 * RADIANCE {
            intensity *= 10.0;
            continue;
        }
        return Ok(intensity * AIM * RADIANCE / got);
    }
    anyhow::bail!("no intensity put roughness {roughness}'s highlight inside the curve")
}

/// What the wall returned, over [`RADIANCE`] — the furnace's own number.
fn energy(
    renderer: &mut OffscreenRenderer,
    world: &World,
    curve: &[(f32, f32)],
) -> anyhow::Result<f32> {
    Ok(radiance(renderer, world, curve)? / RADIANCE)
}

/// The absolute radiance the centre window came back at.
fn radiance(
    renderer: &mut OffscreenRenderer,
    world: &World,
    curve: &[(f32, f32)],
) -> anyhow::Result<f32> {
    let pixels = frame(renderer, world)?;
    let (code, spread) = window(&pixels);
    // The window has to be one surface. At the most oblique angle the wall is
    // twenty-odd pixels wide, and a window that had slipped off it would average
    // wall and sky into a plausible-looking number. The bound is the dither's
    // own span and not zero — a flat surface under triangular noise of one code
    // value legitimately covers a few (§6 M22).
    anyhow::ensure!(
        spread <= 6.0,
        "the centre window spans {spread} code values — more than the dither can account for, \
         so it is not all one surface and the framing has stopped measuring one"
    );
    Ok(invert(curve, code))
}

/// A code value back to the radiance that produced it, by linear interpolation
/// into the measured curve.
fn invert(curve: &[(f32, f32)], code: f32) -> f32 {
    let at = curve.partition_point(|&(_, c)| c < code);
    if at == 0 {
        return curve[0].0;
    }
    if at >= curve.len() {
        return curve[curve.len() - 1].0;
    }
    let (r0, c0) = curve[at - 1];
    let (r1, c1) = curve[at];
    // Equal code values mean a flat run of the curve: the honest answer is the
    // middle of the run, not either end of it.
    if (c1 - c0).abs() < f32::EPSILON {
        return (r0 + r1) * 0.5;
    }
    r0 + (r1 - r0) * (code - c0) / (c1 - c0)
}

/// The mean and the spread of the green channel over the centre window.
fn window(pixels: &[u8]) -> (f32, f32) {
    let (mut total, mut count) = (0.0f32, 0u32);
    let (mut low, mut high) = (255u8, 0u8);
    let centre = (EXTENT.0 / 2, EXTENT.1 / 2);
    for y in centre.1 - WINDOW..=centre.1 + WINDOW {
        for x in centre.0 - WINDOW..=centre.0 + WINDOW {
            let green = pixels[((y * EXTENT.0 + x) * 4 + 1) as usize];
            total += f32::from(green);
            count += 1;
            low = low.min(green);
            high = high.max(green);
        }
    }
    (total / count as f32, f32::from(high) - f32::from(low))
}

/// One frame of whatever world it is handed, through an **orthographic** eye.
///
/// Parallel rays are what make this a measurement rather than a picture: under
/// perspective the view vector turns across the frame, so `n·v` is only exactly
/// the wall's angle at the very centre pixel and a mirror lit head-on puts a
/// highlight a few pixels wide there — the window averaged one bright spot and
/// a lot of black, and reported the mean of the two as a surface. Under ortho
/// the eye direction is a constant (§6 M20 put row 2 of the projection there),
/// so the whole face shades identically and the window is reading one number
/// many times instead of a gradient once.
fn frame(renderer: &mut OffscreenRenderer, world: &World) -> anyhow::Result<Vec<u8>> {
    let view = View {
        ortho: ORTHO_HALF_HEIGHT,
        ..View::default()
    };
    let eye = sim::DVec3::new(0.0, 0.0, EYE_Z);
    let mut extracted = Extracted::default();
    extracted.clear(eye, view.frustum(EXTENT));
    extracted.append::<Renderable>(world)?;
    extracted.append_lights(world)?;
    Ok(renderer
        .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?
        .pixels)
}

/// A uniform environment and nothing in it — the calibration's own scene.
fn empty_sky(radiance: f32) -> anyhow::Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Sky>()?;
    world.register::<Light>()?;
    let sky = world.spawn();
    world.insert(sky, uniform(radiance))?;
    Ok(world)
}

/// White in all three directions, which is what makes it a furnace: the gradient
/// sky projects to spherical harmonics exactly, and a constant projects to its
/// DC band alone — so every direction returns `radiance` with no interpolation
/// anywhere and no chain to resolve.
fn uniform(radiance: f32) -> Sky {
    Sky {
        zenith: 0x00ff_ffff,
        horizon: 0x00ff_ffff,
        ground: 0x00ff_ffff,
        intensity: radiance,
        ..Sky::daylight(radiance)
    }
}

/// One wall of perfect white metal, turned `cosine` away from the camera, inside
/// a uniform sky — or, under `sun`, lit by a single directional light instead.
fn wall(roughness: f32, cosine: f32, sun: Option<f32>) -> anyhow::Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Sky>()?;
    world.register::<Light>()?;

    // The sky is present either way. In the sun leg it carries no radiance at
    // all, which keeps the *pass list* identical between the two legs — a
    // skybox draw appearing in one and not the other would be a difference in
    // the picture that is not a difference in the shading.
    let sky = world.spawn();
    world.insert(sky, uniform(if sun.is_some() { 0.0 } else { RADIANCE }))?;

    let angle = sim::acos(f64::from(cosine.clamp(-1.0, 1.0)));
    let (sin, cos) = sim::sin_cos(angle);
    let normal = render::Vec3::new(sin as f32, 0.0, cos as f32);
    let mut surface = Renderable::boxed(
        sim::DVec3::ZERO,
        sim::Vec3::new(WALL.0, WALL.1, 0.5),
        0x00ff_ffff,
    )
    // `smoothness`, which is the boundary's spelling of the same axis.
    .surfaced(1.0 - roughness, 1.0);
    surface.rotation = sim::DQuat::from_axis_angle(sim::DVec3::Y, angle);
    let entity = world.spawn();
    world.insert(entity, surface)?;

    if let Some(intensity) = sun {
        let lamp = world.spawn();
        world.insert(
            lamp,
            // Straight down the wall's own normal, travelling toward it. Head-on
            // rather than at an angle because the ratio this leg reports must not
            // have a shadow term in it — a grazing light on a flat wall is where
            // acne lives, and while it would scale both legs alike and cancel,
            // "it cancels" is a worse property than "it is not there".
            Light::sun(
                sim::Vec3::new(-normal.x, -normal.y, -normal.z),
                0x00ff_ffff,
                intensity,
            ),
        )?;
    }
    Ok(world)
}

/// The lobe table: one chain tap against the integral the chain approximates.
///
/// Entirely on the CPU and with no device at all, because what is being graded
/// is *where the tap points* and not how a texture stores it. Octahedral
/// projection, mip selection and BC6H are the chain's own fidelity and would
/// only add their error to both columns equally.
fn lobe() -> anyhow::Result<()> {
    let path = std::path::Path::new(PANORAMA);
    anyhow::ensure!(
        path.is_file(),
        "{} is not there — `gg-tools panorama` writes it (§6 M27)",
        path.display()
    );
    let image = image::open(path)?.into_rgb32f();
    let (width, height) = (image.width(), image.height());
    let source: Vec<[f32; 3]> = image.pixels().map(|p| p.0).collect();
    let working = ggc::environment::working_copy(&source, width, height)?;

    println!();
    println!(
        "  lobe — one tap into the n = v = r chain, against the same integral with the view\n  \
         it actually has. Relative error, per cent of truth; demo 06's own panorama."
    );
    println!();
    println!(
        "  rough | {:^20} | {:^20} | {:^20}",
        "n·v 0.70", "n·v 0.40", "n·v 0.20"
    );
    println!(
        "        | {:^20} | {:^20} | {:^20}",
        "mirror  dom-r  dom-a", "mirror  dom-r  dom-a", "mirror  dom-r  dom-a"
    );
    println!("  ------+----------------------+----------------------+----------------------");

    // The normal is fixed and the view is turned, which is the same geometry the
    // rendered legs have and the opposite of how it reads: turning the wall away
    // from a fixed camera *is* moving the eye around a fixed normal.
    //
    // `n·v 1.00` is not a row because it cannot be one: at head-on the view *is*
    // the normal, which is the case the chain was integrated for, and all three
    // columns are zero by construction rather than by measurement.
    let normal = render::Vec3::Z;
    let mut totals = [0.0f32; 3];
    let mut cells = 0u32;
    for roughness in ROUGHNESS {
        let mut row = String::new();
        for cosine in VIEW_COSINE.iter().skip(1) {
            let (sin, cos) = sim::sin_cos(sim::acos(cosine.clamp(-1.0, 1.0)));
            let to_eye = render::Vec3::new(sin, 0.0, cos);
            let truth =
                ggc::environment::reference_value(&working, normal, to_eye, roughness, REFERENCE);
            let reflected = reflect(to_eye, normal);
            let aims = [
                reflected,
                dominant_direction(normal, reflected, roughness),
                // The same fit read on the *other* roughness axis — see
                // `dominant_direction`. Both are printed because the citation is
                // ambiguous about which one it means and the difference is not
                // small; nothing here has to decide from prose.
                dominant_direction(normal, reflected, roughness * roughness),
            ];
            for (slot, aim) in aims.iter().enumerate() {
                let got = ggc::environment::chain_value(&working, *aim, roughness, REFERENCE);
                let e = error(truth, got);
                totals[slot] += e;
                row.push_str(&format!("{e:>7.1}"));
            }
            row.push_str(" |");
            cells += 1;
        }
        println!("  {roughness:>5.2} |{}", row.trim_end_matches('|'));
    }
    println!();
    println!(
        "  mean | mirror {:.1} %   dom-r {:.1} %   dom-a {:.1} %",
        totals[0] / cells.max(1) as f32,
        totals[1] / cells.max(1) as f32,
        totals[2] / cells.max(1) as f32
    );
    Ok(())
}

/// `reflect(-to_eye, normal)`, spelled the way the shader spells it.
fn reflect(to_eye: render::Vec3, normal: render::Vec3) -> render::Vec3 {
    normal * (2.0 * normal.dot(to_eye)) - to_eye
}

/// [LdR14] §4.9.3, and the same two lines `pbr.slang` runs. Written twice on
/// purpose: this is the reference's side of the question, and a reference that
/// imported the implementation could not disagree with it.
///
/// `axis` is *which* roughness — the perceptual one an artist authors, or the
/// `alpha` it squares to. The published listing says "roughness" and the
/// surrounding text uses both words for both things; the two differ by a lot at
/// the smooth end, where the perceptual reading pulls a roughness-0.2 surface
/// twelve per cent off its mirror direction and the alpha reading pulls it two.
/// Which one it should be is a question with a measurement rather than a
/// citation behind it, which is why the caller prints both.
fn dominant_direction(normal: render::Vec3, reflected: render::Vec3, axis: f32) -> render::Vec3 {
    let roughness = axis;
    let smoothness = (1.0 - roughness).clamp(0.0, 1.0);
    let factor = smoothness * (sim::sqrt(smoothness) + roughness);
    (normal + (reflected - normal) * factor).normalize()
}

/// Relative error in per cent, over the mean of the three channels — a
/// panorama's radiance spans two orders of magnitude and an absolute error would
/// report the bright wall and nothing else.
fn error(truth: render::Vec3, got: render::Vec3) -> f32 {
    let denominator = (truth.x + truth.y + truth.z).max(1e-6);
    100.0 * ((got.x - truth.x).abs() + (got.y - truth.y).abs() + (got.z - truth.z).abs())
        / denominator
}
