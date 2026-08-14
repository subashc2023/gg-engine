//! `gg-tools map` — whether demo 13's map is legible, in the two numbers that
//! fail in opposite directions (§6 M38).
//!
//! The map is a **schematic** drawn with the renderer's only vocabulary: lit
//! geometry. Every symbol on it — the star, two planet discs, 128 ring dots and
//! 64 trace dots — is a diffuse ball shaded by one point light, and the whole
//! point of the picture is that its content spans 2.3e11 metres. Those two facts
//! do not sit together: an inverse-square light placed at the star delivers 16x
//! more to a ring at 1 AU than to one at 2.3, so a lux that floors the outer
//! ring blows the inner one to white and a symbol blown to white has lost the
//! only thing it was carrying, which is its colour.
//!
//! So the failures are **blown** — all three channels at the ceiling, a green
//! trace that reads white — and **crushed** — the peak channel under the noise
//! floor, a symbol that is not there. `STAR_LUX` was read off a rendered picture
//! by eye and graded a plateau; the first table below is what that reading
//! should have been, and it says there is no plateau to find.
//!
//! Where a symbol landed is asked of the matrix the frame was drawn with, never
//! rebuilt beside it (`shadow-sweep`'s rule, §6 M22).

use anyhow::Result;
use demo_13_orbit as orbit;
use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::{render, sim};
use gg_render::{OffscreenRenderer, View};

/// The golden's extent, for the golden's reason: a `TRACE_DOT` is 0.3 % of the
/// view height, so a smaller frame measures aliasing rather than shading.
const EXTENT: (u32, u32) = (1280, 720);

/// Both ends of the zoom key and three stations between them, in decades.
const ZOOMS: [f64; 5] = [5.5, 6.8, 8.1, 9.4, 10.75];

/// How far a symbol's *hue* may drift before it has stopped carrying one.
///
/// Measured on the colour normalised by its own peak channel, so brightness is
/// divided out and what is left is the thing a map symbol means: `SHIP_TRACE_INK`
/// is (96, 255, 192) and reads (0.38, 1.00, 0.75). A dot rendered white reads
/// (1.00, 1.00, 1.00) and scores 0.62 — this threshold calls that lost, and it
/// is deliberately generous, since a quarter of full scale off is already a
/// green a player would not name as one.
const HUE_LOST: f32 = 0.25;

/// Peak channel below this and the symbol is not visible against the background
/// it is drawn on — the map's ground is `0x0a0a1e`-ish, not black.
const DIM: u8 = 40;

/// Lux values swept for the star lamp. Spans two decades either side of the
/// shipped 60, which is wide enough that "no value works" is a finding rather
/// than a search that stopped early.
const SWEEP: [f64; 7] = [5.0, 20.0, 60.0, 200.0, 600.0, 2_000.0, 6_000.0];

/// The epoch the `orbit` golden frames — the transfer handed to the star.
const TRANSFER_EPOCH: u64 = 4_925_342;

pub fn run(_args: &[String]) -> Result<()> {
    let mut renderer = OffscreenRenderer::new(EXTENT)?;
    check_projection(&mut renderer)?;
    shipped(&mut renderer)?;
    sweep(&mut renderer)?;
    headlamp(&mut renderer)?;
    Ok(())
}

// ---- the scene ----------------------------------------------------------

/// One drawn thing, in map metres: what the game would put in the world, plus
/// the point this instrument probes it at and the colour it was authored with.
struct Symbol {
    class: &'static str,
    position: sim::DVec3,
    color: u32,
    draw: Renderable,
}

/// A map symbol that is a body — matte, for `trace_segment`'s reason.
fn ball(class: &'static str, position: sim::DVec3, radius: f32, color: u32) -> Symbol {
    Symbol {
        class,
        position,
        color,
        draw: Renderable::ball(position, radius, color).surfaced(0.0, 0.0),
    }
}

/// Which situation the map is in. The user's complaint arrives in the first one
/// and the golden frames the second, and they light differently: parked, every
/// symbol that matters is a million times nearer the eye than the star is.
#[derive(Clone, Copy)]
enum Where {
    /// Tick 0, ship on the authored parking orbit about Verge.
    Parking,
    /// The golden's epoch, ship on the flown transfer about the star.
    Transfer,
}

impl Where {
    const fn name(self) -> &'static str {
        match self {
            Self::Parking => "parking",
            Self::Transfer => "transfer",
        }
    }

    /// Seconds since the world opened.
    fn seconds(self) -> f64 {
        let ticks = match self {
            Self::Parking => 0,
            Self::Transfer => TRANSFER_EPOCH,
        };
        ticks as f64 / f64::from(gg_core::DEFAULT_TICK_HZ)
    }

    /// The ship's conic and the body it is stated about.
    fn ship(self) -> (sim::Orbit, u32) {
        match self {
            Self::Parking => (
                sim::Orbit {
                    semi_major: orbit::PARKING_RADIUS,
                    eccentricity: 0.0,
                    inclination: 0.0,
                    ascending_node: 0.0,
                    argument_of_periapsis: 0.0,
                    mean_anomaly: 0.0,
                    mu: orbit::VERGE.mu,
                },
                orbit::VERGE.index,
            ),
            // The `orbit` golden's own authored elements (§6 M38 item 15).
            Self::Transfer => (
                sim::Orbit {
                    semi_major: 1.882_96e11,
                    eccentricity: 0.234_62,
                    inclination: 0.0,
                    ascending_node: 0.0,
                    argument_of_periapsis: 1.796,
                    mean_anomaly: 0.0,
                    mu: orbit::MU_STAR,
                },
                0,
            ),
        }
    }
}

/// Every symbol `present` would write, in map metres, centred on the ship.
///
/// Dealt from the demo's own tables and stepped by the demo's own `sample`, for
/// the `orbit` golden's reason: a second copy of the stepping is a second thing
/// to keep in step.
fn symbols(at: Where, zoom: f64) -> Vec<Symbol> {
    let seconds = at.seconds();
    let scale = sim::powf(10.0, -zoom);
    let stations: Vec<(&orbit::Planet, sim::DVec3)> = [&orbit::VERGE, &orbit::OCHRE]
        .into_iter()
        .map(|planet| (planet, planet.orbit().state_at(seconds).0))
        .collect();
    let (conic, about) = at.ship();
    let origin = stations
        .iter()
        .find(|(planet, _)| planet.index == about)
        .map_or(sim::DVec3::ZERO, |(_, position)| *position);
    let ship = origin + conic.state_at(0.0).0;
    let map = |absolute: sim::DVec3| (absolute - ship) * scale;

    let mut out = vec![ball(
        "star",
        map(sim::DVec3::ZERO),
        ((orbit::STAR_RADIUS * scale) as f32).max(orbit::MIN_DOT),
        orbit::STAR_GLOW,
    )];
    for (planet, position) in &stations {
        out.push(ball(
            if planet.index == orbit::VERGE.index {
                "Verge"
            } else {
                "Ochre"
            },
            map(*position),
            ((planet.radius * scale) as f32).max(orbit::MIN_DOT),
            planet.color,
        ));
        let class = if planet.index == orbit::VERGE.index {
            "Verge ring"
        } else {
            "Ochre ring"
        };
        let points = orbit::sample(planet.orbit(), sim::DVec3::ZERO, &map);
        for slot in 0..points.len() {
            out.push(segment(class, &points, slot, orbit::dim(planet.color)));
        }
    }
    out.push(ball(
        "ship",
        sim::DVec3::ZERO,
        orbit::SHIP_DOT,
        orbit::SHIP_INK,
    ));
    let points = orbit::sample(conic, origin, &map);
    for slot in 0..points.len() {
        out.push(segment("ship trace", &points, slot, orbit::SHIP_TRACE_INK));
    }
    out
}

/// One ribbon segment as this instrument probes it: measured at its *midpoint*,
/// which is where the game draws its centre.
fn segment(class: &'static str, points: &[sim::DVec3], slot: usize, color: u32) -> Symbol {
    let from = points[slot];
    let to = points[(slot + 1) % points.len()];
    Symbol {
        class,
        position: (from + to) * 0.5,
        color,
        draw: orbit::trace_segment(from, to, color),
    }
}

/// Where the light is and how much of it there is.
#[derive(Clone, Copy)]
enum Lamp {
    /// Demo 13's shipped model: at the star, this many lux at the map's centre.
    Star(f64),
    /// At the eye. A map has no lighting direction to report, so a headlamp
    /// costs the picture nothing and every symbol is lit head-on at the same
    /// distance whatever the zoom.
    Eye(f64),
}

impl Lamp {
    /// The light, in map space. The star arm restates `star_light` because the
    /// lux is the swept knob — the shipped function pins it to `STAR_LUX`, which
    /// is the constant this instrument exists to grade.
    fn light(self, star: sim::DVec3) -> Light {
        match self {
            Self::Star(lux) => {
                let far = star.length().max(1.0);
                Light::point(
                    star,
                    orbit::STAR_INK,
                    (lux * far * far) as f32,
                    (far * 2.0) as f32,
                )
            }
            Self::Eye(lux) => {
                let eye = orbit::eye_position();
                let far = orbit::EYE_RANGE;
                Light::point(
                    eye,
                    orbit::STAR_INK,
                    (lux * far * far) as f32,
                    (far * 4.0) as f32,
                )
            }
        }
    }
}

// ---- one frame, measured ------------------------------------------------

/// What one render says about the symbols in it.
///
/// The two counts fail in opposite directions as the lamp is turned up: a symbol
/// too dark to see, against one bright enough that its channels clip and it
/// reads as the *light's* colour rather than its own. There is a plateau between
/// them only if the frame's symbols sit within one exposure of each other, which
/// is the assumption a map spanning 2.3e11 metres breaks.
#[derive(Default)]
struct Reading {
    /// Symbols whose projected pixel is inside the frame.
    seen: u32,
    /// Of those, how many are under [`DIM`] and how many are past [`HUE_LOST`].
    dim: u32,
    white: u32,
    /// Summed hue error over `seen`, and the worst of them.
    error: f32,
    worst: f32,
    /// The same sum taken at each symbol's centre texel only.
    center: f32,
    /// A ship-trace dot at the median brightness — the symbol the complaint
    /// names, sampled where it is typical rather than where it is brightest.
    trace: [u8; 3],
}

/// A colour normalised by its own peak channel: what is left of it once
/// brightness is divided out.
fn hue(rgb: [f32; 3]) -> [f32; 3] {
    let peak = rgb[0].max(rgb[1]).max(rgb[2]).max(1e-6);
    [rgb[0] / peak, rgb[1] / peak, rgb[2] / peak]
}

/// How far a rendered symbol's hue is from the one it was authored with.
fn hue_error(rendered: [u8; 3], authored: u32) -> f32 {
    let want = hue([
        ((authored >> 16) & 0xff) as f32,
        ((authored >> 8) & 0xff) as f32,
        (authored & 0xff) as f32,
    ]);
    let got = hue([
        f32::from(rendered[0]),
        f32::from(rendered[1]),
        f32::from(rendered[2]),
    ]);
    (0..3).fold(0.0_f32, |worst, c| worst.max((want[c] - got[c]).abs()))
}

fn measure(renderer: &mut OffscreenRenderer, at: Where, zoom: f64, lamp: Lamp) -> Result<Reading> {
    let drawn = symbols(at, zoom);
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Light>()?;
    for symbol in &drawn {
        let entity = world.spawn();
        world.insert(entity, symbol.draw)?;
    }
    let star = drawn
        .first()
        .map_or(sim::DVec3::ZERO, |symbol| symbol.position);
    let lit = world.spawn();
    world.insert(lit, lamp.light(star))?;

    let view = View {
        pitch: orbit::EYE_PITCH,
        ..View::default()
    };
    let mut extracted = Extracted::default();
    extracted.clear(orbit::eye_position(), view.frustum(EXTENT));
    extracted.append::<Renderable>(&world)?;
    extracted.append_lights(&world)?;
    let pixels = renderer
        .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?
        .pixels;

    let matrix = view.view_projection(EXTENT);
    let mut out = Reading::default();
    let mut traces: Vec<[u8; 3]> = Vec::new();
    for symbol in &drawn {
        let Some(rgb) = probe(&pixels, matrix, symbol.position, 2) else {
            continue;
        };
        out.seen += 1;
        let peak = rgb[0].max(rgb[1]).max(rgb[2]);
        let error = hue_error(rgb, symbol.color);
        out.error += error;
        out.worst = out.worst.max(error);
        // The same symbol read at its centre texel alone. A lit sphere is
        // brightest where it faces the light and at its **rim**, where Fresnel
        // takes a dielectric's 4 % specular to 1 — and a three-pixel ball is
        // nearly all rim. If the centre keeps its hue while the window max does
        // not, the map's problem is the shading model and not the exposure.
        if let Some(middle) = probe(&pixels, matrix, symbol.position, 0) {
            out.center += hue_error(middle, symbol.color);
        }
        if peak < DIM {
            out.dim += 1;
        }
        if error > HUE_LOST {
            out.white += 1;
        }
        if symbol.class == "ship trace" {
            traces.push(rgb);
        }
    }
    // The median by brightness, not the brightest: when the failure *is*
    // saturation, the brightest sample is the one guaranteed to have lost its
    // colour and says nothing about the rest of the ring.
    traces.sort_by_key(|rgb| rgb[0].max(rgb[1]).max(rgb[2]));
    if let Some(rgb) = traces.get(traces.len() / 2) {
        out.trace = *rgb;
    }
    Ok(out)
}

/// The brightest pixel within one dot's width of where a map point projects, as
/// RGB. `None` if the point is off frame.
fn probe(
    pixels: &[u8],
    matrix: render::Mat4,
    position: sim::DVec3,
    radius: i32,
) -> Option<[u8; 3]> {
    let relative = position - orbit::eye_position();
    let relative = render::Vec3::new(relative.x as f32, relative.y as f32, relative.z as f32);
    let clip = matrix * relative.extend(1.0);
    if clip.w <= 0.0 {
        return None;
    }
    let ndc = render::Vec3::new(clip.x, clip.y, clip.z) / clip.w;
    if ndc.x.abs() > 0.97 || ndc.y.abs() > 0.97 {
        return None;
    }
    let col = ((ndc.x * 0.5 + 0.5) * EXTENT.0 as f32) as i32;
    let row = ((ndc.y * 0.5 + 0.5) * EXTENT.1 as f32) as i32;
    let mut best = [0u8; 3];
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let (x, y) = (col + dx, row + dy);
            if x < 0 || y < 0 || x >= EXTENT.0 as i32 || y >= EXTENT.1 as i32 {
                continue;
            }
            let at = (y as usize * EXTENT.0 as usize + x as usize) * 4;
            let Some(rgba) = pixels.get(at..at + 3) else {
                continue;
            };
            let peak = rgba[0].max(rgba[1]).max(rgba[2]);
            if peak > best[0].max(best[1]).max(best[2]) {
                best = [rgba[0], rgba[1], rgba[2]];
            }
        }
    }
    Some(best)
}

// ---- the tables ---------------------------------------------------------

/// The projection this instrument probes with has to be the one the frame was
/// drawn with, and the cheapest proof of that is a symbol whose pixel is known
/// independently: the ship sits at the map's origin and the eye looks at the
/// origin, so it lands in the middle of the frame or the matrix is not the one.
fn check_projection(renderer: &mut OffscreenRenderer) -> Result<()> {
    let view = View {
        pitch: orbit::EYE_PITCH,
        ..View::default()
    };
    let matrix = view.view_projection(EXTENT);
    let relative = sim::DVec3::ZERO - orbit::eye_position();
    let relative = render::Vec3::new(relative.x as f32, relative.y as f32, relative.z as f32);
    let clip = matrix * relative.extend(1.0);
    anyhow::ensure!(clip.w > 0.0, "the map's own centre is behind the eye");
    let ndc = render::Vec3::new(clip.x, clip.y, clip.z) / clip.w;
    anyhow::ensure!(
        ndc.x.abs() < 0.02 && ndc.y.abs() < 0.02,
        "the ship sits at the map's origin and the eye looks at it, so it must project to the \
         middle of the frame — this matrix puts it at ndc ({:.3}, {:.3})",
        ndc.x,
        ndc.y
    );
    let _ = renderer;
    Ok(())
}

/// What the shipped map does, at both ends of its own zoom key.
fn shipped(renderer: &mut OffscreenRenderer) -> Result<()> {
    println!(
        "\nthe shipped map — one point light at the eye, MAP_LUX {}\n  \
         (`SHIP_TRACE_INK` is (96,255,192); a ribbon that reads white has lost the only thing \
         it was carrying)",
        orbit::MAP_LUX
    );
    println!(
        "  situation   zoom | symbols |  dim | white | mean hue err | centre px | median trace dot"
    );
    for at in [Where::Parking, Where::Transfer] {
        for zoom in ZOOMS {
            let read = measure(renderer, at, zoom, Lamp::Eye(orbit::MAP_LUX))?;
            println!(
                "  {:9} {zoom:5.2} | {:7} | {:4} | {:5} | {:12.3} | {:9.3} | ({:3},{:3},{:3})",
                at.name(),
                read.seen,
                read.dim,
                read.white,
                read.error / read.seen.max(1) as f32,
                read.center / read.seen.max(1) as f32,
                read.trace[0],
                read.trace[1],
                read.trace[2],
            );
        }
    }
    Ok(())
}

/// Whether any lux makes the map legible. Totalled over both situations and
/// every zoom, because a knob that fixes one end at the other's expense is the
/// failure this table is looking for.
fn sweep(renderer: &mut OffscreenRenderer) -> Result<()> {
    println!(
        "\nthe lamp where it used to be — at the star, swept. Kept because it is the table that \
         says why it moved:\n  no value is a plateau, the two counts never falling together"
    );
    total_table(renderer, Lamp::Star)
}

/// The same measurement with the light at the eye instead. A map has no lighting
/// direction to report, so nothing about the picture is spent by moving it —
/// and every symbol is then the same distance from the lamp whatever the zoom,
/// which is the property the star can never have.
fn headlamp(renderer: &mut OffscreenRenderer) -> Result<()> {
    println!("\nthe light moved to the eye — every symbol lit head-on at EYE_RANGE, any zoom");
    total_table_over(renderer, Lamp::Eye, &FINE)
}

/// The headlamp's own sweep, an order of magnitude below the star's. What it is
/// looking for is the exposure at which the shading multiply lands on **one**: a
/// white light delivering unit irradiance returns the albedo it was handed, and
/// a symbol's albedo is the colour it was authored with.
const FINE: [f64; 7] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0];

fn total_table(renderer: &mut OffscreenRenderer, lamp: fn(f64) -> Lamp) -> Result<()> {
    total_table_over(renderer, lamp, &SWEEP)
}

fn total_table_over(
    renderer: &mut OffscreenRenderer,
    lamp: fn(f64) -> Lamp,
    sweep: &[f64],
) -> Result<()> {
    println!("      lux | symbols |  dim | white | mean hue err | centre px | median trace dot");
    for &lux in sweep {
        let mut total = Reading::default();
        let mut trace = [0u8; 3];
        for at in [Where::Parking, Where::Transfer] {
            for zoom in ZOOMS {
                let read = measure(renderer, at, zoom, lamp(lux))?;
                total.seen += read.seen;
                total.dim += read.dim;
                total.white += read.white;
                total.error += read.error;
                total.center += read.center;
                if read.trace != [0; 3] {
                    trace = read.trace;
                }
            }
        }
        println!(
            "  {lux:7.2} | {:7} | {:4} | {:5} | {:12.3} | {:9.3} | ({:3},{:3},{:3})",
            total.seen,
            total.dim,
            total.white,
            total.error / total.seen.max(1) as f32,
            total.center / total.seen.max(1) as f32,
            trace[0],
            trace[1],
            trace[2],
        );
    }
    Ok(())
}
