//! `gg-tools facets` — a number for a crease in the field's answer (§6 M69).
//!
//! **The metric §6 M68 said it was missing.** That milestone built a `grain`
//! column in `bounce` — the mean absolute step between neighbouring sample points
//! — and recorded it as a metric that did not work: it reads flat across every
//! spacing, because a chevron is a wall-sized saddle and a lag-1 mean is blind to
//! it. Its residual named the operator that would work: *a second difference
//! across a cell boundary, counted rather than averaged*. This is that.
//!
//! # Why a second difference, and why a count
//!
//! Indirect light on a flat wall is a smooth function of position — the field
//! interpolates, and whatever it interpolates *between* varies over metres. So
//! the truth for the second difference of that picture is **zero everywhere**,
//! and no reference renderer is needed to say so. What is not zero is a crease: a
//! texel boundary in a probe's distance record, a cell boundary in the trilinear
//! weights, a probe switching off — each is a step or a kink in an otherwise
//! smooth surface, and a second difference is the operator that spikes at one.
//!
//! A **count** and not a mean, because a crease is a line and a picture is an
//! area: six creases across a 1024x128 wall are a percent of its pixels, so a
//! mean divides the defect by a hundred and reports the quantisation floor
//! instead — which is exactly what §6 M68's `grain` column did. The headline is
//! the *share of the frame* whose second difference clears [`BAR`], and `worst` is
//! beside it because one facet is still one facet.
//!
//! # Every row carries its own floor
//!
//! The readback is 8-bit, so a smooth gradient's second difference is already a
//! code value of rounding wherever the ramp crosses a step, and the room leg's
//! framing is a perspective plane whose own curvature is small but not zero.
//! Rather than argue either down, every row is measured **twice** — once as
//! shipped and once with `r.gi 0`, which is the same framing, the same
//! quantisation and the same geometry with the field removed. The second is the
//! row's floor, and the reported figure means nothing except against it.
//!
//! # What it is not
//!
//! It measures *smoothness*, not correctness. A field uniformly 30 % too dark
//! reads perfectly here, and a field with no probes at all reads at the floor.
//! `bounce` grades accuracy from one chair and `field` grades stability from a
//! moving one; this is the third axis, and it is the one a player reports.

use anyhow::Result;
use gg_ecs::World;
use gg_ecs::boundary::Renderable;
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

use crate::shadow_image::{self, luminance};
use crate::{field, views};

/// The graded frame — wide and short, because the band the structure lives in is
/// wide and short. Ten metres of wall across at a hundred pixels a metre, so a
/// facet a centimetre wide is not the sampling.
const WALL_EXTENT: (u32, u32) = (1024, 128);

/// Metres the ortho view is high, half. With [`WALL_EXTENT`]'s 8:1 it frames
/// 10.4 m of wall across and 1.3 up — the whole band of one probe cell above the
/// slab and nothing else, so every pixel is the wall's own face and no mask is
/// needed.
const WALL_ORTHO: f32 = 0.65;

/// Where the ortho eye sits: level with the middle of that band.
const WALL_EYE: sim::DVec3 = sim::DVec3::new(0.0, 3.4, -4.0);

/// The room leg's chair — `views`' own, so the numbers and the pictures that
/// reported the defect are taken from the same place.
const ROOM_EYE: sim::DVec3 = sim::DVec3::new(6.0, 2.10, 0.5);
const ROOM_YAW: f32 = core::f32::consts::PI;

/// **Aimed at the band, not at the room.** The wall above the shelter's roof is
/// 1.3 m of a 4 m wall eleven metres away — about a ninth of `views`' frame — and
/// a whole-room framing puts the other eight ninths of floor, pillar, stair and
/// sky in the same population. That is not a mask problem, it is the wrong
/// picture: `crease %` is a share, so a subject occupying a ninth of the frame is
/// divided by nine before it is compared to a floor made of everything else.
///
/// So the room leg is framed the way an operator would frame it — a narrow
/// vertical field on the band, wide enough across to hold five probe cells.
const ROOM_EXTENT: (u32, u32) = (1600, 300);
const ROOM_FOV: f32 = 0.15;
const ROOM_PITCH: f32 = 0.108;

/// The uniform environment for the wall leg, `bounce`'s value — with no `Sky` and
/// no `Light` in that world this is the whole of the direct term, and it is a
/// *constant*, so every code value that varies across the wall is the field's.
const AMBIENT: f64 = 0.25;

/// Code values of second difference that count as a crease.
///
/// **Two, and one would not do.** A smooth ramp quantised to 8 bits steps by one
/// code value every few pixels, and each step is a second difference of exactly
/// one — so a bar of one counts the ramp and reads several per cent on a picture
/// with nothing wrong with it. Two is the first value a ramp cannot reach and a
/// step in the shading can, and it is fixed rather than read off each row's floor
/// so that rows stay comparable.
const BAR: i32 = 2;

/// Pixels either side of a geometry or shadow edge the mask drops.
///
/// Three, which is `shadow-bias`' own band and for its reason: the resolve at a
/// silhouette spreads over a couple of pixels, and a second difference reaches one
/// further than the value it is taken of. The room leg keeps four fifths of its
/// frame at this radius, which is printed.
const EDGE_RADIUS: usize = 3;

/// Frames the field is given before a leg is graded, as a floor under the
/// `pending` condition — §6 M67's lesson and §6 M68's: a fixed count is only ever
/// right for one grid.
const FRAMES: usize = 24;

/// Frames after which a field that has not converged is a defect rather than a
/// wait.
const CEILING: usize = 4096;

pub fn run(args: &[String]) -> Result<()> {
    views::apply_sets(args)?;
    println!("gg-tools facets — how flat the field's answer is on a flat wall\n");

    // The dither is a deliberate ±1 code value on a *gradient*, which is the
    // whole of the floor this measures against and hides nothing at a facet's
    // several. Off, and said rather than implied.
    cvars::DITHER.set_float(0.0);
    // An antialias resolves a *geometry* edge; a facet is not one, and leaving it
    // on would measure the resolve kernel's own second difference.
    cvars::AA.set_bool(false);
    // Occlusion is a screen-space term with structure of its own (§6 M62's
    // rotation period) multiplying the same pixels. Off: the subject is the field.
    cvars::AO.set_bool(false);
    cvars::GI_RATE.set_int(0);

    let wall = wall()?;
    let room = field::world()?;
    let was = (cvars::GI_MOMENTS.int(), cvars::GI_SPACING.float());

    tiles(&wall, &room)?;
    spacings(&wall)?;
    cvars::GI_MOMENTS.set_int(was.0);
    cvars::GI_SPACING.set_float(was.1);

    cvars::DITHER.set_float(1.0);
    cvars::AA.set_bool(true);
    cvars::AO.set_bool(true);
    cvars::GI_RATE.set_int(16);
    Ok(())
}

/// What one leg produced.
struct Reading {
    /// Share of the graded population whose second difference clears [`BAR`], per
    /// cent.
    crease: f64,
    worst: i32,
    /// The lag along x at which the mean second difference correlates with itself
    /// best, in pixels — the *period* of whatever structure is there. Zero when
    /// nothing correlated.
    period: usize,
    /// The same share with `r.gi 0`: this framing's rounding, and nothing else.
    floor: f64,
    /// Share of the frame the edge mask kept. Printed rather than assumed: a
    /// metric that silently drops most of its picture reads as a clean one.
    kept: f64,
    /// Mean luminance of what was graded, code values — the scale the figures
    /// above are in, and the column that tells a flat picture from a black one.
    level: f64,
}

/// The tile-edge table — **the one that replaces a flat one**.
///
/// `r.gi_moments`' doc said the knob "does nearly nothing", on the strength of
/// `bounce`'s tile table reading flat to 0.3 % across 2, 4 and 8. That table
/// grades a *mean* against path-traced truth, and the tile's failure is neither
/// mean nor level: a Chebyshev bound point-sampled on an `edge`-by-`edge`
/// octahedral tile is a **discontinuous function of direction**, and its
/// discontinuities, projected from a probe onto a flat wall, are straight lines.
/// That is the reported defect and this is the column it appears in.
fn tiles(wall: &World, room: &World) -> Result<()> {
    println!(
        "the distance tile's edge — the probe visibility bound's angular resolution\n\
         \n  a wall above a slab, orthographic and head-on, lit by a uniform ambient and \
         nothing else\n  {}x{} over {:.1} m, {:.0} px a metre, one probe cell {:.0} px across\n",
        WALL_EXTENT.0,
        WALL_EXTENT.1,
        2.0 * f64::from(WALL_ORTHO) * f64::from(WALL_EXTENT.0) / f64::from(WALL_EXTENT.1),
        px_per_metre(),
        cvars::GI_SPACING.float() * px_per_metre(),
    );
    header();
    for (filter, edge) in reads() {
        cvars::GI_FILTER.set_bool(filter);
        cvars::GI_MOMENTS.set_int(edge);
        let name = format!("tile-{edge}-{}", label(filter));
        let reading = graded(Leg::Wall, wall, &name)?;
        row(&format!("tile {edge}, {:<8}", label(filter)), &reading);
    }
    println!("\n  demo 12's shelter from `views`' own chair — the picture that reported it\n");
    header();
    for (filter, edge) in reads() {
        cvars::GI_FILTER.set_bool(filter);
        cvars::GI_MOMENTS.set_int(edge);
        let name = format!("room-tile-{edge}-{}", label(filter));
        let reading = graded(Leg::Room, room, &name)?;
        row(&format!("tile {edge}, {:<8}", label(filter)), &reading);
    }
    cvars::GI_FILTER.set_bool(true);
    println!();
    Ok(())
}

/// The rows of the tile table: **point-sampled first, because that is what the
/// report was made against.**
///
/// Edge 2 is in the point half and reads clean, and that row is a **limit of this
/// leg rather than a virtue of the setting** — the crease picture confirms it. Four
/// texels over the sphere have only two boundary great circles, and from these
/// probes neither one crosses this framing, so there is no discontinuity in view to
/// find. A leg that frames one wall can only see the boundaries that land on it,
/// which is why the tile edge is not ranked here alone: `bounce`'s leak column is
/// the other half, and it reads edge 2 as leaking exactly as much as edge 8.
fn reads() -> impl Iterator<Item = (bool, i64)> {
    [(false, 2), (false, 4), (false, 8), (true, 4), (true, 8)].into_iter()
}

/// Unpadded, because it is half of a file name as well as half of a table cell —
/// the padding belongs at the table's format site, and putting it here wrote
/// `facets-tile-4-point   .png`.
fn label(filter: bool) -> &'static str {
    if filter { "filtered" } else { "point" }
}

/// The spacing table — what says the structure is the *field's* cell.
///
/// A facet's period should be the probe spacing projected into the frame. A
/// period column that tracks the spacing is the field; one that does not is
/// something else in the frame.
fn spacings(wall: &World) -> Result<()> {
    println!("the probe spacing — the period column is what attributes the structure\n");
    header();
    for spacing in [1.0, 2.0, 3.0, 4.0] {
        cvars::GI_SPACING.set_float(spacing);
        let reading = graded(Leg::Wall, wall, &format!("spacing-{spacing:.0}"))?;
        row(
            &format!("{spacing:.1} m = {:.0} px", spacing * px_per_metre()),
            &reading,
        );
    }
    println!();
    Ok(())
}

/// Pixels a metre of wall takes in the ortho leg.
fn px_per_metre() -> f64 {
    f64::from(WALL_EXTENT.1) / (2.0 * f64::from(WALL_ORTHO))
}

fn header() {
    println!("  leg                    | crease % | worst | period | floor % | graded % | level");
}

fn row(label: &str, r: &Reading) {
    let period = if r.period == 0 {
        "     -".to_owned()
    } else {
        format!("{:>4} px", r.period)
    };
    println!(
        "  {label:<22} | {:>7.3}  | {:>5} | {period} | {:>7.3} | {:>7.1}  | {:>5.1}",
        r.crease, r.worst, r.floor, r.kept, r.level
    );
}

/// Which scene and framing a row is measured in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Leg {
    /// The graded one: orthographic, head-on, truth exactly zero.
    Wall,
    /// The reported one: demo 12's room under perspective, whose floor carries the
    /// projection's own curvature and is printed for exactly that reason.
    Room,
}

impl Leg {
    fn extent(self) -> (u32, u32) {
        match self {
            Leg::Wall => WALL_EXTENT,
            Leg::Room => ROOM_EXTENT,
        }
    }
}

/// One row, on **its own device**.
///
/// §6 M67 and §6 M68 both found a sweep reading the previous row's field:
/// `Grid::covers`' hysteresis and `Grid::place`'s `sticky` are deliberate state,
/// and a leg that changes the tile edge or the spacing without moving the grid
/// leaves `pending` already zero. A device per row is the only version of this
/// that cannot be wrong.
fn graded(leg: Leg, world: &World, name: &str) -> Result<Reading> {
    if leg == Leg::Wall {
        cvars::AMBIENT.set_float(AMBIENT);
    }
    let mut renderer = OffscreenRenderer::new(leg.extent())?;
    let lit = settled(&mut renderer, leg, world)?;
    cvars::GI.set_bool(false);
    let flat = render(&mut renderer, leg, world)?;
    cvars::GI.set_bool(true);
    let report = renderer.shutdown();
    anyhow::ensure!(report.clean(), "unclean render: {report:?}");

    // **The mask comes off the `r.gi 0` leg, never the lit one** —
    // `shadow_image`'s standing argument, and it binds harder here than there. A
    // silhouette or a sun-shadow boundary is a second difference of a couple of
    // hundred code values, two orders above a facet, and it is *unchanged* by the
    // field: masking on the lit leg would let a facet define the band that hides
    // it. What is left is the interiors of surfaces, which is where a crease that
    // corresponds to nothing in the geometry is the whole subject.
    let mask = shadow_image::near_edge(&luminance(&flat), leg.extent(), EDGE_RADIUS);
    let kept = 100.0 * mask.iter().filter(|e| !**e).count() as f64 / mask.len() as f64;
    // The frame itself as well as the derivative. Without it a table reading zero
    // is ambiguous between a flat picture and a black one, and this instrument's
    // first two runs were the second — §6 M68's `views` defect, one module along.
    write_png(&lit, leg.extent(), name)?;
    let level = level(&lit, &mask);
    let lit = second_difference(&lit, leg.extent(), &mask);
    let flat = second_difference(&flat, leg.extent(), &mask);
    write_png(&crease(&lit), leg.extent(), &format!("{name}-crease"))?;
    Ok(Reading {
        crease: share(&lit.values),
        worst: lit.values.iter().copied().max().unwrap_or_default(),
        period: period(&lit.profile),
        floor: share(&flat.values),
        kept,
        level,
    })
}

/// Mean luminance of the graded population, code values — the scale every figure
/// beside it is in.
fn level(pixels: &[u8], mask: &[bool]) -> f64 {
    let luma = luminance(pixels);
    let kept: Vec<i32> = luma
        .iter()
        .zip(mask)
        .filter(|(_, m)| !**m)
        .map(|(v, _)| *v)
        .collect();
    if kept.is_empty() {
        return 0.0;
    }
    f64::from(kept.iter().sum::<i32>()) / kept.len() as f64
}

/// Render until the field has a record for every probe, then once more.
fn settled(renderer: &mut OffscreenRenderer, leg: Leg, world: &World) -> Result<Vec<u8>> {
    let mut pixels = render(renderer, leg, world)?;
    let mut frames = 1;
    // The condition, not the count — `FRAMES` is only the floor under it.
    while frames < FRAMES || renderer.field_pending().0 > 0 {
        pixels = render(renderer, leg, world)?;
        frames += 1;
        let (pending, probes) = renderer.field_pending();
        anyhow::ensure!(
            frames <= CEILING,
            "the field did not converge in {frames} frames: {pending} of {probes} ungathered"
        );
    }
    Ok(pixels)
}

/// One render.
///
/// **The wall leg is orthographic on purpose.** Under perspective a linear
/// function of world position is a ratio of linears in screen space, whose second
/// difference is small but not zero — an argument the floor column would then have
/// to carry. Head-on and orthographic, screen space *is* world space scaled, so
/// the truth for every figure in that table is exactly zero.
fn render(renderer: &mut OffscreenRenderer, leg: Leg, world: &World) -> Result<Vec<u8>> {
    let view = match leg {
        Leg::Wall => View {
            // Looking +z, at the wall.
            yaw: core::f32::consts::PI,
            pitch: 0.0,
            ortho: WALL_ORTHO,
            ..View::default()
        },
        Leg::Room => View {
            yaw: ROOM_YAW,
            pitch: ROOM_PITCH,
            fov_y: ROOM_FOV,
            ..View::default()
        },
    };
    let eye = match leg {
        Leg::Wall => WALL_EYE,
        Leg::Room => ROOM_EYE,
    };
    let extent = leg.extent();
    // `gg-runtime`'s order (`App::extract`), as `views` and `field` restate it:
    // lights, then the caster sweep, then the instances the sweep widened. A
    // different visible set is a different frame, and this leg has to be the same
    // frame the pictures were taken from.
    let mut extracted = Extracted::default();
    extracted.clear(eye, view.frustum(extent));
    extracted.append_lights(world)?;
    extracted.cast_shadows(view.caster_reach(extent));
    extracted.append::<Renderable>(world)?;
    Ok(renderer
        .frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?
        .pixels)
}

/// The absolute second difference of a frame's luminance, both axes.
struct Curvature {
    /// The graded population, two entries per unmasked interior pixel. Both axes
    /// in one population: a facet's boundary is diagonal, so a metric that walked
    /// x alone would report the fraction of it that happened to be vertical —
    /// §6 M66's finding about `ao`'s lag columns, one instrument along.
    values: Vec<i32>,
    /// Summed down the columns, for [`period`]. Rows are summed because a facet's
    /// boundary is a line: one row crosses it at one pixel, and the sum over rows
    /// is what turns that into a signal.
    profile: Vec<f64>,
    /// Every unmasked pixel's `dx + dy`, in frame order with the masked ones zero
    /// — the picture, which is the check on the count.
    image: Vec<i32>,
}

fn second_difference(pixels: &[u8], extent: (u32, u32), mask: &[bool]) -> Curvature {
    let (w, h) = (extent.0 as usize, extent.1 as usize);
    let luma = luminance(pixels);
    let at = |x: usize, y: usize| luma[y * w + x];
    let mut values = Vec::with_capacity(2 * w * h);
    let mut profile = vec![0.0f64; w];
    let mut image = vec![0i32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            if mask[y * w + x] {
                continue;
            }
            let dx = (at(x - 1, y) - 2 * at(x, y) + at(x + 1, y)).abs();
            let dy = (at(x, y - 1) - 2 * at(x, y) + at(x, y + 1)).abs();
            values.push(dx);
            values.push(dy);
            profile[x] += f64::from(dx + dy);
            image[y * w + x] = dx + dy;
        }
    }
    Curvature {
        values,
        profile,
        image,
    }
}

/// Share of the population at or above [`BAR`], per cent.
fn share(values: &[i32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let over = values.iter().filter(|v| **v >= BAR).count();
    100.0 * over as f64 / values.len() as f64
}

/// The lag the profile correlates with itself best at, in pixels.
///
/// A *period* and not a wavelength estimate: the search runs over whole pixel
/// lags from 8 up to half the frame, so a structure repeating at the probe
/// spacing lands within a pixel of it and noise lands nowhere in particular. Zero
/// when the best lag carries no more correlation than the mean — a picture with no
/// structure has no period, and reporting the argmax of noise would be reporting
/// one.
fn period(profile: &[f64]) -> usize {
    let n = profile.len();
    let mean = profile.iter().sum::<f64>() / n as f64;
    let centred: Vec<f64> = profile.iter().map(|p| p - mean).collect();
    let energy: f64 = centred.iter().map(|c| c * c).sum();
    if energy <= f64::EPSILON {
        return 0;
    }
    let mut best = (0usize, 0.0f64);
    for lag in 8..n / 2 {
        let sum: f64 = (0..n - lag).map(|i| centred[i] * centred[i + lag]).sum();
        // Normalised by the *overlap* as well as the energy, or a long lag scores
        // low for having fewer terms and every period reads short.
        let norm = sum / energy * n as f64 / (n - lag) as f64;
        if norm > best.1 {
            best = (lag, norm);
        }
    }
    // A fifth of the zero-lag energy is the bar. Below it the argmax is drift,
    // and a period column reporting drift is worse than an empty one.
    if best.1 < 0.2 { 0 } else { best.0 }
}

/// The second difference as a picture, [`GAIN`] code values to one.
///
/// The tables are the report and this is the check on them: a count is a number
/// about a population, and whether the population is six straight lines or a
/// field of noise is a question only the image answers.
fn crease(c: &Curvature) -> Vec<u8> {
    /// Code values of output per code value of second difference. A facet is
    /// single digits and the floor is one, so this puts the defect in the top
    /// half of the range and the floor at the bottom of it.
    const GAIN: i32 = 40;
    let mut out = Vec::with_capacity(4 * c.image.len());
    for d in &c.image {
        let v = (d * GAIN).min(255) as u8;
        out.extend_from_slice(&[v, v, v, 255]);
    }
    out
}

fn write_png(pixels: &[u8], extent: (u32, u32), name: &str) -> Result<()> {
    let path = crate::output_dir()?.join(format!("facets-{name}.png"));
    let file = std::fs::File::create(&path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), extent.0, extent.1);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(pixels)?;
    Ok(())
}

/// The graded scene: **demo 12's shelter, abstracted to the one thing that
/// produced the report.**
///
/// A floor, a tall wall, and a roofed alcove against it open on one side. The
/// roof's underside makes the probes below it dark and its top leaves the probes
/// above it open, so a point on the wall *above* the roof interpolates a cell with
/// one of each — which is only correct if the visibility bound rejects the dark
/// one, and rejecting it is exactly what the distance tile is for.
///
/// **The side walls are the difference between a measurement and a shrug.** The
/// first version was a floating slab, whose underside is lit from every side, so
/// the probes under it were barely darker than the ones above and the facet the
/// picture plainly has read as a `worst` of two code values. A shelter with sides
/// is a cave, and a cave is what the report is about.
///
/// The roof reaches past the frame on both sides so that nothing in view is near
/// one of its ends, everything is white, and there is no sky and no light: with
/// `r.ambient` the whole of the direct term, a constant, every code value that
/// varies across this wall is the field's.
fn wall() -> Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    let mut box_at = |center: sim::DVec3, half: sim::Vec3| -> Result<()> {
        let entity = world.spawn();
        world.insert(
            entity,
            Renderable::boxed(center, half, 0x00ff_ffff).surfaced(0.0, 0.0),
        )?;
        Ok(())
    };
    box_at(
        sim::DVec3::new(0.0, -0.25, 0.0),
        sim::Vec3::new(12.0, 0.25, 12.0),
    )?;
    box_at(
        sim::DVec3::new(0.0, 4.0, 6.25),
        sim::Vec3::new(12.0, 4.0, 0.25),
    )?;
    // Underside at 2.2 and top at 2.7, so the probe plane at y = 2 is clear of it
    // in the cave below and the plane at y = 4 is clear of it above — one cell
    // straddling the roof, which is the whole configuration.
    // **Eight metres deep, and the depth is the measurement.** At two and a half
    // the alcove is barely darker than the open floor, the probes either side of
    // the roof differ by a few per cent, and the facet the picture plainly has
    // reads as a `worst` of two code values. Deep enough that a probe at the back
    // of it sees the opening as a slot is what gives the two corners of a cell
    // something to disagree about — which is the same 6:1 the shelter's own
    // interior `Sky` declares against the daylight one.
    box_at(
        sim::DVec3::new(0.0, 2.45, 2.0),
        sim::Vec3::new(6.25, 0.25, 4.25),
    )?;
    for side in [-6.0, 6.0] {
        box_at(
            sim::DVec3::new(side, 1.1, 2.0),
            sim::Vec3::new(0.25, 1.1, 4.25),
        )?;
    }
    Ok(world)
}
