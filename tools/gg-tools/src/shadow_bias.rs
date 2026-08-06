//! `gg-tools shadow-bias` — the §6 M11 acne knobs, measured instead of eyeballed.
//!
//! # What it compares against
//!
//! There is no analytic ground truth for "is this fragment self-shadowing", but
//! there is a cheap one: **acne is a function of texel size**, so the same scene
//! through a map eight times denser has an eighth of the error. The reference
//! leg therefore renders at `radius 10 / size 4096` — 4.9 mm texels — and the
//! test leg at `radius 10 / size 512`, which is 39.06 mm, *exactly* the texel
//! the shipping default (`radius 40 / size 2048`) produces. Same slab, same
//! coverage, so nothing falls off a cascade edge in one leg and not the other,
//! and a value tuned here is in texels and therefore transfers.
//!
//! # The two numbers
//!
//! Per pixel, against that reference:
//!
//! - **acne** — the test is *darker*. A surface shadowing itself.
//! - **leak** — the test is *brighter*. Peter-panning at a contact, or a real
//!   shadow the bias pushed a hole through.
//!
//! They pull in opposite directions, which is the whole reason this is a sweep
//! and not a constant: the answer is the **centre of the widest plateau where
//! acne is zero**, not the smallest bias that happens to look clean on one
//! frame at one sun angle. Five sun elevations, because the failure is angular —
//! a value that is clean at noon acnes at 10 degrees, and a value that survives
//! 10 degrees peter-pans at noon.
//!
//! # What is masked out, and why it has to be
//!
//! A denser map also has a *tighter penumbra*: the 3x3 kernel spans 12 cm in the
//! test leg and 1.5 cm in the reference, so every shadow boundary has a band
//! where the two legs disagree for a reason that is not acne and that no bias
//! can fix. Unmasked it floors both numbers around 2.4% and buries the signal.
//!
//! So pixels near a **reference** edge are excluded — from the reference
//! specifically, which is the part that makes it sound: acne is high-gradient
//! too, but it is not in the reference, so an acne stripe is never what defines
//! the mask that would hide it. The excluded fraction is printed; a run that
//! masked most of the frame measured nothing and should say so.

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

use crate::shadow_image::{BACKGROUND, DISAGREEMENT, luminance as luminance_of, near_edge};

/// Frame size. Small on purpose: the metric counts *fractions* of scene pixels,
/// and 320x240 is already tens of thousands of them.
const EXTENT: (u32, u32) = (320, 240);

/// The slab both legs share. Small enough that everything the camera sees is
/// inside one cascade, which is what makes the two legs comparable at all.
const RADIUS: f64 = 10.0;

/// Test-leg map edge: `2 * 10 / 512` is 39.06 mm, the shipping default's texel.
const TEST_SIZE: i64 = 512;

/// Reference-leg map edge — eight times denser, so an eighth of the error.
const REFERENCE_SIZE: i64 = 4096;

/// What the reference renders with. Generous in *texels* and therefore tiny in
/// metres at 4.9 mm a texel: ~24 mm of normal offset at grazing, which is under
/// a pixel of peter-panning at this framing. Deliberately not a value the sweep
/// can reach, so the reference is not judging the answer by its own settings.
const REFERENCE_NORMAL_BIAS: f64 = 4.0;
const REFERENCE_DEPTH_BIAS: f64 = 1.0;

/// Sun elevations above the horizon, and the unit direction light travels at
/// each. Written out rather than computed: `f64::sin` is a §3 grep gate and
/// pulling `gg_math::sim` in to build five constants would be ceremony.
///
/// The ground's incidence angle is 90 minus the elevation, so 10 degrees is
/// where the footprint error is worst and 75 is where peter-panning shows.
/// Azimuth is 45 degrees throughout, which puts the shadows diagonal to the
/// shadow map's texel grid — axis-aligned shadows hide exactly the stair-step
/// the kernel is there to filter.
const SUNS: &[(&str, [f64; 3])] = &[
    ("75deg", [-0.183_013, -0.965_926, -0.183_013]),
    ("55deg", [-0.405_580, -0.819_152, -0.405_580]),
    ("35deg", [-0.579_228, -0.573_576, -0.579_228]),
    ("20deg", [-0.664_463, -0.342_020, -0.664_463]),
    ("10deg", [-0.696_364, -0.173_648, -0.696_364]),
];

/// The `r.shadow_normal_bias` values swept, in shadow texels.
const NORMAL_BIASES: &[f64] = &[0.0, 0.5, 1.0, 1.5, 2.0, 3.0];

/// The `r.shadow_depth_bias` values swept, in shadow texels.
const DEPTH_BIASES: &[f64] = &[0.0, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5];

/// Pixels of dilation around each edge. The test leg's kernel is 1.5 texels of
/// 39 mm, and at this framing the ground runs 2-4 cm a pixel — so a boundary's
/// disagreement band is a handful of pixels wide, and 5 covers it at the near
/// end of the floor as well as the far.
const EDGE_DILATE: usize = 5;

/// How far above the *measured floor* a cell may sit and still count as on the
/// plateau, in fractions of judged pixels.
///
/// Floor-relative rather than absolute because the floor is not zero and cannot
/// be: at 70 degrees of incidence a 39 mm texel simply cannot resolve what a
/// 4.9 mm one does, and that irreducible difference lands in the acne column.
/// 0.0005 is about 30 pixels of 64000 — under the run-to-run scatter, so the
/// plateau it draws is the flat region rather than one lucky cell in it.
const FLOOR_SLACK: f64 = 0.000_5;

/// One cell of the sweep.
struct Cell {
    normal: f64,
    depth: f64,
    /// Worst over the five elevations, which is the only aggregate that means
    /// anything — a mean would let a value that fails at grazing average out
    /// against four angles where nothing is hard.
    worst_acne: f64,
    worst_leak: f64,
    worst_acne_at: &'static str,
    worst_leak_at: &'static str,
    /// Mean absolute output difference over scene pixels, worst elevation. The
    /// threshold-free number, for when two cells both read zero acne.
    worst_mean_delta: f64,
}

pub fn run(args: &[String]) -> anyhow::Result<()> {
    let json = args.iter().any(|a| a == "--json");
    for arg in args {
        anyhow::ensure!(
            arg == "--json",
            "unknown flag {arg:?} — shadow-bias takes --json"
        );
    }

    let mut renderer = OffscreenRenderer::new(EXTENT)?;
    println!(
        "gg-tools shadow-bias: {} on {}",
        EXTENT.0,
        renderer.device().chosen
    );
    println!(
        "  test      radius {RADIUS} / size {TEST_SIZE}  -> {:.2} mm texels (the shipping default's)",
        texel_mm(TEST_SIZE)
    );
    println!(
        "  reference radius {RADIUS} / size {REFERENCE_SIZE} -> {:.2} mm texels, normal \
         {REFERENCE_NORMAL_BIAS} / depth {REFERENCE_DEPTH_BIAS}",
        texel_mm(REFERENCE_SIZE)
    );

    cvars::SHADOW_DISTANCE.set_float(RADIUS);
    let mut cells = Vec::new();
    for &normal in NORMAL_BIASES {
        for &depth in DEPTH_BIASES {
            cells.push(Cell {
                normal,
                depth,
                worst_acne: 0.0,
                worst_leak: 0.0,
                worst_acne_at: "",
                worst_leak_at: "",
                worst_mean_delta: 0.0,
            });
        }
    }

    for &(label, direction) in SUNS {
        let world = scene(direction)?;
        // The reference first, once per elevation — it is the expensive leg and
        // every cell at this elevation is judged against the same pixels.
        cvars::SHADOW_SIZE.set_int(REFERENCE_SIZE);
        cvars::SHADOW_NORMAL_BIAS.set_float(REFERENCE_NORMAL_BIAS);
        cvars::SHADOW_DEPTH_BIAS.set_float(REFERENCE_DEPTH_BIAS);
        let reference = luminance(&mut renderer, &world)?;
        let judged = judged_mask(&reference);
        let kept = judged.iter().filter(|k| **k).count();

        cvars::SHADOW_SIZE.set_int(TEST_SIZE);
        for cell in &mut cells {
            cvars::SHADOW_NORMAL_BIAS.set_float(cell.normal);
            cvars::SHADOW_DEPTH_BIAS.set_float(cell.depth);
            let test = luminance(&mut renderer, &world)?;
            let (acne, leak, mean_delta) = compare(&test, &reference, &judged);
            if acne > cell.worst_acne {
                cell.worst_acne = acne;
                cell.worst_acne_at = label;
            }
            if leak > cell.worst_leak {
                cell.worst_leak = leak;
                cell.worst_leak_at = label;
            }
            cell.worst_mean_delta = cell.worst_mean_delta.max(mean_delta);
        }
        let masked = 100.0 - 100.0 * kept as f64 / judged.len() as f64;
        println!(
            "  swept {} cells at {label} — {kept}/{} pixels judged, {masked:.0}% masked",
            cells.len(),
            judged.len(),
        );
    }

    let report = renderer.shutdown();
    anyhow::ensure!(
        report.clean(),
        "unclean render: {} validation message(s), {} leak(s) — the sweep's numbers are not \
         trustworthy from a device that complained (§4.3, §5.4)",
        report.validation_messages,
        report.leaked_allocations.len(),
    );

    let floor = cells
        .iter()
        .map(|c| c.worst_acne)
        .fold(f64::INFINITY, f64::min);
    print_table(&cells, floor);
    let picked = recommend(&cells, floor);
    if json {
        let path = crate::output_dir()?.join("shadow-bias.json");
        std::fs::write(&path, to_json(&cells, picked))?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

fn texel_mm(size: i64) -> f64 {
    2.0 * RADIUS / size as f64 * 1000.0
}

/// The acne-hostile scene: one big flat receiver, four casters that touch it,
/// one that floats, and a tilted slab.
///
/// Everything is inside 7 m of the camera so it is inside the cascade in both
/// legs. The floating cube is the peter-panning detector — its shadow has a
/// visible gap to look for — and the ones sitting *on* the ground are the
/// opposite one: a contact shadow that detaches is exactly what too much bias
/// costs. The post is there because a long thin shadow at 10 degrees is where a
/// slope term used to punch holes.
fn scene(direction: [f64; 3]) -> anyhow::Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Light>()?;
    let mut put = |r: Renderable| -> anyhow::Result<()> {
        let e = world.spawn();
        world.insert(e, r)?;
        Ok(())
    };
    // Ground: top face at y = -1.6.
    put(Renderable::boxed(
        sim::DVec3::new(0.0, -2.0, -3.0),
        sim::Vec3::new(12.0, 0.4, 12.0),
        0x00b4_b0a8,
    ))?;
    put(Renderable::boxed(
        sim::DVec3::new(-1.2, -1.0, -3.0),
        sim::Vec3::splat(0.6),
        0x00c8_5a3c,
    ))?;
    put(Renderable::boxed(
        sim::DVec3::new(1.4, -1.05, -4.2),
        sim::Vec3::splat(0.55),
        0x003c_8ec8,
    ))?;
    put(Renderable::boxed(
        sim::DVec3::new(-2.6, -0.6, -4.6),
        sim::Vec3::new(0.18, 1.0, 0.18),
        0x0090_c85a,
    ))?;
    put(Renderable::boxed(
        sim::DVec3::new(0.5, -0.4, -2.2),
        sim::Vec3::splat(0.4),
        0x00d2_c85a,
    ))?;
    let mut slab = Renderable::boxed(
        sim::DVec3::new(2.6, -1.0, -2.6),
        sim::Vec3::new(1.2, 0.12, 1.2),
        0x0090_90a0,
    );
    slab.rotation = sim::DQuat::from_axis_angle(sim::DVec3::Z, 0.45);
    put(slab)?;
    // The only directional light, so `Sun::of` takes it as the caster.
    let sun = world.spawn();
    world.insert(
        sun,
        Light::sun(
            sim::Vec3::new(
                direction[0] as f32,
                direction[1] as f32,
                direction[2] as f32,
            ),
            0x00ff_f4e0,
            3.0,
        ),
    )?;
    Ok(world)
}

/// One render, reduced to per-pixel output luminance.
fn luminance(renderer: &mut OffscreenRenderer, world: &World) -> anyhow::Result<Vec<i32>> {
    let view = View {
        // Steep, so the whole visible ground is inside the cascade rather than
        // running to a horizon the single cascade cannot cover.
        pitch: -0.75,
        ..View::default()
    };
    let mut extracted = Extracted::default();
    extracted.transforms::<Renderable>(world, sim::DVec3::ZERO, view.frustum(EXTENT))?;
    extracted.append_lights(world)?;
    let frame = renderer.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?;
    anyhow::ensure!(
        frame.order.iter().any(|name| name.starts_with("shadow")),
        "no shadow pass ran — the sweep would be measuring two identical unlit frames"
    );
    Ok(luminance_of(&frame.pixels))
}

/// Which pixels a comparison is allowed to judge: surface, and clear of any
/// edge in the reference. See the module header for why the mask is taken from
/// the reference and not from the test or from their difference.
fn judged_mask(reference: &[i32]) -> Vec<bool> {
    let dilated = near_edge(reference, EXTENT, EDGE_DILATE);
    reference
        .iter()
        .zip(&dilated)
        .map(|(&l, &near)| !near && l >= BACKGROUND)
        .collect()
}

/// `(acne, leak, mean_delta)` over the judged pixels.
fn compare(test: &[i32], reference: &[i32], judged: &[bool]) -> (f64, f64, f64) {
    let (mut scene, mut acne, mut leak, mut total) = (0u64, 0u64, 0u64, 0i64);
    for ((&t, &r), &judge) in test.iter().zip(reference).zip(judged) {
        if !judge {
            continue;
        }
        scene += 1;
        total += i64::from((t - r).abs());
        if r - t > DISAGREEMENT {
            acne += 1;
        } else if t - r > DISAGREEMENT {
            leak += 1;
        }
    }
    if scene == 0 {
        return (0.0, 0.0, 0.0);
    }
    let scene = scene as f64;
    (
        acne as f64 / scene,
        leak as f64 / scene,
        total as f64 / scene,
    )
}

fn print_table(cells: &[Cell], floor: f64) {
    println!();
    println!("  normal  depth |    acne%          leak%       mean d");
    println!("  --------------+---------------------------------------");
    for cell in cells {
        let plateau = if cell.worst_acne <= floor + FLOOR_SLACK {
            "*"
        } else {
            " "
        };
        println!(
            "{plateau}  {:>5.2}  {:>5.2} | {:>7.3} @{:<6} {:>7.3} @{:<6} {:>6.2}",
            cell.normal,
            cell.depth,
            cell.worst_acne * 100.0,
            cell.worst_acne_at,
            cell.worst_leak * 100.0,
            cell.worst_leak_at,
            cell.worst_mean_delta,
        );
    }
    println!(
        "  (* = on the plateau: worst-elevation acne within {:.3} points of the {:.3}% floor)",
        FLOOR_SLACK * 100.0,
        floor * 100.0,
    );
}

/// The plateau's least-leak cell, and the cliff below it.
///
/// Among the cells at the acne floor the least leak wins, ties to the smallest
/// total offset — everything a tie leaves on the table is peter-panning nobody
/// asked for. Printed with the **cliff**: the largest constant offset that is
/// still *off* the plateau, which is the number saying how much margin a pick
/// has. The angle term contributes nothing head-on, so the cliff is a property
/// of the constant one alone. Both are reported rather than written anywhere;
/// defaults are source, and moving them is a review.
fn recommend(cells: &[Cell], floor: f64) -> Option<&Cell> {
    let plateau: Vec<&Cell> = cells
        .iter()
        .filter(|c| c.worst_acne <= floor + FLOOR_SLACK)
        .collect();
    let picked = plateau.iter().copied().min_by(|a, b| {
        a.worst_leak
            .total_cmp(&b.worst_leak)
            .then((a.normal + a.depth).total_cmp(&(b.normal + b.depth)))
    });
    // Read off the *smallest* normal-bias row, where the angle term contributes
    // nothing head-on: the largest constant offset that still acnes there is the
    // failure boundary. Taken over the whole grid it would find the other edge —
    // cells that fail from too *much* bias — and report the wrong end.
    let cliff = cells
        .iter()
        .filter(|c| c.normal <= NORMAL_BIASES[0] && c.worst_acne > floor + FLOOR_SLACK)
        .map(|c| c.depth)
        .fold(f64::NEG_INFINITY, f64::max);
    println!();
    match picked {
        Some(cell) => {
            println!(
                "  plateau: {}/{} cells. Least leak at normal {} depth {} \
                 (acne {:.3}%, leak {:.3}% @{})",
                plateau.len(),
                cells.len(),
                cell.normal,
                cell.depth,
                cell.worst_acne * 100.0,
                cell.worst_leak * 100.0,
                cell.worst_leak_at,
            );
            if cliff.is_finite() {
                println!(
                    "  cliff: depth {cliff} still acnes head-on. A shipped default wants margin \
                     over that, and leak is what margin costs."
                );
            }
        }
        None => println!("  every cell is at the floor — the grid does not reach the failure"),
    }
    picked
}

/// Hand-written rather than serde's: five fields and one array do not justify a
/// dependency, and `unused_dependencies` (§3) would be right to ask.
fn to_json(cells: &[Cell], picked: Option<&Cell>) -> String {
    let mut s = String::from("{\n  \"texel_mm\": ");
    s.push_str(&format!("{:.4},\n  \"cells\": [\n", texel_mm(TEST_SIZE)));
    for (i, cell) in cells.iter().enumerate() {
        s.push_str(&format!(
            "    {{\"normal\": {}, \"depth\": {}, \"acne\": {:.6}, \"acne_at\": \"{}\", \
             \"leak\": {:.6}, \"leak_at\": \"{}\", \"mean_delta\": {:.4}}}{}\n",
            cell.normal,
            cell.depth,
            cell.worst_acne,
            cell.worst_acne_at,
            cell.worst_leak,
            cell.worst_leak_at,
            cell.worst_mean_delta,
            if i + 1 == cells.len() { "" } else { "," },
        ));
    }
    s.push_str("  ],\n  \"recommended\": ");
    match picked {
        Some(cell) => s.push_str(&format!(
            "{{\"normal\": {}, \"depth\": {}}}\n}}\n",
            cell.normal, cell.depth
        )),
        None => s.push_str("null\n}\n"),
    }
    s
}
