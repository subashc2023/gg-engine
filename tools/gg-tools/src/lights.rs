//! `gg-tools lights` — what a light costs a frame, before and after §6 M30.
//!
//! The cap this prices is `gg_extract::MAX_POINT`, which was 32 for four
//! milestones with a comment saying that raising it was "a measurement rather
//! than a preference". This is that measurement. It renders the same frame at a
//! sweep of light counts under both values of `r.clusters` — which is not two
//! implementations but one: false is `cluster::Assignment` answering "every
//! light, every froxel", so the shader, the buffer and the draw are identical
//! and the only difference is what three thousand runs contain.
//!
//! Two arrangements, and they are chosen to disagree:
//!
//! - **hall** — lamps down a corridor, spread through depth. What a froxel grid
//!   is for: a fragment at the far end is shaded by the two lamps that reach it.
//! - **wall** — the same lamps all at one depth, filling the screen. The honest
//!   worst case: the depth axis has nothing to separate, so a froxel degenerates
//!   towards a screen tile and the win is whatever the tiling alone buys.
//!
//! What it reports per row is the median frame in milliseconds each way, the
//! ratio between them, and the occupancy the grid produced — the busiest froxel,
//! which is what a *pixel* pays, beside the pair total, which is what the loop
//! costs integrated over the frame. The two are not interchangeable: log slices
//! are thick at depth, so the froxel around a vanishing point can hold a third
//! of a hall while the frame's total stays near one percent.
//!
//! Nothing here gates. The frame time is the device's — run it under the pinned
//! lavapipe for a number that compares against another desk, and under the real
//! GPU (`VK_DRIVER_FILES` unset) for the one that answers what to ship.

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

/// A real frame, because the answer is per fragment and a small extent would
/// price the setup instead.
const EXTENT: (u32, u32) = (1280, 720);

/// Light counts swept. 32 is the pre-M30 cap and 256 is `MAX_POINT` after it;
/// the two below 32 are what says whether the grid costs anything when there is
/// nothing to select.
const COUNTS: [usize; 6] = [1, 8, 32, 64, 128, 256];

/// Frames thrown away before the clock starts — pipeline compiles, the first
/// staging ring wrap, and lavapipe's own warm-up.
const WARMUP: usize = 3;

/// Frames measured. The median is reported: a software rasterizer sharing a desk
/// with a browser produces outliers that no amount of averaging removes.
const FRAMES: usize = 11;

const EYE: [f64; 3] = [0.0, 1.5, 4.0];

/// Metres a lamp reaches zero at — long enough to cross the corridor, short
/// against its length.
const REACH: f32 = 3.0;

#[derive(Clone, Copy)]
enum Shape {
    /// Down the corridor: selection has the depth axis to work with.
    Hall,
    /// All at one depth, across the screen: it does not.
    Wall,
}

impl Shape {
    fn name(self) -> &'static str {
        match self {
            Shape::Hall => "hall",
            Shape::Wall => "wall",
        }
    }

    /// Where lamp `i` of `count` sits.
    fn place(self, i: usize, count: usize) -> sim::DVec3 {
        match self {
            Shape::Hall => {
                let side = if i.is_multiple_of(2) { -1.0 } else { 1.0 };
                sim::DVec3::new(side * 2.5, 1.35, -1.5 - (i / 2) as f64 * 1.2)
            }
            // A square-ish lattice across the corridor's far half, at one depth.
            Shape::Wall => {
                let across = (count as f64).sqrt().ceil() as usize;
                let (x, y) = (i % across, i / across);
                sim::DVec3::new(
                    (x as f64 - (across - 1) as f64 * 0.5) * 0.7,
                    0.4 + y as f64 * 0.7,
                    -14.0,
                )
            }
        }
    }
}

/// The corridor, and `count` lamps arranged in it.
fn world(shape: Shape, count: usize) -> anyhow::Result<World> {
    let mut world = World::new();
    world.register::<Renderable>()?;
    world.register::<Light>()?;
    let floor = world.spawn();
    world.insert(
        floor,
        Renderable::boxed(
            sim::DVec3::new(0.0, -0.1, -30.0),
            sim::Vec3::new(3.3, 0.1, 40.0),
            0x0086_8a8e,
        )
        .surfaced(0.7, 0.0),
    )?;
    for side in [-1.0, 1.0] {
        let wall = world.spawn();
        world.insert(
            wall,
            Renderable::boxed(
                sim::DVec3::new(side * 2.9, 2.2, -30.0),
                sim::Vec3::new(0.2, 2.4, 40.0),
                0x008e_8a82,
            )
            .surfaced(0.8, 0.0),
        )?;
    }
    // A back wall, so the `wall` arrangement has something behind it to light
    // and the two shapes shade a comparable number of fragments.
    let back = world.spawn();
    world.insert(
        back,
        Renderable::boxed(
            sim::DVec3::new(0.0, 2.0, -20.0),
            sim::Vec3::new(3.0, 2.4, 0.2),
            0x0090_8e88,
        )
        .surfaced(0.8, 0.0),
    )?;
    for i in 0..count {
        let lamp = world.spawn();
        world.insert(
            lamp,
            Light::point(shape.place(i, count), 0x00ff_e8c0, 7.0, REACH),
        )?;
    }
    Ok(world)
}

/// One row of the sweep: the median frame in ms, and what the grid held.
struct Row {
    ms: f64,
    load: gg_render::ClusterLoad,
    lit: usize,
}

fn measure(
    renderer: &mut OffscreenRenderer,
    world: &World,
    clustered: bool,
) -> anyhow::Result<Row> {
    cvars::CLUSTERS.set_bool(clustered);
    let view = View::default();
    let mut extracted = Extracted::default();
    let mut times: Vec<f64> = Vec::new();
    let mut lit = 0;
    for frame in 0..WARMUP + FRAMES {
        extracted.clear(
            sim::DVec3::new(EYE[0], EYE[1], EYE[2]),
            view.frustum(EXTENT),
        );
        extracted.append::<Renderable>(world)?;
        extracted.append_lights(world)?;
        lit = extracted.lights.len();
        // Wall clock around the whole call, which for the offscreen path is the
        // submit *and* the wait: it blocks until the frame retires, so this is
        // the frame and not the record of it.
        let started = std::time::Instant::now();
        let _ = renderer.frame(&extracted, &view, [0.0, 0.0, 0.0, 1.0], &[])?;
        if frame >= WARMUP {
            times.push(started.elapsed().as_secs_f64() * 1e3);
        }
    }
    times.sort_by(f64::total_cmp);
    Ok(Row {
        ms: times[times.len() / 2],
        load: renderer.cluster_load(),
        lit,
    })
}

pub fn run(args: &[String]) -> anyhow::Result<()> {
    anyhow::ensure!(args.is_empty(), "gg-tools lights takes no arguments");
    let mut renderer = OffscreenRenderer::new(EXTENT)?;
    println!("gg-tools lights — {}", renderer.device().chosen);
    println!("{EXTENT:?}, median of {FRAMES} frames after {WARMUP} warm-up\n");

    for shape in [Shape::Hall, Shape::Wall] {
        println!(
            "{:>5}  {:>4} {:>9} {:>9} {:>6}  {:>6} {:>8} {:>7}",
            shape.name(),
            "lit",
            "froxel ms",
            "frame ms",
            "ratio",
            "worst",
            "pairs",
            "of grid",
        );
        for count in COUNTS {
            let world = world(shape, count)?;
            let on = measure(&mut renderer, &world, true)?;
            let off = measure(&mut renderer, &world, false)?;
            println!(
                "{:>5}  {:>4} {:>9.2} {:>9.2} {:>5.2}x  {:>6} {:>8} {:>6.2}%",
                count,
                on.lit,
                on.ms,
                off.ms,
                off.ms / on.ms,
                on.load.worst,
                on.load.pairs,
                100.0 * on.load.pairs as f64 / (on.load.froxels * on.lit.max(1)) as f64,
            );
        }
        println!();
    }
    cvars::CLUSTERS.set_bool(true);
    // The device's own accounting, which is the half of "it costs nothing" that
    // a stopwatch cannot see.
    let report = renderer.shutdown();
    anyhow::ensure!(
        report.clean(),
        "unclean render: {} validation message(s), {} leak(s)",
        report.validation_messages,
        report.leaked_allocations.len(),
    );
    Ok(())
}
