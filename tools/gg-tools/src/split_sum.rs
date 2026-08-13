//! What [Laz13]'s fit costs, and what a table buys (§6 M34).
//!
//! §6 M33 shipped a multiscatter correction that divides by the split-sum's
//! directional albedo, and named the fit it divides by as the limitation: the
//! fit's `scale + bias` has no view angle in it, so the correction is
//! view-independent by construction and the furnace's `n·v` sweep is flat by
//! algebra rather than by measurement. This is what that costs, in the two
//! places it is spent: `1/E`, which multiplies every direct highlight, and the
//! bias term, which is a dielectric's whole ambient specular.
//!
//! The reference is `gg_render::split_sum::integrate` at a large sample count —
//! the generator and the truth are one function, which is deliberate (§6 M33
//! made `ggc`'s prefilter its own ground truth for the same reason). What that
//! cannot catch is an integrand that is wrong in both, so the first table here
//! is a convergence check and the fit is printed beside it: two implementations
//! written years apart agreeing to a few per cent is the evidence that the
//! integrand is the one everybody else integrates.

use anyhow::Result;
use gg_math::sim::sqrt;
use gg_render::split_sum;

/// Sample count the reference runs at. The convergence table is what this was
/// read off — the estimator's own noise has to be below the error being
/// resolved, or the tables below grade sampling.
const REFERENCE: u32 = 65_536;

/// Sample count the uniform-hemisphere cross-check runs at. Far more than the
/// importance-sampled one needs, and still only good at the rough end — which
/// is the whole reason nobody integrates a BRDF this way except to check one.
const UNIFORM: u32 = 1 << 21;

/// Resolutions swept, so the shipped [`split_sum::EXTENT`] is a measured choice
/// rather than the number everyone else uses.
const EXTENTS: [usize; 4] = [8, 16, 32, 64];

/// Where the tables are read, in both axes. Corners, not centres: the ends of
/// the domain are where the fit is worst and where the correction is most
/// sensitive, so an interior-only sweep would flatter both.
const AXIS: [f32; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

pub fn run(_args: &[String]) -> Result<()> {
    converged();
    cross_check();
    let table = split_sum::table();
    albedo(&table);
    compensation(&table);
    bias(&table);
    resolution();
    Ok(())
}

/// Whether [`REFERENCE`] is enough samples to be a reference — the same
/// integral at a quarter of them, worst disagreement over the domain.
fn converged() {
    let mut worst = 0.0f32;
    for r in AXIS {
        for v in AXIS {
            let (a, b) = split_sum::integrate(r, v, REFERENCE);
            let (a4, b4) = split_sum::integrate(r, v, REFERENCE / 4);
            worst = worst.max((a - a4).abs()).max((b - b4).abs());
        }
    }
    println!(
        "reference: {REFERENCE} samples, worst move against {} is {worst:.5} — everything below \
         resolves differences larger than this and nothing smaller\n",
        REFERENCE / 4
    );
}

/// The importance-sampled estimator against a uniform one — two ways of
/// integrating the same BRDF that share only `D` and the visibility.
///
/// This is the table that decides whether anything below is worth reading. The
/// suspicious part of the reference is the grazing column, where `n·v` is
/// clamped and the importance-sampled weight has a cancelled `n·v` in it; a
/// uniform estimator has no such cancellation and no clamp to lean on.
fn cross_check() {
    println!("E at f0 = 1 — importance-sampled / uniform-hemisphere, {UNIFORM} samples");
    println!("  (uniform does not converge below roughness ~0.3 and is not printed there)");
    header();
    for r in AXIS {
        if r < 0.3 {
            continue;
        }
        print!("  {r:>4.1} |");
        for v in AXIS {
            let (a, b) = split_sum::integrate(r, v, REFERENCE);
            let u = split_sum::integrate_uniform(r, v, UNIFORM);
            print!(" {:>8.3}/{u:.3}", a + b);
        }
        println!();
    }
    println!();
}

/// `E = a + b`: the directional albedo at `f0 = 1`, which is what a white metal
/// in a furnace gives back and what the correction divides by.
fn albedo(table: &[[f32; 2]]) {
    println!("E = scale + bias at f0 = 1 — reference / [Laz13] fit / table");
    header();
    for r in AXIS {
        print!("  {r:>4.1} |");
        for v in AXIS {
            let (a, b) = split_sum::integrate(r, v, REFERENCE);
            let (fa, fb) = split_sum::fit(r, v);
            let (ta, tb) = split_sum::sample(table, r, v);
            print!(" {:.3}/{:.3}/{:.3}", a + b, fa + fb, ta + tb);
        }
        println!();
    }
    // The one claim §6 M33 could only make as algebra. A column that moves in
    // the reference and not in the fit is the fit's missing axis, measured.
    let (flat_r, flat_v) = (1.0, [AXIS[5], AXIS[1]]);
    let e = |v: f32| {
        let (a, b) = split_sum::integrate(flat_r, v, REFERENCE);
        a + b
    };
    let f = |v: f32| {
        let (a, b) = split_sum::fit(flat_r, v);
        a + b
    };
    println!(
        "\n  at roughness 1, n·v {:.1} → {:.1}: reference {:.3} → {:.3} ({:+.1}%), fit {:.3} → \
         {:.3} ({:+.1}%)\n",
        flat_v[0],
        flat_v[1],
        e(flat_v[0]),
        e(flat_v[1]),
        100.0 * (e(flat_v[1]) / e(flat_v[0]) - 1.0),
        f(flat_v[0]),
        f(flat_v[1]),
        100.0 * (f(flat_v[1]) / f(flat_v[0]) - 1.0),
    );
}

/// `1/E` — what §6 M33 multiplies every direct highlight by. The error here is
/// the one a picture shows, because it scales a term that is already the
/// brightest thing in the frame.
fn compensation(table: &[[f32; 2]]) {
    println!("1/E error, per cent of the reference — fit / table");
    header();
    let (mut fit_worst, mut table_worst) = (0.0f32, 0.0f32);
    for r in AXIS {
        print!("  {r:>4.1} |");
        for v in AXIS {
            let (a, b) = split_sum::integrate(r, v, REFERENCE);
            let (fa, fb) = split_sum::fit(r, v);
            let (ta, tb) = split_sum::sample(table, r, v);
            let truth = 1.0 / (a + b).max(1e-3);
            let ef = 100.0 * (1.0 / (fa + fb).max(1e-3) / truth - 1.0);
            let et = 100.0 * (1.0 / (ta + tb).max(1e-3) / truth - 1.0);
            fit_worst = fit_worst.max(ef.abs());
            table_worst = table_worst.max(et.abs());
            print!(" {ef:>+6.1}/{et:>+5.1}");
        }
        println!();
    }
    println!("  worst: fit {fit_worst:.1}%, table {table_worst:.1}%\n");
}

/// The bias term alone — a dielectric's `f0` is 0.04, so `f0 * a + b` is mostly
/// `b` and this is nearly the whole of what a painted wall reflects.
fn bias(table: &[[f32; 2]]) {
    println!("bias, absolute — reference / fit / table");
    header();
    for r in AXIS {
        print!("  {r:>4.1} |");
        for v in AXIS {
            let (_, b) = split_sum::integrate(r, v, REFERENCE);
            let (_, fb) = split_sum::fit(r, v);
            let (_, tb) = split_sum::sample(table, r, v);
            print!(" {b:.3}/{fb:.3}/{tb:.3}");
        }
        println!();
    }
    println!();
}

/// What the table's edge is worth, so [`split_sum::EXTENT`] is read off a sweep.
///
/// Measured at the *sampled* value rather than at texels: a coarse table with
/// bilinear interpolation is much better than its texel count suggests
/// everywhere the function is smooth, and much worse at the one corner where it
/// is not.
fn resolution() {
    println!("table edge vs worst 1/E error, per cent — the cost of the texels");
    println!("  linear n·v axis, then the same table with texels placed at n·v = x²");
    // The thing being replaced, on the same 41x41 grid, so the two are one
    // comparison rather than two tables with different denominators.
    let (mut worst, mut at, mut total) = (0.0f32, (0.0f32, 0.0f32), 0.0f32);
    for i in 0..=40 {
        for j in 0..=40 {
            let (r, v) = (i as f32 / 40.0, j as f32 / 40.0);
            let (a, b) = split_sum::integrate(r, v, REFERENCE);
            let (fa, fb) = split_sum::fit(r, v);
            let truth = 1.0 / (a + b).max(1e-3);
            let e = (100.0 * (1.0 / (fa + fb).max(1e-3) / truth - 1.0)).abs();
            total += e;
            if e > worst {
                (worst, at) = (e, (r, v));
            }
        }
    }
    println!(
        "  [Laz13] fit    (     0 B,    0.0 ms): mean {:.2}%, worst {worst:.2}% at roughness \
         {:.2}, n·v {:.2}",
        total / (41.0 * 41.0),
        at.0,
        at.1,
    );
    for (warp, samples) in [(false, 1024), (true, 1024), (true, 4096), (true, 16384)] {
        for extent in EXTENTS {
            let edge = (extent - 1) as f32;
            let started = std::time::Instant::now();
            let coarse: Vec<[f32; 2]> = (0..extent)
                .flat_map(|y| {
                    (0..extent).map(move |x| {
                        let v = x as f32 / edge;
                        let (a, b) = split_sum::integrate(
                            y as f32 / edge,
                            if warp { v * v } else { v },
                            samples,
                        );
                        [a, b]
                    })
                })
                .collect();
            // What the renderer pays for this edge at startup, once per process.
            let build = started.elapsed();
            // Read off-grid on purpose: on-grid samples would report the sample
            // count's error and say nothing about the interpolation between them.
            let (mut worst, mut at, mut total) = (0.0f32, (0.0f32, 0.0f32), 0.0f32);
            for i in 0..=40 {
                for j in 0..=40 {
                    let (r, v) = (i as f32 / 40.0, j as f32 / 40.0);
                    let (a, b) = split_sum::integrate(r, v, REFERENCE);
                    let (ta, tb) = bilinear(&coarse, extent, r, v, warp);
                    let truth = 1.0 / (a + b).max(1e-3);
                    let e = (100.0 * (1.0 / (ta + tb).max(1e-3) / truth - 1.0)).abs();
                    total += e;
                    if e > worst {
                        (worst, at) = (e, (r, v));
                    }
                }
            }
            let mean = total / (41.0 * 41.0);
            let bytes = extent * extent * 8;
            let axis = if warp { "x²    " } else { "linear" };
            println!(
                "  {axis} {extent:>3}² × {samples:>5} ({bytes:>6} B, {build:>6.1} ms): mean \
                 {mean:.2}%, worst {worst:.2}% at roughness {:.2}, n·v {:.2}",
                at.0,
                at.1,
                build = build.as_secs_f32() * 1e3,
            );
        }
    }
    println!();
}

/// [`split_sum::sample`] at an arbitrary edge, for the resolution sweep.
fn bilinear(
    table: &[[f32; 2]],
    extent: usize,
    roughness: f32,
    n_dot_v: f32,
    warp: bool,
) -> (f32, f32) {
    let edge = (extent - 1) as f32;
    let x = if warp { sqrt(n_dot_v) } else { n_dot_v };
    let (fx, fy) = (x * edge, roughness * edge);
    let (x0, y0) = (fx.floor().min(edge - 1.0), fy.floor().min(edge - 1.0));
    let (tx, ty) = (fx - x0, fy - y0);
    let at = |x: f32, y: f32| table[y as usize * extent + x as usize];
    let (c00, c10) = (at(x0, y0), at(x0 + 1.0, y0));
    let (c01, c11) = (at(x0, y0 + 1.0), at(x0 + 1.0, y0 + 1.0));
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    (
        lerp(lerp(c00[0], c10[0], tx), lerp(c01[0], c11[0], tx), ty),
        lerp(lerp(c00[1], c10[1], tx), lerp(c01[1], c11[1], tx), ty),
    )
}

fn header() {
    print!("  r\\n·v |");
    for v in AXIS {
        print!("{v:>18.1}");
    }
    println!();
}
