//! `orbit` — the two regimes graded against each other (§6 M38 pull 7).
//!
//! §2 splits a solar system in two: the many are *on rails*, where state is
//! `f(elements, tick)` and costs the same every tick, and the few are in a
//! *bubble*, integrated at the tick rate. The way that split fails is silently —
//! by the two halves disagreeing about the same body over the same interval —
//! and nothing in the tree can see it, which is what makes this an instrument
//! before it is a gate.
//!
//! Three tables, in the order the decisions come:
//!
//! - **the solve** — Kepler's equation is the only thing in the analytic half
//!   that is iterative at all, so its residual is where an on-rails body's
//!   error comes from. Swept over step count and eccentricity on both branches;
//!   [`gg_math::sim::KEPLER_STEPS`] is read off the plateau, and the two numbers
//!   failing in opposite directions are cost and residual.
//! - **the regimes** — the same initial state propagated both ways, reduced to
//!   two errors that mean different things and fail in opposite directions
//!   across the integrator: **radial**, which is the orbit being the wrong
//!   *shape* (a periapsis that grazes a planet it should have missed), and
//!   **along-track**, which is the right orbit walked at the wrong *rate* (a
//!   rendezvous that arrives late). A symplectic step bounds the first and lets
//!   the second run; a Runge–Kutta step trades the other way, and reading only
//!   a position error cannot tell them apart.
//! - **the budget** — how long a body may stay in the bubble before the two
//!   halves disagree by more than a metre, which is the number that sizes the
//!   regime boundary rather than a taste about it.
//!
//! The reference is the analytic propagator, and it is a reference because a
//! two-body trajectory conserves energy and angular momentum *exactly* — a
//! closed form that needs no reference implementation to argue with, checked as
//! a gg-math test and re-checked in table 2's own drift column, where the
//! analytic row must read zero.

use gg_math::sim::{self, DVec3, KEPLER_STEPS, Orbit, kepler_anomaly};

/// `GM` of the Sun and of Earth, m³/s². Real magnitudes: an orbit whose numbers
/// are all near 1 is the one case where a lost exponent does not show.
const MU_SUN: f64 = 1.327_124_400_18e20;
const MU_EARTH: f64 = 3.986_004_418e14;
/// 400 km up — the altitude a bubble is actually for.
const LEO_RADIUS: f64 = 6.771e6;
/// The engine's own tick, which is the bubble's step and not a free parameter.
const TICK_HZ: f64 = 60.0;

/// The integrators worth costing: what this tree already writes by habit, the
/// symplectic step, and the accurate one that is not symplectic.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    /// `v += a dt; p += v dt` — demo 11's gravity and demo 12's, unchanged.
    /// Symplectic, first order.
    Euler,
    /// Kick–drift–kick. Symplectic, second order, one force evaluation.
    Leapfrog,
    /// Fourth order, four force evaluations, and not symplectic.
    Rk4,
}

impl Step {
    fn name(self) -> &'static str {
        match self {
            Step::Euler => "semi-implicit euler",
            Step::Leapfrog => "leapfrog",
            Step::Rk4 => "rk4",
        }
    }

    /// Force evaluations per step — the honest cost axis, since a fourth-order
    /// step that costs four first-order ones is not free accuracy.
    fn evaluations(self) -> u32 {
        match self {
            Step::Rk4 => 4,
            _ => 1,
        }
    }
}

/// Two-body acceleration toward a primary at the origin.
fn gravity(position: DVec3, mu: f64) -> DVec3 {
    let r2 = position.length_squared();
    // `r²·√r²` rather than `powf(r, 3)`: one square root, and the banned-list
    // reason for `powf` is that it is not correctly rounded (§4.2.1).
    position * (-mu / (r2 * sim::sqrt(r2)))
}

/// One step of `kind`, in place.
fn advance(kind: Step, state: &mut (DVec3, DVec3), dt: f64, mu: f64) {
    let (position, velocity) = *state;
    *state = match kind {
        Step::Euler => {
            let v = velocity + gravity(position, mu) * dt;
            (position + v * dt, v)
        }
        Step::Leapfrog => {
            let half = velocity + gravity(position, mu) * (dt * 0.5);
            let p = position + half * dt;
            (p, half + gravity(p, mu) * (dt * 0.5))
        }
        Step::Rk4 => {
            let (k1p, k1v) = (velocity, gravity(position, mu));
            let (k2p, k2v) = (
                velocity + k1v * (dt * 0.5),
                gravity(position + k1p * (dt * 0.5), mu),
            );
            let (k3p, k3v) = (
                velocity + k2v * (dt * 0.5),
                gravity(position + k2p * (dt * 0.5), mu),
            );
            let (k4p, k4v) = (velocity + k3v * dt, gravity(position + k3p * dt, mu));
            (
                position + (k1p + k2p * 2.0 + k3p * 2.0 + k4p) * (dt / 6.0),
                velocity + (k1v + k2v * 2.0 + k3v * 2.0 + k4v) * (dt / 6.0),
            )
        }
    };
}

/// Specific orbital energy — the invariant a two-body trajectory holds exactly,
/// so a nonzero drift is the integrator's and never the orbit's.
fn energy(state: (DVec3, DVec3), mu: f64) -> f64 {
    state.1.length_squared() * 0.5 - mu / state.0.length()
}

/// The two errors that mean different things, in metres and signed: the offset
/// resolved onto the analytic body's own radial and in-track directions.
///
/// Projections rather than an angle between the two positions. `acos` of a dot
/// product is the obvious way to write this and it is unusable here — at LEO an
/// along-track error of 0.14 m subtends 2e-8 rad, whose cosine differs from 1 by
/// one double's epsilon, so the whole column quantises to multiples of 0.14 m and
/// reads *plausible*. Found by reading this instrument's first output rather than
/// by suspecting it.
fn split_error(integrated: DVec3, analytic: (DVec3, DVec3)) -> (f64, f64) {
    let offset = integrated - analytic.0;
    let Some(radial) = analytic.0.try_normalize() else {
        return (f64::NAN, f64::NAN);
    };
    let Some(normal) = analytic.0.cross(analytic.1).try_normalize() else {
        return (f64::NAN, f64::NAN);
    };
    (offset.dot(radial), offset.dot(normal.cross(radial)))
}

/// A circular-ish reference orbit about `mu` at `radius`, at eccentricity `e`.
/// Periapsis is held at `radius`, so raising `e` raises apoapsis rather than
/// dropping the body into the primary — the sweep stays a sweep of one thing.
fn path(mu: f64, radius: f64, eccentricity: f64) -> Orbit {
    Orbit {
        semi_major: radius / (1.0 - eccentricity),
        eccentricity,
        inclination: 0.4,
        ascending_node: 0.9,
        argument_of_periapsis: 0.3,
        mean_anomaly: 0.0,
        mu,
    }
}

/// Table 1 — where [`KEPLER_STEPS`] comes from. The residual *is* Kepler's
/// equation, so this grades the solve against its definition and not against a
/// table of answers somebody else computed.
fn table_solve() {
    println!("the solve — worst residual over one revolution, by step count\n");
    print!("  {:>6}", "e");
    for steps in 1..=6 {
        print!("{steps:>11}");
    }
    println!("     branch");
    for &(eccentricity, hyperbolic) in &[
        (0.0, false),
        (0.3, false),
        (0.7, false),
        (0.9, false),
        (0.95, false),
        (0.99, false),
        (0.999, false),
        (1.01, true),
        (1.2, true),
        (2.0, true),
        (10.0, true),
    ] {
        print!("  {eccentricity:>6}");
        for steps in 1..=6 {
            let mut worst: f64 = 0.0;
            for i in 0..512 {
                let t = f64::from(i) / 512.0;
                let (mean, residual) = if hyperbolic {
                    // A wide pass rather than one revolution: the hyperbolic
                    // starter is worst in the tail, where `asinh(M/e)` and the
                    // root diverge.
                    let mean = (t - 0.5) * 200.0;
                    let anomaly = kepler_anomaly(eccentricity, mean, steps);
                    (mean, eccentricity * sim::sinh(anomaly) - anomaly - mean)
                } else {
                    let mean = (t - 0.5) * core::f64::consts::TAU;
                    let anomaly = kepler_anomaly(eccentricity, mean, steps);
                    let raw = anomaly - eccentricity * sim::sin(anomaly) - mean;
                    // Folded: the solve answers for `M` wrapped into one
                    // revolution, so a residual of exactly 2π at the interval's
                    // end is the fold and not an error. Unfolded, every elliptic
                    // row reads a flat 1.52 and the step axis says nothing.
                    let tau = core::f64::consts::TAU;
                    (mean, raw - tau * (raw / tau).round())
                };
                // Relative to the magnitude of the equation's own terms: `e sinh H`
                // reaches 1e5 in the hyperbolic tail, and an absolute bound out
                // there asks for more digits than an f64 holds.
                worst = worst.max(residual.abs() / (mean.abs() + 1.0));
            }
            print!("{worst:>11.2e}");
        }
        println!(
            "     {}",
            if hyperbolic { "hyperbolic" } else { "elliptic" }
        );
    }
    println!(
        "\n  shipped KEPLER_STEPS = {KEPLER_STEPS} (Halley, cubic — the column where the row \
         stops improving)"
    );
}

/// The **envelope** of the disagreement over `revolutions`, plus the energy
/// drift at the end: worst radial, worst in-track, and the secular term.
///
/// The envelope and not the instantaneous error, because a symplectic step's
/// error is a bounded *oscillation* rather than a drift — sampled at one phase
/// it reads near zero and the reader concludes the integrator is exact. Sampled
/// every `STRIDE` ticks, since a bounded oscillation is smooth over a second of
/// a two-hour orbit and a comparison per tick would spend the whole runtime
/// inside the analytic half it is grading.
fn envelope(kind: Step, reference: Orbit, revolutions: f64, hz: f64) -> (f64, f64, f64) {
    const STRIDE: u32 = 64;
    let dt = 1.0 / hz;
    let span = reference.period().unwrap_or(1.0) * revolutions;
    let mut state = reference.state_at(0.0);
    let start = energy(state, MU_EARTH);
    let (mut worst_radial, mut worst_track) = (0.0f64, 0.0f64);
    let mut elapsed = 0.0;
    let mut since = 0;
    while elapsed < span {
        advance(kind, &mut state, dt, MU_EARTH);
        elapsed += dt;
        since += 1;
        if since == STRIDE {
            since = 0;
            let (radial, track) = split_error(state.0, reference.state_at(elapsed));
            worst_radial = worst_radial.max(radial.abs());
            worst_track = worst_track.max(track.abs());
        }
    }
    (
        worst_radial,
        worst_track,
        (energy(state, MU_EARTH) - start) / start.abs(),
    )
}

/// Table 2 — the two regimes over the same interval, split into the two errors
/// that fail in opposite directions.
fn table_regimes() {
    println!(
        "

the regimes — LEO at {TICK_HZ} Hz, worst disagreement in metres
"
    );
    println!(
        "  {:<20}{:>5}{:>7}{:>12}{:>12}{:>13}",
        "integrator", "e", "revs", "radial", "in-track", "energy"
    );
    for &eccentricity in &[0.0, 0.3, 0.7] {
        let reference = path(MU_EARTH, LEO_RADIUS, eccentricity);
        for kind in [Step::Euler, Step::Leapfrog, Step::Rk4] {
            for revolutions in [1.0, 10.0] {
                let (radial, track, drift) = envelope(kind, reference, revolutions, TICK_HZ);
                println!(
                    "  {:<20}{eccentricity:>5}{revolutions:>7}{radial:>12.2e}{track:>12.2e}{drift:>13.2e}",
                    kind.name()
                );
            }
        }
    }
    println!(
        "
  cost is force evaluations per tick: euler {}, leapfrog {}, rk4 {}",
        Step::Euler.evaluations(),
        Step::Leapfrog.evaluations(),
        Step::Rk4.evaluations()
    );
}

/// Table 3 — the same disagreement against the one knob the engine actually has.
/// A bubble's step *is* the tick, so this is the trade the milestone is choosing
/// between: force evaluations per second of sim against metres of disagreement.
fn table_budget() {
    const RATES: [f64; 4] = [15.0, 30.0, 60.0, 120.0];
    println!(
        "

the budget — worst disagreement over one revolution, by tick rate, e = 0.3
"
    );
    print!("  {:<20}", "integrator");
    for hz in RATES {
        print!("{:>20}", format!("{hz:.0} Hz"));
    }
    println!(
        "
  {:<20}{:>80}",
        "", "radial / in-track, metres"
    );
    let reference = path(MU_EARTH, LEO_RADIUS, 0.3);
    for kind in [Step::Euler, Step::Leapfrog, Step::Rk4] {
        print!("  {:<20}", kind.name());
        for hz in RATES {
            let (radial, track, _) = envelope(kind, reference, 1.0, hz);
            print!("{:>20}", format!("{radial:.2e} / {track:.2e}"));
        }
        println!();
    }
    let period = reference.period().unwrap_or(1.0);
    println!(
        "
  one revolution is {period:.0} s ({:.0} ticks at {TICK_HZ} Hz); a tick is {:.1e} of it",
        period * TICK_HZ,
        1.0 / (TICK_HZ * period)
    );
}

/// Table 4 — the analytic half's own cost, which is the claim the regime rests
/// on: a body a century out costs what a body next tick costs.
fn table_cost() {
    println!("\n\nthe analytic cost — one body evaluated at increasing distance in time\n");
    println!("  {:<24}{:>14}", "seconds after epoch", "ns per body");
    let reference = path(MU_SUN, 1.495_978_707e11, 0.2);
    for &seconds in &[0.0, 1.0e3, 1.0e6, 1.0e9, 3.15e9] {
        const RUNS: i32 = 200_000;
        // Best of five, like `hash-scale`'s: the quantity is a floor, and a
        // scheduler that took the thread away has measured the scheduler.
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut sink = DVec3::ZERO;
            let start = std::time::Instant::now();
            for i in 0..RUNS {
                let (position, _) = reference.state_at(seconds + f64::from(i % 64));
                sink += position;
            }
            let each = start.elapsed().as_secs_f64() * 1.0e9 / f64::from(RUNS);
            // Consumed so the loop cannot be folded away.
            std::hint::black_box(sink);
            best = best.min(each);
        }
        println!("  {seconds:<24.3e}{best:>14.1}");
    }
}

pub fn run(_args: &[String]) -> anyhow::Result<()> {
    table_solve();
    table_regimes();
    table_budget();
    table_cost();
    Ok(())
}
