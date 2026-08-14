//! `orbit` — the two regimes graded against each other (§6 M38 pull 7).
//!
//! §2 splits a solar system in two: the many are *on rails*, where state is
//! `f(elements, tick)` and costs the same every tick, and the few are in a
//! *bubble*, integrated at the tick rate. The way that split fails is silently —
//! by the two halves disagreeing about the same body over the same interval —
//! and nothing in the tree can see it, which is what makes this an instrument
//! before it is a gate.
//!
//! Ten tables, in the order the decisions come — the first four grade the two
//! regimes against each other (item 7), the next three grade the *boundary*
//! between them (item 8), which §2 called the hard design problem before any of
//! this existed, and the last three are what **warp** asks of both (item 10):
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
//! - **the round trip** — [`Orbit::from_state`] graded on the two things a
//!   regime condition might be written against, which fail in opposite
//!   directions as an orbit approaches circular or equatorial: the **elements**,
//!   which go arbitrary, and the **state**, which does not move off its floor.
//! - **the boundary** — a body `d` from a planet propagated three ways, where
//!   rails error falls as `1/d²` and the bubble's does not depend on `d` at all,
//!   so they cross; every row carries the reference's own floor, because past
//!   the crossing both columns are measuring it.
//! - **the transition** — whether *crossing* costs anything, with total bubble
//!   time held fixed and only the number of handovers changing.
//! - **the stride** — a warp factor spent as the tick period, which is the
//!   candidate design pull 2 has to choose against. Two currencies: metres for
//!   the regime that steps, nothing at all for the regime that does not.
//! - **the second conic** — the column table 6 never had. A coasting body near a
//!   planet is still analytic, about a *different primary*, and the two conics
//!   fail in opposite directions across the separation axis. Which matters for
//!   warp rather than for accuracy: warp is refused while anything is
//!   integrated, so an analytic answer close in is what makes warp reachable.
//! - **the sampling** — the hazard a stride creates for the *condition* rather
//!   than the physics, since item 8's regime test is read every `k` ticks and a
//!   band crossed in fewer than `k` is crossed invisibly. Measured off a real
//!   encounter's own trajectory, and the answer is that it never binds.
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

/// Radians folded to `(-π, π]`. An element difference of a full turn is the
/// representation and not a disagreement.
fn wrap(angle: f64) -> f64 {
    let tau = core::f64::consts::TAU;
    angle - tau * (angle / tau).round()
}

/// Table 5 — the conversion the boundary is made of, graded on the two things a
/// regime condition might be written against.
///
/// **elements** is what a condition reading `argument_of_periapsis` would see;
/// **state** is what the picture, the hash and the player see. They fail in
/// opposite directions down the degeneracy axis, and the third column is why —
/// the sum the elements are ill-conditioned *around* is not.
fn table_round_trip() {
    println!(
        "

the round trip — Orbit::from_state after state_at, at real magnitudes (1 AU)
"
    );
    println!(
        "  {:<12}{:<12}{:>14}{:>14}{:>14}",
        "e", "i", "worst element", "Ω+ω+M", "state, m"
    );
    for &(eccentricity, inclination) in &[
        (0.3, 0.62),
        (1.0e-3, 0.62),
        (1.0e-6, 0.62),
        (1.0e-9, 0.62),
        (1.0e-12, 0.62),
        (0.0, 0.62),
        (0.3, 1.0e-6),
        (0.3, 1.0e-12),
        (0.3, 0.0),
        (0.0, 0.0),
    ] {
        let reference = Orbit {
            semi_major: 1.495_978_707e11,
            eccentricity,
            inclination,
            ascending_node: -2.4,
            argument_of_periapsis: 1.15,
            mean_anomaly: 0.37,
            mu: MU_SUN,
        };
        let period = reference.period().unwrap_or(1.0);
        let (mut worst_element, mut worst_sum, mut worst_state) = (0.0f64, 0.0f64, 0.0f64);
        for step in 0..16 {
            let seconds = period * f64::from(step) / 16.0;
            let (position, velocity) = reference.state_at(seconds);
            let Some(back) = Orbit::from_state(position, velocity, MU_SUN) else {
                continue;
            };
            // What the elements *should* have been at this epoch: only the mean
            // anomaly has moved, and it moved by a quantity with no cancellation
            // in it.
            let expected = reference.mean_anomaly + reference.mean_motion() * seconds;
            for (had, wanted) in [
                (back.ascending_node, reference.ascending_node),
                (back.argument_of_periapsis, reference.argument_of_periapsis),
                (back.mean_anomaly, expected),
            ] {
                worst_element = worst_element.max(wrap(had - wanted).abs());
            }
            worst_sum = worst_sum.max(
                wrap(
                    (back.ascending_node + back.argument_of_periapsis + back.mean_anomaly)
                        - (reference.ascending_node + reference.argument_of_periapsis + expected),
                )
                .abs(),
            );
            worst_state = worst_state.max((back.state_at(0.0).0 - position).length());
        }
        println!(
            "  {eccentricity:<12.0e}{inclination:<12.0e}\
             {worst_element:>14.2e}{worst_sum:>14.2e}{worst_state:>14.2e}"
        );
    }
    println!(
        "
  an f64 metre at 1 AU is {:.1e} of it — the state column's own floor",
        f64::EPSILON
    );
}

/// The acceleration a bubble body feels: the primary, plus a secondary that is
/// itself on rails — which is what a planet is in this architecture.
fn perturbed(position: DVec3, secondary: Orbit, seconds: f64) -> DVec3 {
    let (centre, _) = secondary.state_at(seconds);
    gravity(position, MU_SUN) + gravity(position - centre, MU_EARTH)
}

/// One step against a *time-varying* field — the perturber moves inside the
/// step, so each stage carries its own time and `advance`'s two-body form cannot
/// be reused.
fn advance_perturbed(kind: Step, state: &mut (DVec3, DVec3), at: f64, dt: f64, secondary: Orbit) {
    let force = |p: DVec3, t: f64| perturbed(p, secondary, t);
    let (position, velocity) = *state;
    *state = match kind {
        Step::Euler => {
            let v = velocity + force(position, at) * dt;
            (position + v * dt, v)
        }
        Step::Leapfrog => {
            let half = velocity + force(position, at) * (dt * 0.5);
            let p = position + half * dt;
            (p, half + force(p, at + dt) * (dt * 0.5))
        }
        Step::Rk4 => {
            let (half, mid) = (dt * 0.5, at + dt * 0.5);
            let (k1p, k1v) = (velocity, force(position, at));
            let (k2p, k2v) = (velocity + k1v * half, force(position + k1p * half, mid));
            let (k3p, k3v) = (velocity + k2v * half, force(position + k2p * half, mid));
            let (k4p, k4v) = (velocity + k3v * dt, force(position + k3p * dt, at + dt));
            (
                position + (k1p + k2p * 2.0 + k3p * 2.0 + k4p) * (dt / 6.0),
                velocity + (k1v + k2v * 2.0 + k3v * 2.0 + k4v) * (dt / 6.0),
            )
        }
    };
}

/// Propagate `state` to each checkpoint, returning the position at each.
///
/// An integer step count per mark, and `at` recovered from it rather than
/// accumulated. A `while at < mark` loop overshoots by up to one step, and at
/// heliocentric speed a 480 Hz step is 62 metres — which is what the first
/// version of table 6 measured, in every column, including the ones where the
/// physics it claimed to be showing had gone to zero. The reference-against-
/// itself line is what caught it.
fn checkpoints(
    kind: Step,
    mut state: (DVec3, DVec3),
    hz: f64,
    marks: &[f64],
    secondary: Orbit,
) -> Vec<DVec3> {
    let dt = 1.0 / hz;
    let mut out = Vec::with_capacity(marks.len());
    let mut taken = 0i64;
    for &mark in marks {
        #[allow(clippy::cast_possible_truncation)]
        let wanted = (mark * hz).round() as i64;
        while taken < wanted {
            #[allow(clippy::cast_precision_loss)]
            let at = taken as f64 * dt;
            advance_perturbed(kind, &mut state, at, dt, secondary);
            taken += 1;
        }
        out.push(state.0);
    }
    out
}

/// Table 6 — the boundary itself. A body near a planet, propagated three ways
/// over the same interval: **rails**, which is the conic about the star with the
/// planet ignored; **bubble**, which is leapfrog at the tick rate with both
/// masses; and a converged reference neither of them is.
///
/// The two numbers fail in opposite directions across the separation axis. Rails
/// error is the neglected planet and falls as `1/d²`; bubble error is the step's
/// own and does not care about `d` at all. Where they cross is where putting a
/// body on rails stops being an approximation and starts being an *improvement*,
/// and the point of measuring rather than asserting is that the crossing is not
/// where the classical sphere of influence puts it.
fn table_boundary() {
    const MARKS: [f64; 3] = [60.0, 600.0, 3600.0];
    // The planet, on its own rails: circular, one AU, and the only body in this
    // table whose own motion is analytic by construction rather than by choice.
    let planet = Orbit {
        semi_major: 1.495_978_707e11,
        eccentricity: 0.0,
        inclination: 0.0,
        ascending_node: 0.0,
        argument_of_periapsis: 0.0,
        mean_anomaly: 0.0,
        mu: MU_SUN,
    };
    println!(
        "

the boundary — a body `d` from a planet, over {:.0} s
",
        MARKS[MARKS.len() - 1]
    );
    println!(
        "  {:<12}{:>12}{:>32}{:>32}{:>12}",
        "d, m",
        "a_p/a_star",
        "rails error at 60/600/3600 s",
        "bubble error at 60/600/3600 s",
        "floor"
    );
    let (centre, drift) = planet.state_at(0.0);
    for &separation in &[1.0e7, 1.0e8, 1.0e9, 1.5e9, 1.0e10, 1.0e11, 1.0e12] {
        // Offset along the star line and given the circular speed for *its* own
        // radius, so the body is on a genuine heliocentric orbit that happens to
        // pass near the planet rather than one bolted to it.
        let Some(radial) = centre.try_normalize() else {
            continue;
        };
        let position = centre + radial * separation;
        let radius = position.length();
        let Some(heading) = drift.try_normalize() else {
            continue;
        };
        let velocity = heading * sim::sqrt(MU_SUN / radius);
        let ratio = MU_EARTH / (separation * separation) / (MU_SUN / (radius * radius));

        let Some(rails) = Orbit::from_state(position, velocity, MU_SUN) else {
            continue;
        };
        let truth = checkpoints(Step::Rk4, (position, velocity), 480.0, &MARKS, planet);
        // The same reference at double the rate. A row whose errors sit at this
        // is a row that measured the reference, and the far rows do — without the
        // column a reader takes the noise for the physics, which is the mistake
        // this table's first version made in every cell.
        let floor = checkpoints(Step::Rk4, (position, velocity), 960.0, &MARKS, planet)
            .iter()
            .zip(&truth)
            .map(|(a, b)| (*a - *b).length())
            .fold(0.0f64, f64::max);
        let bubble = checkpoints(
            Step::Leapfrog,
            (position, velocity),
            TICK_HZ,
            &MARKS,
            planet,
        );
        let rails_error: Vec<f64> = MARKS
            .iter()
            .zip(&truth)
            .map(|(&mark, &want)| (rails.state_at(mark).0 - want).length())
            .collect();
        let bubble_error: Vec<f64> = bubble
            .iter()
            .zip(&truth)
            .map(|(&had, &want)| (had - want).length())
            .collect();
        let show = |errors: &[f64]| {
            errors
                .iter()
                .map(|e| format!("{e:.1e}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        println!(
            "  {separation:<12.1e}{ratio:>12.1e}{:>32}{:>32}{floor:>12.1e}",
            show(&rails_error),
            show(&bubble_error)
        );
    }
    println!(
        "
  the planet's Hill radius is 1.5e9 m — where the classical answer puts the boundary"
    );
}

/// Table 7 — whether crossing the boundary *costs* anything, which nothing else
/// here answers. Total time in the bubble is held fixed and only the number of
/// handovers changes, so a curve that rises with `k` is the conversion's price
/// and a flat one says the price is zero and only residence matters.
fn table_transition() {
    let reference = path(MU_EARTH, LEO_RADIUS, 0.3);
    let period = reference.period().unwrap_or(1.0);
    let dt = 1.0 / TICK_HZ;
    println!(
        "

the transition — one LEO revolution, half of it integrated, split k ways
"
    );
    println!(
        "  {:<10}{:>16}{:>16}{:>18}",
        "handovers", "position, m", "elements, rad", "converted only, m"
    );
    for handovers in [1, 2, 4, 8, 16, 64] {
        let span = period / f64::from(handovers);
        // The control: the same k handovers with *no* integration between them,
        // which is the conversion's price on its own. Without it a curve rising
        // in k cannot be told from the integrator's envelope being sampled more.
        let mut clean = reference.state_at(0.0);
        for step in 0..handovers {
            if let Some(rails) = Orbit::from_state(clean.0, clean.1, MU_EARTH) {
                clean = rails.state_at(span);
            }
            let _ = step;
        }
        let converted = (clean.0 - reference.state_at(period).0).length();

        let mut state = reference.state_at(0.0);
        let mut at = 0.0;
        for _ in 0..handovers {
            // Rails for the first half of the slice: elements from the state we
            // are holding, then evaluated forward — exactly the handover a
            // regime exit performs.
            if let Some(rails) = Orbit::from_state(state.0, state.1, MU_EARTH) {
                state = rails.state_at(span * 0.5);
            }
            at += span * 0.5;
            let until = at + span * 0.5;
            while at < until {
                advance(Step::Leapfrog, &mut state, dt, MU_EARTH);
                at += dt;
            }
        }
        let (wanted, _) = reference.state_at(at);
        let elements = Orbit::from_state(state.0, state.1, MU_EARTH).map_or(f64::NAN, |had| {
            wrap(had.mean_anomaly - reference.mean_anomaly - reference.mean_motion() * at).abs()
        });
        println!(
            "  {handovers:<10}{:>16.2e}{elements:>16.2e}{converted:>18.2e}",
            (state.0 - wanted).length()
        );
    }
    println!(
        "
  the same half-revolution integrated in one stretch is the k = 1 row; the
  leapfrog envelope over a whole revolution is table 2's {:.2e} m",
        envelope(Step::Leapfrog, reference, 1.0, TICK_HZ).0
    );
}

/// The envelope again, sampled every step rather than every 64th.
///
/// Table 8's low rates run tens of steps per revolution, where a stride of 64
/// takes no sample at all and the row reads a confident zero — the same class of
/// mistake as table 6's overshoot, and caught the same way, by a row that had no
/// business being clean. Every step is affordable here precisely because the
/// rates are low.
fn envelope_every_step(reference: Orbit, hz: f64) -> (f64, f64, f64) {
    let dt = 1.0 / hz;
    let span = reference.period().unwrap_or(1.0);
    let mut state = reference.state_at(0.0);
    let start = energy(state, MU_EARTH);
    let (mut worst_radial, mut worst_track) = (0.0f64, 0.0f64);
    #[allow(clippy::cast_possible_truncation)]
    let steps = (span * hz).round().max(1.0) as i64;
    for i in 1..=steps {
        advance(Step::Leapfrog, &mut state, dt, MU_EARTH);
        if !(state.0.length_squared() + state.1.length_squared()).is_finite() {
            return (f64::INFINITY, f64::INFINITY, f64::INFINITY);
        }
        #[allow(clippy::cast_precision_loss)]
        let (radial, track) = split_error(state.0, reference.state_at(i as f64 * dt));
        worst_radial = worst_radial.max(radial.abs());
        worst_track = worst_track.max(track.abs());
    }
    (
        worst_radial,
        worst_track,
        (energy(state, MU_EARTH) - start) / start.abs(),
    )
}

/// Table 8 — the first of pull 2's two candidate warp designs, and the one this
/// kills. Spending a warp factor `k` as tick *stride* means the integrated
/// regime steps `k / hz` seconds at a time while the analytic one is evaluated
/// `k` periods further along — so the two halves pay in different currencies and
/// the table is worth reading as two columns rather than one verdict.
///
/// The analytic column is not measured because there is nothing to measure: a
/// conic evaluated at `t + k` is the same closed form as at `t + 1`, exact at
/// every row by construction, and table 4 already priced it flat in `k`.
fn table_stride() {
    println!(
        "
the stride — a warp factor spent as the tick period, LEO e = 0.3
"
    );
    println!(
        "  {:<10}{:>12}{:>14}{:>26}{:>13}{:>12}",
        "warp k", "dt, s", "steps / rev", "radial / in-track, m", "energy", "analytic"
    );
    let reference = path(MU_EARTH, LEO_RADIUS, 0.3);
    let period = reference.period().unwrap_or(1.0);
    for k in [1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0] {
        let hz = TICK_HZ / k;
        let (radial, track, drift) = envelope_every_step(reference, hz);
        let shape = if radial.is_finite() {
            format!("{radial:.2e} / {track:.2e}")
        } else {
            "diverged".to_string()
        };
        println!(
            "  {k:<10.0}{:>12.3}{:>14.1}{shape:>26}{:>13}{:>12}",
            1.0 / hz,
            period * hz,
            if drift.is_finite() {
                format!("{drift:.2e}")
            } else {
                "—".to_string()
            },
            "exact"
        );
    }
    println!(
        "
  sampled every step, not table 3's every 64th — at the bottom rows a revolution
  is fewer than 64 steps and a strided sample would report a confident zero.
  the energy column stays at the floor while the position column crosses four
  orders: leapfrog is symplectic at any step, so the invariant anyone would have
  checked forgives a stride that has already lost the orbit's shape"
    );
}

/// The absolute position of a body held as a conic about the *planet* — the
/// third answer nothing had asked for, and the one patched conics are made of.
///
/// The planet frame is not inertial, so this is an approximation in its own
/// right and fails in the opposite direction from the star conic: it ignores the
/// star entirely, which is ruinous far out and correct close in.
fn planet_conic(relative: (DVec3, DVec3), planet: Orbit, seconds: f64) -> Option<DVec3> {
    let conic = Orbit::from_state(relative.0, relative.1, MU_EARTH)?;
    Some(planet.state_at(seconds).0 + conic.state_at(seconds).0)
}

/// Table 9 — the column table 6 did not have, and the reason it matters is
/// pull 2 rather than pull 6. Warp is forbidden while anything is being
/// integrated, so the cost of the regime boundary is *residence in the bubble*,
/// and table 6 asked only whether the star conic or an integration was closer.
/// A body coasting near a planet is still analytic — about a different primary —
/// and where the two conics cross is where the primary changes, which is a
/// different question with a different answer.
fn table_second_conic() {
    const MARKS: [f64; 3] = [60.0, 600.0, 3600.0];
    let planet = Orbit {
        semi_major: 1.495_978_707e11,
        eccentricity: 0.0,
        inclination: 0.0,
        ascending_node: 0.0,
        argument_of_periapsis: 0.0,
        mean_anomaly: 0.0,
        mu: MU_SUN,
    };
    println!(
        "

the second conic — the same body, both analytic answers and the integrated one
"
    );
    println!(
        "  {:<12}{:>12}{:>16}{:>16}{:>14}{:>12}",
        "d, m", "a_p/a_star", "about star, m", "about planet, m", "bubble, m", "floor"
    );
    let (centre, drift) = planet.state_at(0.0);
    for &separation in &[1.0e7, 1.0e8, 5.0e8, 1.0e9, 1.5e9, 5.0e9, 1.0e10, 1.0e11] {
        let (Some(radial), Some(heading)) = (centre.try_normalize(), drift.try_normalize()) else {
            continue;
        };
        let position = centre + radial * separation;
        let radius = position.length();
        let velocity = heading * sim::sqrt(MU_SUN / radius);
        let ratio = MU_EARTH / (separation * separation) / (MU_SUN / (radius * radius));

        let truth = checkpoints(Step::Rk4, (position, velocity), 480.0, &MARKS, planet);
        let floor = checkpoints(Step::Rk4, (position, velocity), 960.0, &MARKS, planet)
            .iter()
            .zip(&truth)
            .map(|(a, b)| (*a - *b).length())
            .fold(0.0f64, f64::max);
        let worst = |f: &dyn Fn(f64) -> Option<DVec3>| {
            MARKS
                .iter()
                .zip(&truth)
                .map(|(&mark, &want)| f(mark).map_or(f64::NAN, |had| (had - want).length()))
                .fold(0.0f64, f64::max)
        };
        let star = Orbit::from_state(position, velocity, MU_SUN);
        let relative = (position - centre, velocity - drift);
        let about_star = worst(&|mark| star.map(|o| o.state_at(mark).0));
        let about_planet = worst(&|mark| planet_conic(relative, planet, mark));
        let bubble = checkpoints(
            Step::Leapfrog,
            (position, velocity),
            TICK_HZ,
            &MARKS,
            planet,
        );
        let integrated = bubble
            .iter()
            .zip(&truth)
            .map(|(&had, &want)| (had - want).length())
            .fold(0.0f64, f64::max);
        println!(
            "  {separation:<12.1e}{ratio:>12.1e}{about_star:>16.2e}\
             {about_planet:>16.2e}{integrated:>14.2e}{floor:>12.1e}"
        );
    }
    println!(
        "
  worst of the three marks, so a row is the whole hour and not one instant; the
  planet's Hill radius is 1.5e9 m, and the two analytic columns cross just past
  it — which is the classical answer being right about the question it was asked"
    );
}

/// The acceleration ratio §6 M38 item 8 put the regime condition on, at a body's
/// own state — the quantity the condition reads, evaluated here at whatever
/// cadence a warp stride leaves it.
fn accel_ratio(body: DVec3, planet_at: DVec3) -> f64 {
    let separation = (body - planet_at).length_squared();
    let radius = body.length_squared();
    (MU_EARTH / separation) / (MU_SUN / radius)
}

/// The first root of `ratio - threshold` in `[lo, hi]`, bisected. The ratio is
/// smooth in `t`, so bisection converges on the crossing rather than on a
/// sampling artefact — which is the whole distinction this table is about.
fn bisect(body: Orbit, planet: Orbit, threshold: f64, lo: f64, hi: f64) -> f64 {
    let (mut lo, mut hi) = (lo, hi);
    for _ in 0..80 {
        let mid = (lo + hi) * 0.5;
        let inside = accel_ratio(body.state_at(mid).0, planet.state_at(mid).0) >= threshold;
        if inside { hi = mid } else { lo = mid }
    }
    (lo + hi) * 0.5
}

/// Table 10 — the hazard a stride creates for the *condition* rather than for
/// the physics: item 8's regime test is a function of state, so a warp factor
/// evaluates it every `k` ticks instead of every one, and a band crossed in
/// fewer than `k` ticks is crossed invisibly.
///
/// The two directions are residence and stride. A tighter threshold puts the
/// boundary further out, which makes the band wider in time and the sampling
/// safer — and buys that safety with bubble seconds, which are the seconds warp
/// may not touch. Both columns are here so the trade is one table.
fn table_sampling() {
    let planet = Orbit {
        semi_major: 1.495_978_707e11,
        eccentricity: 0.0,
        inclination: 0.0,
        ascending_node: 0.0,
        argument_of_periapsis: 0.0,
        mean_anomaly: 0.0,
        mu: MU_SUN,
    };
    // An encounter rather than a static separation: closest approach at
    // `ENCOUNTER`, 1e8 m off along the planet's own track, 3 km/s of relative
    // speed across it. Late enough that even the widest band opens after t = 0.
    const ENCOUNTER: f64 = 5.0e7;
    const SCAN: f64 = 200.0;
    let (centre, drift) = planet.state_at(ENCOUNTER);
    let (Some(radial), Some(heading)) = (centre.try_normalize(), drift.try_normalize()) else {
        return;
    };
    let at_encounter = (centre + heading * 1.0e8, drift + radial * 3.0e3);
    let Some(mut body) = Orbit::from_state(at_encounter.0, at_encounter.1, MU_SUN) else {
        return;
    };
    // `from_state` puts the epoch at the state it was handed, so the body's
    // clock would start at closest approach while the planet's starts at zero —
    // two propagators on two clocks, which reads as a body that never meets
    // anything. Rebased so both are on the sim's one clock, which is the same
    // property the regime condition needs and the reason it is worth a line.
    body.mean_anomaly -= body.mean_motion() * ENCOUNTER;
    println!(
        "

the sampling — item 8's condition read every k ticks, one encounter
"
    );
    println!(
        "  {:<12}{:>12}{:>16}{:>14}{:>14}{:>16}",
        "threshold",
        "boundary, m",
        "entry, s before",
        "residence, s",
        "band, ticks",
        "k for 8 samples"
    );
    for threshold in [2.0e-5, 1.0e-3, 1.0e-2, 1.0e-1, 1.0] {
        // The boundary as a separation, for the reader: the ratio is what the
        // condition reads, and a distance is what a level author would picture.
        let boundary = sim::sqrt(MU_EARTH / (threshold * MU_SUN)) * centre.length();
        let inside = |t: f64| accel_ratio(body.state_at(t).0, planet.state_at(t).0) >= threshold;
        // Coarse scan for the sign changes, then bisect each — the residence is
        // measured off the trajectory the body actually flies and not off
        // `2d/v`, which assumes a straight line through a two-body encounter.
        // Scanned from before t = 0, and the pair that *brackets closest
        // approach* is the one taken. At the widest threshold the band already
        // contains the scan's start, so the first edge found is an exit — read
        // as an entry it reports this encounter's residence as the gap until
        // the body's next pass, which is a plausible-looking number for the
        // wrong event.
        let mut edges: Vec<(f64, bool)> = Vec::new();
        let mut t = -ENCOUNTER;
        let mut was = inside(t);
        while t <= ENCOUNTER * 2.0 {
            t += SCAN;
            let now = inside(t);
            if now != was {
                edges.push((bisect(body, planet, threshold, t - SCAN, t), now));
                was = now;
            }
        }
        let entry = edges
            .iter()
            .filter(|(at, entering)| *entering && *at <= ENCOUNTER)
            .map(|(at, _)| *at)
            .fold(f64::NAN, f64::max);
        let exit = edges
            .iter()
            .filter(|(at, entering)| !*entering && *at >= ENCOUNTER)
            .map(|(at, _)| *at)
            .fold(f64::NAN, f64::min);
        let residence = exit - entry;
        if !residence.is_finite() {
            // Not a failure to measure: a threshold whose band has no edge
            // inside ±5e7 s is one the body never crosses, which is a stronger
            // statement about it than any number in the remaining columns.
            println!(
                "  {threshold:<12.0e}{boundary:>12.1e}{:>16}{:>14}{:>14}{:>16}",
                "no edge", "> window", "> window", "unbounded"
            );
            continue;
        }
        let ticks = residence * TICK_HZ;
        println!(
            "  {threshold:<12.0e}{boundary:>12.1e}{:>16.2e}{residence:>14.2e}\
             {ticks:>14.1e}{:>16.1e}",
            ENCOUNTER - entry,
            ticks / 8.0
        );
    }
    println!(
        "
  a 259-day transfer is 1.34e9 ticks, so the coast wants k near 1e5 to fit in a
  few minutes. the last column never falls under it — 1.2e6 at the loosest
  threshold measured — so the sampling hazard this table was built to price is
  the one constraint that does not bind, by an order of magnitude at the worst
  row and three at item 8's. the residence column is the one that does: warp is
  refused while anything is integrated, and 1.6e5 s of it is 44 hours a player
  spends at 1x. boundary, m assumes r = 1 AU; item 8's 1e11 row sat further out"
    );
}

pub fn run(_args: &[String]) -> anyhow::Result<()> {
    table_solve();
    table_regimes();
    table_budget();
    table_cost();
    table_round_trip();
    table_boundary();
    table_transition();
    table_stride();
    table_second_conic();
    table_sampling();
    Ok(())
}
