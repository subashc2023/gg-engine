//! `transfer` — where a departure lands, and where the target has to be for it
//! to land on something (§6 M38 item 14).
//!
//! Demo 13's mission is a crossing between two planets, and `session.rs` flies
//! it closed-loop on everything a pilot can *see*: the burn ends when the conic
//! reaches out far enough, the cruise ends when the game hands the conic over,
//! the capture burn ends when it closes. Two numbers are not visible that way —
//! when to light the engine and how long to hold it — because a pilot cannot
//! measure where a burn it has not made yet will put it. This is what measures
//! them.
//!
//! # Why a tool rather than a search
//!
//! The obvious answer is a grid over the two knobs, scoring each cell by how
//! close the ship comes. That search was run first and it is what named the
//! problem: over a full turn of the parking orbit the best cell missed by
//! 1.6e10 m, which is 175 times the target's grip, because the target's *phase*
//! was authored by hand and no departure can fix a target that is not there.
//!
//! What makes the answer cheap is that **nothing in this world perturbs
//! anything**: planets are on rails, the ship does not pull on them, and the
//! transfer a plan buys is therefore independent of where the target is. So the
//! plan chooses the conic and the target's mean anomaly chooses the meeting,
//! and the two can be solved in that order, once, in closed form:
//!
//! 1. fly the plan and read the heliocentric conic the departure produced;
//! 2. find where that conic crosses the target's — a radius difference with a
//!    sign change, bisected;
//! 3. read the ship's arrival time there off its own mean anomaly;
//! 4. read the mean anomaly the target must have had at epoch 0 to be there
//!    then.
//!
//! # A bullseye is a crater
//!
//! Step 4 aims at the target's *centre*, and the first run of it flew straight
//! into the planet — periapsis 4.1 km above the centre of a body 3389 km in
//! radius. So the last column aims **off** by the impact parameter that puts the
//! incoming hyperbola's periapsis at a chosen number of radii:
//! `b = r_p √(1 + 2μ/(r_p v∞²))`, converted into a shift along the target's own
//! track by the sine of the crossing angle, because moving the target along its
//! orbit is the only knob and a shallow crossing converts less of it. Three
//! radii comes out at 6.6e-5 rad of mean anomaly and lands within 1 % of the
//! periapsis it asked for.
//!
//! Both signs are printed: the ship can pass ahead of the target or behind it,
//! and which one is wanted is a question about the capture burn rather than
//! about the crossing.
//!
//! # What it does not do
//!
//! It does not gate. The numbers it prints are read into `session::FLOWN` and
//! `demo_13_orbit::OCHRE::mean_anomaly` by hand, and what *proves* them is demo
//! 13's own test flying the mission to a capture — a threshold that hardened
//! belongs in `xtask`, and this stays the microscope (CLAUDE.md).

use anyhow::Context as _;
use demo_13_orbit::session::{self, Entry, Plan};
use demo_13_orbit::{MU_STAR, OCHRE, Planet};
use gg_ecs::boundary::{AbiInfo, ComponentsTable, HostApiV1, SystemsTable};
use gg_math::sim::{self, DVec3, Orbit};

// The `gg_game!` exports of the demo, linked into this binary from its rlib.
// Taken as a crate for `cull`'s reason: the mission this measures has to be the
// mission the demo flies, and `default-features = false` keeps the game ABI's
// own exports out of a tool.
unsafe extern "C" {
    fn gg_game_abi() -> AbiInfo;
    fn gg_game_init(api: *const HostApiV1);
    fn gg_game_components() -> ComponentsTable;
    fn gg_game_systems() -> SystemsTable;
}

fn entry() -> Entry {
    Entry {
        abi: gg_game_abi,
        init: gg_game_init,
        components: gg_game_components,
        systems: gg_game_systems,
    }
}

const TAU: f64 = core::f64::consts::TAU;

/// Samples of the true longitude the crossing search brackets over. Two conics
/// cross at most twice, so this only has to be fine enough not to step over a
/// pair of roots that are close together.
const BRACKETS: usize = 2_000;

/// Bisection steps per bracket — enough to reach the double's floor, and the
/// whole search costs nothing beside one flight.
const BISECT: usize = 80;

pub fn run(args: &[String]) -> anyhow::Result<()> {
    let mut plan = session::FLOWN;
    let mut radii = 3.0_f64;
    let mut sweep = None;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = || -> anyhow::Result<String> {
            it.next()
                .cloned()
                .with_context(|| format!("{arg} takes a value"))
        };
        match arg.as_str() {
            "--wait" => plan.wait = value()?.parse()?,
            "--burn" => plan.burn = value()?.parse()?,
            "--radii" => radii = value()?.parse()?,
            "--sweep" => {
                let spec = value()?;
                let parts: Vec<&str> = spec.split(':').collect();
                anyhow::ensure!(parts.len() == 3, "--sweep takes from:to:by, got {spec}");
                sweep = Some((
                    parts[0].parse::<u32>()?,
                    parts[1].parse::<u32>()?,
                    parts[2].parse::<u32>()?,
                ));
            }
            other => anyhow::bail!(
                "transfer takes --wait N --burn N --radii R --sweep from:to:by, got {other}"
            ),
        }
    }

    println!(
        "gg-tools transfer — demo 13's crossing, and the phase it needs (§6 M38 item 14)\n\
         the plan buys the conic, the target's mean anomaly buys the meeting; nothing here \
         perturbs anything, so they solve in that order\n"
    );

    let entry = entry();
    let flight = session::fly(&entry, plan).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!(
        "plan  wait {} at warp {} ({} sim ticks)  burn {} ticks ({:.0} m/s)  escape warp {}",
        plan.wait,
        demo_13_orbit::WARPS[plan.wait_step as usize],
        u64::from(plan.wait) * demo_13_orbit::WARPS[plan.wait_step as usize],
        plan.burn,
        f64::from(plan.burn) * demo_13_orbit::THRUST / 60.0,
        demo_13_orbit::WARPS[plan.escape_step as usize],
    );
    println!(
        "flew  {} in {} ticks, spent {:.0} m/s of {:.0}, closest {:.4e} m, periapsis {}\n",
        outcome_name(flight.outcome),
        flight.frames.len(),
        flight.spent,
        demo_13_orbit::TANK,
        flight.approach,
        flight
            .periapsis
            .map_or_else(|| "none".to_owned(), |p| format!("{p:.4e} m")),
    );

    let Some(rails) = flight.transfer else {
        anyhow::bail!("this plan never reached the star's regime — there is no transfer to aim");
    };
    let ship = rails.orbit;
    println!("the transfer the departure bought\n");
    println!(
        "  a {:.5e} m   e {:.5}   i {:.2e}   peri {:.5e}   apo {:.5e}   handed over at epoch {}",
        ship.semi_major,
        ship.eccentricity,
        ship.inclination,
        ship.semi_major * (1.0 - ship.eccentricity),
        ship.semi_major * (1.0 + ship.eccentricity),
        rails.since,
    );

    let target = orbit_of(&OCHRE);
    println!(
        "  target: peri {:.5e}   apo {:.5e}   grip ~{:.2e} m (equal pull over √{HANDOVER})\n",
        target.semi_major * (1.0 - target.eccentricity),
        target.semi_major * (1.0 + target.eccentricity),
        grip(),
    );

    println!("where the two conics cross, and the phase each crossing asks for\n");
    println!(
        "  {:>9}  {:>11}  {:>11}  {:>9}  {:>9}  {:>10}  {:>10}",
        "longitude", "radius", "arrive (s)", "v∞ (m/s)", "sine", "bullseye", "aimed"
    );
    let crossings = crossings(&ship, &target);
    anyhow::ensure!(
        !crossings.is_empty(),
        "this transfer never reaches the target's orbit — a longer burn is what fixes that"
    );
    for theta in crossings {
        let ahead = wrap(mean_at(&ship, theta) - ship.mean_anomaly);
        let seconds = ahead / ship.mean_motion();
        let epoch = rails.since as f64 + seconds * f64::from(HZ);
        let bullseye =
            wrap(mean_at(&target, theta) - target.mean_motion() * (epoch / f64::from(HZ)));

        let ship_velocity = ship.state_at(seconds).1;
        let aimed = Orbit {
            mean_anomaly: bullseye,
            ..target
        };
        let (target_at, target_velocity) = aimed.state_at(epoch / f64::from(HZ));
        let relative = ship_velocity - target_velocity;
        let v_inf = relative.length();
        let delta = aim_off(radii, v_inf, target_at, target_velocity, relative, theta);
        println!(
            "  {theta:>9.5}  {:>11.4e}  {seconds:>11.4e}  {v_inf:>9.1}  {:>9.4}  {bullseye:>10.6}  \
             {:>10.6}",
            radius_at(&ship, theta),
            sine(target_velocity, relative),
            wrap(bullseye + delta),
        );
        println!(
            "  {:>9}  aim-off {delta:.3e} rad puts the incoming periapsis at {radii} radii \
             ({:.3e} m); the other side is {:.6}",
            "",
            radii * OCHRE.radius,
            wrap(bullseye - delta),
        );
    }

    if let Some((from, to, by)) = sweep {
        println!("\nthe departure window — one turn of the parking orbit is 5828 s\n");
        println!(
            "  {:>7}  {:>12}  {:>11}  {:>11}  {:>9}  {:>7}",
            "wait", "outcome", "closest (m)", "periapsis", "spent", "ticks"
        );
        let mut wait = from;
        while wait < to {
            let leg = Plan { wait, ..plan };
            match session::fly(&entry, leg) {
                Ok(f) => println!(
                    "  {wait:>7}  {:>12}  {:>11.4e}  {:>11}  {:>9.0}  {:>7}",
                    outcome_name(f.outcome),
                    f.approach,
                    f.periapsis
                        .map_or_else(|| "—".to_owned(), |p| format!("{p:.4e}")),
                    f.spent,
                    f.frames.len(),
                ),
                Err(e) => println!("  {wait:>7}  {e}"),
            }
            wait += by.max(1);
        }
    }

    Ok(())
}

/// The rate the session is authored at, and the one every epoch here converts
/// through.
const HZ: u32 = 60;

/// The acceleration ratio demo 13 hands a conic over at.
const HANDOVER: f64 = 2.0;

/// How wide the target's grip is, in metres — where its pull is `HANDOVER`
/// times the star's. What the crossing has to land inside of.
fn grip() -> f64 {
    let ratio = sim::sqrt(OCHRE.mu / (MU_STAR * HANDOVER));
    OCHRE.semi_major * ratio / (1.0 + ratio)
}

fn outcome_name(outcome: u32) -> &'static str {
    match outcome {
        demo_13_orbit::CAPTURED => "captured",
        demo_13_orbit::CRASHED => "crashed",
        demo_13_orbit::STRANDED => "stranded",
        _ => "still flying",
    }
}

fn orbit_of(planet: &Planet) -> Orbit {
    Orbit {
        semi_major: planet.semi_major,
        eccentricity: planet.eccentricity,
        inclination: planet.inclination,
        ascending_node: planet.ascending_node,
        argument_of_periapsis: planet.argument_of_periapsis,
        mean_anomaly: planet.mean_anomaly,
        mu: MU_STAR,
    }
}

/// Longitude of periapsis. Valid while the conic is in the reference plane,
/// which is the case this whole file is about: the ship's only thrust is along
/// its velocity, so a departure from an in-plane parking orbit stays in-plane
/// and the target is authored that way to match.
fn peri_longitude(orbit: &Orbit) -> f64 {
    orbit.ascending_node + orbit.argument_of_periapsis
}

/// Radius at a true longitude.
fn radius_at(orbit: &Orbit, theta: f64) -> f64 {
    let p = orbit.semi_major * (1.0 - orbit.eccentricity * orbit.eccentricity);
    p / (1.0 + orbit.eccentricity * sim::cos(theta - peri_longitude(orbit)))
}

/// Mean anomaly at a true longitude — the `ν → E → M` direction, which is the
/// closed one.
fn mean_at(orbit: &Orbit, theta: f64) -> f64 {
    let nu = theta - peri_longitude(orbit);
    let e = orbit.eccentricity;
    let (sin_nu, cos_nu) = sim::sin_cos(nu);
    let anomaly = sim::atan2(sim::sqrt(1.0 - e * e) * sin_nu, e + cos_nu);
    anomaly - e * sim::sin(anomaly)
}

fn wrap(angle: f64) -> f64 {
    angle.rem_euclid(TAU)
}

/// Where the two conics cross, by sign change of the radius difference.
fn crossings(ship: &Orbit, target: &Orbit) -> Vec<f64> {
    let gap = |theta: f64| radius_at(ship, theta) - radius_at(target, theta);
    let mut found = Vec::new();
    for k in 0..BRACKETS {
        let (lo, hi) = (
            TAU * (k as f64) / (BRACKETS as f64),
            TAU * ((k + 1) as f64) / (BRACKETS as f64),
        );
        if gap(lo).signum() == gap(hi).signum() {
            continue;
        }
        let (mut lo, mut hi) = (lo, hi);
        for _ in 0..BISECT {
            let mid = 0.5 * (lo + hi);
            if gap(lo).signum() == gap(mid).signum() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        found.push(0.5 * (lo + hi));
    }
    found
}

/// How square the crossing is: the sine of the angle between the target's own
/// motion and the relative velocity. It is the conversion factor between "move
/// the target along its track" — the only knob — and "miss by this much".
fn sine(target_velocity: DVec3, relative: DVec3) -> f64 {
    let unit = |v: DVec3| v.try_normalize().unwrap_or(DVec3::ZERO);
    unit(target_velocity).cross(unit(relative)).length()
}

/// Mean anomaly to add to a bullseye so the incoming hyperbola's periapsis
/// lands `radii` target radii up.
fn aim_off(
    radii: f64,
    v_inf: f64,
    target_at: DVec3,
    target_velocity: DVec3,
    relative: DVec3,
    theta: f64,
) -> f64 {
    let peri = radii * OCHRE.radius;
    // The impact parameter a periapsis of `peri` corresponds to. Gravity does
    // the focusing, which is why `b` is larger than the periapsis and by how
    // much depends on how fast the ship arrives.
    let b = peri * sim::sqrt(1.0 + 2.0 * OCHRE.mu / (peri * v_inf * v_inf));
    let shift = b / sine(target_velocity, relative).max(1.0e-6);
    // Arc to mean anomaly. `dν/dM` is not 1 off a circle, and the target's
    // eccentricity is 0.093.
    let e = OCHRE.eccentricity;
    let nu = theta - peri_longitude(&orbit_of(&OCHRE));
    let spread = 1.0 + e * sim::cos(nu);
    let dnu_dm = spread * spread / sim::powf(1.0 - e * e, 1.5);
    shift / (target_at.length() * dnu_dm)
}
