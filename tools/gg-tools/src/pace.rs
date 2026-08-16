//! `gg-tools pace` — what the display rate does to a turn the hand made at a
//! constant speed.
//!
//! The complaint this answers: mouse-look judders, "world sided, like the
//! objects themselves are moving differently when I am moving the mouse AND
//! moving my body at the same time" — on a 240 Hz panel.
//!
//! A hand that turns at a constant rate should paint a constant angle onto every
//! display frame. So **the measurement is the angle each frame adds to the
//! picture**, and no reference renderer is needed to say what it should be: the
//! hand's rate divided by the display's. Every departure from that was invented
//! somewhere between the device and the blend.
//!
//! Nothing here is simulated except the two clocks and the hand. The tick clock
//! is `gg_core::TickClock`, the accumulator and its sharing are `gg_input::Input`,
//! and the frame is composed by `gg_extract::blend_eye` against a pose captured
//! at the top of a tick — the same three pieces `gg_runtime::App` puts in that
//! order, which is why a number this prints is a number the shell would produce.
//!
//! Two figures, failing in opposite directions, on the frame-to-frame angle as a
//! fraction of what it should have been:
//!
//! - **stall** — the smallest. A frame that added nothing reads `0.00`.
//! - **lurch** — the largest. A frame that added four frames' worth reads `4.00`.
//!
//! A pipeline that never invented anything prints `1.00` twice, and one number
//! alone cannot say so: a blend that froze the picture has a perfect `lurch`, and
//! one that showed each tick whole has a perfect `stall`.
//!
//! The third figure is where it came from. **tick** is the same spread over the
//! angle each *tick* turned — the sim's own unevenness, before any blend saw it.
//! A leg with an even `tick` and an uneven frame-to-frame spread is the blend's
//! fault; the reverse is the accumulator's.
//!
//! **age** (§6 M56) is the fourth, and it is the one an even pipeline can still
//! fail: how far *behind* the hand the picture is, in milliseconds, taken
//! against the travel the device has actually reported by the moment the frame
//! landed. Smoothness and freshness are independent — a blend between two ticks
//! is perfectly even and is up to two tick periods stale, which at 60 Hz is
//! 33 ms of a turn the hand finished making. Both tables print it, and the
//! second one is what a latched view does to it.

use core::time::Duration;

use gg_core::{Pace, TickClock};
use gg_ecs::World;
use gg_ecs::boundary::{Eye, Look};
use gg_extract::{Extracted, Latch};
use gg_input::{ActionMap, AxisId, Input};
use gg_math::sim;

/// Demo 12's bindings for the camera — `MouseX`/`MouseY` are raw device counts,
/// which is the axis the complaint is about (§4.9). Both, though the hand here
/// only turns in x: the latch reads whatever axes a `Look` names, and a map
/// holding one of them would grade a protocol half-wired.
const MAP: &str = "[game.axes]\naim_x = [\"MouseX\"]\naim_y = [\"MouseY\"]\n";

/// The axes `MAP` declares, in its order.
const AIM_X: AxisId = AxisId::new(0);
const AIM_Y: AxisId = AxisId::new(1);

/// The sim rate, which is not a variable here: every leg is the same 60 Hz sim
/// under a different panel, because the panel is what changed.
const SIM_HZ: u32 = 60;

/// The hand: device counts per second, held exactly constant for the whole run.
/// A brisk turn — about a radian a second at [`LOOK_PER_COUNT`] — and large
/// enough that the 1/1024 fixed point (§4.7) is not what any of this measures.
const COUNTS_PER_SEC: f64 = 2000.0;

/// Demo 12's `look_per_count` at its default sensitivity, verbatim. It scales
/// every angle here equally and so cancels out of every ratio printed — it is
/// present so that `yaw` stays in the range a real session keeps it in, where an
/// `f32` has the bits to show a 240 Hz frame's share of a turn.
const LOOK_PER_COUNT: f32 = 0.0005;

/// The mouse's own report rate. 1 kHz is what a gaming mouse does, and it
/// matters only in that motion arrives *between* frame polls rather than on
/// them, so a frame's share of the hand is set by when the frame landed.
const MOUSE_HZ: f64 = 1000.0;

/// The second rate the latched table is swept over (§6 M56). 8 kHz is what the
/// current generation of the same hardware does, and it is here to **attribute**
/// a residual rather than to recommend a mouse: latching shows the hand closely
/// enough that the device's own polling becomes the frame-to-frame step, so a
/// leg whose `stall`/`lurch` tightens when only this changes has named the
/// quantizer as the mouse's and not the engine's.
const FAST_MOUSE_HZ: f64 = 8000.0;

/// Seconds each leg runs. Long enough that the frame and tick clocks drift
/// through every phase relationship they have — the rare alignment is the whole
/// question, so a short leg would report the common case and call it the only
/// one.
const SECONDS: f64 = 20.0;

/// Frames skipped before anything is counted: the first tick has no pose to
/// blend away from, and its frames would be charged to the panel.
const WARMUP: usize = 240;

/// Panels, in the order a desk grows through them. 60 is where this engine has
/// been developed and is the control — a leg that judders there is not about
/// the panel at all.
const PANELS: [f64; 6] = [60.0, 120.0, 144.0, 165.0, 240.0, 360.0];

/// Frame-time noise, peak-to-peak microseconds. Zero is the vsync a datasheet
/// describes; the other is the one a scheduler delivers, and the gap between
/// them is how much of the answer is the panel and how much is the machine.
const NOISE_US: [f64; 2] = [0.0, 400.0];

pub fn run(args: &[String]) -> anyhow::Result<()> {
    if let [arg, ..] = args {
        anyhow::bail!("unknown flag {arg:?} — pace takes none");
    }
    println!("gg-tools pace: a {SIM_HZ} Hz sim under panels that are not it");
    println!(
        "  the hand turns at {COUNTS_PER_SEC} counts/s, reporting at {MOUSE_HZ} Hz, for \
         {SECONDS} s — it never changes speed"
    );
    println!("  stall/lurch: the smallest and largest angle a frame added, over what it owed");
    println!("  tick: the same spread over the angle each sim tick turned");
    println!("  whole: the same run with a tick spending the accumulator whole — the control");
    println!("  age: how far behind the reported hand the picture is, mean and worst, ms");
    println!();

    println!("the tick, interpolated — where §6 M21 left it (`r.late_latch 0`)");
    println!(
        "  panel  noise  frames  ticks | stall  lurch | tick lo  tick hi | whole  whole |   age  worst"
    );
    for noise in NOISE_US {
        for hz in PANELS {
            let leg = measure(hz, noise, Spend::Covered, Latched::No, MOUSE_HZ)?;
            let whole = measure(hz, noise, Spend::Whole, Latched::No, MOUSE_HZ)?;
            println!(
                "  {hz:5.0}  {noise:4.0}us  {:6}  {:5} | {:5.2}  {:5.2} | {:7.2} {:8.2} | \
                 {:5.2}  {:5.2} | {:5.1}  {:5.1}",
                leg.frames,
                leg.ticks,
                leg.stall,
                leg.lurch,
                leg.tick_lo,
                leg.tick_hi,
                whole.stall,
                whole.lurch,
                leg.age,
                leg.worst
            );
        }
        println!();
    }

    println!("the hand, latched — §6 M56 (`r.late_latch 1`, the default)");
    println!("  the two poll rates are the attribution: a latched view steps by the device's");
    println!("  own report, so a column that tightens on the mouse alone was never the engine's");
    println!("                              |     1 kHz mouse     |     8 kHz mouse");
    println!("  panel  noise  frames  ticks | stall  lurch    age | stall  lurch    age");
    for noise in NOISE_US {
        for hz in PANELS {
            let leg = measure(hz, noise, Spend::Covered, Latched::Yes, MOUSE_HZ)?;
            let fast = measure(hz, noise, Spend::Covered, Latched::Yes, FAST_MOUSE_HZ)?;
            println!(
                "  {hz:5.0}  {noise:4.0}us  {:6}  {:5} | {:5.2}  {:5.2}  {:5.1} | {:5.2}  {:5.2}  {:5.1}",
                leg.frames,
                leg.ticks,
                leg.stall,
                leg.lurch,
                leg.age,
                fast.stall,
                fast.lurch,
                fast.age
            );
        }
        println!();
    }
    Ok(())
}

/// How a tick decides what part of the accumulator is its own.
#[derive(Clone, Copy, PartialEq)]
enum Spend {
    /// Whatever is in hand, whenever the tick comes due — the shell as it stood
    /// before any of this, and still what a host with no clock to report gets.
    Whole,
    /// One tick period's worth of it, `gg_core::Due::covered` being how much time
    /// the accumulator spans. The shipping path.
    Covered,
}

/// Whether the frame shows the hand's unspent travel (§6 M56) — the `Look` the
/// world carries is the same either way, so what this switches is exactly what
/// `r.late_latch` switches in the shell.
#[derive(Clone, Copy, PartialEq)]
enum Latched {
    No,
    Yes,
}

/// One panel's answer.
struct Leg {
    frames: usize,
    ticks: usize,
    stall: f64,
    lurch: f64,
    tick_lo: f64,
    tick_hi: f64,
    age: f64,
    worst: f64,
}

/// Drive the real clock, the real accumulator and the real extract stage for
/// `SECONDS` at `hz`, and reduce the picture to the numbers above.
///
/// The eye goes through a real `World` and a real [`Extracted`] rather than
/// through [`gg_extract::blend_eye`] by hand, which is what lets the latched leg
/// grade the shipped path: `capture_eye` at the top of a tick, `interpolate`
/// once a frame, `eye(current, latch)` to compose — `gg_runtime::App`'s three
/// calls in `App`'s order, with nothing restated here but the game's own look
/// system.
fn measure(
    hz: f64,
    noise_us: f64,
    spend: Spend,
    latched: Latched,
    mouse_hz: f64,
) -> anyhow::Result<Leg> {
    let mut clock = TickClock::new(SIM_HZ, Pace::Realtime);
    let mut input = Input::new(map());
    // A map's bindings only exist inside a context, and nothing is bound until
    // one is pushed — the shell does this at startup and a leg without it would
    // measure a mouse nothing listens to.
    input.push_named("game");
    let mut noise = Noise::new();

    // Demo 12's look system, which is one line: `yaw = wrap(yaw - x *
    // per_count)`, and nothing else reads the clock.
    let mut yaw = 0.0f32;
    // Demo 12's declaration of that one line, which is what the host latches
    // through. No pitch limit: the hand here never leaves the horizon, and a
    // clamp that never fires would be a column of zeros pretending to be a test.
    let look = Look::fly(
        AIM_X.index() as u32,
        AIM_Y.index() as u32,
        LOOK_PER_COUNT,
        0.0,
    );
    let mut world = World::new();
    let eye = world.spawn();
    world.insert(eye, Eye::at(sim::DVec3::ZERO, 0.0, 0.0))?;
    world.insert(eye, look)?;
    let mut extracted = Extracted::default();

    // Wall time in nanoseconds, and the mouse report the run has delivered up to
    // it. Integer so a 20 s leg does not lose its last frame to a float's ulp.
    let mut wall = 0u64;
    let mut reports = 0u64;
    let period = 1.0e9 / hz;

    let mut shown: Vec<f64> = Vec::new();
    let mut turned: Vec<f64> = Vec::new();
    let mut ages: Vec<f64> = Vec::new();
    let mut last_shown = 0.0f64;
    let mut last_yaw = 0.0f64;
    let mut frames = 0usize;
    let mut ticks = 0usize;

    while (wall as f64) < SECONDS * 1.0e9 {
        let elapsed = (period + noise.next() * noise_us).max(1.0) as u64;
        wall += elapsed;

        // Every report whose timestamp has passed, delivered at the poll that
        // followed it — which is the one thing a frame rate changes about the
        // hand, and the reason this is not just `motion(rate * elapsed)`.
        let due_reports = (wall as f64 * mouse_hz / 1.0e9) as u64;
        let arrived = due_reports.saturating_sub(reports);
        reports = due_reports;
        input.motion((COUNTS_PER_SEC / mouse_hz) as f32 * arrived as f32, 0.0);

        let due = clock.advance(Duration::from_nanos(elapsed));
        // `Whole` never tells the accumulator anything, which leaves it at one
        // period and every tick taking everything in hand — the documented
        // resting behaviour, and exactly what the shell did before this was
        // measured. So the control column is a real supported path, not a
        // reimplementation of one.
        if spend == Spend::Covered {
            input.frame_covered(due.covered);
        }
        for _ in 0..due.count {
            // At the top of the tick, before the tick writes — `App::sim_tick`.
            extracted.capture_eye(&world)?;
            input.tick();
            yaw = wrap(yaw - input.axis(AIM_X) * LOOK_PER_COUNT);
            world.insert(eye, Eye::at(sim::DVec3::ZERO, yaw, 0.0))?;
            ticks += 1;
            if frames >= WARMUP {
                turned.push(step(f64::from(yaw), last_yaw));
            }
            last_yaw = f64::from(yaw);
        }

        extracted.interpolate(clock.alpha());
        let latch = match latched {
            Latched::Yes => Some(Latch::of(
                look,
                (input.pending(AIM_X), input.pending(AIM_Y)),
                (input.spent(AIM_X), input.spent(AIM_Y)),
            )),
            Latched::No => None,
        };
        let current = Eye::of(&world)?;
        let composed = extracted.eye(current, latch);
        let rendered = f64::from(composed.yaw);
        if frames >= WARMUP {
            shown.push(step(rendered, last_shown));
            // Where the *reported* hand is: travel the device has delivered, not
            // travel it has made. A frame cannot show a count that has not
            // arrived, and charging it for one would price the mouse's own rate
            // as engine lag.
            let hand = -(reports as f64) * (COUNTS_PER_SEC / mouse_hz) * f64::from(LOOK_PER_COUNT);
            // The short way, which needs no unwrapping to be exact: the lag
            // being measured is tens of milliseconds of a turn — under 0.05 rad
            // here — so the hand and the picture are never a revolution apart
            // and `step` reduces both sides' wrapping away in one subtraction.
            ages.push(step(hand, rendered) / RATE * 1.0e3);
        }
        last_shown = rendered;
        frames += 1;
    }

    // What each frame owed: the hand's rate over the panel's. Taken from the
    // hand rather than from the mean of what was shown, so a leg that lost or
    // invented angle overall cannot hide it by normalising against itself.
    let owed_frame = RATE / hz;
    let owed_tick = RATE / f64::from(SIM_HZ);
    let (stall, lurch) = spread(&shown, owed_frame);
    let (tick_lo, tick_hi) = spread(&turned, owed_tick);
    let (age, worst) = mean_and_worst(&ages);
    Ok(Leg {
        frames,
        ticks,
        stall,
        lurch,
        tick_lo,
        tick_hi,
        age,
        worst,
    })
}

/// The hand's angular rate, radians a second, signed the way demo 12's look
/// system turns: `yaw -= x * per_count`.
const RATE: f64 = -COUNTS_PER_SEC * LOOK_PER_COUNT as f64;

/// Mean and largest of `ages`, which are milliseconds behind the hand. Both,
/// because a mean alone forgives a pipeline that is fresh three frames in four
/// and a whole tick stale on the fourth — which is the shape of the thing being
/// measured.
fn mean_and_worst(ages: &[f64]) -> (f64, f64) {
    match ages.is_empty() {
        true => (0.0, 0.0),
        false => (
            ages.iter().sum::<f64>() / ages.len() as f64,
            ages.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        ),
    }
}

/// The angle from `previous` to `current`, taken the short way — a turn that
/// crossed ±π is a small step, not a full revolution the other way.
fn step(current: f64, previous: f64) -> f64 {
    let tau = core::f64::consts::TAU;
    let delta = (current - previous) % tau;
    match delta {
        d if d > tau / 2.0 => d - tau,
        d if d < -tau / 2.0 => d + tau,
        d => d,
    }
}

/// Demo 12's `wrap`: keep the angle in ±π, because an unbounded yaw loses the
/// precision this whole measurement is made of.
fn wrap(yaw: f32) -> f32 {
    let tau = core::f32::consts::TAU;
    let yaw = yaw % tau;
    match yaw {
        y if y > core::f32::consts::PI => y - tau,
        y if y < -core::f32::consts::PI => y + tau,
        y => y,
    }
}

/// The smallest and largest of `samples` as a fraction of `owed`, which is
/// signed — a step against the turn reads negative rather than large.
fn spread(samples: &[f64], owed: f64) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &sample in samples {
        let ratio = sample / owed;
        lo = lo.min(ratio);
        hi = hi.max(ratio);
    }
    match samples.is_empty() {
        true => (0.0, 0.0),
        false => (lo, hi),
    }
}

/// The map every leg runs under. Built rather than read from `demos/12-shooter`:
/// what this measures is the accumulator, and a leg that failed because someone
/// rebound a demo would be a confusing way to learn it.
fn map() -> ActionMap {
    // A literal this file owns, parsed by the real parser — the alternative is
    // constructing an `ActionMap` by hand, which no caller outside the crate can
    // do and no caller should learn to.
    match ActionMap::parse(MAP, &[], &["aim_x", "aim_y"]) {
        Ok(map) => map,
        Err(error) => panic!("the instrument's own map does not parse: {error}"),
    }
}

/// Frame-time noise in `-0.5..0.5`, from a fixed seed.
///
/// Deterministic on purpose: a jitter measurement that changes run to run cannot
/// be compared against the last one, and the question is whether a *pattern*
/// exists rather than what one sample of it looked like.
struct Noise(u64);

impl Noise {
    fn new() -> Self {
        Noise(0x2545_f491_4f6c_dd1d)
    }

    fn next(&mut self) -> f64 {
        // xorshift64*, whose only requirement here is that consecutive frames
        // are not correlated — a periodic frame time would beat against the tick
        // clock and report a resonance the desk does not have.
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}
