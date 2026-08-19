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
//!
//! # `--editor` (§6 M65)
//!
//! The same three figures over the **editor's** camera, which is a second eye
//! and not the same one seen from elsewhere: it is host state rather than a
//! world component, it is flown by [`gg_editor::Editor::tick`] rather than by a
//! game's look system, and the shell composes it through
//! [`gg_editor::Editor::eyes`] + [`gg_extract::blend_eye`] + a latch built from
//! [`gg_editor::Editor::look`] rather than through [`Extracted::eye`]. Every
//! one of those is a different line of code from the ones the table above
//! grades, so a green game leg says nothing whatever about it — which is the
//! gap a report of judder in the viewport, on a build whose game camera was
//! already smooth, walks straight into.
//!
//! Driven as the real editor for the reason the game leg drives a real
//! `Extracted`: `Editor::tick` once per sim tick with the look button held and
//! raw counts arriving between frames, then the shell's own frame composition.
//! The *order* is restated here — an instrument cannot link `gg-runtime`, which
//! links `gg-platform` — but none of the arithmetic is: `blend_eye` and
//! `latched` are the shipped functions, called with the shipped arguments.

use core::time::Duration;

use gg_core::{Pace, TickClock};
use gg_ecs::World;
use gg_ecs::boundary::{Eye, Look};
use gg_extract::{Extracted, Latch};
use gg_input::{ActionMap, AxisId, Input, MouseButton};
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
    match args {
        [] => {}
        [flag] if flag == "--editor" => return editor(),
        [arg, ..] => anyhow::bail!("unknown flag {arg:?} — pace takes --editor or nothing"),
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

// ------------------------------------------- §6 M65: the editor's camera ----

/// The surface the editor is laid out on. A real desk's, because
/// `metres_per_unit` — and so the pan, though this leg never pans — is read off
/// the viewport pane's rectangle, and a pane sized for nothing would be a
/// framing no session has.
const EDITOR_EXTENT: (u32, u32) = (1920, 1080);

/// `--editor`: the same question asked of the second camera (§6 M65).
///
/// Printed as one table rather than two, because the interesting comparison is
/// across the row: what the same panel does to this camera interpolated and
/// latched. The game's table above is two because it also has a `Spend`
/// control, and there is no such choice here — the editor's camera reads the
/// same accumulator the game's does, so `Spend::Covered` is not a variable it
/// can be run without.
fn editor() -> anyhow::Result<()> {
    println!("gg-tools pace --editor: the editor camera under panels that are not the sim");
    println!(
        "  the hand turns at {COUNTS_PER_SEC} counts/s, reporting at {MOUSE_HZ} Hz, for \
         {SECONDS} s, with the look button held throughout"
    );
    println!("  a second eye and a second composition (§6 M65): `Editor::tick` flies it, and the");
    println!("  shell blends `Editor::eyes` and latches `Editor::look` — none of which the");
    println!("  game's table grades, so a smooth shooter says nothing about this");
    println!();
    println!("                              |      interpolated      |        latched");
    println!(
        "  panel  noise  frames  ticks | stall  lurch    age | stall  lurch    age | tick lo  tick hi"
    );
    for noise in NOISE_US {
        for hz in PANELS {
            let interpolated = measure_editor(hz, noise, Latched::No)?;
            let leg = measure_editor(hz, noise, Latched::Yes)?;
            println!(
                "  {hz:5.0}  {noise:4.0}us  {:6}  {:5} | {:5.2}  {:5.2}  {:5.1} | {:5.2}  {:5.2}  \
                 {:5.1} | {:7.2} {:8.2}",
                leg.frames,
                leg.ticks,
                interpolated.stall,
                interpolated.lurch,
                interpolated.age,
                leg.stall,
                leg.lurch,
                leg.age,
                leg.tick_lo,
                leg.tick_hi
            );
        }
        println!();
    }
    quantum()?;
    cost()
}

/// What one mouse count is worth (§6 M65) — the column every ratio above is
/// blind to.
///
/// `stall` and `lurch` are fractions of what a frame *owed*, so the look rate
/// divides out of both: a camera turning ten times too fast reads a perfect
/// `1.00` twice, and `age` reads 0.0 because it is fresh — it is fresh and
/// even and stepping. What a hand sees when it moves *slowly* is not the frame
/// cadence at all, it is the smallest angle the device can report, and on a
/// 1920-pixel window that is a distance in pixels.
///
/// Under a pixel is invisible. Over about two, the picture visibly jumps
/// between two counts however slowly the hand moves, which reads as judder that
/// no amount of interpolation or latching touches — demo 12 found this from the
/// other end at §6 M37 and wrote it in its own header, and the editor's camera
/// was ten times coarser than the value that header rejects until §6 M65.
///
/// Graded against demo 12 rather than against a written-down threshold: that
/// camera's rate is the one the operator has confirmed feels right, and both
/// numbers here are read out of the shipped code rather than restated.
fn quantum() -> anyhow::Result<()> {
    let editor = editor_look_rate()?;
    let shooter = f64::from(demo_12_shooter::look_per_count(
        demo_12_shooter::SENS_DEFAULT,
    ));
    println!("what one mouse count is worth — the column a ratio cannot see");
    println!("  every figure above is a fraction of what a frame owed, so the rate cancels out:");
    println!("  a camera turning ten times too fast reads a perfect 1.00 and a fresh 0.0 ms");
    println!(
        "  px/count is across `r.fov`'s {:.0}-degree horizontal window {} pixels wide",
        HORIZONTAL_FOV.to_degrees(),
        QUANTUM_WIDTH
    );
    println!("  under 1 px is invisible; over about 2 the picture steps between two counts");
    println!();
    println!("  camera             rad/count  deg/count  px/count  counts/turn  cm @1600dpi");
    for (name, rate) in [
        ("editor", editor),
        ("demo 12 default", shooter),
        // The value this camera held from §6 M15.2 to M65, printed so the table
        // says what was wrong rather than only what is right — a row that is
        // merely absent is a fix nobody can check.
        ("editor before M65", 0.005),
    ] {
        let turn = core::f64::consts::TAU / rate;
        println!(
            "  {name:<18} {rate:9.6}  {:9.4}  {:8.2}  {:11.0}  {:11.2}",
            rate.to_degrees(),
            rate / HORIZONTAL_FOV * f64::from(QUANTUM_WIDTH),
            turn,
            turn / 1600.0 * 2.54
        );
    }
    println!();
    Ok(())
}

/// The window `px/count` is quoted across. 1080p, the extent every other
/// default in this tree is read at.
const QUANTUM_WIDTH: u32 = 1920;

/// `r.fov`'s 1.0 rad vertical over 16:9, which is what a horizontal step is
/// actually measured against. Written out rather than computed with an
/// `atan` — `gg_math::sim` is the only transcendental this tree may use and a
/// constant needs neither.
const HORIZONTAL_FOV: f64 = 1.5416;

/// The editor camera's radians per count, read off the `Look` it declares
/// rather than off a constant copied here — the rate is a private constant
/// times `d.editor_sensitivity`, and the protocol is where the two meet.
fn editor_look_rate() -> anyhow::Result<f64> {
    let (verbs, bindings) = gg_editor::host::open(&gg_ecs::boundary::Verbs {
        actions: &[],
        axes: &[],
    });
    let map = ActionMap::parse(&bindings, verbs.actions, verbs.axes)?;
    let mut input = Input::new(map);
    input.push_named("game");
    input.mouse_button(MouseButton::Right, true);
    let mut world = World::new();
    world.register::<Eye>()?;
    let eye = world.spawn();
    world.insert(eye, Eye::at(sim::DVec3::ZERO, 0.0, 0.0))?;
    let mut editor = gg_editor::Editor::new(None);
    // One tick with the button held is what makes the camera report a rate at
    // all — `Editor::look` is `None` whenever the operator is not turning it.
    input.tick();
    editor.tick(&mut world, &editor_ui_tick(), &editor_frame(0, &input));
    match editor.look(&input) {
        Some(look) => Ok(f64::from(-look.yaw_rate)),
        None => anyhow::bail!("the editor declared no look with its own button held"),
    }
}

/// What `Editor::tick` costs the frame that owes it (§6 M65).
///
/// The other half of the same complaint, and the half the table above cannot
/// see: those figures are the *composition*, which is arithmetic on two eyes
/// and free. The editor's panels are rebuilt inside a **sim tick**, so their
/// cost lands on one frame in four at 240 Hz and on none of the other three —
/// which is not a lower frame rate, it is a periodic one, and a hand moving at
/// a constant speed across a picture that hitches every fourth frame is exactly
/// the "tiny steps" a flat frame rate does not produce.
///
/// Wall clock, so it is worth exactly what the profile is worth: an
/// `opt-level = 1` instrument is not a shipping build and the header says so.
fn cost() -> anyhow::Result<()> {
    let (verbs, bindings) = gg_editor::host::open(&gg_ecs::boundary::Verbs {
        actions: &[],
        axes: &[],
    });
    let map = ActionMap::parse(&bindings, verbs.actions, verbs.axes)?;
    let mut input = Input::new(map);
    input.push_named("game");
    input.mouse_button(MouseButton::Right, true);
    let mut world = World::new();
    world.register::<Eye>()?;
    let eye = world.spawn();
    world.insert(eye, Eye::at(sim::DVec3::ZERO, 0.0, 0.0))?;
    let mut editor = gg_editor::Editor::new(None);

    let mut costs: Vec<f64> = Vec::new();
    for tick in 0..COST_TICKS {
        input.motion((COUNTS_PER_SEC / f64::from(SIM_HZ)) as f32, 0.0);
        input.tick();
        let at = std::time::Instant::now();
        editor.tick(&mut world, &editor_ui_tick(), &editor_frame(tick, &input));
        let spent = at.elapsed().as_secs_f64() * 1.0e3;
        // The first ticks place the layout and open the panes for the first
        // time; charging a steady-state budget for them would price a cost the
        // session pays once.
        if tick >= COST_WARMUP {
            costs.push(spent);
        }
    }
    costs.sort_by(f64::total_cmp);
    let at = |q: f64| costs[((costs.len() - 1) as f64 * q) as usize];
    let mean = costs.iter().sum::<f64>() / costs.len() as f64;
    println!("what `Editor::tick` costs the frame that owes it — {COST_TICKS} ticks");
    println!(
        "  a sim tick's cost lands on one frame in four at 240 Hz and on none of the other three,"
    );
    println!("  so this is a periodic hitch rather than a frame rate — and 4.17 ms is the budget");
    println!(
        "  profile: {}",
        match cfg!(debug_assertions) {
            true => "debug assertions ON — build with --release before believing a millisecond",
            false => "release",
        }
    );
    println!("    mean  {mean:6.3} ms");
    println!("     p50  {:6.3} ms", at(0.50));
    println!("     p95  {:6.3} ms", at(0.95));
    println!("     p99  {:6.3} ms", at(0.99));
    println!("   worst  {:6.3} ms", at(1.0));
    Ok(())
}

/// Ticks the cost pass runs — a hundred seconds of session at the sim rate,
/// long enough that a p99 is a hundred samples rather than three.
const COST_TICKS: u64 = 6000;

/// Ticks charged to opening the session rather than to running it.
const COST_WARMUP: u64 = 120;

/// [`measure`] against `gg_editor::Editor` instead of a game's look system.
///
/// The editor is driven, not modelled: `Editor::tick` once per sim tick with
/// the map the host appends and the button the map binds, then the frame
/// composed the way `gg_runtime::App::editor_eye` composes it. What that costs
/// is running every panel sixty times a second for the whole leg, which is
/// cheap and is also the point — a camera flown through the real tick is a
/// camera whose ordering against the rest of the editor is being graded too.
fn measure_editor(hz: f64, noise_us: f64, latched: Latched) -> anyhow::Result<Leg> {
    // The map the shell builds with the editor open over a game declaring
    // nothing, which is what makes the axis ids the host's rather than ours.
    let (verbs, bindings) = gg_editor::host::open(&gg_ecs::boundary::Verbs {
        actions: &[],
        axes: &[],
    });
    let map = ActionMap::parse(&bindings, verbs.actions, verbs.axes)?;
    let mut input = Input::new(map);
    input.push_named("game");
    let look_x = axis_named(&input, gg_editor::host::verb::LOOK_X)?;
    let look_y = axis_named(&input, gg_editor::host::verb::LOOK_Y)?;
    // Held for the whole leg, and never released: what is being measured is a
    // drag in progress, so the press and release edges are not in the sample.
    input.mouse_button(MouseButton::Right, true);

    let mut world = World::new();
    world.register::<Eye>()?;
    let eye = world.spawn();
    world.insert(eye, Eye::at(sim::DVec3::ZERO, 0.0, 0.0))?;
    // The game's eye never moves here. That is deliberate: the editor's camera
    // latches from it once and flies its own thereafter, so a frame that showed
    // the world's eye instead of the editor's would read as a dead-flat zero
    // rather than as a plausible-looking wrong answer.
    let mut editor = gg_editor::Editor::new(None);

    let mut clock = TickClock::new(SIM_HZ, Pace::Realtime);
    let mut noise = Noise::new();
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
    // Read off the shipped `Look` rather than written down here: the rate is
    // `d.editor_sensitivity` times a constant private to `gg_editor::camera`,
    // and an instrument holding its own copy would grade itself whenever either
    // moved. Negative, the way the camera turns.
    let mut rate = 0.0f64;

    while (wall as f64) < SECONDS * 1.0e9 {
        let elapsed = (period + noise.next() * noise_us).max(1.0) as u64;
        wall += elapsed;
        let due_reports = (wall as f64 * MOUSE_HZ / 1.0e9) as u64;
        let arrived = due_reports.saturating_sub(reports);
        reports = due_reports;
        input.motion((COUNTS_PER_SEC / MOUSE_HZ) as f32 * arrived as f32, 0.0);

        let due = clock.advance(Duration::from_nanos(elapsed));
        input.frame_covered(due.covered);
        for _ in 0..due.count {
            // `App`'s order: the tick's input frame is folded first, then the
            // editor tick reads it — `Camera::fly` takes `previous` from the
            // eye this tick starts at, so there is no separate capture here the
            // way the game's `capture_eye` is separate.
            input.tick();
            editor.tick(
                &mut world,
                &editor_ui_tick(),
                &editor_frame(ticks as u64, &input),
            );
            ticks += 1;
            let yaw = f64::from(editor.eye(Eye::ORIGIN).yaw);
            if frames >= WARMUP {
                turned.push(step(yaw, last_yaw));
            }
            last_yaw = yaw;
        }

        let alpha = clock.alpha();
        let game = Eye::of(&world)?;
        let (previous, current) = editor.eyes(game);
        let blended = gg_extract::blend_eye(previous, current, alpha);
        let composed = match (latched, editor.look(&input)) {
            (Latched::Yes, Some(look)) => {
                rate = COUNTS_PER_SEC * f64::from(look.yaw_rate);
                gg_extract::latched(
                    blended,
                    Latch::of(
                        look,
                        (input.pending(look_x), input.pending(look_y)),
                        (input.spent(look_x), input.spent(look_y)),
                    ),
                    alpha,
                )
            }
            // The unlatched control reads the rate off the same call, so a leg
            // that never turned cannot pass by dividing by a rate it invented.
            (_, look) => {
                if let Some(look) = look {
                    rate = COUNTS_PER_SEC * f64::from(look.yaw_rate);
                }
                blended
            }
        };
        let rendered = f64::from(composed.yaw);
        if frames >= WARMUP {
            shown.push(step(rendered, last_shown));
            let hand = (reports as f64) * (COUNTS_PER_SEC / MOUSE_HZ) * (rate / COUNTS_PER_SEC);
            ages.push(step(hand, rendered) / rate * 1.0e3);
        }
        last_shown = rendered;
        frames += 1;
    }

    // A leg whose camera never turned would divide every ratio by zero and
    // print a table of NaN that reads like a measurement. Named instead.
    if rate == 0.0 {
        anyhow::bail!(
            "the editor camera never reported a look rate — the drag reached no camera at all, \
             so this leg measures nothing"
        );
    }
    let (stall, lurch) = spread(&shown, rate / hz);
    let (tick_lo, tick_hi) = spread(&turned, rate / f64::from(SIM_HZ));
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

/// The router's tick with nothing in it: this leg turns the camera and clicks
/// nothing, and the camera has read raw device motion rather than the router's
/// pointer since §6 M15.2.
fn editor_ui_tick() -> gg_ui::router::Tick {
    gg_ui::router::Tick::default()
}

/// A stopped editor session over a project, which is the only state its camera
/// flies in.
fn editor_frame<'a>(tick: u64, input: &'a Input) -> gg_editor::Frame<'a> {
    gg_editor::Frame {
        extent: EDITOR_EXTENT,
        dpi: 1.0,
        tick,
        hz: SIM_HZ,
        play: gg_editor::Play::Stopped,
        input: Some(input),
        typed: "",
        passes: &[],
        // Inferred rather than named: the type is `gg-rhi`'s, and a whole
        // dependency for one zeroed struct literal is the wrong trade.
        memory: Default::default(),
        save_path: "target/gg-tools/pace.ggsv",
        title: "gg — pace",
        maximized: false,
        project: Some("pace"),
        projects: &[],
        reload: None,
        draw_cursor: false,
    }
}

/// An axis id by the name the host bound it under. By name for the reason the
/// editor resolves its own verbs by name: the index is a property of whatever
/// game the session is open over.
fn axis_named(input: &Input, name: &str) -> anyhow::Result<AxisId> {
    input
        .map()
        .axis_names()
        .iter()
        .position(|n| n == name)
        .map(AxisId::new)
        .ok_or_else(|| anyhow::anyhow!("the editor's own map has no {name}"))
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
