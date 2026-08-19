//! `gg-tools governor` — does the automatic render scale settle, and how fast
//! (§6 M80)?
//!
//! §6 M78 deferred this milestone with one sentence: a frame-rate governor is a
//! control loop, and a control loop that oscillates is a picture that visibly
//! breathes. That is a claim about *dynamics*, and no amount of staring at a
//! session answers it — a loop can look fine for a minute and ring for the next
//! one, and the frame it rings at is the frame nobody was watching.
//!
//! So the controller is driven here instead, against a **plant** rather than a
//! recording: `frame_ms = floor + fill * scale²`. Squared because `r.scale` is a
//! factor on each *axis* and the cost is the pixel count, which is what makes
//! this a nonlinear plant and the question worth asking at all.
//!
//! The two constants are measured off the knob the governor actually turns, and
//! that is **not** `frame --sweep`'s fit even though both describe the same room
//! on the same device. A sweep moves the *window*, so every pass scales with it
//! and the floor is 1.149 ms; `r.scale` moves only the **scene**, and `post`,
//! the UI and the readback stay the size they were — which is §6 M78's whole
//! selling point and costs three times the fixed floor. Four points of
//! `--set r.scale=` on the integrated Radeon fit `3.257 + 50.60 * scale²`
//! against measurements of 54.041, 31.283, 21.478 and 16.160 at 1.0, 0.75, 0.6
//! and 0.5. Taking the sweep's constants instead read 14 % low at half scale,
//! which flatters every settling number below.
//!
//! The real [`Governor`] is the subject — this restates none of its arithmetic.
//! Beside it runs a **naive** controller that decides every frame from the same
//! window without clearing it, which is the obvious implementation and the one
//! the shipped rule exists to rule out. Two columns of the same table, so the
//! rule is priced rather than asserted.

use anyhow::Result;
use gg_core::governor::Governor;
use std::time::Duration;

/// The plant of demo 12's room at 1920x1080 on the integrated Radeon, measured
/// through `r.scale` itself: 3.257 ms that the knob does not reach — `post`, the
/// UI pass and the readback, which stay the window's size on purpose — and
/// 50.60 ms that is the scene's own pixel count.
const FLOOR_MS: f64 = 3.257;
const FILL_MS: f64 = 50.60;

/// Long enough that a loop which rings slowly has room to do it. A settling
/// time near this is reported as "never", because a controller still moving
/// after twelve hundred frames has not settled, it has been interrupted.
const FRAMES: usize = 1200;

/// The rate a player asks to protect. 60 is the one worth defending on this
/// plant — the machine cannot reach it at full scale and can at the floor, so
/// every trace below has somewhere to go and something to fail at.
const TARGET_HZ: u32 = 60;

pub fn run(_args: &[String]) -> Result<()> {
    println!(
        "the automatic render scale against a plant of {FLOOR_MS:.3} ms + {FILL_MS:.2} ms x \
         scale², protecting {TARGET_HZ} Hz\n"
    );
    println!(
        "  at scale 1.00 this plant runs {:.1} ms ({:.1} Hz); at the 0.50 floor, {:.1} ms ({:.1} Hz)\n",
        plant_ms(1.0, 1.0),
        1000.0 / plant_ms(1.0, 1.0),
        plant_ms(0.5, 1.0),
        1000.0 / plant_ms(0.5, 1.0)
    );

    println!(
        "  trace                shipped                                   naive\n  \
         {:<18} {:>7} {:>7} {:>6} {:>7}   {:>7} {:>7} {:>6} {:>7}",
        "", "settle", "scale", "turns", "late %", "settle", "scale", "turns", "late %"
    );
    for trace in &TRACES {
        let shipped = drive(trace, false);
        let naive = drive(trace, true);
        println!(
            "  {:<18} {:>7} {:>7.3} {:>6} {:>7.1}   {:>7} {:>7.3} {:>6} {:>7.1}",
            trace.name,
            settle(&shipped),
            shipped.factor,
            shipped.turns,
            shipped.late,
            settle(&naive),
            naive.factor,
            naive.turns,
            naive.late
        );
    }
    println!(
        "\n  `turns` is the count of *direction changes* and is the number this exists to read: \
         \n  a loop that steps down four times and stops has settled, one that alternates has not, \
         \n  and both can show the same number of moves. `late %` is the share of frames over the \
         \n  target, which is what a player feels rather than what a controller intends."
    );
    Ok(())
}

/// What a trace does to the machine over time, as a multiplier on the plant.
/// Each is chosen to fail in a way the others cannot.
struct Trace {
    name: &'static str,
    /// The load at frame `i`, and whether this frame is a one-off hitch.
    load: fn(usize) -> (f64, bool),
}

const TRACES: [Trace; 5] = [
    // A machine that was always this slow. The base case, and the only one
    // whose answer is arithmetic: it should walk down and stop.
    Trace {
        name: "steady",
        load: |_| (1.0, false),
    },
    // Fine, then a heavier room. The step response, which is where a loop
    // overshoots if it is going to.
    Trace {
        name: "step",
        load: |i| ((if i < 200 { 0.25 } else { 1.0 }), false),
    },
    // Heavy, then the player walks out of it. Does the scale come back, and how
    // long does a player look at a soft picture they no longer need?
    Trace {
        name: "recovery",
        load: |i| ((if i < 400 { 1.0 } else { 0.25 }), false),
    },
    // A comfortable machine with a shader compile every second. Nothing should
    // move: this is the false-positive test, and the one a controller reading
    // the worst frame rather than the median fails outright.
    Trace {
        name: "hitches",
        load: |i| (0.25, i % 60 == 0),
    },
    // Sat exactly where one step down is too much and none is too little. The
    // dead band's own test — with no dead band this alternates forever.
    Trace {
        name: "boundary",
        load: |_| (0.335, false),
    },
];

/// Frame period at a given scale under a given load, plus the hitch.
fn plant_ms(scale: f64, load: f64) -> f64 {
    FLOOR_MS + FILL_MS * scale * scale * load
}

struct Run {
    /// First frame after which the scale never moved again, if it stopped.
    settled: Option<usize>,
    /// Moves at all — what tells "never touched it" apart from "never stopped",
    /// which are the best and the worst outcome and print the same otherwise.
    moves: usize,
    factor: f64,
    turns: usize,
    late: f64,
}

/// One trace through one controller.
///
/// `naive` swaps the shipped rule for the obvious one — a rolling window that
/// is never cleared, so every decision is taken over frames measured partly at
/// settings no longer in force. It is a straw man and lives here rather than in
/// the engine, but it is the straw man every first implementation of this is.
fn drive(trace: &Trace, naive: bool) -> Run {
    let mut real = Governor::new();
    let mut window: Vec<f64> = Vec::new();
    let mut factor = 1.0_f64;
    let target_ms = 1000.0 / f64::from(TARGET_HZ);

    let mut last_move = None;
    let mut moves = 0;
    let mut turns = 0;
    let mut direction = 0i32;
    let mut late = 0usize;
    for i in 0..FRAMES {
        let (load, hitch) = (trace.load)(i);
        let ms = match hitch {
            true => 300.0,
            false => plant_ms(factor, load),
        };
        if !hitch && ms > target_ms {
            late += 1;
        }
        let was = factor;
        if naive {
            window.push(ms);
            if window.len() > 30 {
                window.remove(0);
            }
            if window.len() == 30 {
                let mut sorted = window.clone();
                sorted.sort_by(f64::total_cmp);
                let median = sorted[15];
                if median > target_ms * 1.05 && factor > 0.5 {
                    factor = (factor * 0.9).max(0.5);
                } else if median < target_ms * 0.80 && factor < 1.0 {
                    factor = (factor * 1.05).min(1.0);
                }
            }
        } else {
            real.observe_at(Duration::from_micros((ms * 1000.0) as u64), TARGET_HZ);
            factor = real.factor();
        }
        if factor != was {
            last_move = Some(i);
            moves += 1;
            // A direction change and not a move: four steps down is a
            // controller converging, and down-up-down-up is one that is not,
            // and the two can have the same number of moves.
            let now = match factor > was {
                true => 1,
                false => -1,
            };
            if direction != 0 && now != direction {
                turns += 1;
            }
            direction = now;
        }
    }
    Run {
        // Settled means it stopped in time to be seen stopping — a move in the
        // last tenth of the run is a loop still going, not one that finished.
        settled: last_move.filter(|i| *i < FRAMES - FRAMES / 10),
        moves,
        factor,
        turns,
        late: 100.0 * late as f64 / FRAMES as f64,
    }
}

fn settle(run: &Run) -> String {
    match (run.moves, run.settled) {
        // Best and worst, and they must not read alike: nothing was ever wrong
        // enough to act on, against a loop still moving when the run ended.
        (0, _) => "none".to_owned(),
        (_, Some(i)) => format!("{i}"),
        (_, None) => "never".to_owned(),
    }
}
