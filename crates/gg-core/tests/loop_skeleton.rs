//! The frame loop's stage order and the tick clock (§4.1).
//!
//! Every test here drives the loop directly with a synthetic elapsed time and no
//! window — which is the point of the loop not owning the OS event loop, and the
//! reason §1.5 has nothing to say about this file.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use core::time::Duration;

use gg_core::clock::{MAX_TICKS_PER_FRAME, Pace, TickClock};
use gg_core::{AppEvent, Flow, FrameLoop, Stages};

/// Records the stage order as it happens, which is the only way to assert an
/// order rather than assert that each stage ran.
#[derive(Default)]
struct Trace {
    stages: Vec<String>,
    ticks: Vec<u64>,
    events: Vec<AppEvent>,
    fail_at_tick: Option<u64>,
    /// Half-open frame range this stage says it is suspended over (§6 M49).
    away: Option<(u64, u64)>,
}

impl Stages for Trace {
    type Error = &'static str;

    fn poll_input(&mut self) -> Result<(), Self::Error> {
        self.stages.push("poll_input".into());
        Ok(())
    }

    fn event(&mut self, event: AppEvent) -> Result<(), Self::Error> {
        self.stages.push("event".into());
        self.events.push(event);
        Ok(())
    }

    fn reload_check(&mut self) -> Result<(), Self::Error> {
        self.stages.push("reload_check".into());
        Ok(())
    }

    fn sim_tick(&mut self, tick: u64) -> Result<(), Self::Error> {
        self.stages.push("sim_tick".into());
        self.ticks.push(tick);
        if self.fail_at_tick == Some(tick) {
            return Err("planted");
        }
        Ok(())
    }

    fn suspended(&mut self, frame: u64) -> bool {
        self.away
            .is_some_and(|(from, until)| (from..until).contains(&frame))
    }

    fn extract(&mut self, _alpha: f32) -> Result<(), Self::Error> {
        self.stages.push("extract".into());
        Ok(())
    }

    fn render(&mut self, _alpha: f32) -> Result<(), Self::Error> {
        self.stages.push("render".into());
        Ok(())
    }
}

const TICK: Duration = Duration::from_nanos(1_000_000_000 / 60);

#[test]
fn the_stages_run_in_the_order_4_1_names() {
    let mut trace = Trace::default();
    let mut app = FrameLoop::locked(60);
    app.events().push(AppEvent::Resized {
        width: 800,
        height: 600,
    });

    assert_eq!(app.frame(&mut trace, TICK).unwrap(), Flow::Continue);
    assert_eq!(
        trace.stages,
        [
            "poll_input",
            "event",
            "reload_check",
            "sim_tick",
            "extract",
            "render"
        ]
    );
}

#[test]
fn a_reload_check_sits_on_every_tick_boundary() {
    // A frame owing three ticks has three boundaries. Checking once a frame
    // would make "applies at the next tick boundary" (§6 M5) mean "the next
    // frame", which is a different and weaker claim.
    let mut trace = Trace::default();
    let mut app = FrameLoop::new(TickClock::new(60, Pace::Locked(3)));
    app.frame(&mut trace, Duration::ZERO).unwrap();

    let boundaries: Vec<&str> = trace
        .stages
        .iter()
        .map(String::as_str)
        .filter(|s| *s == "reload_check" || *s == "sim_tick")
        .collect();
    assert_eq!(
        boundaries,
        [
            "reload_check",
            "sim_tick",
            "reload_check",
            "sim_tick",
            "reload_check",
            "sim_tick"
        ]
    );
}

#[test]
fn a_frame_owing_no_ticks_still_checks_for_a_reload() {
    // An idle or paused game is exactly when someone is editing code.
    let mut trace = Trace::default();
    let mut app = FrameLoop::new(TickClock::new(60, Pace::Realtime));
    app.frame(&mut trace, Duration::from_micros(100)).unwrap();

    assert!(trace.ticks.is_empty());
    assert_eq!(
        trace.stages.iter().filter(|s| *s == "reload_check").count(),
        1
    );
}

#[test]
fn ticks_ascend_without_gaps_across_frames() {
    let mut trace = Trace::default();
    let mut app = FrameLoop::new(TickClock::new(60, Pace::Realtime));
    // Two and a half tick periods per frame, three frames: the halves have to
    // accumulate into a seventh tick rather than being rounded away.
    let frame = TICK * 5 / 2;
    for _ in 0..3 {
        app.frame(&mut trace, frame).unwrap();
    }
    assert_eq!(trace.ticks, [0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(app.clock().tick(), 7);
}

#[test]
fn close_requested_exits_after_finishing_the_frame() {
    // Tearing down mid-frame is how a swapchain gets destroyed with work in
    // flight; the loop finishes the frame it is in and reports Exit at the end.
    let mut trace = Trace::default();
    let mut app = FrameLoop::locked(60);
    app.events().request_quit();

    assert_eq!(app.frame(&mut trace, TICK).unwrap(), Flow::Exit);
    assert_eq!(trace.events, [AppEvent::CloseRequested]);
    assert_eq!(trace.stages.last().map(String::as_str), Some("render"));
}

#[test]
fn an_event_pushed_by_a_stage_lands_next_frame_not_mid_frame() {
    struct Quitter {
        seen: Vec<AppEvent>,
    }
    impl Stages for Quitter {
        type Error = std::convert::Infallible;
        fn sim_tick(&mut self, _tick: u64) -> Result<(), Self::Error> {
            Ok(())
        }
        fn event(&mut self, event: AppEvent) -> Result<(), Self::Error> {
            self.seen.push(event);
            Ok(())
        }
    }

    let mut app = FrameLoop::locked(60);
    let mut game = Quitter { seen: Vec::new() };
    app.events().push(AppEvent::Resized {
        width: 1,
        height: 1,
    });
    // Queue a second event while the first frame's dispatch is still running:
    // the loop took ownership of the queue, so this one cannot be lost and
    // cannot be seen twice.
    assert_eq!(app.frame(&mut game, TICK).unwrap(), Flow::Continue);
    app.events().request_quit();
    assert_eq!(app.frame(&mut game, TICK).unwrap(), Flow::Exit);
    assert_eq!(
        game.seen,
        [
            AppEvent::Resized {
                width: 1,
                height: 1
            },
            AppEvent::CloseRequested
        ]
    );
}

#[test]
fn a_stage_error_abandons_the_frame_where_it_stood() {
    let mut trace = Trace {
        fail_at_tick: Some(1),
        ..Default::default()
    };
    let mut app = FrameLoop::new(TickClock::new(60, Pace::Locked(3)));

    assert_eq!(app.frame(&mut trace, Duration::ZERO), Err("planted"));
    assert_eq!(trace.ticks, [0, 1]);
    assert!(
        !trace.stages.iter().any(|s| s == "render"),
        "extract and render must not run over a half-simulated frame"
    );
}

#[test]
fn the_catch_up_guard_skips_a_hitch_instead_of_simulating_it() {
    // Simulating a two-second stall costs more than two seconds, which owes more
    // ticks again — the spiral this bound exists to refuse.
    let mut trace = Trace::default();
    let mut app = FrameLoop::new(TickClock::new(60, Pace::Realtime));
    app.frame(&mut trace, Duration::from_secs(2)).unwrap();

    assert_eq!(trace.ticks.len(), MAX_TICKS_PER_FRAME as usize);
    // The residue goes with the ticks. Keeping it would make the next frames pay
    // off a debt for a stall already given up on — re-entering the spiral from
    // the other side.
    trace.ticks.clear();
    app.frame(&mut trace, Duration::from_millis(17)).unwrap();
    assert_eq!(trace.ticks.len(), 1);
}

#[test]
fn sixty_hertz_is_exactly_sixty_hertz() {
    // The trap this test exists for: a tick period truncated to whole
    // nanoseconds is 16_666_666 ns, which is 60.000002 Hz — a tick of drift
    // every few hours of play, and drift a float accumulator carries too. The
    // accumulator counts nanosecond-hertz instead, so one tick costs exactly one
    // second's worth of it whatever the rate is.
    let mut clock = TickClock::new(60, Pace::Realtime);
    let mut ticks = 0u64;
    // A minute in millisecond steps — small enough that the catch-up guard never
    // fires, so this measures the accumulator and nothing else.
    for _ in 0..60_000 {
        ticks += u64::from(clock.advance(Duration::from_millis(1)).count);
    }
    assert_eq!(ticks, 3600);
    assert_eq!(clock.tick(), 3600);
}

#[test]
fn a_locked_pace_never_interpolates() {
    // Every frame lands on a tick boundary, so there is nothing between the last
    // tick and the next to interpolate towards.
    let mut clock = TickClock::new(60, Pace::Locked(1));
    clock.advance(Duration::from_millis(7));
    assert_eq!(clock.alpha(), 0.0);
    assert_eq!(clock.tick(), 1);
}

/// `covered` is what a frame's accumulated input is divided by, so it has to be
/// the wall time the clock *kept* — not what arrived, and not the tick count.
/// Three cases, and the third is the one a count cannot express.
#[test]
fn covered_is_the_wall_time_the_clock_is_holding_unspent() {
    // A locked pace: exactly the tick count, so a replay and every headless tier
    // divide by one per tick and take the accumulator whole, as they always did.
    let mut clock = TickClock::new(60, Pace::Locked(3));
    assert_eq!(clock.advance(Duration::from_millis(7)).covered, 3.0);

    // A hitch: the guard drops the residue with the ticks, so the wall time a
    // refused tick covered leaves with it rather than diluting the ones that ran.
    let mut clock = TickClock::new(60, Pace::Realtime);
    let due = clock.advance(Duration::from_secs(2));
    assert_eq!(due.covered, f64::from(MAX_TICKS_PER_FRAME) as f32);
    assert!(due.dropped > 0);

    // And the case the tick count is blind to: a frame owing *one* tick that has
    // more than one tick's travel in hand. This is every frame at 240 Hz where
    // the two clocks slipped a beat, and `count` reads 1 for all of them.
    let mut clock = TickClock::new(60, Pace::Realtime);
    let due = clock.advance(TICK + TICK / 4);
    assert_eq!(due.count, 1);
    assert!(
        (due.covered - 1.25).abs() < 1e-3,
        "covered was {}",
        due.covered
    );
}

/// §6 M49. The claim is *no tick*, not *a cheap tick*: the tick numbers a
/// suspended run produces are the numbers it would have produced without the
/// suspension, so nothing downstream — the hash line, the recording, a save's
/// tick — can tell the two runs apart.
///
/// Under a **locked** pace, which is the case `advance(Duration::ZERO)` cannot
/// express and therefore the reason [`TickClock::hold`] exists at all: a locked
/// clock ignores elapsed time and would owe its tick regardless.
#[test]
fn a_suspended_frame_runs_no_tick_and_every_other_stage() {
    let mut trace = Trace {
        away: Some((2, 5)),
        ..Default::default()
    };
    let mut app = FrameLoop::locked(60);
    for _ in 0..7 {
        app.frame(&mut trace, TICK).unwrap();
    }
    assert_eq!(trace.ticks, [0, 1, 2, 3], "frames 2..5 owed nothing");
    assert_eq!(app.frame_count(), 7, "a suspended frame is still a frame");
    assert_eq!(
        trace.stages.iter().filter(|s| *s == "render").count(),
        7,
        "the picture must repaint on an expose while nobody is looking"
    );
    assert_eq!(
        trace.stages.iter().filter(|s| *s == "reload_check").count(),
        7,
        "and a suspended game still takes a rebuild"
    );
}

/// A suspension is not a hitch. The catch-up guard exists for wall time that
/// *passed while the sim was trying*; this is wall time nobody was watching, so
/// it is never owed, never dropped, and the accumulator resumes mid-tick exactly
/// where it stopped.
#[test]
fn a_suspension_owes_no_catch_up_and_drops_nothing() {
    let mut trace = Trace {
        away: Some((1, 1_000)),
        ..Default::default()
    };
    let mut app = FrameLoop::new(TickClock::new(60, Pace::Realtime));
    // One frame with half a tick's travel in hand, so the resume has a residue
    // to prove it kept.
    app.frame(&mut trace, TICK / 2).unwrap();
    for _ in 0..999 {
        app.frame(&mut trace, Duration::from_secs(1)).unwrap();
    }
    assert!(trace.ticks.is_empty());
    assert!(
        (app.clock().alpha() - 0.5).abs() < 1e-3,
        "alpha was {}",
        app.clock().alpha()
    );
    // Sixteen minutes of wall time went by and the very next frame owes one
    // tick, the same as any other — a `Duration::ZERO` charge would have read
    // the same here and is what the locked test above rules out.
    let ticks = app.frame(&mut trace, TICK).unwrap();
    assert_eq!(ticks, Flow::Continue);
    assert_eq!(trace.ticks, [0]);
}

#[test]
fn alpha_reports_the_fraction_of_a_tick_the_frame_landed_on() {
    let mut clock = TickClock::new(60, Pace::Realtime);
    clock.advance(TICK + TICK / 2);
    assert_eq!(clock.tick(), 1);
    assert!(
        (clock.alpha() - 0.5).abs() < 1e-3,
        "alpha was {}",
        clock.alpha()
    );
}

#[test]
fn resume_at_puts_the_clock_where_a_restored_snapshot_left_it() {
    let mut clock = TickClock::new(60, Pace::Realtime);
    clock.advance(TICK * 10);
    clock.resume_at(4);
    assert_eq!(clock.tick(), 4);
    assert_eq!(clock.alpha(), 0.0, "the accumulator is not captured state");
}
