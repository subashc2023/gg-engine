//! The loop skeleton (§4.1) — stage ordering and nothing else.
//!
//! `gg-core` owns the *shape* of a frame and none of its contents. It sits below
//! `gg-render`, `gg-extract` and `gg-platform` in §3's dependency direction, so
//! it could not call them if it wanted to; the app shell (`gg-runtime`) supplies
//! the concrete stages by implementing [`Stages`] and the loop calls them in the
//! one order there is:
//!
//! ```text
//! poll_input → pump_events → reload_check → sim_ticks(0..n) → extract → render → frame_mark
//! ```
//!
//! **The loop is the scheduler** (§4.1): no system graph, no ordering solver, no
//! parallel dispatch. Game systems run inside `sim_tick` in the order the systems
//! table lists them (§4.2.2), which is what makes system order part of game code
//! and therefore part of what a reload can change.
//!
//! # It does not own the OS event loop
//!
//! [`FrameLoop::frame`] runs exactly one frame and returns; the shell drives it
//! from whatever `gg-platform` hands it, passing the elapsed wall time. Two
//! things fall out, both load-bearing. `gg-core` never links winit, so the
//! sim/replay path stays winit-free (§1.5). And the loop is drivable from a test
//! with a synthetic clock and no window at all — which is how the stage ordering
//! below is actually proven rather than asserted.

use core::time::Duration;

use crate::clock::{Due, Pace, TickClock};
use crate::event::{AppEvent, Events};

/// Whether the shell should come back for another frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    /// Call [`FrameLoop::frame`] again.
    Continue,
    /// Shut down. The frame that returned this one ran to completion.
    Exit,
}

/// The stages a frame runs, supplied by the app shell.
///
/// One trait with `&mut self` rather than a struct of closures: every stage
/// wants mutable access to the same app state, and closures would each need
/// their own borrow of it. All stages but [`Stages::sim_tick`] default to doing
/// nothing, so a test — or demo 00 — implements the one it cares about.
pub trait Stages {
    /// Whatever the shell's stages fail with. `gg-core` never inspects it, which
    /// is how the shell keeps using `anyhow` without `gg-core` depending on it.
    type Error;

    /// Drain the platform's input into `gg-input` and latch this tick's frame.
    fn poll_input(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// One queued [`AppEvent`], in arrival order. Called between ticks, never
    /// during one.
    fn event(&mut self, event: AppEvent) -> Result<(), Self::Error> {
        let _ = event;
        Ok(())
    }

    /// What this frame charged the clock, before the first tick runs.
    ///
    /// The one thing a stage cannot work out for itself, and it matters because
    /// continuous input is accumulated per *frame* and spent per *tick*: the
    /// travel in hand at a tick is whatever a variable number of frames
    /// delivered, and spending it whole turns the camera by a rate that wobbles
    /// with the frame clock while held keys deflect an axis identically
    /// throughout. [`Due::covered`] is the denominator that fixes it. A frame
    /// owing nothing is passed too — it is a frame the accumulator grew on.
    fn ticks_due(&mut self, due: Due) {
        let _ = due;
    }

    /// The game-code swap point (§4.2.2): a rebuilt dylib that passed its
    /// version and fingerprint checks is swapped in here. Compiles to nothing in
    /// dist, where there is no watcher to have found one.
    fn reload_check(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Advance the sim by exactly one tick. `tick` is the sim clock (§2) and
    /// ascends without gaps for the life of the run.
    fn sim_tick(&mut self, tick: u64) -> Result<(), Self::Error>;

    /// Copy render-relevant state into flat per-frame arrays (§4.1) — the
    /// determinism membrane (§1.4). `alpha` is how far between the last tick and
    /// the next this frame lands, in `0.0..1.0`.
    fn extract(&mut self, alpha: f32) -> Result<(), Self::Error> {
        let _ = alpha;
        Ok(())
    }

    /// Record and submit the frame. Present is inside this stage rather than
    /// beside it: `gg-rhi`'s frame call acquires, records, submits and presents
    /// as one unit, and splitting the callback would only hand the shell a
    /// second place to get the order wrong.
    fn render(&mut self, alpha: f32) -> Result<(), Self::Error> {
        let _ = alpha;
        Ok(())
    }

    /// Whether the app itself wants the loop to stop, asked at the end of every
    /// frame — the stage-side twin of [`AppEvent::CloseRequested`].
    ///
    /// A queued event would have been the obvious answer and is the wrong one:
    /// [`Events`](crate::Events) belongs to the loop, and handing a stage a
    /// channel into it would give the shell a second way to end a frame. This is
    /// for a decision a stage makes *about itself* — rejuvenation (§4.2.2) is
    /// the first, where the world has been staged for a successor process and
    /// every remaining frame is one the successor should be running.
    fn quitting(&self) -> bool {
        false
    }
}

/// The frame loop: a tick clock, an event queue, and the stage order.
#[derive(Debug)]
pub struct FrameLoop {
    clock: TickClock,
    events: Events,
    frame: u64,
    quit: bool,
}

impl Default for FrameLoop {
    fn default() -> Self {
        Self::new(TickClock::default())
    }
}

impl FrameLoop {
    /// A loop over `clock`, with an empty event queue.
    pub fn new(clock: TickClock) -> Self {
        Self {
            clock,
            events: Events::new(),
            frame: 0,
            quit: false,
        }
    }

    /// A loop that runs exactly one sim tick per frame and ignores wall time —
    /// what `--frames N`, a replay, and every headless CI run want (§5.6).
    pub fn locked(hz: u32) -> Self {
        Self::new(TickClock::new(hz, Pace::Locked(1)))
    }

    /// The same loop, with the sim clock resuming at `tick` — zero for a fresh
    /// run, or where a rejuvenated session's predecessor stopped (§4.2.2).
    #[must_use]
    pub fn resuming_at(mut self, tick: u64) -> Self {
        self.clock.resume_at(tick);
        self
    }

    /// The tick clock — the sim clock (§2) lives here.
    pub fn clock(&self) -> &TickClock {
        &self.clock
    }

    /// The tick clock, to resume it after a restore or change its pace.
    pub fn clock_mut(&mut self) -> &mut TickClock {
        &mut self.clock
    }

    /// The event queue. The shell pushes platform events into it as they arrive;
    /// they are dispatched at `pump_events`, not where they landed.
    pub fn events(&mut self) -> &mut Events {
        &mut self.events
    }

    /// Frames run so far. Not sim time — a frame is a presentation, and the
    /// number of them a run produced is a property of the machine.
    pub fn frame_count(&self) -> u64 {
        self.frame
    }

    /// Run one frame.
    ///
    /// Returns [`Flow::Exit`] once a [`AppEvent::CloseRequested`] has been
    /// drained, or once [`Stages::quitting`] says so — after finishing the frame
    /// either way, so the shell never tears down a swapchain mid-frame. A stage
    /// error propagates immediately and the frame is abandoned where it stood.
    pub fn frame<S: Stages>(
        &mut self,
        stages: &mut S,
        elapsed: Duration,
    ) -> Result<Flow, S::Error> {
        {
            profiling::scope!("poll_input");
            stages.poll_input()?;
        }

        {
            profiling::scope!("pump_events");
            for event in self.events.take() {
                if event == AppEvent::CloseRequested {
                    self.quit = true;
                }
                stages.event(event)?;
            }
        }

        let due = self.clock.advance(elapsed);
        // Before the loop, not inside it: a stage spending a frame's accumulated
        // input has to know the denominator at the first tick, not the last.
        stages.ticks_due(due);
        if due.dropped > 0 {
            tracing::debug!(
                dropped = due.dropped,
                tick = due.first,
                "catch-up guard skipped sim ticks"
            );
        }

        // `reload_check` runs on every tick boundary, not once a frame: a frame
        // owing three ticks has three of them, and §6 M5 requires a swap at the
        // *next* boundary rather than the next frame. A frame owing none still
        // checks, so a paused or idle game reloads.
        if due.count == 0 {
            reload_check(stages)?;
        }
        for i in 0..due.count {
            reload_check(stages)?;
            profiling::scope!("sim_tick");
            stages.sim_tick(due.first + u64::from(i))?;
        }

        let alpha = self.clock.alpha();
        {
            profiling::scope!("extract");
            stages.extract(alpha)?;
        }
        {
            profiling::scope!("render");
            stages.render(alpha)?;
        }

        // One line per frame, off by default (§4.8's convention), for the
        // question a frame time alone cannot answer: whether a stutter is the
        // *sim* landing unevenly or the picture arriving late. A 60 Hz sim under
        // a faster display runs 0 or 1 ticks a frame in a pattern set by
        // `elapsed`, and `alpha` is where between two poses the frame was
        // interpolated — a smooth session walks alpha evenly, and a session that
        // judders repeats or skips it. `RUST_LOG=gg::pace=debug`.
        tracing::debug!(
            target: "gg::pace",
            frame = self.frame,
            elapsed_us = elapsed.as_micros(),
            ticks = due.count,
            dropped = due.dropped,
            alpha = alpha,
            "paced"
        );

        // Closes the frame *after* every zone above it, so the Tracy timeline
        // shows frames rather than zones straddling them.
        profiling::finish_frame!();
        self.frame += 1;
        Ok(if self.quit || stages.quitting() {
            Flow::Exit
        } else {
            Flow::Continue
        })
    }

    /// Run until a stage or an event asks to stop, or `max_frames` have run.
    ///
    /// The convenience form for bounded and headless runs: every frame is
    /// charged one tick period of elapsed time, so a `Pace::Realtime` loop
    /// driven this way still advances deterministically.
    pub fn run<S: Stages>(&mut self, stages: &mut S, max_frames: u64) -> Result<u64, S::Error> {
        let step = Duration::from_nanos(1_000_000_000 / u64::from(self.clock.hz()));
        let mut frames = 0;
        while frames < max_frames {
            frames += 1;
            if self.frame(stages, step)? == Flow::Exit {
                break;
            }
        }
        Ok(frames)
    }
}

/// Pulled out so the two call sites in [`FrameLoop::frame`] cannot disagree about
/// the zone name.
fn reload_check<S: Stages>(stages: &mut S) -> Result<(), S::Error> {
    profiling::scope!("reload_check");
    stages.reload_check()
}
