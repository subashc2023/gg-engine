//! The tick clock (§4.1, §2 Sim time row) — the accumulator that decouples sim
//! rate from frame rate.
//!
//! Sim time is a `u64` tick count and nothing else. The accumulator here is the
//! one place upstream of render that knows about wall time, and it is deliberately
//! **integer**: §4.1 describes it as float, but nanosecond-hertz costs nothing and
//! removes the residue a float (or a truncated nanosecond period) carries — one
//! tick at 60 Hz is 16.666…ms, and the drift from rounding that is real, unbounded
//! and invisible. The only float the clock produces is [`TickClock::alpha`], which
//! exists for render interpolation and never travels upstream.

use core::time::Duration;

/// Ticks per second when nothing says otherwise (§4.1).
pub const DEFAULT_TICK_HZ: u32 = 60;

/// Sim ticks a single frame may run before the clock stops trying to catch up.
///
/// A frame that owes more than this is a hitch — an OS scheduling gap, a
/// breakpoint, a shader recompile — and simulating it is worse than admitting
/// it: catching up costs more time than the gap did, which owes more ticks
/// again. The catch-up spiral is a longer outage than the honest skip.
pub const MAX_TICKS_PER_FRAME: u32 = 8;

/// The accumulator's unit: one tick costs exactly this much of it, whatever the
/// rate is, because elapsed nanoseconds are scaled by `hz` on the way in. That
/// is what makes a 60 Hz clock exact rather than 60.000002 Hz.
const NANOS_PER_SEC: u64 = 1_000_000_000;

/// What drives the tick count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pace {
    /// Wall time: ticks come due as real time passes, and a slow frame owes
    /// several. What a human plays under.
    Realtime,
    /// Exactly this many ticks per frame, wall time ignored.
    ///
    /// A bounded run (`--frames N`), a replay and every headless CI run use it,
    /// so how many ticks a run simulates is a property of the run rather than of
    /// how fast the machine happened to be (§5.6). [`TickClock::alpha`] is `0.0`
    /// throughout: there is nothing to interpolate towards when every frame
    /// lands exactly on a tick boundary.
    Locked(u32),
}

/// Ticks a frame owes, from [`TickClock::advance`].
///
/// Not `Eq`: [`Due::covered`] is the render side's own float (§1.4), and a
/// clock's report is compared for equality nowhere the sim can see.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Due {
    /// Tick number of the first one owed. Sim ticks are consecutive forever;
    /// this is the count *before* the frame, not an index into the frame.
    pub first: u64,
    /// How many to run, never more than [`MAX_TICKS_PER_FRAME`].
    pub count: u32,
    /// Ticks the catch-up guard refused. Nonzero means the sim skipped forward
    /// in wall-clock terms — worth a log line, never worth simulating.
    pub dropped: u64,
    /// Wall time the clock now holds unspent, in *tick periods*: `count` plus
    /// [`TickClock::alpha`], measured after this frame's charge landed.
    ///
    /// The denominator for anything accumulated per frame and spent per tick.
    /// Continuous input is the one such thing (§4.7): it arrives on the frame's
    /// clock, so the travel in hand at a tick is whatever a *variable* number of
    /// frames delivered — four of them at 240 Hz, and sometimes three or five,
    /// which is a fifth of a turn's rate appearing and disappearing while held
    /// keys deflect an axis identically throughout. Dividing by the tick count
    /// alone cannot see that; dividing by this can, because it is the wall time
    /// the accumulator actually spans.
    ///
    /// A locked pace reports exactly `count` — alpha is zero throughout — so
    /// every replay, golden run and headless tier divides by one per tick and
    /// takes the whole accumulator, as it always did. The catch-up guard is
    /// covered for free: it zeroes the residue, so a dropped hitch's wall time
    /// leaves with the ticks rather than diluting the ones that did run.
    pub covered: f32,
}

/// Fixed-timestep clock: wall time in, whole sim ticks out.
#[derive(Clone, Debug)]
pub struct TickClock {
    hz: u32,
    pace: Pace,
    // Nanosecond-hertz owed but not yet spent. Never a float, never sim state.
    accumulated: u64,
    tick: u64,
}

impl Default for TickClock {
    fn default() -> Self {
        Self::new(DEFAULT_TICK_HZ, Pace::Realtime)
    }
}

impl TickClock {
    /// A clock at `hz` ticks per second.
    ///
    /// # Panics
    ///
    /// If `hz` is zero. The rate is startup configuration, so this is a
    /// programming error rather than a runtime condition — a zero-rate clock
    /// would silently never tick.
    pub fn new(hz: u32, pace: Pace) -> Self {
        assert!(hz > 0, "tick rate must be non-zero");
        Self {
            hz,
            pace,
            accumulated: 0,
            tick: 0,
        }
    }

    /// Ticks per second — a *rate*, which is what crosses the reload boundary
    /// (`TickCtx::tick_hz`) so no `dt` float has to (§2 Sim time row).
    pub fn hz(&self) -> u32 {
        self.hz
    }

    /// Ticks completed so far: the sim clock (§2).
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// What drives the tick count.
    pub fn pace(&self) -> Pace {
        self.pace
    }

    /// How far the accumulator sits between the last tick and the next, in
    /// `0.0..1.0` — the render interpolation factor.
    ///
    /// Render-side by construction: it is derived *from* the clock and never
    /// written back into it, so no sim result can depend on it.
    pub fn alpha(&self) -> f32 {
        self.accumulated as f32 / NANOS_PER_SEC as f32
    }

    /// Charge `elapsed` to the clock and report the ticks it now owes.
    ///
    /// Advances the tick count by the returned `count` — the caller runs the
    /// ticks, it does not decide how many.
    pub fn advance(&mut self, elapsed: Duration) -> Due {
        let first = self.tick;
        let (count, dropped) = match self.pace {
            Pace::Locked(n) => (n, 0),
            Pace::Realtime => {
                let charge = elapsed.as_nanos().saturating_mul(u128::from(self.hz));
                self.accumulated = self
                    .accumulated
                    .saturating_add(u64::try_from(charge).unwrap_or(u64::MAX));

                let owed = self.accumulated / NANOS_PER_SEC;
                let count = owed.min(u64::from(MAX_TICKS_PER_FRAME));
                self.accumulated -= count * NANOS_PER_SEC;
                if owed > count {
                    // Drop the residue too, not just the ticks: keeping it would
                    // make the next frames owe a catch-up for a hitch already
                    // given up on, and re-enter the spiral this guard exists for.
                    self.accumulated = 0;
                }
                (count as u32, owed - count)
            }
        };
        self.tick += u64::from(count);
        Due {
            first,
            count,
            dropped,
            // After the subtraction and after the guard, so it is what the clock
            // *kept*, never what arrived.
            covered: count as f32 + self.alpha(),
        }
    }

    /// What a frame that must not tick reports, whatever the pace (§6 M49).
    ///
    /// Not `advance(Duration::ZERO)`: [`Pace::Locked`] ignores elapsed time and
    /// would still owe its ticks, so the one spelling that works under every
    /// pace is to not charge the clock at all. Nothing is spent and nothing is
    /// dropped, so a realtime accumulator resumes exactly where it stopped
    /// rather than owing a catch-up for time nobody was watching — which is the
    /// difference between suspending a session and hitching it.
    pub fn hold(&self) -> Due {
        Due {
            first: self.tick,
            count: 0,
            dropped: 0,
            covered: self.alpha(),
        }
    }

    /// Resume at `tick` with an empty accumulator — what a restored snapshot
    /// (§4.8) does, since the tick count is captured state and the accumulator
    /// is not.
    pub fn resume_at(&mut self, tick: u64) {
        self.tick = tick;
        self.accumulated = 0;
    }
}
