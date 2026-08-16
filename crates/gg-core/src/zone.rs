//! Frame-scoped CPU zones (§4.8) — the half of a profile a script can read.
//!
//! [`frame`](crate::frame) has emitted the frame's top-level zones through
//! `profiling` since M4, and Tracy is what reads them. Tracy is also a GUI, so
//! no automated run and no measurement taken from a terminal can see a zone at
//! all — which is why §6 M25's regression was visible as a number and not as a
//! place for three milestones. The same macro therefore feeds two sinks:
//! `profiling` for the timeline, and this collector for anything that has to
//! *print* what it measured.
//!
//! Nesting is recorded as a **depth**, not a tree. A frame's zones arrive in
//! close order and the depth is what rebuilds the tree afterwards, which keeps
//! the hot path a push and a subtraction — building a tree during the frame
//! would allocate inside the thing it is timing.
//!
//! Off unless `cpu-timings` is on, where "off" means the guard is a zero-sized
//! struct with an empty `Drop` and [`take`] is an empty `Vec` — lab equipment,
//! and the shipping build pays nothing for it.
//!
//! It rides **no tier** and did not until §6 M58 have a reader either, which is
//! the shape of the defect that milestone found: the collector was built here at
//! M25, `gg-render` emitted into it, and no build in the tree turned it on, so
//! the frame's cost was unmeasurable by anything but a GUI profiler nobody had
//! attached. [`Profile`] is the reader; `xtask run --profile` is the build.

/// One closed zone. `depth` is how many zones were still open around it when it
/// closed, so `0` is a frame's outermost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    /// The literal the zone was opened with — always `'static`, which is what
    /// keeps a sample from owning a string it would have to allocate.
    pub name: &'static str,
    /// Zones still open around this one at close.
    pub depth: u16,
    /// Wall time between open and close.
    pub nanos: u64,
}

/// Open a zone that closes when the returned guard drops.
///
/// Prefer [`zone!`](crate::zone!), which opens one of these *and* a `profiling`
/// scope from one name — two sinks that cannot disagree about what they are
/// called.
#[must_use]
pub fn enter(name: &'static str) -> Guard {
    imp::enter(name)
}

/// Drain this thread's closed zones, leaving the collector empty for the next
/// frame. Call once a frame, after the work and before the next one opens.
#[must_use]
pub fn take() -> Vec<Sample> {
    imp::take()
}

/// Whether this build collects at all — `false` without `cpu-timings`, which is
/// what lets a reader say "no zones" rather than reporting an empty frame.
#[must_use]
pub const fn enabled() -> bool {
    cfg!(feature = "cpu-timings")
}

pub use imp::Guard;

/// A run's zones, accumulated (§6 M58).
///
/// [`take`] hands back one frame and forgets it, which is what a timeline wants
/// and useless to anyone asking where a *session's* milliseconds went. This is
/// the other reader: one row per name, fed every frame, printed once at exit.
/// Bounded by the number of distinct zone names — a session's cost here is a
/// linear scan of a list with a dozen entries, not a buffer that grows with the
/// run (§6 M51's lesson about a log that did).
#[derive(Debug, Default)]
pub struct Profile {
    rows: Vec<Row>,
    frames: u64,
}

/// One zone name's total across a run. `total` divided by [`Profile::frames`] is
/// the per-frame cost — the number a frame budget is argued from — and `worst` is
/// the single occurrence that would be felt as a hitch.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    /// The literal the zone was opened with.
    pub name: &'static str,
    /// The shallowest nesting this name was ever seen at — see
    /// [`Profile::absorb`].
    pub depth: u16,
    /// How many times it opened over the run, which is *not* the frame count: a
    /// frame owing three sim ticks opens `sim_tick` three times.
    pub calls: u64,
    /// Summed wall time, the numerator of the per-frame column.
    pub total_nanos: u64,
    /// The longest single occurrence — a hitch, where the mean is a budget.
    pub worst_nanos: u64,
}

impl Profile {
    /// Fold one frame's drained samples in. Call once a frame with [`take`]'s
    /// result; an empty slice still counts a frame, so a build with no collector
    /// reports zero rows over the right denominator rather than dividing by
    /// nothing.
    pub fn absorb(&mut self, samples: &[Sample]) {
        self.frames += 1;
        for s in samples {
            match self.rows.iter_mut().find(|r| r.name == s.name) {
                Some(row) => {
                    row.calls += 1;
                    row.total_nanos += s.nanos;
                    row.worst_nanos = row.worst_nanos.max(s.nanos);
                    // The shallowest sighting wins: a zone reached through two
                    // paths is one row, and reporting it at the deeper one would
                    // indent it under a parent it does not always have.
                    row.depth = row.depth.min(s.depth);
                }
                None => self.rows.push(Row {
                    name: s.name,
                    depth: s.depth,
                    calls: 1,
                    total_nanos: s.nanos,
                    worst_nanos: s.nanos,
                }),
            }
        }
    }

    /// Frames absorbed — the denominator, and `0` before the first one.
    #[must_use]
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// Say where the session's milliseconds went, at `info` and **once**.
    ///
    /// Here rather than in the shell because the shape of the table is this
    /// type's business, and because §3's line budget for `gg-runtime` is a
    /// budget on the *shell* — a formatter for someone else's struct was never
    /// what those lines were for.
    ///
    /// At exit and never per frame: §6 M51's finding was a log that grew with
    /// the session, and a table printed every second is that same defect with a
    /// better excuse. Silent without `cpu-timings`, and silent before the first
    /// frame — an empty table reads as "nothing costs anything".
    pub fn report(&self) {
        if !enabled() || self.frames == 0 {
            return;
        }
        #[expect(
            clippy::cast_precision_loss,
            reason = "nanoseconds and a frame count, printed to three decimals"
        )]
        let ms = |nanos: u64| nanos as f64 / 1e6;
        let frames = self.frames as f64;
        tracing::info!(frames = self.frames, "frame profile — ms per frame");
        for row in self.rows() {
            // Two columns that fail in opposite directions: the mean is a
            // budget, and the worst single call is a hitch a mean forgives.
            tracing::info!(
                zone = row.name,
                depth = row.depth,
                per_frame = format!("{:.3}", ms(row.total_nanos) / frames),
                worst = format!("{:.3}", ms(row.worst_nanos)),
                calls = row.calls,
                "zone"
            );
        }
    }

    /// The rows, deepest-cost first within each depth. Sorted on read rather
    /// than kept sorted: this runs once at exit and [`Profile::absorb`] runs
    /// every frame.
    #[must_use]
    pub fn rows(&self) -> Vec<Row> {
        let mut rows = self.rows.clone();
        rows.sort_by(|a, b| {
            a.depth
                .cmp(&b.depth)
                .then(b.total_nanos.cmp(&a.total_nanos))
        });
        rows
    }
}

/// Open a CPU zone for the rest of the enclosing block, in both sinks.
///
/// Takes a literal, not an expression: `profiling::scope!` cannot take a dynamic
/// name and a [`Sample`] holds a `&'static str`, so the restriction is the two
/// sinks agreeing rather than this macro's own. A dynamic name is what
/// `gg_ecs::boundary::set_system_zone` exists for.
#[macro_export]
macro_rules! zone {
    ($name:literal) => {
        // The `profiling` sink first, so the Tracy span brackets the collector's
        // — guards drop in reverse, which puts the cheaper one inside.
        $crate::profiling::scope!($name);
        let _gg_zone = $crate::zone::enter($name);
    };
}

#[cfg(feature = "cpu-timings")]
mod imp {
    use std::cell::RefCell;
    use std::time::Instant;

    use super::Sample;

    thread_local! {
        static STATE: RefCell<State> = const { RefCell::new(State::new()) };
    }

    /// Open zones as a stack, closed ones in close order. One frame's worth:
    /// [`take`](super::take) is what bounds the growth, and a host that never
    /// drains is holding a leak it can see.
    struct State {
        open: Vec<(&'static str, Instant)>,
        closed: Vec<Sample>,
    }

    impl State {
        const fn new() -> Self {
            State {
                open: Vec::new(),
                closed: Vec::new(),
            }
        }
    }

    /// Closes its zone on drop. `armed` is false when the collector was already
    /// borrowed — impossible on the frame path, and a dropped sample rather than
    /// a panic if it ever happens, because an instrument must not be the thing
    /// that takes the process down.
    pub struct Guard {
        armed: bool,
    }

    pub fn enter(name: &'static str) -> Guard {
        let armed = STATE.with(|state| {
            let Ok(mut state) = state.try_borrow_mut() else {
                return false;
            };
            state.open.push((name, Instant::now()));
            true
        });
        Guard { armed }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            if !self.armed {
                return;
            }
            STATE.with(|state| {
                let Ok(mut state) = state.try_borrow_mut() else {
                    return;
                };
                let Some((name, start)) = state.open.pop() else {
                    return;
                };
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "a zone longer than 584 years is not a zone; depth is single digits"
                )]
                let sample = Sample {
                    name,
                    depth: state.open.len() as u16,
                    nanos: start.elapsed().as_nanos() as u64,
                };
                state.closed.push(sample);
            });
        }
    }

    pub fn take() -> Vec<Sample> {
        STATE.with(|state| {
            state
                .try_borrow_mut()
                .map(|mut state| core::mem::take(&mut state.closed))
                .unwrap_or_default()
        })
    }
}

#[cfg(not(feature = "cpu-timings"))]
mod imp {
    use super::Sample;

    /// Zero-sized and empty-bodied: without the feature a zone is the name and
    /// nothing else, and this compiles out entirely.
    pub struct Guard;

    pub const fn enter(_name: &'static str) -> Guard {
        Guard
    }

    pub fn take() -> Vec<Sample> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    // The collector is thread-local and these drain it, so they must not
    // interleave on one thread; nextest runs a process per test binary and a
    // thread per test, which is what keeps them apart.
    use super::{enter, take};

    #[test]
    fn nesting_closes_inner_first_and_records_its_depth() {
        let _ = take();
        {
            let _outer = enter("outer");
            {
                let _inner = enter("inner");
            }
        }
        let samples = take();
        if !super::enabled() {
            assert!(samples.is_empty(), "no feature, no samples");
            return;
        }
        // Close order, not open order: the inner zone ends first and is the
        // first sample, which is what a reader rebuilds the tree from.
        let shape: Vec<(&str, u16)> = samples.iter().map(|s| (s.name, s.depth)).collect();
        assert_eq!(shape, [("inner", 1), ("outer", 0)]);
    }

    #[test]
    fn a_profile_divides_by_frames_and_not_by_calls() {
        // The distinction the report is built on: a frame owing three sim ticks
        // opens `sim_tick` three times, and a per-frame budget wants the sum
        // over *frames*. Fed by hand rather than through the collector so the
        // arithmetic is checked in a build without the feature too.
        let sample = |name, depth, nanos| super::Sample { name, depth, nanos };
        let mut profile = super::Profile::default();
        profile.absorb(&[sample("sim_tick", 0, 1_000_000)]);
        profile.absorb(&[
            sample("sim_tick", 0, 2_000_000),
            sample("sim_tick", 0, 3_000_000),
            sample("render", 0, 500_000),
        ]);
        assert_eq!(profile.frames(), 2);
        // Cost order within a depth, which is what makes the table readable —
        // and it is what puts `sim_tick` first, so the row is reached by index
        // rather than by a search this crate's lints would make an `expect`.
        let rows = profile.rows();
        let shape: Vec<(&str, u64, u64, u64)> = rows
            .iter()
            .map(|r| (r.name, r.calls, r.total_nanos, r.worst_nanos))
            .collect();
        assert_eq!(
            shape,
            [
                ("sim_tick", 3, 6_000_000, 3_000_000),
                ("render", 1, 500_000, 500_000),
            ]
        );
    }

    #[test]
    fn a_zone_seen_at_two_depths_is_reported_at_the_shallower_one() {
        // `render.frame` is depth 1 under the loop's `render` and depth 0 to an
        // instrument calling the renderer directly. One row either way, indented
        // where it is always at least that shallow.
        let sample = |depth, nanos| super::Sample {
            name: "render.frame",
            depth,
            nanos,
        };
        let mut profile = super::Profile::default();
        profile.absorb(&[sample(2, 10)]);
        profile.absorb(&[sample(1, 10)]);
        assert_eq!(profile.rows()[0].depth, 1);
    }

    #[test]
    fn taking_leaves_the_collector_empty() {
        let _ = take();
        drop(enter("once"));
        let first = take();
        assert_eq!(first.len(), usize::from(super::enabled()));
        assert!(take().is_empty(), "a drained collector re-fills from empty");
    }
}
