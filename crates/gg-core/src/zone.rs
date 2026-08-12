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
//! struct with an empty `Drop` and [`take`] is an empty `Vec`. The feature rides
//! the instrumented tier beside `gpu-timings`, for that feature's reason: the
//! numbers are lab equipment and the shipping build pays nothing for them.

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
    fn taking_leaves_the_collector_empty() {
        let _ = take();
        drop(enter("once"));
        let first = take();
        assert_eq!(first.len(), usize::from(super::enabled()));
        assert!(take().is_empty(), "a drained collector re-fills from empty");
    }
}
