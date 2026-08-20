//! The render scale nobody chose (§6 M80).
//!
//! §6 M78 built `r.scale` and proved it buys almost the whole frame on a
//! fill-bound one, and §6 M79 made the per-pass table true enough to say so.
//! Neither of them turns it. A player who downloads the folder `xtask ship`
//! assembles and never opens a menu gets whatever their machine can do, which on
//! the integrated Radeon in this desk is nineteen frames a second at 1080p.
//!
//! This is the part that turns it, and it is a control loop, so the thing to be
//! careful about is not the arithmetic but the **feedback**: the quantity being
//! measured is changed by the act of responding to it. A controller that decides
//! from a window straddling its own last move is reading half its own output,
//! and what that looks like on a screen is a picture that breathes.
//!
//! The whole defence is one rule — **a decision is only ever taken over frames
//! measured at the setting currently in force**. Moving the scale clears the
//! window, and no decision happens until it has refilled. There is no separate
//! cooldown counter, no damping term and no gain to tune: the window *is* the
//! cooldown, and it is the shortest one that is honest.
//!
//! §6 M81 found that rule enforced over half its subject: the scale and the
//! **window's own extent** multiply into one pixel count, and only the first of
//! them cleared. What that costs is a resize drag — Win32 runs one inside
//! `DefWindowProc`'s modal loop, so `Event::Frame` stops and the only
//! presentations are `Event::Resized`'s, which arrive when the *hand* moves.
//! Thirty of those are a window full of the mouse's cadence, and `gg-tools
//! governor`'s `drag` trace read a machine that misses no frame at all
//! (`late 0.0 %`) taken to the floor and still short of full scale twelve
//! hundred frames later. So the extent is handed in and a change to it is
//! treated exactly as a move is, because it *is* one.
//!
//! What is deliberately absent: this never looks at a GPU timer. `gpu-timings`
//! is in no shipping tier (it is `tier-dev` and `tier-instrumented` only), so a
//! governor built on `pass_timings` would be a feature that does nothing in the
//! build a player runs. Wall clock is the one measurement every tier has.

use std::time::{Duration, Instant};

/// Frames per decision. Long enough that a median means something and one
/// stalled frame cannot move it, short enough that a machine which cannot keep
/// up is helped within half a second at any rate worth protecting.
const WINDOW: usize = 30;

/// How far down the scale may be taken. Half of each axis is a quarter of the
/// pixels, which is the largest change worth making without asking — below it
/// the picture is the complaint rather than the frame rate, and that is a
/// judgement a player makes in a menu (§6 M78's row).
const FLOOR: f64 = 0.5;

/// The most the scale may be *given back* in one move. There is no matching cap
/// on the way down and that asymmetry is the whole policy: being slow is a
/// complaint and having headroom is not, and a scale handed back too eagerly is
/// one that has to be taken away again, which is the oscillation this avoids.
///
/// `gg-tools governor` is what set it. A fixed step in *both* directions took
/// seven moves and about three seconds to converge on a machine that was slow
/// from its first frame — thirty frames per step, walking somewhere the
/// arithmetic below could name in one — which is why only this direction still
/// takes small ones.
const UP_MAX: f64 = 1.10;

/// How far past the target a frame has to run before it is worth doing anything
/// about, and how much headroom has to be spare before the scale is given back.
/// The gap between them is the dead band, and it is wide on the way up because
/// an increase that has to be undone is the oscillation this exists to avoid.
const SLOW: f64 = 1.05;
const SPARE: f64 = 0.80;

/// Where a correction aims, as a fraction of the target — the **middle of the
/// dead band** rather than its edge. Aiming at the target itself parks the
/// frame just the wrong side of it: the correction ignores the fixed cost (see
/// below) and therefore undershoots, so a player who asked for 60 Hz was held
/// at 57.3 forever, inside the band and never quite right. Landing in the
/// middle leaves neither trigger near, which is what a dead band is for.
const AIM: f64 = 0.92;

/// The host's automatic render scale (§6 M80).
///
/// Host state and nothing else: it is driven by a wall clock, so it must never
/// reach the world. It does not — [`Governor::factor`] is multiplied into the
/// renderer's scale, which is not a component, is in no save and is in no
/// canonical hash. The one field a *player* owns is `Prefs::scale_auto`, which
/// is hashed like every other preference and is the rate handed in below.
#[derive(Debug)]
pub struct Governor {
    /// Frame periods in microseconds, oldest first. Cleared on every move.
    window: Vec<u32>,
    factor: f64,
    /// When [`Governor::observe`] last ran. Held here rather than by the caller
    /// because it is this type's clock and nothing else reads it — a shell that
    /// kept it would be keeping a field on behalf of one method.
    last: Option<Instant>,
    /// The extent the previous frame was measured at. `None` until one has been.
    extent: Option<(u32, u32)>,
}

impl Default for Governor {
    fn default() -> Self {
        Self::new()
    }
}

impl Governor {
    /// Idle: a full scale and an empty window.
    #[must_use]
    pub fn new() -> Self {
        Self {
            window: Vec::with_capacity(WINDOW),
            factor: 1.0,
            last: None,
            extent: None,
        }
    }

    /// The factor to multiply the chosen render scale by. `1.0` until this has
    /// seen a full window of frames it is allowed to judge.
    #[must_use]
    pub fn factor(&self) -> f64 {
        self.factor
    }

    /// Whether the scale is currently being held below what was asked for —
    /// what a menu row or an overlay says out loud, so a player wondering why
    /// the picture is soft is not left to guess.
    #[must_use]
    pub fn engaged(&self) -> bool {
        self.factor < 1.0
    }

    /// One frame, timed against the last one this saw, at the extent it was
    /// drawn into, and the rate the player asked to protect. The shipping entry
    /// point — [`observe_at`](Self::observe_at) is the same decision over a
    /// period somebody else measured, which is what a test and `gg-tools
    /// governor` drive.
    ///
    /// The first call after a gap measures nothing: the interval across bring-up
    /// is a pipeline cache and a swapchain rather than a frame, and handing it
    /// to a controller would soften every session's first second. The extent is
    /// still recorded on that path, so the frame after it is compared against
    /// what was actually on screen rather than against a stale size.
    pub fn observe(&mut self, hz: u32, extent: (u32, u32)) -> bool {
        let now = Instant::now();
        match self.last.replace(now) {
            Some(last) => self.observe_at(now - last, hz, extent),
            None => {
                self.extent = Some(extent);
                false
            }
        }
    }

    /// One frame's wall period, and the rate the player asked to protect.
    ///
    /// `hz` of zero is off, and off means **all the way off**: the factor goes
    /// back to 1.0 rather than freezing wherever it happened to be, because a
    /// setting switched off that leaves the picture soft is a bug report nobody
    /// can act on.
    ///
    /// Returns whether the factor moved, which is what the caller logs.
    ///
    /// A frame drawn at a different `extent` than the one before it is not
    /// comparable to it and is discarded — the window and the clock both, since
    /// the interval *into* a resize is no more a frame period than the interval
    /// out of it. This is the move rule and not a second one: a scale and an
    /// extent multiply into the same pixel count. Deliberately *not* extended to
    /// `Prefs::scale`, which is a player deciding what they want rather than a
    /// swapchain being rebuilt under a measurement; re-converging over one
    /// window after that is the design (§6 M80) and not a defect.
    pub fn observe_at(&mut self, elapsed: Duration, hz: u32, extent: (u32, u32)) -> bool {
        // Changed, not merely unknown: a caller handing a period has already
        // seen two frames, and demanding it repeat the size before anything
        // counts would cost every direct driver a frame for nothing.
        if self.extent.replace(extent).is_some_and(|was| was != extent) {
            self.window.clear();
            self.last = None;
            return false;
        }
        let Some(target_us) = (match hz {
            0 => None,
            hz => Some(1.0e6 / f64::from(hz)),
        }) else {
            let was = self.factor;
            self.window.clear();
            self.factor = 1.0;
            return was != 1.0;
        };

        // Saturating rather than wrapping: a frame longer than an hour is a
        // debugger breakpoint, and it should read as "very slow" rather than as
        // a small number.
        let micros = u32::try_from(elapsed.as_micros()).unwrap_or(u32::MAX);
        self.window.push(micros);
        if self.window.len() < WINDOW {
            return false;
        }

        let median = median_us(&mut self.window);
        // The step is **proportional and the plant is known**: a frame costs
        // roughly `floor + fill * scale²`, so the scale that would have hit a
        // target is `scale * sqrt(target / measured)`. Ignoring the floor is
        // deliberate and is what makes this safe — the floor does not shrink
        // with the pixel count, so a correction computed without it always
        // *undershoots*, and a controller that undershoots converges from one
        // side instead of ringing. There is no gain to tune here; the square
        // root is the plant's own shape.
        let aim = |want: f64| (self.factor * (want / median).sqrt()).clamp(FLOOR, 1.0);
        let moved = match median {
            // Too slow, and there is room to go down.
            m if m > target_us * SLOW && self.factor > FLOOR => {
                self.factor = aim(target_us * AIM);
                true
            }
            // Comfortably inside the budget, and something was taken away
            // earlier that can be given back. The same aim point — the
            // asymmetry between the two directions is the cap and the triggers,
            // not where they are heading, which keeps one number to reason
            // about instead of two.
            m if m < target_us * SPARE && self.factor < 1.0 => {
                self.factor = aim(target_us * AIM).min(self.factor * UP_MAX);
                true
            }
            _ => false,
        };
        // The rule this whole type exists for: a window that straddles a move
        // is half measurements of a setting no longer in force. Cleared on a
        // move so the next decision is taken over frames this factor produced,
        // and *kept* when nothing moved so a steady machine decides every frame
        // rather than every thirtieth.
        match moved {
            true => self.window.clear(),
            false => {
                self.window.remove(0);
            }
        }
        moved
    }
}

/// Median of the window. Takes `&mut` and sorts a scratch copy rather than the
/// window itself — the window is a queue and its order is what makes the oldest
/// sample droppable.
fn median_us(window: &mut [u32]) -> f64 {
    let mut sorted: Vec<u32> = window.to_vec();
    sorted.sort_unstable();
    f64::from(sorted[sorted.len() / 2])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// A window that is not being dragged, which is every test below but one.
    const EXTENT: (u32, u32) = (1920, 1080);

    fn ms(n: f64) -> Duration {
        Duration::from_micros((n * 1000.0) as u64)
    }

    /// A machine that is keeping up is a machine this never touches — the
    /// property every gate in the tree depends on, since a headless or locked
    /// session charges a perfect `1/hz` every frame.
    #[test]
    fn a_machine_that_keeps_up_is_never_touched() {
        let mut g = Governor::new();
        for _ in 0..500 {
            assert!(!g.observe_at(ms(16.6), 60, EXTENT));
        }
        assert_eq!(g.factor(), 1.0);
        assert!(!g.engaged());
    }

    /// Off is off, and it puts back what it took. A preference switched off
    /// that leaves the picture soft is the bug this asserts against.
    #[test]
    fn switching_it_off_gives_the_scale_back() {
        let mut g = Governor::new();
        for _ in 0..200 {
            g.observe_at(ms(50.0), 60, EXTENT);
        }
        assert!(g.engaged(), "50 ms frames against a 16.6 ms target");
        assert!(
            g.observe_at(ms(50.0), 0, EXTENT),
            "turning it off is a move"
        );
        assert_eq!(g.factor(), 1.0);
    }

    /// The floor is a floor. A machine that cannot reach the target at a
    /// quarter of the pixels is one this stops arguing with, rather than
    /// driving the picture to nothing in pursuit of a rate it will never make.
    #[test]
    fn a_hopeless_machine_stops_at_the_floor() {
        let mut g = Governor::new();
        for _ in 0..2000 {
            g.observe_at(ms(500.0), 60, EXTENT);
        }
        assert_eq!(g.factor(), FLOOR);
    }

    /// One stalled frame in a healthy window moves nothing. A shader compile, a
    /// page fault or an alt-tab is not a slow machine, and a controller that
    /// reacted to the maximum rather than the median would drop the resolution
    /// every time one happened.
    #[test]
    fn a_single_hitch_is_not_a_slow_machine() {
        let mut g = Governor::new();
        for i in 0..600 {
            let period = match i % 60 {
                0 => ms(300.0),
                _ => ms(16.0),
            };
            assert!(!g.observe_at(period, 60, EXTENT), "moved on frame {i}");
        }
        assert_eq!(g.factor(), 1.0);
    }

    /// The rule the type exists for: no decision is ever taken over frames
    /// measured at a setting no longer in force. Asserted by construction —
    /// after a move the window is empty, so the next `WINDOW - 1` frames cannot
    /// move it however extreme they are.
    #[test]
    fn a_decision_is_never_taken_over_its_own_last_move() {
        let mut g = Governor::new();
        for _ in 0..WINDOW {
            g.observe_at(ms(50.0), 60, EXTENT);
        }
        let after = g.factor();
        assert!(after < 1.0, "the first window should have moved it");
        for i in 1..WINDOW {
            assert!(
                !g.observe_at(ms(50.0), 60, EXTENT),
                "moved again {i} frames into a window it had just cleared"
            );
            assert_eq!(g.factor(), after);
        }
        assert!(
            g.observe_at(ms(50.0), 60, EXTENT),
            "and moves on the frame it refills"
        );
    }

    /// Down is not the same move as up. Down aims at the target and up aims
    /// short of it, and up is additionally capped — so a scale is taken away in
    /// as few moves as the arithmetic allows and handed back in as many as
    /// caution wants.
    #[test]
    fn it_takes_more_than_it_gives_back() {
        const { assert!(SPARE < 1.0 / SLOW, "the dead band must not be empty") };
        let mut down = Governor::new();
        for _ in 0..WINDOW {
            down.observe_at(ms(60.0), 60, EXTENT);
        }
        // One window, one move, and most of the way there: 16.6 of 60 ms is a
        // factor of 0.527 and the arithmetic should find it in a single step.
        assert!(
            (down.factor() - (16.666_f64 * AIM / 60.0).sqrt()).abs() < 0.01,
            "one step should aim at the dead band's middle, not creep toward it: {}",
            down.factor()
        );

        let mut up = Governor::new();
        for _ in 0..WINDOW {
            up.observe_at(ms(60.0), 60, EXTENT);
        }
        let low = up.factor();
        for _ in 0..WINDOW {
            up.observe_at(ms(1.0), 60, EXTENT);
        }
        assert!(
            up.factor() <= low * UP_MAX + 1e-9,
            "a whole second of idle frames must not hand the scale back at once"
        );
    }

    /// A resize drag is the hand's cadence, not the machine's (§6 M81). Win32
    /// stops the frame loop for the length of one, so every presentation in it
    /// is an `Event::Resized` paint arriving when the mouse moved — and the
    /// machine here is a fast one that would never otherwise be touched.
    ///
    /// The window is what makes this different from a hitch: sixty consecutive
    /// slow paints are a *median* of slow paints, which is exactly what the
    /// median rule cannot save. What saves it is that each one is a different
    /// size.
    #[test]
    fn a_window_being_dragged_is_not_a_slow_machine() {
        let mut g = Governor::new();
        for i in 0..600 {
            let dragging = (300..360).contains(&i);
            let (period, extent) = match dragging {
                true => (ms(120.0), (1920 + i as u32, 1080)),
                false => (ms(4.0), EXTENT),
            };
            assert!(!g.observe_at(period, 60, extent), "moved on frame {i}");
        }
        assert_eq!(g.factor(), 1.0);
    }

    /// The same paints at one size *do* move it, which is what says the test
    /// above is the extent's doing and not the period's. A window resized once
    /// and then held while the machine crawls is a slow machine.
    #[test]
    fn a_window_that_stopped_moving_is_measured_again() {
        let mut g = Governor::new();
        let mut moved = false;
        for i in 0..600 {
            let period = match (300..360).contains(&i) {
                true => ms(120.0),
                false => ms(4.0),
            };
            // Whether it *ever* moved, not where it ended: the fast frames
            // after the slow stretch hand the scale back, so the final state
            // forgives exactly the defect this is the control for.
            moved |= g.observe_at(period, 60, EXTENT);
        }
        assert!(moved, "sixty slow frames at one size are sixty frames");
    }

    /// A resize clears the window rather than merely skipping one sample —
    /// the move rule, over the other half of the pixel count. Without it a
    /// drag's last paint would be judged together with the frames from before
    /// the window ever changed size.
    #[test]
    fn a_resize_clears_the_window_the_way_a_move_does() {
        let mut g = Governor::new();
        for _ in 0..WINDOW - 1 {
            g.observe_at(ms(50.0), 60, EXTENT);
        }
        // The frame the size changed on, which a window filled before it would
        // otherwise decide from on the spot.
        assert!(
            !g.observe_at(ms(50.0), 60, (1280, 720)),
            "decided on the resize frame itself, over samples from another size"
        );
        for i in 1..WINDOW {
            assert!(
                !g.observe_at(ms(50.0), 60, (1280, 720)),
                "decided {i} frames after a resize, over samples from before it"
            );
        }
        assert!(
            g.observe_at(ms(50.0), 60, (1280, 720)),
            "and decides on the frame the new size refills it"
        );
    }

    /// A machine that is only just too slow gets a small correction, and one
    /// that is hopelessly too slow gets a large one — from the same rule, with
    /// nothing between them to tune.
    #[test]
    fn the_step_is_the_size_of_the_problem() {
        let step = |frame_ms: f64| {
            let mut g = Governor::new();
            for _ in 0..WINDOW {
                g.observe_at(ms(frame_ms), 60, EXTENT);
            }
            g.factor()
        };
        let (mild, severe) = (step(20.0), step(200.0));
        // 20 ms against a 16.6 ms budget, aimed at the band's middle, is
        // `sqrt(16.666 * AIM / 20)` — a nudge, from the same rule that sends the
        // row below straight to the floor in one move.
        assert!(
            mild > 0.85,
            "20 ms against a 16.6 ms budget is a nudge: {mild}"
        );
        assert!(
            severe <= FLOOR + 1e-9,
            "200 ms is the floor in one move: {severe}"
        );
        assert!(mild > severe);
    }
}
