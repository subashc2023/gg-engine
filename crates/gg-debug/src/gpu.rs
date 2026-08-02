//! Tracy's GPU zones, fed from the same query pool the overlay reads (§4.8).
//!
//! Measured once, reported twice: if Tracy asked the device separately the two
//! views of one frame could disagree, and the milestone's exit criterion is that
//! they agree. So this takes [`PassTiming`]s that have already come back and
//! turns each into a zone — a frame or two after the fact, which costs nothing
//! because a zone is placed by its *GPU* timestamps and not by when it was sent.

use gg_rhi::{GpuClock, PassTiming};

/// Frames between clock resyncs. AMD parts reset their timestamp counter
/// aggressively in low-power states, which desynchronises Tracy's CPU/GPU
/// alignment without any error to notice; re-anchoring about once a second
/// costs one host-side query and repairs it (`sync_gpu_time`).
#[cfg(feature = "tracy")]
const RESYNC_FRAMES: u32 = 60;

/// A Tracy GPU context and the state needed to feed it in order.
pub struct GpuZones {
    #[cfg(feature = "tracy")]
    context: tracy_client::GpuContext,
    /// The last tick uploaded. Tracy requires timestamps to arrive
    /// monotonically, and two passes the GPU overlapped would otherwise be
    /// uploaded out of order and drawn as nonsense.
    watermark: i64,
    resync_in: u32,
}

impl GpuZones {
    /// Open a context anchored at `clock`. `None` when this build has no Tracy
    /// or the device has no readable clock — both of which cost a pane in a
    /// profiler and nothing else.
    #[cfg(feature = "tracy")]
    pub fn new(name: &str, clock: GpuClock) -> Option<Self> {
        let context = tracy_client::Client::running()?
            .new_gpu_context(
                Some(name),
                tracy_client::GpuContextType::Vulkan,
                clock.ticks,
                clock.period_ns,
            )
            .ok()?;
        Some(Self {
            context,
            watermark: clock.ticks,
            resync_in: RESYNC_FRAMES,
        })
    }

    /// Without Tracy there is no context to open. Separate bodies rather than a
    /// `cfg!` because the context *field* cannot exist without the dependency.
    #[cfg(not(feature = "tracy"))]
    pub fn new(_name: &str, _clock: GpuClock) -> Option<Self> {
        None
    }

    /// Emit one zone per pass of a frame that has come back, and re-anchor the
    /// clock every so often. `clock` is what the re-anchor uses; passing `None`
    /// skips it, which only costs alignment.
    pub fn frame(&mut self, passes: &[PassTiming], clock: Option<GpuClock>) {
        #[cfg(feature = "tracy")]
        {
            for pass in passes {
                let (begin, end) = ordered(pass, self.watermark);
                let Ok(mut span) =
                    self.context
                        .span_alloc(&pass.name, "Renderer::frame", file!(), line!())
                else {
                    // The only failure is 32767 spans awaiting data, which means
                    // the server is gone. Dropping the frame is the right answer.
                    return;
                };
                span.end_zone();
                span.upload_timestamp_start(begin);
                span.upload_timestamp_end(end);
                self.watermark = end;
            }
            self.resync_in = self.resync_in.saturating_sub(1);
            if self.resync_in == 0 {
                self.resync_in = RESYNC_FRAMES;
                if let Some(clock) = clock {
                    self.context.sync_gpu_time(clock.ticks);
                }
            }
        }
        #[cfg(not(feature = "tracy"))]
        {
            let _ = (passes, clock, &mut self.watermark, &mut self.resync_in);
        }
    }
}

/// A pass's ticks, forced into the non-decreasing order Tracy requires.
///
/// Clamped rather than dropped: a pass whose start the GPU overlapped with the
/// previous one still happened, and its *duration* — the number the overlay
/// shows — is what the clamp preserves at the cost of where it sits.
///
/// `cfg`-ed on the tests as well as on Tracy: nothing in a Tracy-less library
/// calls it, but it is the one piece of that path that can be wrong without
/// anything failing, so its tests run in every build.
#[cfg(any(feature = "tracy", test))]
fn ordered(pass: &PassTiming, watermark: i64) -> (i64, i64) {
    let begin = pass.begin.max(watermark);
    (begin, pass.end.max(begin))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass(begin: i64, end: i64) -> PassTiming {
        PassTiming {
            name: "p".into(),
            gpu_ms: 0.0,
            begin,
            end,
        }
    }

    #[test]
    fn ordinary_passes_pass_through_untouched() {
        assert_eq!(ordered(&pass(100, 200), 50), (100, 200));
    }

    #[test]
    fn a_pass_that_started_before_the_last_one_ended_is_pushed_forward() {
        // Overlapping passes are what a GPU does when barriers allow it; Tracy
        // requires monotonic uploads and would otherwise draw the frame as
        // nonsense rather than reject it.
        assert_eq!(ordered(&pass(80, 200), 100), (100, 200));
    }

    #[test]
    fn a_pass_entirely_behind_the_watermark_collapses_rather_than_inverting() {
        // A zero-length zone at the right place beats a negative-length one,
        // which is what a wrapped counter would otherwise produce.
        assert_eq!(ordered(&pass(10, 20), 100), (100, 100));
    }
}
