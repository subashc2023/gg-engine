//! Per-pass GPU timestamps (§4.8) — where the debug overlay's milliseconds and
//! Tracy's GPU zones both come from, measured once so the two cannot disagree.
//!
//! Two timestamps bracket each pass's recorded commands. They are read back a
//! frame later, when the graphics timeline has already proven that frame
//! retired: a query pool read that has to *wait* is a stall the profiler
//! introduced, which would make the profiler the thing being profiled.

use crate::RhiError;
use crate::device::Device;
use ash::vk;

/// One pass's measured GPU time.
#[derive(Clone, Debug)]
pub struct PassTiming {
    /// The name the graph declared the pass under.
    pub name: String,
    /// Device time between the pass's first and last command.
    pub gpu_ms: f32,
    /// The raw device ticks the pass opened and closed at, masked to the bits
    /// the queue actually writes. The overlay wants [`Self::gpu_ms`]; a
    /// timeline profiler wants *where* the pass sat, which a duration cannot
    /// say — see [`GpuClock`] for what puts these on a CPU timeline.
    pub begin: i64,
    /// See [`Self::begin`].
    pub end: i64,
}

/// The device clock, read host-side, for anchoring GPU spans to CPU time
/// (§4.8). `ticks` is in the same units a [`PassTiming`] reports and
/// `period_ns` converts them; both come from the device rather than a constant,
/// because the period is a per-device fact (1 ns on lavapipe and on an RTX
/// 4090, 10 ns on a Radeon iGPU).
#[derive(Clone, Copy, Debug)]
pub struct GpuClock {
    /// The device's timestamp counter at the moment of the call.
    pub ticks: i64,
    /// Nanoseconds one tick represents.
    pub period_ns: f32,
}

/// Passes timed per frame. The v1 list is four (§4.5); a graph past this is
/// timed as far as the pool reaches and says so in the log rather than
/// silently reporting a prefix as the whole frame.
const MAX_TIMED_PASSES: usize = 32;

/// Two queries per pass.
const QUERIES_PER_SLOT: u32 = (MAX_TIMED_PASSES * 2) as u32;

/// The pool and the range of it one frame-in-flight slot owns. Copied into
/// recording, which is why it is small and `Copy`: the recorder must not need a
/// borrow of the whole [`Timings`].
#[derive(Clone, Copy)]
pub(crate) struct Stamps {
    pool: vk::QueryPool,
    base: u32,
}

impl Stamps {
    /// Reset this slot's whole range. Recorded rather than host-side: the reset
    /// must be ordered against the writes on the device, and `hostQueryReset`
    /// would order it against nothing.
    ///
    /// # Safety
    /// `cmd` must be recording.
    pub(crate) unsafe fn reset(&self, device: &Device, cmd: vk::CommandBuffer) {
        // SAFETY: caller contract; the range is this slot's own.
        unsafe {
            device
                .raw()
                .cmd_reset_query_pool(cmd, self.pool, self.base, QUERIES_PER_SLOT)
        };
    }

    /// Whether pass `index` is inside the pool's reach.
    pub(crate) fn covers(&self, index: usize) -> bool {
        index < MAX_TIMED_PASSES
    }

    /// # Safety
    /// `cmd` must be recording and `index` must satisfy [`Stamps::covers`].
    pub(crate) unsafe fn begin(&self, device: &Device, cmd: vk::CommandBuffer, index: usize) {
        // SAFETY: caller contract.
        unsafe { self.write(device, cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, index * 2) };
    }

    /// # Safety
    /// As [`Stamps::begin`], and `begin` must have been called for `index`.
    pub(crate) unsafe fn end(&self, device: &Device, cmd: vk::CommandBuffer, index: usize) {
        // SAFETY: caller contract.
        unsafe {
            self.write(
                device,
                cmd,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                index * 2 + 1,
            )
        };
    }

    /// # Safety
    /// `cmd` is recording and `offset` is within this slot's range.
    unsafe fn write(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        stage: vk::PipelineStageFlags2,
        offset: usize,
    ) {
        // SAFETY: caller contract; the query index is this slot's own.
        unsafe {
            device
                .raw()
                .cmd_write_timestamp2(cmd, stage, self.pool, self.base + offset as u32)
        };
    }
}

/// The query pool, one slot's worth of names per frame in flight, and the last
/// frame's readings.
pub(crate) struct Timings {
    pool: vk::QueryPool,
    /// Nanoseconds per tick.
    period: f32,
    /// Which bits of a reading are real (§4.8: some queues report fewer than 64).
    mask: u64,
    /// Per slot: the names recorded there, empty until a *submitted* frame has
    /// timed that slot — which is what [`Timings::collect`] keys off.
    pending: Vec<Vec<String>>,
    /// The most recent completed frame's per-pass times.
    last: Vec<PassTiming>,
}

impl Timings {
    /// Create the pool, or `None` when this build or this device does not time
    /// passes.
    ///
    /// `cfg!` rather than `#[cfg]` on purpose: the branch folds to a constant
    /// and LLVM drops the rest, so dist creates no pool and allocates no names
    /// (§2, lab equipment unbolts) without this module growing two shapes that
    /// have to be kept in step.
    pub(crate) fn new(device: &Device, slots: usize) -> Result<Option<Self>, RhiError> {
        if !cfg!(feature = "gpu-timings") || device.timestamp_mask() == 0 {
            return Ok(None);
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(QUERIES_PER_SLOT * slots as u32);
        // SAFETY: device is live; info is fully initialized.
        let pool = unsafe { device.raw().create_query_pool(&info, None) }.map_err(RhiError::Vk)?;
        device.set_name(pool, "gg.timings.pool");
        Ok(Some(Self {
            pool,
            period: device.timestamp_period(),
            mask: device.timestamp_mask(),
            pending: vec![Vec::new(); slots],
            last: Vec::new(),
        }))
    }

    /// This slot's write range.
    pub(crate) fn stamps(&self, slot: usize) -> Stamps {
        Stamps {
            pool: self.pool,
            base: QUERIES_PER_SLOT * slot as u32,
        }
    }

    /// Record which passes `slot` is timing.
    ///
    /// # Contract
    /// A registration must not survive a frame that never **submitted**:
    /// [`Timings::collect`] reads a slot on the strength of it, and a frame that
    /// aborted before recording reset none of these queries and wrote none of
    /// them. `Rhi::execute` therefore registers after its submit — its abort
    /// path retries onto the same slot — while `OffscreenRhi` brackets one
    /// synchronous submit between this call and its collect.
    pub(crate) fn expect(&mut self, slot: usize, names: impl Iterator<Item = String>) {
        let pending = &mut self.pending[slot];
        pending.clear();
        pending.extend(names.take(MAX_TIMED_PASSES));
    }

    /// Read back the frame that last used `slot`, which the caller has already
    /// proven retired. A slot no submitted frame has timed reads nothing —
    /// which is the whole guard against reading a query that was never written.
    ///
    /// Errors are logged and swallowed: a profiler that takes the frame down
    /// when a query misbehaves is worse than no profiler.
    pub(crate) fn collect(&mut self, device: &Device, slot: usize) {
        let count = self.pending[slot].len();
        if count == 0 {
            return;
        }
        let mut raw = vec![0u64; count * 2];
        let base = QUERIES_PER_SLOT * slot as u32;
        // SAFETY: the pool is live and `raw` is one u64 per query. The range was
        // reset and written by the frame that registered these names (see
        // `expect`'s contract, which is what keeps an aborted frame's names out
        // of here) and the caller proved that frame retired, so nothing below
        // reads a query that was never written.
        let read = unsafe {
            device.raw().get_query_pool_results(
                self.pool,
                base,
                &mut raw,
                vk::QueryResultFlags::TYPE_64,
            )
        };
        if let Err(err) = read {
            tracing::debug!(?err, "gpu timings unavailable this frame");
            return;
        }
        self.last.clear();
        for (index, name) in self.pending[slot].iter().enumerate() {
            let (begin, end) = (raw[index * 2] & self.mask, raw[index * 2 + 1] & self.mask);
            // Wrapping is the correct arithmetic: the counter is `mask` wide and
            // wraps within a frame far more cheaply than it would be detected.
            let ticks = end.wrapping_sub(begin) & self.mask;
            self.last.push(PassTiming {
                name: name.clone(),
                gpu_ms: ticks as f32 * self.period / 1.0e6,
                begin: begin as i64,
                end: end as i64,
            });
        }
    }

    /// The device clock now, for anchoring a profiler's GPU timeline.
    pub(crate) fn clock(&self, device: &Device) -> Option<GpuClock> {
        Some(GpuClock {
            ticks: device.gpu_ticks_now()? as i64,
            period_ns: self.period,
        })
    }

    /// The most recent completed frame's per-pass times.
    pub(crate) fn last(&self) -> &[PassTiming] {
        &self.last
    }

    /// Tear down. Caller waited the device idle.
    pub(crate) fn destroy(&mut self, device: &Device) {
        // SAFETY: the pool belongs to this device and the GPU is idle.
        unsafe { device.raw().destroy_query_pool(self.pool, None) };
    }
}

/// Warn once when a graph outruns the pool, rather than per frame. Silence
/// here would report a prefix of the frame as the whole of it.
pub(crate) fn warn_pass_overflow(passes: usize) {
    if passes <= MAX_TIMED_PASSES {
        return;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            passes,
            timed = MAX_TIMED_PASSES,
            "graph has more passes than the timestamp pool times; the overlay shows a prefix"
        );
    });
}
