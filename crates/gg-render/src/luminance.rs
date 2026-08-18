//! The overlay's luminance histogram (§6 M11's exit row).
//!
//! What an operator actually needs from it is one question — *is this frame
//! exposed?* — and that question is about the scene's radiance, not about the
//! tonemapped picture. So the scene attachment is resampled into a small float
//! target, copied back to the host, and binned by log2 luminance on the CPU.
//!
//! # Why a readback and not a compute pass
//!
//! `gg-rhi` builds graphics pipelines and nothing else (§4.4), and a histogram
//! is the textbook compute-shader job. Adding a compute path for a debug row
//! would be building the wrong thing first: the row is lab equipment, it is off
//! by default, and 64x36 texels is 36 KiB — a copy small enough that the honest
//! version costs less than the machinery to avoid it. When a compute pass exists
//! for a reason of its own, this moves.
//!
//! # Latency, stated
//!
//! The buffer is per frame-in-flight slot and read at the *start* of the slot's
//! next frame — after `begin_frame` has waited the previous one out, which is
//! the wait that makes the read safe without a fence of its own. So the numbers
//! on screen are two frames old, exactly like the per-pass GPU times beside them
//! (§4.8). Offscreen there is one slot and a blocking submit, so they are
//! current.

use gg_rhi::{BufferDesc, BufferHandle, BufferKind, FRAMES_IN_FLIGHT, ImageFormat, RhiError};

use crate::{GpuHost, cvars};

/// Buckets in the histogram. Wide enough to read as a shape at overlay scale,
/// narrow enough that a 2304-texel sample fills it rather than speckling it.
pub const BINS: usize = 64;

/// The resample grid. Small on purpose — see the module docs on cost.
pub(crate) const GRID: (u32, u32) = (64, 36);

/// The bottom and top of the binned range, in stops about a luminance of 1.
/// Twelve stops covers a dim interior through a bright exterior, which is the
/// span an exposure decision is made over.
const MIN_EV: f32 = -8.0;
const MAX_EV: f32 = 4.0;

/// Rec. 709 luma coefficients, applied to **linear** radiance. The scene
/// attachment is linear (§6 M11), so these are the right weights and no decode
/// happens first — the classic mistake here is weighting sRGB-encoded values.
const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// The per-slot readback buffers and the bins the last read produced.
pub(crate) struct Luminance {
    buffers: Vec<BufferHandle>,
    /// Whether a frame has actually written each slot's buffer. Tracked rather
    /// than assumed: the frame the knob is turned on reads a buffer nothing has
    /// filled, and mapped memory is whatever the allocator last left there.
    written: Vec<bool>,
    bins: [u32; BINS],
    /// Texels the last read binned. Zero until a frame with the pass enabled has
    /// retired, which is what tells the overlay to say so rather than draw an
    /// empty chart.
    sampled: u32,
}

/// Bytes one grid's worth of `Rgba16F` occupies.
fn grid_bytes() -> u64 {
    ImageFormat::Rgba16F.packed_size(GRID)
}

impl Luminance {
    /// Allocate one readback buffer per frame in flight.
    ///
    /// Allocated whether or not the pass is enabled: 36 KiB a slot is less than
    /// the code to allocate them on demand, and a knob that stalls the first
    /// frame it is turned on would be a knob nobody trusts the numbers from.
    ///
    /// # Errors
    /// Whatever the allocation refuses.
    pub(crate) fn new(rhi: &mut impl GpuHost) -> Result<Self, RhiError> {
        let mut buffers = Vec::new();
        for slot in 0..FRAMES_IN_FLIGHT {
            buffers.push(rhi.create_buffer(&BufferDesc {
                name: &format!("render.luminance.{slot}"),
                size: grid_bytes(),
                kind: BufferKind::Readback,
            })?);
        }
        Ok(Luminance {
            written: vec![false; buffers.len()],
            buffers,
            bins: [0; BINS],
            sampled: 0,
        })
    }

    /// Whether the pass runs this frame (§4.8's `r.histogram`). Off by default:
    /// a readback every frame is a cost a row nobody is looking at should not
    /// impose.
    pub(crate) fn enabled() -> bool {
        cvars::HISTOGRAM.bool()
    }

    /// `slot`'s readback buffer.
    pub(crate) fn buffer(&self, slot: u64) -> BufferHandle {
        self.buffers[(slot % FRAMES_IN_FLIGHT) as usize]
    }

    /// Record that this frame's graph writes `slot`'s buffer.
    pub(crate) fn mark(&mut self, slot: u64) {
        let at = (slot % FRAMES_IN_FLIGHT) as usize;
        self.written[at] = true;
    }

    /// Bin whatever `slot`'s buffer holds, if a frame has written it.
    ///
    /// The caller is responsible for the ordering the module docs describe: this
    /// reads mapped memory and takes no fence, because the frames-in-flight wait
    /// already happened.
    ///
    /// # Errors
    /// A buffer handle this struct did not issue, which cannot happen.
    pub(crate) fn read(&mut self, rhi: &impl GpuHost, slot: u64) -> Result<(), RhiError> {
        if !self.written[(slot % FRAMES_IN_FLIGHT) as usize] {
            return Ok(());
        }
        let bytes = rhi.map_buffer(self.buffer(slot))?;
        self.bins = [0; BINS];
        self.sampled = 0;
        for texel in bytes.chunks_exact(8) {
            let channel =
                |i: usize| f16_to_f32(u16::from_le_bytes([texel[i * 2], texel[i * 2 + 1]]));
            let luma = channel(0) * LUMA[0] + channel(1) * LUMA[1] + channel(2) * LUMA[2];
            self.bins[bin_of(luma)] += 1;
            self.sampled += 1;
        }
        Ok(())
    }

    /// Forget the last read — what a frame with the pass switched off does, so
    /// the overlay stops drawing a chart of a frame that is no longer on screen.
    pub(crate) fn clear(&mut self) {
        if self.sampled != 0 {
            self.bins = [0; BINS];
            self.sampled = 0;
        }
        self.written.fill(false);
    }

    /// The bins, or `None` before a frame with the pass enabled has retired.
    pub(crate) fn bins(&self) -> Option<&[u32; BINS]> {
        (self.sampled > 0).then_some(&self.bins)
    }

    /// Release the buffers.
    ///
    /// # Errors
    /// A handle the RHI does not recognise, which cannot happen for one issued
    /// through here.
    pub(crate) fn destroy(self, rhi: &mut impl GpuHost) -> Result<(), RhiError> {
        let mut result = Ok(());
        for buffer in self.buffers {
            let one = rhi.destroy_buffer(buffer);
            if result.is_ok() {
                result = one;
            }
        }
        result
    }
}

/// Which bucket a linear luminance falls in.
///
/// Log2 rather than linear, because exposure is a logarithmic decision: half the
/// buckets of a linear histogram would cover the top stop and none of them would
/// say anything about a shadow.
fn bin_of(luma: f32) -> usize {
    // Zero, negative and NaN all take the bottom bucket. Zero because `log2` of
    // it is negative infinity and a black pixel is real data that belongs
    // somewhere; NaN because it is a bug, and a bug should read as darkness
    // rather than as a bar in an arbitrary place.
    if luma <= 0.0 || luma.is_nan() {
        return 0;
    }
    let ev = log2(luma);
    let position = (ev - MIN_EV) / (MAX_EV - MIN_EV);
    ((position * BINS as f32) as isize).clamp(0, BINS as isize - 1) as usize
}

/// Base-2 logarithm of a positive finite float, from its exponent and a
/// polynomial over the mantissa.
///
/// Written out rather than called, because `f32::log2` is a banned transcendental
/// (§3, `clippy.toml`) and `gg_math::sim` is the wrong side of the membrane for a
/// crate that renders. Accuracy needed here is "which of sixty-four buckets",
/// and the mantissa term is exact at both ends of its range.
fn log2(x: f32) -> f32 {
    let bits = x.to_bits();
    let field = (bits >> 23) & 0xff;
    if field == 0 {
        // Subnormal. Far below the binned range whatever its exact value, and
        // reconstructing the mantissa below would be wrong for it — so it goes
        // to the bottom bucket by way of a number `bin_of` clamps.
        return MIN_EV - 1.0;
    }
    let exponent = field as i32 - 127;
    // The mantissa as a value in [1, 2): the exponent field replaced by 0.
    let mantissa = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    // The `atanh` series: with t = (m-1)/(m+1) in [0, 1/3),
    // ln(m) = 2(t + t^3/3 + t^5/5 + t^7/7 + ...), and log2 is that over ln 2.
    // Truncated after t^7, whose worst-case error at t = 1/3 is 5e-5 — three
    // orders of magnitude finer than a bucket is wide. Plain arithmetic
    // throughout, which is what a banned `log2` leaves available (§3).
    let t = (mantissa - 1.0) / (mantissa + 1.0);
    let square = t * t;
    /// 2 / ln 2.
    const SCALE: f32 = 2.885_39;
    let series = t * (1.0 + square * (1.0 / 3.0 + square * (0.2 + square / 7.0)));
    exponent as f32 + SCALE * series
}

/// IEEE-754 binary16 → `f32`. Bit surgery: both formats put the mantissa at the
/// bottom, so a normal value is a bias swap and a 13-bit shift.
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);
    match exponent {
        0 if mantissa == 0 => f32::from_bits(sign),
        // Subnormal: no implicit leading one, exponent fixed at -14, so the
        // value is mantissa * 2^-24. The literal is that power exactly.
        0 => f32::from_bits(sign | (mantissa as f32 / 16_777_216.0).to_bits()),
        // Inf or NaN. Binned as the top bucket by `bin_of`'s clamp, which is
        // where a blown-out pixel belongs.
        31 => f32::INFINITY,
        e => f32::from_bits(sign | ((e + 112) << 23) | (mantissa << 13)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats_decode_at_the_values_the_histogram_cares_about() {
        // The exact binary16 encodings of 0, 1, 0.5 and 8 — a table rather than
        // a round trip, so a wrong shift cannot agree with itself.
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x4800), 8.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
    }

    #[test]
    fn the_log_is_close_enough_to_pick_a_bucket() {
        // A bucket is (MAX_EV - MIN_EV) / BINS stops wide; the approximation has
        // to be well inside that or the histogram would smear across boundaries.
        let width = (MAX_EV - MIN_EV) / BINS as f32;
        for (value, expected) in [
            (1.0, 0.0),
            (2.0, 1.0),
            (0.5, -1.0),
            (8.0, 3.0),
            (0.003_906_25, -8.0),
            (1.5, 0.584_962_5),
        ] {
            let error = (log2(value) - expected).abs();
            assert!(error < width / 8.0, "log2({value}) off by {error}");
        }
    }

    #[test]
    fn a_black_pixel_lands_in_the_bottom_bucket_and_a_blown_one_in_the_top() {
        // Both are real data. `log2(0)` is negative infinity and the cast of it
        // is the kind of thing that lands in an arbitrary bucket rather than in
        // the one a human would point at.
        assert_eq!(bin_of(0.0), 0);
        assert_eq!(bin_of(-1.0), 0, "a negative radiance is still darkness");
        assert_eq!(bin_of(1.0e-9), 0);
        assert_eq!(bin_of(f32::INFINITY), BINS - 1);
        assert_eq!(bin_of(1.0e9), BINS - 1);
        // And the middle of the range is the middle of the bins: MIN_EV is -8
        // and MAX_EV is +4, so a luminance of 1 (0 EV) sits two thirds along.
        assert_eq!(bin_of(1.0), BINS * 2 / 3);
    }
}
