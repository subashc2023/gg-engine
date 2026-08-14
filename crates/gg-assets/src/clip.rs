//! The clip blob: one header, then mono 16-bit PCM (§6 M43).
//!
//! **The sample format is the engine's, not the file's** — [`mesh`](crate::mesh)'s
//! sentence with two words changed, and for the same reason. A pack cannot
//! describe an encoding: what a clip holds is signed 16-bit mono at whatever
//! rate its source ran at, `ggc` produced it, and the runtime casts it in place.
//!
//! Three choices are baked in here rather than left to a consumer, because each
//! one has exactly one right answer and pushing it downstream would mean every
//! consumer re-deciding it:
//!
//! - **Mono.** The mixer sums to one channel and the device fans that out
//!   (`gg_audio::device`), because [`Sound`](gg_ecs::boundary) carries no pan —
//!   there is no listener pose to pan against, which §3 defers by name. Stereo
//!   PCM has nowhere to go, so the downmix happens once, at bake, where it is
//!   an integer average that every host agrees on.
//! - **Uncompressed.** A clip is therefore *cast in place* like a mesh rather
//!   than produced by a worker like a texture (`crate::load`'s asymmetry). PCM
//!   is the one payload zstd barely moves, and paying the worker path for a few
//!   per cent would put a decode between a cue firing and it sounding.
//! - **The source's own rate**, unresampled. A resampler is a filter, and a
//!   filter is a determinism surface that would have to be pinned and versioned
//!   into the source hash. The voice interpolates at play time instead, against
//!   whatever rate the device turned out to run at — which it has to do anyway,
//!   because no bake can know that number.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;

use crate::pack::SECTION_ALIGN;

/// A clip blob's header. 16 bytes, the smallest multiple of [`SECTION_ALIGN`]
/// that holds it — so the samples behind it start 16-aligned even when a caller
/// lays the blob out by hand.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ClipHeader {
    /// Samples per second the source ran at. Not the device's — see the module
    /// docs for why nothing here resamples.
    pub rate: u32,
    /// Number of `i16` samples. Mono, so this is also the frame count.
    pub frames: u32,
    /// Byte offset of the samples, from the start of the blob. 16-aligned.
    pub samples: u32,
    /// Zero. Where a channel count would go if there were ever a second one to
    /// count — a field with one legal value is not a field.
    pub reserved: u32,
}

const _: () = assert!(size_of::<ClipHeader>() == 16);
const _: () = assert!((size_of::<ClipHeader>() as u64).is_multiple_of(SECTION_ALIGN));

/// Why a clip blob could not be read. Distinct from [`crate::PackError`] for
/// [`MeshError`](crate::MeshError)'s reason: the pack's own validation proves a
/// blob is inside the file, and this proves it is a clip.
#[derive(Debug, Error)]
pub enum ClipError {
    /// The blob is shorter than a header.
    #[error("a clip blob of {len} bytes is shorter than its {header}-byte header")]
    TooShort {
        /// The blob's length.
        len: usize,
        /// What a header needs.
        header: usize,
    },
    /// The samples are not where the header says, or run past the blob.
    #[error("a clip's samples at {offset}..{end} do not fit in {len} bytes")]
    OutOfBounds {
        /// Where the samples start.
        offset: u64,
        /// Where they would end.
        end: u64,
        /// The blob's length.
        len: usize,
    },
    /// The samples do not start on an `i16` boundary.
    #[error("a clip's samples at {offset} are not 2-aligned")]
    Misaligned {
        /// Where they start.
        offset: u32,
    },
    /// A rate of zero, which no arithmetic downstream survives.
    #[error("a clip at 0 Hz is not a clip")]
    Silent,
}

/// A clip, borrowed out of a pack.
#[derive(Clone, Copy, Debug)]
pub struct Clip<'a> {
    header: ClipHeader,
    samples: &'a [i16],
}

impl<'a> Clip<'a> {
    /// Read a clip out of a blob, validating every offset it names.
    ///
    /// # Errors
    /// If the blob is too short, or the samples do not lie inside it, or the
    /// rate is zero.
    pub fn read(blob: &'a [u8]) -> Result<Self, ClipError> {
        let head = blob
            .get(..size_of::<ClipHeader>())
            .ok_or(ClipError::TooShort {
                len: blob.len(),
                header: size_of::<ClipHeader>(),
            })?;
        let header: ClipHeader = bytemuck::pod_read_unaligned(head);
        if header.rate == 0 {
            return Err(ClipError::Silent);
        }
        if !header.samples.is_multiple_of(align_of::<i16>() as u32) {
            return Err(ClipError::Misaligned {
                offset: header.samples,
            });
        }
        let len = u64::from(header.frames) * size_of::<i16>() as u64;
        let end = u64::from(header.samples).saturating_add(len);
        let bytes = usize::try_from(end)
            .ok()
            .and_then(|end| blob.get(header.samples as usize..end))
            .ok_or(ClipError::OutOfBounds {
                offset: u64::from(header.samples),
                end,
                len: blob.len(),
            })?;
        let samples = bytemuck::try_cast_slice(bytes).map_err(|_| ClipError::Misaligned {
            offset: header.samples,
        })?;
        Ok(Self { header, samples })
    }

    /// Samples per second the clip was authored at.
    #[must_use]
    pub fn rate(&self) -> u32 {
        self.header.rate
    }

    /// The mono sample array.
    #[must_use]
    pub fn samples(&self) -> &'a [i16] {
        self.samples
    }
}

/// Lay out a clip blob for [`crate::PackWriter::add`].
///
/// `rate` is clamped to at least 1: a zero would be refused by [`Clip::read`]
/// at every load thereafter, and the moment to fail is the one with the source
/// file's name in scope.
#[must_use]
pub fn encode(rate: u32, samples: &[i16]) -> Vec<u8> {
    let header = ClipHeader {
        rate: rate.max(1),
        frames: samples.len() as u32,
        samples: size_of::<ClipHeader>() as u32,
        reserved: 0,
    };
    let mut blob = Vec::with_capacity(size_of::<ClipHeader>() + size_of_val(samples));
    blob.extend_from_slice(bytemuck::bytes_of(&header));
    blob.extend_from_slice(bytemuck::cast_slice(samples));
    blob
}

#[cfg(test)]
mod tests {
    // unwrap is permitted in tests (§2, Error handling row).
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn a_clip_round_trips_through_its_own_blob() {
        let samples: Vec<i16> = (0..1_000).map(|i| (i * 31 - 16_000) as i16).collect();
        let blob = encode(22_050, &samples);
        let clip = Clip::read(&blob).unwrap();
        assert_eq!(clip.rate(), 22_050);
        assert_eq!(clip.samples(), &samples[..]);
    }

    /// The claim the whole 16-byte header exists for: the samples cast without
    /// a copy, which is what "loaded the moment the file is mapped" means.
    #[test]
    fn the_samples_start_aligned_for_both_a_pack_and_a_hand_written_blob() {
        let blob = encode(48_000, &[1, -1, 2, -2]);
        assert!(size_of::<ClipHeader>().is_multiple_of(SECTION_ALIGN as usize));
        assert_eq!(blob.len(), 16 + 8);
    }

    #[test]
    fn an_empty_clip_is_a_clip_rather_than_an_error() {
        // What a source of nothing but silence compiles to. Legal, and the
        // voice that plays it finishes on its first sample.
        let blob = encode(8_000, &[]);
        let clip = Clip::read(&blob).unwrap();
        assert_eq!(clip.samples().len(), 0);
        assert_eq!(clip.rate(), 8_000);
    }

    #[test]
    fn a_blob_that_is_not_a_clip_is_refused_rather_than_reached_past() {
        assert!(matches!(
            Clip::read(&[0; 8]),
            Err(ClipError::TooShort { len: 8, .. })
        ));

        let mut blob = encode(44_100, &[7; 16]);
        // A frame count the blob cannot hold — the failure a truncated file
        // produces, and the one that would otherwise read past the mapping.
        blob[4..8].copy_from_slice(&9_999_u32.to_le_bytes());
        assert!(matches!(
            Clip::read(&blob),
            Err(ClipError::OutOfBounds { .. })
        ));

        let mut zero_rate = encode(44_100, &[7; 16]);
        zero_rate[0..4].copy_from_slice(&0_u32.to_le_bytes());
        assert!(matches!(Clip::read(&zero_rate), Err(ClipError::Silent)));

        let mut odd = encode(44_100, &[7; 16]);
        odd[8..12].copy_from_slice(&17_u32.to_le_bytes());
        assert!(matches!(
            Clip::read(&odd),
            Err(ClipError::Misaligned { offset: 17 })
        ));
    }

    #[test]
    fn a_rate_of_zero_cannot_be_written_in_the_first_place() {
        let blob = encode(0, &[1, 2]);
        assert_eq!(Clip::read(&blob).unwrap().rate(), 1);
    }
}
