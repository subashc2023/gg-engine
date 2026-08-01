//! Input recording and replay (§4.7) — the backbone of bug repro, the golden
//! harness, and the agent verification loop (§1.12).
//!
//! A replay file is **self-describing**: it names the engine commit it was
//! recorded on, the Determinism Contract version (§4.2.1), the tier it was
//! recorded under, the tick rate, the seed, the game's verb lists in id order,
//! and each segment's game-code hash. "A replay file plus a commit hash" (§1.2)
//! is therefore one file — nothing outside it has to be remembered, including
//! by a player who has never heard of any of this.
//!
//! **Segments** exist for live reload (§4.2.2): a segment opens at the tick a
//! game dylib is swapped in and names that dylib's hash, so a replay recorded
//! across a reload still identifies which code produced which ticks. M4B writes
//! one segment; the mechanism is here because retrofitting a file format after
//! replays exist in the wild is how formats acquire version 2.
//!
//! The recorder is **not lab equipment** (§2) — it is in every tier including
//! dist, and the dist gate checks for its presence rather than its absence.

use crate::map::MAX_AXES;
use crate::state::InputFrame;

/// File magic. A replay is identified by its bytes, not by its extension.
pub const MAGIC: [u8; 4] = *b"GGRP";

/// Replay format version. Bumped when the *encoding* changes; the Determinism
/// Contract version in the header is a separate axis and answers a different
/// question (can these bits still be reproduced, versus can they still be read).
pub const FORMAT: u32 = 1;

/// The engine commit this binary was built from, stamped by `build.rs`, or
/// `"unknown"` where git could not be consulted.
pub const ENGINE_COMMIT: &str = env!("GG_ENGINE_COMMIT");

/// What a replay says about itself before its first frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayMeta {
    /// Engine commit at record time.
    pub engine_commit: String,
    /// Determinism Contract version (§4.2.1) — `gg_math::DETERMINISM_CONTRACT`.
    pub contract: u32,
    /// Build tier: `dev`, `instrumented`, `dist`, `dist-verify`.
    pub tier: String,
    /// Sim ticks per second — the fixed timestep the stream is indexed by.
    pub tick_hz: u32,
    /// The sim's seed.
    pub seed: u64,
    /// Action names in id order.
    pub actions: Vec<String>,
    /// Axis names in id order.
    pub axes: Vec<String>,
}

impl ReplayMeta {
    /// The header a host records under, for `contract`
    /// (`gg_math::DETERMINISM_CONTRACT`, which this crate does not depend on) and
    /// a build tier.
    ///
    /// Everything else is this crate's to answer, and deliberately not a
    /// caller's: the engine commit is [`ENGINE_COMMIT`], and the seed is zero
    /// until a game asks for one — the boundary has no seed to hand across
    /// (§4.2.2), so the field is kept honest rather than invented at each call
    /// site.
    #[must_use]
    pub fn new(contract: u32, tier: &str, tick_hz: u32, actions: &[&str], axes: &[&str]) -> Self {
        ReplayMeta {
            engine_commit: ENGINE_COMMIT.to_owned(),
            contract,
            tier: tier.to_owned(),
            tick_hz,
            seed: 0,
            actions: actions.iter().map(|s| (*s).to_owned()).collect(),
            axes: axes.iter().map(|s| (*s).to_owned()).collect(),
        }
    }
}

/// A run of ticks produced by one build of the game code (§4.2.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    /// First tick this segment covers.
    pub first_tick: u64,
    /// The game code's hash for those ticks. Zero until M5 gives it a source.
    pub code_hash: u128,
}

/// A recorded input stream: the header, the segments, and the ticks at which
/// input changed.
#[derive(Clone, Debug)]
pub struct Replay {
    meta: ReplayMeta,
    segments: Vec<Segment>,
    ticks: u64,
    /// `(tick, frame)` at every tick whose frame differs from the one before —
    /// a held key across 600 ticks is two records, not 600.
    changes: Vec<(u64, InputFrame)>,
}

/// Why a replay could not be read, or could not be trusted once read.
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    /// Not a replay file.
    #[error("not a replay: magic is {found:02x?}, expected {MAGIC:02x?}")]
    Magic {
        /// The first four bytes.
        found: [u8; 4],
    },
    /// Written by a different version of this code.
    #[error("replay format {found}, this build reads {FORMAT}")]
    Format {
        /// The version in the file.
        found: u32,
    },
    /// The file ended mid-record.
    #[error("replay truncated at byte {at} (wanted {wanted} more)")]
    Truncated {
        /// Offset the read started at.
        at: usize,
        /// Bytes the read wanted.
        wanted: usize,
    },
    /// A string field was not UTF-8.
    #[error("replay field at byte {at} is not UTF-8")]
    Utf8 {
        /// Offset of the field.
        at: usize,
    },
    /// The file carries a different number of axis slots than this build.
    #[error("replay carries {found} axis slots, this build has {MAX_AXES}")]
    AxisSlots {
        /// Slots per frame in the file.
        found: usize,
    },
    /// The game's verb list has moved under the file, so ids mean something
    /// else than they did — caught by name rather than replayed wrong.
    #[error("replay's {kind} {index} is `{found}`, this build declares `{expected}`")]
    VerbMismatch {
        /// `"action"` or `"axis"`.
        kind: &'static str,
        /// Which id.
        index: usize,
        /// The name in the file.
        found: String,
        /// The name this build declares.
        expected: String,
    },
    /// The verb list changed length in a way that is not an append.
    #[error("replay declares {found} {kind}s, this build declares {expected}")]
    VerbCount {
        /// `"action"` or `"axis"`.
        kind: &'static str,
        /// How many the file has.
        found: usize,
        /// How many this build has.
        expected: usize,
    },
}

impl Replay {
    /// What the file says about itself.
    pub fn meta(&self) -> &ReplayMeta {
        &self.meta
    }

    /// Restamp the engine commit — for a *generated* replay being blessed into
    /// the tree, which knows its commit only at the moment it is written
    /// (§5.6). A recorded replay gets its commit from the running binary and
    /// has no business changing it afterwards.
    pub fn set_engine_commit(&mut self, commit: &str) {
        self.meta.engine_commit = commit.to_owned();
    }

    /// The segments, in tick order.
    pub fn segments(&self) -> &[Segment] {
        &self.segments
    }

    /// How many ticks the stream covers.
    pub fn ticks(&self) -> u64 {
        self.ticks
    }

    /// How many change records the stream holds — the compression the delta
    /// encoding actually achieved, which tests assert on.
    pub fn change_count(&self) -> usize {
        self.changes.len()
    }

    /// Input at `tick`: the most recent recorded change at or before it, or an
    /// empty frame before the first change. Past the end it holds the last
    /// frame, which is what "the recording stopped" means for a held key.
    pub fn frame(&self, tick: u64) -> InputFrame {
        let at = self.changes.partition_point(|&(t, _)| t <= tick);
        match at.checked_sub(1) {
            Some(i) => self.changes[i].1,
            None => InputFrame::default(),
        }
    }

    /// The game-code hash covering `tick` (§4.2.2).
    pub fn segment_at(&self, tick: u64) -> Option<Segment> {
        let at = self.segments.partition_point(|s| s.first_tick <= tick);
        at.checked_sub(1).map(|i| self.segments[i])
    }

    /// Check that this build's verb lists still mean what the file's ids meant.
    ///
    /// Appending verbs is compatible; reordering or renaming is not, and is
    /// reported by name and index rather than silently replayed onto the wrong
    /// action.
    pub fn check_verbs(&self, actions: &[&str], axes: &[&str]) -> Result<(), ReplayError> {
        check_list("action", &self.meta.actions, actions)?;
        check_list("axis", &self.meta.axes, axes)
    }

    /// Encode to bytes. Little-endian throughout: it is the byte order of every
    /// target in the contract, and picking the host's would make a replay
    /// unreadable on the architecture most likely to be asked to reproduce it.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + self.changes.len() * 48);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT.to_le_bytes());
        out.extend_from_slice(&self.meta.contract.to_le_bytes());
        out.extend_from_slice(&self.meta.tick_hz.to_le_bytes());
        out.extend_from_slice(&self.meta.seed.to_le_bytes());
        out.extend_from_slice(&self.ticks.to_le_bytes());
        out.push(MAX_AXES as u8);
        put_str(&mut out, &self.meta.engine_commit);
        put_str(&mut out, &self.meta.tier);
        put_list(&mut out, &self.meta.actions);
        put_list(&mut out, &self.meta.axes);

        out.extend_from_slice(&(self.segments.len() as u32).to_le_bytes());
        for s in &self.segments {
            out.extend_from_slice(&s.first_tick.to_le_bytes());
            out.extend_from_slice(&s.code_hash.to_le_bytes());
        }
        out.extend_from_slice(&(self.changes.len() as u32).to_le_bytes());
        for (tick, frame) in &self.changes {
            out.extend_from_slice(&tick.to_le_bytes());
            out.extend_from_slice(&frame.buttons.to_le_bytes());
            for axis in frame.axes {
                out.extend_from_slice(&axis.to_le_bytes());
            }
        }
        out
    }

    /// Decode bytes written by [`Replay::encode`].
    pub fn decode(bytes: &[u8]) -> Result<Self, ReplayError> {
        let mut r = Reader { bytes, at: 0 };
        let magic: [u8; 4] = r.array()?;
        if magic != MAGIC {
            return Err(ReplayError::Magic { found: magic });
        }
        let format = r.u32()?;
        if format != FORMAT {
            return Err(ReplayError::Format { found: format });
        }
        let contract = r.u32()?;
        let tick_hz = r.u32()?;
        let seed = r.u64()?;
        let ticks = r.u64()?;
        let slots = r.u8()? as usize;
        if slots != MAX_AXES {
            return Err(ReplayError::AxisSlots { found: slots });
        }
        let engine_commit = r.string()?;
        let tier = r.string()?;
        let actions = r.list()?;
        let axes = r.list()?;

        let mut segments = Vec::new();
        for _ in 0..r.u32()? {
            segments.push(Segment {
                first_tick: r.u64()?,
                code_hash: r.u128()?,
            });
        }
        let mut changes = Vec::new();
        for _ in 0..r.u32()? {
            let tick = r.u64()?;
            let buttons = r.u64()?;
            let mut frame = InputFrame {
                buttons,
                axes: [0; MAX_AXES],
            };
            for axis in &mut frame.axes {
                *axis = r.u32()? as i32;
            }
            changes.push((tick, frame));
        }

        Ok(Replay {
            meta: ReplayMeta {
                engine_commit,
                contract,
                tier,
                tick_hz,
                seed,
                actions,
                axes,
            },
            segments,
            ticks,
            changes,
        })
    }
}

/// Builds a [`Replay`] one tick at a time.
#[derive(Clone, Debug)]
pub struct Recorder {
    replay: Replay,
}

impl Recorder {
    /// Start recording. The first segment opens at tick 0 with no code hash;
    /// M5's reload host replaces it with the loaded dylib's.
    pub fn new(meta: ReplayMeta) -> Self {
        Recorder {
            replay: Replay {
                meta,
                segments: vec![Segment {
                    first_tick: 0,
                    code_hash: 0,
                }],
                ticks: 0,
                changes: Vec::new(),
            },
        }
    }

    /// Record one tick. Ticks must arrive in order; a frame identical to the
    /// previous one costs nothing but the tick count.
    pub fn record(&mut self, tick: u64, frame: InputFrame) {
        self.replay.ticks = self.replay.ticks.max(tick + 1);
        if self.replay.changes.last().map(|&(_, f)| f) != Some(frame) {
            self.replay.changes.push((tick, frame));
        }
    }

    /// Open a segment at `tick` for game code hashing to `code_hash` (§4.2.2).
    pub fn open_segment(&mut self, tick: u64, code_hash: u128) {
        // A reload before any tick ran replaces the opening segment rather than
        // leaving a zero-length one to be reasoned about later.
        if self.replay.segments.last().map(|s| s.first_tick) == Some(tick) {
            if let Some(last) = self.replay.segments.last_mut() {
                last.code_hash = code_hash;
            }
            return;
        }
        self.replay.segments.push(Segment {
            first_tick: tick,
            code_hash,
        });
    }

    /// The recording so far.
    pub fn replay(&self) -> &Replay {
        &self.replay
    }

    /// Finish and take the recording.
    pub fn finish(self) -> Replay {
        self.replay
    }
}

fn check_list(kind: &'static str, found: &[String], expected: &[&str]) -> Result<(), ReplayError> {
    if found.len() > expected.len() {
        return Err(ReplayError::VerbCount {
            kind,
            found: found.len(),
            expected: expected.len(),
        });
    }
    for (index, (f, e)) in found.iter().zip(expected).enumerate() {
        if f != e {
            return Err(ReplayError::VerbMismatch {
                kind,
                index,
                found: f.clone(),
                expected: (*e).to_owned(),
            });
        }
    }
    Ok(())
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u32).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_list(out: &mut Vec<u8>, list: &[String]) {
    out.extend_from_slice(&(list.len() as u32).to_le_bytes());
    for s in list {
        put_str(out, s);
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ReplayError> {
        let end = self.at.saturating_add(n);
        let slice = self.bytes.get(self.at..end).ok_or(ReplayError::Truncated {
            at: self.at,
            wanted: n,
        })?;
        self.at = end;
        Ok(slice)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], ReplayError> {
        let slice = self.take(N)?;
        let mut out = [0u8; N];
        out.copy_from_slice(slice);
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, ReplayError> {
        Ok(self.array::<1>()?[0])
    }

    fn u32(&mut self) -> Result<u32, ReplayError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, ReplayError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn u128(&mut self) -> Result<u128, ReplayError> {
        Ok(u128::from_le_bytes(self.array()?))
    }

    fn string(&mut self) -> Result<String, ReplayError> {
        let at = self.at;
        let len = self.u32()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| ReplayError::Utf8 { at })
    }

    fn list(&mut self) -> Result<Vec<String>, ReplayError> {
        let count = self.u32()?;
        let mut out = Vec::new();
        for _ in 0..count {
            out.push(self.string()?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn meta() -> ReplayMeta {
        ReplayMeta {
            engine_commit: "abc123".into(),
            contract: 1,
            tier: "dev".into(),
            tick_hz: 60,
            seed: 7,
            actions: vec!["look".into(), "spawn".into()],
            axes: vec!["move_right".into()],
        }
    }

    fn frame(buttons: u64, first_axis: i32) -> InputFrame {
        let mut f = InputFrame {
            buttons,
            axes: [0; MAX_AXES],
        };
        f.axes[0] = first_axis;
        f
    }

    fn recorded() -> Replay {
        let mut rec = Recorder::new(meta());
        for tick in 0..10u64 {
            // Down from tick 3, and the axis moves once at tick 7.
            let buttons = u64::from(tick >= 3);
            rec.record(tick, frame(buttons, if tick >= 7 { -512 } else { 0 }));
        }
        rec.finish()
    }

    #[test]
    fn a_held_key_costs_one_record_not_one_per_tick() {
        let replay = recorded();
        assert_eq!(replay.ticks(), 10);
        // tick 0 (up), tick 3 (down), tick 7 (axis moved).
        assert_eq!(replay.change_count(), 3);
        for tick in 0..3 {
            assert_eq!(replay.frame(tick), frame(0, 0), "tick {tick}");
        }
        assert_eq!(replay.frame(6), frame(1, 0));
        assert_eq!(replay.frame(7), frame(1, -512));
        assert_eq!(replay.frame(999), frame(1, -512), "past the end it holds");
    }

    #[test]
    fn a_replay_round_trips_through_its_bytes() {
        let replay = recorded();
        let bytes = replay.encode();
        let back = Replay::decode(&bytes).unwrap();
        assert_eq!(back.meta(), replay.meta());
        assert_eq!(back.ticks(), replay.ticks());
        assert_eq!(back.segments(), replay.segments());
        for tick in 0..12 {
            assert_eq!(back.frame(tick), replay.frame(tick), "tick {tick}");
        }
        // Negative axis values survive the u32 round trip.
        assert_eq!(back.frame(7).axes[0], -512);
    }

    #[test]
    fn a_file_that_is_not_ours_is_refused_rather_than_read() {
        let bytes = recorded().encode();
        assert!(matches!(
            Replay::decode(b"nope").unwrap_err(),
            ReplayError::Magic { .. }
        ));
        let mut wrong_format = bytes.clone();
        wrong_format[4] = 99;
        assert!(matches!(
            Replay::decode(&wrong_format).unwrap_err(),
            ReplayError::Format { found: 99 }
        ));
        for cut in [8, 20, bytes.len() - 1] {
            assert!(
                matches!(
                    Replay::decode(&bytes[..cut]),
                    Err(ReplayError::Truncated { .. })
                ),
                "a replay cut at {cut} decoded anyway"
            );
        }
    }

    #[test]
    fn a_moved_verb_list_is_caught_by_name() {
        let replay = recorded();
        assert!(
            replay
                .check_verbs(&["look", "spawn"], &["move_right"])
                .is_ok()
        );
        // Appending is compatible...
        assert!(
            replay
                .check_verbs(&["look", "spawn", "crouch"], &["move_right"])
                .is_ok()
        );
        // ...reordering is not, and says which id moved.
        let err = replay
            .check_verbs(&["spawn", "look"], &["move_right"])
            .unwrap_err();
        assert!(
            matches!(&err, ReplayError::VerbMismatch { index: 0, found, .. } if found == "look"),
            "{err}"
        );
        let err = replay.check_verbs(&[], &["move_right"]).unwrap_err();
        assert!(
            matches!(err, ReplayError::VerbCount { found: 2, .. }),
            "{err}"
        );
    }

    #[test]
    fn segments_name_the_code_that_produced_each_tick() {
        let mut rec = Recorder::new(meta());
        rec.open_segment(0, 0xaaaa);
        for tick in 0..5 {
            rec.record(tick, frame(0, 0));
        }
        rec.open_segment(5, 0xbbbb);
        for tick in 5..10 {
            rec.record(tick, frame(1, 0));
        }
        let replay = rec.finish();
        assert_eq!(replay.segments().len(), 2, "the opening segment was reused");
        assert_eq!(replay.segment_at(4).map(|s| s.code_hash), Some(0xaaaa));
        assert_eq!(replay.segment_at(5).map(|s| s.code_hash), Some(0xbbbb));
        let back = Replay::decode(&replay.encode()).unwrap();
        assert_eq!(back.segments(), replay.segments());
    }
}
