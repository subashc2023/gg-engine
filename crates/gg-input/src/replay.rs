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
///
/// Stayed 1 after [`MAX_AXES`] doubled, which is the point of the slot count
/// being in the header: the encoding did not change, only how many slots a
/// given file happens to carry, and a bump would have made every replay
/// recorded before the widening unreadable to buy nothing.
///
/// 2 adds the text channel (§6 M16). That *is* an encoding change — a record
/// the reader would otherwise walk straight past into the wrong bytes — so it
/// is the bump the doubling was not.
///
/// 3 adds the knob channel and the recorded surface (§6 M40): the host-side
/// inputs a session was fed that are not verbs. Appended after the text channel
/// for the reason that channel was appended last — a v2 file is a byte-prefix
/// of a v3 one, so nothing in `tests/replays/` is re-blessed to read it.
pub const FORMAT: u32 = 3;

/// The oldest format this build still reads. A v1 file has no text channel,
/// which decodes as the empty one rather than as a refusal: every replay in
/// `tests/replays/` was recorded before this existed, and a determinism
/// baseline that had to be re-blessed to survive a *reader* change would make
/// the bless a formality instead of a reviewed act (§5.6).
pub const FORMAT_MIN: u32 = 1;

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
    /// The surface the session was laid out for, physical pixels, or `(0, 0)`
    /// when the file does not say (§6 M40).
    ///
    /// Hit-testing divides a physical click by a scale computed from this, so a
    /// session replayed against another surface clicks somewhere else — §6
    /// M15.1's named residual, carried by hand as `--editor-extent` until the
    /// file could carry it. The flag remains, now as the override that authors a
    /// session for a *different* surface, which is what it was really for.
    pub surface: (u32, u32),
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
            surface: (0, 0),
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
    /// `(tick, text)` for every tick that produced characters (§6 M16), in tick
    /// order, at most one entry per tick.
    ///
    /// Not delta-encoded and not held, unlike [`Replay::changes`]: typing is an
    /// *impulse* like a wheel notch — it belongs to the tick it happened in and
    /// to no later one — so the query is an exact match rather than "the most
    /// recent at or before". A held-forward text channel would retype the last
    /// character on every tick after it.
    ///
    /// This is deliberately **not** in [`InputFrame`]. That type crosses into
    /// the game dylib by value and its layout is a cross-artifact contract
    /// (§4.2.2); text is host-UI input that no game reads, and widening the ABI
    /// to carry it would put a string in the sim's boundary to feed a panel.
    text: Vec<(u64, String)>,
    /// `(tick, name, value)` for every *recorded* CVar that moved, at the tick
    /// it moved on, in tick order (§6 M40).
    ///
    /// An impulse channel like [`Replay::text`] and for its reason — a knob
    /// change belongs to the tick it happened in — but unlike text its effect
    /// *persists*, since applying it sets a value that stays set. Which is why
    /// this holds only what moved: the opening values are the entries at tick 0,
    /// so a session that ran wholly at defaults carries no bytes at all.
    knobs: Vec<(u64, String, String)>,
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
    /// Written by a version of this code whose encoding this one cannot walk.
    #[error("replay format {found}, this build reads {FORMAT_MIN}..={FORMAT}")]
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
    /// The file carries more axis slots than this build can hold.
    ///
    /// One-directional on purpose, and it is `World::load`'s policy rather than
    /// `World::restore`'s (§6 M14): a file recorded under a *narrower* cap is
    /// zero-extended, because slot `i` is the verb at index `i` in a list
    /// [`Replay::check_verbs`] has already agreed on and the slots past it were
    /// declared by nobody. A file recorded under a wider one is refused, because
    /// truncating it would drop input the sim was recorded reading.
    #[error("replay carries {found} axis slots, this build holds only {MAX_AXES}")]
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

    /// Restamp the recorded surface — for a *generated* replay, which is
    /// authored for an extent rather than recorded against one (§6 M40).
    /// `(0, 0)` is the pre-v3 spelling of "the file does not say", and writing
    /// it deliberately is how a gate builds its own negative control.
    pub fn set_surface(&mut self, surface: (u32, u32)) {
        self.meta.surface = surface;
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

    /// Characters typed *at* `tick`, or `""` — an exact match, for the reason
    /// [`Replay::text`] gives.
    pub fn text_at(&self, tick: u64) -> &str {
        let at = self.text.partition_point(|&(t, _)| t < tick);
        match self.text.get(at) {
            Some((t, s)) if *t == tick => s,
            _ => "",
        }
    }

    /// How many ticks carried text — what a test asserts on to know the channel
    /// survived a round trip rather than being silently empty.
    pub fn typed_count(&self) -> usize {
        self.text.len()
    }

    /// The knob records at exactly `tick` (§6 M40) — the contiguous run, so a
    /// caller applying them allocates nothing.
    ///
    /// An exact match rather than "at or before", for [`Replay::text`]'s reason:
    /// the record is the *change*, and a change held forward would re-apply
    /// itself over whatever the ticks after it did.
    pub fn knobs_at(&self, tick: u64) -> &[(u64, String, String)] {
        let from = self.knobs.partition_point(|&(t, ..)| t < tick);
        let to = self.knobs.partition_point(|&(t, ..)| t <= tick);
        &self.knobs[from..to]
    }

    /// How many knob records the stream carries — zero for every session run at
    /// defaults, which is what a test asserts to know the channel is not
    /// silently recording noise.
    pub fn knob_count(&self) -> usize {
        self.knobs.len()
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
        self.encode_slots(MAX_AXES)
    }

    /// [`encode`](Self::encode) writing `slots` axes per frame instead of this
    /// build's cap — how a narrower build wrote its files, and so the only way
    /// to author one from a build whose cap has already moved.
    ///
    /// Private, and stays private: a recorder writes what it can hold. It exists
    /// because the decoder's zero-extension is otherwise untestable without a
    /// checked-in fixture that would have to be regenerated to stay a fixture.
    fn encode_slots(&self, slots: usize) -> Vec<u8> {
        debug_assert!(slots <= MAX_AXES);
        let mut out = Vec::with_capacity(64 + self.changes.len() * 48);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT.to_le_bytes());
        out.extend_from_slice(&self.meta.contract.to_le_bytes());
        out.extend_from_slice(&self.meta.tick_hz.to_le_bytes());
        out.extend_from_slice(&self.meta.seed.to_le_bytes());
        out.extend_from_slice(&self.ticks.to_le_bytes());
        out.push(slots as u8);
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
            for axis in &frame.axes[..slots] {
                out.extend_from_slice(&axis.to_le_bytes());
            }
        }
        // Last, so a v1 reader's failure is the format check and not a walk off
        // the end of a record it did know how to read.
        out.extend_from_slice(&(self.text.len() as u32).to_le_bytes());
        for (tick, typed) in &self.text {
            out.extend_from_slice(&tick.to_le_bytes());
            put_str(&mut out, typed);
        }
        // v3, and after the v2 channel for the same reason it was written last.
        out.extend_from_slice(&(self.knobs.len() as u32).to_le_bytes());
        for (tick, name, value) in &self.knobs {
            out.extend_from_slice(&tick.to_le_bytes());
            put_str(&mut out, name);
            put_str(&mut out, value);
        }
        out.extend_from_slice(&self.meta.surface.0.to_le_bytes());
        out.extend_from_slice(&self.meta.surface.1.to_le_bytes());
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
        if !(FORMAT_MIN..=FORMAT).contains(&format) {
            return Err(ReplayError::Format { found: format });
        }
        let contract = r.u32()?;
        let tick_hz = r.u32()?;
        let seed = r.u64()?;
        let ticks = r.u64()?;
        let slots = r.u8()? as usize;
        if slots > MAX_AXES {
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
            // `slots`, not `MAX_AXES`: the header's count is how many are in the
            // file, and the rest of the array is already the zero a slot no
            // recorder ever wrote should read as.
            for axis in &mut frame.axes[..slots] {
                *axis = r.u32()? as i32;
            }
            changes.push((tick, frame));
        }
        // Absent in v1, which is the whole reason the channel is last and the
        // count is read only when the header says there is one.
        let mut text = Vec::new();
        if format >= 2 {
            for _ in 0..r.u32()? {
                text.push((r.u64()?, r.string()?));
            }
        }
        // Absent in v1 and v2, which is why both counts are read only when the
        // header's own version says the bytes are there to read.
        let mut knobs = Vec::new();
        let mut surface = (0, 0);
        if format >= 3 {
            for _ in 0..r.u32()? {
                knobs.push((r.u64()?, r.string()?, r.string()?));
            }
            surface = (r.u32()?, r.u32()?);
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
                surface,
            },
            segments,
            ticks,
            changes,
            text,
            knobs,
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
                text: Vec::new(),
                knobs: Vec::new(),
            },
        }
    }

    /// Record the surface the session is laid out for (§6 M40) — physical
    /// pixels, and the host's to know.
    pub fn record_surface(&mut self, surface: (u32, u32)) {
        self.replay.meta.surface = surface;
    }

    /// Record one tick. Ticks must arrive in order; a frame identical to the
    /// previous one costs nothing but the tick count.
    pub fn record(&mut self, tick: u64, frame: InputFrame) {
        self.replay.ticks = self.replay.ticks.max(tick + 1);
        if self.replay.changes.last().map(|&(_, f)| f) != Some(frame) {
            self.replay.changes.push((tick, frame));
        }
    }

    /// Record characters typed at `tick` (§6 M16), appending if the tick
    /// already has some — a tick can deliver several key events, and the panel
    /// reads them as one string in arrival order.
    ///
    /// Empty text is not recorded: the channel is sparse, and a run of empty
    /// entries would be the held-forward semantics [`Replay::text`] rejects,
    /// written out longhand.
    pub fn record_text(&mut self, tick: u64, typed: &str) {
        if typed.is_empty() {
            return;
        }
        self.replay.ticks = self.replay.ticks.max(tick + 1);
        match self.replay.text.last_mut() {
            Some((t, existing)) if *t == tick => existing.push_str(typed),
            _ => self.replay.text.push((tick, typed.to_owned())),
        }
    }

    /// Record the recorded-CVar values that moved at `tick` (§6 M40).
    ///
    /// Whether a knob is worth recording is settled before this is called — the
    /// caller polls the set the registry declares — so this writes what it is
    /// given. Empty is not a record, which is what makes a defaults-only session
    /// cost nothing.
    pub fn record_knobs(&mut self, tick: u64, moved: &[(&str, String)]) {
        if moved.is_empty() {
            return;
        }
        self.replay.ticks = self.replay.ticks.max(tick + 1);
        for (name, value) in moved {
            self.replay
                .knobs
                .push((tick, (*name).to_owned(), value.clone()));
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
            surface: (1280, 720),
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

    /// §4.7's compatibility claim for a widened [`MAX_AXES`]: a file recorded
    /// under a narrower cap replays unchanged, and one recorded under a wider
    /// cap is refused rather than quietly cut short.
    ///
    /// The first half is what let the cap go from 8 to 16 without re-authoring
    /// a single checked-in replay — the count has been in the header since the
    /// format existed, and until now nothing read it as anything but equality.
    #[test]
    fn a_narrower_recording_zero_extends_and_a_wider_one_is_refused() {
        let replay = recorded();
        let narrow = Replay::decode(&replay.encode_slots(1)).unwrap();
        // Slot 0 is `move_right` in both files, which is `check_verbs`'s claim
        // and not an assumption: the slots past it were declared by nobody.
        for tick in 0..12 {
            assert_eq!(narrow.frame(tick), replay.frame(tick), "tick {tick}");
        }
        assert!(replay.encode_slots(1).len() < replay.encode().len());

        // magic 4 + format 4 + contract 4 + tick_hz 4 + seed 8 + ticks 8.
        const SLOTS_AT: usize = 32;
        let mut wider = replay.encode();
        wider[SLOTS_AT] = MAX_AXES as u8 + 1;
        let err = Replay::decode(&wider).unwrap_err();
        assert!(
            matches!(err, ReplayError::AxisSlots { found } if found == MAX_AXES + 1),
            "{err}"
        );
    }

    /// Typing is an impulse, not a held state. The whole channel exists so a
    /// recorded editor session replays the prompt someone typed (§4.7, §6 M16);
    /// held-forward semantics would retype its last character on every tick
    /// after it, which is a session that types forever.
    #[test]
    fn text_belongs_to_the_tick_it_was_typed_in_and_to_no_later_one() {
        let mut rec = Recorder::new(meta());
        rec.record_text(4, "he");
        // Two key events inside one tick are one string in arrival order.
        rec.record_text(4, "llo");
        rec.record_text(9, "!");
        // Empty is not a record, or the sparse channel would be a dense one.
        rec.record_text(5, "");
        let replay = Replay::decode(&rec.finish().encode()).unwrap();

        assert_eq!(replay.typed_count(), 2);
        assert_eq!(replay.text_at(4), "hello");
        assert_eq!(replay.text_at(9), "!");
        for quiet in [0, 3, 5, 8, 10, 999] {
            assert_eq!(replay.text_at(quiet), "", "tick {quiet}");
        }
        // The recorder still counts the tick it typed in.
        assert_eq!(replay.ticks(), 10);
    }

    /// Every replay in `tests/replays/` predates the channel, and re-blessing a
    /// determinism baseline to survive a reader change would make the bless a
    /// formality rather than a reviewed act (§5.6). So v1 reads, with no text.
    #[test]
    fn a_recording_from_before_the_text_channel_still_reads() {
        let mut rec = Recorder::new(meta());
        rec.record(0, frame(1, 0));
        let v2 = rec.finish().encode();

        // A v1 file is this one without the trailing text record — the channel
        // is last precisely so the older layout is a prefix of the newer.
        const FORMAT_AT: usize = 4;
        let mut v1 = v2.clone();
        v1.truncate(v2.len() - size_of::<u32>());
        v1[FORMAT_AT..FORMAT_AT + 4].copy_from_slice(&1u32.to_le_bytes());

        let replay = Replay::decode(&v1).unwrap();
        assert_eq!(replay.typed_count(), 0);
        assert_eq!(replay.text_at(0), "");
        assert_eq!(replay.frame(0), frame(1, 0));

        // A format this build has never heard of is still refused by number.
        let mut future = v2;
        future[FORMAT_AT..FORMAT_AT + 4].copy_from_slice(&(FORMAT + 1).to_le_bytes());
        let err = Replay::decode(&future).unwrap_err();
        assert!(
            matches!(err, ReplayError::Format { found } if found == FORMAT + 1),
            "{err}"
        );
    }

    /// A knob change is an impulse belonging to one tick, like text — but its
    /// *effect* persists, since applying it sets a value that stays set. So the
    /// channel holds only what moved, and the query is an exact match: a record
    /// held forward would re-apply itself over every tick after it, which for a
    /// knob the console moved back is a session that will not let go.
    #[test]
    fn a_knob_belongs_to_the_tick_it_moved_on_and_the_channel_holds_only_moves() {
        let mut rec = Recorder::new(meta());
        rec.record(0, frame(0, 0));
        // Two knobs moving on one tick is two records, and they stay in the
        // order the registry walked them (ascending by name).
        rec.record_knobs(
            0,
            &[("d.editor_scale", "2".into()), ("r.fov", "1.4".into())],
        );
        rec.record_knobs(40, &[("r.fov", "0.9".into())]);
        // Nothing moved is not a record, or a defaults-only session would pay
        // per tick for saying so.
        rec.record_knobs(41, &[]);
        let replay = Replay::decode(&rec.finish().encode()).unwrap();

        assert_eq!(replay.knob_count(), 3);
        let opening = replay.knobs_at(0);
        assert_eq!(opening.len(), 2);
        assert_eq!(opening[0].1, "d.editor_scale");
        assert_eq!(opening[1].2, "1.4");
        assert_eq!(replay.knobs_at(40).len(), 1);
        for quiet in [1, 39, 41, 999] {
            assert!(replay.knobs_at(quiet).is_empty(), "tick {quiet}");
        }
        // The surface rides the header and survives with it.
        assert_eq!(replay.meta().surface, (1280, 720));
    }

    /// The v2 layout is a byte-prefix of the v3 one, which is what keeps every
    /// checked-in replay readable — and re-blessing a determinism baseline to
    /// survive a *reader* change would make the bless a formality (§5.6). A v2
    /// file therefore decodes with an empty channel and no surface, and `(0, 0)`
    /// is how "the file does not say" is spelled.
    #[test]
    fn a_recording_from_before_the_knob_channel_still_reads() {
        let mut rec = Recorder::new(meta());
        rec.record(0, frame(1, 0));
        let v3 = rec.finish().encode();

        const FORMAT_AT: usize = 4;
        // v3 appends a knob count and two extents to what v2 wrote.
        let mut v2 = v3.clone();
        v2.truncate(v3.len() - 3 * size_of::<u32>());
        v2[FORMAT_AT..FORMAT_AT + 4].copy_from_slice(&2u32.to_le_bytes());

        let replay = Replay::decode(&v2).unwrap();
        assert_eq!(replay.knob_count(), 0);
        assert!(replay.knobs_at(0).is_empty());
        assert_eq!(replay.meta().surface, (0, 0));
        assert_eq!(replay.frame(0), frame(1, 0));
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
