//! What a game tells the host to play — the third protocol on the §4.2.2
//! boundary, arriving at §6 M18 item 2.
//!
//! The shape is not designed, it falls out of §3's deny pin: a game crate may
//! link four engine crates and an audio crate is not one of them, exactly as
//! `gg-ui` is not one of them. So sound crosses the way UI crosses and the way
//! [`Renderable`](super::Renderable) crosses — an ordinary component the game
//! declares and the host plays.
//!
//! # The game says what it sounds like, not the host
//!
//! [`Renderable`](super::Renderable) carries a colour rather than a material
//! name, so nothing about how a game *looks* lives in the host. The same
//! sentence with one word changed is why [`Sound`] carries a waveform, a
//! frequency and an envelope rather than an asset id: a host that resolved
//! `"line_clear.wav"` would hold the game's taste, and the pack would have to
//! grow an audio format before a game could make one noise.
//!
//! This is v0 and it is one voice kind, on the same terms `gg-render` was v0 and
//! one pass: what it plays is tones. Samples are the milestone that replaces the
//! synthesizer's insides, and the protocol is picked so that day is additive —
//! a wave constant this build does not know is silence, never a wrong sound.
//!
//! # Triggering without the host writing back
//!
//! [`Widget::state`](super::Widget) comes back *in* the component, which is what
//! puts a click in the state hash. Audio is the opposite case and takes the
//! opposite answer: the host **never writes** to a `Sound`. It reads
//! [`seq`](Sound::seq) and plays when the number changed since the tick before.
//!
//! That is what makes a silent run and a loud run the same run. A host that
//! consumed a trigger by clearing it would put "did anything play" into the
//! world, and §1.5's audio analogue means an automated tier plays nothing — so
//! the determinism gate would be comparing a silent machine against a loud one
//! and §5.6c would fail on the speaker rather than on the sim.
//!
//! A zeroed `Sound` is silent twice over, which matters because `World::restore`
//! zeroes a retyped field (§4.2.2): [`wave::SILENT`] is 0, and a `seq` of 0
//! never differs from the 0 the host saw last. A migration cannot cause a noise.

use crate::Component;

/// How a [`Sound`] is synthesized. Constants rather than an `enum` field, for
/// the reason [`light`](super::light) gives: a discriminant crossing the
/// boundary is a `u32` the compiler on this side did not write, and an unknown
/// one is silence rather than undefined behaviour the moment it is read.
pub mod wave {
    /// Plays nothing. Zero on purpose — a zeroed component makes no sound.
    pub const SILENT: u32 = 0;
    /// Odd harmonics, hard edge. The blip a game reaches for first.
    pub const SQUARE: u32 = 1;
    /// The fundamental alone.
    pub const SINE: u32 = 2;
    /// Odd harmonics again but rolling off fast — square's softer relative.
    pub const TRIANGLE: u32 = 3;
    /// Uniform noise, for percussion. Deterministic per voice: the host seeds a
    /// `gg_math::sim::Rng` from the trigger, so the same tick makes the same
    /// hiss and a recorded session sounds like itself on replay.
    pub const NOISE: u32 = 4;
}

/// The longest a single trigger may sound, milliseconds. A voice that outlived
/// the run holds a slot in the host's fixed bank forever; clamped rather than
/// refused, because the game asking for it is not an error worth halting a sim.
pub const MAX_MS: u32 = 4_000;

/// One voice the game can trigger.
///
/// Every field is the game's. The host reads them and writes none — see the
/// module docs for why that is the load-bearing half of the protocol.
///
/// Nothing here is spatial. Panning and distance attenuation want a listener
/// pose, which is [`Eye`](super::Eye)'s job and a milestone that has not
/// happened; a stereo field invented before then would be one more thing to
/// migrate.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "gg.sound")]
#[repr(C)]
pub struct Sound {
    /// Bump it to play. The host compares against the value it saw on the
    /// previous tick and triggers on *any* difference, so wrapping is fine and
    /// `seq += 1` is the whole idiom.
    ///
    /// Identity is the entity, so a game that wants two sounds at once needs two
    /// of these — one entity is one voice, and re-triggering it before it
    /// finished cuts the previous note off.
    pub seq: u32,
    /// One of [`wave`]'s constants. An unknown one is silent.
    pub wave: u32,
    /// Frequency at the start of the note, Hz. Unread for [`wave::NOISE`].
    pub hz: f32,
    /// Frequency at the end, Hz — the note sweeps linearly between the two.
    /// Equal to [`hz`](Sound::hz) for a steady tone, which is the common case.
    pub hz_to: f32,
    /// Peak amplitude, `0.0..=1.0`. Clamped by the host; the mixer's headroom is
    /// its own business and a game cannot make it clip by asking loudly.
    pub gain: f32,
    /// How long the note lasts, milliseconds. Clamped to [`MAX_MS`].
    pub ms: u32,
    /// Fade in, milliseconds. Zero is a click on every trigger — a square wave
    /// starting at full amplitude is a step, and a step is a broadband snap.
    pub attack_ms: u32,
    /// Fade out, milliseconds, ending at silence. Overlapping attack and release
    /// in a note shorter than their sum is resolved by the host, not refused.
    pub release_ms: u32,
}

impl Sound {
    /// A steady tone, with the short attack and release that keep a trigger from
    /// clicking. The constructor a game reaches for; the fields are public for
    /// the sweep and the envelope.
    #[must_use]
    pub fn tone(wave: u32, hz: f32, ms: u32, gain: f32) -> Self {
        Sound {
            seq: 0,
            wave,
            hz,
            hz_to: hz,
            gain,
            ms,
            attack_ms: 2,
            release_ms: ms / 4,
        }
    }

    /// A tone that slides from `hz` to `hz_to` over its length.
    #[must_use]
    pub fn sweep(wave: u32, hz: f32, hz_to: f32, ms: u32, gain: f32) -> Self {
        Sound {
            hz_to,
            ..Sound::tone(wave, hz, ms, gain)
        }
    }

    /// Retrigger. The one edit a game makes at runtime, after setting whatever
    /// else the moment calls for.
    pub fn play(&mut self) {
        self.seq = self.seq.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_protocol_type_is_flat_and_padding_free() {
        // `Pod` already refuses padding; this pins the number so a field added
        // is a visible edit rather than a silent layout move.
        assert_eq!(size_of::<Sound>(), 32);
        assert_eq!(align_of::<Sound>(), 4);
    }

    /// The property `World::restore` needs: a component zeroed by a migration
    /// asks for nothing. Both halves are checked, because either alone would
    /// carry it and a later edit could take the wrong one away silently.
    #[test]
    fn a_zeroed_sound_is_silent_twice_over() {
        let zeroed: Sound = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.wave, wave::SILENT);
        assert_eq!(zeroed.seq, 0, "and 0 is what a host that saw nothing holds");
    }

    #[test]
    fn playing_bumps_the_sequence_and_wraps_rather_than_overflowing() {
        let mut sound = Sound::tone(wave::SQUARE, 440.0, 60, 0.5);
        sound.play();
        assert_eq!(sound.seq, 1);
        sound.seq = u32::MAX;
        sound.play();
        assert_eq!(
            sound.seq, 0,
            "a wrap is still a difference, which is the test"
        );
    }

    #[test]
    fn a_sweep_is_a_tone_with_a_second_frequency() {
        let sweep = Sound::sweep(wave::SINE, 880.0, 220.0, 200, 0.3);
        assert_eq!(sweep.hz, 880.0);
        assert_eq!(sweep.hz_to, 220.0);
        assert_eq!(Sound::tone(wave::SINE, 880.0, 200, 0.3).hz_to, 880.0);
    }
}
