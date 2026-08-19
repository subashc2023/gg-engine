//! The mixer: triggers in, mono samples out. No device, no threads, no clock —
//! which is why every claim in §6 M18 item 2 about *what a cue sounds like* is
//! provable in an ordinary unit test on a machine that is silent.
//!
//! The split is `gg-rhi`'s: [`device`](super::device) is the only part that
//! talks to an OS, and it is a buffer's worth of arithmetic away from anything
//! testable. Everything with an opinion is here.

use std::sync::Arc;

use gg_ecs::boundary::{MAX_MS, Sound, wave};
use gg_math::sim;

use crate::bank::Bank;

/// Voices that can sound at once. A fixed bank rather than a growing list: the
/// mixer runs in an audio callback, where allocating is the classic way to miss
/// a deadline and produce the very glitch it was called to avoid.
///
/// The bank is not a per-game limit — one `Sound` entity is one voice, and it is
/// the *overlap* that competes. Demo 10 triggers at most four in a tick.
pub const VOICES: usize = 32;

/// Peak of the summed bank before the limiter starts working, and the reason
/// there is one: eight voices at the gain a game would reasonably pick for a
/// single blip sum past 1.0, and a hard clip there is broadband distortion on
/// exactly the frames a game is busiest.
const HEADROOM: f32 = 0.8;

/// What a tick hands the mixer: a [`Sound`] whose `seq` moved, plus the slot it
/// belongs to. Plain data, so the queue between the sim thread and the audio
/// callback carries no references and needs no lifetime.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trigger {
    /// Which voice this replaces. A game's entity, hashed into the bank by the
    /// observer — re-triggering the same entity cuts its previous note off,
    /// which is what "one entity is one voice" means audibly.
    pub slot: usize,
    /// The note, exactly as the game wrote it.
    pub sound: Sound,
}

/// One sounding note.
#[derive(Clone, Copy, Debug)]
struct Voice {
    wave: u32,
    /// Cycles, fractional, wrapped to `0.0..1.0` every sample so a long note
    /// does not lose frequency resolution to a growing float.
    phase: f32,
    hz: f32,
    hz_to: f32,
    gain: f32,
    /// Samples played so far, and the note's whole length. Integers because the
    /// envelope's corners must land on an exact sample — a float countdown
    /// leaves a fraction of a sample of full-amplitude signal at the end, which
    /// is a click.
    at: u32,
    len: u32,
    attack: u32,
    release: u32,
    /// Noise's stream. Seeded from the trigger rather than from a clock, so the
    /// same session sounds the same twice — a run whose hiss differed would make
    /// "record it and listen to it again" useless as a way to hear a bug.
    rng: sim::Rng,
    /// Where in the bank this voice reads, for the clip waves (§6 M43). `None`
    /// for everything the synthesizer produces itself.
    clip: Option<Playback>,
}

/// A clip voice's read head.
///
/// The cursor is **32.32 fixed point** rather than an `f32`, and that is not
/// tidiness: an `f32` accumulator loses its last bit of resolution at 16.7
/// million samples, which a six-minute track passes — a music voice would drift
/// off pitch on its way through, silently and only on long clips.
#[derive(Clone, Copy, Debug)]
struct Playback {
    /// First sample of this clip in the bank's block.
    start: u32,
    frames: u32,
    /// Read position within the clip, 32.32 fixed point.
    cursor: u64,
    /// Samples of source per sample of output, 32.32 — the whole of the
    /// resampling this engine does, and the only place a clip's authored rate
    /// meets the rate the device turned out to run at.
    step: u64,
    /// Repeat rather than finish ([`wave::CLIP_LOOP`]).
    looping: bool,
}

impl Voice {
    /// `None` for a note that would make no sound: an unknown or silent wave, a
    /// zero length, or a gain of nothing. Cheaper to refuse here than to mix
    /// silence for 4000 ms while holding a slot.
    ///
    /// A clip wave naming an id `bank` does not hold is refused the same way —
    /// silence, never an error (`Sound::clip`'s contract). That is the case a
    /// pack rebuilt under a running game produces, and it must not be louder
    /// than the case where a game asked for nothing.
    fn new(sound: &Sound, rate: u32, seed: u64, bank: Option<&Bank>) -> Option<Voice> {
        if sound.wave == wave::SILENT || sound.wave > wave::CLIP_LOOP {
            return None;
        }
        let gain = sound.gain.clamp(0.0, 1.0);
        if gain <= 0.0 {
            return None;
        }
        let looping = sound.wave == wave::CLIP_LOOP;
        let (clip, len) = if sound.wave >= wave::CLIP {
            let entry = bank?.locate(sound.clip)?;
            // Source samples per output sample, 32.32. A clip authored at
            // 22.05 kHz on a 48 kHz device steps by less than one, which is the
            // interpolating direction and the common one.
            let step = (u64::from(entry.rate) << 32) / u64::from(rate.max(1));
            // The note's length at the *device* rate, so the envelope, `at` and
            // `finished` are the same arithmetic they are for a tone.
            let whole = ((u64::from(entry.frames) << 32) / step.max(1)) as u32;
            let len = match (looping, sound.ms) {
                // A loop ends when something else takes the slot. `u32::MAX` is
                // 24.8 hours at 48 kHz — longer than a session, and finite so
                // there is no branch in `finished` that never fires.
                (true, _) => u32::MAX,
                // 0 is the clip's own length, which is why a game names a clip
                // at all; anything else truncates it.
                (false, 0) => whole,
                (false, ms) => whole.min(samples(ms, rate)),
            };
            let playback = Playback {
                start: entry.start,
                frames: entry.frames,
                cursor: 0,
                step,
                looping,
            };
            (Some(playback), len)
        } else {
            (None, samples(sound.ms.min(MAX_MS), rate))
        };
        if len == 0 {
            return None;
        }
        let (attack, release) = envelope(sound, rate, len);
        Some(Voice {
            wave: sound.wave,
            phase: 0.0,
            hz: sound.hz.max(0.0),
            hz_to: sound.hz_to.max(0.0),
            gain,
            at: 0,
            len,
            attack,
            // Kept for a loop as well as a note, and unread while the loop
            // repeats: `len` is `u32::MAX`, so `envelope_at`'s `left` never
            // falls under it. A release applied *per repeat* would pump the
            // loop (`Sound::release_ms`'s contract) and this is not that —
            // it is spent once, by [`Voice::stop`], as the fade out (§6 M77).
            release,
            rng: sim::Rng::from_seed(seed),
            clip,
        })
    }

    /// The next sample, or `None` once the note has finished. `block` is the
    /// bank's sample block, already validated by [`Bank::locate`].
    fn sample(&mut self, rate: f32, block: &[i16]) -> Option<f32> {
        if self.at >= self.len {
            return None;
        }
        if let Some(mut play) = self.clip {
            let value = play.read(block);
            self.clip = Some(play);
            // Envelope before the advance, for the reason the oscillator branch
            // gives below — the first sample of a note must be the attack's own
            // zero and not one step above it.
            let amplitude = self.envelope_at();
            self.at = self.at.saturating_add(1);
            return Some(value * self.gain * amplitude);
        }
        // Sweep on the note's own progress, so a 20 ms chirp and a 2 s slide are
        // the same two lines.
        let t = self.at as f32 / self.len as f32;
        let hz = self.hz + (self.hz_to - self.hz) * t;
        let value = match self.wave {
            wave::SQUARE => {
                if self.phase < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
            // `sim::sin` and not `f32::sin`: clippy.toml bans the std one
            // workspace-wide (§4.2.1), and an allow here would be the exception
            // that makes the ban advisory.
            wave::SINE => sim::sin(self.phase * core::f32::consts::TAU),
            wave::TRIANGLE => 1.0 - 4.0 * (self.phase - 0.5).abs(),
            // Uniform in `-1.0..1.0`, from the top 24 bits — the low bits of a
            // SplitMix64 draw are as good, but taking the top makes the mantissa
            // fill obvious rather than a thing to re-derive.
            _ => (self.rng.next_u32() >> 8) as f32 / 8_388_608.0 - 1.0,
        };
        // Envelope read at the sample being produced, *before* advancing. Read
        // after, the first sample of every note came out one step above silence
        // — which is the click the envelope exists to remove, surviving inside
        // the thing that removes it.
        let amplitude = self.envelope_at();
        self.phase = (self.phase + hz / rate).fract();
        self.at += 1;
        Some(value * self.gain * amplitude)
    }

    /// The amplitude multiplier at the current sample: ramp in, hold, ramp out.
    /// Linear rather than exponential because the ends are what matter — both
    /// reach exactly 0.0 at exactly the right sample, and an exponential curve
    /// approaching zero leaves a step behind.
    fn envelope_at(&self) -> f32 {
        if self.at < self.attack {
            return self.at as f32 / self.attack as f32;
        }
        // `len - 1 - at`, so the *last* sample the note produces is the zero and
        // not the one after it, which is never played.
        let left = self.len - 1 - self.at;
        if left < self.release {
            return left as f32 / self.release as f32;
        }
        1.0
    }

    /// No samples left. Checked after a buffer as well as before one, so
    /// [`Mixer::sounding`] is exact at a note's last sample rather than at the
    /// first sample past it.
    fn finished(&self) -> bool {
        self.at >= self.len
    }

    /// Bring this voice to an end, and say whether it will get there itself.
    ///
    /// `true` only for a **loop with a release**: it is the one voice with no
    /// end of its own, so it is the one that has to be given one. Giving it
    /// an `at + release` end hands the ramp to the envelope that already
    /// exists, and the clip goes on reading — a fade a player hears as the
    /// music receding rather than as a value walked to zero.
    ///
    /// Everything else answers `false` and is dropped where it stands, which
    /// is what it always was: a note is finite, near its own release, and
    /// replacing one is the case §6 M77 measured at a ratio of 0.63 — under
    /// the step the incoming sound makes by itself, so there is nothing there
    /// to hear.
    fn stop(&mut self) -> bool {
        if self.wave != wave::CLIP_LOOP || self.release == 0 {
            return false;
        }
        self.len = self.at.saturating_add(self.release);
        true
    }

    /// Whether this voice is a loop on its way out — [`Voice::stop`] has
    /// given it an end and it has not reached it.
    fn fading(&self) -> bool {
        self.wave == wave::CLIP_LOOP && self.len != u32::MAX
    }

    /// Take a fade back, if `next` is the same loop this voice is already
    /// playing. `true` if it did, and the caller keeps this voice.
    ///
    /// This is a player who paused and changed their mind inside the fade,
    /// and it is worth its own path for a reason larger than the seam: a
    /// fresh voice reads from sample 0, so a pause taken back would drop the
    /// needle at the top of the track. Resuming keeps the phrase.
    ///
    /// The envelope is rejoined rather than reset. Putting `len` back alone
    /// would take the amplitude from wherever the fade had reached straight
    /// to 1.0 — a step *larger* than the one this exists to remove — so `at`
    /// is moved to the point on the attack that reads the level the release
    /// had it at, and the ramp continues upward from exactly there.
    fn resume(&mut self, next: &Voice) -> bool {
        if !self.fading() || next.wave != wave::CLIP_LOOP {
            return false;
        }
        // Same clip, by where it lives in the bank — two names for one blob
        // are one blob, and a different clip is a different track.
        let same = match (self.clip, next.clip) {
            (Some(held), Some(want)) => held.start == want.start && held.frames == want.frames,
            _ => false,
        };
        if !same {
            return false;
        }
        let level = self.envelope_at();
        self.len = u32::MAX;
        self.at = (level * self.attack as f32) as u32;
        // The slider may have moved while it was going.
        self.gain = next.gain;
        true
    }
}

impl Playback {
    /// The next sample of the clip, `-1.0..1.0`, advancing the cursor.
    ///
    /// Linear interpolation between neighbours rather than nearest: at a step
    /// near 1.0 the two are the same, and at 22.05 kHz against a 48 kHz device
    /// nearest-neighbour is audible aliasing on exactly the short bright cues a
    /// game reaches for. Anything better than linear is a filter, and a filter
    /// is a pinned kernel and a determinism surface — the argument
    /// `gg_assets::clip` makes for not resampling at bake time.
    fn read(&mut self, block: &[i16]) -> f32 {
        if self.frames == 0 {
            return 0.0;
        }
        let index = (self.cursor >> 32) as u32;
        if index >= self.frames {
            // Past the end of a one-shot. Silence rather than a wrap, and the
            // note's own `len` is what retires it — the two agree to within the
            // rounding of one fixed-point division, and this is that rounding.
            return 0.0;
        }
        let at = (self.start + index) as usize;
        let next = if index + 1 < self.frames {
            at + 1
        } else if self.looping {
            // The seam. Interpolating into sample 0 is what makes a loop whose
            // ends match join without a click.
            self.start as usize
        } else {
            at
        };
        // `get` rather than an index: `locate` proved the span fits, and the
        // audio callback is the one place in this tree where being wrong about
        // that would be a panic on a thread nobody can catch it on.
        let a = f32::from(block.get(at).copied().unwrap_or(0));
        let b = f32::from(block.get(next).copied().unwrap_or(0));
        // The fraction's top 24 bits, which is every bit an `f32` can hold
        // exactly — so the interpolation is the same arithmetic on every host.
        let frac = ((self.cursor & 0xffff_ffff) >> 8) as f32 / 16_777_216.0;
        self.cursor = self.cursor.wrapping_add(self.step);
        if self.looping {
            let span = u64::from(self.frames) << 32;
            // `%` and not a subtract: a clip shorter than one output sample
            // steps past its whole span in one read.
            self.cursor %= span;
        }
        // 32768 and not 32767: the negative full scale is what divides exactly,
        // and the asymmetry is half a bit at the top of an i16.
        (a + (b - a) * frac) / 32_768.0
    }
}

/// Milliseconds to samples, rounded down but never to zero for a nonzero ask —
/// a 1 ms attack at 48 kHz is 48 samples, and the rounding only bites at rates
/// nothing uses.
fn samples(ms: u32, rate: u32) -> u32 {
    (u64::from(ms) * u64::from(rate) / 1000) as u32
}

/// Attack and release in samples, fitted inside the note.
///
/// A game asking for more envelope than note is scaled rather than refused: the
/// natural way to write a very short blip is to keep the same release you use
/// everywhere and shorten `ms`, and that should sound like a short blip instead
/// of failing. Both are at least one sample, because dividing by the ramp length
/// is how [`Voice::envelope_at`] reads them.
fn envelope(sound: &Sound, rate: u32, len: u32) -> (u32, u32) {
    let attack = samples(sound.attack_ms, rate).max(1);
    let release = samples(sound.release_ms, rate).max(1);
    let total = attack.saturating_add(release);
    if total <= len {
        return (attack, release);
    }
    // Proportional, and the release keeps the remainder: a click at the end of a
    // note is far more audible than a slightly abrupt start.
    let scaled = ((u64::from(attack) * u64::from(len)) / u64::from(total)) as u32;
    let attack = scaled.clamp(1, len.saturating_sub(1).max(1));
    (attack, (len - attack).max(1))
}

/// How long the master takes to reach a value [`Mixer::set_master`] was
/// handed, milliseconds.
///
/// Ten: over the shortest step a click needs to become a slope, and under
/// what a hand moving a slider could tell from instant. It is not a fade —
/// a fade is [`Sound::release_ms`] on a loop and is measured in hundreds of
/// milliseconds — it is the smallest ramp that stops a jump being a step.
const GLIDE_MS: u32 = 10;

/// The voice bank. Owned by the audio callback and reached from nowhere else.
pub struct Mixer {
    voices: [Option<Voice>; VOICES],
    rate: u32,
    /// Counts triggers, and is the noise seed. Not a clock: it advances once per
    /// note rather than with time, which is what makes a replayed session's
    /// noise identical to the original's.
    fired: u64,
    /// The clips (§6 M43). `None` until the shell hands one over, which is what
    /// a game with no pack stays at.
    clips: Option<Arc<Bank>>,
    /// Where [`Mixer::set_master`] is taking the gain, and where `master`
    /// walks to a sample at a time.
    ///
    /// A step rather than a glide was a discontinuity in every looping voice
    /// at once, because master reaches them *live* (see below): §6 M77
    /// measured a suspend at 0.202 against the signal's own 0.030 — a
    /// full-amplitude cut of the music on every lost focus (§6 M49).
    target: f32,
    /// The player's master volume, applied **to looping voices only**.
    ///
    /// A one-shot carries its volume in the trigger's own gain, set when it
    /// fired (§6 M19) — it is over before a slider finishes moving, and that is
    /// what makes `Audio::fired` report what will sound. A loop is not over, so
    /// music that ignored the slider until the next track would be a settings
    /// menu that appears not to work.
    master: f32,
}

impl Mixer {
    /// A silent bank at `rate` samples per second.
    #[must_use]
    pub fn new(rate: u32) -> Mixer {
        Mixer {
            voices: [None; VOICES],
            rate: rate.max(1),
            fired: 0,
            clips: None,
            target: 1.0,
            master: 1.0,
        }
    }

    /// Install the clips a trigger's [`Sound::clip`] resolves against.
    ///
    /// **Cuts every voice.** A voice holds offsets into the block it was
    /// started against ([`Bank::locate`]'s contract), so the alternative is
    /// reading the wrong samples; and a pack rebuilt under a running game
    /// changed the very bytes a sounding note is reading, so continuing it
    /// would be continuing something that no longer exists.
    pub fn set_bank(&mut self, bank: Arc<Bank>) {
        self.silence();
        self.clips = Some(bank);
    }

    /// The player's master volume, `0.0..=1.0`.
    ///
    /// Reached over [`GLIDE_MS`] rather than on the next sample — see
    /// [`Mixer::target`]. A caller that sets the same value twice glides
    /// nowhere, so restating it every tick costs nothing.
    pub fn set_master(&mut self, master: f32) {
        self.target = master.clamp(0.0, 1.0);
    }

    /// Start `trigger`, replacing whatever was in its slot.
    ///
    /// Replacing and not queueing: the slot *is* the game's entity, and a game
    /// that retriggered the same `Sound` twice in three ticks meant the second
    /// one. A note that wanted to ring over its successor is two entities.
    pub fn fire(&mut self, trigger: &Trigger) {
        let slot = trigger.slot % VOICES;
        self.fired = self.fired.wrapping_add(1);
        let bank = self.clips.as_deref();
        match Voice::new(&trigger.sound, self.rate, self.fired, bank) {
            Some(voice) => {
                // The same loop, asked for again while the last request to
                // stop it is still running out — see [`Voice::resume`].
                if !self.voices[slot]
                    .as_mut()
                    .is_some_and(|held| held.resume(&voice))
                {
                    self.voices[slot] = Some(voice);
                }
            }
            // Nothing to start. On a loop that is a **stop**, and a loop
            // stopped on the sample it was asked to is a click: §6 M77
            // measured demo 10's theme cut at 0.202 against a signal whose
            // own largest step was 0.030, which is the noise a player hears
            // every time they pause. Asking the voice to end itself leaves
            // the slot busy for the length of the fade, which is what makes
            // this a stop rather than a deletion.
            None => {
                if !self.voices[slot].as_mut().is_some_and(Voice::stop) {
                    self.voices[slot] = None;
                }
            }
        }
    }

    /// Fill `out` with the sum of every sounding voice, one channel. Overwrites
    /// rather than adds — the caller's buffer is whatever the driver last left
    /// there, which is the previous block of audio if it is anything at all.
    pub fn mix(&mut self, out: &mut [f32]) {
        let rate = self.rate as f32;
        // Borrowed once outside the loop, not per sample: the `Arc` deref is
        // free but the branch on `None` in an inner loop is not, and this loop
        // runs a few hundred thousand times a second.
        let block = self.clips.as_deref().map_or(&[][..], Bank::block);
        let target = self.target;
        // One glide's worth of gain per sample. Computed once: the division
        // is the only one in this loop and it does not vary.
        let glide = 1.0 / samples(GLIDE_MS, self.rate).max(1) as f32;
        let mut master = self.master;
        for sample in out.iter_mut() {
            let delta = target - master;
            master += delta.clamp(-glide, glide);
            if delta.abs() <= glide {
                master = target;
            }
            let mut sum = 0.0;
            for voice in &mut self.voices {
                let Some(playing) = voice else { continue };
                let live = if playing.wave == wave::CLIP_LOOP {
                    master
                } else {
                    1.0
                };
                match playing.sample(rate, block) {
                    Some(value) => sum += value * live,
                    None => *voice = None,
                }
            }
            *sample = limit(sum);
        }
        // Carried across buffers, or the glide would restart from where the last
        // one began and never arrive.
        self.master = master;
        // Retire what ended exactly on this buffer's last sample. Without it a
        // note holds its slot until the *next* buffer asks it for a sample it
        // does not have, which makes `sounding()` off by one buffer — and a
        // count that is only eventually right is not a count a gate can use.
        for voice in &mut self.voices {
            if voice.is_some_and(|v| v.finished()) {
                *voice = None;
            }
        }
    }

    /// Voices currently sounding. The observable the device tests assert on, and
    /// what makes "nothing played" a measurement rather than a listen.
    #[must_use]
    pub fn sounding(&self) -> usize {
        self.voices.iter().flatten().count()
    }

    /// Cut every voice. What a reload does, so a note started by the build being
    /// replaced does not outlive it.
    pub fn silence(&mut self) {
        self.voices = [None; VOICES];
    }
}

/// Soft-knee limiter. Linear to [`HEADROOM`], then asymptotic to 1.0, so a
/// summed bank cannot leave `-1.0..=1.0` and a single quiet voice is untouched
/// by arithmetic it should not be paying for.
fn limit(sum: f32) -> f32 {
    if sum.abs() <= HEADROOM {
        return sum;
    }
    let over = sum.abs() - HEADROOM;
    let compressed = HEADROOM + (1.0 - HEADROOM) * (over / (over + (1.0 - HEADROOM)));
    compressed.copysign(sum)
}

#[cfg(test)]
mod tests {
    // unwrap is permitted in tests (§2, Error handling row).
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const RATE: u32 = 48_000;

    fn fire(mixer: &mut Mixer, sound: Sound) {
        mixer.fire(&Trigger { slot: 0, sound });
    }

    fn peak(samples: &[f32]) -> f32 {
        samples.iter().fold(0.0_f32, |m, s| m.max(s.abs()))
    }

    #[test]
    fn a_fired_note_sounds_and_then_stops_on_its_own() {
        let mut mixer = Mixer::new(RATE);
        fire(&mut mixer, Sound::tone(wave::SQUARE, 440.0, 10, 1.0));
        assert_eq!(mixer.sounding(), 1);

        // 10 ms is 480 samples; mixing 400 leaves the note running.
        let mut out = [0.0; 400];
        mixer.mix(&mut out);
        assert_eq!(mixer.sounding(), 1);
        assert!(peak(&out) > 0.5, "a square at gain 1.0 barely moved");

        mixer.mix(&mut out);
        assert_eq!(mixer.sounding(), 0, "the note outlived its length");
        // And a finished bank writes silence rather than leaving the caller's
        // buffer alone — the driver's buffer holds the previous block.
        let mut stale = [1.0; 16];
        mixer.mix(&mut stale);
        assert!(stale.iter().all(|s| *s == 0.0));
    }

    /// The property the whole envelope exists for. A note that started or ended
    /// at full amplitude is a step, and a step is a click on every trigger.
    #[test]
    fn a_note_starts_and_ends_at_silence() {
        let mut mixer = Mixer::new(RATE);
        // Below `HEADROOM`, so this measures the envelope and not the limiter —
        // a lone voice at gain 1.0 comes out at 0.9 by design, which would make
        // the peak assertion below a test of two things at once.
        fire(&mut mixer, Sound::tone(wave::SQUARE, 200.0, 100, 0.7));
        let mut out = [0.0; 4_800];
        mixer.mix(&mut out);
        assert_eq!(out[0], 0.0, "the first sample is the attack's own zero");
        assert_eq!(*out.last().unwrap(), 0.0, "and the last is the release's");
        assert!(
            (peak(&out) - 0.7).abs() < 1e-6,
            "and full amplitude in between"
        );
    }

    /// A blip shorter than its own envelope: scaled to fit, still silent at both
    /// ends, still audible. Written the way a game writes it — `Sound::tone`
    /// asks for a quarter-length release, so only a hand-set envelope gets here.
    #[test]
    fn an_envelope_longer_than_its_note_is_fitted_rather_than_refused() {
        let mut mixer = Mixer::new(RATE);
        let mut blip = Sound::tone(wave::SINE, 880.0, 4, 1.0);
        blip.attack_ms = 20;
        blip.release_ms = 20;
        fire(&mut mixer, blip);
        let mut out = [0.0; 192];
        mixer.mix(&mut out);
        assert_eq!(mixer.sounding(), 0);
        assert_eq!(out[0], 0.0);
        assert!(peak(&out) > 0.0, "fitting the envelope silenced the note");
    }

    #[test]
    fn a_silent_or_unknown_wave_takes_no_slot() {
        let mut mixer = Mixer::new(RATE);
        fire(&mut mixer, Sound::tone(wave::SILENT, 440.0, 100, 1.0));
        assert_eq!(mixer.sounding(), 0);
        // The forward-compatibility claim in `boundary::audio`: a wave constant
        // a future game knows and this build does not is silence, not a wrong
        // sound and not a panic.
        fire(&mut mixer, Sound::tone(wave::NOISE + 7, 440.0, 100, 1.0));
        assert_eq!(mixer.sounding(), 0);
        fire(&mut mixer, Sound::tone(wave::SQUARE, 440.0, 100, 0.0));
        assert_eq!(mixer.sounding(), 0, "gain 0 is silence with a slot held");
        fire(&mut mixer, Sound::tone(wave::SQUARE, 440.0, 0, 1.0));
        assert_eq!(mixer.sounding(), 0, "and so is a note of no length");
    }

    #[test]
    fn a_note_longer_than_the_cap_is_clamped_rather_than_held_forever() {
        let mut mixer = Mixer::new(1_000); // 1 sample per ms, so lengths are legible
        fire(
            &mut mixer,
            Sound::tone(wave::SQUARE, 100.0, MAX_MS * 4, 1.0),
        );
        let mut half = vec![0.0; MAX_MS as usize / 2];
        mixer.mix(&mut half);
        assert_eq!(
            mixer.sounding(),
            1,
            "clamped to something shorter than MAX_MS"
        );
        mixer.mix(&mut half);
        assert_eq!(
            mixer.sounding(),
            0,
            "the clamp is MAX_MS and it was not applied"
        );
    }

    /// Retriggering a slot cuts the previous note off rather than stacking, and
    /// a second entity is how a game gets two at once.
    #[test]
    fn a_slot_holds_one_note_and_distinct_slots_stack() {
        let mut mixer = Mixer::new(RATE);
        let note = Sound::tone(wave::SQUARE, 440.0, 500, 0.2);
        mixer.fire(&Trigger {
            slot: 0,
            sound: note,
        });
        mixer.fire(&Trigger {
            slot: 0,
            sound: note,
        });
        assert_eq!(mixer.sounding(), 1);
        mixer.fire(&Trigger {
            slot: 1,
            sound: note,
        });
        assert_eq!(mixer.sounding(), 2);
        // Past the bank, a slot wraps rather than panicking or being dropped:
        // the observer hashes an entity into it and must not be able to crash
        // the audio thread by spawning entities.
        mixer.fire(&Trigger {
            slot: VOICES,
            sound: note,
        });
        assert_eq!(
            mixer.sounding(),
            2,
            "slot VOICES is slot 0 and it was already live"
        );
    }

    /// The limiter's whole job. Ten voices at full gain sum to ten; a clip would
    /// be broadband distortion on the busiest frames, and unbounded output would
    /// be whatever the driver does with 10.0.
    #[test]
    fn a_full_bank_stays_inside_the_output_range() {
        let mut mixer = Mixer::new(RATE);
        for slot in 0..10 {
            mixer.fire(&Trigger {
                slot,
                sound: Sound::tone(wave::SQUARE, 100.0 + slot as f32, 100, 1.0),
            });
        }
        let mut out = [0.0; 2_400];
        mixer.mix(&mut out);
        assert!(
            peak(&out) <= 1.0,
            "the bank left the output range: {}",
            peak(&out)
        );
        assert!(
            peak(&out) > HEADROOM,
            "ten voices should be working the limiter"
        );
        // And one quiet voice is not paying for the limiter's existence.
        let mut alone = Mixer::new(RATE);
        alone.fire(&Trigger {
            slot: 0,
            sound: Sound::tone(wave::SQUARE, 100.0, 100, 0.5),
        });
        let mut quiet = [0.0; 2_400];
        alone.mix(&mut quiet);
        assert!(
            (peak(&quiet) - 0.5).abs() < 1e-6,
            "a lone voice was compressed"
        );
    }

    /// Noise is seeded from the trigger count, so the same sequence of fires
    /// produces the same samples — "record it and listen again" is only useful
    /// if the recording sounds like the run.
    #[test]
    fn noise_repeats_for_the_same_sequence_of_triggers() {
        let render = || {
            let mut mixer = Mixer::new(RATE);
            fire(&mut mixer, Sound::tone(wave::NOISE, 0.0, 20, 1.0));
            let mut out = [0.0; 960];
            mixer.mix(&mut out);
            out
        };
        assert_eq!(render(), render());
        assert!(peak(&render()) > 0.5, "the noise voice made no noise");
    }

    #[test]
    fn a_sweep_changes_frequency_across_the_note() {
        // Zero crossings in the first half against the second: a rising sweep
        // has fewer early than late, and counting them beats asserting samples.
        let mut mixer = Mixer::new(RATE);
        fire(
            &mut mixer,
            Sound::sweep(wave::SQUARE, 100.0, 1_600.0, 200, 1.0),
        );
        let mut out = [0.0; 9_600];
        mixer.mix(&mut out);
        let crossings = |half: &[f32]| half.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
        let (early, late) = out.split_at(out.len() / 2);
        assert!(
            crossings(late) > crossings(early) * 2,
            "the sweep did not rise: {} then {}",
            crossings(early),
            crossings(late)
        );
    }

    // --------------------------------------------------------- §6 M43's clips

    /// A clip whose envelope is one sample at each end, so an assertion about a
    /// sample is an assertion about the *clip* rather than about the ramp.
    fn cue(name: &str, gain: f32) -> Sound {
        Sound {
            attack_ms: 0,
            release_ms: 0,
            ..Sound::clip(name, gain)
        }
    }

    fn banked(rate: u32, samples: &[i16]) -> Arc<Bank> {
        let mut bank = Bank::new();
        bank.add(gg_ecs::boundary::asset_id("cue"), rate, samples);
        Arc::new(bank)
    }

    /// The claim the whole milestone rests on: what comes out is what `ggc`
    /// put in. At the device's own rate the resampler steps by exactly one, so
    /// the samples pass through untouched but for gain and the two-sample ramp.
    #[test]
    fn a_clip_plays_the_samples_it_was_handed() {
        let mut mixer = Mixer::new(RATE);
        mixer.set_bank(banked(RATE, &[0, 8_192, 16_384, -8_192, 0]));
        fire(&mut mixer, cue("cue", 1.0));

        let mut out = [0.0; 5];
        mixer.mix(&mut out);
        assert_eq!(out[1], 0.25);
        assert_eq!(out[2], 0.5);
        assert_eq!(out[3], -0.25);
        assert_eq!(out[0], 0.0, "the attack's own zero");
        assert_eq!(out[4], 0.0, "and the release's");
        assert_eq!(mixer.sounding(), 0, "a one-shot outlived its samples");
    }

    /// The resampler, in the direction every real clip takes: authored at half
    /// the device's rate, so one source sample becomes two output ones and the
    /// note lasts twice as long. Nothing about this is knowable at bake time —
    /// which is why `gg_assets::clip` stores the source's rate and stops.
    #[test]
    fn a_clip_authored_at_another_rate_is_stretched_rather_than_pitched() {
        let mut mixer = Mixer::new(RATE);
        mixer.set_bank(banked(RATE / 2, &[16_384; 8]));
        fire(&mut mixer, cue("cue", 1.0));

        let mut short = [0.0; 15];
        mixer.mix(&mut short);
        assert_eq!(mixer.sounding(), 1, "8 frames at half rate is 16 samples");
        let mut last = [0.0; 1];
        mixer.mix(&mut last);
        assert_eq!(mixer.sounding(), 0);
        // And the interpolation did not invent amplitude: a constant clip is
        // constant however it was stepped through.
        assert!(short[1..].iter().all(|s| (*s - 0.5).abs() < 1e-6));
    }

    /// Music: it repeats, and it does not end on its own. What ends it is the
    /// protocol `Sound` already had — a silent trigger on the same slot.
    #[test]
    fn a_loop_repeats_until_something_takes_the_voice() {
        let mut mixer = Mixer::new(RATE);
        mixer.set_bank(banked(RATE, &[16_384, -16_384]));
        fire(&mut mixer, Sound::music("cue", 1.0));

        let mut out = [0.0; 4_096];
        for _ in 0..8 {
            mixer.mix(&mut out);
            assert_eq!(mixer.sounding(), 1, "the loop ended by itself");
        }
        assert!(peak(&out) > 0.4, "the loop went quiet on a later pass");

        // A silent trigger asks it to end, and since §6 M77 it *ends* rather
        // than vanishing: the slot stays busy for the release and then goes.
        // Through M76 this asserted silence on the very next call, which is
        // the cut that measured a step of 0.202 against a signal whose own
        // largest was 0.030.
        fire(&mut mixer, Sound::tone(wave::SILENT, 0.0, 100, 1.0));
        assert_eq!(mixer.sounding(), 1, "the stop is a fade, not a delete");
        let fade = samples(gg_ecs::boundary::FADE_MS, RATE) as usize;
        let mut tail = vec![0.0; fade + 64];
        mixer.mix(&mut tail);
        assert_eq!(mixer.sounding(), 0, "and it is over inside its release");
        assert_eq!(
            tail[fade + 8..].iter().fold(0.0f32, |m, s| m.max(s.abs())),
            0.0,
            "and stays over"
        );
    }

    /// The fade itself: a stopped loop comes down rather than off.
    ///
    /// Graded as a **monotone envelope** rather than as a step count, because
    /// the material underneath is still playing — what has to fall is the
    /// loudest sample in each successive window, and it has to reach zero.
    /// `gg-tools clicks` is the instrument this threshold came off (§6 M77).
    #[test]
    fn a_stopped_loop_comes_down_instead_of_off() {
        let mut mixer = Mixer::new(RATE);
        // A square at full scale: the worst case for a cut, since every
        // sample is at the peak and there is no quiet moment to stop in.
        mixer.set_bank(banked(RATE, &[24_000, -24_000, 24_000, -24_000]));
        fire(&mut mixer, Sound::music("cue", 1.0));
        let fade = samples(gg_ecs::boundary::FADE_MS, RATE) as usize;

        // Past the attack, so what is measured below is the release.
        let mut warm = vec![0.0; fade * 2];
        mixer.mix(&mut warm);
        let loud = peak(&warm);
        assert!(loud > 0.5, "the loop is at full scale before the stop");

        fire(&mut mixer, Sound::tone(wave::SILENT, 0.0, 100, 1.0));
        let mut out = vec![0.0; fade];
        mixer.mix(&mut out);

        // Eight windows across the release, each quieter than the last.
        let step = out.len() / 8;
        let mut last = f32::INFINITY;
        for window in out.chunks(step) {
            let level = peak(window);
            assert!(
                level < last,
                "the fade stopped falling at {level} after {last}"
            );
            last = level;
        }
        assert!(last < loud / 8.0, "and it got most of the way down");

        // The claim the instrument makes: no sample-to-sample step in the
        // fade is larger than the ones the material was already making.
        let own = out
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            own <= loud * 2.0,
            "the fade introduced a step of {own} into a signal of {loud}"
        );
    }

    /// A pause taken back inside the fade keeps the phrase.
    ///
    /// The claim that matters is not the seam — `gg-tools clicks` reads that at
    /// 0.04 where a fresh voice read 1.46 — but *where the music is*: a new
    /// voice reads from sample 0, so before §6 M77 pausing demo 10 and resuming
    /// dropped the needle at the top of the track every time.
    ///
    /// Graded as an A/B against the same act performed a moment later, once the
    /// fade has finished — which is a genuine restart and stays one. Both runs
    /// end the same distance into the session, so the only difference between
    /// them is whether the cursor was kept, and no number here is a threshold.
    #[test]
    fn a_pause_taken_back_inside_the_fade_keeps_its_place() {
        let fade = samples(gg_ecs::boundary::FADE_MS, RATE) as usize;
        // A ramp long enough never to wrap, so a sample *is* a position.
        let steps: Vec<i16> = (0..400_000_i64)
            .map(|i| (i * 32_000 / 400_000) as i16)
            .collect();

        // `gap` is how long the player leaves it before changing their mind.
        let run = |gap: usize| {
            let mut mixer = Mixer::new(RATE);
            mixer.set_bank(banked(RATE, &steps));
            fire(&mut mixer, Sound::music("cue", 1.0));
            let mut warm = vec![0.0; fade * 2];
            mixer.mix(&mut warm);

            fire(&mut mixer, Sound::tone(wave::SILENT, 0.0, 100, 1.0));
            let mut waited = vec![0.0; gap];
            mixer.mix(&mut waited);
            let voices = mixer.sounding();

            fire(&mut mixer, Sound::music("cue", 1.0));
            // Long enough to be past the attack either way, so what is compared
            // is the clip's own value and not an envelope.
            let mut after = vec![0.0; fade * 4];
            mixer.mix(&mut after);
            (voices, peak(&after))
        };

        // Inside the fade: one voice all along, and it never stopped.
        let (voices, resumed) = run(fade / 4);
        assert_eq!(voices, 1, "the fade was still running");

        // And after it: the voice had gone, so this one starts over.
        let (gone, restarted) = run(fade * 2);
        assert_eq!(gone, 0, "the fade had finished and the slot was free");

        assert!(
            resumed > restarted,
            "a pause taken back read {resumed} where starting over reads \
             {restarted} — the resumed voice did not keep its place"
        );
    }

    /// The master reaches a new value over a ramp, not on the next sample.
    ///
    /// A suspend is `set_master(0.0)` (§6 M49), and it reached every looping
    /// voice at once — `gg-tools clicks` measured that cut at 0.202 against a
    /// signal whose own largest step was 0.030.
    #[test]
    fn the_master_glides_to_where_it_was_sent() {
        let mut mixer = Mixer::new(RATE);
        mixer.set_bank(banked(RATE, &[24_000, -24_000, 24_000, -24_000]));
        fire(&mut mixer, Sound::music("cue", 1.0));
        let mut warm = vec![0.0; samples(gg_ecs::boundary::FADE_MS, RATE) as usize * 2];
        mixer.mix(&mut warm);
        let loud = peak(&warm);

        mixer.set_master(0.0);
        // One glide's worth, plus a margin for the sample it lands on.
        let glide = samples(10, RATE) as usize;
        let mut out = vec![0.0; glide * 4];
        mixer.mix(&mut out);
        assert!(
            peak(&out[..glide / 4]) > loud / 4.0,
            "the first samples after a suspend are not silent"
        );
        assert_eq!(
            peak(&out[glide * 2..]),
            0.0,
            "and it is silent well inside the glide"
        );
        // Back up, and the loop is still there to come back — a glide is not
        // a stop.
        mixer.set_master(1.0);
        let mut back = vec![0.0; glide * 4];
        mixer.mix(&mut back);
        assert!(peak(&back) > loud / 2.0, "the music came back");
    }

    /// A pack rebuilt while the game runs, and a game shipped without its pack:
    /// both land here, and both must be silent rather than loud or fatal.
    #[test]
    fn a_clip_no_bank_holds_is_silence_and_takes_no_slot() {
        let mut mixer = Mixer::new(RATE);
        // No bank at all — a run with no `--pack`.
        fire(&mut mixer, cue("cue", 1.0));
        assert_eq!(mixer.sounding(), 0);

        mixer.set_bank(banked(RATE, &[16_384; 64]));
        fire(&mut mixer, cue("a name nothing baked", 1.0));
        assert_eq!(mixer.sounding(), 0);
        // A clip id of 0 is the zeroed component `World::restore` writes.
        let mut nothing = cue("cue", 1.0);
        nothing.clip = 0;
        fire(&mut mixer, nothing);
        assert_eq!(mixer.sounding(), 0);
        // And the one that does resolve still plays, so the three above are
        // refusals rather than a mixer that stopped working.
        fire(&mut mixer, cue("cue", 1.0));
        assert_eq!(mixer.sounding(), 1);
    }

    #[test]
    fn a_length_on_a_clip_truncates_it() {
        let mut mixer = Mixer::new(1_000); // 1 sample per ms
        mixer.set_bank(banked(1_000, &[16_384; 500]));
        let mut clipped = cue("cue", 1.0);
        clipped.ms = 100;
        fire(&mut mixer, clipped);

        let mut out = [0.0; 99];
        mixer.mix(&mut out);
        assert_eq!(mixer.sounding(), 1);
        mixer.mix(&mut [0.0; 1]);
        assert_eq!(mixer.sounding(), 0, "ms did not truncate the clip");

        // And a length past the clip's own is the clip's own, not silence
        // padded out to it — `ms` truncates and never extends.
        let mut long = cue("cue", 1.0);
        long.ms = 10_000;
        fire(&mut mixer, long);
        mixer.mix(&mut [0.0; 500]);
        assert_eq!(mixer.sounding(), 0);
    }

    /// The asymmetry §6 M43 introduces, in both directions: a slider moved
    /// while music plays is heard now, and a one-shot already carries the
    /// volume it fired at (§6 M19) so applying it twice would square it.
    #[test]
    fn the_master_volume_reaches_a_loop_and_leaves_a_one_shot_alone() {
        let mut mixer = Mixer::new(RATE);
        mixer.set_bank(banked(RATE, &[16_384; 512]));

        fire(&mut mixer, Sound::music("cue", 1.0));
        // Past the fade in, and past the glide after each move: since §6 M77 the
        // music arrives over `FADE_MS` and the slider over `GLIDE_MS`, so this
        // measures where the two *land* rather than where they set off.
        let settle = samples(gg_ecs::boundary::FADE_MS, RATE) as usize * 2;
        let mut warm = vec![0.0; settle];
        mixer.mix(&mut warm);
        let mut out = [0.0; 256];
        mixer.mix(&mut out);
        assert!((peak(&out) - 0.5).abs() < 1e-6);

        mixer.set_master(0.5);
        mixer.mix(&mut warm);
        mixer.mix(&mut out);
        assert!(
            (peak(&out) - 0.25).abs() < 1e-6,
            "the slider did not reach the music: {}",
            peak(&out)
        );

        // The one-shot's gain is the trigger's, already scaled by `Audio::tick`.
        let mut alone = Mixer::new(RATE);
        alone.set_bank(banked(RATE, &[16_384; 512]));
        alone.set_master(0.5);
        alone.fire(&Trigger {
            slot: 0,
            sound: cue("cue", 1.0),
        });
        alone.mix(&mut out);
        assert!(
            (peak(&out) - 0.5).abs() < 1e-6,
            "the master was applied to a one-shot twice: {}",
            peak(&out)
        );
    }

    /// Swapping the bank cuts every voice, and it has to: a voice holds offsets
    /// into the block it started against, so the alternative is reading the
    /// wrong samples out of the right array.
    #[test]
    fn installing_a_bank_cuts_what_was_playing_out_of_the_old_one() {
        let mut mixer = Mixer::new(RATE);
        mixer.set_bank(banked(RATE, &[16_384; 4_096]));
        fire(&mut mixer, cue("cue", 1.0));
        assert_eq!(mixer.sounding(), 1);
        mixer.set_bank(banked(RATE, &[8_192; 8]));
        assert_eq!(mixer.sounding(), 0);
    }

    #[test]
    fn silence_cuts_every_voice() {
        let mut mixer = Mixer::new(RATE);
        for slot in 0..4 {
            mixer.fire(&Trigger {
                slot,
                sound: Sound::tone(wave::SINE, 440.0, 1_000, 0.5),
            });
        }
        assert_eq!(mixer.sounding(), 4);
        mixer.silence();
        assert_eq!(mixer.sounding(), 0);
    }
}
