//! The mixer: triggers in, mono samples out. No device, no threads, no clock —
//! which is why every claim in §6 M18 item 2 about *what a cue sounds like* is
//! provable in an ordinary unit test on a machine that is silent.
//!
//! The split is `gg-rhi`'s: [`device`](super::device) is the only part that
//! talks to an OS, and it is a buffer's worth of arithmetic away from anything
//! testable. Everything with an opinion is here.

use gg_ecs::boundary::{MAX_MS, Sound, wave};
use gg_math::sim;

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
}

impl Voice {
    /// `None` for a note that would make no sound: an unknown or silent wave, a
    /// zero length, or a gain of nothing. Cheaper to refuse here than to mix
    /// silence for 4000 ms while holding a slot.
    fn new(sound: &Sound, rate: u32, seed: u64) -> Option<Voice> {
        if sound.wave == wave::SILENT || sound.wave > wave::NOISE {
            return None;
        }
        let gain = sound.gain.clamp(0.0, 1.0);
        let len = samples(sound.ms.min(MAX_MS), rate);
        if gain <= 0.0 || len == 0 {
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
            release,
            rng: sim::Rng::from_seed(seed),
        })
    }

    /// The next sample, or `None` once the note has finished.
    fn sample(&mut self, rate: f32) -> Option<f32> {
        if self.at >= self.len {
            return None;
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

/// The bank. Owned by the audio callback and reached from nowhere else.
pub struct Mixer {
    voices: [Option<Voice>; VOICES],
    rate: u32,
    /// Counts triggers, and is the noise seed. Not a clock: it advances once per
    /// note rather than with time, which is what makes a replayed session's
    /// noise identical to the original's.
    fired: u64,
}

impl Mixer {
    /// A silent bank at `rate` samples per second.
    #[must_use]
    pub fn new(rate: u32) -> Mixer {
        Mixer {
            voices: [None; VOICES],
            rate: rate.max(1),
            fired: 0,
        }
    }

    /// Start `trigger`, replacing whatever was in its slot.
    ///
    /// Replacing and not queueing: the slot *is* the game's entity, and a game
    /// that retriggered the same `Sound` twice in three ticks meant the second
    /// one. A note that wanted to ring over its successor is two entities.
    pub fn fire(&mut self, trigger: &Trigger) {
        let slot = trigger.slot % VOICES;
        self.fired = self.fired.wrapping_add(1);
        self.voices[slot] = Voice::new(&trigger.sound, self.rate, self.fired);
    }

    /// Fill `out` with the sum of every sounding voice, one channel. Overwrites
    /// rather than adds — the caller's buffer is whatever the driver last left
    /// there, which is the previous block of audio if it is anything at all.
    pub fn mix(&mut self, out: &mut [f32]) {
        let rate = self.rate as f32;
        for sample in out.iter_mut() {
            let mut sum = 0.0;
            for voice in &mut self.voices {
                let Some(playing) = voice else { continue };
                match playing.sample(rate) {
                    Some(value) => sum += value,
                    None => *voice = None,
                }
            }
            *sample = limit(sum);
        }
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
