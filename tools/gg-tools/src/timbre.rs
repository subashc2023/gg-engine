//! The sound a synthesizer cannot make, written down and then measured (§6 M43).
//!
//! # Why the audio in this tree is generated and not recorded
//!
//! [`panorama`](super::panorama)'s argument, one medium over: a recorded cue is
//! somebody else's work under somebody else's licence, and a checked-in binary
//! nobody can regenerate quietly becomes load-bearing. Every `.wav` under
//! `demos/10-tetris/assets/` comes out of this file, so it has a source, a diff
//! that changes it is reviewable as *code*, and nothing in the repository is a
//! recording we did not make.
//!
//! # The question it exists to answer
//!
//! §3 stood a non-goal for eleven milestones — "no audio asset format before a
//! milestone asks for one" — and M43 lifts it. So the first thing to establish
//! is that a milestone really is asking, which is a claim about the *protocol*
//! rather than about taste: `Sound` synthesizes one wave at one frequency,
//! swept linearly, under a linear attack-and-release. Four things it therefore
//! cannot produce at all, and each generated clip is built on one of them:
//!
//! - **Inharmonic partials.** A struck bar rings at 1, 2.76, 5.40, 8.93× its
//!   fundamental. One oscillator has one frequency, and every wave in [`wave`]
//!   has *harmonic* overtones — integer multiples — by construction.
//! - **Several pitches at once.** A chord is three frequencies; a voice is one,
//!   and three `Sound` entities is three envelopes a game has to keep in step.
//! - **Exponential decay.** The envelope is linear at both ends. A struck thing
//!   decays exponentially, and the difference is the whole character of a hit.
//! - **Partials that decay at different rates.** The bright ones die first,
//!   which is what makes a struck sound *change colour* as it fades.
//!
//! # The two numbers, and why they fail in opposite directions
//!
//! **Reach** is how far the closest `Sound` this protocol can express lands
//! from the clip, as a fraction of the clip's own spectrum. It is large when
//! the milestone was worth building — a small reach would mean the synthesizer
//! could have made this and the pack is carrying kilobytes for nothing.
//!
//! **Loss** is how far what comes *out of the engine* lands from the `.wav` that
//! went in, through the real bake and the real mixer — `ggc::clip::parse`, the
//! blob, the bank, the voice. It must be small, and it is reported twice: at the
//! clip's own rate, where the resampler steps by exactly one and anything but
//! zero is a defect, and at 48 kHz, where the cost is linear interpolation and
//! is the honest price of not resampling at bake time.
//!
//! Neither number needs a reference implementation. The reach's reference is the
//! shipped `Mixer` rendering the shipped `Sound`, and the loss's reference is the
//! source file — restating none of either to get there.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use gg_audio::{Bank, Mixer, Trigger};
use gg_ecs::boundary::{Sound, asset_id, wave};
use gg_math::sim;

/// Everything here is authored at 22.05 kHz. Half of CD rate, four times the
/// highest partial any of these clips carries, and the rate a game of this kind
/// has always used — the pack stores what the source ran at, so the choice
/// costs nothing downstream but bytes.
const RATE: u32 = 22_050;

/// Samples the spectral comparison runs over. A power of two out of habit
/// rather than necessity — the transform below is the naive one, because this
/// is a manual instrument and an FFT would be sixty lines of index arithmetic
/// nobody would check.
const WINDOW: usize = 1_024;

pub fn run(args: &[String]) -> Result<()> {
    // Under `target/` by default, as `panorama` is: an instrument writes a
    // report, and putting a source file into the tree is a decision someone
    // spells out. `--out demos/10-tetris/assets` is how these got there.
    let mut out = PathBuf::from("target/gg-tools/timbre");
    let mut grade = true;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        match arg.as_str() {
            "--out" => out = PathBuf::from(rest.next().context("--out wants a path")?),
            // For the case where the point is to hear the files rather than to
            // read the table — writing takes a second, grading takes a minute.
            "--no-grade" => grade = false,
            other => bail!("unknown argument {other}"),
        }
    }

    std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
    let clips = compose();
    for (name, samples) in &clips {
        let path = out.join(format!("{name}.wav"));
        write_wav(&path, RATE, samples)?;
        println!(
            "wrote {} — {} samples, {:.2} s, {} KiB",
            path.display(),
            samples.len(),
            samples.len() as f32 / RATE as f32,
            samples.len() * 2 / 1024
        );
    }
    if !grade {
        return Ok(());
    }

    println!(
        "\n{:<10} {:>9} {:>10} {:>11} {:>10}",
        "clip", "reach %", "loss %", "loss@48k %", "partials"
    );
    println!("{}", "-".repeat(54));
    for (name, samples) in &clips {
        let reach = reach(samples);
        let exact = loss(samples, RATE);
        let resampled = loss(samples, 48_000);
        println!(
            "{name:<10} {reach:>9.1} {exact:>10.3} {resampled:>11.2} {:>10}",
            partials(samples)
        );
    }
    println!(
        "\nreach: how far the closest Sound this protocol can express lands from the clip.\n\
         Large is the milestone's justification — a small one would mean the synthesizer\n\
         could have made it. loss: how far the engine's own output lands from the source,\n\
         through ggc, the blob, the bank and the mixer. At the clip's own rate the step is\n\
         exactly one and anything but zero is a defect; at 48 kHz it is the interpolator,\n\
         and it is what not resampling at bake time costs."
    );
    Ok(())
}

// ----------------------------------------------------------------- the clips

/// The four sources, each built on something [`wave`] has no way to produce.
fn compose() -> Vec<(&'static str, Vec<i16>)> {
    vec![
        ("lock", lock()),
        ("level", level()),
        ("topout", topout()),
        ("theme", theme()),
    ]
}

/// A piece landing on the stack: a struck body, which is inharmonic partials
/// with per-partial exponential decay over a noise transient.
///
/// The tone it replaces was `sweep(TRIANGLE, 180 → 90, 60 ms)` — a falling boop.
/// What a lock should sound like is a *thunk*, and the difference is entirely in
/// the four things listed in the module docs.
fn lock() -> Vec<i16> {
    let mut voice = Additive::new(0.30);
    // The free bar's ratios. Nothing here is a multiple of anything here, which
    // is the property a single oscillator cannot have.
    for (ratio, gain, decay) in [
        (1.00, 1.00, 14.0),
        (2.76, 0.55, 26.0),
        (5.40, 0.28, 40.0),
        (8.93, 0.14, 60.0),
    ] {
        voice.partial(150.0 * ratio, gain, decay);
    }
    // The strike. Two milliseconds of noise decaying fast is what makes a hit
    // read as contact rather than as a note starting.
    voice.transient(0.5, 260.0);
    voice.render()
}

/// A level gained: a major triad struck at once, ringing up.
///
/// Three pitches simultaneously, which is three `Sound` entities under the old
/// protocol — and three entities means three `seq` bumps a game has to land on
/// one tick, with three envelopes that drift apart the moment one is retuned.
fn level() -> Vec<i16> {
    let mut voice = Additive::new(0.55);
    // C5–E5–G5, and the octave above the root for brightness. Each partial keeps
    // its own decay, so the chord darkens as it fades.
    for (hz, gain, decay) in [
        (523.25, 0.85, 5.0),
        (659.25, 0.65, 6.0),
        (783.99, 0.55, 7.0),
        (1_046.50, 0.30, 11.0),
    ] {
        voice.partial(hz, gain, decay);
    }
    voice.render()
}

/// The run ending: a minor triad falling a whole register, each note entering
/// after the one before it and all of them still ringing at the end.
///
/// A sweep cannot do this — a sweep is one voice moving, so the note it left is
/// gone. What makes an ending sound final is that nothing is released.
fn topout() -> Vec<i16> {
    let mut voice = Additive::new(1.30);
    // A minor arpeggio down, then the root two octaves below under all of it.
    for (index, hz) in [440.00, 349.23, 261.63, 220.00, 110.00].iter().enumerate() {
        voice.entering(*hz, 0.7 - index as f32 * 0.07, 3.2, index as f32 * 0.16);
    }
    voice.render()
}

/// The music: four bars of i–VI–III–VII at 140, looping seamlessly.
///
/// Seamless is a property of the *file* and not of the mixer: the loop point is
/// a whole number of bars and every voice has decayed by it, so the interpolated
/// step from the last sample back into the first is a step between two zeros.
/// A track that did not end where it began would click once a bar, forever.
fn theme() -> Vec<i16> {
    const BPM: f32 = 140.0;
    let beat = 60.0 / BPM;
    let bars = 4;
    let mut voice = Additive::new(beat * 4.0 * bars as f32);

    // A minor, F, C, G — the progression, as (root, third, fifth) in Hz.
    let chords = [
        (110.00, 130.81, 164.81), // Am
        (87.31, 110.00, 130.81),  // F
        (130.81, 164.81, 196.00), // C
        (98.00, 123.47, 146.83),  // G
    ];
    for (bar, (root, third, fifth)) in chords.iter().enumerate() {
        let at = bar as f32 * beat * 4.0;
        // The bass: root on 1 and 3, an octave down, long and soft.
        for offset in [0.0, beat * 2.0] {
            voice.entering(root / 2.0, 0.42, 3.0, at + offset);
        }
        // The arpeggio: eight notes a beat apart in pairs, two octaves up, each
        // decaying fast enough to leave room for the next.
        for (step, hz) in [*root, *third, *fifth, *third, *root, *fifth, *third, *fifth]
            .iter()
            .enumerate()
        {
            voice.entering(hz * 4.0, 0.16, 11.0, at + step as f32 * beat * 0.5);
        }
    }
    // The hat: a noise tick on every off-beat, which is what makes it move.
    for tick in 0..(bars * 8) {
        voice.tick(0.10, 90.0, (tick as f32 + 0.5) * beat * 0.5);
    }
    voice.render()
}

/// Additive synthesis onto a fixed buffer: partials with independent
/// exponential decays, plus noise transients.
///
/// `sim`'s transcendentals and not `std`'s, for `panorama`'s reason exactly —
/// these files are checked in, so the generator that wrote them must give the
/// same bytes on every host or a regenerated clip is a diff nobody intended.
struct Additive {
    buffer: Vec<f32>,
    rng: sim::Rng,
}

impl Additive {
    fn new(seconds: f32) -> Additive {
        Additive {
            buffer: vec![0.0; (seconds * RATE as f32) as usize],
            // A fixed seed, because the noise is part of the file: a hat that
            // differed between two runs of this tool would make every
            // regeneration a diff.
            rng: sim::Rng::from_seed(0x6767_4d34_3300_0001),
        }
    }

    /// One sine partial from the start, decaying by `decay` nepers a second.
    fn partial(&mut self, hz: f32, gain: f32, decay: f32) {
        self.entering(hz, gain, decay, 0.0);
    }

    /// A partial that starts `at` seconds in, so notes can overlap without
    /// being separate voices — which is the whole of what a chord is.
    fn entering(&mut self, hz: f32, gain: f32, decay: f32, at: f32) {
        let start = (at * RATE as f32) as usize;
        for index in start..self.buffer.len() {
            let t = (index - start) as f32 / RATE as f32;
            let envelope = sim::exp(-decay * t);
            // Below a fifteen-bit code value nothing is left to add, and a
            // four-second theme would otherwise cost every partial a full pass.
            if envelope < 3e-5 {
                break;
            }
            // A two-millisecond ramp on the way in. A partial starting at full
            // amplitude is a step, and a step is the click the engine's own
            // envelope exists to remove — it should not have to fix the file.
            let rise = (t * 500.0).min(1.0);
            self.buffer[index] +=
                gain * rise * envelope * sim::sin(core::f32::consts::TAU * hz * t);
        }
    }

    /// A noise burst at the start, decaying fast — the contact in a hit.
    fn transient(&mut self, gain: f32, decay: f32) {
        self.tick(gain, decay, 0.0);
    }

    /// A noise burst `at` seconds in.
    fn tick(&mut self, gain: f32, decay: f32, at: f32) {
        let start = (at * RATE as f32) as usize;
        for index in start..self.buffer.len() {
            let t = (index - start) as f32 / RATE as f32;
            let envelope = sim::exp(-decay * t);
            if envelope < 1e-4 {
                break;
            }
            let noise = (self.rng.next_u32() >> 8) as f32 / 8_388_608.0 - 1.0;
            self.buffer[index] += gain * envelope * noise;
        }
    }

    /// Normalize to just under full scale and quantize.
    ///
    /// Normalized rather than trusted: additive synthesis sums to whatever it
    /// sums to, and a clip that clipped would be measuring the limiter instead
    /// of the milestone. 0.97 leaves the headroom the mixer's own limiter wants.
    fn render(mut self) -> Vec<i16> {
        let peak = self.buffer.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        if peak > 0.0 {
            let scale = 0.97 / peak;
            for sample in &mut self.buffer {
                *sample *= scale;
            }
        }
        self.buffer
            .iter()
            .map(|s| (s * 32_767.0).clamp(-32_768.0, 32_767.0) as i16)
            .collect()
    }
}

/// A 16-bit mono PCM `.wav`, written by hand.
///
/// By hand because the alternative is a dependency in a tool whose whole point
/// is that the tree owns its own audio path — and because the file `ggc` reads
/// back should be one this repository can also write.
fn write_wav(path: &Path, rate: u32, samples: &[i16]) -> Result<()> {
    let data = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes()); // bytes per second
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data as u32).to_le_bytes());
    for sample in samples {
        out.extend_from_slice(&sample.to_le_bytes());
    }
    let mut file =
        std::fs::File::create(path).with_context(|| format!("creating {}", path.display()))?;
    file.write_all(&out)
        .with_context(|| format!("writing {}", path.display()))
}

// ---------------------------------------------------------------- the grading

/// Magnitude spectrum of the loudest [`WINDOW`] samples, Hann-windowed and
/// normalized to unit sum.
///
/// The *loudest* window rather than the first: a clip whose transient is two
/// milliseconds long would otherwise be compared against a tone over a region
/// where both are near silence, and every candidate would score the same.
/// Normalized because what is under test is the shape of the sound, not how
/// loud it happens to be — gain is the one thing a `Sound` can always match.
fn spectrum(samples: &[i16]) -> Vec<f32> {
    spectrum_at(samples, loudest(samples))
}

/// The same, over the [`WINDOW`] samples beginning at `start`.
fn spectrum_at(samples: &[i16], start: usize) -> Vec<f32> {
    let window: Vec<f32> = (0..WINDOW)
        .map(|i| {
            let sample = samples.get(start + i).copied().unwrap_or(0);
            let hann =
                0.5 - 0.5 * sim::cos(core::f32::consts::TAU * i as f32 / (WINDOW as f32 - 1.0));
            f32::from(sample) / 32_768.0 * hann
        })
        .collect();

    // One turn of the unit circle, once. The naive transform's inner term is
    // `exp(-2πi·bin·i/N)`, whose argument only ever takes `WINDOW` distinct
    // values modulo a turn — so the table below is not an optimization of the
    // arithmetic, it is the arithmetic with the redundancy removed, and it
    // takes the grading pass from minutes to seconds.
    let turn: Vec<(f32, f32)> = (0..WINDOW)
        .map(|k| sim::sin_cos(core::f32::consts::TAU * k as f32 / WINDOW as f32))
        .map(|(sin, cos)| (cos, sin))
        .collect();

    let mut magnitudes = Vec::with_capacity(WINDOW / 2);
    for bin in 0..WINDOW / 2 {
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (i, value) in window.iter().enumerate() {
            let (cos, sin) = turn[bin * i % WINDOW];
            re += value * cos;
            im -= value * sin;
        }
        magnitudes.push((re * re + im * im).sqrt());
    }
    let total: f32 = magnitudes.iter().sum();
    if total > 0.0 {
        for magnitude in &mut magnitudes {
            *magnitude /= total;
        }
    }
    magnitudes
}

/// Where the loudest [`WINDOW`]-sample stretch begins.
fn loudest(samples: &[i16]) -> usize {
    if samples.len() <= WINDOW {
        return 0;
    }
    let mut best = (0usize, 0i64);
    // Stepped rather than slid: an exact argmax over every offset costs a pass
    // per sample for an answer that moves by less than a bin.
    for start in (0..samples.len() - WINDOW).step_by(WINDOW / 8) {
        let energy: i64 = samples[start..start + WINDOW]
            .iter()
            .map(|s| i64::from(*s) * i64::from(*s))
            .sum();
        if energy > best.1 {
            best = (start, energy);
        }
    }
    best.0
}

/// How far two normalized spectra are apart, as a percentage — half the L1
/// distance, so two spectra sharing nothing score 100 and identical ones score
/// 0. (Half, because each unit of magnitude moved is counted at both ends.)
fn distance(a: &[f32], b: &[f32]) -> f32 {
    let sum: f32 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
    sum * 50.0
}

/// Windows a candidate is graded over, spread evenly across the clip.
///
/// More than one, and this is the correction that matters: a single window is
/// 46 ms, and against one window a stationary tone scores respectably on a
/// seven-second progression — which says nothing at all, since one `Sound` has
/// to *be* the whole clip. Graded across the length, a tone has to match every
/// instant of the thing with one sound, which is the actual claim.
const WINDOWS: usize = 6;

/// The closest any `Sound` gets, as a percentage.
///
/// Rendered through the shipped [`Mixer`], not through a model of it: the
/// question is what the *engine* can produce, and a reimplementation here would
/// answer it about this file instead. The candidate is compared window for
/// window at the same *fraction* of its own length, so a tone is never punished
/// for being a different number of samples than the clip.
fn reach(clip: &[i16]) -> f32 {
    let want = spectra(clip);
    let ms = (clip.len() as u32 * 1_000 / RATE).max(1);
    let mut best = f32::INFINITY;
    for shape in [wave::SQUARE, wave::SINE, wave::TRIANGLE, wave::NOISE] {
        // 32 frequencies, geometric from 55 Hz to 3.5 kHz — six octaves at a
        // fifth of a semitone, which is finer than a bin at this window size.
        for step in 0..32 {
            let hz = 55.0 * sim::exp(step as f32 / 31.0 * sim::ln(3_500.0 / 55.0));
            // A steady tone and a sweep to each end of the range: the
            // protocol's own two shapes, given every chance.
            for to in [hz, hz * 0.5, hz * 2.0] {
                let sound = Sound::sweep(shape, hz, to, ms.min(4_000), 1.0);
                let rendered = render(&sound, None, 48_000);
                let got = spectra_of(&rendered);
                let mean: f32 = want
                    .iter()
                    .zip(&got)
                    .map(|(a, b)| distance(a, b))
                    .sum::<f32>()
                    / WINDOWS as f32;
                best = best.min(mean);
            }
        }
    }
    best
}

/// [`WINDOWS`] spectra, at even fractions of the clip's length.
fn spectra(samples: &[i16]) -> Vec<Vec<f32>> {
    let span = samples.len().saturating_sub(WINDOW);
    (0..WINDOWS)
        .map(|i| spectrum_at(samples, span * i / WINDOWS.max(1)))
        .collect()
}

fn spectra_of(rendered: &[f32]) -> Vec<Vec<f32>> {
    let quantized = quantize(rendered);
    spectra(&quantized)
}

/// What the engine's own output loses against the source, as a percentage of
/// full scale (RMS), through the real bake and the real voice.
fn loss(clip: &[i16], device: u32) -> f32 {
    let wav = {
        let mut bytes = Vec::new();
        // Round-tripped through the file format rather than handed over
        // directly, so what is measured is the path a demo actually takes.
        let path = std::env::temp_dir().join("gg-timbre-grade.wav");
        if write_wav(&path, RATE, clip).is_ok()
            && let Ok(read) = std::fs::read(&path)
        {
            bytes = read;
        }
        bytes
    };
    let Ok((rate, decoded)) = ggc::clip::parse(&wav) else {
        return f32::NAN;
    };
    let blob = gg_assets::clip::encode(rate, &decoded);
    let Ok(stored) = gg_assets::Clip::read(&blob) else {
        return f32::NAN;
    };
    let mut bank = Bank::new();
    bank.add(asset_id("graded"), stored.rate(), stored.samples());

    let mut sound = Sound::clip("graded", 1.0);
    // No envelope: the ramp is the engine doing its job and would read as loss.
    sound.attack_ms = 0;
    sound.release_ms = 0;
    let out = render(&sound, Some(bank), device);

    // The source, resampled the way the voice would have to — nearest is enough
    // as a reference here, because what is being measured is the *difference*
    // between two answers to the same question and both see the same input.
    let mut error = 0.0f64;
    let mut count = 0u64;
    for (index, played) in out.iter().enumerate() {
        let source = index as u64 * u64::from(RATE) / u64::from(device);
        let Some(want) = clip.get(source as usize) else {
            break;
        };
        let delta = f64::from(*played) - f64::from(*want) / 32_768.0;
        error += delta * delta;
        count += 1;
    }
    if count == 0 {
        return f32::NAN;
    }
    ((error / count as f64).sqrt() * 100.0) as f32
}

/// Render one `Sound` through the shipped mixer at `rate`.
///
/// The rate is an argument and not a constant because the loss column's whole
/// structure is one measurement taken twice — at the clip's own rate, where the
/// resampler must be a copy, and at a device's, where it must not be much worse
/// than one.
fn render(sound: &Sound, bank: Option<Bank>, rate: u32) -> Vec<f32> {
    let mut mixer = Mixer::new(rate);
    if let Some(bank) = bank {
        mixer.set_bank(std::sync::Arc::new(bank));
    }
    mixer.fire(&Trigger {
        slot: 0,
        sound: *sound,
    });
    // Mixed until the bank goes quiet rather than into a buffer sized by a
    // guess: trailing silence flatters every RMS and drags every spectrum
    // toward zero, and a buffer too short would grade a clip's first half.
    // The cap is for a loop, which by construction never stops.
    let mut out = Vec::new();
    let mut chunk = vec![0.0; 4_096];
    while mixer.sounding() > 0 && out.len() < rate as usize * 10 {
        mixer.mix(&mut chunk);
        out.extend_from_slice(&chunk);
    }
    let last = out.iter().rposition(|s| *s != 0.0).unwrap_or(0);
    out.truncate(last + 1);
    out
}

/// A rendered `f32` buffer in the `i16` domain the analysis works in — so a
/// candidate and a clip are measured by identical code rather than by two
/// spellings of it.
fn quantize(rendered: &[f32]) -> Vec<i16> {
    rendered
        .iter()
        .map(|s| (s * 32_767.0).clamp(-32_768.0, 32_767.0) as i16)
        .collect()
}

/// How many spectral peaks the clip has above a twentieth of its loudest — the
/// count that says *why* no tone fits, since every wave in the protocol has one
/// fundamental and harmonics above it.
fn partials(clip: &[i16]) -> usize {
    let magnitudes = spectrum(clip);
    let peak = magnitudes.iter().fold(0.0f32, |m, s| m.max(*s));
    magnitudes
        .windows(3)
        .filter(|w| w[1] > w[0] && w[1] > w[2] && w[1] > peak / 20.0)
        .count()
}
