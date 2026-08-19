//! Where the mixer clicks (§6 M77).
//!
//! A click is a **discontinuity** — a step between consecutive samples that the
//! signal itself would never have made — so grading one needs no reference
//! implementation and no ear: render the shipped [`Mixer`] a sample at a time,
//! do the thing, and measure the first difference across the sample the thing
//! landed on.
//!
//! # Why the number is a ratio
//!
//! A step alone says nothing. A square wave swings full scale every half period,
//! so an extra 0.2 inside one is inaudible; the same 0.2 in the middle of a
//! sustained chord is the entire complaint. Every row is therefore the step at
//! the event over **the largest step the same signal makes on its own**, away
//! from it. A ratio near 1 is an event the signal could have made itself and
//! nobody can hear; a ratio of twenty is a click.
//!
//! That framing is also what keeps this honest about the rows it *cannot*
//! condemn: a one-shot's start and end read high ratios against a signal that
//! is silent on one side of them, so they are reported against the note's own
//! peak instead — an envelope that reaches exactly zero is the claim, and it is
//! the one `synth.rs` already makes in its own tests.
//!
//! # What it renders
//!
//! Two subjects, because they fail differently. A synthesized **sine** at a
//! known frequency, whose own largest step is arithmetic (`2*pi*f/rate` at the
//! zero crossing) and so says whether the instrument itself is measuring what
//! it thinks. And demo 10's **actual theme**, read from the crate's own `.wav`
//! through `ggc::clip::parse` and the real `Bank`, because that is the music
//! that actually stops when a player pauses.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use gg_audio::{Bank, Mixer, Trigger};
use gg_ecs::boundary::{Sound, asset_id, wave};
use gg_math::sim;

/// The device rate every row is rendered at. 48 kHz because that is what the
/// desk's own device runs and what the resampler therefore actually does.
const RATE: u32 = 48_000;
/// The synthetic subject's frequency — low enough that its own step is small
/// (so a click stands out), high enough to be a tone and not a drift.
const HZ: f32 = 220.0;
/// Samples rendered before an event, and after it. A tenth of a second either
/// side: long enough that the "own step" column has a signal to measure and
/// short enough that seven rows are instant.
const RUN: usize = RATE as usize / 10;
/// How far into a loop [`loudest`] looks. Four seconds, which is longer than
/// the synthetic cycle and most of demo 10's theme.
const SEARCH: usize = RATE as usize * 4;
/// How many samples after the event the step is looked for in. The trigger
/// lands on the sample the mixer next produces, and a fade lands over many —
/// this is only the window the *onset* has to appear in.
const SEAM: usize = 4;

/// Milliseconds as samples at [`RATE`].
fn samples_of(ms: u32) -> usize {
    (RATE as usize * ms as usize) / 1000
}

/// A rendered event: the samples, and where the thing happened.
struct Trace {
    samples: Vec<f32>,
    at: usize,
}

impl Trace {
    /// The largest step across the seam.
    fn step(&self) -> f32 {
        self.window(self.at, self.at + SEAM)
    }

    /// The largest step the signal makes away from the seam — its own.
    ///
    /// Both sides, so a subject that is silent before the event still has the
    /// half after it to be measured against.
    fn own(&self) -> f32 {
        let before = self.window(1, self.at.saturating_sub(1));
        let after = self.window(self.at + SEAM + 1, self.samples.len());
        before.max(after)
    }

    /// The loudest sample anywhere — what a start and an end are judged
    /// against, since a ratio wants a denominator the signal actually has.
    fn peak(&self) -> f32 {
        self.samples.iter().fold(0.0f32, |m, s| m.max(s.abs()))
    }

    fn window(&self, from: usize, to: usize) -> f32 {
        let to = to.min(self.samples.len());
        let mut worst = 0.0f32;
        for index in from.max(1)..to {
            worst = worst.max((self.samples[index] - self.samples[index - 1]).abs());
        }
        worst
    }
}

/// Render `before` samples, do `act`, render `after` more.
///
/// One sample per `mix` call, which is not how a device drives it and is
/// exactly the point: a buffer boundary would hide which sample an event landed
/// on, and that sample is the measurement.
fn trace(mixer: &mut Mixer, before: usize, after: usize, act: impl FnOnce(&mut Mixer)) -> Trace {
    let mut samples = Vec::with_capacity(before + after);
    let mut one = [0.0f32; 1];
    for _ in 0..before {
        mixer.mix(&mut one);
        samples.push(one[0]);
    }
    act(mixer);
    let at = samples.len();
    for _ in 0..after {
        mixer.mix(&mut one);
        samples.push(one[0]);
    }
    Trace { samples, at }
}

/// A mixer holding `bank`, with `sound` already looping in slot 0.
fn sounding(bank: &Arc<Bank>, sound: &Sound) -> Mixer {
    let mut mixer = Mixer::new(RATE);
    mixer.set_bank(Arc::clone(bank));
    mixer.fire(&Trigger {
        slot: 0,
        sound: *sound,
    });
    mixer
}

/// How many samples in the loop is at its loudest, within [`SEARCH`].
///
/// Every event below is fired *there* rather than at a round number of
/// milliseconds, and that is the whole difference between a measurement and a
/// number. The first version warmed up for a tenth of a second and fired into
/// whatever it found: demo 10's theme is a chord progression that opens
/// quietly, so every row read a step of 0.00096 — four different events
/// agreeing to five decimals, which is the signature of an instrument
/// measuring silence. Cutting a signal that is already near zero is not a
/// click, and it is not the case a player produces either: a pause lands
/// wherever the music happens to be, and what a click measurement wants is
/// the worst of those.
fn loudest(bank: &Arc<Bank>, sound: &Sound) -> usize {
    let mut mixer = sounding(bank, sound);
    let mut out = vec![0.0; SEARCH];
    mixer.mix(&mut out);
    let mut at = 0;
    let mut peak = 0.0f32;
    for (index, sample) in out.iter().enumerate() {
        if sample.abs() > peak {
            peak = sample.abs();
            at = index;
        }
    }
    at
}

/// Demo 10's theme, through the bake a pack would give it.
fn theme() -> Result<(Arc<Bank>, Sound)> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .context("the workspace root")?
        .join("demos/10-tetris/assets/theme.wav");
    let bytes = std::fs::read(&path)
        .with_context(|| format!("reading {} — the demo's own music", path.display()))?;
    let (rate, decoded) = ggc::clip::parse(&bytes)
        .map_err(|e| anyhow::anyhow!("{} is not a clip this engine bakes: {e}", path.display()))?;
    let blob = gg_assets::clip::encode(rate, &decoded);
    let stored = gg_assets::Clip::read(&blob)
        .map_err(|e| anyhow::anyhow!("the baked clip does not read back: {e}"))?;
    let mut bank = Bank::new();
    bank.add(asset_id("theme"), stored.rate(), stored.samples());
    // The demo's own call, gain and all (`demos/10-tetris/src/lib.rs`).
    Ok((Arc::new(bank), Sound::music("theme", 0.22)))
}

/// A short smooth clip whose own step is arithmetic: one full cycle of a sine,
/// so the loop seam is also a *phase-continuous* join and any step there is the
/// resampler's rather than the material's.
fn cycle() -> (Arc<Bank>, Sound) {
    let frames = (RATE as f32 / HZ) as usize;
    let mut samples = Vec::with_capacity(frames);
    for index in 0..frames {
        let phase = index as f32 / frames as f32;
        samples.push((sim::sin(phase * core::f32::consts::TAU) * 24_000.0) as i16);
    }
    let mut bank = Bank::new();
    bank.add(asset_id("cycle"), RATE, &samples);
    (Arc::new(bank), Sound::music("cycle", 0.8))
}

/// One row of the report.
struct Row {
    what: &'static str,
    step: f32,
    own: f32,
    /// What the ratio is taken against — the signal's own step, or its peak for
    /// the two rows that have silence on one side.
    against: f32,
    note: &'static str,
}

impl Row {
    fn ratio(&self) -> f32 {
        match self.against {
            0.0 => f32::INFINITY,
            against => self.step / against,
        }
    }
}

/// See the module header.
///
/// # Errors
///
/// If demo 10's `theme.wav` cannot be read or does not bake.
pub fn run(_args: &[String]) -> Result<()> {
    let (theme_bank, theme_sound) = theme()?;
    let (cycle_bank, cycle_sound) = cycle();
    let mut rows = Vec::new();

    // --- the two the envelope already owns -----------------------------------
    let mut mixer = Mixer::new(RATE);
    let mut tone = Sound::tone(wave::SINE, HZ, 200, 0.8);
    tone.attack_ms = 4;
    tone.release_ms = 4;
    let start = trace(&mut mixer, RUN, RUN, |m| {
        m.fire(&Trigger {
            slot: 0,
            sound: tone,
        });
    });
    rows.push(Row {
        what: "one-shot starts",
        step: start.step(),
        own: start.own(),
        against: start.peak(),
        note: "the attack's own first step",
    });

    // Played out rather than interrupted: the end is the envelope's, and the
    // event is the sample the note stops producing.
    let mut mixer = Mixer::new(RATE);
    mixer.fire(&Trigger {
        slot: 0,
        sound: tone,
    });
    let hold = (RATE as usize * 200) / 1000;
    let end = trace(&mut mixer, hold, RUN, |_| {});
    rows.push(Row {
        what: "one-shot ends",
        step: end.step(),
        own: end.own(),
        against: end.peak(),
        note: "the release reaching zero",
    });

    // --- the loop's own seam --------------------------------------------------
    let mut mixer = sounding(&cycle_bank, &cycle_sound);
    let seam = trace(&mut mixer, RUN, RUN, |_| {});
    rows.push(Row {
        what: "loop wraps",
        step: seam.step().max(seam.own()),
        own: seam.own(),
        against: seam.own(),
        note: "measured over the whole run, since a wrap is every cycle",
    });

    // --- the four a mixer does to a loop --------------------------------------
    let mut silent = theme_sound;
    silent.wave = wave::SILENT;
    let loud = loudest(&theme_bank, &theme_sound);
    let mut mixer = sounding(&theme_bank, &theme_sound);
    let stop = trace(&mut mixer, loud, RUN, |m| {
        m.fire(&Trigger {
            slot: 0,
            sound: silent,
        });
    });
    rows.push(Row {
        what: "loop is stopped",
        step: stop.step(),
        own: stop.own(),
        against: stop.own(),
        note: "a player pausing demo 10",
    });

    // A player who pauses and changes their mind inside the fade — the one case
    // the fade itself creates, since before §6 M77 there was no window to land
    // in.
    let mut mixer = sounding(&theme_bank, &theme_sound);
    let mut warm = vec![0.0; loud];
    mixer.mix(&mut warm);
    mixer.fire(&Trigger {
        slot: 0,
        sound: silent,
    });
    let half = samples_of(gg_ecs::boundary::FADE_MS) / 2;
    let resume = trace(&mut mixer, half, RUN, |m| {
        m.fire(&Trigger {
            slot: 0,
            sound: theme_sound,
        });
    });
    rows.push(Row {
        what: "loop resumes mid-fade",
        step: resume.step(),
        own: resume.own(),
        against: resume.own(),
        note: "a pause taken back before the music has gone",
    });

    let mut mixer = sounding(&theme_bank, &theme_sound);
    let cue = Sound::tone(wave::SQUARE, 330.0, 22, 0.16);
    let steal = trace(&mut mixer, loud, RUN, |m| {
        m.fire(&Trigger {
            slot: 0,
            sound: cue,
        });
    });
    rows.push(Row {
        what: "slot is taken",
        step: steal.step(),
        own: steal.own(),
        against: steal.own(),
        note: "a cue whose entity index collides with the music's",
    });

    let mut mixer = sounding(&theme_bank, &theme_sound);
    let swap = trace(&mut mixer, loud, RUN, |m| {
        m.set_bank(Arc::clone(&cycle_bank));
    });
    rows.push(Row {
        what: "bank is swapped",
        step: swap.step(),
        own: swap.own(),
        against: swap.own(),
        note: "a pack rebuilt under a running game",
    });

    let mut mixer = sounding(&theme_bank, &theme_sound);
    let volume = trace(&mut mixer, loud, RUN, |m| m.set_master(0.875));
    rows.push(Row {
        what: "volume moves one notch",
        step: volume.step(),
        own: volume.own(),
        against: volume.own(),
        note: "demo 12's QUIET_STEP, an eighth of the range",
    });

    let mut mixer = sounding(&theme_bank, &theme_sound);
    let hush = trace(&mut mixer, loud, RUN, |m| m.set_master(0.0));
    rows.push(Row {
        what: "session is suspended",
        step: hush.step(),
        own: hush.own(),
        against: hush.own(),
        note: "`Audio::hush` on a lost focus (§6 M49)",
    });

    println!("gg-tools clicks: the mixer's discontinuities at {RATE} Hz (§6 M77)");
    println!();
    println!(
        "  {:<26}  {:>9}  {:>9}  {:>8}  what it is",
        "event", "step", "own", "ratio"
    );
    for row in &rows {
        println!(
            "  {:<26}  {:>9.5}  {:>9.5}  {:>8.2}  {}",
            row.what,
            row.step,
            row.own,
            row.ratio(),
            row.note
        );
    }
    println!();
    println!(
        "  A ratio near 1 is a step the signal could have made itself. The two\n  \
         `one-shot` rows are taken against the note's peak rather than its own\n  \
         step, because silence on one side leaves nothing else to divide by."
    );
    Ok(())
}
