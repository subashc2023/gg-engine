//! The legs that open a real sound card, and therefore the legs no automated
//! tier may run (§1.5's audio analogue, §6 M18 item 2).
//!
//! `#[ignore]` with a reason naming `cargo xtask interactive`, exactly as every
//! window-creating test is marked — the dev machine is also the user's, and a
//! nightly that made a noise at 02:00 is the same violation as one that put a
//! window on the screen. `Audio::device`'s own panic under `GG_HEADLESS=1` is
//! the second lock and is proven in the crate's unit tests, where it costs
//! nothing to run.
//!
//! Everything that can be checked without a device is checked without one: the
//! mixer's tests are pure arithmetic and the observer's run against a `World`.
//! What is left here is the part only a driver can answer — that the format we
//! negotiated is one we can write, and that a note reaches the endpoint.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_audio::Audio;
use gg_ecs::World;
use gg_ecs::boundary::{Sound, wave};

/// Open the default device, play one note, and let it finish.
///
/// The assertion a silent machine cannot make: that `cpal` accepted the config,
/// that the sample format is one `device.rs` writes, and that the callback ran
/// without the stream erroring. What it *cannot* assert is that a human heard
/// anything — which is why this is `interactive` and not a gate.
#[test]
#[ignore = "opens an audio device (§1.5) — run via `cargo xtask interactive`"]
fn a_real_device_takes_a_note() {
    let mut audio = Audio::device().expect("no audio device on this machine");
    assert!(audio.audible());
    let (rate, channels) = audio.format().expect("an open device has a format");
    assert!(rate >= 8_000, "implausible sample rate {rate}");
    assert!(channels >= 1);

    let mut world = World::new();
    world.register::<Sound>().unwrap();
    let entity = world.spawn();
    let mut note = Sound::tone(wave::SINE, 440.0, 250, 0.4);
    world.insert(entity, note).unwrap();
    audio.tick(&world); // first sight registers

    note.play();
    world.insert(entity, note).unwrap();
    audio.tick(&world);
    assert_eq!(audio.fired().len(), 1);

    // Long enough for the callback to have drained the queue and played it out.
    std::thread::sleep(std::time::Duration::from_millis(400));
}

/// The half M43 built and left uncovered: a bank installed on a real device,
/// and a voice reading a slice out of it through the resampler.
///
/// The clip is a **chord**, deliberately. `gg-tools timbre` grades the protocol
/// by how near the closest `Sound` gets to a clip, and its two worst rows are
/// its two chords — one wave at one frequency cannot be three pitches. So an
/// operator who hears a chord has heard the thing a synthesizer provably could
/// not have made, which is the only way an ear can read this leg at all. Its
/// rate is 22 050 on purpose: equal to no device's, so the 32.32 cursor steps
/// by something other than one and the interpolator is in the path.
#[test]
#[ignore = "opens an audio device (§1.5) — run via `cargo xtask interactive`"]
fn a_real_device_takes_a_clip_a_synthesizer_could_not_have_made() {
    const RATE: u32 = 22_050;
    let mut audio = Audio::device().expect("no audio device on this machine");
    assert!(audio.audible());

    let mut world = World::new();
    world.register::<Sound>().unwrap();
    let entity = world.spawn();
    // A major triad, which is what puts it out of `Sound`'s reach.
    let mut chord = Sound::clip("chord", 0.4);
    let mut samples = Vec::new();
    for frame in 0..RATE {
        let t = f64::from(frame) / f64::from(RATE);
        let fade = 1.0 - t; // linear, so the tail does not click
        let voices: f64 = [261.63_f64, 329.63, 392.00]
            .iter()
            .map(|hz| gg_math::sim::sin(t * hz * core::f64::consts::TAU))
            .sum();
        samples.push((voices / 3.0 * fade * 9_000.0) as i16);
    }
    let mut bank = gg_audio::Bank::new();
    bank.add(chord.clip, RATE, &samples);
    assert_eq!(bank.clips(), 1);
    audio.install(bank);
    assert_eq!(
        audio.bank().map(gg_audio::Bank::frames),
        Some(RATE as usize)
    );

    world.insert(entity, chord).unwrap();
    audio.tick(&world); // first sight registers
    chord.play();
    world.insert(entity, chord).unwrap();
    audio.tick(&world);
    assert_eq!(audio.fired().len(), 1);
    std::thread::sleep(std::time::Duration::from_millis(1_200));

    // And the loop, which is the other voice M43 added: it outlives its own
    // length, and silence is how a game stops it rather than a stop call.
    let mut theme = Sound::music("chord", 0.25);
    theme.play();
    world.insert(entity, theme).unwrap();
    audio.tick(&world);
    std::thread::sleep(std::time::Duration::from_millis(1_500));
    let mut hush = Sound::tone(wave::SILENT, 0.0, 0, 0.0);
    hush.play();
    world.insert(entity, hush).unwrap();
    audio.tick(&world);
    std::thread::sleep(std::time::Duration::from_millis(300));
}

/// Dropping the `Audio` stops the device. Nothing observes this from outside the
/// process, so the claim is that it does not hang or panic — a `cpal::Stream`
/// whose drop deadlocked would take the shell's shutdown with it.
#[test]
#[ignore = "opens an audio device (§1.5) — run via `cargo xtask interactive`"]
fn a_device_closes_when_the_observer_drops() {
    for _ in 0..3 {
        let audio = Audio::device().expect("no audio device on this machine");
        assert!(audio.audible());
        drop(audio);
    }
}
