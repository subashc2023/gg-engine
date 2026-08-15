//! The only part of the engine that talks to a sound card, and the whole of
//! `cpal`'s blast radius (§3). Everything with an opinion about how a cue sounds
//! is in [`synth`](super::synth), which is why the interesting claims are
//! testable on a machine that never opens this file's device.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use tracing::{debug, warn};

use crate::bank::Bank;
use crate::synth::{Mixer, Trigger};

/// Somewhere to send notes. One real implementation and one fake, which is the
/// whole reason it exists (§6 M50).
///
/// The supervisor above it — notice a dead endpoint, wait, open another, put the
/// bank back, restart the music — is *policy*, and policy that can only be
/// exercised by unplugging a physical cable is policy nothing gates. §1.5's
/// audio law forbids an automated tier from opening a device at all, so the seam
/// goes here: [`Device`] is the only thing on the far side of it, and everything
/// worth asserting about a reconnect is asserted against a fake on a machine
/// with no sound card.
pub(crate) trait Sink {
    /// Hand over this tick's triggers and the current master volume.
    fn send(&self, triggers: &[Trigger], master: f32) -> usize;
    /// Replace the clips the mixer reads, cutting every sounding voice.
    fn install(&self, bank: Arc<Bank>);
    fn rate(&self) -> u32;
    fn channels(&self) -> u16;
    /// What the host calls this endpoint, for the line that says which one the
    /// player took away.
    fn name(&self) -> &str;
    /// The endpoint is gone and nothing sent here will ever sound again.
    fn lost(&self) -> bool;
    /// The player's default output is no longer the one this sink opened —
    /// headphones plugged in while the game played to the speakers.
    fn superseded(&self) -> bool;
}

/// What the sim thread leaves for the callback to pick up.
///
/// One structure behind one lock rather than three channels: the callback takes
/// the lock once per buffer and everything it might need is behind it, so
/// adding the bank and the volume cost no second chance to lose the race.
#[derive(Default)]
struct Pending {
    triggers: Vec<Trigger>,
    /// Set when the shell installs a pack (§6 M43); taken by the callback.
    bank: Option<Arc<Bank>>,
    /// The player's master volume, read off the world every tick.
    master: f32,
}

/// Triggers the queue holds between one tick and the callback that drains it.
///
/// A tick produces at most one per `Sound` entity, and the callback runs every
/// few milliseconds, so the queue is empty almost always. The cap exists so a
/// stalled device cannot grow it without bound — dropping the *oldest* is wrong
/// (the game's most recent intent is what it meant), so a full queue drops the
/// arriving trigger and says so once.
const QUEUE: usize = 256;

/// Stands in for a device whose name the host will not give up. Compared like
/// any other name, so two unnamed defaults read as the same device rather than
/// as a change — which is what stops a nameless host churning the stream every
/// poll (§6 M50).
const UNNAMED: &str = "<unnamed>";

/// What went wrong reaching a sound card. Never fatal to a run: a machine with
/// no output device plays a game silently, exactly as `--frames` renders one
/// without a window.
#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    #[error("no default audio output device")]
    NoDevice,
    #[error("the device reports no usable output configuration: {0}")]
    NoConfig(#[from] cpal::DefaultStreamConfigError),
    #[error("the device's sample format {0} is not one this build writes")]
    Format(cpal::SampleFormat),
    #[error("building the output stream failed: {0}")]
    Build(#[from] cpal::BuildStreamError),
    #[error("starting the output stream failed: {0}")]
    Play(#[from] cpal::PlayStreamError),
}

/// An open output stream and the queue feeding it.
///
/// `cpal::Stream` stops the device when it drops, so this type *is* the
/// lifetime of the sound — there is nothing to shut down explicitly and no way
/// to leak a running device past the shell that owns it.
pub struct Device {
    queue: Arc<Mutex<Pending>>,
    rate: u32,
    channels: u16,
    /// Set by the error callback when the backend says the endpoint is gone,
    /// read by the supervisor on the sim thread (§6 M50). An atomic rather than
    /// a channel because the writer is the audio thread and the only thing it
    /// has to say is one bit.
    lost: Arc<AtomicBool>,
    /// The device this stream opened, as the host names it. Compared against the
    /// current default to notice a player plugging headphones in — a change no
    /// error callback reports, because nothing went wrong.
    name: String,
    /// Held for its `Drop`. Never read, and `cpal::Stream` is not `Send`, which
    /// is what keeps the whole of `Audio` on the thread that opened it.
    _stream: cpal::Stream,
}

impl Device {
    /// Open the default output device and start it.
    ///
    /// The caller is responsible for §1.5 — [`Audio::device`](crate::Audio) is
    /// where the headless law is enforced, and this function is private to the
    /// crate so there is no way around it.
    pub(crate) fn open() -> Result<Device, DeviceError> {
        let host = cpal::default_host();
        let device = host.default_output_device().ok_or(DeviceError::NoDevice)?;
        let supported = device.default_output_config()?;
        let format = supported.sample_format();
        let config: cpal::StreamConfig = supported.into();
        let rate = config.sample_rate.0;
        let channels = config.channels;
        let name = device.name().unwrap_or_else(|_| UNNAMED.into());
        let queue = Arc::new(Mutex::new(Pending {
            triggers: Vec::with_capacity(QUEUE),
            bank: None,
            master: 1.0,
        }));
        let lost = Arc::new(AtomicBool::new(false));

        // Every format `cpal` names as a device default in practice. An
        // unhandled one is an error rather than a silent cast: writing f32
        // samples into an i16 buffer produces full-scale noise, which is the
        // worst possible failure mode for the subsystem that owns the speakers.
        let stream = match format {
            cpal::SampleFormat::F32 => build::<f32>(&device, &config, &queue, &lost),
            cpal::SampleFormat::I16 => build::<i16>(&device, &config, &queue, &lost),
            cpal::SampleFormat::U16 => build::<u16>(&device, &config, &queue, &lost),
            other => return Err(DeviceError::Format(other)),
        }?;
        stream.play()?;
        debug!(device = name, rate, channels, ?format, "audio device open");
        Ok(Device {
            queue,
            rate,
            channels,
            lost,
            name,
            _stream: stream,
        })
    }
}

impl Sink for Device {
    /// Blocks on the queue's lock, which the callback holds only for the length
    /// of a `Vec` drain — the *callback* is the side that must never wait, and
    /// it does not.
    fn send(&self, triggers: &[Trigger], master: f32) -> usize {
        let Ok(mut queue) = self.queue.lock() else {
            // A poisoned lock means the audio callback panicked. The stream is
            // already dead; saying so every tick would be sixty lines a second.
            return 0;
        };
        queue.master = master;
        let room = QUEUE.saturating_sub(queue.triggers.len());
        let taken = triggers.len().min(room);
        if taken < triggers.len() {
            warn!(
                dropped = triggers.len() - taken,
                "audio trigger queue full — the device is not draining"
            );
        }
        queue.triggers.extend_from_slice(&triggers[..taken]);
        taken
    }

    /// Replaces whatever the callback holds, cutting every sounding voice — see
    /// [`Mixer::set_bank`](crate::Mixer::set_bank). The supervisor puts the
    /// loops back afterwards (§6 M50).
    fn install(&self, bank: Arc<Bank>) {
        if let Ok(mut queue) = self.queue.lock() {
            queue.bank = Some(bank);
        }
    }

    /// Samples per second the device actually runs at.
    fn rate(&self) -> u32 {
        self.rate
    }

    /// Output channels. Mono is written to all of them (§4.2.2's `Sound` carries
    /// no pan, because there is no listener pose to pan against yet).
    fn channels(&self) -> u16 {
        self.channels
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn lost(&self) -> bool {
        self.lost.load(Ordering::Relaxed)
    }

    /// Two property reads against the host, and the reason the supervisor's poll
    /// is measured in seconds rather than ticks.
    ///
    /// Conservative in both directions a host can be vague: a default it cannot
    /// name, or none at all, is *not* a supersession — the first would churn the
    /// device on every poll and the second is [`lost`](Sink::lost)'s to report.
    ///
    /// This is a WASAPI property in practice. ALSA and PulseAudio name their
    /// default `"default"` whatever it is pointing at, so the name never moves
    /// and this never fires — correctly, because on those hosts the *server*
    /// followed the change and the stream is already on the new endpoint.
    fn superseded(&self) -> bool {
        let host = cpal::default_host();
        let Some(current) = host.default_output_device() else {
            return false;
        };
        current.name().unwrap_or_else(|_| UNNAMED.into()) != self.name
    }
}

/// Build the output stream for one sample type. The mixer moves into the
/// callback and is reached from nowhere else, so the audio thread owns its own
/// state outright and the only shared thing is the trigger queue.
fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    queue: &Arc<Mutex<Pending>>,
    lost: &Arc<AtomicBool>,
) -> Result<cpal::Stream, DeviceError>
where
    T: SizedSample + FromSample<f32>,
{
    let mut mixer = Mixer::new(config.sample_rate.0);
    let queue = Arc::clone(queue);
    let lost = Arc::clone(lost);
    let channels = config.channels.max(1) as usize;
    // Grown once on the first callback and reused: allocating inside an audio
    // callback is how a run picks up dropouts under memory pressure.
    let mut mono: Vec<f32> = Vec::new();
    let stream = device.build_output_stream(
        config,
        move |out: &mut [T], _: &cpal::OutputCallbackInfo| {
            // `try_lock`: the audio thread has a deadline and the sim thread
            // does not. Losing the race costs one buffer of latency — about
            // 10 ms, below what anyone hears as late — where blocking costs an
            // audible dropout.
            if let Ok(mut pending) = queue.try_lock() {
                // The bank first: a trigger arriving in the same batch as the
                // pack it names must resolve against the new clips, not miss by
                // one buffer and fall silent.
                if let Some(bank) = pending.bank.take() {
                    mixer.set_bank(bank);
                }
                mixer.set_master(pending.master);
                for trigger in pending.triggers.drain(..) {
                    mixer.fire(&trigger);
                }
            }
            let frames = out.len() / channels;
            mono.resize(frames, 0.0);
            mixer.mix(&mut mono);
            for (frame, sample) in out.chunks_mut(channels).zip(&mono) {
                let value = T::from_sample(*sample);
                frame.fill(value);
            }
        },
        // `DeviceNotAvailable` is the endpoint going away — a cable pulled, a
        // headset disconnecting, a driver reinstalled — and the only thing this
        // thread does about it is set the bit. Logging and reopening are the
        // supervisor's, on the sim thread, because a `tracing` macro allocates
        // and takes a lock and this callback may be running on the driver's.
        //
        // Anything else is left alone deliberately: a backend hiccup that a
        // stream survives is not worth restarting the player's music for.
        move |error| match error {
            cpal::StreamError::DeviceNotAvailable => lost.store(true, Ordering::Relaxed),
            other => warn!(error = %other, "audio output stream error"),
        },
        None,
    )?;
    Ok(stream)
}
