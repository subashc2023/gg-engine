//! The only part of the engine that talks to a sound card, and the whole of
//! `cpal`'s blast radius (§3). Everything with an opinion about how a cue sounds
//! is in [`synth`](super::synth), which is why the interesting claims are
//! testable on a machine that never opens this file's device.

use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};
use tracing::{debug, warn};

use crate::synth::{Mixer, Trigger};

/// Triggers the queue holds between one tick and the callback that drains it.
///
/// A tick produces at most one per `Sound` entity, and the callback runs every
/// few milliseconds, so the queue is empty almost always. The cap exists so a
/// stalled device cannot grow it without bound — dropping the *oldest* is wrong
/// (the game's most recent intent is what it meant), so a full queue drops the
/// arriving trigger and says so once.
const QUEUE: usize = 256;

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
    queue: Arc<Mutex<Vec<Trigger>>>,
    rate: u32,
    channels: u16,
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
        let queue = Arc::new(Mutex::new(Vec::with_capacity(QUEUE)));

        // Every format `cpal` names as a device default in practice. An
        // unhandled one is an error rather than a silent cast: writing f32
        // samples into an i16 buffer produces full-scale noise, which is the
        // worst possible failure mode for the subsystem that owns the speakers.
        let stream = match format {
            cpal::SampleFormat::F32 => build::<f32>(&device, &config, &queue, rate, channels),
            cpal::SampleFormat::I16 => build::<i16>(&device, &config, &queue, rate, channels),
            cpal::SampleFormat::U16 => build::<u16>(&device, &config, &queue, rate, channels),
            other => return Err(DeviceError::Format(other)),
        }?;
        stream.play()?;
        debug!(
            device = device.name().unwrap_or_else(|_| "<unnamed>".into()),
            rate,
            channels,
            ?format,
            "audio device open"
        );
        Ok(Device {
            queue,
            rate,
            channels,
            _stream: stream,
        })
    }

    /// Hand `triggers` to the audio thread. Returns how many were taken.
    ///
    /// Blocks on the queue's lock, which the callback holds only for the length
    /// of a `Vec` drain — the *callback* is the side that must never wait, and
    /// it does not.
    pub(crate) fn send(&self, triggers: &[Trigger]) -> usize {
        let Ok(mut queue) = self.queue.lock() else {
            // A poisoned lock means the audio callback panicked. The stream is
            // already dead; saying so every tick would be sixty lines a second.
            return 0;
        };
        let room = QUEUE.saturating_sub(queue.len());
        let taken = triggers.len().min(room);
        if taken < triggers.len() {
            warn!(
                dropped = triggers.len() - taken,
                "audio trigger queue full — the device is not draining"
            );
        }
        queue.extend_from_slice(&triggers[..taken]);
        taken
    }

    /// Samples per second the device actually runs at.
    pub(crate) fn rate(&self) -> u32 {
        self.rate
    }

    /// Output channels. Mono is written to all of them (§4.2.2's `Sound` carries
    /// no pan, because there is no listener pose to pan against yet).
    pub(crate) fn channels(&self) -> u16 {
        self.channels
    }
}

/// Build the output stream for one sample type. The mixer moves into the
/// callback and is reached from nowhere else, so the audio thread owns its own
/// state outright and the only shared thing is the trigger queue.
fn build<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    queue: &Arc<Mutex<Vec<Trigger>>>,
    rate: u32,
    channels: u16,
) -> Result<cpal::Stream, DeviceError>
where
    T: SizedSample + FromSample<f32>,
{
    let mut mixer = Mixer::new(rate);
    let queue = Arc::clone(queue);
    let channels = channels.max(1) as usize;
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
                for trigger in pending.drain(..) {
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
        // A device that errors mid-run is logged and left alone: the stream is
        // gone, the game is not, and a panic here would be on the audio thread.
        |error| warn!(%error, "audio output stream error"),
        None,
    )?;
    Ok(stream)
}
