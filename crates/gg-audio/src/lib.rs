//! The host half of §4.2.2's audio protocol: [`Sound`] components in, samples
//! out (§6 M18 item 2).
//!
//! `gg-ui` is the shape this follows exactly — a game declares components
//! because §3's deny pin says it may not link the crate that consumes them, and
//! the host turns them into something a device understands. The one difference
//! is the direction of the write, and it is load-bearing: the UI writes
//! [`Widget::state`](gg_ecs::boundary::Widget) back into the world, and **audio
//! writes nothing**. See `gg_ecs::boundary::audio` for why a host that consumed
//! a trigger would make §5.6c fail on the speaker instead of on the sim.
//!
//! # §1.5's audio law
//!
//! The dev machine is the user's gaming PC, and a tier that made a noise at
//! 02:00 would be the same violation as one that opened a window. The law is
//! the same shape and is enforced the same way:
//!
//! - **No automated tier opens an audio device at all.** Not "opens one and
//!   plays nothing" — a device open is a mixer running and a driver holding an
//!   endpoint, and "it happened to be quiet" is not a property CI can rest on.
//! - [`Audio::device`] **panics** under `GG_HEADLESS=1`, exactly as
//!   `gg-platform` panics on a visible window there.
//! - Nothing else in the tree may name it — not the shell, which takes
//!   [`Audio::device_unless_headless`]. `xtask ci`'s grep gate holds the caller
//!   list to this crate, on the same machinery that keeps raw Vulkan tokens
//!   inside `gg-rhi`.
//! - Every test that opens one is `#[ignore]` with a reason naming
//!   `cargo xtask interactive`.
//!
//! [`Audio::silent`] is what everything else gets, and it is not a stub: it runs
//! the whole observer, so "the line-clear cue fired on the tick the row went
//! away" is an ordinary assertion on a silent machine. What it does not do is
//! open a device.

mod device;
pub mod synth;

use gg_ecs::boundary::Sound;
use gg_ecs::{AliasError, Entity, Query, World};
use tracing::{debug, warn};

pub use device::DeviceError;
pub use synth::{Mixer, Trigger, VOICES};

/// True when this process runs under the headless law (§1.5).
///
/// Read here rather than taken from `gg-platform`: audio sits below windowing
/// and must not link it — a game's sound has nothing to do with a swapchain,
/// and the crate that owns the speakers should not be able to open a window by
/// accident. One env var, read the same way in both places.
#[must_use]
pub fn headless() -> bool {
    std::env::var_os("GG_HEADLESS").is_some()
}

/// Panics if this process may not open an audio device (§1.5).
fn enforce_headless_law() {
    assert!(
        !headless(),
        "GG_HEADLESS=1 forbids opening an audio device (§1.5): the dev machine is also the \
         user's, and an automated tier that made a noise is the same violation as one that put a \
         window on the screen. Use `Audio::silent`, which runs the whole observer and opens \
         nothing, or `cargo xtask interactive` if the point is to hear it."
    );
}

/// The host's audio: an observer over the world and somewhere to send what it
/// finds.
///
/// Not `Send`. `cpal::Stream` is not, and that is the useful constraint rather
/// than an obstacle — the device is opened and dropped on the shell's thread,
/// so a device cannot outlive the loop that owns it.
pub struct Audio {
    query: Query<&'static Sound>,
    /// `(entity bits, last seq)`, sorted by the first — a handful of entries,
    /// looked up by binary search. Not an `IndexMap`: this is host state that
    /// never reaches the hash, and a sorted `Vec` needs no dependency.
    seen: Vec<(u64, u32)>,
    /// Next tick's `seen`, built during the walk and swapped in at the end.
    /// Two buffers so a settled tick allocates nothing.
    next: Vec<(u64, u32)>,
    /// This tick's triggers. Kept after sending so a test can read what a tick
    /// decided to play without a device in the picture.
    fired: Vec<Trigger>,
    out: Option<device::Device>,
}

impl Audio {
    /// An observer that opens nothing. What every automated tier gets, and what
    /// a machine with no sound card falls back to.
    ///
    /// # Errors
    /// If the `Sound` query cannot be built, which means the component's access
    /// conflicts with itself — a `gg-ecs` bug, not a runtime condition.
    pub fn silent() -> Result<Audio, AliasError> {
        Ok(Audio {
            query: Query::new()?,
            seen: Vec::new(),
            next: Vec::new(),
            fired: Vec::new(),
            out: None,
        })
    }

    /// An observer with the default output device open and running.
    ///
    /// # Panics
    /// Under `GG_HEADLESS=1` (§1.5). That is the law, not a diagnostic: a tier
    /// that reached here has already decided to make a noise on someone's desk.
    ///
    /// # Errors
    /// If the query cannot be built, or if there is no usable device — the
    /// caller decides whether that is fatal. [`Audio::device_unless_headless`]
    /// decides it is not.
    pub fn device() -> Result<Audio, AudioError> {
        enforce_headless_law();
        let mut audio = Audio::silent()?;
        audio.out = Some(device::Device::open()?);
        Ok(audio)
    }

    /// The shell's constructor: a device when this process is allowed one and
    /// has one, silence otherwise.
    ///
    /// Never fails. A run with no sound card, or one under the headless law, is
    /// a run — the game is the point and the speakers are not.
    ///
    /// # Errors
    /// Only the query, which cannot fail at runtime.
    pub fn device_unless_headless() -> Result<Audio, AliasError> {
        if headless() {
            debug!("audio: silent — GG_HEADLESS=1 (§1.5)");
            return Audio::silent();
        }
        match Audio::device() {
            Ok(audio) => Ok(audio),
            Err(error) => {
                warn!(%error, "audio: silent — no device");
                Audio::silent()
            }
        }
    }

    /// True when a device is open. The observable §1.5's gate asserts on, and
    /// the reason it is a method rather than a log line to grep.
    #[must_use]
    pub fn audible(&self) -> bool {
        self.out.is_some()
    }

    /// Samples per second, and output channels, of the open device. `None` when
    /// silent.
    #[must_use]
    pub fn format(&self) -> Option<(u32, u16)> {
        self.out.as_ref().map(|d| (d.rate(), d.channels()))
    }

    /// Read every [`Sound`] in `world` and play the ones whose `seq` moved.
    ///
    /// `&World` and not `&mut World` by contract, and it is the contract that
    /// matters: the type signature is the proof that a loud run and a silent one
    /// are the same run.
    ///
    /// Called once per **tick**, beside the UI's frame and not beside the
    /// render — a cue fires on the tick a row cleared, and a shell that ran two
    /// sim ticks in one frame would otherwise lose one of them.
    pub fn tick(&mut self, world: &World) {
        let Audio {
            query,
            seen,
            next,
            fired,
            out,
        } = self;
        fired.clear();
        next.clear();
        world.each_ref(query, |entity: Entity, sound: &Sound| {
            let bits = entity.to_bits();
            next.push((bits, sound.seq));
            // A newly-seen entity is registered, never played. `bootstrap`
            // spawns a game's whole cue bank in one tick, and a save restored at
            // §6 M14 carries whatever `seq` the session that wrote it ended on —
            // playing on first sight would make every load and every reload a
            // burst of noise.
            let Ok(at) = seen.binary_search_by_key(&bits, |(e, _)| *e) else {
                return;
            };
            if seen[at].1 != sound.seq {
                fired.push(Trigger {
                    // The entity's slot index, so one entity is one voice for as
                    // long as it lives. Two entities colliding modulo the bank
                    // cut each other off, which is a bank that is too small
                    // rather than a correctness problem.
                    slot: entity.index() as usize % VOICES,
                    sound: *sound,
                });
            }
        });
        // `each_ref` walks archetypes in creation order, so this is not sorted.
        next.sort_unstable_by_key(|(entity, _)| *entity);
        std::mem::swap(seen, next);
        if let Some(device) = out
            && !fired.is_empty()
        {
            device.send(fired);
        }
    }

    /// What the last [`tick`](Audio::tick) decided to play.
    ///
    /// The whole of a cue's testability: a demo asserts that clearing a row
    /// fired a note of the right shape, on a machine with no sound card, in the
    /// fast tier.
    #[must_use]
    pub fn fired(&self) -> &[Trigger] {
        &self.fired
    }

    /// Forget every entity's sequence, so the next tick registers them again
    /// rather than triggering on the difference.
    ///
    /// What a reload and a load both need (§5.11, §6 M14): the world on the
    /// other side is a world this observer has not seen, and the `seq` values in
    /// it are the previous session's. Without this, resuming a save is a chord.
    pub fn forget(&mut self) {
        self.seen.clear();
    }
}

/// Opening a device, all the way down.
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error(transparent)]
    Query(#[from] AliasError),
    #[error(transparent)]
    Device(#[from] DeviceError),
}

#[cfg(test)]
mod tests {
    // unwrap is permitted in tests (§2, Error handling row).
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use gg_ecs::boundary::wave;

    use super::*;

    fn world_with(sounds: usize) -> (World, Vec<Entity>) {
        let mut world = World::new();
        world.register::<Sound>().unwrap();
        let entities = (0..sounds)
            .map(|_| {
                let entity = world.spawn();
                let note = Sound::tone(wave::SQUARE, 440.0, 40, 0.5);
                world.insert(entity, note).unwrap();
                entity
            })
            .collect();
        (world, entities)
    }

    fn bump(world: &mut World, entity: Entity) {
        let mut sound = *world.get::<Sound>(entity).unwrap();
        sound.play();
        world.insert(entity, sound).unwrap();
    }

    #[test]
    fn a_silent_observer_opens_nothing_and_still_observes() {
        let audio = Audio::silent().unwrap();
        assert!(!audio.audible());
        assert_eq!(audio.format(), None);
    }

    /// The property that keeps a load and a reload quiet: the first tick over a
    /// world registers what is in it, whatever `seq` says.
    #[test]
    fn the_first_sight_of_a_sound_registers_it_rather_than_playing_it() {
        let (mut world, entities) = world_with(3);
        // A seq the session that wrote the save ended on.
        let mut loud = *world.get::<Sound>(entities[0]).unwrap();
        loud.seq = 41;
        world.insert(entities[0], loud).unwrap();

        let mut audio = Audio::silent().unwrap();
        audio.tick(&world);
        assert!(audio.fired().is_empty(), "a first tick played something");

        bump(&mut world, entities[0]);
        audio.tick(&world);
        assert_eq!(audio.fired().len(), 1);
        assert_eq!(audio.fired()[0].sound.seq, 42);
    }

    #[test]
    fn only_the_entities_whose_sequence_moved_are_played() {
        let (mut world, entities) = world_with(4);
        let mut audio = Audio::silent().unwrap();
        audio.tick(&world);

        bump(&mut world, entities[1]);
        bump(&mut world, entities[3]);
        audio.tick(&world);
        assert_eq!(audio.fired().len(), 2);
        let slots: Vec<_> = audio.fired().iter().map(|t| t.slot).collect();
        assert!(slots.contains(&(entities[1].index() as usize % VOICES)));
        assert!(slots.contains(&(entities[3].index() as usize % VOICES)));

        // And a tick that changed nothing plays nothing, however many times it
        // runs — the trigger is the *difference*, not the value.
        audio.tick(&world);
        audio.tick(&world);
        assert!(audio.fired().is_empty());
    }

    /// A sim that never touched a `Sound` cannot be made to sound by the passage
    /// of time — which is the difference between a trigger and a poll.
    #[test]
    fn a_world_that_does_not_change_is_silent_forever() {
        let (world, _) = world_with(2);
        let mut audio = Audio::silent().unwrap();
        for _ in 0..1_000 {
            audio.tick(&world);
            assert!(audio.fired().is_empty());
        }
    }

    #[test]
    fn forgetting_makes_the_next_tick_a_first_sight_again() {
        let (mut world, entities) = world_with(2);
        let mut audio = Audio::silent().unwrap();
        audio.tick(&world);
        bump(&mut world, entities[0]);
        audio.tick(&world);
        assert_eq!(audio.fired().len(), 1);

        audio.forget();
        bump(&mut world, entities[0]);
        audio.tick(&world);
        assert!(
            audio.fired().is_empty(),
            "a reload replayed the game's last note"
        );
    }

    /// A despawned entity leaves the table on its own — the walk rebuilds it,
    /// so there is no cleanup path to forget to call and no unbounded growth in
    /// a game that spawns cues per event.
    #[test]
    fn a_despawned_sound_leaves_no_trace() {
        let (mut world, entities) = world_with(3);
        let mut audio = Audio::silent().unwrap();
        audio.tick(&world);
        assert_eq!(audio.seen.len(), 3);
        world.despawn(entities[1]);
        audio.tick(&world);
        assert_eq!(audio.seen.len(), 2);
    }

    /// §1.5, as the machine rather than as the paragraph. The panic is what
    /// makes the law fail loudly instead of making a noise quietly; a
    /// `should_panic` here goes red the moment the enforcement is removed, which
    /// is the "and a gate proves it can fail" half of §6 M18's exit row.
    ///
    /// Safe under `nextest`, which runs each test in its own process — and
    /// under `cargo test`, because the variable is set for the whole binary and
    /// no test in it opens a device.
    #[test]
    #[should_panic(expected = "GG_HEADLESS=1 forbids opening an audio device")]
    fn a_device_under_the_headless_law_is_refused_by_name() {
        // SAFETY: `set_var` is unsound only against a concurrent reader in
        // another thread. This test process reads `GG_HEADLESS` here and in
        // `device_unless_headless`, both on this thread, and `nextest` gives the
        // test a process of its own.
        unsafe { std::env::set_var("GG_HEADLESS", "1") };
        let _ = Audio::device();
    }

    /// And the forgiving direction: under the law the shell's own constructor
    /// succeeds and is silent, so a headless run is a run rather than a refusal.
    #[test]
    fn the_shells_constructor_is_silent_under_the_law_rather_than_fatal() {
        // SAFETY: as above — one thread, one process.
        unsafe { std::env::set_var("GG_HEADLESS", "1") };
        let audio = Audio::device_unless_headless().unwrap();
        assert!(!audio.audible());
    }
}
