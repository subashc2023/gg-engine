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
//!
//! # Two silences (§6 M50)
//!
//! Since a device can be *acquired* mid-session, "silent" had to split, and the
//! law is what splits it. A run under `GG_HEADLESS=1` holds no opener at all, so
//! no elapsed number of ticks can turn it loud; a run that merely found no sound
//! card at launch holds one and keeps asking, because the player who plugs a
//! headset in at the title screen is not asking to restart the game. The
//! permission is a field rather than a check at the call site — the constructors
//! that ask the law are the only ones that install it, and the opener asks again
//! on every attempt so the guarantee stays local.

mod bank;
mod device;
pub mod synth;

use std::sync::Arc;

use gg_ecs::boundary::{Prefs, Sound, wave};
use gg_ecs::{AliasError, Entity, Query, World};
use tracing::{debug, info, warn};

use device::Sink;

pub use bank::Bank;
pub use device::DeviceError;
pub use synth::{Mixer, Trigger, VOICES};

/// Ticks between two questions to the host about the output device — roughly
/// two seconds at 60 Hz (§6 M50).
///
/// Fixed, and deliberately *not* an exponential backoff. A backoff assumes
/// retries are expensive and that a failure predicts the next one; here a poll
/// is two property reads and the thing being waited for is a **human action** —
/// plugging the headphones back in. A player who left the jack out for a minute
/// should not then wait thirty seconds for the sound, which is exactly what a
/// doubling delay would buy them.
const PROBE_TICKS: u32 = 120;

/// True when this process runs under the headless law (§1.5).
///
/// Read here rather than taken from `gg-platform`: audio sits below windowing
/// and must not link it — a game's sound has nothing to do with a swapchain,
/// and the crate that owns the speakers should not be able to open a window by
/// accident. One env var, read the same way in both places — set and not
/// `"0"`/empty, `gg-platform`'s parse exactly.
#[must_use]
pub fn headless() -> bool {
    std::env::var_os("GG_HEADLESS").is_some_and(|v| !v.is_empty() && v != "0")
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
    /// The player's master volume (§6 M19). Read off the world's [`Prefs`]
    /// each tick and applied to the trigger, not the mixer: a note keeps the
    /// volume it started at, and there is no second channel into the audio
    /// thread to race.
    prefs: Query<&'static Prefs>,
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
    /// The clips (§6 M43), kept here as well as sent, so `Bank::holds` is
    /// answerable on a silent machine — which is where the gates run.
    bank: Option<Arc<Bank>>,
    /// The bank before this one, held one swap longer than it is needed.
    ///
    /// Not a leak and not caution: without it the audio thread would be the
    /// last owner of a replaced bank and would run the deallocator *inside the
    /// callback*, which is the textbook way to miss a deadline. One extra
    /// reference moves that free onto this thread, one pack rebuild later.
    retired: Option<Arc<Bank>>,
    out: Option<Box<dyn Sink>>,
    /// How to get another device, and the whole of §1.5's audio law as data:
    /// `None` means this observer may never open anything, however long it runs.
    /// Only the constructors that have already asked the law install one.
    opener: Option<Opener>,
    /// The loops currently holding a voice, by slot, sorted by the first.
    ///
    /// Host state that never reaches the hash, like [`seen`](Audio::seen) beside
    /// it — and the answer to the one question a fresh mixer cannot be asked.
    /// A one-shot is an event and is over; a loop is a **level**, and a device
    /// that came back without it comes back with the music missing.
    sustained: Vec<(usize, Sound)>,
    /// The master volume the last tick sent, so a reconnect between two ticks
    /// hands the new device the level the player set rather than full scale.
    master: f32,
    /// Ticks until the next question to the host. Counted down in
    /// [`tick`](Audio::tick), so a suspended session (§6 M49) does not probe —
    /// correctly, since it is not making a sound to lose.
    probe_in: u32,
}

/// Opens a device, or says why it could not. Boxed rather than a `fn` pointer so
/// a test can supply one that opens nothing at all.
type Opener = Box<dyn FnMut() -> Result<Box<dyn Sink>, DeviceError>>;

/// Why the supervisor wants a device (§6 M50).
///
/// Three ways for one predicate to be true and exactly one recovery, which is
/// the point of naming them together: a player who unplugged a headset, a player
/// who plugged one in, and a player who had neither when the game started are
/// asking the same question in three tones of voice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Want {
    /// Nothing is open — no device at launch, or the last attempt failed.
    Silent,
    /// The backend abandoned the stream: the endpoint is gone.
    Lost,
    /// The player's default output is somewhere else now.
    Superseded,
}

impl Want {
    /// The word the recovery's log line carries, so a session's story reads back
    /// out of `log.txt` without a second line to correlate against.
    fn reason(self) -> &'static str {
        match self {
            Want::Silent => "none was open",
            Want::Lost => "the last one went away",
            Want::Superseded => "the default moved",
        }
    }
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
            prefs: Query::new()?,
            seen: Vec::new(),
            next: Vec::new(),
            fired: Vec::new(),
            bank: None,
            retired: None,
            out: None,
            opener: None,
            sustained: Vec::new(),
            master: 1.0,
            probe_in: 0,
        })
    }

    /// Silent *for now*: nothing open, and permission to keep looking (§6 M50).
    ///
    /// The one difference from [`silent`](Audio::silent), and it is the whole of
    /// the law: this constructor asks §1.5 before handing out an opener, so a
    /// process that may not make a noise cannot acquire the ability to later.
    /// The opener asks again on every attempt rather than trusting this one —
    /// the guarantee is local that way instead of historical.
    fn attended() -> Result<Audio, AliasError> {
        enforce_headless_law();
        let mut audio = Audio::silent()?;
        audio.opener = Some(Box::new(|| {
            enforce_headless_law();
            Ok(Box::new(device::Device::open()?) as Box<dyn Sink>)
        }));
        Ok(audio)
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
        let mut audio = Audio::attended()?;
        audio.out = Some(Box::new(device::Device::open()?));
        Ok(audio)
    }

    /// The shell's constructor: a device when this process is allowed one and
    /// has one, silence otherwise.
    ///
    /// Never fails. A run with no sound card, or one under the headless law, is
    /// a run — the game is the point and the speakers are not.
    ///
    /// Two silences, and §6 M50 is the difference between them. Under the law
    /// this observer opens nothing ever again; on a machine that merely had no
    /// device at launch — headphones still in their case, a USB interface not
    /// plugged in yet — it keeps asking, and the game finds its voice when the
    /// player gives it one.
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
            Err(AudioError::Query(error)) => Err(error),
            Err(AudioError::Device(error)) => {
                warn!(%error, "audio: silent for now — no device, still looking");
                Audio::attended()
            }
        }
    }

    /// True when sound is coming out. The observable §1.5's gate asserts on, and
    /// the reason it is a method rather than a log line to grep.
    ///
    /// Reads the endpoint's own health rather than "a device was once opened"
    /// (§6 M50) — a stream the backend has abandoned is not audible, whatever
    /// this observer is still holding.
    #[must_use]
    pub fn audible(&self) -> bool {
        self.out.as_ref().is_some_and(|out| !out.lost())
    }

    /// Samples per second, and output channels, of the open device. `None` when
    /// silent — and different after a reconnect, because the device a player
    /// plugged in is under no obligation to run at the rate the last one did.
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
        self.supervise();
        let Audio {
            query,
            prefs,
            seen,
            next,
            fired,
            sustained,
            master,
            out,
            ..
        } = self;
        // The first `Prefs` walked, `Prefs`' documented rule; a world that
        // declares none plays at full volume, which is the zeroed default.
        let mut volume = 1.0f32;
        let mut found = false;
        world.each_ref(prefs, |_, p: &Prefs| {
            if !found {
                volume = p.volume();
                found = true;
            }
        });
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
                // Master volume lands in the trigger's own gain (§6 M19): what
                // `fired` reports is what will sound, so a demo's "the cue is
                // half as loud at half volume" is an ordinary assertion.
                //
                // Except for a loop, which outlives the slider (§6 M43). Music
                // pre-multiplied here would keep the volume it started at until
                // the track changed, so the mixer applies the live value to
                // those voices and this must not apply it twice.
                let mut sound = *sound;
                if sound.wave != wave::CLIP_LOOP {
                    sound.gain *= volume;
                }
                fired.push(Trigger {
                    // The entity's slot index, so one entity is one voice for as
                    // long as it lives. Two entities colliding modulo the bank
                    // cut each other off, which is a bank that is too small
                    // rather than a correctness problem.
                    slot: entity.index() as usize % VOICES,
                    sound,
                });
            }
        });
        // `each_ref` walks archetypes in creation order, so this is not sorted.
        next.sort_unstable_by_key(|(entity, _)| *entity);
        std::mem::swap(seen, next);
        // Which slots are holding music, read off what this tick decided rather
        // than tracked inside the walk: one entity is one slot, so a trigger of
        // any other kind *is* the end of the loop that was there (§6 M50).
        for trigger in fired.iter() {
            let looping = trigger.sound.wave == wave::CLIP_LOOP;
            match sustained.binary_search_by_key(&trigger.slot, |(slot, _)| *slot) {
                Ok(at) if looping => sustained[at].1 = trigger.sound,
                Ok(at) => drop(sustained.remove(at)),
                Err(at) if looping => sustained.insert(at, (trigger.slot, trigger.sound)),
                Err(_) => {}
            }
        }
        *master = volume;
        // Sent every tick and not only on a trigger: the volume is a level, and
        // a slider moved on a quiet frame still has to reach a playing loop.
        if let Some(device) = out {
            device.send(fired, volume);
        }
    }

    /// Ask the host whether the device is still the right one, and act if not.
    ///
    /// Free on almost every tick — a counter decrement — and two property reads
    /// twice a second's worth of them. Run from [`tick`](Audio::tick) rather than
    /// per frame, so a suspended session (§6 M49) does not probe: it is not
    /// making a sound, so it has none to lose, and it will ask on its first tick
    /// back.
    fn supervise(&mut self) {
        // §1.5, and the only place it is enforced at runtime after construction:
        // no opener, no attempt, however long the process lives.
        if self.opener.is_none() {
            return;
        }
        if self.probe_in > 0 {
            self.probe_in -= 1;
            return;
        }
        self.probe_in = PROBE_TICKS;
        let Some(want) = self.wanted() else {
            return;
        };
        match (want, &self.out) {
            // Once, by construction: `reconnect` drops the dead sink, so the
            // next poll finds `Silent` rather than this again.
            (Want::Lost, Some(out)) => {
                warn!(
                    device = out.name(),
                    "audio: the output device went away — looking for another"
                );
            }
            (Want::Superseded, _) => debug!("audio: the player's default output moved"),
            _ => {}
        }
        self.reconnect(want);
    }

    /// Why a different device is wanted, or `None` while the one open is fine.
    fn wanted(&self) -> Option<Want> {
        match &self.out {
            None => Some(Want::Silent),
            Some(out) if out.lost() => Some(Want::Lost),
            Some(out) if out.superseded() => Some(Want::Superseded),
            Some(_) => None,
        }
    }

    /// Open a device and put the session back onto it: the clips, then the
    /// loops, then the volume.
    ///
    /// `pub(crate)` rather than private because the operator's leg calls it
    /// directly. A reconnect against a *real* driver is the one claim a fake
    /// cannot make, and "unplug the headphones at the right second" is not a
    /// test — so the ignored leg forces the same path a pulled cable takes.
    pub(crate) fn reconnect(&mut self, want: Want) -> bool {
        // A lost stream is dropped before the attempt and a healthy one is not.
        // The endpoint behind the first is already gone; the second is still
        // playing, and going silent because the device we were *moving to*
        // would not open is the one outcome worse than not moving at all.
        if matches!(want, Want::Lost) {
            self.out = None;
        }
        let Some(opener) = &mut self.opener else {
            return false;
        };
        let sink = match opener() {
            Ok(sink) => sink,
            Err(error) => {
                debug!(%error, "audio: no device to open");
                return false;
            }
        };
        if let Some(bank) = &self.bank {
            sink.install(Arc::clone(bank));
        }
        let revival = self.revival();
        sink.send(&revival, self.master);
        info!(
            rate = sink.rate(),
            channels = sink.channels(),
            music = revival.len(),
            why = want.reason(),
            "audio: output device open"
        );
        self.out = Some(sink);
        true
    }

    /// The triggers a fresh mixer needs to be where the session left it.
    ///
    /// Loops only. A one-shot is an event that already happened, and replaying
    /// the set of them would make a reconnect a chord of every cue the session
    /// ever fired; a loop is a level, and a device that came back without it
    /// came back with the music missing.
    ///
    /// A revived loop restarts at its first sample rather than resuming
    /// mid-phrase, and that is the contract rather than a gap: carrying a read
    /// head onto a device whose rate is not the old one's is a resampler's worth
    /// of machinery to save a player the top of a bar.
    fn revival(&self) -> Vec<Trigger> {
        self.sustained
            .iter()
            .map(|&(slot, sound)| Trigger { slot, sound })
            .collect()
    }

    /// Take the master volume to nothing without touching the world (§6 M49).
    ///
    /// The one thing a suspended session needs and [`Audio::tick`] cannot do,
    /// because `tick` is a *tick's* and a suspension runs none: the device
    /// thread keeps mixing whatever it last held, so a looping track outlives
    /// the window it belongs to. Idempotent, so the caller states the level
    /// every suspended frame rather than tracking an edge, and no resume path is
    /// needed — `tick` sends the live volume every tick and the first one back
    /// restores it.
    ///
    /// A one-shot already sounding plays out, and that is the contract rather
    /// than a gap: its gain was baked when it fired (§6 M19), so this reaches
    /// loops only. What a player hears is the note that was already in the air.
    pub fn hush(&mut self) {
        if let Some(device) = &self.out {
            device.send(&[], 0.0);
        }
    }

    /// Install the clips this session can play (§6 M43).
    ///
    /// Handed the bank whole rather than a pack: this crate parses no audio
    /// format and links no pack, which is what keeps `cpal` a rental and leaves
    /// its sixth dependency slot unspent (§3).
    ///
    /// Replacing one cuts every sounding voice — a pack rebuilt under a running
    /// game changed the samples a note is reading.
    pub fn install(&mut self, bank: Bank) {
        info!(
            clips = bank.clips(),
            frames = bank.frames(),
            "audio: clips installed"
        );
        let bank = Arc::new(bank);
        self.retired = self.bank.replace(bank);
        if let Some(device) = &self.out {
            let Some(bank) = &self.bank else { return };
            device.install(Arc::clone(bank));
            // The install cut every voice, so the loops are stated again — the
            // same level `reconnect` restores, one door down (§6 M50). A pack
            // rebuilt under a running game changed the samples the music was
            // reading, and the music is what a player notices going missing.
            device.send(&self.revival(), self.master);
        }
    }

    /// The installed clips, for the one question a silent machine can answer
    /// about them: whether a cue's id resolves.
    #[must_use]
    pub fn bank(&self) -> Option<&Bank> {
        self.bank.as_deref()
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

    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    use gg_ecs::boundary::wave;

    use super::*;

    /// A sink that opens nothing, so the supervisor's whole policy — notice,
    /// wait, open, put the bank back, restart the music — is exercised on a
    /// machine with no sound card (§1.5, §6 M50).
    #[derive(Default)]
    struct Fake {
        rate: u32,
        channels: u16,
        lost: Cell<bool>,
        superseded: Cell<bool>,
        sent: RefCell<Vec<Trigger>>,
        master: Cell<f32>,
        banks: Cell<usize>,
    }

    impl Fake {
        fn triggers(&self) -> Vec<Trigger> {
            self.sent.borrow().clone()
        }
    }

    impl Sink for Rc<Fake> {
        fn send(&self, triggers: &[Trigger], master: f32) -> usize {
            self.sent.borrow_mut().extend_from_slice(triggers);
            self.master.set(master);
            triggers.len()
        }
        fn install(&self, _bank: Arc<Bank>) {
            self.banks.set(self.banks.get() + 1);
        }
        fn rate(&self) -> u32 {
            self.rate
        }
        fn channels(&self) -> u16 {
            self.channels
        }
        fn name(&self) -> &str {
            "fake"
        }
        fn lost(&self) -> bool {
            self.lost.get()
        }
        fn superseded(&self) -> bool {
            self.superseded.get()
        }
    }

    /// An observer whose opener hands out `plan` in order — a rate for a device
    /// that opens, `None` for an attempt that fails — and fails every attempt
    /// past the end of it.
    struct Bench {
        audio: Audio,
        opened: Rc<RefCell<Vec<Rc<Fake>>>>,
        attempts: Rc<Cell<usize>>,
    }

    impl Bench {
        fn new(plan: &[Option<u32>]) -> Bench {
            let plan = plan.to_vec();
            let opened: Rc<RefCell<Vec<Rc<Fake>>>> = Rc::default();
            let attempts = Rc::new(Cell::new(0usize));
            let (out, count) = (Rc::clone(&opened), Rc::clone(&attempts));
            let mut audio = Audio::silent().unwrap();
            audio.opener = Some(Box::new(move || {
                let at = count.get();
                count.set(at + 1);
                let rate = plan
                    .get(at)
                    .copied()
                    .flatten()
                    .ok_or(DeviceError::NoDevice)?;
                let fake = Rc::new(Fake {
                    rate,
                    channels: 2,
                    master: Cell::new(1.0),
                    ..Fake::default()
                });
                out.borrow_mut().push(Rc::clone(&fake));
                Ok(Box::new(fake) as Box<dyn Sink>)
            }));
            Bench {
                audio,
                opened,
                attempts,
            }
        }

        /// The `n`th device this bench handed out.
        fn device(&self, n: usize) -> Rc<Fake> {
            Rc::clone(&self.opened.borrow()[n])
        }

        fn devices(&self) -> usize {
            self.opened.borrow().len()
        }

        fn run(&mut self, world: &World, ticks: u32) {
            for _ in 0..ticks {
                self.audio.tick(world);
            }
        }
    }

    /// A world holding one music entity and one cue entity, both registered with
    /// the observer and neither yet played.
    fn scored() -> (World, Entity, Entity) {
        let mut world = World::new();
        world.register::<Sound>().unwrap();
        let music = world.spawn();
        world.insert(music, Sound::music("theme", 0.6)).unwrap();
        let cue = world.spawn();
        world
            .insert(cue, Sound::tone(wave::SQUARE, 440.0, 40, 0.5))
            .unwrap();
        (world, music, cue)
    }

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

    /// §6 M19's master volume: the world's `Prefs` scales every trigger's gain,
    /// half volume is half gain, and a world declaring none is full volume —
    /// the zeroed default, which is what a migration writes.
    #[test]
    fn the_worlds_prefs_scale_what_fires() {
        use gg_ecs::boundary::{Prefs, QUIET_MAX, cursor};

        let (mut world, entities) = world_with(1);
        let mut audio = Audio::silent().unwrap();
        audio.tick(&world);
        bump(&mut world, entities[0]);
        audio.tick(&world);
        assert_eq!(audio.fired()[0].sound.gain, 0.5, "no Prefs is full volume");

        world.register::<Prefs>().unwrap();
        let prefs = world.spawn();
        world
            .insert(
                prefs,
                Prefs {
                    cursor: cursor::SOFTWARE,
                    quiet: QUIET_MAX / 2,
                    ..Default::default()
                },
            )
            .unwrap();
        bump(&mut world, entities[0]);
        audio.tick(&world);
        assert_eq!(audio.fired()[0].sound.gain, 0.25, "half of the game's 0.5");
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
    /// Safe under `nextest`, which runs each test in its own process — and that
    /// is now the only thing making it safe: since §6 M50 this binary holds an
    /// `#[ignore]`d leg that *does* open a device, so `cargo test -- --ignored`
    /// in one process could inherit a `GG_HEADLESS` this test set. `xtask
    /// interactive` runs the ignored legs under `nextest`, which is where the
    /// guarantee comes from.
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
    ///
    /// It is the *unattended* silence, which is the other half (§6 M50): a run
    /// forbidden to make a noise must not acquire an opener it could use later.
    #[test]
    fn the_shells_constructor_is_silent_under_the_law_rather_than_fatal() {
        // SAFETY: as above — one thread, one process.
        unsafe { std::env::set_var("GG_HEADLESS", "1") };
        let audio = Audio::device_unless_headless().unwrap();
        assert!(!audio.audible());
        assert!(
            audio.opener.is_none(),
            "a headless run may look for a device"
        );
    }

    /// §1.5 as a property of a *running* observer rather than of its
    /// constructor: no elapsed number of ticks turns a silent one loud.
    #[test]
    fn a_silent_observer_never_opens_anything_however_long_it_runs() {
        let (world, _) = world_with(2);
        let mut audio = Audio::silent().unwrap();
        for _ in 0..10_000 {
            audio.tick(&world);
        }
        assert!(!audio.audible());
        assert!(audio.opener.is_none());
    }

    /// The milestone in one test: a device goes away, another opens, and the
    /// music is playing on it — while the cue that fired before the loss stays
    /// where it belongs, in the past.
    #[test]
    fn a_lost_device_is_replaced_and_the_music_comes_back() {
        let (mut world, music, cue) = scored();
        let mut bench = Bench::new(&[Some(48_000), Some(44_100)]);
        bench.run(&world, 1); // first sight registers both; the device opens
        assert_eq!(bench.devices(), 1);

        bump(&mut world, music);
        bump(&mut world, cue);
        bench.run(&world, 1);
        let first = bench.device(0);
        assert_eq!(first.triggers().len(), 2, "the tick played both");

        first.lost.set(true);
        bench.run(&world, PROBE_TICKS + 1);
        assert_eq!(bench.devices(), 2, "the lost device was not replaced");
        let revived = bench.device(1).triggers();
        assert!(
            !revived.is_empty(),
            "the device came back without the music"
        );
        assert_eq!(revived.len(), 1, "a reconnect replayed the session's cues");
        assert_eq!(
            revived[0].sound.wave,
            wave::CLIP_LOOP,
            "the voice that came back was not the music"
        );
    }

    /// A stream the backend has abandoned is not sound coming out, whatever the
    /// observer is still holding — and no probe is needed to say so.
    #[test]
    fn a_lost_device_stops_being_audible_immediately() {
        let (world, _, _) = scored();
        let mut bench = Bench::new(&[Some(48_000)]);
        bench.run(&world, 1);
        assert!(bench.audio.audible());
        bench.device(0).lost.set(true);
        assert!(!bench.audio.audible(), "a dead stream reported sound");
    }

    /// One entity is one voice, so a cue fired into the slot the music held *is*
    /// the end of that music — and a reconnect must not resurrect it.
    #[test]
    fn a_slot_a_cue_took_over_stops_being_music() {
        let mut world = World::new();
        world.register::<Sound>().unwrap();
        let entity = world.spawn();
        world.insert(entity, Sound::music("theme", 0.6)).unwrap();
        let mut bench = Bench::new(&[Some(48_000), Some(48_000)]);
        bench.run(&world, 1);

        bump(&mut world, entity);
        bench.run(&world, 1);
        // The game's own way of stopping a loop: something else takes the voice.
        let mut over = Sound::tone(wave::SILENT, 0.0, 0, 0.0);
        over.seq = 2;
        world.insert(entity, over).unwrap();
        bench.run(&world, 1);

        bench.device(0).lost.set(true);
        bench.run(&world, PROBE_TICKS + 1);
        assert!(
            bench.device(1).triggers().is_empty(),
            "a loop the game had already stopped came back"
        );
    }

    /// The reconnect is a *poll*, not a spin: a dead device costs one attempt
    /// per probe however many ticks pass between them.
    #[test]
    fn a_reconnect_is_not_attempted_on_every_tick() {
        let (world, _, _) = scored();
        let mut bench = Bench::new(&[Some(48_000), Some(48_000), Some(48_000)]);
        bench.run(&world, 1);
        assert_eq!(bench.attempts.get(), 1);

        bench.device(0).lost.set(true);
        bench.run(&world, 10);
        assert_eq!(bench.attempts.get(), 1, "the host was asked every tick");
        bench.run(&world, PROBE_TICKS);
        assert_eq!(bench.attempts.get(), 2, "the probe never came round");
    }

    /// Headphones plugged in while the game plays to the speakers: nothing
    /// errored, nothing is lost, and the sound has to move anyway.
    #[test]
    fn a_default_output_that_moved_is_followed() {
        let (world, _, _) = scored();
        let mut bench = Bench::new(&[Some(48_000), Some(44_100)]);
        bench.run(&world, 1);
        bench.device(0).superseded.set(true);
        bench.run(&world, PROBE_TICKS + 1);
        assert_eq!(bench.devices(), 2);
        assert_eq!(
            bench.audio.format(),
            Some((44_100, 2)),
            "the rate is the new device's"
        );
    }

    /// And the direction that must not fail open: a healthy device is not given
    /// up for one that would not open. Going silent because the endpoint we were
    /// *moving to* refused is worse than not moving.
    #[test]
    fn a_playing_device_survives_an_attempt_that_fails() {
        let (world, _, _) = scored();
        let mut bench = Bench::new(&[Some(48_000), None]);
        bench.run(&world, 1);
        bench.device(0).superseded.set(true);
        bench.run(&world, PROBE_TICKS + 1);
        assert_eq!(bench.attempts.get(), 2, "the move was never attempted");
        assert_eq!(bench.devices(), 1);
        assert!(
            bench.audio.audible(),
            "a failed move took the sound with it"
        );
        assert_eq!(bench.audio.format(), Some((48_000, 2)));
    }

    /// The third way to want a device, and the one that is nobody's fault: the
    /// machine had none when the game started. A player who plugs a headset in
    /// at the title screen gets sound without restarting.
    #[test]
    fn a_machine_with_no_device_at_launch_picks_one_up_later() {
        let (world, _, _) = scored();
        let mut bench = Bench::new(&[None, None, Some(44_100)]);
        bench.run(&world, 1);
        assert!(!bench.audio.audible());
        bench.run(&world, PROBE_TICKS + 1);
        assert!(!bench.audio.audible());
        bench.run(&world, PROBE_TICKS + 1);
        assert!(
            bench.audio.audible(),
            "the device that appeared was never taken"
        );
        assert_eq!(bench.audio.format(), Some((44_100, 2)));
    }

    /// Everything a fresh mixer cannot know: which clips the session installed,
    /// and how loud the player asked for it.
    #[test]
    fn a_reconnected_device_gets_the_clips_and_the_players_volume() {
        use gg_ecs::boundary::{QUIET_MAX, cursor};

        let (mut world, music, _) = scored();
        world.register::<Prefs>().unwrap();
        let prefs = world.spawn();
        world
            .insert(
                prefs,
                Prefs {
                    cursor: cursor::SOFTWARE,
                    quiet: QUIET_MAX / 2,
                    ..Default::default()
                },
            )
            .unwrap();

        let mut bench = Bench::new(&[Some(48_000), Some(48_000)]);
        let mut bank = Bank::new();
        bank.add(Sound::music("theme", 0.6).clip, 22_050, &[0i16; 64]);
        bench.run(&world, 1);
        bench.audio.install(bank);
        bump(&mut world, music);
        bench.run(&world, 1);

        bench.device(0).lost.set(true);
        bench.run(&world, PROBE_TICKS + 1);
        let second = bench.device(1);
        assert_eq!(
            second.banks.get(),
            1,
            "the clips did not follow the session"
        );
        assert_eq!(
            second.master.get(),
            0.5,
            "the reconnect played at full scale"
        );
    }

    /// A pack rebuilt under a running game cuts every voice, which is right, and
    /// left the music off until something retriggered it, which was not (§6 M50).
    #[test]
    fn a_bank_installed_under_playing_music_restarts_it() {
        let (mut world, music, _) = scored();
        let mut bench = Bench::new(&[Some(48_000)]);
        bench.run(&world, 1);
        bump(&mut world, music);
        bench.run(&world, 1);
        let device = bench.device(0);
        let before = device.triggers().len();

        bench.audio.install(Bank::new());
        let after = device.triggers();
        assert_eq!(device.banks.get(), 1);
        assert_eq!(after.len(), before + 1, "the music was cut and left off");
        assert_eq!(after[before].sound.wave, wave::CLIP_LOOP);
    }

    /// The one claim a fake cannot make: that a **second** `Device::open` in the
    /// same process reaches a real driver, and that the music comes back on it.
    ///
    /// The loss is forced rather than waited for — a test needing a cable pulled
    /// at the right second is not a test, and `reconnect(Want::Lost)` is the
    /// exact path a pulled cable takes. What an operator hears is a chord, a
    /// gap, and the same chord again on whatever the default is now; unplugging
    /// something during the first sleep exercises the *automatic* path instead
    /// and should sound the same.
    #[test]
    #[ignore = "opens an audio device (§1.5) — run via `cargo xtask interactive`"]
    fn a_real_device_comes_back_and_brings_the_music_with_it() {
        const RATE: u32 = 22_050;
        let mut audio = Audio::device().expect("no audio device on this machine");
        assert!(audio.audible());

        let mut world = World::new();
        world.register::<Sound>().unwrap();
        let entity = world.spawn();
        let mut theme = Sound::music("chord", 0.3);
        let mut samples = Vec::new();
        for frame in 0..RATE {
            let t = f64::from(frame) / f64::from(RATE);
            let voices: f64 = [261.63_f64, 329.63, 392.00]
                .iter()
                .map(|hz| gg_math::sim::sin(t * hz * core::f64::consts::TAU))
                .sum();
            samples.push((voices / 3.0 * 9_000.0) as i16);
        }
        let mut bank = Bank::new();
        bank.add(theme.clip, RATE, &samples);
        audio.install(bank);

        world.insert(entity, theme).unwrap();
        audio.tick(&world); // first sight registers
        theme.play();
        world.insert(entity, theme).unwrap();
        audio.tick(&world);
        assert_eq!(audio.fired().len(), 1);
        std::thread::sleep(std::time::Duration::from_millis(2_500));

        assert!(audio.reconnect(Want::Lost), "the device did not reopen");
        assert!(audio.audible());
        std::thread::sleep(std::time::Duration::from_millis(2_500));
    }
}
