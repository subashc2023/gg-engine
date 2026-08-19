//! The recorded session (§6 M37 item 2): **one full round, first spawn to the
//! last miss**, and the hash sequence it produces.
//!
//! One script, three consumers — this crate's own tests drive it through the
//! declared systems table, `xtask replay --bless` encodes it into
//! `tests/replays/demo12-shooter.ggrp`, and `xtask reload --shooter` replays
//! that file through the real shell under three tiers. Authored here for demo
//! 10's reason: a script living in `xtask` cannot run under qemu.
//!
//! # A player, not a frame list
//!
//! Demo 11's driver reads the world back each tick and so does this one, for a
//! stronger reason than a platformer has: **where the targets stand is a draw**
//! (`Range::rng`), so a fixed list of turns would aim at wherever the course
//! dealt when the list was written and hit nothing the day the table gains a
//! spot. So the bot asks — nearest target first, turn to it *through the aim
//! axes* exactly as a mouse would, fire, and read the hit counter to find out
//! whether the round went where it was pointed.
//!
//! That last read is what keeps the room out of this file. A pillar between the
//! eye and a target is not modelled here; it is *observed*, as a shot that did
//! not raise [`Range::hits`], and the slot is not tried again. Nothing of
//! `cast`, `pierce` or the room's geometry is restated — which is demo 11's
//! pull-2 doctrine (claims derive from the scene) applied to a bot that has to
//! see.
//!
//! # The round the stream tells
//!
//! Since §6 M76 it is the game's whole shape rather than its middle: the title
//! screen, left with the key that leaves it; restart (the one press a fresh
//! session never needs, so the verb is in the recording); [`HITS_WANTED`]
//! targets taken, which is one more than a wave, so the stream crosses the
//! difficulty step and not only the first flat stretch of it; then the bot
//! **stops shooting** and the course walks its targets off one at a time until
//! the round is out of misses; then the over screen, and the round started
//! **again from it**, which is the loop closing.
//!
//! A session that only ever fired would print "hit" and never "escaped"; one
//! that only ever stood still would print "escaped" and never "hit"; one that
//! opened straight into the room would print neither bookend. [`LOG`]'s eight
//! milestones are those halves plus every edge of the game's shape, and each
//! fires once.
//!
//! Both screens are left by a **game verb** ([`crate::BEGIN`]) rather than by a
//! button, and that is a constraint rather than a taste: nothing here runs a
//! host, so `Widget::state` is never written, so a click and `ui_press` alike
//! land on nothing. A screen a widget was the only way off would be a screen
//! no gate below the shell could ever leave.
//!
//! # No scene, unlike its neighbour
//!
//! Demo 11's level is `scene.ggsave` and its stream is indexed from the save's
//! tick. This game deals its room from `bootstrap`, so the stream opens at tick
//! 0 and is replayed with no `--load` at all — demo 10's shape, and the reason
//! the baseline's tick column starts where it does.
//!
//! # The entry points are the caller's to supply
//!
//! [`Entry`] for demo 11's reason exactly: two `gg_game!` crates cannot share
//! one binary, so `xtask` resolves these four symbols out of the **built dylib**
//! (§4.2.2) while this crate's own tests hand over its rlib's exports.
//!
//! # The artifact is checked, not decoded
//!
//! Nothing here reads the `.ggrp` — `gg-input` is not a dependency a game crate
//! may take (§3). `xtask ci --push` byte-compares the checked-in file against
//! [`frames`], so a script edited without a `--bless` fails the push tier.

use gg_ecs::boundary::{
    self, AbiInfo, ActionId, ComponentsTable, Eye, HostApiV1, InputFrame, MAX_AXES, Renderable,
    SystemsTable, TickCtx,
};
use gg_ecs::hash::CanonicalHash;
use gg_ecs::{AliasError, Entity, Query, RegistryError, World};

use crate::{
    AIM_X, AIM_Y, BEGIN, FIRE, Gun, PAGE_OVER, RESTART, Range, STATE_OVER, Session, Target, Walker,
    look_per_count,
};
use gg_math::sim;

/// The four `gg_game!` exports, as pointers — resolved by whoever holds the
/// artifact. Demo 11's [`Entry`](crate::session::Entry) verbatim, and for its
/// reason: this crate's tests take them from its own rlib, `xtask` from the
/// built dylib, exactly as the host does (§4.2.2).
#[derive(Clone, Copy)]
pub struct Entry {
    /// `gg_game_abi`.
    pub abi: unsafe extern "C" fn() -> AbiInfo,
    /// `gg_game_init`.
    pub init: unsafe extern "C" fn(*const HostApiV1),
    /// `gg_game_components`.
    pub components: unsafe extern "C" fn() -> ComponentsTable,
    /// `gg_game_systems`.
    pub systems: unsafe extern "C" fn() -> SystemsTable,
}

/// What the shell's log must say, in order, when this session is replayed.
///
/// Eight, and the middle ones are why the game logs anything but its bookends:
/// "ready" and "over" are also what a session that spawned and stood still for
/// three target-lives would print, and a gate that cannot tell that apart is
/// checking that the game starts rather than that it was *played*. "hit" is the
/// half only a shot can reach; "escaped" is the half only patience can; "wave"
/// is the half only a *run* can, since it takes [`crate::WAVE_HITS`] hits in one
/// round to reach it.
///
/// "started" and "again" are §6 M76's, and they are the two edges a room is not
/// a game without: a session that opens on a title screen and a round that can
/// be played once more from the screen that reported it. Each is asserted
/// **once**, which is what makes "again" a claim — a second round is begun and
/// the stream ends inside it, so the loop is shown to close rather than to
/// spin.
pub const LOG: &[&str] = &[
    "shooter: ready",
    "shooter: started",
    "shooter: restarted",
    "shooter: hit",
    "shooter: wave",
    "shooter: escaped",
    "shooter: over",
    "shooter: again",
];

/// This session's name in `tests/replays/` — `.ggrp` beside `.hashes`.
pub const NAME: &str = "demo12-shooter";

/// Targets the bot takes before it stops shooting.
///
/// More than one: a single hit proves the ray reached a box, several prove the
/// streak pays what [`crate::STREAK_STEP`] says and that a taken slot is dealt
/// again on the same tick it was freed. **One more than
/// [`crate::WAVE_HITS`]** since §6 M76, which is the number that matters now:
/// at five the stream would end its shooting exactly on the wave boundary and
/// never deal a target at the shortened life, so the arc would be in the game
/// and not in the gate.
pub const HITS_WANTED: u32 = 6;

/// Full deflection in the fixed-point axis encoding (§4.7's `AXIS_SCALE`).
const STICK: i32 = 1024;

/// Idle ticks after the second round begins. Long enough that the fresh course
/// is dealt and ticked and short enough that nothing in it escapes, so the
/// stream holds exactly one of every milestone.
const TAIL: usize = 30;

/// A ceiling on every drive loop below, not a target: one minute of sim.
/// Reaching it means the policy stopped finishing, which is a named error
/// rather than an unbounded generator.
const MAX_TICKS: usize = 3_600;

/// Where the per-tick baseline lives.
///
/// The manifest dir walked up to the workspace root, demo 02's way: not an env
/// var and not a search, so a harness that cannot find the file fails loudly
/// rather than gating on nothing.
#[must_use]
pub fn baseline_path() -> std::path::PathBuf {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(std::path::Path::parent)
        .unwrap_or(manifest)
        .join("tests/replays")
        .join(format!("{NAME}.hashes"))
}

/// What can go wrong driving a world out of this crate's own tables.
#[derive(Debug)]
pub enum SessionError {
    /// The components table this crate declares was refused.
    Adopt(RegistryError),
    /// A system panicked behind the boundary shim.
    System(boundary::SystemPanic),
    /// The checked-in baseline does not parse, at this one-based line.
    Baseline(usize),
    /// A reader's query asked for the same component twice.
    Alias(AliasError),
    /// The supplied [`Entry`] describes a different build of the boundary.
    Abi,
    /// The policy ran out of ticks before the milestone it was working toward
    /// — the room stopped affording the round this module records.
    Stuck(&'static str),
}

impl core::fmt::Display for SessionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SessionError::Adopt(e) => write!(f, "demo 12's own components were refused: {e}"),
            SessionError::System(e) => write!(f, "{} panicked: {}", e.system, e.message),
            SessionError::Baseline(line) => write!(f, "malformed baseline at line {line}"),
            SessionError::Alias(e) => write!(f, "a reader's query does not hold: {e}"),
            SessionError::Abi => write!(
                f,
                "one build, one description — the entry points came from a different boundary"
            ),
            SessionError::Stuck(waiting) => write!(
                f,
                "the policy spent {MAX_TICKS} ticks without reaching: {waiting}"
            ),
        }
    }
}

impl core::error::Error for SessionError {}

/// The scripted round as an input stream, from tick 0.
///
/// # Errors
///
/// If the table is refused, a system panics, or the room stops affording the
/// round — no target this eye can reach, a weapon that never cools
/// ([`SessionError::Stuck`]).
pub fn frames(entry: &Entry) -> Result<Vec<InputFrame>, SessionError> {
    let mut bot = Bot::open(entry)?;

    // Tick 0 is `bootstrap`'s: the room, the round, the HUD, the title screen
    // and "ready".
    bot.play(idle())?;
    // Off the title (§6 M76). Nothing has run a tick of the round yet — the
    // body, the course and the room were all dealt behind the screen — so this
    // is a screen change and not a second bootstrap.
    bot.play(press(BEGIN))?;
    // The one press a fresh session never needs, so the verb reaches the
    // recording. It also reseeds the course, which is why every milestone
    // below is measured from here rather than from the opening deal.
    bot.play(press(RESTART))?;
    // The body opens in the air by one tick of gravity. A shot fired while it
    // is still settling leaves the eye somewhere between the aim and the
    // trigger, which is a flaky bot rather than an interesting one.
    while bot.walker()?.grounded == 0 {
        bot.play(idle())?;
        bot.check("the floor")?;
    }

    while bot.range()?.hits < HITS_WANTED {
        let Some(mark) = bot.pick()? else {
            // Everything standing has already stopped a bullet somewhere short
            // of itself. The course rotates on its own, so wait for a spot this
            // eye can reach rather than firing into the same pillar again.
            bot.play(idle())?;
            bot.check("a target this eye can reach")?;
            continue;
        };
        let before = bot.range()?.hits;
        let turn = bot.turn_to(mark.at)?;
        bot.play(turn)?;
        bot.play(press(FIRE))?;
        if bot.range()?.hits == before {
            // Observed, not modelled: the room is between the two, and this
            // module never learns which part of it.
            bot.note_miss(mark)?;
        }
        // The kick is given back inside the cooldown ([`crate::RECOIL_TICKS`]
        // under [`crate::FIRE_TICKS`]), so waiting it out is also what leaves
        // the pitch bit-identical to where the last turn put it.
        while bot.gun()?.cooldown > 0 {
            bot.play(idle())?;
            bot.check("the weapon to cool")?;
        }
        bot.check("three targets down")?;
    }

    // Hands off: the course walks its targets off one at a time until the round
    // is out of misses. This is the half a trigger cannot reach.
    while bot.range()?.state != STATE_OVER {
        bot.play(idle())?;
        bot.check("the round to end")?;
    }
    // The panel that reports it is one tick behind the last miss: `menu` reads
    // the state `targets` wrote, and `targets` is after it in the table. Waited
    // for rather than counted, so the day that ordering changes this script
    // still describes a player.
    while bot.screen()?.page != PAGE_OVER {
        bot.play(idle())?;
        bot.check("the over screen")?;
    }
    // The loop closing: the round that just ended, played again from the screen
    // that reported it. The stream ends a moment inside the second round, which
    // is what makes this an edge rather than a button that lit up.
    bot.play(press(BEGIN))?;

    for _ in 0..TAIL {
        bot.play(idle())?;
    }
    Ok(bot.frames)
}

/// Idle ticks the bot watches for after each round leaves, in [`endless`] only.
///
/// The trigger comes off between shots, and this is the one number in the
/// module chosen for the *gate* rather than for the game: a bot holding an
/// automatic weapon fires every [`crate::FIRE_TICKS`], and a retune swap
/// landing in that cadence would have eight ticks to prove a whole world
/// crossed it. Twenty-four is also how a person shoots a course — one round per
/// acquisition, then look at what it did.
const WATCH: usize = 24;

/// The same fight, without an end: the stream `xtask reload --retune` measures
/// a weapon retune over (§6 M37 item 6).
///
/// Demo 10's `endless` and demo 11's at a subject that has to aim — acquire the
/// nearest target this eye can reach, turn onto it, fire one round, wait the
/// weapon out, watch, again — and a round that runs out of misses is
/// **restarted** rather than left standing, so every tick of the stream is a
/// fight in progress and a swap aimed anywhere in the first half lands in one.
/// `blocked` is not cleared across a restart: the course re-deals but the room
/// does not move.
///
/// The cadence is therefore one shot every `WATCH + FIRE_TICKS + 2` ticks, and
/// the gap between them is what the gate spends proving the world crossed the
/// swap untouched.
///
/// # Errors
///
/// As [`frames`], less its milestones — this loop is bounded by `ticks` rather
/// than by a thing it is waiting for, so a room that stops affording a shot
/// yields idle frames instead of [`SessionError::Stuck`].
pub fn endless(entry: &Entry, ticks: usize) -> Result<Vec<InputFrame>, SessionError> {
    let mut bot = Bot::open(entry)?;
    bot.play(idle())?;
    // Off the title, as `frames` does (§6 M76) — a fight in progress cannot
    // begin behind a screen. Every round after this one is restarted with `R`
    // below, which the over screen also answers to.
    bot.play(press(BEGIN))?;
    while bot.frames.len() < ticks && bot.walker()?.grounded == 0 {
        bot.play(idle())?;
    }
    while bot.frames.len() < ticks {
        if bot.finished()? {
            bot.play(press(RESTART))?;
            // Watched after the restart too, and for the gate's reason again:
            // the fresh course is dealt on that same tick, and a shot the tick
            // after would leave the shortest quiet window in the stream at the
            // one moment the whole round was rebuilt.
            for _ in 0..WATCH {
                if bot.frames.len() >= ticks {
                    break;
                }
                bot.play(idle())?;
            }
            continue;
        }
        let Some(mark) = bot.pick()? else {
            bot.play(idle())?;
            continue;
        };
        let turn = bot.turn_to(mark.at)?;
        bot.play(turn)?;
        if bot.frames.len() >= ticks {
            break;
        }
        let before = bot.range()?.hits;
        bot.play(press(FIRE))?;
        if bot.range()?.hits == before {
            bot.note_miss(mark)?;
        }
        // Both waits give up the moment the course runs out, so the stream
        // holds exactly one over tick per round rather than standing over a
        // dead course for the rest of the rest — which is a tick a swap could
        // land on, and the one configuration this gate has nothing to say
        // about.
        while bot.frames.len() < ticks && bot.gun()?.cooldown > 0 && !bot.finished()? {
            bot.play(idle())?;
        }
        for _ in 0..WATCH {
            if bot.frames.len() >= ticks || bot.finished()? {
                break;
            }
            bot.play(idle())?;
        }
    }
    // The opening tick is played unconditionally, so a `ticks` below the
    // bootstrap is the one case that can overshoot.
    bot.frames.truncate(ticks);
    Ok(bot.frames)
}

// ------------------------------------------------------------ the hash side

/// The sequence as text: `tick hash` per line, hex, newline-terminated.
/// Demo 02's format exactly, demo 10's reasons verbatim.
#[must_use]
pub fn encode_baseline(sequence: &[CanonicalHash]) -> String {
    let mut out = String::with_capacity(sequence.len() * 40);
    for (tick, hash) in sequence.iter().enumerate() {
        out.push_str(&format!("{tick} {:032x}\n", hash.get()));
    }
    out
}

/// Parse [`encode_baseline`]'s output.
///
/// # Errors
///
/// On any line that is not `<tick> <32 hex digits>`.
pub fn parse_baseline(text: &str) -> Result<Vec<(u64, u128)>, SessionError> {
    let mut out = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let at = || SessionError::Baseline(index + 1);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (tick, hash) = line.split_once(' ').ok_or_else(at)?;
        out.push((
            tick.parse().map_err(|_| at())?,
            u128::from_str_radix(hash, 16).map_err(|_| at())?,
        ));
    }
    Ok(out)
}

/// Where `sequence` first disagrees with `baseline`, named the way §5.6
/// requires — a bare "hashes differ" over five hundred ticks is a wasted day.
#[must_use]
pub fn divergence(sequence: &[CanonicalHash], baseline: &[(u64, u128)]) -> Option<String> {
    for (tick, (left, (_, right))) in sequence.iter().zip(baseline).enumerate() {
        if left.get() != *right {
            return Some(format!(
                "tick {tick}: this run {:032x}, the baseline {right:032x}",
                left.get()
            ));
        }
    }
    (sequence.len() != baseline.len()).then(|| {
        format!(
            "this run is {} ticks and the baseline is {}",
            sequence.len(),
            baseline.len()
        )
    })
}

/// The canonical hash after each of `frames`, driven through the *declared*
/// systems table. Values are **not** a shell run's — demo 10's `hash_sequence`
/// says why (the shell's world also carries `gg_ui` hit state) — so the gate
/// ties the two by shape, not value.
///
/// # Errors
///
/// If the table is refused or a system panics.
pub fn hash_sequence(
    entry: &Entry,
    frames: &[InputFrame],
) -> Result<Vec<CanonicalHash>, SessionError> {
    drive(entry, frames, |world| Ok(world.canonical_hash()))
}

/// What a gate asks about a tick, read out of the world the systems just ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    /// Targets taken, all round.
    pub hits: u32,
    /// Targets that walked, all round.
    pub misses: u32,
    /// Rounds fired, all round — the accuracy denominator, and the number that
    /// says how much of the room the bot found the hard way.
    pub shots: u32,
    /// Points.
    pub score: u32,
    /// The best this session has seen. The one number that crosses a
    /// restart, which is what makes a second round a *continuation* rather
    /// than a fresh process (§6 M76).
    pub best: u32,
    /// Out of misses; the score is frozen.
    pub over: bool,
    /// Targets standing at the end of the tick.
    pub standing: u32,
    /// A screen is up and the sim is stopped (§6 M76).
    pub stopped: bool,
    /// Which screen — one of the `PAGE_*` constants, and meaningless while
    /// [`Progress::stopped`] is false, where it holds the page to come back to.
    pub page: u32,
}

/// [`Progress`] after each of `frames`, through the same table
/// [`hash_sequence`] drives. `xtask reload --shooter` uses it to name the ticks
/// the first hit, the first escape and the last miss land on — the shape this
/// run has, and the one a shell replaying onto the wrong verb ids cannot fake.
///
/// # Errors
///
/// As [`hash_sequence`].
pub fn progress(entry: &Entry, frames: &[InputFrame]) -> Result<Vec<Progress>, SessionError> {
    let range_q = Query::<&Range>::new().map_err(SessionError::Alias)?;
    let target_q = Query::<&Target>::new().map_err(SessionError::Alias)?;
    let session_q = Query::<&Session>::new().map_err(SessionError::Alias)?;
    drive(entry, frames, |world| {
        let mut out = Progress {
            hits: 0,
            misses: 0,
            shots: 0,
            score: 0,
            best: 0,
            over: false,
            standing: 0,
            stopped: false,
            page: 0,
        };
        world.each_ref(&session_q, |_, session: &Session| {
            out.stopped = session.paused != 0;
            out.page = session.page;
        });
        world.each_ref(&range_q, |_, range: &Range| {
            out.hits = range.hits;
            out.misses = range.misses;
            out.shots = range.shots;
            out.score = range.score;
            out.best = range.best.max(range.score);
            out.over = range.state == STATE_OVER;
        });
        world.each_ref(&target_q, |_, _: &Target| out.standing += 1);
        Ok(out)
    })
}

/// Run `frames` through the declared table, reading something out of the world
/// after each tick. Ticks count from 0 — this game deals its own room, so there
/// is no scene to resume at (demo 11's is the other shape).
fn drive<T>(
    entry: &Entry,
    frames: &[InputFrame],
    mut read: impl FnMut(&World) -> Result<T, SessionError>,
) -> Result<Vec<T>, SessionError> {
    let (table, mut world) = assemble(entry)?;
    let mut out = Vec::with_capacity(frames.len());
    let mut previous = idle();
    for (tick, frame) in frames.iter().enumerate() {
        let ctx = TickCtx::detached(tick as u64, 60, *frame, previous);
        // SAFETY: the table is the entry's own, its entries live for the
        // process, and `ctx` outlives the call.
        unsafe { world.run_systems(&table, &ctx) }.map_err(SessionError::System)?;
        out.push(read(&world)?);
        previous = *frame;
    }
    Ok(out)
}

/// A fresh empty world holding the entry's table — the host's protocol
/// registrations then the entry's adopt, the calls `gg_runtime::app` makes in
/// its order. The six come first even though this game declares all but
/// `gg.model` itself: registration order is archetype order and archetype order
/// is in the hash, so a driver that skipped them would author a baseline no
/// shell could ever reproduce.
fn assemble(entry: &Entry) -> Result<(SystemsTable, World), SessionError> {
    // SAFETY: the caller's contract on [`Entry`] — the four pointers are the
    // `gg_game!` exports of an artifact built from this same source, alive for
    // the process (a dylib is never unloaded, §4.2.2); `host_api()` returns a
    // `&'static` table. The ABI check is the boundary's own refusal.
    unsafe {
        if (entry.abi)() != boundary::abi_info() {
            return Err(SessionError::Abi);
        }
        (entry.init)(core::ptr::from_ref(boundary::host_api()));
        let mut world = World::new();
        world
            .register::<Renderable>()
            .map_err(SessionError::Adopt)?;
        world.register::<Eye>().map_err(SessionError::Adopt)?;
        world
            .register::<boundary::Model>()
            .map_err(SessionError::Adopt)?;
        world
            .register::<boundary::Light>()
            .map_err(SessionError::Adopt)?;
        world
            .register::<boundary::Widget>()
            .map_err(SessionError::Adopt)?;
        world
            .register::<boundary::Sky>()
            .map_err(SessionError::Adopt)?;
        world
            .adopt(&(entry.components)())
            .map_err(SessionError::Adopt)?;
        Ok(((entry.systems)(), world))
    }
}

// ------------------------------------------------------------ the policy

/// The closed loop: a world being driven, the frames chosen so far, and the
/// slots a bullet has already failed to reach.
struct Bot {
    world: World,
    table: SystemsTable,
    tick: u64,
    previous: InputFrame,
    frames: Vec<InputFrame>,
    /// [`Target::slot`]s a shot did not raise the hit counter on. Monotone: the
    /// room does not move, so a spot that was blocked once is blocked for the
    /// rest of the session and costs exactly one round, ever.
    blocked: Vec<u32>,
}

/// A target the bot has chosen, as the three facts it needs after the shot:
/// which spot to blacklist, which entity to check survived it, and where to
/// aim.
#[derive(Clone, Copy)]
struct Mark {
    slot: u32,
    entity: Entity,
    at: sim::DVec3,
}

impl Bot {
    fn open(entry: &Entry) -> Result<Self, SessionError> {
        let (table, world) = assemble(entry)?;
        Ok(Bot {
            world,
            table,
            tick: 0,
            previous: idle(),
            frames: Vec::new(),
            blocked: Vec::new(),
        })
    }

    /// Run one tick under `frame` and record it.
    fn play(&mut self, frame: InputFrame) -> Result<(), SessionError> {
        let ctx = TickCtx::detached(self.tick, 60, frame, self.previous);
        // SAFETY: as `drive` — the entry's own table, `ctx` outlives the call.
        unsafe { self.world.run_systems(&self.table, &ctx) }.map_err(SessionError::System)?;
        self.tick += 1;
        self.previous = frame;
        self.frames.push(frame);
        Ok(())
    }

    fn check(&self, waiting: &'static str) -> Result<(), SessionError> {
        if self.frames.len() >= MAX_TICKS {
            return Err(SessionError::Stuck(waiting));
        }
        Ok(())
    }

    fn one<C: gg_ecs::Component + Copy>(&self, missing: &'static str) -> Result<C, SessionError> {
        let query = Query::<&C>::new().map_err(SessionError::Alias)?;
        let mut out = None;
        self.world.each_ref(&query, |_, held: &C| out = Some(*held));
        out.ok_or(SessionError::Stuck(missing))
    }

    fn walker(&self) -> Result<Walker, SessionError> {
        self.one("a body in the world")
    }

    fn gun(&self) -> Result<Gun, SessionError> {
        self.one("a weapon in the world")
    }

    fn range(&self) -> Result<Range, SessionError> {
        self.one("a round in the world")
    }

    fn screen(&self) -> Result<Session, SessionError> {
        self.one("a session in the world")
    }

    fn finished(&self) -> Result<bool, SessionError> {
        Ok(self.range()?.state == STATE_OVER)
    }

    /// The nearest target this session has not already failed to reach. `None`
    /// is "every one standing has stopped a bullet short of itself".
    fn pick(&self) -> Result<Option<Mark>, SessionError> {
        let eye = self.one::<Eye>("an eye in the world")?.position;
        let query = Query::<(&Target, &Renderable)>::new().map_err(SessionError::Alias)?;
        let mut best: Option<(f64, Mark)> = None;
        self.world
            .each_ref(&query, |entity, (target, shape): (&Target, &Renderable)| {
                if self.blocked.contains(&target.slot) {
                    return;
                }
                let distance = (shape.position - eye).length();
                // Slot breaks a tie, so two targets equidistant from the eye
                // are chosen the same way on every machine.
                let beats = best.is_none_or(|(far, held)| {
                    distance < far || (distance == far && target.slot < held.slot)
                });
                if beats {
                    best = Some((
                        distance,
                        Mark {
                            slot: target.slot,
                            entity,
                            at: shape.position,
                        },
                    ));
                }
            });
        Ok(best.map(|(_, mark)| mark))
    }

    /// A round that did not raise the hit counter, judged.
    ///
    /// The slot is blacklisted only if **that same target is still standing** —
    /// identity by entity rather than by slot, because a spot is dealt again on
    /// the tick it is freed and a target that left between the aim and the
    /// trigger (taken, or walked at the end of its life) teaches nothing about
    /// the room. One round of a 288-tick session could afford the coarser rule;
    /// [`endless`] fires hundreds and would otherwise blacklist the whole
    /// course a spot at a time.
    fn note_miss(&mut self, mark: Mark) -> Result<(), SessionError> {
        if self.world.contains::<Target>(mark.entity) {
            self.blocked.push(mark.slot);
        }
        Ok(())
    }

    /// The frame that turns the view onto `at` — **through the aim axes**, the
    /// counts a mouse would have to travel at this session's sensitivity. A bot
    /// that wrote `Walker::yaw` would be recording a game nobody can play.
    fn turn_to(&self, at: sim::DVec3) -> Result<InputFrame, SessionError> {
        let eye = self.one::<Eye>("an eye in the world")?.position;
        let walker = self.walker()?;
        let sens = self.one::<Session>("a session in the world")?.sens;
        let to = (at - eye)
            .try_normalize()
            .ok_or(SessionError::Stuck("a target the eye is not standing in"))?;
        // `sim::fly_basis` inverted: forward is `(-cosP sinY, sinP, -cosP cosY)`.
        let yaw = sim::atan2(-to.x, -to.z) as f32;
        let pitch = sim::asin(to.y) as f32;
        let per = f64::from(look_per_count(sens));
        // `aim` subtracts the axis, so the counts are the negated delta.
        let counts = |delta: f32| (-f64::from(delta) / per * f64::from(STICK)) as i32;
        let mut frame = idle();
        frame.axes[AIM_X.index()] = counts(yaw - walker.yaw);
        frame.axes[AIM_Y.index()] = counts(pitch - walker.pitch);
        Ok(frame)
    }
}

// ------------------------------------------------------------ frame shapes

fn idle() -> InputFrame {
    InputFrame {
        buttons: 0,
        axes: [0; MAX_AXES],
    }
}

/// The frame with one button down. A press held across ticks is `press` again
/// (`just_pressed` is the edge, which `TickCtx::previous` supplies).
fn press(action: ActionId) -> InputFrame {
    let mut frame = idle();
    frame.buttons = 1 << action.index();
    frame
}
