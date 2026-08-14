//! The recorded session (§6 M38 item 14): **the whole mission, parking orbit to
//! capture**, and the hash sequence it produces.
//!
//! One script, three consumers — demo 12's arrangement exactly: this crate's own
//! tests drive it through the declared systems table, `xtask replay --bless`
//! encodes it into `tests/replays/demo13-orbit.ggrp`, and `xtask reload --orbit`
//! replays that file through the real shell under three tiers. Authored here
//! because a script living in `xtask` cannot run under qemu.
//!
//! # A pilot, and the two numbers it cannot see
//!
//! Every stage below closes its own loop on something the world says: the burn
//! ends when the heliocentric conic reaches the target's orbit, the cruise ends
//! when the *game* hands the conic over, the capture burn ends when the conic
//! about the target closes. Two numbers are not observable that way and are
//! constants ([`Plan`]): how long to wait in the parking orbit before lighting
//! the engine, and how long to hold it lit. They aim the escape asymptote and
//! set the transfer's size — a targeting problem, and the answer is read off
//! `gg-tools transfer` rather than guessed here.
//!
//! That split is the honest one. A pilot cannot *measure* where a burn it has
//! not made yet will put it; every other decision in the mission is a
//! measurement.
//!
//! # Warp is the pilot's problem too
//!
//! §6 M38 item 10 measured the sampling hazard a warp stride creates for the
//! regime *condition* and found the transfer's own encounter is what binds: the
//! target's grip is ~9.2e7 m wide and the ship crosses it in a few hours, so a
//! stride of 1e6 ticks steps straight over it. So the cruise throttles warp by
//! **time to encounter** — the largest step whose stride is under a sixteenth of
//! the distance divided by the closing speed. It is the rule item 10's bubble
//! residence implies, closed on the world rather than tabulated.
//!
//! # The mission the stream tells
//!
//! Spawn, wait, burn, escape the departure planet, cross, be caught by the
//! target, and restart. [`LOG`]'s five milestones are exactly the ones a
//! *flight* reaches and standing still cannot: `escaped` needs the departure
//! burn to have been big enough, `intercept` needs it to have been aimed, and
//! `captured` needs the retro burn to have closed the conic before the ship went
//! past. Each fires once, which is why `lit` and `cut` are not among them —
//! those happen twice, once at each end of the flight.
//!
//! # The entry points are the caller's to supply
//!
//! [`Entry`] for demo 11's reason exactly: two `gg_game!` crates cannot share
//! one binary, so `xtask` resolves these four symbols out of the **built dylib**
//! (§4.2.2) while this crate's own tests hand over its rlib's exports.

use gg_ecs::boundary::{
    self, AbiInfo, ActionId, ComponentsTable, Eye, HostApiV1, InputFrame, MAX_AXES, Renderable,
    SystemsTable, TickCtx,
};
use gg_ecs::hash::CanonicalHash;
use gg_ecs::{AliasError, Entity, Query, RegistryError, World};

use crate::{
    BURN, Body, Bubble, CAPTURED, Epoch, FLYING, OCHRE, RESTART, RETRO, Rails, Ship, VERGE,
    WARP_DOWN, WARP_UP, WARPS,
};
use gg_math::sim;

/// The four `gg_game!` exports, as pointers — resolved by whoever holds the
/// artifact. Demo 12's verbatim, and for its reason: this crate's tests take
/// them from its own rlib, `xtask` from the built dylib, as the host does.
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
/// Five, each exactly once, and every one of them is a claim a session that
/// merely *started* cannot reach. `escaped` is the departure burn having been
/// large enough to leave the planet; `intercept` is it having been aimed, which
/// is a 9.2e7 m window across a 2.3e11 m crossing; `captured` is the retro burn
/// having closed the conic above the target's surface. `restarted` is the last
/// frame of the stream and proves the mission can be flown again.
pub const LOG: &[&str] = &[
    "orbit: ready",
    "orbit: escaped",
    "orbit: intercept",
    "orbit: captured",
    "orbit: restarted",
];

/// This session's name in `tests/replays/` — `.ggrp` beside `.hashes`.
pub const NAME: &str = "demo13-orbit";

/// The two numbers a pilot cannot measure: when to light the engine, and for
/// how long. Everything else in the mission is closed on the world.
///
/// `wait` is in **host** ticks at `wait_step`'s warp, so the departure epoch is
/// `wait * WARPS[wait_step]` sim ticks (plus the taps that got there) and the
/// resolution of the search over it is the stride. Both are read off
/// `gg-tools transfer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    /// Host ticks idled before the engine is lit.
    pub wait: u32,
    /// Index into [`WARPS`] the wait is spent at — the departure epoch's
    /// resolution, and what makes the asymptote steerable to a fraction of a
    /// degree.
    pub wait_step: u32,
    /// Host ticks the departure burn is held. One tick is 4.17 m/s.
    pub burn: u32,
    /// Index into [`WARPS`] used to climb out of the departure planet's grip,
    /// where there is no encounter to throttle against.
    pub escape_step: u32,
}

/// The plan that flies it, read off `gg-tools transfer` (§6 M38 item 14).
///
/// A search over `wait` at this stride sweeps the escape asymptote through a
/// full turn every 5828 s of parking orbit, which is what steers the arrival;
/// `burn` sets how far out the transfer reaches. The pair is a lattice in miss
/// distance, and this is the cell that lands inside the target's grip.
pub const FLOWN: Plan = Plan {
    wait: 0,
    wait_step: 3,
    burn: 900,
    escape_step: 5,
};

/// Where the per-tick baseline lives — demo 12's rule verbatim: the manifest
/// dir walked up to the workspace root, so a harness that cannot find the file
/// fails loudly rather than gating on nothing.
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
    /// The pilot ran out of ticks before the stage it was working toward.
    Stuck(&'static str),
    /// The mission ended somewhere other than captured — [`FLOWN`] no longer
    /// flies it, which is a retuning rather than a bug in the harness.
    Missed(u32),
}

impl core::fmt::Display for SessionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SessionError::Adopt(e) => write!(f, "demo 13's own components were refused: {e}"),
            SessionError::System(e) => write!(f, "{} panicked: {}", e.system, e.message),
            SessionError::Baseline(line) => write!(f, "malformed baseline at line {line}"),
            SessionError::Alias(e) => write!(f, "a reader's query does not hold: {e}"),
            SessionError::Abi => write!(
                f,
                "one build, one description — the entry points came from a different boundary"
            ),
            SessionError::Stuck(waiting) => write!(
                f,
                "the pilot spent {MAX_TICKS} ticks without reaching: {waiting}"
            ),
            SessionError::Missed(outcome) => write!(
                f,
                "the mission ended at outcome {outcome} rather than captured — `session::FLOWN` \
                 no longer aims at the target, and `gg-tools transfer` is what re-aims it"
            ),
        }
    }
}

impl core::error::Error for SessionError {}

/// A ceiling on every drive loop, not a target. Reaching it means a stage
/// stopped finishing, which is a named error rather than an unbounded run.
const MAX_TICKS: usize = 20_000;

/// Idle ticks after the capture, before the restart that closes the stream. A
/// frozen outcome that is never ticked again proves nothing about staying
/// frozen.
const TAIL: usize = 30;

/// How much of the time to encounter one warp stride may spend. Sixteen samples
/// across the closing is what keeps the target's grip from being stepped over
/// (§6 M38 item 10's residence finding, closed on the world).
const GUARD: f64 = 16.0;

/// The scripted mission as an input stream, from tick 0.
///
/// # Errors
///
/// If the table is refused, a system panics, a stage stops finishing, or the
/// flight ends anywhere but captured ([`SessionError::Missed`]).
pub fn frames(entry: &Entry) -> Result<Vec<InputFrame>, SessionError> {
    let flight = fly(entry, FLOWN)?;
    if flight.outcome != CAPTURED {
        return Err(SessionError::Missed(flight.outcome));
    }
    Ok(flight.frames)
}

/// Host tick [`endless`] lights the engine on, and how long it holds it.
///
/// One burn and one only, late: a reload gate's swap is aimed at a tick in the
/// low thousands and lands hundreds late by an amount that is the machine's, so
/// what this stream owes it is a coast long enough that the aim cannot miss and
/// an event far enough past it to be worth calling a window.
pub const LIGHT_AT: usize = 6_000;
/// Ticks the engine is held for — 120 at [`crate::THRUST`] is 500 m/s, well
/// inside a [`crate::TANK`] and enough to reshape the parking conic outright.
pub const LIGHT_FOR: usize = 120;

/// The parking orbit coasted for as long as the caller asks, warped, then one
/// burn.
///
/// [`frames`] ends — a mission is a thing that finishes, which is what left §6
/// M38's exit row a `--retune`-shaped case short. This is the run still in
/// progress when a rebuilt dylib lands under it, and the shape is demo 11's
/// verbatim for demo 11's reason: the ship is **coasting** at the swap, so an
/// engine constant is latent when it arrives and bites exactly at the next
/// light.
///
/// Two warp taps first, which is this game's own contribution to the shape. A
/// world that crosses a rebuild while its *sim clock* is running at 100x is a
/// stronger statement than one that crosses it at 1x, and it costs a stream two
/// frames. Lighting the engine drops warp back to 1 by itself ([`crate::control`]),
/// so the burn needs no undoing.
///
/// Open-loop on purpose: nothing here needs the world, so the stream is a pure
/// function of `ticks`.
#[must_use]
pub fn endless(ticks: usize) -> Vec<InputFrame> {
    let mut out = Vec::with_capacity(ticks);
    while out.len() < ticks {
        let at = out.len();
        out.push(match at {
            // Each tap is an *edge*, so a released tick has to sit between them.
            30 | 32 => press(WARP_UP),
            _ if (LIGHT_AT..LIGHT_AT + LIGHT_FOR).contains(&at) => press(BURN),
            _ => idle(),
        });
    }
    out.truncate(ticks);
    out
}

/// Whether [`endless`] has the ship coasting at this offset — no engine, which
/// is what makes a thrust constant latent. For a caller choosing where a mid-run
/// swap should land.
#[must_use]
pub fn coasting(at: usize) -> bool {
    !(LIGHT_AT..LIGHT_AT + LIGHT_FOR).contains(&at)
}

/// What one [`Plan`] does, measured — the stream it produces and the four
/// numbers a search over plans reads.
#[derive(Clone, Debug)]
pub struct Flight {
    /// The input stream, ready to record.
    pub frames: Vec<InputFrame>,
    /// [`FLYING`] if the horizon arrived first, else how it ended.
    pub outcome: u32,
    /// Closest the ship came to the target, metres. Sampled per host tick, so
    /// the warp throttle is what makes it meaningful near the encounter.
    pub approach: f64,
    /// Periapsis of the conic about the target at the moment it took the ship,
    /// metres above its centre — the number that separates a capture from a
    /// crater, and `None` if there was no intercept.
    pub periapsis: Option<f64>,
    /// Delta-v spent, m/s.
    pub spent: f64,
    /// The heliocentric conic the departure bought, stated at the epoch it was
    /// handed over on. This is what a targeting search is actually searching
    /// *for* — the plan sets the transfer and nothing downstream changes it,
    /// because a planet in this world perturbs nobody.
    pub transfer: Option<Rails>,
}

/// Fly `plan` and report what it did. The stream is returned whatever the
/// outcome: a search reads [`Flight::approach`] off the misses.
///
/// # Errors
///
/// As [`frames`], less the outcome check — a plan that misses is a measurement
/// here rather than an error.
pub fn fly(entry: &Entry, plan: Plan) -> Result<Flight, SessionError> {
    let mut pilot = Pilot::open(entry)?;
    // Tick 0 is `bootstrap`'s: the system, the ship, the HUD, and "ready".
    pilot.play(idle())?;

    // Departure. The wait is what aims the escape asymptote — a parking orbit
    // turns once every 5828 s and the burn leaves along a direction fixed by
    // where in it the engine was lit.
    pilot.set_warp(plan.wait_step)?;
    for _ in 0..plan.wait {
        pilot.play(idle())?;
        pilot.check("the departure window")?;
    }

    // The burn. Warp drops itself the tick the bubble opens (§6 M38 item 10),
    // so nothing here has to ask for 1x.
    for _ in 0..plan.burn {
        pilot.play(press(BURN))?;
        pilot.check("the departure burn")?;
    }
    pilot.play(idle())?; // cut: `from_state` on what the burn produced

    // Out of the planet's grip. There is no encounter to throttle against yet,
    // so this leg is the one place warp is a constant.
    pilot.set_warp(plan.escape_step)?;
    while pilot.primary()? == VERGE.index {
        pilot.play(idle())?;
        pilot.check("the departure planet's grip")?;
        if pilot.outcome()? != FLYING {
            return pilot.report(None);
        }
    }
    pilot.latch_transfer();

    // The crossing. Warp is throttled by time to encounter, which is the whole
    // of what item 10's residence finding asks of a caller.
    let mut horizon = 0usize;
    while pilot.primary()? == 0 && pilot.outcome()? == FLYING {
        let wanted = pilot.warp_for_encounter()?;
        pilot.nudge_warp(wanted)?;
        pilot.play(idle())?;
        horizon += 1;
        if horizon >= MAX_TICKS / 2 {
            // A plan that never arrives is a measurement, not a failure: the
            // search reads the closest approach off exactly these runs.
            return pilot.report(None);
        }
    }
    if pilot.outcome()? != FLYING {
        return pilot.report(None);
    }

    // Caught by the target. The conic it arrived on is a hyperbola about it, so
    // the periapsis below is the one number that says whether a retro burn can
    // end anywhere but the surface.
    let periapsis = pilot.periapsis()?;

    // Fall to periapsis under the same throttle, then burn. Burning from the
    // edge of the grip would spend the tank slowing down where the target's
    // gravity is doing it for free.
    while pilot.to_periapsis()? > 0.0 && pilot.outcome()? == FLYING {
        let wanted = pilot.warp_for_periapsis()?;
        pilot.nudge_warp(wanted)?;
        pilot.play(idle())?;
        pilot.check("periapsis about the target")?;
    }

    // Retrograde until the conic closes. `outcome` only reads a body on rails,
    // so the capture lands on the tick *after* the engine is cut — which is why
    // the release below is a step and not a return.
    while pilot.outcome()? == FLYING && pilot.escaping()? && pilot.fuel()? > 0.0 {
        pilot.play(press(RETRO))?;
        pilot.check("the capture burn")?;
    }
    pilot.play(idle())?;
    while pilot.outcome()? == FLYING {
        pilot.play(idle())?;
        pilot.check("the outcome the capture burn earned")?;
    }

    for _ in 0..TAIL {
        pilot.play(idle())?;
    }
    // The one press a fresh session never needs, and it is the last frame of the
    // stream: `restart` clears the world on the tick it lands, so a stream that
    // ends here says "the mission can be flown again" without paying for a
    // second bootstrap in the baseline.
    pilot.play(press(RESTART))?;
    pilot.report(Some(periapsis))
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
/// requires — a bare "hashes differ" over five thousand ticks is a wasted day.
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
    /// [`Body::index`] of whatever the ship's conic is about.
    pub primary: u32,
    /// [`FLYING`] until it is not.
    pub outcome: u32,
    /// Rows in the flight log.
    pub events: u32,
    /// Whether the engine is lit this tick — the regime, as a gate sees it.
    pub lit: bool,
    /// Sim ticks the world has aged. It is not the host tick and the gap is the
    /// milestone's subject: warp is an [`Epoch`] the *stream* steps, so a shell
    /// that put it in `TickClock` instead would replay a different number of
    /// ticks rather than the same ones over a different span.
    pub epoch: u64,
}

/// [`Progress`] after each of `frames`, through the same table
/// [`hash_sequence`] drives. `xtask reload --orbit` uses it to name the ticks
/// the escape, the intercept and the capture land on — the shape this run has,
/// and the one a shell replaying onto the wrong verb ids cannot fake.
///
/// # Errors
///
/// As [`hash_sequence`].
pub fn progress(entry: &Entry, frames: &[InputFrame]) -> Result<Vec<Progress>, SessionError> {
    let ship_q = Query::<&Ship>::new().map_err(SessionError::Alias)?;
    let event_q = Query::<&crate::Event>::new().map_err(SessionError::Alias)?;
    let rails_q = Query::<(&Ship, &Rails)>::new().map_err(SessionError::Alias)?;
    let bubble_q = Query::<(&Ship, &Bubble)>::new().map_err(SessionError::Alias)?;
    let body_q = Query::<&Body>::new().map_err(SessionError::Alias)?;
    let epoch_q = Query::<&Epoch>::new().map_err(SessionError::Alias)?;
    drive(entry, frames, |world| {
        let mut out = Progress {
            primary: u32::MAX,
            outcome: FLYING,
            events: 0,
            lit: false,
            epoch: 0,
        };
        world.each_ref(&ship_q, |_, ship: &Ship| out.outcome = ship.outcome);
        world.each_ref(&epoch_q, |_, epoch: &Epoch| out.epoch = epoch.elapsed);
        world.each_ref(&event_q, |_, _: &crate::Event| out.events += 1);
        let mut primary = Entity::NONE;
        world.each_ref(&rails_q, |_, (_, rails): (&Ship, &Rails)| {
            primary = rails.primary;
        });
        world.each_ref(&bubble_q, |_, (_, bubble): (&Ship, &Bubble)| {
            primary = bubble.primary;
            out.lit = true;
        });
        world.each_ref(&body_q, |entity, body: &Body| {
            if entity == primary {
                out.primary = body.index;
            }
        });
        Ok(out)
    })
}

/// Run `frames` through the declared table, reading something out of the world
/// after each tick. Ticks count from 0 — this game builds its own system, so
/// there is no scene to resume at.
fn drive<T>(
    entry: &Entry,
    frames: &[InputFrame],
    mut read: impl FnMut(&World) -> Result<T, SessionError>,
) -> Result<Vec<T>, SessionError> {
    let (table, mut world) = assemble(entry)?;
    let mut out = Vec::with_capacity(frames.len());
    let mut previous = idle();
    for (tick, frame) in frames.iter().enumerate() {
        let ctx = TickCtx {
            tick: tick as u64,
            tick_hz: HZ,
            reserved: 0,
            input: *frame,
            previous,
        };
        // SAFETY: the table is the entry's own, its entries live for the
        // process, and `ctx` outlives the call.
        unsafe { world.run_systems(&table, &ctx) }.map_err(SessionError::System)?;
        out.push(read(&world)?);
        previous = *frame;
    }
    Ok(out)
}

/// The rate every stream here is authored at, and the one the baseline means.
const HZ: u32 = 60;

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

// ------------------------------------------------------------ the pilot

/// A body's absolute state, as the pilot computes it.
///
/// The same walk `crate::bodies` does inside the game, restated because a
/// session drives a bare `World` and the game's helpers take a `GameWorld` —
/// demo 12's `turn_to` restates its aim for that reason. Velocity is carried
/// for the reason the game carries it: a frame change that moves only the
/// origin loses the primary's own motion.
#[derive(Clone, Copy)]
struct Station {
    entity: Entity,
    body: Body,
    position: sim::DVec3,
    velocity: sim::DVec3,
}

/// The ship as the pilot sees it: absolute position, velocity **against its
/// primary**, and which body that is.
#[derive(Clone, Copy)]
struct Sighting {
    position: sim::DVec3,
    velocity: sim::DVec3,
    primary: Station,
}

/// The closed loop: a world being driven, the frames chosen so far, and the
/// closest the ship has come to the target.
struct Pilot {
    world: World,
    table: SystemsTable,
    tick: u64,
    previous: InputFrame,
    frames: Vec<InputFrame>,
    approach: f64,
    /// Latched the tick the star takes the conic — the departure's whole
    /// product, and the only thing a targeting search can act on.
    transfer: Option<Rails>,
    /// The ship as it last existed. The stream's final frame is the restart,
    /// which clears the world, so a report read live would find nothing to
    /// report about a mission that had just succeeded.
    last: Option<Ship>,
}

impl Pilot {
    fn open(entry: &Entry) -> Result<Self, SessionError> {
        let (table, world) = assemble(entry)?;
        Ok(Pilot {
            world,
            table,
            tick: 0,
            previous: idle(),
            frames: Vec::new(),
            approach: f64::INFINITY,
            transfer: None,
            last: None,
        })
    }

    /// Run one tick under `frame`, record it, and sample the distance to the
    /// target. The sample is per **host** tick, which is exactly why the cruise
    /// throttles warp: at 1e6 the ship crosses the target's grip between two
    /// samples and this number would never see it.
    fn play(&mut self, frame: InputFrame) -> Result<(), SessionError> {
        let ctx = TickCtx {
            tick: self.tick,
            tick_hz: HZ,
            reserved: 0,
            input: frame,
            previous: self.previous,
        };
        // SAFETY: as `drive` — the entry's own table, `ctx` outlives the call.
        unsafe { self.world.run_systems(&self.table, &ctx) }.map_err(SessionError::System)?;
        self.tick += 1;
        self.previous = frame;
        self.frames.push(frame);
        if let Ok(ship) = self.one::<Ship>("a ship in the world") {
            self.last = Some(ship);
        }
        if let (Ok(sight), Ok(bodies)) = (self.sighting(), self.stations())
            && let Some(target) = bodies.iter().find(|held| held.body.index == OCHRE.index)
        {
            self.approach = self
                .approach
                .min((sight.position - target.position).length());
        }
        Ok(())
    }

    fn check(&self, waiting: &'static str) -> Result<(), SessionError> {
        if self.frames.len() >= MAX_TICKS {
            return Err(SessionError::Stuck(waiting));
        }
        Ok(())
    }

    fn report(mut self, periapsis: Option<f64>) -> Result<Flight, SessionError> {
        let ship = self.last.ok_or(SessionError::Stuck("a ship, ever"))?;
        Ok(Flight {
            frames: core::mem::take(&mut self.frames),
            outcome: ship.outcome,
            approach: self.approach,
            periapsis,
            spent: ship.spent,
            transfer: self.transfer,
        })
    }

    fn one<C: gg_ecs::Component + Copy>(&self, missing: &'static str) -> Result<C, SessionError> {
        let query = Query::<&C>::new().map_err(SessionError::Alias)?;
        let mut out = None;
        self.world.each_ref(&query, |_, held: &C| out = Some(*held));
        out.ok_or(SessionError::Stuck(missing))
    }

    fn outcome(&self) -> Result<u32, SessionError> {
        Ok(self.one::<Ship>("a ship in the world")?.outcome)
    }

    fn fuel(&self) -> Result<f64, SessionError> {
        Ok(self.one::<Ship>("a ship in the world")?.fuel)
    }

    /// Every body's absolute state at the world's current epoch — the pilot's
    /// copy of the walk `flight` does.
    fn stations(&self) -> Result<Vec<Station>, SessionError> {
        let epoch = self.one::<Epoch>("an epoch in the world")?;
        let query = Query::<&Body>::new().map_err(SessionError::Alias)?;
        let mut rows: Vec<(Entity, Body)> = Vec::new();
        self.world
            .each_ref(&query, |entity, body: &Body| rows.push((entity, *body)));
        Ok(rows
            .into_iter()
            .map(|(entity, body)| {
                let (position, velocity) = match self.world.get::<Rails>(entity) {
                    Some(rails) => rails
                        .orbit
                        .state_at(self.seconds(epoch.elapsed, rails.since)),
                    None => (sim::DVec3::ZERO, sim::DVec3::ZERO),
                };
                Station {
                    entity,
                    body,
                    position,
                    velocity,
                }
            })
            .collect())
    }

    /// Seconds between two epochs, at the rate this session is authored for.
    fn seconds(&self, elapsed: u64, since: u64) -> f64 {
        (elapsed.saturating_sub(since) as f64) / f64::from(HZ)
    }

    /// The ship, absolutely.
    fn sighting(&self) -> Result<Sighting, SessionError> {
        let epoch = self.one::<Epoch>("an epoch in the world")?;
        let bodies = self.stations()?;
        let query = Query::<&Ship>::new().map_err(SessionError::Alias)?;
        let mut ship = Entity::NONE;
        self.world
            .each_ref(&query, |entity, _: &Ship| ship = entity);
        let at = |entity: Entity| bodies.iter().copied().find(|held| held.entity == entity);

        if let Some(bubble) = self.world.get::<Bubble>(ship) {
            let primary = at(bubble.primary).ok_or(SessionError::Stuck("the ship's primary"))?;
            return Ok(Sighting {
                position: primary.position + bubble.position,
                velocity: bubble.velocity,
                primary,
            });
        }
        let rails = *self
            .world
            .get::<Rails>(ship)
            .ok_or(SessionError::Stuck("a ship in one of the two regimes"))?;
        let primary = at(rails.primary).ok_or(SessionError::Stuck("the ship's primary"))?;
        let (position, velocity) = rails
            .orbit
            .state_at(self.seconds(epoch.elapsed, rails.since));
        Ok(Sighting {
            position: primary.position + position,
            velocity,
            primary,
        })
    }

    /// Keep the conic the star took the ship over on. Silent on failure: the
    /// transfer is a report, and a flight that has none is one a search reads
    /// as "this plan never left".
    fn latch_transfer(&mut self) {
        let query = match Query::<(&Ship, &Rails)>::new() {
            Ok(query) => query,
            Err(_) => return,
        };
        let mut found = None;
        self.world
            .each_ref(&query, |_, (_, rails): (&Ship, &Rails)| {
                found = Some(*rails)
            });
        self.transfer = found;
    }

    /// [`Body::index`] of whatever the ship's conic is about.
    fn primary(&self) -> Result<u32, SessionError> {
        Ok(self.sighting()?.primary.body.index)
    }

    /// Periapsis of the conic the ship is on, metres from its primary's centre.
    fn periapsis(&self) -> Result<f64, SessionError> {
        let sight = self.sighting()?;
        let orbit = sim::Orbit::from_state(
            sight.position - sight.primary.position,
            sight.velocity,
            sight.primary.body.mu,
        )
        .ok_or(SessionError::Stuck("a conic about the primary"))?;
        Ok(orbit.semi_major * (1.0 - orbit.eccentricity))
    }

    /// Seconds to periapsis on the conic the ship is on, negative once it is
    /// past. Read off the conic's own mean anomaly, which is zero at periapsis
    /// on both branches.
    ///
    /// The estimate `distance / speed` cannot answer this: radial speed vanishes
    /// exactly where the remaining distance does, so it collapses to zero while
    /// hours of fall remain and the warp throttle below it crawls the last leg
    /// at 1x. That cost 8376 ticks of an 11662-tick mission before it was
    /// written this way.
    fn to_periapsis(&self) -> Result<f64, SessionError> {
        let sight = self.sighting()?;
        let orbit = sim::Orbit::from_state(
            sight.position - sight.primary.position,
            sight.velocity,
            sight.primary.body.mu,
        )
        .ok_or(SessionError::Stuck("a conic about the primary"))?;
        Ok(-orbit.mean_anomaly / orbit.mean_motion())
    }

    /// Whether the conic about the current primary is still open, or closed but
    /// grazing — the capture burn's own stopping condition, and the same test
    /// `outcome` will apply on the tick after the cut.
    fn escaping(&self) -> Result<bool, SessionError> {
        let sight = self.sighting()?;
        let Some(orbit) = sim::Orbit::from_state(
            sight.position - sight.primary.position,
            sight.velocity,
            sight.primary.body.mu,
        ) else {
            return Ok(true);
        };
        Ok(orbit.eccentricity >= 1.0
            || orbit.semi_major * (1.0 - orbit.eccentricity) <= sight.primary.body.radius)
    }

    /// The largest warp whose stride spends under a sixteenth of the time to
    /// the encounter. This is the rule §6 M38 item 10's residence finding asks
    /// of anyone who spends a warp factor as a tick stride.
    fn warp_for_encounter(&self) -> Result<u32, SessionError> {
        let sight = self.sighting()?;
        let bodies = self.stations()?;
        let Some(target) = bodies.iter().find(|held| held.body.index == OCHRE.index) else {
            return Ok(0);
        };
        let gap = (sight.position - target.position).length();
        // Closing speed in the *star's* frame, which is the frame both are
        // stated in while the ship is on the transfer.
        let closing = (sight.velocity - target.velocity).length().max(1.0);
        Ok(self.warp_under(gap / closing))
    }

    /// The same rule, aimed at the fall to periapsis rather than at the
    /// encounter — and off the conic's clock rather than off a distance.
    fn warp_for_periapsis(&self) -> Result<u32, SessionError> {
        Ok(self.warp_under(self.to_periapsis()?.max(0.0)))
    }

    /// The largest [`WARPS`] index whose stride is under `seconds / GUARD`.
    fn warp_under(&self, seconds: f64) -> u32 {
        let budget = seconds / GUARD;
        let mut step = 0;
        for (index, warp) in WARPS.iter().enumerate() {
            if (*warp as f64) / f64::from(HZ) <= budget {
                step = index as u32;
            }
        }
        step
    }

    /// Move the warp one step toward `wanted`, or idle. One step per tick is
    /// what the verb affords — this is a controller, not a jump.
    fn nudge_warp(&mut self, wanted: u32) -> Result<(), SessionError> {
        let step = self.one::<Epoch>("an epoch in the world")?.step;
        if step < wanted {
            self.play(press(WARP_UP))?;
            self.play(idle())?;
        } else if step > wanted {
            self.play(press(WARP_DOWN))?;
            self.play(idle())?;
        }
        Ok(())
    }

    /// Climb to `wanted` and stop there. Each step needs its own edge, so the
    /// release between taps is a tick of the stream and not a formality.
    fn set_warp(&mut self, wanted: u32) -> Result<(), SessionError> {
        while self.one::<Epoch>("an epoch in the world")?.step != wanted {
            self.nudge_warp(wanted)?;
            self.check("the warp the stage asked for")?;
        }
        Ok(())
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
