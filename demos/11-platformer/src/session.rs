//! The recorded session (§6 M20): **one full run, first spawn to the goal**,
//! and the hash sequence it produces.
//!
//! One script, three consumers — this crate's own tests drive it through the
//! declared systems table, `xtask replay --bless` encodes it into
//! `tests/replays/demo11-platformer.ggrp`, and `xtask reload --platformer`
//! replays that file through the real shell under three tiers. Authored here
//! for demo 10's reason: a script living in `xtask` cannot run under qemu.
//!
//! # A player, not a frame list — and not even a mirror
//!
//! Demo 10's bot mirrors the well with the game's own pure functions because
//! its choices need the future (where a piece lands). A platformer's choices
//! need only the present, so [`frames`] goes one better: it **drives the real
//! sim** through the declared table while it plays, reading the player and the
//! solids back out of the world each tick to decide the next frame. Nothing of
//! the physics is restated — the policy is jump-when-the-ground-runs-out, and
//! the integrator it answers to is the one being recorded. Re-author the level
//! and the same policy replays it or fails by name ([`SessionError::Stuck`]),
//! which is pull 2's doctrine (claims derive from the scene) applied to the
//! script itself.
//!
//! # The run the stream tells
//!
//! Restart first (the one press a fresh session would never need, so the verb
//! is in the recording), then a deliberate walk into the first pit — the death
//! keeps the clock and proves [`LOG`]'s middle milestone — then the run: over
//! the pit, up the step, along the ledges, into the goal. A session that
//! spawned adjacent to the goal would print "goal" too; it would not print
//! "fell" from the far side of a pit first.
//!
//! # The stream starts at the scene's tick
//!
//! The level is `scene.ggsave` and a replayed session must be handed it
//! explicitly (`--load` — record/replay never probe, §6 M20 item 3), which
//! resumes the world at the save's tick. The frames are therefore indexed from
//! [`opening_tick`], exactly as the shell's `--load --replay` seek expects
//! (§6 M18 item 7's idiom), and the baseline's tick column says the same.
//!
//! # The entry points are the caller's to supply
//!
//! Demo 10's session names the `gg_game!` exports in an `extern` block, which
//! ties every consumer to *linking* them — and two `gg_game!` crates cannot
//! share one binary (the symbols are fixed names; `xtask`'s manifest says so
//! over demo 10). So this driver takes an [`Entry`] instead: the demo's own
//! tests hand it this crate's exports, and `xtask` resolves the same four
//! symbols out of the **built dylib** — the host's own road to them (§4.2.2),
//! which is why the tables under test stay the declared ones either way.
//!
//! # The artifact is checked, not decoded
//!
//! Nothing here reads the `.ggrp` — `gg-input` is not a dependency a game
//! crate may take (§3). `xtask ci --push` byte-compares the checked-in file
//! against [`frames`], so a script edited without a `--bless` fails the push
//! tier by name.

use gg_ecs::boundary::{
    self, AbiInfo, ActionId, ComponentsTable, Eye, HostApiV1, InputFrame, Light, MAX_AXES,
    Renderable, SystemsTable, TickCtx, Widget,
};
use gg_ecs::hash::CanonicalHash;
use gg_ecs::{AliasError, Query, RegistryError, Save, SaveError, World};

use crate::{Goal, JUMP, MOVE, PLAYER_HALF_H, PLAYER_HALF_W, Player, RESTART, Run, Solid};

/// The four `gg_game!` exports, as pointers — resolved by whoever holds the
/// artifact. The demo's tests take them from this crate's own exports; `xtask`
/// takes them from the built dylib, exactly as the host does (§4.2.2).
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
/// "Fell" is the middle milestone for demo 10's reason: "goal" alone is also
/// what a session that spawned beside the goal would print, and this one has
/// to cross a pit it demonstrably fell into first. There is no "ready" — the
/// scene supplies the player, so `bootstrap` stands down in exactly the
/// sessions this stream replays (§6 M20 item 3).
pub const LOG: &[&str] = &[
    "platformer: restarted",
    "platformer: fell",
    "platformer: goal",
];

/// This session's name in `tests/replays/` — `.ggrp` beside `.hashes`.
pub const NAME: &str = "demo11-platformer";

/// Full deflection in the fixed-point axis encoding (§4.7's `AXIS_SCALE`).
const STICK: i32 = 1024;

/// Idle ticks after the goal. A stopped clock that is never ticked again
/// proves nothing about staying stopped.
const TAIL: usize = 30;

/// A ceiling on every drive loop below, not a target: one minute of sim.
/// Reaching it means the policy stopped finishing, which is a named error
/// rather than an unbounded generator.
const MAX_TICKS: usize = 3_600;

/// How far ahead of the leading face a rise must loom before the bot jumps at
/// it. Roughly the horizontal carry of the climb itself at full run speed.
const WALL_LOOK: f64 = 1.3;

/// How far past the leading face the bot probes for ground before treating
/// the edge as a gap.
const GAP_PROBE: f64 = 0.5;

/// How far below the feet a landing still counts as walkable ground rather
/// than the start of a drop worth jumping.
const DROP_TOLERANCE: f64 = 0.6;

/// Where the per-tick baseline lives.
///
/// The manifest dir walked up to the workspace root, demo 02's way: not an env
/// var and not a search, so a harness that cannot find the file fails loudly
/// rather than gating on nothing.
#[must_use]
pub fn baseline_path() -> std::path::PathBuf {
    workspace_root()
        .join("tests/replays")
        .join(format!("{NAME}.hashes"))
}

/// The checked-in level — the same file the shell's probe and `--load` open.
#[must_use]
pub fn scene_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("scene.ggsave")
}

fn workspace_root() -> std::path::PathBuf {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(std::path::Path::parent)
        .unwrap_or(manifest)
        .to_path_buf()
}

/// What can go wrong driving a world out of this crate's own tables and scene.
#[derive(Debug)]
pub enum SessionError {
    /// The components table this crate declares was refused.
    Adopt(RegistryError),
    /// The scene file could not be read.
    Scene(std::io::Error),
    /// The scene file did not decode or refused to load — a damaged file, or
    /// a name the build no longer declares.
    Save(SaveError),
    /// A system panicked behind the boundary shim.
    System(boundary::SystemPanic),
    /// The checked-in baseline does not parse, at this one-based line.
    Baseline(usize),
    /// A reader's query asked for the same component twice.
    Alias(AliasError),
    /// The supplied [`Entry`] describes a different build of the boundary.
    Abi,
    /// The policy ran out of ticks before the milestone it was walking toward
    /// — the level stopped affording the run this module records.
    Stuck(&'static str),
}

impl core::fmt::Display for SessionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SessionError::Adopt(e) => write!(f, "demo 11's own components were refused: {e}"),
            SessionError::Scene(e) => write!(f, "the scene at {:?} unreadable: {e}", scene_path()),
            SessionError::Save(e) => write!(f, "the scene did not open: {e}"),
            SessionError::System(e) => write!(f, "{} panicked: {}", e.system, e.message),
            SessionError::Baseline(line) => write!(f, "malformed baseline at line {line}"),
            SessionError::Alias(e) => write!(f, "a reader's query does not hold: {e}"),
            SessionError::Abi => write!(
                f,
                "one build, one description — the entry points came \
                                            from a different boundary"
            ),
            SessionError::Stuck(waiting) => write!(
                f,
                "the policy spent {MAX_TICKS} ticks without reaching: {waiting}"
            ),
        }
    }
}

impl core::error::Error for SessionError {}

/// The scripted run as an input stream, starting at [`opening_tick`].
///
/// # Errors
///
/// If the scene cannot be opened or the level stops affording the run — a pit
/// the policy cannot die in, a climb it cannot make ([`SessionError::Stuck`]).
pub fn frames(entry: &Entry) -> Result<Vec<InputFrame>, SessionError> {
    let mut bot = Bot::open(entry)?;

    // The one press a fresh session never needs, so the verb reaches the
    // recording: `R` resets the run clock and logs the first milestone.
    bot.play(press(RESTART))?;
    bot.play(idle())?;

    // Walk into the first pit, hands off the stick once airborne so the fall
    // is straight down: the death is LOG's middle milestone, and it proves a
    // death keeps the clock in the replayed run too.
    while bot.deaths()? < 1 {
        let frame = if bot.player()?.grounded != 0 {
            steer(1)
        } else {
            idle()
        };
        bot.play(frame)?;
        bot.check("the first pit")?;
    }

    // The run: over the pit, up the step, along the ledges, into the goal.
    while !bot.finished()? {
        let frame = bot.choose()?;
        bot.play(frame)?;
        bot.check("the goal")?;
    }

    for _ in 0..TAIL {
        bot.play(idle())?;
    }
    Ok(bot.frames)
}

/// The same policy patrolling the start pad for as long as the caller asks —
/// grounded stretches with a full jump every cycle, never near the pit.
///
/// [`frames`] ends; a reload gate needs a run still in progress when a rebuilt
/// dylib lands under it, with the player *on the ground* at the swap so a feel
/// constant is latent when it arrives and bites exactly at the next arc
/// (§6 M20 pull 4). Open-loop on purpose: nothing here needs the world, so the
/// stream is a pure function of `ticks` and the cycle constants below.
#[must_use]
pub fn endless(ticks: usize) -> Vec<InputFrame> {
    let mut out = Vec::with_capacity(ticks);
    while out.len() < ticks {
        let at = out.len();
        out.push(match at {
            // The save opens mid-fall; land and settle before the first cycle.
            0..=59 => idle(),
            _ => match (at - 60) % CYCLE {
                0..=59 => steer(1),
                60..=74 => idle(),
                75..=134 => steer(-1),
                135..=149 => idle(),
                // A held jump: press once, hold through the rise, release
                // past the apex, land inside the same cycle's idle tail.
                150..=175 => press(JUMP),
                _ => idle(),
            },
        });
    }
    out.truncate(ticks);
    out
}

/// One patrol cycle of [`endless`], ticks. Walk right, stop, walk back, stop,
/// one held jump, and enough idle for the landing to be inside the cycle.
pub const CYCLE: usize = 256;

/// Stream offsets (0-based) of [`endless`]'s grounded walking stretches, for
/// a caller choosing where a mid-run swap should land.
#[must_use]
pub fn walking(at: usize) -> bool {
    at >= 60 && matches!((at - 60) % CYCLE, 0..=59 | 75..=134)
}

// ------------------------------------------------------------ the hash side

/// The sequence as text: `tick hash` per line, hex, newline-terminated.
/// Demo 02's format exactly, demo 10's reasons verbatim.
#[must_use]
pub fn encode_baseline(sequence: &[(u64, CanonicalHash)]) -> String {
    let mut out = String::with_capacity(sequence.len() * 40);
    for (tick, hash) in sequence {
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
/// requires.
#[must_use]
pub fn divergence(sequence: &[(u64, CanonicalHash)], baseline: &[(u64, u128)]) -> Option<String> {
    for ((tick, left), (_, right)) in sequence.iter().zip(baseline) {
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

/// The tick the scene resumes at — where the stream's indexing starts.
///
/// # Errors
///
/// If the scene cannot be read or decoded.
pub fn opening_tick() -> Result<u64, SessionError> {
    let bytes = std::fs::read(scene_path()).map_err(SessionError::Scene)?;
    Ok(Save::decode(&bytes).map_err(SessionError::Save)?.tick())
}

/// The canonical hash after each of `frames`, driven through the *declared*
/// systems table over the loaded scene. Values are **not** a shell run's —
/// demo 10's `hash_sequence` says why (the shell's world also carries `gg_ui`
/// hit state) — so the gate ties the two by shape, not value.
///
/// # Errors
///
/// If the scene cannot be opened, the table is refused, or a system panics.
pub fn hash_sequence(
    entry: &Entry,
    frames: &[InputFrame],
) -> Result<Vec<(u64, CanonicalHash)>, SessionError> {
    drive(entry, frames, |tick, world| {
        Ok((tick, world.canonical_hash()))
    })
}

/// What a gate asks about a tick, read out of the world the systems just ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    /// Standing on something at the end of the tick. The feel gate's whole
    /// question: a grounded player is one a gravity retune is *latent* for.
    pub grounded: bool,
    /// Falls so far, all run.
    pub deaths: u32,
    /// The goal has been reached; the clock is stopped.
    pub finished: bool,
}

/// [`Progress`] after each of `frames`, through the same table
/// [`hash_sequence`] drives. `xtask reload --feel` uses it to say the player
/// was *on the ground* when the swap landed — a mid-air swap would prove
/// nothing about a latent constant — and to name the tick the arc next leaves
/// the ground, which is exactly where the runs must part.
///
/// # Errors
///
/// As [`hash_sequence`].
pub fn progress(
    entry: &Entry,
    frames: &[InputFrame],
) -> Result<Vec<(u64, Progress)>, SessionError> {
    // Two queries: the player and the run clock live on different entities.
    let player_q = Query::<&Player>::new().map_err(SessionError::Alias)?;
    let run_q = Query::<&Run>::new().map_err(SessionError::Alias)?;
    drive(entry, frames, |tick, world| {
        let mut out = Progress {
            grounded: false,
            deaths: 0,
            finished: false,
        };
        world.each_ref(&player_q, |_, player: &Player| {
            out.grounded = player.grounded != 0;
        });
        world.each_ref(&run_q, |_, run: &Run| {
            out.deaths = run.deaths;
            out.finished = run.finished_at != 0;
        });
        Ok((tick, out))
    })
}

/// Run `frames` through the declared table over the loaded scene, reading
/// something out of the world after each tick. Ticks are absolute — the
/// scene's own tick first, as the shell's `--load --replay` runs them.
fn drive<T>(
    entry: &Entry,
    frames: &[InputFrame],
    mut read: impl FnMut(u64, &World) -> Result<T, SessionError>,
) -> Result<Vec<T>, SessionError> {
    let (table, mut world, opening) = assemble(entry)?;
    let mut out = Vec::with_capacity(frames.len());
    let mut previous = idle();
    for (offset, frame) in frames.iter().enumerate() {
        let tick = opening + offset as u64;
        let ctx = TickCtx::detached(tick, 60, *frame, previous);
        // SAFETY: the table is this binary's own, its entries live for the
        // process, and `ctx` outlives the call.
        unsafe { world.run_systems(&table, &ctx) }.map_err(SessionError::System)?;
        out.push(read(tick, &world)?);
        previous = *frame;
    }
    Ok(out)
}

/// A fresh world holding the checked-in scene: the host's five protocol
/// registrations, the entry's table adopted, the save loaded clean — the
/// calls `gg_runtime::app` makes, in its order.
fn assemble(entry: &Entry) -> Result<(SystemsTable, World, u64), SessionError> {
    // SAFETY: the caller's contract on [`Entry`] — the four pointers are the
    // `gg_game!` exports of an artifact built from this same source, alive for
    // the process (a dylib is never unloaded, §4.2.2); `host_api()` returns a
    // `&'static` table. The ABI check below is the boundary's own refusal.
    let (table, mut world) = unsafe {
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
        world.register::<Light>().map_err(SessionError::Adopt)?;
        world.register::<Widget>().map_err(SessionError::Adopt)?;
        world
            .adopt(&(entry.components)())
            .map_err(SessionError::Adopt)?;
        ((entry.systems)(), world)
    };
    let bytes = std::fs::read(scene_path()).map_err(SessionError::Scene)?;
    let save = Save::decode(&bytes).map_err(SessionError::Save)?;
    world.load(&save).map_err(SessionError::Save)?;
    Ok((table, world, save.tick()))
}

// ------------------------------------------------------------ the policy

/// The closed loop: a world being driven, the frames chosen so far, and the
/// solids read once (the level does not move).
struct Bot {
    world: World,
    table: SystemsTable,
    tick: u64,
    previous: InputFrame,
    frames: Vec<InputFrame>,
    solids: Vec<(f64, f64, f64, f64)>,
    goal_x: f64,
}

impl Bot {
    fn open(entry: &Entry) -> Result<Self, SessionError> {
        let (table, world, tick) = assemble(entry)?;
        let solids_q = Query::<(&Solid, &Renderable)>::new().map_err(SessionError::Alias)?;
        let mut solids = Vec::new();
        world.each_ref(&solids_q, |_, (_, r): (&Solid, &Renderable)| {
            solids.push((
                r.position.x,
                r.position.y,
                f64::from(r.half_extent.x),
                f64::from(r.half_extent.y),
            ));
        });
        let goal_q = Query::<(&Goal, &Renderable)>::new().map_err(SessionError::Alias)?;
        let mut goal_x = 0.0;
        world.each_ref(&goal_q, |_, (_, r): (&Goal, &Renderable)| {
            goal_x = r.position.x;
        });
        Ok(Bot {
            world,
            table,
            tick,
            previous: idle(),
            frames: Vec::new(),
            solids,
            goal_x,
        })
    }

    /// Run one tick under `frame` and record it.
    fn play(&mut self, frame: InputFrame) -> Result<(), SessionError> {
        let ctx = TickCtx::detached(self.tick, 60, frame, self.previous);
        // SAFETY: as `drive` — this binary's own table, `ctx` outlives the call.
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

    fn player(&self) -> Result<Player, SessionError> {
        let query = Query::<&Player>::new().map_err(SessionError::Alias)?;
        let mut out = None;
        self.world.each_ref(&query, |_, p: &Player| out = Some(*p));
        out.ok_or(SessionError::Stuck("a player in the world"))
    }

    fn run(&self) -> Result<Run, SessionError> {
        let query = Query::<&Run>::new().map_err(SessionError::Alias)?;
        let mut out = None;
        self.world.each_ref(&query, |_, r: &Run| out = Some(*r));
        out.ok_or(SessionError::Stuck("a run clock in the world"))
    }

    fn deaths(&self) -> Result<u32, SessionError> {
        Ok(self.run()?.deaths)
    }

    fn finished(&self) -> Result<bool, SessionError> {
        Ok(self.run()?.finished_at != 0)
    }

    /// The whole policy: run at the goal; jump when the ground runs out or a
    /// rise looms; hold the jump through the rise for full height.
    fn choose(&self) -> Result<InputFrame, SessionError> {
        let p = self.player()?;
        let dir = if self.goal_x >= p.position.x { 1 } else { -1 };
        let mut frame = steer(dir);

        if p.grounded != 0 {
            let feet = p.position.y - f64::from(PLAYER_HALF_H);
            let front = p.position.x + f64::from(dir) * f64::from(PLAYER_HALF_W);
            // A rise ahead: a face in front within WALL_LOOK whose top is
            // above the feet — jump early enough that the apex clears it.
            let wall = self.solids.iter().any(|(sx, sy, shw, shh)| {
                let face = sx - f64::from(dir) * shw;
                let ahead = f64::from(dir) * (face - front);
                sy + shh > feet + 0.05 && (-0.1..=WALL_LOOK).contains(&ahead)
            });
            // The ground running out: nothing walkable under the point just
            // past the leading edge.
            let probe = front + f64::from(dir) * GAP_PROBE;
            let ground = self.solids.iter().any(|(sx, sy, shw, shh)| {
                let top = sy + shh;
                (probe - sx).abs() < shw + f64::from(PLAYER_HALF_W)
                    && top <= feet + 0.05
                    && top >= feet - DROP_TOLERANCE
            });
            if wall || !ground {
                frame.buttons |= 1 << JUMP.index();
            }
        } else if p.velocity.y > 0.0 {
            // Hold through the rise: this session records full jumps; the
            // released-rise gravity is the tests' to pin.
            frame.buttons |= 1 << JUMP.index();
        }
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

fn steer(direction: i32) -> InputFrame {
    let mut frame = idle();
    frame.axes[MOVE.index()] = direction * STICK;
    frame
}

/// The frame with one button down. A press held across ticks is `press` again
/// (`just_pressed` is the edge, which `TickCtx::previous` supplies).
fn press(action: ActionId) -> InputFrame {
    let mut frame = idle();
    frame.buttons = 1 << action.index();
    frame
}
