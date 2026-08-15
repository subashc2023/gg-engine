//! The recorded session (§6 M18): **one full game, first spawn to top-out**,
//! and the hash sequence it produces.
//!
//! One script, three consumers — this crate's own test drives it through the
//! declared systems table, `xtask replay --bless` encodes it into
//! `tests/replays/demo10-tetris.ggrp`, and `xtask reload --tetris` replays that
//! file through the real shell under three tiers. Authored here for demo 07's
//! reason, and demo 02's: a script living in `xtask` cannot run under qemu.
//!
//! # A player, not a frame list
//!
//! A fixed list of button presses cannot clear a line — where a piece *should*
//! go depends on which piece the bag drew, and the bag is world state (§6 M18
//! item 1). So [`frames`] plays: it mirrors the well and emits the taps that
//! carry out its answer. The mirror moves the piece with the game's own
//! [`rotate`], [`landing_row`], [`lock_piece`] and [`clear_rows`], so what is
//! restated here is the tap-to-cell rule and the lock-and-advance order and
//! nothing else — and both are pinned by [`hash_sequence`]'s callers, which run
//! these frames through the *real* systems and assert the outcome the mirror
//! predicted.
//!
//! # Integer scoring, because the frames must not depend on the machine
//!
//! [`place`]'s weights are `i32` and its scan order is fixed. Float scoring
//! would put a rounding difference between two architectures *upstream of the
//! input stream*, which is the one determinism break the canonical hash cannot
//! attribute: aarch64 would go red naming a tick, and the cause would be a bot
//! that chose a different column rather than a sim that computed a different
//! answer. §1.3 covers the sim; this keeps the script out of its way.
//!
//! # Two phases, because a good player does not top out
//!
//! Placing well clears lines and reaches level 2; it also survives, and a
//! session that never ends is not a full game. So the bot plays until
//! [`BUILD_LINES`] rows are gone and then stops choosing — every remaining piece
//! is dropped where it spawned, which stacks four columns and tops out quickly.
//!
//! # The artifact is checked, not decoded
//!
//! Nothing here reads the `.ggrp`. Demo 02's gate decodes its checked-in replay
//! because that file is the artifact under test; here the artifact under test is
//! the *shell's* replay of it, and `gg-input` is not a dependency a game crate
//! may take (§3). The staleness that would open is closed from the other side —
//! `xtask ci --push` byte-compares the checked-in file against [`frames`], so a
//! script edited without a `--bless` fails the push tier by name.

use gg_ecs::boundary::{self, AXIS_SCALE, ActionId, InputFrame, MAX_AXES};
use gg_ecs::hash::CanonicalHash;
use gg_ecs::{AliasError, RegistryError};
// Only what drives a world needs the entry points, and so the feature.
#[cfg(feature = "game")]
use gg_ecs::boundary::{AbiInfo, ComponentsTable, HostApiV1, SystemsTable, TickCtx};
#[cfg(feature = "game")]
use gg_ecs::{Query, World};

use crate::{
    HARD_DROP, HEIGHT, HOLD, LEFT, Piece, Play, RESTART, RIGHT, ROTATE_CW, Rules, SEED, WIDTH,
    Well, clear_rows, collides, draw, gravity_for, landing_row, lock_piece, new_play, rotate,
    spawn_piece,
};

/// What the shell's log must say, in order, when this session is replayed.
///
/// The middle two rows are why the game logs a level at all: "ready" and
/// "topped out" are also what a session that spawned and immediately died would
/// print, and a gate that cannot tell those apart is checking that the game
/// starts rather than that it was played.
pub const LOG: &[&str] = &[
    "tetris: ready",
    "tetris: level 2",
    "tetris: level 3",
    "tetris: topped out",
];

/// This session's name in `tests/replays/` — `.ggrp` beside `.hashes`.
pub const NAME: &str = "demo10-tetris";

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

/// Rows the bot clears before it stops trying to survive. Two whole levels'
/// worth of `Rules::lines_per_level`, so the session crosses the level boundary
/// twice — once is a threshold that could be off by one and still fire.
pub const BUILD_LINES: u32 = 24;

/// Idle ticks after the top-out. A dead board that is never ticked again proves
/// nothing about staying dead.
const TAIL: usize = 30;

/// A ceiling on the loop below, not a target. Reaching it means the bot stopped
/// topping out, which its callers assert against rather than tolerate.
const MAX_PIECES: usize = 400;

/// The scripted game as an input stream.
#[must_use]
pub fn frames() -> Vec<InputFrame> {
    let mut out = Vec::new();
    // The game opens on the title screen (§6 M19); `RESTART` is the keyboard
    // spelling of PLAY there. The board underneath is `bootstrap`'s, untouched
    // by the press, so the mirror below stays the opening position.
    tap(&mut out, RESTART);
    // `bootstrap`'s opening position, mirrored: the same seed and the same first
    // draw, or the bot would be playing a different game from the one it records.
    let mut well = Well {
        cells: [[0; WIDTH]; HEIGHT],
    };
    let mut play = new_play(SEED);
    let mut piece = spawn_piece(draw(&mut play));
    let mut lines = 0;

    for placed in 0..MAX_PIECES {
        // One hold, on the first piece: the swap is a path this session would
        // otherwise never take, and `CUE_HOLD` would never sound.
        if placed == 0 {
            tap(&mut out, HOLD);
            let taken = play.next;
            play.next = draw(&mut play);
            play.hold = piece.kind;
            piece = spawn_piece(taken);
        }

        // Past `BUILD_LINES` the bot stops choosing: every piece is dropped
        // where it spawned, which stacks four columns and tops out quickly.
        let choosing = lines < BUILD_LINES;
        match carry(
            &mut out,
            &mut well,
            &mut play,
            &mut piece,
            choosing,
            level(lines),
        ) {
            Some(cleared) => lines += cleared,
            None => break,
        }
    }

    for _ in 0..TAIL {
        out.push(idle());
    }
    out
}

/// The same bot, never giving up, for as long as the caller asks.
///
/// [`frames`] stops choosing so it can top out; a reload gate needs the
/// opposite — a game still in progress with a stack on the board when a rebuilt
/// dylib lands under it (§6 M18). Playing well is what keeps both true, and the
/// generated stream is trimmed to exactly `ticks` so a caller's swap tick is a
/// position in a known-length run rather than a guess.
///
/// A well-played game still ends eventually, so this mirrors the *restart* too —
/// the one path [`frames`] never takes. The reseed is `Play::rng`'s own next
/// draw, exactly as `step` does it, or the second game would diverge from the
/// first tick of it.
#[must_use]
pub fn endless(ticks: usize) -> Vec<InputFrame> {
    let mut out = Vec::with_capacity(ticks + 32);
    // Off the title screen first, as [`frames`] does (§6 M19).
    tap(&mut out, RESTART);
    let mut well = Well {
        cells: [[0; WIDTH]; HEIGHT],
    };
    let mut play = new_play(SEED);
    let mut piece = spawn_piece(draw(&mut play));
    // The game's own, tracked here because the mirror runs no scoring: `carry`
    // needs the level to know how far a piece falls while it is being placed.
    let mut lines = 0;

    while out.len() < ticks {
        match carry(
            &mut out,
            &mut well,
            &mut play,
            &mut piece,
            true,
            level(lines),
        ) {
            Some(cleared) => lines += cleared,
            None => {
                tap(&mut out, RESTART);
                let seed = play.rng.next_u64();
                well = Well {
                    cells: [[0; WIDTH]; HEIGHT],
                };
                play = new_play(seed);
                piece = spawn_piece(draw(&mut play));
                lines = 0;
            }
        }
    }
    out.truncate(ticks);
    out
}

/// Carry one piece from spawn to lock: the taps that put it where it belongs,
/// and the mirror's own lock-and-advance. `None` means the next piece could not
/// spawn — a top-out, on the tick after the taps this returned.
///
/// `choosing` off drops it where it spawned, unrotated: no taps at all, so a
/// piece costs two ticks and cannot outrun the lock delay on a tall stack.
fn carry(
    out: &mut Vec<InputFrame>,
    well: &mut Well,
    play: &mut Play,
    piece: &mut Piece,
    choosing: bool,
    level: u32,
) -> Option<u32> {
    let (rot, col) = match choosing {
        // A plan the row would change is not a plan this mirror can carry
        // (§6 M51). Dropping where it spawned is the fallback because a hard
        // drop is row-blind by construction: `landing_row` scans down from
        // wherever the piece is and reaches the same cell either way.
        true => {
            let want = place(well, piece);
            if row_blind(well, piece, want, level) {
                want
            } else {
                (0, piece.col)
            }
        }
        false => (0, piece.col),
    };
    for _ in 0..rot {
        tap(out, ROTATE_CW);
        // The real kick, so a rotation that shifts the piece shifts the column
        // the loop below is counting from.
        let _ = rotate(well, piece, true);
    }
    // One cell per tap: a tap is `just_pressed`, which moves once and does not
    // charge DAS. The sim clamps a move into a wall and so does this.
    while piece.col != col {
        let step = if col < piece.col { -1 } else { 1 };
        tap(out, if step < 0 { LEFT } else { RIGHT });
        piece.col += step;
        if collides(well, piece) {
            piece.col -= step;
            break;
        }
    }

    tap(out, HARD_DROP);
    piece.row = landing_row(well, piece);
    lock_piece(well, piece);
    let cleared = clear_rows(well);
    let kind = play.next;
    play.next = draw(play);
    *piece = spawn_piece(kind);
    (!collides(well, piece)).then_some(cleared)
}

/// Whether the taps that carry `piece` to `(rot, col)` mean the same thing at
/// the row the game's piece will actually be on (§6 M51).
///
/// **The one assumption the mirror makes, checked rather than assumed.** This
/// bot positions at the spawn row: it never models the fall, which is what
/// keeps [`carry`] free of the gravity and lock-delay rules. The game's piece is
/// wherever gravity has taken it by the time each tap lands — the same cell
/// while the piece is clear of the stack, and a *different* one once the stack
/// has risen into the distance it falls while being positioned. Nothing said so,
/// and the failure is silent in the worst way: the boards part, the game tops
/// out on a board the script does not have, the script plays on into a dead
/// world and every tick after that is input nobody read. It took a soak's
/// liveness column to see it at all — 35,624 ticks of agreement and then a
/// hundred thousand of a frozen game reading as the flattest possible result.
///
/// So the same taps are walked twice on copies, at the mirror's row and at the
/// game's, and a placement is taken only if the two never disagree. What that
/// buys is not a cleverer bot: it is a bot whose remaining choices narrow as the
/// level rises, exactly as a player's do, until the only move left is to drop
/// where the piece spawned and top out — which is a game ending, not a harness
/// breaking, and [`endless`] answers it with the restart.
fn row_blind(well: &Well, piece: &Piece, (rot, col): (u8, i32), level: u32) -> bool {
    let gravity = gravity_for(&Rules::DEFAULT, level).max(1);
    let (mut flat, mut real) = (*piece, *piece);
    let mut ticks = 0u32;
    // One tap is two ticks — a press and the release that makes the next one an
    // edge — and the fall is charged against both.
    let fall = |real: &mut Piece, ticks: &mut u32| {
        *ticks += 2;
        real.row = piece.row + (*ticks / gravity) as i32;
    };
    for _ in 0..rot {
        let _ = rotate(well, &mut flat, true);
        let _ = rotate(well, &mut real, true);
        fall(&mut real, &mut ticks);
        // A kick resolved against a different neighbourhood, or a piece the fall
        // has already buried: either way the row mattered.
        if flat.col != real.col || flat.rot != real.rot || collides(well, &real) {
            return false;
        }
    }
    while flat.col != col {
        let step = if col < flat.col { -1 } else { 1 };
        flat.col += step;
        real.col += step;
        fall(&mut real, &mut ticks);
        if collides(well, &flat) != collides(well, &real) {
            return false;
        }
        if collides(well, &flat) {
            break;
        }
    }
    true
}

/// Where to put `piece`: the `(rotation, column)` a scoring pass likes best,
/// over every rotation and every column it could occupy.
///
/// Ties go to the first in scan order, which is why the order is written out
/// rather than left to a sort — a stable answer is the whole requirement.
fn place(well: &Well, piece: &Piece) -> (u8, i32) {
    let mut best = (0, piece.col, i32::MIN);
    for rot in 0..4 {
        for col in -2..WIDTH as i32 {
            let probe = Piece { col, rot, ..*piece };
            if collides(well, &probe) {
                continue;
            }
            let mut after = *well;
            let mut landed = probe;
            landed.row = landing_row(well, &landed);
            lock_piece(&mut after, &landed);
            let cleared = clear_rows(&mut after);
            let score = score(&after, cleared);
            if score > best.2 {
                best = (rot, col, score);
            }
        }
    }
    (best.0, best.1)
}

/// How good a board is, in whole numbers: rows gone are worth a lot, and height,
/// buried holes and a jagged surface each cost.
fn score(well: &Well, cleared: u32) -> i32 {
    let mut heights = [0; WIDTH];
    let mut holes = 0;
    for (col, height) in heights.iter_mut().enumerate() {
        let mut top = None;
        for row in 0..HEIGHT {
            if well.cells[row][col] != 0 {
                top.get_or_insert(row);
            } else if top.is_some() {
                // Empty, with something above it: a cell no piece can reach.
                holes += 1;
            }
        }
        *height = top.map_or(0, |row| (HEIGHT - row) as i32);
    }
    let aggregate: i32 = heights.iter().sum();
    let bumpiness: i32 = heights.windows(2).map(|w| (w[0] - w[1]).abs()).sum();
    100 * cleared as i32 - 5 * aggregate - 8 * holes - 3 * bumpiness
}

/// The level `lines` cleared puts a game on — `step`'s own rule, restated here
/// because the mirror runs no scoring and [`row_blind`] needs the gravity that
/// depends on it.
fn level(lines: u32) -> u32 {
    lines / Rules::DEFAULT.lines_per_level.max(1) + 1
}

/// Press for one tick, release for the next — one `just_pressed`, one cell.
fn tap(out: &mut Vec<InputFrame>, action: ActionId) {
    out.push(InputFrame {
        buttons: 1 << action.index(),
        axes: [0; MAX_AXES],
    });
    out.push(idle());
}

fn idle() -> InputFrame {
    InputFrame {
        buttons: 0,
        axes: [0; MAX_AXES],
    }
}

// ------------------------------------------------------------- the menu tour
//
// A second script over the same game, because the first one never touches a
// menu (§6 M19's residual, closed at M44). It is *pointer* input where the game
// session is keyboard input, which is the whole point: a click has to survive
// the canvas fit, the pointer's fixed-point integration and one frame of lag
// before it lands on the row it was aimed at, and none of that is exercised by
// a key. It is also the only recorded session in the tree that **ends itself** —
// EXIT sets `Prefs::close`, so the replay's last tick is the game's decision
// rather than the harness running out of frames.

/// `ui_click`'s id. Named here and nowhere else in the demo: the *game* never
/// reads it — the host routes the click and hands the result back as a `state`
/// bit (§4.9) — but a script has to press the button a player presses.
const UI_CLICK: ActionId = ActionId::new(9);

/// This session's name in `tests/replays/`.
pub const MENU_NAME: &str = "demo10-menu";

/// What the shell's log must say, in order, when the tour is replayed.
///
/// There is no `.hashes` beside this one, and the absence is the structure
/// rather than an omission: a click is a `Widget::state` bit the *host* writes,
/// so an in-process run of these frames registers nothing. The tour is graded
/// against the other tiers running it, demo 07's way (`xtask reload --ui`).
///
/// Every settings row is here because each writes a *different* component —
/// `Options`, `Rules`, `Prefs` — and a tour that only proved one row worked
/// would be checking that the click machinery runs, not that the menu does.
pub const MENU_LOG: &[&str] = &[
    "tetris: ready",
    "tetris: ghost off",
    "tetris: volume changed",
    "tetris: cursor changed",
    "tetris: speed changed",
    "tetris: theme changed",
    "tetris: paused",
    "tetris: resumed",
    "tetris: quit",
    "tetris: closing",
];

/// The scripted tour: every settings row from the title, then a game, then the
/// same screen reached from *pause* — the path that exercises `Screen::from` —
/// then out through QUIT and EXIT.
///
/// Every position is the centre of a row in [`crate::MENU`], so moving a button
/// moves the click rather than breaking the gate (demo 07's rule).
#[must_use]
pub fn menu_frames() -> Vec<InputFrame> {
    let mut out = Vec::new();
    let mut at = (0i32, 0i32);

    // From the title: open settings and touch all five rows. VOLUME twice, so
    // the row that steps through five values is proven to step rather than to
    // toggle.
    for (which, clicks) in [
        (crate::M_TITLE_SETTINGS, 1),
        (crate::M_GHOST, 1),
        (crate::M_VOLUME, 2),
        (crate::M_CURSOR, 1),
        (crate::M_SPEED, 1),
        (crate::M_THEME, 1),
        (crate::M_DONE, 1),
    ] {
        click(&mut out, &mut at, which, clicks);
    }

    // Play a little, with the keyboard, so the pause lands on a live board.
    tap(&mut out, RESTART);
    for _ in 0..40 {
        out.push(idle());
    }

    // Pause with the key, reach settings from *there*, and leave: `Screen::from`
    // is the field that says where DONE returns to, and the title path above
    // cannot tell a working one from a hardcoded `SCREEN_TITLE`.
    tap(&mut out, crate::PAUSE);
    for _ in 0..4 {
        out.push(idle());
    }
    click(&mut out, &mut at, crate::M_PAUSE_SETTINGS, 1);
    click(&mut out, &mut at, crate::M_DONE, 1);
    // Escape resumes, and escape pauses again — the verb M44 took off the shell.
    tap(&mut out, crate::BACK);
    for _ in 0..8 {
        out.push(idle());
    }
    tap(&mut out, crate::BACK);
    for _ in 0..8 {
        out.push(idle());
    }
    // Abandon the run, then close the session from the title.
    click(&mut out, &mut at, crate::M_QUIT, 1);
    click(&mut out, &mut at, crate::M_TITLE_EXIT, 1);
    // The shell needs a tick to see `Prefs::close`; past that the frames are
    // spare, and a replay that outlives its own ending is the point.
    for _ in 0..10 {
        out.push(idle());
    }
    out
}

/// Glide onto row `which`, settle, and press `ui_click` `clicks` times.
fn click(out: &mut Vec<InputFrame>, at: &mut (i32, i32), which: u8, clicks: u32) {
    let rect = crate::MENU[which as usize].rect;
    glide(
        out,
        at,
        (rect[0] + rect[2] / 2.0, rect[1] + rect[3] / 2.0),
        20,
    );
    // Settle: hover resolves against the geometry the *previous* frame declared
    // (§4.9's lag), and a press that finds no hovered widget captures nothing.
    for _ in 0..3 {
        out.push(idle());
    }
    for _ in 0..clicks {
        for _ in 0..2 {
            out.push(InputFrame {
                buttons: 1 << UI_CLICK.index(),
                axes: [0; MAX_AXES],
            });
        }
        // Released, then long enough for `menu` to read the bit and `hud` to
        // redraw what it changed.
        for _ in 0..6 {
            out.push(idle());
        }
    }
}

/// Move the pointer from `at` to `to` over `ticks`, in fixed point.
///
/// Divided against the ticks *remaining* rather than by a constant step, so the
/// last frame carries the remainder and the pointer lands exactly on the target
/// — a click one unit short of a button is a gate that fails on a rounding
/// change and names nothing.
fn glide(out: &mut Vec<InputFrame>, at: &mut (i32, i32), to: (f32, f32), ticks: i32) {
    let target = (
        (to.0 * AXIS_SCALE as f32) as i32,
        (to.1 * AXIS_SCALE as f32) as i32,
    );
    for step in 0..ticks {
        let left = ticks - step;
        let motion = ((target.0 - at.0) / left, (target.1 - at.1) / left);
        at.0 += motion.0;
        at.1 += motion.1;
        let mut axes = [0; MAX_AXES];
        axes[0] = motion.0;
        axes[1] = motion.1;
        out.push(InputFrame { buttons: 0, axes });
    }
}

// ------------------------------------------------------------ the hash side
//
// Behind the `game` feature, because everything below drives the world through
// the symbols `gg_game!` exports and a build without it has none: `gg-golden`
// takes this crate with `default-features = false` for its pure layout
// functions, and an ungated extern block leaves its cdylib unlinkable.

/// What can go wrong driving a world that this crate built out of its own
/// tables: nothing, and it is still not a place to `unwrap`.
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
}

impl core::fmt::Display for SessionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SessionError::Adopt(e) => write!(f, "demo 10's own components were refused: {e}"),
            SessionError::System(e) => write!(f, "{} panicked: {}", e.system, e.message),
            SessionError::Baseline(line) => write!(f, "malformed baseline at line {line}"),
            SessionError::Alias(e) => write!(f, "a reader's query does not hold: {e}"),
        }
    }
}

impl core::error::Error for SessionError {}

/// The sequence as text: `tick hash` per line, hex, newline-terminated.
///
/// Demo 02's format exactly. Text and not a blob so the first differing *line*
/// names the tick, which is most of the answer before anything is re-run; the
/// same shape as its neighbour so a second baseline is not a second thing to
/// learn how to read.
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
/// requires — a bare "hashes differ" over six hundred ticks is a wasted day.
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
/// systems table.
///
/// The game's systems and nothing else, which is why these values are **not** a
/// shell run's: `gg-runtime` also lets `gg_ui` write hit state back into every
/// `Widget` before it hashes, and this game is ~250 of them. Reproducing that
/// here would mean linking `gg-ui` into a game crate's test graph, which §3
/// forbids for better reasons than this one needs. So the two harnesses are tied
/// by the *shape* of their sequences instead — see `xtask reload --tetris`.
///
/// # Errors
///
/// If this crate's own component table is refused, or one of its systems panics.
#[cfg(feature = "game")]
pub fn hash_sequence(frames: &[InputFrame]) -> Result<Vec<CanonicalHash>, SessionError> {
    drive(frames, |world| Ok(world.canonical_hash()))
}

/// What a gate asks about a tick, read out of the world the systems just ran.
#[cfg(feature = "game")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Progress {
    /// Locked cells in the well. Not the falling piece: that is composed into
    /// the drawn grid and never written back into [`Well`].
    pub occupied: u32,
    /// Rows cleared, all game.
    pub lines: u32,
    /// One-based, as `Play::level` is.
    pub level: u32,
    /// Points.
    pub score: u32,
    /// Topped out. Only `RESTART` clears it.
    pub over: bool,
}

/// [`Progress`] after each of `frames`, through the same table [`hash_sequence`]
/// drives.
///
/// The counters a hash covers but cannot show. `xtask reload --rules` uses it to
/// say *which* board crossed a hot reload, and a run whose well happened to be
/// empty at the swap would prove nothing about a stack surviving one.
///
/// # Errors
///
/// As [`hash_sequence`].
#[cfg(feature = "game")]
pub fn progress(frames: &[InputFrame]) -> Result<Vec<Progress>, SessionError> {
    let query = Query::<(&Well, &Play)>::new().map_err(SessionError::Alias)?;
    drive(frames, |world| {
        let mut out = Progress {
            occupied: 0,
            lines: 0,
            level: 0,
            score: 0,
            over: false,
        };
        world.each_ref(&query, |_, (well, play): (&Well, &Play)| {
            for row in &well.cells {
                for cell in row {
                    out.occupied += u32::from(*cell != 0);
                }
            }
            out.lines = play.lines;
            out.level = play.level;
            out.score = play.score;
            out.over = play.over != 0;
        });
        Ok(out)
    })
}

/// What a long session costs, sampled rather than read every tick (§6 M51).
///
/// Two halves that fail in opposite directions, which is why they are one
/// struct. The first three are **growth** — a session that accumulates. The
/// last two are **liveness**, and they exist because a soak whose game quietly
/// stopped playing reports the flattest numbers a growth column can produce:
/// a bot wedged against a wall, a top-out nothing restarted, a world frozen by
/// a system that stopped running would each read as a perfect result.
#[cfg(feature = "game")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Footprint {
    /// Ticks run when this sample was taken.
    pub tick: u64,
    /// Live entities.
    pub entities: u32,
    /// The whole world encoded — what a save of this session would weigh, and
    /// the one number that sees a component growing a field's worth of rows
    /// without spawning anything.
    pub bytes: usize,
    /// Side tables, which is where host-side state a component cannot hold
    /// would accumulate.
    pub side_tables: usize,
    /// Ticks since the previous sample whose canonical hash differed from the
    /// tick before them. Liveness: the world is still moving.
    pub moved: u32,
    /// Rows cleared since tick zero, summed **across restarts** — `Play::lines`
    /// goes back to nothing when a game ends, and a soak's whole subject is the
    /// sessions after the first. Liveness, and the one that distinguishes a bot
    /// still playing well from one merely still moving.
    pub cleared: u32,
}

/// [`Footprint`] every `stride` ticks, through the same table [`hash_sequence`]
/// drives.
///
/// Sampled because the growth columns are not free — encoding the world is what
/// a save costs, and a soak long enough to be worth running is long enough that
/// paying it per tick would price the instrument out of the question it asks.
/// The liveness columns are accumulated over every tick in between, so nothing
/// between two samples goes unwatched; only the weighing is periodic.
///
/// # Errors
///
/// As [`hash_sequence`].
///
/// # Panics
///
/// If `stride` is zero.
#[cfg(feature = "game")]
pub fn footprint(frames: &[InputFrame], stride: usize) -> Result<Vec<Footprint>, SessionError> {
    assert!(stride > 0, "a sample stride of zero samples nothing");
    let query = Query::<&Play>::new().map_err(SessionError::Alias)?;
    let mut tick = 0u64;
    // `Option` rather than a zero sentinel: a hash is an opaque newtype with no
    // reserved value, and inventing one would make tick 0 unobservable.
    let mut previous: Option<CanonicalHash> = None;
    let mut moved = 0;
    // Carried across restarts by hand: `Play::lines` is the *current* game's,
    // and a drop is a game that ended rather than rows un-clearing.
    let (mut cleared, mut last_lines) = (0u32, 0u32);
    let samples = drive(frames, |world| {
        let hash = world.canonical_hash();
        moved += u32::from(previous.is_some_and(|was| was.get() != hash.get()));
        previous = Some(hash);
        let mut lines = last_lines;
        world.each_ref(&query, |_, play: &Play| lines = play.lines);
        cleared += lines.saturating_sub(last_lines);
        last_lines = lines;
        tick += 1;
        let due = tick.is_multiple_of(stride as u64) || tick as usize == frames.len();
        Ok(due.then(|| {
            let out = Footprint {
                tick,
                entities: world.len(),
                bytes: world.snapshot().encode().len(),
                side_tables: world.side_table_count(),
                moved,
                cleared,
            };
            moved = 0;
            out
        }))
    })?;
    Ok(samples.into_iter().flatten().collect())
}

/// Run `frames` through the declared table, reading something out of the world
/// after each tick.
#[cfg(feature = "game")]
fn drive<T>(
    frames: &[InputFrame],
    mut read: impl FnMut(&World) -> Result<T, SessionError>,
) -> Result<Vec<T>, SessionError> {
    // SAFETY: `gg_game_abi` and the tables below are this binary's own symbols,
    // exported by `gg_game!` in this crate and live for the process;
    // `host_api()` returns a `&'static` table (§4.2.2).
    let (table, mut world) = unsafe {
        debug_assert_eq!(
            gg_game_abi(),
            boundary::abi_info(),
            "one build, one description"
        );
        gg_game_init(core::ptr::from_ref(boundary::host_api()));
        let mut world = World::new();
        world
            .adopt(&gg_game_components())
            .map_err(SessionError::Adopt)?;
        (gg_game_systems(), world)
    };

    let mut out = Vec::with_capacity(frames.len());
    let mut previous = idle();
    for (tick, frame) in frames.iter().enumerate() {
        let ctx = TickCtx::detached(tick as u64, 60, *frame, previous);
        // SAFETY: the table is this binary's own, its entries live for the
        // process, and `ctx` outlives the call.
        unsafe { world.run_systems(&table, &ctx) }.map_err(SessionError::System)?;
        out.push(read(&world)?);
        previous = *frame;
    }
    Ok(out)
}

// The symbols `gg_game!` exported into this crate. Named here rather than called
// through the macro because the host reaches them exactly this way, and a
// harness that called the systems directly would be evidence about these
// functions rather than about the table (§4.2.2).
#[cfg(feature = "game")]
unsafe extern "C" {
    fn gg_game_abi() -> AbiInfo;
    fn gg_game_init(api: *const HostApiV1);
    fn gg_game_components() -> ComponentsTable;
    fn gg_game_systems() -> SystemsTable;
}
