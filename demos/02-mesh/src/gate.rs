//! The replay determinism gate's material (§5.6): the curated replay, the
//! chaos generator, the checked-in baselines, and the divergence report.
//!
//! It lives in the demo rather than in `xtask` because the gate has to *run*
//! where the demo runs — as an ordinary test, so the aarch64-under-qemu leg
//! picks it up by adding one `-p` to a `nextest` line, with no way to run an
//! `xtask` binary under emulation. `xtask replay --bless` calls the same code
//! from the other side, so the file that CI checks and the file that `--bless`
//! writes cannot drift apart.
//!
//! At M7 this moves into `gg-golden` alongside replay-driven scene playback
//! (§4.10); it is a demo module now because there is exactly one sim to drive.
//!
//! # What is checked in, and why the two differ
//!
//! - **The curated replay** — a scripted stream, its full per-tick hash
//!   sequence beside it. Precise: a divergence is reported at the exact tick.
//! - **Chaos seeds** (§5.11) — generated, never stored, with *checkpoints*
//!   every [`CHAOS_CHECKPOINT`] ticks. The wide net, kept small enough to read
//!   in a diff. A failing chaos seed gets promoted to a permanent curated
//!   regression replay — with its full sequence — **before** the bug is fixed.

use crate::sim::{self, Plant};
use gg_ecs::{CanonicalHash, Entity, World};
use gg_input::{InputFrame, Recorder, Replay, ReplayMeta};
use std::path::{Path, PathBuf};

/// Ticks in the curated replay: ten seconds at 60 Hz, long enough for the
/// camera to travel, the spin to wrap, and cubes to come and go.
pub const CURATED_TICKS: u64 = 600;

/// The curated replay's name under `tests/replays/`.
pub const CURATED: &str = "demo02-curated";

/// Chaos seeds run nightly on all three architectures (§5.11). Arbitrary but
/// fixed: a rotating seed set would make a nightly failure unreproducible,
/// which is the one thing a chaos suite may not be.
pub const CHAOS_SEEDS: &[u64] = &[1, 2, 3, 5, 8, 13, 21, 34];

/// Ticks per chaos run.
pub const CHAOS_TICKS: u64 = 600;

/// Ticks between chaos checkpoints.
pub const CHAOS_CHECKPOINT: u64 = 32;

/// Where the checked-in replays and baselines live.
pub fn replay_dir() -> PathBuf {
    // The demo's manifest dir, up to the workspace root. Not an env var and not
    // a search: a test that cannot find the baseline must fail loudly, not
    // quietly gate on nothing.
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .and_then(Path::parent)
        .unwrap_or(manifest)
        .join("tests/replays")
}

/// The replay file for `name`.
pub fn replay_path(name: &str) -> PathBuf {
    replay_dir().join(format!("{name}.ggrp"))
}

/// The hash-sequence baseline for `name`.
pub fn baseline_path(name: &str) -> PathBuf {
    replay_dir().join(format!("{name}.hashes"))
}

fn meta(seed: u64) -> ReplayMeta {
    ReplayMeta {
        // Deliberately *not* `gg_input::ENGINE_COMMIT`: a checked-in replay
        // whose header changed with every commit would produce a diff on every
        // commit. The curated replay names the commit it was blessed at, which
        // `--bless` stamps; generated chaos streams name none.
        engine_commit: "generated".to_owned(),
        contract: gg_math::DETERMINISM_CONTRACT,
        tier: "curated".to_owned(),
        tick_hz: sim::TICK_HZ,
        seed,
        actions: sim::ACTIONS.iter().map(|s| (*s).to_owned()).collect(),
        axes: sim::AXES.iter().map(|s| (*s).to_owned()).collect(),
        // A generated stream was laid out for no surface, and this one drives no
        // UI: `(0, 0)` is the same "the file does not say" a v2 file decodes to.
        surface: (0, 0),
    }
}

fn axes(right: i32, up: i32, forward: i32, look_x: i32, look_y: i32) -> [i32; gg_input::MAX_AXES] {
    let mut out = [0; gg_input::MAX_AXES];
    out[sim::MOVE_RIGHT.index()] = right * gg_input::AXIS_SCALE;
    out[sim::MOVE_UP.index()] = up * gg_input::AXIS_SCALE;
    out[sim::MOVE_FORWARD.index()] = forward * gg_input::AXIS_SCALE;
    out[sim::LOOK_X.index()] = look_x;
    out[sim::LOOK_Y.index()] = look_y;
    out
}

/// The curated stream: a scripted flight, written out rather than recorded, so
/// it can be regenerated from source if the file is ever lost.
///
/// It is *scripted* precisely because a human-recorded stream would be a blob
/// nobody could reason about — this one's phases are readable, and each exists
/// to move some state the hash covers.
pub fn curated_replay() -> Replay {
    let mut recorder = Recorder::new(meta(0x5eed));
    for tick in 0..CURATED_TICKS {
        let phase = tick / 100;
        let frame = match phase {
            // Fly forward, then strafe: the camera's `f64` position moves.
            0 => InputFrame {
                buttons: 0,
                axes: axes(0, 0, 1, 0, 0),
            },
            1 => InputFrame {
                buttons: 0,
                axes: axes(1, 0, 0, 0, 0),
            },
            // Turn while moving: yaw and pitch enter the hash, and the basis
            // stops being axis-aligned — libm's sin/cos now decide the path.
            2 => InputFrame {
                buttons: 1 << sim::LOOK.index(),
                axes: axes(0, 1, 1, 24, -8),
            },
            // Spawn on a slow pulse: structural change, archetype churn, and
            // entity-id allocation all join the hash.
            3 => InputFrame {
                buttons: (u64::from(tick % 20 < 2) << sim::SPAWN.index())
                    | (1 << sim::LOOK.index()),
                axes: axes(-1, 0, 0, -16, 4),
            },
            // ...and take them away again, exercising despawn's swap-remove.
            4 => InputFrame {
                buttons: u64::from(tick % 10 < 2) << sim::DESPAWN.index(),
                axes: axes(0, -1, -1, 0, 0),
            },
            // Coast: nothing but the spin counters advancing, which proves the
            // sequence keeps moving when the player has stopped.
            _ => InputFrame {
                buttons: 0,
                axes: axes(0, 0, 0, 0, 0),
            },
        };
        recorder.record(tick, frame);
    }
    recorder.finish()
}

/// xorshift64\*, ours on purpose: `rand`'s output is not a stability guarantee,
/// and a chaos seed has to mean the same stream in five years or the checked-in
/// regression replay it produced means nothing (§5.11).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // A zero state is xorshift's fixed point; seeds are ours to choose, but
        // a generated one should not be able to produce a silent all-zero run.
        Rng(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// A value in `0..n`.
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A randomized-but-seeded input stream (§5.11): movement, turning, and
/// spawn/despawn churn, all of it edges the sim has to agree about.
pub fn chaos_replay(seed: u64, ticks: u64) -> Replay {
    let mut rng = Rng::new(seed);
    let mut recorder = Recorder::new(meta(seed));
    let mut looking = false;
    for tick in 0..ticks {
        if rng.below(32) == 0 {
            looking = !looking;
        }
        let axis = |rng: &mut Rng| rng.below(3) as i32 - 1;
        let frame = InputFrame {
            buttons: (u64::from(looking) << sim::LOOK.index())
                | (u64::from(rng.below(16) == 0) << sim::SPAWN.index())
                | (u64::from(rng.below(24) == 0) << sim::DESPAWN.index()),
            axes: axes(
                axis(&mut rng),
                axis(&mut rng),
                axis(&mut rng),
                rng.below(129) as i32 - 64,
                rng.below(129) as i32 - 64,
            ),
        };
        recorder.record(tick, frame);
    }
    recorder.finish()
}

/// Every [`CHAOS_CHECKPOINT`]th hash of `sequence`, plus its last.
pub fn checkpoints(sequence: &[CanonicalHash]) -> Vec<(u64, CanonicalHash)> {
    let mut out: Vec<(u64, CanonicalHash)> = sequence
        .iter()
        .enumerate()
        .filter(|(i, _)| (*i as u64).is_multiple_of(CHAOS_CHECKPOINT))
        .map(|(i, h)| (i as u64, *h))
        .collect();
    if let Some((last, hash)) = sequence.len().checked_sub(1).zip(sequence.last())
        && out.last().map(|&(t, _)| t) != Some(last as u64)
    {
        out.push((last as u64, *hash));
    }
    out
}

/// A baseline as text: `tick hash` per line, hex, newline-terminated.
///
/// Text and not a binary blob because a divergence is supposed to be *readable*
/// in a diff — the first differing line names the tick, which is most of the
/// answer before anything is re-run.
pub fn encode_baseline(entries: &[(u64, CanonicalHash)]) -> String {
    let mut out = String::with_capacity(entries.len() * 40);
    for (tick, hash) in entries {
        out.push_str(&format!("{tick} {:032x}\n", hash.get()));
    }
    out
}

/// Parse [`encode_baseline`]'s output.
pub fn parse_baseline(text: &str) -> Result<Vec<(u64, u128)>, GateError> {
    let mut out = Vec::new();
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (tick, hash) = line
            .split_once(' ')
            .ok_or(GateError::Baseline { line: index + 1 })?;
        out.push((
            tick.parse()
                .map_err(|_| GateError::Baseline { line: index + 1 })?,
            u128::from_str_radix(hash, 16).map_err(|_| GateError::Baseline { line: index + 1 })?,
        ));
    }
    Ok(out)
}

/// The whole sequence as `(tick, hash)` pairs.
pub fn sequence_entries(sequence: &[CanonicalHash]) -> Vec<(u64, CanonicalHash)> {
    sequence
        .iter()
        .enumerate()
        .map(|(i, h)| (i as u64, *h))
        .collect()
}

/// Where two runs of one replay first stopped agreeing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Divergence {
    /// The first tick whose canonical hash differs.
    pub tick: u64,
    /// This run's hash there.
    pub mine: u128,
    /// The other side's.
    pub theirs: u128,
    /// The first entity whose state differs at that tick — available when both
    /// sides can be reproduced in this process (§5.6a, and the planted-
    /// divergence proof). A checked-in baseline from another architecture
    /// carries hashes and not worlds, so this is `None` there and the tick is
    /// what the report can honestly say.
    pub entity: Option<Entity>,
}

impl std::fmt::Display for Divergence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "first divergence at tick {}: {:032x} here, {:032x} expected",
            self.tick, self.mine, self.theirs
        )?;
        match self.entity {
            Some(e) => write!(
                f,
                "; first diverging entity is index {} generation {}",
                e.index(),
                e.generation()
            ),
            None => write!(
                f,
                "; entity not localized (the other side is a baseline, not a world) — \
                 re-run both sides with `xtask replay` to compare worlds"
            ),
        }
    }
}

/// Why the gate could not run.
#[derive(Debug, thiserror::Error)]
pub enum GateError {
    /// A baseline line was not `tick hash`.
    #[error("baseline line {line} is not `<tick> <hex hash>`")]
    Baseline {
        /// 1-based line number.
        line: usize,
    },
    /// The sim refused to run.
    #[error(transparent)]
    Sim(#[from] crate::sim::SimError),
    /// A checked-in file is missing or unreadable.
    #[error("{path}: {source}")]
    Io {
        /// The file.
        path: String,
        /// What the OS said.
        source: std::io::Error,
    },
}

/// Compare a run against a checked-in baseline, at whatever ticks the baseline
/// names (all of them for the curated replay, checkpoints for chaos).
pub fn compare(sequence: &[CanonicalHash], baseline: &[(u64, u128)]) -> Option<Divergence> {
    for &(tick, expected) in baseline {
        let Some(mine) = sequence.get(tick as usize) else {
            return Some(Divergence {
                tick,
                mine: 0,
                theirs: expected,
                entity: None,
            });
        };
        if mine.get() != expected {
            return Some(Divergence {
                tick,
                mine: mine.get(),
                theirs: expected,
                entity: None,
            });
        }
    }
    None
}

/// Compare two sequences produced *here*, and name the entity as well as the
/// tick — the full report §5.6 asks for.
pub fn compare_runs(
    replay: &Replay,
    mine: &[CanonicalHash],
    theirs: &[CanonicalHash],
    their_plant: Option<Plant>,
) -> Result<Option<Divergence>, GateError> {
    let Some((tick, (a, b))) = mine
        .iter()
        .zip(theirs)
        .enumerate()
        .find(|(_, (a, b))| a != b)
        .map(|(i, pair)| (i as u64, pair))
    else {
        return Ok(None);
    };
    // Re-run both sides *to the diverging tick* and diff entity by entity. The
    // whole-world hash says something moved; this says what.
    let ours = sim::world_at(replay, tick + 1, None)?;
    let theirs_world = sim::world_at(replay, tick + 1, their_plant)?;
    Ok(Some(Divergence {
        tick,
        mine: a.get(),
        theirs: b.get(),
        entity: locate_entity(&ours, &theirs_world),
    }))
}

/// The first entity whose state differs between two worlds, by id order.
///
/// Ids present in one world and not the other count: a spawn or despawn that
/// happened on one side only is exactly the divergence worth naming first.
pub fn locate_entity(a: &World, b: &World) -> Option<Entity> {
    let (ha, hb) = (a.entity_hashes(), b.entity_hashes());
    let (mut i, mut j) = (0, 0);
    while i < ha.len() && j < hb.len() {
        let (ea, hash_a) = ha[i];
        let (eb, hash_b) = hb[j];
        // `Entity`'s own order (index-major) — the order `entity_hashes` walks.
        // `to_bits()` is generation-major, and a merge keyed on it names the
        // wrong entity the moment a recycled index sits across the seam.
        match ea.cmp(&eb) {
            std::cmp::Ordering::Equal => {
                if hash_a != hash_b {
                    return Some(ea);
                }
                i += 1;
                j += 1;
            }
            std::cmp::Ordering::Less => return Some(ea),
            std::cmp::Ordering::Greater => return Some(eb),
        }
    }
    ha.get(i).or(hb.get(j)).map(|&(e, _)| e)
}

/// Read a file, naming it on failure.
pub fn read(path: &Path) -> Result<Vec<u8>, GateError> {
    std::fs::read(path).map_err(|source| GateError::Io {
        path: path.display().to_string(),
        source,
    })
}
