//! The M5 gates that can only be proven by *running the shell over a game
//! dylib* — §5.6c across tiers, replay segments across a reload, and the reload
//! chaos cases (§5.11). And one that proves the opposite: `--launcher` runs the
//! editor **application** over no dylib at all (§6 M15.1 item 4).
//!
//! Everything here is windowless (§1.5): `GG_HEADLESS=1`, so the shell opens no
//! window and its extract and render stages return immediately. The sim, the
//! reload path and the replay stream are identical either way, which is the
//! whole reason CI can exercise the shell it ships.
//!
//! # Why the curated stream is authored rather than recorded
//!
//! §5.6c says "a replay recorded under the true dist configuration". A headless
//! CI run has no hands: with nothing pressed, demo 03's world is static and its
//! canonical hash never moves, so a recorded-under-dist file would be an empty
//! stream and comparing it across tiers would prove that zero equals zero. The
//! criterion is therefore split rather than weakened, and both halves are real:
//!
//! - **The recorder ships and stamps its tier** — the dist shell records a run
//!   and the file decodes with `tier = "dist"`. That is what "recorded under
//!   dist" is protecting: §1.2's bug-report channel out of a shipped build.
//! - **Optimization cannot touch sim results** — a curated stream (scripted
//!   here, like demo 02's, so its phases are readable rather than a blob) drives
//!   the shell under dist-verify, dev and instrumented, and all three canonical
//!   hash sequences must be identical.
//!
//! The one thing the split gives up is a stream whose *bits* came out of a dist
//! binary. Nothing in the sim can tell where an [`InputFrame`] was authored —
//! that is the property that makes a replay a replay — so what is lost is the
//! appearance of provenance, not a check.
//!
//! # No checked-in hash baseline for demo 03, on purpose
//!
//! Demo 02's gate compares against a blessed sequence. Demo 03's must not: its
//! gameplay code is explicitly throwaway and is the crate agents edit all day
//! (§6 M5), so a per-tick baseline would go red on every gameplay tweak and be
//! re-blessed without being read — a gate that trains you to ignore it. The
//! tiers are compared *against each other* in one invocation instead, which is
//! exactly the claim 6c makes and needs no file.

use std::path::{Path, PathBuf};
use std::process::Command;

use gg_input::{InputFrame, Recorder, Replay, ReplayMeta};

use crate::util::{cargo, plain, run as exec, workspace_root};

/// The curated stream's name in `tests/replays/`.
pub const CURATED: &str = "demo03-curated";

/// Ticks it covers. Long enough for every phase below to leave a mark on the
/// hash, short enough that three tier runs stay a nightly cost.
const CURATED_TICKS: u64 = 300;

/// Demo 03's verbs, in the id order the game declares them (§4.7). Repeated here
/// rather than read from the dylib because the shell already refuses a replay
/// whose verb list disagrees with the game's, *by name* — so a drift is a loud
/// failure at load, not a stream replayed onto the wrong actions.
const ACTIONS: &[&str] = &["fire", "spawn", "restart"];
const AXES: &[&str] = &["move_right", "move_up", "move_forward", "aim_x", "aim_y"];

/// The tiers §5.6c compares. `tier-dist` is absent because dist computes no
/// canonical hash — which is the whole reason `dist-verify` exists (§2).
struct Tier {
    name: &'static str,
    features: &'static str,
    profile: &'static str,
    /// Cargo's output directory for that profile.
    out: &'static str,
}

const HASHED_TIERS: &[Tier] = &[
    Tier {
        name: "dev",
        features: "tier-dev",
        profile: "dev",
        out: "debug",
    },
    Tier {
        name: "instrumented",
        features: "tier-instrumented",
        profile: "instrumented",
        out: "instrumented",
    },
    Tier {
        name: "dist-verify",
        features: "tier-dist-verify",
        profile: "dist",
        out: "dist",
    },
];

// ---- the curated stream ------------------------------------------------

fn meta() -> ReplayMeta {
    let mut meta = ReplayMeta::new(
        gg_math::DETERMINISM_CONTRACT,
        "curated",
        gg_core::DEFAULT_TICK_HZ,
        ACTIONS,
        AXES,
    );
    // Not the running binary's commit: a checked-in replay whose header moved
    // with every commit would produce a diff on every commit (demo 02's rule).
    meta.engine_commit = "generated".to_owned();
    meta
}

fn axes(right: i32, up: i32, forward: i32, aim_x: i32, aim_y: i32) -> [i32; gg_input::MAX_AXES] {
    let mut out = [0; gg_input::MAX_AXES];
    let unit = gg_input::AXIS_SCALE;
    out[0] = right * unit;
    out[1] = up * unit;
    out[2] = forward * unit;
    // Aim is pointer motion: already in AXIS_SCALEths, and deliberately not a
    // whole unit — a mouse delta is small and the fixed point is what makes it
    // reproducible (§4.7).
    out[3] = aim_x;
    out[4] = aim_y;
    out
}

/// The scripted session: walk, look, shoot, spawn, restart, coast.
///
/// Each phase exists to move some state the canonical hash covers, and the
/// phases are in the order a person would actually do them — which is also the
/// order that makes a divergence readable ("it went wrong when we started
/// shooting").
pub fn curated_replay() -> Replay {
    let mut recorder = Recorder::new(meta());
    for tick in 0..CURATED_TICKS {
        let frame = match tick / 50 {
            // Walk forward while turning: the eye's f64 position moves and yaw
            // enters the hash, so libm's sin/cos decide the path.
            0 => InputFrame {
                buttons: 0,
                axes: axes(0, 0, 1, 18, -4),
            },
            // Strafe the other way, pitching back.
            1 => InputFrame {
                buttons: 0,
                axes: axes(-1, 0, 0, -12, 6),
            },
            // Fire on a pulse: shots spawn, which is entity allocation plus an
            // archetype the previous phases never touched.
            2 => InputFrame {
                buttons: u64::from(tick % 7 < 2),
                axes: axes(0, 0, 1, 4, 0),
            },
            // Spawn cubes on a slower pulse, while still moving.
            3 => InputFrame {
                buttons: u64::from(tick % 11 < 2) << 1,
                axes: axes(1, 0, 0, -6, 2),
            },
            // Restart once, then coast: despawn-everything followed by a fresh
            // bootstrap is the largest structural change the game can make, and
            // the freelist state it leaves behind is what §4.8 says a restore
            // has to carry.
            4 => InputFrame {
                buttons: u64::from(tick == 200) << 2,
                axes: axes(0, 0, 0, 0, 0),
            },
            // Shots fading out with nothing pressed — the sequence must keep
            // moving after the player has stopped, or the tail proves nothing.
            _ => InputFrame {
                buttons: 0,
                axes: axes(0, 0, 0, 0, 0),
            },
        };
        recorder.record(tick, frame);
    }
    recorder.finish()
}

/// Where the curated stream lives.
pub fn curated_path() -> PathBuf {
    workspace_root()
        .join("tests/replays")
        .join(format!("{CURATED}.ggrp"))
}

/// Rewrite it. Called by `xtask replay --bless`, beside demo 02's.
pub fn bless(commit: &str) -> anyhow::Result<()> {
    let mut replay = curated_replay();
    replay.set_engine_commit(commit);
    std::fs::write(curated_path(), replay.encode())?;
    println!(
        "xtask replay: blessed {CURATED} ({} ticks, {} change records) at {commit}",
        replay.ticks(),
        replay.change_count()
    );
    let mut ui = ui_replay();
    ui.set_engine_commit(commit);
    std::fs::write(ui_path(), ui.encode())?;
    println!(
        "xtask replay: blessed {UI_CURATED} ({} ticks, {} change records) at {commit}",
        ui.ticks(),
        ui.change_count()
    );
    let mut save = save_replay();
    save.set_engine_commit(commit);
    std::fs::write(save_replay_path(), save.encode())?;
    println!(
        "xtask replay: blessed {SAVE_CURATED} ({} ticks, {} change records) at {commit}",
        save.ticks(),
        save.change_count()
    );
    let mut editor = editor_replay()?;
    editor.set_engine_commit(commit);
    std::fs::write(editor_replay_path(), editor.encode())?;
    println!(
        "xtask replay: blessed {EDITOR_CURATED} ({} ticks, {} change records) at {commit}",
        editor.ticks(),
        editor.change_count()
    );
    let mut agent = agent_loop_replay()?;
    agent.set_engine_commit(commit);
    std::fs::write(agent_replay_path(), agent.encode())?;
    println!(
        "xtask replay: blessed {AGENT_CURATED} ({} ticks, {} change records, {} typed) at \
         {commit}",
        agent.ticks(),
        agent.change_count(),
        agent.typed_count()
    );
    let mut tetris = tetris_replay();
    tetris.set_engine_commit(commit);
    std::fs::write(tetris_path(), tetris.encode())?;
    // And the per-tick baseline beside it, driven through demo 10's own
    // declared systems table. This is the file the aarch64 leg compares
    // against, so it is authored by the same code the leg reads it with.
    let sequence = demo_10_tetris::session::hash_sequence(&demo_10_tetris::session::frames())?;
    std::fs::write(
        demo_10_tetris::session::baseline_path(),
        demo_10_tetris::session::encode_baseline(&sequence),
    )?;
    println!(
        "xtask replay: blessed {} ({} ticks, {} change records) and its {}-tick baseline at \
         {commit}",
        demo_10_tetris::session::NAME,
        tetris.ticks(),
        tetris.change_count(),
        sequence.len(),
    );
    let entry = platformer_entry()?;
    let frames = demo_11_platformer::session::frames(&entry)?;
    let opening = demo_11_platformer::session::opening_tick()?;
    let mut platformer = platformer_stream(&frames, opening);
    platformer.set_engine_commit(commit);
    std::fs::write(platformer_path(), platformer.encode())?;
    // Demo 10's pattern: the per-tick baseline beside it, authored by the same
    // code the aarch64 leg reads it with — here over the checked-in scene, so
    // re-authoring the level is what re-authors both files.
    let sequence = demo_11_platformer::session::hash_sequence(&entry, &frames)?;
    std::fs::write(
        demo_11_platformer::session::baseline_path(),
        demo_11_platformer::session::encode_baseline(&sequence),
    )?;
    println!(
        "xtask replay: blessed {} ({} ticks, {} change records) and its {}-tick baseline at \
         {commit}",
        demo_11_platformer::session::NAME,
        platformer.ticks(),
        platformer.change_count(),
        sequence.len(),
    );
    Ok(())
}

/// Demo 10's verbs, in the id order `gg_game!` declares them (§4.7). The axis
/// list stays empty on purpose: the recorded sessions are keyboard games, the
/// game's `ui_x`/`ui_y` are *appended* verbs — which `check_verbs` accepts —
/// and a stream with no axis slots encodes no dead zeroes per frame.
const TETRIS_ACTIONS: &[&str] = &[
    "left",
    "right",
    "soft_drop",
    "hard_drop",
    "rotate_cw",
    "rotate_ccw",
    "hold",
    "restart",
    "pause",
    "ui_click",
    "ui_focus",
];

/// The recorded full game as a replay file — `demo_10_tetris::session`'s frames,
/// so this file, the demo's own tests and the baseline are one script.
pub fn tetris_replay() -> Replay {
    tetris_stream(&demo_10_tetris::session::frames())
}

/// Any of demo 10's scripts as a stream the shell can replay. The verb list is
/// the game's declared order (§4.7) and is what the frames' bits index.
fn tetris_stream(frames: &[InputFrame]) -> Replay {
    let mut meta = ReplayMeta::new(
        gg_math::DETERMINISM_CONTRACT,
        "curated",
        gg_core::DEFAULT_TICK_HZ,
        TETRIS_ACTIONS,
        &[],
    );
    meta.engine_commit = "generated".to_owned();
    let mut recorder = Recorder::new(meta);
    for (tick, frame) in frames.iter().enumerate() {
        recorder.record(tick as u64, *frame);
    }
    recorder.finish()
}

pub fn tetris_path() -> PathBuf {
    workspace_root()
        .join("tests/replays")
        .join(format!("{}.ggrp", demo_10_tetris::session::NAME))
}

/// The checked-in Tetris stream is still the script, byte for byte.
///
/// This closes the hole the demo's own tests leave open by design: they drive
/// `session::frames()` rather than decoding the `.ggrp`, because `gg-input` is
/// not a dependency a game crate may take (§3), and a harness that regenerates
/// its own input passes happily the day the generator changes. So the artifact
/// is checked from the side that *can* read it — here, in the push tier.
///
/// The blessed commit is copied off the file before the compare rather than
/// compared: that field is the one thing `--bless` owns, and a stream is not
/// stale for having been blessed at an older commit (§4.7).
pub fn check_tetris() -> anyhow::Result<()> {
    let path = tetris_path();
    let on_disk = std::fs::read(&path)
        .map_err(|e| anyhow::anyhow!("no Tetris stream at {} ({e})", path.display()))?;
    let decoded = Replay::decode(&on_disk)?;
    let mut fresh = tetris_replay();
    fresh.set_engine_commit(&decoded.meta().engine_commit);
    // Re-encoding what was read separates the two ways this can differ. Equal
    // here means the *stream* is still the script and only the file's encoding
    // version is behind, which is what a `FORMAT` bump does to every generated
    // artifact at once — a failure that blames the script for it sends the next
    // reader after a divergence that is not there.
    anyhow::ensure!(
        fresh.encode() != decoded.encode() || fresh.encode() == on_disk,
        "{} is the stream `demo_10_tetris::session::frames()` still produces, written in an older \
         encoding — `cargo xtask replay --bless` re-authors it at the current `gg_input::replay::\
         FORMAT`",
        path.display(),
    );
    anyhow::ensure!(
        fresh.encode() == on_disk,
        "{} is not what `demo_10_tetris::session::frames()` produces today ({} ticks on disk, {} \
         from the script) — the recording and the script have come apart, and `cargo xtask replay \
         --bless` is the reviewed act that re-joins them",
        path.display(),
        decoded.ticks(),
        fresh.ticks(),
    );
    println!(
        "xtask replay: {} matches its script ({} ticks, blessed at {})",
        demo_10_tetris::session::NAME,
        decoded.ticks(),
        decoded.meta().engine_commit,
    );
    Ok(())
}

/// Demo 11's verbs, in the id order `gg_game!` declares them (§4.7) — and,
/// unlike demo 10's, an axis: a platformer is steered, so its frames carry a
/// `move` value and the stream must declare the slot. The host's appended UI
/// verbs are absent the way `check_verbs` accepts.
const PLATFORMER_ACTIONS: &[&str] = &["jump", "restart"];
const PLATFORMER_AXES: &[&str] = &["move"];

/// Demo 11's entry points, resolved out of the **built dev dylib** — the
/// host's own road to the declared tables (§4.2.2). This binary cannot link
/// them: demo 10 already owns the fixed `gg_game_*` names in this graph (see
/// Cargo.toml), which is exactly why `session::Entry` exists.
fn platformer_entry() -> anyhow::Result<demo_11_platformer::session::Entry> {
    // Once per process: the loaded copy below is leaked, so a second call
    // would copy onto its own locked staging file — a retrying gate (`feel`'s
    // clock resamples) must reuse the entry, not reload it.
    static ENTRY: std::sync::OnceLock<demo_11_platformer::session::Entry> =
        std::sync::OnceLock::new();
    if let Some(entry) = ENTRY.get() {
        return Ok(*entry);
    }
    exec(
        cargo().args(["build", "-p", "demo-11-platformer"]),
        "build demo-11-platformer [dev]",
    )?;
    // A staged copy, never cargo's own output: a loaded DLL is locked on
    // Windows, and holding `target/debug`'s artifact would fail every later
    // `cargo build` of this crate — in this process, a parallel gate, or the
    // user's own session — with LNK1104.
    let built = dylib("debug", "demo_11_platformer");
    let dir = workspace_root().join("target/entry");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(built.file_name().unwrap_or_default());
    std::fs::copy(&built, &path)?;
    // SAFETY: the artifact was built from this same working tree one line up,
    // so the signatures are this build's own; the session's `assemble` refuses
    // a foreign `AbiInfo` before anything is called. The library is leaked —
    // a game dylib is never unloaded (§4.2.2) — so the pointers stay valid for
    // the process.
    unsafe {
        let lib = libloading::Library::new(&path)
            .map_err(|e| anyhow::anyhow!("loading {} failed: {e}", path.display()))?;
        let entry = demo_11_platformer::session::Entry {
            abi: *lib.get(b"gg_game_abi")?,
            init: *lib.get(b"gg_game_init")?,
            components: *lib.get(b"gg_game_components")?,
            systems: *lib.get(b"gg_game_systems")?,
        };
        std::mem::forget(lib);
        Ok(*ENTRY.get_or_init(|| entry))
    }
}

/// The recorded full run as a replay file — `demo_11_platformer::session`'s
/// frames, so this file, the demo's own tests and the baseline are one script.
///
/// # Errors
///
/// If the bot cannot finish the level the checked-in scene holds.
pub fn platformer_replay() -> anyhow::Result<Replay> {
    let entry = platformer_entry()?;
    let frames = demo_11_platformer::session::frames(&entry)?;
    let opening = demo_11_platformer::session::opening_tick()?;
    Ok(platformer_stream(&frames, opening))
}

/// Any of demo 11's scripts as a stream the shell can replay, indexed from the
/// scene's own tick: a replayed session is handed `--load scene.ggsave`
/// explicitly (record/replay never probe, §6 M20 item 3), which resumes the
/// world at the save's tick, and the seek expects the frames there.
fn platformer_stream(frames: &[InputFrame], opening: u64) -> Replay {
    let mut meta = ReplayMeta::new(
        gg_math::DETERMINISM_CONTRACT,
        "curated",
        gg_core::DEFAULT_TICK_HZ,
        PLATFORMER_ACTIONS,
        PLATFORMER_AXES,
    );
    meta.engine_commit = "generated".to_owned();
    let mut recorder = Recorder::new(meta);
    for (offset, frame) in frames.iter().enumerate() {
        recorder.record(opening + offset as u64, *frame);
    }
    recorder.finish()
}

pub fn platformer_path() -> PathBuf {
    workspace_root()
        .join("tests/replays")
        .join(format!("{}.ggrp", demo_11_platformer::session::NAME))
}

/// The level every platformer gate hands the shell — replayed sessions never
/// probe, so the scene is always this explicit spelling.
fn platformer_scene() -> anyhow::Result<String> {
    let scene = workspace_root().join("demos/11-platformer/scene.ggsave");
    anyhow::ensure!(
        scene.is_file(),
        "no level at {} — the checked-in scene is the platformer's stage",
        scene.display()
    );
    Ok(scene.display().to_string())
}

/// The checked-in platformer stream is still the script, byte for byte —
/// `check_tetris`'s reasoning verbatim, including the encoding-version split.
pub fn check_platformer() -> anyhow::Result<()> {
    let path = platformer_path();
    let on_disk = std::fs::read(&path)
        .map_err(|e| anyhow::anyhow!("no platformer stream at {} ({e})", path.display()))?;
    let decoded = Replay::decode(&on_disk)?;
    let mut fresh = platformer_replay()?;
    fresh.set_engine_commit(&decoded.meta().engine_commit);
    anyhow::ensure!(
        fresh.encode() != decoded.encode() || fresh.encode() == on_disk,
        "{} is the stream `demo_11_platformer::session::frames()` still produces, written in an \
         older encoding — `cargo xtask replay --bless` re-authors it at the current \
         `gg_input::replay::FORMAT`",
        path.display(),
    );
    anyhow::ensure!(
        fresh.encode() == on_disk,
        "{} is not what `demo_11_platformer::session::frames()` produces today ({} ticks on disk, \
         {} from the script) — the recording and the script (or the scene it plays) have come \
         apart, and `cargo xtask replay --bless` is the reviewed act that re-joins them",
        path.display(),
        decoded.ticks(),
        fresh.ticks(),
    );
    println!(
        "xtask replay: {} matches its script ({} ticks, blessed at {})",
        demo_11_platformer::session::NAME,
        decoded.ticks(),
        decoded.meta().engine_commit,
    );
    Ok(())
}

/// Demo 07's name in `tests/replays/`, and its verbs in declared order (§4.7).
/// `gg_ui::boundary::binding` finds the four by *name* in the loaded dylib's
/// list; this order is what the recorded frames index.
pub const UI_CURATED: &str = "demo07-ui";
const UI_ACTIONS: &[&str] = &["ui_click", "ui_focus"];
const UI_AXES: &[&str] = &["ui_x", "ui_y"];

/// The scripted UI session as a replay file.
///
/// The frames are `demo_07_ui::session()`'s — authored in the demo so this file
/// and the demo's own in-process test are the same script, and so moving a
/// button in `LAYOUT` moves the click rather than silently missing it.
pub fn ui_replay() -> Replay {
    let mut meta = ReplayMeta::new(
        gg_math::DETERMINISM_CONTRACT,
        "curated",
        gg_core::DEFAULT_TICK_HZ,
        UI_ACTIONS,
        UI_AXES,
    );
    meta.engine_commit = "generated".to_owned();
    let mut recorder = Recorder::new(meta);
    for (tick, frame) in demo_07_ui::session().into_iter().enumerate() {
        recorder.record(tick as u64, frame);
    }
    recorder.finish()
}

pub fn ui_path() -> PathBuf {
    workspace_root()
        .join("tests/replays")
        .join(format!("{UI_CURATED}.ggrp"))
}

/// Demo 08's name in `tests/replays/`, and its verbs in declared order (§4.7).
pub const SAVE_CURATED: &str = "demo08-save";
const SAVE_ACTIONS: &[&str] = &["bank"];
const SAVE_AXES: &[&str] = &["move_x", "move_z"];

/// The scripted walk as a replay file — `demo_08_save::session()`'s frames, so
/// this file and the demo's own tests are the same script and moving a chest
/// moves the walk.
pub fn save_replay() -> Replay {
    let mut meta = ReplayMeta::new(
        gg_math::DETERMINISM_CONTRACT,
        "curated",
        gg_core::DEFAULT_TICK_HZ,
        SAVE_ACTIONS,
        SAVE_AXES,
    );
    meta.engine_commit = "generated".to_owned();
    let mut recorder = Recorder::new(meta);
    for (tick, frame) in demo_08_save::session().into_iter().enumerate() {
        recorder.record(tick as u64, frame);
    }
    recorder.finish()
}

/// The editor session's name in `tests/replays/`, and the *game's* verbs — demo
/// 05's own, which the editor appends to rather than replaces (§6 M15).
///
/// Written out rather than read off the dylib because a replay is authored
/// offline. A drift is not silent: the shell checks a replay's verb lists
/// against the loaded build at load and refuses a mismatch by name (§4.7), so
/// renaming one of demo 05's verbs fails this gate rather than replaying the
/// wrong ones.
pub const EDITOR_CURATED: &str = "demo05-editor";
const EDITOR_ACTIONS: &[&str] = &["freeze"];
const EDITOR_AXES: &[&str] = &["move_right", "move_up", "move_forward", "aim_x", "aim_y"];

/// The verbs a shell binds with the editor open over demo 05, through the
/// host's own append rule rather than a copy of it: a change to which verbs the
/// editor adds moves this file with it.
fn editor_verbs() -> gg_ecs::boundary::Verbs {
    let (verbs, _) = gg_editor::host::open(&gg_ecs::boundary::Verbs {
        actions: EDITOR_ACTIONS,
        axes: EDITOR_AXES,
    });
    verbs
}

/// What `gg_runtime`'s `Stages::surface` reports with no renderer — the extent
/// a headless editor session is laid out at, and therefore the one its script
/// must be aimed at.
const HEADLESS_EXTENT: (u32, u32) = gg_ecs::boundary::CANVAS;

/// And the monitor it reports: none, which is 1.0 (§6 M15.1). The shell's own
/// `App` starts there and a headless run never moves it, so a script aimed with
/// this lands where the replayed session clicks.
const HEADLESS_DPI: f32 = 1.0;

/// The scripted editor session as a replay.
///
/// The frames are `gg_editor::session`'s — authored beside the panels they
/// click, so a panel that moves moves the script rather than missing it (§6
/// M15) — and the ids come out of the augmented verb list, because `ui_click`
/// is action 1 over demo 05 and action 0 over a game that declared it first.
///
/// The script is aimed at [`HEADLESS_EXTENT`] because that is what the shell
/// lays the editor out at with no renderer to ask (`Stages::surface`). Since
/// §6 M15.1 the panes fill their surface, so a script aimed at one extent and
/// replayed at another clicks somewhere else — which is exactly the residual
/// M15.1 names, and here it is closed by the two agreeing on one constant.
pub fn editor_replay() -> anyhow::Result<Replay> {
    editor_replay_at(HEADLESS_EXTENT)
}

/// The same session authored for a different surface — what `--editor-extent`
/// is replayed against, so the flag that closes M15.1's residual is exercised
/// rather than asserted.
pub fn editor_replay_at(extent: (u32, u32)) -> anyhow::Result<Replay> {
    let verbs = editor_verbs();
    let find = |names: &[&str], want: &str| {
        names
            .iter()
            .position(|n| *n == want)
            .ok_or_else(|| anyhow::anyhow!("the editor did not append `{want}`"))
    };
    let mut editor = gg_editor::Editor::new(None);
    editor.place(extent, HEADLESS_DPI);
    let (x, y) = (
        gg_input::AxisId::new(find(verbs.axes, "ui_x")?),
        gg_input::AxisId::new(find(verbs.axes, "ui_y")?),
    );
    let mut frames = gg_editor::session::frames(
        &gg_editor::session::script(&editor),
        gg_input::ActionId::new(find(verbs.actions, "ui_click")?),
        x,
        y,
    );
    // §6 M15.2 item 4: fly the editor camera. Appended here rather than inside
    // `script` because a camera verb aims at no rectangle — the script is
    // authored beside the panels precisely because its clicks are, and this has
    // nothing to be authored beside. It lands *after* the stop above, which is
    // the state the camera is live in.
    let camera = |name| -> anyhow::Result<gg_input::ActionId> {
        Ok(gg_input::ActionId::new(find(verbs.actions, name)?))
    };
    // The look drag, on its own axes and with its own log line: a turn and a
    // move are two claims, and one line for both would let the keys below
    // satisfy a drag that reached no axis at all — which is the failure this
    // pair can have, since the axes are appended per session.
    //
    // Not `(x, y)`: those are cursor motion, which is what this differenced
    // until a wider `MAX_AXES` bought the camera its own axes — and with them a
    // drag that does not stop at the window edge (§6 M15.2's residual).
    let look = |name| -> anyhow::Result<gg_input::AxisId> {
        Ok(gg_input::AxisId::new(find(verbs.axes, name)?))
    };
    frames.extend(gg_editor::session::hold(
        camera(gg_editor::host::verb::LOOK)?,
        24,
        (
            look(gg_editor::host::verb::LOOK_X)?,
            look(gg_editor::host::verb::LOOK_Y)?,
        ),
        (4 * gg_input::AXIS_SCALE, 0),
    ));
    frames.extend(gg_editor::session::hold(
        camera(gg_editor::host::verb::FORWARD)?,
        24,
        (x, y),
        (0, 0),
    ));
    let mut meta = ReplayMeta::new(
        gg_math::DETERMINISM_CONTRACT,
        "curated",
        gg_core::DEFAULT_TICK_HZ,
        verbs.actions,
        verbs.axes,
    );
    meta.engine_commit = "generated".to_owned();
    let mut recorder = Recorder::new(meta);
    for (tick, frame) in frames.into_iter().enumerate() {
        recorder.record(tick as u64, frame);
    }
    Ok(recorder.finish())
}

pub fn editor_replay_path() -> PathBuf {
    workspace_root()
        .join("tests/replays")
        .join(format!("{EDITOR_CURATED}.ggrp"))
}

pub fn save_replay_path() -> PathBuf {
    workspace_root()
        .join("tests/replays")
        .join(format!("{SAVE_CURATED}.ggrp"))
}

// ---- the §1 loop over demo 03 (§6 M16 exit row 4) -----------------------

pub const AGENT_CURATED: &str = "demo03-agent";

/// Demo 03's verbs, in `gg_game!`'s declared order — the id space the frames
/// below index, before the editor appends its own (§4.7).
const AGENT_ACTIONS: &[&str] = &["fire", "spawn", "restart"];
const AGENT_AXES: &[&str] = &["move_right", "move_up", "move_forward", "aim_x", "aim_y"];

/// The §1 loop as one session, in the order a replay can hold it
/// (`gg_editor::session::AgentScript`): ask the panel for the change, hand
/// the game the pointer, and play across wherever the rebuild lands. The
/// typed prompt rides the text channel by tick rather than the action map
/// (§6 M16 item 5), and what it asks for — a wobble — is the field the gate's
/// rebuilt dylib then delivers, because the reload is the *environment's*
/// half of the loop: the gate lands it the way an agent's `cargo build`
/// would, under the watcher, after the ask.
///
/// Long for `agent_session`'s reason: the tail is wall time for the watcher's
/// debounce, so the rebuild lands mid-session rather than after it.
pub fn agent_loop_replay() -> anyhow::Result<Replay> {
    const TICKS: u64 = 60_000;
    let (verbs, _) = gg_editor::host::open(&gg_ecs::boundary::Verbs {
        actions: AGENT_ACTIONS,
        axes: AGENT_AXES,
    });
    let find = |names: &[&str], want: &str| {
        names
            .iter()
            .position(|n| *n == want)
            .ok_or_else(|| anyhow::anyhow!("no verb `{want}` to aim the loop at"))
    };
    let mut editor = gg_editor::Editor::new(None);
    editor.place(HEADLESS_EXTENT, HEADLESS_DPI);
    // Raised for the author only: the replayed editor starts with the pane
    // down, and the script's own tab click is what brings it up there.
    editor.raise(gg_editor::Pane::Agent);
    let script = gg_editor::session::agent_script(&editor).ok_or_else(|| {
        anyhow::anyhow!("the agent pane is not up, so there is nothing to aim at")
    })?;
    let click = gg_input::ActionId::new(find(verbs.actions, "ui_click")?);
    let (x, y) = (
        gg_input::AxisId::new(find(verbs.axes, "ui_x")?),
        gg_input::AxisId::new(find(verbs.axes, "ui_y")?),
    );
    let mut frames = gg_editor::session::frames(&script.focus, click, x, y);
    // Into the settle at the head of the ask piece: focused, click released.
    let typed_at = frames.len() as u64 + 4;
    frames.extend(gg_editor::session::frames_from(
        script.prompt,
        &script.ask,
        click,
        x,
        y,
    ));
    // Then §1's play, in the only order a replay can hold it (see
    // `AgentScript`): the viewport press hands the game the pointer, and the
    // held verbs after it are the hands the sim answers to — running from the
    // ask to the far side of wherever the rebuild lands.
    frames.extend(gg_editor::session::frames_from(
        script.send,
        &script.play,
        click,
        x,
        y,
    ));
    let fire = gg_input::ActionId::new(find(verbs.actions, "fire")?);
    let (right, forward) = (
        gg_input::AxisId::new(find(verbs.axes, "move_right")?),
        gg_input::AxisId::new(find(verbs.axes, "move_forward")?),
    );
    frames.extend(gg_editor::session::hold(
        fire,
        200,
        (right, forward),
        (gg_input::AXIS_SCALE, 2 * gg_input::AXIS_SCALE),
    ));
    let mut meta = ReplayMeta::new(
        gg_math::DETERMINISM_CONTRACT,
        "curated",
        gg_core::DEFAULT_TICK_HZ,
        verbs.actions,
        verbs.axes,
    );
    meta.engine_commit = "generated".to_owned();
    let mut recorder = Recorder::new(meta);
    let played = frames.len() as u64;
    for (tick, frame) in frames.into_iter().enumerate() {
        recorder.record(tick as u64, frame);
    }
    recorder.record_text(typed_at, "add a wobble to the cube, then cargo build");
    // The tail costs one change record: `record` is a delta.
    for tick in played..TICKS {
        recorder.record(tick, InputFrame::default());
    }
    Ok(recorder.finish())
}

pub fn agent_replay_path() -> PathBuf {
    workspace_root()
        .join("tests/replays")
        .join(format!("{AGENT_CURATED}.ggrp"))
}

/// Read it back and say what it claims — the check half.
pub fn describe() -> anyhow::Result<()> {
    let replay = Replay::decode(&std::fs::read(curated_path())?)?;
    println!(
        "xtask replay: {CURATED} — {} ticks, {} change record(s), contract v{}, blessed at {}",
        replay.ticks(),
        replay.change_count(),
        replay.meta().contract,
        replay.meta().engine_commit
    );
    Ok(())
}

// ---- building and running the shell over demo 03 -----------------------

fn exe(dir: &str, name: &str) -> PathBuf {
    workspace_root()
        .join("target")
        .join(dir)
        .join(if cfg!(windows) {
            format!("{name}.exe")
        } else {
            name.to_owned()
        })
}

fn dylib(dir: &str, stem: &str) -> PathBuf {
    workspace_root()
        .join("target")
        .join(dir)
        .join(crate::util::dylib_name(stem))
}

/// Build the shell and the game for one tier, and stage both under
/// `target/tiers/<name>/`.
///
/// Staged rather than run in place because `tier-dist` and `tier-dist-verify`
/// share the `dist` profile and therefore the same output path: building the
/// second would overwrite the first, and a gate that silently compared a tier
/// against itself would be green for the wrong reason.
fn stage(tier: &Tier) -> anyhow::Result<(PathBuf, PathBuf)> {
    stage_game(tier, "demo-03-reload", "demo_03_reload")
}

/// [`stage`], over some other game crate.
fn stage_game(tier: &Tier, package: &str, stem: &str) -> anyhow::Result<(PathBuf, PathBuf)> {
    exec(
        cargo().args([
            "build",
            "-p",
            "gg-runtime",
            "--profile",
            tier.profile,
            "--no-default-features",
            "--features",
            tier.features,
        ]),
        &format!("build gg-runtime [{}]", tier.name),
    )?;
    exec(
        cargo().args(["build", "-p", package, "--profile", tier.profile]),
        &format!("build {package} [{} profile]", tier.profile),
    )?;

    let dir = workspace_root().join("target/tiers").join(tier.name);
    std::fs::create_dir_all(&dir)?;
    let host = dir.join(exe(tier.out, "gg-runtime").file_name().unwrap_or_default());
    let game = dir.join(dylib(tier.out, stem).file_name().unwrap_or_default());
    std::fs::copy(exe(tier.out, "gg-runtime"), &host)?;
    std::fs::copy(dylib(tier.out, stem), &game)?;
    Ok((host, game))
}

/// Run a staged shell over a staged game and return its whole log.
fn play(host: &Path, game: &Path, args: &[&str], hashes: bool) -> anyhow::Result<String> {
    let mut cmd = Command::new(host);
    cmd.arg("--game").arg(game).args(args);
    cmd.env("GG_HEADLESS", "1");
    // The hash target is off by default (one line per tick); this is the gate
    // that wants it.
    cmd.env(
        "RUST_LOG",
        if hashes {
            "info,gg::hash=debug"
        } else {
            "info"
        },
    );
    let out = cmd
        .output()
        .map_err(|e| anyhow::anyhow!("failed to spawn {}: {e}", host.display()))?;
    let log = format!("{}{}", plain(&out.stdout), plain(&out.stderr));
    anyhow::ensure!(
        out.status.success(),
        "{} exited {}\n{log}",
        host.display(),
        out.status
    );
    Ok(log)
}

/// The `(tick, hash)` sequence a run emitted, in tick order.
fn sequence(log: &str) -> anyhow::Result<Vec<(u64, String)>> {
    let mut out = Vec::new();
    for line in log.lines().filter(|l| l.contains("gg::hash")) {
        out.push((
            crate::util::field_u64(line, "tick")?,
            crate::util::field(line, "hash")?.to_owned(),
        ));
    }
    anyhow::ensure!(!out.is_empty(), "the run emitted no state hashes:\n{log}");
    Ok(out)
}

/// Where two sequences first disagree, named the way §5.6 requires — a bare
/// "hashes differ" on tick 9,000 is a wasted day.
fn divergence(a: &(&str, Vec<(u64, String)>), b: &(&str, Vec<(u64, String)>)) -> Option<String> {
    for (left, right) in a.1.iter().zip(&b.1) {
        if left != right {
            return Some(format!(
                "{} and {} diverge at tick {}: {} vs {}",
                a.0, b.0, left.0, left.1, right.1
            ));
        }
    }
    (a.1.len() != b.1.len()).then(|| {
        format!(
            "{} ran {} ticks and {} ran {}",
            a.0,
            a.1.len(),
            b.0,
            b.1.len()
        )
    })
}

// ---- game variants ------------------------------------------------------

/// The crate a generated variant is a copy of.
///
/// Two, because the gates below ask different questions of different games:
/// demo 03 is the migration and latency subject, and demo 10 is the one with a
/// game in progress to change the rules of.
struct Kind {
    /// The demo directory, under the workspace root.
    dir: &'static str,
    /// The artifact's name. The host loads a *path*, so this is the game's
    /// identity as far as the shell is concerned — and why every variant of one
    /// crate builds a library called the same thing.
    lib: &'static str,
    /// Sources beside `lib.rs` that its own `mod` lines name, copied unedited: a
    /// variant differs from its original by the edit under test and nothing else.
    beside: &'static [&'static str],
    /// Features the crate's manifest turns on by default, which the generated
    /// manifest has to declare too — demo 10's `gg_game!` is behind one, and a
    /// variant missing it builds a dylib with no entry points at all.
    features: &'static [&'static str],
}

const DEMO_03: &Kind = &Kind {
    dir: "demos/03-reload",
    lib: "demo_03_reload",
    beside: &[],
    features: &[],
};

const DEMO_10: &Kind = &Kind {
    dir: "demos/10-tetris",
    lib: "demo_10_tetris",
    beside: &["session.rs"],
    features: &["game"],
};

const DEMO_11: &Kind = &Kind {
    dir: "demos/11-platformer",
    lib: "demo_11_platformer",
    beside: &["session.rs"],
    features: &["game"],
};

/// A demo's `lib.rs`, which every variant below is an edit of.
fn game_source(kind: &Kind) -> anyhow::Result<String> {
    Ok(std::fs::read_to_string(
        workspace_root().join(kind.dir).join("src/lib.rs"),
    )?)
}

/// The migration edit: one more field on a hashed component.
///
/// `u64` rather than `f32` because `bytemuck::Pod` refuses a struct with
/// padding, and `Cube` is 48 bytes at align 8 — a four-byte field would need a
/// second one beside it and say nothing extra.
fn with_an_extra_field(source: &str) -> anyhow::Result<String> {
    // Two edits, because a field is not just a declaration: the struct literal
    // that builds a `Cube` has to name it too, and that pair is exactly what an
    // agent types when it adds one.
    let edits = [
        (
            "    /// Half-extent, metres.\n    pub scale: f32,\n}",
            "    /// Half-extent, metres.\n    pub scale: f32,\n    /// Added by the reload \
             gates: a field the running world has never seen, so\n    /// adopting this build is \
             a migration rather than a pointer swap (§4.2.2).\n    pub wobble: u64,\n}",
        ),
        (
            "            scale: 0.5,\n",
            "            scale: 0.5,\n            wobble: 0,\n",
        ),
    ];
    let mut out = source.to_owned();
    for (anchor, replacement) in edits {
        anyhow::ensure!(
            out.contains(anchor),
            "demo 03's source no longer contains `{}` — the migration variant is a text edit, so \
             a rename here is a gate to re-point rather than one that quietly stops migrating \
             anything",
            anchor.trim()
        );
        out = out.replace(anchor, replacement);
    }
    Ok(out)
}

fn dylib_name(kind: &Kind) -> String {
    crate::util::dylib_name(kind.lib)
}

/// Build a variant of demo 03 — its source with one edit applied — as a
/// standalone game dylib the host can be pointed at.
///
/// Generated rather than checked in, and *edited from demo 03's own source*
/// rather than written fresh, because the interesting property is that the two
/// builds differ by exactly the edit under test. A second checked-in game crate
/// would drift away from the first the week after it was added.
///
/// It is its own workspace (the empty `[workspace]` table) so cargo does not try
/// to adopt a package sitting under `target/`, and every variant shares one
/// target directory so the engine crates compile once rather than once per
/// variant. The artifact is copied aside afterwards because they all build a
/// library called `demo_03_reload` — which is the point: the host's path never
/// changes, only the bytes behind it.
fn variant(kind: &Kind, name: &str, source: &str) -> anyhow::Result<PathBuf> {
    let dir = write_variant(kind, name, source)?;
    build_variant(kind, name)?;
    let kept = dir.join(dylib_name(kind));
    std::fs::copy(shared_target().join("debug").join(dylib_name(kind)), &kept)?;
    Ok(kept)
}

/// One target directory for every variant, so the engine crates compile once
/// rather than once per variant.
fn shared_target() -> PathBuf {
    workspace_root().join("target/variants/_target")
}

/// Lay a variant's source and manifest down without building it.
fn write_variant(kind: &Kind, name: &str, source: &str) -> anyhow::Result<PathBuf> {
    let root = workspace_root();
    let dir = root.join("target/variants").join(name);
    std::fs::create_dir_all(dir.join("src"))?;
    std::fs::write(dir.join("src/lib.rs"), source)?;
    for file in kind.beside {
        std::fs::copy(
            root.join(kind.dir).join("src").join(file),
            dir.join("src").join(file),
        )?;
    }
    let crates = root.join("crates").display().to_string().replace('\\', "/");
    let (lib, demo) = (kind.lib, kind.dir);
    // Every feature the original turns on by default, and no way to turn one
    // off: a variant is the crate, edited, not a smaller configuration of it.
    let features = match kind.features {
        [] => String::new(),
        names => format!(
            "[features]\ndefault = [{}]\n{}\n",
            names
                .iter()
                .map(|f| format!("\"{f}\""))
                .collect::<Vec<_>>()
                .join(", "),
            names
                .iter()
                .map(|f| format!("{f} = []"))
                .collect::<Vec<_>>()
                .join("\n")
        ),
    };
    std::fs::write(
        dir.join("Cargo.toml"),
        format!(
            "# Generated by `xtask reload` — a variant of {demo} (§6 M5 gates).\n\
             # Not checked in, not a workspace member, and rewritten on every run.\n\
             [workspace]\n\n\
             [package]\n\
             name = \"gg-variant-{name}\"\n\
             version = \"0.0.0\"\n\
             edition = \"2024\"\n\n\
             [lib]\n\
             name = \"{lib}\"\n\
             crate-type = [\"cdylib\"]\n\
             path = \"src/lib.rs\"\n\n\
             {features}\n\
             [dependencies]\n\
             bytemuck = {{ version = \"1\", features = [\"derive\"] }}\n\
             gg-ecs = {{ path = \"{crates}/gg-ecs\" }}\n\
             gg-math = {{ path = \"{crates}/gg-math\" }}\n\n\
             # The workspace's own dev profile, copied: the latency instrument \
             measures a\n# rebuild, and a rebuild at a different optimization \
             level is a different\n# number (§3 profiles).\n\
             [profile.dev]\n\
             opt-level = 1\n\
             [profile.dev.package.\"*\"]\n\
             opt-level = 3\n"
        ),
    )?;
    Ok(dir)
}

/// Build a variant already written, and return the artifact cargo produced.
fn build_variant(kind: &Kind, name: &str) -> anyhow::Result<PathBuf> {
    let manifest = workspace_root()
        .join("target/variants")
        .join(name)
        .join("Cargo.toml");
    exec(
        cargo().env("CARGO_TARGET_DIR", shared_target()).args([
            "build",
            "--manifest-path",
            &manifest.display().to_string(),
        ]),
        &format!("build game variant `{name}`"),
    )?;
    Ok(shared_target().join("debug").join(dylib_name(kind)))
}

// ---- the gates ----------------------------------------------------------

/// `cargo xtask reload` — every M5 gate that needs the shell driving a game
/// dylib, in one command. The nightly tier calls the same functions
/// individually; this exists so a human can run the set after touching the
/// boundary without waiting out a whole nightly.
pub fn gates(args: &[&str]) -> anyhow::Result<()> {
    let only = |name: &str| args.is_empty() || args.contains(&name);
    if only("--cross-tier") {
        cross_tier()?;
        dist_records()?;
    }
    if only("--segments") {
        segments()?;
    }
    if only("--chaos") {
        chaos_reload()?;
    }
    if only("--latency") {
        latency()?;
    }
    if only("--ui") {
        ui()?;
    }
    if only("--save") {
        save()?;
    }
    if only("--editor") {
        editor()?;
    }
    if only("--launcher") {
        launcher()?;
    }
    if only("--tetris") {
        tetris()?;
    }
    if only("--platformer") {
        platformer()?;
    }
    if only("--rules") {
        rules()?;
    }
    if only("--feel") {
        feel()?;
    }
    if only("--best") {
        best()?;
    }
    if only("--agent") {
        agent()?;
    }
    println!("xtask reload: green");
    Ok(())
}

/// The two points the latency instrument measures, as generated-system counts.
///
/// Each generated system is a real query over a real component, because compile
/// time is bought by *typed* code — a thousand `fn f() {}` would measure the
/// parser and nothing that grows with a game.
///
/// Two points rather than one because they answer different questions. The
/// **small** one is roughly twice demo 03 and is gated against M5's budget: it
/// is the size at which the cooperative loop is supposed to feel immediate, and
/// a regression there is a regression in the thesis. The **fat** one is four
/// times larger again and is gated only against a loose ceiling: its job is to
/// keep the *curve* visible, and asserting a 2 s budget at a size that has
/// already been measured past it would be a gate that is red by design.
const SMALL_SYSTEMS: usize = 25;
const FAT_SYSTEMS: usize = 100;

/// M5's exit budget: save → new behaviour.
const BUDGET_MS: u128 = 2_000;

/// What the fat point may cost before something has gone wrong with the
/// toolchain rather than with the project. Measured at ~3.2 s (2026-08-01).
const FAT_CEILING_MS: u128 = 10_000;

/// Attempts an absolute wall-clock gate gets before its best sample is the
/// verdict. See [`best_under`].
const WALL_CLOCK_TRIES: usize = 3;

/// Sample a wall-clock measurement up to `tries` times and hand back the best,
/// returning early the moment a sample fits `budget`. The fast tier's
/// best-of-five reasoning (`ci.rs`), applied to the other absolute clocks: on a
/// shared desk one slow sample means the machine was busy — an antivirus pass,
/// a parallel build — not that the subject got slower, and a gate that goes red
/// for a reason unrelated to the code under test is the flake class §5 budgets
/// at <0.5%. A subject over the bar on every attempt has actually missed it.
/// The verdict stays the caller's: this returns the number, the caller owns the
/// `ensure!` and its prose.
fn best_under(
    what: &str,
    budget: u128,
    tries: usize,
    mut sample: impl FnMut() -> anyhow::Result<u128>,
) -> anyhow::Result<u128> {
    let mut best = u128::MAX;
    for attempt in 1..=tries {
        best = best.min(sample()?);
        if best <= budget {
            return Ok(best);
        }
        if attempt < tries {
            println!(
                "xtask: {what} is over {budget} ms at attempt {attempt} of {tries} — sampling \
                 again (a busy desk is not the subject being slow)"
            );
        }
    }
    Ok(best)
}

/// Demo 03 with a game's worth of extra code bolted on (§6 M5 finding (e)).
///
/// The cost that grows with a project is the **game crate's** rebuild-and-link —
/// the engine is a compiled dependency and does not recompile — so the
/// instrument grows the one thing that matters and leaves everything else alone.
fn fat_source(source: &str, systems: usize) -> anyhow::Result<String> {
    let mut out = String::with_capacity(source.len() + systems * 400);
    out.push_str(source);
    out.push_str("\n// ---- generated by `xtask reload` (§6 M5 latency instrument) ----\n");
    for i in 0..systems {
        // Behind a condition that never holds: what this instrument buys is
        // *compile and link* time, and three hundred systems each sweeping the
        // world every tick would turn the measured run into a benchmark of the
        // instrument. The code is compiled in full either way — the condition is
        // a runtime value, so nothing here is folded away.
        out.push_str(&format!(
            "\n/// Generated system {i}.\npub fn generated_{i}(world: &mut GameWorld) {{\n    \
             if world.tick() != u64::MAX {{\n        return;\n    }}\n    let mut acc = \
             {i}u64;\n    let _ = world.each::<(&Cube,)>(|_, (cube,)| {{\n        acc = \
             acc.wrapping_add(cube.order ^ {i});\n        acc = acc.rotate_left({});\n    }});\n \
             let _ = world.each::<(&Shot,)>(|_, (shot,)| {{\n        acc = \
             acc.wrapping_mul(3).wrapping_add(u64::from(shot.ticks_left));\n    }});\n    if acc \
             == u64::MAX {{\n        world.log(log_level::TRACE, \"{i}\");\n    }}\n}}\n",
            i % 63
        ));
    }
    out.push_str("\n/// Runs every generated system, so none of them is dead code.\npub fn generated(world: &mut GameWorld) {\n");
    for i in 0..systems {
        out.push_str(&format!("    generated_{i}(world);\n"));
    }
    out.push_str("}\n");

    let anchor = "    systems: [restart, bootstrap, aim, walk, shoot, spawn, fade, present],";
    anyhow::ensure!(
        out.contains(anchor),
        "demo 03's systems list is not where this gate expected it"
    );
    Ok(out.replace(
        anchor,
        "    systems: [restart, bootstrap, aim, walk, shoot, spawn, fade, present, generated],",
    ))
}

/// §6 M5 finding (e): **reload latency, measured at a scale that is not a toy.**
///
/// Every latency number in this project so far came from a game crate of a few
/// hundred lines, and the thing that grows is not the one people assume. World
/// size grows the *snapshot*, which is a column memcpy and is not even paid on
/// the common reload — an edit that moves no schema is a pointer swap with no
/// snapshot at all. Game-crate code size grows the *rebuild and link*, which was
/// 80–90% of every measurement, and it is the only part with a real curve.
///
/// So the instrument is a synthetic fat game crate, edited one line at a time,
/// with the whole loop timed: the rebuild a save triggers, plus the swap the
/// host performs. The budget it is measured against is M5's < 2 s, and the two
/// halves are reported separately because they degrade for different reasons and
/// have different fixes (a faster linker versus a cheaper migration).
fn latency() -> anyhow::Result<()> {
    let small = best_under(
        "the small point's loop",
        BUDGET_MS,
        WALL_CLOCK_TRIES,
        || measure_loop("small", SMALL_SYSTEMS, BUDGET_MS),
    )?;
    anyhow::ensure!(
        small <= BUDGET_MS,
        "at roughly twice demo 03's size the loop takes {small} ms against M5's {BUDGET_MS} ms \
         budget. The fixes are compile-side and none of them is an architecture change (§6 M5): \
         split the game across several dylibs behind the same table, a dev-profile codegen \
         backend, a faster linker, or keep the fast-iteration systems in a small crate."
    );
    let fat = best_under(
        "the fat point's loop",
        FAT_CEILING_MS,
        WALL_CLOCK_TRIES,
        || measure_loop("fat", FAT_SYSTEMS, FAT_CEILING_MS),
    )?;
    anyhow::ensure!(
        fat <= FAT_CEILING_MS,
        "the fat point cost {fat} ms against a {FAT_CEILING_MS} ms ceiling — that is a toolchain \
         or machine event, not a project one"
    );
    Ok(())
}

/// Time one save → new-behaviour loop at a given game-crate size, in
/// milliseconds, and print the breakdown.
/// `budget` is the ceiling *this* point is judged against, passed in rather than
/// read from a constant: the fat point answers to `FAT_CEILING_MS`, and printing
/// M5's 2 s beside a number that is not measured against it reads as a gate that
/// forgives its own failure.
fn measure_loop(name: &str, systems: usize, budget: u128) -> anyhow::Result<u128> {
    let source = fat_source(&game_source(DEMO_03)?, systems)?;
    let lines = source.lines().count();
    let dir = workspace_root().join("target/latency").join(name);
    std::fs::create_dir_all(&dir)?;
    let variant_name = format!("latency-{name}");

    // Build it once, cold, and keep the artifact. This build is *not* the
    // measurement: a game crate compiled for the first time is not what a save
    // costs, and timing it would flatter or damn the loop by whichever the
    // machine's cache happened to hold.
    write_variant(DEMO_03, &variant_name, &source)?;
    let before = dir.join("before.bin");
    std::fs::copy(build_variant(DEMO_03, &variant_name)?, &before)?;

    // The edit: one line inside one system body, which is what M5's exit
    // criterion actually says and the cheapest thing a person types.
    let edited = source.replace(
        "pub const MOVE_PER_TICK: f64 = 0.08;",
        "pub const MOVE_PER_TICK: f64 = 0.09;",
    );
    anyhow::ensure!(edited != source, "the one-line edit changed nothing");
    write_variant(DEMO_03, &variant_name, &edited)?;

    // The rebuild a save triggers, timed on its own. Deliberately *not* measured
    // with the host running underneath: coupling the two would make the run's
    // frame budget a function of how long the compiler took, and a gate that
    // goes red because a build was slow in the wrong way is a gate nobody trusts
    // (§5's flake budget).
    let started = std::time::Instant::now();
    let after = build_variant(DEMO_03, &variant_name)?;
    let rebuild_ms = started.elapsed().as_millis();

    // The swap, measured the way the host measures it: from the file event to
    // the tick boundary the new code first runs at.
    exec(
        cargo().args(["build", "-p", "gg-runtime"]),
        "build the shell [dev]",
    )?;
    let game = dir.join(dylib_name(DEMO_03));
    std::fs::copy(&before, &game)?;
    let log = reload_midway(&game, &after, &["--frames", "60000"])?;
    let reloaded = log
        .lines()
        .find(|l| l.contains("game reloaded"))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "the rebuild produced no reload:
{log}"
            )
        })?;
    let swap_ms = u128::from(crate::util::field_u64(reloaded, "save_to_swap_ms")?);
    let total = rebuild_ms + swap_ms;
    println!(
        "xtask: reload latency, {name} ({lines} game-crate lines, {systems} generated systems): \
         rebuild {rebuild_ms} ms + swap {swap_ms} ms = {total} ms (budget {budget} ms)"
    );
    Ok(total)
}

/// A seeded chaos stream for demo 03 (§5.11), long enough to still be running
/// when a rewrite lands.
///
/// xorshift64\*, ours on purpose, for the reason demo 02's generator gives: a
/// crate's random number generator is not a stability guarantee, and a seed has
/// to mean the same stream in five years or a checked-in failure is not a
/// regression test.
fn chaos_stream(seed: u64, ticks: u64) -> Replay {
    let mut recorder = Recorder::new(meta());
    let mut state = seed | 1;
    let mut next = || {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    for tick in 0..ticks {
        let r = next();
        // Held for a stretch rather than re-rolled every tick: input that
        // changes every frame is not what a player does, and a stream of pure
        // noise compresses to one record per tick, which makes the file the
        // slowest part of the gate.
        if tick % 16 == 0 {
            let frame = InputFrame {
                buttons: r & 0b111,
                axes: axes(
                    (r >> 8) as i32 % 2 - 1,
                    0,
                    (r >> 16) as i32 % 2,
                    (r >> 24) as i32 % 33 - 16,
                    (r >> 32) as i32 % 17 - 8,
                ),
            };
            recorder.record(tick, frame);
        }
    }
    recorder.finish()
}

/// §5 gate 11, from M5: **the chaos generator exercises the reload path.**
///
/// Two cases, and they are the two halves of what a reload is allowed to do:
///
/// - **Reloading an identical dylib is hash-neutral.** The same seeded stream
///   run with and without a same-bytes rewrite mid-flight must produce the same
///   canonical hash sequence, tick for tick. This is the check that would catch
///   a swap leaking allocator state, dropping a tick, or rebuilding a column in
///   a different order — none of which a "did it not crash" test can see.
/// - **A schema-migration reload produces the documented result.** The swap onto
///   a build with one more field on `Cube` reports it by name, and the sequence
///   diverges *at the swap tick and not before* — a migration that started
///   changing state a tick early would be a silent corruption of everything
///   recorded before it.
///
/// Under a chaos stream rather than the curated one because the world has to be
/// busy when the swap lands: a reload into a quiet world is the easy case, and
/// the interesting failures are the ones involving live archetype churn.
fn chaos_reload() -> anyhow::Result<()> {
    const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
    // Set by wall time, not by the check: a headless tick is microseconds and
    // the watcher's debounce alone is 120 ms.
    const TICKS: u64 = 60_000;

    let source = game_source(DEMO_03)?;
    let before = variant(DEMO_03, "baseline", &source)?;
    let after = variant(DEMO_03, "migrated", &with_an_extra_field(&source)?)?;

    let dir = workspace_root().join("target/chaos-reload");
    std::fs::create_dir_all(&dir)?;
    let stream = dir.join("chaos.ggrp");
    std::fs::write(&stream, chaos_stream(SEED, TICKS).encode())?;
    let stream = stream.display().to_string();
    let game = dir.join(dylib_name(DEMO_03));

    exec(
        cargo().args(["build", "-p", "gg-runtime"]),
        "build the shell [dev]",
    )?;
    let host = exe("debug", "gg-runtime");

    std::fs::copy(&before, &game)?;
    let quiet = sequence(&play(&host, &game, &["--replay", &stream], true)?)?;
    anyhow::ensure!(
        quiet.iter().any(|(_, h)| h != &quiet[0].1),
        "the chaos stream never moved the world's hash — it would make both cases below vacuous"
    );

    // Case one: the same bytes, swapped in mid-flight.
    std::fs::copy(&before, &game)?;
    let log = reload_midway(&game, &before, &["--replay", &stream])?;
    let neutral = sequence(&log)?;
    if let Some(found) = divergence(
        &("no reload", quiet.clone()),
        &("identical reload", neutral),
    ) {
        anyhow::bail!("§5.11: reloading an identical dylib was not hash-neutral — {found}");
    }
    let reloaded = log
        .lines()
        .find(|l| l.contains("game reloaded"))
        .unwrap_or_default();
    let swap = crate::util::field_u64(reloaded, "tick")?;
    // The identical case must also be the *cheap* case: unchanged fingerprints
    // take §4.2.2's pointer swap — a snapshot/restore here would mean the 2 s
    // bar is graded on the slow path for every trivial edit.
    anyhow::ensure!(
        reloaded.contains("pointer-swap"),
        "an identical dylib did not take §4.2.2's schema-unchanged pointer swap: {reloaded}"
    );
    println!(
        "xtask: chaos reload — an identical dylib swapped in at tick {swap} changed nothing, \
         through the pointer-swap path"
    );

    // Case two: the migration.
    std::fs::copy(&before, &game)?;
    let log = reload_midway(&game, &after, &["--replay", &stream])?;
    let migrated = sequence(&log)?;
    let reloaded = log
        .lines()
        .find(|l| l.contains("game reloaded"))
        .unwrap_or_default();
    let swap = crate::util::field_u64(reloaded, "tick")?;
    // And the migrating case must be the careful one: a changed fingerprint may
    // never ride the pointer swap (§4.2.2).
    anyhow::ensure!(
        reloaded.contains("restore"),
        "a schema-changing dylib did not take the restore path: {reloaded}"
    );
    let report = log
        .lines()
        .find(|l| l.contains("migrated") && l.contains("demo03.cube"))
        .ok_or_else(|| {
            anyhow::anyhow!("the swap reported no migration of `demo03.cube`:\n{log}")
        })?;
    for expected in ["copied", "position", "defaulted", "wobble"] {
        anyhow::ensure!(
            report.contains(expected),
            "the migration report does not name `{expected}`: {report}"
        );
    }
    // Every component that did *not* move stays quiet — §4.2.2's one line per
    // component that actually changed, which is the difference between a report
    // and a wall of text.
    anyhow::ensure!(
        log.matches("migrated").count() == 1,
        "the swap reported {} migrations; only `demo03.cube` moved",
        log.matches("migrated").count()
    );

    let at = |seq: &[(u64, String)], tick: u64| seq.iter().find(|(t, _)| *t == tick).cloned();
    for tick in 0..swap {
        anyhow::ensure!(
            at(&migrated, tick) == at(&quiet, tick),
            "the migration changed tick {tick}, which ran before the swap at {swap}"
        );
    }
    anyhow::ensure!(
        at(&migrated, swap) != at(&quiet, swap),
        "tick {swap} is unchanged by a component gaining a field, so this gate is comparing a \
         sequence to itself"
    );
    println!(
        "xtask: chaos reload — a migrating dylib swapped in at tick {swap}: reported by name, \
         and the sequence moved there and not one tick earlier"
    );
    Ok(())
}

/// A dev shell playing `game`, with the artifact rewritten underneath it partway
/// through. Returns the whole log.
fn reload_midway(game: &Path, replacement: &Path, args: &[&str]) -> anyhow::Result<String> {
    reload_after(game, replacement, args, crate::util::READY, 400)
}

/// [`reload_midway`] with the moment named: the rewrite lands `nudge_ms` after
/// the child first logs a line containing `marker`. [`crate::util::READY`] is
/// the earliest legal marker — the watcher exists from that line — and a gate
/// that needs the swap near a *tick* passes that tick's own hash line instead,
/// which prices nothing in wall time (see [`rules`]).
fn reload_after(
    game: &Path,
    replacement: &Path,
    args: &[&str],
    marker: &str,
    nudge_ms: u64,
) -> anyhow::Result<String> {
    reload_after_env(game, replacement, args, &[], marker, nudge_ms)
}

/// [`reload_after`] with extra environment for the child — what a leg that
/// also reads the seam record passes `GG_AGENT_DIR` through.
fn reload_after_env(
    game: &Path,
    replacement: &Path,
    args: &[&str],
    envs: &[(&str, &Path)],
    marker: &str,
    nudge_ms: u64,
) -> anyhow::Result<String> {
    use std::process::Stdio;

    exec(
        cargo().args(["build", "-p", "gg-runtime"]),
        "build the shell [dev]",
    )?;
    let mut cmd = Command::new(exe("debug", "gg-runtime"));
    cmd.arg("--game")
        .arg(game)
        .args(args)
        .env("GG_HEADLESS", "1")
        .env("RUST_LOG", "info,gg::hash=debug")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in envs {
        cmd.env(key, value);
    }
    let mut child = cmd.spawn()?;
    let (log, status) = rewrite_once_watching(&mut child, replacement, game, marker, nudge_ms)?;
    anyhow::ensure!(status.success(), "the shell exited {status}\n{log}");
    anyhow::ensure!(
        log.contains("game reloaded"),
        "the rewrite did not produce a reload:\n{log}"
    );
    Ok(log)
}

/// Wait for the child to log a line containing `marker`, then `nudge_ms` more
/// to place the swap, rewrite `game` from `replacement`, and drain the child to
/// exit. Returns `(log, status)`.
///
/// The wait is anchored at the child's own output, never at spawn: before
/// `Watch::new` a file event is a trigger nobody is holding — the flake the
/// rejuvenation-gate correction retired, which a delay-from-spawn re-creates on
/// any machine still loading at the deadline. Every marker a caller passes is
/// at or after [`crate::util::READY`], so the watcher exists by the time the
/// copy lands; whatever follows the marker is placement only, and a busy
/// machine can only move it later into the run.
fn rewrite_once_watching(
    child: &mut std::process::Child,
    replacement: &Path,
    game: &Path,
    marker: &str,
    nudge_ms: u64,
) -> anyhow::Result<(String, std::process::ExitStatus)> {
    let (tx, rx) = std::sync::mpsc::channel();
    let stdout = crate::util::drain(child.stdout.take(), marker.to_owned(), tx.clone());
    let stderr = crate::util::drain(child.stderr.take(), marker.to_owned(), tx);
    // Generous because it measures nothing: the difference between a failure
    // and a hang.
    if rx.recv_timeout(std::time::Duration::from_secs(60)).is_err() {
        let _ = child.kill();
        anyhow::bail!(
            "the shell never logged `{marker}`, so the rewrite would have gone to nobody:\n{}{}",
            stdout.join().unwrap_or_default(),
            stderr.join().unwrap_or_default()
        );
    }
    if nudge_ms > 0 {
        std::thread::sleep(std::time::Duration::from_millis(nudge_ms));
    }
    std::fs::copy(replacement, game)?;
    let status = child.wait()?;
    let log = format!(
        "{}{}",
        stdout.join().unwrap_or_default(),
        stderr.join().unwrap_or_default()
    );
    Ok((log, status))
}

/// §6 M16: **the seam's outcome is a record, and a refusal names itself in it.**
///
/// The outcome has always been a log line — legible to a human watching a
/// terminal and to nothing else. Two legs, failing in opposite directions.
///
/// The **migration** proves the record moves. An added field is in the canonical
/// hash (§4.5), so `state_before` and `state_after` must differ; a gate that
/// only checked "a seam was recorded" would pass against a record that wrote one
/// hash twice, which is the shape this would rot into.
///
/// The **refusal** proves it survives the path with nothing to describe. Garbage
/// over the artifact is refused before a `Reloaded` exists at all — no timings,
/// no second code hash — and it must still arrive named `Open` rather than as an
/// empty seam or no seam. That is the branch M16's exit row is actually about: a
/// refusal in a log file is the thing being replaced.
fn agent() -> anyhow::Result<()> {
    let source = game_source(DEMO_03)?;
    let before = variant(DEMO_03, "agent-baseline", &source)?;
    let after = variant(DEMO_03, "agent-migrated", &with_an_extra_field(&source)?)?;

    let dir = workspace_root().join("target/agent-gate");
    std::fs::create_dir_all(&dir)?;
    let game = dir.join(dylib_name(DEMO_03));
    let records = dir.join("record");

    std::fs::copy(&before, &game)?;
    let (log, record) = agent_session(&game, &after, &records)?;
    anyhow::ensure!(
        log.contains("game reloaded"),
        "the rewrite did not produce a reload:\n{log}"
    );
    for expected in [
        "\"outcome\": \"accepted\"",
        "\"within_budget\": true",
        "\"kind\": \"migrated\"",
        "\"wobble\"",
        "demo03.cube",
    ] {
        anyhow::ensure!(
            record.contains(expected),
            "the accepted seam is missing {expected}:\n{record}"
        );
    }
    let (state_before, state_after) = (
        json_field(&record, "state_before")?,
        json_field(&record, "state_after")?,
    );
    anyhow::ensure!(
        state_before != state_after,
        "the migration left the state hash where it found it ({state_before}) — a field the \
         running world has never seen is *in* the canonical hash (§4.5), so a record that reports \
         it unmoved is reporting a migration that did not happen:\n{record}"
    );
    anyhow::ensure!(
        json_field(&record, "code_before")? != json_field(&record, "code_after")?,
        "the record names one build on both sides of the seam:\n{record}"
    );
    println!("xtask reload --agent: the accepted seam moved the state hash and named both builds");

    // Not a build at all. The watcher stages it, `libloading` refuses it, and
    // the refusal reaches the record through the arm that has no `Reloaded` to
    // read timings off — which is the arm worth a gate.
    std::fs::copy(&before, &game)?;
    let garbage = dir.join("not-a-dylib.bin");
    std::fs::write(&garbage, b"this is not a dynamic library")?;
    let (log, record) = agent_session(&game, &garbage, &records)?;
    anyhow::ensure!(
        log.contains("reload refused") && log.contains("still running the last good build"),
        "garbage over the artifact did not produce a refusal:\n{log}"
    );
    anyhow::ensure!(
        record.contains("\"outcome\": \"refused\"") && record.contains("\"refusal\": \"Open\""),
        "the refusal did not reach the record by name:\n{record}"
    );
    anyhow::ensure!(
        json_field(&record, "state_before")? == json_field(&record, "state_after")?,
        "a refused swap moved the state hash — nothing was adopted, so nothing may have \
         moved:\n{record}"
    );
    // Zero is a measurement and `null` is the absence of one. A record that
    // reported `0 ms` here would make the fastest reload of the session the one
    // that did not happen, and `within_budget: true` would grade §9's bar
    // against it.
    for unknown in [
        "\"code_after\": null",
        "\"load_ms\": null",
        "\"save_to_swap_ms\": null",
        "\"within_budget\": null",
    ] {
        anyhow::ensure!(
            record.contains(unknown),
            "the refusal reported a value where it measured nothing — expected {unknown}:\n{record}"
        );
    }
    println!("xtask reload --agent: the refusal reached the record as `Open`, hash unmoved");
    agent_loop(&before, &after)
}

/// §6 M16 exit row 4: **§1's loop, demonstrated on demo 03 as a recorded
/// session that replays.**
///
/// The session is the checked-in [`AGENT_CURATED`] stream: play on demo 03's
/// own verbs, bring the agent pane up, type the ask through the text channel,
/// click send, keep playing. Under `GG_HEADLESS=1` the send lands and
/// `gg_agent::Ask::spawn` refuses by design (§6 M16 item 3) — the click is in
/// the replay, the paid call is not, and the tick hash stays a function of the
/// world. The rebuild is the gate's to land: anchored on the `editor: agent
/// asked` line itself, so the swap follows the ask the way §1 orders them.
///
/// Three claims, in the order they can fail:
/// - the recorded session **replays** — two undisturbed runs agree tick for
///   tick, editor, prompt and all;
/// - the loop **closes with state intact** — the migrating build swaps in
///   after the ask, `wobble` arrives by name, every pre-swap tick is
///   untouched, and the sequence moves at the swap tick and not before;
/// - the crossing is **graded** — the seam record reports the migration
///   accepted and inside §9's two-second bar.
fn agent_loop(before: &Path, after: &Path) -> anyhow::Result<()> {
    let stream_path = agent_replay_path();
    let stream = Replay::decode(&std::fs::read(&stream_path)?)?;
    anyhow::ensure!(
        stream.typed_count() > 0,
        "the loop's session carries no typed text — the ask would be a click on an empty field"
    );
    let stream = stream_path.display().to_string();

    let dir = workspace_root().join("target/agent-loop");
    std::fs::create_dir_all(&dir)?;
    let game = dir.join(dylib_name(DEMO_03));

    exec(
        cargo().args(["build", "-p", "gg-runtime"]),
        "build the shell [dev]",
    )?;
    let host = exe("debug", "gg-runtime");

    // The replay claim first: same clicks, same prompt, same hashes, twice.
    std::fs::copy(before, &game)?;
    let quiet_log = play(&host, &game, &["--replay", &stream, "--editor"], true)?;
    let asked = quiet_log
        .lines()
        .find(|l| l.contains("editor: agent asked"))
        .ok_or_else(|| anyhow::anyhow!("the session never asked the panel:\n{quiet_log}"))?;
    let ask_tick = crate::util::field_u64(asked, "tick")?;
    let quiet = sequence(&quiet_log)?;
    anyhow::ensure!(
        quiet.iter().any(|(_, h)| h != &quiet[0].1),
        "the session never moved the world's hash — the loop below would close over nothing"
    );
    std::fs::copy(before, &game)?;
    let again = sequence(&play(
        &host,
        &game,
        &["--replay", &stream, "--editor"],
        true,
    )?)?;
    if let Some(found) = divergence(&("the session", quiet.clone()), &("its replay", again)) {
        anyhow::bail!("§6 M16: the recorded session did not replay — {found}");
    }

    // Then the loop: the rebuild lands after the ask, and the session carries
    // its world across the seam.
    std::fs::copy(before, &game)?;
    let log = reload_after_env(
        &game,
        after,
        &["--replay", &stream, "--editor"],
        &[("GG_AGENT_DIR", dir.as_path())],
        "editor: agent asked",
        0,
    )?;
    let reloaded = log
        .lines()
        .find(|l| l.contains("game reloaded"))
        .unwrap_or_default();
    let swap = crate::util::field_u64(reloaded, "tick")?;
    anyhow::ensure!(
        swap > ask_tick,
        "the rebuild landed at tick {swap}, before the ask at {ask_tick} — the anchor is not \
         anchoring"
    );
    anyhow::ensure!(
        reloaded.contains("restore"),
        "a schema-changing dylib did not take the restore path: {reloaded}"
    );
    anyhow::ensure!(
        log.lines()
            .any(|l| l.contains("migrated") && l.contains("demo03.cube") && l.contains("wobble")),
        "the ask's wobble never arrived by name:\n{log}"
    );
    let looped = sequence(&log)?;
    let at = |seq: &[(u64, String)], tick: u64| seq.iter().find(|(t, _)| *t == tick).cloned();
    for tick in 0..swap {
        anyhow::ensure!(
            at(&looped, tick) == at(&quiet, tick),
            "the reload changed tick {tick}, which ran before the swap at {swap} — state was not \
             intact across the seam"
        );
    }
    anyhow::ensure!(
        at(&looped, swap) != at(&quiet, swap),
        "tick {swap} is unchanged by a component gaining a field — the loop closed over nothing"
    );

    // And the grade: the crossing, as the record the panel reads (§6 M16 items
    // 1–2), inside §9's bar.
    let record = std::fs::read_to_string(dir.join("session.json"))?;
    for expected in [
        "\"outcome\": \"accepted\"",
        "\"kind\": \"migrated\"",
        "\"wobble\"",
        "\"within_budget\": true",
    ] {
        anyhow::ensure!(
            record.contains(expected),
            "the loop's seam record is missing {expected}:\n{record}"
        );
    }
    anyhow::ensure!(
        json_field(&record, "state_before")? != json_field(&record, "state_after")?,
        "the record reports the migration without the state moving:\n{record}"
    );
    println!(
        "xtask reload --agent: the loop closed — asked at tick {ask_tick}, rebuilt into the \
         session at {swap}, state intact to the seam and moved at it, inside §9's bar"
    );
    Ok(())
}

/// A dev shell over `game` with `replacement` written under it partway through,
/// publishing its record into `dir`. Returns `(log, record)`.
///
/// Deliberately does not assert a reload happened: half this gate's point is the
/// run where one does *not*.
fn agent_session(game: &Path, replacement: &Path, dir: &Path) -> anyhow::Result<(String, String)> {
    use std::process::Stdio;

    exec(
        cargo().args(["build", "-p", "gg-runtime"]),
        "build the shell [dev]",
    )?;
    let _ = std::fs::remove_dir_all(dir);
    let mut cmd = Command::new(exe("debug", "gg-runtime"));
    cmd.arg("--game")
        .arg(game)
        // Frame count set by wall time, not by ticks needed: the watcher's
        // debounce alone is 120 ms (see `segments`).
        .args(["--frames", "60000"])
        .env("GG_HEADLESS", "1")
        .env("GG_AGENT_DIR", dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = cmd.spawn()?;
    let (log, status) =
        rewrite_once_watching(&mut child, replacement, game, crate::util::READY, 400)?;
    anyhow::ensure!(status.success(), "the shell exited {status}\n{log}");
    let record = std::fs::read_to_string(dir.join("session.json"))?;
    Ok((log, record))
}

/// One `"key": "value"` out of the record, unquoted.
///
/// A reader rather than a parser: the record's shape is this tree's own and the
/// gate wants two strings out of it. A gate that pulled in a JSON stack to
/// compare two hashes would be renting a dependency to read its own file.
fn json_field(record: &str, key: &str) -> anyhow::Result<String> {
    let needle = format!("\"{key}\": \"");
    let at = record
        .find(&needle)
        .ok_or_else(|| anyhow::anyhow!("the record has no `{key}`:\n{record}"))?;
    let rest = &record[at + needle.len()..];
    let end = rest
        .find('"')
        .ok_or_else(|| anyhow::anyhow!("`{key}` is unterminated:\n{record}"))?;
    Ok(rest[..end].to_owned())
}

/// §6 M5: **replay segments close and open across a reload, and the pre-reload
/// segment replays deterministically.**
///
/// The reload is a *migration* — the replacement build has one more field on a
/// hashed component — which is what makes the second half a real comparison
/// rather than a tautology. A headless run has no hands, so demo 03's world only
/// moves when its code does; a same-bytes reload would leave the hash sequence
/// flat on both sides of the swap and "the pre-reload segment reproduces" would
/// be true of any two runs whatsoever.
///
/// So the gate asserts both directions:
///
/// - replaying the recorded stream under the **old** build reproduces the
///   pre-reload ticks exactly, and
/// - it does **not** reproduce the post-reload ticks, because those were
///   produced by code this replay is not running.
///
/// The second is what proves the first is load-bearing. A segment that named the
/// wrong build would be invisible to the first check alone.
fn segments() -> anyhow::Result<()> {
    let source = game_source(DEMO_03)?;
    let before = variant(DEMO_03, "baseline", &source)?;
    let after = variant(DEMO_03, "migrated", &with_an_extra_field(&source)?)?;

    // Its own directory: the watcher watches a *directory*, so rewriting an
    // artifact under `target/debug` would be a reload event for anything else
    // pointed there.
    let dir = workspace_root().join("target/segments");
    std::fs::create_dir_all(&dir)?;
    let game = dir.join(dylib_name(DEMO_03));
    std::fs::copy(&before, &game)?;
    let recorded = dir.join("across-a-reload.ggrp");

    let log = reload_midway(
        &game,
        &after,
        // Long enough to still be running when the rewrite lands: a headless
        // tick is microseconds and the watcher's debounce alone is 120 ms, so
        // the frame count is set by *wall time* rather than by how many ticks
        // the comparison below needs. Sixty thousand is about three
        // seconds here, and a slower machine only makes the window wider.
        &[
            "--frames",
            "60000",
            "--record",
            &recorded.display().to_string(),
        ],
    )?;
    let swap = crate::util::field_u64(
        log.lines()
            .find(|l| l.contains("game reloaded"))
            .unwrap_or_default(),
        "tick",
    )?;
    // The migration itself, reported by name — the same line a human sees when
    // an agent adds a field mid-play (§6 M5).
    anyhow::ensure!(
        log.contains("migrated") && log.contains("demo03.cube"),
        "the swap reported no migration of `demo03.cube`:\n{log}"
    );

    let replay = Replay::decode(&std::fs::read(&recorded)?)?;
    let seg = replay.segments();
    anyhow::ensure!(
        seg.len() == 2,
        "recorded {} segment(s) across one reload, expected 2: {seg:?}",
        seg.len()
    );
    anyhow::ensure!(
        seg[0].first_tick == 0,
        "segment 0 starts at {}",
        seg[0].first_tick
    );
    anyhow::ensure!(
        seg[1].first_tick == swap,
        "the segment opened at tick {} and the swap happened at {swap}",
        seg[1].first_tick
    );
    anyhow::ensure!(
        seg[0].code_hash != seg[1].code_hash && seg[0].code_hash != 0,
        "both segments name the same build, so a stream recorded across a reload cannot say \
         which code produced which ticks: {seg:?}"
    );

    // Replay the whole recording under the *old* build.
    let recorded_hashes = sequence(&log)?;
    let again = play(
        &exe("debug", "gg-runtime"),
        &before,
        &["--replay", &recorded.display().to_string()],
        true,
    )?;
    let replayed = sequence(&again)?;
    let at = |seq: &[(u64, String)], tick: u64| {
        seq.iter().find(|(t, _)| *t == tick).map(|(_, h)| h.clone())
    };
    for tick in 0..swap {
        anyhow::ensure!(
            at(&recorded_hashes, tick) == at(&replayed, tick),
            "the pre-reload segment did not reproduce: tick {tick} was {:?} when recorded and \
             {:?} when replayed under the build segment 0 names",
            at(&recorded_hashes, tick),
            at(&replayed, tick)
        );
    }
    anyhow::ensure!(
        at(&recorded_hashes, swap) != at(&replayed, swap),
        "tick {swap} reproduced under the *old* build, so the migration changed nothing the hash \
         can see and this gate is comparing a constant to itself"
    );
    println!(
        "xtask: replay segments — segment 0 (build {:032x}) covers ticks 0..{swap} and replays \
         exactly; segment 1 (build {:032x}) opens at the swap and does not",
        seg[0].code_hash, seg[1].code_hash
    );
    Ok(())
}

/// §5.6c in full, and the M5 exit criterion that carries it: the same curated
/// stream, replayed by the shell under three codegen configurations, must
/// produce one canonical hash sequence.
///
/// dist-verify is what makes this mean anything (§2): dist proper compiles the
/// hash out, so without it "replayed under dev, hashes identical" would have
/// nothing on the dist side to be identical *to*. Instrumented joins because
/// thin-LTO-at-full-optimization is the one codegen no hash ever measured, and
/// it is the tier a bug replay gets profiled under.
pub fn cross_tier() -> anyhow::Result<()> {
    let replay = curated_path();
    anyhow::ensure!(
        replay.is_file(),
        "no curated stream at {} — `cargo xtask replay --bless` authors it",
        replay.display()
    );
    let replay = replay.display().to_string();

    let mut runs: Vec<(&str, Vec<(u64, String)>)> = Vec::new();
    for tier in HASHED_TIERS {
        let (host, game) = stage(tier)?;
        let log = play(&host, &game, &["--replay", &replay], true)?;
        let seq = sequence(&log)?;
        println!(
            "xtask: {} replayed {} ticks, ending {}",
            tier.name,
            seq.len(),
            seq.last().map(|(_, h)| h.as_str()).unwrap_or("?")
        );
        runs.push((tier.name, seq));
    }

    // A constant sequence would satisfy "all identical" while proving nothing,
    // which is exactly the trap an input-free recorded stream would have walked
    // into. The curated stream is supposed to move the world.
    let first = &runs[0].1;
    anyhow::ensure!(
        first.iter().any(|(_, h)| h != &first[0].1),
        "the curated stream never changed the world's hash — it proves nothing about determinism"
    );

    for pair in runs.windows(2) {
        if let Some(found) = divergence(&pair[0], &pair[1]) {
            anyhow::bail!("§5.6c: {found}");
        }
    }
    println!(
        "xtask: §5.6c green — dev, instrumented and dist-verify agree tick for tick over {} ticks",
        first.len()
    );
    Ok(())
}

/// §6 M13's end-to-end criterion: **a replayed click lands on the same widget
/// every run**, proven through the shell rather than in a unit test.
///
/// The demo's own `tests/game.rs` drives `gg_ui::Ui` over a world in one
/// process and covers the window-size half. What only this can cover is the
/// wiring: the four verb names resolving out of a *loaded dylib*, the shell
/// running the UI on the tick, and `Widget::state` reaching the canonical hash.
/// So the gate has two halves and both are load-bearing —
///
/// - **the clicks landed**: the session's log lines appear, in order. A pointer
///   that hit a neighbouring button writes different lines, and a UI that never
///   routed at all writes none.
/// - **and they landed identically under three codegens**: dev, instrumented
///   and dist-verify agree tick for tick, which is §5.6c's machinery applied to
///   a stream whose hash is *made of* hit state.
fn ui() -> anyhow::Result<()> {
    let replay = ui_path();
    anyhow::ensure!(
        replay.is_file(),
        "no UI stream at {} — `cargo xtask replay --bless` authors it",
        replay.display()
    );
    let replay = replay.display().to_string();

    let mut runs: Vec<(&str, Vec<(u64, String)>)> = Vec::new();
    for tier in HASHED_TIERS {
        let (host, game) = stage_game(tier, "demo-07-ui", "demo_07_ui")?;
        let log = play(&host, &game, &["--replay", &replay], true)?;
        // The shell found the verbs at all. Without this a game that declared
        // none would route nothing, the world would still hash consistently,
        // and three tiers would agree about a UI nobody touched.
        anyhow::ensure!(
            log.contains("ui=true"),
            "the shell did not bind demo 07's UI verbs [{}]:\n{log}",
            tier.name
        );
        let mut at = 0;
        for line in log.lines() {
            if at < demo_07_ui::SESSION_LOG.len() && line.contains(demo_07_ui::SESSION_LOG[at]) {
                at += 1;
            }
        }
        anyhow::ensure!(
            at == demo_07_ui::SESSION_LOG.len(),
            "[{}] the replayed session reached {at} of {} settings changes — expected {:?} in \
             order; a click that landed on a different widget is what this looks like:\n{log}",
            tier.name,
            demo_07_ui::SESSION_LOG.len(),
            demo_07_ui::SESSION_LOG,
        );
        runs.push((tier.name, sequence(&log)?));
    }

    for pair in runs.windows(2) {
        if let Some(found) = divergence(&pair[0], &pair[1]) {
            anyhow::bail!("§6 M13: {found}");
        }
    }
    println!(
        "xtask reload: demo 07's replayed session landed the same {} clicks under dev, \
         instrumented and dist-verify, tick for tick over {} ticks (§6 M13)",
        demo_07_ui::SESSION_LOG.len(),
        runs[0].1.len(),
    );
    Ok(())
}

/// §6 M18's exit row through the shell: **a recorded full game — first spawn to
/// top-out — replays bit-identically under dev, instrumented and dist-verify.**
///
/// The demo's own tests drive the same script in one process and hold the
/// checked-in baseline, which is what the aarch64 leg compares against. Three
/// things only this can add, and each fails differently:
///
/// - **the game was played.** [`session::LOG`](demo_10_tetris::session::LOG) in
///   order, and the top-out exactly once — a shell that replayed onto the wrong
///   verb ids would still hash consistently across three tiers while the board
///   sat untouched, and a stream that restarted would say "ready" twice.
/// - **and it was the recorded game.** The shell's hash *values* are not the
///   baseline's and cannot be: the shell's world also carries what `gg_ui` wrote
///   into `Widget::state`, and demo 10 puts ~250 widgets through it every tick.
///   What does cross the two harnesses is the **shape** of the sequence — the
///   ticks on which nothing moved. A shell replaying onto the wrong verb ids, or
///   an input path that stopped delivering, is quiet on different ticks. Weaker
///   than value equality, and the honest tie available: making it value equality
///   would mean linking `gg-ui` into the qemu leg to reproduce the host's writes,
///   which is the one thing §3 says a game crate's graph must not contain.
/// - **and codegen did not touch it.** dev, instrumented and dist-verify, tick
///   for tick (§5.6c).
fn tetris() -> anyhow::Result<()> {
    let replay = tetris_path();
    anyhow::ensure!(
        replay.is_file(),
        "no Tetris stream at {} — `cargo xtask replay --bless` authors it",
        replay.display()
    );
    let replay = replay.display().to_string();
    let baseline = demo_10_tetris::session::parse_baseline(&std::fs::read_to_string(
        demo_10_tetris::session::baseline_path(),
    )?)
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    let wanted = demo_10_tetris::session::LOG;

    let mut runs: Vec<(&str, Vec<(u64, String)>)> = Vec::new();
    for tier in HASHED_TIERS {
        let (host, game) = stage_game(tier, "demo-10-tetris", "demo_10_tetris")?;
        let log = play(&host, &game, &["--replay", &replay], true)?;
        let at = reaches(&log, wanted);
        anyhow::ensure!(
            at == wanted.len(),
            "[{}] the replayed game reached {at} of {} milestones — expected {wanted:?} in order; \
             a session that spawned and died without being played looks exactly like this:\n{log}",
            tier.name,
            wanted.len(),
        );
        for (once, what) in [
            ("tetris: ready", "started"),
            ("tetris: topped out", "ended"),
        ] {
            let count = log.lines().filter(|l| l.contains(once)).count();
            anyhow::ensure!(
                count == 1,
                "[{}] the session {what} {count} times — one recorded game, one of each",
                tier.name
            );
        }

        let seq = sequence(&log)?;
        anyhow::ensure!(
            seq.len() == baseline.len(),
            "[{}] the shell ran {} ticks and the baseline holds {}",
            tier.name,
            seq.len(),
            baseline.len()
        );
        let (here, there) = (
            quiet_ticks(seq.iter().map(|(_, h)| h.as_str())),
            quiet(&baseline),
        );
        anyhow::ensure!(
            here == there,
            "[{}] the shell's world stood still on {here:?} and the in-process baseline's on \
             {there:?} — the two harnesses did not play the same game",
            tier.name,
        );
        runs.push((tier.name, seq));
    }

    for pair in runs.windows(2) {
        if let Some(found) = divergence(&pair[0], &pair[1]) {
            anyhow::bail!("§6 M18: {found}");
        }
    }
    println!(
        "xtask reload: demo 10's recorded game — spawn to top-out over {} ticks — replayed \
         identically under dev, instrumented and dist-verify, standing still on the same ticks \
         as the baseline the aarch64 leg compares against (§6 M18)",
        runs[0].1.len(),
    );
    Ok(())
}

/// §6 M20's exit row through the shell: **a recorded full run — restart, a real
/// death, spawn to goal — replays bit-identically under dev, instrumented and
/// dist-verify**, over the checked-in scene.
///
/// The tetris gate's three claims, plus the one only this demo can make: the
/// scene crossed as **data**. The stream is replayed with `--load` because
/// record/replay never probe (§6 M20 item 3), so a run that reaches the goal
/// proves the level arrived from the file — a session that opened on a bare
/// world has nothing to stand on and its player is dead in the first two
/// seconds, which "platformer: fell" happening *exactly once* rules out.
fn platformer() -> anyhow::Result<()> {
    let replay = platformer_path();
    anyhow::ensure!(
        replay.is_file(),
        "no platformer stream at {} — `cargo xtask replay --bless` authors it",
        replay.display()
    );
    let replay = replay.display().to_string();
    let scene = platformer_scene()?;
    let baseline = demo_11_platformer::session::parse_baseline(&std::fs::read_to_string(
        demo_11_platformer::session::baseline_path(),
    )?)
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    let wanted = demo_11_platformer::session::LOG;

    // `--frames` bounds the run to the baseline's ticks: the stream's tick
    // span counts from 0 but the scene resumes at its own tick, and the tick
    // past the last recorded frame would be one idle tick nobody blessed.
    let ticks = baseline.len().to_string();

    let mut runs: Vec<(&str, Vec<(u64, String)>)> = Vec::new();
    for tier in HASHED_TIERS {
        let (host, game) = stage_game(tier, "demo-11-platformer", "demo_11_platformer")?;
        let log = play(
            &host,
            &game,
            &["--replay", &replay, "--load", &scene, "--frames", &ticks],
            true,
        )?;
        let at = reaches(&log, wanted);
        anyhow::ensure!(
            at == wanted.len(),
            "[{}] the replayed run reached {at} of {} milestones — expected {wanted:?} in order; \
             a session that spawned beside the goal, or one handed no scene, looks exactly like \
             a shorter prefix of this:\n{log}",
            tier.name,
            wanted.len(),
        );
        // Each milestone exactly once: a second "fell" is a run that lost the
        // level, a second "goal" is a clock that failed to stop, a second
        // "restarted" is a stream this recording never held.
        for once in wanted {
            let count = log.lines().filter(|l| l.contains(once)).count();
            anyhow::ensure!(
                count == 1,
                "[{}] `{once}` appears {count} times — one recorded run, one of each",
                tier.name
            );
        }

        let seq = sequence(&log)?;
        anyhow::ensure!(
            seq.len() == baseline.len(),
            "[{}] the shell ran {} ticks and the baseline holds {}",
            tier.name,
            seq.len(),
            baseline.len()
        );
        let (here, there) = (
            quiet_ticks(seq.iter().map(|(_, h)| h.as_str())),
            quiet(&baseline),
        );
        anyhow::ensure!(
            here == there,
            "[{}] the shell's world stood still on {here:?} and the in-process baseline's on \
             {there:?} — the two harnesses did not play the same run",
            tier.name,
        );
        runs.push((tier.name, seq));
    }

    for pair in runs.windows(2) {
        if let Some(found) = divergence(&pair[0], &pair[1]) {
            anyhow::bail!("§6 M20: {found}");
        }
    }
    println!(
        "xtask reload: demo 11's recorded run — restart, one death, spawn to goal over {} ticks, \
         the level arriving as `--load` data — replayed identically under dev, instrumented and \
         dist-verify, standing still on the same ticks as the baseline the aarch64 leg compares \
         against (§6 M20)",
        runs[0].1.len(),
    );
    Ok(())
}

/// Ticks of the endless stream the rule change is measured over. The swap is
/// aimed at a tick ([`RULES_SWAPS`]) and lands it plus however far the replay
/// raced while the sighting crossed a pipe, so the margin is bought in ticks: a
/// headless replay runs at thousands of ticks a second, and the gate asserts
/// the first half rather than assuming it.
const RULES_TICKS: usize = 8_000;

/// Ticks to aim the swap at — the rewrite lands when the shell's own hash line
/// for that tick is seen, so no wall time is priced (this gate failed the day
/// the trigger was re-anchored, 400 ms of full-speed replay being half the
/// stream). More than one because a swap landing on a clearing tick is
/// uninformative — see the loop in [`rules`].
const RULES_SWAPS: [u64; 3] = [1_200, 2_000, 2_800];

/// The schema edit: `Best` gains a field.
///
/// Two edits, because a field is not just a declaration — the literal that
/// builds one has to name it too, which is exactly what a person types.
fn with_a_wider_record(source: &str) -> anyhow::Result<String> {
    let edits = [
        (
            "    /// Points, over every game this world has played.\n    pub score: u32,\n",
            "    /// Points, over every game this world has played.\n    pub score: u32,\n    \
             /// Added by the save gate: a field the saved world never had, so\n    /// loading it \
             is a migration rather than a copy (§4.5).\n    pub lines: u32,\n",
        ),
        (
            "        Best {\n            score: 0,\n            top: [0; 5],\n        },",
            "        Best {\n            score: 0,\n            lines: 0,\n            top: [0; 5],\n        },",
        ),
    ];
    let mut out = source.to_owned();
    for (anchor, replacement) in edits {
        anyhow::ensure!(
            out.contains(anchor),
            "demo 10's `Best` no longer contains `{}` — the save gate's migration is a text edit, \
             so a rename here is a gate to re-point rather than one that quietly stops migrating \
             anything",
            anchor.trim()
        );
        out = out.replace(anchor, replacement);
    }
    Ok(out)
}

/// The losing edit: the record under a different name, which is what a build
/// that stopped declaring a component looks like to a save.
fn with_the_record_renamed(source: &str) -> anyhow::Result<String> {
    let anchor = "#[component(id = \"tetris.best\")]";
    anyhow::ensure!(
        source.contains(anchor),
        "demo 10's record no longer carries `{anchor}` — this is the gate's *losing* build and a \
         missed anchor would make it an ordinary one"
    );
    Ok(source.replace(anchor, "#[component(id = \"tetris.record\")]"))
}

/// The rule edit: what clearing a row is worth.
///
/// Chosen for what it does **not** do. Gravity, lock delay and DAS all change
/// the very next tick, and a sequence that moves the instant the swap lands
/// cannot tell a world that survived the swap from one that was rebuilt around
/// it — the hashes differ either way. Scoring is inert until the next row goes,
/// so the swapped run *keeps matching* the unswapped one across the boundary,
/// and every tick it matches is a tick the whole board — stack, bag, falling
/// piece, counters — crossed a different build untouched.
///
/// It is also the edit that shows why `Rules::DEFAULT` is not the knob here.
/// That constant is read once, at `bootstrap`; the component it produced is
/// already in the world, and a build that changes it changes nothing about a
/// game in progress. What a reload can change is *code that runs every tick* —
/// and demo 10 has three kinds of constant, which the gate was proven against by
/// pointing this function at each in turn: `EMPTY` (a colour `present` writes
/// every tick) parts the runs at the swap tick and trips the *earlier* branch
/// below, which is the branch a stack that did not survive would trip;
/// `MIN_GRAVITY_TICKS` does not bite until level 40 and trips the *later* one;
/// and `CELL` never bites at all, because a cell's rect is written at bootstrap
/// and only its colour is rewritten after.
fn with_a_different_score(source: &str) -> anyhow::Result<String> {
    let anchor = "pub const LINE_SCORE: [u32; 4] = [100, 300, 500, 800];";
    anyhow::ensure!(
        source.contains(anchor),
        "demo 10's scoring table is not where this gate expected it — the rule edit is a text \
         edit, so a rename here is a gate to re-point rather than one that quietly stops changing \
         any rule at all"
    );
    Ok(source.replace(
        anchor,
        "pub const LINE_SCORE: [u32; 4] = [1000, 3000, 5000, 8000];",
    ))
}

/// §6 M18's other exit row: **a rule changed mid-game reloads with the stack
/// intact, inside §9's two seconds.**
///
/// The subject is a game actually being played — demo 10's bot, never giving up,
/// eight thousand ticks of it — and the edit is one line of the scoring table.
/// Three claims, and the middle one is the milestone's:
///
/// - **the rule changed.** The two runs part, and not at some tick a timing
///   wobble could have chosen: at exactly the first row cleared after the swap,
///   which the in-process script names independently.
/// - **with the stack intact.** They are identical *through* the swap and for
///   every tick between it and that clear. A board that had been rebuilt, reset
///   or half-migrated would diverge at the swap tick, and the same assertion
///   catches a shell that quietly restarted the session. The board it crossed
///   with is a number here, not an assumption: an empty well would make the
///   claim true and vacuous.
/// - **and it was a swap, not a migration.** Nothing in demo 10's schema moved,
///   so the reload reports no component migrated. §5.11's chaos gate covers the
///   other case; this one is the common edit, where the cost is a pointer.
///
/// The budget is M5's `BUDGET_MS`, measured the way `measure_loop` measures it —
/// the rebuild a save triggers, plus the host's own save-to-swap — and judged
/// on the best of up to [`WALL_CLOCK_TRIES`] runs ([`best_under`]). A retry
/// re-runs the *whole* proof: the correctness claims are asserted inside every
/// attempt and are never what a resample waives; only the clock gets another
/// swing.
fn rules() -> anyhow::Result<()> {
    let total = best_under(
        "the rule edit's save-to-new-rule",
        BUDGET_MS,
        WALL_CLOCK_TRIES,
        rules_once,
    )?;
    anyhow::ensure!(
        total <= BUDGET_MS,
        "the rule reached the running game in {total} ms at its best over {WALL_CLOCK_TRIES} \
         attempts against §9's {BUDGET_MS} ms — the per-attempt rebuild/swap breakdown is in the \
         lines above"
    );
    Ok(())
}

/// One full run of the rules gate: every correctness assertion, then the
/// measured save-to-new-rule total in milliseconds — the budget verdict is
/// [`rules`]'s.
fn rules_once() -> anyhow::Result<u128> {
    let source = game_source(DEMO_10)?;
    let edited = with_a_different_score(&source)?;

    // The stream, and the board at every tick of it, from the same table the
    // shell is about to run.
    let frames = demo_10_tetris::session::endless(RULES_TICKS);
    let progress =
        demo_10_tetris::session::progress(&frames).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Its own directory: the watcher watches a *directory*, so a game rewritten
    // under `target/debug` would be a reload event for everything pointed there.
    let dir = workspace_root().join("target/rules");
    std::fs::create_dir_all(&dir)?;
    let stream = dir.join("endless.ggrp");
    std::fs::write(&stream, tetris_stream(&frames).encode())?;
    let stream = stream.display().to_string();

    // One variant name written twice, not two: a second package would rebuild
    // demo 10 from cold, and a cold build is not what a save costs.
    write_variant(DEMO_10, "rules", &source)?;
    let before = dir.join("before.bin");
    std::fs::copy(build_variant(DEMO_10, "rules")?, &before)?;

    write_variant(DEMO_10, "rules", &edited)?;
    let started = std::time::Instant::now();
    let built = build_variant(DEMO_10, "rules")?;
    let rebuild_ms = started.elapsed().as_millis();
    let after = dir.join("after.bin");
    std::fs::copy(&built, &after)?;

    exec(
        cargo().args(["build", "-p", "gg-runtime"]),
        "build the shell [dev]",
    )?;
    let host = exe("debug", "gg-runtime");
    let game = dir.join(dylib_name(DEMO_10));

    std::fs::copy(&before, &game)?;
    let untouched = sequence(&play(&host, &game, &["--replay", &stream], true)?)?;

    // The swap is aimed at a tick, and where it *lands* is still the machine's
    // to choose — the sighting crosses a pipe while the replay races on.
    // Landing on a clearing tick makes the run uninformative rather than red:
    // the new rule bites on the same tick the new build arrives, so every
    // difference — including a world that did not survive — shows in the same
    // place and the two cannot be told apart. The gate takes another swing at a
    // different moment instead of asserting something it did not observe.
    let mut taken = None;
    for aim in RULES_SWAPS {
        std::fs::copy(&before, &game)?;
        let marker = format!("tick={aim} hash=");
        let log = reload_after(&game, &after, &["--replay", &stream], &marker, 0)?;
        let swapped = sequence(&log)?;
        let reloaded = log
            .lines()
            .find(|l| l.contains("game reloaded"))
            .unwrap_or_default();
        let swap = usize::try_from(crate::util::field_u64(reloaded, "tick")?).unwrap_or(usize::MAX);
        let swap_ms = u128::from(crate::util::field_u64(reloaded, "save_to_swap_ms")?);

        anyhow::ensure!(
            log.matches("migrated").count() == 0,
            "the reload migrated a component — a scoring table is code, and nothing in demo 10's \
             schema moved:\n{log}"
        );
        anyhow::ensure!(
            untouched.len() == swapped.len() && untouched.len() == RULES_TICKS,
            "the two runs are {} and {} ticks of a {RULES_TICKS}-tick stream",
            untouched.len(),
            swapped.len()
        );
        anyhow::ensure!(
            swap < RULES_TICKS / 2,
            "the swap landed at tick {swap} of {RULES_TICKS} — past halfway is a machine this \
             stream is too short for, and the next failure it caused would be a confusing one"
        );

        // Where the rule first *can* show: the next row to go, counted from the
        // swap tick itself, because that is the first tick the new build runs.
        let clear = (swap..progress.len())
            .find(|&i| i > 0 && progress[i].lines > progress[i - 1].lines)
            .ok_or_else(|| anyhow::anyhow!("no row was cleared after the swap at tick {swap}"))?;
        if clear == swap {
            println!(
                "xtask: the swap landed on tick {swap}, which is itself a clearing tick — \
                 swinging again"
            );
            continue;
        }
        taken = Some((swap, swap_ms, clear, swapped));
        break;
    }
    let (swap, swap_ms, clear, swapped) = taken.ok_or_else(|| {
        anyhow::anyhow!(
            "every one of {} attempts put the swap on a clearing tick, which is a coincidence \
             worth looking at rather than retrying further",
            RULES_SWAPS.len()
        )
    })?;

    // Mid-game, and on a board worth preserving.
    let at_swap = progress
        .get(swap)
        .copied()
        .ok_or_else(|| anyhow::anyhow!("the swap landed at tick {swap}, past the stream"))?;
    anyhow::ensure!(
        !at_swap.over && at_swap.occupied > 0,
        "the swap landed on a board with {} cells and over={} — a reload onto an empty or dead \
         well proves nothing about a stack surviving one",
        at_swap.occupied,
        at_swap.over
    );

    let parted = untouched
        .iter()
        .zip(&swapped)
        .find(|(a, b)| a.1 != b.1)
        .map(|(a, _)| usize::try_from(a.0).unwrap_or(usize::MAX))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "the two runs never parted — a scoring table the reload did not pick up makes \
                 this gate a comparison of a sequence with itself"
            )
        })?;
    anyhow::ensure!(
        parted == clear,
        "the runs parted at tick {parted}; the first row cleared after the swap at {swap} goes at \
         {clear}. Earlier means the swap itself moved the world — the stack did not cross intact. \
         Later means the new scoring table never ran."
    );

    let total = rebuild_ms + swap_ms;
    println!(
        "xtask reload: demo 10's scoring rule was rewritten under a game in progress — a new build \
         swapped in at tick {swap} onto a well of {} cells at {} lines, hashing identically to the \
         unswapped run through that tick and the {} after it, and differently from the next clear \
         at {clear} on: {total} ms save to new rule (rebuild {rebuild_ms} + swap {swap_ms}, budget \
         {BUDGET_MS}) (§6 M18)",
        at_swap.occupied,
        at_swap.lines,
        clear - swap - 1,
    );
    Ok(total)
}

/// Ticks of demo 11's patrol the feel retune is measured over — long enough to
/// hold every [`FEEL_SWAPS`] aim in its first half.
const FEEL_TICKS: usize = 4_000;

/// Ticks to aim the feel swap at, [`RULES_SWAPS`]' way: anchored on the tick's
/// own hash line, so no wall time is priced. Each sits in one of the patrol's
/// grounded walking stretches (`session::walking`), because a swap that lands
/// mid-air is uninformative — the constant bites immediately and a surviving
/// world cannot be told from a rebuilt one — and where the swap *lands* is
/// still the machine's, so the gate checks and swings again rather than
/// assuming.
const FEEL_SWAPS: [u64; 3] = [601, 1_101, 1_621];

/// The feel edit: how hard the world pulls (§6 M20 pull 4).
///
/// Chosen by the same criterion as demo 10's scoring table, with the platformer
/// twist: gravity runs **every tick** and still changes nothing while the
/// player is grounded — the pull is applied, the ground resolves it away, and
/// the resolved state is bit-identical under any sane constant. It is *latent
/// on the ground and decisive in the air*, so the swapped run keeps matching
/// the unswapped one across the boundary for exactly as long as the player is
/// standing, and parts on the first airborne tick after the next jump — a tick
/// the in-process script names independently.
fn with_a_heavier_pull(source: &str) -> anyhow::Result<String> {
    let anchor = "pub const GRAVITY: f64 = 0.010;";
    anyhow::ensure!(
        source.contains(anchor),
        "demo 11's gravity is not where this gate expected it — the feel edit is a text edit, so \
         a rename here is a gate to re-point rather than one that quietly stops retuning anything"
    );
    Ok(source.replace(anchor, "pub const GRAVITY: f64 = 0.016;"))
}

/// §6 M20's reload row: **a feel constant retuned under a run in progress
/// reloads with the player intact, inside §9's two seconds, and the runs part
/// only where the arc does.**
///
/// The subject is demo 11's patrol — walk, turn, a full jump every cycle — and
/// the edit is one line: gravity, heavier. The three claims are `rules`'s with
/// the platformer's own middle: identical *through* the swap and across every
/// grounded tick after it (a player that had been rebuilt, reset or teleported
/// would part at the swap tick), then different from the first tick the player
/// integrates airborne under the new pull — the jump tick itself is exempt
/// because its gravity is resolved away on the ground and its velocity is the
/// jump's own constant. Nothing in the schema moved, so the reload must report
/// no migration. Budget and retry semantics are [`rules`]'s verbatim.
fn feel() -> anyhow::Result<()> {
    let total = best_under(
        "the feel edit's save-to-new-gravity",
        BUDGET_MS,
        WALL_CLOCK_TRIES,
        feel_once,
    )?;
    anyhow::ensure!(
        total <= BUDGET_MS,
        "the retune reached the running game in {total} ms at its best over {WALL_CLOCK_TRIES} \
         attempts against §9's {BUDGET_MS} ms — the per-attempt rebuild/swap breakdown is in the \
         lines above"
    );
    Ok(())
}

/// One full run of the feel gate: every correctness assertion, then the
/// measured save-to-new-gravity total in milliseconds — the verdict is
/// [`feel`]'s.
fn feel_once() -> anyhow::Result<u128> {
    let source = game_source(DEMO_11)?;
    let edited = with_a_heavier_pull(&source)?;

    let frames = demo_11_platformer::session::endless(FEEL_TICKS);
    let progress = demo_11_platformer::session::progress(&platformer_entry()?, &frames)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let opening = progress.first().map(|(t, _)| *t).unwrap_or(0);
    // State at the end of absolute tick `t`.
    let at = |t: u64| {
        progress
            .get(usize::try_from(t - opening).unwrap_or(usize::MAX))
            .map(|(_, p)| *p)
    };

    let dir = workspace_root().join("target/feel");
    std::fs::create_dir_all(&dir)?;
    let stream = dir.join("patrol.ggrp");
    std::fs::write(&stream, platformer_stream(&frames, opening).encode())?;
    let stream = stream.display().to_string();
    let scene = platformer_scene()?;

    write_variant(DEMO_11, "feel", &source)?;
    let before = dir.join("before.bin");
    std::fs::copy(build_variant(DEMO_11, "feel")?, &before)?;

    write_variant(DEMO_11, "feel", &edited)?;
    let started = std::time::Instant::now();
    let built = build_variant(DEMO_11, "feel")?;
    let rebuild_ms = started.elapsed().as_millis();
    let after = dir.join("after.bin");
    std::fs::copy(&built, &after)?;

    exec(
        cargo().args(["build", "-p", "gg-runtime"]),
        "build the shell [dev]",
    )?;
    let host = exe("debug", "gg-runtime");
    let game = dir.join(dylib_name(DEMO_11));

    std::fs::copy(&before, &game)?;
    // `--frames` for the platformer gate's reason: the bounded run is the
    // blessed one.
    let ticks = FEEL_TICKS.to_string();
    let args = [
        "--replay",
        stream.as_str(),
        "--load",
        scene.as_str(),
        "--frames",
        ticks.as_str(),
    ];
    let untouched = sequence(&play(&host, &game, &args, true)?)?;

    let mut taken = None;
    for aim in FEEL_SWAPS {
        std::fs::copy(&before, &game)?;
        let marker = format!("tick={aim} hash=");
        let log = reload_after(&game, &after, &args, &marker, 0)?;
        let swapped = sequence(&log)?;
        let reloaded = log
            .lines()
            .find(|l| l.contains("game reloaded"))
            .unwrap_or_default();
        let swap = crate::util::field_u64(reloaded, "tick")?;
        let swap_ms = u128::from(crate::util::field_u64(reloaded, "save_to_swap_ms")?);

        // Not `matches("migrated")`: `--load`'s own "save loaded" line carries
        // a `migrated=false` field, and a substring count would trip on it.
        anyhow::ensure!(
            log.lines()
                .filter(|l| l.contains("migrated") && !l.contains("save loaded"))
                .count()
                == 0,
            "the reload migrated a component — gravity is code, and nothing in demo 11's schema \
             moved:\n{log}"
        );
        anyhow::ensure!(
            untouched.len() == swapped.len() && untouched.len() == FEEL_TICKS,
            "the two runs are {} and {} ticks of a {FEEL_TICKS}-tick stream",
            untouched.len(),
            swapped.len()
        );
        anyhow::ensure!(
            swap < opening + FEEL_TICKS as u64 / 2,
            "the swap landed at tick {swap} of {FEEL_TICKS} — past halfway is a machine this \
             stream is too short for, and the next failure it caused would be a confusing one"
        );

        // Mid-patrol, on the ground, run still going: the configuration the
        // claim is about. Anywhere else is uninformative, not red.
        let Some(state) = at(swap) else {
            anyhow::bail!("the swap landed at tick {swap}, past the stream");
        };
        anyhow::ensure!(state.deaths == 0, "the patrol died before tick {swap}");
        anyhow::ensure!(!state.finished, "the patrol finished before tick {swap}");
        if !state.grounded {
            println!(
                "xtask: the swap landed at tick {swap} with the player airborne — swinging again"
            );
            continue;
        }

        // Where the pull first *can* show: the tick after the next airborne
        // tick — the jump tick's own gravity is resolved away on the ground
        // and its velocity is JUMP_VELOCITY either way.
        let arc = (swap..opening + FEEL_TICKS as u64)
            .find(|&t| at(t).is_some_and(|p| !p.grounded))
            .ok_or_else(|| anyhow::anyhow!("the patrol never left the ground after tick {swap}"))?;
        taken = Some((swap, swap_ms, arc + 1, swapped));
        break;
    }
    let (swap, swap_ms, arc, swapped) = taken.ok_or_else(|| {
        anyhow::anyhow!(
            "every one of {} attempts landed the swap mid-air, which is a coincidence worth \
             looking at rather than retrying further",
            FEEL_SWAPS.len()
        )
    })?;

    let parted = untouched
        .iter()
        .zip(&swapped)
        .find(|(a, b)| a.1 != b.1)
        .map(|(a, _)| a.0)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "the two runs never parted — a gravity the reload did not pick up makes this gate \
                 a comparison of a sequence with itself"
            )
        })?;
    anyhow::ensure!(
        parted == arc,
        "the runs parted at tick {parted}; the first airborne integration after the swap at \
         {swap} is tick {arc}. Earlier means the swap itself moved the world — the player did \
         not cross intact. Later means the retuned gravity never ran."
    );

    let total = rebuild_ms + swap_ms;
    println!(
        "xtask reload: demo 11's gravity was retuned under a run in progress — a new build \
         swapped in at tick {swap} with the player on the ground, hashing identically to the \
         unswapped run through that tick and the {} after it, and differently from the first \
         airborne tick at {arc} on: {total} ms save to new feel (rebuild {rebuild_ms} + swap \
         {swap_ms}, budget {BUDGET_MS}) (§6 M20)",
        arc - swap - 1,
    );
    Ok(total)
}

/// Ticks appended to the recorded session for the save gate: idle, one press of
/// `RESTART`, then room for the new game to start. The press is what makes the
/// reopened build print the record it inherited.
const BEST_TAIL: usize = 12;

/// §6 M18's last save row: **the high score survives a close and reopen across a
/// component schema change.**
///
/// A whole recorded game is played and closed with `--save` — the world at the
/// top-out, record and all. Then the file is *reopened by builds that are not
/// the one that wrote it*, and the number is read back out of the log line a new
/// game prints. Three of them, and the third is what makes the first two mean
/// something:
///
/// - **the same build** reopens it and starts a new game against the old record.
///   The control: a failure here is the save path, not the policy.
/// - **a build whose `Best` gained a field** reopens it too, reports the
///   migration by the component's name, and the record is the same number. This
///   is the row: §4.5 says a save may *gain*, and a defaulted field is the
///   gaining half.
/// - **a build that renamed the component** is refused, by that name, before
///   anything is touched. A save policy that could only accept is not a policy —
///   and a rename is what a build that stopped declaring a component looks like
///   from the file's side.
///
/// The expected number is not written here: it is the highest score the
/// in-process script reaches, so an edit to the session moves the gate with it.
fn best() -> anyhow::Result<()> {
    let source = game_source(DEMO_10)?;
    let frames = demo_10_tetris::session::frames();
    let played = frames.len();
    let progress =
        demo_10_tetris::session::progress(&frames).map_err(|e| anyhow::anyhow!("{e}"))?;
    let record = progress.iter().map(|p| p.score).max().unwrap_or(0);
    anyhow::ensure!(
        record > 0,
        "the recorded session scored nothing, so there is no record to lose"
    );
    anyhow::ensure!(
        progress.last().is_some_and(|p| p.over),
        "the session is saved at its last tick and that tick has to be a dead board — a live one \
         would ignore the `RESTART` this gate reopens with"
    );

    // One stream, split by `--frames` the way demo 08's gate splits its own: the
    // save lands mid-file, and what runs after it runs at the ticks the file
    // says, because `--load` sets the tick the replay is seeked to.
    let mut stream = frames;
    stream.push(InputFrame {
        buttons: 0,
        axes: [0; gg_input::MAX_AXES],
    });
    stream.push(InputFrame {
        buttons: 1 << demo_10_tetris::RESTART.index(),
        axes: [0; gg_input::MAX_AXES],
    });
    for _ in 2..BEST_TAIL {
        stream.push(InputFrame {
            buttons: 0,
            axes: [0; gg_input::MAX_AXES],
        });
    }

    let dir = workspace_root().join("target/best");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("session.ggrp");
    std::fs::write(&path, tetris_stream(&stream).encode())?;
    let replay = path.display().to_string();

    // One variant name rewritten three times, so two of the three builds are
    // incremental rather than a demo 10 from cold each.
    let lay = |source: &str, name: &str| -> anyhow::Result<PathBuf> {
        write_variant(DEMO_10, "best", source)?;
        let kept = dir.join(name);
        std::fs::copy(build_variant(DEMO_10, "best")?, &kept)?;
        Ok(kept)
    };
    let same = lay(&source, "same.bin")?;
    let wider = lay(&with_a_wider_record(&source)?, "wider.bin")?;
    let renamed = lay(&with_the_record_renamed(&source)?, "renamed.bin")?;

    exec(
        cargo().args(["build", "-p", "gg-runtime"]),
        "build the shell [dev]",
    )?;
    let host = exe("debug", "gg-runtime");
    let save = dir.join("best.ggsv");
    let save_arg = save.display().to_string();

    let closed = play(
        &host,
        &same,
        &[
            "--replay",
            &replay,
            "--frames",
            &played.to_string(),
            "--save",
            &save_arg,
        ],
        false,
    )?;
    anyhow::ensure!(
        closed.contains("save written") && save.is_file(),
        "the played session wrote no save:\n{closed}"
    );
    anyhow::ensure!(
        !closed.contains("tetris: best"),
        "the closing run already printed a record — this gate reads that line out of the *reopened* \
         build, and a stream that restarted before the save would be reading its own:\n{closed}"
    );

    let tail = (stream.len() - played).to_string();
    for (what, game, migrates) in [
        ("the same build", &same, false),
        ("a wider record", &wider, true),
    ] {
        let log = play(
            &host,
            game,
            &["--load", &save_arg, "--replay", &replay, "--frames", &tail],
            false,
        )?;
        anyhow::ensure!(
            log.contains("save loaded"),
            "[{what}] the shell did not load {save_arg}:\n{log}"
        );
        let named = log.contains("migrated") && log.contains("tetris.best");
        anyhow::ensure!(
            named == migrates,
            "[{what}] the load {} a migration of `tetris.best`, and this build {} one:\n{log}",
            if named { "reported" } else { "reported no" },
            if migrates { "makes" } else { "makes no" },
        );
        let wanted = format!("tetris: best {record}");
        anyhow::ensure!(
            log.contains(&wanted),
            "[{what}] the reopened game did not start against `{wanted}` — the record did not \
             survive the file:\n{log}"
        );
    }

    let out = Command::new(&host)
        .arg("--game")
        .arg(&renamed)
        .args(["--load", &save_arg, "--replay", &replay, "--frames", &tail])
        .env("GG_HEADLESS", "1")
        .env("RUST_LOG", "info")
        .output()?;
    let log = format!("{}{}", plain(&out.stdout), plain(&out.stderr));
    anyhow::ensure!(
        !out.status.success(),
        "a build that stopped declaring `tetris.best` loaded the save anyway — §4.5's `a save may \
         gain, never lose` is not being enforced by the shell:\n{log}"
    );
    anyhow::ensure!(
        log.contains("tetris.best"),
        "the refusal did not name the component that would have been lost:\n{log}"
    );

    println!(
        "xtask reload: demo 10's record of {record} crossed a close and reopen — read back by the \
         same build, by one whose `Best` gained a field (migrated by name), and refused by one \
         that stopped declaring it (§6 M18)"
    );
    Ok(())
}

/// The ticks on which the hash did not move — a run's shape, with its values
/// taken away. Tick 0 is never quiet: there is nothing before it to equal.
fn quiet_ticks<'a>(hashes: impl Iterator<Item = &'a str>) -> Vec<usize> {
    let mut out = Vec::new();
    let mut previous: Option<&str> = None;
    for (tick, hash) in hashes.enumerate() {
        if previous == Some(hash) {
            out.push(tick);
        }
        previous = Some(hash);
    }
    out
}

/// [`quiet_ticks`] over a parsed baseline.
fn quiet(baseline: &[(u64, u128)]) -> Vec<usize> {
    let text: Vec<String> = baseline.iter().map(|(_, h)| format!("{h:032x}")).collect();
    quiet_ticks(text.iter().map(String::as_str))
}

/// The tier `HASHED_TIERS` cannot contain and this gate cannot do without: the
/// shipping one. It computes no canonical hash, so it is compared by the file it
/// writes instead — which is a *stricter* equality than the hash, and the only
/// one available in the configuration that ships.
const DIST_TIER: Tier = Tier {
    name: "dist",
    features: "tier-dist",
    profile: "dist",
    out: "dist",
};

/// Where the gate's files land. Under `target/`, because a save is build output.
fn save_dir() -> anyhow::Result<PathBuf> {
    let dir = workspace_root().join("target/save");
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// §6 M14's cross-tier criterion: **a save written under dev loads under dist
/// and dist-verify and hashes identically in all three**.
///
/// Demo 08's own tests take a save mid-walk and resume it in one process, which
/// covers the format. What only this can cover is the tier axis — that fat LTO,
/// one codegen unit and a stripped binary read a file dev wrote, migrate nothing
/// they should not, and continue the same session. Three claims, and each fails
/// differently:
///
/// - **dev writes a real session**: the walk's log lines appear, in order. A
///   save of a world where nothing happened would pass everything below.
/// - **every tier resumes it identically**: the remaining ticks of the replay
///   run on top of the loaded world and the *world image* inside the re-saved
///   file is byte-identical across all four tiers. Bytes rather than a hash on
///   purpose: dist computes no hash, and an image comparison covers the entity
///   allocator and archetype order that the canonical hash deliberately
///   abstracts over. The *files* differ and must — the container names the
///   dylib that wrote it, and four tiers are four dylibs.
/// - **and the hashed tiers agree tick for tick**: §5.6c's machinery over a
///   stream that started from a file rather than from tick zero.
fn save() -> anyhow::Result<()> {
    let replay = save_replay_path();
    anyhow::ensure!(
        replay.is_file(),
        "no save stream at {} — `cargo xtask replay --bless` authors it",
        replay.display()
    );
    // Every tick count comes off the *file*, never off `session()`. A gate that
    // re-derived its budget from the demo's own constants would follow an edit
    // that moved the chests and stay green over a walk that now misses them —
    // the frozen stream is the whole reason this is a gate (§5.6).
    let total = Replay::decode(&std::fs::read(&replay)?)?.ticks();
    let replay = replay.display().to_string();
    // Stopped mid-walk, so the file holds a session in progress rather than a
    // finished one: chests already open, an avatar somewhere between two.
    let written = (total / 2).to_string();
    let dir = save_dir()?;
    let file = dir.join("demo08.ggsv");
    let path = file.display().to_string();

    let (host, game) = stage_game(&HASHED_TIERS[0], "demo-08-save", "demo_08_save")?;
    let prefix = play(
        &host,
        &game,
        &["--replay", &replay, "--frames", &written, "--save", &path],
        false,
    )?;
    anyhow::ensure!(
        prefix.contains("save written"),
        "dev wrote no save:\n{prefix}"
    );
    anyhow::ensure!(file.is_file(), "the shell said it wrote {path} and did not");

    let remaining = (total - total / 2).to_string();
    let mut runs: Vec<(&str, Vec<(u64, String)>)> = Vec::new();
    let mut files: Vec<(&str, u64, Vec<u8>)> = Vec::new();
    for tier in HASHED_TIERS.iter().chain(std::iter::once(&DIST_TIER)) {
        let hashed = tier.name != DIST_TIER.name;
        let (host, game) = stage_game(tier, "demo-08-save", "demo_08_save")?;
        let out = dir.join(format!("resumed-{}.ggsv", tier.name));
        let log = play(
            &host,
            &game,
            &[
                "--load",
                &path,
                "--replay",
                &replay,
                "--frames",
                &remaining,
                "--save",
                &out.display().to_string(),
            ],
            hashed,
        )?;
        anyhow::ensure!(
            log.contains("save loaded"),
            "[{}] the shell did not load {path}:\n{log}",
            tier.name
        );
        // A migration here is a bug in the gate, not a feature of it: the same
        // build wrote and read this file, so every component must be `Reused`.
        anyhow::ensure!(
            log.contains("migrated=false"),
            "[{}] one build wrote this save and another read it — the tier legs are not \
             building the same demo:\n{log}",
            tier.name
        );
        // The whole walk, across the two runs: the chests dev opened before the
        // save, then the ones this tier opened after loading it. In order, and
        // all of them — a run that resumed a *world* but not the *session*
        // reaches the split and stops.
        let mut at = 0;
        for line in prefix.lines().chain(log.lines()) {
            if at < demo_08_save::SESSION_LOG.len() && line.contains(demo_08_save::SESSION_LOG[at])
            {
                at += 1;
            }
        }
        anyhow::ensure!(
            at == demo_08_save::SESSION_LOG.len(),
            "[{}] the replayed walk reached {at} of {} logged events — expected {:?} in order, \
             across the run that saved and the run that loaded:\n{log}",
            tier.name,
            demo_08_save::SESSION_LOG.len(),
            demo_08_save::SESSION_LOG,
        );
        let resaved = gg_ecs::Save::decode(&std::fs::read(&out)?)?;
        files.push((tier.name, resaved.tick(), resaved.snapshot().encode()));
        if hashed {
            runs.push((tier.name, sequence(&log)?));
        }
    }

    for pair in files.windows(2) {
        anyhow::ensure!(
            (pair[0].1, &pair[0].2) == (pair[1].1, &pair[1].2),
            "§6 M14: the world {} resumed to is not the world {} resumed to — tick {} / {} bytes \
             against tick {} / {} bytes. One file, four codegens, one world was the claim",
            pair[0].0,
            pair[1].0,
            pair[0].1,
            pair[0].2.len(),
            pair[1].1,
            pair[1].2.len()
        );
    }
    for pair in runs.windows(2) {
        if let Some(found) = divergence(&pair[0], &pair[1]) {
            anyhow::bail!("§6 M14: {found}");
        }
    }
    println!(
        "xtask reload: demo 08's save crossed dev → dev, instrumented, dist-verify and dist — \
         one {}-byte world at tick {}, and the three hashed tiers agreed tick for tick over {} \
         resumed ticks (§6 M14)",
        files[0].2.len(),
        files[0].1,
        runs[0].1.len(),
    );
    play_mode()
}

/// §6 M14's other equality: **play → mutate → stop restores a bit-identical
/// world**, through the shell, in every tier that ships one.
///
/// The demo's own test proves it over a world in one process. What this adds is
/// the configuration: dist's codegen taking the snapshot, and the answer coming
/// back out of a binary with no hash to check itself with.
fn play_mode() -> anyhow::Result<()> {
    let total = Replay::decode(&std::fs::read(save_replay_path())?)?.ticks();
    let replay = save_replay_path().display().to_string();
    // Enter early enough that the walk is still opening chests afterwards —
    // stopping a session in which nothing happened would prove nothing.
    let script = format!("{}:{total}", total / 4);
    for tier in HASHED_TIERS.iter().chain(std::iter::once(&DIST_TIER)) {
        let (host, game) = stage_game(tier, "demo-08-save", "demo_08_save")?;
        let log = play(
            &host,
            &game,
            &[
                "--replay",
                &replay,
                "--frames",
                &(total + 1).to_string(),
                "--play",
                &script,
            ],
            false,
        )?;
        anyhow::ensure!(
            captured_entities(&log)? > 0,
            "[{}] the play edge captured an empty world, so the stop below gives back nothing \
             — `changed` and `identical` are both true of that and neither is worth reading:\n\
             {log}",
            tier.name
        );
        // Both halves, or the gate passes on a session in which nothing
        // happened: `changed` is what makes `identical` an achievement.
        anyhow::ensure!(
            log.contains("changed=true"),
            "[{}] the world at the stop tick is the world that entered play — there was nothing \
             to undo, so this run proves nothing:\n{log}",
            tier.name
        );
        anyhow::ensure!(
            log.contains("identical=true"),
            "[{}] stopping play did not give back the world that entered it:\n{log}",
            tier.name
        );
    }
    println!(
        "xtask reload: demo 08 entered play at tick {}, played to {total}, and stopped back onto \
         the same bytes under dev, instrumented, dist-verify and dist (§6 M14)",
        total / 4
    );
    Ok(())
}

/// The log lines §6 M15's session must produce, in order. Each names a distinct
/// claim, and a click that landed on a neighbouring button writes a different
/// set — which is what makes this the half of the gate a hash comparison cannot
/// be: two runs that both clicked on nothing agree perfectly.
const EDITOR_LOG: &[&str] = &[
    // §6 M15.2 post-close: the editor opens Stopped, so the *open* is the first
    // stop — the camera latch fires at tick 0, before anything is clicked —
    // and play is a click: the capture line and the transport line together,
    // because the first is the Stash's own edge and the second is the operator
    // reaching it.
    "editor: camera taken", // item 2: the stopped scene is the operator's to look at
    "play mode entered",
    "editor: play state",       // the title bar started the stopped scene
    "editor: play state",       // then paused the running game
    "component=\"demo05.hub\"", // the inspector reached a *game's* own component
    "field=\"angle\"",          // by name, out of a schema this host never compiled
    "editor: play state",       // and played it again
    "editor: play state",       // and paused it again
    "save written",             // `file` → `save`, not the shell's exit path
    "play mode stopped",        // §6 M15.2: and left play mode, restoring the capture
    // §6 M15.4, in the order the script performs them. The pick is only
    // blind-aimable because the spawn precedes it: a spawned entity lands down
    // the camera's own forward axis and so projects to the centre of the game
    // pane, which is where the click goes.
    "editor: spawned",    // item 5: the tree makes an entity
    "editor: picked",     // item 1: and a ray through the viewport finds it
    "editor: duplicated", // item 5 again, through the registry rather than a type
    "editor: deleted",
    "editor: undo", // item 4: and the delete is taken back out of the ring
    // Item 4 of M15.2, as two lines because it is two claims: a drag reached the
    // axis pair the host appends for it, and the keys reached the camera. One
    // line for both would be satisfied by whichever gesture came first.
    "editor: camera turned",
    "editor: camera flown",
];

/// Nudges the script performs. Pinned, because "at least one edit" would pass
/// over a session whose step button had stopped working.
const EDITOR_NUDGES: usize = 6;

/// §6 M15's fourth exit row: **an editor session is recordable and replayable to
/// the same final state hash** — and, on the way, the first and the fifth.
///
/// The session is `gg_editor::session`'s, frozen into `tests/replays/` and
/// replayed through the real shell over demo 05. Four claims, each failing
/// differently:
///
/// - **the editor's input is in the recorded stream and nowhere else.** The
///   file's verb list is demo 05's *plus* the four `gg-ui` names the host
///   appended, so the same file is **refused** by a shell without `--editor` —
///   asserted below, because it is the whole argument that there is no second
///   input path (§4.7, §4.9).
/// - **the clicks landed on the panels they aimed at**: [`EDITOR_LOG`] in order,
///   and exactly [`EDITOR_NUDGES`] field edits naming a component demo 05
///   declared. An inspector that could only reach the host's own protocol types
///   would name one of those instead.
/// - **an inspector edit is hashed state**: while the sim is paused nothing else
///   in demo 05 moves, so every canonical-hash change inside the paused window
///   is an editor edit — and there must be at least as many as there were
///   nudges. That is §6 M15's "edits go through `World` like every other write",
///   and it is checkable only because pause exists.
/// - **and it replays**: two codegens and a repeat of the first agree tick for
///   tick, and the world each one saved is byte-identical.
fn editor() -> anyhow::Result<()> {
    let path = editor_replay_path();
    anyhow::ensure!(
        path.is_file(),
        "no editor stream at {} — `cargo xtask replay --bless` authors it",
        path.display()
    );
    // Off the file, never off `script()`: a gate that re-derived its length from
    // the editor's own constants would follow a panel that moved and stay green
    // over a session that now clicks the chrome (§6 M14 learned this the hard
    // way).
    let total = Replay::decode(&std::fs::read(&path)?)?.ticks();
    let replay = path.display().to_string();

    let mut runs: Vec<(&str, Vec<(u64, String)>)> = Vec::new();
    let mut worlds: Vec<(&str, u64, Vec<u8>)> = Vec::new();
    // dev, instrumented, then dev again: a session that is a function of its
    // input stream has to reproduce against itself as well as across a codegen.
    for (label, tier) in [
        ("dev", &HASHED_TIERS[0]),
        ("instrumented", &HASHED_TIERS[1]),
        ("dev-again", &HASHED_TIERS[0]),
    ] {
        let (host, game) = stage_game(tier, "demo-05-many", "demo_05_many")?;
        let out = save_dir()?.join(format!("editor-{label}.ggsv"));
        let _ = std::fs::remove_file(&out);
        let log = play(
            &host,
            &game,
            &[
                "--replay",
                &replay,
                "--editor",
                "--save",
                &out.display().to_string(),
            ],
            true,
        )?;
        anyhow::ensure!(
            log.contains("ui=true"),
            "[{label}] the shell bound no UI verbs — the editor has no pointer, so every click \
             below landed on nothing:\n{log}"
        );

        let at = reaches(&log, EDITOR_LOG);
        anyhow::ensure!(
            at == EDITOR_LOG.len(),
            "[{label}] the replayed editor session reached {at} of {} logged events — expected \
             {EDITOR_LOG:?} in order; a click that landed on a different panel is what this \
             looks like:\n{log}",
            EDITOR_LOG.len()
        );
        // Every viewport press in this script is made while the scene is
        // stopped, so the take must never fire: §6 M15.4's pick and the game's
        // mouse-look share one click in one rectangle, and the pointer used to
        // win it — the operator selected an entity and lost the cursor, with
        // Escape the only way back. Absence is the assertion because the
        // session has no running viewport click to assert the other half with;
        // `gg_runtime`'s own `takes_pointer` test carries that direction.
        anyhow::ensure!(
            !log.contains("editor: pointer taken by the game"),
            "[{label}] a stopped session handed the mouse to the game — a click that picks is not \
             a click the player made (§6 M15.4):\n{log}"
        );
        let nudges = log
            .lines()
            .filter(|l| l.contains("editor: field nudged"))
            .count();
        anyhow::ensure!(
            nudges == EDITOR_NUDGES,
            "[{label}] the inspector applied {nudges} edits, not {EDITOR_NUDGES}:\n{log}"
        );

        // The editor opens *Stopped* one tick in (§6 M15.2 post-close), so the
        // play edge is the script's first click and the capture behind it is
        // the bootstrapped world. Both halves asserted: entities prove the
        // bootstrap tick ran before the freeze, and the tick proves nobody
        // auto-entered play — which is what "opens Stopped" looks like in a
        // log.
        anyhow::ensure!(
            captured_entities(&log)? > 0,
            "[{label}] the editor's play edge captured an empty world — the bootstrap tick ran no \
             systems, so a stop hands back a scene the game never made:\n{log}"
        );
        let entered = log
            .lines()
            .find(|l| l.contains("play mode entered"))
            .map(|l| crate::util::field_u64(l, "tick"))
            .transpose()?
            .unwrap_or(0);
        anyhow::ensure!(
            entered > 1,
            "[{label}] play mode entered at tick {entered} — the editor entered play by itself \
             instead of opening Stopped and waiting for the transport (§6 M15.2 post-close):\n{log}"
        );

        // §6 M15.2 item 3, and it is M14's comparison pointed at a button: the
        // stop restored the world play began at. Both halves together, for
        // `play_mode`'s reason — `identical` over a session that changed
        // nothing is a gate that cannot fail, and this session has six nudges
        // in it precisely so `changed` is an achievement too.
        anyhow::ensure!(
            log.contains("changed=true"),
            "[{label}] the world at the stop was the world that entered play — the six nudges \
             above did not reach it, so the restore proves nothing:\n{log}"
        );
        anyhow::ensure!(
            log.contains("identical=true"),
            "[{label}] the transport's stop did not give back the world that entered play \
             (§6 M15.2):\n{log}"
        );

        let seq = sequence(&log)?;
        let moved = hash_changes_while_paused(&log, &seq)?;
        anyhow::ensure!(
            moved >= EDITOR_NUDGES,
            "[{label}] the canonical hash moved {moved} times while the sim was paused and the \
             inspector claims {EDITOR_NUDGES} edits — an edit that did not reach the hash is an \
             edit outside the state this engine is built on (§6 M15):\n{log}"
        );

        let bytes = std::fs::read(&out).map_err(|e| {
            anyhow::anyhow!(
                "[{label}] `file` → `save` wrote nothing to {}: {e}",
                out.display()
            )
        })?;
        let save = gg_ecs::Save::decode(&bytes)?;
        worlds.push((label, save.tick(), save.snapshot().encode()));
        runs.push((label, seq));
    }

    // The negative, and it is the load-bearing one: without `--editor` the same
    // file names verbs the build does not declare, so the shell refuses it at
    // load. An editor whose clicks arrived by a side channel would replay here
    // happily and nothing above would notice.
    let (host, game) = stage_game(&HASHED_TIERS[0], "demo-05-many", "demo_05_many")?;
    let refused = Command::new(&host)
        .arg("--game")
        .arg(&game)
        .args(["--replay", &replay])
        .env("GG_HEADLESS", "1")
        .output()?;
    anyhow::ensure!(
        !refused.status.success(),
        "§6 M15: a shell with no editor replayed the editor's own session — its clicks are then \
         not in the recorded verb space, which is the whole claim that there is no second input \
         path (§4.7)"
    );

    // §6 M15.1's residual, exercised rather than described. A session authored
    // for a *window* is replayed by a shell that has none: the panes fill their
    // surface, so without `--editor-extent` every click below lands on a
    // different pane — which the second half of this asserts, because an escape
    // hatch that would pass anyway is not one.
    let windowed = save_dir()?.join("editor-1080p.ggrp");
    std::fs::write(&windowed, editor_replay_at(WINDOWED_EXTENT)?.encode())?;
    let windowed = windowed.display().to_string();
    let extent = format!("{}x{}", WINDOWED_EXTENT.0, WINDOWED_EXTENT.1);
    let named = play(
        &host,
        &game,
        &[
            "--replay",
            &windowed,
            "--editor",
            "--editor-extent",
            &extent,
        ],
        false,
    )?;
    anyhow::ensure!(
        reaches(&named, EDITOR_LOG) == EDITOR_LOG.len(),
        "§6 M15.1: a session recorded at {extent} did not replay under `--editor-extent {extent}` \
         — the flag is the whole of how a windowed recording is reproduced headlessly:\n{named}"
    );
    let unnamed = play(&host, &game, &["--replay", &windowed, "--editor"], false)?;
    anyhow::ensure!(
        reaches(&unnamed, EDITOR_LOG) < EDITOR_LOG.len(),
        "§6 M15.1: the same session reached every event *without* `--editor-extent`, so this leg \
         proves nothing — either the layout stopped depending on the extent or the script stopped \
         aiming at one:\n{unnamed}"
    );

    for pair in worlds.windows(2) {
        anyhow::ensure!(
            (pair[0].1, &pair[0].2) == (pair[1].1, &pair[1].2),
            "§6 M15: the world {} saved is not the world {} saved — tick {} / {} bytes against \
             tick {} / {} bytes",
            pair[0].0,
            pair[1].0,
            pair[0].1,
            pair[0].2.len(),
            pair[1].1,
            pair[1].2.len()
        );
    }
    for pair in runs.windows(2) {
        if let Some(found) = divergence(&pair[0], &pair[1]) {
            anyhow::bail!("§6 M15: {found}");
        }
    }

    // §6 M15.2 post-close: the opening scene. The save the dev run's `file` →
    // `save` wrote is a world as *data* — staged as `scene.ggsave` beside a
    // copy of the action map, a live session opens from it: the probe names
    // it, the load restores it, and nothing enters play, because nobody
    // clicked. The constant hash is the sharpest half — a world no game tick
    // has touched emits one value for the whole run.
    let project = save_dir()?.join("opening-scene");
    std::fs::create_dir_all(&project)?;
    std::fs::copy(
        workspace_root().join("demos/05-many/input.toml"),
        project.join("input.toml"),
    )?;
    std::fs::copy(
        save_dir()?.join("editor-dev.ggsv"),
        project.join("scene.ggsave"),
    )?;
    let scene = play(
        &host,
        &game,
        &[
            "--input",
            &project.join("input.toml").display().to_string(),
            "--editor",
            "--frames",
            "40",
        ],
        true,
    )?;
    anyhow::ensure!(
        scene.contains("opening scene") && scene.contains("save loaded"),
        "§6 M15.2 post-close: a live editor session beside a `scene.ggsave` did not open from it \
         — the probe is the convention, and a scene that must be named with `--load` is not \
         one:\n{scene}"
    );
    anyhow::ensure!(
        !scene.contains("play mode entered"),
        "§6 M15.2 post-close: a session opened from a scene entered play with nobody clicking — \
         load *paused* is the point of the data:\n{scene}"
    );
    let frozen = sequence(&scene)?;
    anyhow::ensure!(
        frozen.windows(2).all(|pair| pair[0].1 == pair[1].1),
        "§6 M15.2 post-close: the canonical hash moved in a session that never left Stopped — a \
         game tick ran over the loaded scene before the operator asked for one:\n{scene}"
    );

    println!(
        "xtask reload: the recorded editor session replayed over demo 05 under dev and \
         instrumented and again under dev — {EDITOR_NUDGES} inspector edits inside the paused \
         window, identical hashes over {} of {total} ticks, and one {}-byte world out of the \
         title bar's `file` menu (§6 M15); a {}x{} session replayed headlessly under --editor-extent \
         and only under it (§6 M15.1); and the same save reopened as `scene.ggsave` — Stopped at \
         its own tick, hash frozen over 40 ticks, no play edge in the log (§6 M15.2 post-close)",
        runs[0].1.len(),
        worlds[0].2.len(),
        WINDOWED_EXTENT.0,
        WINDOWED_EXTENT.1,
    );
    Ok(())
}

/// The project the launcher gate opens, and the crate that builds it.
const LAUNCHER_PROJECT: &str = "05-many";

/// What a launcher session that picked a project says, in order (§6 M15.1 item
/// 4). Every row is a different claim and the *order* is most of the gate:
/// nothing was loaded, a click landed on a project row, the shell acted on it,
/// and the session that followed was over that game's dylib.
const LAUNCHER_LOG: &[&str] = &[
    // `App::new`'s load line, over the loader's absent variant.
    "<no project>",
    "editor: project picked",
    "opening project",
    // The second session's load line — the dylib, by name.
    "demo_05_many",
    "golden runtime clean exit",
];

/// `cargo xtask reload --launcher` — §6 M15.1 item 4's Exit row: the editor
/// opens with no game, and picking a project from inside it loads one.
///
/// The one gate here that drives an **application** rather than the shell, and
/// it drives it exactly as every other editor gate drives the shell: a recorded
/// click stream, replayed with no window anywhere (§1.5). The stream stops at the
/// pick and cannot do otherwise — its id space is the editor's appended verbs
/// over *no* game, and the session that follows is over demo 05, whose verb list
/// is not that one (§4.7). So `--frames` bounds the second session and the log is
/// what says it happened.
fn launcher() -> anyhow::Result<()> {
    let root = workspace_root();
    // The dylib the *scan* looks for — `target/debug`, not a staged copy: the
    // launcher finds projects by `xtask run`'s own convention and this gate must
    // exercise that convention rather than route around it.
    exec(
        cargo().args(["build", "-p", "demo-05-many"]),
        "cargo build (launcher's project)",
    )?;
    exec(
        cargo().args(["build", "-p", "gg-editor-app"]),
        "cargo build (the launcher)",
    )?;

    // Authored against the same scan the run will do, so "row `i`" means the same
    // project in both. A hard-coded index would drift the day a demo is added.
    let projects = gg_editor::project::scan(&root);
    let row = projects
        .iter()
        .position(|p| p.name == LAUNCHER_PROJECT)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "the scan found no `{LAUNCHER_PROJECT}` under {}",
                root.display()
            )
        })?;
    anyhow::ensure!(
        projects[row].built,
        "`{LAUNCHER_PROJECT}` is not built, so the picker will refuse the click this gate makes"
    );

    let (verbs, _) = gg_editor::host::open(&gg_ecs::boundary::Verbs {
        actions: &[],
        axes: &[],
    });
    let id = |names: &[&str], want: &str| {
        names
            .iter()
            .position(|n| *n == want)
            .ok_or_else(|| anyhow::anyhow!("the editor did not append `{want}`"))
    };
    let mut editor = gg_editor::Editor::new(None);
    editor.place(HEADLESS_EXTENT, HEADLESS_DPI);
    let at = gg_editor::session::aim::project(&editor, row)
        .ok_or_else(|| anyhow::anyhow!("the game pane is not up, so there is nothing to aim at"))?;
    let frames = gg_editor::session::frames(
        &[
            gg_editor::session::Act::To(at),
            gg_editor::session::Act::Settle(3),
            gg_editor::session::Act::Click,
            gg_editor::session::Act::Settle(3),
        ],
        gg_input::ActionId::new(id(verbs.actions, "ui_click")?),
        gg_input::AxisId::new(id(verbs.axes, "ui_x")?),
        gg_input::AxisId::new(id(verbs.axes, "ui_y")?),
    );
    let mut meta = ReplayMeta::new(
        gg_math::DETERMINISM_CONTRACT,
        "curated",
        gg_core::DEFAULT_TICK_HZ,
        verbs.actions,
        verbs.axes,
    );
    meta.engine_commit = "generated".to_owned();
    let mut recorder = Recorder::new(meta);
    let ticks = frames.len();
    for (tick, frame) in frames.into_iter().enumerate() {
        recorder.record(tick as u64, frame);
    }
    // Not checked in: this stream is derived from a scan of *this* tree, so a
    // blessed copy would be a baseline of which demos exist (§4.7's curated
    // streams are the ones whose content is the claim).
    let stream = save_dir()?.join("launcher.ggrp");
    std::fs::write(&stream, recorder.finish().encode())?;

    let app = root.join("target/debug").join(if cfg!(windows) {
        "gg-editor.exe"
    } else {
        "gg-editor"
    });
    let bound = ticks.to_string();
    let run = |args: &[&str]| -> anyhow::Result<String> {
        let out = Command::new(&app)
            .args(args)
            // The scan's root, and the reason it is the working directory: a
            // launcher is started *in* a workspace (`Editing::new`).
            .current_dir(&root)
            .env("GG_HEADLESS", "1")
            .env("RUST_LOG", "info")
            .output()
            .map_err(|e| anyhow::anyhow!("failed to spawn {}: {e}", app.display()))?;
        let log = format!("{}{}", plain(&out.stdout), plain(&out.stderr));
        anyhow::ensure!(
            out.status.success(),
            "the launcher exited {}\n{log}",
            out.status
        );
        Ok(log)
    };

    let picked = run(&[
        "--replay",
        &stream.display().to_string(),
        "--frames",
        &bound,
    ])?;
    let reached = reaches(&picked, LAUNCHER_LOG);
    anyhow::ensure!(
        reached == LAUNCHER_LOG.len(),
        "§6 M15.1 item 4: the launcher reached {reached} of {} logged events — expected \
         {LAUNCHER_LOG:?} in order. A click that landed beside the project row is what this \
         looks like:\n{picked}",
        LAUNCHER_LOG.len()
    );
    // The clean-exit line and not the startup one: boot happens once per process
    // — one logger, one config read — and it is the *session* that repeats.
    let sessions = picked
        .lines()
        .filter(|l| l.contains("golden runtime clean exit"))
        .count();
    anyhow::ensure!(
        sessions == 2,
        "§6 M15.1 item 4: {sessions} session(s), not two — picking a project ends the session it \
         was picked in and starts one over that dylib:\n{picked}"
    );

    // The falsification, and it is the reason the rows above mean anything: the
    // same launcher with nothing driving it opens, finds the same projects, and
    // opens none of them. A gate whose markers appeared without the click would
    // be grading the shell's startup and calling it a picker.
    let idle = run(&["--frames", &bound])?;
    anyhow::ensure!(
        !idle.contains("opening project"),
        "§6 M15.1 item 4: a launcher nobody clicked opened a project anyway:\n{idle}"
    );
    anyhow::ensure!(
        idle.contains("<no project>"),
        "§6 M15.1 item 4: the launcher loaded *something* with no `--game`:\n{idle}"
    );

    println!(
        "xtask reload: launcher — opened with no project, listed {} of them, and a replayed click \
         on `{LAUNCHER_PROJECT}` started a second session over its dylib",
        projects.len()
    );
    Ok(())
}

/// How many entities the play edge captured — the number a stop hands back.
///
/// `changed`/`identical` are both satisfied by a capture of *nothing*: a world
/// emptied by the restore differs from the one that entered play and equals the
/// bytes stashed, so the pair reads green over a stop that deleted the scene.
/// That is not a hypothetical — it is what an editor whose first tick ran no
/// systems did, and neither gate below noticed until this line existed.
fn captured_entities(log: &str) -> anyhow::Result<u64> {
    let line = log
        .lines()
        .find(|l| l.contains("play mode entered"))
        .ok_or_else(|| anyhow::anyhow!("play mode never started:\n{log}"))?;
    let count = line
        .split("entities=")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|n| {
            n.trim_end_matches(|c: char| !c.is_ascii_digit())
                .parse()
                .ok()
        })
        .ok_or_else(|| anyhow::anyhow!("no entity count on the play edge: {line}"))?;
    Ok(count)
}

/// How many of `wanted`, in order, `log` gets through. A partial count is the
/// useful answer: it names how far a session got before its clicks stopped
/// landing where the script meant.
fn reaches(log: &str, wanted: &[&str]) -> usize {
    let mut at = 0;
    for line in log.lines() {
        if at < wanted.len() && line.contains(wanted[at]) {
            at += 1;
        }
    }
    at
}

/// A window for the `--editor-extent` leg, and the choice is not arbitrary.
///
/// What the layout depends on is the *logical canvas* — the extent divided by
/// `gg_editor::ui_scale` — not the pixel count. Every 16:9 window whose scale
/// comes out whole reduces to the same 640×360 canvas as a headless run, so
/// 1080p and 1440p recordings replay headlessly with no flag at all, and using
/// one here would make this leg's negative control fail for the right reason.
/// 1600×900 scales by 2 to an 800×450 canvas, which is a genuinely different
/// layout — the case the flag exists for.
const WINDOWED_EXTENT: (u32, u32) = (1600, 900);

/// How many ticks the canonical hash moved on while the editor had the sim
/// paused.
///
/// The window is read out of the shell's own `editor: play state` lines, so this
/// counts what the *session* did rather than what the script meant to do.
/// Nothing in demo 05 advances while paused — no systems, no hierarchy, no
/// widgets — so every change inside the window is an inspector write.
fn hash_changes_while_paused(log: &str, seq: &[(u64, String)]) -> anyhow::Result<usize> {
    let mut windows: Vec<(u64, u64)> = Vec::new();
    let mut paused_at: Option<u64> = None;
    for line in log.lines().filter(|l| l.contains("editor: play state")) {
        let tick = crate::util::field_u64(line, "tick")?;
        match crate::util::field(line, "playing")? {
            "false" => paused_at = Some(tick),
            _ => {
                if let Some(from) = paused_at.take() {
                    windows.push((from, tick));
                }
            }
        }
    }
    // A window still open at the end runs to the last tick recorded.
    if let (Some(from), Some(last)) = (paused_at, seq.last()) {
        windows.push((from, last.0));
    }
    anyhow::ensure!(
        !windows.is_empty(),
        "the session never paused, so nothing here proves an edit reached the hash:\n{log}"
    );
    Ok(seq
        .windows(2)
        .filter(|pair| {
            pair[0].1 != pair[1].1
                && windows
                    .iter()
                    .any(|(from, to)| pair[1].0 > *from && pair[1].0 <= *to)
        })
        .count())
}

/// The other half of §5.6c's "recorded under dist", and §5.8's recorder-presence
/// check made behavioural: the *shipping* configuration records a replay, and
/// the file it writes decodes and names the tier that wrote it.
///
/// This is §1.2's bug-report channel proven end to end rather than by looking
/// for the format's magic in the binary's bytes.
pub fn dist_records() -> anyhow::Result<()> {
    let tier = Tier {
        name: "dist",
        features: "tier-dist",
        profile: "dist",
        out: "dist",
    };
    let (host, game) = stage(&tier)?;
    let out = workspace_root().join("target/tiers/dist/recorded.ggrp");
    let log = play(
        &host,
        &game,
        &["--frames", "60", "--record", &out.display().to_string()],
        false,
    )?;

    // §2's other dist claim, and the one only a run can make: the dylib arrives
    // once. In dev the same line appears again at every swap, so counting it is
    // how "the reload machinery is absent, not merely idle" reads from outside
    // the process — the graph check upstream says `notify` is not linked, and
    // this says nothing behaves as though it were.
    let loads = log.matches("game loaded").count();
    anyhow::ensure!(
        loads == 1,
        "the dist shell loaded the game {loads} times — dist is load-once (§2)"
    );
    // The dev-only path announces itself; dist must never reach it.
    anyhow::ensure!(
        !log.contains("game reloaded"),
        "the dist shell reloaded a dylib:\n{log}"
    );
    let recorded = Replay::decode(&std::fs::read(&out)?)?;
    anyhow::ensure!(
        recorded.meta().tier == "dist",
        "a replay recorded by the dist shell says tier `{}`",
        recorded.meta().tier
    );
    anyhow::ensure!(
        recorded.ticks() == 60,
        "the dist shell recorded {} ticks of 60",
        recorded.ticks()
    );
    recorded.check_verbs(ACTIONS, AXES)?;
    println!(
        "xtask: the dist shell loaded the game once and recorded a readable replay ({} ticks, \
         tier `{}`, verbs intact) — §1.2's bug-report channel out of a shipped build",
        recorded.ticks(),
        recorded.meta().tier
    );
    Ok(())
}
