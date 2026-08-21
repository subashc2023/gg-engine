//! `gg-runtime`: THE host shell — the one executable game code ever runs under,
//! in every tier (§2 Game-code boundary, §3). Thin in code, fat in linkage: zero
//! engine logic, zero game logic, no public API. Complexity budget: 600
//! CI-counted *code* lines (§3) — 300 through M4, 500 at M5 when the shell took
//! delivery of the window, the renderer, live input and record/replay, 600 at M8
//! for the observability stack. The number lives in `xtask`; this restates it.
//!
//! Boot, observability, the game dylib, and `gg-core`'s loop driven to
//! completion. Everything it does is a choice of *which* engine piece runs;
//! the wiring proper is in [`app`], the window in [`play`], and the loop
//! skeleton both drive is `gg-core`'s (§4.1) — never reimplemented here.
//!
//! # Why this is a library as well as a binary
//!
//! `apps/gg-editor` (§6 M15.1 item 4) opens the editor with no game by driving
//! this shell as [`run`] rather than holding a second copy of it, so `main` is
//! just the argv in front of it — and §3's line budget counts a launcher's own
//! lines as the application's, not the shell's.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use gg_core::config::project::Project;
use gg_core::{DEFAULT_TICK_HZ, FrameLoop};
use gg_input::Replay;
use tracing::{error, info, info_span, warn};

mod app;
mod play;
pub mod player;

/// `--set`, for a host with its own argv to consume it the same way (§4.8).
pub use gg_core::config::SET_FLAG;

/// What the shell was told to run. Hand-parsed: a handful of values, and a
/// parser dependency would be the shell's first gram of fat. Passed to [`App`]
/// whole rather than field by field — one struct is what a session *is*.
///
/// `Default` is derived rather than written out at the one construction site: a
/// new flag then costs the field and its arm, not a third place to forget.
#[derive(Default)]
pub struct Args {
    /// The game dylib. **Empty is a real value** and means no project: the
    /// launcher's opening state, where the host registers its own §4.5 protocol
    /// types and the world stays empty (§6 M15.1 item 4). `parse_args` still
    /// requires the flag — a shell run from a command line with no game is a
    /// mistake, and a launcher does not have a command line.
    pub game: PathBuf,
    /// Bounded run. Headless it is `Pace::Locked` — wall time ignored, so a
    /// run's tick count is a property of the run and not of the machine (§5.6).
    pub frames: Option<u64>,
    /// The action map (§4.7). Resolved against the *game's* declared verbs, so
    /// a binding naming a verb this build does not declare is refused by name.
    pub input: Option<PathBuf>,
    /// Where to write this session's recording.
    pub record: Option<PathBuf>,
    /// A recording to drive this session from.
    pub replay: Option<PathBuf>,
    /// A world staged by this shell's *predecessor* (§4.2.2). Passed by a
    /// rejuvenating process to its successor, never by hand.
    pub restore: Option<PathBuf>,
    /// A save to load before the first tick (§6 M14). Unlike `--restore` this is
    /// the *player's* file: written by one build, read by another, and refused by
    /// name rather than migrated into loss. In every tier, because that is the
    /// point of it.
    pub load: Option<PathBuf>,
    /// Where to write one after the last tick.
    pub save: Option<PathBuf>,
    /// `<enter tick>:<stop tick>` — the editor's play/stop, on a script (§6 M14).
    pub play: Option<String>,
    /// `<from frame>:<until frame>` — the window losing focus, on a script (§6
    /// M49). [`Stages::suspended`](gg_core::Stages::suspended)'s unit, for its
    /// reason: a suspension stops ticks, so it cannot be scripted in them.
    ///
    /// In every tier and not lab equipment, `--play`'s rule: what it drives is
    /// the shipping path, and a suspension that existed only where a window does
    /// could not be graded at all (§1.5).
    ///
    /// A replay bounds its run in *ticks*, so one driven by this and by nothing
    /// else stops short by however many frames were away. Naming `--frames` is
    /// the spelling that means "run this long and suspend inside it", which is
    /// what a gate wants and what the leg uses.
    pub away: Option<String>,
    /// Open the editor over the game (§6 M15). Lab equipment: the flag exists
    /// only in a tier that has the crate, and is refused elsewhere by name
    /// rather than ignored. With it, `--save` names where its save button
    /// writes rather than what the shell writes at exit.
    pub editor: bool,
    /// `<w>x<h>` — lay the editor out for this surface instead of the real one
    /// (§6 M15.1).
    ///
    /// The panes fill their surface, so where a widget *is* depends on how big
    /// that surface is, and a session recorded at one size lands its clicks
    /// somewhere else at another. This is how a recording made in a window is
    /// replayed headlessly: name the extent it was recorded at. It changes
    /// nothing else — the window, the swapchain and the game's own canvas are
    /// still the real one, so passing it to a *windowed* run deliberately draws
    /// an editor that does not match its window.
    pub editor_extent: Option<(u32, u32)>,
    /// Leaked-dylib bytes this session tolerates before rejuvenating. Present so
    /// the forced case — zero, restart on the first reload — is exercisable on
    /// demand instead of after a thousand edits.
    pub leak_budget: Option<u64>,
    /// The pack a game draws out of (§4.6). Which file, and nothing else: what
    /// to draw from it is the game's to say, through `Model`.
    pub pack: Option<PathBuf>,
    /// What the window says (§6 M41). `None` is the dev tree's answer — the
    /// dylib's file stem — which is a developer's name for a game and not a
    /// player's.
    pub title: Option<String>,
    /// Which build of the game this is (§6 M73), for a project that says. Held
    /// as the manifest's own text rather than parsed: what this is for is the
    /// line a bug report quotes, and every comparison happens in the tools that
    /// wrote it — the archive's name and the executable's version resource.
    pub version: Option<String>,
    /// The window to open, for a project that asked for one.
    pub window: Option<(u32, u32)>,
    /// The taskbar picture the manifest named, for a project that shipped one
    /// (§6 M46). A *path* rather than pixels: `parse_args` runs before the
    /// tracing subscriber exists, so a file read here would drop its warning on
    /// the floor — which is how the first version of this passed its own gate.
    /// `session` reads it, on the settings file's policy: warned about and
    /// dropped, because no picture is worth a refusal to run.
    pub icon: Option<PathBuf>,
    /// The folder a manifest was found in, and therefore where this run's
    /// `gg.cfg` and its own files are. `None` is every run driven from a command
    /// line, where the working directory is the answer it has always been.
    pub dir: Option<PathBuf>,
    /// Where this game's own bytes belong (§6 M41 item 4, §6 M42): the log, and
    /// the saves and settings M42 puts beside it. Only a project has one — a dev
    /// run writes under `target/` like everything else.
    pub data: Option<PathBuf>,
}

/// Where a project keeps its opening scene: `scene.ggsave` beside the action
/// map, so a project is still the directory `xtask run` points at (§6 M15.2
/// post-close). The one spelling — the editor's save button writes here and
/// [`opening_scene`] reads here, and a drift between the two would leak a scene
/// one session wrote past the next one's probe.
pub(crate) fn scene_path(input: Option<&std::path::Path>) -> Option<PathBuf> {
    Some(input?.parent()?.join("scene.ggsave"))
}

/// The save a live session opens from, found rather than named. Editor-only
/// until §6 M20 pull 2 made the scene the *project's* data: a game whose level
/// is checked in as `scene.ggsave` must open it in every tier, dist included,
/// or the level would be lab equipment. The probe stays implicit, so a recorded
/// or replayed session still refuses it — a scene appearing beside the project
/// later must not diverge a stream blessed without one — and `--load` stays the
/// explicit spelling, taking precedence so the two cannot both load.
pub(crate) fn opening_scene(args: &Args) -> Option<PathBuf> {
    let live = may_touch_project(args) && args.load.is_none();
    live.then(|| scene_path(args.input.as_deref()))
        .flatten()
        .filter(|scene| scene.is_file())
}

/// The manifest's icon as pixels, or `None` with the reason said out loud (§6
/// M46).
///
/// Here rather than in `parse_args` because that runs before the subscriber
/// does, and a warning nothing prints is the failure this function's whole
/// policy is against. Read on every run, windowed or not: a headless run opens
/// no window and still reports a folder that ships a picture it cannot draw,
/// which is what makes the policy gateable at all (`xtask reload --window`).
fn icon(args: &Args) -> Option<(u32, Vec<u8>)> {
    let path = args.icon.as_ref()?;
    let refused = match std::fs::read(path) {
        Err(e) => e.to_string(),
        Ok(bytes) => match gg_core::config::icon::parse(&bytes) {
            Ok(icon) => return Some(icon),
            Err(e) => e.to_string(),
        },
    };
    warn!(path = %path.display(), refused, "icon: opening without one");
    None
}

/// Whether this run may read or write the project's own files — its scene, its
/// dock layout (§6 M15.1, M15.2), and the player's own two (§6 M42). False under
/// `--record` and `--replay`: a blessed stream must land its clicks against the
/// layout the gate recorded with, and must leave no `scene.ggsave` behind for
/// the next run to open.
pub(crate) fn may_touch_project(args: &Args) -> bool {
    args.record.is_none() && args.replay.is_none()
}

/// One of the player's own files, or `None` when this run may not have it (§6
/// M42). Two things have to be true: the game has a data directory at all —
/// only a project does, a dev run writing under `target/` like everything else —
/// and the run is live. The live rule is [`opening_scene`]'s, for a sharper
/// reason: a knob here moves the pipeline a recorded click travels (§6 M40) and
/// a preference here *is* hashed world state, so a file the game wrote itself
/// would diverge a blessed stream with nobody having typed anything.
pub(crate) fn player_file(args: &Args, name: &str) -> Option<PathBuf> {
    may_touch_project(args)
        .then_some(args.data.as_ref())
        .flatten()
        .map(|dir| dir.join(name))
}

/// One spelling of the handoff flag: [`Args`] parses it and
/// [`gg_core::reload::rejuvenate::restart`] passes it on, and a drift between
/// the two would make a twice-rejuvenated session accumulate argv.
const RESTORE_FLAG: &str = "--restore";

/// The player's session, in the data directory beside `settings.cfg` (§6 M44).
///
/// A fourth file policy, and the three that exist do not cover it. It is a
/// [`gg_ecs::world::save::Save`], so it inherits §4.5's *may gain, never lose* —
/// but nobody typed `--load` to open it, so a build that cannot read one must
/// not refuse to start the way `--load` does. The answer is the one thing a
/// forgiving read cannot be: **skipped and left alone**. The scores are lost to
/// this build and recovered by going back to the one that wrote them, where
/// decoding it leniently would have destroyed them at the next exit.
const PROGRESS: &str = "progress.ggsave";

/// The config file, read from the working directory. Not having written one is
/// the normal case (§4.8), so there is no flag to point elsewhere: a run that
/// wants one value wants `--set`, and a run that wants a different *file* is
/// choosing a different working directory anyway.
const CONFIG: &str = "gg.cfg";

/// Why the shell stopped, and whether a player saw a game first (§6 M81).
///
/// The flag is not diagnostics: it picks the first line of the box. "could not
/// start" is what a stranger reads when the folder is short its dylib, and it is
/// the wrong sentence entirely for a session that played for ten minutes and
/// then failed to write a save — a player told the former goes looking for a
/// broken download.
#[derive(Debug)]
pub struct Exit {
    /// The failure, with its whole chain.
    pub error: anyhow::Error,
    /// `false` for everything up to the first tick — argument parsing,
    /// observability, the config file, the loader.
    pub started: bool,
}

impl From<anyhow::Error> for Exit {
    /// Everything reached by `?` inside [`run`] is startup: the loop's own
    /// failure is built by hand and is the only one that sets the flag.
    fn from(error: anyhow::Error) -> Exit {
        Exit {
            error,
            started: false,
        }
    }
}

/// Boot, then sessions until nothing asks for another.
///
/// `argv` is what `gg_core::config` reads its own flags out of; `args` is what
/// this shell was told, which a launcher constructs rather than parses.
///
/// **The loop is one pass in every ordinary run.** A session ends by asking to
/// open a project (§6 M15.1 item 4) and nothing else does that, so a shell
/// pointed at a game runs exactly once and returns.
///
/// The error carries [`Exit::started`] because a box that says "could not start"
/// to a player who has been playing for ten minutes is describing a different
/// event (§6 M81).
pub fn run(mut args: Args, argv: &[String]) -> Result<(), Exit> {
    // Bound to a named local, not `_`: the guard *is* the Tracy client's
    // lifetime, and `let _ = ..` would drop it here (see `Observability`).
    // Dist has no guard to hold, so the binding is a unit there and the lint is
    // right about that one tier and wrong about the shape.
    #[cfg_attr(not(feature = "debug-tools"), allow(clippy::let_unit_value))]
    let _observability = init_observability(args.data.as_deref(), args.title.as_deref())?;
    // Hazard 5's startup call site (§4.2.1): before any dependency's initializer
    // has had a chance to vandalize MXCSR/FPCR unnoticed. Demos 00–02 do the
    // same; the shell is what every demo from 03 on actually runs under.
    #[cfg(feature = "fp-assert")]
    gg_math::fpenv::assert_fp_env();

    // One closure, so everything reached by `?` here is `anyhow` and lands on
    // `Exit`'s `From` as startup rather than as a session that began.
    (|| -> anyhow::Result<()> {
        let _startup = info_span!("startup").entered();
        info!(
            version = env!("CARGO_PKG_VERSION"),
            tier = active_tier(),
            headless = gg_platform::headless(),
            game = %args.game.display(),
            title = args.title.as_deref().unwrap_or("<no project>"),
            // The game's own, beside the engine's. Two versions rather than one
            // because they move independently and a report naming only the
            // engine's cannot say which build of the game it is about.
            game_version = args.version.as_deref().unwrap_or("<unversioned>"),
            // Where this run's saves, settings and log are going (§6 M75).
            // `None` is a run with no project, which keeps nothing — and a
            // player who cannot find their scores is the one person this line
            // is for.
            data = args.data.as_deref().map_or("<none>", |d| d.to_str().unwrap_or("<path>")),
            "golden runtime online"
        );
        // Game output has nowhere to go until this is set, and two loggers would
        // mean two answers to where a line went — so it is set once, here.
        if gg_ecs::boundary::set_logger(log_from_game).is_err() {
            warn!("a logger was already installed");
        }
        // Per-system CPU zones (§4.8). Only the instrumented graph has anywhere
        // to send them: `gg-debug` is absent from dist (§3), so the ECS's hook
        // stays unset there and its loop keeps a null check.
        #[cfg(feature = "debug-tools")]
        if gg_ecs::boundary::set_system_zone(gg_debug::cpu::system_zone).is_err() {
            warn!("a system zone was already installed");
        }
        // Every crate's knobs registered before anything is applied, or a name
        // the config file uses would be unknown by an accident of ordering.
        gg_render::cvars::register()?;
        #[cfg(feature = "debug-tools")]
        gg_debug::register()?;
        // The editor's two were registered when an editor was *built*, which is
        // after this — so `--set d.editor_scale=3` was "no such cvar" from M15
        // until §6 M40 went looking for a knob to move.
        #[cfg(feature = "editor")]
        gg_editor::register()?;
        // Beside the manifest for a shipped folder, and in the working directory
        // for every other run there has ever been (§6 M41): a game launched from
        // a shortcut must not lose its knobs to whatever directory the shortcut
        // happened to name.
        let config = args.dir.as_deref().unwrap_or(Path::new(".")).join(CONFIG);
        gg_core::config::boot(&config, player_file(&args, CONFIG).as_deref(), argv)?;
        Ok(())
    })()?;

    // The loop's failure is carried rather than raised (§6 M81): a `?` here
    // jumped straight past the verdict below, so the session most likely to have
    // lost the player's disk was the one session that never said so. Whatever
    // went wrong, the sentence about their files is still owed.
    let outcome = loop {
        match session(&args) {
            Ok(Some(next)) => args = next,
            Ok(None) => break Ok(()),
            // Whether it started is the session's own answer and not this
            // loop's: everything up to the frame loop is a game that never
            // opened (§6 M81 post-close — this arm said `true` for all of it,
            // so a folder short its dylib told a stranger their *session* had
            // ended, and `xtask reload --refuse` went red unwatched).
            Err(exit) => break Err(exit),
        }
    };
    // After every session and once per process (§6 M54). A game whose disk
    // refused exits 0 — it played, and the exit code is about the session — so
    // this is the only place the fact reaches anyone but a log reader. `alert`
    // is a no-op under `GG_HEADLESS` that says so (§1.5), which is what lets a
    // gate assert the sentence without an operator dismissing a box.
    if let Some(verdict) = player::verdict() {
        let title = args.title.as_deref().unwrap_or(env!("CARGO_PKG_NAME"));
        let body = unsaved(title, args.data.as_deref(), &verdict);
        error!(
            failures = verdict.failures,
            writes = verdict.writes,
            reason = verdict.reason,
            "the player's files were not written"
        );
        gg_platform::alert(title, &body);
    }
    outcome
}

/// One session: a world over a game, driven to the end of its loop.
///
/// `Some` is a project the operator picked out of the launcher (§6 M15.1 item
/// 4) — the arguments the *next* session runs under, which is a new shell state
/// and not a swap, because a session is built around the dylib it was pointed
/// at. Everything else about the run carries over, `--frames` included.
fn session(args: &Args) -> Result<Option<Args>, Exit> {
    // Every `?` below the frame loop is startup and converts through `Exit`'s
    // `From` as one; the loop itself and everything after it is a session that
    // ran, and says so by hand. Two spellings rather than a flag threaded
    // through, so a new `?` in the setup half cannot get the sentence wrong.
    let began = |error: anyhow::Error| Exit {
        error,
        started: true,
    };
    let staging = std::env::temp_dir().join(format!("gg-runtime-{}", std::process::id()));
    // Windows recycles PIDs and this directory outlives the run that made it —
    // the staged dylib is deliberately never unloaded (§4.2.2), so no
    // destructor of ours can delete a copy the loader still has mapped. The
    // clear is therefore the only end that works, and it is also the one that
    // matters: without it a successor inherits a predecessor's generations
    // (§6 M89).
    let _ = std::fs::remove_dir_all(&staging);
    let bindings = match &args.input {
        Some(path) => {
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?
        }
        None => String::new(),
    };
    let replay = args
        .replay
        .as_ref()
        .map(|path| -> anyhow::Result<_> { Ok(Box::new(Replay::decode(&std::fs::read(path)?)?)) })
        .transpose()?;
    // The player's own keys (§6 M45), read here beside the project's and applied
    // over them. A live session only, like every other file a player owns: a
    // recorded stream drives verb *ids*, so a rebind cannot reach one — but the
    // legend a game draws from the map is world state, and a file the operator
    // edited would otherwise diverge a blessed stream with nobody having played.
    // Absent and unreadable are different sentences (§6 M84): the first is the
    // ordinary case and says nothing, the second is a file the player has and
    // this build did not honour, which they can only act on if it is named.
    // Nothing writes this one back, so there is no overwrite to refuse — the
    // whole loss is the silence, which is why it read `.ok()` for a milestone.
    let rebinds = match player_file(args, gg_input::BINDINGS_FILE) {
        Some(path) if path.is_file() => std::fs::read_to_string(&path).unwrap_or_else(|error| {
            warn!(%error, path = %path.display(),
                  "bindings: found and unreadable — the game's own keys stand");
            String::new()
        }),
        _ => String::new(),
    };
    // Read here rather than at the window, so a headless run reports a folder
    // that ships a picture it cannot draw — the only tier a gate may run (§1.5).
    let icon = icon(args);
    let mut app = app::App::new(args, &staging, DEFAULT_TICK_HZ, bindings, rebinds, replay)?;
    // Before the first frame: the loop's clock resumes at the tick this carries.
    //
    // **And it decides every source below it (§6 M84).** Five things write the
    // world at boot — this, `--load`, `progress`, the opening scene, and
    // `settings` — and until M84 exactly one pair of the ten was reconciled. A
    // `--restore` is not a sixth source competing with the rest: it says this
    // process is a session *continuing*, and the four below describe how a
    // session *starts*. The predecessor already spent them; their effect is in
    // the world it handed over. Applying one again does not add anything, it
    // rewinds — to a checkpoint up to `CHECKPOINT_SECONDS` stale, or to the
    // authored level, or to the preferences the player has since changed.
    //
    // `restart` re-execs with the whole argv (`rejuvenate::restart`), which is
    // how the successor finds its game — and is also why `--load` is on this
    // list rather than exempt from it: the flag the operator typed once is
    // handed to every successor forever.
    // What this session may overwrite: a continuation inherits its predecessor's
    // refusals, a fresh boot starts permitted and earns one by failing to read a
    // file (§6 M44's rule, generalised).
    let mut writable = gg_core::reload::rejuvenate::Writable::default();
    let mut continuing = false;
    if let Some(path) = &args.restore {
        let handoff = gg_core::Handoff::take(path).map_err(anyhow::Error::from)?;
        (writable, continuing) = (handoff.writable, true);
        app.restore(&handoff)?;
    }
    if let Some(path) = &args.load
        && !continuing
    {
        app.load_save(path)?;
    }
    // The player's own session (§6 M44), ahead of the opening scene and never
    // beside it: the scene is what a *new* player gets, and this is what the
    // last one left. A refusal is a warning here rather than an error — see
    // `PROGRESS` — and it takes the exit write down with it.
    // `--load` suppresses the *read* and not the write: the flag says where the
    // session starts, this says where it left off, and every session leaves off
    // somewhere. The editor is out of both, like `--save`.
    let progress = player_file(args, PROGRESS).filter(|_| !args.editor);
    let mut resumed = false;
    if let Some(path) = progress
        .as_deref()
        .filter(|path| path.is_file() && args.load.is_none() && !continuing)
    {
        match app.load_save(path) {
            Ok(()) => {
                info!(progress = %path.display(), "resuming the player's session");
                resumed = true;
            }
            Err(error) => {
                warn!(%error, "progress: this build cannot read it — starting fresh, file untouched");
                writable.progress = false;
            }
        }
    }
    // The project's opening scene (§6 M15.2 post-close; §6 M20 pull 2): the
    // world as data — the editor opens Stopped at the save's tick with no game
    // code run, and a plain run opens the level the project checked in.
    if let Some(scene) = opening_scene(args).filter(|_| !resumed && !continuing) {
        info!(scene = %scene.display(), "opening scene");
        app.load_save(&scene)?;
    }
    // The player's own settings (§6 M42). Absent is *nothing applied*, not a
    // world at defaults: a game whose opening `Prefs` is not all zeros (demo 12
    // asks for MSAA) keeps what it declared until there is a file, and the file
    // is written from the world, so the first exit records it.
    let settings = player_file(args, gg_ecs::boundary::settings::FILE);
    if let Some(path) = settings.as_deref().filter(|_| !continuing) {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let (prefs, stale) = gg_ecs::boundary::settings::decode(&text);
                if !stale.is_empty() {
                    warn!(
                        ?stale,
                        "settings: lines this build does not answer to — ignored"
                    );
                }
                app.want_settings(prefs);
            }
            // Absent is the ordinary case and the comment above is its whole
            // policy: nothing applied, and the first exit writes what the game
            // declared.
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            // Present and unreadable is not that (§6 M84). Until M84 this arm
            // was `.ok()`, so a `settings.cfg` saved as UTF-16 by the editor the
            // file's own header invites — "Yours to edit" — read as absent, and
            // the exit write below then replaced the player's choices with the
            // game's defaults. §6 M44 gave `progress` this rule and this file
            // never got it: what cannot be read must not be overwritten.
            Err(error) => {
                warn!(%error, path = %path.display(),
                      "settings: found and unreadable — the game's own stand, file untouched");
                writable.settings = false;
            }
        }
    }
    // Everything below writes rather than reads, so it is filtered by permission
    // and never by `continuing`: a successor *is* the session now, and a session
    // that stopped checkpointing because it was handed on is one that loses the
    // very seconds the restart existed to preserve.
    let progress = progress.filter(|_| writable.progress);
    let settings = settings.filter(|_| writable.settings);
    app.may_overwrite(writable);
    // From here the session is on the disk as it goes (§6 M48), not only at the
    // exit it may never reach. Same path and same policy as the exit write —
    // Permission included, because a file this build could not read is one it
    // must not overwrite, and a checkpoint is the write that would. Both paths
    // are already filtered by `Writable` above, so this cannot disagree with the
    // exit write below — one place decides and two places obey (§6 M84).
    app.checkpoint_to(progress.clone(), settings.clone());

    // Headless is windowless, not invisible-windowed: §1.5 forbids an automated
    // tier from creating an OS window *at all*, and every `xtask ci` tier sets
    // `GG_HEADLESS=1`. The sim, the reload path and the replay stream are the
    // same either way, which is what lets CI exercise the shell it ships.
    // A replay bounds its own run: the file says how many ticks it covers, and
    // honouring `--frames` over it is how a replay stops reproducing (§4.7).
    let target = args.frames.or_else(|| app.ticks());
    let frames = if gg_platform::headless() {
        let target = target.with_context(
            || "a headless run needs --frames N or --replay: there is no window to close it (§1.5)",
        )?;
        FrameLoop::locked(DEFAULT_TICK_HZ)
            .resuming_at(app.next_tick())
            .run(&mut app, target)
            .map_err(began)?
    } else {
        let title = args.title.clone().unwrap_or_else(|| app.title());
        play::play(&mut app, &title, target, args.window, icon).map_err(began)?
    };

    // First of everything after the loop, because every line below it either
    // writes a player file or leaves the process: the last tick has run, so
    // nothing further will be offered, and bytes still in flight would land on
    // top of an exit write and roll the session back an interval (§6 M48). The
    // rejuvenation restart is the same hazard one step further out — the
    // successor opens these files, and a writer from this process outliving the
    // exec would overwrite what it reads.
    app.checkpoint_stop();
    // The window is down and the GPU is accounted for (§4.3), which is the only
    // point a session may be handed on. Never returns on success.
    if let Some(handoff) = app.handoff() {
        gg_core::reload::rejuvenate::restart(&handoff, RESTORE_FLAG)
            .map_err(|e| began(e.into()))?;
    }
    // Before `finish`, which consumes the app: what a save holds is the world,
    // and the world is gone once the recorder has been taken out of it.
    // Not with the editor open: there `--save` names where its *button* writes,
    // and a second write at exit would bury what the operator actually did.
    if let Some(path) = &args.save
        && !args.editor
    {
        app.write_save(path).map_err(began)?;
    }
    // Same window as the save, and the same reason: what a preference is read
    // out of is the world, and the world goes with the app (§6 M42).
    if let Some(path) = &settings {
        app.write_settings(path);
    }
    // And the session itself (§6 M44). Written however the session ended, which
    // is what leaves the decision with the game: demo 10's EXIT files the score
    // and deals a fresh board before setting `Prefs::close`, so a deliberate
    // quit reopens on the title — and a window closed mid-game reopens on that
    // game, because the game got no tick in which to decide otherwise. A
    // failure here is counted, never fatal: the process is already leaving, and
    // what it says about the disk is `player::note`'s and is said once (§6 M54).
    if let Some(path) = progress {
        let _ = app.write_save(&path);
    }
    // Also before `finish`, and the reason is the same one: the pick is the
    // editor's and the editor goes with the app.
    #[cfg(feature = "editor")]
    let next = app.opening().map(|project| {
        info!(project = project.name, game = %project.game.display(), "opening project");
        Args {
            game: project.game.clone(),
            input: project.input.clone(),
            pack: project.pack.clone(),
            // Only what describes the *run* carries. A recording cannot cross
            // this seam at all — a replay's id space is the verb list it was
            // made against (§4.7), and the launcher's is the editor's appended
            // verbs over no game, which is not any game's. The save and restore
            // flags are the same argument one step out: they named a session,
            // and this is the next one.
            frames: args.frames,
            editor: args.editor,
            editor_extent: args.editor_extent,
            leak_budget: args.leak_budget,
            ..Args::default()
        }
    });
    #[cfg(not(feature = "editor"))]
    let next = None;
    if let (Some(path), Some(recorder)) = (&args.record, app.finish()) {
        let replay = recorder.finish();
        std::fs::write(path, replay.encode()).map_err(|e| began(e.into()))?;
        info!(path = %path.display(), ticks = replay.ticks(),
              changes = replay.change_count(), "replay written");
    }
    info!(frames, "golden runtime clean exit");
    Ok(next)
}

/// `--editor-extent`'s value. Public because the launcher takes the same flag
/// (§6 M15.1 item 4) and two spellings of one parse would drift.
pub fn parse_extent(text: &str) -> anyhow::Result<(u32, u32)> {
    let (w, h) = text
        .split_once(['x', 'X'])
        .with_context(|| format!("--editor-extent wants <w>x<h>, got `{text}`"))?;
    Ok((w.trim().parse()?, h.trim().parse()?))
}

/// The command line, as [`Args`].
pub fn parse_args(argv: &[String]) -> anyhow::Result<Args> {
    let mut args = Args::default();
    let mut project = None;
    let mut argv = argv.iter().cloned();
    while let Some(flag) = argv.next() {
        let mut value = || argv.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            // Consumed and dropped: `gg_core::config` reads the same argv for
            // these, and the shell's job here is only to not call them unknown.
            gg_core::config::SET_FLAG => drop(value()?),
            "--game" => args.game = PathBuf::from(value()?),
            "--frames" => args.frames = Some(value()?.parse()?),
            "--input" => args.input = Some(PathBuf::from(value()?)),
            "--record" => args.record = Some(PathBuf::from(value()?)),
            "--replay" => args.replay = Some(PathBuf::from(value()?)),
            RESTORE_FLAG => args.restore = Some(PathBuf::from(value()?)),
            "--load" => args.load = Some(PathBuf::from(value()?)),
            "--save" => args.save = Some(PathBuf::from(value()?)),
            "--play" => args.play = Some(value()?),
            "--away" => args.away = Some(value()?),
            "--editor" => {
                anyhow::ensure!(
                    cfg!(feature = "editor"),
                    "--editor: this tier has no editor (§3 keeps it out of the dist graph)"
                );
                args.editor = true;
            }
            "--editor-extent" => args.editor_extent = Some(parse_extent(&value()?)?),
            "--leak-budget" => args.leak_budget = Some(value()?.parse()?),
            "--pack" => args.pack = Some(PathBuf::from(value()?)),
            "--project" => project = Some(PathBuf::from(value()?)),
            other => anyhow::bail!("unknown argument `{other}`"),
        }
    }
    // A shipped folder has no command line (§6 M41): the manifest beside the
    // executable is what a double-click passes. Argv wins field by field, which
    // is what keeps every dev-tree invocation and every gate byte-unchanged.
    // The probe happens *only* when no `--game` was given, so a run with a
    // command line never stats a file it did not name.
    let asked = project.is_some() || args.game.as_os_str().is_empty();
    if let Some(found) = if asked {
        Project::found(project.as_deref())?
    } else {
        None
    } {
        args.data = found.data_dir();
        args.dir = Some(found.dir);
        args.title = Some(found.title);
        args.version = found.version.map(|v| v.text().to_owned());
        args.window = found.window;
        args.icon = found.icon;
        args.input = args.input.take().or(found.input);
        args.pack = args.pack.take().or(found.pack);
        // Field by field like the two above, which `game` was not until §6 M42
        // went looking for a way to give a *gate* a data directory: `--project`
        // with a `--game` beside it had blanked the dylib, because a demo's
        // manifest names none and this assigned unconditionally. Argv-wins is
        // the same rule either way — a shipped folder passes no `--game`.
        if args.game.as_os_str().is_empty() {
            args.game = found.game;
        }
    }
    anyhow::ensure!(
        args.record.is_none() || args.replay.is_none(),
        "--record and --replay are two answers to where this run's input comes from"
    );
    anyhow::ensure!(
        args.editor_extent.is_none_or(|(w, h)| w > 0 && h > 0),
        "--editor-extent must be positive: a zero-sized editor lays out nothing"
    );
    // The launcher reaches the same state by *constructing* `Args`, not by
    // omitting a flag: a command line with no `--game` is a mistake, and a
    // shell that opened an empty editor over one would hide it (§6 M15.1 item 4).
    anyhow::ensure!(
        !args.game.as_os_str().is_empty(),
        "--game <path to the game dylib> is required: the shell is the same program in every \
         tier and the game is what makes it a game (§2)"
    );
    Ok(args)
}

/// Game `log` calls, on the host's stream. A forwarding function rather than a
/// `tracing` dependency inside `gg-ecs` (§3) — this is the line that buys.
fn log_from_game(level: u32, message: &str) {
    use gg_ecs::boundary::log_level;
    match level {
        log_level::ERROR => tracing::error!(target: "game", "{message}"),
        log_level::WARN => tracing::warn!(target: "game", "{message}"),
        log_level::DEBUG => tracing::debug!(target: "game", "{message}"),
        log_level::TRACE => tracing::trace!(target: "game", "{message}"),
        _ => tracing::info!(target: "game", "{message}"),
    }
}

/// Which named tier combination this binary was built as (§3). Tiers are
/// meta-features, so exactly one of these is expected per build.
pub fn active_tier() -> &'static str {
    if cfg!(feature = "tier-dev") {
        "dev"
    } else if cfg!(feature = "tier-instrumented") {
        "instrumented"
    } else if cfg!(feature = "tier-dist-verify") {
        "dist-verify"
    } else {
        "dist"
    }
}

/// `gg::hash` off by default: it is one line per sim tick (§5.6c) and wanted
/// only by the gate that compares two runs, which asks for it by `RUST_LOG`.
const LOG_FILTER: &str = "debug,gg::hash=off";

/// Where a shipped run's log goes, in the game's own directory (§6 M41 item 4).
const LOG_FILE: &str = "log.txt";

/// Set when this run actually opened its log file — which is not the same
/// question as whether it has a directory to put one in (§6 M54).
static LOGGING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The file a refusal may point someone at, or `None` when this run's log is a
/// console they are already looking at (§6 M47) — naming a file that does not
/// exist is worse than naming nothing. `cfg!` rather than two bodies so the
/// constant above stays referenced in both tiers.
///
/// The atomic is the second half of that same sentence: a data directory the
/// run could not write is a path a box would otherwise print, sending whoever
/// reads it to look for a file nothing ever created.
fn log_path(data: Option<&Path>) -> Option<PathBuf> {
    match cfg!(feature = "debug-tools") || !LOGGING.load(std::sync::atomic::Ordering::Relaxed) {
        true => None,
        false => data.map(|dir| dir.join(LOG_FILE)),
    }
}

/// What a refusal says to whoever is holding the mouse (§6 M47): the title, so
/// they know what failed; the chain, so they know what to say; the log's path,
/// so a bug report can carry something. Nothing else — a message box is read
/// standing up, and the device table and the stack are in the file it names.
///
/// Separate from — and public alongside — [`refuse`] because it is the only
/// half a test can read: the box is the operator's and stderr is a stream.
/// `started` picks the first line and is the whole of §6 M81's half of this: a
/// failure after the loop began did not fail to *start*, and telling a player it
/// did sends them looking for a broken download.
pub fn refusal(title: &str, error: &anyhow::Error, log: Option<&Path>, started: bool) -> String {
    let opening = if started {
        "stopped and could not continue"
    } else {
        "could not start"
    };
    let mut body = format!("{title} {opening}.\n\n{error}");
    for cause in error.chain().skip(1) {
        body.push_str(&format!("\n\ncaused by: {cause}"));
    }
    if let Some(log) = log {
        body.push_str(&format!("\n\nThe full log is at\n{}", log.display()));
    }
    body
}

/// What a *crash* says, on [`refusal`]'s shape and with one word different (§6
/// M48). The word is the whole difference a player cares about: a refusal never
/// started and lost them nothing, while this had their session and the sentence
/// after it is where they find out how much of it survived.
///
/// `what` is the panic's own message, already formatted — a `PanicHookInfo` is
/// not constructible outside a hook, and the string is the part worth testing.
/// `saved` is [`player::checkpointing`] and not an assumption: the sentence is
/// the most valuable line in the box and a session that was never checkpointing
/// — a replay, a recording, a run with no project — must not be told it.
#[must_use]
pub fn crashed(title: &str, what: &str, log: Option<&Path>, saved: bool) -> String {
    let mut body = format!("{title} stopped unexpectedly.\n\n{what}");
    if saved {
        let seconds = player::CHECKPOINT_SECONDS;
        body.push_str(&format!(
            "\n\nYour progress was saved up to about {seconds} seconds before this."
        ));
    }
    if let Some(log) = log {
        body.push_str(&format!("\n\nThe full log is at\n{}", log.display()));
    }
    body
}

/// What a session whose disk refused says on the way out (§6 M54), on the shape
/// [`refusal`] and [`crashed`] share.
///
/// The tense is the difference: those two describe something the player watched
/// happen, and this describes something that did *not* — a game that played
/// perfectly and saved none of it. Nothing here is a diagnosis, because the
/// causes (a folder in Program Files, a full disk, a scanner holding the file,
/// a directory that is a file) are indistinguishable from in here and the one
/// the OS named is the only honest one to print.
///
/// The directory is the first thing said and the reason the second: a player who
/// is told at exit can move the folder somewhere writable, and one who is told
/// nothing finds out by losing the evening.
#[must_use]
pub fn unsaved(title: &str, dir: Option<&Path>, verdict: &player::Verdict) -> String {
    let mut body = format!("{title} could not save your progress.");
    if let Some(dir) = dir {
        body.push_str(&format!("\n\nIt was writing to\n{}", dir.display()));
    }
    body.push_str(&format!("\n\n{}", verdict.reason));
    // Some against none, because they are different evenings: a session that
    // banked ten checkpoints and then lost the disk kept everything up to the
    // last one, and saying "none of it" to that player would be its own lie.
    body.push_str(match verdict.writes {
        0 => "\n\nNothing from this session was saved.",
        _ => "\n\nSome of this session was saved; the most recent part was not.",
    });
    body
}

/// The shell's last words (§6 M47). Returns the process's code, so `main` is an
/// expression and the one thing it must not do — carry on — is unspellable.
///
/// Called from `main` and not from [`run`] because the earliest failures are
/// `parse_args`'s, which happen before `run` is entered at all; the title and
/// the data directory are arguments for the same reason, since the parse that
/// would have found them is the thing that failed.
///
/// Three sinks, none redundant: the log is what a bug report carries, stderr is
/// the dev tree's answer and every gate's, and the box is the only one a player
/// standing in front of a shipped folder will ever see. The subscriber is a
/// global and outlives `run`'s guard, so the log line lands for every failure
/// that got as far as opening one — the ones that did not are exactly
/// `parse_args`'s, which is what the other two are for.
pub fn refuse(title: Option<&str>, data: Option<&Path>, exit: &Exit) -> std::process::ExitCode {
    let error = &exit.error;
    let said = if exit.started {
        "the session ended early"
    } else {
        "refused to start"
    };
    tracing::error!(error = %format!("{error:#}"), "{said}");
    eprintln!("Error: {error:?}");
    let title = title.unwrap_or(env!("CARGO_PKG_NAME"));
    let log = log_path(data);
    gg_platform::alert(title, &refusal(title, error, log.as_deref(), exit.started));
    std::process::ExitCode::FAILURE
}

/// The instruments (§4.8): Tracy, the log tail a crash report attaches, the
/// console. `gg-debug` is absent from every dist graph by §3, so this is two
/// bodies rather than one with a flag in it — dist keeps the terminal and loses
/// the rest, which is the whole difference.
#[cfg(feature = "debug-tools")]
fn init_observability(
    _data: Option<&Path>,
    _title: Option<&str>,
) -> anyhow::Result<gg_debug::Guard> {
    let guard = gg_debug::init(LOG_FILTER)?;
    // After the tail exists, never before: a report from a process whose logging
    // had not come up yet would attach nothing (§4.8).
    gg_debug::crash::install(gg_debug::crash::Product {
        name: env!("CARGO_PKG_NAME"),
        version: env!("CARGO_PKG_VERSION"),
        tier: active_tier(),
    });
    Ok(guard)
}

#[cfg(not(feature = "debug-tools"))]
fn init_observability(data: Option<&Path>, title: Option<&str>) -> anyhow::Result<()> {
    use tracing_subscriber::fmt::writer::BoxMakeWriter;
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(LOG_FILTER));
    // A shipped folder is linked without a console (§6 M41 item 4), so stdout is
    // a handle nothing is behind: the log is a file in the game's own directory
    // or it is nothing, and nothing is what an itch bug report otherwise holds.
    // Truncated per run rather than rolled — the interesting session is the one
    // that just went wrong, and a rolling appender is a dependency.
    //
    // **Degraded, never fatal (§6 M54).** Until that milestone this was a `?`,
    // so a player whose `%LOCALAPPDATA%` could not be written did not get a game
    // that failed to save — they got one that would not *start*, over a file
    // that exists to describe failures. Diagnostic equipment does not get a veto
    // on the session it is there to observe.
    let mut degraded = None;
    let sink = data
        .and_then(|dir| {
            match (|| -> std::io::Result<_> {
                std::fs::create_dir_all(dir)?;
                std::fs::File::create(dir.join(LOG_FILE))
            })() {
                Ok(file) => {
                    LOGGING.store(true, std::sync::atomic::Ordering::Relaxed);
                    Some(BoxMakeWriter::new(std::sync::Arc::new(file)))
                }
                // Kept, not printed: there is no subscriber to print it to yet,
                // and stderr behind a shipped binary is a handle nothing reads.
                Err(e) => {
                    degraded = Some((dir.to_path_buf(), e));
                    None
                }
            }
        })
        .unwrap_or_else(|| BoxMakeWriter::new(std::io::stdout));
    tracing_subscriber::fmt()
        .with_target(true)
        .with_env_filter(filter)
        .with_ansi(false)
        .with_writer(sink)
        .try_init()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    // The first thing this run says, now that there is somewhere to say it. It
    // goes to a console nobody shipped, which is exactly the point: the sentence
    // is for whoever later asks why the folder has no `log.txt` in it.
    if let Some((dir, e)) = degraded {
        warn!(dir = %dir.display(), error = %e,
              "no log file - this session runs and reports nothing to disk");
    }
    // §3 keeps `gg-debug` and its crash reporter out of the dist graph, so a
    // panic's last words are this hook's or a player watches a window vanish.
    // The words reached the log at M47 and the *window* still vanished (§6 M48):
    // a shipped binary's stderr is a handle nothing is behind, so the box below
    // is the only sink a player has for the failure that costs them a session
    // rather than a launch.
    let previous = std::panic::take_hook();
    let name = title.unwrap_or(env!("CARGO_PKG_NAME")).to_owned();
    let log = log_path(data);
    std::panic::set_hook(Box::new(move |info| {
        tracing::error!(tier = active_tier(), "{info}");
        previous(info);
        // Once per process. `alert` blocks until it is dismissed, so a second
        // panic — on another thread, or inside the box's own event pump — would
        // stack a modal window on top of the failure being reported and bury
        // the first one behind it.
        static SHOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !SHOWN.swap(true, std::sync::atomic::Ordering::SeqCst) {
            let body = crashed(
                &name,
                &info.to_string(),
                log.as_deref(),
                player::checkpointing(),
            );
            gg_platform::alert(&name, &body);
        }
    }));
    Ok(())
}
