//! The §3 complexity budgets, as machines rather than as review standards.
//!
//! Three of them, and they fail for three different reasons: the shell grows
//! *code* (orphan logic hiding in the app shell), an owned crate grows
//! *dependencies* (a charter widening one `Cargo.toml` line at a time), and a
//! game crate grows an *engine* dependency (the reload boundary's blast radius,
//! which the fingerprint's scope is sized against — §4.2.2).
//!
//! §3 called the dependency budgets "`cargo-deny`-counted" and no such count
//! existed; cargo-deny bans crates, it does not budget them. The count is here
//! instead, with its definition stated rather than implied: **every entry in a
//! crate's `[dependencies]` and `[target.'cfg(…)'.dependencies]` tables,
//! workspace-internal ones included** (§3 says the forced `gg-ecs-derive` leaf
//! "counts as one"; a platform-gated edge is still an edge), and
//! dev-dependencies excluded (`gg-ecs`'s own manifest already says they are
//! outside the budget, because they are absent from every runtime graph).

use std::path::Path;

use crate::util::{cargo, run_capture, walk_rs, workspace_root};

/// The `gg-runtime` code-line budget (§3). What licenses a raise: the shell
/// *chooses*, never *implements* — a raise must name the specific alternative
/// home it closes (an owning crate that provably cannot see both sides of
/// whatever decision moved here), not just claim more headroom. A budget that
/// only ever rises is a ratchet, not a budget — §6 M17's refactor is expected
/// to bring this number back down.
///
/// Two raises carry reasoning worth keeping close, because the "every other
/// home is closed" argument is non-obvious:
/// - **M15.1 item 4** (1100 → 1150): the shell became a library as well as a
///   binary so the editor can open with no game at all. An application-level
///   entry point can't own the boot/loop/project-dispatch sequence without
///   reimplementing the shell's own outer loop, and §2 allows exactly one of
///   those.
/// - **§6 M16** (1160 → 1300), the largest raise: the seam *record* — a
///   reload's pre-migration state hash and retired code hash on one side, the
///   migration report and first post-swap tick on the other — can only be
///   taken where both sides of the swap are visible, which is the shell and
///   nothing else.
///
/// The two most recent raises hold deliberate headroom rather than golfing to
/// the exact line count, because a zero-headroom budget is a coin flip on the
/// next comment reflow rather than a tripwire: the post-M18 audit (1300 →
/// 1310, §4.2.2's pointer-swap fast path) and the §6 M15.2 post-close pair
/// (1310 → 1335, `game_fit` and `opening_scene`) each left ~10 spare lines.
///
/// - **Render interpolation** (1335 → 1355, §4.1's `alpha`): a capture is a
///   *tick-boundary* act and a blend is a *frame* act, and one place sees both
///   clocks. `gg-core` owns the loop's shape but sits below `gg-ecs` in §3's
///   dependency direction and so cannot read a world; `gg-extract` owns every
///   line of the arithmetic — the pose table, the shortest-arc quaternion and
///   angle blends, the eye. What landed here is the four calls that place them
///   on the two clocks, and there is nowhere else to place them from.
///
/// - **The settings menu's three answers** (1355 → 1365, §6 M21): who holds the
///   mouse, which edge treatment the frame runs, and whether the player asked to
///   stop. Every piece of *judgement* moved to the crate that owns it —
///   `ActionMap::claims_motion` answers "does this game do mouse-look" from the
///   bindings, `Ui::wants_pointer` answers "is there anything to point at" from
///   the widgets, and `Prefs`' own accessors decide what an unknown mode means.
///   What is left here is the conjunction, and the conjunction is the thing with
///   no other home: `gg-ui` cannot see an action map, `gg-input` cannot see a
///   widget, and neither can see the window whose centre a released cursor is
///   warped to. The nine lines are that sentence, once.
///
/// - **Two clocks meeting** (1365 → 1370, post-M21): four lines, both of them a
///   *sequencing* fact rather than a computation, which is the one thing a stage
///   cannot state about itself.
///
///   `ticks_due` is three of them. `gg-core` owns the wall time a frame charged
///   the tick clock and sits below `gg-input` in §3's direction, so it cannot hand
///   that to an accumulator; `gg-input` owns the accumulator and has never heard
///   of a frame. The shell is the only place both are in scope, and what it says
///   is one sentence — the travel in hand covers this much time, spend a tick of it.
///
///   `cast_shadows` is the fourth. The reach is `gg-render`'s (its split scheme)
///   and the sweep is `gg-extract`'s (its frustum), but the *order* — the sun has
///   to be extracted before the instances are culled against it — is a property of
///   the frame the shell composes and of nothing either crate can see.
///
/// - **A replay and the CVar registry meeting** (1370 → 1385, §6 M40): nine
///   lines, and the same argument the two above make.
///
///   `gg-input` owns the replay and has never heard of a CVar — it depends on
///   `gg-abi` and `thiserror`, which is the whole list, and adding the registry
///   to it would point the input crate at configuration. `gg-core` owns the
///   registry and cannot see a tick, which is why `cvar::apply` takes pairs. So
///   the diff, the seam call and the apply are three lines here because the shell
///   is the only place a file and a knob are both in scope — plus the surface
///   recorded once at tick 0, and `gg_editor::register` moved up beside every
///   other crate's, which is the ordering bug this milestone found.
///
/// - **A folder instead of a command line** (1385 → 1435, §6 M41): the largest
///   raise since M16's, and the only one whose subject is what the shell *is*
///   rather than what it wires together.
///
///   Every line of it answers "where did this run's arguments come from", which
///   is the one question no engine crate may answer: `gg-core` parses the
///   manifest and finds it (`Project::found`, so the probe policy sits beside the
///   format), but `Args` is the shell's own type and mapping one onto the other
///   is the shell's whole job — a session is what it was told to run. The four
///   fields are title, window, the project's directory and the player's; the merge
///   is six lines of "argv wins"; and the rest is the log a shipped run has
///   nowhere else to write, since §3 keeps `gg-debug` and its crash reporter out
///   of the dist graph and the linker takes the console away (§6 M41 item 4).
///
///   Note what did *not* land here: the format, the probe, the extent parse and
///   the slug all went to `gg-core` where they are tested without a shell, and
///   the assembly went to `xtask ship`. What is left is the conjunction, once.
///
/// - **The player's own two files** (1435 → 1485, §6 M42): the settings file and
///   the user `gg.cfg`, and the argument is M41's one step further in.
///
///   Where a player's bytes go is `gg-core`'s (`Project::data_dir`) and what a
///   preference *is* is `gg-ecs`'s (`Prefs`, whose fields are the file's keys and
///   whose codec sits beside them). Neither crate can see the other: `gg-ecs` has
///   never heard of a data directory and does no file IO at all — `Save` is bytes
///   and the shell writes them — while `gg-core` cannot name a boundary component
///   without pointing the loop at the ECS. So the shell is again the only place
///   both are in scope, and what it says is: read here, apply on tick 0, write
///   back at exit. The precedence lives in `gg_core::config::boot` and the policy
///   in `player_file` — one predicate, reused for both files.
///
///   The tick-0 apply is the half with no other home *at all*: a scene arrives as
///   the whole world and can land before tick 0, but a `Prefs` is spawned by the
///   game's own bootstrap, so the only moment a file can reach it is after the
///   systems have run once — which is a fact about the frame the shell composes
///   and about nothing either crate can see.
///
/// - **The clips a game plays** (1485 → 1515, §6 M43): open the pack, walk the
///   entries of one kind, hand the bank over. Thirty lines, and the same shape
///   of argument a third time — with the novelty that here *two* crates are
///   each a step away from doing it, and for different reasons.
///
///   `gg-audio` is the obvious home and is the one place it must not be. Its
///   §3 budget exists to keep a decoder out (its manifest says so in as many
///   words), and a pack reader is the first step of one; it also has five of
///   six slots spent, so the file that opened a pack would be the file that
///   closed the budget. `gg-render` already opens this very file and is the
///   other near miss — but it opens it for the GPU, and under `GG_HEADLESS=1`
///   there is no renderer at all, while a headless run still fires cues and a
///   gate still has to ask whether one resolved.
///
///   So the shell is once more the only place both halves are in scope, and
///   what it says is thirty lines long: a pack that will not open is silence
///   and a log line, a blob that will not read is one clip skipped, and the
///   bank goes to `gg-audio` as a slice — which is exactly the interface that
///   let `gg-audio` spend none of its slots.
///
/// - **The session a player left** (1515 → 1545, §6 M44): the file M42's own
///   directory was missing, and the twenty-odd lines that read it before the
///   opening scene and write it after the loop. It is the *shell's* by the
///   argument M42 already made about `settings.cfg` — `player_file`'s live-only
///   rule is here, the data directory is here, and the world both halves need is
///   here — but it is also the one player file whose policy could not be written
///   anywhere else: `--load` refuses a save it cannot read and this **skips**
///   one, because refusing to launch would brick a patched game and forgiving
///   the loss would destroy the scores at the next exit. That third answer is
///   two lines of `match` and a `keep_progress` flag, and there is nowhere below
///   the shell that knows a launch is at stake.
///
/// - **The keys a player owns** (1545 -> 1585, §6 M45): three seams, and each is
///   one the shell is the only place for. The player's `bindings.cfg` is read
///   beside `settings.cfg` and through the same `player_file` rule, so it is
///   where that rule already lives. The binding table reaches a game through
///   `TickCtx`, which the shell is the only thing that builds. And arbitration
///   is one `match` over the frame between the recorder and the systems table —
///   the recorder holds what the operator did and the game must be handed what
///   the map allows, and there is exactly one line where those two are both in
///   scope. What did **not** stay here is the cost that could move: the
///   spellings and the `keep` mask are `gg_input::Input`'s caches, rebuilt
///   wherever the map or the context stack moves, because a shell that rebuilt
///   them by hand is a shell that forgets to at the next reload.
///
/// - **The window a player owns** (1585 -> 1640, §6 M46): the shell took
///   delivery of the window at M5 and of its *mode* here, which is the same
///   argument a milestone later. Three seams and no fourth. `Prefs::display` is
///   read where every other preference is read and applied in the one closure
///   where a `Window` exists — and applied against what the OS says rather than
///   against a remembered flag, because a player can leave fullscreen by ways
///   nothing here hears about. Alt+Enter is an arm beside Escape's for Escape's
///   stated reason, and it is *unclaimable* where Escape is not: a chord has no
///   spelling an action map can hold, so there is nowhere below the shell for it
///   to live. And the manifest's icon is read here rather than in `parse_args`
///   because the subscriber does not exist yet there — a policy whose whole
///   content is a warning has to be somewhere the warning prints. What did not
///   stay: the present mode is one call into `gg-render`, which owns `r.vsync`
///   and the write-back, and the icon's *format* is `gg-core`'s beside the
///   manifest that names it.
///
/// - **The refusal a player never sees** (1640 -> 1680, §6 M47): the shell is
///   the only thing that holds all three of the game's name, its data directory
///   and the error, so it is the only place the sentence can be *composed* — and
///   `main` is the only place it can be *said*, because a failure in
///   `parse_args` happens before `run` is entered at all, which is why the title
///   and the directory are arguments rather than fields read from somewhere.
///   What did not stay here, and this is the load-bearing half: the decision
///   whether to show anything is `gg-platform`'s, because a message box is an OS
///   window and §1.5 gets one place to watch — an alert put in the shell would
///   be a second window birth site the law does not enforce at. The text a
///   *device* refusal carries stayed in `gg-rhi` for the same reason a milestone
///   over: the crate that knows the fact writes the sentence.
///
/// - **The bytes a crash takes with it** (1680 -> 1820, §6 M48): the largest
///   raise since M5 and the one with the least argument to make, because the
///   responsibility was already here and only half-discharged. The shell owns
///   *which* file, *when*, and *whether this run may* — `player_file`, the exit
///   window, `keep_progress` — and until this milestone it exercised all three
///   exactly once per session, at a line a killed process never reaches. What
///   arrives is the rest of the same job: replacing a player's bytes without
///   passing through a state where they have half of them, and doing it while
///   the session is still running. Both are in `player.rs` rather than in the
///   two functions that call them, which is why the module exists at all — a
///   `replace` written inline at `write_save` is one `write_settings` would not
///   have used. What did **not** stay here: nothing moved out, and that is the
///   honest report. The candidates were real — a save's durability reads like
///   `gg-ecs`'s and a temp-then-rename reads like an OS detail — and both were
///   refused for the same reason, that `gg-ecs` owns bytes and never files and
///   `gg-platform`'s charter is windows. The line is who owns the player's
///   directory, and the answer has been the shell since M42.
///
/// - **The seconds a player was not looking** (1820 -> 1870, §6 M49): a
///   *conjunction* with one place it is in scope, which is the M21 bullet's
///   shape exactly. Whether a frame runs a tick is "the window is not focused"
///   **and** "the world asked to wait", and the two halves live on opposite
///   sides of §3's dependency direction: `gg-platform` raises the focus event
///   and cannot read a world — nor does it exist in a headless run, where every
///   gate that grades this one runs — while `gg-core` owns the loop, the clock
///   and `Stages::suspended` and sits *below* `gg-ecs`, so it cannot read a
///   `Prefs` either. Every piece of judgement is in the crate that owns it, as
///   before: `TickClock::hold` decides what a frame that must not tick reports,
///   `Prefs::pauses_unfocused` decides what an unknown constant means, and
///   `Audio::hush` decides what silence costs a sounding voice. What is left
///   here is that sentence, once, plus `Away` — which is beside `PlayMode` for
///   `PlayMode`'s own reason: a script that drives a shipping path has to live
///   where the path does, or the tier grades a second one.
///
/// - **The disk that says no** (1870 -> 1945, §6 M54): the largest raise since
///   §6 M16, and it buys one sentence the shell is the only thing in a position
///   to say — *the player's files did not get written*. Every piece of judgement
///   is again in the crate that owns it: `player::replace` decides what an
///   atomic write is, `gg_platform::alert` decides what a box is and what
///   headless means for one, and the game decides what it wanted saved. What is
///   here is the *record* — which files were tried, which of them last failed,
///   and the fact that a scanner holding one write is not a disk that refuses —
///   and it has no other home for a reason with two halves. Downward, `gg-ecs`
///   owns bytes and never files and `gg-platform`'s charter is windows, which is
///   M48's line and unchanged. Upward, the answer must **not** be the world: a
///   disk failure is host state, and a `Prefs` field or a component carrying it
///   would make a replay diverge on exactly the machine a bug report comes
///   from. That leaves the shell, which sees the session end, the directory it
///   was writing to, and the title to put on the box, and is the one place all
///   three are in scope at once. The other two thirds are the same argument
///   about equipment: the log sink degrades instead of refusing (a `?` there had
///   been a launch veto held by the file that exists to describe failures), and
///   `log_path` stops naming a file nothing opened.
///
/// Full raise history, one line each: §6 M5, M8, M13, M15.1 (title bar), M15.2
/// (play mode), M18 item 2 (audio), M43 (clips), M44 (the session), M45 (the
/// keys), M46 (the window), M47 (the refusal), M48 (the crash) — each argued the
/// same way.
const SHELL_BUDGET: usize = 1945;

/// Per-crate dependency budgets (§3). Only the crates §3 actually names carry
/// one; a budget invented here would be a rule this file made up.
const DEPENDENCY_BUDGETS: &[(&str, usize)] = &[
    ("gg-ecs", 6),
    ("gg-core", 8),
    ("gg-ui", 10),
    // §6 M15's editor. It consumes engine crates and adds nothing of its own,
    // which is the budget's whole argument — a `gg-editor` that had grown its
    // own dependencies would be a second engine, which is what §6 M15 says it
    // must not be.
    ("gg-editor", 10),
    // §6 M18 item 2. The one place in this tree where the convenient dependency
    // is a *decoder* — and a decoder is what would make `cpal` stop being a
    // rental and start being a stack we did not choose.
    ("gg-audio", 6),
];

/// §3's `gg-ui` acceptance rule, as a machine: the M13 overlay reimplementation
/// "may not exceed the M8 overlay's line count by more than 2×". §3 says the
/// machine-checkable budgets are CI and this one never was — found by M17's read
/// of the tree against the document, which is the one budget §3 states in lines
/// and left to review.
///
/// 510 is the M8 overlay, recorded in §6 M13's status, so the cap is 1020. A
/// constant and not a re-measurement: the crate it was measured against is gone,
/// and a gate that recomputed its own baseline would forgive any drift it was
/// standing next to. What it protects is not the overlay — it is `gg-ui`, whose
/// exit test was that a UI library which cannot cheaply do what 510 lines of
/// immediate-mode drawing did is overbuilt.
///
/// 510 was a **total**-line measurement (§6 M13 records the M13 file as "574
/// lines" on the same basis), so the gate compares total lines — comparing the
/// code-line count against a total-derived cap quietly doubled the intended
/// slack, which the post-M18 audit caught.
const OVERLAY_BUDGET: usize = 1020;

/// Where the widget vocabulary is declared, and where it is turned into
/// geometry. Both are *read* rather than listed, so a kind added to either shows
/// up in [`widget_provenance`] on its own.
const WIDGET_PROTOCOL: &str = "crates/gg-ecs/src/boundary/ui.rs";
const WIDGET_DRAW: &str = "crates/gg-ui/src/boundary.rs";

/// Where the device-address blocks are declared and unpacked (§6 M28).
///
/// This file's structs are not read by the GPU as structs: they mirror a
/// `repr(C)` Rust record byte for byte, and a hand-written loader lifts them out
/// of a raw pointer. The size agreement is asserted on both sides — and *whether
/// every field is actually assigned* was not, which is how §6 M27 shipped three
/// fields the shader declared and `load_frame` never filled.
const SHADER_BLOCKS: &str = "crates/gg-render/shaders/include/pbr.slang";

/// §6 M12's exit row: the template reaches a spinning lit mesh in under 50
/// lines. A budget rather than a claim, because the number is the whole point —
/// it is what caps the ceremony a game pays, and the first time it was measured
/// it came out at 74 and sent two helpers down into `GameWorld` where every
/// game crate had been hand-writing them.
const TEMPLATE_BUDGET: usize = 50;

/// What a game crate may reach engine-side (§3's deny pin, §4.2.2's blast
/// radius). `gg-ecs-derive` is the forced proc-macro leaf `gg-ecs` re-exports,
/// so a game crate reaches it whether or not it names it.
const GAME_CRATE_PIN: &[&str] = &["gg-abi", "gg-ecs", "gg-ecs-derive", "gg-math"];

/// §4.10's reference-set cap. Per-backend PNG sets grow monotonically and a repo
/// that takes minutes to clone fails §9's fresh-clone bar long before git
/// complains; crossing this is a decision (LFS, or a references sub-repo) made in
/// a PR, not discovered when a clone crawls.
const REFERENCE_BUDGET: u64 = 50 * 1024 * 1024;

/// `(crate, dependency, why)` for edges a textual scan cannot see — a dependency
/// reached only through a macro expansion, or linked for its symbols alone.
///
/// **Empty, and deliberately so**, on the same reasoning as the validation
/// suppressions file: the escape hatch exists before it is needed, so the first
/// real case gets a row with a reason instead of the gate getting switched off.
const USED_INVISIBLY: &[(&str, &str, &str)] = &[];

pub fn check() -> anyhow::Result<()> {
    shell_lines()?;
    template_lines()?;
    overlay_lines()?;
    widget_provenance()?;
    shader_block_loaders()?;
    dependencies()?;
    unused_dependencies()?;
    game_crate_pin()?;
    reference_images()
}

/// The template's ceremony, counted (§6 M12).
///
/// Code lines, on the same definition [`shell_lines`] uses and for the same
/// reason: the house comment style is dense and inline, and counting comments
/// would make the two rules fight — resolved by deleting the explanations that
/// are half of what a template is *for*.
fn template_lines() -> anyhow::Result<()> {
    let path = workspace_root().join("demos/99-template/src/lib.rs");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("no template at {}: {e}", path.display()))?;
    let (code, total) = count_code(&text);
    anyhow::ensure!(
        code <= TEMPLATE_BUDGET,
        "demos/99-template is {code} code lines against a {TEMPLATE_BUDGET}-line budget (§6 M12) \
         — the fix is to move the ceremony into the boundary where every game crate gets it, \
         not to raise the number"
    );
    println!("xtask: template budget {code}/{TEMPLATE_BUDGET} code lines ({total} total)");
    Ok(())
}

/// `gg-ui`'s acceptance test, counted (§3, §6 M13).
fn overlay_lines() -> anyhow::Result<()> {
    let path = workspace_root().join("crates/gg-debug/src/overlay.rs");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("no overlay at {}: {e}", path.display()))?;
    let (code, total) = count_code(&text);
    anyhow::ensure!(
        total <= OVERLAY_BUDGET,
        "the overlay is {total} total lines against a {OVERLAY_BUDGET}-line budget (§3) — that cap \
         is 2x the M8 overlay it replaced, and exceeding it is a statement about `gg-ui` rather \
         than about the overlay: a UI library the same screen costs twice as much to draw on is \
         overbuilt (§6 M13's acceptance test)"
    );
    println!("xtask: overlay budget {total}/{OVERLAY_BUDGET} total lines ({code} code, §3)");
    Ok(())
}

/// §3's other `gg-ui` acceptance rule, the one stated in prose: **no widget
/// without a demo that needs it**. The line-count rule above caps how expensive
/// the library is to draw with; this one caps its *vocabulary*, which is the
/// half that grows silently — a kind arrives because the editor wanted it, the
/// editor is host code, and no game ever asks for it again.
///
/// So provenance rather than a count: every kind `gg-ecs`' protocol declares
/// must be reached by a crate under `demos/` that builds a `cdylib` (§2's
/// Game-code boundary row, the same definition the deny pin uses), and must have
/// a `gg-ui` arm that draws it. The second half is not the same claim as the
/// first: [`widget`](gg_ecs) documents that an *unknown* kind draws nothing,
/// which is tolerance for a game sending garbage across the boundary, not a
/// licence for a declared kind to be invisible.
///
/// Reached counts the constant *or* the constructor that sets it — `Widget`'s
/// helpers are how a game names a kind in practice, and a gate that only saw
/// `widget::LABEL` would report demo 10 as having no labels while it draws
/// three.
fn widget_provenance() -> anyhow::Result<()> {
    let root = workspace_root();
    let protocol = std::fs::read_to_string(root.join(WIDGET_PROTOCOL))?;
    let drawn = std::fs::read_to_string(root.join(WIDGET_DRAW))?;
    let kinds = widget_kinds(&protocol);
    anyhow::ensure!(
        !kinds.is_empty(),
        "no widget kinds found in {WIDGET_PROTOCOL} — a check that finds nothing to check passes \
         vacuously (§5.8's rule, applied to §3's `gg-ui` rule)"
    );
    let games: Vec<(String, String)> = game_crate_dirs()?
        .into_iter()
        .map(|(name, dir)| {
            let mut sources = Vec::new();
            walk_rs(&dir, &mut sources);
            let text = sources
                .iter()
                .filter_map(|f| std::fs::read_to_string(f).ok())
                .collect();
            (name, text)
        })
        .collect();
    let (offenders, provenance) = judge_widgets(&kinds, &drawn, &games);
    for line in &provenance {
        println!("xtask: {line}");
    }
    anyhow::ensure!(
        offenders.is_empty(),
        "widget provenance (§3's `no widget without a demo that needs it`):\n  {}\n\nA kind the \
         editor alone wants is host code's business and belongs in `gg-ui`'s own draw list, not \
         in the boundary every game declares against",
        offenders.join("\n  ")
    );
    println!(
        "xtask: {} widget kind(s), each drawn and each needed by a demo (§3)",
        kinds.len()
    );
    Ok(())
}

/// Every field of every block struct in the shader is assigned by a loader (§6
/// M28).
///
/// The failure this exists for is silent in a way the stride assertions are not.
/// A `Frame` field nobody writes is not a compile error in Slang and not a size
/// change on either side; it reads as *whatever was in the register*, which on
/// the pinned rasterizer is zero — so the feature it gates turns itself off and
/// the picture that results is merely a plausible one. §6 M27's prefiltered
/// chain reached the skybox and no shaded surface for exactly this reason, and
/// three blessed references recorded the fallback as though it were the feature.
///
/// The rule is deliberately textual and deliberately weak: a field named `x` on
/// a struct must appear as `.x =` somewhere in the same file. It cannot check
/// that the *offset* is right — the stride assertions and the reference images
/// do that — only that nothing was left behind, which is the half that had no
/// gate at all.
fn shader_block_loaders() -> anyhow::Result<()> {
    let path = workspace_root().join(SHADER_BLOCKS);
    let text = std::fs::read_to_string(&path)
        .map_err(|e| anyhow::anyhow!("no shader blocks at {}: {e}", path.display()))?;
    let blocks = shader_structs(&text);
    anyhow::ensure!(
        !blocks.is_empty(),
        "no block structs found in {SHADER_BLOCKS} — a check that finds nothing to check passes \
         vacuously (§5.8's rule)"
    );
    let fields: usize = blocks.iter().map(|(_, f)| f.len()).sum();
    let offenders = judge_blocks(&text, &blocks);
    anyhow::ensure!(
        offenders.is_empty(),
        "shader block loaders (§6 M28, and §6 M27's defect):\n  {}",
        offenders.join("\n  ")
    );
    println!(
        "xtask: {} shader block(s), {fields} field(s), each one loaded",
        blocks.len()
    );
    Ok(())
}

/// Which declared fields nothing in `text` assigns.
fn judge_blocks(text: &str, blocks: &[(String, Vec<String>)]) -> Vec<String> {
    let mut offenders = Vec::new();
    for (name, declared) in blocks {
        for field in declared {
            if !text.contains(&format!(".{field} =")) {
                offenders.push(format!(
                    "{name}::{field} is declared and never loaded — it reads as whatever the \
                     register held, and the feature it gates silently does not run"
                ));
            }
        }
    }
    offenders
}

/// `(name, fields)` per `struct` in a Slang source, by brace depth rather than
/// by regex — a struct whose fields were reformatted onto one line each is the
/// only shape this has to read, and it is the shape the house style writes.
fn shader_structs(text: &str) -> Vec<(String, Vec<String>)> {
    let mut out = Vec::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let Some(name) = line.trim().strip_prefix("struct ") else {
            continue;
        };
        let name = name.trim().trim_end_matches('{').trim().to_string();
        let mut fields = Vec::new();
        for body in lines.by_ref() {
            let body = body.trim();
            if body == "}" || body == "};" {
                break;
            }
            // The trailing comment first: the house style puts prose after the
            // field, and a sentence with a semicolon in it would otherwise be
            // parsed as a second field named after its last word.
            let body = body.split("//").next().unwrap_or(body).trim();
            // `type name;` or `type name; // why`. A method or a nested brace
            // would end the field list rather than be misread as one.
            let Some(declaration) = body.split(';').next().filter(|_| body.contains(';')) else {
                continue;
            };
            if let Some(field) = declaration.split_whitespace().last() {
                let field = field.trim_end_matches(|c: char| !c.is_alphanumeric() && c != '_');
                // Arrays declare as `float y[N]`; the name is what precedes the
                // bracket.
                let field = field.split('[').next().unwrap_or(field);
                if !field.is_empty() {
                    fields.push(field.to_string());
                }
            }
        }
        if !fields.is_empty() {
            out.push((name, fields));
        }
    }
    out
}

/// `(offenders, one provenance line per covered kind)`.
///
/// Split out of the read so `mod tests` can plant both directions — a gate that
/// has only ever been shown a clean tree is the thing §5 keeps calling a budget
/// nobody has watched go red.
fn judge_widgets(
    kinds: &[(String, Vec<String>)],
    drawn: &str,
    games: &[(String, String)],
) -> (Vec<String>, Vec<String>) {
    let (mut offenders, mut provenance) = (Vec::new(), Vec::new());
    for (kind, constructors) in kinds {
        if !mentions(drawn, &format!("widget::{kind}")) {
            offenders.push(format!(
                "`{kind}` is declared and {WIDGET_DRAW} draws no arm for it"
            ));
        }
        // Qualified, so `label` cannot be satisfied by a local of that name.
        let needs: Vec<&str> = games
            .iter()
            .filter(|(_, text)| {
                mentions(text, &format!("widget::{kind}"))
                    || constructors
                        .iter()
                        .any(|c| text.contains(&format!("Widget::{c}")))
            })
            .map(|(name, _)| name.as_str())
            .collect();
        if needs.is_empty() {
            offenders.push(format!(
                "`{kind}` is declared and no game crate reaches it (constructors: {constructors:?})"
            ));
        } else {
            provenance.push(format!(
                "widget::{kind} — needed by {} (§3)",
                needs.join(", ")
            ));
        }
    }
    (offenders, provenance)
}

/// `(kind, constructors that set it)` off the protocol's own source.
///
/// Comment lines are skipped: the field docs name kinds in prose, and a scan
/// that read them would credit a kind to whichever function happened to be last.
fn widget_kinds(text: &str) -> Vec<(String, Vec<String>)> {
    let mut kinds: Vec<(String, Vec<String>)> = Vec::new();
    let mut in_mod = false;
    let mut current_fn: Option<String> = None;
    for line in text.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            continue;
        }
        if trimmed.starts_with("pub mod widget {") {
            in_mod = true;
        } else if in_mod && line == "}" {
            in_mod = false;
        } else if in_mod {
            if let Some(name) = trimmed
                .strip_prefix("pub const ")
                .and_then(|rest| rest.split(':').next())
            {
                kinds.push((name.to_owned(), Vec::new()));
            }
        } else if let Some(rest) = trimmed.strip_prefix("pub fn ") {
            current_fn = rest.split('(').next().map(str::to_owned);
        }
        if let (Some(func), Some(at)) = (&current_fn, line.find("widget::")) {
            let name: String = line[at + "widget::".len()..]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if let Some((_, ctors)) = kinds.iter_mut().find(|(kind, _)| *kind == name)
                && !ctors.contains(func)
            {
                ctors.push(func.clone());
            }
        }
    }
    kinds
}

/// `(code, total)` lines: code is non-blank and not a `//` comment.
fn count_code(text: &str) -> (usize, usize) {
    let (mut code, mut total) = (0usize, 0usize);
    for line in text.lines() {
        total += 1;
        let line = line.trim_start();
        if !line.is_empty() && !line.starts_with("//") {
            code += 1;
        }
    }
    (code, total)
}

/// Every declared dependency must appear in the crate that declares it.
///
/// The §3 budgets count dependencies for two crates; nothing counted whether a
/// declared one is *reached*, and an unused edge costs a build, a `cargo-deny`
/// surface and an audit line while buying nothing. Textual rather than
/// `cargo-udeps`: this must run in the push tier on pinned stable, and udeps
/// needs a nightly and a full build. The cost of that choice is false positives,
/// paid down by [`USED_INVISIBLY`] rather than by weakening the check.
fn unused_dependencies() -> anyhow::Result<()> {
    let root = workspace_root();
    let mut checked = 0usize;
    let mut offenders = Vec::new();
    for crate_dir in workspace_members(&root)? {
        let manifest: toml::Value =
            toml::from_str(&std::fs::read_to_string(crate_dir.join("Cargo.toml"))?)?;
        let package = crate_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        let mut sources = Vec::new();
        walk_rs(&crate_dir, &mut sources);
        let text: String = sources
            .iter()
            .filter_map(|f| std::fs::read_to_string(f).ok())
            .collect();
        for name in declared_dependencies(&manifest) {
            if USED_INVISIBLY
                .iter()
                .any(|(krate, dep, _)| *dep == name && crate_dir.ends_with(krate))
            {
                continue;
            }
            checked += 1;
            if !mentions(&text, &name.replace('-', "_")) {
                offenders.push(format!("{package} declares `{name}` and never reaches it"));
            }
        }
    }
    anyhow::ensure!(
        offenders.is_empty(),
        "unused dependencies (§3):\n  {}\n\nDelete the line, or — if the use is real and \
         invisible to a textual scan — add it to USED_INVISIBLY with the reason",
        offenders.join("\n  ")
    );
    println!("xtask: {checked} declared dependencies, all reached (§3)");
    Ok(())
}

/// Whole-identifier match: `gg_math` must not be satisfied by `gg_math_sim`, and
/// a substring test would make the gate pass on names that merely overlap.
fn mentions(text: &str, ident: &str) -> bool {
    let boundary = |c: char| !(c.is_alphanumeric() || c == '_');
    text.match_indices(ident).any(|(at, _)| {
        let before = text[..at].chars().next_back().is_none_or(boundary);
        let after = text[at + ident.len()..].chars().next().is_none_or(boundary);
        before && after
    })
}

/// Dependency names from every table a manifest can declare one in, including
/// the `[target.'cfg(...)'.…]` ones — a platform-gated edge is still an edge.
fn declared_dependencies(manifest: &toml::Value) -> Vec<String> {
    const TABLES: [&str; 3] = ["dependencies", "dev-dependencies", "build-dependencies"];
    let mut names = Vec::new();
    let mut take = |table: Option<&toml::Value>| {
        if let Some(table) = table.and_then(toml::Value::as_table) {
            names.extend(table.keys().cloned());
        }
    };
    for table in TABLES {
        take(manifest.get(table));
    }
    if let Some(targets) = manifest.get("target").and_then(toml::Value::as_table) {
        for spec in targets.values() {
            for table in TABLES {
                take(spec.get(table));
            }
        }
    }
    names.sort();
    names.dedup();
    names
}

/// Workspace member directories. Read off the manifest rather than globbed, so a
/// member added without a `members` entry is invisible to this gate for the same
/// reason it is invisible to `cargo`.
fn workspace_members(root: &Path) -> anyhow::Result<Vec<std::path::PathBuf>> {
    let manifest: toml::Value = toml::from_str(&std::fs::read_to_string(root.join("Cargo.toml"))?)?;
    let members = manifest
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(toml::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("the workspace manifest declares no members"))?;
    Ok(members
        .iter()
        .filter_map(toml::Value::as_str)
        .map(|m| root.join(m))
        .filter(|p| p.join("Cargo.toml").is_file())
        .collect())
}

/// The golden reference sets, weighed (§4.10).
fn reference_images() -> anyhow::Result<()> {
    fn weigh(dir: &Path, total: &mut u64, count: &mut usize) -> anyhow::Result<()> {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return Ok(()); // no references yet is not a budget failure
        };
        for entry in entries {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                weigh(&entry.path(), total, count)?;
            } else {
                *total += entry.metadata()?.len();
                *count += 1;
            }
        }
        Ok(())
    }
    let (mut total, mut count) = (0u64, 0usize);
    weigh(
        &workspace_root().join("tests/gg-images"),
        &mut total,
        &mut count,
    )?;
    anyhow::ensure!(
        total <= REFERENCE_BUDGET,
        "golden references are {total} B across {count} file(s) against a {REFERENCE_BUDGET} B \
         budget (§4.10) — Git LFS or a references sub-repository is the decision to make, not a \
         higher number"
    );
    println!(
        "xtask: golden references {} KiB / {} MiB across {count} file(s) (§4.10)",
        total / 1024,
        REFERENCE_BUDGET / (1024 * 1024)
    );
    Ok(())
}

/// Complexity budgets, the CI-counted lines (§3).
///
/// Code rather than every line, because §3's phrase is "thin in code" and the
/// house comment style is dense and inline — counting both would make the two
/// rules fight, and the way that fight resolves is by deleting comments to fit a
/// shell-size cap. The total is printed beside it so comment mass stays visible.
fn shell_lines() -> anyhow::Result<()> {
    let mut files = Vec::new();
    walk_rs(&workspace_root().join("crates/gg-runtime/src"), &mut files);
    let (mut code, mut total) = (0usize, 0usize);
    for text in files.iter().filter_map(|f| std::fs::read_to_string(f).ok()) {
        let (c, t) = count_code(&text);
        code += c;
        total += t;
    }
    anyhow::ensure!(
        code <= SHELL_BUDGET,
        "gg-runtime is {code} code lines against a {SHELL_BUDGET}-line budget (§3) — \
         raising the budget is a PR, not a drift"
    );
    println!("xtask: gg-runtime line budget {code}/{SHELL_BUDGET} code lines ({total} total)");
    Ok(())
}

/// The per-crate dependency budgets of §3, counted off the manifests.
fn dependencies() -> anyhow::Result<()> {
    let root = workspace_root();
    for (name, budget) in DEPENDENCY_BUDGETS {
        let manifest: toml::Value = toml::from_str(&std::fs::read_to_string(
            root.join("crates").join(name).join("Cargo.toml"),
        )?)?;
        let deps = budgeted_dependencies(&manifest).len();
        anyhow::ensure!(
            deps <= *budget,
            "{name} declares {deps} dependencies against a {budget} budget (§3) — raising it is \
             a PR that says what the crate took delivery of, not a drift"
        );
        println!("xtask: {name} dependency budget {deps}/{budget}");
    }
    Ok(())
}

/// What the budget counts: every name in `[dependencies]` **and** in the
/// `[target.'cfg(…)'.dependencies]` tables, deduped — a platform-gated edge is
/// still an edge, the same reading [`declared_dependencies`] gives the
/// unused-deps gate, and `gg-platform`'s `windows-sys` shows the idiom is
/// already in the tree. Dev- and build-dependencies stay outside the budget
/// (absent from every runtime graph).
fn budgeted_dependencies(manifest: &toml::Value) -> Vec<String> {
    let mut names: Vec<String> = manifest
        .get("dependencies")
        .and_then(toml::Value::as_table)
        .map(|t| t.keys().cloned().collect())
        .unwrap_or_default();
    if let Some(targets) = manifest.get("target").and_then(toml::Value::as_table) {
        for spec in targets.values() {
            if let Some(table) = spec.get("dependencies").and_then(toml::Value::as_table) {
                names.extend(table.keys().cloned());
            }
        }
    }
    // One name under two cfgs is one delivery, not two.
    names.sort();
    names.dedup();
    names
}

/// §3's game-crate deny pin, which had no machine behind it: a game crate may
/// reach [`GAME_CRATE_PIN`] engine-side and nothing else.
///
/// This is what makes the boundary fingerprint's scope and a dylib's possible
/// link set the same list (§4.2.2) — an engine crate arriving in a game graph
/// widens the blast radius without widening the fingerprint, and that is a
/// silent hole rather than a loud one. Third-party dependencies are the game's
/// own business; the pin is engine-side by construction.
///
/// Game crates are found rather than listed: a `demos/` package that builds a
/// `cdylib` *is* game code (§2's Game-code boundary row), so demo 04 is covered
/// the day it exists and not the day someone remembers to add it here.
fn game_crate_pin() -> anyhow::Result<()> {
    let root = workspace_root();
    let games = game_crates()?;
    anyhow::ensure!(
        !games.is_empty(),
        "the game-crate deny pin matched no crate — a check that finds nothing to check passes \
         vacuously (§5.8's rule, applied to §3's pin)"
    );
    for name in &games {
        check_one_game_crate(name, &root)?;
    }
    Ok(())
}

/// Every game crate in the workspace, by package name.
///
/// Found rather than listed: a `demos/` package that builds a `cdylib` *is* game
/// code (§2's Game-code boundary row), so demo 04 is covered the day it exists
/// and not the day someone remembers to add it to a constant. Shared with the
/// dist gate, which has the same question and must not answer it differently.
pub fn game_crates() -> anyhow::Result<Vec<String>> {
    Ok(game_crate_dirs()?
        .into_iter()
        .map(|(name, _)| name)
        .collect())
}

/// The same set, with the directory each was found in — what a gate that reads
/// game *source* needs (see [`widget_provenance`]), and what the dist gate's
/// run leg needs to know which demos declare an `assets/` tree.
pub fn game_crate_dirs() -> anyhow::Result<Vec<(String, std::path::PathBuf)>> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(workspace_root().join("demos"))?.flatten() {
        let manifest = entry.path().join("Cargo.toml");
        if !manifest.is_file() {
            continue;
        }
        let parsed: toml::Value = toml::from_str(&std::fs::read_to_string(&manifest)?)?;
        let builds_a_cdylib = parsed
            .get("lib")
            .and_then(|l| l.get("crate-type"))
            .and_then(toml::Value::as_array)
            .is_some_and(|kinds| kinds.iter().any(|k| k.as_str() == Some("cdylib")));
        if !builds_a_cdylib {
            continue;
        }
        let name = parsed
            .get("package")
            .and_then(|p| p.get("name"))
            .and_then(toml::Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("{} has no package name", manifest.display()))?
            .to_owned();
        found.push((name, entry.path()));
    }
    found.sort();
    Ok(found)
}

fn check_one_game_crate(name: &str, root: &Path) -> anyhow::Result<()> {
    // The resolved graph, not the manifest: the pin is about what a dylib can
    // *link*, and a transitive engine crate links exactly as hard as a declared
    // one.
    let tree = run_capture(
        cargo()
            .current_dir(root)
            .args(["tree", "-p", name, "-e", "normal", "--prefix", "none"]),
        &format!("cargo tree ({name} link set)"),
    )?;
    let mut offenders: Vec<&str> = tree
        .lines()
        .filter_map(|l| l.split_whitespace().next())
        .filter(|c| c.starts_with("gg-") && !GAME_CRATE_PIN.contains(c))
        .collect();
    offenders.sort_unstable();
    offenders.dedup();
    anyhow::ensure!(
        offenders.is_empty(),
        "game crate {name} links {offenders:?} — §3 pins game crates to {GAME_CRATE_PIN:?}, which \
         is what keeps the §4.2.2 fingerprint's scope and a dylib's link set the same list"
    );
    println!("xtask: {name} links only the pinned boundary crates (§3)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real shader, parsed — the same reason the widget scan pins its own
    /// protocol: a block file reformatted into something this cannot read would
    /// otherwise leave the gate finding nothing and passing.
    #[test]
    fn the_real_shader_blocks_parse_and_every_field_is_loaded() {
        let text = std::fs::read_to_string(workspace_root().join(SHADER_BLOCKS)).unwrap();
        let blocks = shader_structs(&text);
        let names: Vec<&str> = blocks.iter().map(|(n, _)| n.as_str()).collect();
        assert!(
            names.contains(&"Frame") && names.contains(&"Environment"),
            "the two blocks the frame buffer is made of: {names:?}"
        );
        assert!(judge_blocks(&text, &blocks).is_empty());
    }

    /// Red in the direction that matters, which is §6 M27's actual defect: a
    /// field declared in the struct and dropped from the loader.
    #[test]
    fn a_declared_field_no_loader_fills_is_named() {
        let planted = "struct Frame\n{\n    uint light_count;\n    uint radiance_levels;\n}\n\
                       Frame load_frame(uint64_t a)\n{\n    frame.light_count = u[20];\n}\n";
        let blocks = shader_structs(planted);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].1, vec!["light_count", "radiance_levels"]);
        let offenders = judge_blocks(planted, &blocks);
        assert_eq!(offenders.len(), 1, "{offenders:?}");
        assert!(offenders[0].contains("Frame::radiance_levels"));
    }

    /// An array field declares as `float y[N]`, and its name is what precedes
    /// the bracket — a scan that took the whole token would look for `.y[N] =`
    /// and forgive every array in the file.
    #[test]
    fn an_array_field_is_named_without_its_extent() {
        let planted = "struct Cascade\n{\n    float4x4 view_projection;\n    float taps[4];\n}\n";
        assert_eq!(
            shader_structs(planted)[0].1,
            vec!["view_projection", "taps"]
        );
    }

    /// The real protocol, parsed. Pins the shape the gate depends on rather than
    /// a planted imitation of it: a `widget` module reorganized into something
    /// this scan cannot read would otherwise leave the gate quietly finding
    /// nothing, and the vacuity check only catches the *empty* case.
    #[test]
    fn the_real_protocol_still_parses_into_kinds_and_their_constructors() {
        let text =
            std::fs::read_to_string(workspace_root().join(WIDGET_PROTOCOL)).expect("protocol");
        let kinds = widget_kinds(&text);
        let names: Vec<&str> = kinds.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(
            names,
            ["PANEL", "LABEL", "BUTTON", "LABEL_CENTRE", "LABEL_RIGHT"],
            "{kinds:?}"
        );
        for (kind, ctors) in &kinds {
            assert!(!ctors.is_empty(), "{kind} has no constructor: {kinds:?}");
        }
    }

    fn kinds() -> Vec<(String, Vec<String>)> {
        vec![("LABEL".to_owned(), vec!["label".to_owned()])]
    }

    #[test]
    fn a_kind_a_demo_reaches_by_its_constructor_is_covered() {
        let games = [(
            "demo-10-tetris".to_owned(),
            "Widget::label(r, c, s)".to_owned(),
        )];
        let (offenders, provenance) = judge_widgets(&kinds(), "widget::LABEL => {}", &games);
        assert!(offenders.is_empty(), "{offenders:?}");
        assert_eq!(provenance.len(), 1, "{provenance:?}");
    }

    #[test]
    fn a_kind_only_the_editor_wants_is_rejected() {
        // Host code is not a game crate, so an editor-only kind reaches this
        // gate as a demo list that mentions it nowhere.
        let games = [("demo-07-ui".to_owned(), "Widget::panel(r, c)".to_owned())];
        let (offenders, _) = judge_widgets(&kinds(), "widget::LABEL => {}", &games);
        assert_eq!(offenders.len(), 1, "{offenders:?}");
        assert!(
            offenders[0].contains("no game crate reaches it"),
            "{offenders:?}"
        );
    }

    #[test]
    fn a_declared_kind_gg_ui_never_draws_is_rejected() {
        let games = [(
            "demo-10-tetris".to_owned(),
            "Widget::label(r, c, s)".to_owned(),
        )];
        let (offenders, _) = judge_widgets(&kinds(), "widget::PANEL => {}", &games);
        assert_eq!(offenders.len(), 1, "{offenders:?}");
        assert!(offenders[0].contains("draws no arm"), "{offenders:?}");
    }

    /// A name that merely overlaps must not satisfy either half — the whole
    /// point of matching whole identifiers.
    #[test]
    fn a_longer_name_does_not_stand_in_for_the_kind() {
        let games = [("demo-07-ui".to_owned(), "widget::LABELLED".to_owned())];
        let (offenders, _) = judge_widgets(&kinds(), "widget::LABELLED => {}", &games);
        assert_eq!(offenders.len(), 2, "{offenders:?}");
    }

    /// The budget counts platform-gated edges as edges and one name under two
    /// cfgs as one delivery — while dev- and build-dependencies stay outside it.
    #[test]
    fn a_target_table_edge_is_counted_once_and_dev_deps_are_not() {
        let manifest: toml::Value = toml::from_str(
            r#"
            [dependencies]
            alpha = "1"

            [dev-dependencies]
            outside = "1"

            [build-dependencies]
            also-outside = "1"

            [target.'cfg(windows)'.dependencies]
            windows-sys = "0.5"
            alpha = "1"

            [target.'cfg(unix)'.dependencies]
            windows-sys = "0.5"
            "#,
        )
        .unwrap();
        assert_eq!(budgeted_dependencies(&manifest), ["alpha", "windows-sys"]);
    }
}
