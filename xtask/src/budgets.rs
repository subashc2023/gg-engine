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
/// Full raise history, one line each: §6 M5, M8, M13, M15.1 (title bar), M15.2
/// (play mode), M18 item 2 (audio) — each argued the same way as above.
const SHELL_BUDGET: usize = 1335;

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
        assert_eq!(names, ["PANEL", "LABEL", "BUTTON"], "{kinds:?}");
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
