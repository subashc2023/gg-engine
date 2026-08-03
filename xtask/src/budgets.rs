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
//! crate's `[dependencies]`, workspace-internal ones included** (§3 says the
//! forced `gg-ecs-derive` leaf "counts as one"), and dev-dependencies excluded
//! (`gg-ecs`'s own manifest already says they are outside the budget, because
//! they are absent from every runtime graph).

use std::path::Path;

use crate::util::{cargo, run_capture, walk_rs, workspace_root};

/// The `gg-runtime` code-line budget (§3). Raised 300 → 500 at M5 when the shell
/// grew the window, the renderer's three calls, live input and record/replay,
/// 500 → 600 at M8 for the observability stack (config, the instruments, the
/// overlay, the crash handler, the capture trigger), and 600 → 1000 at M13 when
/// the UI stage arrived: a `gg_ui::Ui` per tick, the canvas→window fit, and the
/// verb lookup that feeds it. Every raise is the same argument — the shell
/// *chooses* these and implements none of them — and every one was spent
/// deliberately rather than discovered.
const SHELL_BUDGET: usize = 1000;

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
];

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
        let deps = manifest
            .get("dependencies")
            .and_then(toml::Value::as_table)
            .map(toml::Table::len)
            .unwrap_or(0);
        anyhow::ensure!(
            deps <= *budget,
            "{name} declares {deps} dependencies against a {budget} budget (§3) — raising it is \
             a PR that says what the crate took delivery of, not a drift"
        );
        println!("xtask: {name} dependency budget {deps}/{budget}");
    }
    Ok(())
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
        found.push(
            parsed
                .get("package")
                .and_then(|p| p.get("name"))
                .and_then(toml::Value::as_str)
                .ok_or_else(|| anyhow::anyhow!("{} has no package name", manifest.display()))?
                .to_owned(),
        );
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
