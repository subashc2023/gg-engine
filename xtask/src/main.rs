//! `cargo xtask` — local-first CI and repo automation. `cargo xtask ci` *is* CI:
//! the definition, not a mirror (§5). Subcommands per §3: ci, shaders, assets,
//! bench, capture, probe, dist — the ones whose first consumer hasn't arrived
//! yet are stubs that say which milestone brings it.

mod bench;
mod budgets;
mod ci;
mod dist;
mod fresh;
mod probe;
mod public_api;
mod replay;
mod run;
mod shaders;
mod shell;
mod timers;
mod util;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str);
    let rest: Vec<&str> = args.iter().skip(1).map(String::as_str).collect();

    let hook = rest.contains(&"--hook");
    let result = match cmd {
        Some("ci") => ci::run(&rest),
        Some("interactive") => ci::interactive(),
        Some("probe") => probe::run(rest.contains(&"--system")),
        Some("shaders") => shaders::build_all(rest.contains(&"--check")),
        Some("dist") => dist::gate(),
        Some("public-api") => public_api::run(&rest),
        Some("replay") => replay::run(&rest),
        Some("reload") => shell::gates(&rest),
        Some("run") => run::run(&rest),
        Some("timers") => timers::run(&rest),
        Some("assets") => stub("assets", "M9 (asset pipeline; `ggc` does not exist yet)"),
        Some("bench") => bench::run(&rest),
        _ => usage(),
    };

    if let Err(err) = result {
        eprintln!("xtask: FAILED\n{err:#}");
        // Stop-hook protocol: exit 2 blocks the agent turn and feeds stderr back
        // to the agent (§6 M0A); plain failures exit 1.
        std::process::exit(if hook { 2 } else { 1 });
    }
}

fn stub(name: &str, lands: &str) -> anyhow::Result<()> {
    println!("xtask {name}: stub — the machine lands at {lands}; see PLAN.md §6.");
    Ok(())
}

fn usage() -> anyhow::Result<()> {
    anyhow::bail!(
        "usage: cargo xtask <command>\n\
         \n\
         ci [--fast|--push|--nightly|--weekly] [--hook]   local-first CI tiers (§5) — windowless by construction (§1.5)\n\
         interactive                                      manual WSI suite: storms + demo runs; two legs present visibly\n\
         run <demo> [shell flags]                         manual: build the game dylib, play it under gg-runtime (creates a window)\n\
         probe [--system]                                 capability table vs pinned lavapipe (spike 2)\n\
         shaders [--check]                                offline shader build + codegen (in-process Slang)\n\
         dist                                             dist gate: build+run tier-dist, symbol absence, crash symbolization (§5.8)\n\
         reload [--cross-tier|--segments|--chaos|--latency]  the M5 shell gates over demo 03; no flag runs the set\n\
         bench [--record]                                 smoke on lavapipe; --record archives real numbers per machine (§4.11)\n\
         assets                                           a stub until M9 (RenderDoc capture is `gg-golden capture`)"
    )
}
