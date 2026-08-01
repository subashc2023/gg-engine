//! `cargo xtask` — local-first CI and repo automation. `cargo xtask ci` *is* CI:
//! the definition, not a mirror (§5). Subcommands per §3: ci, shaders, assets,
//! bench, capture, probe, dist — the ones whose first consumer hasn't arrived
//! yet are stubs that say which milestone brings it.

mod ci;
mod dist;
mod probe;
mod shaders;
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
        Some("assets") => stub("assets", "M9 (asset pipeline; `ggc` does not exist yet)"),
        Some("bench") => stub("bench", "M3 (first benches arrive with gg-ecs)"),
        Some("capture") => stub("capture", "M8 (RenderDoc in-application API)"),
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
         interactive                                      manual windowed suite: storms + demo WSI runs (creates windows)\n\
         probe [--system]                                 capability table vs pinned lavapipe (spike 2)\n\
         shaders [--check]                                offline shader build + codegen (in-process Slang)\n\
         dist                                             dist gate: build+run tier-dist, symbol absence (§5.8)\n\
         assets | bench | capture                         stubs until their milestone"
    )
}
