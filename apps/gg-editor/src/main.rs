//! `gg-editor` — the editor, opened without a game (§6 M15.1 item 4).
//!
//! The whole application: build the arguments a shell would have been given,
//! and hand them to the shell. There is no second loop, no second window and no
//! second copy of the wiring — `gg_runtime::run` is the same code
//! `target/debug/gg-runtime` runs, and the only difference between them is that
//! this one starts with no `--game`.
//!
//! **What "no game" is.** Not an error path: the host registers its own §4.5
//! protocol types as always, the world is empty, and the game pane shows the
//! projects under `demos/` instead of a hole with nothing behind it
//! (`gg_editor::project`). Picking one ends this session and starts the next
//! over that dylib, which is why `run` is a loop rather than a call.
//!
//! **It creates a window** and is therefore manual (§1.5), exactly as `xtask run`
//! is. `GG_HEADLESS=1` is honoured by the shell underneath, which is what lets a
//! gate drive the launcher with nothing on screen — and there a `--frames` bound
//! is required, because there is no window to close.

use anyhow::Context as _;

fn main() -> anyhow::Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    // Deliberately not `gg_runtime::parse_args`: that one *requires* `--game`,
    // and it should — a shell run from a command line with no game is a mistake.
    // The launcher is the one caller that reaches the empty state on purpose, so
    // it constructs the arguments rather than omitting a flag.
    let mut args = gg_runtime::Args {
        editor: true,
        ..gg_runtime::Args::default()
    };
    let mut rest = argv.iter().cloned();
    while let Some(flag) = rest.next() {
        let mut value = || rest.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            // `gg_core::config` reads the same argv and applies these itself; the
            // launcher's job is only to not call them unknown (the shell's own
            // `parse_args` says the same about the same flag). Re-exported rather
            // than spelled, so the two parsers cannot drift apart.
            flag if flag == gg_runtime::SET_FLAG => drop(value()?),
            "--frames" => args.frames = Some(value()?.parse()?),
            // The three that let a launcher session be *driven* rather than
            // played: what the gate uses, and the reason a click on a project is
            // provable with nothing on screen. A recording stops at the pick —
            // the next session is over a game whose verb list is not the one this
            // stream was recorded against (§4.7), and the shell drops it there.
            "--record" => args.record = Some(std::path::PathBuf::from(value()?)),
            "--replay" => args.replay = Some(std::path::PathBuf::from(value()?)),
            // The parse itself is the shell's, for the reason `SET_FLAG` above is.
            "--editor-extent" => {
                args.editor_extent = Some(gg_runtime::parse_extent(&value()?)?);
            }
            other => anyhow::bail!(
                "unknown argument `{other}`: the launcher takes --frames, --record, --replay, \
                 --editor-extent and --set. A session over a named game is `gg-runtime --game \
                 <dylib> --editor` (§6 M15.1 item 4)"
            ),
        }
    }
    gg_runtime::run(args, &argv)
}
