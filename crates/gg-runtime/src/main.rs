//! The shell's command line. Everything it does is [`gg_runtime`]'s — see that
//! crate's header for what the shell *is*, and §6 M15.1 item 4 for why the two
//! are a library and a binary rather than one file.
//!
//! `ExitCode` rather than `anyhow::Result`, and that is the whole of §6 M47 at
//! this end: a shipped shell is linked without a console, so Rust's own `Error:`
//! print goes to a handle nothing is behind and a player sees the screen not
//! change. [`gg_runtime::refuse`] is where a failure is said out loud instead.

fn main() -> std::process::ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    // Before the subscriber and before the manifest is understood, so a failure
    // here has neither a log file nor a title to put on the box.
    let args = match gg_runtime::parse_args(&argv) {
        Ok(args) => args,
        Err(error) => return gg_runtime::refuse(None, None, &error.into()),
    };
    // Copied out because `run` takes the args: what the box says and where the
    // log is are the two things a refusal needs and the session owns.
    let (title, data) = (args.title.clone(), args.data.clone());
    match gg_runtime::run(args, &argv) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(exit) => gg_runtime::refuse(title.as_deref(), data.as_deref(), &exit),
    }
}
