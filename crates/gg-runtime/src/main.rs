//! The shell's command line. Everything it does is [`gg_runtime`]'s — see that
//! crate's header for what the shell *is*, and §6 M15.1 item 4 for why the two
//! are a library and a binary rather than one file.

fn main() -> anyhow::Result<()> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    gg_runtime::run(gg_runtime::parse_args(&argv)?, &argv)
}
