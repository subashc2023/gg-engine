//! The dist symbolization canary (§6 M8 exit: "a dist crash dump symbolizes
//! cleanly against the archived split-debug artifact"). Driven by `xtask dist`.
//!
//! An **example** target, deliberately: it is compiled by the shell's own crate
//! under the same `--profile dist --features tier-dist` invocation the gate
//! builds the shell with — so what it proves is that profile's codegen, fat LTO
//! and `strip` included — while entering no shipped binary. A crash affordance
//! inside `gg-runtime` itself would be product surface bought to test a gate.
//!
//! It exists because the shipping shell has no way to be told to die, and
//! should not grow one. `panic = "abort"` still runs the panic hook, so
//! `RUST_BACKTRACE=1` gets a real backtrace out of a real dist build before the
//! process goes; whether the frames carry *names* is the whole question.

use std::hint::black_box;

/// The frame the gate looks for by name. `inline(never)` because fat LTO with
/// `codegen-units = 1` would otherwise fold this into `main` and the backtrace
/// would be right about a function that no longer exists.
#[inline(never)]
fn the_pass_that_faulted(depth: u32) -> u32 {
    if black_box(depth) == 0 {
        panic!("dist symbolization canary");
    }
    black_box(depth)
}

fn main() {
    // Through `black_box` so the argument is not a constant the optimizer can
    // fold the panic out of.
    the_pass_that_faulted(black_box(0));
}
