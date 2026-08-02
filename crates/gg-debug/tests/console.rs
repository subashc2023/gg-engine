//! The CVar console's command language (§4.8) — the third front door, tested
//! without a window because the parsing and the drawing are different files on
//! purpose.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_core::cvar::{self, CVar, CVarSource};
use gg_debug::console;

/// The registry is process-global and a test that could reset it would not be
/// testing the registry the engine has, so each test declares its own names.
#[test]
fn a_bare_name_reads_and_a_name_with_a_value_sets() {
    static SCALE: CVar = CVar::new_float("t1.renderscale", 1.0, "render target scale");
    cvar::register(&SCALE).unwrap();

    let read = console::run("t1.renderscale");
    assert_eq!(read, ["t1.renderscale = 1 (default)"]);

    // The console is a source like any other, and says so on the way back out.
    assert_eq!(console::run("t1.renderscale 0.5"), ["t1.renderscale = 0.5"]);
    assert_eq!(SCALE.float(), 0.5);
    assert_eq!(SCALE.source(), CVarSource::Console);
    assert_eq!(
        console::run("t1.renderscale"),
        ["t1.renderscale = 0.5 (console)"]
    );
}

#[test]
fn a_refused_value_answers_in_words_rather_than_failing() {
    static WIDTH: CVar = CVar::new_int("t2.width", 1280, "");
    cvar::register(&WIDTH).unwrap();

    let out = console::run("t2.width wide");
    assert_eq!(out.len(), 1);
    assert!(out[0].contains("t2.width"), "{out:?}");
    assert_eq!(WIDTH.int(), 1280, "a refused set leaves the value alone");
    assert_eq!(
        WIDTH.source(),
        CVarSource::Default,
        "and the source with it"
    );

    assert_eq!(console::run("t2.nothing"), ["no cvar named `t2.nothing`"]);
    // Two values is a typo, and guessing which was meant sets the wrong knob.
    let out = console::run("t2.width 1 2");
    assert!(out[0].contains("takes one value"), "{out:?}");
}

#[test]
fn reset_puts_a_knob_back_where_its_declaration_left_it() {
    static VSYNC: CVar = CVar::new_bool("t3.vsync", true, "wait for vertical blank");
    cvar::register(&VSYNC).unwrap();

    console::run("t3.vsync off");
    assert!(!VSYNC.bool());
    // The verb comes first. Reversed, `reset` is just a value the knob refuses,
    // which is the right answer — a console that guessed would reset the wrong
    // thing when a cvar is ever named `reset`.
    let reversed = console::run("t3.vsync reset");
    assert!(reversed[0].contains("cannot take `reset`"), "{reversed:?}");
    assert_eq!(console::run("reset t3.vsync"), ["t3.vsync = 1 (default)"]);
    assert!(VSYNC.bool());
    assert_eq!(VSYNC.source(), CVarSource::Default);
}

#[test]
fn list_filters_by_prefix_and_help_explains_one() {
    static A: CVar = CVar::new_bool("t4.alpha", false, "the first");
    static B: CVar = CVar::new_bool("t4.beta", false, "the second");
    cvar::register_all(&[&A, &B]).unwrap();

    let listed = console::run("list t4.");
    assert_eq!(listed.len(), 2, "{listed:?}");
    assert!(listed[0].starts_with("t4.alpha"), "{listed:?}");
    assert!(
        listed[1].starts_with("t4.beta"),
        "sorted by name: {listed:?}"
    );
    assert!(listed[0].contains("the first"));

    assert_eq!(
        console::run("help t4.beta"),
        ["t4.beta (bool) — the second"]
    );
    assert_eq!(
        console::run("list t4.nothing"),
        ["no cvar starts with `t4.nothing`"]
    );
    // An empty line is not an error and not a reply.
    assert!(console::run("   ").is_empty());
}
