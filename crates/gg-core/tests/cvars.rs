//! The CVar registry and the config that feeds it (§4.8).
//!
//! The registry is process-global, which a test cannot isolate and should not
//! try to — a registry a test could reset is not the one the engine has. So each
//! test declares its own names instead, and the file is correct whether the
//! runner gives every test its own process (nextest, what CI uses) or runs them
//! as threads in one.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_core::config;
use gg_core::cvar::{self, CVar, CVarError, CVarKind, CVarSource};

#[test]
fn a_declared_cvar_starts_at_its_default_and_reads_without_a_lookup() {
    static VSYNC: CVar = CVar::new_bool("a.vsync", true, "wait for vertical blank");
    static WIDTH: CVar = CVar::new_int("a.width", 1280, "window width in pixels");
    static SCALE: CVar = CVar::new_float("a.renderscale", 1.0, "render target scale");
    cvar::register_all(&[&VSYNC, &WIDTH, &SCALE]).unwrap();

    assert!(VSYNC.bool());
    assert!(VSYNC.is_default());
    assert_eq!(WIDTH.int(), 1280);
    assert_eq!(WIDTH.kind(), CVarKind::Int);
    assert_eq!(SCALE.float(), 1.0);
    assert_eq!(SCALE.help(), "render target scale");
}

#[test]
fn declaring_one_name_twice_is_a_startup_error() {
    // Two declarations of a name would leave half the engine reading a value the
    // other half never sees. The failure mode is silence, so the registry is
    // loud instead.
    static FIRST: CVar = CVar::new_bool("b.vsync", true, "the real one");
    static IMPOSTOR: CVar = CVar::new_bool("b.vsync", false, "a second b.vsync");
    cvar::register(&FIRST).unwrap();
    assert!(matches!(
        cvar::register(&IMPOSTOR),
        Err(CVarError::Duplicate { name: "b.vsync" })
    ));
}

#[test]
fn the_three_front_doors_agree_on_what_a_value_means() {
    // Config file, CLI and console all land on `set_from_str`, so `on` cannot
    // mean one thing in a file and another at a prompt.
    static VSYNC: CVar = CVar::new_bool("c.vsync", true, "");
    cvar::register(&VSYNC).unwrap();

    for text in ["0", "false", "off", "no"] {
        VSYNC.set_bool(true);
        cvar::set("c.vsync", text, CVarSource::Console).unwrap();
        assert!(!VSYNC.bool(), "`{text}` should read as false");
    }
    for text in ["1", "true", "on", "yes"] {
        VSYNC.set_bool(false);
        config::apply_str(&format!("c.vsync = {text}"));
        assert!(VSYNC.bool(), "`{text}` should read as true");
    }
}

#[test]
fn a_knob_refuses_a_value_of_the_wrong_shape() {
    static WIDTH: CVar = CVar::new_int("d.width", 1280, "");
    static SCALE: CVar = CVar::new_float("d.renderscale", 1.0, "");
    cvar::register_all(&[&WIDTH, &SCALE]).unwrap();

    assert!(matches!(
        cvar::set("d.width", "wide", CVarSource::Cli),
        Err(CVarError::BadValue { .. })
    ));
    // Non-finite floats are rejected by name: a render scale of NaN is a bug
    // report about the renderer three days later, not a setting.
    for text in ["inf", "-inf", "NaN"] {
        assert!(
            matches!(
                cvar::set("d.renderscale", text, CVarSource::Config),
                Err(CVarError::BadValue { .. })
            ),
            "`{text}` should not be settable"
        );
    }
    assert_eq!(WIDTH.int(), 1280, "a rejected set leaves the old value");
    assert_eq!(
        WIDTH.source(),
        CVarSource::Default,
        "and leaves the source alone: a typo is not a source"
    );
    assert!(matches!(
        cvar::set("d.nosuchthing", "1", CVarSource::Console),
        Err(CVarError::Unknown { .. })
    ));
}

#[test]
fn a_config_file_sets_what_it_names_and_reports_what_it_does_not() {
    static VSYNC: CVar = CVar::new_bool("e.vsync", true, "");
    static WIDTH: CVar = CVar::new_int("e.width", 1280, "");
    cvar::register_all(&[&VSYNC, &WIDTH]).unwrap();

    let report = config::apply_str(
        "# a comment\n\
         \n\
         e.vsync = 0\n\
         e.width = 1920   # trailing comment\n\
         e.gone = 3\n\
         e.width\n",
    );

    assert!(!VSYNC.bool());
    assert_eq!(WIDTH.int(), 1920);
    assert_eq!(report.applied, 2);
    assert_eq!(report.unknown, ["e.gone"]);
    assert_eq!(report.rejected.len(), 1, "a line that is not a setting");
    assert_eq!(report.rejected[0].name, "e.width");
    assert!(!report.is_clean());
}

#[test]
fn the_command_line_beats_the_file_because_it_is_applied_after_it() {
    // Precedence is call order and nothing cleverer, which is why this is two
    // passes over one registry rather than a merge of two parsed trees.
    static SCALE: CVar = CVar::new_float("f.renderscale", 1.0, "");
    cvar::register(&SCALE).unwrap();

    config::apply_str("f.renderscale = 0.5");
    assert_eq!(SCALE.float(), 0.5);

    let report = config::apply_args(["--frames", "100", "--set", "f.renderscale=0.75"]);
    assert_eq!(SCALE.float(), 0.75);
    assert_eq!(report.applied, 1);
    assert!(
        report.is_clean(),
        "the shell's own flags are not config errors: {report:?}"
    );
}

#[test]
fn a_malformed_setting_is_a_rejection_rather_than_a_panic() {
    let report = config::apply_args(["--set"]);
    assert_eq!(report.applied, 0);
    assert_eq!(
        report.rejected.len(),
        1,
        "trailing --set with nothing after"
    );

    let report = config::apply_args(["--set", "g.vsync"]);
    assert_eq!(report.rejected.len(), 1, "no `=` is not a setting");
}

/// §6 M42's precedence, as `boot` applies it: the game's file is what shipped,
/// the player's is what they changed about it, and a flag is someone at a
/// keyboard right now. Written as one `boot` call rather than three `apply_*`
/// ones because the *order* is what is under test, and the order is `boot`'s.
#[test]
fn the_players_file_sits_between_the_games_and_the_command_line() {
    static SCALE: CVar = CVar::new_float("k.renderscale", 1.0, "");
    static QUALITY: CVar = CVar::new_int("k.quality", 0, "");
    cvar::register_all(&[&SCALE, &QUALITY]).unwrap();

    let dir = std::env::temp_dir().join("gg-m42-precedence");
    std::fs::create_dir_all(&dir).unwrap();
    let (shipped, player) = (dir.join("gg.cfg"), dir.join("player.cfg"));
    std::fs::write(&shipped, "k.renderscale = 0.5\nk.quality = 3\n").unwrap();
    std::fs::write(&player, "k.renderscale = 0.75\n").unwrap();

    let flags = ["--set".to_owned(), "k.renderscale=0.9".to_owned()];
    config::boot(&shipped, Some(&player), &flags).unwrap();

    assert_eq!(SCALE.float(), 0.9, "a flag beats both files");
    assert_eq!(SCALE.source(), CVarSource::Cli);
    // The knob the player did not name: theirs is an overlay on the game's
    // file, never a replacement for it.
    assert_eq!(QUALITY.int(), 3);
    assert_eq!(QUALITY.source(), CVarSource::Config);

    // And with no flag, the player wins — under their own name, because "why is
    // this 0" is unanswerable if a file nobody typed reads as `config`.
    SCALE.reset();
    config::boot(&shipped, Some(&player), &[]).unwrap();
    assert_eq!(SCALE.float(), 0.75);
    assert_eq!(SCALE.source(), CVarSource::Player);

    // A player who has never opened the settings menu has no file at all.
    SCALE.reset();
    config::boot(&shipped, Some(&dir.join("absent.cfg")), &[]).unwrap();
    assert_eq!(SCALE.float(), 0.5);
    assert_eq!(SCALE.source(), CVarSource::Config);
}

#[test]
fn a_config_file_that_was_never_written_is_not_a_failure() {
    // Not having written a config yet is the normal case, not an error worth
    // failing a launch over.
    let report = config::apply_file(std::path::Path::new("no-such-config.cfg")).unwrap();
    assert!(report.is_clean());
    assert_eq!(report.applied, 0);
}

#[test]
fn the_listing_is_sorted_by_name_whatever_order_registration_happened_in() {
    // A written-back config file would otherwise be a diff of registration
    // order — §4.2.1 hazard 3 wearing a different hat.
    static Z: CVar = CVar::new_bool("h.zzz", false, "");
    static A: CVar = CVar::new_bool("h.aaa", false, "");
    cvar::register_all(&[&Z, &A]).unwrap();

    let names: Vec<&str> = cvar::all().iter().map(|c| c.name()).collect();
    let mut sorted = names.clone();
    sorted.sort_unstable();
    assert_eq!(names, sorted);
    assert_eq!(cvar::count(), names.len());
}

#[test]
fn to_text_round_trips_through_set_from_str() {
    static SCALE: CVar = CVar::new_float("i.renderscale", 1.0, "");
    cvar::register(&SCALE).unwrap();

    SCALE.set_float(0.625);
    let text = SCALE.to_text();
    SCALE.reset();
    assert!(SCALE.is_default());
    SCALE.set_from_str(&text, CVarSource::Console).unwrap();
    assert_eq!(SCALE.float(), 0.625);
}

#[test]
fn every_source_is_recorded_on_the_cvar_it_set() {
    // §4.8's exit criterion: settable from all three, and the winner is named.
    // Per-CVar rather than per-pass, because the question a session log has to
    // answer is about one knob and not about one pass.
    static SCALE: CVar = CVar::new_float("j.renderscale", 1.0, "");
    cvar::register(&SCALE).unwrap();
    assert_eq!(SCALE.source(), CVarSource::Default);

    config::apply_str("j.renderscale = 0.5");
    assert_eq!(SCALE.source(), CVarSource::Config);
    config::apply_args(["--set", "j.renderscale=0.75"]);
    assert_eq!(SCALE.source(), CVarSource::Cli, "the last writer wins both");
    cvar::set("j.renderscale", "0.9", CVarSource::Console).unwrap();
    assert_eq!(SCALE.source(), CVarSource::Console);

    // Not "the value at startup": a reset undoes the source with the value, or
    // a listing would keep crediting a config line no longer in effect.
    SCALE.reset();
    assert_eq!(SCALE.source(), CVarSource::Default);
    assert_eq!(SCALE.float(), 1.0);

    // A typed setter is a source too — the one case where a knob moves with
    // nobody having asked for it, which is exactly what a log should say.
    SCALE.set_float(0.25);
    assert_eq!(SCALE.source(), CVarSource::Code);
}
