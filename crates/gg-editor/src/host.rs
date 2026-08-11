//! How the editor gets a pointer over a game that never asked for one (§4.7,
//! §4.9).
//!
//! `gg_ui::boundary` routes a *game's* UI through four well-known verbs the
//! game declares. An editor cannot require that: a game declaring `ui_click`
//! would be an editable game and every other one would open into a window with
//! no cursor. So the host **appends** whichever of the four the loaded build did
//! not declare, and binds them.
//!
//! The point of appending rather than opening a side channel is §6 M15's fourth
//! exit row. Editor input then lives in the same [`InputFrame`] the recorder
//! writes, indexed by ordinary verb ids, so a recorded editor session records
//! and replays through §4.7's existing machinery — no second input path, and no
//! second thing to make deterministic.
//!
//! Two consequences a reader should not have to discover:
//!
//! - The verb lists are what a replay header pins (§4.7), so a session recorded
//!   with the editor open **will not replay without it** — and must not: the
//!   ids would name different verbs.
//! - Appending never renumbers what the game declared, because it appends. A
//!   replay of a plain session is unaffected by the editor existing.
//!
//! [`InputFrame`]: gg_input::InputFrame

use gg_ecs::boundary::Verbs;
use gg_ui::boundary::verb as ui;

/// The editor's *own* verbs — what flies its camera (§6 M15.2 item 4).
///
/// Prefixed rather than borrowing a game's movement names: `move_forward` means
/// a game's character, and the editor camera has to fly a scene that has no
/// character in it. Sharing the keys is fine and sharing the verb is not.
///
/// Seven camera actions, the look pair a doubled `MAX_AXES` made affordable —
/// see [`crate::camera`] for what differencing the cursor instead cost while it
/// was not — §6 M16 item 5's two text-editing verbs, and §6 M20 item 10's
/// [`PAN`], [`FRAME`] and [`TOOL`], which are what an *orthographic* scene is
/// authored with: a fly camera's six directions describe a way of moving that
/// a flat view has no use for.
pub mod verb {
    /// Along the eye's forward axis, and against it.
    pub const FORWARD: &str = "editor_forward";
    /// See [`FORWARD`].
    pub const BACK: &str = "editor_back";
    /// Along the eye's right axis, and against it.
    pub const LEFT: &str = "editor_left";
    /// See [`LEFT`].
    pub const RIGHT: &str = "editor_right";
    /// Along **world** up, not the eye's: a fly camera whose lift tilted with
    /// its pitch cannot be flown level, which is the one thing it is for.
    pub const UP: &str = "editor_up";
    /// See [`UP`].
    pub const DOWN: &str = "editor_down";
    /// Held while a drag turns the camera.
    pub const LOOK: &str = "editor_look";
    /// Raw device motion the drag turns by, yaw then pitch.
    ///
    /// Deliberately not the `ui_x`/`ui_y` the panels point with: those are
    /// *cursor* motion, which the OS stops at the window edge, so a drag that
    /// reached it stopped turning (§6 M15.2's residual). A device delta arrives
    /// whatever the cursor is doing, which is also why it is only read while
    /// [`LOOK`] is held.
    pub const LOOK_X: &str = "editor_look_x";
    /// See [`LOOK_X`].
    pub const LOOK_Y: &str = "editor_look_y";
    /// Held while a drag slides the camera across the view plane.
    ///
    /// Turns by the same raw device pair as [`LOOK`] rather than a second one:
    /// a pan and a look are one gesture with a different button, and a third
    /// axis pair would spend `MAX_AXES` headroom to say the same numbers.
    pub const PAN: &str = "editor_pan";
    /// Put the selection in the middle of the view, at a size that fits it —
    /// and, under an orthographic eye, level. The way back from a camera flown
    /// somewhere there is nothing to see.
    pub const FRAME: &str = "editor_frame";
    /// Cycle the gizmo: move → scale → turn (`crate::Tool`).
    pub const TOOL: &str = "editor_tool";
    /// Delete the character before the caret in the focused text field.
    ///
    /// A *verb* and not a character, which is the split the text channel rests
    /// on (§6 M16): printable input is text and has no meaning as an action,
    /// while editing keys are actions and therefore already in the recorded
    /// [`InputFrame`](gg_input::InputFrame) — so the replay carries them with no
    /// new machinery at all.
    pub const TEXT_BACK: &str = "editor_text_back";
    /// Submit the focused text field. See [`TEXT_BACK`] for why it is a verb.
    pub const TEXT_SEND: &str = "editor_text_send";
}

/// The default binding for each verb this host appends. A build that declared
/// its own keeps whatever its bindings file says — the editor is a second
/// consumer of one mouse, not the owner of it.
const DEFAULTS: &[(&str, &str, bool)] = &[
    (ui::CLICK, "Mouse1", true),
    (ui::FOCUS, "Tab", true),
    // `PointerX`, not `MouseX`: the editor wants the arrow the operator can
    // see, and a camera wants raw device deltas. They were one source through
    // M15, so every editor click also aimed the game (§6 M15.1). Note what the
    // split does *not* fix: raw deltas arrive whatever the pointer is over, so
    // a pointer crossing the viewport still swung the camera until the host
    // stopped feeding the game at all while the editor holds the mouse.
    (ui::X, "PointerX", false),
    (ui::Y, "PointerY", false),
    // Actions and not axes, which is `gg_input::Wheel`'s decision rather than
    // this table's: a notch is discrete, and a wheel that took two axis slots
    // would be spending the headroom the look pair below now sits in.
    (ui::SCROLL_UP, "WheelUp", true),
    (ui::SCROLL_DOWN, "WheelDown", true),
    // The camera (§6 M15.2 item 4). Appended **after** the six above so a build
    // predating them resolves `ui_*` to the same ids it always did — appending
    // renumbers nothing, and that is the only reason the order here matters.
    //
    // A demo binding `W` to its own `move_forward` keeps it: one key can feed
    // two verbs, and the collision is harmless because these only fly while the
    // scene is stopped, which is exactly when the game's systems are not
    // running to read theirs.
    (verb::FORWARD, "W", true),
    (verb::BACK, "S", true),
    (verb::LEFT, "A", true),
    (verb::RIGHT, "D", true),
    (verb::UP, "E", true),
    (verb::DOWN, "Q", true),
    (verb::LOOK, "Mouse2", true),
    // Demo 05 binds `MouseX` to its own `aim_x`, and keeps it: the delta reaches
    // both verbs, harmlessly and for the same reason `W` above is shared.
    (verb::LOOK_X, "MouseX", false),
    (verb::LOOK_Y, "MouseY", false),
    // Last, for the same reason the camera verbs were appended after the `ui_*`
    // six: a build predating these resolves every earlier id to what it always
    // did. `Backspace` and `Enter` are bound unconditionally because no demo
    // binds either — and if one ever does it keeps its own, like `W` above.
    (verb::TEXT_BACK, "Backspace", true),
    (verb::TEXT_SEND, "Enter", true),
    // The navigation a *flat* scene needs (§6 M20 item 10). Appended after
    // everything above for the reason that rule keeps earning: `check_verbs`
    // (§4.7) forgives an append and nothing else, so a session recorded before
    // these three still replays, with their bits zero and their behaviour
    // therefore absent. Reordering this table would invalidate every checked-in
    // editor recording at once.
    (verb::PAN, "Mouse3", true),
    (verb::FRAME, "F", true),
    (verb::TOOL, "G", true),
];

/// The verb lists a shell should bind against with the editor open, and the
/// bindings text to append to the game's own.
///
/// Twenty now — M15.1's four, the wheel's two, §6 M15.2's seven camera
/// verbs plus the look pair a widened `MAX_AXES` made room for, §6 M16
/// item 5's text-editing pair, and §6 M20 item 10's pan, frame and tool; a
/// game that declares some of them keeps its own, as it always did.
///
/// The lists are leaked because [`Verbs`] is `&'static` by construction — the
/// dylib's own arrays are, and there must be one type for both. It is a few
/// pointers per session and per reload, in a shell that already leaks every
/// dylib it retires (§4.2.2).
#[must_use]
pub fn open(verbs: &Verbs) -> (Verbs, String) {
    let (mut actions, mut axes) = (verbs.actions.to_vec(), verbs.axes.to_vec());
    let (mut on_actions, mut on_axes) = (String::new(), String::new());
    for (name, source, is_action) in DEFAULTS {
        let (list, text) = match is_action {
            true => (&mut actions, &mut on_actions),
            false => (&mut axes, &mut on_axes),
        };
        if list.contains(name) {
            continue;
        }
        list.push(name);
        text.push_str(&format!("{name} = [\"{source}\"]\n"));
    }
    // The same context the shell pushes for the game (`bind`'s `CONTEXT`), so
    // there is one active context and not a stack whose order would matter.
    let mut bindings = String::new();
    for (section, body) in [("actions", &on_actions), ("axes", &on_axes)] {
        if !body.is_empty() {
            bindings.push_str(&format!("\n[game.{section}]\n{body}"));
        }
    }
    (
        Verbs {
            actions: actions.leak(),
            axes: axes.leak(),
        },
        bindings,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_game_with_no_ui_verbs_gains_all_of_them_without_moving_its_own() {
        let game = Verbs {
            actions: &["freeze"],
            axes: &["move_right", "aim_x"],
        };
        let (verbs, bindings) = open(&game);
        assert_eq!(
            verbs.actions,
            &[
                "freeze",
                ui::CLICK,
                ui::FOCUS,
                ui::SCROLL_UP,
                ui::SCROLL_DOWN,
                verb::FORWARD,
                verb::BACK,
                verb::LEFT,
                verb::RIGHT,
                verb::UP,
                verb::DOWN,
                verb::LOOK,
                verb::TEXT_BACK,
                verb::TEXT_SEND,
                verb::PAN,
                verb::FRAME,
                verb::TOOL,
            ]
        );
        assert_eq!(
            verbs.axes,
            &[
                "move_right",
                "aim_x",
                ui::X,
                ui::Y,
                verb::LOOK_X,
                verb::LOOK_Y
            ]
        );
        // Appended, so every id the game already had still means what it did —
        // which is what keeps a plain replay valid (§4.7).
        assert_eq!(verbs.actions[0], game.actions[0]);
        assert!(bindings.contains("ui_click = [\"Mouse1\"]"));
        assert!(bindings.contains("ui_scroll_up = [\"WheelUp\"]"));
        // The two motion sources, on the two verbs that want different things
        // out of one mouse: the panels point with the cursor, the camera turns
        // by the device (§6 M15.1, §6 M15.2). Asserted as a pair, because the
        // failure worth catching is them collapsing back onto one source.
        assert!(bindings.contains("ui_y = [\"PointerY\"]"));
        assert!(bindings.contains("editor_look_y = [\"MouseY\"]"));
        assert!(bindings.contains("editor_forward = [\"W\"]"));
        // The pan shares the look's axis pair rather than asking for a third:
        // one gesture, two buttons (§6 M20 item 10).
        assert!(bindings.contains("editor_pan = [\"Mouse3\"]"));
        assert_eq!(verbs.axes.len(), game.axes.len() + 4);
        assert!(
            gg_ui::boundary::binding(&verbs).is_some(),
            "all four resolve"
        );
    }

    /// The appended text is a map a shell can actually parse, and no two of the
    /// editor's own verbs share a source. Parsing *is* the first assertion: an
    /// unspellable key is a `MapError` there and is reported nowhere else.
    #[test]
    fn every_appended_verb_parses_and_no_two_share_a_source() {
        let game = Verbs {
            actions: &[],
            axes: &[],
        };
        let (verbs, bindings) = open(&game);
        assert!(
            gg_input::ActionMap::parse(&bindings, verbs.actions, verbs.axes).is_ok(),
            "the editor appended a binding no map can read:\n{bindings}"
        );
        let mut seen: Vec<&str> = Vec::new();
        for (name, source, _) in DEFAULTS {
            assert!(
                !seen.contains(source),
                "`{source}` is bound twice, at {name}"
            );
            seen.push(source);
        }
    }

    /// A game that already declares them keeps its own bindings: the editor
    /// appends nothing and binds nothing, so a HUD's click is still the HUD's.
    #[test]
    fn a_game_that_declares_them_is_left_alone() {
        let game = Verbs {
            actions: &[
                ui::CLICK,
                ui::FOCUS,
                ui::SCROLL_UP,
                ui::SCROLL_DOWN,
                verb::FORWARD,
                verb::BACK,
                verb::LEFT,
                verb::RIGHT,
                verb::UP,
                verb::DOWN,
                verb::LOOK,
                verb::TEXT_BACK,
                verb::TEXT_SEND,
                verb::PAN,
                verb::FRAME,
                verb::TOOL,
            ],
            axes: &[ui::X, ui::Y, verb::LOOK_X, verb::LOOK_Y],
        };
        let (verbs, bindings) = open(&game);
        assert_eq!(verbs.actions, game.actions);
        assert_eq!(verbs.axes, game.axes);
        assert!(bindings.is_empty());
    }

    /// Half-declared is the interesting case: two appended, two kept, and the
    /// bindings text covers exactly the appended pair.
    #[test]
    fn a_partly_declared_build_gains_only_what_it_lacks() {
        let game = Verbs {
            actions: &[ui::CLICK, verb::FORWARD],
            axes: &["aim_x"],
        };
        let (verbs, bindings) = open(&game);
        assert_eq!(
            &verbs.actions[..4],
            &[ui::CLICK, verb::FORWARD, ui::FOCUS, ui::SCROLL_UP]
        );
        assert_eq!(
            verbs.axes,
            &["aim_x", ui::X, ui::Y, verb::LOOK_X, verb::LOOK_Y]
        );
        assert!(!bindings.contains("ui_click"), "already the game's");
        assert!(!bindings.contains("editor_forward"), "and so is that");
        assert!(bindings.contains("ui_focus"));
        assert!(bindings.contains("ui_x") && bindings.contains("ui_y"));
        assert!(bindings.contains("editor_back"));
    }
}
