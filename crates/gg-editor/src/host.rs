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
use gg_ui::boundary::verb;

/// The default binding for each of the four, used only for a verb this host had
/// to append. A build that declared its own keeps whatever its bindings file
/// says — the editor is a second consumer of one mouse, not the owner of it.
const DEFAULTS: &[(&str, &str, bool)] = &[
    (verb::CLICK, "Mouse1", true),
    (verb::FOCUS, "Tab", true),
    // `PointerX`, not `MouseX`: the editor wants the arrow the operator can
    // see, and a camera wants raw device deltas. They were one source through
    // M15, so every editor click also aimed the game (§6 M15.1). Note what the
    // split does *not* fix: raw deltas arrive whatever the pointer is over, so
    // a pointer crossing the viewport still swung the camera until the host
    // stopped feeding the game at all while the editor holds the mouse.
    (verb::X, "PointerX", false),
    (verb::Y, "PointerY", false),
    // Actions and not axes, which is `gg_input::Wheel`'s decision rather than
    // this table's: `MAX_AXES` is 8, and a game declaring six axes of its own
    // would otherwise have no slot left for the editor's pointer — losing the
    // cursor because the wheel wanted a seat.
    (verb::SCROLL_UP, "WheelUp", true),
    (verb::SCROLL_DOWN, "WheelDown", true),
];

/// The verb lists a shell should bind against with the editor open, and the
/// bindings text to append to the game's own.
///
/// Six now rather than M15.1's four; a game that declares some of them keeps
/// its own, as it always did.
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
    fn a_game_with_no_ui_verbs_gains_all_four_without_moving_its_own() {
        let game = Verbs {
            actions: &["freeze"],
            axes: &["move_right", "aim_x"],
        };
        let (verbs, bindings) = open(&game);
        assert_eq!(
            verbs.actions,
            &[
                "freeze",
                verb::CLICK,
                verb::FOCUS,
                verb::SCROLL_UP,
                verb::SCROLL_DOWN
            ]
        );
        assert_eq!(verbs.axes, &["move_right", "aim_x", verb::X, verb::Y]);
        // Appended, so every id the game already had still means what it did —
        // which is what keeps a plain replay valid (§4.7).
        assert_eq!(verbs.actions[0], game.actions[0]);
        assert!(bindings.contains("ui_click = [\"Mouse1\"]"));
        assert!(bindings.contains("ui_scroll_up = [\"WheelUp\"]"));
        // Cursor motion, not look motion — the two sources exist so this line
        // and a camera's `aim_y` are different numbers (§6 M15.1).
        assert!(bindings.contains("ui_y = [\"PointerY\"]"));
        assert!(!bindings.contains("MouseY"));
        assert!(
            gg_ui::boundary::binding(&verbs).is_some(),
            "all four resolve"
        );
    }

    /// A game that already declares them keeps its own bindings: the editor
    /// appends nothing and binds nothing, so a HUD's click is still the HUD's.
    #[test]
    fn a_game_that_declares_them_is_left_alone() {
        let game = Verbs {
            actions: &[verb::CLICK, verb::FOCUS, verb::SCROLL_UP, verb::SCROLL_DOWN],
            axes: &[verb::X, verb::Y],
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
            actions: &[verb::CLICK],
            axes: &["aim_x"],
        };
        let (verbs, bindings) = open(&game);
        assert_eq!(
            verbs.actions,
            &[verb::CLICK, verb::FOCUS, verb::SCROLL_UP, verb::SCROLL_DOWN]
        );
        assert_eq!(verbs.axes, &["aim_x", verb::X, verb::Y]);
        assert!(!bindings.contains("ui_click"), "already the game's");
        assert!(bindings.contains("ui_focus"));
        assert!(bindings.contains("ui_x") && bindings.contains("ui_y"));
    }
}
