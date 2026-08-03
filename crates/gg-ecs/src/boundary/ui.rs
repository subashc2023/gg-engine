//! What a game tells the host about its UI — the second half of the render
//! protocol, arriving at M13 (§4.9, §6 M13's demo 07).
//!
//! Same argument as [`present`](super::present), reached again from a different
//! direction: a game crate may link four engine crates (§3's deny pin) and
//! `gg-ui` is not one of them, so a game cannot build a draw list. If it could,
//! every `gg-ui` change would be inside a game dylib's blast radius, which is
//! the property the pin exists to hold. So a game *declares* its UI as ordinary
//! components and the host's `gg-ui` turns them into geometry.
//!
//! # Why the response comes back in the component
//!
//! [`Widget::state`] is written by the host and read by the game on the next
//! tick, rather than being handed over as a side channel. That puts hover,
//! focus and the click **in the world**, which puts them in the state hash —
//! so §6 M13's "a replayed click lands on the same widget every run" is proven
//! by the determinism gate that already exists (§5.6c) instead of by a new kind
//! of assertion. A click that landed on a different widget is a different hash
//! at that tick, named by tick number, on three architectures.
//!
//! # Canvas units, not pixels
//!
//! [`Widget::rect`] is in [`CANVAS`] units and the host scales that canvas to
//! whatever the window is. A replay must land on the same widget at any window
//! size, and pointer motion arrives as device deltas rather than as a position,
//! so the pointer is integrated *in canvas units* and the window size never
//! enters the arithmetic (§4.7).

use crate::Component;

/// The canvas a game authors its UI against, in logical units. The host fits it
/// to the target with a uniform scale and centres it, so a UI authored once is
/// the same UI at every window size and aspect-correct at 16:9.
pub const CANVAS: (u32, u32) = (640, 360);

/// What a [`Widget`] is. Constants rather than an `enum` field, for the reason
/// [`light`](super::light) gives: a discriminant crossing the boundary is a
/// `u32` the compiler on this side did not write, and an unknown one draws
/// nothing rather than being undefined behaviour the moment it is read.
pub mod widget {
    /// A filled rectangle. Text is unread; this is a background.
    pub const PANEL: u32 = 0;
    /// Text only, left-aligned in the rect, no fill. Not hit-tested.
    pub const LABEL: u32 = 1;
    /// A filled rectangle with its text centred, hit-tested and focusable —
    /// the only kind that ever comes back with [`super::state`] bits set.
    pub const BUTTON: u32 = 2;
}

/// Bits the host writes into [`Widget::state`]. All of them are false for a
/// kind that is not hit-tested.
pub mod state {
    /// The pointer is over it and nothing is drawn on top.
    pub const HOVERED: u32 = 1 << 0;
    /// It took the press and the button is still down — a drag.
    pub const HELD: u32 = 1 << 1;
    /// The press it took was released over it. True for exactly one tick, and
    /// never for a press dragged off and released elsewhere.
    pub const CLICKED: u32 = 1 << 2;
    /// Keys are going here.
    pub const FOCUSED: u32 = 1 << 3;
}

/// Longest label a widget carries, in bytes.
///
/// Fixed because a component is `Pod` (§4.2.1) — a `String` here would put an
/// allocation in a column the snapshot path memcpies. Sized so [`Widget`] comes
/// out at 80 bytes with no padding for `Pod` to refuse.
pub const TEXT: usize = 32;

/// One piece of UI, in canvas units.
///
/// The game fills every field but [`state`](Widget::state); the host writes
/// that one and reads the rest.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "gg.widget")]
#[repr(C)]
pub struct Widget {
    /// Identity across frames — what hover, focus and capture are remembered
    /// against, and what a game names a button by. [`widget_id`] is the
    /// conventional way to produce one; any `u64` the game keeps stable will do.
    ///
    /// Two live widgets sharing an id fight over focus, and the host reports it
    /// rather than arbitrating.
    pub id: u64,
    /// `[x, y, w, h]` in [`CANVAS`] units, origin top-left.
    pub rect: [f32; 4],
    /// `0xAARRGGBB`. Alpha is real: a panel at `0xc0` reads over the scene.
    /// For a [`widget::BUTTON`] this is the fill, and the host lightens it on
    /// hover — the one visual decision the host makes, because a game that had
    /// to author three colours per button would author one and look dead.
    pub color: u32,
    /// `0xAARRGGBB` for the text of a [`widget::LABEL`] or [`widget::BUTTON`].
    pub text_color: u32,
    /// One of [`widget`]'s constants. An unknown kind draws nothing.
    pub kind: u32,
    /// Draw order, ascending; the last one drawn is the topmost and wins a hit.
    /// Ties break on [`id`](Widget::id), so the picture does not depend on world
    /// iteration order.
    pub order: u32,
    /// Host-written: a mask of [`state`]'s bits. Whatever the game puts here is
    /// overwritten every tick the host runs the UI.
    pub state: u32,
    /// Bytes of [`text`](Widget::text) in use. Past [`TEXT`] is clamped.
    pub text_len: u32,
    /// The label, UTF-8, unterminated. Bytes past `text_len` are zero — a
    /// component is hashed whole, so leftovers would make two identical widgets
    /// hash differently.
    pub text: [u8; TEXT],
}

impl Widget {
    /// A background panel.
    #[must_use]
    pub fn panel(rect: [f32; 4], color: u32) -> Self {
        Widget {
            id: 0,
            rect,
            color,
            text_color: 0,
            kind: widget::PANEL,
            order: 0,
            state: 0,
            text_len: 0,
            text: [0; TEXT],
        }
    }

    /// Text at `rect`'s top-left. Truncated to [`TEXT`] bytes on a character
    /// boundary — a HUD row that got long is a clipped row, never a panic and
    /// never a broken code point.
    #[must_use]
    pub fn label(rect: [f32; 4], color: u32, text: &str) -> Self {
        let mut widget = Widget::panel(rect, 0);
        widget.kind = widget::LABEL;
        widget.text_color = color;
        widget.set_text(text);
        widget
    }

    /// A button, hit-tested under `id`.
    #[must_use]
    pub fn button(id: u64, rect: [f32; 4], color: u32, text_color: u32, text: &str) -> Self {
        let mut widget = Widget::panel(rect, color);
        widget.id = id;
        widget.kind = widget::BUTTON;
        widget.text_color = text_color;
        widget.set_text(text);
        widget
    }

    /// Replace the label. The common edit in a HUD, where the rectangle is
    /// fixed and the number in it is not.
    pub fn set_text(&mut self, text: &str) {
        let mut len = text.len().min(TEXT);
        while len > 0 && !text.is_char_boundary(len) {
            len -= 1;
        }
        self.text = [0; TEXT];
        self.text[..len].copy_from_slice(&text.as_bytes()[..len]);
        self.text_len = len as u32;
    }

    /// The label. Empty if the bytes are not UTF-8, which only a hand-built
    /// component can manage.
    #[must_use]
    pub fn text(&self) -> &str {
        let len = (self.text_len as usize).min(TEXT);
        core::str::from_utf8(&self.text[..len]).unwrap_or("")
    }

    /// The pointer is over it.
    #[must_use]
    pub fn hovered(&self) -> bool {
        self.state & state::HOVERED != 0
    }

    /// It was pressed and released over, this tick.
    #[must_use]
    pub fn clicked(&self) -> bool {
        self.state & state::CLICKED != 0
    }

    /// It is being dragged.
    #[must_use]
    pub fn held(&self) -> bool {
        self.state & state::HELD != 0
    }

    /// Keys are going here.
    #[must_use]
    pub fn focused(&self) -> bool {
        self.state & state::FOCUSED != 0
    }
}

/// A widget id from a path. FNV-1a, `const`-evaluable, so an id costs nothing
/// per frame: `const VSYNC: u64 = widget_id("settings.vsync");`.
///
/// Here rather than in `gg-ui` for the same reason
/// [`asset_id`](gg_abi::asset_id) is where it is: a game crate may not link the
/// crate that consumes the value (§3), so the function that produces it has to
/// live on the game's side of the pin. `gg-ui` calls this one — one
/// implementation, so the two sides cannot drift.
#[must_use]
pub const fn widget_id(path: &str) -> u64 {
    let bytes = path.as_bytes();
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut at = 0;
    while at < bytes.len() {
        hash ^= bytes[at] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        at += 1;
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_protocol_type_is_flat_and_padding_free() {
        // `Pod` already refuses padding; this pins the number so a field added
        // is a visible edit rather than a silent layout move.
        assert_eq!(size_of::<Widget>(), 80);
        assert_eq!(align_of::<Widget>(), 8);
    }

    /// A label longer than the field is cut, and cut where a character ends.
    /// The bytes past the cut are zero: a component is hashed whole, so a
    /// leftover tail would make two widgets showing the same text disagree.
    #[test]
    fn a_long_label_is_truncated_on_a_character_boundary_and_leaves_no_tail() {
        let mut widget = Widget::label([0.0; 4], 0xffff_ffff, "a much longer label than fits");
        assert_eq!(widget.text(), "a much longer label than fits");
        let long = "x".repeat(TEXT) + "and more";
        widget.set_text(&long);
        assert_eq!(widget.text().len(), TEXT);

        // A multi-byte character straddling the cut is dropped whole.
        let mut wide = Widget::label([0.0; 4], 0, "");
        wide.set_text(&("y".repeat(TEXT - 1) + "é"));
        assert_eq!(wide.text().len(), TEXT - 1, "the two-byte char did not fit");
        assert!(wide.text[TEXT - 1..].iter().all(|b| *b == 0));

        // And a shorter text after a longer one leaves nothing behind.
        wide.set_text("short");
        assert!(wide.text[5..].iter().all(|b| *b == 0));
    }

    #[test]
    fn state_bits_read_back_as_the_questions_a_game_asks() {
        let mut button = Widget::button(widget_id("ok"), [0.0; 4], 0, 0, "ok");
        assert!(!button.clicked() && !button.hovered());
        button.state = state::HOVERED | state::CLICKED;
        assert!(button.clicked() && button.hovered() && !button.held() && !button.focused());
    }

    #[test]
    fn ids_are_distinct_by_path_and_stable_across_builds() {
        assert_ne!(widget_id("a.b"), widget_id("ab"));
        assert_ne!(widget_id("settings.vsync"), widget_id("settings.vsync "));
        // Pinned: this value is in a replay's hash baseline, so a change to the
        // hash is a change to every recorded UI session.
        assert_eq!(widget_id(""), 0xcbf2_9ce4_8422_2325);
    }
}
