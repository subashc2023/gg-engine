//! Physical input identity: the stable names a rebindable action map binds
//! against (§4.7).
//!
//! Keys are by **position** on a US-QWERTY reference layout, never by the
//! character they produce — a binding that means "the key left of S" must not
//! move when the user switches to AZERTY. The variant name is also the config
//! spelling, generated from one list so a new key cannot arrive with a name the
//! parser has never heard of.

/// One list, three products: the enum, the config spelling, and the parse.
macro_rules! keys {
    ($($variant:ident),* $(,)?) => {
        /// A key by physical position on a US-QWERTY reference layout.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[non_exhaustive]
        #[expect(missing_docs, reason = "one variant per key; the names are the docs")]
        pub enum Key {
            $($variant,)*
        }

        impl Key {
            /// Every key, in declaration order.
            pub const ALL: &'static [Key] = &[$(Key::$variant,)*];

            /// How many keys exist. The held-key bitset is sized off this.
            pub const COUNT: usize = Key::ALL.len();

            /// Position in [`Key::ALL`] — a dense index for bitsets, not a
            /// stable wire value: the declaration order above is source, and a
            /// key inserted in the middle moves every index after it.
            pub const fn index(self) -> usize {
                self as usize
            }

            /// The config spelling — identical to the variant name.
            pub const fn name(self) -> &'static str {
                match self {
                    $(Key::$variant => stringify!($variant),)*
                }
            }

            /// Parse a config spelling. Case-sensitive: `"w"` is not `"W"`,
            /// because a map that silently accepts either teaches nothing when
            /// a typo lands on a key that does exist.
            pub fn from_name(name: &str) -> Option<Self> {
                match name {
                    $(stringify!($variant) => Some(Key::$variant),)*
                    _ => None,
                }
            }
        }
    };
}

keys![
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Digit0,
    Digit1,
    Digit2,
    Digit3,
    Digit4,
    Digit5,
    Digit6,
    Digit7,
    Digit8,
    Digit9,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Left,
    Right,
    Up,
    Down,
    Space,
    Enter,
    Escape,
    Tab,
    Backspace,
    Delete,
    Minus,
    Equal,
    BracketLeft,
    BracketRight,
    Backslash,
    Semicolon,
    Quote,
    Comma,
    Period,
    Slash,
    Backquote,
    ShiftLeft,
    ShiftRight,
    ControlLeft,
    ControlRight,
    AltLeft,
    AltRight,
    SuperLeft,
    SuperRight,
    Home,
    End,
    PageUp,
    PageDown,
    Insert,
];

/// A mouse button, by number rather than by name: "back"/"forward" are OS
/// conventions over the same two physical buttons, and a binding wants the
/// button.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MouseButton {
    /// Primary — left on a right-handed mouse.
    Left,
    /// Secondary.
    Right,
    /// The wheel, pressed.
    Middle,
    /// Thumb buttons, in the order the OS reports them.
    Extra(u8),
}

impl MouseButton {
    /// The config spelling: `Mouse1`…`Mouse3` for the named three, `MouseN`
    /// beyond them.
    pub fn number(self) -> u8 {
        match self {
            MouseButton::Left => 1,
            MouseButton::Right => 2,
            MouseButton::Middle => 3,
            // Extra(0) is the fourth button; the OS numbers them from zero.
            MouseButton::Extra(n) => n.saturating_add(4),
        }
    }

    /// The button with that number, inverse of [`MouseButton::number`].
    pub const fn from_number(n: u8) -> Option<Self> {
        Some(match n {
            1 => MouseButton::Left,
            2 => MouseButton::Right,
            3 => MouseButton::Middle,
            0 => return None,
            n => MouseButton::Extra(n - 4),
        })
    }

    /// Parse a `MouseN` config spelling.
    pub fn from_name(name: &str) -> Option<Self> {
        let digits = name.strip_prefix("Mouse")?;
        Self::from_number(digits.parse::<u8>().ok()?)
    }
}

/// A pointer axis. The two the mouse produces, kept separate from buttons
/// because they are continuous and get quantized on the way into the sim
/// ([`crate::AXIS_SCALE`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MouseAxis {
    /// Horizontal motion, right positive.
    X,
    /// Vertical motion, down positive.
    Y,
}

impl MouseAxis {
    /// Parse the `MouseX` / `MouseY` config spelling.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "MouseX" => Some(MouseAxis::X),
            "MouseY" => Some(MouseAxis::Y),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn every_key_round_trips_through_its_config_spelling() {
        for &k in Key::ALL {
            assert_eq!(Key::from_name(k.name()), Some(k), "{}", k.name());
        }
        assert_eq!(Key::from_name("w"), None, "spellings are case-sensitive");
        assert_eq!(Key::from_name("KeyW"), None, "winit's spelling is not ours");
    }

    #[test]
    fn mouse_buttons_number_from_one_and_parse_back() {
        for (name, button) in [
            ("Mouse1", MouseButton::Left),
            ("Mouse2", MouseButton::Right),
            ("Mouse3", MouseButton::Middle),
            ("Mouse4", MouseButton::Extra(0)),
            ("Mouse5", MouseButton::Extra(1)),
        ] {
            assert_eq!(MouseButton::from_name(name), Some(button));
            assert_eq!(button.number(), name[5..].parse::<u8>().unwrap_or(0));
        }
        assert_eq!(MouseButton::from_name("Mouse0"), None);
        assert_eq!(MouseButton::from_name("Mouse"), None);
    }
}
