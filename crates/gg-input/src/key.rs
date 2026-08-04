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

/// A wheel notch, by direction.
///
/// **An edge and not an axis**, which is a decision worth its sentence. The
/// obvious shape is a third motion pair beside [`MouseAxis`], and it does not
/// fit: `MAX_AXES` is 8, the replay format writes that number into every file's
/// header, and a game declaring six axes of its own would have no slot left for
/// the editor's — so an editor opened over it would lose its pointer entirely
/// rather than lose its wheel. A notch is discrete anyway, and X11 has numbered
/// the two of them among its buttons since before this engine existed.
///
/// What it costs is magnitude: a tick either notched or it did not, so a flick
/// of five notches inside one tick scrolls once. At 60 Hz that ceiling is above
/// what a hand does. P2: a pixel-precise trackpad wants the axis, and wants
/// `MAX_AXES` raised first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Wheel {
    /// Away from the operator — content moves *up* the screen by convention.
    Up,
    /// Toward the operator.
    Down,
}

impl Wheel {
    /// Both, in a fixed order.
    pub const ALL: [Wheel; 2] = [Wheel::Up, Wheel::Down];

    /// Parse the config spelling.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "WheelUp" => Some(Wheel::Up),
            "WheelDown" => Some(Wheel::Down),
            _ => None,
        }
    }

    /// The config spelling.
    pub fn name(self) -> &'static str {
        match self {
            Wheel::Up => "WheelUp",
            Wheel::Down => "WheelDown",
        }
    }
}

/// A pointer axis: continuous, so it is kept apart from buttons and quantized
/// on the way into the sim ([`crate::AXIS_SCALE`]).
///
/// **Four, not two, because one mouse does two jobs.** [`X`](Self::X) and
/// [`Y`](Self::Y) are raw *device* deltas — what a fly camera wants, what keeps
/// arriving when the cursor is against the edge of the screen, and what only a
/// host holding the pointer can deliver sensibly. [`PointerX`](Self::PointerX)
/// and [`PointerY`](Self::PointerY) are *cursor* motion: where the operator's
/// visible arrow went, in the surface's own units. A UI binds the second pair
/// and a camera the first, and a build that binds one verb to each gets both
/// behaviours from one mouse (§6 M15.1).
///
/// They were one pair through M15 and the fusion was the bug: an editor's
/// `ui_x` and a game's `aim_x` were then literally the same number, so a
/// pointer crossing the viewport swung the camera and the cursor had to be
/// captured before anything could be clicked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MouseAxis {
    /// Raw device motion, horizontal, right positive.
    X,
    /// Raw device motion, vertical, down positive.
    Y,
    /// Cursor motion, horizontal, right positive.
    PointerX,
    /// Cursor motion, vertical, down positive.
    PointerY,
}

impl MouseAxis {
    /// Every axis, in the order [`crate::Input`] accumulates them.
    pub const ALL: [MouseAxis; 4] = [
        MouseAxis::X,
        MouseAxis::Y,
        MouseAxis::PointerX,
        MouseAxis::PointerY,
    ];

    /// Parse the config spelling.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "MouseX" => Some(MouseAxis::X),
            "MouseY" => Some(MouseAxis::Y),
            "PointerX" => Some(MouseAxis::PointerX),
            "PointerY" => Some(MouseAxis::PointerY),
            _ => None,
        }
    }

    /// The config spelling, for an error message that has to name it back.
    pub fn name(self) -> &'static str {
        match self {
            MouseAxis::X => "MouseX",
            MouseAxis::Y => "MouseY",
            MouseAxis::PointerX => "PointerX",
            MouseAxis::PointerY => "PointerY",
        }
    }

    /// Whether this axis is cursor motion rather than raw device motion — the
    /// question a host asks to decide which accumulator an event feeds.
    pub fn is_cursor(self) -> bool {
        matches!(self, MouseAxis::PointerX | MouseAxis::PointerY)
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

    /// The two pairs are distinct sources and spell differently: a config that
    /// says `MouseX` must not silently get cursor motion, which is the whole of
    /// what §6 M15.1 separated.
    #[test]
    fn every_pointer_axis_round_trips_and_knows_which_pair_it_is_in() {
        for axis in MouseAxis::ALL {
            assert_eq!(MouseAxis::from_name(axis.name()), Some(axis), "{axis:?}");
        }
        assert_eq!(MouseAxis::from_name("Pointer"), None);
        assert_eq!(MouseAxis::from_name("mousex"), None, "case-sensitive");
        assert!(!MouseAxis::X.is_cursor() && !MouseAxis::Y.is_cursor());
        assert!(MouseAxis::PointerX.is_cursor() && MouseAxis::PointerY.is_cursor());
    }

    #[test]
    fn every_wheel_direction_round_trips_through_its_config_spelling() {
        for wheel in Wheel::ALL {
            assert_eq!(Wheel::from_name(wheel.name()), Some(wheel), "{wheel:?}");
        }
        assert_eq!(Wheel::from_name("Wheel"), None);
        assert_eq!(Wheel::from_name("wheelup"), None, "case-sensitive");
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
