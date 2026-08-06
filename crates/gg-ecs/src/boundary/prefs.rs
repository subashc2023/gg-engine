//! What the player asked the host for — the fourth protocol on the §4.2.2
//! boundary, arriving at §6 M19. Same shape as [`Sound`](super::Sound): a game
//! crate may not link the crates that own the window or the speakers (§3's
//! deny pin), so a preference crosses as an ordinary component the game
//! declares and a menu it draws writes. The host reads it every tick and
//! **writes nothing back** — CVars, not this, are where a machine's own
//! settings (vsync, shadow quality, the overlay) live (§4.8) — so a silent
//! headless replay and a loud windowed run hash alike (§5.6c).
//!
//! `World::restore` zeroes a retyped field (§4.2.2), so every field's zero
//! must mean "what the engine does anyway": [`cursor::SOFTWARE`] is 0, and a
//! [`quiet`](Prefs::quiet) of 0 is full volume. A migration cannot flip a
//! player's cursor or mute their game.

use crate::Component;

/// Which arrow the player sees over a pointed UI. Constants rather than an
/// `enum`, for the reason [`wave`](super::wave) gives: a discriminant crossing
/// the boundary is a `u32` this compiler did not write, and an unknown one
/// falls back to the default rather than to undefined behaviour.
pub mod cursor {
    /// The host hides the OS arrow over the window and `gg-ui` draws the
    /// pointer at the steered position (§4.9). Zero on purpose — the default,
    /// and what an unknown value falls back to.
    pub const SOFTWARE: u32 = 0;
    /// The OS arrow stands in; `gg-ui` draws no second one. The routing is
    /// identical either way — only the picture changes.
    pub const HARDWARE: u32 = 1;
}

/// Full attenuation, the [`Prefs::quiet`] that means silence. Fixed-point like
/// an axis (§4.7): a menu steps an integer, and a float volume would put a
/// rounding rule in hashed state.
pub const QUIET_MAX: u32 = 1024;

/// The player's preferences, read by the host every tick.
///
/// Declare one (usually on the same entity as the rest of a game's globals) and
/// write it from the settings menu; a world without one gets every default.
/// More than one is not an error, but the host reads the first it walks, so one
/// is the number to declare.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "gg.prefs")]
#[repr(C)]
pub struct Prefs {
    /// One of [`cursor`]'s constants. Unknown values are [`cursor::SOFTWARE`].
    pub cursor: u32,
    /// Master attenuation, `0..=`[`QUIET_MAX`]: 0 is full volume, [`QUIET_MAX`]
    /// is silence, linear between. Attenuation and not volume so that zero —
    /// what a migration writes — is loud, not mute. Applied by the host over
    /// every [`Sound::gain`](super::Sound::gain); values past the max clamp.
    pub quiet: u32,
}

impl Prefs {
    /// The master gain this asks for, `0.0..=1.0` — what `gg-audio` reads to
    /// mix every playing [`Sound`](super::Sound).
    #[must_use]
    pub fn volume(&self) -> f32 {
        1.0 - self.quiet.min(QUIET_MAX) as f32 / QUIET_MAX as f32
    }

    /// Whether the OS arrow should stand in for the software cursor.
    #[must_use]
    pub fn hardware_cursor(&self) -> bool {
        self.cursor == cursor::HARDWARE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_protocol_type_is_flat_and_padding_free() {
        // `Pod` already refuses padding; this pins the number so a field added
        // is a visible edit rather than a silent layout move.
        assert_eq!(size_of::<Prefs>(), 8);
        assert_eq!(align_of::<Prefs>(), 4);
    }

    /// The property `World::restore` needs: zeroed is the default, in both
    /// fields, so a migration cannot flip a cursor or mute a game.
    #[test]
    fn a_zeroed_prefs_is_the_default_twice_over() {
        let zeroed: Prefs = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.cursor, cursor::SOFTWARE);
        assert!(!zeroed.hardware_cursor());
        assert_eq!(zeroed.volume(), 1.0);
    }

    /// An unknown cursor constant is the default, not a third behaviour — the
    /// game may be built against a newer boundary than this host (§4.2.2).
    #[test]
    fn an_unknown_cursor_falls_back_to_software() {
        let prefs = Prefs {
            cursor: 9999,
            quiet: 0,
        };
        assert!(!prefs.hardware_cursor());
    }

    #[test]
    fn quiet_attenuates_linearly_and_clamps_at_silence() {
        let at = |quiet| Prefs { cursor: 0, quiet }.volume();
        assert_eq!(at(0), 1.0);
        assert_eq!(at(QUIET_MAX / 2), 0.5);
        assert_eq!(at(QUIET_MAX), 0.0);
        assert_eq!(at(u32::MAX), 0.0, "past the max clamps rather than wraps");
    }
}
