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
//! player's cursor or mute their game. The two fields M21 added keep the rule —
//! [`aa::DEFAULT`] is 0 and leaves the host's own knob alone, and a
//! [`close`](Prefs::close) of 0 is a session that keeps running — which is the
//! constraint that decided both spellings: an "antialiasing on" whose zero was
//! *off* would silently un-antialias a reloaded game, and a `keep_running` flag
//! would end every session a migration touched.

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

/// Which edge treatment the player asked for. Constants for [`cursor`]'s
/// reason, and with one extra: the zero is **not** a mode, it is *"leave the
/// host's own knob alone"*, so a game that never draws a video menu — every one
/// of them before M21 — keeps whatever `r.aa` was configured to, and a
/// migration that zeroes the field hands the choice back rather than making it.
/// Each constant is a complete statement about **both** of the host's knobs —
/// the post pass's and the scene pass's — so the list really is a list of
/// modes: picking one turns the others off, and a game does not have to know
/// that two mechanisms exist to choose between them.
pub mod aa {
    /// The host's own settings (`r.aa`, `r.msaa`) decide. Zero on purpose — see
    /// the module docs.
    pub const DEFAULT: u32 = 0;
    /// No antialiasing at all, whatever the host was configured for.
    pub const OFF: u32 = 1;
    /// One post-process edge pass — cheap, resolution-independent, and the only
    /// mode that costs nothing but a fullscreen read (§6 M21). Softens edges
    /// *inside* a triangle too, which is where a flat-shaded scene's worst
    /// aliasing is, at the price of not being able to tell an authored
    /// one-pixel detail from a staircase.
    pub const FXAA: u32 = 2;
    /// Two samples per pixel. Geometric edges only, and exact on them.
    pub const MSAA_2: u32 = 3;
    /// Four samples per pixel.
    pub const MSAA_4: u32 = 4;
    /// Eight samples per pixel.
    pub const MSAA_8: u32 = 5;

    /// Samples per pixel this mode asks for, or `None` where the count is not
    /// this mode's to state. Here rather than in the host so the mapping from
    /// mode to count has one definition on both sides of the boundary.
    #[must_use]
    pub fn samples(mode: u32) -> Option<u32> {
        match mode {
            OFF | FXAA => Some(1),
            MSAA_2 => Some(2),
            MSAA_4 => Some(4),
            MSAA_8 => Some(8),
            _ => None,
        }
    }
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
///
/// `Default` is derivable *because* of the zero law above: every field's zero
/// is what the engine does anyway, so the derived all-zeros value and
/// `Zeroable::zeroed()` are the same `Prefs`, and a game spelling
/// `Prefs { quiet, ..Default::default() }` cannot accidentally assert a
/// preference it did not mean to hold. A test below pins the two together, so a
/// field whose zero stops being the default breaks here rather than in a game.
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
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
    /// One of [`aa`]'s constants. Unknown values are [`aa::DEFAULT`].
    pub aa: u32,
    /// Nonzero ends the session — the menu's quit button, and the only way a
    /// game has to close its own window.
    ///
    /// Sim state, unlike every other way out of a session (§6 M15.1: "quitting
    /// is not simulated state"), and deliberately: the close button and Escape
    /// are the *operator* stopping the process, which must work identically
    /// while a replay drives, whereas this is the *player* choosing to stop
    /// inside the game — a decision that belongs in the recording, so a replayed
    /// session ends where the session it recorded did.
    ///
    /// Monotone rather than an edge: a game sets it once and the session is
    /// over, so there is no tick on which anyone has to remember to clear it.
    pub close: u32,
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

    /// Whether the post pass should antialias, or `None` to leave `r.aa` alone.
    /// An unknown constant is `None` for [`cursor`]'s reason — the game may be
    /// built against a newer boundary than this host, and a mode this host does
    /// not know is one it must not guess at.
    #[must_use]
    pub fn antialias(&self) -> Option<bool> {
        match self.aa {
            aa::FXAA => Some(true),
            // Every other *known* mode is a statement that the post pass is
            // not the one doing it — including the MSAA ones, which is what
            // makes this a list of modes rather than a pair of flags.
            aa::OFF | aa::MSAA_2 | aa::MSAA_4 | aa::MSAA_8 => Some(false),
            _ => None,
        }
    }

    /// Samples per pixel the scene pass is asked for, or `None` to leave
    /// `r.msaa` alone. A count this device cannot do is reduced by the host and
    /// said out loud there — the game asks, it does not negotiate.
    #[must_use]
    pub fn samples(&self) -> Option<u32> {
        aa::samples(self.aa)
    }

    /// Whether the player asked to end the session.
    #[must_use]
    pub fn closing(&self) -> bool {
        self.close != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_protocol_type_is_flat_and_padding_free() {
        // `Pod` already refuses padding; this pins the number so a field added
        // is a visible edit rather than a silent layout move.
        assert_eq!(size_of::<Prefs>(), 16);
        assert_eq!(align_of::<Prefs>(), 4);
    }

    /// The property `World::restore` needs: zeroed is the default, in every
    /// field, so a migration cannot flip a cursor, mute a game, un-antialias it
    /// or close it.
    #[test]
    fn a_zeroed_prefs_is_the_default_four_times_over() {
        let zeroed: Prefs = bytemuck::Zeroable::zeroed();
        assert_eq!(zeroed.cursor, cursor::SOFTWARE);
        assert!(!zeroed.hardware_cursor());
        assert_eq!(zeroed.volume(), 1.0);
        assert_eq!(zeroed.antialias(), None, "the host's knob is left alone");
        assert_eq!(zeroed.samples(), None, "and so is the other one");
        assert!(!zeroed.closing());
        assert_eq!(zeroed, Prefs::default(), "and `..Default::default()` is it");
    }

    /// Every mode states **both** knobs, and an unknown one states neither —
    /// leaving the host's own settings alone rather than picking for it, on the
    /// same reasoning as the cursor: a newer game, an older host.
    ///
    /// The pairs are the content: an MSAA mode that left the post pass's flag
    /// alone would run whatever `r.aa` happened to be *on top of* the samples,
    /// which is two antialiasers and one of them unasked for.
    #[test]
    fn a_mode_states_both_knobs_and_an_unknown_one_states_neither() {
        let at = |aa| {
            let prefs = Prefs {
                cursor: 0,
                quiet: 0,
                aa,
                close: 0,
            };
            (prefs.antialias(), prefs.samples())
        };
        assert_eq!(at(aa::DEFAULT), (None, None));
        assert_eq!(at(aa::OFF), (Some(false), Some(1)));
        assert_eq!(at(aa::FXAA), (Some(true), Some(1)));
        assert_eq!(at(aa::MSAA_2), (Some(false), Some(2)));
        assert_eq!(at(aa::MSAA_4), (Some(false), Some(4)));
        assert_eq!(at(aa::MSAA_8), (Some(false), Some(8)));
        assert_eq!(at(9999), (None, None));
    }

    /// An unknown cursor constant is the default, not a third behaviour — the
    /// game may be built against a newer boundary than this host (§4.2.2).
    #[test]
    fn an_unknown_cursor_falls_back_to_software() {
        let prefs = Prefs {
            cursor: 9999,
            quiet: 0,
            aa: 0,
            close: 0,
        };
        assert!(!prefs.hardware_cursor());
    }

    #[test]
    fn quiet_attenuates_linearly_and_clamps_at_silence() {
        let at = |quiet| {
            Prefs {
                cursor: 0,
                quiet,
                aa: 0,
                close: 0,
            }
            .volume()
        };
        assert_eq!(at(0), 1.0);
        assert_eq!(at(QUIET_MAX / 2), 0.5);
        assert_eq!(at(QUIET_MAX), 0.0);
        assert_eq!(at(u32::MAX), 0.0, "past the max clamps rather than wraps");
    }
}
