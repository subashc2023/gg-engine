//! What this crate lets a session change without a rebuild (§4.8).
//!
//! Declared here rather than in the shell because a knob belongs to whatever
//! reads it: the shell's whole share of the CVar system is deciding that config
//! is applied at all. The registry lives in `gg-core` precisely so this arrow
//! points downhill (§3, §4.8).

use gg_core::cvar::{self, CVar, CVarError};

/// Vertical field of view, in radians rather than the degrees a human would
/// type: [`crate::View`] is radians everywhere else, and converting here would
/// be a second place for fov to be wrong.
pub static FOV: CVar = CVar::new_float("r.fov", 1.0, "vertical field of view, radians");

/// With reverse-Z and an infinite far plane this is the *only* depth-precision
/// knob there is (§2, Math row) — which is what makes it worth turning without
/// a rebuild.
pub static NEAR: CVar = CVar::new_float("r.near", 0.05, "near plane distance");

/// Make them settable by name. Reads work without this — a read is a load off
/// the `static`, never a lookup — so what registration buys is config, the
/// command line and the console.
pub fn register() -> Result<(), CVarError> {
    cvar::register_all(&[&FOV, &NEAR])
}
