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

/// Bytes of pack content one frame may copy to the device (§4.6).
///
/// A knob rather than a constant because the right value is the machine's, not
/// the engine's: too low and a level takes seconds to finish arriving, too high
/// and the frame that copies is the frame that hitches. 16 MiB is roughly two
/// milliseconds of PCIe and thirty frames' worth of a large level.
pub static UPLOAD_BUDGET: CVar = CVar::new_int(
    "r.upload_budget",
    16 << 20,
    "bytes of pack content uploaded per frame",
);

/// Exposure in **stops**, applied before the tonemap curve (§6 M11). Stops
/// rather than a multiplier because that is the unit a human turning a knob
/// thinks in: `+1` is twice the light, and `0` leaves the scene as authored.
pub static EXPOSURE: CVar = CVar::new_float("r.exposure", 0.0, "exposure, in stops");

/// A flat ambient term, linear. Standing in for indirect light, which is an
/// irradiance probe or a lightmap and is P1 — what this buys today is that a
/// face pointing away from every light is dim rather than pure black.
pub static AMBIENT: CVar = CVar::new_float("r.ambient", 0.03, "flat ambient light, linear");

/// Whether the frame resamples its radiance for the overlay's luminance
/// histogram (§6 M11's exit row, §4.8).
///
/// Off by default and a knob rather than a compile-time thing: it costs a small
/// fullscreen pass and a 36 KiB readback every frame, which is nothing next to
/// the frame and everything next to a row nobody is looking at.
pub static HISTOGRAM: CVar = CVar::new_bool(
    "r.histogram",
    false,
    "resample the frame's radiance for the overlay histogram",
);

/// Shadow map edge in texels, clamped to `[256, 4096]`. A quality knob and a
/// memory one at once: 2048² of `D32_SFLOAT` is 16 MiB.
pub static SHADOW_SIZE: CVar = CVar::new_int("r.shadow_size", 2048, "shadow map edge, texels");

/// Metres the single cascade covers around the camera. The whole quality/extent
/// trade of one cascade lives here: half the radius is twice the texel density
/// and half the range beyond which nothing is shadowed at all.
pub static SHADOW_RADIUS: CVar =
    CVar::new_float("r.shadow_radius", 40.0, "shadow cascade radius, metres");

/// Constant depth bias for the shadow pass, in units of the smallest resolvable
/// depth difference (§6 M11's exit row).
///
/// Too little is **acne** — a surface shadowing itself in stripes. Too much is
/// **peter-panning** — a shadow detached from the foot of the thing casting it.
/// The two failures pull in opposite directions, which is why this is a knob and
/// not a constant somebody picked once on one scene.
///
/// **Negative, because the engine is reverse-Z** (§2, Math row). In an ordinary
/// depth buffer a positive bias pushes a caster *away* from the light; here
/// greater depth means *nearer*, so a positive bias pulls every caster toward
/// the light and shadows the whole scene. That failure looks like a room lit
/// only by the ambient term rather than like acne, which is why it is worth a
/// paragraph: it does not read as a bias problem at all.
pub static SHADOW_BIAS: CVar = CVar::new_float("r.shadow_bias", -2.0, "shadow constant depth bias");

/// Depth bias proportional to the fragment's depth slope. The term that covers a
/// surface seen edge-on by the light, where a constant bias has almost no depth
/// gradient to act through.
pub static SHADOW_SLOPE_BIAS: CVar = CVar::new_float(
    "r.shadow_slope_bias",
    -3.0,
    "shadow slope-scaled depth bias",
);

/// Metres the shadow *lookup* is pushed along the surface normal. The half of
/// the acne fix a depth bias cannot do — it works at any angle, and overdoing it
/// produces the same peter-panning the constant bias does.
pub static SHADOW_NORMAL_BIAS: CVar = CVar::new_float(
    "r.shadow_normal_bias",
    0.03,
    "shadow lookup offset along the normal, metres",
);

/// Make them settable by name. Reads work without this — a read is a load off
/// the `static`, never a lookup — so what registration buys is config, the
/// command line and the console.
pub fn register() -> Result<(), CVarError> {
    cvar::register_all(&[
        &FOV,
        &NEAR,
        &UPLOAD_BUDGET,
        &EXPOSURE,
        &AMBIENT,
        &HISTOGRAM,
        &SHADOW_SIZE,
        &SHADOW_RADIUS,
        &SHADOW_BIAS,
        &SHADOW_SLOPE_BIAS,
        &SHADOW_NORMAL_BIAS,
    ])
}
