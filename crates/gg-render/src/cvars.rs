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

/// The orthographic path's far plane (§6 M20). Perspective keeps its far at
/// infinity (§2) and never reads this; an orthographic projection *must* pick a
/// finite one, and with ortho depth linear in `D32_SFLOAT` a generous value
/// costs nothing — 500 m of slab still resolves to sub-millimetres. A host
/// knob rather than a game field because it is a clipping bound, not framing:
/// the game's [`gg_ecs::boundary::Eye::ortho`] says how much world is *seen*,
/// this says how deep the seen slab reaches.
pub static ORTHO_FAR: CVar =
    CVar::new_float("r.ortho_far", 500.0, "orthographic far plane, metres");

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

/// Antialias the finished picture with one post-process edge pass (§6 M21).
///
/// **Off by default, and that default is load-bearing**: every blessed golden
/// (§4.10) was rendered without it, and an AA pass moves pixels along every
/// silhouette in the frame by construction — turning it on by default would mean
/// re-blessing three backends to change nothing anyone asked to change. A player
/// turns it on through a game's settings menu (`Prefs::aa`), which is a *game's*
/// world state and reaches no gate.
pub static AA: CVar = CVar::new_bool("r.aa", false, "antialias the finished picture (FXAA)");

/// Samples per pixel in the scene pass — MSAA (§6 M21). `1` is off, and off for
/// the same load-bearing reason [`AA`] is: it moves every silhouette.
///
/// Clamped to a power of two in `[1, 8]` and then down to what the device
/// advertises, which is the *only* place a count is silently reduced — a device
/// that cannot do 8× has to draw something, and the alternative is a black
/// window. The reduction is logged once, and [`Renderer::samples`] is what the
/// operator reads back to see which count they actually got.
///
/// [`Renderer::samples`]: crate::Renderer::samples
pub static MSAA: CVar = CVar::new_int("r.msaa", 1, "scene-pass samples per pixel (1, 2, 4, 8)");

/// Shadow map edge in texels, clamped to `[256, 4096]`. A quality knob and a
/// memory one at once: 2048² of `D32_SFLOAT` is 16 MiB.
pub static SHADOW_SIZE: CVar = CVar::new_int("r.shadow_size", 2048, "shadow map edge, texels");

/// Metres from the eye out to which anything is shadowed at all — the *whole*
/// range, which the cascades then divide between them.
///
/// This replaced `r.shadow_radius`, and the rename is the point: a radius was a
/// single slab's half-width, and with cascades no one slab has authority over
/// the range. What a session turns now is how far shadows reach, and
/// [`SHADOW_CASCADES`] is what decides how much resolution that range gets.
pub static SHADOW_DISTANCE: CVar = CVar::new_float(
    "r.shadow_distance",
    80.0,
    "shadow range from the eye, metres",
);

/// How many cascades the range is split into, clamped to `[1, MAX_CASCADES]`.
///
/// Each is a full [`SHADOW_SIZE`]² map, so this multiplies both shadow memory
/// and the shadow pass's draw count — the second is why the fitter culls each
/// cascade's draw list against its own slab rather than redrawing the world four
/// times. 1 is the pre-cascade behaviour and is kept reachable because it is the
/// control a measurement wants.
pub static SHADOW_CASCADES: CVar =
    CVar::new_int("r.shadow_cascades", 4, "shadow cascades over the range");

/// Cull casters, `1` on. **A diagnostic switch, not a quality setting.**
///
/// Two culls stand between a box and the shadow map it belongs in — the swept view
/// frustum at extract (`Extracted::cast_along`) and the per-cascade slab test
/// ([`casts_into`](crate::casts_into)) — and both are a function of where the
/// camera is pointed. So "a shadow that comes and goes as I turn" has two
/// candidate causes that look alike from a chair, and this separates them in one
/// toggle: `r.shadow_cull 0` keeps every caster in every cascade, so if the
/// artifact survives it is the cascade *fit* and not either cull.
///
/// Off, a frame draws the whole world into each cascade — which is the cost the
/// culls exist to avoid, and why this is a knob a session turns and not one a
/// build ships with.
pub static SHADOW_CULL: CVar = CVar::new_int(
    "r.shadow_cull",
    1,
    "cull shadow casters (0 = every caster in every cascade — a diagnostic)",
);

/// How the range is divided, `0` uniform to `1` logarithmic.
///
/// The practical split scheme [ZSXL06]: a logarithmic division gives every
/// cascade the same *relative* texel density, which is what perspective wants,
/// but it spends almost the whole range on the first few metres. A uniform one
/// does the opposite. The blend between them is the knob, and 0.85 is the value
/// that paper and everyone since lands on.
pub static SHADOW_SPLIT_LAMBDA: CVar = CVar::new_float(
    "r.shadow_split_lambda",
    0.85,
    "cascade split blend, 0 uniform to 1 logarithmic",
);

/// Fraction of a cascade's extent over which it cross-fades into the next.
///
/// Without it the split is a visible line where texel density changes — most
/// obvious as a step in a shadow's softness, since the kernel is a fixed three
/// texels and those texels are a different size either side. Costs a second PCF
/// lookup for fragments inside the band, and nothing outside it.
pub static SHADOW_BLEND: CVar = CVar::new_float(
    "r.shadow_blend",
    0.1,
    "cascade cross-fade band, fraction of a cascade",
);

/// Normal-offset reach, in **shadow texels** — the acne knob (§6 M11's exit
/// row).
///
/// Too little is **acne** — a surface shadowing itself in stripes. Too much is
/// **peter-panning** — a shadow detached from the foot of the thing casting it.
/// The two pull in opposite directions, which is why this is a knob and not a
/// constant somebody picked once on one scene; `gg-tools shadow-bias` sweeps it
/// and prints the plateau where neither happens.
///
/// Texels rather than metres because that is the unit the error is in: the map's
/// sampling footprint is one texel wide whatever the cascade covers, so a value
/// in metres stops being right the moment `r.shadow_radius` or `r.shadow_size`
/// moves. The shader scales it by the texel's world size *and* by
/// sin(incidence), so a face-on surface — which needs none — pays none.
///
/// 1.0 clears one texel; the 3x3 kernel's corner tap sits sqrt(2) out. The
/// default is *below* that on purpose — the shader's receiver-plane term already
/// corrects each tap for the receiver's own slope, so what is left for this to
/// cover is the footprint error at a silhouette, where that term is clamped. The
/// sweep measures the plateau at 0.75 to 1.25 texels of total offset and this is
/// the low end of it, because everything above the plateau is peter-panning.
pub static SHADOW_NORMAL_BIAS: CVar = CVar::new_float(
    "r.shadow_normal_bias",
    0.5,
    "shadow normal-offset reach, in shadow texels",
);

/// Constant part of the same offset, in **shadow texels**, angle-free.
///
/// [`SHADOW_NORMAL_BIAS`] vanishes where the light meets a surface head-on,
/// which is correct — there is no footprint error there — but it leaves the one
/// error that does not vanish: the shadow pass and the forward pass compute the
/// same surface's depth through two different rasterizations, and at zero
/// incidence a coin-flip between them shadows the whole floor. This covers that.
///
/// Pushed along the normal like the other term, which at head-on incidence *is*
/// a push toward the light — so the reverse-Z sign trap the old rasterizer bias
/// carried does not exist here: more is always less shadow.
///
/// Replaces `r.shadow_bias`, which drove the rasterizer's constant term. That
/// one was worth about 19 um against a 39 mm texel (Vulkan scales it by an
/// implementation-dependent `r` for a float depth attachment) and was a
/// different distance on every driver — not something a blessed reference
/// (§4.10) may depend on. `r.shadow_slope_bias` is gone with it: unbounded
/// without the optional `depthBiasClamp` feature, and superseded by the shader's
/// receiver-plane term, which corrects per fragment and per tap.
/// 0.75 is twice the measured head-on failure boundary: `gg-tools shadow-bias`
/// puts the cliff between 0.375 and 0.5 texels, and it is a cliff — 0.375 acnes
/// over 9% of the frame and 0.5 over none of it. A default sitting on that edge
/// would be one scene away from failing, and the margin costs 0.2% of pixels in
/// peter-panning.
pub static SHADOW_DEPTH_BIAS: CVar = CVar::new_float(
    "r.shadow_depth_bias",
    0.75,
    "shadow constant normal offset, in shadow texels",
);

/// Make them settable by name. Reads work without this — a read is a load off
/// the `static`, never a lookup — so what registration buys is config, the
/// command line and the console.
pub fn register() -> Result<(), CVarError> {
    cvar::register_all(&[
        &FOV,
        &NEAR,
        &ORTHO_FAR,
        &UPLOAD_BUDGET,
        &EXPOSURE,
        &AMBIENT,
        &HISTOGRAM,
        &AA,
        &MSAA,
        &SHADOW_SIZE,
        &SHADOW_DISTANCE,
        &SHADOW_CASCADES,
        &SHADOW_CULL,
        &SHADOW_SPLIT_LAMBDA,
        &SHADOW_BLEND,
        &SHADOW_NORMAL_BIAS,
        &SHADOW_DEPTH_BIAS,
    ])
}
