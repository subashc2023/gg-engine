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

/// Dither amplitude at the output quantizer, in **code values**; `0` is off.
///
/// The scene is `Rgba16F` all the way to the post pass and the swapchain is
/// 8-bit sRGB, so the last thing that happens to a frame is a quantization to
/// 256 levels a channel — and a gradient that crosses one level over more than a
/// pixel or two is a *contour*, which is what a lit floor or a distant wall is
/// made of. Nothing before this point can help: the banding is manufactured by
/// the quantizer and has to be answered there.
///
/// 1.0 is one LSB of triangular-PDF noise, which is the amplitude that makes the
/// quantization error's mean *and* variance independent of the signal — less
/// leaves the contour partly visible, more is grain for nothing. It is a knob so
/// that a measurement has a control leg (`gg-tools banding`) and so that an
/// operator who suspects the noise can turn it off and look.
pub static DITHER: CVar = CVar::new_float("r.dither", 1.0, "output dither, in code values");

/// A flat ambient term, linear, and what a world declaring no `Sky` still gets —
/// a face pointing away from every light is dim rather than pure black.
///
/// Indirect light itself is §6 M24's environment, and since M28 it varies with
/// *where a fragment is* rather than only with which way it faces. What is still
/// P1 is that the variation is **authored**: a level says where its rooms are,
/// and nothing here derives an irradiance field from the geometry the way a probe
/// bake or a lightmap would. A hand-placed volume is the coarse version of that
/// and is honest about being one.
pub static AMBIENT: CVar = CVar::new_float("r.ambient", 0.03, "flat ambient light, linear");

/// Whether a surface gets back the light multiple microfacet bounces would have
/// returned to it (§6 M33).
///
/// The split-sum's second integral is single-scatter: a ray that leaves the
/// surface, hits it again and *then* leaves is counted as absorbed, so a rough
/// conductor reflects less than it received and the difference goes nowhere. It
/// is worth a third of the energy at roughness 1 and nothing at all at 0, which
/// is why it reads as "the rough end of the chart is too dark" rather than as a
/// bug — the smooth end, where anyone checks, is already right.
///
/// Off is the pre-M33 shading exactly and not a model of it, which is what makes
/// `gg-tools furnace`'s two legs one binary (`r.shadow_cull`'s argument, §6 M32).
pub static MULTISCATTER: CVar = CVar::new_bool(
    "r.multiscatter",
    true,
    "return the energy multiple microfacet bounces would (0 = single scatter)",
);

/// Whether the prefiltered chain is read along the lobe's dominant direction
/// instead of the mirror one (§6 M33).
///
/// The chain was integrated assuming the view, the normal and the reflection all
/// point the same way ([Kar13]) — so one lookup along `reflect` is exact head-on
/// and increasingly wrong toward the silhouette, where the real GGX lobe is
/// stretched and its centre of mass has slid back toward the normal. [LdR14]
/// §4.9.3's two lines move the lookup there.
///
/// Its own knob rather than a bit of [`MULTISCATTER`] because it is a claim
/// about the lobe's *shape* and that one is about its *energy*: they are
/// measured against different references and can be wrong independently — a
/// white furnace cannot see this one at all, since every direction in a uniform
/// environment returns the same radiance.
///
/// It is a *net* improvement and not a uniform one, which is why it is a knob
/// and not a constant: `gg-tools furnace` grades it against an importance-
/// sampled reference and finds the mean error over demo 06's panorama falling
/// from 6.8 % to 5.4 %, roughly halved at the rough end where the error is
/// largest, and slightly worse around roughness 0.4 where every aim is poor.
pub static LOBE: CVar = CVar::new_bool(
    "r.lobe",
    true,
    "read the chain along the lobe's dominant direction, not the mirror one",
);

/// Whether a fragment reads its own froxel's light list or the whole frame's
/// (§6 M30).
///
/// **Off is not a second code path.** `cluster::Assignment::build` answers false
/// by giving every froxel the same run — the entire point-light array — so the
/// shader is unchanged and what the knob selects is a *value*. That is what
/// makes it a measurement rather than a comparison between two implementations
/// with two sets of bugs: `gg-tools lights` sweeps it, and
/// `gg-render/tests/clusters.rs` requires the two renders to be byte-identical,
/// which is the assertion that assignment never under-includes.
pub static CLUSTERS: CVar = CVar::new_bool(
    "r.clusters",
    true,
    "assign lights to froxels instead of looping the frame's",
);

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

/// Point lights cast shadows (§6 M31). Off is every lamp lighting through every
/// wall, which is what the engine did until this milestone and is still the
/// control any measurement of it wants.
pub static LAMP_SHADOWS: CVar = CVar::new_bool(
    "r.lamp_shadows",
    true,
    "point lights cast shadows (0 = they light through walls, as before §6 M31)",
);

/// How many of the frame's point lights cast, clamped to
/// `[0, MAX_LAMPS](crate::lamp::MAX_LAMPS)`. The nearest ones, on extract's own
/// ordering — see `lamp`'s header for why this module invents no second ranking.
///
/// Six faces each, so this multiplies the shadow pass's draw *lists* by six and
/// the atlas's memory by one row. Four is what `gg-tools lamps` reads as the
/// knee on both devices; it is a frame-time knob and not a memory one, which is
/// the opposite of [`LAMP_SIZE`].
pub static LAMPS: CVar = CVar::new_int("r.lamps", 4, "point lights that cast shadows");

/// Edge of one lamp face in texels, clamped to `[128, 512]` and rounded up to a
/// power of two.
///
/// The clamp's ceiling is about the atlas's *width*: six faces across at 512 is
/// 3072, and `maxImageDimension2D` is only guaranteed to be 4096 (§4.3 asks for
/// no more). At the default four lamps a 512 tile is a 3072×2048 `D32_SFLOAT`
/// atlas — 24 MiB, against the sun's 64.
pub static LAMP_SIZE: CVar = CVar::new_int("r.lamp_size", 512, "lamp shadow face edge, texels");

/// Normal-offset reach for the lamp lookup, in **face texels** —
/// [`SHADOW_NORMAL_BIAS`]'s unit and its reasoning, against a texel whose world
/// size a lamp face grows with distance rather than holding fixed.
pub static LAMP_NORMAL_BIAS: CVar = CVar::new_float(
    "r.lamp_normal_bias",
    2.0,
    "lamp normal-offset reach, face texels",
);

/// The angle-free part of the lamp offset, in face texels — [`SHADOW_DEPTH_BIAS`]'s
/// half of the same pair.
pub static LAMP_DEPTH_BIAS: CVar =
    CVar::new_float("r.lamp_depth_bias", 1.0, "lamp depth bias, face texels");

/// Taps in the lamp filter. Lower than [`SHADOW_TAPS`] on purpose: a lamp's
/// filter is paid per casting lamp per fragment, where the sun's is paid once.
pub static LAMP_TAPS: CVar = CVar::new_int("r.lamp_taps", 8, "lamp shadow filter taps");

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

/// The output contract asked of the display: `0` SDR, `1` HDR10, `2` scRGB
/// (§6 M23). **Read once, at swapchain creation** — a colour space is not a
/// per-frame decision — and *written back* by the renderer to whatever the
/// display actually granted.
///
/// That write-back is the point and not a wart. HDR exists only where the
/// monitor, the compositor and the driver all agree, and none of that is
/// knowable before asking; a run that encoded PQ into an sRGB swapchain would be
/// a washed-out grey picture with no error anywhere to explain it. So the value
/// a session reads back is what it *got*, and an operator who set `1` and reads
/// `0` has been told the display said no.
///
/// The rendering has been HDR since M11 whatever this says: `Rgba16F` scene
/// attachment, PBR Neutral, linear throughout. This is only about the last
/// write. What SDR loses is not precision in the pipeline, it is the ability to
/// say "brighter than paper" at all.
pub static HDR: CVar = CVar::new_int("r.hdr", 0, "output: 0 sdr, 1 hdr10, 2 scrgb");

/// What the display should call diffuse white, in nits — the anchor the whole
/// HDR image hangs off, and meaningless under SDR.
///
/// PQ is an **absolute** encoding: a code value names a luminance rather than a
/// fraction of what the panel can do, so something has to say what the sRGB 1.0
/// a tonemapper produces is worth. 200 is the usual answer for a lit room and is
/// what Windows' own SDR-content slider defaults near; a dark room wants less.
/// Too high and the whole picture is glaring, which is the single most common
/// way HDR is set up wrongly.
pub static PAPER_WHITE: CVar = CVar::new_float("r.paper_white", 200.0, "hdr diffuse white, nits");

/// The brightest the display can usefully show, in nits — where the tonemapper's
/// shoulder is put.
///
/// The curve is the same Khronos Neutral one SDR uses, evaluated over
/// `peak / paper_white` instead of over 1: mid-tones come out where they were
/// and only the highlights roll off later, which is what HDR *is*. Claiming more
/// than the panel has clips the top of that roll-off back off again; claiming
/// less leaves headroom unused.
pub static PEAK_NITS: CVar = CVar::new_float("r.peak_nits", 1000.0, "hdr display peak, nits");

/// The narrowest a shadow's penumbra is allowed to be **on the screen**, in
/// pixels, clamped to `[0, 8]` (§6 M23).
///
/// This is the antialiasing floor, and it is the whole of why a shadow edge is
/// not a staircase. A shadow boundary is a shading discontinuity *inside* a
/// triangle: every MSAA sample in the pixel is on the same triangle and takes
/// the same shadow tap, so no sample count resolves it and `r.msaa` may as well
/// be off for the purpose. Coverage antialiasing cannot reach it — the only
/// thing that can is a filter at least a pixel wide, which is what this is.
///
/// Its unit is the point. The physical penumbra ([`SUN_ANGLE`]) is in *shadow
/// texels*, and a texel is a wildly different number of screen pixels in each
/// cascade and at each distance — which is how a three-texel kernel measured on
/// the desk came out **0.01 px** wide and put a 140-level step in one pixel.
/// Anything phrased in texels can go sub-pixel somewhere; only a floor phrased
/// in pixels cannot.
///
/// Zero is the pre-M23 look and is kept reachable because it is the control a
/// measurement wants, not because anything should ship with it.
pub static SHADOW_SOFTNESS: CVar = CVar::new_float(
    "r.shadow_softness",
    2.0,
    "narrowest shadow penumbra, in screen pixels",
);

/// The sun's angular **diameter** in degrees, clamped to `[0, 20]` — what makes
/// a penumbra grow with the distance to whatever cast it (§6 M23).
///
/// The real sun subtends 0.53°, which is the default and is why a crate's shadow
/// is crisp at its own feet and soft where it falls a few metres away. This is
/// the physical half of the filter: [`SHADOW_SOFTNESS`] is a floor the display
/// imposes, this is the width the light actually has. Widening it is an
/// overcast sky, and it is a look knob rather than an error.
///
/// Zero is a point sun — a hard shadow at every distance, floored to a pixel and
/// no wider.
pub static SUN_ANGLE: CVar = CVar::new_float("r.sun_angle", 0.53, "sun angular diameter, degrees");

/// Taps in the filter disk, clamped to `[4, 32]`.
///
/// The kernel is a Vogel disk rotated per pixel, so this trades noise against
/// cost directly and nothing else: `lit` takes one of `taps + 1` values, and the
/// rotation is what turns that quantization into grain instead of into rings.
/// The blocker search reuses the same disk at a quarter the count.
///
/// 16 is the knee. Under 8 the grain is visible on a wide penumbra; over 24 buys
/// nothing a still frame can show.
pub static SHADOW_TAPS: CVar = CVar::new_int("r.shadow_taps", 16, "shadow filter taps");

/// The widest a penumbra may get, in **shadow texels**, clamped to `[1, 64]`.
///
/// Two jobs, and they are the same number on purpose: it caps the filter disk,
/// and it *is* the blocker search radius. A blocker further off the receiver
/// than this cannot widen the penumbra past it, so searching further would find
/// occluders whose contribution the cap discards anyway.
///
/// What it really bounds is noise. A fixed tap count over a growing disk samples
/// ever more sparsely, so the cap is where "physically softer" stops being worth
/// the grain.
pub static SHADOW_PENUMBRA: CVar = CVar::new_float(
    "r.shadow_penumbra",
    16.0,
    "widest shadow penumbra and blocker search, in shadow texels",
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
        &DITHER,
        &HDR,
        &PAPER_WHITE,
        &PEAK_NITS,
        &AMBIENT,
        &MULTISCATTER,
        &LOBE,
        &CLUSTERS,
        &HISTOGRAM,
        &AA,
        &MSAA,
        &SHADOW_SIZE,
        &SHADOW_DISTANCE,
        &SHADOW_CASCADES,
        &SHADOW_CULL,
        &SHADOW_SPLIT_LAMBDA,
        &SHADOW_BLEND,
        &SHADOW_SOFTNESS,
        &SUN_ANGLE,
        &SHADOW_TAPS,
        &SHADOW_PENUMBRA,
        &SHADOW_NORMAL_BIAS,
        &SHADOW_DEPTH_BIAS,
        &LAMP_SHADOWS,
        &LAMPS,
        &LAMP_SIZE,
        &LAMP_NORMAL_BIAS,
        &LAMP_DEPTH_BIAS,
        &LAMP_TAPS,
    ])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    /// Every CVar this module declares is in [`super::register`].
    ///
    /// A declared-but-unregistered CVar is invisible rather than broken: the
    /// engine reads it through its static and gets the default, so nothing
    /// fails — it is simply absent from the console, from a config file and
    /// from the editor's panel, which is exactly where a session would go
    /// looking for it. `r.clusters` shipped that way at §6 M30 and nothing
    /// noticed, so the class gets a gate rather than the instance getting a fix.
    ///
    /// Reads its own source because there is no reflection over statics. Both
    /// sides are shapes this file controls, and a rename that broke the parse
    /// would fail here rather than pass silently — an empty list is refused.
    #[test]
    fn every_cvar_declared_here_is_registered() {
        const SOURCE: &str = include_str!("cvars.rs");
        let declared: Vec<&str> = SOURCE
            .lines()
            .filter_map(|line| line.strip_prefix("pub static "))
            .filter_map(|rest| rest.split_once(": CVar"))
            .map(|(name, _)| name)
            .collect();
        let body = SOURCE
            .split_once("cvar::register_all(&[")
            .and_then(|(_, rest)| rest.split_once("])"))
            .expect("register_all's list has moved")
            .0;
        let registered: Vec<&str> = body
            .lines()
            .filter_map(|line| line.trim().strip_prefix('&'))
            .filter_map(|name| name.strip_suffix(','))
            .collect();
        assert!(declared.len() > 20, "the declaration parse found nothing");
        let missing: Vec<&&str> = declared
            .iter()
            .filter(|name| !registered.contains(name))
            .collect();
        assert!(
            missing.is_empty(),
            "declared but never registered: {missing:?}"
        );
        let unknown: Vec<&&str> = registered
            .iter()
            .filter(|name| !declared.contains(name))
            .collect();
        assert!(
            unknown.is_empty(),
            "registered but not declared here: {unknown:?}"
        );
    }
}
