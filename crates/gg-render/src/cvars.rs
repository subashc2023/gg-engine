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
/// `recorded` because the editor builds its pick ray from it (§6 M40): move
/// this and a recorded click hits a different entity.
pub static FOV: CVar = CVar::new_float("r.fov", 1.0, "vertical field of view, radians").recorded();

/// With reverse-Z and an infinite far plane this is the *only* depth-precision
/// knob there is (§2, Math row) — which is what makes it worth turning without
/// a rebuild. `recorded` for [`FOV`]'s reason: it is the pick ray's other end.
pub static NEAR: CVar = CVar::new_float("r.near", 0.05, "near plane distance").recorded();

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

/// Show the hand's travel this frame instead of next tick (§6 M56); `0` renders
/// the interpolated tick, which is every frame this engine drew before M56.
///
/// A knob for `r.shadow_cull`'s reason rather than because the value is in
/// doubt: what it changes is *feel*, and feel is the one thing no gate in this
/// tree can grade. An operator with a 240 Hz panel toggling this mid-session is
/// the measurement — `gg-tools pace` can say the displayed angle went from 25 ms
/// behind the hand to under one, and cannot say whether that is the complaint.
///
/// Off costs nothing and changes nothing: a locked pace (`--frames`, every
/// replay, every golden run, §5.6) leaves no travel unspent at extract time, so
/// the latch is identically zero there whatever this says.
pub static LATE_LATCH: CVar = CVar::new_bool("r.late_latch", true, "latch the view to the hand");

/// A flat ambient term, linear, and what a world declaring no `Sky` still gets —
/// a face pointing away from every light is dim rather than pure black.
///
/// Indirect light itself is §6 M24's environment — varying with *where a
/// fragment is* since M28, and derived from the geometry rather than authored
/// since M36's field ([`GI`]). This stays the floor under both: what a world
/// declaring neither still gets, and what a probe's ray returns when it escapes
/// a world with no `Sky`, which is what makes `gg-tools bounce`'s ratio leg a
/// measurement of the field alone.
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

/// Paint every fragment by which side of its face the rasterizer thinks it is,
/// green for front and red for back — the one thing about a frame no buffer
/// records (§6 M64).
///
/// Not an entry in [`DEBUG_VIEW`]'s table, and the difference is the whole
/// reason this is a separate knob: that table resolves an *image the frame
/// already produced*, and facing is not an image. It is a per-fragment property
/// of the rasterizer that exists only inside the forward pass and is gone by the
/// time anything could sample it — recording it would cost an attachment on
/// every frame to answer a question asked twice a year.
///
/// What it is for: a surface shaded with the wrong side out does not look like a
/// facing bug, it looks like a *material* bug — no direct light, no albedo, and
/// the environment returned at full strength through a grazing Fresnel, which is
/// a white so bright it reads as a blown highlight. §6 M64 spent an hour on that
/// white before asking this question, and the answer took one frame: the red
/// pixels were exactly the chains and the foliage. It also reads a mesh's
/// *winding*, which is what found demo 06's ten inside-out spheres — the whole
/// sphere comes back red.
pub static FACING: CVar = CVar::new_bool("r.facing", false, "paint face winding, not shading");

/// Paint what the irradiance field made of every fragment instead of shading it
/// (§6 M66) — red where it gave up and used the fallback, green scaled by the
/// probe weight it mustered, blue where the grid's hull is fading out.
///
/// [`FACING`]'s sibling and not an entry in [`DEBUG_VIEW`]'s table, for the same
/// reason: the field's *verdict* is not an image. `gi_irradiance` returns the
/// caller's fallback wherever eight probes cannot see the point, and nothing
/// downstream records which pixels those were.
///
/// What it is for. "The fallback is the honest answer there rather than black" is
/// true only while the fallback is near what the field would have said — and in a
/// room lit *by* the field it is not. Demo 12's shelter reads 3x darker under the
/// fallback than under the field, so the bottom 60 mm of every wall in it came out
/// a hard black band (§6 M66). Nothing about that looks like a threshold in a
/// weight product: it looks like a shadow bug, or a wrong material, or dirt. The
/// only way to tell before this knob was to render the frame twice with `r.gi` off
/// and on and subtract, which finds *that* the field is responsible and never
/// which of its four terms went to zero.
pub static GI_PAINT: CVar = CVar::new_bool(
    "r.gi_paint",
    false,
    "paint the field's own verdict, not shading",
);

/// Paint *which* of the field's three suppressing terms is holding a fragment back
/// (§6 M67) — red burial, green visibility, blue the cosine wrap, white a fragment
/// no probe survived for, black healthy.
///
/// [`GI_PAINT`]'s other half, and the question that one leaves open: it reports how
/// much weight a fragment mustered, which is the *verdict*, and the four ways a
/// fragment can fail want four different repairs — grid placement, the distance
/// record's resolution, the wrap's floor, and a round robin that simply has not
/// arrived yet. §6 M66 answered it by subtracting two renders and reading the
/// shader by hand.
///
/// Its own bit rather than a mode of [`GI_PAINT`]: a paint is a picture and a
/// picture is either drawn or not, which is what makes both of these a `bool` and
/// neither of them an entry in [`DEBUG_VIEW`]'s table. Both set makes this one win,
/// since it is the more specific question.
pub static GI_TERM: CVar = CVar::new_bool(
    "r.gi_term",
    false,
    "paint which field term suppressed a fragment",
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

/// Whether the split-sum's second integral is read from §6 M34's table or from
/// [Laz13]'s analytic fit.
///
/// Off is the pre-M34 shading exactly, which is what makes the difference
/// measurable in one binary. What the fit cannot express is the *view* axis: it
/// returns `(-1.04a + z, 1.04a + w)`, so the directional albedo `scale + bias`
/// is `z + w` and carries no view angle — and that albedo is what §6 M33's
/// compensation divides by. `gg-tools split-sum` grades both against the
/// integral itself: the fit is 17.15 % mean and 121.68 % worst on `1/E`, the
/// table 0.21 % and 1.45 %.
///
/// The worst case is not an accuracy complaint, it is energy: at roughness 1
/// and grazing incidence the true single-scatter albedo is 0.998, so the
/// correction should be ~1, and the fit's constant 0.450 asked for 2.22 — a
/// surface returning more than twice the light that reached it. A white furnace
/// is blind to it, because at `f0 = 1` the ambient path sums to `Ess + (1 - Ess)`
/// whatever `Ess` is.
pub static LUT: CVar = CVar::new_bool(
    "r.lut",
    true,
    "read the split-sum's second integral from the table, not the analytic fit",
);

/// Whether the depth prepass's buffer is read for ambient occlusion (§6 M35).
///
/// The ambient and indirect terms multiply by a surface's occlusion, and until
/// this that number came from a *texture* an artist authored — so a
/// `Renderable`, which has no material maps at all, was lit as though it stood
/// in the open however deep in a corner it was. `gg-tools ao` measures how much
/// of demo 12's room that is: at the default radius, 27.5 % of the visible
/// pixels are occluded and the mean where occluded is 0.777.
///
/// Off is not "AO with the knobs at zero" — the passes are not declared at all
/// and the frame block's slot is zero, so a scene renders exactly the bytes it
/// rendered before this milestone. That is what makes the A/B in `gg-tools ao`
/// one binary and not two (§6 M32's argument for `r.shadow_cull`).
pub static AO: CVar = CVar::new_bool("r.ao", true, "occlude ambient light using the depth buffer");

/// How far ambient occlusion looks for an occluder, metres.
///
/// The knob with the widest defensible range on the list, because it is a
/// statement about *content*: a centimetre-scale radius creases contacts and
/// nothing else, a metre-scale one darkens whole alcoves. Half a metre is read
/// off `gg-tools ao`'s error sweep against the ray-cast reference — past it the
/// screen-space error grows faster than the term it is estimating, because a
/// wider radius asks the depth buffer about geometry further outside the frame.
pub static AO_RADIUS: CVar = CVar::new_float("r.ao_radius", 0.5, "occlusion radius, metres");

/// Exponent on the occlusion value, applied in the AO pass.
///
/// One is the integral itself and is the honest default. It is a knob because
/// the integral is *correct* rather than *pretty* — a scene lit almost entirely
/// by ambient reads flat at 1.0 — and because the instrument needs to be able to
/// prove the shipped value is 1.0 rather than a taste that drifted.
pub static AO_INTENSITY: CVar = CVar::new_float("r.ao_intensity", 1.0, "occlusion exponent");

/// Directions marched through each pixel, and taps along each half of one.
///
/// Both read off `gg-tools ao`. One slice is 0.1091 of mean error, two is
/// 0.0299, three is 0.0247 and four is 0.0248 — so the knee is three and the
/// shipped default is two, deliberately: on the pin three is a 28 % wider pass
/// (+3.93 ms against +3.07) and on a real GPU the difference is inside run-to-run
/// noise, while the 17 % it buys comes off the *smaller* half of the error —
/// the structural 0.13 a depth buffer cannot see is untouched by any slice
/// count. Steps matter far less once `r.ao_bias` is set (0.0329 → 0.0328 from 8
/// to 16), because the 4x4 rotation means a pixel's neighbours march different
/// directions and the denoiser recovers what one pixel's budget cannot.
///
/// **That last clause was false at every count but one until §6 M62**, and this
/// table is where it hid: the rotation stepped by a quarter of PI where the gap
/// between adjacent slices is `PI / slices`, so at two the four phases were two
/// and at four they were one. It is most of why three scored better than four
/// here — three is the count whose slice step a quarter-turn does not divide.
/// The error barely moved when it was fixed (0.0320 → 0.0329 at two, and 0.0589
/// → 0.0419 under the orthographic eye), because a mean absolute error cannot
/// tell occlusion that is wrong every other *column* from occlusion that is
/// wrong at random, and only one of those is something a player sees.
/// `gg-tools ao`'s pattern table is the column that can.
pub static AO_SLICES: CVar = CVar::new_int("r.ao_slices", 2, "occlusion directions per pixel");
/// Taps along each half of a slice — see [`AO_SLICES`].
pub static AO_STEPS: CVar = CVar::new_int("r.ao_steps", 8, "occlusion taps per half-slice");

/// Nearest a tap may be taken, in **pixels**.
///
/// The knob that fails in opposite directions, which is why it is measured and
/// not chosen (`r.shadow_normal_bias`'s shape, §6 M11): too small and a surface
/// at a grazing angle occludes itself, because a tap a fraction of a pixel away
/// differs from the fragment almost entirely along the view axis and the normal
/// that would forgive it is itself reconstructed from depths; too large and the
/// contact crease this term exists to draw is stepped straight over. `gg-tools
/// ao` prints both failures as separate columns — invented occlusion on open
/// surfaces, missed occlusion in creases — and this is the plateau between them.
pub static AO_BIAS: CVar = CVar::new_float("r.ao_bias", 2.0, "nearest occlusion tap, pixels");

/// Radius multiples over which a horizon's contribution fades to nothing.
///
/// What makes the radius a radius: without it a distant wall occludes as hard as
/// a crate on the floor, because the horizon it raises is just as high.
pub static AO_FALLOFF: CVar = CVar::new_float("r.ao_falloff", 1.0, "occlusion falloff, in radii");

/// Whether the raw occlusion target is denoised before the forward pass reads it.
///
/// A 4x4 box, which is exactly one period of the AO pass's rotation pattern and
/// is therefore an unbiased average of sixteen slice orientations rather than a
/// blur — sixteen *distinct* ones only since §6 M62, which is the milestone that
/// made the sentence true. A knob because "is the denoiser doing anything" is a
/// question the instrument should answer with a number: `gg-tools ao` prints the
/// error against the reference either way, and its pattern table prints what the
/// box is there to remove.
pub static AO_BLUR: CVar = CVar::new_bool("r.ao_blur", true, "denoise the occlusion target");

/// Which per-pixel index pair the occlusion pass rotates and offsets by: 0 none,
/// 1 `(x+y, x+2y)`, 2 `(x+y, x-y)`.
///
/// Zero is the control rather than a mode anyone ships — the question "how much
/// of this defect is the pattern at all" has no answer without it, and §6 M71
/// spent two wrong hypotheses before asking it. The other two differ in whether
/// the tap-offset index is *balanced*: see `ao.slang`, and `gg-tools ao
/// --crease` for the sweep the default is read off.
pub static AO_PATTERN: CVar = CVar::new_int("r.ao_pattern", 1, "occlusion's per-pixel index pair");

/// Which of the frame's intermediates is shown instead of the picture (§6 M59).
///
/// An index into [`DEBUG_VIEWS`], where 0 is off. An index rather than a name
/// because a CVar carries a number, and the names are exported beside it so the
/// editor's list and the shader's modes cannot drift apart.
///
/// **Off costs nothing**: no pass is declared, and the two views that need a
/// *sampleable* depth are the only reason a frame with `r.ao` off would allocate
/// one — see `scene_attachments`.
///
/// A view this frame has nothing for (the third cascade in a scene with two, the
/// field with `r.gi` off) shows the picture rather than black. There is no
/// failure to report: an intermediate that does not exist is a pass that did not
/// run, and the graph dump beside it already says so.
pub static DEBUG_VIEW: CVar = CVar::new_int(
    "r.debug_view",
    0,
    "show a frame intermediate; see r.debug_views",
);

/// Multiplier on whatever [`DEBUG_VIEW`] is showing. Metres of white under the
/// depth view, a plain gain everywhere else.
///
/// Zero means the view's own default, which is what makes one knob serve views
/// whose natural range differs by two orders of magnitude — occlusion is already
/// 0..1 and a depth buffer is metres.
pub static DEBUG_SCALE: CVar =
    CVar::new_float("r.debug_scale", 0.0, "debug view gain; 0 is the view's own");

/// What [`DEBUG_VIEW`]'s index means, in order. Index 0 is off.
///
/// Public because the editor lists them and `gg-tools` names them. The claim
/// here used to be that `debug_source` "reads this one" and so a second copy
/// could not go stale; it was **false** (§6 M86). That function is a `match` on
/// bare integers and touched this table only for its `len()`, and
/// `debug_wants_depth` was a third copy of the same ordering — so a view
/// inserted anywhere but the end would have renamed every later one, with the
/// comment saying it could not happen.
///
/// The ordering is now [`views`] and this table is *derived from it*, so a row
/// is a name and an index at once and the renderer's arms are named.
pub const DEBUG_VIEWS: &[&str] = views::NAMES;

/// One ordering for `r.debug_view`: the index, the name, and the arm the
/// renderer resolves (§6 M86).
///
/// A `usize` per view rather than an enum because the CVar carries a number and
/// the editor lists strings — an enum would need a conversion at each end, which
/// is two more places for the ordering to live.
pub mod views {
    /// The names, in index order. Index 0 is off.
    pub const NAMES: &[&str] = &[
        "off", "scene", "depth", "normal", "ao", "ao-raw", "shadow.0", "shadow.1", "shadow.2",
        "shadow.3", "lamps", "field", "moments",
    ];

    /// The picture, unmodified — the one view whose source is the frame itself.
    pub const SCENE: usize = 1;
    /// The camera's depth as metres.
    pub const DEPTH: usize = 2;
    /// The normal `gg-render`'s occlusion pass reconstructs from that depth.
    pub const NORMAL: usize = 3;
    /// The denoised occlusion target.
    pub const AO: usize = 4;
    /// The occlusion pass's own output, before the blur.
    pub const AO_RAW: usize = 5;
    /// The first sun cascade; the four are contiguous.
    pub const SHADOW_0: usize = 6;
    /// One past the last cascade — the half-open end of that run.
    pub const SHADOW_END: usize = 10;
    /// The lamp atlas.
    pub const LAMPS: usize = 10;
    /// The irradiance field's radiance record.
    pub const FIELD: usize = 11;
    /// The field's visibility moments.
    pub const MOMENTS: usize = 12;

    /// Every index has a name and every name its index — the assertion that
    /// makes the two halves above one fact rather than two.
    const _: () = {
        assert!(NAMES.len() == MOMENTS + 1);
        assert!(SHADOW_END == SHADOW_0 + 4);
        // A `&str` comparison is not `const`, so the check is on the bytes.
        const fn is(index: usize, name: &str) -> bool {
            let (a, b) = (NAMES[index].as_bytes(), name.as_bytes());
            if a.len() != b.len() {
                return false;
            }
            let mut i = 0;
            while i < a.len() {
                if a[i] != b[i] {
                    return false;
                }
                i += 1;
            }
            true
        }
        assert!(is(0, "off"));
        assert!(is(SCENE, "scene"));
        assert!(is(DEPTH, "depth"));
        assert!(is(NORMAL, "normal"));
        assert!(is(AO, "ao"));
        assert!(is(AO_RAW, "ao-raw"));
        assert!(is(SHADOW_0, "shadow.0"));
        assert!(is(SHADOW_END - 1, "shadow.3"));
        assert!(is(LAMPS, "lamps"));
        assert!(is(FIELD, "field"));
        assert!(is(MOMENTS, "moments"));
    };
}

/// Whether the irradiance nobody authored is gathered at all (§6 M36).
///
/// Off is not "the field at zero": no probe renders, neither field image is
/// created, the frame block's slot is zero, and `ambient_light` composites the
/// authored [`Sky`](gg_ecs::boundary::Sky) volumes exactly as it did before this
/// milestone. That is what makes the A/B in `gg-tools bounce` one binary
/// (§6 M32's argument for `r.shadow_cull`) — and what a scene blessed before M36
/// still renders.
///
/// **Off by default since §6 M71, and that is a judgement rather than a
/// measurement.** The numbers favour the field and always have: `gg-tools
/// bounce` puts the authored volumes +0.152 too bright on a room's dark half and
/// -0.087 too dark on its bright one, against a field that is 2.9 % short of
/// energy and wrong in no bucket by that much. What the numbers do not grade is
/// three milestones of reports about how it *looks* — chevrons on a flat wall
/// (M68), facets a `r.gi_moments` bump subdivides rather than removes (M69), and
/// a shortfall M70 attributed to the four coefficients a probe stores and priced
/// three fixes for, all of which were declined. A term that is more accurate on
/// average and visibly structured in place is worse than one that is neither,
/// and the operator's call is that the fancy term waits until the plain ones are
/// right. Nothing is deleted: `r.gi 1` is the field, every instrument that grades
/// it still does, and the residuals stay named.
pub static GI: CVar = CVar::new_bool("r.gi", false, "gather bounced light from the scene itself");

/// Metres between probes.
///
/// The knob that sets what the field can *resolve*: light changes over the
/// distance a wall is from the floor beside it, and a spacing wider than that
/// averages the two together — which is what a chevron on a flat wall is, since
/// interpolation across a cell is a saddle and its level sets are hyperbolas.
///
/// **It was a floor rather than a promise until §6 M68, and that is what the
/// coupling cost.** `Grid::fit` widened it by a whole multiple until the axis cap
/// spanned the scene's *widest* axis, so a 24 m room could not have cells finer than
/// 4 m and Sponza not finer than 8 m, whatever was asked for. `Grid::place` anchors
/// an axis that cannot fit instead of widening every axis to suit it, so this is now
/// **exact** in a session and the widening is reachable only through `Grid::fit`
/// itself.
///
/// Two metres is read off the picture and off `bounce --energy`'s slab rather than off
/// that command's mean-error tables, and the footing is worth stating: those tables
/// prefer a *coarse* field at every spacing, because a coarse grid leaks light through
/// walls and the leak cancels the ~5 % of energy the record is short (§6 M68). Until
/// that shortfall is fixed, a mean absolute error cannot rank the resolution.
pub static GI_SPACING: CVar =
    CVar::new_float("r.gi_spacing", 2.0, "metres between probes, at least");

/// Where the probe lattice sits between the multiples of its own spacing, as a
/// fraction of a cell (§6 M67).
///
/// Zero is a lattice through the world origin, which is what `Grid::fit` did for
/// three milestones and what put whole probe planes *inside* demo 12's floor and
/// walls: authored geometry lands on round numbers and the spacing divides them.
/// §6 M66 fixed the probe that lands there and named this as the follow-up; §6 M67
/// measured it and `gg-tools bounce`'s offset table is what the default is read
/// off.
///
/// A knob rather than a constant because it is the one number here whose right
/// value is a property of *content* — how a level author rounds — and because the
/// table it is read off has to be re-readable on a scene that is not demo 12's
/// room.
pub static GI_OFFSET: CVar = CVar::new_float(
    "r.gi_offset",
    LATTICE_OFFSET,
    "probe lattice offset, in cells",
);

/// How fast a probe's weight falls with the fraction of its own sphere that came
/// back a back face (§6 M67) — zero trusts a fully buried probe as much as an open
/// one, and two rejects one at the half-buried point.
///
/// A probe inside a wall recorded the inside of that wall, so its record is a colour
/// and not a light. What makes this a knob rather than a constant is that its right
/// value is a *plateau* between two failures — light through a wall against light
/// missing from a crease — and §6 M66 read that plateau off a table which, §6 M67
/// found, had been measured through a field two thirds of which held the previous
/// sweep row's records. `gg-tools bounce`'s burial table is the corrected reading.
pub static GI_BURIAL: CVar =
    CVar::new_float("r.gi_burial", 1.5, "how fast a buried probe's weight falls");

/// How far a shading point is pushed along its normal before it is located in the
/// grid — **in metres, and it was in probe spacings until §6 M68**.
///
/// The push exists because a shading point sits exactly on geometry the probes
/// recorded, so without it its own wall is at distance zero from it and every
/// visibility bound reads as occluded by the very surface being shaded.
///
/// The unit is half the finding. It was `0.15 * spacing`, which is 0.60 m at the
/// 4 m grid the axis cap used to force and 0.30 m at 2 m — so every attempt to make
/// the cells finer also halved the term that escapes self-occlusion. A surface's own
/// thickness is not a function of how far apart the probes are, so the unit is
/// metres and 0.6 is what the old expression meant at the only spacing it ever ran
/// at.
///
/// **The other half is why it is not larger**, because `gg-tools bounce` says larger
/// is better and that reading is a trap. Its bias table at 2 m falls monotonically
/// to the end of the sweep — .1023 total error at 0.6 m against .0974 at 1.2 — and
/// 1.2 m is six tenths of a cell, which does not push a shading point off its
/// surface, it locates it somewhere else. What the improvement is: every extra metre
/// of push brightens the field, and the field is **5 % short of energy at every
/// spacing from 1 m to 6 m** on the one leg whose truth is exactly 1.0
/// (`bounce --energy`'s slab). So the knob is being paid to compensate for a deficit
/// it has nothing to do with, and it reads as an improvement because the metric is a
/// mean. 0.6 m is where the curve *turns* in the one configuration where it turns at
/// all — a genuine two-sided plateau at 4 m spacing, .0750 either side of .0723 —
/// and a monotone gain that is a known defect wearing a disguise is not a plateau.
pub static GI_BIAS: CVar = CVar::new_float(
    "r.gi_bias",
    0.6,
    "shading-point offset off its surface, metres",
);

/// [`GI_OFFSET`]'s default: **zero, and the argument for moving it was wrong**.
///
/// §6 M66 named an offset lattice as its sized follow-up with numbers in hand — 17.5 %
/// of the truth invented falling to 4.9 %, total error down 23 %. §6 M67 built the
/// knob, fixed the two convergence defects that measurement was taken through, and got
/// the opposite answer: every nonzero offset is worse, and the best of them (a half) is
/// still 31 % worse in total error than a lattice through the world origin.
///
/// **Why, and it is not subtle once the sweep exists.** The offset's premise is that a
/// probe coplanar with a surface is a wasted probe. That is true of a *wall*, which is
/// thin — a third of a cell puts the probe through it and out the far side. It is false
/// of a *floor* under an open sky, where a probe on the surface sees the whole sky
/// hemisphere and is the best-placed probe in the room, because the floor is where the
/// receivers are. At demo 12's 4 m effective spacing the vertical axis has two useful
/// planes, so moving them off the floor costs more than moving them out of the walls
/// buys. The offset trades a wall problem for a floor problem and the floor problem is
/// bigger.
///
/// Kept as a knob rather than deleted: the trade is a property of *content* — how tall
/// a level is against how thick its walls are — and the next level to ask this question
/// deserves the table rather than this paragraph.
pub const LATTICE_OFFSET: f64 = 0.0;

/// Probes gathered per frame, or `0` for all of them.
///
/// A probe is **six draws of the scene**, which is the price of a field that
/// sees whatever the renderer draws rather than a second description of the
/// world. So a session spreads them: the field converges over
/// `probes / r.gi_rate` frames and `Probes::pending` is what is left, the way
/// `Content::pending` is for a streaming pack.
///
/// Zero is for a *reference*: a harness that wants the whole field in one frame
/// and will pay for it. It is not a sensible session value and is not the
/// default for that reason.
pub static GI_RATE: CVar = CVar::new_int("r.gi_rate", 16, "probes gathered per frame (0 = all)");

/// Edge of one probe face, texels.
///
/// Six of these are the whole sphere around a probe, so the sample count is
/// `6 * edge²` — 384 at the default. What it has to resolve is *irradiance*,
/// which is the scene convolved with a cosine lobe, so it is a far smaller
/// number than a shadow map's: [`GI_FACE`] trades against nothing visible until
/// a small bright thing (a lamp, a window) falls between samples.
///
/// **Swept at last in §6 M70**, against `gg-tools bounce --energy`'s slab, whose truth
/// is 1.0 by construction — and the sweep says the argument above is right and the
/// default is one row past where it stops paying:
///
///   face | samples | quadrature |  field | short | vs sum
///      4 |      96 |    +1.53 % | 0.9784 | 2.2 % |  3.7 %
///      8 |     384 |    +0.38 % | 0.9729 | 2.7 % |  3.1 %   (shipped)
///     16 |    1536 |    +0.10 % | 0.9715 | 2.9 % |  2.9 %
///     32 |    6144 |    +0.02 % | 0.9711 | 2.9 % |  2.9 %
///
/// The `quadrature` column is why the sweep needed no reference renderer and why the
/// `short` column alone would have misled: `probe_face_solid_angle` is a midpoint rule
/// on the gnomonic Jacobian, it **over**-counts, and an SH projection is not
/// normalized by its own measure — so the coarse rows were being flattered by exactly
/// the amount they were wrong. Corrected (`vs sum`), the error is flat from 8 upward,
/// and what is left is the record's four coefficients rather than its sampling
/// (`probe_irradiance`).
pub static GI_FACE: CVar = CVar::new_int("r.gi_face", 8, "probe face edge, texels");

/// Octahedral edge of a probe's distance tile.
///
/// Directions the visibility test can tell apart: `edge²` of them, 64 at the
/// default.
///
/// **It was 4, and the sentence that kept it there was a measurement error of
/// kind** (§6 M69). The doc said the knob "does nearly nothing" because
/// `gg-tools bounce`'s tile table is flat to 0.3 % across 2, 4 and 8 — which it
/// is, and which says nothing, because that table grades a *mean* against
/// path-traced truth and the tile's failure is structure rather than level. What
/// the tile's resolution actually costs is a bound quantised to `edge²`
/// directions, and until M69 that bound was point-sampled, so it was
/// discontinuous in direction and its discontinuities projected from a probe onto
/// a flat wall as straight lines. A report about "large geometric triangular
/// shadows" above the shelter's roof was one triangle per probe cell at edge 4.
///
/// `gg-tools facets` is the table it is now read off, and what that table says
/// about *this* knob is a warning: 4 to 8 does not remove the facets, it
/// **subdivides** them — three large triangles become eight small ones and the
/// crease count moves by a tenth. The fix is [`GI_FILTER`]; 8 is here because with
/// the bound continuous the edge is a quality knob again, and 8 is the finest the
/// [`GI_FACE`] gather can feed (its kernel half-angle is 15.3° against a face
/// texel's 11.25°; at 12 the kernel is narrower than the samples it averages).
pub static GI_MOMENTS: CVar = CVar::new_int("r.gi_moments", 8, "probe distance tile edge, texels");

/// Whether the visibility bound's distance tile is filtered (§6 M69).
///
/// **The fix, and off is the pre-M69 renderer exactly** — `r.shadow_cull`'s and
/// `r.lut`'s arrangement (§6 M32, M34), so `gg-tools facets` grades two algorithms
/// in one binary instead of two builds. The tile carries a one-texel border either
/// way (`probe::MOMENT_BORDER`), because a border costs 56 % of a 1 MiB atlas and
/// a second code path for the unbordered layout would cost a reader far more.
///
/// What it buys, from `gg-tools facets` — the graded leg is a wall above a roofed
/// alcove seen head-on, where the truth for a second difference is exactly zero,
/// and the room leg is demo 12's shelter from the chair the report was made at,
/// whose floor is that framing with the field off:
///
///   read            | wall crease % | room crease % | room worst
///   tile 4, point   |     0.307     |     0.365     |     63       (pre-M69)
///   tile 8, point   |     0.006     |     0.341     |     38
///   tile 4, filtered|     0.001     |     0.153     |     50
///   tile 8, filtered|     0.002     |     0.153     |     38       (shipped)
///   the room's floor:              |     0.136     |
///
/// Read the room column against its floor: point-sampled at edge 4 is 0.229 above
/// it and filtered is 0.017, which is thirteen times less crease for the same
/// picture. Read the two point rows against each other for why the edge is not the
/// fix — 4 to 8 moves that excess by a tenth, because a finer tile has more
/// boundaries each carrying a smaller step, and the crease pictures show three
/// large triangles becoming eight small ones.
///
/// **And it costs nothing the accuracy tables can see.** `bounce`'s tile table
/// reads 0.1001–0.1002 whole error across all six of those rows — the same
/// flatness that hid the defect now says the fix is free. On the 4090 the forward
/// pass goes 0.307 → 0.315 ms for the four taps and `probe-moments` does not move
/// at all; on the pinned lavapipe, which is a CPU rasterizer, the border and the
/// finer edge cost the moment pass 3.0 → 4.1 ms and the frame 7.6 %.
pub static GI_FILTER: CVar = CVar::new_bool(
    "r.gi_filter",
    true,
    "filter the probe visibility bound's tile",
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

/// What fraction of the window the *scene* is drawn at, before the post pass
/// puts it on the screen (§6 M78). Clamped to `[0.25, 2.0]`; `1.0` is off and is
/// exactly the frame this renderer drew before the knob existed.
///
/// It is the only knob here whose value is read off a measurement rather than a
/// look, and the measurement is `gg-tools frame --sweep`: demo 12's room is
/// **99.7 % fragments** on the integrated Radeon in this desk — 37.7 ms per
/// megapixel with a floor of 0.2 — so the frame is very nearly proportional to
/// the pixel count and nothing else. 79.4 ms at 1920x1080 is 34.0 at 1280x720,
/// which is twelve frames a second becoming twenty-nine on a machine that
/// changed nothing but this.
///
/// **What it does not touch is the point.** The UI, the HUD and the pointer are
/// a separate pass into the backbuffer at the window's own size, so text stays
/// exactly as sharp as it was — which is what makes this a knob a player will
/// actually leave turned down. Nor does it reach the shadow maps, the lamp
/// atlas or the probe atlas: those are sized by their own knobs, and the sweep's
/// per-pass table reads 1–37 % fill on them for that reason.
///
/// Above `1.0` it is supersampling, and composes with [`MSAA`] rather than
/// competing with it — at exactly `2.0` the post pass's bilinear tap lands
/// between four texels and is an exact box filter.
pub static SCALE: CVar = CVar::new_float(
    "r.scale",
    1.0,
    "scene resolution, as a fraction of the window",
);

/// Whether a `transparency` above zero draws as glass (§6 M92) — the
/// transparent pass, the missing shadow, the missing prepass entry, all of it.
/// Off files every such surface back under opaque, which is the world before
/// the field existed, not a world with holes in it: the switch exists so
/// `gg-tools frame --attribute` can price the pass and an operator can ask
/// whether an artefact is the blend or the surface, and a surface that
/// *vanished* would answer neither question.
pub static GLASS: CVar = CVar::new_bool("r.glass", true, "draw transparency as glass");

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

/// Every cascade records the *widest* one's depth (§6 M60). Off is the older
/// behaviour, where each derived its own from its own width.
///
/// A knob for `r.shadow_cull`'s reason: the value is not in question, and what
/// makes it worth keeping is that the two sides of a correctness fix are then one
/// binary a flag apart. Off, a caster further up-light than a small cascade's
/// light eye is absent from that map, and absent depth is absent blocker — so the
/// receiver renders **lit**. It is worst where the map is tightest, which is the
/// near field, and it gets worse as `r.shadow_distance` comes *down*.
pub static SHADOW_REACH: CVar = CVar::new_bool(
    "r.shadow_reach",
    true,
    "every cascade records the widest one's depth (0 = its own, as before §6 M60)",
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

/// When a finished frame reaches the glass: `1` wait for the blank, `2` replace
/// the queued frame, `3` straight to the glass (§6 M46). [`HDR`]'s write-back,
/// for [`HDR`]'s reason — only vsync is guaranteed to exist, so an operator who
/// set `2` and reads `3` has been told the driver has no mailbox.
///
/// Unlike [`HDR`] this is read **every frame**, because a present mode does not
/// change what the numbers in the backbuffer mean and so is not a decision that
/// has to be made before the pipelines are built. A player's own choice
/// (`Prefs::present`) overrides it when the game offers one; a game that draws
/// no video menu leaves it alone, which is what makes this the host's knob
/// rather than a duplicate of the boundary's.
///
/// Default is vsync and stays vsync: tearing is a thing a player asks for, never
/// a thing a default hands them, and every golden in the tree is rendered
/// through a path that presents exactly once per frame.
pub static VSYNC: CVar = CVar::new_int("r.vsync", 1, "present: 1 vsync, 2 fast, 3 immediate");

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
        &FACING,
        &GI_PAINT,
        &GI_TERM,
        &LATE_LATCH,
        &HDR,
        &VSYNC,
        &PAPER_WHITE,
        &PEAK_NITS,
        &AMBIENT,
        &MULTISCATTER,
        &LOBE,
        &LUT,
        &AO,
        &AO_RADIUS,
        &AO_INTENSITY,
        &AO_SLICES,
        &AO_STEPS,
        &AO_FALLOFF,
        &AO_BIAS,
        &AO_BLUR,
        &AO_PATTERN,
        &DEBUG_VIEW,
        &DEBUG_SCALE,
        &GI,
        &GI_SPACING,
        &GI_OFFSET,
        &GI_BURIAL,
        &GI_BIAS,
        &GI_RATE,
        &GI_FACE,
        &GI_MOMENTS,
        &GI_FILTER,
        &CLUSTERS,
        &HISTOGRAM,
        &AA,
        &MSAA,
        &SCALE,
        &GLASS,
        &SHADOW_SIZE,
        &SHADOW_DISTANCE,
        &SHADOW_CASCADES,
        &SHADOW_CULL,
        &SHADOW_REACH,
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
