// The §4.5 v1 fullscreen post pass, which since M11 is the tonemapper: exposure,
// then a curve, from the `Rgba16F` scene attachment onto whatever the frame lands
// in. The pass already existed for the graph's sake — a consumer of a colour
// attachment another pass wrote — and this is the work §6 M11 said would land in
// it without a new pass.
//
// The target is sRGB and Vulkan encodes on write, so everything here is linear
// and nothing applies a gamma by hand. A second encode is the classic washed-out
// frame. `dither` is the one exception and it round-trips: it has to perturb the
// value the *quantizer* sees, so it encodes, adds, and decodes straight back for
// the hardware to encode again.

// The global set (§4.3). Binding 1 (storage images) has no consumer here.
[[vk::binding(0, 0)]] Texture2D<float4> g_textures[];
[[vk::binding(2, 0)]] SamplerState g_samplers[];

struct PostPush
{
    uint texture_index; // the scene attachment's slot in g_textures
    uint sampler_index; // NearestClamp resolving, LinearClamp when aa != 0
    float exposure;     // stops, so 0 is unchanged and +1 is twice the light
    uint aa;            // 0 off, 1 fxaa — `r.aa`, or the game's Prefs (§6 M21)
    float2 texel;       // 1/extent of the source, in uv units
    float dither;       // `r.dither`, in output code values; 0 is off (§6 M22)
    uint output;        // the swapchain's granted contract, `gg_rhi::Output` (§6 M23)
    float paper_white;  // what to call diffuse white, in nits; unread under SDR
    float white;        // the display's peak in units of paper white; 1.0 under SDR
}

[[vk::push_constant]]
ConstantBuffer<PostPush> push;

struct VOut
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
}

// A fullscreen *triangle*, not a quad: three vertices generated from the index
// alone (nothing binds a vertex buffer — §4.3), and no diagonal seam where two
// triangles would meet. uv 0 lands at clip -1, which is the top row in Vulkan's
// Y-down clip space — the same row as texel row 0, so nothing is flipped.
[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.uv = uv;
    o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

// Khronos PBR Neutral (KhronosGroup/ToneMapping, 2024) — the reference
// implementation, transcribed.
//
// Chosen over AgX and over ACES for one reason that outranks taste: it is
// **hue-preserving and value-preserving in the mid-range**, so an artist's base
// colour arrives on screen as the colour they picked until it is bright enough
// to need compressing. ACES tints and shifts hue at ordinary intensities, and
// AgX needs either a LUT or a large polynomial fit — a checked-in table would be
// a second thing to keep in sync with the shader that reads it.
//
// The three constants are the reference's own and are not tuned here. Retuning a
// published curve without a viewing environment to justify it produces a look
// nobody can reproduce.
static const float START_COMPRESSION = 0.8 - 0.04;
static const float DESATURATION = 0.15;

float3 pbr_neutral(float3 color)
{
    // Pull the darkest channel toward zero, which is what keeps a saturated dark
    // colour from washing out: without it the compression below lifts the whole
    // triple together.
    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < START_COMPRESSION)
    {
        // Below the knee the curve is the identity, which is the property that
        // makes an authored mid-tone survive to the screen unchanged.
        return color;
    }

    float d = 1.0 - START_COMPRESSION;
    float new_peak = 1.0 - d * d / (peak + d - START_COMPRESSION);
    color *= new_peak / peak;

    // Desaturate toward white as the peak is compressed — a very bright
    // saturated colour reads as its own hue at full strength otherwise, which is
    // not what a real sensor or an eye does.
    float g = 1.0 - 1.0 / (DESATURATION * (peak - new_peak) + 1.0);
    return lerp(color, new_peak.xxx, g);
}

// One tap of the finished picture: the scene's radiance, exposed and tonemapped.
//
// Everything the antialiaser below does is done to *this* — display-referred
// values in 0..1 — and never to the radiance behind it. That is not a detail of
// convenience: an edge between 0.4 and 40.0 nits blended in HDR and tonemapped
// afterwards lands wherever the curve's knee happens to put the mean, which is
// nowhere near the halfway point an eye expects. Tonemapping each tap and
// blending the results is the order that makes a blended edge read as half.
float3 tap(float2 uv)
{
    float3 radiance = g_textures[push.texture_index].Sample(g_samplers[push.sampler_index], uv).rgb;
    // Exposure in stops, applied *before* the curve: it is the choice of what
    // counts as "1.0", and applying it after would be adjusting a picture that
    // had already been compressed.
    float3 exposed = radiance * exp2(push.exposure);
    // The same curve over a wider range, which is the whole of what HDR output
    // changes about the *look* (§6 M23). `white` is the display's peak in units
    // of diffuse white — 1.0 under SDR, and five under a 1000-nit panel calling
    // paper 200. Evaluating the curve over `[0, white]` and scaling back keeps
    // the identity region identical (the knee is at 0.76 of white either way, and
    // below it this is exactly `exposed`) and moves only the shoulder, so an
    // authored mid-tone arrives where it did and highlights get somewhere to go.
    // Scaling the *output* of an SDR curve instead would brighten the whole
    // picture, which is the usual way this is got wrong.
    return pbr_neutral(exposed / push.white) * push.white;
}

// Perceptual weight, Rec. 709 without blue. Blue's coefficient is 0.07 and
// dropping it is the reference implementation's own economy — the term costs a
// multiply-add per tap and moves an edge decision almost never.
float luma(float3 c)
{
    return c.r * 0.299 + c.g * 0.587;
}

// FXAA, the console variant (Lottes, NVIDIA 2011, FXAA 3.11 `FxaaPixelShader`
// with `FXAA_PC == 0`) — transcribed, with its three constants unchanged.
//
// Why this and not MSAA: MSAA antialiases *geometry* and would have to reach
// into `gg-rhi`, which today creates every image and every pipeline at one
// sample (`SampleCountFlags::TYPE_1`) — sample counts on attachments, on
// pipelines, on the depth prepass the forward pass shares, and a resolve the
// post pass could sample. That is a milestone with a public-API bless and a
// re-bless of every golden on three backends in it. This is a fullscreen read,
// costs nothing the frame notices, needs no RHI change, and — unlike MSAA —
// also softens the edges *inside* a triangle, which is where a flat-shaded
// engine's worst aliasing actually is.
//
// The cost it does have is the honest one: it cannot tell a real edge from an
// authored one, so a high-contrast texture detail one pixel wide is softened
// along with the staircase. That is why it is a mode and not a constant.
static const float FXAA_SPAN_MAX = 8.0;
static const float FXAA_REDUCE_MUL = 1.0 / 8.0;
static const float FXAA_REDUCE_MIN = 1.0 / 128.0;

float3 fxaa(float2 uv, float3 centre)
{
    // The four corners are sampled *between* texels, so a linear sampler makes
    // each one the mean of the four it sits among — which is what turns nine
    // reads into the sixteen-texel neighbourhood the filter reasons about. With
    // a nearest sampler this degenerates to four diagonal texels and the edge
    // direction below is quantized to the diagonals; that is why the mode picks
    // `LinearClamp` and the resolve path does not.
    float2 half_texel = push.texel * 0.5;
    float nw = luma(tap(uv + float2(-half_texel.x, -half_texel.y)));
    float ne = luma(tap(uv + float2(half_texel.x, -half_texel.y)));
    float sw = luma(tap(uv + float2(-half_texel.x, half_texel.y)));
    float se = luma(tap(uv + float2(half_texel.x, half_texel.y)));
    float m = luma(centre);

    float lo = min(m, min(min(nw, ne), min(sw, se)));
    float hi = max(m, max(max(nw, ne), max(sw, se)));

    // The edge runs *along* the smaller gradient, which is why the two rows are
    // crossed: a vertical staircase has a large horizontal luma step and is
    // blended horizontally.
    float2 dir = float2(-((nw + ne) - (sw + se)), (nw + sw) - (ne + se));

    // Reduction keeps the step bounded where the neighbourhood is dark: `dir`
    // is a difference of luma, so on a near-black edge it is a tiny number
    // whose reciprocal would otherwise fling the taps across the screen. The
    // `MIN` term is what stops a perfectly flat neighbourhood dividing by zero.
    float reduce = max((nw + ne + sw + se) * 0.25 * FXAA_REDUCE_MUL, FXAA_REDUCE_MIN);
    float rcp_min = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcp_min, -FXAA_SPAN_MAX.xx, FXAA_SPAN_MAX.xx) * push.texel;

    // Two taps inside the pixel, two out at the span's ends. The inner pair is
    // the safe answer; the outer pair sharpens a long shallow edge and is
    // discarded below when it reaches past the neighbourhood's own range.
    float3 inner = 0.5 * (tap(uv + dir * (1.0 / 3.0 - 0.5)) + tap(uv + dir * (2.0 / 3.0 - 0.5)));
    float3 outer = inner * 0.5 + 0.25 * (tap(uv + dir * -0.5) + tap(uv + dir * 0.5));

    // The whole guard against over-blurring: a result outside what the 3x3
    // neighbourhood contained came from somewhere this pixel cannot see, so the
    // inner pair stands instead. Without it a thin bright line is smeared into
    // whatever is two pixels away from it.
    float outer_luma = luma(outer);
    return (outer_luma < lo || outer_luma > hi) ? inner : outer;
}

// The sRGB transfer pair, IEC 61966-2-1. Present here for one reason: the
// quantizer this pass feeds sits on the *other* side of it — the attachment is
// `_SRGB` and the hardware encodes on write — so a perturbation applied to a
// linear value is a different number of code values at every brightness, and at
// the dark end where banding is worst it is a fraction of one.
float3 linear_to_srgb(float3 c)
{
    c = max(c, 0.0);
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

// Linear value one *code value* is worth at this brightness — the derivative of
// the decode, evaluated where the colour already sits. Above the toe
// `linear = ((e + 0.055) / 1.055)^2.4`, so `dlinear/de = 2.4 * linear / (e +
// 0.055)` and the value we already have does the work of the second power.
float3 code_value(float3 colour, float3 encoded)
{
    return select(encoded <= 0.04045, (1.0 / 12.92).xxx,
                  2.4 * colour / (encoded + 0.055));
}

// A 32-bit integer hash — exact on every device by construction, which is what
// makes a dithered frame reproducible enough to bless (§4.10). A `frac(sin(...))`
// noise would not be: the transcendental is the driver's to round.
// (Jenkins-style mix, the `lowbias32` constants.)
uint hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Uniform noise in (-0.5, 0.5).
//
// **Uniform and not triangular**, which is a deliberate trade against the
// classical result (Lipshitz, Wannamaker & Vanderkooy, JAES 1992: a triangular
// PDF two LSBs wide makes the quantization error's variance independent of the
// signal as well as its mean; a uniform one leaves the variance modulating, so
// the noise faintly breathes across a gradient). Half a code value is what the
// window below can scale to zero at a code; an LSB could not, and a flat region
// that is exact on one driver and not another is worse than a variance term
// nobody can see at one part in 255.
float noise(uint seed)
{
    return float(hash(seed) >> 8) * (1.0 / 16777216.0) - 0.5;
}

// Code-value distance from a code over which the dither ramps back up to full.
// See `dither` — it is a bound on the *driver's* encode error, not a taste knob.
static const float DITHER_DEADZONE = 0.25;

// Half a code value of the *output* is what a smooth gradient has to be broken
// up by — scaled by how far the value sits from a code, and applied through the
// decode's slope rather than by a round trip. Both of those are load-bearing.
//
// **The window.** A contour is where the round *changes*, which is where the
// value sits halfway between two codes; a value sitting *on* a code is already
// exactly representable and has no error to spread. So the amplitude ramps to
// zero over the last quarter of a code either side, and is full over the middle
// half where every contour is. That is what makes the dither safe rather than
// merely tidy: the hardware's own encode is an approximation — llvmpipe's is
// good to about a tenth of a code value, not to the ULP — so no fixed margin
// under half a step survives it, while a perturbation of exactly zero does. The
// deadzone has to be wider than that disagreement, which is what a quarter buys
// and an eighth would not.
//
// **The slope.** Encoding, adding and decoding back is the obvious spelling and
// is wrong for the same family of reason: `pow` is the driver's to round, so a
// colour that arrived on a code would come back a part in 10^7 off it. Scaling
// the noise into linear and adding leaves the signal itself untouched.
//
// Per channel and not per pixel: three independent draws read as fine chroma
// noise, one shared draw reads as luminance grain — the same amplitude, and the
// eye is far more sensitive to the second. The seed is the pixel alone, so a
// still frame is still: this adds no shimmer to a static camera and nothing here
// is a function of time.
float3 dither(float3 colour, float2 pixel)
{
    if (push.dither <= 0.0)
    {
        return colour;
    }
    uint2 p = uint2(pixel);
    uint seed = p.x * 73856093u ^ p.y * 19349663u;
    float3 n = float3(noise(seed), noise(seed + 1u), noise(seed + 2u));
    float3 encoded = linear_to_srgb(colour);
    float3 f = frac(encoded * 255.0);
    float3 window = saturate(min(f, 1.0 - f) * (1.0 / DITHER_DEADZONE));
    return colour +
           n * window * push.dither * (1.0 / 255.0) * code_value(colour, encoded);
}


// The three output contracts, mirroring `gg_rhi::Output`. Which one is live is
// the *swapchain's* answer and not a preference: `r.hdr` is written back to
// whatever the display granted before this ever reads it.
static const uint OUTPUT_SDR = 0;
static const uint OUTPUT_HDR10 = 1;
static const uint OUTPUT_SCRGB = 2;

// scRGB's definition: 1.0 is 80 nits. Not a knob and not a guess — it is what
// `EXTENDED_SRGB_LINEAR_EXT` means, and the reason that path needs no transfer
// function at all, only a scale.
static const float SCRGB_NITS = 80.0;

// Rec.709 -> Rec.2020 primaries. Both are D65, so this is a pure primary
// conversion with no chromatic adaptation in it. HDR10 is defined on Rec.2020
// and a frame handed over without this arrives oversaturated — the most common
// way an HDR path looks "better" while being wrong.
static const float3x3 REC709_TO_REC2020 = float3x3(0.6274039, 0.3292830, 0.0433131,
                                                   0.0690970, 0.9195404, 0.0113626,
                                                   0.0163914, 0.0880133, 0.8955953);

// ST.2084 (PQ) inverse EOTF: absolute luminance in nits to a [0, 1] code value.
//
// Absolute is the word that matters. Unlike sRGB, a PQ code names a luminance
// rather than a fraction of whatever the panel can do, which is why this path
// needs `r.paper_white` at all and why getting that number wrong is a picture
// that is uniformly too bright rather than one that is mis-shaped.
//
// The five constants are the standard's, as rationals rather than decimals so a
// reader can check them against the text. Anchors, for the same reason: 0 nits
// encodes to 0.0, 100 to 0.5081, 1000 to 0.7518, and 10000 to exactly 1.0.
float3 pq_encode(float3 nits)
{
    const float m1 = 2610.0 / 16384.0;
    const float m2 = 128.0 * 2523.0 / 4096.0;
    const float c1 = 3424.0 / 4096.0;
    const float c2 = 32.0 * 2413.0 / 4096.0;
    const float c3 = 32.0 * 2392.0 / 4096.0;
    float3 y = pow(max(nits, 0.0) / 10000.0, m1);
    return pow((c1 + c2 * y) / (1.0 + c3 * y), m2);
}

// Dither in the *encoded* domain, for a quantizer that takes `levels` steps.
//
// The SDR path cannot use this and does not: there the shader writes linear
// values into an sRGB-format attachment and the *hardware* encodes, so the noise
// has to be spent through the decode's slope before the write (`dither` above).
// Here the shader owns the encode, so the noise goes in where the quantizer is
// and the whole slope argument evaporates. Two dithers because there are
// genuinely two quantizers in two places, not because one of them is wrong.
//
// The deadzone is the same idea and is here for a sharper reason: PQ is
// absolute, so pure black is an exact code on every display, and a half-step of
// noise on it is a grey floor in a dark room.
float3 dither_code(float3 encoded, float2 pixel, float levels)
{
    if (push.dither <= 0.0)
    {
        return encoded;
    }
    uint2 p = uint2(pixel);
    uint seed = p.x * 73856093u ^ p.y * 19349663u;
    float3 n = float3(noise(seed), noise(seed + 1u), noise(seed + 2u));
    float3 f = frac(encoded * levels);
    float3 window = saturate(min(f, 1.0 - f) * (1.0 / DITHER_DEADZONE));
    return encoded + n * window * push.dither * (1.0 / levels);
}

// The finished picture in whatever the swapchain agreed to take.
//
// Ten-bit is where the dither goes on the HDR path: scRGB is `Rgba16F` and has
// no quantizer worth the noise, and HDR10 has four times SDR's code count over a
// range that is fifty times wider — so per code value it is *coarser* where it
// matters, and PQ's whole design is to put those codes where an eye can see the
// steps.
float4 encode(float3 colour, float2 pixel)
{
    if (push.output == OUTPUT_SCRGB)
    {
        return float4(colour * (push.paper_white / SCRGB_NITS), 1.0);
    }
    if (push.output == OUTPUT_HDR10)
    {
        float3 wide = mul(REC709_TO_REC2020, colour * push.paper_white);
        return float4(dither_code(pq_encode(wide), pixel, 1023.0), 1.0);
    }
    // SDR: linear out, and the attachment's own sRGB format does the OETF on
    // write. Untouched by this milestone on purpose — it is the path every
    // blessed golden (§4.10) holds.
    return float4(dither(colour, pixel), 1.0);
}

[shader("fragment")]
float4 fs_main(VOut i) : SV_Target
{
    float3 colour = tap(i.uv);
    if (push.aa != 0)
    {
        colour = fxaa(i.uv, colour);
    }
    // Last, after the edge pass: dither is for the quantizer immediately after
    // this write, and anything that filtered it afterwards would average the
    // noise back out and the banding back in.
    return encode(colour, i.pos.xy);
}

// The overlay's luminance histogram (§6 M11's exit row), first half: resample
// the scene attachment into a small float target the CPU can afford to read
// back and bin.
//
// A *sample* and not an average, and the difference is worth stating: with a
// linear sampler each output texel is the bilinear blend of four source texels,
// not the mean of the whole region it stands for. That makes the histogram a
// stratified sample of the frame — which is what a histogram is for — rather
// than an exact distribution, and pretending otherwise would take a mip chain
// this attachment does not have.
//
// Untonemapped on purpose: the point of the row is what the *scene* is doing, so
// that an operator can pick an exposure. A histogram of the tonemapped image
// would show the curve's output and always look reasonable.
[shader("fragment")]
float4 fs_downsample(VOut i) : SV_Target
{
    return float4(g_textures[push.texture_index].Sample(g_samplers[push.sampler_index], i.uv).rgb,
                  1.0);
}
