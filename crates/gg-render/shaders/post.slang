// The §4.5 v1 fullscreen post pass, which since M11 is the tonemapper: exposure,
// then a curve, from the `Rgba16F` scene attachment onto whatever the frame lands
// in. The pass already existed for the graph's sake — a consumer of a colour
// attachment another pass wrote — and this is the work §6 M11 said would land in
// it without a new pass.
//
// The target is sRGB and Vulkan encodes on write, so everything here is linear
// and nothing applies a gamma by hand. A second encode is the classic washed-out
// frame.

// The global set (§4.3). Binding 1 (storage images) has no consumer here.
[[vk::binding(0, 0)]] Texture2D<float4> g_textures[];
[[vk::binding(2, 0)]] SamplerState g_samplers[];

struct PostPush
{
    uint texture_index; // the scene attachment's slot in g_textures
    uint sampler_index; // NearestClamp: this is a resolve, not a filter
    float exposure;     // stops, so 0 is unchanged and +1 is twice the light
    uint reserved;
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

[shader("fragment")]
float4 fs_main(VOut i) : SV_Target
{
    float3 radiance =
        g_textures[push.texture_index].Sample(g_samplers[push.sampler_index], i.uv).rgb;
    // Exposure in stops, applied *before* the curve: it is the choice of what
    // counts as "1.0", and applying it after would be adjusting a picture that
    // had already been compressed.
    return float4(pbr_neutral(radiance * exp2(push.exposure)), 1.0);
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
