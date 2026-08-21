// The frame taken apart (§6 M59): one intermediate, shown instead of the
// picture.
//
// Every pass in this graph writes an image, and until this shader existed
// exactly one of them was ever visible — the tonemapped scene. A question about
// any of the others ("what does the AO pass think this wall is?", "is the third
// cascade covering anything?") could only be answered by an offscreen instrument
// rendering its own scene, which is a different scene from the one being
// complained about.
//
// It is a *blit* and not a second renderer: the source is already in the
// bindless array because a pass wrote it, and the only work here is deciding
// what a texel means. That is why the mode is a push constant rather than five
// pipelines — the branch is uniform across the draw and costs nothing a
// fullscreen triangle notices.

#include "include/dither.slang"
// Not a transcription of the occlusion pass any more (§6 M86): MODE_NORMAL is a
// picture of what that pass reconstructs, so the reconstruction is one function
// and `PROJECTION_*` is one pair of constants.
#include "include/depth.slang"

// The global set (§4.3). Binding 1 (storage images) has no consumer here.
[[vk::binding(0, 0)]] Texture2D<float4> g_textures[];
[[vk::binding(2, 0)]] SamplerState g_samplers[];

// How to read a texel of `source_index`.
static const uint MODE_SCENE = 0u;  // linear radiance, exposed, no curve
static const uint MODE_DEPTH = 1u;  // reverse-Z depth as metres, `scale` white
static const uint MODE_NORMAL = 2u; // the depth buffer's own normal, as colour
static const uint MODE_RED = 3u;    // `.r * scale` as grey — occlusion, a map
static const uint MODE_RGB = 4u;    // `.rgb * scale` — a field's record

struct DebugPush
{
    uint source_index;   // what is being shown, in g_textures
    uint sampler_index;  // NearestClamp — a debug view must not invent a texel
    uint mode;           // MODE_*
    uint projection;     // PROJECTION_*, for the two depth-derived modes
    float2 texel;        // 1/extent of the *source*, uv units
    float2 project;      // `ao.slang`'s, and derived by the same host code
    float2 depth_params; // as `project`
    float2 fit_scale;    // screen uv -> source uv, affine; (1,1) fills (§6 M62)
    float2 fit_bias;     // as `fit_scale`
    float2 fit_limit;    // the written corner of the source, in its own uv
    uint2 group;         // (keep, stride) source columns; (1,1) shows them all
    float scale;         // metres of white under MODE_DEPTH, a gain otherwise
    float exposure;      // stops, MODE_SCENE only
    float dither;        // `r.dither`, in output code values; 0 is off (§6 M62)
}

[[vk::push_constant]]
ConstantBuffer<DebugPush> push;

struct VOut
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
}

// `post.slang`'s fullscreen triangle, unchanged.
[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.uv = uv;
    o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

float4 fetch(float2 uv)
{
    return g_textures[push.source_index].SampleLevel(g_samplers[push.sampler_index], uv, 0);
}

// The source read as a depth buffer, for the two modes that reconstruct from it.
// `source_index` is the depth image in exactly those two cases, which is what
// makes one `DepthView` serve a blit that mostly shows something else.
DepthView depth_view()
{
    DepthView v;
    v.image = g_textures[push.source_index];
    v.samp = g_samplers[push.sampler_index];
    v.projection = push.projection;
    v.texel = push.texel;
    v.project = push.project;
    v.depth_params = push.depth_params;
    return v;
}

// Outside the source: what a letterboxed atlas sits on. A mid grey rather than
// black, because half these views are of images whose *content* is black in
// places and a margin that agrees with them says nothing.
static const float3 LETTERBOX = float3(0.06, 0.06, 0.07);

// Skip the columns of a record that are not what the view is *of* (§6 M62).
//
// The irradiance image stores four texels per probe and only three of them are
// radiance; the fourth is that probe's state, whose red channel is a distance in
// metres. Shown as colour at a gain meant for irradiance it is a saturated red
// bar every fourth column, which is a quarter of the view and is the first thing
// an eye lands on. `group` is `(keep, stride)` — `(3, 4)` here, `(1, 1)`
// everywhere else, where this whole function is skipped and the uv is untouched.
//
// The host has already sized the fit to the *kept* columns, so `uv.x` walks the
// shown ones and this expands each back to where it lives.
float2 regroup(float2 uv)
{
    float column = uv.x / max(push.texel.x, 1e-12);
    // Clamped, because the letterbox test admits `uv == fit_limit` — it has to,
    // or every full-screen view loses its last column to the margin — and one
    // pixel past the last kept column would expand to a texel nothing wrote.
    uint columns = uint(push.fit_limit.x / max(push.texel.x, 1e-12) + 0.5);
    uint index = min(uint(column), columns - 1u);
    uint source = (index / push.group.x) * push.group.y + (index % push.group.x);
    return float2((float(source) + frac(column)) * push.texel.x, uv.y);
}

// What this view shows, linear and already in 0..1.
float3 shown(VOut i, float2 uv)
{
    float4 texel = fetch(uv);
    float gain = max(push.scale, 1e-6);

    if (push.mode == MODE_SCENE)
    {
        // No curve on purpose: the question this view answers is what the
        // forward pass *wrote*, and a tonemapper is the thing standing between
        // that and every other view of it. Values above one clip to white, which
        // is the honest reading of "this pixel is over-bright".
        return saturate(texel.rgb * exp2(push.exposure) * gain);
    }
    if (push.mode == MODE_RGB)
    {
        return saturate(texel.rgb * gain);
    }
    if (push.mode == MODE_RED)
    {
        return saturate(texel.rrr * gain);
    }

    // Both remaining modes read the camera's depth, where reverse-Z clears to
    // zero and nothing renders at the far plane — so a zero is background and is
    // the one value neither may reconstruct a position from.
    float d = texel.r;
    if (d <= 0.0)
    {
        return float3(0.0);
    }
    float3 p = view_position(depth_view(), uv, d);
    if (push.mode == MODE_NORMAL)
    {
        // Half-and-half, the usual encoding: a normal is signed and the target
        // is not, and a view where "up" is a recognisable colour is worth more
        // than one where half the range is clipped to black.
        return reconstruct_normal(depth_view(), uv, p) * 0.5 + 0.5;
    }
    // MODE_DEPTH. Distance along the view axis rather than to the eye: it is
    // what the reconstruction produces and what every consumer of this buffer
    // reasons in, and a radial distance would disagree with all of them at the
    // corners.
    return saturate(-p.z / gain).xxx;
}

// The target is sRGB and the hardware encodes on write, so everything returned
// here is linear — a debug view that applied a gamma by hand would be the
// washed-out frame `post.slang` warns about, one pass later.
//
// **The source is fitted rather than stretched** (§6 M62), which for the views
// that fill the screen is the identity and for the four that do not is the
// difference between a picture and a smear. Two things were wrong and the same
// affine map fixes both: the two field images are allocated at the largest grid
// the renderer will ever build and written only as far as *this* scene's grid
// reaches, so a straight blit showed the unwritten remainder as a black margin
// and called it data; and an atlas is not the screen's shape, so a 28x48 record
// across 16:9 had texels eight times wider than tall. `fit_scale` is (1,1) and
// `fit_bias` (0,0) whenever the source is the view's own size, which is exactly
// when `i.uv` passes through untouched.
//
// **Dithered like the picture** (§6 M62). This pass is a blit onto the same
// 8-bit swapchain and had no dither of its own, so any view holding a smooth
// ramp arrived as contours: depth over fifty metres of white is a fifth of a
// metre per code, which on a floor receding away from the eye is bands metres
// wide. The banding is the *output's* and not the source's — the depth buffer
// is `D32_SFLOAT` and the occlusion target a half float — which is exactly the
// case `r.dither` exists for and is why this is the same function `post.slang`
// calls rather than a second one sized by eye.
[shader("fragment")]
float4 fs_main(VOut i) : SV_Target
{
    float2 uv = i.uv * push.fit_scale + push.fit_bias;
    if (any(uv < 0.0) || any(uv > push.fit_limit))
    {
        return float4(dither_linear(LETTERBOX, i.pos.xy, push.dither), 1.0);
    }
    if (push.group.y > push.group.x)
    {
        uv = regroup(uv);
    }
    return float4(dither_linear(shown(i, uv), i.pos.xy, push.dither), 1.0);
}
