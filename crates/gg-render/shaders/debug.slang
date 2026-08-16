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

// The global set (§4.3). Binding 1 (storage images) has no consumer here.
[[vk::binding(0, 0)]] Texture2D<float4> g_textures[];
[[vk::binding(2, 0)]] SamplerState g_samplers[];

// `ao.slang`'s two, and they mean the same thing — this shader reconstructs
// position exactly as the occlusion pass does, because half of what it is for is
// showing what that pass saw.
static const uint PROJECTION_PERSPECTIVE = 0u;
static const uint PROJECTION_ORTHOGRAPHIC = 1u;

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
    float scale;         // metres of white under MODE_DEPTH, a gain otherwise
    float exposure;      // stops, MODE_SCENE only
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

// `ao.slang`'s, transcribed rather than shared: an include would put the whole
// occlusion pass in this translation unit for four lines, and the two are
// checked against each other by the fact that MODE_NORMAL is *read* — a
// reconstruction that drifted from the one AO marches against would show it.
float3 view_position(float2 uv, float d)
{
    float2 ndc = uv * 2.0 - 1.0;
    if (push.projection == PROJECTION_ORTHOGRAPHIC)
    {
        return float3(ndc * push.project, d * push.depth_params.x + push.depth_params.y);
    }
    float linear_depth = push.depth_params.x / max(d, 1e-9);
    return float3(ndc * push.project * linear_depth, -linear_depth);
}

float3 position_at(float2 uv)
{
    return view_position(uv, fetch(uv).r);
}

float3 reconstruct_normal(float2 uv, float3 p)
{
    float3 right = position_at(uv + float2(push.texel.x, 0.0));
    float3 left = position_at(uv - float2(push.texel.x, 0.0));
    float3 down = position_at(uv + float2(0.0, push.texel.y));
    float3 up = position_at(uv - float2(0.0, push.texel.y));
    float3 dx = abs(right.z - p.z) < abs(p.z - left.z) ? right - p : p - left;
    float3 dy = abs(down.z - p.z) < abs(p.z - up.z) ? down - p : p - up;
    float3 n = cross(dy, dx);
    float len = length(n);
    if (len < 1e-12)
    {
        return float3(0.0, 0.0, 1.0);
    }
    n /= len;
    float3 to_eye = push.projection == PROJECTION_ORTHOGRAPHIC ? float3(0.0, 0.0, 1.0)
                                                               : normalize(-p);
    return dot(n, to_eye) < 0.0 ? -n : n;
}

// The target is sRGB and the hardware encodes on write, so everything returned
// here is linear — a debug view that applied a gamma by hand would be the
// washed-out frame `post.slang` warns about, one pass later.
[shader("fragment")]
float4 fs_main(VOut i) : SV_Target
{
    float4 texel = fetch(i.uv);
    float gain = max(push.scale, 1e-6);

    if (push.mode == MODE_SCENE)
    {
        // No curve on purpose: the question this view answers is what the
        // forward pass *wrote*, and a tonemapper is the thing standing between
        // that and every other view of it. Values above one clip to white, which
        // is the honest reading of "this pixel is over-bright".
        return float4(saturate(texel.rgb * exp2(push.exposure) * gain), 1.0);
    }
    if (push.mode == MODE_RGB)
    {
        return float4(saturate(texel.rgb * gain), 1.0);
    }
    if (push.mode == MODE_RED)
    {
        return float4(saturate(texel.rrr * gain), 1.0);
    }

    // Both remaining modes read the camera's depth, where reverse-Z clears to
    // zero and nothing renders at the far plane — so a zero is background and is
    // the one value neither may reconstruct a position from.
    float d = texel.r;
    if (d <= 0.0)
    {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    float3 p = view_position(i.uv, d);
    if (push.mode == MODE_NORMAL)
    {
        // Half-and-half, the usual encoding: a normal is signed and the target
        // is not, and a view where "up" is a recognisable colour is worth more
        // than one where half the range is clipped to black.
        return float4(reconstruct_normal(i.uv, p) * 0.5 + 0.5, 1.0);
    }
    // MODE_DEPTH. Distance along the view axis rather than to the eye: it is
    // what the reconstruction produces and what every consumer of this buffer
    // reasons in, and a radial distance would disagree with all of them at the
    // corners.
    return float4(saturate(-p.z / gain).xxx, 1.0);
}
