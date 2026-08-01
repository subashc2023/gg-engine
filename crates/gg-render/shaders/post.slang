// The §4.5 v1 fullscreen post pass. What it does today is resolve the scene
// attachment onto whatever the frame lands in — one texture read per pixel and
// nothing else. That is deliberate: the pass exists so the *graph* has a
// consumer of a colour attachment written by another pass (the transition the
// derivation has to get right), and so tone mapping has somewhere to land at
// M11 without a new pass. An invented look here would be a look M11 deletes.

// The global set (§4.3). Binding 1 (storage images) has no consumer here.
[[vk::binding(0, 0)]] Texture2D<float4> g_textures[];
[[vk::binding(2, 0)]] SamplerState g_samplers[];

struct PostPush
{
    uint texture_index; // the scene attachment's slot in g_textures
    uint sampler_index; // NearestClamp: this is a resolve, not a filter
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

[shader("fragment")]
float4 fs_main(VOut i) : SV_Target
{
    return float4(g_textures[push.texture_index].Sample(g_samplers[push.sampler_index], i.uv).rgb,
                  1.0);
}
