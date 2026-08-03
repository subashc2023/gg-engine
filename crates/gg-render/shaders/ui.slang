// The UI pass (§4.9's draw layer, seeded by M8's overlay — §4.8). One pipeline
// for every rectangle and every glyph the frame draws: a glyph quad is a
// rectangle whose uv lands on a coverage atlas, and a solid rectangle is one
// whose uv lands on a texel that is always 1. No branch, no second pipeline,
// and one draw call for the whole layer.
//
// Vertices are *pulled* through a buffer device address (§4.3); nothing binds a
// vertex buffer anywhere in this engine.

// The global set (§4.3). Binding 1 (storage images) has no consumer here.
[[vk::binding(0, 0)]] Texture2D<float4> g_textures[];
[[vk::binding(2, 0)]] SamplerState g_samplers[];

struct UiPush
{
    uint64_t vertices;  // device address of this frame's vertex array
    uint texture_index; // the coverage atlas's slot in g_textures
    uint sampler_index; // NearestClamp: the atlas is authored at final size
    float2 inv_extent;  // 1 / target size — pixels to [0,1]
}

[[vk::push_constant]]
ConstantBuffer<UiPush> push;

// HLSL has no `sizeof`; the stride is written here and asserted by the Rust
// `UiVertex` beside it, so a drift is a build error rather than garbled quads.
static const uint UI_VERTEX_STRIDE = 20;

struct VOut
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
}

// `0xAARRGGBB`, the same byte order the render protocol's `Renderable` uses so
// one colour constant means one thing engine-wide. rgb is sRGB, linearized by
// the exact EOTF — the same curve `srgb_to_linear` tabulates CPU-side, evaluated
// here because the GPU is not on the determinism membrane (§1.4) and `pow` is
// free to use. Alpha is coverage and is already linear.
float3 srgb_to_linear(float3 c)
{
    return select(c <= 0.04045, c / 12.92, pow((c + 0.055) / 1.055, 2.4));
}

float4 unpack(uint c)
{
    float3 rgb = float3((c >> 16) & 0xffu, (c >> 8) & 0xffu, c & 0xffu) / 255.0;
    return float4(srgb_to_linear(rgb), float((c >> 24) & 0xffu) / 255.0);
}

[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    // A `float*` rather than a `UiVertex*`: the pointee's layout would then be
    // Slang's to choose while the Rust side is packed `repr(C)`, and scalar
    // strides are the one indexing both agree on.
    float* v = (float*)(push.vertices + uint64_t(vid) * UI_VERTEX_STRIDE);
    VOut o;
    // Pixels with the origin top-left. The viewport is not Y-flipped, so clip
    // -1 already *is* the top row and the mapping is the plain one.
    o.pos = float4(float2(v[0], v[1]) * push.inv_extent * 2.0 - 1.0, 0.0, 1.0);
    o.uv = float2(v[2], v[3]);
    o.color = unpack(asuint(v[4]));
    return o;
}

[shader("fragment")]
float4 fs_main(VOut i) : SV_Target
{
    float coverage = g_textures[push.texture_index].Sample(g_samplers[push.sampler_index], i.uv).r;
    return float4(i.color.rgb, i.color.a * coverage);
}
