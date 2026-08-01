// Demo 02 — Real Geometry (§6 M4A). Three M4A mechanisms meet in one shader:
// vertices are *pulled* through a buffer device address (nothing binds a
// vertex buffer — §4.3), the texture is reached by index into the one global
// bindless array, and the pipeline supplies reverse-Z depth.

// The global set (§4.3). Binding numbers match gg-rhi's `bindless.rs`; the
// arrays are unbounded and partially bound, so declaring them whole costs
// nothing per unused slot. Binding 1 (storage images) has no consumer here.
[[vk::binding(0, 0)]] Texture2D<float4> g_textures[];
[[vk::binding(2, 0)]] SamplerState g_samplers[];

struct MeshPush
{
    float4x4 view_projection; // world -> clip, reverse-Z infinite far
    uint64_t vertices;        // device address of the vertex array
    uint texture_index;       // slot in g_textures
    uint sampler_index;       // slot in g_samplers
}

[[vk::push_constant]]
ConstantBuffer<MeshPush> push;

// One vertex's byte layout. HLSL has no `sizeof`, so the stride is written
// here and mirrored by demo 02's Rust `Vertex` — which asserts it, making a
// drift a build error rather than a garbled mesh.
static const uint VERTEX_STRIDE = 32;

struct VOut
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
    float3 normal : NORMAL0;
}

[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    // A `float*` rather than a `Vertex*`: the pointee's layout would then be
    // Slang's to choose (float3 aligns to 16 in std430), and the Rust side is
    // packed `repr(C)`. Scalar strides are the one indexing both agree on.
    float* v = (float*)(push.vertices + uint64_t(vid) * VERTEX_STRIDE);
    float3 position = float3(v[0], v[1], v[2]);
    float3 normal = float3(v[3], v[4], v[5]);
    float2 uv = float2(v[6], v[7]);

    VOut o;
    o.pos = mul(push.view_projection, float4(position, 1.0));
    o.uv = uv;
    o.normal = normal;
    return o;
}

[shader("fragment")]
float4 fs_main(VOut i) : SV_Target
{
    float4 albedo = g_textures[push.texture_index].Sample(g_samplers[push.sampler_index], i.uv);
    // A fixed directional term, not lighting: enough to tell the cube's faces
    // apart in a golden image. Real lights are M11's job.
    float3 light = normalize(float3(0.35, 0.85, 0.4));
    float lambert = saturate(dot(normalize(i.normal), light)) * 0.75 + 0.25;
    return float4(albedo.rgb * lambert, 1.0);
}
