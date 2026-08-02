// The §4.6 pass: geometry and materials out of a pack, rather than the one box
// `ugly.slang` draws. Same light, same reverse-Z, same pulled vertices — what is
// new here is that the stride is a *file's* and the colour comes from a texture.
//
// Base colour only. The pack also carries normal, metallic-roughness and
// occlusion maps and this samples none of them: they are lighting inputs and
// lighting is M11's, so a shading model invented here would be the first thing
// M11 deletes. The maps are in the file and unread, which is the honest state.
//
// Vertices are *pulled* through a buffer device address (§4.3); nothing binds a
// vertex buffer anywhere in this engine.

// The global set (§4.3). Binding 1 (storage images) has no consumer here.
[[vk::binding(0, 0)]] Texture2D<float4> g_textures[];
[[vk::binding(2, 0)]] SamplerState g_samplers[];

struct ScenePush
{
    float4x4 mvp;       // object -> clip, camera-relative, reverse-Z infinite far
    uint64_t vertices;  // device address of the mesh's vertex array
    float4 tint;        // linear rgb: the game's tint times the material's base colour
    float4 rotation;    // the instance's orientation, xyzw — normals only
    uint texture_index; // base colour's slot in g_textures; a 1x1 white texel if the
                        // material names none, so the fragment path has no branch
    uint sampler_index; // LinearRepeat: a mip chain exists and uvs may leave [0,1]
}

[[vk::push_constant]]
ConstantBuffer<ScenePush> push;

// HLSL has no `sizeof`, so the stride is written here and asserted against
// `gg_assets::Vertex` on the Rust side. A drift is a build error, not a mesh
// that reads its own normals as positions.
static const uint VERTEX_STRIDE = 32;

// See `ugly.slang`: positions carry scale through `mvp`, which non-uniform
// scale would shear a normal by, so normals take the rotation on its own.
float3 rotate_by(float4 q, float3 v)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

struct VOut
{
    float4 pos : SV_Position;
    float3 normal : NORMAL0;
    float2 uv : TEXCOORD0;
}

[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    // A `float*` rather than a `Vertex*`: the pointee's layout would then be
    // Slang's to choose (float3 aligns to 16 in std430) while the pack's is
    // packed `repr(C)`. Scalar strides are the one indexing both agree on.
    float* v = (float*)(push.vertices + uint64_t(vid) * VERTEX_STRIDE);
    VOut o;
    o.pos = mul(push.mvp, float4(v[0], v[1], v[2], 1.0));
    o.normal = rotate_by(push.rotation, float3(v[3], v[4], v[5]));
    o.uv = float2(v[6], v[7]);
    return o;
}

[shader("fragment")]
float4 fs_main(VOut i) : SV_Target
{
    float3 light = normalize(float3(0.35, 0.85, 0.4));
    float lambert = saturate(dot(normalize(i.normal), light)) * 0.7 + 0.3;
    // The texture is BC7 sRGB, so the sampler decodes and this arithmetic is
    // linear throughout — the tint arrives linear from the CPU for the same
    // reason (§4.5's `srgb_to_linear`).
    float3 base = g_textures[push.texture_index].Sample(g_samplers[push.sampler_index], i.uv).rgb;
    return float4(base * push.tint.rgb * lambert, 1.0);
}

// The depth prepass (§4.5 v1). It shares `ScenePush` with the pass above rather
// than declaring a smaller block, because the forward pass must re-draw
// *exactly* where this one put a fragment — a second matrix path is a second
// chance to disagree, and `GREATER_OR_EQUAL` only forgives bit-identical depth.
[shader("vertex")]
float4 vs_depth(uint vid: SV_VertexID) : SV_Position
{
    float* v = (float*)(push.vertices + uint64_t(vid) * VERTEX_STRIDE);
    return mul(push.mvp, float4(v[0], v[1], v[2], 1.0));
}

[shader("fragment")]
void fs_depth()
{
}
