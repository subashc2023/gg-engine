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
//
// # Instanced since M10
//
// One draw covers a run of instances that share a mesh, so what varies per
// instance — the matrix, the tint, the orientation — moved out of the push
// constants and into a device-address array indexed by `SV_InstanceID`. What
// stays in the push block is what a *batch* shares: the mesh, the material's
// two indices, and where the batch starts in that array.
//
// The matrix is still the finished object-to-clip product, computed on the CPU
// and stored per instance rather than assembled here from a view-projection and
// a model. That is deliberate and not laziness: multiplying in a different place
// changes the low bits, and every pixel of the blessed `hall` reference (§4.10)
// depends on it not changing.

// The global set (§4.3). Binding 1 (storage images) has no consumer here.
[[vk::binding(0, 0)]] Texture2D<float4> g_textures[];
[[vk::binding(2, 0)]] SamplerState g_samplers[];

struct ScenePush
{
    uint64_t instances;  // device address of the instance array
    uint64_t vertices;   // device address of this batch's mesh vertices
    uint instance_base;  // index of this batch's first instance
    uint texture_index;  // base colour's slot in g_textures; a 1x1 white texel if the
                         // material names none, so the fragment path has no branch
    uint sampler_index;  // LinearRepeat: a mip chain exists and uvs may leave [0,1]
    uint reserved;
}

[[vk::push_constant]]
ConstantBuffer<ScenePush> push;

// HLSL has no `sizeof`, so the strides are written here and asserted against the
// Rust types. A drift is a build error, not a mesh that reads its own normals as
// positions.
static const uint VERTEX_STRIDE = 32;
static const uint INSTANCE_STRIDE = 96;

// See `ugly.slang`: positions carry scale through the matrix, which non-uniform
// scale would shear a normal by, so normals take the rotation on its own.
float3 rotate_by(float4 q, float3 v)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// One instance's record. Read as scalars for the same reason vertices are: a
// `float4x4*` would let Slang pick the layout while the Rust side is packed
// `repr(C)`, and 16 consecutive floats are what both agree on. The constructor
// fills row-major, which is the order `gg_math::render::rows` writes.
struct Instance
{
    float4x4 mvp;
    float4 tint;
    float4 rotation;
}

Instance load_instance(uint id)
{
    float* f = (float*)(push.instances + uint64_t(push.instance_base + id) * INSTANCE_STRIDE);
    Instance i;
    i.mvp = float4x4(f[0], f[1], f[2], f[3],
                     f[4], f[5], f[6], f[7],
                     f[8], f[9], f[10], f[11],
                     f[12], f[13], f[14], f[15]);
    i.tint = float4(f[16], f[17], f[18], f[19]);
    i.rotation = float4(f[20], f[21], f[22], f[23]);
    return i;
}

struct VOut
{
    float4 pos : SV_Position;
    float3 normal : NORMAL0;
    float2 uv : TEXCOORD0;
    float3 tint : COLOR0;
}

[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID, uint iid: SV_InstanceID)
{
    // A `float*` rather than a `Vertex*`: the pointee's layout would then be
    // Slang's to choose (float3 aligns to 16 in std430) while the pack's is
    // packed `repr(C)`. Scalar strides are the one indexing both agree on.
    float* v = (float*)(push.vertices + uint64_t(vid) * VERTEX_STRIDE);
    Instance inst = load_instance(iid);
    VOut o;
    o.pos = mul(inst.mvp, float4(v[0], v[1], v[2], 1.0));
    o.normal = rotate_by(inst.rotation, float3(v[3], v[4], v[5]));
    o.uv = float2(v[6], v[7]);
    o.tint = inst.tint.rgb;
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
    return float4(base * i.tint * lambert, 1.0);
}

// The depth prepass (§4.5 v1). It shares `ScenePush` with the pass above rather
// than declaring a smaller block, because the forward pass must re-draw
// *exactly* where this one put a fragment — a second matrix path is a second
// chance to disagree, and `GREATER_OR_EQUAL` only forgives bit-identical depth.
// Same argument covers `load_instance`: one loader, not two.
[shader("vertex")]
float4 vs_depth(uint vid: SV_VertexID, uint iid: SV_InstanceID) : SV_Position
{
    float* v = (float*)(push.vertices + uint64_t(vid) * VERTEX_STRIDE);
    return mul(load_instance(iid).mvp, float4(v[0], v[1], v[2], 1.0));
}

[shader("fragment")]
void fs_depth()
{
}
