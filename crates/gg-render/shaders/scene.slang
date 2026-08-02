// The §4.6 pass: geometry and materials out of a pack, shaded by §6 M11's
// metal-rough model. Every equation and its citation is in `include/pbr.slang`;
// what is here is the plumbing — what a batch shares, what an instance carries,
// and which texture goes where.
//
// Vertices are *pulled* through a buffer device address (§4.3); nothing binds a
// vertex buffer anywhere in this engine.
//
// # Three pipelines, one push block
//
// Prepass, shadow and forward all take `ScenePush` rather than each declaring
// the smallest block it needs. The forward pass must re-draw *exactly* where the
// prepass put a fragment — a second matrix path is a second chance to disagree,
// and `GREATER_OR_EQUAL` only forgives a bit-identical depth. The shadow pass
// shares it for the plainer reason that it wants the same instance loader.
//
// The matrix is the finished object-to-clip product, computed on the CPU and
// stored per instance rather than assembled here from a view-projection and a
// model. That is deliberate: multiplying in a different place changes the low
// bits, and blessed references (§4.10) are byte comparisons against pixels that
// depend on them. The camera-relative world position is carried *beside* it as a
// translation and a scale, because lighting needs the position and the clip
// coordinate cannot be inverted back to one.

#include "include/pbr.slang"

// The global set (§4.3). Binding 1 (storage images) has no consumer here.
[[vk::binding(0, 0)]] Texture2D<float4> g_textures[];
[[vk::binding(2, 0)]] SamplerState g_samplers[];

struct ScenePush
{
    uint64_t instances;   // device address of the instance array
    uint64_t vertices;    // device address of this batch's mesh vertices
    uint64_t frame;       // device address of this frame's lighting environment
    uint instance_base;   // index of this batch's first instance
    uint base_color;      // slot in g_textures; a 1x1 white texel if the material names none
    uint normal_map;      // slot in g_textures; a 1x1 flat texel if none
    uint metallic_rough;  // slot in g_textures; roughness in r, metalness in g (§4.6)
    uint occlusion;       // slot in g_textures; occlusion in r
    uint sampler_index;   // LinearRepeat: a mip chain exists and uvs may leave [0,1]
    float metallic;       // factor, multiplied with the map
    float roughness;      // factor, multiplied with the map
}

[[vk::push_constant]]
ConstantBuffer<ScenePush> push;

// HLSL has no `sizeof`, so the strides are written here and asserted against the
// Rust types. A drift is a build error, not a mesh that reads its own normals as
// positions.
static const uint VERTEX_STRIDE = 48;
static const uint INSTANCE_STRIDE = 128;

// Positions carry scale through the matrix, which non-uniform scale would shear
// a normal by, so normals and tangents take the rotation on its own.
float3 rotate_by(float4 q, float3 v)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// One instance's record. Read as scalars for the same reason vertices are: a
// `float4x4*` would let Slang pick the layout while the Rust side is packed
// `repr(C)`, and consecutive floats are what both agree on. The constructor
// fills row-major, which is the order `gg_math::render::rows` writes.
struct Instance
{
    float4x4 mvp;
    float4 tint;
    float4 rotation;
    float3 offset; // camera-relative translation
    float3 scale;  // per-axis, applied before the rotation
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
    i.offset = float3(f[24], f[25], f[26]);
    i.scale = float3(f[28], f[29], f[30]);
    return i;
}

struct VOut
{
    float4 pos : SV_Position;
    float3 world : TEXCOORD1;     // camera-relative, which makes the eye the origin
    float3 normal : NORMAL0;
    float4 tangent : TANGENT0;    // xyz plus the handedness bit
    float2 uv : TEXCOORD0;
    float3 tint : COLOR0;
    float3 light_clip : TEXCOORD2; // already NDC: the sun's projection is orthographic
}

// Where the shadow map is sampled from, offset along the normal.
//
// The normal offset is the half of the acne fix a depth bias cannot do: bias
// moves a fragment along the light's depth axis, and a surface lit edge-on has
// almost no depth gradient to move it through. Pushing the *lookup* off the
// surface instead works at any angle, and the cost of overdoing it is the same
// peter-panning the constant bias produces — which is why both are CVars.
float3 to_light_clip(Frame frame, float3 world, float3 normal)
{
    float4 biased = float4(world + normal * frame.shadow_normal_bias, 1.0);
    return mul(frame.sun_view_projection, biased).xyz;
}

[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID, uint iid: SV_InstanceID)
{
    // A `float*` rather than a `Vertex*`: the pointee's layout would then be
    // Slang's to choose (float3 aligns to 16 in std430) while the pack's is
    // packed `repr(C)`. Scalar strides are the one indexing both agree on.
    float* v = (float*)(push.vertices + uint64_t(vid) * VERTEX_STRIDE);
    Instance inst = load_instance(iid);
    Frame frame = load_frame(push.frame);

    float3 local = float3(v[0], v[1], v[2]);
    float3 normal = rotate_by(inst.rotation, float3(v[3], v[4], v[5]));

    VOut o;
    o.pos = mul(inst.mvp, float4(local, 1.0));
    o.world = inst.offset + rotate_by(inst.rotation, local * inst.scale);
    o.normal = normal;
    o.tangent = float4(rotate_by(inst.rotation, float3(v[8], v[9], v[10])), v[11]);
    o.uv = float2(v[6], v[7]);
    o.tint = inst.tint.rgb;
    o.light_clip = to_light_clip(frame, o.world, normalize(normal));
    return o;
}

[shader("fragment")]
float4 fs_main(VOut i) : SV_Target
{
    Frame frame = load_frame(push.frame);
    SamplerState sampler = g_samplers[push.sampler_index];

    // The base-colour map is BC7 sRGB, so the sampler decodes and everything
    // below is linear. The tint arrives linear from the CPU for the same reason
    // (§4.5's `srgb_to_linear`).
    float3 base = g_textures[push.base_color].Sample(sampler, i.uv).rgb * i.tint;
    // glTF packs roughness in green and metalness in blue; `ggc` repacks the
    // pair into BC5's two channels once at import (§4.6), so this reads r and g
    // and no shader has to remember which convention it is looking at.
    float2 mr = g_textures[push.metallic_rough].Sample(sampler, i.uv).rg;
    float occlusion = g_textures[push.occlusion].Sample(sampler, i.uv).r;
    float2 normal_sample = g_textures[push.normal_map].Sample(sampler, i.uv).rg;

    Surface s = make_surface(apply_normal_map(i.normal, i.tangent, normal_sample), base,
                             mr.g * push.metallic, mr.r * push.roughness, occlusion);
    // Radiance, unbounded above — the scene attachment is `Rgba16F` and the
    // tonemapper is what brings this into [0, 1] (§6 M11).
    return float4(shade_all(push.frame, frame, s, i.world, i.light_clip, g_textures, g_samplers),
                  1.0);
}

// The depth prepass (§4.5 v1) and the shadow pass. Both are position-only, and
// both share `ScenePush` and `load_instance` with the pass above — one loader,
// not three.
[shader("vertex")]
float4 vs_depth(uint vid: SV_VertexID, uint iid: SV_InstanceID) : SV_Position
{
    float* v = (float*)(push.vertices + uint64_t(vid) * VERTEX_STRIDE);
    return mul(load_instance(iid).mvp, float4(v[0], v[1], v[2], 1.0));
}

// The shadow pass reprojects through the sun rather than the camera, which is
// why it cannot reuse `mvp` and rebuilds the world position the way `vs_main`
// does. No normal offset here: the bias belongs on the *lookup*, so that the
// depth this records stays the geometry's own.
[shader("vertex")]
float4 vs_shadow(uint vid: SV_VertexID, uint iid: SV_InstanceID) : SV_Position
{
    float* v = (float*)(push.vertices + uint64_t(vid) * VERTEX_STRIDE);
    Instance inst = load_instance(iid);
    Frame frame = load_frame(push.frame);
    float3 world = inst.offset + rotate_by(inst.rotation, float3(v[0], v[1], v[2]) * inst.scale);
    return mul(frame.sun_view_projection, float4(world, 1.0));
}

[shader("fragment")]
void fs_depth()
{
}
