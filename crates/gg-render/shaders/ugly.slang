// The §4.5 v0 pass: one box, lit by one hardcoded direction, tinted by whatever
// colour the game put in its `Renderable` (§6 M5). No textures, no materials,
// no lights — M9 and M11 own those, and a placeholder that pretended to would
// be the thing they have to delete first.
//
// Vertices are *pulled* through a buffer device address (§4.3): nothing binds a
// vertex buffer anywhere in this engine.

struct UglyPush
{
    float4x4 mvp;      // object -> clip, camera-relative, reverse-Z infinite far
    uint64_t vertices; // device address of the vertex array
    float4 tint;       // linear rgb; w unused
    float4 rotation;   // the instance's orientation, xyzw — normals only
}

[[vk::push_constant]]
ConstantBuffer<UglyPush> push;

// HLSL has no `sizeof`, so the stride is written here and asserted by the Rust
// `Vertex` beside it. A drift is a build error, not a garbled mesh.
static const uint VERTEX_STRIDE = 24;

// Positions carry the instance's scale through `mvp`, which non-uniform scale
// would shear a normal by; the rotation is applied to normals separately for
// that reason. A box's face normals are unaffected by axis-aligned scale, which
// is what makes the shortcut exact here rather than merely cheap.
float3 rotate_by(float4 q, float3 v)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

struct VOut
{
    float4 pos : SV_Position;
    float3 normal : NORMAL0;
}

[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    // A `float*` rather than a `Vertex*`: the pointee's layout would then be
    // Slang's to choose (float3 aligns to 16 in std430) while the Rust side is
    // packed `repr(C)`. Scalar strides are the one indexing both agree on.
    float* v = (float*)(push.vertices + uint64_t(vid) * VERTEX_STRIDE);
    VOut o;
    o.pos = mul(push.mvp, float4(v[0], v[1], v[2], 1.0));
    o.normal = rotate_by(push.rotation, float3(v[3], v[4], v[5]));
    return o;
}

[shader("fragment")]
float4 fs_main(VOut i) : SV_Target
{
    float3 light = normalize(float3(0.35, 0.85, 0.4));
    float lambert = saturate(dot(normalize(i.normal), light)) * 0.7 + 0.3;
    return float4(push.tint.rgb * lambert, 1.0);
}
