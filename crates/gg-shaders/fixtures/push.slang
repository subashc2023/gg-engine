// Reflection→codegen test fixture: a push-constant block chosen to exercise
// the layout machinery — a matrix, a vec3 whose successor packs into its tail,
// a scalar, and a vec2 that forces interior padding. The unit tests assert the
// reflected offsets and the generated assertions against hand-computed std430.

struct PushFixture
{
    float4x4 transform; // offset 0, size 64
    float3 tint;        // offset 64, size 12
    float exposure;     // offset 76 (packs into tint's tail)
    float2 viewport;    // offset 80
    uint frame;         // offset 88
}                       // size 92 -> padded to 96 (align 16)

[[vk::push_constant]]
ConstantBuffer<PushFixture> push;

struct VOut
{
    float4 pos : SV_Position;
    float3 color : COLOR0;
}

[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    static const float2 verts[3] = {
        float2(0.0, 0.5),
        float2(-0.5, -0.5),
        float2(0.5, -0.5),
    };
    VOut o;
    o.pos = mul(push.transform, float4(verts[vid] * push.viewport, 0.0, 1.0));
    o.color = push.tint * push.exposure * float(push.frame % 2 + 1);
    return o;
}

[shader("fragment")]
float4 fs_main(VOut i) : SV_Target
{
    return float4(i.color, 1.0);
}
