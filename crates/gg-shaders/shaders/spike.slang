// Spike 1 artifact (PLAN.md §6, M0A): a trivial-but-reflective module. Two entry
// points and a uniform block, so the test exercises compilation AND reflection —
// the two capabilities the shader-slang bindings must prove before M2 commits
// to the in-process path.

struct Uniforms
{
    float4x4 mvp;
    float4 tint;
}

ConstantBuffer<Uniforms> uniforms;

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
    o.pos = mul(uniforms.mvp, float4(verts[vid], 0.0, 1.0));
    o.color = uniforms.tint.rgb;
    return o;
}

[shader("fragment")]
float4 fs_main(VOut i) : SV_Target
{
    return float4(i.color, 1.0);
}
