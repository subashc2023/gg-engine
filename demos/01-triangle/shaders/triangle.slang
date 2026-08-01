// Demo 01 — the Golden Triangle (§6 M2). The vertex colors below are the
// hot-reloadable bit: edit them while the demo runs and the new triangle is
// on screen in under 500 ms (§4.4). A broken edit keeps the last-good
// pipeline and prints this file's compile errors instead.

struct TrianglePush
{
    float4x4 transform; // aspect correction (and whatever the host dreams up)
    float4 tint;        // host-side color control; white leaves colors as-is
}

[[vk::push_constant]]
ConstantBuffer<TrianglePush> push;

struct VOut
{
    float4 pos : SV_Position;
    float3 color : COLOR0;
}

[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    static const float2 positions[3] = {
        float2(0.0, -0.6),
        float2(0.55, 0.5),
        float2(-0.55, 0.5),
    };
    // Hot-reload me: try swapping these, or making the triangle gold.
    static const float3 colors[3] = {
        float3(1.0, 0.2, 0.1),
        float3(0.1, 1.0, 0.2),
        float3(0.1, 0.2, 1.0),
    };
    VOut o;
    o.pos = mul(push.transform, float4(positions[vid], 0.0, 1.0));
    o.color = colors[vid] * push.tint.rgb;
    return o;
}

[shader("fragment")]
float4 fs_main(VOut i) : SV_Target
{
    return float4(i.color, 1.0);
}
