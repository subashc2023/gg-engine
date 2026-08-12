// The environment, drawn where nothing else was (§6 M24).
//
// Not a separate pass in the graph. It is one more draw at the end of
// `forward-opaque`, over the same attachment and against the same depth buffer,
// because that is all it needs to be: reverse-Z clears depth to 0.0 and compares
// `GREATER_OR_EQUAL`, so a triangle emitted at exactly 0.0 passes precisely
// where nothing has been drawn and fails everywhere else. No stencil, no second
// attachment, no barrier, and the depth test does the masking for free.
//
// Drawn last rather than first, which is the opposite of the intuitive order and
// is what makes it cost almost nothing: after the opaque pass, every pixel the
// scene covers has already failed this draw's depth test before its fragment
// shader would have run.
//
// # Why the ray comes from an inverted matrix
//
// A skybox needs the world-space direction through each pixel, and the frame
// block carries world -> clip. Inverting it *per frame on the host* costs one
// matrix; carrying the camera basis instead would mean this shader rebuilding
// the projection's inverse from a field of view and an aspect ratio it would
// have to be handed anyway, and getting the reverse-Z convention right a second
// time. Under an infinite far plane the clip point (x, y, 0, 1) maps to a
// homogeneous point with w = 0 — a direction exactly, with nothing to divide.

struct SkyPush
{
    float4x4 inverse_view_projection;
    // Linear rgb with `Sky::intensity` already folded in, so this shader is a
    // lerp and holds no opinion about the gradient's scale. `w` unused in all
    // three.
    float4 zenith;
    float4 horizon;
    float4 ground;
}

[[vk::push_constant]]
ConstantBuffer<SkyPush> push;

struct VOut
{
    float4 pos : SV_Position;
    // The direction *before* normalization: interpolating a unit vector across a
    // triangle does not give a unit vector, so the normalize belongs in the
    // fragment shader and this stays the raw homogeneous direction.
    float3 direction : DIRECTION;
}

[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    float2 uv = float2((vid << 1) & 2, vid & 2);
    float2 ndc = uv * 2.0 - 1.0;
    VOut o;
    // z = 0 is the far plane under reverse-Z, which is both where the sky is and
    // the depth value the attachment was cleared to.
    o.pos = float4(ndc, 0.0, 1.0);
    o.direction = mul(push.inverse_view_projection, float4(ndc, 0.0, 1.0)).xyz;
    return o;
}

[shader("fragment")]
float4 fs_main(VOut i) : SV_Target
{
    // The same gradient `sky.rs`'s `radiance` projects to spherical harmonics,
    // and it has to be: this is what a mirror is *supposed* to be reflecting, so
    // a shader that drew one curve while the lighting integrated another would
    // put a reflection beside its source that did not match it.
    //
    // Met along the square root of altitude, which concentrates the transition
    // in the twenty degrees above the horizon where a real sky keeps it.
    float3 d = normalize(i.direction);
    float3 pole = d.y >= 0.0 ? push.zenith.rgb : push.ground.rgb;
    return float4(lerp(push.horizon.rgb, pole, sqrt(min(abs(d.y), 1.0))), 1.0);
}
