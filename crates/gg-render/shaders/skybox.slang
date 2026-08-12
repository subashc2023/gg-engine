// The environment, drawn where nothing else was (§6 M24) — the gradient, or the
// compiled panorama's own mirror level (§6 M27).
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
// # One draw per environment (§6 M29)
//
// M28 made this a *pick* — the innermost volume containing the camera, cut where
// the shading faded — and named the seam. It is now one of these draws per
// environment whose volume reaches the camera, outermost first, each carrying
// its own straight alpha. Back-to-front `over` with raw weights is algebraically
// the same composite as `ambient_light`'s front-to-back budget, so the
// background fades exactly where the shading does. The count is not a cost worth
// avoiding: a camera deep inside a room emits one draw, and only a camera in a
// doorway emits two. Depth is tested and never written, so every draw meets the
// same `GREATER_OR_EQUAL` against the cleared 0.0 and none of them masks another.
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

#include "include/pbr.slang"

// The global set (§4.3) — the radiance chain is reached through it, like any
// texture. Declared here rather than inherited: `pbr.slang` holds equations, and
// every entry point that samples declares the arrays it samples through.
[[vk::binding(0, 0)]] Texture2D<float4> g_textures[];
[[vk::binding(2, 0)]] SamplerState g_samplers[];

struct SkyPush
{
    float4x4 inverse_view_projection;
    // Linear rgb with `Sky::intensity` already folded in, so this shader is a
    // lerp and holds no opinion about the gradient's scale. `w` unused in all
    // three. Unread when a chain is present.
    float4 zenith;
    float4 horizon;
    float4 ground;
    // The frame block and this draw's index into its environment array. The
    // chain, its scale and the volume all come from there rather than from three
    // more push fields (§6 M29): the block already carries them for the shading
    // path, a second copy is a second thing to keep in step, and at 128 bytes
    // this block is exactly at Vulkan's guaranteed floor with nothing spare.
    uint64_t frame;
    uint environment;
    // What fraction of the background this draw is, straight alpha (§6 M29).
    // One draw per environment whose volume reaches the camera, outermost first,
    // so `Blend::Alpha` composites them in the order that makes the sum equal
    // `ambient_light`'s front-to-back budget.
    float weight;
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
    Environment e = load_environment(push.frame, push.environment);
    if (e.radiance_levels > 0)
    {
        // Parallax-corrected from the camera, which is the origin of this
        // shader's world (§6 M29). The background and a mirror in front of it
        // have to be the same picture, and a bounded environment's picture
        // depends on where it is read from — so a background that skipped the
        // correction would disagree with every reflection of it in the frame.
        float3 aim = normalize(parallax_direction(e, float3(0.0, 0.0, 0.0), d));
        // Level 0, which is the chain's *mirror* level — a skybox is the one
        // consumer of this texture that wants no lobe at all. The same tap a
        // perfectly smooth surface takes.
        return float4(g_textures[e.radiance_texture]
                              .SampleLevel(g_samplers[SAMPLER_LINEAR_CLAMP],
                                           octahedral_uv(aim), 0.0)
                              .rgb
                          * e.radiance_scale,
                      push.weight);
    }
    // No correction on this branch, for `ambient_light`'s reason: a gradient is
    // a function of direction with nothing in it that is anywhere.
    float3 pole = d.y >= 0.0 ? push.zenith.rgb : push.ground.rgb;
    return float4(lerp(push.horizon.rgb, pole, sqrt(min(abs(d.y), 1.0))), push.weight);
}
