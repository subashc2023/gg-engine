// The irradiance field's integration (§6 M36): six rendered faces of a probe,
// reduced to the two records the shading pass reads.
//
// Two entry points over one atlas row, because the two records want opposite
// treatment and `probe.rs`'s header is where that is argued: irradiance is
// low-frequency and becomes four coefficients per channel, distance is a step
// function and stays per direction as [Don05]'s two moments. Both walk the same
// loop — every texel of the probe's six faces, weighted by the solid angle a
// cube-face texel actually covers — so the sampling is one thing to be right
// about rather than two.
//
// # Where the sky comes from
//
// A face texel nothing drew into is `w == 0`, and it is filled with the
// environment along that direction rather than left black. That is what makes
// the field *replace* the environments' diffuse term instead of adding to it: a
// probe's record already contains the sky it can see and the part of it the
// walls took away, so a room's interior darkens because its probes were walled
// off rather than because a separate occlusion term said so.

#include "include/pbr.slang"

// The global set (§4.3). Binding 1 (storage images) has no consumer here.
[[vk::binding(0, 0)]] Texture2D<float4> g_textures[];
[[vk::binding(2, 0)]] SamplerState g_samplers[];

struct ProbePush
{
    uint64_t frame;      // this frame's lighting environment, and `probe_views`
    uint atlas;          // the radiance atlas in g_textures
    uint slot;           // which row of it — this probe's place in the batch
    uint face_edge;      // one face tile's edge, texels
    uint moment_edge;    // one octahedral distance tile's edge, texels
    uint2 origin;        // this draw's viewport corner, in target pixels
    float distance_cap;  // metres a recorded distance is clamped to
}

[[vk::push_constant]]
ConstantBuffer<ProbePush> push;

struct VOut
{
    float4 pos : SV_Position;
}

// `post.slang`'s fullscreen triangle without the uv: every fragment here finds
// itself from `SV_Position` and the push's viewport corner, because a probe's
// target rectangle is four texels wide and an interpolated uv across it would
// be a fraction of a texel to be wrong about.
[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

// One texel of one face: which direction it looks along, how much of the sphere
// it covers, and what came back.
struct Sample
{
    float3 direction;
    float solid_angle;
    float3 radiance;
    float distance;  // clamped, and already positive
    bool back;       // the face saw the inside of something this way
    bool miss;       // nothing was drawn this way at all
};

Sample face_sample(uint face, uint2 texel, float edge)
{
    float2 ndc = (float2(texel) + 0.5) / edge * 2.0 - 1.0;
    Sample s;
    s.direction = probe_face_direction(face, ndc);
    s.solid_angle = probe_face_solid_angle(ndc, edge);
    float4 read = g_textures[push.atlas].Load(
        int3(int(face * push.face_edge + texel.x), int(push.slot * push.face_edge + texel.y), 0));
    s.radiance = read.rgb;
    // Exactly zero is the clear, which is "nothing was drawn this way" — a
    // fragment at a probe's own position would also read zero and is a
    // degenerate case that has no radiance to contribute either.
    s.back = read.w < 0.0;
    s.miss = read.w == 0.0;
    s.distance = s.miss ? push.distance_cap : min(abs(read.w), push.distance_cap);
    return s;
}

// The environment along `direction` at the probe — `ambient_light`'s walk, for
// radiance in one direction rather than irradiance over a hemisphere.
//
// The same front-to-back budget and the same two accumulators, and it has to be:
// a probe filling its misses from a different rule than the one that shades the
// pixels around it would put a seam at the edge of every doorway.
float3 environment_radiance(Frame frame, float3 world, float3 direction)
{
    if (frame.environment_count == 0)
    {
        return frame.ambient;
    }
    float3 sh[SH_COEFFICIENTS];
    [ForceUnroll]
    for (uint c = 0; c < SH_COEFFICIENTS; ++c)
    {
        sh[c] = float3(0.0, 0.0, 0.0);
    }
    float3 sampled = float3(0.0, 0.0, 0.0);
    float budget = 1.0;
    for (uint i = 0; i < frame.environment_count && i < MAX_SKY; ++i)
    {
        if (budget <= 1e-3)
        {
            break;
        }
        Environment e = load_environment(push.frame, i);
        float weight = min(environment_weight(e, world), 1.0) * budget;
        if (weight <= 0.0)
        {
            continue;
        }
        budget -= weight;
        if (e.radiance_levels > 0)
        {
            // Alpha zero: this is what a probe *saw*, not what a rough surface
            // would reflect, so the chain is asked for its sharpest level.
            float3 aim = parallax_direction(e, world, direction);
            sampled += weight * prefiltered_radiance(e, aim, 0.0, g_textures, g_samplers);
        }
        else
        {
            accumulate_sh(push.frame, i, weight, sh);
        }
    }
    return sampled + sh_radiance(sh, direction, 0.0);
}

// The irradiance record: three texels of L1 per channel and one of probe state.
//
// Which of the four this fragment is comes from `SV_Position` against the push's
// corner — the draw is one row of `PROBE_SH_TEXELS` pixels, so the four run this
// same loop over four different reductions of it rather than one shader writing
// four targets.
[shader("fragment")]
float4 fs_sh(VOut i) : SV_Target
{
    uint channel = uint(i.pos.x) - push.origin.x;
    Frame frame = load_frame(push.frame);
    float3 probe = load_probe_position(load_probe_views(push.frame), push.slot);
    float edge = float(push.face_edge);

    // `(c_y, c_z, c_x, c_0)` — `sh_basis`'s own order with the constant last,
    // which is the order `probe_irradiance` reads.
    float4 coefficients = float4(0.0, 0.0, 0.0, 0.0);
    float solid = 0.0;
    float distance = 0.0;
    float behind = 0.0;
    for (uint face = 0; face < PROBE_FACES; ++face)
    {
        for (uint y = 0; y < push.face_edge; ++y)
        {
            for (uint x = 0; x < push.face_edge; ++x)
            {
                Sample s = face_sample(face, uint2(x, y), edge);
                solid += s.solid_angle;
                distance += s.solid_angle * s.distance;
                behind += s.back ? s.solid_angle : 0.0;
                if (channel >= PROBE_STATE_TEXEL)
                {
                    continue;
                }
                float3 radiance =
                    s.miss ? environment_radiance(frame, probe, s.direction) : s.radiance;
                float value = radiance[channel] * s.solid_angle;
                float4 basis = float4(0.48860251 * s.direction.y, 0.48860251 * s.direction.z,
                                      0.48860251 * s.direction.x, 0.28209479);
                coefficients += value * basis;
            }
        }
    }
    if (channel < PROBE_STATE_TEXEL)
    {
        // No normalization by the total solid angle: an SH projection is an
        // integral over the sphere and the weights already are the measure. What
        // it is *not* is an average, and dividing here would scale every
        // coefficient by 1/4pi and darken the field by that much.
        return coefficients;
    }
    // The state texel. `behind / solid` is the fraction of the probe's own
    // sphere that came back a back face, which is how much of it is buried —
    // the one number `gi_irradiance` reads out of this texel.
    float total = max(solid, 1e-6);
    return float4(distance / total, behind / total, 0.0, 1.0);
}

// The distance record: one octahedral tile per probe, two moments a texel, with
// a one-texel border (§6 M69).
//
// The kernel's exponent is derived rather than tuned. A texel of an `edge` tile
// covers `4pi / edge^2` steradians, whose half-angle `t` satisfies
// `cos(t)^k = 1/2` at the width where the kernel should fall to half — which
// works out at `k ~= 0.3 * edge^2` across the range this tile is allowed to be.
// A fixed exponent would be a filter that is a blur at one tile size and a point
// sample at another.
//
// **The border is one line of this shader** and no second pass: the viewport is
// `edge + 2` square, the uv of a border texel lands outside `[0, 1]`, and
// `octahedral_wrap` folds it back to the direction it stands for. So a border
// texel is the same integral in the same direction as the interior texel it
// mirrors — exactly equal rather than a replication, which is what
// `the_moment_tile_border_is_its_own_mirror` asserts.
[shader("fragment")]
float4 fs_moments(VOut i) : SV_Target
{
    uint2 texel = uint2(i.pos.xy) - push.origin;
    float2 uv =
        (float2(texel) + 0.5 - float(PROBE_MOMENT_BORDER)) / float(push.moment_edge);
    float3 direction = octahedral_direction(octahedral_wrap(uv));
    float sharpness = 0.3 * float(push.moment_edge * push.moment_edge);
    float edge = float(push.face_edge);

    float weight = 0.0;
    float mean = 0.0;
    float square = 0.0;
    for (uint face = 0; face < PROBE_FACES; ++face)
    {
        for (uint y = 0; y < push.face_edge; ++y)
        {
            for (uint x = 0; x < push.face_edge; ++x)
            {
                Sample s = face_sample(face, uint2(x, y), edge);
                float cosine = dot(s.direction, direction);
                if (cosine <= 0.0)
                {
                    continue;
                }
                float w = s.solid_angle * pow(cosine, sharpness);
                weight += w;
                mean += w * s.distance;
                square += w * s.distance * s.distance;
            }
        }
    }
    // A texel no face landed in cannot happen — the six faces cover the sphere
    // and the kernel spans more than one of their texels — but the cap is the
    // answer that reads as "nothing that way" rather than as a wall at zero.
    if (weight <= 1e-9)
    {
        return float4(push.distance_cap, push.distance_cap * push.distance_cap, 0.0, 1.0);
    }
    return float4(mean / weight, square / weight, 0.0, 1.0);
}
