// The shading model (§6 M11), shared by the two forward pipelines through
// `#include`. Not a module of its own: it has no entry points, and
// `xtask shaders` compiles what is directly under `shaders/` — which is why this
// lives one level down, and why its bytes are folded into every including
// module's source hash.
//
// # References
//
// The equations below are standard and are cited rather than derived. Where a
// choice existed between two published forms, the citation is the one this
// implements, so a future reader can check the code against a paper instead of
// against an opinion.
//
//   [CT82]  Cook & Torrance, "A Reflectance Model for Computer Graphics",
//           SIGGRAPH 1982 — the microfacet framework: D, F and G over
//           4*NdotL*NdotV.
//   [Wal07] Walter, Marschner, Li & Torrance, "Microfacet Models for Refraction
//           through Rough Surfaces", EGSR 2007 — GGX / Trowbridge-Reitz, eq. 33.
//   [Bur12] Burley, "Physically-Based Shading at Disney", SIGGRAPH 2012 course
//           — roughness is perceptual and alpha = roughness^2. Every glTF asset
//           is authored against that remap, so it is not optional here.
//   [Kar13] Karis, "Real Shading in Unreal Engine 4", SIGGRAPH 2013 course —
//           the Schlick exponent as an exp2, and the windowed inverse-square
//           falloff.
//   [Hei14] Heitz, "Understanding the Masking-Shadowing Function in
//           Microfacet-Based BRDFs", JCGT 3(2) — the height-correlated Smith
//           visibility term used below.
//   [LdR14] Lagarde & de Rousiers, "Moving Frostbite to Physically Based
//           Rendering", SIGGRAPH 2014 course — the V-vs-G bookkeeping, and the
//           argument for folding the 4*NdotL*NdotV denominator into V.
//   [Sch94] Schlick, "An Inexpensive BRDF Model for Physically-based
//           Rendering", Eurographics 1994 — the Fresnel approximation.

static const float PI = 3.14159265358979323846;

// `gg_ecs::boundary::light`'s constants, on the other side of the wire.
static const uint LIGHT_DIRECTIONAL = 0;
static const uint LIGHT_POINT = 1;

// Strides written here and asserted against the Rust records beside them, the
// same agreement `VERTEX_STRIDE` and `INSTANCE_STRIDE` are half of. HLSL has no
// `sizeof`, so a drift is a build error rather than a garbled read.
static const uint FRAME_STRIDE = 176;
static const uint LIGHT_STRIDE = 48;

// Dielectric reflectance at normal incidence — 4%, the value [Bur12] and every
// engine since use for "not a metal". A metal has no single number here, which
// is why a metal's F0 is its base colour instead.
static const float3 DIELECTRIC_F0 = float3(0.04, 0.04, 0.04);

// One frame's lighting environment, read once per shader invocation out of the
// per-frame buffer. Scalars rather than a struct pointer, for the same reason
// vertices are: the pointee's layout would otherwise be Slang's to choose while
// the Rust side is packed `repr(C)`.
struct Frame
{
    // The camera's own projection, here rather than in a push block because the
    // box pass has one push per box and no room for another 64 bytes under
    // Vulkan's guaranteed 128. The pack pass does not read it: its instances
    // carry the finished object-to-clip product for the bit-exactness reason
    // `scene.slang` gives.
    float4x4 view_projection;
    float4x4 sun_view_projection; // camera-relative world -> the sun's clip space
    float3 ambient;               // linear rgb, a flat term standing in for GI (P1)
    uint light_count;
    uint shadow_texture;      // the shadow map's slot in g_textures
    uint shadow_sampler;      // NearestClamp: the comparison is per tap, below
    uint shadow_enabled;      // 0 when no directional light casts
    float shadow_texel;       // 1 / shadow map edge, in uv
    float shadow_normal_bias; // metres along the normal, applied at projection time
}

struct Light
{
    float3 position;  // camera-relative; zero for a directional light
    float range;      // metres to exactly zero; zero for a directional light
    float3 direction; // the direction light *travels*; zero for a point light
    uint kind;
    float3 color;     // linear rgb
    float intensity;
}

Frame load_frame(uint64_t address)
{
    float* f = (float*)address;
    Frame frame;
    frame.view_projection = float4x4(f[0], f[1], f[2], f[3],
                                     f[4], f[5], f[6], f[7],
                                     f[8], f[9], f[10], f[11],
                                     f[12], f[13], f[14], f[15]);
    frame.sun_view_projection = float4x4(f[16], f[17], f[18], f[19],
                                         f[20], f[21], f[22], f[23],
                                         f[24], f[25], f[26], f[27],
                                         f[28], f[29], f[30], f[31]);
    frame.ambient = float3(f[32], f[33], f[34]);
    uint* u = (uint*)address;
    frame.light_count = u[36];
    frame.shadow_texture = u[37];
    frame.shadow_sampler = u[38];
    frame.shadow_enabled = u[39];
    frame.shadow_texel = f[40];
    frame.shadow_normal_bias = f[41];
    return frame;
}

Light load_light(uint64_t frame_address, uint index)
{
    uint64_t at = frame_address + uint64_t(FRAME_STRIDE) + uint64_t(index) * LIGHT_STRIDE;
    float* f = (float*)at;
    uint* u = (uint*)at;
    Light light;
    light.position = float3(f[0], f[1], f[2]);
    light.range = f[3];
    light.direction = float3(f[4], f[5], f[6]);
    light.kind = u[7];
    light.color = float3(f[8], f[9], f[10]);
    light.intensity = f[11];
    return light;
}

// What the BRDF needs about a point on a surface, resolved from the material
// before any light is considered.
struct Surface
{
    float3 normal;    // unit, camera-relative world space, after normal mapping
    float3 diffuse;   // albedo scaled by (1 - metallic): a metal has no diffuse lobe
    float3 f0;        // specular reflectance at normal incidence
    float alpha;      // roughness^2, the [Bur12] remap
    float occlusion;  // ambient occlusion, 1 for none
}

Surface make_surface(float3 normal, float3 base_color, float metallic, float roughness,
                     float occlusion)
{
    Surface s;
    s.normal = normal;
    // Metal/rough: a metal's specular colour *is* its base colour, and it has no
    // diffuse lobe at all — the two interpolate on `metallic` rather than
    // branching, which is what makes a half-metal legal instead of undefined.
    s.diffuse = base_color * (1.0 - metallic);
    s.f0 = lerp(DIELECTRIC_F0, base_color, metallic);
    // Clamped away from zero: alpha == 0 is a perfect mirror, where D_GGX is a
    // Dirac delta that a single sample cannot represent and which shows up as a
    // fireflying pixel rather than as a highlight.
    s.alpha = max(roughness * roughness, 1e-3);
    s.occlusion = occlusion;
    return s;
}

// GGX / Trowbridge-Reitz normal distribution — [Wal07] eq. 33.
float d_ggx(float n_dot_h, float alpha)
{
    float a2 = alpha * alpha;
    float d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

// Height-correlated Smith visibility — [Hei14], in the form [LdR14] gives.
//
// A *visibility* term and not a geometry term: the 4*NdotL*NdotV of [CT82]'s
// denominator is already folded in here, which is why the caller multiplies D,
// F and this together and divides by nothing.
float v_smith_correlated(float n_dot_v, float n_dot_l, float alpha)
{
    float a2 = alpha * alpha;
    float lambda_v = n_dot_l * sqrt(n_dot_v * n_dot_v * (1.0 - a2) + a2);
    float lambda_l = n_dot_v * sqrt(n_dot_l * n_dot_l * (1.0 - a2) + a2);
    return 0.5 / max(lambda_v + lambda_l, 1e-7);
}

// Fresnel — [Sch94], with [Kar13]'s spherical-Gaussian form of the fifth power.
float3 f_schlick(float3 f0, float v_dot_h)
{
    return f0 + (1.0 - f0) * exp2((-5.55473 * v_dot_h - 6.98316) * v_dot_h);
}

// One light's contribution. `to_light` is unit and points *from* the surface
// toward the light; `radiance` is the light's colour times its intensity times
// whatever attenuation applies.
float3 shade(Surface s, float3 to_eye, float3 to_light, float3 radiance)
{
    float3 h = normalize(to_eye + to_light);
    float n_dot_l = saturate(dot(s.normal, to_light));
    if (n_dot_l <= 0.0)
    {
        return float3(0.0, 0.0, 0.0);
    }
    // Nudged off zero rather than saturated to it: at grazing incidence the
    // visibility term divides by something proportional to this, and a hard zero
    // is a NaN on the silhouette of every object.
    float n_dot_v = max(dot(s.normal, to_eye), 1e-4);
    float n_dot_h = saturate(dot(s.normal, h));
    float v_dot_h = saturate(dot(to_eye, h));

    float3 fresnel = f_schlick(s.f0, v_dot_h);
    float3 specular = fresnel * d_ggx(n_dot_h, s.alpha) * v_smith_correlated(n_dot_v, n_dot_l, s.alpha);
    // Energy conservation, the cheap way: what the surface reflected specularly
    // is not available to scatter. Exact only for a single interface, which is
    // the approximation every forward renderer of this class makes.
    float3 diffuse = (1.0 - fresnel) * s.diffuse / PI;
    return (diffuse + specular) * radiance * n_dot_l;
}

// Inverse-square, windowed to reach *exactly* zero at `range` — [Kar13].
//
// Exactly zero is the part that matters here: `Light::range` is documented as
// the bound `gg-extract` culls against, and a falloff that only asymptotes would
// make that documentation false — a light would be dropped while it was still
// contributing.
float point_falloff(float distance_squared, float range)
{
    float ratio = distance_squared / max(range * range, 1e-6);
    float window = saturate(1.0 - ratio * ratio);
    // The denominator is clamped rather than the distance: a surface exactly at
    // a light's position is a division by zero, and the clamp is what stops it
    // from being an infinity that survives the tonemapper as a white pixel.
    return window * window / max(distance_squared, 1e-4);
}

// How much of the sun reaches `light_clip`, in [0, 1].
//
// 3x3 percentage-closer filtering with the comparison written out per tap rather
// than done by a comparison sampler. That is deliberate: a `SamplerComparisonState`
// would mean either a fourth immutable sampler or aliasing the existing binding
// with a second type, and neither is worth it for nine taps whose arithmetic is
// this simple. It also keeps the reverse-Z sense visible at the site that
// depends on it.
float sun_shadow(Frame frame, float3 light_clip,
                 Texture2D<float4> textures[], SamplerState samplers[])
{
    if (frame.shadow_enabled == 0)
    {
        return 1.0;
    }
    // The light's projection is orthographic, so its w is 1 and this is already
    // NDC. x and y go from [-1, 1] to [0, 1]; y needs no flip, because
    // `orthographic_reverse_z` is Y-down like every other projection here.
    float2 uv = light_clip.xy * 0.5 + 0.5;
    // Outside the cascade is **lit**, never shadowed. One cascade covers a slab
    // around the camera and there is simply no depth recorded past it; treating
    // the absence as shadow would put a hard black edge at the cascade boundary,
    // which is the single most recognizable artifact of getting this wrong.
    if (any(uv < 0.0) || any(uv > 1.0) || light_clip.z <= 0.0 || light_clip.z > 1.0)
    {
        return 1.0;
    }
    float lit = 0.0;
    for (int y = -1; y <= 1; ++y)
    {
        for (int x = -1; x <= 1; ++x)
        {
            float2 tap = uv + float2(float(x), float(y)) * frame.shadow_texel;
            float caster = textures[frame.shadow_texture]
                               .SampleLevel(samplers[frame.shadow_sampler], tap, 0)
                               .r;
            // Reverse-Z: nearer compares *greater*. A fragment is lit when it is
            // at least as near the light as whatever the map recorded.
            lit += light_clip.z >= caster ? 1.0 : 0.0;
        }
    }
    return lit / 9.0;
}

// Every light in the frame, over one surface. `world` is camera-relative, so the
// eye is the origin and the view vector is just its negation — which is the same
// property that lets a camera 10^12 m out render without jitter (§1.4).
float3 shade_all(uint64_t frame_address, Frame frame, Surface s, float3 world,
                 float3 light_clip, Texture2D<float4> textures[], SamplerState samplers[])
{
    float3 to_eye = normalize(-world);
    // Ambient is flat and occluded, and it is a placeholder saying so: real
    // indirect light is an irradiance probe or a lightmap, and both are P1. What
    // it buys today is that an unlit face is dim rather than pure black.
    float3 total = frame.ambient * s.diffuse * s.occlusion;
    bool shadowed_sun_seen = false;
    for (uint i = 0; i < frame.light_count; ++i)
    {
        Light light = load_light(frame_address, i);
        float3 radiance = light.color * light.intensity;
        float3 to_light;
        if (light.kind == LIGHT_DIRECTIONAL)
        {
            to_light = -light.direction;
            // The *first* directional light casts, and only it: a second shadow
            // map is a second pass and a cascade decision, and cascades are P1
            // (§6 M11). The rest still light, they just do not occlude.
            if (!shadowed_sun_seen)
            {
                shadowed_sun_seen = true;
                radiance *= sun_shadow(frame, light_clip, textures, samplers);
            }
        }
        else
        {
            float3 offset = light.position - world;
            float distance_squared = dot(offset, offset);
            radiance *= point_falloff(distance_squared, light.range);
            to_light = offset * rsqrt(max(distance_squared, 1e-8));
        }
        total += shade(s, to_eye, to_light, radiance);
    }
    return total;
}

// A tangent-space normal-map sample, brought into world space.
//
// The map is BC5: two channels, z reconstructed. A material naming none samples
// a 1x1 (0.5, 0.5) texel, which decodes to (0, 0, 1) — the flat normal — so the
// path has no branch and an absent map costs one texture read rather than a
// divergence.
float3 apply_normal_map(float3 normal, float4 tangent, float2 sampled)
{
    float2 xy = sampled * 2.0 - 1.0;
    // z from the unit-length constraint. Clamped under the root because BC5's
    // block endpoints can put a compressed texel slightly outside the disc, and
    // a NaN normal is a black pixel that looks like a lighting bug.
    float3 tangent_normal = float3(xy, sqrt(saturate(1.0 - dot(xy, xy))));
    float3 n = normalize(normal);
    // Gram-Schmidt again at shading time: interpolation across a triangle leaves
    // the tangent no longer exactly perpendicular to the normal.
    float3 t = normalize(tangent.xyz - n * dot(n, tangent.xyz));
    // glTF stores the bitangent as a handedness bit, which is what `tangent.w`
    // is; recomputing the bitangent from it cannot disagree with the normal the
    // way an interpolated one would.
    float3 b = cross(n, t) * tangent.w;
    return normalize(tangent_normal.x * t + tangent_normal.y * b + tangent_normal.z * n);
}
