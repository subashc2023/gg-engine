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
//   [Isi06] Isidoro, "Shadow Mapping: GPU-based Tips and Techniques", GDC 2006
//           — receiver-plane depth bias: the 2x2 solve below, and the argument
//           that a PCF kernel's own footprint is what needs biasing.
//   [Hol11] Holbert, "Saying Goodbye to Shadow Acne", 2011 — normal-offset
//           scaled by the shadow texel's world size and by sin(incidence).
//   [ZSXL06] Zhang, Sun, Xu & Lu, "Parallel-Split Shadow Maps for Large-scale
//           Virtual Environments", 2006 — the cascade split scheme the CPU side
//           fits against, and the containment-based selection below.

static const float PI = 3.14159265358979323846;

// `gg_ecs::boundary::light`'s constants, on the other side of the wire.
static const uint LIGHT_DIRECTIONAL = 0;
static const uint LIGHT_POINT = 1;

// Strides written here and asserted against the Rust records beside them, the
// same agreement `VERTEX_STRIDE` and `INSTANCE_STRIDE` are half of. HLSL has no
// `sizeof`, so a drift is a build error rather than a garbled read.
static const uint FRAME_STRIDE = 448;
static const uint LIGHT_STRIDE = 48;
static const uint CASCADE_STRIDE = 80;
// Where the cascade array starts inside the frame block.
static const uint CASCADE_BASE = 128;
// Must equal `lighting.rs`'s MAX_CASCADES: the loops below are unrolled over it
// and a mismatch would read past the block.
static const uint MAX_CASCADES = 4;

// How far the receiver-plane extrapolation below is allowed to lean, in NDC
// depth per unit of shadow uv. The slab `Sun::of` fits spans 2*radius across
// the map and 4*radius through it, so this quantity is tan(incidence)/2 for any
// radius — 4.0 is 83 degrees. Past that the derivative is more likely to be a
// screen-space silhouette than a surface, and the normal offset carries it.
static const float SHADOW_PLANE_SLOPE_LIMIT = 4.0;

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
    float3 ambient;            // linear rgb, a flat term standing in for GI (P1)
    uint light_count;
    uint shadow_sampler;       // NearestClamp: the comparison is per tap, below
    uint cascade_count;        // 0 when no directional light casts
    float shadow_texel;        // 1 / shadow map edge, in uv; one size for every cascade
    float3 sun_direction;      // the direction the casting light travels; zero when none casts
    float shadow_normal_bias;  // *texels* of normal offset, angle-scaled in `to_light_clip`
    float shadow_depth_bias;   // the angle-free part of the normal offset, also in texels
    float shadow_blend;        // fraction of a cascade it cross-fades into the next over
}

// One cascade. `texel_world` is per-cascade and that is the point: the bias is
// in texels, and a texel is a different number of metres in each.
struct Cascade
{
    float4x4 view_projection;
    uint texture;
    float texel_world;
    float split_far;
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
    frame.ambient = float3(f[16], f[17], f[18]);
    uint* u = (uint*)address;
    frame.light_count = u[20];
    frame.shadow_sampler = u[21];
    frame.cascade_count = u[22];
    frame.shadow_texel = f[23];
    frame.sun_direction = float3(f[24], f[25], f[26]);
    frame.shadow_normal_bias = f[27];
    frame.shadow_depth_bias = f[28];
    frame.shadow_blend = f[29];
    return frame;
}

Cascade load_cascade(uint64_t frame_address, uint index)
{
    uint64_t at = frame_address + uint64_t(CASCADE_BASE) + uint64_t(index) * CASCADE_STRIDE;
    float* f = (float*)at;
    uint* u = (uint*)at;
    Cascade c;
    c.view_projection = float4x4(f[0], f[1], f[2], f[3],
                                 f[4], f[5], f[6], f[7],
                                 f[8], f[9], f[10], f[11],
                                 f[12], f[13], f[14], f[15]);
    c.texture = u[16];
    c.texel_world = f[17];
    c.split_far = f[18];
    return c;
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

// Where the shadow map is sampled from, pushed off the surface along its normal
// — [Hol11]. Shared by both forward passes so the two cannot drift.
//
// Scaled to the *texel* and to sin(incidence) rather than being a fixed number
// of metres. What the offset has to clear is the map's own sampling footprint,
// which is one texel wide whatever the cascade covers, and the depth error that
// footprint hides grows with how obliquely the light meets the surface. An
// angle-free offset in metres is simultaneously too much face-on — where it
// peter-pans a contact shadow off its own caster — and far too little at
// grazing, which is where the acne actually is.
//
// `shadow_normal_bias` is therefore a *reach in texels*: 1.0 clears one, and the
// 3x3 kernel's corner tap sits sqrt(2) out. Applied here, at projection time, and
// deliberately not in the shadow pass — the depth the map records stays the
// geometry's own, so one caster's bias cannot follow it onto another receiver.
//
// `shadow_depth_bias` is the part that must *not* vanish head-on. There is no
// footprint error at zero incidence, but there is still a disagreement: the
// shadow pass and this one rasterize the same surface through two different
// projections, and a coin flip between them shadows an entire flat floor. Head-on
// the normal *is* the direction of the light, so pushing along it is exactly a
// depth push — which is also why nothing here has a reverse-Z sign to get wrong.
float3 to_light_clip(Frame frame, Cascade cascade, float3 world, float3 normal)
{
    float n_dot_l = dot(normal, -frame.sun_direction);
    float sin_incidence = sqrt(saturate(1.0 - n_dot_l * n_dot_l));
    float offset = cascade.texel_world *
                   (frame.shadow_depth_bias + frame.shadow_normal_bias * sin_incidence);
    return mul(cascade.view_projection, float4(world + normal * offset, 1.0)).xyz;
}

// The receiver's own depth gradient in shadow-map uv — [Isi06]'s 2x2 solve.
//
// Screen-space derivatives give dz/dx and duv/dx; the inverse of the uv-by-screen
// Jacobian turns those into dz/du and dz/dv, which is what lets each PCF tap
// compare against the *plane the fragment lies in* rather than against the
// fragment's single depth. That is the acne the kernel itself manufactures: a tap
// a texel away is looking at a part of the same surface that is legitimately
// nearer the light, and without this it reads as an occluder.
//
// Must be called from uniform control flow — which with cascades means called for
// *every* cascade and then selected, never called for the chosen one. Two
// fragments of one quad can land in different cascades, and a derivative taken
// after that branch reads whatever the helper lanes held.
float2 receiver_plane_slope(float2 uv, float z)
{
    // All four before any branch, and the degenerate case selected rather than
    // returned early: the rule this function's contract states applies to it as
    // well, and a `ddx` after an `if` reads whatever the helper lanes held.
    float2 duvdx = ddx(uv);
    float2 duvdy = ddy(uv);
    float dzdx = ddx(z);
    float dzdy = ddy(z);
    // Degenerate where the surface is edge-on to the *camera*: the quad's two
    // derivatives are parallel and the slope they imply is noise, not a plane.
    // Divided by a floored magnitude so the arithmetic below stays finite; the
    // select is what discards it.
    float det = duvdx.x * duvdy.y - duvdx.y * duvdy.x;
    bool usable = abs(det) >= 1e-12;
    float2 solved = float2(duvdy.y * dzdx - duvdx.y * dzdy,
                           duvdx.x * dzdy - duvdy.x * dzdx) /
                    (usable ? det : 1.0);
    float2 slope = usable ? solved : float2(0.0, 0.0);
    // Clamped, because at a screen-space silhouette the derivative straddles two
    // surfaces and reports a plane belonging to neither — unclamped, that is a
    // tap depth far enough off to punch a lit hole through a real shadow.
    float magnitude = length(slope);
    return magnitude > SHADOW_PLANE_SLOPE_LIMIT ? slope * (SHADOW_PLANE_SLOPE_LIMIT / magnitude)
                                                : slope;
}

// Whether a point projected into a cascade actually landed in its map.
bool inside_cascade(float3 clip)
{
    return all(abs(clip.xy) <= 1.0) && clip.z > 0.0 && clip.z <= 1.0;
}

// How far out toward a cascade's border a point is: 0 at the centre, 1 at the
// edge. What the cross-fade to the next cascade is parameterized on — a distance
// in *this* cascade's own extent, which is the quantity that says "about to run
// out of map" whatever the split distances happen to be.
float cascade_edge(float3 clip)
{
    return max(abs(clip.x), abs(clip.y));
}

// 3x3 percentage-closer filtering in one cascade, the comparison written out per
// tap rather than done by a comparison sampler. That is deliberate: a
// `SamplerComparisonState` would mean either a fourth immutable sampler or
// aliasing the existing binding with a second type, and neither is worth it for
// nine taps whose arithmetic is this simple. It also keeps the reverse-Z sense
// visible at the site that depends on it — and it is what makes the per-tap plane
// extrapolation possible at all, since a comparison sampler takes one depth for
// the whole kernel.
//
// No derivatives here, which is what makes it safe under the divergent
// per-fragment cascade choice: every tap is an explicit-LOD sample.
float pcf(Frame frame, uint texture, float3 clip, float2 slope,
          Texture2D<float4> textures[], SamplerState samplers[])
{
    // The light's projection is orthographic, so its w is 1 and this is already
    // NDC. x and y go from [-1, 1] to [0, 1]; y needs no flip, because
    // `orthographic_reverse_z` is Y-down like every other projection here.
    float2 uv = clip.xy * 0.5 + 0.5;
    float lit = 0.0;
    for (int y = -1; y <= 1; ++y)
    {
        for (int x = -1; x <= 1; ++x)
        {
            float2 offset = float2(float(x), float(y)) * frame.shadow_texel;
            float caster = textures[texture]
                               .SampleLevel(samplers[frame.shadow_sampler], uv + offset, 0)
                               .r;
            // Reverse-Z: nearer compares *greater*. A fragment is lit when it is
            // at least as near the light as whatever the map recorded — measured
            // where the tap is, along the plane, not where the fragment is.
            lit += clip.z + dot(offset, slope) >= caster ? 1.0 : 0.0;
        }
    }
    return lit / 9.0;
}

// How much of the sun reaches this fragment, in [0, 1].
//
// # Why this takes a world position rather than a projected one
//
// The cascade a point belongs to is a property of the point, not of the vertex it
// was interpolated from: a triangle spanning a split has fragments in two
// cascades, and a single `light_clip` varying could only ever have been right for
// one of them. So the projection happens here, per fragment, per cascade — which
// is also what lets the normal offset use the chosen cascade's own texel size.
//
// # The two phases, which are not stylistic
//
// Phase one runs over **every** cascade under uniform control flow, because
// `receiver_plane_slope` takes screen-space derivatives and neighbouring
// fragments may choose differently. Phase two selects and filters, and may
// diverge freely — `pcf` samples at an explicit LOD and takes no derivatives.
float sun_shadow(uint64_t frame_address, Frame frame, float3 world, float3 normal,
                 Texture2D<float4> textures[], SamplerState samplers[])
{
    // Uniform across the whole draw: it comes out of the frame block, so this
    // branch never splits a quad.
    if (frame.cascade_count == 0)
    {
        return 1.0;
    }

    float3 clip[MAX_CASCADES];
    float2 slope[MAX_CASCADES];
    uint texture[MAX_CASCADES];
    [ForceUnroll]
    for (uint i = 0; i < MAX_CASCADES; ++i)
    {
        // Clamped rather than skipped: the loop bound has to be the compile-time
        // maximum for the derivatives inside to be uniform, so an unused cascade
        // recomputes the last live one instead of branching around it.
        Cascade c = load_cascade(frame_address, min(i, frame.cascade_count - 1));
        clip[i] = to_light_clip(frame, c, world, normal);
        slope[i] = receiver_plane_slope(clip[i].xy * 0.5 + 0.5, clip[i].z);
        texture[i] = c.texture;
    }

    // The tightest cascade containing the point, which for splits fitted
    // nearest-first is the first one that does — containment rather than view
    // depth, because it asks the question that matters directly: is there a map
    // here.
    uint chosen = MAX_CASCADES;
    for (uint i = 0; i < frame.cascade_count; ++i)
    {
        if (inside_cascade(clip[i]))
        {
            chosen = i;
            break;
        }
    }
    // Past the last cascade is **lit**, never shadowed. There is simply no depth
    // recorded out there, and treating the absence as shadow would put a hard
    // black horizon at the shadow distance — the single most recognizable way of
    // getting this wrong.
    if (chosen == MAX_CASCADES)
    {
        return 1.0;
    }

    float lit = pcf(frame, texture[chosen], clip[chosen], slope[chosen], textures, samplers);

    // Cross-fade into the next cascade over the outer `shadow_blend` of this one.
    // Without it the split is a visible line: the kernel is a fixed three texels
    // and those texels are a different size either side, so the softness of every
    // shadow steps as it crosses.
    uint next = chosen + 1;
    float border = 1.0 - frame.shadow_blend;
    float edge = cascade_edge(clip[chosen]);
    if (next < frame.cascade_count && edge > border && inside_cascade(clip[next]))
    {
        float t = saturate((edge - border) / max(frame.shadow_blend, 1e-6));
        float far = pcf(frame, texture[next], clip[next], slope[next], textures, samplers);
        lit = lerp(lit, far, t);
    }
    return lit;
}

// Every light in the frame, over one surface. `world` is camera-relative, so the
// eye is the origin and the view vector is just its negation — which is the same
// property that lets a camera 10^12 m out render without jitter (§1.4).
float3 shade_all(uint64_t frame_address, Frame frame, Surface s, float3 world,
                 Texture2D<float4> textures[], SamplerState samplers[])
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
                radiance *= sun_shadow(frame_address, frame, world, s.normal, textures, samplers);
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
