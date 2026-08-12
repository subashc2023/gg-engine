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

// The instance scale with every component pulled off zero, sign kept.
//
// Here, in the one file both vertex stages include, because it is a numerical
// guard the two must *agree* on: a normal takes `R * S^-1` (the inverse
// transpose of a diagonal `T*R*S`), so both divide by this, and two copies that
// drifted would shade the same geometry differently through the two pipelines.
//
// Only the tangent-frame paths use it; positions take the raw scale, because
// prepass and forward agree on a fragment's depth only if they compute it from
// bit-identical inputs. A zero component collapses the instance into a plane,
// whose reciprocal is infinite and whose normalize is NaN — a black pixel that
// reads as a lighting bug. Clamped, the collapsed axis merely dominates, so the
// normal points along it: the direction a flattened surface faces. The sign
// survives because a negative scale is a mirror and the normal flips with it.
float3 nonzero_scale(float3 s)
{
    float3 magnitude = max(abs(s), 1e-8);
    return select(s < 0.0, -magnitude, magnitude);
}

// `gg_ecs::boundary::light`'s constants, on the other side of the wire.
static const uint LIGHT_DIRECTIONAL = 0;
static const uint LIGHT_POINT = 1;

// Strides written here and asserted against the Rust records beside them, the
// same agreement `VERTEX_STRIDE` and `INSTANCE_STRIDE` are half of. HLSL has no
// `sizeof`, so a drift is a build error rather than a garbled read.
static const uint FRAME_STRIDE = 624;
static const uint LIGHT_STRIDE = 48;
static const uint CASCADE_STRIDE = 80;
// Where the cascade array starts inside the frame block.
static const uint CASCADE_BASE = 304;
// Coefficients in the order-2 environment projection (§6 M24), and where they
// start. Must equal `sky.rs`'s SH_COEFFICIENTS.
static const uint SH_COEFFICIENTS = 9;
static const uint SH_BASE = 160;
// Must equal `lighting.rs`'s MAX_CASCADES: the loops below are unrolled over it
// and a mismatch would read past the block.
static const uint MAX_CASCADES = 4;

// How far the receiver-plane extrapolation below is allowed to lean, in NDC
// depth per unit of shadow uv. The slab `Sun::of` fits spans 2*radius across
// the map and 4*radius through it, so this quantity is tan(incidence)/2 for any
// radius — 4.0 is 83 degrees. Past that the receiver is nearly parallel to the
// light and a tap's depth is not the plane's to predict; the normal offset
// carries what this declines to.
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
    // Vulkan's guaranteed 128. The pack pass does not *transform* by it — its
    // instances carry the finished object-to-clip product for the bit-exactness
    // reason `scene.slang` gives — but both passes read row 2 in `shade_all`,
    // which is where the view vector learns the projection has no eye point.
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
    float sun_tan;             // tan of the sun's angular *radius* — the penumbra's growth rate
    float shadow_softness;     // narrowest penumbra, in *screen pixels* (§6 M23)
    float shadow_penumbra;     // penumbra cap and blocker search radius, in texels
    uint shadow_taps;          // taps in the filter disk; the search uses a quarter
    float sky;                 // nonzero when the block holds an environment (§6 M24)
}

// One cascade. `texel_world` is per-cascade and that is the point: the bias is
// in texels, and a texel is a different number of metres in each.
struct Cascade
{
    float4x4 view_projection;
    uint texture;
    float texel_world;
    float split_far;
    float depth_world;  // metres one unit of this cascade's clip depth covers
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
    frame.sun_tan = f[30];
    frame.shadow_softness = f[31];
    frame.shadow_penumbra = f[32];
    frame.shadow_taps = u[33];
    frame.sky = f[34];
    return frame;
}

// The environment's coefficients, read out of the same block — but **not** by
// `load_frame`, and that is a measured decision. Loading them eagerly is 27
// scalar loads a fragment for the environment's sake in every scene that has
// none, and it cost the frame macro 6 % on the pinned rasterizer with nothing
// to show for it. `Frame::sky` is uniform across a draw, so the branch that
// guards this splits no quad and a scene without a sky pays nothing.
void load_sh(uint64_t address, out float3 sh[SH_COEFFICIENTS])
{
    float* f = (float*)(address + uint64_t(SH_BASE));
    [ForceUnroll]
    for (uint i = 0; i < SH_COEFFICIENTS; ++i)
    {
        sh[i] = float3(f[i * 4], f[i * 4 + 1], f[i * 4 + 2]);
    }
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
    c.depth_world = f[19];
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
    float3 plane;     // the *geometric* normal — the receiver plane the shadow
                      // extrapolation leans on. A normal map perturbs shading,
                      // not the surface the shadow map rasterized, and a plane
                      // taken from the mapped normal re-acnes every mapped
                      // receiver. `make_surface` defaults it to `normal`;
                      // a mapped pass overrides it with the vertex normal.
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
    s.plane = normal;
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

// ---------------------------------------------------------------------------
// Image-based lighting (§6 M24)
// ---------------------------------------------------------------------------
//
//   [RH01] Ramamoorthi & Hanrahan, "An Efficient Representation for Irradiance
//          Environment Maps", SIGGRAPH 2001 — the three band factors below, and
//          the result that a clamped-cosine kernel has no fourth band.
//   [Slo08] Sloan, "Stupid Spherical Harmonics (SH) Tricks", GDC 2008 — the
//          basis in polynomial form, and band-wise convolution as a product.
//   [Laz13] Lazarov, "Getting More Physical in Call of Duty: Black Ops II",
//          SIGGRAPH 2013 course — the analytic fit to the split-sum's second
//          integral, as [Kar13] publishes it for mobile.

// The real SH basis, bands 0..=2. **The same literals as `sky.rs`'s `basis`,
// and neither is derived from the other** — this evaluates what that one
// projected, they never meet at runtime, and a drift between them is an
// environment that lights from the wrong direction with nothing to fail.
//
// `z` is the polar axis here and the engine's up is `y`; see `sky.rs` for why
// that is deliberate and harmless.
void sh_basis(float3 d, out float y[SH_COEFFICIENTS])
{
    y[0] = 0.28209479;
    y[1] = 0.48860251 * d.y;
    y[2] = 0.48860251 * d.z;
    y[3] = 0.48860251 * d.x;
    y[4] = 1.09254840 * d.x * d.y;
    y[5] = 1.09254840 * d.y * d.z;
    y[6] = 0.31539157 * (3.0 * d.z * d.z - 1.0);
    y[7] = 1.09254840 * d.x * d.z;
    y[8] = 0.54627420 * (d.x * d.x - d.y * d.y);
}

// Irradiance arriving at a surface facing `normal` — [RH01]. The band factors
// are the clamped-cosine kernel's, and there are only three of them because
// that kernel has only three nonzero bands: this is not a truncation of the
// diffuse integral, it is the diffuse integral.
//
// Clamped at zero because a band-limited reconstruction rings: an environment
// with a strong gradient goes slightly negative facing away from it, and a
// negative irradiance is a surface that subtracts light from its neighbours.
float3 sh_irradiance(float3 sh[SH_COEFFICIENTS], float3 normal)
{
    float y[SH_COEFFICIENTS];
    sh_basis(normal, y);
    // pi, 2pi/3, pi/4 — [RH01]'s A0, A1, A2.
    float3 e = 3.14159265 * sh[0] * y[0];
    e += 2.09439510 * (sh[1] * y[1] + sh[2] * y[2] + sh[3] * y[3]);
    e += 0.78539816 * (sh[4] * y[4] + sh[5] * y[5] + sh[6] * y[6] + sh[7] * y[7] + sh[8] * y[8]);
    return max(e, 0.0);
}

// The environment along `direction`, blurred by a lobe of width `alpha` — the
// split-sum's *first* integral, which is conventionally a prefiltered cubemap
// with roughness on the mip axis.
//
// It is not one here, and the reason is what the environment is rather than a
// shortcut. Convolving with a spherical Gaussian is a per-band product ([Slo08])
// — band `l` scaled by `exp(-l(l+1) sigma / 2)` — so an environment already
// represented in three bands convolves *exactly*, for free, with no second
// resource and no mip chain to keep in step with the first. What that cannot
// represent is a sharp reflection of something small, because three bands do
// not hold anything small: a mirror shows the low-order sky and not the sun.
// That is precisely the trade this milestone's environment is chosen to make,
// and the day a loaded panorama arrives it is also the day the prefiltered chain
// has to (§6 M24 exit).
//
// `alpha` and not roughness, so the width is in the same unit GGX's lobe is.
float3 sh_radiance(float3 sh[SH_COEFFICIENTS], float3 direction, float alpha)
{
    float y[SH_COEFFICIENTS];
    sh_basis(direction, y);
    // exp(-x) as exp2(-x * log2 e). Band 2's weight is band 1's cubed —
    // l(l+1)/2 is 1 and 3 — so the whole convolution costs one exp2.
    float w1 = exp2(-1.44269504 * alpha);
    float w2 = w1 * w1 * w1;
    float3 l = sh[0] * y[0];
    l += w1 * (sh[1] * y[1] + sh[2] * y[2] + sh[3] * y[3]);
    l += w2 * (sh[4] * y[4] + sh[5] * y[5] + sh[6] * y[6] + sh[7] * y[7] + sh[8] * y[8]);
    return max(l, 0.0);
}

// The split-sum's *second* integral — how much of the environment a surface of
// this roughness reflects, and with what tint — as [Laz13]'s analytic fit rather
// than as a lookup table. A 2D LUT would be a texture, a bindless slot, a
// generation pass and a resource to bless; the fit is four constants and is
// within a percent of the table over the whole domain.
//
// Returns the scale and bias applied to F0: `f0 * a + b`.
float2 env_brdf(float roughness, float n_dot_v)
{
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    return float2(-1.04, 1.04) * a004 + r.zw;
}

// Fresnel for an *environment* rather than for one direction — [Kar13]. The
// ceiling is pulled down by roughness because a rough surface's grazing
// reflection is spread over directions that are no longer all grazing, and the
// mirror-grade rise to 1.0 at the silhouette is what makes a rough object look
// like it is wrapped in foil.
float3 f_schlick_ambient(float3 f0, float n_dot_v, float roughness)
{
    float3 ceiling = max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), f0);
    return f0 + (ceiling - f0) * exp2((-5.55473 * n_dot_v - 6.98316) * n_dot_v);
}

// Everything the environment contributes, or the flat ambient term when there
// is no environment — the one place the two are chosen between, so a scene
// blessed before §6 M24 and a scene with a sky do not each need their own path
// through `shade_all`.
float3 ambient_light(uint64_t frame_address, Frame frame, Surface s, float3 to_eye)
{
    if (frame.sky <= 0.0)
    {
        // §6 M11's placeholder, unchanged: a flat term times albedo, occluded.
        return frame.ambient * s.diffuse * s.occlusion;
    }
    float3 sh[SH_COEFFICIENTS];
    load_sh(frame_address, sh);
    float n_dot_v = max(dot(s.normal, to_eye), 1e-4);
    float roughness = sqrt(s.alpha);
    float3 fresnel = f_schlick_ambient(s.f0, n_dot_v, roughness);

    // Lambertian: irradiance over pi, times albedo, times what the specular
    // lobe did not already reflect. `s.diffuse` is albedo with metallic already
    // folded out, so a conductor's diffuse term is zero here by construction.
    float3 diffuse = (1.0 - fresnel) * s.diffuse * sh_irradiance(sh, s.normal) / PI;

    // Specular: the reflected environment through the split-sum. `reflect`
    // takes the incident direction, which is the negation of the one toward the
    // eye — the sign that is wrong in half the implementations of this on the
    // internet and which shows up as a reflection that slides the wrong way.
    float3 reflected = reflect(-to_eye, s.normal);
    float2 ab = env_brdf(roughness, n_dot_v);
    float3 specular = sh_radiance(sh, reflected, s.alpha) * (s.f0 * ab.x + ab.y);

    // Occlusion applies to both. Strictly it is a diffuse quantity — a specular
    // lobe is narrow and mostly misses whatever the occlusion term is standing
    // in for — but a mirror in a crevice reflecting an unoccluded sky is worse
    // than the theory being slightly wrong, and the map is a hemisphere average
    // rather than a measurement anyway.
    return (diffuse + specular) * s.occlusion;
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

// The receiver's own depth gradient in shadow-map uv — what lets each PCF tap
// compare against the *plane the fragment lies in* rather than against the
// fragment's single depth. That is the acne the kernel itself manufactures: a tap
// a texel away is looking at a part of the same surface that is legitimately
// nearer the light, and without this it reads as an occluder.
//
// Analytic, from the surface normal — not [Isi06]'s screen-derivative solve. A
// derivative is a quad's, not a surface's: wherever one 2x2 quad straddles two
// surfaces — every contact corner, every silhouette — it reports a plane
// belonging to neither, and the flat camera (§6 M20) renders that as a dark
// band at the player's feet, two rows tall because a quad is, bias-proof
// because no offset fixes a wrong plane. The normal is the surface's own on
// exactly those fragments, needs no uniform control flow, and costs three dots.
//
// The projection's linear rows are an orthographic slab's — scaled rotation
// rows, mutually orthogonal — so the inverse-transpose that carries a plane
// normal into clip space is each row over its own squared length, no matrix
// inverse anywhere. With that normal m: m . dclip = 0, and uv = clip.xy / 2 +
// 1/2 makes the tap extrapolation dz/duv = -2 m.xy / m.z.
float2 receiver_plane_slope(Cascade cascade, float3 normal)
{
    float3 r0 = cascade.view_projection[0].xyz;
    float3 r1 = cascade.view_projection[1].xyz;
    float3 r2 = cascade.view_projection[2].xyz;
    float3 m = float3(dot(r0, normal) / dot(r0, r0),
                      dot(r1, normal) / dot(r1, r1),
                      dot(r2, normal) / dot(r2, r2));
    // Edge-on to the light m.z vanishes: the floor keeps the division finite,
    // and the clamp below is what actually rules out there, as it always has.
    float mz = (m.z < 0.0 ? -1.0 : 1.0) * max(abs(m.z), 1e-8);
    float2 slope = -2.0 * m.xy / mz;
    // Clamped, because a receiver nearly parallel to the light has more depth
    // per uv than any extrapolation should claim to know — unclamped, that is
    // a tap depth far enough off to punch a lit hole through a real shadow.
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

// The side of a square covering the same area of this surface as one screen
// pixel, in metres. **Must be called from uniform control flow** — it takes
// screen-space derivatives, so a quad whose four fragments did not all reach it
// gets an undefined answer.
//
// A derivative and not an analytic term, and the difference is the whole point.
// The analytic version — the frustum's half-height over the frame's, times the
// view depth — is the pixel's *lateral* width, which is right for a wall square
// to the eye and badly wrong for the case that reported this defect: a floor
// seen at a grazing angle, where one pixel covers several times more world along
// the view than across it. Measured against the lateral width, a floor's
// boundary stayed under a pixel at every softness the sweep could reach.
//
// The area's square root rather than the longer or the shorter side: a shadow
// boundary can run any direction across the pixel, the longer side would blur
// the narrow axis by the full anisotropy (tens, at a grazing angle), and the
// shorter is the under-blur this replaced. The isotropic footprint is the same
// compromise a mip LOD makes, for the same reason.
float pixel_footprint(float3 world)
{
    return sqrt(length(cross(ddx(world), ddy(world))));
}

// Golden-angle spiral over a unit disk: `count` points no two of which share a
// radius, so a partial sum of them is a fair estimate at any count and the
// pattern has no axis for a straight edge to line up with.
//
// **Advanced by a rotation rather than recomputed**, which is a performance
// decision with a measured number behind it. Written the obvious way — an angle
// `i * VOGEL_ANGLE + rotation` and a `sincos` per tap — twenty taps of
// transcendentals a fragment moved the frame macro on the pinned software
// rasterizer from 7.3 ms to 11.9 ms, most of it in `forward-opaque`. Turning by a
// *constant* angle each step is a 2x2 rotation whose entries are compile-time
// constants, so a whole loop costs one `sincos` and each tap costs a complex
// multiply. The drift over thirty-two rotations is far below a texel, and it is
// deterministic, so nothing about a blessed golden depends on the difference.
static const float VOGEL_COS = -0.73736888;
static const float VOGEL_SIN = 0.67549029;

float2 spin(float2 dir)
{
    return float2(dir.x * VOGEL_COS - dir.y * VOGEL_SIN,
                  dir.x * VOGEL_SIN + dir.y * VOGEL_COS);
}

// The spiral's radius at tap `i` of `count`: `sqrt` of the fraction, which is
// what spreads the points by *area* rather than crowding them at the centre.
float spiral_radius(uint i, uint count)
{
    return sqrt((float(i) + 0.5) / float(count));
}

// `lowbias32`, the same mix `post.slang` dithers with. Integer, so it is exact
// on every device and a blessed golden (§4.10) survives it — a float noise hash
// would leave two backends disagreeing about the picture for a reason that is
// not the picture.
uint shadow_hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// This pixel's tap rotation, radians. Screen-space and not world-space: a
// world-space hash is stable under camera motion but aliases against itself at
// distance, where a screen-space one is stable against the display it is being
// hidden in.
float tap_rotation(float2 pixel)
{
    uint2 p = uint2(pixel);
    uint h = shadow_hash(p.x * 73856093u ^ p.y * 19349663u);
    return float(h >> 8) * (1.0 / 16777216.0) * (2.0 * PI);
}

// The parts of a cascade the filter needs. Carried instead of the whole
// `Cascade` because `sun_shadow` holds one of these per cascade and four 4x4
// matrices in registers is a real cost for two floats' worth of question.
struct CascadeShape
{
    uint texture;
    float texel_world;
    float depth_world;
}

// Percentage-closer *soft* shadows in one cascade (§6 M23): a blocker search
// that sizes the penumbra, then a disk filter at that size.
//
// # Why the radius cannot be a fixed number of texels
//
// It was one until this milestone, and a screenshot off the desk is why it is
// not. A shadow boundary is a shading discontinuity *inside* a triangle: every
// MSAA sample in the pixel lies on the same triangle and takes the same tap, so
// coverage antialiasing cannot touch it however many samples are paid for — the
// silhouette of the very cube casting the shadow comes out smooth beside it,
// which is what makes the defect so recognizable. Measured off that frame, the
// boundary crossed 140 levels in **one pixel**: a three-texel kernel that
// projected to 0.01 px, because a texel is a wildly different number of pixels
// in each cascade and at each distance. Nothing phrased in texels can promise
// otherwise.
//
// So the radius is the larger of two quantities answering to different masters:
//
//   * the **physical** penumbra, `sun_tan * gap` — how far the blocker stands
//     off the receiver, times the sun's angular radius. This is what makes a
//     crate's shadow crisp at its own feet and soft where it falls away, and it
//     is in metres because an angular size is.
//   * the **screen floor**, `r.shadow_softness` pixels wide, converted through
//     the host's analytic pixel footprint. This is the antialiasing, and it is
//     the half that had to be added.
//
// # Why a rotated disk and not a grid
//
// The floor makes the radius whatever a pixel is worth in texels, which in a far
// cascade is tens of them. A fixed grid sampled that sparsely is regular
// undersampling — the aliasing it was meant to cure, wearing a hat. A Vogel
// spiral rotated per pixel turns it into grain instead, and grain is all that
// `lit` taking one of `taps + 1` values looks like once nothing correlates the
// pattern with the geometry.
//
// # Why there is no fully-lit early-out
//
// The textbook shape returns early when the blocker search finds nothing. No
// finite search is sound: the innermost ring of a Vogel disk of `n` sits at
// `sqrt(0.5/n)` of the radius, so a blocker directly overhead falls straight
// through the middle of it and the early-out punches a lit hole in the deepest
// part of a shadow. Uniform cost is worth more than a branch here, and the pair
// still totals fewer taps than the 5x5 box it replaces.
//
// The comparison stays per tap rather than going through a comparison sampler: a
// `SamplerComparisonState` would mean either a fourth immutable sampler or
// aliasing the existing binding with a second type, it keeps the reverse-Z sense
// visible at the site that depends on it, and it is what makes the per-tap plane
// extrapolation possible at all — a comparison sampler takes one depth for the
// whole kernel. Filtering *after* comparing is also the only order that is
// correct: a filtered depth compared once is a lie about a silhouette, which is
// why the sampler stays `NearestClamp`.
//
// No derivatives anywhere, which is what keeps it safe under the divergent
// per-fragment cascade choice: every tap is an explicit-LOD sample and the pixel
// footprint arrives as an analytic term from the host rather than as an
// `fwidth`.
float pcss(Frame frame, CascadeShape cascade, float3 clip, float2 slope, float footprint,
           float rotation, Texture2D<float4> textures[], SamplerState samplers[])
{
    // The light's projection is orthographic, so its w is 1 and this is already
    // NDC. x and y go from [-1, 1] to [0, 1]; y needs no flip, because
    // `orthographic_reverse_z` is Y-down like every other projection here.
    float2 uv = clip.xy * 0.5 + 0.5;
    uint taps = max(frame.shadow_taps, 4u);
    uint search_taps = max(taps / 4u, 4u);

    // Pixels to texels, through the two conversions that meet here: the host's
    // metres-per-pixel at this depth, and this cascade's own metres-per-texel.
    // Halved because `shadow_softness` is a width and this is a radius.
    float floor_texels = 0.5 * frame.shadow_softness * footprint / cascade.texel_world;

    // Blocker search at the widest a penumbra may get — searching further would
    // only find occluders whose contribution the same cap discards.
    float gap_sum = 0.0;
    float found = 0.0;
    float2 dir = float2(cos(rotation), sin(rotation));
    for (uint i = 0; i < search_taps; ++i)
    {
        float2 offset =
            dir * (spiral_radius(i, search_taps) * frame.shadow_penumbra * frame.shadow_texel);
        dir = spin(dir);
        float caster = textures[cascade.texture]
                           .SampleLevel(samplers[frame.shadow_sampler], uv + offset, 0)
                           .r;
        // Reverse-Z: nearer the light compares greater. Measured along the
        // receiver's own plane, exactly as the filter does below, so a floor's
        // up-slope away from the light is not counted as a thing standing on it.
        float reference = clip.z + max(dot(offset, slope), 0.0);
        float gap = caster - reference;
        if (gap > 0.0)
        {
            gap_sum += gap;
            found += 1.0;
        }
    }

    // The **mean** gap and not the nearest blocker's: a penumbra is cast by
    // everything standing between the point and the sun, and sizing it off the
    // nearest alone makes the width jump wherever one caster passes in front of
    // another. Clip depth to metres through the cascade's own slab depth.
    float gap_metres = found > 0.0 ? (gap_sum / found) * cascade.depth_world : 0.0;
    float penumbra = frame.sun_tan * gap_metres / cascade.texel_world;
    float radius = clamp(max(penumbra, floor_texels), 0.0, frame.shadow_penumbra);

    // The spiral restarts rather than continuing where the search left off, so
    // the two are the same pattern at two counts instead of one long sequence
    // whose second half depends on how many taps the first half happened to use.
    dir = float2(cos(rotation), sin(rotation));
    float lit = 0.0;
    for (uint i = 0; i < taps; ++i)
    {
        float2 offset = dir * (spiral_radius(i, taps) * radius * frame.shadow_texel);
        dir = spin(dir);
        float caster = textures[cascade.texture]
                           .SampleLevel(samplers[frame.shadow_sampler], uv + offset, 0)
                           .r;
        // One-sided on purpose: the plane may promise a tap is *nearer*
        // (excusing the receiver's own up-slope, the acne the kernel
        // manufactures) but never farther, because nothing farther from the
        // light than the fragment can occlude it — a farther prediction only
        // manufactures shadow. Two-sided, the overshoot past the neighbouring
        // surface at a concave corner was half of §6 M20's contact band
        // (`gg-tools shadow-flat` is the measurement).
        lit += clip.z + max(dot(offset, slope), 0.0) >= caster ? 1.0 : 0.0;
    }
    return lit / float(taps);
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
// Nothing here takes a screen-space derivative — the receiver plane comes off
// the normal and the pixel footprint off the host — so the per-fragment cascade
// choice and everything after it may diverge freely; `pcss` samples at an
// explicit LOD.
// `normal` drives the normal-offset bias (the shading normal, as the offset has
// always been scaled); `plane` is the geometric normal the receiver-plane
// extrapolation is taken from — see `Surface::plane` for why they may differ.
// `pixel` is the fragment's position on the screen, which seeds the tap
// rotation. `footprint` is the side of a square covering the same area of *this
// surface* as one screen pixel, in metres — what a penumbra floored in pixels is
// floored against, and the one quantity here that a derivative has to supply.
float sun_shadow(uint64_t frame_address, Frame frame, float3 world, float3 normal,
                 float3 plane, float2 pixel, float footprint,
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
    CascadeShape shape[MAX_CASCADES];
    [ForceUnroll]
    for (uint i = 0; i < MAX_CASCADES; ++i)
    {
        // Clamped rather than skipped: the loop is unrolled to the compile-time
        // maximum, so an unused slot recomputes the last live cascade instead
        // of reading past the block.
        Cascade c = load_cascade(frame_address, min(i, frame.cascade_count - 1));
        clip[i] = to_light_clip(frame, c, world, normal);
        slope[i] = receiver_plane_slope(c, plane);
        shape[i].texture = c.texture;
        shape[i].texel_world = c.texel_world;
        shape[i].depth_world = c.depth_world;
    }

    float rotation = tap_rotation(pixel);

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

    float lit = pcss(frame, shape[chosen], clip[chosen], slope[chosen], footprint, rotation,
                     textures, samplers);

    // Cross-fade into the next cascade over the outer `shadow_blend` of this one.
    // Less load-bearing than it was — a radius floored in *pixels* is the same
    // number of pixels either side of a split, so the softness no longer steps
    // where the texel size does. What survives is the map's own grain changing,
    // which is the part a fade is the right answer to.
    uint next = chosen + 1;
    float border = 1.0 - frame.shadow_blend;
    float edge = cascade_edge(clip[chosen]);
    if (next < frame.cascade_count && edge > border && inside_cascade(clip[next]))
    {
        float t = saturate((edge - border) / max(frame.shadow_blend, 1e-6));
        float far = pcss(frame, shape[next], clip[next], slope[next], footprint, rotation,
                         textures, samplers);
        lit = lerp(lit, far, t);
    }
    return lit;
}

// Every light in the frame, over one surface. `world` is camera-relative, so under
// perspective the eye is the origin and the view vector is just its negation — the
// same property that lets a camera 10^12 m out render without jitter (§1.4).
// `pixel` and `footprint` are threaded through for the shadow filter alone — see
// `sun_shadow` for why a shading term needs the screen grid, and `pixel_footprint`
// for why the caller computes the second of them rather than this.
float3 shade_all(uint64_t frame_address, Frame frame, Surface s, float3 world, float2 pixel,
                 float footprint, Texture2D<float4> textures[], SamplerState samplers[])
{
    // Orthographic has no eye point: every view ray is the camera's axis, so the
    // negation above would turn depth into a specular slide across the screen —
    // foreshortening in the shading, under the one projection that has none. The
    // matrix decides, as it does for the culler (`gg_extract::frustum`'s *Five
    // planes, or six*): row 2 is `backward / depth` under
    // `orthographic_reverse_z` and exactly zero under `perspective_reverse_z`,
    // which is `[0, 0, 0, near]` by construction rather than by rounding.
    float3 axis = frame.view_projection[2].xyz;
    float3 to_eye = dot(axis, axis) > 1e-12 ? normalize(axis) : normalize(-world);
    // The environment, before any light with a direction — image-based since
    // §6 M24, and the flat placeholder in a frame that declared no sky.
    float3 total = ambient_light(frame_address, frame, s, to_eye);
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
                radiance *= sun_shadow(frame_address, frame, world, s.normal, s.plane,
                                       pixel, footprint, textures, samplers);
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
