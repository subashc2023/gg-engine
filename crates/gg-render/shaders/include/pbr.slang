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
//   [FA19] Fdez-Aguera, "A Multiple-Scattering Microfacet Model for Real-Time
//           Image-Based Lighting", JCGT 8(1) — the energy a single-scatter
//           split-sum loses, returned in closed form from the same two numbers
//           the split-sum already computed.
//   [Lag12] Lagarde & Zanuttini, "Local Image-based Lighting With
//           Parallax-corrected Cubemap", SIGGRAPH 2012 talks — intersect the
//           reflected ray with the environment's own volume and re-aim from its
//           centre, which is what makes a bounded environment a *place*.
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
static const uint FRAME_STRIDE = 4368;
static const uint LIGHT_STRIDE = 48;
static const uint CASCADE_STRIDE = 80;
// Where the cascade array starts inside the frame block.
static const uint CASCADE_BASE = 976;
// Coefficients in the order-2 environment projection (§6 M24), and where they
// start. Must equal `sky.rs`'s SH_COEFFICIENTS.
static const uint SH_COEFFICIENTS = 9;
// The environment array (§6 M28). Must equal `gg_extract::MAX_SKY`; the loop
// below is bounded by it and a mismatch would read past the block.
static const uint MAX_SKY = 4;
static const uint ENV_STRIDE = 192;

// The immutable sampler set's LinearClamp slot — `gg_rhi::Sampler::index`, and
// the one this file names rather than reading out of the frame block: the
// shadow sampler is a *choice* the block records, and this is a fixed property
// of how an octahedral chain must be filtered.
static const uint SAMPLER_LINEAR_CLAMP = 1;
// Where the environment array starts, and therefore where the innermost
// environment's coefficients start — `sh` is the first field of an environment
// precisely so M24's read is still the read.
static const uint ENV_BASE = 208;
// Must equal `lighting.rs`'s MAX_CASCADES: the loops below are unrolled over it
// and a mismatch would read past the block.
static const uint MAX_CASCADES = 4;

// The casting lamps (§6 M31). Must equal `lamp.rs`'s MAX_LAMPS and FACES; a lamp
// is six 4x4 matrices and nothing else, because the light record beside it
// already carries the position and range this needs.
static const uint MAX_LAMPS = 8;
static const uint LAMP_FACES = 6;
static const uint LAMP_STRIDE = 384;
static const uint LAMP_BASE = 1296;
// A shadow draw's `shadow_view` index: cascades first, then lamp faces. One
// number rather than a second push field, so the shadow vertex shader takes one
// branch on a value that is uniform across the draw.
static const uint LAMP_VIEW_BASE = MAX_CASCADES;

// The froxel grid (§6 M30). Must equal `cluster.rs`'s GRID, which is where the
// dimensions are argued; here they are only the arithmetic that finds a list.
static const uint CLUSTER_X = 16;
static const uint CLUSTER_Y = 8;
static const uint CLUSTER_Z = 24;
// Lights the frame's array holds — `gg_extract::MAX_DIRECTIONAL + MAX_POINT`,
// and the only one of these five a stride change cannot derive.
static const uint MAX_LIGHTS = 260;
// Where the froxel ranges start, and where the indices they point into do.
// Derived rather than written down, so a block that grows moves both.
static const uint CLUSTER_BASE = FRAME_STRIDE + MAX_LIGHTS * LIGHT_STRIDE;
static const uint INDEX_BASE = CLUSTER_BASE + CLUSTER_X * CLUSTER_Y * CLUSTER_Z * 8;

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
    uint environment_count;    // live entries in the environment array; 0 is "no sky" (§6 M28)
    uint directional_count;    // suns at the front of the light array (§6 M30); the rest are froxel-listed
    float3 view_forward;       // unit; what a camera-relative position projects onto to get view depth
    float cluster_scale;       // slice = log2(depth) * scale + bias, floored
    float cluster_bias;
    float2 cluster_tile;       // froxels per pixel, across and down
    uint lamp_count;           // point lights that cast, immediately after the suns (§6 M31)
    uint lamp_texture;         // the face atlas's bindless slot; unread at lamp_count 0
    float2 lamp_uv_scale;      // tile space -> atlas uv: 1/6 across, 1/lamps down
    float lamp_inv_tile;       // one face texel in tile space
    float lamp_normal_bias;    // *face* texels of normal offset, angle-scaled like the sun's
    float lamp_depth_bias;     // the angle-free part, also in face texels
    uint lamp_taps;            // taps in the lamp filter; its own knob because it is paid per lamp
    uint shading;              // SHADING_* bits; the frame block's last padding word (§6 M33)
}

// What `Frame::shading` carries. Mirrored in `lighting.rs`, which is where the
// CVars behind them are read — a bit crosses, a knob does not.
static const uint SHADING_MULTISCATTER = 1u;
static const uint SHADING_LOBE = 2u;

// One environment: what it radiates, and the volume it radiates in (§6 M28).
//
// `sh` is loaded separately for `load_sh`'s measured reason — 27 scalars a
// fragment is worth a branch — so this holds everything needed to decide
// *whether* to load them.
struct Environment
{
    float3 offset;         // the volume's centre, camera-relative
    float fade;            // metres outside the box it falls to nothing over; 0 is a hard edge
    float3 half_extent;    // zero is unbounded, which weighs 1 everywhere
    float radiance_scale;  // Sky::intensity, applied to the chain's texels
    uint radiance_texture; // the prefiltered chain's bindless slot (§6 M27)
    uint radiance_levels;  // levels in it, and the switch: 0 is "SH only"
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
    frame.environment_count = u[34];
    frame.directional_count = u[35];
    frame.view_forward = float3(f[36], f[37], f[38]);
    frame.cluster_scale = f[39];
    frame.cluster_bias = f[40];
    frame.cluster_tile = float2(f[41], f[42]);
    frame.lamp_count = u[43];
    frame.lamp_texture = u[44];
    frame.lamp_uv_scale = float2(f[45], f[46]);
    frame.lamp_inv_tile = f[47];
    frame.lamp_normal_bias = f[48];
    frame.lamp_depth_bias = f[49];
    frame.lamp_taps = u[50];
    frame.shading = u[51];
    return frame;
}

// One face of one lamp's projection, straight out of the block.
//
// Read a face at a time and never six: a fragment picks its face by major axis
// before it asks, so five of them would be sixty scalar loads to discard.
float4x4 load_lamp_face(uint64_t frame_address, uint lamp, uint face)
{
    float* f = (float*)(frame_address + LAMP_BASE + lamp * LAMP_STRIDE + face * 64);
    return float4x4(f[0], f[1], f[2], f[3],
                    f[4], f[5], f[6], f[7],
                    f[8], f[9], f[10], f[11],
                    f[12], f[13], f[14], f[15]);
}

// Which of a lamp's six faces a direction falls in: `2 * major axis + negative`.
//
// The same table `lamp.rs`'s `AXES` holds, and the agreement between them is the
// whole of the convention — these are six ordinary 2D maps, so there is no cube
// standard to defer to and nothing that would catch the two drifting apart
// except the picture. `to` is light → fragment.
uint lamp_face(float3 to)
{
    float3 a = abs(to);
    if (a.x >= a.y && a.x >= a.z)
    {
        return to.x < 0.0 ? 1 : 0;
    }
    if (a.y >= a.z)
    {
        return to.y < 0.0 ? 3 : 2;
    }
    return to.z < 0.0 ? 5 : 4;
}

// The froxel this fragment is in: its tile from `SV_Position`, its slice from
// the view depth. `cluster.rs`'s `Grid::slice_of` is the host's copy of the
// second half, and the two are reconciled by widening every sphere there rather
// than by hoping two `log2` implementations agree on a boundary.
uint cluster_of(Frame frame, float3 world, float2 pixel)
{
    float depth = max(dot(world, frame.view_forward), 1e-6);
    int k = int(floor(log2(depth) * frame.cluster_scale + frame.cluster_bias));
    uint i = uint(clamp(int(pixel.x * frame.cluster_tile.x), 0, int(CLUSTER_X) - 1));
    uint j = uint(clamp(int(pixel.y * frame.cluster_tile.y), 0, int(CLUSTER_Y) - 1));
    // Clamped rather than discarded: the last slice reaches infinity and the
    // first starts at the near plane, so a depth outside them belongs to the
    // froxel at the end — whose list was built for a froxel that reaches it.
    uint slice = uint(clamp(k, 0, int(CLUSTER_Z) - 1));
    return (slice * CLUSTER_Y + j) * CLUSTER_X + i;
}

// One froxel's run: where it starts in the index array, and how long it is.
uint2 load_cluster(uint64_t frame_address, uint cluster)
{
    uint* u = (uint*)(frame_address + uint64_t(CLUSTER_BASE) + uint64_t(cluster) * 8);
    return uint2(u[0], u[1]);
}

// One entry of a run: an index into the frame's light array.
uint load_cluster_index(uint64_t frame_address, uint at)
{
    uint* u = (uint*)(frame_address + uint64_t(INDEX_BASE));
    return u[at];
}

Environment load_environment(uint64_t frame_address, uint index)
{
    uint64_t at = frame_address + uint64_t(ENV_BASE) + uint64_t(index) * ENV_STRIDE;
    // Past the coefficients: they are the first 144 bytes and are read only if
    // this record says the environment applies here.
    float* f = (float*)(at + 144);
    uint* u = (uint*)(at + 144);
    Environment e;
    e.offset = float3(f[0], f[1], f[2]);
    e.fade = f[3];
    e.half_extent = float3(f[4], f[5], f[6]);
    e.radiance_scale = f[7];
    e.radiance_texture = u[8];
    e.radiance_levels = u[9];
    return e;
}

// How much of `e` applies at `world`, camera-relative — 1 inside the box, 0
// beyond `fade` metres outside it, smoothstepped between.
//
// **Duplicated in `gg_extract::ExtractedSky::weight_at`**, which decides which
// background to draw where this decides how to shade. The two never meet at
// runtime, so nothing can assert them equal; both are three lines of the same
// arithmetic, and a drift shows as a reflection that does not match the sky
// behind it.
float environment_weight(Environment e, float3 world)
{
    if (all(e.half_extent == 0.0))
    {
        return 1.0;
    }
    float3 outside = max(abs(world - e.offset) - e.half_extent, 0.0);
    float distance = length(outside);
    // The comparison and not a division: a zero `fade` is a hard edge, and
    // dividing by it would be a NaN weight and a black room.
    if (e.fade <= 0.0)
    {
        return distance <= 0.0 ? 1.0 : 0.0;
    }
    return 1.0 - smoothstep(0.0, e.fade, distance);
}

// `direction` re-aimed at where it actually leaves `e`'s volume (§6 M29).
//
// An environment is a function of direction alone, which is a statement that its
// contents are infinitely far away. That is true of a sky and false of a room:
// inside a bounded volume the wall a reflection lands on is metres away, so two
// points on a floor reflecting the *same* direction must see different parts of
// it. Intersecting the ray with the volume and re-aiming from the box centre —
// where the chain was integrated from — is [Lag12]'s correction, and it is what
// makes a reflection slide along a wall instead of riding the camera.
//
// Unbounded environments are returned unchanged: infinitely far away is what
// they are, and there is no box to leave. Returned **unnormalized** — every
// consumer normalizes — because the exit point relative to the centre is already
// the direction wanted.
float3 parallax_direction(Environment e, float3 world, float3 direction)
{
    if (all(e.half_extent == 0.0))
    {
        return direction;
    }
    // The reciprocal is taken against a floored magnitude rather than the
    // component: a ray exactly parallel to a slab gives 0 * inf = NaN in the
    // products below, and a NaN t survives `min` on some drivers and not others.
    float3 inverse = select(direction >= 0.0, 1.0, -1.0) / max(abs(direction), 1e-5);
    float3 local = world - e.offset;
    float3 far = max((e.half_extent - local) * inverse, (-e.half_extent - local) * inverse);
    float t = min(min(far.x, far.y), far.z);
    // Outside the box and pointing away from it: there is no exit point, and the
    // honest answer is the uncorrected direction rather than a point behind the
    // viewer. Written as `!(t > 0)` so a NaN that got this far falls here too.
    if (!(t > 0.0))
    {
        return direction;
    }
    return local + direction * t;
}

// One environment's coefficients, read out of the same block — but **not** by
// `load_frame`, and that is a measured decision. Loading them eagerly is 27
// scalar loads a fragment for the environment's sake in every scene that has
// none, and it cost the frame macro 6 % on the pinned rasterizer with nothing
// to show for it. `Frame::environment_count` is uniform across a draw, so the
// branch that guards this splits no quad and a scene without a sky pays nothing.
//
// **Accumulated with a weight rather than assigned** (§6 M28). An order-2
// projection is linear, so the weighted sum of two environments' coefficients is
// the projection of the weighted sum of the two environments — not an
// approximation of it. That is the whole of why blending environments costs nine
// multiply-adds and one evaluation instead of a second lighting path, and it is
// why a doorway is a fade instead of a pop.
void accumulate_sh(uint64_t address, uint index, float weight, inout float3 sh[SH_COEFFICIENTS])
{
    float* f = (float*)(address + uint64_t(ENV_BASE) + uint64_t(index) * ENV_STRIDE);
    [ForceUnroll]
    for (uint i = 0; i < SH_COEFFICIENTS; ++i)
    {
        sh[i] += weight * float3(f[i * 4], f[i * 4 + 1], f[i * 4 + 2]);
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

// The projection one shadow draw renders through: a cascade below
// `LAMP_VIEW_BASE`, a lamp face at or above it.
//
// One index and one branch, rather than a second push field the forward passes
// would carry and never read. The branch is uniform across a draw — the shadow
// pass is recorded once per view — so it costs a shadow vertex nothing.
float4x4 shadow_view_projection(uint64_t frame_address, uint view)
{
    if (view < LAMP_VIEW_BASE)
    {
        return load_cascade(frame_address, view).view_projection;
    }
    uint at = view - LAMP_VIEW_BASE;
    return load_lamp_face(frame_address, at / LAMP_FACES, at % LAMP_FACES);
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
//
// `compensation` is what the multiple bounces this BRDF does not model would
// have returned (§6 M33) — 1 when the correction is off, and computed once per
// fragment by the caller rather than here, because it depends on the view and
// the roughness and on nothing this light brings.
float3 shade(Surface s, float3 to_eye, float3 to_light, float3 radiance, float3 compensation)
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
    // Single-scatter GGX, then the bounces it dropped. D*V*F counts a ray that
    // left the surface once; a rough microsurface shadows itself, and the ray
    // that hits a second facet and *then* leaves is energy this term throws away
    // (§6 M33). `compensation` is the tint-correct scale that puts it back, and
    // it is exactly 1 at roughness 0 where there is no second facet to hit.
    float3 specular = fresnel * d_ggx(n_dot_h, s.alpha)
                      * v_smith_correlated(n_dot_v, n_dot_l, s.alpha) * compensation;
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

// The octahedral uv a direction lands at — the inverse of `ggc`'s
// `octahedral_decode`, folded about **y** for the same reason.
//
// The two are written twice and never meet, exactly as the SH basis is. What
// keeps them honest is that each is the textbook mapping and neither is derived
// from the other; a disagreement would show as an environment reflected through
// a crease, which is the one artifact here nobody could mistake for shading.
float2 octahedral_uv(float3 d)
{
    float3 n = d / (abs(d.x) + abs(d.y) + abs(d.z));
    float2 p = float2(n.x, n.z);
    // The lower hemisphere folds outward across the diagonals. `sign` is wrong
    // here at exactly zero — it returns 0 and would collapse the fold to the
    // origin — so the comparison is written out.
    float2 folded = (1.0 - abs(float2(n.z, n.x))) * select(p >= 0.0, 1.0, -1.0);
    p = n.y >= 0.0 ? p : folded;
    return p * 0.5 + 0.5;
}

// The prefiltered environment along `direction` at lobe width `alpha` — the
// split-sum's *first* integral, as `ggc` actually computed it (§6 M27).
//
// Roughness picks the mip, linearly across the whole chain, because that is the
// axis the levels were filtered along. `sqrt(alpha)` and not `alpha`: the chain
// is parameterized by roughness and `alpha` is its square, and getting this
// backwards puts every level's blur on the wrong surface — a subtle wrongness
// that reads as "the rough things are too shiny" rather than as a bug.
//
// The clamp at half a texel from each edge is the seam, answered at the only
// place it can still bite: every level was integrated over the sphere so the
// texels are right, but a bilinear tap centred *on* the outer edge would blend
// across the fold into the opposite hemisphere.
float3 prefiltered_radiance(Environment e, float3 direction, float alpha,
                            Texture2D<float4> textures[], SamplerState samplers[])
{
    float level = sqrt(alpha) * float(e.radiance_levels - 1);
    float2 uv = octahedral_uv(normalize(direction));
    // Scaled here and not at import: the pack holds the panorama's own absolute
    // radiance, and exposure is the world's. The SH beside it is scaled on the
    // host, and the two must agree or a mirror stops matching its room.
    return textures[e.radiance_texture]
               .SampleLevel(samplers[SAMPLER_LINEAR_CLAMP], uv, level)
               .rgb
           * e.radiance_scale;
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

// How much of the environment a surface of this roughness reflects when nothing
// absorbs any of it — the split-sum's second integral at `f0 = 1`, which is what
// `a + b` is by construction.
//
// This is the number the whole of §6 M33 turns on. It is the *single-scatter*
// directional albedo, so it is below 1 wherever the microsurface can shadow
// itself: ~1.0 at roughness 0 and ~0.68 at roughness 1, head-on. A renderer that
// stops there has silently absorbed the difference, and `gg-tools furnace`
// measures exactly this curve by rendering it.
float directional_albedo(float2 ab)
{
    // Floored because it divides below, and because the analytic fit is free to
    // return a hair under zero at a corner of its domain where the table it
    // approximates does not.
    return max(ab.x + ab.y, 1e-3);
}

// The scale that returns what a single-scatter lobe lost — [FA19], in the form
// [LdR14] gives for direct lights.
//
// `1/E` is the whole idea: if a white surface only gave back `E` of what it
// received, the bounces that never happened were worth `1/E` of what did. The
// `f0` factor is what makes it tint-correct — the light went round again and was
// Fresnel-filtered a second time, so a copper surface's returned energy is
// copper-coloured and a dielectric's is nearly white and nearly nothing.
float3 energy_compensation(float3 f0, float2 ab)
{
    return 1.0 + f0 * (1.0 / directional_albedo(ab) - 1.0);
}

// Where the lobe actually points — [LdR14] §4.9.3.
//
// The chain was prefiltered at `n = v = r` ([Kar13]), which is a statement that
// the lobe is round and centred on the mirror direction. It is round only
// head-on: toward the silhouette the GGX lobe stretches along the surface and
// its centre of mass slides back toward the normal, so a lookup along `reflect`
// asks the chain for a direction the lobe is no longer centred on. Two lines
// move it, and at alpha 0 the interpolation returns the mirror direction
// exactly, which is the one case that was already right.
//
// **`alpha`, not the perceptual roughness, and that was measured rather than
// read.** The published listing says "roughness" and its own text uses that word
// for both quantities; the two differ by a lot at the smooth end, where the
// perceptual reading pulls a roughness-0.2 surface twelve per cent off its
// mirror direction and this one pulls it two. `gg-tools furnace` prints both
// against an importance-sampled reference over demo 06's panorama, and this is
// the column that won (§6 M33).
float3 dominant_direction(float3 normal, float3 reflected, float alpha)
{
    float smoothness = saturate(1.0 - alpha);
    return normalize(lerp(normal, reflected, smoothness * (sqrt(smoothness) + alpha)));
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

// Everything the environments contribute, or the flat ambient term when there
// are none — the one place the two are chosen between, so a scene blessed before
// §6 M24 and a scene with a sky do not each need their own path through
// `shade_all`.
//
// # Compositing, and why it is front to back (§6 M28)
//
// `frame.environments` is ordered innermost first, so walking it and spending a
// budget of weight is exactly "the smallest volume containing this fragment
// wins, and what it leaves over falls through to whatever contains *it*". The
// world's sky is unbounded and last, which makes it the fallback without being a
// special case. The loop stops when the budget runs out, so a fragment deep
// inside a room reads one environment and a fragment in a doorway reads two.
//
// Two accumulators and not one, because the specular half has two sources that
// cannot be summed before evaluation: an environment with a prefiltered chain
// contributes a *sample*, and one without contributes a three-band convolution.
// Both are linear in what they are given, so the chainless ones are accumulated
// as coefficients and evaluated once at the end — which is why blending N
// environments still costs one `sh_irradiance` and one `sh_radiance`.
float3 ambient_light(uint64_t frame_address, Frame frame, Surface s, float3 world, float3 to_eye,
                     Texture2D<float4> textures[], SamplerState samplers[])
{
    if (frame.environment_count == 0)
    {
        // §6 M11's placeholder, unchanged: a flat term times albedo, occluded.
        return frame.ambient * s.diffuse * s.occlusion;
    }
    float n_dot_v = max(dot(s.normal, to_eye), 1e-4);
    float roughness = sqrt(s.alpha);
    // `reflect` takes the incident direction, which is the negation of the one
    // toward the eye — the sign that is wrong in half the implementations of
    // this on the internet and which shows up as a reflection that slides the
    // wrong way.
    float3 reflected = reflect(-to_eye, s.normal);
    // Where the chain is *asked*, which is not the mirror direction (§6 M33).
    // Corrected before the parallax intersection and not after: the ray whose
    // exit point the room wants is the one the lobe actually looks along, and
    // doing it the other way round re-aims a direction that has already been
    // turned into a point on a wall.
    float3 lobe = (frame.shading & SHADING_LOBE) != 0
                      ? dominant_direction(s.normal, reflected, s.alpha)
                      : reflected;

    float3 sh[SH_COEFFICIENTS];
    float3 sh_specular[SH_COEFFICIENTS];
    [ForceUnroll]
    for (uint c = 0; c < SH_COEFFICIENTS; ++c)
    {
        sh[c] = float3(0.0, 0.0, 0.0);
        sh_specular[c] = float3(0.0, 0.0, 0.0);
    }
    float3 sampled = float3(0.0, 0.0, 0.0);
    float budget = 1.0;
    for (uint i = 0; i < frame.environment_count && i < MAX_SKY; ++i)
    {
        if (budget <= 1e-3)
        {
            break;
        }
        Environment e = load_environment(frame_address, i);
        float weight = min(environment_weight(e, world), 1.0) * budget;
        if (weight <= 0.0)
        {
            continue;
        }
        budget -= weight;
        accumulate_sh(frame_address, i, weight, sh);
        // The chain when there is one, the three-band convolution when there is
        // not. The fallback is not a degraded path — it is exact for a gradient,
        // which is the only environment that has no chain (§6 M24).
        //
        // Parallax-corrected on this side only, and that is a claim rather than
        // an omission (§6 M29): a chain is an image *of a place* and has a wall
        // to slide against, while a gradient is a direction function with no
        // spatial content at all — box-projecting one would invent detail it
        // does not have, and would cost the chainless environments the single
        // evaluation that makes blending them free.
        if (e.radiance_levels > 0)
        {
            float3 aim = parallax_direction(e, world, lobe);
            sampled += weight * prefiltered_radiance(e, aim, s.alpha, textures, samplers);
        }
        else
        {
            accumulate_sh(frame_address, i, weight, sh_specular);
        }
    }

    // The two quantities everything below is built from: the average radiance a
    // Lambertian surface sees, and the radiance the specular lobe sees.
    //
    // No parallax on the first and there cannot be: irradiance is a hemispherical
    // integral rather than a direction, so there is no ray to intersect — what
    // corrects a diffuse term for position is a field of probes, not a box.
    // `s.diffuse` is albedo with metallic already folded out, so a conductor's
    // diffuse term is zero here by construction.
    float3 irradiance = sh_irradiance(sh, s.normal) / PI;
    // The mirror direction and not `lobe`: this is the three-band convolution a
    // chainless environment contributes, and §6 M29's argument for declining
    // parallax on a gradient is the same one here. A function with no detail at
    // the scale the correction moves cannot be improved by moving it, and every
    // gradient-lit scene would have shifted for nothing.
    float3 radiance = sampled + sh_radiance(sh_specular, reflected, s.alpha);

    // The split-sum's second integral: how much of that radiance comes back, and
    // tinted how. `f0 * a + b` is `FssEss` — the *single-scatter* answer, which
    // is where every renderer of this class used to stop.
    float2 ab = env_brdf(roughness, n_dot_v);
    float3 fss_ess = s.f0 * ab.x + ab.y;
    float3 diffuse;
    float3 specular;
    if ((frame.shading & SHADING_MULTISCATTER) != 0)
    {
        // [FA19], and it is bookkeeping rather than a model: `1 - Ess` is the
        // fraction the single-scatter lobe dropped, `Favg` is what a bounce
        // averaged over the hemisphere costs in Fresnel, and `Fms` is the sum of
        // the geometric series of surviving bounces. The energy comes back as a
        // *hemispherical* term — it left in every direction, so what it re-reads
        // is the irradiance and not the lobe's own radiance.
        float ems = 1.0 - directional_albedo(ab);
        float3 favg = s.f0 + (1.0 - s.f0) / 21.0;
        float3 fms = fss_ess * favg / max(1.0 - ems * favg, 1e-4);
        float3 multiscatter = fms * ems;
        specular = fss_ess * radiance + multiscatter * irradiance;
        // The diffuse lobe gets what the specular one demonstrably did *not*
        // take, which is the part this replaces rather than corrects: the old
        // `1 - F` was a plausible split, and this is the same split derived from
        // the energy actually accounted for above. A white furnace is what tells
        // the two apart — the old one does not close, and this one does.
        diffuse = s.diffuse * max(1.0 - fss_ess - multiscatter, 0.0) * irradiance;
    }
    else
    {
        // Pre-§6 M33 exactly, so `r.multiscatter 0` is the old renderer and not a
        // reconstruction of it — including this `1 - F` split, which is the term
        // the branch above stops using.
        specular = fss_ess * radiance;
        diffuse = (1.0 - f_schlick_ambient(s.f0, n_dot_v, roughness)) * s.diffuse * irradiance;
    }

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

// Radius of the lamp filter, in face texels. At most `lamp.rs`'s GUTTER, which
// is what lets the clamp below be lossless: every tap of a fragment that picked
// this face reads depth this face rendered, never its neighbour's.
static const float LAMP_FILTER_RADIUS = 1.5;

// How much of lamp `lamp` reaches this fragment, in [0, 1] (§6 M31).
//
// # What this does not do that `sun_shadow` does
//
// No cascade choice — a lamp's six faces cover every direction at one
// resolution, and which face is a property of the direction rather than of the
// view. No contact hardening: PCSS sizes a penumbra from a blocker search, and
// four lamps' worth of searches is four times the cost of the sun's for a
// softness a lamp with no angular size does not have. And **no receiver-plane
// extrapolation** — `receiver_plane_slope` is derived from an orthographic
// slab's mutually-orthogonal rows, which a perspective face does not have, so
// what carries the kernel's own acne here is the normal offset alone. That is
// the trade a small radius buys: 1.5 texels of filter is close enough to the
// depth the fragment recorded that the plane term has little left to correct.
float lamp_shadow(uint64_t frame_address, Frame frame, uint lamp, Light light, float3 world,
                  float3 normal, float rotation, Texture2D<float4> textures[],
                  SamplerState samplers[])
{
    float3 to = world - light.position;
    uint face = lamp_face(to);
    // The face's own axis distance, which is what its 90 degrees are measured
    // from — a face spans 2*major metres across `tile` texels at that distance,
    // so a texel is worth that much world here. Growing with distance is the
    // difference from a cascade, whose texel is one size across its whole slab.
    float major = max(max(abs(to.x), abs(to.y)), abs(to.z));
    float texel_world = 2.0 * major * frame.lamp_inv_tile;
    // `to` points away from the light, so the direction *to* the light is its
    // negation — the same n_dot_l the shading term uses, and the same
    // sin(incidence) scaling `to_light_clip` explains at length.
    float n_dot_l = dot(normal, -normalize(to));
    float sin_incidence = sqrt(saturate(1.0 - n_dot_l * n_dot_l));
    float offset = texel_world * (frame.lamp_depth_bias + frame.lamp_normal_bias * sin_incidence);

    float4 clip = mul(load_lamp_face(frame_address, lamp, face),
                      float4(world + normal * offset, 1.0));
    // Behind the face's own eye. Only reachable when the normal offset pushed a
    // fragment at the light's own position through it, and lit is the answer
    // that does not put a black speck at a lamp's centre.
    if (clip.w <= 0.0)
    {
        return 1.0;
    }
    float3 ndc = clip.xyz / clip.w;
    float2 centre = ndc.xy * 0.5 + 0.5;

    // Half a texel in from the tile's edge, and the gutter is what makes that
    // inset land inside the face's own 90 degrees rather than short of it.
    float inset = 0.5 * frame.lamp_inv_tile;
    float2 origin = float2(float(face), float(lamp));
    uint taps = max(frame.lamp_taps, 4u);
    float2 dir = float2(cos(rotation), sin(rotation));
    float lit = 0.0;
    for (uint i = 0; i < taps; ++i)
    {
        float2 offset_uv =
            dir * (spiral_radius(i, taps) * LAMP_FILTER_RADIUS * frame.lamp_inv_tile);
        dir = spin(dir);
        float2 uv = clamp(centre + offset_uv, inset, 1.0 - inset);
        float caster = textures[frame.lamp_texture]
                           .SampleLevel(samplers[frame.shadow_sampler],
                                        (origin + uv) * frame.lamp_uv_scale, 0)
                           .r;
        // Reverse-Z, exactly as the sun's filter compares: nearer the light is
        // greater, and a tap at least as near as the recorded depth is lit.
        lit += ndc.z >= caster ? 1.0 : 0.0;
    }
    return lit / float(taps);
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
    float3 total = ambient_light(frame_address, frame, s, world, to_eye, textures, samplers);
    // What the direct lobes below lost to bounces they do not model (§6 M33).
    //
    // Once here rather than inside `shade`, because it is a function of the view
    // and the roughness and of nothing a light brings — a scene with twenty
    // lamps on a froxel would otherwise evaluate the same fit twenty times. It
    // is a second `env_brdf` after `ambient_light`'s and deliberately not shared
    // with it: that one returns early in every frame that declared no sky, and a
    // sun still lights a surface there.
    float3 compensation = float3(1.0, 1.0, 1.0);
    if ((frame.shading & SHADING_MULTISCATTER) != 0)
    {
        float n_dot_v = max(dot(s.normal, to_eye), 1e-4);
        compensation = energy_compensation(s.f0, env_brdf(sqrt(s.alpha), n_dot_v));
    }
    // The suns, by count. A light with no position reaches every fragment, so a
    // froxel list holding one would be the same answer stored three thousand
    // times — they are the front of the array instead (§6 M30), which is where
    // `Extracted::append_lights` puts them.
    for (uint i = 0; i < frame.directional_count; ++i)
    {
        Light light = load_light(frame_address, i);
        float3 radiance = light.color * light.intensity;
        // The *first* directional light casts, and only it: a second shadow map
        // is a second pass and a cascade decision, and cascades are P1 (§6 M11).
        // The rest still light, they just do not occlude.
        if (i == 0)
        {
            radiance *= sun_shadow(frame_address, frame, world, s.normal, s.plane,
                                   pixel, footprint, textures, samplers);
        }
        total += shade(s, to_eye, -light.direction, radiance, compensation);
    }
    // And the lamps this froxel was handed. Ascending, which is what makes this
    // sum a *subsequence* of the pre-M30 loop's rather than a reordering of it —
    // the falloff is exactly zero past `range`, so what was dropped was adding
    // zero and the two sums agree bit for bit.
    uint2 run = load_cluster(frame_address, cluster_of(frame, world, pixel));
    // The lamps that cast are the front of the point half (§6 M31), so "does
    // this one occlude" is a comparison against a count and not a search — the
    // same shape the suns above are found by, one array segment along.
    uint lamp_end = frame.directional_count + frame.lamp_count;
    float rotation = tap_rotation(pixel);
    for (uint n = 0; n < run.y; ++n)
    {
        uint index = load_cluster_index(frame_address, run.x + n);
        Light light = load_light(frame_address, index);
        float3 offset = light.position - world;
        float distance_squared = dot(offset, offset);
        float3 radiance = light.color * light.intensity;
        float falloff = point_falloff(distance_squared, light.range);
        radiance *= falloff;
        // Guarded on the falloff as well as the count: past `range` the term is
        // exactly zero, so shadowing it would be six texture reads to scale a
        // zero — and it is exactly zero rather than nearly, which is the
        // property §6 M30's froxel lists are built on.
        if (index < lamp_end && falloff > 0.0)
        {
            radiance *= lamp_shadow(frame_address, frame, index - frame.directional_count, light,
                                    world, s.normal, rotation, textures, samplers);
        }
        total += shade(s, to_eye, offset * rsqrt(max(distance_squared, 1e-8)), radiance,
                       compensation);
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
