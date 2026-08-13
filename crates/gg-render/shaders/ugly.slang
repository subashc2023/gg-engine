// The §4.5 v0 pass: one box, tinted by whatever colour the game put in its
// `Renderable` (§6 M5) — and, since M11, shaded by the same model the pack pass
// uses. Two lighting models in one engine would be exactly the thing this
// milestone exists to delete, so the equations live once, in `include/pbr.slang`.
//
// Vertices are *pulled* through a buffer device address (§4.3): nothing binds a
// vertex buffer anywhere in this engine.
//
// # Why this push block has no matrix
//
// Vulkan guarantees 128 bytes of push constants and this pass has one push per
// box, so the object-to-clip product cannot live here beside a position, a
// rotation and a scale. It is assembled in the vertex shader from the camera
// projection the frame buffer carries instead. The pack pass makes the opposite
// trade for a reason it states: it batches, so its instances are in a buffer
// where a precomputed matrix is cheap, and its blessed references depend on the
// low bits of that product. Both prepass and forward here run the *same*
// expression, which is the only agreement `GREATER_OR_EQUAL` needs.

#include "include/pbr.slang"

// The global set (§4.3) — the shadow map is reached through it, like any texture.
[[vk::binding(0, 0)]] Texture2D<float4> g_textures[];
[[vk::binding(2, 0)]] SamplerState g_samplers[];

struct UglyPush
{
    uint64_t vertices; // device address of the vertex array
    uint64_t frame;    // device address of this frame's lighting environment
    float4 tint;       // linear rgb; w unused
    float4 rotation;   // the instance's orientation, xyzw
    float4 offset;     // xyz camera-relative centre; w unused
    float4 scale;      // xyz half-extent, applied before the rotation; w unused
    float4 material;   // x smoothness, y metallic (§6 M24); zw reserved
    uint shadow_view;  // which projection `vs_shadow` uses — see `shadow_view_projection`
}

[[vk::push_constant]]
ConstantBuffer<UglyPush> push;

// HLSL has no `sizeof`, so the stride is written here and asserted by the Rust
// `Vertex` beside it. A drift is a build error, not a garbled mesh.
static const uint VERTEX_STRIDE = 24;

float3 rotate_by(float4 q, float3 v)
{
    float3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

// Positions carry the instance's scale, which shears a normal; the inverse
// transpose of a diagonal `T*R*S` is `R*S^-1`, so a normal divides where a
// position multiplies — `scene.slang`'s vertex stage, on the pass that always
// had meshes to get this wrong on.
//
// Exact for both primitives and free for one of them: a box's face normal is
// axis-aligned, so dividing it by the scale changes only its length and
// `normalize` puts that back. That is why this was a rotation alone until §6
// M26 — correct by luck for every instance the pass had ever drawn, and wrong
// the moment one of them curved.
float3 normal_of(float3 n)
{
    return normalize(rotate_by(push.rotation, n / nonzero_scale(push.scale.xyz)));
}

// Camera-relative world position of one of the primitive's vertices. Shared by all
// three entry points below so the three cannot disagree about where a fragment
// is — which for prepass and forward is a hard requirement, not tidiness.
float3 world_of(uint vid)
{
    float* v = (float*)(push.vertices + uint64_t(vid) * VERTEX_STRIDE);
    return push.offset.xyz + rotate_by(push.rotation, float3(v[0], v[1], v[2]) * push.scale.xyz);
}

struct VOut
{
    float4 pos : SV_Position;
    float3 world : TEXCOORD1;
    float3 normal : NORMAL0;
}

[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    // A `float*` rather than a `Vertex*`: the pointee's layout would then be
    // Slang's to choose (float3 aligns to 16 in std430) while the Rust side is
    // packed `repr(C)`. Scalar strides are the one indexing both agree on.
    float* v = (float*)(push.vertices + uint64_t(vid) * VERTEX_STRIDE);
    Frame frame = load_frame(push.frame);
    float3 world = world_of(vid);
    float3 normal = normal_of(float3(v[3], v[4], v[5]));

    VOut o;
    o.pos = mul(frame.view_projection, float4(world, 1.0));
    o.world = world;
    o.normal = normal;
    // No light_clip: which cascade a point belongs to is a per-fragment
    // question, so the projection happens in `sun_shadow` (§6 M15.3).
    return o;
}

[shader("fragment")]
float4 fs_main(VOut i) : SV_Target
{
    Frame frame = load_frame(push.frame);
    // No normal map and no occlusion map: a box has neither, so the tangent
    // frame this pass would need does not exist and is not invented.
    // Smoothness inverts to roughness here and nowhere else — see
    // `Renderable::smoothness` for why the boundary carries the other one.
    Surface s = make_surface(normalize(i.normal), push.tint.rgb, push.material.y,
                             1.0 - push.material.x, 1.0);
    // The footprint before anything else touches it: `pixel_footprint` takes
    // derivatives, so it has to be reached by every fragment of a quad.
    float footprint = pixel_footprint(i.world);
    return float4(shade_all(push.frame, frame, s, i.world, i.pos.xy, footprint, g_textures,
                            g_samplers),
                  1.0);
}

// The depth prepass (§4.5 v1). It shares `UglyPush` and `world_of` with the pass
// above rather than declaring a smaller block, because the forward pass must
// re-draw *exactly* where this one put a fragment — a second position path is a
// second chance to disagree, and `GREATER_OR_EQUAL` only forgives a bit-identical
// depth.
[shader("vertex")]
float4 vs_depth(uint vid: SV_VertexID) : SV_Position
{
    return mul(load_frame(push.frame).view_projection, float4(world_of(vid), 1.0));
}

// The probe pass (§6 M36): the same geometry, through one probe face's
// projection, shaded from the probe rather than from the camera.
//
// `vs_probe` is `vs_shadow` plus the two interpolants shading needs, and takes
// its projection through the same `shadow_view` index — `shadow_view_projection`
// is what resolves a cascade, a lamp face or a probe face out of one number.
// Depth is *written* here, unlike the forward pass: a probe face has no prepass
// in front of it, and six per probe would be six times a pass whose whole point
// is that it is small.
[shader("vertex")]
VOut vs_probe(uint vid: SV_VertexID)
{
    float* v = (float*)(push.vertices + uint64_t(vid) * VERTEX_STRIDE);
    float3 world = world_of(vid);
    VOut o;
    o.pos = mul(shadow_view_projection(push.frame, push.shadow_view), float4(world, 1.0));
    o.world = world;
    o.normal = normal_of(float3(v[3], v[4], v[5]));
    return o;
}

// Radiance toward the probe in rgb, and how far away this fragment is in `w` —
// **signed**, negative on a back face.
//
// The sign is the whole of "is this probe inside something": a probe in open air
// sees the outsides of things, a probe buried in a wall sees the inside of it,
// and the fraction of its sphere that came back negative is the number the
// integration writes into the state texel. Cheaper and more direct than any
// geometric test the host could do, because the rasterizer already knows.
[shader("fragment")]
float4 fs_probe(VOut i, bool front: SV_IsFrontFace) : SV_Target
{
    Frame frame = load_frame(push.frame);
    float3 probe =
        load_probe_position(load_probe_views(push.frame),
                            (push.shadow_view - PROBE_VIEW_BASE) / PROBE_FACES);
    float3 offset = probe - i.world;
    float distance = length(offset);
    Surface s = make_surface(normalize(i.normal), push.tint.rgb, push.material.y,
                             1.0 - push.material.x, 1.0);
    // No `pixel_footprint`: that is a screen-derivative estimate of a filter
    // width, and the screen here is an eight-texel face. Zero asks the shadow
    // filter for its floor, which is the right answer at this resolution.
    float3 radiance = shade_all_from(push.frame, frame, s, i.world,
                                     offset / max(distance, 1e-5), i.pos.xy, 0.0, false,
                                     g_textures, g_samplers);
    return float4(radiance, front ? distance : -distance);
}

// The shadow pass: the same geometry through one shadow view's projection,
// chosen by the push rather than by the frame, because the pass is recorded once
// per view. No normal offset here — the bias belongs on the lookup, so the depth
// this records stays the geometry's own.
[shader("vertex")]
float4 vs_shadow(uint vid: SV_VertexID) : SV_Position
{
    return mul(shadow_view_projection(push.frame, push.shadow_view), float4(world_of(vid), 1.0));
}

[shader("fragment")]
void fs_depth()
{
}
