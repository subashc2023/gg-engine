// Ambient occlusion from the depth prepass (§6 M35).
//
// The prepass has written a depth buffer since M6 and nothing had ever read it.
// This is its consumer: the fraction of the hemisphere a fragment can see,
// multiplied into the ambient and indirect terms — which arrive unoccluded
// otherwise on every surface whose material carries no baked occlusion map,
// meaning every `Renderable` in the tree.
//
// **The algorithm is GTAO** (Jimenez et al., "Practical Real-Time Strategies for
// Accurate Indirect Occlusion", SIGGRAPH 2016), and it is chosen for a reason
// that is not performance: it converges to the *cosine-weighted visibility
// integral* rather than to a heuristic that looks like one. That is exactly the
// quantity `gg_render::ao` casts on the CPU, so the two can be compared as
// numbers and `gg-tools ao` prints how far apart they are. An algorithm graded
// against ground truth has to be aiming at the ground truth.
//
// What it structurally cannot do, and no tuning changes: it sees a depth buffer,
// not a scene. An occluder off the edge of the frame, or hidden behind a nearer
// surface, is not there. That error is measured rather than apologized for.

// The global set (§4.3). Binding 1 (storage images) has no consumer here.
[[vk::binding(0, 0)]] Texture2D<float4> g_textures[];
[[vk::binding(2, 0)]] SamplerState g_samplers[];

// Depth reconstruction modes, mirroring `View::ortho`'s zero-means-perspective.
static const uint PROJECTION_PERSPECTIVE = 0u;
static const uint PROJECTION_ORTHOGRAPHIC = 1u;

struct AoPush
{
    uint depth_index;    // the prepass depth (or its resolve) in g_textures
    uint source_index;   // fs_blur only: the raw AO target
    uint sampler_index;  // NearestClamp — every read here is a texel, not a blend
    uint projection;     // PROJECTION_*
    float2 texel;        // 1/extent, uv units
    float2 extent;       // pixels
    // ndc -> view scale per axis. Perspective: (aspect/h, -1/h) against a linear
    // depth of `near/d`. Orthographic: (half_width, -half_height) against a
    // linear depth of `d * range + z0`.
    float2 project;
    float2 depth_params; // perspective: (near, unused); ortho: (range, z0)
    float radius;        // world metres — how far "local" reaches
    float intensity;     // exponent on the result; 1.0 is the integral itself
    uint slices;         // directions through the pixel
    uint steps;          // taps along each half-slice
    float falloff;       // radius multiples over which a horizon fades to nothing
    float min_pixels;    // nearest a tap may be taken, in pixels — see `fs_main`
}

[[vk::push_constant]]
ConstantBuffer<AoPush> push;

struct VOut
{
    float4 pos : SV_Position;
    float2 uv : TEXCOORD0;
}

// `post.slang`'s fullscreen triangle, unchanged — see its comment for why a
// triangle and why uv 0 is the top row.
[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.uv = uv;
    o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

float raw_depth(float2 uv)
{
    return g_textures[push.depth_index].SampleLevel(g_samplers[push.sampler_index], uv, 0).r;
}

// Reverse-Z clears to 0.0 and nothing renders *at* the far plane, so a zero here
// is background — the one depth this pass must not reconstruct a position from.
bool is_background(float d)
{
    return d <= 0.0;
}

// View-space position from a uv and its depth. Camera-relative, like everything
// downstream of extract (§1.4), so view space and world space differ by a
// rotation and this is the frame the reference casts in.
//
// The two projections are two lines rather than an inverse matrix: reverse-Z
// with an infinite far gives `depth = near / -z` exactly, and the orthographic
// one is affine in z. Inverting a 4x4 per pixel to recover two constants would
// be the expensive way to be less exact.
float3 view_position(float2 uv, float d)
{
    float2 ndc = uv * 2.0 - 1.0;
    if (push.projection == PROJECTION_ORTHOGRAPHIC)
    {
        return float3(ndc * push.project, d * push.depth_params.x + push.depth_params.y);
    }
    float linear_depth = push.depth_params.x / max(d, 1e-9);
    return float3(ndc * push.project * linear_depth, -linear_depth);
}

float3 position_at(float2 uv)
{
    return view_position(uv, raw_depth(uv));
}

// The direction from a surface point toward the viewer, which every horizon in
// this pass is measured against.
//
// **Orthographic has no eye point** — `shade_all`'s note, and the same trap one
// pass along. Under parallel rays every view ray is the camera's axis, so
// normalizing `-p` turns a fragment's *screen position* into a view direction
// and tilts every slice plane by however far off centre it sits — by the frame's
// half-angle at its corners.
//
// What that is worth, measured rather than asserted (`gg-tools ao`'s projection
// row): it takes the orthographic leg's mean rendered occlusion from 0.9728 to
// 0.9585 against a reference's 0.9536, and its worst error from 0.6155 to
// 0.4856. Its mean *absolute* error rises, 0.0479 to 0.0598, and that is the
// honest shape of it — the wrong vector under-estimates everywhere and the right
// one is wrong in both directions instead. No golden in the tree can see the
// difference, because the only orthographic scene among them is demo 11's flat
// framing, whose visible faces are one coplanar sheet with no depth for a
// screen-space estimate to work from.
float3 to_eye(float3 p)
{
    if (push.projection == PROJECTION_ORTHOGRAPHIC)
    {
        return float3(0.0, 0.0, 1.0);
    }
    return normalize(-p);
}

// The normal, from the depth buffer rather than from a G-buffer the prepass does
// not write.
//
// The nearer neighbour on each axis is the one differenced, which is what keeps
// a silhouette from producing a normal that spans two surfaces: at a depth
// discontinuity one of the two candidates belongs to a different object, and it
// is always the further one.
float3 reconstruct_normal(float2 uv, float3 p)
{
    float3 right = position_at(uv + float2(push.texel.x, 0.0));
    float3 left = position_at(uv - float2(push.texel.x, 0.0));
    float3 down = position_at(uv + float2(0.0, push.texel.y));
    float3 up = position_at(uv - float2(0.0, push.texel.y));
    float3 dx = abs(right.z - p.z) < abs(p.z - left.z) ? right - p : p - left;
    float3 dy = abs(down.z - p.z) < abs(p.z - up.z) ? down - p : p - up;
    float3 n = cross(dy, dx);
    float len = length(n);
    if (len < 1e-12)
    {
        // A perfectly degenerate neighbourhood — a single-pixel island, or the
        // frame's own edge. Facing the camera is the answer that occludes
        // nothing rather than the one that occludes everything.
        return float3(0.0, 0.0, 1.0);
    }
    n /= len;
    // The prepass renders front faces, so the normal points back at the eye. The
    // guard is against the *reconstruction's* sign, not the geometry's.
    return dot(n, to_eye(p)) < 0.0 ? -n : n;
}

// How many world metres the radius is worth in pixels at this depth.
//
// Constant across the frame under an orthographic eye and inversely proportional
// to distance under a perspective one, which is the whole of the difference and
// is why `project.y` carries the vertical scale for both.
float radius_in_pixels(float3 p)
{
    float scale = push.extent.y * 0.5 / max(-push.project.y, 1e-6);
    if (push.projection == PROJECTION_ORTHOGRAPHIC)
    {
        return push.radius * scale;
    }
    return push.radius * scale / max(-p.z, 1e-4);
}

// One tap's horizon: the cosine of the angle between the view vector and the
// direction to the sampled point, attenuated by distance.
//
// The attenuation is what makes the radius a radius. Without it a wall ten
// metres away occludes as hard as a crate against the floor, because the horizon
// it raises is just as high; with it the contribution falls off over the radius
// and the term stays local, which is the definition AO is written to.
//
// It is also the one place a depth buffer's blindness is admitted rather than
// worked around: a sample is a surface with no back, so without a distance term
// a thin object would occlude everything behind it forever.
float horizon(float2 uv, float3 p, float3 v)
{
    float d = raw_depth(uv);
    if (is_background(d))
    {
        return -1.0;
    }
    float3 delta = view_position(uv, d) - p;
    float len = length(delta);
    if (len < 1e-6)
    {
        return -1.0;
    }
    float cos_h = dot(delta / len, v);
    // Attenuate toward "no horizon" rather than clipping, so a sample crossing
    // the radius does not step the result.
    //
    // **Quadratic, not linear**, and that is the definition rather than a taste:
    // occlusion within a radius is a *binary* visibility term — `gg_render::ao`
    // counts any hit inside it — so an attenuation that is already at 0.5 half
    // way out is estimating a different quantity, and does so by a third of the
    // term in a deep crease. Squared stays near one over most of the radius and
    // falls off only at its edge, which is as close to the definition as a
    // continuous weight gets without the popping a hard cutoff would give.
    float reach = len / (push.radius * max(push.falloff, 1e-3));
    float falloff = saturate(1.0 - reach * reach);
    return lerp(-1.0, cos_h, falloff);
}

// The visibility of one slice's arc, GTAO's inner integral in closed form.
//
// `n` is the angle of the normal projected into the slice plane; `h` is a
// horizon angle already clamped to within a right angle of it. This is the
// cosine-weighted visibility of the wedge between the two, and it is the reason
// the algorithm converges to the reference instead of to something that
// resembles it — an occlusion built from `1 - cos(h)` is the uniform-weighted
// answer and is wrong by a factor that varies with the surface's tilt.
float arc(float h, float n, float cos_n, float sin_n)
{
    return 0.25 * (-cos(2.0 * h - n) + cos_n + 2.0 * h * sin_n);
}

[shader("fragment")]
float4 fs_main(VOut i) : SV_Target
{
    float d = raw_depth(i.uv);
    if (is_background(d))
    {
        return float4(1.0);
    }
    float3 p = view_position(i.uv, d);
    float3 n = reconstruct_normal(i.uv, p);
    float3 v = to_eye(p);

    float pixels = radius_in_pixels(p);
    if (pixels <= push.min_pixels)
    {
        // The radius does not reach past the bias, so there is nothing to march
        // and every horizon would be this fragment's own. Reporting "open" is
        // the honest answer and it is also what a distant surface should get.
        return float4(1.0);
    }

    // Per-pixel rotation and offset, so `slices * steps` taps cover far more
    // directions across a neighbourhood than they do at one pixel. What this
    // buys in coverage it pays for in noise, which is `fs_blur`'s business — and
    // the 4x4 period below is chosen so that a 4x4 box filter averages exactly
    // one whole period and is therefore unbiased.
    float total = 0.0;
    float slices = float(max(push.slices, 1u));
    float steps = float(max(push.steps, 1u));

    // **The rotation tiles the gap between adjacent slices, not the half-turn**
    // (§6 M62). A slice is a line and the loop below marches both ways along it,
    // so the `slices` directions already span a whole PI; rotating by a quarter
    // of PI therefore lands a rotated slice *on* an unrotated one whenever the
    // step divides PI/slices. At the shipped two slices that is every other
    // column — rotations 0 and PI/2 name the same pair {0, PI/2} — and at four
    // it is every column, where the rotation did nothing whatsoever. Measured
    // rather than read: `gg-tools ao`'s pattern table prints the mean difference
    // at lag 1, 2, 4 and 8 px, and a lag that differs *less* than a nearer one
    // is a period with no reference renderer needed to name it. It read
    // 13.06/7.02 at lag 1 and 2 across columns, which is a period of two.
    //
    // The two indices are **not separable**, and that is the other half. A
    // rotation that is a function of x alone and an offset a function of y alone
    // put the same slice set down every column and the same tap phase along
    // every row, which is a grid rather than noise however unbiased its average
    // is. `(x+y, x+2y)` mod 4 has determinant 1 and is therefore a bijection of
    // the 4x4 tile: all sixteen pairs still occur exactly once, so `fs_blur`
    // stays the unbiased average it claims to be, and neither index is constant
    // along either axis. `pattern_is_a_bijection` is the check.
    uint2 pixel = uint2(i.pos.xy);
    float rotation = float((pixel.x + pixel.y) & 3u) * (0.25 * 3.14159265 / slices);
    float offset = float((pixel.x + 2u * pixel.y) & 3u) * 0.25;
    for (uint s = 0u; s < push.slices; ++s)
    {
        float phi = (float(s) / slices) * 3.14159265 + rotation;
        float2 dir = float2(cos(phi), sin(phi));

        // The slice plane: it contains the view vector and the marched
        // direction. `axis` is its normal, and the normal projected into it is
        // what the arc integral is measured against.
        //
        // **The y is negated**, and it is not cosmetic: `dir` is a step in uv,
        // Vulkan's clip space is Y-down, and `view_position` therefore maps an
        // increasing uv.y to a *decreasing* view y. Marching down the screen
        // while calling it +y puts `ortho` on the wrong side of the slice for
        // every direction with a vertical component, which flips the sign of the
        // normal's angle and makes a flat floor occlude itself. It is worth 0.45
        // of mean error against `gg_render::ao` — the whole term, and the
        // measurement is how it was found rather than the code being read.
        float3 slice_dir = float3(dir.x, -dir.y, 0.0);
        float3 axis = normalize(cross(slice_dir, v));
        float3 projected = n - axis * dot(n, axis);
        float projected_len = length(projected);
        if (projected_len < 1e-6)
        {
            continue;
        }
        float3 unit_projected = projected / projected_len;
        float3 ortho = normalize(slice_dir - v * dot(slice_dir, v));
        float sign_n = dot(unit_projected, ortho) < 0.0 ? -1.0 : 1.0;
        float cos_angle = clamp(dot(unit_projected, v), -1.0, 1.0);
        float angle_n = sign_n * acos(cos_angle);
        float cos_n = cos(angle_n);
        float sin_n = sin(angle_n);

        // Cosines of the two horizons, one either way along the slice. Starting
        // at -1 is "nothing found", which the clamps below turn into the
        // unoccluded right angle. `plus` marches along `ortho`, so it is the
        // side `sign_n` calls positive.
        float cos_plus = -1.0;
        float cos_minus = -1.0;
        for (uint t = 0u; t < push.steps; ++t)
        {
            float march = (float(t) + offset + 0.5) / steps;
            // Squared spacing: the near taps are where a contact shadow lives and
            // a linear march spends most of its samples on the far half of the
            // radius, where the falloff has already thrown the contribution away.
            //
            // **From `min_pixels`, not from zero**, and that is the difference
            // between shading and dirt. A tap a fraction of a pixel away on a
            // surface seen at a grazing angle differs from this fragment almost
            // entirely along the view axis, so the horizon it raises is near
            // vertical — the surface occluding itself. The clamp against the
            // normal is supposed to forgive exactly that, and does so only as
            // accurately as a normal reconstructed from neighbouring depths,
            // which at grazing incidence is not accurately at all. Measured
            // rather than argued: `gg-tools ao` sweeps this, and at zero it
            // invents 0.14 of occlusion on a hundred thousand open pixels.
            float2 delta = dir * lerp(push.min_pixels, pixels, march * march) * push.texel;
            cos_plus = max(cos_plus, horizon(i.uv + delta, p, v));
            cos_minus = max(cos_minus, horizon(i.uv - delta, p, v));
        }

        // Back to angles, and clamped to a right angle either side of the
        // normal: a horizon behind the surface is a sample the surface cannot
        // see and is not evidence of occlusion.
        const float HALF_PI = 1.5707963;
        float h2 = acos(clamp(cos_plus, -1.0, 1.0));
        float h1 = -acos(clamp(cos_minus, -1.0, 1.0));
        h1 = angle_n + max(h1 - angle_n, -HALF_PI);
        h2 = angle_n + min(h2 - angle_n, HALF_PI);
        total += projected_len * (arc(h1, angle_n, cos_n, sin_n) + arc(h2, angle_n, cos_n, sin_n));
    }

    float visibility = saturate(total / slices);
    return float4(pow(visibility, max(push.intensity, 0.0)));
}

// The denoiser: a 4x4 box over the raw target, which is exactly one period of
// the rotation-and-offset pattern above and is therefore an *unbiased* average
// of sixteen different slice orientations rather than a blur.
//
// Depth-aware, because a box that crossed a silhouette would drag an occluded
// crease onto the surface behind it — the halo that makes screen-space AO look
// like a filter rather than like shading.
[shader("fragment")]
float4 fs_blur(VOut i) : SV_Target
{
    float centre = raw_depth(i.uv);
    if (is_background(centre))
    {
        return float4(1.0);
    }
    float centre_z = -view_position(i.uv, centre).z;
    float total = 0.0;
    float weight = 0.0;
    for (int y = -2; y <= 1; ++y)
    {
        for (int x = -2; x <= 1; ++x)
        {
            float2 uv = i.uv + float2(float(x), float(y)) * push.texel;
            float d = raw_depth(uv);
            if (is_background(d))
            {
                continue;
            }
            // Within 2 % of the centre's depth is the same surface for this
            // purpose: a proportional threshold rather than an absolute one,
            // because a metre at the far plane is a pixel and a metre at the
            // near plane is the whole frame.
            float z = -view_position(uv, d).z;
            if (abs(z - centre_z) > 0.02 * centre_z)
            {
                continue;
            }
            total += g_textures[push.source_index]
                         .SampleLevel(g_samplers[push.sampler_index], uv, 0)
                         .r;
            weight += 1.0;
        }
    }
    return float4(weight > 0.0 ? total / weight : 1.0);
}
