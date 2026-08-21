// Position and normal from a depth buffer (§6 M86), shared by the two passes
// that reconstruct one.
//
// The occlusion pass marches horizons against the prepass depth; the debug blit
// shows what that pass saw. They must agree *by definition* — one is a picture
// of the other's input — and until this file they agreed by transcription, with
// `debug.slang` carrying a note arguing the copy was safe because MODE_NORMAL is
// read. It was not a machine: §6 M71 replaced the differencing rule here and the
// copy kept the old one, so for fourteen milestones the view an operator opens to
// look at a crease was drawing the artifact M71 removed — in the one place where
// seeing it would be read as evidence about the *occlusion* pass.
//
// Every function takes its state instead of reading `push`. The two blocks name
// the depth image differently (`depth_index` against `source_index`), so a shared
// body reaching for a global would bind to whichever push constant the include
// landed above — an agreement enforced by include order, which is the same kind
// of agreement this file exists to delete.

// Mirrors `View::ortho`'s zero-means-perspective. Generated into Rust beside the
// push-constant struct, so the host cannot spell the discriminant by hand.
static const uint PROJECTION_PERSPECTIVE = 0u;
static const uint PROJECTION_ORTHOGRAPHIC = 1u;

/// A depth buffer and everything needed to undo the projection that wrote it.
struct DepthView
{
    Texture2D<float4> image; // the prepass depth, or its resolve
    SamplerState samp;       // NearestClamp — a reconstruction reads texels
    uint projection;         // PROJECTION_*
    float2 texel;            // 1/extent of `image`, uv units
    // ndc -> view scale per axis. Perspective: (aspect/h, -1/h) against a linear
    // depth of `near/d`. Orthographic: (half_width, -half_height) against a
    // linear depth of `d * range + z0`.
    float2 project;
    float2 depth_params; // perspective: (near, unused); ortho: (range, z0)
}

float depth_at(DepthView v, float2 uv)
{
    return v.image.SampleLevel(v.samp, uv, 0).r;
}

// Reverse-Z clears to 0.0 and nothing renders *at* the far plane, so a zero here
// is background — the one depth no caller may reconstruct a position from.
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
float3 view_position(DepthView v, float2 uv, float d)
{
    float2 ndc = uv * 2.0 - 1.0;
    if (v.projection == PROJECTION_ORTHOGRAPHIC)
    {
        return float3(ndc * v.project, d * v.depth_params.x + v.depth_params.y);
    }
    float linear_depth = v.depth_params.x / max(d, 1e-9);
    return float3(ndc * v.project * linear_depth, -linear_depth);
}

float3 position_at(DepthView v, float2 uv)
{
    return view_position(v, uv, depth_at(v, uv));
}

// The direction from a surface point toward the viewer, which every horizon in
// the occlusion pass is measured against.
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
float3 to_eye(DepthView v, float3 p)
{
    if (v.projection == PROJECTION_ORTHOGRAPHIC)
    {
        return float3(0.0, 0.0, 1.0);
    }
    return normalize(-p);
}

// The normal, from the depth buffer rather than from a G-buffer the prepass does
// not write.
//
// One of the two one-sided differences on each axis is the surface's and the
// other may belong to something else, and the whole difficulty is telling which.
//
// **By curvature, not by depth difference** (§6 M71). Differencing toward the
// *nearer* neighbour is the obvious rule and it is wrong in exactly one place,
// which happens to be the place a normal matters most: at a crease, where a
// surface seen at a glancing angle meets one seen closer to head-on. The
// glancing surface's own neighbour is several per cent of the depth away while
// the head-on surface across the seam is barely any distance at all, so the
// nearer-neighbour rule reaches *across* the crease and returns a normal
// belonging to the other wall — and which way it goes flips with the seam's
// sub-pixel position, so the answer walks as the camera turns. It cost twenty
// code values along a seam whose truth is a constant, against three for the same
// seam cast on the CPU, and no sampling knob touched it: `gg-tools ao --crease`.
//
// A second difference asks the question that is actually being asked. Three
// samples on one surface are collinear in view space whatever the angle, so
// `|2*a1 - a2 - p|` is zero on the co-planar side and is the kink's full size on
// the side that crosses a crease or a silhouette. Four taps an axis rather than
// two, which is the price.
float3 reconstruct_normal(DepthView v, float2 uv, float3 p)
{
    float2 sx = float2(v.texel.x, 0.0);
    float2 sy = float2(0.0, v.texel.y);
    float3 right = position_at(v, uv + sx);
    float3 left = position_at(v, uv - sx);
    float3 down = position_at(v, uv + sy);
    float3 up = position_at(v, uv - sy);
    float bend_right = abs(2.0 * right.z - position_at(v, uv + 2.0 * sx).z - p.z);
    float bend_left = abs(2.0 * left.z - position_at(v, uv - 2.0 * sx).z - p.z);
    float bend_down = abs(2.0 * down.z - position_at(v, uv + 2.0 * sy).z - p.z);
    float bend_up = abs(2.0 * up.z - position_at(v, uv - 2.0 * sy).z - p.z);
    float3 dx = bend_right < bend_left ? right - p : p - left;
    float3 dy = bend_down < bend_up ? down - p : p - up;
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
    return dot(n, to_eye(v, p)) < 0.0 ? -n : n;
}
