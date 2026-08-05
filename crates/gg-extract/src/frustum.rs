//! Frustum culling (§6 M10), on the render side of the membrane.
//!
//! It lives here rather than in `gg-render` because the test wants
//! camera-relative `f32` and this crate is where positions become that. Culling
//! against absolute `f64` world positions would either narrow twice or do plane
//! arithmetic in `f64` for a decision that is a `bool` — and a bounding volume
//! is a *local* quantity, exactly like a half-extent (§1.4).
//!
//! # Five planes, not six
//!
//! The far plane is at infinity (§2, Math row: reverse-Z with an infinite far
//! plane), and the constraint it would impose — `z_clip >= 0` — is satisfied by
//! every point in front of the eye. Extracting it anyway yields a degenerate
//! plane with a zero normal, which normalizes to `NaN` and quietly rejects
//! everything. So there are five, and this comment is why.
//!
//! # Spheres, not boxes
//!
//! A sphere test is rotation-invariant, so an instance that spins costs no
//! per-frame bound rebuild. It over-keeps a long thin box turned diagonally,
//! which is the cheap direction to be wrong in: a kept instance costs a draw,
//! and a wrongly culled one is a hole in the picture.

use gg_math::render;

/// One half-space, `ax + by + cz + d >= 0` inside, with a unit normal — the
/// normalization is what makes `d` comparable against a radius in metres.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Plane {
    normal: render::Vec3,
    distance: f32,
}

impl Plane {
    /// Signed distance from `point`, in metres, positive inside.
    fn distance_to(&self, point: render::Vec3) -> f32 {
        self.normal.dot(point) + self.distance
    }
}

/// The camera-relative view frustum, as five inward half-spaces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    planes: [Plane; 5],
    /// Set by [`Frustum::UNBOUNDED`] — a frustum that keeps everything, for
    /// hosts and tests that do not cull. A flag rather than five planes placed
    /// at infinity, because a plane at infinity is exactly the degenerate case
    /// the module doc warns about.
    unbounded: bool,
}

impl Frustum {
    /// Keeps everything. What a caller that does not cull passes, and what
    /// [`Frustum::contains`] answers `true` for without touching a plane.
    pub const UNBOUNDED: Frustum = Frustum {
        planes: [Plane {
            normal: render::Vec3::ZERO,
            distance: 0.0,
        }; 5],
        unbounded: true,
    };

    /// The five planes of a camera-relative view-projection matrix, by the
    /// Gribb–Hartmann rows.
    ///
    /// `view_projection` must be the *camera-relative* one — the eye at the
    /// origin — which is the only kind this engine builds (see
    /// [`gg_math::render::perspective_reverse_z`] and `gg-render`'s `View`).
    #[must_use]
    pub fn from_view_projection(view_projection: render::Mat4) -> Frustum {
        let m = render::rows(view_projection);
        let plane = |row: [f32; 4]| {
            let normal = render::Vec3::new(row[0], row[1], row[2]);
            // Normalizing is not cosmetic: `distance_to` is compared against a
            // radius in metres, and an unnormalized plane's units are the
            // matrix's, which vary with fov and aspect.
            let length = normal.length();
            Plane {
                normal: normal / length,
                distance: row[3] / length,
            }
        };
        let combine = |a: [f32; 4], b: [f32; 4], add: bool| {
            let sign = if add { 1.0 } else { -1.0 };
            [
                a[0] + sign * b[0],
                a[1] + sign * b[1],
                a[2] + sign * b[2],
                a[3] + sign * b[3],
            ]
        };
        Frustum {
            planes: [
                plane(combine(m[3], m[0], true)),  // left
                plane(combine(m[3], m[0], false)), // right
                plane(combine(m[3], m[1], true)),  // bottom
                plane(combine(m[3], m[1], false)), // top
                // Near. Under reverse-Z this is the `z <= w` plane; the `z >= 0`
                // one is the far plane, and it is at infinity.
                plane(combine(m[3], m[2], false)),
            ],
            unbounded: false,
        }
    }

    /// Whether a sphere at `center` (camera-relative metres) of `radius` has any
    /// part inside.
    ///
    /// A non-finite radius keeps the instance: an asset whose bounds are not
    /// known yet must not vanish while it streams in (§4.6).
    ///
    /// P1: the short-circuiting `all` looks cheaper than it is. A variant that
    /// evaluated all five distances up front — *and* a needless `sqrt` on each —
    /// ran the extract bench's serial narrow leg at 8.2 ns/entity against 9.5,
    /// because a world spread across the frustum mispredicts the early exit
    /// every few rows. Found by accident while falsifying that bench's gates,
    /// and not yet measured without the `sqrt`, which is what would settle it.
    #[must_use]
    pub fn contains(&self, center: render::Vec3, radius: f32) -> bool {
        if self.unbounded || !radius.is_finite() {
            return true;
        }
        self.planes
            .iter()
            .all(|plane| plane.distance_to(center) >= -radius)
    }
}

#[cfg(test)]
mod tests {
    use core::f32::consts::FRAC_PI_3;

    use super::*;

    /// 90° vertical fov, square, near 0.1 — a frustum whose planes sit at 45°.
    fn frustum() -> Frustum {
        Frustum::from_view_projection(render::perspective_reverse_z(FRAC_PI_3, 1.0, 0.1))
    }

    #[test]
    fn a_point_down_the_view_axis_is_inside_and_one_behind_the_eye_is_not() {
        let f = frustum();
        // The engine looks down -Z in view space (right-handed, Y up).
        assert!(f.contains(render::Vec3::new(0.0, 0.0, -10.0), 0.0));
        assert!(!f.contains(render::Vec3::new(0.0, 0.0, 10.0), 0.0));
    }

    #[test]
    fn something_just_behind_the_eye_is_kept_if_it_is_big_enough_to_reach_in() {
        let f = frustum();
        let just_behind = render::Vec3::new(0.0, 0.0, 0.05);
        assert!(!f.contains(just_behind, 0.0), "a point is out");
        assert!(f.contains(just_behind, 1.0), "a metre-wide thing is not");
    }

    #[test]
    fn the_far_plane_does_not_exist() {
        // The whole reason there are five planes: at 10^9 m the sixth would
        // have rejected this, and reverse-Z put it at infinity instead.
        let f = frustum();
        assert!(f.contains(render::Vec3::new(0.0, 0.0, -1.0e9), 1.0));
    }

    #[test]
    fn something_far_off_to_the_side_is_culled() {
        let f = frustum();
        assert!(!f.contains(render::Vec3::new(1000.0, 0.0, -10.0), 1.0));
        assert!(f.contains(render::Vec3::new(1000.0, 0.0, -10.0), 10_000.0));
    }

    #[test]
    fn an_unbounded_frustum_keeps_what_a_real_one_rejects() {
        let behind = render::Vec3::new(0.0, 0.0, 10.0);
        assert!(!frustum().contains(behind, 0.0));
        assert!(Frustum::UNBOUNDED.contains(behind, 0.0));
    }

    #[test]
    fn an_unknown_radius_is_never_culled() {
        // What a mesh that has not streamed in yet reports.
        assert!(frustum().contains(render::Vec3::new(0.0, 0.0, 1.0e6), f32::INFINITY));
    }

    #[test]
    fn the_planes_are_normalized_so_a_radius_is_in_metres() {
        let f = frustum();
        for plane in &f.planes {
            assert!((plane.normal.length() - 1.0).abs() < 1e-5, "{plane:?}");
        }
        // A sphere centred exactly on the near plane's outside, of radius equal
        // to its distance, must just touch: that only holds in metres.
        let on_axis = render::Vec3::new(0.0, 0.0, 0.0);
        assert!(f.contains(on_axis, 0.1001));
    }
}
