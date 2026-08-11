//! Frustum culling (§6 M10), on the render side of the membrane.
//!
//! It lives here rather than in `gg-render` because the test wants
//! camera-relative `f32` and this crate is where positions become that. Culling
//! against absolute `f64` world positions would either narrow twice or do plane
//! arithmetic in `f64` for a decision that is a `bool` — and a bounding volume
//! is a *local* quantity, exactly like a half-extent (§1.4).
//!
//! # Five planes, or six
//!
//! The perspective far plane is at infinity (§2, Math row: reverse-Z with an
//! infinite far plane), and the constraint it would impose — `z_clip >= 0` — is
//! satisfied by every point in front of the eye. Extracting it anyway yields a
//! degenerate plane with a zero normal, which normalizes to `NaN` and quietly
//! rejects everything. An orthographic projection (§6 M20) takes a *finite* far,
//! and dropping that one keeps every instance behind it — a hole no picture
//! shows, because more is drawn rather than less (§6 M18 item 8). So the count
//! is read off the matrix: a far row with no normal is the infinite case and
//! contributes no plane, one with a normal is a sixth plane like any other.
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

/// The camera-relative view frustum, as five or six inward half-spaces.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frustum {
    planes: [Plane; 6],
    /// Live planes in `planes` — 5 for an infinite far, 6 for a finite one. The
    /// slot past `count` is never read, so it holds the degenerate plane rather
    /// than a sentinel.
    count: usize,
    /// Set by [`Frustum::UNBOUNDED`] — a frustum that keeps everything, for
    /// hosts and tests that do not cull. A flag rather than planes placed
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
        }; 6],
        count: 0,
        unbounded: true,
    };

    /// The planes of a camera-relative view-projection matrix, by the
    /// Gribb–Hartmann rows — five for a perspective matrix (far at infinity),
    /// six for an orthographic one (finite far). See the module's *Five planes,
    /// or six* note for how the matrix itself is what decides.
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
        // The far constraint — Vulkan clips to `0 <= z <= w`, so `z >= 0` is
        // row 2 on its own rather than a combination of two (the OpenGL-shaped
        // `m[3] + m[2]` is for a `[-w, w]` clip range this engine does not
        // use). `perspective_reverse_z` leaves the row `[0, 0, 0, near]` — no
        // normal at all, the degenerate plane the module doc warns about —
        // while `orthographic_reverse_z` puts `1 / (far - near)` there, orders
        // of magnitude above this threshold for any range a camera would use.
        let finite_far = render::Vec3::new(m[2][0], m[2][1], m[2][2]).length_squared() > 1e-12;
        Frustum {
            planes: [
                plane(combine(m[3], m[0], true)),  // left
                plane(combine(m[3], m[0], false)), // right
                // Top before bottom, unlike X: row 1 carries Vulkan's Y flip
                // (`-1/half_height`), so `m[3] + m[1]` is the *upper* bound.
                plane(combine(m[3], m[1], true)),  // top
                plane(combine(m[3], m[1], false)), // bottom
                // Near: Vulkan's `z <= w`.
                plane(combine(m[3], m[2], false)),
                // Far: row 2 alone, normalized only when it exists — slot 5 is
                // never read in the five-plane case, and normalizing a zero
                // normal is the NaN the count exists to avoid.
                match finite_far {
                    true => plane(m[2]),
                    false => Plane {
                        normal: render::Vec3::ZERO,
                        distance: 0.0,
                    },
                },
            ],
            count: match finite_far {
                true => 6,
                false => 5,
            },
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
    /// evaluated every plane's distance up front — *and* a needless `sqrt` on
    /// each, needless because [`Frustum::from_view_projection`] normalizes once
    /// at construction — ran the extract bench's serial narrow leg at
    /// 8.2 ns/entity against 9.5, because a world spread across the frustum
    /// mispredicts the early exit every few rows. Found by accident while
    /// falsifying that bench's gates, and not yet measured without the `sqrt`,
    /// which is what would settle it. That bench's frustum is perspective, so it
    /// prices the five-plane case only.
    #[must_use]
    pub fn contains(&self, center: render::Vec3, radius: f32) -> bool {
        if self.unbounded || !radius.is_finite() {
            return true;
        }
        self.planes[..self.count]
            .iter()
            .all(|plane| plane.distance_to(center) >= -radius)
    }
}

#[cfg(test)]
mod tests {
    use core::f32::consts::FRAC_PI_3;

    use super::*;

    /// 60° vertical fov, square, near 0.1 — side planes 30° off the view axis.
    fn frustum() -> Frustum {
        Frustum::from_view_projection(render::perspective_reverse_z(FRAC_PI_3, 1.0, 0.1))
    }

    /// A 16 m half-extent slab from 0.1 to 200 m — an orthographic camera's
    /// frustum, which is a box.
    fn ortho() -> Frustum {
        Frustum::from_view_projection(render::orthographic_reverse_z(16.0, 16.0, 0.1, 200.0))
    }

    /// §6 M18's third gap, closed rather than fenced: the five-plane culler
    /// silently kept everything past a finite far, then M18 item 8 made that a
    /// debug panic, and this is the test the defect never had — a point beyond
    /// `far` is *culled*. The near assertion is what makes it a far-plane test:
    /// a culler that rejected the whole axis would pass the first line too.
    #[test]
    fn an_orthographic_frustum_has_a_far_plane_and_it_culls() {
        let f = ortho();
        assert!(
            !f.contains(render::Vec3::new(0.0, 0.0, -250.0), 0.0),
            "past far"
        );
        assert!(
            f.contains(render::Vec3::new(0.0, 0.0, -150.0), 0.0),
            "inside"
        );
        // And the plane is where the matrix put it, not merely somewhere: a
        // sphere centred 10 m past it that reaches back in is kept.
        assert!(f.contains(render::Vec3::new(0.0, 0.0, -210.0), 10.001));
        assert!(!f.contains(render::Vec3::new(0.0, 0.0, -210.0), 9.999));
    }

    /// The four side planes of an orthographic frustum are parallel to the view
    /// axis: the box is as wide at 150 m as at 1 m, where a perspective frustum
    /// this test would forgive opens with depth.
    #[test]
    fn an_orthographic_frustum_is_a_box_not_a_pyramid() {
        let f = ortho();
        for depth in [-1.0, -150.0] {
            assert!(
                f.contains(render::Vec3::new(15.9, 0.0, depth), 0.0),
                "at {depth}"
            );
            assert!(
                !f.contains(render::Vec3::new(16.1, 0.0, depth), 0.0),
                "at {depth}"
            );
        }
        // The control: at 150 m a perspective frustum is far wider than 16 m, so
        // the rejections above are the box's parallel sides and not "any frustum
        // rejects 16.1". It says nothing about the far plane — both depths here
        // sit inside it, and proving *that* is the test above.
        assert!(frustum().contains(render::Vec3::new(16.1, 0.0, -150.0), 0.0));
    }

    /// Behind the near plane is still out — the sixth plane must not have
    /// displaced the fifth.
    #[test]
    fn an_orthographic_frustum_still_clips_at_the_near_plane() {
        let f = ortho();
        assert!(!f.contains(render::Vec3::new(0.0, 0.0, 0.0), 0.05));
        assert!(f.contains(render::Vec3::new(0.0, 0.0, -0.2), 0.0));
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
        for f in [frustum(), ortho()] {
            for plane in &f.planes[..f.count] {
                assert!((plane.normal.length() - 1.0).abs() < 1e-5, "{plane:?}");
            }
        }
        let f = frustum();
        // A sphere centred exactly on the near plane's outside, of radius equal
        // to its distance, must just touch: that only holds in metres.
        let on_axis = render::Vec3::new(0.0, 0.0, 0.0);
        assert!(f.contains(on_axis, 0.1001));
    }
}
