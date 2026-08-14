//! Ground truth for §6 M35's ambient occlusion: the visibility integral itself,
//! cast ray by ray on the CPU.
//!
//! What the shader computes is an approximation of one number —
//!
//! ```text
//! AO(p, n) = (1/π) ∫ V(p, ω) (n·ω) dω     over the hemisphere about n
//! ```
//!
//! — where `V` is 0 when a ray leaving `p` meets geometry within
//! [`radius`](Params::radius) and 1 when it escapes. This module is that
//! integral with nothing approximated: cosine-weighted Hammersley directions,
//! an analytic intersection against the scene's own primitives, and a count.
//! One is fully open and zero is sealed, which is the sense
//! [`Surface::occlusion`](../shaders/include/pbr.slang) already multiplies in.
//!
//! **Why a reference and not a golden.** A blessed image proves the AO pass
//! renders what it rendered last time; it cannot distinguish darkening a corner
//! by the right amount from darkening it by some other amount that looks
//! plausible — the §6 M34 failure, one milestone on. Screen-space AO has a
//! *principled* error besides: it sees a depth buffer rather than a scene, so an
//! occluder outside the frame or behind another surface is invisible to it and
//! no amount of tuning recovers that. A number is the only way to say how large
//! that gap is, and `gg-tools ao` is where it is printed.
//!
//! It casts against [`Occluder`]s rather than against a `World` on purpose: the
//! primitives are the two [`shape`](gg_ecs::boundary::shape)s a `Renderable`
//! can be, so a box scene is exact here, and a pack mesh is not representable —
//! which is honest, because a triangle-soup reference is a renderer and this is
//! a measurement.

use gg_math::sim;

/// One primitive a ray can meet. Camera-relative, like everything downstream of
/// `gg-extract` (§1.4) — the eye is the origin.
///
/// [`albedo`](Self::albedo) and [`emission`](Self::emission) are what a *bounce*
/// needs ([`crate::bounce`], §6 M36) and [`occlusion`] ignores both: a shape
/// blocks a ray whatever colour it is. They live here rather than in a parallel
/// array because one scene with two slices to keep in step is a bug waiting for
/// a caller who appends to one of them.
#[derive(Clone, Copy, Debug)]
pub struct Occluder {
    /// Centre, metres from the eye.
    pub center: sim::Vec3,
    /// Orientation. Identity for anything axis-aligned.
    pub rotation: sim::Quat,
    /// Half-extent per axis *before* rotation.
    pub half_extent: sim::Vec3,
    /// An ellipsoid of the same half-extent rather than a box —
    /// [`shape::SPHERE`](gg_ecs::boundary::shape::SPHERE).
    pub sphere: bool,
    /// Diffuse reflectance, **linear** and per channel. Physically below 1;
    /// nothing here clamps it, and a scene that sets it higher is a furnace
    /// whose series does not converge.
    pub albedo: sim::Vec3,
    /// Radiance this surface emits on its own, linear. Zero for everything the
    /// engine can currently declare — a `Renderable` has no emissive term — and
    /// present because the one closed form that grades a *bounce count* is an
    /// enclosure that emits ([`bounce`](crate::bounce)).
    pub emission: sim::Vec3,
}

/// Where a ray met something.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    /// Distance along the (unit) direction.
    pub distance: f32,
    /// Outward surface normal at the meeting point.
    pub normal: sim::Vec3,
}

/// The knobs the reference and the shader must agree on, or they are integrating
/// two different quantities and their difference means nothing.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// How far a ray looks before it counts as escaped, metres. AO is a
    /// *local* term by definition — the far field is the ambient itself — and
    /// this is where local stops.
    pub radius: f32,
    /// Rays per point. 256 is inside a thousandth of the converged value on
    /// every cell `gg-tools ao` sweeps; the instrument's own reference uses
    /// more because it can afford to.
    pub samples: u32,
}

impl Default for Params {
    /// [`cvars::AO_RADIUS`](crate::cvars::AO_RADIUS)'s default, so a reference
    /// built without argument grades the shipped configuration.
    fn default() -> Params {
        Params {
            radius: 0.5,
            samples: 256,
        }
    }
}

/// Where a ray from `origin` along `direction` first meets `scene`, if it does.
///
/// `direction` must be unit length; nothing here normalizes it, because the two
/// callers both have one already and a silent renormalize is a silent
/// disagreement about what `distance` means.
///
/// Hits behind `near` are ignored, which is what keeps a point that *lies on* a
/// primitive from occluding itself — a surface sample is on the boundary of its
/// own slab and the slab test is entitled to report `0.0`.
#[must_use]
pub fn trace(
    origin: sim::Vec3,
    direction: sim::Vec3,
    scene: &[Occluder],
    near: f32,
    far: f32,
) -> Option<Hit> {
    nearest(origin, direction, scene, near, far).map(|(_, hit)| hit)
}

/// [`trace`], and *which* primitive was met — the index into `scene`.
///
/// A bounce needs it and occlusion does not: what leaves a surface is a property
/// of the surface, so [`crate::bounce`] has to get back from a hit to the thing
/// it hit.
#[must_use]
pub fn nearest(
    origin: sim::Vec3,
    direction: sim::Vec3,
    scene: &[Occluder],
    near: f32,
    far: f32,
) -> Option<(usize, Hit)> {
    let mut best: Option<(usize, Hit)> = None;
    for (index, occluder) in scene.iter().enumerate() {
        let Some(hit) = meet(origin, direction, occluder, near, far) else {
            continue;
        };
        if best.is_none_or(|(_, b)| hit.distance < b.distance) {
            best = Some((index, hit));
        }
    }
    best
}

/// True cosine-weighted ambient occlusion at `point`, facing `normal`.
///
/// One is unoccluded. `normal` must be unit length.
#[must_use]
pub fn occlusion(point: sim::Vec3, normal: sim::Vec3, scene: &[Occluder], params: Params) -> f32 {
    // Off the surface by a whisker before casting: the slab test is exact, and
    // exact on a boundary is the one place it may answer either way.
    let origin = point + normal * (1e-4 * params.radius.max(1.0));
    let (tangent, bitangent) = frame(normal);
    let mut open = 0u32;
    for i in 0..params.samples {
        let (u1, u2) = (
            (i as f32 + 0.5) / params.samples as f32,
            crate::split_sum::radical_inverse(i),
        );
        // Cosine-weighted, so the (n·ω) of the integrand is the density and the
        // estimator is a plain count rather than a weighted sum.
        let r = sim::sqrt(u1);
        let (sin_phi, cos_phi) = sim::sin_cos(core::f32::consts::TAU * u2);
        let direction =
            tangent * (r * cos_phi) + bitangent * (r * sin_phi) + normal * sim::sqrt(1.0 - u1);
        if trace(origin, direction, scene, 1e-4, params.radius).is_none() {
            open += 1;
        }
    }
    open as f32 / params.samples as f32
}

/// Any orthonormal pair square to `n` — [Duf17]'s branchless frame, which has no
/// singularity to guard and no `atan2` in it.
///
/// Which pair does not matter: the estimator is rotationally symmetric about
/// `n`, so a frame that spins with the normal changes the order of the samples
/// and not their distribution.
fn frame(n: sim::Vec3) -> (sim::Vec3, sim::Vec3) {
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    (
        sim::Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
        sim::Vec3::new(b, sign + n.y * n.y * a, -n.y),
    )
}

/// One primitive's intersection, in its own local frame.
fn meet(
    origin: sim::Vec3,
    direction: sim::Vec3,
    occluder: &Occluder,
    near: f32,
    far: f32,
) -> Option<Hit> {
    let inverse = occluder.rotation.conjugate();
    let local_origin = inverse.rotate(origin - occluder.center);
    let local_direction = inverse.rotate(direction);
    let (distance, local_normal) = if occluder.sphere {
        ellipsoid(local_origin, local_direction, occluder.half_extent, near)?
    } else {
        slab(local_origin, local_direction, occluder.half_extent, near)?
    };
    (distance <= far).then(|| Hit {
        distance,
        normal: occluder.rotation.rotate(local_normal),
    })
}

/// Ray versus an origin-centred box: [`sim::Ray::aabb`]'s interval (§6 M37
/// item 1), and this pass's own answer to what that interval means.
fn slab(
    origin: sim::Vec3,
    direction: sim::Vec3,
    half: sim::Vec3,
    near: f32,
) -> Option<(f32, sim::Vec3)> {
    let span = sim::Ray::new(origin, direction).aabb(half)?;
    // Past `near` on the way in, or — for a ray that started *inside* — on the
    // way out. The second branch needs `enter` clearly behind rather than merely
    // not ahead, and that is the whole difficulty: a ray leaving a point that
    // lies **on** this box's face is tangent to one of its slabs and enters at
    // 0.0, which is neither in front nor behind. Reading that as "started
    // inside" makes every surface occlude itself along its own plane.
    let distance = if span.enter > near {
        span.enter
    } else if span.enter < -near && span.exit > near {
        span.exit
    } else {
        return None;
    };
    Some((distance, span.normal))
}

/// Ray versus an origin-centred ellipsoid, by squashing both into the unit
/// sphere's frame — where the quadratic is the one everyone knows.
fn ellipsoid(
    origin: sim::Vec3,
    direction: sim::Vec3,
    half: sim::Vec3,
    near: f32,
) -> Option<(f32, sim::Vec3)> {
    let scale = sim::Vec3::new(
        1.0 / half.x.max(1e-6),
        1.0 / half.y.max(1e-6),
        1.0 / half.z.max(1e-6),
    );
    let o = sim::Vec3::new(origin.x * scale.x, origin.y * scale.y, origin.z * scale.z);
    let d = sim::Vec3::new(
        direction.x * scale.x,
        direction.y * scale.y,
        direction.z * scale.z,
    );
    let (a, b, c) = (d.dot(d), o.dot(d), o.dot(o) - 1.0);
    if a < 1e-12 {
        return None;
    }
    let discriminant = b * b - a * c;
    if discriminant < 0.0 {
        return None;
    }
    let root = sim::sqrt(discriminant);
    let (t0, t1) = ((-b - root) / a, (-b + root) / a);
    // The squash is not a similarity, so these are distances in the *unit
    // sphere's* space; the ratio is what carries them back, and it is the same
    // ratio for both roots.
    let distance = if t0 > near { t0 } else { t1 };
    if distance <= near {
        return None;
    }
    let point = o + d * distance;
    // The gradient of the implicit surface, squashed back — a normal transforms
    // by the inverse transpose, which for a diagonal scale is the scale again.
    let normal = sim::Vec3::new(point.x * scale.x, point.y * scale.y, point.z * scale.z);
    Some((distance, normal.try_normalize()?))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn wall(center: sim::Vec3, half: sim::Vec3) -> Occluder {
        Occluder {
            center,
            rotation: sim::Quat::IDENTITY,
            half_extent: half,
            sphere: false,
            albedo: sim::Vec3::ZERO,
            emission: sim::Vec3::ZERO,
        }
    }

    #[test]
    fn an_open_plane_occludes_nothing() {
        let floor = [wall(
            sim::Vec3::new(0.0, -1.0, 0.0),
            sim::Vec3::new(50.0, 1.0, 50.0),
        )];
        let open = occlusion(sim::Vec3::ZERO, sim::Vec3::Y, &floor, Params::default());
        assert!(
            open > 0.999,
            "a point on an unobstructed floor is open to the whole hemisphere, not {open}"
        );
    }

    /// A half-space through the sample point takes exactly half the weight,
    /// whatever the radius and whatever the cosine lobe does — the one AO value
    /// that is a symmetry rather than an integral, so it grades the estimator's
    /// *bias* with nothing to be approximately right about.
    ///
    /// The point sits a thousandth of a radius clear of the wall's face rather
    /// than on it. On it, every blocked direction is exactly tangent to the
    /// slab, which is the measure-zero case [`slab`] refuses on purpose and a
    /// terrible thing to build an assertion on.
    #[test]
    fn a_half_space_takes_half() {
        let scene = [
            wall(
                sim::Vec3::new(0.0, -1.0, 0.0),
                sim::Vec3::new(50.0, 1.0, 50.0),
            ),
            // Face at x = 0, ten metres tall: nothing escapes over the top
            // within the radius.
            wall(
                sim::Vec3::new(-1.0, 5.0, 0.0),
                sim::Vec3::new(1.0, 5.0, 50.0),
            ),
        ];
        let open = occlusion(
            sim::Vec3::new(1e-3, 0.0, 0.0),
            sim::Vec3::Y,
            &scene,
            Params {
                samples: 4096,
                ..Params::default()
            },
        );
        assert!(
            (open - 0.5).abs() < 0.01,
            "a point against a half-space sees half the hemisphere, not {open}"
        );
    }

    /// A lid `h` above the sample admits exactly the ring below elevation
    /// `asin(h/R)`, whose cosine-weighted share is `(h/R)²` — closed form, and
    /// the reason this is a test of the estimator rather than of a scene.
    ///
    /// It is also the assertion that catches a radius that is not the radius:
    /// the value depends on `h/R` alone, so a [`Params::radius`] the caster
    /// silently squared or halved lands nowhere near it.
    #[test]
    fn a_lid_admits_the_ring_the_radius_leaves() {
        for height in [0.05f32, 0.15, 0.3] {
            let scene = [
                wall(
                    sim::Vec3::new(0.0, -1.0, 0.0),
                    sim::Vec3::new(50.0, 1.0, 50.0),
                ),
                wall(
                    sim::Vec3::new(0.0, height + 0.5, 0.0),
                    sim::Vec3::new(50.0, 0.5, 50.0),
                ),
            ];
            let params = Params {
                samples: 4096,
                ..Params::default()
            };
            let open = occlusion(sim::Vec3::ZERO, sim::Vec3::Y, &scene, params);
            let exact = (height / params.radius).powi(2);
            assert!(
                (open - exact).abs() < 0.005,
                "a lid {height} above the sample left {open} open against the ring's {exact}"
            );
        }
    }

    #[test]
    fn a_sphere_is_met_where_the_arithmetic_says() {
        let ball = Occluder {
            center: sim::Vec3::new(0.0, 0.0, -5.0),
            rotation: sim::Quat::IDENTITY,
            half_extent: sim::Vec3::splat(2.0),
            sphere: true,
            albedo: sim::Vec3::ZERO,
            emission: sim::Vec3::ZERO,
        };
        let hit = trace(
            sim::Vec3::ZERO,
            sim::Vec3::new(0.0, 0.0, -1.0),
            &[ball],
            1e-4,
            100.0,
        )
        .unwrap();
        assert!((hit.distance - 3.0).abs() < 1e-3, "{}", hit.distance);
        assert!(hit.normal.z > 0.999, "the near face points back at the ray");
    }

    /// The estimator's own convergence, which is what licenses
    /// [`Params::samples`]'s default.
    #[test]
    fn the_count_settles_by_the_default_sample_count() {
        let scene = [
            wall(
                sim::Vec3::new(0.0, -1.0, 0.0),
                sim::Vec3::new(50.0, 1.0, 50.0),
            ),
            wall(
                sim::Vec3::new(-1.0, 5.0, 0.0),
                sim::Vec3::new(1.0, 5.0, 50.0),
            ),
        ];
        let at = |samples| {
            occlusion(
                sim::Vec3::new(0.3, 0.0, 0.0),
                sim::Vec3::Y,
                &scene,
                Params {
                    samples,
                    ..Params::default()
                },
            )
        };
        let (coarse, fine) = (at(256), at(16_384));
        assert!(
            (coarse - fine).abs() < 0.02,
            "256 rays reported {coarse} against 16384's {fine}"
        );
    }
}
