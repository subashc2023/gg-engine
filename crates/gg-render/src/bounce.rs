//! Ground truth for §6 M36's derived irradiance: the light that bounced, cast
//! path by path on the CPU.
//!
//! `r.ambient` is a constant and §6 M28's environments are boxes a level author
//! drew, so nothing in the engine derives indirect light from the geometry that
//! actually produces it. What that costs is a number rather than a matter of
//! taste, and this module is the number's source — the rendering equation,
//! Lambertian, restricted to [`ao::Occluder`]'s two primitives:
//!
//! ```text
//! E(p, n) = ∫ L(p, ω) (n·ω) dω        over the hemisphere about n
//! L(x, ω) = Le(x) + (ρ(x)/π) E(x, nx)
//! ```
//!
//! — with `E` the irradiance a fragment's diffuse lobe is handed and `L` the
//! radiance arriving along a direction, which is the environment's when the ray
//! escapes and a surface's own when it does not.
//!
//! **Why this and not a golden, and not the shipped path in a loop.** The same
//! argument [`ao`](crate::ao) makes, one term along: a blessed image proves a
//! room is lit the way it was lit last time, and cannot tell light that bounced
//! off the floor from a constant somebody tuned until it looked similar. The
//! difference between a probe field and this is made of *discretization* —
//! probe spacing, an order-2 projection, a bounce count, how a sample near a
//! wall decides which side of it it is on — and each of those is a knob whose
//! cost is only visible against something that has none of them.
//!
//! **It grades itself against a closed form that needs no reference
//! implementation** ([`tests`], and §6 M33's `furnace` argument transplanted).
//! A closed enclosure whose walls all emit `Le` and all reflect `ρ` reaches
//! equilibrium radiance `Le/(1-ρ)` *everywhere inside it*, whatever its shape —
//! so irradiance is `πLe/(1-ρ)` at every point on every wall, and truncating at
//! `k` bounces leaves exactly `πLe(1-ρᵏ)/(1-ρ)`. That last is the useful half:
//! it is a closed form for the estimator's own [`Params::bounces`], so a tracer
//! that drops a bounce, double-counts one, or loses the `π` fails it by a
//! ratio anyone can read.
//!
//! # References
//!
//!   [Kaj86] Kajiya, "The Rendering Equation", SIGGRAPH 1986.

use gg_math::sim;

use crate::ao::{self, Occluder};

/// A light with no position: it reaches everything, and its shadow ray runs to
/// the far plane rather than to a point.
#[derive(Clone, Copy, Debug)]
pub struct Sun {
    /// The direction the light *travels*, unit — `gg_ecs::boundary::Light`'s
    /// convention, so a surface is lit when its normal opposes this.
    pub direction: sim::Vec3,
    /// Colour times intensity, linear. The shader's `light.color *
    /// light.intensity` exactly, because a reference in other units grades a
    /// unit conversion rather than a renderer.
    pub radiance: sim::Vec3,
}

/// A light with a position and a range, windowed to reach exactly zero at it.
#[derive(Clone, Copy, Debug)]
pub struct Lamp {
    /// Camera-relative, like everything else here.
    pub position: sim::Vec3,
    /// Colour times intensity, linear, before the falloff.
    pub radiance: sim::Vec3,
    /// Metres to exactly zero — `Light::range`.
    pub range: f32,
}

/// What a path can meet: the geometry, and the two kinds of light that are not
/// geometry.
#[derive(Clone, Copy, Debug)]
pub struct Scene<'a> {
    /// Shapes, with the albedo a bounce leaves on the light.
    pub solids: &'a [Occluder],
    /// Directional lights. The first one casting and the rest not is the
    /// *shader's* simplification (one shadow map); the reference shadows every
    /// one, because what it is for is saying what the shipped path missed.
    pub suns: &'a [Sun],
    /// Point lights.
    pub lamps: &'a [Lamp],
}

/// The knobs. Unlike [`ao::Params`] these are the *reference's* and not the
/// shader's — there is no shipped `r.` knob any of them corresponds to, which
/// is the point: the estimator has no discretization the field it grades will
/// share, so a difference cannot be blamed on a setting they had in common.
#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Paths per point.
    pub samples: u32,
    /// Bounce vertices along a path. `1` is single-bounce indirect — light
    /// leaves a lit surface once and arrives — and is what a probe field
    /// carrying no feedback term can hope to match.
    pub bounces: u32,
    /// How far a path segment looks before it counts as having escaped to the
    /// environment, metres.
    pub far: f32,
}

impl Default for Params {
    fn default() -> Params {
        Params {
            samples: 512,
            bounces: 4,
            far: 1e4,
        }
    }
}

/// **Indirect** irradiance arriving at `point` on a surface facing `normal`,
/// linear rgb — the quantity a Lambertian lobe multiplies by `albedo/π`.
///
/// `sky` is what the environment radiates along a direction that escapes; pass
/// a closure returning zero for a scene lit only by its own lights. `normal`
/// must be unit length.
///
/// Indirect and not total, and that falls out of the integral rather than being
/// a subtraction: [`Sun`] and [`Lamp`] are *delta* lights, so they carry no
/// solid angle for a hemisphere integral to find and appear here only through
/// the surfaces they lit. What arrives at `point` straight from one of them is
/// [`direct`], which the engine gets exactly from a shadow map and which is not
/// what a probe field is for. Sum the two for the whole.
#[must_use]
pub fn irradiance<F: Fn(sim::Vec3) -> sim::Vec3>(
    point: sim::Vec3,
    normal: sim::Vec3,
    scene: &Scene,
    sky: F,
    params: Params,
) -> sim::Vec3 {
    let mut total = sim::Vec3::ZERO;
    for i in 0..params.samples {
        total += path(point, normal, scene, &sky, params, i);
    }
    total / params.samples.max(1) as f32
}

/// One path's estimate of the integral. Cosine-weighted throughout, which is
/// why the estimator carries `π` and never a cosine: the density *is* the
/// integrand's `(n·ω)/π`, so the two cancel at every vertex and what is left of
/// a Lambertian bounce is exactly its albedo.
fn path<F: Fn(sim::Vec3) -> sim::Vec3>(
    point: sim::Vec3,
    normal: sim::Vec3,
    scene: &Scene,
    sky: &F,
    params: Params,
    sample: u32,
) -> sim::Vec3 {
    let mut total = sim::Vec3::ZERO;
    let mut throughput = sim::Vec3::splat(core::f32::consts::PI);
    let (mut origin, mut n) = (offset(point, normal), normal);
    for depth in 0..params.bounces.max(1) {
        // Each vertex advances the low-discrepancy sequence by the depth so a
        // path does not reuse one direction's pair at every bounce, which is
        // what turns 512 paths into 512 copies of one shape.
        let direction = hemisphere(n, sample, params.samples, depth);
        let Some((index, hit)) = ao::nearest(origin, direction, scene.solids, 1e-4, params.far)
        else {
            total += modulate(throughput, sky(direction));
            break;
        };
        let solid = &scene.solids[index];
        let surface = origin + direction * hit.distance;
        // What leaves that surface toward us: its own emission plus its
        // Lambertian response to every light that reaches it.
        let lit = direct(surface, hit.normal, scene);
        let out = solid.emission + modulate(solid.albedo, lit) * core::f32::consts::FRAC_1_PI;
        total += modulate(throughput, out);
        throughput = modulate(throughput, solid.albedo);
        (origin, n) = (offset(surface, hit.normal), hit.normal);
    }
    total
}

/// Irradiance at `point` from the scene's lights alone, shadowed — no bounce,
/// no environment.
#[must_use]
pub fn direct(point: sim::Vec3, normal: sim::Vec3, scene: &Scene) -> sim::Vec3 {
    let origin = offset(point, normal);
    let mut total = sim::Vec3::ZERO;
    for sun in scene.suns {
        let toward = -sun.direction;
        let cosine = normal.dot(toward);
        if cosine <= 0.0 || ao::trace(origin, toward, scene.solids, 1e-4, 1e6).is_some() {
            continue;
        }
        total += sun.radiance * cosine;
    }
    for lamp in scene.lamps {
        let offset_to = lamp.position - origin;
        let distance_squared = offset_to.dot(offset_to);
        let falloff = point_falloff(distance_squared, lamp.range);
        if falloff <= 0.0 {
            continue;
        }
        let distance = sim::sqrt(distance_squared);
        let toward = offset_to / distance.max(1e-6);
        let cosine = normal.dot(toward);
        // The shadow ray stops *at* the lamp: geometry past it is not between.
        if cosine <= 0.0 || ao::trace(origin, toward, scene.solids, 1e-4, distance - 1e-3).is_some()
        {
            continue;
        }
        total += lamp.radiance * (falloff * cosine);
    }
    total
}

/// [Kar13]'s windowed inverse square — `pbr.slang`'s `point_falloff`, restated
/// rather than shared because the two sides are different languages and the
/// only defence against them drifting is that both are four lines anyone can
/// read.
fn point_falloff(distance_squared: f32, range: f32) -> f32 {
    let ratio = distance_squared / (range * range).max(1e-6);
    let window = (1.0 - ratio * ratio).clamp(0.0, 1.0);
    window * window / distance_squared.max(1e-4)
}

/// Off the surface by a whisker before casting — [`ao::occlusion`]'s reason,
/// and the scale is absolute here because a bounce has no radius to take it
/// from.
fn offset(point: sim::Vec3, normal: sim::Vec3) -> sim::Vec3 {
    point + normal * 1e-4
}

/// A cosine-weighted direction about `n`, from sample `i` of `count`,
/// decorrelated per `depth`.
///
/// `count` is a parameter and not a constant, which is the whole of what the
/// convergence leg of `gg-tools bounce` was built to check: stratifying against
/// a fixed denominator leaves a smaller run sampling a *prefix* of the
/// hemisphere — every direction crowded toward the normal — and no closed-form
/// test here catches it, because the integrands they use are constant over
/// exactly the region a prefix covers.
fn hemisphere(n: sim::Vec3, i: u32, count: u32, depth: u32) -> sim::Vec3 {
    // Cranley-Patterson: one shift per depth, so every vertex sees the same
    // low-discrepancy set moved rather than a worse set.
    let shift = crate::split_sum::radical_inverse(depth.wrapping_mul(0x9e37_79b9));
    let u1 = ((i as f32 + 0.5) / count.max(1) as f32 + shift).fract();
    let u2 = (crate::split_sum::radical_inverse(i) + shift * 0.5).fract();
    let (tangent, bitangent) = frame(n);
    let r = sim::sqrt(u1);
    let (sin_phi, cos_phi) = sim::sin_cos(core::f32::consts::TAU * u2);
    tangent * (r * cos_phi) + bitangent * (r * sin_phi) + n * sim::sqrt((1.0 - u1).max(0.0))
}

/// Componentwise product. `sim::Vec3` has scalar `Mul` only — deliberately, the
/// sim having no use for a Hadamard product — and every line here that needs one
/// is a colour meeting a reflectance.
fn modulate(a: sim::Vec3, b: sim::Vec3) -> sim::Vec3 {
    sim::Vec3::new(a.x * b.x, a.y * b.y, a.z * b.z)
}

/// [Duf17]'s branchless frame — [`ao`]'s, and the same reasoning: which pair
/// does not matter, only that it is orthonormal.
fn frame(n: sim::Vec3) -> (sim::Vec3, sim::Vec3) {
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    (
        sim::Vec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
        sim::Vec3::new(b, sign + n.y * n.y * a, -n.y),
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    const NO_SKY: fn(sim::Vec3) -> sim::Vec3 = |_| sim::Vec3::ZERO;

    fn solid(center: sim::Vec3, half: sim::Vec3, albedo: f32, emission: f32) -> Occluder {
        Occluder {
            center,
            rotation: sim::Quat::IDENTITY,
            half_extent: half,
            sphere: false,
            albedo: sim::Vec3::splat(albedo),
            emission: sim::Vec3::splat(emission),
        }
    }

    /// Six slabs enclosing the origin, every one emitting and reflecting the
    /// same amount. Thick, so a path cannot leave through a corner seam.
    fn enclosure(albedo: f32, emission: f32) -> Vec<Occluder> {
        let (inner, thickness) = (2.0f32, 1.0f32);
        let mid = inner + thickness;
        let long = sim::Vec3::new(mid, thickness, mid);
        [
            (sim::Vec3::new(0.0, -mid, 0.0), long),
            (sim::Vec3::new(0.0, mid, 0.0), long),
            (
                sim::Vec3::new(-mid, 0.0, 0.0),
                sim::Vec3::new(thickness, mid, mid),
            ),
            (
                sim::Vec3::new(mid, 0.0, 0.0),
                sim::Vec3::new(thickness, mid, mid),
            ),
            (
                sim::Vec3::new(0.0, 0.0, -mid),
                sim::Vec3::new(mid, mid, thickness),
            ),
            (
                sim::Vec3::new(0.0, 0.0, mid),
                sim::Vec3::new(mid, mid, thickness),
            ),
        ]
        .into_iter()
        .map(|(c, h)| solid(c, h, albedo, emission))
        .collect()
    }

    /// An unobstructed point under a uniform environment receives exactly `πL`,
    /// whatever `L` is and wherever it faces. The estimator's normalization and
    /// nothing else — if the `π` is wrong this is the test that says so, and it
    /// says so by a ratio rather than by a tolerance anyone chose.
    #[test]
    fn an_open_point_under_a_uniform_sky_takes_pi_times_it() {
        let scene = Scene {
            solids: &[],
            suns: &[],
            lamps: &[],
        };
        for normal in [
            sim::Vec3::Y,
            -sim::Vec3::Y,
            sim::Vec3::new(0.3, 0.8, -0.5).try_normalize().unwrap(),
        ] {
            let e = irradiance(
                sim::Vec3::ZERO,
                normal,
                &scene,
                |_| sim::Vec3::new(0.25, 0.5, 1.0),
                Params::default(),
            );
            let exact = sim::Vec3::new(0.25, 0.5, 1.0) * core::f32::consts::PI;
            assert!(
                (e - exact).length() / exact.length() < 1e-4,
                "{normal:?}: {e:?} against {exact:?}"
            );
        }
    }

    /// A sky that varies over the hemisphere, against its closed form — the
    /// test the constant-sky one above cannot be: with `L(ω) = n·ω` the
    /// integral is `∫cos²θ dω = 2π/3`, and an estimator whose directions crowd
    /// toward the normal reports something approaching `π` instead.
    ///
    /// At small counts deliberately. This is the bug the convergence leg of
    /// `gg-tools bounce` found — a stratification denominator that was a
    /// constant rather than the sample count — and every closed form in this
    /// module was blind to it, because each one integrates something that does
    /// not vary over the part of the hemisphere a short prefix covers.
    #[test]
    fn a_sky_that_varies_is_integrated_at_every_sample_count() {
        let scene = Scene {
            solids: &[],
            suns: &[],
            lamps: &[],
        };
        let normal = sim::Vec3::new(0.3, 0.8, -0.5).try_normalize().unwrap();
        let exact = 2.0 * core::f32::consts::PI / 3.0;
        for samples in [64u32, 128, 512, 1024, 4096] {
            let e = irradiance(
                sim::Vec3::ZERO,
                normal,
                &scene,
                |d| sim::Vec3::splat(normal.dot(d).max(0.0)),
                Params {
                    samples,
                    bounces: 1,
                    ..Params::default()
                },
            );
            assert!(
                (e.x / exact - 1.0).abs() < 0.02,
                "{samples} paths: {} against {exact}",
                e.x
            );
        }
    }

    /// The whole tracer, against the one closed form that grades a *bounce
    /// count*: an enclosure emitting `Le` and reflecting `ρ` truncated at `k`
    /// bounces delivers exactly `πLe(1-ρᵏ)/(1-ρ)`.
    ///
    /// Two albedos and four bounce counts, and the assertion is on the ratio so
    /// a tracer that lost the `π` fails every cell rather than the tolerance
    /// absorbing it. This is the milestone's `furnace` (§6 M33): the answer is
    /// arithmetic, so nothing here has to be right about a scene.
    #[test]
    fn an_emitting_enclosure_sums_its_own_series() {
        for albedo in [0.3f32, 0.7] {
            let solids = enclosure(albedo, 1.0);
            let scene = Scene {
                solids: &solids,
                suns: &[],
                lamps: &[],
            };
            for bounces in [1u32, 2, 4, 8] {
                let e = irradiance(
                    sim::Vec3::new(0.0, -2.0 + 1e-3, 0.0),
                    sim::Vec3::Y,
                    &scene,
                    NO_SKY,
                    Params {
                        samples: 4096,
                        bounces,
                        ..Params::default()
                    },
                );
                let exact =
                    core::f32::consts::PI * (1.0 - albedo.powi(bounces as i32)) / (1.0 - albedo);
                let ratio = e.x / exact;
                assert!(
                    (ratio - 1.0).abs() < 0.02,
                    "albedo {albedo}, {bounces} bounces: {} against {exact}",
                    e.x
                );
            }
        }
    }

    /// A sealed enclosure that emits nothing is black however many times the
    /// light bounces — the control leg, and the one that would catch a `sky`
    /// leaking in through a seam between two slabs.
    #[test]
    fn a_sealed_enclosure_that_emits_nothing_stays_dark() {
        let solids = enclosure(0.9, 0.0);
        let scene = Scene {
            solids: &solids,
            suns: &[],
            lamps: &[],
        };
        let e = irradiance(
            sim::Vec3::new(0.0, -2.0 + 1e-3, 0.0),
            sim::Vec3::Y,
            &scene,
            |_| sim::Vec3::splat(1000.0),
            Params {
                samples: 1024,
                ..Params::default()
            },
        );
        assert!(e.length() < 1e-3, "a sealed box let {e:?} in");
    }

    /// The direct term against the definition: a sun straight overhead delivers
    /// `radiance * cos` and a wall in the way delivers nothing.
    #[test]
    fn a_sun_delivers_its_cosine_and_a_blocker_removes_it() {
        let suns = [Sun {
            direction: -sim::Vec3::Y,
            radiance: sim::Vec3::splat(4.0),
        }];
        let open = Scene {
            solids: &[],
            suns: &suns,
            lamps: &[],
        };
        let tilted = sim::Vec3::new(0.6, 0.8, 0.0).try_normalize().unwrap();
        let e = direct(sim::Vec3::ZERO, tilted, &open);
        assert!((e.x - 4.0 * 0.8).abs() < 1e-4, "{e:?}");

        let lid = [solid(
            sim::Vec3::new(0.0, 5.0, 0.0),
            sim::Vec3::new(50.0, 1.0, 50.0),
            0.0,
            0.0,
        )];
        let shaded = Scene {
            solids: &lid,
            ..open
        };
        assert_eq!(direct(sim::Vec3::ZERO, tilted, &shaded), sim::Vec3::ZERO);
    }

    /// One bounce off a coloured floor is the milestone's whole claim, and it
    /// is checked here as a *ratio* against the closed form for exactly this
    /// arrangement: a point just above an infinite Lambertian plane of albedo
    /// `ρ` lit by irradiance `E` sees the plane fill its hemisphere, so it
    /// receives `ρE` — no geometry term, no π, because the plane's radiance
    /// `ρE/π` integrated over a full hemisphere against the cosine is `πρE/π`.
    #[test]
    fn a_point_over_a_lit_floor_receives_the_floors_albedo_times_its_light() {
        let floor = [solid(
            sim::Vec3::new(0.0, -1.0, 0.0),
            sim::Vec3::new(500.0, 1.0, 500.0),
            0.6,
            0.0,
        )];
        let suns = [Sun {
            direction: -sim::Vec3::Y,
            radiance: sim::Vec3::splat(2.0),
        }];
        let scene = Scene {
            solids: &floor,
            suns: &suns,
            lamps: &[],
        };
        // Facing *down* at the floor from just above it, so the hemisphere the
        // integral covers is the plane and only the plane.
        let e = irradiance(
            sim::Vec3::new(0.0, 1e-2, 0.0),
            -sim::Vec3::Y,
            &scene,
            NO_SKY,
            Params {
                samples: 4096,
                bounces: 1,
                ..Params::default()
            },
        );
        let exact = 0.6 * 2.0;
        assert!((e.x / exact - 1.0).abs() < 0.01, "{} against {exact}", e.x);
    }
}
