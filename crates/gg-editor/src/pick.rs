//! What a click in the viewport is aimed at (§6 M15.4 item 1): a ray, cast
//! sim-side, against the boxes the game declared.
//!
//! **Not an ID buffer.** That would be a render target, a readback and a frame
//! of latency, spent to answer a question the sim already holds the exact answer
//! to — [`Renderable`] *is* centre, rotation and half-extent, so a ray-vs-OBB
//! test over one query is exact rather than approximate.
//!
//! Both ends are `f64` and neither crosses §1.4's membrane: the ray starts at
//! the eye's un-narrowed [`sim::DVec3`] and is tested against un-narrowed
//! centres, so picking at 10^12 m out works for the same reason rendering there
//! does. The one number borrowed from the render side is the field of view,
//! which is a CVar — a ray built from a different one than the frame was drawn
//! with misses by however much the console last moved it, so the caller reads it
//! per tick rather than this module caching one.

use gg_ecs::boundary::{Eye, Renderable};
use gg_math::sim;

/// The camera a viewport was drawn through, in both directions.
///
/// One type rather than a cast function beside a project function, because the
/// two must be inverses: a marker drawn through a different camera than the
/// click was cast through is an outline that sits beside the box it outlines,
/// and nothing about either one alone would look wrong.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Lens {
    origin: sim::DVec3,
    right: sim::DVec3,
    up: sim::DVec3,
    forward: sim::DVec3,
    /// The image plane's half-height: `tan(fov_y / 2)` at unit depth for a
    /// perspective eye, world metres at every depth for an orthographic one.
    extent: f64,
    aspect: f64,
    /// The eye declared `Eye::ortho` (§6 M20): rays are parallel, `extent` is
    /// metres, and depth scales nothing.
    ortho: bool,
    /// Metres in front of the eye that the frame's near plane sits at — where a
    /// segment reaching behind the camera has to be cut.
    pub(crate) near: f64,
}

impl Lens {
    /// The camera at `eye` with `fov_y` radians of vertical view over `aspect`,
    /// clipped at `near` — or, when `eye.ortho` is nonzero, the orthographic
    /// camera that same eye is (§6 M20), where `fov_y` goes unread exactly as
    /// the renderer leaves it unread.
    ///
    /// `aspect` is the *rendered* rectangle's, not the pane's: the two differ by
    /// the rounding `viewport_rect` does, and the picture was drawn with the
    /// first one. `fov_y` and `near` are CVars — read per tick by the caller,
    /// because a ray built from a stale field of view misses by however much the
    /// console last moved it.
    pub(crate) fn new(eye: Eye, fov_y: f64, aspect: f64, near: f64) -> Lens {
        let (right, up, forward) = basis(eye.yaw, eye.pitch);
        let ortho = eye.ortho > 0.0;
        Lens {
            origin: eye.position,
            right,
            up,
            forward,
            extent: match ortho {
                true => f64::from(eye.ortho),
                // `gg_math::sim` and not `std`, per §3's ban: this decides which
                // entity a click selects and a libm the two hosts disagree about
                // would decide it differently.
                false => sim::tan(fov_y * 0.5),
            },
            aspect,
            ortho,
            near,
        }
    }

    /// The ray through `at` — a position **inside** the viewport in `[0, 1]`,
    /// `(0, 0)` at its top-left corner.
    ///
    /// Orthographic rays all travel `forward`; what varies with `at` is where
    /// they start. The perspective shape is the reverse — one origin, a fan of
    /// directions — and everything downstream takes either, because a ray is a
    /// ray.
    pub(crate) fn ray(&self, at: (f64, f64)) -> Ray {
        let x = (at.0 * 2.0 - 1.0) * self.extent * self.aspect;
        // Flipped: `at` counts down the pane and the camera's up axis counts up.
        let y = (1.0 - at.1 * 2.0) * self.extent;
        if self.ortho {
            return Ray {
                origin: self.origin + self.right * x + self.up * y,
                direction: self.forward,
            };
        }
        let direction = self.forward + self.right * x + self.up * y;
        Ray {
            origin: self.origin,
            // Never degenerate — `forward` is unit and the other two are
            // perpendicular to it — but a fallback beats an `unwrap` in a crate
            // that bans them.
            direction: direction.try_normalize().unwrap_or(self.forward),
        }
    }

    /// Where `point` lands, as [`Lens::ray`]'s `at` and the metres in front of
    /// the eye it is at.
    ///
    /// The depth is returned rather than used: a point at or behind the eye has
    /// no position on the pane at all, and which of `near`, "skip it" or "cut
    /// the segment here" is right belongs to whoever is drawing. An
    /// orthographic pane position ignores depth — parallel projection is what
    /// that means — but the depth still orders and still cuts.
    pub(crate) fn project(&self, point: sim::DVec3) -> ((f64, f64), f64) {
        let to = point - self.origin;
        let depth = to.dot(self.forward);
        let scale = self.extent
            * match self.ortho {
                true => 1.0,
                false => depth,
            };
        let at = (
            (to.dot(self.right) / (scale * self.aspect) + 1.0) * 0.5,
            (1.0 - to.dot(self.up) / scale) * 0.5,
        );
        (at, depth)
    }

    /// How many world metres one unit down a pane `height` units tall covers, at
    /// `depth` in front of the eye.
    ///
    /// What sizes a gizmo that has to stay the same size on screen however far
    /// away the thing it is attached to is — the alternative being a handle that
    /// is a pixel wide across a room and fills the window up close. Under an
    /// orthographic eye the answer never depended on `depth`, which is the same
    /// fact as the renderer's equal-width boxes.
    pub(crate) fn metres_per_unit(&self, depth: f64, height: f64) -> f64 {
        let reach = match self.ortho {
            true => self.extent,
            false => self.extent * depth,
        };
        2.0 * reach / height.max(1.0)
    }

    /// `from` moved along the segment to `to` until it is `near` in front of the
    /// eye, for a segment with one end behind the camera. The caller has already
    /// established that the two ends straddle the plane.
    pub(crate) fn clip_near(&self, from: sim::DVec3, to: sim::DVec3) -> sim::DVec3 {
        let (a, b) = (
            (from - self.origin).dot(self.forward),
            (to - self.origin).dot(self.forward),
        );
        // The straddle guarantees the denominator is the gap between the two
        // depths and so is nonzero; a zero would mean both ends are at `near`.
        let span = b - a;
        match span.abs() < f64::EPSILON {
            true => from,
            false => from + (to - from) * ((self.near - a) / span),
        }
    }
}

/// A ray in world space, un-narrowed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ray {
    /// The eye it left, which is [`crate::Editor::eye`]'s answer and not
    /// necessarily the game's (§6 M15.2 item 2).
    pub(crate) origin: sim::DVec3,
    /// Unit length, so a `t` out of [`Ray::hits`] is metres.
    pub(crate) direction: sim::DVec3,
}

impl Ray {
    /// Metres to where this enters `box_`, or `None` for a miss. Zero when the
    /// ray starts inside it, which is a hit and the nearest one there can be.
    ///
    /// The slab test, in the box's own frame: rotating the *ray* by the inverse
    /// is one quaternion multiply against the three a rotated OBB's axes would
    /// otherwise cost, and it leaves the comparison axis-aligned.
    pub(crate) fn hits(&self, box_: &Renderable) -> Option<f64> {
        let inverse = box_.rotation.conjugate();
        let origin = inverse.rotate(self.origin - box_.position);
        let direction = inverse.rotate(self.direction);
        let half = sim::DVec3::new(
            f64::from(box_.half_extent.x).abs(),
            f64::from(box_.half_extent.y).abs(),
            f64::from(box_.half_extent.z).abs(),
        );
        let (mut enter, mut exit) = (f64::NEG_INFINITY, f64::INFINITY);
        for axis in 0..3 {
            let (o, d, h) = (
                component(origin, axis),
                component(direction, axis),
                component(half, axis),
            );
            // Parallel to this pair of slabs: inside them or nowhere. Guarded
            // rather than left to `0.0 / 0.0`, which is a NaN that compares
            // false against everything and would report a miss as a hit.
            if d.abs() < f64::EPSILON {
                if o.abs() > h {
                    return None;
                }
                continue;
            }
            let (near, far) = ((-h - o) / d, (h - o) / d);
            enter = enter.max(near.min(far));
            exit = exit.min(near.max(far));
        }
        // `exit >= 0` is what keeps a box *behind* the eye from being picked;
        // without it every ray hits everything on its line, in both directions.
        (exit >= enter.max(0.0)).then(|| enter.max(0.0))
    }
}

fn component(v: sim::DVec3, axis: usize) -> f64 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

/// The fly camera's basis at `yaw`/`pitch` — right, up, forward — in the one
/// order `gg_render::View::rotation` builds it: yaw about world +Y, then pitch
/// about the rotated right axis, never roll.
///
/// Shared with [`crate::camera`] rather than written twice, because a pick and
/// a picture disagreeing about which way the operator is facing is a bug with no
/// symptom except "it selects the wrong thing near the edges" — and off
/// [`sim::fly_basis`] rather than another copy of the trig, so the editor's idea
/// of forward is the demos' to the bit.
pub(crate) fn basis(yaw: f32, pitch: f32) -> (sim::DVec3, sim::DVec3, sim::DVec3) {
    let (forward, right) = sim::fly_basis(yaw, pitch);
    (right, right.cross(forward), forward)
}

/// A box's eight corners in world space, indexed so that bit 0 is the sign of
/// the local X, bit 1 of Y and bit 2 of Z — which is what makes [`EDGES`] the
/// pairs differing in exactly one bit.
pub(crate) fn corners(box_: &Renderable) -> [sim::DVec3; 8] {
    let half = sim::DVec3::new(
        f64::from(box_.half_extent.x),
        f64::from(box_.half_extent.y),
        f64::from(box_.half_extent.z),
    );
    let mut out = [sim::DVec3::ZERO; 8];
    for (i, corner) in out.iter_mut().enumerate() {
        let sign = |bit: usize| match i & (1 << bit) {
            0 => -1.0,
            _ => 1.0,
        };
        let local = sim::DVec3::new(half.x * sign(0), half.y * sign(1), half.z * sign(2));
        *corner = box_.position + box_.rotation.rotate(local);
    }
    out
}

/// Which pairs of [`corners`] are joined by an edge: the twelve that differ in
/// one axis, four per axis.
pub(crate) const EDGES: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// The nearest [`Renderable`] `ray` hits, or `None`.
///
/// Ties go to the **lowest entity index**, on [`Eye::of`]'s reasoning and not as
/// a formality: two boxes at the same distance is what a duplicate is, and
/// iteration order is archetype order, so without this the answer would move
/// when an unrelated component was added to one of them.
///
/// # Errors
///
/// `None` if the world refuses the query, which one read alone cannot cause.
pub(crate) fn nearest(world: &gg_ecs::World, ray: &Ray) -> Option<gg_ecs::Entity> {
    let query = gg_ecs::Query::<&Renderable>::new().ok()?;
    let mut best: Option<(f64, u32, gg_ecs::Entity)> = None;
    world.each_ref(&query, |entity, box_: &Renderable| {
        let Some(t) = ray.hits(box_) else { return };
        let index = entity.index();
        if best.is_none_or(|(far, other, _)| t < far || (t == far && index < other)) {
            best = Some((t, index, entity));
        }
    });
    best.map(|(_, _, entity)| entity)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const FOV: f64 = 1.0;
    const NEAR: f64 = 0.05;

    fn lens(aspect: f64) -> Lens {
        Lens::new(Eye::ORIGIN, FOV, aspect, NEAR)
    }

    fn box_at(x: f64, y: f64, z: f64) -> Renderable {
        Renderable::boxed(sim::DVec3::new(x, y, z), sim::Vec3::splat(0.5), 0x00ff_ffff)
    }

    /// The centre of the pane looks where the camera looks: down -Z at rest.
    #[test]
    fn the_middle_of_the_viewport_is_the_way_the_camera_faces() {
        let ray = lens(16.0 / 9.0).ray((0.5, 0.5));
        assert_eq!(ray.origin, sim::DVec3::ZERO);
        assert!((ray.direction - sim::DVec3::new(0.0, 0.0, -1.0)).length() < 1e-12);
    }

    /// The two directions are inverses, which is the whole reason they are one
    /// type: a point put on the pane and then cast back through is on the ray.
    /// Checked from a camera that is neither at the origin nor level, since a
    /// mismatched basis is exactly what a rest-pose round trip forgives.
    #[test]
    fn a_projected_point_is_on_the_ray_cast_back_through_it() {
        let eye = Eye::at(sim::DVec3::new(-3.0, 12.0, 40.0), 0.7, -0.35);
        let lens = Lens::new(eye, FOV, 21.0 / 9.0, NEAR);
        for point in [
            sim::DVec3::new(0.0, 0.0, 0.0),
            sim::DVec3::new(-9.0, 14.0, 20.0),
            sim::DVec3::new(4.0, 11.0, 33.0),
        ] {
            let (at, depth) = lens.project(point);
            assert!(depth > 0.0, "{point:?} is in front of the camera");
            let ray = lens.ray(at);
            // The ray is unit, so `depth / cos` metres along it is the point.
            let along = ray.origin + ray.direction * (depth / ray.direction.dot(lens.forward));
            assert!(
                (along - point).length() < 1e-9,
                "{point:?} landed at {along:?}"
            );
        }
    }

    /// The orthographic lens (§6 M20): rays are parallel — one direction, a
    /// plane of origins — and the projection is its inverse from an off-centre,
    /// turned camera, exactly as the perspective round trip is checked.
    #[test]
    fn an_orthographic_lens_casts_parallel_rays_and_projects_back() {
        let eye = Eye {
            ortho: 4.5,
            ..Eye::at(sim::DVec3::new(-3.0, 12.0, 40.0), 0.7, -0.35)
        };
        let lens = Lens::new(eye, FOV, 16.0 / 9.0, NEAR);
        let (centre, corner) = (lens.ray((0.5, 0.5)), lens.ray((0.9, 0.1)));
        assert!(
            (centre.direction - corner.direction).length() < 1e-12,
            "every ray travels the same way"
        );
        assert!(
            (corner.origin - centre.origin).length() > 1.0,
            "and they differ by where they start"
        );
        for point in [
            sim::DVec3::new(0.0, 0.0, 0.0),
            sim::DVec3::new(-9.0, 14.0, 20.0),
            sim::DVec3::new(4.0, 11.0, 33.0),
        ] {
            let (at, depth) = lens.project(point);
            assert!(depth > 0.0, "{point:?} is in front of the camera");
            let along = lens.ray(at);
            let landed = along.origin + along.direction * depth;
            assert!(
                (landed - point).length() < 1e-9,
                "{point:?} landed at {landed:?}"
            );
        }
        // A gizmo under this lens is the same size at any depth — the pick-side
        // statement of "no foreshortening", and what keeps a handle grabbable
        // on whatever slab the operator parked furthest away.
        assert_eq!(
            lens.metres_per_unit(2.0, 360.0),
            lens.metres_per_unit(200.0, 360.0)
        );
    }

    /// A segment reaching behind the camera is cut at the near plane and not at
    /// the eye — cutting at zero puts a corner at a projection that divides by
    /// nothing, which is where an outline becomes a stripe across the window.
    #[test]
    fn a_segment_through_the_camera_is_cut_at_the_near_plane() {
        let lens = lens(1.0);
        let behind = sim::DVec3::new(0.0, 0.0, 4.0);
        let front = sim::DVec3::new(0.0, 0.0, -6.0);
        let cut = lens.clip_near(behind, front);
        assert!((lens.project(cut).1 - NEAR).abs() < 1e-12);
        // And from the other end it lands on the same point: a segment has no
        // direction, and an outline drawn twice from opposite ends would not
        // meet at the corner if this did.
        let back = lens.clip_near(front, behind);
        assert!((cut - back).length() < 1e-9);
    }

    /// Eight distinct corners, twelve edges joining them, and every edge one
    /// half-extent long on exactly one axis — which is the invariant a hand-
    /// written index table gets wrong by transposing two digits.
    #[test]
    fn a_box_has_eight_corners_and_twelve_edges_of_its_own_extents() {
        let mut box_ = box_at(2.0, -1.0, 5.0);
        box_.half_extent = sim::Vec3::new(3.0, 1.0, 0.5);
        let corners = corners(&box_);
        for (i, a) in corners.iter().enumerate() {
            for b in corners.iter().skip(i + 1) {
                assert!((*a - *b).length() > 1e-9, "two corners coincide");
            }
        }
        let mut lengths: Vec<f64> = EDGES
            .iter()
            .map(|(a, b)| (corners[*a] - corners[*b]).length())
            .collect();
        lengths.sort_by(f64::total_cmp);
        // Four of each, and each is the full extent rather than the half.
        for (i, want) in [1.0, 2.0, 6.0].into_iter().enumerate() {
            for length in &lengths[i * 4..i * 4 + 4] {
                assert!((length - want).abs() < 1e-9, "{lengths:?}");
            }
        }
        // And they are the box's own axes once it turns.
        let spun = super::corners(&Renderable {
            rotation: sim::DQuat::from_axis_angle(sim::DVec3::Y, 0.6),
            ..box_
        });
        let mut turned: Vec<f64> = EDGES
            .iter()
            .map(|(a, b)| (spun[*a] - spun[*b]).length())
            .collect();
        turned.sort_by(f64::total_cmp);
        assert!(
            turned
                .iter()
                .zip(&lengths)
                .all(|(a, b)| (a - b).abs() < 1e-9)
        );
    }

    /// Right of centre aims right, above centre aims up — and the aspect ratio
    /// widens the first without touching the second. A pick built with the two
    /// swapped is wrong by the pane's shape and looks nearly right on a square
    /// one, which is why this asserts the asymmetry rather than a sign.
    #[test]
    fn the_corners_open_by_the_fov_and_widen_by_the_aspect() {
        let aspect = 2.0;
        let right = lens(aspect).ray((1.0, 0.5));
        let up = lens(aspect).ray((0.5, 0.0));
        let half = sim::tan(FOV * 0.5);
        // Normalized, so the ratio to the forward component is the tangent.
        assert!((right.direction.x / -right.direction.z - half * aspect).abs() < 1e-12);
        assert!(right.direction.y.abs() < 1e-12);
        assert!((up.direction.y / -up.direction.z - half).abs() < 1e-12);
        assert!(up.direction.x.abs() < 1e-12);
    }

    /// The exit criterion: two boxes on one ray, and the click takes the near
    /// one. Both are hit — asserted, or the test would pass with a far box that
    /// was simply missed.
    #[test]
    fn two_boxes_on_one_ray_select_the_nearer() {
        let mut world = gg_ecs::World::new();
        let (near, far) = (world.spawn(), world.spawn());
        world.insert(far, box_at(0.0, 0.0, -20.0)).unwrap();
        world.insert(near, box_at(0.0, 0.0, -5.0)).unwrap();
        let ray = lens(1.0).ray((0.5, 0.5));
        assert!(ray.hits(&box_at(0.0, 0.0, -20.0)).is_some());
        assert_eq!(nearest(&world, &ray), Some(near));
        // And the far one alone is still reachable, so "nearer" is a choice
        // between two hits rather than the only hit there was.
        let mut lone = gg_ecs::World::new();
        let only = lone.spawn();
        lone.insert(only, box_at(0.0, 0.0, -20.0)).unwrap();
        assert_eq!(nearest(&lone, &ray), Some(only));
    }

    /// A box behind the eye is not picked, however exactly the ray's line runs
    /// through it.
    #[test]
    fn what_is_behind_the_camera_is_not_on_the_ray() {
        let mut world = gg_ecs::World::new();
        let behind = world.spawn();
        world.insert(behind, box_at(0.0, 0.0, 5.0)).unwrap();
        let ray = lens(1.0).ray((0.5, 0.5));
        assert_eq!(nearest(&world, &ray), None);
    }

    /// A click into empty sky hits nothing — the case that clears a selection,
    /// and the one an over-eager slab test reports as a hit at infinity.
    #[test]
    fn a_ray_past_everything_hits_nothing() {
        let mut world = gg_ecs::World::new();
        let entity = world.spawn();
        world.insert(entity, box_at(0.0, 0.0, -5.0)).unwrap();
        // Hard right, well outside a half-metre box five metres out.
        let ray = lens(1.0).ray((1.0, 0.5));
        assert_eq!(nearest(&world, &ray), None);
    }

    /// The rotation is applied to the *box*, and applied **inverted**: a long
    /// thin bar is missed where it lies, hit once turned an eighth of a circle
    /// onto the ray, and missed again when turned the same amount the other way.
    ///
    /// The third leg is the point of the test. A symmetric case cannot tell the
    /// conjugate from the rotation itself — the one substitution that compiles,
    /// looks right, and puts every rotated pick on the mirror image of where the
    /// operator clicked.
    #[test]
    fn a_rotated_box_is_tested_in_its_own_inverted_frame() {
        let mut bar = box_at(0.0, 0.0, -5.0);
        bar.half_extent = sim::Vec3::new(2.0, 0.1, 0.1);
        // Straight down through a point a metre and a half along the axis one
        // turn swings the bar onto, and well off the axis it lies on now.
        let ray = Ray {
            origin: sim::DVec3::new(1.06, 3.0, -6.06),
            direction: sim::DVec3::new(0.0, -1.0, 0.0),
        };
        let turned = |angle: f64| Renderable {
            rotation: sim::DQuat::from_axis_angle(sim::DVec3::Y, angle),
            ..bar
        };
        assert_eq!(
            ray.hits(&bar),
            None,
            "the bar lies along X, and this is not"
        );
        let eighth = core::f64::consts::FRAC_PI_4;
        let t = ray
            .hits(&turned(eighth))
            .expect("turned onto the ray, the bar is under the pointer");
        // Three metres up, and the bar's top face is a tenth of one thick.
        assert!((t - 2.9).abs() < 0.05, "entered at {t}");
        assert_eq!(ray.hits(&turned(-eighth)), None, "and swung the other way");
    }

    /// A ray starting inside a box is a hit at zero, not a miss and not a
    /// negative distance — what a camera flown into the floor does.
    #[test]
    fn a_ray_that_starts_inside_enters_at_zero() {
        let ray = lens(1.0).ray((0.5, 0.5));
        assert_eq!(ray.hits(&box_at(0.0, 0.0, 0.0)), Some(0.0));
    }

    /// The basis is the renderer's, checked against a pitched-and-yawed case
    /// rather than at rest: right stays level whatever the pitch, which is the
    /// property `camera::fly`'s world-up lift and this module's `up` both rest
    /// on.
    #[test]
    fn the_basis_never_rolls() {
        let (right, up, forward) = basis(0.9, -0.4);
        assert!(right.y.abs() < 1e-12, "right is level");
        assert!((right.length() - 1.0).abs() < 1e-12);
        assert!((forward.length() - 1.0).abs() < 1e-12);
        assert!((up.length() - 1.0).abs() < 1e-12);
        assert!(right.dot(forward).abs() < 1e-12);
        assert!(up.y > 0.0, "up points up");
    }
}
