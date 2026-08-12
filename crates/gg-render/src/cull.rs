//! Whether a sphere reaches a shadow view, in one verdict for both kinds of
//! view (§6 M32).
//!
//! §6 M15.3 culled the sun's casters per cascade and §6 M31 culled a lamp's per
//! face, each with its own predicate answering yes or no. Both are sphere tests
//! against a convex volume, and a *batch* needs a third answer neither offered:
//! whether the sphere is **wholly** inside, which is what lets a run of
//! instances be drawn through its own untouched range rather than walked and
//! copied. So the two predicates became one function with three outcomes, and
//! [`ScenePass`](crate::scene::ScenePass) asks it per batch before it asks it
//! per instance.
//!
//! The soundness argument the pass rests on: an instance's sphere is contained
//! in its batch's by construction, and both volumes here are intersections of
//! halfspaces and a ball. A sphere sitting `radius` inside every one of those
//! therefore contains no sub-sphere that is not, so [`Fit::Inside`] implies
//! every instance would have passed — the walk it skips could not have rejected
//! anything.

use gg_math::render;

use crate::{cvars, lamp, lighting};

/// How a sphere sits against a shadow view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fit {
    /// Nothing of it reaches: the draw goes away entirely.
    Outside,
    /// Some of it might: a batch has to test its instances one at a time and
    /// compact the survivors.
    Partial,
    /// All of it does, so a batch needs neither the walk nor the copy.
    Inside,
}

/// What the last frame's shadow cull came to (§6 M32).
///
/// Public, for [`ClusterLoad`](crate::ClusterLoad)'s reason one milestone along:
/// "every batch is no longer drawn into every view" is the claim this exists
/// for, and a claim nothing can count is a claim nobody checks. A cull that
/// quietly began passing everything through renders an identical picture — the
/// golden suite cannot see it, and only a number can. `gg-tools lamps` prints
/// it and `gg-render/tests/cull.rs` gates it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShadowCull {
    /// Instances drawn into a shadow view, summed over every cascade and lamp
    /// face. What the shadow passes cost.
    pub drawn: u32,
    /// Instances a view rejected — what those passes *would* have cost, since
    /// before this milestone every batch was drawn into every view.
    pub rejected: u32,
    /// (batch, view) pairs that produced no draw at all: rejected whole, or
    /// compacted down to nothing.
    pub dropped: u32,
    /// Shadow views this frame — cascades plus lamp faces. The denominator a
    /// caller needs to turn the two counts above into a ratio, without a second
    /// constant to import.
    pub views: u32,
}

impl ShadowCull {
    /// Both passes' figures as one. `views` is the same list walked twice, so it
    /// is the larger rather than the sum.
    pub(crate) fn merge(self, other: ShadowCull) -> ShadowCull {
        ShadowCull {
            drawn: self.drawn + other.drawn,
            rejected: self.rejected + other.rejected,
            dropped: self.dropped + other.dropped,
            views: self.views.max(other.views),
        }
    }
}

/// The sun's basis, as the slab cull needs it.
///
/// Taken apart from [`lighting::Sun`] rather than borrowed whole because the
/// cull reads three vectors of it and a `Sun` carries four cascades — and
/// because a test can then state a basis in one line instead of fitting one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Basis {
    /// The direction the light travels, unit.
    pub(crate) direction: render::Vec3,
    /// The two axes across the slab. The cull has to know which way the slab's
    /// *square* is turned: a distance from its axis cannot tell a corner from
    /// an edge.
    pub(crate) right: render::Vec3,
    pub(crate) above: render::Vec3,
}

impl Basis {
    pub(crate) fn of(sun: &lighting::Sun) -> Basis {
        Basis {
            direction: sun.direction,
            right: sun.right,
            above: sun.above,
        }
    }
}

/// One shadow view: a cascade's slab, or one face of one lamp.
#[derive(Clone, Copy, Debug)]
pub(crate) enum View<'a> {
    /// The sun's slab, which is a **prism**: a caster outside it but between it
    /// and the sun still shadows into it, so the along-light extent is checked
    /// against the depth the map records rather than against the slab's own.
    Slab {
        cascade: &'a lighting::Cascade,
        basis: Basis,
    },
    /// One face's pyramid through the lamp, and the lamp's own reach.
    Face { lamp: &'a lamp::Lamp, face: usize },
}

impl View<'_> {
    /// Where the sphere at `centre` of `radius` sits in this view.
    ///
    /// `r.shadow_cull 0` answers [`Fit::Inside`] for everything, which is the
    /// diagnostic off switch meaning exactly what it says: every caster in every
    /// view, and no compaction to be wrong about either. An artifact that
    /// survives it is the *fit's* rather than a cull's.
    pub(crate) fn fit(&self, centre: render::Vec3, radius: f32) -> Fit {
        if cvars::SHADOW_CULL.int() == 0 {
            return Fit::Inside;
        }
        match *self {
            View::Slab { cascade, basis } => slab(cascade, &basis, centre, radius),
            View::Face { lamp, face } => lamp.fit(centre, radius, face),
        }
    }
}

/// The cascade prism (§6 M15.3), as three independent extents.
///
/// The footprint is a **square**, not a disc, and mistaking the two was a real
/// defect rather than a tightness argument: `lighting::fit` builds the
/// projection with `orthographic_reverse_z(radius, radius, ..)`, so the slab
/// reaches `radius` up each axis and `radius * sqrt(2)` into its corners. A cull
/// testing distance from the slab's axis dropped every caster whose shadow
/// landed in a corner — 21% of the footprint, sliding across the world as the
/// camera turned (`gg-tools shadow-sweep`).
fn slab(cascade: &lighting::Cascade, basis: &Basis, centre: render::Vec3, radius: f32) -> Fit {
    let to = centre - cascade.centre;
    // `radius * 2` each way: the eye sits two radii up-light of the centre and
    // the far plane four, so that is exactly the depth the map records.
    let depth = cascade.radius * 2.0;
    let signed = to.dot(basis.direction);
    let along = signed.abs();
    if along > depth + radius {
        return Fit::Outside;
    }
    let across = to - basis.direction * signed;
    let (x, y) = (across.dot(basis.right).abs(), across.dot(basis.above).abs());
    // Distance from the *square*, per axis: zero inside it, and the shortest way
    // out of it at a corner. A sphere reaches the slab iff that is under `radius`.
    let out = |d: f32| (d - cascade.radius).max(0.0);
    if out(x) * out(x) + out(y) * out(y) > radius * radius {
        return Fit::Outside;
    }
    if along + radius <= depth && x + radius <= cascade.radius && y + radius <= cascade.radius {
        return Fit::Inside;
    }
    Fit::Partial
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lit straight down, so the slab's square lies in the x/z plane.
    const DOWN: Basis = Basis {
        direction: render::Vec3::NEG_Y,
        right: render::Vec3::X,
        above: render::Vec3::Z,
    };

    fn slab_of(radius: f32) -> lighting::Cascade {
        lighting::Cascade {
            view_projection: render::Mat4::IDENTITY,
            centre: render::Vec3::ZERO,
            radius,
            texel_world: 0.0,
            split_far: 0.0,
            depth_world: 0.0,
        }
    }

    #[test]
    fn a_sphere_well_inside_a_slab_needs_no_per_instance_walk() {
        let cascade = slab_of(10.0);
        assert_eq!(slab(&cascade, &DOWN, render::Vec3::ZERO, 1.0), Fit::Inside);
        // Straddling the edge is *not* Inside: one instance under that sphere
        // could still sit outside, which is the whole distinction.
        let edge = render::Vec3::new(9.5, 0.0, 0.0);
        assert_eq!(slab(&cascade, &DOWN, edge, 1.0), Fit::Partial);
        let gone = render::Vec3::new(20.0, 0.0, 0.0);
        assert_eq!(slab(&cascade, &DOWN, gone, 1.0), Fit::Outside);
    }

    #[test]
    fn the_footprint_is_a_square_so_a_corner_is_kept() {
        // 14 m from the axis but inside the square's corner: a disc test would
        // drop this and take 21% of the shadow with it.
        let corner = render::Vec3::new(9.9, 0.0, 9.9);
        assert_eq!(slab(&slab_of(10.0), &DOWN, corner, 0.1), Fit::Inside);
    }

    #[test]
    fn a_caster_between_the_slab_and_the_sun_still_reaches_it() {
        let cascade = slab_of(10.0);
        // Straight up-light and inside the depth the map records: it shadows
        // into the slab though it is nowhere near it. Partial rather than
        // Inside — the sphere sits at the far edge of that depth.
        let above = render::Vec3::new(0.0, 19.0, 0.0);
        assert_ne!(slab(&cascade, &DOWN, above, 0.5), Fit::Outside);
        let higher = render::Vec3::new(0.0, 40.0, 0.0);
        assert_eq!(slab(&cascade, &DOWN, higher, 0.5), Fit::Outside);
    }

    #[test]
    fn an_instance_inside_the_batch_sphere_passes_whatever_the_batch_did() {
        // The soundness claim in this module's header, as a sweep: wherever the
        // batch sphere is Inside, every sub-sphere of it is at worst Partial —
        // never Outside, which is the verdict that would have lost a shadow.
        let cascade = slab_of(6.0);
        for i in 0..40 {
            for j in 0..40 {
                let centre = render::Vec3::new(i as f32 * 0.5 - 10.0, 3.0, j as f32 * 0.5 - 10.0);
                let radius = 1.5;
                if slab(&cascade, &DOWN, centre, radius) != Fit::Inside {
                    continue;
                }
                // Every offset that keeps the sub-sphere inside the batch's.
                for k in 0..8 {
                    let dir = render::Vec3::new(
                        if k & 1 == 0 { 1.0 } else { -1.0 },
                        if k & 2 == 0 { 1.0 } else { -1.0 },
                        if k & 4 == 0 { 1.0 } else { -1.0 },
                    )
                    .normalize();
                    let inner = 0.4;
                    let at = centre + dir * (radius - inner);
                    assert_ne!(
                        slab(&cascade, &DOWN, at, inner),
                        Fit::Outside,
                        "sub-sphere at {at:?} of a batch Inside at {centre:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_off_switch_keeps_everything_and_compacts_nothing() {
        let cascade = slab_of(1.0);
        let view = View::Slab {
            cascade: &cascade,
            basis: DOWN,
        };
        let far = render::Vec3::splat(1000.0);
        assert_eq!(view.fit(far, 0.1), Fit::Outside);
        cvars::SHADOW_CULL.set_int(0);
        assert_eq!(view.fit(far, 0.1), Fit::Inside);
        cvars::SHADOW_CULL.set_int(1);
    }
}
