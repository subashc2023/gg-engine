//! Which lights reach which part of the view (§6 M30).
//!
//! Until this milestone the forward pass looped over *every* light in the frame
//! for every fragment, which is why [`MAX_POINT`](gg_extract::MAX_POINT) was 32:
//! the cap was a per-pixel cost. The frustum is cut into froxels here, each one
//! gets the list of lights whose sphere of influence touches it, and a fragment
//! reads its own froxel's list instead of the frame's.
//!
//! # The property that makes this free
//!
//! The point falloff reaches **exactly zero** at
//! [`Light::range`](gg_ecs::boundary::Light::range) — documented there, and
//! asserted. So a light the old loop considered and rejected added `0.0` to the
//! sum, and skipping it instead changes no bit. Assignment therefore has one
//! obligation and it is one-sided: **it may over-include and must never
//! under-include.** Everything below is arranged around that asymmetry, and it
//! is why the twenty scenes blessed before this milestone do not move by a
//! pixel.
//!
//! The two sides do not compute the froxel index with the same instructions —
//! the host's `log2` is `gg_math::sim`'s and the shader's is the driver's — so
//! every sphere is widened by [`SLACK`] before it is tested. Over-inclusion
//! costs a fragment one iteration; under-inclusion is a seam along a froxel
//! boundary, which is the failure this is bought against.

use gg_extract::{ExtractedLight, light};
use gg_math::{render, sim};

use crate::View;

/// Froxels across, down, and through. [Ols12]'s scheme at [Col16]'s dimensions —
/// the depth axis is where the resolution has to be, because a screen tile
/// spanning near and far geometry is exactly what tiled shading gets wrong.
///
///   [Ols12] Olsson, Billeter & Assarsson, "Clustered Deferred and Forward
///           Shading", HPG 2012.
///   [Col16] Collin, "Doom (2016) — Graphics Study" / id Tech 6's 16x8x24.
pub(crate) const GRID: (usize, usize, usize) = (16, 8, 24);

/// Froxels in the grid. Fixed at compile time so the buffer is, and so the
/// shader's constants can be asserted against these rather than uploaded.
pub(crate) const CLUSTERS: usize = GRID.0 * GRID.1 * GRID.2;

/// Light indices the whole grid may hold, over every froxel.
///
/// A pair `(froxel, light)` is four bytes, so this is 128 KiB a frame — the
/// same order as one shadow cascade, and written only up to what was used. A
/// frame that wants more drops the pairs it could not fit **in light order**,
/// which is extract's own rule arriving one layer down: the array reaching this
/// point is already sorted nearest-first, so what a full grid loses is the
/// farthest light in the busiest froxel.
pub(crate) const MAX_INDICES: usize = 32768;

/// Depth the slice distribution is spread over, metres, when the projection has
/// no far plane of its own.
///
/// **Distribution and never correctness.** The last slice extends to infinity
/// whatever this says — a fragment beyond it clamps into it and finds a list
/// built for a froxel that reaches it. Changing this moves where the resolution
/// is, not what is in the lists.
const FAR: f32 = 1024.0;

/// How much every sphere is widened before it is tested, as a fraction of its
/// own radius and as an absolute floor in metres.
///
/// Covers the host and the shader disagreeing about which froxel a boundary
/// belongs to. One part in ten thousand is some four orders above any plausible
/// `log2` divergence and still far below a froxel.
const SLACK: f32 = 1e-4;

/// The view frustum cut into froxels: enough to say which ones a sphere touches,
/// and the two numbers the shader needs to find its own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Grid {
    /// Camera basis. Camera-relative positions project onto these; there is no
    /// translation, because the eye is the origin (§1.4).
    ///
    /// **Down and not up**, which is the one place this file has to know a
    /// convention: both projections flip Y because Vulkan's clip space is
    /// Y-down, and the tile a fragment reads is indexed off `SV_Position`, which
    /// counts rows downward. A grid built on the world's up would be mirrored
    /// against the one the shader looks in — and mirrored is the failure mode
    /// that still lights most of the screen correctly.
    right: render::Vec3,
    down: render::Vec3,
    forward: render::Vec3,
    /// Half-width and half-height of the view at unit depth under perspective,
    /// and in metres outright under an orthographic eye.
    half: (f32, f32),
    /// Zero renders perspective — [`View::ortho`]'s contract, carried here
    /// because the froxel is a wedge under one projection and a box under the
    /// other.
    ortho: bool,
    near: f32,
    far: f32,
    /// `slice = log2(z) * scale + bias`, floored.
    ///
    /// Logarithmic under *both* projections, and that is a simplification worth
    /// saying out loud: the distribution is derived for perspective, where a
    /// screen tile's world footprint grows with depth. An orthographic eye's
    /// does not, so log slices merely put more resolution near the camera than
    /// it needs — non-optimal, never wrong, and one expression instead of a
    /// branch the shader would also have to carry.
    scale: f32,
    bias: f32,
    /// Froxels per pixel, across and down — what the shader multiplies
    /// `SV_Position` by.
    tile: (f32, f32),
}

impl Grid {
    /// Fit the grid to what the frame will be rendered with.
    ///
    /// `extent` is the *view* extent — the attachment the forward pass covers,
    /// which under a viewport is not the surface (`crate::view_extent`).
    pub(crate) fn new(view: &View, extent: (u32, u32)) -> Self {
        let rotation = view.rotation();
        let aspect = extent.0.max(1) as f32 / extent.1.max(1) as f32;
        let near = view.near.max(1e-4);
        let ortho = view.ortho > 0.0;
        let far = match ortho {
            true => view.ortho_far.max(near * 2.0),
            false => FAR.max(near * 2.0),
        };
        let half_y = match ortho {
            true => view.ortho,
            // The same half-angle `perspective_reverse_z` is built from, so a
            // froxel's wedge is the projection's and not an approximation of it.
            false => sim::tan(view.fov_y * 0.5),
        };
        let slices = GRID.2 as f32;
        let span = sim::log2(far / near);
        Grid {
            right: rotation * render::Vec3::X,
            down: rotation * render::Vec3::NEG_Y,
            forward: rotation * render::Vec3::NEG_Z,
            half: (half_y * aspect, half_y),
            ortho,
            near,
            far,
            scale: slices / span,
            bias: -slices * sim::log2(near) / span,
            tile: (
                GRID.0 as f32 / extent.0.max(1) as f32,
                GRID.1 as f32 / extent.1.max(1) as f32,
            ),
        }
    }

    /// What the frame block carries: the basis, the two slice constants, and the
    /// froxels-per-pixel pair.
    pub(crate) fn shader_terms(&self) -> ([f32; 3], f32, f32, [f32; 2]) {
        (
            self.forward.to_array(),
            self.scale,
            self.bias,
            [self.tile.0, self.tile.1],
        )
    }

    /// View-space depth of the front face of slice `k`.
    fn slice_depth(&self, k: usize) -> f32 {
        self.near * sim::exp2(k as f32 / self.scale.max(f32::MIN_POSITIVE))
    }

    /// The slice a depth falls in, floored and clamped — the host's copy of the
    /// shader's `cluster_of`.
    fn slice_of(&self, z: f32) -> usize {
        let raw = sim::log2(z.max(self.near)) * self.scale + self.bias;
        (raw.max(0.0) as usize).min(GRID.2 - 1)
    }

    /// Normalized horizontal position of a view-space `x` at depth `z`: -1 at
    /// the left edge of the frustum and +1 at the right. Depth divides under
    /// perspective and does not under an orthographic eye, which is the whole of
    /// the difference between the two froxel shapes.
    fn across(&self, x: f32, z: f32, half: f32) -> f32 {
        match self.ortho {
            true => x / half,
            false => x / (z * half),
        }
    }

    /// Tile index a normalized coordinate lands in, unclamped, so a caller can
    /// see that a range fell entirely outside the grid.
    fn tile_of(n: f32, count: usize) -> i32 {
        ((n + 1.0) * 0.5 * count as f32).floor() as i32
    }
}

/// What the last frame's assignment came to (§6 M30).
///
/// Public, and for [`Renderer::draw_counts`](crate::Renderer::draw_counts)'s
/// reason: "a fragment iterates its own froxel rather than the frame" is the
/// claim this milestone exists for, and a claim nothing can count is a claim
/// nobody checks. `gg-tools lights` sweeps it and
/// `gg-render/tests/clusters.rs` uses it as the control that stops its equality
/// from passing on a grid that let everybody in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClusterLoad {
    /// `(froxel, light)` pairs written — the loop the whole grid costs, against
    /// the [`froxels`](ClusterLoad::froxels) times the light count an
    /// unclustered frame would.
    pub pairs: usize,
    /// Froxels in the grid. Carried so a caller comparing against "every light
    /// everywhere" has the denominator without a second constant to import — and
    /// so the grid's dimensions stay this crate's to change.
    pub froxels: usize,
    /// The busiest froxel's run. What a *pixel* can be made to pay, which is the
    /// number a frame time is a function of.
    pub worst: u32,
    /// Pairs [`MAX_INDICES`] had no room for.
    pub dropped: usize,
}

/// One frame's froxel lists: where each froxel's run starts, how long it is, and
/// the runs themselves.
///
/// Reused across frames — the two vectors are cleared and refilled, because this
/// is rebuilt every frame and nothing waits on it.
#[derive(Debug, Default)]
pub(crate) struct Assignment {
    /// `(offset, count)` per froxel, in `k * Y * X + j * X + i` order — a slice
    /// contiguous, which is the order the build walks.
    ranges: Vec<[u32; 2]>,
    /// Indices into the frame's light array, ascending within each run.
    ///
    /// Ascending is not cosmetic: it makes a froxel's list a *subsequence* of
    /// the order the old loop summed in, so the floating-point sum is the same
    /// sum and not merely a close one.
    indices: Vec<u32>,
    /// `(froxel, light)` pairs [`MAX_INDICES`] had no room for.
    dropped: usize,
    /// Scratch, kept across frames for its allocation.
    pairs: Vec<(u32, u32)>,
    counts: Vec<u32>,
}

impl Assignment {
    pub(crate) fn ranges(&self) -> &[[u32; 2]] {
        &self.ranges
    }

    pub(crate) fn indices(&self) -> &[u32] {
        &self.indices
    }

    pub(crate) fn load(&self) -> ClusterLoad {
        ClusterLoad {
            pairs: self.indices.len(),
            froxels: CLUSTERS,
            worst: self.ranges.iter().map(|[_, n]| *n).max().unwrap_or(0),
            dropped: self.dropped,
        }
    }

    /// Assign `lights` to froxels. Directional lights are **not** assigned: they
    /// reach everything, so a list holding them would hold all of them
    /// everywhere — the shader loops those out of the front of the light array
    /// instead, which is where extract already puts them.
    ///
    /// `clustered` false gives every froxel the same run: the whole point-light
    /// array, once. That is the pre-M30 loop as a *value* of this function
    /// rather than as a second code path — which is what lets the equality
    /// between the two be a test that renders (`tests/clusters.rs`) instead of a
    /// claim about deleted code.
    pub(crate) fn build(&mut self, grid: &Grid, lights: &[ExtractedLight], clustered: bool) {
        self.ranges.clear();
        self.indices.clear();
        self.dropped = 0;
        // Extract's contract: directional lights are the front of the array
        // (`Extracted::append_lights`), which is what lets the shader loop them
        // by count. Nothing here relies on it — a kind that is not `POINT` is
        // skipped wherever it sits — but a caller that broke it would have a
        // light neither loop reached, so it is asserted where a test would see
        // it and a frame would not pay for it.
        debug_assert!(
            lights
                .iter()
                .skip_while(|l| l.kind == light::DIRECTIONAL)
                .all(|l| l.kind != light::DIRECTIONAL),
            "directional lights must be the prefix of the array",
        );
        if !lights.iter().any(|l| l.kind == light::POINT) {
            self.ranges.resize(CLUSTERS, [0, 0]);
            return;
        }

        if !clustered {
            self.indices.extend(
                lights
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| l.kind == light::POINT)
                    .map(|(i, _)| i as u32),
            );
            self.ranges.resize(CLUSTERS, [0, self.indices.len() as u32]);
            return;
        }

        self.pairs.clear();
        for (index, light) in lights.iter().enumerate() {
            if light.kind != light::POINT {
                continue;
            }
            self.touching(grid, light, index as u32);
        }

        // Counting sort by froxel. Stable by construction — the pairs were
        // pushed in light order and are read back in it — which is what keeps
        // each run ascending.
        let Assignment {
            ranges,
            indices,
            pairs,
            counts,
            ..
        } = self;
        counts.clear();
        counts.resize(CLUSTERS + 1, 0);
        for &(cluster, _) in pairs.iter() {
            counts[cluster as usize + 1] += 1;
        }
        for i in 0..CLUSTERS {
            counts[i + 1] += counts[i];
        }
        ranges.extend((0..CLUSTERS).map(|i| [counts[i], counts[i + 1] - counts[i]]));
        indices.resize(pairs.len(), 0);
        for &(cluster, light) in pairs.iter() {
            let at = &mut counts[cluster as usize];
            indices[*at as usize] = light;
            *at += 1;
        }
    }

    /// Push a pair for every froxel `light`'s sphere reaches.
    fn touching(&mut self, grid: &Grid, light: &ExtractedLight, index: u32) {
        let centre = light.offset;
        let radius = light.range * (1.0 + SLACK) + SLACK;
        let view = render::Vec3::new(
            centre.dot(grid.right),
            centre.dot(grid.down),
            centre.dot(grid.forward),
        );
        let (lo, hi) = (view.z - radius, view.z + radius);
        // Behind the eye entirely, or beyond nothing: the frustum cull upstream
        // already dropped what misses the view, and this is the residue of it.
        if hi <= 0.0 {
            return;
        }
        for k in grid.slice_of(lo.max(grid.near))..=grid.slice_of(hi) {
            // The slice intersected with the sphere's own depth range, which is
            // what keeps the last slice's box finite: it reaches infinity, and
            // a sphere does not.
            let z0 = grid.slice_depth(k).max(lo);
            let z1 = match k + 1 == GRID.2 {
                true => hi,
                false => grid.slice_depth(k + 1).min(hi),
            };
            if z0 > z1 {
                continue;
            }
            let (i0, i1) = span(grid, view.x, radius, z0, z1, grid.half.0, GRID.0);
            let (j0, j1) = span(grid, view.y, radius, z0, z1, grid.half.1, GRID.1);
            for j in j0..=j1 {
                for i in i0..=i1 {
                    if !reaches(grid, view, radius, (i, j), z0, z1) {
                        continue;
                    }
                    if self.pairs.len() == MAX_INDICES {
                        self.dropped += 1;
                        continue;
                    }
                    let cluster = (k * GRID.1 + j) * GRID.0 + i;
                    self.pairs.push((cluster as u32, index));
                }
            }
        }
    }
}

/// Tiles a sphere can reach along one axis, over the depth range `z0..z1`.
///
/// Conservative on purpose: it bounds the sphere by its extreme normalized
/// coordinates at the two ends of the range, which over-counts the corners.
/// [`reaches`] is what makes the answer tight.
fn span(
    grid: &Grid,
    centre: f32,
    radius: f32,
    z0: f32,
    z1: f32,
    half: f32,
    count: usize,
) -> (usize, usize) {
    let lo = grid
        .across(centre - radius, z0, half)
        .min(grid.across(centre - radius, z1, half));
    let hi = grid
        .across(centre + radius, z0, half)
        .max(grid.across(centre + radius, z1, half));
    let last = count as i32 - 1;
    (
        Grid::tile_of(lo, count).clamp(0, last) as usize,
        Grid::tile_of(hi, count).clamp(0, last) as usize,
    )
}

/// Does the sphere reach the froxel in tile `(i, j)`, whose depth range is
/// already cut to the sphere's?
///
/// Sphere against the froxel's *bounding box* rather than against the wedge
/// itself. That over-includes near the corners of a deep froxel and is the
/// permitted direction — the box contains the wedge, so a sphere the box rejects
/// cannot touch the wedge.
fn reaches(
    grid: &Grid,
    view: render::Vec3,
    radius: f32,
    (i, j): (usize, usize),
    z0: f32,
    z1: f32,
) -> bool {
    let (x0, x1) = extent(grid, i, GRID.0, grid.half.0, z0, z1);
    let (y0, y1) = extent(grid, j, GRID.1, grid.half.1, z0, z1);
    let dx = (x0 - view.x).max(view.x - x1).max(0.0);
    let dy = (y0 - view.y).max(view.y - y1).max(0.0);
    // Depth was cut to the sphere's own range by the caller, so the z term is
    // zero by construction and is not recomputed here.
    dx * dx + dy * dy <= radius * radius
}

/// One tile's view-space extent along an axis, over `z0..z1`.
fn extent(grid: &Grid, i: usize, count: usize, half: f32, z0: f32, z1: f32) -> (f32, f32) {
    let lo = (2.0 * i as f32 / count as f32 - 1.0) * half;
    let hi = (2.0 * (i + 1) as f32 / count as f32 - 1.0) * half;
    match grid.ortho {
        true => (lo, hi),
        // Each edge scales with depth, so the box's is the wider of the two ends
        // — which for a negative edge is the *far* one.
        false => (
            lo * if lo >= 0.0 { z0 } else { z1 },
            hi * if hi >= 0.0 { z1 } else { z0 },
        ),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn a_view() -> View {
        View {
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 1.0,
            near: 0.05,
            ortho: 0.0,
            ortho_far: 100.0,
            aa: false,
            samples: gg_rhi::Samples::X1,
        }
    }

    fn a_light(at: [f32; 3], range: f32) -> ExtractedLight {
        ExtractedLight {
            offset: render::Vec3::from_array(at),
            direction: render::Vec3::ZERO,
            color: 0x00FF_FFFF,
            intensity: 1.0,
            range,
            kind: light::POINT,
        }
    }

    fn a_sun() -> ExtractedLight {
        ExtractedLight {
            offset: render::Vec3::ZERO,
            direction: render::Vec3::NEG_Y,
            color: 0x00FF_FFFF,
            intensity: 1.0,
            range: 0.0,
            kind: light::DIRECTIONAL,
        }
    }

    /// The froxel a point lands in, by the host's own rule — the shader's
    /// `cluster_of` on this side of the wire.
    fn cluster_of(grid: &Grid, world: render::Vec3, extent: (u32, u32)) -> usize {
        let clip = a_view().view_projection(extent) * world.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        let pixel = (
            (ndc.x * 0.5 + 0.5) * extent.0 as f32,
            (ndc.y * 0.5 + 0.5) * extent.1 as f32,
        );
        let i = ((pixel.0 * grid.tile.0) as usize).min(GRID.0 - 1);
        let j = ((pixel.1 * grid.tile.1) as usize).min(GRID.1 - 1);
        let k = grid.slice_of(world.dot(grid.forward));
        (k * GRID.1 + j) * GRID.0 + i
    }

    /// The obligation, stated as a test: a light present in *any* froxel the old
    /// loop would have lit must be present in the froxel the fragment reads.
    ///
    /// Sampled over a lattice of world points rather than over froxels, because
    /// the froxel a fragment reads is the shader's business and a test that
    /// asked the same function twice would prove nothing.
    #[test]
    fn a_light_is_in_every_froxel_it_reaches() {
        let extent = (640, 360);
        let grid = Grid::new(&a_view(), extent);
        let lights: Vec<ExtractedLight> = vec![
            a_light([0.0, 0.0, -2.0], 1.5),
            a_light([3.0, 1.0, -12.0], 6.0),
            a_light([-1.0, -0.5, -0.6], 0.4),
            a_light([0.0, 0.0, -60.0], 40.0),
        ];
        let mut assignment = Assignment::default();
        assignment.build(&grid, &lights, true);

        let mut checked = 0;
        for xi in -20..=20 {
            for yi in -12..=12 {
                for zi in 1..40 {
                    let z = -0.2 * zi as f32 * zi as f32 / 8.0 - 0.1;
                    let world = render::Vec3::new(xi as f32 * 0.6, yi as f32 * 0.5, z);
                    // Only points the frame can actually shade: outside the
                    // frustum the shader never asks.
                    let clip = a_view().view_projection(extent) * world.extend(1.0);
                    if clip.w <= 0.0
                        || clip.x.abs() > clip.w
                        || clip.y.abs() > clip.w
                        || clip.z < 0.0
                    {
                        continue;
                    }
                    let cluster = cluster_of(&grid, world, extent);
                    let [offset, count] = assignment.ranges()[cluster];
                    let run =
                        &assignment.indices()[offset as usize..offset as usize + count as usize];
                    for (index, light) in lights.iter().enumerate() {
                        let reach = (light.offset - world).length();
                        if reach >= light.range {
                            continue;
                        }
                        checked += 1;
                        assert!(
                            run.contains(&(index as u32)),
                            "light {index} lights {world} and is absent from froxel {cluster}",
                        );
                    }
                }
            }
        }
        assert!(checked > 500, "the lattice missed the lights: {checked}");
    }

    /// And the reason it is worth doing: over-inclusion is bounded. A froxel
    /// holding every light would be the old loop with extra steps.
    #[test]
    fn a_froxel_holds_far_fewer_lights_than_the_frame_does() {
        let grid = Grid::new(&a_view(), (640, 360));
        // Lamps down a hall — spread through *depth* as well as across the
        // screen, which is the arrangement clustering exists for and the one a
        // purely tiled grid cannot help with. A flat wall of them all at one
        // depth is the honest worst case and is measured by `gg-tools lights`
        // rather than asserted here; it is a scene, not a property.
        let lights: Vec<ExtractedLight> = (0..128)
            .map(|i| {
                let x = (i % 8) as f32 * 1.6 - 5.6;
                let z = -1.5 - (i / 8) as f32 * 2.0;
                a_light([x, 0.0, z], 2.0)
            })
            .collect();
        let mut assignment = Assignment::default();
        assignment.build(&grid, &lights, true);
        let counts: Vec<u32> = assignment.ranges().iter().map(|[_, n]| *n).collect();
        let worst = counts.iter().copied().max().unwrap();
        let occupied = counts.iter().filter(|n| **n > 0).count();
        assert_eq!(assignment.load().dropped, 0);
        // A tenth of the frame in the busiest froxel, and a fortieth on average
        // over the froxels that hold anything at all. Both are what the loop
        // costs, so both are the claim.
        let total: u32 = counts.iter().sum::<u32>();
        let mean = total as f32 / occupied as f32;
        // Two numbers, because they answer different questions. The **worst**
        // froxel is what a pixel can be made to pay — 24 here, against 128, and
        // it is bounded loosely because it is a property of one arrangement.
        // The **total** is the loop that was removed: 7444 pairs against the
        // 393216 an every-light-everywhere assignment writes, which is the same
        // 2% the shader stops iterating.
        assert!(worst * 4 < 128, "worst froxel holds {worst} of 128");
        assert!(mean < 6.0, "mean occupied froxel holds {mean} of 128");
        assert!(
            total as usize * 40 < CLUSTERS * 128,
            "{total} pairs of a possible {}",
            CLUSTERS * 128,
        );
    }

    /// Directional lights are the front of the array and are never assigned —
    /// the shader loops them by count, not by list.
    #[test]
    fn the_lists_hold_point_lights_only() {
        let grid = Grid::new(&a_view(), (640, 360));
        let lights = vec![a_sun(), a_sun(), a_light([0.0, 0.0, -3.0], 4.0)];
        let mut assignment = Assignment::default();
        assignment.build(&grid, &lights, true);
        assert!(
            assignment.indices().iter().all(|&i| i == 2),
            "a directional light reached a froxel list",
        );
        let lights = vec![a_sun()];
        assignment.build(&grid, &lights, true);
        assert!(assignment.indices().is_empty());
        assert_eq!(assignment.ranges().len(), CLUSTERS);
    }

    /// The unclustered value: every froxel, the same run, and it is exactly the
    /// point-light array.
    #[test]
    fn unclustered_is_every_light_in_every_froxel() {
        let grid = Grid::new(&a_view(), (640, 360));
        let lights = vec![
            a_sun(),
            a_light([0.0, 0.0, -3.0], 4.0),
            a_light([9.0; 3], 1.0),
        ];
        let mut assignment = Assignment::default();
        assignment.build(&grid, &lights, false);
        assert_eq!(assignment.indices(), &[1, 2]);
        assert!(assignment.ranges().iter().all(|&r| r == [0, 2]));
    }

    /// Under an orthographic eye a froxel is a box rather than a wedge, and a
    /// light far off the axis must not reach the middle of the screen.
    #[test]
    fn an_orthographic_froxel_does_not_widen_with_depth() {
        let view = View {
            ortho: 8.0,
            ortho_far: 100.0,
            ..a_view()
        };
        let grid = Grid::new(&view, (640, 360));
        let lights = vec![a_light([13.0, 0.0, -50.0], 1.0)];
        let mut assignment = Assignment::default();
        assignment.build(&grid, &lights, true);
        let reached: Vec<usize> = assignment
            .ranges()
            .iter()
            .enumerate()
            .filter(|(_, [_, count])| *count > 0)
            .map(|(index, _)| index % GRID.0)
            .collect();
        assert!(!reached.is_empty(), "the light reached nothing at all");
        assert!(
            reached.iter().all(|&i| i >= GRID.0 - 2),
            "an orthographic light 13 m right reached tiles {reached:?}",
        );
    }

    /// A grid that cannot fit its pairs drops them and says how many, rather
    /// than growing or silently truncating a run.
    #[test]
    fn a_full_grid_counts_what_it_dropped() {
        let grid = Grid::new(&a_view(), (640, 360));
        // Every light covering the whole frustum: `MAX_INDICES / CLUSTERS` of
        // them fit, and the rest cannot.
        let fill = MAX_INDICES / CLUSTERS + 2;
        let lights: Vec<ExtractedLight> = (0..fill)
            .map(|_| a_light([0.0, 0.0, -1.0], 4000.0))
            .collect();
        let mut assignment = Assignment::default();
        assignment.build(&grid, &lights, true);
        assert_eq!(assignment.indices().len(), MAX_INDICES);
        assert!(assignment.load().dropped > 0);
    }
}
