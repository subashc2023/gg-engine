//! The irradiance field, and the probes that gather it (§6 M36).
//!
//! §6 M28 gave a level author boxes to draw an environment in, and `gg-tools
//! bounce` measured what that leaves out: in demo 12's room 41.9 % of the
//! indirect light arrives by bouncing off geometry, and the authored volume —
//! right to five per cent *on average* — over-lights the room's dark half by
//! 0.152 and under-lights its bright half by 0.087. It is a constant with a
//! doorway in it, because that is the most a box can be. This is the field that
//! is not authored.
//!
//! # What a probe is here
//!
//! A point that **renders the scene**, six 2D faces of it into tiles of one
//! atlas, exactly the way a casting lamp does (`crate::lamp`, whose [`AXES`]
//! table this shares rather than repeats). The faces are then integrated into
//! two records per probe: the irradiance itself, and how far away the geometry
//! that produced it was.
//!
//! Rendering rather than ray-casting is what makes the field see *whatever the
//! renderer draws* — pack geometry, the lamps, the sky, and next frame's field
//! through last frame's, which is how the second bounce arrives for free. What
//! it costs is that a probe is six draws of the scene, so a frame updates a few
//! of them and the field converges over many. Nothing here hides that: [`
//! Probes::pending`] is the count that has never been gathered, and a reference
//! image rendered while it is nonzero is a reference of a half-built field.
//!
//! # Why the irradiance is L1 and the distance is not
//!
//! Irradiance off a diffuse bounce is low-frequency by construction — a cosine
//! lobe is a 3-band filter, so the fourth band of a projection is already almost
//! zero. Four coefficients per channel, stored as one texel per channel as
//! `(c_y, c_z, c_x, c_0)`, makes a probe's irradiance three texels and an
//! eight-probe blend twenty-four taps of it. `crate::sky`'s nine are for a
//! *sky*, where a bright window is the whole point.
//!
//! An L1 record is not a *truncation* of the answer wherever the surrounding
//! radiance is a hemispherical step — sky above, floor below — which is the case
//! a probe over a floor actually is: the band-1 term and the constant cancel
//! exactly against the cosine kernel, and the surface receives `pi * L_sky`. It
//! is an approximation of everything more structured than that.
//!
//! Distance is the opposite: it is a step function, its whole job is to say
//! which side of a wall a probe is on, and a projection onto four smooth bands
//! would average the wall away. So it is stored per direction, in a small
//! octahedral tile per probe, as [Don05]'s two moments — mean and mean of the
//! square — which is what the shader's Chebyshev bound reads. Point-sampled
//! rather than filtered, which is why the tiles need no border: a bound is soft
//! already, and the seam of an octahedral map is exactly where bilinear would
//! otherwise fetch a direction from the other side of the probe.
//!
//! [Don05] Donnelly & Lauritzen, "Variance Shadow Maps", I3D 2005.

use gg_math::{render, sim};
use gg_rhi::{
    Blend, BufferDesc, BufferHandle, ColorTarget, DepthMode, DeviceAddress, DrawSpec,
    FRAMES_IN_FLIGHT, ImageFormat, PipelineDesc, PipelineHandle, RhiError, Samples, TextureIndex,
};

use crate::cull::Fit;
use crate::lamp::{AXES, FACES};
use crate::shaders_gen::probe as probe_shader;
use crate::{GpuHost, cvars};

/// A probe's faces, and the index space its views live in — `pbr.slang`'s
/// `PROBE_FACES` and `PROBE_VIEW_BASE`, one segment past the lamps'.
pub(crate) use crate::lamp::FACES as PROBE_FACES;
pub(crate) const VIEW_BASE: usize =
    crate::lighting::MAX_CASCADES + crate::lamp::MAX_LAMPS * crate::lamp::FACES;

/// Texels one probe's irradiance record occupies, in a row.
///
/// Three carry L1 per channel and the fourth is the probe's own state — how far
/// the geometry around it was on average, and whether it is inside any. Four
/// rather than three so the state is not a second image with a second layout to
/// keep in step with this one.
pub(crate) const SH_TEXELS: u32 = 4;

/// Where the state texel sits in that row — `pbr.slang`'s
/// `PROBE_STATE_TEXEL`, which is what reads it.
pub(crate) const STATE_TEXEL: u32 = 3;

const _: () = assert!(STATE_TEXEL < SH_TEXELS);

/// Probes per axis, at most.
///
/// **A cost ceiling before it is an atlas one.** A probe is six draws of the
/// scene, so an 8-cube is 512 probes and 3072 renders for the field's first
/// sweep — which is what a harness sits through before it may judge a picture
/// and what a session pays over its first second. A 16-cube is eight times that,
/// and the number it produced was twenty-four thousand renders of a golden scene
/// for one reference image. The *atlas* would have taken 16: its height is
/// `y * z * edge`, and 16 × 16 × 8 is 2048 against §4.3's 4096 floor — so the
/// limit that bites is the one nobody was counting.
///
/// A field wanting more than an 8-cube covers less of the world per probe, not
/// more of it — the grid fits the scene either way. What it cannot do is cover a
/// *large* world at a *fine* spacing, which is the scrolling, camera-anchored
/// grid §6 M36 names as its residual rather than builds.
const MAX_PER_AXIS: u32 = 8;

/// Octahedral edge of a probe's distance tile, at most — see [`MAX_PER_AXIS`].
const MAX_MOMENT_EDGE: u32 = 8;

/// Relative headroom the roundings in [`Grid::fit`] give a bound that lands on
/// one of their risers (§6 M57).
///
/// Every rounding there is a step function, and a bound sitting exactly on a
/// step makes the fit a coin toss. The bounds arrive through §1.4's membrane —
/// an absolute position narrowed to `f32` against the camera origin and widened
/// back — so one that is *exactly* a whole spacing wobbles by an `f32` ULP of its
/// distance as the eye moves, and that is enough. Demo 12's walls top out at
/// exactly `y = 4.0` against a 4 m spacing: the vertical probe count alternated 3
/// and 4, and [`Probes::update`] threw the whole field away on every flip.
/// `gg-tools field` measured sixteen refits in a sixty-frame jump, each moving
/// 6-10 % of the room's light, against zero for the same eye walking and turning.
///
/// **Relative and not absolute**, which the first version got wrong: a
/// thousandth of a spacing is four millimetres at demo 12's, three orders above
/// the noise it exists to absorb, and it moved a golden whose fit was never in
/// doubt. `f32` carries about `6e-8` of relative precision, so `1e-5` is two
/// orders of margin over the wobble and still narrower than any fit decision that
/// means anything — it can only change an answer that was already a coin toss.
const SLACK: f64 = 1e-5;

/// `q` rounded down, counting a value within [`SLACK`] *below* an integer as that
/// integer. `magnitude` is the scale the inputs were computed at, since a `q`
/// that is a difference of large numbers has lost precision that `q` itself no
/// longer shows.
fn floor_steady(q: f64, magnitude: f64) -> f64 {
    (q + SLACK * magnitude.abs().max(1.0)).floor()
}

/// `q` rounded up, counting a value within [`SLACK`] *above* an integer as that
/// integer — [`floor_steady`]'s mirror, and it has to be the mirror: the two are
/// applied to the same bound from opposite sides.
fn ceil_steady(q: f64, magnitude: f64) -> f64 {
    (q - SLACK * magnitude.abs().max(1.0)).ceil()
}

/// Metres the near plane of a probe's face sits at. `crate::lamp`'s reasoning
/// and its value: inside anything a probe could be mounted in, and not so small
/// that reverse-Z spends its precision on the first millimetre.
const NEAR: f32 = 0.05;

/// Probes one frame may gather, whatever `r.gi_rate` asks for.
///
/// The view buffer's ceiling, and therefore the atlas's: a slot is 400 bytes and
/// a batch row of the atlas is six faces, so sixty-four is 25 KiB of buffer and
/// a 48x512 image at the default face size. `r.gi_rate 0` means "as many as this
/// holds" rather than "the whole grid" — the round robin is what makes the
/// difference a number of frames rather than a number of probes.
pub(crate) const MAX_BATCH: usize = 64;

/// Bytes one probe occupies in the view buffer — `pbr.slang`'s
/// `PROBE_SLOT_STRIDE`.
pub(crate) const SLOT_BYTES: u64 = core::mem::size_of::<GpuProbeView>() as u64;

/// Both field images' format.
///
/// `Rgba16F` for the irradiance because a coefficient is a radiance and radiance
/// is what the whole pipeline is in; for the moments because the *square* of a
/// distance is what half of that texel holds, and eight bits of exponent is what
/// keeps a two-metre cell and a two-hundred-metre one in the same image.
pub(crate) const FIELD_FORMAT: ImageFormat = ImageFormat::Rgba16F;

/// Bytes one of its texels takes — four halves.
const FIELD_TEXEL_BYTES: usize = 8;

/// Multiples of the probe spacing a recorded distance is clamped to.
///
/// Two, because that is the furthest a shading point can be from a probe that
/// still weighs in its own cell's interpolation — a moment recorded past it
/// bounds nothing, and clamping keeps the squared term inside `Rgba16F`'s
/// comfortable range rather than spending its exponent on a sky at infinity.
const DISTANCE_CAP: f32 = 2.0;

/// One gathering probe as the shaders read it — `pbr.slang`'s `load_probe_face`
/// and `load_probe_position`, whose `PROBE_SLOT_STRIDE` is this type's size.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuProbeView {
    /// Camera-relative world → face clip, in [`AXES`] order.
    faces: [[[f32; 4]; 4]; FACES],
    /// Where the probe is, camera-relative — the eye the faces shade from and
    /// the point their distances are measured to. Carried rather than recovered
    /// from a matrix, `lamp::Lamp::tan_half`'s reason: `faces` is a product.
    position: [f32; 3],
    /// Zero, to the 16-byte multiple the stride has to be.
    reserved: f32,
}

const _: () = {
    assert!(SLOT_BYTES == 400);
    assert!(core::mem::offset_of!(GpuProbeView, position) == 384);
};

/// What the frame block needs to know about the field this frame (§6 M36).
///
/// Assembled by the renderer rather than by `Probes`, for `lighting::Maps`'
/// reason at one remove: the bindless slots are the graph's to hand over and the
/// camera origin is extract's, so a field that resolved either itself would be
/// writing a block whose lookups land in whatever else is in those slots.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Field {
    pub(crate) sh: TextureIndex,
    pub(crate) moments: TextureIndex,
    pub(crate) grid: Grid,
    pub(crate) moment_edge: u32,
    /// The grid's corner probe, camera-relative.
    pub(crate) origin: render::Vec3,
    /// Metres a shading point is pushed off its surface before it is located.
    pub(crate) bias: f32,
}

/// A probe grid: where it starts, how far apart the probes are, how many.
///
/// The origin is **absolute** (§1.4) and the spacing is a whole multiple of
/// itself away from the world origin, which is what makes the grid a function of
/// the scene rather than of the frame — two frames whose bounds differ by a
/// millimetre fit the same grid, so the field is not thrown away every time
/// something moves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Grid {
    origin: sim::DVec3,
    spacing: f64,
    counts: [u32; 3],
}

impl Grid {
    /// The grid covering `min..max`, snapped out to the spacing.
    ///
    /// **`r.gi_spacing` is a floor and not a promise**, and the measurement that
    /// settled it is `gg-tools bounce`'s field table: with the spacing taken
    /// literally and the counts clamped per axis, demo 12's room got a grid over
    /// one corner of itself — **115 of 786 visible points inside the field** at
    /// the shipped default — and the other 85 % were lit by the flat term M36
    /// exists to replace. "The fallback is honest at a distant corner" was the
    /// argument for clamping and it is true; what it does not survive is the
    /// corner being most of the room. Widened to a whole multiple of the asked-for
    /// spacing, which is what keeps this a function of the scene rather than of
    /// the frame: an integer multiple of a constant, driven by bounds that move
    /// only when the content does.
    pub(crate) fn fit(min: sim::DVec3, max: sim::DVec3, spacing: f64) -> Grid {
        let asked = spacing.max(0.05);
        let reach = f64::from(MAX_PER_AXIS - 1);
        let widest = (max.x - min.x).max(max.y - min.y).max(max.z - min.z);
        // `widest` may be non-finite; `max(1.0)` on a NaN takes the 1.0 branch,
        // and the count below clamps whatever survives that.
        let multiple = if widest.is_finite() {
            let reached = widest / (asked * reach);
            ceil_steady(reached, reached).clamp(1.0, 1e6)
        } else {
            1.0
        };
        let spacing = asked * multiple;
        // Every rounding here is steadied, and a bound on a riser has to fall the
        // *same* way in all of them: a `min` that snapped down a whole spacing
        // while its span rounded up would grow the grid by a probe on one side
        // and lose it on the other.
        let snap = |v: f64| floor_steady(v / spacing, v / spacing) * spacing;
        let origin = sim::DVec3::new(snap(min.x), snap(min.y), snap(min.z));
        let count = |lo: f64, hi: f64| {
            // +1 because a span of one spacing needs two probes to bracket it,
            // and 2 at the floor because one probe is a constant with a position.
            // Clamped *before* the cast rather than after: a span that is not
            // finite — an instance with an infinite radius, a scene bounded by
            // nothing — saturates a `u32` and the `+ 1` below overflows, which
            // is a panic in a renderer instead of a coarse grid.
            // The magnitude is `|hi| + |lo|` and not the span: a span is a
            // *difference*, and one metre between two points a kilometre out
            // carries the precision of the kilometre, not of the metre.
            let spans = ceil_steady((hi - lo) / spacing, (hi.abs() + lo.abs()) / spacing)
                .clamp(0.0, f64::from(MAX_PER_AXIS)) as u32;
            (spans + 1).clamp(2, MAX_PER_AXIS)
        };
        Grid {
            origin,
            spacing,
            counts: [
                count(origin.x, max.x),
                count(origin.y, max.y),
                count(origin.z, max.z),
            ],
        }
    }

    /// Whether `min..max` is inside what this grid reaches — the hysteresis that
    /// keeps a moving crate from refitting the field every frame.
    pub(crate) fn covers(&self, min: sim::DVec3, max: sim::DVec3) -> bool {
        let far = self.corner();
        // Steadied for the same reason `fit` is, and to the same width: a bound
        // sitting exactly on the far corner must not answer this one way and the
        // fit the other. Scaled by the corner's own distance, since that is where
        // the precision went.
        let reach = far
            .x
            .abs()
            .max(far.y.abs())
            .max(far.z.abs())
            .max(self.origin.x.abs())
            .max(self.origin.y.abs())
            .max(self.origin.z.abs());
        let slack = SLACK * (self.spacing + reach);
        min.x >= self.origin.x - slack
            && min.y >= self.origin.y - slack
            && min.z >= self.origin.z - slack
            && max.x <= far.x + slack
            && max.y <= far.y + slack
            && max.z <= far.z + slack
    }

    /// The probe diagonally opposite the origin.
    fn corner(&self) -> sim::DVec3 {
        sim::DVec3::new(
            self.origin.x + f64::from(self.counts[0] - 1) * self.spacing,
            self.origin.y + f64::from(self.counts[1] - 1) * self.spacing,
            self.origin.z + f64::from(self.counts[2] - 1) * self.spacing,
        )
    }

    pub(crate) fn counts(&self) -> [u32; 3] {
        self.counts
    }

    /// Origin, spacing and counts — what [`crate::Renderer::field_grid`] hands a
    /// caller that has to know where the field reaches.
    pub(crate) fn report(&self) -> (sim::DVec3, f32, [u32; 3]) {
        (self.origin, self.spacing as f32, self.counts)
    }

    pub(crate) fn spacing(&self) -> f32 {
        self.spacing as f32
    }

    /// Probes in the grid.
    pub(crate) fn probes(&self) -> usize {
        self.counts.iter().map(|c| *c as usize).product()
    }

    /// Probe `index`'s grid coordinate. The order is x fastest, then y, then z —
    /// the same order [`Grid::sh_row`] lays the field out in, because two
    /// orderings is one more than the shader can be told about.
    pub(crate) fn coord(&self, index: usize) -> [u32; 3] {
        let (x, rest) = (
            index % self.counts[0] as usize,
            index / self.counts[0] as usize,
        );
        let (y, z) = (
            rest % self.counts[1] as usize,
            rest / self.counts[1] as usize,
        );
        [x as u32, y as u32, z as u32]
    }

    /// Where probe `index` sits, **camera-relative** — what a face projection is
    /// built through, and what the shader compares a shading point against.
    pub(crate) fn position(&self, index: usize, camera_origin: sim::DVec3) -> render::Vec3 {
        let [x, y, z] = self.coord(index);
        let at = sim::DVec3::new(
            self.origin.x + f64::from(x) * self.spacing,
            self.origin.y + f64::from(y) * self.spacing,
            self.origin.z + f64::from(z) * self.spacing,
        ) - camera_origin;
        // Narrowed through the camera origin and nowhere else — §1.4's membrane
        // is that an absolute position never becomes an `f32` by any other route.
        render::Vec3::new(at.x as f32, at.y as f32, at.z as f32)
    }

    /// The grid origin the shader locates a shading point against, in the same
    /// camera-relative frame as [`Grid::position`].
    pub(crate) fn origin_relative(&self, camera_origin: sim::DVec3) -> render::Vec3 {
        let at = self.origin - camera_origin;
        render::Vec3::new(at.x as f32, at.y as f32, at.z as f32)
    }

    /// The row of the field images probe `index` lives on: y and z flattened, so
    /// one image holds a volume.
    fn sh_row(&self, index: usize) -> u32 {
        let [_, y, z] = self.coord(index);
        z * self.counts[1] + y
    }

    /// The irradiance image's extent: one probe's row of texels across, one grid
    /// row down.
    pub(crate) fn sh_extent(&self) -> (u32, u32) {
        (self.counts[0] * SH_TEXELS, self.counts[1] * self.counts[2])
    }

    /// Where probe `index`'s irradiance row starts, in texels.
    pub(crate) fn sh_texel(&self, index: usize) -> (u32, u32) {
        let [x, _, _] = self.coord(index);
        (x * SH_TEXELS, self.sh_row(index))
    }

    /// The moment image's extent, at octahedral edge `edge`.
    pub(crate) fn moment_extent(&self, edge: u32) -> (u32, u32) {
        (
            self.counts[0] * edge,
            self.counts[1] * self.counts[2] * edge,
        )
    }

    /// Where probe `index`'s octahedral distance tile starts, in texels.
    pub(crate) fn moment_tile(&self, index: usize, edge: u32) -> gg_rhi::Viewport {
        let [x, _, _] = self.coord(index);
        gg_rhi::Viewport {
            x: x * edge,
            y: self.sh_row(index) * edge,
            width: edge,
            height: edge,
        }
    }
}

/// One probe being gathered this frame: where it is, and the six projections its
/// faces are rendered through.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Gather {
    /// Which probe of the grid.
    pub(crate) index: usize,
    /// Camera-relative, matching the face matrices beside it.
    pub(crate) position: render::Vec3,
    /// Camera-relative world → face clip, in [`AXES`] order.
    pub(crate) faces: [render::Mat4; FACES],
}

impl Gather {
    /// Where a sphere of `radius` at `offset` sits against face `face`.
    ///
    /// [`lamp::Lamp::fit`](crate::lamp::Lamp::fit)'s four side planes with the
    /// reach sphere removed and the tangent fixed at 1 — a probe sees the whole
    /// scene, so nothing about distance rejects a caster here, and its faces are
    /// exactly ninety degrees because nothing filters across their seams.
    ///
    /// The near plane is absent for that function's reason: a caster straddling
    /// the probe must be drawn by whichever faces its parts fall into rather
    /// than rejected by a plane through the middle of it.
    pub(crate) fn fit(&self, offset: render::Vec3, radius: f32, face: usize) -> Fit {
        let to = offset - self.position;
        let (forward, up) = AXES[face];
        let right = forward.cross(up);
        // The side plane's normal is `axis - forward` at tangent 1, whose length
        // is sqrt(2) — the unit the radius is measured in here.
        let scale = radius * core::f32::consts::SQRT_2;
        let along = to.dot(forward);
        let past = |axis: render::Vec3| to.dot(axis).abs() - along;
        let (x, y) = (past(right), past(up));
        if x > scale || y > scale {
            return Fit::Outside;
        }
        if x <= -scale && y <= -scale {
            return Fit::Inside;
        }
        Fit::Partial
    }
}

/// The field: the grid it is gathered on, and which probes have been.
///
/// Refit is a *loss*, so it is deliberately rare — a grid that changed is a
/// field with nothing in it, and [`Probes::pending`] goes back to the full count
/// so a harness can wait the same way it waits for a pack to stream.
#[derive(Clone, Debug, Default)]
pub(crate) struct Probes {
    grid: Option<Grid>,
    /// Where the round robin has got to.
    cursor: usize,
    /// Probes never gathered since the grid was fitted, by index.
    ungathered: Vec<bool>,
    /// This frame's batch, kept so a frame allocates nothing.
    batch: Vec<Gather>,
    face: u32,
    moment_edge: u32,
    /// The batch as the shaders read it, one region per frame in flight.
    /// `None` until [`Probes::open`] — the tests below drive the schedule and
    /// the layouts, which are the half that needs no device.
    views: Option<BufferHandle>,
    address: DeviceAddress,
    /// Staged once per frame, reused so a frame allocates nothing.
    staged: Vec<GpuProbeView>,
    /// The field itself: the L1 records and the distance tiles, both owned
    /// across frames and imported into the graph rather than pooled.
    images: Option<Images>,
}

/// The field's two images, at their **largest** extent whatever grid is fitted.
///
/// Sized once and never resized, which is the whole reason: an imported image is
/// read by frames still in flight, so recreating one on a refit would be a
/// destroy racing a read with no fence to hang it on. At [`MAX_PER_AXIS`] and
/// [`MAX_MOMENT_EDGE`] that is 128 KiB and 2 MiB, and every layout below
/// addresses from the origin — so a smaller grid uses a corner and the rest is
/// texels nothing reads.
#[derive(Clone, Copy, Debug)]
struct Images {
    sh: gg_rhi::ImageHandle,
    sh_texture: TextureIndex,
    moments: gg_rhi::ImageHandle,
    moment_texture: TextureIndex,
}

impl Probes {
    /// Fit the grid to `min..max` if the one in hand does not reach, then take
    /// the next `r.gi_rate` probes off the round robin.
    ///
    /// Returns nothing: the batch is [`Probes::batch`], and the field's own
    /// dimensions are the grid's. A frame with the field off, or with no
    /// geometry to bound, leaves the grid alone rather than clearing it — a
    /// CVar toggled twice should not cost the field it already gathered.
    pub(crate) fn update(&mut self, min: sim::DVec3, max: sim::DVec3, camera_origin: sim::DVec3) {
        self.batch.clear();
        self.face = face_size();
        self.moment_edge = moment_edge();
        let spacing = cvars::GI_SPACING.float().max(0.05);
        let fitted = Grid::fit(min, max, spacing);
        let refit = match self.grid {
            // A grid that no longer reaches is worth replacing only by a
            // *different* one. A scene larger than [`MAX_PER_AXIS`] spacings
            // never satisfies `covers` — the fit is clamped, so it does not
            // reach either — and refitting to the same grid every frame would
            // reset the field every frame and leave it one batch old forever.
            Some(grid) => {
                grid.spacing != fitted.spacing || (!grid.covers(min, max) && grid != fitted)
            }
            None => true,
        };
        if refit {
            // A refit discards every probe, so it is an event and not a detail:
            // §6 M57's flicker was sixteen of these in a sixty-frame jump and
            // nothing named one. `debug` rather than `info` because a session
            // legitimately refits while a level streams in.
            tracing::debug!(
                target: "gg::field",
                spacing = fitted.spacing,
                counts = ?fitted.counts,
                origin = ?fitted.origin,
                probes = fitted.probes(),
                was = ?self.grid.map(|g| (g.spacing, g.counts)),
                "probe field refitted"
            );
            self.ungathered = vec![true; fitted.probes()];
            self.cursor = 0;
            self.grid = Some(fitted);
        }
        let Some(grid) = self.grid else { return };
        let probes = grid.probes();
        let rate = match cvars::GI_RATE.int() {
            // Zero is "as many as one batch holds", for a harness that wants the
            // field in as few frames as it can be had and is willing to pay six
            // draws per probe for them.
            n if n <= 0 => probes,
            n => (n as usize).min(probes),
        }
        .min(MAX_BATCH);
        // Widened by the gutter exactly as a lamp face is, and for the same
        // reason: the integration reads the tile clamped to itself, so a face's
        // own edge has to sit inside the tile's.
        let tan_half = 1.0;
        let projection = render::cube_face_reverse_z(tan_half, NEAR);
        for _ in 0..rate {
            let index = self.cursor % probes;
            self.cursor = index + 1;
            let position = grid.position(index, camera_origin);
            self.batch.push(Gather {
                index,
                position,
                faces: core::array::from_fn(|face| {
                    let (forward, up) = AXES[face];
                    projection * render::camera::rh::view::look_to_mat4(position, forward, up)
                }),
            });
            if let Some(slot) = self.ungathered.get_mut(index) {
                *slot = false;
            }
        }
    }

    /// Allocate the view buffer — one [`MAX_BATCH`] region per frame in flight,
    /// so a frame writing this one cannot move the matrices the previous frame's
    /// probe pass is still reading.
    ///
    /// # Errors
    /// Whatever the allocation refuses.
    pub(crate) fn open(&mut self, rhi: &mut impl GpuHost) -> Result<(), RhiError> {
        let buffer = rhi.create_buffer(&BufferDesc {
            name: "render.probes.views",
            size: SLOT_BYTES * MAX_BATCH as u64 * FRAMES_IN_FLIGHT,
            kind: gg_rhi::BufferKind::Dynamic,
        })?;
        self.address = rhi.buffer_address(buffer)?;
        self.views = Some(buffer);
        let sh = field_image(rhi, "render.probes.sh", MAX_SH_EXTENT)?;
        let moments = field_image(rhi, "render.probes.moments", MAX_MOMENT_EXTENT)?;
        rhi.flush_uploads()?;
        self.images = Some(Images {
            sh,
            sh_texture: rhi.register_texture(sh)?,
            moments,
            moment_texture: rhi.register_texture(moments)?,
        });
        Ok(())
    }

    /// Name both field images in `frame`, as images this outlives it.
    ///
    /// `None` before [`Probes::open`], which is every renderer's first frame and
    /// no other — and then nothing downstream declares a field pass or a field
    /// sample, so the graph is M35's exactly.
    pub(crate) fn import(
        &self,
        frame: &mut crate::graph::Frame<'_>,
    ) -> Option<(crate::graph::ResourceId, crate::graph::ResourceId)> {
        let images = self.images?;
        Some((
            frame.imported("probe.sh", images.sh, images.sh_texture),
            frame.imported("probe.moments", images.moments, images.moment_texture),
        ))
    }

    /// What the frame block carries, once the field holds something and `r.gi`
    /// asks for it.
    ///
    /// **The CVar is read here and not only in [`Probes::refit`]**, and the
    /// difference is a session rather than a fresh renderer: `refit` stops the
    /// *gathering*, which is invisible to a frame that already has a grid, so
    /// switching the term off left the last field it gathered lighting every
    /// frame after. `gg-tools bounce`'s field table is what found it — it grades
    /// a ratio of the two renders, and the ratio was exactly 1 everywhere.
    /// Nothing is cleared: the grid and its records survive the toggle, so
    /// turning it back on costs the frame it costs and not the field.
    ///
    /// A probe the round robin has not reached yet is a *different* question and
    /// is answered in the field itself — `probe.slang` writes 1 into the state
    /// texel's alpha and [`Probes::open`] uploaded 0, so the shader can tell a
    /// record from the zeroes it started as. Without that an ungathered probe
    /// contributes a tiny weight and an irradiance of zero, which is not "no
    /// opinion" but black.
    ///
    /// The **view buffer's address is not in here**, and that separation is
    /// deliberate: a frame gathers whether or not it has a field to sample, so
    /// its probe vertex shaders read that address on frames when this is `None`
    /// — carrying it inside would hand them a null pointer.
    pub(crate) fn field(&self, camera_origin: sim::DVec3) -> Option<Field> {
        if !cvars::GI.bool() {
            return None;
        }
        let (grid, images) = (self.grid?, self.images?);
        Some(Field {
            sh: images.sh_texture,
            moments: images.moment_texture,
            grid,
            moment_edge: self.moment_edge,
            origin: grid.origin_relative(camera_origin),
            bias: self.bias(),
        })
    }

    /// Write this frame's batch, and hand back the address the frame block
    /// carries. `None` is a frame with nothing to gather, or one before
    /// [`Probes::open`].
    ///
    /// # Errors
    /// Whatever the buffer write refuses.
    pub(crate) fn stage(
        &mut self,
        rhi: &mut impl GpuHost,
        slot: u64,
    ) -> Result<Option<DeviceAddress>, RhiError> {
        let (Some(buffer), false) = (self.views, self.batch.is_empty()) else {
            return Ok(None);
        };
        self.staged.clear();
        self.staged
            .extend(self.batch.iter().map(|gather| GpuProbeView {
                faces: core::array::from_fn(|face| render::rows(gather.faces[face])),
                position: gather.position.to_array(),
                reserved: 0.0,
            }));
        let region = SLOT_BYTES * MAX_BATCH as u64;
        let offset = (slot % FRAMES_IN_FLIGHT) * region;
        rhi.write_buffer(buffer, offset, bytemuck::cast_slice(&self.staged))?;
        Ok(Some(self.address + offset))
    }

    /// Release the view buffer.
    ///
    /// # Errors
    /// A handle the RHI does not recognise, which cannot happen for one issued
    /// through here.
    pub(crate) fn close(&mut self, rhi: &mut impl GpuHost) -> Result<(), RhiError> {
        if let Some(images) = self.images.take() {
            rhi.destroy_image(images.sh)?;
            rhi.destroy_image(images.moments)?;
        }
        match self.views.take() {
            Some(buffer) => rhi.destroy_buffer(buffer),
            None => Ok(()),
        }
    }

    /// Fit the grid to what this frame is drawing, and take the next probes off
    /// the round robin.
    ///
    /// The bounds are the **visible** instances', which is what extract hands
    /// over: what survived the frustum, plus what `cast_shadows` swept back in.
    /// There is no whole-world bound short of a second pass over the ECS, and
    /// §1.4 puts the absolute positions that would need on the far side of the
    /// membrane. What it costs is the residual §6 M36 names rather than hides —
    /// the grid becomes the union of what the camera has looked at, because it
    /// only ever grows.
    pub(crate) fn refit(&mut self, extracted: &gg_extract::Extracted) {
        match cvars::GI.bool().then(|| bounds_of(extracted)).flatten() {
            None => self.idle(),
            Some((min, max)) => self.update(min, max, extracted.camera_origin),
        }
    }

    /// Gather nothing this frame, keeping the grid and everything in it.
    ///
    /// `r.gi` off, or a frame with no geometry to bound. Not a clear: a CVar
    /// toggled twice should not cost the field it already gathered, and the
    /// shading path's own switch is `SHADING_GI` rather than an empty field.
    pub(crate) fn idle(&mut self) {
        self.batch.clear();
    }

    pub(crate) fn grid(&self) -> Option<Grid> {
        self.grid
    }

    /// Metres a recorded distance is clamped to, and the value a face texel that
    /// drew nothing stands for — see [`DISTANCE_CAP`].
    pub(crate) fn distance_cap(&self) -> f32 {
        self.grid.map_or(0.0, |grid| DISTANCE_CAP * grid.spacing())
    }

    /// Metres a shading point is pushed along its normal before it is located in
    /// the grid.
    ///
    /// A fraction of the spacing rather than a number of metres: what the offset
    /// has to clear is the probe's own view of the surface being shaded, whose
    /// distance error is a face texel's worth of the cell it is in.
    pub(crate) fn bias(&self) -> f32 {
        self.grid.map_or(0.0, |grid| 0.15 * grid.spacing())
    }

    pub(crate) fn batch(&self) -> &[Gather] {
        &self.batch
    }

    /// Probes that have never been gathered since the grid was fitted.
    ///
    /// The number a reference render must wait out. Nonzero means the field is
    /// partly whatever the allocator handed over, and a golden blessed there
    /// would be a reference of the warm-up rather than of the feature — which is
    /// `Content::pending`'s argument (§6 M27), at one remove.
    pub(crate) fn pending(&self) -> usize {
        self.ungathered.iter().filter(|u| **u).count()
    }

    /// [`Probes::pending`] beside the count it is out of — what a harness waits
    /// on, and what a session with the field off reports as `(0, 0)`.
    pub(crate) fn progress(&self) -> (usize, usize) {
        (self.pending(), self.grid.map_or(0, |grid| grid.probes()))
    }

    /// The radiance atlas this frame's batch needs: six faces across, one row
    /// per probe being gathered.
    pub(crate) fn atlas_extent(&self) -> (u32, u32) {
        (
            self.face * FACES as u32,
            self.face * self.batch.len().max(1) as u32,
        )
    }

    /// Where probe `slot` of the batch draws face `face`, in atlas pixels.
    pub(crate) fn tile_at(&self, slot: usize, face: usize) -> gg_rhi::Viewport {
        gg_rhi::Viewport {
            x: face as u32 * self.face,
            y: slot as u32 * self.face,
            width: self.face,
            height: self.face,
        }
    }

    pub(crate) fn face(&self) -> u32 {
        self.face
    }

    pub(crate) fn moment_edge(&self) -> u32 {
        self.moment_edge
    }
}

/// The two passes that turn a gathered atlas into the field (§6 M36).
///
/// A struct of its own rather than two more pipelines on `BoxPass`, because
/// nothing here draws geometry: both are one fullscreen triangle per probe over
/// a rectangle the push names, and the push blocks they borrow have to outlive
/// the [`DrawSpec`]s built from them.
pub(crate) struct Integrate {
    sh: PipelineHandle,
    moments: PipelineHandle,
    /// One push per gathering probe, per target. Two lists because the only
    /// field that differs is the viewport corner, and a `DrawSpec` borrows the
    /// bytes rather than copying them.
    sh_pushes: Vec<probe_shader::ProbePush>,
    moment_pushes: Vec<probe_shader::ProbePush>,
}

impl Integrate {
    /// # Errors
    /// Whatever pipeline creation refuses.
    pub(crate) fn new(rhi: &mut impl GpuHost) -> Result<Self, RhiError> {
        Ok(Integrate {
            sh: rhi.create_pipeline(&record_desc(
                "probe.sh",
                probe_shader::FS_SH_SPIRV,
                probe_shader::FS_SH_ENTRY,
            ))?,
            moments: rhi.create_pipeline(&record_desc(
                "probe.moments",
                probe_shader::FS_MOMENTS_SPIRV,
                probe_shader::FS_MOMENTS_ENTRY,
            ))?,
            sh_pushes: Vec::new(),
            moment_pushes: Vec::new(),
        })
    }

    /// This frame's pushes — one per probe in the batch, per record.
    pub(crate) fn build(&mut self, probes: &Probes, frame: DeviceAddress, atlas: TextureIndex) {
        self.sh_pushes.clear();
        self.moment_pushes.clear();
        let Some(grid) = probes.grid() else { return };
        // The one place the grid's own extents are checked against the images
        // they are written into: `MAX_PER_AXIS` and `MAX_MOMENT_EDGE` are what
        // make a corner of a fixed-size image enough, and a grid that outgrew
        // either would write outside its own rows in silence.
        debug_assert!(
            grid.sh_extent().0 <= MAX_SH_EXTENT.0 && grid.sh_extent().1 <= MAX_SH_EXTENT.1,
            "the field outgrew its image: {:?}",
            grid.sh_extent()
        );
        debug_assert!(
            grid.moment_extent(probes.moment_edge()).0 <= MAX_MOMENT_EXTENT.0
                && grid.moment_extent(probes.moment_edge()).1 <= MAX_MOMENT_EXTENT.1,
            "the distance tiles outgrew their image: {:?}",
            grid.moment_extent(probes.moment_edge())
        );
        let cap = probes.distance_cap();
        for (slot, gather) in probes.batch().iter().enumerate() {
            let (x, y) = grid.sh_texel(gather.index);
            self.sh_pushes.push(probe_shader::ProbePush::new(
                frame,
                atlas.get(),
                slot as u32,
                probes.face(),
                probes.moment_edge(),
                [x, y],
                cap,
            ));
            let tile = grid.moment_tile(gather.index, probes.moment_edge());
            self.moment_pushes.push(probe_shader::ProbePush::new(
                frame,
                atlas.get(),
                slot as u32,
                probes.face(),
                probes.moment_edge(),
                [tile.x, tile.y],
                cap,
            ));
        }
    }

    /// The irradiance record's draws — one per probe, over its own four texels.
    pub(crate) fn sh_draws(&self, probes: &Probes) -> Vec<DrawSpec<'_>> {
        let Some(grid) = probes.grid() else {
            return Vec::new();
        };
        self.draws(self.sh, &self.sh_pushes, probes, |index| {
            let (x, y) = grid.sh_texel(index);
            gg_rhi::Viewport {
                x,
                y,
                width: SH_TEXELS,
                height: 1,
            }
        })
    }

    /// The distance record's — one per probe, over its own octahedral tile.
    pub(crate) fn moment_draws(&self, probes: &Probes) -> Vec<DrawSpec<'_>> {
        let Some(grid) = probes.grid() else {
            return Vec::new();
        };
        let edge = probes.moment_edge();
        self.draws(self.moments, &self.moment_pushes, probes, |index| {
            grid.moment_tile(index, edge)
        })
    }

    fn draws<'a>(
        &self,
        pipeline: PipelineHandle,
        pushes: &'a [probe_shader::ProbePush],
        probes: &Probes,
        rect: impl Fn(usize) -> gg_rhi::Viewport,
    ) -> Vec<DrawSpec<'a>> {
        probes
            .batch()
            .iter()
            .zip(pushes)
            .map(|(gather, push)| DrawSpec {
                pipeline,
                push_constants: bytemuck::bytes_of(push),
                count: crate::FULLSCREEN_VERTICES,
                index_buffer: None,
                indirect: None,
                depth_bias: None,
                viewport: Some(rect(gather.index)),
            })
            .collect()
    }

    /// # Errors
    /// A handle the RHI does not recognise, which cannot happen for one issued
    /// through here.
    pub(crate) fn destroy(self, rhi: &mut impl GpuHost) -> Result<(), RhiError> {
        rhi.destroy_pipeline(self.sh)?;
        rhi.destroy_pipeline(self.moments)
    }
}

/// One record's pipeline: the fullscreen triangle, no depth, into an `Rgba16F`
/// rectangle the push names.
fn record_desc(
    name: &'static str,
    fs: &'static [u8],
    entry: &'static str,
) -> PipelineDesc<'static> {
    PipelineDesc {
        name,
        vs_spirv: probe_shader::VS_MAIN_SPIRV,
        vs_entry: probe_shader::VS_MAIN_ENTRY,
        fs_spirv: fs,
        fs_entry: entry,
        push_constant_size: core::mem::size_of::<probe_shader::ProbePush>() as u32,
        color: ColorTarget::Format(FIELD_FORMAT),
        blend: Blend::Off,
        depth: DepthMode::Off,
        samples: Samples::X1,
        depth_bias: false,
    }
}

/// The absolute box this frame's geometry occupies, widened back through the
/// camera origin the instances were narrowed by.
///
/// A **box** per instance and not the bounding sphere the culler carries, which
/// is the difference between a field over the room and a field over a corner of
/// the sky above it. Demo 12's floor is 25 m across and half a metre thick, so
/// its radius is 17.7: bounded as a sphere it puts the grid's ceiling eighteen
/// metres over a room four metres tall, and with the per-axis clamp the eight
/// probes an axis land in empty space. `gg-tools bounce`'s field table is what
/// found it — every spacing from 0.75 m to 2 m graded *identically*, because at
/// none of them did the grid reach the geometry at all.
///
/// A conservative bound is the right thing for a culler, which must not drop
/// what it cannot see; it is the wrong thing for a grid, which spends a fixed
/// number of probes on whatever it is told to cover. Models keep the radius:
/// their `half_extent` is a scale rather than an extent (`gg_extract::Instance`),
/// so a sphere is the only bound this has.
///
/// **Every visible instance counts, including one that is there for three
/// ticks** (§6 M37 item 3, `tests/transient.rs`). Demo 12's room is closed on
/// five sides and open to the sky, so a shot aimed up draws a 40 m tracer
/// through a ceiling that is not there — and that sliver widens the spacing from
/// 4 m to 8 m, refits the grid, and throws the gathered field away. Twice per
/// shot: once when it appears and once when it dies, `covers` being no help
/// because the fitted *spacing* differs either way.
///
/// P2: bound the grid by something a three-tick sliver cannot dominate. The
/// candidates all have a cost and none is obviously right — a volume floor
/// discards a small distant receiver, growth hysteresis makes the grid a
/// function of the frame the way this doc says it is not, and a receiver flag
/// widens the render protocol for the renderer's convenience rather than
/// because a game asked (§2). Named rather than settled by argument, and the
/// test pins today's numbers so the day one is chosen the diff says what moved.
fn bounds_of(extracted: &gg_extract::Extracted) -> Option<(sim::DVec3, sim::DVec3)> {
    let mut bounds: Option<(render::Vec3, render::Vec3)> = None;
    let boxes = extracted.visible().iter().map(|instance| {
        // The rotated box's own axis-aligned reach: the absolute rotation matrix
        // times the half-extent, which is the standard result and exact rather
        // than conservative for the axis-aligned case every `Renderable` is.
        let m = render::Mat3::from_quat(instance.rotation);
        let absolute = render::Mat3::from_cols(m.x_axis.abs(), m.y_axis.abs(), m.z_axis.abs());
        (instance.offset, absolute * instance.half_extent)
    });
    let models = extracted
        .visible_models()
        .iter()
        .map(|instance| (instance.offset, render::Vec3::splat(instance.radius)));
    for (offset, reach) in boxes.chain(models) {
        let (lo, hi) = (offset - reach, offset + reach);
        bounds = Some(match bounds {
            Some((min, max)) => (min.min(lo), max.max(hi)),
            None => (lo, hi),
        });
    }
    let (min, max) = bounds?;
    // A scene bounded by a non-finite number is a scene with nothing to say
    // about where probes go; the field keeps the grid it had.
    if !(min.is_finite() && max.is_finite()) {
        return None;
    }
    // Widening is not narrowing and needs no membrane: what §1.4 forbids is an
    // absolute position becoming an `f32`, and this is the inverse — the camera
    // origin is `f64` and the offset was already narrow.
    let absolute = |v: render::Vec3| {
        extracted.camera_origin + sim::DVec3::new(f64::from(v.x), f64::from(v.y), f64::from(v.z))
    };
    Some((absolute(min), absolute(max)))
}

/// The irradiance image at its largest — see [`Images`].
const MAX_SH_EXTENT: (u32, u32) = (MAX_PER_AXIS * SH_TEXELS, MAX_PER_AXIS * MAX_PER_AXIS);
/// The distance image at its largest, and the pair [`MAX_PER_AXIS`] is set by.
const MAX_MOMENT_EXTENT: (u32, u32) = (
    MAX_PER_AXIS * MAX_MOMENT_EDGE,
    MAX_PER_AXIS * MAX_PER_AXIS * MAX_MOMENT_EDGE,
);

/// One field image: a colour target a later pass samples, which is exactly
/// [`gg_rhi::ImageUse::ColorTarget`] — **uploaded as zeroes**, which is what
/// puts it in the layout its bindless descriptor declares before any frame runs
/// (`graph::Frame::imported` argues why that has to happen at creation) and what
/// makes an ungathered probe's record readable rather than undefined.
///
/// Not flushed here: the caller uploads both and flushes once, which is the same
/// "one flush at startup" every other pass observes (§6 M9).
fn field_image(
    rhi: &mut impl GpuHost,
    name: &'static str,
    extent: (u32, u32),
) -> Result<gg_rhi::ImageHandle, RhiError> {
    let image = rhi.create_image(&gg_rhi::ImageDesc {
        name,
        extent,
        format: FIELD_FORMAT,
        usage: gg_rhi::ImageUse::ColorTarget,
        mip_levels: 1,
        samples: Samples::X1,
    })?;
    let bytes = extent.0 as usize * extent.1 as usize * FIELD_TEXEL_BYTES;
    rhi.upload_image(image, 0, &vec![0; bytes])?;
    Ok(image)
}

/// The face edge a session asked for, clamped to what an atlas can be. Powers of
/// two so the integration's stride is a shift and the tile boundary is exact.
fn face_size() -> u32 {
    (cvars::GI_FACE.int().clamp(4, 32) as u32).next_power_of_two()
}

/// The octahedral edge of a distance tile — see [`MAX_PER_AXIS`] for why this
/// ceiling and the probe count's are the same constraint.
fn moment_edge() -> u32 {
    (cvars::GI_MOMENTS.int().clamp(2, MAX_MOMENT_EDGE as i64) as u32).next_power_of_two()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(x: f64, y: f64, z: f64) -> sim::DVec3 {
        sim::DVec3::new(x, y, z)
    }

    /// A field survives a scene that moves. The snap alone cannot promise this —
    /// it has boundaries wherever the multiples are, and bounds straddling one
    /// fit two different grids — so what promises it is that [`Probes::update`]
    /// refits only when [`Grid::covers`] says no.
    ///
    /// The cost of that rule, and it is deliberate: a grid only ever grows. A
    /// level that shrinks keeps probes over the space it used to occupy, which
    /// is a field slightly too big rather than a field thrown away, and the
    /// second is much the worse of the two.
    #[test]
    fn a_scene_that_moves_inside_its_grid_keeps_the_field_it_gathered() {
        let mut probes = Probes::default();
        cvars::GI_RATE.set_int(0);
        cvars::GI_SPACING.set_float(2.0);
        probes.update(at(-4.9, -0.1, -4.9), at(4.9, 3.1, 4.9), sim::DVec3::ZERO);
        let grid = probes.grid();
        // Each of these fits a *different* grid on its own — the third straddles
        // the snap boundary at y = 0 — and none of them costs the field.
        for (dx, dy) in [(0.3, -0.2), (-0.45, 0.6), (0.9, 0.9), (0.0, 2.4)] {
            probes.update(
                at(-4.9 + dx, -0.1 + dy, -4.9 + dx),
                at(4.9 - dx, 3.1, 4.9 - dx),
                sim::DVec3::ZERO,
            );
            assert_eq!(probes.grid(), grid, "moved by ({dx}, {dy})");
        }
        assert_ne!(
            Grid::fit(at(-4.9, 0.5, -4.9), at(4.9, 3.1, 4.9), 2.0),
            grid.unwrap_or_else(|| Grid::fit(at(0.0, 0.0, 0.0), at(0.0, 0.0, 0.0), 2.0)),
            "the wobble crossed a snap boundary, so `covers` is what held the grid"
        );
    }

    /// A bound sitting exactly on a spacing multiple fits *one* grid, however the
    /// membrane rounds it (§6 M57).
    ///
    /// Demo 12's own numbers, because they are what found this: a 25 m room whose
    /// walls top out at exactly `y = 4.0` against the 4 m spacing `r.gi_spacing`
    /// widens to. The bounds reach the renderer as `f32` offsets against the
    /// camera origin (§1.4) and are widened back, so `max.y` arrives a ULP either
    /// side of 4.0 depending on where the eye is — and the eye's *height* is what
    /// moves it, which is why a player saw this while jumping and not while
    /// walking. Without [`SLACK`] the vertical count alternates 3, 4, 3, 4 and
    /// [`Probes::update`] discards the whole field on every flip.
    ///
    /// `covers` cannot be what saves this: a 25 m room does not fit eight probes
    /// at 4 m from a snapped origin, so it is false every frame here and the
    /// refit test is `grid != fitted` alone. The fit *is* the hysteresis.
    #[test]
    fn a_bound_on_a_spacing_multiple_fits_one_grid_however_the_membrane_rounds_it() {
        let mut probes = Probes::default();
        cvars::GI_RATE.set_int(0);
        cvars::GI_SPACING.set_float(2.0);
        // Well past an `f32` ULP at these magnitudes (~1e-6 m) and far under the
        // 4 mm [`SLACK`] is worth, which is the band the claim is about.
        let wobbles = [0.0, 1e-7, -1e-7, 4e-7, -4e-7, 2e-6, -2e-6];
        let fitted: Vec<Grid> = wobbles
            .iter()
            .map(|w| {
                Grid::fit(
                    at(-12.5 + w, -0.5 - w, -12.5 - w),
                    at(12.5, 4.0 + w, 12.5),
                    2.0,
                )
            })
            .collect();
        assert!(
            fitted.windows(2).all(|p| p[0] == p[1]),
            "the fit moved under a rounding: {fitted:?}"
        );
        assert_eq!(fitted[0].spacing(), 4.0, "the room is wider than 8 x 2 m");
        assert!(
            !fitted[0].covers(at(-12.5, -0.5, -12.5), at(12.5, 4.0, 12.5)),
            "if the clamped grid ever covers this room, the test below proves nothing"
        );
        // And the field survives the whole wobble, which is the observable half:
        // one grid, and `pending` never climbing back off a refit.
        let (min, max) = (at(-12.5, -0.5, -12.5), at(12.5, 4.0, 12.5));
        probes.update(min, max, sim::DVec3::ZERO);
        let grid = probes.grid();
        // `r.gi_rate 0` is one *batch* and not one grid ([`MAX_BATCH`]), so a
        // 192-probe field takes three passes however fast it is asked for.
        for _ in 0..grid.map_or(0, |g| g.probes().div_ceil(MAX_BATCH)) {
            probes.update(min, max, sim::DVec3::ZERO);
        }
        assert_eq!(probes.pending(), 0, "the field never converged");
        for w in wobbles {
            probes.update(
                at(-12.5 + w, -0.5 - w, -12.5 - w),
                at(12.5, 4.0 + w, 12.5),
                sim::DVec3::ZERO,
            );
            assert_eq!(probes.grid(), grid, "refit on a wobble of {w}");
            assert_eq!(probes.pending(), 0, "the field was thrown away at {w}");
        }
    }

    /// The grid brackets what it was fitted to — every corner of the bounds is
    /// inside the probe hull, or a shading point there interpolates against
    /// probes on one side of it only.
    #[test]
    fn every_corner_of_the_bounds_is_inside_the_probe_hull() {
        let (min, max) = (at(-5.3, -0.2, -4.1), at(4.4, 3.9, 3.2));
        let grid = Grid::fit(min, max, 1.5);
        assert!(
            grid.covers(min, max),
            "the fit does not reach its own bounds"
        );
        let far = grid.corner();
        assert!(
            far.x - max.x < 1.5 && far.y - max.y < 1.5 && far.z - max.z < 1.5,
            "and does not overshoot by a whole spacing: {far:?} against {max:?}"
        );
        // And it keeps bracketing them when the scene is wider than
        // [`MAX_PER_AXIS`] spacings, which is what `r.gi_spacing` being a floor
        // buys: the spacing widens to a whole multiple of the asked-for one
        // rather than the grid retreating into a corner of the scene.
        let wide = at(60.0, 3.9, 3.2);
        let coarse = Grid::fit(min, wide, 1.5);
        assert_eq!(coarse.counts()[0], MAX_PER_AXIS);
        assert!(
            coarse.covers(min, wide),
            "the widened fit still does not reach"
        );
        let multiple = f64::from(coarse.spacing()) / 1.5;
        assert!(
            (multiple - multiple.round()).abs() < 1e-9 && multiple >= 1.0,
            "the spacing is a whole multiple of the session's: {multiple}"
        );
        // A scene that already fits keeps the spacing it asked for — the
        // widening is a consequence of the clamp and not a policy of its own.
        assert_eq!(grid.spacing(), 1.5);
    }

    /// Index → coordinate → texel is one ordering, used three times. A field
    /// laid out in one order and read in another is a bug with no symptom
    /// except that the light is in the wrong place.
    #[test]
    fn the_layout_gives_every_probe_its_own_texels() {
        let grid = Grid::fit(at(0.0, 0.0, 0.0), at(6.0, 4.0, 8.0), 2.0);
        assert_eq!(grid.counts(), [4, 3, 5]);
        assert_eq!(grid.probes(), 60);
        let (w, h) = grid.sh_extent();
        assert_eq!((w, h), (4 * SH_TEXELS, 15));

        let mut seen = vec![false; (w * h) as usize];
        let edge = 4;
        let mut tiles =
            vec![false; (grid.moment_extent(edge).0 * grid.moment_extent(edge).1) as usize];
        for index in 0..grid.probes() {
            let (tx, ty) = grid.sh_texel(index);
            for c in 0..SH_TEXELS {
                let slot = (ty * w + tx + c) as usize;
                assert!(!seen[slot], "probe {index} shares texel {slot}");
                seen[slot] = true;
            }
            let tile = grid.moment_tile(index, edge);
            let (mw, _) = grid.moment_extent(edge);
            for v in 0..edge {
                for u in 0..edge {
                    let slot = ((tile.y + v) * mw + tile.x + u) as usize;
                    assert!(!tiles[slot], "probe {index} shares moment texel {slot}");
                    tiles[slot] = true;
                }
            }
        }
        assert!(seen.iter().all(|s| *s), "the field has texels nothing owns");
        assert!(tiles.iter().all(|s| *s));
    }

    /// A probe's position is its coordinate times the spacing, off the grid
    /// origin — and it narrows through the camera origin and nowhere else, so a
    /// far-away eye does not quantize the field it is standing in.
    #[test]
    fn a_probe_sits_where_its_coordinate_says_however_far_the_eye_is() {
        let grid = Grid::fit(at(0.0, 0.0, 0.0), at(6.0, 4.0, 8.0), 2.0);
        let index = 1 + 4 * (1 + 3 * 2); // coord (1, 1, 2)
        assert_eq!(grid.coord(index), [1, 1, 2]);
        for eye in [at(0.0, 0.0, 0.0), at(1.0e6, 0.0, -1.0e6)] {
            let expected = render::Vec3::new(
                (2.0 - eye.x) as f32,
                (2.0 - eye.y) as f32,
                (4.0 - eye.z) as f32,
            );
            let got = grid.position(index, eye);
            assert!(
                (got - expected).length() < 1e-3,
                "at eye {eye:?}: {got:?} against {expected:?}"
            );
            // The origin the shader locates a point against is the same frame.
            let origin = grid.origin_relative(eye);
            assert!((origin + render::Vec3::new(2.0, 2.0, 4.0) - got).length() < 1e-3);
        }
    }

    /// The round robin reaches every probe and then comes back around, which is
    /// what makes `pending` a countdown rather than a number that gets stuck.
    #[test]
    fn the_round_robin_gathers_every_probe_before_it_repeats() {
        let mut probes = Probes::default();
        let (min, max) = (at(0.0, 0.0, 0.0), at(4.0, 2.0, 4.0));
        cvars::GI_RATE.set_int(7);
        cvars::GI_SPACING.set_float(2.0);
        probes.update(min, max, sim::DVec3::ZERO);
        let total = probes.grid().map_or(0, |g| g.probes());
        assert_eq!(total, 3 * 2 * 3);
        assert_eq!(probes.pending(), total - 7);
        let mut seen = vec![0u32; total];
        for gather in probes.batch() {
            seen[gather.index] += 1;
        }
        // Round up: the field is whole after ceil(total / rate) frames.
        for _ in 1..total.div_ceil(7) {
            probes.update(min, max, sim::DVec3::ZERO);
            for gather in probes.batch() {
                seen[gather.index] += 1;
            }
        }
        assert_eq!(probes.pending(), 0, "a frame short of the whole field");
        assert!(
            seen.iter().all(|n| *n >= 1),
            "the robin skipped a probe: {seen:?}"
        );
        cvars::GI_RATE.set_int(0);
        probes.update(min, max, sim::DVec3::ZERO);
        assert_eq!(probes.batch().len(), total, "zero is all of them");
    }

    /// A refit is a loss and says so — the field goes back to empty rather than
    /// keeping records addressed against a grid that no longer exists.
    #[test]
    fn a_grid_that_had_to_move_reports_a_field_with_nothing_in_it() {
        let mut probes = Probes::default();
        cvars::GI_RATE.set_int(0);
        cvars::GI_SPACING.set_float(2.0);
        probes.update(at(0.0, 0.0, 0.0), at(4.0, 2.0, 4.0), sim::DVec3::ZERO);
        assert_eq!(probes.pending(), 0);
        let before = probes.grid();

        // Inside the grid: nothing moves, nothing is lost.
        probes.update(at(1.0, 0.5, 1.0), at(3.0, 1.5, 3.0), sim::DVec3::ZERO);
        assert_eq!(probes.grid(), before);

        // Outside it: a new grid, and every probe ungathered again.
        cvars::GI_RATE.set_int(1);
        probes.update(at(0.0, 0.0, 0.0), at(40.0, 2.0, 4.0), sim::DVec3::ZERO);
        assert_ne!(probes.grid(), before);
        assert_eq!(
            probes.pending(),
            probes.grid().map_or(0, |g| g.probes()) - 1,
            "a refit that kept its `pending` would let a golden bless a stale field"
        );
        cvars::GI_RATE.set_int(0);
    }

    /// The one place `pbr.slang`'s `PROBE_FORWARD`/`PROBE_UP` table is checked
    /// against the matrices the host actually built.
    ///
    /// The shader has to turn a face texel back into the direction that drew it
    /// — the integration is a projection onto directions — and it does that from
    /// a transcription of [`AXES`] rather than from the matrix, which is one
    /// convention with two ends and no runtime meeting point. This is the
    /// meeting point: the direction the shader's formula returns, projected
    /// through the matrix `Probes::update` built, must land back on the texel it
    /// came from. A sign error anywhere in either half fails here rather than
    /// mirroring the field.
    #[test]
    fn the_shader_reads_a_face_texel_as_the_direction_the_matrix_drew_it() {
        let mut probes = Probes::default();
        cvars::GI_RATE.set_int(0);
        cvars::GI_SPACING.set_float(2.0);
        probes.update(at(0.0, 0.0, 0.0), at(2.0, 2.0, 2.0), sim::DVec3::ZERO);
        let gather = probes.batch()[0];
        for (face, (forward, up)) in AXES.iter().copied().enumerate() {
            // `probe_face_direction`, transcribed: right is `forward x up` and
            // the v term is *subtracted* because Vulkan's clip space is y-down.
            let right = forward.cross(up);
            let above = right.cross(forward);
            for (u, v) in [(-0.7, -0.3), (0.4, 0.9), (0.0, 0.0), (0.85, -0.85)] {
                let direction = (forward + right * u - above * v).normalize();
                let clip =
                    gather.faces[face] * render::Vec4::from((gather.position + direction, 1.0));
                assert!(clip.w > 0.0, "face {face} does not contain ({u}, {v})");
                let (x, y) = (clip.x / clip.w, clip.y / clip.w);
                assert!(
                    (x - u).abs() < 1e-5 && (y - v).abs() < 1e-5,
                    "face {face}: ({u}, {v}) came back as ({x}, {y})"
                );
            }
        }
    }

    /// Every probe's six faces cover the sphere: a direction out of the probe
    /// lands in exactly one face's pyramid, so the integration sees each
    /// direction once and none twice.
    #[test]
    fn the_six_faces_of_a_probe_partition_the_directions_around_it() {
        let mut probes = Probes::default();
        cvars::GI_RATE.set_int(0);
        cvars::GI_SPACING.set_float(2.0);
        probes.update(at(0.0, 0.0, 0.0), at(2.0, 2.0, 2.0), sim::DVec3::ZERO);
        let gather = probes.batch()[0];
        for i in 0..64 {
            // A spread of directions with no axis alignment, so nothing lands
            // exactly on a seam where two faces may both claim it.
            let d = render::Vec3::new(
                (i % 7) as f32 - 3.3,
                ((i / 7) % 5) as f32 - 2.2,
                ((i / 35) % 3) as f32 - 1.1,
            )
            .normalize();
            let inside = (0..FACES)
                .filter(|face| {
                    let clip = gather.faces[*face] * render::Vec4::from((gather.position + d, 1.0));
                    clip.w > 0.0
                        && clip.x.abs() <= clip.w
                        && clip.y.abs() <= clip.w
                        && clip.z >= 0.0
                })
                .count();
            assert_eq!(inside, 1, "direction {d:?} lands in {inside} faces");
        }
    }
}
