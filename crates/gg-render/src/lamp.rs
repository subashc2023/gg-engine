//! The lamps that cast, and the atlas their faces land in (§6 M31).
//!
//! §6 M15.3 gave the sun cascades and said "point lights still cast nothing";
//! §6 M30 made two hundred and fifty-six of them affordable and said the same
//! thing louder, because a lamp that lights through the wall beside it is a
//! defect that now scales. This is that pass.
//!
//! # Six 2D maps, not a cube map
//!
//! `gg-rhi` has no array layers and no cube views (`ImageDesc` is one layer, one
//! face), and adding them would buy this pass nothing: the geometry has to be
//! recorded per face either way, which is `crate::shadow_maps`' argument for
//! four separate cascade images arriving at the opposite answer here only
//! because *six times four* images would be twenty-four passes and twenty-four
//! bindless slots. So the faces are tiles of one depth atlas, six across and one
//! row per lamp, and the tile is selected by
//! [`DrawSpec::viewport`](gg_rhi::DrawSpec::viewport) — the one piece of RHI
//! surface this milestone added, and the smallest thing that could have said it.
//!
//! # Which lamps cast, and why nothing here ranks them
//!
//! The nearest [`cvars::LAMPS`] of them, which is not a ranking this module
//! invents: [`Extracted::append_lights`](gg_extract::Extracted::append_lights)
//! already sorts point lights by distance and `MAX_POINT` already truncates on
//! exactly that order. A second score — brightness, screen coverage, solid angle
//! — would be a second opinion about importance that disagrees with the
//! truncation, so that the light a frame *dropped* and the light it *shadowed*
//! were ranked by different questions. The casters are therefore the array's own
//! prefix, and the shader needs no search: an index below `directional_count +
//! lamp_count` casts, which is [`GpuFrame`](crate::lighting)'s directional
//! prefix trick at one remove.
//!
//! What that costs is named in §6 M31 rather than hidden: a big lamp just behind
//! a small one loses to it, and two lamps swapping distance order swap shadows
//! in one frame.
//!
//! # The gutter
//!
//! Each face is rendered slightly wider than 90° so that its own edge sits
//! [`GUTTER`] texels inside the tile's. A filter clamped to the tile then reads
//! depth this face rendered rather than the neighbouring lamp's, and the clamp
//! costs nothing at the seam instead of costing a wrong answer there. Faces
//! still do not filter *across* each other — a kernel straddling an edge sees
//! one side of it — which is the honest limit of an atlas and is what the
//! gutter's width is traded against.

use gg_extract::ExtractedLight;
use gg_math::render;

use crate::cvars;

/// Faces a lamp has. Six, and it is not a knob: fewer would leave a direction
/// with no map, and the number is baked into the atlas's width and the shader's
/// major-axis pick.
pub(crate) const FACES: usize = 6;

/// Lamps the frame block has room for. The *budget* is [`cvars::LAMPS`] and is
/// usually lower; this is the ceiling that sizes [`GpuLamp`](crate::lighting)'s
/// array, at 384 bytes of matrices each.
///
/// Eight rather than four because the number a session should run is a
/// measurement (`gg-tools lamps`) and a ceiling that equalled the default would
/// leave nothing to measure — §6 M30's `MAX_POINT` lesson, applied on the way in.
pub(crate) const MAX_LAMPS: usize = 8;

/// Texels of overshoot on every side of every face. Two: enough for the filter
/// radius the lamp lookup actually uses, and cheap — the gutter is taken *out*
/// of the face's angular coverage, so it is resolution given up.
const GUTTER: f32 = 2.0;

/// Metres the near plane sits at. Small enough to be inside anything a lamp is
/// mounted on, large enough that reverse-Z's precision is spent on the range the
/// lamp actually reaches rather than on the first millimetre.
const NEAR: f32 = 0.05;

/// The six faces as (forward, up) in world axes, in the order the shader picks
/// them: `2 * major_axis + (component < 0)`.
///
/// Our own table rather than Vulkan's cube-face convention, because nothing here
/// is a cube map — these are six ordinary 2D depth maps, and what matters is
/// that this table and `pbr.slang`'s `lamp_face` are the *same* table. A
/// standard one would be a second thing to be right about.
const AXES: [(render::Vec3, render::Vec3); FACES] = [
    (render::Vec3::X, render::Vec3::Y),
    (render::Vec3::NEG_X, render::Vec3::Y),
    // Y needs a different up, or the look-to basis is degenerate.
    (render::Vec3::Y, render::Vec3::Z),
    (render::Vec3::NEG_Y, render::Vec3::NEG_Z),
    (render::Vec3::Z, render::Vec3::Y),
    (render::Vec3::NEG_Z, render::Vec3::Y),
];

/// One casting lamp: where it is, how far it reaches, and the six projections
/// its faces were rendered through.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Lamp {
    /// Camera-relative, matching the light record the shader reads beside this.
    pub(crate) position: render::Vec3,
    /// The light's own range — the sphere the caster cull tests against, and
    /// the distance past which its falloff is exactly zero.
    pub(crate) range: f32,
    /// Camera-relative world → this face's clip space, in [`AXES`] order.
    pub(crate) faces: [render::Mat4; FACES],
    /// The tangent the faces were built with — 45° widened by [`GUTTER`].
    /// Carried rather than recovered from a matrix: `faces` is projection times
    /// view, so the column that held `1/tan_half` before the multiply does not
    /// afterwards.
    tan_half: f32,
}

impl Lamp {
    /// Whether a caster of `radius` at `offset` can put anything into face
    /// `face`'s map.
    ///
    /// Two tests, cheapest first: the lamp's own sphere, then the face's pyramid
    /// as the four side planes through the light. The far plane is deliberately
    /// absent — the projection has none — and the near plane is too, because a
    /// caster straddling the light is inside the sphere and must be drawn by
    /// whichever faces its parts fall into rather than rejected by a plane that
    /// runs through the middle of it.
    pub(crate) fn casts_into(&self, offset: render::Vec3, radius: f32, face: usize) -> bool {
        let to = offset - self.position;
        let reach = self.range + radius;
        if to.length_squared() > reach * reach {
            return false;
        }
        let (forward, up) = AXES[face];
        let right = forward.cross(up);
        // A side plane of a pyramid with half-angle tangent `t` has unnormalized
        // outward normal `axis - t * forward`, whose length is `sqrt(1 + t*t)` —
        // so that is what the radius is measured in. Not `sqrt(2)`: the gutter
        // makes `t` a little over 1, and rounding it down here would cull a
        // caster the face is about to render.
        let t = self.tan_half;
        let scale = radius * (1.0 + t * t).sqrt();
        let along = to.dot(forward);
        let inside = |axis: render::Vec3| to.dot(axis).abs() - along * t <= scale;
        inside(right) && inside(up)
    }
}

/// This frame's casting lamps, and the atlas that holds their faces.
#[derive(Clone, Debug, Default)]
pub(crate) struct Lamps {
    lamps: Vec<Lamp>,
    tile: u32,
}

impl Lamps {
    /// The nearest lights that fit the budget, fitted with six faces each.
    ///
    /// `lights` is extract's array — directional first, then point lights
    /// nearest-first — so this takes a prefix of the point half and the shader's
    /// "index below `directional_count + lamp_count` casts" is true by
    /// construction. The suns are counted here rather than passed in: one place
    /// deciding where the point half starts is one place to be wrong.
    pub(crate) fn build(lights: &[ExtractedLight]) -> Lamps {
        let directional = lights
            .iter()
            .take_while(|light| light.kind == gg_extract::light::DIRECTIONAL)
            .count();
        let budget = (cvars::LAMPS.int().max(0) as usize).min(MAX_LAMPS);
        let tile = tile_size();
        if budget == 0 || !cvars::LAMP_SHADOWS.bool() {
            return Lamps {
                lamps: Vec::new(),
                tile,
            };
        }
        // The gutter, as the tangent of the half-angle it widens 45° to. Exact:
        // a ratio of two integers, and `cube_face_reverse_z` takes it unrounded.
        let tan_half = tile as f32 / (tile as f32 - 2.0 * GUTTER);
        let projection = render::cube_face_reverse_z(tan_half, NEAR);
        let lamps = lights
            .iter()
            .skip(directional)
            .take(budget)
            // A range of zero reaches nothing, so six maps of it would be six
            // passes over an empty sphere.
            .filter(|light| light.range > 0.0)
            .map(|light| Lamp {
                position: light.offset,
                range: light.range,
                faces: core::array::from_fn(|face| {
                    let (forward, up) = AXES[face];
                    projection * render::camera::rh::view::look_to_mat4(light.offset, forward, up)
                }),
                tan_half,
            })
            .collect();
        Lamps { lamps, tile }
    }

    pub(crate) fn lamps(&self) -> &[Lamp] {
        &self.lamps
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.lamps.is_empty()
    }

    /// Tile edge in texels.
    pub(crate) fn tile(&self) -> u32 {
        self.tile
    }

    /// The atlas's extent: six faces across, one row per lamp.
    ///
    /// Sized to the live count rather than to the budget, so a scene with one
    /// lamp allocates one row. The transient pool keys images by size, so the
    /// count changing costs an allocation — which it does about as often as a
    /// scene gains its second lamp.
    pub(crate) fn extent(&self) -> (u32, u32) {
        (
            self.tile * FACES as u32,
            self.tile * self.lamps.len() as u32,
        )
    }

    /// Where lamp `lamp`'s face `face` is drawn, in atlas pixels.
    pub(crate) fn tile_at(&self, lamp: usize, face: usize) -> gg_rhi::Viewport {
        gg_rhi::Viewport {
            x: face as u32 * self.tile,
            y: lamp as u32 * self.tile,
            width: self.tile,
            height: self.tile,
        }
    }

    /// What the shader multiplies a tile-space uv by to reach the atlas: one
    /// sixth across, one row down.
    pub(crate) fn uv_scale(&self) -> [f32; 2] {
        [1.0 / FACES as f32, 1.0 / self.lamps.len().max(1) as f32]
    }
}

/// The tile edge a session asked for, clamped to what an atlas can be.
///
/// The upper bound is the width, not the height: six tiles across at 512 is
/// 3072, and `maxImageDimension2D` is only guaranteed to be 4096 (§4.3 requires
/// no more). A session wanting a sharper lamp raises the count of texels it can
/// afford elsewhere first.
fn tile_size() -> u32 {
    (cvars::LAMP_SIZE.int().clamp(128, 512) as u32).next_power_of_two()
}
