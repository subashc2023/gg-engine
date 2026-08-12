//! What a game tells the host to draw — the render half of the §4.2.2 boundary.
//!
//! The host holds no game types. It cannot instantiate a query over `Cube`, and
//! §3's deny pin means a type both sides name can live in exactly three crates —
//! `gg-abi` (below [`Component`](crate::Component), so it cannot carry persisted
//! identity), `gg-math` (types, no identity either), and this one. So the render
//! protocol is *three ordinary components* defined here, declared by the game
//! like any other, and read by the host through a typed query.
//!
//! The consequence worth naming: **nothing about how the game looks lives in the
//! host.** A system fills these in from whatever the game's own components say,
//! which puts colour, size and pose inside the dylib — reloadable, and editable
//! while someone is playing. The host's renderer knows two shapes, one colour
//! channel, and how to resolve a name in the pack it was handed; that is the
//! whole of its opinion — and the second shape arrived at §6 M26 as a field on
//! a component, not as a renderer setting, for exactly that reason.
//!
//! Layout agreement is not asserted here because it is already checked where it
//! matters: the schema hash covers every field's name, type token, offset and
//! size plus the struct's size and alignment, and `World::adopt` refuses a
//! declared component whose schema disagrees with the one already registered
//! (§4.2.2). Two compilations that laid these out differently would be refused
//! by name at load rather than drawing garbage.

use gg_abi::asset_id;
use gg_math::sim;

use crate::Component;

/// One primitive to draw, in world space.
///
/// Two of them since §6 M26 — a box and a sphere, chosen by
/// [`shape`](Renderable::shape). A cube, a floor slab and a tracer are all the
/// first with different [`half_extent`](Renderable::half_extent)s; the second
/// exists because a *curved* surface is the only one that shows a whole
/// specular lobe at once, and a material is not readable without that.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "gg.renderable")]
#[repr(C)]
pub struct Renderable {
    /// World-space centre, `f64` and un-narrowed — `gg-extract` is what makes it
    /// camera-relative `f32` (§1.4).
    pub position: sim::DVec3,
    /// Orientation. Identity for anything axis-aligned, which is most of it.
    pub rotation: sim::DQuat,
    /// Half-extent per axis, metres, *before* rotation. Non-uniform on purpose:
    /// a beam is a long thin box, and one primitive that stretches beats a
    /// second primitive.
    pub half_extent: sim::Vec3,
    /// `0x00RRGGBB`, sRGB. An integer rather than three floats: it is a colour
    /// the game picked, not a value anything computes with.
    pub color: u32,
    /// How polished the surface is, `0.0..=1.0` — 0 is chalk, 1 is a mirror.
    ///
    /// **Smoothness and not roughness, which is the field every renderer names
    /// and the inverse of this one.** The reason is [`quiet`](super::Prefs::quiet)'s,
    /// applied to a component that predates it: `World::restore` zeroes a field
    /// a migration could not carry (§4.2.2), and a zeroed *roughness* is a
    /// perfect mirror — so the first reload after this field appeared would
    /// silver every box in the world. A zeroed smoothness is a matte surface,
    /// which is the one answer that cannot look like a bug. The shader converts
    /// once, at [`make_surface`], and nothing downstream sees this spelling.
    ///
    /// Perceptual, not linear in the microfacet distribution: the BRDF squares
    /// `1 - smoothness` to get GGX's alpha, the [Bur12] remap every glTF asset
    /// is authored against, so a chart stepped evenly here reads as evenly
    /// stepped.
    pub smoothness: f32,
    /// Whether the surface is a conductor, `0.0..=1.0`. A metal's specular
    /// colour *is* its base colour and it has no diffuse lobe at all; a
    /// dielectric reflects a flat 4 % and scatters the rest. In between is not a
    /// physical material — it is the blend a texture authored at a boundary
    /// needs — but it is legal, and interpolating beats branching.
    ///
    /// Zero is a dielectric, which is both the common case and the safe
    /// migration, for [`smoothness`](Renderable::smoothness)'s reason.
    pub metallic: f32,
    /// Which primitive this draws — [`shape::BOX`] or [`shape::SPHERE`].
    ///
    /// A number with associated constants rather than an `enum`, for
    /// [`light`]'s reason: a bare enum in a component is refused by the derive,
    /// and an unknown value here draws the box rather than nothing. Zero is the
    /// box for [`smoothness`](Renderable::smoothness)'s reason — a migration
    /// that cannot carry this field zeroes it, and what a world full of boxes
    /// turns into on its first reload must be the world that was already there.
    ///
    /// `u64` rather than `u32`, and that is arithmetic and not ambition: `f64`
    /// puts this struct's alignment at 8, so a trailing `u32` would be followed
    /// by four bytes of implicit padding — which `bytemuck::Pod` refuses. Both
    /// spellings cost the same 88 bytes; only the narrow one needs a `_pad`
    /// beside it in the schema, in the inspector, and in every struct literal.
    pub shape: u64,
}

/// What [`Renderable::shape`] is. Associated constants rather than an `enum`
/// field, for [`light`]'s reason — see that module.
pub mod shape {
    /// An axis-aligned box of [`half_extent`](super::Renderable::half_extent),
    /// rotated. Zero, so it is what every migration and every zeroed field
    /// produces.
    pub const BOX: u64 = 0;
    /// An ellipsoid of the same half-extent — a sphere when the three axes
    /// agree, which is the case worth having and the only one a chart uses.
    pub const SPHERE: u64 = 1;
}

/// What [`Renderable::boxed`] leaves a surface at: middling-rough, dielectric.
///
/// Exactly the constants the box pass held before the material crossed the
/// boundary (§6 M5's `BOX_ROUGHNESS` was 0.6), so every game written before this
/// field existed draws the same picture it always did — the widening is visible
/// in the schema hash and nowhere in the frame.
pub const DEFAULT_SMOOTHNESS: f32 = 0.4;

impl Renderable {
    /// An axis-aligned box. The common case, spelled once so game code does not
    /// repeat `DQuat::IDENTITY`.
    #[must_use]
    pub fn boxed(position: sim::DVec3, half_extent: sim::Vec3, color: u32) -> Self {
        Renderable {
            position,
            rotation: sim::DQuat::IDENTITY,
            half_extent,
            color,
            smoothness: DEFAULT_SMOOTHNESS,
            metallic: 0.0,
            shape: shape::BOX,
        }
    }

    /// A sphere of `radius`, unrotated — the same primitive a
    /// [`boxed`](Renderable::boxed) call makes, drawn round.
    ///
    /// Uniform by construction: an ellipsoid is reachable by setting
    /// [`half_extent`](Renderable::half_extent) directly and is legal, but the
    /// thing a game asks for by name is a ball.
    #[must_use]
    pub fn ball(position: sim::DVec3, radius: f32, color: u32) -> Self {
        Renderable {
            shape: shape::SPHERE,
            ..Renderable::boxed(position, sim::Vec3::splat(radius), color)
        }
    }

    /// The same primitive with a material on it — the two knobs a game has over
    /// how a surface answers light, past the colour it already picked.
    #[must_use]
    pub fn surfaced(mut self, smoothness: f32, metallic: f32) -> Self {
        self.smoothness = smoothness;
        self.metallic = metallic;
        self
    }
}

/// One piece of pack content to draw, in world space (§4.6).
///
/// [`Renderable`] is the shape the host knows; this is the shape the *content*
/// knows. `asset` names a mesh or a scene in the pack the host was given, and
/// the host draws whatever that name resolves to — so the game still says what
/// it looks like, at the only granularity a file format leaves it.
///
/// An `asset` the pack does not contain draws nothing. Not an error: a pack is
/// rebuilt while the game runs (§4.6 watch mode), and a frame that failed
/// because a mesh was mid-rebuild would make an artist's save look like a crash.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "gg.model")]
#[repr(C)]
pub struct Model {
    /// World-space origin, `f64` and un-narrowed — as [`Renderable::position`].
    pub position: sim::DVec3,
    /// Orientation.
    pub rotation: sim::DQuat,
    /// The pack asset this draws: `gg_assets::AssetId`'s value, or 0 for none.
    /// A `u64` rather than that type because a game crate may not link the
    /// crate that defines it (§3), which is also why [`asset_id`] is where it
    /// is — see that function for the whole argument.
    pub asset: u64,
    /// Scale per axis, applied before rotation. `1.0` is the size the asset was
    /// authored at, which is what makes an unscaled placement the common case.
    pub scale: sim::Vec3,
    /// `0x00RRGGBB`, sRGB, multiplied into the material's base colour. White
    /// leaves the asset looking as it was authored.
    pub tint: u32,
}

impl Model {
    /// White, unrotated, unscaled — an asset placed as authored.
    #[must_use]
    pub fn at(name: &str, position: sim::DVec3) -> Self {
        Model {
            position,
            rotation: sim::DQuat::IDENTITY,
            asset: asset_id(name),
            scale: sim::Vec3::splat(1.0),
            tint: 0x00ff_ffff,
        }
    }
}

/// What a [`Light`] is. Associated constants rather than an `enum` field: a
/// bare enum in a component is refused by the derive (a value outside the
/// declared discriminants is UB the moment it is read, and the dylib on the
/// other side of the boundary is not the compiler that wrote it), so the
/// discriminant crosses as a `u32` and the host treats an unknown one as
/// nothing to shade with.
pub mod light {
    /// Parallel rays from infinitely far away — a sun. Position is unread;
    /// direction and colour are the whole of it.
    pub const DIRECTIONAL: u32 = 0;
    /// A point emitting in every direction, falling off with distance. Position
    /// and range are read; direction is unread.
    pub const POINT: u32 = 1;
}

/// One light, in world space (§6 M11).
///
/// A component for the same reason [`Renderable`] is one: the host holds no
/// game types, so "there is a sun here, and it is this colour" has to be data
/// the game *declares* rather than a renderer setting somebody edits in the
/// engine. A game with no lights renders unlit, which is visible and obviously
/// wrong — the same choice [`Eye::ORIGIN`] makes.
///
/// Intensity is separate from colour because they answer different questions:
/// colour is the light's tint and lives in eight bits a human picks, intensity
/// is a physical quantity with no upper bound and lives in a float. Multiplying
/// them into one `f32` triple at the boundary would make "twice as bright"
/// unrepresentable above white.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "gg.light")]
#[repr(C)]
pub struct Light {
    /// World-space position, `f64` and un-narrowed — as [`Renderable::position`].
    /// Unread for [`light::DIRECTIONAL`].
    pub position: sim::DVec3,
    /// The direction the light *travels*, unit length, for
    /// [`light::DIRECTIONAL`]. Travel rather than "toward the light", because
    /// that is the one a game author can point at the ground without thinking
    /// about the shading equation.
    ///
    /// `f32` because a direction is not a position: it never gains planetary
    /// magnitude, which is the only thing `f64` buys on this side of §1.4.
    pub direction: sim::Vec3,
    /// `0x00RRGGBB`, sRGB — the light's tint, as [`Renderable::color`].
    pub color: u32,
    /// Radiance multiplier, linear. Unbounded above: an outdoor sun and a candle
    /// differ by orders of magnitude, which is the whole reason the scene
    /// attachment is a float target.
    pub intensity: f32,
    /// Metres at which a [`light::POINT`] contributes nothing. The falloff is
    /// inverse-square windowed to reach exactly zero here, so a range is a
    /// culling bound the renderer can trust rather than a fade that never quite
    /// ends. Unread for [`light::DIRECTIONAL`].
    pub range: f32,
    /// One of [`light`]'s constants.
    pub kind: u32,
    /// Padding, spelled out. `Pod` refuses a struct with holes in it, and the
    /// alternative to naming this field is a layout that changes the moment
    /// somebody reorders the ones above.
    pub reserved: u32,
}

impl Light {
    /// A sun: parallel rays travelling in `direction`.
    ///
    /// # Panics
    /// Never — a zero direction is left as it is and shades nothing, which is
    /// the same "visible and obviously wrong" rule the rest of the protocol
    /// follows.
    #[must_use]
    pub fn sun(direction: sim::Vec3, color: u32, intensity: f32) -> Self {
        Light {
            position: sim::DVec3::ZERO,
            direction,
            color,
            intensity,
            range: 0.0,
            kind: light::DIRECTIONAL,
            reserved: 0,
        }
    }

    /// A point light at `position`, dark by `range` metres.
    #[must_use]
    pub fn point(position: sim::DVec3, color: u32, intensity: f32, range: f32) -> Self {
        Light {
            position,
            direction: sim::Vec3::ZERO,
            color,
            intensity,
            range,
            kind: light::POINT,
            reserved: 0,
        }
    }
}

/// The environment every surface is lit by when no [`Light`] is pointing at it
/// — image-based lighting's *image*, declared the way everything else on this
/// boundary is (§6 M24).
///
/// # Why three colours and not a texture
///
/// An IBL is conventionally a captured HDR panorama, and a panorama is a file:
/// it would need a pack format, an importer, and a decision about whose taste
/// the default one is. What actually lights a surface is the low-frequency part
/// of that panorama — a diffuse lobe integrates over a whole hemisphere and a
/// rough specular one over most of it — so the machinery this protocol has to
/// exist for is the *projection*, not the source. A vertical gradient is the
/// smallest thing that exercises all of it: the host projects it to spherical
/// harmonics exactly as it would project a panorama, and the day a panorama
/// arrives it changes what feeds the projection and nothing downstream of it.
///
/// # Why the sun is not in here
///
/// A real sky photograph has the sun burnt into it, and an engine that both
/// lit from that pixel *and* shaded the [`Light`] pointing the same way would
/// count the same photon twice. The split is therefore: the sun is a
/// [`Light`], and this is everything else the sky does — which is also why a
/// surface facing away from the sun is lit at all.
///
/// # The zero
///
/// [`intensity`](Sky::intensity) of zero is *no environment*, and a world that
/// declares no `Sky` is exactly the flat `r.ambient` term that came before this
/// (§6 M11). So the migration `World::restore` performs — every field zeroed —
/// puts a world back to the lighting it had, and every scene blessed before
/// this component existed still renders itself.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "gg.sky")]
#[repr(C)]
pub struct Sky {
    /// `0x00RRGGBB`, sRGB — straight up.
    pub zenith: u32,
    /// The same, at the horizon. Reached along a square root of altitude rather
    /// than linearly, which is what puts most of the gradient in the bottom of
    /// the sky where a real one keeps it.
    pub horizon: u32,
    /// The same, straight down: what the ground bounces back up. The half of an
    /// environment a gradient usually forgets, and the reason the underside of
    /// a thing is dim rather than black.
    pub ground: u32,
    /// Linear multiplier over all three, and the switch: zero is no environment
    /// at all. Separate from the colours for [`Light::intensity`]'s reason — a
    /// sky brighter than white is not expressible in eight bits.
    pub intensity: f32,
}

impl Sky {
    /// A plain daylight environment: pale blue overhead, near-white at the
    /// horizon, dim warm grey below. Somewhere to start rather than a
    /// measurement — a game with an opinion sets the fields.
    #[must_use]
    pub fn daylight(intensity: f32) -> Self {
        Sky {
            zenith: 0x0059_8fd8,
            horizon: 0x00c8_d4e0,
            ground: 0x0035_3129,
            intensity,
        }
    }
}

/// Where the game is looked at from — and, since §6 M20, *how*.
///
/// Yaw and pitch rather than a quaternion: the host builds a fly camera's basis
/// from them in one order (Y then X, never roll), and a quaternion here would
/// make "which way is up" the game's problem to get right per frame.
///
/// With more than one in a world the **lowest entity index** wins — see
/// [`Eye::of`] for why that and not iteration order. A game with no eye renders
/// from the origin — visible, wrong, and obviously so, which beats a black
/// screen that could mean anything.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "gg.eye")]
#[repr(C)]
pub struct Eye {
    /// World-space eye position.
    pub position: sim::DVec3,
    /// Rotation about +Y, radians.
    pub yaw: f32,
    /// Rotation about the camera's right axis, radians.
    pub pitch: f32,
    /// Vertical half-height of an **orthographic** view, in metres — how much
    /// world the window shows, which is framing and therefore the game's
    /// (§6 M20). `0.0` is perspective, and that zero is a migration contract
    /// rather than a convenience: `World::restore` zero-fills a field an older
    /// world never held, so every world that predates this field reopens
    /// perspective — exactly what it was. Sim state like the rest of the eye:
    /// a zoom is a hashed, replayed fact.
    pub ortho: f32,
    /// Padding, spelled out — `Pod` refuses a struct with holes in it, and
    /// naming the hole keeps the layout a visible edit (as [`Light::reserved`]).
    pub reserved: u32,
}

impl Eye {
    /// Unrotated, at the world origin — what a game that declares no eye is
    /// rendered from.
    pub const ORIGIN: Eye = Eye {
        position: sim::DVec3::ZERO,
        yaw: 0.0,
        pitch: 0.0,
        ortho: 0.0,
        reserved: 0,
    };

    /// A perspective eye at `position`, looking along `yaw`/`pitch` in radians.
    /// Spelled once here so a game does not repeat the literal, on the same
    /// reasoning as [`Renderable::boxed`] — and because the M12 template
    /// criterion counts every line of ceremony a game pays (§6 M12).
    #[must_use]
    pub fn at(position: sim::DVec3, yaw: f32, pitch: f32) -> Self {
        Eye {
            position,
            yaw,
            pitch,
            ortho: 0.0,
            reserved: 0,
        }
    }

    /// An orthographic eye at `position`, looking straight down -Z over
    /// `half_height` metres of vertical view — the 2D game's camera (§6 M20).
    /// The half-*height*, with width following the window's aspect, because a
    /// platformer meets a wider window with more world and never with a
    /// stretch.
    #[must_use]
    pub fn flat(position: sim::DVec3, half_height: f32) -> Self {
        Eye {
            position,
            yaw: 0.0,
            pitch: 0.0,
            ortho: half_height,
            reserved: 0,
        }
    }

    /// The eye a host renders from: the live one with the **lowest entity
    /// index**, or [`Eye::ORIGIN`] when the game declared none.
    ///
    /// A rule and not an accident (§6 M15.2). Iteration order is archetype
    /// order, so "the first one" is decided by which archetype a camera landed
    /// in — adding an unrelated component to the observer would silently move
    /// the scene onto the other eye. Index order is stable under that, and a
    /// game wanting the other camera moves the spawn rather than the layout.
    ///
    /// Visible and obviously wrong beats a black screen, which could mean the
    /// game declared no eye, spawned nothing, or crashed.
    ///
    /// # Errors
    ///
    /// If the world refuses the query, which one read alone cannot cause.
    pub fn of(world: &crate::World) -> Result<Eye, crate::AliasError> {
        let query = crate::Query::<&Eye>::new()?;
        let mut found: Option<(u32, Eye)> = None;
        world.each_ref(&query, |entity, eye: &Eye| {
            let index = entity.index();
            if found.is_none_or(|(best, _)| index < best) {
                found = Some((index, *eye));
            }
        });
        Ok(found.map_or(Eye::ORIGIN, |(_, eye)| eye))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn the_protocol_types_are_flat_and_padding_free() {
        // `Pod` already refuses padding; these pin the numbers so a field added
        // to either one is a visible edit rather than a silent layout move.
        // 88 since §6 M26's `shape`, and the eight is the whole argument for
        // spelling it `u64`: at align 8 a trailing `u32` would have cost the
        // same eight bytes and a `_pad` field beside it.
        assert_eq!(size_of::<Renderable>(), 88);
        assert_eq!(align_of::<Renderable>(), 8);
        assert_eq!(size_of::<Eye>(), 40);
        assert_eq!(align_of::<Eye>(), 8);
        assert_eq!(size_of::<Model>(), 80);
        assert_eq!(align_of::<Model>(), 8);
        assert_eq!(size_of::<Light>(), 56);
        assert_eq!(align_of::<Light>(), 8);
        assert_eq!(size_of::<Sky>(), 16);
        assert_eq!(align_of::<Sky>(), 4);
    }

    /// The migration law this protocol's zeros are chosen against (§4.2.2): a
    /// field `World::restore` could not carry comes back zeroed, and every one
    /// of those zeros has to be what the engine did before the field existed.
    #[test]
    fn a_zeroed_sky_is_no_environment_and_a_zeroed_material_is_chalk() {
        let sky: Sky = bytemuck::Zeroable::zeroed();
        assert_eq!(
            sky.intensity, 0.0,
            "which is the flat ambient term, as before"
        );
        let mut box3 = Renderable::boxed(sim::DVec3::ZERO, sim::Vec3::splat(0.5), 0x00ff_ffff);
        box3.smoothness = 0.0;
        box3.metallic = 0.0;
        assert_eq!(1.0 - box3.smoothness, 1.0, "fully rough, never a mirror");
        let zeroed: Renderable = bytemuck::Zeroable::zeroed();
        assert_eq!(
            zeroed.shape,
            shape::BOX,
            "a world that lost this field on a reload comes back the boxes it was"
        );
    }

    /// The two spellings a game reaches for, and the one difference between
    /// them. Not a tautology: `ball` is `boxed` with one field moved, so an
    /// edit that made it a separate constructor could silently drop the
    /// material default or the identity rotation.
    #[test]
    fn a_ball_is_a_box_that_differs_only_in_its_shape() {
        let at = sim::DVec3::new(1.0, 2.0, 3.0);
        let ball = Renderable::ball(at, 0.5, 0x0012_3456);
        let boxed = Renderable::boxed(at, sim::Vec3::splat(0.5), 0x0012_3456);
        assert_eq!(ball.shape, shape::SPHERE);
        assert_eq!(boxed.shape, shape::BOX);
        assert_eq!(
            Renderable {
                shape: shape::BOX,
                ..ball
            },
            boxed
        );
    }

    #[test]
    fn a_sun_carries_no_position_and_a_point_light_no_direction() {
        // The unread field of each is *zero* rather than left over: a component
        // is `Pod` and a game may memcpy one, so an unread field that happened
        // to hold a stale value would hash differently for two lights that are
        // the same light.
        let sun = Light::sun(sim::Vec3::new(0.0, -1.0, 0.0), 0x00ff_eedd, 3.0);
        assert_eq!(sun.kind, light::DIRECTIONAL);
        assert_eq!(sun.position, sim::DVec3::ZERO);
        assert_eq!(sun.range, 0.0);

        let lamp = Light::point(sim::DVec3::new(1.0, 2.0, 3.0), 0x00ff_8800, 12.0, 8.0);
        assert_eq!(lamp.kind, light::POINT);
        assert_eq!(lamp.direction, sim::Vec3::ZERO);
        assert_eq!(lamp.range, 8.0);
        assert_eq!(lamp.reserved, 0, "padding is written, not inherited");
    }

    #[test]
    fn a_model_names_its_asset_by_the_hash_the_pack_stores() {
        let model = Model::at("hall/scene", sim::DVec3::ZERO);
        assert_eq!(model.asset, asset_id("hall/scene"));
        assert_eq!(model.scale, sim::Vec3::splat(1.0));
        assert_eq!(model.tint, 0x00ff_ffff, "authored colours, untinted");
    }

    #[test]
    fn a_boxed_renderable_is_unrotated() {
        let r = Renderable::boxed(sim::DVec3::ZERO, sim::Vec3::splat(0.5), 0x00ff_8000);
        assert_eq!(r.rotation, sim::DQuat::IDENTITY);
        assert_eq!(r.color, 0x00ff_8000);
    }

    /// The two orders are made to **disagree** on purpose: the low-index eye is
    /// pushed into a later archetype by gaining a second component, so "first
    /// in iteration order" and "lowest index" name different cameras. Without
    /// that the test would pass against the rule it is meant to pin (§6 M15.2).
    #[test]
    fn two_eyes_render_from_the_lower_index_whatever_archetype_holds_it() {
        let mut world = crate::World::new();
        let (low, high) = (world.spawn(), world.spawn());
        assert!(low.index() < high.index(), "spawn order is index order");

        // `high` reaches the plain-`Eye` archetype first, so iteration arrives
        // there before the `(Eye, Renderable)` one `low` ends up in.
        world
            .insert(high, Eye::at(sim::DVec3::new(0.0, 0.0, 9.0), 0.0, 0.0))
            .unwrap();
        world
            .insert(low, Eye::at(sim::DVec3::new(0.0, 0.0, 1.0), 0.0, 0.0))
            .unwrap();
        world
            .insert(
                low,
                Renderable::boxed(sim::DVec3::ZERO, sim::Vec3::splat(0.5), 0x00ff_8000),
            )
            .unwrap();

        // The disagreement is asserted rather than assumed: if archetype order
        // ever reached `low` first this test would pass against the rule it
        // replaced and prove nothing.
        let query = crate::Query::<&Eye>::new().unwrap();
        let mut first = None;
        world.each_ref(&query, |entity, _: &Eye| first = first.or(Some(entity)));
        assert_eq!(
            first.unwrap(),
            high,
            "iteration order reaches the high index"
        );

        assert_eq!(
            Eye::of(&world).unwrap().position.z,
            1.0,
            "lowest index wins"
        );

        // And the rule holds when the survivor is the *other* one: despawning
        // the low index hands the scene to the eye that is left, rather than to
        // whichever archetype happens to be walked first.
        world.despawn(low);
        assert_eq!(Eye::of(&world).unwrap().position.z, 9.0);
    }

    #[test]
    fn a_world_with_no_eye_renders_from_the_origin() {
        let world = crate::World::new();
        assert_eq!(Eye::of(&world).unwrap(), Eye::ORIGIN);
    }
}
