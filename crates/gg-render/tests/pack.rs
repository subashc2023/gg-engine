//! A pack on the screen, and the rebuild that replaces it (§4.6).
//!
//! The whole chain with no window in it: a game component naming a scene,
//! extract's expansion and narrowing, residency's upload, the pack pass's
//! pipelines, and pixels back through §4.5's readback pass. The shell runs the
//! same passes and is manual (§1.5), so this is where they are proven.
//!
//! The texture path is deliberately *not* here: these packs are hand-written
//! and a hand-written BC7 block would prove something about this file's
//! encoder. Real `ggc` texels reach the screen through the golden suite, which
//! is where a colour anyone can look at belongs (§4.10).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use gg_assets::pack::{AssetId, AssetKind};
use gg_assets::texture::{self, TextureFormat};
use gg_assets::{Material, Node, PackWriter, Vertex, mesh, scene};
use gg_ecs::World;
use gg_ecs::boundary::{Light, Model};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View};

const EXTENT: (u32, u32) = (64, 64);
const SCENE: &str = "test/scene";
const MESH: &str = "test/mesh/0.0";
const MATERIAL: &str = "test/material/0";

/// The facet normal: a quad in the `z = 0` plane faces +Z.
const FLAT: [f32; 3] = [0.0, 0.0, 1.0];

/// A quad facing +Z, one metre to a side, so an unrotated eye sees its face.
///
/// `normal` is *authored* per vertex and need not be the facet's — every
/// smooth-shaded mesh in existence carries one that is not. That is what
/// [`a_stretched_instance_shades_by_its_inverse_transposed_normal`] exploits:
/// the transform under test is the only thing between it and the pixel.
fn quad(normal: [f32; 3]) -> Vec<u8> {
    quad_with_tangent(normal, [1.0, 0.0, 0.0, 1.0])
}

fn quad_with_tangent(normal: [f32; 3], tangent: [f32; 4]) -> Vec<u8> {
    let corner = |x: f32, y: f32, u: f32, v: f32| Vertex {
        position: [x, y, 0.0],
        normal,
        uv: [u, v],
        tangent,
    };
    let vertices = [
        corner(-1.0, -1.0, 0.0, 1.0),
        corner(1.0, -1.0, 1.0, 1.0),
        corner(1.0, 1.0, 1.0, 0.0),
        corner(-1.0, 1.0, 0.0, 0.0),
    ];
    mesh::encode(&vertices, &[0, 1, 2, 0, 2, 3], AssetId::of(MATERIAL))
}

/// A one-node scene four metres down -Z, drawn in `base_color`.
fn pack_bytes(base_color: [f32; 4], normal: [f32; 3]) -> Vec<u8> {
    let mut writer = PackWriter::new();
    writer.add(MESH, AssetKind::Mesh, 0, quad(normal)).unwrap();
    writer
        .add(
            MATERIAL,
            AssetKind::Material,
            1,
            Material {
                base_color,
                // A plain dielectric, so the pixel this test reads is the base
                // colour and not a metal's specular response to a light that is
                // not there — glTF's default is fully metallic, which renders
                // black without one (§6 M11).
                metallic: 0.0,
                roughness: 1.0,
                ..Material::default()
            }
            .encode(),
        )
        .unwrap();
    writer
        .add(
            SCENE,
            AssetKind::Scene,
            2,
            scene::encode(&[Node {
                mesh: AssetId::of(MESH),
                translation: [0.0, 0.0, -4.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
                reserved: [0; 5],
            }]),
        )
        .unwrap();
    writer.finish().unwrap()
}

/// Write a pack the way `ggc` does — to a temporary, then renamed over. On
/// Windows that is also the only form that can replace a file another process
/// has mapped, which is exactly the situation watch mode creates.
fn write_pack(path: &PathBuf, bytes: &[u8]) {
    let temporary = path.with_extension("tmp");
    std::fs::write(&temporary, bytes).unwrap();
    std::fs::rename(&temporary, path).unwrap();
}

fn temp(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("gg-render-{}-{name}.ggpack", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path
}

/// A world holding one model that names the scene, and a sun to see it by.
fn world() -> World {
    let mut world = World::new();
    world.register::<Model>().unwrap();
    world.register::<Light>().unwrap();
    let entity = world.spawn();
    world
        .insert(entity, Model::at(SCENE, sim::DVec3::ZERO))
        .unwrap();
    // Straight down -Z, so it lands square on a quad facing +Z. Without it the
    // quad is lit by the ambient term alone and this file would be asserting
    // that a knob is not zero.
    let sun = world.spawn();
    world
        .insert(
            sun,
            Light::sun(sim::Vec3::new(0.0, 0.0, -1.0), 0x00ff_ffff, 3.0),
        )
        .unwrap();
    world
}

/// Frames that run before an idle reading is believed. The first frame's
/// request is what *loads* the scene — extract cannot expand a scene the
/// renderer has not been shown yet — so frame one reads idle for the reason
/// that it is the state before the meshes are even known about, and streaming
/// is frames deep by design (§4.6).
const SETTLE_MIN: usize = 4;

/// The bound, not the target. Reached only if something never becomes resident,
/// which is a failure worth naming rather than a picture worth returning: an
/// under-settled frame draws the fallback and fails whatever *shading* claim
/// the caller was making, several inferences away from the cause.
const SETTLE_MAX: usize = 240;

/// Render until nothing is pending and one whole frame has been drawn since,
/// then keep the last.
///
/// A count was what this used to be, and a count is a proxy: four frames is
/// enough on an idle desk and not obviously enough on one running the rest of
/// the suite against the same GPU. `pack.rs`'s tangent test was seen to fail
/// once inside a full-workspace run on 2026-08-04 and did not reproduce in six
/// further runs, so this is the leading suspect rather than a confirmed cause —
/// but the two idle frames are what the count was *approximating*, and if
/// residency is ever the reason again the panic below says so by name.
fn settle(renderer: &mut OffscreenRenderer, world: &World) -> Vec<u8> {
    let mut extracted = Extracted::default();
    let mut idle = 0;
    for frame in 1..=SETTLE_MAX {
        extracted.clear(sim::DVec3::ZERO, gg_extract::Frustum::UNBOUNDED);
        extracted
            .append_models::<Model>(world, renderer.scenes())
            .unwrap();
        extracted.append_lights(world).unwrap();
        let pixels = renderer
            .frame(&extracted, &View::default(), [0.0, 0.0, 0.0, 1.0], &[])
            .unwrap()
            .pixels;
        // Two, not one: the frame that drains the last request may have drawn
        // before it, so the second is the first drawn wholly resident.
        idle = if renderer.pack().is_none_or(|pack| pack.pending() == 0) {
            idle + 1
        } else {
            0
        };
        if frame >= SETTLE_MIN && idle >= 2 {
            return pixels;
        }
    }
    panic!(
        "still streaming after {SETTLE_MAX} frames: {} assets pending",
        renderer
            .pack()
            .map_or(0, gg_render::content::Content::pending)
    );
}

fn at(pixels: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * EXTENT.0 + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2]]
}

/// A normal 45° between +X and +Z. Off the axis the instance is stretched along,
/// which is the whole point, and still square enough to the eye that the
/// specular term stays tame — a normal perpendicular to the view rides
/// `n_dot_v`'s clamp, where the visibility term is at its most sensitive.
const TILTED: [f32; 3] = [
    std::f32::consts::FRAC_1_SQRT_2,
    0.0,
    std::f32::consts::FRAC_1_SQRT_2,
];

/// Where the quad's centre ends up: the scene node's translation, unscaled,
/// because both stretches below leave `x` and `z` of a zero-`x` offset alone.
const QUAD: sim::DVec3 = sim::DVec3::new(0.0, 0.0, -4.0);

/// A point light rather than a sun, for two reasons that both matter here: a
/// sun draws a shadow pass, and a sun grazing this quad's *facet* would put the
/// shadow bias knobs between the normal and the pixel; and a point light far
/// enough out is parallel to within a rounding at the one pixel this reads.
/// 16 m against a 64 m range keeps the windowed falloff near 1.
const LIGHT_DISTANCE: f64 = 16.0;
const LIGHT_RANGE: f32 = 64.0;
/// Chosen so the control reading lands mid-range: bright enough to be well off
/// the ambient floor, dim enough to stay under the tonemapper's knee, where the
/// curve is the identity and a change in shading is a change in the pixel.
const LIGHT_INTENSITY: f32 = 520.0;

/// The quad at `scale`, lit from `toward_light` — a unit axis pointing *from*
/// the quad at the lamp.
fn stretched_world(scale: sim::Vec3, toward_light: sim::DVec3) -> World {
    let mut world = World::new();
    world.register::<Model>().unwrap();
    world.register::<Light>().unwrap();
    let entity = world.spawn();
    let mut model = Model::at(SCENE, sim::DVec3::ZERO);
    model.scale = scale;
    world.insert(entity, model).unwrap();
    let lamp = world.spawn();
    world
        .insert(
            lamp,
            Light::point(
                QUAD + toward_light * LIGHT_DISTANCE,
                0x00ff_ffff,
                LIGHT_INTENSITY,
                LIGHT_RANGE,
            ),
        )
        .unwrap();
    world
}

/// Render one configuration and return the centre pixel summed over rgb. Grey
/// in, grey out — the sum is the same reading at three times the resolution.
fn lit(path: &Path, scale: sim::Vec3, toward_light: sim::DVec3) -> u32 {
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    renderer.open_pack(path).unwrap();
    let pixels = settle(&mut renderer, &stretched_world(scale, toward_light));
    let middle = at(&pixels, EXTENT.0 / 2, EXTENT.1 / 2);
    assert!(renderer.shutdown().clean(), "no leaks, no validation");
    u32::from(middle[0]) + u32::from(middle[1]) + u32::from(middle[2])
}

#[test]
fn a_stretched_instance_shades_by_its_inverse_transposed_normal() {
    // The model matrix is T*R*S with S diagonal, so a normal transforms by
    // R*S^-1 and not by R alone (`scene.slang`). Stretching X by four tips a
    // normal that sat 45° between +X and +Z over toward +Z: `atan(1/4)` off the
    // axis instead of 45°. So the same instance, lit along +X, must go *dimmer*
    // than the unstretched one, and lit along +Z must go *brighter* — and the
    // two must move by different amounts, since only one of them is the axis
    // the stretch acts on.
    //
    // Under the rotation-only shortcut the instance's scale never reaches the
    // normal at all and all four readings below collapse to one number, which is
    // what makes this a gate rather than a comparison.
    let path = temp("stretch");
    write_pack(&path, &pack_bytes([1.0, 1.0, 1.0, 1.0], TILTED));

    let plus_x = sim::DVec3::new(1.0, 0.0, 0.0);
    let plus_z = sim::DVec3::new(0.0, 0.0, 1.0);
    let unit = sim::Vec3::splat(1.0);
    let wide = sim::Vec3::new(4.0, 1.0, 1.0);

    let control_x = lit(&path, unit, plus_x);
    let control_z = lit(&path, unit, plus_z);
    let wide_x = lit(&path, wide, plus_x);
    let wide_z = lit(&path, wide, plus_z);

    // The control is the symmetry the rest rests on: at 45° the normal faces
    // both lamps equally, so the two readings agree. A setup that failed this
    // would make the comparisons below a claim about the lamp, not the scale.
    assert!(
        control_x.abs_diff(control_z) <= 6,
        "the unstretched quad faces both lamps alike: {control_x} vs {control_z}"
    );
    // Off the ambient floor and under the knee, so the readings below are the
    // shading and not the tonemapper.
    assert!(
        (150..690).contains(&control_x),
        "the control is mid-range: {control_x}"
    );

    assert!(
        wide_x + 90 < control_x,
        "stretched along the lamp's own axis, the normal turns away: \
         {wide_x} vs {control_x}"
    );
    assert!(
        wide_z > control_z + 30,
        "and toward the other one: {wide_z} vs {control_z}"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_scene_named_by_a_game_reaches_the_target_as_pack_geometry() {
    let path = temp("draw");
    write_pack(&path, &pack_bytes([1.0, 0.0, 0.0, 1.0], FLAT));
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    renderer.open_pack(&path).unwrap();
    let world = world();

    let pixels = settle(&mut renderer, &world);
    let middle = at(&pixels, EXTENT.0 / 2, EXTENT.1 / 2);
    assert!(middle[0] > 0x40, "the quad is there and red: {middle:?}");
    // Not "and only red": since M11 a dielectric carries a white specular
    // highlight, so the other two channels are small rather than zero. What is
    // still true — and what this file is actually about — is that the base
    // colour in the pack is the colour that dominates on the screen.
    assert!(
        middle[0] > middle[1] * 3 && middle[0] > middle[2] * 3,
        "red dominates: {middle:?}"
    );
    // A one-metre quad four metres out at the default fov leaves the corners
    // clear, which is what makes the middle a claim about geometry and not
    // about a fullscreen fill.
    assert_eq!(at(&pixels, 0, 0), [0, 0, 0], "the corners are the clear");

    let pack = renderer.pack().unwrap();
    assert_eq!(pack.pending(), 0);
    assert!(
        pack.ready_at().is_some(),
        "the load clock stops when the last asset lands (§6 M9)"
    );

    assert!(renderer.shutdown().clean(), "no leaks, no validation");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_rebuilt_pack_is_re_uploaded_without_the_game_noticing() {
    // §4.6 watch mode, end to end and windowless: the artist saves, `ggc`
    // renames a new pack over the old one, and the next frame is the new
    // colour. Nothing in the world changed — the game's `Model` still names
    // the same scene by the same name.
    let path = temp("reload");
    write_pack(&path, &pack_bytes([1.0, 0.0, 0.0, 1.0], FLAT));
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    renderer.open_pack(&path).unwrap();
    let world = world();

    let before = settle(&mut renderer, &world);
    assert!(at(&before, 32, 32)[0] > 0x40, "red first");

    // A stamp is (mtime, len) and both packs are the same length, so a rebuild
    // inside one filesystem timestamp tick would be missed. Sleeping is the
    // honest fix in a test; a real edit is never this fast.
    std::thread::sleep(std::time::Duration::from_millis(20));
    write_pack(&path, &pack_bytes([0.0, 1.0, 0.0, 1.0], FLAT));
    assert!(renderer.reload_pack().unwrap(), "the rewrite was noticed");

    let after = settle(&mut renderer, &world);
    let middle = at(&after, 32, 32);
    assert!(middle[1] > 0x40, "green after the rebuild: {middle:?}");
    // The red left in the pixel is the white specular highlight, not the old
    // base colour: green out-dominates it by the same margin it used to.
    assert!(
        middle[1] > middle[0] * 3 && middle[1] > middle[2] * 3,
        "nothing of the old colour: {middle:?}"
    );

    assert!(renderer.shutdown().clean(), "no leaks, no validation");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_model_naming_nothing_the_pack_holds_draws_nothing_and_does_not_fail() {
    // The state between a save and a finished rebuild. A frame that failed
    // here would make Ctrl-S look like a crash.
    let path = temp("absent");
    write_pack(&path, &pack_bytes([1.0, 0.0, 0.0, 1.0], FLAT));
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    renderer.open_pack(&path).unwrap();

    let mut world = World::new();
    world.register::<Model>().unwrap();
    let entity = world.spawn();
    world
        .insert(entity, Model::at("nothing/at/all", sim::DVec3::ZERO))
        .unwrap();

    let pixels = settle(&mut renderer, &world);
    assert_eq!(
        at(&pixels, 32, 32),
        [0, 0, 0],
        "an empty frame, not a panic"
    );

    assert!(renderer.shutdown().clean(), "no leaks, no validation");
    let _ = std::fs::remove_file(&path);
}

// ---- the tangent half of the same transform -----------------------------

const NORMAL_TEX: &str = "test/normal/0";

/// A BC5 block whose sixteen texels all decode to `(r, g)`.
///
/// The module doc declines hand-written BC7 because a hand-written block would
/// be asserting something about this file's encoder. This is the one case that
/// carries no encoder: a BC4 block whose two endpoints are equal decodes to
/// that endpoint at every index, in either interpolation mode, so the six index
/// bytes can be zero and the result is exact by construction. BC5 is two of
/// them, red then green.
fn flat_bc5(r: u8, g: u8) -> Vec<u8> {
    let mut block = vec![0u8; 16];
    (block[0], block[1]) = (r, r);
    (block[8], block[9]) = (g, g);
    block
}

/// The map's tangent-space normal, `sampled * 2 - 1` (`apply_normal_map`):
/// `(0.6, 0.0)`, so `z` reconstructs to `0.8`. Leaning hard along +t is what
/// makes the shaded normal a function of the tangent frame at all — a flat
/// `(0, 0, 1)` map would shade identically whatever the tangent did.
const TS_LEAN_X: u8 = 204; // 0.8 * 255, decoding to +0.6
const TS_ZERO_Y: u8 = 128; // 0.502 * 255, decoding to ~0

/// A tangent 45° between +X and +Y, in the plane the quad's normal is
/// perpendicular to — so Gram-Schmidt leaves it alone and the *only* thing a
/// stretch along x changes is its direction.
const DIAGONAL_TANGENT: [f32; 4] = [
    std::f32::consts::FRAC_1_SQRT_2,
    std::f32::consts::FRAC_1_SQRT_2,
    0.0,
    1.0,
];

/// The scene of [`a_stretched_instance_shades_by_its_scaled_tangent`]: the same
/// quad, a normal map that leans along the tangent, and a diagonal tangent for
/// the stretch to act on.
fn normal_mapped_pack() -> Vec<u8> {
    let mut writer = PackWriter::new();
    writer
        .add(
            MESH,
            AssetKind::Mesh,
            0,
            quad_with_tangent(FLAT, DIAGONAL_TANGENT),
        )
        .unwrap();
    writer
        .add(
            NORMAL_TEX,
            AssetKind::Texture,
            1,
            texture::encode(
                TextureFormat::Bc5Unorm,
                4,
                4,
                &[flat_bc5(TS_LEAN_X, TS_ZERO_Y)],
            )
            .unwrap(),
        )
        .unwrap();
    writer
        .add(
            MATERIAL,
            AssetKind::Material,
            2,
            Material {
                base_color: [1.0; 4],
                metallic: 0.0,
                roughness: 1.0,
                normal_texture: AssetId::of(NORMAL_TEX),
                ..Material::default()
            }
            .encode(),
        )
        .unwrap();
    writer
        .add(
            SCENE,
            AssetKind::Scene,
            3,
            scene::encode(&[Node {
                mesh: AssetId::of(MESH),
                translation: [0.0, 0.0, -4.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                scale: [1.0; 3],
                reserved: [0; 5],
            }]),
        )
        .unwrap();
    writer.finish().unwrap()
}

#[test]
fn a_stretched_instance_shades_by_its_scaled_tangent() {
    // The other half of the transform the test above gates (`scene.slang`): a
    // normal goes by R*S^-1, a *tangent* by R*S. Both readings below share one
    // geometric normal — (0,0,1) is parallel to itself under any diagonal S —
    // so the tangent frame is the only thing between them and the pixel.
    //
    // Stretching x by four swings the 45° tangent to within 14° of +X, and the
    // mapped normal, which leans along that tangent, swings with it. Lit from
    // +Y the stretched quad must therefore go markedly *dimmer*. Under a
    // tangent that ignored scale — `rotate_by(q, t)`, which is what shipped
    // before M12's audit — the two readings are one number.
    let path = temp("tangent");
    write_pack(&path, &normal_mapped_pack());

    let plus_y = sim::DVec3::new(0.0, 1.0, 0.0);
    let control = lit(&path, sim::Vec3::splat(1.0), plus_y);
    let stretched = lit(&path, sim::Vec3::new(4.0, 1.0, 1.0), plus_y);

    assert!(
        control > stretched,
        "stretching along the tangent must turn the mapped normal away from a \
         +Y light: control {control}, stretched {stretched}"
    );
    // Measured 414 against 243 on both lavapipe and the 4090; the reverted
    // shader gives 414 against 414. The threshold sits well above rounding and
    // well below that gap — what this guards is the collapse to equality, not a
    // drift.
    assert!(
        control - stretched > 40,
        "the scale never reached the tangent: control {control}, stretched {stretched}"
    );
}
