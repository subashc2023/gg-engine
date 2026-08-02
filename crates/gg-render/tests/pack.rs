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

use std::path::PathBuf;

use gg_assets::pack::{AssetId, AssetKind};
use gg_assets::{Material, Node, PackWriter, Vertex, mesh, scene};
use gg_ecs::World;
use gg_ecs::boundary::Model;
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View};

const EXTENT: (u32, u32) = (64, 64);
const SCENE: &str = "test/scene";
const MESH: &str = "test/mesh/0.0";
const MATERIAL: &str = "test/material/0";

/// A quad facing +Z, one metre to a side, so an unrotated eye sees its face.
fn quad() -> Vec<u8> {
    let corner = |x: f32, y: f32, u: f32, v: f32| Vertex {
        position: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [u, v],
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
fn pack_bytes(base_color: [f32; 4]) -> Vec<u8> {
    let mut writer = PackWriter::new();
    writer.add(MESH, AssetKind::Mesh, 0, quad()).unwrap();
    writer
        .add(
            MATERIAL,
            AssetKind::Material,
            1,
            Material {
                base_color,
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

/// A world holding one model that names the scene.
fn world() -> World {
    let mut world = World::new();
    world.register::<Model>().unwrap();
    let entity = world.spawn();
    world
        .insert(entity, Model::at(SCENE, sim::DVec3::ZERO))
        .unwrap();
    world
}

/// Render a few frames and keep the last.
///
/// A fixed count rather than "until nothing is pending", because the first
/// frame's request is what *loads* the scene: extract cannot expand a scene the
/// renderer has not been shown yet, so an idle-looking frame one is the state
/// before the meshes are even known about. Streaming is frames deep by design
/// (§4.6) and a settle that stopped at the first idle would stop before it.
const SETTLE_FRAMES: usize = 4;

fn settle(renderer: &mut OffscreenRenderer, world: &World) -> Vec<u8> {
    let mut extracted = Extracted::default();
    let mut pixels = Vec::new();
    for _ in 0..SETTLE_FRAMES {
        extracted.clear(sim::DVec3::ZERO);
        extracted
            .append_models::<Model>(world, renderer.scenes())
            .unwrap();
        pixels = renderer
            .frame(&extracted, &View::default(), [0.0, 0.0, 0.0, 1.0], &[])
            .unwrap()
            .pixels;
    }
    pixels
}

fn at(pixels: &[u8], x: u32, y: u32) -> [u8; 3] {
    let i = ((y * EXTENT.0 + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2]]
}

#[test]
fn a_scene_named_by_a_game_reaches_the_target_as_pack_geometry() {
    let path = temp("draw");
    write_pack(&path, &pack_bytes([1.0, 0.0, 0.0, 1.0]));
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    renderer.open_pack(&path).unwrap();
    let world = world();

    let pixels = settle(&mut renderer, &world);
    let middle = at(&pixels, EXTENT.0 / 2, EXTENT.1 / 2);
    assert!(middle[0] > 0x40, "the quad is there and red: {middle:?}");
    assert_eq!((middle[1], middle[2]), (0, 0), "and only red");
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
    write_pack(&path, &pack_bytes([1.0, 0.0, 0.0, 1.0]));
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    renderer.open_pack(&path).unwrap();
    let world = world();

    let before = settle(&mut renderer, &world);
    assert!(at(&before, 32, 32)[0] > 0x40, "red first");

    // A stamp is (mtime, len) and both packs are the same length, so a rebuild
    // inside one filesystem timestamp tick would be missed. Sleeping is the
    // honest fix in a test; a real edit is never this fast.
    std::thread::sleep(std::time::Duration::from_millis(20));
    write_pack(&path, &pack_bytes([0.0, 1.0, 0.0, 1.0]));
    assert!(renderer.reload_pack().unwrap(), "the rewrite was noticed");

    let after = settle(&mut renderer, &world);
    let middle = at(&after, 32, 32);
    assert!(middle[1] > 0x40, "green after the rebuild: {middle:?}");
    assert_eq!(middle[0], 0, "and nothing of the old colour");

    assert!(renderer.shutdown().clean(), "no leaks, no validation");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_model_naming_nothing_the_pack_holds_draws_nothing_and_does_not_fail() {
    // The state between a save and a finished rebuild. A frame that failed
    // here would make Ctrl-S look like a crash.
    let path = temp("absent");
    write_pack(&path, &pack_bytes([1.0, 0.0, 0.0, 1.0]));
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
