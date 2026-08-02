//! The two fixed-shape blobs: a material record and a scene's node list.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_assets::material::{MaterialError, flags};
use gg_assets::scene::{self, Scene, SceneError, SceneHeader};
use gg_assets::{AssetId, Material, Node};

#[test]
fn a_material_round_trips_and_defaults_to_gltfs_own_defaults() {
    let plain = Material::default();
    assert_eq!(plain.base_color, [1.0; 4]);
    assert_eq!(plain.metallic, 1.0);
    assert_eq!(plain.roughness, 1.0);
    assert!(plain.base_color_texture.is_none());

    let brass = Material {
        base_color: [0.8, 0.6, 0.2, 1.0],
        metallic: 1.0,
        roughness: 0.35,
        flags: flags::DOUBLE_SIDED,
        base_color_texture: AssetId::of("textures/brass"),
        ..Material::default()
    };
    let read = Material::read(&brass.encode()).unwrap();
    assert_eq!(read.base_color, brass.base_color);
    assert_eq!(read.roughness, brass.roughness);
    assert_eq!(read.flags & flags::DOUBLE_SIDED, flags::DOUBLE_SIDED);
    assert_eq!(read.base_color_texture, brass.base_color_texture);
}

#[test]
fn a_blob_of_the_wrong_length_is_not_read_as_a_material() {
    // Exact, not "at least": a longer blob here means the entry's kind is
    // wrong, and reading the first 64 bytes of a mesh as a material would
    // succeed and produce nonsense.
    let mut bytes = Material::default().encode();
    bytes.push(0);
    let err = Material::read(&bytes).unwrap_err();
    assert!(matches!(err, MaterialError { len: 65, .. }), "{err}");
}

fn node(mesh: &str, x: f64) -> Node {
    Node {
        mesh: AssetId::of(mesh),
        translation: [x, 1.5, -2.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
        reserved: [0; 5],
    }
}

#[test]
fn a_scene_round_trips_and_keeps_its_translations_in_f64() {
    // The number below is the point of the test: it is representable in f64 and
    // not in f32, so a scene blob that narrowed would fail here rather than in
    // a large world nobody has built yet (§1.4).
    let far = 6_378_137.123_456_7;
    let nodes = [node("meshes/arch", far), node("meshes/lion", -far)];
    let blob = scene::encode(&nodes);
    let read = Scene::read(&blob).unwrap();
    assert_eq!(read.nodes().len(), 2);
    assert_eq!(read.nodes()[0].translation[0], far);
    assert_eq!(read.nodes()[1].translation[0], -far);
    assert_ne!(f64::from(far as f32), far);
    assert_eq!(read.nodes()[0].mesh, AssetId::of("meshes/arch"));
}

#[test]
fn an_empty_scene_reads_back_empty() {
    let blob = scene::encode(&[]);
    let read = Scene::read(&blob).unwrap();
    assert!(read.nodes().is_empty());
}

#[test]
fn a_scene_whose_nodes_leave_the_blob_is_refused() {
    let mut blob = scene::encode(&[node("meshes/arch", 0.0)]);
    let mut header: SceneHeader = bytemuck::pod_read_unaligned(&blob[..size_of::<SceneHeader>()]);
    header.node_count = 64;
    blob[..size_of::<SceneHeader>()].copy_from_slice(bytemuck::bytes_of(&header));
    let err = Scene::read(&blob).unwrap_err();
    assert!(matches!(err, SceneError::OutOfBounds { .. }), "{err}");

    let err = Scene::read(&[0; 4]).unwrap_err();
    assert!(matches!(err, SceneError::TooShort { .. }), "{err}");
}
