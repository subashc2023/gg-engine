//! The mesh blob: the round trip, the precomputed bounds, and what a blob that
//! is not a mesh does at the reader.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_assets::mesh::{self, Mesh, MeshError, MeshHeader};
use gg_assets::{AssetId, AssetKind, Pack, PackWriter, Vertex};

fn tetra() -> (Vec<Vertex>, Vec<u32>) {
    let at = |x: f32, y: f32, z: f32| Vertex {
        position: [x, y, z],
        normal: [0.0, 1.0, 0.0],
        uv: [x, z],
    };
    (
        vec![
            at(0.0, 0.0, 0.0),
            at(2.0, 0.0, 0.0),
            at(0.0, 3.0, 0.0),
            at(0.0, 0.0, -4.0),
        ],
        vec![0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2],
    )
}

#[test]
fn a_mesh_round_trips_through_a_pack_without_a_copy() {
    let (vertices, indices) = tetra();
    let material = AssetId::of("materials/brass");
    let blob = mesh::encode(&vertices, &indices, material);

    let mut writer = PackWriter::new();
    writer
        .add("meshes/tetra", AssetKind::Mesh, 0, blob)
        .unwrap();
    let bytes = writer.finish().unwrap();

    let pack = Pack::from_bytes(&bytes).unwrap();
    let entry = pack.find_by_name("meshes/tetra").unwrap();
    let read = Mesh::read(pack.blob(entry)).unwrap();

    assert_eq!(read.vertices(), vertices.as_slice());
    assert_eq!(read.indices(), indices.as_slice());
    assert_eq!(read.header.material, material);
    // The arrays are the pack's own bytes, not a copy of them: the vertex
    // slice has to point inside the blob it was read from.
    let blob = pack.blob(entry);
    let base = blob.as_ptr() as usize;
    let vertex = read.vertices().as_ptr() as usize;
    assert!((base..base + blob.len()).contains(&vertex));
}

#[test]
fn bounds_are_computed_at_import_rather_than_trusted() {
    let (vertices, indices) = tetra();
    let blob = mesh::encode(&vertices, &indices, AssetId::NONE);
    let read = Mesh::read(&blob).unwrap();
    assert_eq!(read.header.bounds_min, [0.0, 0.0, -4.0]);
    assert_eq!(read.header.bounds_max, [2.0, 3.0, 0.0]);
    assert!(read.header.material.is_none());
}

#[test]
fn an_empty_mesh_bounds_to_a_point_rather_than_to_infinity() {
    // A ±infinity box unions with anything to give ±infinity, so one empty
    // mesh would make a scene's bounds useless and the cause would be nowhere
    // near the culler that read them.
    let blob = mesh::encode(&[], &[], AssetId::NONE);
    let read = Mesh::read(&blob).unwrap();
    assert_eq!(read.header.bounds_min, [0.0; 3]);
    assert_eq!(read.header.bounds_max, [0.0; 3]);
    assert!(read.vertices().is_empty());
    assert!(read.indices().is_empty());
}

#[test]
fn a_blob_too_short_for_a_header_is_refused_before_it_is_cast() {
    let err = Mesh::read(&[0; 8]).unwrap_err();
    assert!(matches!(err, MeshError::TooShort { .. }), "{err}");
}

#[test]
fn a_run_that_leaves_the_blob_is_refused() {
    let (vertices, indices) = tetra();
    let mut blob = mesh::encode(&vertices, &indices, AssetId::NONE);
    let mut header: MeshHeader = bytemuck::pod_read_unaligned(&blob[..size_of::<MeshHeader>()]);
    header.vertex_count = 4096;
    blob[..size_of::<MeshHeader>()].copy_from_slice(bytemuck::bytes_of(&header));
    let err = Mesh::read(&blob).unwrap_err();
    assert!(matches!(err, MeshError::OutOfBounds { .. }), "{err}");
    assert!(format!("{err}").contains("the vertices"));
}

#[test]
fn a_run_at_an_offset_its_records_cannot_be_read_from_is_refused() {
    let (vertices, indices) = tetra();
    let mut blob = mesh::encode(&vertices, &indices, AssetId::NONE);
    let mut header: MeshHeader = bytemuck::pod_read_unaligned(&blob[..size_of::<MeshHeader>()]);
    header.indices += 1;
    blob[..size_of::<MeshHeader>()].copy_from_slice(bytemuck::bytes_of(&header));
    let err = Mesh::read(&blob).unwrap_err();
    assert!(matches!(err, MeshError::Misaligned { .. }), "{err}");
    assert!(format!("{err}").contains("the indices"));
}
