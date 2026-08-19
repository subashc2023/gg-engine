//! The runtime half: handles, reference counts, load states, and what the
//! worker pool does and does not change.
//!
//! The properties worth guarding are the ones a threaded loader gets wrong:
//! that a second `load` of one name is one asset and not two, that a texture is
//! usable only after a pump, that arrival order never reaches a consumer, and
//! that a mesh is never copied out of the mapping in the first place.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_assets::load::{Assets, LoadState, asset};
use gg_assets::pack::{AssetId, AssetKind};
use gg_assets::texture::{self, TextureFormat, full_mip_count, level_extent};
use gg_assets::{PackWriter, Vertex, mesh, scene};

/// A texture blob with a full chain, distinguishable by `seed`.
fn texture_blob(seed: u8, width: u32, height: u32) -> Vec<u8> {
    let levels: Vec<Vec<u8>> = (0..full_mip_count(width, height))
        .map(|level| {
            let (w, h) = level_extent(width, height, level);
            let len = TextureFormat::Bc7Srgb.level_bytes(w, h) as usize;
            (0..len)
                .map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed))
                .collect()
        })
        .collect();
    texture::encode(TextureFormat::Bc7Srgb, width, height, &levels).unwrap()
}

fn mesh_blob() -> Vec<u8> {
    let vertices = [
        Vertex {
            position: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            position: [1.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            uv: [1.0, 0.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
            position: [0.0, 0.0, 1.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
        },
    ];
    mesh::encode(&vertices, &[0, 1, 2], AssetId::of("mat/a"))
}

/// Three textures, a mesh, a material and a scene — enough for every path.
fn pack_bytes() -> Vec<u8> {
    let mut writer = PackWriter::new();
    for (index, name) in ["tex/a", "tex/b", "tex/c"].iter().enumerate() {
        writer
            .add(
                name,
                AssetKind::Texture,
                index as u64,
                texture_blob(index as u8, 16, 16),
            )
            .unwrap();
    }
    writer
        .add("mesh/a", AssetKind::Mesh, 10, mesh_blob())
        .unwrap();
    writer
        .add(
            "mat/a",
            AssetKind::Material,
            11,
            gg_assets::Material::default().encode(),
        )
        .unwrap();
    writer
        .add("scene/a", AssetKind::Scene, 12, scene::encode(&[]))
        .unwrap();
    writer.finish().unwrap()
}

fn assets() -> Assets {
    Assets::new(gg_assets::Pack::from_bytes(&pack_bytes()).unwrap())
}

#[test]
fn everything_but_a_texture_is_ready_the_moment_the_pack_is_mapped() {
    let mut assets = assets();
    let mesh = assets.load::<asset::Mesh>("mesh/a").unwrap();
    let material = assets.load::<asset::Material>("mat/a").unwrap();
    let scene = assets.load::<asset::Scene>("scene/a").unwrap();

    // No pump has run. A mapping is already loaded, and pretending otherwise
    // would mean copying meshes out of it to make them look like textures.
    assert_eq!(mesh.state(), LoadState::Ready);
    assert_eq!(material.state(), LoadState::Ready);
    assert_eq!(scene.state(), LoadState::Ready);

    let read = assets.mesh(&mesh).unwrap();
    assert_eq!(read.vertices().len(), 3);
    assert_eq!(read.indices(), &[0, 1, 2]);
    assert_eq!(assets.scene(&scene).unwrap().nodes().len(), 0);
    assert_eq!(assets.material(&material).unwrap().base_color, [1.0; 4]);
}

#[test]
fn a_mesh_is_read_out_of_the_mapping_and_never_copied_into_the_registry() {
    let mut assets = assets();
    let mesh = assets.load::<asset::Mesh>("mesh/a").unwrap();
    let vertices = assets.mesh(&mesh).unwrap().vertices();

    // The vertices lie inside the pack's own bytes. If the registry had copied
    // them, this pointer would be somewhere else entirely.
    let pack = assets.pack();
    let entry = pack.find_by_name("mesh/a").unwrap();
    let blob = pack.blob(entry);
    let inside = vertices.as_ptr() as usize;
    let start = blob.as_ptr() as usize;
    assert!(
        inside >= start && inside < start + blob.len(),
        "the vertex slice is not a view into the blob"
    );
    // And nothing was made resident: only textures cost bytes here.
    assert_eq!(assets.resident_bytes(), 0);
}

#[test]
fn a_texture_is_loading_until_it_is_pumped_and_then_holds_every_level() {
    let mut assets = assets();
    let handle = assets.load::<asset::Texture>("tex/a").unwrap();
    assert_eq!(handle.state(), LoadState::Loading);
    assert!(assets.texture(&handle).is_none());

    assets.pump_until_idle();
    assert_eq!(handle.state(), LoadState::Ready);

    let data = assets.texture(&handle).unwrap();
    assert_eq!(data.format, TextureFormat::Bc7Srgb);
    assert_eq!(data.extent, (16, 16));
    assert_eq!(data.levels.len(), full_mip_count(16, 16) as usize);
    for (level, bytes) in data.levels.iter().enumerate() {
        let (w, h) = level_extent(16, 16, level as u32);
        assert_eq!(
            bytes.len() as u64,
            TextureFormat::Bc7Srgb.level_bytes(w, h),
            "level {level}"
        );
    }
    assert_eq!(assets.resident_bytes(), data.bytes());
}

#[test]
fn loading_one_name_twice_is_one_asset_and_two_references() {
    let mut assets = assets();
    let first = assets.load::<asset::Texture>("tex/a").unwrap();
    let second = assets.load::<asset::Texture>("tex/a").unwrap();
    assert_eq!(first, second);
    assert_eq!(assets.tracked(), 1, "one slot, not two");

    assets.pump_until_idle();
    // Both see the same state through the same slot.
    assert!(first.ready() && second.ready());

    // One reference released is not an eviction; the other still holds it.
    drop(first);
    assert_eq!(assets.evict_unreferenced(), 0);
    assert!(assets.texture(&second).is_some());

    drop(second);
    assert_eq!(assets.evict_unreferenced(), 1);
    assert_eq!(assets.resident_bytes(), 0);
    assert_eq!(assets.tracked(), 0);
}

#[test]
fn a_load_that_lands_after_its_eviction_is_dropped_rather_than_leaked() {
    // The one order nothing else here exercises (§6 M81): the handle is
    // released and the slot evicted while the decompression is still in flight,
    // so the payload arrives with nowhere to belong. Inserted anyway it is
    // unreachable — every reader goes through a slot — and invisible to every
    // later eviction, which walks slots too, so it is resident for the life of
    // the process.
    let mut assets = assets();
    let handle = assets.load::<asset::Texture>("tex/a").unwrap();
    drop(handle);
    assert_eq!(
        assets.evict_unreferenced(),
        1,
        "the slot went while loading"
    );
    assets.pump_until_idle();
    assert_eq!(assets.resident_bytes(), 0, "the late payload was dropped");
    assert_eq!(assets.tracked(), 0);
}

#[test]
fn a_handles_type_is_checked_against_the_pack_once_at_load() {
    let mut assets = assets();
    let err = assets.load::<asset::Mesh>("tex/a").unwrap_err().to_string();
    assert!(
        err.contains("is a Texture, and the handle asks for a Mesh"),
        "got: {err}"
    );

    let err = assets
        .load::<asset::Mesh>("mesh/nope")
        .unwrap_err()
        .to_string();
    assert!(err.contains("no asset named `mesh/nope`"), "got: {err}");
}

#[test]
fn an_absent_cross_reference_is_none_rather_than_an_error() {
    let mut assets = assets();
    // A material with no normal map spells it `AssetId::NONE`, which the pack
    // refuses to contain — so following it must not be a failed lookup.
    let absent = assets.load_id::<asset::Texture>(AssetId::NONE).unwrap();
    assert!(absent.is_none());

    // A real id resolves, and a mesh's material is reached the same way.
    let mesh = assets.load::<asset::Mesh>("mesh/a").unwrap();
    let material_id = assets.mesh(&mesh).unwrap().header.material;
    let material = assets
        .load_id::<asset::Material>(material_id)
        .unwrap()
        .expect("the mesh names a material that is in the pack");
    assert_eq!(material.id(), AssetId::of("mat/a"));
}

#[test]
fn every_texture_in_a_batch_lands_whatever_order_the_workers_finished_in() {
    let mut assets = assets();
    let handles: Vec<_> = ["tex/a", "tex/b", "tex/c"]
        .iter()
        .map(|name| assets.load::<asset::Texture>(name).unwrap())
        .collect();
    assets.pump_until_idle();

    // Distinct seeds, so a payload delivered to the wrong slot is visible
    // rather than merely possible.
    for (index, handle) in handles.iter().enumerate() {
        let data = assets.texture(handle).unwrap();
        let first = data.levels[0][0];
        assert_eq!(
            first, index as u8,
            "tex {index} got another texture's bytes"
        );
    }
}

#[test]
fn a_corrupt_texture_fails_its_handle_rather_than_the_registry() {
    let mut writer = PackWriter::new();
    writer
        .add("tex/good", AssetKind::Texture, 0, texture_blob(1, 8, 8))
        .unwrap();
    // A blob of the right kind that is not a container at all.
    writer
        .add("tex/bad", AssetKind::Texture, 1, vec![0u8; 96])
        .unwrap();
    let mut assets = Assets::new(gg_assets::Pack::from_bytes(&writer.finish().unwrap()).unwrap());

    let good = assets.load::<asset::Texture>("tex/good").unwrap();
    let bad = assets.load::<asset::Texture>("tex/bad").unwrap();
    assets.pump_until_idle();

    assert_eq!(bad.state(), LoadState::Failed);
    assert!(assets.texture(&bad).is_none());
    // The failure is one asset's, not the batch's.
    assert_eq!(good.state(), LoadState::Ready);
    assert!(assets.texture(&good).is_some());
}

#[test]
fn unloading_keeps_the_handle_and_reloading_fills_it_again() {
    let mut assets = assets();
    let handle = assets.load::<asset::Texture>("tex/a").unwrap();
    assets.pump_until_idle();
    let before = assets.texture(&handle).unwrap().levels[0].clone();

    // §4.6 watch mode: the residency layer drops the payload and asks for it
    // back, and the game's handle never learns anything happened.
    assets.unload(handle.id());
    assert_eq!(handle.state(), LoadState::Unloaded);
    assert!(assets.texture(&handle).is_none());

    assert_eq!(assets.reload_unloaded(), 1);
    assets.pump_until_idle();
    assert_eq!(handle.state(), LoadState::Ready);
    assert_eq!(assets.texture(&handle).unwrap().levels[0], before);
}
