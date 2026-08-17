//! `ggc`'s own gates (§6 M9): what a glTF compiles to, that two clean runs over
//! identical inputs are bit-identical, and that a warm build reuses rather than
//! recompiles.
//!
//! The source is written by the test rather than checked in. A fixture glTF in
//! the tree would be a second thing to keep in step with the importer, and this
//! one is small enough to read: one triangle, one material, one node, with the
//! buffer as a `data:` URI so nothing here is binary.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use gg_assets::{AssetId, AssetKind, Clip, Material, Mesh, Pack, Scene, Texture, TextureFormat};
use image::ImageEncoder;

/// Three positions and three `u16` indices, base64. Positions are (0,0,0),
/// (1,0,0), (0,1,0) — a triangle in the xy plane, so its flat normal is +z and
/// the generated-normal path has an answer worth asserting.
const BUFFER: &str = "AAAAAAAAAAAAAAAAAACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAABAAIA";

/// A glTF 2.0 document with no NORMAL and no TEXCOORD, so the importer's two
/// optional-attribute paths both run. `translate` moves the single node, which
/// is how a test asks for a *different* source with the same shape.
fn document(translate: f64) -> String {
    format!(
        r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0] }}],
  "nodes": [{{ "mesh": 0, "translation": [{translate}, 0.0, 0.0] }}],
  "meshes": [{{ "primitives": [{{ "attributes": {{ "POSITION": 0 }}, "indices": 1, "material": 0 }}] }}],
  "materials": [{{
    "pbrMetallicRoughness": {{ "baseColorFactor": [0.25, 0.5, 0.75, 1.0], "roughnessFactor": 0.5 }},
    "doubleSided": true
  }}],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
       "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0] }},
    {{ "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }}
  ],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": 0, "byteLength": 36 }},
    {{ "buffer": 0, "byteOffset": 36, "byteLength": 6 }}
  ],
  "buffers": [{{ "byteLength": 42, "uri": "data:application/octet-stream;base64,{BUFFER}" }}]
}}"#
    )
}

/// The same document with one image, referenced by all four material slots.
///
/// That is the ORM case, and it is the common one: glTF authors routinely point
/// occlusion, roughness and metalness at three channels of a single PNG. One
/// image, four assets, three formats.
fn textured_document() -> String {
    format!(
        r#"{{
  "asset": {{ "version": "2.0" }},
  "scene": 0,
  "scenes": [{{ "nodes": [0] }}],
  "nodes": [{{ "mesh": 0 }}],
  "meshes": [{{ "primitives": [{{ "attributes": {{ "POSITION": 0 }}, "indices": 1, "material": 0 }}] }}],
  "materials": [{{
    "pbrMetallicRoughness": {{
      "baseColorTexture": {{ "index": 0 }},
      "metallicRoughnessTexture": {{ "index": 0 }}
    }},
    "normalTexture": {{ "index": 0 }},
    "occlusionTexture": {{ "index": 0 }},
    "doubleSided": true
  }}],
  "images": [{{ "uri": "checker.png" }}],
  "textures": [{{ "source": 0 }}],
  "accessors": [
    {{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
       "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0] }},
    {{ "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" }}
  ],
  "bufferViews": [
    {{ "buffer": 0, "byteOffset": 0, "byteLength": 36 }},
    {{ "buffer": 0, "byteOffset": 36, "byteLength": 6 }}
  ],
  "buffers": [{{ "byteLength": 42, "uri": "data:application/octet-stream;base64,{BUFFER}" }}]
}}"#
    )
}

/// An 8x8 PNG with something in every channel, written beside its glTF.
///
/// A file rather than a `data:` URI on purpose: it puts the importer's URI path
/// and the source hash's "everything this document references" walk under test,
/// which is what makes a texture edit invalidate the model that uses it.
fn write_png(path: &Path, seed: u8) {
    let mut rgba = Vec::with_capacity(8 * 8 * 4);
    for y in 0..8u8 {
        for x in 0..8u8 {
            let lit = (x / 2 + y / 2) % 2 == 0;
            rgba.extend_from_slice(&[
                if lit { 220 } else { 30 },
                x.wrapping_mul(31).wrapping_add(seed),
                y.wrapping_mul(17),
                255,
            ]);
        }
    }
    let file = std::fs::File::create(path).unwrap();
    image::codecs::png::PngEncoder::new(file)
        .write_image(&rgba, 8, 8, image::ExtendedColorType::Rgba8)
        .unwrap();
}

/// A scratch directory of its own, so tests can run in parallel processes.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("ggc-tests").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("src")).unwrap();
    dir
}

fn write_source(dir: &Path, relative: &str, translate: f64) {
    let path = dir.join("src").join(relative);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, document(translate)).unwrap();
}

#[test]
fn a_gltf_compiles_to_a_mesh_a_material_and_a_scene() {
    let dir = scratch("compiles");
    write_source(&dir, "shapes/tri.gltf", 0.0);
    let out = dir.join("assets.ggpack");

    let stats = ggc::build::build(&dir.join("src"), &out).unwrap();
    assert_eq!(stats.compiled, 1);
    assert_eq!(stats.reused, 0);
    assert_eq!(stats.assets, 3);
    assert_eq!(stats.triangles, 1);

    let pack = Pack::open(&out).unwrap();
    // The name is built from the source's path under the root and glTF's own
    // indices — never from a glTF `name`, which is optional and duplicated.
    let entry = pack.find_by_name("shapes/tri/mesh/0.0").unwrap();
    assert_eq!(entry.kind(), Some(AssetKind::Mesh));

    let mesh = Mesh::read(pack.blob(entry)).unwrap();
    assert_eq!(mesh.vertices().len(), 3);
    assert_eq!(mesh.indices(), &[0, 1, 2]);
    assert_eq!(mesh.header.bounds_min, [0.0; 3]);
    assert_eq!(mesh.header.bounds_max, [1.0, 1.0, 0.0]);
    // glTF says a primitive without NORMAL is flat-shaded, and this triangle
    // faces +z. A zero normal here would light it as facing nowhere.
    assert_eq!(mesh.vertices()[0].normal, [0.0, 0.0, 1.0]);
    assert_eq!(mesh.header.material, AssetId::of("shapes/tri/material/0"));

    let scene = pack.find_by_name("shapes/tri/scene").unwrap();
    let scene = Scene::read(pack.blob(scene)).unwrap();
    assert_eq!(scene.nodes().len(), 1);
    assert_eq!(scene.nodes()[0].mesh, AssetId::of("shapes/tri/mesh/0.0"));
}

/// Write the textured document and its PNG into `dir/src/<name>/`.
fn write_textured(dir: &Path, name: &str, seed: u8) {
    let at = dir.join("src").join(name);
    std::fs::create_dir_all(&at).unwrap();
    std::fs::write(at.join("model.gltf"), textured_document()).unwrap();
    write_png(&at.join("checker.png"), seed);
}

#[test]
fn one_image_in_four_slots_compiles_to_four_assets_in_three_formats() {
    let dir = scratch("textures");
    write_textured(&dir, "town", 0);
    let out = dir.join("assets.ggpack");

    let stats = ggc::build::build(&dir.join("src"), &out).unwrap();
    assert_eq!(stats.textures, 4, "one image, four slots");
    assert_eq!(stats.assets, 3 + 4);

    let pack = Pack::open(&out).unwrap();
    // The name carries the role because the slot, not the file, decides what
    // the texels mean — the same PNG is a colour here and a mask there.
    for (role, format) in [
        ("base_color", TextureFormat::Bc7Srgb),
        ("normal", TextureFormat::Bc5Unorm),
        ("metallic_roughness", TextureFormat::Bc5Unorm),
        ("occlusion", TextureFormat::Bc4Unorm),
    ] {
        let name = format!("town/model/texture/0/{role}");
        let entry = pack.find_by_name(&name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(entry.kind(), Some(AssetKind::Texture));

        let blob = pack.blob(entry);
        // The blob is a KTX2 file, not a private layout wearing the name.
        assert_eq!(&blob[..12], &gg_assets::texture::IDENTIFIER, "{name}");
        let texture = Texture::read(blob).unwrap();
        assert_eq!(texture.format, format, "{name}");
        assert_eq!((texture.width, texture.height), (8, 8));
        assert_eq!(texture.level_count(), 4, "8,4,2,1");
        // 8x8 is four blocks; 4x4, 2x2 and 1x1 are one each, since a partial
        // block at the edge is still a whole block in the file.
        assert_eq!(
            texture.decompressed_bytes(),
            7 * u64::from(format.block_bytes()),
            "{name}"
        );
        for level in 0..texture.level_count() {
            assert!(
                texture.level(level).unwrap().is_ok(),
                "{name} level {level}"
            );
        }
    }

    // sRGB is a property of the base colour asset and of nothing else, decided
    // by the slot at import (§4.6) — the four assets share one source image.
    let colour = pack
        .find_by_name("town/model/texture/0/base_color")
        .unwrap();
    let mask = pack.find_by_name("town/model/texture/0/occlusion").unwrap();
    assert!(Texture::read(pack.blob(colour)).unwrap().format.is_srgb());
    assert!(!Texture::read(pack.blob(mask)).unwrap().format.is_srgb());
    assert_ne!(pack.blob(colour), pack.blob(mask));

    // And the material points at all four by id, which is what makes the names
    // above load-bearing rather than decorative.
    let material = pack.find_by_name("town/model/material/0").unwrap();
    let material = Material::read(pack.blob(material)).unwrap();
    assert_eq!(
        material.base_color_texture,
        AssetId::of("town/model/texture/0/base_color")
    );
    assert_eq!(
        material.normal_texture,
        AssetId::of("town/model/texture/0/normal")
    );
    assert_eq!(
        material.metallic_roughness_texture,
        AssetId::of("town/model/texture/0/metallic_roughness")
    );
    // The document says `doubleSided`, so the pack must too (§6 M64): the flag
    // crosses a serialization boundary to reach the one thing that reads it, and
    // its failure mode is silent — a two-sided surface shaded as a solid looks
    // like a *material* bug, not a missing bit.
    assert_ne!(
        material.flags & gg_assets::material::flags::DOUBLE_SIDED,
        0,
        "the pack dropped doubleSided"
    );
    assert_eq!(
        material.occlusion_texture,
        AssetId::of("town/model/texture/0/occlusion")
    );
}

#[test]
fn editing_a_texture_recompiles_the_model_that_references_it() {
    // The source hash covers everything the document names, not just the
    // document — otherwise a texture edit would be invisible to the cache and
    // the pack would keep serving the old texels forever.
    let dir = scratch("texture-edit");
    write_textured(&dir, "town", 0);
    let out = dir.join("assets.ggpack");

    assert_eq!(
        ggc::build::build(&dir.join("src"), &out).unwrap().compiled,
        1
    );
    assert_eq!(ggc::build::build(&dir.join("src"), &out).unwrap().reused, 1);

    write_png(&dir.join("src/town/checker.png"), 40);
    let after = ggc::build::build(&dir.join("src"), &out).unwrap();
    assert_eq!((after.compiled, after.reused), (1, 0));
}

#[test]
fn a_textured_pack_is_bit_identical_across_two_clean_runs() {
    // Byte-reproducibility (§4.6) has to survive the codec: the BC encoder, the
    // mip filter and zstd are all in this path, and any one of them being
    // schedule- or host-dependent would show here.
    let dir = scratch("texture-reproducible");
    write_textured(&dir, "town", 0);
    write_textured(&dir, "field", 7);

    let first = dir.join("first.ggpack");
    let second = dir.join("second.ggpack");
    ggc::build::build(&dir.join("src"), &first).unwrap();
    ggc::build::build(&dir.join("src"), &second).unwrap();
    assert_eq!(
        std::fs::read(&first).unwrap(),
        std::fs::read(&second).unwrap()
    );
}

#[test]
fn two_clean_runs_over_identical_inputs_produce_bit_identical_packs() {
    // §6 M9's exit row, and the property a content-hashed cache rests on: if
    // the bytes moved between runs, the incremental build below would be
    // rebuilding the world and calling it a cache.
    let dir = scratch("reproducible");
    write_source(&dir, "b/second.gltf", 2.0);
    write_source(&dir, "a/first.gltf", 1.0);
    write_source(&dir, "a/nested/third.gltf", 3.0);

    let first = dir.join("first.ggpack");
    let second = dir.join("second.ggpack");
    ggc::build::build(&dir.join("src"), &first).unwrap();
    ggc::build::build(&dir.join("src"), &second).unwrap();

    assert_eq!(
        std::fs::read(&first).unwrap(),
        std::fs::read(&second).unwrap()
    );
}

#[test]
fn a_warm_build_reuses_what_did_not_change_and_recompiles_what_did() {
    let dir = scratch("incremental");
    write_source(&dir, "a.gltf", 1.0);
    write_source(&dir, "b.gltf", 2.0);
    let out = dir.join("assets.ggpack");

    let cold = ggc::build::build(&dir.join("src"), &out).unwrap();
    assert_eq!((cold.compiled, cold.reused), (2, 0));

    let warm = ggc::build::build(&dir.join("src"), &out).unwrap();
    assert_eq!((warm.compiled, warm.reused), (0, 2));
    // The reused blobs are the ones that shipped, so a warm build's output is
    // the cold build's output byte for byte.
    let warm_bytes = std::fs::read(&out).unwrap();

    write_source(&dir, "b.gltf", 9.0);
    let partial = ggc::build::build(&dir.join("src"), &out).unwrap();
    assert_eq!((partial.compiled, partial.reused), (1, 1));
    assert_ne!(std::fs::read(&out).unwrap(), warm_bytes);
}

/// A 16-bit stereo `.wav`, written byte for byte beside its glTF neighbours —
/// same reason the glTF above is a string literal and not a fixture.
fn write_wav(dir: &Path, relative: &str, rate: u32, interleaved: &[i16]) {
    let data: Vec<u8> = interleaved.iter().flat_map(|s| s.to_le_bytes()).collect();
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&1u16.to_le_bytes()); // PCM
    fmt.extend_from_slice(&2u16.to_le_bytes()); // stereo
    fmt.extend_from_slice(&rate.to_le_bytes());
    fmt.extend_from_slice(&(rate * 4).to_le_bytes());
    fmt.extend_from_slice(&4u16.to_le_bytes());
    fmt.extend_from_slice(&16u16.to_le_bytes());

    let mut body = b"WAVE".to_vec();
    for (id, payload) in [(b"fmt ", &fmt), (b"data", &data)] {
        body.extend_from_slice(id);
        body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        body.extend_from_slice(payload);
    }
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);

    let path = dir.join("src").join(relative);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, out).unwrap();
}

#[test]
fn a_wav_compiles_to_one_clip_named_by_its_stem() {
    let dir = scratch("clip");
    write_wav(&dir, "sfx/pickup.wav", 22_050, &[100, 300, -1000, 0, 7, 8]);
    let out = dir.join("assets.ggpack");

    let stats = ggc::build::build(&dir.join("src"), &out).unwrap();
    assert_eq!((stats.compiled, stats.assets, stats.clips), (1, 1, 1));

    let pack = Pack::open(&out).unwrap();
    // The stem alone, with no suffix: a game names the path it put the file at.
    let entry = pack.find_by_name("sfx/pickup").unwrap();
    assert_eq!(entry.kind(), Some(AssetKind::Clip));
    let clip = Clip::read(pack.blob(entry)).unwrap();
    // Unresampled, and the stereo pairs averaged as integers — (7+8)/2 truncates.
    assert_eq!(clip.rate(), 22_050);
    assert_eq!(clip.samples(), [200, -500, 7]);
}

#[test]
fn a_pack_with_a_clip_is_bit_identical_across_two_clean_runs() {
    // §4.6 through the audio path: the downmix and the depth conversion are the
    // two places a host could have disagreed, and neither is allowed to.
    let dir = scratch("clip-reproducible");
    let samples: Vec<i16> = (0..2_000).map(|i| (i * 37 - 20_000) as i16).collect();
    write_wav(&dir, "sfx/hum.wav", 44_100, &samples);
    write_wav(&dir, "music/theme.wav", 48_000, &samples[..100]);

    let first = dir.join("first.ggpack");
    let second = dir.join("second.ggpack");
    ggc::build::build(&dir.join("src"), &first).unwrap();
    ggc::build::build(&dir.join("src"), &second).unwrap();
    assert_eq!(
        std::fs::read(&first).unwrap(),
        std::fs::read(&second).unwrap()
    );
}

#[test]
fn a_warm_build_reuses_a_clip_rather_than_recompiling_it() {
    // What `import::owns` recognising a bare `{stem}` buys. Before it, a source
    // whose only asset carries the stem itself was owned by nobody, so the cache
    // found nothing to copy and rebuilt it on every run.
    let dir = scratch("clip-incremental");
    write_wav(&dir, "sfx/pickup.wav", 22_050, &[1, 2, 3, 4]);
    let out = dir.join("assets.ggpack");

    let cold = ggc::build::build(&dir.join("src"), &out).unwrap();
    assert_eq!((cold.compiled, cold.reused), (1, 0));
    let cold_bytes = std::fs::read(&out).unwrap();

    let warm = ggc::build::build(&dir.join("src"), &out).unwrap();
    assert_eq!((warm.compiled, warm.reused), (0, 1));
    assert_eq!(std::fs::read(&out).unwrap(), cold_bytes);

    write_wav(&dir, "sfx/pickup.wav", 22_050, &[9, 9, 9, 9]);
    let after = ggc::build::build(&dir.join("src"), &out).unwrap();
    assert_eq!((after.compiled, after.reused), (1, 0));
    assert_ne!(std::fs::read(&out).unwrap(), cold_bytes);
}

#[test]
fn a_wav_that_is_not_pcm_fails_the_build_by_name() {
    // A compressed source must not become a clip full of noise: the pack is cast
    // in place, so bake is the last place this can be refused.
    let dir = scratch("clip-refused");
    let path = dir.join("src").join("bad.wav");
    let mut fmt = Vec::new();
    fmt.extend_from_slice(&0x0011u16.to_le_bytes()); // IMA ADPCM
    fmt.extend_from_slice(&1u16.to_le_bytes());
    fmt.extend_from_slice(&22_050u32.to_le_bytes());
    fmt.extend_from_slice(&11_025u32.to_le_bytes());
    fmt.extend_from_slice(&256u16.to_le_bytes());
    fmt.extend_from_slice(&4u16.to_le_bytes());
    let mut body = b"WAVE".to_vec();
    body.extend_from_slice(b"fmt ");
    body.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
    body.extend_from_slice(&fmt);
    body.extend_from_slice(b"data\x08\0\0\0");
    body.extend_from_slice(&[0; 8]);
    let mut out = b"RIFF".to_vec();
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    std::fs::write(&path, out).unwrap();

    let err = ggc::build::build(&dir.join("src"), &dir.join("assets.ggpack")).unwrap_err();
    let message = format!("{err:#}");
    assert!(message.contains("bad.wav"), "{message}");
    assert!(message.contains("17"), "{message}");
}

#[test]
fn bumping_the_importer_invalidates_every_asset_it_built() {
    // A compiler that changed what it emits and then reused blobs built by the
    // old rules would ship a pack no single version of `ggc` ever produced.
    // `IMPORT_VERSION` is in the hash so that cannot happen, and the reuse test
    // above only proves the *other* direction.
    let dir = scratch("import-version");
    write_source(&dir, "a.gltf", 1.0);
    let source = dir.join("src").join("a.gltf");

    let now = ggc::build::source_hash_at(&source, ggc::build::IMPORT_VERSION).unwrap();
    let next = ggc::build::source_hash_at(&source, ggc::build::IMPORT_VERSION + 1).unwrap();
    assert_ne!(now, next);
}

#[test]
fn one_sources_assets_share_one_hash_so_reuse_is_all_or_nothing() {
    // The source file is the unit that is hashed, so reusing one of its assets
    // and recompiling another would mean trusting a hash that covers both.
    let dir = scratch("one-hash");
    write_source(&dir, "a.gltf", 1.0);
    let out = dir.join("assets.ggpack");
    ggc::build::build(&dir.join("src"), &out).unwrap();

    let pack = Pack::open(&out).unwrap();
    let first = pack.entries()[0].source_hash;
    assert!(pack.entries().iter().all(|e| e.source_hash == first));
}

#[test]
fn a_source_tree_with_nothing_in_it_produces_an_empty_pack() {
    let dir = scratch("empty");
    let out = dir.join("assets.ggpack");
    let stats = ggc::build::build(&dir.join("src"), &out).unwrap();
    assert_eq!(stats.assets, 0);
    assert!(Pack::open(&out).unwrap().entries().is_empty());
}

#[test]
fn a_missing_source_directory_is_named_rather_than_treated_as_empty() {
    let dir = scratch("missing");
    let err = ggc::build::build(&dir.join("absent"), &dir.join("assets.ggpack")).unwrap_err();
    assert!(format!("{err}").contains("is not a directory"), "{err}");
}
