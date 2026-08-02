//! The pack format's own gates (§4.6): the round trip, byte-reproducibility,
//! and every way a pack is refused.
//!
//! The rejection half is the half worth writing. A loader that reads a good
//! pack is easy; §6 M9 asks for a version mismatch that is "rejected at load
//! with a named error rather than a garbage read", and the only way to know
//! that is to hand it bad packs and read what it says.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_assets::pack::{Entry, Header};
use gg_assets::{AssetId, AssetKind, FORMAT_VERSION, MAGIC, Pack, PackError, PackWriter};

/// Three assets with distinguishable blobs, added in a fixed order.
fn sample() -> Vec<u8> {
    let mut writer = PackWriter::new();
    writer
        .add("meshes/cube", AssetKind::Mesh, 0xc0be, vec![1; 3])
        .unwrap();
    writer
        .add("textures/albedo", AssetKind::Texture, 0xa1be, vec![2; 17])
        .unwrap();
    writer
        .add("scenes/main", AssetKind::Scene, 0x5cee, vec![3; 64])
        .unwrap();
    writer.finish().unwrap()
}

#[test]
fn a_pack_round_trips_through_its_own_writer() {
    let bytes = sample();
    let pack = Pack::from_bytes(&bytes).unwrap();
    assert_eq!(pack.entries().len(), 3);

    let cube = pack.find_by_name("meshes/cube").unwrap();
    assert_eq!(pack.name(cube), "meshes/cube");
    assert_eq!(cube.kind(), Some(AssetKind::Mesh));
    assert_eq!(cube.source_hash, 0xc0be);
    assert_eq!(pack.blob(cube), &[1; 3]);

    let albedo = pack.find_by_name("textures/albedo").unwrap();
    assert_eq!(albedo.kind(), Some(AssetKind::Texture));
    assert_eq!(pack.blob(albedo), &[2; 17]);

    let scene = pack.find_by_name("scenes/main").unwrap();
    assert_eq!(pack.blob(scene), &[3; 64]);

    assert!(pack.find(AssetId::of("meshes/nothing")).is_none());
}

#[test]
fn every_section_and_blob_lands_16_aligned() {
    let bytes = sample();
    let pack = Pack::from_bytes(&bytes).unwrap();
    for entry in pack.entries() {
        assert!(
            entry.offset.is_multiple_of(16),
            "{} starts at {}",
            pack.name(entry),
            entry.offset
        );
    }
    // The 3-byte blob is followed by 13 bytes of padding, and they are zeros:
    // gaps are content too, or two runs that agree stop agreeing (§4.6).
    let cube = pack.find_by_name("meshes/cube").unwrap();
    let pad = (cube.offset + cube.len) as usize..(cube.offset + 16) as usize;
    assert!(bytes[pad].iter().all(|&b| b == 0));
}

#[test]
fn the_bytes_do_not_depend_on_the_order_assets_were_added_in() {
    // §4.6's byte-reproducibility, at the level this crate can prove it: `ggc`
    // may parallelize, and the order its threads finish in is exactly what
    // reaches `add`. If that reached the file, a content-hashed cache would be
    // rebuilding the world and calling it a cache.
    let forwards = sample();

    let mut backwards = PackWriter::new();
    backwards
        .add("scenes/main", AssetKind::Scene, 0x5cee, vec![3; 64])
        .unwrap();
    backwards
        .add("textures/albedo", AssetKind::Texture, 0xa1be, vec![2; 17])
        .unwrap();
    backwards
        .add("meshes/cube", AssetKind::Mesh, 0xc0be, vec![1; 3])
        .unwrap();

    assert_eq!(forwards, backwards.finish().unwrap());
}

#[test]
fn an_empty_pack_is_legal_and_reads_back_empty() {
    let bytes = PackWriter::new().finish().unwrap();
    let pack = Pack::from_bytes(&bytes).unwrap();
    assert!(pack.entries().is_empty());
    assert!(pack.find_by_name("anything").is_none());
}

#[test]
fn the_same_name_twice_is_named_rather_than_shadowed() {
    let mut writer = PackWriter::new();
    writer.add("a", AssetKind::Mesh, 0, vec![1]).unwrap();
    writer.add("a", AssetKind::Mesh, 0, vec![2]).unwrap();
    let err = writer.finish().unwrap_err();
    assert!(
        format!("{err}").contains("a and a both have id"),
        "unhelpful: {err}"
    );
}

#[test]
fn a_file_that_is_not_a_pack_says_so_before_reading_anything() {
    let err = Pack::from_bytes(&b"\x89PNG".repeat(32)).unwrap_err();
    assert!(matches!(err, PackError::NotAPack { .. }), "{err}");
    assert!(format!("{err}").contains("a pack starts"));

    let err = Pack::from_bytes(b"short").unwrap_err();
    assert!(matches!(err, PackError::TooShort { .. }), "{err}");
}

#[test]
fn a_future_version_is_refused_by_number() {
    // The exit criterion (§6 M9): rejected at load with a named error rather
    // than a garbage read. `magic` at 0 and `format_version` at 8 are frozen
    // precisely so this check can run against a layout we have never seen.
    let mut bytes = sample();
    bytes[8..12].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
    let err = Pack::from_bytes(&bytes).unwrap_err();
    assert!(matches!(err, PackError::Version { found } if found == FORMAT_VERSION + 1));
    assert!(format!("{err}").contains("this build reads version"));
}

#[test]
fn a_truncated_pack_is_truncation_rather_than_whatever_ran_off_the_end_first() {
    let bytes = sample();
    let err = Pack::from_bytes(&bytes[..bytes.len() - 16]).unwrap_err();
    assert!(matches!(err, PackError::Truncated { .. }), "{err}");
}

/// Patch the header of a good pack, then reload it.
fn with_header(mut edit: impl FnMut(&mut Header)) -> PackError {
    let mut bytes = sample();
    let mut header: Header = bytemuck::pod_read_unaligned(&bytes[..size_of::<Header>()]);
    edit(&mut header);
    bytes[..size_of::<Header>()].copy_from_slice(bytemuck::bytes_of(&header));
    Pack::from_bytes(&bytes).unwrap_err()
}

/// Patch entry `index` of a good pack, then reload it.
fn with_entry(index: usize, mut edit: impl FnMut(&mut Entry)) -> PackError {
    let mut bytes = sample();
    let header: Header = bytemuck::pod_read_unaligned(&bytes[..size_of::<Header>()]);
    let at = header.entries as usize + index * size_of::<Entry>();
    let mut entry: Entry = bytemuck::pod_read_unaligned(&bytes[at..at + size_of::<Entry>()]);
    edit(&mut entry);
    bytes[at..at + size_of::<Entry>()].copy_from_slice(bytemuck::bytes_of(&entry));
    Pack::from_bytes(&bytes).unwrap_err()
}

#[test]
fn a_section_that_runs_past_the_end_is_named() {
    let err = with_header(|h| h.entries = h.file_len - 16);
    assert!(matches!(err, PackError::OutOfBounds { .. }), "{err}");
    assert!(format!("{err}").contains("the entry table"));

    let err = with_header(|h| h.names_len = h.file_len);
    assert!(matches!(err, PackError::OutOfBounds { .. }), "{err}");
    assert!(format!("{err}").contains("the name table"));
}

#[test]
fn a_misaligned_section_is_refused_rather_than_cast() {
    let err = with_header(|h| h.entries += 8);
    assert!(matches!(err, PackError::Misaligned { .. }), "{err}");
}

#[test]
fn a_blob_that_leaves_the_file_is_named_by_its_asset() {
    let err = with_entry(0, |e| e.len = u64::MAX);
    assert!(matches!(err, PackError::OutOfBounds { .. }), "{err}");
    // The overflow arithmetic has to survive this: offset + u64::MAX wraps.
    let text = format!("{err}");
    assert!(
        text.contains("meshes/") || text.contains("textures/") || text.contains("scenes/"),
        "the message should name the asset, not the offset: {text}"
    );
}

#[test]
fn an_unknown_kind_is_refused_because_the_wrong_header_would_be_cast() {
    let err = with_entry(0, |e| e.kind = 99);
    assert!(
        matches!(err, PackError::UnknownKind { kind: 99, .. }),
        "{err}"
    );
}

#[test]
fn an_out_of_order_table_is_refused_because_lookup_rests_on_it() {
    let err = with_entry(1, |e| e.id = AssetId(1));
    assert!(matches!(err, PackError::Unsorted { .. }), "{err}");
}

#[test]
fn the_reserved_id_may_not_name_a_real_asset() {
    // `AssetId::NONE` is what an absent material or normal map is spelled as,
    // so an asset carrying it would answer a lookup that means "nothing".
    let err = with_entry(0, |e| e.id = AssetId::NONE);
    assert!(matches!(err, PackError::ReservedId { .. }), "{err}");
}

#[test]
fn a_name_slice_outside_the_name_table_is_refused() {
    let err = with_entry(0, |e| e.name_len = 4096);
    assert!(matches!(err, PackError::BadName { .. }), "{err}");
}

#[test]
fn a_pack_on_disk_maps_and_reads_the_same_as_one_in_memory() {
    // The mapped path is the shipping one and `from_bytes` is not it: the two
    // backings differ in how they come by their alignment, which is the single
    // thing every cast in the reader rests on.
    let dir = std::env::temp_dir().join("gg-assets-open");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sample.ggpack");
    std::fs::write(&path, sample()).unwrap();

    let pack = Pack::open(&path).unwrap();
    let cube = pack.find_by_name("meshes/cube").unwrap();
    assert_eq!(pack.blob(cube), &[1; 3]);
    assert_eq!(pack.entries().len(), 3);

    let err = Pack::open(&dir.join("absent.ggpack")).unwrap_err();
    assert!(matches!(err, PackError::Io { .. }), "{err}");
    assert!(format!("{err}").contains("absent.ggpack"));
}

#[test]
fn the_magic_and_the_version_never_move() {
    // Not a test of behaviour but of the promise the version check rests on:
    // a reader of any version finds these two at these two offsets.
    let bytes = sample();
    assert_eq!(&bytes[..8], &MAGIC);
    assert_eq!(&bytes[8..12], &FORMAT_VERSION.to_le_bytes());
}
