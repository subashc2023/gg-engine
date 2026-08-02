//! The KTX2 container: what it writes, what it refuses, and what the bytes are.
//!
//! Half of these assert *layout* rather than behaviour — the identifier, the
//! header's field offsets, the data format descriptor byte for byte. A container
//! whose reader and writer only agree with each other is a private format
//! wearing a standard's name, and the point of writing KTX2 by hand was that
//! `ktx validate` and RenderDoc can read it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_assets::texture::{
    self, IDENTIFIER, Texture, TextureError, TextureFormat, full_mip_count, level_extent,
};

/// A full mip chain of plausible-looking blocks. Not real BC7 — the container
/// never interprets a block, and a test that ran a codec would be testing one.
fn chain(format: TextureFormat, width: u32, height: u32) -> Vec<Vec<u8>> {
    (0..full_mip_count(width, height))
        .map(|level| {
            let (w, h) = level_extent(width, height, level);
            let len = format.level_bytes(w, h) as usize;
            (0..len)
                .map(|i| (i as u8).wrapping_mul(31).wrapping_add(level as u8))
                .collect()
        })
        .collect()
}

fn word(blob: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(blob[at..at + 4].try_into().unwrap())
}

fn long(blob: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(blob[at..at + 8].try_into().unwrap())
}

#[test]
fn a_texture_round_trips_every_level_of_its_chain() {
    let levels = chain(TextureFormat::Bc7Srgb, 64, 32);
    assert_eq!(levels.len(), 7); // 64,32,16,8,4,2,1
    let blob = texture::encode(TextureFormat::Bc7Srgb, 64, 32, &levels).unwrap();

    let read = Texture::read(&blob).unwrap();
    assert_eq!(read.format, TextureFormat::Bc7Srgb);
    assert_eq!((read.width, read.height), (64, 32));
    assert_eq!(read.level_count(), 7);
    for (level, expected) in levels.iter().enumerate() {
        let got = read.level(level as u32).unwrap().unwrap();
        assert_eq!(&got, expected, "level {level}");
    }
    assert!(read.level(7).is_none());
    assert_eq!(
        read.decompressed_bytes(),
        levels.iter().map(|l| l.len() as u64).sum::<u64>()
    );
}

#[test]
fn the_mip_tail_is_square_whatever_the_aspect_ratio() {
    // A 64x1 texture still has seven levels, and the six after the first are
    // 32x1 down to 1x1 — the axis that ran out clamps rather than ending the
    // chain, which is the rule the block count depends on.
    assert_eq!(full_mip_count(64, 1), 7);
    assert_eq!(level_extent(64, 1, 3), (8, 1));
    assert_eq!(level_extent(64, 1, 6), (1, 1));
    assert_eq!(level_extent(1, 1, 0), (1, 1));
    let levels = chain(TextureFormat::Bc4Unorm, 64, 1);
    let blob = texture::encode(TextureFormat::Bc4Unorm, 64, 1, &levels).unwrap();
    let read = Texture::read(&blob).unwrap();
    assert_eq!(read.extent(6), (1, 1));
    // One 4x4 block covers a 1x1 texture; BC4's block is eight bytes.
    assert_eq!(read.level(6).unwrap().unwrap().len(), 8);
}

#[test]
fn the_blob_is_a_ktx2_file_with_its_fields_where_the_spec_puts_them() {
    let levels = chain(TextureFormat::Bc5Unorm, 16, 16);
    let blob = texture::encode(TextureFormat::Bc5Unorm, 16, 16, &levels).unwrap();

    assert_eq!(&blob[..12], &IDENTIFIER);
    assert_eq!(word(&blob, 12), 141, "vkFormat: VK_FORMAT_BC5_UNORM_BLOCK");
    assert_eq!(word(&blob, 16), 1, "typeSize");
    assert_eq!(word(&blob, 20), 16, "pixelWidth");
    assert_eq!(word(&blob, 24), 16, "pixelHeight");
    assert_eq!(word(&blob, 28), 0, "pixelDepth: 2D");
    assert_eq!(word(&blob, 32), 0, "layerCount: not an array");
    assert_eq!(word(&blob, 36), 1, "faceCount: not a cube map");
    assert_eq!(word(&blob, 40), 5, "levelCount");
    assert_eq!(word(&blob, 44), 2, "supercompressionScheme: zstd");
    assert_eq!(long(&blob, 64), 0, "sgdByteOffset");
    assert_eq!(long(&blob, 72), 0, "sgdByteLength");

    // The four sections the index names must lie inside the file, in order, and
    // the DFD must start where the level index ends.
    let (dfd_at, dfd_len) = (word(&blob, 48), word(&blob, 52));
    let (kvd_at, kvd_len) = (word(&blob, 56), word(&blob, 60));
    assert_eq!(dfd_at as usize, 80 + 5 * 24);
    assert_eq!(kvd_at, dfd_at + dfd_len);
    assert!((kvd_at + kvd_len) as usize <= blob.len());
    // dfdTotalSize is the first word of the DFD and must agree with the index.
    assert_eq!(word(&blob, dfd_at as usize), dfd_len);
}

#[test]
fn level_data_is_stored_smallest_first_even_though_the_index_is_not() {
    let levels = chain(TextureFormat::Bc7Srgb, 32, 32);
    let blob = texture::encode(TextureFormat::Bc7Srgb, 32, 32, &levels).unwrap();
    let offsets: Vec<u64> = (0..levels.len())
        .map(|level| long(&blob, 80 + level * 24))
        .collect();

    // The index runs level 0 first; the data it points at runs the other way
    // (KTX 2.0 §3.9.1), so a reader can get the small mips from a prefix.
    assert!(
        offsets.windows(2).all(|w| w[0] > w[1]),
        "level offsets should descend with level: {offsets:?}"
    );
    // Nothing between the sections and nothing between the levels: with
    // supercompression there is no mip padding to insert.
    let lengths: Vec<u64> = (0..levels.len())
        .map(|level| long(&blob, 80 + level * 24 + 8))
        .collect();
    let smallest = levels.len() - 1;
    assert_eq!(
        offsets[smallest],
        (word(&blob, 56) + word(&blob, 60)) as u64,
        "the smallest level should start where the KVD ends"
    );
    assert_eq!(
        offsets[0] + lengths[0],
        blob.len() as u64,
        "level 0 should end at the end of the file"
    );
}

#[test]
fn the_data_format_descriptor_is_the_bytes_the_spec_asks_for() {
    // Hand-checked against KHR_DF §5 and KTX 2.0 §3.10. Written out as a
    // literal rather than recomputed: a test that derives the expectation the
    // same way the writer does proves the writer agrees with itself.
    let bc7 = texture::encode(
        TextureFormat::Bc7Srgb,
        4,
        4,
        &chain(TextureFormat::Bc7Srgb, 4, 4),
    )
    .unwrap();
    let at = word(&bc7, 48) as usize;
    let len = word(&bc7, 52) as usize;
    #[rustfmt::skip]
    let expected: [u8; 44] = [
        44, 0, 0, 0,            // dfdTotalSize
        0, 0, 0, 0,             // vendorId 0 (Khronos), descriptorType 0 (basic)
        2, 0, 40, 0,            // versionNumber 2, descriptorBlockSize 40
        134, 1, 2, 0,           // model BC7, primaries BT709, transfer sRGB, flags 0
        3, 3, 0, 0,             // texelBlockDimension: 4x4x1x1, stored as extent-1
        16, 0, 0, 0, 0, 0, 0, 0, // bytesPlane: a 16-byte block, one plane
        0, 0,                   // sample 0: bitOffset
        127, 0,                 // bitLength 128-1, channelType BC7_COLOR
        0, 0, 0, 0,             // samplePosition
        0, 0, 0, 0,             // sampleLower
        255, 255, 255, 255,     // sampleUpper
    ];
    assert_eq!(len, expected.len());
    assert_eq!(&bc7[at..at + len], &expected);

    // BC5 differs in four places and nowhere else: the model, the primaries and
    // transfer function (a measurement has neither), and a second sample.
    let bc5 = texture::encode(
        TextureFormat::Bc5Unorm,
        4,
        4,
        &chain(TextureFormat::Bc5Unorm, 4, 4),
    )
    .unwrap();
    let at = word(&bc5, 48) as usize;
    let block = &bc5[at..at + word(&bc5, 52) as usize];
    assert_eq!(word(block, 0), 60, "dfdTotalSize: 4 + 24 + two samples");
    assert_eq!(block[12], 132, "KHR_DF_MODEL_BC5");
    assert_eq!(block[13], 0, "primaries unspecified: not a colour");
    assert_eq!(block[14], 1, "KHR_DF_TRANSFER_LINEAR");
    // Samples start after the 4-byte total size and the 24-byte fixed part.
    assert_eq!(block[28 + 2], 63, "sample 0 is the red half-block");
    assert_eq!(block[28 + 3], 0, "KHR_DF_CHANNEL_BC5_RED");
    assert_eq!(u16::from_le_bytes([block[44], block[45]]), 64);
    assert_eq!(block[44 + 3], 1, "KHR_DF_CHANNEL_BC5_GREEN");
}

#[test]
fn srgb_is_recorded_in_the_format_and_in_the_descriptor_not_inferred_at_load() {
    assert!(TextureFormat::Bc7Srgb.is_srgb());
    assert!(!TextureFormat::Bc5Unorm.is_srgb());
    assert!(!TextureFormat::Bc4Unorm.is_srgb());
    assert_eq!(TextureFormat::Bc7Srgb.vk_format(), 146);
    assert_eq!(TextureFormat::Bc4Unorm.vk_format(), 139);
    assert_eq!(TextureFormat::Bc4Unorm.block_bytes(), 8);
}

#[test]
fn the_same_texels_encode_to_the_same_bytes() {
    // §4.6's byte-reproducibility reaches into the codec: zstd is deterministic
    // for a level and a library version, and `codec_version` is what makes the
    // second of those a cache key rather than a hope.
    let levels = chain(TextureFormat::Bc7Srgb, 16, 8);
    let once = texture::encode(TextureFormat::Bc7Srgb, 16, 8, &levels).unwrap();
    let twice = texture::encode(TextureFormat::Bc7Srgb, 16, 8, &levels).unwrap();
    assert_eq!(once, twice);
    assert_ne!(texture::codec_version(), 0);
}

#[test]
fn a_blob_that_is_not_ktx2_says_so_before_reading_anything() {
    let png = b"\x89PNG\r\n\x1a\n".repeat(16);
    assert!(matches!(
        Texture::read(&png).unwrap_err(),
        TextureError::NotKtx2
    ));

    let truncated = [0u8; 40];
    let err = Texture::read(&truncated).unwrap_err();
    assert!(
        matches!(err, TextureError::TooShort { len: 40, .. }),
        "{err}"
    );
}

#[test]
fn the_features_this_reader_does_not_implement_are_refused_by_name() {
    let levels = chain(TextureFormat::Bc7Srgb, 8, 8);
    let good = texture::encode(TextureFormat::Bc7Srgb, 8, 8, &levels).unwrap();

    // A cube map: six faces of texels this loader would upload as one.
    let mut cube = good.clone();
    cube[36..40].copy_from_slice(&6u32.to_le_bytes());
    let err = Texture::read(&cube).unwrap_err();
    assert!(
        matches!(
            err,
            TextureError::Unsupported {
                what: "faceCount",
                found: 6,
                ..
            }
        ),
        "{err}"
    );

    // Basis supercompression, whose levels are not zstd frames at all.
    let mut basis = good.clone();
    basis[44..48].copy_from_slice(&1u32.to_le_bytes());
    let err = Texture::read(&basis).unwrap_err();
    assert!(
        matches!(
            err,
            TextureError::Unsupported {
                what: "supercompressionScheme",
                ..
            }
        ),
        "{err}"
    );

    // An uncompressed format: real KTX2, and nothing a pack may contain.
    let mut rgba = good.clone();
    rgba[12..16].copy_from_slice(&37u32.to_le_bytes()); // VK_FORMAT_R8G8B8A8_UNORM
    assert!(matches!(
        Texture::read(&rgba).unwrap_err(),
        TextureError::UnknownFormat(37)
    ));
}

#[test]
fn a_level_that_lies_about_its_size_is_caught_before_it_is_decompressed() {
    let levels = chain(TextureFormat::Bc7Srgb, 8, 8);
    let good = texture::encode(TextureFormat::Bc7Srgb, 8, 8, &levels).unwrap();

    // uncompressedByteLength is what the staging ring would be sized from, so a
    // wrong one is a buffer overrun waiting for a loader that trusted it.
    let mut lying = good.clone();
    lying[80 + 16..80 + 24].copy_from_slice(&(1u64 << 40).to_le_bytes());
    let err = Texture::read(&lying).unwrap_err();
    assert!(
        matches!(
            err,
            TextureError::WrongLevelSize {
                level: 0,
                expected: 64,
                ..
            }
        ),
        "{err}"
    );

    // A hostile byteLength must report a span past the end, never wrap into one
    // that fits.
    let mut hostile = good.clone();
    hostile[80 + 8..80 + 16].copy_from_slice(&u64::MAX.to_le_bytes());
    let err = Texture::read(&hostile).unwrap_err();
    assert!(matches!(err, TextureError::OutOfBounds { .. }), "{err}");

    // More levels than a 8x8 texture has: the index would read past the header.
    let mut deep = good.clone();
    deep[40..44].copy_from_slice(&9u32.to_le_bytes());
    let err = Texture::read(&deep).unwrap_err();
    assert!(
        matches!(
            err,
            TextureError::TooManyLevels {
                levels: 9,
                most: 4,
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn the_writer_refuses_a_level_whose_texels_do_not_fill_it() {
    let mut levels = chain(TextureFormat::Bc7Srgb, 8, 8);
    levels[1].truncate(8);
    let err = texture::encode(TextureFormat::Bc7Srgb, 8, 8, &levels).unwrap_err();
    assert!(
        matches!(
            err,
            TextureError::LevelLength {
                level: 1,
                found: 8,
                expected: 16,
                ..
            }
        ),
        "{err}"
    );

    let err = texture::encode(TextureFormat::Bc7Srgb, 8, 8, &[]).unwrap_err();
    assert!(
        matches!(err, TextureError::Empty { levels: 0, .. }),
        "{err}"
    );

    let err = texture::encode(TextureFormat::Bc7Srgb, 0, 8, &levels).unwrap_err();
    assert!(matches!(err, TextureError::Empty { width: 0, .. }), "{err}");

    // A partial chain is legal — a texture may stop above 1x1 — but a chain
    // longer than the extent allows names levels that do not exist.
    let short = chain(TextureFormat::Bc7Srgb, 8, 8)[..2].to_vec();
    assert!(texture::encode(TextureFormat::Bc7Srgb, 8, 8, &short).is_ok());
    let too_many = vec![chain(TextureFormat::Bc7Srgb, 4, 4)[0].clone(); 5];
    let err = texture::encode(TextureFormat::Bc7Srgb, 4, 4, &too_many).unwrap_err();
    assert!(
        matches!(err, TextureError::TooManyLevels { most: 3, .. }),
        "{err}"
    );
}
