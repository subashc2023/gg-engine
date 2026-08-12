//! The texture blob: a complete KTX2 container, byte for byte (§4.6).
//!
//! A blob here is a file, not a private layout. That is the whole reason §2
//! names a standard container: a level can be written out and handed to
//! RenderDoc, `ktx validate` or any texture viewer without this crate present.
//! The pack is our container for *assets*; KTX2 is the industry's for *texels*,
//! and nesting them costs the 80-odd bytes of a header we would otherwise have
//! invented.
//!
//! ```text
//! 0                     a texture blob, in file order
//! +-----------------------------------------------------------------+
//! | identifier   12 bytes, «KTX 20»\r\n\x1A\n                        |
//! | header       36 bytes, format and dimensions                     |
//! | index        32 bytes, where the four sections below are         |
//! | level index  24 bytes per level, level 0 first                   |
//! +-----------------------------------------------------------------+
//! | DFD          the Khronos basic data format descriptor            |
//! | KVD          KTXorientation, KTXwriter                           |
//! +-----------------------------------------------------------------+
//! | level n-1    smallest mip first — see `LEVELS_ARE_STORED_SMALL`  |
//! | ...                                                              |
//! | level 0      each a zstd frame (supercompressionScheme 2)        |
//! +-----------------------------------------------------------------+
//! ```
//!
//! **Textures are the one asset that is not read in place.** Meshes and scenes
//! are cast out of the mapping and used as they lie; a level is a zstd frame and
//! has to be decompressed somewhere. That is not a wart — texels are copied to
//! the GPU through the staging ring (§4.3) regardless, so the decompression has
//! a buffer to land in that the mesh path does not.
//!
//! **The reader is strict, and deliberately narrower than KTX2.** It accepts the
//! subset [`encode`] writes — 2D, one face, no array layers, one of four block
//! formats, always zstd — and refuses everything else by name. A permissive
//! reader would be a second importer running at load time, on the ship, on the
//! player's machine; `ggc` is where a file gets interpreted.

use core::mem::size_of;

use thiserror::Error;

/// The 12 bytes every KTX2 file starts with (KTX 2.0 §3.1).
pub const IDENTIFIER: [u8; 12] = [
    0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

/// `supercompressionScheme` 2 — Zstandard (KTX 2.0 §3.7). Always, for every
/// texture we write, so there is one level-data path and not two.
const SUPERCOMPRESSION_ZSTD: u32 = 2;

/// Where the level index starts: identifier + header + index.
const LEVEL_INDEX_AT: usize = 80;

/// One level index record: `byteOffset`, `byteLength`, `uncompressedByteLength`.
const LEVEL_INDEX_BYTES: usize = 24;

/// Every block format here is 4x4.
const BLOCK_EXTENT: u32 = 4;

/// The largest extent a texture blob may declare, per axis.
///
/// A bound, not a preference: `pixelWidth`/`pixelHeight` are two words of an
/// 80-byte header, and everything downstream is sized from them *before* a
/// payload byte is read — [`TextureFormat::level_bytes`], the staging ring's
/// reservation (§4.3), and `zstd`'s output capacity, which reserves up front. A
/// ~900-byte file declaring 65536x65536 BC7 would otherwise name 4 GiB a level
/// and ~5.6 GiB a chain, and 2^31 square would reach `handle_alloc_error` — an
/// abort, on a worker thread, uncatchable. 16384 is `maxImageDimension2D` on
/// every target we ship to, so a larger texture could not be created even if the
/// file were honest; it caps one level at 256 MiB. Packs are build output today
/// and a mod tomorrow (§4.6), and this is the difference.
pub const MAX_DIMENSION: u32 = 16_384;

/// zstd compression level. Block-compressed texels are already near-random, so
/// the ratio curve is flat past single digits — 9 is where more time stops
/// buying bytes. Deterministic given the level and the linked zstd, which is
/// why [`codec_version`] exists.
const ZSTD_LEVEL: i32 = 9;

/// The writer identity in the KVD. A *constant*, not a build version string: a
/// field that moved between builds would make an identical source compile to a
/// different pack and take §4.6's byte-reproducibility with it.
const WRITER: &str = "ggc";

/// What a texture blob holds. The discriminant is Vulkan's format number, so
/// the loader hands it to the API rather than mapping it (§4.6 — sRGB and
/// linear are decided at import and never guessed at runtime).
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TextureFormat {
    /// BC7, sRGB-encoded. Base colour and nothing else — the only role whose
    /// texels are a colour rather than a measurement.
    Bc7Srgb = 146,
    /// BC5, two channels, linear. Tangent-space normals (x, y — z is
    /// reconstructed) and metallic-roughness (roughness, metalness).
    Bc5Unorm = 141,
    /// BC4, one channel, linear. Occlusion and other masks.
    Bc4Unorm = 139,
    /// BC6H, three channels, *unsigned half float*, linear. The environment and
    /// nothing else (§6 M27). The only format here whose texels may exceed one:
    /// a sky is the one measurement in a pack that is a radiance rather than a
    /// reflectance, and eight bits per channel cannot hold a sun.
    Bc6hUfloat = 143,
}

impl TextureFormat {
    const fn from_u32(raw: u32) -> Option<Self> {
        match raw {
            146 => Some(Self::Bc7Srgb),
            141 => Some(Self::Bc5Unorm),
            139 => Some(Self::Bc4Unorm),
            143 => Some(Self::Bc6hUfloat),
            _ => None,
        }
    }

    /// Vulkan's `VkFormat` number for this format.
    #[must_use]
    pub const fn vk_format(self) -> u32 {
        self as u32
    }

    /// Bytes one 4x4 block occupies.
    #[must_use]
    pub const fn block_bytes(self) -> u32 {
        match self {
            Self::Bc7Srgb | Self::Bc5Unorm | Self::Bc6hUfloat => 16,
            Self::Bc4Unorm => 8,
        }
    }

    /// Whether texels are sRGB-encoded. The sampler undoes it; nothing else may.
    #[must_use]
    pub const fn is_srgb(self) -> bool {
        matches!(self, Self::Bc7Srgb)
    }

    /// Bytes a whole level of this format occupies at `width` x `height`,
    /// saturating where [`Self::checked_level_bytes`] would give `None`. A
    /// saturated answer is one no level index can match, so a caller that
    /// skipped [`MAX_DIMENSION`] refuses rather than acting on a wrapped size.
    #[must_use]
    pub const fn level_bytes(self, width: u32, height: u32) -> u64 {
        match self.checked_level_bytes(width, height) {
            Some(bytes) => bytes,
            None => u64::MAX,
        }
    }

    /// [`Self::level_bytes`], or `None` when the product does not fit a `u64`:
    /// `blocks_across(u32::MAX)² * 16` is exactly 2^64, which panics where
    /// overflow checks are on and wraps to *zero* where they are not — a
    /// zero-byte level that then matches a hostile index (§4.6).
    #[must_use]
    pub const fn checked_level_bytes(self, width: u32, height: u32) -> Option<u64> {
        let across = blocks_across(width) as u64;
        let down = blocks_across(height) as u64;
        match across.checked_mul(down) {
            Some(blocks) => blocks.checked_mul(self.block_bytes() as u64),
            None => None,
        }
    }
}

/// Blocks needed to cover `texels` along one axis — a partial block at the edge
/// is a whole block in the file.
#[must_use]
pub const fn blocks_across(texels: u32) -> u32 {
    texels.div_ceil(BLOCK_EXTENT)
}

/// A level's extent: halved per level, floored at one texel. The mip tail is
/// 2x1 and 1x1 whatever the aspect ratio, which is why this clamps rather than
/// stopping when an axis runs out.
#[must_use]
pub const fn level_extent(width: u32, height: u32, level: u32) -> (u32, u32) {
    (
        if width >> level > 0 {
            width >> level
        } else {
            1
        },
        if height >> level > 0 {
            height >> level
        } else {
            1
        },
    )
}

/// Levels in a full chain down to 1x1.
#[must_use]
pub const fn full_mip_count(width: u32, height: u32) -> u32 {
    let longest = if width > height { width } else { height };
    32 - longest.leading_zeros()
}

/// The codec's identity, for the hash a warm rebuild keys on (§4.6).
///
/// Folded into `ggc`'s source hash rather than left to a human constant: zstd's
/// output is deterministic for a given library and level, so a dependency bump
/// silently re-authors every texture in the pack. A cache that did not notice
/// would serve the old bytes forever and `--check`, which builds twice with the
/// *same* library, would agree with itself and pass.
#[must_use]
pub fn codec_version() -> u32 {
    // Low 16 bits ours, high 16 zstd's — one number, two reasons to change.
    (zstd::zstd_safe::version_number() << 16) | u32::from(CONTAINER_VERSION)
}

/// Bumped when [`encode`] would lay the same texels out differently.
const CONTAINER_VERSION: u16 = 1;

/// Why a texture blob could not be read.
#[derive(Debug, Error)]
pub enum TextureError {
    /// Shorter than the fixed part of a KTX2 file.
    #[error("a texture blob of {len} bytes is shorter than a {need}-byte KTX2 header")]
    TooShort {
        /// The blob's length.
        len: usize,
        /// What was needed to read this far.
        need: usize,
    },
    /// The first twelve bytes are not KTX2's.
    #[error("the blob does not start with the KTX2 identifier — it is not a texture")]
    NotKtx2,
    /// A field names something the loader does not implement.
    #[error(
        "{what} is {found}, and this reader accepts only {accepted} — `ggc` writes the narrow \
         subset on purpose, so a wider file came from elsewhere"
    )]
    Unsupported {
        /// The field, named.
        what: &'static str,
        /// What the file said.
        found: u32,
        /// What is accepted.
        accepted: &'static str,
    },
    /// `vkFormat` is not one of the three block formats.
    #[error("vkFormat {0} is not one of the block formats a pack may contain (BC7/BC5/BC4)")]
    UnknownFormat(u32),
    /// A section or level runs past the end of the blob.
    #[error("{what} spans {offset}..{end} of a {len}-byte texture blob")]
    OutOfBounds {
        /// Which run, named.
        what: &'static str,
        /// Where it starts.
        offset: u64,
        /// Where it would end.
        end: u64,
        /// The blob's length.
        len: usize,
    },
    /// A level claims a decompressed size its format and extent contradict.
    #[error(
        "level {level} claims {found} decompressed bytes, but {width}x{height} of that format is \
         {expected}"
    )]
    WrongLevelSize {
        /// Which level.
        level: u32,
        /// What the level index said.
        found: u64,
        /// What the format and extent require.
        expected: u64,
        /// The level's width.
        width: u32,
        /// The level's height.
        height: u32,
    },
    /// An extent past [`MAX_DIMENSION`] — no image could be created at it, and
    /// unbounded it is an allocation sized by the file (§4.6).
    #[error(
        "{width}x{height} is past the {most}-texel bound a texture blob may declare — no image \
         could be created at it, and the size arithmetic below it is the file's to choose"
    )]
    DimensionTooLarge {
        /// What the file said, or what a caller passed.
        width: u32,
        /// The same, for the other axis.
        height: u32,
        /// The bound: [`MAX_DIMENSION`].
        most: u32,
    },
    /// The level index has more entries than the dimensions allow.
    #[error("{levels} levels is more than the {most} a {width}x{height} texture has")]
    TooManyLevels {
        /// What the file said.
        levels: u32,
        /// What a full chain would be.
        most: u32,
        /// The base width.
        width: u32,
        /// The base height.
        height: u32,
    },
    /// A level's zstd frame would not decompress.
    #[error("level {level} is supercompressed with zstd and did not decompress: {source}")]
    Decompress {
        /// Which level.
        level: u32,
        /// The underlying failure.
        source: std::io::Error,
    },
    /// A caller handed [`encode`] a level of the wrong length.
    #[error("level {level} is {found} bytes, but {width}x{height} of that format is {expected}")]
    LevelLength {
        /// Which level.
        level: u32,
        /// What was passed.
        found: usize,
        /// What the format and extent require.
        expected: u64,
        /// The level's width.
        width: u32,
        /// The level's height.
        height: u32,
    },
    /// A caller handed [`encode`] no levels, or a degenerate extent.
    #[error(
        "a texture needs at least one level and a non-zero extent; got {levels} at {width}x{height}"
    )]
    Empty {
        /// How many levels were passed.
        levels: usize,
        /// The base width.
        width: u32,
        /// The base height.
        height: u32,
    },
}

/// One level's place in the file, as the level index states it.
#[derive(Clone, Copy, Debug)]
struct LevelIndex {
    offset: u64,
    length: u64,
    uncompressed: u64,
}

/// A texture, borrowed out of a pack's mapping. Parsing is done and validated;
/// the texels are still compressed until [`Texture::level`] is called.
#[derive(Clone, Debug)]
pub struct Texture<'a> {
    /// The block format, decided at import.
    pub format: TextureFormat,
    /// Base width in texels.
    pub width: u32,
    /// Base height in texels.
    pub height: u32,
    levels: Vec<LevelIndex>,
    blob: &'a [u8],
}

impl<'a> Texture<'a> {
    /// Read a texture out of a blob, validating every offset and every field
    /// this reader will later act on.
    pub fn read(blob: &'a [u8]) -> Result<Self, TextureError> {
        let head = blob.get(..LEVEL_INDEX_AT).ok_or(TextureError::TooShort {
            len: blob.len(),
            need: LEVEL_INDEX_AT,
        })?;
        if head[..IDENTIFIER.len()] != IDENTIFIER {
            return Err(TextureError::NotKtx2);
        }
        let word =
            |at: usize| u32::from_le_bytes([head[at], head[at + 1], head[at + 2], head[at + 3]]);

        let format =
            TextureFormat::from_u32(word(12)).ok_or(TextureError::UnknownFormat(word(12)))?;
        // Every one of these is a whole feature of KTX2 that the loader has no
        // code for. Refusing by name beats reading a cube map as a 2D texture.
        for (what, found, want, accepted) in [
            ("typeSize", word(16), 1, "1 (block-compressed)"),
            ("pixelDepth", word(28), 0, "0 (2D)"),
            ("layerCount", word(32), 0, "0 (not an array)"),
            ("faceCount", word(36), 1, "1 (not a cube map)"),
            (
                "supercompressionScheme",
                word(44),
                SUPERCOMPRESSION_ZSTD,
                "2 (zstd)",
            ),
        ] {
            if found != want {
                return Err(TextureError::Unsupported {
                    what,
                    found,
                    accepted,
                });
            }
        }
        let (width, height, level_count) = (word(20), word(24), word(40));
        if width == 0 || height == 0 || level_count == 0 {
            return Err(TextureError::Empty {
                levels: level_count as usize,
                width,
                height,
            });
        }
        // Before `full_mip_count`, before the level index is sized, before any
        // arithmetic on the extent at all: every allocation below is a function
        // of these two words, and they came out of the file.
        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(TextureError::DimensionTooLarge {
                width,
                height,
                most: MAX_DIMENSION,
            });
        }
        let most = full_mip_count(width, height);
        if level_count > most {
            return Err(TextureError::TooManyLevels {
                levels: level_count,
                most,
                width,
                height,
            });
        }

        let index_end = LEVEL_INDEX_AT + level_count as usize * LEVEL_INDEX_BYTES;
        let index = blob
            .get(LEVEL_INDEX_AT..index_end)
            .ok_or(TextureError::TooShort {
                len: blob.len(),
                need: index_end,
            })?;
        let mut levels = Vec::with_capacity(level_count as usize);
        for (level, record) in index.chunks_exact(LEVEL_INDEX_BYTES).enumerate() {
            let long = |at: usize| {
                let mut bytes = [0u8; 8];
                bytes.copy_from_slice(&record[at..at + 8]);
                u64::from_le_bytes(bytes)
            };
            let entry = LevelIndex {
                offset: long(0),
                length: long(8),
                uncompressed: long(16),
            };
            let (w, h) = level_extent(width, height, level as u32);
            // Checked, though the bound above already puts every level three
            // orders under the overflow: the size arithmetic on this path has no
            // branch that panics in dev and wraps in dist.
            let expected =
                format
                    .checked_level_bytes(w, h)
                    .ok_or(TextureError::DimensionTooLarge {
                        width,
                        height,
                        most: MAX_DIMENSION,
                    })?;
            if entry.uncompressed != expected {
                return Err(TextureError::WrongLevelSize {
                    level: level as u32,
                    found: entry.uncompressed,
                    expected,
                    width: w,
                    height: h,
                });
            }
            // Saturating, then checked against the length: a hostile `length`
            // of u64::MAX must report a span past the file, never wrap into one
            // that fits inside it.
            let end = entry.offset.saturating_add(entry.length);
            if end > blob.len() as u64 {
                return Err(TextureError::OutOfBounds {
                    what: "a level's data",
                    offset: entry.offset,
                    end,
                    len: blob.len(),
                });
            }
            levels.push(entry);
        }

        Ok(Self {
            format,
            width,
            height,
            levels,
            blob,
        })
    }

    /// How many mip levels this texture carries.
    #[must_use]
    pub fn level_count(&self) -> u32 {
        self.levels.len() as u32
    }

    /// A level's extent in texels.
    #[must_use]
    pub fn extent(&self, level: u32) -> (u32, u32) {
        level_extent(self.width, self.height, level)
    }

    /// Decompress one level's texels, ready for the staging ring. `None` for a
    /// level this texture does not have.
    pub fn level(&self, level: u32) -> Option<Result<Vec<u8>, TextureError>> {
        let entry = *self.levels.get(level as usize)?;
        // Cannot fail: `read` checked this span against the blob's length.
        let frame = self
            .blob
            .get(entry.offset as usize..(entry.offset + entry.length) as usize)?;
        // `decompress` reserves `uncompressed` before it touches the payload, so
        // that number *is* an allocation the file asked for — bounded only
        // because `read` refused an extent past `MAX_DIMENSION` first.
        Some(
            zstd::bulk::decompress(frame, entry.uncompressed as usize)
                .map_err(|source| TextureError::Decompress { level, source }),
        )
    }

    /// Every level's texels, base first — what an upload walks.
    pub fn levels(&self) -> impl Iterator<Item = Result<Vec<u8>, TextureError>> + '_ {
        (0..self.level_count()).filter_map(|level| self.level(level))
    }

    /// Bytes the whole chain occupies once decompressed, which is what the
    /// staging ring must have room for.
    #[must_use]
    pub fn decompressed_bytes(&self) -> u64 {
        self.levels.iter().map(|l| l.uncompressed).sum()
    }

    /// The KTX2 file itself, for handing to a tool that speaks it.
    #[must_use]
    pub fn ktx2(&self) -> &'a [u8] {
        self.blob
    }
}

/// Lay out a texture blob for [`crate::PackWriter::add`].
///
/// `levels[0]` is the base and each subsequent level is half the extent,
/// already block-compressed to `format` — the codec is `ggc`'s, because this
/// crate is in the runtime's graph and an encoder there would ship a compressor
/// nothing calls.
pub fn encode(
    format: TextureFormat,
    width: u32,
    height: u32,
    levels: &[Vec<u8>],
) -> Result<Vec<u8>, TextureError> {
    if levels.is_empty() || width == 0 || height == 0 {
        return Err(TextureError::Empty {
            levels: levels.len(),
            width,
            height,
        });
    }
    // The writer refuses what the reader refuses, so `ggc` cannot author a pack
    // its own loader would reject at load time on a player's machine.
    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(TextureError::DimensionTooLarge {
            width,
            height,
            most: MAX_DIMENSION,
        });
    }
    let most = full_mip_count(width, height);
    if levels.len() as u32 > most {
        return Err(TextureError::TooManyLevels {
            levels: levels.len() as u32,
            most,
            width,
            height,
        });
    }

    let mut frames = Vec::with_capacity(levels.len());
    for (level, texels) in levels.iter().enumerate() {
        let (w, h) = level_extent(width, height, level as u32);
        let expected = format.level_bytes(w, h);
        if texels.len() as u64 != expected {
            return Err(TextureError::LevelLength {
                level: level as u32,
                found: texels.len(),
                expected,
                width: w,
                height: h,
            });
        }
        // Infallible in practice; an allocation failure inside zstd is the only
        // way out, and it arrives as an io::Error like a decompression failure.
        frames.push(zstd::bulk::compress(texels, ZSTD_LEVEL).map_err(|source| {
            TextureError::Decompress {
                level: level as u32,
                source,
            }
        })?);
    }

    let dfd = data_format_descriptor(format);
    let kvd = key_value_data();
    let index_end = LEVEL_INDEX_AT + levels.len() * LEVEL_INDEX_BYTES;
    let dfd_at = index_end;
    let kvd_at = dfd_at + dfd.len();
    let data_at = kvd_at + kvd.len();

    // LEVELS_ARE_STORED_SMALL: the level *index* runs level 0 first, but the
    // level *data* runs smallest first (KTX 2.0 §3.9.1). That ordering is the
    // container's one concession to streaming — a reader that wants a usable
    // picture from a prefix of the file gets the small mips without seeking —
    // and inverting it here is the whole of honouring it.
    let mut payload = Vec::new();
    let mut index = vec![[0u64; 3]; levels.len()];
    for (level, frame) in frames.iter().enumerate().rev() {
        let (w, h) = level_extent(width, height, level as u32);
        index[level] = [
            (data_at + payload.len()) as u64,
            frame.len() as u64,
            format.level_bytes(w, h),
        ];
        payload.extend_from_slice(frame);
    }

    let mut out = Vec::with_capacity(data_at + payload.len());
    out.extend_from_slice(&IDENTIFIER);
    for field in [
        format.vk_format(),
        1, // typeSize: one byte, the only value a block format may use
        width,
        height,
        0, // pixelDepth: 2D
        0, // layerCount: not an array
        1, // faceCount: not a cube map
        levels.len() as u32,
        SUPERCOMPRESSION_ZSTD,
        dfd_at as u32,
        dfd.len() as u32,
        kvd_at as u32,
        kvd.len() as u32,
    ] {
        out.extend_from_slice(&field.to_le_bytes());
    }
    // sgdByteOffset / sgdByteLength: zstd carries its parameters in the frame,
    // so supercompression global data exists only for Basis and stays zero.
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    for record in &index {
        for field in record {
            out.extend_from_slice(&field.to_le_bytes());
        }
    }
    out.extend_from_slice(&dfd);
    out.extend_from_slice(&kvd);
    out.extend_from_slice(&payload);
    debug_assert_eq!(out.len(), data_at + payload.len());
    Ok(out)
}

/// The Khronos basic data format descriptor (KHR_DF §5), which KTX2 requires
/// even though `vkFormat` already names the format: the DFD is what a reader
/// that does not know Vulkan's enum uses, and `ktx validate` checks the two
/// agree.
fn data_format_descriptor(format: TextureFormat) -> Vec<u8> {
    // (channel type, bit offset, bit length) per sample. Channel ids are the
    // per-model ones: BC7_COLOR = 0; BC5_RED = 0, BC5_GREEN = 1; BC4_DATA = 0;
    // BC6H_COLOR = 0, with KHR_DF_SAMPLE_DATATYPE_FLOAT (0x80) set in the high
    // nibble — the one format here whose decoded texels are floats, and the one
    // bit that says so to a reader not consulting `vkFormat`.
    let samples: &[(u8, u16, u8)] = match format {
        TextureFormat::Bc7Srgb => &[(0, 0, 127)],
        TextureFormat::Bc5Unorm => &[(0, 0, 63), (1, 64, 63)],
        TextureFormat::Bc4Unorm => &[(0, 0, 63)],
        TextureFormat::Bc6hUfloat => &[(0x80, 0, 127)],
    };
    // KHR_DF_MODEL_BC7 = 134, BC6H = 133, BC5 = 132, BC4 = 131.
    let color_model: u8 = match format {
        TextureFormat::Bc7Srgb => 134,
        TextureFormat::Bc6hUfloat => 133,
        TextureFormat::Bc5Unorm => 132,
        TextureFormat::Bc4Unorm => 131,
    };
    // The range one sample spans, as the spec's own bit patterns. Normalized
    // formats say 0..=u32::MAX; BC6H says 0.0..=1.0 *as float bits*, because a
    // float channel's bounds are read as floats and `u32::MAX` reinterpreted
    // that way is a NaN.
    let (lower, upper): (u32, u32) = match format {
        TextureFormat::Bc6hUfloat => (0, 0x3f80_0000),
        _ => (0, u32::MAX),
    };
    // BT709 primaries for the one format carrying a colour; UNSPECIFIED for the
    // two carrying measurements, which have no primaries to speak of.
    let primaries: u8 = if format.is_srgb() { 1 } else { 0 };
    // KHR_DF_TRANSFER_SRGB = 2, KHR_DF_TRANSFER_LINEAR = 1. This byte is the
    // entire "sRGB decided at import" claim (§4.6), written down.
    let transfer: u8 = if format.is_srgb() { 2 } else { 1 };

    let block_size = 24 + 16 * samples.len();
    let mut dfd = Vec::with_capacity(4 + block_size);
    dfd.extend_from_slice(&((4 + block_size) as u32).to_le_bytes());
    // vendorId 0 (Khronos) in bits 0..17, descriptorType 0 (basic) above it.
    dfd.extend_from_slice(&0u32.to_le_bytes());
    // versionNumber 2 in bits 0..16 — KTX2 requires KHR_DF 1.3 — then the
    // block's own size, which excludes the four bytes of dfdTotalSize.
    dfd.extend_from_slice(&(2u32 | ((block_size as u32) << 16)).to_le_bytes());
    // KHR_DF_FLAG_ALPHA_STRAIGHT: premultiplied alpha is a decision the importer
    // does not make and the shader would have to undo.
    let flags: u8 = 0;
    dfd.extend_from_slice(&[color_model, primaries, transfer, flags]);
    // texelBlockDimension is stored as extent-1, so a 4x4x1x1 block is 3,3,0,0.
    dfd.extend_from_slice(&[BLOCK_EXTENT as u8 - 1, BLOCK_EXTENT as u8 - 1, 0, 0]);
    // bytesPlane0 is the block's size; a block format has one plane.
    let mut planes = [0u8; 8];
    planes[0] = format.block_bytes() as u8;
    dfd.extend_from_slice(&planes);
    for &(channel, offset, length) in samples {
        dfd.extend_from_slice(&offset.to_le_bytes());
        // `channel` carries its own qualifier nibble — zero for the three
        // unsigned normalized formats, FLOAT for BC6H.
        dfd.extend_from_slice(&[length, channel]);
        dfd.extend_from_slice(&[0u8; 4]); // samplePosition, all axes
        dfd.extend_from_slice(&lower.to_le_bytes()); // sampleLower
        dfd.extend_from_slice(&upper.to_le_bytes()); // sampleUpper
    }
    debug_assert_eq!(dfd.len(), 4 + block_size);
    dfd
}

/// The key/value data (KTX 2.0 §3.11). Keys are in codepoint order because the
/// spec requires it, which here means orientation before writer.
fn key_value_data() -> Vec<u8> {
    let mut kvd = Vec::new();
    for (key, value) in [
        // Origin top-left, rows downward: glTF's uv convention, and therefore
        // the one the importer never has to flip.
        ("KTXorientation", "rd"),
        ("KTXwriter", WRITER),
    ] {
        let len = key.len() + 1 + value.len() + 1;
        kvd.extend_from_slice(&(len as u32).to_le_bytes());
        kvd.extend_from_slice(key.as_bytes());
        kvd.push(0);
        kvd.extend_from_slice(value.as_bytes());
        kvd.push(0);
        // valuePadding: every entry ends 4-aligned, the last one included, so
        // whatever follows the KVD is aligned without a second rule.
        kvd.resize(kvd.len().next_multiple_of(size_of::<u32>()), 0);
    }
    kvd
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// A well-formed 8x8 BC7 blob to vandalise. The contents are irrelevant —
    /// the reader never interprets a block.
    fn blob() -> Vec<u8> {
        let levels: Vec<Vec<u8>> = (0..full_mip_count(8, 8))
            .map(|level| {
                let (w, h) = level_extent(8, 8, level);
                vec![level as u8; TextureFormat::Bc7Srgb.level_bytes(w, h) as usize]
            })
            .collect();
        encode(TextureFormat::Bc7Srgb, 8, 8, &levels).unwrap()
    }

    /// The same blob with `pixelWidth`/`pixelHeight` rewritten — the two words a
    /// hostile pack costs nothing to change.
    fn with_extent(width: u32, height: u32) -> Vec<u8> {
        let mut blob = blob();
        blob[20..24].copy_from_slice(&width.to_le_bytes());
        blob[24..28].copy_from_slice(&height.to_le_bytes());
        blob
    }

    #[test]
    fn an_extent_no_image_could_have_is_refused_before_anything_is_reserved() {
        // 65536² BC7 is 4 GiB a level and ~5.6 GiB a chain, named by ~900 bytes.
        let err = Texture::read(&with_extent(65_536, 65_536)).unwrap_err();
        assert!(
            matches!(err, TextureError::DimensionTooLarge { most, .. } if most == MAX_DIMENSION),
            "{err}"
        );
        // 2^31 square is the one that reached `Vec::with_capacity(2^62)`:
        // `handle_alloc_error` on a decode worker, which aborts the process and
        // cannot be caught by the caller that asked for the texture.
        let err = Texture::read(&with_extent(1 << 31, 1 << 31)).unwrap_err();
        assert!(
            matches!(err, TextureError::DimensionTooLarge { .. }),
            "{err}"
        );
        // A bound, not a shrug: the largest legal extent is still *read* as an
        // extent, and refused for what it then gets wrong.
        let err = Texture::read(&with_extent(MAX_DIMENSION, MAX_DIMENSION)).unwrap_err();
        assert!(matches!(err, TextureError::WrongLevelSize { .. }), "{err}");
    }

    #[test]
    fn the_size_arithmetic_answers_instead_of_overflowing() {
        // Exactly 2^64: a panic where overflow checks are on (and that panic is
        // inside a decode worker holding the inbox mutex, which poisons it and
        // takes the whole pool with it), a wrap to zero where they are not.
        assert_eq!(
            TextureFormat::Bc7Srgb.checked_level_bytes(u32::MAX, u32::MAX),
            None
        );
        assert_eq!(
            TextureFormat::Bc7Srgb.level_bytes(u32::MAX, u32::MAX),
            u64::MAX,
            "saturated, so no level index can match it"
        );
        assert_eq!(
            TextureFormat::Bc7Srgb.checked_level_bytes(MAX_DIMENSION, MAX_DIMENSION),
            Some(256 * 1024 * 1024)
        );
        assert_eq!(TextureFormat::Bc4Unorm.level_bytes(u32::MAX, 4), 1 << 33);
    }

    #[test]
    fn the_writer_refuses_the_extent_the_reader_refuses() {
        let level = vec![0u8; 16];
        let err = encode(TextureFormat::Bc7Srgb, MAX_DIMENSION + 1, 4, &[level]).unwrap_err();
        assert!(
            matches!(err, TextureError::DimensionTooLarge { .. }),
            "{err}"
        );
    }
}
