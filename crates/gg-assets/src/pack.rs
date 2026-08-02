//! The `gg-pack` container. **This module is the format's specification** (§6
//! M9 exit): the layout is what [`Header`] and [`Entry`] say it is, and there is
//! no prose elsewhere that could disagree with them.
//!
//! ```text
//! 0                       a pack, in file order
//! +---------------------------------------------------------------+
//! | Header            64 bytes, fixed forever                      |
//! +---------------------------------------------------------------+
//! | Entry[entry_count]  48 bytes each, sorted by id, ascending     |
//! +---------------------------------------------------------------+
//! | name table          utf-8, not terminated, sliced by Entry     |
//! +---------------------------------------------------------------+
//! | blob                16-aligned, padded with zeros              |
//! | blob                                                           |
//! | ...                                                            |
//! +---------------------------------------------------------------+
//! ```
//!
//! Four properties are load-bearing and each is enforced at load:
//!
//! - **Little-endian, `repr(C)`, 16-byte-aligned sections** (§4.6). Sections are
//!   cast in place out of the mapping; nothing here is deserialized. 16 is the
//!   alignment of the widest thing a blob may contain, so a blob's own header
//!   casts without the loader knowing what kind it is.
//! - **`magic` at 0 and `format_version` at 8, for every version there will
//!   ever be.** That is what lets a version mismatch be a named error rather
//!   than a garbage read: a reader that cannot understand the file can still
//!   understand *why*.
//! - **The entry table is sorted by id, strictly ascending.** One invariant
//!   doing three jobs: lookup is a binary search, duplicate ids are impossible
//!   to represent, and the table's order is a function of content rather than
//!   of the order `ggc` happened to finish its work in (§4.6,
//!   byte-reproducibility).
//! - **Every offset is validated before it is trusted.** A truncated or hostile
//!   pack must fail with a sentence, not with a read past the end of a mapping.

use core::mem::size_of;
use std::fs::File;
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;
use thiserror::Error;

/// The first eight bytes of every pack, at offset 0 for every version.
pub const MAGIC: [u8; 8] = *b"GG-PACK\0";

/// The layout version this build reads and writes. Bumped by any change to
/// [`Header`], [`Entry`] or a blob's own layout; a mismatch is refused at load.
pub const FORMAT_VERSION: u32 = 1;

/// Section and blob alignment (§4.6). Also the granularity every offset in the
/// file is a multiple of.
pub const SECTION_ALIGN: u64 = 16;

/// A pack-wide asset identity: [`gg_abi::asset_id`] of the asset's name.
///
/// The name is hashed exactly as given. Normalizing it — separators, case — is
/// the importer's job and has to happen before this, or a pack built on Windows
/// and a pack built on Linux name the same asset differently.
///
/// The hash lives in `gg-abi` rather than here because a game crate names
/// assets and may not link this one (§3's deny pin); see that module for why
/// that is one definition rather than two.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Pod, Zeroable)]
pub struct AssetId(pub u64);

impl AssetId {
    /// The absent reference — a material with no normal map, a mesh with no
    /// material. Reserved rather than merely unused: a pack may not contain it
    /// (both the writer and the loader refuse it), so an optional id needs no
    /// flag beside it to say whether it means anything.
    pub const NONE: Self = Self(0);

    /// Hash `name` into an id. `const`, so a name written in source costs
    /// nothing at runtime.
    #[must_use]
    pub const fn of(name: &str) -> Self {
        Self(gg_abi::asset_id(name))
    }

    /// Whether this is [`Self::NONE`].
    #[must_use]
    pub const fn is_none(self) -> bool {
        self.0 == Self::NONE.0
    }
}

impl core::fmt::Display for AssetId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:#018x}", self.0)
    }
}

/// What a blob holds. Stored as `u32`; an unknown value is refused at load
/// rather than reached past, since the reader would otherwise cast a blob to a
/// header of the wrong shape.
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AssetKind {
    /// A mesh blob: one mesh header, then vertices and indices.
    Mesh = 1,
    /// A KTX2 container: compressed texels, mip chain, format (§4.6).
    Texture = 2,
    /// A material blob: one material's texture ids and factors.
    Material = 3,
    /// A scene blob: the node list a demo instantiates.
    Scene = 4,
}

impl AssetKind {
    const fn from_u32(raw: u32) -> Option<Self> {
        match raw {
            1 => Some(Self::Mesh),
            2 => Some(Self::Texture),
            3 => Some(Self::Material),
            4 => Some(Self::Scene),
            _ => None,
        }
    }
}

/// The pack header: 64 bytes, at offset 0, the same size in every version.
///
/// Its size is frozen so that a *future* pack still parses far enough here to
/// be rejected by name. Fields after `format_version` may be reinterpreted by a
/// version bump; those two may not move.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Header {
    /// [`MAGIC`]. Offset 0, forever.
    pub magic: [u8; 8],
    /// [`FORMAT_VERSION`]. Offset 8, forever.
    pub format_version: u32,
    /// Number of [`Entry`] records in the table.
    pub entry_count: u32,
    /// Byte offset of the entry table from the start of the file.
    pub entries: u64,
    /// Byte offset of the name table.
    pub names: u64,
    /// Length of the name table in bytes. Not padded — the first blob's offset
    /// carries the alignment instead.
    pub names_len: u64,
    /// The file's own length. Checked against the real one before any other
    /// offset is trusted, so a truncated pack fails as truncation rather than
    /// as whichever offset happened to run off the end first.
    pub file_len: u64,
    /// Zero. Present so the header is 64 bytes and the entry table starts
    /// aligned with no arithmetic.
    pub reserved: [u8; 16],
}

/// One asset's record. 48 bytes, three times [`SECTION_ALIGN`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Entry {
    /// The asset's identity, and the table's sort key.
    pub id: AssetId,
    /// An [`AssetKind`] discriminant.
    pub kind: u32,
    /// Reserved, zero. Kind-specific bits live in the blob's own header, where
    /// they can grow without the table changing size.
    pub flags: u32,
    /// Byte offset of the blob from the start of the file. 16-aligned.
    pub offset: u64,
    /// The blob's length in bytes, before the padding that follows it.
    pub len: u64,
    /// xxhash3 of the source bytes and the import settings that produced this
    /// blob. `ggc` compares it to decide whether a rebuild has anything to do
    /// (§4.6); nothing at runtime reads it.
    pub source_hash: u64,
    /// Byte offset of the name within the name table.
    pub name_offset: u32,
    /// Length of the name in bytes.
    pub name_len: u32,
}

impl Entry {
    /// The entry's kind. `None` is unreachable for an entry of an open
    /// [`Pack`] — validation refuses an unknown discriminant — and stays an
    /// `Option` rather than a fallback, because a kind guessed wrong casts a
    /// blob to the wrong header.
    #[must_use]
    pub fn kind(&self) -> Option<AssetKind> {
        AssetKind::from_u32(self.kind)
    }
}

const _: () = assert!(size_of::<Header>() == 64);
const _: () = assert!(size_of::<Entry>() == 48);
const _: () = assert!((size_of::<Entry>() as u64).is_multiple_of(SECTION_ALIGN));

/// Why a pack could not be opened. Every variant names the fault; none of them
/// means "the bytes were read anyway".
#[derive(Debug, Error)]
pub enum PackError {
    /// The file is shorter than a header, so not even the magic can be checked.
    #[error("not a gg-pack: {len} bytes is shorter than the {header}-byte header")]
    TooShort {
        /// The file's length.
        len: usize,
        /// [`Header`]'s size.
        header: usize,
    },
    /// The first eight bytes are not [`MAGIC`].
    #[error("not a gg-pack: it starts {found:02x?}, and a pack starts {MAGIC:02x?}")]
    NotAPack {
        /// What was there instead.
        found: [u8; 8],
    },
    /// A pack, but of a layout this build does not read.
    #[error("gg-pack format version {found}, and this build reads version {FORMAT_VERSION}")]
    Version {
        /// The version in the file.
        found: u32,
    },
    /// The header's own length field disagrees with the file's.
    #[error("truncated gg-pack: the header claims {claimed} bytes, the file holds {actual}")]
    Truncated {
        /// What the header says.
        claimed: u64,
        /// What is there.
        actual: u64,
    },
    /// A section or blob runs past the end of the file.
    #[error("{what} spans {offset}..{end} of a {actual}-byte gg-pack")]
    OutOfBounds {
        /// Which section, named.
        what: String,
        /// Where it starts.
        offset: u64,
        /// Where it would end.
        end: u64,
        /// The file's length.
        actual: u64,
    },
    /// An offset that must be 16-aligned is not.
    #[error("{what} starts at {offset}, which is not a multiple of {SECTION_ALIGN}")]
    Misaligned {
        /// Which section, named.
        what: String,
        /// Where it starts.
        offset: u64,
    },
    /// An entry claims [`AssetId::NONE`], which optional references use to mean
    /// "absent" and so may not name a real asset.
    #[error("entry {index} ({name}) claims the reserved id {0}", AssetId::NONE)]
    ReservedId {
        /// The entry's index in the table.
        index: u32,
        /// The entry's name.
        name: String,
    },
    /// An entry's `kind` is not an [`AssetKind`].
    #[error("entry {index} ({name}) has kind {kind}, which this build does not know")]
    UnknownKind {
        /// The entry's index in the table.
        index: u32,
        /// The entry's name, if it was readable.
        name: String,
        /// The unknown discriminant.
        kind: u32,
    },
    /// The table is out of order, which would break both lookup and §4.6's
    /// byte-reproducibility.
    #[error("entry {index} has id {id} after {previous}: the table is not sorted by id")]
    Unsorted {
        /// The offending entry's index.
        index: u32,
        /// Its id.
        id: AssetId,
        /// The id before it.
        previous: AssetId,
    },
    /// An entry's name slice is outside the name table.
    #[error("entry {index} names bytes {offset}..{end} of a {len}-byte name table")]
    BadName {
        /// The offending entry's index.
        index: u32,
        /// Where the name starts.
        offset: u64,
        /// Where it would end.
        end: u64,
        /// The name table's length.
        len: u64,
    },
    /// The name table is not utf-8.
    #[error("the name table is not utf-8: {0}")]
    NameNotUtf8(#[source] core::str::Utf8Error),
    /// The file could not be opened or mapped.
    #[error("{path}: {source}")]
    Io {
        /// The path that was tried.
        path: String,
        /// What the OS said.
        #[source]
        source: std::io::Error,
    },
}

/// A 16-aligned heap buffer, so an owned pack casts exactly like a mapped one.
///
/// A `Vec<u8>` is align-1 *by type* and bytemuck is right to refuse it; an
/// allocator that happened to hand back an aligned pointer would make the
/// difference between working and not a property of the machine.
#[repr(C, align(16))]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Chunk([u8; SECTION_ALIGN as usize]);

enum Backing {
    Mapped(Mmap),
    Owned {
        chunks: Box<[Chunk]>,
        /// The true byte length; the last chunk is zero-padded.
        len: usize,
    },
}

impl Backing {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Mapped(map) => map,
            // Validation proved `len` fits: the buffer was sized from it.
            Self::Owned { chunks, len } => &bytemuck::cast_slice(chunks)[..*len],
        }
    }
}

/// An open pack.
///
/// The blobs stay in the mapping and are handed out as borrowed slices — that
/// is what mmap is for. The entry and name tables are *copied* at open, which
/// costs 48 bytes an asset and buys away every later question about whether a
/// cast is aligned; they are the only two sections read more than once.
pub struct Pack {
    backing: Backing,
    entries: Box<[Entry]>,
    names: Box<str>,
}

impl core::fmt::Debug for Pack {
    /// Counts, never contents: a pack is megabytes and a `{:?}` that printed
    /// them would be a debugger that hangs the terminal it was called from.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Pack")
            .field("assets", &self.entries.len())
            .field("bytes", &self.backing.bytes().len())
            .finish()
    }
}

impl Pack {
    /// Map `path` and validate it. The mapping is read-only and lives as long
    /// as the returned pack.
    ///
    /// A pack is expected to be immutable while open: a rebuilt pack is a *new*
    /// [`Pack`] over the new file (§4.6 watch mode), never an edit under a live
    /// mapping.
    pub fn open(path: &Path) -> Result<Self, PackError> {
        let io = |source| PackError::Io {
            path: path.display().to_string(),
            source,
        };
        let file = File::open(path).map_err(io)?;
        // SAFETY: mmap is unsound against another process truncating or writing
        // the file under us; the contract above is that a pack is immutable
        // while open, and `ggc` writes a new file and renames rather than
        // editing in place, so a rebuild replaces the inode instead of the
        // bytes this mapping sees.
        let map = unsafe { Mmap::map(&file) }.map_err(io)?;
        Self::validate(Backing::Mapped(map))
    }

    /// Copy `bytes` into an aligned buffer and validate them. For tests, and
    /// for a pack that arrived as bytes rather than as a file.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PackError> {
        let chunks = bytes.len().div_ceil(SECTION_ALIGN as usize);
        let mut buffer = vec![Chunk([0; SECTION_ALIGN as usize]); chunks].into_boxed_slice();
        bytemuck::cast_slice_mut::<Chunk, u8>(&mut buffer)[..bytes.len()].copy_from_slice(bytes);
        Self::validate(Backing::Owned {
            chunks: buffer,
            len: bytes.len(),
        })
    }

    /// The entry table, sorted by id.
    #[must_use]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// `entry`'s name. Cheap: the name table is resident.
    ///
    /// # Panics
    /// Never for an entry of this pack — the slice was validated at open. An
    /// entry from a *different* pack is a caller bug and may panic.
    #[must_use]
    pub fn name(&self, entry: &Entry) -> &str {
        let start = entry.name_offset as usize;
        &self.names[start..start + entry.name_len as usize]
    }

    /// `entry`'s blob, borrowed straight out of the mapping.
    ///
    /// # Panics
    /// Same contract as [`Self::name`].
    #[must_use]
    pub fn blob(&self, entry: &Entry) -> &[u8] {
        let start = entry.offset as usize;
        &self.backing.bytes()[start..start + entry.len as usize]
    }

    /// Find an asset by id. Binary search over the sorted table.
    #[must_use]
    pub fn find(&self, id: AssetId) -> Option<&Entry> {
        let index = self.entries.binary_search_by_key(&id, |e| e.id).ok()?;
        self.entries.get(index)
    }

    /// Find an asset by name. Hashes and then [`Self::find`]s — a pack holds no
    /// name index, because a lookup by name is a load-time convenience and a
    /// lookup by id is the hot path.
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<&Entry> {
        self.find(AssetId::of(name))
    }

    /// Read the header out of `bytes` far enough to name what is wrong with it.
    /// Split out because these three checks must work on *any* version.
    fn header(bytes: &[u8]) -> Result<Header, PackError> {
        let head = bytes
            .get(..size_of::<Header>())
            .ok_or(PackError::TooShort {
                len: bytes.len(),
                header: size_of::<Header>(),
            })?;
        // `pod_read_unaligned`, not `from_bytes`: the mapping is page-aligned
        // and the owned buffer is 16-aligned, but proving that here would be a
        // second place the alignment argument lives.
        let header: Header = bytemuck::pod_read_unaligned(head);
        if header.magic != MAGIC {
            return Err(PackError::NotAPack {
                found: header.magic,
            });
        }
        if header.format_version != FORMAT_VERSION {
            return Err(PackError::Version {
                found: header.format_version,
            });
        }
        Ok(header)
    }

    fn validate(backing: Backing) -> Result<Self, PackError> {
        let bytes = backing.bytes();
        let header = Self::header(bytes)?;
        let actual = bytes.len() as u64;
        if header.file_len != actual {
            return Err(PackError::Truncated {
                claimed: header.file_len,
                actual,
            });
        }

        let table_len = u64::from(header.entry_count) * size_of::<Entry>() as u64;
        let table = span(bytes, "the entry table", header.entries, table_len, true)?;
        let names = span(
            bytes,
            "the name table",
            header.names,
            header.names_len,
            false,
        )?;
        let names = core::str::from_utf8(names).map_err(PackError::NameNotUtf8)?;

        let entries: Box<[Entry]> = table
            .chunks_exact(size_of::<Entry>())
            .map(bytemuck::pod_read_unaligned::<Entry>)
            .collect();

        // Ascending from the reserved id, so `AssetId::NONE` is refused at
        // index 0 by the same comparison that refuses a duplicate later.
        let mut previous = AssetId::NONE;
        for (index, entry) in entries.iter().enumerate() {
            let index = index as u32;
            let name_end = u64::from(entry.name_offset) + u64::from(entry.name_len);
            let name = names
                .get(entry.name_offset as usize..name_end as usize)
                .ok_or(PackError::BadName {
                    index,
                    offset: u64::from(entry.name_offset),
                    end: name_end,
                    len: header.names_len,
                })?;
            if entry.id.is_none() {
                return Err(PackError::ReservedId {
                    index,
                    name: name.to_owned(),
                });
            }
            if AssetKind::from_u32(entry.kind).is_none() {
                return Err(PackError::UnknownKind {
                    index,
                    name: name.to_owned(),
                    kind: entry.kind,
                });
            }
            // Strictly ascending, so equal ids are unrepresentable rather than
            // merely unwanted — a duplicate cannot survive to shadow its twin.
            if entry.id <= previous {
                return Err(PackError::Unsorted {
                    index,
                    id: entry.id,
                    previous,
                });
            }
            previous = entry.id;
            span(bytes, name, entry.offset, entry.len, true)?;
        }

        let names: Box<str> = names.into();
        Ok(Self {
            backing,
            entries,
            names,
        })
    }
}

/// Check that `offset..offset + len` is inside `bytes`, and 16-aligned if
/// `aligned`. Returns the slice so a caller cannot forget to use the checked
/// one.
fn span<'a>(
    bytes: &'a [u8],
    what: &str,
    offset: u64,
    len: u64,
    aligned: bool,
) -> Result<&'a [u8], PackError> {
    if aligned && !offset.is_multiple_of(SECTION_ALIGN) {
        return Err(PackError::Misaligned {
            what: what.to_owned(),
            offset,
        });
    }
    let actual = bytes.len() as u64;
    // Saturating, not checked: a hostile `len` of `u64::MAX` must report a span
    // that ends past the file, not wrap around to one that fits inside it.
    let end = offset.saturating_add(len);
    let inside = end <= actual && usize::try_from(end).is_ok();
    if !inside {
        return Err(PackError::OutOfBounds {
            what: what.to_owned(),
            offset,
            end,
            actual,
        });
    }
    bytes
        .get(offset as usize..end as usize)
        .ok_or(PackError::OutOfBounds {
            what: what.to_owned(),
            offset,
            end,
            actual,
        })
}
