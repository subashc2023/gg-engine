//! Building a pack. Here rather than in `ggc` because a second definition of
//! the layout in [`crate::pack`] is how a format acquires a dialect.
//!
//! **Byte-reproducibility is a property of this module, not of its caller**
//! (§4.6). [`PackWriter::finish`] sorts by id and lays out from there, so the
//! order assets were added in — which is the order `ggc`'s threads happened to
//! finish in — cannot reach the bytes. A caller that wanted to break it would
//! have to change an asset's *name*.

use core::mem::size_of;

use thiserror::Error;

use crate::pack::{AssetId, AssetKind, Entry, FORMAT_VERSION, Header, MAGIC, SECTION_ALIGN};

/// Why a pack could not be built.
#[derive(Debug, Error)]
pub enum WriteError {
    /// Two assets hash to one id. Almost always the same name added twice; a
    /// genuine xxhash3 collision between two different names is the other
    /// reading, and the message carries both names so the difference is
    /// visible rather than assumed.
    #[error("{first} and {second} both have id {id} — the same asset twice, or a hash collision")]
    DuplicateId {
        /// The name already in the pack.
        first: String,
        /// The name that collided with it.
        second: String,
        /// The id they share.
        id: AssetId,
    },
    /// A name hashed to [`AssetId::NONE`], which optional references spend on
    /// "absent". One name in 2^64; refused so the sentinel stays a sentinel,
    /// and the fix is to rename the asset.
    #[error(
        "{name} hashes to the reserved id {0}, which means \"no asset\"",
        AssetId::NONE
    )]
    ReservedId {
        /// The name that hashed to it.
        name: String,
    },
    /// A name too long for the table's 32-bit slice, or a pack too large for
    /// its own offsets. Both are absurd sizes; refused rather than truncated.
    #[error("{what} does not fit the pack format: {size} exceeds {limit}")]
    TooLarge {
        /// Which quantity overflowed.
        what: &'static str,
        /// What it was.
        size: u64,
        /// What the format allows.
        limit: u64,
    },
}

struct Pending {
    id: AssetId,
    name: String,
    kind: AssetKind,
    source_hash: u64,
    blob: Vec<u8>,
}

/// Accumulates assets, then lays out a pack.
#[derive(Default)]
pub struct PackWriter {
    pending: Vec<Pending>,
}

impl PackWriter {
    /// An empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one asset. `name` is hashed exactly as given — normalize separators
    /// before this, or two hosts produce two packs.
    ///
    /// `source_hash` is whatever the importer keys its incremental rebuild on:
    /// the source bytes and the import settings, hashed together, so a settings
    /// change invalidates as surely as an edit does (§4.6).
    pub fn add(
        &mut self,
        name: &str,
        kind: AssetKind,
        source_hash: u64,
        blob: Vec<u8>,
    ) -> Result<AssetId, WriteError> {
        let id = AssetId::of(name);
        if id.is_none() {
            return Err(WriteError::ReservedId {
                name: name.to_owned(),
            });
        }
        if name.len() > u32::MAX as usize {
            return Err(WriteError::TooLarge {
                what: "an asset name",
                size: name.len() as u64,
                limit: u64::from(u32::MAX),
            });
        }
        self.pending.push(Pending {
            id,
            name: name.to_owned(),
            kind,
            source_hash,
            blob,
        });
        Ok(id)
    }

    /// How many assets are staged.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Whether nothing is staged. An empty pack is legal — a scene with no
    /// assets compiles to one, and refusing it would make the empty case a
    /// special case at every caller.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Lay out the pack and return its bytes.
    pub fn finish(mut self) -> Result<Vec<u8>, WriteError> {
        // The one line that makes the output a function of content: everything
        // below is positional, and this is what fixes the positions.
        self.pending.sort_unstable_by_key(|p| p.id);
        // Duplicates are adjacent now, which is why the check is here and not
        // in `add` — a scan per insert is quadratic, and Sponza is thousands.
        if let Some(pair) = self.pending.windows(2).find(|w| w[0].id == w[1].id) {
            return Err(WriteError::DuplicateId {
                first: pair[0].name.clone(),
                second: pair[1].name.clone(),
                id: pair[0].id,
            });
        }

        let count = u32::try_from(self.pending.len()).map_err(|_| WriteError::TooLarge {
            what: "the asset count",
            size: self.pending.len() as u64,
            limit: u64::from(u32::MAX),
        })?;

        let mut names = String::new();
        let mut entries = Vec::with_capacity(self.pending.len());
        for asset in &self.pending {
            let name_offset = u32::try_from(names.len()).map_err(|_| WriteError::TooLarge {
                what: "the name table",
                size: names.len() as u64,
                limit: u64::from(u32::MAX),
            })?;
            names.push_str(&asset.name);
            entries.push(Entry {
                id: asset.id,
                kind: asset.kind as u32,
                flags: 0,
                // Patched below, once the blob region's start is known.
                offset: 0,
                len: asset.blob.len() as u64,
                source_hash: asset.source_hash,
                name_offset,
                name_len: asset.name.len() as u32,
            });
        }

        let table_at = size_of::<Header>() as u64;
        let names_at = table_at + u64::from(count) * size_of::<Entry>() as u64;
        let mut cursor = align_up(names_at + names.len() as u64);
        for entry in &mut entries {
            entry.offset = cursor;
            cursor = align_up(cursor + entry.len);
        }
        let file_len = cursor;

        let mut out = Vec::with_capacity(file_len as usize);
        out.extend_from_slice(bytemuck::bytes_of(&Header {
            magic: MAGIC,
            format_version: FORMAT_VERSION,
            entry_count: count,
            entries: table_at,
            names: names_at,
            names_len: names.len() as u64,
            file_len,
            reserved: [0; 16],
        }));
        for entry in &entries {
            out.extend_from_slice(bytemuck::bytes_of(entry));
        }
        out.extend_from_slice(names.as_bytes());
        for (entry, asset) in entries.iter().zip(&self.pending) {
            // Padding is zeros rather than whatever was in the buffer: two runs
            // that agree on content must agree on every byte, gaps included.
            out.resize(entry.offset as usize, 0);
            out.extend_from_slice(&asset.blob);
        }
        out.resize(file_len as usize, 0);
        Ok(out)
    }
}

const fn align_up(offset: u64) -> u64 {
    offset.div_ceil(SECTION_ALIGN) * SECTION_ALIGN
}
