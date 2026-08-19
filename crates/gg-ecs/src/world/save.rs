//! The durable save (§6 M14) — a snapshot that has to survive a *different
//! build*, on disk, in the hands of someone who did not write it.
//!
//! [`Snapshot`] is a handoff between two processes of the same generation: no
//! checksum by design, and a format word that rejects a stale file rather than
//! carrying one forward (§4.2.2). A save inverts every one of those choices. It
//! is written by one build and read by the next, it sits on a disk that can
//! half-write it, and the data in it is the player's rather than the
//! developer's. So it wraps the same image in a container that says who wrote
//! it, when, and whether the bytes are all there — and it applies a *stricter*
//! load policy than a reload does.
//!
//! # A save may gain, never lose
//!
//! [`World::restore`] tolerates loss on purpose: a component the new build
//! stopped declaring is `Dropped`, a field that changed type is zeroed and
//! reported, and a field the new build no longer declares is discarded and
//! reported. That is right for a reload, where the developer just deleted the
//! thing and is watching the log. It is wrong for a save, where the same event
//! silently deletes someone's afternoon. [`World::load`] therefore refuses all
//! three, **by name**, before the world is touched — while migrating everything a
//! reload would: fields matched by name, reordered fields moved with their data,
//! new fields defaulted, new components simply absent from the image.
//!
//! That policy is what makes the version question answer itself.
//!
//! # There is no version number, and that is the design
//!
//! A save carries no "game version N" field. It carries the schema manifest —
//! every declared id, schema hash, field name, type token, offset and width the
//! writing build had — and *that* is the version, in a form the compiler
//! maintains instead of a human. A newer game's save read by an older build
//! trips the rule above the moment it names a component that build has never
//! heard of, and is refused by that component's name rather than by a number
//! nobody remembered to increment. The only version word here is the
//! container's own format, which is the *engine's* framing and a different
//! question entirely.
//!
//! # What does not ride along
//!
//! **The replay.** A save is world state at a tick; a replay is an input stream
//! from tick zero. Carrying one inside the other would grow the file with the
//! session and promise a reproducibility a save cannot deliver — restoring
//! mid-stream and then replaying applies the same inputs twice. Exactly one
//! number crosses: the tick, so a resumed session's clock continues instead of
//! restarting under a world that remembers — the same reason
//! `gg_core::reload::rejuvenate`'s handoff carries it.
//!
//! **Side-table contents.** P2: the strategy layer will want them, and it cannot
//! have them yet. [`SideTable`](crate::side_table::SideTable) is `StateHash`-only
//! by construction — a table is hashed, never copied — so carrying contents
//! means a serialization method on the trait, which is §7's named non-goal and
//! which no demo has pulled. The save records the declared *ids*, as the
//! snapshot does, so a load into a differently-equipped host is refused rather
//! than silently short of state.

use super::World;
use super::snapshot::{ComponentOutcome, MigrationReport, Snapshot, SnapshotError};

/// The container's magic and format word. Unlike the snapshot's, this one is
/// expected to be read by builds that did not write it — so it gates the
/// *engine's* framing only, and never the game's schema (see the module docs).
const SAVE_MAGIC: [u8; 4] = *b"GGSV";
const SAVE_FORMAT: u32 = 1;

/// Bytes of blake3 digest at the tail. The one hash primitive (§4.2.1), here for
/// the one job a snapshot deliberately does not do: catching a half-written file.
const CHECKSUM_BYTES: usize = 32;

/// Why a save could not be read, or could be read and not loaded.
///
/// Every variant names the artifact or the component at fault, because the
/// reader is holding a file from one build and running another and needs to know
/// which of the two is wrong.
#[derive(Debug, thiserror::Error)]
pub enum SaveError {
    /// The file does not start with the save magic.
    #[error("not a save file: magic is {found:02x?}, expected {SAVE_MAGIC:02x?}")]
    Magic {
        /// The four bytes actually found.
        found: [u8; 4],
    },
    /// A container format this build does not read.
    #[error("save file format {found}, this build reads {SAVE_FORMAT}")]
    Format {
        /// The format word found in the file.
        found: u32,
    },
    /// The file ends inside a field.
    #[error("save file ends early: {need} more bytes needed, {left} left")]
    Truncated {
        /// Bytes the field still needed.
        need: usize,
        /// Bytes actually remaining.
        left: usize,
    },
    /// The digest at the tail does not match the bytes ahead of it.
    #[error(
        "save file is damaged: it carries checksum {found:032x} but its contents hash to \
         {computed:032x}"
    )]
    Checksum {
        /// The digest the file claims.
        found: u128,
        /// The digest its bytes actually have.
        computed: u128,
    },
    /// A component in the save that this build no longer declares.
    #[error(
        "save holds component \"{declared}\", which this build does not declare. A reload may \
         drop a component the developer just deleted; a save may not, because the data is the \
         player's — and a save from a *newer* game reaches this same refusal, by the name of the \
         component that build added."
    )]
    Dropped {
        /// The declared id with nowhere to land.
        declared: String,
    },
    /// A field that kept its name and changed its type.
    #[error(
        "save holds \"{declared}.{field}\" under a type this build no longer declares it with — \
         migrating by name would copy bytes that no longer mean what they meant"
    )]
    Retyped {
        /// The component's declared id.
        declared: String,
        /// The field within it.
        field: String,
    },
    /// A field the save holds and this build no longer declares. The
    /// field-granularity twin of [`SaveError::Dropped`] and refused on the same
    /// grounds — the row lands, so nothing else here would ever notice.
    #[error(
        "save holds \"{declared}.{field}\", which this build no longer declares — the rest of \
         that component would migrate cleanly and this field's data would be gone, which is a \
         loss and not a migration"
    )]
    Removed {
        /// The component's declared id.
        declared: String,
        /// The field within it that has nowhere to land.
        field: String,
    },
    /// The image inside the container was refused on its own terms.
    #[error(transparent)]
    Snapshot(#[from] SnapshotError),
}

/// A world, on disk, meant to outlive the build that wrote it.
#[derive(Clone, Debug)]
pub struct Save {
    tick: u64,
    provenance: u128,
    snapshot: Snapshot,
}

impl Save {
    /// Wrap `snapshot` for durable storage.
    ///
    /// `tick` is the sim clock the session stops at. `provenance` identifies the
    /// build that wrote it — the shell passes the game dylib's content hash
    /// (§4.2.2) — and is **never** a gate: a save that could only be loaded by
    /// the build that wrote it would be a handoff, which already exists.
    #[must_use]
    pub fn new(snapshot: Snapshot, tick: u64, provenance: u128) -> Save {
        Save {
            tick,
            provenance,
            snapshot,
        }
    }

    /// The sim clock this save stops at.
    #[must_use]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// What the writing build identified itself by. Log it; do not branch on it.
    #[must_use]
    pub fn provenance(&self) -> u128 {
        self.provenance
    }

    /// The image inside, for a caller that wants to inspect the manifest before
    /// deciding anything ([`Snapshot::dry_run`]).
    #[must_use]
    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    /// Serialize, checksum included.
    ///
    /// Fixed-width little-endian fields and one length-prefixed run of bytes
    /// whose width was already known — §7's no-serialization non-goal survives
    /// this for the same reason [`Snapshot::encode`] does not break it. The
    /// digest goes at the tail so verifying is one pass over everything ahead of
    /// it, with nothing to exclude.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let image = self.snapshot.encode();
        let mut out = Vec::with_capacity(image.len() + 64);
        out.extend_from_slice(&SAVE_MAGIC);
        out.extend_from_slice(&SAVE_FORMAT.to_le_bytes());
        out.extend_from_slice(&self.tick.to_le_bytes());
        out.extend_from_slice(&self.provenance.to_le_bytes());
        out.extend_from_slice(&(image.len() as u32).to_le_bytes());
        out.extend_from_slice(&image);
        out.extend_from_slice(blake3::hash(&out).as_bytes());
        out
    }

    /// Read a file [`Save::encode`] wrote, rejecting damage before schema.
    ///
    /// Validates framing and the digest only. Whether the world inside can
    /// actually land in *this* build is [`World::load`]'s question, and a
    /// decoded `Save` is well formed rather than loadable.
    pub fn decode(bytes: &[u8]) -> Result<Save, SaveError> {
        // The digest is checked before a single length is believed: every field
        // read below trusts a number the file supplied, and a damaged file is
        // the case this container exists for.
        let split = bytes
            .len()
            .checked_sub(CHECKSUM_BYTES)
            .ok_or(SaveError::Truncated {
                need: CHECKSUM_BYTES,
                left: bytes.len(),
            })?;
        let (body, tail) = bytes.split_at(split);
        let computed = blake3::hash(body);
        if tail != computed.as_bytes() {
            return Err(SaveError::Checksum {
                found: prefix(tail),
                computed: prefix(computed.as_bytes()),
            });
        }

        let mut r = Reader { bytes: body, at: 0 };
        let magic: [u8; 4] = r.array()?;
        if magic != SAVE_MAGIC {
            return Err(SaveError::Magic { found: magic });
        }
        let format = u32::from_le_bytes(r.array()?);
        if format != SAVE_FORMAT {
            return Err(SaveError::Format { found: format });
        }
        let tick = u64::from_le_bytes(r.array()?);
        let provenance = u128::from_le_bytes(r.array()?);
        let len = u32::from_le_bytes(r.array()?) as usize;
        let snapshot = Snapshot::decode(r.take(len)?)?;
        Ok(Save {
            tick,
            provenance,
            snapshot,
        })
    }
}

/// The leading 16 bytes of a digest, for the message only — the comparison is
/// over all 32. A refusal that printed nothing would send the reader to a hex
/// editor to learn that two files differ.
fn prefix(bytes: &[u8]) -> u128 {
    let mut head = [0u8; 16];
    head.copy_from_slice(&bytes[..16]);
    u128::from_le_bytes(head)
}

/// A cursor over the container's header. Every read is bounds-checked for the
/// same reason the snapshot's is, minus one: the digest already agreed, so a
/// truncation here is a file that was *written* short rather than damaged.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], SaveError> {
        let end = self.at.saturating_add(count);
        let taken = self.bytes.get(self.at..end).ok_or(SaveError::Truncated {
            need: count,
            left: self.bytes.len() - self.at,
        })?;
        self.at = end;
        Ok(taken)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], SaveError> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }
}

impl World {
    /// Load `save` into this world, migrating forward and refusing to lose.
    ///
    /// The registry is this world's, as with [`World::restore`] — whatever is
    /// registered here is the running build's view and the save is an older
    /// build's data. What differs is the verdict on the gap: a component this
    /// build does not declare, or a field whose type moved under a name that
    /// stayed, is [`SaveError::Dropped`]/[`SaveError::Retyped`] rather than a
    /// line in a report.
    ///
    /// Refuses before touching anything: the check runs on
    /// [`Snapshot::dry_run`], so a rejected save leaves the player in the world
    /// they had.
    pub fn load(&mut self, save: &Save) -> Result<MigrationReport, SaveError> {
        for (declared, outcome) in save.snapshot.dry_run(self)? {
            match outcome {
                ComponentOutcome::Dropped => return Err(SaveError::Dropped { declared }),
                // `defaulted` is a field this build *added* — the gaining half of
                // the policy, and the whole point of migrating forward. The other
                // two are the losing half, and both are refused by name.
                ComponentOutcome::Migrated {
                    retyped, removed, ..
                } => {
                    if let Some(field) = retyped.into_iter().next() {
                        return Err(SaveError::Retyped { declared, field });
                    }
                    if let Some(field) = removed.into_iter().next() {
                        return Err(SaveError::Removed { declared, field });
                    }
                }
                ComponentOutcome::Reused => {}
            }
        }
        Ok(self.restore(&save.snapshot)?)
    }
}
