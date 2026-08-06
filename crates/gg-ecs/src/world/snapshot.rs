//! Snapshot, restore, and offset-mapped migration (§4.8, §4.2.2) — an M5
//! deliverable homed here because it is ECS machinery, not observability.
//!
//! A snapshot is a `Pod` column memcpy plus a manifest keyed by stable component
//! id and schema hash. That is the whole trick, and it is only available because
//! components are flat: §7's "no serialization" non-goal stands, because nothing
//! here reflects over a type, walks a trait object, or names a format. It copies
//! bytes whose width it was already told.
//!
//! # A child module on purpose
//!
//! Restore rebuilds archetypes, locations and the allocator together, from the
//! outside in. Doing that through accessors would open `World`'s internals to
//! the whole crate for one caller's sake; a child module sees its parent's
//! private fields and nobody else gains anything.
//!
//! # What a snapshot does not carry
//!
//! **Side tables.** They are host-owned sim state that cannot cross the reload
//! boundary at all — installing one is generic, and no generic crosses (§4.2.2) —
//! so no reload can change their layout and migration has no work to do on them.
//! [`World::restore`] leaves the live tables in place and *refuses* a snapshot
//! captured against a different set, rather than silently reviving a world with
//! state missing. The limit is named because the caller that will eventually
//! need more is rejuvenation, which restarts the process and takes the tables
//! with it.

use super::{Archetype, ArchetypeId, Location, World};
use crate::component::FieldDesc;
use crate::entity::{EntitiesImage, Entity};
use crate::hash::{ComponentId, SchemaHash};
use crate::registry::Registry;

/// The flattened image's magic and format word. Writer and reader are the *same
/// build* — a successor process (§4.2.2) — so the format word exists to reject a
/// stale file on disk, not to carry versions forward.
const IMAGE_MAGIC: [u8; 4] = *b"GGSN";
const IMAGE_FORMAT: u32 = 1;

/// Why a snapshot could not be restored.
#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    /// The image does not start with the snapshot magic.
    #[error("not a snapshot image: magic is {found:02x?}, expected {IMAGE_MAGIC:02x?}")]
    Magic {
        /// The four bytes actually found.
        found: [u8; 4],
    },
    /// A format word this build does not write. Snapshots are same-build only
    /// (§4.2.2), so this rejects a stale file rather than starting a migration.
    #[error("snapshot image format {found}, this build writes {IMAGE_FORMAT}")]
    Format {
        /// The format word found in the image.
        found: u32,
    },
    /// The image ends inside a record.
    #[error("snapshot image ends early: {need} more bytes needed, {left} left")]
    Truncated {
        /// Bytes the record still needed.
        need: usize,
        /// Bytes actually remaining.
        left: usize,
    },
    /// A declared id or type name in the image is not UTF-8.
    #[error("snapshot image holds a name that is not UTF-8")]
    NotUtf8,
    /// The captured allocator does not satisfy the allocation contract
    /// ([`crate::entity`]). Decoding bounds-checks truncation only, and a
    /// rejuvenation handoff is a file a *previous process* wrote and carries no
    /// checksum by design (§4.2.2) — so restore is the one place a corrupt
    /// freelist can be caught before it panics in `alloc`, hands one index to two
    /// live entities, or wraps a generation past `RETIRED` and resurrects every
    /// stale handle at that index.
    #[error("snapshot's entity allocator is not well formed: {detail}")]
    Allocator {
        /// Which rule, and the index that broke it.
        detail: String,
    },
    /// The world's installed side tables differ from the captured set. Side
    /// tables are host-owned, so a difference means the *host* changed.
    #[error(
        "snapshot was captured with side tables [{captured}] but this world has [{present}]. \
         Side tables are host-owned and cannot cross the reload boundary, so a difference means \
         the host changed — which is not something a snapshot carries."
    )]
    SideTableMismatch {
        /// Declared ids present when the snapshot was captured.
        captured: String,
        /// Declared ids installed in the world being restored into.
        present: String,
    },
    /// One declared id kept its schema hash while its size moved — which the
    /// hash covers, so one of the two artifacts is lying about itself.
    #[error(
        "component \"{declared}\" kept its schema hash but changed size ({captured} bytes when \
         captured, {present} now). Size is hashed into the schema, so one of the two artifacts \
         is not what it claims to be."
    )]
    SchemaContradiction {
        /// The declared id in contradiction.
        declared: String,
        /// Component size recorded in the image.
        captured: usize,
        /// Component size this build reports.
        present: usize,
    },
    /// The image's own arithmetic does not hold — a column shorter than its
    /// rows, a stride disagreeing with its component's size, a field past its
    /// component's end. Decoding proves framing, not arithmetic, and a
    /// rejuvenation handoff carries no checksum at all (§4.2.2), so this is
    /// refused *before* the world is torn down rather than panicking half-way
    /// through a restore.
    #[error("snapshot image is malformed: {detail}")]
    Malformed {
        /// What failed to add up, and where.
        detail: String,
    },
}

/// One field's layout, owned.
///
/// Owned rather than borrowed from the `&'static [FieldDesc]` it came from: the
/// descriptors live in the *old* dylib, and a snapshot exists precisely so the
/// old code can stop being consulted. Dev leaks dylibs, so borrowing would even
/// work — right up to rejuvenation, where the process owning those bytes is the
/// one being restarted.
#[derive(Clone, Debug, PartialEq, Eq)]
struct FieldImage {
    name: String,
    ty: String,
    offset: usize,
    size: usize,
}

/// One component type as it was when captured.
#[derive(Clone, Debug)]
struct ComponentImage {
    id: ComponentId,
    schema: SchemaHash,
    declared_id: String,
    size: usize,
    fields: Vec<FieldImage>,
}

/// One archetype's rows.
#[derive(Clone, Debug)]
struct ArchetypeImage {
    /// Ascending; parallel to `columns` and `strides`.
    ids: Vec<ComponentId>,
    /// Bytes per row *as captured*, which is not the destination's stride once a
    /// component has been migrated. Stored rather than divided out of the column
    /// length, which would be undefined for an empty archetype.
    strides: Vec<usize>,
    entities: Vec<Entity>,
    /// Row-major bytes, `entities.len() * strides[i]` each.
    columns: Vec<Vec<u8>>,
}

/// A world's component state, frozen.
///
/// Restoring into a world with the same registered components reproduces it
/// exactly, canonical hash included. Restoring into a world whose components
/// have moved runs the migration path (§4.2.2).
#[derive(Clone, Debug)]
pub struct Snapshot {
    entities: EntitiesImage,
    /// The manifest, ascending by id.
    components: Vec<ComponentImage>,
    archetypes: Vec<ArchetypeImage>,
    /// Declared ids of the side tables installed at capture, ascending.
    side_tables: Vec<String>,
}

impl Snapshot {
    /// Live entities captured.
    #[must_use]
    pub fn entity_count(&self) -> u32 {
        self.entities.live()
    }

    /// Component types captured, ascending by stable id.
    pub fn components(&self) -> impl Iterator<Item = (ComponentId, SchemaHash)> + '_ {
        self.components.iter().map(|c| (c.id, c.schema))
    }

    /// Flatten to bytes, for the one consumer that cannot be handed a `Snapshot`
    /// by value: rejuvenation, which gives this world to a *successor process*
    /// (§4.2.2).
    ///
    /// §7's no-serialization non-goal survives this for the same reason
    /// [`World::snapshot`] does not break it — fixed-width little-endian fields
    /// and length-prefixed runs of bytes whose widths were already known.
    /// Nothing reflects over a type; the columns are copied, never interpreted.
    /// A column carries no length of its own: `rows × stride` is the only length
    /// it can have, so a disagreement is not representable.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&IMAGE_MAGIC);
        put_u32(&mut out, IMAGE_FORMAT);

        put_u32(&mut out, self.entities.generations.len() as u32);
        for &generation in &self.entities.generations {
            put_u32(&mut out, generation);
        }
        put_u32(&mut out, self.entities.free.len() as u32);
        for &index in &self.entities.free {
            put_u32(&mut out, index);
        }
        put_u32(&mut out, self.entities.live);

        put_u32(&mut out, self.components.len() as u32);
        for component in &self.components {
            out.extend_from_slice(&component.id.get().to_le_bytes());
            out.extend_from_slice(&component.schema.get().to_le_bytes());
            put_u32(&mut out, component.size as u32);
            put_str(&mut out, &component.declared_id);
            put_u32(&mut out, component.fields.len() as u32);
            for field in &component.fields {
                put_str(&mut out, &field.name);
                put_str(&mut out, &field.ty);
                put_u32(&mut out, field.offset as u32);
                put_u32(&mut out, field.size as u32);
            }
        }

        put_u32(&mut out, self.archetypes.len() as u32);
        for archetype in &self.archetypes {
            put_u32(&mut out, archetype.ids.len() as u32);
            for (at, id) in archetype.ids.iter().enumerate() {
                out.extend_from_slice(&id.get().to_le_bytes());
                put_u32(&mut out, archetype.strides[at] as u32);
            }
            put_u32(&mut out, archetype.entities.len() as u32);
            for entity in &archetype.entities {
                out.extend_from_slice(&entity.to_bits().to_le_bytes());
            }
            for column in &archetype.columns {
                out.extend_from_slice(column);
            }
        }

        put_u32(&mut out, self.side_tables.len() as u32);
        for table in &self.side_tables {
            put_str(&mut out, table);
        }
        out
    }

    /// What restoring into `world` would do to each captured component, having
    /// decided nothing and touched nothing.
    ///
    /// The planning is [`World::restore`]'s own, so a caller that refuses on an
    /// outcome here refuses on the outcome it would actually have got — which is
    /// what makes it usable as a *precondition* rather than a second opinion.
    /// [`world::save`](super::save) is the caller: a save may not lose data a
    /// reload is allowed to drop, and that verdict has to be reached while the
    /// world is still the one the player had.
    pub fn dry_run(&self, world: &World) -> Result<Vec<(String, ComponentOutcome)>, SnapshotError> {
        world.check_side_tables(self)?;
        Ok(self.plan(&world.registry)?.1)
    }

    /// The image's structural arithmetic — everything `restore_archetype` later
    /// indexes by, proven while the world is still whole. `decode` checks
    /// framing only ("well *formed*, not well *founded*"), and the rejuvenation
    /// handoff is a file a previous process wrote with no checksum (§4.2.2),
    /// so without this a malformed image panics *after* `reset_storage` and the
    /// player's world is already gone.
    fn check_consistent(&self) -> Result<(), SnapshotError> {
        let bad = |detail: String| Err(SnapshotError::Malformed { detail });
        for component in &self.components {
            for field in &component.fields {
                if field.offset + field.size > component.size {
                    return bad(format!(
                        "component \"{}\" field \"{}\" spans {}..{} of a {}-byte row",
                        component.declared_id,
                        field.name,
                        field.offset,
                        field.offset + field.size,
                        component.size
                    ));
                }
            }
        }
        for (at, image) in self.archetypes.iter().enumerate() {
            if image.strides.len() != image.ids.len() || image.columns.len() != image.ids.len() {
                return bad(format!(
                    "archetype {at} has {} ids, {} strides, {} columns",
                    image.ids.len(),
                    image.strides.len(),
                    image.columns.len()
                ));
            }
            if !image.ids.windows(2).all(|w| w[0] < w[1]) {
                return bad(format!("archetype {at}'s component ids are not ascending"));
            }
            for (column, (&id, &stride)) in image.ids.iter().zip(&image.strides).enumerate() {
                let Some(component) = self.components.iter().find(|c| c.id == id) else {
                    return bad(format!(
                        "archetype {at} column {column} holds a component the manifest omits"
                    ));
                };
                if stride != component.size {
                    return bad(format!(
                        "archetype {at} strides \"{}\" at {stride} bytes; the manifest says {}",
                        component.declared_id, component.size
                    ));
                }
                if image.columns[column].len() != image.entities.len() * stride {
                    return bad(format!(
                        "archetype {at} column \"{}\" holds {} bytes for {} rows of {stride}",
                        component.declared_id,
                        image.columns[column].len(),
                        image.entities.len()
                    ));
                }
            }
        }
        Ok(())
    }

    /// One plan and one outcome per captured component, decided once rather than
    /// per row. Ascending by stable id, so `restore_archetype` can binary-search.
    fn plan(&self, registry: &Registry) -> Result<PlannedRestore, SnapshotError> {
        let mut plans: Vec<(ComponentId, Option<Plan>)> = Vec::new();
        let mut components = Vec::with_capacity(self.components.len());
        for old in &self.components {
            let Some(new) = registry.get(old.id) else {
                plans.push((old.id, None));
                components.push((old.declared_id.clone(), ComponentOutcome::Dropped));
                continue;
            };
            if new.schema == old.schema {
                if new.size != old.size {
                    return Err(SnapshotError::SchemaContradiction {
                        declared: old.declared_id.clone(),
                        captured: old.size,
                        present: new.size,
                    });
                }
                plans.push((old.id, Some(Plan::Verbatim)));
                components.push((old.declared_id.clone(), ComponentOutcome::Reused));
                continue;
            }
            let (plan, outcome) = migrate_fields(&old.fields, new.fields);
            plans.push((old.id, Some(plan)));
            components.push((old.declared_id.clone(), outcome));
        }
        Ok((plans, components))
    }

    /// Read back an image [`Snapshot::encode`] wrote.
    ///
    /// Validates only that the bytes are *this build's* image and are all there.
    /// Everything else — which components exist, whether a schema moved, whether
    /// the side tables agree, and whether the captured allocator satisfies its
    /// own contract — is [`World::restore`]'s to decide, and it already does. In
    /// particular a decoded [`Snapshot`] is well *formed*, not well *founded*.
    pub fn decode(bytes: &[u8]) -> Result<Snapshot, SnapshotError> {
        let mut r = Reader { bytes, at: 0 };
        let magic: [u8; 4] = r.array()?;
        if magic != IMAGE_MAGIC {
            return Err(SnapshotError::Magic { found: magic });
        }
        let format = r.u32()?;
        if format != IMAGE_FORMAT {
            return Err(SnapshotError::Format { found: format });
        }

        let entities = EntitiesImage {
            generations: r.u32s()?,
            free: r.u32s()?,
            live: r.u32()?,
        };

        let mut components = Vec::new();
        for _ in 0..r.count()? {
            let id = ComponentId::from_raw(r.u64()?);
            let schema = SchemaHash::from_raw(r.u128()?);
            let size = r.count()?;
            let declared_id = r.string()?;
            let mut fields = Vec::new();
            for _ in 0..r.count()? {
                fields.push(FieldImage {
                    name: r.string()?,
                    ty: r.string()?,
                    offset: r.count()?,
                    size: r.count()?,
                });
            }
            components.push(ComponentImage {
                id,
                schema,
                declared_id,
                size,
                fields,
            });
        }

        let mut archetypes = Vec::new();
        for _ in 0..r.count()? {
            let mut ids = Vec::new();
            let mut strides = Vec::new();
            for _ in 0..r.count()? {
                ids.push(ComponentId::from_raw(r.u64()?));
                strides.push(r.count()?);
            }
            let mut entities = Vec::new();
            for _ in 0..r.count()? {
                entities.push(Entity::from_bits(r.u64()?));
            }
            let mut columns = Vec::new();
            for &stride in &strides {
                // Saturating, not checked: a corrupt length can only make the
                // run larger than what is left, which `take` reports as the
                // truncation it is.
                columns.push(r.take(entities.len().saturating_mul(stride))?.to_vec());
            }
            archetypes.push(ArchetypeImage {
                ids,
                strides,
                entities,
                columns,
            });
        }

        let mut side_tables = Vec::new();
        for _ in 0..r.count()? {
            side_tables.push(r.string()?);
        }

        Ok(Snapshot {
            entities,
            components,
            archetypes,
            side_tables,
        })
    }
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_u32(out, value.len() as u32);
    out.extend_from_slice(value.as_bytes());
}

/// A cursor over an encoded image. Every read is bounds-checked: the file is
/// written by a process that is *exiting*, so a truncated one is a disk that
/// filled, not a case worth trusting.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], SnapshotError> {
        let end = self.at.saturating_add(count);
        let taken = self
            .bytes
            .get(self.at..end)
            .ok_or(SnapshotError::Truncated {
                need: count,
                left: self.bytes.len() - self.at,
            })?;
        self.at = end;
        Ok(taken)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], SnapshotError> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, SnapshotError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, SnapshotError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn u128(&mut self) -> Result<u128, SnapshotError> {
        Ok(u128::from_le_bytes(self.array()?))
    }

    /// A length or an index, as `usize`.
    fn count(&mut self) -> Result<usize, SnapshotError> {
        Ok(self.u32()? as usize)
    }

    fn u32s(&mut self) -> Result<Vec<u32>, SnapshotError> {
        let count = self.count()?;
        (0..count).map(|_| self.u32()).collect()
    }

    fn string(&mut self) -> Result<String, SnapshotError> {
        let count = self.count()?;
        let bytes = self.take(count)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| SnapshotError::NotUtf8)
    }
}

/// What happened to one component type across a restore.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComponentOutcome {
    /// Schema unchanged: the column was copied verbatim.
    Reused,
    /// Schema moved: fields were matched by name and copied by offset.
    Migrated {
        /// Matched by name *and* type, copied.
        copied: Vec<String>,
        /// Present in the new build only — zeroed.
        defaulted: Vec<String>,
        /// Present in both under different types — zeroed, and the case worth a
        /// warning, because here data was *lost* rather than never present.
        retyped: Vec<String>,
    },
    /// The new build no longer declares this component; its data was dropped.
    Dropped,
}

/// What a restore did, for the one log line per component §4.2.2 asks for.
#[derive(Clone, Debug)]
pub struct MigrationReport {
    /// Live entities restored.
    pub entities: u32,
    /// `(declared id, outcome)`, ascending by stable id. Every captured
    /// component appears exactly once.
    pub components: Vec<(String, ComponentOutcome)>,
}

impl MigrationReport {
    /// The report of a reload that never captured anything: §4.2.2's
    /// schema-unchanged pointer swap, where the world is untouched and no
    /// component has an outcome to name.
    #[must_use]
    pub fn untouched(entities: u32) -> MigrationReport {
        MigrationReport {
            entities,
            components: Vec::new(),
        }
    }

    /// Did every component come back untouched? True means the restore was a
    /// pure reload of identical state — the case worth *not* logging.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.components
            .iter()
            .all(|(_, o)| *o == ComponentOutcome::Reused)
    }
}

/// [`Snapshot::plan`]'s two parallel results: the per-component copy plan the
/// rows are rebuilt through, and the outcomes the report is made of.
type PlannedRestore = (
    Vec<(ComponentId, Option<Plan>)>,
    Vec<(String, ComponentOutcome)>,
);

/// How one surviving component's bytes move from the old column to the new.
enum Plan {
    /// Same schema: one memcpy per row.
    Verbatim,
    /// `(old offset, new offset, size)` per matched field. Everything else in
    /// the destination row stays zero, which is what "new fields default" means.
    Fields(Vec<(usize, usize, usize)>),
}

impl World {
    /// Freeze this world's component state (§4.8).
    ///
    /// Captures the allocator, the manifest and every column. Cost is a memcpy
    /// per column — reload- and rejuvenation-time machinery, never per-frame.
    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        let mut components: Vec<ComponentImage> = self
            .registry
            .iter()
            .map(|info| ComponentImage {
                id: info.id,
                schema: info.schema,
                declared_id: info.declared_id.to_string(),
                size: info.size,
                fields: info
                    .fields
                    .iter()
                    .map(|f| FieldImage {
                        name: f.name.to_string(),
                        ty: f.ty.to_string(),
                        offset: f.offset,
                        size: f.size,
                    })
                    .collect(),
            })
            .collect();
        components.sort_by_key(|c| c.id);

        let archetypes = self
            .archetypes
            .iter()
            .map(|a| ArchetypeImage {
                ids: a.ids().to_vec(),
                strides: (0..a.ids().len()).map(|at| a.column(at).stride()).collect(),
                entities: a.entities_slice().to_vec(),
                columns: (0..a.ids().len())
                    .map(|at| a.column(at).as_bytes().to_vec())
                    .collect(),
            })
            .collect();

        Snapshot {
            entities: self.entities.image(),
            components,
            archetypes,
            side_tables: self.side_tables.declared_ids(),
        }
    }

    /// Rebuild this world's component state from `snapshot`, migrating any
    /// component whose schema moved (§4.2.2).
    ///
    /// The *registry* is this world's, not the snapshot's: whatever is
    /// registered here is the new build's view of the world, and the snapshot is
    /// the old build's data. That asymmetry is the migration.
    ///
    /// Side tables are left in place and must match what was captured — see the
    /// module docs.
    pub fn restore(&mut self, snapshot: &Snapshot) -> Result<MigrationReport, SnapshotError> {
        self.check_side_tables(snapshot)?;
        snapshot.check_consistent()?;
        let (plans, components) = snapshot.plan(&self.registry)?;

        // Before anything is torn down: a refused allocator must leave this
        // world as it was rather than half-restored.
        self.entities.restore(&snapshot.entities)?;
        self.reset_storage();
        for image in &snapshot.archetypes {
            self.restore_archetype(image, &plans);
        }

        Ok(MigrationReport {
            entities: snapshot.entities.live(),
            components,
        })
    }

    /// The captured side tables must be the installed ones — see the module
    /// docs. First check of both entry points, because a host that changed is
    /// not a migration and there is nothing further worth planning.
    fn check_side_tables(&self, snapshot: &Snapshot) -> Result<(), SnapshotError> {
        let present = self.side_tables.declared_ids();
        if present == snapshot.side_tables {
            return Ok(());
        }
        Err(SnapshotError::SideTableMismatch {
            captured: snapshot.side_tables.join(", "),
            present: present.join(", "),
        })
    }

    /// Drop every archetype and location, keeping the registry, the side tables,
    /// and the empty archetype at index 0 — the invariant `spawn` relies on.
    fn reset_storage(&mut self) {
        self.archetypes.clear();
        self.by_ids.clear();
        self.archetypes.push(Archetype::new(Vec::new(), &[]));
        self.by_ids.push((Vec::new(), ArchetypeId::EMPTY));
        self.locations.clear();
    }

    /// Rebuild one captured archetype, dropping components the new build no
    /// longer declares.
    ///
    /// Two captured archetypes collapse into one when a dropped component was
    /// all that distinguished them, so the destination is looked up by component
    /// set rather than created blindly: `archetype_for` merges them and the rows
    /// append.
    fn restore_archetype(&mut self, image: &ArchetypeImage, plans: &[(ComponentId, Option<Plan>)]) {
        let mut kept: Vec<(usize, &Plan)> = Vec::new();
        let mut ids: Vec<ComponentId> = Vec::new();
        for (at, id) in image.ids.iter().enumerate() {
            let found = plans.binary_search_by_key(id, |(i, _)| *i);
            if let Ok(found) = found
                && let Some(plan) = plans[found].1.as_ref()
            {
                kept.push((at, plan));
                ids.push(*id);
            }
        }

        let target = self.archetype_for(ids);
        let archetype = &mut self.archetypes[target.index() as usize];
        let mut placed = Vec::with_capacity(image.entities.len());
        for (source_row, &entity) in image.entities.iter().enumerate() {
            let row = archetype.push_zeroed(entity);
            for (column_at, &(source_at, plan)) in kept.iter().enumerate() {
                let source_stride = image.strides[source_at];
                let from = &image.columns[source_at][source_row * source_stride..][..source_stride];
                let into = archetype.column_mut(column_at).row_mut(row as usize);
                match plan {
                    Plan::Verbatim => into.copy_from_slice(from),
                    Plan::Fields(moves) => {
                        for &(old_at, new_at, size) in moves {
                            into[new_at..new_at + size]
                                .copy_from_slice(&from[old_at..old_at + size]);
                        }
                    }
                }
            }
            placed.push((entity, row));
        }
        // Locations are set after the rows land: `set_location` needs `&mut
        // self`, and the archetype borrow above is still live inside the loop.
        for (entity, row) in placed {
            self.set_location(
                entity,
                Some(Location {
                    archetype: target,
                    row,
                }),
            );
        }
    }
}

/// Match new fields against old ones by name and decide what each one gets.
///
/// Name is the key, never offset or declaration order: reordering a struct's
/// fields must move the data with them, which is only possible if identity
/// survives the reorder.
fn migrate_fields(old: &[FieldImage], new: &[FieldDesc]) -> (Plan, ComponentOutcome) {
    let mut moves = Vec::new();
    let mut copied = Vec::new();
    let mut defaulted = Vec::new();
    let mut retyped = Vec::new();
    for field in new {
        match old.iter().find(|o| o.name == field.name) {
            // Type token *and* width must agree. The token alone is syntactic,
            // so an alias swap moves it; the width alone would happily copy a
            // `u32` into an `f32` — bit-identical, semantically nonsense.
            Some(o) if o.ty == field.ty && o.size == field.size => {
                moves.push((o.offset, field.offset, field.size));
                copied.push(field.name.to_string());
            }
            Some(_) => retyped.push(field.name.to_string()),
            None => defaulted.push(field.name.to_string()),
        }
    }
    (
        Plan::Fields(moves),
        ComponentOutcome::Migrated {
            copied,
            defaulted,
            retyped,
        },
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::World;

    /// Must refuse a malformed image before `reset_storage` runs — a save or
    /// rejuvenation handoff carries no checksum (§4.2.2). In-module because
    /// the corruption has to be built directly: `decode` only proves framing.
    #[test]
    fn a_malformed_image_is_refused_before_the_world_is_touched() {
        let mut world = World::new();
        let victim = world.spawn();
        let mut snapshot = world.snapshot();
        snapshot.components.push(ComponentImage {
            id: ComponentId::of("test.cube"),
            schema: SchemaHash::from_raw(1),
            declared_id: "test.cube".to_owned(),
            size: 8,
            fields: Vec::new(),
        });
        snapshot.archetypes.push(ArchetypeImage {
            ids: vec![ComponentId::of("test.cube")],
            strides: vec![8],
            entities: vec![victim],
            // One row of a claimed 8-byte stride, four bytes present: the
            // arithmetic `restore_archetype` would panic on mid-copy.
            columns: vec![vec![0u8; 4]],
        });

        let before = world.canonical_hash();
        let err = world.restore(&snapshot).unwrap_err();
        assert!(matches!(err, SnapshotError::Malformed { .. }), "{err}");
        assert_eq!(
            world.canonical_hash(),
            before,
            "a refusal must leave the world exactly as it found it"
        );
    }
}
