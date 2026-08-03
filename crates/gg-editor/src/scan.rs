//! What the editor knows about a world it was not compiled against.
//!
//! Everything here is built from API that shipped before this milestone —
//! [`Registry`](gg_ecs::registry::Registry) for the schema, [`QueryAccess`] and
//! `views_ref` for the storage — which is §6 M15's fifth exit row held at its
//! strong end: the editor asks the ECS nothing a host could not already ask.
//!
//! # Archetypes are the unit, not entities
//!
//! A per-entity component list over demo 05 would be ten thousand rows rebuilt
//! sixty times a second. It is not needed: every entity in an archetype has the
//! same components, so the scan is `archetypes × components` — a handful times a
//! handful — and paging into it copies only the rows on screen.
//!
//! The two walks are matched by the **entity slice's pointer**. `views_ref`
//! visits archetypes in one order and a component's walk is a subsequence of the
//! full one, so a cursor that only ever advances lines them up exactly.
//!
//! What that order *is* matters to anyone reading the tree: it is the ECS's own
//! archetype order and dense row within it, not spawn order and not entity
//! index. Two entities created one after another land far apart if a component
//! separates them, which is why the first row of demo 05's tree is its sun and
//! the second is a hub a hundred children later. Sorting would mean a full
//! ordering over ten thousand entities every tick; the tree shows the storage
//! and says so. P2: a name column, once anything in the engine gives an entity
//! one — an index is a poor thing to hunt through pages of.

use gg_ecs::component::FieldDesc;
use gg_ecs::hash::ComponentId;
use gg_ecs::{Entity, QueryAccess, World};

/// A component as the panels need it: everything copied out, because the
/// registry borrow cannot survive the `&mut World` an edit takes.
#[derive(Clone, Copy)]
pub struct Slot {
    /// Stable id — what an access set is built from.
    pub id: ComponentId,
    /// The name the game declared, which is what the tree and inspector show.
    pub declared: &'static str,
    /// Row stride, and the length of one entity's bytes.
    pub size: usize,
    /// The schema, straight off the component's `Component` impl.
    pub fields: &'static [FieldDesc],
}

/// One archetype: where its entity slice lives, how many rows it holds, and
/// which [`Slot`]s it carries.
#[derive(Clone, Copy)]
struct Arch {
    /// The entity slice's address — the key that lines the two walks up.
    at: usize,
    mask: u64,
}

/// The world's shape this tick.
#[derive(Default)]
pub struct Scan {
    /// Registry order, capped at 64 — the mask is a `u64` and a game declaring
    /// more components than that has outgrown a panel, not the editor.
    pub slots: Vec<Slot>,
    arch: Vec<Arch>,
    /// Live entities across every archetype.
    pub total: usize,
}

/// Components a mask can hold. Past this the tree stops showing the extras
/// rather than mis-attributing them.
pub const MAX_SLOTS: usize = 64;

impl Scan {
    /// Re-read the world's schema and shape. Cheap enough to run every tick,
    /// which is what keeps the tree honest across a reload.
    pub fn run(&mut self, world: &World) {
        self.slots.clear();
        self.slots
            .extend(world.registry().iter().take(MAX_SLOTS).map(|info| Slot {
                id: info.id,
                declared: info.declared_id,
                size: info.size,
                fields: info.fields,
            }));

        self.arch.clear();
        self.total = 0;
        // Empty read set, empty write set: `contains_all` of nothing is true, so
        // this matches every archetype and takes no columns.
        let Ok(all) = QueryAccess::new(&[], &[]) else {
            return;
        };
        let (arch, total) = (&mut self.arch, &mut self.total);
        world.views_ref(&all, |view| {
            arch.push(Arch {
                at: view.entities().as_ptr() as usize,
                mask: 0,
            });
            *total += view.len();
        });

        for (bit, slot) in self.slots.iter().enumerate() {
            let Ok(access) = QueryAccess::new(&[slot.id], &[]) else {
                continue;
            };
            let mut cursor = 0;
            let arch = &mut self.arch;
            world.views_ref(&access, |view| {
                let at = view.entities().as_ptr() as usize;
                while cursor < arch.len() && arch[cursor].at != at {
                    cursor += 1;
                }
                if let Some(found) = arch.get_mut(cursor) {
                    found.mask |= 1 << bit;
                    cursor += 1;
                }
            });
        }
    }

    /// Copy `count` entities starting at flat index `from`, with the mask of the
    /// archetype each came out of. Only what a page shows is ever copied.
    pub fn page(&self, world: &World, from: usize, count: usize, out: &mut Vec<(Entity, u64)>) {
        out.clear();
        let Ok(all) = QueryAccess::new(&[], &[]) else {
            return;
        };
        let (mut seen, mut cursor) = (0usize, 0usize);
        world.views_ref(&all, |view| {
            let mask = self.arch.get(cursor).map_or(0, |a| a.mask);
            cursor += 1;
            let entities = view.entities();
            if out.len() < count && seen + entities.len() > from {
                let start = from.saturating_sub(seen);
                for e in entities.iter().skip(start).take(count - out.len()) {
                    out.push((*e, mask));
                }
            }
            seen += entities.len();
        });
    }

    /// Which components `entity` carries, by the same mask the page uses.
    pub fn mask_of(&self, world: &World, entity: Entity) -> u64 {
        let Ok(all) = QueryAccess::new(&[], &[]) else {
            return 0;
        };
        let (mut found, mut cursor) = (0u64, 0usize);
        world.views_ref(&all, |view| {
            if found == 0 && view.entities().contains(&entity) {
                found = self.arch.get(cursor).map_or(0, |a| a.mask);
            }
            cursor += 1;
        });
        found
    }
}

/// Copy one entity's bytes for one component, or `None` if it does not carry it.
pub fn read_row(world: &World, entity: Entity, slot: &Slot, out: &mut Vec<u8>) -> bool {
    out.clear();
    let Ok(access) = QueryAccess::new(&[slot.id], &[]) else {
        return false;
    };
    let mut found = false;
    world.views_ref(&access, |view| {
        if found {
            return;
        }
        if let Some(row) = view.entities().iter().position(|e| *e == entity) {
            let bytes = view.read_bytes(0);
            let at = row * slot.size;
            if let Some(slice) = bytes.get(at..at + slot.size) {
                out.extend_from_slice(slice);
                found = true;
            }
        }
    });
    found
}

/// Hand one entity's bytes to `edit`, in place. `false` if the entity does not
/// carry the component — the tree can be a tick stale across a despawn, and a
/// nudge landing on a gone entity is a no-op rather than a panic.
pub fn write_row(
    world: &mut World,
    entity: Entity,
    slot: &Slot,
    edit: impl FnOnce(&mut [u8]),
) -> bool {
    let Ok(access) = QueryAccess::new(&[], &[slot.id]) else {
        return false;
    };
    let mut edit = Some(edit);
    let mut done = false;
    world.views(&access, |mut view| {
        if done {
            return;
        }
        let Some(row) = view.entities().iter().position(|e| *e == entity) else {
            return;
        };
        let bytes = view.write_bytes(0);
        let at = row * slot.size;
        if let Some(slice) = bytes.get_mut(at..at + slot.size)
            && let Some(edit) = edit.take()
        {
            edit(slice);
            done = true;
        }
    });
    done
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use gg_ecs::Component;
    use gg_math::sim;

    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
    #[component(id = "test.pos")]
    #[repr(C)]
    struct Pos {
        p: sim::DVec3,
    }

    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
    #[component(id = "test.tag")]
    #[repr(C)]
    struct Tag {
        n: u32,
    }

    /// Two archetypes and two components: the mask has to say which entity has
    /// what, and the page has to hand back the right slice of a flat list.
    #[test]
    fn the_scan_places_every_entity_in_the_archetype_it_lives_in() {
        let mut world = World::new();
        let mut both = Vec::new();
        for i in 0..6u32 {
            let e = world.spawn();
            world
                .insert(
                    e,
                    Pos {
                        p: sim::DVec3::new(f64::from(i), 0.0, 0.0),
                    },
                )
                .unwrap();
            if i % 2 == 0 {
                world.insert(e, Tag { n: i }).unwrap();
                both.push(e);
            }
        }
        let mut scan = Scan::default();
        scan.run(&world);
        assert_eq!(scan.total, 6);
        assert_eq!(scan.slots.len(), 2);
        let tag = scan
            .slots
            .iter()
            .position(|s| s.declared == "test.tag")
            .unwrap();

        let mut page = Vec::new();
        scan.page(&world, 0, 16, &mut page);
        assert_eq!(page.len(), 6);
        let tagged: Vec<_> = page
            .iter()
            .filter(|(_, m)| m & (1 << tag) != 0)
            .map(|(e, _)| *e)
            .collect();
        assert_eq!(tagged, both, "the mask followed the archetype");
        for (e, m) in &page {
            assert_eq!(*m, scan.mask_of(&world, *e), "page and lookup agree");
        }

        // Paging is a window on one flat list, not per archetype.
        let mut window = Vec::new();
        scan.page(&world, 2, 3, &mut window);
        assert_eq!(window.len(), 3);
        assert_eq!(window[0].0, page[2].0);
        assert_eq!(window[2].0, page[4].0);
    }

    #[test]
    fn a_row_reads_and_writes_through_the_schema() {
        let mut world = World::new();
        let e = world.spawn();
        world
            .insert(
                e,
                Pos {
                    p: sim::DVec3::new(1.0, 2.0, 3.0),
                },
            )
            .unwrap();
        let mut scan = Scan::default();
        scan.run(&world);
        let slot = scan.slots[0];

        let mut bytes = Vec::new();
        assert!(read_row(&world, e, &slot, &mut bytes));
        assert_eq!(bytes.len(), slot.size);
        assert!(write_row(&mut world, e, &slot, |b| {
            let field = &slot.fields[0];
            crate::value::nudge(b, field, 1, 0.5);
        }));
        assert_eq!(world.get::<Pos>(e).unwrap().p.y, 2.5);

        // A dead entity is a miss, not a panic.
        world.despawn(e);
        assert!(!read_row(&world, e, &slot, &mut bytes));
        assert!(!write_row(&mut world, e, &slot, |_| {}));
    }
}
