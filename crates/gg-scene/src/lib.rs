//! `gg-scene` (§4.7) — the transform hierarchy.
//!
//! A game spells a parent link with [`Node`]; this crate turns those local
//! transforms into the world transforms the render protocol carries, writing
//! the result back into [`Renderable`] and [`Model`]. It runs **inside the
//! tick, before the canonical hash**, so a derived transform is hashed state
//! like any other and the three-way gate covers the compose itself rather than
//! only the locals feeding it.
//!
//! That placement is also why there is no `rayon` here and no `gg_math::render`:
//! this crate is determinism-critical by what it writes, not by what it reads.
//!
//! # Four columnar passes, no random access
//!
//! The obvious implementation — walk the tree, `World::get_mut` each entity —
//! pays an entity-to-row lookup per node, twice, every frame. Instead:
//!
//! 1. gather every `Renderable`/`Model` transform into a dense array;
//! 2. gather every `Node`;
//! 3. compose, in parent-before-child order, touching no ECS storage at all;
//! 4. write the changed ones back through a columnar query.
//!
//! Every pass is a column walk and the arithmetic step is a loop over a flat
//! array, which is what makes 10k nodes (§6 M10) a sub-millisecond cost.
//!
//! # What "dirty" buys, and what it costs
//!
//! `gg-ecs` has no change detection, so the dirty *set* is found by comparing
//! each local against the one seen last frame — O(n) regardless. What dirtiness
//! skips is the **compose**: a quaternion multiply and a rotate per node, plus
//! the write-back. A still hierarchy therefore costs a scan and nothing else.
//! Trading a memcmp for a compose is the whole of the claim, and it is worth
//! stating plainly rather than implying subtree-sized savings that only a real
//! change-detection bit could deliver.

#![warn(missing_docs)]

use gg_ecs::boundary::{Model, Node, Renderable};
use gg_ecs::{AliasError, Entity, Query, World};
use gg_math::sim;

/// A position and an orientation — what composes down the hierarchy.
///
/// No scale, deliberately: see [`Node`] for why a TRS hierarchy that composes
/// non-uniform scale stops being TRS.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    /// World-space position, `f64` and un-narrowed (§1.4).
    pub position: sim::DVec3,
    /// World-space orientation.
    pub rotation: sim::DQuat,
}

impl Transform {
    /// At the origin, unrotated.
    pub const IDENTITY: Transform = Transform {
        position: sim::DVec3::ZERO,
        rotation: sim::DQuat::IDENTITY,
    };

    /// `local` expressed in the frame `self` describes.
    ///
    /// The offset is rotated by the parent's orientation before it is added, so
    /// turning a parent swings its children rather than sliding them — the same
    /// composition `gg-extract` applies to a scene's nodes (§4.6), spelled once
    /// per side of the membrane because the two operate on different types.
    #[must_use]
    pub fn compose(&self, local: &Transform) -> Transform {
        Transform {
            position: self.position + self.rotation.rotate(local.position),
            rotation: self.rotation.mul(local.rotation),
        }
    }
}

/// Why a hierarchy could not be resolved. Every variant names the entity, since
/// "the hierarchy is invalid" is not a sentence anyone can act on.
#[derive(Debug, thiserror::Error)]
pub enum SceneError {
    /// A node's parent chain returns to itself, reported at the entity the walk
    /// re-entered — which is on the cycle.
    #[error("entity {0:?} is its own ancestor")]
    Cycle(Entity),
    /// A node names a parent that is not alive.
    #[error("entity {child:?} names parent {parent:?}, which is not alive")]
    MissingParent {
        /// The node holding the dangling reference.
        child: Entity,
        /// What it named.
        parent: Entity,
    },
    /// A node names a live parent carrying no transform to be relative to — no
    /// `Renderable`, no `Model`, and no `Node` of its own.
    ///
    /// An invisible pivot is the case this refuses, and refusing it is
    /// deliberate: the alternative is treating an absent transform as the
    /// identity, which puts a subtree at the world origin and reads as a bug in
    /// the game rather than in its hierarchy.
    #[error("entity {child:?} names parent {parent:?}, which has no transform")]
    ParentWithoutTransform {
        /// The node.
        child: Entity,
        /// The parent lacking a transform.
        parent: Entity,
    },
    /// The world refused a query.
    #[error(transparent)]
    Alias(#[from] AliasError),
}

/// What one propagation did. Numbers rather than a bool, because "the hierarchy
/// got expensive this frame" is the thing worth putting on the overlay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Propagated {
    /// Nodes considered — the scan, which is paid whatever moved.
    pub nodes: usize,
    /// Nodes recomposed: the quaternion work actually done.
    pub composed: usize,
    /// Component writes back into the render protocol. An entity carrying both
    /// `Renderable` and `Model` counts twice, because both were written.
    pub written: usize,
}

/// One entity's row in the dense table, indexed by [`Entity::index`] — a lookup
/// is an array read rather than a map probe, which is the reason a 10k-node
/// hierarchy costs what it costs.
#[derive(Clone, Copy)]
struct Slot {
    /// Which incarnation of this index the row describes. A recycled index
    /// whose generation moved on is an empty row, never stale data.
    generation: u32,
    /// This frame's world transform.
    world: Transform,
    /// What it was last frame, for the moved comparison.
    world_prev: Transform,
    /// Last frame's local, likewise.
    local_prev: Transform,
    /// Set when this frame's world differs from last frame's.
    changed: bool,
    /// Whether this entity has a transform at all this frame. What makes
    /// [`SceneError::ParentWithoutTransform`] detectable.
    present: bool,
    /// Whether a `Node` was seen for it this frame.
    is_node: bool,
}

impl Slot {
    const EMPTY: Slot = Slot {
        generation: u32::MAX,
        world: Transform::IDENTITY,
        world_prev: Transform::IDENTITY,
        local_prev: Transform::IDENTITY,
        changed: false,
        present: false,
        is_node: false,
    };
}

/// Depth-walk state, separate from [`Slot`] because it exists only while the
/// order is being rebuilt.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Visit {
    Unvisited,
    OnStack,
    Done,
}

/// The hierarchy's frame-to-frame memory: the dense table, the parent-before-
/// child order, and enough of last frame to tell what moved.
///
/// Host-owned and deliberately **not** a `SideTable`: everything here is
/// derived from state the hash already covers, and hashing a cache is how a
/// corrupt cache becomes a replay divergence instead of a wrong picture.
pub struct Hierarchy {
    slots: Vec<Slot>,
    /// This frame's nodes, in world iteration order.
    nodes: Vec<(Entity, Node)>,
    /// Indices into `nodes`, parent before child. Rebuilt only when the shape
    /// changes, which is rare next to the transforms moving.
    shape_order: Vec<u32>,
    /// Fingerprint of `(entity, parent, parent-has-a-transform)` over `nodes` —
    /// what decides whether `shape_order` still describes this frame's tree.
    shape: u64,
    /// Set for the frame after a rebuild: every node composes once, so a first
    /// frame cannot mistake an untouched `local` for an up-to-date `world`.
    force: bool,
    visit: Vec<Visit>,
    /// Reused by the iterative walk, so a deep chain costs no recursion.
    stack: Vec<u32>,
    /// `nodes` position by entity index, rebuilt with the order.
    node_at: Vec<u32>,
}

impl Default for Hierarchy {
    fn default() -> Self {
        Self::new()
    }
}

impl Hierarchy {
    /// An empty hierarchy. One per world, held by whoever runs the tick.
    #[must_use]
    pub fn new() -> Self {
        Hierarchy {
            slots: Vec::new(),
            nodes: Vec::new(),
            shape_order: Vec::new(),
            // Not 0: an empty hierarchy and one whose fingerprint happens to be
            // 0 would agree, and the first frame would skip its rebuild.
            shape: u64::MAX,
            force: true,
            visit: Vec::new(),
            stack: Vec::new(),
            node_at: Vec::new(),
        }
    }

    /// Compose every [`Node`]'s world transform and write the changed ones back
    /// into `Renderable`/`Model`.
    ///
    /// Call once per tick, after the game's systems and before the canonical
    /// hash. A world with no nodes costs two column walks and returns zeroes.
    ///
    /// # Errors
    ///
    /// [`SceneError::Cycle`], [`SceneError::MissingParent`] or
    /// [`SceneError::ParentWithoutTransform`], each naming the entity at fault;
    /// or [`SceneError::Alias`] if the world refuses a query. Nothing is written
    /// when the hierarchy is refused — a half-propagated frame is a worse thing
    /// to hand a hash than an unpropagated one.
    pub fn propagate(&mut self, world: &mut World) -> Result<Propagated, SceneError> {
        self.gather(world)?;
        self.reorder(world)?;
        let composed = self.compose();
        let written = self.write_back(world)?;
        self.force = false;
        Ok(Propagated {
            nodes: self.nodes.len(),
            composed,
            written,
        })
    }

    /// Passes 1 and 2: every transform, then every node, into the dense table.
    fn gather(&mut self, world: &World) -> Result<(), SceneError> {
        for slot in &mut self.slots {
            slot.present = false;
            slot.is_node = false;
            slot.changed = false;
        }
        self.nodes.clear();

        let renderables = Query::<&Renderable>::new()?;
        let slots = &mut self.slots;
        world.each_ref(&renderables, |entity, r: &Renderable| {
            let slot = slot_mut(slots, entity);
            slot.world = Transform {
                position: r.position,
                rotation: r.rotation,
            };
            slot.present = true;
        });
        let models = Query::<&Model>::new()?;
        world.each_ref(&models, |entity, m: &Model| {
            // An entity carrying both is written from both and read from one;
            // the two agree because this crate is what wrote them last frame.
            let slot = slot_mut(slots, entity);
            slot.world = Transform {
                position: m.position,
                rotation: m.rotation,
            };
            slot.present = true;
        });

        let query = Query::<&Node>::new()?;
        let nodes = &mut self.nodes;
        world.each_ref(&query, |entity, node: &Node| {
            let slot = slot_mut(slots, entity);
            slot.present = true;
            slot.is_node = true;
            nodes.push((entity, *node));
        });

        // A root's world transform arrived from the game in the passes above,
        // so its change is detectable here and nowhere else.
        for slot in &mut self.slots {
            if slot.present && !slot.is_node {
                slot.changed = slot.world != slot.world_prev;
                slot.world_prev = slot.world;
            }
        }

        // Presence is folded into the fingerprint, not just parentage: a parent
        // that was despawned leaves every `Node` byte-identical, and without
        // this the stale order would be reused and the subtree would quietly
        // keep composing against last frame's transform.
        let mut shape = 0xcbf2_9ce4_8422_2325_u64;
        for &(entity, node) in &self.nodes {
            let present = self
                .slots
                .get(node.parent.index() as usize)
                .is_some_and(|s| s.generation == node.parent.generation() && s.present);
            shape = mix(
                mix(shape, entity.to_bits()),
                node.parent.to_bits() ^ u64::from(present),
            );
        }
        if shape != self.shape {
            self.shape = shape;
            self.shape_order.clear();
            self.force = true;
        }
        Ok(())
    }

    /// Rebuild the parent-before-child order, if this frame's shape asked for
    /// one. Also where a cycle, a dead parent and a transformless parent are
    /// caught: all three are properties of the *shape*, so the frames that
    /// reuse the order pay nothing for the checks.
    fn reorder(&mut self, world: &World) -> Result<(), SceneError> {
        if !self.shape_order.is_empty() || self.nodes.is_empty() {
            return Ok(());
        }
        self.node_at.clear();
        self.node_at.resize(self.slots.len(), u32::MAX);
        for (position, (entity, _)) in self.nodes.iter().enumerate() {
            self.node_at[entity.index() as usize] = position as u32;
        }
        self.visit.clear();
        self.visit.resize(self.nodes.len(), Visit::Unvisited);
        self.shape_order.reserve(self.nodes.len());

        for start in 0..self.nodes.len() as u32 {
            if self.visit[start as usize] != Visit::Unvisited {
                continue;
            }
            self.stack.clear();
            let mut current = start;
            loop {
                match self.visit[current as usize] {
                    // Already ordered: everything above it is too, so this
                    // chain is resolved from here up.
                    Visit::Done => break,
                    Visit::OnStack => {
                        return Err(SceneError::Cycle(self.nodes[current as usize].0));
                    }
                    Visit::Unvisited => {}
                }
                self.visit[current as usize] = Visit::OnStack;
                self.stack.push(current);
                let (child, node) = self.nodes[current as usize];
                let parent = node.parent;
                if !world.is_alive(parent) {
                    return Err(SceneError::MissingParent { child, parent });
                }
                let present = self
                    .slots
                    .get(parent.index() as usize)
                    .is_some_and(|s| s.generation == parent.generation() && s.present);
                if !present {
                    return Err(SceneError::ParentWithoutTransform { child, parent });
                }
                match self.node_at.get(parent.index() as usize).copied() {
                    Some(at) if at != u32::MAX => current = at,
                    // The parent is a root, so the chain ends here.
                    _ => break,
                }
            }
            // The stack holds child-first, highest-ancestor-last. Popping walks
            // it ancestor-first, which is exactly the order to emit — no
            // reverse, and chains are emitted contiguously.
            while let Some(index) = self.stack.pop() {
                self.visit[index as usize] = Visit::Done;
                self.shape_order.push(index);
            }
        }
        Ok(())
    }

    /// Pass 3: the composes, over flat arrays, touching no ECS storage.
    fn compose(&mut self) -> usize {
        let mut composed = 0;
        for position in 0..self.shape_order.len() {
            let index = self.shape_order[position] as usize;
            let (entity, node) = self.nodes[index];
            let local = Transform {
                position: node.position,
                rotation: node.rotation,
            };
            let parent = node.parent.index() as usize;
            let me = entity.index() as usize;
            if !self.force && !self.slots[parent].changed && local == self.slots[me].local_prev {
                self.slots[me].changed = false;
                continue;
            }
            let world = self.slots[parent].world.compose(&local);
            let slot = &mut self.slots[me];
            slot.local_prev = local;
            // Compared against last frame's rather than set unconditionally: a
            // parent that moved and moved back leaves its children where they
            // were, and their own children should not recompose for it.
            slot.changed = world != slot.world_prev;
            slot.world = world;
            slot.world_prev = world;
            composed += 1;
        }
        composed
    }

    /// Pass 4: the changed transforms back into the render protocol.
    fn write_back(&mut self, world: &mut World) -> Result<usize, SceneError> {
        let mut written = 0;
        let slots = &self.slots;
        let renderables = Query::<(&Node, &mut Renderable)>::new()?;
        world.each(&renderables, |entity, (_, r): (&Node, &mut Renderable)| {
            let slot = &slots[entity.index() as usize];
            if slot.changed {
                r.position = slot.world.position;
                r.rotation = slot.world.rotation;
                written += 1;
            }
        });
        let models = Query::<(&Node, &mut Model)>::new()?;
        world.each(&models, |entity, (_, m): (&Node, &mut Model)| {
            let slot = &slots[entity.index() as usize];
            if slot.changed {
                m.position = slot.world.position;
                m.rotation = slot.world.rotation;
                written += 1;
            }
        });
        Ok(written)
    }
}

/// The row for `entity`, grown on demand and reset when the index was recycled.
fn slot_mut(slots: &mut Vec<Slot>, entity: Entity) -> &mut Slot {
    let index = entity.index() as usize;
    if index >= slots.len() {
        slots.resize(index + 1, Slot::EMPTY);
    }
    let slot = &mut slots[index];
    if slot.generation != entity.generation() {
        *slot = Slot::EMPTY;
        slot.generation = entity.generation();
    }
    slot
}

/// FNV-1a over one word. The fingerprint only has to notice a *different* tree
/// and is recomputed from scratch every frame, so a collision costs one frame
/// of a stale order for an identically-shaped hierarchy — the same order.
const fn mix(hash: u64, value: u64) -> u64 {
    let mut hash = hash ^ value;
    hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    hash ^= hash >> 29;
    hash
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn world_with_root(position: sim::DVec3) -> (World, Entity) {
        let mut world = World::new();
        world.register::<Renderable>().unwrap();
        world.register::<Node>().unwrap();
        let root = world.spawn();
        world.insert(root, boxed(position)).unwrap();
        (world, root)
    }

    fn boxed(position: sim::DVec3) -> Renderable {
        Renderable::boxed(position, sim::Vec3::splat(0.5), 0x00ff_ffff)
    }

    /// Attach a child at `offset` from `parent`, drawn as a box.
    fn attach(world: &mut World, parent: Entity, offset: sim::DVec3) -> Entity {
        let child = world.spawn();
        world.insert(child, Node::at(parent, offset)).unwrap();
        world.insert(child, boxed(sim::DVec3::ZERO)).unwrap();
        child
    }

    fn position_of(world: &World, entity: Entity) -> sim::DVec3 {
        world.get::<Renderable>(entity).unwrap().position
    }

    #[test]
    fn a_child_lands_at_its_parents_position_plus_its_own_offset() {
        let (mut world, root) = world_with_root(sim::DVec3::new(10.0, 0.0, 0.0));
        let child = attach(&mut world, root, sim::DVec3::new(0.0, 2.0, 0.0));

        let mut hierarchy = Hierarchy::new();
        let done = hierarchy.propagate(&mut world).unwrap();
        assert_eq!(done.nodes, 1);
        assert_eq!(done.composed, 1);
        assert_eq!(position_of(&world, child), sim::DVec3::new(10.0, 2.0, 0.0));
    }

    #[test]
    fn a_parents_rotation_swings_its_children_rather_than_sliding_them() {
        let (mut world, root) = world_with_root(sim::DVec3::ZERO);
        let child = attach(&mut world, root, sim::DVec3::new(1.0, 0.0, 0.0));
        let mut hierarchy = Hierarchy::new();
        hierarchy.propagate(&mut world).unwrap();

        // A quarter turn about +Y takes +X to -Z.
        world.get_mut::<Renderable>(root).unwrap().rotation =
            sim::DQuat::from_axis_angle(sim::DVec3::Y, core::f64::consts::FRAC_PI_2);
        hierarchy.propagate(&mut world).unwrap();
        let at = position_of(&world, child);
        assert!(at.x.abs() < 1e-12, "{at:?}");
        assert!((at.z + 1.0).abs() < 1e-12, "{at:?}");
    }

    #[test]
    fn a_grandchild_follows_the_root_through_two_composes() {
        let (mut world, root) = world_with_root(sim::DVec3::ZERO);
        let child = attach(&mut world, root, sim::DVec3::new(1.0, 0.0, 0.0));
        let grandchild = attach(&mut world, child, sim::DVec3::new(1.0, 0.0, 0.0));

        let mut hierarchy = Hierarchy::new();
        hierarchy.propagate(&mut world).unwrap();
        assert_eq!(
            position_of(&world, grandchild),
            sim::DVec3::new(2.0, 0.0, 0.0)
        );

        world.get_mut::<Renderable>(root).unwrap().position = sim::DVec3::new(0.0, 5.0, 0.0);
        let moved = hierarchy.propagate(&mut world).unwrap();
        assert_eq!(moved.composed, 2, "the whole chain, and only it");
        assert_eq!(
            position_of(&world, grandchild),
            sim::DVec3::new(2.0, 5.0, 0.0)
        );
    }

    #[test]
    fn a_still_hierarchy_composes_nothing_and_writes_nothing() {
        let (mut world, root) = world_with_root(sim::DVec3::ZERO);
        for i in 0..8 {
            attach(&mut world, root, sim::DVec3::new(f64::from(i), 0.0, 0.0));
        }
        let mut hierarchy = Hierarchy::new();
        assert_eq!(hierarchy.propagate(&mut world).unwrap().composed, 8);

        let second = hierarchy.propagate(&mut world).unwrap();
        assert_eq!(second.nodes, 8, "still scanned");
        assert_eq!(second.composed, 0, "and nothing recomposed");
        assert_eq!(second.written, 0);
    }

    #[test]
    fn moving_one_hub_recomposes_its_subtree_and_no_other() {
        let (mut world, left) = world_with_root(sim::DVec3::ZERO);
        let right = world.spawn();
        world
            .insert(right, boxed(sim::DVec3::new(100.0, 0.0, 0.0)))
            .unwrap();
        for _ in 0..3 {
            attach(&mut world, left, sim::DVec3::X);
        }
        for _ in 0..5 {
            attach(&mut world, right, sim::DVec3::X);
        }

        let mut hierarchy = Hierarchy::new();
        hierarchy.propagate(&mut world).unwrap();
        world.get_mut::<Renderable>(left).unwrap().position = sim::DVec3::new(0.0, 1.0, 0.0);
        let moved = hierarchy.propagate(&mut world).unwrap();
        assert_eq!(moved.composed, 3, "only the left hub's children");
        assert_eq!(moved.written, 3);
    }

    #[test]
    fn a_cycle_is_refused_by_an_entity_on_it() {
        let (mut world, root) = world_with_root(sim::DVec3::ZERO);
        let child = attach(&mut world, root, sim::DVec3::X);
        world.insert(root, Node::at(child, sim::DVec3::X)).unwrap();

        let mut hierarchy = Hierarchy::new();
        let err = hierarchy.propagate(&mut world).unwrap_err();
        assert!(matches!(err, SceneError::Cycle(_)), "{err}");
    }

    #[test]
    fn a_despawned_parent_is_refused_by_name_rather_than_left_composing() {
        let (mut world, root) = world_with_root(sim::DVec3::ZERO);
        let child = attach(&mut world, root, sim::DVec3::X);
        let mut hierarchy = Hierarchy::new();
        hierarchy.propagate(&mut world).unwrap();

        // The `Node` is byte-identical after this: only the parent's presence
        // changed, which is why presence is in the shape fingerprint.
        world.despawn(root);
        match hierarchy.propagate(&mut world).unwrap_err() {
            SceneError::MissingParent { child: c, parent } => {
                assert_eq!(c, child);
                assert_eq!(parent, root);
            }
            other => panic!("{other}"),
        }
    }

    #[test]
    fn a_parent_with_no_transform_is_refused_rather_than_treated_as_the_origin() {
        let mut world = World::new();
        world.register::<Renderable>().unwrap();
        world.register::<Node>().unwrap();
        let pivot = world.spawn(); // alive, but carries nothing
        let child = attach(&mut world, pivot, sim::DVec3::X);

        let mut hierarchy = Hierarchy::new();
        match hierarchy.propagate(&mut world).unwrap_err() {
            SceneError::ParentWithoutTransform { child: c, parent } => {
                assert_eq!(c, child);
                assert_eq!(parent, pivot);
            }
            other => panic!("{other}"),
        }
    }

    #[test]
    fn a_model_is_propagated_the_same_way_a_renderable_is() {
        let mut world = World::new();
        world.register::<Model>().unwrap();
        world.register::<Node>().unwrap();
        let root = world.spawn();
        world
            .insert(root, Model::at("hub", sim::DVec3::new(4.0, 0.0, 0.0)))
            .unwrap();
        let child = world.spawn();
        world
            .insert(child, Node::at(root, sim::DVec3::new(0.0, 3.0, 0.0)))
            .unwrap();
        world
            .insert(child, Model::at("bit", sim::DVec3::ZERO))
            .unwrap();

        let mut hierarchy = Hierarchy::new();
        hierarchy.propagate(&mut world).unwrap();
        assert_eq!(
            world.get::<Model>(child).unwrap().position,
            sim::DVec3::new(4.0, 3.0, 0.0)
        );
    }

    #[test]
    fn a_world_with_no_nodes_is_a_scan_and_nothing_else() {
        let (mut world, _) = world_with_root(sim::DVec3::ZERO);
        let mut hierarchy = Hierarchy::new();
        assert_eq!(
            hierarchy.propagate(&mut world).unwrap(),
            Propagated::default()
        );
    }
}
