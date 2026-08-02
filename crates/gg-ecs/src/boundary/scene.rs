//! How one entity's transform relates to another's — the §4.7 hierarchy's
//! game-facing half.
//!
//! [`present`](super::present) says what a thing looks like *in world space*.
//! This says where that world space came from: an entity carrying [`Node`] has
//! its `Renderable`/`Model` position and rotation **written** each tick by
//! `gg-scene`, composed from its parent's. One that does not carries its own.
//! Presence in the archetype is therefore the whole answer to "is this
//! transform derived?", which is the question the ECS is fastest at.
//!
//! Same three-legal-homes argument as the render protocol (see
//! [`present`](super::present)): a game may not link `gg-scene`, so the *link*
//! is a component and the *machinery* is the host's.

use gg_abi::entity::Entity;
use gg_math::sim;

use crate::Component;

/// A transform expressed relative to another entity's.
///
/// The parent must exist and the graph must be acyclic; `gg-scene` refuses
/// either failure by name rather than looping or silently dropping a subtree.
/// A root is spelled by *not* carrying this — there is no null parent, so a
/// dangling reference cannot be written by accident.
///
/// # Why there is no scale
///
/// Composing non-uniform scale down a hierarchy produces a matrix that is no
/// longer translation-rotation-scale — a rotated child under a non-uniformly
/// scaled parent shears, and no TRS triple can express the result. So size
/// stays where the game already spells it, in `Renderable::half_extent` and
/// `Model::scale`. An assembly that must scale as a unit is a *scene* (§4.6),
/// which does compose scale, and does it at import where the question is
/// answered once instead of every tick.
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "gg.node")]
#[repr(C)]
pub struct Node {
    /// Whose space [`position`](Node::position) and [`rotation`](Node::rotation)
    /// are in.
    pub parent: Entity,
    /// Offset from the parent, in the parent's rotated frame. `f64` for the
    /// same reason a world position is (§1.4): a child of something planetary
    /// is planetary.
    pub position: sim::DVec3,
    /// Orientation relative to the parent's.
    pub rotation: sim::DQuat,
}

impl Node {
    /// Attached, unrotated, at an offset — the common case.
    #[must_use]
    pub fn at(parent: Entity, position: sim::DVec3) -> Self {
        Node {
            parent,
            position,
            rotation: sim::DQuat::IDENTITY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_node_is_flat_and_padding_free() {
        // 8 + 24 + 32, in that order. A field of any other size would need
        // filler invented to satisfy `Pod` — see the doc on why scale is absent.
        assert_eq!(size_of::<Node>(), 64);
        assert_eq!(align_of::<Node>(), 8);
    }

    #[test]
    fn attaching_leaves_the_child_unrotated() {
        let parent = Entity::new(3, 1);
        let node = Node::at(parent, sim::DVec3::new(0.0, 2.0, 0.0));
        assert_eq!(node.parent, parent);
        assert_eq!(node.rotation, sim::DQuat::IDENTITY);
    }
}
