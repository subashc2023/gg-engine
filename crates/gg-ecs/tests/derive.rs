//! `#[derive(Component)]` / `#[derive(StateHash)]` behaviour (§4.2, §4.2.1).
//!
//! The rejection half lives in `derive_rejects.rs` as `trybuild` cases — a
//! compile *failure* cannot be asserted from inside a compiling test.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::{Component, Entity, StateHash, StateHasher, component_id, schema_hash};
use gg_math::sim;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "position")]
#[repr(C)]
struct Position {
    p: sim::DVec3,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "velocity")]
#[repr(C)]
struct Velocity {
    v: sim::DVec3,
}

/// Same layout as `Position`, different declared id — the pair that proves the
/// schema hash is not layout alone.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "anchor")]
#[repr(C)]
struct Anchor {
    p: sim::DVec3,
}

/// A marker component: no fields, hashes as its id.
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "frozen")]
#[repr(C)]
struct Frozen;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "target")]
#[repr(C)]
struct Target {
    who: Entity,
    priority: u32,
    _pad: u32,
}

/// The §4.2.1 side-table shape: protocol-hashable without being `Pod`.
#[derive(StateHash)]
struct ProductionQueue {
    owner: Entity,
    orders: Vec<u32>,
    label: String,
}

mod moved {
    //! The same component, declared in another module. §4.2's promise is that
    //! this changes neither the id nor the schema hash.
    use gg_ecs::Component;
    use gg_math::sim;

    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
    #[component(id = "position")]
    #[repr(C)]
    pub struct PositionRenamedAndMoved {
        pub p: sim::DVec3,
    }
}

fn hash_of(v: &impl StateHash) -> gg_ecs::CanonicalHash {
    let mut h = StateHasher::canonical();
    v.state_hash(&mut h);
    h.finish_canonical()
}

#[test]
fn moving_and_renaming_a_component_changes_neither_id_nor_schema() {
    // The whole point of declaring ids explicitly (§4.2): source identity and
    // persisted identity are decoupled, so agents can refactor freely.
    assert_eq!(
        component_id::<Position>(),
        component_id::<moved::PositionRenamedAndMoved>()
    );
    assert_eq!(
        schema_hash::<Position>(),
        schema_hash::<moved::PositionRenamedAndMoved>()
    );
}

#[test]
fn distinct_ids_produce_distinct_component_ids_and_schemas() {
    assert_ne!(component_id::<Position>(), component_id::<Velocity>());
    // Anchor is byte-identical to Position in layout; only the id differs.
    assert_eq!(
        core::mem::size_of::<Anchor>(),
        core::mem::size_of::<Position>()
    );
    assert_ne!(
        schema_hash::<Anchor>(),
        schema_hash::<Position>(),
        "schema hash must cover the declared id, or migration could confuse \
         two same-layout components"
    );
}

#[test]
fn field_descriptors_carry_real_offsets_and_sizes() {
    let fields = <Target as Component>::FIELDS;
    assert_eq!(fields.len(), 3);
    assert_eq!(fields[0].name, "who");
    assert_eq!(fields[0].ty, "Entity");
    assert_eq!(fields[0].offset, 0);
    assert_eq!(fields[0].size, 8);
    assert_eq!(fields[1].name, "priority");
    assert_eq!(fields[1].offset, 8);
    // These are §4.2.2's migration input, so they must describe the real type.
    assert_eq!(
        fields.iter().map(|f| f.size).sum::<usize>(),
        core::mem::size_of::<Target>(),
        "a gap here means the component has padding, which Pod should forbid"
    );
}

#[test]
fn renaming_a_field_moves_the_schema_hash() {
    // Layout-identical, name-different: migration matches by name (§4.2.2), so
    // this *must* be a schema change.
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
    #[component(id = "position")]
    #[repr(C)]
    struct PositionFieldRenamed {
        origin: sim::DVec3,
    }
    assert_eq!(
        component_id::<Position>(),
        component_id::<PositionFieldRenamed>()
    );
    assert_ne!(
        schema_hash::<Position>(),
        schema_hash::<PositionFieldRenamed>()
    );
}

#[test]
fn marker_components_are_legal_and_hash_as_nothing() {
    assert_eq!(<Frozen as Component>::FIELDS.len(), 0);
    assert_eq!(core::mem::size_of::<Frozen>(), 0);
    // Two markers hash identically as *values*; their identity is their id,
    // which the archetype carries, not the payload.
    assert_eq!(hash_of(&Frozen), hash_of(&Frozen));
}

#[test]
fn the_derived_encoding_is_field_by_field_not_raw_memory() {
    let t = Target {
        who: Entity::from_bits((1u64 << 32) | 5),
        priority: 9,
        _pad: 0,
    };
    let mut manual = StateHasher::canonical();
    manual.u64(t.who.to_bits());
    manual.u32(t.priority);
    manual.u32(t._pad);
    assert_eq!(hash_of(&t), manual.finish_canonical());
}

#[test]
fn side_tables_hash_without_being_pod() {
    let a = ProductionQueue {
        owner: Entity::from_bits((1u64 << 32) | 2),
        orders: vec![7, 8, 9],
        label: "yard".into(),
    };
    let b = ProductionQueue {
        owner: a.owner,
        orders: vec![7, 8],
        label: "yard".into(),
    };
    assert_ne!(hash_of(&a), hash_of(&b));
    assert_eq!(hash_of(&a), hash_of(&a));
}
