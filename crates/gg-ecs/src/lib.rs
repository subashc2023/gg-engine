//! `gg-ecs` (§4.2) — our archetype ECS, built before any sim code exists to be
//! written against something else.
//!
//! Storage and queries only. **No scheduler, no system graph, no reflection, no
//! serialization** (§7, standing non-goals): the frame loop in §4.1 *is* the
//! schedule. The public surface is deliberately small and machine-held — a
//! `cargo-public-api` baseline (§5.10) makes widening it a reviewed act rather
//! than a drive-by convenience. Internals are explicitly fair game for rework;
//! the canonical state hash is what makes rework falsifiable, because it
//! measures the simulation rather than the layout.
//!
//! # Writing a component
//!
//! ```
//! use gg_ecs::{Component, World};
//! use gg_math::sim;
//!
//! #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
//! #[component(id = "position")]
//! #[repr(C)]
//! struct Position {
//!     p: sim::DVec3,
//! }
//!
//! let mut world = World::new();
//! let e = world.spawn();
//! world.insert(e, Position { p: sim::DVec3::new(1.0, 0.0, 0.0) })?;
//! # Ok::<(), gg_ecs::WorldError>(())
//! ```
//!
//! The `id` is **persisted identity** and is mandatory. It is not the Rust type
//! path, so moving or renaming `Position` costs nothing and changing the id is
//! the visible act a schema change ought to be (§4.2).
//!
//! # Components are `Pod`, and what to do about it
//!
//! A component must be `bytemuck::Pod` — flat, `repr(C)`, padding-free, no
//! pointers. That is not a hashing rule (the [`StateHash`] protocol is happy
//! with a `Vec`); it is load-bearing three other ways: the reload boundary
//! hands raw column pointers across a C ABI (§4.2.2), heap-backed component
//! data would put a leaked dylib's allocations inside host-owned state, and
//! offset-mapped migration is only a column copy because columns are flat.
//!
//! It is a real ergonomic tax, so the shapes it forces are stated here rather
//! than improvised per component (§4.2.1):
//!
//! - **Enums** → a `#[repr(transparent)]` newtype over `u8`/`u32` with
//!   validated accessors. A bare enum's discriminant representation is not
//!   pinned by the language, so its bytes are not a contract.
//! - **Flags** → an explicit bit set over a fixed-width integer, never
//!   `Vec<bool>`.
//! - **Dynamic data** → a fixed-capacity inline array when the bound is known,
//!   otherwise an index into a host-owned arena.
//! - **Anything genuinely growable** → a [`side_table`], which is hashed in the
//!   canonical pass without being `Pod`. Strategy-layer state — fleets,
//!   missions, inventories — is *expected* to live there, with components
//!   holding handles.
//!
//! The derive rejects the rest at compile time, naming the field: `usize`, raw
//! pointers, references, and randomly-ordered containers all fail before they
//! can reach a hash (§4.2.1).

#![deny(missing_docs)]

// `#[derive(Component)]` emits `::gg_ecs::` paths, which is what lets a game
// crate name one crate (§3). The alias is how the derive also works *inside*
// this crate, so the boundary's own components are derived rather than
// hand-written — a hand-written `FIELDS` is a schema hash waiting to drift.
extern crate self as gg_ecs;

pub mod archetype;
pub mod boundary;
pub mod chunked;
mod column;
pub mod component;
pub mod entity;
pub mod hash;
pub mod nan_scan;
pub mod query;
pub mod registry;
pub mod side_table;
pub mod state_hash;
pub mod view;
pub mod world;

pub use archetype::ArchetypeId;
pub use boundary::{GameWorld, host_api};
pub use component::{Component, FieldDesc, component_id, schema_hash};
pub use entity::{Entities, Entity};
pub use hash::{CanonicalHash, ComponentId, SchemaHash, SideTableId, StateHasher};
pub use nan_scan::{NanHit, NanSite, SideTableNan, UnclassifiedField};
pub use query::{Query, QueryData, ReadOnly};
pub use registry::{ComponentInfo, Registry, RegistryError};
pub use side_table::{SideTable, SideTableError};
pub use state_hash::StateHash;
pub use view::{AliasError, ArchetypeView, ColumnView, QueryAccess};
pub use world::save::{Save, SaveError};
pub use world::snapshot::{ComponentOutcome, MigrationReport, Snapshot, SnapshotError};
pub use world::{World, WorldError};

/// The derives (§4.2, §4.2.1), re-exported so game crates name `gg-ecs` alone
/// and the §4.2.2 boundary deny pin stays a three-crate list.
pub use gg_ecs_derive::{Component, SideTable, StateHash};
