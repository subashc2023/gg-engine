//! The component registry (§4.2): stable id ↔ layout, with collision checking.
//!
//! Registration is where a declared id stops being a string literal and becomes
//! the world's persisted identity, so it is also where every check that would
//! otherwise be an audit finding gets made: id collisions, over-aligned
//! components, and — because `Component` cannot be implemented without them —
//! the `Pod` and schema-hash requirements of §4.2.1 hazard 4.

use crate::column::MAX_ALIGN;
use crate::component::{Component, FieldDesc, component_id, schema_hash};
use crate::hash::{ComponentId, SchemaHash, StateHasher};

/// Encode one component's bytes through the explicit protocol (§4.2.1).
///
/// Captured at registration, which is the only place the concrete type is
/// known. `bytes` is exactly one row of that component's column.
pub type ProtocolHashFn = fn(bytes: &[u8], h: &mut StateHasher);

fn protocol_hash_of<T: Component>(bytes: &[u8], h: &mut StateHasher) {
    crate::StateHash::state_hash(bytemuck::from_bytes::<T>(bytes), h);
}

/// Everything the type-erased side of the ECS needs about one component.
///
/// No `PartialEq`: it would compare `protocol_hash`, and function-pointer
/// equality is not meaningful (identical monomorphizations may be merged).
/// Compare `id` and `schema`, which is what callers actually mean.
#[derive(Clone, Debug)]
pub struct ComponentInfo {
    pub id: ComponentId,
    pub schema: SchemaHash,
    /// The declared id string — persisted identity, and what a migration report
    /// names.
    pub declared_id: &'static str,
    /// The Rust type name. Diagnostics only.
    pub type_name: &'static str,
    pub size: usize,
    pub align: usize,
    pub fields: &'static [FieldDesc],
    /// The protocol encoding, kept so CI can prove the raw-bytes fast path
    /// agrees with it (§4.2.1). Not used by [`World::canonical_hash`] itself.
    ///
    /// [`World::canonical_hash`]: crate::World::canonical_hash
    pub protocol_hash: ProtocolHashFn,
}

/// Registration failures. Each is a startup error, never a silent remap
/// (§4.2) — a component whose identity is wrong corrupts saves rather than
/// crashing, so it must not be recoverable.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error(
        "component id collision: `{first_type}` (id \"{first_declared}\") and `{second_type}` \
         (id \"{second_declared}\") both hash to {id:?}. Change one declared id — they are the \
         world's persisted identities and cannot be shared."
    )]
    IdCollision {
        id: ComponentId,
        first_type: &'static str,
        first_declared: &'static str,
        second_type: &'static str,
        second_declared: &'static str,
    },
    #[error(
        "component id \"{declared}\" is declared by two types, `{first_type}` and \
         `{second_type}`. A declared id is persisted identity: two types claiming one id would \
         load each other's saved state."
    )]
    DuplicateDeclaration {
        declared: &'static str,
        first_type: &'static str,
        second_type: &'static str,
    },
    #[error(
        "component `{type_name}` needs {align}-byte alignment; columns provide {MAX_ALIGN}. \
         Over-aligned components are a render-side concern on the far side of the §1.4 membrane — \
         sim state has no reason to be SIMD-aligned."
    )]
    OverAligned {
        type_name: &'static str,
        align: usize,
    },
}

/// The registry. Entries are kept **sorted by [`ComponentId`]**, which is what
/// makes every downstream walk over components deterministic without anyone
/// having to remember to sort (§4.2.1 hazard 3).
#[derive(Clone, Default)]
pub struct Registry {
    sorted: Vec<ComponentInfo>,
}

impl Registry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.sorted.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }

    /// Register `T`, or return its existing entry.
    ///
    /// Re-registering the same type is a no-op: `gg-ecs` cannot know whether a
    /// caller registered eagerly at startup or lazily on first use, and both
    /// are legitimate.
    pub fn register<T: Component>(&mut self) -> Result<ComponentId, RegistryError> {
        let id = component_id::<T>();
        let info = ComponentInfo {
            id,
            schema: schema_hash::<T>(),
            declared_id: T::DECLARED_ID,
            type_name: T::TYPE_NAME,
            size: size_of::<T>(),
            align: align_of::<T>(),
            fields: T::FIELDS,
            protocol_hash: protocol_hash_of::<T>,
        };

        if info.align > MAX_ALIGN {
            return Err(RegistryError::OverAligned {
                type_name: info.type_name,
                align: info.align,
            });
        }

        match self.sorted.binary_search_by_key(&id, |e| e.id) {
            Ok(at) => {
                let existing = &self.sorted[at];
                if existing.type_name == info.type_name {
                    return Ok(id); // idempotent re-registration
                }
                // Same 64-bit id from two types. Which of the two failures it
                // is depends on whether the *declared strings* also match:
                // equal strings mean two types claimed one identity, different
                // strings mean the hash collided. The remedies differ, so the
                // messages do too.
                Err(if existing.declared_id == info.declared_id {
                    RegistryError::DuplicateDeclaration {
                        declared: info.declared_id,
                        first_type: existing.type_name,
                        second_type: info.type_name,
                    }
                } else {
                    RegistryError::IdCollision {
                        id,
                        first_type: existing.type_name,
                        first_declared: existing.declared_id,
                        second_type: info.type_name,
                        second_declared: info.declared_id,
                    }
                })
            }
            Err(at) => {
                self.sorted.insert(at, info);
                Ok(id)
            }
        }
    }

    #[must_use]
    pub fn get(&self, id: ComponentId) -> Option<&ComponentInfo> {
        self.sorted
            .binary_search_by_key(&id, |e| e.id)
            .ok()
            .map(|at| &self.sorted[at])
    }

    /// Registered components, ascending by id — the canonical walk order.
    pub fn iter(&self) -> impl Iterator<Item = &ComponentInfo> + '_ {
        self.sorted.iter()
    }

    /// Fold the registry into the canonical hash (§4.2.1).
    ///
    /// Schemas are hashed, not just ids: two worlds whose components share ids
    /// but differ in layout are not the same world, and saying so here is what
    /// makes a reload's schema check and the hash agree about what "changed"
    /// means.
    pub fn hash_into(&self, h: &mut StateHasher) {
        h.u64(self.sorted.len() as u64);
        for info in &self.sorted {
            h.u64(info.id.get());
            h.u128(info.schema.get());
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::component::FieldDesc;

    /// Hand-written `Component` impls, because the collision cases cannot be
    /// produced through the derive — it would need two types to hash alike.
    macro_rules! fake_component {
        ($name:ident, $declared:expr) => {
            #[derive(Clone, Copy)]
            #[repr(C)]
            struct $name {
                v: u32,
            }
            // SAFETY: `repr(C)` single `u32` — no padding, all bit patterns valid.
            unsafe impl bytemuck::Zeroable for $name {}
            // SAFETY: as above, plus `Copy`.
            unsafe impl bytemuck::Pod for $name {}
            impl crate::StateHash for $name {
                fn state_hash(&self, h: &mut StateHasher) {
                    h.u32(self.v);
                }
            }
            impl Component for $name {
                const DECLARED_ID: &'static str = $declared;
                const TYPE_NAME: &'static str = stringify!($name);
                const FIELDS: &'static [FieldDesc] = &[FieldDesc {
                    name: "v",
                    ty: "u32",
                    offset: 0,
                    size: 4,
                }];
            }
        };
    }

    fake_component!(Alpha, "alpha");
    fake_component!(Beta, "beta");
    fake_component!(AlphaTwin, "alpha");

    #[test]
    fn registration_is_idempotent_and_sorted() {
        let mut r = Registry::new();
        let a = r.register::<Alpha>().unwrap();
        let b = r.register::<Beta>().unwrap();
        assert_eq!(r.register::<Alpha>().unwrap(), a);
        assert_eq!(r.len(), 2);
        let ids: Vec<ComponentId> = r.iter().map(|i| i.id).collect();
        assert!(
            ids.windows(2).all(|w| w[0] < w[1]),
            "registry must be sorted"
        );
        assert!(ids.contains(&a) && ids.contains(&b));
    }

    #[test]
    fn two_types_declaring_one_id_is_an_error_naming_both() {
        let mut r = Registry::new();
        r.register::<Alpha>().unwrap();
        let err = r.register::<AlphaTwin>().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("Alpha"), "{msg}");
        assert!(msg.contains("AlphaTwin"), "{msg}");
        assert!(msg.contains("alpha"), "{msg}");
    }

    #[test]
    fn an_over_aligned_component_is_refused_by_name() {
        #[derive(Clone, Copy)]
        #[repr(C, align(32))]
        struct Wide {
            v: [u64; 4],
        }
        // SAFETY: `repr(C, align(32))` over `[u64; 4]` is 32 bytes of payload
        // in a 32-byte allocation — no padding, all bit patterns valid.
        unsafe impl bytemuck::Zeroable for Wide {}
        // SAFETY: as above, plus `Copy`.
        unsafe impl bytemuck::Pod for Wide {}
        impl crate::StateHash for Wide {
            fn state_hash(&self, h: &mut StateHasher) {
                self.v.state_hash(h);
            }
        }
        impl Component for Wide {
            const DECLARED_ID: &'static str = "wide";
            const TYPE_NAME: &'static str = "Wide";
            const FIELDS: &'static [FieldDesc] = &[FieldDesc {
                name: "v",
                ty: "[u64; 4]",
                offset: 0,
                size: 32,
            }];
        }

        let err = Registry::new().register::<Wide>().unwrap_err();
        assert!(err.to_string().contains("Wide"), "{err}");
    }

    #[test]
    fn the_registry_hash_covers_schemas_not_only_ids() {
        let mut a = Registry::new();
        a.register::<Alpha>().unwrap();
        let mut b = Registry::new();
        b.register::<Beta>().unwrap();

        let mut ha = StateHasher::canonical();
        a.hash_into(&mut ha);
        let mut hb = StateHasher::canonical();
        b.hash_into(&mut hb);
        assert_ne!(ha.finish_canonical(), hb.finish_canonical());
    }
}
