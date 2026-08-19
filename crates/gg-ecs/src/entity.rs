//! Entity ids and their allocator (§4.2, determinism guarantee 2).
//!
//! # The allocation contract
//!
//! A documented API guarantee, not an implementation detail: the same sequence
//! of spawns and despawns yields the same ids on every run and every
//! architecture. It must be, because entity ids appear *inside* components as
//! references and the canonical hash sorts by entity id (§4.2.1) —
//! nondeterministic allocation would poison the hash even with perfect
//! iteration order.
//!
//! 1. **Fresh indices ascend from 0.** Empty freelist ⇒ next index is the
//!    current slot count.
//! 2. **The freelist is LIFO.** Despawn pushes, spawn pops the most recently
//!    freed. FIFO would delay reuse and surface stale-handle bugs more loudly,
//!    a real argument — but *specified* is the property that matters here, and
//!    LIFO keeps reused rows cache-warm.
//! 3. **Generation parity is liveness.** Odd = live, even = dead. Allocation
//!    bumps to odd, despawn bumps to even, so a slot's liveness needs no
//!    side table and iteration never scans the freelist. First allocation of
//!    an index yields generation 1, so all-zeroes is never live and a zeroed
//!    `Pod` field holding an [`Entity`] reads as dead rather than as entity 0.
//! 4. **A saturated index retires.** [`RETIRED`] is even, so it reads as dead,
//!    and it is never pushed to the freelist — the index leaves circulation
//!    instead of wrapping. Wrapping would eventually resurrect a stale handle
//!    as valid, the one failure this scheme exists to prevent.

use crate::hash::StateHasher;

/// The handle itself lives in `gg-abi` (§4.2.2): it crosses the reload boundary
/// both as an archetype match's entity slice and inside `Pod` component bytes,
/// which makes its layout a contract between two separately-compiled artifacts
/// rather than a storage detail. The *contract above* — how ids are handed out —
/// is still this module's, and it is the part determinism depends on.
pub use gg_abi::Entity;

/// Terminal generation for an index that has been despawned `u32::MAX / 2`
/// times. Even (⇒ dead by rule 3) and never re-issued (rule 4).
const RETIRED: u32 = u32::MAX - 1;

/// The entity allocator: one generation per index, plus the freelist.
///
/// Deliberately separate from storage. Archetypes move rows constantly; id
/// allocation must not, and keeping them apart means the storage rewrite §4.2
/// promises is fair game cannot perturb the id sequence the canonical hash
/// depends on.
#[derive(Clone, Default)]
pub struct Entities {
    /// `generations[i]` is index `i`'s generation — odd live, even dead,
    /// [`RETIRED`] terminal.
    generations: Vec<u32>,
    /// Free indices, LIFO. Retired indices are absent, permanently.
    free: Vec<u32>,
    /// Live count. Not derivable from `generations.len()`, which counts dead
    /// and retired slots too.
    live: u32,
}

impl Entities {
    /// An empty allocator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Live entity count.
    #[must_use]
    pub const fn len(&self) -> u32 {
        self.live
    }

    /// Whether nothing is live. Slots may still be allocated — this counts
    /// entities, not capacity.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Allocate, per the contract at the top of this module.
    ///
    /// # Panics
    ///
    /// If the index space is exhausted (`u32::MAX` slots). A panic rather than
    /// a `Result` because every caller would unwrap it: 4 billion indices is
    /// not a condition gameplay code recovers from, and §2's on-rails regime
    /// is the answer to scale, not a wider id.
    pub fn alloc(&mut self) -> Entity {
        self.live += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.generations[index as usize];
            *slot += 1; // even -> odd: live again
            return Entity::new(index, *slot);
        }
        assert!(
            self.generations.len() < u32::MAX as usize,
            "gg-ecs: entity index space exhausted (u32::MAX indices)"
        );
        let index = self.generations.len() as u32;
        self.generations.push(1);
        Entity::new(index, 1)
    }

    /// Free an entity. Returns whether it was alive — a double despawn is a
    /// reported no-op, not a panic, because despawn-during-iteration patterns
    /// legitimately produce duplicates.
    pub fn free(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        self.live -= 1;
        let slot = &mut self.generations[entity.index() as usize];
        *slot += 1; // odd -> even: dead
        if *slot != RETIRED {
            self.free.push(entity.index());
        }
        true
    }

    /// Whether the handle names a currently-live entity. Cheap and total: an
    /// out-of-range index, a stale generation, [`Entity::NONE`], and a
    /// hand-built even-generation handle all answer `false` without touching
    /// storage.
    #[must_use]
    pub fn is_alive(&self, entity: Entity) -> bool {
        entity.generation() % 2 == 1
            && self
                .generations
                .get(entity.index() as usize)
                .is_some_and(|&g| g == entity.generation())
    }

    /// Every live entity, ascending by index — the order the canonical hash
    /// walks (§4.2.1), and also ascending `Entity` order, since an index
    /// appears at most once.
    pub fn iter(&self) -> impl Iterator<Item = Entity> + '_ {
        self.generations
            .iter()
            .enumerate()
            .filter(|&(_, &g)| g % 2 == 1)
            .map(|(index, &generation)| Entity::new(index as u32, generation))
    }

    /// Freeze allocator state for a snapshot (§4.8).
    ///
    /// The freelist and the retired slots come along, not just the live ids:
    /// two allocators with identical live entities but different freelists hand
    /// out *different ids next tick*, so restoring one as the other would be a
    /// divergence one spawn later — the same reasoning that puts the freelist in
    /// the canonical hash.
    pub(crate) fn image(&self) -> EntitiesImage {
        EntitiesImage {
            generations: self.generations.clone(),
            free: self.free.clone(),
            live: self.live,
        }
    }

    /// Replace allocator state wholesale. Restores the id sequence exactly.
    ///
    /// # Errors
    ///
    /// If `image` breaks any rule at the top of this module. This is the only
    /// path that installs an allocator it did not build, and the image reaches it
    /// off disk from a previous process through a handoff that carries no
    /// checksum (§4.2.2) — so the rules are checked here rather than discovered
    /// later as a panic in [`alloc`](Self::alloc), as two live entities sharing
    /// one index, or as a wrapped generation reviving every stale handle.
    /// Nothing is written until every check has passed.
    pub(crate) fn restore(&mut self, image: &EntitiesImage) -> Result<(), crate::SnapshotError> {
        image.check()?;
        self.generations.clone_from(&image.generations);
        self.free.clone_from(&image.free);
        self.live = image.live;
        Ok(())
    }

    /// Fold allocator state into the canonical hash (§4.2.1).
    ///
    /// The freelist is hashed too, deliberately: two worlds with identical live
    /// entities but different freelists hand out *different ids next tick*, so
    /// they are not the same state. Omitting it would let a divergence hide for
    /// exactly one spawn.
    pub fn hash_into(&self, h: &mut StateHasher) {
        h.u32(self.live);
        h.u64(self.generations.len() as u64);
        for &g in &self.generations {
            h.u32(g);
        }
        h.u64(self.free.len() as u64);
        for &i in &self.free {
            h.u32(i);
        }
    }
}

/// Allocator state as a snapshot carries it (§4.8).
///
/// Public because [`Snapshot`](crate::Snapshot) is, but opaque: the fields are
/// the allocation contract's private business, and a caller that could edit them
/// could hand out a live id twice.
#[derive(Clone, Debug)]
pub struct EntitiesImage {
    pub(crate) generations: Vec<u32>,
    pub(crate) free: Vec<u32>,
    pub(crate) live: u32,
}

impl EntitiesImage {
    /// Live entities in the captured world.
    #[must_use]
    pub fn live(&self) -> u32 {
        self.live
    }

    /// [`Entities::is_alive`] against the *captured* allocator, for the
    /// consistency check a restore runs before it touches the world (§6 M81).
    pub(crate) fn is_live(&self, entity: Entity) -> bool {
        entity.generation() % 2 == 1
            && self
                .generations
                .get(entity.index() as usize)
                .is_some_and(|&g| g == entity.generation())
    }

    /// The allocation contract, against the image alone (§6 M81).
    ///
    /// Split out of [`Entities::restore`] so a snapshot's *structural* check can
    /// run it first: the row checks beside it read generations out of this
    /// image, and reporting "entity 3 is not live" about an image whose
    /// generation array is itself corrupt names the symptom instead of the
    /// cause.
    pub(crate) fn check(&self) -> Result<(), crate::SnapshotError> {
        let image = self;
        let bad = |detail: String| crate::SnapshotError::Allocator { detail };
        let mut live: u32 = 0;
        for (index, &generation) in image.generations.iter().enumerate() {
            if generation == u32::MAX {
                return Err(bad(format!(
                    "index {index} holds generation {generation}, which reads as live and would \
                     wrap to 0 on the next despawn — release builds do not trap it (rule 4)"
                )));
            }
            live += u32::from(generation % 2 == 1);
        }
        if live != image.live {
            return Err(bad(format!(
                "live count is {} but {live} generations are odd (rule 3)",
                image.live
            )));
        }

        // A seen-flag per slot rather than a sort: uniqueness is the check, and
        // the freelist's *order* is state the canonical hash covers (§4.2.1).
        let mut seen = vec![false; image.generations.len()];
        for &index in &image.free {
            let Some(slot) = seen.get_mut(index as usize) else {
                return Err(bad(format!(
                    "freelist names index {index}, past the {} slots that exist — the next alloc \
                     would index out of range",
                    image.generations.len()
                )));
            };
            if core::mem::replace(slot, true) {
                return Err(bad(format!(
                    "freelist names index {index} twice — two live entities would share one \
                     index, and one row would silently overwrite the other"
                )));
            }
            let generation = image.generations[index as usize];
            if generation % 2 == 1 {
                return Err(bad(format!(
                    "freelist names index {index}, which is live at generation {generation} \
                     (rule 3)"
                )));
            }
            if generation == RETIRED {
                return Err(bad(format!(
                    "freelist names index {index}, which is retired — a retired index leaves \
                     circulation permanently (rule 4)"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn allocation_sequence_is_the_documented_one() {
        // This test *is* the contract (§4.2 guarantee 2). A failure is either a
        // bug or a change to the documented discipline — and the second is a
        // Determinism Contract event, not a refactor.
        let mut e = Entities::new();
        let (a, b, c) = (e.alloc(), e.alloc(), e.alloc());
        assert_eq!((a.index(), a.generation()), (0, 1));
        assert_eq!((b.index(), b.generation()), (1, 1));
        assert_eq!((c.index(), c.generation()), (2, 1));

        // LIFO: freeing 0 then 1 hands 1 back first. Generations land on 3,
        // not 2 — parity, rule 3.
        assert!(e.free(a));
        assert!(e.free(b));
        let (d, f) = (e.alloc(), e.alloc());
        assert_eq!((d.index(), d.generation()), (1, 3));
        assert_eq!((f.index(), f.generation()), (0, 3));
    }

    #[test]
    fn stale_handles_are_dead_and_none_is_never_alive() {
        let mut e = Entities::new();
        let a = e.alloc();
        assert!(e.is_alive(a));
        assert!(e.free(a));
        assert!(!e.is_alive(a));

        let b = e.alloc();
        assert_eq!(b.index(), a.index());
        assert!(e.is_alive(b));
        assert!(
            !e.is_alive(a),
            "reused index must not revive the old handle"
        );

        assert!(!e.is_alive(Entity::NONE));
        assert!(!e.is_alive(Entity::from_bits((1u64 << 32) | 999)), "range");
        // A hand-built even generation is rejected on parity alone.
        assert!(!e.is_alive(Entity::from_bits(2u64 << 32)));
    }

    #[test]
    fn double_free_is_a_reported_no_op() {
        let mut e = Entities::new();
        let a = e.alloc();
        assert!(e.free(a));
        assert!(!e.free(a));
        assert_eq!(e.len(), 0);
    }

    #[test]
    fn a_saturated_index_retires_rather_than_wrapping() {
        let mut e = Entities::new();
        let a = e.alloc();
        // Fast-forward index 0 to the last live generation it can hold.
        e.generations[a.index() as usize] = RETIRED - 1;
        let aged = Entity::from_bits(u64::from(RETIRED - 1) << 32);
        assert!(e.is_alive(aged));
        assert!(e.free(aged));

        // Index 0 must not return: the next alloc takes a fresh index.
        assert_eq!(e.alloc().index(), 1);
        // And nothing can be alive at index 0 ever again.
        assert!(!e.is_alive(Entity::from_bits(u64::from(RETIRED) << 32)));
        assert!(e.iter().all(|x| x.index() != 0));
    }

    #[test]
    fn iteration_is_ascending_and_skips_freed_and_retired() {
        let mut e = Entities::new();
        let all: Vec<Entity> = (0..5).map(|_| e.alloc()).collect();
        assert!(e.free(all[1]));
        assert!(e.free(all[3]));
        let seen: Vec<u32> = e.iter().map(Entity::index).collect();
        assert_eq!(seen, vec![0, 2, 4]);
    }

    #[test]
    fn a_restored_allocator_must_satisfy_the_contract_it_was_not_built_by() {
        // `restore` is the one entry point that installs an allocator nobody
        // here built — the image arrives off disk from a previous process
        // through a checksum-free handoff (§4.2.2). Each case below is a real
        // failure of the rules at the top of this module, and the point is that
        // it is caught *here* rather than in the next `alloc`.
        let good = EntitiesImage {
            generations: vec![1, 2, 1],
            free: vec![1],
            live: 2,
        };
        assert!(Entities::new().restore(&good).is_ok());

        let refused = |image: EntitiesImage, needle: &str| {
            let mut allocator = Entities::new();
            let message = allocator.restore(&image).unwrap_err().to_string();
            assert!(message.contains(needle), "{message}");
            assert_eq!(allocator.len(), 0, "a refused image writes nothing");
        };

        refused(
            EntitiesImage {
                generations: vec![1, 2],
                free: vec![7],
                live: 1,
            },
            "past the 2 slots",
        );
        refused(
            EntitiesImage {
                generations: vec![2, 2],
                free: vec![0, 0],
                live: 0,
            },
            "twice",
        );
        refused(
            EntitiesImage {
                generations: vec![1, 2],
                free: vec![0],
                live: 1,
            },
            "live at generation 1",
        );
        refused(
            EntitiesImage {
                generations: vec![RETIRED],
                free: vec![0],
                live: 0,
            },
            "retired",
        );
        // Odd, so it reads as live; the next despawn wraps it to 0 under
        // release overflow rules, the index returns to the freelist, and the
        // next alloc reissues generation 1 — reviving every stale handle ever
        // minted for that index, which is the one thing RETIRED exists to stop.
        refused(
            EntitiesImage {
                generations: vec![u32::MAX],
                free: Vec::new(),
                live: 1,
            },
            "wrap to 0",
        );
        refused(
            EntitiesImage {
                generations: vec![1, 1],
                free: Vec::new(),
                live: 1,
            },
            "live count is 1",
        );
    }

    #[test]
    fn the_freelist_is_part_of_the_hashed_state() {
        // Identical live sets, different freelists: they diverge on the next
        // spawn, so they must not hash alike.
        let mut a = Entities::new();
        let mut b = Entities::new();
        for _ in 0..3 {
            a.alloc();
            b.alloc();
        }
        assert!(a.free(Entity::from_bits((1u64 << 32) | 1)));
        // Restore the live set so only the freelist differs.
        a.alloc();

        let mut ha = StateHasher::canonical();
        a.hash_into(&mut ha);
        let mut hb = StateHasher::canonical();
        b.hash_into(&mut hb);
        assert_ne!(ha.finish_canonical(), hb.finish_canonical());
    }

    #[test]
    fn a_long_churn_sequence_is_reproducible() {
        // The property the canonical hash actually leans on: replayed
        // operations reproduce ids exactly, including through freelist reuse.
        fn churn() -> Vec<u64> {
            let mut e = Entities::new();
            let mut live: Vec<Entity> = Vec::new();
            let mut out = Vec::new();
            for tick in 0..500u32 {
                for _ in 0..3 {
                    let x = e.alloc();
                    live.push(x);
                    out.push(x.to_bits());
                }
                // Deterministic pseudo-churn: no RNG, no hashing, no time.
                if tick % 2 == 0 && !live.is_empty() {
                    let victim = live.swap_remove((tick as usize * 7) % live.len());
                    assert!(e.free(victim));
                }
            }
            out.push(u64::from(e.len()));
            out
        }
        assert_eq!(churn(), churn());
    }
}
