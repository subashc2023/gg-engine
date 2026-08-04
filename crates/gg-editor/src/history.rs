//! Undo, as snapshots (§6 M15.4 item 4).
//!
//! **Not a command log.** A log would need every editor write to be invertible,
//! and it would rot the first time a field changed type — an inverse recorded
//! against an `f32` lane cannot be applied to the `f64` a reload turned it into.
//! M14's encoder already round-trips a whole world and already refuses a
//! mismatched schema *by name* (`gg_ecs::world::save`), so the format that
//! survives a reload is the one the ring is made of.
//!
//! The bytes are the *editor's* here, where M15.2 item 3's play capture is the
//! *shell's*, and the line is where the edge is: a play capture is taken when
//! the transport says so and the transport is the host's, while an undo step is
//! taken on the edge of an edit and only the thing making the edit knows where
//! that is. Neither is in the world, so neither is in the canonical hash —
//! though *re-applying* one is, which is the whole point.
//!
//! What it costs is stated rather than hidden: a step is a copy of the entire
//! world, so a ring of sixteen over ten thousand entities is tens of megabytes.
//! [`DEPTH`] is the knob, and this crate is dev-only by construction (§3's deny
//! pin).
//!
//! A step is recorded **before** the change it undoes, which is what makes the
//! first undo of a session go somewhere. And the whole ring is dropped whenever
//! the transport moves: a snapshot taken inside play mode restored into a
//! stopped scene would put the game's own simulation back on top of the scene,
//! which is the one thing stop exists to prevent.

use gg_core::cvar::CVar;

/// How many steps are kept. A CVar because the right depth is a trade against
/// the memory a world costs, and only the operator knows how big theirs is.
pub(crate) static DEPTH: CVar = CVar::new_int("d.editor_undo", 16, "editor undo steps kept");

/// The ring, and what has been undone out of it.
#[derive(Default)]
pub(crate) struct History {
    /// Worlds as they were before each edit, oldest first.
    done: Vec<Vec<u8>>,
    /// Worlds undone out of [`History::done`], newest last — a redo stack, and
    /// emptied by the next fresh edit, because a branch nobody took is not a
    /// future anyone can return to.
    undone: Vec<Vec<u8>>,
}

impl History {
    /// Record the world as it stands, because something is about to change it.
    ///
    /// Cheap to call on a tick that turns out to change nothing — it costs a
    /// snapshot — but callers should still call it on the *edge* of a gesture
    /// rather than every tick of one, or a gizmo drag would fill the ring with
    /// sixty steps a second and undo would walk back through the drag frame by
    /// frame.
    pub(crate) fn edit(&mut self, world: &gg_ecs::World) {
        self.undone.clear();
        self.done.push(world.snapshot().encode());
        let depth = DEPTH.int().max(0) as usize;
        // Oldest first, so the drop is off the front. A `VecDeque` would save
        // the copy; a ring this shallow makes it once per edit past the depth.
        if self.done.len() > depth {
            self.done.drain(..self.done.len() - depth);
        }
    }

    /// Put the world back as it was before the last recorded edit. `false` when
    /// there is nothing to go back to.
    pub(crate) fn undo(&mut self, world: &mut gg_ecs::World) -> bool {
        let Some(bytes) = self.done.pop() else {
            return false;
        };
        let now = world.snapshot().encode();
        if !restore(world, &bytes) {
            return false;
        }
        self.undone.push(now);
        true
    }

    /// Re-apply the last undo.
    pub(crate) fn redo(&mut self, world: &mut gg_ecs::World) -> bool {
        let Some(bytes) = self.undone.pop() else {
            return false;
        };
        let now = world.snapshot().encode();
        if !restore(world, &bytes) {
            return false;
        }
        self.done.push(now);
        true
    }

    /// Drop everything — what a change of play state does, see the module docs.
    pub(crate) fn clear(&mut self) {
        self.done.clear();
        self.undone.clear();
    }

    /// Steps back and steps forward, for the label that says so.
    pub(crate) fn depths(&self) -> (usize, usize) {
        (self.done.len(), self.undone.len())
    }
}

/// Decode and apply, reporting rather than panicking: these bytes came out of
/// this process moments ago, so a failure here is a bug in the encoder and not
/// a stale file — but an editor that aborted on one would take the session with
/// it.
fn restore(world: &mut gg_ecs::World, bytes: &[u8]) -> bool {
    let snapshot = match gg_ecs::Snapshot::decode(bytes) {
        Ok(snapshot) => snapshot,
        Err(error) => {
            tracing::error!(%error, "editor: an undo step will not decode");
            return false;
        }
    };
    match world.restore(&snapshot) {
        Ok(report) => {
            // A same-process round trip is `Reused` throughout; anything else
            // here is the encoder disagreeing with itself one tick later.
            if !report.is_clean() {
                tracing::warn!("editor: an undo step did not round-trip");
            }
            true
        }
        Err(error) => {
            tracing::error!(%error, "editor: an undo step will not restore");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use gg_ecs::World;
    use gg_ecs::boundary::Renderable;
    use gg_math::sim;

    fn world_at(x: f64) -> (World, gg_ecs::Entity) {
        let mut world = World::new();
        let entity = world.spawn();
        world
            .insert(
                entity,
                Renderable::boxed(
                    sim::DVec3::new(x, 0.0, 0.0),
                    sim::Vec3::splat(0.5),
                    0x00ff_ffff,
                ),
            )
            .unwrap();
        (world, entity)
    }

    fn at(world: &World, entity: gg_ecs::Entity) -> f64 {
        world.get::<Renderable>(entity).unwrap().position.x
    }

    /// The exit criterion, as bytes: an undo restores the world the edit was
    /// made against, and a redo re-applies the edit. Byte-compared by M14's own
    /// encoder rather than field by field, so a step that restored the field
    /// under test and lost an unrelated one would fail here.
    #[test]
    fn an_undo_restores_the_world_byte_for_byte_and_a_redo_re_applies_it() {
        let (mut world, entity) = world_at(0.0);
        let before = world.snapshot().encode();
        let mut history = History::default();

        history.edit(&world);
        world.get_mut::<Renderable>(entity).unwrap().position.x = 7.0;
        let after = world.snapshot().encode();
        assert_ne!(before, after, "the edit changed nothing");

        assert!(history.undo(&mut world));
        assert_eq!(world.snapshot().encode(), before);
        assert!(history.redo(&mut world));
        assert_eq!(world.snapshot().encode(), after);
        assert_eq!(at(&world, entity), 7.0);
    }

    /// Steps stack, and they come back in reverse — the property a ring that
    /// kept only the newest would pass a one-edit test without having.
    #[test]
    fn undo_walks_back_through_the_edits_in_the_order_they_were_made() {
        let (mut world, entity) = world_at(0.0);
        let mut history = History::default();
        for step in 1..=3 {
            history.edit(&world);
            world.get_mut::<Renderable>(entity).unwrap().position.x = f64::from(step);
        }
        assert_eq!(history.depths(), (3, 0));
        for want in [2.0, 1.0, 0.0] {
            assert!(history.undo(&mut world));
            assert_eq!(at(&world, entity), want);
        }
        assert!(!history.undo(&mut world), "the ring is empty");
        assert_eq!(history.depths(), (0, 3));
        for want in [1.0, 2.0, 3.0] {
            assert!(history.redo(&mut world));
            assert_eq!(at(&world, entity), want);
        }
        assert!(!history.redo(&mut world));
    }

    /// A fresh edit after an undo drops the redo stack: the branch nobody took
    /// is not a future to return to, and keeping it would let a redo overwrite
    /// an edit made after it.
    #[test]
    fn an_edit_after_an_undo_forgets_what_was_undone() {
        let (mut world, entity) = world_at(0.0);
        let mut history = History::default();
        history.edit(&world);
        world.get_mut::<Renderable>(entity).unwrap().position.x = 5.0;
        assert!(history.undo(&mut world));
        assert_eq!(history.depths(), (0, 1));

        history.edit(&world);
        world.get_mut::<Renderable>(entity).unwrap().position.x = 9.0;
        assert_eq!(history.depths(), (1, 0), "the redo survived a fresh edit");
        assert!(!history.redo(&mut world));
        assert_eq!(at(&world, entity), 9.0);
    }

    /// The ring is bounded by its CVar, and it drops the *oldest* — a ring that
    /// dropped the newest would keep undo working and make it undo the wrong
    /// thing.
    #[test]
    fn the_ring_is_bounded_and_forgets_from_the_far_end() {
        let (mut world, entity) = world_at(0.0);
        let mut history = History::default();
        let was = DEPTH.int();
        DEPTH.set_int(3);
        for step in 1..=6 {
            history.edit(&world);
            world.get_mut::<Renderable>(entity).unwrap().position.x = f64::from(step);
        }
        assert_eq!(history.depths(), (3, 0));
        // Three left, and they are the three most recent: 5, 4, 3.
        for want in [5.0, 4.0, 3.0] {
            assert!(history.undo(&mut world));
            assert_eq!(at(&world, entity), want);
        }
        assert!(!history.undo(&mut world));
        DEPTH.set_int(was);
    }
}
