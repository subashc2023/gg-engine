//! The §4.2.2 game-code boundary — both sides of it.
//!
//! `gg-abi` defines the seam's *layout*; this module is the code on either end
//! of it. [`host_api`] implements the versioned function table over a real
//! [`World`](crate::World); [`GameWorld`] and [`gg_game!`](crate::gg_game) are
//! what a game crate writes against, so gameplay code names components and never
//! sees a raw pointer.
//!
//! # Why it lives in `gg-ecs`
//!
//! Not preference — arithmetic. A game crate may link `gg-abi`, `gg-ecs`,
//! `gg-ecs-derive` and `gg-math` and nothing else (§3's deny pin, which is
//! what makes the fingerprint's scope and a dylib's link set the same list).
//! `gg-abi` is `no_std` and sits *below* [`Component`](crate::Component), so
//! it cannot assemble a table keyed on component identity; `gg-core` may not
//! depend on `gg-ecs` at all (§3's closed charter, and the reason
//! `GameLib::load` takes the table as an argument). That leaves one crate,
//! and it is this one.
//!
//! It is not a widening of §7's non-goals: no scheduler, no system graph. The
//! table's order *is* the execution order, chosen by the game and read by the
//! loop (§4.1).
//!
//! # The two directions are trusted differently
//!
//! Host-side entry points are handed pointers by a dylib that may have been
//! built from anything, so every one of them validates before it dereferences,
//! and an out-of-range request is an [`AbiStatus`] rather than a panic. Guest
//! side, the host is the trusted, statically-linked engine — a column whose
//! stride disagrees with its component is a host bug, and the assertion that
//! catches it is a panic the shim converts into a status.

mod audio;
mod declare;
mod game;
mod host;
mod load;
mod macros;
mod prefs;
mod present;
mod scene;
mod ui;

pub use audio::{FADE_MS, MAX_MS, Sound, wave};
pub use declare::{Table, assert_protocol_matches_raw, entry, init, layout_of, run, verbs};
pub use game::{Access, BoundaryData, BoundaryError, Columns, GameWorld};
pub use host::{SystemZone, host_api, set_logger, set_system_zone};
pub use load::{SystemPanic, Verbs, read_verbs, same_schemas};
pub use prefs::{Prefs, QUIET_MAX, aa, cursor, display, frame, settings, unfocused};
pub use present::{DEFAULT_SMOOTHNESS, Eye, Light, Look, Model, Renderable, Sky, light, shape};
pub use scene::Node;
pub use ui::{CANVAS, CELL, TEXT, Widget, state, text_width, widget, widget_id};

pub use crate::hash::system_id;

/// Register every component **this module** declares (§6 M89).
///
/// A host reads these whatever the game says — the renderer draws a
/// [`Renderable`] and weighs a [`Sky`], the mixer plays a [`Sound`], the shell
/// applies a [`Prefs`] — so registration cannot be conditional on the game
/// having remembered to declare them. It is idempotent against a game that
/// did: same id, same schema, and [`World::adopt`](crate::World::adopt) accepts
/// that and refuses only a *collision*.
///
/// It exists because the shell kept the list by hand and `Sky` was never added
/// to it. Demo 06 spawned one from §6 M27 to M89 and the insert was refused
/// every time, one `ERROR` line deep, in the demo whose whole subject is light
/// — and no picture noticed, because `gg-golden` builds its own worlds and
/// registers `Sky` itself.
///
/// # Errors
///
/// A collision, which is a startup error and never a silent remap (§4.2).
pub fn register_all(world: &mut crate::World) -> Result<(), crate::RegistryError> {
    world.register::<Sound>()?;
    world.register::<Prefs>()?;
    world.register::<Renderable>()?;
    world.register::<Model>()?;
    world.register::<Light>()?;
    world.register::<Sky>()?;
    world.register::<Eye>()?;
    world.register::<Look>()?;
    world.register::<Node>()?;
    world.register::<Widget>()?;
    Ok(())
}

// Re-exported so `gg_game!`'s expansion names exactly one crate: a game crate is
// pinned to three engine dependencies (§3) and a macro that silently required a
// fourth path in scope would be a pin held by hope.
pub use gg_abi::host::log_level;
pub use gg_abi::{
    AXIS_SCALE, AbiInfo, AbiStatus, ActionId, AxisId, BINDING_NAME, Binding, ComponentLayout,
    ComponentsTable, HostApiV1, InputFrame, MAX_AXES, SystemEntry, SystemStatus, SystemsTable,
    TickCtx, VerbName, VerbsTable, WorldHandle, asset_id,
};

/// What this build of the boundary says about itself — the value
/// `gg_game_abi()` returns.
///
/// Both halves come from `gg-abi` rather than from anything local: the whole
/// point of the check is that host and dylib compare *the same constant*
/// compiled twice.
#[must_use]
pub const fn abi_info() -> AbiInfo {
    AbiInfo {
        host_api_version: gg_abi::HOST_API_VERSION,
        fingerprint: gg_abi::BOUNDARY_FINGERPRINT,
    }
}

/// The world as the boundary sees it: an opaque handle, never a layout.
///
/// The cast is the entire mechanism — [`WorldHandle`] is a zero-sized `repr(C)`
/// marker, so this is a pointer with a different name and no promise attached to
/// it. The dylib cannot do anything with the result except hand it back.
#[must_use]
pub fn handle(world: &mut crate::World) -> *mut WorldHandle {
    core::ptr::from_mut(world).cast()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    /// Every `gg.` component this module declares is one [`super::register_all`]
    /// registers.
    ///
    /// A text scan over this module's own sources, for the reason `xtask`'s
    /// cross-file gates are one: the alternative is a registry the derive writes
    /// into, which is a linker-order inventory rather than a list a reader can
    /// check. What it catches is the omission that produced it — a component
    /// added to the protocol and not to the host's list.
    #[test]
    fn every_boundary_component_is_one_the_host_registers() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/boundary");
        let mut declared = Vec::new();
        for entry in std::fs::read_dir(&dir).expect("the boundary module's own directory") {
            let path = entry.expect("an entry").path();
            if path.extension().is_none_or(|e| e != "rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("a source file");
            // The declared id, then the `pub struct` under it — the type name
            // is what `register_all` spells and the id is what makes it ours.
            for (at, _) in text.match_indices("#[component(id = \"gg.") {
                let name = text[at..]
                    .split_once("pub struct ")
                    .and_then(|(_, tail)| tail.split(|c: char| !c.is_alphanumeric()).next())
                    .expect("a struct under the attribute")
                    .to_owned();
                declared.push(name);
            }
        }
        declared.sort_unstable();
        // §6 M87: a scan that matched nothing would make the loop below a
        // universal quantifier over an empty set, which is true and worthless.
        assert!(
            declared.len() >= 10,
            "the scan found {} boundary components — it stopped matching this module's \
             declarations rather than the module losing them (§6 M87)",
            declared.len()
        );

        let body = include_str!("boundary.rs")
            .split_once("pub fn register_all")
            .expect("register_all is in this file")
            .1;
        let body = &body[..body.find("\n}").expect("its closing brace")];
        for name in &declared {
            assert!(
                body.contains(&format!("register::<{name}>()")),
                "`{name}` is a boundary component the host does not register — a game that \
                 spawns one without declaring it is refused `UnknownComponent` and told so in \
                 a log line nothing reads (§6 M89)"
            );
        }
    }
}
