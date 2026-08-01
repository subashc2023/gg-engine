//! The dormant static-link variant of the game-code boundary (§2, §5.9).
//!
//! §2 makes the systems table the boundary *in every tier* and calls how the
//! table arrives "a one-line difference instead of a second program". This crate
//! is that claim, kept honest: the same five `extern "C"` entry points, the same
//! [`GameLib`], the same version and fingerprint checks — reached through the
//! linker rather than through `dlopen`.
//!
//! # It exists to be compiled, not to be run
//!
//! §5.9 gates it with a `cargo check` and nothing else. That is a deliberate
//! bound, not an oversight: a check proves the variant still *type-checks*
//! against `gg-core`'s host API, which is the thing that actually rots when a
//! signature moves under a path nobody builds. Proving the symbols *resolve*
//! wants a link, and a link is what activation brings — together with §5.6e's
//! link-mode equivalence gate, which is specified and inactive for the same
//! reason this is (§2, Game-code boundary row).
//!
//! # What activation would change
//!
//! One line in `gg-runtime`: [`GameLib::load`](gg_core::GameLib::load) becomes
//! [`link`], behind a tier feature. Everything downstream of it — adopting the
//! components, driving the systems, the replay's verb check — is written against
//! `GameLib` and does not know which mode produced it. Reload does not survive
//! activation, and that is the trade §2 names: a platform that forbids dylibs
//! forbids the cooperative loop with them.

use gg_core::reload::{GameEntryPoints, GameLib, ReloadError};
use gg_ecs::boundary::{AbiInfo, ComponentsTable, HostApiV1, SystemsTable, VerbsTable, host_api};

// `gg_game!` gives a game crate no new Rust items, only new symbols (§4.2.2) —
// so nothing here can *name* the game, and an unnamed rlib is one rustc leaves
// out of the link entirely. This line is the whole reason the static variant
// links at all, and it is load-bearing in a way a `cargo check` cannot see: with
// it removed the crate still compiles and the linker reports five unresolved
// externals.
extern crate demo_03_reload as _;

// The game's exports, resolved by the linker rather than by a symbol table.
unsafe extern "C" {
    fn gg_game_abi() -> AbiInfo;
    fn gg_game_init(api: *const HostApiV1);
    fn gg_game_components() -> ComponentsTable;
    fn gg_game_verbs() -> VerbsTable;
    fn gg_game_systems() -> SystemsTable;
}

/// The statically linked game, verified and ready to drive.
///
/// # Safety
///
/// The entry points are this binary's own, so the provenance obligation
/// [`GameLib::load`] puts on the operator is discharged by the linker — but the
/// build must be a matched set, which is what the version and fingerprint checks
/// inside verify rather than assume.
pub unsafe fn link() -> Result<GameLib, ReloadError> {
    let entries = GameEntryPoints {
        abi: gg_game_abi,
        init: gg_game_init,
        components: gg_game_components,
        verbs: gg_game_verbs,
        systems: gg_game_systems,
    };
    // SAFETY: the five entry points are one game crate's — the linker resolved
    // them from the single game rlib in this graph — and `host_api()` is
    // `&'static`, which is the lifetime `linked` requires.
    unsafe { GameLib::linked(&entries, host_api()) }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    /// One test, and it is the reason this crate is a `lib` rather than a stub:
    /// building the test harness is the only thing in the workspace that
    /// actually **links** a game crate statically, and linking is where a
    /// `cargo check` stops. §5.9 asks for a compile check; this costs the same
    /// tier one link and answers the harder question — do the symbols resolve,
    /// does the table arrive, does the same verification pass on this route.
    ///
    /// It does not activate the variant (§5.6e stays inactive): nothing here
    /// runs a world or compares a hash across link modes. It proves the fallback
    /// is a working fallback rather than a compiling one.
    #[test]
    fn the_game_arrives_through_the_linker_with_its_tables_intact() {
        // SAFETY: the entry points are this test binary's own, statically linked.
        let game = unsafe { super::link() }.unwrap();
        assert!(game.systems().len > 0, "no systems came across");
        assert!(game.components().len > 0, "no components came across");
        assert_eq!(game.bytes(), 0, "a statically linked game leaks no dylib");
    }
}
