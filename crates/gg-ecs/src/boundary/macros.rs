//! [`gg_game!`](crate::gg_game) — the `extern "C"` symbols a game dylib exports,
//! assembled from the lists a game declares (§4.2.2).

/// Declare this crate's components and systems as a game dylib.
///
/// Expands to the symbols the host resolves — `gg_game_abi`, `gg_game_init`,
/// `gg_game_components`, `gg_game_verbs`, `gg_game_systems` — with every system
/// wrapped in a generated shim that catches panics on *this* side of the seam
/// (see [`boundary::run`](crate::boundary::run)). Invoke it once per game crate.
///
/// ```ignore
/// gg_ecs::gg_game! {
///     components: [Position, Velocity],
///     actions: ["fire", "restart"],
///     axes: ["move_right", "aim_x"],
///     systems: [movement, gravity],
/// }
/// ```
///
/// Systems are named by **identifier, not path**, because the entry's stable id
/// is the hash of that name: a path would tie persisted identity to how the
/// source is organized, which is the same trade §4.2 already refused for
/// component ids. `use` a system into scope to rename or relocate it freely.
///
/// A system is any `fn(&mut GameWorld)`. Order in the list is execution order
/// (§4.1) — which is why ordering is game code, and reloadable.
///
/// The verb lists are string literals whose **order is the id space** a replay
/// records (§4.7): appending is free, reordering is a replay-format change. Both
/// may be omitted by a game that takes no input.
#[macro_export]
macro_rules! gg_game {
    (
        components: [$($component:ty),* $(,)?],
        systems: [$($system:ident),* $(,)?] $(,)?
    ) => {
        $crate::gg_game! {
            components: [$($component),*],
            actions: [],
            axes: [],
            systems: [$($system),*],
        }
    };
    (
        components: [$($component:ty),* $(,)?],
        actions: [$($action:literal),* $(,)?],
        axes: [$($axis:literal),* $(,)?],
        systems: [$($system:ident),* $(,)?] $(,)?
    ) => {
        // A `const` block so the game crate gets no new names, only new symbols:
        // `#[unsafe(no_mangle)]` forces external linkage regardless of scope.
        const _: () = {
            #[unsafe(no_mangle)]
            extern "C" fn gg_game_abi() -> $crate::boundary::AbiInfo {
                $crate::boundary::abi_info()
            }

            #[unsafe(no_mangle)]
            unsafe extern "C" fn gg_game_init(api: *const $crate::boundary::HostApiV1) {
                // SAFETY: the host's contract for `gg_game_init` (§4.2.2) — it
                // owns the table for as long as this dylib can be called into.
                unsafe { $crate::boundary::init(api) }
            }

            #[unsafe(no_mangle)]
            extern "C" fn gg_game_components() -> $crate::boundary::ComponentsTable {
                static TABLE: $crate::boundary::Table<$crate::boundary::ComponentLayout> =
                    $crate::boundary::Table::new();
                TABLE.components(&[$($crate::boundary::layout_of::<$component>),*])
            }

            #[unsafe(no_mangle)]
            extern "C" fn gg_game_verbs() -> $crate::boundary::VerbsTable {
                static ACTIONS: $crate::boundary::Table<$crate::boundary::VerbName> =
                    $crate::boundary::Table::new();
                static AXES: $crate::boundary::Table<$crate::boundary::VerbName> =
                    $crate::boundary::Table::new();
                $crate::boundary::verbs(&ACTIONS, &AXES, &[$($action),*], &[$($axis),*])
            }

            #[unsafe(no_mangle)]
            extern "C" fn gg_game_systems() -> $crate::boundary::SystemsTable {
                static TABLE: $crate::boundary::Table<$crate::boundary::SystemEntry> =
                    $crate::boundary::Table::new();
                TABLE.systems(&[$($crate::boundary::entry(
                    ::core::stringify!($system),
                    {
                        unsafe extern "C" fn shim(
                            world: *mut $crate::boundary::WorldHandle,
                            ctx: *const $crate::boundary::TickCtx,
                        ) -> $crate::boundary::SystemStatus {
                            // SAFETY: the host calls a systems-table entry with
                            // the handle and context it built for this tick.
                            unsafe { $crate::boundary::run(world, ctx, $system) }
                        }
                        shim
                    },
                )),*])
            }
        };
    };
}
