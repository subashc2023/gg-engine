//! Per-system CPU zones (§4.8) — the host half of `gg_ecs::boundary`'s
//! `SystemZone` hook, and the last piece of §4.8 to land.
//!
//! The systems loop is inside `gg-ecs`, which has neither `tracing` nor
//! `profiling` by an explicit §3 decision, and `profiling::scope!` cannot take a
//! dynamic name — a system's name is table data read out of a dylib. So the ECS
//! buys one `fn` pointer and this crate spends it, which keeps the zone in the
//! same client, on the same thread, and beside the frame-stage zones the shell
//! already emits.

/// Run one system inside a Tracy zone named after it.
///
/// Install once, beside `set_logger`:
/// `gg_ecs::boundary::set_system_zone(gg_debug::cpu::system_zone)`.
///
/// Calls `body` exactly once on every path — a zone that swallowed the call
/// would half-simulate the tick, which `run_systems` reports rather than
/// tolerates.
pub fn system_zone(name: &str, body: &mut dyn FnMut()) {
    #[cfg(feature = "tracy")]
    if let Some(client) = tracy_client::Client::running() {
        // The span ends when it drops, which is after `body` returns — the
        // reason the hook takes the body rather than being a begin/end pair.
        // `Span` is `!Send`, so the pair would have needed host-side
        // thread-local storage to survive the gap between its halves.
        let _span = client.span_alloc(Some(name), "World::run_systems", file!(), line!(), 0);
        body();
        return;
    }
    // Without Tracy the name has nowhere to go, and the hook is still installed
    // — the shell's `debug-tools` feature is what links this crate, not `tracy`.
    #[cfg(not(feature = "tracy"))]
    let _ = name;
    body();
}
