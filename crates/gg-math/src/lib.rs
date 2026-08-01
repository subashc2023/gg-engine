//! `gg-math` — the §1.4 membrane as a crate (§4.2.1): [`sim`] holds our own
//! scalar types in both widths with `libm` transcendentals — portable by
//! construction, everything `bytemuck::Pod`; [`render`] is full-SIMD `glam`.
//! Conversion between them is explicit and one-way — positions narrow only
//! through a camera origin ([`render::camera_relative`]), so an absolute
//! `f32` position cannot exist render-side by accident.
//!
//! Complexity budget (§9): not a general-purpose math library. `sim` is the
//! minimal set of scalar types the sim needs; new surface arrives with a named
//! consumer, never speculatively. [`fpenv`] is hazard 5's guard (§4.2.1) —
//! determinism infrastructure, not math surface; its named consumers are the
//! M1 startup/per-tick assertions and §4.2.2's post-reload re-assert.

pub mod fpenv;
pub mod render;
pub mod sim;

/// **Determinism Contract version** (§4.2.1): the pinned toolchain, the three
/// target triples, the FP environment and the state-hash protocol, taken
/// together as one versioned claim — `canonical_hash(frame N)` is bit-identical
/// across targets, tiers and link modes for a given replay and seed.
///
/// It lives here because `gg-math` is where the claim is *tested* (the FP
/// baseline is its gate) rather than where it is consumed. Replay headers
/// record it (§4.7) so a file that predates a contract event says so instead of
/// being quietly replayed against different arithmetic.
///
/// Bumping it is the honest name for a toolchain or `libm` upgrade that moves
/// result bits. Contract events are logged in PLAN.md §4.2.1; the one taken so
/// far (rustc 1.92 → 1.97.1) moved nothing, which is why this still reads 1.
pub const DETERMINISM_CONTRACT: u32 = 1;
