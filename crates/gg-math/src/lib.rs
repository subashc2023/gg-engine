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
