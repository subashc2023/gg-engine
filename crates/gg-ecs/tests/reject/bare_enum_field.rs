//! A bare enum reaches no syntactic blocklist — it is rejected by the *absence*
//! of a `StateHash` impl, which is what makes rejection total (§4.2.1).
//!
//! Every `gg_math::render` type is refused by this identical mechanism. There
//! is no fixture for one because `gg-ecs` is determinism-critical (§3) and
//! cannot name those types at all — a stronger guarantee than a rejection test.

use gg_ecs::StateHash;

#[derive(Clone, Copy)]
enum Stance {
    Idle,
    Alert,
}

#[derive(StateHash)]
struct Unit {
    stance: Stance,
}

fn main() {}
