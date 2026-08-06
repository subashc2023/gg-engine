//! A `HashMap` in a component: randomly-seeded iteration order, §4.2.1
//! hazard 3. The workspace clippy ban covers engine crates; this proves the
//! derive's own message is what a *game* crate sees, since a game is compiled
//! without our lint configuration.

use gg_ecs::Component;
use std::collections::HashMap;

// Deliberately no `Copy`/`Pod`: a `HashMap` satisfies neither, and layer 1
// fires on the field's type first — deriving them would add unrelated errors.
#[derive(Component)]
#[component(id = "lookup")]
#[repr(C)]
struct Lookup {
    by_name: HashMap<u32, u32>,
}

fn main() {}
