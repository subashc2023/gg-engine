//! Deriving `Component` on an enum: the discriminant representation is not
//! pinned by the language, so the newtyped-integer pattern is the answer.

use gg_ecs::Component;

#[derive(Clone, Copy, Component)]
#[component(id = "stance")]
#[repr(C)]
enum Stance {
    Idle,
    Alert,
}

fn main() {}
