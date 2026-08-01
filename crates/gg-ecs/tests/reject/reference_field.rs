//! A reference in a component: no meaning across the §4.2.2 boundary.

use gg_ecs::Component;

// `&'static` so the case under test is the *reference*, not the generic
// parameter a borrowed field would otherwise introduce (which has its own ban).
#[derive(Clone, Copy, Component)]
#[component(id = "borrowed")]
#[repr(C)]
struct Borrowed {
    who: &'static u32,
}

fn main() {}
