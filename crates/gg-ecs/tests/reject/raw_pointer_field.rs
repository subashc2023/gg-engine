//! A raw pointer in a component: an address is not a value, so the same
//! simulation would hash differently on every run (§4.2.1).

use gg_ecs::Component;

// No `bytemuck::Pod` derive: layer 1 names the field before the `Pod` bound
// could, and a second unrelated error would bury the message under test.
#[derive(Clone, Copy, Component)]
#[component(id = "cursor")]
#[repr(C)]
struct Cursor {
    at: *const u32,
}

fn main() {}
