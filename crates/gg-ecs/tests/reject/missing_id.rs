//! No `#[component(id = "…")]`: the attribute is mandatory (§4.2).

use gg_ecs::Component;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[repr(C)]
struct Nameless {
    v: u32,
}

fn main() {}
