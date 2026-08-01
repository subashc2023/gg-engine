//! `usize` in a component: platform-width, so rejected (§4.2.1).

use gg_ecs::Component;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "counter")]
#[repr(C)]
struct Counter {
    ticks: usize,
}

fn main() {}
