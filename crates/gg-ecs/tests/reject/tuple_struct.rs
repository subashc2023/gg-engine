//! Positional fields cannot migrate by name (§4.2.2), so they are refused.

use gg_ecs::Component;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "score")]
#[repr(C)]
struct Score(u32);

fn main() {}
