//! A padded component: `u8` beside a `u32` leaves three bytes of padding, and
//! `Pod` refuses it (§4.2.1 hazard 4). The exact case the plan names as one an
//! agent will write and no human will catch in review.

use gg_ecs::Component;

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
#[component(id = "health")]
#[repr(C)]
struct Health {
    flag: u8,
    amount: u32,
}

fn main() {}
