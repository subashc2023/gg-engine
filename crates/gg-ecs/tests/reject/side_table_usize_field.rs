//! §4.2.1: dropping the `Pod` rule for side tables does not drop the hazard
//! list with it. A platform-width integer is still a cross-architecture
//! divergence waiting to happen.

#[derive(gg_ecs::SideTable)]
#[side_table(id = "fleet-index")]
struct FleetIndex {
    ships: Vec<u64>,
    cursor: usize,
}

fn main() {}
