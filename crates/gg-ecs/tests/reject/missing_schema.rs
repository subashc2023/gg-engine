//! §4.2: a component without a schema hash cannot exist. `FIELDS` is what the
//! schema hash is computed from and what §4.2.2's offset-mapped migration
//! reads, so hand-writing the trait around the derive must not be a way to
//! register a component with no layout fingerprint.

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Unfingerprinted {
    v: u32,
}

impl gg_ecs::StateHash for Unfingerprinted {
    fn state_hash(&self, h: &mut gg_ecs::StateHasher) {
        h.u32(self.v);
    }
}

impl gg_ecs::Component for Unfingerprinted {
    const DECLARED_ID: &'static str = "unfingerprinted";
    // Stated, so the error below is about the schema and only the schema.
    const COMPONENT_ID: gg_ecs::ComponentId = gg_ecs::component_id_of!("unfingerprinted");
    const TYPE_NAME: &'static str = "Unfingerprinted";
}

fn main() {}
