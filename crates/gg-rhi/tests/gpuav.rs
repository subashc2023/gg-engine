//! The GPU-assisted-validation gate's own negative control (§5 gate 4's
//! nightly half, §8's sync/descriptor row): GPU-AV instruments shaders, so the
//! only honest proof it is engaged is a fault it alone can see.
//!
//! Both tests below run *only* under `GG_GPUAV=1` — `cargo xtask gpuav` is
//! what sets it — and return early otherwise. Not `#[ignore]`: `xtask
//! interactive` sweeps this crate's ignored tests as the §1.5 windowed suite,
//! and a deliberate device-address fault is not a windowed test.
//!
//! The fault is an out-of-bounds read through a buffer device address, which
//! is §4.3's *only* route from a buffer into a shader — the engine's most
//! load-bearing unchecked pointer, and invisible to ordinary validation.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use gg_rhi::{BufferDesc, BufferKind, DrawSpec, OffscreenRhi, PipelineDesc, PipelineHandle};

/// Entries in the fixture buffer. Small on purpose: the fault index below is
/// far outside it, so no allocator rounding can accidentally make it legal.
const ENTRIES: usize = 2;

/// The out-of-bounds entry the fault case reads. 1 MiB past the buffer — past
/// any suballocation the same `VkDeviceMemory` might hold, so what GPU-AV
/// reports is an address fault and not a neighbour's data.
const FAULT_INDEX: u32 = 1 << 16;

fn init_tracing() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
}

/// Whether this process was launched by the GPU-AV leg. Without the layer
/// setting the fault below is undefined behaviour that reports nothing, so the
/// test would assert on noise.
fn gpuav_on() -> bool {
    let on = std::env::var("GG_GPUAV").is_ok_and(|v| v == "1");
    if !on {
        println!("skipped: GPU-assisted validation off — run `cargo xtask gpuav`");
    }
    on
}

/// Read entry `index` of a `float4` array reached by device address, and paint
/// it. Legal at index 0..ENTRIES, a fault past it.
const SOURCE: &str = r#"
struct P { uint64_t colors; uint index; uint _pad; }
[[vk::push_constant]] ConstantBuffer<P> push;

struct VOut { float4 pos : SV_Position; }

[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    static const float2 verts[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut o;
    o.pos = float4(verts[vid], 0.0, 1.0);
    return o;
}

[shader("fragment")]
float4 fs_main() : SV_Target
{
    float* c = (float*)(push.colors + uint64_t(push.index) * 16);
    return float4(c[0], c[1], c[2], c[3]);
}
"#;

fn pipeline(rhi: &mut OffscreenRhi) -> PipelineHandle {
    let scratch = common::Scratch::new("gpuav");
    let search = scratch.slang("gpuav.slang", SOURCE);
    let module = gg_shaders::compile_module(&search, "gpuav.slang").unwrap();
    let find = |stage| {
        module
            .entry_points
            .iter()
            .find(|e| e.stage == stage)
            .unwrap()
    };
    let vs = find(gg_shaders::Stage::Vertex);
    let fs = find(gg_shaders::Stage::Fragment);
    rhi.create_pipeline(&PipelineDesc {
        name: "gpuav-bda",
        vs_spirv: &vs.spirv,
        vs_entry: &vs.spirv_entry,
        fs_spirv: &fs.spirv,
        fs_entry: &fs.spirv_entry,
        push_constant_size: module
            .push_constants
            .as_ref()
            .map(|p| p.size as u32)
            .unwrap_or(0),
        color: gg_rhi::ColorTarget::Backbuffer,
        blend: gg_rhi::Blend::Off,
        depth: gg_rhi::DepthMode::Off,
        samples: gg_rhi::Samples::X1,
        depth_bias: false,
    })
    .unwrap()
}

/// Render one instrumented frame reading entry `index`, and report whether the
/// run ended clean. Everything is destroyed either way — an instrumented fault
/// must still leave the §4.3 accounting intact.
fn render_reading(index: u32) -> gg_rhi::ShutdownReport {
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
    let handle = pipeline(&mut rhi);
    let colors = [1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    assert_eq!(colors.len(), ENTRIES * 4);
    let buffer = rhi
        .create_buffer(&BufferDesc {
            name: "gpuav.colors",
            size: std::mem::size_of_val(&colors) as u64,
            kind: BufferKind::Storage,
        })
        .unwrap();
    rhi.upload_buffer(buffer, 0, bytemuck::cast_slice(&colors))
        .unwrap();
    rhi.flush_uploads().unwrap();

    let mut push = [0u8; 16];
    push[..8].copy_from_slice(&rhi.buffer_address(buffer).unwrap().to_le_bytes());
    push[8..12].copy_from_slice(&index.to_le_bytes());
    common::render(
        &mut rhi,
        [0.0, 0.0, 0.0, 1.0],
        &[DrawSpec {
            pipeline: handle,
            push_constants: &push,
            count: 3,
            index_buffer: None,
            indirect: None,
            depth_bias: None,
            viewport: None,
        }],
    )
    .unwrap();

    rhi.destroy_buffer(buffer).unwrap();
    rhi.destroy_pipeline(handle).unwrap();
    rhi.shutdown()
}

/// The forgiving half: an in-bounds read through the same instrumented shader
/// is silent. Without this the fault case below proves only that GPU-AV is
/// noisy, not that it is discriminating.
#[test]
fn instrumented_in_bounds_read_is_silent() {
    init_tracing();
    if !gpuav_on() {
        return;
    }
    let report = render_reading(1);
    assert!(
        report.clean(),
        "instrumented clean frame is not clean: {report:?}"
    );
}

/// The failing half: a read 1 MiB past the buffer's end is caught at shader
/// execution. Ordinary validation sees a legal draw with a legal push constant
/// here — this message exists only because the shader was instrumented.
#[test]
fn out_of_bounds_device_address_read_is_caught() {
    init_tracing();
    if !gpuav_on() {
        return;
    }
    let report = render_reading(FAULT_INDEX);
    assert!(
        report.validation_messages > 0,
        "GPU-AV reported nothing for a read {FAULT_INDEX} entries past a {ENTRIES}-entry \
         buffer — the leg proves nothing in this state: {report:?}"
    );
    assert!(
        report.leaked_allocations.is_empty(),
        "an instrumented fault must not disturb the §4.3 accounting: {report:?}"
    );
}
