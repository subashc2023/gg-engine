//! Offscreen render → readback (§4.10 v0's GPU spine), against whatever
//! driver the environment provides (the nightly gate re-runs these on the
//! pinned lavapipe, §5.4). Same zero-mystery bar as every gg-rhi test: any
//! validation message or leak fails the run via the shutdown report.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use gg_rhi::{DrawSpec, OffscreenRhi, PipelineDesc};

fn init_tracing() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
}

/// A clear with no draw reads back as the exact sRGB-encoded bytes.
#[test]
fn clear_readback_is_exact() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
    let pixels = common::render(&mut rhi, [1.0, 0.0, 0.0, 1.0], &[]).unwrap();
    assert_eq!(pixels.len(), 8 * 8 * 4);
    for px in pixels.chunks_exact(4) {
        // Linear 1.0 encodes to 255; linear 0.0 to 0 — exactly, per the sRGB
        // transfer function's endpoints.
        assert_eq!(px, [255, 0, 0, 255]);
    }
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// A full-target triangle drawn through a runtime-compiled pipeline covers
/// every pixel; the pipeline cache lands on disk at shutdown (§4.4).
#[test]
fn draw_covers_target_and_cache_persists() {
    init_tracing();
    let dir = std::env::temp_dir().join(format!("gg-rhi-offscreen-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("fullscreen.slang"),
        r#"
struct VOut { float4 pos : SV_Position; }

[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    // One triangle covering clip space entirely.
    static const float2 verts[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut o;
    o.pos = float4(verts[vid], 0.0, 1.0);
    return o;
}

[shader("fragment")]
float4 fs_main() : SV_Target { return float4(0.0, 1.0, 0.0, 1.0); }
"#,
    )
    .unwrap();
    let module = gg_shaders::compile_module(&dir.to_string_lossy(), "fullscreen.slang").unwrap();
    let find = |stage| {
        module
            .entry_points
            .iter()
            .find(|e| e.stage == stage)
            .unwrap()
    };
    let vs = find(gg_shaders::Stage::Vertex);
    let fs = find(gg_shaders::Stage::Fragment);

    let mut rhi = OffscreenRhi::new((64, 64)).unwrap();
    let pipeline = rhi
        .create_pipeline(&PipelineDesc {
            name: "test-fullscreen",
            vs_spirv: &vs.spirv,
            vs_entry: &vs.spirv_entry,
            fs_spirv: &fs.spirv,
            fs_entry: &fs.spirv_entry,
            push_constant_size: 0,
            color: gg_rhi::ColorTarget::Backbuffer,
            blend: gg_rhi::Blend::Off,
            depth: gg_rhi::DepthMode::Off,
        })
        .unwrap();
    let pixels = common::render(
        &mut rhi,
        [1.0, 0.0, 0.0, 1.0],
        &[DrawSpec {
            pipeline,
            push_constants: &[],
            count: 3,
            index_buffer: None,
        }],
    )
    .unwrap();
    for px in pixels.chunks_exact(4) {
        assert_eq!(px, [0, 255, 0, 255], "draw must cover every pixel");
    }

    // A dead handle after destruction is an error, not a dangle (§4.4).
    rhi.destroy_pipeline(pipeline).unwrap();
    let err = common::render(
        &mut rhi,
        [0.0; 4],
        &[DrawSpec {
            pipeline,
            push_constants: &[],
            count: 3,
            index_buffer: None,
        }],
    )
    .unwrap_err();
    assert!(err.to_string().contains("not live"), "got: {err}");

    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");

    // The disk-backed cache (§4.4): something was persisted for this device.
    let cache_dir = std::path::Path::new("target/gg-cache");
    assert!(
        cache_dir
            .read_dir()
            .map(|mut d| d.next().is_some())
            .unwrap_or(false),
        "pipeline cache directory is empty after shutdown"
    );
}

/// Wrong-size push constants fail precisely, before anything is recorded.
#[test]
fn wrong_push_constant_size_is_an_error() {
    init_tracing();
    let dir = std::env::temp_dir().join(format!("gg-rhi-push-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("pushy.slang"),
        r#"
struct P { float4 color; }
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
float4 fs_main() : SV_Target { return push.color; }
"#,
    )
    .unwrap();
    let module = gg_shaders::compile_module(&dir.to_string_lossy(), "pushy.slang").unwrap();
    assert_eq!(module.push_constants.as_ref().unwrap().size, 16);
    let find = |stage| {
        module
            .entry_points
            .iter()
            .find(|e| e.stage == stage)
            .unwrap()
    };
    let vs = find(gg_shaders::Stage::Vertex);
    let fs = find(gg_shaders::Stage::Fragment);

    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
    let pipeline = rhi
        .create_pipeline(&PipelineDesc {
            name: "test-pushy",
            vs_spirv: &vs.spirv,
            vs_entry: &vs.spirv_entry,
            fs_spirv: &fs.spirv,
            fs_entry: &fs.spirv_entry,
            push_constant_size: 16,
            color: gg_rhi::ColorTarget::Backbuffer,
            blend: gg_rhi::Blend::Off,
            depth: gg_rhi::DepthMode::Off,
        })
        .unwrap();

    let err = common::render(
        &mut rhi,
        [0.0; 4],
        &[DrawSpec {
            pipeline,
            push_constants: &[0u8; 12], // wrong: pipeline declares 16
            count: 3,
            index_buffer: None,
        }],
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("12"),
        "error names the size: {err}"
    );

    // The right size draws fine afterwards — the failed call recorded nothing.
    let pixels = common::render(
        &mut rhi,
        [0.0, 0.0, 0.0, 1.0],
        &[DrawSpec {
            pipeline,
            push_constants: bytemuck::bytes_of(&[1.0f32, 1.0, 1.0, 1.0]),
            count: 3,
            index_buffer: None,
        }],
    )
    .unwrap();
    assert_eq!(&pixels[0..4], &[255, 255, 255, 255]);

    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// Several draws in one pass all land, in order — the M4B shape, since the sim
/// decides how many things exist and a caller forced to present per object
/// would show only the last one.
#[test]
fn every_draw_in_one_pass_reaches_the_target() {
    init_tracing();
    let dir = std::env::temp_dir().join(format!("gg-rhi-many-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("many.slang"),
        r#"
// Push constants place the quad and colour it, so one pipeline draws each
// instance differently — the same shape demo 02's per-cube push takes.
struct P { float4 color; float2 offset; float2 half_size; }
[[vk::push_constant]] ConstantBuffer<P> push;

struct VOut { float4 pos : SV_Position; }

[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    static const float2 quad[6] = {
        float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0),
        float2(-1.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0),
    };
    VOut o;
    o.pos = float4(quad[vid] * push.half_size + push.offset, 0.5, 1.0);
    return o;
}

[shader("fragment")]
float4 fs_main() : SV_Target { return push.color; }
"#,
    )
    .unwrap();
    let module = gg_shaders::compile_module(&dir.to_string_lossy(), "many.slang").unwrap();
    let find = |stage| {
        module
            .entry_points
            .iter()
            .find(|e| e.stage == stage)
            .unwrap()
    };
    let (vs, fs) = (
        find(gg_shaders::Stage::Vertex),
        find(gg_shaders::Stage::Fragment),
    );

    let extent = (16u32, 16u32);
    let mut rhi = OffscreenRhi::new(extent).unwrap();
    let pipeline = rhi
        .create_pipeline(&PipelineDesc {
            name: "many",
            vs_spirv: &vs.spirv,
            vs_entry: &vs.spirv_entry,
            fs_spirv: &fs.spirv,
            fs_entry: &fs.spirv_entry,
            push_constant_size: 32,
            color: gg_rhi::ColorTarget::Backbuffer,
            blend: gg_rhi::Blend::Off,
            depth: gg_rhi::DepthMode::Off,
        })
        .unwrap();

    // Left half red, right half green, and a third draw over the right half in
    // blue — which must win there, proving draws land *in order* rather than
    // merely all landing.
    let push = |color: [f32; 4], offset: [f32; 2]| {
        let mut bytes = Vec::with_capacity(32);
        for f in color.iter().chain(&offset).chain(&[0.5f32, 1.0]) {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        bytes
    };
    let (red, green, blue) = (
        push([1.0, 0.0, 0.0, 1.0], [-0.5, 0.0]),
        push([0.0, 1.0, 0.0, 1.0], [0.5, 0.0]),
        push([0.0, 0.0, 1.0, 1.0], [0.5, 0.0]),
    );
    fn draw(pipeline: gg_rhi::PipelineHandle, bytes: &[u8]) -> DrawSpec<'_> {
        DrawSpec {
            pipeline,
            push_constants: bytes,
            count: 6,
            index_buffer: None,
        }
    }
    let pixels = common::render(
        &mut rhi,
        [0.0, 0.0, 0.0, 1.0],
        &[
            draw(pipeline, &red),
            draw(pipeline, &green),
            draw(pipeline, &blue),
        ],
    )
    .unwrap();

    let pixel = |x: u32, y: u32| {
        let at = ((y * extent.0 + x) * 4) as usize;
        &pixels[at..at + 4]
    };
    assert_eq!(pixel(4, 8), [255, 0, 0, 255], "the first draw is missing");
    assert_eq!(pixel(12, 8), [0, 0, 255, 255], "the last draw must win");

    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// `GG_ADAPTER` is a choice, not a preference: a name nothing matches fails the
/// run rather than falling back to the highest-scoring device, so a result
/// reported against the second vendor cannot secretly be the first's. Relies on
/// nextest's process-per-test; `cargo test` would leak the variable sideways.
#[test]
fn an_unmatched_adapter_override_is_an_error_not_a_fallback() {
    init_tracing();
    // SAFETY: own process, before anything has touched the Vulkan loader — no
    // other thread exists to observe the change.
    unsafe { std::env::set_var("GG_ADAPTER", "no-such-adapter-exists") };
    let err = OffscreenRhi::new((8, 8)).err().expect("override matched");
    assert!(
        matches!(&err, gg_rhi::RhiError::NoSuitableDevice(r) if r.contains("GG_ADAPTER")),
        "{err:?}"
    );
}
