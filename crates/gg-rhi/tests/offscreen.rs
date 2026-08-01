//! Offscreen render → readback (§4.10 v0's GPU spine), against whatever
//! driver the environment provides (the nightly gate re-runs these on the
//! pinned lavapipe, §5.4). Same zero-mystery bar as every gg-rhi test: any
//! validation message or leak fails the run via the shutdown report.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_rhi::{DrawSpec, OffscreenRhi, PipelineDesc};

fn init_tracing() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
}

/// A clear with no draw reads back as the exact sRGB-encoded bytes.
#[test]
fn clear_readback_is_exact() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
    let pixels = rhi.render([1.0, 0.0, 0.0, 1.0], None).unwrap();
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
        })
        .unwrap();
    let pixels = rhi
        .render(
            [1.0, 0.0, 0.0, 1.0],
            Some(&DrawSpec {
                pipeline,
                push_constants: &[],
                vertex_count: 3,
            }),
        )
        .unwrap();
    for px in pixels.chunks_exact(4) {
        assert_eq!(px, [0, 255, 0, 255], "draw must cover every pixel");
    }

    // A dead handle after destruction is an error, not a dangle (§4.4).
    rhi.destroy_pipeline(pipeline).unwrap();
    let err = rhi
        .render(
            [0.0; 4],
            Some(&DrawSpec {
                pipeline,
                push_constants: &[],
                vertex_count: 3,
            }),
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
        })
        .unwrap();

    let err = rhi
        .render(
            [0.0; 4],
            Some(&DrawSpec {
                pipeline,
                push_constants: &[0u8; 12], // wrong: pipeline declares 16
                vertex_count: 3,
            }),
        )
        .unwrap_err();
    assert!(
        err.to_string().contains("12"),
        "error names the size: {err}"
    );

    // The right size draws fine afterwards — the failed call recorded nothing.
    let pixels = rhi
        .render(
            [0.0, 0.0, 0.0, 1.0],
            Some(&DrawSpec {
                pipeline,
                push_constants: bytemuck::bytes_of(&[1.0f32, 1.0, 1.0, 1.0]),
                vertex_count: 3,
            }),
        )
        .unwrap();
    assert_eq!(&pixels[0..4], &[255, 255, 255, 255]);

    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}
