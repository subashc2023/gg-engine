//! M4A's three new mechanisms, proven on the GPU (§4.3): buffers reached by
//! device address, textures reached by index into the one global bindless set,
//! and reverse-Z depth. Runs against whatever driver the environment provides;
//! the nightly gate re-runs it on the pinned lavapipe (§5.4) and on the real
//! GPU, which is the only place the queue-family ownership transfer executes
//! at all.
//!
//! Same zero-mystery bar as every gg-rhi test: any validation message or leak
//! fails the run through the shutdown report.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_rhi::{
    BufferDesc, BufferKind, DrawSpec, ImageDesc, ImageFormat, ImageUse, OffscreenRhi, PipelineDesc,
    PipelineHandle,
};

fn init_tracing() {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
}

/// Compile a Slang source string and build a pipeline from it. Every test here
/// needs one and they differ only in the source and whether depth is on.
fn pipeline(
    rhi: &mut OffscreenRhi,
    name: &str,
    source: &str,
    depth: bool,
) -> (PipelineHandle, u32) {
    let dir = std::env::temp_dir().join(format!("gg-rhi-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = format!("{name}.slang");
    std::fs::write(dir.join(&file), source).unwrap();
    let module = gg_shaders::compile_module(&dir.to_string_lossy(), &file).unwrap();
    let find = |stage| {
        module
            .entry_points
            .iter()
            .find(|e| e.stage == stage)
            .unwrap()
    };
    let vs = find(gg_shaders::Stage::Vertex);
    let fs = find(gg_shaders::Stage::Fragment);
    let push_size = module
        .push_constants
        .as_ref()
        .map(|p| p.size as u32)
        .unwrap_or(0);
    let handle = rhi
        .create_pipeline(&PipelineDesc {
            name,
            vs_spirv: &vs.spirv,
            vs_entry: &vs.spirv_entry,
            fs_spirv: &fs.spirv,
            fs_entry: &fs.spirv_entry,
            push_constant_size: push_size,
            depth,
        })
        .unwrap();
    (handle, push_size)
}

/// A triangle covering clip space, so every pixel is the fragment shader's.
const FULLSCREEN_VS: &str = r#"
struct VOut { float4 pos : SV_Position; }

[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    static const float2 verts[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut o;
    o.pos = float4(verts[vid], 0.0, 1.0);
    return o;
}
"#;

fn top_left(pixels: &[u8]) -> [u8; 4] {
    [pixels[0], pixels[1], pixels[2], pixels[3]]
}

/// Pixel at `(x, y)` of a `width`-wide RGBA8 readback.
fn pixel(pixels: &[u8], width: u32, x: u32, y: u32) -> [u8; 4] {
    let at = ((y * width + x) * 4) as usize;
    [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
}

/// §4.3: a buffer's *only* route into a shader is its device address. Upload
/// two colors, hand the shader the address, and read the second one back.
#[test]
fn a_buffer_reaches_the_shader_through_its_device_address() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
    let source = format!(
        r#"
struct P {{ uint64_t colors; uint index; uint _pad; }}
[[vk::push_constant]] ConstantBuffer<P> push;
{FULLSCREEN_VS}
[shader("fragment")]
float4 fs_main() : SV_Target
{{
    float* c = (float*)(push.colors + uint64_t(push.index) * 16);
    return float4(c[0], c[1], c[2], c[3]);
}}
"#
    );
    let (handle, push_size) = pipeline(&mut rhi, "bda", &source, false);
    assert_eq!(push_size, 16, "uint64 + 2 uints, std430");

    let colors: [f32; 8] = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let buffer = rhi
        .create_buffer(&BufferDesc {
            name: "test.colors",
            size: std::mem::size_of_val(&colors) as u64,
            kind: BufferKind::Storage,
        })
        .unwrap();
    rhi.upload_buffer(buffer, 0, bytemuck::cast_slice(&colors))
        .unwrap();
    rhi.flush_uploads().unwrap();

    let address = rhi.buffer_address(buffer).unwrap();
    assert_ne!(address, 0, "a live buffer has a device address");
    let mut push = [0u8; 16];
    push[..8].copy_from_slice(&address.to_le_bytes());
    push[8..12].copy_from_slice(&1u32.to_le_bytes());

    let pixels = rhi
        .render(
            [0.0, 0.0, 0.0, 1.0],
            Some(&DrawSpec {
                pipeline: handle,
                push_constants: &push,
                count: 3,
                index_buffer: None,
            }),
        )
        .unwrap();
    assert_eq!(
        top_left(&pixels),
        [0, 255, 0, 255],
        "the shader read entry 1, not entry 0 and not garbage"
    );

    rhi.destroy_buffer(buffer).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// §4.3: a material is an index. Two textures share one descriptor set, and
/// the second is registered *after* the set has already been bound in a
/// submitted command buffer — which is what update-after-bind buys and what
/// the layout flag is enabled for.
#[test]
fn textures_are_sampled_by_index_out_of_the_one_global_set() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
    let source = format!(
        r#"
[[vk::binding(0, 0)]] Texture2D<float4> g_textures[];
[[vk::binding(2, 0)]] SamplerState g_samplers[];
struct P {{ uint texture; uint sampler; }}
[[vk::push_constant]] ConstantBuffer<P> push;
{FULLSCREEN_VS}
[shader("fragment")]
float4 fs_main() : SV_Target
{{
    return g_textures[push.texture].Sample(g_samplers[push.sampler], float2(0.5, 0.5));
}}
"#
    );
    let (handle, push_size) = pipeline(&mut rhi, "bindless", &source, false);
    assert_eq!(push_size, 8);

    let solid = |color: [u8; 4]| -> Vec<u8> { color.repeat(4 * 4) };
    let make = |rhi: &mut OffscreenRhi, name: &'static str, color: [u8; 4]| {
        let image = rhi
            .create_image(&ImageDesc {
                name,
                extent: (4, 4),
                format: ImageFormat::Rgba8Unorm,
                usage: ImageUse::Sampled,
            })
            .unwrap();
        rhi.upload_image(image, &solid(color)).unwrap();
        rhi.flush_uploads().unwrap();
        (image, rhi.register_texture(image).unwrap())
    };

    let draw = |rhi: &mut OffscreenRhi, texture: u32| {
        let mut push = [0u8; 8];
        push[..4].copy_from_slice(&texture.to_le_bytes());
        rhi.render(
            [0.0, 0.0, 0.0, 1.0],
            Some(&DrawSpec {
                pipeline: handle,
                push_constants: &push,
                count: 3,
                index_buffer: None,
            }),
        )
        .unwrap()
    };

    let (red_image, red) = make(&mut rhi, "test.red", [255, 0, 0, 255]);
    assert_eq!(top_left(&draw(&mut rhi, red.get())), [255, 0, 0, 255]);

    // Registered after a draw has already bound and submitted this set.
    let (blue_image, blue) = make(&mut rhi, "test.blue", [0, 0, 255, 255]);
    assert_ne!(red.get(), blue.get(), "each texture gets its own slot");
    assert_eq!(top_left(&draw(&mut rhi, blue.get())), [0, 0, 255, 255]);
    // The first texture is still where it was: a new registration writes one
    // descriptor, it does not rebuild the set.
    assert_eq!(top_left(&draw(&mut rhi, red.get())), [255, 0, 0, 255]);

    rhi.destroy_image(red_image).unwrap();
    rhi.destroy_image(blue_image).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// Reverse-Z (§2, Math row): the buffer clears to 0.0 and the comparison is
/// GREATER_OR_EQUAL, so the *nearer* fragment survives — including when the
/// farther one is drawn second. Under a conventional depth test this frame
/// comes out entirely green.
#[test]
fn reverse_z_keeps_the_nearer_fragment_when_the_farther_one_is_drawn_after_it() {
    init_tracing();
    let extent = (16u32, 16u32);
    let mut rhi = OffscreenRhi::new(extent).unwrap();
    let source = r#"
struct VOut { float4 pos : SV_Position; float4 color : COLOR0; }

// Two quads in one draw, ordered near-then-far. Quad 0 covers the left half at
// depth 0.9 (near, under reverse-Z) in red; quad 1 covers everything at depth
// 0.1 (far) in green, and must lose on the left.
[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    static const float2 quad[6] = {
        float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0),
        float2(-1.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0),
    };
    uint which = vid / 6;
    float2 p = quad[vid % 6];
    VOut o;
    if (which == 0)
    {
        o.pos = float4(p.x * 0.5 - 0.5, p.y, 0.9, 1.0);
        o.color = float4(1.0, 0.0, 0.0, 1.0);
    }
    else
    {
        o.pos = float4(p, 0.1, 1.0);
        o.color = float4(0.0, 1.0, 0.0, 1.0);
    }
    return o;
}

[shader("fragment")]
float4 fs_main(VOut i) : SV_Target { return i.color; }
"#;
    let (handle, _) = pipeline(&mut rhi, "reversez", source, true);
    let pixels = rhi
        .render(
            [0.0, 0.0, 0.0, 1.0],
            Some(&DrawSpec {
                pipeline: handle,
                push_constants: &[],
                count: 12,
                index_buffer: None,
            }),
        )
        .unwrap();
    assert_eq!(
        pixel(&pixels, extent.0, 3, 8),
        [255, 0, 0, 255],
        "the near quad must survive the far quad drawn over it"
    );
    assert_eq!(
        pixel(&pixels, extent.0, 12, 8),
        [0, 255, 0, 255],
        "the far quad still covers where nothing is nearer"
    );

    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// The staging ring wraps rather than growing (§4.3), and a wrap waits on the
/// transfer timeline instead of overwriting bytes a submitted batch is reading.
/// Four flushed 3 MiB uploads is one and a half times around an 8 MiB ring.
#[test]
fn the_staging_ring_wraps_without_losing_the_last_write() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
    let source = format!(
        r#"
struct P {{ uint64_t data; uint _pad0; uint _pad1; }}
[[vk::push_constant]] ConstantBuffer<P> push;
{FULLSCREEN_VS}
[shader("fragment")]
float4 fs_main() : SV_Target
{{
    float* c = (float*)push.data;
    return float4(c[0], c[1], c[2], c[3]);
}}
"#
    );
    let (handle, _) = pipeline(&mut rhi, "ring", &source, false);

    const CHUNK: usize = 3 << 20;
    let buffer = rhi
        .create_buffer(&BufferDesc {
            name: "test.ring-target",
            size: CHUNK as u64,
            kind: BufferKind::Storage,
        })
        .unwrap();

    let mut bytes = vec![0u8; CHUNK];
    for round in 0..4u32 {
        // Each round writes a different color into the first 16 bytes; only
        // the last one may survive.
        let color: [f32; 4] = [(round % 2) as f32, ((round / 2) % 2) as f32, 0.0, 1.0];
        bytes[..16].copy_from_slice(bytemuck::cast_slice(&color));
        rhi.upload_buffer(buffer, 0, &bytes).unwrap();
        rhi.flush_uploads().unwrap();
    }

    let address = rhi.buffer_address(buffer).unwrap();
    let mut push = [0u8; 16];
    push[..8].copy_from_slice(&address.to_le_bytes());
    let pixels = rhi
        .render(
            [0.0, 0.0, 0.0, 1.0],
            Some(&DrawSpec {
                pipeline: handle,
                push_constants: &push,
                count: 3,
                index_buffer: None,
            }),
        )
        .unwrap();
    // Round 3: (1, 1, 0).
    assert_eq!(top_left(&pixels), [255, 255, 0, 255]);

    rhi.destroy_buffer(buffer).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// Uploads that cannot work say so precisely, before anything is recorded.
#[test]
fn impossible_uploads_are_refused_by_name() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();

    // Larger than the whole staging ring: chunked streaming arrives with the
    // asset pipeline, and until then this is an error rather than a hang.
    let huge = rhi
        .create_buffer(&BufferDesc {
            name: "test.huge",
            size: 16 << 20,
            kind: BufferKind::Storage,
        })
        .unwrap();
    let err = rhi
        .upload_buffer(huge, 0, &vec![0u8; 9 << 20])
        .unwrap_err()
        .to_string();
    assert!(err.contains("staging ring"), "got: {err}");

    // Past the end of the destination.
    let small = rhi
        .create_buffer(&BufferDesc {
            name: "test.small",
            size: 64,
            kind: BufferKind::Storage,
        })
        .unwrap();
    let err = rhi
        .upload_buffer(small, 32, &[0u8; 64])
        .unwrap_err()
        .to_string();
    assert!(err.contains("overruns"), "got: {err}");

    // An image upload whose byte count does not match the format's packed
    // size — the check that stops a BC7 block count from being guessed.
    let texture = rhi
        .create_image(&ImageDesc {
            name: "test.bc7",
            extent: (8, 8),
            format: ImageFormat::Bc7Srgb,
            usage: ImageUse::Sampled,
        })
        .unwrap();
    assert_eq!(ImageFormat::Bc7Srgb.packed_size((8, 8)), 64);
    let err = rhi
        .upload_image(texture, &[0u8; 63])
        .unwrap_err()
        .to_string();
    assert!(err.contains("packs to 64 bytes"), "got: {err}");

    // A depth image is an attachment, not a material.
    let depth = rhi
        .create_image(&ImageDesc {
            name: "test.depth",
            extent: (8, 8),
            format: ImageFormat::Depth32,
            usage: ImageUse::Depth,
        })
        .unwrap();
    let err = rhi.register_texture(depth).unwrap_err().to_string();
    assert!(err.contains("not a bindless texture"), "got: {err}");

    // And a depth format asked for as anything else is refused at creation.
    let err = rhi
        .create_image(&ImageDesc {
            name: "test.confused",
            extent: (8, 8),
            format: ImageFormat::Depth32,
            usage: ImageUse::Sampled,
        })
        .unwrap_err()
        .to_string();
    assert!(err.contains("disagree"), "got: {err}");

    rhi.destroy_buffer(huge).unwrap();
    rhi.destroy_buffer(small).unwrap();
    rhi.destroy_image(texture).unwrap();
    rhi.destroy_image(depth).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// The queue-family ownership transfer (§4.3). Whether the release/acquire
/// pair *executes* is a property of the hardware, not of this code: lavapipe
/// has one family and there is nothing to transfer. The test therefore asserts
/// the two facts agree and says which path ran, so a green CI run on lavapipe
/// is never mistaken for coverage of the cross-family path — the nightly GPU
/// leg on the real device is where that half is proven.
#[test]
fn uploads_are_clean_on_whichever_queue_topology_this_device_has() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
    let crosses = rhi.transfer_crosses_queue_families();
    assert_eq!(
        crosses,
        rhi.device_report().transfer_dedicated,
        "the ownership-transfer path keys off the same fact the report prints"
    );
    if crosses {
        tracing::info!("cross-family ownership transfer exercised on this device");
    } else {
        tracing::warn!(
            "single queue family: the release/acquire pair is a no-op here — cross-family \
             coverage comes from the nightly run on the real GPU (§5)"
        );
    }

    // Both an image and a buffer, because they take different barrier shapes.
    let buffer = rhi
        .create_buffer(&BufferDesc {
            name: "test.transfer.buffer",
            size: 256,
            kind: BufferKind::Index,
        })
        .unwrap();
    let image = rhi
        .create_image(&ImageDesc {
            name: "test.transfer.image",
            extent: (16, 16),
            format: ImageFormat::Rgba8Srgb,
            usage: ImageUse::Sampled,
        })
        .unwrap();
    rhi.upload_buffer(buffer, 0, &[7u8; 256]).unwrap();
    rhi.upload_image(image, &[9u8; 16 * 16 * 4]).unwrap();
    rhi.flush_uploads().unwrap();
    // The acquires are recorded by the next render; running one is what proves
    // the barrier pair validates.
    let _ = rhi.register_texture(image).unwrap();
    let _ = rhi.render([0.0, 0.0, 0.0, 1.0], None).unwrap();

    rhi.destroy_buffer(buffer).unwrap();
    rhi.destroy_image(image).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// A storage image takes a slot in the other global array and the layout
/// transition that array's descriptors declare (§4.3).
#[test]
fn a_storage_image_takes_a_slot_in_the_storage_array() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
    let image = rhi
        .create_image(&ImageDesc {
            name: "test.storage",
            extent: (8, 8),
            format: ImageFormat::Rgba8Unorm,
            usage: ImageUse::Storage,
        })
        .unwrap();
    let index = rhi.register_storage_image(image).unwrap();
    assert_eq!(index.get(), 0, "first registration takes slot 0");
    rhi.destroy_image(image).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}
