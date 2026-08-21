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

mod common;

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
    depth: gg_rhi::DepthMode,
) -> (PipelineHandle, u32) {
    let scratch = common::Scratch::new(name);
    let file = format!("{name}.slang");
    let module = gg_shaders::compile_module(&scratch.slang(&file, source), &file).unwrap();
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
            color: gg_rhi::ColorTarget::Backbuffer,
            blend: gg_rhi::Blend::Off,
            depth,
            samples: gg_rhi::Samples::X1,
            depth_bias: false,
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
    let (handle, push_size) = pipeline(&mut rhi, "bda", &source, gg_rhi::DepthMode::Off);
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

    let pixels = common::render(
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
    let (handle, push_size) = pipeline(&mut rhi, "bindless", &source, gg_rhi::DepthMode::Off);
    assert_eq!(push_size, 8);

    let solid = |color: [u8; 4]| -> Vec<u8> { color.repeat(4 * 4) };
    let make = |rhi: &mut OffscreenRhi, name: &'static str, color: [u8; 4]| {
        let image = rhi
            .create_image(&ImageDesc {
                name,
                extent: (4, 4),
                format: ImageFormat::Rgba8Unorm,
                usage: ImageUse::Sampled,
                mip_levels: 1,
                samples: gg_rhi::Samples::X1,
            })
            .unwrap();
        rhi.upload_image(image, 0, &solid(color)).unwrap();
        rhi.flush_uploads().unwrap();
        (image, rhi.register_texture(image).unwrap())
    };

    let draw = |rhi: &mut OffscreenRhi, texture: u32| {
        let mut push = [0u8; 8];
        push[..4].copy_from_slice(&texture.to_le_bytes());
        common::render(
            rhi,
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
    let (handle, _) = pipeline(&mut rhi, "reversez", source, gg_rhi::DepthMode::Write);
    let pixels = common::render_with_depth(
        &mut rhi,
        [0.0, 0.0, 0.0, 1.0],
        &[DrawSpec {
            pipeline: handle,
            push_constants: &[],
            count: 12,
            index_buffer: None,
            indirect: None,
            depth_bias: None,
            viewport: None,
        }],
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
    let (handle, _) = pipeline(&mut rhi, "ring", &source, gg_rhi::DepthMode::Off);

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
    let pixels = common::render(
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
    // Round 3: (1, 1, 0).
    assert_eq!(top_left(&pixels), [255, 255, 0, 255]);

    rhi.destroy_buffer(buffer).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// A buffer larger than the whole ring is split and reassembled at the
/// destination (§4.6 streaming). 20 MiB is two and a half rings, so the copy
/// spans chunks, a wrap, and the early submit a batch that outgrew the ring
/// forces — and the readback proves every byte landed at its own offset, not
/// merely that the calls returned.
#[test]
fn an_upload_larger_than_the_staging_ring_arrives_whole() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();

    const SIZE: usize = 20 << 20;
    let readback = rhi
        .create_buffer(&BufferDesc {
            name: "test.oversize",
            size: SIZE as u64,
            kind: BufferKind::Readback,
        })
        .unwrap();

    // A pattern with a long period, so a chunk landing at the wrong offset
    // cannot coincide with the bytes that belong there.
    let source: Vec<u8> = (0..SIZE).map(|i| (i % 251) as u8).collect();
    rhi.upload_buffer(readback, 0, &source).unwrap();
    rhi.flush_uploads().unwrap();

    let landed = rhi.map_buffer(readback).unwrap();
    let first = (0..SIZE).find(|&i| landed[i] != source[i]);
    assert_eq!(first, None, "first byte that disagrees");

    rhi.destroy_buffer(readback).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// A block that has to wrap is charged for the bytes it *uses*, not for the
/// dead space it steps over to stay contiguous (§4.3).
///
/// The cursor's phase is the whole test. A fresh uploader and the uniform-sized
/// rounds above leave the head on a ring boundary, where a wrap skips nothing;
/// one 256-byte upload first puts it 256 bytes past one, so the 8 MiB chunk
/// that follows must skip almost the entire ring. Those skipped bytes belong to
/// no batch and no wait can reclaim them, so charging them to the reservation
/// made a legal upload unsatisfiable — the ring refused it by name rather than
/// wrapping.
#[test]
fn a_wrapping_upload_is_not_charged_for_the_dead_space_it_skips() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();

    // Flushed on its own: this exists only to leave the head off a boundary.
    let phase = rhi
        .create_buffer(&BufferDesc {
            name: "test.phase",
            size: 256,
            kind: BufferKind::Readback,
        })
        .unwrap();
    rhi.upload_buffer(phase, 0, &[0xa5u8; 256]).unwrap();
    rhi.flush_uploads().unwrap();

    // 12 MiB: chunk 0 is the ring exactly and has nowhere to sit but the next
    // boundary; chunk 1 is the 4 MiB remainder.
    const SIZE: usize = 12 << 20;
    let big = rhi
        .create_buffer(&BufferDesc {
            name: "test.wrapped",
            size: SIZE as u64,
            kind: BufferKind::Readback,
        })
        .unwrap();
    let source: Vec<u8> = (0..SIZE).map(|i| (i % 251) as u8).collect();
    rhi.upload_buffer(big, 0, &source).unwrap();
    rhi.flush_uploads().unwrap();

    let landed = rhi.map_buffer(big).unwrap();
    assert_eq!(
        (0..SIZE).find(|&i| landed[i] != source[i]),
        None,
        "first byte that disagrees"
    );
    // The wrap lands on top of where the first upload staged, so it may only
    // proceed once that batch has retired.
    assert!(
        rhi.map_buffer(phase).unwrap().iter().all(|b| *b == 0xa5),
        "the wrap overwrote staging bytes a batch was still reading"
    );

    rhi.destroy_buffer(big).unwrap();
    rhi.destroy_buffer(phase).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// The same accounting, at sizes where nothing is split — `gg-render`'s
/// residency shape (§4.6): a small mesh, then a large one, each behind its own
/// flush. 4 MiB leaves 4 MiB before the wrap point, so the 5 MiB that follows
/// cannot start where the cursor stands and skips to the next boundary; under
/// the old accounting it then needed 9 MiB of an 8 MiB ring and failed.
#[test]
fn a_large_upload_after_a_smaller_one_wraps_instead_of_failing() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();

    const SMALL: usize = 4 << 20;
    const LARGE: usize = 5 << 20;
    let small = rhi
        .create_buffer(&BufferDesc {
            name: "test.mesh.small",
            size: SMALL as u64,
            kind: BufferKind::Readback,
        })
        .unwrap();
    let large = rhi
        .create_buffer(&BufferDesc {
            name: "test.mesh.large",
            size: LARGE as u64,
            kind: BufferKind::Readback,
        })
        .unwrap();

    // Distinct long-period patterns: a chunk at the wrong offset cannot
    // coincide with the bytes that belong there.
    let small_src: Vec<u8> = (0..SMALL).map(|i| (i % 251) as u8).collect();
    let large_src: Vec<u8> = (0..LARGE).map(|i| (i % 241 + 7) as u8).collect();
    rhi.upload_buffer(small, 0, &small_src).unwrap();
    rhi.flush_uploads().unwrap();
    rhi.upload_buffer(large, 0, &large_src).unwrap();
    rhi.flush_uploads().unwrap();

    let landed_large = rhi.map_buffer(large).unwrap();
    let landed_small = rhi.map_buffer(small).unwrap();
    assert_eq!(
        (0..LARGE).find(|&i| landed_large[i] != large_src[i]),
        None,
        "first byte of the wrapped upload that disagrees"
    );
    assert_eq!(
        (0..SMALL).find(|&i| landed_small[i] != small_src[i]),
        None,
        "the wrap overwrote the earlier upload's staging bytes"
    );

    rhi.destroy_buffer(large).unwrap();
    rhi.destroy_buffer(small).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// A mip chain arrives level by level and every level is sampled from the same
/// image (§4.6). The chain is uploaded smallest-first — the order a KTX2 file
/// stores it in — to prove no level depends on another having gone before it.
#[test]
fn a_mip_chain_uploads_level_by_level_and_samples_from_every_one() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();

    // 4 levels: 8x8, 4x4, 2x2, 1x1. Each is a flat colour, so which level a
    // sample came from is readable off the pixel. Only 0 and 255 per channel:
    // the offscreen target is sRGB and would re-encode anything between them,
    // which is a colour-space bug wearing a wrong-level costume.
    const EXTENT: (u32, u32) = (8, 8);
    const LEVELS: u32 = 4;
    const COLORS: [[u8; 4]; LEVELS as usize] = [
        [255, 0, 0, 255],
        [0, 255, 0, 255],
        [0, 0, 255, 255],
        [255, 255, 0, 255],
    ];
    let image = rhi
        .create_image(&ImageDesc {
            name: "test.chain",
            extent: EXTENT,
            format: ImageFormat::Rgba8Unorm,
            usage: ImageUse::Sampled,
            mip_levels: LEVELS,
            samples: gg_rhi::Samples::X1,
        })
        .unwrap();
    for level in (0..LEVELS).rev() {
        let (w, h) = gg_rhi::mip_extent(EXTENT, level);
        let bytes: Vec<u8> = COLORS[level as usize].repeat((w * h) as usize);
        rhi.upload_image(image, level, &bytes).unwrap();
    }
    rhi.flush_uploads().unwrap();
    let index = rhi.register_texture(image).unwrap();

    // Sample each level explicitly. `SampleLevel` rather than a derivative, so
    // the level under test is named rather than inferred from the quad.
    let source = format!(
        r#"
struct P {{ uint texture; uint level; uint _pad0; uint _pad1; }}
[[vk::push_constant]] ConstantBuffer<P> push;
[[vk::binding(0, 0)]] Texture2D<float4> textures[];
[[vk::binding(2, 0)]] SamplerState samplers[];
{FULLSCREEN_VS}
[shader("fragment")]
float4 fs_main() : SV_Target
{{
    return textures[push.texture].SampleLevel(samplers[1], float2(0.5, 0.5), push.level);
}}
"#
    );
    let (handle, _) = pipeline(&mut rhi, "chain", &source, gg_rhi::DepthMode::Off);

    for level in 0..LEVELS {
        let mut push = [0u8; 16];
        push[..4].copy_from_slice(&index.get().to_le_bytes());
        push[4..8].copy_from_slice(&level.to_le_bytes());
        let pixels = common::render(
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
        assert_eq!(
            top_left(&pixels),
            COLORS[level as usize],
            "level {level} sampled as the wrong one"
        );
    }

    rhi.destroy_image(image).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// Uploads that cannot work say so precisely, before anything is recorded.
#[test]
fn impossible_uploads_are_refused_by_name() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();

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
            mip_levels: 1,
            samples: gg_rhi::Samples::X1,
        })
        .unwrap();
    assert_eq!(ImageFormat::Bc7Srgb.packed_size((8, 8)), 64);
    let err = rhi
        .upload_image(texture, 0, &[0u8; 63])
        .unwrap_err()
        .to_string();
    assert!(err.contains("packs to 64 bytes"), "got: {err}");

    // A level the image was never allocated. Refused by name rather than
    // recorded against a subresource that does not exist.
    let err = rhi
        .upload_image(texture, 1, &[0u8; 16])
        .unwrap_err()
        .to_string();
    assert!(err.contains("allocated with 1"), "got: {err}");

    // More levels than the extent reduces to: 8x8 is four, never five.
    let err = rhi
        .create_image(&ImageDesc {
            name: "test.overlong",
            extent: (8, 8),
            format: ImageFormat::Rgba8Unorm,
            usage: ImageUse::Sampled,
            mip_levels: 5,
            samples: gg_rhi::Samples::X1,
        })
        .unwrap_err()
        .to_string();
    assert!(err.contains("5 mip levels, and (8, 8) has 4"), "got: {err}");

    // A depth image is an attachment, not a material.
    let depth = rhi
        .create_image(&ImageDesc {
            name: "test.depth",
            extent: (8, 8),
            format: ImageFormat::Depth32,
            usage: ImageUse::Depth,
            mip_levels: 1,
            samples: gg_rhi::Samples::X1,
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
            mip_levels: 1,
            samples: gg_rhi::Samples::X1,
        })
        .unwrap_err()
        .to_string();
    assert!(err.contains("disagree"), "got: {err}");

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
            mip_levels: 1,
            samples: gg_rhi::Samples::X1,
        })
        .unwrap();
    rhi.upload_buffer(buffer, 0, &[7u8; 256]).unwrap();
    rhi.upload_image(image, 0, &[9u8; 16 * 16 * 4]).unwrap();
    rhi.flush_uploads().unwrap();
    // The acquires are recorded by the next render; running one is what proves
    // the barrier pair validates.
    let _ = rhi.register_texture(image).unwrap();
    let _ = common::render(&mut rhi, [0.0, 0.0, 0.0, 1.0], &[]).unwrap();

    rhi.destroy_buffer(buffer).unwrap();
    rhi.destroy_image(image).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// A resource uploaded and then retired *before* the frame that would have
/// read it leaves no barrier behind it.
///
/// The acquire queue holds raw handles (`upload::Acquire`), so a retire between
/// the flush and the next frame used to leave that frame recording a barrier
/// over a destroyed image — `VUID-VkImageMemoryBarrier2-image-parameter`, then
/// an access violation. Found by replacing the editor's glyph atlas on the
/// frame after the one that uploaded it (§6 M15.1), which is ordinary use.
///
/// **Vacuous where one queue family owns everything**, which is every lavapipe
/// run and so every automated tier: nothing is owed and nothing can be stale.
/// The real assertion runs on the desk's GPU (`GG_ADAPTER`) and in the nightly
/// real-GPU leg — the same split `uploads_are_clean_on_whichever_queue_topology`
/// documents.
#[test]
fn a_resource_retired_before_its_first_read_leaves_no_barrier_owed() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
    if !rhi.transfer_crosses_queue_families() {
        tracing::warn!("single queue family: nothing is owed here, so nothing can be stale");
    }
    let buffer = rhi
        .create_buffer(&BufferDesc {
            name: "test.retired.buffer",
            size: 256,
            kind: BufferKind::Index,
        })
        .unwrap();
    let image = rhi
        .create_image(&ImageDesc {
            name: "test.retired.image",
            extent: (16, 16),
            format: ImageFormat::Rgba8Srgb,
            usage: ImageUse::Sampled,
            mip_levels: 1,
            samples: gg_rhi::Samples::X1,
        })
        .unwrap();
    rhi.upload_buffer(buffer, 0, &[7u8; 256]).unwrap();
    rhi.upload_image(image, 0, &[9u8; 16 * 16 * 4]).unwrap();
    let _ = rhi.register_texture(image).unwrap();
    rhi.flush_uploads().unwrap();
    // Both retired with their acquires still owed. The render is the recording
    // that would name them.
    rhi.destroy_buffer(buffer).unwrap();
    rhi.destroy_image(image).unwrap();
    let _ = common::render(&mut rhi, [0.0, 0.0, 0.0, 1.0], &[]).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// A frame that would read an upload nobody submitted is refused by name (§6
/// M81), and the refusal costs the *recorded* batch nothing — flushing after it
/// still lands every byte.
///
/// This is the one hazard in the file whose failure is invisible on both halves
/// at once. The copy never ran, so the destination holds what it held — on a
/// fresh buffer that is zeroes, which reads as a black texture rather than as an
/// error. And `Uploader::acquires` is pushed when the release is *recorded*, so
/// where the transfer family is its own engine the frame acquires ownership of a
/// resource whose release is still sitting in an unsubmitted command buffer,
/// with `timeline_value` unmoved and the graphics submit therefore waiting on
/// nothing at all. Unlike the ownership tests above this one is **not** vacuous
/// on lavapipe: an open batch is an open batch whatever the queue topology.
#[test]
fn a_frame_is_refused_while_an_upload_it_would_read_is_still_unflushed() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
    let buffer = rhi
        .create_buffer(&BufferDesc {
            name: "test.unflushed",
            size: 256,
            kind: BufferKind::Readback,
        })
        .unwrap();

    // Nothing recorded: the check must be silent, or every frame in the tree
    // fails and the gate is about the check rather than about the hazard.
    common::render(&mut rhi, [0.0, 0.0, 0.0, 1.0], &[]).unwrap();

    rhi.upload_buffer(buffer, 0, &[0x5au8; 256]).unwrap();
    let err = common::render(&mut rhi, [0.0, 0.0, 0.0, 1.0], &[])
        .unwrap_err()
        .to_string();
    assert!(err.contains("never flushed"), "got: {err}");
    assert!(err.contains("256 bytes"), "the batch's own size: {err}");

    // Refused, not consumed. The recorded batch is still there to flush.
    rhi.flush_uploads().unwrap();
    common::render(&mut rhi, [0.0, 0.0, 0.0, 1.0], &[]).unwrap();
    assert!(
        rhi.map_buffer(buffer).unwrap().iter().all(|b| *b == 0x5a),
        "the refused frame cost the batch its bytes"
    );

    rhi.destroy_buffer(buffer).unwrap();
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
            mip_levels: 1,
            samples: gg_rhi::Samples::X1,
        })
        .unwrap();
    let index = rhi.register_storage_image(image).unwrap();
    assert_eq!(index.get(), 0, "first registration takes slot 0");
    rhi.destroy_image(image).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// The indirect path (§6 M10): the same geometry, drawn twice, once with the
/// counts on the CPU and once with them in device memory. Equality is the whole
/// assertion — an indirect draw that renders *something* proves far less than
/// one that renders exactly what its direct equivalent does.
///
/// It lives beside the BDA and bindless tests rather than in `offscreen.rs`
/// because it is the same claim they make: geometry is reached through §4.3's
/// resource rules and nothing is bound but the index stream.
#[test]
fn an_indirect_draw_renders_what_the_direct_one_does_and_zero_instances_render_nothing() {
    init_tracing();
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
    // No `format!`: this fixture interpolates nothing, unlike the ones above
    // that splice `FULLSCREEN_VS` in.
    let source = r#"
struct P { uint64_t positions; }
[[vk::push_constant]] ConstantBuffer<P> push;
[shader("vertex")]
float4 vs_main(uint vid : SV_VertexID) : SV_Position
{
    float* p = (float*)(push.positions + uint64_t(vid) * 8);
    return float4(p[0], p[1], 0.5, 1.0);
}
[shader("fragment")]
float4 fs_main() : SV_Target { return float4(0.0, 1.0, 0.0, 1.0); }
"#;
    let (handle, push_size) = pipeline(&mut rhi, "indirect", source, gg_rhi::DepthMode::Off);
    assert_eq!(push_size, 8);

    // One oversized triangle, so every pixel is covered whichever way it is
    // drawn — this test is about the draw parameters, not about rasterization.
    let positions: [f32; 6] = [-3.0, -3.0, 3.0, -3.0, 0.0, 3.0];
    let vertices = rhi
        .create_buffer(&BufferDesc {
            name: "test.indirect.positions",
            size: std::mem::size_of_val(&positions) as u64,
            kind: BufferKind::Storage,
        })
        .unwrap();
    let indices = rhi
        .create_buffer(&BufferDesc {
            name: "test.indirect.indices",
            size: 12,
            kind: BufferKind::Index,
        })
        .unwrap();
    rhi.upload_buffer(vertices, 0, bytemuck::cast_slice(&positions))
        .unwrap();
    rhi.upload_buffer(indices, 0, bytemuck::cast_slice(&[0u32, 1, 2]))
        .unwrap();
    rhi.flush_uploads().unwrap();
    let push = rhi.buffer_address(vertices).unwrap().to_le_bytes();

    // Host-visible and written without a submit: a draw list is rebuilt every
    // frame, which is why `Indirect` is `Dynamic`-shaped (§4.3).
    let commands = rhi
        .create_buffer(&BufferDesc {
            name: "test.indirect.commands",
            size: 64,
            kind: BufferKind::Indirect,
        })
        .unwrap();

    let direct = common::render(
        &mut rhi,
        [0.0, 0.0, 0.0, 1.0],
        &[DrawSpec {
            pipeline: handle,
            push_constants: &push,
            count: 3,
            index_buffer: Some(indices),
            indirect: None,
            depth_bias: None,
            viewport: None,
        }],
    )
    .unwrap();
    assert_eq!(top_left(&direct), [0, 255, 0, 255], "the control drew");

    rhi.write_buffer(
        commands,
        0,
        bytemuck::bytes_of(&gg_rhi::IndirectCommand {
            index_count: 3,
            instance_count: 1,
            ..Default::default()
        }),
    )
    .unwrap();
    let indirect = common::render(
        &mut rhi,
        [0.0, 0.0, 0.0, 1.0],
        &[DrawSpec {
            pipeline: handle,
            push_constants: &push,
            // Deliberately wrong: the count in the buffer is what must win, and
            // a path that quietly read this one would still pass a test that
            // set both to 3.
            count: 0,
            index_buffer: Some(indices),
            indirect: Some(gg_rhi::Indirect {
                buffer: commands,
                offset: 0,
            }),
            depth_bias: None,
            viewport: None,
        }],
    )
    .unwrap();
    assert_eq!(
        indirect, direct,
        "indirect must match the direct draw exactly"
    );

    // Zero instances is how a culling pass rejects a batch without rewriting
    // the list, so it has to mean "draw nothing" rather than "draw one".
    rhi.write_buffer(
        commands,
        0,
        bytemuck::bytes_of(&gg_rhi::IndirectCommand {
            index_count: 3,
            instance_count: 0,
            ..Default::default()
        }),
    )
    .unwrap();
    let culled = common::render(
        &mut rhi,
        [0.0, 0.0, 0.0, 1.0],
        &[DrawSpec {
            pipeline: handle,
            push_constants: &push,
            count: 3,
            index_buffer: Some(indices),
            indirect: Some(gg_rhi::Indirect {
                buffer: commands,
                offset: 0,
            }),
            depth_bias: None,
            viewport: None,
        }],
    )
    .unwrap();
    assert_eq!(
        top_left(&culled),
        [0, 0, 0, 255],
        "zero instances drew nothing"
    );

    rhi.destroy_buffer(commands).unwrap();
    rhi.destroy_buffer(indices).unwrap();
    rhi.destroy_buffer(vertices).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}
