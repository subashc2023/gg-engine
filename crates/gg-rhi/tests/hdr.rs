//! What M11 added to the execution layer: a float colour target, a depth
//! attachment a later pass may sample, and dynamic depth bias.
//!
//! Each of the three is a thing a green run would otherwise say nothing about.
//! An `Rgba16F` target that silently clamped would leave the tonemapper a curve
//! over values that were already 1.0; a shadow map that could not be sampled
//! fails at the descriptor write, three layers away from the mistake; and an
//! unset dynamic bias draws with whatever the command buffer had, which is a
//! shadow that acnes on one driver and not the next.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_rhi::{
    Access, BufferDesc, BufferKind, ColorAttachment, DepthBias, DrawSpec, ImageDesc, ImageFormat,
    ImageUse, OffscreenRhi, Pass, PassKind, PipelineDesc, RhiError, Target, Transition,
};

/// The value the fragment shader writes — well above 1.0, which is the whole
/// question this file asks of the format.
const BRIGHT: f32 = 8.0;

fn shader(source: &str, stem: &str) -> gg_shaders::CompiledModule {
    let dir = std::env::temp_dir().join(format!("gg-rhi-hdr-{}-{stem}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = format!("{stem}.slang");
    std::fs::write(dir.join(&file), source).unwrap();
    gg_shaders::compile_module(&dir.to_string_lossy(), &file).unwrap()
}

/// IEEE-754 binary16 → `f32`, for reading a float target back. Ten lines rather
/// than a dependency: this is the only place in the tree that decodes one, and
/// the values under test are ordinary finite numbers.
fn f16(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);
    // Bit surgery rather than arithmetic: both formats put the mantissa at the
    // bottom, so a normal value is a bias swap (15 → 127) and a 13-bit shift.
    match exponent {
        0 if mantissa == 0 => f32::from_bits(sign),
        // Subnormal: no implicit leading one, exponent fixed at -14, so the
        // value is mantissa * 2^-24. The literal is that power exactly.
        0 => f32::from_bits(sign | (mantissa as f32 / 16_777_216.0).to_bits()),
        // 31 is inf/NaN; nothing here produces one, and reporting it as the
        // largest finite value would hide a bug rather than surface it.
        31 => f32::INFINITY,
        e => f32::from_bits(sign | ((e + 112) << 23) | (mantissa << 13)),
    }
}

/// A float target keeps what a forward pass wrote above 1.0. An `Rgba8` one
/// would hand the tonemapper a wall of 1.0 and every highlight would be gone
/// before the curve that exists to shape them ever ran.
#[test]
fn a_float_target_carries_radiance_past_one() {
    let module = shader(
        &format!(
            r#"
struct VOut {{ float4 pos : SV_Position; }}

[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{{
    static const float2 verts[3] = {{ float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) }};
    VOut o;
    o.pos = float4(verts[vid], 0.0, 1.0);
    return o;
}}

[shader("fragment")]
float4 fs_main() : SV_Target {{ return float4({BRIGHT}, 0.5, 0.0, 1.0); }}
"#
        ),
        "bright",
    );
    let find = |stage| {
        module
            .entry_points
            .iter()
            .find(|e| e.stage == stage)
            .unwrap()
    };
    let vs = find(gg_shaders::Stage::Vertex);
    let fs = find(gg_shaders::Stage::Fragment);

    const EXTENT: (u32, u32) = (8, 8);
    let mut rhi = OffscreenRhi::new(EXTENT).unwrap();
    let pipeline = rhi
        .create_pipeline(&PipelineDesc {
            name: "hdr",
            vs_spirv: &vs.spirv,
            vs_entry: &vs.spirv_entry,
            fs_spirv: &fs.spirv,
            fs_entry: &fs.spirv_entry,
            push_constant_size: 0,
            color: gg_rhi::ColorTarget::Format(ImageFormat::Rgba16F),
            blend: gg_rhi::Blend::Off,
            depth: gg_rhi::DepthMode::Off,
            depth_bias: false,
        })
        .unwrap();
    let target = rhi
        .create_image(&ImageDesc {
            name: "test.hdr",
            extent: EXTENT,
            format: ImageFormat::Rgba16F,
            usage: ImageUse::ColorTarget,
            mip_levels: 1,
        })
        .unwrap();
    let readback = rhi
        .create_buffer(&BufferDesc {
            name: "test.hdr.readback",
            size: ImageFormat::Rgba16F.packed_size(EXTENT),
            kind: BufferKind::Readback,
        })
        .unwrap();

    let draws = [DrawSpec {
        pipeline,
        push_constants: &[],
        count: 3,
        index_buffer: None,
        indirect: None,
        depth_bias: None,
    }];
    let to_attachment = [Transition {
        target: Target::Image(target),
        from: Access::None,
        to: Access::ColorWrite,
    }];
    let to_copy = [Transition {
        target: Target::Image(target),
        from: Access::ColorWrite,
        to: Access::CopySrc,
    }];
    let to_host = [Transition {
        target: Target::Buffer(readback),
        from: Access::CopyDst,
        to: Access::HostRead,
    }];
    rhi.execute(&[
        Pass {
            name: "test.hdr.draw",
            transitions: &to_attachment,
            kind: PassKind::Render {
                color: Some(ColorAttachment {
                    target: Target::Image(target),
                    clear: Some([0.0, 0.0, 0.0, 1.0]),
                }),
                depth: None,
                viewport: None,
                draws: &draws,
            },
        },
        Pass {
            name: "test.hdr.readback",
            transitions: &to_copy,
            kind: PassKind::Readback {
                source: Target::Image(target),
                dest: readback,
            },
        },
        Pass {
            name: "test.hdr.host-visible",
            transitions: &to_host,
            kind: PassKind::Barriers,
        },
    ])
    .unwrap();

    let bytes = rhi.map_buffer(readback).unwrap().to_vec();
    assert_eq!(bytes.len() as u64, ImageFormat::Rgba16F.packed_size(EXTENT));
    for texel in bytes.chunks_exact(8) {
        let channel = |i: usize| f16(u16::from_le_bytes([texel[i * 2], texel[i * 2 + 1]]));
        // binary16 holds 8.0 and 0.5 exactly, so this is equality rather than a
        // tolerance — the failure being guarded against is a *clamp*, not
        // rounding.
        assert_eq!(channel(0), BRIGHT, "red clamped");
        assert_eq!(channel(1), 0.5, "green moved");
    }

    rhi.destroy_buffer(readback).unwrap();
    rhi.destroy_image(target).unwrap();
    rhi.destroy_pipeline(pipeline).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// Only a depth image that asked to be sampled gets a bindless slot. Refusing
/// the other one *here* is the point: without the check it fails inside the
/// descriptor write, with a message about a slot rather than about the image.
#[test]
fn only_a_shadow_map_gets_a_bindless_slot() {
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
    let prepass = rhi
        .create_image(&ImageDesc {
            name: "test.prepass",
            extent: (8, 8),
            format: ImageFormat::Depth32,
            usage: ImageUse::Depth,
            mip_levels: 1,
        })
        .unwrap();
    let refused = rhi.register_texture(prepass);
    assert!(
        matches!(&refused, Err(RhiError::Loader(m)) if m.contains("DepthSampled")),
        "{refused:?}"
    );

    let shadow = rhi
        .create_image(&ImageDesc {
            name: "test.shadow",
            extent: (8, 8),
            format: ImageFormat::Depth32,
            usage: ImageUse::DepthSampled,
            mip_levels: 1,
        })
        .unwrap();
    rhi.register_texture(shadow).unwrap();

    // A depth *format* with a colour usage stays refused: the two have to agree
    // whichever way they disagree.
    let mismatch = rhi.create_image(&ImageDesc {
        name: "test.confused",
        extent: (8, 8),
        format: ImageFormat::Depth32,
        usage: ImageUse::ColorTarget,
        mip_levels: 1,
    });
    assert!(mismatch.is_err(), "a depth format is a depth attachment");

    rhi.destroy_image(shadow).unwrap();
    rhi.destroy_image(prepass).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// Declared and supplied, or neither. Both mismatches are silent at runtime —
/// an unset dynamic state biases by whatever was left in the command buffer,
/// and a bias handed to a pipeline that never enabled it is a knob the operator
/// turns while nothing moves.
#[test]
fn a_depth_bias_is_required_exactly_where_it_is_declared() {
    let module = shader(
        r#"
struct VOut { float4 pos : SV_Position; }

[shader("vertex")]
VOut vs_main(uint vid: SV_VertexID)
{
    static const float2 verts[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut o;
    o.pos = float4(verts[vid], 0.5, 1.0);
    return o;
}

[shader("fragment")]
float4 fs_main() : SV_Target { return float4(0.0, 1.0, 0.0, 1.0); }
"#,
        "biased",
    );
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
    let mut make = |name: &str, depth_bias: bool| {
        rhi.create_pipeline(&PipelineDesc {
            name,
            vs_spirv: &vs.spirv,
            vs_entry: &vs.spirv_entry,
            fs_spirv: &fs.spirv,
            fs_entry: &fs.spirv_entry,
            push_constant_size: 0,
            color: gg_rhi::ColorTarget::Backbuffer,
            blend: gg_rhi::Blend::Off,
            depth: gg_rhi::DepthMode::Off,
            depth_bias,
        })
        .unwrap()
    };
    let plain = make("plain", false);
    let biased = make("biased", true);

    let bias = Some(DepthBias {
        constant: 2.0,
        slope: 1.5,
    });
    let spec = |pipeline, depth_bias| DrawSpec {
        pipeline,
        push_constants: &[],
        count: 3,
        index_buffer: None,
        indirect: None,
        depth_bias,
    };

    let mut run = |draw| {
        let draws = [draw];
        let to_attachment = [Transition {
            target: Target::Backbuffer,
            from: Access::None,
            to: Access::ColorWrite,
        }];
        rhi.execute(&[Pass {
            name: "test.bias",
            transitions: &to_attachment,
            kind: PassKind::Render {
                color: Some(ColorAttachment {
                    target: Target::Backbuffer,
                    clear: Some([0.0, 0.0, 0.0, 1.0]),
                }),
                depth: None,
                viewport: None,
                draws: &draws,
            },
        }])
    };

    assert!(run(spec(plain, bias)).is_err(), "undeclared, supplied");
    assert!(run(spec(biased, None)).is_err(), "declared, unsupplied");
    run(spec(plain, None)).unwrap();
    run(spec(biased, bias)).unwrap();

    rhi.destroy_pipeline(plain).unwrap();
    rhi.destroy_pipeline(biased).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}
