//! The deliberate-hang canary (§6 M8): a fragment shader that never returns,
//! and a device-lost report that names the pass it was in.
//!
//! **Ignored, and it stays ignored.** This wedges the GPU on purpose. On the
//! desk that means a driver reset — the display blanks, and every other GPU
//! process on the machine is entitled to die with it — so it is a diagnostic a
//! human runs on the real-GPU box, never a CI gate. Ordinary CI proves the same
//! machinery without wedging anything: `crash.rs` unit-tests the report the
//! marks produce, and `breadcrumbs.rs` proves the marks are written by a device.
//!
//! Lavapipe cannot answer this at all: there is no watchdog behind a CPU
//! rasterizer, so the hang is this process hanging across every core it threads
//! over, and it has no `VK_EXT_device_fault` either (§6 M8, measured). The test
//! reads the device name and refuses a software rasterizer before it compiles
//! anything — a guard, not a convenience: `xtask interactive` swept this test in
//! through `--run-ignored ignored-only` and ran it on the pinned lavapipe, and
//! the desk stopped answering. What CI *can* prove about the canary is proven
//! beside it: `the_hang_shader_loops_on_a_trip_count_it_reads_from_memory` runs
//! the same shader with a trip count small enough to retire, so a real-GPU run
//! — which costs a driver reset — is never spent finding a wiring bug.
//!
//! Run it against real hardware:
//!
//! ```text
//! GG_ADAPTER=4090 cargo nextest run -p gg-rhi --run-ignored ignored-only device_lost
//! ```
//!
//! The desk's own shell is PowerShell, which has no inline env-var prefix — and
//! `$env:` assignments outlive the command, so a stray `GG_ADAPTER` would
//! silently redirect every later GPU run in that session:
//!
//! ```text
//! $env:GG_ADAPTER='4090'; cargo nextest run -p gg-rhi --run-ignored ignored-only device_lost; Remove-Item Env:GG_ADAPTER
//! ```
//!
//! `GG_ADAPTER` needs `VK_DRIVER_FILES` unset, or the pin wins and the guard
//! above skips the run rather than the shader wedging the desk.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use gg_rhi::{
    Access, BufferDesc, BufferHandle, BufferKind, ColorAttachment, ColorTarget, DepthMode,
    DrawSpec, OffscreenRhi, Pass, PassKind, PipelineDesc, PipelineHandle, RhiError, Target,
    Transition,
};

/// Words the shader indexes through: `[0]` is the trip count, the rest are the
/// body's loads. 256 of them so the body's address varies with `i`.
const WORDS: usize = 257;

/// A full-target triangle whose fragment shader loops on a bound *read out of a
/// buffer*, adding a per-iteration load whose address moves with the counter.
///
/// The arithmetic version this replaced — a 4-billion trip count folded out of
/// `SV_Position` and a dependent FMA chain for a body — is a real `DontUnroll`
/// loop in the SPIR-V and still returned in under a second on an RTX 4090
/// (measured 2026-08-02). A driver back end may do what the SPIR-V says or
/// something it can prove equivalent, and a pure-arithmetic loop whose result
/// saturates is exactly the shape a back end is entitled to reason about. Data
/// it has to fetch is not: neither the bound nor the body can be folded,
/// hoisted or strength-reduced without reading device memory at compile time.
/// This version does hang the 4090 — measured at the TDR timeout — which is how
/// `Device::check_alive` came to exist: the wait a reset unblocks reports
/// success, and the loss was arriving a frame after the crumbs that name it.
const HANG: &str = r#"
struct P { uint64_t data; }
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
    uint* w = (uint*)push.data;
    float acc = 0.0;
    uint limit = w[0];
    [loop] for (uint i = 0; i < limit; ++i) { acc += float(w[1 + (i & 255u)]); }
    return float4(1.0, acc, 0.0, 1.0);
}
"#;

#[test]
#[ignore = "wedges the GPU on purpose — real-GPU canary, run by hand (see the module docs)"]
fn a_hung_pass_is_named_by_the_device_lost_report() {
    // The device first, and a refusal before anything is compiled: behind a CPU
    // rasterizer there is no watchdog, so this shader does not wedge a device —
    // it wedges *this process* on every core the driver threads across, and the
    // desk stops answering. It is the difference between a canary and a fork
    // bomb, and it is one environment variable away at all times.
    let mut rhi = OffscreenRhi::new((1024, 1024)).unwrap();
    let device = rhi.device_report().chosen.clone();
    let lower = device.to_lowercase();
    if ["llvmpipe", "lavapipe", "swiftshader", "software"]
        .iter()
        .any(|name| lower.contains(name))
    {
        eprintln!(
            "skipped: `{device}` is a software rasterizer, which has no watchdog to notice a \
             shader that never returns (§6 M8, measured: it has no VK_EXT_device_fault either). \
             Unset VK_DRIVER_FILES and name a real device: GG_ADAPTER=4090"
        );
        return;
    }

    // The trip count the shader reads: every iteration, for every pixel, and
    // the zeros behind it keep `acc` finite so a survivor is a compiler fact
    // rather than an overflow.
    let mut words = [0u32; WORDS];
    words[0] = u32::MAX;
    // Nothing is destroyed on the way out: the device is expected to be dead by
    // the time this returns, and `Drop` is the backstop for that.
    let (pipeline, _buffer, push) = arm(&mut rhi, &words);

    // No readback pass: the frame is expected to die, and a graph whose teardown
    // also touches the dead device would replace the error worth reading.
    let transitions = [Transition {
        target: Target::Backbuffer,
        from: Access::None,
        to: Access::ColorWrite,
    }];
    let draws = [DrawSpec {
        pipeline,
        push_constants: &push,
        count: 3,
        index_buffer: None,
        indirect: None,
        depth_bias: None,
    }];
    let passes = [Pass {
        name: "hang.draw",
        transitions: &transitions,
        kind: PassKind::Render {
            color: Some(ColorAttachment {
                target: Target::Backbuffer,
                clear: Some(BLUE),
            }),
            depth: None,
            viewport: None,
            draws: &draws,
        },
    }];

    match rhi.execute(&passes) {
        Err(RhiError::DeviceLost(report)) => {
            println!("{report}");
            assert!(
                report.contains("pass `hang.draw` was in flight"),
                "the report did not name the hung pass:\n{report}"
            );
        }
        Err(other) => panic!("expected a device-lost report, got {other:?}"),
        // A driver that survived it is a fact worth failing on: the canary's
        // whole job is proving this path fires on real hardware. Fail with the
        // *pixels*, not just the verdict — a survivor is one of two very
        // different bugs and the target says which (see `survived`).
        Ok(()) => panic!("{}", survived(&mut rhi, &draws)),
    }
}

/// The clear, chosen to be a colour the shader never writes: it returns red in
/// x, so the target discriminates the two ways this test can survive.
const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// Compile [`HANG`] and hand it a buffer holding `words`. Both tests below go
/// through here on purpose: a canary proven by a *different* shader than the one
/// CI can run proves nothing about the one that has to hang.
fn arm(
    rhi: &mut OffscreenRhi,
    words: &[u32; WORDS],
) -> (PipelineHandle, BufferHandle, [u8; size_of::<u64>()]) {
    let dir = std::env::temp_dir().join(format!("gg-rhi-hang-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("hang.slang"), HANG).unwrap();
    let module = gg_shaders::compile_module(&dir.to_string_lossy(), "hang.slang").unwrap();
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
        .unwrap();
    assert_eq!(push_size, 8, "one device address");

    let buffer = rhi
        .create_buffer(&BufferDesc {
            name: "hang.words",
            size: std::mem::size_of_val(words) as u64,
            kind: BufferKind::Storage,
        })
        .unwrap();
    rhi.upload_buffer(buffer, 0, bytemuck::cast_slice(words))
        .unwrap();
    rhi.flush_uploads().unwrap();
    let push = rhi.buffer_address(buffer).unwrap().to_le_bytes();

    let pipeline = rhi
        .create_pipeline(&PipelineDesc {
            name: "hang",
            vs_spirv: &vs.spirv,
            vs_entry: &vs.spirv_entry,
            fs_spirv: &fs.spirv,
            fs_entry: &fs.spirv_entry,
            push_constant_size: push_size,
            color: ColorTarget::Backbuffer,
            blend: gg_rhi::Blend::Off,
            depth: DepthMode::Off,
            depth_bias: false,
        })
        .unwrap();
    (pipeline, buffer, push)
}

/// The half of the canary CI can run: the *same* shader with a trip count small
/// enough to retire, proving the loop is driven by memory rather than folded.
///
/// This is what keeps a real-GPU run — which costs a driver reset — from being
/// spent discovering a wiring bug. The witness is the last word the loop reads:
/// it is only reached on the final iteration, so a loop that ran short, ran
/// zero times, or read a hoisted address leaves the green channel at 0.
#[test]
fn the_hang_shader_loops_on_a_trip_count_it_reads_from_memory() {
    let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
    const TRIPS: u32 = 200;
    let mut words = [0u32; WORDS];
    words[0] = TRIPS;
    // Read at `i == TRIPS - 1`, the one iteration a short loop skips.
    words[TRIPS as usize] = 1;
    let (pipeline, buffer, push) = arm(&mut rhi, &words);

    let pixels = common::render(
        &mut rhi,
        BLUE,
        &[DrawSpec {
            pipeline,
            push_constants: &push,
            count: 3,
            index_buffer: None,
            indirect: None,
            depth_bias: None,
        }],
    )
    .unwrap();
    assert_eq!(
        [pixels[0], pixels[1], pixels[2], pixels[3]],
        [255, 255, 0, 255],
        "red says the shader ran; green says it read the last word of its {TRIPS} iterations"
    );

    rhi.destroy_buffer(buffer).unwrap();
    rhi.destroy_pipeline(pipeline).unwrap();
    let report = rhi.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// Re-run the same draw *with* a readback and turn the pixels into a diagnosis.
/// Safe by the branch that calls it: the device just proved it is alive.
fn survived(rhi: &mut OffscreenRhi, draws: &[DrawSpec<'_>]) -> String {
    let pixels = match common::render(rhi, BLUE, draws) {
        Ok(pixels) => pixels,
        Err(err) => {
            return format!(
                "the device survived a shader that never returns; the \
             readback that would say why failed too: {err:?}"
            );
        }
    };
    let center = {
        let (w, h) = rhi.extent();
        let at = (((h / 2) * w + w / 2) * 4) as usize;
        [pixels[at], pixels[at + 1], pixels[at + 2], pixels[at + 3]]
    };
    let verdict = match center {
        // Blue survived the draw: no fragment ever ran, so the loop is not what
        // completed early — the triangle, the viewport or the draw is.
        [_, _, b, _] if b > 128 => {
            "the target still holds the clear, so the draw produced no \
             fragments at all — this is a geometry/graph bug, not a driver one"
        }
        // Red: the shader ran to completion. It read its trip count from a
        // buffer, so a back end cannot have bounded the loop — suspect the
        // upload (a zero limit word reaching the GPU reads as a finished loop).
        [r, _, _, _] if r > 128 => {
            "the shader ran and returned, so the loop terminated: check \
             that the trip-count word reached the device as u32::MAX"
        }
        _ => "the target holds neither the clear nor the shader's red",
    };
    format!("the device survived a shader that never returns — center pixel {center:?}: {verdict}")
}
