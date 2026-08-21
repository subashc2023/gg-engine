//! The two-pass graph gg-rhi's own tests render through.
//!
//! Spelled out rather than borrowed from `gg-render`: this crate cannot depend
//! on the one that declares graphs, and these are the *execution* layer's tests
//! — a helper that hid the declarations would hide exactly what they exercise.

#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

use gg_rhi::{
    Access, BufferDesc, BufferHandle, BufferKind, ColorAttachment, DEPTH_CLEAR, DepthAttachment,
    DrawSpec, ImageDesc, ImageFormat, ImageHandle, ImageUse, OffscreenRhi, Pass, PassKind,
    RhiError, Target, Transition,
};

/// Bytes per pixel of the offscreen target's RGBA8 format.
const BYTES_PER_PIXEL: u64 = 4;

/// A scratch directory a recycled PID cannot make lie (§6 M89).
///
/// Named for the process because that is what makes a failure's leavings
/// findable, and **cleared on the way in** because a name is not an isolation
/// claim: Windows reuses PIDs, nothing here removed its directory, and
/// `draw_covers_target_and_cache_persists` read a four-day-old 4090 pipeline
/// blob as its own run's output. Removed on the way out *unless the test is
/// failing*, which is the only reading under which both halves are wanted —
/// evidence survives a red run and a green one leaves nothing behind.
pub struct Scratch(std::path::PathBuf);

impl Scratch {
    /// `%TEMP%/gg-rhi-<what>-<pid>`, empty, existing.
    pub fn new(what: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("gg-rhi-{what}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    /// The directory, for the `&str` every `gg_shaders` entry point takes.
    pub fn path(&self) -> &std::path::Path {
        &self.0
    }

    /// Write `source` into the scratch and hand back the directory as Slang
    /// wants it — a search path plus a file name, never a joined path.
    pub fn slang(&self, file: &str, source: &str) -> String {
        std::fs::write(self.0.join(file), source).unwrap();
        self.0.to_string_lossy().into_owned()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if !std::thread::panicking() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

/// Clear the target to `clear`, run `draws` into it, and copy it back:
/// tightly packed RGBA8, row-major, sRGB-encoded (PNG-ready bytes, §4.10).
pub fn render(
    rhi: &mut OffscreenRhi,
    clear: [f32; 4],
    draws: &[DrawSpec<'_>],
) -> Result<Vec<u8>, RhiError> {
    render_inner(rhi, clear, draws, None, None)
}

/// As [`render`], with a depth attachment on the draw pass — what a pipeline
/// in [`DepthMode::Write`] needs, since dynamic rendering makes a pipeline and
/// its pass agree on attachment formats.
pub fn render_with_depth(
    rhi: &mut OffscreenRhi,
    clear: [f32; 4],
    draws: &[DrawSpec<'_>],
) -> Result<Vec<u8>, RhiError> {
    let depth = rhi.create_image(&ImageDesc {
        name: "test.depth",
        extent: rhi.extent(),
        format: ImageFormat::Depth32,
        usage: ImageUse::Depth,
        mip_levels: 1,
        samples: gg_rhi::Samples::X1,
    })?;
    let pixels = render_inner(rhi, clear, draws, None, Some(depth));
    rhi.destroy_image(depth)?;
    pixels
}

/// As [`render`], into a caller-owned readback buffer.
pub fn render_into(
    rhi: &mut OffscreenRhi,
    clear: [f32; 4],
    draws: &[DrawSpec<'_>],
    dest: BufferHandle,
) -> Result<Vec<u8>, RhiError> {
    render_inner(rhi, clear, draws, Some(dest), None)
}

fn render_inner(
    rhi: &mut OffscreenRhi,
    clear: [f32; 4],
    draws: &[DrawSpec<'_>],
    dest: Option<BufferHandle>,
    depth: Option<ImageHandle>,
) -> Result<Vec<u8>, RhiError> {
    let owned = match dest {
        Some(dest) => dest,
        None => {
            let extent = rhi.extent();
            rhi.create_buffer(&BufferDesc {
                name: "test.readback",
                size: u64::from(extent.0) * u64::from(extent.1) * BYTES_PER_PIXEL,
                kind: BufferKind::Readback,
            })?
        }
    };
    let pixels = record(rhi, clear, draws, owned, depth);
    if dest.is_none() {
        rhi.destroy_buffer(owned)?;
    }
    pixels
}

fn record(
    rhi: &mut OffscreenRhi,
    clear: [f32; 4],
    draws: &[DrawSpec<'_>],
    dest: BufferHandle,
    depth: Option<ImageHandle>,
) -> Result<Vec<u8>, RhiError> {
    let mut to_attachment = vec![Transition {
        target: Target::Backbuffer,
        from: Access::None,
        to: Access::ColorWrite,
    }];
    if let Some(depth) = depth {
        to_attachment.push(Transition {
            target: Target::Image(depth),
            from: Access::None,
            to: Access::DepthWrite,
        });
    }
    let to_copy = [Transition {
        target: Target::Backbuffer,
        from: Access::ColorWrite,
        to: Access::CopySrc,
    }];
    let to_host = [Transition {
        target: Target::Buffer(dest),
        from: Access::CopyDst,
        to: Access::HostRead,
    }];
    let passes = [
        Pass {
            name: "test.draw",
            transitions: &to_attachment,
            kind: PassKind::Render {
                color: Some(ColorAttachment {
                    target: Target::Backbuffer,
                    clear: Some(clear),
                    resolve: None,
                }),
                depth: depth.map(|target| DepthAttachment {
                    target: Target::Image(target),
                    clear: Some(DEPTH_CLEAR),
                    store: false,
                    read_only: false,
                    resolve: None,
                }),
                viewport: None,
                draws,
            },
        },
        Pass {
            name: "test.readback",
            transitions: &to_copy,
            kind: PassKind::Readback {
                source: Target::Backbuffer,
                dest,
            },
        },
        Pass {
            name: "test.host-visible",
            transitions: &to_host,
            kind: PassKind::Barriers,
        },
    ];
    rhi.execute(&passes)?;
    Ok(rhi.map_buffer(dest)?.to_vec())
}
