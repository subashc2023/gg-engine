//! The render graph's **execution layer** (§4.5) — the Vulkan half of the seam.
//!
//! §4.5 splits the graph in two. Passes declare *virtual resource + usage* in
//! engine terms, and that half is API-neutral by construction: it is
//! `gg-render`'s, and a second backend re-derives its own synchronization from
//! the same declarations (§2, Console portability row). Deriving
//! `synchronization2` barriers and layout transitions from those declarations is
//! Vulkan-specific, so it lives here — and **only** here. This module is the one
//! place in the engine that writes a barrier for a pass; the grep gate says so.
//!
//! What crosses the seam is [`Pass`]: a name, the [`Transition`]s the graph
//! derived for it, and what it records. What does not cross is a stage mask, an
//! access mask or a layout — those are [`Access`]'s three private mappings
//! below, and nothing outside this crate can name one.
//!
//! The staging ring's queue-family ownership transfers are deliberately not
//! here (`upload.rs`). An ownership transfer is a property of the *transfer*,
//! paired across two queues and two submits and ordered by a timeline; the v1
//! graph is single-queue and models pass dependencies, not queue handoffs.

use crate::RhiError;
use crate::crash::Crumbs;
use crate::gpu::Gpu;
use crate::pipeline::{ResolvedDraw, record_draw};
use crate::resource::{BufferHandle, ImageHandle};
use crate::timing::Stamps;
use crate::{DrawSpec, device::Device};
use ash::vk;

/// How a pass touches a resource — the whole vocabulary a pass declares in.
///
/// Every variant maps to exactly one (stage, access, layout) triple below, and
/// a transition between two of them is one barrier. Buffer-only variants have
/// no layout; image-only variants are never applied to a buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// Nothing has touched it yet this frame and its contents are discardable —
    /// the state every attachment starts in, and the only one whose layout is
    /// `UNDEFINED`.
    None,
    /// Written as a color attachment.
    ColorWrite,
    /// Tested and written as a depth attachment.
    DepthWrite,
    /// Tested but not written — a forward pass reading a prepass's depth.
    DepthRead,
    /// Read through the bindless sampled-image array in a fragment shader.
    SampledRead,
    /// Source of a copy.
    CopySrc,
    /// Destination of a copy.
    CopyDst,
    /// Read by the host after a copy (buffers).
    HostRead,
    /// Reachable through the bindless storage-image array. Not a pass
    /// declaration: it is the resting layout a storage image is registered
    /// into, and the graph never moves one.
    StorageReadWrite,
    /// Handed to the presentation engine. Terminal: nothing follows it.
    Present,
}

impl Access {
    /// Where the access happens. `None` is `ALL_COMMANDS` rather than
    /// `TOP_OF_PIPE` because it pairs with an `UNDEFINED` old layout, where the
    /// barrier's job is the transition and not a wait — the access mask is
    /// empty, so the conservative stage costs an execution dependency and no
    /// cache flush.
    fn stage(self) -> vk::PipelineStageFlags2 {
        match self {
            Access::None => vk::PipelineStageFlags2::ALL_COMMANDS,
            Access::ColorWrite => vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            Access::DepthWrite | Access::DepthRead => {
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
            }
            Access::SampledRead => vk::PipelineStageFlags2::FRAGMENT_SHADER,
            Access::StorageReadWrite => vk::PipelineStageFlags2::ALL_COMMANDS,
            Access::CopySrc | Access::CopyDst => vk::PipelineStageFlags2::COPY,
            Access::HostRead => vk::PipelineStageFlags2::HOST,
            // Present is a queue operation ordered by the present semaphore, not
            // by a stage; naming one here would claim a dependency the barrier
            // does not provide.
            Access::Present => vk::PipelineStageFlags2::NONE,
        }
    }

    fn mask(self) -> vk::AccessFlags2 {
        match self {
            Access::None | Access::Present => vk::AccessFlags2::NONE,
            Access::ColorWrite => {
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::COLOR_ATTACHMENT_READ
            }
            Access::DepthWrite => {
                vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
            }
            Access::DepthRead => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            Access::SampledRead => vk::AccessFlags2::SHADER_SAMPLED_READ,
            Access::StorageReadWrite => {
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE
            }
            Access::CopySrc => vk::AccessFlags2::TRANSFER_READ,
            Access::CopyDst => vk::AccessFlags2::TRANSFER_WRITE,
            Access::HostRead => vk::AccessFlags2::HOST_READ,
        }
    }

    fn layout(self) -> vk::ImageLayout {
        match self {
            Access::None => vk::ImageLayout::UNDEFINED,
            Access::ColorWrite => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            Access::DepthWrite => vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            Access::DepthRead => vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL,
            Access::SampledRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            Access::StorageReadWrite => vk::ImageLayout::GENERAL,
            Access::CopySrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            Access::CopyDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            Access::Present => vk::ImageLayout::PRESENT_SRC_KHR,
            // Host access is a buffer's business; an image never reaches it.
            Access::HostRead => vk::ImageLayout::GENERAL,
        }
    }
}

/// What a declaration points at. `Backbuffer` is late-bound on purpose: on a
/// windowed context it is whichever swapchain image the acquire handed over,
/// which is not known until after the graph is compiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Where this context's frame lands: the acquired swapchain image, or the
    /// offscreen target.
    Backbuffer,
    /// An image the caller created.
    Image(ImageHandle),
    /// A buffer the caller created.
    Buffer(BufferHandle),
}

/// One resource moving from one access to another — the unit the graph derives
/// and this module turns into a barrier.
#[derive(Clone, Copy, Debug)]
pub struct Transition {
    /// What moves.
    pub target: Target,
    /// Where it has been. `Access::None` discards the contents.
    pub from: Access,
    /// What the pass about to run needs.
    pub to: Access,
}

/// A pass's color attachment. Always stored: a color attachment nothing reads
/// is a pass the graph should not have run.
#[derive(Clone, Copy, Debug)]
pub struct ColorAttachment {
    /// What is written.
    pub target: Target,
    /// Clear to this (linear values; an sRGB target encodes) or load what is
    /// there.
    pub clear: Option<[f32; 4]>,
    /// Where a multisampled `target` is averaged down to, at the end of the
    /// pass. Present exactly when `target` has more than one sample.
    ///
    /// Part of the attachment rather than a pass of its own: dynamic rendering
    /// resolves *as* the render ends, so there is no second pass to order, no
    /// barrier between the two, and — the reason it matters — the multisample
    /// attachment is then never stored at all. What the next pass reads is this
    /// one image, so the graph derives one dependency and not three.
    pub resolve: Option<Target>,
}

/// Where in its attachments a pass draws, in pixels from the top-left — the
/// viewport and the scissor, which are one rectangle here because a pass that
/// set them apart would be a pass drawing where it said it would not.
///
/// Absent on a pass means the whole attachment, which is what every pass that
/// owns its target wants. It is present when a pass *composites*: the render
/// area still covers the attachment, so a clear reaches every pixel, while the
/// draws land only here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Viewport {
    /// Left edge.
    pub x: u32,
    /// Top edge.
    pub y: u32,
    /// Width in pixels. Zero draws nothing rather than being an error — a
    /// window mid-resize is allowed to have no room for a viewport.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Viewport {
    /// The whole of an attachment `extent` pixels across.
    #[must_use]
    pub fn whole(extent: (u32, u32)) -> Viewport {
        Viewport {
            x: 0,
            y: 0,
            width: extent.0,
            height: extent.1,
        }
    }

    /// This rectangle cut down to an attachment `extent` pixels across.
    ///
    /// Clamped rather than refused because the caller is a window: a viewport
    /// derived from last frame's extent arrives one frame stale after a resize,
    /// and a scissor past the render area is a validation error the frame after
    /// it would fix by itself.
    #[must_use]
    pub fn clamped(self, extent: (u32, u32)) -> Viewport {
        let x = self.x.min(extent.0);
        let y = self.y.min(extent.1);
        Viewport {
            x,
            y,
            width: self.width.min(extent.0 - x),
            height: self.height.min(extent.1 - y),
        }
    }
}

/// A pass's depth attachment. The three flags are three Vulkan knobs, named
/// one-to-one so a pass says what it means: `clear` is the load op, `store` the
/// store op, and `read_only` the layout — which must agree with the pipeline's
/// [`DepthMode`](crate::DepthMode).
#[derive(Clone, Copy, Debug)]
pub struct DepthAttachment {
    /// What is tested against.
    pub target: Target,
    /// Reverse-Z clears to `0.0` (§2, Math row).
    pub clear: Option<f32>,
    /// Keep the result for a later pass. A depth prepass stores; a forward pass
    /// that owns its own depth does not.
    pub store: bool,
    /// Tested but not written — pair with `Access::DepthRead`.
    pub read_only: bool,
}

/// What a pass records once its transitions are in.
pub enum PassKind<'a> {
    /// A dynamic-rendering pass. Either attachment may be absent: a depth
    /// prepass has no color, and a fullscreen pass has no depth.
    Render {
        /// The color attachment, if any.
        color: Option<ColorAttachment>,
        /// The depth attachment, if any.
        depth: Option<DepthAttachment>,
        /// Where the draws land. `None` is the whole attachment.
        viewport: Option<Viewport>,
        /// Drawn in order.
        draws: &'a [DrawSpec<'a>],
    },
    /// Nothing but this pass's transitions — the frame's handoff to the
    /// presentation engine, and any other pure state change a graph derives.
    /// A pass rather than a special case in the executor, so that "every
    /// barrier belongs to a pass" stays true without exception.
    Barriers,
    /// Image → buffer copy: the §4.5 readback pass, and the only way pixels
    /// reach the host.
    Readback {
        /// The image copied out. Must be in `Access::CopySrc`.
        source: Target,
        /// A [`BufferKind::Readback`](crate::BufferKind) buffer, large enough
        /// for the tightly packed image.
        dest: BufferHandle,
    },
}

/// One compiled pass: what it is called, what has to move before it runs, and
/// what it records. `gg-render` produces these; nothing else should.
pub struct Pass<'a> {
    /// Debug name — the label a capture shows, and the name the dump prints.
    pub name: &'a str,
    /// Derived by the graph from the declarations, in the order they apply.
    pub transitions: &'a [Transition],
    /// The body.
    pub kind: PassKind<'a>,
}

/// What the recorder writes *beside* a frame's own commands (§4.8): the
/// timestamp pair and the breadcrumb marks. One argument because they bracket
/// the same span, are prepared together, and neither is the frame's business.
#[derive(Clone, Copy)]
pub(crate) struct Instruments {
    /// Absent without the `gpu-timings` feature or on a device that writes no
    /// usable timestamp; the breadcrumbs are unconditional.
    pub stamps: Option<Stamps>,
    pub crumbs: Crumbs,
}

/// An image resolved to the handles recording needs.
#[derive(Clone, Copy)]
pub(crate) struct Bound {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub extent: (u32, u32),
    pub aspect: vk::ImageAspectFlags,
}

/// A resolved reference, still late-bound for the backbuffer.
#[derive(Clone, Copy)]
enum Ref {
    Bound(Bound),
    Backbuffer,
}

impl Ref {
    fn get(self, backbuffer: &Bound) -> Bound {
        match self {
            Ref::Bound(b) => b,
            Ref::Backbuffer => *backbuffer,
        }
    }
}

enum ResolvedTransition {
    Image {
        target: Ref,
        from: Access,
        to: Access,
    },
    Buffer {
        buffer: vk::Buffer,
        size: u64,
        from: Access,
        to: Access,
    },
}

/// What a render pass draws into, resolved. A struct because the three travel
/// together everywhere and the recorder takes them all at once.
struct Attachments {
    color: Option<(Ref, Option<[f32; 4]>, Option<Ref>)>,
    depth: Option<(Ref, Option<f32>, bool, bool)>,
    viewport: Option<Viewport>,
}

enum ResolvedKind<'a> {
    Render {
        into: Attachments,
        draws: Vec<ResolvedDraw<'a>>,
    },
    Readback {
        source: Ref,
        dest: vk::Buffer,
    },
    Barriers,
}

/// A pass with every handle looked up — so recording, which happens after the
/// swapchain acquire, cannot fail. See [`Rhi::execute`](crate::Rhi::execute).
pub(crate) struct ResolvedPass<'a> {
    name: &'a str,
    transitions: Vec<ResolvedTransition>,
    kind: ResolvedKind<'a>,
}

/// Look every handle up and validate every draw, ahead of recording.
///
/// This runs *before* the acquire deliberately: a stale handle or a
/// push-constant length mismatch is a normal caller-reachable error, and
/// returning one between acquire and submit would leave a signaled semaphore
/// with no waiter — which the next frame to reuse that slot trips over as a
/// validation error, permanently failing the §4.3 zero-message report.
pub(crate) fn resolve<'a>(
    gpu: &Gpu,
    passes: &[Pass<'a>],
) -> Result<Vec<ResolvedPass<'a>>, RhiError> {
    passes
        .iter()
        .map(|pass| {
            let transitions = pass
                .transitions
                .iter()
                .map(|t| resolve_transition(gpu, t))
                .collect::<Result<Vec<_>, _>>()?;
            let kind = match &pass.kind {
                PassKind::Render {
                    color,
                    depth,
                    viewport,
                    draws,
                } => {
                    if color.is_none() && depth.is_none() {
                        return Err(RhiError::Loader(format!(
                            "pass `{}` renders into nothing — a pass with no attachment has no \
                             render area",
                            pass.name
                        )));
                    }
                    ResolvedKind::Render {
                        into: Attachments {
                            color: color
                                .map(|c| {
                                    Ok::<_, RhiError>((
                                        resolve_image(gpu, c.target)?,
                                        c.clear,
                                        c.resolve.map(|r| resolve_image(gpu, r)).transpose()?,
                                    ))
                                })
                                .transpose()?,
                            depth: depth
                                .map(|d| {
                                    Ok::<_, RhiError>((
                                        resolve_image(gpu, d.target)?,
                                        d.clear,
                                        d.store,
                                        d.read_only,
                                    ))
                                })
                                .transpose()?,
                            viewport: *viewport,
                        },
                        draws: resolve_draws(gpu, draws)?,
                    }
                }
                PassKind::Readback { source, dest } => ResolvedKind::Readback {
                    source: resolve_image(gpu, *source)?,
                    dest: gpu.resources.buffer(*dest)?.raw,
                },
                PassKind::Barriers => ResolvedKind::Barriers,
            };
            Ok(ResolvedPass {
                name: pass.name,
                transitions,
                kind,
            })
        })
        .collect()
}

fn resolve_image(gpu: &Gpu, target: Target) -> Result<Ref, RhiError> {
    match target {
        Target::Backbuffer => Ok(Ref::Backbuffer),
        Target::Image(handle) => {
            let image = gpu.resources.image(handle)?;
            Ok(Ref::Bound(Bound {
                image: image.raw,
                view: image.view,
                extent: image.extent,
                aspect: image.format.aspect(),
            }))
        }
        Target::Buffer(_) => Err(RhiError::Loader(
            "a buffer cannot be an attachment or a copy source".into(),
        )),
    }
}

fn resolve_transition(gpu: &Gpu, t: &Transition) -> Result<ResolvedTransition, RhiError> {
    match t.target {
        Target::Buffer(handle) => {
            let buffer = gpu.resources.buffer(handle)?;
            Ok(ResolvedTransition::Buffer {
                buffer: buffer.raw,
                size: buffer.size,
                from: t.from,
                to: t.to,
            })
        }
        target => Ok(ResolvedTransition::Image {
            target: resolve_image(gpu, target)?,
            from: t.from,
            to: t.to,
        }),
    }
}

/// Resolve draws, checking the push-constant length here rather than at record
/// time so `record` is infallible (see [`resolve`]).
fn resolve_draws<'a>(gpu: &Gpu, draws: &[DrawSpec<'a>]) -> Result<Vec<ResolvedDraw<'a>>, RhiError> {
    draws
        .iter()
        .map(|d| {
            let entry = *gpu.pipelines.get(d.pipeline)?;
            if d.push_constants.len() != entry.push_constant_size as usize {
                return Err(RhiError::Loader(format!(
                    "draw passed {} push-constant bytes; the pipeline declares {} (§4.4: layouts \
                     are checked, not assumed)",
                    d.push_constants.len(),
                    entry.push_constant_size
                )));
            }
            // Both directions, because both are silent failures: an unset
            // dynamic state biases by whatever was left in the command buffer,
            // and a bias handed to a pipeline that did not enable it is a knob
            // the operator turns while nothing moves.
            if d.depth_bias.is_some() != entry.depth_bias {
                return Err(RhiError::Loader(format!(
                    "draw {} a depth bias and its pipeline {} one — dynamic state is required \
                     exactly where it is declared",
                    if d.depth_bias.is_some() {
                        "passes"
                    } else {
                        "passes no"
                    },
                    if entry.depth_bias {
                        "declares"
                    } else {
                        "declares no"
                    },
                )));
            }
            let index_buffer = d
                .index_buffer
                .map(|h| gpu.resources.buffer(h).map(|b| b.raw))
                .transpose()?;
            let indirect = d
                .indirect
                .map(|i| gpu.resources.buffer(i.buffer).map(|b| (b.raw, i.offset)))
                .transpose()?;
            Ok(ResolvedDraw {
                entry,
                push_constants: d.push_constants,
                count: d.count,
                index_buffer,
                indirect,
                depth_bias: d.depth_bias,
                viewport: d.viewport,
            })
        })
        .collect()
}

/// Record every pass into `cmd`: its barriers, then its body. Infallible by
/// construction — [`resolve`] did everything that can fail.
///
/// `stamps` brackets each pass with a timestamp pair (§4.8). The opening one is
/// written *before* the barrier deliberately: §4.5 makes every barrier belong to
/// a pass, so a pass that stalls on its own transition should read as expensive
/// rather than as free. `crumbs` brackets the same span with breadcrumb marks,
/// which is what names the pass a lost device was inside.
///
/// # Safety
/// `cmd` must be recording on the graphics family, `backbuffer` must name a live
/// image of this device, and `set` must be the global bindless set.
pub(crate) unsafe fn record(
    device: &Device,
    cmd: vk::CommandBuffer,
    set: vk::DescriptorSet,
    passes: &[ResolvedPass<'_>],
    backbuffer: Bound,
    instruments: Instruments,
) {
    let Instruments { stamps, crumbs } = instruments;
    if let Some(stamps) = stamps {
        // SAFETY: caller contract — cmd is recording, and the range is the
        // caller's slot, whose previous frame the caller proved retired.
        unsafe { stamps.reset(device, cmd) };
    }
    for (index, pass) in passes.iter().enumerate() {
        let timed = stamps.filter(|s| s.covers(index));
        let tracked = crumbs.covers(index);
        // SAFETY: caller contract, and every handle below came from resolve.
        unsafe {
            device.begin_label(cmd, pass.name);
            if let Some(stamps) = timed {
                stamps.begin(device, cmd, index);
            }
            if tracked {
                crumbs.begin(device, cmd, index);
            }
            barrier(device, cmd, &pass.transitions, &backbuffer);
            match &pass.kind {
                ResolvedKind::Render { into, draws } => {
                    render(device, cmd, set, into, draws, &backbuffer);
                }
                ResolvedKind::Readback { source, dest } => {
                    readback(device, cmd, source.get(&backbuffer), *dest);
                }
                ResolvedKind::Barriers => {}
            }
            if let Some(stamps) = timed {
                stamps.end(device, cmd, index);
            }
            if tracked {
                crumbs.end(device, cmd, index);
            }
            device.end_label(cmd);
        }
    }
}

/// One `vkCmdPipelineBarrier2` per pass — every transition it declared, in one
/// dependency, which is what `synchronization2` is for.
///
/// # Safety
/// `cmd` is recording; every handle is live.
unsafe fn barrier(
    device: &Device,
    cmd: vk::CommandBuffer,
    transitions: &[ResolvedTransition],
    backbuffer: &Bound,
) {
    if transitions.is_empty() {
        return;
    }
    let mut images = Vec::new();
    let mut buffers = Vec::new();
    for t in transitions {
        match t {
            ResolvedTransition::Image { target, from, to } => {
                let bound = target.get(backbuffer);
                images.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(bound.image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(bound.aspect)
                                .level_count(1)
                                .layer_count(1),
                        )
                        .old_layout(from.layout())
                        .new_layout(to.layout())
                        .src_stage_mask(from.stage())
                        .src_access_mask(from.mask())
                        .dst_stage_mask(to.stage())
                        .dst_access_mask(to.mask()),
                );
            }
            ResolvedTransition::Buffer {
                buffer,
                size,
                from,
                to,
            } => buffers.push(
                vk::BufferMemoryBarrier2::default()
                    .buffer(*buffer)
                    .size(*size)
                    .src_stage_mask(from.stage())
                    .src_access_mask(from.mask())
                    .dst_stage_mask(to.stage())
                    .dst_access_mask(to.mask()),
            ),
        }
    }
    let info = vk::DependencyInfo::default()
        .image_memory_barriers(&images)
        .buffer_memory_barriers(&buffers);
    // SAFETY: caller contract; the arrays outlive the call.
    unsafe { device.raw().cmd_pipeline_barrier2(cmd, &info) };
}

/// # Safety
/// `cmd` is recording, the attachments are live, `set` is the global set.
unsafe fn render(
    device: &Device,
    cmd: vk::CommandBuffer,
    set: vk::DescriptorSet,
    into: &Attachments,
    draws: &[ResolvedDraw<'_>],
    backbuffer: &Bound,
) {
    let color = into
        .color
        .map(|(r, clear, resolve)| (r.get(backbuffer), clear, resolve.map(|r| r.get(backbuffer))));
    let depth = into
        .depth
        .map(|(r, clear, store, read_only)| (r.get(backbuffer), clear, store, read_only));
    // The render area is the color attachment's, or the depth one's when a
    // prepass has no color. resolve() refused the pass that has neither.
    let extent = color
        .map(|(b, ..)| b.extent)
        .or(depth.map(|(b, ..)| b.extent))
        .unwrap_or((1, 1));

    let color_attachments: Vec<vk::RenderingAttachmentInfo<'_>> = color
        .iter()
        .map(|(bound, clear, resolve)| {
            let info = vk::RenderingAttachmentInfo::default()
                .image_view(bound.view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                // A resolved attachment is never stored: the averaged image is
                // the pass's whole output, and storing the samples beside it
                // would write N× the bytes for something nothing reads.
                .store_op(match resolve {
                    Some(_) => vk::AttachmentStoreOp::DONT_CARE,
                    None => vk::AttachmentStoreOp::STORE,
                });
            let info = match resolve {
                Some(into) => info
                    .resolve_mode(vk::ResolveModeFlags::AVERAGE)
                    .resolve_image_view(into.view)
                    .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
                None => info,
            };
            match clear {
                Some(c) => info
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { float32: *c },
                    }),
                None => info.load_op(vk::AttachmentLoadOp::LOAD),
            }
        })
        .collect();

    let depth_attachment = depth.map(|(bound, clear, store, read_only)| {
        let layout = if read_only {
            vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL
        } else {
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
        };
        let info = vk::RenderingAttachmentInfo::default()
            .image_view(bound.view)
            .image_layout(layout)
            .store_op(if store {
                vk::AttachmentStoreOp::STORE
            } else {
                vk::AttachmentStoreOp::DONT_CARE
            });
        match clear {
            Some(d) => info
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: d,
                        stencil: 0,
                    },
                }),
            None => info.load_op(vk::AttachmentLoadOp::LOAD),
        }
    });

    let mut rendering = vk::RenderingInfo::default()
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: vk::Extent2D {
                width: extent.0,
                height: extent.1,
            },
        })
        .layer_count(1)
        .color_attachments(&color_attachments);
    if let Some(depth) = &depth_attachment {
        rendering = rendering.depth_attachment(depth);
    }

    // Clamped to the render area declared above: a scissor reaching past it is
    // a validation error, and the rectangle came from a window that may have
    // resized since (`Viewport::clamped`).
    let region = into
        .viewport
        .unwrap_or(Viewport::whole(extent))
        .clamped(extent);

    // SAFETY: caller contract — cmd is recording, every view is live, each
    // draw's pipeline was validated against its push constants in resolve(),
    // and `region` was just clamped into the render area.
    unsafe {
        device.raw().cmd_begin_rendering(cmd, &rendering);
        for draw in draws {
            // A draw's own rectangle if it declared one (§6 M31's atlas tiles),
            // clamped by the same rule so an override is no more able to scissor
            // past the render area than the pass's is.
            let region = draw.viewport.map_or(region, |v| v.clamped(extent));
            record_draw(device, cmd, region, set, draw);
        }
        device.raw().cmd_end_rendering(cmd);
    }
}

/// One image, one transition, outside any pass — what bindless registration
/// needs. Here rather than at the call site so the barrier vocabulary has one
/// definition (§4.5, and the grep gate behind it).
///
/// # Safety
/// `cmd` must be recording on the graphics family and `image` must be live.
pub(crate) unsafe fn one_shot_transition(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    from: Access,
    to: Access,
) {
    let bound = Bound {
        image,
        view: vk::ImageView::null(),
        extent: (0, 0),
        aspect,
    };
    let transitions = [ResolvedTransition::Image {
        target: Ref::Bound(bound),
        from,
        to,
    }];
    // SAFETY: caller contract.
    unsafe { barrier(device, cmd, &transitions, &bound) };
}

/// # Safety
/// `cmd` is recording; `source` is in `TRANSFER_SRC_OPTIMAL` and `dest` is large
/// enough for the tightly packed image.
unsafe fn readback(device: &Device, cmd: vk::CommandBuffer, source: Bound, dest: vk::Buffer) {
    let regions = [vk::BufferImageCopy2::default()
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(source.aspect)
                .layer_count(1),
        )
        .image_extent(vk::Extent3D {
            width: source.extent.0,
            height: source.extent.1,
            depth: 1,
        })];
    let copy = vk::CopyImageToBufferInfo2::default()
        .src_image(source.image)
        .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .dst_buffer(dest)
        .regions(&regions);
    // SAFETY: caller contract; regions outlive the call.
    unsafe { device.raw().cmd_copy_image_to_buffer2(cmd, &copy) };
}
