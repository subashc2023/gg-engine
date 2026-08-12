//! The render graph's **declaration layer** (§4.5) — the API-neutral half.
//!
//! Passes declare virtual resources and what they do with them; [`compile`]
//! linearizes the declarations, derives every state change between them, and
//! hands `gg-rhi` a pass list carrying those changes. Nothing here names a
//! stage, an access mask or a layout: the vocabulary is [`Access`], and turning
//! one into a barrier is the execution layer's job behind the §3 seam. That is
//! what makes the *contract* portable — a second backend re-derives its own
//! synchronization from these same declarations (§2, Console portability row).
//!
//! Two things this v1 deliberately does not do. It does not **cull** passes: a
//! four-pass graph has nothing to cull, and a pass that silently stopped running
//! is a worse failure than a pass that ran for nothing. It does not **alias**
//! transient memory (§4.5 marks that P2) — [`Transients`] pools whole images by
//! size and format, which is the recycling that pays for itself.

use gg_rhi::{
    Access, BufferHandle, ColorAttachment, DepthAttachment, DrawSpec, GraphContext, ImageDesc,
    ImageFormat, ImageHandle, ImageUse, Pass, PassKind, RhiError, Samples, Target, TextureIndex,
    Viewport,
};

/// A virtual resource inside one frame's graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceId(usize);

/// What a resource is, once the pool has given it something real.
#[derive(Clone, Copy)]
struct Resource {
    name: &'static str,
    target: Target,
    /// Where it has got to, walked forward as passes are compiled.
    state: Access,
}

/// What a pass does with the depth attachment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthUse {
    /// Tested and written, and the result kept for a later pass — a prepass.
    WriteStore,
    /// Tested and written, discarded afterwards — a forward pass that owns its
    /// own depth.
    Write,
    /// Tested only, against depth a previous pass stored.
    Test,
}

/// What a color attachment starts the pass with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Load {
    /// Clear to these linear values (an sRGB target encodes them).
    Clear([f32; 4]),
    /// Keep what is there.
    Keep,
}

/// What a pass does.
pub enum Body<'a> {
    /// Draw into attachments.
    Draw {
        /// Written as color, if any.
        color: Option<(ResourceId, Load)>,
        /// Where a multisampled `color` is averaged down at the end of the pass
        /// (§6 M21). Required exactly when `color` has more than one sample —
        /// the samples are never stored, so this is the pass's only output, and
        /// a later pass samples *this* rather than the attachment drawn into.
        resolve: Option<ResourceId>,
        /// Tested against, if any.
        depth: Option<(ResourceId, DepthUse)>,
        /// Where in the attachments the draws land; `None` is all of them. A
        /// [`Load::Clear`] still covers the whole attachment — that pairing is
        /// what fills the area a composited pass does not draw into.
        viewport: Option<Viewport>,
        /// Read through the bindless sampled array. Declared even though the
        /// draws reach them by index: an undeclared read is a missing barrier,
        /// and the index is exactly what hides it.
        samples: &'a [ResourceId],
        /// Recorded in order.
        draws: &'a [DrawSpec<'a>],
    },
    /// Copy an attachment out to a host-readable buffer (§4.5's readback pass).
    Readback {
        /// The attachment copied out.
        source: ResourceId,
        /// An imported `BufferKind::Readback` buffer, big enough for the packed
        /// image — see [`Frame::readback_buffer`].
        dest: ResourceId,
    },
}

/// The smallest graph there is: one pass into the backbuffer. The handoff to
/// the presentation engine is [`Frame::compile`]'s, so this really is one.
///
/// Here rather than in each demo because a demo that spelled out its own
/// transitions would be a demo writing barriers — which is the thing §6 M6
/// deletes from the tree.
pub fn single_pass<'a>(
    name: &'static str,
    backbuffer: ResourceId,
    load: Load,
    draws: &'a [DrawSpec<'a>],
) -> [Declared<'a>; 1] {
    [Declared {
        name,
        body: Body::Draw {
            color: Some((backbuffer, load)),
            resolve: None,
            depth: None,
            viewport: None,
            samples: &[],
            draws,
        },
    }]
}

/// The §4.5 readback pass: copy an attachment into a host-readable buffer.
/// What the golden harness appends to whatever graph a scene declares, so the
/// image it judges is the frame the demo actually renders (§4.10).
pub fn readback_pass<'a>(source: ResourceId, dest: ResourceId) -> Declared<'a> {
    Declared {
        name: "readback",
        body: Body::Readback { source, dest },
    }
}

/// One declared pass.
pub struct Declared<'a> {
    /// Debug name — the capture label, and what the dump prints.
    pub name: &'static str,
    /// What it does.
    pub body: Body<'a>,
}

/// A compiled frame: the passes in execution order, each with the transitions
/// the graph derived for it. Owns the transitions, so [`Compiled::passes`]
/// borrows rather than allocates.
pub struct Compiled<'a> {
    passes: Vec<Compiled1<'a>>,
    /// Every resource, for the dump — including any nothing touched.
    resources: Vec<(&'static str, Target)>,
}

struct Compiled1<'a> {
    name: &'static str,
    transitions: Vec<gg_rhi::Transition>,
    /// The named form of the same transitions, for the dump. Derived from one
    /// walk rather than recomputed, so what is printed cannot drift from what
    /// executes — which is half of §6 M6's exit criterion.
    edges: Vec<Edge>,
    body: CompiledBody<'a>,
}

/// One derived state change, before it loses its resource's name.
struct Edge {
    name: &'static str,
    target: Target,
    from: Access,
    to: Access,
}

enum CompiledBody<'a> {
    Draw {
        color: Option<ColorAttachment>,
        depth: Option<DepthAttachment>,
        viewport: Option<Viewport>,
        draws: &'a [DrawSpec<'a>],
    },
    Readback {
        source: Target,
        dest: BufferHandle,
    },
    /// Transitions only — see [`PassKind::Barriers`].
    Barriers,
}

impl<'a> Compiled<'a> {
    /// The pass list to hand `gg-rhi`, in execution order.
    pub fn passes(&self) -> Vec<Pass<'_>> {
        self.passes
            .iter()
            .map(|p| Pass {
                name: p.name,
                transitions: &p.transitions,
                kind: match &p.body {
                    CompiledBody::Draw {
                        color,
                        depth,
                        viewport,
                        draws,
                    } => PassKind::Render {
                        color: *color,
                        depth: *depth,
                        viewport: *viewport,
                        draws,
                    },
                    CompiledBody::Readback { source, dest } => PassKind::Readback {
                        source: *source,
                        dest: *dest,
                    },
                    CompiledBody::Barriers => PassKind::Barriers,
                },
            })
            .collect()
    }

    /// The executed order, by name. What `--dump-render-graph` claims and what
    /// [`Compiled::passes`] hands the RHI come from this one list.
    pub fn order(&self) -> impl Iterator<Item = &str> {
        self.passes.iter().map(|p| p.name)
    }

    /// The §4.5 text dump: every pass in execution order with the transitions
    /// derived for it.
    pub fn dump(&self) -> String {
        let mut out = String::from("render graph\n");
        out.push_str("  resources\n");
        for (name, target) in &self.resources {
            let what = match target {
                Target::Backbuffer => "backbuffer".to_owned(),
                Target::Image(_) => "attachment".to_owned(),
                Target::Buffer(_) => "buffer".to_owned(),
            };
            out.push_str(&format!("    {name} ({what})\n"));
        }
        out.push_str("  passes (execution order)\n");
        for (i, pass) in self.passes.iter().enumerate() {
            out.push_str(&format!("    {i}. {}\n", pass.name));
            for edge in &pass.edges {
                out.push_str(&format!(
                    "       barrier {}: {:?} -> {:?}\n",
                    edge.name, edge.from, edge.to
                ));
            }
            if let CompiledBody::Draw { draws, .. } = &pass.body {
                out.push_str(&format!("       {} draw(s)\n", draws.len()));
            }
        }
        out
    }

    /// The same graph as GraphViz, for when the text one stops fitting (§4.5).
    pub fn dot(&self) -> String {
        let mut out = String::from("digraph render {\n  rankdir=LR;\n");
        for (i, pass) in self.passes.iter().enumerate() {
            out.push_str(&format!("  p{i} [shape=box,label=\"{}\"];\n", pass.name));
            if i > 0 {
                out.push_str(&format!("  p{} -> p{i} [style=dotted];\n", i - 1));
            }
            for edge in &pass.edges {
                out.push_str(&format!(
                    "  \"{}\" -> p{i} [label=\"{:?}->{:?}\"];\n",
                    edge.name, edge.from, edge.to
                ));
            }
        }
        out.push_str("}\n");
        out
    }
}

/// The resources one frame's passes may name, and the pool behind them.
///
/// Built per frame: acquire what the frame needs, build the draws (a post pass
/// needs its input's bindless index, which is why acquisition comes first), then
/// [`Frame::compile`] the declarations.
pub struct Frame<'p> {
    pool: &'p mut Transients,
    ctx: &'p mut dyn GraphContext,
    extent: (u32, u32),
    resources: Vec<Resource>,
    textures: Vec<Option<TextureIndex>>,
    presents: bool,
}

impl<'p> Frame<'p> {
    /// Where this frame lands: the acquired swapchain image, or the offscreen
    /// target.
    pub fn backbuffer(&mut self) -> ResourceId {
        self.push("backbuffer", Target::Backbuffer, None)
    }

    /// A pooled color attachment at the frame's extent, registered in the
    /// bindless array so a later pass can sample it.
    ///
    /// The format is the caller's: the scene attachment is `Rgba16F` since M11
    /// (radiance leaves the forward pass above 1.0 and the tonemapper is what
    /// brings it down), while a pass resolving to the screen writes 8-bit sRGB.
    ///
    /// # Errors
    ///
    /// Allocation, or a full bindless array.
    pub fn color(
        &mut self,
        name: &'static str,
        format: ImageFormat,
    ) -> Result<ResourceId, RhiError> {
        let entry = self.pool.acquire(
            self.ctx,
            name,
            self.extent,
            format,
            ImageUse::ColorTarget,
            Samples::X1,
        )?;
        Ok(self.push(name, Target::Image(entry.image), entry.texture))
    }

    /// A pooled depth attachment at the frame's extent, written and never read
    /// as a texture — a prepass target.
    ///
    /// # Errors
    ///
    /// Allocation.
    pub fn depth(&mut self, name: &'static str) -> Result<ResourceId, RhiError> {
        self.depth_at(name, self.extent, Samples::X1)
    }

    /// A pooled depth attachment at **its own** extent — the camera's depth
    /// buffer when the scene is composited into part of the target rather than
    /// all of it, and so is smaller than the frame.
    ///
    /// [`Self::shadow`]'s [`ImageUse`] would do the wrong thing here: nothing
    /// samples a prepass target, and asking for `SAMPLED` costs some drivers
    /// the compressed depth layout it should keep.
    ///
    /// # Errors
    ///
    /// Allocation.
    pub fn depth_at(
        &mut self,
        name: &'static str,
        extent: (u32, u32),
        samples: Samples,
    ) -> Result<ResourceId, RhiError> {
        let entry = self.pool.acquire(
            self.ctx,
            name,
            extent,
            ImageFormat::Depth32,
            ImageUse::Depth,
            samples,
        )?;
        Ok(self.push(name, Target::Image(entry.image), None))
    }

    /// A pooled color attachment at **its own** extent, with a bindless slot —
    /// the luminance resample's small target (§6 M11).
    ///
    /// # Errors
    ///
    /// Allocation, or a full bindless array.
    /// A multisampled one has **no bindless slot**: nothing samples it, the
    /// resolve target beside it is what a later pass reads, and a `2DMS`
    /// descriptor is not something the global set declares. [`Frame::texture`]
    /// therefore returns `None` for it, which is the shape that makes "the post
    /// pass samples the resolve" the only spelling that compiles.
    pub fn color_at(
        &mut self,
        name: &'static str,
        extent: (u32, u32),
        format: ImageFormat,
        samples: Samples,
    ) -> Result<ResourceId, RhiError> {
        let entry = self.pool.acquire(
            self.ctx,
            name,
            extent,
            format,
            ImageUse::ColorTarget,
            samples,
        )?;
        Ok(self.push(name, Target::Image(entry.image), entry.texture))
    }

    /// A pooled depth attachment at **its own** extent, with a bindless slot —
    /// the shadow map (§6 M11).
    ///
    /// Its own extent because a shadow map's resolution is a quality knob and
    /// not the window's, and its own [`ImageUse`] because `SAMPLED` costs some
    /// drivers the compressed depth layout that the prepass target — which
    /// nothing samples — should keep.
    ///
    /// # Errors
    ///
    /// Allocation, or a full bindless array.
    pub fn shadow(
        &mut self,
        name: &'static str,
        extent: (u32, u32),
    ) -> Result<ResourceId, RhiError> {
        let entry = self.pool.acquire(
            self.ctx,
            name,
            extent,
            ImageFormat::Depth32,
            ImageUse::DepthSampled,
            // Never multisampled, whatever the camera's count: a shadow map is
            // sampled — many times, at an offset — so its samples would have to
            // be resolved before the lookup, and what the lookup wants is the
            // nearest depth rather than the average of eight.
            Samples::X1,
        )?;
        Ok(self.push(name, Target::Image(entry.image), entry.texture))
    }

    /// Import a host-readable buffer as a resource, so a readback pass can name
    /// it and the graph can derive the copy → host dependency behind it.
    pub fn readback_buffer(&mut self, name: &'static str, handle: BufferHandle) -> ResourceId {
        self.push(name, Target::Buffer(handle), None)
    }

    /// The bindless slot a pass samples `id` through — what goes in a post
    /// pass's push constants.
    pub fn texture(&self, id: ResourceId) -> Option<TextureIndex> {
        self.textures.get(id.0).copied().flatten()
    }

    fn push(
        &mut self,
        name: &'static str,
        target: Target,
        texture: Option<TextureIndex>,
    ) -> ResourceId {
        self.resources.push(Resource {
            name,
            target,
            state: Access::None,
        });
        self.textures.push(texture);
        ResourceId(self.resources.len() - 1)
    }

    /// Linearize the declarations and derive every transition between them.
    ///
    /// Execution order is declaration order: a v1 graph whose passes are
    /// written in the order they run needs no topological sort, and inventing
    /// one would let a reordering happen that no author asked for. What *is*
    /// derived is the synchronization — which is the half that is error-prone.
    ///
    /// The backbuffer is left in [`Access::Present`], whether or not a pass
    /// wrote it: a swapchain image handed back in any other layout is a
    /// validation error, and forgetting the last transition is exactly the bug
    /// the graph exists to prevent.
    ///
    /// # Errors
    ///
    /// A pass naming a resource this frame does not have, or a readback of one
    /// that is not an image.
    pub fn compile<'a>(mut self, declared: &'a [Declared<'a>]) -> Result<Compiled<'a>, RhiError> {
        let mut passes = Vec::with_capacity(declared.len() + 1);
        for pass in declared {
            let mut edges = Vec::new();
            let body = match &pass.body {
                Body::Draw {
                    color,
                    resolve,
                    depth,
                    viewport,
                    samples,
                    draws,
                } => {
                    let resolve = *resolve;
                    let color = color
                        .map(|(id, load)| {
                            let target = self.transition(&mut edges, id, Access::ColorWrite)?;
                            // The resolve target is written by the pass as
                            // surely as the multisample one is — later, and by
                            // fixed function rather than by a draw, but a pass
                            // that did not declare it would be a pass whose
                            // output nothing barriered.
                            let into = resolve
                                .map(|id| self.transition(&mut edges, id, Access::ColorWrite))
                                .transpose()?;
                            Ok::<_, RhiError>(ColorAttachment {
                                target,
                                clear: match load {
                                    Load::Clear(c) => Some(c),
                                    Load::Keep => None,
                                },
                                resolve: into,
                            })
                        })
                        .transpose()?;
                    let depth = depth
                        .map(|(id, use_)| {
                            let access = match use_ {
                                DepthUse::Test => Access::DepthRead,
                                _ => Access::DepthWrite,
                            };
                            let target = self.transition(&mut edges, id, access)?;
                            Ok::<_, RhiError>(DepthAttachment {
                                target,
                                // Reverse-Z clears to the far plane; a pass that
                                // tests against a stored buffer loads it.
                                clear: (use_ != DepthUse::Test).then_some(gg_rhi::DEPTH_CLEAR),
                                store: use_ == DepthUse::WriteStore,
                                read_only: use_ == DepthUse::Test,
                            })
                        })
                        .transpose()?;
                    for id in samples.iter() {
                        self.transition(&mut edges, *id, Access::SampledRead)?;
                    }
                    CompiledBody::Draw {
                        color,
                        depth,
                        viewport: *viewport,
                        draws,
                    }
                }
                Body::Readback { source, dest } => {
                    let target = self.transition(&mut edges, *source, Access::CopySrc)?;
                    let into = self.transition(&mut edges, *dest, Access::CopyDst)?;
                    let Target::Buffer(into) = into else {
                        return Err(RhiError::Loader(format!(
                            "pass `{}` reads back into something that is not a buffer",
                            pass.name
                        )));
                    };
                    CompiledBody::Readback {
                        source: target,
                        dest: into,
                    }
                }
            };
            passes.push(Compiled1 {
                name: pass.name,
                transitions: edges
                    .iter()
                    .map(|e| gg_rhi::Transition {
                        target: e.target,
                        from: e.from,
                        to: e.to,
                    })
                    .collect(),
                edges,
                body,
            });
        }

        // The frame's terminal states, and the ones no author should have to
        // remember: a swapchain image handed back in any layout but `Present`
        // is a validation error, and a readback buffer the host reads without a
        // dependency on the copy is a race the timeline wait only usually wins.
        //
        // `Present` only where something presents — an offscreen target put in
        // a present layout is an image no WSI owns in a layout only a WSI reads.
        let mut edges = Vec::new();
        let terminal: Vec<(usize, Access)> = self
            .resources
            .iter()
            .enumerate()
            .filter_map(|(i, r)| match (r.target, r.state) {
                (Target::Backbuffer, state) if self.presents && state != Access::Present => {
                    Some((i, Access::Present))
                }
                (Target::Buffer(_), Access::CopyDst) => Some((i, Access::HostRead)),
                _ => None,
            })
            .collect();
        for (index, access) in terminal {
            self.transition(&mut edges, ResourceId(index), access)?;
        }
        if !edges.is_empty() {
            passes.push(Compiled1 {
                name: "frame-end",
                transitions: edges
                    .iter()
                    .map(|e| gg_rhi::Transition {
                        target: e.target,
                        from: e.from,
                        to: e.to,
                    })
                    .collect(),
                edges,
                // A pass with no body: the transition *is* the work. Expressing
                // it as a pass rather than as a special case in the executor is
                // what keeps "every barrier belongs to a pass" true.
                body: CompiledBody::Barriers,
            });
        }

        Ok(Compiled {
            passes,
            resources: self.resources.iter().map(|r| (r.name, r.target)).collect(),
        })
    }

    /// Move `id` to `access`, recording the edge when it actually changes.
    fn transition(
        &mut self,
        edges: &mut Vec<Edge>,
        id: ResourceId,
        access: Access,
    ) -> Result<Target, RhiError> {
        let resource = self
            .resources
            .get_mut(id.0)
            .ok_or_else(|| RhiError::Loader(format!("resource {} is not this frame's", id.0)))?;
        if resource.state != access {
            edges.push(Edge {
                name: resource.name,
                target: resource.target,
                from: resource.state,
                to: access,
            });
            resource.state = access;
        }
        Ok(resource.target)
    }
}

/// One pooled attachment.
#[derive(Clone, Copy)]
struct Entry {
    image: ImageHandle,
    texture: Option<TextureIndex>,
}

struct Slot {
    name: String,
    slot: u64,
    extent: (u32, u32),
    format: ImageFormat,
    samples: Samples,
    entry: Entry,
    last_used: u64,
}

/// The transient attachment pool (§4.5): images recycled by name, size, format
/// **and frame slot**.
///
/// The slot is not decoration. With two frames in flight, one image shared by
/// both is frame N+1 clearing an attachment frame N is still reading, with no
/// barrier anywhere that orders them — a hazard no single-frame test can see.
/// Keying on the slot gives each in-flight frame its own, which is what the
/// frames-in-flight wait already paces.
#[derive(Default)]
pub struct Transients {
    slots: Vec<Slot>,
    frame: u64,
}

/// Frames a pooled image may go unused before it is retired. Two frames in
/// flight plus one, so a resize does not thrash the pool while the old size is
/// still on the GPU.
const IDLE_FRAMES: u64 = gg_rhi::FRAMES_IN_FLIGHT + 1;

impl Transients {
    /// Begin a frame's graph at `extent`, retiring anything the last few frames
    /// stopped asking for (a resize leaves the old size behind).
    ///
    /// # Errors
    ///
    /// Retiring a stale handle.
    pub fn frame<'p>(
        &'p mut self,
        ctx: &'p mut dyn GraphContext,
        extent: (u32, u32),
    ) -> Result<Frame<'p>, RhiError> {
        self.frame += 1;
        let frame = self.frame;
        let mut retired = Ok(());
        self.slots.retain(|slot| {
            if frame.saturating_sub(slot.last_used) <= IDLE_FRAMES {
                return true;
            }
            if retired.is_ok() {
                retired = ctx.destroy_image(slot.entry.image);
            }
            false
        });
        retired?;
        let presents = ctx.presents();
        Ok(Frame {
            pool: self,
            ctx,
            extent,
            resources: Vec::new(),
            textures: Vec::new(),
            presents,
        })
    }

    /// Every attachment the pool still holds, retired. Callers do this before
    /// shutting the context down; the leak report is the check behind it.
    ///
    /// # Errors
    ///
    /// A handle already retired.
    pub fn destroy(&mut self, ctx: &mut dyn GraphContext) -> Result<(), RhiError> {
        let mut result = Ok(());
        for slot in self.slots.drain(..) {
            let one = ctx.destroy_image(slot.entry.image);
            if result.is_ok() {
                result = one;
            }
        }
        result
    }

    fn acquire(
        &mut self,
        ctx: &mut dyn GraphContext,
        name: &str,
        extent: (u32, u32),
        format: ImageFormat,
        usage: ImageUse,
        samples: Samples,
    ) -> Result<Entry, RhiError> {
        let slot = ctx.frame_slot();
        let frame = self.frame;
        // The sample count is part of the key for the same reason the extent
        // is: a mode switch mid-session must not hand back the old count's
        // image, and the one it stops asking for retires by going idle.
        if let Some(existing) = self.slots.iter_mut().find(|s| {
            s.name == name
                && s.slot == slot
                && s.extent == extent
                && s.format == format
                && s.samples == samples
        }) {
            existing.last_used = frame;
            return Ok(existing.entry);
        }
        let image = ctx.create_image(&ImageDesc {
            name: &format!("gg.graph.{name}.{slot}"),
            extent,
            format,
            usage,
            // A transient is written and sampled at full size; a chain would be
            // levels no pass declares and no upload fills.
            mip_levels: 1,
            samples,
        })?;
        // Sampled attachments get their bindless slot once, here, rather than
        // per frame: a descriptor write every frame would be a cost the graph
        // charges for nothing, since the pool hands back the same image.
        //
        // A multisample attachment gets none at all — the global set's arrays
        // are single-sample, and what a later pass reads is the resolve.
        let texture = match usage {
            ImageUse::Depth => None,
            _ if samples.multisampled() => None,
            _ => Some(ctx.register_texture(image)?),
        };
        let entry = Entry { image, texture };
        self.slots.push(Slot {
            name: name.to_owned(),
            slot,
            extent,
            format,
            samples,
            entry,
            last_used: frame,
        });
        Ok(entry)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use gg_rhi::{ImageHandle, OffscreenRhi};

    /// A real context with one thing overridden: which frame-in-flight slot it
    /// claims to be building for.
    ///
    /// A windowed `Rhi` is the only thing that reports more than one, and it is
    /// windowed and therefore manual (§1.5) — so the property that matters here
    /// (two frames in flight never share an attachment) would otherwise be
    /// provable only by hand. Allocation stays real; the slot is the lie.
    struct Slotted {
        rhi: OffscreenRhi,
        live: Vec<ImageHandle>,
        slot: u64,
    }

    impl Slotted {
        fn new() -> Self {
            Slotted {
                rhi: OffscreenRhi::new((8, 8)).unwrap(),
                live: Vec::new(),
                slot: 0,
            }
        }
    }

    impl GraphContext for Slotted {
        fn create_image(&mut self, desc: &ImageDesc<'_>) -> Result<ImageHandle, RhiError> {
            let handle = self.rhi.create_image(desc)?;
            self.live.push(handle);
            Ok(handle)
        }
        fn destroy_image(&mut self, handle: ImageHandle) -> Result<(), RhiError> {
            let at = self
                .live
                .iter()
                .position(|h| *h == handle)
                .ok_or_else(|| RhiError::Loader("retired twice".into()))?;
            self.live.remove(at);
            self.rhi.destroy_image(handle)
        }
        fn register_texture(&mut self, handle: ImageHandle) -> Result<TextureIndex, RhiError> {
            self.rhi.register_texture(handle)
        }
        fn frame_slot(&self) -> u64 {
            self.slot
        }
        fn presents(&self) -> bool {
            true
        }
    }

    fn depth_of(pool: &mut Transients, ctx: &mut Slotted, extent: (u32, u32)) -> ImageHandle {
        let mut frame = pool.frame(ctx, extent).unwrap();
        let id = frame.depth("scene.depth").unwrap();
        let declared = [Declared {
            name: "pass",
            body: Body::Draw {
                color: None,
                resolve: None,
                depth: Some((id, DepthUse::Write)),
                viewport: None,
                samples: &[],
                draws: &[],
            },
        }];
        let compiled = frame.compile(&declared).unwrap();
        match compiled.passes()[0].kind {
            PassKind::Render {
                depth: Some(depth), ..
            } => match depth.target {
                Target::Image(handle) => handle,
                other => panic!("depth is an image, not {other:?}"),
            },
            _ => panic!("the pass declares a depth attachment"),
        }
    }

    /// Two frames in flight get two attachments, and the third frame is back on
    /// the first one. Sharing one across the pair would be frame N+1 clearing
    /// what frame N is still reading, with no barrier between them — the
    /// frames-in-flight wait paces the slots and the pool keys on them.
    #[test]
    fn each_frame_in_flight_gets_its_own_attachment() {
        let mut pool = Transients::default();
        let mut ctx = Slotted::new();

        ctx.slot = 0;
        let first = depth_of(&mut pool, &mut ctx, (64, 64));
        ctx.slot = 1;
        let second = depth_of(&mut pool, &mut ctx, (64, 64));
        assert_ne!(first, second, "one image cannot serve two frames in flight");

        ctx.slot = 0;
        assert_eq!(
            depth_of(&mut pool, &mut ctx, (64, 64)),
            first,
            "and the slot comes back around to the same one"
        );
        assert_eq!(ctx.live.len(), 2, "two slots, two images, nothing else");

        pool.destroy(&mut ctx).unwrap();
        assert!(ctx.live.is_empty(), "the pool hands everything back");
        assert!(
            ctx.rhi.shutdown().clean(),
            "no leaks, no validation messages"
        );
    }

    /// A resize is a new key, and the old size is retired once the frames that
    /// could still be holding it are past.
    #[test]
    fn a_resize_retires_the_old_size() {
        let mut pool = Transients::default();
        let mut ctx = Slotted::new();
        let small = depth_of(&mut pool, &mut ctx, (64, 64));
        let large = depth_of(&mut pool, &mut ctx, (128, 128));
        assert_ne!(small, large);
        for _ in 0..IDLE_FRAMES + 2 {
            depth_of(&mut pool, &mut ctx, (128, 128));
        }
        assert_eq!(
            ctx.live,
            [large],
            "the old extent is gone, the new one stays"
        );
        pool.destroy(&mut ctx).unwrap();
        assert!(
            ctx.rhi.shutdown().clean(),
            "no leaks, no validation messages"
        );
    }

    /// The frame's terminal states are derived, not declared: a graph that never
    /// mentions the backbuffer still hands it over in a present layout.
    #[test]
    fn the_backbuffer_always_ends_in_present() {
        let mut pool = Transients::default();
        let mut ctx = Slotted::new();
        let mut frame = pool.frame(&mut ctx, (8, 8)).unwrap();
        let backbuffer = frame.backbuffer();
        let declared = single_pass("clear", backbuffer, Load::Clear([0.0; 4]), &[]);
        let compiled = frame.compile(&declared).unwrap();
        let order: Vec<&str> = compiled.order().collect();
        assert_eq!(order, ["clear", "frame-end"]);
        let last = compiled.passes();
        let last = last.last().unwrap();
        assert_eq!(last.transitions.len(), 1);
        assert_eq!(last.transitions[0].to, Access::Present);
        pool.destroy(&mut ctx).unwrap();
        assert!(
            ctx.rhi.shutdown().clean(),
            "no leaks, no validation messages"
        );
    }
}
