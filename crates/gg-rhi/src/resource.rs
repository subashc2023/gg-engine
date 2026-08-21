//! GPU resources (§4.3): buffers, images, and the fixed sampler set — newtypes
//! owning an allocation plus its metadata, reached through opaque handles and
//! destroyed behind the timeline like everything else.
//!
//! The shape follows §4.3's bindless rule rather than Vulkan's usage matrix:
//! **every** buffer is shader-readable through its device address, because no
//! per-resource descriptor set exists to bind one to, so `BufferKind` records
//! only what the fixed-function index stream also needs. Images are sampled or
//! written through the one global set (`bindless.rs`), which is why nothing
//! here hands out a descriptor.

use crate::RhiError;
use crate::deletion::{Deferred, DeletionQueue};
use crate::device::Device;
use ash::vk;
use std::collections::BTreeMap;

/// A GPU-visible pointer to a buffer's first byte. Buffers reach shaders this
/// way and no other (§4.3), so this is the only "binding" a buffer has.
pub type DeviceAddress = u64;

/// What a buffer is for. Deliberately not a usage mask: with all access by
/// device address, the one distinction that survives is whether the
/// fixed-function index stream reads it too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferKind {
    /// Shader-readable through its [`DeviceAddress`]. Vertices, materials,
    /// instance data, scene buffers — all of it.
    Storage,
    /// As [`BufferKind::Storage`], and bindable as an indexed draw's index
    /// stream (`u32` indices).
    Index,
    /// Host-visible, copied *into* by a readback pass and read with
    /// [`Rhi::map_buffer`](crate::Rhi::map_buffer). The third variant is a
    /// memory *location* rather than a usage — the one distinction a device
    /// address genuinely cannot express (§4.5's readback pass).
    Readback,
    /// Host-visible the other way: written by the CPU with
    /// [`Rhi::write_buffer`](crate::Rhi::write_buffer) and read by shaders
    /// through its address, with no staging copy and no submit.
    ///
    /// The staging ring (§4.3) is for data that outlives the frame that uploads
    /// it; a stream rebuilt from scratch every frame would pay a transfer
    /// submit and a wait to hand over what one pass reads once. Nothing here
    /// waits, so **the caller owns the frames-in-flight hazard** — one region
    /// per [`FRAMES_IN_FLIGHT`](crate::FRAMES_IN_FLIGHT) slot, or a frame
    /// overwrites what its predecessor is still reading.
    Dynamic,
    /// As [`BufferKind::Dynamic`], and readable by an indirect draw as its
    /// parameters (§6 M10).
    ///
    /// The enum's own doc says it records "only what the fixed-function index
    /// stream also needs" — this is the second such consumer, and the last one
    /// core Vulkan has. It is `Dynamic` rather than `Storage` because a
    /// CPU-built draw list is rebuilt every frame, so **the caller owns the
    /// frames-in-flight hazard exactly as it does there**: one region per
    /// [`FRAMES_IN_FLIGHT`](crate::FRAMES_IN_FLIGHT) slot, or a frame overwrites
    /// the commands its predecessor is still executing.
    Indirect,
}

/// Pixel format. Short on purpose (§3 budget): one entry per job that exists,
/// new entries arrive with a consumer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageFormat {
    /// 8-bit RGBA, sRGB-encoded — color textures.
    Rgba8Srgb,
    /// 8-bit RGBA, linear — masks and data textures.
    Rgba8Unorm,
    /// BC7 block-compressed, sRGB-encoded (§2, Texture pipeline row).
    Bc7Srgb,
    /// BC5, two linear channels — normal maps, and glTF's metallic-roughness
    /// pair after `ggc` repacks it (§4.6).
    Bc5Unorm,
    /// BC4, one linear channel — masks. Half the block size of the other two.
    Bc4Unorm,
    /// BC6H, three channels of unsigned half float — the environment (§6 M27).
    /// The one block format here that does not clamp at 1.0, which is the whole
    /// reason it exists: a sky is a radiance and a sun is four digits of it.
    Bc6hUfloat,
    /// 16-bit float RGBA — the scene attachment since M11. Radiance leaves the
    /// forward pass unbounded above 1.0 and the tonemapper is what brings it
    /// down, so an 8-bit target would clip every highlight before the curve that
    /// exists to shape them ever ran.
    Rgba16F,
    /// One 8-bit linear channel — the ambient-occlusion target (§6 M35).
    ///
    /// A whole format for one channel because the alternative is four times the
    /// bandwidth on a full-screen target the forward pass reads once per
    /// fragment. Eight bits is enough by what it feeds: occlusion multiplies an
    /// ambient term, so a code value is a 0.4 % change in a fraction of the
    /// picture's dimmest light.
    R8Unorm,
    /// 32-bit float depth. The only depth format we ask for: reverse-Z's
    /// precision argument *is* a float-depth argument (§2, Math row).
    Depth32,
}

impl ImageFormat {
    pub(crate) fn vk(self) -> vk::Format {
        match self {
            ImageFormat::Rgba8Srgb => vk::Format::R8G8B8A8_SRGB,
            ImageFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
            ImageFormat::Bc7Srgb => vk::Format::BC7_SRGB_BLOCK,
            ImageFormat::Bc5Unorm => vk::Format::BC5_UNORM_BLOCK,
            ImageFormat::Bc4Unorm => vk::Format::BC4_UNORM_BLOCK,
            ImageFormat::Bc6hUfloat => vk::Format::BC6H_UFLOAT_BLOCK,
            ImageFormat::R8Unorm => vk::Format::R8_UNORM,
            ImageFormat::Rgba16F => vk::Format::R16G16B16A16_SFLOAT,
            ImageFormat::Depth32 => vk::Format::D32_SFLOAT,
        }
    }

    pub(crate) fn aspect(self) -> vk::ImageAspectFlags {
        match self {
            ImageFormat::Depth32 => vk::ImageAspectFlags::DEPTH,
            _ => vk::ImageAspectFlags::COLOR,
        }
    }

    /// Whether this format is a depth format — the one query callers outside
    /// the crate legitimately make (a depth image is attached, not sampled).
    pub fn is_depth(self) -> bool {
        matches!(self, ImageFormat::Depth32)
    }

    /// The texel block's edge: 4 for the BC formats, 1 for the rest. A copy's
    /// height must be a multiple of this unless it reaches the mip's edge,
    /// which is what makes a banded upload legal (§4.3).
    pub fn block_extent(self) -> u32 {
        match self {
            ImageFormat::Bc7Srgb
            | ImageFormat::Bc5Unorm
            | ImageFormat::Bc4Unorm
            | ImageFormat::Bc6hUfloat => 4,
            ImageFormat::Rgba8Srgb
            | ImageFormat::Rgba8Unorm
            | ImageFormat::R8Unorm
            | ImageFormat::Rgba16F
            | ImageFormat::Depth32 => 1,
        }
    }

    /// Bytes a tightly packed `extent` image occupies. Block formats round the
    /// extent up to whole 4x4 blocks — a 5x5 BC7 image is 2x2 blocks, and
    /// getting this wrong is a buffer overrun the copy would not catch.
    pub fn packed_size(self, extent: (u32, u32)) -> u64 {
        let (w, h) = (u64::from(extent.0), u64::from(extent.1));
        match self {
            ImageFormat::Bc7Srgb | ImageFormat::Bc5Unorm | ImageFormat::Bc6hUfloat => {
                w.div_ceil(4) * h.div_ceil(4) * 16
            }
            ImageFormat::Bc4Unorm => w.div_ceil(4) * h.div_ceil(4) * 8,
            ImageFormat::R8Unorm => w * h,
            ImageFormat::Rgba8Srgb | ImageFormat::Rgba8Unorm | ImageFormat::Depth32 => w * h * 4,
            ImageFormat::Rgba16F => w * h * 8,
        }
    }
}

/// `extent` at mip `level`, halved per level and clamped at 1 — the Vulkan
/// rule, and the one `ggc` encodes a chain against.
///
/// Mirrored in `gg_assets::texture` rather than shared: this crate is below the
/// asset format and must be able to *check* a caller's arithmetic without
/// linking the crate that produced it.
#[must_use]
pub fn mip_extent(extent: (u32, u32), level: u32) -> (u32, u32) {
    ((extent.0 >> level).max(1), (extent.1 >> level).max(1))
}

/// How many mips `extent` reduces to before both axes reach 1.
#[must_use]
pub fn full_mip_count(extent: (u32, u32)) -> u32 {
    32 - extent.0.max(extent.1).max(1).leading_zeros()
}

/// What an image is for. Drives usage flags and the layout it settles in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageUse {
    /// Uploaded once, then read through the bindless sampled-image array.
    Sampled,
    /// A depth attachment. Never uploaded, never sampled.
    Depth,
    /// A depth attachment a later pass also samples — the shadow map (§6 M11).
    /// Separate from [`ImageUse::Depth`] because `SAMPLED` costs some drivers
    /// the compressed depth layout, and the prepass target that is never read
    /// back should not pay for the one that is.
    DepthSampled,
    /// Written through the bindless storage-image array.
    Storage,
    /// A graph attachment: written as color, then either sampled by a later
    /// pass or copied out by a readback one. All three usages at once, because
    /// which of them a transient is put to is the graph's business and not the
    /// allocation's (§4.5).
    ColorTarget,
}

impl ImageUse {
    /// Whether this usage is a depth attachment — the question
    /// [`ImageFormat::is_depth`] has to agree with.
    #[must_use]
    pub fn is_depth(self) -> bool {
        matches!(self, ImageUse::Depth | ImageUse::DepthSampled)
    }

    /// Whether an image for this job may be multisampled at all. Only the two
    /// attachment kinds can: a sampled or storage image is reached through the
    /// bindless array, whose descriptors are single-sample by declaration, and
    /// a multisample image is illegal as a copy source besides.
    #[must_use]
    pub fn is_attachment(self) -> bool {
        matches!(self, ImageUse::Depth | ImageUse::ColorTarget)
    }
}

/// How many samples one rasterization covers — MSAA's only knob (§6 M21).
///
/// Ordered, so clamping to what a device advertises is `min`. The set stops at
/// eight because that is where every desktop driver's advertised mask stops
/// being universal, and a ninth entry would be a mode with no hardware.
///
/// A count is **asked for and refused**, never quietly downgraded: an operator
/// who set 8× and silently got 4× would be judging 8× by the wrong picture.
/// [`DeviceReport::max_samples`](crate::DeviceReport::max_samples) is where the
/// asking happens.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Samples {
    /// No multisampling — one sample per pixel, and the only count that needs
    /// no resolve.
    #[default]
    X1,
    /// Two samples.
    X2,
    /// Four samples. The one count above 1 that every driver in the execution
    /// matrix advertises, including pinned lavapipe — so it is the only one a
    /// headless gate can prove (§8 residual).
    X4,
    /// Eight samples.
    X8,
}

impl Samples {
    /// Every count, ascending.
    pub const ALL: [Samples; 4] = [Samples::X1, Samples::X2, Samples::X4, Samples::X8];

    /// Samples per pixel.
    #[must_use]
    pub fn count(self) -> u32 {
        match self {
            Samples::X1 => 1,
            Samples::X2 => 2,
            Samples::X4 => 4,
            Samples::X8 => 8,
        }
    }

    /// The count `n` names, or `None` when it is not one we have — including
    /// 16 and 32, which exist in Vulkan and not here.
    #[must_use]
    pub fn from_count(n: u32) -> Option<Samples> {
        Samples::ALL.into_iter().find(|s| s.count() == n)
    }

    /// Whether this count needs a resolve attachment beside it.
    #[must_use]
    pub fn multisampled(self) -> bool {
        self != Samples::X1
    }

    pub(crate) fn vk(self) -> vk::SampleCountFlags {
        match self {
            Samples::X1 => vk::SampleCountFlags::TYPE_1,
            Samples::X2 => vk::SampleCountFlags::TYPE_2,
            Samples::X4 => vk::SampleCountFlags::TYPE_4,
            Samples::X8 => vk::SampleCountFlags::TYPE_8,
        }
    }
}

/// The whole sampler set (§4.3: samplers are a small *immutable* set baked
/// into the bindless layout — a material names one by index and no descriptor
/// is ever allocated for it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampler {
    /// Linear filtering, repeating — surface textures.
    LinearRepeat,
    /// Linear filtering, clamped — full-screen and atlas sampling.
    LinearClamp,
    /// Nearest filtering, clamped — data lookups that must not blend
    /// neighbouring texels.
    NearestClamp,
}

impl Sampler {
    /// How many samplers the immutable set holds.
    pub const COUNT: u32 = 3;

    /// This sampler's index in the bindless sampler array — what a shader
    /// indexes with, and what a material stores.
    pub fn index(self) -> u32 {
        match self {
            Sampler::LinearRepeat => 0,
            Sampler::LinearClamp => 1,
            Sampler::NearestClamp => 2,
        }
    }

    pub(crate) fn all() -> [Sampler; Self::COUNT as usize] {
        [
            Sampler::LinearRepeat,
            Sampler::LinearClamp,
            Sampler::NearestClamp,
        ]
    }

    pub(crate) fn create(self, device: &Device) -> Result<vk::Sampler, RhiError> {
        let (filter, mode) = match self {
            Sampler::LinearRepeat => (vk::Filter::LINEAR, vk::SamplerAddressMode::REPEAT),
            Sampler::LinearClamp => (vk::Filter::LINEAR, vk::SamplerAddressMode::CLAMP_TO_EDGE),
            Sampler::NearestClamp => (vk::Filter::NEAREST, vk::SamplerAddressMode::CLAMP_TO_EDGE),
        };
        let mipmap = match filter {
            vk::Filter::NEAREST => vk::SamplerMipmapMode::NEAREST,
            _ => vk::SamplerMipmapMode::LINEAR,
        };
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(filter)
            .min_filter(filter)
            .mipmap_mode(mipmap)
            .address_mode_u(mode)
            .address_mode_v(mode)
            .address_mode_w(mode)
            .max_lod(vk::LOD_CLAMP_NONE);
        // SAFETY: device is live; info is fully initialized above.
        unsafe { device.raw().create_sampler(&info, None) }.map_err(RhiError::vk)
    }
}

/// How to create a buffer. `name` is not optional (§4.3): an unnamed handle is
/// an unreadable validation message and an unattributable leak report.
pub struct BufferDesc<'a> {
    /// Debug name (§1.6). Compiles out in dist with the rest of the naming.
    pub name: &'a str,
    /// Size in bytes. Zero is refused — a zero-size buffer has no address.
    pub size: u64,
    /// What the buffer is for.
    pub kind: BufferKind,
}

/// How to create an image.
pub struct ImageDesc<'a> {
    /// Debug name (§1.6).
    pub name: &'a str,
    /// Width and height in pixels (or in texels for block formats).
    pub extent: (u32, u32),
    /// Pixel format.
    pub format: ImageFormat,
    /// What the image is for.
    pub usage: ImageUse,
    /// Mip levels, 1 for no chain. Never *generated* here: a chain is built
    /// offline by `ggc` and uploaded level by level (§4.6), so this allocates
    /// the levels and the caller fills every one of them.
    pub mip_levels: u32,
    /// Samples per pixel. Anything above [`Samples::X1`] makes this an
    /// attachment and nothing else — not sampled, not copied, not mipped — and
    /// the pass that writes it must resolve into a single-sample image.
    pub samples: Samples,
}

/// An opaque handle to a buffer. Plain data, like [`PipelineHandle`]: a handle
/// outliving its buffer fails a lookup with a precise error, never dangles.
///
/// [`PipelineHandle`]: crate::PipelineHandle
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BufferHandle(u64);

/// What the device is holding, live (§4.8). Counts beside bytes because the
/// number that says a frame is leaking is the count, not the total.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryUse {
    /// Live buffers.
    pub buffers: u32,
    /// Bytes allocated for them.
    pub buffer_bytes: u64,
    /// Live images, including the graph's pooled attachments.
    pub images: u32,
    /// Bytes allocated for them.
    pub image_bytes: u64,
}

impl MemoryUse {
    /// Buffers and images together.
    pub fn total_bytes(&self) -> u64 {
        self.buffer_bytes + self.image_bytes
    }
}

/// An opaque handle to an image.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ImageHandle(u64);

/// A buffer: its Vulkan handle, its allocation, its size, and the device
/// address shaders reach it by.
pub(crate) struct Buffer {
    pub raw: vk::Buffer,
    pub alloc: Option<gpu_allocator::vulkan::Allocation>,
    pub size: u64,
    pub address: DeviceAddress,
}

/// An image and its view. No layout field: every transition a frame makes is
/// derived from the graph's declarations (§4.5) and an upload's pair is
/// recorded per mip level, so a single current-layout here could only ever be
/// a copy of one of those — and one that a half-uploaded chain makes wrong.
pub(crate) struct Image {
    pub raw: vk::Image,
    pub view: vk::ImageView,
    pub alloc: Option<gpu_allocator::vulkan::Allocation>,
    pub extent: (u32, u32),
    pub format: ImageFormat,
    /// What it was created for — the only thing that says whether a depth image
    /// is allowed a bindless slot.
    pub usage: ImageUse,
    pub mip_levels: u32,
    /// Bindless slots this image occupies, so retiring it returns them rather
    /// than leaking the array one entry at a time.
    pub texture_slot: Option<crate::TextureIndex>,
    pub storage_slot: Option<crate::StorageImageIndex>,
}

/// The bindless slots a retired image was holding — the caller returns them to
/// the arrays once the GPU is past the frames that indexed them.
pub(crate) struct RetiredSlots {
    pub texture: Option<crate::TextureIndex>,
    pub storage: Option<crate::StorageImageIndex>,
}

/// Every buffer and image the engine owns, plus the immutable samplers.
pub(crate) struct Resources {
    next_id: u64,
    buffers: BTreeMap<u64, Buffer>,
    images: BTreeMap<u64, Image>,
    samplers: Vec<vk::Sampler>,
}

impl Resources {
    pub fn new(device: &Device) -> Result<Self, RhiError> {
        crate::inject::point("Resources::new")?;
        let mut samplers = Vec::with_capacity(Sampler::COUNT as usize);
        for s in Sampler::all() {
            match s.create(device) {
                Ok(raw) => {
                    device.set_name(raw, &format!("gg.sampler.{s:?}"));
                    samplers.push(raw);
                }
                Err(e) => {
                    // SAFETY: every handle in `samplers` was created just above
                    // and is unused; the device outlives this loop.
                    unsafe {
                        for raw in samplers.drain(..) {
                            device.raw().destroy_sampler(raw, None);
                        }
                    }
                    return Err(e);
                }
            }
        }
        Ok(Self {
            next_id: 1,
            buffers: BTreeMap::new(),
            images: BTreeMap::new(),
            samplers,
        })
    }

    /// The immutable sampler handles, in [`Sampler::index`] order — what the
    /// bindless layout bakes in.
    pub fn samplers(&self) -> &[vk::Sampler] {
        &self.samplers
    }

    pub fn create_buffer(
        &mut self,
        device: &mut Device,
        desc: &BufferDesc<'_>,
    ) -> Result<BufferHandle, RhiError> {
        if desc.size == 0 {
            return Err(RhiError::Loader(format!(
                "buffer `{}`: size 0 has no device address",
                desc.name
            )));
        }
        let usage = buffer_usage(desc.kind);
        let location = match desc.kind {
            BufferKind::Readback => gpu_allocator::MemoryLocation::GpuToCpu,
            // CpuToGpu, so a discrete part places it in write-combined BAR
            // memory the GPU can read directly: the CPU writes it sequentially
            // and never reads it back, which is exactly what that memory is bad
            // and good at respectively.
            BufferKind::Dynamic | BufferKind::Indirect => gpu_allocator::MemoryLocation::CpuToGpu,
            _ => gpu_allocator::MemoryLocation::GpuOnly,
        };
        let buffer = create_raw_buffer_in(device, desc.name, desc.size, usage, location)?;
        let address = {
            let info = vk::BufferDeviceAddressInfo::default().buffer(buffer.raw);
            // SAFETY: the buffer was created with SHADER_DEVICE_ADDRESS and is
            // bound to memory; bufferDeviceAddress is an asserted feature.
            unsafe { device.raw().get_buffer_device_address(&info) }
        };
        let id = self.next_id;
        self.next_id += 1;
        self.buffers.insert(id, Buffer { address, ..buffer });
        Ok(BufferHandle(id))
    }

    pub fn create_image(
        &mut self,
        device: &mut Device,
        desc: &ImageDesc<'_>,
    ) -> Result<ImageHandle, RhiError> {
        if desc.extent.0 == 0 || desc.extent.1 == 0 {
            return Err(RhiError::Loader(format!(
                "image `{}`: extent {:?} must be nonzero",
                desc.name, desc.extent
            )));
        }
        if desc.format.is_depth() != desc.usage.is_depth() {
            return Err(RhiError::Loader(format!(
                "image `{}`: {:?} and {:?} disagree — a depth format is a depth attachment and \
                 nothing else",
                desc.name, desc.format, desc.usage
            )));
        }
        // A chain longer than the extent supports would allocate levels no
        // upload can name, and every one of them would sample as undefined.
        let full = full_mip_count(desc.extent);
        if desc.mip_levels == 0 || desc.mip_levels > full {
            return Err(RhiError::Loader(format!(
                "image `{}`: {} mip levels, and {:?} has {full}",
                desc.name, desc.mip_levels, desc.extent
            )));
        }
        // The three ways a multisample image is not an ordinary one, refused
        // here rather than left to validation: only an attachment can be one,
        // a chain of them is levels no pass can name, and a count the device
        // does not advertise is a mode the operator must be told they cannot
        // have (§6 M21) — never one quietly rounded down.
        if desc.samples.multisampled() {
            if !desc.usage.is_attachment() {
                return Err(RhiError::Loader(format!(
                    "image `{}`: {:?} at {}x — only an attachment can be multisampled, since \
                     everything else is reached through the bindless array or a copy",
                    desc.name,
                    desc.usage,
                    desc.samples.count()
                )));
            }
            if desc.mip_levels != 1 {
                return Err(RhiError::Loader(format!(
                    "image `{}`: {} mip levels at {}x — a multisample image has one level",
                    desc.name,
                    desc.mip_levels,
                    desc.samples.count()
                )));
            }
            if !device.supports_samples(desc.samples) {
                return Err(RhiError::Loader(format!(
                    "image `{}`: this device does not do {}x — it advertises up to {}x",
                    desc.name,
                    desc.samples.count(),
                    device.report().max_samples().count()
                )));
            }
        }
        let usage = match desc.usage {
            ImageUse::Sampled => vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            ImageUse::Depth => vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ImageUse::DepthSampled => {
                vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED
            }
            ImageUse::Storage => vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
            // A multisample target keeps the attachment usage and drops the
            // other two: sampling one needs a `2DMS` descriptor the global set
            // does not declare, and it is not a legal copy source at all. What
            // reads it is the resolve, and a resolve is not either of those.
            ImageUse::ColorTarget if desc.samples.multisampled() => {
                vk::ImageUsageFlags::COLOR_ATTACHMENT
            }
            ImageUse::ColorTarget => {
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    // And written *into* by an upload, which is the fourth thing
                    // a graph resource may be put to (§6 M36): an image that
                    // outlives the frame has to start somewhere, and the ordinary
                    // upload path is what puts it in the layout its bindless
                    // descriptor declares before any frame runs.
                    | vk::ImageUsageFlags::TRANSFER_DST
            }
        };
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(desc.format.vk())
            .extent(vk::Extent3D {
                width: desc.extent.0,
                height: desc.extent.1,
                depth: 1,
            })
            .mip_levels(desc.mip_levels)
            .array_layers(1)
            .samples(desc.samples.vk())
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: device is live; info is fully initialized above.
        let raw = unsafe { device.raw().create_image(&info, None) }.map_err(RhiError::vk)?;
        device.set_name(raw, desc.name);

        // SAFETY: image is live.
        let requirements = unsafe { device.raw().get_image_memory_requirements(raw) };
        let alloc = match device.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name: desc.name,
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(a) => a,
            Err(e) => {
                // SAFETY: the image was created above and owns no memory yet.
                unsafe { device.raw().destroy_image(raw, None) };
                return Err(e);
            }
        };
        // SAFETY: fresh image, fresh memory, offset from the allocator.
        if let Err(e) = unsafe {
            device
                .raw()
                .bind_image_memory(raw, alloc.memory(), alloc.offset())
        } {
            let _ = device.free(alloc);
            // SAFETY: as above — the image is unbound and unused.
            unsafe { device.raw().destroy_image(raw, None) };
            return Err(RhiError::vk(e));
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(raw)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(desc.format.vk())
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(desc.format.aspect())
                    .level_count(desc.mip_levels)
                    .layer_count(1),
            );
        // SAFETY: image is live and bound.
        let view = match unsafe { device.raw().create_image_view(&view_info, None) } {
            Ok(v) => v,
            Err(e) => {
                let _ = device.free(alloc);
                // SAFETY: as above.
                unsafe { device.raw().destroy_image(raw, None) };
                return Err(RhiError::vk(e));
            }
        };
        device.set_name(view, &format!("{}.view", desc.name));

        let id = self.next_id;
        self.next_id += 1;
        self.images.insert(
            id,
            Image {
                raw,
                view,
                alloc: Some(alloc),
                extent: desc.extent,
                format: desc.format,
                usage: desc.usage,
                mip_levels: desc.mip_levels,
                texture_slot: None,
                storage_slot: None,
            },
        );
        Ok(ImageHandle(id))
    }

    pub fn buffer(&self, handle: BufferHandle) -> Result<&Buffer, RhiError> {
        self.buffers
            .get(&handle.0)
            .ok_or_else(|| RhiError::Loader(format!("buffer handle {} is not live", handle.0)))
    }

    /// A readback buffer's host-visible bytes.
    pub fn mapped(&self, handle: BufferHandle) -> Result<&[u8], RhiError> {
        let buffer = self.buffer(handle)?;
        buffer
            .alloc
            .as_ref()
            .and_then(gpu_allocator::vulkan::Allocation::mapped_slice)
            .map(|bytes| &bytes[..buffer.size as usize])
            .ok_or_else(|| host_visible_error(handle))
    }

    /// A [`BufferKind::Dynamic`] buffer's mapping, to write into.
    pub fn mapped_mut(&mut self, handle: BufferHandle) -> Result<&mut [u8], RhiError> {
        let buffer = self
            .buffers
            .get_mut(&handle.0)
            .ok_or_else(|| RhiError::Loader(format!("buffer handle {} is not live", handle.0)))?;
        let size = buffer.size as usize;
        buffer
            .alloc
            .as_mut()
            .and_then(gpu_allocator::vulkan::Allocation::mapped_slice_mut)
            .map(|bytes| &mut bytes[..size])
            .ok_or_else(|| host_visible_error(handle))
    }

    /// What the device is holding right now, for the overlay's memory row
    /// (§4.8). Allocated bytes rather than requested: alignment and the
    /// allocator's block size are real, and a row that hid them would report a
    /// number no tool agrees with.
    pub fn memory(&self) -> MemoryUse {
        let bytes = |alloc: &Option<gpu_allocator::vulkan::Allocation>| {
            alloc
                .as_ref()
                .map_or(0, gpu_allocator::vulkan::Allocation::size)
        };
        MemoryUse {
            buffers: self.buffers.len() as u32,
            buffer_bytes: self.buffers.values().map(|b| bytes(&b.alloc)).sum(),
            images: self.images.len() as u32,
            image_bytes: self.images.values().map(|i| bytes(&i.alloc)).sum(),
        }
    }

    pub fn image(&self, handle: ImageHandle) -> Result<&Image, RhiError> {
        self.images
            .get(&handle.0)
            .ok_or_else(|| RhiError::Loader(format!("image handle {} is not live", handle.0)))
    }

    pub fn image_mut(&mut self, handle: ImageHandle) -> Result<&mut Image, RhiError> {
        self.images
            .get_mut(&handle.0)
            .ok_or_else(|| RhiError::Loader(format!("image handle {} is not live", handle.0)))
    }

    /// Retire a buffer behind the timeline: destroyed once the GPU is provably
    /// past `after_value`.
    pub fn retire_buffer(
        &mut self,
        handle: BufferHandle,
        deletions: &mut DeletionQueue,
        after_value: u64,
    ) -> Result<(), RhiError> {
        let mut buffer = self
            .buffers
            .remove(&handle.0)
            .ok_or_else(|| RhiError::Loader(format!("buffer handle {} retired twice", handle.0)))?;
        deletions.defer(after_value, Deferred::Buffer(buffer.raw));
        if let Some(alloc) = buffer.alloc.take() {
            deletions.defer(after_value, Deferred::Allocation(Box::new(alloc)));
        }
        Ok(())
    }

    /// Retire an image and its view behind the timeline, handing back the
    /// bindless slots it held.
    pub fn retire_image(
        &mut self,
        handle: ImageHandle,
        deletions: &mut DeletionQueue,
        after_value: u64,
    ) -> Result<RetiredSlots, RhiError> {
        let mut image = self
            .images
            .remove(&handle.0)
            .ok_or_else(|| RhiError::Loader(format!("image handle {} retired twice", handle.0)))?;
        deletions.defer(after_value, Deferred::ImageView(image.view));
        deletions.defer(after_value, Deferred::Image(image.raw));
        if let Some(alloc) = image.alloc.take() {
            deletions.defer(after_value, Deferred::Allocation(Box::new(alloc)));
        }
        Ok(RetiredSlots {
            texture: image.texture_slot,
            storage: image.storage_slot,
        })
    }

    /// Teardown: destroy everything still held. Caller waited the device idle.
    pub fn destroy(&mut self, device: &mut Device) {
        for (_, mut buffer) in std::mem::take(&mut self.buffers) {
            // SAFETY: handle belongs to this device; GPU idle per contract.
            unsafe { device.raw().destroy_buffer(buffer.raw, None) };
            if let Some(alloc) = buffer.alloc.take() {
                let _ = device.free(alloc);
            }
        }
        for (_, mut image) in std::mem::take(&mut self.images) {
            // SAFETY: as above.
            unsafe {
                device.raw().destroy_image_view(image.view, None);
                device.raw().destroy_image(image.raw, None);
            }
            if let Some(alloc) = image.alloc.take() {
                let _ = device.free(alloc);
            }
        }
        for raw in self.samplers.drain(..) {
            // SAFETY: as above.
            unsafe { device.raw().destroy_sampler(raw, None) };
        }
    }
}

fn host_visible_error(handle: BufferHandle) -> RhiError {
    RhiError::Loader(format!(
        "buffer handle {} is not host-visible — only BufferKind::Readback, ::Dynamic and \
         ::Indirect are",
        handle.0
    ))
}

fn buffer_usage(kind: BufferKind) -> vk::BufferUsageFlags {
    // SHADER_DEVICE_ADDRESS on every buffer is the §4.3 bindless rule, not a
    // convenience: shaders have no other way to reach one.
    let base = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        | vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST;
    match kind {
        BufferKind::Storage | BufferKind::Readback | BufferKind::Dynamic => base,
        BufferKind::Index => base | vk::BufferUsageFlags::INDEX_BUFFER,
        BufferKind::Indirect => base | vk::BufferUsageFlags::INDIRECT_BUFFER,
    }
}

pub(crate) fn create_raw_buffer_in(
    device: &mut Device,
    name: &str,
    size: u64,
    usage: vk::BufferUsageFlags,
    location: gpu_allocator::MemoryLocation,
) -> Result<Buffer, RhiError> {
    let info = vk::BufferCreateInfo::default()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: device is live; info is fully initialized above.
    let raw = unsafe { device.raw().create_buffer(&info, None) }.map_err(RhiError::vk)?;
    device.set_name(raw, name);
    // SAFETY: buffer is live.
    let requirements = unsafe { device.raw().get_buffer_memory_requirements(raw) };
    let alloc = match device.allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
        name,
        requirements,
        location,
        linear: true,
        allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
    }) {
        Ok(a) => a,
        Err(e) => {
            // SAFETY: the buffer was created above and owns no memory yet.
            unsafe { device.raw().destroy_buffer(raw, None) };
            return Err(e);
        }
    };
    // SAFETY: fresh buffer, fresh memory, offset from the allocator.
    if let Err(e) = unsafe {
        device
            .raw()
            .bind_buffer_memory(raw, alloc.memory(), alloc.offset())
    } {
        let _ = device.free(alloc);
        // SAFETY: as above — the buffer is unbound and unused.
        unsafe { device.raw().destroy_buffer(raw, None) };
        return Err(RhiError::vk(e));
    }
    Ok(Buffer {
        raw,
        alloc: Some(alloc),
        size,
        address: 0,
    })
}
