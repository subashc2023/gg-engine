//! Uploads (§4.3): a ring-buffer staging arena on the dedicated transfer
//! queue, timeline-signaled, with a synchronous "upload now" path for tools
//! and tests beside the async path a frame uses.
//!
//! The hard part is not the copy, it is the **queue-family ownership
//! transfer**. A resource written by the transfer family and read by the
//! graphics family needs a matching pair of barriers — a *release* recorded on
//! the transfer command buffer and an *acquire* recorded on the graphics one,
//! with identical layouts and family indices — and the two are ordered by the
//! graphics submit waiting on the transfer timeline. Skipping the pair leaves
//! contents undefined on hardware where the families are genuinely different
//! engines, which is precisely the hardware nobody develops on. When the
//! device has one family (lavapipe), there is nothing to transfer and a single
//! barrier on the transfer buffer does the whole job:
//! [`Uploader::take_acquires`] comes back empty and the graphics side records
//! nothing.
//!
//! # Nothing here is bounded by the ring
//!
//! An upload larger than [`RING_BYTES`] is split — buffers by byte range,
//! images by bands of whole texel-block rows — and a *batch* that would outgrow
//! the ring submits itself rather than failing, because its own bytes are the
//! ones it would otherwise be waiting on. A caller streaming a pack in
//! therefore never has to know this file's constants (§4.6).

use crate::RhiError;
use crate::device::Device;
use crate::resource::{Buffer, ImageFormat, create_raw_buffer_in, mip_extent};
use ash::vk;
use std::collections::VecDeque;

/// Staging ring capacity. Nothing is refused for exceeding it: an upload
/// larger than the ring is split, and a *batch* that outgrows it is submitted
/// early (see [`Uploader::stage`]). A pack holds more texels than any ring
/// worth allocating, so the size is a throughput knob rather than a limit
/// (§4.6 streaming).
pub(crate) const RING_BYTES: u64 = 8 << 20;

/// Offset alignment for a staged block. 64 covers both hard requirements at
/// once — 4 bytes for buffer copies, the 16-byte texel block for BC7 — with
/// room for the copy alignment drivers prefer.
const STAGE_ALIGN: u64 = 64;

/// Command buffers in rotation. One would mean waiting out each batch before
/// recording the next, which is exactly the stall streaming exists to avoid;
/// four lets the CPU stay three batches ahead of the transfer queue.
const BATCH_SLOTS: usize = 4;

/// A barrier the *graphics* queue must record before it first reads something
/// the transfer queue wrote. Produced only when the families differ.
#[derive(Clone, Copy)]
pub(crate) enum Acquire {
    Image {
        image: vk::Image,
        aspect: vk::ImageAspectFlags,
        /// One mip level per acquire, because one upload writes one level.
        level: u32,
    },
    Buffer {
        buffer: vk::Buffer,
        size: u64,
    },
}

/// One flushed batch: the transfer timeline value that proves it done, and
/// where the ring's head stood afterwards.
struct Batch {
    timeline_value: u64,
    head_after: u64,
}

/// The staging arena and the transfer-queue command buffers it feeds.
pub(crate) struct Uploader {
    ring: Buffer,
    /// Monotonic byte cursors into the ring. `head - tail <= RING_BYTES` is the
    /// invariant; the ring offset is the cursor modulo capacity. `tail` is the
    /// first byte no *live* block owns, so it moves both when a batch retires
    /// and when [`Uploader::stage`] steps over dead space nothing will read.
    head: u64,
    tail: u64,
    /// Where the *unflushed* batch began. A batch cannot outgrow the ring —
    /// nothing in flight owns its bytes, so waiting could never reclaim them —
    /// which is why reaching that point submits rather than fails.
    batch_start: u64,
    in_flight: VecDeque<Batch>,
    pool: vk::CommandPool,
    cmds: [vk::CommandBuffer; BATCH_SLOTS],
    /// Which slot the current (or next) batch records into, and the timeline
    /// value each slot's last submission signaled — the only proof a buffer is
    /// free to reset. Zero means never submitted.
    slot: usize,
    slot_retire: [u64; BATCH_SLOTS],
    recording: bool,
    timeline_value: u64,
    acquires: Vec<Acquire>,
}

impl Uploader {
    pub fn new(device: &mut Device) -> Result<Self, RhiError> {
        let ring = create_raw_buffer_in(
            device,
            "gg.staging.ring",
            RING_BYTES,
            vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::CpuToGpu,
        )?;
        // RESET_COMMAND_BUFFER, not a pool reset: the slots retire independently
        // and resetting the pool would recycle buffers still in flight.
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )
            .queue_family_index(device.transfer.family);
        // SAFETY: device is live; the family index came from device creation.
        let pool = match unsafe { device.raw().create_command_pool(&pool_info, None) } {
            Ok(p) => p,
            Err(e) => {
                destroy_ring(device, ring);
                return Err(RhiError::Vk(e));
            }
        };
        device.set_name(pool, "gg.upload.pool");
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(BATCH_SLOTS as u32);
        // SAFETY: pool is live.
        let allocated = match unsafe { device.raw().allocate_command_buffers(&alloc_info) } {
            Ok(bufs) => bufs,
            Err(e) => {
                // SAFETY: the pool was created above and owns no buffers yet.
                unsafe { device.raw().destroy_command_pool(pool, None) };
                destroy_ring(device, ring);
                return Err(RhiError::Vk(e));
            }
        };
        let Ok(cmds) = <[vk::CommandBuffer; BATCH_SLOTS]>::try_from(allocated.as_slice()) else {
            // SAFETY: as above; the buffers are freed with the pool.
            unsafe { device.raw().destroy_command_pool(pool, None) };
            destroy_ring(device, ring);
            return Err(RhiError::Loader("no upload command buffers".into()));
        };
        for (index, cmd) in cmds.iter().enumerate() {
            device.set_name(*cmd, &format!("gg.upload.cmd.{index}"));
        }

        Ok(Self {
            ring,
            head: 0,
            tail: 0,
            batch_start: 0,
            in_flight: VecDeque::new(),
            pool,
            cmds,
            slot: 0,
            slot_retire: [0; BATCH_SLOTS],
            recording: false,
            timeline_value: 0,
            acquires: Vec::new(),
        })
    }

    /// Barriers the graphics queue owes for everything flushed so far, taken
    /// out (they are recorded once). Empty when the device has a single queue
    /// family — see the module docs.
    pub fn take_acquires(&mut self) -> Vec<Acquire> {
        std::mem::take(&mut self.acquires)
    }

    /// Drop the barriers owed for an image about to be destroyed.
    ///
    /// Uploading a resource and retiring it before the next frame is ordinary
    /// use — a texture replaced twice between two presents does it — and the
    /// acquire queue holds *raw* handles, so without this the next frame
    /// records a barrier naming a destroyed image. That is a validation error
    /// and, on a real driver, a crash. Nothing is lost by dropping it: the
    /// barrier exists to hand ownership to a reader, and a retired resource
    /// has none.
    ///
    /// Unreachable where one family owns everything, which is every lavapipe
    /// run and therefore every automated tier (§4.3).
    pub fn forget_image_acquires(&mut self, image: vk::Image) {
        self.acquires
            .retain(|owed| !matches!(owed, Acquire::Image { image: dead, .. } if *dead == image));
    }

    /// [`Uploader::forget_image_acquires`] for a buffer.
    pub fn forget_buffer_acquires(&mut self, buffer: vk::Buffer) {
        self.acquires.retain(
            |owed| !matches!(owed, Acquire::Buffer { buffer: dead, .. } if *dead == buffer),
        );
    }

    /// The transfer timeline value the last [`Uploader::flush`] signaled — what
    /// a graphics submit waits on to order its acquires after the releases.
    pub fn timeline_value(&self) -> u64 {
        self.timeline_value
    }

    /// Copy `bytes` into `dst` at `dst_offset` (§4.3 async path: recorded now,
    /// submitted by [`Uploader::flush`]). `dst_size` is the destination's full
    /// size, which is what the range is bounds-checked against.
    pub fn upload_buffer(
        &mut self,
        device: &Device,
        dst: vk::Buffer,
        dst_size: u64,
        dst_offset: u64,
        bytes: &[u8],
    ) -> Result<(), RhiError> {
        if bytes.is_empty() {
            return Ok(());
        }
        let end = dst_offset
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| RhiError::Loader("upload range overflows".into()))?;
        if end > dst_size {
            return Err(RhiError::Loader(format!(
                "upload of {} bytes at offset {dst_offset} overruns a {dst_size}-byte buffer",
                bytes.len()
            )));
        }
        // Split at the ring's capacity rather than refusing: a mesh larger than
        // the ring is ordinary at Sponza's size, and the chunk boundary is
        // invisible to the destination — one buffer, one release barrier.
        for (index, chunk) in bytes.chunks(RING_BYTES as usize).enumerate() {
            let src_offset = self.stage(device, chunk)?;
            self.begin(device)?;
            let at = dst_offset + (index as u64) * RING_BYTES;
            let regions = [vk::BufferCopy2::default()
                .src_offset(src_offset)
                .dst_offset(at)
                .size(chunk.len() as u64)];
            let copy = vk::CopyBufferInfo2::default()
                .src_buffer(self.ring.raw)
                .dst_buffer(dst)
                .regions(&regions);
            // SAFETY: cmd is recording (begin above); both buffers are live and
            // the range was bounds-checked against dst_size.
            unsafe { device.raw().cmd_copy_buffer2(self.cmd(), &copy) };
        }

        self.release_buffer(device, dst, dst_size);
        Ok(())
    }

    /// Copy `bytes` into one mip `level` of `image`, transitioning that level
    /// into shader-read layout. `extent` is the image's level-0 size; `bytes`
    /// must be the tightly packed content of the *level*, whose size follows
    /// from the two ([`ImageFormat::packed_size`] of [`mip_extent`]).
    ///
    /// One level per call, and each level's barrier pair names only itself, so
    /// a chain uploaded over several frames leaves the levels it has not
    /// reached yet untouched rather than repeatedly discarded.
    pub fn upload_image(
        &mut self,
        device: &Device,
        image: vk::Image,
        format: ImageFormat,
        extent: (u32, u32),
        level: u32,
        bytes: &[u8],
    ) -> Result<(), RhiError> {
        let level_extent = mip_extent(extent, level);
        let expected = format.packed_size(level_extent);
        if bytes.len() as u64 != expected {
            return Err(RhiError::Loader(format!(
                "image upload: {format:?} level {level} of {extent:?} is {level_extent:?} and \
                 packs to {expected} bytes, got {}",
                bytes.len()
            )));
        }
        let aspect = format.aspect();
        let subresource = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect)
            .base_mip_level(level)
            .level_count(1)
            .layer_count(1);

        self.begin(device)?;
        let to_dst = [vk::ImageMemoryBarrier2::default()
            .image(image)
            .subresource_range(subresource)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)];
        // SAFETY: cmd is recording; the image is live and this level has not
        // been written, so discarding its contents via UNDEFINED is correct.
        unsafe {
            device.raw().cmd_pipeline_barrier2(
                self.cmd(),
                &vk::DependencyInfo::default().image_memory_barriers(&to_dst),
            )
        };

        // Bands of whole texel-block rows. A copy's height must be a multiple
        // of the block edge *unless* it reaches the mip's bottom, which is what
        // makes the last band legal at a non-block-aligned height.
        let block = format.block_extent();
        // Exactly `rows * band_bytes == bytes.len()`: both round the extent up
        // to whole blocks the same way, which is what lets the last band be a
        // slice rather than a special case.
        let band_bytes = format.packed_size((level_extent.0, block));
        let rows = u64::from(level_extent.1.div_ceil(block));
        let per_band = (RING_BYTES / band_bytes.max(1)).max(1);
        let mut first_row = 0;
        while first_row < rows {
            let band_rows = per_band.min(rows - first_row);
            // Both fit u32: the band lies inside a level whose height does.
            let y = (first_row * u64::from(block)) as u32;
            let height = ((band_rows * u64::from(block)) as u32).min(level_extent.1 - y);
            let from = (first_row * band_bytes) as usize;
            let to = from + (band_rows * band_bytes) as usize;
            let src_offset = self.stage(device, &bytes[from..to])?;
            self.begin(device)?;
            first_row += band_rows;

            let regions = [vk::BufferImageCopy2::default()
                .buffer_offset(src_offset)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(aspect)
                        .mip_level(level)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D {
                    x: 0,
                    y: y as i32,
                    z: 0,
                })
                .image_extent(vk::Extent3D {
                    width: level_extent.0,
                    height,
                    depth: 1,
                })];
            let copy = vk::CopyBufferToImageInfo2::default()
                .src_buffer(self.ring.raw)
                .dst_image(image)
                .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .regions(&regions);
            // SAFETY: cmd is recording; the barrier above put this level in the
            // named layout and the band is inside the staged range.
            unsafe { device.raw().cmd_copy_buffer_to_image2(self.cmd(), &copy) };
        }

        self.release_image(device, image, aspect, level);
        Ok(())
    }

    /// Submit the recorded batch on the transfer queue, signaling the transfer
    /// timeline. Returns the signaled value (0 when there was nothing to do).
    pub fn flush(&mut self, device: &Device) -> Result<u64, RhiError> {
        if !self.recording {
            // Nothing recorded, but `stage` may still have advanced the head;
            // leaving `batch_start` behind it would make the next batch look
            // bigger than it is and submit early forever.
            self.batch_start = self.head;
            return Ok(self.timeline_value);
        }
        // SAFETY: cmd has been recording since `begin`.
        unsafe { device.raw().end_command_buffer(self.cmd()) }.map_err(RhiError::Vk)?;
        self.recording = false;

        let signal_value = self.timeline_value + 1;
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(device.transfer.timeline)
            .value(signal_value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let cmd_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(self.cmd())];
        let submit = vk::SubmitInfo2::default()
            .command_buffer_infos(&cmd_infos)
            .signal_semaphore_infos(&signal);
        // SAFETY: queue and semaphore are live; cmd finished recording above.
        unsafe {
            device
                .raw()
                .queue_submit2(device.transfer.raw, &[submit], vk::Fence::null())
        }
        .map_err(RhiError::Vk)?;

        self.timeline_value = signal_value;
        self.in_flight.push_back(Batch {
            timeline_value: signal_value,
            head_after: self.head,
        });
        self.batch_start = self.head;
        self.slot_retire[self.slot] = signal_value;
        self.slot = (self.slot + 1) % BATCH_SLOTS;
        Ok(signal_value)
    }

    /// Flush and wait — the §4.3 "upload now" path for tools and tests. The
    /// acquire barriers still have to be recorded by the graphics queue before
    /// it reads anything; waiting proves the *release* happened, not that the
    /// resource is visible to a shader.
    pub fn flush_blocking(&mut self, device: &Device) -> Result<(), RhiError> {
        let value = self.flush(device)?;
        if value > 0 {
            device.wait_transfer_timeline(value)?;
        }
        Ok(())
    }

    /// Record the graphics-side half of every pending ownership transfer.
    ///
    /// # Safety
    /// `cmd` must be a recording command buffer on the graphics family, and
    /// its submission must wait on the transfer timeline at
    /// [`Uploader::timeline_value`] so the acquires follow their releases.
    pub unsafe fn record_acquires(device: &Device, cmd: vk::CommandBuffer, acquires: &[Acquire]) {
        if acquires.is_empty() {
            return;
        }
        let (src, dst) = (device.transfer.family, device.graphics.family);
        let mut image_barriers = Vec::new();
        let mut buffer_barriers = Vec::new();
        for acquire in acquires {
            match *acquire {
                Acquire::Image {
                    image,
                    aspect,
                    level,
                } => image_barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(image)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(aspect)
                                .base_mip_level(level)
                                .level_count(1)
                                .layer_count(1),
                        )
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(src)
                        .dst_queue_family_index(dst)
                        // An acquire's source scopes are empty by definition:
                        // the release already made the write available, and the
                        // semaphore ordered the two.
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(SHADER_READ_STAGES)
                        .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ),
                ),
                Acquire::Buffer { buffer, size } => buffer_barriers.push(
                    vk::BufferMemoryBarrier2::default()
                        .buffer(buffer)
                        .size(size)
                        .src_queue_family_index(src)
                        .dst_queue_family_index(dst)
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(BUFFER_READ_STAGES)
                        .dst_access_mask(BUFFER_READ_ACCESS),
                ),
            }
        }
        let info = vk::DependencyInfo::default()
            .image_memory_barriers(&image_barriers)
            .buffer_memory_barriers(&buffer_barriers);
        // SAFETY: caller contract — cmd is recording on the graphics family and
        // its submit waits on the transfer timeline.
        unsafe { device.raw().cmd_pipeline_barrier2(cmd, &info) };
    }

    /// Teardown. Caller waited the device idle.
    pub fn destroy(&mut self, device: &mut Device) {
        // SAFETY: handles belong to this device; GPU idle per contract.
        unsafe { device.raw().destroy_command_pool(self.pool, None) };
        let ring = std::mem::replace(
            &mut self.ring,
            Buffer {
                raw: vk::Buffer::null(),
                alloc: None,
                size: 0,
                address: 0,
            },
        );
        destroy_ring(device, ring);
    }

    // ---- internals -------------------------------------------------------

    fn cmd(&self) -> vk::CommandBuffer {
        self.cmds[self.slot]
    }

    fn begin(&mut self, device: &Device) -> Result<(), RhiError> {
        if self.recording {
            return Ok(());
        }
        // Resetting a buffer whose submission is still executing is undefined,
        // and the ring's reclaim wait does not prove it retired — a small batch
        // never touches the ring's far side. The timeline does prove it, and
        // with BATCH_SLOTS in rotation this only waits once the CPU is that far
        // ahead of the transfer queue.
        let retire = self.slot_retire[self.slot];
        if retire > 0 {
            device.wait_transfer_timeline(retire)?;
        }
        // SAFETY: the wait above proves this buffer is no longer in flight.
        unsafe {
            device
                .raw()
                .reset_command_buffer(self.cmd(), vk::CommandBufferResetFlags::empty())
                .map_err(RhiError::Vk)?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .raw()
                .begin_command_buffer(self.cmd(), &begin)
                .map_err(RhiError::Vk)?;
        }
        self.recording = true;
        Ok(())
    }

    /// Reserve `bytes.len()` in the ring and copy into it, returning the byte
    /// offset. Blocks on the transfer timeline when the ring has wrapped onto
    /// bytes a submitted batch may still be reading.
    fn stage(&mut self, device: &Device, bytes: &[u8]) -> Result<u64, RhiError> {
        let size = bytes.len() as u64;
        if size > RING_BYTES {
            return Err(RhiError::Loader(format!(
                "staging {size} bytes exceeds the {RING_BYTES}-byte staging ring; the caller \
                 splits an upload this large before it reaches here"
            )));
        }
        // A batch cannot outgrow the ring: its own bytes are the ones the loop
        // below would have to wait on, and nothing in flight owns them. So the
        // batch is submitted here instead — which is what lets a caller push a
        // whole pack through without knowing the ring's size (§4.6). Measured
        // from `placed` rather than from the raw head: the gap a wrap steps over
        // lies between the batch's first byte and this block, so a batch that
        // only outgrows the ring *because* of a wrap has to submit here too —
        // otherwise the loop below reaches its own unreclaimable bytes.
        if placed(self.head, size) + size - self.batch_start > RING_BYTES {
            self.flush(device)?;
        }
        let start = loop {
            let start = placed(self.head, size);
            // The gap that alignment and the wrap step over is *dead*, not
            // reserved: no batch owns it, nothing is ever read from it, and no
            // wait can release it. Once everything below has retired — nothing
            // in flight past `tail`, current batch empty — the cursor passes it
            // in one step instead of charging it to this batch, which is what
            // used to refuse a wrapping stage outright.
            if self.tail == self.head && self.batch_start == self.head {
                self.tail = start;
                self.batch_start = start;
            }
            if start + size - self.tail <= RING_BYTES {
                self.head = start + size;
                break start;
            }
            let Some(batch) = self.in_flight.pop_front() else {
                // Unreachable: draining to empty leaves `tail` at `batch_start`,
                // and the submit above left that either equal to `head` — the
                // retirement above then applies — or near enough that
                // `start + size` is within a ring of it. Named rather than
                // unwrapped, because a deadlock is the alternative.
                return Err(RhiError::Loader(format!(
                    "{size} bytes cannot be staged in the {RING_BYTES}-byte ring with nothing in \
                     flight to reclaim"
                )));
            };
            device.wait_transfer_timeline(batch.timeline_value)?;
            self.tail = batch.head_after;
        };

        let offset = start % RING_BYTES;
        let mapped = self
            .ring
            .alloc
            .as_mut()
            .and_then(|a| a.mapped_slice_mut())
            .ok_or_else(|| RhiError::Allocator("staging ring is not host-visible".into()))?;
        let at = offset as usize;
        mapped[at..at + bytes.len()].copy_from_slice(bytes);
        Ok(offset)
    }

    /// The transfer-queue half of an image's ownership transfer — or, when one
    /// family owns everything, the whole barrier.
    fn release_image(
        &mut self,
        device: &Device,
        image: vk::Image,
        aspect: vk::ImageAspectFlags,
        level: u32,
    ) {
        let subresource = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect)
            .base_mip_level(level)
            .level_count(1)
            .layer_count(1);
        let barrier = vk::ImageMemoryBarrier2::default()
            .image(image)
            .subresource_range(subresource)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE);
        let barrier = if device.transfer_crosses_families() {
            self.acquires.push(Acquire::Image {
                image,
                aspect,
                level,
            });
            // A release's destination scopes must be empty, and its stage mask
            // must be one the transfer family supports — which is why the
            // shader-read stage lives on the acquire and not here.
            barrier
                .src_queue_family_index(device.transfer.family)
                .dst_queue_family_index(device.graphics.family)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_access_mask(vk::AccessFlags2::NONE)
        } else {
            barrier
                .dst_stage_mask(SHADER_READ_STAGES)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
        };
        let barriers = [barrier];
        // SAFETY: cmd is recording; the image is live and in TRANSFER_DST after
        // the copy this follows.
        unsafe {
            device.raw().cmd_pipeline_barrier2(
                self.cmd(),
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            )
        };
    }

    /// The transfer-queue half of a buffer's ownership transfer. With one
    /// family this is a no-op: the timeline signal already makes the write
    /// available and the wait makes it visible.
    fn release_buffer(&mut self, device: &Device, dst: vk::Buffer, dst_size: u64) {
        if !device.transfer_crosses_families() {
            return;
        }
        self.acquires.push(Acquire::Buffer {
            buffer: dst,
            size: dst_size,
        });
        let barriers = [vk::BufferMemoryBarrier2::default()
            .buffer(dst)
            .size(dst_size)
            .src_queue_family_index(device.transfer.family)
            .dst_queue_family_index(device.graphics.family)
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::NONE)];
        // SAFETY: cmd is recording; the buffer is live and was just written.
        unsafe {
            device.raw().cmd_pipeline_barrier2(
                self.cmd(),
                &vk::DependencyInfo::default().buffer_memory_barriers(&barriers),
            )
        };
    }
}

/// Where a `size`-byte block lands with the ring cursor at `head`: aligned, and
/// pushed on to the next ring boundary rather than straddling the wrap point,
/// because one copy reads one contiguous range. Pure, so [`Uploader::stage`]
/// can ask the same question before it commits to anything.
fn placed(head: u64, size: u64) -> u64 {
    let start = head.next_multiple_of(STAGE_ALIGN);
    if start % RING_BYTES + size > RING_BYTES {
        (start / RING_BYTES + 1) * RING_BYTES
    } else {
        start
    }
}

/// Stages that read an uploaded texture. Both shader stages, because a
/// material index may be resolved in either (§4.3: materials are indices).
const SHADER_READ_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::VERTEX_SHADER.as_raw()
        | vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw(),
);

/// Stages and accesses that read an uploaded buffer: pulled through its device
/// address in a shader, or consumed as the fixed-function index stream.
const BUFFER_READ_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::VERTEX_SHADER.as_raw()
        | vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw()
        | vk::PipelineStageFlags2::INDEX_INPUT.as_raw(),
);
const BUFFER_READ_ACCESS: vk::AccessFlags2 = vk::AccessFlags2::from_raw(
    vk::AccessFlags2::SHADER_STORAGE_READ.as_raw() | vk::AccessFlags2::INDEX_READ.as_raw(),
);

fn destroy_ring(device: &mut Device, mut ring: Buffer) {
    if ring.raw != vk::Buffer::null() {
        // SAFETY: the ring belongs to this device and is destroyed once — the
        // caller either failed bring-up or waited idle at teardown.
        unsafe { device.raw().destroy_buffer(ring.raw, None) };
    }
    if let Some(alloc) = ring.alloc.take() {
        let _ = device.free(alloc);
    }
}
