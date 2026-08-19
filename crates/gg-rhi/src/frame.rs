//! Frames-in-flight (§4.3): 2, expressed as a wait on the graphics timeline —
//! no fences-per-resource, no semaphore pools. Each flight slot owns a command
//! pool, one primary command buffer, and the binary acquire semaphore WSI
//! demands (per slot: its reuse is proven safe by the timeline wait).

use crate::RhiError;
use crate::device::Device;
use ash::vk;

/// How many frames may be in flight at once (§4.3: 2).
pub const FRAMES_IN_FLIGHT: u64 = 2;

/// One flight slot's resources.
pub(crate) struct FrameSlot {
    pub pool: vk::CommandPool,
    pub cmd: vk::CommandBuffer,
    pub acquire: vk::Semaphore,
}

/// The flight slots, indexed by `frame_index % FRAMES_IN_FLIGHT`.
pub(crate) struct Frames {
    pub slots: Vec<FrameSlot>,
}

impl Frames {
    pub fn new(device: &Device) -> Result<Self, RhiError> {
        let mut slots = Vec::with_capacity(FRAMES_IN_FLIGHT as usize);
        for i in 0..FRAMES_IN_FLIGHT {
            let pool_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                .queue_family_index(device.graphics.family);
            // SAFETY: device is live.
            let pool = unsafe { device.raw().create_command_pool(&pool_info, None) }
                .map_err(RhiError::vk)?;
            device.set_name(pool, &format!("gg.frame{i}.pool"));

            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: pool is live.
            let cmd = unsafe { device.raw().allocate_command_buffers(&alloc_info) }
                .map_err(RhiError::vk)?
                .into_iter()
                .next()
                .ok_or_else(|| RhiError::Loader("no command buffer allocated".into()))?;
            device.set_name(cmd, &format!("gg.frame{i}.cmd"));

            let semaphore_info = vk::SemaphoreCreateInfo::default();
            // SAFETY: device is live.
            let acquire = unsafe { device.raw().create_semaphore(&semaphore_info, None) }
                .map_err(RhiError::vk)?;
            device.set_name(acquire, &format!("gg.frame{i}.acquire"));

            slots.push(FrameSlot { pool, cmd, acquire });
        }
        Ok(Self { slots })
    }

    /// Teardown. Caller waited the device idle.
    pub fn destroy(&mut self, device: &Device) {
        for slot in self.slots.drain(..) {
            // SAFETY: handles belong to this device; GPU idle per contract.
            // Destroying the pool frees its command buffer.
            unsafe {
                device.raw().destroy_semaphore(slot.acquire, None);
                device.raw().destroy_command_pool(slot.pool, None);
            }
        }
    }
}
