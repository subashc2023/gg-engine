//! The deletion queue (§4.3): destruction is deferred and keyed to graphics
//! timeline values — nothing is destroyed while the GPU may still read it.
//! First customer: retired swapchains and their views on recreation.

use crate::device::Device;
use ash::vk;

/// A resource waiting for the GPU to be provably past it. `Allocation` is
/// boxed: it is an order of magnitude larger than the handles beside it, and
/// the queue is a `Vec` of these.
pub(crate) enum Deferred {
    Swapchain(vk::SwapchainKHR),
    ImageView(vk::ImageView),
    Image(vk::Image),
    Buffer(vk::Buffer),
    Allocation(Box<gpu_allocator::vulkan::Allocation>),
    Semaphore(vk::Semaphore),
    Pipeline(vk::Pipeline),
    PipelineLayout(vk::PipelineLayout),
}

impl Deferred {
    fn destroy(self, device: &mut Device) {
        // Memory is returned to the allocator, not destroyed — and it must be
        // returned *after* the object bound to it, which the queue guarantees
        // by insertion order (retire_* pushes the handle first).
        if let Deferred::Allocation(alloc) = self {
            let _ = device.free(*alloc);
            return;
        }
        // SAFETY (all arms): the handle belongs to this device, was retired at
        // a timeline value the caller proved complete, and is destroyed once —
        // entries are drained out of the queue.
        unsafe {
            match self {
                Deferred::Swapchain(s) => device.swapchain_fns().destroy_swapchain(s, None),
                Deferred::ImageView(v) => device.raw().destroy_image_view(v, None),
                Deferred::Image(i) => device.raw().destroy_image(i, None),
                Deferred::Buffer(b) => device.raw().destroy_buffer(b, None),
                Deferred::Semaphore(s) => device.raw().destroy_semaphore(s, None),
                Deferred::Pipeline(p) => device.raw().destroy_pipeline(p, None),
                Deferred::PipelineLayout(l) => device.raw().destroy_pipeline_layout(l, None),
                Deferred::Allocation(_) => unreachable!("handled above"),
            }
        }
    }
}

/// Resources pending destruction, each keyed to the graphics timeline value
/// whose completion proves the GPU is done with it.
#[derive(Default)]
pub(crate) struct DeletionQueue {
    pending: Vec<(u64, Deferred)>,
}

impl DeletionQueue {
    /// Destroy `resource` once the graphics timeline reaches `after_value`.
    pub fn defer(&mut self, after_value: u64, resource: Deferred) {
        self.pending.push((after_value, resource));
    }

    /// Destroy everything whose timeline value has completed, **in insertion
    /// order**. Order is load-bearing since M4A: `retire_image`/`retire_buffer`
    /// queue the handle before the allocation backing it, and returning memory
    /// to the allocator while a live `VkBuffer` is still bound to it is a
    /// validation error. A `swap_remove` compaction permutes exactly that pair.
    pub fn collect(&mut self, device: &mut Device, completed_value: u64) {
        let mut keep = Vec::new();
        for (value, resource) in std::mem::take(&mut self.pending) {
            if value <= completed_value {
                resource.destroy(device);
            } else {
                keep.push((value, resource));
            }
        }
        self.pending = keep;
    }

    /// Teardown path: destroy everything. Caller must have waited the device
    /// idle first.
    pub fn drain_all(&mut self, device: &mut Device) {
        for (_, resource) in self.pending.drain(..) {
            resource.destroy(device);
        }
    }
}
