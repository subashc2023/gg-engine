//! The deletion queue (§4.3): destruction is deferred and keyed to graphics
//! timeline values — nothing is destroyed while the GPU may still read it.
//! First customer: retired swapchains and their views on recreation.

use crate::device::Device;
use ash::vk;

/// A resource waiting for the GPU to be provably past it.
pub(crate) enum Deferred {
    Swapchain(vk::SwapchainKHR),
    ImageView(vk::ImageView),
    Semaphore(vk::Semaphore),
    Pipeline(vk::Pipeline),
    PipelineLayout(vk::PipelineLayout),
}

impl Deferred {
    fn destroy(self, device: &Device) {
        // SAFETY (all arms): the handle belongs to this device, was retired at
        // a timeline value the caller proved complete, and is destroyed once —
        // entries are drained out of the queue.
        unsafe {
            match self {
                Deferred::Swapchain(s) => device.swapchain_fns().destroy_swapchain(s, None),
                Deferred::ImageView(v) => device.raw().destroy_image_view(v, None),
                Deferred::Semaphore(s) => device.raw().destroy_semaphore(s, None),
                Deferred::Pipeline(p) => device.raw().destroy_pipeline(p, None),
                Deferred::PipelineLayout(l) => device.raw().destroy_pipeline_layout(l, None),
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

    /// Destroy everything whose timeline value has completed.
    pub fn collect(&mut self, device: &Device, completed_value: u64) {
        let mut i = 0;
        while i < self.pending.len() {
            if self.pending[i].0 <= completed_value {
                let (_, resource) = self.pending.swap_remove(i);
                resource.destroy(device);
            } else {
                i += 1;
            }
        }
    }

    /// Teardown path: destroy everything. Caller must have waited the device
    /// idle first.
    pub fn drain_all(&mut self, device: &Device) {
        for (_, resource) in self.pending.drain(..) {
            resource.destroy(device);
        }
    }
}
