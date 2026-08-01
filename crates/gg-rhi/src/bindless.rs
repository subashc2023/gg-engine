//! The global descriptor set (§4.3, bindless from day one). **One** set for
//! the whole engine: an unbounded sampled-image array, an unbounded
//! storage-image array, and the immutable sampler set. Materials are indices
//! into these arrays; buffers are not here at all, because they are reached by
//! device address (`resource.rs`).
//!
//! This is the descriptor-set-layout combinatorics §4.3 refuses, deleted at the
//! root: there is one layout, every pipeline uses it, and it is bound once per
//! command buffer. Adding a texture is a descriptor *write*, never a new set,
//! never a new layout, never a pipeline rebuild — which is what
//! `descriptorBindingSampledImageUpdateAfterBind` buys and why the feature is
//! asserted at bring-up rather than probed for here.

use crate::RhiError;
use crate::device::Device;
use crate::resource::Resources;
use ash::vk;

/// Binding numbers inside the global set. Shaders must agree; the generated
/// Slang side names the same three (§4.4).
pub(crate) const BINDING_SAMPLED_IMAGES: u32 = 0;
pub(crate) const BINDING_STORAGE_IMAGES: u32 = 1;
pub(crate) const BINDING_SAMPLERS: u32 = 2;

/// What we ask for before the device's own limit clamps it. Descriptors are
/// cheap and the arrays are partially bound, so the cost of a generous array
/// is the pool's bookkeeping, not memory per slot.
const WANT_SAMPLED_IMAGES: u32 = 4096;
const WANT_STORAGE_IMAGES: u32 = 256;

/// A texture's slot in the global sampled-image array — the whole of what a
/// material stores about a texture (§4.3: materials are indices).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TextureIndex(u32);

impl TextureIndex {
    /// The raw index, for the push constant or scene buffer that carries it.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// A slot in the global storage-image array.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct StorageImageIndex(u32);

impl StorageImageIndex {
    /// The raw index, for the push constant that carries it.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// A bump-and-free-list slot allocator over one descriptor array.
struct Slots {
    capacity: u32,
    next: u32,
    free: Vec<u32>,
}

impl Slots {
    fn new(capacity: u32) -> Self {
        Self {
            capacity,
            next: 0,
            free: Vec::new(),
        }
    }

    fn alloc(&mut self, what: &str) -> Result<u32, RhiError> {
        if let Some(index) = self.free.pop() {
            return Ok(index);
        }
        if self.next >= self.capacity {
            return Err(RhiError::Loader(format!(
                "the global {what} array is full ({} slots)",
                self.capacity
            )));
        }
        let index = self.next;
        self.next += 1;
        Ok(index)
    }

    fn release(&mut self, index: u32) {
        self.free.push(index);
    }
}

/// The one descriptor set, its layout, and its pool.
pub(crate) struct Bindless {
    layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    sampled: Slots,
    storage: Slots,
}

impl Bindless {
    pub fn new(device: &Device, resources: &Resources) -> Result<Self, RhiError> {
        let limits = device.descriptor_limits();
        let sampled_capacity = WANT_SAMPLED_IMAGES.min(limits.sampled_images).max(1);
        let storage_capacity = WANT_STORAGE_IMAGES.min(limits.storage_images).max(1);
        tracing::info!(
            sampled_images = sampled_capacity,
            storage_images = storage_capacity,
            samplers = resources.samplers().len(),
            "bindless global set"
        );

        let samplers = resources.samplers();
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_SAMPLED_IMAGES)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(sampled_capacity)
                .stage_flags(vk::ShaderStageFlags::ALL),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_STORAGE_IMAGES)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(storage_capacity)
                .stage_flags(vk::ShaderStageFlags::ALL),
            // Immutable: the handles are baked into the layout, so no sampler
            // descriptor is ever written and none can be wrong at draw time.
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_SAMPLERS)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .descriptor_count(samplers.len() as u32)
                .stage_flags(vk::ShaderStageFlags::ALL)
                .immutable_samplers(samplers),
        ];
        let image_flags = vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING;
        let binding_flags = [
            image_flags,
            image_flags,
            vk::DescriptorBindingFlags::empty(),
        ];
        let mut flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .bindings(&bindings)
            .push_next(&mut flags_info);
        // SAFETY: device is live; every array pointed at outlives the call.
        let layout = unsafe {
            device
                .raw()
                .create_descriptor_set_layout(&layout_info, None)
        }
        .map_err(RhiError::Vk)?;
        device.set_name(layout, "gg.bindless.layout");

        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(sampled_capacity),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(storage_capacity),
            // Immutable samplers still occupy pool slots: the layout supplies
            // the handles, the pool still allocates the descriptors.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(samplers.len() as u32),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(1)
            .pool_sizes(&sizes);
        // SAFETY: device is live; sizes outlive the call.
        let pool = match unsafe { device.raw().create_descriptor_pool(&pool_info, None) } {
            Ok(p) => p,
            Err(e) => {
                // SAFETY: the layout was created above and is unused.
                unsafe { device.raw().destroy_descriptor_set_layout(layout, None) };
                return Err(RhiError::Vk(e));
            }
        };
        device.set_name(pool, "gg.bindless.pool");

        let layouts = [layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        // SAFETY: pool and layout are live; one set fits max_sets(1).
        let set = match unsafe { device.raw().allocate_descriptor_sets(&alloc_info) } {
            Ok(sets) => sets.into_iter().next(),
            Err(e) => {
                // SAFETY: pool and layout were created above; nothing was
                // allocated from the pool.
                unsafe {
                    device.raw().destroy_descriptor_pool(pool, None);
                    device.raw().destroy_descriptor_set_layout(layout, None);
                }
                return Err(RhiError::Vk(e));
            }
        };
        let Some(set) = set else {
            // SAFETY: as above.
            unsafe {
                device.raw().destroy_descriptor_pool(pool, None);
                device.raw().destroy_descriptor_set_layout(layout, None);
            }
            return Err(RhiError::Loader(
                "descriptor set allocation returned nothing".into(),
            ));
        };
        device.set_name(set, "gg.bindless.set");

        Ok(Self {
            layout,
            pool,
            set,
            sampled: Slots::new(sampled_capacity),
            storage: Slots::new(storage_capacity),
        })
    }

    pub fn layout(&self) -> vk::DescriptorSetLayout {
        self.layout
    }

    pub fn set(&self) -> vk::DescriptorSet {
        self.set
    }

    /// Give `view` a slot in the sampled-image array and write the descriptor.
    /// Legal while frames are in flight — that is what update-after-bind means
    /// — provided no in-flight draw indexes *this* slot, which a fresh slot
    /// cannot be.
    pub fn register_sampled(
        &mut self,
        device: &Device,
        view: vk::ImageView,
    ) -> Result<TextureIndex, RhiError> {
        let index = self.sampled.alloc("sampled-image")?;
        self.write_image(
            device,
            BINDING_SAMPLED_IMAGES,
            index,
            vk::DescriptorType::SAMPLED_IMAGE,
            view,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        Ok(TextureIndex(index))
    }

    /// Give `view` a slot in the storage-image array and write the descriptor.
    pub fn register_storage(
        &mut self,
        device: &Device,
        view: vk::ImageView,
    ) -> Result<StorageImageIndex, RhiError> {
        let index = self.storage.alloc("storage-image")?;
        self.write_image(
            device,
            BINDING_STORAGE_IMAGES,
            index,
            vk::DescriptorType::STORAGE_IMAGE,
            view,
            vk::ImageLayout::GENERAL,
        );
        Ok(StorageImageIndex(index))
    }

    /// Return a sampled slot to the free list. The descriptor is deliberately
    /// *not* cleared: PARTIALLY_BOUND makes a stale descriptor legal as long
    /// as no shader indexes it, and clearing it would be a second write for a
    /// slot the next `register_sampled` overwrites anyway.
    pub fn release_sampled(&mut self, index: TextureIndex) {
        self.sampled.release(index.0);
    }

    /// Return a storage slot to the free list.
    pub fn release_storage(&mut self, index: StorageImageIndex) {
        self.storage.release(index.0);
    }

    fn write_image(
        &self,
        device: &Device,
        binding: u32,
        index: u32,
        ty: vk::DescriptorType,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    ) {
        let info = [vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(layout)];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(binding)
            .dst_array_element(index)
            .descriptor_type(ty)
            .image_info(&info)];
        // SAFETY: the set and view are live, the slot index is inside the
        // declared capacity, and UPDATE_AFTER_BIND permits the write while
        // the set is bound to in-flight command buffers.
        unsafe { device.raw().update_descriptor_sets(&write, &[]) };
    }

    /// Teardown. Caller waited the device idle. Destroying the pool frees the
    /// set, so the set is not destroyed separately.
    pub fn destroy(&mut self, device: &Device) {
        // SAFETY: handles belong to this device; GPU idle per contract.
        unsafe {
            device.raw().destroy_descriptor_pool(self.pool, None);
            device
                .raw()
                .destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
