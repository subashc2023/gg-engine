//! GPU residency for pack assets (§4.6's runtime half, §4.3's staging ring).
//!
//! [`gg_assets::Assets`] answers "are the bytes here"; this answers "is it on
//! the device". The two are separate because they fail and finish at different
//! times: a texture's levels are decompressed by a worker thread, and the
//! upload that follows is main-thread work bounded by a per-frame byte budget.
//!
//! # Why the budget is bytes and not assets
//!
//! One 4K base level is 22 MiB and one 1x1 mip is 16 bytes. A budget counted in
//! *assets* would let a frame spend a hundred milliseconds copying or nothing
//! at all, depending only on what the queue happened to hold — which is a
//! stutter that moves when the art does. Bytes are what the transfer queue
//! actually charges for.
//!
//! # Nothing is sampled until it is whole
//!
//! A mip chain arrives level by level, possibly across frames. An image is
//! given its bindless slot only once every level has landed: a level still in
//! `UNDEFINED` layout is not merely stale, it is an invalid read, and the one
//! thing worse than a blurry texture is a validation message that only appears
//! when the disk is slow.

use std::collections::{BTreeMap, VecDeque};

use gg_assets::load::{Assets, LoadState, asset};
use gg_assets::pack::AssetId;
use gg_assets::texture::TextureFormat;
use gg_assets::{Handle, MeshError};
use gg_rhi::{
    BufferDesc, BufferHandle, BufferKind, DeviceAddress, ImageDesc, ImageFormat, ImageHandle,
    ImageUse, RhiError, TextureIndex, full_mip_count,
};

use crate::GpuHost;

/// A mesh on the device. Both buffers are reached the §4.3 way — the vertex
/// stream by device address in a push constant, the index stream as the one
/// piece of fixed function left.
#[derive(Clone, Copy, Debug)]
pub struct ResidentMesh {
    /// The vertex buffer's handle, for teardown. `None` for a mesh with no
    /// geometry, which a pack may hold and no allocator will give a buffer for
    /// — resident with nothing to draw, rather than perpetually loading.
    pub vertices: Option<BufferHandle>,
    /// What a shader pulls vertices through. Zero when there are none.
    pub address: DeviceAddress,
    /// The index buffer, bound by an indexed draw.
    pub indices: Option<BufferHandle>,
    /// Indices to draw.
    pub index_count: u32,
    /// The material this mesh names, or [`AssetId::NONE`].
    pub material: AssetId,
}

/// A texture on the device: the image, and the only thing a shader learns
/// about it (§4.3, materials are indices).
#[derive(Clone, Copy, Debug)]
pub struct ResidentTexture {
    /// The image handle, for teardown.
    pub image: ImageHandle,
    /// Its slot in the global sampled-image array.
    pub index: TextureIndex,
    /// Level 0's extent in texels.
    pub extent: (u32, u32),
}

/// What one [`Residency::pump`] managed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Progress {
    /// Bytes copied into the staging ring this call.
    pub uploaded: usize,
    /// Assets that became fully resident.
    pub finished: usize,
    /// Assets still waiting — for their bytes, or for a later budget.
    pub pending: usize,
}

impl Progress {
    /// Whether everything asked for is on the device.
    #[must_use]
    pub fn idle(&self) -> bool {
        self.pending == 0
    }
}

/// Why an asset could not be made resident.
#[derive(Debug, thiserror::Error)]
pub enum ResidencyError {
    /// The RHI refused an allocation or an upload.
    #[error(transparent)]
    Rhi(#[from] RhiError),
    /// A mesh blob would not read.
    #[error("mesh {id}: {source}")]
    Mesh {
        /// Which asset.
        id: AssetId,
        /// What the reader said.
        #[source]
        source: MeshError,
    },
    /// A texture whose block format has no `gg-rhi` equivalent. Unreachable
    /// through `ggc`, which writes three; named because the pack is a file and
    /// a file can come from anywhere.
    #[error("texture {id}: {format:?} is not a format this renderer samples")]
    Format {
        /// Which asset.
        id: AssetId,
        /// What the container declared.
        format: TextureFormat,
    },
}

/// An asset asked for and not yet finished.
enum Pending {
    Mesh(Handle<asset::Mesh>),
    /// A texture, and how many of its levels have been uploaded. The image is
    /// allocated on the first visit, which is why it is an `Option` rather than
    /// something the queue entry was built with.
    Texture {
        handle: Handle<asset::Texture>,
        image: Option<ImageHandle>,
        next_level: u32,
    },
}

impl Pending {
    fn id(&self) -> AssetId {
        match self {
            Self::Mesh(handle) => handle.id(),
            Self::Texture { handle, .. } => handle.id(),
        }
    }

    fn state(&self) -> LoadState {
        match self {
            Self::Mesh(handle) => handle.state(),
            Self::Texture { handle, .. } => handle.state(),
        }
    }
}

/// Every pack asset this renderer has on the device, and the queue of the ones
/// on their way there.
#[derive(Default)]
pub struct Residency {
    meshes: BTreeMap<AssetId, ResidentMesh>,
    textures: BTreeMap<AssetId, ResidentTexture>,
    queue: VecDeque<Pending>,
    /// Ids in `queue` or already resident, so asking twice costs nothing.
    known: BTreeMap<AssetId, ()>,
}

impl core::fmt::Debug for Residency {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Residency")
            .field("meshes", &self.meshes.len())
            .field("textures", &self.textures.len())
            .field("queued", &self.queue.len())
            .finish()
    }
}

impl Residency {
    /// Nothing resident, nothing queued.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask for a mesh. Idempotent: a second call for an asset already resident
    /// or already queued does nothing.
    pub fn want_mesh(&mut self, handle: &Handle<asset::Mesh>) {
        if self.known.insert(handle.id(), ()).is_none() {
            self.queue.push_back(Pending::Mesh(handle.clone()));
        }
    }

    /// Ask for a texture. Idempotent, as [`Self::want_mesh`].
    pub fn want_texture(&mut self, handle: &Handle<asset::Texture>) {
        if self.known.insert(handle.id(), ()).is_none() {
            self.queue.push_back(Pending::Texture {
                handle: handle.clone(),
                image: None,
                next_level: 0,
            });
        }
    }

    /// A resident mesh, or `None` while it is still on its way.
    #[must_use]
    pub fn mesh(&self, id: AssetId) -> Option<&ResidentMesh> {
        self.meshes.get(&id)
    }

    /// A resident texture, or `None` while it is still on its way.
    #[must_use]
    pub fn texture(&self, id: AssetId) -> Option<&ResidentTexture> {
        self.textures.get(&id)
    }

    /// How many assets are still on their way.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// Move up to `budget` bytes onto the device.
    ///
    /// Assets whose bytes have not arrived yet are rotated to the back rather
    /// than blocking the ones behind them, so a slow texture never holds up a
    /// mesh that is ready — and one that has *failed* is dropped, because there
    /// is nothing left to wait for.
    ///
    /// Call once a frame, after [`Assets::pump`]. One flush at the end covers
    /// everything this call recorded (§4.3).
    ///
    /// # Errors
    /// Whatever the RHI refuses, or a blob that will not read.
    pub fn pump(
        &mut self,
        rhi: &mut impl GpuHost,
        assets: &Assets,
        budget: usize,
    ) -> Result<Progress, ResidencyError> {
        let mut progress = Progress::default();
        // Bounded by the queue's length at entry: an asset rotated to the back
        // must not be reconsidered in the same call, or a queue of one waiting
        // texture would spin here until its worker finished.
        let mut visits = self.queue.len();
        while visits > 0 && progress.uploaded < budget {
            visits -= 1;
            let Some(mut item) = self.queue.pop_front() else {
                break;
            };
            match item.state() {
                LoadState::Failed => {
                    // Dropped, not retried: `Assets` already logged why, and a
                    // queue that kept it would report `pending` forever.
                    self.known.remove(&item.id());
                    continue;
                }
                LoadState::Ready => {}
                LoadState::Loading | LoadState::Unloaded => {
                    self.queue.push_back(item);
                    continue;
                }
            }
            let done = match &mut item {
                Pending::Mesh(handle) => {
                    let bytes = self.resident_mesh(rhi, assets, handle)?;
                    progress.uploaded += bytes;
                    true
                }
                Pending::Texture {
                    handle,
                    image,
                    next_level,
                } => self.resident_texture(
                    rhi,
                    assets,
                    handle,
                    image,
                    next_level,
                    budget - progress.uploaded,
                    &mut progress.uploaded,
                )?,
            };
            if done {
                progress.finished += 1;
            } else {
                // Out of budget mid-chain: the front, not the back — the next
                // frame finishes what it started rather than interleaving.
                self.queue.push_front(item);
                break;
            }
        }
        if progress.uploaded > 0 {
            rhi.flush_uploads()?;
        }
        progress.pending = self.queue.len();
        Ok(progress)
    }

    /// Allocate and fill a mesh's two buffers. Meshes are all-or-nothing: the
    /// blob is already in a mapping, so splitting the copy would buy a
    /// half-drawn mesh and nothing else. The staging ring splits it internally
    /// when it is larger than the ring (§4.3).
    fn resident_mesh(
        &mut self,
        rhi: &mut impl GpuHost,
        assets: &Assets,
        handle: &Handle<asset::Mesh>,
    ) -> Result<usize, ResidencyError> {
        let id = handle.id();
        let mesh = assets
            .mesh(handle)
            .map_err(|source| ResidencyError::Mesh { id, source })?;
        let vertex_bytes: &[u8] = bytemuck::cast_slice(mesh.vertices());
        let index_bytes: &[u8] = bytemuck::cast_slice(mesh.indices());
        if vertex_bytes.is_empty() || index_bytes.is_empty() {
            // A mesh with no geometry is legal in a pack and undrawable here;
            // recording it as resident with zero indices is what lets a scene
            // walk it without a special case.
            self.meshes.insert(
                id,
                ResidentMesh {
                    vertices: None,
                    address: 0,
                    indices: None,
                    index_count: 0,
                    material: mesh.header.material,
                },
            );
            return Ok(0);
        }

        let vertices = rhi.create_buffer(&BufferDesc {
            name: &format!("assets.{id}.vertices"),
            size: vertex_bytes.len() as u64,
            kind: BufferKind::Storage,
        })?;
        let indices = rhi.create_buffer(&BufferDesc {
            name: &format!("assets.{id}.indices"),
            size: index_bytes.len() as u64,
            kind: BufferKind::Index,
        })?;
        rhi.upload_buffer(vertices, 0, vertex_bytes)?;
        rhi.upload_buffer(indices, 0, index_bytes)?;
        self.meshes.insert(
            id,
            ResidentMesh {
                vertices: Some(vertices),
                address: rhi.buffer_address(vertices)?,
                indices: Some(indices),
                index_count: mesh.indices().len() as u32,
                material: mesh.header.material,
            },
        );
        Ok(vertex_bytes.len() + index_bytes.len())
    }

    /// Upload as much of a texture's chain as the budget allows. Returns
    /// whether the last level landed — the point at which it gets its slot.
    #[allow(clippy::too_many_arguments)] // every one is a distinct piece of the resume point
    fn resident_texture(
        &mut self,
        rhi: &mut impl GpuHost,
        assets: &Assets,
        handle: &Handle<asset::Texture>,
        image: &mut Option<ImageHandle>,
        next_level: &mut u32,
        budget: usize,
        uploaded: &mut usize,
    ) -> Result<bool, ResidencyError> {
        let id = handle.id();
        let Some(data) = assets.texture(handle) else {
            // Ready with no payload is a slot that was unloaded between the
            // state read and here; try again next frame.
            return Ok(false);
        };
        let format = rhi_format(data.format).ok_or(ResidencyError::Format {
            id,
            format: data.format,
        })?;
        let levels = data.levels.len() as u32;

        let handle_image = match *image {
            Some(existing) => existing,
            None => {
                // The chain's length is the pack's, clamped to what the extent
                // can hold: `ggc` writes a full chain, and a partial one is a
                // shorter image rather than an error the player sees.
                let mip_levels = levels.min(full_mip_count(data.extent)).max(1);
                let created = rhi.create_image(&ImageDesc {
                    name: &format!("assets.{id}"),
                    extent: data.extent,
                    format,
                    usage: ImageUse::Sampled,
                    mip_levels,
                })?;
                *image = Some(created);
                created
            }
        };
        let mip_levels = levels.min(full_mip_count(data.extent)).max(1);

        while *next_level < mip_levels {
            let level = *next_level as usize;
            let Some(bytes) = data.levels.get(level) else {
                break;
            };
            if *uploaded >= budget && *next_level > 0 {
                // Level 0 always goes, whatever the budget says: an image with
                // no levels at all is a slot nothing can be drawn with, and a
                // budget that could starve one indefinitely is a leak of GPU
                // memory rather than a limit on bandwidth.
                return Ok(false);
            }
            rhi.upload_image(handle_image, *next_level, bytes)?;
            *uploaded += bytes.len();
            *next_level += 1;
        }

        self.textures.insert(
            id,
            ResidentTexture {
                image: handle_image,
                index: rhi.register_texture(handle_image)?,
                extent: data.extent,
            },
        );
        Ok(true)
    }

    /// Forget one asset's device resources, retiring them behind the frame
    /// timeline. The next [`Self::want_mesh`] or [`Self::want_texture`] for it
    /// re-uploads — which is what an asset hot reload is (§4.6 watch mode).
    ///
    /// # Errors
    /// A handle the RHI does not recognise, which cannot happen for one issued
    /// here.
    pub fn drop_asset(
        &mut self,
        rhi: &mut impl GpuHost,
        id: AssetId,
    ) -> Result<bool, ResidencyError> {
        self.known.remove(&id);
        if let Some(mesh) = self.meshes.remove(&id) {
            destroy_mesh(rhi, mesh)?;
            return Ok(true);
        }
        if let Some(texture) = self.textures.remove(&id) {
            rhi.destroy_image(texture.image)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Release everything. Called at shutdown, before the RHI's own accounting
    /// (§4.3) — anything still held here shows up there as a leak.
    ///
    /// # Errors
    /// As [`Self::drop_asset`].
    pub fn destroy(&mut self, rhi: &mut impl GpuHost) -> Result<(), ResidencyError> {
        self.queue.clear();
        self.known.clear();
        for (_, mesh) in std::mem::take(&mut self.meshes) {
            destroy_mesh(rhi, mesh)?;
        }
        for (_, texture) in std::mem::take(&mut self.textures) {
            rhi.destroy_image(texture.image)?;
        }
        Ok(())
    }
}

fn destroy_mesh(rhi: &mut impl GpuHost, mesh: ResidentMesh) -> Result<(), ResidencyError> {
    for buffer in [mesh.vertices, mesh.indices].into_iter().flatten() {
        rhi.destroy_buffer(buffer)?;
    }
    Ok(())
}

/// The pack's block formats, as the RHI's. Total by construction on `ggc`'s
/// three; `None` is what a pack from elsewhere gets.
fn rhi_format(format: TextureFormat) -> Option<ImageFormat> {
    Some(match format {
        TextureFormat::Bc7Srgb => ImageFormat::Bc7Srgb,
        TextureFormat::Bc5Unorm => ImageFormat::Bc5Unorm,
        TextureFormat::Bc4Unorm => ImageFormat::Bc4Unorm,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use gg_assets::pack::AssetKind;
    use gg_assets::texture::{full_mip_count as tex_mips, level_extent};
    use gg_assets::{Pack, PackWriter, Vertex, mesh, texture};
    use gg_rhi::OffscreenRhi;

    const EXTENT: (u32, u32) = (64, 64);
    const MESH_NAME: &str = "demo/mesh/0.0";
    const TEX_NAME: &str = "demo/texture/0/base_color";

    fn vertex(x: f32) -> Vertex {
        Vertex {
            position: [x, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
        }
    }

    /// A pack with one mesh and one BC7 texture whose chain is full.
    fn pack() -> Pack {
        let levels: Vec<Vec<u8>> = (0..tex_mips(EXTENT.0, EXTENT.1))
            .map(|level| {
                let (w, h) = level_extent(EXTENT.0, EXTENT.1, level);
                vec![level as u8; TextureFormat::Bc7Srgb.level_bytes(w, h) as usize]
            })
            .collect();
        let mut writer = PackWriter::new();
        writer
            .add(
                MESH_NAME,
                AssetKind::Mesh,
                0,
                mesh::encode(
                    &[vertex(0.0), vertex(1.0), vertex(2.0)],
                    &[0, 1, 2],
                    AssetId::of("demo/material/0"),
                ),
            )
            .unwrap();
        writer
            .add(
                TEX_NAME,
                AssetKind::Texture,
                1,
                texture::encode(TextureFormat::Bc7Srgb, EXTENT.0, EXTENT.1, &levels).unwrap(),
            )
            .unwrap();
        Pack::from_bytes(&writer.finish().unwrap()).unwrap()
    }

    /// A generous budget: one pump takes everything.
    const WHOLE: usize = 64 << 20;

    #[test]
    fn a_pack_mesh_and_texture_reach_the_device_and_are_accounted_for() {
        let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
        let mut assets = Assets::new(pack());
        let mut residency = Residency::new();

        let mesh_handle = assets.load::<asset::Mesh>(MESH_NAME).unwrap();
        let tex_handle = assets.load::<asset::Texture>(TEX_NAME).unwrap();
        residency.want_mesh(&mesh_handle);
        residency.want_texture(&tex_handle);
        // Asking twice is not two uploads.
        residency.want_mesh(&mesh_handle);
        assert_eq!(residency.pending(), 2);

        // Before the decode lands, the texture is skipped and the mesh is not:
        // a slow texture must not hold up geometry behind it in the queue.
        let progress = residency.pump(&mut rhi, &assets, WHOLE).unwrap();
        assert_eq!(progress.finished, 1, "the mesh, whose bytes were mapped");
        assert!(residency.mesh(mesh_handle.id()).is_some());

        assets.pump_until_idle();
        let progress = residency.pump(&mut rhi, &assets, WHOLE).unwrap();
        assert!(progress.idle(), "{progress:?}");

        let mesh = residency.mesh(mesh_handle.id()).unwrap();
        assert_eq!(mesh.index_count, 3);
        assert_ne!(mesh.address, 0, "a resident mesh has a device address");
        assert_eq!(mesh.material, AssetId::of("demo/material/0"));

        let texture = residency.texture(tex_handle.id()).unwrap();
        assert_eq!(texture.extent, EXTENT);
        // Two images exist by now — the offscreen target and this one — so the
        // slot is only meaningful as "it got one", not as a fixed number.
        let _ = texture.index.get();

        residency.destroy(&mut rhi).unwrap();
        let report = rhi.shutdown();
        assert!(report.clean(), "unclean: {report:?}");
    }

    #[test]
    fn a_budget_spreads_a_chain_across_frames_without_registering_a_partial_one() {
        let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
        let mut assets = Assets::new(pack());
        let mut residency = Residency::new();

        let tex_handle = assets.load::<asset::Texture>(TEX_NAME).unwrap();
        residency.want_texture(&tex_handle);
        assets.pump_until_idle();

        // One byte of budget: level 0 goes anyway (an image with no levels is
        // GPU memory nothing can draw), and the rest waits.
        let first = residency.pump(&mut rhi, &assets, 1).unwrap();
        assert!(!first.idle(), "the chain is not finished: {first:?}");
        assert_eq!(first.finished, 0);
        assert!(
            residency.texture(tex_handle.id()).is_none(),
            "a half-uploaded chain must not get a bindless slot"
        );
        let base = TextureFormat::Bc7Srgb.level_bytes(EXTENT.0, EXTENT.1) as usize;
        assert_eq!(first.uploaded, base, "level 0 and nothing after it");

        // Frames until it finishes. Bounded: each pump moves at least a level.
        let mut pumps = 1;
        while !residency.pump(&mut rhi, &assets, 1).unwrap().idle() {
            pumps += 1;
            assert!(pumps < 32, "the chain never finished");
        }
        assert!(
            pumps > 1,
            "a one-byte budget uploaded the whole chain at once"
        );
        assert!(residency.texture(tex_handle.id()).is_some());

        residency.destroy(&mut rhi).unwrap();
        let report = rhi.shutdown();
        assert!(report.clean(), "unclean: {report:?}");
    }

    #[test]
    fn dropping_an_asset_and_asking_again_re_uploads_it() {
        let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
        let mut assets = Assets::new(pack());
        let mut residency = Residency::new();

        let tex_handle = assets.load::<asset::Texture>(TEX_NAME).unwrap();
        residency.want_texture(&tex_handle);
        assets.pump_until_idle();
        residency.pump(&mut rhi, &assets, WHOLE).unwrap();
        let first = residency.texture(tex_handle.id()).unwrap().image;

        // §4.6 watch mode's shape: the image is retired, the game's handle is
        // untouched, and the next want re-uploads through the same handle.
        assert!(residency.drop_asset(&mut rhi, tex_handle.id()).unwrap());
        assert!(residency.texture(tex_handle.id()).is_none());

        residency.want_texture(&tex_handle);
        let progress = residency.pump(&mut rhi, &assets, WHOLE).unwrap();
        assert!(progress.idle());
        let second = residency.texture(tex_handle.id()).unwrap().image;
        assert_ne!(
            first, second,
            "a re-upload is a new image, not the retired one"
        );

        residency.destroy(&mut rhi).unwrap();
        let report = rhi.shutdown();
        assert!(report.clean(), "unclean: {report:?}");
    }

    #[test]
    fn a_failed_asset_leaves_the_queue_rather_than_pending_forever() {
        let mut rhi = OffscreenRhi::new((8, 8)).unwrap();
        let mut writer = PackWriter::new();
        // Right kind, wrong bytes: this fails in the worker, not here.
        writer
            .add("tex/bad", AssetKind::Texture, 0, vec![7u8; 128])
            .unwrap();
        let mut assets = Assets::new(Pack::from_bytes(&writer.finish().unwrap()).unwrap());
        let mut residency = Residency::new();

        let handle = assets.load::<asset::Texture>("tex/bad").unwrap();
        residency.want_texture(&handle);
        assets.pump_until_idle();
        assert_eq!(handle.state(), LoadState::Failed);

        let progress = residency.pump(&mut rhi, &assets, WHOLE).unwrap();
        assert!(
            progress.idle(),
            "a failure is not something to keep waiting on"
        );
        assert_eq!(progress.finished, 0);
        assert!(residency.texture(handle.id()).is_none());

        residency.destroy(&mut rhi).unwrap();
        let report = rhi.shutdown();
        assert!(report.clean(), "unclean: {report:?}");
    }
}
