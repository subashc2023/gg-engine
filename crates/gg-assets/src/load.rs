//! The runtime half of §4.6: typed ref-counted [`Handle`]s, [`LoadState`], and
//! the worker pool that turns a pack entry into something a consumer can use.
//!
//! # Only textures are loaded
//!
//! The pack exists to be cast in place (see [`crate::pack`]), so a mesh, a
//! material and a scene are *already* loaded the moment the file is mapped —
//! [`Assets::mesh`] hands back a view into the mapping and copies nothing. A
//! texture level is a zstd frame and is the one thing that has to be produced
//! rather than pointed at, which is why it is the only kind that reaches a
//! worker thread and the only kind [`Assets`] holds bytes for.
//!
//! That asymmetry is the format's, not this module's. Hiding it behind a
//! uniform "everything loads" API would mean copying meshes out of the mapping
//! to make them look like textures, which is the cost mmap was chosen to avoid.
//!
//! # Threads and determinism
//!
//! Decompression runs on a small pool. This is downstream of the simulation and
//! of the hash (§4.1): which worker finishes first changes *when* an asset
//! becomes ready and nothing else. Nothing here feeds a tick, and the order
//! assets arrive in is never observable to game code — [`Assets::pump`] applies
//! them in id order, so even the arrival *sequence* is a function of content.

use std::path::Path;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::{collections::BTreeMap, marker::PhantomData, thread};

use crate::pack::{AssetId, AssetKind, Entry, Pack, PackError};
use crate::texture::{Texture, TextureError, TextureFormat};

/// Type tags for [`Handle`]. They hold nothing: what a handle points at lives
/// in [`Assets`], because a borrowed view of a mapping cannot outlive the
/// borrow that produced it — and a handle outlives every frame.
pub mod asset {
    /// A [`crate::Mesh`]: vertices and indices, read in place.
    pub struct Mesh;
    /// A [`crate::Texture`]: a KTX2 container whose levels are decompressed.
    pub struct Texture;
    /// A [`crate::Material`]: factors and the ids of its four maps.
    pub struct Material;
    /// A [`crate::Scene`]: the node list a demo instantiates.
    pub struct Scene;
}

/// What a handle's type parameter promises about the entry behind it. The
/// pack's `kind` is checked once, at [`Assets::load`], so a `Handle<Mesh>` that
/// exists at all names a mesh.
pub trait Asset: 'static {
    /// The entry kind this tag accepts.
    const KIND: AssetKind;
}

impl Asset for asset::Mesh {
    const KIND: AssetKind = AssetKind::Mesh;
}
impl Asset for asset::Texture {
    const KIND: AssetKind = AssetKind::Texture;
}
impl Asset for asset::Material {
    const KIND: AssetKind = AssetKind::Material;
}
impl Asset for asset::Scene {
    const KIND: AssetKind = AssetKind::Scene;
}

/// Where an asset stands (§4.6). Read off a [`Handle`] with no registry in
/// hand, which is what lets a system holding only handles decide whether it can
/// draw yet.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum LoadState {
    /// Named but not asked for. Only reachable through [`Assets::unload`]:
    /// a handle handed out by `load` is at least `Loading`.
    Unloaded = 0,
    /// A worker has it, or is about to.
    Loading = 1,
    /// Usable now.
    Ready = 2,
    /// It will not become usable. The reason was logged where it was found;
    /// this is the state a consumer branches on.
    Failed = 3,
}

impl LoadState {
    fn from_u8(raw: u8) -> Self {
        match raw {
            1 => Self::Loading,
            2 => Self::Ready,
            3 => Self::Failed,
            _ => Self::Unloaded,
        }
    }
}

/// The shared cell a [`Handle`] and [`Assets`] both point at. One allocation
/// per asset, and its strong count *is* the reference count §4.6 asks for.
struct Slot {
    id: AssetId,
    state: AtomicU8,
}

/// A typed, ref-counted reference to one asset.
///
/// Cloning is a reference count bump and dropping is a decrement; when the last
/// one outside [`Assets`] goes, the asset becomes evictable (see
/// [`Assets::evict_unreferenced`]). Nothing is freed at the drop itself — an
/// asset the GPU is still reading must outlive the frame that named it, and a
/// `Drop` cannot know which frame that was.
pub struct Handle<T: Asset> {
    slot: Arc<Slot>,
    tag: PhantomData<fn() -> T>,
}

impl<T: Asset> Handle<T> {
    /// The asset's identity — what a blob's cross-references store.
    #[must_use]
    pub fn id(&self) -> AssetId {
        self.slot.id
    }

    /// Where this asset stands, without consulting the registry.
    #[must_use]
    pub fn state(&self) -> LoadState {
        LoadState::from_u8(self.slot.state.load(Ordering::Acquire))
    }

    /// Whether the asset is usable now.
    #[must_use]
    pub fn ready(&self) -> bool {
        self.state() == LoadState::Ready
    }
}

impl<T: Asset> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            slot: Arc::clone(&self.slot),
            tag: PhantomData,
        }
    }
}

impl<T: Asset> core::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Handle({}, {:?})", self.slot.id, self.state())
    }
}

impl<T: Asset> PartialEq for Handle<T> {
    /// Identity, not pointer equality: two handles to one asset are equal even
    /// if they were loaded through different names for it.
    fn eq(&self, other: &Self) -> bool {
        self.slot.id == other.slot.id
    }
}

impl<T: Asset> Eq for Handle<T> {}

/// A decompressed texture, level 0 first. Owned rather than borrowed: this is
/// the one asset the pack cannot hand over in place (see the module docs).
pub struct TextureData {
    /// The block format `ggc` chose from the material slot (§4.6).
    pub format: TextureFormat,
    /// Level 0's extent in texels.
    pub extent: (u32, u32),
    /// Every level, largest first, each tightly packed for `format`.
    pub levels: Vec<Vec<u8>>,
}

impl TextureData {
    /// Total decompressed bytes — what a residency budget charges against.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.levels.iter().map(Vec::len).sum()
    }
}

/// Why an asset could not be produced.
#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    /// The pack itself would not open.
    #[error(transparent)]
    Pack(#[from] PackError),
    /// No entry of that name.
    #[error("no asset named `{name}` in the pack")]
    NotFound {
        /// The name that was looked up.
        name: String,
    },
    /// The entry exists but is not what the handle's type says.
    #[error("`{name}` is a {found:?}, and the handle asks for a {wanted:?}")]
    WrongKind {
        /// The name that was looked up.
        name: String,
        /// What the pack holds.
        found: AssetKind,
        /// What the caller's type parameter promised.
        wanted: AssetKind,
    },
    /// A texture blob would not decode.
    #[error("`{name}`: {source}")]
    Texture {
        /// The asset's name.
        name: String,
        /// What the container reader said.
        #[source]
        source: TextureError,
    },
}

/// One worker's job: decompress every level of a texture entry.
struct Request {
    id: AssetId,
    entry: Entry,
}

/// What came back. `Err` carries the id so the slot can be marked failed.
struct Finished {
    id: AssetId,
    result: Result<TextureData, TextureError>,
}

/// A pack, the handles into it, and the pool that decompresses its textures.
///
/// One pack per `Assets`. A game with several loads several, because the entry
/// table is a binary search over *one* file's ids and merging two packs would
/// mean rebuilding it — which is `ggc`'s job, offline, where a duplicate name
/// is an error rather than a silent shadow.
pub struct Assets {
    pack: Arc<Pack>,
    slots: BTreeMap<AssetId, Arc<Slot>>,
    textures: BTreeMap<AssetId, TextureData>,
    /// Names, kept for error messages only — the pack has them, but a failed
    /// entry's name is wanted after the borrow that would produce it is gone.
    names: BTreeMap<AssetId, String>,
    requests: Option<mpsc::Sender<Request>>,
    finished: mpsc::Receiver<Finished>,
    workers: Vec<thread::JoinHandle<()>>,
    /// Textures still with a worker. `pump` is done when this reaches zero,
    /// which is what [`Assets::pump_until_idle`] waits on.
    in_flight: usize,
}

impl core::fmt::Debug for Assets {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Assets")
            .field("pack", &self.pack)
            .field("handles", &self.slots.len())
            .field("resident_textures", &self.textures.len())
            .field("in_flight", &self.in_flight)
            .finish()
    }
}

/// Decompression threads. Small: zstd is fast and the ceiling here is the
/// staging ring's throughput, not the CPU's (§4.3).
fn worker_count() -> usize {
    thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1).clamp(1, 4))
        .unwrap_or(1)
}

impl Assets {
    /// Map `path` and stand up the worker pool.
    ///
    /// # Errors
    /// Whatever [`Pack::open`] refuses.
    pub fn open(path: &Path) -> Result<Self, AssetError> {
        Ok(Self::new(Pack::open(path)?))
    }

    /// Take an already-open pack — for tests, and for a pack that arrived as
    /// bytes rather than as a file.
    #[must_use]
    pub fn new(pack: Pack) -> Self {
        let pack = Arc::new(pack);
        let (requests, inbox) = mpsc::channel::<Request>();
        let (outbox, finished) = mpsc::channel::<Finished>();
        // One queue behind a mutex rather than a queue per worker: the jobs
        // differ in cost by orders of magnitude (a 1x1 mip against a 4K base),
        // so handing them out round-robin would idle three threads behind one.
        let inbox = Arc::new(Mutex::new(inbox));

        let workers = (0..worker_count())
            .map(|index| {
                let inbox = Arc::clone(&inbox);
                let outbox = outbox.clone();
                let pack = Arc::clone(&pack);
                thread::Builder::new()
                    .name(format!("gg-assets-{index}"))
                    .spawn(move || decode_loop(&inbox, &outbox, &pack))
            })
            .filter_map(Result::ok)
            .collect();

        Self {
            pack,
            slots: BTreeMap::new(),
            textures: BTreeMap::new(),
            names: BTreeMap::new(),
            requests: Some(requests),
            finished,
            workers,
            in_flight: 0,
        }
    }

    /// The pack behind these handles — for the entry table, and for the
    /// in-place reads [`Self::mesh`] and friends are built on.
    #[must_use]
    pub fn pack(&self) -> &Pack {
        &self.pack
    }

    /// How many assets have a handle out.
    #[must_use]
    pub fn tracked(&self) -> usize {
        self.slots.len()
    }

    /// Resident texture bytes — what a memory budget reads (§4.8).
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.textures.values().map(TextureData::bytes).sum()
    }

    /// Take a reference to `name`, starting the work if it is not started.
    ///
    /// Returns immediately: a texture is `Loading` until a [`Self::pump`] picks
    /// it up, everything else is `Ready` here, because the mapping already is.
    ///
    /// # Errors
    /// No such name, or an entry whose kind is not `T`'s.
    pub fn load<T: Asset>(&mut self, name: &str) -> Result<Handle<T>, AssetError> {
        let entry = *self
            .pack
            .find_by_name(name)
            .ok_or_else(|| AssetError::NotFound {
                name: name.to_owned(),
            })?;
        // `kind()` cannot be None for an entry of an open pack; treating it as
        // a mismatch rather than unwrapping keeps that fact out of the panics.
        let found = entry.kind().unwrap_or(T::KIND);
        if found != T::KIND {
            return Err(AssetError::WrongKind {
                name: name.to_owned(),
                found,
                wanted: T::KIND,
            });
        }
        Ok(self.handle_for(entry, name))
    }

    /// As [`Self::load`], by id rather than name — what a blob's
    /// cross-references (a mesh's material, a material's maps) are followed by.
    ///
    /// [`AssetId::NONE`] is absent by construction and yields `Ok(None)` rather
    /// than an error, so an optional map needs no flag beside it.
    ///
    /// # Errors
    /// No such id, or a kind mismatch.
    pub fn load_id<T: Asset>(&mut self, id: AssetId) -> Result<Option<Handle<T>>, AssetError> {
        if id.is_none() {
            return Ok(None);
        }
        let entry = *self.pack.find(id).ok_or_else(|| AssetError::NotFound {
            name: id.to_string(),
        })?;
        let found = entry.kind().unwrap_or(T::KIND);
        if found != T::KIND {
            return Err(AssetError::WrongKind {
                name: id.to_string(),
                found,
                wanted: T::KIND,
            });
        }
        let name = self.pack.name(&entry).to_owned();
        Ok(Some(self.handle_for(entry, &name)))
    }

    /// The slot for `entry`, created and queued if this is its first reference.
    fn handle_for<T: Asset>(&mut self, entry: Entry, name: &str) -> Handle<T> {
        if let Some(slot) = self.slots.get(&entry.id) {
            return Handle {
                slot: Arc::clone(slot),
                tag: PhantomData,
            };
        }
        // Everything but a texture is usable the moment the pack is mapped.
        let queue = T::KIND == AssetKind::Texture && !self.textures.contains_key(&entry.id);
        let state = if queue {
            LoadState::Loading
        } else {
            LoadState::Ready
        };
        let slot = Arc::new(Slot {
            id: entry.id,
            state: AtomicU8::new(state as u8),
        });
        self.slots.insert(entry.id, Arc::clone(&slot));
        self.names.insert(entry.id, name.to_owned());
        if queue {
            self.submit(entry);
        }
        Handle {
            slot,
            tag: PhantomData,
        }
    }

    /// Hand a texture entry to the pool, or decode it here if the pool is gone.
    fn submit(&mut self, entry: Entry) {
        let request = Request {
            id: entry.id,
            entry,
        };
        match self.requests.as_ref().map(|tx| tx.send(request)) {
            Some(Ok(())) => self.in_flight += 1,
            // No pool, or every worker died: decode inline rather than leave
            // the handle Loading forever. Slower, never wrong.
            _ => {
                let result = decode(&self.pack, &entry);
                self.deliver(Finished {
                    id: entry.id,
                    result,
                });
            }
        }
    }

    /// Apply everything the workers have finished. Call once a frame.
    ///
    /// Returns how many assets became `Ready` or `Failed`. Applied in id order
    /// rather than arrival order, so two runs of the same load produce the same
    /// sequence of states whatever the threads did (§4.1).
    pub fn pump(&mut self) -> usize {
        let mut batch: Vec<Finished> = self.finished.try_iter().collect();
        batch.sort_unstable_by_key(|done| done.id);
        let count = batch.len();
        for done in batch {
            self.in_flight = self.in_flight.saturating_sub(1);
            self.deliver(done);
        }
        count
    }

    /// Pump until nothing is outstanding — the synchronous load path (§4.6:
    /// "synchronous load path first"). Blocks on the workers rather than
    /// spinning.
    pub fn pump_until_idle(&mut self) {
        self.pump();
        while self.in_flight > 0 {
            let Ok(done) = self.finished.recv() else {
                // Every worker is gone and nothing more can arrive; the slots
                // still Loading are failed by `submit`'s inline fallback next
                // time, and leaving them here would be a silent hang.
                self.in_flight = 0;
                break;
            };
            self.in_flight -= 1;
            self.deliver(done);
        }
    }

    fn deliver(&mut self, done: Finished) {
        let state = match done.result {
            Ok(data) => {
                self.textures.insert(done.id, data);
                LoadState::Ready
            }
            Err(source) => {
                let name = self
                    .names
                    .get(&done.id)
                    .cloned()
                    .unwrap_or_else(|| done.id.to_string());
                tracing::error!(%name, error = %source, "asset failed to load");
                LoadState::Failed
            }
        };
        if let Some(slot) = self.slots.get(&done.id) {
            slot.state.store(state as u8, Ordering::Release);
        }
    }

    /// A mesh, read in place out of the mapping.
    ///
    /// # Errors
    /// A blob whose header does not describe it — which an open pack's
    /// bounds checks do not cover, because they check the blob's *extent*
    /// and this checks its contents.
    pub fn mesh(&self, handle: &Handle<asset::Mesh>) -> Result<crate::Mesh<'_>, crate::MeshError> {
        crate::Mesh::read(self.blob(handle.id()))
    }

    /// A material, read in place.
    ///
    /// # Errors
    /// A blob of the wrong length.
    pub fn material(
        &self,
        handle: &Handle<asset::Material>,
    ) -> Result<crate::Material, crate::material::MaterialError> {
        crate::Material::read(self.blob(handle.id()))
    }

    /// A scene, read in place.
    ///
    /// # Errors
    /// A blob whose node count disagrees with its length.
    pub fn scene(
        &self,
        handle: &Handle<asset::Scene>,
    ) -> Result<crate::Scene<'_>, crate::scene::SceneError> {
        crate::Scene::read(self.blob(handle.id()))
    }

    /// A decompressed texture, or `None` while it is not [`LoadState::Ready`].
    #[must_use]
    pub fn texture(&self, handle: &Handle<asset::Texture>) -> Option<&TextureData> {
        self.textures.get(&handle.id())
    }

    /// The KTX2 container itself, undecompressed — for a tool writing a level
    /// back out, and for the residency path that wants the header without the
    /// texels.
    ///
    /// # Errors
    /// A blob that is not the KTX2 subset this engine writes.
    pub fn container(
        &self,
        handle: &Handle<asset::Texture>,
    ) -> Result<Texture<'_>, crate::TextureError> {
        Texture::read(self.blob(handle.id()))
    }

    /// The raw blob. Empty for an id with no entry, which cannot happen for a
    /// handle this registry issued.
    fn blob(&self, id: AssetId) -> &[u8] {
        self.pack
            .find(id)
            .map_or(&[], |entry| self.pack.blob(entry))
    }

    /// Drop the decompressed bytes of a texture nobody holds a handle to, and
    /// forget its slot. Returns how many were released.
    ///
    /// Explicit rather than automatic on the last `Drop`: an asset the GPU is
    /// still reading has to outlive the frame that named it, and only the
    /// caller knows where that boundary is (§4.3's deletion queue makes the
    /// same argument about images).
    pub fn evict_unreferenced(&mut self) -> usize {
        // Strong count 1 means this map holds the only reference.
        let dead: Vec<AssetId> = self
            .slots
            .iter()
            .filter(|(_, slot)| Arc::strong_count(slot) == 1)
            .map(|(id, _)| *id)
            .collect();
        for id in &dead {
            self.slots.remove(id);
            self.names.remove(id);
            self.textures.remove(id);
        }
        dead.len()
    }

    /// Forget one asset's payload and reset its slot to
    /// [`LoadState::Unloaded`], keeping every handle to it alive.
    ///
    /// This is the asset half of hot reload (§4.6 watch mode): a rebuilt pack
    /// is a new [`Assets`], but a *re-request* of the same handle inside one is
    /// what a residency layer needs to re-upload without the game noticing its
    /// handles changed.
    pub fn unload(&mut self, id: AssetId) {
        self.textures.remove(&id);
        if let Some(slot) = self.slots.get(&id) {
            slot.state
                .store(LoadState::Unloaded as u8, Ordering::Release);
        }
    }

    /// Re-queue everything currently `Unloaded` that still has a handle out.
    /// Returns how many were re-queued.
    pub fn reload_unloaded(&mut self) -> usize {
        let stale: Vec<AssetId> = self
            .slots
            .iter()
            .filter(|(_, slot)| {
                LoadState::from_u8(slot.state.load(Ordering::Acquire)) == LoadState::Unloaded
            })
            .map(|(id, _)| *id)
            .collect();
        for id in &stale {
            let Some(entry) = self.pack.find(*id).copied() else {
                continue;
            };
            if let Some(slot) = self.slots.get(id) {
                slot.state
                    .store(LoadState::Loading as u8, Ordering::Release);
            }
            self.submit(entry);
        }
        stale.len()
    }
}

impl Drop for Assets {
    fn drop(&mut self) {
        // Dropping the sender is what the workers see as "no more work"; the
        // join is what keeps a thread from reading the pack's mapping after
        // this Arc's last strong reference goes with it.
        self.requests = None;
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn decode_loop(
    inbox: &Mutex<mpsc::Receiver<Request>>,
    outbox: &mpsc::Sender<Finished>,
    pack: &Pack,
) {
    loop {
        // The lock is held across `recv`, so a worker blocks holding it and
        // wakes exactly one of the others when the queue is empty. Cheap: the
        // contention is a channel op against a decompression.
        let request = match inbox.lock() {
            Ok(inbox) => inbox.recv(),
            Err(_) => return,
        };
        let Ok(request) = request else { return };
        let result = decode(pack, &request.entry);
        if outbox
            .send(Finished {
                id: request.id,
                result,
            })
            .is_err()
        {
            return;
        }
    }
}

fn decode(pack: &Pack, entry: &Entry) -> Result<TextureData, TextureError> {
    let blob = pack.blob(entry);
    let container = Texture::read(blob)?;
    let levels = container.levels().collect::<Result<Vec<_>, _>>()?;
    Ok(TextureData {
        format: container.format,
        extent: (container.width, container.height),
        levels,
    })
}
