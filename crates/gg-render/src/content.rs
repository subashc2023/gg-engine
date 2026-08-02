//! The pack a renderer was handed, and everything it has asked for (§4.6).
//!
//! [`Residency`](crate::residency::Residency) answers "is it on the device" and
//! [`Assets`] answers "are the bytes here". This is the layer above both: it
//! takes the asset ids a *game* named, resolves what they depend on, and holds
//! the small per-mesh facts a draw needs — which texture, which tint — so the
//! draw loop is a lookup rather than a chain of blob reads.
//!
//! # Requesting is transitive and cheap
//!
//! A game names one scene; a scene names meshes; a mesh names a material; a
//! material names four textures. Every link but the last is a blob already in
//! the mapping (§4.6), so following all of them costs a `BTreeMap` insert and
//! no I/O — which is why [`Content::request`] resolves the whole chain at once
//! instead of discovering it a frame at a time.
//!
//! # A missing asset is not an error
//!
//! `ggc watch` rewrites the pack under a running game, and an id that resolves
//! to nothing is the normal state for the moment between a save and a rebuild.
//! Requests for unknown ids are dropped, malformed blobs are logged once, and
//! the frame draws what it has. The alternative is a renderer that treats an
//! artist pressing Ctrl-S as a fatal error.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use gg_assets::load::{Assets, asset};
use gg_assets::pack::{AssetId, AssetKind};
use gg_assets::{AssetError, Handle};
use gg_extract::{Placement, Scenes};
use gg_math::sim;
use gg_rhi::RhiError;

use crate::GpuHost;
use crate::residency::{Progress, Residency, ResidencyError, ResidentMesh, ResidentTexture};

/// What one mesh needs at draw time beyond its buffers, resolved when the mesh
/// is requested rather than per frame.
#[derive(Clone, Copy, Debug)]
pub struct DrawMaterial {
    /// Linear base colour, already through the sRGB decode the game's tint is.
    pub base_color: [f32; 4],
    /// The base-colour map, or [`AssetId::NONE`] for a flat material.
    pub base_color_texture: AssetId,
}

impl Default for DrawMaterial {
    /// What a mesh naming no material draws as: white, untextured. Visible and
    /// obviously plain, rather than invisible.
    fn default() -> Self {
        DrawMaterial {
            base_color: [1.0; 4],
            base_color_texture: AssetId::NONE,
        }
    }
}

/// Why a pack could not be opened. Everything *after* opening is recoverable
/// and logged rather than returned — see the module docs.
#[derive(Debug, thiserror::Error)]
pub enum ContentError {
    /// The file would not open, or is not a pack.
    #[error(transparent)]
    Asset(#[from] AssetError),
    /// Making an asset resident failed at the RHI.
    #[error(transparent)]
    Residency(#[from] ResidencyError),
}

/// Identity of the file on disk, for the watch (§4.6). Length as well as time
/// because a rebuild that lands in the same clock tick still changes size, and
/// filesystem timestamp granularity is coarser than a fast incremental build.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Stamp {
    modified: Option<SystemTime>,
    len: u64,
}

impl Stamp {
    fn of(path: &Path) -> Option<Self> {
        let meta = std::fs::metadata(path).ok()?;
        Some(Stamp {
            modified: meta.modified().ok(),
            len: meta.len(),
        })
    }
}

/// A pack, what has been asked of it, and what is on the device.
pub struct Content {
    assets: Assets,
    residency: Residency,
    /// Per-mesh draw facts, keyed by mesh id.
    materials: BTreeMap<AssetId, DrawMaterial>,
    /// Scene handles, so [`Scenes::expand`] reads without loading — that method
    /// takes `&self` and expansion must never be what triggers a load.
    scenes: BTreeMap<AssetId, Handle<asset::Scene>>,
    /// Every id a game has named, so a reload can ask for them all again.
    wanted: BTreeMap<AssetId, ()>,
    path: PathBuf,
    stamp: Option<Stamp>,
    opened: Instant,
    ready_at: Option<Duration>,
}

impl core::fmt::Debug for Content {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Content")
            .field("path", &self.path)
            .field("wanted", &self.wanted.len())
            .field("residency", &self.residency)
            .finish()
    }
}

impl Content {
    /// Map `path` and start the load-to-first-frame clock (§6 M9's exit row).
    ///
    /// # Errors
    /// A file that will not open, or a pack the loader refuses by name.
    pub fn open(path: &Path) -> Result<Self, ContentError> {
        Ok(Content {
            assets: Assets::open(path)?,
            residency: Residency::new(),
            materials: BTreeMap::new(),
            scenes: BTreeMap::new(),
            wanted: BTreeMap::new(),
            path: path.to_path_buf(),
            stamp: Stamp::of(path),
            opened: Instant::now(),
            ready_at: None,
        })
    }

    /// Ask for `id` and everything it names. Idempotent, and silent about ids
    /// the pack does not contain — see the module docs.
    pub fn request(&mut self, id: AssetId) {
        if id.is_none() || self.wanted.insert(id, ()).is_some() {
            return;
        }
        let Some(kind) = self.assets.pack().find(id).and_then(|entry| entry.kind()) else {
            return;
        };
        match kind {
            AssetKind::Scene => self.request_scene(id),
            AssetKind::Mesh => self.request_mesh(id),
            // A game naming a material or a texture directly asks for nothing:
            // both are reached *through* a mesh, and a texture with no mesh
            // behind it has no draw to be sampled by.
            AssetKind::Material | AssetKind::Texture => {}
        }
    }

    fn request_scene(&mut self, id: AssetId) {
        let handle = match self.assets.load_id::<asset::Scene>(id) {
            Ok(Some(handle)) => handle,
            Ok(None) => return,
            Err(error) => return warn(id, &error),
        };
        // Collected before recursing: `nodes()` borrows the mapping through
        // `self.assets`, and the requests below take `&mut self`.
        let meshes: Vec<AssetId> = match self.assets.scene(&handle) {
            Ok(scene) => scene.nodes().iter().map(|node| node.mesh).collect(),
            Err(error) => return warn(id, &error),
        };
        self.scenes.insert(id, handle);
        for mesh in meshes {
            self.request(mesh);
        }
    }

    fn request_mesh(&mut self, id: AssetId) {
        let handle = match self.assets.load_id::<asset::Mesh>(id) {
            Ok(Some(handle)) => handle,
            Ok(None) => return,
            Err(error) => return warn(id, &error),
        };
        let material_id = match self.assets.mesh(&handle) {
            Ok(mesh) => mesh.header.material,
            Err(error) => return warn(id, &error),
        };
        self.residency.want_mesh(&handle);
        let material = self.resolve_material(material_id);
        self.materials.insert(id, material);
        self.request_texture(material.base_color_texture);
    }

    /// A material's draw facts, or the default for one that is absent or
    /// unreadable. The three maps it also names are deliberately unread: they
    /// are lighting inputs and M11 owns the pass that reads them.
    fn resolve_material(&mut self, id: AssetId) -> DrawMaterial {
        let handle = match self.assets.load_id::<asset::Material>(id) {
            Ok(Some(handle)) => handle,
            Ok(None) => return DrawMaterial::default(),
            Err(error) => {
                warn(id, &error);
                return DrawMaterial::default();
            }
        };
        match self.assets.material(&handle) {
            // `base_color` is already linear in the pack (§4.6), so nothing is
            // decoded here — unlike the game's `0x00RRGGBB` tint, which is.
            Ok(material) => DrawMaterial {
                base_color: material.base_color,
                base_color_texture: material.base_color_texture,
            },
            Err(error) => {
                warn(id, &error);
                DrawMaterial::default()
            }
        }
    }

    fn request_texture(&mut self, id: AssetId) {
        match self.assets.load_id::<asset::Texture>(id) {
            Ok(Some(handle)) => self.residency.want_texture(&handle),
            Ok(None) => {}
            Err(error) => warn(id, &error),
        }
    }

    /// Decompress what the workers finished and move up to `budget` bytes onto
    /// the device. Call once a frame.
    ///
    /// # Errors
    /// Whatever the *device* refuses. A blob that will not read is content and
    /// is logged instead: it has already left the queue, so the next pump
    /// carries on without it and the frame draws what it has.
    pub fn pump(&mut self, rhi: &mut impl GpuHost, budget: usize) -> Result<Progress, RhiError> {
        self.assets.pump();
        let progress = match self.residency.pump(rhi, &self.assets, budget) {
            Ok(progress) => progress,
            Err(ResidencyError::Rhi(refused)) => return Err(refused),
            Err(content) => {
                tracing::warn!(error = %content, "asset skipped");
                Progress::default()
            }
        };
        // Not "nothing pending" alone: that is also true before a game has
        // named anything, and a clock that started at zero would report the
        // load time of an empty pack.
        if self.ready_at.is_none() && !self.wanted.is_empty() && progress.idle() {
            let elapsed = self.opened.elapsed();
            self.ready_at = Some(elapsed);
            // Reported here rather than by the host: the clock belongs to
            // whatever owns the file, and a shell that had to remember whether
            // it had already logged this would be keeping renderer state (§3).
            tracing::info!(
                pack = %self.path.display(),
                load_ms = elapsed.as_millis(),
                assets = self.wanted.len(),
                "pack resident"
            );
        }
        Ok(progress)
    }

    /// How long from [`Content::open`] to every requested asset being on the
    /// device — §6 M9's "load to first frame < 500 ms". `None` until then.
    #[must_use]
    pub fn ready_at(&self) -> Option<Duration> {
        self.ready_at
    }

    /// A resident mesh, or `None` while it is on its way.
    #[must_use]
    pub fn mesh(&self, id: AssetId) -> Option<&ResidentMesh> {
        self.residency.mesh(id)
    }

    /// A resident texture, or `None` while it is on its way.
    #[must_use]
    pub fn texture(&self, id: AssetId) -> Option<&ResidentTexture> {
        self.residency.texture(id)
    }

    /// One mesh's draw facts. Defaults for a mesh nothing has requested, which
    /// is what a draw for an id from another pack would ask about.
    #[must_use]
    pub fn material(&self, mesh: AssetId) -> DrawMaterial {
        self.materials.get(&mesh).copied().unwrap_or_default()
    }

    /// Assets still on their way to the device.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.residency.pending()
    }

    /// Every scene in the pack, in id order. What a harness timing a whole
    /// level asks for (§6 M9's exit row) — a game asks for the scenes it draws
    /// and no more.
    #[must_use]
    pub fn scene_ids(&self) -> Vec<u64> {
        self.assets
            .pack()
            .entries()
            .iter()
            .filter(|entry| entry.kind() == Some(AssetKind::Scene))
            .map(|entry| entry.id.0)
            .collect()
    }

    /// The pack file this was opened from.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Re-open the pack if `ggc watch` has rewritten it, re-uploading
    /// everything the game had asked for (§4.6). `true` if it did.
    ///
    /// The comparison is the file's identity rather than its contents: a pack
    /// is written to a temporary and renamed over, so a changed stamp means a
    /// *finished* build and there is no half-written file to catch. That is why
    /// there is no settle rule here and one in `ggc watch` — the two ends of one
    /// rename need opposite treatment.
    ///
    /// # Errors
    /// A rewritten pack the loader refuses, or an RHI that will not release the
    /// old resources. A refused pack leaves the old one mapped and playing.
    pub fn poll_reload(&mut self, rhi: &mut impl GpuHost) -> Result<bool, ContentError> {
        let stamp = Stamp::of(&self.path);
        if stamp.is_none() || stamp == self.stamp {
            return Ok(false);
        }
        // Read before anything is torn down: a rebuild that produced a broken
        // file must leave the running game exactly as it was.
        let assets = Assets::open(&self.path)?;
        let previously = core::mem::take(&mut self.wanted);
        self.residency.destroy(rhi)?;
        self.assets = assets;
        self.stamp = stamp;
        self.materials.clear();
        self.scenes.clear();
        // The ids a game named, asked for again. Names are stable across a
        // rebuild and ids are hashes of names, so this is the same set — an
        // asset the artist deleted simply resolves to nothing now.
        for id in previously.into_keys() {
            self.request(id);
        }
        // Deliberately not reset: the clock measures the first load, and a
        // watch-mode reload is not that. `ready_at` staying put is what keeps
        // the number an artist's save cannot move.
        tracing::info!(pack = %self.path.display(), "pack rebuilt — re-uploading");
        Ok(true)
    }

    /// Release every device resource. Before the RHI's own accounting (§4.3).
    ///
    /// # Errors
    /// A handle the RHI does not recognise, which cannot happen for one issued
    /// through here.
    pub fn destroy(&mut self, rhi: &mut impl GpuHost) -> Result<(), ResidencyError> {
        self.residency.destroy(rhi)
    }
}

/// One line per asset that would not read. `warn` rather than an error return:
/// a pack is content and a frame is not the place to fail over it.
fn warn(id: AssetId, error: &dyn std::error::Error) {
    tracing::warn!(asset = %id, error = %error, "asset skipped");
}

impl Scenes for Content {
    fn expand(&self, asset: u64, visit: &mut dyn FnMut(Placement)) {
        let Some(handle) = self.scenes.get(&AssetId(asset)) else {
            return;
        };
        let Ok(scene) = self.assets.scene(handle) else {
            return;
        };
        for node in scene.nodes() {
            visit(Placement {
                mesh: node.mesh.0,
                translation: sim::DVec3::new(
                    node.translation[0],
                    node.translation[1],
                    node.translation[2],
                ),
                rotation: sim::DQuat::from_xyzw(
                    f64::from(node.rotation[0]),
                    f64::from(node.rotation[1]),
                    f64::from(node.rotation[2]),
                    f64::from(node.rotation[3]),
                ),
                scale: sim::Vec3::new(node.scale[0], node.scale[1], node.scale[2]),
            });
        }
    }
}
