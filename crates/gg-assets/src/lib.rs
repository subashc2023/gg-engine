//! `gg-assets` (§4.6, §3) — the `gg-pack` format, its reader and its writer.
//!
//! The runtime never parses a source asset and never parses JSON (§2, Asset
//! interchange row). `ggc` compiles glTF and images offline; what ships is a
//! pack, and what this crate does is map one and hand out borrowed slices of
//! it. [`pack`] *is* the format's specification — the layout is what the
//! commented `repr(C)` structs there say it is (§6 M9 exit).
//!
//! The writer lives here too, in [`write`], for the reason a format usually
//! ends up with a dialect: two definitions of one layout drift, and the round
//! trip is only a proof if both halves read the same struct.
//!
//! ```
//! use gg_assets::{AssetKind, Pack, PackWriter};
//!
//! let mut writer = PackWriter::new();
//! writer.add("meshes/cube", AssetKind::Mesh, 0, vec![1, 2, 3])?;
//! let bytes = writer.finish()?;
//!
//! let pack = Pack::from_bytes(&bytes)?;
//! let cube = pack.find_by_name("meshes/cube").ok_or("no cube")?;
//! assert_eq!(pack.blob(cube), &[1, 2, 3]);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(missing_docs)]

pub mod load;
pub mod material;
pub mod mesh;
pub mod pack;
pub mod scene;
pub mod texture;
pub mod write;

pub use load::{AssetError, Assets, Handle, LoadState, TextureData, asset};
pub use material::Material;
pub use mesh::{Mesh, MeshError, MeshHeader, Vertex};
pub use pack::{
    AssetId, AssetKind, Entry, FORMAT_VERSION, Header, MAGIC, Pack, PackError, SECTION_ALIGN,
};
pub use scene::{Node, Scene, SceneHeader};
pub use texture::{Texture, TextureError, TextureFormat};
pub use write::{PackWriter, WriteError};
