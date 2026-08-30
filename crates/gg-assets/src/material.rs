//! The material blob: one 64-byte record, no arrays, no strings.
//!
//! Metallic-roughness only — glTF's own model, so the importer translates
//! nothing and the runtime inherits no second convention to convert between.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;

use crate::pack::AssetId;

/// Bits in [`Material::flags`].
pub mod flags {
    /// Alpha is a mask: a fragment below `alpha_cutoff` is discarded. Without
    /// it a material is opaque and `alpha_cutoff` is not read. The pass that can
    /// sort exists since §6 M92 — the primitive path's `forward-transparent` —
    /// but no `BLEND` flag joins this module until an asset in the tree declares
    /// the mode: import surface over an empty population is what §6 M87 refuses,
    /// and the mesh path's half (a per-batch blended fork, a sorted bucket in
    /// `scene.rs`'s `sort_key`) is priced there, not here.
    pub const ALPHA_MASK: u32 = 1 << 0;
    /// Draw both faces. Sponza's foliage needs it; most of Sponza does not,
    /// and it is per-material because that is where glTF puts it.
    pub const DOUBLE_SIDED: u32 = 1 << 1;
}

/// A material. Textures are [`AssetId`]s into the same pack, or
/// [`AssetId::NONE`] — which is why that id is reserved.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Material {
    /// Linear RGBA, multiplied with the base colour texture.
    pub base_color: [f32; 4],
    /// Metalness factor, multiplied with the green channel of
    /// [`Self::metallic_roughness_texture`].
    pub metallic: f32,
    /// Roughness factor, multiplied with that texture's red channel.
    ///
    /// glTF puts roughness in green and metalness in blue; BC5 has two channels
    /// and a sampler returns them as red and green, so the importer repacks the
    /// pair once rather than leaving every shader to remember which convention
    /// it is reading.
    pub roughness: f32,
    /// Discard threshold, read only under [`flags::ALPHA_MASK`].
    pub alpha_cutoff: f32,
    /// [`flags`].
    pub flags: u32,
    /// sRGB colour, BC7.
    pub base_color_texture: AssetId,
    /// Tangent-space normals, BC5 — two channels, z reconstructed.
    pub normal_texture: AssetId,
    /// Roughness in red, metalness in green, BC5 linear — repacked from glTF's
    /// green and blue at import.
    pub metallic_roughness_texture: AssetId,
    /// Ambient occlusion in red, BC4 linear.
    pub occlusion_texture: AssetId,
}

const _: () = assert!(size_of::<Material>() == 64);

impl Default for Material {
    /// glTF's own defaults, so a material that declares nothing compiles to the
    /// same thing a glTF reader would have produced.
    fn default() -> Self {
        Self {
            base_color: [1.0; 4],
            metallic: 1.0,
            roughness: 1.0,
            alpha_cutoff: 0.5,
            flags: 0,
            base_color_texture: AssetId::NONE,
            normal_texture: AssetId::NONE,
            metallic_roughness_texture: AssetId::NONE,
            occlusion_texture: AssetId::NONE,
        }
    }
}

/// Why a material blob could not be read.
#[derive(Debug, Error)]
#[error("a material blob of {len} bytes is not the {expected} a material is")]
pub struct MaterialError {
    /// The blob's length.
    pub len: usize,
    /// [`Material`]'s size.
    pub expected: usize,
}

impl Material {
    /// Read a material out of a blob. Exact length: a material is one record,
    /// so a longer blob is a kind confusion rather than a forward-compatible
    /// pack — growing the record is a `FORMAT_VERSION` bump.
    pub fn read(blob: &[u8]) -> Result<Self, MaterialError> {
        if blob.len() != size_of::<Self>() {
            return Err(MaterialError {
                len: blob.len(),
                expected: size_of::<Self>(),
            });
        }
        Ok(bytemuck::pod_read_unaligned(blob))
    }

    /// The blob's bytes, for [`crate::PackWriter::add`].
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        bytemuck::bytes_of(self).to_vec()
    }
}
