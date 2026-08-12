//! The environment blob: an order-2 SH projection and the radiance chain it was
//! projected from, in one 160-byte record (§6 M27).
//!
//! **Two halves of one measurement, so they are one asset.** A diffuse surface
//! gathers a hemisphere and reads the nine coefficients; a specular one gathers
//! a lobe and reads a mip of [`Environment::radiance`]. Both are integrals of
//! the *same* panorama, and a pack that let them be named separately would let
//! a scene pair yesterday's sky with today's coefficients — a mismatch nothing
//! downstream could detect, because each half is individually plausible.
//!
//! The SH lives here rather than beside the texels because it is not texels:
//! nine `float3`s do not survive a block format, and asking the runtime to
//! project them from a BC6H image would mean decompressing one on the CPU at
//! load to recover numbers `ggc` already had in float.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;

use crate::pack::AssetId;

/// Coefficients in an order-2 real SH projection: bands 0, 1 and 2 are 1 + 3 +
/// 5. **Must equal `gg_render::sky::SH_COEFFICIENTS` and `pbr.slang`'s.**
pub const SH_COEFFICIENTS: usize = 9;

/// One compiled environment.
///
/// `[f32; 4]` per coefficient rather than `[f32; 3]`: the shader reads these out
/// of a device address at a 16-byte stride like every other `float3` in the
/// frame block, and a packed 12-byte stride would be the one array in this
/// engine whose element size disagreed with its alignment.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Environment {
    /// The order-2 projection of the source panorama, in linear radiance. `w`
    /// is unread padding.
    pub sh: [[f32; 4]; SH_COEFFICIENTS],
    /// The prefiltered radiance chain: an octahedral BC6H texture whose mip
    /// level *is* the roughness axis (§4.6). [`AssetId::NONE`] for an
    /// environment that is diffuse-only — legal, and what a sky too dim to
    /// reflect anything compiles to.
    pub radiance: AssetId,
    /// Levels in that chain. Zero when there is no chain; otherwise the divisor
    /// that turns a roughness into a mip, kept here so the shader never has to
    /// ask the texture how many it has.
    pub levels: u32,
    /// Padding to a 16-byte multiple, so the record's tail is not `Pod`'s
    /// implicit-padding refusal in a different spelling.
    pub _pad: u32,
}

const _: () = assert!(size_of::<Environment>() == 160);

impl Default for Environment {
    /// No environment: black in every band, no chain. What a `Sky` naming an
    /// asset the pack does not hold resolves to, which is dark rather than a
    /// failure for [`crate::Material`]'s reason.
    fn default() -> Self {
        Self {
            sh: [[0.0; 4]; SH_COEFFICIENTS],
            radiance: AssetId::NONE,
            levels: 0,
            _pad: 0,
        }
    }
}

/// Why an environment blob could not be read.
#[derive(Debug, Error)]
#[error("an environment blob of {len} bytes is not the {expected} an environment is")]
pub struct EnvironmentError {
    /// The blob's length.
    pub len: usize,
    /// [`Environment`]'s size.
    pub expected: usize,
}

impl Environment {
    /// Read an environment out of a blob. Exact length, for [`crate::Material`]'s
    /// reason: growing the record is a `FORMAT_VERSION` bump.
    pub fn read(blob: &[u8]) -> Result<Self, EnvironmentError> {
        if blob.len() != size_of::<Self>() {
            return Err(EnvironmentError {
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
