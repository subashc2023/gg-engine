//! The scene blob: a header and a flat node list.
//!
//! **Flat, not a hierarchy.** glTF's node tree is a way of *authoring*
//! transforms; what a renderer draws is a list of meshes with world transforms
//! on them. The importer walks the tree once and multiplies it out, so the
//! runtime never does — and the transform hierarchy that M10's `gg-scene`
//! grows is about transforms that *change*, which an imported scene's do not.
//!
//! **Translation is `f64`** (§1.4, §4.2.1). A scene's node positions are
//! absolute and sim-side; narrowing them here would bake the membrane's failure
//! into the file, where no camera origin can undo it. Rotation is a unit
//! quaternion and scale is a ratio — neither has a magnitude to lose — so both
//! stay `f32`.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;

use crate::pack::AssetId;

/// A scene blob's header. 16 bytes; the nodes behind it are 8-aligned.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SceneHeader {
    /// Number of [`Node`] records.
    pub node_count: u32,
    /// Byte offset of the nodes from the start of the blob.
    pub nodes: u32,
    /// Zero.
    pub reserved: [u32; 2],
}

/// One placed mesh. 80 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Node {
    /// The mesh this node draws. Never [`AssetId::NONE`] — a node that draws
    /// nothing is a node the importer did not emit.
    pub mesh: AssetId,
    /// World-space translation, absolute (§1.4).
    pub translation: [f64; 3],
    /// World-space rotation as a unit quaternion, `xyzw`.
    pub rotation: [f32; 4],
    /// World-space scale.
    pub scale: [f32; 3],
    /// Zero.
    pub reserved: [u32; 5],
}

const _: () = assert!(size_of::<SceneHeader>() == 16);
const _: () = assert!(size_of::<Node>() == 80);

/// Why a scene blob could not be read.
#[derive(Debug, Error)]
pub enum SceneError {
    /// The blob is shorter than a header.
    #[error("a scene blob of {len} bytes is shorter than its {header}-byte header")]
    TooShort {
        /// The blob's length.
        len: usize,
        /// [`SceneHeader`]'s size.
        header: usize,
    },
    /// The node array leaves the blob, or starts where nodes cannot be read.
    #[error("{count} nodes at {offset} do not fit a {len}-byte scene blob")]
    OutOfBounds {
        /// How many nodes were claimed.
        count: u32,
        /// Where they start.
        offset: u32,
        /// The blob's length.
        len: usize,
    },
}

/// A scene, borrowed out of a pack's mapping.
#[derive(Clone, Copy, Debug)]
pub struct Scene<'a> {
    nodes: &'a [Node],
}

impl<'a> Scene<'a> {
    /// Read a scene out of a blob, validating the node array it names.
    pub fn read(blob: &'a [u8]) -> Result<Self, SceneError> {
        let head = blob
            .get(..size_of::<SceneHeader>())
            .ok_or(SceneError::TooShort {
                len: blob.len(),
                header: size_of::<SceneHeader>(),
            })?;
        let header: SceneHeader = bytemuck::pod_read_unaligned(head);
        let end = u64::from(header.nodes)
            .saturating_add(u64::from(header.node_count) * size_of::<Node>() as u64);
        let bytes = usize::try_from(end)
            .ok()
            .filter(|_| u64::from(header.nodes).is_multiple_of(align_of::<Node>() as u64))
            .and_then(|end| blob.get(header.nodes as usize..end))
            .ok_or(SceneError::OutOfBounds {
                count: header.node_count,
                offset: header.nodes,
                len: blob.len(),
            })?;
        let nodes = bytemuck::try_cast_slice(bytes).map_err(|_| SceneError::OutOfBounds {
            count: header.node_count,
            offset: header.nodes,
            len: blob.len(),
        })?;
        Ok(Self { nodes })
    }

    /// The node list, in importer order — which is glTF's document order, so a
    /// pack's draw order is a property of the source rather than of the walk.
    #[must_use]
    pub fn nodes(&self) -> &'a [Node] {
        self.nodes
    }
}

/// Lay out a scene blob for [`crate::PackWriter::add`].
#[must_use]
pub fn encode(nodes: &[Node]) -> Vec<u8> {
    let header = SceneHeader {
        node_count: nodes.len() as u32,
        nodes: size_of::<SceneHeader>() as u32,
        reserved: [0; 2],
    };
    let mut blob = Vec::with_capacity(size_of::<SceneHeader>() + size_of_val(nodes));
    blob.extend_from_slice(bytemuck::bytes_of(&header));
    blob.extend_from_slice(bytemuck::cast_slice(nodes));
    blob
}
