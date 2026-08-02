//! The mesh blob: one header, then vertices, then indices.
//!
//! **The vertex format is the engine's, not the file's** (§4.6). A pack cannot
//! describe a layout — there is one, it is [`Vertex`], and changing it bumps
//! [`crate::FORMAT_VERSION`] so an old pack is refused by number instead of
//! read with the fields transposed. That is the whole reason the runtime does
//! no format negotiation: `ggc` already did it.
//!
//! Geometry is *pulled* (§4.3): the shader takes the vertex buffer's device
//! address as a `uint64_t` and indexes it by scalar, so [`Vertex`]'s job is to
//! be a stride both sides agree on. It is `repr(C)` and all `f32` for exactly
//! that reason — no `float3`, whose alignment would be Slang's to choose.

use core::mem::size_of;

use bytemuck::{Pod, Zeroable};
use thiserror::Error;

use crate::pack::{AssetId, SECTION_ALIGN};

/// The engine's one vertex format: 48 bytes, position, normal, uv, tangent.
///
/// The tangent arrived at M11 with normal mapping, which is what the v1 note
/// said would bring it — and it arrived as a [`crate::FORMAT_VERSION`] bump, so
/// a pack built before it is refused by number rather than read with its fields
/// transposed. `uv` kept its offset so the change is additive on the shader
/// side; the tangent went on the end.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vertex {
    /// Object-space position.
    pub position: [f32; 3],
    /// Unit normal, object space.
    pub normal: [f32; 3],
    /// Texture coordinates, glTF convention (origin top-left).
    pub uv: [f32; 2],
    /// Object-space tangent in `xyz` and **bitangent handedness** in `w`, ±1 —
    /// glTF's own `TANGENT` convention, so a document that supplies one is
    /// copied rather than reinterpreted.
    ///
    /// `w` rather than a stored bitangent: the bitangent is
    /// `cross(normal, tangent) * w`, which is four bytes instead of twelve and
    /// cannot disagree with the normal it is derived from after interpolation.
    pub tangent: [f32; 4],
}

const _: () = assert!(size_of::<Vertex>() == 48);
const _: () = assert!(core::mem::offset_of!(Vertex, uv) == 24);
const _: () = assert!(core::mem::offset_of!(Vertex, tangent) == 32);

/// A mesh blob's header. 64 bytes, so the vertices behind it start 16-aligned
/// even when a caller lays the blob out by hand.
///
/// One mesh is one glTF *primitive*: a single material and a single draw. A
/// glTF "mesh" with three primitives compiles to three assets, because the unit
/// the renderer submits is the unit the pack should name.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MeshHeader {
    /// Number of [`Vertex`] records.
    pub vertex_count: u32,
    /// Number of `u32` indices. Always `u32`: two index widths would be a
    /// branch in every consumer to save a megabyte on a scene that costs
    /// hundreds, and §4.3's `BufferKind::Index` exists once.
    pub index_count: u32,
    /// Byte offset of the vertices, from the start of the blob. 16-aligned.
    pub vertices: u32,
    /// Byte offset of the indices, from the start of the blob. 16-aligned.
    pub indices: u32,
    /// Object-space bounds, precomputed at import (§4.6) — the renderer culls
    /// against these and must never walk the vertices to find them.
    pub bounds_min: [f32; 3],
    /// The other corner.
    pub bounds_max: [f32; 3],
    /// The material this mesh draws with, or [`AssetId::NONE`].
    pub material: AssetId,
    /// Zero.
    pub reserved: [u32; 4],
}

impl MeshHeader {
    /// Radius of the sphere about the **object origin** that contains the
    /// bounds, metres.
    ///
    /// About the origin rather than about the box's centre because that is
    /// where an instance is placed: a sphere around the centre would need the
    /// centre carried alongside it through extract and the culler, to tighten a
    /// bound that is already loose. The farthest corner is therefore the one
    /// whose per-axis extent is the larger of the two faces.
    #[must_use]
    pub fn bounding_radius(&self) -> f32 {
        let mut square = 0.0;
        for axis in 0..3 {
            let extent = self.bounds_min[axis].abs().max(self.bounds_max[axis].abs());
            square += extent * extent;
        }
        square.sqrt()
    }
}

const _: () = assert!(size_of::<MeshHeader>() == 64);
const _: () = assert!((size_of::<MeshHeader>() as u64).is_multiple_of(SECTION_ALIGN));

/// Why a mesh blob could not be read. Distinct from
/// [`crate::PackError`]: the pack's own validation proves a blob is *inside the
/// file*, and this proves it is a mesh.
#[derive(Debug, Error)]
pub enum MeshError {
    /// The blob is shorter than a header.
    #[error("a mesh blob of {len} bytes is shorter than its {header}-byte header")]
    TooShort {
        /// The blob's length.
        len: usize,
        /// [`MeshHeader`]'s size.
        header: usize,
    },
    /// A run of vertices or indices leaves the blob.
    #[error("{what} spans {offset}..{end} of a {len}-byte mesh blob")]
    OutOfBounds {
        /// Which run, named.
        what: &'static str,
        /// Where it starts.
        offset: u64,
        /// Where it would end.
        end: u64,
        /// The blob's length.
        len: usize,
    },
    /// A run starts at an offset its element type cannot be cast at.
    #[error("{what} starts at {offset}, which {size}-byte records cannot be read from")]
    Misaligned {
        /// Which run, named.
        what: &'static str,
        /// Where it starts.
        offset: u32,
        /// The element's alignment requirement.
        size: usize,
    },
}

/// A mesh, borrowed out of a pack's mapping. No copy happens here — the arrays
/// are the file's bytes, cast in place.
#[derive(Clone, Copy, Debug)]
pub struct Mesh<'a> {
    /// The blob's header, read by value: 64 bytes is cheaper to copy than to
    /// keep a reference to and re-borrow.
    pub header: MeshHeader,
    vertices: &'a [Vertex],
    indices: &'a [u32],
}

impl<'a> Mesh<'a> {
    /// Read a mesh out of a blob, validating every offset it names.
    pub fn read(blob: &'a [u8]) -> Result<Self, MeshError> {
        let head = blob
            .get(..size_of::<MeshHeader>())
            .ok_or(MeshError::TooShort {
                len: blob.len(),
                header: size_of::<MeshHeader>(),
            })?;
        let header: MeshHeader = bytemuck::pod_read_unaligned(head);
        let vertices = run(blob, "the vertices", header.vertices, header.vertex_count)?;
        let indices = run(blob, "the indices", header.indices, header.index_count)?;
        Ok(Self {
            header,
            vertices,
            indices,
        })
    }

    /// The vertex array.
    #[must_use]
    pub fn vertices(&self) -> &'a [Vertex] {
        self.vertices
    }

    /// The index array. Values are not range-checked against `vertex_count`:
    /// the check is `ggc`'s, at import, where a bad index is a source problem
    /// with a name attached rather than a per-load cost on every ship.
    #[must_use]
    pub fn indices(&self) -> &'a [u32] {
        self.indices
    }
}

/// Cast `count` records of `T` at `offset` out of `blob`.
fn run<'a, T: Pod>(
    blob: &'a [u8],
    what: &'static str,
    offset: u32,
    count: u32,
) -> Result<&'a [T], MeshError> {
    if !u64::from(offset).is_multiple_of(align_of::<T>() as u64) {
        return Err(MeshError::Misaligned {
            what,
            offset,
            size: align_of::<T>(),
        });
    }
    let len = u64::from(count) * size_of::<T>() as u64;
    let end = u64::from(offset).saturating_add(len);
    let bytes = usize::try_from(end)
        .ok()
        .and_then(|end| blob.get(offset as usize..end))
        .ok_or(MeshError::OutOfBounds {
            what,
            offset: u64::from(offset),
            end,
            len: blob.len(),
        })?;
    // Cannot fail: the offset is aligned against `T` above, and a pack's blobs
    // start 16-aligned, so the base is too.
    bytemuck::try_cast_slice(bytes).map_err(|_| MeshError::Misaligned {
        what,
        offset,
        size: align_of::<T>(),
    })
}

/// Lay out a mesh blob for [`crate::PackWriter::add`].
///
/// Bounds are computed here rather than taken from the caller, because a bound
/// that disagrees with the vertices is a culling bug that shows as geometry
/// vanishing at an angle and is diagnosed nowhere near the importer.
#[must_use]
pub fn encode(vertices: &[Vertex], indices: &[u32], material: AssetId) -> Vec<u8> {
    let mut bounds_min = [f32::INFINITY; 3];
    let mut bounds_max = [f32::NEG_INFINITY; 3];
    for vertex in vertices {
        for axis in 0..3 {
            bounds_min[axis] = bounds_min[axis].min(vertex.position[axis]);
            bounds_max[axis] = bounds_max[axis].max(vertex.position[axis]);
        }
    }
    if vertices.is_empty() {
        // An empty mesh's bounds are a point at the origin, not ±infinity: the
        // latter poisons every union a scene takes with it.
        bounds_min = [0.0; 3];
        bounds_max = [0.0; 3];
    }

    let vertices_at = size_of::<MeshHeader>() as u32;
    let indices_at = vertices_at + size_of_val(vertices) as u32;
    let header = MeshHeader {
        vertex_count: vertices.len() as u32,
        index_count: indices.len() as u32,
        vertices: vertices_at,
        indices: indices_at,
        bounds_min,
        bounds_max,
        material,
        reserved: [0; 4],
    };

    let mut blob = Vec::with_capacity(indices_at as usize + size_of_val(indices));
    blob.extend_from_slice(bytemuck::bytes_of(&header));
    blob.extend_from_slice(bytemuck::cast_slice(vertices));
    blob.extend_from_slice(bytemuck::cast_slice(indices));
    blob
}
