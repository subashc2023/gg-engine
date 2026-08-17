//! glTF 2.0 → the pack's blobs.
//!
//! **Names are derived from glTF's own indices, never from its strings.** A
//! glTF `name` is optional, frequently duplicated and frequently absent; an
//! index is stable within a document and stable across runs, which is what
//! §4.6's byte-reproducibility needs from an importer. So a primitive becomes
//! `<stem>/mesh/<mesh>.<primitive>` and nothing here consults a name.
//!
//! **One glTF primitive is one mesh asset.** A primitive is the unit with a
//! single material and a single draw, which is the unit the renderer submits.
//!
//! **One image plus one material slot is one texture asset.** glTF routinely
//! points two slots at one file — an ORM map is occlusion, roughness and
//! metalness sharing three channels of the same PNG — and the slot decides the
//! format, the channel packing and the colour space, so the pair is the unit
//! and not the image (§4.6, and [`crate::texture`] for what each role does).

use anyhow::{Context, Result, bail};
use gg_assets::material::flags;
use gg_assets::mesh::Vertex;
use gg_assets::{AssetId, AssetKind, Material, Node, mesh, scene};
use gg_math::render::{DMat4, DVec3};
use gltf::mesh::Mode;

use crate::texture::Role;

/// One compiled asset on its way into the pack.
pub struct Compiled {
    /// The pack-wide name, and therefore the id.
    pub name: String,
    /// What the blob holds.
    pub kind: AssetKind,
    /// The blob.
    pub blob: Vec<u8>,
}

/// What one glTF document compiled to.
pub struct Import {
    /// Meshes, materials and one scene, in a defined order — though the pack
    /// re-sorts by id regardless, so this order is for reporting only.
    pub assets: Vec<Compiled>,
    /// Counts for the build log.
    pub meshes: usize,
    /// Triangles across all primitives.
    pub triangles: usize,
    /// Compiled (image, role) pairs.
    pub textures: usize,
    /// Compiled clips.
    pub clips: usize,
}

/// Compile one source. `stem` is its pack-relative path with the extension
/// removed, and every asset name is built from it.
///
/// Dispatch is on the extension because the extension is what [`crate::build`]
/// walked for. An `.hdr` is not a document with meshes in it and shares nothing
/// with the glTF path below but this function's signature.
pub fn document(path: &std::path::Path, stem: &str) -> Result<Import> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("hdr") => panorama(path, stem),
        Some("wav") => clip(path, stem),
        _ => gltf_document(path, stem),
    }
}

/// Compile one `.wav` into a clip (§6 M43) — one source, one asset, named by the
/// stem alone, so a game writes `Sound::clip("sfx/pickup")` for the path it put
/// the file at. [`crate::clip`] is where the refusals live.
fn clip(path: &std::path::Path, stem: &str) -> Result<Import> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let (rate, samples) =
        crate::clip::parse(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    Ok(Import {
        assets: vec![Compiled {
            name: stem.to_owned(),
            kind: AssetKind::Clip,
            blob: gg_assets::clip::encode(rate, &samples),
        }],
        meshes: 0,
        triangles: 0,
        textures: 0,
        clips: 1,
    })
}

/// Compile one equirectangular `.hdr` into an environment and its prefiltered
/// radiance chain (§6 M27) — two assets, named off one source, because the SH
/// and the chain are two integrals of the same panorama.
fn panorama(path: &std::path::Path, stem: &str) -> Result<Import> {
    let image = image::open(path)
        .with_context(|| format!("reading {}", path.display()))?
        .into_rgb32f();
    let (width, height) = (image.width(), image.height());
    let texels: Vec<[f32; 3]> = image.pixels().map(|p| p.0).collect();
    let compiled = crate::environment::compile(&texels, width, height)
        .with_context(|| format!("prefiltering {}", path.display()))?;

    let radiance_name = format!("{stem}/radiance");
    let environment = gg_assets::Environment {
        sh: compiled.sh,
        radiance: AssetId::of(&radiance_name),
        levels: crate::environment::LEVELS,
        _pad: 0,
    };
    Ok(Import {
        assets: vec![
            Compiled {
                name: radiance_name,
                kind: AssetKind::Texture,
                blob: compiled.radiance,
            },
            // The environment is named by the *stem* alone, so a game writes
            // `Sky::image("sky/kloofendal")` — the path it put the file at, with
            // no suffix to remember. The chain hanging off it is an
            // implementation detail of that name.
            Compiled {
                name: stem.to_string(),
                kind: AssetKind::Environment,
                blob: environment.encode(),
            },
        ],
        meshes: 0,
        triangles: 0,
        textures: 1,
        clips: 0,
    })
}

fn gltf_document(path: &std::path::Path, stem: &str) -> Result<Import> {
    let gltf = gltf::Gltf::open(path).with_context(|| format!("reading {}", path.display()))?;
    let base = path.parent();
    let buffers = gltf::import_buffers(&gltf.document, base, gltf.blob.clone())
        .with_context(|| format!("loading the buffers of {}", path.display()))?;

    let mut assets = Vec::new();
    let mut triangles = 0;
    let mut meshes = 0;

    for material in gltf.document.materials() {
        let Some(index) = material.index() else {
            // The default material: glTF's index-less fallback. Meshes that use
            // it reference `AssetId::NONE` and the renderer supplies defaults,
            // so emitting an asset for it would be a second way to say the same
            // thing.
            continue;
        };
        assets.push(Compiled {
            name: format!("{stem}/material/{index}"),
            kind: AssetKind::Material,
            blob: convert_material(&material, stem).encode(),
        });
    }

    let wanted = referenced_images(&gltf.document);
    let textures = wanted.len();
    if textures > 0 {
        // Decoded here rather than per slot: an ORM map referenced by three
        // materials is one PNG, and decoding it three times would be the
        // pipeline's single most expensive way to get the same texels.
        let images = gltf::import_images(&gltf.document, base, &buffers)
            .with_context(|| format!("decoding the images of {}", path.display()))?;
        for (image, role) in wanted {
            let source = images
                .get(image)
                .with_context(|| format!("{stem} references image {image}, which is not there"))?;
            let (rgba, width, height) = to_rgba8(source, stem, image)?;
            assets.push(Compiled {
                name: texture_name(stem, image, role),
                kind: AssetKind::Texture,
                blob: crate::texture::compile(&rgba, width, height, role)
                    .with_context(|| format!("compiling image {image} as {}", role.suffix()))?,
            });
        }
    }

    for gltf_mesh in gltf.document.meshes() {
        for (index, primitive) in gltf_mesh.primitives().enumerate() {
            if primitive.mode() != Mode::Triangles {
                // Points, lines and strips: skipped loudly. A silent skip is
                // how a scene loses geometry and nobody finds out until the
                // picture is wrong.
                tracing::warn!(
                    mesh = gltf_mesh.index(),
                    primitive = index,
                    mode = ?primitive.mode(),
                    "skipped: only triangles are imported"
                );
                continue;
            }
            let name = format!("{stem}/mesh/{}.{index}", gltf_mesh.index());
            let material = primitive.material().index().map_or(AssetId::NONE, |i| {
                AssetId::of(&format!("{stem}/material/{i}"))
            });
            let (vertices, indices) = primitive_geometry(&primitive, &buffers, &name)?;
            triangles += indices.len() / 3;
            meshes += 1;
            assets.push(Compiled {
                name,
                kind: AssetKind::Mesh,
                blob: mesh::encode(&vertices, &indices, material),
            });
        }
    }

    let nodes = flatten(&gltf, stem);
    assets.push(Compiled {
        name: format!("{stem}/scene"),
        kind: AssetKind::Scene,
        blob: scene::encode(&nodes),
    });

    Ok(Import {
        assets,
        meshes,
        triangles,
        textures,
        clips: 0,
    })
}

/// A texture asset's pack-wide name. The role is in it because the same image
/// compiles differently per slot, and two assets that shared a name would be
/// one asset with whichever bytes were written last.
fn texture_name(stem: &str, image: usize, role: Role) -> String {
    format!("{stem}/texture/{image}/{}", role.suffix())
}

/// Whether pack entry `name` was built from the source named `stem` — the
/// inverse of the naming above, and the ownership test `reuse` runs against the
/// previous pack. A bare `{stem}/` prefix match is not it: `props.gltf` and
/// `props/chair.gltf` share that prefix, so a stem would claim its namesake
/// directory's assets — refusing reuse forever while the hashes differ, and
/// failing a warm build with `DuplicateId` the day the two files are identical.
///
/// An empty remainder is the *exact* name and therefore owned: an `.hdr`'s
/// environment and a `.wav`'s clip are both named by the stem alone. Missing it
/// meant every `.hdr` recompiled on every warm build, silently — the pack was
/// still correct, so nothing failed except the cache.
pub fn owns(stem: &str, name: &str) -> bool {
    name.strip_prefix(stem).is_some_and(|rest| {
        rest.is_empty()
            || rest == "/scene"
            || rest == "/radiance"
            || ["/mesh/", "/material/", "/texture/"]
                .iter()
                .any(|kind| rest.starts_with(kind))
    })
}

/// Which of a material's four slots point at an image, and at which one.
fn slots(material: &gltf::Material<'_>) -> Vec<(Role, usize, u32)> {
    let pbr = material.pbr_metallic_roughness();
    let mut found = Vec::new();
    if let Some(info) = pbr.base_color_texture() {
        found.push((
            Role::BaseColor,
            info.texture().source().index(),
            info.tex_coord(),
        ));
    }
    if let Some(info) = pbr.metallic_roughness_texture() {
        found.push((
            Role::MetallicRoughness,
            info.texture().source().index(),
            info.tex_coord(),
        ));
    }
    if let Some(normal) = material.normal_texture() {
        found.push((
            Role::Normal,
            normal.texture().source().index(),
            normal.tex_coord(),
        ));
    }
    if let Some(occlusion) = material.occlusion_texture() {
        found.push((
            Role::Occlusion,
            occlusion.texture().source().index(),
            occlusion.tex_coord(),
        ));
    }
    found
}

/// Every (image, role) pair some material references, deduplicated and in a
/// defined order.
///
/// Sorted rather than left in discovery order: this decides which image is
/// decoded when, and a build log that reads differently on two machines is the
/// first thing to go when byte-reproducibility (§4.6) starts to slip.
fn referenced_images(document: &gltf::Document) -> Vec<(usize, Role)> {
    let mut wanted = Vec::new();
    for material in document.materials() {
        for (role, image, uv) in slots(&material) {
            if uv != 0 {
                // Only TEXCOORD_0 is imported (see `primitive_geometry`), so a
                // slot asking for a second set would sample coordinates the
                // vertex does not carry.
                tracing::warn!(
                    material = material.index(),
                    slot = role.suffix(),
                    tex_coord = uv,
                    "sampled with TEXCOORD_0 instead: only one uv set is imported"
                );
            }
            wanted.push((image, role));
        }
    }
    wanted.sort_unstable_by_key(|&(image, role)| (image, role.suffix()));
    wanted.dedup();
    wanted
}

/// Normalize a decoded glTF image to RGBA8, the one layout the codec takes.
fn to_rgba8(image: &gltf::image::Data, stem: &str, index: usize) -> Result<(Vec<u8>, u32, u32)> {
    use gltf::image::Format;
    let texels = (image.width as usize) * (image.height as usize);
    // Missing channels are filled the way a sampler would read them: an absent
    // colour channel is black, an absent alpha is opaque.
    let mut rgba = vec![0u8; texels * 4];
    let spread = |rgba: &mut [u8], source: &[u8], have: usize, stride: usize, take: usize| {
        for (texel, out) in rgba.chunks_exact_mut(4).enumerate() {
            for channel in 0..have {
                out[channel] = source[texel * stride + channel * take];
            }
            if have < 4 {
                out[3] = u8::MAX;
            }
        }
    };
    match image.format {
        // The 8-bit cases: `have` channels present, one byte each.
        Format::R8 => spread(&mut rgba, &image.pixels, 1, 1, 1),
        Format::R8G8 => spread(&mut rgba, &image.pixels, 2, 2, 1),
        Format::R8G8B8 => spread(&mut rgba, &image.pixels, 3, 3, 1),
        Format::R8G8B8A8 => rgba.copy_from_slice(&image.pixels),
        // 16-bit PNG, narrowed by taking the high byte — little-endian in the
        // decoder's output, so that is the second of each pair. Every block
        // format here is 8-bit, so the low byte has nowhere to go.
        Format::R16 => spread(&mut rgba, &image.pixels[1..], 1, 2, 2),
        Format::R16G16 => spread(&mut rgba, &image.pixels[1..], 2, 4, 2),
        Format::R16G16B16 => spread(&mut rgba, &image.pixels[1..], 3, 6, 2),
        Format::R16G16B16A16 => spread(&mut rgba, &image.pixels[1..], 4, 8, 2),
        // glTF permits PNG and JPEG and nothing else, neither of which decodes
        // to float — so this is a source that arrived by another route, and
        // guessing an exposure for it is not the importer's call.
        other => bail!(
            "{stem} image {index} decoded as {other:?}, which glTF does not permit — PNG or JPEG"
        ),
    }
    Ok((rgba, image.width, image.height))
}

/// Pull one primitive's vertices and indices into the engine's format.
fn primitive_geometry(
    primitive: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
    name: &str,
) -> Result<(Vec<Vertex>, Vec<u32>)> {
    let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(|b| &b.0[..]));
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .with_context(|| format!("{name} has no POSITION attribute"))?
        .collect();

    // glTF makes NORMAL optional and says a mesh without one is flat-shaded.
    // Generating the flat normals here is the whole of that rule; leaving them
    // zero would light the mesh as if it faced nowhere.
    let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(Iterator::collect);
    // TEXCOORD_0 only. A second uv set is a lightmap or a detail layer, and
    // neither has a consumer until there is a material model that reads one.
    let uvs: Option<Vec<[f32; 2]>> = reader
        .read_tex_coords(0)
        .map(|uv| uv.into_f32().collect::<Vec<_>>());
    // glTF's TANGENT is already `xyz` plus a handedness `w`, which is the
    // engine's format too — so a document that supplies one is copied, not
    // reinterpreted. Only a document that omits it gets a generated frame, and
    // an authored one always wins: the artist's normal map was baked against
    // *their* tangents, and regenerating would light every seam differently.
    let tangents: Option<Vec<[f32; 4]>> = reader.read_tangents().map(Iterator::collect);

    let mut indices: Vec<u32> = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect(),
        // An index-less primitive draws its vertices in order. Materializing
        // the indices costs 4 bytes a vertex and buys one draw path.
        None => (0..positions.len() as u32).collect(),
    };
    if !indices.len().is_multiple_of(3) {
        bail!(
            "{name} has {} indices, which is not whole triangles",
            indices.len()
        );
    }
    for (at, &index) in indices.iter().enumerate() {
        if index as usize >= positions.len() {
            // Checked here so the runtime never has to: a bad index is a source
            // problem, and this is the only place that knows the source's name.
            bail!(
                "{name} index {at} refers to vertex {index} of {}",
                positions.len()
            );
        }
    }

    let mut vertices: Vec<Vertex> = positions
        .iter()
        .enumerate()
        // `get` rather than an index: glTF permits accessors of differing
        // counts and a short NORMAL array is a malformed source, not a panic.
        .map(|(i, &position)| Vertex {
            position,
            normal: normals
                .as_ref()
                .and_then(|n| n.get(i))
                .copied()
                .unwrap_or([0.0; 3]),
            uv: uvs
                .as_ref()
                .and_then(|u| u.get(i))
                .copied()
                .unwrap_or([0.0; 2]),
            tangent: tangents
                .as_ref()
                .and_then(|t| t.get(i))
                .copied()
                .unwrap_or([0.0; 4]),
        })
        .collect();
    match normals.is_none() {
        // Derived from the winding, so they agree with it by construction and
        // there is nothing here to check.
        true => flat_normals(&mut vertices, &indices),
        false => check_winding(&vertices, &indices, name)?,
    }
    if tangents.is_none() {
        generate_tangents(&mut vertices, &indices);
    }
    let vertices = optimize(vertices, &mut indices);
    Ok((vertices, indices))
}

/// meshoptimizer's three reordering passes, in the order it requires (§2, Mesh
/// pipeline row). Every one permutes; none changes the set of triangles or a
/// vertex's contents, so the picture is identical and the fetch pattern is not.
///
/// Deterministic, and load-bearing that it is: meshoptimizer is single-threaded
/// C over IEEE basic operations, so two hosts produce the same permutation and
/// §4.6's byte-reproducibility survives the optimizer.
fn optimize(vertices: Vec<Vertex>, indices: &mut [u32]) -> Vec<Vertex> {
    if vertices.is_empty() || indices.is_empty() {
        return vertices;
    }
    // Post-transform cache first: overdraw is documented to take *its* output,
    // and fetch last because it renumbers the vertices the other two ranked.
    meshopt::optimize_vertex_cache_in_place(indices, vertices.len());
    if let Ok(positions) =
        meshopt::VertexDataAdapter::new(bytemuck::cast_slice(&vertices), size_of::<Vertex>(), 0)
    {
        // 1.05: up to 5% of cache efficiency traded for overdraw, upstream's own
        // suggested default and not a number measured here.
        meshopt::optimize_overdraw_in_place(indices, &positions, 1.05);
    }
    meshopt::optimize_vertex_fetch(indices, &vertices)
}

/// Refuse a primitive whose triangles wind against their own authored normals.
///
/// A defect with no other witness (§6 M64), which is what earns it a gate:
/// nothing downstream reads winding until something asks the rasterizer which
/// side of a face it is on, so an inverted primitive imports clean, renders
/// right for as long as every pass ignores facing, and then shades inside-out
/// the day the forward pass starts honouring `SV_IsFrontFace`. Demo 06's ten
/// spheres were exactly that, checked in at §6 M11 and found four milestones
/// later by a golden that moved for what looked like an unrelated reason.
///
/// Graded per *primitive* and on a **majority**, not on one triangle: a smooth
/// low-poly surface can legitimately carry a vertex normal on the far side of a
/// sliver's geometric one, and a pole fan's degenerate triangles have no
/// geometric normal to disagree with. What this catches is the shape a mistake
/// has — every triangle in the primitive, because the whole index buffer was
/// emitted the other way round.
fn check_winding(vertices: &[Vertex], indices: &[u32], name: &str) -> Result<()> {
    let (mut against, mut counted) = (0usize, 0usize);
    for triangle in indices.chunks_exact(3) {
        let corner = |i: usize| &vertices[triangle[i] as usize];
        let at = |i: usize| DVec3::from(corner(i).position.map(f64::from));
        // The three authored normals summed, against the winding's own normal.
        // Summed rather than averaged: only the sign of the dot product is read.
        let authored = (0..3).fold(DVec3::ZERO, |sum, i| {
            sum + DVec3::from(corner(i).normal.map(f64::from))
        });
        let geometric = (at(1) - at(0)).cross(at(2) - at(0));
        // Exactly zero is a degenerate triangle or an unset normal — neither is
        // evidence either way, so it does not vote.
        match geometric.dot(authored).partial_cmp(&0.0) {
            Some(core::cmp::Ordering::Less) => (against, counted) = (against + 1, counted + 1),
            Some(core::cmp::Ordering::Greater) => counted += 1,
            _ => {}
        }
    }
    if against * 2 > counted {
        anyhow::bail!(
            "{name} winds against its own normals in {against} of {counted} triangles — the \
             source's index order is reversed. Nothing reads winding until a pass asks which \
             side of a face it is on, so this renders correctly until one does and then shades \
             inside-out (§6 M64)."
        );
    }
    Ok(())
}

/// Give every vertex the normal of the last triangle that names it. Correct
/// only because glTF's flat-shading rule means such a mesh has no shared
/// vertices to disagree about — a mesh that does share them was authored with
/// normals.
fn flat_normals(vertices: &mut [Vertex], indices: &[u32]) {
    for triangle in indices.chunks_exact(3) {
        let corner = |i: usize| DVec3::from(vertices[triangle[i] as usize].position.map(f64::from));
        let normal = (corner(1) - corner(0))
            .cross(corner(2) - corner(0))
            .normalize_or_zero();
        let normal = [normal.x as f32, normal.y as f32, normal.z as f32];
        for &index in triangle {
            vertices[index as usize].normal = normal;
        }
    }
}

/// Build a tangent frame for a primitive whose glTF document supplied none.
///
/// Lengyel's method (*Mathematics for 3D Game Programming*, §7.8, and the same
/// derivation in the glTF sample viewer): for each triangle, solve the 2x2
/// system that maps the uv edges onto the position edges, which gives the
/// object-space directions in which u and v increase. Accumulate both per
/// vertex, then orthonormalize the tangent against the interpolated normal by
/// Gram-Schmidt and store the bitangent's sign.
///
/// This is *not* MikkTSpace, and the difference is worth naming: MikkTSpace
/// splits vertices where the frame is discontinuous, and this averages across
/// the seam instead. For an asset whose normal map was baked against MikkTSpace
/// that is visible at the seam — which is exactly why an authored `TANGENT`
/// wins, and why a pipeline that bakes its own maps should supply one.
///
/// `f64` throughout and narrowed once, like [`flat_normals`]: §4.6's
/// byte-reproducibility means two hosts must produce the same pack, and the
/// accumulation order is the file's triangle order on both.
fn generate_tangents(vertices: &mut [Vertex], indices: &[u32]) {
    let mut tangents = vec![DVec3::ZERO; vertices.len()];
    let mut bitangents = vec![DVec3::ZERO; vertices.len()];
    for triangle in indices.chunks_exact(3) {
        let at = |i: usize| &vertices[triangle[i] as usize];
        let position = |i: usize| DVec3::from(at(i).position.map(f64::from));
        let uv = |i: usize| (f64::from(at(i).uv[0]), f64::from(at(i).uv[1]));

        let (edge1, edge2) = (position(1) - position(0), position(2) - position(0));
        let ((u0, v0), (u1, v1), (u2, v2)) = (uv(0), uv(1), uv(2));
        let (du1, dv1) = (u1 - u0, v1 - v0);
        let (du2, dv2) = (u2 - u0, v2 - v0);
        let determinant = du1 * dv2 - du2 * dv1;
        // A degenerate uv triangle — a face with no texture area, which is
        // ordinary on untextured geometry — contributes nothing rather than an
        // infinity that would poison every vertex it touches.
        if determinant == 0.0 {
            continue;
        }
        let scale = 1.0 / determinant;
        let tangent = (edge1 * dv2 - edge2 * dv1) * scale;
        let bitangent = (edge2 * du1 - edge1 * du2) * scale;
        for &index in triangle {
            tangents[index as usize] += tangent;
            bitangents[index as usize] += bitangent;
        }
    }

    for (vertex, (tangent, bitangent)) in vertices
        .iter_mut()
        .zip(tangents.into_iter().zip(bitangents))
    {
        let normal = DVec3::from(vertex.normal.map(f64::from)).normalize_or_zero();
        // Gram-Schmidt: the accumulated tangent is only approximately in the
        // surface plane once several triangles have averaged into it.
        let orthogonal = (tangent - normal * normal.dot(tangent)).normalize_or_zero();
        let orthogonal = match orthogonal == DVec3::ZERO {
            // No usable tangent: an untextured face, or one whose uvs collapsed.
            // Any vector in the surface plane will do, and picking it from the
            // world axis *least* aligned with the normal is what stops the cross
            // product from being near-zero.
            true => normal.cross(least_aligned_axis(normal)).normalize_or_zero(),
            false => orthogonal,
        };
        // glTF stores the bitangent as a sign, so this is the one bit that says
        // whether the uv layout was mirrored on this face.
        let handedness = match normal.cross(orthogonal).dot(bitangent) < 0.0 {
            true => -1.0,
            false => 1.0,
        };
        vertex.tangent = [
            orthogonal.x as f32,
            orthogonal.y as f32,
            orthogonal.z as f32,
            handedness,
        ];
    }
}

/// The world axis `normal` is least aligned with — the one whose cross product
/// with it is furthest from zero.
fn least_aligned_axis(normal: DVec3) -> DVec3 {
    let (x, y, z) = (normal.x.abs(), normal.y.abs(), normal.z.abs());
    if x <= y && x <= z {
        DVec3::X
    } else if y <= z {
        DVec3::Y
    } else {
        DVec3::Z
    }
}

/// glTF's material model is ours, so this copies rather than converts.
fn convert_material(material: &gltf::Material<'_>, stem: &str) -> Material {
    let pbr = material.pbr_metallic_roughness();
    let mut flags = 0;
    if material.alpha_mode() == gltf::material::AlphaMode::Mask {
        flags |= flags::ALPHA_MASK;
    }
    if material.double_sided() {
        flags |= flags::DOUBLE_SIDED;
    }
    // glTF scales the sampled normal and the sampled occlusion by per-slot
    // factors the 64-byte record has no room for. Warned rather than dropped in
    // silence: both change the picture, and a value that is not 1 was authored
    // deliberately by someone who will wonder where it went.
    if material.normal_texture().is_some_and(|n| n.scale() != 1.0) {
        tracing::warn!(
            material = material.index(),
            "normal scale is not 1 and is not imported"
        );
    }
    if material
        .occlusion_texture()
        .is_some_and(|o| o.strength() != 1.0)
    {
        tracing::warn!(
            material = material.index(),
            "occlusion strength is not 1 and is not imported"
        );
    }

    let mut converted = Material {
        base_color: pbr.base_color_factor(),
        metallic: pbr.metallic_factor(),
        roughness: pbr.roughness_factor(),
        alpha_cutoff: material.alpha_cutoff().unwrap_or(0.5),
        flags,
        ..Material::default()
    };
    // The ids are derived from the names, so this needs nothing from the
    // texture pass beyond agreeing with it on what a name is. A slot pointing
    // at an image the document does not have would already have failed there.
    for (role, image, _) in slots(material) {
        let id = AssetId::of(&texture_name(stem, image, role));
        match role {
            Role::BaseColor => converted.base_color_texture = id,
            Role::Normal => converted.normal_texture = id,
            Role::MetallicRoughness => converted.metallic_roughness_texture = id,
            Role::Occlusion => converted.occlusion_texture = id,
        }
    }
    converted
}

/// Walk the default scene's node tree and multiply it out into a flat list.
///
/// Composition is `f64` throughout: a node ten kilometres from the origin is a
/// number `f32` cannot hold to the millimetre, and the pack's translation field
/// is `f64` precisely so the importer is not the place that decides otherwise.
fn flatten(gltf: &gltf::Gltf, stem: &str) -> Vec<Node> {
    let mut nodes = Vec::new();
    let Some(scene) = gltf
        .document
        .default_scene()
        .or_else(|| gltf.document.scenes().next())
    else {
        return nodes;
    };
    for root in scene.nodes() {
        visit(&root, DMat4::IDENTITY, stem, &mut nodes);
    }
    nodes
}

fn visit(node: &gltf::Node<'_>, parent: DMat4, stem: &str, out: &mut Vec<Node>) {
    let local = DMat4::from_cols_array_2d(&node.transform().matrix().map(|c| c.map(f64::from)));
    let world = parent * local;
    if let Some(gltf_mesh) = node.mesh() {
        // A node draws every primitive of its mesh, so a three-primitive mesh
        // becomes three nodes sharing one transform. The alternative — a node
        // that names a mesh and lets the runtime expand it — would put a level
        // of indirection in the draw loop to save bytes in the file.
        let (scale, rotation, translation) = world.to_scale_rotation_translation();
        for (index, primitive) in gltf_mesh.primitives().enumerate() {
            // The same skip the importer made, and the same index it counted
            // with: a node that named a primitive nobody compiled would be a
            // dangling id the loader could only report as a missing asset.
            if primitive.mode() != Mode::Triangles {
                continue;
            }
            out.push(Node {
                mesh: AssetId::of(&format!("{stem}/mesh/{}.{index}", gltf_mesh.index())),
                translation: [translation.x, translation.y, translation.z],
                rotation: [
                    rotation.x as f32,
                    rotation.y as f32,
                    rotation.z as f32,
                    rotation.w as f32,
                ],
                scale: [scale.x as f32, scale.y as f32, scale.z as f32],
                reserved: [0; 5],
            });
        }
    }
    for child in node.children() {
        visit(&child, world, stem, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cache-reuse collision: `props.gltf` and `props/chair.gltf` share the
    /// `props/` prefix, so a prefix match handed one source the other's assets.
    #[test]
    fn ownership_is_by_source_not_by_prefix() {
        assert!(owns("props", "props/scene"));
        assert!(owns("props", "props/mesh/0.1"));
        assert!(owns("props", "props/material/2"));
        assert!(owns("props", "props/texture/0/albedo"));
        assert!(!owns("props", "props/chair/scene"));
        assert!(!owns("props", "props/chair/mesh/0.0"));
        assert!(owns("props/chair", "props/chair/mesh/0.0"));
        assert!(!owns("props/chair", "props/scene"));
        // A stem that merely *starts* like another claims nothing of its
        // namesake either direction.
        assert!(!owns("prop", "props/scene"));
        // The same hazard one character in, which is what the empty remainder
        // below must not weaken: `a` owns `a`, never anything of `ab`.
        assert!(!owns("a", "ab"));
    }

    /// The bare-`{stem}` regression: an `.hdr` names its environment by the stem
    /// alone, so an ownership test that required a `/` suffix found none of a
    /// panorama's assets and recompiled every one of them on every warm build.
    #[test]
    fn a_source_whose_asset_carries_the_bare_stem_is_still_owned() {
        assert!(owns("sky/dusk", "sky/dusk"), "the environment");
        assert!(owns("sky/dusk", "sky/dusk/radiance"), "its chain");
        // And a clip, which is the whole of a `.wav`'s output.
        assert!(owns("sfx/pickup", "sfx/pickup"));
        assert!(!owns("sfx/pick", "sfx/pickup"));
    }

    /// A grid of quads, emitted in scanline order — the order that maximizes
    /// vertex cache misses, so the optimizer has something to do.
    fn grid(side: u32) -> (Vec<Vertex>, Vec<u32>) {
        let stride = side + 1;
        let vertices = (0..stride * stride)
            .map(|i| Vertex {
                position: [(i % stride) as f32, (i / stride) as f32, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0; 2],
                tangent: [1.0, 0.0, 0.0, 1.0],
            })
            .collect();
        let mut indices = Vec::new();
        for y in 0..side {
            for x in 0..side {
                let at = y * stride + x;
                indices.extend_from_slice(&[at, at + 1, at + stride]);
                indices.extend_from_slice(&[at + 1, at + stride + 1, at + stride]);
            }
        }
        (vertices, indices)
    }

    /// The triangles a mesh draws, by *position* rather than by index — the one
    /// thing three reordering passes may not change.
    fn triangles(vertices: &[Vertex], indices: &[u32]) -> Vec<[[u32; 3]; 3]> {
        let mut faces: Vec<[[u32; 3]; 3]> = indices
            .chunks_exact(3)
            .map(|face| {
                let corner = |i: usize| vertices[face[i] as usize].position.map(|c| c.to_bits());
                [corner(0), corner(1), corner(2)]
            })
            .collect();
        faces.sort_unstable();
        faces
    }

    #[test]
    fn the_optimizer_permutes_a_mesh_without_changing_what_it_draws() {
        let (vertices, mut indices) = grid(8);
        let before = triangles(&vertices, &indices);
        let original = indices.clone();

        let optimized = optimize(vertices.clone(), &mut indices);

        assert_eq!(
            optimized.len(),
            vertices.len(),
            "no vertex is added or lost"
        );
        assert_ne!(
            indices, original,
            "the passes should have reordered something"
        );
        // Same triangles, same winding, different order in the buffer.
        assert_eq!(triangles(&optimized, &indices), before);
    }

    #[test]
    fn optimizing_the_same_mesh_twice_gives_the_same_answer() {
        // Byte-reproducibility (§4.6) reaches through meshoptimizer: if the
        // permutation moved between runs so would every mesh blob.
        let (vertices, mut first) = grid(6);
        let mut second = first.clone();
        let one = optimize(vertices.clone(), &mut first);
        let two = optimize(vertices, &mut second);
        assert_eq!(first, second);
        assert_eq!(one, two);
    }

    #[test]
    fn a_mesh_with_no_geometry_survives_the_optimizer() {
        // meshoptimizer is C and takes raw pointers; an empty run must not reach
        // it at all rather than reach it with a null.
        let mut indices = Vec::new();
        assert!(optimize(Vec::new(), &mut indices).is_empty());
    }

    /// One quad in the XY plane facing +Z, with uv increasing along +X and +Y —
    /// the case whose right answer is written down rather than derived.
    fn quad(uv: [[f32; 2]; 4]) -> (Vec<Vertex>, Vec<u32>) {
        let corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let vertices = corners
            .iter()
            .zip(uv)
            .map(|(&position, uv)| Vertex {
                position,
                normal: [0.0, 0.0, 1.0],
                uv,
                tangent: [0.0; 4],
            })
            .collect();
        (vertices, vec![0, 1, 2, 0, 2, 3])
    }

    /// The quad's corners run counter-clockwise seen from +Z and its normals are
    /// +Z, so the winding agrees and nothing is refused (§6 M64).
    #[test]
    fn a_quad_wound_with_its_normals_imports() {
        let (vertices, indices) = quad([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        assert!(check_winding(&vertices, &indices, "quad").is_ok());
    }

    /// Demo 06's ten spheres, in miniature: the same surface with its index
    /// buffer emitted the other way round. Both triangles disagree, so the
    /// majority is the whole primitive.
    #[test]
    fn a_quad_wound_against_its_normals_is_refused() {
        let (vertices, indices) = quad([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let flipped: Vec<u32> = indices
            .chunks_exact(3)
            .flat_map(|t| [t[0], t[2], t[1]])
            .collect();
        let Err(error) = check_winding(&vertices, &flipped, "sphere") else {
            panic!("an inverted primitive imported clean");
        };
        let error = error.to_string();
        assert!(error.contains("2 of 2 triangles"), "{error}");
        assert!(error.contains("sphere"), "{error}");
    }

    /// A single disagreeing triangle is not a verdict: the gate is a majority,
    /// because a smoothed surface may legitimately carry a vertex normal on the
    /// far side of one sliver's geometric one.
    #[test]
    fn one_triangle_against_the_grain_is_not_a_refusal() {
        let (vertices, indices) = quad([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let mut one_bad = indices.clone();
        one_bad.swap(1, 2);
        assert!(check_winding(&vertices, &one_bad, "quad").is_ok());
    }

    /// A degenerate triangle has no geometric normal, so it votes neither way —
    /// and a primitive made only of them is not evidence of anything. A pole fan
    /// is where these come from, which is why demo 06's spheres counted 414 of
    /// 432 rather than all of them.
    #[test]
    fn degenerate_triangles_do_not_vote() {
        let (vertices, _) = quad([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        assert!(check_winding(&vertices, &[0, 1, 1], "sliver").is_ok());
    }

    #[test]
    fn a_generated_tangent_points_the_way_u_increases() {
        // u along +X, v along +Y: the tangent is +X and the frame is
        // right-handed, so the handedness bit is +1.
        let (mut vertices, indices) = quad([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        generate_tangents(&mut vertices, &indices);
        for vertex in &vertices {
            assert!(
                (vertex.tangent[0] - 1.0).abs() < 1e-5,
                "{:?}",
                vertex.tangent
            );
            assert!(vertex.tangent[1].abs() < 1e-5);
            assert!(vertex.tangent[2].abs() < 1e-5);
            assert_eq!(vertex.tangent[3], 1.0);
        }
    }

    #[test]
    fn a_mirrored_uv_layout_flips_the_handedness_bit_and_nothing_else() {
        // v runs the other way. The tangent still points along +X; what changes
        // is the sign that tells the shader which way the bitangent goes — which
        // is the whole reason the bitangent is a bit rather than a vector.
        let (mut vertices, indices) = quad([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]);
        generate_tangents(&mut vertices, &indices);
        for vertex in &vertices {
            assert!(
                (vertex.tangent[0] - 1.0).abs() < 1e-5,
                "{:?}",
                vertex.tangent
            );
            assert_eq!(vertex.tangent[3], -1.0, "mirrored");
        }
    }

    #[test]
    fn a_face_with_no_texture_area_still_gets_a_usable_frame() {
        // Every uv identical: the 2x2 system is singular. A zero tangent would
        // make the TBN singular too and normal mapping would come out black, so
        // the fallback has to be a real vector in the surface plane.
        let (mut vertices, indices) = quad([[0.5, 0.5]; 4]);
        generate_tangents(&mut vertices, &indices);
        for vertex in &vertices {
            let t = DVec3::new(
                f64::from(vertex.tangent[0]),
                f64::from(vertex.tangent[1]),
                f64::from(vertex.tangent[2]),
            );
            assert!((t.length() - 1.0).abs() < 1e-5, "{:?}", vertex.tangent);
            // And it lies in the surface: perpendicular to the normal.
            assert!(t.z.abs() < 1e-5, "{:?}", vertex.tangent);
        }
    }

    #[test]
    fn generating_tangents_twice_gives_the_same_bytes() {
        // §4.6's byte-reproducibility reaches through this too: the accumulation
        // is `f64` over the file's own triangle order, so two hosts agree.
        let (mut first, indices) = quad([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let mut second = first.clone();
        generate_tangents(&mut first, &indices);
        generate_tangents(&mut second, &indices);
        assert_eq!(first, second);
    }
}
