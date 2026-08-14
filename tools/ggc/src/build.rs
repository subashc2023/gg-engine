//! The build: walk a source tree, compile what changed, write a pack.
//!
//! **The pack is its own cache** (§4.6's incremental rebuilds). Every entry
//! already carries the xxhash3 of the source and settings that produced it, so
//! a warm build opens the previous pack, hashes each source, and copies the
//! blobs whose hash still agrees. There is no sidecar cache file to go stale,
//! to be deleted by a clean, or to disagree with the artifact beside it — and
//! the thing being reused is the exact bytes that shipped last time.
//!
//! **The settings are part of the hash, and so is this compiler.**
//! [`IMPORT_VERSION`] folds into every source hash, so changing what the
//! importer emits invalidates every asset rather than silently reusing blobs
//! built by the old rules.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use gg_assets::{AssetKind, Pack, PackWriter};
use twox_hash::xxhash3_64::Hasher as Xxh3;

use crate::import;

/// Bumped whenever the importer's output changes for unchanged input. Part of
/// every source hash, which is what makes that statement true rather than
/// aspirational.
/// 2 since M11: meshes carry a tangent, generated where the document has none.
/// 3 since M27: `.hdr` sources compile, and every prefilter constant beside them
/// is part of what a blob means.
/// 4 since M43: `.wav` sources compile to clips.
pub const IMPORT_VERSION: u32 = 4;

/// What a build did, for the log and for the tests.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Sources compiled from scratch.
    pub compiled: usize,
    /// Sources whose assets were copied out of the previous pack.
    pub reused: usize,
    /// Assets in the finished pack.
    pub assets: usize,
    /// Triangles across every mesh compiled this run.
    pub triangles: usize,
    /// (image, role) pairs block-compressed this run.
    pub textures: usize,
    /// Clips compiled this run.
    pub clips: usize,
}

/// Compile every source under `source` into a pack at `out`.
pub fn build(source: &Path, out: &Path) -> Result<Stats> {
    if !source.is_dir() {
        bail!("{} is not a directory", source.display());
    }
    let sources = walk(source)?;
    // The previous pack is opened read-only and outlives the writer, because
    // the blobs copied out of it are borrowed from its mapping.
    let previous = Pack::open(out).ok();
    if previous.is_none() && out.exists() {
        // An unreadable pack is a full rebuild, not a failure: the common cause
        // is a `FORMAT_VERSION` bump, and refusing to build until someone
        // deletes a file would be a compiler that cannot recover from its own
        // release.
        tracing::info!(pack = %out.display(), "the previous pack could not be read — rebuilding it");
    }

    let mut writer = PackWriter::new();
    let mut stats = Stats::default();
    for path in &sources {
        let stem = stem_of(source, path)?;
        let hash = source_hash(path)?;
        if let Some(previous) = &previous
            && let Some(reused) = reuse(previous, &stem, hash, &mut writer)?
        {
            tracing::debug!(source = %stem, assets = reused, "reused");
            stats.reused += 1;
            stats.assets += reused;
            continue;
        }
        let import = import::document(path, &stem)?;
        tracing::info!(
            source = %stem,
            meshes = import.meshes,
            triangles = import.triangles,
            textures = import.textures,
            clips = import.clips,
            "compiled"
        );
        stats.compiled += 1;
        stats.triangles += import.triangles;
        stats.textures += import.textures;
        stats.clips += import.clips;
        for asset in import.assets {
            writer.add(&asset.name, asset.kind, hash, asset.blob)?;
            stats.assets += 1;
        }
    }

    let bytes = writer.finish()?;
    write_atomically(out, &bytes)?;
    Ok(stats)
}

/// Copy `stem`'s assets out of `previous` if every one of them still carries
/// `hash`. All or nothing: a source is the unit that is hashed, so reusing half
/// of one would mean trusting a hash that covers the other half too.
fn reuse(previous: &Pack, stem: &str, hash: u64, writer: &mut PackWriter) -> Result<Option<usize>> {
    // Ownership by the naming shape, not by prefix — `import::owns` is the
    // inverse of the naming `import::document` writes.
    let owned: Vec<_> = previous
        .entries()
        .iter()
        .filter(|e| import::owns(stem, previous.name(e)))
        .collect();
    if owned.is_empty() || owned.iter().any(|e| e.source_hash != hash) {
        return Ok(None);
    }
    for entry in &owned {
        let Some(kind) = entry.kind() else {
            return Ok(None);
        };
        writer.add(
            previous.name(entry),
            kind,
            hash,
            previous.blob(entry).to_vec(),
        )?;
    }
    Ok(Some(owned.len()))
}

/// Every source under `root`, sorted by path.
///
/// Sorted because the walk order is the filesystem's and differs between
/// machines; the pack's own sort by id would hide that, but the *build log*
/// and any error's ordering would not.
fn walk(root: &Path) -> Result<Vec<PathBuf>> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in
            std::fs::read_dir(&dir).with_context(|| format!("reading {}", dir.display()))?
        {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("gltf" | "glb" | "hdr" | "wav")
            ) {
                found.push(path);
            }
        }
    }
    found.sort();
    Ok(found)
}

/// A source's pack-relative name: its path under the source root, without an
/// extension, with forward slashes.
///
/// Forward slashes unconditionally, because the name is what is *hashed* into
/// an id — a pack built on Windows and one built on Linux must name the same
/// asset identically or nothing that references it resolves (§4.6).
fn stem_of(root: &Path, path: &Path) -> Result<String> {
    let relative = path
        .strip_prefix(root)
        .with_context(|| format!("{} is not under {}", path.display(), root.display()))?
        .with_extension("");
    let mut stem = String::new();
    for component in relative.components() {
        if !stem.is_empty() {
            stem.push('/');
        }
        let component = component
            .as_os_str()
            .to_str()
            .with_context(|| format!("{} is not utf-8", path.display()))?;
        stem.push_str(component);
    }
    Ok(stem)
}

/// Hash a source and everything it references.
///
/// The document's own bytes plus every external buffer and image it names, in
/// document order. Reading the images here costs a pass over bytes the codec
/// will read again, and buys the property that editing a texture invalidates
/// the model that uses it — which is the edit hot reload exists for.
fn source_hash(path: &Path) -> Result<u64> {
    source_hash_at(path, IMPORT_VERSION)
}

/// [`source_hash`] with the importer version as an argument, so a gate can ask
/// what a *different* version would have produced. The const cannot be varied
/// at runtime, and "bumping this invalidates everything" is worth more than a
/// comment claiming it.
pub fn source_hash_at(path: &Path, import_version: u32) -> Result<u64> {
    let mut hasher = Xxh3::new();
    core::hash::Hasher::write_u32(&mut hasher, import_version);
    // The codec's identity, not just ours: zstd's output moves between library
    // versions, so a `cargo update` re-authors every texture in the pack. Left
    // to [`IMPORT_VERSION`] it would be a step someone has to remember, and
    // `xtask assets --check` could not catch the omission — it builds twice
    // against the *same* library and would agree with itself.
    core::hash::Hasher::write_u32(&mut hasher, gg_assets::texture::codec_version());
    // And the machine's SIMD, for the same reason one step further out: the BC7
    // kernel is chosen by CPUID and its variants do not agree bit for bit (see
    // [`crate::texture::encoder_isa`]). Without this, a pack carried from one
    // desk to another reuses the first machine's blobs beside the second's
    // recompiles — one pack, two encoders — and `--check` cannot see it.
    core::hash::Hasher::write_u32(&mut hasher, crate::texture::encoder_isa());
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    core::hash::Hasher::write(&mut hasher, &bytes);

    // An `.hdr` and a `.wav` are single self-contained files: their bytes above
    // are the whole source, and there is no document to walk for references.
    // Checked here rather than by trying glTF first, because "it did not parse
    // as glTF" is also what a corrupt glTF looks like.
    if matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("hdr" | "wav")
    ) {
        return Ok(core::hash::Hasher::finish(&hasher));
    }

    let gltf =
        gltf::Gltf::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))?;
    let base = path.parent().unwrap_or(Path::new("."));
    let mut referenced = Vec::new();
    for buffer in gltf.document.buffers() {
        if let gltf::buffer::Source::Uri(uri) = buffer.source() {
            referenced.push(uri.to_owned());
        }
    }
    for image in gltf.document.images() {
        if let gltf::image::Source::Uri { uri, .. } = image.source() {
            referenced.push(uri.to_owned());
        }
    }
    for uri in referenced {
        // A data: URI carries its bytes in the document, which is already
        // hashed above; only a file on disk can change without it.
        if uri.starts_with("data:") {
            continue;
        }
        let decoded = percent_decode(&uri);
        let file = base.join(&decoded);
        let bytes = std::fs::read(&file)
            .with_context(|| format!("{} references {}", path.display(), file.display()))?;
        core::hash::Hasher::write(&mut hasher, &bytes);
    }
    Ok(core::hash::Hasher::finish(&hasher))
}

/// glTF URIs are percent-encoded (the spec says so, and Sponza's paths contain
/// spaces). Only the escape needs undoing — these are relative file paths, not
/// URLs, so there is no scheme, host or query to parse.
fn percent_decode(uri: &str) -> String {
    let bytes = uri.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut at = 0;
    while at < bytes.len() {
        match bytes[at] {
            b'%' if at + 2 < bytes.len() => {
                let hex = core::str::from_utf8(&bytes[at + 1..at + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(byte) => {
                        out.push(byte);
                        at += 3;
                    }
                    Err(_) => {
                        out.push(b'%');
                        at += 1;
                    }
                }
            }
            byte => {
                out.push(byte);
                at += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Write to a temporary beside the target and rename over it.
///
/// A pack is mmap'd by whoever has it open, and writing in place would change
/// bytes under a live mapping — [`Pack::open`]'s contract is that a rebuild
/// replaces the inode. It also means an interrupted build leaves the old pack
/// intact rather than a half-written one.
fn write_atomically(out: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let temporary = out.with_extension("ggpack.tmp");
    std::fs::write(&temporary, bytes)
        .with_context(|| format!("writing {}", temporary.display()))?;
    std::fs::rename(&temporary, out)
        .with_context(|| format!("renaming {} onto {}", temporary.display(), out.display()))?;
    Ok(())
}

/// Print a pack's table, widest column first. `ggc list`.
pub fn list(path: &Path) -> Result<()> {
    let pack = Pack::open(path)?;
    println!("{} — {} assets", path.display(), pack.entries().len());
    for entry in pack.entries() {
        let kind = match entry.kind() {
            Some(AssetKind::Mesh) => "mesh",
            Some(AssetKind::Texture) => "texture",
            Some(AssetKind::Material) => "material",
            Some(AssetKind::Scene) => "scene",
            Some(AssetKind::Environment) => "environment",
            Some(AssetKind::Clip) => "clip",
            None => "?",
        };
        println!(
            "  {:8} {:>10} bytes  {}  {}",
            kind,
            entry.len,
            entry.id,
            pack.name(entry)
        );
    }
    Ok(())
}
