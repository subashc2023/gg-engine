//! The picture and the version a Windows *shell* reads (§6 M73) — a PE resource
//! section, written by hand and handed to the linker as an object file.
//!
//! §6 M46 gave the game an icon and §6 M41 priced this one and declined it: the
//! window's picture is `WindowDesc::with_icon` and arrives when the process is
//! already running, while the **executable's** is a `.rsrc` section the loader
//! never reads and Explorer reads before anything of ours runs. The shipped
//! artifact had no resource directory at all — six sections, none of them
//! `.rsrc` — so a stranger who downloads the zip sees a blank generic icon in
//! their Downloads folder, a blank *Details* tab, and no way to tell two builds
//! apart. That is the first thing about this game anybody sees.
//!
//! # Why by hand, again
//!
//! The ordinary road is `rc.exe` → `cvtres.exe` (the Windows SDK, which the
//! Vulkan SDK does not bring) or a build script on `winres`/`embed-resource`,
//! which puts a crate in `gg-runtime`'s *build* graph to produce four kilobytes
//! of well-documented structs. `deps.rs` already reads PE import tables with no
//! `dumpbin` and `ship.rs` already writes a zip with no deflate crate; this is
//! the third of that family and the argument has not changed. What it costs is
//! this file. What it buys is that the bytes are a function of the manifest and
//! the icon, checked by a reader in the same file, on a machine that has only
//! the two SDKs the tree already requires.
//!
//! # The shape
//!
//! A linker builds `.rsrc` from a **COFF object** holding two grouped sections:
//! `.rsrc$01` is the directory tree (type → id → language, then one
//! `IMAGE_RESOURCE_DATA_ENTRY` per leaf) and `.rsrc$02` is the bytes those
//! leaves point at. The one thing an object cannot know is where the image will
//! be loaded, so every leaf's `OffsetToData` — an **RVA** in the finished file —
//! is emitted as an offset into `.rsrc$02` plus a relocation against that
//! section's symbol, and `link.exe` turns the pair into the address. That is
//! precisely what `cvtres` emits, which is why the linker recognises it.
//!
//! Three resource types go in:
//!
//! - `RT_ICON`, one per size, each a `BITMAPINFOHEADER` with its height doubled
//!   over bottom-up BGRA and an all-zero AND mask. The doubled height is not a
//!   mistake in this file: an icon image is a colour bitmap and a mask stacked,
//!   and the header describes both.
//! - `RT_GROUP_ICON` id 1 — the directory Windows actually asks for, naming each
//!   `RT_ICON` by ordinal. Explorer picks the **lowest-numbered** group in the
//!   file, so being the only one is not enough; being id 1 is.
//! - `RT_VERSION` id 1 — `VS_VERSIONINFO`, which is the *Details* tab.
//!
//! # The sizes, and why none of them is a resample
//!
//! [`icon::SIDE`](gg_core::config::icon::SIDE) is 96 and its doc comment already
//! says why: it divides by 6, 4, 3 and 2, so 16, 24, 32 and 48 are each an exact
//! integer box filter of the same pixels. This is the milestone that spends that
//! choice. The 96 goes in as well, so the sizes above it are the OS upscaling a
//! real image rather than a 48. No filter is designed here, nothing rounds twice,
//! and the whole chain is a function of the one checked-in file.
//!
//! The average is taken **premultiplied** and undone afterwards, which matters
//! only at an edge and matters completely there: straight-averaging RGB across a
//! boundary drags in the colour of pixels whose alpha says they are not visible,
//! and a generated icon is nothing but hard edges.

use anyhow::{Context, Result, bail, ensure};

/// Resource type ids, from `winuser.h`. Ascending, which is also the order a
/// resource directory has to list them in.
const RT_ICON: u32 = 3;
const RT_GROUP_ICON: u32 = 14;
const RT_VERSION: u32 = 16;

/// US English. One language, because the strings below are the manifest's and a
/// manifest has one spelling of them.
const LANG: u32 = 0x0409;

/// The codepage half of a `StringFileInfo` key: 1200, UTF-16.
const CODEPAGE: u32 = 0x04b0;

/// What an OS asks for, largest last. Every one an exact divisor of
/// [`icon::SIDE`](gg_core::config::icon::SIDE) — asserted, not assumed.
const SIDES: [u32; 5] = [16, 24, 32, 48, 96];

/// What the shell shows about this build. Everything here is the manifest's
/// except [`engine`](Marks::engine), which is the tree's.
pub struct Marks {
    /// The game's name — `FileDescription` and `ProductName`, and the bold line
    /// at the top of a properties dialog.
    pub title: String,
    /// The version as written, suffix and all. `None` writes `0.0.0.0` and says
    /// so, because a `VS_VERSIONINFO` with no version in it is worse than none.
    pub version: Option<gg_core::config::project::Version>,
    /// The directory a player's bytes live in — `InternalName`, which is the
    /// field a support answer wants and the one nothing else in the file says.
    pub slug: String,
    /// The executable's own file name, which `OriginalFilename` exists to record
    /// so a renamed copy still names itself.
    pub exe: String,
    /// The engine and its version, into `Comments`. A bug report that reaches
    /// this tree is about this, and a player can read it off the file.
    pub engine: String,
}

impl Marks {
    /// `major.minor.patch.build`, or all zeros for an unversioned build.
    fn quad(&self) -> [u16; 4] {
        self.version.as_ref().map_or([0; 4], |v| v.quad())
    }

    /// What the two version *strings* say. Deliberately not `"unknown"`: the
    /// numeric field is `0.0.0.0` either way, and two spellings of the same
    /// absence is how a comparison goes wrong.
    fn text(&self) -> String {
        self.version
            .as_ref()
            .map_or_else(|| "0.0.0.0".to_owned(), |v| v.text().to_owned())
    }

    /// The `StringFileInfo` table, in the order it is written. Order is not
    /// load-bearing for a reader, and being fixed is what makes the object a
    /// function of its input.
    fn strings(&self) -> Vec<(&'static str, String)> {
        vec![
            ("Comments", self.engine.clone()),
            ("FileDescription", self.title.clone()),
            ("FileVersion", self.text()),
            ("InternalName", self.slug.clone()),
            ("OriginalFilename", self.exe.clone()),
            ("ProductName", self.title.clone()),
            ("ProductVersion", self.text()),
        ]
    }
}

/// The COFF object to hand `link.exe`, and the icon images it carries.
///
/// The images come back because the gate compares them against what it finds in
/// the linked executable, and reading them out of here rather than regenerating
/// them is what makes that a round trip rather than two runs of one function.
#[derive(Debug)]
pub struct Built {
    /// The object file's bytes.
    pub object: Vec<u8>,
    /// `(side, image)` per `RT_ICON`, in resource-id order starting at 1.
    pub icons: Vec<(u32, Vec<u8>)>,
    /// The `VS_VERSIONINFO` blob, as written.
    pub version: Vec<u8>,
}

/// Build the object for `icon` (a [`icon`](gg_core::config::icon) file, or
/// `None` for a manifest that declares a version and no picture) and `marks`.
///
/// The icon is optional and the version record is not: a build with neither has
/// no reason to be here, and a build with only a version still owes Explorer a
/// *Details* tab (§6 M81 — before it, an iconless manifest silently dropped the
/// version too, and with it the only check that covers the linker).
///
/// # Errors
/// If the icon file is not one, or its side is not divisible by every entry in
/// [`SIDES`] — the second is a tree-level mistake rather than a runtime one, and
/// it fails here rather than shipping a stretched picture.
pub fn object(icon: Option<&[u8]>, marks: &Marks) -> Result<Built> {
    let mut icons = Vec::new();
    if let Some(icon) = icon {
        let (side, rgba) =
            gg_core::config::icon::parse(icon).context("reading the icon to ship")?;
        for want in SIDES {
            ensure!(
                want <= side && side.is_multiple_of(want),
                "an icon {side}px square cannot become {want}px by an integer box filter — either \
                 the source side or `SIDES` moved (§6 M73)"
            );
            icons.push((want, image(want, &reduce(side, &rgba, want))));
        }
    }
    let version = version_info(marks);

    // Leaves, in the order the directory will list them: every `RT_ICON` id
    // ascending, then the group, then the version. An iconless build writes the
    // version leaf alone — a group with no members is a picture Explorer would
    // ask for and not find.
    let mut leaves: Vec<(u32, u32, &[u8])> = Vec::new();
    for (i, (_, bytes)) in icons.iter().enumerate() {
        leaves.push((RT_ICON, i as u32 + 1, bytes));
    }
    let group = group(&icons);
    if !icons.is_empty() {
        leaves.push((RT_GROUP_ICON, 1, &group));
    }
    leaves.push((RT_VERSION, 1, &version));

    let (directory, blobs, fixups) = tree(&leaves);
    Ok(Built {
        object: coff(&directory, &blobs, &fixups),
        icons,
        version,
    })
}

/// One size of the icon, box-filtered from `side` down to `to`.
///
/// `side % to == 0` is the caller's, checked in [`object`]. The average is taken
/// on premultiplied colour and undone after, so a transparent pixel contributes
/// its coverage and not its colour.
fn reduce(side: u32, rgba: &[u8], to: u32) -> Vec<u8> {
    if side == to {
        return rgba.to_vec();
    }
    let n = side / to;
    let count = n * n;
    let mut out = vec![0u8; (to as usize) * (to as usize) * 4];
    for y in 0..to {
        for x in 0..to {
            let (mut rgb, mut alpha) = ([0u32; 3], 0u32);
            for dy in 0..n {
                for dx in 0..n {
                    let at = (((y * n + dy) * side + (x * n + dx)) * 4) as usize;
                    let a = u32::from(rgba[at + 3]);
                    for (c, sum) in rgb.iter_mut().enumerate() {
                        *sum += u32::from(rgba[at + c]) * a;
                    }
                    alpha += a;
                }
            }
            let to_at = ((y * to + x) * 4) as usize;
            for (c, sum) in rgb.iter().enumerate() {
                // Divided by the *alpha* sum rather than the sample count:
                // that is what undoes the premultiply. `checked_div` is the
                // whole of the transparent case — a block nothing is visible in
                // has no colour to keep, and zero is that answer.
                out[to_at + c] = (sum + alpha / 2).checked_div(alpha).unwrap_or(0) as u8;
            }
            out[to_at + 3] = ((alpha + count / 2) / count) as u8;
        }
    }
    out
}

/// One `RT_ICON`: a `BITMAPINFOHEADER` over bottom-up BGRA and an AND mask.
///
/// The mask is all zeros and is not optional. A 32-bit icon's transparency is
/// its alpha channel, but the structure predates alpha and every consumer still
/// reads past the colour bits — a resource one mask short is a resource whose
/// last rows are somebody else's memory.
fn image(side: u32, rgba: &[u8]) -> Vec<u8> {
    let stride = mask_stride(side);
    let mask = (stride * side) as usize;
    let colour = (side as usize) * (side as usize) * 4;
    let mut out = Vec::with_capacity(40 + colour + mask);
    out.extend_from_slice(&40u32.to_le_bytes()); // biSize
    out.extend_from_slice(&side.to_le_bytes()); // biWidth
    // Colour rows and mask rows in one bitmap, which is what doubles it.
    out.extend_from_slice(&(side * 2).to_le_bytes()); // biHeight
    out.extend_from_slice(&1u16.to_le_bytes()); // biPlanes
    out.extend_from_slice(&32u16.to_le_bytes()); // biBitCount
    out.extend_from_slice(&0u32.to_le_bytes()); // biCompression = BI_RGB
    out.extend_from_slice(&((colour + mask) as u32).to_le_bytes()); // biSizeImage
    out.extend_from_slice(&[0u8; 16]); // resolution, palette: none of it read
    for y in (0..side).rev() {
        for x in 0..side {
            let at = ((y * side + x) * 4) as usize;
            out.extend_from_slice(&[rgba[at + 2], rgba[at + 1], rgba[at], rgba[at + 3]]);
        }
    }
    out.resize(out.len() + mask, 0);
    out
}

/// Bytes per row of a 1-bit AND mask: whole 32-bit words, like every DIB row.
fn mask_stride(side: u32) -> u32 {
    side.div_ceil(32) * 4
}

/// The `RT_GROUP_ICON` directory naming each image by its resource id.
///
/// The sizes here are `u8`, which is the format's own ceiling and the reason
/// 256 is spelled `0`. Nothing in [`SIDES`] reaches it, and a size that did
/// would need that spelling rather than a wider field.
fn group(icons: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::with_capacity(6 + icons.len() * 14);
    out.extend_from_slice(&0u16.to_le_bytes()); // reserved
    out.extend_from_slice(&1u16.to_le_bytes()); // type: icon
    out.extend_from_slice(&(icons.len() as u16).to_le_bytes());
    for (i, (side, bytes)) in icons.iter().enumerate() {
        let byte = u8::try_from(*side).unwrap_or(0);
        out.extend_from_slice(&[byte, byte, 0, 0]); // width, height, colours, pad
        out.extend_from_slice(&1u16.to_le_bytes()); // planes
        out.extend_from_slice(&32u16.to_le_bytes()); // bits
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&(i as u16 + 1).to_le_bytes()); // the RT_ICON id
    }
    out
}

// ---------------------------------------------------------------- VS_VERSIONINFO

/// A `VS_VERSIONINFO` node: three counts, a UTF-16 key, a value and children,
/// each part starting on a four-byte boundary.
///
/// `wValueLength` counts **bytes** for a binary value and **UTF-16 units** for a
/// text one, which is the format's one genuine trap — the same number means two
/// things one field apart. Every node is padded to four here and `wLength`
/// covers the padding, so a reader that rounds up and one that does not step
/// identically.
fn node(key: &str, text: bool, value: &[u8], children: &[u8]) -> Vec<u8> {
    let units = if text { value.len() / 2 } else { value.len() };
    let mut out = Vec::with_capacity(6 + key.len() * 2 + 2 + value.len() + children.len() + 12);
    out.extend_from_slice(&0u16.to_le_bytes()); // wLength, patched below
    out.extend_from_slice(&(units as u16).to_le_bytes());
    out.extend_from_slice(&u16::from(text).to_le_bytes());
    out.extend_from_slice(&utf16(key));
    pad4(&mut out);
    out.extend_from_slice(value);
    pad4(&mut out);
    out.extend_from_slice(children);
    pad4(&mut out);
    let len = out.len() as u16;
    out[..2].copy_from_slice(&len.to_le_bytes());
    out
}

/// NUL-terminated UTF-16LE. `encode_utf16` is lossless for anything a manifest
/// can hold, and a title with an emoji in it is a title.
fn utf16(text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len() * 2 + 2);
    for unit in text.encode_utf16().chain(std::iter::once(0)) {
        out.extend_from_slice(&unit.to_le_bytes());
    }
    out
}

fn pad4(out: &mut Vec<u8>) {
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
}

/// The whole `RT_VERSION` blob.
fn version_info(marks: &Marks) -> Vec<u8> {
    let q = marks.quad();
    let mut fixed = Vec::with_capacity(52);
    for word in [
        0xfeef_04bdu32,                          // dwSignature
        0x0001_0000,                             // dwStrucVersion
        u32::from(q[0]) << 16 | u32::from(q[1]), // dwFileVersionMS
        u32::from(q[2]) << 16 | u32::from(q[3]), // dwFileVersionLS
        u32::from(q[0]) << 16 | u32::from(q[1]), // dwProductVersionMS
        u32::from(q[2]) << 16 | u32::from(q[3]), // dwProductVersionLS
        0x0000_003f,                             // dwFileFlagsMask
        0,                                       // dwFileFlags: released
        0x0004_0004,                             // dwFileOS: VOS_NT_WINDOWS32
        1,                                       // dwFileType: VFT_APP
        0,                                       // dwFileSubtype
        0,                                       // dwFileDateMS
        0,                                       // dwFileDateLS
    ] {
        fixed.extend_from_slice(&word.to_le_bytes());
    }

    let mut strings = Vec::new();
    for (key, value) in marks.strings() {
        strings.extend_from_slice(&node(key, true, &utf16(&value), &[]));
    }
    // The key is language and codepage as eight hex digits — the one place in
    // this format where a number is spelled as text.
    let table = node(&format!("{LANG:04x}{CODEPAGE:04x}"), true, &[], &strings);
    let string_info = node("StringFileInfo", true, &[], &table);

    let mut translation = Vec::with_capacity(4);
    translation.extend_from_slice(&(LANG as u16).to_le_bytes());
    translation.extend_from_slice(&(CODEPAGE as u16).to_le_bytes());
    let var = node("Translation", false, &translation, &[]);
    let var_info = node("VarFileInfo", true, &[], &var);

    let mut children = string_info;
    children.extend_from_slice(&var_info);
    node("VS_VERSION_INFO", false, &fixed, &children)
}

// -------------------------------------------------------------- the resource tree

/// Build `.rsrc$01` (directories and leaf records) and `.rsrc$02` (the bytes),
/// plus the offsets in the first that need relocating against the second.
///
/// `leaves` must already be sorted by `(type, id)`: a resource directory is
/// binary-searched by the loader and by every tool that reads one, so an
/// out-of-order entry is a resource nothing can find.
fn tree(leaves: &[(u32, u32, &[u8])]) -> (Vec<u8>, Vec<u8>, Vec<usize>) {
    let mut types: Vec<u32> = leaves.iter().map(|(t, _, _)| *t).collect();
    types.dedup();

    // Three levels of directory, then the leaf records, so every offset below is
    // known before a byte is written.
    let dir = |n: usize| 16 + n * 8;
    let mut at = dir(types.len());
    let mut name_dirs = Vec::new();
    for ty in &types {
        name_dirs.push(at);
        at += dir(leaves.iter().filter(|(t, _, _)| t == ty).count());
    }
    let mut lang_dirs = Vec::new();
    for _ in leaves {
        lang_dirs.push(at);
        at += dir(1);
    }
    let records = at;

    let mut out = Vec::new();
    let header = |out: &mut Vec<u8>, count: usize| {
        out.extend_from_slice(&[0u8; 12]); // characteristics, stamp, version
        out.extend_from_slice(&0u16.to_le_bytes()); // named entries: never any
        out.extend_from_slice(&(count as u16).to_le_bytes());
    };
    // Level 1: by type.
    header(&mut out, types.len());
    for (i, ty) in types.iter().enumerate() {
        out.extend_from_slice(&ty.to_le_bytes());
        out.extend_from_slice(&(0x8000_0000u32 | name_dirs[i] as u32).to_le_bytes());
    }
    // Level 2: by id, per type.
    for ty in &types {
        let of_type: Vec<usize> = (0..leaves.len()).filter(|&i| leaves[i].0 == *ty).collect();
        header(&mut out, of_type.len());
        for i in of_type {
            out.extend_from_slice(&leaves[i].1.to_le_bytes());
            out.extend_from_slice(&(0x8000_0000u32 | lang_dirs[i] as u32).to_le_bytes());
        }
    }
    // Level 3: one language, pointing at the leaf record rather than a directory.
    for i in 0..leaves.len() {
        header(&mut out, 1);
        out.extend_from_slice(&LANG.to_le_bytes());
        out.extend_from_slice(&((records + i * 16) as u32).to_le_bytes());
    }

    let mut blobs = Vec::new();
    let mut fixups = Vec::new();
    for (_, _, bytes) in leaves {
        fixups.push(out.len());
        // The RVA the linker will write. What is here now is the offset inside
        // `.rsrc$02`, which is the relocation's addend.
        out.extend_from_slice(&(blobs.len() as u32).to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        out.extend_from_slice(&[0u8; 8]); // codepage, reserved
        blobs.extend_from_slice(bytes);
        while !blobs.len().is_multiple_of(8) {
            blobs.push(0);
        }
    }
    (out, blobs, fixups)
}

// -------------------------------------------------------------------- the object

/// `IMAGE_REL_AMD64_ADDR32NB` — a 32-bit address relative to the image base,
/// which is exactly what an RVA is. x86-64 only, which is the only Windows this
/// tree ships.
const ADDR32NB: u16 = 0x0003;

/// The COFF object: two grouped sections, two section symbols, one relocation
/// per leaf.
///
/// `TimeDateStamp` is zero for `ship.rs`'s reason — an artifact that differs by
/// when it was built cannot be compared with one that was built elsewhere.
fn coff(directory: &[u8], blobs: &[u8], fixups: &[usize]) -> Vec<u8> {
    let headers = 20 + 40 * 2;
    let dir_at = headers;
    let reloc_at = dir_at + directory.len();
    let blob_at = reloc_at + fixups.len() * 10;
    let symbols_at = blob_at + blobs.len();

    let mut out = Vec::with_capacity(symbols_at + 18 * 2 + 4);
    out.extend_from_slice(&0x8664u16.to_le_bytes()); // Machine: AMD64
    out.extend_from_slice(&2u16.to_le_bytes()); // NumberOfSections
    out.extend_from_slice(&0u32.to_le_bytes()); // TimeDateStamp
    out.extend_from_slice(&(symbols_at as u32).to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes()); // NumberOfSymbols
    out.extend_from_slice(&0u16.to_le_bytes()); // SizeOfOptionalHeader
    out.extend_from_slice(&0u16.to_le_bytes()); // Characteristics

    // CNT_INITIALIZED_DATA | ALIGN_4BYTES | MEM_READ. The `$` suffixes are what
    // the linker sorts on: `.rsrc$01` lands before `.rsrc$02` in one `.rsrc`,
    // which is the whole reason the offsets above are laid out that way.
    let flags = 0x4030_0040u32;
    for (name, size, at, relocs, reloc_at) in [
        (b".rsrc$01", directory.len(), dir_at, fixups.len(), reloc_at),
        (b".rsrc$02", blobs.len(), blob_at, 0, 0),
    ] {
        out.extend_from_slice(name);
        out.extend_from_slice(&0u32.to_le_bytes()); // VirtualSize
        out.extend_from_slice(&0u32.to_le_bytes()); // VirtualAddress
        out.extend_from_slice(&(size as u32).to_le_bytes());
        out.extend_from_slice(&(at as u32).to_le_bytes());
        out.extend_from_slice(&(reloc_at as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // PointerToLinenumbers
        out.extend_from_slice(&(relocs as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // NumberOfLinenumbers
        out.extend_from_slice(&flags.to_le_bytes());
    }

    out.extend_from_slice(directory);
    for at in fixups {
        out.extend_from_slice(&(*at as u32).to_le_bytes()); // VirtualAddress
        out.extend_from_slice(&1u32.to_le_bytes()); // SymbolTableIndex: .rsrc$02
        out.extend_from_slice(&ADDR32NB.to_le_bytes());
    }
    out.extend_from_slice(blobs);

    // Two static section symbols and nothing else. Both names are exactly eight
    // bytes, which is the short form and why there is no string table to fill.
    for (i, name) in [b".rsrc$01", b".rsrc$02"].into_iter().enumerate() {
        out.extend_from_slice(name);
        out.extend_from_slice(&0u32.to_le_bytes()); // Value
        out.extend_from_slice(&(i as i16 + 1).to_le_bytes()); // SectionNumber
        out.extend_from_slice(&0u16.to_le_bytes()); // Type
        out.push(3); // IMAGE_SYM_CLASS_STATIC
        out.push(0); // NumberOfAuxSymbols
    }
    // An empty string table is still four bytes saying it is empty.
    out.extend_from_slice(&4u32.to_le_bytes());
    out
}

// -------------------------------------------------------------------- reading back

/// What a linked executable was found to carry.
#[derive(Debug)]
pub struct Found {
    /// `(type, id, bytes)` for every leaf, in the order the tree lists them.
    pub leaves: Vec<(u32, u32, Vec<u8>)>,
}

impl Found {
    fn of(&self, ty: u32, id: u32) -> Option<&[u8]> {
        self.leaves
            .iter()
            .find(|(t, i, _)| *t == ty && *i == id)
            .map(|(_, _, b)| b.as_slice())
    }
}

/// Walk a linked PE's resource directory back out of the file.
///
/// The point of reading rather than trusting: between [`object`] and here sit
/// `link.exe`, a section merge and a relocation, none of which this tree
/// controls. A gate that compares what was asked for against what a *reader*
/// finds in the artifact is the only one that covers them.
///
/// # Errors
/// A file that is not a PE, or one with no resource directory — which is the
/// pre-M73 artifact and is the failure this exists to name.
pub fn read(exe: &[u8]) -> Result<Found> {
    let at = |off: usize| -> Result<u32> {
        let bytes = exe
            .get(off..off + 4)
            .context("truncated PE: a header ran off the end of the file")?;
        Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    };
    ensure!(exe.starts_with(b"MZ"), "not a PE image");
    let pe = at(0x3c)? as usize;
    ensure!(exe.get(pe..pe + 4) == Some(b"PE\0\0"), "not a PE image");
    let sections = u16::from_le_bytes([exe[pe + 6], exe[pe + 7]]) as usize;
    let optional = u16::from_le_bytes([exe[pe + 20], exe[pe + 21]]) as usize;
    let magic = u16::from_le_bytes([exe[pe + 24], exe[pe + 25]]);
    // PE32+ puts eight more bytes in front of the directory count than PE32.
    let count_at = pe + 24 + if magic == 0x20b { 108 } else { 92 };
    let directories = at(count_at)? as usize;
    // Entry 2 of the data directory is the resource table.
    ensure!(
        directories > 2,
        "this image declares {directories} data directories and the resource table is the third"
    );
    let table = pe + 24 + if magic == 0x20b { 112 } else { 96 };
    let (root, size) = (at(table + 16)?, at(table + 20)?);
    ensure!(
        root != 0 && size != 0,
        "no resource directory: this executable carries no icon and no version (§6 M73)"
    );

    // RVA to file offset, through the section table.
    let table_at = pe + 24 + optional;
    let mut map = Vec::with_capacity(sections);
    for i in 0..sections {
        let s = table_at + i * 40;
        map.push((at(s + 12)?, at(s + 16)?, at(s + 20)?));
    }
    let file_at = |rva: u32| -> Result<usize> {
        for (va, size, raw) in &map {
            if rva >= *va && rva < va + size {
                return Ok((raw + (rva - va)) as usize);
            }
        }
        bail!("resource RVA {rva:#x} is in no section")
    };

    let base = file_at(root)?;
    let mut leaves = Vec::new();
    walk(exe, base, base, 0, (0, 0), &mut leaves, &file_at)?;
    Ok(Found { leaves })
}

/// One directory level. The three levels differ only in what an entry's name
/// *means* — type, then id, then language — so `carry` collects the first two on
/// the way down and the third has nothing left to name but the leaf.
fn walk(
    exe: &[u8],
    base: usize,
    here: usize,
    depth: usize,
    carry: (u32, u32),
    out: &mut Vec<(u32, u32, Vec<u8>)>,
    file_at: &dyn Fn(u32) -> Result<usize>,
) -> Result<()> {
    let word = |at: usize| -> Result<u32> {
        let b = exe
            .get(at..at + 4)
            .context("truncated resource directory")?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    ensure!(exe.len() > here + 16, "truncated resource directory");
    let named = u16::from_le_bytes([exe[here + 12], exe[here + 13]]) as usize;
    let ids = u16::from_le_bytes([exe[here + 14], exe[here + 15]]) as usize;
    for i in 0..named + ids {
        let entry = here + 16 + i * 8;
        let (name, offset) = (word(entry)?, word(entry + 4)?);
        let carry = match depth {
            0 => (name, carry.1),
            1 => (carry.0, name),
            _ => carry,
        };
        if offset & 0x8000_0000 != 0 {
            let to = base + (offset & 0x7fff_ffff) as usize;
            walk(exe, base, to, depth + 1, carry, out, file_at)?;
            continue;
        }
        // A leaf record: an RVA, a size, and two fields nothing reads.
        let record = base + offset as usize;
        let (rva, size) = (word(record)?, word(record + 4)?);
        let from = file_at(rva)?;
        let bytes = exe
            .get(from..from + size as usize)
            .with_context(|| format!("resource at {rva:#x} runs {size} bytes past the file"))?;
        out.push((carry.0, carry.1, bytes.to_vec()));
    }
    Ok(())
}

/// Grade a linked executable against what [`object`] put in it.
///
/// # Errors
/// Anything missing, anything different, and the pre-M73 case of no resource
/// directory at all — each by name.
pub fn verify(exe: &[u8], built: &Built, marks: &Marks) -> Result<usize> {
    let found = read(exe)?;
    // Only what went in is looked for: an iconless build carries the version
    // alone (§6 M81), and demanding a group there would fail a link that did
    // exactly what it was asked.
    if !built.icons.is_empty() {
        let group = found.of(RT_GROUP_ICON, 1).context(
            "no RT_GROUP_ICON id 1: Explorer asks for the lowest-numbered group and would \
                  find none",
        )?;
        ensure!(
            group == self::group(&built.icons),
            "the icon directory in the executable is not the one that went in: {}",
            differ(group, &self::group(&built.icons))
        );
        for (i, (side, want)) in built.icons.iter().enumerate() {
            let id = i as u32 + 1;
            let have = found.of(RT_ICON, id).with_context(|| {
                format!("no RT_ICON id {id} — the {side}px image did not survive")
            })?;
            ensure!(
                have == want.as_slice(),
                "RT_ICON id {id} ({side}px) is not the image that went in: {}",
                differ(have, want)
            );
        }
    }
    let version = found
        .of(RT_VERSION, 1)
        .context("no RT_VERSION id 1: the Details tab would be blank")?;
    ensure!(
        version == built.version,
        "the version record in the executable is not the one that went in: {}",
        differ(version, &built.version)
    );
    // The strings, read back out of the blob rather than compared to it: the
    // comparison above proves the bytes crossed, and this proves they *say* what
    // the manifest does, which is the claim a player checks.
    for (key, value) in marks.strings() {
        let want = utf16(&value);
        ensure!(
            find(version, &utf16(key)) && find(version, &want),
            "the version record does not carry `{key} = {value}`"
        );
    }
    Ok(found.leaves.len())
}

/// How two blobs differ, in the terms that tell a truncation from a corruption
/// apart — the first is a link that dropped something and the second is not.
fn differ(have: &[u8], want: &[u8]) -> String {
    if have.len() != want.len() {
        return format!("{} bytes against {}", have.len(), want.len());
    }
    let at = have.iter().zip(want).position(|(a, b)| a != b);
    let count = have.iter().zip(want).filter(|(a, b)| a != b).count();
    format!(
        "same {} bytes, {count} of them different, first at {}",
        have.len(),
        at.unwrap_or(0)
    )
}

fn find(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marks() -> Marks {
        Marks {
            title: "Falling Blocks".to_owned(),
            version: Some(gg_core::config::project::Version::parse("1.2.3-rc4").unwrap()),
            slug: "falling-blocks".to_owned(),
            exe: "falling-blocks.exe".to_owned(),
            engine: "GGEngine 0.1.0".to_owned(),
        }
    }

    /// A gradient with a hard transparent quadrant, so the premultiply matters.
    fn source(side: u32) -> Vec<u8> {
        let mut rgba = Vec::with_capacity((side * side * 4) as usize);
        for y in 0..side {
            for x in 0..side {
                let clear = x < side / 2 && y < side / 2;
                rgba.extend_from_slice(&[
                    (x * 255 / (side - 1)) as u8,
                    (y * 255 / (side - 1)) as u8,
                    0x40,
                    if clear { 0 } else { 255 },
                ]);
            }
        }
        rgba
    }

    /// The claim the whole size chain rests on: 96 divides by every side we ask
    /// for, so no filter is designed and no ratio is approximate.
    #[test]
    fn every_size_an_os_asks_for_is_an_exact_reduction_of_the_one_file() {
        for side in SIDES {
            assert!(
                gg_core::config::icon::SIDE.is_multiple_of(side),
                "{side} does not divide the checked-in icon"
            );
        }
    }

    /// A flat block reduces to its own colour, and a transparent one does not
    /// bleed into its neighbour — which is the only thing the premultiply buys
    /// and the thing a straight average gets wrong.
    #[test]
    fn a_box_filter_keeps_a_flat_colour_and_an_edge_keeps_its_side() {
        let flat: Vec<u8> = std::iter::repeat_n([9u8, 8, 7, 255], 16)
            .flatten()
            .collect();
        assert_eq!(reduce(4, &flat, 1), vec![9, 8, 7, 255]);
        // Two pixels opaque red, two fully transparent green: the colour is red
        // at half coverage, not a red/green average.
        let edge = [
            255, 0, 0, 255, 255, 0, 0, 255, //
            0, 255, 0, 0, 0, 255, 0, 0,
        ];
        assert_eq!(reduce(2, &edge, 1), vec![255, 0, 0, 128]);
        // Nothing visible has no colour to keep.
        assert_eq!(reduce(2, &[0; 16], 1), vec![0, 0, 0, 0]);
    }

    /// The structure a consumer walks: header, colour bits bottom-up, and a mask
    /// nobody looks at that everybody reads past.
    #[test]
    fn an_icon_image_is_a_doubled_bitmap_with_the_mask_it_promises() {
        let side = 4;
        let rgba = source(side);
        let bytes = image(side, &rgba);
        assert_eq!(&bytes[..4], &40u32.to_le_bytes());
        assert_eq!(&bytes[4..8], &side.to_le_bytes());
        assert_eq!(&bytes[8..12], &(side * 2).to_le_bytes(), "doubled height");
        assert_eq!(&bytes[14..16], &32u16.to_le_bytes());
        let colour = (side * side * 4) as usize;
        let mask = (mask_stride(side) * side) as usize;
        assert_eq!(bytes.len(), 40 + colour + mask);
        assert!(
            bytes[40 + colour..].iter().all(|b| *b == 0),
            "mask is clear"
        );
        // The first colour row in the file is the *last* row of the image, and
        // BGRA rather than RGBA.
        let last = ((side - 1) * side * 4) as usize;
        assert_eq!(
            &bytes[40..44],
            &[rgba[last + 2], rgba[last + 1], rgba[last], rgba[last + 3]]
        );
    }

    /// Every node's length covers itself, so a reader that steps by `wLength`
    /// lands on the next node and finishes exactly at the end.
    #[test]
    fn a_version_record_is_a_tree_a_reader_can_step_through() {
        let marks = marks();
        let blob = version_info(&marks);
        let len = u16::from_le_bytes([blob[0], blob[1]]) as usize;
        assert_eq!(len, blob.len(), "the root covers the whole record");
        assert!(len.is_multiple_of(4), "every node is four-aligned");
        // The fixed record is at a known place: six bytes of counts, the key,
        // then padding to four.
        let key = utf16("VS_VERSION_INFO");
        let fixed = 6 + key.len() + (4 - (6 + key.len()) % 4) % 4;
        assert_eq!(
            &blob[fixed..fixed + 4],
            &0xfeef_04bdu32.to_le_bytes(),
            "VS_FIXEDFILEINFO signature"
        );
        assert_eq!(
            u16::from_le_bytes([blob[2], blob[3]]) as usize,
            52,
            "wValueLength of a binary value is bytes"
        );
        // A text node's is UTF-16 units, which is the trap this pair pins.
        let one = node("k", true, &utf16("ab"), &[]);
        assert_eq!(
            u16::from_le_bytes([one[2], one[3]]),
            3,
            "\"ab\" and its NUL"
        );
        assert_eq!(u16::from_le_bytes([one[4], one[5]]), 1, "wType: text");
    }

    /// The suffix reaches the strings and never the numbers, and an unversioned
    /// build says `0.0.0.0` in both places rather than in neither.
    #[test]
    fn the_version_a_shell_reads_is_the_one_the_manifest_wrote() {
        let blob = version_info(&marks());
        assert!(find(&blob, &utf16("1.2.3-rc4")));
        assert!(find(&blob, &utf16("Falling Blocks")));
        assert!(find(&blob, &utf16("falling-blocks.exe")));
        // dwFileVersionMS/LS: 1.2 and 3.0, the suffix nowhere near them.
        assert!(find(&blob, &(1u32 << 16 | 2).to_le_bytes()));
        assert!(find(&blob, &(3u32 << 16).to_le_bytes()));
        let bare = Marks {
            version: None,
            ..marks()
        };
        assert!(find(&version_info(&bare), &utf16("0.0.0.0")));
    }

    /// The object as the linker will read it, without needing one: two grouped
    /// sections, one relocation per leaf, and every relocation aimed at the
    /// symbol for the section the bytes are actually in.
    #[test]
    fn the_object_declares_two_sections_and_relocates_every_leaf() {
        let icon = gg_core::config::icon::encode(
            gg_core::config::icon::SIDE,
            &source(gg_core::config::icon::SIDE),
        );
        let built = object(Some(&icon), &marks()).unwrap();
        let obj = &built.object;
        assert_eq!(&obj[..2], &0x8664u16.to_le_bytes(), "AMD64");
        assert_eq!(u16::from_le_bytes([obj[2], obj[3]]), 2, "two sections");
        assert_eq!(&obj[4..8], &[0; 4], "no clock in the bytes");
        assert_eq!(&obj[20..28], b".rsrc$01");
        assert_eq!(&obj[60..68], b".rsrc$02");
        let relocs = u16::from_le_bytes([obj[52], obj[53]]) as usize;
        assert_eq!(
            relocs,
            SIDES.len() + 2,
            "an icon each, the group, the version"
        );
        assert_eq!(
            u16::from_le_bytes([obj[92], obj[93]]),
            0,
            "the blob section relocates nothing"
        );
        // PointerToRelocations, 24 bytes into the first section header.
        let at = u32::from_le_bytes([obj[44], obj[45], obj[46], obj[47]]) as usize;
        for i in 0..relocs {
            let r = at + i * 10;
            assert_eq!(
                u32::from_le_bytes([obj[r + 4], obj[r + 5], obj[r + 6], obj[r + 7]]),
                1,
                "relocation {i} names the second section's symbol"
            );
            assert_eq!(u16::from_le_bytes([obj[r + 8], obj[r + 9]]), ADDR32NB);
        }
    }

    /// The claim no amount of reading this file can make: **`link.exe` accepts
    /// it**, merges the two grouped sections, applies the relocations and fills
    /// the resource data directory.
    ///
    /// Everything else here grades the bytes against the specification, which is
    /// worth nothing if the specification and the linker disagree. `xtask ship`
    /// does the same round trip on the real artifact and is manual (§6 M41), so
    /// without this the one risk in the milestone would be checked only when
    /// somebody released something. An empty `main` links in about a second,
    /// which is what makes that affordable in the fast tier.
    ///
    /// Windows only, because a `.rsrc` section is.
    #[test]
    #[cfg_attr(not(windows), ignore = "a PE resource section is Windows'")]
    fn a_hand_written_object_is_one_the_real_linker_accepts() {
        let dir = std::env::temp_dir().join(format!("gg-rsrc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let icon = gg_core::config::icon::encode(
            gg_core::config::icon::SIDE,
            &source(gg_core::config::icon::SIDE),
        );
        let marks = marks();
        let built = object(Some(&icon), &marks).unwrap();
        let obj = dir.join("marks.obj");
        std::fs::write(&obj, &built.object).unwrap();
        let main = dir.join("main.rs");
        std::fs::write(&main, "fn main() {}\n").unwrap();
        let exe = dir.join("marked.exe");
        let out = std::process::Command::new("rustc")
            .arg(&main)
            .arg("-o")
            .arg(&exe)
            .arg(format!("-Clink-arg={}", obj.display()))
            .output()
            .expect("rustc");
        assert!(
            out.status.success(),
            "the linker refused the object:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let bytes = std::fs::read(&exe).unwrap();
        let leaves = verify(&bytes, &built, &marks).expect("what the linker produced");
        assert_eq!(leaves, SIDES.len() + 2);
        // And the same binary without it has none, which is what says the check
        // above is reading the object rather than something every PE carries.
        let bare = dir.join("bare.exe");
        assert!(
            std::process::Command::new("rustc")
                .arg(&main)
                .arg("-o")
                .arg(&bare)
                .output()
                .expect("rustc")
                .status
                .success()
        );
        let refused = read(&std::fs::read(&bare).unwrap())
            .unwrap_err()
            .to_string();
        assert!(refused.contains("no resource directory"), "{refused}");

        // And the iconless manifest through the same linker (§6 M81): the
        // version leaf alone, which is what a `version` with no `icon` used to
        // drop on the floor along with this whole check.
        let only = object(None, &marks).unwrap();
        assert!(only.icons.is_empty());
        let obj = dir.join("version-only.obj");
        std::fs::write(&obj, &only.object).unwrap();
        let exe = dir.join("version-only.exe");
        let out = std::process::Command::new("rustc")
            .arg(&main)
            .arg("-o")
            .arg(&exe)
            .arg(format!("-Clink-arg={}", obj.display()))
            .output()
            .expect("rustc");
        assert!(
            out.status.success(),
            "the linker refused the iconless object:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let found = verify(&std::fs::read(&exe).unwrap(), &only, &marks)
            .expect("a version record with no picture beside it");
        assert_eq!(found, 1, "the version leaf and nothing else");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// One input, one object — the property `ship.rs`'s archive already has and
    /// the reason `TimeDateStamp` is zero.
    #[test]
    fn an_object_is_a_function_of_its_input() {
        // A 4px source cannot make the chain, and says which size stopped it.
        let icon = gg_core::config::icon::encode(4, &source(4));
        let refused = object(Some(&icon), &marks()).unwrap_err().to_string();
        assert!(refused.contains("16px"), "{refused}");
        let whole = gg_core::config::icon::encode(
            gg_core::config::icon::SIDE,
            &source(gg_core::config::icon::SIDE),
        );
        assert_eq!(
            object(Some(&whole), &marks()).unwrap().object,
            object(Some(&whole), &marks()).unwrap().object
        );
    }
}
