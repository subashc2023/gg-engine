//! RGBA8 PNG read/write — the reference-image format (§4.10). Deliberately
//! strict: references are written by `bless` as 8-bit RGBA and read back the
//! same way; anything else in the references directory is an error, not a
//! conversion.

use std::path::Path;

pub fn write(path: &Path, pixels: &[u8], extent: (u32, u32)) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, encode(pixels, extent)?)?;
    Ok(())
}

/// The same PNG, in memory — what the HTML report embeds so the artifact is one
/// file a reviewer can open or attach without carrying a directory with it.
pub fn encode(pixels: &[u8], extent: (u32, u32)) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        pixels.len() == extent.0 as usize * extent.1 as usize * 4,
        "pixel buffer does not match {}x{} RGBA8",
        extent.0,
        extent.1
    );
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, extent.0, extent.1);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    // Finish explicitly: the IEND chunk is otherwise written by `Drop`, where a
    // write error has nowhere to go and a truncated PNG rides into the report.
    let mut writer = encoder.write_header()?;
    writer.write_image_data(pixels)?;
    writer.finish()?;
    Ok(out)
}

pub fn read(path: &Path) -> anyhow::Result<(Vec<u8>, (u32, u32))> {
    let file = std::fs::File::open(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
    let decoder = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = decoder.read_info()?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| anyhow::anyhow!("{}: output size overflows", path.display()))?;
    let mut buf = vec![0u8; size];
    let info = reader.next_frame(&mut buf)?;
    anyhow::ensure!(
        info.color_type == png::ColorType::Rgba && info.bit_depth == png::BitDepth::Eight,
        "{}: references are 8-bit RGBA, this is {:?}/{:?}",
        path.display(),
        info.color_type,
        info.bit_depth
    );
    buf.truncate(info.buffer_size());
    Ok((buf, (info.width, info.height)))
}
