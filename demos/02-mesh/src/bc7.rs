//! A minimal BC7 encoder — **mode 6 only** — so demo 02 has real
//! block-compressed texels to upload before the asset pipeline exists.
//!
//! Mode 6 is the one BC7 mode with no partitioning, no rotation and no index
//! selection: a single subset, RGBA endpoints at 7 bits plus a P-bit each, and
//! 4-bit indices. That makes it ~100 lines here while still producing bytes
//! the hardware block decoder must interpret — which is the point, since what
//! M4A is proving is the *upload and sampling* path for a compressed format,
//! not compression quality.
//!
//! `ggc` replaces this with `intel_tex_2` at M9 (§2, Texture pipeline row;
//! §4.6). Nothing outside this demo should grow a dependency on it.

/// Interpolation weights for 4-bit indices, from the BC7 specification.
const WEIGHTS: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

/// Bytes one BC7 block occupies (4x4 texels).
pub const BLOCK_BYTES: usize = 16;

/// Encode a whole RGBA8 image to BC7. `extent` must be a multiple of 4 in
/// both axes — partial blocks are a real format feature and a real amount of
/// code, and the demo's textures are powers of two.
pub fn encode(rgba: &[u8], extent: (u32, u32)) -> Vec<u8> {
    assert!(
        extent.0.is_multiple_of(4) && extent.1.is_multiple_of(4),
        "BC7 encodes whole 4x4 blocks; {extent:?} is not block-aligned"
    );
    assert_eq!(rgba.len(), (extent.0 * extent.1 * 4) as usize);
    let (bw, bh) = (extent.0 / 4, extent.1 / 4);
    let mut out = Vec::with_capacity((bw * bh) as usize * BLOCK_BYTES);
    for by in 0..bh {
        for bx in 0..bw {
            let mut texels = [[0u8; 4]; 16];
            for ty in 0..4u32 {
                for tx in 0..4u32 {
                    let x = bx * 4 + tx;
                    let y = by * 4 + ty;
                    let at = ((y * extent.0 + x) * 4) as usize;
                    texels[(ty * 4 + tx) as usize].copy_from_slice(&rgba[at..at + 4]);
                }
            }
            out.extend_from_slice(&encode_block(&texels));
        }
    }
    out
}

/// Encode one 4x4 block as a BC7 mode-6 block.
fn encode_block(texels: &[[u8; 4]; 16]) -> [u8; BLOCK_BYTES] {
    // Endpoints are the per-channel bounding box of the block. Optimal for the
    // flat and two-tone blocks this demo's textures are made of, and honest
    // (if unremarkable) for anything else.
    let mut lo = [255u8; 4];
    let mut hi = [0u8; 4];
    for t in texels {
        for c in 0..4 {
            lo[c] = lo[c].min(t[c]);
            hi[c] = hi[c].max(t[c]);
        }
    }
    // 7-bit endpoint + P-bit: the decoder reconstructs `(q << 1) | p`, so
    // pinning both P-bits to 1 costs at most one code point of accuracy and
    // removes a search dimension.
    let mut q0 = [0u8; 4];
    let mut q1 = [0u8; 4];
    for c in 0..4 {
        q0[c] = lo[c] >> 1;
        q1[c] = hi[c] >> 1;
    }
    let e0 = expand(q0);
    let e1 = expand(q1);

    let mut indices = [0u8; 16];
    for (i, t) in texels.iter().enumerate() {
        indices[i] = best_index(t, &e0, &e1);
    }

    // The anchor index carries one bit fewer (its high bit is implicitly 0),
    // so a block whose first texel sits past the midpoint has to be flipped.
    let (q0, q1, indices) = if indices[0] >= 8 {
        let flipped = indices.map(|i| 15 - i);
        (q1, q0, flipped)
    } else {
        (q0, q1, indices)
    };

    let mut bits = [0u8; BLOCK_BYTES];
    let mut at = 0usize;
    put(&mut bits, &mut at, 1 << 6, 7); // mode 6: six zeros then a one
    for c in 0..4 {
        put(&mut bits, &mut at, u32::from(q0[c]), 7);
        put(&mut bits, &mut at, u32::from(q1[c]), 7);
    }
    put(&mut bits, &mut at, 1, 1); // P0
    put(&mut bits, &mut at, 1, 1); // P1
    put(&mut bits, &mut at, u32::from(indices[0]), 3);
    for index in &indices[1..] {
        put(&mut bits, &mut at, u32::from(*index), 4);
    }
    debug_assert_eq!(at, 128);
    bits
}

/// 7-bit endpoint plus its P-bit (always 1 here) to the 8-bit value the
/// decoder interpolates between.
fn expand(q: [u8; 4]) -> [u8; 4] {
    q.map(|v| (v << 1) | 1)
}

fn best_index(texel: &[u8; 4], e0: &[u8; 4], e1: &[u8; 4]) -> u8 {
    let mut best = (u32::MAX, 0u8);
    for (i, w) in WEIGHTS.iter().enumerate() {
        let mut error = 0u32;
        for c in 0..4 {
            let v = interpolate(e0[c], e1[c], *w);
            let d = i32::from(v) - i32::from(texel[c]);
            error += (d * d) as u32;
        }
        if error < best.0 {
            best = (error, i as u8);
        }
    }
    best.1
}

fn interpolate(a: u8, b: u8, w: u32) -> u8 {
    (((64 - w) * u32::from(a) + w * u32::from(b) + 32) >> 6) as u8
}

/// Append `count` low bits of `value`, LSB-first — the order BC7 blocks are
/// read in.
fn put(bits: &mut [u8; BLOCK_BYTES], at: &mut usize, value: u32, count: usize) {
    for i in 0..count {
        if (value >> i) & 1 == 1 {
            bits[(*at + i) / 8] |= 1 << ((*at + i) % 8);
        }
    }
    *at += count;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode a mode-6 block the way hardware does, so the encoder is checked
    /// against the format rather than against itself.
    fn decode_block(bits: &[u8; BLOCK_BYTES]) -> [[u8; 4]; 16] {
        let mut at = 0usize;
        let mode = get(bits, &mut at, 7);
        assert_eq!(mode, 1 << 6, "not a mode-6 block");
        let mut q0 = [0u8; 4];
        let mut q1 = [0u8; 4];
        for c in 0..4 {
            q0[c] = get(bits, &mut at, 7) as u8;
            q1[c] = get(bits, &mut at, 7) as u8;
        }
        let p0 = get(bits, &mut at, 1) as u8;
        let p1 = get(bits, &mut at, 1) as u8;
        let e0 = q0.map(|v| (v << 1) | p0);
        let e1 = q1.map(|v| (v << 1) | p1);
        let mut out = [[0u8; 4]; 16];
        for (i, texel) in out.iter_mut().enumerate() {
            let index = get(bits, &mut at, if i == 0 { 3 } else { 4 }) as usize;
            for c in 0..4 {
                texel[c] = interpolate(e0[c], e1[c], WEIGHTS[index]);
            }
        }
        assert_eq!(at, 128);
        out
    }

    fn get(bits: &[u8; BLOCK_BYTES], at: &mut usize, count: usize) -> u32 {
        let mut value = 0u32;
        for i in 0..count {
            let bit = (bits[(*at + i) / 8] >> ((*at + i) % 8)) & 1;
            value |= u32::from(bit) << i;
        }
        *at += count;
        value
    }

    #[test]
    fn a_flat_block_round_trips_within_one_code_point() {
        let texels = [[37u8, 200, 9, 255]; 16];
        let decoded = decode_block(&encode_block(&texels));
        for t in &decoded {
            for c in 0..4 {
                assert!(
                    t[c].abs_diff(texels[0][c]) <= 1,
                    "flat block drifted: {t:?} vs {:?}",
                    texels[0]
                );
            }
        }
    }

    #[test]
    fn a_two_tone_block_keeps_both_tones_apart() {
        // The case the demo's checker texture is made of: two colors, no
        // gradient. Both must survive, and the anchor-index flip must keep the
        // block legal whichever tone lands in texel 0.
        for (a, b) in [
            ([15u8, 15, 15, 255], [241u8, 200, 33, 255]),
            ([241, 200, 33, 255], [15, 15, 15, 255]),
        ] {
            let mut texels = [a; 16];
            for (i, t) in texels.iter_mut().enumerate() {
                if i % 2 == 1 {
                    *t = b;
                }
            }
            let decoded = decode_block(&encode_block(&texels));
            for (i, t) in decoded.iter().enumerate() {
                let want = if i % 2 == 1 { b } else { a };
                for c in 0..4 {
                    assert!(
                        t[c].abs_diff(want[c]) <= 2,
                        "texel {i} channel {c}: {} vs {}",
                        t[c],
                        want[c]
                    );
                }
            }
        }
    }

    #[test]
    fn the_anchor_index_never_needs_its_missing_bit() {
        // Texel 0 deliberately at the far endpoint: without the flip its index
        // would be 15, which does not fit in the anchor's three bits.
        let mut texels = [[0u8, 0, 0, 255]; 16];
        texels[0] = [255, 255, 255, 255];
        let block = encode_block(&texels);
        let mut at = 7 + 56 + 2;
        assert!(get(&block, &mut at, 3) < 8);
        let decoded = decode_block(&block);
        assert!(
            decoded[0][0] > 250,
            "anchor texel decoded as {:?}",
            decoded[0]
        );
        assert!(
            decoded[1][0] < 5,
            "non-anchor texel decoded as {:?}",
            decoded[1]
        );
    }

    #[test]
    fn an_image_encodes_to_one_block_per_sixteen_texels() {
        let rgba = vec![64u8; 8 * 8 * 4];
        assert_eq!(encode(&rgba, (8, 8)).len(), 4 * BLOCK_BYTES);
    }
}
