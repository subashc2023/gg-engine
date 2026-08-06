//! What `shadow-bias` and `shadow-fit` both do to a frame before they can count
//! anything: reduce it to luminance, and find the pixels near a shadow edge.
//!
//! Shared because the two instruments ask different questions of the *same*
//! reduction — a mask taken from the reference leg, never from the test one,
//! since acne and a staircase are both high-gradient and would otherwise define
//! the band that hides them. Two copies of that argument is two places for the
//! thresholds to drift apart, which is what this module exists to prevent.

/// Output levels of disagreement that count as one. A single PCF tap is 1/9 of
/// the sun's contribution, which at these scenes' contrast is roughly 20 levels;
/// 8 catches half a tap and stays well clear of tonemapper rounding.
pub const DISAGREEMENT: i32 = 8;

/// Below this in both legs the pixel is background, not surface.
pub const BACKGROUND: i32 = 4;

/// Reference-luminance range across a 3x3 window that marks a shadow (or
/// geometric) edge. Well above the couple of levels a smooth gradient moves and
/// well under the ~190 a shadow boundary does.
pub const EDGE_GRADIENT: i32 = 20;

/// RGBA8 output reduced to per-pixel luminance.
///
/// Post-tonemap and therefore not linear, which is the point: the question is
/// what a viewer would see disagree, not what the radiance was. Rec. 709 weights
/// in fixed point, so the metric has no float rounding of its own to explain.
pub fn luminance(pixels: &[u8]) -> Vec<i32> {
    pixels
        .chunks_exact(4)
        .map(|p| (2126 * i32::from(p[0]) + 7152 * i32::from(p[1]) + 722 * i32::from(p[2])) / 10000)
        .collect()
}

/// Pixels within `radius` of an [`EDGE_GRADIENT`] edge in `image`.
///
/// The dilation is separable — a square structuring element is two 1-D max
/// passes, which is the difference between this being instant and quadratic.
pub fn near_edge(image: &[i32], extent: (u32, u32), radius: usize) -> Vec<bool> {
    let (w, h) = (extent.0 as usize, extent.1 as usize);
    let at = |x: usize, y: usize| image[y * w + x];
    let mut edge = vec![false; image.len()];
    for y in 0..h {
        for x in 0..w {
            let (mut lo, mut hi) = (i32::MAX, i32::MIN);
            for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
                for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                    lo = lo.min(at(nx, ny));
                    hi = hi.max(at(nx, ny));
                }
            }
            edge[y * w + x] = hi - lo > EDGE_GRADIENT;
        }
    }
    let mut wide = vec![false; edge.len()];
    for y in 0..h {
        for x in 0..w {
            let lo = x.saturating_sub(radius);
            let hi = (x + radius).min(w - 1);
            wide[y * w + x] = edge[y * w + lo..=y * w + hi].iter().any(|e| *e);
        }
    }
    let mut out = vec![false; edge.len()];
    for y in 0..h {
        for x in 0..w {
            let lo = y.saturating_sub(radius);
            let hi = (y + radius).min(h - 1);
            out[y * w + x] = (lo..=hi).any(|ny| wide[ny * w + x]);
        }
    }
    out
}
