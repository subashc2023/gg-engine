//! The v0 compare (§4.10): exact-ish gate — per-channel tolerance plus a
//! maximum differing-pixel count. The perceptual gate (FLIP/DSSIM) and the
//! HTML report land with v1 at M7; the *contract* — every rendering change
//! answers to a reference image — starts here.

/// Per-test comparison policy.
#[derive(Clone, Copy)]
pub struct Policy {
    /// Max absolute per-channel delta that still counts as "same pixel".
    pub tolerance: u8,
    /// Max number of differing pixels that still passes the test.
    pub max_diff_pixels: usize,
}

/// What a comparison found.
pub struct Comparison {
    /// Pixels whose worst channel delta exceeded the tolerance.
    pub diff_pixels: usize,
    /// The worst per-channel delta seen anywhere.
    pub max_delta: u8,
    /// Per-pixel worst-channel delta, for the heatmap (len = w*h).
    pub deltas: Vec<u8>,
}

impl Comparison {
    /// Does this comparison pass under `policy`?
    pub fn passes(&self, policy: Policy) -> bool {
        self.diff_pixels <= policy.max_diff_pixels
    }
}

/// Compare two RGBA8 buffers of identical dimensions.
pub fn compare(actual: &[u8], reference: &[u8], policy: Policy) -> anyhow::Result<Comparison> {
    anyhow::ensure!(
        actual.len() == reference.len(),
        "buffer sizes differ: actual {} B vs reference {} B",
        actual.len(),
        reference.len()
    );
    let mut deltas = Vec::with_capacity(actual.len() / 4);
    let mut diff_pixels = 0usize;
    let mut max_delta = 0u8;
    for (a, r) in actual.chunks_exact(4).zip(reference.chunks_exact(4)) {
        let delta = (0..4).map(|i| a[i].abs_diff(r[i])).max().unwrap_or(0);
        deltas.push(delta);
        max_delta = max_delta.max(delta);
        if delta > policy.tolerance {
            diff_pixels += 1;
        }
    }
    Ok(Comparison {
        diff_pixels,
        max_delta,
        deltas,
    })
}

/// Render the deltas as a heatmap: black where equal, red ramp where not —
/// the failure artifact a human (or agent) diagnoses from (§4.10).
pub fn heatmap(comparison: &Comparison, tolerance: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(comparison.deltas.len() * 4);
    for &d in &comparison.deltas {
        if d == 0 {
            out.extend_from_slice(&[0, 0, 0, 255]);
        } else if d <= tolerance {
            // Within tolerance: dim blue, visible but calm.
            out.extend_from_slice(&[0, 0, 96, 255]);
        } else {
            let heat = 128u8.saturating_add(d.saturating_mul(8));
            out.extend_from_slice(&[heat, 32, 0, 255]);
        }
    }
    out
}
