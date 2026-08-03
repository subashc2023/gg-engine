//! The compare layer (§4.10): two gates over one pair of images.
//!
//! The **exact gate** is per-channel tolerance plus a differing-pixel budget —
//! it answers "did any pixel move?". The **perceptual gate** is DSSIM over
//! overlapping windows — it answers "did the *picture* move?", which is the
//! question that separates a driver's last-bit rounding from a regression.
//! Neither subsumes the other, so every scene answers to both (M7).

/// Per-test comparison policy.
#[derive(Clone, Copy)]
pub struct Policy {
    /// Max absolute per-channel delta that still counts as "same pixel".
    pub tolerance: u8,
    /// Max number of differing pixels that still passes the exact gate.
    pub max_diff_pixels: usize,
    /// Ceiling on *benign* drift: above this per-channel delta no amount of
    /// structural agreement rescues a frame. Without it the perceptual gate is
    /// an escape hatch — with it, drift must be both small everywhere and
    /// structurally invisible to be forgiven.
    pub benign_delta: u8,
    /// Max DSSIM any single window may reach. Worst-window, not mean: a
    /// structural break is local, and a mean drowns a moved edge in the clean
    /// 99% of the frame around it.
    pub max_dssim: f64,
    /// Max |mean signed error|, in LSB, that still counts as drift — applied
    /// per channel over the frame and, scaled by [`REGION_SLACK`], per
    /// [`REGION`] tile. DSSIM is contrast-and-structure and is blind to a
    /// *level* shift, so without this the perceptual gate would forgive a wrong
    /// colour constant. Rounding noise cancels; a systematic shift does not,
    /// which is the whole discriminator.
    pub max_bias: f64,
}

impl Policy {
    /// What one [`REGION`] tile may drift by. Looser than the frame's allowance
    /// because a tile is a few hundredths of its area and its mean is that much
    /// noisier — see [`REGION_SLACK`].
    #[must_use]
    pub fn max_region_bias(self) -> f64 {
        self.max_bias * REGION_SLACK
    }
}

/// The side of one bias region, in pixels. Large enough that uncorrelated
/// rounding averages to ~1/32 LSB inside one, small enough that an error
/// confined to a lit wall fills tiles instead of being diluted by the frame
/// around it.
pub const REGION: usize = 32;

/// How much looser a region's bias allowance is than the frame's. Four, because
/// a region's mean is over ~1/50th of the pixels and correspondingly noisier —
/// and a level error big enough to see (≥1 LSB across a wall) still clears the
/// product by a wide margin.
const REGION_SLACK: f64 = 4.0;

/// What a comparison found. Both gates are always evaluated — the verdict picks
/// between them, and the report prints both numbers either way.
pub struct Comparison {
    /// Pixels whose worst channel delta exceeded the tolerance.
    pub diff_pixels: usize,
    /// The worst per-channel delta seen anywhere.
    pub max_delta: u8,
    /// Per-pixel worst-channel delta, for the heatmap (len = w*h).
    pub deltas: Vec<u8>,
    /// Worst window's DSSIM, in `0.0..=1.0`; 0 is structurally identical.
    pub dssim_worst: f64,
    /// Mean DSSIM over all windows — context for the worst, never the gate.
    pub dssim_mean: f64,
    /// Mean signed error (actual − reference) in LSB, **per channel**. Summed
    /// across RGB it would cancel: +8 on red against −8 on blue is a warm cast
    /// over the whole frame and a joint mean of exactly zero.
    pub channel_bias: [f64; 3],
    /// The worst [`REGION`] tile's worst channel, signed. A frame-wide mean is
    /// blind to an error that changes sign across the *picture* — a sun 3%
    /// brighter against a shadow 3% darker nets zero — and so is DSSIM, which
    /// sees contrast and not level. This is the term that sees both.
    pub region_bias: f64,
}

/// The signed value of the larger magnitude. Signed because *which way* a level
/// error went is what identifies it; magnitude because the gate judges size.
fn worse(a: f64, b: f64) -> f64 {
    if b.abs() > a.abs() { b } else { a }
}

/// The joined verdict of both gates.
pub enum Verdict {
    /// Inside the exact gate's budget and structurally clean.
    Pass,
    /// Over the pixel budget, but nothing moved far and nothing moved
    /// structurally: benign precision drift, accepted and *recorded* as such
    /// (§4.10 — the drift is reported, never silently swallowed).
    BenignDrift,
    /// A structural regression, or drift too large to call benign.
    Fail,
}

impl Comparison {
    /// The frame's worst channel bias, signed — one number for a report, and
    /// the frame-wide half of the bias gate.
    #[must_use]
    pub fn mean_bias(&self) -> f64 {
        self.channel_bias.iter().copied().fold(0.0, worse)
    }

    /// Whether either bias term objects: a level error over the whole frame, or
    /// one confined to a region.
    #[must_use]
    pub fn biased(&self, policy: Policy) -> bool {
        self.mean_bias().abs() > policy.max_bias
            || self.region_bias.abs() > policy.max_region_bias()
    }

    /// Judge against `policy`. The exact gate decides whether a frame is clean
    /// or merely drifting; the perceptual gate can only ever *add* a failure or
    /// forgive a drift — it cannot overrule a structural objection.
    pub fn verdict(&self, policy: Policy) -> Verdict {
        // Both terms are asked on *both* branches, and that is load-bearing:
        // `delta > tolerance` is strict, so an error that moves every channel of
        // every pixel by exactly `tolerance` differs from nothing at all as far
        // as `diff_pixels` is concerned — 1.6% of full scale at the atrium's 4 —
        // and DSSIM sees contrast rather than level. A bias consulted only once
        // the pixel budget had already been blown would forgive precisely the
        // wrong colour constant it exists to catch.
        if self.dssim_worst > policy.max_dssim || self.biased(policy) {
            return Verdict::Fail;
        }
        if self.diff_pixels <= policy.max_diff_pixels {
            return Verdict::Pass;
        }
        // Over the pixel budget, structurally clean and unbiased: benign only if
        // nothing moved far either.
        if self.max_delta <= policy.benign_delta {
            Verdict::BenignDrift
        } else {
            Verdict::Fail
        }
    }
}

/// Compare two RGBA8 buffers of identical dimensions.
pub fn compare(
    actual: &[u8],
    reference: &[u8],
    extent: (u32, u32),
    policy: Policy,
) -> anyhow::Result<Comparison> {
    anyhow::ensure!(
        actual.len() == reference.len(),
        "buffer sizes differ: actual {} B vs reference {} B",
        actual.len(),
        reference.len()
    );
    let (w, h) = (extent.0 as usize, extent.1 as usize);
    anyhow::ensure!(
        actual.len() == w * h * 4,
        "a {w}x{h} RGBA8 frame is {} B, not {}",
        w * h * 4,
        actual.len()
    );
    // Tiles floor-divide and the last one absorbs the remainder, so every region
    // is at least REGION x REGION: a four-pixel-wide sliver of a tile would have
    // a mean too noisy to judge and nothing to say.
    let (across, down) = ((w / REGION).max(1), (h / REGION).max(1));
    let mut regions = vec![([0i64; 3], 0usize); across * down];
    let mut deltas = Vec::with_capacity(w * h);
    let mut diff_pixels = 0usize;
    let mut max_delta = 0u8;
    let mut frame = [0i64; 3];
    for y in 0..h {
        let row = (y / REGION).min(down - 1) * across;
        for x in 0..w {
            let at = (y * w + x) * 4;
            let (a, r) = (&actual[at..at + 4], &reference[at..at + 4]);
            let delta = (0..4).map(|i| a[i].abs_diff(r[i])).max().unwrap_or(0);
            deltas.push(delta);
            max_delta = max_delta.max(delta);
            if delta > policy.tolerance {
                diff_pixels += 1;
            }
            let region = &mut regions[row + (x / REGION).min(across - 1)];
            region.1 += 1;
            // RGB only: alpha is a flat 255 in a readback frame and would do
            // nothing but dilute the mean it is averaged into. Kept per channel
            // and per region — a sum over the three cancels a colour cast, and a
            // sum over the frame cancels an error that changes sign across it.
            for c in 0..3 {
                let signed = i64::from(a[c]) - i64::from(r[c]);
                frame[c] += signed;
                region.0[c] += signed;
            }
        }
    }
    let pixels = (w * h).max(1) as f64;
    let region_bias = regions
        .iter()
        .flat_map(|&(sums, count)| sums.map(move |sum| sum as f64 / count.max(1) as f64))
        .fold(0.0f64, worse);
    let (dssim_worst, dssim_mean) = structural(actual, reference, extent);
    Ok(Comparison {
        diff_pixels,
        max_delta,
        deltas,
        dssim_worst,
        dssim_mean,
        channel_bias: frame.map(|sum| sum as f64 / pixels),
        region_bias,
    })
}

/// Overlapping 8×8 windows: sensitive enough that a two-pixel edge shift lands
/// wholly inside several of them, cheap enough to stay unnoticed in the tier.
const WINDOW: usize = 8;
const STRIDE: usize = 4;
const WINDOW_PIXELS: f64 = (WINDOW * WINDOW) as f64;
/// The standard 8-bit SSIM stabilisers (K1 = 0.01, K2 = 0.03 over a 255 range).
/// They exist to keep a flat window from dividing by ~0, nothing more.
const C1: f64 = 6.5025;
const C2: f64 = 58.5225;

/// Rec.709 luma. Alpha is the exact gate's business — a readback frame is
/// opaque, and an alpha regression is a channel delta, not a structural one.
fn luma(px: &[u8]) -> f64 {
    0.2126 * f64::from(px[0]) + 0.7152 * f64::from(px[1]) + 0.0722 * f64::from(px[2])
}

/// `(worst, mean)` DSSIM. Pure `+ - * /` on `f64` — no transcendentals, so the
/// number is bit-identical on every host that runs the gate, which a threshold
/// checked into the repo needs to be.
fn structural(actual: &[u8], reference: &[u8], extent: (u32, u32)) -> (f64, f64) {
    let (w, h) = (extent.0 as usize, extent.1 as usize);
    let (mut worst, mut sum, mut windows) = (0.0f64, 0.0f64, 0usize);
    let mut y = 0;
    while y + WINDOW <= h {
        let mut x = 0;
        while x + WINDOW <= w {
            let (mut sa, mut sr, mut saa, mut srr, mut sar) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for row in 0..WINDOW {
                for col in 0..WINDOW {
                    let i = ((y + row) * w + x + col) * 4;
                    let (a, r) = (luma(&actual[i..i + 4]), luma(&reference[i..i + 4]));
                    sa += a;
                    sr += r;
                    saa += a * a;
                    srr += r * r;
                    sar += a * r;
                }
            }
            // Naive second moments: luma is 0..255 in f64, so the cancellation
            // that makes this form dangerous at scale cannot reach here.
            let (ma, mr) = (sa / WINDOW_PIXELS, sr / WINDOW_PIXELS);
            let va = saa / WINDOW_PIXELS - ma * ma;
            let vr = srr / WINDOW_PIXELS - mr * mr;
            let cov = sar / WINDOW_PIXELS - ma * mr;
            let ssim = ((2.0 * ma * mr + C1) * (2.0 * cov + C2))
                / ((ma * ma + mr * mr + C1) * (va + vr + C2));
            let dssim = ((1.0 - ssim) / 2.0).clamp(0.0, 1.0);
            worst = worst.max(dssim);
            sum += dssim;
            windows += 1;
            x += STRIDE;
        }
        y += STRIDE;
    }
    if windows == 0 {
        return (0.0, 0.0); // frame smaller than one window: nothing to say
    }
    (worst, sum / windows as f64)
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const EXTENT: (u32, u32) = (64, 64);

    fn policy() -> Policy {
        Policy {
            tolerance: 0,
            max_diff_pixels: 0,
            benign_delta: 2,
            max_dssim: 0.02,
            max_bias: 0.25,
        }
    }

    /// A horizontal luma gradient with a hard vertical edge at `edge` — enough
    /// structure that SSIM has something to lose.
    fn scene(edge: usize) -> Vec<u8> {
        let mut px = Vec::new();
        for y in 0..EXTENT.1 as usize {
            for x in 0..EXTENT.0 as usize {
                let v = if x < edge { (x * 3) as u8 } else { 200 };
                px.extend_from_slice(&[v, v.wrapping_add(y as u8), 40, 255]);
            }
        }
        px
    }

    #[test]
    fn identical_frames_are_structurally_identical() {
        let a = scene(32);
        let c = compare(&a, &a, EXTENT, policy()).unwrap();
        assert_eq!((c.diff_pixels, c.max_delta), (0, 0));
        assert_eq!(c.dssim_worst, 0.0);
        assert!(matches!(c.verdict(policy()), Verdict::Pass));
    }

    /// The M7 exit row: a benign precision change fails the exact gate — every
    /// pixel differs — and the perceptual gate forgives it, *as a recorded
    /// verdict* rather than by widening a tolerance until the gate stops asking.
    /// The drift is signed alternately because that is what rounding noise is:
    /// a driver rounds both ways, and the next test is what happens when it
    /// doesn't.
    #[test]
    fn one_lsb_rounding_noise_is_benign_and_says_so() {
        let reference = scene(32);
        let actual: Vec<u8> = reference
            .chunks_exact(4)
            .enumerate()
            .flat_map(|(i, p)| {
                let d = if i % 2 == 0 { 1i16 } else { -1 };
                [nudge(p[0], d), nudge(p[1], -d), nudge(p[2], d), p[3]]
            })
            .collect();
        let c = compare(&actual, &reference, EXTENT, policy()).unwrap();
        assert_eq!(c.max_delta, 1);
        assert!(c.diff_pixels > 0, "the exact gate must object first");
        // Alternating every pixel is the *worst* case a ±1 noise can present to
        // SSIM — real driver rounding is spatially correlated and scores lower —
        // and it still sits an order of magnitude under the 0.02 budget.
        assert!(c.dssim_worst < 0.005, "worst window {}", c.dssim_worst);
        assert!(c.mean_bias().abs() < 0.02, "bias {}", c.mean_bias());
        // And inside every tile, not only over the frame: noise that cancelled
        // globally while shifting a region would be a level error in part of the
        // picture, which is the next test but one.
        assert!(c.region_bias.abs() < 0.1, "region bias {}", c.region_bias);
        assert!(matches!(c.verdict(policy()), Verdict::BenignDrift));
    }

    /// The first blind spot, closed: bias summed across R+G+B cancels a cast.
    /// `+2` on red against `−2` on blue moves no luma to speak of, moves every
    /// pixel by exactly as much as the benign ceiling allows, and — as a joint
    /// mean — reports as exactly zero. It is a warm cast over the whole frame,
    /// obvious side by side, and it used to pass.
    #[test]
    fn opposite_signs_on_two_channels_do_not_cancel_into_clean() {
        let reference = mid_scene();
        let actual: Vec<u8> = reference
            .chunks_exact(4)
            .flat_map(|p| [nudge(p[0], 2), p[1], nudge(p[2], -2), p[3]])
            .collect();
        let c = compare(&actual, &reference, EXTENT, policy()).unwrap();
        let joint: f64 = c.channel_bias.iter().sum();
        assert!(
            joint.abs() < 1e-9,
            "the joint mean is the blind spot: {joint}"
        );
        assert_eq!(c.max_delta, 2, "inside the benign ceiling");
        assert!(c.dssim_worst < 0.001, "worst window {}", c.dssim_worst);
        assert!(matches!(c.verdict(policy()), Verdict::Fail));
        let blind = Policy {
            max_bias: f64::INFINITY,
            ..policy()
        };
        assert!(
            matches!(c.verdict(blind), Verdict::BenignDrift),
            "the per-channel bias is what rejects this — drop it and the cast passes"
        );
    }

    /// The second, closed by the region term: an error that changes sign across
    /// the picture. Half the frame lifted and half lowered nets zero in *every*
    /// frame-wide mean, per channel or not, and DSSIM reads a smooth level swing
    /// as contrast it already agrees about. Only a bounded region sees it — a
    /// sun 3% brighter against a shadow 3% darker is the real shape of this.
    #[test]
    fn an_antisymmetric_shift_is_caught_by_the_region_it_lives_in() {
        let reference = mid_scene();
        let width = EXTENT.0 as usize;
        // Stepped rather than a hard split down the middle: a step is an *edge*,
        // and DSSIM would object to that on its own, which is not the blind spot
        // under test. The four steps sum to zero across every row.
        let swing = |x: usize| match x * 4 / width {
            0 => 2i16,
            1 => 1,
            2 => -1,
            _ => -2,
        };
        let actual: Vec<u8> = reference
            .chunks_exact(4)
            .enumerate()
            .flat_map(|(i, p)| {
                let d = swing(i % width);
                [nudge(p[0], d), nudge(p[1], d), nudge(p[2], d), p[3]]
            })
            .collect();
        let c = compare(&actual, &reference, EXTENT, policy()).unwrap();
        assert!(
            c.mean_bias().abs() < 1e-9,
            "every frame-wide mean is zero: {:?}",
            c.channel_bias
        );
        assert_eq!(c.max_delta, 2, "inside the benign ceiling");
        assert!(
            c.dssim_worst < policy().max_dssim,
            "dssim {}",
            c.dssim_worst
        );
        assert!(c.region_bias.abs() > 1.4, "region bias {}", c.region_bias);
        assert!(matches!(c.verdict(policy()), Verdict::Fail));
        let blind = Policy {
            max_bias: f64::INFINITY,
            ..policy()
        };
        assert!(
            matches!(c.verdict(blind), Verdict::BenignDrift),
            "the region term is what rejects this — drop it and half a frame may shift"
        );
    }

    /// The escape hatch, closed. A uniform +1 LSB on every channel is small
    /// everywhere and — above the stabilisers, which is most of any real frame —
    /// structurally invisible: DSSIM sees contrast, not level. Without the bias
    /// term the perceptual gate would wave a wrong colour constant through, and
    /// the second half of this test is that claim, not an opinion about it.
    #[test]
    fn a_uniform_shift_is_not_drift_however_small() {
        let reference = mid_scene();
        let actual: Vec<u8> = reference
            .chunks_exact(4)
            .flat_map(|p| [nudge(p[0], 1), nudge(p[1], 1), nudge(p[2], 1), p[3]])
            .collect();
        let c = compare(&actual, &reference, EXTENT, policy()).unwrap();
        assert_eq!(c.max_delta, 1);
        assert!(c.dssim_worst < 0.001, "worst window {}", c.dssim_worst);
        assert!(c.mean_bias() > 0.99, "bias {}", c.mean_bias());
        assert!(matches!(c.verdict(policy()), Verdict::Fail));
        let blind = Policy {
            max_bias: f64::INFINITY,
            ..policy()
        };
        assert!(
            matches!(c.verdict(blind), Verdict::BenignDrift),
            "the bias term is what rejects this — drop it and the frame passes"
        );
    }

    /// Mid-tone everywhere: SSIM's stabilisers only matter near black, so this
    /// is where "invisible to DSSIM" is a real claim rather than an artifact of
    /// C1 dominating a dark window.
    fn mid_scene() -> Vec<u8> {
        let mut px = Vec::new();
        for y in 0..EXTENT.1 as usize {
            for x in 0..EXTENT.0 as usize {
                px.extend_from_slice(&[120 + (x % 40) as u8, 140 + (y % 30) as u8, 160, 255]);
            }
        }
        px
    }

    fn nudge(v: u8, d: i16) -> u8 {
        (i16::from(v) + d).clamp(0, 255) as u8
    }

    /// The other half: a *structural* change — the same edge, two pixels over —
    /// is rejected however the pixel budget is set, because DSSIM sees the
    /// picture move rather than the numbers.
    #[test]
    fn a_shifted_edge_is_structural_and_no_budget_forgives_it() {
        let c = compare(&scene(34), &scene(32), EXTENT, policy()).unwrap();
        assert!(c.dssim_worst > 0.02, "worst window {}", c.dssim_worst);
        let generous = Policy {
            tolerance: 255,
            max_diff_pixels: usize::MAX,
            ..policy()
        };
        assert!(matches!(c.verdict(generous), Verdict::Fail));
    }

    /// Small *and* structurally invisible are both required: a handful of pixels
    /// wrenched to a different colour stays inside a generous pixel budget and
    /// is still not benign.
    #[test]
    fn a_large_delta_is_never_benign_drift() {
        let reference = scene(32);
        let mut actual = reference.clone();
        for p in actual.chunks_exact_mut(4).take(4) {
            p[0] = p[0].wrapping_add(120);
        }
        let c = compare(&actual, &reference, EXTENT, policy()).unwrap();
        assert_eq!(c.max_delta, 120);
        assert!(matches!(c.verdict(policy()), Verdict::Fail));
    }
}
