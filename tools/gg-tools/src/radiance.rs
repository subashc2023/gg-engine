// The display pipeline run backwards (§6 M70): a readback code value as the scene
// radiance behind it.
//
// # Why this exists, and what it cost not to
//
// Three instruments here grade a render by a **ratio of the same pixel** — the
// frame with a pass on against the frame with it off — and read that ratio as the
// pass's answer in physical units. `bounce` grades the irradiance field that way,
// `ao` the occlusion pass, `field` a frame-to-frame swing. The method is exact and
// needs no reference renderer, on one condition nobody had checked: that everything
// between a radiance and a code value is **linear**.
//
// It is not. `post.slang`'s tonemapper opens with
//
//     float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
//     color -= offset;
//
// so a mid-tone arrives at the swapchain as `x - 0.04` and a dark one as `6.25x²`.
// A *ratio* through the first exaggerates every deviation from 1 by `x / (x - 0.04)`
// — 1.42x where these rooms sit — and a ratio through the second **squares** it.
// Every percentage those three commands have ever printed was inflated by a factor
// they did not know they had, and the factor is not a constant: it depends on the
// pixel's own brightness, so a table whose two columns fail in opposite directions
// (`invented` against `missed`, the shape every plateau in this tree is read off)
// had its two sides scaled *differently*. Bright errors were corrected less than
// dark ones. That is what makes this a correctness fix rather than a calibration.
//
// # It is a bijection, and the inverse is closed form
//
// Not a fit and not a solve. Every stage of `pbr_neutral` is invertible from the
// output alone, which is worth stating because the obvious reading of that curve is
// that its shoulder loses information:
//
//   - the desaturation lerps toward `new_peak`, and the lerp's other end has
//     `new_peak` as its own maximum, so **`max(output)` is `new_peak`** whatever
//     the blend did;
//   - `new_peak` determines `peak` by one reciprocal, and the two together
//     determine the desaturation blend, so the lerp and the rescale come apart;
//   - the opening offset is a monotone map of the *minimum* channel alone, and its
//     two branches meet at `0.08 -> 0.04`, so one comparison picks the branch.
//
// What genuinely loses information is the quantizer: a channel at code 255 could
// have been any radiance at all, and that is what [`Scene::Clipped`] says.
//
// The forward curve is restated in this module's tests, which is a second copy of
// the shader's arithmetic and is deliberate — the test's whole content is that the
// two compose to the identity, and it fails on a transcription slip in either.

use anyhow::Result;
use gg_math::sim;

/// `post.slang`'s `START_COMPRESSION` and `DESATURATION`. Both are the reference
/// implementation's own constants (KhronosGroup/ToneMapping), and the shader's
/// comment declines to tune them; a drift between the two files would show up in
/// `the_curve_and_its_inverse_compose_to_the_identity` as a round-trip error rather
/// than as a wrong picture.
const START: f32 = 0.8 - 0.04;
const DESATURATION: f32 = 0.15;

/// What one readback pixel says about the radiance that produced it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scene {
    /// The radiance, in the units the renderer was handed.
    Linear([f32; 3]),
    /// A channel at code 255 — the quantizer's ceiling, above which the curve's
    /// shoulder is arbitrarily steep and one code value spans an unbounded range.
    /// Counted rather than guessed at: an instrument that averaged a made-up number
    /// here would report its own ceiling as the scene's.
    Clipped,
}

/// sRGB decode, IEC 61966-2-1 — `post.slang`'s encode run backwards.
pub fn decode(code: u8) -> f32 {
    let e = f32::from(code) / 255.0;
    if e <= 0.040_45 {
        e / 12.92
    } else {
        sim::powf((e + 0.055) / 1.055, 2.4)
    }
}

/// The opening offset backwards, on the minimum channel.
///
/// Forward it is `m -> m < 0.08 ? 6.25m² : m - 0.04`, which is monotone, continuous at
/// `0.08 -> 0.04`, and the only stage of the curve that reads a channel other than the
/// one it writes — so one comparison picks the branch in either direction. Only the
/// inverse is here: the forward form lives in this module's tests, where it has a job.
fn unpedestal(low: f32) -> f32 {
    if low < 0.04 {
        sim::sqrt(low.max(0.0) / 6.25)
    } else {
        low + 0.04
    }
}

/// One readback pixel as scene radiance — `pbr_neutral` and the sRGB encode both
/// run backwards, then the exposure and the white point divided back out.
///
/// `exposure` is in stops and `white` is the display's peak in units of diffuse
/// white, exactly as `post_push` passes them: 0 and 1 for every offscreen render,
/// which is what [`scene`] assumes and [`scene_at`] does not.
pub fn scene_at(pixel: &[u8], exposure: f32, white: f32) -> Scene {
    if pixel.iter().take(3).any(|c| *c == u8::MAX) {
        return Scene::Clipped;
    }
    let out = [decode(pixel[0]), decode(pixel[1]), decode(pixel[2])];
    // Over `[0, white]`, which is the range the curve was evaluated on.
    let out = [out[0] / white, out[1] / white, out[2] / white];
    let high = out[0].max(out[1]).max(out[2]);
    // Below the knee the curve returned its input untouched, so the only stage to
    // undo is the pedestal. `high` is the post-pedestal peak either way — see the
    // header — so this one comparison is the same test the shader made.
    let post = if high < START {
        out
    } else {
        // `high` *is* `new_peak`: the desaturation's far end is the constant
        // `new_peak`, whose maximum is itself.
        let d = 1.0 - START;
        let peak = d * d / (1.0 - high).max(1e-6) - d + START;
        let g = 1.0 - 1.0 / (DESATURATION * (peak - high) + 1.0);
        let scale = peak / high.max(1e-6);
        // The lerp, then the rescale: `out = lerp(c * new_peak/peak, new_peak, g)`.
        [
            ((out[0] - g * high) / (1.0 - g)) * scale,
            ((out[1] - g * high) / (1.0 - g)) * scale,
            ((out[2] - g * high) / (1.0 - g)) * scale,
        ]
    };
    let low = post[0].min(post[1]).min(post[2]);
    let offset = unpedestal(low) - low;
    let gain = white / sim::exp2(exposure);
    Scene::Linear([
        (post[0] + offset) * gain,
        (post[1] + offset) * gain,
        (post[2] + offset) * gain,
    ])
}

/// [`scene_at`] at the offscreen path's own exposure and white point.
///
/// `r.exposure` because a leg is free to move it; `white` is 1.0 rather than
/// `white_point()`, and that is a property of the renderer these instruments drive
/// — `OffscreenRhi` presents to no display, so `r.hdr` is 0 and the curve is
/// evaluated over `[0, 1]`. Asserted rather than assumed.
pub fn scene(pixel: &[u8]) -> Result<Scene> {
    anyhow::ensure!(
        gg_render::cvars::HDR.int() == 0,
        "the offscreen path has no display to grant HDR; r.hdr is {}",
        gg_render::cvars::HDR.int()
    );
    Ok(scene_at(
        pixel,
        gg_render::cvars::EXPOSURE.float() as f32,
        1.0,
    ))
}

/// Every pixel of a readback as scene radiance, **RGBA in, RGBA out** — the shape
/// `OffscreenFrame::radiance` comes in, so a caller can grade the two against each
/// other without a second reduction that could disagree.
///
/// A clipped pixel comes back infinite rather than guessed at. **A caller grading a
/// ratio must handle that** — [`scene`] is the one that names it per pixel.
pub fn scene_luminances(pixels: &[u8]) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(pixels.len());
    for p in pixels.chunks_exact(4) {
        match scene(p)? {
            Scene::Linear(c) => out.extend_from_slice(&[c[0], c[1], c[2], 1.0]),
            Scene::Clipped => out.extend_from_slice(&[f32::INFINITY; 4]),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `post.slang`'s `pbr_neutral`, transcribed — the forward curve this module
    /// inverts. A second copy of the shader's arithmetic on purpose: the test's
    /// content is that the two compose to the identity, so a slip in either end
    /// fails it.
    fn pbr_neutral(color: [f32; 3]) -> [f32; 3] {
        let x = color[0].min(color[1]).min(color[2]);
        let offset = if x < 0.08 { x - 6.25 * x * x } else { 0.04 };
        let mut c = [color[0] - offset, color[1] - offset, color[2] - offset];
        let peak = c[0].max(c[1]).max(c[2]);
        if peak < START {
            return c;
        }
        let d = 1.0 - START;
        let new_peak = 1.0 - d * d / (peak + d - START);
        for v in &mut c {
            *v *= new_peak / peak;
        }
        let g = 1.0 - 1.0 / (DESATURATION * (peak - new_peak) + 1.0);
        [
            c[0] + (new_peak - c[0]) * g,
            c[1] + (new_peak - c[1]) * g,
            c[2] + (new_peak - c[2]) * g,
        ]
    }

    /// sRGB encode, and the quantizer — `post.slang`'s own last two steps, so the
    /// round trip below starts where a readback does.
    fn encode(v: f32) -> u8 {
        let c = v.clamp(0.0, 1.0);
        let e = if c <= 0.003_130_8 {
            c * 12.92
        } else {
            1.055 * sim::powf(c, 1.0 / 2.4) - 0.055
        };
        (e * 255.0 + 0.5) as u8
    }

    /// The whole claim: the curve is a bijection and this module holds its inverse.
    ///
    /// Graded in *radiance* and not in code values, because that is the direction an
    /// instrument reads — and to a tolerance set by the quantizer rather than by
    /// floating point: one code value at the darkest sample here is worth about a
    /// per cent of it, so the assertion is against the round trip of the *quantized*
    /// value and the 8-bit step is what it is allowed.
    #[test]
    fn the_curve_and_its_inverse_compose_to_the_identity() {
        // Deliberately across the pedestal's branch (0.08), the knee (0.76) and well
        // into the shoulder, saturated and neutral both.
        let cases: &[[f32; 3]] = &[
            [0.004, 0.004, 0.004],
            [0.02, 0.02, 0.02],
            [0.079, 0.079, 0.079],
            [0.081, 0.081, 0.081],
            [0.125, 0.125, 0.125],
            [0.3, 0.3, 0.3],
            [0.7, 0.7, 0.7],
            [0.9, 0.9, 0.9],
            [1.5, 1.5, 1.5],
            [3.0, 3.0, 3.0],
            [0.4, 0.1, 0.02],
            [0.9, 0.3, 0.05],
            [2.0, 0.5, 0.1],
            [0.05, 0.2, 0.6],
        ];
        for want in cases {
            let display = pbr_neutral(*want);
            let pixel = [
                encode(display[0]),
                encode(display[1]),
                encode(display[2]),
                255,
            ];
            let Scene::Linear(got) = scene_at(&pixel, 0.0, 1.0) else {
                panic!("{want:?} came back clipped at {display:?}");
            };
            // The 8-bit step in *radiance* at this brightness — one code value on
            // every channel, not on one: the pedestal is a function of the minimum
            // channel, so a single-channel bump leaves it where it was and would
            // price the step at the encode's resolution rather than the chain's.
            // Which matters most exactly where the chain is coarsest: the quadratic
            // branch takes a radiance of 0.004 to code **0**, so the darkest samples
            // here are resolved to about 7e-3 and no better.
            let step = {
                let up = [
                    pixel[0].saturating_add(1),
                    pixel[1].saturating_add(1),
                    pixel[2].saturating_add(1),
                    255,
                ];
                match scene_at(&up, 0.0, 1.0) {
                    Scene::Linear(n) => (0..3).fold(1e-4f32, |m, c| m.max(n[c] - got[c])),
                    Scene::Clipped => 1e-4,
                }
            };
            for c in 0..3 {
                assert!(
                    (got[c] - want[c]).abs() <= step,
                    "channel {c} of {want:?}: got {got:?}, one code value is {step}"
                );
            }
        }
    }

    /// The pedestal is what the milestone is about, so its leverage on a *ratio* is
    /// asserted rather than left in prose: two radiances five per cent apart arrive
    /// at the swapchain further apart than that, and by how much depends on where
    /// they sit.
    #[test]
    fn a_ratio_of_code_values_is_not_a_ratio_of_radiances() {
        for (base, least) in [(0.125f32, 1.35f32), (0.3, 1.10), (0.05, 1.5)] {
            let lo = pbr_neutral([base * 0.95, base * 0.95, base * 0.95])[1];
            let hi = pbr_neutral([base, base, base])[1];
            let display = 1.0 - lo / hi;
            assert!(
                display >= 0.05 * least,
                "at {base} a 5 % gap reads {:.3} % on screen, expected at least {:.3} %",
                display * 100.0,
                5.0 * least
            );
        }
    }

    /// A blown pixel is named, not averaged — the one place the chain really is
    /// lossy.
    #[test]
    fn a_clipped_channel_is_refused() {
        assert_eq!(scene_at(&[255, 12, 12, 255], 0.0, 1.0), Scene::Clipped);
        assert_eq!(scene_at(&[12, 12, 255, 255], 0.0, 1.0), Scene::Clipped);
        assert!(matches!(
            scene_at(&[254, 12, 12, 255], 0.0, 1.0),
            Scene::Linear(_)
        ));
    }

    /// Exposure and the white point divide back out, so a leg that moves either
    /// still reads radiance in the units it handed the renderer.
    #[test]
    fn the_exposure_and_the_white_point_come_back_out() {
        let want = [0.2f32, 0.15, 0.1];
        for (exposure, white) in [(0.0f32, 1.0f32), (1.0, 1.0), (-1.0, 1.0), (0.0, 4.0)] {
            let scaled = [
                want[0] * sim::exp2(exposure) / white,
                want[1] * sim::exp2(exposure) / white,
                want[2] * sim::exp2(exposure) / white,
            ];
            let display = pbr_neutral(scaled);
            let pixel = [
                encode(display[0] * white),
                encode(display[1] * white),
                encode(display[2] * white),
                255,
            ];
            let Scene::Linear(got) = scene_at(&pixel, exposure, white) else {
                panic!("clipped at exposure {exposure} white {white}");
            };
            for c in 0..3 {
                assert!(
                    (got[c] - want[c]).abs() < 0.01 * white,
                    "exposure {exposure} white {white} channel {c}: {got:?} vs {want:?}"
                );
            }
        }
    }
}
