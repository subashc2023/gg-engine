// Breaking up the quantizer immediately after a fullscreen write (§6 M22).
//
// Lifted out of `post.slang` at §6 M62 for the reason a second caller always is:
// the debug view is a *blit* onto the same 8-bit swapchain and had none, so a
// depth ramp over fifty metres arrived as twenty-centimetre contours — the
// `r.dither` measurement in `gg-tools banding` describes that pass exactly as
// well as it describes the tonemapper, and the two now meet the quantizer
// through one piece of code rather than through one and a copy.
//
// The amount is a parameter rather than a push field, because the two callers
// spell their push blocks separately and a shared function that read
// `push.dither` would silently bind to whichever one included it.

// A 32-bit integer hash — exact on every device by construction, which is what
// makes a dithered frame reproducible enough to bless (§4.10). A `frac(sin(...))`
// noise would not be: the transcendental is the driver's to round.
// (Jenkins-style mix, the `lowbias32` constants.)
uint dither_hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// Uniform noise in (-0.5, 0.5).
//
// **Uniform and not triangular**, which is a deliberate trade against the
// classical result (Lipshitz, Wannamaker & Vanderkooy, JAES 1992: a triangular
// PDF two LSBs wide makes the quantization error's variance independent of the
// signal as well as its mean; a uniform one leaves the variance modulating, so
// the noise faintly breathes across a gradient). Half a code value is what the
// window below can scale to zero at a code; an LSB could not, and a flat region
// that is exact on one driver and not another is worse than a variance term
// nobody can see at one part in 255.
float dither_noise(uint seed)
{
    return float(dither_hash(seed) >> 8) * (1.0 / 16777216.0) - 0.5;
}

// The seed is the pixel alone, so a still frame is still: this adds no shimmer
// to a static camera and nothing here is a function of time.
float3 dither_triple(float2 pixel)
{
    uint2 p = uint2(pixel);
    uint seed = p.x * 73856093u ^ p.y * 19349663u;
    // Per channel and not per pixel: three independent draws read as fine chroma
    // noise, one shared draw reads as luminance grain — the same amplitude, and
    // the eye is far more sensitive to the second.
    return float3(dither_noise(seed), dither_noise(seed + 1u), dither_noise(seed + 2u));
}

// Code-value distance from a code over which the dither ramps back up to full.
// See `dither` — it is a bound on the *driver's* encode error, not a taste knob.
static const float DITHER_DEADZONE = 0.25;

// **The window.** A contour is where the round *changes*, which is where the
// value sits halfway between two codes; a value sitting *on* a code is already
// exactly representable and has no error to spread. So the amplitude ramps to
// zero over the last quarter of a code either side, and is full over the middle
// half where every contour is. That is what makes the dither safe rather than
// merely tidy: the hardware's own encode is an approximation — llvmpipe's is
// good to about a tenth of a code value, not to the ULP — so no fixed margin
// under half a step survives it, while a perturbation of exactly zero does. The
// deadzone has to be wider than that disagreement, which is what a quarter buys
// and an eighth would not.
float3 dither_window(float3 encoded, float levels)
{
    float3 f = frac(encoded * levels);
    return saturate(min(f, 1.0 - f) * (1.0 / DITHER_DEADZONE));
}

// The sRGB transfer pair, IEC 61966-2-1. Present here for one reason: the
// quantizer a fullscreen pass feeds sits on the *other* side of it — the
// attachment is `_SRGB` and the hardware encodes on write — so a perturbation
// applied to a linear value is a different number of code values at every
// brightness, and at the dark end where banding is worst it is a fraction of one.
float3 linear_to_srgb(float3 c)
{
    c = max(c, 0.0);
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

// Linear value one *code value* is worth at this brightness — the derivative of
// the decode, evaluated where the colour already sits. Above the toe
// `linear = ((e + 0.055) / 1.055)^2.4`, so `dlinear/de = 2.4 * linear / (e +
// 0.055)` and the value we already have does the work of the second power.
float3 code_value(float3 colour, float3 encoded)
{
    return select(encoded <= 0.04045, (1.0 / 12.92).xxx,
                  2.4 * colour / (encoded + 0.055));
}

// Half a code value of the *output* is what a smooth gradient has to be broken
// up by — scaled by how far the value sits from a code, and applied through the
// decode's slope rather than by a round trip.
//
// **The slope.** Encoding, adding and decoding back is the obvious spelling and
// is wrong for the driver's-rounding reason above: `pow` is the driver's to
// round, so a colour that arrived on a code would come back a part in 10^7 off
// it. Scaling the noise into linear and adding leaves the signal itself
// untouched.
//
// The caller passes a *linear* colour bound for an `_SRGB` attachment. A pass
// that owns its own encode wants `dither_code` instead.
float3 dither_linear(float3 colour, float2 pixel, float amount)
{
    if (amount <= 0.0)
    {
        return colour;
    }
    float3 encoded = linear_to_srgb(colour);
    return colour + dither_triple(pixel) * dither_window(encoded, 255.0) * amount *
                        (1.0 / 255.0) * code_value(colour, encoded);
}

// Dither in the *encoded* domain, for a quantizer that takes `levels` steps.
//
// The SDR path cannot use this and does not: there the shader writes linear
// values into an sRGB-format attachment and the *hardware* encodes, so the noise
// has to be spent through the decode's slope before the write (`dither_linear`).
// Here the shader owns the encode, so the noise goes in where the quantizer is
// and the whole slope argument evaporates. Two dithers because there are
// genuinely two quantizers in two places, not because one of them is wrong.
//
// The deadzone is the same idea and is here for a sharper reason: PQ is
// absolute, so pure black is an exact code on every display, and a half-step of
// noise on it is a grey floor in a dark room.
float3 dither_code(float3 encoded, float2 pixel, float levels, float amount)
{
    if (amount <= 0.0)
    {
        return encoded;
    }
    return encoded + dither_triple(pixel) * dither_window(encoded, levels) * amount *
                         (1.0 / levels);
}
