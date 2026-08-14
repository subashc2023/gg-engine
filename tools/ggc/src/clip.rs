//! RIFF/WAVE → mono 16-bit PCM, the one shape [`gg_assets::clip`] holds.
//!
//! Hand-written and dependency-free, which is a choice about *refusals* rather
//! than about code size: a decoder crate takes a compressed `.wav` and hands
//! back samples, and an ADPCM source silently becoming a clip is a bug nobody
//! hears until the game does. Everything below either produces PCM or names what
//! it will not read.
//!
//! **The conversion is exact integer and rational work — no transcendental, no
//! resampler, no dither.** Depth changes are shifts, the downmix is an integer
//! average, and the float path is one multiply, one clamp and a round-half-away
//! -from-zero built out of add-and-truncate. Every one of those is an IEEE basic
//! operation or plain integer arithmetic, so a pack built on this desk and one
//! built in WSL agree byte for byte (§4.6) without a pinned library between them.

use anyhow::{Result, bail};

/// WAVE format tags. `EXTENSIBLE` is a wrapper: the real tag is the first two
/// bytes of the sub-format GUID, which is why it never reaches the match below.
const PCM: u16 = 1;
const FLOAT: u16 = 3;
const EXTENSIBLE: u16 = 0xFFFE;

/// What `fmt ` said.
struct Format {
    tag: u16,
    channels: u16,
    rate: u32,
    bits: u16,
}

/// Parse a `.wav` into its sample rate and mono 16-bit samples.
///
/// # Errors
/// If the container is not RIFF/WAVE, if `fmt ` or `data` is missing or short,
/// or if the format tag or bit depth is one this refuses — the message names the
/// offending value, because a source that cannot be baked is a content problem
/// and the number is what the author needs.
pub fn parse(bytes: &[u8]) -> Result<(u32, Vec<i16>)> {
    let riff = bytes.get(..12).unwrap_or_default();
    if riff.get(..4) != Some(b"RIFF") || riff.get(8..12) != Some(b"WAVE") {
        bail!("not a RIFF/WAVE file");
    }
    // The RIFF size counts from byte 8, and files in the wild routinely
    // disagree with their own — clamped rather than trusted, so a truncated
    // download fails as a short chunk instead of an out-of-bounds slice.
    let end = u32_at(bytes, 4)
        .map_or(bytes.len(), |size| (size as usize).saturating_add(8))
        .min(bytes.len());

    let mut format = None;
    let mut data: &[u8] = &[];
    let mut at = 12;
    while at + 8 <= end {
        let id = &bytes[at..at + 4];
        let size = u32_at(bytes, at + 4).unwrap_or(0) as usize;
        let body = bytes
            .get(at + 8..)
            .and_then(|rest| rest.get(..size.min(end - at - 8)))
            .unwrap_or_default();
        match id {
            b"fmt " => format = Some(parse_format(body)?),
            b"data" => data = body,
            // `LIST`, `fact`, `cue `, whatever a DAW wrote: skipped by size.
            // Unknown metadata is not an error — refusing it would fail on
            // sources that are perfectly ordinary PCM.
            _ => {}
        }
        // A chunk of odd size carries a pad byte that is not counted in it.
        at += 8 + size + (size & 1);
    }

    let Some(format) = format else {
        bail!("no `fmt ` chunk");
    };
    if data.is_empty() {
        bail!("no `data` chunk, or an empty one");
    }
    Ok((format.rate, convert(&format, data)))
}

fn parse_format(body: &[u8]) -> Result<Format> {
    if body.len() < 16 {
        bail!(
            "a `fmt ` chunk of {} bytes is short of the 16 a WAVE header needs",
            body.len()
        );
    }
    let mut tag = u16_at(body, 0).unwrap_or(0);
    let channels = u16_at(body, 2).unwrap_or(0);
    let rate = u32_at(body, 4).unwrap_or(0);
    let bits = u16_at(body, 14).unwrap_or(0);
    if tag == EXTENSIBLE {
        // WAVE_FORMAT_EXTENSIBLE: a 40-byte `fmt ` whose sub-format GUID starts
        // with the tag it stands in for. Only those two bytes matter — the rest
        // of the GUID is the fixed KSDATAFORMAT_SUBTYPE suffix.
        tag = u16_at(body, 24).unwrap_or(0);
    }
    match (tag, bits) {
        (PCM, 8 | 16 | 24 | 32) | (FLOAT, 32) => {}
        (PCM | FLOAT, bits) => bail!("{bits}-bit samples (format tag {tag}) are not imported"),
        (tag, bits) => bail!(
            "format tag {tag} is compressed or unknown ({bits}-bit) — bake PCM instead, \
             so nothing decodes at load"
        ),
    }
    if channels == 0 {
        bail!("a `fmt ` chunk claiming 0 channels");
    }
    if rate == 0 {
        bail!("a `fmt ` chunk claiming 0 Hz");
    }
    Ok(Format {
        tag,
        channels,
        rate,
        bits,
    })
}

/// Frames of interleaved source samples → mono `i16`.
///
/// Every depth lands in the `i16` domain *first* and the channels are averaged
/// there, which is what the test comparing an 8-bit, a 16-bit and a float source
/// asserts: one sample value, three spellings of it.
fn convert(format: &Format, data: &[u8]) -> Vec<i16> {
    let width = (format.bits / 8) as usize;
    let channels = format.channels as usize;
    let stride = width * channels;
    let frames = data.len() / stride; // a partial trailing frame is dropped
    let mut samples = Vec::with_capacity(frames);
    for frame in 0..frames {
        let mut sum = 0i32;
        for channel in 0..channels {
            let at = frame * stride + channel * width;
            sum += sample(format, &data[at..at + width]);
        }
        // The downmix: an integer average, so a stereo file is the same on every
        // host without a normalization pass to disagree about.
        samples
            .push((sum / channels as i32).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16);
    }
    samples
}

/// One source sample, in the `i16` domain.
fn sample(format: &Format, bytes: &[u8]) -> i32 {
    match (format.tag, format.bits) {
        // 8-bit WAVE is *unsigned*, biased by 128, and the only depth that is.
        (_, 8) => (i32::from(bytes[0]) - 128) * 256,
        (_, 16) => i32::from(i16::from_le_bytes([bytes[0], bytes[1]])),
        // 24- and 32-bit narrow by an arithmetic shift: exact for anything a
        // 16-bit source could have held, and truncating toward -inf for the low
        // bits that have nowhere to go. Rounding them would cost a branch per
        // sample to move the error by half a code value nobody can hear.
        (_, 24) => i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]]) >> 16,
        (FLOAT, _) => {
            let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            // Clamped before scaling: float WAVE is nominally -1..=1 but nothing
            // enforces it, and a mastering overshoot must clip rather than wrap.
            // A NaN survives every step and lands on 0 at the cast, which is
            // Rust's saturating float→int rule and the one value that is silent.
            // Round half away from zero: `as` truncates toward zero, so the half
            // carries the sample's own sign.
            let scaled = value.clamp(-1.0, 1.0) * 32_767.0;
            (scaled + if scaled < 0.0 { -0.5 } else { 0.5 }) as i32
        }
        (_, _) => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) >> 16,
    }
}

fn u16_at(bytes: &[u8], at: usize) -> Option<u16> {
    bytes
        .get(at..at + 2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

#[cfg(test)]
mod tests {
    // unwrap is permitted in tests (§2, Error handling row).
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// Assemble a `.wav` byte for byte, so every expectation below is arithmetic
    /// against a header this file wrote rather than against a checked-in
    /// fixture nobody can read.
    fn wav(
        tag: u16,
        channels: u16,
        rate: u32,
        bits: u16,
        data: &[u8],
        extra: &[(&[u8], &[u8])],
    ) -> Vec<u8> {
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&tag.to_le_bytes());
        fmt.extend_from_slice(&channels.to_le_bytes());
        fmt.extend_from_slice(&rate.to_le_bytes());
        let align = u32::from(channels) * u32::from(bits) / 8;
        fmt.extend_from_slice(&(rate * align).to_le_bytes());
        fmt.extend_from_slice(&(align as u16).to_le_bytes());
        fmt.extend_from_slice(&bits.to_le_bytes());

        let mut body = b"WAVE".to_vec();
        let mut chunk = |id: &[u8], payload: &[u8]| {
            body.extend_from_slice(id);
            body.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            body.extend_from_slice(payload);
            if payload.len() % 2 == 1 {
                body.push(0);
            }
        };
        chunk(b"fmt ", &fmt);
        for &(id, payload) in extra {
            chunk(id, payload);
        }
        chunk(b"data", data);

        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);
        out
    }

    /// The reference samples, and the values every other depth is written to
    /// reproduce: multiples of 256 so the 8-bit spelling is exact.
    const MONO: [i16; 4] = [0, 8_192, -8_192, 32_512];

    fn le16(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[test]
    fn a_stereo_16_bit_source_averages_to_the_mono_samples() {
        // Left and right chosen so the average is written down rather than
        // derived: (100 + 300)/2, (-1000 + 0)/2, (7 + 8)/2 truncating toward 0.
        let interleaved = le16(&[100, 300, -1000, 0, 7, 8]);
        let (rate, samples) = parse(&wav(PCM, 2, 44_100, 16, &interleaved, &[])).unwrap();
        assert_eq!(rate, 44_100);
        assert_eq!(samples, [200, -500, 7]);
    }

    #[test]
    fn eight_bit_and_float_land_where_sixteen_bit_does() {
        let reference = parse(&wav(PCM, 1, 22_050, 16, &le16(&MONO), &[])).unwrap();
        assert_eq!(reference.1, MONO);

        // 8-bit is unsigned and biased by 128; every reference value is a
        // multiple of 256, so this is the same number with the low byte absent.
        let eight: Vec<u8> = MONO.iter().map(|s| ((s / 256) + 128) as u8).collect();
        assert_eq!(parse(&wav(PCM, 1, 22_050, 8, &eight, &[])).unwrap().1, MONO);

        // Float, scaled by 32767 on the way back — so the round trip is exact
        // only where the source divided evenly, and 32_512 does not.
        let float: Vec<u8> = MONO
            .iter()
            .flat_map(|&s| (f32::from(s) / 32_767.0).to_le_bytes())
            .collect();
        let (_, samples) = parse(&wav(FLOAT, 1, 22_050, 32, &float, &[])).unwrap();
        for (got, want) in samples.iter().zip(MONO) {
            assert!(
                (i32::from(*got) - i32::from(want)).abs() <= 1,
                "{got} vs {want}"
            );
        }
    }

    #[test]
    fn float_clips_rather_than_wrapping_and_rounds_symmetrically() {
        // An overshoot past unity, then the two sides of a boundary. Truncation
        // would give 2 and -2 for the .6 pair; rounding *to even* would break
        // the sign symmetry the .6/-.6 pair asserts. Offsets of 0.1 rather than
        // exactly 0.5 because no f32 divided by 32767 lands on a half — testing
        // at the boundary would be testing the double rounding, not the rule.
        let values: [f32; 6] = [
            2.0,
            -2.0,
            2.6 / 32_767.0,
            -2.6 / 32_767.0,
            2.4 / 32_767.0,
            -2.4 / 32_767.0,
        ];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (_, samples) = parse(&wav(FLOAT, 1, 48_000, 32, &data, &[])).unwrap();
        assert_eq!(samples, [32_767, -32_767, 3, -3, 2, -2]);
    }

    #[test]
    fn a_float_source_full_of_nonsense_bakes_to_silence_rather_than_noise() {
        // NaN and both infinities: the clamp passes NaN through and the
        // saturating `as` cast lands it on 0, which is the only value that
        // cannot be heard. An infinity clamps first and is a full-scale sample.
        let values: [f32; 3] = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (_, samples) = parse(&wav(FLOAT, 1, 48_000, 32, &data, &[])).unwrap();
        assert_eq!(samples, [0, 32_767, -32_767]);
    }

    #[test]
    fn twenty_four_and_thirty_two_bit_narrow_to_their_high_bits() {
        // 24-bit: 0x123456 >> 8 is 0x1234. Written little-endian, low byte first.
        let data = [0x56, 0x34, 0x12, 0xAA, 0xBB, 0xCC];
        let (_, samples) = parse(&wav(PCM, 1, 8_000, 24, &data, &[])).unwrap();
        assert_eq!(samples, [0x1234, 0xCCBB_u16 as i16]);

        let data: Vec<u8> = [0x1234_5678_i32, -0x0123_4567]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (_, samples) = parse(&wav(PCM, 1, 8_000, 32, &data, &[])).unwrap();
        assert_eq!(samples, [0x1234, (-0x0123_4567_i32 >> 16) as i16]);
    }

    #[test]
    fn extensible_is_read_through_its_sub_format_guid() {
        // A 40-byte `fmt `: cbSize, valid bits, channel mask, then the GUID
        // whose first two bytes are the tag this stands in for.
        let mut fmt = Vec::new();
        fmt.extend_from_slice(&EXTENSIBLE.to_le_bytes());
        fmt.extend_from_slice(&1u16.to_le_bytes());
        fmt.extend_from_slice(&44_100u32.to_le_bytes());
        fmt.extend_from_slice(&88_200u32.to_le_bytes());
        fmt.extend_from_slice(&2u16.to_le_bytes());
        fmt.extend_from_slice(&16u16.to_le_bytes());
        fmt.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        fmt.extend_from_slice(&16u16.to_le_bytes()); // valid bits
        fmt.extend_from_slice(&4u32.to_le_bytes()); // channel mask
        fmt.extend_from_slice(&PCM.to_le_bytes());
        fmt.extend_from_slice(&[0; 14]);

        let mut body = b"WAVE".to_vec();
        body.extend_from_slice(b"fmt ");
        body.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        body.extend_from_slice(&fmt);
        let data = le16(&MONO);
        body.extend_from_slice(b"data");
        body.extend_from_slice(&(data.len() as u32).to_le_bytes());
        body.extend_from_slice(&data);
        let mut out = b"RIFF".to_vec();
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&body);

        assert_eq!(parse(&out).unwrap().1, MONO);
    }

    #[test]
    fn a_compressed_format_tag_is_refused_by_name() {
        // 0x0011 is IMA ADPCM. A decoder crate would have produced samples here;
        // this has to fail at bake, since the runtime casts blobs in place.
        let err = parse(&wav(0x0011, 1, 22_050, 4, &[0; 16], &[])).unwrap_err();
        let message = format!("{err}");
        assert!(message.contains("17"), "{message}");
        assert!(message.contains("4-bit"), "{message}");
    }

    #[test]
    fn an_unsupported_depth_is_refused_with_the_depth_in_the_message() {
        let err = parse(&wav(PCM, 1, 22_050, 12, &[0; 16], &[])).unwrap_err();
        assert!(format!("{err}").contains("12-bit"), "{err}");
        // 64-bit float is a legal WAVE and still not something the blob holds.
        let err = parse(&wav(FLOAT, 1, 22_050, 64, &[0; 16], &[])).unwrap_err();
        assert!(format!("{err}").contains("64-bit"), "{err}");
    }

    #[test]
    fn unknown_chunks_between_fmt_and_data_are_skipped_rather_than_fatal() {
        // A `LIST` of odd length, so the pad byte is under test too: miscounting
        // it puts the walk one byte into `data`'s id and loses the samples.
        let extra: [(&[u8], &[u8]); 3] = [
            (b"LIST", b"INFOISFTLavf59"),
            (b"fact", &[4, 0, 0, 0]),
            (b"junk", b"odd"),
        ];
        let (rate, samples) = parse(&wav(PCM, 1, 32_000, 16, &le16(&MONO), &extra)).unwrap();
        assert_eq!(rate, 32_000);
        assert_eq!(samples, MONO);
    }

    #[test]
    fn a_data_chunk_that_is_not_whole_frames_is_truncated_to_them() {
        // Three bytes of a fourth stereo frame — a truncated recording, and the
        // one shape that would otherwise read past the end of the chunk.
        let mut data = le16(&[1, 2, 3, 4, 5, 6]);
        data.extend_from_slice(&[7, 0, 8]);
        let (_, samples) = parse(&wav(PCM, 2, 16_000, 16, &data, &[])).unwrap();
        assert_eq!(samples, [1, 3, 5]);
    }

    #[test]
    fn a_file_that_is_not_wave_is_refused_before_anything_is_read() {
        assert!(parse(b"").is_err());
        assert!(parse(b"RIFFxxxxAVI ").is_err());
        assert!(parse(b"OggS\0\0\0\0\0\0\0\0").is_err());
        // RIFF/WAVE with no chunks at all: named for what is missing.
        let err = parse(b"RIFF\x04\0\0\0WAVE").unwrap_err();
        assert!(format!("{err}").contains("fmt"), "{err}");
    }

    #[test]
    fn parsing_the_same_file_twice_gives_the_same_samples() {
        // §4.6 reaches through this the way it reaches through the optimizer:
        // the conversion is integer and IEEE-basic throughout, so it is a pure
        // function of the bytes and of nothing about the host.
        let data: Vec<u8> = (0..600u32)
            .flat_map(|i| ((i * 97) as f32 / 30_000.0 - 1.0).to_le_bytes())
            .collect();
        let file = wav(FLOAT, 3, 48_000, 32, &data, &[]);
        assert_eq!(parse(&file).unwrap().1, parse(&file).unwrap().1);
    }
}
