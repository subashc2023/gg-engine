//! The picture in the taskbar (§6 M46) — raw RGBA behind an eight-byte header.
//!
//! **A format rather than a PNG, and the reason is a ban rather than a
//! preference.** `deny.toml` keeps every image decoder out of the dist graph
//! (§3), and a window icon is wanted at window creation — before the pack that
//! would otherwise carry it is open, and before the renderer exists at all. So
//! the shipped bytes are the pixels: a magic, a side, and `side * side * 4`.
//!
//! Written by `gg-tools icon`, never by hand, on `panorama`'s and `timbre`'s
//! rule — a checked-in binary in this tree has a source that regenerates it.
//! What that buys here beyond tidiness: an icon is the one artifact a reviewer
//! cannot diff, so the diff is the generator.
//!
//! Square because every platform's icon is, and one size because the OS scales:
//! Windows wants 16 through 256 and picks, X11 takes a list, and shipping one
//! good 64 is what the alternative — a chain nobody looks at — is worth.

/// What a file of these starts with. Four bytes, so a truncated or misnamed
/// file fails by name rather than by drawing noise.
pub const MAGIC: [u8; 4] = *b"GGIC";

/// Bytes before the pixels.
const HEADER: usize = 8;

/// The side this tree writes. Not a limit — [`parse`] takes what the file says —
/// but the one `gg-tools icon` produces and the one `xtask ship` stages.
///
/// 96 rather than a power of two: it halves to 48, 24 and 12 exactly, so every
/// size an OS is likely to ask for is an integer box filter of these pixels
/// rather than a resample, and it divides by the six-cell grid `gg-tools icon`
/// draws on.
pub const SIDE: u32 = 96;

/// Why a file was not an icon.
#[derive(Debug, thiserror::Error)]
pub enum IconError {
    /// Not this format at all.
    #[error("not an icon: expected the bytes `GGIC`")]
    Magic,
    /// The header and the body disagree, which is a truncated file in every case
    /// anyone has seen — so it says the two numbers rather than "corrupt".
    #[error("an icon {side}px square wants {want} bytes of pixels and the file has {have}")]
    Truncated {
        /// The side the header claims.
        side: u32,
        /// Bytes the claim needs.
        want: usize,
        /// Bytes present.
        have: usize,
    },
    /// A side of zero, or one large enough that `side * side * 4` overflows.
    #[error("`{0}` is not a plausible icon side")]
    Side(u32),
}

/// The pixels a file holds, as `(side, rgba)` — row-major from the top left,
/// eight bits a channel, alpha last and not premultiplied (winit's convention,
/// which is the only consumer).
///
/// # Errors
/// [`IconError`] for a file that is not this format or does not hold what its
/// own header claims. Never panics on hostile bytes: this reads a file from a
/// folder a player unzipped.
pub fn parse(bytes: &[u8]) -> Result<(u32, Vec<u8>), IconError> {
    if bytes.len() < HEADER || bytes[..4] != MAGIC {
        return Err(IconError::Magic);
    }
    let side = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    // 1024 is well past any platform's largest and keeps the multiply below
    // inside a `usize` on every target this builds for.
    let want = match side {
        1..=1024 => side as usize * side as usize * 4,
        _ => return Err(IconError::Side(side)),
    };
    let have = bytes.len() - HEADER;
    if have != want {
        return Err(IconError::Truncated { side, want, have });
    }
    Ok((side, bytes[HEADER..].to_vec()))
}

/// The file for `side` square of `rgba`, ready to write.
///
/// # Panics
/// If `rgba` is not `side * side * 4` bytes — the caller generated it and a
/// mismatch is a bug in the generator rather than a runtime condition.
#[must_use]
pub fn encode(side: u32, rgba: &[u8]) -> Vec<u8> {
    assert_eq!(
        rgba.len(),
        side as usize * side as usize * 4,
        "an icon's pixels must match its side"
    );
    let mut out = Vec::with_capacity(HEADER + rgba.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&side.to_le_bytes());
    out.extend_from_slice(rgba);
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn an_icon_round_trips_through_its_own_file() {
        let rgba: Vec<u8> = (0..4 * 4 * 4).map(|i| (i % 251) as u8).collect();
        let (side, back) = parse(&encode(4, &rgba)).unwrap();
        assert_eq!((side, back), (4, rgba));
    }

    /// Every way a file in a player's folder can be wrong, refused by name. The
    /// subject is a folder someone unzipped, so "never panics" is the claim and
    /// the empty slice is the one that would.
    #[test]
    fn a_file_that_is_not_an_icon_is_refused_rather_than_drawn() {
        assert!(matches!(parse(&[]), Err(IconError::Magic)));
        assert!(matches!(parse(b"PNG\x0d"), Err(IconError::Magic)));
        assert!(matches!(parse(b"GGIC"), Err(IconError::Magic)), "no side");
        let whole = encode(2, &[0; 2 * 2 * 4]);
        assert!(matches!(
            parse(&whole[..whole.len() - 1]),
            Err(IconError::Truncated { side: 2, .. })
        ));
        let mut zero = whole.clone();
        zero[4..8].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(parse(&zero), Err(IconError::Side(0))));
        let mut huge = whole;
        huge[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(parse(&huge), Err(IconError::Side(u32::MAX))));
    }
}
