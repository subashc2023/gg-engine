//! Field values, read and nudged through the schema rather than through a type
//! (§6 M15's inspector).
//!
//! A component chosen at runtime has no `T` to name, so a field is reached by
//! `(offset, size)` out of [`FieldDesc`] and interpreted by its **type token**.
//! The vocabulary is closed and is the same one `gg_ecs::nan_scan` classifies
//! against — deliberately, because a token neither of them knows is a field the
//! NaN scan does not cover *and* the inspector must not pretend to understand.
//! An unknown token shows its bytes in hex and refuses to be edited.
//!
//! # Why every edit is a nudge
//!
//! The inspector has no text entry. Typing straight off the keyboard would mean
//! keystrokes outside the action map — the path `gg_debug`'s console takes — and
//! those are not in a replay (§4.7); a recorded editor session that replayed to
//! a different world would fail §6 M15's fourth exit row. The one place text is
//! typed is the agent panel's prompt (`chat`), sound for the opposite reason:
//! printable input rides the replay's own text channel. Nothing routes that
//! channel into a field, so the inspector edits by `-` and `+` and nothing else.
//! The cost is real and is the right price: reaching 100.0 takes ten clicks at
//! step 10.

use gg_ecs::component::FieldDesc;

/// What one lane of a field is. Widths come from the token, never from the
/// field size — `[f32; 4]` and `Vec4` are four `F32` lanes either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// IEEE binary32.
    F32,
    /// IEEE binary64.
    F64,
    /// Unsigned integer of `n` bytes.
    U(usize),
    /// Signed integer of `n` bytes.
    I(usize),
    /// One byte, shown as `true`/`false`.
    Bool,
    /// Two `u32`s, shown as `index:generation`.
    Entity,
}

impl Kind {
    /// Bytes one lane occupies.
    pub fn width(self) -> usize {
        match self {
            Kind::F32 => 4,
            Kind::F64 => 8,
            Kind::U(n) | Kind::I(n) => n,
            Kind::Bool => 1,
            Kind::Entity => 8,
        }
    }
}

/// The lane kind a type token denotes, or `None` for one outside the closed
/// vocabulary — the same universe `gg_ecs::nan_scan` places, for the reason the
/// module docs give.
///
/// Arrays strip to their element; the *count* is the field's size over the
/// lane width, so `[[f32; 4]; 4]` and `Mat4` are both sixteen lanes.
pub fn kind_of(token: &str) -> Option<Kind> {
    let mut squeezed = String::with_capacity(token.len());
    squeezed.extend(token.chars().filter(|c| !c.is_whitespace()));
    let mut t = squeezed.as_str();
    // `[T; N]`, nested — the length binds last, so strip from the right.
    while let Some(inner) = t.strip_prefix('[') {
        t = inner.strip_suffix(']')?.rsplit_once(';')?.0;
    }
    Some(match t.rsplit("::").next()? {
        "f32" | "Vec2" | "Vec3" | "Vec4" | "Quat" | "Mat3" | "Mat4" => Kind::F32,
        "f64" | "DVec2" | "DVec3" | "DVec4" | "DQuat" | "DMat3" | "DMat4" => Kind::F64,
        "bool" => Kind::Bool,
        "Entity" => Kind::Entity,
        "u8" => Kind::U(1),
        "u16" => Kind::U(2),
        "u32" => Kind::U(4),
        "u64" | "ComponentId" => Kind::U(8),
        "u128" => Kind::U(16),
        "i8" => Kind::I(1),
        "i16" => Kind::I(2),
        "i32" => Kind::I(4),
        "i64" => Kind::I(8),
        "i128" => Kind::I(16),
        _ => return None,
    })
}

/// Lanes in `field`, or `0` for a token this build cannot place.
pub fn lanes(field: &FieldDesc) -> usize {
    kind_of(field.ty).map_or(0, |k| field.size / k.width())
}

/// `lane`'s byte range within the component, or `None` if the field's token is
/// unknown or the lane is past its end.
pub fn span(field: &FieldDesc, lane: usize) -> Option<(Kind, core::ops::Range<usize>)> {
    let kind = kind_of(field.ty)?;
    let at = field.offset + lane * kind.width();
    (at + kind.width() <= field.offset + field.size).then(|| (kind, at..at + kind.width()))
}

/// Read one lane out of a component's bytes.
///
/// Native-endian throughout: these bytes were `memcpy`'d out of a live column,
/// so they are in the layout the compiler chose and nothing else would read
/// them correctly.
pub fn read(bytes: &[u8], field: &FieldDesc, lane: usize) -> Option<f64> {
    let (kind, at) = span(field, lane)?;
    let b = bytes.get(at)?;
    let n = |i: usize| u128::from(b[i]);
    let unsigned: u128 = (0..kind.width()).fold(0, |acc, i| acc | (n(i) << (8 * i)));
    Some(match kind {
        Kind::F32 => f64::from(f32::from_bits(unsigned as u32)),
        Kind::F64 => f64::from_bits(unsigned as u64),
        Kind::U(_) | Kind::Bool | Kind::Entity => unsigned as f64,
        // Sign-extend from the field's own width before widening.
        Kind::I(w) => {
            let shift = 128 - 8 * w;
            ((unsigned as i128) << shift >> shift) as f64
        }
    })
}

/// Format one lane as the inspector shows it. Floats carry two decimals — a
/// nudge of 0.25 has to be visible or the button looks broken.
pub fn show(bytes: &[u8], field: &FieldDesc, lane: usize) -> String {
    let Some((kind, at)) = span(field, lane) else {
        return "??".to_owned();
    };
    let Some(value) = read(bytes, field, lane) else {
        return "??".to_owned();
    };
    match kind {
        Kind::F32 | Kind::F64 => format!("{value:.2}"),
        Kind::Bool => if value == 0.0 { "false" } else { "true" }.to_owned(),
        Kind::Entity => {
            let halves = |o: usize| {
                u32::from_ne_bytes(
                    bytes[at.start + o..at.start + o + 4]
                        .try_into()
                        .unwrap_or([0; 4]),
                )
            };
            format!("{}:{}", halves(0), halves(4))
        }
        Kind::U(_) | Kind::I(_) => format!("{value:.0}"),
    }
}

/// The whole field as hex, for a token outside the vocabulary. Truncated,
/// because a 32-byte label in a 28-column panel is a wall.
pub fn hex(bytes: &[u8], field: &FieldDesc) -> String {
    let end = (field.offset + field.size).min(bytes.len());
    let shown = &bytes[field.offset.min(end)..end];
    let mut out = String::new();
    for byte in shown.iter().take(6) {
        out.push_str(&format!("{byte:02x}"));
    }
    if shown.len() > 6 {
        out.push('~');
    }
    out
}

/// Add `step` to one lane, in place. `false` if the field's token is unknown —
/// which is what makes an unrecognized field read-only rather than corruptible.
///
/// Integers saturate at their own width and booleans toggle; a float takes the
/// step as written. Ordinary IEEE addition, which is bit-identical on every
/// target §1.3 covers — this writes hashed state, so it has to be.
pub fn nudge(bytes: &mut [u8], field: &FieldDesc, lane: usize, step: f64) -> bool {
    let Some((kind, at)) = span(field, lane) else {
        return false;
    };
    if bytes.len() < at.end {
        return false;
    }
    let current = read(bytes, field, lane).unwrap_or(0.0);
    match kind {
        Kind::F32 => bytes[at].copy_from_slice(&((current as f32) + step as f32).to_ne_bytes()),
        Kind::F64 => bytes[at].copy_from_slice(&(current + step).to_ne_bytes()),
        Kind::Bool => bytes[at.start] = u8::from(current == 0.0),
        // An entity handle is a pair of indices, not a number: stepping one
        // would mint a handle at random. Left alone deliberately.
        Kind::Entity => return false,
        Kind::U(w) => {
            let stepped = (current + step).max(0.0).min(u128::MAX as f64) as u128;
            let ceiling = if w >= 16 {
                u128::MAX
            } else {
                (1u128 << (8 * w)) - 1
            };
            let value = stepped.min(ceiling);
            for (i, slot) in bytes[at].iter_mut().enumerate() {
                *slot = (value >> (8 * i)) as u8;
            }
        }
        Kind::I(w) => {
            let span = 1i128 << (8 * w - 1);
            let value = ((current + step) as i128).clamp(-span, span - 1);
            for (i, slot) in bytes[at].iter_mut().enumerate() {
                *slot = (value >> (8 * i)) as u8;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn field(ty: &'static str, size: usize) -> FieldDesc {
        FieldDesc {
            name: "f",
            ty,
            offset: 0,
            size,
        }
    }

    #[test]
    fn a_vector_is_its_scalar_repeated_and_an_array_strips_to_its_element() {
        assert_eq!(kind_of("gg_math::sim::DVec3"), Some(Kind::F64));
        assert_eq!(lanes(&field("DVec3", 24)), 3);
        assert_eq!(lanes(&field("[f32; 4]", 16)), 4);
        assert_eq!(lanes(&field("[[f32; 4]; 4]", 64)), 16);
        assert_eq!(lanes(&field("Mat4", 64)), 16);
        // Outside the vocabulary is zero lanes, not a guess.
        assert_eq!(kind_of("SomeStruct"), None);
        assert_eq!(lanes(&field("SomeStruct", 12)), 0);
    }

    #[test]
    fn a_float_lane_reads_and_steps_where_the_schema_says_it_is() {
        let f = FieldDesc {
            name: "p",
            ty: "DVec3",
            offset: 8,
            size: 24,
        };
        let mut bytes = vec![0u8; 32];
        bytes[8..16].copy_from_slice(&1.5f64.to_ne_bytes());
        bytes[24..32].copy_from_slice(&(-2.0f64).to_ne_bytes());
        assert_eq!(read(&bytes, &f, 0), Some(1.5));
        assert_eq!(read(&bytes, &f, 2), Some(-2.0));
        assert_eq!(read(&bytes, &f, 3), None, "past the field");
        assert!(nudge(&mut bytes, &f, 2, 0.25));
        assert_eq!(show(&bytes, &f, 2), "-1.75");
        // The neighbouring lane is untouched — the offsets are the schema's.
        assert_eq!(read(&bytes, &f, 0), Some(1.5));
    }

    #[test]
    fn integers_saturate_at_their_own_width_and_bools_toggle() {
        let u8f = field("u8", 1);
        let mut bytes = vec![250u8];
        assert!(nudge(&mut bytes, &u8f, 0, 10.0));
        assert_eq!(bytes[0], 255, "clamped, not wrapped");
        assert!(nudge(&mut bytes, &u8f, 0, -300.0));
        assert_eq!(bytes[0], 0);

        let i16f = field("i16", 2);
        let mut signed = (-5i16).to_ne_bytes().to_vec();
        assert_eq!(read(&signed, &i16f, 0), Some(-5.0), "sign extended");
        assert!(nudge(&mut signed, &i16f, 0, -40_000.0));
        assert_eq!(read(&signed, &i16f, 0), Some(f64::from(i16::MIN)));

        let b = field("bool", 1);
        let mut flag = vec![0u8];
        assert!(nudge(&mut flag, &b, 0, 1.0));
        assert_eq!(show(&flag, &b, 0), "true");
        assert!(nudge(&mut flag, &b, 0, 1.0));
        assert_eq!(show(&flag, &b, 0), "false");
    }

    /// Two things the inspector must refuse rather than mangle: a handle, and a
    /// type this build has no idea about.
    #[test]
    fn a_handle_and_an_unknown_token_are_read_only() {
        let e = field("Entity", 8);
        let mut bytes = [7u32.to_ne_bytes(), 2u32.to_ne_bytes()].concat();
        assert_eq!(show(&bytes, &e, 0), "7:2");
        assert!(!nudge(&mut bytes, &e, 0, 1.0));
        assert_eq!(show(&bytes, &e, 0), "7:2", "untouched");

        let odd = field("Mystery", 4);
        let mut raw = vec![0xde, 0xad, 0xbe, 0xef];
        assert!(!nudge(&mut raw, &odd, 0, 1.0));
        assert_eq!(hex(&raw, &odd), "deadbeef");
        assert_eq!(raw, vec![0xde, 0xad, 0xbe, 0xef]);
    }
}
