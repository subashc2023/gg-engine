//! The per-tick bitwise NaN scan over hashed state (§1.13 hazard 6).
//!
//! IEEE 754 pins *when* a NaN is produced but not its sign or payload, and the
//! architectures disagree — `0.0 / 0.0` is `0xFFC0_0000` under SSE2 and
//! `0x7FC0_0000` on AArch64. The canonical hash absorbs raw bits, so one NaN
//! reaching a hashed component is a §5.6 cross-architecture divergence wearing a
//! math bug's clothes. It is banned rather than canonicalized around, which
//! makes naming it *the tick it is born* the whole job.
//!
//! Bitwise, never `f32::is_nan`: the scalar is read out of the column with
//! `from_le_bytes` at an unaligned offset and its exponent/mantissa pattern
//! tested directly, so nothing between the bytes and the test can canonicalize a
//! payload away.
//!
//! Where the floats *are* comes from [`FieldDesc::ty`](crate::FieldDesc::ty),
//! the syntactic source token, classified once at registration into contiguous
//! same-width runs — re-parsing tokens sixty times a second is not the cheap
//! pass §1.13 promises. A token the classifier does not know is reported through
//! [`Registry::unclassified_fields`](crate::Registry::unclassified_fields)
//! rather than skipped: an uncovered field is a hole in the gate, and a silent
//! hole is worse than no gate.
//!
//! Side tables are hashed state as well and have no layout to walk, so they are
//! covered the other way round: hashed into a throwaway
//! [`StateHasher`](crate::StateHasher), which witnesses every float it absorbs.
//! Coarser — it names the table and not the field, because the protocol carries
//! no field names — and structural, since a float cannot reach the canonical
//! hash without passing the funnel that watches it.

use core::fmt;

use crate::component::FieldDesc;
use crate::entity::Entity;

/// One contiguous run of same-width IEEE scalars inside a component row.
///
/// Per *field*, never merged across two: the report has to name the field, and
/// a merged span could not. Within a field the run is whole — a `sim::DMat4` is
/// sixteen lanes at one offset, not sixteen spans.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FloatSpan {
    offset: u32,
    lanes: u32,
    /// `f64` lanes rather than `f32`.
    wide: bool,
    field: &'static str,
}

/// All-ones exponent with a nonzero mantissa — quiet, signalling and both signs
/// alike, which is the point of testing bits rather than calling `is_nan()`.
/// `±inf` has a zero mantissa and is deliberately not a hit.
pub(crate) const fn is_nan32(bits: u32) -> bool {
    bits & 0x7f80_0000 == 0x7f80_0000 && bits & 0x007f_ffff != 0
}

/// As [`is_nan32`], at double width.
pub(crate) const fn is_nan64(bits: u64) -> bool {
    bits & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000 && bits & 0x000f_ffff_ffff_ffff != 0
}

impl FloatSpan {
    /// The first NaN lane in `row`, as `(lane, bits)`. `row` is one component's
    /// bytes; the span is in range by construction (see [`spans_of`]).
    fn first_nan(self, row: &[u8]) -> Option<(u32, u64)> {
        let start = self.offset as usize;
        let width = if self.wide { 8 } else { 4 };
        // Sliced once, then `chunks_exact` — one bounds check per span instead
        // of one per lane, which is most of what this pass costs. The span is in
        // range by construction (see `spans_of`); `?` keeps a storage bug from
        // becoming a panic mid-tick.
        let span = row.get(start..start + self.lanes as usize * width)?;
        for (lane, scalar) in span.chunks_exact(width).enumerate() {
            // Little-endian by target (§4.2.1's raw fast path rests on the same
            // fact); read as bits, never as a value, so nothing on the way here
            // can canonicalize a payload.
            let nan = if self.wide {
                let bits = u64::from_le_bytes(scalar.try_into().ok()?);
                is_nan64(bits).then_some(bits)
            } else {
                let bits = u32::from_le_bytes(scalar.try_into().ok()?);
                is_nan32(bits).then_some(u64::from(bits))
            };
            if let Some(bits) = nan {
                return Some((lane as u32, bits));
            }
        }
        None
    }
}

/// A field whose `ty` token the scan could not place — the gate's own blind
/// spot, published so a caller can say so out loud (§1.13 hazard 6).
///
/// Reached two ways: a nested `#[derive(StateHash, Component)]` struct field,
/// whose token names a type this crate has no table entry for; and a token that
/// *is* known but whose declared size disagrees with it, which is a game type
/// shadowing an engine name rather than the engine type it looks like.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnclassifiedField {
    /// The component's declared id — persisted identity, as a report names it.
    pub component: &'static str,
    /// The Rust type name. Diagnostics only.
    pub type_name: &'static str,
    /// The field name.
    pub field: &'static str,
    /// The source type token that could not be placed.
    pub ty: &'static str,
}

impl fmt::Display for UnclassifiedField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}::{} (component \"{}\") has type token `{}`, which the NaN scan cannot classify",
            self.type_name, self.field, self.component, self.ty
        )
    }
}

/// Where a NaN was found, the tick it was found (§1.13 hazard 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NanHit {
    /// The entity whose row holds it.
    pub entity: Entity,
    /// The component's declared id.
    pub component: &'static str,
    /// The Rust type name.
    pub type_name: &'static str,
    /// The field within the component.
    pub field: &'static str,
    /// Scalar index within that field — which of a `DMat4`'s sixteen.
    pub lane: u32,
    /// Scalar width in bits: 32 or 64.
    pub width: u32,
    /// The offending bit pattern, zero-extended for `f32`. Reported because the
    /// payload is exactly what differs between architectures.
    pub bits: u64,
}

impl fmt::Display for NanHit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NaN in {:?}: {}::{} lane {} (component \"{}\", f{}, bits {:#018x})",
            self.entity,
            self.type_name,
            self.field,
            self.lane,
            self.component,
            self.width,
            self.bits
        )
    }
}

/// A NaN in a side table (§4.2.1) — hashed state that is not flat `Pod` memory,
/// so it is caught by witnessing the hash rather than by walking a layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SideTableNan {
    /// The table's declared id — persisted identity.
    pub declared: &'static str,
    /// The Rust type name.
    pub type_name: &'static str,
    /// Scalar width in bits: 32 or 64.
    pub width: u32,
    /// The offending bit pattern, zero-extended for `f32`.
    pub bits: u64,
}

impl fmt::Display for SideTableNan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NaN in side table {} (declared \"{}\", f{}, bits {:#018x})",
            self.type_name, self.declared, self.width, self.bits
        )
    }
}

/// Where in hashed state a NaN was found (§1.13 hazard 6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NanSite {
    /// A `Pod` component column, located to the entity, field and lane.
    Column(NanHit),
    /// A side table, located to the table and no further. The hash protocol
    /// carries values and not field names, so the type is as fine as the answer
    /// gets — enough to name whose author has the next question.
    SideTable(SideTableNan),
}

impl NanSite {
    /// The column hit, or `None` for a side table. Mostly for tests, which
    /// assert on coordinates only a column has.
    #[must_use]
    pub const fn column(self) -> Option<NanHit> {
        match self {
            Self::Column(hit) => Some(hit),
            Self::SideTable(_) => None,
        }
    }
}

impl fmt::Display for NanSite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Column(hit) => hit.fmt(f),
            Self::SideTable(hit) => hit.fmt(f),
        }
    }
}

/// What one `ty` token means to the scan.
#[derive(Clone, Copy)]
struct Class {
    /// Scalar count. Zero means classified and float-free — an integer field is
    /// not a hole.
    lanes: usize,
    wide: bool,
    /// Byte size the token implies, cross-checked against the declared size.
    size: usize,
}

const fn scalars(lanes: usize, wide: bool) -> Option<Class> {
    Some(Class {
        lanes,
        wide,
        size: lanes * if wide { 8 } else { 4 },
    })
}

const fn opaque(size: usize) -> Option<Class> {
    Some(Class {
        lanes: 0,
        wide: false,
        size,
    })
}

/// Classify one source type token, or `None` for a token with no entry.
///
/// The token is syntactic and its path prefix is whatever the declaring module
/// wrote (`sim :: DVec3`, `DVec3`, `gg_math :: sim :: DVec3` all name one type),
/// so matching is on the last path segment. That alone would accept a game's own
/// `Vec3`; the declared size is cross-checked at the call site, which is what
/// turns a shadowed name into an [`UnclassifiedField`] instead of a false NaN.
fn classify(token: &str) -> Option<Class> {
    // The derive normalizes to single spaces and the wire carries that string
    // through unchanged; stripping the rest makes `[u32 ; 4]` and `[u32;4]` one
    // token without a parser.
    let mut squeezed = String::with_capacity(token.len());
    squeezed.extend(token.chars().filter(|c| !c.is_whitespace()));
    classify_squeezed(&squeezed)
}

fn classify_squeezed(t: &str) -> Option<Class> {
    if let Some(inner) = t.strip_prefix('[') {
        // `[T;N]`, and nested — the length binds last, so split from the right.
        let (elem, count) = inner.strip_suffix(']')?.rsplit_once(';')?;
        let n: usize = count.parse().ok()?;
        let c = classify_squeezed(elem)?;
        return Some(Class {
            lanes: c.lanes * n,
            wide: c.wide,
            size: c.size * n,
        });
    }
    // The universe is closed: it is exactly the `StateHash` impls (§4.2.1), and
    // anything else is deliberately a hole rather than a guess.
    match t.rsplit("::").next()? {
        "f32" => scalars(1, false),
        "f64" => scalars(1, true),
        "Vec2" => scalars(2, false),
        "Vec3" => scalars(3, false),
        "Vec4" | "Quat" => scalars(4, false),
        "Mat3" => scalars(9, false),
        "Mat4" => scalars(16, false),
        "DVec2" => scalars(2, true),
        "DVec3" => scalars(3, true),
        "DVec4" | "DQuat" => scalars(4, true),
        "DMat3" => scalars(9, true),
        "DMat4" => scalars(16, true),
        "u8" | "i8" | "bool" => opaque(1),
        "u16" | "i16" => opaque(2),
        "u32" | "i32" => opaque(4),
        "u64" | "i64" | "Entity" | "ComponentId" => opaque(8),
        "u128" | "i128" => opaque(16),
        _ => None,
    }
}

/// The float spans of one component, and whatever could not be placed.
///
/// Called from [`Registry::insert`](crate::Registry) — once per component, not
/// once per tick, which is what keeps the scan a straight linear pass.
pub(crate) fn spans_of(
    declared_id: &'static str,
    type_name: &'static str,
    size: usize,
    fields: &'static [FieldDesc],
    holes: &mut Vec<UnclassifiedField>,
) -> Vec<FloatSpan> {
    let mut spans = Vec::new();
    for field in fields {
        // Size and bounds are both checks, not assumptions: a wire-declared
        // component's descriptors are a dylib's claim, and a span past the end
        // of a row would read another entity's bytes.
        let placed = classify(field.ty)
            .filter(|c| c.size == field.size && field.offset.saturating_add(field.size) <= size);
        match placed {
            Some(c) if c.lanes > 0 => spans.push(FloatSpan {
                offset: field.offset as u32,
                lanes: c.lanes as u32,
                wide: c.wide,
                field: field.name,
            }),
            Some(_) => {}
            None => holes.push(UnclassifiedField {
                component: declared_id,
                type_name,
                field: field.name,
                ty: field.ty,
            }),
        }
    }
    spans
}

/// The first NaN in any hashed column, or `None` — the body of
/// [`World::scan_for_nan`](crate::World::scan_for_nan).
///
/// Archetype creation order, then dense row, then span: the answer is a
/// function of the world and not of the search. Column-major would touch each
/// span's bytes contiguously; row-major is what makes stopping at the first hit
/// stop early, and stopping is the point.
pub(crate) fn scan(
    archetypes: &[crate::archetype::Archetype],
    registry: &crate::Registry,
) -> Option<NanHit> {
    for archetype in archetypes {
        if archetype.is_empty() {
            continue;
        }
        let entities = archetype.entities_slice();
        for (at, id) in archetype.ids().iter().enumerate() {
            // One registry lookup per column, never per row.
            let Some((info, spans)) = registry.floats_of(*id) else {
                continue;
            };
            if spans.is_empty() {
                continue;
            }
            let column = archetype.column(at);
            let stride = column.stride();
            let bytes = column.as_bytes();
            for (row, &entity) in entities.iter().enumerate() {
                let Some(row_bytes) = bytes.get(row * stride..(row + 1) * stride) else {
                    break;
                };
                for span in spans {
                    if let Some((lane, bits)) = span.first_nan(row_bytes) {
                        return Some(NanHit {
                            entity,
                            component: info.declared_id,
                            type_name: info.type_name,
                            field: span.field,
                            lane,
                            width: if span.wide { 64 } else { 32 },
                            bits,
                        });
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use gg_math::sim;

    use super::*;
    use crate::World;
    use crate::boundary::{Eye, Light, Model, Node, Renderable};
    use crate::hash::{ComponentId, SchemaHash};
    use crate::registry::ComponentInfo;

    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, crate::Component)]
    #[component(id = "nan-scan.narrow")]
    #[repr(C)]
    struct Narrow {
        offset: sim::Vec3,
        strength: f32,
    }

    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, crate::Component)]
    #[component(id = "nan-scan.wide")]
    #[repr(C)]
    struct Wide {
        position: sim::DVec3,
        rotation: sim::DQuat,
        weights: [f64; 2],
    }

    /// The false positive that would make the gate useless: integer bytes that
    /// spell a NaN.
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, crate::Component)]
    #[component(id = "nan-scan.bits")]
    #[repr(C)]
    struct Bits {
        pair: u64,
        packed: u32,
        spare: u32,
        mask: [u32; 4],
    }

    fn nan_f32() -> f32 {
        f32::from_bits(0x7fc0_0001)
    }

    fn nan_f64() -> f64 {
        f64::from_bits(0x7ff8_0000_0000_0001)
    }

    #[test]
    fn a_clean_world_reports_nothing() {
        let mut world = World::new();
        for i in 0..64u32 {
            let e = world.spawn();
            world
                .insert(
                    e,
                    Narrow {
                        offset: sim::Vec3::splat(i as f32),
                        strength: 1.0,
                    },
                )
                .unwrap();
            world
                .insert(
                    e,
                    Wide {
                        position: sim::DVec3::splat(f64::from(i)),
                        rotation: sim::DQuat::IDENTITY,
                        weights: [0.0, -1.0],
                    },
                )
                .unwrap();
        }
        assert!(world.scan_for_nan().is_none());
    }

    #[test]
    fn a_narrow_nan_names_its_entity_component_and_field() {
        let mut world = World::new();
        let clean = world.spawn();
        world
            .insert(
                clean,
                Narrow {
                    offset: sim::Vec3::ZERO,
                    strength: 1.0,
                },
            )
            .unwrap();
        let dirty = world.spawn();
        world
            .insert(
                dirty,
                Narrow {
                    offset: sim::Vec3::new(0.0, nan_f32(), 0.0),
                    strength: 1.0,
                },
            )
            .unwrap();

        let hit = world
            .scan_for_nan()
            .and_then(NanSite::column)
            .expect("the NaN must be found");
        assert_eq!(hit.entity, dirty);
        assert_eq!(hit.component, "nan-scan.narrow");
        assert_eq!(hit.type_name, "Narrow");
        assert_eq!(hit.field, "offset");
        assert_eq!(hit.lane, 1, "the y lane, not the field");
        assert_eq!(hit.width, 32);
        assert_eq!(hit.bits, 0x7fc0_0001);
    }

    #[test]
    fn a_wide_nan_is_found_in_the_field_that_holds_it() {
        let mut world = World::new();
        let e = world.spawn();
        world
            .insert(
                e,
                Wide {
                    position: sim::DVec3::ZERO,
                    rotation: sim::DQuat::IDENTITY,
                    weights: [0.0, nan_f64()],
                },
            )
            .unwrap();
        let hit = world
            .scan_for_nan()
            .and_then(NanSite::column)
            .expect("the NaN must be found");
        assert_eq!(hit.entity, e);
        assert_eq!(hit.field, "weights");
        assert_eq!(hit.lane, 1);
        assert_eq!(hit.width, 64);
        assert_eq!(hit.bits, 0x7ff8_0000_0000_0001);
    }

    #[test]
    fn every_sign_and_payload_counts_as_a_nan() {
        // The architectures disagree on exactly these bits (§1.13 hazard 6), so
        // a scan that only caught the positive quiet one would be blind on half
        // the targets. `-inf` and `inf` are *not* NaNs and must pass.
        for bits in [
            0x7fc0_0000u32,
            0xffc0_0000,
            0x7f80_0001,
            0xffff_ffff,
            0x7f80_0000,
            0xff80_0000,
        ] {
            let mut world = World::new();
            let e = world.spawn();
            world
                .insert(
                    e,
                    Narrow {
                        offset: sim::Vec3::ZERO,
                        strength: f32::from_bits(bits),
                    },
                )
                .unwrap();
            let found = world.scan_for_nan().is_some();
            let expected = bits & 0x7f80_0000 == 0x7f80_0000 && bits & 0x007f_ffff != 0;
            assert_eq!(found, expected, "bits {bits:#010x}");
        }
    }

    #[test]
    fn integer_bytes_that_spell_a_nan_are_not_reported() {
        let mut world = World::new();
        let e = world.spawn();
        world
            .insert(
                e,
                Bits {
                    pair: 0x7ff8_0000_0000_0001,
                    packed: 0x7fc0_0001,
                    spare: 0xffc0_0001,
                    mask: [0xffff_ffff; 4],
                },
            )
            .unwrap();
        assert!(
            world.scan_for_nan().is_none(),
            "integer columns are not float columns, whatever their bytes spell"
        );
    }

    #[test]
    fn every_engine_declared_component_classifies() {
        let mut world = World::new();
        world.register::<Renderable>().unwrap();
        world.register::<Eye>().unwrap();
        world.register::<Model>().unwrap();
        world.register::<Light>().unwrap();
        world.register::<Node>().unwrap();
        let holes = world.registry().unclassified_fields();
        assert!(
            holes.is_empty(),
            "the engine's own components must leave the scan no hole: {holes:?}"
        );
    }

    #[test]
    fn a_nested_struct_field_is_a_named_hole_not_a_silent_skip() {
        // What a game reaches by putting a `#[derive(StateHash, Component)]`
        // struct inside a component: a token with no table entry.
        let fields: &'static [FieldDesc] = &[FieldDesc {
            name: "pose",
            ty: "game :: Pose",
            offset: 0,
            size: 8,
        }];
        let mut world = World::new();
        world
            .registry_mut()
            .register_declared(ComponentInfo {
                id: ComponentId::of("nested"),
                schema: SchemaHash::from_raw(1),
                declared_id: "nested",
                type_name: "Nested",
                size: 8,
                align: 8,
                fields,
                protocol_hash: |bytes, h| h.bytes(bytes),
            })
            .unwrap();
        let holes = world.registry().unclassified_fields();
        assert_eq!(holes.len(), 1);
        assert_eq!(holes[0].field, "pose");
        assert_eq!(holes[0].ty, "game :: Pose");
        assert!(holes[0].to_string().contains("Nested::pose"));
    }

    #[test]
    fn a_known_name_with_the_wrong_size_is_a_hole_not_a_float_run() {
        // A game type called `Vec3` that is three `u32`s: same name, same size
        // as `sim::Vec3`, different meaning. The size check cannot separate
        // those two, so this uses one that differs — which is the case the check
        // exists for.
        let fields: &'static [FieldDesc] = &[FieldDesc {
            name: "v",
            ty: "Vec3",
            offset: 0,
            size: 6,
        }];
        let mut world = World::new();
        world
            .registry_mut()
            .register_declared(ComponentInfo {
                id: ComponentId::of("shadowed"),
                schema: SchemaHash::from_raw(2),
                declared_id: "shadowed",
                type_name: "Shadowed",
                size: 6,
                align: 2,
                fields,
                protocol_hash: |bytes, h| h.bytes(bytes),
            })
            .unwrap();
        assert_eq!(world.registry().unclassified_fields().len(), 1);
    }

    #[test]
    fn a_dylib_declared_component_carries_its_type_token_through_the_wire() {
        // The whole point of the `ty` token crossing §4.2.2: the host holds no
        // type for this component and the scan still knows where its floats are.
        const NAME: &str = "Foreign";
        const DECLARED: &str = "nan-scan.foreign";
        const FIELD: &str = "velocity";
        const TY: &str = "sim :: DVec3";
        // A local rather than a `static`: `FieldLayout` holds raw pointers and
        // is not `Sync`. It outlives `adopt`, which is all the wire needs — what
        // `adopt` keeps is `&'static str`s over the three literals below.
        let fields = [gg_abi::FieldLayout {
            name: FIELD.as_ptr(),
            ty: TY.as_ptr(),
            name_len: FIELD.len() as u32,
            ty_len: TY.len() as u32,
            offset: 0,
            size: 24,
        }];
        let id = ComponentId::of(DECLARED);
        let layout = gg_abi::ComponentLayout {
            name: NAME.as_ptr(),
            fields: fields.as_ptr(),
            declared: DECLARED.as_ptr(),
            id: id.get(),
            schema_hash: [7; 16],
            name_len: NAME.len() as u32,
            declared_len: DECLARED.len() as u32,
            field_len: 1,
            size: 24,
            align: 8,
        };
        let table = gg_abi::ComponentsTable {
            entries: core::ptr::from_ref(&layout),
            len: 1,
            reserved: 0,
        };

        let mut world = World::new();
        // SAFETY: `layout`, `fields` and the four `&'static str`s all outlive
        // this call, which is what `adopt`'s never-unloaded-dylib rule buys.
        let declared = unsafe { world.adopt(&table) }.unwrap();
        assert_eq!(declared, 1);
        assert!(world.registry().unclassified_fields().is_empty());

        let e = world.spawn();
        let mut bytes = [0u8; 24];
        bytes[8..16].copy_from_slice(&nan_f64().to_le_bytes());
        assert_eq!(
            world.insert_raw(e, id, &bytes),
            gg_abi::AbiStatus::Ok,
            "the raw insert is the only way in for a host-typeless component"
        );

        let hit = world
            .scan_for_nan()
            .and_then(NanSite::column)
            .expect("the NaN must be found");
        assert_eq!(hit.entity, e);
        assert_eq!(hit.component, DECLARED);
        assert_eq!(hit.field, FIELD);
        assert_eq!(hit.lane, 1);
        assert_eq!(hit.width, 64);
    }

    #[test]
    fn array_tokens_flatten_including_nested_ones() {
        let c = classify("[[f32 ; 3] ; 4]").unwrap();
        assert_eq!((c.lanes, c.wide, c.size), (12, false, 48));
        let c = classify("[sim :: DQuat ; 2]").unwrap();
        assert_eq!((c.lanes, c.wide, c.size), (8, true, 64));
        let c = classify("[u32 ; 4]").unwrap();
        assert_eq!((c.lanes, c.size), (0, 16));
        assert!(classify("[Pose ; 2]").is_none());
    }

    #[test]
    fn the_path_prefix_does_not_change_the_answer() {
        for token in ["DVec3", "sim :: DVec3", "gg_math :: sim :: DVec3"] {
            let c = classify(token).unwrap();
            assert_eq!((c.lanes, c.wide, c.size), (3, true, 24), "{token}");
        }
    }
}
