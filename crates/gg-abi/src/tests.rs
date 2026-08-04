//! Layout assertions — the teeth behind [`HOST_API_VERSION`](crate::HOST_API_VERSION).
//!
//! The version number is only load-bearing if it moves whenever the layout
//! does. Nothing makes that automatic, so it is made *loud*: every size, every
//! alignment, every offset is pinned here, and any change to a boundary type
//! turns this file red before it can reach a running session as a silently
//! reinterpreted struct. Fixing one of these by editing the number without
//! bumping the version is the one wrong move available.
//!
//! The offsets matter as much as the sizes. Two artifacts agreeing on a struct's
//! total size while disagreeing about where a field sits is exactly the failure
//! this boundary is `repr(C)` to prevent.

use core::mem::{align_of, offset_of, size_of};

use crate::*;

#[test]
fn entity_is_two_words_with_no_padding() {
    assert_eq!(size_of::<Entity>(), 8);
    assert_eq!(align_of::<Entity>(), 4);
    assert_eq!(offset_of!(Entity, index), 0);
    assert_eq!(offset_of!(Entity, generation), 4);
}

#[test]
fn an_entity_survives_the_bit_encoding_and_zero_is_never_alive() {
    let e = Entity::new(7, 3);
    assert_eq!(Entity::from_bits(e.to_bits()), e);
    // The encoding is generation-high, which is *not* the memory order — the
    // point of `to_bits` being a value encoding rather than a transmute.
    assert_eq!(e.to_bits(), (3u64 << 32) | 7);
    assert!(Entity::NONE.is_none());
    assert!(!e.is_none());
}

#[test]
fn input_frame_is_seventy_two_bytes() {
    // Both: the first says the `repr(C)` shape carries no padding, the second
    // pins the absolute number so raising `MAX_AXES` is an edit here rather than
    // a silent growth of every replay change record.
    assert_eq!(size_of::<InputFrame>(), 8 + 4 * MAX_AXES);
    assert_eq!(size_of::<InputFrame>(), 72);
    assert_eq!(align_of::<InputFrame>(), 8);
    assert_eq!(offset_of!(InputFrame, buttons), 0);
    assert_eq!(offset_of!(InputFrame, axes), 8);
    assert_eq!(size_of::<ActionId>(), 1);
    assert_eq!(size_of::<AxisId>(), 1);
}

#[test]
fn an_axis_reads_back_exactly_because_the_scale_is_a_power_of_two() {
    let mut axes = [0; MAX_AXES];
    axes[0] = AXIS_SCALE;
    axes[1] = -AXIS_SCALE / 2;
    let frame = InputFrame {
        buttons: 0b101,
        axes,
    };
    assert!(frame.pressed(ActionId::new(0)));
    assert!(!frame.pressed(ActionId::new(1)));
    assert!(frame.pressed(ActionId::new(2)));
    assert_eq!(frame.axis(AxisId::new(0)), 1.0);
    assert_eq!(frame.axis(AxisId::new(1)), -0.5);
    assert!(AXIS_SCALE.unsigned_abs().is_power_of_two());
}

#[test]
fn tick_ctx_has_no_implicit_padding() {
    assert_eq!(offset_of!(TickCtx, tick), 0);
    assert_eq!(offset_of!(TickCtx, tick_hz), 8);
    assert_eq!(offset_of!(TickCtx, reserved), 12);
    assert_eq!(offset_of!(TickCtx, input), 16);
    // Two `InputFrame`s, so both numbers move with `MAX_AXES` — 16 bytes of
    // header, then this tick's frame and the one before it.
    assert_eq!(offset_of!(TickCtx, previous), 88);
    assert_eq!(size_of::<TickCtx>(), 160);
    assert_eq!(align_of::<TickCtx>(), 8);
}

#[test]
fn system_types_are_pinned() {
    assert_eq!(size_of::<SystemStatus>(), 16);
    assert_eq!(offset_of!(SystemStatus, code), 0);
    assert_eq!(offset_of!(SystemStatus, message_len), 4);
    assert_eq!(offset_of!(SystemStatus, message), 8);

    assert_eq!(size_of::<SystemEntry>(), 32);
    assert_eq!(offset_of!(SystemEntry, name), 0);
    assert_eq!(offset_of!(SystemEntry, run), 8);
    assert_eq!(offset_of!(SystemEntry, id), 16);
    assert_eq!(offset_of!(SystemEntry, name_len), 24);

    assert_eq!(size_of::<SystemsTable>(), 16);
}

#[test]
fn only_exactly_ok_reads_as_success() {
    assert!(SystemStatus::ok().is_ok());
    let garbage = SystemStatus {
        code: 9999,
        ..SystemStatus::ok()
    };
    assert!(!garbage.is_ok(), "an unknown code is a failure, not a pass");
}

#[test]
fn query_types_are_pinned() {
    assert_eq!(size_of::<ColumnView>(), 24);
    assert_eq!(offset_of!(ColumnView, ptr), 0);
    assert_eq!(offset_of!(ColumnView, len), 8);
    assert_eq!(offset_of!(ColumnView, stride), 16);

    assert_eq!(size_of::<QueryDesc>(), 24);
    assert_eq!(offset_of!(QueryDesc, read_len), 16);
    assert_eq!(offset_of!(QueryDesc, write_len), 20);

    assert_eq!(offset_of!(ArchetypeMatch, columns), 16);
    assert_eq!(size_of::<ArchetypeMatch>(), 16 + 24 * MAX_QUERY_COLUMNS);
}

#[test]
fn layout_descriptors_are_pinned() {
    assert_eq!(size_of::<FieldLayout>(), 32);
    assert_eq!(offset_of!(FieldLayout, offset), 24);

    assert_eq!(size_of::<ComponentLayout>(), 72);
    assert_eq!(offset_of!(ComponentLayout, declared), 16);
    assert_eq!(offset_of!(ComponentLayout, id), 24);
    assert_eq!(offset_of!(ComponentLayout, schema_hash), 32);
    assert_eq!(offset_of!(ComponentLayout, declared_len), 52);
    assert_eq!(offset_of!(ComponentLayout, size), 60);
    assert_eq!(offset_of!(ComponentLayout, align), 64);

    assert_eq!(size_of::<ComponentsTable>(), 16);
}

#[test]
fn verb_descriptors_are_pinned() {
    assert_eq!(size_of::<VerbName>(), 16);
    assert_eq!(offset_of!(VerbName, name_len), 8);

    assert_eq!(size_of::<VerbsTable>(), 24);
    assert_eq!(offset_of!(VerbsTable, axes), 8);
    assert_eq!(offset_of!(VerbsTable, action_len), 16);
    assert_eq!(offset_of!(VerbsTable, axis_len), 20);
}

#[test]
fn the_self_description_read_before_trust_is_pointer_free() {
    // `AbiInfo` is read out of an artifact that has not been proven compatible
    // yet, so it must be interpretable without following anything.
    assert_eq!(size_of::<AbiInfo>(), 4 + FINGERPRINT_BYTES);
    assert_eq!(align_of::<AbiInfo>(), 4);
    assert_eq!(offset_of!(AbiInfo, fingerprint), 4);
}

#[test]
fn the_host_table_is_a_version_then_eight_calls() {
    assert_eq!(offset_of!(HostApiV1, version), 0);
    assert_eq!(offset_of!(HostApiV1, spawn), 8);
    // Counted in function pointers rather than words: adding a call to the table
    // is a version bump, and this is the line that says so.
    assert_eq!(size_of::<HostApiV1>(), 8 + 8 * size_of::<extern "C" fn()>());
}

#[test]
fn every_exported_symbol_name_is_nul_terminated() {
    // These are handed to `dlsym`/`GetProcAddress` as-is; a missing NUL is a
    // read past the end of a `&'static [u8]`, not a failed lookup.
    for sym in [
        SYM_GAME_ABI,
        SYM_GAME_INIT,
        SYM_GAME_COMPONENTS,
        SYM_GAME_VERBS,
        SYM_GAME_SYSTEMS,
    ] {
        assert_eq!(sym.last(), Some(&0));
        assert!(!sym[..sym.len() - 1].contains(&0));
    }
}
