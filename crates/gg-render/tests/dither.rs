//! `r.dither` (§6 M22) — the half-LSB the post pass adds before the 8-bit
//! quantizer, proven offscreen where a test can read the pixels.
//!
//! Two claims that fail in opposite directions, because either alone is
//! satisfied by something wrong:
//!
//! - A smooth gradient must stop holding one code value for tens of pixels.
//!   That is the band, and a dither that did nothing would leave it.
//! - A flat region must move by **at most one code value**, and a value that
//!   already landed on a code must not move at all. That is not a nicety, it is
//!   the reason the noise is uniform and half an LSB wide rather than the
//!   triangular LSB the signal-processing literature would prescribe: the round
//!   is still the nearest one, so an exactly-representable value survives and
//!   nothing else can travel further than the neighbour it was already between.
//!   Half the suite's "this corner is the clear colour" claims are black corners
//!   and they hold unchanged; a dither that had to be excused by them would be
//!   the wrong dither.
//!
//! `gg-tools banding` is where the amplitude was chosen; this is only the pair of
//! claims a gate can hold.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_ecs::World;
use gg_ecs::boundary::{Light, Renderable};
use gg_extract::Extracted;
use gg_math::sim;
use gg_render::{OffscreenRenderer, View, cvars};

/// Wide, because a band is measured in pixels and the same gradient across half
/// the width is a band half as wide — a narrow frame would report the fix as
/// having less to do than it does.
const EXTENT: (u32, u32) = (512, 288);

/// Pitch, radians. Shallow enough that the top of the frame is well above the
/// horizon — the flat claim below needs rows the floor cannot reach, and at this
/// vertical field the horizon sits about a fifth of the way down.
const PITCH: f32 = -0.35;

/// Rows the floor fills at this framing, and the ones the gradient is measured
/// over. Comfortably below the horizon at [`PITCH`].
const FLOOR: std::ops::Range<u32> = 120..288;

/// Rows the geometry reaches none of, where the clear colour is what every pixel
/// holds. Well above the horizon — a "sky" row that grazed the far floor would
/// make this a claim about very dark floor instead.
const FLAT: [(u32, u32); 4] = [(0, 0), (3, 8), (EXTENT.0 - 1, 0), (EXTENT.0 - 4, 12)];

/// Linear, and not black: a clear of zero would make the flat claim pass for the
/// wrong reason, since zero is the one value no rounding can move off.
const CLEAR: [f32; 4] = [0.05, 0.06, 0.09, 1.0];

/// Nothing but floor under one point light — a single inverse-square falloff, so
/// every pixel of `FLOOR` is on a smooth ramp and a run of one code value there
/// is a contour and nothing else.
fn world() -> World {
    let mut world = World::new();
    world.register::<Renderable>().unwrap();
    world.register::<Light>().unwrap();
    let floor = world.spawn();
    world
        .insert(
            floor,
            Renderable::boxed(
                sim::DVec3::new(0.0, -0.1, 0.0),
                sim::Vec3::new(60.0, 0.1, 60.0),
                0x009a_9488,
            ),
        )
        .unwrap();
    let lamp = world.spawn();
    world
        .insert(
            lamp,
            Light::point(sim::DVec3::new(2.0, 2.6, -5.0), 0x00ff_c890, 14.0, 11.0),
        )
        .unwrap();
    world
}

fn render_with(
    renderer: &mut OffscreenRenderer,
    world: &World,
    dither: f64,
    clear: [f32; 4],
) -> Vec<u8> {
    cvars::DITHER.set_float(dither);
    let view = View {
        pitch: PITCH,
        ..View::default()
    };
    let mut extracted = Extracted::default();
    extracted.clear(sim::DVec3::new(0.0, 1.7, 0.0), view.frustum(EXTENT));
    extracted.append_lights(world).unwrap();
    extracted.append::<Renderable>(world).unwrap();
    renderer
        .frame(&extracted, &view, clear, &[])
        .unwrap()
        .pixels
}

fn render(renderer: &mut OffscreenRenderer, world: &World, dither: f64) -> Vec<u8> {
    render_with(renderer, world, dither, CLEAR)
}

/// Mean run of identical green values along the floor's rows, in pixels.
///
/// Green because luminance is a weighted sum whose quantization partly cancels,
/// which would report a picture less banded than the one on the screen. The
/// *mean* run and not the longest: correct dither still leaves long runs wherever
/// the signal sits on a code exactly, and those are not contours.
fn mean_run(pixels: &[u8]) -> f64 {
    let w = EXTENT.0 as usize;
    let (mut total, mut runs) = (0usize, 0usize);
    for y in FLOOR {
        let row: Vec<u8> = (0..w)
            .map(|x| pixels[((y as usize * w) + x) * 4 + 1])
            .collect();
        total += w;
        runs += (1..w).filter(|&x| row[x] != row[x - 1]).count() + 1;
    }
    total as f64 / runs as f64
}

fn pixel(pixels: &[u8], x: u32, y: u32) -> [u8; 4] {
    let at = ((y * EXTENT.0 + x) * 4) as usize;
    pixels[at..at + 4].try_into().unwrap()
}

#[test]
fn dither_breaks_the_bands_in_a_gradient_and_leaves_flat_colour_alone() {
    let world = world();
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    let off = render(&mut renderer, &world, 0.0);
    let on = render(&mut renderer, &world, 1.0);

    let (banded, dithered) = (mean_run(&off), mean_run(&on));
    println!("mean run: {banded:.2} px undithered, {dithered:.2} px dithered");
    assert!(
        banded > 6.0,
        "the undithered gradient's mean run is {banded:.2} px — this framing is not banding, so \
         the claim below would pass for free"
    );
    // An absolute bound rather than a ratio against the row above: what "the
    // bands are gone" means is that a code value holds for a pixel or three, and
    // that number does not depend on how steep this particular framing's falloff
    // came out.
    assert!(
        dithered < 4.0,
        "dither moved the mean run from {banded:.2} px to {dithered:.2} px — the bands are still \
         there"
    );

    for (x, y) in FLAT {
        let (a, b) = (pixel(&off, x, y), pixel(&on, x, y));
        for channel in 0..3 {
            assert!(
                a[channel].abs_diff(b[channel]) <= 1,
                "dither moved flat clear colour at ({x}, {y}) from {a:?} to {b:?}; half an LSB \
                 reaches the neighbouring code and no further"
            );
        }
    }

    // And the sharper half of the same claim, which is what the suite's black
    // corners rest on: a value already *on* a code cannot move, because the
    // round is still the nearest one.
    let black = render_with(&mut renderer, &world, 1.0, [0.0, 0.0, 0.0, 1.0]);
    for (x, y) in FLAT {
        assert_eq!(
            pixel(&black, x, y),
            [0, 0, 0, 255],
            "a black clear moved at ({x}, {y}) — half an LSB reached an exact code"
        );
    }
}

/// Off by default it is not — and that is the claim every blessed golden (§4.10)
/// now rests on from the other side, so it is worth stating where a reader of
/// `View` will find it. Free of a device: this is arithmetic on a CVar.
#[test]
fn the_shipping_amplitude_is_half_a_code_value() {
    assert_eq!(
        cvars::DITHER.float(),
        1.0,
        "`r.dither` is the peak-to-peak width in code values; 1.0 is the ±0.5 the goldens hold"
    );
}
