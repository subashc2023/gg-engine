//! The rented text stack end to end, on a real device: face → shaper →
//! rasterizer → packer → UI pass → pixels.
//!
//! Offscreen, through `OffscreenRenderer` — the entry point the golden harness
//! uses — because §1.5 keeps the windowed shell out of every automated tier and
//! this path would otherwise be proven only by a human looking at it.
//!
//! What it is really for is the *shared* atlas: [`gg_ui::font`]'s band and a
//! packed outline glyph are one bitmap and therefore one draw (§4.9), and the
//! way that breaks is silently — a solid rectangle that started sampling a
//! glyph's coverage still draws, just wrongly.

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_extract::Extracted;
use gg_render::{OffscreenRenderer, View};
use gg_ui::{DrawList, Fonts, Rect};

const EXTENT: (u32, u32) = (256, 128);
const RED: u32 = 0xffff_0000;
const GREEN: u32 = 0xff00_ff00;
/// Where the glyph run's top-left goes — clear of the rectangle below.
const PEN: (f32, f32) = (96.0, 32.0);
/// Big enough that a dropped row is visible as a count, not as rounding.
const PX: u16 = 32;

struct Target {
    pixels: Vec<u8>,
}

impl Target {
    fn count(&self, colour: [u8; 3]) -> usize {
        self.pixels
            .chunks_exact(4)
            .filter(|p| p[..3] == colour)
            .count()
    }

    fn at(&self, x: u32, y: u32) -> [u8; 3] {
        let i = ((y * EXTENT.0 + x) * 4) as usize;
        [self.pixels[i], self.pixels[i + 1], self.pixels[i + 2]]
    }
}

fn face_bytes() -> Vec<u8> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crates/gg-ui is two below the workspace root");
    std::fs::read(root.join("assets/fonts/FiraMono-Regular.ttf")).expect("the vendored face")
}

/// A solid rectangle and one shaped glyph, in two colours so each can be
/// counted without the other.
fn frame() -> (Target, Rect) {
    let mut fonts = Fonts::default();
    let face = fonts.load(face_bytes(), 0).unwrap();
    let mut list = DrawList::default();
    list.rect(Rect::new(0.0, 0.0, 32.0, 16.0), RED);
    // `l` in a monospace face: a stem with clear space either side, so an atlas
    // that was sampled at the wrong slot would be a block or nothing at all.
    fonts.layout(face, PX, "l");
    let glyph = fonts.glyphs()[0].rect;
    list.glyphs(PEN.0, PEN.1, fonts.glyphs(), GREEN);

    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    // The upload the version counter exists to schedule: after layout, because
    // that is when the packer had something new to say.
    renderer.set_ui_atlas(&fonts.coverage()).unwrap();
    let captured = renderer
        .frame(
            &Extracted::default(),
            &View::default(),
            [0.0, 0.0, 0.0, 1.0],
            list.vertices(),
        )
        .unwrap();
    let target = Target {
        pixels: captured.pixels,
    };
    let report = renderer.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
    (
        target,
        Rect::new(PEN.0 + glyph.x, PEN.1 + glyph.y, glyph.w, glyph.h),
    )
}

/// A packed outline glyph draws its shape, and the fallback band's solid texel
/// still draws solid — from the same bitmap, in the same draw.
#[test]
fn a_rasterized_glyph_and_a_solid_rectangle_share_one_atlas() {
    let (target, glyph) = frame();

    assert_eq!(
        target.count([0xff, 0, 0]),
        32 * 16,
        "the solid texel is solid"
    );
    assert_eq!(target.at(16, 8), [0xff, 0, 0]);

    let ink = target.count([0, 0xff, 0]);
    let cell = (glyph.w * glyph.h) as usize;
    assert!(ink > 0, "the glyph drew nothing");
    assert!(
        cell > 0 && ink < cell,
        "the glyph filled its box: {ink} of {cell}"
    );
    // And it drew where it was placed, not at the origin: every green pixel is
    // inside the rectangle `layout` reported.
    for y in 0..EXTENT.1 {
        for x in 0..EXTENT.0 {
            if target.at(x, y) == [0, 0xff, 0] {
                assert!(
                    glyph.contains(x as f32, y as f32),
                    "green at {x},{y} is outside {glyph:?}"
                );
            }
        }
    }
}

/// Replacing the atlas between two frames is clean — the shell does exactly
/// this when the editor's resident glyph size changes under a resize (§6
/// M15.1), and the retired image's uploads must not outlive it.
///
/// On a device with a dedicated transfer family the first upload leaves an
/// ownership acquire the *next* frame records, so a replace-then-render is a
/// barrier over a destroyed image unless the retire drops it
/// (`gg-rhi/tests/uploads.rs` names the same bug at its own level). Lavapipe
/// has one family and cannot see it; this is written to run on either.
#[test]
fn an_atlas_replaced_between_frames_leaves_nothing_behind() {
    let mut fonts = Fonts::default();
    let face = fonts.load(face_bytes(), 0).unwrap();
    let mut list = DrawList::default();
    list.rect(Rect::new(0.0, 0.0, 8.0, 8.0), RED);

    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    let frame = |renderer: &mut OffscreenRenderer, list: &DrawList| {
        renderer
            .frame(
                &Extracted::default(),
                &View::default(),
                [0.0, 0.0, 0.0, 1.0],
                list.vertices(),
            )
            .map(|_| ())
    };
    renderer.set_ui_atlas(&fonts.coverage()).unwrap();
    frame(&mut renderer, &list).unwrap();
    // A second atlas, uploaded and handed over with the first still resident.
    fonts.layout(face, PX, "replaced");
    renderer.set_ui_atlas(&fonts.coverage()).unwrap();
    frame(&mut renderer, &list).unwrap();
    // And one replaced *before* any frame reads it, which is the tighter case.
    fonts.layout(face, PX * 2, "again");
    renderer.set_ui_atlas(&fonts.coverage()).unwrap();
    fonts.layout(face, PX * 3, "once more");
    renderer.set_ui_atlas(&fonts.coverage()).unwrap();
    frame(&mut renderer, &list).unwrap();

    let report = renderer.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
}

/// The upload schedule: the version moves while glyphs are arriving and stops
/// once they are resident, which is what makes a steady-state frame cost no
/// atlas traffic at all (§6 M13's zero-allocation row is the CPU half of the
/// same claim).
#[test]
fn the_atlas_version_moves_only_while_glyphs_are_new() {
    let mut fonts = Fonts::default();
    let face = fonts.load(face_bytes(), 0).unwrap();
    let cold = fonts.version();
    fonts.layout(face, PX, "steady");
    let warm = fonts.version();
    assert!(warm > cold, "six glyphs arrived");
    for _ in 0..16 {
        fonts.layout(face, PX, "steady");
    }
    assert_eq!(fonts.version(), warm, "nothing to re-upload");
}
