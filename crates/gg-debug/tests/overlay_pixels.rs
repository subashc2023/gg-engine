//! The overlay end to end, on a real device: font → draw layer → UI pass →
//! pixels.
//!
//! §1.5 keeps the windowed shell out of every automated tier, so without this
//! the whole UI path would be proven only by a human looking at it. Everything
//! here is offscreen, which is the one kind of rendering an automated tier is
//! allowed to do — and it goes through `OffscreenRenderer`, the same entry point
//! the golden harness uses, so what it proves is the pass the screen runs.
//!
//! Golden references would be the other way to gate this, and are deliberately
//! not: a reference image containing an overlay contains a *frame counter*, and
//! adding a scene means blessing it on every backend (§4.10).

// unwrap is permitted in tests (§2, Error handling row).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use gg_debug::draw::{DrawList, Rect};
use gg_debug::{Overlay, atlas, overlay};
use gg_extract::Extracted;
use gg_render::{OffscreenRenderer, View};
use gg_rhi::MemoryUse;

const EXTENT: (u32, u32) = (256, 128);
/// The clear the scene lands on, in the sRGB bytes it reads back as.
const CLEAR: [u8; 3] = [0, 0, 0];

struct Target {
    pixels: Vec<u8>,
}

impl Target {
    fn at(&self, x: u32, y: u32) -> [u8; 3] {
        let i = ((y * EXTENT.0 + x) * 4) as usize;
        [self.pixels[i], self.pixels[i + 1], self.pixels[i + 2]]
    }

    fn lit(&self) -> usize {
        self.pixels
            .chunks_exact(4)
            .filter(|p| p[..3] != CLEAR)
            .count()
    }
}

/// Render a UI layer over an empty scene and read the target back.
fn render(vertices: &[gg_render::ui::UiVertex]) -> Target {
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    renderer.set_ui_atlas(&atlas()).unwrap();
    let frame = renderer
        .frame(
            &Extracted::default(),
            &View::default(),
            [0.0, 0.0, 0.0, 1.0],
            vertices,
        )
        .unwrap();
    let target = Target {
        pixels: frame.pixels,
    };
    let report = renderer.shutdown();
    assert!(report.clean(), "unclean: {report:?}");
    target
}

/// The pass exists in the graph only when there is something to draw — an
/// empty layer must not cost a pipeline bind, and a dist build hands over an
/// empty one every frame.
#[test]
fn an_empty_layer_declares_no_pass_and_a_full_one_does() {
    let mut renderer = OffscreenRenderer::new(EXTENT).unwrap();
    renderer.set_ui_atlas(&atlas()).unwrap();
    let empty = renderer
        .frame(&Extracted::default(), &View::default(), [0.0; 4], &[])
        .unwrap();
    assert!(
        !empty.order.iter().any(|name| name == "ui"),
        "{:?}",
        empty.order
    );

    let mut list = DrawList::default();
    list.rect(Rect::new(0.0, 0.0, 8.0, 8.0), 0xffff_ffff);
    let drawn = renderer
        .frame(
            &Extracted::default(),
            &View::default(),
            [0.0; 4],
            list.vertices(),
        )
        .unwrap();
    assert!(
        drawn.order.iter().any(|name| name == "ui"),
        "{:?}",
        drawn.order
    );
    // Between post and the readback: the image judged is the one the screen
    // would get, UI included.
    let position = |name: &str| drawn.order.iter().position(|n| n == name).unwrap();
    assert!(position("post") < position("ui"));
    assert!(position("ui") < position("readback"));
    assert!(renderer.shutdown().clean());
}

/// A solid rectangle lands where it was asked for, in the colour it was asked
/// for, and nowhere else. This is the whole vertex path — pixel coordinates
/// through the push constants, the solid texel, and the `0xAARRGGBB` unpack.
#[test]
fn a_rectangle_lands_on_the_pixels_it_named() {
    let mut list = DrawList::default();
    // Opaque pure red, so the blend and the sRGB encode are both identities on
    // it and the readback is exactly what was authored.
    list.rect(Rect::new(16.0, 8.0, 32.0, 16.0), 0xffff_0000);
    let target = render(list.vertices());

    assert_eq!(target.at(0, 0), CLEAR, "the corner is untouched");
    assert_eq!(target.at(32, 16), [0xff, 0, 0], "inside");
    assert_eq!(
        target.at(16, 8),
        [0xff, 0, 0],
        "top-left corner is inclusive"
    );
    assert_eq!(target.at(48, 8), CLEAR, "and the right edge is exclusive");
    assert_eq!(target.at(15, 16), CLEAR, "nothing bleeds left");
    assert_eq!(target.lit(), 32 * 16, "exactly the area asked for");
}

/// Alpha blending, which is what separates a readable overlay from a black box
/// over the game: half-alpha white over black reads back as mid grey.
#[test]
fn a_translucent_rectangle_blends_with_what_is_under_it() {
    let mut list = DrawList::default();
    list.rect(Rect::new(0.0, 0.0, 16.0, 16.0), 0xffff_ffff);
    list.rect(Rect::new(0.0, 0.0, 16.0, 16.0), 0x8000_0000);
    let target = render(list.vertices());
    let [r, g, b] = target.at(8, 8);
    assert_eq!((r, g), (b, b), "grey");
    assert!(
        (100..=200).contains(&r),
        "black at ~50% over white is mid grey, got {r}"
    );
}

/// A glyph is ink *and* gaps. A quad whose every pixel came out lit would mean
/// the atlas sample was ignored and every character is a solid block — which
/// still looks like "text is drawing" from a distance.
#[test]
fn a_glyph_draws_its_shape_and_not_its_cell() {
    let mut list = DrawList::default();
    // `I` at scale 4: a stem with clear space either side of it, so an atlas
    // that failed to sample would be obvious rather than merely wrong.
    list.push_transform((8.0, 8.0), 4.0);
    list.text(0.0, 0.0, "I", 0xffff_ffff);
    list.pop_transform();
    let target = render(list.vertices());

    let cell = 5 * 4 * 7 * 4;
    let lit = target.lit();
    assert!(lit > 0, "the glyph drew nothing at all");
    assert!(
        lit < cell,
        "the glyph filled its whole cell: {lit} of {cell}"
    );
    // The stem is the middle column of `I`; its left neighbour is blank on the
    // rows between the serifs.
    assert_eq!(target.at(8 + 2 * 4 + 1, 8 + 3 * 4 + 1), [0xff; 3], "stem");
    assert_eq!(target.at(8 + 1, 8 + 3 * 4 + 1), CLEAR, "beside the stem");
}

/// The overlay as the shell drives it: whatever it builds must survive being
/// handed to the renderer, and it must actually put ink on the target.
#[test]
fn the_overlay_the_shell_builds_reaches_the_screen() {
    overlay::register().unwrap();
    let mut ui = Overlay::default();
    let vertices = ui
        .build(&overlay::Stats {
            extent: EXTENT,
            tick: 1234,
            passes: &[],
            memory: MemoryUse::default(),
        })
        .to_vec();
    assert!(!vertices.is_empty(), "the default overlay draws");
    let target = render(&vertices);
    assert!(target.lit() > 0, "and reaches the target");
    assert_eq!(
        target.at(EXTENT.0 - 1, EXTENT.1 - 1),
        CLEAR,
        "the panel is a panel, not the whole screen"
    );
}
