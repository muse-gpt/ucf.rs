//! Panel documents: pages of rectangular panels carrying artwork references,
//! self-sizing caption boxes and speech balloons, plus the numbered
//! contact-sheet plan.
//!
//! The semantics were absorbed from the OpenComic core (Apache-2.0,
//! © OpenComic Authors) with the values preserved: the text-box auto-sizing
//! (pad 0.6 × size, line height 1.25 × size, widest-line measure, baseline
//! 0.82 × size), the balloon corner-cut octagon and tail-ray geometry, the
//! even grid math and the 16 px contact-sheet gutter with `cell_h / 8` bold
//! page labels.
//!
//! Ownership boundaries:
//! - this crate owns the document model ([`model`]), the host input contract
//!   ([`spec`]), the grid math ([`grid`]), the text-box metrics ([`metrics`]),
//!   the annotation paths ([`paths`]) and the plans ([`plan`], [`sheet`]);
//!   it never paints pixels and never writes files;
//! - lowering a plan through `media.raster` and encoding PNGs live in
//!   `media.render`'s `layout-export` entry;
//! - wrapping and measuring go through `media.text` behind
//!   [`metrics::TextMeasure`] — there is no second shaping engine here and no
//!   system font discovery: hosts inject explicit font bytes;
//! - panel artwork is an explicit resource too: [`plan::ImageStore`] resolves
//!   a key to decoded pixels, and a missing or undecodable image fails the
//!   whole page — the source behaviour locked by its `a_missing_image_is_an_error`
//!   test;
//! - the input contract ([`spec`]) carries the absorbed binding defaults, and
//!   a malformed colour, an unknown `fit` spelling or an out-of-range size is
//!   an explicit `invalid_input` error instead of the binding's silent
//!   fallback.
//!
//! ```rust
//! use media_layout::{ComicSpec, Rgba, grid};
//!
//! let rects = grid(2, 1, 400.0, 300.0, 20.0, 16.0);
//! assert_eq!(rects.len(), 2);
//! let spec: ComicSpec = serde_json::from_str(
//!     r##"{"width":400,"height":300,"pages":[{"panels":[{"rect":[0.0,0.0,200.0,300.0],"fill":"#b42828"}]}]}"##,
//! )
//! .expect("spec parses");
//! let doc = spec.resolve().expect("valid document");
//! assert_eq!(doc.pages[0].panels[0].fill, Rgba([180, 40, 40, 255]));
//! ```

pub mod grid;
pub mod metrics;
pub mod model;
pub mod paths;
pub mod plan;
pub mod sheet;
pub mod spec;

// The value types that appear in the public surface, re-exported so hosts and
// the render lowering can name them without importing the providers.
pub use geometry::Rgba;
pub use grid::grid;
pub use media_image::Fit;
pub use media_text::FontId;
pub use metrics::{TextMeasure, TextBox, anchor_origin, text_box};
pub use model::{Border, ComicDoc, Page, Panel, TextItem, TextKind};
pub use paths::{rect_path, round_rect_path, tail_base};
pub use plan::{DrawOp, ImageStore, PagePlan, page_plan};
pub use sheet::{SheetCell, SheetPlan, sheet_plan};
pub use spec::{
    ComicSpec, BorderSpec, MAX_SHEET_SCALE, MIN_SHEET_SCALE, MIN_TEXT_SIZE, PageSpec, PanelSpec, TextSpec, DEFAULT_BACKGROUND,
    DEFAULT_BORDER_COLOR, DEFAULT_BORDER_WIDTH, DEFAULT_PANEL_FILL, DEFAULT_TEXT_AT, DEFAULT_TEXT_COLOR, DEFAULT_TEXT_FILL,
    DEFAULT_TEXT_MAX_WIDTH, DEFAULT_TEXT_SIZE, DEFAULT_TEXT_STROKE,
};

#[cfg(test)]
pub(crate) mod testutil {
    use media_types::Error;

    use crate::metrics::TextMeasure;
    use crate::plan::ImageStore;

    /// A scripted measurer: fixed lines, fixed advance per character. Enough
    /// to pin the box formulas without binding the suite to any installed
    /// font — the real-font path is pinned by the `layout-export` digests.
    pub(crate) struct FixedText {
        pub lines: Vec<String>,
        pub advance: f32,
    }

    impl FixedText {
        pub fn new(lines: &[&str], advance: f32) -> Self {
            Self { lines: lines.iter().map(|line| line.to_string()).collect(), advance }
        }
    }

    impl TextMeasure for FixedText {
        fn wrap_text(&mut self, _text: &str, _size: f32, _bold: bool, _max_width: f32) -> Vec<String> {
            self.lines.clone()
        }

        fn line_width(&mut self, text: &str, _size: f32, _bold: bool) -> f32 {
            text.chars().count() as f32 * self.advance
        }
    }

    /// A store that only knows the keys it was given; anything else fails
    /// with the same `io`-coded error the codec-backed store uses.
    pub(crate) struct KeyStore {
        store: std::collections::HashMap<String, media_image::Bitmap>,
    }

    impl KeyStore {
        pub fn new(keys: &[&str]) -> Self {
            Self {
                store: keys.iter().map(|key| (key.to_string(), media_image::Bitmap::new(1, 1, vec![0, 0, 0, 0]))).collect(),
            }
        }
    }

    impl ImageStore for KeyStore {
        fn resolve(&mut self, key: &str) -> Result<&media_image::Bitmap, Error> {
            self.store.get(key).ok_or_else(|| Error::io(format!("image {key}: not supplied")))
        }
    }

    /// The demo page 1 shape (one bold caption on a wide card) as a wire
    /// fixture for the end-to-end resolve-and-plan tests.
    pub(crate) fn demo_spec() -> crate::spec::ComicSpec {
        serde_json::from_str(
            r##"{"width":1200,"height":800,"background":"#f6f4f0","pages":[{"panels":[
                {"rect":[40.0,24.0,360.0,752.0],"texts":[{"kind":"caption","text":"EST. NIGHT EXTERIOR","at":[52.0,36.0],"size":22.0,"width":336.0,"bold":true,"fill":"#e8b86d","stroke":"#141418"}]}
            ]}]}"##,
        )
        .expect("demo fixture parses")
    }
}
