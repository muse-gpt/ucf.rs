//! The drawable page plan: one resolved, ordered sequence of drawing
//! operations per page. This crate never paints pixels — `media.render`'s
//! `layout-export` entry lowers a [`PagePlan`] through `media.raster` and
//! `media.text` onto a `media.image` frame, in exactly this order.

use geometry::Rgba;
use media_image::{Bitmap, Fit};
use media_text::FontId;
use media_types::Error;

use crate::metrics::{BASELINE_FACTOR, LINE_HEIGHT_FACTOR, PAD_FACTOR, TextMeasure, text_box};
use crate::model::{ComicDoc, Panel, TextItem, TextKind};
use crate::paths;

/// Stroke width of every caption box and balloon outline (the source fixed).
pub const BOX_STROKE_WIDTH: f32 = 2.0;
/// Balloon corner radius is `box height / 3`, capped at [`BALLOON_MAX_RADIUS`].
pub const BALLOON_RADIUS_DIVISOR: f32 = 3.0;
/// Cap for the balloon corner radius.
pub const BALLOON_MAX_RADIUS: f32 = 12.0;

/// Explicit image resources: a panel image key resolves to decoded pixels.
///
/// The plan builder only checks resolvability — a missing or undecodable
/// image fails the whole page, the behaviour the source test
/// `a_missing_image_is_an_error` locks. The render lowering reads the same
/// store again for the pixels, so one resolution path serves both halves.
pub trait ImageStore {
    /// Resolves a panel image key, failing with an explicit error when the
    /// resource is absent or cannot be decoded.
    fn resolve(&mut self, key: &str) -> Result<&Bitmap, Error>;
}

/// One resolved drawing step. The variants map one-to-one onto the
/// `media.raster` `Paint` calls and `media.text` line draws the source made,
/// with the fixed call parameters (stroke progress 1.0, non-additive, image
/// alpha 1.0 and radius 0.0) already applied.
#[derive(Clone, Debug, PartialEq)]
pub enum DrawOp {
    FillRect { rect: [f32; 4], color: Rgba },
    FillRoundRect { rect: [f32; 4], radius: f32, color: Rgba },
    StrokePolyline { points: Vec<[f32; 2]>, width: f32, color: Rgba },
    FillPolygon { points: Vec<[f32; 2]>, color: Rgba },
    Image { key: String, rect: [f32; 4], fit: Fit },
    TextLine { text: String, x: f32, baseline: f32, size: f32, font: FontId, color: Rgba },
}

/// One page as a frame header plus its ordered operations.
#[derive(Clone, Debug, PartialEq)]
pub struct PagePlan {
    pub width: u32,
    pub height: u32,
    pub background: Rgba,
    pub ops: Vec<DrawOp>,
}

/// Builds the plan for one page. An unknown page index and an unresolvable
/// panel image are explicit errors; neither paints a partial page.
pub fn page_plan(
    comic: &ComicDoc,
    page: usize,
    text: &mut dyn TextMeasure,
    images: &mut dyn ImageStore,
) -> Result<PagePlan, Error> {
    let page = comic.pages.get(page).ok_or_else(|| Error::invalid_input(format!("no page {page}")))?;
    let mut ops = Vec::new();
    for panel in &page.panels {
        paint_panel(&mut ops, text, images, panel)?;
    }
    Ok(PagePlan { width: comic.width, height: comic.height, background: comic.background, ops })
}

/// The source paint order: fill rect, artwork, border, then texts.
fn paint_panel(ops: &mut Vec<DrawOp>, text: &mut dyn TextMeasure, images: &mut dyn ImageStore, panel: &Panel) -> Result<(), Error> {
    ops.push(DrawOp::FillRect { rect: panel.rect, color: panel.fill });
    if let Some(key) = &panel.image {
        images.resolve(key)?;
        ops.push(DrawOp::Image { key: key.clone(), rect: panel.rect, fit: panel.fit });
    }
    if let Some(border) = &panel.border {
        ops.push(DrawOp::StrokePolyline { points: paths::rect_path(panel.rect), width: border.width, color: border.color });
    }
    for item in &panel.texts {
        paint_text(ops, text, item);
    }
    Ok(())
}

/// The source text order: box fill, box outline, tail, then the lines. The
/// tail is drawn after the box outline so its fill opens the outline where it
/// joins — the call order the pixels confirm (the source's stale doc comment
/// claiming the opposite is deliberately not carried over).
fn paint_text(ops: &mut Vec<DrawOp>, text: &mut dyn TextMeasure, item: &TextItem) {
    let box_ = text_box(text, item);
    let [x0, y0, box_w, box_h] = box_.rect;
    match item.kind {
        TextKind::Caption => {
            ops.push(DrawOp::FillRect { rect: box_.rect, color: item.fill });
            ops.push(DrawOp::StrokePolyline { points: paths::rect_path(box_.rect), width: BOX_STROKE_WIDTH, color: item.stroke });
        }
        TextKind::Balloon => {
            let radius = (box_h / BALLOON_RADIUS_DIVISOR).min(BALLOON_MAX_RADIUS);
            ops.push(DrawOp::FillRoundRect { rect: box_.rect, radius, color: item.fill });
            ops.push(DrawOp::StrokePolyline {
                points: paths::round_rect_path(box_.rect, radius),
                width: BOX_STROKE_WIDTH,
                color: item.stroke,
            });
            if let Some(tail) = item.tail {
                let [b0, b1] = paths::tail_base(box_.rect, tail);
                ops.push(DrawOp::FillPolygon { points: vec![b0, tail, b1], color: item.fill });
                ops.push(DrawOp::StrokePolyline { points: vec![b0, tail, b1], width: BOX_STROKE_WIDTH, color: item.stroke });
            }
        }
    }
    let font = if item.bold { FontId::Bold } else { FontId::Regular };
    let pad = item.size * PAD_FACTOR;
    let line_height = item.size * LINE_HEIGHT_FACTOR;
    let mut top = y0 + pad;
    for (line, width) in box_.lines.iter().zip(&box_.widths) {
        let x = match item.kind {
            TextKind::Caption => x0 + pad,
            TextKind::Balloon => x0 + (box_w - width) / 2.0,
        };
        ops.push(DrawOp::TextLine { text: line.clone(), x, baseline: top + item.size * BASELINE_FACTOR, size: item.size, font, color: item.color });
        top += line_height;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Border;
    use crate::spec::{DEFAULT_TEXT_COLOR, DEFAULT_TEXT_FILL, DEFAULT_TEXT_STROKE};
    use crate::testutil::{FixedText, KeyStore, demo_spec};
    use crate::Page;

    fn page(panels: Vec<Panel>) -> ComicDoc {
        let mut comic = ComicDoc::new(400, 300, Rgba([250, 250, 252, 255]));
        comic.pages.push(Page { panels });
        comic
    }

    #[test]
    fn ops_follow_the_source_paint_order() {
        let comic = page(vec![Panel {
            rect: [10.0, 10.0, 100.0, 80.0],
            fill: Rgba([255, 255, 255, 255]),
            image: Some("panel.png".to_string()),
            fit: Fit::Cover,
            border: Some(Border { color: Rgba([20, 20, 24, 255]), width: 3.0 }),
            texts: Vec::new(),
        }]);
        let mut measurer = FixedText::new(&[], 10.0);
        let mut store = KeyStore::new(&["panel.png"]);
        let plan = page_plan(&comic, 0, &mut measurer, &mut store).expect("plan");
        assert_eq!(
            plan.ops,
            vec![
                DrawOp::FillRect { rect: [10.0, 10.0, 100.0, 80.0], color: Rgba([255, 255, 255, 255]) },
                DrawOp::Image { key: "panel.png".to_string(), rect: [10.0, 10.0, 100.0, 80.0], fit: Fit::Cover },
                DrawOp::StrokePolyline { points: paths::rect_path([10.0, 10.0, 100.0, 80.0]), width: 3.0, color: Rgba([20, 20, 24, 255]) },
            ]
        );
    }

    #[test]
    fn a_missing_image_fails_the_whole_page() {
        // The source contract, locked by `a_missing_image_is_an_error`.
        let comic = page(vec![Panel {
            rect: [0.0, 0.0, 100.0, 100.0],
            fill: Rgba([255, 255, 255, 255]),
            image: Some("E:/no-such-panel.png".to_string()),
            fit: Fit::Contain,
            border: None,
            texts: Vec::new(),
        }]);
        let mut measurer = FixedText::new(&[], 10.0);
        let mut store = KeyStore::new(&[]);
        let error = page_plan(&comic, 0, &mut measurer, &mut store).expect_err("broken panel path must not plan silently");
        assert_eq!(error.code(), "io");
        assert!(error.message.contains("image E:/no-such-panel.png"), "{}", error.message);
    }

    #[test]
    fn an_unknown_page_fails_loudly() {
        let comic = ComicDoc::new(400, 300, Rgba([250, 250, 252, 255]));
        let mut measurer = FixedText::new(&[], 10.0);
        let mut store = KeyStore::new(&[]);
        let error = page_plan(&comic, 0, &mut measurer, &mut store).expect_err("empty document has no page 0");
        assert_eq!(error.code(), "invalid_input");
        assert_eq!(error.message, "no page 0");
    }

    #[test]
    fn caption_plan_locks_the_box_and_line_geometry() {
        // size 20, advance 10/char, "AB CD" → box [30,30,74,49].
        let item = TextItem::caption("AB CD").size(20.0).at([30.0, 30.0]);
        let comic = page(vec![Panel::new([0.0, 0.0, 400.0, 300.0]).text(item)]);
        let mut measurer = FixedText::new(&["AB CD"], 10.0);
        let mut store = KeyStore::new(&[]);
        let plan = page_plan(&comic, 0, &mut measurer, &mut store).expect("plan");
        assert_eq!(
            &plan.ops[1..],
            &[
                DrawOp::FillRect { rect: [30.0, 30.0, 74.0, 49.0], color: DEFAULT_TEXT_FILL },
                DrawOp::StrokePolyline { points: paths::rect_path([30.0, 30.0, 74.0, 49.0]), width: 2.0, color: DEFAULT_TEXT_STROKE },
                DrawOp::TextLine {
                    text: "AB CD".to_string(),
                    x: 42.0,
                    baseline: 58.4, // 30 + pad 12 + 20 × 0.82
                    size: 20.0,
                    font: FontId::Regular,
                    color: DEFAULT_TEXT_COLOR,
                },
            ]
        );
    }

    #[test]
    fn balloon_plan_centers_lines_and_draws_the_tail_last() {
        // Box [163,125.5,74,49]; radius min(49/3, 12) = 12; tail exits the
        // bottom edge at (200, 174.5) with a 7.0 half base.
        let item = TextItem::balloon("AB CD", [200.0, 240.0]).size(20.0).at([200.0, 150.0]);
        let comic = page(vec![Panel::new([0.0, 0.0, 400.0, 300.0]).text(item)]);
        let mut measurer = FixedText::new(&["AB CD"], 10.0);
        let mut store = KeyStore::new(&[]);
        let plan = page_plan(&comic, 0, &mut measurer, &mut store).expect("plan");
        let rect = [163.0, 125.5, 74.0, 49.0];
        let [b0, b1] = paths::tail_base(rect, [200.0, 240.0]);
        assert_eq!(b0, [193.0, 174.5]);
        assert_eq!(b1, [207.0, 174.5]);
        assert_eq!(
            &plan.ops[1..],
            &[
                DrawOp::FillRoundRect { rect, radius: 12.0, color: DEFAULT_TEXT_FILL },
                DrawOp::StrokePolyline { points: paths::round_rect_path(rect, 12.0), width: 2.0, color: DEFAULT_TEXT_STROKE },
                DrawOp::FillPolygon { points: vec![b0, [200.0, 240.0], b1], color: DEFAULT_TEXT_FILL },
                DrawOp::StrokePolyline { points: vec![b0, [200.0, 240.0], b1], width: 2.0, color: DEFAULT_TEXT_STROKE },
                DrawOp::TextLine {
                    text: "AB CD".to_string(),
                    x: 175.0, // centred: 163 + (74 - 50) / 2
                    baseline: 153.9,
                    size: 20.0,
                    font: FontId::Regular,
                    color: DEFAULT_TEXT_COLOR,
                },
            ]
        );
    }

    #[test]
    fn multi_line_staggers_by_line_height() {
        // lines ["AAA","BB"] at size 10: pad 6, baselines 14.2 and 26.7.
        let item = TextItem::caption("AAA BB").size(10.0).at([0.0, 0.0]);
        let comic = page(vec![Panel::new([0.0, 0.0, 400.0, 300.0]).text(item)]);
        let mut measurer = FixedText::new(&["AAA", "BB"], 10.0);
        let mut store = KeyStore::new(&[]);
        let plan = page_plan(&comic, 0, &mut measurer, &mut store).expect("plan");
        let baselines: Vec<f32> =
            plan.ops.iter().filter_map(|op| if let DrawOp::TextLine { baseline, .. } = op { Some(*baseline) } else { None }).collect();
        assert_eq!(baselines, vec![14.2, 26.7]);
    }

    #[test]
    fn bold_text_switches_the_font_role() {
        let item = TextItem::caption("HI").size(10.0).bold();
        let comic = page(vec![Panel::new([0.0, 0.0, 400.0, 300.0]).text(item)]);
        let mut measurer = FixedText::new(&["HI"], 10.0);
        let mut store = KeyStore::new(&[]);
        let plan = page_plan(&comic, 0, &mut measurer, &mut store).expect("plan");
        let DrawOp::TextLine { font, .. } = plan.ops.last().expect("line") else { panic!("last op must be the text line") };
        assert_eq!(*font, FontId::Bold);
    }

    #[test]
    fn a_resolved_spec_plans_without_surprises() {
        // End-to-end through the input contract: the demo's wide card.
        let comic = demo_spec().resolve().expect("demo spec resolves");
        let mut measurer = FixedText::new(&["EST. NIGHT EXTERIOR"], 11.0);
        let mut store = KeyStore::new(&[]);
        let plan = page_plan(&comic, 0, &mut measurer, &mut store).expect("plan");
        assert_eq!((plan.width, plan.height), (1200, 800));
        assert_eq!(plan.background, Rgba([246, 244, 240, 255]));
    }
}
