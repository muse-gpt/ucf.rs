//! The text-box auto-sizing contract: the author places an anchor, the box
//! sizes itself to the wrapped text.
//!
//! Every factor here is absorbed value-for-value from the OpenComic core's
//! `paint_text`: pad = 0.6 × size, line height = 1.25 × size, box width =
//! widest measured line + 2 × pad, box height = lines × 1.25 × size + 2 × pad,
//! baseline = top + 0.82 × size. Wrapping and measuring themselves go through
//! [`TextMeasure`], implemented by `media.text`'s [`media_text::Fonts`] — this
//! crate holds the comic box contract, never a second shaping engine.

use media_text::Fonts;

use crate::model::{TextItem, TextKind};

/// Box padding around the text, as a factor of the font size.
pub const PAD_FACTOR: f32 = 0.6;
/// Line advance, as a factor of the font size.
pub const LINE_HEIGHT_FACTOR: f32 = 1.25;
/// Baseline offset below the text top, as a factor of the font size.
pub const BASELINE_FACTOR: f32 = 0.82;

/// The measurement seam every plan builder reads text through.
///
/// `media.text`'s [`Fonts`] is the production implementation; hosts may
/// inject a deterministic stand-in for tests. The seam never licenses a
/// second engine — the wrapping rules live in `media.text`.
pub trait TextMeasure {
    /// Breaks `text` into lines no wider than `max_width`.
    fn wrap_text(&mut self, text: &str, size: f32, bold: bool, max_width: f32) -> Vec<String>;
    /// The pixel advance width of one already-wrapped line.
    fn line_width(&mut self, text: &str, size: f32, bold: bool) -> f32;
}

impl TextMeasure for Fonts {
    fn wrap_text(&mut self, text: &str, size: f32, bold: bool, max_width: f32) -> Vec<String> {
        self.wrap(text, size, bold, false, Some(max_width))
    }

    fn line_width(&mut self, text: &str, size: f32, bold: bool) -> f32 {
        self.measure(text, size, bold, false)
    }
}

/// A measured, anchored text box: the resolved rect plus the wrapped lines
/// and their widths the painter centers on.
#[derive(Clone, Debug, PartialEq)]
pub struct TextBox {
    pub rect: [f32; 4],
    pub lines: Vec<String>,
    pub widths: Vec<f32>,
}

/// The anchoring rule in one place: a caption's anchor is its box's top-left
/// corner, a balloon's anchor is its box centre.
#[must_use]
pub fn anchor_origin(kind: TextKind, at: [f32; 2], box_w: f32, box_h: f32) -> [f32; 2] {
    match kind {
        TextKind::Caption => at,
        TextKind::Balloon => [at[0] - box_w / 2.0, at[1] - box_h / 2.0],
    }
}

/// Measures one text item into its box, applying the absorbed formulas.
pub fn text_box(text: &mut dyn TextMeasure, item: &TextItem) -> TextBox {
    let pad = item.size * PAD_FACTOR;
    let line_height = item.size * LINE_HEIGHT_FACTOR;
    let lines = text.wrap_text(&item.text, item.size, item.bold, item.max_width);
    let widths: Vec<f32> = lines.iter().map(|line| text.line_width(line, item.size, item.bold)).collect();
    let inner = widths.iter().copied().fold(0.0f32, f32::max);
    let box_w = inner + pad * 2.0;
    let box_h = lines.len() as f32 * line_height + pad * 2.0;
    let [x0, y0] = anchor_origin(item.kind, item.at, box_w, box_h);
    TextBox { rect: [x0, y0, box_w, box_h], lines, widths }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::FixedText;

    #[test]
    fn caption_box_sizes_to_the_wrapped_text() {
        // pad = 0.6 × 20 = 12, line height = 1.25 × 20 = 25,
        // inner = 5 chars × 10 advance = 50 → box 74 × 49, anchored top-left.
        let mut measurer = FixedText::new(&["AB CD"], 10.0);
        let item = TextItem::caption("AB CD").size(20.0).at([30.0, 30.0]);
        let box_ = text_box(&mut measurer, &item);
        assert_eq!(box_.rect, [30.0, 30.0, 74.0, 49.0]);
        assert_eq!(box_.lines, vec!["AB CD".to_string()]);
        assert_eq!(box_.widths, vec![50.0]);
    }

    #[test]
    fn balloon_anchor_is_the_box_centre() {
        let mut measurer = FixedText::new(&["AB CD"], 10.0);
        let item = TextItem::balloon("AB CD", [200.0, 240.0]).size(20.0).at([200.0, 150.0]);
        let box_ = text_box(&mut measurer, &item);
        assert_eq!(box_.rect, [163.0, 125.5, 74.0, 49.0]);
    }

    #[test]
    fn widest_line_sets_the_box_width() {
        // lines ["AAA","BB"] at size 10: pad 6, inner 30, box 42 × 37.
        let mut measurer = FixedText::new(&["AAA", "BB"], 10.0);
        let item = TextItem::caption("AAA BB").size(10.0).at([0.0, 0.0]);
        let box_ = text_box(&mut measurer, &item);
        assert_eq!(box_.rect, [0.0, 0.0, 42.0, 37.0]);
    }

    #[test]
    fn the_three_anchor_rules() {
        assert_eq!(anchor_origin(TextKind::Caption, [7.0, 9.0], 40.0, 20.0), [7.0, 9.0]);
        assert_eq!(anchor_origin(TextKind::Balloon, [40.0, 30.0], 40.0, 20.0), [20.0, 20.0]);
        // A zero-size box anchors exactly on its point in both kinds.
        assert_eq!(anchor_origin(TextKind::Balloon, [40.0, 30.0], 0.0, 0.0), [40.0, 30.0]);
    }
}
