//! The numbered contact-sheet plan: every page scaled into a cell, tiled and
//! labelled like a shot card. Pure geometry — painting and encoding live in
//! `media.render`'s `layout-export` entry.
//!
//! The geometry is absorbed value-for-value from the OpenComic core's `sheet`:
//! a fixed 16 px gutter, cells of `round(page × scale).max(1)`, label height
//! `cell_h / 8`, bold page numbers at `(x + 4, y + label)` in ink
//! `(20, 20, 24, 235)` over a `(236, 236, 238)` sheet, and an empty document
//! still producing a one-row sheet.

use geometry::Rgba;
use media_text::FontId;
use media_types::Error;

use crate::model::ComicDoc;
use crate::spec::{MAX_SHEET_SCALE, MIN_SHEET_SCALE};

/// Fixed gutter between cells and around the sheet edge.
pub const SHEET_GUTTER: u32 = 16;
/// Sheet background colour.
pub const SHEET_BACKGROUND: Rgba = Rgba([236, 236, 238, 255]);
/// Page-number label colour.
pub const SHEET_LABEL_INK: Rgba = Rgba([20, 20, 24, 235]);
/// Horizontal label offset inside the cell.
pub const SHEET_LABEL_OFFSET_X: f32 = 4.0;
/// Label height is the cell height divided by this.
pub const SHEET_LABEL_DIVISOR: u32 = 8;

/// One tiled page: where its scaled picture goes and what it is labelled.
#[derive(Clone, Debug, PartialEq)]
pub struct SheetCell {
    pub page: usize,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub label: String,
}

/// The whole sheet as a paintable plan.
#[derive(Clone, Debug, PartialEq)]
pub struct SheetPlan {
    pub width: u32,
    pub height: u32,
    pub background: Rgba,
    pub cells: Vec<SheetCell>,
    /// Label text size in pixels (the source `cell_h / 8`).
    pub label_size: u32,
    pub label_font: FontId,
    pub label_color: Rgba,
    pub label_offset_x: f32,
}

/// Lays out the sheet. The scale and column count are validated instead of
/// the source's silent clamps.
pub fn sheet_plan(comic: &ComicDoc, scale: f32, cols: usize) -> Result<SheetPlan, Error> {
    if !scale.is_finite() || !(MIN_SHEET_SCALE..=MAX_SHEET_SCALE).contains(&scale) {
        return Err(Error::invalid_input(format!(
            "sheet scale {scale}: must be finite and within {MIN_SHEET_SCALE}..={MAX_SHEET_SCALE}"
        )));
    }
    if cols < 1 {
        return Err(Error::invalid_input("sheet cols must be at least 1"));
    }
    let cell_w = (comic.width as f32 * scale).round().max(1.0) as u32;
    let cell_h = (comic.height as f32 * scale).round().max(1.0) as u32;
    let rows = if comic.pages.is_empty() { 1 } else { (comic.pages.len() + cols - 1) / cols };
    let cells: Vec<SheetCell> = (0..comic.pages.len())
        .map(|index| {
            let col = (index % cols) as u32;
            let row = (index / cols) as u32;
            SheetCell {
                page: index,
                x: SHEET_GUTTER + col * (cell_w + SHEET_GUTTER),
                y: SHEET_GUTTER + row * (cell_h + SHEET_GUTTER),
                width: cell_w,
                height: cell_h,
                label: (index + 1).to_string(),
            }
        })
        .collect();
    Ok(SheetPlan {
        width: SHEET_GUTTER * (cols as u32 + 1) + cell_w * cols as u32,
        height: SHEET_GUTTER * (rows as u32 + 1) + cell_h * rows as u32,
        background: SHEET_BACKGROUND,
        cells,
        label_size: cell_h / SHEET_LABEL_DIVISOR,
        label_font: FontId::Bold,
        label_color: SHEET_LABEL_INK,
        label_offset_x: SHEET_LABEL_OFFSET_X,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Page;

    fn comic(pages: usize, width: u32, height: u32) -> ComicDoc {
        let mut comic = ComicDoc::new(width, height, Rgba([255, 255, 255, 255]));
        for _ in 0..pages {
            comic.pages.push(Page::new());
        }
        comic
    }

    #[test]
    fn the_sheet_tiles_every_page() {
        // The source test, kept: 3 pages of 400x300 at scale 0.5, 2 columns.
        let plan = sheet_plan(&comic(3, 400, 300), 0.5, 2).expect("sheet");
        assert_eq!((plan.width, plan.height), (16 * 3 + 200 * 2, 16 * 3 + 150 * 2));
        assert_eq!(plan.cells.len(), 3);
        assert_eq!((plan.cells[0].x, plan.cells[0].y), (16, 16));
        assert_eq!((plan.cells[1].x, plan.cells[1].y), (16 + 216, 16));
        assert_eq!((plan.cells[2].x, plan.cells[2].y), (16, 16 + 166));
    }

    #[test]
    fn labels_number_pages_from_one() {
        let plan = sheet_plan(&comic(3, 400, 300), 0.5, 2).expect("sheet");
        let labels: Vec<&str> = plan.cells.iter().map(|cell| cell.label.as_str()).collect();
        assert_eq!(labels, ["1", "2", "3"]);
        assert_eq!(plan.label_size, 150 / 8);
        assert_eq!(plan.label_font, FontId::Bold);
        assert_eq!(plan.label_color, Rgba([20, 20, 24, 235]));
        assert_eq!(plan.label_offset_x, 4.0);
        assert_eq!(plan.background, Rgba([236, 236, 238, 255]));
    }

    #[test]
    fn an_empty_document_is_a_one_row_sheet() {
        // Source boundary, kept: no pages still yields a background-only sheet.
        let plan = sheet_plan(&comic(0, 400, 300), 0.5, 2).expect("sheet");
        assert!(plan.cells.is_empty());
        assert_eq!(plan.height, 16 * 2 + 150);
    }

    #[test]
    fn cell_sizes_round_and_floor_like_the_source() {
        let plan = sheet_plan(&comic(1, 400, 300), 0.333, 1).expect("sheet");
        assert_eq!((plan.cells[0].width, plan.cells[0].height), (133, 100));
        let tiny = sheet_plan(&comic(1, 1, 1), 0.05, 1).expect("sheet");
        assert_eq!((tiny.cells[0].width, tiny.cells[0].height), (1, 1));
    }

    #[test]
    fn sheet_parameters_are_validated() {
        // The source clamped these silently; now they reject explicitly.
        assert_eq!(sheet_plan(&comic(1, 400, 300), 0.5, 0).expect_err("zero cols rejects").code(), "invalid_input");
        assert_eq!(sheet_plan(&comic(1, 400, 300), 0.01, 2).expect_err("tiny scale rejects").code(), "invalid_input");
        assert_eq!(sheet_plan(&comic(1, 400, 300), 5.0, 2).expect_err("huge scale rejects").code(), "invalid_input");
        let error = sheet_plan(&comic(1, 400, 300), f32::INFINITY, 2).expect_err("infinite scale rejects");
        assert!(error.message.contains("finite"), "{}", error.message);
    }
}
