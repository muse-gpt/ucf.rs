//! Panel grid math for comic pages and storyboard cards.

/// Even grid of `cols × rows` panels inside a page of `width × height` pixels.
///
/// `margin` is the paper border, `gutter` the gap between neighbours. Cells are
/// emitted reading order: left to right, top to bottom. Ported value-for-value
/// from the OpenComic core, including the clamp of degenerate extents to zero.
#[must_use]
pub fn grid(cols: u32, rows: u32, width: f32, height: f32, margin: f32, gutter: f32) -> Vec<[f32; 4]> {
    let (cols, rows) = (cols.max(1), rows.max(1));
    let cell_w = ((width - 2.0 * margin - (cols - 1) as f32 * gutter) / cols as f32).max(0.0);
    let cell_h = ((height - 2.0 * margin - (rows - 1) as f32 * gutter) / rows as f32).max(0.0);
    let mut rects = Vec::with_capacity((cols * rows) as usize);
    for row in 0..rows {
        for col in 0..cols {
            rects.push([
                margin + col as f32 * (cell_w + gutter),
                margin + row as f32 * (cell_h + gutter),
                cell_w,
                cell_h,
            ]);
        }
    }
    rects
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_grid_fills_the_page() {
        let rects = grid(2, 3, 400.0, 300.0, 20.0, 10.0);
        assert_eq!(rects.len(), 6);
        for rect in &rects {
            assert!(rect[0] >= 20.0 && rect[1] >= 20.0);
            assert!(rect[0] + rect[2] <= 380.0 + 0.001, "{rect:?} runs past the margin");
            assert!(rect[1] + rect[3] <= 280.0 + 0.001, "{rect:?} runs past the margin");
        }
    }

    #[test]
    fn cells_are_read_in_order() {
        let rects = grid(2, 2, 100.0, 100.0, 0.0, 0.0);
        assert_eq!(rects[0][0], 0.0);
        assert_eq!(rects[1][0], 50.0);
        assert_eq!(rects[2][1], 50.0);
    }

    #[test]
    fn a_single_cell_is_the_page() {
        let rects = grid(1, 1, 120.0, 80.0, 10.0, 8.0);
        assert_eq!(rects, vec![[10.0, 10.0, 100.0, 60.0]]);
    }
}
