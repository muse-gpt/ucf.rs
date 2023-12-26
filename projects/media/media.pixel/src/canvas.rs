// 源自 OpenPixel（Apache-2.0, © OpenPixel Authors），索引语义逐值保持。

//! The indexed canvas and its tools.

use media_image::Bitmap;

use crate::palette::Palette;

/// A grid of palette indices.
///
/// Storage is `Vec<u8>` indices for the whole lifetime of the canvas:
/// every tool reads and writes indices, and colours appear only when
/// [`PixelCanvas::to_bitmap`] expands the sprite at the export boundary.
/// Index 0 is the transparent index.
#[derive(Clone, Debug)]
pub struct PixelCanvas {
    pub width: u32,
    pub height: u32,
    pixels: Vec<u8>,
}

impl PixelCanvas {
    /// Creates an empty (all-transparent) canvas. Zero dimensions clamp to
    /// one, exactly as the absorbed implementation does. The raw API is
    /// uncapped by design; the input budget for hostile sizes lives on the
    /// spec entry [`crate::spec::render_sprite`].
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        Self {
            width,
            height,
            pixels: vec![0; (width as usize) * (height as usize)],
        }
    }

    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Reads an index; out-of-bounds is transparent (0).
    ///
    /// Paired with [`PixelCanvas::set`]: the silent out-of-bounds contract
    /// is load-bearing for [`PixelCanvas::line`] and
    /// [`PixelCanvas::flood`], which feed it wrapped and negative-intent
    /// coordinates on purpose. Turning either into an error breaks those
    /// fills; the pair is locked by tests.
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> u8 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        self.pixels[(y * self.width + x) as usize]
    }

    /// Writes an index; out-of-bounds is ignored.
    ///
    /// See [`PixelCanvas::get`] for why the silence is a contract, not an
    /// oversight.
    pub fn set(&mut self, x: u32, y: u32, index: u8) {
        if x >= self.width || y >= self.height {
            return;
        }
        self.pixels[(y * self.width + x) as usize] = index;
    }

    /// Bresenham line.
    pub fn line(&mut self, x0: u32, y0: u32, x1: u32, y1: u32, index: u8) {
        let (mut x0, mut y0) = (x0 as i32, y0 as i32);
        let (x1, y1) = (x1 as i32, y1 as i32);
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut error = dx + dy;
        loop {
            if x0 >= 0 && y0 >= 0 {
                self.set(x0 as u32, y0 as u32, index);
            }
            if x0 == x1 && y0 == y1 {
                break;
            }
            let doubled = 2 * error;
            if doubled >= dy {
                error += dy;
                x0 += sx;
            }
            if doubled <= dx {
                error += dx;
                y0 += sy;
            }
        }
    }

    /// Rectangle outline or fill.
    pub fn rect(&mut self, x: u32, y: u32, w: u32, h: u32, index: u8, filled: bool) {
        if w == 0 || h == 0 {
            return;
        }
        if filled {
            for py in y..y + h {
                for px in x..x + w {
                    self.set(px, py, index);
                }
            }
            return;
        }
        self.line(x, y, x + w - 1, y, index);
        self.line(x, y + h - 1, x + w - 1, y + h - 1, index);
        self.line(x, y, x, y + h - 1, index);
        self.line(x + w - 1, y, x + w - 1, y + h - 1, index);
    }

    /// Four-way flood fill from a seed.
    ///
    /// Neighbours are pushed with `wrapping_sub`, so the stack legitimately
    /// carries `u32::MAX` coordinates at the edges. They are digested by
    /// the bounds checks here together with the silent out-of-bounds
    /// `get`/`set` contract — that is why the fill uses wrapping arithmetic
    /// without a panic or an error path. Replacing the silent contract with
    /// validation would require rewriting this loop; the pair is locked by
    /// `flood_survives_wrapped_neighbour_coordinates`.
    pub fn flood(&mut self, x: u32, y: u32, index: u8) {
        if x >= self.width || y >= self.height {
            return;
        }
        let target = self.get(x, y);
        if target == index {
            return;
        }
        let mut stack = vec![(x, y)];
        while let Some((px, py)) = stack.pop() {
            if px >= self.width || py >= self.height {
                continue;
            }
            if self.get(px, py) != target {
                continue;
            }
            self.set(px, py, index);
            stack.push((px.wrapping_sub(1), py));
            stack.push((px + 1, py));
            stack.push((px, py.wrapping_sub(1)));
            stack.push((px, py + 1));
        }
    }

    /// Nearest-neighbour upscale: the only correct way to enlarge pixel art.
    #[must_use]
    pub fn scale(&self, factor: u32) -> Self {
        let factor = factor.max(1);
        let mut out = Self::new(self.width * factor, self.height * factor);
        for y in 0..out.height {
            for x in 0..out.width {
                out.set(x, y, self.get(x / factor, y / factor));
            }
        }
        out
    }

    /// Expands the indices through the palette into an RGBA bitmap.
    ///
    /// This is the crate's only colour-producing edge; everything upstream
    /// of it stays in the index domain. Encoding the bitmap (PNG and
    /// friends) is composed by the caller with `media.codec`.
    #[must_use]
    pub fn to_bitmap(&self, palette: &Palette) -> Bitmap {
        let mut rgba = vec![0u8; self.pixels.len() * 4];
        for (pixel, slot) in self.pixels.iter().enumerate() {
            let color = palette.color(*slot);
            let offset = pixel * 4;
            rgba[offset..offset + 4].copy_from_slice(&color);
        }
        Bitmap::new(self.width, self.height, rgba)
    }

    /// Replaces every occurrence of one index; used for recolouring.
    pub fn replace(&mut self, from: u8, to: u8) {
        for slot in &mut self.pixels {
            if *slot == from {
                *slot = to;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pixel_stores_an_index() {
        let mut canvas = PixelCanvas::new(4, 4);
        canvas.set(2, 2, 5);
        assert_eq!(canvas.get(2, 2), 5);
        assert_eq!(canvas.get(0, 0), 0);
    }

    #[test]
    fn out_of_bounds_is_ignored_rather_than_panicking() {
        let mut canvas = PixelCanvas::new(4, 4);
        canvas.set(99, 99, 3);
        assert_eq!(canvas.get(99, 99), 0);
    }

    #[test]
    fn lines_are_continuous() {
        let mut canvas = PixelCanvas::new(16, 16);
        canvas.line(0, 0, 15, 15, 7);
        for i in 0..16 {
            assert_eq!(canvas.get(i, i), 7, "diagonal breaks at {i}");
        }
    }

    #[test]
    fn filled_rect_covers_the_area() {
        let mut canvas = PixelCanvas::new(8, 8);
        canvas.rect(2, 2, 3, 3, 4, true);
        assert_eq!(canvas.get(2, 2), 4);
        assert_eq!(canvas.get(4, 4), 4);
        assert_eq!(canvas.get(5, 5), 0);
    }

    #[test]
    fn flood_fill_stops_at_other_colors() {
        let mut canvas = PixelCanvas::new(5, 5);
        canvas.rect(1, 1, 3, 3, 2, true);
        canvas.flood(0, 0, 6);
        assert_eq!(canvas.get(0, 0), 6);
        assert_eq!(canvas.get(1, 1), 2, "the island must survive");
        assert_eq!(canvas.get(4, 4), 6);
    }

    #[test]
    fn scaling_is_nearest_neighbour() {
        let mut canvas = PixelCanvas::new(2, 2);
        canvas.set(0, 0, 1);
        let scaled = canvas.scale(3);
        assert_eq!((scaled.width, scaled.height), (6, 6));
        assert_eq!(scaled.get(2, 2), 1);
        assert_eq!(scaled.get(3, 3), 0);
    }

    #[test]
    fn palette_maps_indices_to_rgba() {
        let palette = Palette::default();
        let mut canvas = PixelCanvas::new(1, 1);
        canvas.set(0, 0, 1);
        let bitmap = canvas.to_bitmap(&palette);
        assert_eq!((bitmap.width, bitmap.height), (1, 1));
        assert_eq!(&bitmap.rgba[0..4], &palette.color(1));
    }

    #[test]
    fn editing_stays_in_the_index_domain() {
        // The hard boundary: tools write indices, storage holds indices and
        // colours appear only through the export call.
        let mut canvas = PixelCanvas::new(2, 2);
        canvas.set(0, 0, 1);
        canvas.rect(1, 0, 1, 1, 2, true);
        assert_eq!(canvas.pixels(), &[1, 2, 0, 0]);
        let bitmap = canvas.to_bitmap(&Palette::default());
        assert_eq!(&bitmap.rgba[0..4], &Palette::default().color(1));
        assert_eq!(&bitmap.rgba[4..8], &Palette::default().color(2));
        assert_eq!(&bitmap.rgba[8..12], &[0, 0, 0, 0], "index 0 exports transparent");
    }

    #[test]
    fn empty_dimensions_clamp_to_one() {
        let canvas = PixelCanvas::new(0, 0);
        assert_eq!((canvas.width, canvas.height), (1, 1));
        assert_eq!(canvas.pixels().len(), 1);
    }

    #[test]
    fn line_endpoints_are_inclusive() {
        let mut canvas = PixelCanvas::new(8, 8);
        canvas.line(2, 2, 5, 2, 9);
        for x in 2..=5 {
            assert_eq!(canvas.get(x, 2), 9, "endpoint span");
        }
        assert_eq!(canvas.get(1, 2), 0);
        assert_eq!(canvas.get(6, 2), 0);
        canvas.line(3, 3, 3, 3, 4);
        assert_eq!(canvas.get(3, 3), 4, "degenerate line paints its point");
    }

    #[test]
    fn rect_outline_shares_corners_and_zero_size_is_noop() {
        let mut canvas = PixelCanvas::new(8, 8);
        canvas.rect(1, 1, 3, 3, 5, false);
        assert_eq!(canvas.get(1, 1), 5);
        assert_eq!(canvas.get(3, 1), 5);
        assert_eq!(canvas.get(1, 3), 5);
        assert_eq!(canvas.get(3, 3), 5);
        assert_eq!(canvas.get(2, 2), 0, "outline leaves the interior");
        canvas.rect(5, 5, 0, 4, 6, true);
        canvas.rect(5, 5, 4, 0, 6, false);
        assert_eq!(canvas.get(5, 5), 0, "zero extent never paints");
    }

    #[test]
    fn flood_survives_wrapped_neighbour_coordinates() {
        // Locks the pair: the wrapping neighbour push plus the silent
        // out-of-bounds read/write contract. Corner seeds immediately put
        // u32::MAX coordinates on the stack.
        let mut canvas = PixelCanvas::new(3, 3);
        canvas.flood(0, 0, 5);
        for y in 0..3 {
            for x in 0..3 {
                assert_eq!(canvas.get(x, y), 5, "({x},{y})");
            }
        }
        let mut single = PixelCanvas::new(1, 1);
        single.flood(0, 0, 2);
        assert_eq!(single.get(0, 0), 2);
        let mut untouched = PixelCanvas::new(2, 2);
        untouched.set(1, 1, 4);
        untouched.flood(99, 99, 7);
        assert_eq!(untouched.get(1, 1), 4, "out-of-bounds seed is a no-op");
        untouched.flood(1, 1, 4);
        assert_eq!(untouched.get(1, 1), 4, "same-colour seed is a no-op");
    }

    #[test]
    fn scale_clamps_factor_below_one() {
        let mut canvas = PixelCanvas::new(2, 3);
        canvas.set(1, 2, 8);
        let same = canvas.scale(0);
        assert_eq!((same.width, same.height), (2, 3));
        assert_eq!(same.get(1, 2), 8);
    }

    #[test]
    fn replace_swaps_every_occurrence() {
        let mut canvas = PixelCanvas::new(2, 2);
        canvas.set(0, 0, 3);
        canvas.set(1, 1, 3);
        canvas.replace(3, 9);
        assert_eq!(canvas.pixels(), &[9, 0, 0, 9]);
        canvas.replace(0, 1);
        assert_eq!(canvas.pixels(), &[9, 1, 1, 9]);
    }

    #[test]
    fn pseudo_random_operations_keep_the_index_invariants() {
        // Property-style sweep without a fuzz dependency: any mix of tools
        // must keep storage in the index domain, never panic and keep the
        // out-of-bounds contract.
        let mut state: u32 = 0x9E37_79B9;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut canvas = PixelCanvas::new(8, 8);
        let mut seen = vec![0u8];
        for _ in 0..300 {
            let (a, b, c, d, i) = (next() % 10, next() % 10, next() % 10, next() % 10, next() as u8);
            match next() % 5 {
                0 => canvas.set(a, b, i),
                1 => canvas.line(a, b, c, d, i),
                2 => canvas.rect(a, b, c % 4, d % 4, i, next() % 2 == 0),
                3 => canvas.flood(a, b, i),
                _ => canvas.replace(i, i.wrapping_add(1)),
            }
            if i < 8 {
                seen.push(i);
            }
            assert_eq!(canvas.pixels().len(), 64, "storage never leaves the index grid");
            assert_eq!(canvas.get(100, 100), 0);
            canvas.set(100, 100, i);
            assert_eq!(canvas.get(100, 100), 0);
        }
        let bitmap = canvas.to_bitmap(&Palette::default());
        assert_eq!(bitmap.rgba.len(), 8 * 8 * 4);
    }
}
