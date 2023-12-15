//! Owned RGBA8 frame storage: pixels, clip bounds, sampling and the encoder
//! format conversion. Painting semantics live in `media.raster`.

/// An RGBA8 framebuffer owned by the renderer.
pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
    clip: Option<[u32; 4]>,
}

impl Frame {
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, pixels: vec![0; (width as usize) * (height as usize) * 4], clip: None }
    }

    /// Restricts every subsequent paint to `rect` (x0, y0, x1, y1).
    pub fn set_clip(&mut self, rect: [f32; 4]) {
        let x0 = rect[0].max(0.0) as u32;
        let y0 = rect[1].max(0.0) as u32;
        let x1 = rect[2].min(self.width as f32) as u32;
        let y1 = rect[3].min(self.height as f32) as u32;
        self.clip = if x1 > x0 && y1 > y0 { Some([x0, y0, x1, y1]) } else { Some([0, 0, 0, 0]) };
    }

    pub fn clear_clip(&mut self) {
        self.clip = None;
    }

    /// The active clip rectangle (x0, y0, x1, y1), if one is set.
    #[must_use]
    pub fn clip(&self) -> Option<[u32; 4]> {
        self.clip
    }

    /// Samples the frame with bilinear filtering at fractional coordinates.
    #[inline]
    #[must_use]
    pub fn sample(&self, sx: f32, sy: f32) -> [f32; 4] {
        let w = self.width.saturating_sub(1) as f32;
        let h = self.height.saturating_sub(1) as f32;
        let x = sx.clamp(0.0, w);
        let y = sy.clamp(0.0, h);
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        let (x0, y0) = (x0 as u32, y0 as u32);
        let x1 = (x0 + 1).min(self.width.saturating_sub(1));
        let y1 = (y0 + 1).min(self.height.saturating_sub(1));
        let at = |px: u32, py: u32| -> [f32; 4] {
            let i = ((py * self.width + px) as usize) * 4;
            [self.pixels[i] as f32, self.pixels[i + 1] as f32, self.pixels[i + 2] as f32, self.pixels[i + 3] as f32]
        };
        let (p00, p10, p01, p11) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
        let mut out = [0.0f32; 4];
        for k in 0..4 {
            let top = p00[k] + (p10[k] - p00[k]) * fx;
            let bottom = p01[k] + (p11[k] - p01[k]) * fx;
            out[k] = top + (bottom - top) * fy;
        }
        out
    }

    /// Converts RGBA8 to NV12 (BT.601, full-range) for the encoder.
    #[must_use]
    pub fn to_nv12(&self) -> Vec<u8> {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut out = vec![0u8; w * h + w * h / 2];
        let mut di = 0;
        for y in 0..h {
            let row = y * w * 4;
            for x in 0..w {
                let p = row + x * 4;
                let r = self.pixels[p] as f32 / 255.0;
                let g = self.pixels[p + 1] as f32 / 255.0;
                let b = self.pixels[p + 2] as f32 / 255.0;
                let yy = 0.299 * r + 0.587 * g + 0.114 * b;
                out[di] = (yy * 255.0).round().clamp(0.0, 255.0) as u8;
                di += 1;
            }
        }
        let mut uv = w * h;
        for by in (0..h).step_by(2) {
            for bx in (0..w).step_by(2) {
                let mut acc_u = 0.0f32;
                let mut acc_v = 0.0f32;
                let mut n = 0.0f32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let p = (by + dy) * w * 4 + (bx + dx) * 4;
                        let r = self.pixels[p] as f32 / 255.0;
                        let g = self.pixels[p + 1] as f32 / 255.0;
                        let b = self.pixels[p + 2] as f32 / 255.0;
                        acc_u += -0.168736 * r - 0.331264 * g + 0.5 * b;
                        acc_v += 0.5 * r - 0.418688 * g - 0.081312 * b;
                        n += 1.0;
                    }
                }
                out[uv] = ((acc_u / n) * 255.0 + 128.0).round().clamp(0.0, 255.0) as u8;
                out[uv + 1] = ((acc_v / n) * 255.0 + 128.0).round().clamp(0.0, 255.0) as u8;
                uv += 2;
            }
        }
        out
    }
}
