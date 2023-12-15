//! Decoded RGBA8 image storage, decoupled from whatever decoded it.

/// A decoded RGBA8 image, decoupled from whatever decoded it.
pub struct Bitmap {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl Bitmap {
    #[must_use]
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Self {
        Self { width, height, rgba }
    }

    /// Converts a BGRA8 buffer (the layout MF hands back) into RGBA8.
    ///
    /// MF's RGB32 has no alpha channel, so the fourth byte is undefined and is
    /// replaced with full opacity.
    #[must_use]
    pub fn from_bgra(width: u32, height: u32, bgra: &[u8], stride: usize) -> Self {
        let mut rgba = vec![0u8; (width as usize) * (height as usize) * 4];
        for y in 0..height as usize {
            let src = y * stride;
            let dst = y * (width as usize) * 4;
            for x in 0..width as usize {
                let s = src + x * 4;
                if s + 3 >= bgra.len() {
                    break;
                }
                let d = dst + x * 4;
                rgba[d] = bgra[s + 2];
                rgba[d + 1] = bgra[s + 1];
                rgba[d + 2] = bgra[s];
                rgba[d + 3] = 255;
            }
        }
        Self { width, height, rgba }
    }

    /// Bilinear sample at normalized-free pixel coordinates.
    #[inline]
    #[must_use]
    pub fn sample(&self, sx: f32, sy: f32) -> [f32; 4] {
        let x = sx.clamp(0.0, self.width.saturating_sub(1) as f32);
        let y = sy.clamp(0.0, self.height.saturating_sub(1) as f32);
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width.saturating_sub(1) as usize);
        let y1 = (y0 + 1).min(self.height.saturating_sub(1) as usize);
        let fx = x - x0 as f32;
        let fy = y - y0 as f32;
        let at = |px: usize, py: usize| -> [f32; 4] {
            let i = (py * self.width as usize + px) * 4;
            [self.rgba[i] as f32, self.rgba[i + 1] as f32, self.rgba[i + 2] as f32, self.rgba[i + 3] as f32]
        };
        let p00 = at(x0, y0);
        let p10 = at(x1, y0);
        let p01 = at(x0, y1);
        let p11 = at(x1, y1);
        let mut out = [0.0f32; 4];
        for k in 0..4 {
            let top = p00[k] + (p10[k] - p00[k]) * fx;
            let bottom = p01[k] + (p11[k] - p01[k]) * fx;
            out[k] = top + (bottom - top) * fy;
        }
        out
    }
}
