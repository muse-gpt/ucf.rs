//! Round brushes and stroke stamping.

use geometry::Rgba;
use media_image::Frame;

/// Whether a stamp adds light or replaces paint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BrushMode {
    #[default]
    Paint,
    Erase,
    Add,
}

/// A soft round brush.
#[derive(Clone, Copy, Debug)]
pub struct Brush {
    /// Radius in pixels.
    pub radius: f32,
    /// 0 is a hard disc, 1 is a wide falloff.
    pub hardness: f32,
    pub color: Rgba,
    /// 0..1 per stamp.
    pub flow: f32,
    pub mode: BrushMode,
}

impl Brush {
    #[must_use]
    pub fn round(radius: f32, color: Rgba) -> Self {
        Self {
            radius: radius.max(0.5),
            hardness: 0.6,
            color,
            flow: 1.0,
            mode: BrushMode::Paint,
        }
    }

    #[must_use]
    pub fn soft(radius: f32, color: Rgba) -> Self {
        Self {
            hardness: 0.15,
            ..Self::round(radius, color)
        }
    }

    #[must_use]
    pub fn eraser(radius: f32) -> Self {
        Self {
            mode: BrushMode::Erase,
            ..Self::round(radius, Rgba([0, 0, 0, 255]))
        }
    }

    /// Stamps one dab at `(cx, cy)`.
    pub fn stamp(&self, frame: &mut Frame, cx: f32, cy: f32) {
        let radius = self.radius.max(0.5);
        let x0 = (cx - radius).max(0.0) as u32;
        let y0 = (cy - radius).max(0.0) as u32;
        let x1 = ((cx + radius) as u32).min(frame.width.saturating_sub(1));
        let y1 = ((cy + radius) as u32).min(frame.height.saturating_sub(1));
        let inner = radius * (1.0 - self.hardness.clamp(0.0, 1.0));
        for y in y0..=y1 {
            for x in x0..=x1 {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let distance = (dx * dx + dy * dy).sqrt();
                if distance > radius {
                    continue;
                }
                // Soft edge between `inner` and `radius`.
                let falloff = if distance <= inner {
                    1.0
                } else {
                    ((radius - distance) / (radius - inner).max(1e-3)).clamp(0.0, 1.0)
                };
                let gain = falloff * self.flow.clamp(0.0, 1.0);
                if gain <= 0.001 {
                    continue;
                }
                let index = ((y * frame.width + x) * 4) as usize;
                match self.mode {
                    BrushMode::Paint => {
                        let alpha = self.color.0[3] as f32 / 255.0 * gain;
                        for k in 0..3 {
                            let src = self.color.0[k] as f32;
                            let dst = frame.pixels[index + k] as f32;
                            frame.pixels[index + k] = (dst + (src - dst) * alpha) as u8;
                        }
                        frame.pixels[index + 3] = 255;
                    }
                    BrushMode::Erase => {
                        frame.pixels[index + 3] =
                            (frame.pixels[index + 3] as f32 * (1.0 - gain)) as u8;
                    }
                    BrushMode::Add => {
                        for k in 0..3 {
                            let added = self.color.0[k] as f32 * gain;
                            frame.pixels[index + k] =
                                (frame.pixels[index + k] as f32 + added).min(255.0) as u8;
                        }
                        frame.pixels[index + 3] = 255;
                    }
                }
            }
        }
    }

    /// Stamps along a polyline with a fixed spacing.
    pub fn stroke(&self, frame: &mut Frame, points: &[[f32; 2]]) {
        if points.is_empty() {
            return;
        }
        let spacing = (self.radius * 0.33).max(0.75);
        self.stamp(frame, points[0][0], points[0][1]);
        for pair in points.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            let dx = b[0] - a[0];
            let dy = b[1] - a[1];
            let length = (dx * dx + dy * dy).sqrt();
            if length <= f32::EPSILON {
                continue;
            }
            let steps = (length / spacing).ceil() as u32;
            for step in 1..=steps {
                let t = step as f32 / steps as f32;
                self.stamp(frame, a[0] + dx * t, a[1] + dy * t);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_raster::Paint;

    #[test]
    fn a_stamp_paints_the_center() {
        let mut frame = Frame::new(32, 32);
        Brush::round(6.0, Rgba([255, 255, 255, 255])).stamp(&mut frame, 16.0, 16.0);
        let center = ((16 * 32 + 16) * 4) as usize;
        assert_eq!(&frame.pixels[center..center + 3], &[255, 255, 255]);
        assert_eq!(frame.pixels[center + 3], 255);
    }

    #[test]
    fn a_stroke_covers_the_whole_segment() {
        let mut frame = Frame::new(64, 16);
        Brush::round(3.0, Rgba([255, 255, 255, 255])).stroke(&mut frame, &[[4.0, 8.0], [60.0, 8.0]]);
        for x in [6u32, 20, 40, 58] {
            let index = ((8 * 64 + x) * 4) as usize;
            assert_eq!(frame.pixels[index + 3], 255, "x={x}");
        }
    }

    #[test]
    fn erasing_clears_alpha() {
        let mut frame = Frame::new(16, 16);
        frame.fill(Rgba([200, 200, 200, 255]));
        Brush::eraser(4.0).stamp(&mut frame, 8.0, 8.0);
        let center = ((8 * 16 + 8) * 4) as usize;
        assert!(frame.pixels[center + 3] < 40);
    }

    #[test]
    fn additive_stamps_only_brighten() {
        let mut frame = Frame::new(8, 8);
        frame.fill(Rgba([10, 10, 10, 255]));
        let brush = Brush {
            mode: BrushMode::Add,
            ..Brush::round(4.0, Rgba([60, 60, 60, 255]))
        };
        brush.stamp(&mut frame, 4.0, 4.0);
        let center = ((4 * 8 + 4) * 4) as usize;
        assert!(frame.pixels[center] >= 60, "{}", frame.pixels[center]);
    }
}
