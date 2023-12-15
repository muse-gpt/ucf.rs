//! Software rasterizer: paint operations on the `Frame` storage owned by
//! `media.image`. Everything here is plain integer/floating math; there is
//! no GPU and no external imaging library.

use geometry::Rgba;
use media_image::{Bitmap, Fit, Frame};
use media_types::hash;

/// Paint semantics for an RGBA8 frame: blending, primitives, image placement
/// and camera resampling. The pixel storage itself lives in `media.image`.
pub trait Paint {
    /// Source-over blend of one pixel, honoring the active clip.
    fn blend(&mut self, x: u32, y: u32, c: Rgba);

    /// Additive blend of one pixel (light, glow, overlays).
    fn add_pixel(&mut self, x: u32, y: u32, c: Rgba);

    /// Paints a solid color everywhere.
    fn fill(&mut self, c: Rgba);

    /// Paints a vertical gradient through the given color stops (evenly spaced).
    fn v_gradient(&mut self, stops: &[Rgba]);

    /// Paints a radial gradient from `center` color to `edge` color.
    fn radial(&mut self, center: Rgba, edge: Rgba);

    /// Source-over rectangle. Coordinates are clamped to the frame.
    fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, c: Rgba);

    /// Rounded rectangle with a corner radius in pixels.
    fn round_rect(&mut self, x: f32, y: f32, w: f32, h: f32, radius: f32, c: Rgba);

    /// Camera transform: samples `src` around `focus` at `zoom` and optional
    /// rotation, then writes the result into this frame. Used to push, pull and
    /// pan a whole scene like a real lens.
    fn resample_from(&mut self, src: &Frame, zoom: f32, focus: [f32; 2], rotate_deg: f32);

    /// Paints a soft-edged band across the frame at `angle` degrees.
    ///
    /// Falloff is gaussian so the band reads as light in the air rather than as
    /// a painted stripe.
    fn band(&mut self, center_y: f32, angle_deg: f32, thickness: f32, c: Rgba);

    /// Sprinkles deterministic grain; `seed` changes per frame.
    fn grain(&mut self, density: u32, strength: f32, seed: u32);

    /// Fills a closed polygon with 4x vertical supersampling for soft edges.
    ///
    /// Coverage is accumulated per pixel and composited once, so a solid fill
    /// stays solid instead of stacking four translucent passes.
    fn fill_polygon(&mut self, points: &[[f32; 2]], c: Rgba, additive: bool);

    /// Strokes a polyline, optionally only the first `progress` of its length.
    fn stroke_polyline(&mut self, points: &[[f32; 2]], width: f32, c: Rgba, progress: f32, additive: bool);

    /// Ring or disc.
    fn circle(&mut self, cx: f32, cy: f32, radius: f32, c: Rgba, stroke: Option<f32>);

    /// Straight line with thickness.
    fn line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32, c: Rgba);

    /// Additive soft glow dot for star fields and highlights.
    fn glow(&mut self, cx: f32, cy: f32, radius: f32, c: Rgba);

    /// Draws `bitmap` into the box `(x, y, w, h)` honoring `fit`.
    ///
    /// `alpha` scales the source alpha; `radius` clips the result to rounded
    /// corners (0 draws a plain rectangle).
    fn draw_bitmap(&mut self, bitmap: &Bitmap, x: f32, y: f32, w: f32, h: f32, fit: Fit, alpha: f32, radius: f32);

    /// Deterministic star field: hash-based positions, sine twinkle, slow drift.
    fn starfield(&mut self, count: u32, tint: Rgba, drift_px_per_sec: f32, twinkle: f32, seconds: f32);

    /// Darkens the frame edges; `vignette` holds one alpha byte per pixel.
    fn apply_vignette(&mut self, factor: &[u8]);
}

#[inline]
fn inside_clip(frame: &Frame, x: u32, y: u32) -> bool {
    match frame.clip() {
        None => true,
        Some([x0, y0, x1, y1]) => x >= x0 && x < x1 && y >= y0 && y < y1,
    }
}

#[inline]
fn set_pixel(frame: &mut Frame, x: u32, y: u32, c: Rgba) {
    let i = ((y * frame.width + x) as usize) * 4;
    frame.pixels[i] = c.0[0];
    frame.pixels[i + 1] = c.0[1];
    frame.pixels[i + 2] = c.0[2];
    frame.pixels[i + 3] = c.0[3];
}

#[inline]
fn add(frame: &mut Frame, x: u32, y: u32, c: Rgba) {
    let i = ((y * frame.width + x) as usize) * 4;
    for k in 0..3 {
        let v = frame.pixels[i + k] as u32 + (c.0[k] as u32 * c.0[3] as u32 / 255);
        frame.pixels[i + k] = v.min(255) as u8;
    }
}

fn add_pixel_line(frame: &mut Frame, x0: f32, y0: f32, x1: f32, y1: f32, c: Rgba) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let steps = dx.abs().max(dy.abs()).max(1.0) as u32;
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let px = x0 + dx * t;
        let py = y0 + dy * t;
        if px < 0.0 || py < 0.0 || px >= frame.width as f32 || py >= frame.height as f32 {
            continue;
        }
        frame.add_pixel(px as u32, py as u32, c);
    }
}

fn stamp_disc(frame: &mut Frame, cx: f32, cy: f32, radius: f32, c: Rgba, additive: bool) {
    let x0 = (cx - radius).max(0.0) as u32;
    let y0 = (cy - radius).max(0.0) as u32;
    let x1 = (cx + radius).min(frame.width as f32 - 1.0) as u32;
    let y1 = (cy + radius).min(frame.height as f32 - 1.0) as u32;
    for py in y0..=y1 {
        for px in x0..=x1 {
            let dx = px as f32 - cx;
            let dy = py as f32 - cy;
            if dx * dx + dy * dy <= radius * radius {
                if additive {
                    frame.add_pixel(px, py, c);
                } else {
                    frame.blend(px, py, c);
                }
            }
        }
    }
}

impl Paint for Frame {
    #[inline]
    fn blend(&mut self, x: u32, y: u32, c: Rgba) {
        if !inside_clip(self, x, y) {
            return;
        }
        let i = ((y * self.width + x) as usize) * 4;
        let a = c.0[3] as u32;
        if a >= 255 {
            set_pixel(self, x, y, c);
            return;
        }
        if a == 0 {
            return;
        }
        let inv = 255 - a;
        for k in 0..4 {
            let dst = self.pixels[i + k] as u32;
            self.pixels[i + k] = ((c.0[k] as u32 * a + dst * inv) / 255) as u8;
        }
    }

    #[inline]
    fn add_pixel(&mut self, x: u32, y: u32, c: Rgba) {
        if !inside_clip(self, x, y) {
            return;
        }
        add(self, x, y, c);
    }

    fn fill(&mut self, c: Rgba) {
        for chunk in self.pixels.chunks_mut(4) {
            chunk.copy_from_slice(&c.0);
        }
    }

    fn v_gradient(&mut self, stops: &[Rgba]) {
        if stops.is_empty() {
            return;
        }
        if stops.len() == 1 {
            self.fill(stops[0]);
            return;
        }
        let h = self.height;
        let w = self.width;
        let segments = stops.len() - 1;
        for y in 0..h {
            let t = (y as f32 / (h - 1).max(1) as f32) * segments as f32;
            let seg = (t as usize).min(segments - 1);
            let local = t - seg as f32;
            let c = stops[seg].lerp(stops[seg + 1], local);
            let row = (y as usize) * (w as usize) * 4;
            for px in self.pixels[row..row + (w as usize) * 4].chunks_mut(4) {
                px.copy_from_slice(&c.0);
            }
        }
    }

    fn radial(&mut self, center: Rgba, edge: Rgba) {
        let cx = self.width as f32 / 2.0;
        let cy = self.height as f32 / 2.0;
        let max_r = (cx * cx + cy * cy).sqrt();
        for y in 0..self.height {
            for x in 0..self.width {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let t = (dx * dx + dy * dy).sqrt() / max_r;
                let c = center.lerp(edge, t);
                set_pixel(self, x, y, c);
            }
        }
    }

    fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, c: Rgba) {
        let x0 = x.max(0.0) as u32;
        let y0 = y.max(0.0) as u32;
        let x1 = (x + w).min(self.width as f32) as u32;
        let y1 = (y + h).min(self.height as f32) as u32;
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        for py in y0..y1 {
            for px in x0..x1 {
                self.blend(px, py, c);
            }
        }
    }

    fn round_rect(&mut self, x: f32, y: f32, w: f32, h: f32, radius: f32, c: Rgba) {
        if radius <= 0.01 {
            self.rect(x, y, w, h, c);
            return;
        }
        let r = radius.min(w / 2.0).min(h / 2.0);
        let x0 = x.max(0.0) as u32;
        let y0 = y.max(0.0) as u32;
        let x1 = (x + w).min(self.width as f32) as u32;
        let y1 = (y + h).min(self.height as f32) as u32;
        for py in y0..y1 {
            for px in x0..x1 {
                let fx = px as f32;
                let fy = py as f32;
                let cx = if fx < x + r {
                    x + r
                } else if fx > x + w - r {
                    x + w - r
                } else {
                    fx
                };
                let cy = if fy < y + r {
                    y + r
                } else if fy > y + h - r {
                    y + h - r
                } else {
                    fy
                };
                let dx = fx - cx;
                let dy = fy - cy;
                if dx * dx + dy * dy <= r * r {
                    self.blend(px, py, c);
                }
            }
        }
    }

    fn resample_from(&mut self, src: &Frame, zoom: f32, focus: [f32; 2], rotate_deg: f32) {
        let zoom = zoom.max(0.05);
        let angle = rotate_deg.to_radians();
        let (sin, cos) = angle.sin_cos();
        let w = self.width as f32;
        let h = self.height as f32;
        // How many source texels land on one output pixel. Above 1 the frame is
        // shrinking, so a single bilinear tap would shimmer and alias; average a
        // small box instead.
        let ratio = src.width as f32 / (w * zoom);
        let taps = (ratio.ceil() as u32).clamp(1, 3);
        let step = ratio / taps as f32;
        for y in 0..self.height {
            let v = (y as f32 + 0.5) / h - 0.5;
            for x in 0..self.width {
                let u = (x as f32 + 0.5) / w - 0.5;
                // Undo zoom, then rotate the sample position.
                let zu = u / zoom;
                let zv = v / zoom;
                let ru = zu * cos + zv * sin;
                let rv = -zu * sin + zv * cos;
                let sx = (focus[0] + ru) * src.width as f32 - 0.5;
                let sy = (focus[1] + rv) * src.height as f32 - 0.5;
                let texel = if taps == 1 {
                    src.sample(sx, sy)
                } else {
                    let mut acc = [0.0f32; 4];
                    let half = (taps as f32 - 1.0) / 2.0;
                    for ty in 0..taps {
                        for tx in 0..taps {
                            let t = src.sample(sx + (tx as f32 - half) * step, sy + (ty as f32 - half) * step);
                            for k in 0..4 {
                                acc[k] += t[k];
                            }
                        }
                    }
                    let n = (taps * taps) as f32;
                    [acc[0] / n, acc[1] / n, acc[2] / n, acc[3] / n]
                };
                let i = ((y * self.width + x) as usize) * 4;
                self.pixels[i] = texel[0] as u8;
                self.pixels[i + 1] = texel[1] as u8;
                self.pixels[i + 2] = texel[2] as u8;
                self.pixels[i + 3] = 255;
            }
        }
    }

    fn band(&mut self, center_y: f32, angle_deg: f32, thickness: f32, c: Rgba) {
        let angle = angle_deg.to_radians();
        let (sin, cos) = angle.sin_cos();
        let half = thickness / 2.0;
        let steps = thickness.ceil().max(1.0) as u32;
        for s in 0..steps {
            let offset = s as f32 - half;
            let normalized = offset / half;
            let falloff = (-normalized * normalized * 2.6).exp();
            if falloff <= 0.002 {
                continue;
            }
            let alpha = (c.0[3] as f32 * falloff) as u8;
            if alpha == 0 {
                continue;
            }
            let color = Rgba([c.0[0], c.0[1], c.0[2], alpha]);
            let cx = self.width as f32 / 2.0 + offset * -sin;
            let cy = center_y + offset * cos;
            let dx = cos * self.width as f32;
            let dy = sin * self.width as f32;
            add_pixel_line(self, cx - dx, cy - dy, cx + dx, cy + dy, color);
        }
    }

    fn grain(&mut self, density: u32, strength: f32, seed: u32) {
        let amount = ((strength.clamp(0.0, 1.0)) * 255.0) as u8;
        for i in 0..density {
            let h = hash(seed.wrapping_mul(2_654_435_761).wrapping_add(i));
            let x = (h & 0xffff) % self.width;
            let y = ((h >> 16) & 0xffff) % self.height;
            let bright = ((h >> 8) & 1) == 1;
            let value = if bright { amount } else { 0 };
            let c = Rgba([value, value, value, (amount as f32 * 0.5) as u8]);
            self.add_pixel(x, y, c);
        }
    }

    fn fill_polygon(&mut self, points: &[[f32; 2]], c: Rgba, additive: bool) {
        if points.len() < 3 || c.0[3] == 0 {
            return;
        }
        let min_y = points.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min).max(0.0);
        let max_y = points.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max).min(self.height as f32 - 1.0);
        if max_y <= min_y {
            return;
        }
        let samples = 4u32;
        let mut xs: Vec<f32> = Vec::with_capacity(16);
        let mut coverage = vec![0u8; self.width as usize];
        for row in min_y.floor() as u32..=max_y.ceil() as u32 {
            if row >= self.height {
                break;
            }
            coverage.fill(0);
            for s in 0..samples {
                let y = row as f32 + (s as f32 + 0.5) / samples as f32;
                xs.clear();
                for i in 0..points.len() {
                    let a = points[i];
                    let b = points[(i + 1) % points.len()];
                    if (a[1] <= y) == (b[1] <= y) {
                        continue;
                    }
                    let t = (y - a[1]) / (b[1] - a[1]);
                    xs.push(a[0] + (b[0] - a[0]) * t);
                }
                if xs.is_empty() {
                    continue;
                }
                xs.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
                // Even-odd rule: fill between consecutive crossings.
                for pair in xs.as_chunks::<2>().0 {
                    let (x0, x1) = (pair[0], pair[1]);
                    if x1 <= x0 {
                        continue;
                    }
                    let start = x0.max(0.0) as u32;
                    let end = x1.min(self.width as f32 - 1.0) as u32;
                    for px in start..=end {
                        coverage[px as usize] = coverage[px as usize].saturating_add(1);
                    }
                }
            }
            for px in 0..self.width {
                let covered = coverage[px as usize];
                if covered == 0 {
                    continue;
                }
                let alpha = c.0[3] as f32 * covered as f32 / samples as f32;
                if alpha < 1.0 {
                    continue;
                }
                let color = Rgba([c.0[0], c.0[1], c.0[2], alpha as u8]);
                if additive {
                    self.add_pixel(px, row, color);
                } else {
                    self.blend(px, row, color);
                }
            }
        }
    }

    fn stroke_polyline(&mut self, points: &[[f32; 2]], width: f32, c: Rgba, progress: f32, additive: bool) {
        if points.len() < 2 {
            return;
        }
        let progress = progress.clamp(0.0, 1.0);
        let mut total = 0.0f32;
        for i in 1..points.len() {
            total += distance(points[i - 1], points[i]);
        }
        if total <= 0.0 {
            return;
        }
        let limit = total * progress;
        let mut walked = 0.0f32;
        let half = (width / 2.0).max(0.5);
        for i in 1..points.len() {
            let (a, b) = (points[i - 1], points[i]);
            let segment = distance(a, b);
            if segment <= 0.0001 {
                continue;
            }
            let usable = (limit - walked).min(segment);
            if usable <= 0.0 {
                break;
            }
            let t = usable / segment;
            let end = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
            let steps = (usable * 1.5).max(2.0) as u32;
            for s in 0..=steps {
                let f = s as f32 / steps as f32;
                let px = a[0] + (end[0] - a[0]) * f;
                let py = a[1] + (end[1] - a[1]) * f;
                stamp_disc(self, px, py, half, c, additive);
            }
            walked += usable;
            if walked >= limit {
                break;
            }
        }
    }

    fn circle(&mut self, cx: f32, cy: f32, radius: f32, c: Rgba, stroke: Option<f32>) {
        let r = radius;
        let x0 = (cx - r).max(0.0) as u32;
        let y0 = (cy - r).max(0.0) as u32;
        let x1 = (cx + r).min(self.width as f32 - 1.0) as u32;
        let y1 = (cy + r).min(self.height as f32 - 1.0) as u32;
        let inner = stroke.map(|s| (r - s).max(0.0));
        for py in y0..=y1 {
            for px in x0..=x1 {
                let dx = px as f32 - cx;
                let dy = py as f32 - cy;
                let d = (dx * dx + dy * dy).sqrt();
                if d <= r && inner.is_none_or(|ri| d >= ri) {
                    self.blend(px, py, c);
                }
            }
        }
    }

    fn line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, thickness: f32, c: Rgba) {
        let dx = x1 - x0;
        let dy = y1 - y0;
        let len = (dx * dx + dy * dy).max(0.001);
        let steps = (len * 2.0) as u32;
        let half = thickness / 2.0;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let px = x0 + dx * t;
            let py = y0 + dy * t;
            self.circle(px, py, half, c, None);
        }
    }

    fn glow(&mut self, cx: f32, cy: f32, radius: f32, c: Rgba) {
        let x0 = (cx - radius).max(0.0) as u32;
        let y0 = (cy - radius).max(0.0) as u32;
        let x1 = (cx + radius).min(self.width as f32 - 1.0) as u32;
        let y1 = (cy + radius).min(self.height as f32 - 1.0) as u32;
        for py in y0..=y1 {
            for px in x0..=x1 {
                let dx = px as f32 - cx;
                let dy = py as f32 - cy;
                let d = (dx * dx + dy * dy).sqrt() / radius;
                if d >= 1.0 {
                    continue;
                }
                let falloff = (1.0 - d) * (1.0 - d);
                let a = c.0[3] as f32 * falloff / 255.0;
                add(self, px, py, Rgba([c.0[0], c.0[1], c.0[2], (a * 255.0) as u8]));
            }
        }
    }

    fn draw_bitmap(&mut self, bitmap: &Bitmap, x: f32, y: f32, w: f32, h: f32, fit: Fit, alpha: f32, radius: f32) {
        if w <= 0.0 || h <= 0.0 || bitmap.width == 0 || bitmap.height == 0 {
            return;
        }
        let (src_w, src_h) = (bitmap.width as f32, bitmap.height as f32);
        // Scale factors between the source and the destination box.
        let (sx, sy) = match fit {
            Fit::Stretch => (src_w / w, src_h / h),
            Fit::Contain | Fit::Cover => {
                let scale = if fit == Fit::Contain { (w / src_w).min(h / src_h) } else { (w / src_w).max(h / src_h) };
                let drawn_w = src_w * scale;
                let drawn_h = src_h * scale;
                // Undrawn area is centered; for cover the overflow is cropped.
                (src_w / drawn_w, src_h / drawn_h)
            }
        };
        let drawn_w = src_w / sx;
        let drawn_h = src_h / sy;
        let ox = x + (w - drawn_w) / 2.0;
        let oy = y + (h - drawn_h) / 2.0;

        let clip_x0 = x.max(0.0) as u32;
        let clip_y0 = y.max(0.0) as u32;
        let clip_x1 = (x + w).min(self.width as f32) as u32;
        let clip_y1 = (y + h).min(self.height as f32) as u32;
        let r = radius.min(w / 2.0).min(h / 2.0);
        for py in clip_y0..clip_y1 {
            for px in clip_x0..clip_x1 {
                if r > 0.01 {
                    let fx = px as f32 + 0.5;
                    let fy = py as f32 + 0.5;
                    let cx = fx.clamp(x + r, x + w - r);
                    let cy = fy.clamp(y + r, y + h - r);
                    let dx = fx - cx;
                    let dy = fy - cy;
                    if dx * dx + dy * dy > r * r {
                        continue;
                    }
                }
                let u = (px as f32 + 0.5 - ox) / drawn_w;
                let v = (py as f32 + 0.5 - oy) / drawn_h;
                if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                    continue;
                }
                let texel = bitmap.sample(u * (src_w - 1.0), v * (src_h - 1.0));
                let a = (texel[3] / 255.0) * alpha;
                if a <= 0.001 {
                    continue;
                }
                self.blend(px, py, Rgba([texel[0] as u8, texel[1] as u8, texel[2] as u8, (a * 255.0) as u8]));
            }
        }
    }

    fn starfield(&mut self, count: u32, tint: Rgba, drift_px_per_sec: f32, twinkle: f32, seconds: f32) {
        for i in 0..count {
            let h = hash(i);
            let base_x = (h & 0xffff) as f32 / 65535.0;
            let base_y = ((h >> 16) & 0xffff) as f32 / 65535.0;
            let phase = ((h >> 3) & 255) as f32 / 255.0 * std::f32::consts::PI * 2.0;
            let speed = 0.4 + ((h >> 11) & 63) as f32 / 63.0 * 1.4;
            let size = 1.0 + ((h >> 20) & 3) as f32;
            let x = (base_x * self.width as f32 + drift_px_per_sec * seconds).rem_euclid(self.width as f32);
            let y = base_y * self.height as f32;
            let brightness = (1.0 - twinkle + twinkle * (0.5 + 0.5 * (seconds * speed + phase).sin())).clamp(0.0, 1.0);
            let c = Rgba([tint.0[0], tint.0[1], tint.0[2], (brightness * 230.0) as u8]);
            if size > 2.0 {
                self.glow(x, y, size * 3.0, c.with_alpha(brightness * 60.0));
            }
            self.rect(x, y, size, size, c);
        }
    }

    fn apply_vignette(&mut self, factor: &[u8]) {
        let mut i = 0;
        for y in 0..self.height {
            for x in 0..self.width {
                let v = factor[i] as u32;
                if v != 0 {
                    let p = i * 4;
                    self.pixels[p] = (self.pixels[p] as u32 * (255 - v) / 255) as u8;
                    self.pixels[p + 1] = (self.pixels[p + 1] as u32 * (255 - v) / 255) as u8;
                    self.pixels[p + 2] = (self.pixels[p + 2] as u32 * (255 - v) / 255) as u8;
                    let _ = (x, y);
                }
                i += 1;
            }
        }
    }
}

/// Euclidean distance between two points.
#[must_use]
pub fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    (dx * dx + dy * dy).sqrt()
}

/// Resamples a polyline as a Catmull-Rom curve, so a handful of control points
/// becomes a drawn line. `segments` is the resolution between two points.
#[must_use]
pub fn smooth_polyline(points: &[[f32; 2]], closed: bool, segments: u32) -> Vec<[f32; 2]> {
    if points.len() < 2 {
        return points.to_vec();
    }
    let count = points.len();
    let at = |i: isize| -> [f32; 2] {
        let index =
            if closed { ((i % count as isize) + count as isize) % count as isize } else { i.clamp(0, count as isize - 1) };
        points[index as usize]
    };
    let mut out = Vec::with_capacity(count * segments as usize + 1);
    let last = if closed { count as isize } else { count as isize - 1 };
    for i in 0..last {
        let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
        for s in 0..segments {
            let t = s as f32 / segments as f32;
            let t2 = t * t;
            let t3 = t2 * t;
            let mut point = [0.0f32; 2];
            for k in 0..2 {
                point[k] = 0.5
                    * ((2.0 * p1[k])
                        + (-p0[k] + p2[k]) * t
                        + (2.0 * p0[k] - 5.0 * p1[k] + 4.0 * p2[k] - p3[k]) * t2
                        + (-p0[k] + 3.0 * p1[k] - 3.0 * p2[k] + p3[k]) * t3);
            }
            out.push(point);
        }
    }
    if closed {
        out.push(out[0]);
    } else {
        out.push(points[count - 1]);
    }
    out
}

/// Builds the vignette factor map once per resolution.
#[must_use]
pub fn build_vignette(width: u32, height: u32, strength: f32) -> Vec<u8> {
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;
    let max_r = (cx * cx + cy * cy).sqrt();
    let mut map = vec![0u8; (width as usize) * (height as usize)];
    let mut i = 0;
    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let t = (dx * dx + dy * dy).sqrt() / max_r;
            let v = (t - 0.45).max(0.0) / 0.55 * strength;
            map[i] = (v.clamp(0.0, 1.0) * 180.0) as u8;
            i += 1;
        }
    }
    map
}
