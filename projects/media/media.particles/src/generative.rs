//! Generative fields: drifting sparks or dust, and volumetric wedges fanning
//! out of a point.
//!
//! Both are stateless per moment: `apply` consumes only the frame plus the
//! `seconds` derived upstream from the single rational time base. Sizes are
//! authored relative to the short side, inheriting the post-processing radius
//! contract; the pixel measures come from the frame itself.

use geometry_types::Rgba;
use media_image::Frame;
use media_types::hash;

/// Short side of the frame (at least 1), the unit both fields are authored
/// against.
fn short_side(frame: &Frame) -> f32 {
    frame.width.min(frame.height).max(1) as f32
}

/// Drifting sparks or dust.
pub struct Particles {
    pub count: u32,
    pub seed: u32,
    pub color: Rgba,
    /// Dot radius as a fraction of the short side.
    pub size: f32,
    /// Vertical travel in frame heights per second.
    pub speed: f32,
    /// Sideways drift as a fraction of the width.
    pub drift: f32,
    /// 0..1, how much the dots fade toward the edge of their life.
    pub twinkle: f32,
}

impl Particles {
    /// Slow upward motes: size 0.0035, speed 0.05, drift 0.02, twinkle 0.6.
    #[must_use]
    pub fn dust(count: u32, seed: u32, color: Rgba) -> Self {
        Self {
            count,
            seed,
            color,
            size: 0.0035,
            speed: 0.05,
            drift: 0.02,
            twinkle: 0.6,
        }
    }

    /// Fast rising embers: size 0.0025, speed -0.18, drift 0.04, twinkle 0.85.
    #[must_use]
    pub fn sparks(count: u32, seed: u32, color: Rgba) -> Self {
        Self {
            count,
            seed,
            color,
            size: 0.0025,
            speed: -0.18,
            drift: 0.04,
            twinkle: 0.85,
        }
    }

    /// Stamps every particle at its position for `seconds`, additively.
    pub fn apply(&self, frame: &mut Frame, seconds: f32) {
        let width = frame.width as f32;
        let height = frame.height as f32;
        let radius = (self.size.clamp(0.0005, 0.05) * short_side(frame)).max(0.8);
        for i in 0..self.count {
            let h = hash(self.seed.wrapping_mul(2_654_435_761).wrapping_add(i));
            let u = (h & 0xffff) as f32 / 65_535.0;
            let v = ((h >> 16) & 0xffff) as f32 / 65_535.0;
            // Each particle has its own phase so they never move as a block.
            let phase = u * 6.28;
            let x = (u * width + (seconds * self.drift + phase).sin() * width * 0.01).rem_euclid(width);
            let travel = seconds * self.speed * (0.4 + v);
            let y = (v * height + travel * height).rem_euclid(height);
            let blink = 0.55
                + 0.45
                    * (seconds * (0.6 + v) * 6.28 + phase).sin()
                    * self.twinkle.clamp(0.0, 1.0);
            stamp(frame, x, y, radius, self.color, blink.clamp(0.0, 1.0));
        }
    }
}

/// Volumetric wedges fanning out of a point.
pub struct Beams {
    pub count: u32,
    pub seed: u32,
    pub color: Rgba,
    /// Source point in normalized coordinates.
    pub origin: [f32; 2],
    /// Overall intensity.
    pub strength: f32,
    /// Sway in degrees.
    pub sway: f32,
}

impl Beams {
    /// Default stage: origin `[0.5, 0.28]`, strength 0.35, sway 6°.
    #[must_use]
    pub fn new(count: u32, seed: u32, color: Rgba) -> Self {
        Self {
            count,
            seed,
            color,
            origin: [0.5, 0.28],
            strength: 0.35,
            sway: 6.0,
        }
    }

    /// Marches every wedge from the origin for `seconds`, additively.
    pub fn apply(&self, frame: &mut Frame, seconds: f32) {
        let width = frame.width as f32;
        let height = frame.height as f32;
        let ox = self.origin[0].clamp(0.0, 1.0) * width;
        let oy = self.origin[1].clamp(0.0, 1.0) * height;
        let length = (width * width + height * height).sqrt();
        let half_width = short_side(frame) * 0.035;
        for i in 0..self.count {
            let h = hash(self.seed.wrapping_add(i).wrapping_mul(2_654_435_761));
            let base = (h & 0xff) as f32 / 255.0 * 360.0;
            let wobble = (seconds * (0.3 + (h >> 8 & 0xff) as f32 / 255.0) * 6.28).sin() * self.sway;
            let angle = (base + wobble).to_radians();
            let (sin, cos) = angle.sin_cos();
            let intensity = 0.25 + ((h >> 16) & 0xff) as f32 / 255.0 * 0.75;
            let steps = (length / 2.0).max(1.0) as u32;
            for step in 0..steps {
                let t = step as f32 / steps as f32;
                let distance = t * length * 0.62;
                let cx = ox + cos * distance;
                let cy = oy + sin * distance;
                let spread = half_width * (0.35 + t * 1.6);
                let falloff = (1.0 - t).powi(2) * self.strength * intensity;
                stamp(frame, cx, cy, spread, self.color, falloff);
            }
        }
    }
}

/// Additive soft dot. RGB only — straight alpha passes through.
fn stamp(frame: &mut Frame, cx: f32, cy: f32, radius: f32, color: Rgba, gain: f32) {
    let radius = radius.max(0.6);
    let x0 = (cx - radius).max(0.0) as u32;
    let y0 = (cy - radius).max(0.0) as u32;
    let x1 = ((cx + radius) as u32).min(frame.width.saturating_sub(1));
    let y1 = ((cy + radius) as u32).min(frame.height.saturating_sub(1));
    for y in y0..=y1 {
        for x in x0..=x1 {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let distance = (dx * dx + dy * dy).sqrt() / radius;
            if distance >= 1.0 {
                continue;
            }
            let falloff = (1.0 - distance) * (1.0 - distance) * gain;
            let index = ((y * frame.width + x) * 4) as usize;
            for k in 0..3 {
                let added = (color.0[k] as f32 * falloff) as u32;
                frame.pixels[index + k] = (frame.pixels[index + k] as u32 + added).min(255) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const AMBER: Rgba = Rgba([255, 196, 120, 255]);

    fn blank() -> Frame {
        let mut frame = Frame::new(96, 54);
        for chunk in frame.pixels.chunks_exact_mut(4) {
            chunk[3] = 255;
        }
        frame
    }

    #[test]
    fn particles_are_deterministic_and_seeded() {
        let mut a = blank();
        Particles::sparks(40, 9, AMBER).apply(&mut a, 0.4);
        let mut b = blank();
        Particles::sparks(40, 9, AMBER).apply(&mut b, 0.4);
        assert_eq!(a.pixels, b.pixels);
        let mut c = blank();
        Particles::sparks(40, 10, AMBER).apply(&mut c, 0.4);
        assert_ne!(a.pixels, c.pixels);
        assert!(!a.pixels.iter().step_by(4).all(|v| *v == 0), "the field paints");
    }

    #[test]
    fn particles_move_between_moments() {
        let mut a = blank();
        let mut b = blank();
        Particles::dust(60, 3, AMBER).apply(&mut a, 0.0);
        Particles::dust(60, 3, AMBER).apply(&mut b, 0.5);
        assert_ne!(a.pixels, b.pixels);
    }

    #[test]
    fn out_of_order_moments_reproduce_the_sequential_field() {
        // The stateless contract: evaluating 11 before 0 (or skipping 7)
        // changes nothing about the pixels each moment owns.
        let field = Particles::sparks(30, 31, AMBER);
        let mut sequential = Vec::new();
        for seconds in [0.0f32, 0.5, 1.0, 1.5] {
            let mut frame = blank();
            field.apply(&mut frame, seconds);
            sequential.push(frame.pixels);
        }
        let mut shuffled = vec![Vec::new(); 4];
        for (index, seconds) in [(3, 1.5), (0, 0.0), (2, 1.0), (1, 0.5)] {
            let mut frame = blank();
            field.apply(&mut frame, seconds);
            shuffled[index] = frame.pixels;
        }
        for (want, got) in sequential.iter().zip(shuffled.iter()) {
            assert_eq!(want, got);
        }
    }

    #[test]
    fn parallel_frame_fetches_match_the_serial_field() {
        let handles: Vec<_> = (0..4u32)
            .map(|f| {
                std::thread::spawn(move || {
                    let beams = Beams::new(5, 11, AMBER);
                    let mut frame = blank();
                    beams.apply(&mut frame, f as f32 / 30.0);
                    frame.pixels
                })
            })
            .collect();
        for (f, handle) in (0..4u32).zip(handles) {
            let mut serial = blank();
            Beams::new(5, 11, AMBER).apply(&mut serial, f as f32 / 30.0);
            assert_eq!(handle.join().expect("thread"), serial.pixels);
        }
    }

    #[test]
    fn beams_sway_with_time_and_keep_alpha() {
        let mut a = blank();
        let mut b = blank();
        Beams::new(7, 11, AMBER).apply(&mut a, 0.0);
        Beams::new(7, 11, AMBER).apply(&mut b, 2.0);
        assert_ne!(a.pixels, b.pixels, "the sway phase moves the wedges");
        for i in (0..a.pixels.len()).step_by(4) {
            assert_eq!(a.pixels[i + 3], 255, "alpha untouched");
        }
    }

    #[test]
    fn dust_and_sparks_carry_their_recipe_constants() {
        let dust = Particles::dust(1, 0, AMBER);
        assert_eq!((dust.size, dust.speed, dust.drift, dust.twinkle), (0.0035, 0.05, 0.02, 0.6));
        let sparks = Particles::sparks(1, 0, AMBER);
        assert_eq!((sparks.size, sparks.speed, sparks.drift, sparks.twinkle), (0.0025, -0.18, 0.04, 0.85));
        let beams = Beams::new(1, 0, AMBER);
        assert_eq!(beams.origin, [0.5, 0.28]);
        assert_eq!((beams.strength, beams.sway), (0.35, 6.0));
    }
}
