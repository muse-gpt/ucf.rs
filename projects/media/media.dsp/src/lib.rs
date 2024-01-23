//! Deterministic DSP primitives shared by every sound path.
//!
//! [`Rng`] is the single formal random stream contract for audio in this
//! workspace: xorshift32, seeded per sound, no global state. The same seed
//! always produces the same stream, which is what lets a voice be regenerated
//! byte for byte. Visual crates keep their own stateless integer hash
//! (`media.types::hash`) for per-index grain; that is a different concern and
//! must not be conflated with this stream — audio paths consume `Rng`, never
//! `hash`.
//!
//! The primitives were absorbed from the OpenSFX core (Apache-2.0,
//! © OpenSFX Authors) with every constant preserved; block scheduling (e.g.
//! rebuilding a sweep filter every 64 samples) belongs to the voices in
//! `media.synthesis`, not to this crate.
//!
//! ```rust
//! use media_dsp::Rng;
//!
//! let mut a = Rng::new(7);
//! let mut b = Rng::new(7);
//! assert_eq!(a.next_u32(), b.next_u32());
//! ```

/// xorshift32: the only randomness in the audio domain, seeded per sound.
pub struct Rng(u32);

impl Rng {
    #[must_use]
    pub fn new(seed: u32) -> Self {
        Self(seed.max(1) | 1)
    }

    #[must_use]
    pub fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    /// Uniform in `0..1`.
    #[must_use]
    pub fn unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16_777_216.0
    }

    /// Uniform in `-1..1`.
    #[must_use]
    pub fn bipolar(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }
}

/// A two-pole biquad, enough for the low / high / band shapes a sound needs.
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    fn coefficients(kind: u8, cutoff: f32, q: f32, sample_rate: f32) -> (f32, f32, f32, f32, f32) {
        let w0 = std::f32::consts::TAU * cutoff.clamp(20.0, sample_rate * 0.49) / sample_rate;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q.max(0.1));
        let a0 = 1.0 + alpha;
        match kind {
            // low-pass
            0 => ((1.0 - cos) / 2.0 / a0, (1.0 - cos) / a0, (1.0 - cos) / 2.0 / a0, -2.0 * cos / a0, (1.0 - alpha) / a0),
            // high-pass
            1 => ((1.0 + cos) / 2.0 / a0, -(1.0 + cos) / a0, (1.0 + cos) / 2.0 / a0, -2.0 * cos / a0, (1.0 - alpha) / a0),
            // band-pass, constant skirt gain
            _ => (alpha / a0, 0.0, -alpha / a0, -2.0 * cos / a0, (1.0 - alpha) / a0),
        }
    }

    #[must_use]
    pub fn low_pass(cutoff: f32, q: f32, sample_rate: f32) -> Self {
        Self::from(Self::coefficients(0, cutoff, q, sample_rate))
    }

    #[must_use]
    pub fn high_pass(cutoff: f32, q: f32, sample_rate: f32) -> Self {
        Self::from(Self::coefficients(1, cutoff, q, sample_rate))
    }

    #[must_use]
    pub fn band_pass(center: f32, q: f32, sample_rate: f32) -> Self {
        Self::from(Self::coefficients(2, center, q, sample_rate))
    }

    fn from(coeffs: (f32, f32, f32, f32, f32)) -> Self {
        Self { b0: coeffs.0, b1: coeffs.1, b2: coeffs.2, a1: coeffs.3, a2: coeffs.4, x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0 }
    }

    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Attack / decay envelope in seconds, sampled at `t`.
#[must_use]
pub fn env_ad(t: f32, attack: f32, decay: f32) -> f32 {
    if t < attack {
        if attack <= 0.0 { 1.0 } else { (t / attack).clamp(0.0, 1.0) }
    } else {
        let x = (t - attack) / decay.max(1e-4);
        (-4.5 * x).exp().clamp(0.0, 1.0)
    }
}

/// Exponential pitch glide from `start` to `end` over `duration`.
#[must_use]
pub fn glide(t: f32, duration: f32, start: f32, end: f32) -> f32 {
    let k = (t / duration.max(1e-4)).clamp(0.0, 1.0);
    start * (end / start).powf(k)
}

/// Equal-power pan: returns (left, right) gains for `pan` in -1..1.
#[must_use]
pub fn pan_gains(pan: f32) -> (f32, f32) {
    let p = (pan.clamp(-1.0, 1.0) + 1.0) * 0.25 * std::f32::consts::PI;
    (p.cos(), p.sin())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_zero_and_one_share_one_stream() {
        let mut a = Rng::new(0);
        let mut b = Rng::new(1);
        for _ in 0..64 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn unit_and_bipolar_stay_in_range() {
        let mut rng = Rng::new(1234);
        for _ in 0..1000 {
            let u = rng.unit();
            assert!((0.0..1.0).contains(&u));
            let b = rng.bipolar();
            assert!((-1.0..1.0).contains(&b));
        }
    }

    #[test]
    fn biquad_passes_dc_at_full_band_edge() {
        // A low-pass well above the signal rate must converge to unity gain on
        // a constant input without inventing values.
        let mut lp = Biquad::low_pass(20_000.0, 0.707, 44_100.0);
        let mut y = 0.0f32;
        for _ in 0..4000 {
            y = lp.process(1.0);
        }
        assert!((y - 1.0).abs() < 1e-3, "dc gain {y}");
    }

    #[test]
    fn envelope_and_glide_shapes_match_contract() {
        assert_eq!(env_ad(0.0, 0.0, 0.1), 1.0);
        assert_eq!(env_ad(0.05, 0.1, 0.1), 0.5);
        assert!((env_ad(0.1, 0.1, 0.1) - 1.0).abs() < 1e-6);
        assert!((env_ad(0.2, 0.1, 0.1) - (-4.5f32).exp()).abs() < 1e-6);
        assert_eq!(glide(0.0, 1.0, 100.0, 400.0), 100.0);
        assert!((glide(1.0, 1.0, 100.0, 400.0) - 400.0).abs() < 1e-3);
        assert!((glide(0.5, 1.0, 100.0, 400.0) - 200.0).abs() < 1e-3);
    }

    #[test]
    fn pan_is_equal_power_and_clamped() {
        let (l, r) = pan_gains(0.0);
        assert!((l - r).abs() < 1e-6);
        assert!((l * l + r * r - 1.0).abs() < 1e-6);
        let (l, r) = pan_gains(-5.0);
        assert!((l - 1.0).abs() < 1e-6 && r.abs() < 1e-6);
    }
}
