//! Rational time base: one type derives frame, fps and seconds.

/// Frames per second as an exact rational `numerator / denominator`.
///
/// A single time base replaces the parallel bare values (frame index, fps,
/// seconds) that could drift apart: seconds and sample offsets are always
/// derived from frames through this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TimeBase {
    numerator: u32,
    denominator: u32,
}

impl TimeBase {
    /// Whole-number fps, e.g. 24, 30, 60.
    #[must_use]
    pub const fn from_fps(fps: u32) -> Self {
        Self { numerator: fps, denominator: 1 }
    }

    /// Exact rational fps, e.g. NTSC `30000 / 1001`.
    #[must_use]
    pub const fn rational(numerator: u32, denominator: u32) -> Option<Self> {
        if numerator == 0 || denominator == 0 {
            None
        } else {
            Some(Self { numerator, denominator })
        }
    }

    #[must_use]
    pub const fn numerator(&self) -> u32 {
        self.numerator
    }

    #[must_use]
    pub const fn denominator(&self) -> u32 {
        self.denominator
    }

    /// Frames per second as a float.
    #[must_use]
    pub fn fps(&self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }

    /// Seconds covered by `frames`.
    #[must_use]
    pub fn to_seconds(&self, frames: u32) -> f64 {
        frames as f64 * self.denominator as f64 / self.numerator as f64
    }

    /// Sample index where frame `frames` begins at the given rate.
    #[must_use]
    pub fn to_samples(&self, frames: u32, sample_rate: u32) -> usize {
        (frames as u64 * sample_rate as u64 * self.denominator as u64 / self.numerator as u64) as usize
    }

    /// Sample count covering `frames` at the given rate, rounded up so the
    /// bed never truncates a partial frame.
    #[must_use]
    pub fn to_samples_ceil(&self, frames: u32, sample_rate: u32) -> usize {
        let scaled = frames as u64 * sample_rate as u64 * self.denominator as u64;
        ((scaled + self.numerator as u64 - 1) / self.numerator as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whole_fps_derives_seconds_and_samples() {
        let tb = TimeBase::from_fps(15);
        assert_eq!(tb.fps(), 15.0);
        assert_eq!(tb.to_seconds(30), 2.0);
        assert_eq!(tb.to_samples(5, 44_100), 14_700);
        assert_eq!(tb.to_samples_ceil(30, 44_100), 88_200);
    }

    #[test]
    fn rational_fps_is_exact() {
        let tb = TimeBase::rational(30_000, 1_001).expect("valid");
        assert_eq!(tb.to_samples(30, 44_100), 44_144);
        assert_eq!(tb.to_samples_ceil(30, 44_100), 44_145);
        assert!(TimeBase::rational(0, 1).is_none());
    }
}
