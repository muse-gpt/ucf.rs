//! Easing curves and the shared fade envelope.

use serde::{Deserialize, Serialize};

/// Named easing curves. The set is deliberately small: one curve per intent.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Ease {
    #[default]
    Linear,
    InOutCubic,
    OutCubic,
    OutQuint,
    InOutSine,
    OutExpo,
    InExpo,
    InOutQuart,
    OutBack,
}

/// Maps a normalized progress 0..=1 through the named curve.
#[must_use]
pub fn apply(ease: Ease, t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    match ease {
        Ease::Linear => t,
        Ease::InOutCubic => {
            if t < 0.5 {
                4.0 * t * t * t
            } else {
                1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
            }
        }
        Ease::OutCubic => 1.0 - (1.0 - t).powi(3),
        Ease::OutQuint => 1.0 - (1.0 - t).powi(5),
        Ease::InOutSine => (1.0 - (std::f32::consts::PI * t).cos()) / 2.0,
        // Fast take-off, long settle: reveals and camera pushes.
        Ease::OutExpo => {
            if t >= 1.0 {
                1.0
            } else {
                1.0 - (-10.0 * t).exp2()
            }
        }
        Ease::InExpo => {
            if t <= 0.0 {
                0.0
            } else {
                (-10.0 * (1.0 - t)).exp2()
            }
        }
        // Symmetric and calm: reads as a deliberate move rather than a slide.
        Ease::InOutQuart => {
            if t < 0.5 {
                8.0 * t * t * t * t
            } else {
                1.0 - (-2.0 * t + 2.0).powi(4) / 2.0
            }
        }
        // Overshoot is a personality, not a default: playful objects only.
        Ease::OutBack => {
            const C1: f32 = 1.701_58;
            const C3: f32 = C1 + 1.0;
            1.0 + C3 * (t - 1.0).powi(3) + C1 * (t - 1.0).powi(2)
        }
    }
}

/// Alpha envelope for an element living `dur` frames with fades at both ends.
#[must_use]
pub fn fade(local: u32, dur: u32, fade_in: u32, fade_out: u32) -> f32 {
    if dur == 0 {
        return 0.0;
    }
    let fade_in = fade_in.min(dur / 2 + 1);
    let fade_out = fade_out.min(dur / 2 + 1);
    if local < fade_in {
        apply(Ease::OutCubic, local as f32 / (fade_in.max(1)) as f32)
    } else if local + fade_out > dur {
        let remain = dur - local;
        apply(Ease::OutCubic, remain as f32 / (fade_out.max(1)) as f32)
    } else {
        1.0
    }
}
