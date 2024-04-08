//! Keyframed property tracks.
//!
//! A track owns one property over time: its keys, the easing between them and
//! what happens outside the keyed range. Evaluation never mutates the track, so
//! seek, loop and offline sampling are all the same call.

use serde::{Deserialize, Serialize};

use crate::Ease;

/// A value that can be interpolated between two keys.
pub trait Lerp: Clone {
    /// Interpolates from `a` at `t = 0` to `b` at `t = 1`.
    fn lerp(a: &Self, b: &Self, t: f32) -> Self;
}

impl Lerp for f32 {
    fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        a + (b - a) * t
    }
}

impl Lerp for f64 {
    fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        a + (b - a) * f64::from(t)
    }
}

impl Lerp for [f32; 2] {
    fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        [f32::lerp(&a[0], &b[0], t), f32::lerp(&a[1], &b[1], t)]
    }
}

impl Lerp for [f32; 3] {
    fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        [f32::lerp(&a[0], &b[0], t), f32::lerp(&a[1], &b[1], t), f32::lerp(&a[2], &b[2], t)]
    }
}

impl Lerp for [f32; 4] {
    fn lerp(a: &Self, b: &Self, t: f32) -> Self {
        [
            f32::lerp(&a[0], &b[0], t),
            f32::lerp(&a[1], &b[1], t),
            f32::lerp(&a[2], &b[2], t),
            f32::lerp(&a[3], &b[3], t),
        ]
    }
}

/// One key: a value at a local frame, with the curve that leads into it.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Key<T> {
    /// Local frame this key lands on.
    pub at: u32,
    pub value: T,
    /// Curve used from the previous key to this one.
    #[serde(default)]
    pub ease: Ease,
}

impl<T> Key<T> {
    /// Builds a key on the default (`linear`) curve.
    pub fn linear(at: u32, value: T) -> Self {
        Self { at, value, ease: Ease::Linear }
    }

    /// Builds a key with an explicit curve.
    pub fn eased(at: u32, value: T, ease: Ease) -> Self {
        Self { at, value, ease }
    }
}

/// One property over time.
///
/// Keys must be authored in ascending `at` order; [`Track::is_sorted`] reports
/// a violation so the IR layer can reject it instead of evaluating a track
/// whose meaning depends on read order.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Track<T> {
    pub keys: Vec<Key<T>>,
    /// Hold the first/last value outside the keyed range. `false` means "only
    /// exists between the first and last key".
    #[serde(default = "crate::track::default_clamp")]
    pub clamp: bool,
}

pub(crate) fn default_clamp() -> bool {
    true
}

impl<T> Track<T> {
    /// Builds a clamped track from keys.
    pub fn new(keys: Vec<Key<T>>) -> Self {
        Self { keys, clamp: true }
    }

    /// Whether the keys are strictly ascending.
    #[must_use]
    pub fn is_sorted(&self) -> bool {
        self.keys.windows(2).all(|pair| pair[0].at < pair[1].at)
    }
}

impl<T: Lerp> Track<T> {
    /// The value at local `frame` (fractional frames allowed for sub-frame
    /// sampling).
    ///
    /// Returns `None` for an empty track, or for a frame outside the keyed
    /// range when `clamp` is off.
    #[must_use]
    pub fn at(&self, frame: f32) -> Option<T> {
        let first = self.keys.first()?;
        let last = self.keys.last()?;
        if frame <= first.at as f32 {
            return self.clamp.then(|| first.value.clone());
        }
        if frame >= last.at as f32 {
            return self.clamp.then(|| last.value.clone());
        }
        let mut index = 0usize;
        while index + 1 < self.keys.len() && self.keys[index + 1].at as f32 <= frame {
            index += 1;
        }
        let left = &self.keys[index];
        let right = &self.keys[index + 1];
        let span = (right.at - left.at) as f32;
        if span <= 0.0 {
            return Some(right.value.clone());
        }
        let t = ((frame - left.at as f32) / span).clamp(0.0, 1.0);
        Some(T::lerp(&left.value, &right.value, geometry::apply(right.ease, t)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp() -> Track<f32> {
        Track::new(vec![
            Key::linear(0, 0.0),
            Key::eased(10, 10.0, Ease::Linear),
            Key::linear(20, 0.0),
        ])
    }

    #[test]
    fn interpolates_between_keys_and_holds_the_ends() {
        let track = ramp();
        assert_eq!(track.at(0.0), Some(0.0));
        assert_eq!(track.at(-5.0), Some(0.0));
        assert_eq!(track.at(5.0), Some(5.0));
        assert_eq!(track.at(15.0), Some(5.0));
        assert_eq!(track.at(25.0), Some(0.0));
        // Sub-frame sampling stays continuous.
        let halfway = track.at(2.5).unwrap();
        assert!((halfway - 2.5).abs() < 1e-6);
    }

    #[test]
    fn an_unclamped_track_only_exists_inside_its_keys() {
        let mut track = ramp();
        track.clamp = false;
        assert_eq!(track.at(-1.0), None);
        assert_eq!(track.at(30.0), None);
        assert_eq!(track.at(10.0), Some(10.0));
    }

    #[test]
    fn unsorted_keys_are_reported() {
        let track = Track::new(vec![Key::linear(10, 1.0), Key::linear(0, 0.0)]);
        assert!(!track.is_sorted());
        assert!(ramp().is_sorted());
    }

    #[test]
    fn vector_values_interpolate_per_component() {
        let track = Track::new(vec![Key::linear(0, [0.0, 10.0]), Key::linear(10, [10.0, 0.0])]);
        assert_eq!(track.at(5.0), Some([5.0, 5.0]));
    }
}
