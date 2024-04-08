//! The authoring shape for node motion and its evaluated result.

use serde::{Deserialize, Serialize};

use crate::track::Track;

/// A path a node travels along.
///
/// The travelled position comes from the node's `progress` track, so timing a
/// follow is authored exactly like any other property instead of with a second
/// clock.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(deny_unknown_fields)]
pub struct Follow {
    /// Control points in normalized frame coordinates.
    pub points: Vec<[f32; 2]>,
    /// Treat the points as a smooth curve instead of a polyline.
    #[serde(default)]
    pub smooth: bool,
    /// Close the path before sampling.
    #[serde(default)]
    pub closed: bool,
}

/// Motion authored on one node.
///
/// Every property is optional: a node that declares nothing keeps the values its
/// item carries. Tracks are in the node's local frames, which is the same clock
/// as the item's `at`/`dur`.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct Animated {
    /// Horizontal position, normalized frame coordinates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x: Option<Track<f32>>,
    /// Vertical position, normalized frame coordinates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub y: Option<Track<f32>>,
    /// Opacity multiplier, 0..=1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity: Option<Track<f32>>,
    /// Item progress (a line drawing itself, a path being traced), 0..=1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<Track<f32>>,
    /// When present, `progress` also drives position along this path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub follow: Option<Follow>,
}

/// One node's motion at one local frame.
///
/// Plain values, no tracks: the renderer evaluates once per node and then paints
/// exactly as it would for a fully static item.
#[derive(Clone, Copy, Debug)]
pub struct Motion {
    pub x: Option<f32>,
    pub y: Option<f32>,
    pub opacity: f32,
    pub progress: Option<f32>,
    /// Position along [`Follow`], when the node follows one.
    pub follow: Option<[f32; 2]>,
}

impl Default for Motion {
    fn default() -> Self {
        Self { x: None, y: None, opacity: 1.0, progress: None, follow: None }
    }
}

impl Animated {
    /// Whether the node declares no motion at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.x.is_none() && self.y.is_none() && self.opacity.is_none() && self.progress.is_none() && self.follow.is_none()
    }

    /// Evaluates every declared property at local `frame`.
    #[must_use]
    pub fn sample(&self, frame: f32) -> Motion {
        let opacity = self.opacity.as_ref().and_then(|track| track.at(frame)).unwrap_or(1.0).clamp(0.0, 1.0);
        let progress = self.progress.as_ref().and_then(|track| track.at(frame));
        let follow = match (&self.follow, progress) {
            (Some(follow), Some(progress)) => follow.point_at(progress),
            _ => None,
        };
        Motion {
            x: self.x.as_ref().and_then(|track| track.at(frame)),
            y: self.y.as_ref().and_then(|track| track.at(frame)),
            opacity,
            progress,
            follow,
        }
    }

    /// Reports motion that cannot be evaluated, with the node it belongs to.
    pub fn validate(&self, where_: &str, problems: &mut Vec<String>) {
        for (name, track) in [
            ("x", self.x.as_ref()),
            ("y", self.y.as_ref()),
            ("opacity", self.opacity.as_ref()),
            ("progress", self.progress.as_ref()),
        ] {
            if let Some(track) = track {
                if track.keys.is_empty() {
                    problems.push(format!("{where_}: `{name}` track has no keys"));
                }
                if !track.is_sorted() {
                    problems.push(format!("{where_}: `{name}` track keys must be in ascending order"));
                }
            }
        }
        if let Some(follow) = &self.follow {
            if follow.points.len() < 2 {
                problems.push(format!("{where_}: a follow path needs at least two points"));
            }
            if self.progress.is_none() {
                problems.push(format!("{where_}: following a path needs a `progress` track to time it"));
            }
        }
    }
}

impl Follow {
    /// The point at normalized `progress` along the path.
    #[must_use]
    pub fn point_at(&self, progress: f32) -> Option<[f32; 2]> {
        let points = if self.smooth {
            geometry::smooth_polyline(&self.points, self.closed, 12)
        } else if self.closed && !self.points.is_empty() {
            let mut points = self.points.clone();
            points.push(self.points[0]);
            points
        } else {
            self.points.clone()
        };
        if points.len() < 2 {
            return None;
        }
        // Walk by arc length so progress means distance travelled, not "index
        // into the point list": a path with uneven spacing must not race.
        let mut lengths = Vec::with_capacity(points.len() - 1);
        let mut total = 0.0f32;
        for pair in points.windows(2) {
            let dx = pair[1][0] - pair[0][0];
            let dy = pair[1][1] - pair[0][1];
            let length = (dx * dx + dy * dy).sqrt();
            lengths.push(length);
            total += length;
        }
        if total <= f32::EPSILON {
            return Some(points[0]);
        }
        let mut remaining = progress.clamp(0.0, 1.0) * total;
        for (index, length) in lengths.iter().enumerate() {
            if remaining <= *length {
                let t = if *length > 0.0 { remaining / length } else { 0.0 };
                let a = points[index];
                let b = points[index + 1];
                return Some([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
            }
            remaining -= length;
        }
        points.last().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Key, Track};

    #[test]
    fn sampling_reads_each_track_at_the_same_frame() {
        let animated = Animated {
            x: Some(Track::new(vec![Key::linear(0, 0.0), Key::linear(10, 1.0)])),
            opacity: Some(Track::new(vec![Key::linear(0, 1.0), Key::linear(10, 0.0)])),
            ..Animated::default()
        };
        let motion = animated.sample(5.0);
        assert!((motion.x.unwrap() - 0.5).abs() < 1e-6);
        assert!((motion.opacity - 0.5).abs() < 1e-6);
        assert_eq!(motion.y, None);
    }

    #[test]
    fn follow_uses_arc_length_not_point_index() {
        // The second segment is nine times longer than the first.
        let follow = Follow { points: vec![[0.0, 0.0], [0.1, 0.0], [1.0, 0.0]], smooth: false, closed: false };
        let quarter = follow.point_at(0.25).unwrap();
        assert!((quarter[0] - 0.25).abs() < 1e-5, "got {quarter:?}");
        assert_eq!(follow.point_at(0.0), Some([0.0, 0.0]));
        assert_eq!(follow.point_at(1.0), Some([1.0, 0.0]));
    }

    #[test]
    fn follow_without_progress_is_reported() {
        let animated = Animated {
            follow: Some(Follow { points: vec![[0.0, 0.0], [1.0, 1.0]], smooth: false, closed: false }),
            ..Animated::default()
        };
        let mut problems = Vec::new();
        animated.validate("node a", &mut problems);
        assert_eq!(problems.len(), 1);
        assert!(problems[0].contains("progress"));
    }
}
