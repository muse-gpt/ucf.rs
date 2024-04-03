//! Keyed camera curves.

use serde::{Deserialize, Serialize};

use crate::ease::{Ease, apply};
use geometry_types::Vec2;

/// A camera pose sampled from the curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraPose {
    pub zoom: f32,
    pub focus: Vec2,
    pub rotate: f32,
}

impl Default for CameraPose {
    fn default() -> Self {
        Self { zoom: 1.0, focus: [0.5, 0.5], rotate: 0.0 }
    }
}

impl CameraPose {
    /// The static geometry camera for this pose.
    #[must_use]
    pub fn camera(&self) -> geometry_types::Camera {
        geometry_types::Camera { zoom: self.zoom, focus: self.focus, rotate: self.rotate }
    }
}

/// A key on a camera curve, positioned by absolute frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CameraKey {
    pub at: u32,
    pub zoom: f32,
    #[serde(default = "default_center")]
    pub focus: Vec2,
    #[serde(default)]
    pub rotate: f32,
    /// Easing applied from this key to the next.
    #[serde(default)]
    pub ease: Ease,
}

fn default_center() -> Vec2 {
    [0.5, 0.5]
}

/// Resolves a camera curve at `frame`.
///
/// Keys clamp at both ends and every segment is eased with the outgoing key's
/// curve, so a film can be one continuous move with no cuts.
#[must_use]
pub fn camera_at(keys: &[CameraKey], frame: u32) -> CameraPose {
    match keys.len() {
        0 => CameraPose::default(),
        1 => CameraPose { zoom: keys[0].zoom, focus: keys[0].focus, rotate: keys[0].rotate },
        _ => {
            let first = &keys[0];
            if frame <= first.at {
                return CameraPose { zoom: first.zoom, focus: first.focus, rotate: first.rotate };
            }
            for pair in keys.windows(2) {
                let (a, b) = (&pair[0], &pair[1]);
                if frame <= b.at {
                    let span = b.at.saturating_sub(a.at).max(1) as f32;
                    let t = apply(a.ease, (frame - a.at) as f32 / span);
                    return CameraPose {
                        zoom: a.zoom + (b.zoom - a.zoom) * t,
                        focus: [a.focus[0] + (b.focus[0] - a.focus[0]) * t, a.focus[1] + (b.focus[1] - a.focus[1]) * t],
                        rotate: a.rotate + (b.rotate - a.rotate) * t,
                    };
                }
            }
            let last = &keys[keys.len() - 1];
            CameraPose { zoom: last.zoom, focus: last.focus, rotate: last.rotate }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys() -> Vec<CameraKey> {
        vec![
            CameraKey { at: 0, zoom: 1.0, focus: [0.5, 0.5], rotate: 0.0, ease: Ease::Linear },
            CameraKey { at: 10, zoom: 2.0, focus: [0.5, 0.5], rotate: 0.0, ease: Ease::Linear },
        ]
    }

    #[test]
    fn clamps_outside_the_key_range() {
        let keys = keys();
        assert!((camera_at(&keys, 0).zoom - 1.0).abs() < 1e-6);
        assert!((camera_at(&keys, 99).zoom - 2.0).abs() < 1e-6);
    }

    #[test]
    fn interpolates_between_keys() {
        let mid = camera_at(&keys(), 5);
        assert!((mid.zoom - 1.5).abs() < 1e-5, "{mid:?}");
    }

    #[test]
    fn a_single_key_is_a_still_camera() {
        let one = vec![CameraKey { at: 3, zoom: 1.4, focus: [0.2, 0.3], rotate: 5.0, ease: Ease::OutExpo }];
        let pose = camera_at(&one, 99);
        assert!((pose.zoom - 1.4).abs() < 1e-6);
        assert_eq!(pose.focus, [0.2, 0.3]);
    }
}
