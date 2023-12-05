//! Geometry facade.
//!
//! This crate only re-exports: value types live in `geometry.types` and motion
//! curves in `geometry.motion`.

pub use geometry_motion as motion;
pub use geometry_types as types;

pub use geometry_motion::{CameraKey, CameraPose, Ease, apply, camera_at, fade};
pub use geometry_types::{
    Affine2, Camera, LinearRgb, Rect, Rgba, Size, Vec2, centroid, distance, lerp, parse, parse_or, point_in_polygon, smooth_polyline,
};
