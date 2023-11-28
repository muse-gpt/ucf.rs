//! Geometry value types shared by every media crate.
//!
//! Nothing here rasterizes, allocates frames or talks to a device: this crate
//! owns the meaning of points, colors, transforms and cameras so that scene
//! data, raster backends and downstream products agree on the same math.

pub mod color;
pub mod geom;
pub mod path;
pub mod transform;

pub use color::{Rgba, parse, parse_or};
pub use geom::{Rect, Size, Vec2, distance, lerp};
pub use path::{centroid, point_in_polygon, smooth_polyline};
pub use transform::{Affine2, Camera};
