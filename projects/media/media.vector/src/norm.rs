//! The normalized drawing representation shared by both export paths.
//!
//! A [`NormDoc`] is the single resolved reading of a [`crate::doc::VectorDoc`]:
//! smoothing expanded, closure expressed as a flag, rounded radii clamped,
//! hidden and effectively transparent layers dropped and layer opacity folded
//! into every paint's alpha. The SVG encoder in `media.codec` and the raster
//! lowering in `media.render` consume this same value, so the two artifacts
//! cannot diverge on geometry, layer order, hidden layers or transparency —
//! the OpenPaint baseline's rounded-rect stroke and dropped-alpha split is
//! structurally impossible here.
//!
//! This crate never paints pixels; producing a `media.image::Frame` from a
//! `NormDoc` is the `vector-export` composition entry's job.

use geometry::{Rgba, Vec2, smooth_polyline};

use crate::doc::{Shape, VectorDoc};

/// Samples per segment when a smooth path is expanded. Locked from the source
/// implementation (`smooth_polyline(points, closed, 12)`); both export paths
/// see the same tessellation because expansion happens once, here.
pub const SMOOTH_SEGMENTS: u32 = 12;

/// Samples per corner arc when a rounded-rect stroke is tessellated for the
/// raster path. The SVG encoder keeps the analytic `<rect rx>` element; the
/// clamp below is what makes the two readings agree.
pub const ROUNDED_CORNER_SEGMENTS: u32 = 12;

/// Layer opacity at or below this threshold drops the whole layer. Absorbed
/// value-for-value from the source implementation, which applied the same
/// cutoff in both the raster loop and the SVG writer.
pub const MIN_LAYER_OPACITY: f32 = 0.001;

/// Geometry of a normalized shape, resolved to one of three forms.
#[derive(Clone, Debug, PartialEq)]
pub enum NormGeometry {
    /// Expanded point list. `closed == true` means the last point connects
    /// back to the first without repeating it: the raster lowering appends
    /// the closure point before stroking, the SVG encoder emits `Z`.
    Outline { points: Vec<Vec2>, closed: bool },
    /// Axis-aligned rectangle; `radius` is already clamped into
    /// `0..=min(w, h) / 2` and is `0` below the source's `0.01` sharpness
    /// cutoff. The rounded-rect stroke is the same rounded boundary in both
    /// products: the raster path tessellates it with
    /// [`ROUNDED_CORNER_SEGMENTS`], the SVG path keeps `rx`.
    RoundRect { origin: Vec2, size: Vec2, radius: f32 },
    /// Analytic circle; `radius` is clamped to be non-negative.
    Disc { center: Vec2, radius: f32 },
}

/// One drawable with resolved geometry and folded paint.
#[derive(Clone, Debug, PartialEq)]
pub struct NormShape {
    pub geometry: NormGeometry,
    /// Fill with the layer opacity already folded into its alpha channel.
    pub fill: Option<Rgba>,
    /// Stroke with the layer opacity already folded into its alpha channel.
    pub stroke: Option<Rgba>,
    /// Stroke width, clamped to be non-negative.
    pub width: f32,
}

/// A rendered layer: hidden and effectively transparent layers are gone, so
/// consumers paint everything they receive.
#[derive(Clone, Debug, PartialEq)]
pub struct NormLayer {
    pub name: String,
    pub shapes: Vec<NormShape>,
}

/// The whole document in its normalized form.
#[derive(Clone, Debug, PartialEq)]
pub struct NormDoc {
    pub width: u32,
    pub height: u32,
    pub background: Rgba,
    pub layers: Vec<NormLayer>,
}

/// Resolves `doc` into the shared representation both export paths consume.
#[must_use]
pub fn normalize(doc: &VectorDoc) -> NormDoc {
    let mut layers = Vec::with_capacity(doc.layers.len());
    for layer in &doc.layers {
        if !layer.visible || layer.opacity <= MIN_LAYER_OPACITY {
            continue;
        }
        let shapes = layer.shapes.iter().map(|shape| norm_shape(shape, layer.opacity)).collect();
        layers.push(NormLayer { name: layer.name.clone(), shapes });
    }
    NormDoc { width: doc.width, height: doc.height, background: doc.background, layers }
}

/// Expands a smooth polyline with the locked segment count.
pub(crate) fn expand_smooth(points: &[Vec2], closed: bool) -> Vec<Vec2> {
    smooth_polyline(points, closed, SMOOTH_SEGMENTS)
}

fn norm_shape(shape: &Shape, opacity: f32) -> NormShape {
    let (geometry, fill, stroke, width) = match shape {
        Shape::Path { points, smooth, closed, fill, stroke, width } => {
            let mut points = if *smooth { expand_smooth(points, *closed) } else { points.clone() };
            if *closed {
                // One closure convention for both paths: no repeated final
                // point. `smooth_polyline` appends one for closed curves.
                if points.len() >= 2 && points.first() == points.last() {
                    points.pop();
                }
            }
            (NormGeometry::Outline { points, closed: *closed }, fill, stroke, width)
        }
        Shape::Rect { origin, size, radius, fill, stroke, width } => {
            let radius = radius.max(0.0).min(size[0] / 2.0).min(size[1] / 2.0);
            // The raster round_rect treats sub-0.01 radii as sharp corners;
            // normalizing to exactly 0 keeps the SVG `rx` on the same shape.
            let radius = if radius <= 0.01 { 0.0 } else { radius };
            (NormGeometry::RoundRect { origin: *origin, size: *size, radius }, fill, stroke, width)
        }
        Shape::Circle { center, radius, fill, stroke, width } => {
            (NormGeometry::Disc { center: *center, radius: radius.max(0.0) }, fill, stroke, width)
        }
    };
    NormShape {
        geometry,
        fill: fill.map(|color| fold(color, opacity)),
        stroke: stroke.map(|color| fold(color, opacity)),
        width: width.max(0.0),
    }
}

/// Folds the layer opacity into the paint, replacing its alpha channel —
/// the source implementation's `with_alpha` semantics, kept so a color's own
/// alpha never double-counts and both paths agree on one number.
fn fold(color: Rgba, opacity: f32) -> Rgba {
    color.with_alpha(opacity.clamp(0.0, 1.0))
}
