//! Vector drawing: a document is a size plus layers of paths and shapes.
//!
//! The semantics were absorbed from the OpenPaint core (Apache-2.0,
//! © OpenPaint Authors) with the values preserved. Two corrections are
//! deliberate and contract-level:
//! - a malformed hex colour is an explicit [`media_types::Error::invalid_input`]
//!   instead of the binding layer's silent drop of the paint;
//! - both export paths are driven by one [`norm::NormDoc`], so the rounded-rect
//!   stroke, closure handling, transparency, layer order and hidden layers can
//!   no longer disagree between the raster preview and the SVG artifact.
//!
//! Ownership boundaries:
//! - this crate owns the document model ([`doc`]), the host input contract
//!   ([`input`]) and the normalized drawing representation ([`norm`]);
//! - path math (`smooth_polyline`, `rounded_rect_outline`) lives in
//!   `geometry.types`, the shared geometric truth both encoders read;
//! - painting pixels lives in `media.raster` and is driven from `NormDoc` by
//!   the `media.render` `vector-export` entry; nothing here touches pixels;
//! - SVG encoding lives in `media.codec` (its `svg` feature consumes `NormDoc`
//!   one-way; this crate never imports the codec).
//!
//! ```rust
//! use geometry::Rgba;
//! use media_vector::{NormGeometry, Shape, VectorDoc};
//!
//! let mut doc = VectorDoc::new(128, 96, Rgba([12, 16, 30, 255]));
//! doc.add_layer("ink");
//! doc.add_shape(0, Shape::circle([64.0, 48.0], 20.0).filled(Rgba([255, 200, 120, 255])));
//! let norm = doc.normalize();
//! let NormGeometry::Disc { center, radius } = norm.layers[0].shapes[0].geometry else {
//!     panic!("a circle normalizes to a disc");
//! };
//! assert_eq!((center, radius), ([64.0, 48.0], 20.0));
//! ```

pub mod doc;
pub mod input;
pub mod norm;

pub use doc::{Shape, VectorDoc, VectorLayer};
pub use input::{DEFAULT_BACKGROUND, DEFAULT_LAYER_NAME, DEFAULT_STROKE_WIDTH, LayerSpec, ShapeSpec, VectorSpec};
pub use norm::{
    MIN_LAYER_OPACITY, NormDoc, NormGeometry, NormLayer, NormShape, ROUNDED_CORNER_SEGMENTS, SMOOTH_SEGMENTS, normalize,
};

impl VectorDoc {
    /// Resolves the document into the shared normalized representation both
    /// the SVG encoder and the raster lowering consume.
    #[must_use]
    pub fn normalize(&self) -> NormDoc {
        norm::normalize(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geometry::Rgba;

    fn sample() -> VectorDoc {
        let mut doc = VectorDoc::new(64, 48, Rgba([10, 12, 20, 255]));
        let _ = doc.add_layer("base");
        doc.add_shape(
            0,
            Shape::rect([4.0, 4.0], [40.0, 24.0], 6.0)
                .filled(Rgba([30, 40, 70, 255]))
                .stroked(Rgba([120, 160, 255, 255]), 2.0),
        );
        doc.add_shape(
            0,
            Shape::path(vec![[8.0, 30.0], [24.0, 12.0], [56.0, 28.0]])
                .smooth()
                .stroked(Rgba([255, 200, 120, 255]), 3.0),
        );
        doc
    }

    #[test]
    fn smooth_paths_expand_with_the_locked_segment_count() {
        let norm = sample().normalize();
        let NormGeometry::Outline { points, closed } = &norm.layers[0].shapes[1].geometry else {
            panic!("a path normalizes to an outline");
        };
        // 3 input points, open: 2 segments x 12 samples plus the true endpoint.
        assert_eq!(points.len(), 2 * SMOOTH_SEGMENTS as usize + 1);
        assert!(!closed);
        assert_eq!(*points, Shape::path(vec![[8.0, 30.0], [24.0, 12.0], [56.0, 28.0]]).smooth().outline());
    }

    #[test]
    fn closed_paths_use_the_flag_not_a_repeated_point() {
        let mut doc = VectorDoc::new(32, 32, Rgba::default());
        let _ = doc.add_layer("ink");
        doc.add_shape(0, Shape::path(vec![[4.0, 4.0], [20.0, 4.0], [12.0, 20.0]]).closed().filled(Rgba([1, 2, 3, 255])));
        let NormGeometry::Outline { points, closed } = &doc.normalize().layers[0].shapes[0].geometry else {
            panic!("outline");
        };
        assert!(closed);
        assert_eq!(points.len(), 3, "the closure is expressed once, as the flag");
        assert_ne!(points.first(), points.last());

        // A smooth closed curve drops the duplicated seam point too.
        doc.add_shape(0, Shape::path(vec![[4.0, 4.0], [20.0, 4.0], [12.0, 20.0]]).smooth().closed());
        let NormGeometry::Outline { points, closed } = &doc.normalize().layers[0].shapes[1].geometry else {
            panic!("outline");
        };
        assert!(closed);
        assert_eq!(points.len(), 3 * SMOOTH_SEGMENTS as usize, "3 wrapped segments x 12, seam point removed");
        assert_ne!(points.first(), points.last());
    }

    #[test]
    fn rect_radius_is_clamped_into_the_shape() {
        let mut doc = VectorDoc::new(32, 32, Rgba::default());
        let _ = doc.add_layer("ink");
        doc.add_shape(0, Shape::rect([0.0, 0.0], [30.0, 10.0], 99.0));
        doc.add_shape(0, Shape::rect([0.0, 0.0], [30.0, 10.0], 0.005));
        doc.add_shape(0, Shape::rect([0.0, 0.0], [30.0, 10.0], -3.0));
        let norm = doc.normalize();
        let NormGeometry::RoundRect { radius, .. } = norm.layers[0].shapes[0].geometry else {
            panic!("rect");
        };
        assert_eq!(radius, 5.0, "clamped to half the short edge");
        let NormGeometry::RoundRect { radius, .. } = norm.layers[0].shapes[1].geometry else {
            panic!("rect");
        };
        assert_eq!(radius, 0.0, "sub-0.01 radii are sharp, matching the raster cutoff");
        let NormGeometry::RoundRect { radius, .. } = norm.layers[0].shapes[2].geometry else {
            panic!("rect");
        };
        assert_eq!(radius, 0.0, "negative radii clamp to zero");
    }

    #[test]
    fn layer_opacity_folds_into_every_paint() {
        let mut doc = VectorDoc::new(32, 32, Rgba::default());
        let _ = doc.add_layer("ink");
        doc.layer_mut(0).expect("layer").opacity = 0.5;
        doc.add_shape(0, Shape::rect([0.0, 0.0], [8.0, 8.0], 0.0).filled(Rgba([255, 0, 0, 255])));
        let norm = doc.normalize();
        let fill = norm.layers[0].shapes[0].fill.expect("paint");
        assert_eq!(fill.0[3], 128, "the layer opacity replaces the paint alpha, as the source raster path did");
        assert_eq!(&fill.0[..3], &[255, 0, 0], "the color channels pass through untouched");
    }

    #[test]
    fn hidden_and_transparent_layers_are_dropped() {
        let mut doc = sample();
        let _ = doc.add_layer("ghost");
        doc.layer_mut(1).expect("layer").visible = false;
        let _ = doc.add_layer("faint");
        doc.layer_mut(2).expect("layer").opacity = MIN_LAYER_OPACITY;
        let norm = doc.normalize();
        assert_eq!(norm.layers.len(), 1, "only the visible base layer survives");
        assert_eq!(norm.layers[0].name, "base");

        // Just above the threshold the layer renders, matching the source's
        // `opacity <= 0.001` cutoff exactly.
        doc.layer_mut(2).expect("layer").opacity = MIN_LAYER_OPACITY + 0.000_1;
        assert_eq!(doc.normalize().layers.len(), 2);
    }

    #[test]
    fn layer_and_shape_order_survive_normalization() {
        let mut doc = VectorDoc::new(8, 8, Rgba([0, 0, 0, 255]));
        let _ = doc.add_layer("first");
        let _ = doc.add_layer("second");
        doc.add_shape(1, Shape::rect([0.0, 0.0], [8.0, 8.0], 0.0).filled(Rgba([255, 0, 0, 255])));
        doc.add_shape(0, Shape::rect([0.0, 0.0], [8.0, 8.0], 0.0).filled(Rgba([0, 255, 0, 255])));
        doc.add_shape(0, Shape::circle([4.0, 4.0], 2.0).filled(Rgba([0, 0, 255, 255])));
        let norm = doc.normalize();
        assert_eq!(norm.layers.iter().map(|l| l.name.as_str()).collect::<Vec<_>>(), vec!["first", "second"]);
        assert_eq!(norm.layers[0].shapes.len(), 2, "shapes keep their append order within the layer");
    }

    #[test]
    fn out_of_range_layer_append_is_dropped_silently() {
        let mut doc = sample();
        doc.add_shape(99, Shape::circle([1.0, 1.0], 1.0));
        assert_eq!(doc.normalize().layers[0].shapes.len(), 2, "the source model dropped these appends without an error");
    }

    #[test]
    fn degenerate_numbers_clamp_instead_of_reaching_encoders() {
        let mut doc = VectorDoc::new(32, 32, Rgba::default());
        let _ = doc.add_layer("ink");
        doc.add_shape(0, Shape::circle([4.0, 4.0], -7.0).stroked(Rgba([9, 9, 9, 255]), -3.0));
        let shape = &doc.normalize().layers[0].shapes[0];
        let NormGeometry::Disc { radius, .. } = shape.geometry else {
            panic!("disc");
        };
        assert_eq!(radius, 0.0);
        assert_eq!(shape.width, 0.0, "a negative stroke width never reaches an encoder");
    }
}
