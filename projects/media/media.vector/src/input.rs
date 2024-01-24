//! The host input contract: serde shapes, defaults and validation.
//!
//! These types absorb the OpenPaint N-API's `DocDto`/`LayerDto`/`ShapeDto`
//! defaults and rules — the wire shape is unchanged, but the authoritative
//! behavior moved out of the binding: white background, `"layer"` names,
//! opacity clamped to `0..=1`, stroke width `2.0`, and — the deliberate
//! correction — a malformed hex colour is an explicit `invalid_input` error
//! instead of the binding's silent drop of the paint.
//!
//! Bindings extract host values, deserialize one of these specs and call
//! [`VectorSpec::build`]; they must not re-implement defaults or validation.

use geometry::{Rgba, parse};
use media_types::Error;
use serde::Deserialize;

use crate::doc::{Shape, VectorDoc};

/// Background used when a document omits one. Absorbed from the binding layer.
pub const DEFAULT_BACKGROUND: Rgba = Rgba([255, 255, 255, 255]);
/// Layer name used when a layer omits one.
pub const DEFAULT_LAYER_NAME: &str = "layer";
/// Stroke width used when a shape omits one.
pub const DEFAULT_STROKE_WIDTH: f32 = 2.0;

fn default_background() -> Option<String> {
    None
}

fn default_name() -> String {
    DEFAULT_LAYER_NAME.to_string()
}

fn default_width() -> f32 {
    DEFAULT_STROKE_WIDTH
}

/// A whole document as the host sends it.
#[derive(Deserialize, Debug)]
pub struct VectorSpec {
    pub width: u32,
    pub height: u32,
    #[serde(default = "default_background")]
    pub background: Option<String>,
    #[serde(default)]
    pub layers: Vec<LayerSpec>,
}

impl VectorSpec {
    /// Applies the input contract and builds the domain document.
    ///
    /// A present-but-malformed colour, a non-finite number or an unknown
    /// shape kind is an `invalid_input` error; absent fields take the
    /// documented defaults.
    pub fn build(&self) -> Result<VectorDoc, Error> {
        let background = match self.background.as_deref() {
            Some(text) => parse_color("background", Some(text))?.expect("Some input parses to Some"),
            None => DEFAULT_BACKGROUND,
        };
        let mut doc = VectorDoc::new(self.width, self.height, background);
        for layer in &self.layers {
            layer.validate()?;
            let index = doc.add_layer(layer.name.clone());
            // Safe: add_layer just pushed this index.
            let target = doc.layer_mut(index).expect("layer just added");
            target.opacity = layer.opacity.clamp(0.0, 1.0);
            target.visible = layer.visible;
            for shape in &layer.shapes {
                target.shapes.push(shape.build()?);
            }
        }
        Ok(doc)
    }
}

/// One layer as the host sends it.
#[derive(Deserialize, Debug)]
pub struct LayerSpec {
    #[serde(default = "default_name")]
    pub name: String,
    #[serde(default = "one")]
    pub opacity: f32,
    #[serde(default = "one_bool")]
    pub visible: bool,
    #[serde(default)]
    pub shapes: Vec<ShapeSpec>,
}

impl LayerSpec {
    fn validate(&self) -> Result<(), Error> {
        if !self.opacity.is_finite() {
            return Err(Error::invalid_input(format!("layer opacity {} must be finite", self.opacity)));
        }
        Ok(())
    }
}

fn one() -> f32 {
    1.0
}

fn one_bool() -> bool {
    true
}

/// One shape as the host sends it. The wire discriminant (`kind`, kebab-case)
/// matches the historical binding DTO exactly.
#[derive(Deserialize, Debug)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ShapeSpec {
    Path {
        points: Vec<[f32; 2]>,
        #[serde(default)]
        smooth: bool,
        #[serde(default)]
        closed: bool,
        #[serde(default)]
        fill: Option<String>,
        #[serde(default)]
        stroke: Option<String>,
        #[serde(default = "default_width")]
        width: f32,
    },
    Rect {
        origin: [f32; 2],
        size: [f32; 2],
        #[serde(default)]
        radius: f32,
        #[serde(default)]
        fill: Option<String>,
        #[serde(default)]
        stroke: Option<String>,
        #[serde(default = "default_width")]
        width: f32,
    },
    Circle {
        center: [f32; 2],
        radius: f32,
        #[serde(default)]
        fill: Option<String>,
        #[serde(default)]
        stroke: Option<String>,
        #[serde(default = "default_width")]
        width: f32,
    },
}

impl ShapeSpec {
    fn build(&self) -> Result<Shape, Error> {
        match self {
            Self::Path { points, smooth, closed, fill, stroke, width } => {
                check_points("points", points)?;
                check_finite("width", *width)?;
                let mut shape = Shape::path(points.clone());
                if *smooth {
                    shape = shape.smooth();
                }
                if *closed {
                    shape = shape.closed();
                }
                if let Some(color) = parse_color("fill", fill.as_deref())? {
                    shape = shape.filled(color);
                }
                if let Some(color) = parse_color("stroke", stroke.as_deref())? {
                    shape = shape.stroked(color, *width);
                }
                Ok(shape)
            }
            Self::Rect { origin, size, radius, fill, stroke, width } => {
                check_point("origin", origin)?;
                check_point("size", size)?;
                check_finite("radius", *radius)?;
                check_finite("width", *width)?;
                let mut shape = Shape::rect(*origin, *size, *radius);
                if let Some(color) = parse_color("fill", fill.as_deref())? {
                    shape = shape.filled(color);
                }
                if let Some(color) = parse_color("stroke", stroke.as_deref())? {
                    shape = shape.stroked(color, *width);
                }
                Ok(shape)
            }
            Self::Circle { center, radius, fill, stroke, width } => {
                check_point("center", center)?;
                check_finite("radius", *radius)?;
                check_finite("width", *width)?;
                let mut shape = Shape::circle(*center, *radius);
                if let Some(color) = parse_color("fill", fill.as_deref())? {
                    shape = shape.filled(color);
                }
                if let Some(color) = parse_color("stroke", stroke.as_deref())? {
                    shape = shape.stroked(color, *width);
                }
                Ok(shape)
            }
        }
    }
}

/// Parses a paint colour, rejecting malformed input explicitly. `None` means
/// "no paint"; a present string that does not parse is an error, replacing
/// the binding layer's silent drop.
fn parse_color(field: &str, input: Option<&str>) -> Result<Option<Rgba>, Error> {
    match input {
        None => Ok(None),
        Some(text) => parse(text)
            .map(Some)
            .ok_or_else(|| Error::invalid_input(format!("{field}: invalid color '{text}'"))),
    }
}

fn check_finite(field: &str, value: f32) -> Result<(), Error> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Error::invalid_input(format!("{field}: {value} must be finite")))
    }
}

fn check_point(field: &str, point: &[f32; 2]) -> Result<(), Error> {
    check_finite(field, point[0])?;
    check_finite(field, point[1])
}

fn check_points(field: &str, points: &[[f32; 2]]) -> Result<(), Error> {
    for point in points {
        check_point(field, point)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::doc::Shape;

    fn spec(json: &str) -> VectorSpec {
        serde_json::from_str(json).expect("input parses")
    }

    #[test]
    fn minimal_document_takes_the_absorbed_defaults() {
        let doc = spec(r#"{"width":10,"height":20,"layers":[{}]}"#).build().expect("build");
        assert_eq!((doc.width, doc.height), (10, 20));
        assert_eq!(doc.background, DEFAULT_BACKGROUND, "an absent background is white, as the binding made it");
        let layer = &doc.layers[0];
        assert_eq!(layer.name, DEFAULT_LAYER_NAME);
        assert_eq!(layer.opacity, 1.0);
        assert!(layer.visible);
        assert!(layer.shapes.is_empty());
    }

    #[test]
    fn zero_canvas_clamps_to_one() {
        let doc = spec(r#"{"width":0,"height":0}"#).build().expect("build");
        assert_eq!((doc.width, doc.height), (1, 1), "the source model's max(1) rule");
    }

    #[test]
    fn wire_shape_matches_the_historical_binding_dto() {
        let json = r##"{
            "width": 64, "height": 48, "background": "#10141e",
            "layers": [{
                "name": "base", "opacity": 0.8, "visible": true,
                "shapes": [
                    {"kind": "path", "points": [[8, 30], [24, 12]], "smooth": true, "closed": false,
                     "fill": null, "stroke": "#ffc878", "width": 3.0},
                    {"kind": "rect", "origin": [4, 4], "size": [40, 24], "radius": 6,
                     "fill": "#222846", "stroke": null},
                    {"kind": "circle", "center": [32, 24], "radius": 10,
                     "fill": null, "stroke": "#78a0ff", "width": 2.0}
                ]
            }]
        }"##;
        let doc = spec(json).build().expect("build");
        assert_eq!(doc.background, Rgba([16, 20, 30, 255]));
        let layer = &doc.layers[0];
        assert_eq!(layer.name, "base");
        assert_eq!(layer.opacity, 0.8);
        assert_eq!(layer.shapes.len(), 3);
        let Shape::Path { points, smooth, stroke, width, .. } = &layer.shapes[0] else {
            panic!("path");
        };
        assert_eq!(points, &vec![[8.0, 30.0], [24.0, 12.0]]);
        assert!(smooth);
        assert_eq!(*stroke, Some(Rgba([255, 200, 120, 255])));
        assert_eq!(*width, 3.0);
        let Shape::Rect { radius, fill, .. } = &layer.shapes[1] else {
            panic!("rect");
        };
        assert_eq!(*radius, 6.0);
        assert_eq!(*fill, Some(Rgba([34, 40, 70, 255])));
    }

    #[test]
    fn shape_defaults_apply_per_field() {
        let doc = spec(r#"{"width":8,"height":8,"layers":[{"shapes":[
            {"kind": "path", "points": [[0, 0], [4, 4]]},
            {"kind": "rect", "origin": [0, 0], "size": [4, 4]},
            {"kind": "circle", "center": [2, 2], "radius": 1.0}
        ]}]}"#)
            .build()
            .expect("build");
        for shape in &doc.layers[0].shapes {
            match shape {
                Shape::Path { smooth, closed, fill, stroke, width, .. } => {
                    assert!(!smooth && !closed);
                    assert!(fill.is_none() && stroke.is_none());
                    assert_eq!(*width, DEFAULT_STROKE_WIDTH);
                }
                Shape::Rect { radius, fill, stroke, width, .. } => {
                    assert_eq!(*radius, 0.0);
                    assert!(fill.is_none() && stroke.is_none());
                    assert_eq!(*width, DEFAULT_STROKE_WIDTH);
                }
                Shape::Circle { fill, stroke, width, .. } => {
                    assert!(fill.is_none() && stroke.is_none());
                    assert_eq!(*width, DEFAULT_STROKE_WIDTH);
                }
            }
        }
    }

    #[test]
    fn opacity_is_clamped_into_the_valid_range() {
        let doc = spec(r#"{"width":8,"height":8,"layers":[
            {"opacity": 2.5}, {"opacity": -1.0}
        ]}"#)
            .build()
            .expect("build");
        assert_eq!(doc.layers[0].opacity, 1.0);
        assert_eq!(doc.layers[1].opacity, 0.0);
    }

    #[test]
    fn malformed_colors_are_explicit_errors() {
        // The deliberate correction: the binding used to drop the paint in
        // silence; every bad hex now names itself.
        for (json, field) in [
            (r#"{"width":8,"height":8,"background":"nope"}"#, "background"),
            (r##"{"width":8,"height":8,"layers":[{"shapes":[
                {"kind":"circle","center":[2,2],"radius":1,"fill":"#zzzzzz"}]}]}"##, "fill"),
            (r##"{"width":8,"height":8,"layers":[{"shapes":[
                {"kind":"rect","origin":[0,0],"size":[4,4],"stroke":"#fff0"}]}]}"##, "stroke"),
        ] {
            let error = spec(json).build().expect_err("bad color rejects");
            assert_eq!(error.code(), "invalid_input", "{json}");
            assert!(error.message.contains(field), "{} missing {field}: {error}", error.message);
        }
    }

    #[test]
    fn non_finite_numbers_are_explicit_errors() {
        // Out-of-range JSON literals are rejected at deserialization; they
        // never become a silently clamped document.
        assert!(serde_json::from_str::<VectorSpec>(r#"{"width":8,"height":8,"layers":[{"opacity":1e999}]}"#).is_err());

        // Values a host converts into a non-finite f32 are caught by the
        // build contract before any shape exists.
        let shape = ShapeSpec::Circle { center: [2.0, f32::INFINITY], radius: 1.0, fill: None, stroke: None, width: 2.0 };
        assert_eq!(shape.build().expect_err("infinite coordinate rejects").code(), "invalid_input");

        let shape = ShapeSpec::Path {
            points: vec![[0.0, 0.0], [f32::NAN, 1.0]],
            smooth: false,
            closed: false,
            fill: None,
            stroke: None,
            width: 2.0,
        };
        assert_eq!(shape.build().expect_err("NaN point rejects").code(), "invalid_input");

        let spec = VectorSpec {
            width: 8,
            height: 8,
            background: None,
            layers: vec![LayerSpec { name: "l".to_string(), opacity: f32::NAN, visible: true, shapes: Vec::new() }],
        };
        assert_eq!(spec.build().expect_err("NaN opacity rejects").code(), "invalid_input");
    }
}
