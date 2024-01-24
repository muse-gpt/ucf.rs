//! The vector document model.
//!
//! Shapes carry resolved drawing intent in absolute pixel coordinates; they
//! know nothing about pixels. Raster painting and SVG encoding consume the
//! [`crate::norm`] output of [`VectorDoc::normalize`], composed by
//! `media.render` and `media.codec` respectively.

use geometry::Rgba;

/// One drawing primitive. Coordinates are absolute pixels.
#[derive(Clone, Debug)]
pub enum Shape {
    Path {
        points: Vec<[f32; 2]>,
        smooth: bool,
        closed: bool,
        fill: Option<Rgba>,
        stroke: Option<Rgba>,
        width: f32,
    },
    Rect {
        origin: [f32; 2],
        size: [f32; 2],
        radius: f32,
        fill: Option<Rgba>,
        stroke: Option<Rgba>,
        width: f32,
    },
    Circle {
        center: [f32; 2],
        radius: f32,
        fill: Option<Rgba>,
        stroke: Option<Rgba>,
        width: f32,
    },
}

impl Shape {
    #[must_use]
    pub fn path(points: Vec<[f32; 2]>) -> Self {
        Self::Path {
            points,
            smooth: false,
            closed: false,
            fill: None,
            stroke: None,
            width: 2.0,
        }
    }

    #[must_use]
    pub fn rect(origin: [f32; 2], size: [f32; 2], radius: f32) -> Self {
        Self::Rect {
            origin,
            size,
            radius,
            fill: None,
            stroke: None,
            width: 2.0,
        }
    }

    #[must_use]
    pub fn circle(center: [f32; 2], radius: f32) -> Self {
        Self::Circle {
            center,
            radius,
            fill: None,
            stroke: None,
            width: 2.0,
        }
    }

    #[must_use]
    pub fn smooth(mut self) -> Self {
        if let Self::Path { smooth, .. } = &mut self {
            *smooth = true;
        }
        self
    }

    #[must_use]
    pub fn closed(mut self) -> Self {
        if let Self::Path { closed, .. } = &mut self {
            *closed = true;
        }
        self
    }

    #[must_use]
    pub fn filled(mut self, color: Rgba) -> Self {
        match &mut self {
            Self::Path { fill, .. } | Self::Rect { fill, .. } | Self::Circle { fill, .. } => {
                *fill = Some(color);
            }
        }
        self
    }

    #[must_use]
    pub fn stroked(mut self, color: Rgba, width: f32) -> Self {
        match &mut self {
            Self::Path {
                stroke,
                width: w, ..
            }
            | Self::Rect {
                stroke,
                width: w, ..
            }
            | Self::Circle {
                stroke,
                width: w, ..
            } => {
                *stroke = Some(color);
                *w = width;
            }
        }
        self
    }

    /// Absolute pixel outline of a path shape, with smoothing applied.
    ///
    /// Kept for consumers that only need the resolved point list; the full
    /// normalization contract (closure convention, radius clamp, alpha fold)
    /// lives in [`crate::norm::normalize`].
    #[must_use]
    pub fn outline(&self) -> Vec<[f32; 2]> {
        match self {
            Self::Path {
                points,
                smooth,
                closed,
                ..
            } => {
                let mut outline = if *smooth {
                    crate::norm::expand_smooth(points, *closed)
                } else {
                    points.clone()
                };
                if *closed && !*smooth && !outline.is_empty() {
                    outline.push(outline[0]);
                }
                outline
            }
            Self::Rect { .. } | Self::Circle { .. } => Vec::new(),
        }
    }
}

/// A group of shapes with its own opacity.
#[derive(Clone, Debug)]
pub struct VectorLayer {
    pub name: String,
    pub opacity: f32,
    pub visible: bool,
    pub shapes: Vec<Shape>,
}

impl VectorLayer {
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            opacity: 1.0,
            visible: true,
            shapes: Vec::new(),
        }
    }
}

/// A vector document: canvas size, background and layers.
#[derive(Clone, Debug)]
pub struct VectorDoc {
    pub width: u32,
    pub height: u32,
    pub background: Rgba,
    pub layers: Vec<VectorLayer>,
}

impl VectorDoc {
    #[must_use]
    pub fn new(width: u32, height: u32, background: Rgba) -> Self {
        Self {
            width: width.max(1),
            height: height.max(1),
            background,
            layers: Vec::new(),
        }
    }

    #[must_use]
    pub fn add_layer(&mut self, name: impl Into<String>) -> usize {
        self.layers.push(VectorLayer::new(name));
        self.layers.len() - 1
    }

    /// Appends `shape` to `layer`. An out-of-range layer index is dropped
    /// silently — the source model's documented behaviour, preserved so a
    /// host racing layer removal never turns an append into an error.
    pub fn add_shape(&mut self, layer: usize, shape: Shape) {
        if let Some(target) = self.layers.get_mut(layer) {
            target.shapes.push(shape);
        }
    }

    #[must_use]
    pub fn layer_mut(&mut self, index: usize) -> Option<&mut VectorLayer> {
        self.layers.get_mut(index)
    }
}
