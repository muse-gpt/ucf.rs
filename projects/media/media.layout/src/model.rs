//! The resolved document model: pages of panels with artwork keys and
//! self-sizing text boxes.
//!
//! Values here are already validated: colours are [`Rgba`], sizes are finite
//! and in range. The host-facing shape with defaults and validation lives in
//! [`crate::spec`]; this module is what Rust authors build directly and what
//! [`crate::plan`] consumes.

use geometry::Rgba;
use media_image::Fit;

use crate::spec::{
    DEFAULT_PANEL_FILL, DEFAULT_TEXT_AT, DEFAULT_TEXT_COLOR, DEFAULT_TEXT_FILL, DEFAULT_TEXT_MAX_WIDTH, DEFAULT_TEXT_SIZE,
    DEFAULT_TEXT_STROKE,
};

/// What a text item is drawn as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextKind {
    /// A rectangular narration box anchored at its top-left corner.
    Caption,
    /// A rounded speech balloon anchored at its centre, with a tail.
    Balloon,
}

/// One piece of text on a panel. The box sizes itself to the wrapped text,
/// so the author places the anchor and never the box.
#[derive(Clone, Debug)]
pub struct TextItem {
    pub kind: TextKind,
    pub text: String,
    /// Caption: top-left of the box. Balloon: the box centre.
    pub at: [f32; 2],
    /// Balloon only: the point the tail points at.
    pub tail: Option<[f32; 2]>,
    pub size: f32,
    /// Wrap width for the text inside the box.
    pub max_width: f32,
    pub bold: bool,
    pub color: Rgba,
    pub fill: Rgba,
    pub stroke: Rgba,
}

impl TextItem {
    #[must_use]
    pub fn caption(text: impl Into<String>) -> Self {
        Self::new(TextKind::Caption, text, None)
    }

    #[must_use]
    pub fn balloon(text: impl Into<String>, tail: [f32; 2]) -> Self {
        Self::new(TextKind::Balloon, text, Some(tail))
    }

    fn new(kind: TextKind, text: impl Into<String>, tail: Option<[f32; 2]>) -> Self {
        Self {
            kind,
            text: text.into(),
            at: DEFAULT_TEXT_AT,
            tail,
            size: DEFAULT_TEXT_SIZE,
            max_width: DEFAULT_TEXT_MAX_WIDTH,
            bold: false,
            color: DEFAULT_TEXT_COLOR,
            fill: DEFAULT_TEXT_FILL,
            stroke: DEFAULT_TEXT_STROKE,
        }
    }

    #[must_use]
    pub fn at(mut self, at: [f32; 2]) -> Self {
        self.at = at;
        self
    }

    #[must_use]
    pub fn size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    #[must_use]
    pub fn max_width(mut self, max_width: f32) -> Self {
        self.max_width = max_width;
        self
    }

    #[must_use]
    pub fn bold(mut self) -> Self {
        self.bold = true;
        self
    }

    #[must_use]
    pub fn colored(mut self, color: Rgba) -> Self {
        self.color = color;
        self
    }

    #[must_use]
    pub fn boxed(mut self, fill: Rgba, stroke: Rgba) -> Self {
        self.fill = fill;
        self.stroke = stroke;
        self
    }
}

/// A panel border: one stroke of fixed width around the panel rect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Border {
    pub color: Rgba,
    pub width: f32,
}

/// One rectangular panel on a page.
#[derive(Clone, Debug)]
pub struct Panel {
    /// `x, y, width, height` in page pixels.
    pub rect: [f32; 4],
    pub fill: Rgba,
    /// A key the render-time [`crate::plan::ImageStore`] resolves to decoded
    /// pixels. The store, not this model, owns where the bytes come from.
    pub image: Option<String>,
    pub fit: Fit,
    pub border: Option<Border>,
    pub texts: Vec<TextItem>,
}

impl Panel {
    #[must_use]
    pub fn new(rect: [f32; 4]) -> Self {
        Self { rect, fill: DEFAULT_PANEL_FILL, image: None, fit: Fit::Contain, border: None, texts: Vec::new() }
    }

    #[must_use]
    pub fn filled(mut self, color: Rgba) -> Self {
        self.fill = color;
        self
    }

    /// Attaches an artwork key the render-time [`crate::plan::ImageStore`]
    /// resolves, together with how it fits the panel rect.
    #[must_use]
    pub fn image(mut self, key: impl Into<String>, fit: Fit) -> Self {
        self.image = Some(key.into());
        self.fit = fit;
        self
    }

    #[must_use]
    pub fn bordered(mut self, color: Rgba, width: f32) -> Self {
        self.border = Some(Border { color, width });
        self
    }

    #[must_use]
    pub fn text(mut self, item: TextItem) -> Self {
        self.texts.push(item);
        self
    }
}

/// One page: an ordered list of panels.
#[derive(Default, Debug)]
pub struct Page {
    pub panels: Vec<Panel>,
}

impl Page {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_panel(&mut self, panel: Panel) -> usize {
        self.panels.push(panel);
        self.panels.len() - 1
    }

    #[must_use]
    pub fn panel_mut(&mut self, index: usize) -> Option<&mut Panel> {
        self.panels.get_mut(index)
    }
}

/// A comic document: page size, paper colour and pages.
///
/// Direct Rust construction trusts the author; the zero-size and range
/// checks of the host contract live in [`crate::spec`].
#[derive(Debug)]
pub struct ComicDoc {
    pub width: u32,
    pub height: u32,
    pub background: Rgba,
    pub pages: Vec<Page>,
}

impl ComicDoc {
    #[must_use]
    pub fn new(width: u32, height: u32, background: Rgba) -> Self {
        Self { width, height, background, pages: Vec::new() }
    }

    pub fn add_page(&mut self) -> usize {
        self.pages.push(Page::new());
        self.pages.len() - 1
    }

    #[must_use]
    pub fn page_mut(&mut self, index: usize) -> Option<&mut Page> {
        self.pages.get_mut(index)
    }
}
