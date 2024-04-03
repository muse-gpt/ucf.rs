//! The host input contract: serde shape, absorbed defaults and validation.
//!
//! These types absorb the OpenComic N-API's `DocDto`/`PanelDto`/`TextDto`
//! defaults with the domain wire shape (struct fields snake_case, the
//! `kind` tag kebab-case, and the deliberate author-facing short key `width`
//! for the text wrap width). Bindings map JS-style keys at their own boundary.
//! The deliberate corrections, per the plan's error-unification ruling:
//! - a malformed colour is an explicit [`media_types::Error::invalid_input`],
//!   not the binding's silent `parse_or` fallback;
//! - an unknown `fit` spelling fails deserialization instead of silently
//!   becoming `Contain`;
//! - sizes, border widths, page dimensions and sheet parameters that the
//!   source clamped silently are validated and rejected explicitly.
//!
//! Bindings extract host values, deserialize a [`ComicSpec`] and call
//! [`ComicSpec::resolve`]; they must not re-implement defaults or validation.

use geometry::{Rgba, parse};
use media_image::Fit;
use media_types::Error;
use serde::Deserialize;

use crate::model::{Border, ComicDoc, Panel, TextItem, TextKind};

/// Page background used when the host omits one. Absorbed from the binding.
pub const DEFAULT_BACKGROUND: Rgba = Rgba([255, 255, 255, 255]);
/// Panel fill used when the host omits one. Absorbed from the binding.
pub const DEFAULT_PANEL_FILL: Rgba = Rgba([255, 255, 255, 255]);
/// Border colour used when a border carries none. Absorbed from the binding.
pub const DEFAULT_BORDER_COLOR: Rgba = Rgba([20, 20, 24, 255]);
/// Border width used when a border carries none. Absorbed from the binding.
pub const DEFAULT_BORDER_WIDTH: f32 = 3.0;
/// Text anchor used when the author places none (the source Rust default).
pub const DEFAULT_TEXT_AT: [f32; 2] = [0.0, 0.0];
/// Text size used when the host omits one. Absorbed from the core default.
pub const DEFAULT_TEXT_SIZE: f32 = 18.0;
/// Wrap width used when the host omits one. Absorbed from the core default.
pub const DEFAULT_TEXT_MAX_WIDTH: f32 = 240.0;
/// Ink colour used when the host omits one. Absorbed from the core default.
pub const DEFAULT_TEXT_COLOR: Rgba = Rgba([16, 16, 20, 255]);
/// Box fill used when the host omits one. Absorbed from the core default.
pub const DEFAULT_TEXT_FILL: Rgba = Rgba([255, 255, 255, 255]);
/// Box stroke used when the host omits one. Absorbed from the core default.
pub const DEFAULT_TEXT_STROKE: Rgba = Rgba([20, 20, 24, 255]);
/// Smallest accepted text size (the source clamped silently to this).
pub const MIN_TEXT_SIZE: f32 = 1.0;
/// Smallest accepted contact-sheet scale (the source clamped silently).
pub const MIN_SHEET_SCALE: f32 = 0.05;
/// Largest accepted contact-sheet scale (the source clamped silently).
pub const MAX_SHEET_SCALE: f32 = 4.0;

/// A whole comic document as the host sends it.
#[derive(Deserialize, Debug)]
pub struct ComicSpec {
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub background: Option<String>,
    #[serde(default)]
    pub pages: Vec<PageSpec>,
}

/// One page as the host sends it.
#[derive(Deserialize, Debug)]
pub struct PageSpec {
    #[serde(default)]
    pub panels: Vec<PanelSpec>,
}

/// One panel as the host sends it.
#[derive(Deserialize, Debug)]
pub struct PanelSpec {
    pub rect: [f32; 4],
    #[serde(default)]
    pub fill: Option<String>,
    #[serde(default)]
    pub image: Option<String>,
    #[serde(default)]
    pub fit: Option<Fit>,
    #[serde(default)]
    pub border: Option<BorderSpec>,
    #[serde(default)]
    pub texts: Vec<TextSpec>,
}

/// A panel border as the host sends it.
#[derive(Deserialize, Debug)]
pub struct BorderSpec {
    #[serde(default)]
    pub color: Option<String>,
    #[serde(default)]
    pub width: Option<f32>,
}

/// One text item as the host sends it: a tagged union on `kind`, matching
/// the historical binding DTO.
#[derive(Deserialize, Debug)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum TextSpec {
    Caption {
        text: String,
        #[serde(default)]
        at: Option<[f32; 2]>,
        #[serde(default)]
        size: Option<f32>,
        // The author-facing short key for `max_width`, kept from the
        // historical DTO as a deliberate per-field exception.
        #[serde(rename = "width", default)]
        max_width: Option<f32>,
        #[serde(default)]
        bold: bool,
        #[serde(default)]
        color: Option<String>,
        #[serde(default)]
        fill: Option<String>,
        #[serde(default)]
        stroke: Option<String>,
    },
    Balloon {
        text: String,
        #[serde(default)]
        at: Option<[f32; 2]>,
        tail: [f32; 2],
        #[serde(default)]
        size: Option<f32>,
        #[serde(rename = "width", default)]
        max_width: Option<f32>,
        #[serde(default)]
        bold: bool,
        #[serde(default)]
        color: Option<String>,
        #[serde(default)]
        fill: Option<String>,
        #[serde(default)]
        stroke: Option<String>,
    },
}

impl ComicSpec {
    /// Validates and resolves the host shape into the document model.
    /// Every default above is applied here and nowhere else.
    pub fn resolve(&self) -> Result<ComicDoc, Error> {
        if self.width < 1 {
            return Err(Error::invalid_input("page width must be at least 1"));
        }
        if self.height < 1 {
            return Err(Error::invalid_input("page height must be at least 1"));
        }
        let background = resolve_color(self.background.as_deref(), DEFAULT_BACKGROUND, "background")?;
        let mut doc = ComicDoc::new(self.width, self.height, background);
        for (index, page) in self.pages.iter().enumerate() {
            let target = doc.add_page();
            let page_mut = doc.page_mut(target).expect("page just added");
            page_mut.panels = page.panels.iter().enumerate().map(|(i, panel)| panel.resolve(index, i)).collect::<Result<Vec<Panel>, Error>>()?;
        }
        Ok(doc)
    }
}

impl PanelSpec {
    pub(crate) fn resolve(&self, page: usize, index: usize) -> Result<Panel, Error> {
        let field = format!("page {page} panel {index}");
        for value in self.rect {
            finite(value, &format!("{field} rect"))?;
        }
        if self.rect[2] < 0.0 || self.rect[3] < 0.0 {
            return Err(Error::invalid_input(format!("{field} rect: width and height must not be negative")));
        }
        let fill = resolve_color(self.fill.as_deref(), DEFAULT_PANEL_FILL, &format!("{field} fill"))?;
        let mut panel = Panel {
            rect: self.rect,
            fill,
            image: self.image.clone(),
            fit: self.fit.unwrap_or_default(),
            border: None,
            texts: Vec::new(),
        };
        if let Some(border) = &self.border {
            let width = match border.width {
                None => DEFAULT_BORDER_WIDTH,
                Some(width) => {
                    finite(width, &format!("{field} border width"))?;
                    if width < 0.0 {
                        return Err(Error::invalid_input(format!("{field} border width: must not be negative")));
                    }
                    width
                }
            };
            panel.border = Some(Border {
                color: resolve_color(border.color.as_deref(), DEFAULT_BORDER_COLOR, &format!("{field} border color"))?,
                width,
            });
        }
        for (text_index, text) in self.texts.iter().enumerate() {
            panel.texts.push(text.resolve(&format!("{field} text {text_index}"))?);
        }
        Ok(panel)
    }
}

impl TextSpec {
    pub(crate) fn resolve(&self, field: &str) -> Result<TextItem, Error> {
        let (kind, text, at, tail, size, max_width, bold, color, fill, stroke) = match self {
            Self::Caption { text, at, size, max_width, bold, color, fill, stroke } => {
                (TextKind::Caption, text, *at, None, *size, *max_width, *bold, color, fill, stroke)
            }
            Self::Balloon { text, at, tail, size, max_width, bold, color, fill, stroke } => {
                (TextKind::Balloon, text, *at, Some(*tail), *size, *max_width, *bold, color, fill, stroke)
            }
        };
        let at = at.unwrap_or(DEFAULT_TEXT_AT);
        finite(at[0], &format!("{field} at"))?;
        finite(at[1], &format!("{field} at"))?;
        if let Some(tail) = tail {
            finite(tail[0], &format!("{field} tail"))?;
            finite(tail[1], &format!("{field} tail"))?;
        }
        let size = match size {
            None => DEFAULT_TEXT_SIZE,
            Some(size) => {
                finite(size, &format!("{field} size"))?;
                if size < MIN_TEXT_SIZE {
                    return Err(Error::invalid_input(format!("{field} size: must be at least {MIN_TEXT_SIZE}")));
                }
                size
            }
        };
        let max_width = match max_width {
            None => DEFAULT_TEXT_MAX_WIDTH,
            Some(max_width) => {
                finite(max_width, &format!("{field} width"))?;
                if max_width < MIN_TEXT_SIZE {
                    return Err(Error::invalid_input(format!("{field} width: must be at least {MIN_TEXT_SIZE}")));
                }
                max_width
            }
        };
        Ok(TextItem {
            kind,
            text: text.clone(),
            at,
            tail,
            size,
            max_width,
            bold,
            color: resolve_color(color.as_deref(), DEFAULT_TEXT_COLOR, &format!("{field} color"))?,
            fill: resolve_color(fill.as_deref(), DEFAULT_TEXT_FILL, &format!("{field} fill"))?,
            stroke: resolve_color(stroke.as_deref(), DEFAULT_TEXT_STROKE, &format!("{field} stroke"))?,
        })
    }
}

fn resolve_color(value: Option<&str>, fallback: Rgba, field: &str) -> Result<Rgba, Error> {
    match value {
        None => Ok(fallback),
        Some(text) => parse(text).ok_or_else(|| Error::invalid_input(format!("{field}: invalid color '{text}'"))),
    }
}

fn finite(value: f32, field: &str) -> Result<(), Error> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(Error::invalid_input(format!("{field}: values must be finite")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(json: &str) -> ComicSpec {
        serde_json::from_str(json).expect("input parses")
    }

    #[test]
    fn minimal_document_takes_the_absorbed_defaults() {
        let doc = spec(r#"{"width":400,"height":300,"pages":[{"panels":[{"rect":[0.0,0.0,100.0,100.0],"texts":[{"kind":"caption","text":"HI"}]}]}]}"#)
            .resolve()
            .expect("defaults resolve");
        assert_eq!(doc.width, 400);
        assert_eq!(doc.height, 300);
        assert_eq!(doc.background, DEFAULT_BACKGROUND);
        let panel = &doc.pages[0].panels[0];
        assert_eq!(panel.fill, DEFAULT_PANEL_FILL);
        assert_eq!(panel.fit, Fit::Contain);
        assert!(panel.border.is_none());
        let text = &panel.texts[0];
        assert_eq!(text.kind, TextKind::Caption);
        assert_eq!(text.at, DEFAULT_TEXT_AT);
        assert_eq!(text.size, DEFAULT_TEXT_SIZE);
        assert_eq!(text.max_width, DEFAULT_TEXT_MAX_WIDTH);
        assert!(!text.bold);
        assert_eq!(text.color, DEFAULT_TEXT_COLOR);
        assert_eq!(text.fill, DEFAULT_TEXT_FILL);
        assert_eq!(text.stroke, DEFAULT_TEXT_STROKE);
        assert_eq!(text.tail, None);
    }

    #[test]
    fn wire_shape_matches_the_historical_binding_dto() {
        let doc = spec(
            r##"{"width":1200,"height":800,"background":"#f6f4f0","pages":[{"panels":[
                {"rect":[40.0,24.0,360.0,200.0],"fill":"#ffffff","image":"panel.png","fit":"cover",
                 "border":{"color":"#141418","width":3.0},
                 "texts":[
                    {"kind":"caption","text":"EST. NIGHT","at":[52.0,36.0],"size":22.0,"width":336.0,"bold":true,"fill":"#e8b86d","stroke":"#141418"},
                    {"kind":"balloon","text":"It works.","at":[200.0,150.0],"tail":[200.0,240.0],"color":"#101014"}
                 ]}
            ]}]}"##,
        )
        .resolve()
        .expect("full wire resolves");
        assert_eq!(doc.background, Rgba([246, 244, 240, 255]));
        let panel = &doc.pages[0].panels[0];
        assert_eq!(panel.image.as_deref(), Some("panel.png"));
        assert_eq!(panel.fit, Fit::Cover);
        let border = panel.border.expect("border");
        assert_eq!((border.color, border.width), (Rgba([20, 20, 24, 255]), 3.0));
        let caption = &panel.texts[0];
        assert_eq!(caption.at, [52.0, 36.0]);
        assert_eq!(caption.size, 22.0);
        assert_eq!(caption.max_width, 336.0);
        assert!(caption.bold);
        assert_eq!(caption.fill, Rgba([232, 184, 109, 255]));
        let balloon = &panel.texts[1];
        assert_eq!(balloon.kind, TextKind::Balloon);
        assert_eq!(balloon.tail, Some([200.0, 240.0]));
    }

    #[test]
    fn malformed_colors_are_explicit_errors() {
        let error = spec(r##"{"width":10,"height":10,"background":"#zzz"}"##).resolve().expect_err("bad background rejects");
        assert_eq!(error.code(), "invalid_input");
        assert!(error.message.contains("background: invalid color '#zzz'"), "{}", error.message);
        let error = spec(r##"{"width":10,"height":10,"pages":[{"panels":[{"rect":[0.0,0.0,1.0,1.0],"fill":"nope"}]}]}"##)
            .resolve()
            .expect_err("bad panel fill rejects");
        assert!(error.message.contains("page 0 panel 0 fill"), "{}", error.message);
    }

    #[test]
    fn unknown_fit_spelling_fails_deserialization() {
        // The deliberate correction: the binding silently fell back to Contain.
        assert!(serde_json::from_str::<ComicSpec>(r##"{"width":10,"height":10,"pages":[{"panels":[{"rect":[0.0,0.0,1.0,1.0],"fit":"tile"}]}]}"##).is_err());
        for accepted in ["contain", "cover", "stretch"] {
            let json = format!(r#"{{"width":10,"height":10,"pages":[{{"panels":[{{"rect":[0.0,0.0,1.0,1.0],"fit":"{accepted}"}}]}}]}}"#);
            serde_json::from_str::<ComicSpec>(&json).expect("accepted fit spelling");
        }
    }

    #[test]
    fn out_of_range_numbers_are_explicit_errors() {
        // The source clamped all of these silently; now they reject.
        let error = spec(r##"{"width":0,"height":10}"##).resolve().expect_err("zero width rejects");
        assert_eq!(error.code(), "invalid_input");
        let error = spec(r##"{"width":10,"height":0}"##).resolve().expect_err("zero height rejects");
        assert_eq!(error.code(), "invalid_input");
        let error = spec(r#"{"width":10,"height":10,"pages":[{"panels":[{"rect":[0.0,0.0,-1.0,1.0]}]}]}"#)
            .resolve()
            .expect_err("negative rect extent rejects");
        assert!(error.message.contains("must not be negative"), "{}", error.message);
        let error = spec(r#"{"width":10,"height":10,"pages":[{"panels":[{"rect":[0.0,0.0,1.0,1.0],"border":{"width":-2.0}}]}]}"#)
            .resolve()
            .expect_err("negative border rejects");
        assert!(error.message.contains("border width"), "{}", error.message);
        let error = spec(r#"{"width":10,"height":10,"pages":[{"panels":[{"rect":[0.0,0.0,1.0,1.0],"texts":[{"kind":"caption","text":"x","size":0.5}]}]}]}"#)
            .resolve()
            .expect_err("sub-pixel size rejects");
        assert!(error.message.contains("size: must be at least 1"), "{}", error.message);
        let error = spec(r#"{"width":10,"height":10,"pages":[{"panels":[{"rect":[0.0,0.0,1.0,1.0],"texts":[{"kind":"caption","text":"x","size":1e300}]}]}]}"#)
            .resolve()
            .expect_err("infinite size rejects");
        assert!(error.message.contains("size"), "{}", error.message);
    }

    #[test]
    fn non_finite_rect_coordinates_are_explicit_errors() {
        let error = spec(r#"{"width":10,"height":10,"pages":[{"panels":[{"rect":[1e300,0.0,1.0,1.0]}]}]}"#)
            .resolve()
            .expect_err("infinite coordinate rejects");
        assert!(error.message.contains("rect: values must be finite"), "{}", error.message);
    }

    #[test]
    fn balloon_requires_a_tail_and_caption_ignores_one() {
        assert!(serde_json::from_str::<ComicSpec>(r#"{"width":10,"height":10,"pages":[{"panels":[{"rect":[0.0,0.0,1.0,1.0],"texts":[{"kind":"balloon","text":"x"}]}]}]}"#).is_err());
        let doc = spec(r#"{"width":10,"height":10,"pages":[{"panels":[{"rect":[0.0,0.0,1.0,1.0],"texts":[{"kind":"caption","text":"x","tail":[1.0,2.0]}]}]}]}"#)
            .resolve()
            .expect("extra key is ignored like the historical DTO");
        assert_eq!(doc.pages[0].panels[0].texts[0].tail, None);
    }
}
