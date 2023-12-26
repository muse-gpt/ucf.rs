//! The sprite spec: a declarative input contract plus its replay.
//!
//! This is the upstream home of what the OpenPixel binding layer used to
//! own: the palette default (empty list → the warm sixteen), the export
//! scale default (absent → 1), hex colour parsing with explicit errors
//! and the ordered replay of drawing operations. A binding converts host
//! values into [`SpriteSpec`] and calls [`render_sprite`] — nothing else.
//! Writing the bitmap to a file is composed by the caller with
//! `media.codec`.

use media_image::Bitmap;
use media_types::Error;
use serde::Deserialize;

use crate::canvas::PixelCanvas;
use crate::palette::Palette;

/// Upper bound on both the canvas area and the exported area, in pixels,
/// accepted by [`render_sprite`].
///
/// The raw [`PixelCanvas`] API keeps the absorbed implementation's uncapped
/// behaviour; this budget guards the spec entry, where the OpenPixel
/// original had none and a hostile size went straight to an allocation
/// failure. 16 Mi index pixels bound the storage to 16 MiB and the RGBA
/// export to 64 MiB.
pub const MAX_SPRITE_PIXELS: usize = 16 * 1024 * 1024;

/// A whole sprite description: canvas size, optional palette, optional
/// export scale and the operations to replay in order.
#[derive(Clone, Debug, Deserialize)]
pub struct SpriteSpec {
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub scale: Option<u32>,
    #[serde(default)]
    pub palette: Vec<String>,
    #[serde(default)]
    pub ops: Vec<PixelOp>,
}

/// One drawing operation of a [`SpriteSpec`].
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum PixelOp {
    Pixel { x: u32, y: u32, index: u8 },
    Line {
        from: [u32; 2],
        to: [u32; 2],
        index: u8,
    },
    Rect {
        at: [u32; 2],
        size: [u32; 2],
        index: u8,
        #[serde(default)]
        filled: bool,
    },
    Flood { at: [u32; 2], index: u8 },
}

impl PixelOp {
    /// Applies one operation. Coordinates beyond the canvas stay silent by
    /// design — the same contract the raw tools document.
    pub fn apply(&self, canvas: &mut PixelCanvas) {
        match self {
            Self::Pixel { x, y, index } => canvas.set(*x, *y, *index),
            Self::Line { from, to, index } => canvas.line(from[0], from[1], to[0], to[1], *index),
            Self::Rect {
                at,
                size,
                index,
                filled,
            } => canvas.rect(at[0], at[1], size[0], size[1], *index, *filled),
            Self::Flood { at, index } => canvas.flood(at[0], at[1], *index),
        }
    }
}

/// Executes a sprite spec into an RGBA bitmap: input budget, palette
/// resolution (empty → default table, malformed hex → explicit
/// `invalid_input`), ordered replay, nearest-neighbour export scale and
/// index expansion at the boundary.
pub fn render_sprite(spec: &SpriteSpec) -> Result<Bitmap, Error> {
    let width = spec.width.max(1);
    let height = spec.height.max(1);
    check_area(width, height)?;
    let palette = if spec.palette.is_empty() {
        Palette::default()
    } else {
        Palette::from_hex(&spec.palette)?
    };
    let mut canvas = PixelCanvas::new(spec.width, spec.height);
    for op in &spec.ops {
        op.apply(&mut canvas);
    }
    let factor = spec.scale.unwrap_or(1).max(1);
    let scaled_width = width
        .checked_mul(factor)
        .ok_or_else(|| Error::invalid_input(format!("sprite scale {factor} overflows {width}x{height}")))?;
    let scaled_height = height
        .checked_mul(factor)
        .ok_or_else(|| Error::invalid_input(format!("sprite scale {factor} overflows {width}x{height}")))?;
    check_area(scaled_width, scaled_height)?;
    let scaled = canvas.scale(factor);
    Ok(scaled.to_bitmap(&palette))
}

fn check_area(width: u32, height: u32) -> Result<(), Error> {
    let area = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| Error::invalid_input(format!("sprite area {width}x{height} overflows")))?;
    if area > MAX_SPRITE_PIXELS {
        return Err(Error::invalid_input(format!(
            "sprite area {area} exceeds the {MAX_SPRITE_PIXELS} pixel budget"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn smoke_spec() -> SpriteSpec {
        SpriteSpec {
            width: 16,
            height: 16,
            scale: Some(8),
            palette: vec![
                "#000000".to_string(),
                "#343a54".to_string(),
                "#cc4448".to_string(),
                "#f6c470".to_string(),
                "#4884c8".to_string(),
            ],
            ops: vec![
                PixelOp::Rect {
                    at: [2, 2],
                    size: [12, 12],
                    index: 1,
                    filled: true,
                },
                PixelOp::Rect {
                    at: [4, 4],
                    size: [8, 8],
                    index: 2,
                    filled: true,
                },
                PixelOp::Pixel {
                    x: 8,
                    y: 8,
                    index: 3,
                },
                PixelOp::Line {
                    from: [1, 1],
                    to: [14, 1],
                    index: 4,
                },
                PixelOp::Flood { at: [0, 0], index: 4 },
            ],
        }
    }

    fn pixel(bitmap: &Bitmap, x: u32, y: u32) -> [u8; 4] {
        let offset = ((y * bitmap.width + x) as usize) * 4;
        bitmap.rgba[offset..offset + 4].try_into().unwrap()
    }

    #[test]
    fn smoke_sprite_replays_to_expected_bitmap() {
        // Golden of the OpenPixel binding smoke input: five ops over a
        // custom five-colour palette, nearest-neighbour export at 8x.
        let bitmap = render_sprite(&smoke_spec()).unwrap();
        assert_eq!((bitmap.width, bitmap.height), (128, 128));
        // (0,0) was index 0, flooded to index 4 (#4884c8).
        assert_eq!(pixel(&bitmap, 0, 0), [72, 132, 200, 255]);
        // (3,3) sits inside the first rect (index 1, #343a54).
        assert_eq!(pixel(&bitmap, 24, 24), [52, 58, 84, 255]);
        // (5,5) inside the second rect (index 2, #cc4448).
        assert_eq!(pixel(&bitmap, 40, 40), [204, 68, 72, 255]);
        // (8,8) is the lone pixel op (index 3, #f6c470).
        assert_eq!(pixel(&bitmap, 64, 64), [246, 196, 112, 255]);
        // Nearest-neighbour blocks: the whole 8x8 block of (0,0) agrees.
        assert_eq!(pixel(&bitmap, 7, 7), [72, 132, 200, 255]);
        // The pixel op's block starts at (64,64); (63,63) still maps to
        // canvas (7,7), the second rect's index 2.
        assert_eq!(pixel(&bitmap, 63, 63), [204, 68, 72, 255], "block boundary");
        assert_eq!(pixel(&bitmap, 71, 71), [246, 196, 112, 255], "block interior");
    }

    #[test]
    fn sprite_defaults_apply_palette_and_scale() {
        let spec = SpriteSpec {
            width: 2,
            height: 2,
            scale: None,
            palette: vec![],
            ops: vec![PixelOp::Pixel {
                x: 0,
                y: 0,
                index: 1,
            }],
        };
        let bitmap = render_sprite(&spec).unwrap();
        assert_eq!((bitmap.width, bitmap.height), (2, 2));
        assert_eq!(pixel(&bitmap, 0, 0), Palette::default().color(1));
        assert_eq!(pixel(&bitmap, 1, 1), [0, 0, 0, 0]);
    }

    #[test]
    fn ops_replay_in_order() {
        let spec = SpriteSpec {
            width: 4,
            height: 4,
            scale: Some(0),
            palette: vec![],
            ops: vec![
                PixelOp::Pixel {
                    x: 1,
                    y: 1,
                    index: 2,
                },
                PixelOp::Rect {
                    at: [0, 0],
                    size: [3, 3],
                    index: 5,
                    filled: true,
                },
                PixelOp::Flood { at: [3, 3], index: 9 },
            ],
        };
        let bitmap = render_sprite(&spec).unwrap();
        assert_eq!((bitmap.width, bitmap.height), (4, 4), "scale 0 clamps to 1");
        assert_eq!(pixel(&bitmap, 1, 1), Palette::default().color(5), "later op wins");
        assert_eq!(pixel(&bitmap, 3, 3), Palette::default().color(9));
    }

    #[test]
    fn zero_dimensions_still_render() {
        let spec = SpriteSpec {
            width: 0,
            height: 0,
            scale: Some(3),
            palette: vec![],
            ops: vec![],
        };
        let bitmap = render_sprite(&spec).unwrap();
        assert_eq!((bitmap.width, bitmap.height), (3, 3));
    }

    #[test]
    fn malformed_palette_hex_errors_the_whole_sprite() {
        // The deliberate correction: the OpenPixel binding silently turned
        // a bad colour transparent; the domain entry rejects it.
        let spec = SpriteSpec {
            width: 2,
            height: 2,
            scale: None,
            palette: vec!["#000".to_string(), "red".to_string()],
            ops: vec![],
        };
        let error = render_sprite(&spec).err().unwrap();
        assert_eq!(error.code(), "invalid_input");
    }

    #[test]
    fn oversized_canvas_hits_the_input_budget() {
        let spec = SpriteSpec {
            width: u32::MAX,
            height: 2,
            scale: None,
            palette: vec![],
            ops: vec![],
        };
        let error = render_sprite(&spec).err().unwrap();
        assert_eq!(error.code(), "invalid_input");

        let at_budget = SpriteSpec {
            width: 4096,
            height: 4096,
            scale: None,
            palette: vec![],
            ops: vec![],
        };
        assert!(render_sprite(&at_budget).is_ok(), "16 Mi canvas is inside the budget");
    }

    #[test]
    fn oversized_export_scale_hits_the_input_budget() {
        let spec = SpriteSpec {
            width: 16,
            height: 16,
            scale: Some(100_000),
            palette: vec![],
            ops: vec![],
        };
        let error = render_sprite(&spec).err().unwrap();
        assert_eq!(error.code(), "invalid_input");
    }
}
