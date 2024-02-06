//! SVG encoding of the normalized vector representation.
//!
//! A hand-written writer keeps the artifact readable and diffable: one line
//! per shape, no external serializer, no digit noise beyond two decimals.
//! The writer was absorbed from the OpenPaint SVG module with the number and
//! escaping contracts preserved; two semantics are deliberately corrected,
//! both driven by the fact that this encoder reads the same [`NormDoc`] the
//! raster lowering paints:
//! - transparency survives: paints carry `fill-opacity` / `stroke-opacity`
//!   instead of the old `#rrggbb` that silently dropped the alpha channel;
//! - a rounded rect keeps its clamped radius in `rx`, so its stroked outline
//!   is the same rounded boundary the raster path tessellates.
//!
//! Layer opacity is already folded into every paint by `media.vector`, so
//! `<g>` carries identity only, never a second transparency.

use media_types::Error;
use media_vector::{NormDoc, NormGeometry, NormShape, Rgba};

/// Writes the normalized document as an SVG file, returning the byte count.
pub fn write_svg(doc: &NormDoc, path: &std::path::Path) -> Result<u64, Error> {
    let svg = to_svg_string(doc);
    std::fs::write(path, svg.as_bytes())?;
    Ok(svg.len() as u64)
}

/// Renders the normalized document as an SVG string.
#[must_use]
pub fn to_svg_string(doc: &NormDoc) -> String {
    let mut out = String::with_capacity(1024);
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n",
        doc.width, doc.height, doc.width, doc.height
    ));
    out.push_str("  <rect width=\"100%\" height=\"100%\"");
    paint(&mut out, Some(doc.background), "fill");
    out.push_str("/>\n");
    for layer in &doc.layers {
        out.push_str(&format!("  <g id=\"{}\">\n", escape(&layer.name)));
        for shape in &layer.shapes {
            out.push_str("    ");
            out.push_str(&element(shape));
            out.push('\n');
        }
        out.push_str("  </g>\n");
    }
    out.push_str("</svg>\n");
    out
}

fn element(shape: &NormShape) -> String {
    let mut out = match &shape.geometry {
        NormGeometry::Outline { points, closed } => {
            let commands: Vec<String> = points
                .iter()
                .enumerate()
                .map(|(index, point)| {
                    let command = if index == 0 { "M" } else { "L" };
                    format!("{command}{} {}", number(point[0]), number(point[1]))
                })
                .collect();
            let mut d = commands.join(" ");
            if *closed {
                d.push_str(" Z");
            }
            format!("<path d=\"{d}\"")
        }
        NormGeometry::RoundRect { origin, size, radius } => {
            let mut out = format!(
                "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"",
                number(origin[0]),
                number(origin[1]),
                number(size[0]),
                number(size[1])
            );
            if *radius > 0.0 {
                out.push_str(&format!(" rx=\"{}\"", number(*radius)));
            }
            out
        }
        NormGeometry::Disc { center, radius } => format!(
            "<circle cx=\"{}\" cy=\"{}\" r=\"{}\"",
            number(center[0]),
            number(center[1]),
            number(*radius)
        ),
    };
    paint(&mut out, shape.fill, "fill");
    paint(&mut out, shape.stroke, "stroke");
    if shape.stroke.is_some() {
        out.push_str(&format!(" stroke-width=\"{}\"", number(shape.width)));
    }
    out.push_str("/>");
    out
}

/// Emits one paint attribute pair: `fill`/`stroke` plus its opacity when the
/// folded alpha is not fully opaque. A fully transparent paint encodes as
/// `none`, matching the raster blend no-op.
fn paint(out: &mut String, color: Option<Rgba>, name: &str) {
    match color {
        None | Some(Rgba([_, _, _, 0])) => out.push_str(&format!(" {name}=\"none\"")),
        Some(color) => {
            let [r, g, b, a] = color.0;
            out.push_str(&format!(" {name}=\"#{r:02x}{g:02x}{b:02x}\""));
            if a < 255 {
                out.push_str(&format!(" {name}-opacity=\"{}\"", alpha(a as f32 / 255.0)));
            }
        }
    }
}

/// Alpha with enough precision to round-trip a byte channel exactly.
fn alpha(value: f32) -> String {
    let text = format!("{value:.4}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn number(value: f32) -> String {
    let text = format!("{value:.2}");
    text.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_vector::{Shape, VectorDoc};

    fn sample() -> NormDoc {
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
        doc.add_shape(0, Shape::circle([32.0, 24.0], 10.0).filled(Rgba([9, 9, 9, 255])));
        doc.normalize()
    }

    #[test]
    fn svg_contains_every_shape_kind() {
        let svg = to_svg_string(&sample());
        assert!(svg.starts_with("<svg"));
        assert!(svg.contains("<rect"), "{svg}");
        assert!(svg.contains("<path"), "{svg}");
        assert!(svg.contains("<circle"), "{svg}");
        assert!(svg.ends_with("</svg>\n"));
    }

    #[test]
    fn transparency_survives_the_encoding() {
        // The deliberate correction: the old writer emitted `#rrggbb` only,
        // so a translucent PNG paint became an opaque SVG shape. Layer
        // opacity is what the normalization folds into the paint alpha.
        let mut doc = VectorDoc::new(8, 8, Rgba([255, 255, 255, 255]));
        let _ = doc.add_layer("half");
        doc.layer_mut(0).expect("layer").opacity = 0.5;
        doc.add_shape(0, Shape::rect([0.0, 0.0], [8.0, 8.0], 0.0).filled(Rgba([255, 0, 0, 255])));
        let _ = doc.add_layer("faint");
        doc.layer_mut(1).expect("layer").opacity = 40.0 / 255.0;
        doc.add_shape(1, Shape::circle([4.0, 4.0], 2.0).stroked(Rgba([0, 255, 0, 255]), 2.0));
        let svg = to_svg_string(&doc.normalize());
        assert!(svg.contains("fill=\"#ff0000\" fill-opacity=\"0.502\""), "{svg}");
        assert!(svg.contains("stroke=\"#00ff00\" stroke-opacity=\"0.1569\""), "{svg}");
        // Fully opaque paints carry no opacity attribute at all.
        let header = svg.lines().nth(1).expect("background line");
        assert!(header.contains("fill=\"#ffffff\"") && !header.contains("fill-opacity"), "{header}");
    }

    #[test]
    fn fully_transparent_paint_encodes_as_none() {
        let mut doc = VectorDoc::new(8, 8, Rgba([0, 0, 0, 0]));
        let _ = doc.add_layer("ink");
        // Just above the drop threshold: the folded alpha rounds to zero,
        // and the raster blend of that paint is a no-op too.
        doc.layer_mut(0).expect("layer").opacity = 0.0015;
        doc.add_shape(0, Shape::circle([4.0, 4.0], 2.0).filled(Rgba([255, 0, 0, 200])));
        let svg = to_svg_string(&doc.normalize());
        assert!(svg.contains("<circle cx=\"4\" cy=\"4\" r=\"2\" fill=\"none\" stroke=\"none\"/>"), "{svg}");
        assert!(svg.contains("<rect width=\"100%\" height=\"100%\" fill=\"none\"/>"), "{svg}");
    }

    #[test]
    fn rounded_rect_keeps_the_clamped_radius_as_rx() {
        let mut doc = VectorDoc::new(32, 32, Rgba::default());
        let _ = doc.add_layer("ink");
        doc.add_shape(0, Shape::rect([0.0, 0.0], [30.0, 10.0], 99.0).stroked(Rgba([1, 1, 1, 255]), 2.0));
        doc.add_shape(0, Shape::rect([0.0, 0.0], [8.0, 8.0], 0.0).filled(Rgba([2, 2, 2, 255])));
        let svg = to_svg_string(&doc.normalize());
        assert!(svg.contains("rx=\"5\""), "radius clamped to half the short edge: {svg}");
        let sharp = svg.lines().find(|line| line.contains("width=\"8\"")).expect("sharp rect");
        assert!(!sharp.contains("rx"), "a sharp rect carries no radius: {sharp}");
    }

    #[test]
    fn closed_paths_end_with_the_z_command() {
        let mut doc = VectorDoc::new(32, 32, Rgba::default());
        let _ = doc.add_layer("ink");
        doc.add_shape(0, Shape::path(vec![[4.0, 4.0], [20.0, 4.0], [12.0, 20.0]]).closed().filled(Rgba([1, 2, 3, 255])));
        doc.add_shape(0, Shape::path(vec![[4.0, 4.0], [20.0, 4.0]]).stroked(Rgba([1, 2, 3, 255]), 2.0));
        let svg = to_svg_string(&doc.normalize());
        assert!(svg.contains("d=\"M4 4 L20 4 L12 20 Z\""), "{svg}");
        assert!(svg.contains("d=\"M4 4 L20 4\""), "{svg}");
    }

    #[test]
    fn numbers_lose_trailing_zeros() {
        assert_eq!(number(3.0), "3");
        assert_eq!(number(0.5), "0.5");
        assert_eq!(number(2.25), "2.25");
        assert_eq!(number(0.0), "0");
        assert_eq!(alpha(128.0 / 255.0), "0.502");
        assert_eq!(alpha(40.0 / 255.0), "0.1569");
    }

    #[test]
    fn layer_names_are_escaped() {
        let mut doc = VectorDoc::new(8, 8, Rgba::default());
        let _ = doc.add_layer("a<b&\"c");
        let svg = to_svg_string(&doc.normalize());
        assert!(svg.contains("<g id=\"a&lt;b&amp;&quot;c\">"), "{svg}");
    }

    #[test]
    fn hidden_layers_never_reach_the_document() {
        let mut doc = sample();
        doc.layers.push(media_vector::NormLayer { name: "ghost".to_string(), shapes: Vec::new() });
        let svg = to_svg_string(&doc);
        assert_eq!(svg.matches("<g ").count(), 2, "normalization already dropped hidden layers, but an appended empty layer still renders its group");

        let mut doc = VectorDoc::new(64, 48, Rgba::default());
        let _ = doc.add_layer("shown");
        let _ = doc.add_layer("hidden");
        doc.layer_mut(1).expect("layer").visible = false;
        let svg = to_svg_string(&doc.normalize());
        assert_eq!(svg.matches("<g ").count(), 1, "{svg}");
        assert!(!svg.contains("hidden"), "{svg}");
    }

    #[test]
    fn write_svg_produces_the_same_bytes_it_returns() {
        let dir = std::env::temp_dir().join("media-codec-svg-tests");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("sample.svg");
        let bytes = write_svg(&sample(), &path).expect("write");
        let on_disk = std::fs::read(&path).expect("read");
        assert_eq!(bytes as usize, on_disk.len());
        assert_eq!(on_disk, to_svg_string(&sample()).as_bytes());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
