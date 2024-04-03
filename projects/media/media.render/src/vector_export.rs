//! Vector export: the authoritative vector-plus-raster-plus-codec composition.
//!
//! A document is normalized by `media.vector`, then driven through exactly
//! one of two encoders over the same [`NormDoc`]: the raster path paints it
//! with `media.raster` into `media.image` storage and encodes a PNG through
//! `media.codec`, the vector path encodes SVG through the codec's `svg`
//! feature. Because both readings consume the one resolved model, the
//! OpenPaint baseline's rounded-rect stroke split cannot recur: the raster
//! tessellates the clamped radius while the SVG keeps it as `rx`.
//!
//! File IO is gated behind the `vector-export` feature, mirroring how the
//! scene, sfx and image-pipeline entries gate theirs. Bindings map host
//! values in and results out — the `format` default (PNG) is resolved here,
//! not in the binding.

use std::path::Path;

use geometry::rounded_rect_outline;
use media_image::Frame;
use media_raster::Paint;
use media_types::Error;
use media_vector::{NormDoc, NormGeometry, NormShape, ROUNDED_CORNER_SEGMENTS, VectorDoc};

/// Export format. Absorbed from the OpenPaint binding contract: an absent
/// format means PNG.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorFormat {
    Png,
    Svg,
}

impl VectorFormat {
    /// Resolves the host spelling. Anything unknown is an explicit
    /// `invalid_input` error instead of the binding's silent PNG fallback.
    pub fn parse(name: Option<&str>) -> Result<Self, Error> {
        match name {
            None | Some("png") => Ok(Self::Png),
            Some("svg") => Ok(Self::Svg),
            Some(other) => Err(Error::invalid_input(format!("vector export: unknown format '{other}'"))),
        }
    }

    /// The stable spelling, the same as the serde discriminant.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Svg => "svg",
        }
    }
}

/// Facts about a finished vector export run.
#[derive(serde::Serialize)]
pub struct ExportedVector {
    pub out: String,
    pub format: &'static str,
    pub bytes: u64,
}

/// Paints the normalized document through `media.raster` into RGBA8 storage.
///
/// This is the raster half of the shared-representation contract: every
/// geometry decision already happened in `normalize`, so this function only
/// dispatches resolved primitives onto the `Paint` trait.
#[must_use]
pub fn render_frame(doc: &NormDoc) -> Frame {
    let mut frame = Frame::new(doc.width, doc.height);
    frame.fill(doc.background);
    for layer in &doc.layers {
        for shape in &layer.shapes {
            paint_shape(&mut frame, shape);
        }
    }
    frame
}

fn paint_shape(frame: &mut Frame, shape: &NormShape) {
    match &shape.geometry {
        NormGeometry::Outline { points, closed } => {
            let mut outline = points.clone();
            if *closed && outline.len() >= 2 {
                outline.push(outline[0]);
            }
            if outline.len() < 2 {
                return;
            }
            if let Some(color) = shape.fill {
                frame.fill_polygon(&outline, color, false);
            }
            if let Some(color) = shape.stroke {
                frame.stroke_polyline(&outline, shape.width, color, 1.0, false);
            }
        }
        NormGeometry::RoundRect { origin, size, radius } => {
            if let Some(color) = shape.fill {
                if *radius > 0.0 {
                    frame.round_rect(origin[0], origin[1], size[0], size[1], *radius, color);
                } else {
                    frame.rect(origin[0], origin[1], size[0], size[1], color);
                }
            }
            if let Some(color) = shape.stroke {
                // The deliberate correction: the stroked boundary is the same
                // rounded outline the SVG `rx` describes, not the sharp
                // four-edge polyline the source raster path used.
                let outline = rounded_rect_outline(*origin, *size, *radius, ROUNDED_CORNER_SEGMENTS);
                frame.stroke_polyline(&outline, shape.width, color, 1.0, false);
            }
        }
        NormGeometry::Disc { center, radius } => {
            if let Some(color) = shape.fill {
                frame.circle(center[0], center[1], *radius, color, None);
            }
            if let Some(color) = shape.stroke {
                frame.circle(center[0], center[1], *radius, color, Some(shape.width));
            }
        }
    }
}

/// Normalizes `doc` and exports it as `format` to `out`, creating the output
/// directory.
pub fn write_vector(doc: &VectorDoc, out: &Path, format: VectorFormat) -> Result<ExportedVector, Error> {
    write_normalized(&doc.normalize(), out, format)
}

/// Exports an already-normalized document. Hosts that normalize once and
/// encode both artifacts call this directly.
pub fn write_normalized(doc: &NormDoc, out: &Path, format: VectorFormat) -> Result<ExportedVector, Error> {
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    match format {
        VectorFormat::Svg => {
            media_codec::write_svg(doc, out)?;
        }
        VectorFormat::Png => {
            let frame = render_frame(doc);
            media_codec::write_png(out, frame.width, frame.height, &frame.pixels)?;
        }
    }
    let bytes = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    Ok(ExportedVector { out: out.display().to_string(), format: format.as_str(), bytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_vector::{Shape, VectorDoc};

    fn ink() -> media_vector::Rgba {
        media_vector::Rgba([255, 0, 0, 255])
    }

    fn pixel(frame: &Frame, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * frame.width + x) as usize) * 4;
        [frame.pixels[i], frame.pixels[i + 1], frame.pixels[i + 2], frame.pixels[i + 3]]
    }

    fn is_red(p: [u8; 4]) -> bool {
        p[0] > 128 && p[1] < 128 && p[2] < 128
    }

    #[test]
    fn rounded_stroke_agrees_between_the_two_paths() {
        // The §5.2 target: the old raster path stroked a radius-8 rect as a
        // sharp four-edge polyline while the SVG carried `rx`. Both readings
        // now follow the rounded boundary.
        let mut doc = VectorDoc::new(40, 40, media_vector::Rgba([1, 1, 1, 255]));
        let _ = doc.add_layer("ink");
        doc.add_shape(0, Shape::rect([10.0, 10.0], [20.0, 20.0], 8.0).stroked(ink(), 2.0));
        let norm = doc.normalize();

        let frame = render_frame(&norm);
        assert!(is_red(pixel(&frame, 20, 10)), "the top edge midpoint is on the stroke");
        assert!(!is_red(pixel(&frame, 10, 10)), "the old sharp corner is outside the rounded stroke");
        assert!(!is_red(pixel(&frame, 11, 11)), "the corner cut is rounded away (1.9px off the arc, stroke half-width is 1)");

        let svg = media_codec::to_svg_string(&norm);
        assert!(svg.contains("rx=\"8\""), "the SVG keeps the same clamped radius: {svg}");
        assert!(svg.contains("stroke-width=\"2\""), "{svg}");
    }

    #[test]
    fn sharp_rect_stroke_is_the_corner_polyline() {
        // radius 0 keeps the source behavior exactly: corners are stroked.
        let mut doc = VectorDoc::new(40, 40, media_vector::Rgba([1, 1, 1, 255]));
        let _ = doc.add_layer("ink");
        doc.add_shape(0, Shape::rect([10.0, 10.0], [20.0, 20.0], 0.0).stroked(ink(), 2.0));
        let frame = render_frame(&doc.normalize());
        assert!(is_red(pixel(&frame, 10, 10)), "a sharp corner is painted");
        assert!(is_red(pixel(&frame, 20, 10)));
    }

    #[test]
    fn transparency_agrees_between_the_two_paths() {
        let mut doc = VectorDoc::new(32, 32, media_vector::Rgba([0, 0, 0, 255]));
        let _ = doc.add_layer("half");
        doc.layer_mut(0).expect("layer").opacity = 0.5;
        doc.add_shape(0, Shape::rect([4.0, 4.0], [24.0, 24.0], 0.0).filled(ink()));
        let norm = doc.normalize();

        let frame = render_frame(&norm);
        // Exact integer arithmetic of the raster blend: red 255 at folded
        // alpha 128 over opaque black lands on (128, 0, 0, 191) — the same
        // 0.502 the SVG attribute encodes, neither path dropping the alpha.
        let center = pixel(&frame, 16, 16);
        assert_eq!(center, [128, 0, 0, 191], "the raster composites the folded alpha");

        let svg = media_codec::to_svg_string(&norm);
        assert!(svg.contains("fill=\"#ff0000\" fill-opacity=\"0.502\""), "the SVG carries the same alpha: {svg}");
    }

    #[test]
    fn hidden_layers_and_order_render_identically_in_both_paths() {
        let mut doc = VectorDoc::new(8, 8, media_vector::Rgba([0, 0, 0, 255]));
        let _ = doc.add_layer("ink");
        let _ = doc.add_layer("ghost");
        doc.layer_mut(1).expect("layer").visible = false;
        doc.add_shape(0, Shape::rect([0.0, 0.0], [8.0, 8.0], 0.0).filled(media_vector::Rgba([255, 0, 0, 255])));
        doc.add_shape(0, Shape::rect([0.0, 0.0], [8.0, 8.0], 0.0).filled(media_vector::Rgba([0, 0, 255, 255])));
        let norm = doc.normalize();

        // Later shapes win, matching the source ordering test.
        let frame = render_frame(&norm);
        assert_eq!(&frame.pixels[0..3], &[0, 0, 255]);
        let svg = media_codec::to_svg_string(&norm);
        assert_eq!(svg.matches("<rect").count(), 3, "background plus two shapes, no ghost");
        assert!(!svg.contains("ghost"), "{svg}");
        let blue = svg.rfind("fill=\"#0000ff\"").expect("blue paint");
        let red = svg.rfind("fill=\"#ff0000\"").expect("red paint");
        assert!(blue > red, "the SVG document order paints blue last");
    }

    #[test]
    fn an_empty_document_is_just_the_background() {
        let doc = VectorDoc::new(4, 3, media_vector::Rgba([7, 8, 9, 255]));
        let norm = doc.normalize();
        let frame = render_frame(&norm);
        assert_eq!((frame.width, frame.height), (4, 3));
        for i in 0..4usize * 3 {
            assert_eq!(&frame.pixels[i * 4..i * 4 + 3], &[7, 8, 9]);
        }
        let svg = media_codec::to_svg_string(&norm);
        assert!(!svg.contains("<g"), "no layers, no groups: {svg}");
        assert!(svg.contains("fill=\"#070809\""), "{svg}");
    }

    #[test]
    fn format_default_is_png_and_unknown_is_explicit() {
        assert_eq!(VectorFormat::parse(None).expect("default"), VectorFormat::Png);
        assert_eq!(VectorFormat::parse(Some("png")).expect("png"), VectorFormat::Png);
        assert_eq!(VectorFormat::parse(Some("svg")).expect("svg"), VectorFormat::Svg);
        let error = VectorFormat::parse(Some("pdf")).expect_err("unknown format rejects");
        assert_eq!(error.code(), "invalid_input");
    }

    #[test]
    fn write_vector_produces_both_artifacts_and_creates_directories() {
        let dir = std::env::temp_dir().join(format!("media-vector-export-{}", std::process::id()));
        let mut doc = VectorDoc::new(16, 16, media_vector::Rgba([255, 255, 255, 255]));
        let _ = doc.add_layer("ink");
        doc.add_shape(0, Shape::circle([8.0, 8.0], 5.0).filled(ink()));

        let png = write_vector(&doc, &dir.join("nested/out.png"), VectorFormat::Png).expect("png");
        assert_eq!(png.format, "png");
        assert!(png.bytes > 0);
        let svg = write_vector(&doc, &dir.join("nested/out.svg"), VectorFormat::Svg).expect("svg");
        assert_eq!(svg.format, "svg");
        assert!(svg.bytes > 0);
        assert_eq!(std::fs::read(&dir.join("nested/out.svg")).expect("read").len() as u64, svg.bytes);

        // The PNG round-trips through the decoder-free writer contract: the
        // signature is enough here because the pixels are asserted in the
        // dedicated path tests above.
        let png_bytes = std::fs::read(&dir.join("nested/out.png")).expect("read png");
        assert_eq!(&png_bytes[0..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    /// Rounded fill, smooth path, circle and a 0.5 layer — no rounded-rect
    /// stroke, so the absorbed raster path must reproduce the source exactly.
    fn scene_a() -> VectorDoc {
        let mut doc = VectorDoc::new(64, 48, media_vector::Rgba([10, 12, 20, 255]));
        let _ = doc.add_layer("base");
        doc.add_shape(0, Shape::rect([4.0, 4.0], [40.0, 24.0], 6.0).filled(media_vector::Rgba([30, 40, 70, 255])));
        doc.add_shape(
            0,
            Shape::path(vec![[8.0, 30.0], [24.0, 12.0], [56.0, 28.0]])
                .smooth()
                .stroked(media_vector::Rgba([255, 200, 120, 255]), 3.0),
        );
        doc.add_shape(
            0,
            Shape::circle([50.0, 36.0], 6.0)
                .filled(media_vector::Rgba([200, 60, 60, 255]))
                .stroked(media_vector::Rgba([255, 255, 255, 255]), 1.5),
        );
        let _ = doc.add_layer("faint");
        doc.layer_mut(1).expect("layer").opacity = 0.5;
        doc.add_shape(1, Shape::rect([20.0, 20.0], [30.0, 20.0], 0.0).filled(media_vector::Rgba([120, 160, 255, 255])));
        doc
    }

    /// The baseline's divergence case: a rounded rect with a stroke.
    fn scene_b() -> VectorDoc {
        let mut doc = VectorDoc::new(64, 48, media_vector::Rgba([10, 12, 20, 255]));
        let _ = doc.add_layer("base");
        doc.add_shape(
            0,
            Shape::rect([4.0, 4.0], [40.0, 24.0], 6.0)
                .filled(media_vector::Rgba([30, 40, 70, 255]))
                .stroked(media_vector::Rgba([120, 160, 255, 255]), 2.0),
        );
        doc
    }

    #[test]
    fn the_raster_path_reproduces_the_source_except_for_the_fixed_stroke() {
        // Digests measured against a verbatim port of the OpenPaint core at
        // the absorbed revision (G6a-2 evidence harness, FNV-1a 64 over the
        // RGBA8 buffer).
        assert_eq!(
            fnv1a64(&render_frame(&scene_a().normalize()).pixels),
            0x50d9_ae2e_52d3_5849,
            "scene A must byte-match the source raster path"
        );
        // The source stroked the sharp four-edge polyline (digest
        // 0xa2fc_dc22_e8b5_520f); the normalized path strokes the rounded
        // outline the SVG `rx` describes. Deliberate correction, pinned here
        // so the fixed geometry can never silently drift again.
        assert_eq!(
            fnv1a64(&render_frame(&scene_b().normalize()).pixels),
            0xd859_3ef4_cbe6_a991,
            "scene B is the corrected rounded-rect stroke"
        );
    }
}
