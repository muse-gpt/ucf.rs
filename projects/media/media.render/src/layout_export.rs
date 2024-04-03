//! Layout export: the authoritative layout-plus-raster-plus-text-plus-codec
//! composition.
//!
//! A [`media_layout::ComicSpec`] resolves through `media.layout` into the
//! document model, a page plan or a contact-sheet plan is built with explicit
//! font and image resources, the plan is lowered through `media.raster` and
//! the `paint` layer of `media.text`, and the frame is encoded through
//! `media.codec` as PNG. The page and the overview are the two artifacts of
//! one deterministic pipeline: same spec, same font bytes, same image bytes,
//! same pixels, byte for byte.
//!
//! File IO and font decoding are gated behind the `layout-export` feature,
//! mirroring the scene, sfx, image-pipeline, vector-export and voxel-export
//! entries. Resources are always explicit inputs: this entry never probes a
//! system path for fonts and never opens a panel image by itself — hosts
//! inject font bytes and an [`media_layout::ImageStore`] (the codec-backed
//! [`MediaImages`] is provided for path-style artwork).

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use media_codec::{DecodeLimits, MediaLibrary};
use media_image::{Bitmap, Frame};
use media_layout::{ComicDoc, ComicSpec, DrawOp, ImageStore, PagePlan, SheetPlan, page_plan, sheet_plan};
use media_raster::Paint;
use media_text::Fonts;
use media_types::Error;

/// Explicit font bytes for the three roles `media.text` rasterizes through.
/// There is no discovery and no default: a host that wants system fonts reads
/// the files itself and passes the bytes here.
#[derive(Clone, Copy, Debug)]
pub struct FontAssets<'a> {
    pub regular: &'a [u8],
    pub bold: &'a [u8],
    pub mono: &'a [u8],
}

/// Everything the pipeline pulls from the host: font bytes and an image store.
pub struct LayoutAssets<'a> {
    pub fonts: FontAssets<'a>,
    pub images: &'a mut dyn ImageStore,
}

/// An [`ImageStore`] over the codec's [`MediaLibrary`]: panel image keys are
/// file paths decoded through the host codecs, cached per store, and a
/// missing or undecodable resource fails the whole page with the source's
/// `image {path}: …` message shape.
pub struct MediaImages<'a> {
    library: &'a mut MediaLibrary,
    limits: DecodeLimits,
    cache: HashMap<String, Rc<Bitmap>>,
}

impl<'a> MediaImages<'a> {
    /// Wraps a library with the unbounded decode budget, matching the
    /// historical per-render `MediaLibrary` behaviour.
    #[must_use]
    pub fn new(library: &'a mut MediaLibrary) -> Self {
        Self::with_limits(library, DecodeLimits::unbounded())
    }

    #[must_use]
    pub fn with_limits(library: &'a mut MediaLibrary, limits: DecodeLimits) -> Self {
        Self { library, limits, cache: HashMap::new() }
    }
}

impl ImageStore for MediaImages<'_> {
    fn resolve(&mut self, key: &str) -> Result<&Bitmap, Error> {
        if !self.cache.contains_key(key) {
            let bitmap =
                self.library.image(key, self.limits).map_err(|e| Error::io(format!("image {key}: {e}")))?;
            self.cache.insert(key.to_string(), bitmap);
        }
        Ok(&self.cache[key])
    }
}

/// Facts about a finished layout export run.
#[derive(Debug, serde::Serialize)]
pub struct ExportedPage {
    pub out: String,
    pub width: u32,
    pub height: u32,
    pub bytes: u64,
}

/// Lowers one page plan through `media.raster` and `media.text`.
///
/// Every sizing, anchoring and tail decision already happened in the
/// `media.layout` plan, so this function only dispatches resolved operations
/// onto the `Paint` trait and the line drawer, exactly as the absorbed paint
/// loops did.
pub fn render_page(plan: &PagePlan, fonts: &mut Fonts, images: &mut dyn ImageStore) -> Result<Frame, Error> {
    let mut frame = Frame::new(plan.width, plan.height);
    frame.fill(plan.background);
    for op in &plan.ops {
        match op {
            DrawOp::FillRect { rect, color } => frame.rect(rect[0], rect[1], rect[2], rect[3], *color),
            DrawOp::FillRoundRect { rect, radius, color } => frame.round_rect(rect[0], rect[1], rect[2], rect[3], *radius, *color),
            DrawOp::StrokePolyline { points, width, color } => frame.stroke_polyline(points, *width, *color, 1.0, false),
            DrawOp::FillPolygon { points, color } => frame.fill_polygon(points, *color, false),
            DrawOp::Image { key, rect, fit } => {
                let bitmap = images.resolve(key)?;
                frame.draw_bitmap(bitmap, rect[0], rect[1], rect[2], rect[3], *fit, 1.0, 0.0);
            }
            DrawOp::TextLine { text, x, baseline, size, font, color } => fonts.draw_line(&mut frame, text, *x, *baseline, *size, *font, *color),
        }
    }
    Ok(frame)
}

/// Paints the whole contact sheet: each cell renders its page through the
/// same plan-and-lower path, scaled into the cell and labelled like the
/// source shot card.
pub fn render_overview(comic: &ComicDoc, plan: &SheetPlan, fonts: &mut Fonts, images: &mut dyn ImageStore) -> Result<Frame, Error> {
    let mut sheet = Frame::new(plan.width, plan.height);
    sheet.fill(plan.background);
    for cell in &plan.cells {
        let page = page_plan(comic, cell.page, fonts, images)?;
        let frame = render_page(&page, fonts, images)?;
        let bitmap = Bitmap::new(frame.width, frame.height, frame.pixels);
        sheet.draw_bitmap(&bitmap, cell.x as f32, cell.y as f32, cell.width as f32, cell.height as f32, media_image::Fit::Stretch, 1.0, 0.0);
        fonts.draw_line(
            &mut sheet,
            &cell.label,
            cell.x as f32 + plan.label_offset_x,
            cell.y as f32 + plan.label_size as f32,
            plan.label_size as f32,
            plan.label_font,
            plan.label_color,
        );
    }
    Ok(sheet)
}

/// Resolves the spec, builds the page plan and writes one PNG.
///
/// The input contract is validated before any font byte is parsed, so a
/// malformed colour reports as `invalid_input` regardless of the assets.
pub fn write_page(spec: &ComicSpec, page: usize, assets: &mut LayoutAssets, out: &Path) -> Result<ExportedPage, Error> {
    let comic = spec.resolve()?;
    let mut fonts = Fonts::from_bytes(assets.fonts.regular, assets.fonts.bold, assets.fonts.mono)?;
    let plan = page_plan(&comic, page, &mut fonts, &mut *assets.images)?;
    let frame = render_page(&plan, &mut fonts, &mut *assets.images)?;
    let bytes = export_png(&frame, out)?;
    Ok(ExportedPage { out: out.display().to_string(), width: frame.width, height: frame.height, bytes })
}

/// Resolves the spec, lays out the numbered overview and writes one PNG.
pub fn write_overview(spec: &ComicSpec, scale: f32, cols: usize, assets: &mut LayoutAssets, out: &Path) -> Result<ExportedPage, Error> {
    let comic = spec.resolve()?;
    let plan = sheet_plan(&comic, scale, cols)?;
    let mut fonts = Fonts::from_bytes(assets.fonts.regular, assets.fonts.bold, assets.fonts.mono)?;
    let frame = render_overview(&comic, &plan, &mut fonts, &mut *assets.images)?;
    let bytes = export_png(&frame, out)?;
    Ok(ExportedPage { out: out.display().to_string(), width: frame.width, height: frame.height, bytes })
}

fn export_png(frame: &Frame, out: &Path) -> Result<u64, Error> {
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    media_codec::write_png(out, frame.width, frame.height, &frame.pixels)?;
    Ok(std::fs::metadata(out).map(|m| m.len()).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_layout::{ComicSpec, ImageStore};

    /// The three faces the source `Fonts::load` picked on this machine. The
    /// digest goldens bind to these exact files so the comparison with the
    /// absorbed render path is byte-for-byte, not "some installed font".
    fn system_font_bytes() -> Option<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        let read = |path: &str| std::fs::read(path).ok();
        Some((
            read("C:\\Windows\\Fonts\\msyh.ttc")?,
            read("C:\\Windows\\Fonts\\msyhbd.ttc")?,
            read("C:\\Windows\\Fonts\\consola.ttf")?,
        ))
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    // Measured by the G6c-2 evidence harness, where the verbatim port of the
    // OpenComic render path and this pipeline produced byte-equal pixels and
    // byte-equal PNG files on the pinned font files.
    const PAGE0_DIGEST: u64 = 0x0ba2_5d0b_8ff3_1025;
    const PAGE1_DIGEST: u64 = 0x5a3b_3044_3a4f_e69c;
    const PAGE2_DIGEST: u64 = 0x14bb_7e4c_6e3f_2af8;
    const SHEET_DIGEST: u64 = 0xb666_8e1e_af0b_e5cf;

    /// The source test document: two half panels, a bordered panel, one
    /// caption and one balloon.
    fn source_test_doc() -> ComicSpec {
        serde_json::from_str(
            r##"{"width":400,"height":300,"background":"#fafafc","pages":[
                {"panels":[
                    {"rect":[0.0,0.0,200.0,300.0],"fill":"#b42828"},
                    {"rect":[200.0,0.0,200.0,300.0],"fill":"#2828b4"}
                ]},
                {"panels":[
                    {"rect":[20.0,20.0,100.0,80.0],"fill":"#ffffff","border":{"color":"#141418","width":3.0}}
                ]},
                {"panels":[
                    {"rect":[20.0,20.0,360.0,260.0],"texts":[
                        {"kind":"caption","text":"EST. NIGHT EXTERIOR","at":[30.0,30.0]},
                        {"kind":"balloon","text":"It works.","at":[200.0,150.0],"tail":[200.0,240.0]}
                    ]}
                ]}
            ]}"##,
        )
        .expect("test doc parses")
    }

    fn render_all_pages(spec: &ComicSpec, fonts_bytes: &(Vec<u8>, Vec<u8>, Vec<u8>)) -> Vec<Frame> {
        let comic = spec.resolve().expect("test doc resolves");
        let mut fonts = Fonts::from_bytes(&fonts_bytes.0, &fonts_bytes.1, &fonts_bytes.2).expect("fonts");
        let mut store = media_codec::MediaLibrary::new();
        let mut images = MediaImages::new(&mut store);
        (0..comic.pages.len())
            .map(|page| {
                let plan = page_plan(&comic, page, &mut fonts, &mut images).expect("plan");
                render_page(&plan, &mut fonts, &mut images).expect("frame")
            })
            .collect()
    }

    #[test]
    fn malformed_input_rejects_before_any_font_is_parsed() {
        // Garbage font bytes prove the order: the colour error must surface.
        let spec: ComicSpec =
            serde_json::from_str(r##"{"width":10,"height":10,"background":"#zzz"}"##).expect("parses");
        let garbage = [0u8; 8];
        let mut store = media_codec::MediaLibrary::new();
        let mut images = MediaImages::new(&mut store);
        let mut assets = LayoutAssets {
            fonts: FontAssets { regular: &garbage, bold: &garbage, mono: &garbage },
            images: &mut images,
        };
        let error = write_page(&spec, 0, &mut assets, Path::new("unused.png")).expect_err("bad colour rejects");
        assert_eq!(error.code(), "invalid_input");
    }

    #[test]
    fn undecodable_font_bytes_are_an_explicit_io_error() {
        let spec: ComicSpec = serde_json::from_str(r#"{"width":10,"height":10}"#).expect("parses");
        let garbage = [0u8; 8];
        let mut store = media_codec::MediaLibrary::new();
        let mut images = MediaImages::new(&mut store);
        let mut assets = LayoutAssets {
            fonts: FontAssets { regular: &garbage, bold: &garbage, mono: &garbage },
            images: &mut images,
        };
        let error = write_page(&spec, 0, &mut assets, Path::new("unused.png")).expect_err("bad font rejects");
        assert_eq!(error.code(), "io");
        assert!(error.message.contains("regular font"), "{}", error.message);
    }

    #[test]
    fn a_missing_panel_image_fails_the_whole_page() {
        // The codec-backed store against a path that cannot exist.
        let spec: ComicSpec = serde_json::from_str(
            r#"{"width":10,"height":10,"pages":[{"panels":[{"rect":[0.0,0.0,10.0,10.0],"image":"E:/no-such-opencomic-panel.png","fit":"cover"}]}]}"#,
        )
        .expect("parses");
        let Some(fonts_bytes) = system_font_bytes() else {
            eprintln!("SKIP: pinned system fonts absent");
            return;
        };
        let mut store = media_codec::MediaLibrary::new();
        let mut images = MediaImages::new(&mut store);
        let mut assets = LayoutAssets {
            fonts: FontAssets {
                regular: &fonts_bytes.0,
                bold: &fonts_bytes.1,
                mono: &fonts_bytes.2,
            },
            images: &mut images,
        };
        let error = write_page(&spec, 0, &mut assets, Path::new("unused.png")).expect_err("broken path rejects");
        assert_eq!(error.code(), "io");
        assert!(error.message.contains("image E:/no-such-opencomic-panel.png"), "{}", error.message);
    }

    #[test]
    fn the_page_pixels_byte_match_the_absorbed_source() {
        // Digests measured against a verbatim port of the OpenComic render
        // path at the absorbed revision (G6c-2 evidence harness, FNV-1a 64
        // over the RGBA8 buffer, same pinned font files). The old pipeline
        // painted directly through `media.raster`; the new one lowers the
        // `media.layout` plan — the pictures must not drift.
        let Some(fonts_bytes) = system_font_bytes() else {
            eprintln!("SKIP: pinned system fonts absent");
            return;
        };
        let frames = render_all_pages(&source_test_doc(), &fonts_bytes);
        assert_eq!(fnv1a64(&frames[0].pixels), PAGE0_DIGEST, "two half panels");
        assert_eq!(fnv1a64(&frames[1].pixels), PAGE1_DIGEST, "bordered panel");
        assert_eq!(fnv1a64(&frames[2].pixels), PAGE2_DIGEST, "caption and balloon");
    }

    #[test]
    fn the_overview_pixels_byte_match_the_absorbed_source() {
        let Some(fonts_bytes) = system_font_bytes() else {
            eprintln!("SKIP: pinned system fonts absent");
            return;
        };
        let comic = source_test_doc().resolve().expect("doc");
        let plan = sheet_plan(&comic, 0.5, 2).expect("sheet plan");
        assert_eq!((plan.width, plan.height), (16 * 3 + 200 * 2, 16 * 3 + 150 * 2));
        let mut fonts = Fonts::from_bytes(&fonts_bytes.0, &fonts_bytes.1, &fonts_bytes.2).expect("fonts");
        let mut store = media_codec::MediaLibrary::new();
        let mut images = MediaImages::new(&mut store);
        let sheet = render_overview(&comic, &plan, &mut fonts, &mut images).expect("sheet");
        assert_eq!(fnv1a64(&sheet.pixels), SHEET_DIGEST, "numbered overview");
    }

    #[test]
    fn the_pipeline_is_self_deterministic() {
        let Some(fonts_bytes) = system_font_bytes() else {
            eprintln!("SKIP: pinned system fonts absent");
            return;
        };
        let a = render_all_pages(&source_test_doc(), &fonts_bytes);
        let b = render_all_pages(&source_test_doc(), &fonts_bytes);
        for (page, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(fnv1a64(&x.pixels), fnv1a64(&y.pixels), "page {page} drifted");
        }
    }

    #[test]
    fn write_page_produces_both_artifacts_and_creates_directories() {
        let Some(fonts_bytes) = system_font_bytes() else {
            eprintln!("SKIP: pinned system fonts absent");
            return;
        };
        let dir = std::env::temp_dir().join(format!("media-layout-export-{}", std::process::id()));
        let mut store = media_codec::MediaLibrary::new();
        let mut images = MediaImages::new(&mut store);
        let mut assets = LayoutAssets {
            fonts: FontAssets { regular: &fonts_bytes.0, bold: &fonts_bytes.1, mono: &fonts_bytes.2 },
            images: &mut images,
        };
        let page = write_page(&source_test_doc(), 2, &mut assets, &dir.join("nested/page.png")).expect("page");
        assert_eq!((page.width, page.height), (400, 300));
        assert!(page.bytes > 0);
        let sheet = write_overview(&source_test_doc(), 0.5, 2, &mut assets, &dir.join("nested/sheet.png")).expect("sheet");
        assert_eq!((sheet.width, sheet.height), (16 * 3 + 200 * 2, 16 * 3 + 150 * 2));
        for name in ["page.png", "sheet.png"] {
            let bytes = std::fs::read(dir.join("nested").join(name)).expect("read");
            assert_eq!(&bytes[0..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A], "{name}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn media_images_surfaces_the_codec_error_shape() {
        let mut store = media_codec::MediaLibrary::new();
        let mut images = MediaImages::new(&mut store);
        let error = images.resolve("no-such-file.png").err().expect("absent file rejects");
        assert_eq!(error.code(), "io");
        assert!(error.message.starts_with("image no-such-file.png: "), "{}", error.message);
    }
}
