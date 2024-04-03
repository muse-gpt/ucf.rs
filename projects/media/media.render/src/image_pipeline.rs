//! The scene-free image pipeline: decode → process → encode, one call.
//!
//! This is the authoritative composition entry the photo and VFX hosts call
//! instead of orchestrating inside their bindings: `media.codec` decodes, one
//! `media.post` domain operation runs on the RGBA8 frame, `media.codec`
//! encodes the PNG. The output directory is prepared here, the decode budget
//! is the project-layer default, and the processing moment for time-varying
//! stacks is derived from the single rational time base. Bindings map host
//! values in and results out — nothing else.

use std::path::Path;

use media_codec::{DecodeLimits, MediaLibrary};
use media_image::Frame;
use media_post::{Adjustments, EffectContext, EffectStack};
use media_types::{Error, TimeBase};

/// Facts about a finished image pipeline run.
#[derive(serde::Serialize)]
pub struct ProcessedImage {
    pub out: String,
    pub width: u32,
    pub height: u32,
    pub bytes: u64,
    /// Applied stack names in order; empty for the develop recipe.
    pub effects: Vec<&'static str>,
}

/// What runs between decode and encode.
pub enum ImageOperation<'a> {
    /// The deterministic develop recipe (display-space).
    Develop(&'a Adjustments),
    /// One effect stack at one moment (display-space, time-varying operators
    /// derive their phase from `time_base`).
    Stack {
        stack: &'a EffectStack,
        frame: u32,
        time_base: TimeBase,
    },
}

/// Decodes `src`, applies `operation` and encodes the result as a PNG at
/// `out`, returning the observable facts of the run.
pub fn process_image(src: &Path, out: &Path, operation: &ImageOperation<'_>) -> Result<ProcessedImage, Error> {
    let src_str = src
        .to_str()
        .ok_or_else(|| Error::invalid_input("source path is not valid UTF-8"))?;
    let mut library = MediaLibrary::new();
    // The project-layer decode budget: this entry trusts container sizes;
    // hosts with tighter budgets call the codec directly instead.
    let bitmap = library.image(src_str, DecodeLimits::unbounded())?;
    let (width, height) = (bitmap.width, bitmap.height);
    let mut frame = Frame::new(width, height);
    frame.pixels.copy_from_slice(&bitmap.rgba);
    let effects = match operation {
        ImageOperation::Develop(recipe) => {
            recipe.apply(&mut frame);
            Vec::new()
        }
        ImageOperation::Stack { stack, frame: index, time_base } => {
            stack.apply(&mut frame, &EffectContext::new(*index, *time_base));
            stack.names()
        }
    };
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    media_codec::write_png(out, width, height, &frame.pixels)?;
    let bytes = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    Ok(ProcessedImage {
        out: out.display().to_string(),
        width,
        height,
        bytes,
        effects,
    })
}

#[cfg(all(test, windows))]
mod tests {
    // Image import goes through the host decoder (WIC on Windows); other
    // platforms report `unsupported`, so the end-to-end assertions live here.
    use super::*;
    use media_post::StackSpec;

    fn fixture(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("media-image-pipeline-tests");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(name);
        let (width, height) = (16u32, 8u32);
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.extend_from_slice(&[(x * 16) as u8, (y * 32) as u8, 64, 255]);
            }
        }
        media_codec::write_png(&path, width, height, &pixels).expect("fixture png");
        path
    }

    fn decode(path: &Path) -> (u32, u32, Vec<u8>) {
        let mut library = MediaLibrary::new();
        let bitmap = library
            .image(path.to_str().unwrap(), DecodeLimits::unbounded())
            .expect("decode");
        (bitmap.width, bitmap.height, bitmap.rgba.clone())
    }

    #[test]
    fn develop_round_trip_brightens_and_reports() {
        let src = fixture("develop-src.png");
        let out = std::env::temp_dir().join("media-image-pipeline-tests/out-develop.png");
        let stats = process_image(
            &src,
            &out,
            &ImageOperation::Develop(&Adjustments {
                exposure: 1.0,
                ..Adjustments::default()
            }),
        )
        .expect("develop");
        assert_eq!((stats.width, stats.height), (16, 8));
        assert!(stats.bytes > 0);
        assert!(stats.effects.is_empty());
        let (_, _, before) = decode(&src);
        let (_, _, after) = decode(&out);
        // Pixel (1,0): red 16 doubles to 32 under one stop; the pinned
        // display-space formula, end to end through codec and post.
        assert_eq!(before[4], 16);
        assert_eq!(after[4], 32);
        assert!(after[0] >= before[0]);
    }

    #[test]
    fn stack_round_trip_reports_applied_names() {
        let src = fixture("stack-src.png");
        let out = std::env::temp_dir().join("media-image-pipeline-tests/nested/deep/out-stack.png");
        let stack = StackSpec::from_json(
            r##"{"effects":[{"kind":"chroma","amount":0.05},{"kind":"grain","strength":0.1,"density":1000}]}"##,
        )
        .expect("spec")
        .compile()
        .expect("stack");
        let stats = process_image(
            &src,
            &out,
            &ImageOperation::Stack {
                stack: &stack,
                frame: 2,
                time_base: TimeBase::from_fps(30),
            },
        )
        .expect("stack run");
        assert_eq!(stats.effects, vec!["chroma", "grain"]);
        assert_eq!((stats.width, stats.height), (16, 8));
        assert!(out.exists(), "nested output directories are created");
    }

    #[test]
    fn the_moment_moves_a_time_varying_stack() {
        let src = fixture("moment-src.png");
        let stack = media_post::preset("fireside").expect("preset");
        let render = |frame: u32| {
            let out = std::env::temp_dir().join(format!("media-image-pipeline-tests/moment-{frame}.png"));
            process_image(
                &src,
                &out,
                &ImageOperation::Stack {
                    stack: &stack,
                    frame,
                    time_base: TimeBase::from_fps(30),
                },
            )
            .expect("render moment");
            decode(&out).2
        };
        assert_eq!(render(0), render(0), "the same moment reproduces");
        assert_ne!(render(0), render(45), "the moment drives the phase");
    }

    #[test]
    fn a_missing_source_is_an_explicit_media_error() {
        let error = process_image(
            Path::new("no-such-file.png"),
            &std::env::temp_dir().join("media-image-pipeline-tests/none.png"),
            &ImageOperation::Develop(&Adjustments::default()),
        )
        .err()
        .expect("decode fails");
        assert_eq!(error.code(), "media");
    }
}
