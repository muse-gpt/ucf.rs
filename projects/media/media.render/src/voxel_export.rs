//! Voxel export: the authoritative voxel-plus-raster-plus-codec composition.
//!
//! A [`VoxelSpec`] is resolved by `media.voxel` into a grid, then driven
//! through exactly one of two encoders over the same exposed-face list: the
//! PNG path paints the preview plan with `media.raster` onto `media.image`
//! storage and encodes through `media.codec`, the OBJ path encodes the face
//! list through the codec's `obj` feature. Because both readings consume the
//! one deterministic face sequence, the source promise that the preview and
//! the mesh never disagree holds across the ownership split.
//!
//! The source binding wrote the PNG always and the OBJ only when the host
//! passed a second path; the composed entry picks one artifact per call and
//! hosts that want both call it twice over the same grid. File IO is gated
//! behind the `voxel-export` feature, mirroring the scene, sfx,
//! image-pipeline and vector-export entries. The preview defaults, the
//! background and the shading strategy are resolved here, not in a binding.

use std::path::Path;

use media_image::Frame;
use media_raster::Paint;
use media_types::Error;
use media_voxel::{Shading, VoxelPreview, VoxelSpec, faces, preview};

/// Export format. An absent host spelling means PNG, the artifact the
/// source always wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxelFormat {
    Png,
    Obj,
}

impl VoxelFormat {
    /// Resolves the host spelling. Anything unknown is an explicit
    /// `invalid_input` error instead of a silent fallback.
    pub fn parse(name: Option<&str>) -> Result<Self, Error> {
        match name {
            None | Some("png") => Ok(Self::Png),
            Some("obj") => Ok(Self::Obj),
            Some(other) => Err(Error::invalid_input(format!("voxel export: unknown format '{other}'"))),
        }
    }

    /// The stable spelling, the same as the serde discriminant.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Png => "png",
            Self::Obj => "obj",
        }
    }
}

/// Facts about a finished voxel export run.
#[derive(Debug, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportedVoxel {
    pub out: String,
    pub format: &'static str,
    pub bytes: u64,
}

/// Paints the preview plan through `media.raster` into RGBA8 storage.
///
/// This is the raster half of the shared-face-list contract: every
/// projection, fit, sort and shading decision already happened in the
/// `media.voxel` plan, so this function only dispatches resolved quads onto
/// the `Paint` trait, exactly as the absorbed preview loop did.
#[must_use]
pub fn render_preview(plan: &VoxelPreview) -> Frame {
    let mut frame = Frame::new(plan.width, plan.height);
    frame.fill(plan.background);
    for polygon in &plan.polygons {
        frame.fill_polygon(&polygon.points, polygon.color, false);
    }
    frame
}

/// Exports the scene with the default [`Shading::FixedSixDir`] preset.
pub fn write_voxel(spec: &VoxelSpec, out: &Path, format: VoxelFormat) -> Result<ExportedVoxel, Error> {
    write_voxel_shaded(spec, out, format, Shading::default())
}

/// Exports the scene under an explicit shading strategy.
pub fn write_voxel_shaded(spec: &VoxelSpec, out: &Path, format: VoxelFormat, shading: Shading) -> Result<ExportedVoxel, Error> {
    let grid = spec.grid()?;
    let background = spec.background()?;
    if let Some(parent) = out.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    match format {
        VoxelFormat::Obj => {
            std::fs::write(out, media_codec::to_obj_string(&faces(&grid)).as_bytes())?;
        }
        VoxelFormat::Png => {
            let plan = preview(&grid, spec.width(), spec.height(), background, shading);
            let frame = render_preview(&plan);
            media_codec::write_png(out, frame.width, frame.height, &frame.pixels)?;
        }
    }
    let bytes = std::fs::metadata(out).map(|m| m.len()).unwrap_or(0);
    Ok(ExportedVoxel { out: out.display().to_string(), format: format.as_str(), bytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_voxel::VoxelEntry;

    fn spec(width: u32, height: u32, background: &str, voxels: &[([i32; 3], &str)]) -> VoxelSpec {
        VoxelSpec {
            width: Some(width),
            height: Some(height),
            background: Some(background.to_string()),
            voxels: voxels.iter().map(|(at, color)| VoxelEntry { at: *at, color: (*color).to_string() }).collect(),
        }
    }

    fn lone_cube() -> VoxelSpec {
        spec(64, 64, "#000000", &[([0, 0, 0], "#c8783c")])
    }

    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    fn painted(spec: &VoxelSpec, shading: Shading) -> Frame {
        let grid = spec.grid().expect("grid");
        let plan = preview(&grid, spec.width(), spec.height(), spec.background().expect("background"), shading);
        render_preview(&plan)
    }

    #[test]
    fn the_preview_pixels_byte_match_the_absorbed_source() {
        // Digests measured against a verbatim port of the OpenVoxel core at
        // the absorbed revision (G6b-2 evidence harness, FNV-1a 64 over the
        // RGBA8 buffer). The shading preset is the only brightness source
        // and the default resolves to it, so the picture cannot drift.
        assert_eq!(fnv1a64(&painted(&lone_cube(), Shading::default()).pixels), 0xb9e4_fec8_5e40_d7f4, "lone cube at 64x64");
        let bar = spec(128, 128, "#000000", &[([-7, -3, 5], "#0c2238"), ([-6, -3, 5], "#0c2238")]);
        assert_eq!(fnv1a64(&painted(&bar, Shading::default()).pixels), 0x0a4c_0d15_a5c2_146d, "negative-coordinate bar");
        let cluster = spec(
            256,
            256,
            "#10141c",
            &[
                ([0, 0, 0], "#60a874"),
                ([1, 0, 0], "#60a874"),
                ([0, 0, 1], "#60a874"),
                ([1, 0, 1], "#60a874"),
                ([0, 1, 0], "#60a874"),
                ([1, 1, 0], "#ffffff"),
                ([0, 1, 1], "#60a874"),
            ],
        );
        assert_eq!(fnv1a64(&painted(&cluster, Shading::default()).pixels), 0x5e66_b14b_47be_7c05, "smoke-like cluster");
    }

    #[test]
    fn the_shading_parameter_changes_the_picture_and_the_default_does_not() {
        // The strategy is a real render parameter: a uniform multiplier
        // paints different bytes, while the default preset keeps every
        // digest above valid.
        let default = painted(&lone_cube(), Shading::default());
        let flat = painted(&lone_cube(), Shading::Uniform(0.5));
        assert_ne!(fnv1a64(&default.pixels), fnv1a64(&flat.pixels));
        assert_eq!(fnv1a64(&default.pixels), 0xb9e4_fec8_5e40_d7f4);
    }

    #[test]
    fn the_preview_paints_pixels() {
        // The source test, kept.
        let frame = painted(&lone_cube(), Shading::default());
        let lit = frame.pixels.chunks(4).filter(|p| p[0] > 20).count();
        assert!(lit > 50, "only {lit} lit pixels");
    }

    #[test]
    fn an_empty_scene_is_just_background() {
        let frame = painted(&spec(32, 32, "#07090c", &[]), Shading::default());
        assert!(frame.pixels.chunks(4).all(|p| p[0] == 7 && p[1] == 9 && p[2] == 12));
    }

    #[test]
    fn obj_and_png_come_from_the_same_face_list() {
        // The cross-encoder half of the never-disagree promise: the OBJ
        // `f` lines and the preview polygons are the same quads.
        let cluster = spec(
            256,
            256,
            "#10141c",
            &[
                ([0, 0, 0], "#60a874"),
                ([1, 0, 0], "#60a874"),
                ([0, 0, 1], "#60a874"),
                ([1, 0, 1], "#60a874"),
                ([0, 1, 0], "#60a874"),
                ([1, 1, 0], "#ffffff"),
                ([0, 1, 1], "#60a874"),
            ],
        );
        let grid = cluster.grid().expect("grid");
        let obj = media_codec::to_obj_string(&faces(&grid));
        let plan = preview(&grid, 256, 256, cluster.background().expect("background"), Shading::default());
        assert_eq!(obj.lines().filter(|line| line.starts_with("f ")).count(), plan.polygons.len());
    }

    #[test]
    fn format_default_is_png_and_unknown_is_explicit() {
        assert_eq!(VoxelFormat::parse(None).expect("default"), VoxelFormat::Png);
        assert_eq!(VoxelFormat::parse(Some("png")).expect("png"), VoxelFormat::Png);
        assert_eq!(VoxelFormat::parse(Some("obj")).expect("obj"), VoxelFormat::Obj);
        let error = VoxelFormat::parse(Some("stl")).expect_err("unknown format rejects");
        assert_eq!(error.code(), "invalid_input");
    }

    #[test]
    fn write_voxel_produces_both_artifacts_and_creates_directories() {
        let dir = std::env::temp_dir().join(format!("media-voxel-export-{}", std::process::id()));
        let png = write_voxel(&lone_cube(), &dir.join("nested/out.png"), VoxelFormat::Png).expect("png");
        assert_eq!(png.format, "png");
        assert!(png.bytes > 0);
        let obj = write_voxel(&lone_cube(), &dir.join("nested/out.obj"), VoxelFormat::Obj).expect("obj");
        assert_eq!(obj.format, "obj");

        // The deliberate artifact change recorded by the contract: no brand
        // header, the document starts at the first vertex.
        let obj_bytes = std::fs::read(dir.join("nested/out.obj")).expect("read obj");
        assert!(obj_bytes.starts_with(b"v 0 1 0\n"), "{}", String::from_utf8_lossy(&obj_bytes));
        assert!(!obj_bytes.contains(&b'#'), "no header of any kind");
        assert_eq!(obj.bytes as usize, obj_bytes.len());

        let png_bytes = std::fs::read(dir.join("nested/out.png")).expect("read png");
        assert_eq!(&png_bytes[0..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_voxel_propagates_the_input_contract_errors() {
        let dir = std::env::temp_dir().join(format!("media-voxel-export-err-{}", std::process::id()));
        let bad = spec(64, 64, "#000000", &[([0, 0, 0], "nope")]);
        let error = write_voxel(&bad, &dir.join("out.png"), VoxelFormat::Png).expect_err("bad color rejects");
        assert_eq!(error.code(), "invalid_input");
        let bad_background = spec(64, 64, "#zzz", &[]);
        let error = write_voxel(&bad_background, &dir.join("out.png"), VoxelFormat::Png).expect_err("bad background rejects");
        assert_eq!(error.code(), "invalid_input");
    }
}
