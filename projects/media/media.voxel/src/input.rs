//! The host input contract: serde shape, defaults and validation.
//!
//! These types absorb the OpenVoxel N-API's `SceneDto`/`VoxelDto` defaults
//! — width `960`, height `640`, background `#0a0e1a` — with the wire shape
//! unchanged. The deliberate correction: the binding's silent fallback of a
//! malformed voxel colour to light grey is now an explicit
//! [`media_types::Error::invalid_input`].
//!
//! Bindings extract host values, deserialize a [`VoxelSpec`] and call
//! [`VoxelSpec::grid`]; they must not re-implement defaults or validation.

use geometry::{Rgba, parse};
use media_types::Error;
use serde::Deserialize;

use crate::mesh::VoxelGrid;

/// Preview width used when the host omits one. Absorbed from the binding.
pub const DEFAULT_WIDTH: u32 = 960;
/// Preview height used when the host omits one. Absorbed from the binding.
pub const DEFAULT_HEIGHT: u32 = 640;
/// Preview background used when the host omits one. Absorbed from the binding.
pub const DEFAULT_BACKGROUND: Rgba = Rgba([10, 14, 26, 255]);

/// A whole scene as the host sends it.
#[derive(Deserialize, Debug)]
pub struct VoxelSpec {
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub background: Option<String>,
    #[serde(default)]
    pub voxels: Vec<VoxelEntry>,
}

/// One voxel as the host sends it. Both fields are required, matching the
/// historical DTO: a missing position or colour fails deserialization.
#[derive(Deserialize, Debug)]
pub struct VoxelEntry {
    pub at: [i32; 3],
    pub color: String,
}

impl VoxelSpec {
    /// Builds the grid. A malformed hex colour is an explicit error,
    /// replacing the binding's silent light-grey fallback.
    pub fn grid(&self) -> Result<VoxelGrid, Error> {
        let mut grid = VoxelGrid::new();
        for entry in &self.voxels {
            let color = parse(&entry.color).ok_or_else(|| {
                Error::invalid_input(format!(
                    "voxel at [{}, {}, {}]: invalid color '{}'",
                    entry.at[0], entry.at[1], entry.at[2], entry.color
                ))
            })?;
            grid.set(entry.at, color);
        }
        Ok(grid)
    }

    /// Resolved preview width.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.width.unwrap_or(DEFAULT_WIDTH)
    }

    /// Resolved preview height.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.height.unwrap_or(DEFAULT_HEIGHT)
    }

    /// Resolved preview background. A present-but-malformed colour is an
    /// explicit error instead of the binding's silent fallback.
    pub fn background(&self) -> Result<Rgba, Error> {
        match self.background.as_deref() {
            None => Ok(DEFAULT_BACKGROUND),
            Some(text) => parse(text).ok_or_else(|| Error::invalid_input(format!("background: invalid color '{text}'"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(json: &str) -> VoxelSpec {
        serde_json::from_str(json).expect("input parses")
    }

    #[test]
    fn minimal_scene_takes_the_absorbed_defaults() {
        let parsed = spec("{}");
        assert_eq!(parsed.width(), DEFAULT_WIDTH);
        assert_eq!(parsed.height(), DEFAULT_HEIGHT);
        assert_eq!(parsed.background().expect("default background"), Rgba([10, 14, 26, 255]));
        assert!(parsed.grid().expect("empty grid").is_empty());
    }

    #[test]
    fn wire_shape_matches_the_historical_binding_dto() {
        let parsed = spec(
            r##"{"width":256,"height":256,"background":"#10141c","voxels":[
                {"at":[0,0,0],"color":"#78a0ff"},
                {"at":[-1,2,-3],"color":"#ffc878"}
            ]}"##,
        );
        assert_eq!((parsed.width(), parsed.height()), (256, 256));
        assert_eq!(parsed.background().expect("background"), Rgba([16, 20, 28, 255]));
        let grid = parsed.grid().expect("build");
        assert_eq!(grid.len(), 2);
        assert_eq!(grid.get([0, 0, 0]), Some(Rgba([120, 160, 255, 255])));
        assert_eq!(grid.get([-1, 2, -3]), Some(Rgba([255, 200, 120, 255])));
    }

    #[test]
    fn malformed_voxel_color_is_an_explicit_error() {
        // The deliberate correction: the binding fell back to light grey.
        let parsed = spec(r#"{"voxels":[{"at":[0,0,0],"color":"nope"}]}"#);
        let error = parsed.grid().expect_err("bad color rejects");
        assert_eq!(error.code(), "invalid_input");
        assert!(error.message.contains("invalid color 'nope'"), "{}", error.message);
        assert!(error.message.contains("voxel at [0, 0, 0]"), "{}", error.message);
    }

    #[test]
    fn malformed_background_is_an_explicit_error() {
        let parsed = spec(r##"{"background":"#zzz"}"##);
        let error = parsed.background().expect_err("bad background rejects");
        assert_eq!(error.code(), "invalid_input");
        assert!(error.message.contains("background"), "{}", error.message);
    }

    #[test]
    fn missing_voxel_fields_fail_deserialization() {
        assert!(serde_json::from_str::<VoxelSpec>(r#"{"voxels":[{"at":[0,0,0]}]}"#).is_err());
        assert!(serde_json::from_str::<VoxelSpec>(r##"{"voxels":[{"color":"#fff"}]}"##).is_err());
    }

    #[test]
    fn duplicate_cells_last_write_wins_like_the_binding() {
        let parsed = spec(r##"{"voxels":[{"at":[1,1,1],"color":"#ff0000"},{"at":[1,1,1],"color":"#0000ff"}]}"##);
        let grid = parsed.grid().expect("build");
        assert_eq!(grid.len(), 1);
        assert_eq!(grid.get([1, 1, 1]), Some(Rgba([0, 0, 255, 255])));
    }
}
