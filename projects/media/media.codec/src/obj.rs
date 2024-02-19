//! Wavefront OBJ encoding of a voxel mesh view.
//!
//! The encoder consumes the ordered quad sequence `media.voxel` produces
//! and adds no domain semantics of its own: one `v` line per unique corner
//! in first-seen order, one `f` line per quad with 1-based indices carrying
//! the winding the face already has. Vertex identity is the exact bit
//! pattern of the corner, so the encoder never merges two positions it
//! cannot prove equal.
//!
//! Two deliberate corrections against the absorbed writer:
//! - the brand comment line is gone — the artifact now starts at the first
//!   `v` line, which is the only output change by design;
//! - the old `unwrap_or(&1)` index fallback is gone — the single-pass
//!   construction assigns every corner before its face is emitted, so a
//!   lookup can no longer silently point at vertex one.

use std::collections::BTreeMap;

use media_types::Error;
use media_voxel::Face;

/// Writes the mesh view as an OBJ file, returning the byte count.
pub fn write_obj(faces: &[Face], path: &std::path::Path) -> Result<u64, Error> {
    let obj = to_obj_string(faces);
    std::fs::write(path, obj.as_bytes())?;
    Ok(obj.len() as u64)
}

/// Renders the mesh view as an OBJ string.
#[must_use]
pub fn to_obj_string(faces: &[Face]) -> String {
    let mut index: BTreeMap<[u32; 3], usize> = BTreeMap::new();
    let mut corners: Vec<[f32; 3]> = Vec::new();
    let mut quads: Vec<[usize; 4]> = Vec::with_capacity(faces.len());
    for face in faces {
        let mut slots = [0usize; 4];
        for (position, corner) in face.corners.iter().enumerate() {
            let key = corner.map(f32::to_bits);
            let slot = *index.entry(key).or_insert_with(|| {
                corners.push(*corner);
                corners.len()
            });
            slots[position] = slot;
        }
        quads.push(slots);
    }
    let mut out = String::new();
    for corner in &corners {
        out.push_str(&format!("v {} {} {}\n", corner[0], corner[1], corner[2]));
    }
    for slots in &quads {
        out.push_str(&format!("f {} {} {} {}\n", slots[0], slots[1], slots[2], slots[3]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use media_voxel::{Rgba, Shading, VoxelGrid, faces, preview};

    fn cube() -> VoxelGrid {
        let mut grid = VoxelGrid::new();
        grid.set([0, 0, 0], Rgba([200, 120, 60, 255]));
        grid
    }

    #[test]
    fn the_lone_cube_is_the_locked_full_text() {
        // Byte-for-byte golden: vertices in first-seen order over the fixed
        // face sequence, quads with 1-based indices in the face winding.
        let expected = "v 0 1 0\n\
            v 1 1 0\n\
            v 1 1 1\n\
            v 0 1 1\n\
            v 0 0 0\n\
            v 0 0 1\n\
            v 1 0 1\n\
            v 1 0 0\n\
            f 1 2 3 4\n\
            f 5 6 7 8\n\
            f 8 2 3 7\n\
            f 6 4 1 5\n\
            f 7 3 4 6\n\
            f 5 1 2 8\n";
        assert_eq!(to_obj_string(&faces(&cube())), expected);
    }

    #[test]
    fn the_document_starts_at_the_first_vertex_not_a_brand_header() {
        // The deliberate output change recorded by the contract: the
        // absorbed writer began with `# openvoxel\n`, the codec emits no
        // header of any kind.
        let obj = to_obj_string(&faces(&cube()));
        assert!(obj.starts_with("v 0 1 0\n"), "{obj}");
        assert!(!obj.contains('#'), "{obj}");
    }

    #[test]
    fn obj_export_lists_vertices_and_faces() {
        // The source count assertions, kept.
        let obj = to_obj_string(&faces(&cube()));
        assert_eq!(obj.lines().filter(|line| line.starts_with("v ")).count(), 8);
        assert_eq!(obj.lines().filter(|line| line.starts_with("f ")).count(), 6);
    }

    #[test]
    fn shared_corners_are_deduplicated() {
        // A 1x2 bar: 3x2x2 corner lattice, ten exposed quads.
        let mut grid = cube();
        grid.set([1, 0, 0], Rgba([200, 120, 60, 255]));
        let obj = to_obj_string(&faces(&grid));
        assert_eq!(obj.lines().filter(|line| line.starts_with("v ")).count(), 12);
        assert_eq!(obj.lines().filter(|line| line.starts_with("f ")).count(), 10);
    }

    #[test]
    fn negative_corners_encode_verbatim() {
        let mut grid = VoxelGrid::new();
        grid.set([-1, -2, -3], Rgba([1, 2, 3, 255]));
        let obj = to_obj_string(&faces(&grid));
        assert!(obj.starts_with("v -1 -1 -3\n"), "{obj}");
        assert!(obj.contains("v 0 -2 -2\n"), "{obj}");
    }

    #[test]
    fn an_empty_view_encodes_nothing() {
        assert_eq!(to_obj_string(&[]), "");
    }

    #[test]
    fn obj_and_preview_never_disagree() {
        // The core promise, checked across the ownership split: the encoder
        // and the preview plan read the same face list, so the artifact and
        // the picture describe the same mesh, polygon for polygon.
        let mut grid = VoxelGrid::new();
        for x in 0..2 {
            for z in 0..2 {
                grid.set([x, 0, z], Rgba([96, 168, 116, 255]));
            }
        }
        grid.set([0, 1, 0], Rgba([255, 255, 255, 255]));
        let quads = faces(&grid);
        let obj = to_obj_string(&quads);
        let plan = preview(&grid, 96, 64, Rgba([10, 14, 26, 255]), Shading::default());
        assert_eq!(obj.lines().filter(|line| line.starts_with("f ")).count(), plan.polygons.len());
        assert_eq!(plan.polygons.len(), quads.len());
    }

    #[test]
    fn write_obj_produces_the_same_bytes_it_returns() {
        let dir = std::env::temp_dir().join("media-codec-obj-tests");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("cube.obj");
        let quads = faces(&cube());
        let bytes = write_obj(&quads, &path).expect("write");
        let on_disk = std::fs::read(&path).expect("read");
        assert_eq!(bytes as usize, on_disk.len());
        assert_eq!(on_disk, to_obj_string(&quads).as_bytes());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
