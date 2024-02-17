//! The isometric preview plan.
//!
//! A fixed `2:1` isometric projection: one unit right-down, one unit
//! left-down, one unit up. Faces are ordered back to front by mean depth,
//! so painting them in plan order needs no z-buffer. This module owns the
//! projection, the fit into the frame, the depth sort and the shading call;
//! it never touches pixels — `media.render`'s `voxel-export` entry paints
//! the plan through `media.raster`.
//!
//! The sort is a stable sort over the deterministic face list, so the same
//! grid always produces the same plan, polygon for polygon.

use geometry::Rgba;

use crate::mesh::{Face, VoxelGrid, faces};
use crate::shade::Shading;

/// One quad in frame coordinates, ready to paint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PreviewPolygon {
    pub points: [[f32; 2]; 4],
    pub color: Rgba,
}

/// The whole preview: a background plus polygons in draw order.
#[derive(Clone, Debug, PartialEq)]
pub struct VoxelPreview {
    pub width: u32,
    pub height: u32,
    pub background: Rgba,
    pub polygons: Vec<PreviewPolygon>,
}

/// The fixed 2:1 isometric projection of a world corner.
#[must_use]
pub fn project(corner: [f32; 3]) -> [f32; 2] {
    [(corner[0] - corner[2]) * 0.5, (corner[0] + corner[2]) * 0.25 - corner[1] * 0.5]
}

/// The projected depth of a face: the mean of `(x + z - y)` over its four
/// corners. Lower draws first (back to front).
#[must_use]
pub fn depth(face: &Face) -> f32 {
    face.corners.iter().map(|c| c[0] + c[2] - c[1]).sum::<f32>() / 4.0
}

/// Builds the draw-ordered preview plan for `grid`.
///
/// The projected bounds are fitted into the frame with a `0.9` margin on
/// both axes, keeping the smaller scale, and centred; a degenerate span is
/// clamped to `0.001` exactly as the source did. An empty grid yields a
/// background-only plan.
#[must_use]
pub fn preview(grid: &VoxelGrid, width: u32, height: u32, background: Rgba, shading: Shading) -> VoxelPreview {
    let quads = faces(grid);
    if quads.is_empty() {
        return VoxelPreview { width, height, background, polygons: Vec::new() };
    }

    let mut min = [f32::MAX; 2];
    let mut max = [f32::MIN; 2];
    let mut projected: Vec<(f32, [[f32; 2]; 4], Rgba)> = Vec::with_capacity(quads.len());
    for face in &quads {
        let mut points = [[0.0f32; 2]; 4];
        let face_depth = depth(face);
        for (slot, corner) in face.corners.iter().enumerate() {
            let point = project(*corner);
            points[slot] = point;
            min[0] = min[0].min(point[0]);
            min[1] = min[1].min(point[1]);
            max[0] = max[0].max(point[0]);
            max[1] = max[1].max(point[1]);
        }
        projected.push((face_depth, points, shading.paint(face.direction, face.color)));
    }

    let span_x = (max[0] - min[0]).max(0.001);
    let span_y = (max[1] - min[1]).max(0.001);
    let scale = ((width as f32 * 0.9) / span_x).min((height as f32 * 0.9) / span_y);
    let offset_x = (width as f32 - span_x * scale) * 0.5 - min[0] * scale;
    let offset_y = (height as f32 - span_y * scale) * 0.5 - min[1] * scale;

    projected.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let polygons = projected
        .into_iter()
        .map(|(_, points, color)| PreviewPolygon {
            points: points.map(|point| [point[0] * scale + offset_x, point[1] * scale + offset_y]),
            color,
        })
        .collect();
    VoxelPreview { width, height, background, polygons }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube() -> VoxelGrid {
        let mut grid = VoxelGrid::new();
        grid.set([0, 0, 0], Rgba([200, 120, 60, 255]));
        grid
    }

    #[test]
    fn the_projection_is_the_fixed_two_to_one_contract() {
        assert_eq!(project([0.0, 0.0, 0.0]), [0.0, 0.0]);
        assert_eq!(project([1.0, 0.0, 0.0]), [0.5, 0.25]);
        assert_eq!(project([0.0, 0.0, 1.0]), [-0.5, 0.25]);
        assert_eq!(project([0.0, 1.0, 0.0]), [0.0, -0.5]);
    }

    #[test]
    fn an_empty_grid_yields_a_background_only_plan() {
        let plan = preview(&VoxelGrid::new(), 32, 32, Rgba([7, 9, 12, 255]), Shading::default());
        assert!(plan.polygons.is_empty());
        assert_eq!((plan.width, plan.height, plan.background), (32, 32, Rgba([7, 9, 12, 255])));
    }

    #[test]
    fn the_lone_cube_draws_back_to_front_in_the_locked_order() {
        // Depths: PosY 0, NegY 1, PosX 1, NegX 0, PosZ 1, NegZ 0. The
        // stable sort keeps the face-list order inside ties, so the draw
        // order is PosY, NegX, NegZ, NegY, PosX, PosZ.
        let plan = preview(&cube(), 64, 64, Rgba([0, 0, 0, 255]), Shading::default());
        assert_eq!(plan.polygons.len(), 6);
        let painted: Vec<Rgba> = plan.polygons.iter().map(|p| p.color).collect();
        assert_eq!(
            painted,
            vec![
                Rgba([200, 120, 60, 255]), // PosY * 1.0
                Rgba([124, 74, 37, 255]),  // NegX * 0.62
                Rgba([132, 79, 39, 255]),  // NegZ * 0.66
                Rgba([90, 54, 27, 255]),   // NegY * 0.45
                Rgba([144, 86, 43, 255]),  // PosX * 0.72
                Rgba([116, 69, 34, 255]),  // PosZ * 0.58
            ],
            "depth-sorted draw order with the preset shading"
        );
    }

    #[test]
    fn the_fit_is_margin_ninety_and_centred() {
        // Lone cube projects to a 1x1 bounds; at 64x64 the scale is 57.6
        // and the centre offsets are 32.
        let plan = preview(&cube(), 64, 64, Rgba([0, 0, 0, 255]), Shading::default());
        let top = &plan.polygons[0];
        let near = |a: f32, b: f32| (a - b).abs() < 1e-3;
        assert!(near(top.points[0][0], 32.0) && near(top.points[0][1], 3.2), "{:?}", top.points);
        assert!(near(top.points[2][0], 32.0) && near(top.points[2][1], 32.0), "{:?}", top.points);
    }

    #[test]
    fn preview_and_mesh_never_disagree() {
        // The core promise: the plan is the face list, reordered and shaded,
        // never re-derived. Same grid, same polygons, byte-stable.
        let mut grid = VoxelGrid::new();
        for (x, z) in [(0, 0), (1, 0), (0, 1), (1, 1), (0, 0)] {
            grid.set([x, 0, z], Rgba([80, 160, 240, 255]));
        }
        grid.set([0, 1, 0], Rgba([255, 255, 255, 255]));
        let quads = faces(&grid);
        let plan = preview(&grid, 96, 64, Rgba([10, 14, 26, 255]), Shading::default());
        assert_eq!(plan.polygons.len(), quads.len(), "one polygon per exposed face");

        let again = preview(&grid, 96, 64, Rgba([10, 14, 26, 255]), Shading::default());
        assert_eq!(plan, again, "the plan is byte-stable across runs");

        // Every painted colour is the shading of some face in the list.
        for polygon in &plan.polygons {
            assert!(
                quads.iter().any(|face| polygon.color == Shading::default().paint(face.direction, face.color)),
                "no polygon invents a colour"
            );
        }
    }

    #[test]
    fn negative_coordinate_scenes_fit_and_stay_deterministic() {
        let mut grid = VoxelGrid::new();
        grid.set([-7, -3, 5], Rgba([12, 34, 56, 255]));
        grid.set([-6, -3, 5], Rgba([12, 34, 56, 255]));
        let plan = preview(&grid, 128, 128, Rgba([0, 0, 0, 255]), Shading::default());
        assert_eq!(plan.polygons.len(), 10, "a two-cube bar keeps ten faces");
        assert!(
            plan.polygons
                .iter()
                .flat_map(|p| p.points.iter())
                .all(|point| point[0] >= 0.0 && point[0] <= 128.0 && point[1] >= 0.0 && point[1] <= 128.0),
            "the fit keeps the scene inside the frame"
        );
        assert_eq!(plan, preview(&grid, 128, 128, Rgba([0, 0, 0, 255]), Shading::default()));
    }
}
