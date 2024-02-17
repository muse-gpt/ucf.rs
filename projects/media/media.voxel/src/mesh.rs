//! Sparse voxel grid and exposed-face extraction.
//!
//! The grid is a `BTreeMap` keyed by integer cell coordinates, so negative
//! cells are first-class and every traversal visits the cells in one global
//! order. [`faces`] walks cells in that order and tests the six directions
//! in the fixed sequence +Y, -Y, +X, -X, +Z, -Z. The resulting list is the
//! shared truth: the isometric preview plan and the OBJ encoder both consume
//! it unchanged, which is what keeps preview and export from ever
//! disagreeing. Determinism is the contract — replacing the `BTreeMap` with
//! a hash map or parallelizing the walk would break it.
//!
//! Culling looks at occupancy only: a stored voxel hides the shared face
//! even when its colour is fully transparent.

use std::collections::BTreeMap;

use geometry::Rgba;

type Cell = (i32, i32, i32);

/// A sparse grid of unit cubes.
#[derive(Default, Debug)]
pub struct VoxelGrid {
    cells: BTreeMap<Cell, Rgba>,
}

impl VoxelGrid {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Stores `color` at `at`, replacing any voxel already there.
    pub fn set(&mut self, at: [i32; 3], color: Rgba) {
        self.cells.insert((at[0], at[1], at[2]), color);
    }

    /// Removes the voxel at `at`, returning its colour if one was stored.
    pub fn remove(&mut self, at: [i32; 3]) -> Option<Rgba> {
        self.cells.remove(&(at[0], at[1], at[2]))
    }

    #[must_use]
    pub fn get(&self, at: [i32; 3]) -> Option<Rgba> {
        self.cells.get(&(at[0], at[1], at[2])).copied()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

/// One of the six axis-aligned cube directions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    PosY,
    NegY,
    PosX,
    NegX,
    PosZ,
    NegZ,
}

/// An exposed quad of the mesh.
///
/// `corners` carry the fixed winding the OBJ `f` lines and the preview
/// polygons are emitted with: the order itself is the contract, pinned by
/// golden tests, not a handedness claim. Brightness is deliberately not a
/// field here — it is a preview-time decision made through
/// [`crate::Shading`], so the mesh stays a neutral geometry view the
/// encoder can consume without inventing any voxel semantics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Face {
    pub corners: [[f32; 3]; 4],
    pub direction: Direction,
    pub color: Rgba,
}

/// Every face that touches empty space, in cell order times the fixed
/// direction order.
#[must_use]
pub fn faces(grid: &VoxelGrid) -> Vec<Face> {
    let mut out = Vec::new();
    for (cell, color) in &grid.cells {
        let (x, y, z) = *cell;
        let (fx, fy, fz) = (x as f32, y as f32, z as f32);
        // +Y (top)
        if grid.get([x, y + 1, z]).is_none() {
            out.push(Face {
                corners: [
                    [fx, fy + 1.0, fz],
                    [fx + 1.0, fy + 1.0, fz],
                    [fx + 1.0, fy + 1.0, fz + 1.0],
                    [fx, fy + 1.0, fz + 1.0],
                ],
                direction: Direction::PosY,
                color: *color,
            });
        }
        // -Y (bottom)
        if grid.get([x, y - 1, z]).is_none() {
            out.push(Face {
                corners: [[fx, fy, fz], [fx, fy, fz + 1.0], [fx + 1.0, fy, fz + 1.0], [fx + 1.0, fy, fz]],
                direction: Direction::NegY,
                color: *color,
            });
        }
        // +X
        if grid.get([x + 1, y, z]).is_none() {
            out.push(Face {
                corners: [
                    [fx + 1.0, fy, fz],
                    [fx + 1.0, fy + 1.0, fz],
                    [fx + 1.0, fy + 1.0, fz + 1.0],
                    [fx + 1.0, fy, fz + 1.0],
                ],
                direction: Direction::PosX,
                color: *color,
            });
        }
        // -X
        if grid.get([x - 1, y, z]).is_none() {
            out.push(Face {
                corners: [[fx, fy, fz + 1.0], [fx, fy + 1.0, fz + 1.0], [fx, fy + 1.0, fz], [fx, fy, fz]],
                direction: Direction::NegX,
                color: *color,
            });
        }
        // +Z
        if grid.get([x, y, z + 1]).is_none() {
            out.push(Face {
                corners: [
                    [fx + 1.0, fy, fz + 1.0],
                    [fx + 1.0, fy + 1.0, fz + 1.0],
                    [fx, fy + 1.0, fz + 1.0],
                    [fx, fy, fz + 1.0],
                ],
                direction: Direction::PosZ,
                color: *color,
            });
        }
        // -Z
        if grid.get([x, y, z - 1]).is_none() {
            out.push(Face {
                corners: [[fx, fy, fz], [fx, fy + 1.0, fz], [fx + 1.0, fy + 1.0, fz], [fx + 1.0, fy, fz]],
                direction: Direction::NegZ,
                color: *color,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use geometry::Rgba;

    fn cube() -> VoxelGrid {
        let mut grid = VoxelGrid::new();
        grid.set([0, 0, 0], Rgba([200, 120, 60, 255]));
        grid
    }

    #[test]
    fn a_lone_cube_keeps_every_face() {
        assert_eq!(faces(&cube()).len(), 6);
    }

    #[test]
    fn shared_faces_are_culled() {
        let mut grid = cube();
        grid.set([1, 0, 0], Rgba([200, 120, 60, 255]));
        // 12 faces minus the two that touch.
        assert_eq!(faces(&grid).len(), 10);
    }

    #[test]
    fn removing_a_voxel_brings_faces_back() {
        let mut grid = cube();
        grid.set([1, 0, 0], Rgba([0, 0, 0, 255]));
        grid.remove([1, 0, 0]);
        assert_eq!(faces(&grid).len(), 6);
    }

    #[test]
    fn culling_looks_at_occupancy_not_color() {
        // The source semantics, kept deliberately: a fully transparent
        // stored voxel still hides the shared face.
        let mut grid = cube();
        grid.set([1, 0, 0], Rgba([0, 0, 0, 0]));
        assert_eq!(faces(&grid).len(), 10);
    }

    #[test]
    fn negative_coordinates_are_first_class_cells() {
        let mut grid = VoxelGrid::new();
        grid.set([-1, -2, -3], Rgba([10, 20, 30, 255]));
        let quads = faces(&grid);
        assert_eq!(quads.len(), 6);
        assert!(quads.iter().all(|face| face.corners.iter().all(|c| c[0] == -1.0 || c[0] == 0.0)), "the cube occupies x -1..0");
        let top = &quads[0];
        assert_eq!(top.direction, Direction::PosY);
        assert_eq!(top.corners[0], [-1.0, -1.0, -3.0]);
    }

    #[test]
    fn the_six_windings_are_the_locked_corner_orders() {
        // Byte-for-byte golden of the absorbed corner sequences. The face
        // order is the fixed direction sequence: +Y, -Y, +X, -X, +Z, -Z.
        let expected: [([[f32; 3]; 4], Direction); 6] = [
            ([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]], Direction::PosY),
            ([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]], Direction::NegY),
            ([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]], Direction::PosX),
            ([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], Direction::NegX),
            ([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], Direction::PosZ),
            ([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], Direction::NegZ),
        ];
        let quads = faces(&cube());
        assert_eq!(quads.len(), 6);
        for (face, (corners, direction)) in quads.iter().zip(expected) {
            assert_eq!(face.corners, corners, "{direction:?} winding drifted");
            assert_eq!(face.direction, direction);
            assert_eq!(face.color, Rgba([200, 120, 60, 255]));
        }
    }

    #[test]
    fn face_order_is_cell_order_times_direction_order() {
        // BTreeMap visits (-5,4,9) before (0,0,0) before (2,-3,-1), and each
        // cell emits its six faces in the fixed direction sequence.
        let mut grid = VoxelGrid::new();
        grid.set([2, -3, -1], Rgba([1, 1, 1, 255]));
        grid.set([-5, 4, 9], Rgba([2, 2, 2, 255]));
        grid.set([0, 0, 0], Rgba([3, 3, 3, 255]));
        let quads = faces(&grid);
        assert_eq!(quads.len(), 18);
        let cycle = [Direction::PosY, Direction::NegY, Direction::PosX, Direction::NegX, Direction::PosZ, Direction::NegZ];
        for chunk in quads.chunks(6) {
            assert_eq!(chunk.iter().map(|face| face.direction).collect::<Vec<_>>(), cycle.to_vec());
        }
        let first_corners: Vec<[f32; 3]> = quads.chunks(6).map(|chunk| chunk[0].corners[0]).collect();
        assert_eq!(
            first_corners,
            vec![[-5.0, 5.0, 9.0], [0.0, 1.0, 0.0], [2.0, -2.0, -1.0]],
            "cells visit in BTreeMap order, top faces first"
        );
    }

    #[test]
    fn traversal_is_stable_across_insert_orders() {
        // The BTreeMap erases insertion history: the same cell set always
        // yields the same face list.
        let cells = [[0, 0, 0], [3, -1, 2], [-4, 5, -5], [1, 1, 1]];
        let mut a = VoxelGrid::new();
        for (index, cell) in cells.iter().enumerate() {
            a.set(*cell, Rgba([index as u8, 9, 9, 255]));
        }
        let mut b = VoxelGrid::new();
        for (index, cell) in cells.iter().rev().enumerate() {
            b.set(*cell, Rgba([(3 - index) as u8, 9, 9, 255]));
        }
        assert_eq!(faces(&a), faces(&b), "insertion order cannot leak into the face list");
    }
}
