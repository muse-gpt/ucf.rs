//! Voxels: a sparse grid of coloured unit cubes, the exposed-face mesh and
//! the isometric preview plan.
//!
//! The semantics were absorbed from the OpenVoxel core (Apache-2.0,
//! © OpenVoxel Authors) with the values preserved. The mesher keeps only
//! the faces that touch empty space, which is what turns a voxel model
//! into a low-poly mesh. The same ordered face list feeds the preview plan
//! and the OBJ encoder, so what you see is what you ship: determinism is a
//! contract — cells visit in `BTreeMap` order, directions test in one fixed
//! sequence and the depth sort is stable. Replacing any of that with hash
//! maps or parallel iteration would break the promise.
//!
//! One correction is deliberate and contract-level: the six fixed shadow
//! constants no longer masquerade as a material. They became the named
//! [`Shading::FixedSixDir`] preset, an explicit preview parameter whose
//! default keeps every painted byte identical.
//!
//! Ownership boundaries:
//! - this crate owns the grid, face extraction ([`mesh`]), the shading
//!   strategy ([`shade`]) and the preview plan ([`preview`]); it never
//!   paints pixels and never writes files;
//! - OBJ encoding lives in `media.codec` (its `obj` feature consumes the
//!   [`Face`] sequence one-way; this crate never imports the codec);
//! - painting the plan and writing artifacts live in `media.render`'s
//!   `voxel-export` entry;
//! - the host input contract ([`input`]) carries the absorbed binding
//!   defaults, and a malformed colour is an explicit `invalid_input` error.
//!
//! ```rust
//! use geometry::Rgba;
//! use media_voxel::{VoxelGrid, faces};
//!
//! let mut grid = VoxelGrid::new();
//! grid.set([0, 0, 0], Rgba([120, 180, 255, 255]));
//! assert_eq!(faces(&grid).len(), 6, "a lone cube shows six faces");
//! ```

pub mod input;
pub mod mesh;
pub mod preview;
pub mod shade;

// The value types that appear in the public surface, re-exported so encoders
// and hosts can name a colour without importing `geometry` themselves.
pub use geometry::Rgba;
pub use input::{DEFAULT_BACKGROUND, DEFAULT_HEIGHT, DEFAULT_WIDTH, VoxelEntry, VoxelSpec};
pub use mesh::{Direction, Face, VoxelGrid, faces};
pub use preview::{PreviewPolygon, VoxelPreview, depth, preview, project};
pub use shade::Shading;
