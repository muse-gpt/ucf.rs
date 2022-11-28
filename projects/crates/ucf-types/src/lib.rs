//! Shared UCF value types.
//!
//! This crate is the bottom of the dependency stack. It has no graph logic,
//! no scheduler, and no backend bindings — only serializable data shapes.

mod dep;
mod ids;
mod objective;
mod resource;
mod shader;
mod task;

pub use dep::DepKind;
pub use ids::{ResourceId, ShaderId, TaskId};
pub use objective::{Objective, Priority};
pub use resource::{Access, Domain, ResourceKind, ResourceNode};
pub use shader::{ShaderOp, ShaderProgram};
pub use task::{Dispatch, ParamValue, TaskKind, TaskNode};
