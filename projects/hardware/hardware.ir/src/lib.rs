//! UCF graph intermediate representation.
//!
//! [`hardware_types`] holds shared value shapes; this crate adds graph containers
//! and validation (unique ids, edge references, topological order).
#![warn(missing_docs)]

mod binary;
mod builder;
mod error;
mod graph;

pub use binary::{decode, encode, MAGIC, WIRE_MAJOR};
pub use builder::{chain_edges, GraphBuilder};
pub use error::{Error, Result};
pub use graph::{DepEdge, Graph, ResourceGraph, TaskGraph};

// Re-export shared types so existing `hardware_ir::TaskId` imports keep working.
pub use hardware_types::{
    Access, DepKind, Dispatch, Domain, Objective, ParamValue, Priority, ResourceId,
    ResourceKind, ResourceNode, ShaderId, TaskId, TaskKind, TaskNode,
};
