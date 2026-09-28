use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::ids::{ShaderId, TaskId};
use crate::objective::{Objective, Priority};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskKind {
    Raster,
    Dispatch,
    /// Write a constant into a buffer (CPU reference path).
    Fill,
    MatMul,
    RtTrace,
    Copy,
    Custom(String),
}

/// Grid / tile sizing strategy. Backends may auto-tune when set to `Auto`.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub enum Dispatch {
    #[default]
    Auto,
    Explicit { x: u32, y: u32, z: u32 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskNode {
    pub id: TaskId,
    pub kind: TaskKind,
    pub shader: ShaderId,
    pub params: BTreeMap<String, ParamValue>,
    pub dispatch: Dispatch,
    pub objective: Objective,
    pub priority: Priority,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ParamValue {
    Bool(bool),
    I64(i64),
    F64(f64),
    Str(String),
}
