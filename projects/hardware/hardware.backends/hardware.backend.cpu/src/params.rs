use hardware_ir::{ParamValue, TaskNode};
use hardware_types::ResourceId;

/// CPU backend failure message (mapped to [`hardware_scheduler::Error::Backend`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendError(pub String);

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

pub fn resource_param(task: &TaskNode, key: &str) -> Result<ResourceId, BackendError> {
    match task.params.get(key) {
        Some(ParamValue::I64(v)) if *v >= 0 => Ok(ResourceId(*v as u64)),
        Some(other) => Err(BackendError(format!(
            "task {} param `{key}` must be non-negative i64, got {other:?}",
            task.id.0
        ))),
        None => Err(BackendError(format!(
            "task {} missing param `{key}`",
            task.id.0
        ))),
    }
}

pub fn u32_param(task: &TaskNode, key: &str) -> Result<u32, BackendError> {
    match task.params.get(key) {
        Some(ParamValue::I64(v)) if *v >= 0 && *v <= u32::MAX as i64 => Ok(*v as u32),
        Some(other) => Err(BackendError(format!(
            "task {} param `{key}` must be u32-range i64, got {other:?}",
            task.id.0
        ))),
        None => Err(BackendError(format!(
            "task {} missing param `{key}`",
            task.id.0
        ))),
    }
}

pub fn f32_param(task: &TaskNode, key: &str) -> Result<f32, BackendError> {
    f64_param(task, key).map(|v| v as f32)
}

pub fn f64_param(task: &TaskNode, key: &str) -> Result<f64, BackendError> {
    match task.params.get(key) {
        Some(ParamValue::F64(v)) => Ok(*v),
        Some(ParamValue::I64(v)) => Ok(*v as f64),
        Some(other) => Err(BackendError(format!(
            "task {} param `{key}` must be numeric, got {other:?}",
            task.id.0
        ))),
        None => Err(BackendError(format!(
            "task {} missing param `{key}`",
            task.id.0
        ))),
    }
}

pub fn optional_f64_param(task: &TaskNode, key: &str, default: f64) -> f64 {
    match task.params.get(key) {
        Some(ParamValue::F64(v)) => *v,
        Some(ParamValue::I64(v)) => *v as f64,
        _ => default,
    }
}
