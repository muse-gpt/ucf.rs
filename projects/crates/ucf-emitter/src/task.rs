use ucf_types::{ParamValue, ShaderOp, ShaderProgram, TaskKind, TaskNode};

/// `prepare` 阶段应使用的严格 lowering：未知 `Custom` / `Dispatch` 不得静默退化为 Fill。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgramLowerError {
    UnsupportedTaskKind { kind: String },
    UnsupportedCustomOp { name: String },
}

impl std::fmt::Display for ProgramLowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedTaskKind { kind } => write!(f, "unsupported task kind: {kind}"),
            Self::UnsupportedCustomOp { name } => write!(f, "unsupported custom op: {name}"),
        }
    }
}

impl std::error::Error for ProgramLowerError {}

/// 严格 lowering。未知领域自定义算子必须失败。
pub fn try_program_from_task(task: &TaskNode) -> Result<ShaderProgram, ProgramLowerError> {
    match &task.kind {
        TaskKind::Copy => Ok(ShaderProgram::mem_copy("ucf_copy")),
        TaskKind::Fill => {
            let value = match task.params.get("value") {
                Some(ParamValue::F64(v)) => *v as f32,
                Some(ParamValue::I64(v)) => *v as f32,
                _ => 0.0,
            };
            Ok(ShaderProgram::dispatch_fill("ucf_fill", value))
        }
        TaskKind::MatMul => Ok(ShaderProgram::matmul("ucf_matmul")),
        TaskKind::Custom(name) if name == "denoise" => {
            Ok(ShaderProgram::rgba8_denoise("ucf_denoise"))
        }
        TaskKind::Custom(name) if name == "attention" => {
            Ok(ShaderProgram::attention("ucf_attention"))
        }
        TaskKind::Custom(name) => Err(ProgramLowerError::UnsupportedCustomOp {
            name: name.clone(),
        }),
        TaskKind::Dispatch => Err(ProgramLowerError::UnsupportedTaskKind {
            kind: "Dispatch".into(),
        }),
        TaskKind::Raster => Err(ProgramLowerError::UnsupportedTaskKind {
            kind: "Raster".into(),
        }),
        TaskKind::RtTrace => Err(ProgramLowerError::UnsupportedTaskKind {
            kind: "RtTrace".into(),
        }),
    }
}

/// Build a backend-facing [`ShaderProgram`] from a task node.
///
/// Lower a task strictly. Unsupported tasks return an explicit error.
pub fn program_from_task(task: &TaskNode) -> Result<ShaderProgram, ProgramLowerError> {
    try_program_from_task(task)
}

/// Whether the program needs a device kernel (vs memcpy-only).
pub fn needs_kernel(program: &ShaderProgram) -> bool {
    program.ops.iter().any(|op| {
        matches!(
            op,
            ShaderOp::IotaFill { .. }
                | ShaderOp::MatMul
                | ShaderOp::Rgba8Denoise
                | ShaderOp::Attention
        )
    })
}
