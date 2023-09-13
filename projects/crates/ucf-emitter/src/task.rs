use ucf_types::{ParamValue, ShaderOp, ShaderProgram, TaskKind, TaskNode};

/// `prepare` 阶段应使用的严格 lowering：未知 `Custom` / `Dispatch` 不得静默退化为 Fill。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProgramLowerError {
    UnsupportedTaskKind { kind: String },
    UnsupportedCustomOp { name: String },
}

/// 严格 lowering；黑洞等领域 `ton_*` 自定义算子在此必须失败。
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
/// 遗留路径：未知 `Custom` 仍映射为占位 Fill（待后端全面改用 `try_program_from_task`）。
/// `Raster` / `RtTrace` 同理为占位，图形后端消费 present 参数而非此处 kernel。
pub fn program_from_task(task: &TaskNode) -> ShaderProgram {
    match try_program_from_task(task) {
        Ok(p) => p,
        Err(ProgramLowerError::UnsupportedCustomOp { .. })
        | Err(ProgramLowerError::UnsupportedTaskKind { kind: _ }) => {
            ShaderProgram::dispatch_fill("ucf_dispatch", 1.0)
        }
    }
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
