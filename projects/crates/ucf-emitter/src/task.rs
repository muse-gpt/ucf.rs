use ucf_types::{ParamValue, ShaderOp, ShaderProgram, TaskKind, TaskNode};

/// Build a backend-facing [`ShaderProgram`] from a task node.
pub fn program_from_task(task: &TaskNode) -> ShaderProgram {
    match &task.kind {
        TaskKind::Copy => ShaderProgram::mem_copy("ucf_copy"),
        TaskKind::Fill => {
            let value = match task.params.get("value") {
                Some(ParamValue::F64(v)) => *v as f32,
                Some(ParamValue::I64(v)) => *v as f32,
                _ => 0.0,
            };
            ShaderProgram::dispatch_fill("ucf_fill", value)
        }
        TaskKind::MatMul => ShaderProgram::matmul("ucf_matmul"),
        TaskKind::Dispatch | TaskKind::Custom(_) => {
            ShaderProgram::dispatch_fill("ucf_dispatch", 1.0)
        }
        TaskKind::Raster | TaskKind::RtTrace => {
            ShaderProgram::dispatch_fill("ucf_gfx_placeholder", 0.0)
        }
    }
}

/// Whether the program needs a device kernel (vs memcpy-only).
pub fn needs_kernel(program: &ShaderProgram) -> bool {
    program
        .ops
        .iter()
        .any(|op| matches!(op, ShaderOp::IotaFill { .. } | ShaderOp::MatMul))
}
