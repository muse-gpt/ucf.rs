use ucf_types::{ParamValue, ShaderOp, ShaderProgram, TaskKind, TaskNode};

/// Build a backend-facing [`ShaderProgram`] from a task node.
///
/// `Raster` / `RtTrace` keep a placeholder program: clear / present params are
/// consumed by graphics backends (`dst`, `width`, `height`, optional `r`/`g`/`b`/`a`),
/// not lowered to a compute kernel here.
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
        TaskKind::Custom(name) if name == "denoise" => {
            ShaderProgram::rgba8_denoise("ucf_denoise")
        }
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
    program.ops.iter().any(|op| {
        matches!(
            op,
            ShaderOp::IotaFill { .. } | ShaderOp::MatMul | ShaderOp::Rgba8Denoise
        )
    })
}
