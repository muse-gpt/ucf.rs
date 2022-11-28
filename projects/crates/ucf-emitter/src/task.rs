use ucf_types::{ShaderOp, ShaderProgram, TaskKind, TaskNode};

pub fn program_from_task(task: &TaskNode) -> ShaderProgram {
    match task.kind {
        TaskKind::Copy => ShaderProgram::mem_copy("ucf_copy"),
        TaskKind::MatMul | TaskKind::Dispatch | TaskKind::Custom(_) => {
            ShaderProgram::dispatch_fill("ucf_dispatch", 1.0)
        }
        TaskKind::Raster | TaskKind::RtTrace => {
            ShaderProgram::dispatch_fill("ucf_gfx_placeholder", 0.0)
        }
    }
}

pub fn needs_kernel(program: &ShaderProgram) -> bool {
    program
        .ops
        .iter()
        .any(|op| matches!(op, ShaderOp::IotaFill { .. }))
}
