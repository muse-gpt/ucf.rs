//! Minimal end-to-end skeleton via the `ucf` facade.

use ucf::prelude::*;

fn main() -> ucf_scheduler::Result<()> {
    let graph = Graph {
        resources: ResourceGraph {
            nodes: vec![ResourceNode {
                id: ResourceId(1),
                kind: ResourceKind::Tensor,
                domain: Domain::Vram,
                access: Access::ReadWrite,
                byte_size: None,
            }],
        },
        tasks: TaskGraph {
            nodes: vec![TaskNode {
                id: TaskId(1),
                kind: TaskKind::MatMul,
                shader: ShaderId(1),
                params: Default::default(),
                dispatch: Default::default(),
                objective: Objective::MaxThroughput,
                priority: Priority::Batch,
            }],
            edges: vec![DepEdge {
                from_task: None,
                from_resource: Some(ResourceId(1)),
                to_task: TaskId(1),
                kind: DepKind::Data,
            }],
        },
    };

    let mut runtime = Runtime::new();
    runtime.register_backend(Box::new(CudaBackend::new(0)));
    runtime.run(&graph)?;
    Ok(())
}
