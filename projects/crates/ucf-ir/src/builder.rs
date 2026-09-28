use std::collections::BTreeMap;

use crate::{
    Access, DepEdge, DepKind, Dispatch, Domain, Graph, Objective, ParamValue, Priority,
    ResourceGraph, ResourceId, ResourceKind, ResourceNode, ShaderId, TaskGraph, TaskId, TaskKind,
    TaskNode,
};

/// Ergonomic builder for [`Graph`] documents (resources, tasks, dependency edges).
pub struct GraphBuilder {
    resources: Vec<ResourceNode>,
    tasks: Vec<TaskNode>,
    edges: Vec<DepEdge>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            resources: Vec::new(),
            tasks: Vec::new(),
            edges: Vec::new(),
        }
    }

    pub fn buffer(&mut self, id: u64, domain: Domain) -> ResourceId {
        self.resource(id, ResourceKind::Buffer, domain)
    }

    pub fn texture(&mut self, id: u64, domain: Domain) -> ResourceId {
        self.resource(id, ResourceKind::Texture, domain)
    }

    pub fn tensor(&mut self, id: u64, domain: Domain) -> ResourceId {
        self.resource(id, ResourceKind::Tensor, domain)
    }

    pub fn resource(&mut self, id: u64, kind: ResourceKind, domain: Domain) -> ResourceId {
        self.resources.push(ResourceNode {
            id: ResourceId(id),
            kind,
            domain,
            access: Access::ReadWrite,
            byte_size: None,
        });
        ResourceId(id)
    }

    pub fn gpu(&mut self, id: u64, op: &str) -> TaskId {
        self.task(id, op, "Gpu", task_kind(op), Objective::MaxUtilization, Priority::Batch)
    }

    pub fn gpu_matmul(&mut self, id: u64, op: &str) -> TaskId {
        self.task(id, op, "Gpu", TaskKind::MatMul, Objective::MaxThroughput, Priority::Batch)
    }

    pub fn gpu_raster(&mut self, id: u64, op: &str) -> TaskId {
        self.task(
            id,
            op,
            "Gpu",
            TaskKind::Raster,
            Objective::MinLatency { deadline_micros: 8_000 },
            Priority::Interactive,
        )
    }

    pub fn executor(
        &mut self,
        id: u64,
        op: &str,
        executor: &str,
        kind: TaskKind,
        objective: Objective,
        priority: Priority,
    ) -> TaskId {
        self.task(id, op, executor, kind, objective, priority)
    }

    pub fn task(
        &mut self,
        id: u64,
        op: &str,
        executor: &str,
        kind: TaskKind,
        objective: Objective,
        priority: Priority,
    ) -> TaskId {
        let mut params = BTreeMap::new();
        params.insert("op".into(), ParamValue::Str(op.into()));
        params.insert("executor".into(), ParamValue::Str(executor.into()));
        self.tasks.push(TaskNode {
            id: TaskId(id),
            kind,
            shader: ShaderId(id),
            params,
            dispatch: Dispatch::Auto,
            objective,
            priority,
        });
        TaskId(id)
    }

    pub fn chain(&mut self, ids: &[u64]) {
        self.edges.extend(chain_edges(ids));
    }

    pub fn build(self) -> Graph {
        Graph {
            resources: ResourceGraph {
                nodes: self.resources,
            },
            tasks: TaskGraph {
                nodes: self.tasks,
                edges: self.edges,
            },
        }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Linear execution-order edges between task ids.
pub fn chain_edges(ids: &[u64]) -> Vec<DepEdge> {
    ids.windows(2)
        .map(|pair| DepEdge {
            from_task: Some(TaskId(pair[0])),
            from_resource: None,
            to_task: TaskId(pair[1]),
            kind: DepKind::Execution,
        })
        .collect()
}

fn task_kind(op: &str) -> TaskKind {
    match op {
        "MatMul" => TaskKind::MatMul,
        "Clear" | "Swap" => TaskKind::Copy,
        "RasterShader" | "CullShader" | "VertexShader" | "PixelShader" | "Rasterize" | "RayGen"
        | "Traverse" | "ClosestHit" | "Present" => TaskKind::Raster,
        other => TaskKind::Custom(other.into()),
    }
}
