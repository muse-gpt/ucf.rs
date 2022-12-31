use ucf_ir::Graph;
use ucf_optimize::{apply, Optimization};
use ucf_scheduler::{Backend, Scheduler};

use crate::capacity::{apply_capacity, CapacityPolicy};

/// Owns backends, capacity policy, and graph execution.
pub struct Runtime {
    scheduler: Scheduler,
    capacity: CapacityPolicy,
}

impl Runtime {
    /// Default runtime with [`CapacityPolicy::Infinite`].
    pub fn new() -> Self {
        Self {
            scheduler: Scheduler::new(),
            capacity: CapacityPolicy::default(),
        }
    }

    /// Runtime with an explicit capacity policy.
    pub fn with_capacity(capacity: CapacityPolicy) -> Self {
        Self {
            scheduler: Scheduler::new(),
            capacity,
        }
    }

    /// Register a backend for the scheduler.
    pub fn register_backend(&mut self, backend: Box<dyn Backend>) {
        self.scheduler.register(backend);
    }

    /// Current capacity policy.
    pub fn capacity(&self) -> CapacityPolicy {
        self.capacity
    }

    /// Run a graph under the current capacity policy.
    pub fn run(&mut self, graph: &Graph) -> ucf_scheduler::Result<()> {
        self.run_with_opts(graph, &[])
    }

    /// Apply optimizations, then capacity, then schedule.
    pub fn run_with_opts(
        &mut self,
        graph: &Graph,
        opts: &[Optimization],
    ) -> ucf_scheduler::Result<()> {
        let mut graph = graph.clone();
        apply(&mut graph, opts)?;
        apply_capacity(&mut graph, self.capacity)?;
        self.scheduler.execute(&graph)
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_ir::{
        Access, Domain, ResourceGraph, ResourceId, ResourceKind, ResourceNode, TaskGraph,
    };
    use ucf_scheduler::NoopBackend;

    #[test]
    fn run_rejects_soft_limit_overflow() {
        let graph = Graph {
            resources: ResourceGraph {
                nodes: vec![ResourceNode {
                    id: ResourceId(1),
                    kind: ResourceKind::Buffer,
                    domain: Domain::Vram,
                    access: Access::ReadWrite,
                    byte_size: Some(200),
                }],
            },
            tasks: TaskGraph {
                nodes: vec![],
                edges: vec![],
            },
        };
        let mut rt = Runtime::with_capacity(CapacityPolicy::SoftLimit {
            domain: Domain::Vram,
            bytes: 100,
        });
        rt.register_backend(Box::new(NoopBackend));
        let err = rt.run(&graph).expect_err("should exceed");
        assert!(matches!(
            err,
            ucf_scheduler::Error::CapacityExceeded { .. }
        ));
    }

    #[test]
    fn run_migrates_then_succeeds() {
        let graph = Graph {
            resources: ResourceGraph {
                nodes: vec![
                    ResourceNode {
                        id: ResourceId(1),
                        kind: ResourceKind::Buffer,
                        domain: Domain::Vram,
                        access: Access::ReadWrite,
                        byte_size: Some(80),
                    },
                    ResourceNode {
                        id: ResourceId(2),
                        kind: ResourceKind::Buffer,
                        domain: Domain::Vram,
                        access: Access::ReadWrite,
                        byte_size: Some(80),
                    },
                ],
            },
            tasks: TaskGraph {
                nodes: vec![],
                edges: vec![],
            },
        };
        let mut rt = Runtime::with_capacity(CapacityPolicy::MigrateToHost {
            domain: Domain::Vram,
            bytes: 100,
        });
        rt.register_backend(Box::new(NoopBackend));
        rt.run(&graph).expect("migrate then run");
    }
}
