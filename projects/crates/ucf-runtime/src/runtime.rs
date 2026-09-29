use ucf_capability::CapabilityReport;
use ucf_ir::Graph;
use ucf_optimize::{apply, Optimization};
use ucf_scheduler::{Backend, Scheduler};

use crate::capacity::{apply_capacity, CapacityPolicy};
use crate::diagnostics::ExecutionDiagnostics;

/// Owns backends, capacity policy, and graph execution.
pub struct Runtime {
    scheduler: Scheduler,
    capacity: CapacityPolicy,
    diagnostics: ExecutionDiagnostics,
}

impl Runtime {
    /// Default runtime with [`CapacityPolicy::Infinite`].
    pub fn new() -> Self {
        Self {
            scheduler: Scheduler::new(),
            capacity: CapacityPolicy::default(),
            diagnostics: ExecutionDiagnostics::default(),
        }
    }

    /// Runtime with an explicit capacity policy.
    pub fn with_capacity(capacity: CapacityPolicy) -> Self {
        Self {
            scheduler: Scheduler::new(),
            capacity,
            diagnostics: ExecutionDiagnostics::default(),
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

    /// Capability rows for every registered backend.
    pub fn capabilities(&self) -> CapabilityReport {
        self.scheduler.capabilities()
    }

    /// Accumulated execution diagnostics (application contract probe).
    pub fn diagnostics(&self) -> &ExecutionDiagnostics {
        &self.diagnostics
    }

    /// Clear diagnostics between runs.
    pub fn clear_diagnostics(&mut self) {
        self.diagnostics.clear();
    }

    /// Validate and prepare resources on every backend (idempotent on size match).
    pub fn prepare(&mut self, graph: &Graph) -> ucf_scheduler::Result<()> {
        self.diagnostics
            .push_simple("graph_validate", None, None, "runtime");
        let result = self.scheduler.prepare(graph);
        if result.is_ok() {
            self.diagnostics
                .push_simple("resource_prepare", None, None, "runtime");
        }
        result
    }

    /// Flush every registered backend.
    pub fn flush(&mut self) -> ucf_scheduler::Result<()> {
        let result = self.scheduler.flush();
        if result.is_ok() {
            self.diagnostics
                .push_simple("flush", None, None, "runtime");
        }
        result
    }

    /// Apply capacity, then submit a previously prepared graph (no second prepare).
    pub fn run_prepared(&mut self, graph: &Graph) -> ucf_scheduler::Result<()> {
        self.run_prepared_with_opts(graph, &[])
    }

    /// Optimizations + capacity + submit without re-preparing.
    pub fn run_prepared_with_opts(
        &mut self,
        graph: &Graph,
        opts: &[Optimization],
    ) -> ucf_scheduler::Result<()> {
        let mut graph = graph.clone();
        apply(&mut graph, opts)?;
        apply_capacity(&mut graph, self.capacity)?;
        let result = self.scheduler.submit_prepared(&graph);
        if result.is_ok() {
            self.diagnostics
                .push_simple("task_submit", None, None, "runtime");
        }
        result
    }

    /// Run a graph under the current capacity policy (prepare + submit).
    pub fn run(&mut self, graph: &Graph) -> ucf_scheduler::Result<()> {
        self.run_with_opts(graph, &[])
    }

    /// Apply optimizations, then capacity, then schedule (prepare + submit).
    pub fn run_with_opts(
        &mut self,
        graph: &Graph,
        opts: &[Optimization],
    ) -> ucf_scheduler::Result<()> {
        let mut graph = graph.clone();
        apply(&mut graph, opts)?;
        apply_capacity(&mut graph, self.capacity)?;
        self.diagnostics
            .push_simple("graph_validate", None, None, "runtime");
        let result = self.scheduler.execute(&graph);
        if result.is_ok() {
            self.diagnostics
                .push_simple("resource_prepare", None, None, "runtime");
            self.diagnostics
                .push_simple("task_submit", None, None, "runtime");
        }
        result
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

    #[test]
    fn prepare_flush_capabilities_record_diagnostics() {
        let graph = Graph {
            resources: ResourceGraph {
                nodes: vec![],
            },
            tasks: TaskGraph {
                nodes: vec![],
                edges: vec![],
            },
        };
        let mut rt = Runtime::new();
        rt.register_backend(Box::new(NoopBackend));
        assert!(!rt.capabilities().backends.is_empty());
        rt.prepare(&graph).expect("prepare");
        rt.flush().expect("flush");
        assert!(rt.diagnostics().contains_kind("graph_validate"));
        assert!(rt.diagnostics().contains_kind("resource_prepare"));
        assert!(rt.diagnostics().contains_kind("flush"));
    }
}
