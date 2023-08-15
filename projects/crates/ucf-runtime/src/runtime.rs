use ucf_capability::CapabilityReport;
use ucf_ir::Graph;
use ucf_optimize::{apply, Optimization};
use ucf_scheduler::{Backend, ExecutionBindings, Scheduler};

use crate::capacity::{apply_capacity, CapacityPolicy};
use crate::diagnostics::{kinds, ExecutionDiagnostics, ExecutionEvent};

/// Owns backends, capacity policy, and graph execution.
pub struct Runtime {
    scheduler: Scheduler,
    capacity: CapacityPolicy,
    diagnostics: ExecutionDiagnostics,
    bindings: ExecutionBindings,
}

impl Runtime {
    /// Default runtime with [`CapacityPolicy::Infinite`].
    pub fn new() -> Self {
        Self {
            scheduler: Scheduler::new(),
            capacity: CapacityPolicy::default(),
            diagnostics: ExecutionDiagnostics::default(),
            bindings: ExecutionBindings::new(),
        }
    }

    /// Runtime with an explicit capacity policy.
    pub fn with_capacity(capacity: CapacityPolicy) -> Self {
        Self {
            scheduler: Scheduler::new(),
            capacity,
            diagnostics: ExecutionDiagnostics::default(),
            bindings: ExecutionBindings::new(),
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

    /// Replace live external buffer / stream bindings (applied on next prepare).
    pub fn set_bindings(&mut self, bindings: ExecutionBindings) {
        self.bindings = bindings;
    }

    /// Borrow current bindings.
    pub fn bindings(&self) -> &ExecutionBindings {
        &self.bindings
    }

    /// Capability rows for every registered backend.
    pub fn capabilities(&self) -> CapabilityReport {
        self.scheduler.capabilities()
    }

    /// Accumulated execution diagnostics (application contract probe).
    pub fn diagnostics(&self) -> &ExecutionDiagnostics {
        &self.diagnostics
    }

    /// Mutable diagnostics (sessions stamp upload / readback).
    pub fn diagnostics_mut(&mut self) -> &mut ExecutionDiagnostics {
        &mut self.diagnostics
    }

    /// Assign a graph id stamped onto subsequent diagnostic events.
    pub fn set_graph_id(&mut self, id: Option<u64>) {
        self.diagnostics.set_graph_id(id);
    }

    /// Clear diagnostics between runs.
    pub fn clear_diagnostics(&mut self) {
        self.diagnostics.clear();
    }

    fn apply_bindings(&mut self) -> ucf_scheduler::Result<()> {
        if self.bindings.buffers.is_empty() && self.bindings.stream.is_none() {
            return Ok(());
        }
        self.diagnostics.push_simple(
            kinds::RESOURCE_BIND,
            None,
            None,
            "runtime",
        );
        let result = self.scheduler.bind_externals(&self.bindings);
        if let Err(ref err) = result {
            self.diagnostics.push_error("runtime", err.code());
        }
        result
    }

    /// Validate and prepare resources on every backend (idempotent on size match).
    pub fn prepare(&mut self, graph: &Graph) -> ucf_scheduler::Result<()> {
        self.diagnostics
            .push_simple(kinds::GRAPH_VALIDATE, None, None, "runtime");
        self.apply_bindings()?;
        let result = self.scheduler.prepare(graph);
        match &result {
            Ok(()) => self.diagnostics.push_simple(
                kinds::RESOURCE_PREPARE,
                None,
                None,
                "runtime",
            ),
            Err(err) => self.diagnostics.push_error("runtime", err.code()),
        }
        result
    }

    /// Flush every registered backend.
    pub fn flush(&mut self) -> ucf_scheduler::Result<()> {
        let result = self.scheduler.flush();
        match &result {
            Ok(()) => self
                .diagnostics
                .push_simple(kinds::FLUSH, None, None, "runtime"),
            Err(err) => self.diagnostics.push_error("runtime", err.code()),
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
        match &result {
            Ok(()) => {
                self.diagnostics.push(
                    ExecutionEvent::builder(kinds::TASK_SUBMIT, "runtime")
                        .stamp_now()
                        .build(),
                );
                self.diagnostics.push(
                    ExecutionEvent::builder(kinds::TASK_COMPLETE, "runtime")
                        .stamp_now()
                        .build(),
                );
            }
            Err(err) => self.diagnostics.push_error("runtime", err.code()),
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
            .push_simple(kinds::GRAPH_VALIDATE, None, None, "runtime");
        self.apply_bindings()?;
        let result = self.scheduler.execute(&graph);
        match &result {
            Ok(()) => {
                self.diagnostics.push_simple(
                    kinds::RESOURCE_PREPARE,
                    None,
                    None,
                    "runtime",
                );
                self.diagnostics
                    .push_simple(kinds::TASK_SUBMIT, None, None, "runtime");
                self.diagnostics
                    .push_simple(kinds::TASK_COMPLETE, None, None, "runtime");
            }
            Err(err) => self.diagnostics.push_error("runtime", err.code()),
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
        rt.set_graph_id(Some(7));
        rt.prepare(&graph).expect("prepare");
        rt.flush().expect("flush");
        assert!(rt.diagnostics().contains_kind(kinds::GRAPH_VALIDATE));
        assert!(rt.diagnostics().contains_kind(kinds::RESOURCE_PREPARE));
        assert!(rt.diagnostics().contains_kind(kinds::FLUSH));
        assert_eq!(
            rt.diagnostics()
                .first_of_kind(kinds::GRAPH_VALIDATE)
                .unwrap()
                .graph_id,
            Some(7)
        );
    }

    #[test]
    fn prepare_error_emits_backend_error_event() {
        let bad = Graph {
            resources: ResourceGraph { nodes: vec![] },
            tasks: TaskGraph {
                nodes: vec![],
                edges: vec![ucf_ir::DepEdge {
                    from_task: None,
                    from_resource: Some(ResourceId(1)),
                    to_task: ucf_ir::TaskId(1),
                    kind: ucf_ir::DepKind::Data,
                }],
            },
        };
        let mut rt = Runtime::new();
        rt.register_backend(Box::new(NoopBackend));
        let err = rt.prepare(&bad).expect_err("invalid");
        assert_eq!(err.code(), ucf_scheduler::ErrorCode::Ir);
        let e = rt
            .diagnostics()
            .first_of_kind(kinds::BACKEND_ERROR)
            .expect("error event");
        assert_eq!(e.error_code, Some(ucf_scheduler::ErrorCode::Ir));
    }
}
