//! CPU application session: prepare → write → run → flush → readback.

use ucf_backend_cpu::{shared_store, CpuBackend, SharedHostStore};
use ucf_capability::CapabilityReport;
use ucf_ir::Graph;
use ucf_runtime::{ExecutionDiagnostics, Runtime};
use ucf_scheduler::{Error as SchedulerError, Result as SchedulerResult};
use ucf_types::ResourceId;

/// Host-CPU session implementing the frozen application contract surface.
///
/// Lifecycle: [`CpuSession::open`] → [`prepare`](Self::prepare) →
/// [`write_f32`](Self::write_f32) → [`run_prepared`](Self::run_prepared) →
/// [`flush`](Self::flush) → [`read_f32`](Self::read_f32).
pub struct CpuSession {
    store: SharedHostStore,
    runtime: Runtime,
}

impl CpuSession {
    /// Open a session with a fresh shared host store and CPU backend.
    pub fn open() -> Self {
        let store = shared_store();
        let mut runtime = Runtime::new();
        runtime.register_backend(Box::new(CpuBackend::with_store(store.clone())));
        Self { store, runtime }
    }

    /// Validate and allocate resources (idempotent when sizes match).
    pub fn prepare(&mut self, graph: &Graph) -> SchedulerResult<()> {
        self.runtime.prepare(graph)
    }

    /// Write little-endian `f32` values into a prepared resource.
    pub fn write_f32(&self, id: ResourceId, values: &[f32]) -> SchedulerResult<()> {
        self.store
            .lock()
            .map_err(|_| SchedulerError::Backend("cpu".into(), "host store poisoned".into()))?
            .write_f32(id, values)
            .map_err(|e| SchedulerError::Backend("cpu".into(), e.0))
    }

    /// Submit a previously prepared graph (no second prepare).
    pub fn run_prepared(&mut self, graph: &Graph) -> SchedulerResult<()> {
        self.runtime.run_prepared(graph)
    }

    /// Prepare then submit (convenient one-shot).
    pub fn run(&mut self, graph: &Graph) -> SchedulerResult<()> {
        self.runtime.run(graph)
    }

    /// Flush the CPU backend so subsequent reads observe submitted work.
    pub fn flush(&mut self) -> SchedulerResult<()> {
        self.runtime.flush()
    }

    /// Read little-endian `f32` values after flush.
    pub fn read_f32(&self, id: ResourceId) -> SchedulerResult<Vec<f32>> {
        self.store
            .lock()
            .map_err(|_| SchedulerError::Backend("cpu".into(), "host store poisoned".into()))?
            .read_f32(id)
            .map_err(|e| SchedulerError::Backend("cpu".into(), e.0))
    }

    /// Capability rows for the registered CPU backend.
    pub fn capabilities(&self) -> CapabilityReport {
        self.runtime.capabilities()
    }

    /// Execution diagnostics recorded by the runtime.
    pub fn diagnostics(&self) -> &ExecutionDiagnostics {
        self.runtime.diagnostics()
    }

    /// Clear diagnostics between contract runs.
    pub fn clear_diagnostics(&mut self) {
        self.runtime.clear_diagnostics();
    }

    /// Shared host store (advanced adapters / tests).
    pub fn store(&self) -> &SharedHostStore {
        &self.store
    }
}
