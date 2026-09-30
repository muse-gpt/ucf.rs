//! CPU application session: prepare → write → run → flush → readback.

use std::sync::Arc;

use ucf_backend_cpu::{shared_store, CpuBackend, HostExternalBuffer, SharedHostStore};
use ucf_capability::CapabilityReport;
use ucf_ir::Graph;
use ucf_runtime::{
    event_kinds as kinds, ExecutionDiagnostics, ExecutionEvent, Runtime,
};
use ucf_scheduler::{
    Error as SchedulerError, ExecutionBindings, ImmediateBridge, ImmediateEvent, ImmediateStream,
    Result as SchedulerResult, StreamEventBridge,
};
use ucf_types::ResourceId;

/// Host-CPU session implementing the frozen application contract surface.
///
/// Lifecycle: [`CpuSession::open`] → optional [`bind_host_buffer`](Self::bind_host_buffer) /
/// [`set_stream`](Self::set_stream) → [`prepare`](Self::prepare) →
/// [`write_f32`](Self::write_f32) → [`run_prepared`](Self::run_prepared) →
/// [`flush`](Self::flush) → [`read_f32`](Self::read_f32).
pub struct CpuSession {
    store: SharedHostStore,
    runtime: Runtime,
    bindings: ExecutionBindings,
    bridge: ImmediateBridge,
}

impl CpuSession {
    /// Open a session with a fresh shared host store and CPU backend.
    pub fn open() -> Self {
        let store = shared_store();
        let mut runtime = Runtime::new();
        runtime.register_backend(Box::new(CpuBackend::with_store(store.clone())));
        Self {
            store,
            runtime,
            bindings: ExecutionBindings::new(),
            bridge: ImmediateBridge,
        }
    }

    /// Assign a graph id stamped onto subsequent diagnostic events.
    pub fn set_graph_id(&mut self, id: Option<u64>) {
        self.runtime.set_graph_id(id);
    }

    /// Allocate and bind a host external buffer for `id` (must match graph size).
    pub fn bind_host_buffer(&mut self, id: ResourceId, byte_size: u64) -> SchedulerResult<()> {
        let buf = HostExternalBuffer::allocate(self.store.clone(), id, byte_size)?;
        self.bindings.bind_buffer(id, buf);
        Ok(())
    }

    /// Prefer the CPU immediate stream for subsequent prepares.
    pub fn set_stream(&mut self, stream: Arc<ImmediateStream>) {
        self.bindings.set_stream(stream);
    }

    /// CPU stream/event bridge (upload→compute style waits are no-ops on host).
    pub fn stream_bridge(&self) -> &dyn StreamEventBridge {
        &self.bridge
    }

    /// Record `event` on `stream` and emit [`kinds::EVENT_SIGNAL`].
    pub fn record_event(
        &mut self,
        stream: &dyn ucf_scheduler::ExecStream,
        event: &dyn ucf_scheduler::ExecEvent,
    ) -> SchedulerResult<()> {
        let result = self.bridge.record_event(stream, event);
        match &result {
            Ok(()) => self.runtime.diagnostics_mut().push(
                ExecutionEvent::builder(kinds::EVENT_SIGNAL, "cpu")
                    .stream_id("immediate")
                    .stamp_now()
                    .build(),
            ),
            Err(err) => self.runtime.diagnostics_mut().push_error("cpu", err.code()),
        }
        result
    }

    /// Wait for `event` on `stream` and emit [`kinds::STREAM_WAIT`].
    pub fn wait_event(
        &mut self,
        stream: &dyn ucf_scheduler::ExecStream,
        event: &dyn ucf_scheduler::ExecEvent,
    ) -> SchedulerResult<()> {
        let result = self.bridge.wait_event(stream, event);
        match &result {
            Ok(()) => self.runtime.diagnostics_mut().push(
                ExecutionEvent::builder(kinds::STREAM_WAIT, "cpu")
                    .stream_id("immediate")
                    .stamp_now()
                    .build(),
            ),
            Err(err) => self.runtime.diagnostics_mut().push_error("cpu", err.code()),
        }
        result
    }

    /// Validate and allocate resources (idempotent when sizes match).
    pub fn prepare(&mut self, graph: &Graph) -> SchedulerResult<()> {
        self.runtime.set_bindings(self.bindings.clone());
        self.runtime.prepare(graph)
    }

    /// Write little-endian `f32` values into a prepared resource.
    pub fn write_f32(&mut self, id: ResourceId, values: &[f32]) -> SchedulerResult<()> {
        let result = self
            .store
            .lock()
            .map_err(|_| SchedulerError::Backend("cpu".into(), "host store poisoned".into()))?
            .write_f32(id, values)
            .map_err(|e| SchedulerError::Backend("cpu".into(), e.0));
        match &result {
            Ok(()) => self.runtime.diagnostics_mut().push(
                ExecutionEvent::builder(kinds::RESOURCE_UPLOAD, "cpu")
                    .resource_id(id.0)
                    .host_transfer(true)
                    .stream_id("immediate")
                    .stamp_now()
                    .build(),
            ),
            Err(err) => self.runtime.diagnostics_mut().push_error("cpu", err.code()),
        }
        result
    }

    /// Submit a previously prepared graph (no second prepare).
    pub fn run_prepared(&mut self, graph: &Graph) -> SchedulerResult<()> {
        self.runtime.run_prepared(graph)
    }

    /// Prepare then submit (convenient one-shot).
    pub fn run(&mut self, graph: &Graph) -> SchedulerResult<()> {
        self.runtime.set_bindings(self.bindings.clone());
        self.runtime.run(graph)
    }

    /// Flush the CPU backend so subsequent reads observe submitted work.
    pub fn flush(&mut self) -> SchedulerResult<()> {
        self.runtime.flush()
    }

    /// Read little-endian `f32` values after flush.
    pub fn read_f32(&mut self, id: ResourceId) -> SchedulerResult<Vec<f32>> {
        let result = self
            .store
            .lock()
            .map_err(|_| SchedulerError::Backend("cpu".into(), "host store poisoned".into()))?
            .read_f32(id)
            .map_err(|e| SchedulerError::Backend("cpu".into(), e.0));
        match &result {
            Ok(_) => self.runtime.diagnostics_mut().push(
                ExecutionEvent::builder(kinds::READBACK, "cpu")
                    .resource_id(id.0)
                    .host_transfer(true)
                    .stream_id("immediate")
                    .stamp_now()
                    .build(),
            ),
            Err(err) => self.runtime.diagnostics_mut().push_error("cpu", err.code()),
        }
        result
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

/// Convenience constructors for CPU immediate sync tokens.
pub fn immediate_stream() -> Arc<ImmediateStream> {
    Arc::new(ImmediateStream)
}

/// Convenience constructor for a CPU immediate event.
pub fn immediate_event() -> Arc<ImmediateEvent> {
    Arc::new(ImmediateEvent)
}
