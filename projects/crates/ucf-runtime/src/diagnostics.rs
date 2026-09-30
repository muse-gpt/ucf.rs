//! Stable execution event schema for downstream inspect / debugger adapters.
//!
//! UCF emits these payloads only. It does **not** depend on DXO inspect or
//! `spark-debugger`.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ucf_scheduler::ErrorCode;

/// Stable event category strings (adapters match on these, not free text).
pub mod kinds {
    /// Graph passed / failed [`ucf_ir::Graph::validate`].
    pub const GRAPH_VALIDATE: &str = "graph_validate";
    /// External buffers / streams bound into backends.
    pub const RESOURCE_BIND: &str = "resource_bind";
    /// Backend resource allocation / bind during prepare.
    pub const RESOURCE_PREPARE: &str = "resource_prepare";
    /// Host → device (or host store) upload.
    pub const RESOURCE_UPLOAD: &str = "resource_upload";
    /// Task(s) submitted to a backend.
    pub const TASK_SUBMIT: &str = "task_submit";
    /// Task completed (when the backend reports completion).
    pub const TASK_COMPLETE: &str = "task_complete";
    /// Placement crossed backends (flush of previous).
    pub const BACKEND_SWITCH: &str = "backend_switch";
    /// Stream waited on an event.
    pub const STREAM_WAIT: &str = "stream_wait";
    /// Event recorded / signaled on a stream.
    pub const EVENT_SIGNAL: &str = "event_signal";
    /// Module / kernel compile started or finished.
    pub const MODULE_COMPILE: &str = "module_compile";
    /// Module cache hit.
    pub const MODULE_CACHE_HIT: &str = "module_cache_hit";
    /// Module cache miss.
    pub const MODULE_CACHE_MISS: &str = "module_cache_miss";
    /// Device / host readback.
    pub const READBACK: &str = "readback";
    /// Backend flush / synchronize.
    pub const FLUSH: &str = "flush";
    /// Capacity policy moved a resource between domains (logical placement).
    pub const RESOURCE_MIGRATE: &str = "resource_migrate";
    /// Stable failure marker with [`super::ExecutionEvent::error_code`].
    pub const BACKEND_ERROR: &str = "backend_error";
}

/// One probeable execution step (downstream maps into inspect / debugger).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionEvent {
    /// Stable category string (see [`kinds`]).
    pub kind: String,
    /// Logical graph identity when known (often hash / caller-assigned).
    pub graph_id: Option<u64>,
    /// Optional task id when known.
    pub task_id: Option<u64>,
    /// Optional resource id when known.
    pub resource_id: Option<u64>,
    /// Backend or runtime label (`cpu`, `cuda`, `runtime`, …).
    pub backend: String,
    /// Device fingerprint string when known (adapter-filled).
    pub device_fingerprint: Option<String>,
    /// Stream / queue identifier when known.
    pub stream_id: Option<String>,
    /// True when this step performed (or observed) a host↔device transfer.
    pub host_transfer: bool,
    /// Unix-epoch millis when the step started (`None` if not stamped).
    pub started_unix_ms: Option<u64>,
    /// Wall duration if timed; `None` for instantaneous markers.
    pub duration: Option<Duration>,
    /// Stable error class when [`kinds::BACKEND_ERROR`] or a failed step.
    pub error_code: Option<ErrorCode>,
}

impl ExecutionEvent {
    /// Start a builder for `kind` on `backend`.
    pub fn builder(kind: impl Into<String>, backend: impl Into<String>) -> ExecutionEventBuilder {
        ExecutionEventBuilder {
            event: ExecutionEvent {
                kind: kind.into(),
                graph_id: None,
                task_id: None,
                resource_id: None,
                backend: backend.into(),
                device_fingerprint: None,
                stream_id: None,
                host_transfer: false,
                started_unix_ms: None,
                duration: None,
                error_code: None,
            },
        }
    }
}

/// Fluent builder for [`ExecutionEvent`].
#[derive(Debug, Clone)]
pub struct ExecutionEventBuilder {
    event: ExecutionEvent,
}

impl ExecutionEventBuilder {
    /// Set graph id.
    pub fn graph_id(mut self, id: u64) -> Self {
        self.event.graph_id = Some(id);
        self
    }

    /// Set task id.
    pub fn task_id(mut self, id: u64) -> Self {
        self.event.task_id = Some(id);
        self
    }

    /// Set resource id.
    pub fn resource_id(mut self, id: u64) -> Self {
        self.event.resource_id = Some(id);
        self
    }

    /// Set device fingerprint.
    pub fn device_fingerprint(mut self, fp: impl Into<String>) -> Self {
        self.event.device_fingerprint = Some(fp.into());
        self
    }

    /// Set stream / queue id.
    pub fn stream_id(mut self, id: impl Into<String>) -> Self {
        self.event.stream_id = Some(id.into());
        self
    }

    /// Mark host transfer involvement.
    pub fn host_transfer(mut self, yes: bool) -> Self {
        self.event.host_transfer = yes;
        self
    }

    /// Stamp start time as now (unix millis).
    pub fn stamp_now(mut self) -> Self {
        self.event.started_unix_ms = unix_millis_now();
        self
    }

    /// Attach wall duration from `started`.
    pub fn duration_since(mut self, started: Instant) -> Self {
        self.event.duration = Some(started.elapsed());
        self
    }

    /// Attach explicit duration.
    pub fn duration(mut self, d: Duration) -> Self {
        self.event.duration = Some(d);
        self
    }

    /// Attach stable error code.
    pub fn error_code(mut self, code: ErrorCode) -> Self {
        self.event.error_code = Some(code);
        self
    }

    /// Finish the event.
    pub fn build(self) -> ExecutionEvent {
        self.event
    }
}

fn unix_millis_now() -> Option<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|d| d.as_millis() as u64)
}

/// Accumulated events for one application session or run batch.
#[derive(Debug, Default, Clone)]
pub struct ExecutionDiagnostics {
    events: Vec<ExecutionEvent>,
    /// Caller-assigned graph id stamped on subsequent events when set.
    active_graph_id: Option<u64>,
}

impl ExecutionDiagnostics {
    /// Empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow recorded events in order.
    pub fn events(&self) -> &[ExecutionEvent] {
        &self.events
    }

    /// Graph id applied to new events (if any).
    pub fn active_graph_id(&self) -> Option<u64> {
        self.active_graph_id
    }

    /// Set graph id for subsequent pushes (does not rewrite past events).
    pub fn set_graph_id(&mut self, id: Option<u64>) {
        self.active_graph_id = id;
    }

    /// Drop all events and clear active graph id.
    pub fn clear(&mut self) {
        self.events.clear();
        self.active_graph_id = None;
    }

    /// Append a fully built event (fills `graph_id` from active when unset).
    pub fn push(&mut self, mut event: ExecutionEvent) {
        if event.graph_id.is_none() {
            event.graph_id = self.active_graph_id;
        }
        self.events.push(event);
    }

    /// Append a marker without timing (compat helper).
    pub fn push_simple(
        &mut self,
        kind: impl Into<String>,
        task_id: Option<u64>,
        resource_id: Option<u64>,
        backend: impl Into<String>,
    ) {
        let mut b = ExecutionEvent::builder(kind, backend).stamp_now();
        if let Some(t) = task_id {
            b = b.task_id(t);
        }
        if let Some(r) = resource_id {
            b = b.resource_id(r);
        }
        self.push(b.build());
    }

    /// Append a timed span starting at `started` (compat helper).
    pub fn push_timed(
        &mut self,
        kind: impl Into<String>,
        task_id: Option<u64>,
        resource_id: Option<u64>,
        backend: impl Into<String>,
        started: Instant,
    ) {
        let mut b = ExecutionEvent::builder(kind, backend)
            .stamp_now()
            .duration_since(started);
        if let Some(t) = task_id {
            b = b.task_id(t);
        }
        if let Some(r) = resource_id {
            b = b.resource_id(r);
        }
        self.push(b.build());
    }

    /// Record a failure marker with stable [`ErrorCode`].
    pub fn push_error(&mut self, backend: impl Into<String>, code: ErrorCode) {
        self.push(
            ExecutionEvent::builder(kinds::BACKEND_ERROR, backend)
                .stamp_now()
                .error_code(code)
                .build(),
        );
    }

    /// True if any event of `kind` was recorded.
    pub fn contains_kind(&self, kind: &str) -> bool {
        self.events.iter().any(|e| e.kind == kind)
    }

    /// First event of `kind`, if any.
    pub fn first_of_kind(&self, kind: &str) -> Option<&ExecutionEvent> {
        self.events.iter().find(|e| e.kind == kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_and_graph_id_propagation() {
        let mut d = ExecutionDiagnostics::new();
        d.set_graph_id(Some(42));
        d.push_simple(kinds::GRAPH_VALIDATE, None, None, "runtime");
        let e = d.first_of_kind(kinds::GRAPH_VALIDATE).unwrap();
        assert_eq!(e.graph_id, Some(42));
        assert!(e.started_unix_ms.is_some());
    }

    #[test]
    fn push_error_records_code() {
        let mut d = ExecutionDiagnostics::new();
        d.push_error("cpu", ErrorCode::Ir);
        let e = d.first_of_kind(kinds::BACKEND_ERROR).unwrap();
        assert_eq!(e.error_code, Some(ErrorCode::Ir));
    }
}
