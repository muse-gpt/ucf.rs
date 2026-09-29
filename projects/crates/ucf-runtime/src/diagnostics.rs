//! Lightweight execution event log for application adapters.

use std::time::{Duration, Instant};

/// One probeable execution step (downstream maps into inspect / debugger).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionEvent {
    /// Stable category string (`graph_validate`, `resource_prepare`, …).
    pub kind: String,
    /// Optional task id when known.
    pub task_id: Option<u64>,
    /// Optional resource id when known.
    pub resource_id: Option<u64>,
    /// Backend or runtime label (`cpu`, `cuda`, `runtime`, …).
    pub backend: String,
    /// Wall duration if timed; `None` for instantaneous markers.
    pub duration: Option<Duration>,
}

/// Accumulated events for one application session or run batch.
#[derive(Debug, Default, Clone)]
pub struct ExecutionDiagnostics {
    events: Vec<ExecutionEvent>,
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

    /// Drop all events.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Append a marker without timing.
    pub fn push_simple(
        &mut self,
        kind: impl Into<String>,
        task_id: Option<u64>,
        resource_id: Option<u64>,
        backend: impl Into<String>,
    ) {
        self.events.push(ExecutionEvent {
            kind: kind.into(),
            task_id,
            resource_id,
            backend: backend.into(),
            duration: None,
        });
    }

    /// Append a timed span starting at `started`.
    pub fn push_timed(
        &mut self,
        kind: impl Into<String>,
        task_id: Option<u64>,
        resource_id: Option<u64>,
        backend: impl Into<String>,
        started: Instant,
    ) {
        self.events.push(ExecutionEvent {
            kind: kind.into(),
            task_id,
            resource_id,
            backend: backend.into(),
            duration: Some(started.elapsed()),
        });
    }

    /// True if any event of `kind` was recorded.
    pub fn contains_kind(&self, kind: &str) -> bool {
        self.events.iter().any(|e| e.kind == kind)
    }
}
