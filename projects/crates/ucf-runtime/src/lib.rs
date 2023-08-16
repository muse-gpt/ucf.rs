//! UCF runtime: backend registration and graph execution.
#![warn(missing_docs)]

mod capacity;
mod diagnostics;
mod runtime;

pub use capacity::{
    apply_capacity, check_soft_limit, domain_bytes, migrate_overflow_to_host,
    migrate_overflow_to_host_logged, CapacityPolicy, ResourceMigration,
};
pub use diagnostics::{
    kinds as event_kinds, ExecutionDiagnostics, ExecutionEvent, ExecutionEventBuilder,
};
pub use runtime::Runtime;

use ucf_ir::Graph;
use ucf_scheduler::{NoopBackend, Result};

/// Validate a graph and schedule it on the noop backend (no hardware).
pub fn dry_run(graph: &Graph) -> Result<()> {
    graph.validate().map_err(ucf_scheduler::Error::from)?;
    let mut runtime = Runtime::new();
    runtime.register_backend(Box::new(NoopBackend));
    runtime.run(graph)
}
