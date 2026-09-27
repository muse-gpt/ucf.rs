mod capacity;
mod runtime;

pub use capacity::CapacityPolicy;
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
