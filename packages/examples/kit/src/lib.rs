mod builder;
mod noop;

pub use builder::{chain_edges, GraphBuilder};
pub use noop::NoopBackend;

use ucf_ir::Graph;
use ucf_runtime::Runtime;

pub fn dry_run(graph: &Graph) -> Result<(), String> {
    graph.validate().map_err(|e| e.to_string())?;
    let mut runtime = Runtime::new();
    runtime.register_backend(Box::new(NoopBackend));
    runtime.run(graph).map_err(|e| e.to_string())
}
