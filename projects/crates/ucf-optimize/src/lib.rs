mod optimization;

pub use optimization::Optimization;

use ucf_ir::{Graph, Result};

/// Applies optimization hints to a graph. Skeleton: no-op until auto-tune lands.
pub fn apply(graph: &mut Graph, _opts: &[Optimization]) -> Result<()> {
    graph.validate()?;
    Ok(())
}
