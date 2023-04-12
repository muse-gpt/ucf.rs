//! Optimization hint descriptors (auto-tune skeleton).
#![warn(missing_docs)]

mod metrics;
mod optimization;
mod perf;

pub use metrics::{attention_flops, AttentionMetrics, matmul_flops, MatmulMetrics};
pub use optimization::Optimization;
pub use perf::{BackendPerfRow, PerfReport, TimedSample};

use ucf_ir::{Graph, Result};

/// Applies optimization hints to a graph. Skeleton: no-op until auto-tune lands.
pub fn apply(graph: &mut Graph, _opts: &[Optimization]) -> Result<()> {
    graph.validate()?;
    Ok(())
}
