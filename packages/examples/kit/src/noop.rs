use ucf_capability::FeatureSet;
use ucf_ir::{Graph, TaskNode};
use ucf_scheduler::{Backend, Result};

pub struct NoopBackend;

impl Backend for NoopBackend {
    fn name(&self) -> &str {
        "noop"
    }

    fn features(&self) -> FeatureSet {
        FeatureSet::new()
    }

    fn submit_task(&mut self, graph: &Graph, _task: &TaskNode) -> Result<()> {
        graph.validate().map_err(ucf_scheduler::Error::from)
    }
}
