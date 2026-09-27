use ucf_ir::Graph;
use ucf_optimize::{apply, Optimization};
use ucf_scheduler::{Backend, Scheduler};

use crate::capacity::CapacityPolicy;

pub struct Runtime {
    scheduler: Scheduler,
    capacity: CapacityPolicy,
}

impl Runtime {
    pub fn new() -> Self {
        Self {
            scheduler: Scheduler::new(),
            capacity: CapacityPolicy::default(),
        }
    }

    pub fn with_capacity(capacity: CapacityPolicy) -> Self {
        Self {
            scheduler: Scheduler::new(),
            capacity,
        }
    }

    pub fn register_backend(&mut self, backend: Box<dyn Backend>) {
        self.scheduler.register(backend);
    }

    pub fn capacity(&self) -> CapacityPolicy {
        self.capacity
    }

    pub fn run(&mut self, graph: &Graph) -> ucf_scheduler::Result<()> {
        self.run_with_opts(graph, &[])
    }

    pub fn run_with_opts(
        &mut self,
        graph: &Graph,
        opts: &[Optimization],
    ) -> ucf_scheduler::Result<()> {
        let mut graph = graph.clone();
        apply(&mut graph, opts)?;
        self.scheduler.execute(&graph)
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}
