use ucf_ir::{Graph, Objective, Priority};

use crate::backend::Backend;
use crate::error::{Error, Result};

pub struct Scheduler {
    backends: Vec<Box<dyn Backend>>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self { backends: Vec::new() }
    }

    pub fn register(&mut self, backend: Box<dyn Backend>) {
        self.backends.push(backend);
    }

    pub fn execute(&mut self, graph: &Graph) -> Result<()> {
        graph.validate()?;

        let order = graph.tasks.topological_order()?;
        for task_id in order {
            let task = graph
                .tasks
                .nodes
                .iter()
                .find(|t| t.id == task_id)
                .expect("task in order must exist");
            let backend_idx = self.pick_backend_index(task)?;
            self.backends[backend_idx].submit_task(graph, task)?;
        }
        Ok(())
    }

    fn pick_backend_index(&self, task: &ucf_ir::TaskNode) -> Result<usize> {
        let prefer_cuda = matches!(task.kind, ucf_ir::TaskKind::MatMul | ucf_ir::TaskKind::Dispatch);
        let prefer_graphics =
            matches!(task.kind, ucf_ir::TaskKind::Raster | ucf_ir::TaskKind::RtTrace);

        let idx = self
            .backends
            .iter()
            .position(|b| {
                if prefer_cuda {
                    b.name().contains("cuda")
                } else if prefer_graphics {
                    b.name().contains("dx") || b.name().contains("vulkan")
                } else {
                    true
                }
            })
            .or_else(|| {
                if task.priority == Priority::Interactive
                    || matches!(task.objective, Objective::MinLatency { .. })
                {
                    self.backends.iter().position(|b| b.name().contains("dx"))
                } else {
                    None
                }
            })
            .or_else(|| if self.backends.is_empty() { None } else { Some(0) });

        idx.ok_or_else(|| Error::NoBackend(format!("{task_kind:?}", task_kind = task.kind)))
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}
