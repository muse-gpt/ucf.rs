use std::collections::{BTreeMap, BTreeSet};

use ucf_ir::{Graph, Objective, Priority, TaskId};

use crate::backend::Backend;
use crate::deadline::effective_deadlines;
use crate::error::{Error, Result};

/// Places and submits tasks onto registered backends.
pub struct Scheduler {
    backends: Vec<Box<dyn Backend>>,
}

impl Scheduler {
    /// Empty scheduler with no backends.
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    /// Register a backend for placement.
    pub fn register(&mut self, backend: Box<dyn Backend>) {
        self.backends.push(backend);
    }

    /// Validate, prepare, then submit tasks in dependency order.
    ///
    /// Ready tasks prefer [`Priority::Interactive`] and tighter effective deadlines
    /// (see [`crate::effective_deadlines`]) before batch work.
    pub fn execute(&mut self, graph: &Graph) -> Result<()> {
        graph.validate()?;

        for backend in &mut self.backends {
            backend.prepare(graph)?;
        }

        let deadlines = effective_deadlines(graph);
        let order = schedule_order(graph, &deadlines)?;
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
        let prefer_cuda = matches!(
            task.kind,
            ucf_ir::TaskKind::MatMul | ucf_ir::TaskKind::Dispatch
        );
        let prefer_cpu = matches!(
            task.kind,
            ucf_ir::TaskKind::Copy | ucf_ir::TaskKind::Fill
        );
        let prefer_graphics =
            matches!(task.kind, ucf_ir::TaskKind::Raster | ucf_ir::TaskKind::RtTrace);

        let idx = self
            .backends
            .iter()
            .position(|b| {
                if prefer_cuda {
                    b.name().contains("cuda") || b.name() == "cpu"
                } else if prefer_cpu {
                    b.name() == "cpu"
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
            .or_else(|| {
                if self.backends.is_empty() {
                    None
                } else {
                    Some(0)
                }
            });

        idx.ok_or_else(|| Error::NoBackend(format!("{task_kind:?}", task_kind = task.kind)))
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

fn schedule_order(
    graph: &Graph,
    deadlines: &BTreeMap<TaskId, Option<u64>>,
) -> Result<Vec<TaskId>> {
    let mut indegree: BTreeMap<TaskId, usize> = BTreeMap::new();
    let mut successors: BTreeMap<TaskId, Vec<TaskId>> = BTreeMap::new();
    let mut nodes: BTreeMap<TaskId, &ucf_ir::TaskNode> = BTreeMap::new();

    for task in &graph.tasks.nodes {
        indegree.insert(task.id, 0);
        successors.insert(task.id, Vec::new());
        nodes.insert(task.id, task);
    }
    for edge in &graph.tasks.edges {
        if let Some(from) = edge.from_task {
            *indegree.entry(edge.to_task).or_insert(0) += 1;
            successors.entry(from).or_default().push(edge.to_task);
        }
    }

    let mut ready: BTreeSet<(i32, u64, TaskId)> = BTreeSet::new();
    for (&id, &deg) in &indegree {
        if deg == 0 {
            ready.insert(ready_key(nodes[&id], deadlines));
        }
    }

    let mut order = Vec::with_capacity(graph.tasks.nodes.len());
    while let Some(key) = ready.iter().next().copied() {
        ready.remove(&key);
        let id = key.2;
        order.push(id);
        for &succ in successors.get(&id).into_iter().flatten() {
            let deg = indegree.get_mut(&succ).expect("succ in indegree");
            *deg -= 1;
            if *deg == 0 {
                ready.insert(ready_key(nodes[&succ], deadlines));
            }
        }
    }

    if order.len() != graph.tasks.nodes.len() {
        return Err(Error::Ir(ucf_ir::Error::Cycle));
    }
    Ok(order)
}

fn ready_key(
    task: &ucf_ir::TaskNode,
    deadlines: &BTreeMap<TaskId, Option<u64>>,
) -> (i32, u64, TaskId) {
    let prio_rank = match task.priority {
        Priority::Interactive => 0,
        Priority::Streaming => 1,
        Priority::Background => 2,
        Priority::Batch => 3,
    };
    let deadline = deadlines
        .get(&task.id)
        .copied()
        .flatten()
        .unwrap_or(u64::MAX);
    (prio_rank, deadline, task.id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap as Map;
    use std::sync::{Arc, Mutex};

    use ucf_ir::{
        DepEdge, DepKind, Graph, Objective, Priority, ResourceGraph, ShaderId, TaskGraph, TaskKind,
        TaskNode,
    };

    use crate::noop::NoopBackend;

    struct OrderBackend {
        name: String,
        order: Arc<Mutex<Vec<TaskId>>>,
    }

    impl Backend for OrderBackend {
        fn name(&self) -> &str {
            &self.name
        }

        fn features(&self) -> ucf_capability::FeatureSet {
            ucf_capability::FeatureSet::new()
        }

        fn submit_task(&mut self, _graph: &Graph, task: &ucf_ir::TaskNode) -> Result<()> {
            self.order.lock().unwrap().push(task.id);
            Ok(())
        }
    }

    fn task(id: u64, priority: Priority, objective: Objective) -> TaskNode {
        TaskNode {
            id: TaskId(id),
            kind: TaskKind::Custom("x".into()),
            shader: ShaderId(id),
            params: Map::new(),
            dispatch: Default::default(),
            objective,
            priority,
        }
    }

    #[test]
    fn interactive_before_batch_when_independent() {
        let graph = Graph {
            resources: ResourceGraph { nodes: vec![] },
            tasks: TaskGraph {
                nodes: vec![
                    task(1, Priority::Batch, Objective::MaxThroughput),
                    task(2, Priority::Interactive, Objective::MinLatency { deadline_micros: 10 }),
                ],
                edges: vec![],
            },
        };
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut sched = Scheduler::new();
        sched.register(Box::new(OrderBackend {
            name: "cpu".into(),
            order: order.clone(),
        }));
        sched.execute(&graph).expect("execute");
        assert_eq!(*order.lock().unwrap(), vec![TaskId(2), TaskId(1)]);
    }

    #[test]
    fn tighter_deadline_first_when_same_priority() {
        let graph = Graph {
            resources: ResourceGraph { nodes: vec![] },
            tasks: TaskGraph {
                nodes: vec![
                    task(
                        1,
                        Priority::Batch,
                        Objective::MinLatency { deadline_micros: 200 },
                    ),
                    task(
                        2,
                        Priority::Batch,
                        Objective::MinLatency { deadline_micros: 50 },
                    ),
                ],
                edges: vec![],
            },
        };
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut sched = Scheduler::new();
        sched.register(Box::new(OrderBackend {
            name: "cpu".into(),
            order: order.clone(),
        }));
        // Also register noop so Custom can land on first backend (cpu).
        let _ = NoopBackend;
        sched.execute(&graph).expect("execute");
        assert_eq!(*order.lock().unwrap(), vec![TaskId(2), TaskId(1)]);
    }

    #[test]
    fn respects_dependency_before_priority() {
        let graph = Graph {
            resources: ResourceGraph { nodes: vec![] },
            tasks: TaskGraph {
                nodes: vec![
                    task(1, Priority::Batch, Objective::MaxThroughput),
                    task(2, Priority::Interactive, Objective::MaxThroughput),
                ],
                edges: vec![DepEdge {
                    from_task: Some(TaskId(1)),
                    from_resource: None,
                    to_task: TaskId(2),
                    kind: DepKind::Execution,
                }],
            },
        };
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut sched = Scheduler::new();
        sched.register(Box::new(OrderBackend {
            name: "cpu".into(),
            order: order.clone(),
        }));
        sched.execute(&graph).expect("execute");
        assert_eq!(*order.lock().unwrap(), vec![TaskId(1), TaskId(2)]);
    }
}
