use std::collections::BTreeMap;

use ucf_ir::{Graph, Objective, TaskId};

/// Effective deadline (micros) per task after backward propagation along deps.
///
/// A task's effective deadline is the minimum of:
/// - its own [`Objective::MinLatency`] deadline, if any
/// - the effective deadlines of every dependent successor
///
/// Tasks without a reachable latency objective get `None`.
pub fn effective_deadlines(graph: &Graph) -> BTreeMap<TaskId, Option<u64>> {
    let mut own: BTreeMap<TaskId, Option<u64>> = BTreeMap::new();
    for task in &graph.tasks.nodes {
        let d = match task.objective {
            Objective::MinLatency { deadline_micros } => Some(deadline_micros),
            _ => None,
        };
        own.insert(task.id, d);
    }

    // successors: from -> [to]
    let mut successors: BTreeMap<TaskId, Vec<TaskId>> = BTreeMap::new();
    for task in &graph.tasks.nodes {
        successors.entry(task.id).or_default();
    }
    for edge in &graph.tasks.edges {
        if let Some(from) = edge.from_task {
            successors.entry(from).or_default().push(edge.to_task);
        }
    }

    // Reverse topo: start from sinks, pull min deadline backward.
    let order = match graph.tasks.topological_order() {
        Ok(o) => o,
        Err(_) => return own,
    };

    let mut effective = own;
    for task_id in order.into_iter().rev() {
        let mut best = effective.get(&task_id).copied().flatten();
        if let Some(succs) = successors.get(&task_id) {
            for &succ in succs {
                if let Some(sd) = effective.get(&succ).copied().flatten() {
                    best = Some(match best {
                        Some(bd) => bd.min(sd),
                        None => sd,
                    });
                }
            }
        }
        effective.insert(task_id, best);
    }
    effective
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap as Map;

    use ucf_ir::{
        Access, DepEdge, DepKind, Domain, Graph, Objective, Priority, ResourceGraph, ResourceId,
        ResourceKind, ResourceNode, ShaderId, TaskGraph, TaskKind, TaskNode,
    };

    fn task(id: u64, objective: Objective) -> TaskNode {
        TaskNode {
            id: TaskId(id),
            kind: TaskKind::Dispatch,
            shader: ShaderId(id),
            params: Map::new(),
            dispatch: Default::default(),
            objective,
            priority: Priority::Batch,
        }
    }

    fn edge(from: u64, to: u64) -> DepEdge {
        DepEdge {
            from_task: Some(TaskId(from)),
            from_resource: None,
            to_task: TaskId(to),
            kind: DepKind::Execution,
        }
    }

    #[test]
    fn propagates_successor_deadline_backward() {
        let graph = Graph {
            resources: ResourceGraph {
                nodes: vec![ResourceNode {
                    id: ResourceId(1),
                    kind: ResourceKind::Buffer,
                    domain: Domain::Vram,
                    access: Access::ReadWrite,
                    byte_size: Some(16),
                }],
            },
            tasks: TaskGraph {
                nodes: vec![
                    task(1, Objective::MaxThroughput),
                    task(2, Objective::MinLatency { deadline_micros: 1000 }),
                    task(3, Objective::MinLatency { deadline_micros: 500 }),
                ],
                edges: vec![edge(1, 2), edge(2, 3)],
            },
        };

        let d = effective_deadlines(&graph);
        assert_eq!(d.get(&TaskId(3)).copied().flatten(), Some(500));
        assert_eq!(d.get(&TaskId(2)).copied().flatten(), Some(500));
        assert_eq!(d.get(&TaskId(1)).copied().flatten(), Some(500));
    }

    #[test]
    fn keeps_tighter_own_deadline() {
        let graph = Graph {
            resources: ResourceGraph { nodes: vec![] },
            tasks: TaskGraph {
                nodes: vec![
                    task(1, Objective::MinLatency { deadline_micros: 100 }),
                    task(2, Objective::MinLatency { deadline_micros: 1000 }),
                ],
                edges: vec![edge(1, 2)],
            },
        };
        let d = effective_deadlines(&graph);
        assert_eq!(d.get(&TaskId(1)).copied().flatten(), Some(100));
        assert_eq!(d.get(&TaskId(2)).copied().flatten(), Some(1000));
    }
}
