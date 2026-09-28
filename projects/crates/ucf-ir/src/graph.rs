use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use ucf_types::{DepKind, ResourceId, ResourceNode, TaskId, TaskNode};

use crate::error::{Error, Result};

/// Directed dependency from a task and/or resource into a consumer task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepEdge {
    /// Producer task, when the edge is task-ordered.
    pub from_task: Option<TaskId>,
    /// Producer or operand resource, when the edge carries data/memory.
    pub from_resource: Option<ResourceId>,
    /// Consumer task that cannot start until the source is ready.
    pub to_task: TaskId,
    /// Kind of dependency.
    pub kind: DepKind,
}

/// Resource nodes belonging to one IR document.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceGraph {
    /// Resource declarations.
    pub nodes: Vec<ResourceNode>,
}

/// Tasks and dependency edges belonging to one IR document.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TaskGraph {
    /// Task declarations.
    pub nodes: Vec<TaskNode>,
    /// Ordering and data/memory edges.
    pub edges: Vec<DepEdge>,
}

/// Combined IR document exchanged between frontends, scheduler, and ucf-backends.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    /// Resource subgraph.
    pub resources: ResourceGraph,
    /// Task subgraph.
    pub tasks: TaskGraph,
}

impl Graph {
    /// Validate structural contracts, then ensure the task graph is acyclic.
    ///
    /// Checks:
    /// - resource and task ids are unique
    /// - every edge names an existing `to_task`
    /// - every edge has at least one source (`from_task` and/or `from_resource`)
    /// - named sources exist in this graph
    /// - task-to-task edges form a DAG
    pub fn validate(&self) -> Result<()> {
        self.check_unique_ids()?;
        self.check_edges()?;
        self.tasks.topological_order()?;
        Ok(())
    }

    fn check_unique_ids(&self) -> Result<()> {
        let mut resources = BTreeSet::new();
        for node in &self.resources.nodes {
            if !resources.insert(node.id) {
                return Err(Error::DuplicateResource(node.id.0));
            }
        }

        let mut tasks = BTreeSet::new();
        for node in &self.tasks.nodes {
            if !tasks.insert(node.id) {
                return Err(Error::DuplicateTask(node.id.0));
            }
        }
        Ok(())
    }

    fn check_edges(&self) -> Result<()> {
        let resources: BTreeSet<ResourceId> =
            self.resources.nodes.iter().map(|n| n.id).collect();
        let tasks: BTreeSet<TaskId> = self.tasks.nodes.iter().map(|n| n.id).collect();

        for edge in &self.tasks.edges {
            if !tasks.contains(&edge.to_task) {
                return Err(Error::UnknownTask(edge.to_task.0));
            }
            if edge.from_task.is_none() && edge.from_resource.is_none() {
                return Err(Error::EmptyEdgeSource(edge.to_task.0));
            }
            if let Some(from) = edge.from_task {
                if !tasks.contains(&from) {
                    return Err(Error::UnknownTask(from.0));
                }
            }
            if let Some(resource) = edge.from_resource {
                if !resources.contains(&resource) {
                    return Err(Error::UnknownResource(resource.0));
                }
            }
        }
        Ok(())
    }
}

impl TaskGraph {
    /// Returns tasks in dependency order. Fails on cycles.
    ///
    /// Only `from_task` edges contribute to ordering. Resource-only edges are
    /// data/memory annotations validated by [`Graph::validate`].
    pub fn topological_order(&self) -> Result<Vec<TaskId>> {
        let task_ids: BTreeSet<TaskId> = self.nodes.iter().map(|n| n.id).collect();
        let mut indegree = std::collections::BTreeMap::<TaskId, usize>::new();
        for node in &self.nodes {
            indegree.insert(node.id, 0);
        }
        for edge in &self.edges {
            let Some(from) = edge.from_task else {
                continue;
            };
            if !task_ids.contains(&from) || !task_ids.contains(&edge.to_task) {
                return Err(Error::UnknownTask(if !task_ids.contains(&from) {
                    from.0
                } else {
                    edge.to_task.0
                }));
            }
            *indegree.get_mut(&edge.to_task).expect("edge target") += 1;
        }

        let mut ready: Vec<TaskId> = indegree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(id, _)| *id)
            .collect();
        ready.sort_by_key(|id| id.0);

        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(task) = ready.first().copied() {
            ready.remove(0);
            order.push(task);

            for edge in &self.edges {
                if edge.from_task == Some(task) {
                    let entry = indegree.get_mut(&edge.to_task).expect("edge target");
                    *entry -= 1;
                    if *entry == 0 {
                        ready.push(edge.to_task);
                        ready.sort_by_key(|id| id.0);
                    }
                }
            }
        }

        if order.len() != self.nodes.len() {
            return Err(Error::Cycle);
        }
        Ok(order)
    }
}
