use serde::{Deserialize, Serialize};

use ucf_types::{DepKind, ResourceId, ResourceNode, TaskId, TaskNode};

use crate::error::{Error, Result};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DepEdge {
    pub from_task: Option<TaskId>,
    pub from_resource: Option<ResourceId>,
    pub to_task: TaskId,
    pub kind: DepKind,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceGraph {
    pub nodes: Vec<ResourceNode>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TaskGraph {
    pub nodes: Vec<TaskNode>,
    pub edges: Vec<DepEdge>,
}

/// Combined IR document exchanged between frontends, scheduler, and ucf-backends.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Graph {
    pub resources: ResourceGraph,
    pub tasks: TaskGraph,
}

impl Graph {
    pub fn validate(&self) -> Result<()> {
        self.tasks.topological_order()?;
        Ok(())
    }
}

impl TaskGraph {
    /// Returns tasks in dependency order. Fails on cycles.
    pub fn topological_order(&self) -> Result<Vec<TaskId>> {
        let mut indegree = std::collections::BTreeMap::<TaskId, usize>::new();
        for node in &self.nodes {
            indegree.insert(node.id, 0);
        }
        for edge in &self.edges {
            if edge.from_task.is_some() {
                if let Some(deg) = indegree.get_mut(&edge.to_task) {
                    *deg += 1;
                }
            }
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
