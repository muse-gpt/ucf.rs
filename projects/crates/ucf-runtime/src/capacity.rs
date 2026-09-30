use ucf_ir::{Domain, Graph, ResourceId};
use ucf_scheduler::{Error, Result};

/// One logical domain change applied by capacity policy before run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceMigration {
    /// Logical resource id moved.
    pub resource_id: ResourceId,
    /// Domain before migration.
    pub from: Domain,
    /// Domain after migration.
    pub to: Domain,
}

/// How the runtime hides finite VRAM / host memory from the user-facing IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacityPolicy {
    /// Assume unbounded logical capacity; no soft budget check.
    Infinite,
    /// Enforce a soft byte budget per domain before run (reject if over).
    SoftLimit { domain: Domain, bytes: u64 },
    /// Soft budget with automatic overflow migration to [`Domain::Host`].
    MigrateToHost { domain: Domain, bytes: u64 },
}

impl Default for CapacityPolicy {
    fn default() -> Self {
        Self::Infinite
    }
}

/// Sum of `byte_size` for resources in `domain` (missing sizes count as 0).
pub fn domain_bytes(graph: &Graph, domain: Domain) -> u64 {
    graph
        .resources
        .nodes
        .iter()
        .filter(|n| n.domain == domain)
        .map(|n| n.byte_size.unwrap_or(0))
        .sum()
}

/// Reject the graph when SoftLimit is exceeded.
pub fn check_soft_limit(graph: &Graph, policy: CapacityPolicy) -> Result<()> {
    match policy {
        CapacityPolicy::Infinite | CapacityPolicy::MigrateToHost { .. } => Ok(()),
        CapacityPolicy::SoftLimit { domain, bytes } => {
            let used = domain_bytes(graph, domain);
            if used > bytes {
                Err(Error::CapacityExceeded {
                    domain,
                    used,
                    limit: bytes,
                })
            } else {
                Ok(())
            }
        }
    }
}

/// Move overflow from `from` into `Domain::Host` until `budget` fits.
///
/// Resources are considered in ascending id order. Returns how many nodes moved.
pub fn migrate_overflow_to_host(graph: &mut Graph, from: Domain, budget: u64) -> usize {
    migrate_overflow_to_host_logged(graph, from, budget).len()
}

/// Like [`migrate_overflow_to_host`], but returns per-resource migration records.
pub fn migrate_overflow_to_host_logged(
    graph: &mut Graph,
    from: Domain,
    budget: u64,
) -> Vec<ResourceMigration> {
    let mut used = domain_bytes(graph, from);
    if used <= budget {
        return Vec::new();
    }
    let mut migrations = Vec::new();
    let mut ids: Vec<_> = graph
        .resources
        .nodes
        .iter()
        .filter(|n| n.domain == from)
        .map(|n| n.id)
        .collect();
    ids.sort_by_key(|id| id.0);
    for id in ids {
        if used <= budget {
            break;
        }
        if let Some(node) = graph.resources.nodes.iter_mut().find(|n| n.id == id) {
            let size = node.byte_size.unwrap_or(0);
            migrations.push(ResourceMigration {
                resource_id: id,
                from,
                to: Domain::Host,
            });
            node.domain = Domain::Host;
            used = used.saturating_sub(size);
        }
    }
    migrations
}

/// Apply capacity policy: migrate and/or reject before scheduling.
pub fn apply_capacity(graph: &mut Graph, policy: CapacityPolicy) -> Result<Vec<ResourceMigration>> {
    match policy {
        CapacityPolicy::Infinite => Ok(Vec::new()),
        CapacityPolicy::SoftLimit { .. } => {
            check_soft_limit(graph, policy)?;
            Ok(Vec::new())
        }
        CapacityPolicy::MigrateToHost { domain, bytes } => {
            let migrations = migrate_overflow_to_host_logged(graph, domain, bytes);
            let used = domain_bytes(graph, domain);
            if used > bytes {
                Err(Error::CapacityExceeded {
                    domain,
                    used,
                    limit: bytes,
                })
            } else {
                Ok(migrations)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_ir::{
        Access, ResourceGraph, ResourceId, ResourceKind, ResourceNode, TaskGraph,
    };

    fn buf(id: u64, domain: Domain, bytes: u64) -> ResourceNode {
        ResourceNode {
            id: ResourceId(id),
            kind: ResourceKind::Buffer,
            domain,
            access: Access::ReadWrite,
            byte_size: Some(bytes),
        }
    }

    #[test]
    fn soft_limit_rejects_over_budget() {
        let graph = Graph {
            resources: ResourceGraph {
                nodes: vec![buf(1, Domain::Vram, 100), buf(2, Domain::Vram, 50)],
            },
            tasks: TaskGraph {
                nodes: vec![],
                edges: vec![],
            },
        };
        let err = check_soft_limit(
            &graph,
            CapacityPolicy::SoftLimit {
                domain: Domain::Vram,
                bytes: 120,
            },
        )
        .expect_err("over budget");
        match err {
            Error::CapacityExceeded {
                domain,
                used,
                limit,
            } => {
                assert_eq!(domain, Domain::Vram);
                assert_eq!(used, 150);
                assert_eq!(limit, 120);
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn migrate_moves_smallest_ids_first() {
        let mut graph = Graph {
            resources: ResourceGraph {
                nodes: vec![
                    buf(1, Domain::Vram, 40),
                    buf(2, Domain::Vram, 40),
                    buf(3, Domain::Vram, 40),
                ],
            },
            tasks: TaskGraph {
                nodes: vec![],
                edges: vec![],
            },
        };
        let moved = migrate_overflow_to_host(&mut graph, Domain::Vram, 80);
        assert_eq!(moved, 1);
        assert_eq!(graph.resources.nodes[0].domain, Domain::Host);
        assert_eq!(graph.resources.nodes[1].domain, Domain::Vram);
        assert_eq!(domain_bytes(&graph, Domain::Vram), 80);
    }
}
