use ucf_ir::Domain;

/// How the runtime hides finite VRAM / host memory from the user-facing IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapacityPolicy {
    /// Assume unbounded logical capacity; scheduler migrates across domains.
    Infinite,
    /// Enforce a soft byte budget per domain before eviction.
    SoftLimit { domain: Domain, bytes: u64 },
}

impl Default for CapacityPolicy {
    fn default() -> Self {
        Self::Infinite
    }
}
