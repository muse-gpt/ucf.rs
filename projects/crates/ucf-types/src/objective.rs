use serde::{Deserialize, Serialize};

/// Scheduling goal attached to a task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Objective {
    /// Hard or soft latency bound (games, interactive rendering).
    MinLatency { deadline_micros: u64 },
    /// Throughput-first (DL training, batch inference).
    MaxThroughput,
    /// Keep hardware busy without a specific latency target.
    MaxUtilization,
    /// Power / energy budget.
    MinEnergy,
}

/// Preemption and reservation hint for the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Priority {
    Interactive,
    Streaming,
    Background,
    Batch,
}
