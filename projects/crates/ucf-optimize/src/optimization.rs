use serde::{Deserialize, Serialize};

use ucf_types::{Domain, TaskId};

/// Declarative optimization hint (auto-tune may emit these later).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Optimization {
    /// Pin a task to a memory domain.
    Place {
        /// Target task.
        task: TaskId,
        /// Preferred domain.
        domain: Domain,
    },
    /// Scale launch parallelism.
    ChangeParallelism {
        /// Target task.
        task: TaskId,
        /// Multiplier relative to the baseline launch.
        factor: u32,
    },
    /// Override a string task parameter.
    TuneParam {
        /// Target task.
        task: TaskId,
        /// Parameter key.
        key: String,
        /// Parameter value.
        value: String,
    },
    /// Request a relative submit order (soft hint).
    Reorder {
        /// Task that should run earlier.
        before: TaskId,
        /// Task that should run later.
        after: TaskId,
    },
}
