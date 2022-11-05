use serde::{Deserialize, Serialize};

use ucf_types::{Domain, TaskId};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Optimization {
    Place { task: TaskId, domain: Domain },
    ChangeParallelism { task: TaskId, factor: u32 },
    TuneParam { task: TaskId, key: String, value: String },
    Reorder { before: TaskId, after: TaskId },
}
