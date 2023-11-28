use std::collections::BTreeMap;

use hardware_emitter::{try_program_from_task, ProgramLowerError};
use hardware_types::{Objective, Priority, ShaderId, TaskId, TaskKind, TaskNode};

#[test]
fn ton_custom_op_fails_strict_lower() {
    let task = TaskNode {
        id: TaskId(12),
        kind: TaskKind::Custom("ton_trace_geodesics".into()),
        shader: ShaderId(12),
        params: BTreeMap::new(),
        dispatch: Default::default(),
        objective: Objective::MaxThroughput,
        priority: Priority::Batch,
    };
    let err = try_program_from_task(&task).unwrap_err();
    assert_eq!(
        err,
        ProgramLowerError::UnsupportedCustomOp {
            name: "ton_trace_geodesics".into(),
        }
    );
}
