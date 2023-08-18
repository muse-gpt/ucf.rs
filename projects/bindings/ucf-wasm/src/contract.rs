//! Reference application-contract graph（与 `ucf/tests/application_contract.rs` 同语义）。

use std::collections::BTreeMap;

use ucf::prelude::*;
use ucf::ParamValue;
use ucf::SchedulerErrorCode as ErrorCode;

const DIAG_GRAPH_VALIDATE: u32 = 1 << 0;
const DIAG_RESOURCE_PREPARE: u32 = 1 << 1;
const DIAG_RESOURCE_UPLOAD: u32 = 1 << 2;
const DIAG_TASK_SUBMIT: u32 = 1 << 3;
const DIAG_TASK_COMPLETE: u32 = 1 << 4;
const DIAG_FLUSH: u32 = 1 << 5;
const DIAG_READBACK: u32 = 1 << 6;

fn buffer(id: u64, floats: usize) -> ResourceNode {
    ResourceNode {
        id: ResourceId(id),
        kind: ResourceKind::Buffer,
        domain: Domain::Host,
        access: Access::ReadWrite,
        byte_size: Some((floats * 4) as u64),
    }
}

fn task(id: u64, kind: TaskKind, params: BTreeMap<String, ParamValue>) -> TaskNode {
    TaskNode {
        id: TaskId(id),
        kind,
        shader: ShaderId(id),
        params,
        dispatch: Default::default(),
        objective: Objective::MaxThroughput,
        priority: Priority::Batch,
    }
}

fn i(key: &str, v: i64) -> (String, ParamValue) {
    (key.into(), ParamValue::I64(v))
}

fn f(key: &str, v: f64) -> (String, ParamValue) {
    (key.into(), ParamValue::F64(v))
}

/// Copy → Fill → MatMul reference graph used by the application contract test.
pub fn reference_graph() -> Graph {
    Graph {
        resources: ResourceGraph {
            nodes: vec![
                buffer(1, 4),
                buffer(2, 4),
                buffer(3, 4),
                buffer(4, 4),
            ],
        },
        tasks: TaskGraph {
            nodes: vec![
                task(
                    10,
                    TaskKind::Copy,
                    BTreeMap::from([i("src", 1), i("dst", 2)]),
                ),
                task(
                    11,
                    TaskKind::Fill,
                    BTreeMap::from([i("dst", 3), f("value", 1.0)]),
                ),
                task(
                    12,
                    TaskKind::MatMul,
                    BTreeMap::from([
                        i("a", 2),
                        i("b", 3),
                        i("out", 4),
                        i("m", 2),
                        i("n", 2),
                        i("k", 2),
                    ]),
                ),
            ],
            edges: vec![
                DepEdge {
                    from_task: None,
                    from_resource: Some(ResourceId(1)),
                    to_task: TaskId(10),
                    kind: DepKind::Data,
                },
                DepEdge {
                    from_task: Some(TaskId(10)),
                    from_resource: None,
                    to_task: TaskId(12),
                    kind: DepKind::Execution,
                },
                DepEdge {
                    from_task: Some(TaskId(11)),
                    from_resource: None,
                    to_task: TaskId(12),
                    kind: DepKind::Execution,
                },
            ],
        },
    }
}

/// Stable negative code for C ABI (`0` = success).
pub fn error_code_to_i32(code: ErrorCode) -> i32 {
    match code {
        ErrorCode::Ir => -1,
        ErrorCode::NoBackend => -2,
        ErrorCode::Backend => -3,
        ErrorCode::CapacityExceeded => -4,
    }
}

/// Run encode → decode → prepare → write → run → flush → readback on CPU.
pub fn run_matmul_contract() -> Result<Vec<f32>, ErrorCode> {
    let original = reference_graph();
    let bytes = encode(&original).map_err(|_| ErrorCode::Ir)?;
    let graph = decode(&bytes).map_err(|_| ErrorCode::Ir)?;

    let mut session = CpuSession::open();
    if !session.capabilities().contains_backend("cpu") {
        return Err(ErrorCode::NoBackend);
    }

    session.prepare(&graph).map_err(|e| e.code())?;
    session
        .write_f32(ResourceId(1), &[1.0, 2.0, 3.0, 4.0])
        .map_err(|e| e.code())?;
    session.run_prepared(&graph).map_err(|e| e.code())?;
    session.flush().map_err(|e| e.code())?;
    session
        .read_f32(ResourceId(4))
        .map(|v| v.to_vec())
        .map_err(|e| e.code())
}

/// Bitmask of diagnostic kinds observed after [`run_matmul_contract`].
pub fn run_diag_mask() -> Result<u32, ErrorCode> {
    let original = reference_graph();
    let graph = decode(&encode(&original).map_err(|_| ErrorCode::Ir)?).map_err(|_| ErrorCode::Ir)?;

    let mut session = CpuSession::open();
    session.prepare(&graph).map_err(|e| e.code())?;
    session
        .write_f32(ResourceId(1), &[1.0, 2.0, 3.0, 4.0])
        .map_err(|e| e.code())?;
    session.run_prepared(&graph).map_err(|e| e.code())?;
    session.flush().map_err(|e| e.code())?;
    let _ = session.read_f32(ResourceId(4)).map_err(|e| e.code())?;

    let diag = session.diagnostics();
    let mut mask = 0u32;
    if diag.contains_kind(event_kinds::GRAPH_VALIDATE) {
        mask |= DIAG_GRAPH_VALIDATE;
    }
    if diag.contains_kind(event_kinds::RESOURCE_PREPARE) {
        mask |= DIAG_RESOURCE_PREPARE;
    }
    if diag.contains_kind(event_kinds::RESOURCE_UPLOAD) {
        mask |= DIAG_RESOURCE_UPLOAD;
    }
    if diag.contains_kind(event_kinds::TASK_SUBMIT) {
        mask |= DIAG_TASK_SUBMIT;
    }
    if diag.contains_kind(event_kinds::TASK_COMPLETE) {
        mask |= DIAG_TASK_COMPLETE;
    }
    if diag.contains_kind(event_kinds::FLUSH) {
        mask |= DIAG_FLUSH;
    }
    if diag.contains_kind(event_kinds::READBACK) {
        mask |= DIAG_READBACK;
    }
    Ok(mask)
}

/// Expected MatMul readback for the reference graph.
pub const EXPECTED_MATMUL: [f32; 4] = [3.0, 3.0, 7.0, 7.0];

/// Expected diagnostic mask bits after a successful contract run.
pub const EXPECTED_DIAG_MASK: u32 = DIAG_GRAPH_VALIDATE
    | DIAG_RESOURCE_PREPARE
    | DIAG_RESOURCE_UPLOAD
    | DIAG_TASK_SUBMIT
    | DIAG_TASK_COMPLETE
    | DIAG_FLUSH
    | DIAG_READBACK;
