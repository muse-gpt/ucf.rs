//! Application contract: encode → decode → prepare → write → run → flush → readback.

use std::collections::BTreeMap;

use ucf::prelude::*;
use ucf::ParamValue;

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

fn reference_graph() -> Graph {
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

/// Long-lived reference: `.ucf` round-trip plus the frozen CPU application lifecycle.
#[test]
fn application_contract_encode_prepare_run_flush_readback() {
    let original = reference_graph();
    let bytes = encode(&original).expect("encode .ucf");
    assert_eq!(&bytes[0..4], UCF_MAGIC.as_slice());
    assert_eq!(
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        UCF_WIRE_MAJOR
    );

    let graph = decode(&bytes).expect("decode .ucf");
    assert_eq!(graph, original);

    let mut session = CpuSession::open();
    let caps = session.capabilities();
    assert!(caps.contains_backend("cpu"));

    session.prepare(&graph).expect("prepare");
    session
        .write_f32(ResourceId(1), &[1.0, 2.0, 3.0, 4.0])
        .expect("write");
    session.run_prepared(&graph).expect("run_prepared");
    session.flush().expect("flush");

    let out = session.read_f32(ResourceId(4)).expect("readback");
    assert_eq!(out, vec![3.0, 3.0, 7.0, 7.0]);

    let diag = session.diagnostics();
    assert!(diag.contains_kind("graph_validate"));
    assert!(diag.contains_kind("resource_prepare"));
    assert!(diag.contains_kind("task_submit"));
    assert!(diag.contains_kind("flush"));
}

#[test]
fn application_contract_maps_ir_errors() {
    let mut session = CpuSession::open();
    let bad = Graph {
        resources: ResourceGraph { nodes: vec![] },
        tasks: TaskGraph {
            nodes: vec![],
            edges: vec![DepEdge {
                from_task: None,
                from_resource: Some(ResourceId(1)),
                to_task: TaskId(1),
                kind: DepKind::Data,
            }],
        },
    };
    let err = session.prepare(&bad).expect_err("invalid graph");
    assert_eq!(err.code(), SchedulerErrorCode::Ir);
}
