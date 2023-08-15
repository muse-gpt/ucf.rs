//! External host buffers + immediate stream bindings on the CPU application path.

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

#[test]
fn external_host_buffers_and_immediate_stream_run() {
    let graph = reference_graph();
    let mut session = CpuSession::open();

    for node in &graph.resources.nodes {
        let bytes = node.byte_size.expect("sized");
        session
            .bind_host_buffer(node.id, bytes)
            .expect("bind external");
    }
    session.set_stream(immediate_stream());

    let stream = immediate_stream();
    let upload = immediate_event();
    let compute = immediate_event();
    session
        .record_event(stream.as_ref(), upload.as_ref())
        .expect("record upload");
    session
        .wait_event(stream.as_ref(), upload.as_ref())
        .expect("upload → compute");
    session
        .record_event(stream.as_ref(), compute.as_ref())
        .expect("record compute");

    session.prepare(&graph).expect("prepare");
    session
        .write_f32(ResourceId(1), &[1.0, 2.0, 3.0, 4.0])
        .expect("write");
    session.run_prepared(&graph).expect("run");
    session.flush().expect("flush");

    let out = session.read_f32(ResourceId(4)).expect("readback");
    assert_eq!(out, vec![3.0, 3.0, 7.0, 7.0]);
    let diag = session.diagnostics();
    assert!(diag.contains_kind("resource_bind"));
    assert!(diag.contains_kind("event_signal"));
    assert!(diag.contains_kind("stream_wait"));
    assert!(diag.contains_kind("resource_upload"));
    assert!(diag.contains_kind("readback"));
}

#[test]
fn reject_foreign_store_external_buffer() {
    let foreign = shared_store();
    let buf = HostExternalBuffer::allocate(foreign, ResourceId(1), 16).expect("alloc");
    let mut bindings = ExecutionBindings::new();
    bindings.bind_buffer(ResourceId(1), buf);

    let store = shared_store();
    let mut backend = CpuBackend::with_store(store);
    let err = backend
        .bind_externals(&bindings)
        .expect_err("foreign store");
    assert_eq!(err.code(), SchedulerErrorCode::Backend);
}
