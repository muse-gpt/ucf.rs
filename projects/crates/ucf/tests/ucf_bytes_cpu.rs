//! Encode a reference graph as `.ucf` bytes, decode, then run on the CPU backend.

use std::collections::BTreeMap;

use ucf::prelude::*;
use ucf::ParamValue;
use ucf_types::ResourceId as Rid;

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
fn encode_decode_run_copy_fill_matmul_on_cpu() {
    let original = reference_graph();
    let bytes = encode(&original).expect("encode .ucf");
    assert_eq!(&bytes[0..4], UCF_MAGIC.as_slice());
    assert_eq!(
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        UCF_WIRE_MAJOR
    );

    let graph = decode(&bytes).expect("decode .ucf");
    assert_eq!(graph, original);
    graph.validate().expect("valid");

    let store = shared_store();
    {
        let mut prep = CpuBackend::with_store(store.clone());
        prep.prepare(&graph).expect("cpu prepare");
    }
    store
        .lock()
        .unwrap()
        .write_f32(Rid(1), &[1.0, 2.0, 3.0, 4.0])
        .expect("seed");

    let mut runtime = Runtime::new();
    runtime.register_backend(Box::new(CpuBackend::with_store(store.clone())));
    runtime.run(&graph).expect("run");

    let out = store.lock().unwrap().read_f32(Rid(4)).expect("read");
    assert_eq!(out, vec![3.0, 3.0, 7.0, 7.0]);
}
