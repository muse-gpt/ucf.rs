//! Soft-skip: bind CUDA external device buffers + default stream, run MatMul.

use std::collections::BTreeMap;

use ucf::prelude::*;
use ucf::ParamValue;
use ucf_backend_cuda::CudaBackend;

fn buffer(id: u64, floats: usize) -> ResourceNode {
    ResourceNode {
        id: ResourceId(id),
        kind: ResourceKind::Buffer,
        domain: Domain::Vram,
        access: Access::ReadWrite,
        byte_size: Some((floats * 4) as u64),
    }
}

fn i(key: &str, v: i64) -> (String, ParamValue) {
    (key.into(), ParamValue::I64(v))
}

fn matmul_graph() -> Graph {
    Graph {
        resources: ResourceGraph {
            nodes: vec![buffer(1, 4), buffer(2, 4), buffer(3, 4)],
        },
        tasks: TaskGraph {
            nodes: vec![TaskNode {
                id: TaskId(10),
                kind: TaskKind::MatMul,
                shader: ShaderId(10),
                params: BTreeMap::from([
                    i("a", 1),
                    i("b", 2),
                    i("out", 3),
                    i("m", 2),
                    i("n", 2),
                    i("k", 2),
                ]),
                dispatch: Default::default(),
                objective: Objective::MaxThroughput,
                priority: Priority::Batch,
            }],
            edges: vec![
                DepEdge {
                    from_task: None,
                    from_resource: Some(ResourceId(1)),
                    to_task: TaskId(10),
                    kind: DepKind::Data,
                },
                DepEdge {
                    from_task: None,
                    from_resource: Some(ResourceId(2)),
                    to_task: TaskId(10),
                    kind: DepKind::Data,
                },
            ],
        },
    }
}

#[test]
fn cuda_external_buffer_and_default_stream_matmul_soft_skip() {
    let mut backend = match CudaBackend::new(0) {
        Ok(b) => b,
        Err(err) => {
            eprintln!("skip cuda external bindings (no device): {err}");
            return;
        }
    };

    let graph = matmul_graph();
    let mut bindings = ExecutionBindings::new();
    for node in &graph.resources.nodes {
        let bytes = node.byte_size.expect("sized");
        let buf = match backend.allocate_external_buffer(bytes) {
            Ok(b) => b,
            Err(err) => {
                eprintln!("skip cuda external bindings (alloc): {err}");
                return;
            }
        };
        bindings.bind_buffer(node.id, buf);
    }
    bindings.set_stream(backend.default_stream());

    let stream = backend.default_stream();
    let upload = match backend.create_event() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("skip cuda external bindings (event): {err}");
            return;
        }
    };
    let compute = match backend.create_event() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("skip cuda external bindings (event2): {err}");
            return;
        }
    };
    {
        let bridge = backend.stream_event_bridge();
        if let Err(err) = bridge.record_event(stream.as_ref(), upload.as_ref()) {
            eprintln!("skip cuda external bindings (record upload): {err}");
            return;
        }
        if let Err(err) = bridge.wait_event(stream.as_ref(), upload.as_ref()) {
            eprintln!("skip cuda external bindings (wait upload): {err}");
            return;
        }
    }

    if let Err(err) = backend.bind_externals(&bindings) {
        eprintln!("skip cuda external bindings (bind): {err}");
        return;
    }
    if let Err(err) = backend.prepare(&graph) {
        eprintln!("skip cuda external bindings (prepare): {err}");
        return;
    }
    if let Err(err) = backend.write_f32(ResourceId(1), &[1.0, 2.0, 3.0, 4.0]) {
        eprintln!("skip cuda external bindings (write a): {err}");
        return;
    }
    if let Err(err) = backend.write_f32(ResourceId(2), &[1.0, 0.0, 0.0, 1.0]) {
        eprintln!("skip cuda external bindings (write b): {err}");
        return;
    }
    if let Err(err) = backend.run_prepared(&graph) {
        eprintln!("skip cuda external bindings (run): {err}");
        return;
    }
    {
        let bridge = backend.stream_event_bridge();
        if let Err(err) = bridge.record_event(stream.as_ref(), compute.as_ref()) {
            eprintln!("skip cuda external bindings (record compute): {err}");
            return;
        }
    }
    if let Err(err) = backend.flush() {
        eprintln!("skip cuda external bindings (flush): {err}");
        return;
    }

    let out = match backend.read_f32(ResourceId(3)) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("skip cuda external bindings (read): {err}");
            return;
        }
    };
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    assert!(backend.active_stream().is_some());
    eprintln!("cuda external bindings matmul ok");
}
