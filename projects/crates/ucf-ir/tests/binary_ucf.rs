use ucf_ir::{
    decode, encode, Access, DepEdge, DepKind, Domain, Error, Graph, Objective, Priority,
    ResourceGraph, ResourceId, ResourceKind, ResourceNode, ShaderId, TaskGraph, TaskId, TaskKind,
    TaskNode, MAGIC, WIRE_MAJOR,
};

fn sample_graph() -> Graph {
    Graph {
        resources: ResourceGraph {
            nodes: vec![ResourceNode {
                id: ResourceId(1),
                kind: ResourceKind::Buffer,
                domain: Domain::Vram,
                access: Access::ReadWrite,
                byte_size: Some(4096),
            }],
        },
        tasks: TaskGraph {
            nodes: vec![TaskNode {
                id: TaskId(10),
                kind: TaskKind::MatMul,
                shader: ShaderId(100),
                params: Default::default(),
                dispatch: Default::default(),
                objective: Objective::MaxThroughput,
                priority: Priority::Batch,
            }],
            edges: vec![DepEdge {
                from_task: None,
                from_resource: Some(ResourceId(1)),
                to_task: TaskId(10),
                kind: DepKind::Data,
            }],
        },
    }
}

#[test]
fn binary_ucf_roundtrip() {
    let graph = sample_graph();
    let bytes = encode(&graph).expect("encode");
    assert_eq!(&bytes[0..4], MAGIC.as_slice());
    assert_eq!(
        u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
        WIRE_MAJOR
    );
    let decoded = decode(&bytes).expect("decode");
    assert_eq!(graph, decoded);
    decoded.validate().expect("valid");
}

#[test]
fn binary_ucf_rejects_bad_magic() {
    let mut bytes = encode(&sample_graph()).expect("encode");
    bytes[0] = b'X';
    assert_eq!(decode(&bytes), Err(Error::InvalidMagic));
}

#[test]
fn binary_ucf_rejects_unsupported_major() {
    let mut bytes = encode(&sample_graph()).expect("encode");
    bytes[4..8].copy_from_slice(&999u32.to_le_bytes());
    assert_eq!(
        decode(&bytes),
        Err(Error::UnsupportedWireMajor {
            found: 999,
            supported: WIRE_MAJOR,
        })
    );
}

#[test]
fn binary_ucf_rejects_truncated_header() {
    assert!(matches!(
        decode(b"UCF"),
        Err(Error::TruncatedBinary(_))
    ));
}
