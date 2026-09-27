use ucf_ir::{
    Access, DepEdge, DepKind, Domain, Graph, Objective, Priority, ResourceGraph, ResourceId,
    ResourceKind, ResourceNode, ShaderId, TaskGraph, TaskId, TaskKind, TaskNode,
};

#[test]
fn graph_serde_roundtrip() {
    let graph = Graph {
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
    };

    let json = serde_json::to_string(&graph).expect("serialize");
    let decoded: Graph = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(graph, decoded);
    decoded.validate().expect("valid graph");
}
