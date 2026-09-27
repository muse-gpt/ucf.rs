use ucf::prelude::*;

#[test]
fn facade_graph_roundtrip() {
    let graph = Graph {
        resources: ResourceGraph::default(),
        tasks: TaskGraph {
            nodes: vec![TaskNode {
                id: TaskId(1),
                kind: TaskKind::Dispatch,
                shader: ShaderId(1),
                params: Default::default(),
                dispatch: Default::default(),
                objective: Objective::MaxThroughput,
                priority: Priority::Batch,
            }],
            edges: vec![],
        },
    };

    let json = serde_json::to_string(&graph).expect("serialize");
    let decoded: Graph = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(graph, decoded);
}
