//! IR must not carry graphics backend dirt (PSO / Heap / Barrier object names).

use ucf_ir::{
    Access, DepEdge, DepKind, Domain, Graph, Objective, Priority, ResourceGraph, ResourceId,
    ResourceKind, ResourceNode, ShaderId, TaskGraph, TaskId, TaskKind, TaskNode,
};
use ucf_types::{ParamValue, ShaderOp, ShaderProgram};

const FORBIDDEN: &[&str] = &[
    "PSO",
    "PsoPrecache",
    "DescriptorHeap",
    "DescriptorBuffer",
    "PipelineState",
    "ID3D12",
    "VkPipeline",
    "Barrier",
    "Heap",
];

fn sample_graph() -> Graph {
    Graph {
        resources: ResourceGraph {
            nodes: vec![
                ResourceNode {
                    id: ResourceId(1),
                    kind: ResourceKind::Buffer,
                    domain: Domain::Vram,
                    access: Access::ReadWrite,
                    byte_size: Some(64),
                },
                ResourceNode {
                    id: ResourceId(2),
                    kind: ResourceKind::Texture,
                    domain: Domain::Vram,
                    access: Access::ReadWrite,
                    byte_size: Some(256),
                },
            ],
        },
        tasks: TaskGraph {
            nodes: vec![
                TaskNode {
                    id: TaskId(10),
                    kind: TaskKind::Raster,
                    shader: ShaderId(1),
                    params: [(
                        "draw".into(),
                        ParamValue::Str("tri".into()),
                    )]
                    .into(),
                    dispatch: Default::default(),
                    objective: Objective::MinLatency { deadline_micros: 8_000 },
                    priority: Priority::Interactive,
                },
                TaskNode {
                    id: TaskId(11),
                    kind: TaskKind::MatMul,
                    shader: ShaderId(2),
                    params: Default::default(),
                    dispatch: Default::default(),
                    objective: Objective::MaxThroughput,
                    priority: Priority::Batch,
                },
                TaskNode {
                    id: TaskId(12),
                    kind: TaskKind::Custom("attention".into()),
                    shader: ShaderId(3),
                    params: Default::default(),
                    dispatch: Default::default(),
                    objective: Objective::MaxThroughput,
                    priority: Priority::Batch,
                },
            ],
            edges: vec![
                DepEdge {
                    from_task: Some(TaskId(10)),
                    from_resource: None,
                    to_task: TaskId(11),
                    kind: DepKind::Execution,
                },
                DepEdge {
                    from_task: None,
                    from_resource: Some(ResourceId(1)),
                    to_task: TaskId(11),
                    kind: DepKind::Data,
                },
            ],
        },
    }
}

fn assert_clean(label: &str, text: &str) {
    for needle in FORBIDDEN {
        assert!(
            !text.contains(needle),
            "{label} must not contain graphics dirt `{needle}`: {text}"
        );
    }
}

#[test]
fn graph_json_excludes_graphics_dirt_names() {
    let graph = sample_graph();
    let json = serde_json::to_string(&graph).expect("serialize");
    assert_clean("graph json", &json);
    graph.validate().expect("valid");
}

#[test]
fn task_and_resource_kind_debug_excludes_graphics_dirt() {
    let kinds = [
        format!("{:?}", TaskKind::Raster),
        format!("{:?}", TaskKind::Dispatch),
        format!("{:?}", TaskKind::Fill),
        format!("{:?}", TaskKind::MatMul),
        format!("{:?}", TaskKind::RtTrace),
        format!("{:?}", TaskKind::Copy),
        format!("{:?}", TaskKind::Custom("attention".into())),
        format!("{:?}", ResourceKind::Buffer),
        format!("{:?}", ResourceKind::Texture),
        format!("{:?}", ResourceKind::Tensor),
        format!("{:?}", ResourceKind::AccelStruct),
        format!("{:?}", ResourceKind::Sampler),
        format!("{:?}", DepKind::Data),
        format!("{:?}", DepKind::Memory),
        format!("{:?}", DepKind::Execution),
        format!("{:?}", Domain::Vram),
        format!("{:?}", Domain::Host),
    ];
    for text in kinds {
        assert_clean("kind debug", &text);
    }
}

#[test]
fn shader_ops_exclude_graphics_dirt_names() {
    let ops = [
        format!("{:?}", ShaderOp::MatMul),
        format!("{:?}", ShaderOp::Attention),
        format!("{:?}", ShaderOp::Rgba8Denoise),
        format!("{:?}", ShaderOp::MemCopy),
        format!("{:?}", ShaderOp::IotaFill { value: 1.0 }),
        format!("{:?}", ShaderProgram::attention("ucf_attention")),
    ];
    for text in ops {
        assert_clean("shader op", &text);
    }
}
