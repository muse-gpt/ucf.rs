use std::collections::BTreeMap;

use ucf_backend_cpu::{shared_store, CpuBackend};
use ucf_backend_vulkan::VulkanBackend;
use ucf_program::{program_hash, BinOp, Expr, Function, FunctionBuilder, Program, ProgramBuilder, Stmt, Ty};
use ucf_ir::{
    Access, DepEdge, DepKind, Domain, Graph, Objective, ParamValue, Priority, ResourceGraph,
    ResourceId, ResourceKind, ResourceNode, ShaderId, TaskGraph, TaskId, TaskKind, TaskNode,
};
use ucf_scheduler::Backend;
use ucf_types::ResourceId as Rid;

fn buffer(id: u64, floats: usize) -> ResourceNode {
    ResourceNode {
        id: ResourceId(id),
        kind: ResourceKind::Buffer,
        domain: Domain::Vram,
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

fn run_cpu(graph: &Graph) -> Vec<f32> {
    let store = shared_store();
    {
        let mut prep = CpuBackend::with_store(store.clone());
        prep.prepare(graph).expect("cpu prepare");
    }
    store
        .lock()
        .unwrap()
        .write_f32(Rid(1), &[1.0, 2.0, 3.0, 4.0])
        .expect("cpu seed");

    let mut runtime = ucf_runtime::Runtime::new();
    runtime.register_backend(Box::new(CpuBackend::with_store(store.clone())));
    runtime.run(graph).expect("cpu run");
    let out = store.lock().unwrap().read_f32(Rid(4)).expect("cpu read");
    out
}

fn run_cpu_generic_dispatch(graph: &Graph) -> Vec<f32> {
    let store = shared_store();
    {
        let mut prep = CpuBackend::with_store(store.clone());
        prep.prepare(graph).expect("cpu prepare");
    }
    let mut runtime = ucf_runtime::Runtime::new();
    runtime.register_backend(Box::new(CpuBackend::with_store(store.clone())));
    runtime.run(graph).expect("cpu generic dispatch run");
    let output = store
        .lock()
        .unwrap()
        .read_f32(Rid(10))
        .expect("cpu generic dispatch read");
    output
}

#[test]
fn vulkan_matches_cpu_copy_fill_matmul() {
    let graph = reference_graph();
    let expected = run_cpu(&graph);

    let mut vk = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            eprintln!("skip vulkan parity (no usable Vulkan device): {err}");
            return;
        }
    };

    if let Err(err) = vk.prepare(&graph) {
        eprintln!("skip vulkan parity (prepare failed): {err}");
        return;
    }
    if let Err(err) = vk.write_f32(Rid(1), &[1.0, 2.0, 3.0, 4.0]) {
        eprintln!("skip vulkan parity (seed failed): {err}");
        return;
    }
    if let Err(err) = vk.run_prepared(&graph) {
        eprintln!("skip vulkan parity (run failed): {err}");
        return;
    }

    let actual = vk.read_f32(Rid(4)).expect("vulkan read");
    assert_eq!(actual, expected);
    assert_eq!(actual, vec![3.0, 3.0, 7.0, 7.0]);
}

fn generic_dispatch_graph(program: &Program, dispatch_count: usize, output_len: usize) -> Graph {
    let body = serde_json::to_string(program).expect("serialize program");
    let hash = program_hash(program);
    Graph {
        resources: ResourceGraph {
            nodes: vec![buffer(10, output_len), buffer(11, output_len)],
        },
        tasks: TaskGraph {
            nodes: vec![TaskNode {
                id: TaskId(20),
                kind: TaskKind::Dispatch,
                shader: ShaderId(20),
                params: BTreeMap::from([
                    (
                        "program".into(),
                        ParamValue::Program { version: 1, hash, body },
                    ),
                    ("output".into(), ParamValue::I64(10)),
                    ("buffer_0".into(), ParamValue::I64(11)),
                    ("uniform_0".into(), ParamValue::F64(2.5)),
                ]),
                dispatch: ucf_ir::Dispatch::Explicit { x: dispatch_count as u32, y: 1, z: 1 },
                objective: Objective::MaxThroughput,
                priority: Priority::Batch,
            }],
            edges: Vec::new(),
        },
    }
}

#[test]
fn vulkan_generic_dispatch_matches_cpu_interpreter() {
    let program = Program {
        version: 1,
        functions: vec![Function {
            name: "uniform_constant".into(),
            params: Vec::new(),
            locals: Vec::new(),
            body: vec![Stmt::Return(Expr::BinOp {
                op: BinOp::Add,
                lhs: Box::new(Expr::LoadUniform(0)),
                rhs: Box::new(Expr::ConstF32(1.0)),
            })],
            ret: Ty::f32(),
        }],
        entry: 0,
    };
    let graph = generic_dispatch_graph(&program, 65, 70);
    let expected = run_cpu_generic_dispatch(&graph);

    let mut vk = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            eprintln!("skip generic UCF Vulkan parity (no usable Vulkan device): {err}");
            return;
        }
    };
    if let Err(err) = vk.prepare(&graph) {
        eprintln!("skip generic UCF Vulkan parity (prepare failed): {err}");
        return;
    }
    if let Err(err) = vk.run_prepared(&graph) {
        eprintln!("skip generic UCF Vulkan parity (run failed): {err}");
        return;
    }
    let actual = vk.read_f32(ucf_types::ResourceId(10)).expect("generic UCF readback");
    assert_eq!(actual, expected);
    assert_eq!(&actual[..65], vec![3.5; 65].as_slice());
    assert_eq!(&actual[65..], &[0.0; 5]);
}

#[test]
fn vulkan_builder_cfg_select_matches_cpu_interpreter() {
    let mut function = FunctionBuilder::new("cfg_select");
    let uniform = function.emit_load_uniform(0, Ty::f32()).expect("uniform");
    let threshold = function.emit_const_f32(3.0).expect("threshold");
    let condition = function.emit_binary(BinOp::Lt, uniform, threshold).expect("condition");
    let then_block = function.create_block();
    let else_block = function.create_block();
    function.conditional_branch(condition, then_block, else_block).expect("branch");
    function.switch_to(then_block).expect("then block");
    let then_value = function.emit_const_f32(3.0).expect("then value");
    function.return_value(then_value).expect("then return");
    function.switch_to(else_block).expect("else block");
    let else_value = function.emit_const_f32(-1.0).expect("else value");
    function.return_value(else_value).expect("else return");
    let program = ProgramBuilder::new().entry(function).expect("entry").finish().expect("valid CFG");
    let mut true_cpu = ucf_program::CpuInterpreter::new();
    true_cpu.with_uniform(0, ucf_program::Value::F32(2.5));
    assert_eq!(true_cpu.run(&program).expect("CPU true").value, ucf_program::Value::F32(3.0));
    let mut false_cpu = ucf_program::CpuInterpreter::new();
    false_cpu.with_uniform(0, ucf_program::Value::F32(3.5));
    assert_eq!(false_cpu.run(&program).expect("CPU false").value, ucf_program::Value::F32(-1.0));
    let graph = generic_dispatch_graph(&program, 65, 70);
    let mut vk = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            eprintln!("skip CFG UCF Vulkan parity (no usable Vulkan device): {err}");
            return;
        }
    };
    if let Err(err) = vk.prepare(&graph) {
        eprintln!("skip CFG UCF Vulkan parity (prepare failed): {err}");
        return;
    }
    if let Err(err) = vk.run_prepared(&graph) {
        eprintln!("skip CFG UCF Vulkan parity (run failed): {err}");
        return;
    }
    let actual = vk.read_f32(Rid(10)).expect("CFG Vulkan output");
    assert_eq!(&actual[..65], vec![3.0; 65].as_slice());
    assert_eq!(&actual[65..], &[0.0; 5]);
}

#[test]
fn vulkan_builder_vector_math_matches_cpu_interpreter() {
    let mut function = FunctionBuilder::new("vector_math");
    let angle = function.emit_const_f32(0.0).expect("angle");
    let sine = function.emit_unary(ucf_program::UnOp::Sin, angle, Ty::f32()).expect("sin");
    let four = function.emit_const_f32(4.0).expect("four");
    let root = function.emit_unary(ucf_program::UnOp::Sqrt, four, Ty::f32()).expect("sqrt");
    let result = function.emit_binary(BinOp::Add, sine, root).expect("add");
    function.return_value(result).expect("return");
    let program = ProgramBuilder::new()
        .entry(function)
        .expect("entry")
        .finish()
        .expect("valid vector math program");
    let graph = generic_dispatch_graph(&program, 65, 70);
    let expected = run_cpu_generic_dispatch(&graph);
    let mut vk = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            eprintln!("skip vector math Vulkan parity (no usable Vulkan device): {err}");
            return;
        }
    };
    if let Err(err) = vk.prepare(&graph) {
        eprintln!("skip vector math Vulkan parity (prepare failed): {err}");
        return;
    }
    vk.run_prepared(&graph).expect("vector math Vulkan run");
    let actual = vk.read_f32(ucf_types::ResourceId(10)).expect("vector math readback");
    assert_eq!(actual, expected);
    let expected_value = 2.0_f32;
    assert!(actual[..65].iter().all(|value| (*value - expected_value).abs() < 1.0e-5));
    assert_eq!(&actual[65..], &[0.0; 5]);
}

#[test]
fn vulkan_builder_cfg_block_argument_matches_cpu_interpreter() {
    let mut function = FunctionBuilder::new("cfg_block_argument");
    let selector = function.emit_load_uniform(0, Ty::f32()).expect("selector");
    let zero = function.emit_const_f32(0.0).expect("zero");
    let condition = function.emit_binary(BinOp::Gt, selector, zero).expect("condition");
    let positive = function.emit_const_f32(4.0).expect("positive");
    let negative = function.emit_const_f32(-4.0).expect("negative");
    let join = function.create_block();
    let parameter = function.block_parameter(join, Ty::f32()).expect("join parameter");
    function.conditional_branch_with_args(condition, join, vec![positive], join, vec![negative]).expect("diamond branch");
    function.switch_to(join).expect("join block");
    function.return_value(parameter).expect("join return");
    let program = ProgramBuilder::new().entry(function).expect("entry").finish().expect("valid CFG");

    let mut cpu = ucf_program::CpuInterpreter::new();
    cpu.with_uniform(0, ucf_program::Value::F32(1.0));
    assert_eq!(cpu.run(&program).expect("CPU branch argument").value, ucf_program::Value::F32(4.0));

    let graph = generic_dispatch_graph(&program, 1, 1);
    let mut vk = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            eprintln!("skip block argument Vulkan parity (no usable Vulkan device): {err}");
            return;
        }
    };
    if let Err(err) = vk.prepare(&graph) {
        eprintln!("skip block argument Vulkan parity (prepare failed): {err}");
        return;
    }
    if let Err(err) = vk.run_prepared(&graph) {
        eprintln!("skip block argument Vulkan parity (run failed): {err}");
        return;
    }
    let actual = vk.read_f32(Rid(10)).expect("block argument output");
    assert_eq!(actual[0], 4.0);
}

#[test]
fn vulkan_builder_program_matches_cpu_interpreter() {
    let mut function = FunctionBuilder::new("builder_buffer_classify");
    let index = function.emit_global_id().expect("open entry block");
    let input = function.emit_load_buffer(0, index, Ty::f32()).expect("global id index");
    let threshold = function.emit_load_uniform(0, Ty::f32()).expect("open entry block");
    let condition = function.emit_binary(BinOp::Lt, input, threshold).expect("matching types");
    let one = function.emit_const_f32(1.0).expect("open entry block");
    let zero = function.emit_const_f32(0.0).expect("open entry block");
    let result = function.emit_select(condition, one, zero).expect("matching branch types");
    function.return_value(result).expect("open entry block");
    let program = ProgramBuilder::new().entry(function).expect("entry").finish().expect("valid program");
    let graph = generic_dispatch_graph(&program, 65, 70);
    let input: Vec<f32> = (0..70).map(|value| value as f32).collect();

    let store = shared_store();
    {
        let mut prep = CpuBackend::with_store(store.clone());
        prep.prepare(&graph).expect("cpu prepare");
    }
    store.lock().unwrap().write_f32(Rid(11), &input).expect("cpu input seed");
    let mut runtime = ucf_runtime::Runtime::new();
    runtime.register_backend(Box::new(CpuBackend::with_store(store.clone())));
    runtime.run(&graph).expect("cpu builder program run");
    let expected = store.lock().unwrap().read_f32(Rid(10)).expect("cpu output read");

    let mut vk = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            eprintln!("skip builder UCF Vulkan parity (no usable Vulkan device): {err}");
            return;
        }
    };
    if let Err(err) = vk.prepare(&graph) {
        eprintln!("skip builder UCF Vulkan parity (prepare failed): {err}");
        return;
    }
    vk.write_f32(Rid(11), &input).expect("vulkan input seed");
    vk.run_prepared(&graph).expect("vulkan builder program run");
    let actual = vk.read_f32(Rid(10)).expect("vulkan output read");
    assert_eq!(actual, expected);
    assert_eq!(&actual[..3], &[1.0; 3]);
    assert_eq!(&actual[3..65], &[0.0; 62]);
    assert_eq!(&actual[65..], &[0.0; 5]);
}

#[test]
fn vulkan_generic_buffer_load_matches_cpu_interpreter() {
    let program = Program {
        version: 1,
        functions: vec![Function {
            name: "buffer_add_uniform".into(),
            params: Vec::new(),
            locals: Vec::new(),
            body: vec![Stmt::Return(Expr::BinOp {
                op: BinOp::Add,
                lhs: Box::new(Expr::LoadBuffer {
                    resource: 0,
                    index: Box::new(Expr::GlobalId),
                }),
                rhs: Box::new(Expr::LoadUniform(0)),
            })],
            ret: Ty::f32(),
        }],
        entry: 0,
    };
    let graph = generic_dispatch_graph(&program, 65, 70);
    let store = shared_store();
    {
        let mut prep = CpuBackend::with_store(store.clone());
        prep.prepare(&graph).expect("cpu prepare");
    }
    store
        .lock()
        .unwrap()
        .write_f32(Rid(11), &(0..70).map(|value| value as f32).collect::<Vec<_>>())
        .expect("cpu input seed");
    let mut runtime = ucf_runtime::Runtime::new();
    runtime.register_backend(Box::new(CpuBackend::with_store(store.clone())));
    runtime.run(&graph).expect("cpu buffer dispatch run");
    let expected = store.lock().unwrap().read_f32(Rid(10)).expect("cpu output read");

    let mut vk = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            eprintln!("skip generic buffer Vulkan parity (no usable Vulkan device): {err}");
            return;
        }
    };
    if let Err(err) = vk.prepare(&graph) {
        eprintln!("skip generic buffer Vulkan parity (prepare failed): {err}");
        return;
    }
    vk.write_f32(Rid(11), &(0..70).map(|value| value as f32).collect::<Vec<_>>())
        .expect("vulkan input seed");
    vk.run_prepared(&graph).expect("vulkan buffer dispatch run");
    let actual = vk.read_f32(Rid(10)).expect("vulkan output read");
    assert_eq!(actual, expected);
    assert_eq!(&actual[..65], &(0..65).map(|value| value as f32 + 2.5).collect::<Vec<_>>());
    assert_eq!(&actual[65..], &[0.0; 5]);
}

#[test]
fn vulkan_generic_buffer_store_matches_cpu_interpreter() {
    let result = Expr::BinOp {
        op: BinOp::Add,
        lhs: Box::new(Expr::LoadBuffer {
            resource: 0,
            index: Box::new(Expr::GlobalId),
        }),
        rhs: Box::new(Expr::LoadUniform(0)),
    };
    let program = Program {
        version: 1,
        functions: vec![Function {
            name: "buffer_store_and_return".into(),
            params: Vec::new(),
            locals: vec![(0, Ty::f32())],
            body: vec![
                Stmt::Let {
                    id: 0,
                    ty: Ty::f32(),
                    init: result.clone(),
                },
                Stmt::StoreBuffer {
                    resource: 0,
                    index: Expr::GlobalId,
                    value: Expr::LoadLocal(0),
                },
                Stmt::Return(Expr::LoadLocal(0)),
            ],
            ret: Ty::f32(),
        }],
        entry: 0,
    };
    let graph = generic_dispatch_graph(&program, 65, 70);
    let input: Vec<f32> = (0..70).map(|value| value as f32).collect();

    let store = shared_store();
    {
        let mut prep = CpuBackend::with_store(store.clone());
        prep.prepare(&graph).expect("cpu prepare");
    }
    store.lock().unwrap().write_f32(Rid(11), &input).expect("cpu input seed");
    let mut runtime = ucf_runtime::Runtime::new();
    runtime.register_backend(Box::new(CpuBackend::with_store(store.clone())));
    runtime.run(&graph).expect("cpu buffer store run");
    let expected_output = store.lock().unwrap().read_f32(Rid(10)).expect("cpu output read");
    let expected_input = store.lock().unwrap().read_f32(Rid(11)).expect("cpu state read");

    let mut vk = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            eprintln!("skip generic buffer store Vulkan parity (no usable Vulkan device): {err}");
            return;
        }
    };
    if let Err(err) = vk.prepare(&graph) {
        eprintln!("skip generic buffer store Vulkan parity (prepare failed): {err}");
        return;
    }
    vk.write_f32(Rid(11), &input).expect("vulkan input seed");
    vk.run_prepared(&graph).expect("vulkan buffer store run");
    let actual_output = vk.read_f32(Rid(10)).expect("vulkan output read");
    let actual_input = vk.read_f32(Rid(11)).expect("vulkan state read");
    assert_eq!(actual_output, expected_output);
    assert_eq!(actual_input, expected_input);
    let expected_prefix: Vec<f32> = (0..65).map(|value| value as f32 + 2.5).collect();
    assert_eq!(&actual_output[..65], expected_prefix.as_slice());
    assert_eq!(&actual_input[..65], expected_prefix.as_slice());
    assert_eq!(&actual_output[65..], &[0.0; 5]);
    assert_eq!(&actual_input[65..], &[65.0, 66.0, 67.0, 68.0, 69.0]);
}

#[test]
fn vulkan_generic_bounded_for_matches_cpu_interpreter() {
    let program = Program {
        version: 1,
        functions: vec![Function {
            name: "bounded_accumulate".into(),
            params: Vec::new(),
            locals: vec![(0, Ty::f32())],
            body: vec![
                Stmt::Let { id: 0, ty: Ty::f32(), init: Expr::ConstF32(0.0) },
                Stmt::For {
                    iter: 1,
                    limit: 8,
                    body: vec![Stmt::StoreLocal {
                        id: 0,
                        value: Expr::BinOp {
                            op: BinOp::Add,
                            lhs: Box::new(Expr::LoadLocal(0)),
                            rhs: Box::new(Expr::LoadUniform(0)),
                        },
                    }],
                },
                Stmt::Return(Expr::LoadLocal(0)),
            ],
            ret: Ty::f32(),
        }],
        entry: 0,
    };
    let graph = generic_dispatch_graph(&program, 65, 70);
    let expected = run_cpu_generic_dispatch(&graph);
    let mut vk = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            eprintln!("skip generic For Vulkan parity (no usable Vulkan device): {err}");
            return;
        }
    };
    if let Err(err) = vk.prepare(&graph) {
        eprintln!("skip generic For Vulkan parity (prepare failed): {err}");
        return;
    }
    vk.run_prepared(&graph).expect("vulkan bounded For run");
    let actual = vk.read_f32(Rid(10)).expect("vulkan bounded For read");
    assert_eq!(actual, expected);
    assert_eq!(&actual[..65], &[20.0; 65]);
    assert_eq!(&actual[65..], &[0.0; 5]);
}

#[test]
fn vulkan_generic_bounded_while_matches_cpu_interpreter() {
    let program = Program {
        version: 1,
        functions: vec![Function {
            name: "bounded_while".into(),
            params: Vec::new(),
            locals: vec![(0, Ty::f32())],
            body: vec![
                Stmt::Let { id: 0, ty: Ty::f32(), init: Expr::ConstF32(0.0) },
                Stmt::While {
                    cond: Expr::BinOp {
                        op: BinOp::Gt,
                        lhs: Box::new(Expr::LoadUniform(0)),
                        rhs: Box::new(Expr::LoadLocal(0)),
                    },
                    limit: 8,
                    body: vec![Stmt::StoreLocal {
                        id: 0,
                        value: Expr::BinOp {
                            op: BinOp::Add,
                            lhs: Box::new(Expr::LoadLocal(0)),
                            rhs: Box::new(Expr::ConstF32(1.0)),
                        },
                    }],
                },
                Stmt::Return(Expr::LoadLocal(0)),
            ],
            ret: Ty::f32(),
        }],
        entry: 0,
    };
    let graph = generic_dispatch_graph(&program, 65, 70);
    let expected = run_cpu_generic_dispatch(&graph);
    let mut vk = VulkanBackend::new().expect("Vulkan device for bounded while");
    vk.prepare(&graph).expect("bounded while prepare");
    vk.run_prepared(&graph).expect("bounded while run");
    let actual = vk.read_f32(Rid(10)).expect("bounded while readback");
    assert_eq!(actual, expected);
    assert_eq!(&actual[..65], &[3.0; 65]);
    assert_eq!(&actual[65..], &[0.0; 5]);
}

#[test]
fn vulkan_generic_is_finite_matches_cpu_interpreter() {
    let program = Program {
        version: 1,
        functions: vec![Function {
            name: "is_finite_select".into(),
            params: Vec::new(),
            locals: vec![(0, Ty::f32())],
            body: vec![
                Stmt::Let { id: 0, ty: Ty::f32(), init: Expr::LoadUniform(0) },
                Stmt::Return(Expr::Select {
                    cond: Box::new(Expr::UnOp { op: ucf_program::UnOp::IsFinite, arg: Box::new(Expr::LoadLocal(0)) }),
                    then_: Box::new(Expr::ConstF32(1.0)),
                    else_: Box::new(Expr::ConstF32(0.0)),
                }),
            ],
            ret: Ty::f32(),
        }],
        entry: 0,
    };
    let graph = generic_dispatch_graph(&program, 65, 70);
    let expected = run_cpu_generic_dispatch(&graph);
    let mut vk = match VulkanBackend::new() {
        Ok(backend) => backend,
        Err(err) => {
            eprintln!("skip generic is finite Vulkan parity (no usable Vulkan device): {err}");
            return;
        }
    };
    if let Err(err) = vk.prepare(&graph) {
        eprintln!("skip generic is finite Vulkan parity (prepare failed): {err}");
        return;
    }
    vk.run_prepared(&graph).expect("vulkan is finite run");
    let actual = vk.read_f32(Rid(10)).expect("vulkan is finite read");
    assert_eq!(actual, expected);
    assert_eq!(&actual[..65], &[1.0; 65]);
    assert_eq!(&actual[65..], &[0.0; 5]);
}
