use hardware_program::{
    bounded_accumulate, parallel_path_steps, program_hash, scalar_step_once, validate_program,
    vector_math_normalize, CpuInterpreter, Value,
};

#[test]
fn structured_loop_control_flow_is_preserved() {
    use hardware_program::{BinOp, Expr, Function, Program, ScalarTy, Stmt, Ty};

    let program = Program {
        version: 1,
        functions: vec![Function {
            name: "entry".into(),
            params: Vec::new(),
            locals: vec![(0, Ty::f64()), (1, Ty::f64())],
            body: vec![
                Stmt::Let { id: 0, ty: Ty::f64(), init: Expr::ConstF64(0.0) },
                Stmt::For {
                    iter: 1,
                    limit: 8,
                    body: vec![
                        Stmt::If {
                            cond: Expr::BinOp {
                                op: BinOp::Eq,
                                lhs: Box::new(Expr::LoadLocal(1)),
                                rhs: Box::new(Expr::ConstU32(2)),
                            },
                            then_body: vec![Stmt::Continue],
                            else_body: Vec::new(),
                        },
                        Stmt::If {
                            cond: Expr::BinOp {
                                op: BinOp::Eq,
                                lhs: Box::new(Expr::LoadLocal(1)),
                                rhs: Box::new(Expr::ConstU32(5)),
                            },
                            then_body: vec![Stmt::Break],
                            else_body: Vec::new(),
                        },
                        Stmt::StoreLocal {
                            id: 0,
                            value: Expr::BinOp {
                                op: BinOp::Add,
                                lhs: Box::new(Expr::LoadLocal(0)),
                                rhs: Box::new(Expr::ConstF64(1.0)),
                            },
                        },
                    ],
                },
                Stmt::Return(Expr::LoadLocal(0)),
            ],
            ret: Ty::Scalar(ScalarTy::F64),
        }],
        entry: 0,
    };
    let mut cpu = CpuInterpreter::new();
    let result = cpu.run(&program).expect("control flow execution");
    assert_eq!(result.value, Value::F64(4.0));
    assert_eq!(result.loop_steps, 6);
}

#[test]
fn resource_load_store_is_bounds_checked() {
    use hardware_program::{Expr, Function, Program, ScalarTy, Stmt, Ty};

    let program = Program {
        version: 1,
        functions: vec![Function {
            name: "copy_instance".into(),
            params: Vec::new(),
            locals: vec![(0, Ty::Scalar(ScalarTy::U32)), (1, Ty::f64())],
            body: vec![
                Stmt::Let { id: 0, ty: Ty::Scalar(ScalarTy::U32), init: Expr::GlobalId },
                Stmt::Let {
                    id: 1,
                    ty: Ty::f64(),
                    init: Expr::LoadBuffer { resource: 0, index: Box::new(Expr::LoadLocal(0)) },
                },
                Stmt::StoreBuffer {
                    resource: 1,
                    index: Expr::LoadLocal(0),
                    value: Expr::LoadLocal(1),
                },
                Stmt::Return(Expr::LoadLocal(1)),
            ],
            ret: Ty::f64(),
        }],
        entry: 0,
    };
    let mut cpu = CpuInterpreter::new();
    cpu.with_buffer(0, vec![Value::F64(2.0), Value::F64(4.0), Value::F64(8.0)])
        .with_buffer(1, vec![Value::F64(0.0), Value::F64(0.0), Value::F64(0.0)]);
    let results = cpu.run_parallel(&program, 3).expect("resource execution");
    assert_eq!(results[2].value, Value::F64(8.0));
    assert_eq!(cpu.buffer(1).unwrap()[1], Value::F64(4.0));

    let error = cpu
        .with_buffer(0, vec![Value::F64(1.0)])
        .run_parallel(&program, 2)
        .expect_err("out of bounds must fail");
    assert!(error.to_string().contains("out of bounds"));
}

#[test]
fn vector_math_normalize_cpu() {
    let prog = vector_math_normalize();
    validate_program(&prog).expect("valid");
    let hash = program_hash(&prog);
    assert_ne!(hash, 0);

    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::Vec3([3.0, 4.0, 0.0]));
    let out = cpu.run(&prog).expect("run");
    let v = out.value.as_vec3().expect("vec3");
    assert!((v[0] - 0.6).abs() < 1e-6);
    assert!((v[1] - 0.8).abs() < 1e-6);
    assert!(v[2].abs() < 1e-6);
}

#[test]
fn scalar_step_once_cpu() {
    let prog = scalar_step_once();
    validate_program(&prog).expect("valid");

    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::F64(10.0));
    cpu.with_uniform(1, Value::F64(-0.5));
    cpu.with_uniform(2, Value::F64(2.0));
    let out = cpu.run(&prog).expect("run");
    assert!((out.value.as_f64().expect("f64") - 9.0).abs() < 1e-9);
}

#[test]
fn bounded_accumulate_cpu() {
    let prog = bounded_accumulate();
    validate_program(&prog).expect("valid");

    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::F64(1.0));
    let out = cpu.run(&prog).expect("run");
    assert!((out.value.as_f64().expect("f64") - 64.0).abs() < 1e-9);
    assert_eq!(out.loop_steps, 64);
}

#[test]
fn parallel_path_cpu() {
    let prog = parallel_path_steps();
    validate_program(&prog).expect("valid");
    let mut cpu = CpuInterpreter::new();
    let results = cpu.run_parallel(&prog, 4).expect("parallel");
    assert_eq!(results.len(), 4);
    assert!((results[0].value.as_f64().expect("f64") - 1.0).abs() < 1e-9);
    assert!((results[3].value.as_f64().expect("f64") - 4.0).abs() < 1e-9);
}
