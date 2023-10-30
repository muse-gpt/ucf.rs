use ucf_program::{BinOp, CpuInterpreter, Expr, FunctionBuilder, Program, ProgramBuilder, StructuredFunctionBuilder, Ty, UnOp, Value};

fn binary(op: BinOp, left: Expr, right: Expr) -> Expr {
    Expr::BinOp { op, lhs: Box::new(left), rhs: Box::new(right) }
}

fn execute(expression: Expr, ty: Ty) -> Value {
    let builder = StructuredFunctionBuilder::new("scalar_precision");
    let program = Program { version: 1, entry: 0, functions: vec![builder.finish(ty, expression)] };
    CpuInterpreter::new().run(&program).unwrap().value
}

#[test]
fn structured_f32_rounds_each_operation_while_f64_preserves_precision() {
    let f32_expression = binary(BinOp::Sub,
        binary(BinOp::Add, Expr::ConstF32(16_777_216.0), Expr::ConstF32(1.0)), Expr::ConstF32(16_777_216.0));
    let f64_expression = binary(BinOp::Sub,
        binary(BinOp::Add, Expr::ConstF64(16_777_216.0), Expr::ConstF64(1.0)), Expr::ConstF64(16_777_216.0));
    assert_eq!(execute(f32_expression, Ty::f32()), Value::F32(0.0));
    assert_eq!(execute(f64_expression, Ty::f64()), Value::F64(1.0));
}

#[test]
fn cfg_f32_rounding_matches_structured_execution() {
    let mut builder = FunctionBuilder::new("cfg_precision");
    let large = builder.emit_const_f32(16_777_216.0).unwrap();
    let one = builder.emit_const_f32(1.0).unwrap();
    let sum = builder.emit_binary(BinOp::Add, large, one).unwrap();
    let difference = builder.emit_binary(BinOp::Sub, sum, large).unwrap();
    builder.return_value(difference).unwrap();
    let program = ProgramBuilder::new().entry(builder).unwrap().finish().unwrap();
    assert_eq!(CpuInterpreter::new().run(&program).unwrap().value, Value::F32(0.0));
}

#[test]
fn cfg_uniform_binding_obeys_declared_float_precision() {
    for (ty, expected) in [(Ty::f32(), Value::F32(16_777_216.0)), (Ty::f64(), Value::F64(16_777_217.0))] {
        let mut builder = FunctionBuilder::new("typed_uniform");
        let uniform = builder.emit_load_uniform(0, ty).unwrap();
        builder.return_value(uniform).unwrap();
        let program = ProgramBuilder::new().entry(builder).unwrap().finish().unwrap();
        let mut cpu = CpuInterpreter::new();
        cpu.with_uniform(0, Value::F64(16_777_217.0));
        assert_eq!(cpu.run(&program).unwrap().value, expected);
    }
}

#[test]
fn scalar_f32_operations_preserve_runtime_type_and_rounding() {
    for (op, expected) in [
        (BinOp::Add, 2.3f32 + 0.7), (BinOp::Sub, 2.3f32 - 0.7),
        (BinOp::Mul, 2.3f32 * 0.7), (BinOp::Div, 2.3f32 / 0.7),
        (BinOp::Min, 2.3f32.min(0.7)), (BinOp::Max, 2.3f32.max(0.7)),
        (BinOp::Mod, 2.3f32 % 0.7), (BinOp::Pow, 2.3f32.powf(0.7)),
    ] {
        assert_eq!(execute(binary(op, Expr::ConstF32(2.3), Expr::ConstF32(0.7)), Ty::f32()), Value::F32(expected), "{op:?}");
    }
    for (op, expected) in [
        (UnOp::Neg, -2.3f32), (UnOp::Sqrt, 2.3f32.sqrt()), (UnOp::Sin, 2.3f32.sin()),
        (UnOp::Cos, 2.3f32.cos()), (UnOp::Atan, 2.3f32.atan()), (UnOp::Abs, 2.3f32.abs()), (UnOp::Ln, 2.3f32.ln()),
    ] {
        assert_eq!(execute(Expr::UnOp { op, arg: Box::new(Expr::ConstF32(2.3)) }, Ty::f32()), Value::F32(expected), "{op:?}");
    }
}

#[test]
fn scalar_float_domain_errors_are_observable_non_finite_values() {
    let Value::F32(infinity) = execute(binary(BinOp::Div, Expr::ConstF32(1.0), Expr::ConstF32(0.0)), Ty::f32()) else { panic!("lost f32 type"); };
    assert!(infinity.is_infinite());
    let Value::F32(nan) = execute(Expr::UnOp { op: UnOp::Sqrt, arg: Box::new(Expr::ConstF32(-1.0)) }, Ty::f32()) else { panic!("lost f32 type"); };
    assert!(nan.is_nan());
    let Value::F64(nan) = execute(binary(BinOp::Div, Expr::ConstF64(0.0), Expr::ConstF64(0.0)), Ty::f64()) else { panic!("lost f64 type"); };
    assert!(nan.is_nan());
    let Value::F32(nan) = execute(binary(BinOp::Mod, Expr::ConstF32(1.0), Expr::ConstF32(0.0)), Ty::f32()) else { panic!("lost f32 type"); };
    assert!(nan.is_nan());
    let finite = Expr::UnOp { op: UnOp::IsFinite, arg: Box::new(binary(BinOp::Div, Expr::ConstF32(1.0), Expr::ConstF32(0.0))) };
    assert_eq!(execute(finite, Ty::Scalar(ucf_program::ScalarTy::Bool)), Value::Bool(false));
}
