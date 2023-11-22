use ucf_program::{BinOp, FunctionBuilder, ProgramBuilder, CpuInterpreter, Value};

#[test]
fn u32_arithmetic_preserves_integer_index_semantics() {
    let mut builder = FunctionBuilder::new("u32_arithmetic");
    let gid = builder.emit_global_id().expect("global id");
    let width = builder.emit_const_u32(2).expect("width");
    let x = builder.emit_binary(BinOp::Mod, gid, width).expect("mod");
    let gid = builder.emit_global_id().expect("global id");
    let width = builder.emit_const_u32(2).expect("width");
    let y = builder.emit_binary(BinOp::Div, gid, width).expect("div");
    let stride = builder.emit_const_u32(3).expect("stride");
    let base = builder.emit_binary(BinOp::Mul, y, stride).expect("mul");
    let index = builder.emit_binary(BinOp::Add, base, x).expect("add");
    builder.return_value(index).expect("return");
    let program = ProgramBuilder::new().entry(builder).expect("entry").finish().expect("program");

    let mut interpreter = CpuInterpreter::new();
    let results = interpreter.run_parallel(&program, 4).expect("execute");
    assert_eq!(results.iter().map(|result| &result.value).collect::<Vec<_>>(), vec![
        &Value::U32(0), &Value::U32(1), &Value::U32(3), &Value::U32(4),
    ]);
}
