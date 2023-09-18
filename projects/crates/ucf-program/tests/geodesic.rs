use ucf_program::{
    geodesic_capture_classify, geodesic_radial_infall_steps, validate_program, CpuInterpreter,
    Value,
};

#[test]
fn geodesic_capture_classify_cpu() {
    let prog = geodesic_capture_classify();
    validate_program(&prog).expect("valid");

    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::F64(2.0));
    cpu.with_uniform(1, Value::F64(5.196152));
    let captured = cpu.run(&prog).expect("run").value.as_f64().expect("f64");
    assert!((captured - 1.0).abs() < 1e-9);

    cpu.with_uniform(0, Value::F64(8.0));
    let escaped = cpu.run(&prog).expect("run").value.as_f64().expect("f64");
    assert!(escaped.abs() < 1e-9);
}

#[test]
fn geodesic_radial_infall_steps_cpu() {
    let prog = geodesic_radial_infall_steps();
    validate_program(&prog).expect("valid");

    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::F64(10.0));
    cpu.with_uniform(1, Value::F64(2.0));
    cpu.with_uniform(2, Value::F64(0.08));
    let steps = cpu.run(&prog).expect("run").value.as_f64().expect("f64");
    assert!((steps - 100.0).abs() < 1e-9);

    cpu.with_uniform(0, Value::F64(2.5));
    cpu.with_uniform(1, Value::F64(2.0));
    cpu.with_uniform(2, Value::F64(0.08));
    let at_horizon = cpu.run(&prog).expect("run").value.as_f64().expect("f64");
    assert!((at_horizon - 7.0).abs() < 1e-9);
}
