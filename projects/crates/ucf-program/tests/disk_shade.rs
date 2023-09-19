use ucf_program::{disk_shade_novikov, validate_program, CpuInterpreter, Value};

#[test]
fn disk_shade_novikov_out_of_band_is_black() {
    let prog = disk_shade_novikov();
    validate_program(&prog).expect("valid");

    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::F64(2.0));
    cpu.with_uniform(1, Value::F64(1.0));
    cpu.with_uniform(2, Value::F64(6.0));
    cpu.with_uniform(3, Value::F64(30.0));
    cpu.with_uniform(4, Value::F64(1.0));
    cpu.with_uniform(5, Value::F64(1.0));
    let rgb = cpu.run(&prog).expect("run").value.as_vec3().expect("vec3");
    assert!(rgb[0].abs() < 1e-9);
    assert!(rgb[1].abs() < 1e-9);
    assert!(rgb[2].abs() < 1e-9);
}

#[test]
fn disk_shade_novikov_in_band_is_warm_hdr() {
    let prog = disk_shade_novikov();
    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::F64(12.0));
    cpu.with_uniform(1, Value::F64(1.5));
    cpu.with_uniform(2, Value::F64(6.0));
    cpu.with_uniform(3, Value::F64(30.0));
    cpu.with_uniform(4, Value::F64(1.0));
    cpu.with_uniform(5, Value::F64(1.0));
    let rgb = cpu.run(&prog).expect("run").value.as_vec3().expect("vec3");
    assert!(rgb[0] > rgb[2], "warm disk should be redder than blue: {rgb:?}");
    assert!(rgb[0] > 0.05, "expected non-trivial HDR radiance: {rgb:?}");
}
