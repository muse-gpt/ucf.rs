use ucf_program::{
    orbit_loop_accumulate, orbit_step_once, program_hash, ray_batch_steps, validate_program,
    vector_math_normalize, CpuInterpreter, Value,
};

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
fn orbit_step_once_cpu() {
    let prog = orbit_step_once();
    validate_program(&prog).expect("valid");

    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::F64(10.0));
    cpu.with_uniform(1, Value::F64(-0.5));
    cpu.with_uniform(2, Value::F64(2.0));
    let out = cpu.run(&prog).expect("run");
    assert!((out.value.as_f64().expect("f64") - 9.0).abs() < 1e-9);
}

#[test]
fn orbit_loop_bounded_cpu() {
    let prog = orbit_loop_accumulate();
    validate_program(&prog).expect("valid");

    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::F64(0.0));
    let out = cpu.run(&prog).expect("run");
    // sum 0..63 = 63*64/2 = 2016
    assert!((out.value.as_f64().expect("f64") - 2016.0).abs() < 1e-9);
    assert_eq!(out.loop_steps, 64);
}

#[test]
fn ray_batch_parallel_cpu() {
    let prog = ray_batch_steps();
    validate_program(&prog).expect("valid");
    let mut cpu = CpuInterpreter::new();
    let results = cpu.run_parallel(&prog, 4).expect("parallel");
    assert_eq!(results.len(), 4);
    assert!((results[0].value.as_f64().expect("f64") - 1.0).abs() < 1e-9);
    assert!((results[3].value.as_f64().expect("f64") - 4.0).abs() < 1e-9);
}
