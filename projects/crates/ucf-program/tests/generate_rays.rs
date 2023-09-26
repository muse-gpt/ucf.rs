use ucf_program::{generate_camera_ray, validate_program, CpuInterpreter, Value};

#[test]
fn generate_camera_ray_looks_forward_at_center() {
    let prog = generate_camera_ray();
    validate_program(&prog).expect("valid");

    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::F64(0.0));
    cpu.with_uniform(1, Value::F64(0.0));
    cpu.with_uniform(2, Value::F64(0.5));
    cpu.with_uniform(3, Value::Vec3([0.0, 0.0, -1.0]));
    cpu.with_uniform(4, Value::Vec3([1.0, 0.0, 0.0]));
    cpu.with_uniform(5, Value::Vec3([0.0, 1.0, 0.0]));
    let dir = cpu.run(&prog).expect("run").value.as_vec3().expect("vec3");
    assert!((dir[0]).abs() < 1e-9);
    assert!((dir[1]).abs() < 1e-9);
    assert!((dir[2] + 1.0).abs() < 1e-9);
}

#[test]
fn generate_camera_ray_tilts_with_ndc() {
    let prog = generate_camera_ray();
    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::F64(1.0));
    cpu.with_uniform(1, Value::F64(0.0));
    cpu.with_uniform(2, Value::F64(0.5));
    cpu.with_uniform(3, Value::Vec3([0.0, 0.0, -1.0]));
    cpu.with_uniform(4, Value::Vec3([1.0, 0.0, 0.0]));
    cpu.with_uniform(5, Value::Vec3([0.0, 1.0, 0.0]));
    let dir = cpu.run(&prog).expect("run").value.as_vec3().expect("vec3");
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    assert!((len - 1.0).abs() < 1e-9);
    assert!(dir[0] > 0.0, "positive u should tilt toward +right: {dir:?}");
}
