use ucf_program::{post_bloom_extract, tone_map_aces, validate_program, CpuInterpreter, Value};

#[test]
fn tone_map_aces_clamps_hot_channel() {
    let prog = tone_map_aces();
    validate_program(&prog).expect("valid");

    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::Vec3([4.0, 2.0, 0.5]));
    let rgb = cpu.run(&prog).expect("run").value.as_vec3().expect("vec3");
    assert!(rgb[0] <= 1.0 && rgb[1] <= 1.0 && rgb[2] <= 1.0);
    assert!(rgb[0] > 0.9, "hot red channel should stay near white: {rgb:?}");
}

#[test]
fn tone_map_aces_black_stays_black() {
    let prog = tone_map_aces();
    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::Vec3([0.0, 0.0, 0.0]));
    let rgb = cpu.run(&prog).expect("run").value.as_vec3().expect("vec3");
    assert!(rgb[0].abs() < 1e-9);
    assert!(rgb[1].abs() < 1e-9);
    assert!(rgb[2].abs() < 1e-9);
}

#[test]
fn post_bloom_extract_below_threshold_is_black() {
    let prog = post_bloom_extract();
    validate_program(&prog).expect("valid");

    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::Vec3([0.2, 0.1, 0.05]));
    cpu.with_uniform(1, Value::F64(0.85));
    let rgb = cpu.run(&prog).expect("run").value.as_vec3().expect("vec3");
    assert!(rgb[0].abs() < 1e-9);
    assert!(rgb[1].abs() < 1e-9);
    assert!(rgb[2].abs() < 1e-9);
}

#[test]
fn post_bloom_extract_above_threshold_scales_color() {
    let prog = post_bloom_extract();
    let mut cpu = CpuInterpreter::new();
    cpu.with_uniform(0, Value::Vec3([4.0, 2.0, 0.5]));
    cpu.with_uniform(1, Value::F64(0.85));
    let rgb = cpu.run(&prog).expect("run").value.as_vec3().expect("vec3");
    assert!(rgb[0] > 0.0);
    assert!(rgb[0] < 4.0, "scaled bright pass should be below source HDR: {rgb:?}");
}
