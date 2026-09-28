use ucf_emitter::emit_ptx;
use ucf_types::ShaderProgram;

#[test]
fn ptx_contains_entry_and_no_cuda_c() {
    let program = ShaderProgram::dispatch_fill("ucf_dispatch", 1.0);
    let ptx = emit_ptx(&program).expect("ptx");
    let text = String::from_utf8(ptx).expect("utf8");
    assert!(text.contains(".entry ucf_dispatch"));
    assert!(!text.contains("#include"));
    assert!(!text.contains("__global__"));
}

#[test]
fn matmul_ptx_has_entry_and_fma() {
    let program = ShaderProgram::matmul("ucf_matmul");
    let ptx = emit_ptx(&program).expect("ptx");
    let text = String::from_utf8(ptx).expect("utf8");
    assert!(text.contains(".entry ucf_matmul"));
    assert!(text.contains("fma.rn.f32"));
}
