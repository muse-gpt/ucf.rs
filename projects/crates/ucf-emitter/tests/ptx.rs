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

#[test]
fn denoise_ptx_has_entry_and_u8_ops() {
    let program = ShaderProgram::rgba8_denoise("ucf_denoise");
    let ptx = emit_ptx(&program).expect("ptx");
    let text = String::from_utf8(ptx).expect("utf8");
    assert!(text.contains(".entry ucf_denoise"));
    assert!(text.contains("ld.global.u8"));
    assert!(text.contains("st.global.u8"));
}

#[test]
fn attention_ptx_has_entry_and_fma() {
    let program = ShaderProgram::attention("ucf_attention");
    let ptx = emit_ptx(&program).expect("ptx");
    let text = String::from_utf8(ptx).expect("utf8");
    assert!(text.contains(".entry ucf_attention"));
    assert!(text.contains("fma.rn.f32"));
}
