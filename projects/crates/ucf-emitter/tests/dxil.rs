use ucf_emitter::emit_dxil;
use ucf_types::ShaderProgram;

#[test]
fn dxil_container_starts_with_dxbc() {
    let program = ShaderProgram::dispatch_fill("ucf_dispatch", 1.0);
    let blob = emit_dxil(&program).expect("dxil");
    assert_eq!(&blob[0..4], b"DXBC");
    assert!(blob.windows(4).any(|w| w == b"DXIL"));
}
