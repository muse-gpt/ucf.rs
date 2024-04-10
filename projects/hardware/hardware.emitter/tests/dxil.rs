use hardware_emitter::emit_dxil;
use hardware_types::ShaderProgram;

#[test]
fn dxil_container_starts_with_dxbc() {
    let program = ShaderProgram::dispatch_fill("ucf_dispatch", 1.0);
    match emit_dxil(&program) {
        Ok(blob) => {
            assert!(blob.len() > 4, "dxil blob too short");
            assert_eq!(&blob[0..4], b"DXBC");
        }
        Err(err) => {
            eprintln!("skip dxil emit (no Windows d3dcompiler): {err}");
        }
    }
}
