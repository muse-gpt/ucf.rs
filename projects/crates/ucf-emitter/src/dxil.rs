use ucf_types::{ShaderOp, ShaderProgram};

/// Emit DXBC compute bytecode (`cs_5_1`) for the given UCF program.
///
/// On Windows this compiles built-in HLSL through the system `d3dcompiler`.
/// Authors never write HLSL; the source lives only inside this emitter.
pub fn emit_dxil(program: &ShaderProgram) -> Result<Vec<u8>, String> {
    for op in &program.ops {
        match op {
            ShaderOp::MemCopy => continue,
            ShaderOp::IotaFill { value } => {
                return compile_cs(&hlsl_fill(&program.entry, *value), &program.entry);
            }
            ShaderOp::MatMul => {
                return compile_cs(&hlsl_matmul(&program.entry), &program.entry);
            }
        }
    }
    Err("program has no emittable DXIL ops".into())
}

fn hlsl_fill(entry: &str, value: f32) -> String {
    // Value is baked like CUDA PTX; `count` and the UAV are bound at dispatch.
    format!(
        r#"
cbuffer UcfFillParams : register(b0) {{
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}};
RWByteAddressBuffer OutBuf : register(u0);

[numthreads(64, 1, 1)]
void {entry}(uint3 dtid : SV_DispatchThreadID) {{
    if (dtid.x >= count) return;
    float value = {value:?};
    OutBuf.Store(dtid.x * 4, asuint(value));
}}
"#
    )
}

fn hlsl_matmul(entry: &str) -> String {
    format!(
        r#"
cbuffer UcfMatMulParams : register(b0) {{
    uint m;
    uint n;
    uint k;
    uint _pad;
}};
ByteAddressBuffer ABuf : register(t0);
ByteAddressBuffer BBuf : register(t1);
RWByteAddressBuffer OutBuf : register(u0);

[numthreads(64, 1, 1)]
void {entry}(uint3 dtid : SV_DispatchThreadID) {{
    uint idx = dtid.x;
    if (idx >= m * n) return;
    uint row = idx / n;
    uint col = idx % n;
    float acc = 0.0;
    for (uint i = 0; i < k; i++) {{
        float a = asfloat(ABuf.Load((row * k + i) * 4));
        float b = asfloat(BBuf.Load((i * n + col) * 4));
        acc += a * b;
    }}
    OutBuf.Store(idx * 4, asuint(acc));
}}
"#
    )
}

#[cfg(windows)]
fn compile_cs(hlsl: &str, entry: &str) -> Result<Vec<u8>, String> {
    use windows::core::PCSTR;
    use windows::Win32::Graphics::Direct3D::Fxc::{D3DCompile, D3DCOMPILE_OPTIMIZATION_LEVEL3};
    use windows::Win32::Graphics::Direct3D::ID3DBlob;

    let source = hlsl.as_bytes();
    let entry_c = std::ffi::CString::new(entry).map_err(|e| e.to_string())?;
    let target = windows::core::s!("cs_5_1");
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    let result = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            None,
            None,
            None,
            PCSTR(entry_c.as_ptr().cast()),
            target,
            D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(err) = result {
        let detail = errors
            .as_ref()
            .map(|blob| unsafe {
                let ptr = blob.GetBufferPointer() as *const u8;
                let len = blob.GetBufferSize();
                String::from_utf8_lossy(std::slice::from_raw_parts(ptr, len)).into_owned()
            })
            .unwrap_or_else(|| err.message());
        return Err(format!("D3DCompile failed for `{entry}`: {detail}"));
    }
    let code = code.ok_or_else(|| "D3DCompile returned null bytecode".to_string())?;
    unsafe {
        let ptr = code.GetBufferPointer() as *const u8;
        let len = code.GetBufferSize();
        Ok(std::slice::from_raw_parts(ptr, len).to_vec())
    }
}

#[cfg(not(windows))]
fn compile_cs(_hlsl: &str, _entry: &str) -> Result<Vec<u8>, String> {
    Err("DXBC emission requires Windows d3dcompiler".into())
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use ucf_types::ShaderProgram;

    #[test]
    fn emit_fill_and_matmul_dxbc() {
        let fill = emit_dxil(&ShaderProgram::dispatch_fill("ucf_fill", 1.0)).expect("fill");
        assert!(fill.len() > 32);
        assert_eq!(&fill[0..4], b"DXBC");
        let matmul = emit_dxil(&ShaderProgram::matmul("ucf_matmul")).expect("matmul");
        assert!(matmul.len() > 32);
        assert_eq!(&matmul[0..4], b"DXBC");
    }
}
