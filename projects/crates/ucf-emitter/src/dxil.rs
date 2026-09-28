use ucf_types::{ShaderOp, ShaderProgram};

/// VS + PS DXBC for the Raster triangle thin gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RasterTriDxbc {
    /// Vertex shader bytecode (`vs_5_0`).
    pub vs: Vec<u8>,
    /// Pixel shader bytecode (`ps_5_0`).
    pub ps: Vec<u8>,
}

/// Emit DXBC compute bytecode (`cs_5_1`) for the given UCF program.
///
/// On Windows this compiles built-in HLSL through the system `d3dcompiler`.
/// Authors never write HLSL; the source lives only inside this emitter.
pub fn emit_dxil(program: &ShaderProgram) -> Result<Vec<u8>, String> {
    for op in &program.ops {
        match op {
            ShaderOp::MemCopy => continue,
            ShaderOp::IotaFill { value } => {
                return compile_shader(&hlsl_fill(&program.entry, *value), &program.entry, "cs_5_1");
            }
            ShaderOp::MatMul => {
                return compile_shader(&hlsl_matmul(&program.entry), &program.entry, "cs_5_1");
            }
        }
    }
    Err("program has no emittable DXIL ops".into())
}

/// Emit a hard-coded NDC triangle VS/PS pair (yellow solid) for Raster `draw=tri`.
pub fn emit_raster_tri_dxbc() -> Result<RasterTriDxbc, String> {
    let hlsl = r#"
struct VSOut {
    float4 pos : SV_Position;
    float4 col : COLOR0;
};

VSOut vs_main(uint vid : SV_VertexID) {
    float2 verts[3] = {
        float2(-0.8, -0.8),
        float2( 0.8, -0.8),
        float2( 0.0,  0.8)
    };
    VSOut o;
    o.pos = float4(verts[vid], 0.0, 1.0);
    o.col = float4(1.0, 1.0, 0.0, 1.0);
    return o;
}

float4 ps_main(VSOut i) : SV_Target0 {
    return i.col;
}
"#;
    Ok(RasterTriDxbc {
        vs: compile_shader(hlsl, "vs_main", "vs_5_0")?,
        ps: compile_shader(hlsl, "ps_main", "ps_5_0")?,
    })
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
fn compile_shader(hlsl: &str, entry: &str, target: &str) -> Result<Vec<u8>, String> {
    use windows::core::PCSTR;
    use windows::Win32::Graphics::Direct3D::Fxc::{D3DCompile, D3DCOMPILE_OPTIMIZATION_LEVEL3};
    use windows::Win32::Graphics::Direct3D::ID3DBlob;

    let source = hlsl.as_bytes();
    let entry_c = std::ffi::CString::new(entry).map_err(|e| e.to_string())?;
    let target_c = std::ffi::CString::new(target).map_err(|e| e.to_string())?;
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
            PCSTR(target_c.as_ptr().cast()),
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
        return Err(format!("D3DCompile failed for `{entry}` ({target}): {detail}"));
    }
    let code = code.ok_or_else(|| "D3DCompile returned null bytecode".to_string())?;
    unsafe {
        let ptr = code.GetBufferPointer() as *const u8;
        let len = code.GetBufferSize();
        Ok(std::slice::from_raw_parts(ptr, len).to_vec())
    }
}

#[cfg(not(windows))]
fn compile_shader(_hlsl: &str, _entry: &str, _target: &str) -> Result<Vec<u8>, String> {
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

    #[test]
    fn emit_raster_tri_vs_ps_dxbc() {
        let tri = emit_raster_tri_dxbc().expect("tri");
        assert!(tri.vs.len() > 32);
        assert!(tri.ps.len() > 32);
        assert_eq!(&tri.vs[0..4], b"DXBC");
        assert_eq!(&tri.ps[0..4], b"DXBC");
    }
}
