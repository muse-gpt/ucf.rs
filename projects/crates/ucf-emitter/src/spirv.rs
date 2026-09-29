use ucf_types::{ShaderOp, ShaderProgram};

/// SPIR-V module for the Raster triangle thin gate (vertex + fragment entries).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RasterTriSpirv {
    /// Module bytes containing `vs_main` and `fs_main`.
    pub module: Vec<u8>,
}

/// Emit SPIR-V words (as little-endian bytes) for the given UCF program.
///
/// Built-in WGSL is compiled with `naga`. Authors never write shaders.
pub fn emit_spirv(program: &ShaderProgram) -> Result<Vec<u8>, String> {
    for op in &program.ops {
        match op {
            ShaderOp::MemCopy => continue,
            ShaderOp::IotaFill { value } => {
                return compile_wgsl(&wgsl_fill(&program.entry, *value), &program.entry);
            }
            ShaderOp::MatMul => {
                return compile_wgsl(&wgsl_matmul(&program.entry), &program.entry);
            }
            ShaderOp::Rgba8Denoise => {
                return Err("RGBA8 denoise is CUDA-only in this thin gate".into());
            }
        }
    }
    Err("program has no emittable SPIR-V ops".into())
}

/// Emit a hard-coded NDC yellow triangle (VS+FS) for Raster `draw=tri`.
pub fn emit_raster_tri_spirv() -> Result<RasterTriSpirv, String> {
    let wgsl = r#"
struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) col: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    var verts = array<vec2<f32>, 3>(
        vec2<f32>(-0.8, -0.8),
        vec2<f32>( 0.8, -0.8),
        vec2<f32>( 0.0,  0.8)
    );
    var o: VSOut;
    o.pos = vec4<f32>(verts[vid], 0.0, 1.0);
    o.col = vec4<f32>(1.0, 1.0, 0.0, 1.0);
    return o;
}

@fragment
fn fs_main(@location(0) col: vec4<f32>) -> @location(0) vec4<f32> {
    return col;
}
"#;
    Ok(RasterTriSpirv {
        module: compile_wgsl(wgsl, "vs_main")?,
    })
}

fn wgsl_fill(entry: &str, value: f32) -> String {
    format!(
        r#"
struct Params {{
    count: u32,
}}

@group(0) @binding(0)
var<storage, read_write> out_buf: array<f32>;

var<push_constant> params: Params;

@compute @workgroup_size(64)
fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= params.count) {{
        return;
    }}
    out_buf[gid.x] = {value:?};
}}
"#
    )
}

fn wgsl_matmul(entry: &str) -> String {
    format!(
        r#"
struct Dims {{
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}}

@group(0) @binding(0)
var<storage, read> a_buf: array<f32>;

@group(0) @binding(1)
var<storage, read> b_buf: array<f32>;

@group(0) @binding(2)
var<storage, read_write> out_buf: array<f32>;

var<push_constant> dims: Dims;

@compute @workgroup_size(64)
fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    if (idx >= dims.m * dims.n) {{
        return;
    }}
    let row = idx / dims.n;
    let col = idx % dims.n;
    var acc = 0.0;
    for (var i = 0u; i < dims.k; i = i + 1u) {{
        acc = acc + a_buf[row * dims.k + i] * b_buf[i * dims.n + col];
    }}
    out_buf[idx] = acc;
}}
"#
    )
}

fn compile_wgsl(source: &str, entry: &str) -> Result<Vec<u8>, String> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| format!("WGSL parse failed for `{entry}`: {e}"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|e| format!("WGSL validate failed for `{entry}`: {e}"))?;

    let spv_options = naga::back::spv::Options {
        lang_version: (1, 3),
        flags: naga::back::spv::WriterFlags::empty(),
        binding_map: Default::default(),
        capabilities: None,
        bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
        zero_initialize_workgroup_memory: naga::back::spv::ZeroInitializeWorkgroupMemoryMode::Native,
        debug_info: None,
    };
    let words = naga::back::spv::write_vec(&module, &info, &spv_options, None)
        .map_err(|e| format!("SPIR-V write failed for `{entry}`: {e}"))?;
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_types::ShaderProgram;

    #[test]
    fn emit_fill_and_matmul_spirv() {
        let fill = emit_spirv(&ShaderProgram::dispatch_fill("ucf_fill", 1.0)).expect("fill");
        assert!(fill.len() > 20);
        assert_eq!(&fill[0..4], &[0x03, 0x02, 0x23, 0x07]); // SPIR-V magic LE
        let matmul = emit_spirv(&ShaderProgram::matmul("ucf_matmul")).expect("matmul");
        assert!(matmul.len() > 20);
        assert_eq!(&matmul[0..4], &[0x03, 0x02, 0x23, 0x07]);
    }

    #[test]
    fn emit_raster_tri_spirv_module() {
        let tri = emit_raster_tri_spirv().expect("tri");
        assert!(tri.module.len() > 20);
        assert_eq!(&tri.module[0..4], &[0x03, 0x02, 0x23, 0x07]);
    }
}
